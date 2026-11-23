//! Expert reads from disk.
//!
//! A load is split into pieces (one per matrix) served by a small set of I/O threads, so many reads are
//! queued at the drive at once. Each piece is read into the thread's own aligned scratch buffer
//! and repacked straight into the expert's destination buffer: no intermediate copies.
//!
//! Demand loads go before prefetches, and those before cache warm-up, which only starts reads
//! while no demand is being read. Background reads get at most a quarter of the threads. A demand for an expert that is already being
//! loaded in the background joins that load and moves its remaining pieces to the demand queue.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

use parking_lot::{Condvar, Mutex};

use crate::experts::{Key, MatLoc, key};

/// MXFP4 block: [scale: 1][values: 16]
const BLK: usize = 17;
pub const THREADS: usize = 16;
/// Blocks per piece. Large enough that an expert matrix (6.3 MB) is one read: on the drive this
/// was tuned on (DRAM-less NVMe) 1 MB pieces ran at 0.6 GB/s and whole matrices at 2 GB/s.
const PIECE_BLOCKS: usize = (8 << 20) / BLK;

#[repr(C, align(4096))]
#[derive(Clone, Copy)]
struct Page([u8; 4096]);

/// Reads `len` bytes at `off` using only page-aligned I/O (required by unbuffered handles).
fn read_at<'a>(f: &File, off: u64, len: usize, scratch: &'a mut Vec<Page>) -> std::io::Result<&'a [u8]> {
    let a0 = off & !4095;
    let a1 = (off + len as u64 + 4095) & !4095;
    let n = (a1 - a0) as usize;
    if scratch.len() < n / 4096 {
        scratch.resize(n / 4096, Page([0; 4096]));
    }
    let buf: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut u8, n) };
    let mut done = 0usize;
    while done < n {
        #[cfg(windows)]
        let r = std::os::windows::fs::FileExt::seek_read(f, &mut buf[done..], a0 + done as u64)?;
        #[cfg(unix)]
        let r = std::os::unix::fs::FileExt::read_at(f, &mut buf[done..], a0 + done as u64)?;
        done += r;
        // unbuffered reads only stop short at end of file; a retry would start unaligned
        if r == 0 || done % 4096 != 0 {
            break;
        }
    }
    let s = (off - a0) as usize;
    if done < s + len {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short expert read"));
    }
    Ok(&buf[s..s + len])
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Demand,
    Prefetch,
    /// cache warm-up: only while nothing else is queued
    Warm,
}

/// One expert being loaded.
pub struct Job {
    key: Key,
    t0: Instant,
    dst: *mut u8,
    /// owns the allocation behind `dst`; taken when the last piece finishes
    buf: Mutex<Option<Vec<u8>>>,
    remaining: AtomicUsize,
    /// some piece has been handed to a thread (guarded by the queue lock)
    started: AtomicBool,
    kind: Kind,
    /// a demand has joined this background load
    claimed: AtomicBool,
    error: Mutex<Option<String>>,
    result: Mutex<Option<Result<Arc<Vec<u8>>, String>>>,
    cv: Condvar,
}

// `dst` points into `buf`, and pieces write disjoint ranges of it.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    /// True once, for the first demand that joins a prefetch.
    pub fn claim_prefetch(&self) -> bool {
        self.kind == Kind::Prefetch && !self.claimed.swap(true, Ordering::Relaxed)
    }
    pub fn wait(&self) -> anyhow::Result<Arc<Vec<u8>>> {
        let mut r = self.result.lock();
        while r.is_none() {
            self.cv.wait(&mut r);
        }
        r.as_ref().unwrap().clone().map_err(|e| anyhow::anyhow!("expert read: {e}"))
    }
}

struct Piece {
    job: Arc<Job>,
    mat: usize,
    first: usize,
    n: usize,
}

#[derive(Default)]
struct Queues {
    demand: VecDeque<Piece>,
    prefetch: VecDeque<Piece>,
    warm: VecDeque<Piece>,
    background_active: usize,
    demand_active: usize,
    inflight: HashMap<Key, Arc<Job>>,
    stop: bool,
}

/// What finished loads report to the owner, before the waiters are released.
pub struct Finished<'a> {
    pub key: Key,
    pub data: &'a Arc<Vec<u8>>,
    pub prefetch: bool,
    pub ns: u64,
}
pub type OnDone = Box<dyn Fn(Finished) + Send + Sync>;

struct Shared {
    locs: Arc<Vec<[MatLoc; 3]>>,
    mat_bytes: usize,
    q: Mutex<Queues>,
    cv: Condvar,
    on_done: OnDone,
    max_background: usize,
}

pub struct Loader {
    sh: Arc<Shared>,
    threads: Vec<std::thread::JoinHandle<()>>,
}

impl Loader {
    /// `open` opens the shards; it is called once per thread, because Windows runs the reads on
    /// one synchronous file handle strictly one after another.
    pub fn new(open: impl Fn() -> std::io::Result<Vec<File>>, locs: Arc<Vec<[MatLoc; 3]>>, mat_bytes: usize, threads: usize, on_done: OnDone) -> std::io::Result<Self> {
        let sh = Arc::new(Shared { locs, mat_bytes, q: Default::default(), cv: Condvar::new(), on_done, max_background: (threads / 4).max(1) });
        let threads = (0..threads)
            .map(|i| {
                let (s, files) = (sh.clone(), open()?);
                std::thread::Builder::new().name(format!("expert-io-{i}")).spawn(move || worker(&s, &files))
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        Ok(Self { sh, threads })
    }

    /// Start loading an expert (or join a load in progress). `cached` is consulted under the
    /// queue lock, so a load that finished just before this call is not repeated.
    pub fn demand(&self, layer: usize, e: usize, cached: impl FnOnce() -> Option<Arc<Vec<u8>>>, alloc: impl FnOnce() -> anyhow::Result<Vec<u8>>) -> anyhow::Result<Load> {
        let k = key(layer, e);
        let mut q = self.sh.q.lock();
        if let Some(j) = q.inflight.get(&k).cloned() {
            if j.kind != Kind::Demand {
                let is_mine = |p: &Piece| p.job.key == k;
                let (a, rest): (VecDeque<Piece>, VecDeque<Piece>) = std::mem::take(&mut q.prefetch).into_iter().partition(is_mine);
                q.prefetch = rest;
                let (b, rest): (VecDeque<Piece>, VecDeque<Piece>) = std::mem::take(&mut q.warm).into_iter().partition(is_mine);
                q.warm = rest;
                q.demand.extend(a);
                q.demand.extend(b);
                drop(q);
                self.sh.cv.notify_all();
            }
            return Ok(Load::Pending(j));
        }
        if let Some(v) = cached() {
            return Ok(Load::Ready(v));
        }
        Ok(Load::Pending(self.start(&mut q, k, Kind::Demand, alloc()?)))
    }

    /// Queue a background load unless it is already running. Returns false if it was.
    pub fn prefetch(&self, layer: usize, e: usize, alloc: impl FnOnce() -> anyhow::Result<Vec<u8>>) -> bool {
        let k = key(layer, e);
        let mut q = self.sh.q.lock();
        if q.inflight.contains_key(&k) {
            return false;
        }
        match alloc() {
            Ok(b) => {
                self.start(&mut q, k, Kind::Prefetch, b);
                true
            }
            Err(_) => false,
        }
    }

    /// Queue a cache warm-up load, unless the expert is already being loaded.
    pub fn warm(&self, layer: usize, e: usize, alloc: impl FnOnce() -> anyhow::Result<Vec<u8>>) -> Option<Arc<Job>> {
        let k = key(layer, e);
        let mut q = self.sh.q.lock();
        if q.inflight.contains_key(&k) {
            return None;
        }
        let b = alloc().ok()?;
        Some(self.start(&mut q, k, Kind::Warm, b))
    }

    fn start(&self, q: &mut Queues, k: Key, kind: Kind, mut buf: Vec<u8>) -> Arc<Job> {
        assert_eq!(buf.len(), self.sh.mat_bytes * 3);
        let nblk = self.sh.mat_bytes / BLK;
        let per_mat = nblk.div_ceil(PIECE_BLOCKS);
        let job = Arc::new(Job {
            key: k,
            t0: Instant::now(),
            dst: buf.as_mut_ptr(),
            buf: Mutex::new(Some(buf)),
            remaining: AtomicUsize::new(3 * per_mat),
            started: AtomicBool::new(false),
            kind,
            claimed: AtomicBool::new(false),
            error: Mutex::new(None),
            result: Mutex::new(None),
            cv: Condvar::new(),
        });
        let queue = match kind {
            Kind::Demand => &mut q.demand,
            Kind::Prefetch => &mut q.prefetch,
            Kind::Warm => &mut q.warm,
        };
        for mat in 0..3 {
            for p in 0..per_mat {
                let first = p * PIECE_BLOCKS;
                queue.push_back(Piece { job: job.clone(), mat, first, n: (nblk - first).min(PIECE_BLOCKS) });
            }
        }
        q.inflight.insert(k, job.clone());
        self.sh.cv.notify_all();
        job
    }

    /// Drop queued prefetches of `layer` that no thread has started on: the router has spoken.
    /// Returns their unused buffers.
    pub fn cancel_prefetches(&self, layer: usize) -> Vec<Vec<u8>> {
        let mut q = self.sh.q.lock();
        let stale = |j: &Job| (j.key >> 16) as usize == layer && !j.started.load(Ordering::Relaxed);
        let (gone, keep): (VecDeque<Piece>, VecDeque<Piece>) = std::mem::take(&mut q.prefetch).into_iter().partition(|p| stale(&p.job));
        q.prefetch = keep;
        let mut bufs = Vec::new();
        for p in gone {
            q.inflight.remove(&p.job.key);
            if let Some(b) = p.job.buf.lock().take() {
                bufs.push(b);
            }
        }
        bufs
    }
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.sh.q.lock().stop = true;
        self.sh.cv.notify_all();
        for t in self.threads.drain(..) {
            // the last owner can be one of the I/O threads' callbacks: never join ourselves
            if t.thread().id() != std::thread::current().id() {
                let _ = t.join();
            }
        }
    }
}

pub enum Load {
    Ready(Arc<Vec<u8>>),
    Pending(Arc<Job>),
}

impl Load {
    pub fn wait(self) -> anyhow::Result<Arc<Vec<u8>>> {
        match self {
            Load::Ready(v) => Ok(v),
            Load::Pending(j) => j.wait(),
        }
    }
}

fn worker(sh: &Shared, files: &[File]) {
    let mut scratch: Vec<Page> = Vec::new();
    loop {
        let (piece, background) = {
            let mut q = sh.q.lock();
            loop {
                if q.stop {
                    return;
                }
                if let Some(p) = q.demand.pop_front() {
                    q.demand_active += 1;
                    break (p, false);
                }
                if q.background_active < sh.max_background {
                    let next = match q.prefetch.pop_front() {
                        Some(p) => Some(p),
                        None if q.demand_active == 0 => q.warm.pop_front(),
                        None => None,
                    };
                    if let Some(p) = next {
                        q.background_active += 1;
                        p.job.started.store(true, Ordering::Relaxed);
                        break (p, true);
                    }
                }
                sh.cv.wait(&mut q);
            }
        };
        if let Err(e) = run_piece(sh, files, &piece, &mut scratch) {
            *piece.job.error.lock() = Some(e.to_string());
        }
        {
            let mut q = sh.q.lock();
            if background {
                q.background_active -= 1;
            } else {
                q.demand_active -= 1;
            }
            if q.background_active < sh.max_background && !(q.prefetch.is_empty() && q.warm.is_empty()) {
                sh.cv.notify_one();
            }
        }
        let job = piece.job;
        if job.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
            let buf = job.buf.lock().take().expect("expert buffer");
            let err = job.error.lock().take();
            let res = match err {
                Some(e) => Err(e),
                None => {
                    let data = Arc::new(buf);
                    (sh.on_done)(Finished { key: job.key, data: &data, prefetch: job.kind == Kind::Prefetch && !job.claimed.load(Ordering::Relaxed), ns: job.t0.elapsed().as_nanos() as u64 });
                    Ok(data)
                }
            };
            sh.q.lock().inflight.remove(&job.key);
            *job.result.lock() = Some(res);
            job.cv.notify_all();
        }
    }
}

fn run_piece(sh: &Shared, files: &[File], p: &Piece, scratch: &mut Vec<Page>) -> std::io::Result<()> {
    let m = &sh.locs[(p.job.key >> 16) as usize][p.mat];
    let nblk = sh.mat_bytes / BLK;
    let base = p.mat * sh.mat_bytes;
    // SAFETY: every piece owns a distinct block range of the job's buffer, which stays alive
    // until `remaining` reaches zero.
    let (qs, sc) = unsafe {
        (
            std::slice::from_raw_parts_mut(p.job.dst.add(base + p.first * 16), p.n * 16),
            std::slice::from_raw_parts_mut(p.job.dst.add(base + nblk * 16 + p.first), p.n),
        )
    };
    if m.shard == usize::MAX {
        // dev-partial stand-in: all-zero blocks with scale exponent 127
        qs.fill(0);
        sc.fill(127);
        return Ok(());
    }
    let e = (p.job.key & 0xffff) as u64;
    let raw = read_at(&files[m.shard], m.offset + e * m.stride + (p.first * BLK) as u64, p.n * BLK, scratch)?;
    repack_mxfp4(raw, qs, sc);
    Ok(())
}

/// 17-byte MXFP4 blocks -> values and scales stored separately
fn repack_mxfp4(raw: &[u8], qs: &mut [u8], sc: &mut [u8]) {
    for ((b, q), s) in raw.chunks_exact(BLK).zip(qs.chunks_exact_mut(16)).zip(sc) {
        *s = b[0];
        q.copy_from_slice(&b[1..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(blocks: usize, experts: usize) -> (std::path::PathBuf, Vec<u8>, Arc<Vec<[MatLoc; 3]>>) {
        static N: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!("dsv4-io-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
        let stride = blocks * BLK;
        let pad = 123; // unaligned tensor start
        let data: Vec<u8> = (0..pad + 3 * experts * stride).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let loc = |m: usize| MatLoc { shard: 0, offset: (pad + m * experts * stride) as u64, stride: stride as u64 };
        (path, data, Arc::new(vec![[loc(0), loc(1), loc(2)]]))
    }

    fn expect(data: &[u8], locs: &[[MatLoc; 3]], e: usize, blocks: usize) -> Vec<u8> {
        let mut out = Vec::new();
        for m in &locs[0] {
            let raw = &data[m.offset as usize + e * m.stride as usize..][..m.stride as usize];
            raw.chunks(BLK).for_each(|b| out.extend(&b[1..]));
            raw.chunks(BLK).for_each(|b| out.push(b[0]));
            assert_eq!(raw.len(), blocks * BLK);
        }
        out
    }

    #[test]
    fn loads_match_file_and_join() {
        let blocks = PIECE_BLOCKS * 2 + 77;
        let (path, data, locs) = fixture(blocks, 4);
        let mb = blocks * BLK;
        let done = Arc::new(AtomicUsize::new(0));
        let d2 = done.clone();
        let l = Loader::new(|| Ok(vec![File::open(&path)?]), locs.clone(), mb, 4, Box::new(move |_| { d2.fetch_add(1, Ordering::Relaxed); })).unwrap();
        for e in 0..3 {
            assert!(l.prefetch(0, e, || Ok(vec![0; 3 * mb])));
        }
        let loads: Vec<_> = (0..4).map(|e| l.demand(0, e, || None, || Ok(vec![0; 3 * mb])).unwrap()).collect();
        for (e, j) in loads.into_iter().enumerate() {
            assert!(*j.wait().unwrap() == expect(&data, &locs, e, blocks), "expert {e}");
        }
        assert!(l.sh.q.lock().inflight.is_empty());
        // prefetches may finish before being demanded, in which case they load twice (no cache here)
        assert!(done.load(Ordering::Relaxed) >= 4);
        drop(l);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn read_errors_reach_the_waiter() {
        let (path, _, locs) = fixture(64, 1);
        let l = Loader::new(|| Ok(vec![File::open(&path)?]), locs, 64 * BLK, 2, Box::new(|_| {})).unwrap();
        // expert 5 lies past the end of the file
        let j = l.demand(0, 5, || None, || Ok(vec![0; 3 * 64 * BLK])).unwrap();
        assert!(j.wait().is_err());
        drop(l);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cancelled_prefetches_never_run() {
        let blocks = PIECE_BLOCKS * 4;
        let (path, data, locs) = fixture(blocks, 4);
        let mb = blocks * BLK;
        let l = Loader::new(|| Ok(vec![File::open(&path)?]), locs.clone(), mb, 1, Box::new(|_| {})).unwrap();
        for e in 0..4 {
            l.prefetch(0, e, || Ok(vec![0; 3 * mb]));
        }
        let bufs = l.cancel_prefetches(0);
        assert!(bufs.iter().all(|b| b.len() == 3 * mb));
        // a cancelled expert can still be demanded
        let j = l.demand(0, 3, || None, || Ok(vec![0; 3 * mb])).unwrap();
        assert!(*j.wait().unwrap() == expect(&data, &locs, 3, blocks));
        drop(l);
        let _ = std::fs::remove_file(path);
    }
}
