//! Routed-expert streaming. Only experts the router actually selects are brought to the GPU.
//!
//! Three tiers:
//!   VRAM  - fixed slots in a few large slab buffers, LRU replaced
//!   RAM   - repacked (GPU-ready) expert bytes, LRU with a byte budget
//!   disk  - positioned reads straight from the GGUF shards (unbuffered on Windows so the
//!           OS page cache does not duplicate the RAM tier)
//!
//! An expert is gate|up|down MXFP4 matrices, each stored on the GPU as
//! [qs: rows*nb*16 bytes][e8m0 scales: rows*nb bytes] (nb = K/32 blocks per row).

use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use parking_lot::Mutex;
use rayon::prelude::*;

use crate::config::Config;
use crate::gguf::{GGML_MXFP4, Gguf};
use crate::gpu::{Buf, Gpu};
use crate::stats::Stats;

pub type Key = u32;
pub fn key(layer: usize, e: usize) -> Key {
    (layer as u32) << 16 | e as u32
}

#[derive(Clone, Copy)]
struct MatLoc {
    shard: usize,
    offset: u64,   // file offset of expert 0
    stride: u64, // bytes per expert
}

pub struct ExpertGeom {
    pub inter: usize,
    pub dim: usize,
    /// bytes of one repacked matrix (all three are equal in size)
    pub mat_bytes: u64,
    pub expert_bytes: u64,
}

struct RamCache {
    map: HashMap<Key, (Arc<Vec<u8>>, u64)>,
    lru: BTreeMap<u64, Key>,
    bytes: u64,
    limit: u64,
    tick: u64,
}

impl RamCache {
    fn get(&mut self, k: Key) -> Option<Arc<Vec<u8>>> {
        self.tick += 1;
        let t = self.tick;
        let e = self.map.get_mut(&k)?;
        self.lru.remove(&e.1);
        e.1 = t;
        self.lru.insert(t, k);
        Some(e.0.clone())
    }
    fn put(&mut self, k: Key, v: Arc<Vec<u8>>) {
        if self.limit == 0 || self.map.contains_key(&k) {
            return;
        }
        let sz = v.len() as u64;
        while self.bytes + sz > self.limit {
            let Some((&t, &old)) = self.lru.iter().next() else { break };
            self.lru.remove(&t);
            if let Some((b, _)) = self.map.remove(&old) {
                self.bytes -= b.len() as u64;
            }
        }
        self.tick += 1;
        self.map.insert(k, (v, self.tick));
        self.lru.insert(self.tick, k);
        self.bytes += sz;
    }
}

pub struct Slab {
    pub buf: Buf,
}

struct VramState {
    slot_key: Vec<Option<Key>>,
    last_use: Vec<u64>,
    map: HashMap<Key, usize>,
    tick: u64,
}

pub struct ExpertStore {
    pub geom: ExpertGeom,
    locs: Vec<[MatLoc; 3]>,
    files: Vec<File>,
    ram: Mutex<RamCache>,
    pub slabs: Vec<Slab>,
    slots_per_slab: usize,
    vram: Mutex<VramState>,
    stats: Arc<Stats>,
    pub n_slots: usize,
    freq: Mutex<HashMap<Key, (u64, f32)>>,
    inflight: Mutex<std::collections::HashSet<Key>>,
    io: rayon::ThreadPool,
}

/// A resident expert: which slab, and the byte offset of its slot within it.
#[derive(Clone, Copy, Debug)]
pub struct SlotRef {
    pub slab: usize,
    pub offset: u64,
}

#[repr(C, align(4096))]
#[derive(Clone, Copy)]
struct Page([u8; 4096]);

fn read_aligned(f: &File, off: u64, len: usize) -> std::io::Result<Vec<u8>> {
    let a0 = off & !4095;
    let a1 = (off + len as u64 + 4095) & !4095;
    let n = (a1 - a0) as usize;
    let mut pages: Vec<Page> = vec![Page([0; 4096]); n / 4096];
    let buf: &mut [u8] = bytemuck_pages(&mut pages);
    let mut done = 0usize;
    while done < n {
        #[cfg(windows)]
        let r = std::os::windows::fs::FileExt::seek_read(f, &mut buf[done..], a0 + done as u64)?;
        #[cfg(unix)]
        let r = std::os::unix::fs::FileExt::read_at(f, &mut buf[done..], a0 + done as u64)?;
        if r == 0 {
            break;
        }
        done += r;
        // unbuffered reads only stop short at end of file; a retry would start unaligned
        if done % 4096 != 0 {
            break;
        }
    }
    let s = (off - a0) as usize;
    if done < s + len {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short expert read"));
    }
    Ok(buf[s..s + len].to_vec())
}

fn bytemuck_pages(p: &mut [Page]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(p.as_mut_ptr() as *mut u8, p.len() * 4096) }
}

fn open_direct(path: &std::path::Path) -> std::io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        if let Ok(f) = OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING).open(path) {
            return Ok(f);
        }
    }
    OpenOptions::new().read(true).open(path)
}

/// MXFP4 rows of 17-byte blocks -> [qs 16B/block][e8m0 1B/block]
fn repack_mxfp4(raw: &[u8], out: &mut [u8]) {
    let nblk = raw.len() / 17;
    let (q, s) = out.split_at_mut(nblk * 16);
    for i in 0..nblk {
        let b = &raw[i * 17..i * 17 + 17];
        s[i] = b[0];
        q[i * 16..i * 16 + 16].copy_from_slice(&b[1..17]);
    }
}

impl ExpertStore {
    pub fn new(g: &Gguf, gpu: &Gpu, cfg: &Config, vram_bytes: u64, ram_bytes: u64, stats: Arc<Stats>) -> Result<Self> {
        let mut locs = Vec::new();
        for l in 0..cfg.n_layers {
            let mut m = [MatLoc { shard: 0, offset: 0, stride: 0 }; 3];
            for (i, nm) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter().enumerate() {
                let name = format!("blk.{l}.{nm}.weight");
                if crate::gguf::dev_partial() && !g.has(&name) {
                    // stand-in geometry; load_disk returns zeros for shard == usize::MAX
                    let (k, rows) = if i == 2 { (cfg.moe_inter, cfg.dim) } else { (cfg.dim, cfg.moe_inter) };
                    m[i] = MatLoc { shard: usize::MAX, offset: 0, stride: (rows * k / 32 * 17) as u64 };
                    continue;
                }
                let t = g.tensor(&name)?;
                if t.ty != GGML_MXFP4 {
                    bail!("blk.{l}.{nm}: expected MXFP4 experts (type 39), got {}", t.ty);
                }
                let ne = t.dims[2];
                m[i] = MatLoc { shard: t.shard, offset: t.file_offset, stride: t.nbytes / ne };
            }
            locs.push(m);
        }
        let mat_bytes = locs[0][0].stride;
        if locs.iter().any(|m| m.iter().any(|x| x.stride != mat_bytes)) {
            bail!("expert matrices of unequal size are not supported");
        }
        let geom = ExpertGeom { inter: cfg.moe_inter, dim: cfg.dim, mat_bytes, expert_bytes: mat_bytes * 3 };
        let files = g.shards.iter().map(|s| open_direct(&s.path)).collect::<std::io::Result<Vec<_>>>()?;

        let max_bind = gpu.max_binding().min(1 << 31);
        let slots_per_slab = ((max_bind / geom.expert_bytes) as usize).clamp(1, 64);
        let mut n_slots = (vram_bytes / geom.expert_bytes) as usize;
        n_slots = n_slots.max(cfg.n_active * 2);
        let mut slabs = Vec::new();
        let mut left = n_slots;
        while left > 0 {
            let s = left.min(slots_per_slab);
            slabs.push(Slab { buf: gpu.buffer("expert_slab", s as u64 * geom.expert_bytes) });
            left -= s;
        }
        Stats::set(&stats.vram_slots_total, n_slots as u64);
        Stats::set(&stats.ram_cache_limit, ram_bytes);
        Ok(Self {
            geom,
            locs,
            files,
            ram: Mutex::new(RamCache { map: HashMap::new(), lru: BTreeMap::new(), bytes: 0, limit: ram_bytes, tick: 0 }),
            slabs,
            slots_per_slab,
            vram: Mutex::new(VramState { slot_key: vec![None; n_slots], last_use: vec![0; n_slots], map: HashMap::new(), tick: 0 }),
            stats,
            n_slots,
            freq: Mutex::new(HashMap::new()),
            inflight: Mutex::new(Default::default()),
            io: rayon::ThreadPoolBuilder::new().num_threads(8).thread_name(|i| format!("expert-io-{i}")).build()?,
        })
    }

    fn slot_ref(&self, slot: usize) -> SlotRef {
        SlotRef { slab: slot / self.slots_per_slab, offset: (slot % self.slots_per_slab) as u64 * self.geom.expert_bytes }
    }

    /// Read + repack one expert from disk.
    fn load_disk(&self, layer: usize, e: usize) -> Result<Vec<u8>> {
        let t0 = Instant::now();
        let n = self.geom.expert_bytes as usize;
        let mut out: Vec<u8> = Vec::new();
        if out.try_reserve_exact(n).is_err() {
            // commit limit reached: give back a quarter of the RAM tier and retry
            let mut r = self.ram.lock();
            r.limit = r.limit / 4 * 3;
            while r.bytes > r.limit {
                let Some((&t, &old)) = r.lru.iter().next() else { break };
                r.lru.remove(&t);
                if let Some((b, _)) = r.map.remove(&old) {
                    r.bytes -= b.len() as u64;
                }
            }
            Stats::set(&self.stats.ram_cache_limit, r.limit);
            drop(r);
            self.stats.event("allocation failed: shrinking the expert RAM cache");
            out.try_reserve_exact(n).map_err(|e| anyhow::anyhow!("out of memory for expert buffer: {e}"))?;
        }
        out.resize(n, 0);
        let mb = self.geom.mat_bytes as usize;
        let raws: Vec<Result<Vec<u8>>> = self.locs[layer]
            .par_iter()
            .map(|m| {
                if m.shard == usize::MAX {
                    // dev-partial stand-in: all-zero blocks with scale exponent 127
                    let mut z = vec![0u8; m.stride as usize];
                    z.chunks_mut(17).for_each(|c| c[0] = 127);
                    return Ok(z);
                }
                Ok(read_aligned(&self.files[m.shard], m.offset + e as u64 * m.stride, m.stride as usize)?)
            })
            .collect();
        for (i, r) in raws.into_iter().enumerate() {
            let raw = r?;
            repack_mxfp4(&raw, &mut out[i * mb..(i + 1) * mb]);
        }
        Stats::add(&self.stats.disk_bytes, self.geom.expert_bytes);
        Stats::add(&self.stats.disk_ns, t0.elapsed().as_nanos() as u64);
        Stats::add(&self.stats.expert_disk_loads, 1);
        Ok(out)
    }

    /// Read `n` random experts from disk with `threads` parallel loaders (I/O benchmark).
    pub fn bench_disk(&self, n: usize, threads: usize) -> Result<(f64, f64)> {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        let mut seed = 0x2545f4914f6cdd1du64;
        let picks: Vec<(usize, usize)> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let real: Vec<usize> = (0..self.locs.len()).filter(|&l| self.locs[l].iter().all(|m| m.shard != usize::MAX)).collect();
                (real[(seed % real.len() as u64) as usize], ((seed >> 20) % 384) as usize)
            })
            .collect();
        let t0 = Instant::now();
        let r: Result<Vec<usize>> = pool.install(|| picks.par_iter().map(|&(l, e)| self.load_disk(l, e).map(|v| v.len())).collect());
        let bytes: usize = r?.iter().sum();
        let s = t0.elapsed().as_secs_f64();
        Ok((bytes as f64 / s / 1e9, s / n as f64 * 1e3))
    }

    /// Host copy of an expert: RAM tier, else wait for an in-flight (prefetch) load, else disk.
    pub fn fetch_host(&self, layer: usize, e: usize) -> Result<Arc<Vec<u8>>> {
        self.fetch_host_inner(layer, e, true)
    }

    fn fetch_host_inner(&self, layer: usize, e: usize, count: bool) -> Result<Arc<Vec<u8>>> {
        let k = key(layer, e);
        let use_ram = self.ram.lock().limit > 0;
        loop {
            if let Some(v) = self.ram.lock().get(k) {
                if count {
                    Stats::add(&self.stats.expert_ram_hits, 1);
                }
                return Ok(v);
            }
            if !use_ram || self.inflight.lock().insert(k) {
                break;
            }
            // another thread (usually the prefetcher) is reading it right now
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        let res = self.load_disk(layer, e).map(|v| {
            let v = Arc::new(v);
            let mut r = self.ram.lock();
            r.put(k, v.clone());
            Stats::set(&self.stats.ram_cache_bytes, r.bytes);
            Stats::set(&self.stats.ram_cache_entries, r.map.len() as u64);
            v
        });
        if use_ram {
            self.inflight.lock().remove(&k);
        }
        res
    }

    /// Start background reads of experts that are likely to be needed soon.
    pub fn prefetch(self: &Arc<Self>, layer: usize, experts: &[usize]) {
        for &e in experts {
            let k = key(layer, e);
            if self.is_resident(layer, e) || self.ram.lock().map.contains_key(&k) || self.inflight.lock().contains(&k) {
                continue;
            }
            Stats::add(&self.stats.expert_prefetches, 1);
            let me = self.clone();
            self.io.spawn(move || {
                let _ = me.fetch_host_inner(layer, e, false);
            });
        }
    }

    /// Make the given experts of `layer` resident; returns their slots in the same order.
    /// At most `n_slots` experts can be requested at once.
    pub fn ensure(&self, gpu: &Gpu, layer: usize, experts: &[usize]) -> Result<Vec<SlotRef>> {
        assert!(experts.len() <= self.n_slots);
        let t0 = Instant::now();
        let mut result = vec![None; experts.len()];
        let mut missing: Vec<(usize, usize)> = Vec::new(); // (index in request, expert)
        {
            let mut v = self.vram.lock();
            v.tick += 1;
            let tick = v.tick;
            for (i, &e) in experts.iter().enumerate() {
                if let Some(&s) = v.map.get(&key(layer, e)) {
                    v.last_use[s] = tick;
                    result[i] = Some(s);
                } else {
                    missing.push((i, e));
                }
            }
        }
        let hits = experts.len() - missing.len();
        Stats::add(&self.stats.expert_vram_hits, hits as u64);
        {
            let mut lh = self.stats.layer_hits.lock();
            if lh.len() <= layer {
                lh.resize(layer + 1, (0, 0));
            }
            lh[layer].0 += hits as u64;
            lh[layer].1 += missing.len() as u64;
        }
        if !missing.is_empty() {
            // fetch host copies in parallel (RAM hits are instant, disk loads overlap)
            let datas: Vec<Result<Arc<Vec<u8>>>> = missing.par_iter().map(|&(_, e)| self.fetch_host(layer, e)).collect();
            let mut v = self.vram.lock();
            let tick = v.tick;
            for (&(i, e), d) in missing.iter().zip(datas) {
                let d = d?;
                // victim: least recently used slot not touched in this request
                let mut best = usize::MAX;
                let mut best_t = u64::MAX;
                for s in 0..self.n_slots {
                    if v.slot_key[s].is_none() {
                        best = s;
                        break;
                    }
                    if v.last_use[s] < best_t && v.last_use[s] != tick {
                        best_t = v.last_use[s];
                        best = s;
                    }
                }
                if best == usize::MAX {
                    bail!("no free expert slot");
                }
                if let Some(old) = v.slot_key[best].take() {
                    v.map.remove(&old);
                }
                v.slot_key[best] = Some(key(layer, e));
                v.last_use[best] = tick;
                v.map.insert(key(layer, e), best);
                let r = self.slot_ref(best);
                gpu.write(&self.slabs[r.slab].buf, r.offset, &d);
                Stats::add(&self.stats.expert_uploads, 1);
                result[i] = Some(best);
            }
            let used = v.slot_key.iter().filter(|s| s.is_some()).count();
            Stats::set(&self.stats.vram_slots_used, used as u64);
        }
        Stats::add(&self.stats.t_expert_ns, t0.elapsed().as_nanos() as u64);
        Ok(result.into_iter().map(|s| self.slot_ref(s.unwrap())).collect())
    }

    /// Record one use of an expert; returns its decayed usage score (uses per ~100 steps).
    pub fn note_use(&self, layer: usize, e: usize, step: u64) -> f32 {
        let mut f = self.freq.lock();
        let ent = f.entry(key(layer, e)).or_insert((step, 0.0));
        let decay = 0.97f32.powi((step - ent.0).min(10_000) as i32);
        ent.1 = ent.1 * decay + 1.0;
        ent.0 = step;
        ent.1
    }

    pub fn is_resident(&self, layer: usize, e: usize) -> bool {
        self.vram.lock().map.contains_key(&key(layer, e))
    }

}
