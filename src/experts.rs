//! Routed-expert streaming. Only experts the router actually selects are brought to the GPU.
//!
//! Three tiers:
//!   VRAM  - fixed slots in a few large slab buffers
//!   RAM   - repacked (GPU-ready) expert bytes with a byte budget
//!   disk  - positioned reads straight from the GGUF shards, or from an expert pack (see
//!           pack.rs); unbuffered on Windows so the OS page cache does not duplicate the RAM tier
//!
//! Both caches evict the expert with the lowest *heat*: a use count that halves every
//! `HEAT_HALF_LIFE` forward steps, started from a routing profile. The profile also decides what
//! the caches are filled with while the disk is otherwise idle after startup.
//!
//! An expert is gate|up|down matrices. MXFP4 ones are stored as [qs: rows*nb*16 bytes]
//! [e8m0 scales: rows*nb bytes] (nb = K/32 blocks per row) in both tiers; low-bit ones keep their
//! GGUF bytes in RAM and become 4-bit codes plus scales in VRAM (quants.rs).

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fs::{File, OpenOptions};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use parking_lot::Mutex;

use crate::config::Config;
use crate::expert_io::{Finished, Homes, LayerParts, Load, Loader, Part, PartKind, THREADS};
use crate::gguf::{GGML_MXFP4, Gguf, type_block};
use crate::gpu::{Buf, Gpu};
use crate::quants;
use crate::stats::Stats;

pub type Key = u32;
pub fn key(layer: usize, e: usize) -> Key {
    (layer as u32) << 16 | e as u32
}

/// The expert matrices of one layer, in the order gate, up, down.
#[derive(Clone, Copy, Debug)]
pub struct LayerGeom {
    /// ggml types
    pub types: [u32; 3],
    /// bytes of each in the RAM tier: what it takes in the GGUF
    pub host: [usize; 3],
}

impl LayerGeom {
    pub fn host_len(&self) -> usize {
        self.host.iter().sum()
    }
    /// The NUMA node of every part of an expert in the RAM tier: each matrix's rows are divided
    /// among the nodes. Empty with one node.
    pub fn homes(&self, inter: usize, dim: usize, shares: &[usize]) -> Homes {
        let mut h = Vec::new();
        if shares.len() < 2 {
            return h;
        }
        let mut at = 0;
        for i in 0..3 {
            let (rows, k) = if i == 2 { (dim, inter) } else { (inter, dim) };
            // MXFP4: all values, then all scales
            let regions = if self.types[i] == GGML_MXFP4 { vec![k / 2, k / 32] } else { vec![self.host[i] / rows] };
            for row in regions {
                h.extend(crate::cpu::split(shares, rows).into_iter().enumerate().map(|(n, r)| (n, at + r.start * row..at + r.end * row)));
                at += rows * row;
            }
        }
        h
    }
    /// bytes of each matrix in a VRAM slot
    pub fn slot_lens(&self, inter: usize, dim: usize) -> [usize; 3] {
        self.types.map(|t| quants::slot_len(t, inter, dim))
    }
    /// The VRAM slot contents for an expert as the RAM tier holds it.
    pub fn slot_data<'a>(&self, inter: usize, dim: usize, host: &'a [u8]) -> Cow<'a, [u8]> {
        if self.types == [GGML_MXFP4; 3] {
            return Cow::Borrowed(host);
        }
        let lens = self.slot_lens(inter, dim);
        let mut out = vec![0u8; lens.iter().sum()];
        let (mut src, mut dst) = (host, &mut out[..]);
        for i in 0..3 {
            let (s, d);
            (s, src) = src.split_at(self.host[i]);
            (d, dst) = dst.split_at_mut(lens[i]);
            let (rows, k) = if i == 2 { (dim, inter) } else { (inter, dim) };
            match self.types[i] {
                GGML_MXFP4 => d.copy_from_slice(s),
                t => quants::to_slot(t, s, rows, k, d),
            }
        }
        Cow::Owned(out)
    }
}

pub struct ExpertGeom {
    pub inter: usize,
    pub dim: usize,
    pub layers: Vec<LayerGeom>,
    /// bytes of a VRAM slot: the largest expert
    pub slot_bytes: u64,
    /// mean bytes of an expert in the RAM tier
    pub mean_host: f64,
}

/// The expert matrices of every layer, and where they are in the shards.
pub fn layer_geoms(g: &Gguf, cfg: &Config) -> Result<(Vec<LayerGeom>, Vec<LayerParts>)> {
    let (mut geoms, mut layers) = (Vec::new(), Vec::new());
    for l in 0..cfg.n_layers {
        let mut geom = LayerGeom { types: [GGML_MXFP4; 3], host: [0; 3] };
        let mut parts = Vec::new();
        let mut dst = 0;
        for (i, nm) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"].iter().enumerate() {
            let name = format!("blk.{l}.{nm}.weight");
            let (k, rows) = if i == 2 { (cfg.moe_inter, cfg.dim) } else { (cfg.dim, cfg.moe_inter) };
            let part = if crate::gguf::dev_partial() && !g.has(&name) {
                Part { file: 0, offset: 0, stride: 0, dst, len: rows * k / 32 * 17, kind: PartKind::Zero }
            } else {
                let t = g.tensor(&name)?;
                if t.ty != GGML_MXFP4 && !quants::is_low_bit(t.ty) {
                    bail!("blk.{l}.{nm}: {} experts are not supported", crate::gguf::type_name(t.ty));
                }
                let (be, bb) = type_block(t.ty).unwrap();
                let len = (rows * k) as u64 / be * bb;
                if t.dims != [k as u64, rows as u64, cfg.n_experts as u64] || t.nbytes != len * cfg.n_experts as u64 {
                    bail!("blk.{l}.{nm}: unexpected shape {:?}", t.dims);
                }
                geom.types[i] = t.ty;
                let kind = if t.ty == GGML_MXFP4 { PartKind::Mxfp4 } else { PartKind::Raw };
                Part { file: t.shard, offset: t.file_offset, stride: len, dst, len: len as usize, kind }
            };
            geom.host[i] = part.len;
            dst += part.len;
            parts.push(part);
        }
        if geom.types[0] != geom.types[1] {
            bail!("blk.{l}: gate and up experts of different types are not supported");
        }
        if geom.types != [GGML_MXFP4; 3] && (cfg.dim % 256 != 0 || cfg.moe_inter % 256 != 0) {
            bail!("low-bit experts need dimensions that are multiples of 256");
        }
        geoms.push(geom);
        layers.push(LayerParts { len: dst, parts });
    }
    Ok((geoms, layers))
}

/// forward steps after which a use counts half (chosen by replaying recorded routing traces)
pub const HEAT_HALF_LIFE: f64 = 16.0;
/// background warm-up loads queued at once
const WARM_QUEUE: usize = 4;
/// evicted host buffers kept for reuse
const RECYCLE: usize = 64;

/// Decayed use counts. Stored as log2(count) + step / half-life, so values written at different
/// steps compare directly and never need refreshing.
struct Heat {
    v: Vec<f64>,
    n_experts: usize,
    step: u64,
}

impl Heat {
    fn idx(&self, k: Key) -> usize {
        (k >> 16) as usize * self.n_experts + (k & 0xffff) as usize
    }
    fn get(&self, k: Key) -> f64 {
        self.v[self.idx(k)]
    }
    /// Count one use at `step`; returns the decayed use count.
    fn note(&mut self, k: Key, step: u64) -> f64 {
        self.step = self.step.max(step);
        let t = step as f64 / HEAT_HALF_LIFE;
        let i = self.idx(k);
        let n = (self.v[i] - t).exp2() + 1.0;
        self.v[i] = n.log2() + t;
        n
    }
    /// What `k`'s heat would become if it were used now (for prefetched experts).
    fn if_used(&self, k: Key) -> f64 {
        let t = self.step as f64 / HEAT_HALF_LIFE;
        ((self.get(k) - t).exp2() + 1.0).log2() + t
    }
}

fn ordered(h: f64) -> i64 {
    (h * (1u64 << 24) as f64) as i64
}

struct RamEntry {
    data: Arc<Vec<u8>>,
    prio: i64,
    /// brought in by a prefetch and not used so far
    unused_prefetch: bool,
}

struct RamCache {
    map: HashMap<Key, RamEntry>,
    /// coldest first
    order: BTreeSet<(i64, Key)>,
    bytes: u64,
    limit: u64,
    /// buffers of evicted experts, reused so loads do not fault in fresh memory
    free: Vec<Vec<u8>>,
}

impl RamCache {
    fn get(&mut self, k: Key, heat: f64) -> Option<(Arc<Vec<u8>>, bool)> {
        let e = self.map.get_mut(&k)?;
        self.order.remove(&(e.prio, k));
        e.prio = ordered(heat);
        self.order.insert((e.prio, k));
        Some((e.data.clone(), std::mem::take(&mut e.unused_prefetch)))
    }
    fn evict(&mut self) -> bool {
        let Some((_, k)) = self.order.pop_first() else { return false };
        if let Some(e) = self.map.remove(&k) {
            self.bytes -= e.data.len() as u64;
            if let Ok(b) = Arc::try_unwrap(e.data) {
                self.recycle(b);
            }
        }
        true
    }
    /// A recycled buffer of `n` bytes, if there is one.
    fn reuse(&mut self, n: usize) -> Option<Vec<u8>> {
        let i = self.free.iter().position(|b| b.len() == n)?;
        Some(self.free.swap_remove(i))
    }
    fn recycle(&mut self, b: Vec<u8>) {
        if self.free.len() < RECYCLE {
            self.free.push(b);
        }
    }
    fn fits(&self, sz: u64) -> bool {
        self.bytes + sz <= self.limit
    }
    fn put(&mut self, k: Key, v: Arc<Vec<u8>>, heat: f64, unused_prefetch: bool) {
        let sz = v.len() as u64;
        if self.limit < sz || self.map.contains_key(&k) {
            return;
        }
        while !self.fits(sz) && self.evict() {}
        let prio = ordered(heat);
        self.map.insert(k, RamEntry { data: v, prio, unused_prefetch });
        self.order.insert((prio, k));
        self.bytes += sz;
    }
    /// Commit limit reached: give back a quarter of the tier.
    fn shrink(&mut self) {
        self.limit = self.limit / 4 * 3;
        while self.bytes > self.limit && self.evict() {}
        self.free.clear();
    }
}

/// `n` zeroed bytes straight from the allocator (untouched pages until written).
pub fn alloc_buffer(n: usize) -> Option<Vec<u8>> {
    let layout = std::alloc::Layout::array::<u8>(n).ok()?;
    let p = unsafe { std::alloc::alloc_zeroed(layout) };
    // SAFETY: allocated with Vec<u8>'s own layout, zero-initialized
    (!p.is_null()).then(|| unsafe { Vec::from_raw_parts(p, n, n) })
}

pub struct Slab {
    pub buf: Buf,
}

struct VramState {
    /// slots holding pinned experts are never chosen as victims
    pinned: Vec<bool>,
    slot_key: Vec<Option<Key>>,
    last_use: Vec<u64>,
    /// slots being filled by a promotion
    busy: Vec<bool>,
    map: HashMap<Key, usize>,
    tick: u64,
}

/// Promotions (CPU-run experts copied to VRAM in the background) outstanding at once.
const MAX_PROMOTIONS: usize = 2;

pub struct ExpertStore {
    pub geom: ExpertGeom,
    /// layers with missing (dev-partial) experts are not worth reading
    real_layers: Vec<usize>,
    loader: Loader,
    ram: Arc<Mutex<RamCache>>,
    heat: Arc<Mutex<Heat>>,
    pub slabs: Vec<Slab>,
    slots_per_slab: usize,
    vram: Mutex<VramState>,
    /// held while commands that reference expert slots are recorded and submitted, and while a
    /// promotion takes a slot away
    pub slot_gate: Mutex<()>,
    promoting: std::sync::atomic::AtomicUsize,
    stats: Arc<Stats>,
    pub n_slots: usize,
    /// experts kept in host memory for the whole run (never evicted)
    pinned_ram: Mutex<HashMap<Key, Arc<Vec<u8>>>>,
    /// lifetime selection counts [layer][expert], for building usage profiles
    counts: Mutex<Vec<Vec<u64>>>,
    /// DSV4_EXPERT_OVERRIDE=layer:path - that layer's experts come from a replacement file
    /// (all experts, repacked layout, in index order; written by `dsv4 distill-pilot --emit`)
    override_layer: Option<(usize, File)>,
}

/// A resident expert: which slab, and the byte offset of its slot within it.
#[derive(Clone, Copy, Debug)]
pub struct SlotRef {
    pub slab: usize,
    pub offset: u64,
}

pub fn open_direct(path: &std::path::Path) -> std::io::Result<File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        if std::env::var("DSV4_BUFFERED_IO").is_ok_and(|v| v == "1") {
            return OpenOptions::new().read(true).open(path);
        }
        if let Ok(f) = OpenOptions::new().read(true).custom_flags(FILE_FLAG_NO_BUFFERING).open(path) {
            return Ok(f);
        }
    }
    OpenOptions::new().read(true).open(path)
}

impl ExpertStore {
    pub fn new(g: &Gguf, gpu: &Gpu, cfg: &Config, vram_bytes: u64, ram_bytes: u64, ranking: Option<&[Key]>, pack: Option<&std::path::Path>, stats: Arc<Stats>) -> Result<Self> {
        let (layers, mut locs) = layer_geoms(g, cfg)?;
        let real_layers = (0..locs.len()).filter(|&l| locs[l].parts.iter().all(|p| p.kind != PartKind::Zero)).collect();
        if let Some(p) = pack {
            let packed = crate::pack::open(p, g, &layers, cfg.n_experts, g.shards.len())?;
            stats.event(format!("expert pack: {} of {} layers", packed.len(), locs.len()));
            for (l, p) in locs.iter_mut().zip(packed) {
                *l = p;
            }
        }
        let slot_bytes = layers.iter().map(|l| l.slot_lens(cfg.moe_inter, cfg.dim).iter().sum::<usize>() as u64).max().unwrap();
        let mean_host = layers.iter().map(|l| l.host_len() as f64).sum::<f64>() / layers.len() as f64;
        let geom = ExpertGeom { inter: cfg.moe_inter, dim: cfg.dim, layers, slot_bytes, mean_host };

        let max_bind = gpu.max_binding().min(1 << 31);
        let slots_per_slab = ((max_bind / geom.slot_bytes) as usize).clamp(1, 64);
        let mut n_slots = (vram_bytes / geom.slot_bytes) as usize;
        n_slots = n_slots.max(cfg.n_active * 2);
        let mut slabs = Vec::new();
        let mut left = n_slots;
        while left > 0 {
            let s = left.min(slots_per_slab);
            slabs.push(Slab { buf: gpu.buffer("expert_slab", s as u64 * geom.slot_bytes) });
            left -= s;
        }
        Stats::set(&stats.vram_slots_total, n_slots as u64);
        Stats::set(&stats.ram_cache_limit, ram_bytes);
        let override_layer = match std::env::var("DSV4_EXPERT_OVERRIDE") {
            Ok(v) => {
                let (l, p) = v.split_once(':').ok_or_else(|| anyhow::anyhow!("DSV4_EXPERT_OVERRIDE must be layer:path"))?;
                let l: usize = l.parse()?;
                let f = File::open(p).map_err(|e| anyhow::anyhow!("override file {p}: {e}"))?;
                let want = (geom.layers[l].host_len() * cfg.n_experts) as u64;
                if f.metadata()?.len() != want {
                    bail!("override file {p} is {} bytes, expected {want}", f.metadata()?.len());
                }
                stats.event(format!("layer {l} experts overridden from {p}"));
                Some((l, f))
            }
            Err(_) => None,
        };
        let mut heat = Heat { v: vec![-20.0; cfg.n_layers * cfg.n_experts], n_experts: cfg.n_experts, step: 0 };
        // prior: one use for the hottest, falling to zero for the coldest
        for (i, &k) in ranking.unwrap_or_default().iter().enumerate() {
            let ix = heat.idx(k);
            if ix < heat.v.len() {
                heat.v[ix] = (1.0 - i as f64 / ranking.unwrap().len() as f64).log2();
            }
        }
        let heat = Arc::new(Mutex::new(heat));
        let ram = Arc::new(Mutex::new(RamCache { map: HashMap::new(), order: BTreeSet::new(), bytes: 0, limit: ram_bytes, free: Vec::new() }));
        let paths: Vec<_> = g.shards.iter().map(|s| s.path.as_path()).chain(pack).collect();
        let (r2, h2, s2) = (ram.clone(), heat.clone(), stats.clone());
        let on_done = move |f: Finished| {
            Stats::add(&s2.disk_bytes, f.data.len() as u64);
            Stats::add(&s2.disk_ns, f.ns);
            Stats::add(&s2.expert_disk_loads, 1);
            let heat = {
                let h = h2.lock();
                if f.prefetch { h.if_used(f.key) } else { h.get(f.key) }
            };
            let mut r = r2.lock();
            r.put(f.key, f.data.clone(), heat, f.prefetch);
            Stats::set(&s2.ram_cache_bytes, r.bytes);
            Stats::set(&s2.ram_cache_entries, r.map.len() as u64);
        };
        let homes = geom.layers.iter().map(|l| l.homes(cfg.moe_inter, cfg.dim, crate::cpu::shares())).collect();
        let loader = Loader::new(|| paths.iter().map(|p| open_direct(p)).collect(), Arc::new(locs), homes, THREADS, Box::new(on_done))?;
        Ok(Self {
            geom,
            real_layers,
            loader,
            ram,
            heat,
            slabs,
            slots_per_slab,
            vram: Mutex::new(VramState {
                pinned: vec![false; n_slots],
                slot_key: vec![None; n_slots],
                last_use: vec![0; n_slots],
                busy: vec![false; n_slots],
                map: HashMap::new(),
                tick: 0,
            }),
            slot_gate: Mutex::new(()),
            promoting: Default::default(),
            stats,
            n_slots,
            pinned_ram: Mutex::new(HashMap::new()),
            counts: Mutex::new(vec![vec![0; cfg.n_experts]; cfg.n_layers]),
            override_layer,
        })
    }

    fn slot_ref(&self, slot: usize) -> SlotRef {
        SlotRef { slab: slot / self.slots_per_slab, offset: (slot % self.slots_per_slab) as u64 * self.geom.slot_bytes }
    }

    /// A buffer for one expert of `layer`: recycled if possible.
    fn buffer(&self, layer: usize) -> Result<Vec<u8>> {
        let n = self.geom.layers[layer].host_len();
        if let Some(b) = self.ram.lock().reuse(n) {
            return Ok(b);
        }
        if let Some(b) = alloc_buffer(n) {
            return Ok(b);
        }
        let mut r = self.ram.lock();
        r.shrink();
        Stats::set(&self.stats.ram_cache_limit, r.limit);
        drop(r);
        self.stats.event("allocation failed: shrinking the expert RAM cache");
        alloc_buffer(n).ok_or_else(|| anyhow::anyhow!("out of memory for expert buffer"))
    }

    /// Start reading one expert from disk, unless `cached` (checked atomically) has it.
    fn load(&self, layer: usize, e: usize, cached: impl FnOnce() -> Option<Arc<Vec<u8>>>) -> Result<Load> {
        if let Some((ol, f)) = &self.override_layer {
            if *ol == layer {
                let n = self.geom.layers[layer].host_len();
                let mut out = self.buffer(layer)?;
                #[cfg(windows)]
                std::os::windows::fs::FileExt::seek_read(f, &mut out, (e * n) as u64)?;
                #[cfg(unix)]
                std::os::unix::fs::FileExt::read_exact_at(f, &mut out, (e * n) as u64)?;
                Stats::add(&self.stats.expert_disk_loads, 1);
                let out = Arc::new(out);
                let k = key(layer, e);
                let heat = self.heat.lock().get(k);
                self.ram.lock().put(k, out.clone(), heat, false);
                return Ok(Load::Ready(out));
            }
        }
        self.loader.demand(layer, e, cached, || self.buffer(layer))
    }

    /// Read `n` random experts from disk with `threads` requested at a time (I/O benchmark).
    pub fn bench_disk(&self, n: usize, threads: usize, max_layers: usize) -> Result<(f64, f64)> {
        // a fresh set every time: drives are faster on regions read recently
        let mut seed = 0x2545f4914f6cdd1du64 ^ std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
        let real: Vec<usize> = self.real_layers.iter().copied().filter(|&l| l < max_layers).collect();
        let picks: Vec<(usize, usize)> = (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (real[(seed % real.len() as u64) as usize], ((seed >> 20) % 384) as usize)
            })
            .collect();
        let t0 = Instant::now();
        let mut bytes = 0;
        for batch in picks.chunks(threads) {
            let loads = batch.iter().map(|&(l, e)| self.load(l, e, || None)).collect::<Result<Vec<_>>>()?;
            for l in loads {
                let v = l.wait()?;
                bytes += v.len();
                if let Ok(b) = Arc::try_unwrap(v) {
                    self.ram.lock().recycle(b);
                }
            }
        }
        let s = t0.elapsed().as_secs_f64();
        Ok((bytes as f64 / s / 1e9, s / n as f64 * 1e3))
    }

    /// Host copy of an expert: RAM tier, else join a load in progress, else read from disk.
    /// Returns without waiting for the disk, so several requests can be queued together.
    pub fn request_host(&self, layer: usize, e: usize) -> Result<Load> {
        let k = key(layer, e);
        if let Some(v) = self.pinned_ram.lock().get(&k) {
            Stats::add(&self.stats.expert_ram_hits, 1);
            return Ok(Load::Ready(v.clone()));
        }
        let heat = self.heat.lock().get(k);
        let from_ram = || {
            let (v, first_prefetch_use) = self.ram.lock().get(k, heat)?;
            Stats::add(&self.stats.expert_ram_hits, 1);
            if first_prefetch_use {
                Stats::add(&self.stats.expert_prefetch_hits, 1);
            }
            Some(v)
        };
        if let Some(v) = from_ram() {
            return Ok(Load::Ready(v));
        }
        let load = self.load(layer, e, from_ram)?;
        if let Load::Pending(j) = &load {
            if j.claim_prefetch() {
                Stats::add(&self.stats.expert_prefetch_hits, 1);
            }
        }
        Ok(load)
    }

    /// Start background reads of experts that are likely to be needed soon.
    pub fn prefetch(&self, layer: usize, experts: &[usize]) {
        for &e in experts {
            let k = key(layer, e);
            if self.override_layer.as_ref().is_some_and(|o| o.0 == layer)
                || self.is_resident(layer, e)
                || self.pinned_ram.lock().contains_key(&k)
                || self.ram.lock().map.contains_key(&k)
            {
                continue;
            }
            if self.loader.prefetch(layer, e, || self.buffer(layer)) {
                Stats::add(&self.stats.expert_prefetches, 1);
            }
        }
    }

    /// `layer` has been routed: drop its queued prefetches that nobody asked for.
    pub fn end_prefetch(&self, layer: usize) {
        let bufs = self.loader.cancel_prefetches(layer);
        if !bufs.is_empty() {
            Stats::add(&self.stats.expert_prefetch_cancels, bufs.len() as u64);
            let mut r = self.ram.lock();
            bufs.into_iter().for_each(|b| r.recycle(b));
        }
    }

    /// Fill the caches with `ranking` (hottest first) in the background. Reads only use the disk
    /// while no demand or prefetch is waiting, and stop when the RAM tier is full.
    pub fn warm_up(self: &Arc<Self>, gpu: &Arc<Gpu>, ranking: Vec<Key>) {
        let me = Arc::downgrade(self);
        let gpu = gpu.clone();
        let _ = std::thread::Builder::new().name("expert-warm".into()).spawn(move || {
            let t0 = Instant::now();
            let mut queue = std::collections::VecDeque::new();
            let mut n = 0usize;
            let mut it = ranking.into_iter();
            loop {
                let Some(s) = me.upgrade() else { return };
                while queue.len() < WARM_QUEUE {
                    let Some(k) = it.next() else { break };
                    let (l, e) = ((k >> 16) as usize, (k & 0xffff) as usize);
                    if l >= s.geom.layers.len() || s.override_layer.as_ref().is_some_and(|o| o.0 == l) || s.ram.lock().map.contains_key(&k) {
                        continue;
                    }
                    if !s.ram.lock().fits(s.geom.slot_bytes * (queue.len() as u64 + 1)) {
                        it.by_ref().for_each(drop);
                        break;
                    }
                    if let Some(j) = s.loader.warm(l, e, || s.buffer(l)) {
                        queue.push_back((k, j));
                    }
                }
                let Some((k, j)) = queue.pop_front() else {
                    s.stats.event(format!("cache warm-up: {n} experts in {:.0} s", t0.elapsed().as_secs_f64()));
                    return;
                };
                // do not keep the store alive while waiting on the disk
                drop(s);
                let Ok(v) = j.wait() else { return };
                let Some(s) = me.upgrade() else { return };
                n += 1;
                if s.has_free_slot() {
                    s.promote_now(&gpu, k, &v);
                }
            }
        });
    }

    fn has_free_slot(&self) -> bool {
        let v = self.vram.lock();
        (0..self.n_slots).any(|s| v.slot_key[s].is_none() && !v.busy[s])
    }

    /// The slot to replace: a free one, else the coldest not pinned, busy, or used in `tick`.
    fn victim(&self, v: &VramState, heat: &Heat, tick: u64) -> Option<usize> {
        let usable = |s: &usize| !v.pinned[*s] && !v.busy[*s];
        if let Some(s) = (0..self.n_slots).find(|s| usable(s) && v.slot_key[*s].is_none()) {
            return Some(s);
        }
        (0..self.n_slots)
            .filter(|s| usable(s) && v.last_use[*s] != tick)
            .min_by(|&a, &b| heat.get(v.slot_key[a].unwrap()).total_cmp(&heat.get(v.slot_key[b].unwrap())))
    }

    /// Would this expert earn a VRAM slot? It must have been used repeatedly, and be clearly
    /// hotter than the expert it would replace.
    pub fn worth_promoting(&self, layer: usize, e: usize, uses: f64) -> bool {
        const MIN_USES: f64 = 2.5;
        const MARGIN: f64 = 0.6; // log2: ~1.5x the victim's count
        if uses < MIN_USES || self.promoting.load(std::sync::atomic::Ordering::Relaxed) >= MAX_PROMOTIONS {
            return false;
        }
        let heat = self.heat.lock();
        let v = self.vram.lock();
        match self.victim(&v, &heat, u64::MAX) {
            Some(s) => v.slot_key[s].is_none_or(|old| heat.get(key(layer, e)) > heat.get(old) + MARGIN),
            None => false,
        }
    }

    /// Copy an expert that just ran on the CPU into VRAM, in the background.
    pub fn promote(self: &Arc<Self>, gpu: &Arc<Gpu>, layer: usize, e: usize, data: Arc<Vec<u8>>) {
        use std::sync::atomic::Ordering::Relaxed;
        if self.promoting.fetch_add(1, Relaxed) >= MAX_PROMOTIONS {
            self.promoting.fetch_sub(1, Relaxed);
            return;
        }
        let (me, gpu) = (self.clone(), gpu.clone());
        // a plain thread: it may wait for the slot gate, which must not hold up the rayon pool
        std::thread::spawn(move || {
            me.promote_now(&gpu, key(layer, e), &data);
            me.promoting.fetch_sub(1, Relaxed);
        });
    }

    fn slot_data<'a>(&self, layer: usize, host: &'a [u8]) -> Cow<'a, [u8]> {
        self.geom.layers[layer].slot_data(self.geom.inter, self.geom.dim, host)
    }

    fn promote_now(&self, gpu: &Gpu, k: Key, data: &[u8]) {
        let data = self.slot_data((k >> 16) as usize, data);
        let slot = {
            // once the gate is ours, every command using the victim's old contents is submitted;
            // the queue orders the write below after them
            let _gate = self.slot_gate.lock();
            let heat = self.heat.lock();
            let mut v = self.vram.lock();
            if v.map.contains_key(&k) {
                return;
            }
            let Some(s) = self.victim(&v, &heat, u64::MAX) else { return };
            if let Some(old) = v.slot_key[s].take() {
                v.map.remove(&old);
            }
            v.busy[s] = true;
            s
        };
        let r = self.slot_ref(slot);
        gpu.write(&self.slabs[r.slab].buf, r.offset, &data);
        Stats::add(&self.stats.expert_uploads, 1);
        let mut v = self.vram.lock();
        v.busy[slot] = false;
        // a prefill wave may have brought the same expert in meanwhile
        if !v.map.contains_key(&k) {
            v.slot_key[slot] = Some(k);
            v.map.insert(k, slot);
        }
        let used = v.slot_key.iter().filter(|s| s.is_some()).count();
        Stats::set(&self.stats.vram_slots_used, used as u64);
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
            // queue all reads before waiting for any
            let loads = missing.iter().map(|&(_, e)| self.request_host(layer, e)).collect::<Result<Vec<_>>>()?;
            let datas = loads.into_iter().map(Load::wait).collect::<Result<Vec<_>>>()?;
            let heat = self.heat.lock();
            let mut v = self.vram.lock();
            let tick = v.tick;
            for (&(i, e), d) in missing.iter().zip(datas) {
                let Some(best) = self.victim(&v, &heat, tick) else { bail!("no free expert slot") };
                if let Some(old) = v.slot_key[best].take() {
                    v.map.remove(&old);
                }
                v.slot_key[best] = Some(key(layer, e));
                v.last_use[best] = tick;
                v.map.insert(key(layer, e), best);
                let r = self.slot_ref(best);
                gpu.write(&self.slabs[r.slab].buf, r.offset, &self.slot_data(layer, &d));
                Stats::add(&self.stats.expert_uploads, 1);
                result[i] = Some(best);
            }
            let used = v.slot_key.iter().filter(|s| s.is_some()).count();
            Stats::set(&self.stats.vram_slots_used, used as u64);
        }
        Stats::add(&self.stats.t_expert_ns, t0.elapsed().as_nanos() as u64);
        Ok(result.into_iter().map(|s| self.slot_ref(s.unwrap())).collect())
    }

    /// Record one use of an expert; returns its decayed use count.
    pub fn note_use(&self, layer: usize, e: usize, step: u64) -> f64 {
        self.counts.lock()[layer][e] += 1;
        self.heat.lock().note(key(layer, e), step)
    }

    /// Most experts one `ensure` call may bring in: at most a quarter of the cache, so a prefill
    /// does not flush the hot experts, and never a pinned slot or one a promotion may take.
    pub fn streaming_slots(&self) -> usize {
        let free = self.vram.lock().pinned.iter().filter(|&&p| !p).count();
        free.saturating_sub(MAX_PROMOTIONS + 1).min((self.n_slots / 4).max(8)).max(1)
    }

    pub fn is_resident(&self, layer: usize, e: usize) -> bool {
        self.vram.lock().map.contains_key(&key(layer, e))
    }

    /// Which experts of `layer` are in VRAM or RAM right now.
    pub fn held(&self, layer: usize, n_experts: usize) -> Vec<bool> {
        let v = self.vram.lock();
        let p = self.pinned_ram.lock();
        let r = self.ram.lock();
        (0..n_experts).map(|e| key(layer, e)).map(|k| v.map.contains_key(&k) || p.contains_key(&k) || r.map.contains_key(&k)).collect()
    }

    /// Load `set` (layer, expert) into pinned host memory, then pin as many as fit into VRAM,
    /// keeping `reserve` slots free for streaming anything else. Returns how many went to VRAM.
    pub fn pin(&self, gpu: &Gpu, set: &[(usize, usize)], reserve: usize) -> Result<usize> {
        let mut pr = self.pinned_ram.lock();
        let mut v = self.vram.lock();
        let room = self.n_slots.saturating_sub(reserve);
        let mut in_vram = 0;
        let mut pending = std::collections::VecDeque::new();
        let mut it = set.iter();
        loop {
            while pending.len() < 16 {
                let Some(&(l, e)) = it.next() else { break };
                pending.push_back(((l, e), self.load(l, e, || None)?));
            }
            let Some(((l, e), load)) = pending.pop_front() else { break };
            let b = load.wait()?;
            if in_vram < room {
                let s = in_vram;
                v.slot_key[s] = Some(key(l, e));
                v.pinned[s] = true;
                v.map.insert(key(l, e), s);
                let r = self.slot_ref(s);
                gpu.write(&self.slabs[r.slab].buf, r.offset, &self.slot_data(l, &b));
                in_vram += 1;
                if in_vram % 16 == 0 {
                    gpu.flush();
                }
            }
            pr.insert(key(l, e), b);
        }
        gpu.flush();
        Stats::set(&self.stats.vram_slots_used, in_vram as u64);
        Ok(in_vram)
    }

    /// Add this run's selection counts to the profile at `path` (created if missing).
    pub fn save_profile(&self, path: &std::path::Path) -> Result<()> {
        let mut total: Vec<Vec<u64>> = self.counts.lock().clone();
        if let Ok(old) = load_profile(path) {
            for (t, o) in total.iter_mut().zip(old) {
                for (a, b) in t.iter_mut().zip(o) {
                    *a += b;
                }
            }
        }
        std::fs::write(path, serde_json::to_vec(&serde_json::json!({ "counts": total }))?)?;
        for c in self.counts.lock().iter_mut() {
            c.iter_mut().for_each(|x| *x = 0);
        }
        Ok(())
    }
}

/// Selection counts [layer][expert] from a profile written by `save_profile`.
pub fn load_profile(path: &std::path::Path) -> Result<Vec<Vec<u64>>> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    Ok(serde_json::from_value(v["counts"].clone())?)
}

/// The `n` most-selected experts of every layer according to `counts`.
pub fn top_experts(counts: &[Vec<u64>], n: usize) -> Vec<Vec<usize>> {
    counts
        .iter()
        .map(|c| {
            let mut idx: Vec<usize> = (0..c.len()).collect();
            idx.sort_by_key(|&e| std::cmp::Reverse(c[e]));
            idx.truncate(n);
            idx.sort_unstable();
            idx
        })
        .collect()
}

/// Every expert of a profile, most used first.
pub fn rank_counts(counts: &[Vec<u64>]) -> Vec<Key> {
    let mut all: Vec<(u64, Key)> = counts.iter().enumerate().flat_map(|(l, c)| c.iter().enumerate().map(move |(e, &n)| (n, key(l, e)))).collect();
    all.sort_by_key(|&(n, k)| (std::cmp::Reverse(n), k));
    all.into_iter().map(|x| x.1).collect()
}

/// Built-in ranking of all experts by routing frequency: (u16 layer, u16 expert) pairs, most
/// used first. From Basalt's deepseek41-40x384.prof (MIT, see data/NOTICE).
pub fn builtin_ranking(cfg: &Config) -> Option<Vec<Key>> {
    static DATA: &[u8] = include_bytes!("../data/expert_ranking.bin");
    let keys: Vec<Key> = DATA
        .chunks_exact(4)
        .map(|b| key(u16::from_le_bytes([b[0], b[1]]) as usize, u16::from_le_bytes([b[2], b[3]]) as usize))
        .collect();
    (cfg.n_layers == 40 && cfg.n_experts == 384 && keys.len() == 40 * 384).then_some(keys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_decays_and_orders() {
        let mut h = Heat { v: vec![-20.0; 8], n_experts: 4, step: 0 };
        for s in 1..=4 {
            h.note(key(0, 1), s);
        }
        assert!((h.note(key(0, 2), 4) - 1.0).abs() < 0.01);
        assert!(h.get(key(0, 1)) > h.get(key(0, 2)));
        // one fresh use outweighs four uses several half-lives ago
        h.note(key(1, 0), 4 + 4 * HEAT_HALF_LIFE as u64);
        assert!(h.get(key(1, 0)) > h.get(key(0, 1)));
    }

    #[test]
    fn ram_cache_evicts_coldest_and_recycles() {
        let mut r = RamCache { map: HashMap::new(), order: BTreeSet::new(), bytes: 0, limit: 30, free: Vec::new() };
        r.put(1, Arc::new(vec![1; 10]), 5.0, false);
        r.put(2, Arc::new(vec![2; 10]), 1.0, true);
        r.put(3, Arc::new(vec![3; 10]), 3.0, false);
        assert_eq!(r.get(2, 9.0).map(|x| x.1), Some(true));
        assert_eq!(r.get(2, 9.0).map(|x| x.1), Some(false));
        r.put(4, Arc::new(vec![4; 10]), 4.0, false);
        assert!(!r.map.contains_key(&3) && r.map.len() == 3 && r.bytes == 30);
        assert_eq!(r.free.len(), 1);
    }

    #[test]
    fn homes_cover_every_byte_once() {
        let (inter, dim) = (2304, 5120);
        let mx = LayerGeom { types: [GGML_MXFP4; 3], host: [dim * inter / 32 * 17; 3] };
        let low = LayerGeom { types: [quants::Q2_K; 3], host: [dim * inter / 256 * 84; 3] };
        for g in [mx, low] {
            assert!(g.homes(inter, dim, &[1]).is_empty());
            for shares in [&[1, 1][..], &[32, 8], &[1, 2, 3]] {
                let mut at = 0;
                for (n, r) in g.homes(inter, dim, shares) {
                    assert!(n < shares.len() && r.start == at);
                    at = r.end;
                }
                assert_eq!(at, g.host_len());
            }
        }
    }

    #[test]
    fn builtin_ranking_is_a_permutation() {
        let data = include_bytes!("../data/expert_ranking.bin");
        let mut k: Vec<[u8; 4]> = data.chunks_exact(4).map(|c| c.try_into().unwrap()).collect();
        k.sort();
        k.dedup();
        assert_eq!(k.len(), 40 * 384);
    }
}
