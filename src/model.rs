//! DeepSeek-V4.1 forward pass on wgpu.
//!
//! A prompt is processed in *segments* of up to `seg` tokens, layer-major: every token of the
//! segment goes through layer l before any goes through layer l+1. Attention runs in chunks of
//! `chunk` tokens (its state is causal and append-only), while the MoE runs over the whole
//! segment at once so each routed expert is fetched from RAM/disk at most once per layer per
//! segment. Decode is simply a one-token segment.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::config::Config;
use crate::engram::Engram;
use crate::experts::ExpertStore;
use crate::gguf::{Gguf, TensorInfo};
use crate::gpu::{B, Batch, Buf, Gpu};
use crate::stats::Stats;
use crate::weights::{Layer, Q8W, f32_buf};

pub struct ModelOptions {
    pub max_ctx: usize,
    pub chunk: usize,
    pub seg: usize,
    /// None: use the free VRAM measured after the dense weights are resident
    pub expert_vram_bytes: Option<u64>,
    pub expert_ram_bytes: u64,
    /// experts in one file (see pack.rs)
    pub expert_pack: Option<std::path::PathBuf>,
    pub backend: Option<String>,
    pub cpu_experts: bool,
    /// All experts, hottest first (None: built-in ranking)
    pub expert_ranking: Option<Vec<crate::experts::Key>>,
    /// Decode picks its experts as if those in VRAM or RAM scored this much higher (lossy; 0: off)
    pub route_prior: f32,
    /// Decode drops picks of this weight rank or lower that would need the disk (lossy; 0: off)
    pub route_tail_skip: usize,
    /// Skip the work of prompt positions that cannot influence the prompt's end
    pub prompt_skip: bool,
    /// Read likely experts of the next layer ahead of its routing
    pub prefetch: bool,
    /// Fill the expert caches from the ranking in the background
    pub warm_up: bool,
    /// Experiment: route only among these experts per layer (pinned in VRAM/RAM up front)
    pub expert_subset: Option<Vec<Vec<usize>>>,
}

/// Per-layer persistent attention state.
struct LayerState {
    wring: Buf,             // [R][hd] sliding-window KV (post-RoPE, fp8-simulated)
    kraw: Option<Buf>,      // [R][hd] compressor kv (kv sources)
    sraw: Option<Buf>,      // [R][hd] compressor gate scores (ratio > 1)
    ccache: Option<Buf>,    // [max_ctx/r][hd] compressed KV (kv sources)
    idxk: Option<Buf>,      // [max_ctx/r][index_hd] indexer keys (kv sources)
}

struct Acts {
    // segment-wide
    h: Buf,
    mix_a: Buf,
    mix_f: Buf,
    xnf: Buf,
    rlog: Buf,
    /// the next layer's router applied to this layer's FFN input, to read ahead
    rnext: Buf,
    hc_part: Buf,
    sg: Buf,
    su: Buf,
    sa: Buf,
    sh_out: Buf,
    o6: Buf,
    hmid: Buf,
    route: Buf,
    comp_idx: Buf,
    cand: Buf,
    widx: Buf,
    // chunk-wide
    xn: Buf,
    qa_raw: Buf,
    qr: Buf,
    q: Buf,
    kv_raw: Buf,
    kvt: Buf,
    ckv: Buf,
    csc: Buf,
    lat: Buf,
    ik_raw: Buf,
    ik: Buf,
    iq: Buf,
    iw: Buf,
    isc: Buf,
    bsc: Buf,
    cidx_tmp: Buf,
    att: Buf,
    o: Buf,
    oa: Buf,
    y: Buf,
    emb: Buf,
    ekv: Buf,
    tmp: Buf,
    logits: Buf,
    dummy: Buf,
}

pub struct Model {
    pub cfg: Config,
    pub g: Arc<Gguf>,
    pub gpu: Arc<Gpu>,
    layers: Vec<Layer>,
    output_norm: Buf,
    head: Q8W,
    embd: TensorInfo,
    pub experts: Arc<ExpertStore>,
    /// router weights on the host, for speculative next-layer prefetch
    engram: Option<Arc<Engram>>,
    rope: Buf,
    rope_yarn_off: u32,
    states: Vec<LayerState>,
    acts: Acts,
    pub max_ctx: usize,
    pub chunk: usize,
    pub seg: usize,
    ring: usize,
    cand_words: usize,
    /// tokens currently represented in the KV state
    pub tokens: Vec<u32>,
    stats: Arc<Stats>,
    cpu_experts: bool,
    prefetch: bool,
    route_prior: f32,
    route_tail_skip: usize,
    prompt_skip: bool,
    /// (from, to): states of `from < n < to` tokens cannot be resumed, their last layers' window
    /// rows were skipped
    gaps: Vec<(usize, usize)>,
    step: std::sync::atomic::AtomicU64,
    snapshot: Option<Snapshot>,
    /// [layer][expert] -> may be routed to (None: all)
    allowed: Option<Vec<Vec<bool>>>,
    /// add expert selection counts to this profile after every request
    pub profile_out: Option<std::path::PathBuf>,
    /// record the routed FFN inputs of one layer (expert distillation experiments)
    pub capture: Option<Capture>,
}

/// Writes one record per (token, routed expert) of `layer`:
/// [u16 expert][u16 slot][f32 route weight][f32 x; dim], x being the normalized FFN input.
pub struct Capture {
    pub layer: usize,
    pub out: std::io::BufWriter<std::fs::File>,
    /// skip the layers after `layer` (the logits are then meaningless)
    pub stop: bool,
    pub records: u64,
}

impl Capture {
    fn write(&mut self, routing: &Routing, x: &[f32], d: usize) -> Result<()> {
        use std::io::Write;
        for (e, list) in routing.per_expert.iter().enumerate() {
            for &(t, slot, w) in list {
                self.out.write_all(&(e as u16).to_le_bytes())?;
                self.out.write_all(&(slot as u16).to_le_bytes())?;
                self.out.write_all(&w.to_le_bytes())?;
                self.out.write_all(bytemuck::cast_slice(&x[t as usize * d..(t as usize + 1) * d]))?;
                self.records += 1;
            }
        }
        Ok(())
    }
}

struct Snapshot {
    tokens: Vec<u32>,
    gaps: Vec<(usize, usize)>,
    bufs: Vec<(Option<Buf>, Option<Buf>, Option<Buf>)>,
}

fn free_vram() -> Option<u64> {
    let n = nvml_wrapper::Nvml::init().ok()?;
    let d = n.device_by_index(0).ok()?;
    Some(d.memory_info().ok()?.free)
}

fn grid1d(n: usize) -> (u32, u32, u32) {
    let groups = (n as u32).div_ceil(256).max(1);
    if groups <= 65535 { (groups, 1, 1) } else { (65535, groups.div_ceil(65535), 1) }
}

fn f(v: f32) -> u32 {
    v.to_bits()
}

/// cos/sin table [max_pos][dim/2][2], optionally YaRN-scaled (matches precompute_freqs_cis).
fn rope_table(dim: usize, max_pos: usize, base: f32, yarn: Option<(usize, f32, f32, f32)>) -> Vec<f32> {
    let half = dim / 2;
    let mut freqs: Vec<f32> = (0..half).map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32)).collect();
    if let Some((orig, factor, beta_fast, beta_slow)) = yarn {
        if orig > 0 {
            let cd = |rot: f32| dim as f32 * ((orig as f32) / (rot * 2.0 * std::f32::consts::PI)).ln() / (2.0 * base.ln());
            let low = cd(beta_fast).floor().max(0.0);
            let high = cd(beta_slow).ceil().min(dim as f32 - 1.0);
            for (i, fr) in freqs.iter_mut().enumerate() {
                let ramp = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *fr = *fr / factor * (1.0 - smooth) + *fr * smooth;
            }
        }
    }
    let mut t = vec![0f32; max_pos * half * 2];
    t.par_chunks_mut(half * 2).enumerate().for_each(|(p, row)| {
        for i in 0..half {
            let a = (p as f32 * freqs[i]) as f64;
            row[2 * i] = a.cos() as f32;
            row[2 * i + 1] = a.sin() as f32;
        }
    });
    t
}

struct Guard<F: FnMut()>(F);
impl<F: FnMut()> Drop for Guard<F> {
    fn drop(&mut self) {
        (self.0)()
    }
}
fn scopeguard<F: FnMut()>(f: F) -> Guard<F> {
    Guard(f)
}

pub struct Routing {
    /// per expert: list of (token, k-slot, weight)
    pub per_expert: Vec<Vec<(u32, u32, f32)>>,
    /// picks changed / dropped by a `RoutePref`
    pub moved: u64,
    pub dropped: u64,
}

/// The longest resumable state of at most `n` tokens: inside a gap, fall back to its start.
fn resumable(n: usize, gaps: &[(usize, usize)]) -> usize {
    gaps.iter().rev().fold(n, |n, &(a, b)| if a < n && n < b { a } else { n })
}

/// Lossy decode routing that favours experts already in memory.
pub struct RoutePref {
    /// added to the score held experts are picked by (their weights are not changed)
    pub prior: f32,
    /// picks of this weight rank (0-based) or lower that are not held are dropped; 0: none
    pub tail_skip: usize,
    /// [expert] -> in VRAM or RAM
    pub held: Vec<bool>,
}

pub fn route_score(x: f32) -> f32 {
    (if x > 20.0 { x } else { x.exp().ln_1p() }).sqrt()
}

/// Longest input whose next layer's experts are guessed and read ahead.
const PREFETCH_MAX_TOKENS: usize = 4;

/// Pick the experts of every token from row `from` on; rows before it get none.
pub fn route(cfg: &Config, logits: &[f32], bias: &[f32], t: usize, from: usize, allowed: Option<&[bool]>, pref: Option<&RoutePref>) -> Routing {
    let ne = cfg.n_experts;
    let k = cfg.n_active;
    let mut per_expert: Vec<Vec<(u32, u32, f32)>> = vec![Vec::new(); ne];
    let picks: Vec<(Vec<(usize, f32)>, u64, u64)> = (0..t)
        .into_par_iter()
        .map(|ti| {
            if ti < from {
                return (Vec::new(), 0, 0);
            }
            let row = &logits[ti * ne..(ti + 1) * ne];
            let scores: Vec<f32> = row.iter().map(|&x| route_score(x)).collect();
            let mut idx: Vec<usize> = match allowed {
                Some(m) => (0..ne).filter(|&e| m[e]).collect(),
                None => (0..ne).collect(),
            };
            let k = k.min(idx.len());
            let plain = |e: usize| scores[e] + bias[e];
            idx.select_nth_unstable_by(k - 1, |&a, &b| plain(b).total_cmp(&plain(a)));
            let (mut moved, mut dropped) = (0, 0);
            let mut sel = idx[..k].to_vec();
            if let Some(p) = pref.filter(|p| p.prior > 0.0) {
                let s = |e: usize| plain(e) + if p.held[e] { p.prior } else { 0.0 };
                idx.select_nth_unstable_by(k - 1, |&a, &b| s(b).total_cmp(&s(a)));
                moved = idx[..k].iter().filter(|e| !sel.contains(e)).count() as u64;
                sel = idx[..k].to_vec();
            }
            let sum: f32 = sel.iter().map(|&e| scores[e]).sum::<f32>() + 1e-20;
            if let Some(p) = pref.filter(|p| p.tail_skip > 0) {
                sel.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
                let kept: Vec<usize> = sel.iter().enumerate().filter(|&(r, &e)| r < p.tail_skip || p.held[e]).map(|x| *x.1).collect();
                dropped = (sel.len() - kept.len()) as u64;
                sel = kept;
            }
            (sel.iter().map(|&e| (e, scores[e] / sum * cfg.route_scale)).collect(), moved, dropped)
        })
        .collect();
    let (mut moved, mut dropped) = (0, 0);
    for (ti, (p, m, d)) in picks.iter().enumerate() {
        moved += m;
        dropped += d;
        for (slot, &(e, w)) in p.iter().enumerate() {
            per_expert[e].push((ti as u32, slot as u32, w));
        }
    }
    Routing { per_expert, moved, dropped }
}

impl Model {
    pub fn load(path: &std::path::Path, opt: &ModelOptions, stats: Arc<Stats>) -> Result<Self> {
        stats.set_phase("opening GGUF");
        let g = Arc::new(Gguf::open(path)?);
        let cfg = Config::from_gguf(&g)?;
        *stats.model_name.lock() = g.str("general.name").unwrap_or("DeepSeek-V4.1").to_string();
        stats.event(format!(
            "model: {} layers, dim {}, {} experts (top-{}), {} shards, {:.1} GB",
            cfg.n_layers,
            cfg.dim,
            cfg.n_experts,
            cfg.n_active,
            g.shards.len(),
            g.total_bytes() as f64 / 1e9
        ));
        stats.set_phase("initializing GPU");
        let gpu = Arc::new(Gpu::new(stats.clone(), opt.backend.as_deref())?);
        *stats.adapter.lock() = format!("{} ({:?})", gpu.info.name, gpu.info.backend);
        stats.event(format!(
            "GPU: {} via {:?}, max binding {:.2} GB",
            gpu.info.name,
            gpu.info.backend,
            gpu.max_binding() as f64 / 1e9
        ));

        stats.set_phase("uploading dense weights");
        let mut layers = Vec::new();
        for l in 0..cfg.n_layers {
            layers.push(Layer::load(&g, &gpu, &cfg, l)?);
            gpu.flush();
            Stats::set(&stats.load_progress_milli, ((l + 1) * 1000 / (cfg.n_layers + 1)) as u64);
            Stats::set(&stats.gpu_total_bytes, gpu.gpu_bytes.load(std::sync::atomic::Ordering::Relaxed));
        }
        let output_norm = f32_buf(&g, &gpu, "output_norm.weight")?;
        let head = Q8W::load(&g, &gpu, "output.weight")?;
        gpu.flush();
        let embd = g.tensor("token_embd.weight")?.clone();
        if !crate::quants::can_dequantize(embd.ty) {
            bail!("token_embd: {} is not supported", crate::gguf::type_name(embd.ty));
        }
        let dense = gpu.gpu_bytes.load(std::sync::atomic::Ordering::Relaxed);
        Stats::set(&stats.gpu_dense_bytes, dense);
        stats.event(format!("dense weights resident: {:.2} GB", dense as f64 / 1e9));

        let max_ctx = opt.max_ctx.next_multiple_of(256);
        let chunk = opt.chunk.clamp(1, 512);
        let seg = opt.seg.max(chunk);
        let ring = 128 + 896; // window + rollback / chunk slack
        assert!(chunk + cfg.window <= ring);
        let hd = cfg.head_dim;
        let ihd = cfg.index_head_dim;

        // rope tables: plain (ratio-0 layers) and YaRN at the compress theta
        let rd = cfg.rope_dim;
        let mut tab = rope_table(rd, max_ctx, cfg.rope_theta, None);
        let yarn_off = tab.len() as u32;
        tab.extend(rope_table(rd, max_ctx, cfg.compress_rope_theta, Some((cfg.rope_orig_ctx, cfg.rope_factor, cfg.beta_fast, cfg.beta_slow))));
        let rope = gpu.buffer_init("rope", bytemuck::cast_slice(&tab));

        let mut states = Vec::new();
        for l in 0..cfg.n_layers {
            let r = cfg.ratio(l);
            let src = cfg.is_kv_source(l);
            states.push(LayerState {
                wring: gpu.buffer("wring", (ring * hd * 4) as u64),
                kraw: src.then(|| gpu.buffer("kraw", (ring * hd * 4) as u64)),
                sraw: (src && r > 1).then(|| gpu.buffer("sraw", (ring * hd * 4) as u64)),
                ccache: src.then(|| gpu.buffer("ccache", ((max_ctx / r + 1) * hd * 4) as u64)),
                idxk: src.then(|| gpu.buffer("idxk", ((max_ctx / r + 1) * ihd * 4) as u64)),
            });
        }
        let d = cfg.dim;
        let inter = cfg.moe_inter;
        let na = cfg.n_active;
        let topk = cfg.index_topk;
        // bitset row stride, padded so per-chunk sub-ranges stay 256-byte aligned
        let cand_words = (max_ctx / cfg.candidate_block.max(1)).div_ceil(32).next_multiple_of(64);
        let fb = |n: usize| (n * 4) as u64;
        let a = Acts {
            h: gpu.buffer("h", fb(seg * cfg.hc_mult * d)),
            mix_a: gpu.buffer("mix_a", fb(seg * 24)),
            mix_f: gpu.buffer("mix_f", fb(seg * 24)),
            xnf: gpu.buffer("xnf", fb(seg * d)),
            rnext: gpu.buffer("rnext", fb(PREFETCH_MAX_TOKENS * cfg.n_experts)),
            hc_part: gpu.buffer("hc_part", fb(seg * 25 * (cfg.hc_mult * cfg.dim / 4).div_ceil(crate::kernels::HC_PART as usize))),
            rlog: gpu.buffer("rlog", fb(seg * cfg.n_experts)),
            sg: gpu.buffer("sg", fb(seg * inter)),
            su: gpu.buffer("su", fb(seg * inter)),
            sa: gpu.buffer("sa", fb(seg * inter)),
            sh_out: gpu.buffer("sh_out", fb(seg * d)),
            o6: gpu.buffer("o6", fb(seg * na * d)),
            hmid: gpu.buffer("hmid", fb(seg * na * inter)),
            route: gpu.buffer("route", fb(seg * na * 3)),
            comp_idx: gpu.buffer("comp_idx", fb(seg * topk)),
            cand: gpu.buffer("cand", fb(seg * cand_words)),
            widx: gpu.buffer("widx", fb(seg * cfg.window)),
            xn: gpu.buffer("xn", fb(chunk * d)),
            qa_raw: gpu.buffer("qa_raw", fb(chunk * cfg.q_lora_rank)),
            qr: gpu.buffer("qr", fb(chunk * cfg.q_lora_rank)),
            q: gpu.buffer("q", fb(chunk * cfg.n_heads * hd)),
            kv_raw: gpu.buffer("kv_raw", fb(chunk * hd)),
            kvt: gpu.buffer("kvt", fb(chunk * hd)),
            ckv: gpu.buffer("ckv", fb(chunk * hd)),
            csc: gpu.buffer("csc", fb(chunk * hd)),
            lat: gpu.buffer("lat", fb((chunk + 1) * hd)),
            ik_raw: gpu.buffer("ik_raw", fb((chunk + 1) * ihd)),
            ik: gpu.buffer("ik", fb((chunk + 1) * ihd)),
            iq: gpu.buffer("iq", fb(chunk * cfg.index_heads * ihd)),
            iw: gpu.buffer("iw", fb(chunk * cfg.index_heads)),
            isc: gpu.buffer("isc", fb(chunk * (max_ctx + 1))),
            bsc: gpu.buffer("bsc", fb(chunk * (max_ctx / cfg.candidate_block.max(1) + 1))),
            cidx_tmp: gpu.buffer("cidx_tmp", fb(chunk * cfg.candidate_topk_blocks.max(1))),
            att: gpu.buffer("att", fb(chunk * cfg.n_heads * (cfg.window + topk))),
            o: gpu.buffer("o", fb(chunk * cfg.n_heads * hd)),
            oa: gpu.buffer("oa", fb(chunk * cfg.o_groups * cfg.o_lora_rank)),
            y: gpu.buffer("y", fb(chunk * d)),
            emb: gpu.buffer("emb", fb(chunk * cfg.n_hash_cols().max(1) * cfg.engram_head_dim.max(1))),
            ekv: gpu.buffer("ekv", fb(chunk * (cfg.hc_mult + 1) * d)),
            tmp: gpu.buffer("tmp", fb(d)),
            logits: gpu.buffer("logits", fb(cfg.vocab)),
            dummy: gpu.buffer("dummy", 256),
        };
        let act_bytes = gpu.gpu_bytes.load(std::sync::atomic::Ordering::Relaxed) - dense;
        stats.event(format!("activations + KV state: {:.2} GB (ctx {max_ctx}, segment {seg}, chunk {chunk})", act_bytes as f64 / 1e9));

        stats.set_phase("allocating expert cache");
        let vram = match opt.expert_vram_bytes {
            Some(v) => v,
            None => {
                gpu.flush();
                let free = free_vram().unwrap_or(4_000_000_000);
                let v = free.saturating_sub(700_000_000);
                stats.event(format!("free VRAM {:.2} GB -> {:.2} GB for experts", free as f64 / 1e9, v as f64 / 1e9));
                v
            }
        };
        let ranking = opt.expert_ranking.clone().or_else(|| crate::experts::builtin_ranking(&cfg));
        let experts = Arc::new(ExpertStore::new(&g, &gpu, &cfg, vram, opt.expert_ram_bytes, ranking.as_deref(), opt.expert_pack.as_deref(), stats.clone())?);
        let allowed = match &opt.expert_subset {
            Some(sub) => {
                stats.set_phase("pinning expert subset");
                let set: Vec<(usize, usize)> = sub.iter().enumerate().flat_map(|(l, es)| es.iter().map(move |&e| (l, e))).collect();
                let in_vram = experts.pin(&gpu, &set, cfg.n_active * 2)?;
                stats.event(format!(
                    "expert subset: {} per layer, {} pinned ({} in VRAM, {} in RAM, {:.1} GB)",
                    sub.first().map_or(0, |s| s.len()),
                    set.len(),
                    in_vram,
                    set.len() - in_vram,
                    set.len() as f64 * experts.geom.mean_host / 1e9
                ));
                Some(
                    sub.iter()
                        .map(|es| {
                            let mut m = vec![false; cfg.n_experts];
                            es.iter().for_each(|&e| m[e] = true);
                            m
                        })
                        .collect(),
                )
            }
            None => None,
        };
        stats.event(format!(
            "expert cache: {} VRAM slots ({:.2} GB), RAM tier {:.1} GB, {:.1} MB/expert",
            experts.n_slots,
            experts.n_slots as f64 * experts.geom.slot_bytes as f64 / 1e9,
            opt.expert_ram_bytes as f64 / 1e9,
            experts.geom.mean_host / 1e6
        ));
        if let (true, Some(r), None) = (opt.warm_up, ranking, &opt.expert_subset) {
            experts.warm_up(&gpu, r);
        }
        let engram = Engram::new(&g, &cfg, stats.clone())?.map(Arc::new);
        Stats::set(&stats.gpu_total_bytes, gpu.gpu_bytes.load(std::sync::atomic::Ordering::Relaxed));
        Stats::set(&stats.ctx_max, max_ctx as u64);
        Stats::set(&stats.load_progress_milli, 1000);
        stats.set_phase("ready");
        Ok(Self {
            cfg,
            g,
            gpu,
            layers,
            output_norm,
            head,
            embd,
            experts,
            engram,
            rope,
            rope_yarn_off: yarn_off,
            states,
            acts: a,
            max_ctx,
            chunk,
            seg,
            ring,
            cand_words,
            tokens: Vec::new(),
            stats,
            prefetch: opt.prefetch,
            route_prior: opt.route_prior,
            route_tail_skip: opt.route_tail_skip,
            prompt_skip: opt.prompt_skip,
            gaps: Vec::new(),
            cpu_experts: opt.cpu_experts && crate::cpu_moe::has_simd(),
            step: std::sync::atomic::AtomicU64::new(0),
            snapshot: None,
            allowed,
            profile_out: None,
            capture: None,
        })
    }

    /// How far back the rings still hold every position a rollback target needs.
    fn slack(&self) -> usize {
        self.ring - self.cfg.window - 1
    }

    fn usable(&self, state_tokens: &[u32], gaps: &[(usize, usize)], prompt: &[u32]) -> usize {
        let common = state_tokens.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let common = common.min(prompt.len().saturating_sub(1)); // always recompute the last token
        // inside a gap: fall back to its start
        let common = resumable(common, gaps);
        if state_tokens.len() - common <= self.slack() { common } else { 0 }
    }

    /// Bring the KV state to the longest reusable prefix of `prompt`, using either the live state
    /// or the snapshot taken after the previous prefill, and return its length. The compressed
    /// caches are position-indexed and append-only, so only the rings need restoring.
    pub fn prepare(&mut self, prompt: &[u32]) -> usize {
        let live = self.usable(&self.tokens, &self.gaps, prompt);
        let snap = self.snapshot.as_ref().map(|s| self.usable(&s.tokens, &s.gaps, prompt)).unwrap_or(0);
        if snap > live {
            let s = self.snapshot.as_ref().unwrap();
            let mut b = self.gpu.begin();
            for (st, sb) in self.states.iter().zip(&s.bufs) {
                for (dst, src) in [(Some(&st.wring), sb.0.as_ref()), (st.kraw.as_ref(), sb.1.as_ref()), (st.sraw.as_ref(), sb.2.as_ref())] {
                    if let (Some(d), Some(s)) = (dst, src) {
                        b.copy(s, 0, &d.buf, 0, d.size);
                    }
                }
            }
            b.submit();
            self.tokens = s.tokens.clone();
            self.gaps = s.gaps.clone();
            self.truncate(snap);
            self.stats.event(format!("restored KV snapshot ({snap} tokens reused)"));
            snap
        } else {
            self.truncate(live);
            live
        }
    }

    pub fn drop_snapshot(&mut self) {
        if let Some(s) = &mut self.snapshot {
            s.tokens.clear();
        }
    }

    /// Save the ring buffers so a later request can resume from the current position.
    pub fn take_snapshot(&mut self) {
        let snap = self.snapshot.get_or_insert_with(|| Snapshot {
            tokens: Vec::new(),
            gaps: Vec::new(),
            bufs: self
                .states
                .iter()
                .map(|st| {
                    let mk = |b: &Buf| self.gpu.buffer("snapshot", b.size);
                    (Some(mk(&st.wring)), st.kraw.as_ref().map(mk), st.sraw.as_ref().map(mk))
                })
                .collect(),
        });
        let mut b = self.gpu.begin();
        for (st, sb) in self.states.iter().zip(&snap.bufs) {
            for (src, dst) in [(Some(&st.wring), sb.0.as_ref()), (st.kraw.as_ref(), sb.1.as_ref()), (st.sraw.as_ref(), sb.2.as_ref())] {
                if let (Some(s), Some(d)) = (src, dst) {
                    b.copy(s, 0, &d.buf, 0, s.size);
                }
            }
        }
        b.submit();
        snap.tokens = self.tokens.clone();
        snap.gaps = self.gaps.clone();
    }

    pub fn truncate(&mut self, n: usize) {
        self.tokens.truncate(n);
        self.gaps.retain(|g| g.0 < n);
        Stats::set(&self.stats.ctx_len, n as u64);
    }

    fn embed(&self, toks: &[u32]) -> Vec<f32> {
        let d = self.cfg.dim;
        let raw = self.g.bytes(&self.embd);
        let row_bytes = raw.len() / (self.embd.n_elements() as usize / d);
        let hc = self.cfg.hc_mult;
        let mut out = vec![0f32; toks.len() * hc * d];
        out.par_chunks_mut(hc * d).zip(toks.par_iter()).for_each(|(o, &t)| {
            let r = &raw[t as usize * row_bytes..(t as usize + 1) * row_bytes];
            crate::quants::dequantize(self.embd.ty, r, &mut o[..d]);
            let (first, rest) = o.split_at_mut(d);
            for c in rest.chunks_exact_mut(d) {
                c.copy_from_slice(first);
            }
        });
        out
    }

    fn rmsnorm(&self, b: &mut Batch, x: &Buf, x_off: usize, w: Option<&Buf>, y: &Buf, y_off: usize, d: usize, rows: usize) {
        let wb = w.unwrap_or(&self.acts.dummy);
        b.run(
            "rmsnorm",
            &[x.all(), wb.all(), y.all()],
            &[d as u32, x_off as u32, y_off as u32, d as u32, d as u32, f(self.cfg.norm_eps), w.is_some() as u32],
            (rows as u32, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn rope(&self, b: &mut Batch, x: &Buf, rows: usize, heads: usize, stride: usize, pos_base: usize, pos_step: usize, yarn: bool, inverse: bool) {
        let half = self.cfg.rope_dim / 2;
        b.run(
            "rope",
            &[x.all(), self.rope.all()],
            &[
                rows as u32,
                heads as u32,
                stride as u32,
                (stride - self.cfg.rope_dim) as u32,
                pos_base as u32,
                pos_step as u32,
                inverse as u32,
                if yarn { self.rope_yarn_off } else { 0 },
                0,
                half as u32,
            ],
            grid1d(rows * half),
        );
    }

    fn quant(&self, b: &mut Batch, x: &Buf, rows: usize, row_len: usize, block: usize, mode: u32) {
        let nb = rows * row_len / block;
        b.run("quant_sim", &[x.all()], &[nb as u32, block as u32, mode, 0, row_len as u32, row_len as u32], grid1d(nb));
    }

    /// Hyper-connection mixes of tokens `c0..c0+m` of the residual stream.
    fn hc_mix(&self, b: &mut Batch, wf: &Buf, sb: &Buf, mix: &Buf, c0: usize, m: usize) {
        let (cfg, a) = (&self.cfg, &self.acts);
        let width = (cfg.hc_mult * cfg.dim) as u32;
        let parts = (width / 4).div_ceil(crate::kernels::HC_PART);
        b.run("hc_dot", &[a.h.all(), wf.all(), a.hc_part.all()], &[width, c0 as u32], (parts, m as u32, 1));
        b.run(
            "hc_mix",
            &[a.hc_part.all(), sb.all(), mix.all()],
            &[width, cfg.hc_mult as u32, cfg.hc_iters as u32, f(cfg.norm_eps), f(cfg.hc_eps), 0, (c0 * 24) as u32, parts],
            (m as u32, 1, 1),
        );
    }

    fn copy(&self, b: &mut Batch, src: &Buf, so: usize, dst: &Buf, d_o: usize, n: usize) {
        b.run("copy", &[src.all(), dst.all()], &[n as u32, so as u32, d_o as u32], grid1d(n));
    }

    /// Run the model over `toks` (appended at the current position). Returns logits of the last
    /// token.
    pub fn forward(&mut self, toks: &[u32]) -> Result<Vec<f32>> {
        let p0 = self.tokens.len();
        let end = p0 + toks.len();
        // Every state of `resume` or more tokens stays usable for a later request that diverges
        // near the end of this prompt; only what those states need is computed.
        let margin = (toks.len() / 8).clamp(16, 512);
        let resume = (self.prompt_skip && self.capture.is_none() && self.cfg.skip_layer().is_some() && end > p0 + margin).then(|| end - margin);
        let mut logits = Vec::new();
        let mut i = 0;
        while i < toks.len() {
            let n = (toks.len() - i).min(self.seg);
            logits = self.forward_segment(&toks[i..i + n], resume)?;
            i += n;
        }
        if let Some(r) = resume {
            self.gaps.push((p0, r));
        }
        Ok(logits)
    }

    /// First positions whose window keys, and whose whole output, layer `l` must produce so that
    /// position `resume` and all later ones come out exactly.
    fn needed_from(&self, l: usize, resume: Option<usize>) -> (usize, usize) {
        let (Some(r), Some(y)) = (resume, self.cfg.skip_layer()) else { return (0, 0) };
        if l < y {
            return (0, 0);
        }
        // each layer looks back one window further than the one above it
        let back = self.cfg.window - 1;
        let out = r.saturating_sub(back * (self.cfg.n_layers - 1 - l));
        (out.saturating_sub(back), out)
    }

    fn forward_segment(&mut self, toks: &[u32], resume: Option<usize>) -> Result<Vec<f32>> {
        let cfg = self.cfg.clone();
        let t_all = toks.len();
        let p0 = self.tokens.len();
        if p0 + t_all > self.max_ctx {
            bail!("context overflow: {} + {} > {}", p0, t_all, self.max_ctx);
        }
        self.tokens.extend_from_slice(toks);
        self.step.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let gpu = self.gpu.clone();
        let d = cfg.dim;
        let w = cfg.window;
        let a = &self.acts;

        // engram rows: start the disk reads now, they only depend on token ids; each layer waits for its own
        let mut eng_rows: Vec<crate::engram::Rows> = match &self.engram {
            Some(e) => (0..e.layers.len()).map(|li| e.start(&self.g, li, &self.tokens, p0, t_all)).collect(),
            None => Vec::new(),
        };

        // residual stream: hc_mult copies of the embedding
        let h0 = self.embed(toks);
        gpu.write(&a.h, 0, bytemuck::cast_slice(&h0));
        // sliding-window slots per token
        let mut widx = vec![-1i32; t_all * w];
        for i in 0..t_all {
            let p = p0 + i;
            for j in 0..w {
                if p + j + 1 >= w {
                    let q = p + j + 1 - w;
                    widx[i * w + j] = (q % self.ring) as i32;
                }
            }
        }
        gpu.write(&a.widx, 0, bytemuck::cast_slice(&widx));

        let n_chunks = t_all.div_ceil(self.chunk);
        let mut cand_active = vec![false; n_chunks];
        let mut batch = gpu.begin();
        for l in 0..cfg.n_layers {
            let ly = &self.layers[l];
            let ratio = cfg.ratio(l);
            let yarn = ratio > 0;
            let (keys_from, out_from) = self.needed_from(l, resume);
            let last = p0 + t_all - 1;
            // ---------------- attention, chunk by chunk ----------------
            for c in 0..n_chunks {
                let c0 = c * self.chunk;
                let m = (t_all - c0).min(self.chunk);
                let pc = p0 + c0;
                let keys_only = pc + m - 1 < out_from;
                if pc + m - 1 < keys_from && self.cfg.skip_layer() != Some(l) {
                    continue;
                }
                if let (Some(e), true) = (&ly.engram, self.engram.is_some()) {
                    let li = cfg.engram_layers.iter().position(|&x| x == l).unwrap();
                    let width = self.engram.as_ref().unwrap().cols() * self.engram.as_ref().unwrap().head_dim;
                    // the emb buffer is reused per chunk, so each chunk needs its own submit
                    batch.submit();
                    gpu.write(&a.emb, 0, bytemuck::cast_slice(&eng_rows[li].get()[c0 * width..(c0 + m) * width]));
                    batch = gpu.begin();
                    e.kv.run(&mut batch, &a.emb, 0, width, &a.ekv, 0, (cfg.hc_mult + 1) * d, m, 0, false);
                    batch.run("engram", &[a.h.all(), a.ekv.all(), e.gate.all()], &[d as u32, f(cfg.norm_eps), c0 as u32], (m as u32, cfg.hc_mult as u32, 1));
                }
                // hyper-connection mixes for attention, then collapse with the previous FFN's pre mix
                self.hc_mix(&mut batch, &ly.hc_attn_fn, &ly.hc_attn_sb, &a.mix_a, c0, m);
                batch.run(
                    "hc_pre_norm",
                    &[a.h.all(), a.mix_f.all(), ly.attn_norm.all(), a.xn.all()],
                    &[d as u32, (l == 0) as u32, f(cfg.norm_eps), c0 as u32, 0, 0],
                    (m as u32, 1, 1),
                );
                self.attention(&mut batch, l, c, m, pc, c0, yarn, &mut cand_active, keys_only)?;
                if keys_only {
                    continue;
                }
                batch.run(
                    "hc_post",
                    &[a.h.all(), a.y.all(), a.dummy.all(), a.mix_a.all()],
                    &[d as u32, m as u32, 0, 0, c0 as u32],
                    grid1d(m * d),
                );
            }
            if last < out_from {
                // nothing here is needed past this layer
                batch.submit();
                Stats::add(&self.stats.prompt_rows_skipped, (t_all * (cfg.n_layers - l)) as u64);
                return Ok(Vec::new());
            }
            let row_from = out_from.saturating_sub(p0);
            Stats::add(&self.stats.prompt_rows_skipped, row_from as u64);
            // ---------------- MoE over the whole segment ----------------
            self.hc_mix(&mut batch, &ly.hc_ffn_fn, &ly.hc_ffn_sb, &a.mix_f, 0, t_all);
            batch.run(
                "hc_pre_norm",
                &[a.h.all(), a.mix_a.all(), ly.ffn_norm.all(), a.xnf.all()],
                &[d as u32, 0, f(cfg.norm_eps), 0, 0, 0],
                (t_all as u32, 1, 1),
            );
            ly.gate_inp.run(&mut batch, &a.xnf, 0, d, &a.rlog, 0, cfg.n_experts, t_all);
            // guess the next layer's experts by applying its router to this layer's FFN input (the
            // residual stream changes slowly)
            let guess = self.prefetch && self.cpu_experts && l + 1 < cfg.n_layers && t_all <= PREFETCH_MAX_TOKENS && self.allowed.is_none();
            if guess {
                self.layers[l + 1].gate_inp.run(&mut batch, &a.xnf, 0, d, &a.rnext, 0, cfg.n_experts, t_all);
            }
            // shared expert (overlaps with the routing readback)
            ly.sh_gate.run(&mut batch, &a.xnf, 0, d, &a.sg, 0, cfg.moe_inter, t_all, 0, false);
            ly.sh_up.run(&mut batch, &a.xnf, 0, d, &a.su, 0, cfg.moe_inter, t_all, 0, false);
            batch.run("swiglu", &[a.sg.all(), a.su.all(), a.sa.all()], &[(t_all * cfg.moe_inter) as u32, f(cfg.swiglu_limit)], grid1d(t_all * cfg.moe_inter));
            ly.sh_down.run(&mut batch, &a.sa, 0, cfg.moe_inter, &a.sh_out, 0, d, t_all, 0, false);
            let tw = Instant::now();
            let mut ranges = vec![(&a.rlog, 0u64, (t_all * cfg.n_experts * 4) as u64)];
            let want_x = self.cpu_experts || self.capture.as_ref().is_some_and(|c| c.layer == l);
            if want_x {
                ranges.push((&a.xnf, 0, (t_all * d * 4) as u64));
            }
            if guess {
                ranges.push((&a.rnext, 0, (t_all * cfg.n_experts * 4) as u64));
            }
            let mut back = gpu.read_multi(batch, &ranges);
            Stats::add(&self.stats.t_gpu_wait_ns, tw.elapsed().as_nanos() as u64);
            let next: Option<Vec<f32>> = guess.then(|| bytemuck::cast_slice(&back.pop().unwrap()).to_vec());
            let xnf_host: Option<Vec<f32>> = if want_x { Some(bytemuck::cast_slice(&back.pop().unwrap()).to_vec()) } else { None };
            let rl: Vec<f32> = bytemuck::cast_slice(&back.pop().unwrap()).to_vec();
            let pref = (t_all == 1 && (self.route_prior > 0.0 || self.route_tail_skip > 0))
                .then(|| RoutePref { prior: self.route_prior, tail_skip: self.route_tail_skip, held: self.experts.held(l, cfg.n_experts) });
            let routing = route(&cfg, &rl, &ly.exp_bias, t_all, row_from, self.allowed.as_ref().map(|a| a[l].as_slice()), pref.as_ref());
            Stats::add(&self.stats.route_moved, routing.moved);
            Stats::add(&self.stats.route_dropped, routing.dropped);
            if let Some(cap) = self.capture.as_mut().filter(|c| c.layer == l) {
                cap.write(&routing, xnf_host.as_deref().unwrap(), d)?;
                if cap.stop {
                    return Ok(vec![0.0; cfg.vocab]);
                }
            }
            if let Some(next) = next {
                self.prefetch_next(l + 1, &next, t_all);
            }
            self.run_experts(l, &routing, xnf_host.as_deref())?;
            batch = gpu.begin();
            batch.run(
                "hc_post",
                &[a.h.all(), a.sh_out.all(), a.o6.all(), a.mix_f.all()],
                &[d as u32, t_all as u32, cfg.n_active as u32, 0, 0],
                grid1d(t_all * d),
            );
        }
        // final collapse + head on the last token
        let a = &self.acts;
        batch.run(
            "hc_pre_norm",
            &[a.h.all(), a.mix_f.all(), self.output_norm.all(), a.tmp.all()],
            &[d as u32, 0, f(cfg.norm_eps), (t_all - 1) as u32, 0, 0],
            (1, 1, 1),
        );
        self.head.run(&mut batch, &a.tmp, 0, d, &a.logits, 0, cfg.vocab, 1, 0, false);
        let logits = gpu.read_f32(batch, &a.logits, 0, cfg.vocab);
        Stats::set(&self.stats.ctx_len, self.tokens.len() as u64);
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(&self, b: &mut Batch, l: usize, c: usize, m: usize, pc: usize, c0: usize, yarn: bool, cand_active: &mut [bool], keys_only: bool) -> Result<()> {
        let cfg = &self.cfg;
        let a = &self.acts;
        let ly = &self.layers[l];
        let st = &self.states[l];
        let d = cfg.dim;
        let hd = cfg.head_dim;
        let nh = cfg.n_heads;
        let w = cfg.window;
        let ratio = cfg.ratio(l);
        let ihd = cfg.index_head_dim;
        let ih = cfg.index_heads;
        let topk = cfg.index_topk;

        // q
        ly.q_a.run(b, &a.xn, 0, d, &a.qa_raw, 0, cfg.q_lora_rank, m, 0, false);
        self.rmsnorm(b, &a.qa_raw, 0, Some(&ly.q_a_norm), &a.qr, 0, cfg.q_lora_rank, m);
        ly.q_b.run(b, &a.qr, 0, cfg.q_lora_rank, &a.q, 0, nh * hd, m, 0, false);
        self.rope(b, &a.q, m * nh, nh, hd, pc, 1, yarn, false);
        // sliding-window kv
        ly.kv.run(b, &a.xn, 0, d, &a.kv_raw, 0, hd, m, 0, false);
        self.rmsnorm(b, &a.kv_raw, 0, Some(&ly.kv_norm), &a.kvt, 0, hd, m);
        self.rope(b, &a.kvt, m, 1, hd, pc, 1, yarn, false);
        self.quant(b, &a.kvt, m, hd, 32, 0);
        b.run("copy_ring", &[a.kvt.all(), st.wring.all()], &[m as u32, hd as u32, pc as u32, self.ring as u32, 0, 0], grid1d(m * hd));

        let mut n_slots = w;
        let mut kc: &Buf = &a.dummy;
        if ratio > 0 {
            let src = cfg.kv_source_for(l).unwrap();
            let sst = &self.states[src];
            if cfg.is_kv_source(l) {
                let comp = ly.compressor.as_ref().unwrap();
                comp.kv.run(b, &a.xn, 0, d, &a.ckv, 0, hd, m, 0, false);
                b.run("copy_ring", &[a.ckv.all(), st.kraw.as_ref().unwrap().all()], &[m as u32, hd as u32, pc as u32, self.ring as u32, 0, 0], grid1d(m * hd));
                if let Some(gw) = &comp.gate {
                    gw.run(b, &a.xn, 0, d, &a.csc, 0, hd, m, 0, false);
                    b.run("copy_ring", &[a.csc.all(), st.sraw.as_ref().unwrap().all()], &[m as u32, hd as u32, pc as u32, self.ring as u32, 0, 0], grid1d(m * hd));
                }
                let j_lo = (pc + ratio) / ratio - 1;
                let j_hi = (pc + m) / ratio;
                if j_hi > j_lo {
                    let g = j_hi - j_lo;
                    let sr = st.sraw.as_ref().unwrap_or(&a.dummy);
                    b.run(
                        "compress",
                        &[st.kraw.as_ref().unwrap().all(), sr.all(), comp.norm.all(), a.lat.all()],
                        &[g as u32, hd as u32, j_lo as u32, ratio as u32, self.ring as u32, f(cfg.norm_eps)],
                        (g as u32, 1, 1),
                    );
                    if let Some(ind) = &ly.indexer {
                        if let (Some(wk), Some(kn)) = (&ind.attn_k, &ind.k_norm) {
                            wk.run(b, &a.lat, 0, hd, &a.ik_raw, 0, ihd, g, 0, false);
                            self.rmsnorm(b, &a.ik_raw, 0, Some(kn), &a.ik, 0, ihd, g);
                            self.rope(b, &a.ik, g, 1, ihd, j_lo * ratio, ratio, yarn, false);
                            self.quant(b, &a.ik, g, ihd, 32, 1);
                            self.copy(b, &a.ik, 0, st.idxk.as_ref().unwrap(), j_lo * ihd, g * ihd);
                        }
                    }
                    self.rope(b, &a.lat, g, 1, hd, j_lo * ratio, ratio, yarn, false);
                    self.quant(b, &a.lat, g, hd, 16, 2);
                    self.copy(b, &a.lat, 0, st.ccache.as_ref().unwrap(), j_lo * hd, g * hd);
                }
            }
            if keys_only {
                return Ok(());
            }
            let n_comp = (pc + m) / ratio;
            let k_eff = topk.min(n_comp);
            if cfg.is_index_source(l) && k_eff > 0 {
                let ind = ly.indexer.as_ref().unwrap();
                ind.q_b.run(b, &a.qr, 0, cfg.q_lora_rank, &a.iq, 0, ih * ihd, m, 0, false);
                self.rope(b, &a.iq, m * ih, ih, ihd, pc, 1, yarn, false);
                self.quant(b, &a.iq, m * ih, ihd, 32, 1);
                ind.proj.run(b, &a.xn, 0, d, &a.iw, 0, ih, m, 0, false);
                let cand_src = cfg.candidate_source.unwrap_or(usize::MAX);
                let use_cand = l > cand_src && cand_active[c];
                let cand_bytes = (self.cand_words * 4) as u64;
                let cand_b = a.cand.range(c0 as u64 * cand_bytes, (m as u64 * cand_bytes).max(16));
                b.run(
                    "idx_scores",
                    &[a.iq.all(), a.iw.all(), sst.idxk.as_ref().unwrap().all(), cand_b, a.isc.all()],
                    &[
                        n_comp as u32,
                        ih as u32,
                        ihd as u32,
                        pc as u32,
                        ratio as u32,
                        f((ihd as f32).powf(-0.5) * (ih as f32).powf(-0.5)),
                        use_cand as u32,
                        cfg.candidate_block as u32,
                        self.cand_words as u32,
                    ],
                    ((n_comp as u32).div_ceil(256), m as u32, 1),
                );
                let blk = cfg.candidate_block.max(1);
                let n_blocks = n_comp.div_ceil(blk);
                if l == cand_src && n_blocks > cfg.candidate_topk_blocks {
                    let kb = cfg.candidate_topk_blocks;
                    b.run("block_max", &[a.isc.all(), a.bsc.all()], &[n_comp as u32, blk as u32, n_blocks as u32, pc as u32, ratio as u32], ((n_blocks as u32).div_ceil(256), m as u32, 1));
                    b.run("topk", &[a.bsc.all(), a.cidx_tmp.all()], &[n_blocks as u32, kb as u32, 0, 1, kb as u32, 0, n_blocks as u32, 0], (m as u32, 1, 1));
                    let words = m * self.cand_words;
                    b.run("zero", &[a.cand.all()], &[words as u32, (c0 * self.cand_words) as u32], grid1d(words));
                    b.run("cand_mark", &[a.cidx_tmp.all(), cand_b], &[kb as u32, self.cand_words as u32, m as u32], ((kb as u32).div_ceil(256), m as u32, 1));
                    cand_active[c] = true;
                }
                b.run(
                    "topk",
                    &[a.isc.all(), a.comp_idx.all()],
                    &[n_comp as u32, k_eff as u32, pc as u32, ratio as u32, topk as u32, (c0 * topk) as u32, 0, 0],
                    (m as u32, 1, 1),
                );
            }
            if k_eff > 0 {
                n_slots = w + k_eff;
                kc = sst.ccache.as_ref().unwrap();
            }
        }

        let widx_b = a.widx.range((c0 * w * 4) as u64, (m * w * 4) as u64);
        let scale = (hd as f32).powf(-0.5);
        let ab: [B; 6] = [a.q.all(), st.wring.all(), kc.all(), widx_b, a.comp_idx.all(), a.att.all()];
        b.run(
            "attn_scores",
            &ab,
            &[n_slots as u32, w as u32, topk as u32, nh as u32, hd as u32, f(scale), (c0 * topk) as u32],
            ((n_slots as u32).div_ceil(16), (nh / 16) as u32, m as u32),
        );
        b.run("attn_softmax", &[a.att.all(), ly.sinks.all()], &[n_slots as u32, nh as u32], (nh as u32, m as u32, 1));
        let pb: [B; 6] = [a.att.all(), st.wring.all(), kc.all(), widx_b, a.comp_idx.all(), a.o.all()];
        b.run(
            "attn_pv",
            &pb,
            &[n_slots as u32, w as u32, topk as u32, nh as u32, hd as u32, 0, (c0 * topk) as u32],
            ((hd / 16) as u32, (nh / 16) as u32, m as u32),
        );
        self.rope(b, &a.o, m * nh, nh, hd, pc, 1, yarn, true);
        self.quant(b, &a.o, m * nh, hd, 32, 0);
        ly.out_a.run(b, &a.o, 0, nh * hd, &a.oa, 0, cfg.o_groups * cfg.o_lora_rank, m, cfg.o_lora_rank, false);
        ly.out_b.run(b, &a.oa, 0, cfg.o_groups * cfg.o_lora_rank, &a.y, 0, d, m, 0, false);
        Ok(())
    }

    /// Routed experts for one layer over the segment.
    ///
    /// Resident experts (and experts used by many tokens) run on the GPU in waves that fit the
    /// VRAM slots. With `cpu_experts`, the remaining cold experts
    /// run on the CPU straight from the RAM tier, concurrently with the GPU work, and their
    /// outputs are uploaded into the same per-(token, k) rows. Those that keep getting picked are
    /// then copied to VRAM in the background for later steps.
    fn run_experts(&self, l: usize, routing: &Routing, xnf: Option<&[f32]>) -> Result<()> {
        const CPU_MAX_TOKENS: usize = 4;
        let cfg = &self.cfg;
        let gpu = &self.gpu;
        let step = self.step.load(std::sync::atomic::Ordering::Relaxed);
        trace_routing(step, l, routing);
        if routing.dropped > 0 {
            // a dropped pick's output row still holds an earlier layer's values
            let mut used = vec![false; cfg.n_active];
            routing.per_expert.iter().flatten().for_each(|&(_, k, _)| used[k as usize] = true);
            let zero = vec![0u8; cfg.dim * 4];
            for k in (0..cfg.n_active).filter(|&k| !used[k]) {
                gpu.write(&self.acts.o6, (k * cfg.dim * 4) as u64, &zero);
            }
        }
        let mut gpu_list: Vec<usize> = Vec::new();
        let mut cpu_list: Vec<usize> = Vec::new();
        let mut uses = vec![0f64; cfg.n_experts];
        for e in 0..cfg.n_experts {
            let n = routing.per_expert[e].len();
            if n == 0 {
                continue;
            }
            uses[e] = self.experts.note_use(l, e, step);
            let to_gpu = xnf.is_none() || n > CPU_MAX_TOKENS || self.experts.is_resident(l, e);
            if to_gpu {
                gpu_list.push(e)
            } else {
                cpu_list.push(e)
            }
        }
        // resident experts first: they need no transfer
        gpu_list.sort_by_key(|&e| !self.experts.is_resident(l, e));

        let t_moe = std::time::Instant::now();
        // queue every disk read before waiting on any
        let loads = if xnf.is_some() {
            cpu_list.iter().map(|&e| self.experts.request_host(l, e)).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        self.experts.end_prefetch(l);
        let (gpu_res, cpu_res) = rayon::join(
            || self.run_experts_gpu(l, routing, &gpu_list),
            || -> Result<CpuOut> {
                let Some(x) = xnf else { return Ok(Default::default()) };
                let t_cpu = std::time::Instant::now();
                let _timer = scopeguard(|| Stats::add(&self.stats.t_cpu_moe_ns, t_cpu.elapsed().as_nanos() as u64));
                let d = cfg.dim;
                let geom = &self.experts.geom;
                let run = |batch: Vec<(usize, Arc<Vec<u8>>)>, out: &mut CpuOut| {
                    let jobs: Vec<crate::cpu_moe::Job> = batch
                        .iter()
                        .map(|(e, blob)| crate::cpu_moe::Job {
                            geom: geom.layers[l],
                            blob,
                            xs: routing.per_expert[*e].iter().map(|&(t, _, _)| &x[t as usize * d..(t as usize + 1) * d]).collect(),
                            ws: routing.per_expert[*e].iter().map(|&(_, _, w)| w).collect(),
                        })
                        .collect();
                    let t_run = std::time::Instant::now();
                    let ys = crate::cpu_moe::experts_forward(&jobs, geom.inter, d, cfg.swiglu_limit);
                    Stats::add(&self.stats.t_cpu_compute_ns, t_run.elapsed().as_nanos() as u64);
                    for ((e, _), ys) in batch.iter().zip(ys) {
                        out.rows.extend(routing.per_expert[*e].iter().zip(ys).map(|(&(t, k, _), y)| (t as usize * cfg.n_active + k as usize, y)));
                    }
                    out.blobs.extend(batch);
                };
                // experts already in memory compute while the others are being read
                let (ready, pending): (Vec<_>, Vec<_>) =
                    cpu_list.iter().copied().zip(loads).partition(|(_, l)| matches!(l, crate::expert_io::Load::Ready(_)));
                let mut all = CpuOut::default();
                let ready = ready.into_iter().map(|(e, l)| Ok((e, l.wait()?))).collect::<Result<Vec<_>>>()?;
                // in groups, so a prompt's many cold experts do not need all their buffers at once
                let mut ready = ready.into_iter();
                loop {
                    let group: Vec<_> = ready.by_ref().take(16).collect();
                    if group.is_empty() {
                        break;
                    }
                    run(group, &mut all);
                }
                for (e, l) in pending {
                    run(vec![(e, l.wait()?)], &mut all);
                }
                Ok(all)
            },
        );
        Stats::add(&self.stats.t_moe_ns, t_moe.elapsed().as_nanos() as u64);
        gpu_res?;
        let out = cpu_res?;
        Stats::add(&self.stats.expert_cpu_runs, cpu_list.len() as u64);
        for (row, y) in out.rows {
            gpu.write(&self.acts.o6, (row * cfg.dim * 4) as u64, bytemuck::cast_slice(&y));
        }
        for (e, blob) in out.blobs {
            if self.experts.worth_promoting(l, e, uses[e]) {
                self.experts.promote(gpu, l, e, blob);
            }
        }
        Ok(())
    }

    /// Start reading the experts that `logits`, a guess of the next layer's router output, picks.
    fn prefetch_next(&self, next: usize, logits: &[f32], t_all: usize) {
        let cfg = &self.cfg;
        let bias = &self.layers[next].exp_bias;
        // expect what `route` will pick
        let held = (self.route_prior > 0.0).then(|| self.experts.held(next, cfg.n_experts));
        let keep = if self.route_tail_skip > 0 { self.route_tail_skip.min(cfg.n_active) } else { cfg.n_active };
        let mut want: Vec<(usize, usize)> = Vec::new();
        for t in 0..t_all {
            let scores: Vec<f32> = logits[t * cfg.n_experts..(t + 1) * cfg.n_experts]
                .iter()
                .enumerate()
                .map(|(e, &z)| {
                    route_score(z) + bias[e] + if held.as_ref().is_some_and(|h| h[e]) { self.route_prior } else { 0.0 }
                })
                .collect();
            let mut idx: Vec<usize> = (0..cfg.n_experts).collect();
            // best guesses first: the rest is dropped if the disk has not got to it in time
            idx.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));
            for (r, &e) in idx[..keep].iter().enumerate() {
                want.push((r, e));
            }
        }
        want.sort_unstable();
        let mut seen = std::collections::HashSet::new();
        let want: Vec<usize> = want.into_iter().map(|x| x.1).filter(|&e| seen.insert(e)).collect();
        self.experts.prefetch(next, &want);
    }

    fn run_experts_gpu(&self, l: usize, routing: &Routing, list: &[usize]) -> Result<()> {
        let cfg = &self.cfg;
        let a = &self.acts;
        let gpu = &self.gpu;
        let cap = self.experts.streaming_slots();
        let geom = &self.experts.geom;
        let types = geom.layers[l].types;
        let lens = geom.layers[l].slot_lens(geom.inter, geom.dim);
        let (up_word, down_word) = ((lens[0] / 4) as u32, ((lens[0] + lens[1]) / 4) as u32);
        // p7.. of the low-bit kernels
        let extra = |ty: u32| -> [u32; 5] {
            let lut: &[u32; 4] = bytemuck::cast_ref(crate::quants::codes(ty).0);
            [crate::quants::has_min(ty) as u32, lut[0], lut[1], lut[2], lut[3]]
        };
        let lut = types.map(|t| t != crate::gguf::GGML_MXFP4);
        let mut waves = list.chunks(cap).peekable();
        while let Some(wave) = waves.next() {
            // read the next wave while this one is uploaded and recorded
            if let Some(next) = waves.peek() {
                self.experts.prefetch(l, next);
            }
            // slots must stay put from lookup until the commands using them are submitted
            let gate = self.experts.slot_gate.lock();
            let slots = self.experts.ensure(gpu, l, wave)?;
            let mut entries: Vec<u32> = Vec::new();
            let mut ranges = Vec::new();
            for &e in wave {
                let start = entries.len() / 3;
                for &(tok, slot, wt) in &routing.per_expert[e] {
                    entries.extend_from_slice(&[tok, tok * cfg.n_active as u32 + slot, wt.to_bits()]);
                }
                ranges.push((start, routing.per_expert[e].len()));
            }
            gpu.write(&a.route, 0, bytemuck::cast_slice(&entries));
            let mut b = gpu.begin();
            for (s, &(start, cnt)) in slots.iter().zip(&ranges) {
                let slab = &self.experts.slabs[s.slab].buf;
                let sb = slab.range(s.offset, geom.slot_bytes);
                let tm = if cnt == 1 { 1 } else { 8 };
                let n = (tm != 1) as usize;
                let (gk, dk) = (crate::kernels::moe(types[0])[n], crate::kernels::moe(types[2])[2 + n]);
                let mut gp = vec![geom.inter as u32, geom.dim as u32, cnt as u32, start as u32, f(cfg.swiglu_limit), up_word, cfg.dim as u32];
                if lut[0] {
                    gp.extend(extra(types[0]));
                }
                let mut dp = vec![geom.dim as u32, geom.inter as u32, cnt as u32, start as u32, 0, down_word];
                if lut[2] {
                    dp.push(0);
                    dp.extend(extra(types[2]));
                }
                b.run(
                    gk,
                    &[sb, a.xnf.all(), a.route.all(), a.hmid.all()],
                    &gp,
                    ((geom.inter as u32).div_ceil(8), (cnt as u32).div_ceil(tm), 1),
                );
                b.run(
                    dk,
                    &[sb, a.hmid.all(), a.route.all(), a.o6.all()],
                    &dp,
                    ((geom.dim as u32).div_ceil(8), (cnt as u32).div_ceil(tm), 1),
                );
            }
            b.submit();
            drop(gate);
        }
        Ok(())
    }
}

/// Results of one layer's CPU experts: (output row, values), and the weights they ran from.
#[derive(Default)]
struct CpuOut {
    rows: Vec<(usize, Vec<f32>)>,
    blobs: Vec<(usize, Arc<Vec<u8>>)>,
}

/// DSV4_ROUTE_TRACE=path appends every layer's routed experts as one line,
/// `step layer expert:tokens ...`, for simulating cache policies offline.
fn trace_routing(step: u64, l: usize, routing: &Routing) {
    use std::io::Write;
    static OUT: std::sync::OnceLock<Option<std::sync::Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let out = OUT.get_or_init(|| {
        let p = std::env::var("DSV4_ROUTE_TRACE").ok()?;
        Some(std::sync::Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(p).ok()?))
    });
    let Some(f) = out else { return };
    let mut line = format!("{step} {l}");
    for (e, list) in routing.per_expert.iter().enumerate().filter(|(_, v)| !v.is_empty()) {
        line.push_str(&format!(" {e}:{}", list.len()));
    }
    line.push(char::from(10));
    let _ = f.lock().unwrap().write_all(line.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pick(r: &Routing) -> Vec<(usize, f32)> {
        r.per_expert.iter().enumerate().flat_map(|(e, l)| l.iter().map(move |x| (e, x.2))).collect()
    }

    #[test]
    fn resume_avoids_gaps() {
        let gaps = [(0, 100), (130, 400)];
        for (n, want) in [(50, 0), (100, 100), (120, 120), (130, 130), (399, 130), (400, 400), (450, 450)] {
            assert_eq!(resumable(n, &gaps), want, "{n}");
        }
        // gaps that meet
        assert_eq!(resumable(250, &[(0, 210), (200, 300)]), 0);
    }

    #[test]
    fn route_preferences() {
        let mut cfg = Config::default();
        cfg.n_experts = 6;
        cfg.n_active = 2;
        cfg.route_scale = 1.0;
        let logits = [3.0, 2.0, 1.95, 0.0, -1.0, -2.0f32];
        let bias = [0.0; 6];
        let plain = route(&cfg, &logits, &bias, 1, 0, None, None);
        assert_eq!(pick(&plain).iter().map(|x| x.0).collect::<Vec<_>>(), [0, 1]);
        // expert 2 is nearly as good as 1 and in memory
        let held = vec![false, false, true, false, false, true];
        let p = RoutePref { prior: 0.1, tail_skip: 0, held: held.clone() };
        let r = route(&cfg, &logits, &bias, 1, 0, None, Some(&p));
        assert_eq!(pick(&r).iter().map(|x| x.0).collect::<Vec<_>>(), [0, 2]);
        assert_eq!((r.moved, r.dropped), (1, 0));
        // weights come from the unbiased scores
        let w = route_score(3.0) / (route_score(3.0) + route_score(1.95));
        assert!((pick(&r)[0].1 - w).abs() < 1e-6);
        // the tail skip keeps the remaining weight as it was
        let p = RoutePref { prior: 0.0, tail_skip: 1, held };
        let r = route(&cfg, &logits, &bias, 1, 0, None, Some(&p));
        assert_eq!(pick(&r), vec![pick(&plain)[0]]);
        assert_eq!(r.dropped, 1);
        // rows before `from` get nothing
        let two: Vec<f32> = logits.iter().chain(&logits).copied().collect();
        let r = route(&cfg, &two, &bias, 2, 1, None, None);
        assert!(r.per_expert.iter().flatten().all(|x| x.0 == 1));
    }
}
