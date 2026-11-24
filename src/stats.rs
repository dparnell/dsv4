//! Counters shared between the engine, server and TUI. Everything is lock-free atomics except
//! a small event log and the per-request history.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::Mutex;

/// One API request, for the status page.
#[derive(Clone, serde::Serialize)]
pub struct ReqRow {
    pub id: u64,
    pub endpoint: &'static str,
    /// unix time it arrived
    pub at: f64,
    /// queued, prefill, generating, or how it ended
    pub state: String,
    pub prompt: usize,
    pub cached: usize,
    pub generated: usize,
    pub prefill_s: Option<f64>,
    /// time to the first token, waiting included
    pub ttft_s: Option<f64>,
    pub tps: Option<f64>,
}

const KEEP_REQUESTS: usize = 100;

#[derive(Default)]
pub struct Stats {
    /// recent requests, oldest first, and the number of all there were
    pub requests: Mutex<(VecDeque<ReqRow>, u64)>,
    // token throughput
    pub prompt_tokens: AtomicU64,
    pub gen_tokens: AtomicU64,
    pub prefill_ns: AtomicU64,
    pub decode_ns: AtomicU64,
    /// last decode step latency (ns) and last prefill rate (tokens/s * 1000)
    pub last_decode_ns: AtomicU64,
    pub last_prefill_tps_milli: AtomicU64,
    // data movement
    pub upload_bytes: AtomicU64,
    pub download_bytes: AtomicU64,
    pub disk_bytes: AtomicU64,
    pub disk_ns: AtomicU64,
    pub engram_bytes: AtomicU64,
    // expert cache
    pub expert_vram_hits: AtomicU64,
    pub expert_ram_hits: AtomicU64,
    pub expert_disk_loads: AtomicU64,
    pub expert_cpu_runs: AtomicU64,
    pub expert_prefetches: AtomicU64,
    /// prefetches that were then used / dropped before they started
    pub expert_prefetch_hits: AtomicU64,
    pub expert_prefetch_cancels: AtomicU64,
    pub expert_uploads: AtomicU64,
    /// routed picks chosen because of --route-prior / dropped by --route-tail-skip
    pub route_moved: AtomicU64,
    pub route_dropped: AtomicU64,
    /// (token, layer) pairs of prompts whose experts were not needed
    pub prompt_rows_skipped: AtomicU64,
    pub vram_slots_total: AtomicU64,
    pub vram_slots_used: AtomicU64,
    pub ram_cache_bytes: AtomicU64,
    pub ram_cache_limit: AtomicU64,
    pub ram_cache_entries: AtomicU64,
    // timing breakdown of the last token (ns)
    pub t_gpu_wait_ns: AtomicU64,
    pub t_expert_ns: AtomicU64,
    pub t_sample_ns: AtomicU64,
    pub t_moe_ns: AtomicU64,
    pub t_cpu_moe_ns: AtomicU64,
    /// the part of it spent computing rather than waiting for experts to be read
    pub t_cpu_compute_ns: AtomicU64,
    // state
    pub ctx_len: AtomicU64,
    pub ctx_max: AtomicU64,
    pub requests_active: AtomicU64,
    pub requests_queued: AtomicU64,
    pub requests_done: AtomicU64,
    pub gpu_dense_bytes: AtomicU64,
    pub gpu_total_bytes: AtomicU64,
    pub load_progress_milli: AtomicU64,
    pub phase: Mutex<String>,
    pub model_name: Mutex<String>,
    pub adapter: Mutex<String>,
    pub events: Mutex<VecDeque<(Instant, String)>>,
    /// per-layer expert hit counters: (vram hits, misses)
    pub layer_hits: Mutex<Vec<(u64, u64)>>,
    /// where the time of recent decoded tokens went
    pub decode_times: Mutex<VecDeque<DecodeTime>>,
    /// recent decode tok/s samples for the sparkline
    pub tps_history: Mutex<VecDeque<u64>>,
    /// tail of the text being generated (for the dashboard)
    pub transcript: Mutex<String>,
}

/// Nanoseconds of one decoded token, or the running totals they are differences of.
#[derive(Clone, Copy, Default)]
pub struct DecodeTime {
    pub total: u64,
    gpu_wait: u64,
    moe: u64,
    cpu_moe: u64,
    cpu_compute: u64,
}

impl DecodeTime {
    /// The counters now; `total` is left to the caller.
    pub fn read(s: &Stats) -> Self {
        let g = Stats::get;
        DecodeTime { total: 0, gpu_wait: g(&s.t_gpu_wait_ns), moe: g(&s.t_moe_ns), cpu_moe: g(&s.t_cpu_moe_ns), cpu_compute: g(&s.t_cpu_compute_ns) }
    }

    /// From `before` to now, taking `total` ns.
    pub fn since(s: &Stats, before: &Self, total: u64) -> Self {
        let n = Self::read(s);
        DecodeTime {
            total,
            gpu_wait: n.gpu_wait - before.gpu_wait,
            moe: n.moe - before.moe,
            cpu_moe: n.cpu_moe - before.cpu_moe,
            cpu_compute: n.cpu_compute - before.cpu_compute,
        }
    }
}

const KEEP_DECODE_TIMES: usize = 100;

impl Stats {
    pub fn push_decode_time(&self, t: DecodeTime) {
        let mut d = self.decode_times.lock();
        d.push_back(t);
        if d.len() > KEEP_DECODE_TIMES {
            d.pop_front();
        }
    }

    /// Mean milliseconds per recent token: computing experts on the CPU, the rest of the CPU path
    /// (nearly all of it waiting for experts to be read), expert work on the GPU side that outlasts
    /// the CPU path, waiting for the rest of the model on the GPU, anything else.
    pub fn decode_breakdown(&self) -> Option<[(&'static str, f64); 5]> {
        let d = self.decode_times.lock();
        if d.is_empty() {
            return None;
        }
        let ms = |f: &dyn Fn(&DecodeTime) -> u64| d.iter().map(f).sum::<u64>() as f64 / 1e6 / d.len() as f64;
        Some([
            ("CPU experts", ms(&|t| t.cpu_compute)),
            ("waiting for disk", ms(&|t| t.cpu_moe.saturating_sub(t.cpu_compute))),
            ("GPU experts", ms(&|t| t.moe.saturating_sub(t.cpu_moe))),
            ("rest of the model (GPU)", ms(&|t| t.gpu_wait)),
            ("other", ms(&|t| t.total.saturating_sub(t.moe + t.gpu_wait))),
        ])
    }

    pub fn add(c: &AtomicU64, v: u64) {
        c.fetch_add(v, Ordering::Relaxed);
    }
    pub fn get(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed)
    }
    pub fn set(c: &AtomicU64, v: u64) {
        c.store(v, Ordering::Relaxed);
    }

    pub fn event(&self, msg: impl Into<String>) {
        let msg = msg.into();
        tracing::info!("{msg}");
        let mut e = self.events.lock();
        e.push_back((Instant::now(), msg));
        while e.len() > 200 {
            e.pop_front();
        }
    }

    pub fn push_text(&self, t: &str) {
        let mut s = self.transcript.lock();
        s.push_str(t);
        if s.len() > 8192 {
            let cut = s.len() - 6144;
            let cut = (cut..s.len()).find(|&i| s.is_char_boundary(i)).unwrap_or(0);
            s.drain(..cut);
        }
    }

    pub fn set_phase(&self, p: impl Into<String>) {
        *self.phase.lock() = p.into();
    }

    /// Log a new request; returns its id.
    pub fn log_request(&self, endpoint: &'static str, prompt: usize) -> u64 {
        let at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
        let mut r = self.requests.lock();
        r.1 += 1;
        let id = r.1;
        r.0.push_back(ReqRow { id, endpoint, at, state: "queued".into(), prompt, cached: 0, generated: 0, prefill_s: None, ttft_s: None, tps: None });
        if r.0.len() > KEEP_REQUESTS {
            r.0.pop_front();
        }
        id
    }

    pub fn update_request(&self, id: u64, f: impl FnOnce(&mut ReqRow)) {
        if let Some(r) = self.requests.lock().0.iter_mut().rev().find(|r| r.id == id) {
            f(r);
        }
    }

    pub fn requests_json(&self) -> serde_json::Value {
        serde_json::json!(self.requests.lock().0.iter().rev().collect::<Vec<_>>())
    }

    pub fn snapshot_json(&self) -> serde_json::Value {
        let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
        serde_json::json!({
            "prompt_tokens": g(&self.prompt_tokens),
            "generated_tokens": g(&self.gen_tokens),
            "prefill_seconds": g(&self.prefill_ns) as f64 / 1e9,
            "decode_seconds": g(&self.decode_ns) as f64 / 1e9,
            "last_decode_ms": g(&self.last_decode_ns) as f64 / 1e6,
            "upload_bytes": g(&self.upload_bytes),
            "disk_bytes": g(&self.disk_bytes),
            "engram_bytes": g(&self.engram_bytes),
            "expert_vram_hits": g(&self.expert_vram_hits),
            "expert_ram_hits": g(&self.expert_ram_hits),
            "expert_disk_loads": g(&self.expert_disk_loads),
            "vram_slots": [g(&self.vram_slots_used), g(&self.vram_slots_total)],
            "ram_cache_bytes": g(&self.ram_cache_bytes),
            "ctx_len": g(&self.ctx_len),
            "requests_done": g(&self.requests_done),
            "gpu_bytes": g(&self.gpu_total_bytes),
        })
    }
}
