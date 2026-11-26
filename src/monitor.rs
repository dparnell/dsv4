//! What the web status page shows: a thread samples the counters, the GPU and this process once
//! a second and keeps a few minutes of history.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{Value, json};
use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::stats::Stats;
use crate::tui::{Gpu, GpuSample};

const KEEP: usize = 300;

/// One second. `None`: nothing to measure (idle, or no NVML).
#[derive(Clone, Default)]
struct Sample {
    decode_tps: Option<f64>,
    disk_bps: f64,
    upload_bps: f64,
    /// share of the experts asked for that were in VRAM or RAM
    held: Option<f64>,
    cpu: f64,
    rss: u64,
    gpu: Option<(u32, u64, u64)>,
}

#[derive(Default)]
struct State {
    samples: VecDeque<Sample>,
    gpu: Option<GpuSample>,
    ram: (u64, u64),
}

pub struct Monitor {
    stats: Arc<Stats>,
    started: Instant,
    state: Mutex<State>,
}

/// The counters rates are made from.
#[derive(Clone, Copy)]
struct Counters {
    t: Instant,
    generated: u64,
    disk: u64,
    upload: u64,
    hits: u64,
    loads: u64,
}

impl Counters {
    fn read(s: &Stats) -> Self {
        let g = Stats::get;
        Counters {
            t: Instant::now(),
            generated: g(&s.gen_tokens),
            disk: g(&s.disk_bytes),
            upload: g(&s.upload_bytes),
            hits: g(&s.expert_vram_hits) + g(&s.expert_ram_hits),
            loads: g(&s.expert_disk_loads),
        }
    }
}

impl Monitor {
    pub fn start(stats: Arc<Stats>) -> Arc<Self> {
        let m = Arc::new(Monitor { stats, started: Instant::now(), state: Default::default() });
        let weak = Arc::downgrade(&m);
        let _ = std::thread::Builder::new().name("monitor".into()).spawn(move || {
            let mut sys = System::new().ok();
            let pid = Pid::from_u32(std::process::id());
            let mut gpu = Gpu::new("");
            let mut adapter = String::new();
            let Some(m) = weak.upgrade() else { return };
            let mut prev = Counters::read(&m.stats);
            drop(m);
            loop {
                std::thread::sleep(Duration::from_secs(1));
                let Some(m) = weak.upgrade() else { return };
                // the adapter is known once the model starts loading
                let name = m.stats.adapter.lock().clone();
                if name != adapter {
                    gpu = Gpu::new(&name);
                    adapter = name;
                }
                let now = Counters::read(&m.stats);
                let dt = now.t.duration_since(prev.t).as_secs_f64().max(1e-3);
                let rate = |a: u64, b: u64| a.saturating_sub(b) as f64 / dt;
                let last = Stats::get(&m.stats.last_decode_ns);
                // steps can take longer than a second: show the last step's speed while decoding
                let decoding = now.generated > prev.generated || (last > 0 && *m.stats.phase.lock() == "decoding");
                let (hits, loads) = (now.hits.saturating_sub(prev.hits), now.loads.saturating_sub(prev.loads));
                let mut s = Sample {
                    decode_tps: (decoding && last > 0).then(|| 1e9 / last as f64),
                    disk_bps: rate(now.disk, prev.disk),
                    upload_bps: rate(now.upload, prev.upload),
                    held: (hits + loads > 0).then(|| hits as f64 / (hits + loads) as f64),
                    ..Default::default()
                };
                prev = now;
                let mut ram = (0, 0);
                if let Some(sys) = &mut sys {
                    sys.refresh_cpu_usage();
                    sys.refresh_memory();
                    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
                    s.cpu = sys.global_cpu_usage() as f64;
                    s.rss = sys.process(pid).map_or(0, |p| p.memory());
                    ram = (sys.used_memory(), sys.total_memory());
                }
                // only a GPU recognised as the one in use says anything
                let g = if adapter.is_empty() { None } else { gpu.sample().filter(|g| adapter.contains(&g.name)) };
                s.gpu = g.as_ref().map(|g| (g.util, g.mem_used, g.mem_total));
                let mut st = m.state.lock();
                st.gpu = g;
                st.ram = ram;
                st.samples.push_back(s);
                if st.samples.len() > KEEP {
                    st.samples.pop_front();
                }
            }
        });
        m
    }

    pub fn snapshot(&self) -> Value {
        let s = &self.stats;
        let g = Stats::get;
        let st = self.state.lock();
        let col = |f: &dyn Fn(&Sample) -> Value| st.samples.iter().map(f).collect::<Vec<_>>();
        let events: Vec<Value> = {
            let e = s.events.lock();
            e.iter().rev().take(80).map(|(t, m)| json!({"age": t.elapsed().as_secs_f64(), "text": m})).collect()
        };
        let layers: Vec<[u64; 2]> = s.layer_hits.lock().iter().map(|&(h, m)| [h, m]).collect();
        json!({
            "uptime": self.started.elapsed().as_secs_f64(),
            "phase": *s.phase.lock(),
            "loading": g(&s.load_progress_milli) as f64 / 1000.0,
            "model": *s.model_name.lock(),
            "adapter": *s.adapter.lock(),
            "quant": *s.quant.lock(),
            "profiling": s.profiling.load(std::sync::atomic::Ordering::Relaxed),
            "metrics": s.snapshot_json(),
            "ctx_max": g(&s.ctx_max),
            "active": g(&s.requests_active),
            "queued": g(&s.requests_queued),
            "last_prefill_tps": g(&s.last_prefill_tps_milli) as f64 / 1000.0,
            "cpu_runs": g(&s.expert_cpu_runs),
            "uploads": g(&s.expert_uploads),
            "prefetches": g(&s.expert_prefetches),
            "prefetch_hits": g(&s.expert_prefetch_hits),
            "ram_cache_limit": g(&s.ram_cache_limit),
            "ram_cache_entries": g(&s.ram_cache_entries),
            "system_ram": [st.ram.0, st.ram.1],
            "gpu": st.gpu.as_ref().map(|g| json!({"temp": g.temp, "power": g.power_w})),
            "samples": {
                "decode_tps": col(&|x| json!(x.decode_tps)),
                "disk_bps": col(&|x| json!(x.disk_bps)),
                "upload_bps": col(&|x| json!(x.upload_bps)),
                "held": col(&|x| json!(x.held)),
                "cpu": col(&|x| json!(x.cpu)),
                "rss": col(&|x| json!(x.rss)),
                "gpu_util": col(&|x| json!(x.gpu.map(|g| g.0))),
                "vram_used": col(&|x| json!(x.gpu.map(|g| g.1))),
                "vram_total": st.samples.back().and_then(|x| x.gpu).map(|g| g.2),
            },
            "layers": layers,
            "cpu": crate::cpu::describe(),
            "decode_breakdown": s.decode_breakdown().map(|b| b.map(|(name, ms)| json!([name, ms]))),
            "requests": s.requests_json(),
            "events": events,
        })
    }
}
