//! Optional terminal dashboard: throughput, memory, CPU/GPU utilisation, data transfer rates,
//! expert-cache behaviour and recent events.

use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Bar, BarChart, BarGroup, Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline};
use ratatui::Frame;
use sysinfo::{Pid, ProcessesToUpdate, System};

use crate::stats::Stats;

pub struct Gpu {
    nvml: Option<nvml_wrapper::Nvml>,
    index: u32,
}

#[derive(Default, Clone)]
pub struct GpuSample {
    pub name: String,
    pub util: u32,
    pub mem_used: u64,
    pub mem_total: u64,
    pub temp: u32,
    pub power_w: f64,
    pcie_tx_kbs: u32,
    pcie_rx_kbs: u32,
}

impl Gpu {
    pub fn new(adapter: &str) -> Self {
        let nvml = nvml_wrapper::Nvml::init().ok();
        let mut index = 0;
        if let Some(n) = &nvml {
            for i in 0..n.device_count().unwrap_or(0) {
                if let Ok(d) = n.device_by_index(i) {
                    if d.name().map(|nm| adapter.contains(&nm)).unwrap_or(false) {
                        index = i;
                    }
                }
            }
        }
        Self { nvml, index }
    }
    pub fn sample(&self) -> Option<GpuSample> {
        use nvml_wrapper::enum_wrappers::device::{PcieUtilCounter, TemperatureSensor};
        let d = self.nvml.as_ref()?.device_by_index(self.index).ok()?;
        let mem = d.memory_info().ok();
        Some(GpuSample {
            name: d.name().unwrap_or_default(),
            util: d.utilization_rates().map(|u| u.gpu).unwrap_or(0),
            mem_used: mem.as_ref().map(|m| m.used).unwrap_or(0),
            mem_total: mem.as_ref().map(|m| m.total).unwrap_or(0),
            temp: d.temperature(TemperatureSensor::Gpu).unwrap_or(0),
            power_w: d.power_usage().unwrap_or(0) as f64 / 1000.0,
            pcie_tx_kbs: d.pcie_throughput(PcieUtilCounter::Send).unwrap_or(0),
            pcie_rx_kbs: d.pcie_throughput(PcieUtilCounter::Receive).unwrap_or(0),
        })
    }
}

/// Rates derived from counter deltas between two ticks.
#[derive(Default, Clone, Copy)]
struct Rates {
    decode_tps: f64,
    prefill_tps: f64,
    upload_bps: f64,
    disk_bps: f64,
    engram_bps: f64,
    proc_read_bps: f64,
}

struct Prev {
    t: Instant,
    ntok: u64,
    prompt: u64,
    upload: u64,
    disk: u64,
    engram: u64,
    proc_read: u64,
}

fn gb(b: f64) -> String {
    if b >= 1e9 { format!("{:.2} GB", b / 1e9) } else if b >= 1e6 { format!("{:.1} MB", b / 1e6) } else { format!("{:.0} KB", b / 1e3) }
}

fn rate(b: f64) -> String {
    format!("{}/s", gb(b))
}

pub fn run(stats: Arc<Stats>, listen: String) -> anyhow::Result<()> {
    let mut term = ratatui::init();
    let mut sys = System::new().expect("sysinfo");
    let pid = Pid::from_u32(std::process::id());
    let gpu = Gpu::new(&stats.adapter.lock().clone());
    let start = Instant::now();
    let g = |c: &std::sync::atomic::AtomicU64| Stats::get(c);
    let mut prev = Prev { t: Instant::now(), ntok: 0, prompt: 0, upload: 0, disk: 0, engram: 0, proc_read: 0 };
    let mut rates = Rates::default();
    let mut gs = GpuSample::default();
    let mut cpu_hist: Vec<u64> = Vec::new();
    let mut disk_hist: Vec<u64> = Vec::new();
    let mut last_tick = Instant::now() - Duration::from_secs(1);
    let mut proc_mem = 0u64;
    let mut proc_cpu = 0f32;
    let result = loop {
        if last_tick.elapsed() >= Duration::from_millis(500) {
            last_tick = Instant::now();
            sys.refresh_cpu_usage();
            sys.refresh_memory();
            sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
            let (pr, pm, pc) = sys.process(pid).map(|p| (p.disk_usage().total_read_bytes, p.memory(), p.cpu_usage())).unwrap_or((0, 0, 0.0));
            proc_mem = pm;
            proc_cpu = pc;
            let dt = prev.t.elapsed().as_secs_f64().max(1e-3);
            let now = Prev {
                t: Instant::now(),
                ntok: g(&stats.gen_tokens),
                prompt: g(&stats.prompt_tokens),
                upload: g(&stats.upload_bytes),
                disk: g(&stats.disk_bytes),
                engram: g(&stats.engram_bytes),
                proc_read: pr,
            };
            let d = |a: u64, b: u64| a.saturating_sub(b) as f64 / dt;
            rates = Rates {
                decode_tps: d(now.ntok, prev.ntok),
                prefill_tps: d(now.prompt, prev.prompt),
                upload_bps: d(now.upload, prev.upload),
                disk_bps: d(now.disk, prev.disk),
                engram_bps: d(now.engram, prev.engram),
                proc_read_bps: d(now.proc_read, prev.proc_read),
            };
            prev = now;
            cpu_hist.push(sys.global_cpu_usage() as u64);
            disk_hist.push((rates.disk_bps / 1e6) as u64);
            for h in [&mut cpu_hist, &mut disk_hist] {
                if h.len() > 200 {
                    h.remove(0);
                }
            }
            if let Some(s) = gpu.sample() {
                gs = s;
            }
        }
        let snapshot = (rates, gs.clone(), sys.global_cpu_usage(), sys.used_memory(), sys.total_memory(), proc_mem, proc_cpu);
        term.draw(|f| draw(f, &stats, &snapshot, &cpu_hist, &disk_hist, start, &listen))?;
        if event::poll(Duration::from_millis(200))? {
            if let Event::Key(k) = event::read()? {
                if k.code == KeyCode::Char('q') || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL)) {
                    break Ok(());
                }
            }
        }
    };
    ratatui::restore();
    result
}

type Snap = (Rates, GpuSample, f32, u64, u64, u64, f32);

fn draw(f: &mut Frame, st: &Stats, s: &Snap, cpu_hist: &[u64], disk_hist: &[u64], start: Instant, listen: &str) {
    let (r, gs, cpu, mem_used, mem_total, proc_mem, proc_cpu) = s;
    let g = |c: &std::sync::atomic::AtomicU64| Stats::get(c);
    let title = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Length(9), Constraint::Length(10), Constraint::Min(8)])
        .split(area);

    // header
    let up = start.elapsed().as_secs();
    let header = Paragraph::new(vec![Line::from(vec![
        Span::styled(format!(" {} ", st.model_name.lock()), title),
        Span::styled(format!("{} ", st.quant.lock()), Style::default().fg(Color::Magenta)),
        Span::raw(format!("│ {} │ ", st.adapter.lock())),
        Span::styled(format!("{} ", st.phase.lock()), Style::default().fg(Color::Yellow)),
        if st.profiling.load(std::sync::atomic::Ordering::Relaxed) {
            Span::styled("PROFILING (DSV4_PROFILE) ", Style::default().fg(Color::White).bg(Color::Red).add_modifier(Modifier::BOLD))
        } else {
            Span::raw("")
        },
        Span::raw(format!("│ http://{listen} │ up {:02}:{:02}:{:02} │ q to quit", up / 3600, up / 60 % 60, up % 60)),
    ])])
    .block(Block::default().borders(Borders::ALL).title(" dsv4 "));
    f.render_widget(header, rows[0]);

    // row 1: throughput | memory | gpu
    let r1 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(36), Constraint::Percentage(32), Constraint::Percentage(32)])
        .split(rows[1]);
    let ntok = g(&st.gen_tokens);
    let dec_s = g(&st.decode_ns) as f64 / 1e9;
    let pre_s = g(&st.prefill_ns) as f64 / 1e9;
    let last_ms = g(&st.last_decode_ns) as f64 / 1e6;
    let tp = Paragraph::new(vec![
        Line::from(format!("decode  {:>6.2} tok/s  last {:.0} ms", if last_ms > 0.0 { 1000.0 / last_ms } else { 0.0 }, last_ms)),
        Line::from(format!("  live  {:>6.2} tok/s  avg {:.2}", r.decode_tps, ntok as f64 / dec_s.max(1e-9))),
        Line::from(format!("prefill {:>6.1} tok/s  last {:.1}", r.prefill_tps, g(&st.last_prefill_tps_milli) as f64 / 1000.0)),
        Line::from(format!("tokens    {} prompt / {} generated", g(&st.prompt_tokens), ntok)),
        Line::from(format!("time      prefill {:.1}s  decode {:.1}s", pre_s, dec_s)),
        Line::from(format!("context   {} / {}", g(&st.ctx_len), g(&st.ctx_max))),
        Line::from(format!("requests  {} active, {} queued, {} done", g(&st.requests_active), g(&st.requests_queued), g(&st.requests_done))),
    ])
    .block(Block::default().borders(Borders::ALL).title(Span::styled(" Throughput ", title)));
    f.render_widget(tp, r1[0]);

    let ram_lim = g(&st.ram_cache_limit).max(1);
    let memp = Paragraph::new(vec![
        Line::from(format!("system RAM   {} / {}", gb(*mem_used as f64), gb(*mem_total as f64))),
        Line::from(format!("process RSS  {}", gb(*proc_mem as f64))),
        Line::from(format!("expert RAM {} / {} ({})", gb(g(&st.ram_cache_bytes) as f64), gb(ram_lim as f64), g(&st.ram_cache_entries))),
        Line::from(format!("VRAM (ours)  {} (dense {})", gb(g(&st.gpu_total_bytes) as f64), gb(g(&st.gpu_dense_bytes) as f64))),
        Line::from(format!("VRAM (device) {} / {}", gb(gs.mem_used as f64), gb(gs.mem_total as f64))),
        Line::from(format!("expert slots {} / {}", g(&st.vram_slots_used), g(&st.vram_slots_total))),
        Line::from(format!("CPU  {:>5.1}% total, {:>5.1}% process", cpu, proc_cpu)),
    ])
    .block(Block::default().borders(Borders::ALL).title(Span::styled(" Memory / CPU ", title)));
    f.render_widget(memp, r1[1]);

    let gpup = Paragraph::new(vec![
        Line::from(format!("{}", if gs.name.is_empty() { "NVML unavailable".to_string() } else { gs.name.clone() })),
        Line::from(format!("util {:>3}%  temp {}°C  power {:.0} W", gs.util, gs.temp, gs.power_w)),
        Line::from(format!("PCIe in {}  out {}", rate(gs.pcie_rx_kbs as f64 * 1e3), rate(gs.pcie_tx_kbs as f64 * 1e3))),
        Line::from(format!("uploads  {}", rate(r.upload_bps))),
        Line::from(format!("disk (experts) {}", rate(r.disk_bps))),
        Line::from(format!("disk (process) {}", rate(r.proc_read_bps))),
        Line::from(format!("engram rows    {}", rate(r.engram_bps))),
    ])
    .block(Block::default().borders(Borders::ALL).title(Span::styled(" GPU / Transfers ", title)));
    f.render_widget(gpup, r1[2]);

    // row 2: expert cache | sparklines
    let r2 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(36), Constraint::Percentage(64)])
        .split(rows[2]);
    let vh = g(&st.expert_vram_hits);
    let rh = g(&st.expert_ram_hits);
    let dl = g(&st.expert_disk_loads);
    let tot = (vh + rh + dl).max(1) as f64;
    let ex = Paragraph::new(vec![
        Line::from(format!("VRAM hits  {:>9} ({:>5.1}%)", vh, vh as f64 / tot * 100.0)),
        Line::from(format!("RAM hits   {:>9} ({:>5.1}%)", rh, rh as f64 / tot * 100.0)),
        Line::from(format!("disk loads {:>9} ({:>5.1}%)", dl, dl as f64 / tot * 100.0)),
        Line::from(format!("uploads    {:>9}   CPU runs {}", g(&st.expert_uploads), g(&st.expert_cpu_runs))),
        Line::from(format!("prefetches {:>9}", g(&st.expert_prefetches))),
        Line::from(format!("disk read  {}", gb(g(&st.disk_bytes) as f64))),
        Line::from(match st.decode_breakdown() {
            Some([cpu, disk, gpu, rest, other]) => format!(
                "ms/token   CPU experts {:.0}  disk wait {:.0}  GPU experts {:.0}  rest of GPU {:.0}  other {:.0}",
                cpu.1, disk.1, gpu.1, rest.1, other.1
            ),
            None => "ms/token   –".into(),
        }),
        Line::from(format!("uploaded   {}", gb(g(&st.upload_bytes) as f64))),
    ])
    .block(Block::default().borders(Borders::ALL).title(Span::styled(" Expert cache ", title)));
    f.render_widget(ex, r2[0]);
    let sp = Layout::default().direction(Direction::Vertical).constraints([Constraint::Ratio(1, 3); 3]).split(r2[1]);
    let tps: Vec<u64> = st.tps_history.lock().iter().copied().collect();
    let w = sp[0].width.saturating_sub(2) as usize;
    let tail = |v: &[u64]| v[v.len().saturating_sub(w)..].to_vec();
    f.render_widget(
        Sparkline::default().block(Block::default().borders(Borders::ALL).title(" decode tok/s x100 ")).data(&tail(&tps)).style(Style::default().fg(Color::Green)),
        sp[0],
    );
    f.render_widget(
        Sparkline::default().block(Block::default().borders(Borders::ALL).title(" CPU % ")).data(&tail(cpu_hist)).max(100).style(Style::default().fg(Color::Magenta)),
        sp[1],
    );
    f.render_widget(
        Sparkline::default().block(Block::default().borders(Borders::ALL).title(" expert disk MB/s ")).data(&tail(disk_hist)).style(Style::default().fg(Color::Yellow)),
        sp[2],
    );

    // row 3: per-layer VRAM hit rate | events
    let r3 = Layout::default().direction(Direction::Horizontal).constraints([Constraint::Percentage(50), Constraint::Percentage(50)]).split(rows[3]);
    let left = Layout::default().direction(Direction::Vertical).constraints([Constraint::Length(8), Constraint::Min(4)]).split(r3[0]);
    let r3 = [left[0], r3[1], left[1]];
    let lh = st.layer_hits.lock().clone();
    let bars: Vec<Bar> = lh
        .iter()
        .enumerate()
        .map(|(i, &(h, m))| {
            let pct = if h + m > 0 { h * 100 / (h + m) } else { 0 };
            Bar::default().value(pct).label(Line::from(format!("{i}"))).text_value(String::new())
        })
        .collect();
    let bw = ((r3[0].width.saturating_sub(2)) / (lh.len().max(1) as u16)).saturating_sub(1).max(1);
    f.render_widget(
        BarChart::default()
            .block(Block::default().borders(Borders::ALL).title(" experts run on the GPU: already in VRAM, % by layer "))
            .data(BarGroup::default().bars(&bars))
            .bar_width(bw)
            .bar_gap(1)
            .max(100)
            .bar_style(Style::default().fg(Color::Blue)),
        r3[0],
    );
    let ev = st.events.lock();
    let h = r3[1].height.saturating_sub(2) as usize;
    let items: Vec<ListItem> = ev
        .iter()
        .rev()
        .take(h)
        .rev()
        .map(|(t, m)| ListItem::new(format!("{:>6.0}s {}", t.duration_since(start).as_secs_f64(), m)))
        .collect();
    f.render_widget(List::new(items).block(Block::default().borders(Borders::ALL).title(" Events ")), r3[1]);

    // tail of the current generation, wrapped to the panel
    let out = st.transcript.lock().clone();
    let width = r3[2].width.saturating_sub(2).max(1) as usize;
    let mut lines: Vec<String> = Vec::new();
    for raw in out.split('\n') {
        let chars: Vec<char> = raw.chars().collect();
        if chars.is_empty() {
            lines.push(String::new());
        }
        for c in chars.chunks(width) {
            lines.push(c.iter().collect());
        }
    }
    let hgt = r3[2].height.saturating_sub(2) as usize;
    let shown: Vec<Line> = lines[lines.len().saturating_sub(hgt)..].iter().map(|l| Line::from(l.clone())).collect();
    f.render_widget(Paragraph::new(shown).block(Block::default().borders(Borders::ALL).title(" Output ")), r3[2]);

    let lp = g(&st.load_progress_milli);
    if lp < 1000 {
        let ga = Rect { x: area.x + 2, y: area.y + area.height.saturating_sub(2), width: area.width.saturating_sub(4), height: 1 };
        f.render_widget(Gauge::default().ratio(lp as f64 / 1000.0).label(format!("loading {:.0}%", lp as f64 / 10.0)), ga);
    }
}
