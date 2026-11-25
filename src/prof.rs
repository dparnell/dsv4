//! Diagnostic timers for CPU experts (DSV4_PROF=1). One row per number of experts in a call.
use parking_lot::Mutex;
use std::sync::OnceLock;
use std::time::Instant;

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DSV4_PROF").is_some())
}
pub fn now() -> u64 {
    static T: OnceLock<Instant> = OnceLock::new();
    T.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// One thread's part in one parallel phase.
#[derive(Clone, Copy, Default)]
pub struct Part {
    pub start: u64,
    pub end: u64,
    pub busy: u64,
    pub tasks: u32,
    pub stolen: u32,
}

#[derive(Clone, Copy, Default)]
struct Row {
    calls: u64,
    experts: u64,
    wall: u64,
    phases: u64,
    phase_wall: u64,
    first: u64,
    start: u64,
    tail: u64,
    threads: u64,
    busy: u64,
    tasks: u64,
    stolen: u64,
}

#[derive(Default)]
struct State {
    rows: [[Row; 9]; 2],
    cur: Row,
    placed: (usize, usize),
    last_end: u64,
    gap: u64,
    n: u64,
}
static S: Mutex<Option<State>> = Mutex::new(None);

pub fn phase(t0: u64, parts: &[Part]) {
    let t1 = now();
    let mut g = S.lock();
    let c = &mut g.get_or_insert_default().cur;
    let used: Vec<&Part> = parts.iter().filter(|p| p.tasks > 0).collect();
    c.phases += 1;
    c.phase_wall += t1 - t0;
    c.first += used.iter().map(|p| p.start - t0).min().unwrap_or(0);
    c.threads += used.len() as u64;
    for p in used {
        c.start += p.start - t0;
        c.tail += t1 - p.end;
        c.busy += p.busy;
        c.tasks += p.tasks as u64;
        c.stolen += p.stolen as u64;
    }
}

/// `decode`: every expert has one token.
pub fn call(t0: u64, n: usize, decode: bool, place: impl FnOnce() -> (usize, usize)) {
    let t1 = now();
    let mut g = S.lock();
    let s = g.get_or_insert_default();
    let c = std::mem::take(&mut s.cur);
    s.gap += t0.saturating_sub(s.last_end);
    s.n += 1;
    let r = &mut s.rows[!decode as usize][n.min(8)];
    r.calls += 1;
    r.experts += n as u64;
    r.wall += t1 - t0;
    r.phases += c.phases;
    r.phase_wall += c.phase_wall;
    r.first += c.first;
    r.start += c.start;
    r.tail += c.tail;
    r.threads += c.threads;
    r.busy += c.busy;
    r.tasks += c.tasks;
    r.stolen += c.stolen;
    let check = s.n % 64 == 1;
    if check {
        let (a, b) = place();
        s.placed.0 += a;
        s.placed.1 += b;
    }
    s.last_end = now();
}

pub fn reset() {
    if on() {
        *S.lock() = None;
    }
}

pub fn dump() {
    if !on() {
        return;
    }
    let mut g = S.lock();
    let Some(s) = g.as_mut() else { return };
    use std::io::Write;
    let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open("dsv4-prof.txt") else { return };
    let _ = writeln!(f, "\nCPU experts, {}", crate::cpu::describe());
    let _ = writeln!(f, "mean gap between calls {:.3} ms; pages on their node {}, elsewhere {}", s.gap as f64 / 1e6 / s.n.max(1) as f64, s.placed.0, s.placed.1);
    for (k, what) in ["decode (one token per expert)", "several tokens"].iter().enumerate() {
        let _ = writeln!(f, "{what}\n  n  calls  ms/call ms/expert | serial ms/call | per phase, us: wall first-start mean-start mean-tail | threads used  busy%  stolen%  ms-cpu/expert");
        let mut total = 0;
        for (n, r) in s.rows[k].iter().enumerate().filter(|r| r.1.calls > 0) {
            total += r.wall;
            let (c, p, t) = (r.calls as f64, r.phases.max(1) as f64, r.threads.max(1) as f64);
            let _ = writeln!(f, 
                " {n:>2} {:>6} {:>8.3} {:>9.3} | {:>8.3} | {:>7.0} {:>7.0} {:>7.0} {:>7.0} | {:>5.1} {:>6.1} {:>6.1} {:>8.2}",
                r.calls,
                r.wall as f64 / 1e6 / c,
                r.wall as f64 / 1e6 / r.experts as f64,
                (r.wall - r.phase_wall) as f64 / 1e6 / c,
                r.phase_wall as f64 / 1e3 / p,
                r.first as f64 / 1e3 / p,
                r.start as f64 / 1e3 / t,
                r.tail as f64 / 1e3 / t,
                t / p,
                100.0 * r.busy as f64 / (r.phase_wall as f64 * rayon::current_num_threads() as f64 ),
                100.0 * r.stolen as f64 / r.tasks.max(1) as f64,
                r.busy as f64 / 1e6 / r.experts as f64,
            );
        }
        let _ = writeln!(f, "  total {:.1} ms", total as f64 / 1e6);
    }
}
