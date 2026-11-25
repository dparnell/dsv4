//! Threads that stay awake between the expert calls of a decode step.
//!
//! Rayon's workers sleep when they run out of work. On some machines (two sockets, deep idle
//! states) getting all of them going again takes 0.5 ms, at every layer. A team's threads watch for
//! the next call for [`SPIN`] before they sleep, and a call never waits for a thread that is late.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::Result;
use parking_lot::Mutex;

/// Long enough for the GPU's part of a layer; short enough that an idle server is idle.
const SPIN: Duration = Duration::from_millis(10);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, clap::ValueEnum)]
pub enum Spin {
    /// On if there are at least two logical processors per worker thread
    #[default]
    Auto,
    On,
    Off,
}

impl Spin {
    pub fn wanted(self, threads: usize) -> bool {
        match self {
            Spin::On => threads > 1,
            Spin::Off => false,
            Spin::Auto => threads > 1 && crate::cpu::logical() >= 2 * threads,
        }
    }
}

type Call = *const (dyn Fn() + Sync);

/// `state` is a call's number, a bit telling whether threads may still join it, and how many
/// threads are in it.
const OPEN: u64 = 1 << 31;
const COUNT: u64 = OPEN - 1;
const NUMBER: u64 = 1 << 32;

struct Shared {
    state: AtomicU64,
    /// written only while no thread is in a call and none can join
    call: UnsafeCell<Option<Call>>,
    parked: Vec<AtomicBool>,
    threads: OnceLock<Vec<std::thread::Thread>>,
    stop: AtomicBool,
    panicked: AtomicBool,
    spin: Duration,
}

// SAFETY: `call` is guarded by `state` as described there; `Team::run` keeps what it points to alive
// until every thread has left the call.
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

pub struct Team {
    sh: Arc<Shared>,
    /// one call at a time
    turn: Mutex<()>,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Team {
    /// A team of `n` including the calling thread, spread like [`crate::cpu::pool`].
    pub fn new(use_groups: usize, n: usize) -> Result<Team> {
        Self::with_spin(use_groups, n, SPIN)
    }

    fn with_spin(use_groups: usize, n: usize, spin: Duration) -> Result<Team> {
        let n = n.max(1) - 1;
        let sh = Arc::new(Shared {
            state: AtomicU64::new(0),
            call: UnsafeCell::new(None),
            parked: (0..n).map(|_| AtomicBool::new(false)).collect(),
            threads: OnceLock::new(),
            stop: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
            spin,
        });
        let mut team = Team { sh: sh.clone(), turn: Mutex::new(()), handles: Vec::new() };
        let place = crate::cpu::binder(use_groups, n + 1);
        for i in 0..n {
            let (s, place) = (sh.clone(), place.clone());
            team.handles.push(std::thread::Builder::new().name(format!("expert-cpu-{i}")).spawn(move || {
                // the caller is the first of the n + 1
                place(i + 1);
                s.work(i)
            })?);
        }
        let _ = sh.threads.set(team.handles.iter().map(|h| h.thread().clone()).collect());
        Ok(team)
    }

    /// Run `f` on the calling thread and on every thread of the team that gets there before the
    /// calling thread is through. Returns false, without running it, if another call is under way.
    pub fn run(&self, f: &(dyn Fn() + Sync)) -> bool {
        let Some(_turn) = self.turn.try_lock() else { return false };
        let sh = &*self.sh;
        // SAFETY: the lifetime is only erased until `Leave` is dropped below
        let f_static: Call = unsafe { std::mem::transmute::<*const (dyn Fn() + Sync + '_), Call>(f) };
        let number = sh.state.load(Ordering::Relaxed) & !(NUMBER - 1);
        // left over if the last call's own thread panicked too
        sh.panicked.store(false, Ordering::Relaxed);
        unsafe { *sh.call.get() = Some(f_static) };
        sh.state.store(number.wrapping_add(NUMBER) | OPEN, Ordering::SeqCst);
        {
            let _leave = Leave(sh);
            sh.wake(2);
            f();
        }
        if sh.panicked.swap(false, Ordering::Relaxed) {
            panic!("a CPU expert thread panicked");
        }
        true
    }
}

/// Ends a call, also when it panics: nobody may join any more, and those in it finish.
struct Leave<'a>(&'a Shared);

impl Drop for Leave<'_> {
    fn drop(&mut self) {
        self.0.state.fetch_and(!OPEN, Ordering::AcqRel);
        let mut n = 0u32;
        while self.0.state.load(Ordering::Acquire) & COUNT != 0 {
            n += 1;
            if n % 1024 == 0 { std::thread::yield_now() } else { std::hint::spin_loop() }
        }
    }
}

impl Drop for Team {
    fn drop(&mut self) {
        self.sh.stop.store(true, Ordering::SeqCst);
        for h in self.handles.drain(..) {
            h.thread().unpark();
            let _ = h.join();
        }
    }
}

impl Shared {
    /// Wake up to `n` sleeping threads. Those wake others in turn, so that no thread has to spend
    /// the time it takes to wake all.
    fn wake(&self, n: usize) {
        let Some(threads) = self.threads.get() else { return };
        let mut left = n;
        for (p, t) in self.parked.iter().zip(threads) {
            if left == 0 {
                return;
            }
            if p.load(Ordering::SeqCst) && p.swap(false, Ordering::SeqCst) {
                t.unpark();
                left -= 1;
            }
        }
    }

    fn work(&self, i: usize) {
        let mut last = 0;
        let mut idle = Instant::now();
        let mut n = 0u32;
        loop {
            let s = self.state.load(Ordering::SeqCst);
            if s & OPEN != 0 && s & !(NUMBER - 1) != last {
                if self.state.compare_exchange(s, s + 1, Ordering::AcqRel, Ordering::Relaxed).is_err() {
                    continue;
                }
                last = s & !(NUMBER - 1);
                self.wake(2);
                // SAFETY: counted in `state`, so the caller is waiting for us
                let f = unsafe { &*(*self.call.get()).unwrap() };
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err() {
                    self.panicked.store(true, Ordering::Relaxed);
                }
                self.state.fetch_sub(1, Ordering::AcqRel);
                idle = Instant::now();
                continue;
            }
            if self.stop.load(Ordering::Relaxed) {
                return;
            }
            n = n.wrapping_add(1);
            if n % 64 != 0 || idle.elapsed() < self.spin {
                std::hint::spin_loop();
                continue;
            }
            // announce first, then look once more: a call that starts in between sees one or is seen
            self.parked[i].store(true, Ordering::SeqCst);
            let s = self.state.load(Ordering::SeqCst);
            let work = s & OPEN != 0 && s & !(NUMBER - 1) != last;
            if !work && !self.stop.load(Ordering::SeqCst) {
                std::thread::park();
            }
            self.parked[i].store(false, Ordering::SeqCst);
        }
    }
}

static TEAM: OnceLock<Option<Team>> = OnceLock::new();

/// Set up the team used for experts, if wanted.
pub fn init(spin: Spin) -> Result<()> {
    let n = rayon::current_num_threads();
    let team = if spin.wanted(n) { Some(Team::new(crate::cpu::groups().len(), n)?) } else { None };
    let _ = TEAM.set(team);
    Ok(())
}

pub fn get() -> Option<&'static Team> {
    TEAM.get()?.as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Every item of many calls is done exactly once, with threads spinning, asleep, or in between.
    #[test]
    fn runs_everything_once() {
        for spin_us in [0, 20, 10_000] {
            let team = Team::with_spin(0, 6, Duration::from_micros(spin_us)).unwrap();
            for c in 0..3000usize {
                let items: Vec<AtomicUsize> = (0..1 + c % 40).map(|_| AtomicUsize::new(0)).collect();
                let next = AtomicUsize::new(0);
                let ok = team.run(&|| {
                    while let Some(i) = items.get(next.fetch_add(1, Ordering::Relaxed)) {
                        i.fetch_add(1, Ordering::Relaxed);
                    }
                });
                assert!(ok && items.iter().all(|i| i.load(Ordering::Relaxed) == 1));
                if c % 500 == 0 {
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
        }
    }

    #[test]
    fn helpers_join_and_wake_from_sleep() {
        let team = Team::with_spin(0, 4, Duration::ZERO).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        let inside = AtomicUsize::new(0);
        // each waits until all four are here
        team.run(&|| {
            inside.fetch_add(1, Ordering::SeqCst);
            let t = Instant::now();
            while inside.load(Ordering::SeqCst) < 4 && t.elapsed() < Duration::from_secs(20) {
                std::thread::yield_now();
            }
        });
        assert_eq!(inside.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn survives_panics_and_nested_calls() {
        let team = Team::with_spin(0, 3, SPIN).unwrap();
        for _ in 0..20 {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| team.run(&|| panic!("expected by this test"))));
            assert!(r.is_err());
        }
        let nested = AtomicBool::new(false);
        team.run(&|| {
            if team.run(&|| ()) {
                nested.store(true, Ordering::Relaxed);
            }
        });
        assert!(!nested.load(Ordering::Relaxed));
        let n = AtomicUsize::new(0);
        assert!(team.run(&|| {
            n.fetch_add(1, Ordering::Relaxed);
        }));
        assert!(n.load(Ordering::Relaxed) >= 1);
    }
}
