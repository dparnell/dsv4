//! Where worker threads and expert memory live on machines with several processors.
//!
//! Two things are dealt with:
//! - Windows puts at most 64 logical processors in a *group* and, before Windows 11, runs a process
//!   in one group only: a dual-socket machine uses half its cores unless threads are moved.
//! - With several *NUMA nodes* (sockets, or parts of one) memory of another node is slower. Threads
//!   then stay on one node each, and every expert's rows are divided among the nodes: a node's
//!   threads compute the rows in its own memory ([`split`], [`place`]).
//!
//! Memory goes to a node by being written first from there, which Windows and Linux both honour.
//! Anywhere else (one node, macOS) there is nothing to do and rayon's defaults stay.

use std::cell::Cell;
use std::ops::Range;
use std::sync::OnceLock;

use anyhow::Result;

/// Processors that threads can be bound to together: those of one NUMA node, within one processor
/// group on Windows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Group {
    /// Windows processor group
    pub id: u16,
    /// the operating system's number of the NUMA node
    pub node: u32,
    pub cores: usize,
    /// logical processors, numbered within the group
    pub cpus: Vec<u16>,
}

impl Group {
    pub fn logical(&self) -> usize {
        self.cpus.len()
    }
}

fn add_core(groups: &mut Vec<Group>, id: u16, node: u32, cpus: impl IntoIterator<Item = u16>) {
    let i = groups.iter().position(|g| g.id == id && g.node == node).unwrap_or_else(|| {
        groups.push(Group { id, node, ..Default::default() });
        groups.len() - 1
    });
    groups[i].cores += 1;
    groups[i].cpus.extend(cpus);
}

/// Parse the `PROCESSOR_RELATIONSHIP` core records of `GetLogicalProcessorInformationEx`: u32
/// relationship, u32 record size, 22 bytes of no interest, u16 number of groups, then one
/// `GROUP_AFFINITY` (u64 mask, u16 group, 6 bytes padding) each. `node` tells the NUMA node of a
/// processor (group, number).
#[cfg_attr(not(all(windows, target_pointer_width = "64")), allow(dead_code))]
fn parse_windows(mut b: &[u8], node: impl Fn(u16, u16) -> u32) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    while b.len() >= 48 {
        let size = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        if size < 48 || size > b.len() {
            break;
        }
        if b[..4] == [0; 4] {
            let mask = u64::from_le_bytes(b[32..40].try_into().unwrap());
            let id = u16::from_le_bytes([b[40], b[41]]);
            let cpus: Vec<u16> = (0..64).filter(|i| mask >> i & 1 == 1).collect();
            if let Some(&first) = cpus.first() {
                add_core(&mut groups, id, node(id, first), cpus);
            }
        }
        b = &b[size..];
    }
    groups.sort_by_key(|g| (g.id, g.node));
    groups
}

/// "0-3,8,10-11"
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_cpu_list(s: &str) -> Vec<u16> {
    let mut v = Vec::new();
    for p in s.trim().split(',') {
        let (a, b) = p.split_once('-').unwrap_or((p, p));
        if let (Ok(a), Ok(b)) = (a.trim().parse::<u16>(), b.trim().parse::<u16>()) {
            v.extend(a..=b);
        }
    }
    v
}

/// The NUMA nodes under /sys/devices/system, of which only `allowed` processors count.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_linux(sys: &std::path::Path, allowed: impl Fn(u16) -> bool) -> Vec<Group> {
    let mut groups = Vec::new();
    for d in std::fs::read_dir(sys.join("node")).into_iter().flatten().flatten() {
        let Some(node) = d.file_name().to_str().and_then(|n| n.strip_prefix("node")?.parse::<u32>().ok()) else { continue };
        let list = std::fs::read_to_string(d.path().join("cpulist")).unwrap_or_default();
        let cpus: Vec<u16> = parse_cpu_list(&list).into_iter().filter(|&c| allowed(c)).collect();
        let mut seen: Vec<u16> = Vec::new();
        for &c in &cpus {
            if seen.contains(&c) {
                continue;
            }
            let siblings = std::fs::read_to_string(sys.join(format!("cpu/cpu{c}/topology/thread_siblings_list"))).unwrap_or_default();
            let mut core: Vec<u16> = parse_cpu_list(&siblings).into_iter().filter(|s| cpus.contains(s)).collect();
            if !core.contains(&c) {
                core = vec![c];
            }
            seen.extend(&core);
            add_core(&mut groups, 0, node, core);
        }
    }
    groups.sort_by_key(|g| g.node);
    groups
}

#[cfg(all(windows, target_pointer_width = "64"))]
mod sys {
    use super::Group;

    #[repr(C)]
    struct GroupAffinity {
        mask: usize,
        group: u16,
        reserved: [u16; 3],
    }

    #[repr(C)]
    struct ProcessorNumber {
        group: u16,
        number: u8,
        reserved: u8,
    }

    type Handle = *mut core::ffi::c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(kind: i32, buf: *mut u8, len: *mut u32) -> i32;
        fn GetNumaProcessorNodeEx(p: *const ProcessorNumber, node: *mut u16) -> i32;
        fn GetCurrentThread() -> Handle;
        fn GetCurrentProcess() -> Handle;
        fn SetThreadGroupAffinity(thread: Handle, new: *const GroupAffinity, old: *mut GroupAffinity) -> i32;
        fn SetThreadIdealProcessorEx(thread: Handle, new: *const ProcessorNumber, old: *mut ProcessorNumber) -> i32;
        fn K32QueryWorkingSetEx(process: Handle, info: *mut [usize; 2], bytes: u32) -> i32;
    }

    fn node(group: u16, number: u16) -> u32 {
        let mut n = 0u16;
        let ok = unsafe { GetNumaProcessorNodeEx(&ProcessorNumber { group, number: number as u8, reserved: 0 }, &mut n) };
        if ok != 0 && n != u16::MAX { n as u32 } else { 0 }
    }

    pub fn groups() -> Vec<Group> {
        const CORES: i32 = 0;
        let mut len = 0u32;
        // u64s: the records contain pointer-sized fields
        let mut buf: Vec<u64> = Vec::new();
        unsafe {
            GetLogicalProcessorInformationEx(CORES, std::ptr::null_mut(), &mut len);
            buf.resize((len as usize).div_ceil(8), 0);
            if len == 0 || GetLogicalProcessorInformationEx(CORES, buf.as_mut_ptr().cast(), &mut len) == 0 {
                return Vec::new();
            }
        }
        super::parse_windows(&bytemuck::cast_slice(&buf)[..len as usize], node)
    }

    /// Let the calling thread run on any processor of `g`, and on no other.
    pub fn enter(g: &Group) -> bool {
        let mask = g.cpus.iter().fold(0usize, |m, c| m | 1 << c);
        let a = GroupAffinity { mask, group: g.id, reserved: [0; 3] };
        unsafe { SetThreadGroupAffinity(GetCurrentThread(), &a, std::ptr::null_mut()) != 0 }
    }

    /// New memory comes from the node of the thread's *ideal* processor, not of the one it is on.
    pub fn allocate_from(g: &Group) {
        let p = ProcessorNumber { group: g.id, number: g.cpus[0] as u8, reserved: 0 };
        unsafe { SetThreadIdealProcessorEx(GetCurrentThread(), &p, std::ptr::null_mut()) };
    }

    /// The NUMA node of each of the pages at these addresses, where it has one.
    pub fn page_nodes(pages: &[*const u8]) -> Vec<Option<u32>> {
        let mut info: Vec<[usize; 2]> = pages.iter().map(|&p| [p as usize, 0]).collect();
        let ok = unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), info.as_mut_ptr(), (info.len() * 16) as u32) };
        // bit 0: valid; bits 16-21: node
        info.iter().map(|i| (ok != 0 && i[1] & 1 == 1).then_some((i[1] >> 16 & 63) as u32)).collect()
    }
}

#[cfg(target_os = "linux")]
mod sys {
    use super::Group;

    type Mask = [u64; 1024 / 64];

    pub fn groups() -> Vec<Group> {
        let mut m: Mask = [0; 16];
        if unsafe { libc::sched_getaffinity(0, size_of::<Mask>(), m.as_mut_ptr().cast()) } != 0 {
            return Vec::new();
        }
        super::parse_linux(std::path::Path::new("/sys/devices/system"), |c| m.get(c as usize / 64).is_some_and(|w| w >> (c % 64) & 1 == 1))
    }

    /// Let the calling thread run on any processor of `g`, and on no other.
    pub fn enter(g: &Group) -> bool {
        let mut m: Mask = [0; 16];
        for &c in &g.cpus {
            m[c as usize / 64] |= 1 << (c % 64);
        }
        unsafe { libc::sched_setaffinity(0, size_of::<Mask>(), m.as_ptr().cast()) == 0 }
    }

    /// New memory comes from the node the thread runs on.
    pub fn allocate_from(_: &Group) {}

    /// The NUMA node of each of the pages at these addresses, where it has one.
    pub fn page_nodes(pages: &[*const u8]) -> Vec<Option<u32>> {
        let mut status = vec![-1i32; pages.len()];
        // without target nodes this only asks; a failure leaves the -1s
        unsafe {
            libc::syscall(libc::SYS_move_pages, 0usize, pages.len(), pages.as_ptr(), std::ptr::null::<i32>(), status.as_mut_ptr(), 0usize);
        }
        status.into_iter().map(|s| u32::try_from(s).ok()).collect()
    }
}

#[cfg(not(any(all(windows, target_pointer_width = "64"), target_os = "linux")))]
mod sys {
    use super::Group;

    pub fn groups() -> Vec<Group> {
        Vec::new()
    }
    pub fn enter(_: &Group) -> bool {
        false
    }
    pub fn allocate_from(_: &Group) {}
    pub fn page_nodes(pages: &[*const u8]) -> Vec<Option<u32>> {
        vec![None; pages.len()]
    }
}

struct Topology {
    /// empty unless there are several
    groups: Vec<Group>,
    /// index into `shares` of each group
    node_of: Vec<usize>,
    /// cores of each NUMA node
    shares: Vec<usize>,
    /// made up by DSV4_NUMA=<n>: threads are not bound
    pretend: bool,
}

impl Topology {
    fn new(mut groups: Vec<Group>, pretend: bool) -> Self {
        if groups.len() < 2 {
            groups.clear();
        }
        let mut nodes: Vec<u32> = groups.iter().map(|g| g.node).collect();
        nodes.sort_unstable();
        nodes.dedup();
        let node_of: Vec<usize> = groups.iter().map(|g| nodes.binary_search(&g.node).unwrap()).collect();
        let mut shares = vec![0; nodes.len()];
        for (g, &n) in groups.iter().zip(&node_of) {
            shares[n] += g.cores;
        }
        if shares.len() < 2 {
            shares = vec![1];
        }
        Topology { groups, node_of, shares, pretend }
    }
}

/// One node per processor group, all the same one.
fn merge_nodes(groups: Vec<Group>) -> Vec<Group> {
    let mut out: Vec<Group> = Vec::new();
    for g in groups {
        match out.iter_mut().find(|o| o.id == g.id) {
            Some(o) => {
                o.cores += g.cores;
                o.cpus.extend(g.cpus);
            }
            None => out.push(Group { node: 0, ..g }),
        }
    }
    out
}

fn topology() -> &'static Topology {
    static T: OnceLock<Topology> = OnceLock::new();
    T.get_or_init(|| match std::env::var("DSV4_NUMA").as_deref() {
        Ok("off" | "0" | "1") => Topology::new(merge_nodes(sys::groups()), false),
        // for testing: that many nodes on a machine that may not have them
        Ok(n) if n.parse::<usize>().is_ok_and(|n| n <= 64) => {
            let n: usize = n.parse().unwrap();
            let logical = std::thread::available_parallelism().map_or(n, |p| p.get());
            let cores = (logical / n).max(1);
            let g = |i| Group { id: 0, node: i as u32, cores, cpus: (i * cores..(i + 1) * cores).map(|c| c as u16).collect() };
            Topology::new((0..n).map(g).collect(), true)
        }
        _ => Topology::new(sys::groups(), false),
    })
}

/// The groups of processors to spread threads over, if there are several; otherwise none.
pub fn groups() -> &'static [Group] {
    &topology().groups
}

/// How much of every expert belongs to each NUMA node: one entry per node.
pub fn shares() -> &'static [usize] {
    &topology().shares
}

/// The rows of a matrix that belong to each node, in order.
pub fn split(shares: &[usize], rows: usize) -> Vec<Range<usize>> {
    let total: usize = shares.iter().sum();
    let (mut sum, mut from) = (0, 0);
    shares
        .iter()
        .map(|s| {
            sum += s;
            let to = rows * sum / total;
            std::mem::replace(&mut from, to)..to
        })
        .collect()
}

thread_local! {
    static NODE: Cell<usize> = const { Cell::new(0) };
}

/// The NUMA node (index into [`shares`]) the calling worker thread is on.
pub fn node() -> usize {
    NODE.get()
}

fn enter(i: usize) -> bool {
    let t = topology();
    NODE.set(t.node_of[i]);
    t.pretend || sys::enter(&t.groups[i])
}

/// The smallest page size in use anywhere.
const PAGE: usize = 4096;

/// Give the pages of `buf` that are not in memory yet to the nodes that use them. `buf` lies at
/// `at` in a larger buffer of which `homes` tells the node of each range; its contents are not kept.
///
/// Binds the calling thread to one node after another; it ends on the one numbered `home`.
pub fn place(buf: &mut [u8], at: usize, homes: &[(usize, Range<usize>)], home: usize) {
    let t = topology();
    let n = t.shares.len();
    if n < 2 || homes.is_empty() {
        return;
    }
    let base = buf.as_ptr() as usize;
    for node in (1..=n).map(|i| (home + i) % n) {
        let group = t.node_of.iter().position(|&g| g == node).unwrap();
        enter(group);
        if !t.pretend {
            sys::allocate_from(&t.groups[group]);
        }
        for (_, r) in homes.iter().filter(|h| h.0 == node) {
            let (from, to) = (r.start.max(at), r.end.min(at + buf.len()));
            let mut i = from - at;
            while at + i < to {
                // SAFETY: to <= at + buf.len(), so i < buf.len()
                unsafe { buf.as_mut_ptr().add(i).write_volatile(0) };
                // the start of the next page
                i = ((base + i) / PAGE + 1) * PAGE - base;
            }
        }
    }
}

/// [`place`], then `write(buf, from, to)` for each part of `buf` (indices into it) from a thread of
/// the node it belongs to, so that what is written is in that node's caches as well as its memory.
/// Every part of `buf` is written exactly once. Returns false, writing nothing, with fewer than two
/// nodes.
pub fn place_write(buf: &mut [u8], at: usize, homes: &[(usize, Range<usize>)], home: usize, mut write: impl FnMut(&mut [u8], usize, usize)) -> bool {
    let t = topology();
    let n = t.shares.len();
    if n < 2 || homes.is_empty() {
        return false;
    }
    place(buf, at, homes, home);
    let (start, end) = (at, at + buf.len());
    let mut done = 0;
    for node in (1..=n).map(|i| (home + i) % n) {
        let group = t.node_of.iter().position(|&g| g == node).unwrap();
        enter(group);
        for (_, r) in homes.iter().filter(|h| h.0 == node) {
            let (from, to) = (r.start.max(start), r.end.min(end));
            if from < to {
                write(buf, from - at, to - at);
                done += to - from;
            }
        }
    }
    // not covered by `homes` (they cover whole experts, so this is for safety only)
    if done < buf.len() {
        let mut covered: Vec<(usize, usize)> = homes.iter().map(|(_, r)| (r.start.max(start), r.end.min(end))).filter(|r| r.0 < r.1).collect();
        covered.sort_unstable();
        let mut pos = start;
        for (a, b) in covered.into_iter().chain([(end, end)]) {
            if a > pos {
                write(buf, pos - at, a - at);
            }
            pos = pos.max(b);
        }
    }
    true
}

/// Of the pages of `buf`, how many are on the node `homes` wants, and how many are elsewhere.
pub fn placed(buf: &[u8], homes: &[(usize, Range<usize>)]) -> (usize, usize) {
    let t = topology();
    let mut res = (0, 0);
    for (node, r) in homes {
        // whole pages only
        let p0 = (buf.as_ptr() as usize + r.start).div_ceil(PAGE);
        let p1 = (buf.as_ptr() as usize + r.end) / PAGE;
        let pages: Vec<*const u8> = (p0..p1).map(|p| (p * PAGE) as *const u8).collect();
        if pages.is_empty() {
            continue;
        }
        for n in sys::page_nodes(&pages).into_iter().flatten() {
            let right = t.groups.iter().zip(&t.node_of).any(|(g, i)| g.node == n && i == node);
            *(if right { &mut res.0 } else { &mut res.1 }) += 1;
        }
    }
    res
}

/// The group (index) of each of `n` threads, spread over the groups in proportion to their cores.
fn assign(groups: &[Group], n: usize) -> Vec<usize> {
    let mut used = vec![0usize; groups.len()];
    (0..n)
        .map(|_| {
            // the group with the fewest threads per core; the first of equals
            let i = (0..groups.len())
                .min_by(|&a, &b| ((used[a] + 1) * groups[b].cores.max(1)).cmp(&((used[b] + 1) * groups[a].cores.max(1))))
                .unwrap();
            used[i] += 1;
            i
        })
        .collect()
}

/// A rayon pool of `n` threads (0: rayon's choice) over the first `use_groups` groups (0: wherever
/// the OS puts them).
pub fn pool(use_groups: usize, n: usize) -> rayon::ThreadPoolBuilder {
    let b = rayon::ThreadPoolBuilder::new().num_threads(n);
    if use_groups == 0 || n == 0 {
        return b;
    }
    let bind = binder(use_groups, n);
    b.start_handler(move |i| bind(i))
}

/// Binds the calling thread as the given one of `n` over the first `use_groups` groups.
pub fn binder(use_groups: usize, n: usize) -> impl Fn(usize) + Clone + Send + Sync + 'static {
    let groups = &groups()[..use_groups];
    let place = if groups.is_empty() { Vec::new() } else { assign(groups, n) };
    move |i| {
        if place.get(i).is_some_and(|&g| !enter(g)) {
            tracing::warn!("worker {i} could not be bound to its processors");
        }
    }
}

/// Like [`binder`], but each thread on a core of its own (both its logical processors), the
/// `n` threads of a group on its cores in turn. Without groups, or with DSV4_PIN_CORES=0, the same
/// as [`binder`].
pub fn core_binder(use_groups: usize, n: usize) -> impl Fn(usize) + Clone + Send + Sync + 'static {
    let t = topology();
    let groups = &groups()[..use_groups];
    let off = std::env::var("DSV4_PIN_CORES").is_ok_and(|v| v == "0");
    let place: Vec<Option<Group>> = if groups.is_empty() || t.pretend || off {
        Vec::new()
    } else {
        let mut seen = vec![0usize; groups.len()];
        assign(groups, n)
            .into_iter()
            .map(|gi| {
                let g = &groups[gi];
                let k = seen[gi];
                seen[gi] += 1;
                // logical processors of a core are listed together
                let per = g.logical() / g.cores.max(1);
                (per > 0 && g.logical() == per * g.cores).then(|| {
                    let c = k % g.cores;
                    Group { cores: 1, cpus: g.cpus[c * per..(c + 1) * per].to_vec(), ..g.clone() }
                })
            })
            .collect()
    };
    let plain = binder(use_groups, n);
    move |i| match place.get(i) {
        Some(Some(core)) => {
            NODE.set(t.node_of[t.groups.iter().position(|g| g.id == core.id && g.node == core.node).unwrap()]);
            if !sys::enter(core) {
                tracing::warn!("worker {i} could not be bound to its core");
            }
        }
        _ => plain(i),
    }
}

/// Logical processors the threads can use.
pub fn logical() -> usize {
    let t = topology();
    if t.groups.is_empty() || t.pretend {
        return std::thread::available_parallelism().map_or(1, |n| n.get());
    }
    t.groups.iter().map(|g| g.logical()).sum()
}

/// Set up the global pool. Without `threads`: one per core when there are several groups, else
/// what rayon does by itself.
pub fn init(threads: Option<usize>) -> Result<()> {
    let groups = groups();
    // rayon reads the variable itself; with several groups it is read here
    let asked = threads
        .or_else(|| std::env::var("RAYON_NUM_THREADS").ok()?.parse().ok())
        .filter(|&n| n > 0);
    let n = match asked {
        Some(n) => n,
        None if groups.is_empty() => return Ok(()),
        // memory bandwidth, not arithmetic, limits experts: hyper-threads only add contention
        None => groups.iter().map(|g| g.cores).sum(),
    };
    pool(groups.len(), n).build_global()?;
    Ok(())
}

/// One line for the log.
pub fn describe() -> String {
    let d = describe_threads();
    if crate::team::get().is_some() { d + ", awake between layers" } else { d }
}

fn describe_threads() -> String {
    let n = rayon::current_num_threads();
    let t = topology();
    if t.groups.is_empty() {
        return format!("CPU: {n} worker threads");
    }
    let sizes: Vec<String> = t.groups.iter().map(|g| format!("{} cores / {} threads", g.cores, g.logical())).collect();
    let what = match t.shares.len() {
        1 => format!("{} processor groups", t.groups.len()),
        k if k == t.groups.len() => format!("{k} NUMA nodes"),
        k => format!("{k} NUMA nodes in {} processor groups", t.groups.len()),
    };
    format!("CPU: {n} worker threads over {what} ({}){}", sizes.join(", "), if t.pretend { ", pretended" } else { "" })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(kind: u32, group: u16, mask: u64) -> Vec<u8> {
        let mut r = vec![0u8; 48];
        r[..4].copy_from_slice(&kind.to_le_bytes());
        r[4..8].copy_from_slice(&48u32.to_le_bytes());
        r[30] = 1;
        r[32..40].copy_from_slice(&mask.to_le_bytes());
        r[40..42].copy_from_slice(&group.to_le_bytes());
        r
    }

    /// Two Xeon E5-2699 v4: 22 cores with two threads each, per group.
    fn dual_xeon() -> Vec<Group> {
        let mut b = Vec::new();
        for g in 0..2 {
            for c in 0..22 {
                b.extend(record(0, g, 3 << (2 * c)));
            }
        }
        // another kind of record, and a truncated one
        b.extend(record(3, 0, 1));
        b.extend([0u8; 20]);
        parse_windows(&b, |g, _| g as u32)
    }

    fn sizes(g: &[Group]) -> Vec<(u32, usize, usize)> {
        g.iter().map(|g| (g.node, g.cores, g.logical())).collect()
    }

    #[test]
    fn counts_cores_by_group() {
        let want = |id: u16| Group { id, node: id as u32, cores: 22, cpus: (0..44).collect() };
        assert_eq!(dual_xeon(), [want(0), want(1)]);
        assert!(parse_windows(&[], |_, _| 0).is_empty());
        // two nodes in one group
        let b: Vec<u8> = (0..8).flat_map(|c| record(0, 0, 3 << (2 * c))).collect();
        let g = parse_windows(&b, |_, c| c as u32 / 8);
        assert_eq!(sizes(&g), [(0, 4, 8), (1, 4, 8)]);
        assert_eq!(merge_nodes(g).len(), 1);
        assert_eq!(Topology::new(dual_xeon(), false).shares, [22, 22]);
        // a machine with nothing to tell apart
        let t = Topology::new(dual_xeon()[..1].to_vec(), false);
        assert!(t.groups.is_empty() && t.shares == [1]);
        // DSV4_NUMA=off keeps the processor groups
        let t = Topology::new(merge_nodes(dual_xeon()), false);
        assert!(t.groups.len() == 2 && t.shares == [1]);
    }

    /// A dual-socket Linux machine as /sys describes it, plus a node with memory only.
    #[test]
    fn reads_linux_nodes() {
        let d = std::env::temp_dir().join(format!("dsv4-sys-{}", std::process::id()));
        let put = |p: String, s: &str| {
            let p = d.join(p);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, s).unwrap();
        };
        put("node/node0/cpulist".into(), "0-3,8-11\n");
        put("node/node1/cpulist".into(), "4-7,12-15\n");
        put("node/node2/cpulist".into(), "\n");
        put("node/online".into(), "0-2\n");
        for c in 0..16 {
            put(format!("cpu/cpu{c}/topology/thread_siblings_list"), &format!("{},{}\n", c % 8, c % 8 + 8));
        }
        let g = parse_linux(&d, |_| true);
        assert_eq!(sizes(&g), [(0, 4, 8), (1, 4, 8)]);
        assert_eq!(g[1].cpus, [4, 12, 5, 13, 6, 14, 7, 15]);
        // restricted to some processors (taskset, containers)
        assert_eq!(sizes(&parse_linux(&d, |c| c < 6)), [(0, 4, 4), (1, 2, 2)]);
        assert_eq!(parse_linux(&d, |c| c < 4).len(), 1);
        std::fs::remove_dir_all(&d).unwrap();
        assert!(parse_linux(&d, |_| true).is_empty());
    }

    #[test]
    fn spreads_threads_by_cores() {
        let count = |p: &[usize], id| p.iter().filter(|&&g| g == id).count();
        let g = dual_xeon();
        for n in [2, 44, 88] {
            let p = assign(&g, n);
            assert_eq!((count(&p, 0), count(&p, 1)), (n / 2, n / 2));
        }
        let uneven = [Group { cores: 32, ..Default::default() }, Group { id: 1, cores: 8, ..Default::default() }];
        let p = assign(&uneven, 40);
        assert_eq!((count(&p, 0), count(&p, 1)), (32, 8));
    }

    #[test]
    fn splits_cover_all_rows() {
        assert_eq!(split(&[1], 7), [0..7]);
        assert_eq!(split(&[22, 22], 2304), [0..1152, 1152..2304]);
        assert_eq!(split(&[1, 1, 1], 5), [0..1, 1..3, 3..5]);
        assert_eq!(split(&[32, 8], 10), [0..8, 8..10]);
    }

    /// The real calls, on whatever this machine has.
    #[cfg(any(all(windows, target_pointer_width = "64"), target_os = "linux"))]
    #[test]
    fn threads_can_enter_every_group() {
        let g = sys::groups();
        if g.is_empty() {
            return; // no /sys in this sandbox
        }
        let logical: usize = g.iter().map(|g| g.logical()).sum();
        assert!(logical >= std::thread::available_parallelism().unwrap().get());
        assert!(g.iter().all(|g| g.cores > 0 && g.logical() >= g.cores));
        // memory written from a node is on that node
        let nodes = std::thread::spawn(move || {
            g.iter()
                .map(|g| {
                    assert!(sys::enter(g));
                    sys::allocate_from(g);
                    let v = vec![1u8; 1 << 22];
                    (g.node, sys::page_nodes(&[v.as_ptr().wrapping_add(1 << 21)])[0])
                })
                .collect::<Vec<_>>()
        })
        .join()
        .unwrap();
        println!("{nodes:?}");
        for (want, got) in nodes {
            assert!(got.is_none_or(|n| n == want), "{got:?}, not {want}");
        }
    }
}
