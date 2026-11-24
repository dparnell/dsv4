//! Worker threads for machines with several Windows processor groups.
//!
//! Windows puts at most 64 logical processors in a group and, before Windows 11, runs a process in
//! one group only: a dual-socket machine uses half its cores unless threads are moved explicitly.
//! Elsewhere rayon's defaults already cover the whole machine, so nothing is changed.

use anyhow::Result;

/// Cores and logical processors of one processor group.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Group {
    pub id: u16,
    pub cores: usize,
    pub logical: usize,
    /// its active processors
    pub mask: u64,
}

/// Parse the `PROCESSOR_RELATIONSHIP` core records of `GetLogicalProcessorInformationEx`: u32
/// relationship, u32 record size, 22 bytes of no interest, u16 number of groups, then one
/// `GROUP_AFFINITY` (u64 mask, u16 group, 6 bytes padding) each.
#[cfg_attr(not(all(windows, target_pointer_width = "64")), allow(dead_code))]
fn parse_cores(mut b: &[u8]) -> Vec<Group> {
    let mut groups: Vec<Group> = Vec::new();
    while b.len() >= 48 {
        let size = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        if size < 48 || size > b.len() {
            break;
        }
        if b[..4] == [0; 4] {
            let mask = u64::from_le_bytes(b[32..40].try_into().unwrap());
            let id = u16::from_le_bytes([b[40], b[41]]);
            let i = groups.iter().position(|g| g.id == id).unwrap_or_else(|| {
                groups.push(Group { id, ..Default::default() });
                groups.len() - 1
            });
            groups[i].cores += 1;
            groups[i].logical += mask.count_ones() as usize;
            groups[i].mask |= mask;
        }
        b = &b[size..];
    }
    groups.sort_by_key(|g| g.id);
    groups
}

#[cfg(all(windows, target_pointer_width = "64"))]
mod sys {
    #[repr(C)]
    #[derive(Default)]
    pub struct GroupAffinity {
        pub mask: usize,
        pub group: u16,
        pub reserved: [u16; 3],
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(kind: i32, buf: *mut u8, len: *mut u32) -> i32;
        fn GetCurrentThread() -> *mut core::ffi::c_void;
        fn SetThreadGroupAffinity(thread: *mut core::ffi::c_void, new: *const GroupAffinity, old: *mut GroupAffinity) -> i32;
    }

    pub fn groups() -> Vec<super::Group> {
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
        super::parse_cores(&bytemuck::cast_slice(&buf)[..len as usize])
    }

    /// Let the calling thread run on any processor of `g`, and on no other.
    pub fn enter(g: &super::Group) -> bool {
        let a = GroupAffinity { mask: g.mask as usize, group: g.id, reserved: [0; 3] };
        unsafe { SetThreadGroupAffinity(GetCurrentThread(), &a, std::ptr::null_mut()) != 0 }
    }
}

#[cfg(not(all(windows, target_pointer_width = "64")))]
mod sys {
    pub fn groups() -> Vec<super::Group> {
        Vec::new()
    }
    pub fn enter(_: &super::Group) -> bool {
        false
    }
}

/// The processor groups, if there are several; otherwise none.
pub fn groups() -> Vec<Group> {
    let g = sys::groups();
    if g.len() > 1 { g } else { Vec::new() }
}

/// The group of each of `n` threads, spread over the groups in proportion to their cores.
fn assign(groups: &[Group], n: usize) -> Vec<Group> {
    let mut used = vec![0usize; groups.len()];
    (0..n)
        .map(|_| {
            // the group with the fewest threads per core; the first of equals
            let i = (0..groups.len())
                .min_by(|&a, &b| ((used[a] + 1) * groups[b].cores.max(1)).cmp(&((used[b] + 1) * groups[a].cores.max(1))))
                .unwrap();
            used[i] += 1;
            groups[i]
        })
        .collect()
}

/// A rayon pool of `n` threads (0: rayon's choice) over `groups` (none: wherever the OS puts them).
pub fn pool(groups: &[Group], n: usize) -> rayon::ThreadPoolBuilder {
    let b = rayon::ThreadPoolBuilder::new().num_threads(n);
    if groups.is_empty() || n == 0 {
        return b;
    }
    let place = assign(groups, n);
    b.start_handler(move |i| {
        if !sys::enter(&place[i]) {
            tracing::warn!("worker {i} could not be moved to processor group {}", place[i].id);
        }
    })
}

/// Set up the global pool. Without `threads`: one per core over all groups when there are several,
/// else what rayon does by itself.
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
    pool(&groups, n).build_global()?;
    Ok(())
}

/// One line for the log.
pub fn describe() -> String {
    let n = rayon::current_num_threads();
    let groups = groups();
    if groups.is_empty() {
        return format!("CPU: {n} worker threads");
    }
    let sizes: Vec<String> = groups.iter().map(|g| format!("{} cores / {} threads", g.cores, g.logical)).collect();
    format!("CPU: {n} worker threads over {} processor groups ({})", groups.len(), sizes.join(", "))
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
        parse_cores(&b)
    }

    #[test]
    fn counts_cores_by_group() {
        let want = |id| Group { id, cores: 22, logical: 44, mask: (1 << 44) - 1 };
        assert_eq!(dual_xeon(), [want(0), want(1)]);
        assert!(parse_cores(&[]).is_empty());
    }

    #[test]
    fn spreads_threads_by_cores() {
        let count = |p: &[Group], id| p.iter().filter(|g| g.id == id).count();
        let g = dual_xeon();
        for n in [2, 44, 88] {
            let p = assign(&g, n);
            assert_eq!((count(&p, 0), count(&p, 1)), (n / 2, n / 2));
        }
        let uneven = [Group { id: 0, cores: 32, logical: 64, mask: 0 }, Group { id: 1, cores: 8, logical: 16, mask: 0 }];
        let p = assign(&uneven, 40);
        assert_eq!((count(&p, 0), count(&p, 1)), (32, 8));
    }

    /// The real calls, on whatever this machine has.
    #[cfg(all(windows, target_pointer_width = "64"))]
    #[test]
    fn threads_can_enter_every_group() {
        let g = sys::groups();
        let logical: usize = g.iter().map(|g| g.logical).sum();
        assert!(logical >= std::thread::available_parallelism().unwrap().get());
        assert!(g.iter().all(|g| g.cores > 0 && g.logical >= g.cores));
        let ok = std::thread::spawn(move || g.iter().all(sys::enter)).join().unwrap();
        assert!(ok);
    }
}
