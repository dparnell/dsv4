//! Expert packs: all routed experts in one file, each expert contiguous.
//!
//! A GGUF stores the three matrices of an expert far apart, so loading one takes three reads.
//! A pack holds the same bytes in the form the RAM tier keeps them (gate | up | down, see
//! `LayerGeom`), so it takes one; drives are noticeably faster that way, the more so the smaller
//! the matrices are. Nothing is requantized.
//!
//! Layout: a 64 KiB header (magic, JSON length, JSON), then the experts in (layer, expert) order,
//! each padded to 4 KiB.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::expert_io::{LayerParts, Load, Loader, Part, PartKind, THREADS};
use crate::experts::LayerGeom;
use crate::gguf::Gguf;

const MAGIC: &[u8; 8] = b"DSV4EXP1";
const HEADER: u64 = 64 << 10;
const ALIGN: u64 = 4096;
/// Experts read ahead of the writer.
const WINDOW: usize = 48;

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct Header {
    fingerprint: String,
    n_experts: usize,
    /// per layer: ggml types and stored bytes of gate, up, down
    types: Vec<[u32; 3]>,
    bytes: Vec<[usize; 3]>,
}

impl Header {
    fn new(g: &Gguf, geoms: &[LayerGeom], n_experts: usize) -> Result<Self> {
        Ok(Self { fingerprint: fingerprint(g)?, n_experts, types: geoms.iter().map(|l| l.types).collect(), bytes: geoms.iter().map(|l| l.host).collect() })
    }
    fn encode(&self) -> Result<Vec<u8>> {
        let j = serde_json::to_vec(self)?;
        let mut h = MAGIC.to_vec();
        h.extend((j.len() as u64).to_le_bytes());
        h.extend(j);
        if h.len() as u64 > HEADER {
            bail!("expert pack header too large");
        }
        h.resize(HEADER as usize, 0);
        Ok(h)
    }
    fn strides(&self) -> impl Iterator<Item = u64> + '_ {
        self.bytes.iter().map(|b| stride(b))
    }
    fn file_len(&self) -> u64 {
        HEADER + self.strides().sum::<u64>() * self.n_experts as u64
    }
}

fn stride(bytes: &[usize; 3]) -> u64 {
    (bytes.iter().sum::<usize>() as u64).next_multiple_of(ALIGN)
}

/// Identifies the GGUF a pack was made from: its shards' names and sizes, and where its tensors are.
fn fingerprint(g: &Gguf) -> Result<String> {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut add = |b: &[u8]| b.iter().for_each(|&x| h = (h ^ x as u64).wrapping_mul(0x100_0000_01b3));
    for s in &g.shards {
        add(s.path.file_name().context("shard name")?.to_string_lossy().as_bytes());
        add(&(s.mmap.len() as u64).to_le_bytes());
    }
    for n in &g.tensor_order {
        let t = &g.tensors[n];
        add(n.as_bytes());
        add(&t.file_offset.to_le_bytes());
        add(&t.ty.to_le_bytes());
    }
    Ok(format!("{h:016x}"))
}

/// Where a pack of the model at `first_shard` is looked for.
pub fn default_path(first_shard: &Path) -> PathBuf {
    first_shard.with_file_name("experts.pack")
}

/// Check that the pack at `path` belongs to this model; returns where its experts are, as parts
/// of file number `file`. It may hold only the first layers.
pub fn open(path: &Path, g: &Gguf, geoms: &[LayerGeom], n_experts: usize, file: usize) -> Result<Vec<LayerParts>> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut head = vec![0u8; HEADER as usize];
    f.read_exact(&mut head).context("expert pack header")?;
    let have = decode(&head)?;
    let want = Header::new(g, &geoms[..have.bytes.len().min(geoms.len())], n_experts)?;
    if have != want {
        bail!("{} was made from a different model; run `dsv4 repack` again", path.display());
    }
    if f.metadata()?.len() != want.file_len() {
        bail!("{} is incomplete", path.display());
    }
    Ok(parts(&want, file))
}

fn decode(head: &[u8]) -> Result<Header> {
    if &head[..8] != MAGIC {
        bail!("not an expert pack");
    }
    let n = u64::from_le_bytes(head[8..16].try_into()?) as usize;
    Ok(serde_json::from_slice(head.get(16..16usize.saturating_add(n)).context("expert pack header")?)?)
}

fn parts(h: &Header, file: usize) -> Vec<LayerParts> {
    let mut base = HEADER;
    h.bytes
        .iter()
        .map(|b| {
            let (len, st) = (b.iter().sum(), stride(b));
            let offset = base;
            base += st * h.n_experts as u64;
            LayerParts { len, parts: vec![Part { file, offset, stride: st, dst: 0, len, kind: PartKind::Raw }] }
        })
        .collect()
}

fn free_space(path: &Path) -> Option<u64> {
    let p = std::path::absolute(path).ok()?;
    let disks = sysinfo::Disks::new_with_refreshed_list().ok()?;
    disks
        .iter()
        .filter(|d| p.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

/// Write a pack of the model's experts to `out`, resuming an interrupted one, then compare a
/// sample of it with the model. `gguf` says where the experts are in the GGUF shards. Only
/// the first `n_layers` layers are packed.
pub fn repack(g: &Gguf, geoms_all: &[LayerGeom], mut gguf: Vec<LayerParts>, n_layers: usize, n_experts: usize, out: &Path) -> Result<()> {
    let geoms = &geoms_all[..n_layers.min(geoms_all.len())];
    gguf.truncate(geoms.len());
    let h = Header::new(g, geoms, n_experts)?;
    if out.exists() {
        match open(out, g, geoms_all, n_experts, 0) {
            Ok(p) if p.len() != geoms.len() => bail!("{} holds {} layers; delete it or pass another --out", out.display(), p.len()),
            Ok(_) => {
                println!("{} is already complete; checking it", out.display());
                return check(g, &h, gguf, out);
            }
            Err(e) => bail!("{} exists and is not a pack of this model ({e:#}); delete it or pass another --out", out.display()),
        }
    }
    let tmp = {
        let mut n = out.as_os_str().to_owned();
        n.push(".part");
        PathBuf::from(n)
    };
    let mut f = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&tmp).with_context(|| format!("create {}", tmp.display()))?;
    let len = f.metadata()?.len();
    let mut start = 0usize; // experts already written
    if len >= HEADER {
        let mut head = vec![0u8; HEADER as usize];
        f.read_exact(&mut head)?;
        if decode(&head).ok().as_ref() != Some(&h) {
            bail!("{} belongs to another model; delete it first", tmp.display());
        }
        let mut pos = HEADER;
        for st in h.strides().flat_map(|s| std::iter::repeat_n(s, n_experts)) {
            if pos + st > len {
                break;
            }
            pos += st;
            start += 1;
        }
        // the end may not have reached the disk
        start = start.saturating_sub(2 * WINDOW);
    }
    let total = h.file_len();
    let strides: Vec<u64> = h.strides().flat_map(|s| std::iter::repeat_n(s, n_experts)).collect();
    let mut pos = HEADER + strides[..start].iter().sum::<u64>();
    // what is there now is truncated to `pos` below
    if let Some(free) = free_space(out) {
        let need = total - pos;
        let free = free + len.saturating_sub(pos);
        if free < need + (1 << 30) {
            bail!("{:.0} GB needed for {}, {:.0} GB free", need as f64 / 1e9, out.display(), free as f64 / 1e9);
        }
    }
    if start == 0 {
        f.set_len(0)?;
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&h.encode()?)?;
    } else {
        println!("resuming at {:.1} GB", pos as f64 / 1e9);
    }
    f.set_len(pos)?;
    f.seek(SeekFrom::Start(pos))?;

    let src = loader(g, gguf.clone(), None)?;
    let bar = indicatif::ProgressBar::new(total);
    bar.set_style(indicatif::ProgressStyle::with_template("{bar:40} {bytes}/{total_bytes} {bytes_per_sec} ETA {eta}")?);
    bar.set_position(pos);
    bar.reset_eta();
    let mut queue = std::collections::VecDeque::new();
    let mut next = start;
    let pad = vec![0u8; ALIGN as usize];
    let mut unsynced = 0u64;
    loop {
        while queue.len() < WINDOW && next < strides.len() {
            let (l, e) = (next / n_experts, next % n_experts);
            queue.push_back(src.demand(l, e, || None, || Ok(vec![0; gguf[l].len]))?);
            next += 1;
        }
        let Some(load) = queue.pop_front() else { break };
        let data = load.wait()?;
        f.write_all(&data)?;
        f.write_all(&pad[..(data.len() as u64).next_multiple_of(ALIGN) as usize - data.len()])?;
        let n = (data.len() as u64).next_multiple_of(ALIGN);
        pos += n;
        unsynced += n;
        bar.set_position(pos);
        if unsynced > 512 << 20 {
            // keeps dirty pages bounded, and what a resume trusts on the disk
            f.sync_data()?;
            unsynced = 0;
        }
    }
    f.sync_all()?;
    bar.finish();
    if pos != total {
        bail!("wrote {pos} bytes, expected {total}");
    }
    drop(f);
    std::fs::rename(&tmp, out)?;
    check(g, &h, gguf, out)
}

fn loader(g: &Gguf, parts: Vec<LayerParts>, pack: Option<&Path>) -> Result<Loader> {
    let paths: Vec<PathBuf> = g.shards.iter().map(|s| s.path.clone()).chain(pack.map(Path::to_path_buf)).collect();
    Ok(Loader::new(|| paths.iter().map(|p| crate::experts::open_direct(p)).collect(), Arc::new(parts), Vec::new(), THREADS, Box::new(|_| {}))?)
}

/// Compare experts spread over the whole pack with the GGUF.
fn check(g: &Gguf, h: &Header, gguf: Vec<LayerParts>, out: &Path) -> Result<()> {
    let lens: Vec<usize> = gguf.iter().map(|l| l.len).collect();
    let a = loader(g, gguf, None)?;
    let b = loader(g, parts(h, g.shards.len()), Some(out))?;
    let total = lens.len() * h.n_experts;
    // first and last expert, and a stride coprime to the layer size in between
    let mut picks: Vec<usize> = (0..total).step_by((total / 128).max(1) | 1).collect();
    picks.push(total - 1);
    for p in picks.chunks(16) {
        let ask = |l: &Loader| -> Result<Vec<Load>> { p.iter().map(|i| l.demand(i / h.n_experts, i % h.n_experts, || None, || Ok(vec![0; lens[i / h.n_experts]]))).collect() };
        for ((x, y), i) in ask(&a)?.into_iter().zip(ask(&b)?).zip(p) {
            if x.wait()? != y.wait()? {
                bail!("layer {} expert {} differs from the GGUF: delete {} and repack", i / h.n_experts, i % h.n_experts, out.display());
            }
        }
    }
    println!("{}: {:.1} GB, {} of {total} experts compared with the GGUF, all equal", out.display(), h.file_len() as f64 / 1e9, picks.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GGML_MXFP4, Shard, TensorInfo};

    fn fixture(dir: &Path) -> (Gguf, Vec<LayerGeom>, Vec<LayerParts>) {
        let path = dir.join("m.gguf");
        // two layers, 5 experts, MXFP4 gate/up/down of 34 and 51 bytes x 1000 blocks
        let data: Vec<u8> = (0..4_000_000u32).map(|i| (i.wrapping_mul(2654435761) >> 9) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let f = File::open(&path).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&f).unwrap() };
        let mut geoms = Vec::new();
        let mut layers = Vec::new();
        let mut off = 77u64;
        for l in 0..2 {
            let bytes = 17 * 1000 * (l + 2);
            let mut p = Vec::new();
            for m in 0..3 {
                p.push(Part { file: 0, offset: off, stride: bytes as u64, dst: m * bytes, len: bytes, kind: PartKind::Mxfp4 });
                off += 5 * bytes as u64;
            }
            geoms.push(LayerGeom { types: [GGML_MXFP4; 3], host: [bytes; 3] });
            layers.push(LayerParts { len: 3 * bytes, parts: p });
        }
        let t = TensorInfo { name: "t".into(), dims: vec![1], ty: 0, shard: 0, file_offset: 77, nbytes: 4 };
        let g = Gguf { shards: vec![Shard { path, mmap }], kv: Default::default(), tensors: [("t".to_string(), t)].into(), tensor_order: vec!["t".into()] };
        (g, geoms, layers)
    }

    #[test]
    fn pack_round_trip_resume_and_mismatch() {
        let dir = std::env::temp_dir().join(format!("dsv4-pack-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (g, geoms, layers) = fixture(&dir);
        let out = dir.join("experts.pack");
        repack(&g, &geoms, layers.clone(), 2, 5, &out).unwrap();
        let whole = std::fs::read(&out).unwrap();
        assert_eq!(whole.len() as u64, HEADER + 5 * (102_400 + 155_648));

        // interrupted after some of the data: same result
        std::fs::remove_file(&out).unwrap();
        std::fs::write(dir.join("experts.pack.part"), &whole[..300_000]).unwrap();
        repack(&g, &geoms, layers.clone(), 2, 5, &out).unwrap();
        assert!(std::fs::read(&out).unwrap() == whole);

        // a second run only checks
        repack(&g, &geoms, layers.clone(), 2, 5, &out).unwrap();
        // corrupt an expert: the check notices
        let mut bad = whole.clone();
        bad[HEADER as usize + 10] ^= 1;
        std::fs::write(&out, &bad).unwrap();
        assert!(repack(&g, &geoms, layers.clone(), 2, 5, &out).is_err());
        // another model's geometry
        assert!(open(&out, &g, &[geoms[1], geoms[0]], 5, 1).is_err());
        // truncated
        std::fs::write(&out, &whole[..whole.len() - 1]).unwrap();
        assert!(open(&out, &g, &geoms, 5, 1).is_err());
        // only the first layer
        let one = dir.join("one.pack");
        repack(&g, &geoms, layers.clone(), 1, 5, &one).unwrap();
        assert_eq!(open(&one, &g, &geoms, 5, 1).unwrap().len(), 1);
        assert!(repack(&g, &geoms, layers.clone(), 2, 5, &one).is_err());
        drop(g);
        let _ = std::fs::remove_dir_all(dir);
    }
}
