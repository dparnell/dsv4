//! Minimal multi-shard GGUF reader. Every shard is memory mapped; tensors are looked up by
//! name across shards and exposed as byte slices plus their absolute file location, so the
//! expert streamer can also issue its own positioned reads.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;

pub const GGML_F32: u32 = 0;
pub const GGML_F16: u32 = 1;
pub const GGML_Q8_0: u32 = 8;
pub const GGML_BF16: u32 = 30;
pub const GGML_MXFP4: u32 = 39;
pub const GGML_MXFP8: u32 = 43;

/// (block elements, block bytes)
pub fn type_block(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        GGML_F32 => (1, 4),
        GGML_F16 | GGML_BF16 => (1, 2),
        GGML_Q8_0 => (32, 34),
        GGML_MXFP4 => (32, 17),
        GGML_MXFP8 => (32, 33),
        2 => (32, 18),   // Q4_0
        3 => (32, 20),   // Q4_1
        6 => (32, 22),   // Q5_0
        7 => (32, 24),   // Q5_1
        10 => (256, 84), // Q2_K
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        20 => (32, 18), // IQ4_NL
        23 => (256, 136),
        _ => return None,
    })
}

pub fn type_name(t: u32) -> &'static str {
    match t {
        GGML_F32 => "F32",
        GGML_F16 => "F16",
        GGML_Q8_0 => "Q8_0",
        GGML_BF16 => "BF16",
        GGML_MXFP4 => "MXFP4",
        GGML_MXFP8 => "MXFP8",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        20 => "IQ4_NL",
        _ => "?",
    }
}

#[derive(Debug, Clone)]
pub enum Value {
    U(u64),
    I(i64),
    F(f64),
    Bool(#[allow(dead_code)] bool),
    Str(String),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(v) => Some(*v),
            Value::I(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::U(v) => Some(*v as i64),
            Value::I(v) => Some(*v),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F(v) => Some(*v),
            Value::U(v) => Some(*v as f64),
            Value::I(v) => Some(*v as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the innermost (contiguous) dimension
    pub dims: Vec<u64>,
    pub ty: u32,
    pub shard: usize,
    /// absolute byte offset of the tensor data within its shard file
    pub file_offset: u64,
    pub nbytes: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

pub struct Shard {
    pub path: PathBuf,
    pub mmap: Mmap,
}

pub struct Gguf {
    pub shards: Vec<Shard>,
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub tensor_order: Vec<String>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.b.len() {
            bail!("unexpected end of GGUF header");
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U(self.u8()? as u64),
            1 => Value::I(self.u8()? as i8 as i64),
            2 => Value::U(self.u16()? as u64),
            3 => Value::I(self.u16()? as i16 as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(self.u32()? as i32 as i64),
            6 => Value::F(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.str()?),
            9 => {
                let at = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(at)?);
                }
                Value::Arr(v)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(self.u64()? as i64),
            12 => Value::F(f64::from_bits(self.u64()?)),
            _ => bail!("unknown GGUF value type {t}"),
        })
    }
}

/// `DSV4_DEV_PARTIAL=1`: load an incomplete download (missing experts read as zeros). Output is
/// meaningless; this only exercises the pipeline end to end.
pub fn dev_partial() -> bool {
    std::env::var("DSV4_DEV_PARTIAL").is_ok_and(|v| v == "1")
}

/// Given any shard path (or the first one), find all `-NNNNN-of-MMMMM.gguf` siblings.
pub fn shard_paths(first: &Path) -> Result<Vec<PathBuf>> {
    let name = first.file_name().and_then(|n| n.to_str()).context("bad model path")?;
    if let Some(pos) = name.rfind("-of-") {
        let count: usize = name[pos + 4..].trim_end_matches(".gguf").parse().unwrap_or(1);
        let stem = &name[..pos - 5]; // strip "NNNNN"
        let dir = first.parent().unwrap_or(Path::new("."));
        let paths: Vec<PathBuf> =
            (1..=count).map(|i| dir.join(format!("{stem}{i:05}-of-{count:05}.gguf"))).collect();
        if dev_partial() {
            // development only: use whatever is on disk, including in-progress .part downloads
            return Ok(paths
                .into_iter()
                .filter_map(|p| {
                    if p.exists() {
                        return Some(p);
                    }
                    let part = PathBuf::from(format!("{}.part", p.display()));
                    part.exists().then_some(part)
                })
                .collect());
        }
        for p in &paths {
            if !p.exists() {
                bail!("missing shard {}", p.display());
            }
        }
        return Ok(paths);
    }
    Ok(vec![first.to_path_buf()])
}

/// Resolve a model argument: a directory containing a quant folder, a quant folder, or a gguf file.
pub fn resolve_model_path(p: &Path) -> Result<PathBuf> {
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    if p.is_dir() {
        let mut found: Vec<PathBuf> = Vec::new();
        for dir in std::iter::once(p.to_path_buf()).chain(std::fs::read_dir(p)?.flatten().map(|e| e.path()).filter(|e| e.is_dir())) {
            if let Ok(rd) = std::fs::read_dir(&dir) {
                for e in rd.flatten() {
                    let n = e.file_name().to_string_lossy().to_string();
                    if n.ends_with("-00001-of-00009.gguf") || (n.ends_with(".gguf") && n.contains("00001-of-")) {
                        found.push(e.path());
                    }
                }
            }
        }
        found.sort();
        if let Some(f) = found.into_iter().next() {
            return Ok(f);
        }
    }
    bail!("could not find a GGUF model at {}", p.display())
}

impl Gguf {
    pub fn open(first: &Path) -> Result<Self> {
        let paths = shard_paths(first)?;
        let mut shards = Vec::new();
        let mut kv = HashMap::new();
        let mut tensors = HashMap::new();
        let mut tensor_order = Vec::new();
        for (si, path) in paths.iter().enumerate() {
            let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
            let mmap = unsafe { Mmap::map(&f)? };
            let mut c = Cursor { b: &mmap, p: 0 };
            if c.u32()? != 0x4655_4747 {
                bail!("{} is not a GGUF file", path.display());
            }
            let version = c.u32()?;
            if version < 2 {
                bail!("unsupported GGUF version {version}");
            }
            let nt = c.u64()? as usize;
            let nkv = c.u64()? as usize;
            let mut local_kv = HashMap::new();
            for _ in 0..nkv {
                let k = c.str()?;
                let t = c.u32()?;
                let v = c.value(t)?;
                local_kv.insert(k, v);
            }
            let alignment = local_kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
            let mut infos = Vec::with_capacity(nt);
            for _ in 0..nt {
                let name = c.str()?;
                let nd = c.u32()? as usize;
                let mut dims = Vec::with_capacity(nd);
                for _ in 0..nd {
                    dims.push(c.u64()?);
                }
                let ty = c.u32()?;
                let off = c.u64()?;
                infos.push((name, dims, ty, off));
            }
            let data_start = (c.p as u64).div_ceil(alignment) * alignment;
            for (name, dims, ty, off) in infos {
                let (be, bb) = type_block(ty).with_context(|| format!("tensor {name}: unsupported ggml type {ty}"))?;
                let n: u64 = dims.iter().product();
                let nbytes = n / be * bb;
                let file_offset = data_start + off;
                if file_offset + nbytes > mmap.len() as u64 {
                    bail!("tensor {name} extends past the end of {}", path.display());
                }
                tensor_order.push(name.clone());
                tensors.insert(name.clone(), TensorInfo { name, dims, ty, shard: si, file_offset, nbytes });
            }
            if si == 0 {
                kv = local_kv;
            } else {
                for (k, v) in local_kv {
                    kv.entry(k).or_insert(v);
                }
            }
            shards.push(Shard { path: path.clone(), mmap });
        }
        Ok(Self { shards, kv, tensors, tensor_order })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors.get(name).with_context(|| format!("tensor {name} not found"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn bytes(&self, t: &TensorInfo) -> &[u8] {
        let m = &self.shards[t.shard].mmap;
        &m[t.file_offset as usize..(t.file_offset + t.nbytes) as usize]
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }

    pub fn u64(&self, key: &str) -> Result<u64> {
        self.get(key).and_then(|v| v.as_u64()).with_context(|| format!("missing GGUF key {key}"))
    }

    pub fn f64(&self, key: &str) -> Result<f64> {
        self.get(key).and_then(|v| v.as_f64()).with_context(|| format!("missing GGUF key {key}"))
    }

    pub fn arr_i64(&self, key: &str) -> Result<Vec<i64>> {
        self.get(key)
            .and_then(|v| v.as_arr())
            .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
            .with_context(|| format!("missing GGUF array {key}"))
    }

    pub fn arr_f64(&self, key: &str) -> Result<Vec<f64>> {
        self.get(key)
            .and_then(|v| v.as_arr())
            .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
            .with_context(|| format!("missing GGUF array {key}"))
    }

    pub fn str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }

    pub fn total_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.mmap.len() as u64).sum()
    }
}
