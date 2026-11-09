//! Dense (non-expert) weights: read from the mmap, repacked into GPU-friendly layouts and
//! uploaded once at startup. These stay resident in VRAM (~8 GB for the Q4 quant).

use anyhow::{Result, bail};
use half::f16;
use rayon::prelude::*;

use crate::config::Config;
use crate::gguf::{GGML_BF16, GGML_F16, GGML_F32, GGML_Q8_0, Gguf, TensorInfo};
use crate::gpu::{Batch, Buf, Gpu};

/// Q8_0 weight [n rows][k] repacked as [int8 n*k][f16 scales n*k/32], possibly split into
/// row ranges so that no part exceeds the storage binding limit.
pub struct Q8W {
    #[allow(dead_code)]
    pub n: usize,
    pub k: usize,
    pub parts: Vec<Q8Part>,
}

pub struct Q8Part {
    pub buf: Buf,
    pub row0: usize,
    pub rows: usize,
    pub scale_off_words: u32,
}

pub fn repack_q8(raw: &[u8], rows: usize, k: usize) -> Vec<u8> {
    let nb = k / 32;
    let nblocks = rows * nb;
    let qbytes = nblocks * 32;
    let sbytes = (nblocks * 2).div_ceil(4) * 4;
    let mut out = vec![0u8; qbytes + sbytes];
    let (q, s) = out.split_at_mut(qbytes);
    q.par_chunks_mut(32 * nb).zip(s.par_chunks_mut(2 * nb)).enumerate().for_each(|(r, (qr, sr))| {
        for b in 0..nb {
            let src = &raw[(r * nb + b) * 34..(r * nb + b + 1) * 34];
            sr[b * 2..b * 2 + 2].copy_from_slice(&src[0..2]);
            qr[b * 32..b * 32 + 32].copy_from_slice(&src[2..34]);
        }
    });
    out
}

impl Q8W {
    pub fn load(g: &Gguf, gpu: &Gpu, name: &str) -> Result<Self> {
        let t = g.tensor(name)?;
        if t.ty != GGML_Q8_0 {
            bail!("{name}: expected Q8_0, got type {}", t.ty);
        }
        let k = t.dims[0] as usize;
        let n = (t.n_elements() / t.dims[0]) as usize;
        let raw = g.bytes(t);
        let row_bytes = (k / 32 * 34) as u64;
        let max_rows = ((gpu.max_binding() - 4096) / (row_bytes)) as usize;
        let mut parts = Vec::new();
        let mut row0 = 0;
        while row0 < n {
            let rows = (n - row0).min(max_rows);
            let src = &raw[row0 * (k / 32 * 34)..(row0 + rows) * (k / 32 * 34)];
            let packed = repack_q8(src, rows, k);
            let buf = gpu.buffer_init(name, &packed);
            parts.push(Q8Part { buf, row0, rows, scale_off_words: (rows * k / 4) as u32 });
            row0 += rows;
        }
        Ok(Self { n, k, parts })
    }

    /// Y[m, y_off + y_stride*m + n] = W . X[m]  for m in 0..M.
    #[allow(clippy::too_many_arguments)]
    pub fn run(&self, b: &mut Batch, x: &Buf, x_off: usize, x_stride: usize, y: &Buf, y_off: usize, y_stride: usize, m: usize, group_rows: usize, accumulate: bool) {
        let tm = if m == 1 { 1 } else { 8 };
        let kern = if tm == 1 { "mm_q8_1" } else { "mm_q8_8" };
        for p in &self.parts {
            let params = [
                p.rows as u32,
                self.k as u32,
                m as u32,
                x_stride as u32,
                y_stride as u32,
                (x_off + if group_rows > 0 { p.row0 / group_rows * self.k } else { 0 }) as u32,
                y_off as u32,
                p.scale_off_words,
                group_rows as u32,
                accumulate as u32,
                p.row0 as u32,
            ];
            b.run(kern, &[p.buf.all(), x.all(), y.all()], &params, ((p.rows as u32).div_ceil(8), (m as u32).div_ceil(tm), 1));
        }
    }
}

/// Read any float tensor as f32 on the host.
pub fn read_f32(g: &Gguf, t: &TensorInfo) -> Result<Vec<f32>> {
    let b = g.bytes(t);
    Ok(match t.ty {
        GGML_F32 => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        GGML_F16 => b.chunks_exact(2).map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect(),
        GGML_BF16 => b.chunks_exact(2).map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect(),
        GGML_Q8_0 => {
            let mut out = Vec::with_capacity(t.n_elements() as usize);
            for blk in b.chunks_exact(34) {
                let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                out.extend(blk[2..].iter().map(|&q| q as i8 as f32 * d));
            }
            out
        }
        _ => bail!("{}: cannot read type {} as f32", t.name, t.ty),
    })
}

pub fn f32_buf(g: &Gguf, gpu: &Gpu, name: &str) -> Result<Buf> {
    let v = read_f32(g, g.tensor(name)?)?;
    Ok(gpu.buffer_init(name, bytemuck::cast_slice(&v)))
}

pub struct F32W {
    pub buf: Buf,
    pub n: usize,
    pub k: usize,
}

impl F32W {
    pub fn load(g: &Gguf, gpu: &Gpu, name: &str) -> Result<Self> {
        let t = g.tensor(name)?;
        let k = t.dims[0] as usize;
        let n = (t.n_elements() / t.dims[0]) as usize;
        Ok(Self { buf: f32_buf(g, gpu, name)?, n, k })
    }

    pub fn run(&self, b: &mut Batch, x: &Buf, x_off: usize, x_stride: usize, y: &Buf, y_off: usize, y_stride: usize, m: usize) {
        let tm = if m == 1 { 1 } else { 8 };
        let kern = if tm == 1 { "mm_f32_1" } else { "mm_f32_8" };
        let params = [self.n as u32, self.k as u32, m as u32, x_stride as u32, y_stride as u32, x_off as u32, y_off as u32];
        b.run(kern, &[self.buf.all(), x.all(), y.all()], &params, ((self.n as u32).div_ceil(8), (m as u32).div_ceil(tm), 1));
    }
}

pub struct Compressor {
    pub norm: Buf,
    pub kv: Q8W,
    pub gate: Option<Q8W>,
}

pub struct Indexer {
    pub k_norm: Option<Buf>,
    pub attn_k: Option<Q8W>,
    pub proj: Q8W,
    pub q_b: Q8W,
}

pub struct Engram {
    pub gate: Buf,
    pub kv: Q8W,
}

pub struct Layer {
    pub attn_norm: Buf,
    pub q_a: Q8W,
    pub q_a_norm: Buf,
    pub q_b: Q8W,
    pub kv: Q8W,
    pub kv_norm: Buf,
    pub out_a: Q8W,
    pub out_b: Q8W,
    pub sinks: Buf,
    pub compressor: Option<Compressor>,
    pub indexer: Option<Indexer>,
    pub ffn_norm: Buf,
    pub gate_inp: F32W,
    pub exp_bias: Vec<f32>,
    pub sh_gate: Q8W,
    pub sh_up: Q8W,
    pub sh_down: Q8W,
    pub hc_attn_fn: Buf,
    pub hc_attn_sb: Buf,
    pub hc_ffn_fn: Buf,
    pub hc_ffn_sb: Buf,
    pub engram: Option<Engram>,
}

impl Layer {
    pub fn load(g: &Gguf, gpu: &Gpu, cfg: &Config, l: usize) -> Result<Self> {
        let n = |s: &str| format!("blk.{l}.{s}");
        let q8 = |s: &str| Q8W::load(g, gpu, &n(s));
        let f = |s: &str| f32_buf(g, gpu, &n(s));
        let sb = |kind: &str| -> Result<Buf> {
            let mut v = read_f32(g, g.tensor(&n(&format!("hc_{kind}_scale.weight")))?)?;
            v.extend(read_f32(g, g.tensor(&n(&format!("hc_{kind}_base.weight")))?)?);
            Ok(gpu.buffer_init("hc_sb", bytemuck::cast_slice(&v)))
        };
        let compressor = if g.has(&n("attn_compressor_kv.weight")) {
            Some(Compressor {
                norm: f("attn_compressor_norm.weight")?,
                kv: q8("attn_compressor_kv.weight")?,
                gate: if g.has(&n("attn_compressor_gate.weight")) { Some(q8("attn_compressor_gate.weight")?) } else { None },
            })
        } else {
            None
        };
        let indexer = if g.has(&n("indexer.attn_q_b.weight")) {
            Some(Indexer {
                k_norm: if g.has(&n("indexer.k_norm.weight")) { Some(f("indexer.k_norm.weight")?) } else { None },
                attn_k: if g.has(&n("indexer.attn_k.weight")) { Some(q8("indexer.attn_k.weight")?) } else { None },
                proj: q8("indexer.proj.weight")?,
                q_b: q8("indexer.attn_q_b.weight")?,
            })
        } else {
            None
        };
        let engram = if cfg.engram_layers.contains(&l) {
            Some(Engram { gate: f("engram_gate.weight")?, kv: q8("engram_kv.weight")? })
        } else {
            None
        };
        if cfg.is_kv_source(l) && compressor.is_none() {
            bail!("layer {l}: kv source without compressor weights");
        }
        if cfg.is_index_source(l) && indexer.is_none() {
            bail!("layer {l}: index source without indexer weights");
        }
        Ok(Self {
            attn_norm: f("attn_norm.weight")?,
            q_a: q8("attn_q_a.weight")?,
            q_a_norm: f("attn_q_a_norm.weight")?,
            q_b: q8("attn_q_b.weight")?,
            kv: q8("attn_kv.weight")?,
            kv_norm: f("attn_kv_a_norm.weight")?,
            out_a: q8("attn_output_a.weight")?,
            out_b: q8("attn_output_b.weight")?,
            sinks: f("attn_sinks.weight")?,
            compressor,
            indexer,
            ffn_norm: f("ffn_norm.weight")?,
            gate_inp: F32W::load(g, gpu, &n("ffn_gate_inp.weight"))?,
            exp_bias: read_f32(g, g.tensor(&n("exp_probs_b.bias"))?)?,
            sh_gate: q8("ffn_gate_shexp.weight")?,
            sh_up: q8("ffn_up_shexp.weight")?,
            sh_down: q8("ffn_down_shexp.weight")?,
            hc_attn_fn: f("hc_attn_fn.weight")?,
            hc_attn_sb: sb("attn")?,
            hc_ffn_fn: f("hc_ffn_fn.weight")?,
            hc_ffn_sb: sb("ffn")?,
            engram,
        })
    }
}
