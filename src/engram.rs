//! Engram n-gram embeddings. Hashing happens on the CPU (it only needs token ids); the rows are
//! gathered straight from the memory-mapped MXFP8 table, so only the ~6 KB per token per layer
//! that is actually used ever leaves the disk.

use std::sync::Arc;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::config::Config;
use crate::gguf::{GGML_MXFP8, Gguf, TensorInfo};
use crate::stats::Stats;

pub struct Engram {
    pub layers: Vec<usize>,
    ngram: usize,
    heads: usize,
    pub head_dim: usize,
    mult: Vec<i64>,
    offsets: Vec<i64>,
    primes: Vec<i64>,
    token_map: Vec<i64>,
    pad: i64,
    table: TensorInfo,
    row_bytes: usize,
    e4m3: [f32; 256],
    stats: Arc<Stats>,
}

fn e4m3_lut() -> [f32; 256] {
    let mut t = [0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        let s = if i & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = ((i >> 3) & 0xf) as i32;
        let m = (i & 7) as f32;
        *v = if e == 0 { s * m / 8.0 * 2f32.powi(-6) } else if (i & 0x7f) == 0x7f { 0.0 } else { s * (1.0 + m / 8.0) * 2f32.powi(e - 7) };
    }
    t
}

impl Engram {
    pub fn new(g: &Gguf, cfg: &Config, stats: Arc<Stats>) -> Result<Option<Self>> {
        if cfg.engram_layers.is_empty() {
            return Ok(None);
        }
        if crate::gguf::dev_partial() && !g.has("per_layer_token_embd.weight") {
            tracing::warn!("dev-partial: engram table missing, engram disabled");
            return Ok(None);
        }
        let table = g.tensor("per_layer_token_embd.weight")?.clone();
        let hd = table.dims[0] as usize;
        let row_bytes = match table.ty {
            GGML_MXFP8 => hd / 32 * 33,
            t if crate::quants::can_dequantize(t) && hd as u64 % crate::gguf::type_block(t).unwrap().0 == 0 => {
                let (be, bb) = crate::gguf::type_block(t).unwrap();
                hd / be as usize * bb as usize
            }
            t => bail!("engram table type {} is not supported", crate::gguf::type_name(t)),
        };
        let n_cols = cfg.engram_layers.len() * cfg.n_hash_cols();
        if cfg.engram_offsets.len() != n_cols || cfg.engram_primes.len() != n_cols {
            bail!("engram metadata mismatch: {} offsets for {} columns", cfg.engram_offsets.len(), n_cols);
        }
        if cfg.engram_token_map.is_empty() {
            bail!("GGUF lacks deepseek41.engram.token_map");
        }
        let pad = cfg.engram_token_map[cfg.engram_pad_token as usize];
        Ok(Some(Self {
            layers: cfg.engram_layers.clone(),
            ngram: cfg.engram_ngram,
            heads: cfg.engram_heads,
            head_dim: hd,
            mult: cfg.engram_multipliers.clone(),
            offsets: cfg.engram_offsets.clone(),
            primes: cfg.engram_primes.clone(),
            token_map: cfg.engram_token_map.clone(),
            pad,
            table,
            row_bytes,
            e4m3: e4m3_lut(),
            stats,
        }))
    }

    pub fn cols(&self) -> usize {
        (self.ngram - 1) * self.heads
    }

    /// Row ids for engram layer `li` at position `p` of `tokens`.
    fn rows(&self, li: usize, tokens: &[u32], p: usize, out: &mut Vec<u64>) {
        let mut rolling = 0i64;
        for s in 0..self.ngram {
            let t = if p < s { self.pad } else { self.token_map[tokens[p - s] as usize] };
            let prod = t.wrapping_mul(self.mult[li * self.ngram + s]);
            if s == 0 {
                rolling = prod;
                continue;
            }
            rolling ^= prod;
            for h in 0..self.heads {
                let c = li * self.cols() + (s - 1) * self.heads + h;
                let r = rolling.rem_euclid(self.primes[c]) + self.offsets[c];
                out.push(r as u64);
            }
        }
    }

    fn row_f32(&self, g: &Gguf, row: u64, out: &mut [f32]) {
        let shard = &g.shards[self.table.shard].mmap;
        let off = self.table.file_offset as usize + row as usize * self.row_bytes;
        let b = &shard[off..off + self.row_bytes];
        match self.table.ty {
            GGML_MXFP8 => {
                for (blk, o) in b.chunks_exact(33).zip(out.chunks_exact_mut(32)) {
                    // e8m0: 2^(e-127); e == 0 is the subnormal 2^-127
                    let sc = if blk[0] == 0 { f32::from_bits(1 << 22) } else { f32::from_bits((blk[0] as u32) << 23) };
                    for (q, v) in blk[1..].iter().zip(o.iter_mut()) {
                        *v = self.e4m3[*q as usize] * sc;
                    }
                }
            }
            ty => crate::quants::dequantize(ty, b, out),
        }
    }

    /// Embeddings for positions `pos0..pos0+m` of `tokens`, for engram layer index `li`.
    /// Returns [m][cols * head_dim].
    pub fn lookup(&self, g: &Gguf, li: usize, tokens: &[u32], pos0: usize, m: usize) -> Vec<f32> {
        let width = self.cols() * self.head_dim;
        let mut rows = Vec::with_capacity(m * self.cols());
        for i in 0..m {
            self.rows(li, tokens, pos0 + i, &mut rows);
        }
        // one task per row: these are random reads into a 200 GB mapping, so keep many in flight
        let mut out = vec![0f32; m * width];
        out.par_chunks_mut(self.head_dim).zip(rows.par_iter()).for_each(|(o, &r)| self.row_f32(g, r, o));
        Stats::add(&self.stats.engram_bytes, (m * self.cols() * self.row_bytes) as u64);
        out
    }
}
