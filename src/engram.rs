//! Engram n-gram embeddings. Hashing happens on the CPU (it only needs token ids); the rows are
//! gathered straight from the memory-mapped tables, so only the ~6 KB per token per layer
//! that is actually used ever leaves the disk.

use std::sync::Arc;

use anyhow::{Result, bail};
use rayon::prelude::*;

use crate::config::Config;
use crate::gguf::{ENGRAM_TABLE, GGML_I8, GGML_MXFP8, Gguf, TensorInfo};
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
    /// per layer; the same one for all if the file has a single table
    tables: Vec<TensorInfo>,
    row_bytes: usize,
    e4m3: [f32; 256],
    /// threads that wait for the disk, so that those of the global pool can compute meanwhile
    io: rayon::ThreadPool,
    stats: Arc<Stats>,
}

const IO_THREADS: usize = 32;

/// Rows being read; see [`Engram::start`].
pub struct Rows {
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
    data: Option<Vec<f32>>,
}

impl Rows {
    /// The values, once they are all there.
    pub fn get(&mut self) -> &[f32] {
        self.data.get_or_insert_with(|| self.rx.recv().expect("engram read failed"))
    }
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

/// Values and bytes per row.
fn row_format(t: &TensorInfo) -> Result<(usize, usize)> {
    let d = t.dims[0] as usize;
    Ok(match t.ty {
        GGML_MXFP8 if d % 32 == 0 => (d, d / 32 * 33),
        // raw bytes: ds4's e4m3_e8m0_32_row264
        GGML_I8 if d % 33 == 0 => (d / 33 * 32, d),
        ty if crate::quants::can_dequantize(ty) && d as u64 % crate::gguf::type_block(ty).unwrap().0 == 0 => {
            let (be, bb) = crate::gguf::type_block(ty).unwrap();
            (d, d / be as usize * bb as usize)
        }
        ty => bail!("engram table type {} is not supported", crate::gguf::type_name(ty)),
    })
}

/// First row of every hash column: a column has `prime` rows and they follow one another, in one
/// table for all layers (`shared`) or starting over in each layer's own.
fn row_offsets(primes: &[i64], cols: usize, shared: bool) -> Vec<i64> {
    let mut at = 0;
    primes
        .iter()
        .enumerate()
        .map(|(c, p)| {
            if !shared && c % cols == 0 {
                at = 0;
            }
            at += p;
            at - p
        })
        .collect()
}

impl Engram {
    pub fn new(g: &Gguf, cfg: &Config, stats: Arc<Stats>) -> Result<Option<Self>> {
        if cfg.engram_layers.is_empty() {
            return Ok(None);
        }
        let tables: Vec<TensorInfo> = if g.has(ENGRAM_TABLE) {
            vec![g.tensor(ENGRAM_TABLE)?.clone(); cfg.engram_layers.len()]
        } else {
            let own: Result<Vec<_>> = cfg.engram_layers.iter().map(|l| Ok(g.tensor(&format!("blk.{l}.engram_embd.weight"))?.clone())).collect();
            match own {
                Ok(t) => t,
                Err(_) if crate::gguf::dev_partial() => {
                    tracing::warn!("dev-partial: engram table missing, engram disabled");
                    return Ok(None);
                }
                Err(_) => bail!("the Engram table is missing: if the repo has it as a separate file, put that next to the model (a name with 'engram')"),
            }
        };
        let (hd, row_bytes) = row_format(&tables[0])?;
        if hd != cfg.engram_head_dim || tables.iter().any(|t| t.ty != tables[0].ty || t.dims[0] != tables[0].dims[0]) {
            bail!("unexpected engram table shape {:?}", tables[0].dims);
        }
        let n_cols = cfg.engram_layers.len() * cfg.n_hash_cols();
        if cfg.engram_primes.len() != n_cols || cfg.engram_multipliers.len() != cfg.engram_layers.len() * cfg.engram_ngram {
            bail!("engram metadata mismatch: {} primes for {} columns", cfg.engram_primes.len(), n_cols);
        }
        let offsets = row_offsets(&cfg.engram_primes, cfg.n_hash_cols(), g.has(ENGRAM_TABLE));
        for (c, (o, p)) in offsets.iter().zip(&cfg.engram_primes).enumerate() {
            if (o + p) as u64 > tables[c / cfg.n_hash_cols()].dims[1] {
                bail!("engram table has {} rows, too few for its hash columns", tables[c / cfg.n_hash_cols()].dims[1]);
            }
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
            offsets,
            primes: cfg.engram_primes.clone(),
            token_map: cfg.engram_token_map.clone(),
            pad,
            tables,
            row_bytes,
            e4m3: e4m3_lut(),
            io: rayon::ThreadPoolBuilder::new().num_threads(IO_THREADS).thread_name(|i| format!("engram-io-{i}")).build()?,
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

    fn row_f32(&self, g: &Gguf, li: usize, row: u64, out: &mut [f32]) {
        let t = &self.tables[li];
        let shard = &g.shards[t.shard].mmap;
        let off = t.file_offset as usize + row as usize * self.row_bytes;
        let b = &shard[off..off + self.row_bytes];
        // e8m0: 2^(e-127); e == 0 is the subnormal 2^-127
        let e8m0 = |e: u8| if e == 0 { f32::from_bits(1 << 22) } else { f32::from_bits((e as u32) << 23) };
        let block = |q: &[u8], sc: u8, o: &mut [f32]| {
            let sc = e8m0(sc);
            for (q, v) in q.iter().zip(o) {
                *v = self.e4m3[*q as usize] * sc;
            }
        };
        match t.ty {
            GGML_MXFP8 => b.chunks_exact(33).zip(out.chunks_exact_mut(32)).for_each(|(c, o)| block(&c[1..], c[0], o)),
            // the same numbers, a row's values first and then its scales
            GGML_I8 => {
                let (q, sc) = b.split_at(out.len());
                q.chunks_exact(32).zip(sc).zip(out.chunks_exact_mut(32)).for_each(|((q, sc), o)| block(q, *sc, o));
            }
            ty => crate::quants::dequantize(ty, b, out),
        }
    }

    /// Embeddings for positions `pos0..pos0+m` of `tokens`, for engram layer index `li`.
    /// Start reading embeddings into [m][cols * head_dim].
    pub fn start(self: &Arc<Self>, g: &Arc<Gguf>, li: usize, tokens: &[u32], pos0: usize, m: usize) -> Rows {
        let width = self.cols() * self.head_dim;
        let mut rows = Vec::with_capacity(m * self.cols());
        for i in 0..m {
            self.rows(li, tokens, pos0 + i, &mut rows);
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let (me, g) = (self.clone(), g.clone());
        self.io.spawn(move || {
            // one task per row: these are random reads into a 200 GB mapping, so keep many in flight
            let mut out = vec![0f32; m * width];
            out.par_chunks_mut(me.head_dim).zip(rows.par_iter()).for_each(|(o, &r)| me.row_f32(&g, li, r, o));
            Stats::add(&me.stats.engram_bytes, (m * me.cols() * me.row_bytes) as u64);
            let _ = tx.send(out);
        });
        Rows { rx, data: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_follow_the_tables() {
        assert_eq!(row_offsets(&[3, 5, 7, 11], 2, true), [0, 3, 8, 15]);
        assert_eq!(row_offsets(&[3, 5, 7, 11], 2, false), [0, 3, 0, 7]);
    }
}
