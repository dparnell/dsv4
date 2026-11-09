//! Expert distillation pilot.
//!
//! Given the real routed inputs of one layer's experts (recorded by `dsv4 capture`), fit
//! smaller replacements for every expert and measure how well they reproduce the original
//! expert's output on held-out inputs (every 5th sample). Methods, each at 2x/4x/8x fewer
//! bytes than the original expert:
//!
//!   prune      keep the hidden units with the largest contribution, drop the rest
//!   prune+fit  same units, then refit the down projection by ridge regression so the kept
//!              units also absorb what the dropped ones did
//!   lowrank    per-expert input and output subspaces (SVD of that expert's inputs/outputs);
//!              gate/up read only r input directions, down writes only r output directions
//!   shared     one input and one output subspace for the whole layer (eigenvectors of the
//!              pooled covariance), so per-expert storage is just the projected matrices
//!
//! Every method can also be written out as a drop-in replacement for the layer (full-size
//! MXFP4 blobs whose content is the compressed expert) to test the end-to-end effect.

use std::io::{BufReader, Read, Write};
use std::path::Path;

use anyhow::{Context, Result, bail};
use faer::prelude::*;
use faer::{Mat, MatRef, Side};
use rayon::prelude::*;

use crate::config::Config;
use crate::gguf::Gguf;

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const MIN_TRAIN: usize = 16;

fn e8m0(e: u8) -> f32 {
    2f32.powi(e as i32 - 127)
}

fn decode(code: u8) -> f32 {
    let v = E2M1[(code & 7) as usize];
    if code & 8 != 0 { -v } else { v }
}

/// GGUF MXFP4 rows (17-byte blocks: e8m0 scale, then 16 bytes holding elements j / j+16 in the
/// low / high nibbles) -> f32 [rows][k].
pub fn dequant_raw(raw: &[u8], rows: usize, k: usize) -> Mat<f32> {
    let nb = k / 32;
    Mat::from_fn(rows, k, |i, j| {
        let b = &raw[(i * nb + j / 32) * 17..];
        let jj = j % 32;
        let byte = b[1 + jj % 16];
        decode(if jj < 16 { byte & 15 } else { byte >> 4 }) * e8m0(b[0])
    })
}

/// Repacked (GPU slot) layout of one matrix -> f32.
fn dequant_repacked(blob: &[u8], rows: usize, k: usize) -> Mat<f32> {
    let nb = k / 32;
    let (qs, sc) = blob.split_at(rows * nb * 16);
    Mat::from_fn(rows, k, |i, j| {
        let blk = i * nb + j / 32;
        let jj = j % 32;
        let byte = qs[blk * 16 + jj % 16];
        decode(if jj < 16 { byte & 15 } else { byte >> 4 }) * e8m0(sc[blk])
    })
}

fn nearest(x: f32) -> u8 {
    let a = x.abs();
    let mut best = 0;
    for i in 1..8 {
        if (E2M1[i] - a).abs() < (E2M1[best] - a).abs() {
            best = i;
        }
    }
    best as u8 | if x < 0.0 && best > 0 { 8 } else { 0 }
}

/// f32 [rows][k] -> MXFP4 in the repacked layout [qs rows*nb*16][e8m0 rows*nb].
pub fn quant_repacked(w: MatRef<f32>) -> Vec<u8> {
    let (rows, k) = (w.nrows(), w.ncols());
    let nb = k / 32;
    let mut qs = vec![0u8; rows * nb * 16];
    let mut sc = vec![0u8; rows * nb];
    qs.par_chunks_mut(nb * 16).zip(sc.par_chunks_mut(nb)).enumerate().for_each(|(i, (q, s))| {
        for b in 0..nb {
            let vals: Vec<f32> = (0..32).map(|j| w[(i, b * 32 + j)]).collect();
            let amax = vals.iter().fold(0f32, |a, v| a.max(v.abs()));
            let e = if amax > 0.0 { ((amax / 6.0).log2().ceil() as i32 + 127).clamp(1, 254) } else { 1 };
            s[b] = e as u8;
            let inv = 1.0 / e8m0(e as u8);
            for (j, v) in vals.iter().enumerate() {
                let c = nearest(v * inv);
                if j < 16 {
                    q[b * 16 + j] |= c;
                } else {
                    q[b * 16 + j - 16] |= c << 4;
                }
            }
        }
    });
    qs.extend_from_slice(&sc);
    qs
}

fn requant(w: MatRef<f32>) -> Mat<f32> {
    dequant_repacked(&quant_repacked(w), w.nrows(), w.ncols())
}

/// Raw GGUF expert -> repacked GPU layout (what the engine stores), for the int8 reference path.
fn repack_raw(raw: &[u8], out: &mut Vec<u8>) {
    let nblk = raw.len() / 17;
    let start = out.len();
    out.resize(start + nblk * 17, 0);
    let (q, s) = out[start..].split_at_mut(nblk * 16);
    for i in 0..nblk {
        let b = &raw[i * 17..i * 17 + 17];
        s[i] = b[0];
        q[i * 16..i * 16 + 16].copy_from_slice(&b[1..17]);
    }
}

pub struct Expert {
    pub gate: Mat<f32>, // inter x dim
    pub up: Mat<f32>,   // inter x dim
    pub down: Mat<f32>, // dim x inter
    raw: [Vec<u8>; 3],
}

pub fn load_expert(g: &Gguf, cfg: &Config, layer: usize, e: usize) -> Result<Expert> {
    let mut raws: Vec<Vec<u8>> = Vec::new();
    for nm in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
        let t = g.tensor(&format!("blk.{layer}.{nm}.weight"))?;
        let stride = (t.nbytes / t.dims[2]) as usize;
        raws.push(g.bytes(t)[e * stride..(e + 1) * stride].to_vec());
    }
    let (d, h) = (cfg.dim, cfg.moe_inter);
    Ok(Expert {
        gate: dequant_raw(&raws[0], h, d),
        up: dequant_raw(&raws[1], h, d),
        down: dequant_raw(&raws[2], d, h),
        raw: [raws.remove(0), raws.remove(0), raws.remove(0)],
    })
}

fn act(g: f32, u: f32, l: f32) -> f32 {
    let g = g.min(l);
    let u = u.clamp(-l, l);
    g / (1.0 + (-g).exp()) * u
}

fn hidden(e: &Expert, x: MatRef<f32>, limit: f32) -> Mat<f32> {
    let g = x * e.gate.transpose();
    let u = x * e.up.transpose();
    Mat::from_fn(x.nrows(), g.ncols(), |i, j| act(g[(i, j)], u[(i, j)], limit))
}

fn rows(m: MatRef<f32>, idx: &[usize]) -> Mat<f32> {
    Mat::from_fn(idx.len(), m.ncols(), |i, j| m[(idx[i], j)])
}

fn cols(m: MatRef<f32>, idx: &[usize]) -> Mat<f32> {
    Mat::from_fn(m.nrows(), idx.len(), |i, j| m[(i, idx[j])])
}

fn leading(m: MatRef<f32>, r: usize) -> Mat<f32> {
    Mat::from_fn(m.nrows(), r.min(m.ncols()), |i, j| m[(i, j)])
}

/// Top-r eigenvectors (as columns) of a symmetric matrix.
fn top_eigvecs(c: MatRef<f32>, r: usize) -> Result<Mat<f32>> {
    let c64 = Mat::<f64>::from_fn(c.nrows(), c.ncols(), |i, j| c[(i, j)] as f64);
    let evd = c64.self_adjoint_eigen(Side::Lower).map_err(|e| anyhow::anyhow!("eigendecomposition failed: {e:?}"))?;
    let u = evd.U();
    let n = u.ncols();
    // eigenvalues ascend: the last r columns are the leading directions
    Ok(Mat::from_fn(u.nrows(), r, |i, j| u[(i, n - 1 - j)] as f32))
}

fn sq(m: MatRef<f32>) -> f64 {
    m.squared_norm_l2() as f64
}

#[derive(Default, Clone, Copy)]
struct Acc {
    err: f64,
    norm: f64,
}

impl Acc {
    fn add(&mut self, yhat: MatRef<f32>, y: MatRef<f32>) {
        self.err += sq((yhat - y).as_ref());
        self.norm += sq(y);
    }
    fn rel(&self) -> f64 {
        (self.err / self.norm.max(1e-30)).sqrt()
    }
}

/// Captured inputs, grouped by expert: flat [n][dim] rows.
pub fn load_capture(path: &Path, n_experts: usize, d: usize) -> Result<Vec<Vec<f32>>> {
    let f = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::with_capacity(64 << 20, f);
    let mut per: Vec<Vec<f32>> = vec![Vec::new(); n_experts];
    let mut head = [0u8; 8];
    let mut x = vec![0u8; d * 4];
    loop {
        match r.read_exact(&mut head) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        r.read_exact(&mut x)?;
        let e = u16::from_le_bytes([head[0], head[1]]) as usize;
        if e >= n_experts {
            bail!("corrupt capture: expert {e}");
        }
        per[e].extend_from_slice(bytemuck::cast_slice(&x));
    }
    Ok(per)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    Prune,
    PruneFit,
    LowRank,
    Shared,
}

impl Method {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "prune" => Method::Prune,
            "prune+fit" | "prunefit" => Method::PruneFit,
            "lowrank" => Method::LowRank,
            "shared" => Method::Shared,
            _ => bail!("unknown method '{s}' (prune, prune+fit, lowrank, shared)"),
        })
    }
    fn name(self) -> &'static str {
        match self {
            Method::Prune => "prune",
            Method::PruneFit => "prune+fit",
            Method::LowRank => "lowrank",
            Method::Shared => "shared",
        }
    }
}

/// Bytes-per-expert ratio -> method parameter (hidden units kept, or subspace rank).
fn param(m: Method, ratio: usize, d: usize, h: usize) -> usize {
    match m {
        Method::Prune | Method::PruneFit => h / ratio,
        // r*(2d) for the two bases + r*(3h) for the projected gate, up, down
        Method::LowRank => 3 * d * h / (ratio * (2 * d + 3 * h)),
        // bases are shared by the layer, so only the projected matrices count
        Method::Shared => d / ratio,
    }
}

struct Split {
    xtr: Mat<f32>,
    xte: Mat<f32>,
}

fn split(flat: &[f32], d: usize) -> Split {
    let n = flat.len() / d;
    let te: Vec<usize> = (0..n).filter(|i| i % 5 == 4).collect();
    let tr: Vec<usize> = (0..n).filter(|i| i % 5 != 4).collect();
    let mk = |idx: &[usize]| Mat::from_fn(idx.len(), d, |i, j| flat[idx[i] * d + j]);
    Split { xtr: mk(&tr), xte: mk(&te) }
}

/// The compressed expert as full-size matrices (pruned units zeroed, subspace projections
/// multiplied out), quantized back to MXFP4 when `quant` is set.
struct Fitted {
    gate: Mat<f32>,
    up: Mat<f32>,
    down: Mat<f32>,
}

struct Ctx<'a> {
    cfg: &'a Config,
    shared_in: Option<Mat<f32>>,
    shared_out: Option<Mat<f32>>,
}

fn fit(c: &Ctx, e: &Expert, s: &Split, m: Method, ratio: usize, ytr: MatRef<f32>, htr: MatRef<f32>) -> Result<Fitted> {
    let (d, h) = (c.cfg.dim, c.cfg.moe_inter);
    let p = param(m, ratio, d, h);
    Ok(match m {
        Method::Prune | Method::PruneFit => {
            // contribution of unit i: rms activation x norm of its output column
            let mut score: Vec<(f32, usize)> = (0..h)
                .map(|i| {
                    let a: f32 = (0..htr.nrows()).map(|t| htr[(t, i)] * htr[(t, i)]).sum::<f32>() / htr.nrows() as f32;
                    let w: f32 = (0..d).map(|r| e.down[(r, i)] * e.down[(r, i)]).sum();
                    ((a * w).sqrt(), i)
                })
                .collect();
            score.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut keep: Vec<usize> = score[..p].iter().map(|x| x.1).collect();
            keep.sort_unstable();
            let mut mask = vec![false; h];
            keep.iter().for_each(|&i| mask[i] = true);
            let gate = Mat::from_fn(h, d, |i, j| if mask[i] { e.gate[(i, j)] } else { 0.0 });
            let up = Mat::from_fn(h, d, |i, j| if mask[i] { e.up[(i, j)] } else { 0.0 });
            let down_k: Mat<f32> = if m == Method::PruneFit {
                // ridge toward the original columns: (Hk'Hk + lam I) D' = Hk'Y + lam Wk'
                let hk = cols(htr, &keep);
                let wk = cols(e.down.as_ref(), &keep);
                let hk64 = Mat::<f64>::from_fn(hk.nrows(), hk.ncols(), |i, j| hk[(i, j)] as f64);
                let mut gram = hk64.transpose() * &hk64;
                let mean_diag = (0..p).map(|i| gram[(i, i)]).sum::<f64>() / p as f64;
                let lam = 0.05 * mean_diag.max(1e-12);
                for i in 0..p {
                    gram[(i, i)] += lam;
                }
                let y64 = Mat::<f64>::from_fn(ytr.nrows(), ytr.ncols(), |i, j| ytr[(i, j)] as f64);
                let w64 = Mat::<f64>::from_fn(wk.nrows(), wk.ncols(), |i, j| wk[(i, j)] as f64);
                let rhs = hk64.transpose() * &y64 + w64.transpose() * faer::Scale(lam);
                let sol = gram.llt(Side::Lower).map_err(|e| anyhow::anyhow!("cholesky failed: {e:?}"))?.solve(&rhs); // p x d
                Mat::from_fn(d, p, |i, j| sol[(j, i)] as f32)
            } else {
                cols(e.down.as_ref(), &keep)
            };
            let mut down = Mat::<f32>::zeros(d, h);
            for (jj, &j) in keep.iter().enumerate() {
                for i in 0..d {
                    down[(i, j)] = down_k[(i, jj)];
                }
            }
            Fitted { gate, up, down }
        }
        Method::LowRank | Method::Shared => {
            let (vin, pout) = if m == Method::LowRank {
                let svx = s.xtr.thin_svd().map_err(|e| anyhow::anyhow!("svd failed: {e:?}"))?;
                let svy = ytr.thin_svd().map_err(|e| anyhow::anyhow!("svd failed: {e:?}"))?;
                (leading(svx.V(), p), leading(svy.V(), p))
            } else {
                (leading(c.shared_in.as_ref().unwrap().as_ref(), p), leading(c.shared_out.as_ref().unwrap().as_ref(), p))
            };
            // W x  ->  (W V)(V' x):  gate' = gate V V'
            let proj_in = &vin * vin.transpose();
            let proj_out = &pout * pout.transpose();
            Fitted { gate: &e.gate * &proj_in, up: &e.up * &proj_in, down: &proj_out * &e.down }
        }
    })
}

fn quantize(f: Fitted) -> Fitted {
    Fitted { gate: requant(f.gate.as_ref()), up: requant(f.up.as_ref()), down: requant(f.down.as_ref()) }
}

fn eval(f: &Fitted, x: MatRef<f32>, limit: f32) -> Mat<f32> {
    let e = Expert { gate: f.gate.clone(), up: f.up.clone(), down: f.down.clone(), raw: Default::default() };
    hidden(&e, x, limit) * e.down.transpose()
}

pub struct PilotArgs<'a> {
    pub layer: usize,
    pub capture: &'a Path,
    pub ratios: Vec<usize>,
    pub methods: Vec<Method>,
    /// write the replacement layer for (method, ratio) here
    pub emit: Option<(Method, usize, &'a Path)>,
    /// cap on experts analysed (most-used first), 0 = all
    pub max_experts: usize,
}

pub fn pilot(g: &Gguf, cfg: &Config, a: &PilotArgs) -> Result<()> {
    let (d, h, ne) = (cfg.dim, cfg.moe_inter, cfg.n_experts);
    let limit = cfg.swiglu_limit;
    let t0 = std::time::Instant::now();
    let per = load_capture(a.capture, ne, d)?;
    let counts: Vec<usize> = per.iter().map(|v| v.len() / d).collect();
    let total: usize = counts.iter().sum();
    let used = counts.iter().filter(|&&c| c > 0).count();
    let fit_ok = counts.iter().filter(|&&c| c * 4 / 5 >= MIN_TRAIN).count();
    eprintln!(
        "layer {}: {total} routed samples, {used}/{ne} experts seen, {fit_ok} with >= {MIN_TRAIN} training samples ({:.0}s)",
        a.layer,
        t0.elapsed().as_secs_f64()
    );
    let mut order: Vec<usize> = (0..ne).filter(|&e| counts[e] * 4 / 5 >= MIN_TRAIN).collect();
    order.sort_by_key(|&e| std::cmp::Reverse(counts[e]));
    if a.max_experts > 0 {
        order.truncate(a.max_experts);
    }

    // shared subspaces need the pooled input / output covariance of the layer first
    let need_shared = a.methods.contains(&Method::Shared) || a.emit.is_some_and(|x| x.0 == Method::Shared);
    let mut ctx = Ctx { cfg, shared_in: None, shared_out: None };
    if need_shared {
        let mut cin = Mat::<f32>::zeros(d, d);
        let mut cout = Mat::<f32>::zeros(d, d);
        for (i, &e) in order.iter().enumerate() {
            let s = split(&per[e], d);
            let ex = load_expert(g, cfg, a.layer, e)?;
            let y = hidden(&ex, s.xtr.as_ref(), limit) * ex.down.transpose();
            cin += s.xtr.transpose() * &s.xtr;
            cout += y.transpose() * &y;
            if i % 64 == 63 {
                eprintln!("  covariance {}/{} ({:.0}s)", i + 1, order.len(), t0.elapsed().as_secs_f64());
            }
        }
        let rmax = a.ratios.iter().map(|&r| param(Method::Shared, r, d, h)).max().unwrap_or(d / 2);
        ctx.shared_in = Some(top_eigvecs(cin.as_ref(), rmax)?);
        ctx.shared_out = Some(top_eigvecs(cout.as_ref(), rmax)?);
        eprintln!("  shared subspaces ready ({:.0}s)", t0.elapsed().as_secs_f64());
    }

    // results[method][ratio] for exact and requantized weights
    let nm = a.methods.len();
    let nr = a.ratios.len();
    let mut exact = vec![vec![Acc::default(); nr]; nm];
    let mut quant = vec![vec![Acc::default(); nr]; nm];
    let mut int8 = Acc::default();
    let mut per_expert_rows: Vec<String> = Vec::new();
    let mut emit_file = match a.emit {
        Some((_, _, p)) => Some(std::io::BufWriter::with_capacity(64 << 20, std::fs::File::create(p)?)),
        None => None,
    };
    let analysed: std::collections::HashSet<usize> = order.iter().copied().collect();

    // the emitted layer must hold every expert in index order; unanalysed ones stay original
    let emit_order: Vec<usize> = if a.emit.is_some() { (0..ne).collect() } else { order.clone() };
    for (i, &e) in emit_order.iter().enumerate() {
        let ex = load_expert(g, cfg, a.layer, e)?;
        if !analysed.contains(&e) {
            if let Some(w) = emit_file.as_mut() {
                let mut blob = Vec::new();
                for r in &ex.raw {
                    repack_raw(r, &mut blob);
                }
                w.write_all(&blob)?;
            }
            continue;
        }
        let s = split(&per[e], d);
        let htr = hidden(&ex, s.xtr.as_ref(), limit);
        let ytr = &htr * ex.down.transpose();
        let yte = hidden(&ex, s.xte.as_ref(), limit) * ex.down.transpose();
        let mut row = format!("{e:>3} n={:>5}", counts[e]);

        // the engine's own int8-activation CPU path, as a reference error level
        {
            let mut blob = Vec::new();
            for r in &ex.raw {
                repack_raw(r, &mut blob);
            }
            let xs: Vec<Vec<f32>> = (0..s.xte.nrows()).map(|t| (0..d).map(|j| s.xte[(t, j)]).collect()).collect();
            let xr: Vec<&[f32]> = xs.iter().map(|v| v.as_slice()).collect();
            let ys = crate::cpu_moe::expert_forward(&blob, h, d, ex.raw[0].len(), &xr, &vec![1.0; xr.len()], limit);
            let yi = Mat::from_fn(ys.len(), d, |t, j| ys[t][j]);
            int8.add(yi.as_ref(), yte.as_ref());
        }

        for (mi, &m) in a.methods.iter().enumerate() {
            for (ri, &ratio) in a.ratios.iter().enumerate() {
                let f = fit(&ctx, &ex, &s, m, ratio, ytr.as_ref(), htr.as_ref())?;
                let mut ea = Acc::default();
                ea.add(eval(&f, s.xte.as_ref(), limit).as_ref(), yte.as_ref());
                exact[mi][ri].err += ea.err;
                exact[mi][ri].norm += ea.norm;
                let fq = quantize(f);
                let mut qa = Acc::default();
                qa.add(eval(&fq, s.xte.as_ref(), limit).as_ref(), yte.as_ref());
                quant[mi][ri].err += qa.err;
                quant[mi][ri].norm += qa.norm;
                row.push_str(&format!("  {}/{ratio}x {:.3}", m.name(), qa.rel()));
            }
        }
        if let (Some((m, ratio, _)), Some(w)) = (a.emit, emit_file.as_mut()) {
            let f = fit(&ctx, &ex, &s, m, ratio, ytr.as_ref(), htr.as_ref())?;
            for mat in [&f.gate, &f.up, &f.down] {
                w.write_all(&quant_repacked(mat.as_ref()))?;
            }
        }
        per_expert_rows.push(row);
        if i % 16 == 15 || i + 1 == emit_order.len() {
            eprintln!("  {}/{} experts ({:.0}s)", i + 1, emit_order.len(), t0.elapsed().as_secs_f64());
        }
    }
    if let Some(mut w) = emit_file {
        w.flush()?;
        let (m, ratio, p) = a.emit.unwrap();
        eprintln!("wrote {} {}x replacement for layer {} -> {}", m.name(), ratio, a.layer, p.display());
    }

    println!("\nlayer {} — relative output error on held-out samples (RMS, usage-weighted)", a.layer);
    println!("{} experts analysed, {} samples; reference: engine int8 path {:.4}", order.len(), total, int8.rel());
    println!("{:<12} {:>6} {:>8} {:>10} {:>12}", "method", "ratio", "param", "exact", "MXFP4");
    for (mi, &m) in a.methods.iter().enumerate() {
        for (ri, &ratio) in a.ratios.iter().enumerate() {
            println!(
                "{:<12} {:>5}x {:>8} {:>10.4} {:>12.4}",
                m.name(),
                ratio,
                param(m, ratio, d, h),
                exact[mi][ri].rel(),
                quant[mi][ri].rel()
            );
        }
    }
    if std::env::var("DSV4_PILOT_VERBOSE").is_ok() {
        for r in per_expert_rows {
            println!("{r}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_roundtrip_is_stable() {
        let w = Mat::<f32>::from_fn(4, 64, |i, j| ((i * 64 + j) as f32 * 0.37).sin() * (1.0 + i as f32));
        let q1 = requant(w.as_ref());
        let q2 = requant(q1.as_ref());
        assert!(sq((&q1 - &q2).as_ref()) < 1e-12, "requantizing quantized values must be exact");
        let rel = (sq((&q1 - &w).as_ref()) / sq(w.as_ref())).sqrt();
        assert!(rel < 0.15, "mxfp4 error {rel}");
    }
}
