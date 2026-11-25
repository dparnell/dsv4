//! `dsv4 selftest`: run every GPU kernel on random data and compare with a CPU reference.

use std::sync::Arc;

use anyhow::Result;
use half::f16;

use crate::gpu::Gpu;
use crate::stats::Stats;

struct R(u64);
impl R {
    fn u(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.u() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| self.f() * s).collect()
    }
}

fn check(name: &str, got: &[f32], want: &[f32], tol: f32) -> bool {
    let mut worst = 0f32;
    let mut wi = 0;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let e = (g - w).abs() / (1.0 + w.abs());
        if !(e <= worst) {
            worst = e;
            wi = i;
        }
    }
    let ok = worst <= tol && got.len() == want.len();
    println!(
        "{} {name:<16} max rel err {worst:.2e} (at {wi}: got {} want {})",
        if ok { "PASS" } else { "FAIL" },
        got.get(wi).copied().unwrap_or(f32::NAN),
        want.get(wi).copied().unwrap_or(f32::NAN)
    );
    ok
}

fn bits(v: f32) -> u32 {
    v.to_bits()
}

// CPU versions of the shader quantizers
fn pow2_ceil(x: f32) -> f32 {
    let b = x.to_bits();
    let mut e = ((b >> 23) & 0xff) as i32 - 127;
    if b & 0x7fffff != 0 {
        e += 1;
    }
    2f32.powi(e)
}
fn rne(x: f32) -> f32 {
    x.round_ties_even()
}
fn to_e4m3(v: f32) -> f32 {
    let a = v.abs().min(448.0);
    let q = if a < 0.015625 {
        rne(a * 512.0) / 512.0
    } else {
        let e = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
        let step = 2f32.powi(e - 3);
        (rne(a / step) * step).min(448.0)
    };
    if v < 0.0 { -q } else { q }
}
fn to_e2m1(v: f32) -> f32 {
    let a = v.abs().min(6.0);
    let q = if a < 2.0 { rne(a * 2.0) * 0.5 } else if a < 4.0 { rne(a) } else { rne(a * 0.5) * 2.0 };
    if v < 0.0 { -q } else { q }
}

pub fn run() -> Result<()> {
    let stats = Arc::new(Stats::default());
    let gpu = Gpu::new(stats, None)?;
    println!("adapter: {} ({:?}), max binding {} MB", gpu.info.name, gpu.info.backend, gpu.max_binding() >> 20);
    let mut r = R(0x9e3779b97f4a7c15);
    let mut all = true;
    let fb = |v: &[f32]| gpu.buffer_init("t", bytemuck::cast_slice(v));

    // ---- mm_q8 (M = 1 and M = 11, grouped and plain) ----
    for &(n, k, m, groups) in &[(96usize, 256usize, 1usize, 0usize), (64, 128, 11, 0), (64, 64, 3, 16)] {
        let mut raw = Vec::new();
        let mut wdeq = vec![0f32; n * k];
        for row in 0..n {
            for b in 0..k / 32 {
                let d = f16::from_f32(r.f().abs() * 0.05 + 0.001);
                raw.extend_from_slice(&d.to_le_bytes());
                for i in 0..32 {
                    let q = (r.u() % 255) as i32 - 127;
                    raw.push(q as i8 as u8);
                    wdeq[row * k + b * 32 + i] = q as f32 * d.to_f32();
                }
            }
        }
        let packed = crate::weights::repack_q8(&raw, n, k);
        let wb = gpu.buffer_init("w", &packed);
        let kin = if groups > 0 { k * (n / groups) } else { k };
        let x = r.vec(m * kin, 1.0);
        let xb = fb(&x);
        let yb = gpu.buffer("y", (m * n * 4) as u64);
        let mut b = gpu.begin();
        let tm = if m == 1 { 1 } else { 8 };
        let params = [n as u32, k as u32, m as u32, kin as u32, n as u32, 0, 0, (n * k / 4) as u32, groups as u32, 0, 0];
        b.run(if tm == 1 { "mm_q8_1" } else { "mm_q8_8" }, &[wb.all(), xb.all(), yb.all()], &params, ((n as u32).div_ceil(8), (m as u32).div_ceil(tm), 1));
        let got = gpu.read_f32(b, &yb, 0, m * n);
        let mut want = vec![0f32; m * n];
        for t in 0..m {
            for row in 0..n {
                let xo = if groups > 0 { row / groups * k } else { 0 };
                want[t * n + row] = (0..k).map(|i| wdeq[row * k + i] * x[t * kin + xo + i]).sum();
            }
        }
        all &= check(&format!("mm_q8 m={m} g={groups}"), &got, &want, 1e-4);
    }

    // ---- mm_f32 ----
    {
        let (n, k, m) = (40, 128, 9);
        let w = r.vec(n * k, 1.0);
        let x = r.vec(m * k, 1.0);
        let (wb, xb, yb) = (fb(&w), fb(&x), gpu.buffer("y", (m * n * 4) as u64));
        let mut b = gpu.begin();
        b.run("mm_f32_8", &[wb.all(), xb.all(), yb.all()], &[n as u32, k as u32, m as u32, k as u32, n as u32, 0, 0], ((n as u32).div_ceil(8), (m as u32).div_ceil(8), 1));
        let got = gpu.read_f32(b, &yb, 0, m * n);
        let want: Vec<f32> = (0..m * n).map(|i| (0..k).map(|j| w[(i % n) * k + j] * x[(i / n) * k + j]).sum()).collect();
        all &= check("mm_f32", &got, &want, 1e-4);
    }

    // ---- MXFP4 expert: gate_up + down ----
    {
        let (inter, dim) = (64usize, 128usize);
        let kv = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
        let mut mk = |rows: usize, k: usize| {
            let mut raw = Vec::new();
            let mut deq = vec![0f32; rows * k];
            for row in 0..rows {
                for b in 0..k / 32 {
                    let e = 120 + (r.u() % 8) as u8;
                    raw.push(e);
                    let sc = 2f32.powi(e as i32 - 127);
                    let mut qs = [0u8; 16];
                    for j in 0..16 {
                        let lo = (r.u() % 16) as u8;
                        let hi = (r.u() % 16) as u8;
                        qs[j] = lo | hi << 4;
                        deq[row * k + b * 32 + j] = kv[lo as usize] * sc;
                        deq[row * k + b * 32 + j + 16] = kv[hi as usize] * sc;
                    }
                    raw.extend_from_slice(&qs);
                }
            }
            let mut rep = vec![0u8; raw.len()];
            let nblk = raw.len() / 17;
            for i in 0..nblk {
                rep[nblk * 16 + i] = raw[i * 17];
                rep[i * 16..i * 16 + 16].copy_from_slice(&raw[i * 17 + 1..i * 17 + 17]);
            }
            (rep, deq)
        };
        let (g, gd) = mk(inter, dim);
        let (u, ud) = mk(inter, dim);
        let (dn, dd) = mk(dim, inter);
        let mat = g.len();
        let mut slot = g.clone();
        slot.extend(&u);
        slot.extend(&dn);
        let sb = gpu.buffer_init("slot", &slot);
        let t = 5usize;
        let x = r.vec(t * dim, 1.0);
        let xb = fb(&x);
        // 3 routed rows: tokens 0, 3, 4 ; output rows tok*2+slot
        let entries: Vec<(u32, u32, f32)> = vec![(0, 1, 0.7), (3, 0, 1.2), (4, 1, 0.3)];
        let mut ent = Vec::new();
        for &(tk, s, w) in &entries {
            ent.extend_from_slice(&[tk, tk * 2 + s, w.to_bits()]);
        }
        let rb = gpu.buffer_init("route", bytemuck::cast_slice(&ent));
        let hb = gpu.buffer("h", (entries.len() * inter * 4) as u64);
        let ob = gpu.buffer("o", (t * 2 * dim * 4) as u64);
        let lim = 10.0f32;
        let mut b = gpu.begin();
        b.run("moe_gate_up_8", &[sb.all(), xb.all(), rb.all(), hb.all()], &[inter as u32, dim as u32, 3, 0, bits(lim), (mat / 4) as u32, dim as u32], ((inter as u32).div_ceil(8), 1, 1));
        b.run("moe_down_8", &[sb.all(), hb.all(), rb.all(), ob.all()], &[dim as u32, inter as u32, 3, 0, 0, (2 * mat / 4) as u32], ((dim as u32).div_ceil(8), 1, 1));
        let got = gpu.read_f32(b, &ob, 0, t * 2 * dim);
        let mut want = vec![0f32; t * 2 * dim];
        for &(tk, s, w) in &entries {
            let xs = &x[tk as usize * dim..(tk as usize + 1) * dim];
            let h: Vec<f32> = (0..inter)
                .map(|n| {
                    let gv = (0..dim).map(|i| gd[n * dim + i] * xs[i]).sum::<f32>().min(lim);
                    let uv = (0..dim).map(|i| ud[n * dim + i] * xs[i]).sum::<f32>().clamp(-lim, lim);
                    gv / (1.0 + (-gv).exp()) * uv * w
                })
                .collect();
            let o = (tk * 2 + s) as usize;
            for n in 0..dim {
                want[o * dim + n] = (0..inter).map(|i| dd[n * inter + i] * h[i]).sum();
            }
        }
        all &= check("moe mxfp4", &got, &want, 1e-4);
        let xs: Vec<&[f32]> = entries.iter().map(|&(tk, _, _)| &x[tk as usize * dim..(tk as usize + 1) * dim]).collect();
        let ws: Vec<f32> = entries.iter().map(|e| e.2).collect();
        let ys = crate::cpu_moe::expert_forward(&slot, inter, dim, mat, &xs, &ws, lim);
        let mut cpu = vec![0f32; t * 2 * dim];
        for (i, &(tk, s, _)) in entries.iter().enumerate() {
            let o = (tk * 2 + s) as usize * dim;
            cpu[o..o + dim].copy_from_slice(&ys[i]);
        }
        // int8 activations: judge the error relative to the output range
        let range = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let err = cpu.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / range;
        let ok = err < 0.03;
        println!("{} moe cpu q8       max err {err:.2e} of output range ({:?})", if ok { "PASS" } else { "FAIL" }, crate::cpu_moe::isa());
        all &= ok;
    }

    // ---- low-bit experts ----
    {
        use crate::quants as q;
        let (inter, dim) = (256usize, 512usize);
        for (ty, dn, name) in [
            (q::IQ1_S, q::Q2_K, "iq1_s/q2_k"),
            (q::IQ2_XXS, q::IQ2_XS, "iq2_xxs/xs"),
            (q::IQ2_S, q::IQ3_XXS, "iq2_s/iq3_xxs"),
            (q::Q3_K, q::IQ3_S, "q3_k/iq3_s"),
            (q::Q2_K, q::IQ1_S, "q2_k/iq1_s"),
        ] {
            let types = [ty, ty, dn];
            let mut host = Vec::new();
            let mut lens = [0usize; 3];
            let mut deq: Vec<Vec<f32>> = Vec::new();
            for (i, &t) in types.iter().enumerate() {
                let raw = q::random_blocks(t, inter * dim / 256, || (r.u() >> 7) as u8);
                let mut d = vec![0f32; inter * dim];
                q::dequantize(t, &raw, &mut d);
                lens[i] = raw.len();
                host.extend(raw);
                deq.push(d);
            }
            let geom = crate::experts::LayerGeom { types, host: lens };
            let slot = geom.slot_data(inter, dim, &host);
            let sl = geom.slot_lens(inter, dim);
            let sb = gpu.buffer_init("slot", &slot);
            let toks = 3usize;
            let x = r.vec(toks * dim, 1.0);
            let xs: Vec<&[f32]> = x.chunks(dim).collect();
            let ws = [0.7f32, 1.2, 0.3];
            let lim = 10.0f32;
            let mut want = Vec::new();
            for (x, w) in xs.iter().zip(ws) {
                let h: Vec<f32> = (0..inter)
                    .map(|n| {
                        let gv = (0..dim).map(|i| deq[0][n * dim + i] * x[i]).sum::<f32>().min(lim);
                        let uv = (0..dim).map(|i| deq[1][n * dim + i] * x[i]).sum::<f32>().clamp(-lim, lim);
                        gv / (1.0 + (-gv).exp()) * uv * w
                    })
                    .collect();
                want.extend((0..dim).map(|n| (0..inter).map(|i| deq[2][n * inter + i] * h[i]).sum::<f32>()));
            }
            let lutp = |t: u32| -> Vec<u32> {
                let l: &[u32; 4] = bytemuck::cast_ref(q::codes(t).0);
                vec![q::has_min(t) as u32, l[0], l[1], l[2], l[3]]
            };
            let (xb, hb, ob) = (fb(&x), gpu.buffer("h", (toks * inter * 4) as u64), gpu.buffer("o", (toks * dim * 4) as u64));
            let ent: Vec<u32> = (0..toks as u32).flat_map(|t| [t, t, ws[t as usize].to_bits()]).collect();
            let rb = gpu.buffer_init("route", bytemuck::cast_slice(&ent));
            let mut b = gpu.begin();
            let mut gp = vec![inter as u32, dim as u32, 3, 0, bits(lim), (sl[0] / 4) as u32, dim as u32];
            gp.extend(lutp(ty));
            b.run("moe_gate_up_lut_8", &[sb.all(), xb.all(), rb.all(), hb.all()], &gp, ((inter as u32).div_ceil(8), 1, 1));
            let mut dp = vec![dim as u32, inter as u32, 3, 0, 0, ((sl[0] + sl[1]) / 4) as u32, 0];
            dp.extend(lutp(dn));
            b.run("moe_down_lut_8", &[sb.all(), hb.all(), rb.all(), ob.all()], &dp, ((dim as u32).div_ceil(8), 1, 1));
            let got = gpu.read_f32(b, &ob, 0, toks * dim);
            let range = want.iter().fold(0f32, |a, v| a.max(v.abs()));
            let worst = |got: &[f32]| got.iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / range;
            // and one token through the single-token kernels
            let mut b = gpu.begin();
            (gp[2], dp[2]) = (1, 1);
            b.run("moe_gate_up_lut_1", &[sb.all(), xb.all(), rb.all(), hb.all()], &gp, ((inter as u32).div_ceil(8), 1, 1));
            b.run("moe_down_lut_1", &[sb.all(), hb.all(), rb.all(), ob.all()], &dp, ((dim as u32).div_ceil(8), 1, 1));
            let got1 = gpu.read_f32(b, &ob, 0, toks * dim);
            let (e8, e1) = (worst(&got), worst(&got1));
            let ok = e8 < 1e-4 && e1 < 1e-4 && range > 0.0;
            println!("{} moe {name:<14} max err {:.2e} of output range", if ok { "PASS" } else { "FAIL" }, e8.max(e1));
            all &= ok;
            let job = crate::cpu_moe::Job { geom, blob: &host, xs, ws: ws.to_vec() };
            let e = worst(&crate::cpu_moe::experts_forward(&[job], inter, dim, lim)[0].concat());
            let ok = e < 0.03;
            println!("{} moe cpu {name:<10} max err {e:.2e} of output range", if ok { "PASS" } else { "FAIL" });
            all &= ok;
        }
    }

    // ---- rmsnorm ----
    {
        let (d, rows) = (5120, 3);
        let x = r.vec(rows * d, 2.0);
        let w = r.vec(d, 1.0);
        let (xb, wb, yb) = (fb(&x), fb(&w), gpu.buffer("y", (rows * d * 4) as u64));
        let mut b = gpu.begin();
        b.run("rmsnorm", &[xb.all(), wb.all(), yb.all()], &[d as u32, 0, 0, d as u32, d as u32, bits(1e-20), 1], (rows as u32, 1, 1));
        let got = gpu.read_f32(b, &yb, 0, rows * d);
        let mut want = vec![0f32; rows * d];
        for rr in 0..rows {
            let s: f32 = x[rr * d..(rr + 1) * d].iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (s + 1e-20).sqrt();
            for i in 0..d {
                want[rr * d + i] = x[rr * d + i] * inv * w[i];
            }
        }
        all &= check("rmsnorm", &got, &want, 1e-5);
    }

    // ---- rope (+ inverse round trip) ----
    {
        let (rows, heads, hd, half) = (6usize, 3usize, 128usize, 32usize);
        let x = r.vec(rows * hd, 1.0);
        let maxp = 64;
        let mut tab = vec![0f32; maxp * half * 2];
        for p in 0..maxp {
            for i in 0..half {
                let a = p as f32 * (1.0 / 10000f32.powf(2.0 * i as f32 / 64.0));
                tab[(p * half + i) * 2] = a.cos();
                tab[(p * half + i) * 2 + 1] = a.sin();
            }
        }
        let (xb, tb) = (fb(&x), fb(&tab));
        let mut b = gpu.begin();
        let p = |inv: u32| [rows as u32, heads as u32, hd as u32, (hd - 64) as u32, 7, 1, inv, 0, 0, half as u32];
        b.run("rope", &[xb.all(), tb.all()], &p(0), (((rows * half) as u32).div_ceil(256), 1, 1));
        let got = gpu.read_f32(b, &xb, 0, rows * hd);
        let mut want = x.clone();
        for rr in 0..rows {
            let pos = 7 + rr / heads;
            for i in 0..half {
                let (c, s) = (tab[(pos * half + i) * 2], tab[(pos * half + i) * 2 + 1]);
                let o = rr * hd + 64 + 2 * i;
                let (a, bb) = (x[o], x[o + 1]);
                want[o] = a * c - bb * s;
                want[o + 1] = a * s + bb * c;
            }
        }
        all &= check("rope", &got, &want, 1e-5);
        let mut b = gpu.begin();
        b.run("rope", &[xb.all(), tb.all()], &p(1), (((rows * half) as u32).div_ceil(256), 1, 1));
        let back = gpu.read_f32(b, &xb, 0, rows * hd);
        all &= check("rope inverse", &back, &x, 1e-5);
    }

    // ---- quant simulation ----
    for (mode, block, name) in [(0u32, 32usize, "quant fp8"), (1, 32, "quant fp4/e8m0"), (2, 16, "quant fp4/e4m3")] {
        let n = 512 * 4;
        let x = r.vec(n, 3.0);
        let xb = fb(&x);
        let mut b = gpu.begin();
        b.run("quant_sim", &[xb.all()], &[(n / block) as u32, block as u32, mode, 0, 512, 512], (((n / block) as u32).div_ceil(256), 1, 1));
        let got = gpu.read_f32(b, &xb, 0, n);
        let mut want = x.clone();
        for blk in want.chunks_mut(block) {
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            match mode {
                0 => {
                    let s = pow2_ceil(amax.max(1e-4) / 448.0);
                    blk.iter_mut().for_each(|v| *v = to_e4m3((*v / s).clamp(-448.0, 448.0)) * s);
                }
                1 => {
                    let s = pow2_ceil(amax.max(6.0 * f32::MIN_POSITIVE) / 6.0);
                    blk.iter_mut().for_each(|v| *v = to_e2m1((*v / s).clamp(-6.0, 6.0)) * s);
                }
                _ => {
                    let s = to_e4m3(amax.max(6.0 / 512.0) / 6.0);
                    blk.iter_mut().for_each(|v| *v = to_e2m1((*v / s).clamp(-6.0, 6.0)) * s);
                }
            }
        }
        all &= check(name, &got, &want, 1e-6);
    }

    // ---- hyper-connection mix (Sinkhorn) + pre + post ----
    {
        let (d, t) = (256usize, 2usize);
        let hdim = 4 * d;
        let h = r.vec(t * hdim, 1.0);
        let wf = r.vec(24 * hdim, 0.05);
        let mut sb = vec![0.7f32, 0.9, 1.1];
        sb.extend(r.vec(24, 0.5));
        let (hb, wb, sbb, mb) = (fb(&h), fb(&wf), fb(&sb), gpu.buffer("mix", (t * 24 * 4) as u64));
        let mut b = gpu.begin();
        let parts = (hdim as u32 / 4).div_ceil(crate::kernels::HC_PART);
        let pb = gpu.buffer("part", t as u64 * parts as u64 * 100);
        b.run("hc_dot", &[hb.all(), wb.all(), pb.all()], &[hdim as u32, 0], (parts, t as u32, 1));
        b.run("hc_mix", &[pb.all(), sbb.all(), mb.all()], &[hdim as u32, 4, 20, bits(1e-20), bits(1e-6), 0, 0, parts], (t as u32, 1, 1));
        let got = gpu.read_f32(b, &mb, 0, t * 24);
        let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
        let mut want = Vec::new();
        for tk in 0..t {
            let x = &h[tk * hdim..(tk + 1) * hdim];
            let rs = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / hdim as f32 + 1e-20).sqrt();
            let m: Vec<f32> = (0..24).map(|o| (0..hdim).map(|i| wf[o * hdim + i] * x[i]).sum::<f32>() * rs).collect();
            let eps = 1e-6;
            for j in 0..4 {
                want.push(sig(m[j] * sb[0] + sb[3 + j]) + eps);
            }
            for j in 0..4 {
                want.push(2.0 * sig(m[4 + j] * sb[1] + sb[7 + j]));
            }
            let mut c = [[0f32; 4]; 4];
            for j in 0..4 {
                for k in 0..4 {
                    c[j][k] = m[8 + j * 4 + k] * sb[2] + sb[11 + j * 4 + k];
                }
                let mx = c[j].iter().fold(f32::MIN, |a, &b| a.max(b));
                let s: f32 = c[j].iter().map(|v| (v - mx).exp()).sum();
                for k in 0..4 {
                    c[j][k] = (c[j][k] - mx).exp() / s + eps;
                }
            }
            let colnorm = |c: &mut [[f32; 4]; 4]| {
                for k in 0..4 {
                    let s: f32 = (0..4).map(|j| c[j][k]).sum::<f32>() + eps;
                    (0..4).for_each(|j| c[j][k] /= s);
                }
            };
            colnorm(&mut c);
            for _ in 1..20 {
                for j in 0..4 {
                    let s: f32 = c[j].iter().sum::<f32>() + eps;
                    c[j].iter_mut().for_each(|v| *v /= s);
                }
                colnorm(&mut c);
            }
            for row in c {
                want.extend(row);
            }
        }
        all &= check("hc_mix", &got, &want, 1e-4);

        // hc_post with these mixes
        let y = r.vec(t * d, 1.0);
        let yb = fb(&y);
        let dmy = gpu.buffer("d", 16);
        let mut b = gpu.begin();
        b.run("hc_post", &[hb.all(), yb.all(), dmy.all(), mb.all()], &[d as u32, t as u32, 0, 0, 0], (((t * d) as u32).div_ceil(256), 1, 1));
        let got = gpu.read_f32(b, &hb, 0, t * hdim);
        let mut wantp = vec![0f32; t * hdim];
        for tk in 0..t {
            let mix = &want[tk * 24..tk * 24 + 24];
            for k in 0..4 {
                for i in 0..d {
                    let mut v = mix[4 + k] * y[tk * d + i];
                    for j in 0..4 {
                        v += mix[8 + j * 4 + k] * h[tk * hdim + j * d + i];
                    }
                    wantp[tk * hdim + k * d + i] = v;
                }
            }
        }
        all &= check("hc_post", &got, &wantp, 1e-4);
    }

    // ---- top-k (radix select): ascending indices, ties to the lowest index ----
    for (n, k, tied) in [(3000usize, 512usize, false), (3000, 512, true), (513, 512, true), (700, 512, true), (300, 7, true), (400, 512, true)] {
        let rows = 2usize;
        let mut s = r.vec(rows * n, 10.0);
        if tied {
            for v in s.iter_mut() {
                *v = v.round().max(0.0);
            }
        }
        let (sb, ob) = (fb(&s), gpu.buffer("o", (rows * k * 4) as u64));
        let mut b = gpu.begin();
        b.run("topk", &[sb.all(), ob.all()], &[n as u32, k as u32, 0, 1, k as u32, 0, n as u32, 0], (rows as u32, 1, 1));
        let raw = gpu.read(b, &ob, 0, (rows * k * 4) as u64);
        let got: &[i32] = bytemuck::cast_slice(&raw);
        let mut ok = true;
        for rr in 0..rows {
            let row = &s[rr * n..(rr + 1) * n];
            let mut want: Vec<i32> = (0..n as i32).collect();
            want.sort_by(|&x, &y| row[y as usize].total_cmp(&row[x as usize]).then(x.cmp(&y)));
            want.truncate(k);
            want.sort();
            want.resize(k, -1);
            ok &= got[rr * k..(rr + 1) * k] == want[..];
        }
        println!("{} topk             {rows} rows of {n}, k={k}{}", if ok { "PASS" } else { "FAIL" }, if tied { ", ties" } else { "" });
        all &= ok;
    }

    // ---- sparse attention with sink ----
    {
        let (m, heads, hd, w, tk) = (2usize, 16usize, 64usize, 8usize, 4usize);
        let ring = 16;
        let q = r.vec(m * heads * hd, 1.0);
        let kw = r.vec(ring * hd, 1.0);
        let kc = r.vec(10 * hd, 1.0);
        let widx: Vec<i32> = (0..m * w).map(|i| if i % w < 2 && i < w { -1 } else { (i * 3 % ring) as i32 }).collect();
        let cidx: Vec<i32> = (0..m * tk).map(|i| if i % tk == 3 { -1 } else { (i * 7 % 10) as i32 }).collect();
        let sink = r.vec(heads, 1.0);
        let ns = w + tk;
        let scale = 0.125f32;
        let (qb, kwb, kcb) = (fb(&q), fb(&kw), fb(&kc));
        let wib = gpu.buffer_init("wi", bytemuck::cast_slice(&widx));
        let cib = gpu.buffer_init("ci", bytemuck::cast_slice(&cidx));
        let skb = fb(&sink);
        let att = gpu.buffer("att", (m * heads * ns * 4) as u64);
        let ob = gpu.buffer("o", (m * heads * hd * 4) as u64);
        let mut b = gpu.begin();
        let pp = [ns as u32, w as u32, tk as u32, heads as u32, hd as u32, bits(scale), 0];
        b.run("attn_scores", &[qb.all(), kwb.all(), kcb.all(), wib.all(), cib.all(), att.all()], &pp, ((ns as u32).div_ceil(16), (heads / 16) as u32, m as u32));
        b.run("attn_softmax", &[att.all(), skb.all()], &[ns as u32, heads as u32], (heads as u32, m as u32, 1));
        b.run("attn_pv", &[att.all(), kwb.all(), kcb.all(), wib.all(), cib.all(), ob.all()], &pp, ((hd / 16) as u32, (heads / 16) as u32, m as u32));
        let got = gpu.read_f32(b, &ob, 0, m * heads * hd);
        let mut want = vec![0f32; m * heads * hd];
        for mm in 0..m {
            let keys: Vec<Option<&[f32]>> = (0..ns)
                .map(|j| {
                    let ix = if j < w { widx[mm * w + j] } else { cidx[mm * tk + j - w] };
                    if ix < 0 {
                        None
                    } else if j < w {
                        Some(&kw[ix as usize * hd..(ix as usize + 1) * hd])
                    } else {
                        Some(&kc[ix as usize * hd..(ix as usize + 1) * hd])
                    }
                })
                .collect();
            for h in 0..heads {
                let qh = &q[(mm * heads + h) * hd..(mm * heads + h + 1) * hd];
                let sc: Vec<Option<f32>> = keys.iter().map(|k| k.map(|k| k.iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * scale)).collect();
                let mx = sc.iter().flatten().fold(sink[h], |a, &b| a.max(b));
                let den: f32 = sc.iter().flatten().map(|s| (s - mx).exp()).sum::<f32>() + (sink[h] - mx).exp();
                for dd in 0..hd {
                    want[(mm * heads + h) * hd + dd] = sc.iter().zip(&keys).filter_map(|(s, k)| Some((s.as_ref()? - mx).exp() / den * k.as_ref()?[dd])).sum();
                }
            }
        }
        all &= check("sparse attention", &got, &want, 1e-4);
    }

    // ---- hc_pre_norm ----
    {
        let (d, t) = (512usize, 3usize);
        let h = r.vec(t * 4 * d, 1.0);
        let mix = r.vec(t * 24, 1.0);
        let w = r.vec(d, 1.0);
        let (hb, mb, wb, yb) = (fb(&h), fb(&mix), fb(&w), gpu.buffer("y", (2 * d * 4) as u64));
        let mut b = gpu.begin();
        // tokens 1..3, output rows 0..2
        b.run("hc_pre_norm", &[hb.all(), mb.all(), wb.all(), yb.all()], &[d as u32, 0, bits(1e-20), 1, 0, 0], (2, 1, 1));
        let got = gpu.read_f32(b, &yb, 0, 2 * d);
        let mut want = Vec::new();
        for tk in 1..3 {
            let x: Vec<f32> = (0..d).map(|i| (0..4).map(|j| mix[tk * 24 + j] * h[(tk * 4 + j) * d + i]).sum()).collect();
            let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / d as f32 + 1e-20).sqrt();
            want.extend((0..d).map(|i| x[i] * inv * w[i]));
        }
        all &= check("hc_pre_norm", &got, &want, 1e-5);
    }

    // ---- engram gate ----
    {
        let (d, t) = (256usize, 2usize);
        let h = r.vec(t * 4 * d, 1.0);
        let ekv = r.vec(t * 5 * d, 1.0);
        let gw = r.vec(4 * d, 1.0);
        let (hb, eb, gb) = (fb(&h), fb(&ekv), fb(&gw));
        let mut b = gpu.begin();
        b.run("engram", &[hb.all(), eb.all(), gb.all()], &[d as u32, bits(1e-20), 0], (t as u32, 4, 1));
        let got = gpu.read_f32(b, &hb, 0, t * 4 * d);
        let mut want = h.clone();
        for tk in 0..t {
            for j in 0..4 {
                let hh = &h[(tk * 4 + j) * d..(tk * 4 + j + 1) * d];
                let key = &ekv[tk * 5 * d + j * d..tk * 5 * d + (j + 1) * d];
                let val = &ekv[tk * 5 * d + 4 * d..tk * 5 * d + 5 * d];
                let ms = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>() / d as f32;
                let rstd = 1.0 / (ms(hh) + 1e-20).sqrt() / (ms(key) + 1e-20).sqrt();
                let dot = (0..d).map(|i| hh[i] * gw[j * d + i] * key[i]).sum::<f32>() * rstd / (d as f32).sqrt();
                let z = dot.abs().max(1e-6).sqrt().copysign(dot);
                let g = 1.0 / (1.0 + (-z).exp());
                for i in 0..d {
                    want[(tk * 4 + j) * d + i] += g * val[i];
                }
            }
        }
        all &= check("engram", &got, &want, 1e-5);
    }

    // ---- compressor pooling (ratio 2) from rings ----
    {
        let (d, ring, ratio, j0, g) = (64usize, 16usize, 2usize, 6usize, 3usize);
        let kr = r.vec(ring * d, 1.0);
        let sr = r.vec(ring * d, 2.0);
        let w = r.vec(d, 1.0);
        let (kb, sb, wb, ob) = (fb(&kr), fb(&sr), fb(&w), gpu.buffer("o", (g * d * 4) as u64));
        let mut b = gpu.begin();
        b.run("compress", &[kb.all(), sb.all(), wb.all(), ob.all()], &[g as u32, d as u32, j0 as u32, ratio as u32, ring as u32, bits(1e-20)], (g as u32, 1, 1));
        let got = gpu.read_f32(b, &ob, 0, g * d);
        let mut want = Vec::new();
        for gi in 0..g {
            let j = j0 + gi;
            let v: Vec<f32> = (0..d)
                .map(|dd| {
                    let rows: Vec<usize> = (0..ratio).map(|i| ((j * ratio + i) % ring) * d + dd).collect();
                    let mx = rows.iter().map(|&r| sr[r]).fold(f32::MIN, f32::max);
                    let den: f32 = rows.iter().map(|&r| (sr[r] - mx).exp()).sum();
                    rows.iter().map(|&r| (sr[r] - mx).exp() / den * kr[r]).sum()
                })
                .collect();
            let inv = 1.0 / (v.iter().map(|x| x * x).sum::<f32>() / d as f32 + 1e-20).sqrt();
            want.extend((0..d).map(|i| v[i] * inv * w[i]));
        }
        all &= check("compress", &got, &want, 1e-5);
    }

    // ---- indexer scores (+ candidate mask) ----
    {
        let (heads, hd, n, m, pos0, ratio) = (32usize, 128usize, 300usize, 2usize, 597usize, 2usize);
        let q = r.vec(m * heads * hd, 1.0);
        let wp = r.vec(m * heads, 1.0);
        let kc = r.vec(n * hd, 1.0);
        let words = 64usize;
        let mut cand = vec![0u32; m * words];
        for mm in 0..m {
            for blk in (0..n.div_ceil(8)).step_by(3) {
                cand[mm * words + blk / 32] |= 1 << (blk % 32);
            }
        }
        let (qb, wb, kb) = (fb(&q), fb(&wp), fb(&kc));
        let cb = gpu.buffer_init("cand", bytemuck::cast_slice(&cand));
        let sb = gpu.buffer("s", (m * n * 4) as u64);
        let scale = 0.01f32;
        let mut b = gpu.begin();
        b.run(
            "idx_scores",
            &[qb.all(), wb.all(), kb.all(), cb.all(), sb.all()],
            &[n as u32, heads as u32, hd as u32, pos0 as u32, ratio as u32, bits(scale), 1, 8, words as u32],
            ((n as u32).div_ceil(256), m as u32, 1),
        );
        let got = gpu.read_f32(b, &sb, 0, m * n);
        let mut want = vec![0f32; m * n];
        for mm in 0..m {
            let nvis = (pos0 + mm + 1) / ratio;
            for t in 0..n {
                let blk = t / 8;
                let okc = cand[mm * words + blk / 32] >> (blk % 32) & 1 == 1;
                want[mm * n + t] = if t < nvis && okc {
                    (0..heads)
                        .map(|h| {
                            let dd: f32 = (0..hd).map(|i| q[(mm * heads + h) * hd + i] * kc[t * hd + i]).sum();
                            dd.max(0.0) * wp[mm * heads + h] * scale
                        })
                        .sum()
                } else {
                    -3.0e38
                };
            }
        }
        all &= check("idx_scores", &got, &want, 1e-4);
    }

    // ---- ring copy ----
    {
        let (m, d, ring, base) = (5usize, 8usize, 6usize, 4usize);
        let src = r.vec(m * d, 1.0);
        let (sb, db) = (fb(&src), gpu.buffer("dst", (ring * d * 4) as u64));
        let mut b = gpu.begin();
        b.run("copy_ring", &[sb.all(), db.all()], &[m as u32, d as u32, base as u32, ring as u32, 0, 0], (1, 1, 1));
        let got = gpu.read_f32(b, &db, 0, ring * d);
        let mut want = vec![0f32; ring * d];
        for i in 0..m {
            want[((base + i) % ring) * d..((base + i) % ring + 1) * d].copy_from_slice(&src[i * d..(i + 1) * d]);
        }
        all &= check("copy_ring", &got, &want, 0.0);
    }

    println!("{}", if all { "all kernels OK" } else { "SOME KERNELS FAILED" });
    if !all {
        anyhow::bail!("selftest failed");
    }
    Ok(())
}
