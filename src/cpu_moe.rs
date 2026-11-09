//! CPU execution of routed experts that are not resident in VRAM.
//!
//! For decode, a missed expert is used by a single token: running it on the CPU straight from
//! the RAM tier reads its 18.8 MB once at DRAM speed, instead of pushing it over PCIe first.
//! Expert bytes use the same repacked layout as the GPU slots:
//! [gate | up | down], each [qs rows*nb*16][e8m0 rows*nb].

use rayon::prelude::*;

const KV2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12]; // doubled e2m1

#[inline]
fn e8m0_half(e: u8) -> f32 {
    // 2^(e-127) / 2 (the LUT holds doubled values)
    if e < 2 { 2f32.powi(e as i32 - 128) } else { f32::from_bits(((e as u32) - 1) << 23) }
}

/// x quantized per 32-block to int8 with one f32 scale (like Q8_0)
pub struct Q8x {
    q: Vec<i8>,
    d: Vec<f32>,
}

pub fn quant_x(x: &[f32]) -> Q8x {
    let nb = x.len() / 32;
    let mut q = vec![0i8; x.len()];
    let mut d = vec![0f32; nb];
    for b in 0..nb {
        let xb = &x[b * 32..b * 32 + 32];
        let amax = xb.iter().fold(0f32, |a, v| a.max(v.abs()));
        let s = amax / 127.0;
        d[b] = s;
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for i in 0..32 {
            q[b * 32 + i] = (xb[i] * inv).round() as i8;
        }
    }
    Q8x { q, d }
}

fn e8m0_table() -> &'static [f32; 256] {
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (e, v) in t.iter_mut().enumerate() {
            *v = e8m0_half(e as u8);
        }
        t
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_row_avx2(qs: &[u8], sc: &[u8], x: &Q8x) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let tab = e8m0_table();
        let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(KV2.as_ptr() as *const __m128i));
        let mask = _mm_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        for (b, s) in sc.iter().enumerate() {
            let q = _mm_loadu_si128(qs.as_ptr().add(b * 16) as *const __m128i);
            let idx = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), mask), _mm_and_si128(q, mask));
            let w = _mm256_shuffle_epi8(lut, idx);
            let xv = _mm256_loadu_si256(x.q.as_ptr().add(b * 32) as *const __m256i);
            let p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(xv, w));
            let p32 = _mm256_madd_epi16(p16, ones);
            let scale = _mm256_set1_ps(tab[*s as usize] * *x.d.get_unchecked(b));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scale, acc);
        }
        let s4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
        _mm_cvtss_f32(s1)
    }
}

fn dot_row_q8_scalar(qs: &[u8], sc: &[u8], x: &Q8x) -> f32 {
    let tab = e8m0_table();
    let mut acc = 0f32;
    for (b, s) in sc.iter().enumerate() {
        let q = &qs[b * 16..b * 16 + 16];
        let xb = &x.q[b * 32..b * 32 + 32];
        let mut t = 0i32;
        for j in 0..16 {
            t += KV2[(q[j] & 15) as usize] as i32 * xb[j] as i32 + KV2[(q[j] >> 4) as usize] as i32 * xb[j + 16] as i32;
        }
        acc += t as f32 * tab[*s as usize] * x.d[b];
    }
    acc
}

pub fn has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && std::env::var("DSV4_NO_AVX2").is_err()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn dot_row(avx2: bool, qs: &[u8], sc: &[u8], x: &Q8x) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return unsafe { dot_row_avx2(qs, sc, x) };
    }
    let _ = avx2;
    dot_row_q8_scalar(qs, sc, x)
}

/// One matrix view inside an expert blob.
struct Mat<'a> {
    qs: &'a [u8],
    sc: &'a [u8],
    nb: usize,
}

impl<'a> Mat<'a> {
    fn new(blob: &'a [u8], rows: usize, k: usize) -> Self {
        let nb = k / 32;
        let (qs, sc) = blob.split_at(rows * nb * 16);
        Mat { qs, sc: &sc[..rows * nb], nb }
    }
    #[inline]
    fn row(&self, avx2: bool, r: usize, x: &Q8x) -> f32 {
        dot_row(avx2, &self.qs[r * self.nb * 16..(r + 1) * self.nb * 16], &self.sc[r * self.nb..(r + 1) * self.nb], x)
    }
}

/// y = W2 (w * silu(min(W1 x, l)) * clamp(W3 x, -l, l)) for several tokens of one expert.
/// `xs` are the token inputs (dim each); returns one dim-vector per token.
pub fn expert_forward(blob: &[u8], inter: usize, dim: usize, mat_bytes: usize, xs: &[&[f32]], ws: &[f32], limit: f32) -> Vec<Vec<f32>> {
    let avx2 = has_avx2();
    let gate = Mat::new(&blob[..mat_bytes], inter, dim);
    let up = Mat::new(&blob[mat_bytes..2 * mat_bytes], inter, dim);
    let down = Mat::new(&blob[2 * mat_bytes..3 * mat_bytes], dim, inter);
    let n = xs.len();
    let xq: Vec<Q8x> = xs.iter().map(|x| quant_x(x)).collect();
    // hidden[t][i], token-major
    let mut hidden = vec![0f32; n * inter];
    hidden.par_chunks_mut(128).enumerate().for_each(|(c, out)| {
        for (o, slot) in out.iter_mut().enumerate() {
            let idx = c * 128 + o;
            let (t, i) = (idx / inter, idx % inter);
            let g = gate.row(avx2, i, &xq[t]).min(limit);
            let u = up.row(avx2, i, &xq[t]).clamp(-limit, limit);
            *slot = g / (1.0 + (-g).exp()) * u * ws[t];
        }
    });
    let hq: Vec<Q8x> = hidden.chunks(inter).map(quant_x).collect();
    let mut out = vec![0f32; n * dim];
    out.par_chunks_mut(128).enumerate().for_each(|(c, o)| {
        for (j, slot) in o.iter_mut().enumerate() {
            let idx = c * 128 + j;
            let (t, r) = (idx / dim, idx % dim);
            *slot = down.row(avx2, r, &hq[t]);
        }
    });
    out.chunks(dim).map(|c| c.to_vec()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_dot_matches_f32() {
        let nb = 4;
        let mut seed = 12345u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let qs: Vec<u8> = (0..nb * 16).map(|_| rnd() as u8).collect();
        let sc: Vec<u8> = (0..nb).map(|_| 120 + (rnd() % 8) as u8).collect();
        let x: Vec<f32> = (0..nb * 32).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
        let mut want = 0f32;
        for b in 0..nb {
            for j in 0..16 {
                let lo = KV2[(qs[b * 16 + j] & 15) as usize] as f32;
                let hi = KV2[(qs[b * 16 + j] >> 4) as usize] as f32;
                want += (lo * x[b * 32 + j] + hi * x[b * 32 + 16 + j]) * e8m0_half(sc[b]);
            }
        }
        let xq = quant_x(&x);
        let s = dot_row_q8_scalar(&qs, &sc, &xq);
        println!("want {want} scalar {s}");
        assert!((s - want).abs() < 0.02 * (1.0 + want.abs()));
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            let a = unsafe { dot_row_avx2(&qs, &sc, &xq) };
            println!("avx2 {a}");
            assert!((a - want).abs() < 0.02 * (1.0 + want.abs()));
        }
    }

    #[test]
    fn expert_matches_f32() {
        let (inter, dim) = (64usize, 128usize);
        let mut seed = 777u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut blob = Vec::new();
        let mut deq: Vec<Vec<f32>> = Vec::new();
        for &(rows, k) in &[(inter, dim), (inter, dim), (dim, inter)] {
            let nb = k / 32;
            let qs: Vec<u8> = (0..rows * nb * 16).map(|_| rnd() as u8).collect();
            let sc: Vec<u8> = (0..rows * nb).map(|_| 116 + (rnd() % 4) as u8).collect();
            let mut d = vec![0f32; rows * k];
            for r in 0..rows {
                for b in 0..nb {
                    for j in 0..16 {
                        let q = qs[(r * nb + b) * 16 + j];
                        let s = e8m0_half(sc[r * nb + b]);
                        d[r * k + b * 32 + j] = KV2[(q & 15) as usize] as f32 * s;
                        d[r * k + b * 32 + 16 + j] = KV2[(q >> 4) as usize] as f32 * s;
                    }
                }
            }
            blob.extend(qs);
            blob.extend(sc);
            deq.push(d);
        }
        let mat = blob.len() / 3;
        let x: Vec<f32> = (0..dim).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect();
        let lim = 10.0;
        let got = expert_forward(&blob, inter, dim, mat, &[&x], &[1.2], lim);
        let h: Vec<f32> = (0..inter)
            .map(|n| {
                let g = (0..dim).map(|i| deq[0][n * dim + i] * x[i]).sum::<f32>().min(lim);
                let u = (0..dim).map(|i| deq[1][n * dim + i] * x[i]).sum::<f32>().clamp(-lim, lim);
                g / (1.0 + (-g).exp()) * u * 1.2
            })
            .collect();
        let want: Vec<f32> = (0..dim).map(|r| (0..inter).map(|i| deq[2][r * inter + i] * h[i]).sum()).collect();
        let range = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        let worst = got[0].iter().zip(&want).fold(0f32, |a, (g, w)| a.max((g - w).abs())) / range;
        println!("mat {mat} worst {worst}");
        assert!(worst < 0.02);
    }
}
