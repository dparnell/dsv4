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
    /// zero-padded to whole 64-byte vectors
    q: Vec<i8>,
    d: Vec<f32>,
    /// -12 * (sum of each 4 values of q): undoes the +12 that makes weights unsigned for VNNI
    corr: Vec<i32>,
}

fn quant_x(x: &[f32], isa: Isa) -> Q8x {
    let nb = x.len() / 32;
    let mut q = vec![0i8; nb.next_multiple_of(2) * 32];
    let mut d = vec![0f32; nb];
    #[cfg(target_arch = "x86_64")]
    if isa != Isa::Scalar {
        unsafe { quant_blocks_avx2(x, &mut q, &mut d) };
    }
    if isa == Isa::Scalar {
        for b in 0..nb {
            let xb = &x[b * 32..b * 32 + 32];
            let s = xb.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
            d[b] = s;
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for i in 0..32 {
                q[b * 32 + i] = (xb[i] * inv).round() as i8;
            }
        }
    }
    let corr = match isa {
        Isa::AvxVnni | Isa::Avx512Vnni => q.chunks_exact(4).map(|c| -12 * c.iter().map(|&v| v as i32).sum::<i32>()).collect(),
        _ => Vec::new(),
    };
    Q8x { q, d, corr }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quant_blocks_avx2(x: &[f32], q: &mut [i8], d: &mut [f32]) {
    use std::arch::x86_64::*;
    unsafe {
        let sign = _mm256_set1_ps(-0.0);
        // the largest float below 0.5: adding it and truncating rounds half away from zero without
        // pushing 0.49999997 up to 1
        let half = _mm256_set1_ps(f32::from_bits(0x3eff_ffff));
        let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        for (b, db) in d.iter_mut().enumerate() {
            let p = x.as_ptr().add(b * 32);
            let v = [_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8)), _mm256_loadu_ps(p.add(16)), _mm256_loadu_ps(p.add(24))];
            let mut m = _mm256_setzero_ps();
            for w in v {
                m = _mm256_max_ps(_mm256_andnot_ps(sign, w), m);
            }
            let m = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
            let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
            let m = _mm_cvtss_f32(_mm_max_ss(m, _mm_shuffle_ps(m, m, 1)));
            let s = m / 127.0;
            *db = s;
            let inv = _mm256_set1_ps(if s > 0.0 { 1.0 / s } else { 0.0 });
            let mut i = [_mm256_setzero_si256(); 4];
            for (o, w) in i.iter_mut().zip(v) {
                let w = _mm256_mul_ps(w, inv);
                *o = _mm256_cvttps_epi32(_mm256_add_ps(w, _mm256_or_ps(half, _mm256_and_ps(w, sign))));
            }
            let packed = _mm256_packs_epi16(_mm256_packs_epi32(i[0], i[1]), _mm256_packs_epi32(i[2], i[3]));
            _mm256_storeu_si256(q.as_mut_ptr().add(b * 32) as *mut __m256i, _mm256_permutevar8x32_epi32(packed, order));
        }
    }
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

/// Instruction set of the row kernels, chosen at run time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Isa {
    Scalar,
    Avx2,
    AvxVnni,
    Avx512Vnni,
}

impl Isa {
    const ALL: [Isa; 4] = [Isa::Scalar, Isa::Avx2, Isa::AvxVnni, Isa::Avx512Vnni];

    fn available(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            let avx2 = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
            match self {
                Isa::Scalar => true,
                Isa::Avx2 => avx2,
                Isa::AvxVnni => avx2 && is_x86_feature_detected!("avxvnni"),
                Isa::Avx512Vnni => {
                    avx2 && is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512vnni")
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self == Isa::Scalar
        }
    }
}

/// The best kernel this CPU has; DSV4_CPU_ISA=scalar|avx2|avxvnni|avx512vnni picks another.
pub fn isa() -> Isa {
    static I: std::sync::OnceLock<Isa> = std::sync::OnceLock::new();
    *I.get_or_init(|| {
        if std::env::var("DSV4_NO_AVX2").is_ok() {
            return Isa::Scalar;
        }
        let want = std::env::var("DSV4_CPU_ISA").ok().map(|v| v.to_lowercase());
        let usable: Vec<Isa> = Isa::ALL.into_iter().filter(|i| i.available()).collect();
        let named = usable.iter().copied().find(|i| Some(format!("{i:?}").to_lowercase()) == want);
        if want.is_some() && named.is_none() {
            tracing::warn!("DSV4_CPU_ISA is not available on this CPU; ignored");
        }
        named.or(usable.last().copied()).unwrap_or(Isa::Scalar)
    })
}

/// Fast enough to be worth running experts on.
pub fn has_simd() -> bool {
    isa() != Isa::Scalar
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(acc: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let s4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    _mm_cvtss_f32(_mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1)))
}

// Each row kernel computes one row against N inputs, unpacking the weights once; ACC separate
// sums per input keep the fused multiply-adds from waiting on one another. Callers guarantee
// qs = 16 bytes per block and inputs of sc.len() blocks.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_avx2<const N: usize, const ACC: usize>(qs: &[u8], sc: &[u8], x: [&Q8x; N]) -> [f32; N] {
    use std::arch::x86_64::*;
    unsafe {
        let tab = e8m0_table();
        let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(KV2.as_ptr() as *const __m128i));
        let mask = _mm_set1_epi8(0x0f);
        let ones = _mm256_set1_epi16(1);
        let mut acc = [[_mm256_setzero_ps(); ACC]; N];
        for (b, s) in sc.iter().enumerate() {
            let q = _mm_loadu_si128(qs.as_ptr().add(b * 16) as *const __m128i);
            let idx = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), mask), _mm_and_si128(q, mask));
            let w = _mm256_shuffle_epi8(lut, idx);
            let wa = _mm256_abs_epi8(w);
            let ws = tab[*s as usize];
            for t in 0..N {
                let xv = _mm256_loadu_si256(x[t].q.as_ptr().add(b * 32) as *const __m256i);
                let p16 = _mm256_maddubs_epi16(wa, _mm256_sign_epi8(xv, w));
                let p32 = _mm256_madd_epi16(p16, ones);
                let scale = _mm256_set1_ps(ws * *x[t].d.get_unchecked(b));
                let a = &mut acc[t][b % ACC];
                *a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scale, *a);
            }
        }
        let mut out = [0f32; N];
        for t in 0..N {
            let mut s = acc[t][0];
            for a in &acc[t][1..] {
                s = _mm256_add_ps(s, *a);
            }
            out[t] = hsum256(s);
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avxvnni")]
unsafe fn rows_avxvnni<const N: usize, const ACC: usize>(qs: &[u8], sc: &[u8], x: [&Q8x; N]) -> [f32; N] {
    use std::arch::x86_64::*;
    unsafe {
        let tab = e8m0_table();
        let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(KV2.as_ptr() as *const __m128i));
        let lut = _mm256_add_epi8(lut, _mm256_set1_epi8(12));
        let mask = _mm_set1_epi8(0x0f);
        let mut acc = [[_mm256_setzero_ps(); ACC]; N];
        for (b, s) in sc.iter().enumerate() {
            let q = _mm_loadu_si128(qs.as_ptr().add(b * 16) as *const __m128i);
            let idx = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), mask), _mm_and_si128(q, mask));
            let w = _mm256_shuffle_epi8(lut, idx);
            let ws = tab[*s as usize];
            for t in 0..N {
                let xv = _mm256_loadu_si256(x[t].q.as_ptr().add(b * 32) as *const __m256i);
                let c = _mm256_loadu_si256(x[t].corr.as_ptr().add(b * 8) as *const __m256i);
                let p32 = _mm256_dpbusd_avx_epi32(c, w, xv);
                let scale = _mm256_set1_ps(ws * *x[t].d.get_unchecked(b));
                let a = &mut acc[t][b % ACC];
                *a = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), scale, *a);
            }
        }
        let mut out = [0f32; N];
        for t in 0..N {
            let mut s = acc[t][0];
            for a in &acc[t][1..] {
                s = _mm256_add_ps(s, *a);
            }
            out[t] = hsum256(s);
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vnni")]
unsafe fn rows_avx512<const N: usize, const ACC: usize>(qs: &[u8], sc: &[u8], x: [&Q8x; N]) -> [f32; N] {
    use std::arch::x86_64::*;
    unsafe {
        let nb = sc.len();
        let lut = _mm512_broadcast_i32x4(_mm_loadu_si128(KV2.as_ptr() as *const __m128i));
        let lut = _mm512_add_epi8(lut, _mm512_set1_epi8(12));
        let mask = _mm512_set1_epi8(0x0f);
        let lanes = _mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3);
        let shifts = _mm512_setr_epi64(0, 0, 4, 4, 0, 0, 4, 4);
        let pair = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        let mut acc = [[_mm512_setzero_ps(); ACC]; N];
        let mut b0 = 0;
        while b0 < nb {
            let n = (nb - b0).min(16);
            let m16 = (0xffffu32 >> (16 - n)) as __mmask16;
            let e = if n == 16 {
                _mm_loadu_si128(sc.as_ptr().add(b0) as *const __m128i)
            } else {
                let mut t = [0u8; 16];
                t[..n].copy_from_slice(&sc[b0..]);
                _mm_loadu_si128(t.as_ptr() as *const __m128i)
            };
            // 2^(e-127) / 2, with the two smallest exponents (below 2^-126) as zero
            let e = _mm512_max_epi32(_mm512_sub_epi32(_mm512_cvtepu8_epi32(e), _mm512_set1_epi32(1)), _mm512_setzero_si512());
            let ws = _mm512_castsi512_ps(_mm512_slli_epi32(e, 23));
            let mut p = [_mm512_setzero_ps(); N];
            for t in 0..N {
                p[t] = _mm512_mul_ps(ws, _mm512_maskz_loadu_ps(m16, x[t].d.as_ptr().add(b0)));
            }
            for i in 0..n.div_ceil(2) {
                let b = b0 + 2 * i;
                let qp = qs.as_ptr().add(b * 16);
                let q = if b + 2 <= nb {
                    _mm256_loadu_si256(qp as *const __m256i)
                } else {
                    // the padding block has zero inputs, so whatever its weights are
                    _mm256_zextsi128_si256(_mm_loadu_si128(qp as *const __m128i))
                };
                let idx = _mm512_permutexvar_epi64(lanes, _mm512_castsi256_si512(q));
                let w = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srlv_epi64(idx, shifts), mask));
                let sel = _mm512_add_epi32(pair, _mm512_set1_epi32(2 * i as i32));
                for t in 0..N {
                    let xv = _mm512_loadu_si512(x[t].q.as_ptr().add(b * 32) as *const __m512i);
                    let c = _mm512_loadu_si512(x[t].corr.as_ptr().add(b * 8) as *const __m512i);
                    let p32 = _mm512_dpbusd_epi32(c, w, xv);
                    let a = &mut acc[t][i % ACC];
                    *a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p32), _mm512_permutexvar_ps(sel, p[t]), *a);
                }
            }
            b0 += 16;
        }
        let mut out = [0f32; N];
        for t in 0..N {
            let mut s = acc[t][0];
            for a in &acc[t][1..] {
                s = _mm512_add_ps(s, *a);
            }
            out[t] = _mm512_reduce_add_ps(s);
        }
        out
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

/// Most inputs one row kernel call takes.
const MAX_N: usize = 4;

/// One row times several inputs, `out[t] = row . xs[t]`.
#[inline]
fn dot_row(isa: Isa, qs: &[u8], sc: &[u8], xs: &[Q8x], out: &mut [f32]) {
    assert!(qs.len() == sc.len() * 16 && xs.len() == out.len());
    for (xs, out) in xs.chunks(MAX_N).zip(out.chunks_mut(MAX_N)) {
        assert!(xs.iter().all(|x| x.d.len() == sc.len()));
        #[cfg(target_arch = "x86_64")]
        macro_rules! by_count {
            ($f:ident) => {
                unsafe {
                    match xs {
                        [a] => out.copy_from_slice(&$f::<1, 4>(qs, sc, [a])),
                        [a, b] => out.copy_from_slice(&$f::<2, 2>(qs, sc, [a, b])),
                        [a, b, c] => out.copy_from_slice(&$f::<3, 2>(qs, sc, [a, b, c])),
                        [a, b, c, d] => out.copy_from_slice(&$f::<4, 1>(qs, sc, [a, b, c, d])),
                        _ => unreachable!(),
                    }
                }
            };
        }
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => by_count!(rows_avx2),
            #[cfg(target_arch = "x86_64")]
            Isa::AvxVnni => by_count!(rows_avxvnni),
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512Vnni => by_count!(rows_avx512),
            _ => xs.iter().zip(out).for_each(|(x, o)| *o = dot_row_q8_scalar(qs, sc, x)),
        }
    }
}

/// One matrix view inside an expert blob.
#[derive(Clone, Copy)]
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
    fn row(&self, isa: Isa, r: usize, xs: &[Q8x], out: &mut [f32]) {
        dot_row(isa, &self.qs[r * self.nb * 16..(r + 1) * self.nb * 16], &self.sc[r * self.nb..(r + 1) * self.nb], xs, out)
    }
}

/// One expert and the tokens it runs for.
pub struct Job<'a> {
    pub blob: &'a [u8],
    /// token inputs, `dim` each
    pub xs: Vec<&'a [f32]>,
    pub ws: Vec<f32>,
}

/// Rows per parallel task: a few hundred KB of weights, large next to the cost of handing it out.
const TASK: usize = 128;

/// y = W2 (w * silu(min(W1 x, l)) * clamp(W3 x, -l, l)) for several experts at once, each with
/// several tokens. Returns one dim-vector per token of each job.
pub fn experts_forward(jobs: &[Job], inter: usize, dim: usize, mat_bytes: usize, limit: f32) -> Vec<Vec<Vec<f32>>> {
    forward_isa(jobs, inter, dim, mat_bytes, limit, isa())
}

fn forward_isa(jobs: &[Job], inter: usize, dim: usize, mat_bytes: usize, limit: f32, isa: Isa) -> Vec<Vec<Vec<f32>>> {
    // every expert's rows go into one list of tasks, so the threads meet twice however many experts
    let xq: Vec<Vec<Q8x>> = jobs.iter().map(|j| j.xs.iter().map(|x| quant_x(x, isa)).collect()).collect();
    // row-major: hidden[i][t]
    let mut hidden: Vec<Vec<f32>> = jobs.iter().map(|j| vec![0f32; j.xs.len() * inter]).collect();
    let mut tasks = Vec::new();
    for ((j, xq), h) in jobs.iter().zip(&xq).zip(&mut hidden) {
        let gate = Mat::new(&j.blob[..mat_bytes], inter, dim);
        let up = Mat::new(&j.blob[mat_bytes..2 * mat_bytes], inter, dim);
        tasks.extend(h.chunks_mut(TASK * j.xs.len().max(1)).enumerate().map(|(c, out)| (c, out, j, xq, gate, up)));
    }
    tasks.into_par_iter().with_max_len(1).for_each(|(c, out, j, xq, gate, up)| {
        let n = j.xs.len();
        let (mut g, mut u) = (vec![0f32; n], vec![0f32; n]);
        for (o, row) in out.chunks_mut(n).enumerate() {
            let i = c * TASK + o;
            gate.row(isa, i, xq, &mut g);
            up.row(isa, i, xq, &mut u);
            for t in 0..n {
                let gv = g[t].min(limit);
                row[t] = gv / (1.0 + (-gv).exp()) * u[t].clamp(-limit, limit) * j.ws[t];
            }
        }
    });
    let column = |m: &[f32], n: usize, t: usize| -> Vec<f32> { m.iter().skip(t).step_by(n).copied().collect() };
    let hq: Vec<Vec<Q8x>> = jobs
        .iter()
        .zip(&hidden)
        .map(|(j, h)| (0..j.xs.len()).map(|t| quant_x(&column(h, j.xs.len(), t), isa)).collect())
        .collect();
    let mut out: Vec<Vec<f32>> = jobs.iter().map(|j| vec![0f32; j.xs.len() * dim]).collect();
    let mut tasks = Vec::new();
    for ((j, hq), o) in jobs.iter().zip(&hq).zip(&mut out) {
        let down = Mat::new(&j.blob[2 * mat_bytes..3 * mat_bytes], dim, inter);
        tasks.extend(o.chunks_mut(2 * TASK * hq.len().max(1)).enumerate().map(|(c, out)| (c, out, hq, down)));
    }
    tasks.into_par_iter().with_max_len(1).for_each(|(c, out, hq, down)| {
        for (o, row) in out.chunks_mut(hq.len()).enumerate() {
            down.row(isa, c * 2 * TASK + o, hq, row);
        }
    });
    jobs.iter().zip(&out).map(|(j, o)| (0..j.xs.len()).map(|t| column(o, j.xs.len(), t)).collect()).collect()
}

/// [`experts_forward`] for one expert.
pub fn expert_forward(blob: &[u8], inter: usize, dim: usize, mat_bytes: usize, xs: &[&[f32]], ws: &[f32], limit: f32) -> Vec<Vec<f32>> {
    experts_forward(&[Job { blob, xs: xs.to_vec(), ws: ws.to_vec() }], inter, dim, mat_bytes, limit).pop().unwrap()
}

/// Time experts on the CPU for the batch shapes decode and short prompts produce.
pub fn bench(dim: usize, inter: usize) -> anyhow::Result<()> {
    let mat = dim * inter / 2 + dim * inter / 32;
    // enough experts that every pass comes from DRAM, not a cache
    let blobs: Vec<Vec<u8>> = (0..48u64)
        .into_par_iter()
        .map(|i| {
            let mut z = i * 0x9E37_79B9 + 1;
            let mut b = vec![0u8; 3 * mat];
            for (k, v) in b.iter_mut().enumerate() {
                z ^= z << 13;
                z ^= z >> 7;
                z ^= z << 17;
                let is_scale = k % mat >= dim * inter / 2;
                *v = if is_scale { 114 + (z % 6) as u8 } else { z as u8 };
            }
            b
        })
        .collect();
    let x: Vec<f32> = (0..dim).map(|i| ((i * 7919) % 1000) as f32 / 500.0 - 1.0).collect();
    let have: Vec<Isa> = Isa::ALL.into_iter().filter(|i| i.available()).collect();
    println!("expert {:.1} MB, kernels {have:?}, using {:?}", 3.0 * mat as f64 / 1e6, isa());
    let cores = std::thread::available_parallelism()?.get();
    let mut counts = vec![1, 4, 8, cores / 2, cores * 3 / 4, cores];
    counts.sort();
    counts.dedup();
    for threads in counts.into_iter().filter(|&t| t > 0) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
        let mut line = format!("{threads:>3} threads:");
        for (name, group, nt) in [("1 expert", 1, 1), ("6 experts", 6, 1), ("6 experts x 4 tokens", 6, 4)] {
            let mut best = f64::MAX;
            for _ in 0..3 {
                let t0 = std::time::Instant::now();
                pool.install(|| {
                    for g in blobs.chunks(group) {
                        let jobs: Vec<Job> = g.iter().map(|b| Job { blob: b, xs: vec![&x[..]; nt], ws: vec![1.0; nt] }).collect();
                        std::hint::black_box(experts_forward(&jobs, inter, dim, mat, 10.0));
                    }
                });
                best = best.min(t0.elapsed().as_secs_f64() * 1e3 / blobs.len() as f64);
            }
            line += &format!("  {name} {best:.2} ms ({:.0} GB/s)", 3.0 * mat as f64 / best / 1e6);
        }
        println!("{line}");
    }
    Ok(())
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
        let xq = quant_x(&x, Isa::Scalar);
        let s = dot_row_q8_scalar(&qs, &sc, &xq);
        println!("want {want} scalar {s}");
        assert!((s - want).abs() < 0.02 * (1.0 + want.abs()));
    }

    /// Every kernel, input count and block count agrees with the scalar code; the integer
    /// arithmetic is the same, only the order of the float sums differs.
    #[test]
    fn kernels_agree() {
        let mut seed = 99u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for nb in [1usize, 2, 3, 15, 16, 17, 31, 72, 160] {
            let qs: Vec<u8> = (0..nb * 16).map(|_| rnd() as u8).collect();
            let sc: Vec<u8> = (0..nb).map(|_| 110 + (rnd() % 12) as u8).collect();
            let x: Vec<Vec<f32>> = (0..9).map(|_| (0..nb * 32).map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0).collect()).collect();
            for isa in Isa::ALL.into_iter().filter(|i| i.available()) {
                for n in 1..=9 {
                    let xq: Vec<Q8x> = x[..n].iter().map(|x| quant_x(x, isa)).collect();
                    let mut got = vec![0f32; n];
                    dot_row(isa, &qs, &sc, &xq, &mut got);
                    for (t, g) in got.iter().enumerate() {
                        let xs = quant_x(&x[t], Isa::Scalar);
                        assert_eq!(xq[t].q, xs.q, "{isa:?} quantization");
                        assert_eq!(xq[t].d, xs.d);
                        let want = dot_row_q8_scalar(&qs, &sc, &xs);
                        // an upper bound on the size of the terms summed
                        let mag: f32 = sc.iter().map(|&e| e8m0_half(e) * 12.0 * 32.0).sum();
                        assert!((g - want).abs() <= 1e-6 * mag, "{isa:?} nb {nb} n {n}: {g} vs {want}");
                    }
                }
            }
        }
    }

    /// A batch gives every token what it would get alone (rows and tokens are not mixed up).
    #[test]
    fn batch_matches_single() {
        let (inter, dim) = (96usize, 160usize);
        let mat = dim * inter / 2 + dim * inter / 32;
        let mk = |i: usize| -> Vec<u8> { (0..3 * mat).map(|k| if k % mat >= dim * inter / 2 { 116 + (k % 4) as u8 } else { (k * 31 + i * 17) as u8 }).collect() };
        let blobs: Vec<Vec<u8>> = (0..3).map(mk).collect();
        let xs: Vec<Vec<f32>> = (0..5).map(|t| (0..dim).map(|i| ((i * 37 + t * 101) % 200) as f32 / 100.0 - 1.0).collect()).collect();
        for isa in Isa::ALL.into_iter().filter(|i| i.available()) {
            let jobs: Vec<Job> = blobs
                .iter()
                .enumerate()
                .map(|(i, b)| Job { blob: b, xs: xs[..i + 3].iter().map(|x| &x[..]).collect(), ws: (0..i + 3).map(|t| 0.5 + t as f32).collect() })
                .collect();
            let all = forward_isa(&jobs, inter, dim, mat, 10.0, isa);
            for (j, got) in jobs.iter().zip(&all) {
                for t in 0..j.xs.len() {
                    let one = Job { blob: j.blob, xs: vec![j.xs[t]], ws: vec![j.ws[t]] };
                    let want = &forward_isa(&[one], inter, dim, mat, 10.0, Isa::Scalar)[0][0];
                    let range = want.iter().fold(0f32, |a, v| a.max(v.abs()));
                    let worst = got[t].iter().zip(want).fold(0f32, |a, (g, w)| a.max((g - w).abs()));
                    assert!(worst <= 0.02 * range, "{isa:?} token {t}: {worst} of {range}");
                }
            }
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
