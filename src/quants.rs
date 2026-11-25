//! Decoders for ggml's block formats.
//!
//! The formats used for routed experts all have the form
//!     weight = w * s * d  (- m * dm for Q2_K, Q4_K and Q5_K)
//! with a small integer w, an integer s per 16 weights and a float d per 256. `decode` returns
//! those parts; w takes at most 16 values in most formats, so on the GPU it is stored as 4-bit codes
//! into a table (`codes`), except for the `wide` ones, which keep a byte. Nothing is rounded on the way.
//!
//! Formats, tables and bit layouts are ggml's (MIT, see data/NOTICE).

use std::sync::OnceLock;

use half::f16;

use crate::gguf::{GGML_BF16, GGML_F16, GGML_F32, GGML_MXFP4, GGML_Q8_0};

pub const Q2_K: u32 = 10;
pub const Q3_K: u32 = 11;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;
pub const IQ2_XXS: u32 = 16;
pub const IQ2_XS: u32 = 17;
pub const IQ3_XXS: u32 = 18;
pub const IQ1_S: u32 = 19;
pub const IQ4_NL: u32 = 20;
pub const IQ3_S: u32 = 21;
pub const IQ2_S: u32 = 22;
pub const IQ4_XS: u32 = 23;
pub const IQ1_M: u32 = 29;

/// Formats `decode` handles, i.e. those routed experts can have besides MXFP4.
pub fn is_low_bit(ty: u32) -> bool {
    LOW_BIT.contains(&ty)
}

pub const LOW_BIT: [u32; 13] = [Q2_K, Q3_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K];

/// Does w take more values than 4-bit codes can tell apart?
pub fn wide(ty: u32) -> bool {
    matches!(ty, Q5_K | Q6_K)
}

/// Does the format have the `m * dm` offset?
pub fn has_min(ty: u32) -> bool {
    matches!(ty, Q2_K | Q4_K | Q5_K)
}

struct Tables {
    g2xxs: Vec<u64>,
    g2xs: Vec<u64>,
    g2s: Vec<u64>,
    g3xxs: Vec<u32>,
    g3s: Vec<u32>,
    /// 8 * (grid + 1/8) and 8 * (grid - 1/8)
    g1: [Vec<u64>; 2],
    /// bytes of 0xff where the sign bit is set
    signs: [u64; 256],
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let raw = include_bytes!("../data/iq_grids.bin");
        let mut p = 0;
        let mut take = |n: usize| {
            p += n;
            &raw[p - n..p]
        };
        let q = |b: &[u8]| b.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
        let d = |b: &[u8]| b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
        let (g2xxs, g2xs, g2s) = (q(take(256 * 8)), q(take(512 * 8)), q(take(1024 * 8)));
        let (g3xxs, g3s) = (d(take(256 * 4)), d(take(512 * 4)));
        let g1s = q(take(2048 * 8));
        let shifted = |by: i8| g1s.iter().map(|v| u64::from_le_bytes(v.to_le_bytes().map(|b| (b as i8 * 8 + by) as u8))).collect();
        let mut signs = [0u64; 256];
        for (i, s) in signs.iter_mut().enumerate() {
            *s = u64::from_le_bytes(std::array::from_fn(|j| if i >> j & 1 == 1 { 0xff } else { 0 }));
        }
        Tables { g2xxs, g2xs, g2s, g3xxs, g3s, g1: [shifted(1), shifted(-1)], signs }
    })
}

/// 256 weights.
#[derive(Clone)]
#[repr(align(64))]
pub struct Block {
    pub w: [i8; 256],
    pub s: [i8; 16],
    pub m: [u8; 16],
    pub d: f32,
    pub dm: f32,
}

impl Block {
    pub fn new() -> Self {
        Block { w: [0; 256], s: [0; 16], m: [0; 16], d: 0.0, dm: 0.0 }
    }
}

fn half_at(b: &[u8], o: usize) -> f32 {
    f16::from_le_bytes([b[o], b[o + 1]]).to_f32()
}

/// Negate the bytes of `g` (each nonzero and positive) where `m` is 0xff.
#[inline(always)]
fn apply_signs(g: u64, m: u64) -> u64 {
    // ~x + 1 cannot carry out of a byte: ~x is at most 0xfe
    (g ^ m) + (m & 0x0101_0101_0101_0101)
}

/// The 8-bit sign pattern from 7 stored bits: the last one makes the parity even.
#[inline(always)]
fn even(bits: u32) -> usize {
    let b = bits & 127;
    (b | (b.count_ones() & 1) << 7) as usize
}

#[inline(always)]
fn put(o: &mut Block, i: usize, v: u64) {
    let c: [u8; 8] = v.to_le_bytes();
    o.w[i * 8..i * 8 + 8].copy_from_slice(bytemuck::cast_slice(&c));
}

/// Decode one 256-weight block of a low-bit format.
#[inline(always)]
pub fn decode(ty: u32, b: &[u8], o: &mut Block) {
    let t = tables();
    let pair = |lo: u32, hi: u32| lo as u64 | (hi as u64) << 32;
    let le32 = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
    let le16 = |i: usize| u16::from_le_bytes(b[i..i + 2].try_into().unwrap()) as usize;
    let odd = |v: u32| (2 * v + 1) as i8;
    match ty {
        IQ2_XXS => {
            let b: &[u8; 66] = b.try_into().unwrap();
            o.d = half_at(b, 0) / 8.0;
            for i in 0..8 {
                let c = le32(6 + 8 * i);
                o.s[2 * i] = odd(c >> 28);
                o.s[2 * i + 1] = o.s[2 * i];
                for l in 0..4 {
                    put(o, 4 * i + l, apply_signs(t.g2xxs[b[2 + 8 * i + l] as usize], t.signs[even(c >> (7 * l))]));
                }
            }
        }
        IQ2_XS => {
            let b: &[u8; 74] = b.try_into().unwrap();
            o.d = half_at(b, 0) / 8.0;
            for i in 0..8 {
                o.s[2 * i] = odd(b[66 + i] as u32 & 15);
                o.s[2 * i + 1] = odd(b[66 + i] as u32 >> 4);
                for l in 0..4 {
                    let q = le16(2 + 2 * (4 * i + l));
                    put(o, 4 * i + l, apply_signs(t.g2xs[q & 511], t.signs[even(q as u32 >> 9)]));
                }
            }
        }
        IQ2_S => {
            let b: &[u8; 82] = b.try_into().unwrap();
            o.d = half_at(b, 0) / 8.0;
            for i in 0..8 {
                o.s[2 * i] = odd(b[74 + i] as u32 & 15);
                o.s[2 * i + 1] = odd(b[74 + i] as u32 >> 4);
                let hi = b[66 + i] as usize;
                for l in 0..4 {
                    let g = t.g2s[b[2 + 4 * i + l] as usize | (hi << (8 - 2 * l) & 0x300)];
                    put(o, 4 * i + l, apply_signs(g, t.signs[b[34 + 4 * i + l] as usize]));
                }
            }
        }
        IQ3_XXS => {
            let b: &[u8; 98] = b.try_into().unwrap();
            o.d = half_at(b, 0) / 4.0;
            for i in 0..8 {
                let c = le32(66 + 4 * i);
                o.s[2 * i] = odd(c >> 28);
                o.s[2 * i + 1] = o.s[2 * i];
                for l in 0..4 {
                    let g = pair(t.g3xxs[b[2 + 8 * i + 2 * l] as usize], t.g3xxs[b[3 + 8 * i + 2 * l] as usize]);
                    put(o, 4 * i + l, apply_signs(g, t.signs[even(c >> (7 * l))]));
                }
            }
        }
        IQ3_S => {
            let b: &[u8; 110] = b.try_into().unwrap();
            o.d = half_at(b, 0);
            for i in 0..8 {
                o.s[2 * i] = odd(b[106 + i / 2] as u32 >> (4 * (i & 1)) & 15);
                o.s[2 * i + 1] = o.s[2 * i];
                let hi = b[66 + i] as usize;
                for l in 0..4 {
                    let g = pair(
                        t.g3s[b[2 + 8 * i + 2 * l] as usize | (hi << (8 - 2 * l) & 256)],
                        t.g3s[b[3 + 8 * i + 2 * l] as usize | (hi << (7 - 2 * l) & 256)],
                    );
                    put(o, 4 * i + l, apply_signs(g, t.signs[b[74 + 4 * i + l] as usize]));
                }
            }
        }
        IQ1_S => {
            let b: &[u8; 50] = b.try_into().unwrap();
            o.d = half_at(b, 0) / 8.0;
            for i in 0..8 {
                let q = le16(34 + 2 * i);
                o.s[2 * i] = odd((q >> 12 & 7) as u32);
                o.s[2 * i + 1] = o.s[2 * i];
                let g = &t.g1[q >> 15];
                for l in 0..4 {
                    put(o, 4 * i + l, g[b[2 + 4 * i + l] as usize | (q >> (3 * l) & 7) << 8]);
                }
            }
        }
        Q3_K => {
            let b: &[u8; 110] = b.try_into().unwrap();
            o.d = half_at(b, 108);
            let s = &b[96..108];
            for i in 0..16 {
                let lo = if i < 8 { s[i] & 15 } else { s[i - 8] >> 4 };
                let hi = s[8 + i % 4] >> (2 * (i / 4)) & 3;
                o.s[i] = (lo | hi << 4) as i8 - 32;
            }
            for (j, w) in o.w.chunks_exact_mut(32).enumerate() {
                let q = &b[32 + j / 4 * 32..][..32];
                let sh = 2 * (j % 4);
                for ((w, q), h) in w.iter_mut().zip(q).zip(&b[..32]) {
                    *w = ((q >> sh & 3) | (h >> j & 1) << 2) as i8 - 4;
                }
            }
        }
        Q2_K => {
            let b: &[u8; 84] = b.try_into().unwrap();
            o.d = half_at(b, 80);
            o.dm = half_at(b, 82);
            for i in 0..16 {
                o.s[i] = (b[i] & 15) as i8;
                o.m[i] = b[i] >> 4;
            }
            for (j, w) in o.w.chunks_exact_mut(32).enumerate() {
                let q = &b[16 + j / 4 * 32..][..32];
                let sh = 2 * (j % 4);
                for (w, q) in w.iter_mut().zip(q) {
                    *w = (q >> sh & 3) as i8;
                }
            }
        }
        IQ1_M => {
            let b: &[u8; 56] = b.try_into().unwrap();
            let sc: [usize; 4] = std::array::from_fn(|i| le16(48 + 2 * i));
            o.d = f16::from_bits((sc[0] >> 12 | sc[1] >> 8 & 0xf0 | sc[2] >> 4 & 0xf00 | sc[3] & 0xf000) as u16).to_f32() / 8.0;
            for i in 0..16 {
                let h = b[32 + i] as usize;
                o.s[i] = odd((sc[i / 4] >> (3 * (i % 4)) & 7) as u32);
                put(o, 2 * i, t.g1[h >> 3 & 1][b[2 * i] as usize | (h & 7) << 8]);
                put(o, 2 * i + 1, t.g1[h >> 7][b[2 * i + 1] as usize | (h >> 4 & 7) << 8]);
            }
        }
        IQ4_XS => {
            let b: &[u8; 136] = b.try_into().unwrap();
            o.d = half_at(b, 0);
            let hi = le16(2);
            for j in 0..8 {
                let s = (b[4 + j / 2] >> (4 * (j & 1)) & 15) as i8 | ((hi >> (2 * j) & 3) << 4) as i8;
                o.s[2 * j] = s - 32;
                o.s[2 * j + 1] = s - 32;
                for l in 0..16 {
                    o.w[32 * j + l] = IQ4_NL_VALUES[(b[8 + 16 * j + l] & 15) as usize];
                    o.w[32 * j + 16 + l] = IQ4_NL_VALUES[(b[8 + 16 * j + l] >> 4) as usize];
                }
            }
        }
        Q4_K | Q5_K => {
            let low = if ty == Q4_K { 16 } else { 48 };
            let b = &b[..low + 128];
            o.d = half_at(b, 0);
            o.dm = half_at(b, 2);
            let s = &b[4..16];
            for j in 0..8 {
                let (sc, m) = if j < 4 {
                    (s[j] & 63, s[j + 4] & 63)
                } else {
                    (s[j + 4] & 15 | s[j - 4] >> 6 << 4, s[j + 4] >> 4 | s[j] >> 6 << 4)
                };
                (o.s[2 * j], o.s[2 * j + 1]) = (sc as i8, sc as i8);
                (o.m[2 * j], o.m[2 * j + 1]) = (m, m);
            }
            for (j, w) in o.w.chunks_exact_mut(32).enumerate() {
                let q = &b[low + j / 2 * 32..][..32];
                let sh = 4 * (j & 1);
                for (w, q) in w.iter_mut().zip(q) {
                    *w = (q >> sh & 15) as i8;
                }
                if ty == Q5_K {
                    for (w, h) in w.iter_mut().zip(&b[16..48]) {
                        *w |= ((h >> j & 1) << 4) as i8;
                    }
                }
            }
        }
        Q6_K => {
            let b: &[u8; 210] = b.try_into().unwrap();
            o.d = half_at(b, 208);
            for i in 0..16 {
                o.s[i] = b[192 + i] as i8;
            }
            for (j, w) in o.w.chunks_exact_mut(32).enumerate() {
                let (n, part) = (j / 4, j % 4);
                let lo = &b[n * 64 + 32 * (part & 1)..][..32];
                let hi = &b[128 + n * 32..][..32];
                let (a, c) = (4 * (part >> 1), 2 * part);
                for ((w, lo), hi) in w.iter_mut().zip(lo).zip(hi) {
                    *w = ((lo >> a & 15) | (hi >> c & 3) << 4) as i8 - 32;
                }
            }
        }
        _ => unreachable!("not a low-bit format: {ty}"),
    }
}

/// `decode` for a whole row: the weights, their scales s * d, and for formats with them the offsets m * dm.
#[inline(always)]
fn row_by_block(ty: u32, raw: &[u8], w: &mut [i8], s: &mut [f32], m: &mut [f32]) {
    let bb = crate::gguf::type_block(ty).unwrap().1 as usize;
    let mut blk = Block::new();
    for (b, raw) in raw.chunks_exact(bb).enumerate() {
        decode(ty, raw, &mut blk);
        w[b * 256..][..256].copy_from_slice(&blk.w);
        for (o, &v) in s[b * 16..][..16].iter_mut().zip(&blk.s) {
            *o = v as f32 * blk.d;
        }
        if has_min(ty) {
            for (o, &v) in m[b * 16..][..16].iter_mut().zip(&blk.m) {
                *o = v as f32 * blk.dm;
            }
        }
    }
}

/// `avx2`: the CPU has AVX2, FMA, F16C and POPCNT. The results are the same without.
pub fn decode_row(avx2: bool, ty: u32, raw: &[u8], w: &mut [i8], s: &mut [f32], m: &mut [f32]) {
    let bb = crate::gguf::type_block(ty).unwrap().1 as usize;
    let nb = raw.len() / bb;
    assert!(raw.len() == nb * bb && w.len() == nb * 256 && s.len() == nb * 16 && m.len() == if has_min(ty) { nb * 16 } else { 0 });
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        return unsafe { x86::decode_row(ty, raw, w, s, m) };
    }
    let _ = avx2;
    row_by_block(ty, raw, w, s, m)
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    use super::*;

    /// Negate 32 weights where a bit is set: bits 0-7 of each 64-bit lane are for 8 weights.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn negate(w: *mut i8, bits: __m256i) {
        unsafe {
            let spread = _mm256_shuffle_epi8(bits, _mm256_setr_epi64x(0, 0x0808080808080808, 0, 0x0808080808080808));
            let bit = _mm256_set1_epi64x(0x8040201008040201u64 as i64);
            let neg = _mm256_cmpeq_epi8(_mm256_and_si256(spread, bit), bit);
            let v = _mm256_loadu_si256(w as *const __m256i);
            _mm256_storeu_si256(w as *mut __m256i, _mm256_sub_epi8(_mm256_xor_si256(v, neg), neg));
        }
    }

    /// Bit 7 of each lane from the 7 below it (see `even`).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn with_parity(v: __m256i) -> __m256i {
        let v = _mm256_and_si256(v, _mm256_set1_epi64x(127));
        let x = _mm256_and_si256(_mm256_xor_si256(v, _mm256_srli_epi64(v, 4)), _mm256_set1_epi64x(15));
        let h = -128i8;
        let odd = _mm256_broadcastsi128_si256(_mm_setr_epi8(0, h, h, 0, h, 0, 0, h, h, 0, 0, h, 0, h, h, 0));
        // the other bytes of a lane look up entry 0
        _mm256_or_si256(v, _mm256_shuffle_epi8(odd, x))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn signs_28(w: *mut i8, c: u32) {
        unsafe { negate(w, with_parity(_mm256_srlv_epi64(_mm256_set1_epi64x(c as i64), _mm256_setr_epi64x(0, 7, 14, 21)))) }
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn signs_32(w: *mut i8, c: u32) {
        unsafe { negate(w, _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(c as i32))) }
    }

    #[inline]
    #[target_feature(enable = "avx2,f16c")]
    unsafe fn scales(s: *mut f32, v: &[i8; 16], d: u16, by: f32) {
        unsafe {
            let d = _mm256_set1_ps(_mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(d as i32))) * by);
            for i in [0, 8] {
                let v = _mm256_cvtepi8_epi32(_mm_loadl_epi64(v.as_ptr().add(i) as *const __m128i));
                _mm256_storeu_ps(s.add(i), _mm256_mul_ps(_mm256_cvtepi32_ps(v), d));
            }
        }
    }

    /// The caller checked the lengths. Grid formats: first the magnitudes of the whole row, then the
    /// signs, so that those do not load what has only just been stored in smaller pieces.
    #[target_feature(enable = "avx2,fma,f16c,popcnt")]
    pub unsafe fn decode_row(ty: u32, raw: &[u8], w: &mut [i8], s: &mut [f32], m: &mut [f32]) {
        let t = tables();
        let (wp, sp) = (w.as_mut_ptr(), s.as_mut_ptr());
        let mut sc = [0i8; 16];
        let odd = |v: u32| (2 * v + 1) as i8;
        let nib = |v: u8| (v as u32 & 15, v as u32 >> 4);
        let le16 = |b: &[u8], i: usize| u16::from_le_bytes([b[i], b[i + 1]]);
        let le32 = |b: &[u8], i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        // sized as their indices, so that looking up needs no check
        let (g2xxs, g2xs, g2s): (&[u64; 256], &[u64; 512], &[u64; 1024]) =
            (t.g2xxs[..].try_into().unwrap(), t.g2xs[..].try_into().unwrap(), t.g2s[..].try_into().unwrap());
        let (g3xxs, g3s): (&[u32; 256], &[u32; 512]) = (t.g3xxs[..].try_into().unwrap(), t.g3s[..].try_into().unwrap());
        unsafe {
            let put8 = |at: usize, v: u64| (wp.add(at) as *mut u64).write_unaligned(v);
            let put4 = |at: usize, v: u32| (wp.add(at) as *mut u32).write_unaligned(v);
            // for group i of 32 weights, at o
            macro_rules! grid {
                ($n:literal, |$b:ident, $o:ident, $i:ident| $body:expr) => {
                    for (n, b) in raw.chunks_exact($n).enumerate() {
                        let $b: &[u8; $n] = b.try_into().unwrap();
                        for $i in 0..8 {
                            let $o = n * 256 + $i * 32;
                            $body
                        }
                    }
                };
            }
            // the same; the result is the group's two scales
            macro_rules! finish {
                ($n:literal, $by:expr, |$b:ident, $o:ident, $i:ident| $body:expr) => {
                    for (n, b) in raw.chunks_exact($n).enumerate() {
                        let $b: &[u8; $n] = b.try_into().unwrap();
                        for $i in 0..8 {
                            let $o = wp.add(n * 256 + $i * 32);
                            let (s0, s1) = $body;
                            (sc[2 * $i], sc[2 * $i + 1]) = (odd(s0), odd(s1));
                        }
                        scales(sp.add(n * 16), &sc, le16($b, 0), $by);
                    }
                };
            }
            match ty {
                IQ2_XXS => {
                    grid!(66, |b, o, i| for l in 0..4 {
                        put8(o + 8 * l, g2xxs[b[2 + 8 * i + l] as usize]);
                    });
                    finish!(66, 0.125, |b, o, i| {
                        let c = le32(b, 6 + 8 * i);
                        signs_28(o, c);
                        (c >> 28, c >> 28)
                    });
                }
                IQ2_XS => {
                    grid!(74, |b, o, i| for l in 0..4 {
                        put8(o + 8 * l, g2xs[le16(b, 2 + 2 * (4 * i + l)) as usize & 511]);
                    });
                    finish!(74, 0.125, |b, o, i| {
                        let q = _mm256_cvtepu16_epi64(_mm_loadl_epi64(b[2 + 8 * i..][..8].as_ptr() as *const __m128i));
                        negate(o, with_parity(_mm256_srli_epi64(q, 9)));
                        nib(b[66 + i])
                    });
                }
                IQ2_S => {
                    grid!(82, |b, o, i| {
                        let hi = b[66 + i] as usize;
                        for l in 0..4 {
                            put8(o + 8 * l, g2s[b[2 + 4 * i + l] as usize | (hi << (8 - 2 * l) & 0x300)]);
                        }
                    });
                    finish!(82, 0.125, |b, o, i| {
                        signs_32(o, le32(b, 34 + 4 * i));
                        nib(b[74 + i])
                    });
                }
                IQ3_XXS => {
                    grid!(98, |b, o, i| for l in 0..8 {
                        put4(o + 4 * l, g3xxs[b[2 + 8 * i + l] as usize]);
                    });
                    finish!(98, 0.25, |b, o, i| {
                        let c = le32(b, 66 + 4 * i);
                        signs_28(o, c);
                        (c >> 28, c >> 28)
                    });
                }
                IQ3_S => {
                    grid!(110, |b, o, i| {
                        let hi = b[66 + i] as usize;
                        for l in 0..8 {
                            put4(o + 4 * l, g3s[b[2 + 8 * i + l] as usize | (hi << (8 - l) & 256)]);
                        }
                    });
                    finish!(110, 1.0, |b, o, i| {
                        signs_32(o, le32(b, 74 + 4 * i));
                        let v = b[106 + i / 2] as u32 >> (4 * (i & 1)) & 15;
                        (v, v)
                    });
                }
                IQ1_S => finish!(50, 0.125, |b, o, i| {
                    let q = le16(b, 34 + 2 * i) as usize;
                    let g = &t.g1[q >> 15];
                    for l in 0..4 {
                        let v = g[b[2 + 4 * i + l] as usize | (q >> (3 * l) & 7) << 8];
                        (o.add(8 * l) as *mut u64).write_unaligned(v);
                    }
                    let v = (q >> 12 & 7) as u32;
                    (v, v)
                }),
                _ => row_by_block(ty, raw, w, s, m),
            }
        }
    }
}

/// The values w can take in a format that is not `wide`, as a 16-entry table, and the table entry of each value.
pub fn codes(ty: u32) -> (&'static [i8; 16], &'static [u8; 256]) {
    static T: [OnceLock<([i8; 16], [u8; 256])>; 7] = [const { OnceLock::new() }; 7];
    let (which, vals): (usize, &[i8]) = match ty {
        IQ1_S | IQ1_M => (0, &[-9, -7, -1, 1, 7, 9]),
        IQ2_XXS | IQ2_XS | IQ2_S => (1, &[8, 25, 43, -8, -25, -43]),
        IQ3_S => (4, &[1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15]),
        IQ3_XXS => (2, &[4, 12, 20, 28, 36, 44, 52, 62, -4, -12, -20, -28, -36, -44, -52, -62]),
        IQ4_XS => (5, &IQ4_NL_VALUES),
        Q4_K => (6, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        // Q3_K, Q2_K
        _ => (3, &[-4, -3, -2, -1, 0, 1, 2, 3]),
    };
    let (lut, inv) = T[which].get_or_init(|| {
        let (mut lut, mut inv) = ([0i8; 16], [0u8; 256]);
        for (i, &v) in vals.iter().enumerate() {
            lut[i] = v;
            inv[v as u8 as usize] = i as u8;
        }
        (lut, inv)
    });
    (lut, inv)
}

/// Bytes of a [rows][k] matrix in the form the GPU kernels read.
pub fn slot_len(ty: u32, rows: usize, k: usize) -> usize {
    let n = rows * k;
    if ty == GGML_MXFP4 {
        return n / 2 + n / 32;
    }
    (if wide(ty) { n } else { n / 2 }) + n / 16 + n / 64 + if has_min(ty) { n / 16 + n / 64 } else { 0 }
}

/// Convert a low-bit matrix to the form of the `*_lut` kernels (`*_wide` ones: an int8 per weight, in order):
/// [4-bit codes: of 32 weights the first 16 in the low nibbles][s: i8][d: f32] (`has_min`: [m: u8][dm: f32])
pub fn to_slot(ty: u32, raw: &[u8], rows: usize, k: usize, out: &mut [u8]) {
    use rayon::prelude::*;
    let (be, bb) = crate::gguf::type_block(ty).unwrap();
    let (nb, bb) = (k / be as usize, bb as usize);
    assert!(be == 256 && k % 256 == 0 && raw.len() == rows * nb * bb && out.len() == slot_len(ty, rows, k));
    let n = rows * k;
    let (qk, inv) = if wide(ty) { (k, None) } else { (k / 2, Some(codes(ty).1)) };
    let (q, rest) = out.split_at_mut(rows * qk);
    let (s, rest) = rest.split_at_mut(n / 16);
    let (d, rest) = rest.split_at_mut(n / 64);
    // keep the parallel zip uniform: without offsets, every row gets empty slices
    let (mk, dk) = if has_min(ty) { (n / 16, n / 64) } else { (0, 0) };
    let (m, dm) = rest.split_at_mut(mk);
    assert_eq!(dm.len(), dk);
    let mut m_rows: Vec<&mut [u8]> = if mk > 0 { m.chunks_mut(k / 16).collect() } else { (0..rows).map(|_| Default::default()).collect() };
    let mut dm_rows: Vec<&mut [u8]> = if dk > 0 { dm.chunks_mut(k / 64).collect() } else { (0..rows).map(|_| Default::default()).collect() };
    q.par_chunks_mut(qk)
        .zip(s.par_chunks_mut(k / 16))
        .zip(d.par_chunks_mut(k / 64))
        .zip(m_rows.par_iter_mut().zip(dm_rows.par_iter_mut()))
        .enumerate()
        .with_min_len(16)
        .for_each_init(Block::new, |blk, (r, (((q, s), d), (m, dm)))| {
            for b in 0..nb {
                decode(ty, &raw[(r * nb + b) * bb..][..bb], blk);
                if let Some(inv) = inv {
                    for (c, o) in blk.w.chunks_exact(32).zip(q[b * 128..].chunks_exact_mut(16)) {
                        for j in 0..16 {
                            o[j] = inv[c[j] as u8 as usize] | inv[c[j + 16] as u8 as usize] << 4;
                        }
                    }
                } else {
                    q[b * 256..][..256].copy_from_slice(bytemuck::cast_slice(&blk.w));
                }
                s[b * 16..b * 16 + 16].copy_from_slice(bytemuck::cast_slice(&blk.s));
                d[b * 4..b * 4 + 4].copy_from_slice(&blk.d.to_le_bytes());
                if !m.is_empty() {
                    m[b * 16..b * 16 + 16].copy_from_slice(&blk.m);
                    dm[b * 4..b * 4 + 4].copy_from_slice(&blk.dm.to_le_bytes());
                }
            }
        });
}

const IQ4_NL_VALUES: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

/// Can `dequantize` read it?
pub fn can_dequantize(ty: u32) -> bool {
    is_low_bit(ty) || matches!(ty, GGML_F32 | GGML_F16 | GGML_BF16 | GGML_Q8_0 | GGML_MXFP4 | IQ4_NL)
}

/// Whole blocks to f32.
pub fn dequantize(ty: u32, b: &[u8], out: &mut [f32]) {
    match ty {
        GGML_F32 => b.chunks_exact(4).zip(out).for_each(|(c, o)| *o = f32::from_le_bytes(c.try_into().unwrap())),
        GGML_F16 => b.chunks_exact(2).zip(out).for_each(|(c, o)| *o = half_at(c, 0)),
        GGML_BF16 => b.chunks_exact(2).zip(out).for_each(|(c, o)| *o = half::bf16::from_le_bytes([c[0], c[1]]).to_f32()),
        GGML_Q8_0 => {
            for (c, o) in b.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
                let d = half_at(c, 0);
                c[2..].iter().zip(o).for_each(|(&q, o)| *o = q as i8 as f32 * d);
            }
        }
        IQ4_NL => {
            for (c, o) in b.chunks_exact(18).zip(out.chunks_exact_mut(32)) {
                let d = half_at(c, 0);
                for j in 0..16 {
                    o[j] = d * IQ4_NL_VALUES[(c[2 + j] & 15) as usize] as f32;
                    o[j + 16] = d * IQ4_NL_VALUES[(c[2 + j] >> 4) as usize] as f32;
                }
            }
        }
        GGML_MXFP4 => {
            const V: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
            for (c, o) in b.chunks_exact(17).zip(out.chunks_exact_mut(32)) {
                let d = 2f32.powi(c[0] as i32 - 127);
                for j in 0..16 {
                    o[j] = d * V[(c[1 + j] & 15) as usize];
                    o[j + 16] = d * V[(c[1 + j] >> 4) as usize];
                }
            }
        }
        _ => {
            let mut blk = Block::new();
            let bb = crate::gguf::type_block(ty).unwrap().1 as usize;
            for (c, o) in b.chunks_exact(bb).zip(out.chunks_exact_mut(256)) {
                decode(ty, c, &mut blk);
                for (i, o) in o.iter_mut().enumerate() {
                    *o = blk.w[i] as f32 * (blk.s[i / 16] as f32 * blk.d);
                    if has_min(ty) {
                        *o -= blk.m[i / 16] as f32 * blk.dm;
                    }
                }
            }
        }
    }
}

/// f32 to Q8_0 blocks (as ggml: the largest magnitude of 32 becomes 127).
pub fn quantize_q8_0(x: &[f32], out: &mut [u8]) {
    for (x, o) in x.chunks_exact(32).zip(out.chunks_exact_mut(34)) {
        let d = x.iter().fold(0f32, |a, v| a.max(v.abs())) / 127.0;
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        o[..2].copy_from_slice(&f16::from_f32(d).to_le_bytes());
        x.iter().zip(&mut o[2..]).for_each(|(v, o)| *o = (v * inv).round() as i8 as u8);
    }
}

/// Random blocks of a low-bit format with sane scales, for tests.
pub fn random_blocks(ty: u32, n: usize, mut byte: impl FnMut() -> u8) -> Vec<u8> {
    let bb = crate::gguf::type_block(ty).unwrap().1 as usize;
    let mut out: Vec<u8> = (0..n * bb).map(|_| byte()).collect();
    let at = match ty {
        Q2_K => 80,
        Q3_K => 108,
        Q6_K => 208,
        _ => 0,
    };
    for (i, b) in out.chunks_exact_mut(bb).enumerate() {
        let d = f16::from_f32(0.001 * (1 + i % 7) as f32);
        if ty == IQ1_M {
            // spread over the top nibbles of four words
            for j in 0..4 {
                b[49 + 2 * j] = b[49 + 2 * j] & 15 | ((d.to_bits() >> (4 * j) & 15) as u8) << 4;
            }
            continue;
        }
        b[at..at + 2].copy_from_slice(&d.to_le_bytes());
        if has_min(ty) {
            b[at + 2..at + 4].copy_from_slice(&f16::from_f32(0.0015).to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random blocks and their values from a separate transcription of ggml's decoders, which
    /// was checked against real weights.
    fn vectors() -> Vec<(u32, &'static [u8], Vec<f32>)> {
        let mut b: &'static [u8] = include_bytes!("../data/quant_vectors.bin");
        let mut out = Vec::new();
        while !b.is_empty() {
            let ty = u32::from_le_bytes(b[..4].try_into().unwrap());
            let n = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
            let y = b[8 + n..8 + n + 2048].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            out.push((ty, &b[8..8 + n], y));
            b = &b[8 + n + 2048..];
        }
        out
    }

    #[test]
    fn decoders_match_reference() {
        let v = vectors();
        assert_eq!(v.len(), 14);
        for (ty, raw, want) in v {
            let mut got = vec![0f32; 512];
            dequantize(ty, raw, &mut got);
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                // Q2_K and Q5_K subtract: judge those against the size of the terms
                assert!((g - w).abs() <= 1e-6 * (w.abs() + 0.05), "type {ty} at {i}: {g} vs {w}");
            }
        }
    }

    #[test]
    fn rows_match_blocks() {
        let mut z = 11u64;
        let mut byte = || {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            (z >> 9) as u8
        };
        let k = 256 * 64;
        for ty in LOW_BIT {
            let raw = random_blocks(ty, k / 256, &mut byte);
            let run = |avx2| {
                let (mut w, mut s) = (vec![0i8; k], vec![0f32; k / 16]);
                let mut m = vec![0f32; if has_min(ty) { k / 16 } else { 0 }];
                decode_row(avx2, ty, &raw, &mut w, &mut s, &mut m);
                (w, s, m)
            };
            assert!(run(false) == run(crate::cpu_moe::has_simd()), "type {ty}");
        }
    }

    #[test]
    fn slots_hold_the_same_weights() {
        let mut z = 5u64;
        let mut byte = || {
            z ^= z << 13;
            z ^= z >> 7;
            z ^= z << 17;
            (z >> 9) as u8
        };
        let (rows, k) = (3usize, 512usize);
        for ty in LOW_BIT {
            let raw = random_blocks(ty, rows * k / 256, &mut byte);
            let mut want = vec![0f32; rows * k];
            dequantize(ty, &raw, &mut want);
            let mut slot = vec![0u8; slot_len(ty, rows, k)];
            to_slot(ty, &raw, rows, k, &mut slot);
            let (n, lut) = (rows * k, codes(ty).0);
            let qn = if wide(ty) { n } else { n / 2 };
            let f = |o: usize| f32::from_le_bytes(slot[o..o + 4].try_into().unwrap());
            for (i, w) in want.iter().enumerate() {
                let q = slot[i / 32 * 16 + i % 16] >> (if i % 32 < 16 { 0 } else { 4 }) & 15;
                let v = if wide(ty) { slot[i] as i8 } else { lut[q as usize] };
                let mut g = v as f32 * (slot[qn + i / 16] as i8 as f32 * f(qn + n / 16 + i / 256 * 4));
                if has_min(ty) {
                    let o = qn + n / 16 + n / 64;
                    g -= slot[o + i / 16] as f32 * f(o + n / 16 + i / 256 * 4);
                }
                assert_eq!(g, *w, "type {ty} at {i}");
            }
        }
    }

    #[test]
    fn mxfp4_values() {
        let mut b = [0u8; 17];
        (b[0], b[1], b[2], b[16]) = (126, 0x71, 0xf2, 0x80);
        let mut y = [0f32; 32];
        dequantize(GGML_MXFP4, &b, &mut y);
        assert_eq!((y[0], y[16], y[1], y[17], y[15], y[31]), (0.25, 3.0, 0.5, -3.0, 0.0, 0.0));
    }

    #[test]
    fn q8_0_round_trip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut b = vec![0u8; 68];
        quantize_q8_0(&x, &mut b);
        let mut y = vec![0f32; 64];
        dequantize(GGML_Q8_0, &b, &mut y);
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 0.005));
    }
}
