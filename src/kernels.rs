//! WGSL compute kernels. Every kernel is its own module with entry point `main` and reads its
//! scalar parameters from a 64-byte immediate block `P` (p0..p15, floats passed as bits).

const PRELUDE: &str = r#"
struct Params {
  p0: u32, p1: u32, p2: u32, p3: u32, p4: u32, p5: u32, p6: u32, p7: u32,
  p8: u32, p9: u32, p10: u32, p11: u32, p12: u32, p13: u32, p14: u32, p15: u32,
}
var<immediate> P: Params;
const NEG_INF: f32 = -3.0e38;

fn fbits(u: u32) -> f32 { return bitcast<f32>(u); }

// 2^ceil(log2(x)) for x > 0 (exact powers of two map to themselves)
fn pow2_ceil(x: f32) -> f32 {
  let b = bitcast<u32>(x);
  var e = i32((b >> 23u) & 0xffu) - 127;
  if ((b & 0x7fffffu) != 0u) { e = e + 1; }
  return ldexp(1.0, e);
}

// round to the nearest float8 e4m3fn value (ties to even), saturating at 448
fn to_e4m3(v: f32) -> f32 {
  let a = min(abs(v), 448.0);
  var q: f32;
  if (a < 0.015625) {
    q = round(a * 512.0) / 512.0;
  } else {
    let e = i32((bitcast<u32>(a) >> 23u) & 0xffu) - 127;
    let step = ldexp(1.0, e - 3);
    q = min(round(a / step) * step, 448.0);
  }
  return select(q, -q, v < 0.0);
}

// round to the nearest float4 e2m1 value (ties to even), saturating at 6
fn to_e2m1(v: f32) -> f32 {
  let a = min(abs(v), 6.0);
  var q: f32;
  if (a < 2.0) { q = round(a * 2.0) * 0.5; }
  else if (a < 4.0) { q = round(a); }
  else { q = round(a * 0.5) * 2.0; }
  return select(q, -q, v < 0.0);
}
"#;

/// Q8_0 matmul, weights repacked as [int8 x N*K][f16 scale x N*K/32].
/// Y[m, row_off + n] (+)= sum_k W[n, k] X[m, xg(n) + k]
/// p0=N p1=K p2=M p3=x_stride p4=y_stride p5=x_off p6=y_off p7=scale_word_off
/// p8=group_rows(0: none) p9=accumulate p10=row_off(Y column offset)
const MM_Q8: &str = r#"
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<array<f32, TM>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let N = P.p0; let K = P.p1; let M = P.p2;
  let lane = li & 31u;
  let row = wid.x * 8u + (li >> 5u);
  let t0 = wid.y * TM;
  let nb = K / 32u;
  var acc: array<f32, TM>;
  for (var t = 0u; t < TM; t++) { acc[t] = 0.0; }
  if (row < N) {
    var xg = 0u;
    if (P.p8 != 0u) { xg = (row / P.p8) * K; }
    for (var b = lane; b < nb; b += 32u) {
      let s = row * nb + b;
      let sc = unpack2x16float(W[P.p7 + (s >> 1u)])[s & 1u];
      let wb = s * 8u;
      var w: array<vec4<f32>, 8>;
      for (var i = 0u; i < 8u; i++) { w[i] = vec4<f32>(unpack4xI8(W[wb + i])); }
      for (var t = 0u; t < TM; t++) {
        if (t0 + t < M) {
          let xb = (P.p5 + (t0 + t) * P.p3 + xg + b * 32u) >> 2u;
          var d = 0.0;
          for (var i = 0u; i < 8u; i++) { d += dot(w[i], X[xb + i]); }
          acc[t] += d * sc;
        }
      }
    }
  }
  for (var t = 0u; t < TM; t++) { red[li][t] = acc[t]; }
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) {
      for (var t = 0u; t < TM; t++) { red[li][t] += red[li + s][t]; }
    }
    workgroupBarrier();
  }
  if (lane == 0u && row < N) {
    for (var t = 0u; t < TM; t++) {
      if (t0 + t < M) {
        let yi = P.p6 + (t0 + t) * P.p4 + P.p10 + row;
        if (P.p9 != 0u) { Y[yi] += red[li][t]; } else { Y[yi] = red[li][t]; }
      }
    }
  }
}
"#;

/// F32-weight matmul (router, small): same contract as MM_Q8 without scales/groups.
const MM_F32: &str = r#"
@group(0) @binding(0) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<array<f32, TM>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let N = P.p0; let K = P.p1; let M = P.p2;
  let lane = li & 31u;
  let row = wid.x * 8u + (li >> 5u);
  let t0 = wid.y * TM;
  let nv = K / 4u;
  var acc: array<f32, TM>;
  for (var t = 0u; t < TM; t++) { acc[t] = 0.0; }
  if (row < N) {
    for (var v = lane; v < nv; v += 32u) {
      let w = W[row * nv + v];
      for (var t = 0u; t < TM; t++) {
        if (t0 + t < M) { acc[t] += dot(w, X[((P.p5 + (t0 + t) * P.p3) >> 2u) + v]); }
      }
    }
  }
  for (var t = 0u; t < TM; t++) { red[li][t] = acc[t]; }
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { for (var t = 0u; t < TM; t++) { red[li][t] += red[li + s][t]; } }
    workgroupBarrier();
  }
  if (lane == 0u && row < N) {
    for (var t = 0u; t < TM; t++) {
      if (t0 + t < M) { Y[P.p6 + (t0 + t) * P.p4 + row] = red[li][t]; }
    }
  }
}
"#;

/// MXFP4 expert helpers. A slot holds gate | up | down, each [qs N*nb*4 words][e8m0 N*nb bytes].
const MXFP4_COMMON: &str = r#"
var<private> KV4: array<f32, 16> = array<f32, 16>(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                                 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0);
fn e8m0(S: u32) -> f32 { return ldexp(1.0, i32(S) - 127); }
fn lo4(w: u32) -> vec4<f32> {
  return vec4<f32>(KV4[w & 15u], KV4[(w >> 8u) & 15u], KV4[(w >> 16u) & 15u], KV4[(w >> 24u) & 15u]);
}
fn hi4(w: u32) -> vec4<f32> {
  return vec4<f32>(KV4[(w >> 4u) & 15u], KV4[(w >> 12u) & 15u], KV4[(w >> 20u) & 15u], KV4[(w >> 28u) & 15u]);
}
"#;

/// Fused expert gate+up for a list of routed rows.
/// p0=N(inter) p1=K(dim) p2=count p3=entry_off p4=limit(f32) p5=mat_words(one matrix) p6=x_stride
/// route entry e: R[3e]=token R[3e+1]=output row (token*n_active+k) R[3e+2]=weight bits. H[e][n]
const MOE_GATE_UP: &str = r#"
@group(0) @binding(0) var<storage, read> S: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> R: array<u32>;
@group(0) @binding(3) var<storage, read_write> H: array<f32>;
var<workgroup> red: array<array<vec2<f32>, TM>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let N = P.p0; let K = P.p1; let cnt = P.p2;
  let lane = li & 31u;
  let row = wid.x * 8u + (li >> 5u);
  let t0 = wid.y * TM;
  let nb = K / 32u;
  let qw = N * nb * 4u;          // qs words per matrix
  let up = P.p5;                  // word offset of the up matrix
  var acc: array<vec2<f32>, TM>;
  for (var t = 0u; t < TM; t++) { acc[t] = vec2<f32>(0.0); }
  if (row < N) {
    for (var b = lane; b < nb; b += 32u) {
      let s = row * nb + b;
      let sg = e8m0((S[qw + (s >> 2u)] >> ((s & 3u) * 8u)) & 0xffu);
      let su = e8m0((S[up + qw + (s >> 2u)] >> ((s & 3u) * 8u)) & 0xffu);
      var g: array<vec4<f32>, 8>;
      var u: array<vec4<f32>, 8>;
      for (var i = 0u; i < 4u; i++) {
        let wg = S[s * 4u + i];
        let wu = S[up + s * 4u + i];
        g[i] = lo4(wg); g[i + 4u] = hi4(wg);
        u[i] = lo4(wu); u[i + 4u] = hi4(wu);
      }
      for (var t = 0u; t < TM; t++) {
        if (t0 + t < cnt) {
          let tok = R[(P.p3 + t0 + t) * 3u];
          let xb = (tok * P.p6 + b * 32u) >> 2u;
          var dg = 0.0; var du = 0.0;
          for (var i = 0u; i < 8u; i++) { let x = X[xb + i]; dg += dot(g[i], x); du += dot(u[i], x); }
          acc[t] += vec2<f32>(dg * sg, du * su);
        }
      }
    }
  }
  for (var t = 0u; t < TM; t++) { red[li][t] = acc[t]; }
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { for (var t = 0u; t < TM; t++) { red[li][t] += red[li + s][t]; } }
    workgroupBarrier();
  }
  if (lane == 0u && row < N) {
    let lim = fbits(P.p4);
    for (var t = 0u; t < TM; t++) {
      if (t0 + t < cnt) {
        let e = P.p3 + t0 + t;
        let w = fbits(R[e * 3u + 2u]);
        let gv = min(red[li][t].x, lim);
        let uv = clamp(red[li][t].y, -lim, lim);
        H[e * N + row] = gv / (1.0 + exp(-gv)) * uv * w;
      }
    }
  }
}
"#;

/// Expert down projection: O[R[3e+1]][n] = sum_k Wd[n,k] H[e][k]
/// p0=N(dim) p1=K(inter) p2=count p3=entry_off p5=down word offset
const MOE_DOWN: &str = r#"
@group(0) @binding(0) var<storage, read> S: array<u32>;
@group(0) @binding(1) var<storage, read> Hh: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> R: array<u32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
var<workgroup> red: array<array<f32, TM>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let N = P.p0; let K = P.p1; let cnt = P.p2;
  let lane = li & 31u;
  let row = wid.x * 8u + (li >> 5u);
  let t0 = wid.y * TM;
  let nb = K / 32u;
  let base = P.p5;
  let qw = N * nb * 4u;
  var acc: array<f32, TM>;
  for (var t = 0u; t < TM; t++) { acc[t] = 0.0; }
  if (row < N) {
    for (var b = lane; b < nb; b += 32u) {
      let s = row * nb + b;
      let sc = e8m0((S[base + qw + (s >> 2u)] >> ((s & 3u) * 8u)) & 0xffu);
      var w: array<vec4<f32>, 8>;
      for (var i = 0u; i < 4u; i++) {
        let ww = S[base + s * 4u + i];
        w[i] = lo4(ww); w[i + 4u] = hi4(ww);
      }
      for (var t = 0u; t < TM; t++) {
        if (t0 + t < cnt) {
          let xb = ((P.p3 + t0 + t) * K + b * 32u) >> 2u;
          var d = 0.0;
          for (var i = 0u; i < 8u; i++) { d += dot(w[i], Hh[xb + i]); }
          acc[t] += d * sc;
        }
      }
    }
  }
  for (var t = 0u; t < TM; t++) { red[li][t] = acc[t]; }
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { for (var t = 0u; t < TM; t++) { red[li][t] += red[li + s][t]; } }
    workgroupBarrier();
  }
  if (lane == 0u && row < N) {
    for (var t = 0u; t < TM; t++) {
      if (t0 + t < cnt) {
        let orow = R[(P.p3 + t0 + t) * 3u + 1u];
        O[orow * N + row] = red[li][t];
      }
    }
  }
}
"#;

/// Expert matrices that are not MXFP4, as laid out by `quants::to_slot`: 4-bit codes into the 16 int8
/// values of p8..p11 (`wide` ones: int8 values, see `lut`), an int8 scale per 16 weights and a float per 256 (p7 != 0: also
/// a uint8 offset per 16 and its float per 256). K is a multiple of 256.
const LUT_COMMON: &str = r#"
var<private> KV4: array<f32, 16>;
fn init_lut() {
  let w = array<u32, 4>(P.p8, P.p9, P.p10, P.p11);
  for (var j = 0u; j < 4u; j++) {
    let v = vec4<f32>(unpack4xI8(w[j]));
    for (var i = 0u; i < 4u; i++) { KV4[j * 4u + i] = v[i]; }
  }
}
fn lo4(w: u32) -> vec4<f32> {
  return vec4<f32>(KV4[w & 15u], KV4[(w >> 8u) & 15u], KV4[(w >> 16u) & 15u], KV4[(w >> 24u) & 15u]);
}
fn hi4(w: u32) -> vec4<f32> {
  return vec4<f32>(KV4[(w >> 4u) & 15u], KV4[(w >> 12u) & 15u], KV4[(w >> 20u) & 15u], KV4[(w >> 28u) & 15u]);
}
// scales of the two halves of block b (32 weights) in a row of the matrix at word `m`
fn scales(m: u32, row: u32, b: u32) -> vec2<f32> {
  let n = P.p0 * P.p1;
  let sb = m + n / WPW;
  let i = row * (P.p1 / 16u) + b * 2u;
  let v = unpack4xI8(S[sb + (i >> 2u)]);
  let d = fbits(S[sb + n / 64u + row * (P.p1 / 256u) + (b >> 3u)]);
  return vec2<f32>(f32(v[i & 3u]), f32(v[(i & 3u) + 1u])) * d;
}
fn mins(m: u32, row: u32, b: u32) -> vec2<f32> {
  if (P.p7 == 0u) { return vec2<f32>(0.0); }
  let n = P.p0 * P.p1;
  let mb = m + n / WPW + n / 64u + n / 256u;
  let i = row * (P.p1 / 16u) + b * 2u;
  let v = unpack4xU8(S[mb + (i >> 2u)]);
  let d = fbits(S[mb + n / 64u + row * (P.p1 / 256u) + (b >> 3u)]);
  return vec2<f32>(f32(v[i & 3u]), f32(v[(i & 3u) + 1u])) * d;
}
"#;

/// MOE_GATE_UP for low-bit matrices. Also p7=offsets p8..p11=values
const MOE_GATE_UP_LUT: &str = r#"
@group(0) @binding(0) var<storage, read> S: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> R: array<u32>;
@group(0) @binding(3) var<storage, read_write> H: array<f32>;
var<workgroup> red: array<array<vec2<f32>, TM>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let N = P.p0; let K = P.p1; let cnt = P.p2;
  let lane = li & 31u;
  let row = wid.x * 8u + (li >> 5u);
  let t0 = wid.y * TM;
  let nb = K / 32u;
  init_lut();
  let up = P.p5;                  // word offset of the up matrix
  var acc: array<vec2<f32>, TM>;
  for (var t = 0u; t < TM; t++) { acc[t] = vec2<f32>(0.0); }
  if (row < N) {
    for (var b = lane; b < nb; b += 32u) {
      let s = row * nb + b;
      let sg = scales(0u, row, b);
      let su = scales(up, row, b);
      let mg = mins(0u, row, b);
      let mu = mins(up, row, b);
      var g: array<vec4<f32>, 8>;
      var u: array<vec4<f32>, 8>;
      for (var i = 0u; i < 4u; i++) {
        FETCH
      }
      for (var t = 0u; t < TM; t++) {
        if (t0 + t < cnt) {
          let tok = R[(P.p3 + t0 + t) * 3u];
          let xb = (tok * P.p6 + b * 32u) >> 2u;
          var dg = vec2<f32>(0.0); var du = vec2<f32>(0.0); var sx = vec2<f32>(0.0);
          for (var i = 0u; i < 4u; i++) {
            let x = X[xb + i]; let y = X[xb + i + 4u];
            dg += vec2<f32>(dot(g[i], x), dot(g[i + 4u], y));
            du += vec2<f32>(dot(u[i], x), dot(u[i + 4u], y));
            if (P.p7 != 0u) { sx += vec2<f32>(dot(vec4<f32>(1.0), x), dot(vec4<f32>(1.0), y)); }
          }
          acc[t] += vec2<f32>(dot(dg, sg) - dot(sx, mg), dot(du, su) - dot(sx, mu));
        }
      }
    }
  }
  for (var t = 0u; t < TM; t++) { red[li][t] = acc[t]; }
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { for (var t = 0u; t < TM; t++) { red[li][t] += red[li + s][t]; } }
    workgroupBarrier();
  }
  if (lane == 0u && row < N) {
    let lim = fbits(P.p4);
    for (var t = 0u; t < TM; t++) {
      if (t0 + t < cnt) {
        let e = P.p3 + t0 + t;
        let w = fbits(R[e * 3u + 2u]);
        let gv = min(red[li][t].x, lim);
        let uv = clamp(red[li][t].y, -lim, lim);
        H[e * N + row] = gv / (1.0 + exp(-gv)) * uv * w;
      }
    }
  }
}
"#;

/// MOE_DOWN for low-bit matrices. Also p7=offsets p8..p11=values
const MOE_DOWN_LUT: &str = r#"
@group(0) @binding(0) var<storage, read> S: array<u32>;
@group(0) @binding(1) var<storage, read> Hh: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> R: array<u32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
var<workgroup> red: array<array<f32, TM>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let N = P.p0; let K = P.p1; let cnt = P.p2;
  let lane = li & 31u;
  let row = wid.x * 8u + (li >> 5u);
  let t0 = wid.y * TM;
  let nb = K / 32u;
  init_lut();
  let base = P.p5;
  var acc: array<f32, TM>;
  for (var t = 0u; t < TM; t++) { acc[t] = 0.0; }
  if (row < N) {
    for (var b = lane; b < nb; b += 32u) {
      let s = row * nb + b;
      let sc = scales(base, row, b);
      let mn = mins(base, row, b);
      var w: array<vec4<f32>, 8>;
      for (var i = 0u; i < 4u; i++) {
        FETCH
      }
      for (var t = 0u; t < TM; t++) {
        if (t0 + t < cnt) {
          let xb = ((P.p3 + t0 + t) * K + b * 32u) >> 2u;
          var d = vec2<f32>(0.0); var sx = vec2<f32>(0.0);
          for (var i = 0u; i < 4u; i++) {
            let x = Hh[xb + i]; let y = Hh[xb + i + 4u];
            d += vec2<f32>(dot(w[i], x), dot(w[i + 4u], y));
            if (P.p7 != 0u) { sx += vec2<f32>(dot(vec4<f32>(1.0), x), dot(vec4<f32>(1.0), y)); }
          }
          acc[t] += dot(d, sc) - dot(sx, mn);
        }
      }
    }
  }
  for (var t = 0u; t < TM; t++) { red[li][t] = acc[t]; }
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { for (var t = 0u; t < TM; t++) { red[li][t] += red[li + s][t]; } }
    workgroupBarrier();
  }
  if (lane == 0u && row < N) {
    for (var t = 0u; t < TM; t++) {
      if (t0 + t < cnt) {
        let orow = R[(P.p3 + t0 + t) * 3u + 1u];
        O[orow * N + row] = red[li][t];
      }
    }
  }
}
"#;

/// RMSNorm over rows. p0=D p1=in_off p2=out_off p3=in_stride p4=out_stride p5=eps p6=has_weight
const RMSNORM: &str = r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32) {
  let D = P.p0;
  let xi = P.p1 + wid.x * P.p3;
  let yi = P.p2 + wid.x * P.p4;
  var s = 0.0;
  for (var i = li; i < D; i += 256u) { let v = X[xi + i]; s += v * v; }
  s = subgroupAdd(s);
  if (sl == 0u) { red[li / ss] = s; }
  workgroupBarrier();
  var tot = 0.0;
  for (var i = 0u; i < 256u / ss; i++) { tot += red[i]; }
  let r = inverseSqrt(tot / f32(D) + fbits(P.p5));
  for (var i = li; i < D; i += 256u) {
    var v = X[xi + i] * r;
    if (P.p6 != 0u) { v = v * Wt[i]; }
    Y[yi + i] = v;
  }
}
"#;

/// RoPE on the trailing rope_dim of each head, adjacent pairs, in place.
/// p0=n_rows(tokens*heads) p1=heads p2=head_stride p3=rope_off p4=pos_base p5=pos_step p6=inverse
/// p7=table_off(floats) p8=x_off p9=half(rope_dim/2)
const ROPE: &str = r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> T: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let half = P.p9;
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= P.p0 * half) { return; }
  let row = i / half; let k = i % half;
  let tok = row / P.p1;
  let pos = P.p4 + tok * P.p5;
  let c = T[P.p7 + (pos * half + k) * 2u];
  var s = T[P.p7 + (pos * half + k) * 2u + 1u];
  if (P.p6 != 0u) { s = -s; }
  let o = P.p8 + row * P.p2 + P.p3 + 2u * k;
  let a = X[o]; let b = X[o + 1u];
  X[o] = a * c - b * s;
  X[o + 1u] = a * s + b * c;
}
"#;

/// Quantize-dequantize simulation, in place. One thread per block.
/// p0=n_blocks p1=block p2=mode(0 fp8 pow2 / 1 fp4 e8m0 / 2 fp4 e4m3-scale) p3=x_off
/// p4=row_len p5=row_stride (blocks never straddle rows)
const QUANT_SIM: &str = r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let bi = gid.x + gid.y * 65535u * 256u;
  if (bi >= P.p0) { return; }
  let bs = P.p1;
  let per_row = P.p4 / bs;
  let base = P.p3 + (bi / per_row) * P.p5 + (bi % per_row) * bs;
  var amax = 0.0;
  for (var i = 0u; i < bs; i++) { amax = max(amax, abs(X[base + i])); }
  let mode = P.p2;
  if (mode == 0u) {
    let s = pow2_ceil(max(amax, 1e-4) / 448.0);
    for (var i = 0u; i < bs; i++) { X[base + i] = to_e4m3(clamp(X[base + i] / s, -448.0, 448.0)) * s; }
  } else {
    var s: f32;
    if (mode == 1u) { s = pow2_ceil(max(amax, 6.0 * 1.1754944e-38) / 6.0); }
    else { s = to_e4m3(max(amax, 6.0 / 512.0) / 6.0); }
    for (var i = 0u; i < bs; i++) { X[base + i] = to_e2m1(clamp(X[base + i] / s, -6.0, 6.0)) * s; }
  }
}
"#;

/// vec4s of H that one workgroup of `hc_dot` reduces.
pub const HC_PART: u32 = 128;

/// First half of the hyper-connection mixes: the products of H with each row of Wf (and H with
/// itself), in parts so that many workgroups share the work.
/// H [M][hc*D] ; Wf [24][hc*D] ; out PART [M][parts][25], parts = hc*D/4/HC_PART rounded up
/// p0=hc*D p1=token offset into H; dispatch (parts, M, 1)
const HC_DOT: &str = r#"
@group(0) @binding(0) var<storage, read> H: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Wf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> PART: array<f32>;
var<workgroup> red: array<array<f32, 32>, 25>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32) {
  let L = P.p0 / 4u;
  let i = wid.x * 128u + li;
  var acc: array<f32, 25>;
  for (var o = 0u; o < 25u; o++) { acc[o] = 0.0; }
  if (i < L) {
    let x = H[(P.p1 + wid.y) * L + i];
    acc[24] = dot(x, x);
    for (var o = 0u; o < 24u; o++) { acc[o] = dot(x, Wf[o * L + i]); }
  }
  for (var o = 0u; o < 25u; o++) {
    let v = subgroupAdd(acc[o]);
    if (sl == 0u) { red[o][li / ss] = v; }
  }
  workgroupBarrier();
  if (li < 25u) {
    var t = 0.0;
    for (var s = 0u; s < 128u / ss; s++) { t += red[li][s]; }
    PART[(wid.y * nw.x + wid.x) * 25u + li] = t;
  }
}
"#;

/// Second half: hyper-connection mixes for one token per workgroup.
/// SB = scale[3] ++ base[24] ; out MIX [M][24] = pre4 post4 comb16
/// p0=hc*D p1=hc p2=iters p3=norm_eps p4=hc_eps p5=sb_off(floats in SB buffer) p6=mix_off(first token)
/// p7=parts
const HC_MIX: &str = r#"
@group(0) @binding(0) var<storage, read> PART: array<f32>;
@group(0) @binding(1) var<storage, read> SB: array<f32>;
@group(0) @binding(2) var<storage, read_write> MIX: array<f32>;
var<workgroup> tot: array<f32, 25>;

@compute @workgroup_size(25)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  var t = 0.0;
  for (var i = 0u; i < P.p7; i++) { t += PART[(wid.x * P.p7 + i) * 25u + li]; }
  tot[li] = t;
  workgroupBarrier();
  if (li == 0u) {
    let hc = P.p1;
    let eps = fbits(P.p4);
    let r = inverseSqrt(tot[24] / f32(P.p0) + fbits(P.p3));
    let so = P.p5;
    let mo = P.p6 + wid.x * 24u;
    var m: array<f32, 24>;
    for (var o = 0u; o < 24u; o++) { m[o] = tot[o] * r; }
    for (var j = 0u; j < 4u; j++) {
      MIX[mo + j] = 1.0 / (1.0 + exp(-(m[j] * SB[so] + SB[so + 3u + j]))) + eps;
      MIX[mo + 4u + j] = 2.0 / (1.0 + exp(-(m[4u + j] * SB[so + 1u] + SB[so + 3u + 4u + j])));
    }
    var c: array<f32, 16>;
    for (var j = 0u; j < 4u; j++) {
      var mx = NEG_INF;
      for (var k = 0u; k < 4u; k++) {
        c[j * 4u + k] = m[8u + j * 4u + k] * SB[so + 2u] + SB[so + 3u + 8u + j * 4u + k];
        mx = max(mx, c[j * 4u + k]);
      }
      var sum = 0.0;
      for (var k = 0u; k < 4u; k++) { c[j * 4u + k] = exp(c[j * 4u + k] - mx); sum += c[j * 4u + k]; }
      for (var k = 0u; k < 4u; k++) { c[j * 4u + k] = c[j * 4u + k] / sum + eps; }
    }
    for (var k = 0u; k < 4u; k++) {
      let cs = c[k] + c[4u + k] + c[8u + k] + c[12u + k] + eps;
      for (var j = 0u; j < 4u; j++) { c[j * 4u + k] /= cs; }
    }
    for (var it = 1u; it < P.p2; it++) {
      for (var j = 0u; j < 4u; j++) {
        let rs = c[j * 4u] + c[j * 4u + 1u] + c[j * 4u + 2u] + c[j * 4u + 3u] + eps;
        for (var k = 0u; k < 4u; k++) { c[j * 4u + k] /= rs; }
      }
      for (var k = 0u; k < 4u; k++) {
        let cs = c[k] + c[4u + k] + c[8u + k] + c[12u + k] + eps;
        for (var j = 0u; j < 4u; j++) { c[j * 4u + k] /= cs; }
      }
    }
    for (var o = 0u; o < 16u; o++) { MIX[mo + 8u + o] = c[o]; }
  }
}
"#;

/// x = rmsnorm(sum_j pre[j] * H[j]) * w, one token per workgroup.
/// p0=D p1=identity_pre p2=eps p3=token_off(first token) p4=out_off p5=mix_off
const HC_PRE_NORM: &str = r#"
@group(0) @binding(0) var<storage, read> H: array<f32>;
@group(0) @binding(1) var<storage, read> MIX: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32) {
  let D = P.p0;
  let t = P.p3 + wid.x;
  var pre = vec4<f32>(1.0, 0.0, 0.0, 0.0);
  if (P.p1 == 0u) {
    let mo = P.p5 + t * 24u;
    pre = vec4<f32>(MIX[mo], MIX[mo + 1u], MIX[mo + 2u], MIX[mo + 3u]);
  }
  let hb = t * 4u * D;
  let yb = P.p4 + wid.x * D;
  var s = 0.0;
  for (var i = li; i < D; i += 256u) {
    let v = pre.x * H[hb + i] + pre.y * H[hb + D + i] + pre.z * H[hb + 2u * D + i] + pre.w * H[hb + 3u * D + i];
    Y[yb + i] = v;
    s += v * v;
  }
  s = subgroupAdd(s);
  if (sl == 0u) { red[li / ss] = s; }
  workgroupBarrier();
  var tot = 0.0;
  for (var i = 0u; i < 256u / ss; i++) { tot += red[i]; }
  let r = inverseSqrt(tot / f32(D) + fbits(P.p2));
  for (var i = li; i < D; i += 256u) { Y[yb + i] = Y[yb + i] * r * Wt[i]; }
}
"#;

/// H[m][k] = post[k] * y + sum_j comb[j][k] H[m][j]; y = Y[m] (+ SH[m] + sum_s O6[m*ns+s])
/// p0=D p1=M p2=n_slots(0 = attention output only) p3=mix_off p4=token offset into H / MIX
const HC_POST: &str = r#"
@group(0) @binding(0) var<storage, read_write> H: array<f32>;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read> O6: array<f32>;
@group(0) @binding(3) var<storage, read> MIX: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let D = P.p0;
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= P.p1 * D) { return; }
  let m = i / D; let d = i % D;
  var y = Y[i];
  for (var s = 0u; s < P.p2; s++) { y += O6[(m * P.p2 + s) * D + d]; }
  let tk = P.p4 + m;
  let mo = P.p3 + tk * 24u;
  let hb = tk * 4u * D + d;
  let h = vec4<f32>(H[hb], H[hb + D], H[hb + 2u * D], H[hb + 3u * D]);
  for (var k = 0u; k < 4u; k++) {
    let c = vec4<f32>(MIX[mo + 8u + k], MIX[mo + 12u + k], MIX[mo + 16u + k], MIX[mo + 20u + k]);
    H[hb + k * D] = MIX[mo + 4u + k] * y + dot(c, h);
  }
}
"#;

/// Engram gate: per (token, copy) workgroup. EKV [M][(hc+1)*D] = keys(hc*D) ++ value(D); GW [hc][D]
/// p0=D p1=eps p2=token offset into H
const ENGRAM: &str = r#"
@group(0) @binding(0) var<storage, read_write> H: array<f32>;
@group(0) @binding(1) var<storage, read> EKV: array<f32>;
@group(0) @binding(2) var<storage, read> GW: array<f32>;
var<workgroup> red: array<vec3<f32>, 32>;
var<workgroup> gate: f32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32) {
  let D = P.p0;
  let m = wid.x; let j = wid.y;
  let hb = ((P.p2 + m) * 4u + j) * D;
  let kb = m * 5u * D + j * D;
  let vb = m * 5u * D + 4u * D;
  var a = vec3<f32>(0.0);
  for (var i = li; i < D; i += 256u) {
    let h = H[hb + i]; let k = EKV[kb + i];
    a += vec3<f32>(h * h, k * k, h * GW[j * D + i] * k);
  }
  a = subgroupAdd(a);
  if (sl == 0u) { red[li / ss] = a; }
  workgroupBarrier();
  if (li == 0u) {
    var t = vec3<f32>(0.0);
    for (var i = 0u; i < 256u / ss; i++) { t += red[i]; }
    let eps = fbits(P.p1);
    let rstd = inverseSqrt(t.x / f32(D) + eps) * inverseSqrt(t.y / f32(D) + eps);
    let dt = t.z * rstd * inverseSqrt(f32(D));
    let sq = sqrt(max(abs(dt), 1e-6));
    let z = select(sq, -sq, dt < 0.0);
    gate = 1.0 / (1.0 + exp(-z));
  }
  workgroupBarrier();
  let g = gate;
  for (var i = li; i < D; i += 256u) { H[hb + i] += g * EKV[vb + i]; }
}
"#;

/// SwiGLU with clamps for the shared expert: A = silu(min(G, l)) * clamp(U, -l, l)
/// p0=n p1=limit
const SWIGLU: &str = r#"
@group(0) @binding(0) var<storage, read> G: array<f32>;
@group(0) @binding(1) var<storage, read> U: array<f32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= P.p0) { return; }
  let l = fbits(P.p1);
  let g = min(G[i], l);
  A[i] = g / (1.0 + exp(-g)) * clamp(U[i], -l, l);
}
"#;

/// Copy rows into a ring: DST[(base + i) % R] = SRC[i]. p0=M p1=D p2=base p3=R p4=src_off p5=dst_off
const COPY_RING: &str = r#"
@group(0) @binding(0) var<storage, read> SRC: array<f32>;
@group(0) @binding(1) var<storage, read_write> DST: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= P.p0 * P.p1) { return; }
  let r = i / P.p1; let d = i % P.p1;
  DST[P.p5 + ((P.p2 + r) % P.p3) * P.p1 + d] = SRC[P.p4 + i];
}
"#;

/// Pool compress groups j0.. from the raw kv/score rings, then RMSNorm with weight.
/// p0=G p1=D p2=j0 p3=ratio p4=R p5=eps
const COMPRESS: &str = r#"
@group(0) @binding(0) var<storage, read> KR: array<f32>;
@group(0) @binding(1) var<storage, read> SR: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
var<workgroup> red: array<f32, 32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32) {
  let D = P.p1; let r = P.p3; let R = P.p4;
  let j = P.p2 + wid.x;
  var s2 = 0.0;
  for (var d = li; d < D; d += 256u) {
    var v = 0.0;
    if (r == 1u) {
      v = KR[(j % R) * D + d];
    } else {
      var mx = NEG_INF;
      for (var i = 0u; i < r; i++) { mx = max(mx, SR[((j * r + i) % R) * D + d]); }
      var den = 0.0;
      for (var i = 0u; i < r; i++) {
        let row = ((j * r + i) % R) * D + d;
        let e = exp(SR[row] - mx);
        den += e; v += e * KR[row];
      }
      v = v / den;
    }
    OUT[wid.x * D + d] = v;
    s2 += v * v;
  }
  s2 = subgroupAdd(s2);
  if (sl == 0u) { red[li / ss] = s2; }
  workgroupBarrier();
  var tot = 0.0;
  for (var i = 0u; i < 256u / ss; i++) { tot += red[i]; }
  let rr = inverseSqrt(tot / f32(D) + fbits(P.p5));
  for (var d = li; d < D; d += 256u) { OUT[wid.x * D + d] = OUT[wid.x * D + d] * rr * Wt[d]; }
}
"#;

/// Plain row copy: DST[dst_off + i] = SRC[src_off + i]. p0=n p1=src_off p2=dst_off
const COPY: &str = r#"
@group(0) @binding(0) var<storage, read> SRC: array<f32>;
@group(0) @binding(1) var<storage, read_write> DST: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= P.p0) { return; }
  DST[P.p2 + i] = SRC[P.p1 + i];
}
"#;

/// Indexer scores: S[m][t] = sum_h relu(q[m,h] . k[t]) * w[m,h] * scale, t < n_vis(m) else -inf.
/// Q [M][heads*hd], WP [M][heads], K [n][hd]. p0=n_cols(row stride of S) p1=heads p2=hd
/// p3=pos0 p4=ratio p5=scale p6=use_cand p7=cand_block p8=cand_stride(u32 words per row)
const IDX_SCORES: &str = r#"
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> WP: array<f32>;
@group(0) @binding(2) var<storage, read> Kc: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> CAND: array<u32>;
@group(0) @binding(4) var<storage, read_write> S: array<f32>;
var<workgroup> qs: array<vec4<f32>, 1024>;
var<workgroup> ws: array<f32, 64>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let heads = P.p1; let hd = P.p2; let hv = hd / 4u;
  let m = wid.y;
  for (var i = li; i < heads * hv; i += 256u) {
    let b = m * heads * hd + i * 4u;
    qs[i] = vec4<f32>(Q[b], Q[b + 1u], Q[b + 2u], Q[b + 3u]);
  }
  if (li < heads) { ws[li] = WP[m * heads + li] * fbits(P.p5); }
  workgroupBarrier();
  let t = wid.x * 256u + li;
  if (t >= P.p0) { return; }
  let nvis = (P.p3 + m + 1u) / P.p4;
  var out = NEG_INF;
  var ok = t < nvis;
  if (ok && P.p6 != 0u) {
    let blk = t / P.p7;
    ok = ((CAND[m * P.p8 + (blk >> 5u)] >> (blk & 31u)) & 1u) != 0u;
  }
  if (ok) {
    var acc = 0.0;
    for (var h = 0u; h < heads; h++) {
      var d = 0.0;
      for (var i = 0u; i < hv; i++) { d += dot(qs[h * hv + i], Kc[t * hv + i]); }
      acc += max(d, 0.0) * ws[h];
    }
    out = acc;
  }
  S[m * P.p0 + t] = out;
}
"#;

/// Block maxima for candidate selection; the block with the newest visible position is pinned
/// to +big. p0=n_cols p1=block p2=n_blocks(row stride of B) p3=pos0 p4=ratio
const BLOCK_MAX: &str = r#"
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> B: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x; let m = gid.y;
  if (b >= P.p2) { return; }
  let nvis = (P.p3 + m + 1u) / P.p4;
  var mx = NEG_INF;
  for (var i = 0u; i < P.p1; i++) {
    let t = b * P.p1 + i;
    if (t < nvis) { mx = max(mx, S[m * P.p0 + t]); }
  }
  if (nvis > 0u && b == (nvis - 1u) / P.p1) { mx = 3.0e38; }
  B[m * P.p2 + b] = mx;
}
"#;

/// Mark selected candidate blocks in a bitset. IDX [M][k] block ids (-1 = none).
/// p0=k p1=words per row p2=M
const CAND_MARK: &str = r#"
@group(0) @binding(0) var<storage, read> IDX: array<i32>;
@group(0) @binding(1) var<storage, read_write> CAND: array<atomic<u32>>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let m = gid.y;
  if (i >= P.p0 || m >= P.p2) { return; }
  let b = IDX[m * P.p0 + i];
  if (b >= 0) { atomicOr(&CAND[m * P.p1 + (u32(b) >> 5u)], 1u << (u32(b) & 31u)); }
}
"#;

/// Zero a u32 range. p0=n p1=off
const ZERO: &str = r#"
@group(0) @binding(0) var<storage, read_write> X: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 65535u * 256u;
  if (i < P.p0) { X[P.p1 + i] = 0u; }
}
"#;

/// Exact top-k per row by 4-pass radix select. Writes indices in ascending order, -1 padded;
/// ties at the threshold go to the lowest index, so the result never depends on scheduling.
/// S [M][n_cols]. p0=n_cols p1=k p2=pos0 p3=ratio (n_vis = (pos0+m+1)/ratio, capped at n_cols)
/// p4=out_stride p5=out_off p6=n_vis_override(0 = use formula) p7=add_offset
const TOPK: &str = r#"
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read_write> OUT: array<i32>;
var<workgroup> hist: array<atomic<u32>, 256>;
var<workgroup> prefix: u32;
var<workgroup> remaining: u32;
var<workgroup> n_gt: array<u32, 256>;
var<workgroup> n_eq: array<u32, 256>;

fn okey(f: f32) -> u32 {
  let b = bitcast<u32>(f);
  return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let m = wid.x;
  let k = P.p1;
  var nvis = P.p6;
  if (nvis == 0u) { nvis = (P.p2 + m + 1u) / P.p3; }
  nvis = min(nvis, P.p0);
  let ob = P.p5 + m * P.p4;
  let row = m * P.p0;
  if (nvis <= k) {
    for (var i = li; i < k; i += 256u) {
      OUT[ob + i] = select(-1, i32(i + P.p7), i < nvis);
    }
    return;
  }
  if (li == 0u) { prefix = 0u; remaining = k; }
  for (var ps = 0u; ps < 4u; ps++) {
    let shift = 24u - ps * 8u;
    atomicStore(&hist[li], 0u);
    workgroupBarrier();
    let pf = prefix;
    let hmask = select(0u, 0xffffffffu << ((shift + 8u) & 31u), ps > 0u);
    for (var i = li; i < nvis; i += 256u) {
      let key = okey(S[row + i]);
      if ((key & hmask) == (pf & hmask)) { atomicAdd(&hist[(key >> shift) & 255u], 1u); }
    }
    workgroupBarrier();
    if (li == 0u) {
      var rem = remaining;
      var bin = 255u;
      loop {
        let c = atomicLoad(&hist[bin]);
        if (c >= rem || bin == 0u) { break; }
        rem -= c;
        bin -= 1u;
      }
      remaining = rem;
      prefix = pf | (bin << shift);
    }
    workgroupBarrier();
  }
  let thr = prefix;
  let need_eq = remaining;
  // Stable compaction: each invocation owns a contiguous block of columns.
  let blk = (nvis + 255u) / 256u;
  let lo = min(li * blk, nvis);
  let hi = min(lo + blk, nvis);
  var g = 0u;
  var e = 0u;
  for (var i = lo; i < hi; i++) {
    let key = okey(S[row + i]);
    if (key > thr) { g++; } else if (key == thr) { e++; }
  }
  n_gt[li] = g;
  n_eq[li] = e;
  workgroupBarrier();
  if (li == 0u) {
    var sg = 0u;
    var se = 0u;
    for (var i = 0u; i < 256u; i++) {
      let cg = n_gt[i]; let ce = n_eq[i];
      n_gt[i] = sg; n_eq[i] = se;
      sg += cg; se += ce;
    }
  }
  workgroupBarrier();
  g = n_gt[li];
  e = n_eq[li];
  for (var i = lo; i < hi; i++) {
    let key = okey(S[row + i]);
    if (key > thr) {
      OUT[ob + g + min(e, need_eq)] = i32(i + P.p7);
      g++;
    } else if (key == thr) {
      if (e < need_eq) { OUT[ob + g + e] = i32(i + P.p7); }
      e++;
    }
  }
}
"#;

/// Sparse attention scores: S[m][h][j] = scale * q[m,h] . kv(idx[m][j]) for j over
/// window slots (WIDX [M][W] into the ring) then compressed slots (CIDX [M][T] into the cache).
/// p0=n_slots p1=W p2=T(cidx stride) p3=heads p4=hd p5=scale p6=cidx_off
const ATTN_SCORES: &str = r#"
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> KW: array<f32>;
@group(0) @binding(2) var<storage, read> KC: array<f32>;
@group(0) @binding(3) var<storage, read> WIDX: array<i32>;
@group(0) @binding(4) var<storage, read> CIDX: array<i32>;
@group(0) @binding(5) var<storage, read_write> S: array<f32>;
var<workgroup> qt: array<array<f32, 17>, 16>;
var<workgroup> kt: array<array<f32, 17>, 16>;
var<workgroup> kidx: array<i32, 16>;
var<workgroup> ksrc: array<u32, 16>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let ns = P.p0; let W = P.p1; let heads = P.p3; let hd = P.p4;
  let m = wid.z;
  let j0 = wid.x * 16u; let h0 = wid.y * 16u;
  let ty = li / 16u; let tx = li % 16u;
  if (li < 16u) {
    let j = j0 + li;
    var ix = -1; var src = 0u;
    if (j < ns) {
      if (j < W) { ix = WIDX[m * W + j]; } else { ix = CIDX[P.p6 + m * P.p2 + (j - W)]; src = 1u; }
    }
    kidx[li] = ix; ksrc[li] = src;
  }
  workgroupBarrier();
  var acc = 0.0;
  for (var d0 = 0u; d0 < hd; d0 += 16u) {
    qt[ty][tx] = Q[(m * heads + h0 + ty) * hd + d0 + tx];
    let ix = kidx[ty];
    var kv = 0.0;
    if (ix >= 0) {
      if (ksrc[ty] == 0u) { kv = KW[u32(ix) * hd + d0 + tx]; } else { kv = KC[u32(ix) * hd + d0 + tx]; }
    }
    kt[ty][tx] = kv;
    workgroupBarrier();
    for (var d = 0u; d < 16u; d++) { acc += qt[ty][d] * kt[tx][d]; }
    workgroupBarrier();
  }
  let j = j0 + tx;
  if (j < ns) {
    var v = acc * fbits(P.p5);
    if (kidx[tx] < 0) { v = NEG_INF; }
    S[(m * heads + h0 + ty) * ns + j] = v;
  }
}
"#;

/// Softmax with an attention sink, in place over each (m, h) row. p0=n_slots p1=heads
const ATTN_SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<storage, read> SINK: array<f32>;
var<workgroup> red: array<f32, 32>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32) {
  let ns = P.p0;
  let h = wid.x; let m = wid.y;
  let base = (m * P.p1 + h) * ns;
  let sink = SINK[h];
  var mx = sink;
  for (var j = li; j < ns; j += 256u) { mx = max(mx, S[base + j]); }
  mx = subgroupMax(mx);
  if (sl == 0u) { red[li / ss] = mx; }
  workgroupBarrier();
  var gm = NEG_INF;
  for (var i = 0u; i < 256u / ss; i++) { gm = max(gm, red[i]); }
  workgroupBarrier();
  var sum = 0.0;
  for (var j = li; j < ns; j += 256u) {
    let s = S[base + j];
    var e = 0.0;
    if (s > -1.0e38) { e = exp(s - gm); }
    S[base + j] = e;
    sum += e;
  }
  sum = subgroupAdd(sum);
  if (sl == 0u) { red[li / ss] = sum; }
  workgroupBarrier();
  var tot = exp(sink - gm);
  for (var i = 0u; i < 256u / ss; i++) { tot += red[i]; }
  let inv = 1.0 / tot;
  for (var j = li; j < ns; j += 256u) { S[base + j] *= inv; }
}
"#;

/// O[m][h][d] = sum_j P[m][h][j] * kv(idx[m][j])[d]   (same slot layout as ATTN_SCORES)
const ATTN_PV: &str = r#"
@group(0) @binding(0) var<storage, read> PR: array<f32>;
@group(0) @binding(1) var<storage, read> KW: array<f32>;
@group(0) @binding(2) var<storage, read> KC: array<f32>;
@group(0) @binding(3) var<storage, read> WIDX: array<i32>;
@group(0) @binding(4) var<storage, read> CIDX: array<i32>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
var<workgroup> pt: array<array<f32, 17>, 16>;
var<workgroup> vt: array<array<f32, 17>, 16>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let ns = P.p0; let W = P.p1; let heads = P.p3; let hd = P.p4;
  let m = wid.z;
  let d0 = wid.x * 16u; let h0 = wid.y * 16u;
  let ty = li / 16u; let tx = li % 16u;
  var acc = 0.0;
  for (var j0 = 0u; j0 < ns; j0 += 16u) {
    let jp = j0 + tx;
    var p = 0.0;
    if (jp < ns) { p = PR[(m * heads + h0 + ty) * ns + jp]; }
    pt[ty][tx] = p;
    let j = j0 + ty;
    var v = 0.0;
    if (j < ns) {
      var ix = -1; var src = 0u;
      if (j < W) { ix = WIDX[m * W + j]; } else { ix = CIDX[P.p6 + m * P.p2 + (j - W)]; src = 1u; }
      if (ix >= 0) {
        if (src == 0u) { v = KW[u32(ix) * hd + d0 + tx]; } else { v = KC[u32(ix) * hd + d0 + tx]; }
      }
    }
    vt[ty][tx] = v;
    workgroupBarrier();
    for (var jj = 0u; jj < 16u; jj++) { acc += pt[ty][jj] * vt[jj][tx]; }
    workgroupBarrier();
  }
  O[(m * heads + h0 + ty) * hd + d0 + tx] = acc;
}
"#;

fn k(body: &str) -> String {
    format!("{PRELUDE}\n{body}")
}

/// A `*_LUT` kernel: how it reads the weights (FETCH) and where the scales follow them (WPW,
/// weights per word) depends on whether those are 4-bit codes or bytes.
fn lut(body: &str, tm: &str, wide: bool) -> String {
    let gate_up = body.contains("let up =");
    let fetch = match (gate_up, wide) {
        (true, false) => "let wg = S[s * 4u + i]; let wu = S[up + s * 4u + i];
        g[i] = lo4(wg); g[i + 4u] = hi4(wg); u[i] = lo4(wu); u[i + 4u] = hi4(wu);",
        (false, false) => "let ww = S[base + s * 4u + i]; w[i] = lo4(ww); w[i + 4u] = hi4(ww);",
        (true, true) => "g[i] = vec4<f32>(unpack4xI8(S[s * 8u + i])); g[i + 4u] = vec4<f32>(unpack4xI8(S[s * 8u + i + 4u]));
        u[i] = vec4<f32>(unpack4xI8(S[up + s * 8u + i])); u[i + 4u] = vec4<f32>(unpack4xI8(S[up + s * 8u + i + 4u]));",
        (false, true) => "w[i] = vec4<f32>(unpack4xI8(S[base + s * 8u + i])); w[i + 4u] = vec4<f32>(unpack4xI8(S[base + s * 8u + i + 4u]));",
    };
    let src = format!("{LUT_COMMON}{body}").replace("FETCH", fetch).replace("WPW", if wide { "4u" } else { "8u" });
    k(&src.replace("TM", tm))
}

/// The kernels for expert matrices of a type: gate and up, down; for one token and for up to eight.
pub fn moe(ty: u32) -> [&'static str; 4] {
    if ty == crate::gguf::GGML_MXFP4 {
        ["moe_gate_up_1", "moe_gate_up_8", "moe_down_1", "moe_down_8"]
    } else if crate::quants::wide(ty) {
        ["moe_gate_up_wide_1", "moe_gate_up_wide_8", "moe_down_wide_1", "moe_down_wide_8"]
    } else {
        ["moe_gate_up_lut_1", "moe_gate_up_lut_8", "moe_down_lut_1", "moe_down_lut_8"]
    }
}

pub fn all() -> Vec<(&'static str, String)> {
    let mut v = vec![
        ("mm_q8_1", k(&MM_Q8.replace("TM", "1u"))),
        ("mm_q8_8", k(&MM_Q8.replace("TM", "8u"))),
        ("mm_f32_1", k(&MM_F32.replace("TM", "1u"))),
        ("mm_f32_8", k(&MM_F32.replace("TM", "8u"))),
        ("moe_gate_up_1", k(&format!("{MXFP4_COMMON}{}", MOE_GATE_UP.replace("TM", "1u")))),
        ("moe_gate_up_8", k(&format!("{MXFP4_COMMON}{}", MOE_GATE_UP.replace("TM", "8u")))),
        ("moe_down_1", k(&format!("{MXFP4_COMMON}{}", MOE_DOWN.replace("TM", "1u")))),
        ("moe_down_8", k(&format!("{MXFP4_COMMON}{}", MOE_DOWN.replace("TM", "8u")))),
        ("moe_gate_up_lut_1", lut(MOE_GATE_UP_LUT, "1u", false)),
        ("moe_gate_up_lut_8", lut(MOE_GATE_UP_LUT, "8u", false)),
        ("moe_down_lut_1", lut(MOE_DOWN_LUT, "1u", false)),
        ("moe_down_lut_8", lut(MOE_DOWN_LUT, "8u", false)),
        ("moe_gate_up_wide_1", lut(MOE_GATE_UP_LUT, "1u", true)),
        ("moe_gate_up_wide_8", lut(MOE_GATE_UP_LUT, "8u", true)),
        ("moe_down_wide_1", lut(MOE_DOWN_LUT, "1u", true)),
        ("moe_down_wide_8", lut(MOE_DOWN_LUT, "8u", true)),
        ("rmsnorm", k(RMSNORM)),
        ("rope", k(ROPE)),
        ("quant_sim", k(QUANT_SIM)),
        ("hc_dot", k(HC_DOT)),
        ("hc_mix", k(HC_MIX)),
        ("hc_pre_norm", k(HC_PRE_NORM)),
        ("hc_post", k(HC_POST)),
        ("engram", k(ENGRAM)),
        ("swiglu", k(SWIGLU)),
        ("copy_ring", k(COPY_RING)),
        ("compress", k(COMPRESS)),
        ("copy", k(COPY)),
        ("idx_scores", k(IDX_SCORES)),
        ("block_max", k(BLOCK_MAX)),
        ("cand_mark", k(CAND_MARK)),
        ("zero", k(ZERO)),
        ("topk", k(TOPK)),
        ("attn_scores", k(ATTN_SCORES)),
        ("attn_softmax", k(ATTN_SOFTMAX)),
        ("attn_pv", k(ATTN_PV)),
    ];
    v.shrink_to_fit();
    v
}
