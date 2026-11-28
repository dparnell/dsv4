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

/// MM_Q8 for one token (same parameters), reading each row in 16-byte pieces so that 32 lanes read
/// 512 bytes in a row: 1.3 to 1.5 times as fast as MM_Q8 reading 32-byte blocks per lane.
const MM_Q8_1: &str = r#"
@group(0) @binding(0) var<storage, read> W: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;

fn q4(w: u32, x: vec4<f32>) -> f32 { return dot(vec4<f32>(unpack4xI8(w)), x); }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let N = P.p0; let K = P.p1;
  let lane = li & 31u;
  let row = wid.x * 8u + (li >> 5u);
  var acc = 0.0;
  if (row < N) {
    var xg = 0u;
    if (P.p8 != 0u) { xg = (row / P.p8) * K; }
    let kv = K / 16u;
    let rv = row * kv;
    let xb = (P.p5 + xg) >> 2u;
    for (var j = lane; j < kv; j += 32u) {
      let w = W[rv + j];
      // a block of 32 weights is two of these
      let s = (rv + j) >> 1u;
      let sw = P.p7 + (s >> 1u);
      let sc = unpack2x16float(W[sw >> 2u][sw & 3u])[s & 1u];
      let xi = xb + j * 4u;
      acc += (q4(w.x, X[xi]) + q4(w.y, X[xi + 1u]) + q4(w.z, X[xi + 2u]) + q4(w.w, X[xi + 3u])) * sc;
    }
  }
  red[li] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < N) {
    let yi = P.p6 + P.p10 + row;
    if (P.p9 != 0u) { Y[yi] += red[li]; } else { Y[yi] = red[li]; }
  }
}
"#;


/// Two MM_Q8_1 rows at once (gate and up, same shape, one part each) and SWIGLU of the pair: one
/// token, each row summed exactly as MM_Q8_1 sums it. A[n] = silu(min(g, l)) * clamp(u, -l, l).
/// p0=N p1=K p5=x_off p7=scale_word_off (the same in both) p11=limit
const MM_Q8_1_GLU: &str = r#"
@group(0) @binding(0) var<storage, read> WG: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> WU: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
var<workgroup> rg: array<f32, 256>;
var<workgroup> ru: array<f32, 256>;

fn q4(w: u32, x: vec4<f32>) -> f32 { return dot(vec4<f32>(unpack4xI8(w)), x); }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let N = P.p0; let K = P.p1;
  let lane = li & 31u;
  let row = wid.x * 8u + (li >> 5u);
  var ag = 0.0;
  var au = 0.0;
  if (row < N) {
    let kv = K / 16u;
    let rv = row * kv;
    let xb = P.p5 >> 2u;
    for (var j = lane; j < kv; j += 32u) {
      let wg = WG[rv + j];
      let wu = WU[rv + j];
      let s = (rv + j) >> 1u;
      let sw = P.p7 + (s >> 1u);
      let sg = unpack2x16float(WG[sw >> 2u][sw & 3u])[s & 1u];
      let su = unpack2x16float(WU[sw >> 2u][sw & 3u])[s & 1u];
      let xi = xb + j * 4u;
      let x0 = X[xi]; let x1 = X[xi + 1u]; let x2 = X[xi + 2u]; let x3 = X[xi + 3u];
      ag += (q4(wg.x, x0) + q4(wg.y, x1) + q4(wg.z, x2) + q4(wg.w, x3)) * sg;
      au += (q4(wu.x, x0) + q4(wu.y, x1) + q4(wu.z, x2) + q4(wu.w, x3)) * su;
    }
  }
  rg[li] = ag;
  ru[li] = au;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { rg[li] += rg[li + s]; ru[li] += ru[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < N) {
    let l = fbits(P.p11);
    let g = min(rg[li], l);
    A[row] = g / (1.0 + exp(-g)) * clamp(ru[li], -l, l);
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

/// Two MM_F32 products of one token's x at once (this layer's router and the next layer's, the
/// same shape): z = 0 uses W0 -> Y0, z = 1 W1 -> Y1, each row summed exactly as MM_F32 sums it.
/// p0=N p1=K p5=x_off p6=y_off
const MM_F32_1_PAIR: &str = r#"
@group(0) @binding(0) var<storage, read> W0: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> W1: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> X: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> Y0: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y1: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let N = P.p0; let K = P.p1;
  let lane = li & 31u;
  let row = wid.x * 8u + (li >> 5u);
  let nv = K / 4u;
  var acc = 0.0;
  if (row < N) {
    for (var v = lane; v < nv; v += 32u) {
      var w: vec4<f32>;
      if (wid.z == 0u) { w = W0[row * nv + v]; } else { w = W1[row * nv + v]; }
      acc += dot(w, X[(P.p5 >> 2u) + v]);
    }
  }
  red[li] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && row < N) {
    if (wid.z == 0u) { Y0[P.p6 + row] = red[li]; } else { Y1[P.p6 + row] = red[li]; }
  }
}
"#;

/// MXFP4 expert helpers. A slot holds gate | up | down, each [qs N*nb*4 words][e8m0 N*nb bytes].
const MXFP4_COMMON: &str = r#"
fn e8m0(S: u32) -> f32 { return bitcast<f32>(S << 23u); }
// 0, 0.5, 1, 1.5, 2, 3, 4, 6 from a table of 4-bit fields (an array of 16 floats made the kernels
// three times slower); bit 3 is the sign
fn val(w: vec4<u32>) -> vec4<f32> {
  let m = vec4<f32>((vec4<u32>(0xc8643210u) >> ((w & vec4<u32>(7u)) << vec4<u32>(2u))) & vec4<u32>(15u)) * 0.5;
  return select(m, -m, (w & vec4<u32>(8u)) != vec4<u32>(0u));
}
fn lo4(w: u32) -> vec4<f32> { return val(vec4<u32>(w) >> vec4<u32>(0u, 8u, 16u, 24u)); }
fn hi4(w: u32) -> vec4<f32> { return val(vec4<u32>(w) >> vec4<u32>(4u, 12u, 20u, 28u)); }
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

/// What replaces LUT_COMMON for the `compact` formats (quants.rs): `oct` gives 8 weights of the
/// group of 32 number s in the matrix at word m; WORDS, BYTES and FLOATS are its layout.
const COMPACT_COMMON: &str = r#"
fn init_lut() {}
fn word(m: u32, s: u32, i: u32) -> u32 { return S[m + s * WORDS + i]; }
fn tail(m: u32, s: u32) -> u32 {
  let i = s * BYTES;
  return S[m + P.p0 * P.p1 / 32u * WORDS + (i >> 2u)] >> ((i & 3u) * 8u);
}
fn float(m: u32, s: u32, i: u32) -> f32 {
  let g = P.p0 * P.p1 / 32u;
  return fbits(S[m + g * WORDS + g * BYTES / 4u + (s >> 3u) * FLOATS + i]);
}
fn odd2(a: u32, b: u32) -> vec2<f32> { return vec2<f32>(f32(2u * a + 1u), f32(2u * b + 1u)); }
fn nib2(v: u32) -> vec2<f32> { return odd2(v & 15u, (v >> 4u) & 15u); }
// the last of 8 signs makes the parity even
fn even(v: u32) -> u32 { let b = v & 127u; return b | ((countOneBits(b) & 1u) << 7u); }
fn int4(w: u32) -> vec4<f32> { return vec4<f32>(unpack4xI8(w)); }
fn neg4(v: vec4<f32>, bits: u32) -> vec4<f32> {
  return select(v, -v, (vec4<u32>(bits) & vec4<u32>(1u, 2u, 4u, 8u)) != vec4<u32>(0u));
}
fn eight(a: u32, b: u32, bits: u32) -> mat2x4<f32> {
  return mat2x4<f32>(neg4(int4(G[GRID + a]), bits), neg4(int4(G[GRID + b]), bits >> 4u));
}
fn scales(m: u32, row: u32, b: u32) -> vec2<f32> {
  let s = row * (P.p1 / 32u) + b;
  return scale2(m, s) * float(m, s, 0u);
}
fn mins(m: u32, row: u32, b: u32) -> vec2<f32> {
  let s = row * (P.p1 / 32u) + b;
  return MINS;
}
"#;

const COMPACT_Q2_K: &str = r#"
fn scale2(m: u32, s: u32) -> vec2<f32> { let t = tail(m, s); return vec2<f32>(f32(t & 15u), f32((t >> 8u) & 15u)); }
fn min2(m: u32, s: u32) -> vec2<f32> { let t = tail(m, s); return vec2<f32>(f32((t >> 4u) & 15u), f32((t >> 12u) & 15u)); }
fn oct(m: u32, s: u32, l: u32) -> mat2x4<f32> {
  let w = word(m, s, l >> 1u) >> ((l & 1u) * 16u);
  let sh = vec4<u32>(0u, 2u, 4u, 6u);
  let a = (vec4<u32>(w) >> sh) & vec4<u32>(3u);
  let b = (vec4<u32>(w >> 8u) >> sh) & vec4<u32>(3u);
  return mat2x4<f32>(vec4<f32>(a), vec4<f32>(b));
}
"#;

const COMPACT_IQ2_XXS: &str = r#"
fn scale2(m: u32, s: u32) -> vec2<f32> { let c = word(m, s, 1u) >> 28u; return odd2(c, c); }
fn oct(m: u32, s: u32, l: u32) -> mat2x4<f32> {
  let i = ((word(m, s, 0u) >> (l * 8u)) & 255u) * 2u;
  return eight(i, i + 1u, even(word(m, s, 1u) >> (l * 7u)));
}
"#;

const COMPACT_IQ2_XS: &str = r#"
fn scale2(m: u32, s: u32) -> vec2<f32> { return nib2(tail(m, s)); }
fn oct(m: u32, s: u32, l: u32) -> mat2x4<f32> {
  let q = word(m, s, l >> 1u) >> ((l & 1u) * 16u);
  let i = (q & 511u) * 2u;
  return eight(i, i + 1u, even(q >> 9u));
}
"#;

const COMPACT_IQ2_S: &str = r#"
fn scale2(m: u32, s: u32) -> vec2<f32> { return nib2(tail(m, s) >> 8u); }
fn oct(m: u32, s: u32, l: u32) -> mat2x4<f32> {
  let hi = ((tail(m, s) >> (l * 2u)) & 3u) << 8u;
  let i = (((word(m, s, 0u) >> (l * 8u)) & 255u) | hi) * 2u;
  return eight(i, i + 1u, word(m, s, 1u) >> (l * 8u));
}
"#;

const COMPACT_IQ3_XXS: &str = r#"
fn scale2(m: u32, s: u32) -> vec2<f32> { let c = word(m, s, 2u) >> 28u; return odd2(c, c); }
fn oct(m: u32, s: u32, l: u32) -> mat2x4<f32> {
  let q = word(m, s, l >> 1u) >> ((l & 1u) * 16u);
  return eight(q & 255u, (q >> 8u) & 255u, even(word(m, s, 2u) >> (l * 7u)));
}
"#;

const COMPACT_IQ3_S: &str = r#"
fn scale2(m: u32, s: u32) -> vec2<f32> { let c = (tail(m, s) >> 8u) & 15u; return odd2(c, c); }
fn oct(m: u32, s: u32, l: u32) -> mat2x4<f32> {
  let q = word(m, s, l >> 1u) >> ((l & 1u) * 16u);
  let hi = tail(m, s) >> (l * 2u);
  return eight((q & 255u) | ((hi & 1u) << 8u), ((q >> 8u) & 255u) | ((hi & 2u) << 7u), word(m, s, 2u) >> (l * 8u));
}
"#;

const COMPACT_IQ1_S: &str = r#"
fn scale2(m: u32, s: u32) -> vec2<f32> { let c = (tail(m, s) >> 12u) & 7u; return odd2(c, c); }
fn oct(m: u32, s: u32, l: u32) -> mat2x4<f32> {
  let q = tail(m, s);
  let i = (((word(m, s, 0u) >> (l * 8u)) & 255u) | (((q >> (l * 3u)) & 7u) << 8u)) * 2u + ((q >> 15u) & 1u) * 4096u;
  return mat2x4<f32>(int4(G[GRID + i]), int4(G[GRID + i + 1u]));
}
"#;

const COMPACT_IQ1_M: &str = r#"
fn scale2(m: u32, s: u32) -> vec2<f32> { let t = word(m, s, 1u) >> 16u; return odd2(t & 7u, (t >> 4u) & 7u); }
fn oct(m: u32, s: u32, l: u32) -> mat2x4<f32> {
  let h = word(m, s, 1u) >> (l * 4u);
  let i = (((word(m, s, 0u) >> (l * 8u)) & 255u) | ((h & 7u) << 8u)) * 2u + ((h >> 3u) & 1u) * 4096u;
  return mat2x4<f32>(int4(G[GRID + i]), int4(G[GRID + i + 1u]));
}
"#;

/// The formats of the COMPACT kernels.
const COMPACT: [(u32, &str, [&str; 4]); 8] = {
    use crate::quants::*;
    macro_rules! names {
        ($t:literal) => {
            [concat!("moe_gate_up_", $t, "_1"), concat!("moe_gate_up_", $t, "_8"), concat!("moe_down_", $t, "_1"), concat!("moe_down_", $t, "_8")]
        };
    }
    [
        (Q2_K, COMPACT_Q2_K, names!("q2_k")),
        (IQ2_XXS, COMPACT_IQ2_XXS, names!("iq2_xxs")),
        (IQ2_XS, COMPACT_IQ2_XS, names!("iq2_xs")),
        (IQ2_S, COMPACT_IQ2_S, names!("iq2_s")),
        (IQ3_XXS, COMPACT_IQ3_XXS, names!("iq3_xxs")),
        (IQ3_S, COMPACT_IQ3_S, names!("iq3_s")),
        (IQ1_S, COMPACT_IQ1_S, names!("iq1_s")),
        (IQ1_M, COMPACT_IQ1_M, names!("iq1_m")),
    ]
};

/// The compact kernels, built when first used: a model has a few of these formats at most.
pub fn on_demand() -> Vec<(&'static str, String)> {
    let mut v = Vec::new();
    for (ty, oct, names) in COMPACT {
        let c = crate::quants::compact(ty).unwrap();
        for (i, name) in names.into_iter().enumerate() {
            let (body, fetch) = if i < 2 {
                (
                    MOE_GATE_UP_LUT,
                    "let og = oct(0u, s, i); let ou = oct(up, s, i);
        g[2u * i] = og[0]; g[2u * i + 1u] = og[1]; u[2u * i] = ou[0]; u[2u * i + 1u] = ou[1];",
                )
            } else {
                (MOE_DOWN_LUT, "let o = oct(base, s, i); w[2u * i] = o[0]; w[2u * i + 1u] = o[1];")
            };
            let bind = if c.grid.is_some() { "@group(0) @binding(4) var<storage, read> G: array<u32>;" } else { "" };
            let mut common = COMPACT_COMMON.to_string();
            if c.grid.is_none() {
                // no grid, no signs
                let a = common.find("fn eight").unwrap();
                let b = common.find("fn scales").unwrap();
                common.replace_range(a..b, "");
            }
            let src = format!("{bind}\n{common}{oct}{body}")
                .replace("FETCH", fetch)
                .replace("MINS", if crate::quants::has_min(ty) { "min2(m, s) * float(m, s, 1u)" } else { "vec2<f32>(0.0)" })
                .replace("P.p7 != 0u", if crate::quants::has_min(ty) { "true" } else { "false" })
                .replace("WORDS", &format!("{}u", c.words))
                .replace("BYTES", &format!("{}u", c.bytes))
                .replace("FLOATS", &format!("{}u", c.floats))
                .replace("GRID", &format!("{}u", c.grid.unwrap_or(0)))
                .replace("TM", if i % 2 == 0 { "1u" } else { "8u" });
            v.push((name, k(&src)));
        }
    }
    v
}

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

/// One row per workgroup (rows = tokens * heads, each `L` long, L <= 1024): optionally RMSNorm with
/// a weight, RoPE on the trailing 2*half values, quantize-dequantize in blocks, then write in place
/// or to D (linearly, or into a ring of R rows). Each step computes exactly what RMSNORM, ROPE and
/// QUANT_SIM do, in the same order. p0=L p1=heads p2=x_off p3=flags (1 norm, 2 rope, 4 inverse
/// rope, 8 write to D) p4=eps p5=pos_base p6=pos_step p7=table_off p8=half p9=quant block (0: none)
/// p10=quant mode p11=dst_off p12=ring_base p13=ring rows (0: linear) p14=rows
const ROW_POST: &str = r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> T: array<f32>;
@group(0) @binding(3) var<storage, read_write> D: array<f32>;
var<workgroup> red: array<f32, 128>;
var<workgroup> sv: array<f32, 1024>;

fn rp_scale(amax: f32, mode: u32) -> f32 {
  if (mode == 0u) { return pow2_ceil(max(amax, 1e-4) / 448.0); }
  if (mode == 1u) { return pow2_ceil(max(amax, 6.0 * 1.1754944e-38) / 6.0); }
  return to_e4m3(max(amax, 6.0 / 512.0) / 6.0);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32) {
  let L = P.p0; let fl = P.p3;
  let row = wid.x + wid.y * 65535u;
  if (row >= P.p14) { return; }
  let xb = P.p2 + row * L;
  var v: array<f32, 4>;
  for (var q = 0u; q < 4u; q++) {
    let i = li + q * 256u;
    v[q] = 0.0;
    if (i < L) { v[q] = X[xb + i]; }
  }
  if ((fl & 1u) != 0u) {
    var s = 0.0;
    for (var q = 0u; q < 4u; q++) { s += v[q] * v[q]; }
    s = subgroupAdd(s);
    if (sl == 0u) { red[li / ss] = s; }
    workgroupBarrier();
    var tot = 0.0;
    for (var i = 0u; i < 256u / ss; i++) { tot += red[i]; }
    let r = inverseSqrt(tot / f32(L) + fbits(P.p4));
    for (var q = 0u; q < 4u; q++) {
      let i = li + q * 256u;
      if (i < L) { v[q] = v[q] * r * Wt[i]; }
    }
  }
  if ((fl & 2u) != 0u) {
    let half = P.p8;
    let r0 = L - 2u * half;
    let pos = P.p5 + (row / P.p1) * P.p6;
    for (var q = 0u; q < 4u; q++) {
      let i = li + q * 256u;
      let other = subgroupShuffleXor(v[q], 1u);
      if (i < L && i >= r0) {
        let k = (i - r0) / 2u;
        let c = T[P.p7 + (pos * half + k) * 2u];
        var s = T[P.p7 + (pos * half + k) * 2u + 1u];
        if ((fl & 4u) != 0u) { s = -s; }
        if ((i & 1u) == 0u) { v[q] = v[q] * c - other * s; } else { v[q] = other * s + v[q] * c; }
      }
    }
  }
  let bs = P.p9;
  if (bs != 0u) {
    let mode = P.p10;
    if (ss >= bs) {
      for (var q = 0u; q < 4u; q++) {
        var amax = abs(v[q]);
        for (var o = 1u; o < bs; o <<= 1u) { amax = max(amax, subgroupShuffleXor(amax, o)); }
        let s = rp_scale(amax, mode);
        if (mode == 0u) { v[q] = to_e4m3(clamp(v[q] / s, -448.0, 448.0)) * s; }
        else { v[q] = to_e2m1(clamp(v[q] / s, -6.0, 6.0)) * s; }
      }
    } else {
      for (var q = 0u; q < 4u; q++) { sv[li + q * 256u] = v[q]; }
      workgroupBarrier();
      for (var q = 0u; q < 4u; q++) {
        let i = li + q * 256u;
        let b0 = i / bs * bs;
        var amax = 0.0;
        for (var j = 0u; j < bs; j++) { amax = max(amax, abs(sv[b0 + j])); }
        let s = rp_scale(amax, mode);
        if (mode == 0u) { v[q] = to_e4m3(clamp(v[q] / s, -448.0, 448.0)) * s; }
        else { v[q] = to_e2m1(clamp(v[q] / s, -6.0, 6.0)) * s; }
      }
    }
  }
  var ob = xb;
  if ((fl & 8u) != 0u) {
    var r = row;
    if (P.p13 != 0u) { r = (P.p12 + row) % P.p13; }
    ob = P.p11 + r * L;
  }
  for (var q = 0u; q < 4u; q++) {
    let i = li + q * 256u;
    if (i < L) {
      if ((fl & 8u) != 0u) { D[ob + i] = v[q]; } else { X[ob + i] = v[q]; }
    }
  }
}
"#;

/// Quantize-dequantize simulation, in place. One thread per element: the lanes of a block (16 or
/// 32, never more than a subgroup) take its largest magnitude by shuffles; with smaller subgroups the
/// first thread of each block does the whole block. Dispatch n_blocks * block threads.
/// p0=n_blocks p1=block p2=mode(0 fp8 pow2 / 1 fp4 e8m0 / 2 fp4 e4m3-scale) p3=x_off
/// p4=row_len p5=row_stride (blocks never straddle rows)
const QUANT_SIM: &str = r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;

fn qs_scale(amax: f32, mode: u32) -> f32 {
  if (mode == 0u) { return pow2_ceil(max(amax, 1e-4) / 448.0); }
  if (mode == 1u) { return pow2_ceil(max(amax, 6.0 * 1.1754944e-38) / 6.0); }
  return to_e4m3(max(amax, 6.0 / 512.0) / 6.0);
}

fn qs_one(v: f32, s: f32, mode: u32) -> f32 {
  if (mode == 0u) { return to_e4m3(clamp(v / s, -448.0, 448.0)) * s; }
  return to_e2m1(clamp(v / s, -6.0, 6.0)) * s;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(subgroup_size) ss: u32) {
  let t = gid.x + gid.y * 65535u * 256u;
  let bs = P.p1;
  let bi = t / bs;
  let live = bi < P.p0;
  let per_row = P.p4 / bs;
  let base = P.p3 + (min(bi, P.p0 - 1u) / per_row) * P.p5 + (min(bi, P.p0 - 1u) % per_row) * bs;
  let mode = P.p2;
  if (ss >= bs) {
    let at = base + t % bs;
    var v = 0.0;
    if (live) { v = X[at]; }
    var amax = abs(v);
    for (var o = 1u; o < bs; o <<= 1u) { amax = max(amax, subgroupShuffleXor(amax, o)); }
    if (live) { X[at] = qs_one(v, qs_scale(amax, mode), mode); }
    return;
  }
  if (!live || t % bs != 0u) { return; }
  var amax = 0.0;
  for (var i = 0u; i < bs; i++) { amax = max(amax, abs(X[base + i])); }
  let s = qs_scale(amax, mode);
  for (var i = 0u; i < bs; i++) { X[base + i] = qs_one(X[base + i], s, mode); }
}
"#;

/// vec4s of H that one workgroup of `hc_dot` reduces.
pub const HC_PART: u32 = 128;

/// Workgroups of `hc_dot` along z, each taking 24 / HC_DOT_SPLIT rows of Wf.
pub const HC_DOT_SPLIT: u32 = 4;

/// First half of the hyper-connection mixes: the products of H with each row of Wf (and H with
/// itself), in parts so that many workgroups share the work.
/// H [M][hc*D] ; Wf [24][hc*D] ; out PART [M][parts][25], parts = hc*D/4/HC_PART rounded up
/// p0=hc*D p1=token offset into H; dispatch (parts, M, HC_DOT_SPLIT): z = which 6 rows of Wf
/// (z = 0 also H with itself)
const HC_DOT: &str = r#"
@group(0) @binding(0) var<storage, read> H: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> Wf: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> PART: array<f32>;
var<workgroup> red: array<array<f32, 32>, 7>;

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>,
        @builtin(subgroup_invocation_id) sl: u32, @builtin(subgroup_size) ss: u32) {
  let L = P.p0 / 4u;
  let i = wid.x * 128u + li;
  let o0 = wid.z * 6u;
  var acc: array<f32, 7>;
  for (var o = 0u; o < 7u; o++) { acc[o] = 0.0; }
  if (i < L) {
    let x = H[(P.p1 + wid.y) * L + i];
    acc[6] = dot(x, x);
    for (var o = 0u; o < 6u; o++) { acc[o] = dot(x, Wf[(o0 + o) * L + i]); }
  }
  for (var o = 0u; o < 7u; o++) {
    let v = subgroupAdd(acc[o]);
    if (sl == 0u) { red[o][li / ss] = v; }
  }
  workgroupBarrier();
  let row = (wid.y * nw.x + wid.x) * 25u;
  if (li < 6u || (li == 6u && wid.z == 0u)) {
    var t = 0.0;
    for (var s = 0u; s < 128u / ss; s++) { t += red[li][s]; }
    PART[row + select(o0 + li, 24u, li == 6u)] = t;
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
var<workgroup> cm: array<f32, 16>;

// Thread li < 16 holds comb entry (j, k) = (li / 4, li % 4); 16..23 the pre and post mixes. Every
// sum is taken in the same order as one thread doing it all would.
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(subgroup_size) ss: u32) {
  if (li < 25u) {
    // loads issued eight at a time, added in order
    var t = 0.0;
    let pb = wid.x * P.p7 * 25u + li;
    var i = 0u;
    for (; i + 8u <= P.p7; i += 8u) {
      var v: array<f32, 8>;
      for (var q = 0u; q < 8u; q++) { v[q] = PART[pb + (i + q) * 25u]; }
      for (var q = 0u; q < 8u; q++) { t += v[q]; }
    }
    for (; i < P.p7; i++) { t += PART[pb + i * 25u]; }
    tot[li] = t;
  }
  workgroupBarrier();
  let eps = fbits(P.p4);
  let r = inverseSqrt(tot[24] / f32(P.p0) + fbits(P.p3));
  let so = P.p5;
  let mo = P.p6 + wid.x * 24u;
  let j = li / 4u;
  let k = li % 4u;
  if (li >= 16u && li < 20u) {
    let q = li - 16u;
    MIX[mo + q] = 1.0 / (1.0 + exp(-(tot[q] * r * SB[so] + SB[so + 3u + q]))) + eps;
  } else if (li >= 20u && li < 24u) {
    let q = li - 20u;
    MIX[mo + 4u + q] = 2.0 / (1.0 + exp(-(tot[4u + q] * r * SB[so + 1u] + SB[so + 3u + 4u + q])));
  }
  // Lanes of one subgroup trade comb entries by shuffles (this path needs subgroups of 16 or more;
  // the workgroup is one subgroup then, and every lane runs the same steps).
  if (ss >= 16u) {
    let e = min(li, 15u);
    var c = tot[8u + e] * r * SB[so + 2u] + SB[so + 3u + 8u + e];
    let rb = j % 4u * 4u;
    let mx = max(max(max(max(NEG_INF, subgroupShuffle(c, rb)), subgroupShuffle(c, rb + 1u)), subgroupShuffle(c, rb + 2u)), subgroupShuffle(c, rb + 3u));
    c = exp(c - mx);
    c = c / (0.0 + subgroupShuffle(c, rb) + subgroupShuffle(c, rb + 1u) + subgroupShuffle(c, rb + 2u) + subgroupShuffle(c, rb + 3u)) + eps;
    c /= subgroupShuffle(c, k) + subgroupShuffle(c, 4u + k) + subgroupShuffle(c, 8u + k) + subgroupShuffle(c, 12u + k) + eps;
    for (var it = 1u; it < P.p2; it++) {
      c /= subgroupShuffle(c, rb) + subgroupShuffle(c, rb + 1u) + subgroupShuffle(c, rb + 2u) + subgroupShuffle(c, rb + 3u) + eps;
      c /= subgroupShuffle(c, k) + subgroupShuffle(c, 4u + k) + subgroupShuffle(c, 8u + k) + subgroupShuffle(c, 12u + k) + eps;
    }
    if (li < 16u) { MIX[mo + 8u + li] = c; }
    return;
  }
  var c = 0.0;
  if (li < 16u) { c = tot[8u + li] * r * SB[so + 2u] + SB[so + 3u + 8u + li]; cm[li] = c; }
  workgroupBarrier();
  if (li < 16u) {
    let rb = j * 4u;
    let mx = max(max(max(max(NEG_INF, cm[rb]), cm[rb + 1u]), cm[rb + 2u]), cm[rb + 3u]);
    c = exp(c - mx);
  }
  workgroupBarrier();
  if (li < 16u) { cm[li] = c; }
  workgroupBarrier();
  if (li < 16u) {
    let rb = j * 4u;
    let sum = 0.0 + cm[rb] + cm[rb + 1u] + cm[rb + 2u] + cm[rb + 3u];
    c = c / sum + eps;
  }
  workgroupBarrier();
  if (li < 16u) { cm[li] = c; }
  workgroupBarrier();
  if (li < 16u) { c /= cm[k] + cm[4u + k] + cm[8u + k] + cm[12u + k] + eps; }
  for (var it = 1u; it < P.p2; it++) {
    workgroupBarrier();
    if (li < 16u) { cm[li] = c; }
    workgroupBarrier();
    if (li < 16u) { let rb = j * 4u; c /= cm[rb] + cm[rb + 1u] + cm[rb + 2u] + cm[rb + 3u] + eps; }
    workgroupBarrier();
    if (li < 16u) { cm[li] = c; }
    workgroupBarrier();
    if (li < 16u) { c /= cm[k] + cm[4u + k] + cm[8u + k] + cm[12u + k] + eps; }
  }
  if (li < 16u) { MIX[mo + 8u + li] = c; }
}
"#;

/// x = rmsnorm(sum_j pre[j] * H[j]) * w, one token per workgroup.
/// p0=D p1=identity_pre p2=eps p3=token_off(first token) p4=out_off p5=mix_off
const HC_PRE_NORM: &str = r#"
@group(0) @binding(0) var<storage, read> H: array<f32>;
@group(0) @binding(1) var<storage, read> MIX: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 128>;

@compute @workgroup_size(1024)
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
  for (var i = li; i < D; i += 1024u) {
    let v = pre.x * H[hb + i] + pre.y * H[hb + D + i] + pre.z * H[hb + 2u * D + i] + pre.w * H[hb + 3u * D + i];
    Y[yb + i] = v;
    s += v * v;
  }
  s = subgroupAdd(s);
  if (sl == 0u) { red[li / ss] = s; }
  workgroupBarrier();
  var tot = 0.0;
  for (var i = 0u; i < 1024u / ss; i++) { tot += red[i]; }
  let r = inverseSqrt(tot / f32(D) + fbits(P.p2));
  for (var i = li; i < D; i += 1024u) { Y[yb + i] = Y[yb + i] * r * Wt[i]; }
}
"#;

/// H[m][k] = post[k] * y + sum_j comb[j][k] H[m][j]; y = Y[m] (+ SH[m] + sum_s O6[m*ns+s])
/// p0=D p1=M p2=n_slots(0 = attention output only) p3=mix_off p4=token offset into H / MIX
/// p5=slots (M = 1 only) read from O6C[s] instead of O6
const HC_POST: &str = r#"
@group(0) @binding(0) var<storage, read_write> H: array<f32>;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<storage, read> O6: array<f32>;
@group(0) @binding(3) var<storage, read> MIX: array<f32>;
@group(0) @binding(4) var<storage, read> O6C: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let D = P.p0;
  let i = gid.x + gid.y * 65535u * 256u;
  if (i >= P.p1 * D) { return; }
  let m = i / D; let d = i % D;
  var y = Y[i];
  // p5 bit s: slot s (one token only) was computed on the CPU and is in O6C
  for (var s = 0u; s < P.p2; s++) {
    var o: f32;
    if (((P.p5 >> s) & 1u) != 0u) { o = O6C[s * D + d]; } else { o = O6[(m * P.p2 + s) * D + d]; }
    y += o;
  }
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
/// One thread per (t, head), IDX_T_PER_WG values of t per workgroup (heads * IDX_T_PER_WG = 256);
/// dispatch (n_cols / IDX_T_PER_WG rounded up, M, 1). The heads are added in order by one thread.
const IDX_SCORES: &str = r#"
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> WP: array<f32>;
@group(0) @binding(2) var<storage, read> Kc: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> CAND: array<u32>;
@group(0) @binding(4) var<storage, read_write> S: array<f32>;
var<workgroup> qs: array<vec4<f32>, 1024>;
var<workgroup> ws: array<f32, 64>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let heads = P.p1; let hd = P.p2; let hv = hd / 4u;
  let m = wid.y;
  // q transposed, [i][h], so the lanes of one t read consecutive vec4s
  for (var e = li; e < heads * hv; e += 256u) {
    let h = e / hv; let i = e % hv;
    let b = m * heads * hd + e * 4u;
    qs[i * heads + h] = vec4<f32>(Q[b], Q[b + 1u], Q[b + 2u], Q[b + 3u]);
  }
  if (li < heads) { ws[li] = WP[m * heads + li] * fbits(P.p5); }
  workgroupBarrier();
  let per = 256u / heads;
  let h = li % heads;
  let t = wid.x * per + li / heads;
  let nvis = (P.p3 + m + 1u) / P.p4;
  var ok = t < nvis && t < P.p0;
  if (ok && P.p6 != 0u) {
    let blk = t / P.p7;
    ok = ((CAND[m * P.p8 + (blk >> 5u)] >> (blk & 31u)) & 1u) != 0u;
  }
  var d = 0.0;
  if (ok) {
    for (var i = 0u; i < hv; i++) { d += dot(qs[i * heads + h], Kc[t * hv + i]); }
  }
  part[li] = d;
  workgroupBarrier();
  if (h == 0u && t < P.p0) {
    var out = NEG_INF;
    if (ok) {
      var acc = 0.0;
      for (var g = 0u; g < heads; g++) { acc += max(part[li + g], 0.0) * ws[g]; }
      out = acc;
    }
    S[m * P.p0 + t] = out;
  }
}
"#;

/// Values of t per workgroup of `idx_scores` for `heads` indexer heads.
pub fn idx_t_per_wg(heads: usize) -> u32 {
    assert!(heads > 0 && 256 % heads == 0 && heads <= 64, "idx_scores needs heads dividing 256, at most 64");
    (256 / heads) as u32
}

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

/// One small workgroup doing arithmetic for p0 iterations and nothing else: run while the GPU would
/// otherwise wait for the CPU, so that the driver keeps the memory at full clock.
const KEEP_BUSY: &str = r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  var a = 1.0 + f32(lid.x) * 1e-3;
  for (var i = 0u; i < P.p0; i++) { a = fma(a, 0.9999999, 1e-7); }
  // never true, but the compiler cannot know: keeps the loop
  if (a == -1.0) { X[lid.x] = a; }
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
/// p0=n_slots p1=W p2=T(cidx stride) p3=heads p4=hd p5=scale p6=cidx_off; with p10 != 0, q gets RoPE
/// as it is read (trailing 2*p8 values, table RT at p7 floats, position p9 + token; 2*p8 a multiple
/// of 4)
const ATTN_SCORES: &str = r#"
@group(0) @binding(0) var<storage, read> Q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> KW: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> KC: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> WIDX: array<i32>;
@group(0) @binding(4) var<storage, read> CIDX: array<i32>;
@group(0) @binding(5) var<storage, read_write> S: array<f32>;
@group(0) @binding(6) var<storage, read> RT: array<f32>;
var<workgroup> qt: array<array<vec4<f32>, 33>, 16>;
var<workgroup> kt: array<array<vec4<f32>, 33>, 16>;
var<workgroup> kidx: array<i32, 16>;
var<workgroup> ksrc: array<u32, 16>;

// 16 heads x 16 slots per workgroup, hd in steps of 128 (hd a multiple of 16), as vec4s. Four sums
// per thread (one per vec4 lane), so the adds do not wait on each other.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let ns = P.p0; let W = P.p1; let heads = P.p3; let hd = P.p4;
  let hv = hd / 4u;
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
  var acc = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  for (var c0 = 0u; c0 < hv; c0 += 32u) {
    let cn = min(32u, hv - c0);
    var qv: array<vec4<f32>, 2>;
    var kv: array<vec4<f32>, 2>;
    for (var q = 0u; q < 2u; q++) {
      let e = q * 256u + li;
      let r = e / 32u; let c = e % 32u;
      qv[q] = vec4<f32>(0.0); kv[q] = vec4<f32>(0.0);
      if (c < cn) {
        qv[q] = Q[(m * heads + h0 + r) * hv + c0 + c];
        let e = 4u * (c0 + c);
        let r0 = hd - 2u * P.p8;
        if (P.p10 != 0u && e >= r0) {
          // RoPE of q as ROPE does it, at position p9 + token
          let half = P.p8;
          let k0 = (e - r0) / 2u;
          let tb = P.p7 + ((P.p9 + m) * half + k0) * 2u;
          let ca = RT[tb]; let sa = RT[tb + 1u]; let cb = RT[tb + 2u]; let sb = RT[tb + 3u];
          let o = qv[q];
          qv[q] = vec4<f32>(o.x * ca - o.y * sa, o.x * sa + o.y * ca, o.z * cb - o.w * sb, o.z * sb + o.w * cb);
        }
        let ix = kidx[r];
        if (ix >= 0) {
          if (ksrc[r] == 0u) { kv[q] = KW[u32(ix) * hv + c0 + c]; } else { kv[q] = KC[u32(ix) * hv + c0 + c]; }
        }
      }
    }
    for (var q = 0u; q < 2u; q++) {
      let e = q * 256u + li;
      qt[e / 32u][e % 32u] = qv[q];
      kt[e / 32u][e % 32u] = kv[q];
    }
    workgroupBarrier();
    // cn is even (hd a multiple of 16)
    for (var c = 0u; c < cn; c += 2u) {
      acc = fma(qt[ty][c], kt[tx][c], acc);
      acc2 = fma(qt[ty][c + 1u], kt[tx][c + 1u], acc2);
    }
    workgroupBarrier();
  }
  let j = j0 + tx;
  if (j < ns) {
    let a = acc + acc2;
    var v = ((a.x + a.y) + (a.z + a.w)) * fbits(P.p5);
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
var<workgroup> pt: array<array<f32, 65>, 16>;
var<workgroup> vt: array<array<f32, 17>, 64>;

// 16 heads x 16 dims per workgroup, slots in steps of 64; each step's loads issued before any is
// stored, two sums per thread.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {
  let ns = P.p0; let W = P.p1; let heads = P.p3; let hd = P.p4;
  let m = wid.z;
  let d0 = wid.x * 16u; let h0 = wid.y * 16u;
  let ty = li / 16u; let tx = li % 16u;
  var acc = 0.0;
  var acc2 = 0.0;
  for (var j0 = 0u; j0 < ns; j0 += 64u) {
    var pv: array<f32, 4>;
    var vv: array<f32, 4>;
    for (var q = 0u; q < 4u; q++) {
      let e = q * 256u + li;
      let ph = e / 64u; let pj = j0 + e % 64u;
      pv[q] = 0.0;
      if (pj < ns) { pv[q] = PR[(m * heads + h0 + ph) * ns + pj]; }
      let j = j0 + e / 16u;
      vv[q] = 0.0;
      if (j < ns) {
        var ix = -1; var src = 0u;
        if (j < W) { ix = WIDX[m * W + j]; } else { ix = CIDX[P.p6 + m * P.p2 + (j - W)]; src = 1u; }
        if (ix >= 0) {
          if (src == 0u) { vv[q] = KW[u32(ix) * hd + d0 + e % 16u]; } else { vv[q] = KC[u32(ix) * hd + d0 + e % 16u]; }
        }
      }
    }
    for (var q = 0u; q < 4u; q++) {
      let e = q * 256u + li;
      pt[e / 64u][e % 64u] = pv[q];
      vt[e / 16u][e % 16u] = vv[q];
    }
    workgroupBarrier();
    for (var jj = 0u; jj < 64u; jj += 2u) {
      acc += pt[ty][jj] * vt[jj][tx];
      acc2 += pt[ty][jj + 1u] * vt[jj + 1u][tx];
    }
    workgroupBarrier();
  }
  O[(m * heads + h0 + ty) * hd + d0 + tx] = acc + acc2;
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
    if let Some(c) = COMPACT.iter().find(|c| c.0 == ty) {
        c.2
    } else if ty == crate::gguf::GGML_MXFP4 {
        ["moe_gate_up_1", "moe_gate_up_8", "moe_down_1", "moe_down_8"]
    } else if crate::quants::wide(ty) {
        ["moe_gate_up_wide_1", "moe_gate_up_wide_8", "moe_down_wide_1", "moe_down_wide_8"]
    } else {
        ["moe_gate_up_lut_1", "moe_gate_up_lut_8", "moe_down_lut_1", "moe_down_lut_8"]
    }
}

pub fn all() -> Vec<(&'static str, String)> {
    let mut v = vec![
        ("mm_q8_8", k(&MM_Q8.replace("TM", "8u"))),
        ("mm_q8_1", k(MM_Q8_1)),
        ("mm_q8_1_glu", k(MM_Q8_1_GLU)),
        ("mm_q8_1_blocks", k(&MM_Q8.replace("TM", "1u"))),
        ("mm_f32_1", k(&MM_F32.replace("TM", "1u"))),
        ("mm_f32_8", k(&MM_F32.replace("TM", "8u"))),
        ("mm_f32_1_pair", k(MM_F32_1_PAIR)),
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
        ("row_post", k(ROW_POST)),
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
        ("keep_busy", k(KEEP_BUSY)),
        ("topk", k(TOPK)),
        ("attn_scores", k(ATTN_SCORES)),
        ("attn_softmax", k(ATTN_SOFTMAX)),
        ("attn_pv", k(ATTN_PV)),
    ];
    v.shrink_to_fit();
    v
}
