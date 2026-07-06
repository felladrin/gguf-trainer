// WGSL kernel source generators plus the shared codegen/dispatch helpers, split
// out of webgpu.ts so "which kernels exist and what WGSL they emit" lives apart
// from "how the backend runs them". Every function is a pure string/number
// builder with no device state; shapes and hyperparameters are baked into the
// source, so each kernel compiles once and is cached by its source string.
// Consumed by webgpu.ts (the backend) and the GPU optimizers (muon_gpu, adamw_gpu).

export function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}

/** WebGPU's per-dimension workgroup-count cap (spec default, and what Metal/M1 reports). */
export const MAX_WG = 65535;

/**
 * Split a 1-D per-element dispatch of `n` items (256 threads/workgroup) into a
 * 2-D workgroup grid that stays under MAX_WG per dimension. `roww` (= gridX·256)
 * is baked into the kernel so it rebuilds the flat index as `gid.y*roww + gid.x`.
 * A flat ceilDiv(n,256) grid overflows past ~16.7M elements — reached by SwiGLU
 * on [T,ffn], cross-entropy on [T,V], or a large embedding at long context /
 * large vocab. Below the cap this is identical to a 1-D dispatch (gridY=1).
 */
export function grid2D(n: number): { x: number; y: number; roww: number } {
  const total = Math.max(1, ceilDiv(n, 256));
  const x = Math.min(total, MAX_WG);
  return { x, y: ceilDiv(total, x), roww: x * 256 };
}

/** 2-D grid for one-workgroup-per-row kernels (rmsNorm): row = wg.y*gridX + wg.x. */
export function gridRows(rows: number): { x: number; y: number } {
  const x = Math.min(Math.max(1, rows), MAX_WG);
  return { x, y: ceilDiv(Math.max(1, rows), x) };
}

/** Format a JS number as a WGSL f32 literal. */
export function f32lit(v: number): string {
  const s = String(v);
  return /[.e]/i.test(s) ? s : `${s}.0`;
}

// --- WGSL kernel sources --------------------------------------------------------

export function bindF32(i: number, name: string, access: "read" | "read_write"): string {
  return `@group(0) @binding(${i}) var<storage, ${access}> ${name}: array<f32>;`;
}
function bindU32(i: number, name: string): string {
  return `@group(0) @binding(${i}) var<storage, read> ${name}: array<u32>;`;
}

// Register-tiled GEMM block dims. A workgroup computes a BM×BN output tile in
// BK-deep K steps; each of its (BM/TM)×(BN/TN) threads owns a TM×TN micro-tile
// of accumulators, so every value staged into shared memory is reused across TM
// (or TN) MACs instead of one — the arithmetic-intensity lever that lifts this
// off the "SMEM-tiled, 1 output/thread" rung. Single-sourced with the dispatch
// grid (gemm/prepareGemm below) so block size and workgroup count never drift.
export const GEMM_BM = 64, GEMM_BN = 64, GEMM_BK = 8, GEMM_TM = 4, GEMM_TN = 4;
export const GEMM_WG = (GEMM_BM / GEMM_TM) * (GEMM_BN / GEMM_TN); // threads/workgroup (256)

/**
 * Register-tiled GEMM: C[M,N] = sum_k A'[m,k] * B'[k,n], optionally accumulating
 * into C. Transpose flavors cover the forward and both backward products of
 * `linear` without ever materializing a transposed matrix:
 *   NT: A stored [M,K], B stored [N,K]   (y = x·Wᵀ)
 *   NN: A stored [M,K], B stored [K,N]   (dX = dY·W)
 *   TN: A stored [K,M], B stored [K,N]   (dW = dYᵀ·x)
 * The TM×TN accumulator tile and its A/B fragments are UNROLLED into named
 * scalars (acc0_0 … acc{TM-1}_{TN-1}): a WGSL array indexed by loop variables is
 * not promoted to registers, so a rolled version spills the accumulators to
 * memory and runs *slower* than the 1-output/thread kernel (measured ~3× on
 * M1 Max). Bounds are guarded per load and per store, so M/N/K need not be
 * multiples of the block dims (the parity suite covers tiny, odd, multi-block).
 */
export function srcGemm(
  kind: "NT" | "NN" | "TN",
  accum: boolean,
  M: number,
  N: number,
  K: number,
  f16 = false,
): string {
  const [BM, BN, BK, TM, TN, WG] = [GEMM_BM, GEMM_BN, GEMM_BK, GEMM_TM, GEMM_TN, GEMM_WG];
  // gr = global row (m), gc = global col (n), gk = global k index.
  const aLoad = kind === "TN" ? "AB[gk * M + gr]" : "AB[gr * K + gk]";
  const bLoad = kind === "NT" ? "BB[gc * K + gk]" : "BB[gk * N + gc]";
  // Mixed precision: stage the tiles as f16 and multiply in f16 (2x ALU on
  // hardware with packed f16, e.g. Strix Halo), but ACCUMULATE in f32 so the
  // K-length reduction keeps full precision. Operands (activations/weights/grads)
  // round to f16 for the multiply only; buffers, grads, and the optimizer stay
  // f32, so no loss scaling is needed. sh = shared-tile scalar type.
  const sh = f16 ? "f16" : "f32";
  const toSh = f16 ? "f16(v)" : "v";
  const prod = (i: number, j: number) => f16 ? `f32(a${i} * b${j})` : `a${i} * b${j}`;
  // Unroll the tile so every accumulator/fragment is a compile-time-named
  // scalar (stays in registers) rather than a dynamically-indexed array.
  let decl = "", fragA = "", fragB = "", macs = "", stores = "";
  for (let i = 0; i < TM; i++) fragA += `      let a${i} = As[(tRow + ${i}u) * ${BK}u + kc];\n`;
  for (let j = 0; j < TN; j++) fragB += `      let b${j} = Bs[kc * ${BN}u + tCol + ${j}u];\n`;
  for (let i = 0; i < TM; i++) {
    for (let j = 0; j < TN; j++) {
      decl += `  var acc${i}_${j} = 0.0;\n`;
      macs += `      acc${i}_${j} = acc${i}_${j} + ${prod(i, j)};\n`;
      const idx = `(blockRow + tRow + ${i}u) * N + (blockCol + tCol + ${j}u)`;
      const rhs = accum ? `CB[ci] + acc${i}_${j}` : `acc${i}_${j}`;
      stores += `  if (blockRow + tRow + ${i}u < M && blockCol + tCol + ${j}u < N) ` +
        `{ let ci = ${idx}; CB[ci] = ${rhs}; }\n`;
    }
  }
  return `${f16 ? "enable f16;\n" : ""}
${bindF32(0, "AB", "read")}
${bindF32(1, "BB", "read")}
${bindF32(2, "CB", "read_write")}
const M: u32 = ${M}u; const N: u32 = ${N}u; const K: u32 = ${K}u;
var<workgroup> As: array<${sh}, ${BM * BK}>;
var<workgroup> Bs: array<${sh}, ${BK * BN}>;
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
  let blockRow = wg.y * ${BM}u;
  let blockCol = wg.x * ${BN}u;
  let tRow = (lidx / ${BN / TN}u) * ${TM}u; // this thread's first row in the tile
  let tCol = (lidx % ${BN / TN}u) * ${TN}u; // this thread's first col in the tile
${decl}  var kk = 0u;
  loop {
    if (kk >= K) { break; }
    // Cooperatively stage the A and B tiles (grid-strided).
    for (var t = lidx; t < ${BM * BK}u; t += ${WG}u) {
      let r = t / ${BK}u;
      let kc = t % ${BK}u;
      let gr = blockRow + r;
      let gk = kk + kc;
      var v = 0.0;
      if (gr < M && gk < K) { v = ${aLoad}; }
      As[r * ${BK}u + kc] = ${toSh};
    }
    for (var t = lidx; t < ${BK * BN}u; t += ${WG}u) {
      let kc = t / ${BN}u;
      let c = t % ${BN}u;
      let gk = kk + kc;
      let gc = blockCol + c;
      var v = 0.0;
      if (gk < K && gc < N) { v = ${bLoad}; }
      Bs[kc * ${BN}u + c] = ${toSh};
    }
    workgroupBarrier();
    for (var kc = 0u; kc < ${BK}u; kc++) {
${fragA}${fragB}${macs}    }
    workgroupBarrier();
    kk = kk + ${BK}u;
  }
${stores}}`;
}

/** Elementwise kernel over N threads; body sees index `i`. */
export function srcElementwise(bindings: string[], n: number, body: string): string {
  return `
${bindings.join("\n")}
const N: u32 = ${n}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * ${grid2D(n).roww}u + gid.x;
  if (i >= N) { return; }
  ${body}
}`;
}

/**
 * RMSNorm forward, one workgroup per row: parallel-reduce the mean square,
 * stash 1/sqrt(ms+eps) for backward, then normalize+scale. Also serves
 * rmsNormHeads: a [T, H*hd] tensor per-head-normalized is exactly rmsNorm on
 * a [T*H, hd] view because rows are contiguous.
 */
export function srcRmsNormFwd(rows: number, d: number, eps: number): string {
  return `
${bindF32(0, "XB", "read")}
${bindF32(1, "WB", "read")}
${bindF32(2, "YB", "read_write")}
${bindF32(3, "RINV", "read_write")}
const D: u32 = ${d}u; const ROWS: u32 = ${rows}u; const EPS: f32 = ${f32lit(eps)};
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let row = wg.y * ${gridRows(rows).x}u + wg.x;
  if (row >= ROWS) { return; } // whole workgroup: row is uniform, no barrier reached
  let lid = li.x;
  var s = 0.0;
  for (var j = lid; j < D; j += 128u) { let v = XB[row * D + j]; s = s + v * v; }
  red[lid] = s;
  workgroupBarrier();
  var off = 64u;
  loop {
    if (lid < off) { red[lid] = red[lid] + red[lid + off]; }
    workgroupBarrier();
    if (off == 1u) { break; }
    off = off / 2u;
  }
  let r = 1.0 / sqrt(red[0] / f32(D) + EPS);
  if (lid == 0u) { RINV[row] = r; }
  for (var j = lid; j < D; j += 128u) { YB[row * D + j] = XB[row * D + j] * r * WB[j]; }
}`;
}

/** RMSNorm backward wrt x. Same row-parallel shape as forward; S = Σ g·w·x. */
export function srcRmsNormBwdX(rows: number, d: number): string {
  return `
${bindF32(0, "XB", "read")}
${bindF32(1, "WB", "read")}
${bindF32(2, "GB", "read")}
${bindF32(3, "RINV", "read")}
${bindF32(4, "DX", "read_write")}
const D: u32 = ${d}u; const ROWS: u32 = ${rows}u;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let row = wg.y * ${gridRows(rows).x}u + wg.x;
  if (row >= ROWS) { return; } // whole workgroup: row is uniform, no barrier reached
  let lid = li.x;
  var s = 0.0;
  for (var j = lid; j < D; j += 128u) {
    s = s + GB[row * D + j] * WB[j] * XB[row * D + j];
  }
  red[lid] = s;
  workgroupBarrier();
  var off = 64u;
  loop {
    if (lid < off) { red[lid] = red[lid] + red[lid + off]; }
    workgroupBarrier();
    if (off == 1u) { break; }
    off = off / 2u;
  }
  let bigS = red[0];
  let r = RINV[row];
  for (var j = lid; j < D; j += 128u) {
    let g = GB[row * D + j];
    DX[row * D + j] = DX[row * D + j] + WB[j] * r * g -
      (XB[row * D + j] / f32(D)) * r * r * r * bigS;
  }
}`;
}

/** RMSNorm backward wrt weight: thread per column j, reduce over rows. */
export function srcRmsNormBwdW(rows: number, d: number): string {
  return `
${bindF32(0, "XB", "read")}
${bindF32(1, "GB", "read")}
${bindF32(2, "RINV", "read")}
${bindF32(3, "DW", "read_write")}
const ROWS: u32 = ${rows}u; const D: u32 = ${d}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= D) { return; }
  var a = 0.0;
  for (var row = 0u; row < ROWS; row++) {
    a = a + GB[row * D + j] * XB[row * D + j] * RINV[row];
  }
  DW[j] = DW[j] + a;
}`;
}

export function srcEmbeddingFwd(T: number, d: number): string {
  return `
${bindF32(0, "WB", "read")}
${bindU32(1, "IDS")}
${bindF32(2, "YB", "read_write")}
const N: u32 = ${T * d}u; const D: u32 = ${d}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * ${grid2D(T * d).roww}u + gid.x;
  if (i >= N) { return; }
  let t = i / D;
  let j = i % D;
  YB[i] = WB[IDS[t] * D + j];
}`;
}

/**
 * Embedding backward: thread per (vocab row, column) scanning the id list.
 * Deterministic and race-free — repeated ids just extend the scan — at the
 * cost of V*d*T work, which is trivial at trainable-here scales.
 */
export function srcEmbeddingBwd(T: number, d: number, V: number): string {
  return `
${bindU32(0, "IDS")}
${bindF32(1, "GB", "read")}
${bindF32(2, "DW", "read_write")}
const N: u32 = ${V * d}u; const D: u32 = ${d}u; const T: u32 = ${T}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * ${grid2D(V * d).roww}u + gid.x;
  if (i >= N) { return; }
  let v = i / D;
  let j = i % D;
  var a = 0.0;
  for (var t = 0u; t < T; t++) {
    if (IDS[t] == v) { a = a + GB[t * D + j]; }
  }
  DW[i] = DW[i] + a;
}`;
}

export function srcRope(
  T: number,
  H: number,
  hd: number,
  base: number,
  posOffset: number,
  backward: boolean,
): string {
  const half = hd / 2;
  // Backward is the transposed rotation of forward.
  const body = backward
    ? `DX[b + j] = DX[b + j] + c * g0 + s * g1;
  DX[b + j + HALF] = DX[b + j + HALF] - s * g0 + c * g1;`
    : `DX[b + j] = c * g0 - s * g1;
  DX[b + j + HALF] = s * g0 + c * g1;`;
  return `
${bindF32(0, "GB", "read")}
${bindF32(1, "DX", "read_write")}
const N: u32 = ${T * H * half}u; const H: u32 = ${H}u; const HD: u32 = ${hd}u;
const HALF: u32 = ${half}u; const ROW: u32 = ${H * hd}u;
const BASE: f32 = ${f32lit(base)}; const POS: f32 = ${f32lit(posOffset)};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * ${grid2D(T * H * half).roww}u + gid.x;
  if (i >= N) { return; }
  let j = i % HALF;
  let th = i / HALF;
  let h = th % H;
  let t = th / H;
  let b = t * ROW + h * HD;
  let freq = pow(BASE, -2.0 * f32(j) / f32(HD));
  let ang = (f32(t) + POS) * freq;
  let c = cos(ang);
  let s = sin(ang);
  let g0 = GB[b + j];
  let g1 = GB[b + j + HALF];
  ${body}
}`;
}

export interface AttnDims {
  T: number;
  Hq: number;
  Hkv: number;
  hd: number;
}

function attnConsts(a: AttnDims): string {
  const scale = 1 / Math.sqrt(a.hd);
  return `const T: u32 = ${a.T}u; const HQ: u32 = ${a.Hq}u; const HKV: u32 = ${a.Hkv}u;
const HD: u32 = ${a.hd}u; const GROUP: u32 = ${a.Hq / a.Hkv}u;
const QS: u32 = ${a.Hq * a.hd}u; const KS: u32 = ${a.Hkv * a.hd}u;
const SCALE: f32 = ${f32lit(scale)};`;
}

// --- Attention, small-T regime: materialized [Hq,T,T] probabilities ----------
// Two attention implementations coexist on purpose. These five kernels write
// the full probability (and dScore) matrix but parallelize over hd as well —
// T·Hq·hd threads with O(T)-deep loops. The flash kernels below never touch a
// [Hq,T,T] buffer but run one thread per (head, query row) — Hq·T threads with
// O(T·hd)-deep loops — which underoccupies the GPU and lengthens the serial
// chain at small T. Measured on M1 Max (fwd+bwd+sync, Hq=4 Hkv=2 hd=32, same
// harness on both): flash-only was 1.4–2× SLOWER than these below T≈1536 and
// only wins from T≈2048 up. attention() picks per problem size; the crossover
// lives in WebGPUBackend.attnFlashMinT.

/** Causal softmax probabilities per (head, query): P[h,t,s] for s<=t. */
export function srcAttnProbs(a: AttnDims): string {
  return `
${bindF32(0, "QB", "read")}
${bindF32(1, "KB", "read")}
${bindF32(2, "PB", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= HQ * T) { return; }
  let h = i / T;
  let t = i % T;
  let kv = h / GROUP;
  var maxS = -3.0e38;
  for (var si = 0u; si <= t; si++) {
    var dot = 0.0;
    for (var d = 0u; d < HD; d++) { dot = dot + QB[t * QS + h * HD + d] * KB[si * KS + kv * HD + d]; }
    maxS = max(maxS, dot * SCALE);
  }
  var sum = 0.0;
  for (var si = 0u; si <= t; si++) {
    var dot = 0.0;
    for (var d = 0u; d < HD; d++) { dot = dot + QB[t * QS + h * HD + d] * KB[si * KS + kv * HD + d]; }
    let e = exp(dot * SCALE - maxS);
    PB[(h * T + t) * T + si] = e;
    sum = sum + e;
  }
  for (var si = 0u; si <= t; si++) {
    PB[(h * T + t) * T + si] = PB[(h * T + t) * T + si] / sum;
  }
}`;
}

/** out[t,h,d] = Σ_{s<=t} P[h,t,s]·V[s,kv,d] */
export function srcAttnOut(a: AttnDims): string {
  return `
${bindF32(0, "PB", "read")}
${bindF32(1, "VB", "read")}
${bindF32(2, "YB", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= T * HQ * HD) { return; }
  let d = i % HD;
  let th = i / HD;
  let h = th % HQ;
  let t = th / HQ;
  let kv = h / GROUP;
  var a = 0.0;
  for (var si = 0u; si <= t; si++) {
    a = a + PB[(h * T + t) * T + si] * VB[si * KS + kv * HD + d];
  }
  YB[t * QS + h * HD + d] = a;
}`;
}

/**
 * Softmax backward to scores: dS[h,t,s] = P·(dP − Σ P·dP)·scale, where
 * dP[s] = Σ_d dOut[t,h,d]·V[s,kv,d]. dP is recomputed in the second loop
 * instead of stored — avoids per-thread arrays, and the extra T·hd multiplies
 * are cheaper than the register/spill cost at these sizes.
 */
export function srcAttnDScore(a: AttnDims): string {
  return `
${bindF32(0, "GB", "read")}
${bindF32(1, "VB", "read")}
${bindF32(2, "PB", "read")}
${bindF32(3, "DS", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= HQ * T) { return; }
  let h = i / T;
  let t = i % T;
  let kv = h / GROUP;
  var dot = 0.0;
  for (var si = 0u; si <= t; si++) {
    var dp = 0.0;
    for (var d = 0u; d < HD; d++) { dp = dp + GB[t * QS + h * HD + d] * VB[si * KS + kv * HD + d]; }
    dot = dot + PB[(h * T + t) * T + si] * dp;
  }
  for (var si = 0u; si <= t; si++) {
    var dp = 0.0;
    for (var d = 0u; d < HD; d++) { dp = dp + GB[t * QS + h * HD + d] * VB[si * KS + kv * HD + d]; }
    DS[(h * T + t) * T + si] = PB[(h * T + t) * T + si] * (dp - dot) * SCALE;
  }
}`;
}

/** dQ[t,h,d] += Σ_{s<=t} dS[h,t,s]·K[s,kv,d] */
export function srcAttnDq(a: AttnDims): string {
  return `
${bindF32(0, "DS", "read")}
${bindF32(1, "KB", "read")}
${bindF32(2, "DQ", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= T * HQ * HD) { return; }
  let d = i % HD;
  let th = i / HD;
  let h = th % HQ;
  let t = th / HQ;
  let kv = h / GROUP;
  var a = 0.0;
  for (var si = 0u; si <= t; si++) {
    a = a + DS[(h * T + t) * T + si] * KB[si * KS + kv * HD + d];
  }
  DQ[t * QS + h * HD + d] = DQ[t * QS + h * HD + d] + a;
}`;
}

/**
 * dK[s,kv,d] += Σ_{h in group} Σ_{t>=s} dS[h,t,s]·Q[t,h,d]. One thread per
 * dK element owns the double reduction, so GQA head-sharing needs no atomics.
 * srcAttnDv is the same shape with (P, dOut) in place of (dS, Q).
 */
export function srcAttnDkv(a: AttnDims, useProbs: boolean): string {
  const lhs = useProbs ? "PB" : "DS";
  return `
${bindF32(0, lhs === "PB" ? "PB" : "DS", "read")}
${bindF32(1, "SRC", "read")}
${bindF32(2, "DST", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= T * HKV * HD) { return; }
  let d = i % HD;
  let skv = i / HD;
  let kv = skv % HKV;
  let si = skv / HKV;
  var a = 0.0;
  for (var h = kv * GROUP; h < kv * GROUP + GROUP; h++) {
    for (var t = si; t < T; t++) {
      a = a + ${lhs}[(h * T + t) * T + si] * SRC[t * QS + h * HD + d];
    }
  }
  DST[si * KS + kv * HD + d] = DST[si * KS + kv * HD + d] + a;
}`;
}

// --- Attention, large-T regime: flash-style, no [Hq,T,T] buffer ---------------

/**
 * Fused causal attention forward, flash-style: one thread per (head, query
 * row) runs an online softmax — running max `m`, running denominator `l`,
 * and an HD-wide output accumulator rescaled by exp(m−mNew) whenever the max
 * moves — so the [Hq,T,T] probability matrix is never materialized. The only
 * side product is the per-row logsumexp (O(Hq·T)), from which backward
 * recomputes any probability as exp(score − LSE). The final m/l equal the
 * two-pass CPU values mathematically; only summation-order float noise
 * differs. HD is baked per pipeline, so the private arrays are static.
 *
 * Deliberately NOT staged through workgroup memory: consecutive threads are
 * consecutive query rows of one head walking keys in lockstep from s=0, so
 * every K/V fetch is already wavefront-uniform and cache-served. A 32-key
 * shared-tile variant measured ~17% slower at T=4096 on M1 Max (barrier
 * overhead, no bandwidth saved). Contrast srcAttnBwdDkv, where thread loop
 * bounds diverge and staging wins.
 */
export function srcAttnFwd(a: AttnDims): string {
  return `
${bindF32(0, "QB", "read")}
${bindF32(1, "KB", "read")}
${bindF32(2, "VB", "read")}
${bindF32(3, "YB", "read_write")}
${bindF32(4, "LSE", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= HQ * T) { return; }
  let h = i / T;
  let t = i % T;
  let kv = h / GROUP;
  var qr: array<f32, HD>;
  var acc: array<f32, HD>;
  for (var d = 0u; d < HD; d++) {
    qr[d] = QB[t * QS + h * HD + d];
    acc[d] = 0.0;
  }
  var m = -3.0e38;
  var l = 0.0;
  for (var s = 0u; s <= t; s++) {
    var dot = 0.0;
    for (var d = 0u; d < HD; d++) { dot = dot + qr[d] * KB[s * KS + kv * HD + d]; }
    let sc = dot * SCALE;
    let mNew = max(m, sc);
    let corr = exp(m - mNew);
    let p = exp(sc - mNew);
    l = l * corr + p;
    for (var d = 0u; d < HD; d++) { acc[d] = acc[d] * corr + p * VB[s * KS + kv * HD + d]; }
    m = mNew;
  }
  for (var d = 0u; d < HD; d++) { YB[t * QS + h * HD + d] = acc[d] / l; }
  LSE[h * T + t] = m + log(l);
}`;
}

/**
 * D[h,t] = Σ_d dOut[t,h,d]·out[t,h,d] — the softmax-backward row constant
 * (equals Σ_s P·dP), precomputed once so neither backward kernel re-derives
 * it inside its O(T) scan.
 */
export function srcAttnBwdD(a: AttnDims): string {
  return `
${bindF32(0, "GB", "read")}
${bindF32(1, "YB", "read")}
${bindF32(2, "DB", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= HQ * T) { return; }
  let h = i / T;
  let t = i % T;
  var s = 0.0;
  for (var d = 0u; d < HD; d++) { s = s + GB[t * QS + h * HD + d] * YB[t * QS + h * HD + d]; }
  DB[h * T + t] = s;
}`;
}

/**
 * dQ[t,h,:] += Σ_{s<=t} dS[t,s]·K[s,kv,:] with dS recomputed on the fly:
 * p = exp(q·k·scale − LSE[h,t]), dP = Σ_d dOut·V[s], dS = p·(dP − D[h,t])·scale.
 * One thread per (head, query row) owns the whole dQ row — each (t,s) score
 * is computed once and reused across HD via the private accumulator, and no
 * atomics are needed. Direct loads for the same reason as srcAttnFwd: the
 * lockstep key scan is already uniform per wavefront (shared-tile variant
 * measured slower).
 */
export function srcAttnBwdDq(a: AttnDims): string {
  return `
${bindF32(0, "QB", "read")}
${bindF32(1, "KB", "read")}
${bindF32(2, "VB", "read")}
${bindF32(3, "GB", "read")}
${bindF32(4, "LSE", "read")}
${bindF32(5, "DB", "read")}
${bindF32(6, "DQ", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= HQ * T) { return; }
  let h = i / T;
  let t = i % T;
  let kv = h / GROUP;
  var qr: array<f32, HD>;
  var go: array<f32, HD>;
  var dq: array<f32, HD>;
  for (var d = 0u; d < HD; d++) {
    qr[d] = QB[t * QS + h * HD + d];
    go[d] = GB[t * QS + h * HD + d];
    dq[d] = 0.0;
  }
  let lse = LSE[h * T + t];
  let dRow = DB[h * T + t];
  for (var s = 0u; s <= t; s++) {
    var dot = 0.0;
    var dp = 0.0;
    for (var d = 0u; d < HD; d++) {
      dot = dot + qr[d] * KB[s * KS + kv * HD + d];
      dp = dp + go[d] * VB[s * KS + kv * HD + d];
    }
    let ds = exp(dot * SCALE - lse) * (dp - dRow) * SCALE;
    for (var d = 0u; d < HD; d++) { dq[d] = dq[d] + ds * KB[s * KS + kv * HD + d]; }
  }
  for (var d = 0u; d < HD; d++) {
    DQ[t * QS + h * HD + d] = DQ[t * QS + h * HD + d] + dq[d];
  }
}`;
}

/**
 * dK[s,kv,:] += Σ_{h in group} Σ_{t>=s} dS[t,s]·Q[t,h,:] and
 * dV[s,kv,:] += Σ_{h in group} Σ_{t>=s} P[t,s]·dOut[t,h,:], fused so each
 * (t,s) score is computed once and serves both. One thread per (key, kv head)
 * owns both gradient rows and reduces over the GQA head group itself, so
 * head-sharing needs no atomics. Workgroup = (kv head, 64-key block); Q/dOut
 * rows and the LSE/D statistics stream through workgroup memory in 32-query
 * tiles per group head. Query tiles start at the block's first key — the
 * causal t>=s means earlier tiles can't contribute to any key in the block.
 * Staging pays off here (unlike srcAttnFwd/srcAttnBwdDq, ~1.8x at T=4096):
 * each thread's scan starts at its own key t=s, so direct loads would be
 * wavefront-divergent; the shared tile re-aligns the whole block to one
 * uniform t range. Shared footprint 2·32·HD+64 f32 ≈ 8.3 KiB at HD=32, under
 * the 16 KiB spec-default maxComputeWorkgroupStorageSize. Tail threads
 * (s >= T) still run the cooperative loads: barriers must stay uniform.
 */
export function srcAttnBwdDkv(a: AttnDims): string {
  return `
${bindF32(0, "QB", "read")}
${bindF32(1, "KB", "read")}
${bindF32(2, "VB", "read")}
${bindF32(3, "GB", "read")}
${bindF32(4, "LSE", "read")}
${bindF32(5, "DB", "read")}
${bindF32(6, "DK", "read_write")}
${bindF32(7, "DV", "read_write")}
${attnConsts(a)}
const BT: u32 = 32u;
var<workgroup> Qs: array<f32, ${32 * a.hd}>;
var<workgroup> Gs: array<f32, ${32 * a.hd}>;
var<workgroup> Ls: array<f32, 32>;
var<workgroup> Ds: array<f32, 32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let kv = wg.y;
  let s = wg.x * 64u + li.x;
  let live = s < T;
  var kr: array<f32, HD>;
  var vr: array<f32, HD>;
  var dk: array<f32, HD>;
  var dv: array<f32, HD>;
  if (live) {
    for (var d = 0u; d < HD; d++) {
      kr[d] = KB[s * KS + kv * HD + d];
      vr[d] = VB[s * KS + kv * HD + d];
      dk[d] = 0.0;
      dv[d] = 0.0;
    }
  }
  for (var h = kv * GROUP; h < kv * GROUP + GROUP; h++) {
    for (var t0 = wg.x * 64u; t0 < T; t0 += BT) {
      for (var j = li.x; j < BT * HD; j += 64u) {
        let t = t0 + j / HD;
        let d = j % HD;
        var qval = 0.0;
        var gval = 0.0;
        if (t < T) {
          qval = QB[t * QS + h * HD + d];
          gval = GB[t * QS + h * HD + d];
        }
        Qs[j] = qval;
        Gs[j] = gval;
      }
      if (li.x < BT) {
        let t = t0 + li.x;
        var lval = 0.0;
        var dval = 0.0;
        if (t < T) {
          lval = LSE[h * T + t];
          dval = DB[h * T + t];
        }
        Ls[li.x] = lval;
        Ds[li.x] = dval;
      }
      workgroupBarrier();
      if (live) {
        let tEnd = min(t0 + BT, T);
        for (var t = max(t0, s); t < tEnd; t++) {
          let b = (t - t0) * HD;
          var dot = 0.0;
          var dp = 0.0;
          for (var d = 0u; d < HD; d++) {
            dot = dot + Qs[b + d] * kr[d];
            dp = dp + Gs[b + d] * vr[d];
          }
          let p = exp(dot * SCALE - Ls[t - t0]);
          let ds = p * (dp - Ds[t - t0]) * SCALE;
          for (var d = 0u; d < HD; d++) {
            dk[d] = dk[d] + ds * Qs[b + d];
            dv[d] = dv[d] + p * Gs[b + d];
          }
        }
      }
      workgroupBarrier();
    }
  }
  if (live) {
    for (var d = 0u; d < HD; d++) {
      DK[s * KS + kv * HD + d] = DK[s * KS + kv * HD + d] + dk[d];
      DV[s * KS + kv * HD + d] = DV[s * KS + kv * HD + d] + dv[d];
    }
  }
}`;
}

/** Per-row softmax + NLL; stashes probs for backward, per-row loss for reduce. */
export function srcCeFwd(T: number, V: number): string {
  return `
${bindF32(0, "LOG", "read")}
${bindU32(1, "TGT")}
${bindF32(2, "PROBS", "read_write")}
${bindF32(3, "LT", "read_write")}
const T: u32 = ${T}u; const V: u32 = ${V}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= T) { return; }
  var mx = -3.0e38;
  for (var v = 0u; v < V; v++) { mx = max(mx, LOG[t * V + v]); }
  var sum = 0.0;
  for (var v = 0u; v < V; v++) {
    let e = exp(LOG[t * V + v] - mx);
    PROBS[t * V + v] = e;
    sum = sum + e;
  }
  for (var v = 0u; v < V; v++) { PROBS[t * V + v] = PROBS[t * V + v] / sum; }
  let tgt = TGT[t];
  if (tgt == 0xffffffffu) { LT[t] = 0.0; }             // ignore-index: no loss
  else { LT[t] = -log(PROBS[t * V + tgt] + 1e-12); }
}`;
}

/** Single-thread mean of the per-row losses — T is tiny; simplicity wins. */
export function srcCeReduce(T: number): string {
  return `
${bindF32(0, "LT", "read")}
${bindF32(1, "DIV", "read")}
${bindF32(2, "LOSS", "read_write")}
const T: u32 = ${T}u;
@compute @workgroup_size(1)
fn main() {
  var a = 0.0;
  for (var t = 0u; t < T; t++) { a = a + LT[t]; }
  LOSS[0] = a / DIV[0];                                 // mean over kept rows
}`;
}

export function srcCeBwd(T: number, V: number): string {
  return `
${bindF32(0, "PROBS", "read")}
${bindU32(1, "TGT")}
${bindF32(2, "LG", "read")}
${bindF32(3, "DIV", "read")}
${bindF32(4, "DLOG", "read_write")}
const N: u32 = ${T * V}u; const T: u32 = ${T}u; const V: u32 = ${V}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * ${grid2D(T * V).roww}u + gid.x;
  if (i >= N) { return; }
  let t = i / V;
  let tgt = TGT[t];
  if (tgt == 0xffffffffu) { return; }                  // ignore-index: no gradient
  let v = i % V;
  var ind = 0.0;
  if (v == tgt) { ind = 1.0; }
  DLOG[i] = DLOG[i] + (LG[0] / DIV[0]) * (PROBS[i] - ind);
}`;
}
