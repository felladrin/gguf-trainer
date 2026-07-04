// WebGPU backend: the CPU op set from ../model/autograd.ts as WGSL compute
// shaders, forward AND backward, behind the same Tensor interface.
//
// How it plugs in: WebGPUBackend implements OpsBackend and registers itself via
// setOpsBackend() (install()/uninstall()). ../model/qwen3.ts and the autograd
// backward() walk run unchanged — op calls route here, return ordinary Tensor
// objects, and _backward closures encode GPU dispatches instead of CPU loops.
//
// Execution model: ops record dispatches into a shared command encoder and
// return immediately; tensor data lives in GPU buffers tracked in a side
// table. Host `data`/`grad` arrays are STALE until `await sync()`, the one
// unavoidable async point (WebGPU readback is async-only). A training step is:
// uploadParams -> forwards/backwards (sync code) -> await sync() -> optimizer
// step on host arrays. See train_gpu.ts.
//
// Why shapes are baked into WGSL source instead of uniform buffers: training
// shapes are static, so every kernel compiles exactly once (pipelines cached
// by source string) and bind groups stay trivial. Generation compiles one
// variant per sequence length — acceptable at demo scale.
//
// Gradient kernels avoid atomics entirely: every backward pass is expressed so
// one thread owns one gradient element and reduces over the other axis (GEMM
// transpose flavors for linear; vocab-row scan for embedding; key-axis
// reductions for attention). Accumulation across graph fan-out and across
// batch micro-steps happens by `+=` into a gradient buffer that is zeroed
// once per step, matching the CPU semantics.

import { setOpsBackend, Tensor } from "../model/autograd.ts";
import type { OpsBackend } from "../model/autograd.ts";

// --- Minimal structural WebGPU types -----------------------------------------
// Deliberately local: TypeScript's WebGPU declarations vary across runtimes and
// lib configs, and this file must merely type-check everywhere (it only *runs*
// where navigator.gpu exists). Numeric usage flags are fixed by the WebGPU spec.

const USAGE = { MAP_READ: 0x0001, COPY_SRC: 0x0004, COPY_DST: 0x0008, STORAGE: 0x0080 } as const;
const MAP_MODE_READ = 0x0001;

export interface GpuBuffer {
  mapAsync(mode: number): Promise<void>;
  getMappedRange(): ArrayBuffer;
  unmap(): void;
  destroy(): void;
}
interface GpuPipeline {
  getBindGroupLayout(index: number): unknown;
}
interface GpuComputePass {
  setPipeline(p: GpuPipeline): void;
  setBindGroup(index: number, group: unknown): void;
  dispatchWorkgroups(x: number, y?: number, z?: number): void;
  end(): void;
}
interface GpuCommandEncoder {
  beginComputePass(): GpuComputePass;
  clearBuffer(buffer: GpuBuffer): void;
  copyBufferToBuffer(
    src: GpuBuffer,
    srcOff: number,
    dst: GpuBuffer,
    dstOff: number,
    size: number,
  ): void;
  finish(): unknown;
}
interface GpuQueue {
  submit(buffers: unknown[]): void;
  writeBuffer(buffer: GpuBuffer, offset: number, data: ArrayBufferView): void;
}
interface GpuDevice {
  createBuffer(desc: { size: number; usage: number }): GpuBuffer;
  createShaderModule(desc: { code: string }): unknown;
  createComputePipeline(desc: unknown): GpuPipeline;
  createBindGroup(desc: unknown): unknown;
  createCommandEncoder(): GpuCommandEncoder;
  queue: GpuQueue;
  limits?: { maxStorageBufferBindingSize?: number };
  lost?: Promise<{ message?: string }>;
  onuncapturederror?: ((ev: { error?: { message?: string } }) => void) | null;
}

interface Entry {
  data: GpuBuffer;
  grad: GpuBuffer;
  bytes: number;
  external: boolean; // created outside backend ops (params, test inputs)
  gradNeedsClear: boolean;
}

export function ceilDiv(a: number, b: number): number {
  return Math.ceil(a / b);
}

/** Format a JS number as a WGSL f32 literal. */
export function f32lit(v: number): string {
  const s = String(v);
  return /[.e]/i.test(s) ? s : `${s}.0`;
}

// --- Buffer pool --------------------------------------------------------------
// Storage buffers keyed by 256-byte-aligned size. Training reuses identical
// shapes every step, so after step one this is pure recycling. Buffers come
// back DIRTY: anything that needs zeros (gradient accumulators) is clearBuffer'd
// explicitly before backward runs.

class BufferPool {
  private free = new Map<number, GpuBuffer[]>();
  private device: GpuDevice;

  constructor(device: GpuDevice) {
    this.device = device;
  }

  acquire(bytes: number): { buf: GpuBuffer; size: number } {
    const size = Math.max(256, Math.ceil(bytes / 256) * 256);
    const list = this.free.get(size);
    if (list && list.length > 0) return { buf: list.pop()!, size };
    const buf = this.device.createBuffer({
      size,
      usage: USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST,
    });
    return { buf, size };
  }

  release(size: number, buf: GpuBuffer) {
    let list = this.free.get(size);
    if (!list) {
      list = [];
      this.free.set(size, list);
    }
    list.push(buf);
  }

  destroyAll() {
    for (const list of this.free.values()) for (const b of list) b.destroy();
    this.free.clear();
  }
}

// --- WGSL kernel sources --------------------------------------------------------

export function bindF32(i: number, name: string, access: "read" | "read_write"): string {
  return `@group(0) @binding(${i}) var<storage, ${access}> ${name}: array<f32>;`;
}
function bindU32(i: number, name: string): string {
  return `@group(0) @binding(${i}) var<storage, read> ${name}: array<u32>;`;
}

/**
 * Tiled 16x16 GEMM: C[M,N] = sum_k A'[m,k] * B'[k,n], optionally accumulating
 * into C. Transpose flavors cover the forward and both backward products of
 * `linear` without ever materializing a transposed matrix:
 *   NT: A stored [M,K], B stored [N,K]   (y = x·Wᵀ)
 *   NN: A stored [M,K], B stored [K,N]   (dX = dY·W)
 *   TN: A stored [K,M], B stored [K,N]   (dW = dYᵀ·x)
 */
function srcGemm(
  kind: "NT" | "NN" | "TN",
  accum: boolean,
  M: number,
  N: number,
  K: number,
): string {
  const aLoad = kind === "TN" ? "AB[ka * M + m]" : "AB[m * K + ka]";
  const bLoad = kind === "NT" ? "BB[n * K + kb]" : "BB[kb * N + n]";
  const store = accum ? "CB[m * N + n] = CB[m * N + n] + acc;" : "CB[m * N + n] = acc;";
  return `
${bindF32(0, "AB", "read")}
${bindF32(1, "BB", "read")}
${bindF32(2, "CB", "read_write")}
const M: u32 = ${M}u; const N: u32 = ${N}u; const K: u32 = ${K}u;
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let m = wg.y * 16u + li.y;
  let n = wg.x * 16u + li.x;
  var acc = 0.0;
  let tiles = (K + 15u) / 16u;
  for (var tt = 0u; tt < tiles; tt++) {
    let ka = tt * 16u + li.x;
    var av = 0.0;
    if (m < M && ka < K) { av = ${aLoad}; }
    As[li.y * 16u + li.x] = av;
    let kb = tt * 16u + li.y;
    var bv = 0.0;
    if (n < N && kb < K) { bv = ${bLoad}; }
    Bs[li.y * 16u + li.x] = bv;
    workgroupBarrier();
    for (var k = 0u; k < 16u; k++) {
      acc = acc + As[li.y * 16u + k] * Bs[k * 16u + li.x];
    }
    workgroupBarrier();
  }
  if (m < M && n < N) { ${store} }
}`;
}

/** Elementwise kernel over N threads; body sees index `i`. */
function srcElementwise(bindings: string[], n: number, body: string): string {
  return `
${bindings.join("\n")}
const N: u32 = ${n}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
function srcRmsNormFwd(d: number, eps: number): string {
  return `
${bindF32(0, "XB", "read")}
${bindF32(1, "WB", "read")}
${bindF32(2, "YB", "read_write")}
${bindF32(3, "RINV", "read_write")}
const D: u32 = ${d}u; const EPS: f32 = ${f32lit(eps)};
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let row = wg.x;
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
function srcRmsNormBwdX(d: number): string {
  return `
${bindF32(0, "XB", "read")}
${bindF32(1, "WB", "read")}
${bindF32(2, "GB", "read")}
${bindF32(3, "RINV", "read")}
${bindF32(4, "DX", "read_write")}
const D: u32 = ${d}u;
var<workgroup> red: array<f32, 128>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let row = wg.x;
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
function srcRmsNormBwdW(rows: number, d: number): string {
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

function srcEmbeddingFwd(T: number, d: number): string {
  return `
${bindF32(0, "WB", "read")}
${bindU32(1, "IDS")}
${bindF32(2, "YB", "read_write")}
const N: u32 = ${T * d}u; const D: u32 = ${d}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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
function srcEmbeddingBwd(T: number, d: number, V: number): string {
  return `
${bindU32(0, "IDS")}
${bindF32(1, "GB", "read")}
${bindF32(2, "DW", "read_write")}
const N: u32 = ${V * d}u; const D: u32 = ${d}u; const T: u32 = ${T}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
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

function srcRope(
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
  let i = gid.x;
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

interface AttnDims {
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
function srcAttnProbs(a: AttnDims): string {
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
function srcAttnOut(a: AttnDims): string {
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
function srcAttnDScore(a: AttnDims): string {
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
function srcAttnDq(a: AttnDims): string {
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
function srcAttnDkv(a: AttnDims, useProbs: boolean): string {
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
function srcAttnFwd(a: AttnDims): string {
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
function srcAttnBwdD(a: AttnDims): string {
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
function srcAttnBwdDq(a: AttnDims): string {
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
function srcAttnBwdDkv(a: AttnDims): string {
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
function srcCeFwd(T: number, V: number): string {
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
  LT[t] = -log(PROBS[t * V + TGT[t]] + 1e-12);
}`;
}

/** Single-thread mean of the per-row losses — T is tiny; simplicity wins. */
function srcCeReduce(T: number): string {
  return `
${bindF32(0, "LT", "read")}
${bindF32(1, "LOSS", "read_write")}
const T: u32 = ${T}u;
@compute @workgroup_size(1)
fn main() {
  var a = 0.0;
  for (var t = 0u; t < T; t++) { a = a + LT[t]; }
  LOSS[0] = a / f32(T);
}`;
}

function srcCeBwd(T: number, V: number): string {
  return `
${bindF32(0, "PROBS", "read")}
${bindU32(1, "TGT")}
${bindF32(2, "LG", "read")}
${bindF32(3, "DLOG", "read_write")}
const N: u32 = ${T * V}u; const T: u32 = ${T}u; const V: u32 = ${V}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let t = i / V;
  let v = i % V;
  var ind = 0.0;
  if (v == TGT[t]) { ind = 1.0; }
  DLOG[i] = DLOG[i] + (LG[0] / f32(T)) * (PROBS[i] - ind);
}`;
}

// --- The backend ---------------------------------------------------------------

export class WebGPUBackend implements OpsBackend {
  readonly adapterName: string;
  /**
   * Context-length threshold for the flash-style attention kernels. Below this
   * T, the materialized [Hq,T,T] kernels win because flash's one-thread-per-row
   * structure underoccupies the GPU when T is small (measured crossover
   * M1 Max: flash is 1.4–2x slower below T~1536 and only wins from ~T=2048).
   * Above this threshold flash removes the O(T²) buffer and improves bandwidth.
   * The large-T tests require flash; small-T parity (T=67/130/193) uses the
   * materialized path and must also pass — attention() routes automatically.
   */
  attnFlashMinT = 2048;
  /** Bytes staged back to the host by the most recent sync() (profiling). */
  lastSyncReadbackBytes = 0;
  private device: GpuDevice;
  private queue: GpuQueue;
  private pool: BufferPool;
  private pipelines = new Map<string, GpuPipeline>();
  private entries = new WeakMap<Tensor, Entry>();
  private touchedExternals = new Set<Tensor>();
  private gradKeptOnDevice = new WeakSet<Tensor>();
  private transients: { buf: GpuBuffer; size: number }[] = [];
  private pendingClears: GpuBuffer[] = [];
  private enc: GpuCommandEncoder | null = null;
  private pass: GpuComputePass | null = null;
  private backwardBegun = false;
  private deviceError: string | null = null;

  constructor(device: GpuDevice, adapterName: string) {
    this.device = device;
    this.queue = device.queue;
    this.pool = new BufferPool(device);
    this.adapterName = adapterName;
    device.lost?.then((info) => {
      this.deviceError = `WebGPU device lost: ${info?.message ?? "unknown"}`;
    });
    try {
      device.onuncapturederror = (ev) => {
        this.deviceError = `WebGPU error: ${ev?.error?.message ?? "unknown"}`;
      };
    } catch {
      // Runtime without onuncapturederror support; sync() still surfaces device loss.
    }
  }

  /** Route the autograd op set through this backend. */
  install() {
    setOpsBackend(this);
  }

  uninstall() {
    setOpsBackend(null);
  }

  /**
   * Push current host data of these tensors to the GPU. Must be called after
   * anything mutates tensor.data on the host (e.g. an optimizer step) and
   * before the next forward. First use of a tensor uploads automatically.
   */
  uploadParams(tensors: Tensor[]) {
    for (const t of tensors) {
      const e = this.entryFor(t);
      this.queue.writeBuffer(e.data, 0, t.data);
    }
  }

  /**
   * Flush all recorded GPU work, then read back: the listed tensors' data and
   * the accumulated gradients of every external tensor (with requiresGrad)
   * touched since the last sync. After this resolves, those host arrays are
   * valid and all transient buffers are recycled.
   */
  async sync(reads: Tensor[] = []): Promise<void> {
    this.endPass();
    if (!this.enc) this.enc = this.device.createCommandEncoder();
    for (const b of this.pendingClears) this.enc.clearBuffer(b);
    this.pendingClears = [];

    const stagings: { stage: GpuBuffer; dst: Float32Array }[] = [];
    const readSet = new Set(reads);
    for (const t of readSet) {
      const e = this.entries.get(t);
      if (!e) continue; // never materialized on GPU — host copy is already authoritative
      stagings.push({ stage: this.copyToStaging(e.data, t.size * 4), dst: t.data });
    }
    for (const t of this.touchedExternals) {
      const e = this.entries.get(t)!;
      if (t.requiresGrad && !this.gradKeptOnDevice.has(t)) {
        stagings.push({ stage: this.copyToStaging(e.grad, t.size * 4), dst: t.grad });
      }
      e.gradNeedsClear = true;
    }
    this.lastSyncReadbackBytes = stagings.reduce((a, s) => a + s.dst.length * 4, 0);

    this.submit();
    await Promise.all(stagings.map((s) => s.stage.mapAsync(MAP_MODE_READ)));
    for (const s of stagings) {
      s.dst.set(new Float32Array(s.stage.getMappedRange(), 0, s.dst.length));
      s.stage.unmap();
      s.stage.destroy();
    }

    // The mapAsync completions above prove all submitted work finished, so
    // every transient buffer is idle and safe to recycle.
    for (const tr of this.transients) this.pool.release(tr.size, tr.buf);
    this.transients = [];
    this.touchedExternals.clear();
    this.backwardBegun = false;

    if (this.deviceError) throw new Error(this.deviceError);
  }

  /** Free every pooled GPU buffer. The backend is unusable afterwards. */
  destroy() {
    this.uninstall();
    this.pool.destroyAll();
  }

  // --- ops (OpsBackend) ---------------------------------------------------------

  linear(x: Tensor, w: Tensor): Tensor {
    this.beginForwardOp();
    const [T, inDim] = x.shape;
    const [outDim, inDim2] = w.shape;
    if (inDim !== inDim2) throw new Error(`linear dim mismatch ${inDim} vs ${inDim2}`);
    const ex = this.entryFor(x);
    const ew = this.entryFor(w);
    const { t: out, e: eo } = this.makeOut([T, outDim], [x, w]);
    this.gemm("NT", false, T, outDim, inDim, ex.data, ew.data, eo.data);
    out._backward = () => {
      this.ensureBackwardBegun();
      this.gemm("NN", true, T, inDim, outDim, eo.grad, ew.data, ex.grad);
      this.gemm("TN", true, outDim, inDim, T, eo.grad, ex.data, ew.grad);
    };
    return out;
  }

  add(a: Tensor, b: Tensor): Tensor {
    this.beginForwardOp();
    const ea = this.entryFor(a);
    const eb = this.entryFor(b);
    const { t: out, e: eo } = this.makeOut(a.shape, [a, b]);
    const n = out.size;
    this.dispatch(
      srcElementwise(
        [bindF32(0, "AB", "read"), bindF32(1, "BB", "read"), bindF32(2, "YB", "read_write")],
        n,
        "YB[i] = AB[i] + BB[i];",
      ),
      [ea.data, eb.data, eo.data],
      ceilDiv(n, 256),
    );
    out._backward = () => {
      this.ensureBackwardBegun();
      this.dispatch(
        srcElementwise(
          [
            bindF32(0, "GB", "read"),
            bindF32(1, "DA", "read_write"),
            bindF32(2, "DB", "read_write"),
          ],
          n,
          "DA[i] = DA[i] + GB[i]; DB[i] = DB[i] + GB[i];",
        ),
        [eo.grad, ea.grad, eb.grad],
        ceilDiv(n, 256),
      );
    };
    return out;
  }

  mul(a: Tensor, b: Tensor): Tensor {
    this.beginForwardOp();
    const ea = this.entryFor(a);
    const eb = this.entryFor(b);
    const { t: out, e: eo } = this.makeOut(a.shape, [a, b]);
    const n = out.size;
    this.dispatch(
      srcElementwise(
        [bindF32(0, "AB", "read"), bindF32(1, "BB", "read"), bindF32(2, "YB", "read_write")],
        n,
        "YB[i] = AB[i] * BB[i];",
      ),
      [ea.data, eb.data, eo.data],
      ceilDiv(n, 256),
    );
    out._backward = () => {
      this.ensureBackwardBegun();
      this.dispatch(
        srcElementwise(
          [
            bindF32(0, "GB", "read"),
            bindF32(1, "AB", "read"),
            bindF32(2, "BB", "read"),
            bindF32(3, "DA", "read_write"),
            bindF32(4, "DB", "read_write"),
          ],
          n,
          "DA[i] = DA[i] + GB[i] * BB[i]; DB[i] = DB[i] + GB[i] * AB[i];",
        ),
        [eo.grad, ea.data, eb.data, ea.grad, eb.grad],
        ceilDiv(n, 256),
      );
    };
    return out;
  }

  silu(x: Tensor): Tensor {
    this.beginForwardOp();
    const ex = this.entryFor(x);
    const { t: out, e: eo } = this.makeOut(x.shape, [x]);
    const n = out.size;
    this.dispatch(
      srcElementwise(
        [bindF32(0, "XB", "read"), bindF32(1, "YB", "read_write")],
        n,
        "let s = 1.0 / (1.0 + exp(-XB[i])); YB[i] = XB[i] * s;",
      ),
      [ex.data, eo.data],
      ceilDiv(n, 256),
    );
    out._backward = () => {
      this.ensureBackwardBegun();
      // Recompute sigmoid from the saved input instead of stashing it — one
      // fewer buffer, and the input is alive in the graph anyway.
      this.dispatch(
        srcElementwise(
          [bindF32(0, "GB", "read"), bindF32(1, "XB", "read"), bindF32(2, "DX", "read_write")],
          n,
          "let s = 1.0 / (1.0 + exp(-XB[i])); DX[i] = DX[i] + GB[i] * (s + XB[i] * s * (1.0 - s));",
        ),
        [eo.grad, ex.data, ex.grad],
        ceilDiv(n, 256),
      );
    };
    return out;
  }

  rmsNorm(x: Tensor, weight: Tensor, eps: number): Tensor {
    const [T, d] = x.shape;
    return this.rmsNormRows(x, weight, T, d, eps, [T, d]);
  }

  rmsNormHeads(x: Tensor, weight: Tensor, T: number, H: number, hd: number, eps: number): Tensor {
    // Per-(token, head) rows are contiguous, so this is rmsNorm on [T*H, hd].
    return this.rmsNormRows(x, weight, T * H, hd, eps, [T, H * hd]);
  }

  embedding(weight: Tensor, ids: number[]): Tensor {
    this.beginForwardOp();
    const [V, d] = weight.shape;
    const T = ids.length;
    const ew = this.entryFor(weight);
    const idsBuf = this.uploadU32(ids);
    const { t: out, e: eo } = this.makeOut([T, d], [weight]);
    this.dispatch(srcEmbeddingFwd(T, d), [ew.data, idsBuf, eo.data], ceilDiv(T * d, 256));
    out._backward = () => {
      this.ensureBackwardBegun();
      this.dispatch(srcEmbeddingBwd(T, d, V), [idsBuf, eo.grad, ew.grad], ceilDiv(V * d, 256));
    };
    return out;
  }

  rope(x: Tensor, T: number, H: number, hd: number, base: number, posOffset: number): Tensor {
    this.beginForwardOp();
    const ex = this.entryFor(x);
    const { t: out, e: eo } = this.makeOut([T, H * hd], [x]);
    const n = T * H * (hd / 2);
    this.dispatch(srcRope(T, H, hd, base, posOffset, false), [ex.data, eo.data], ceilDiv(n, 256));
    out._backward = () => {
      this.ensureBackwardBegun();
      this.dispatch(srcRope(T, H, hd, base, posOffset, true), [eo.grad, ex.grad], ceilDiv(n, 256));
    };
    return out;
  }

  attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    T: number,
    Hq: number,
    Hkv: number,
    hd: number,
  ): Tensor {
    this.beginForwardOp();
    const a: AttnDims = { T, Hq, Hkv, hd };
    const eq = this.entryFor(q);
    const ek = this.entryFor(k);
    const ev = this.entryFor(v);
    const { t: out, e: eo } = this.makeOut([T, Hq * hd], [q, k, v]);
    if (T >= this.attnFlashMinT) {
      // Flash path: per-row logsumexp (O(Hq·T)) is the only side buffer —
      // no [Hq,T,T] allocation, so context length is no longer capped by
      // maxStorageBufferBindingSize.
      const lse = this.acquireTransient(Hq * T * 4);
      this.dispatch(
        srcAttnFwd(a),
        [eq.data, ek.data, ev.data, eo.data, lse],
        ceilDiv(Hq * T, 64),
      );
      out._backward = () => {
        this.ensureBackwardBegun();
        const dRow = this.acquireTransient(Hq * T * 4);
        this.dispatch(srcAttnBwdD(a), [eo.grad, eo.data, dRow], ceilDiv(Hq * T, 64));
        this.dispatch(
          srcAttnBwdDq(a),
          [eq.data, ek.data, ev.data, eo.grad, lse, dRow, eq.grad],
          ceilDiv(Hq * T, 64),
        );
        this.dispatch(
          srcAttnBwdDkv(a),
          [eq.data, ek.data, ev.data, eo.grad, lse, dRow, ek.grad, ev.grad],
          ceilDiv(T, 64),
          Hkv,
        );
      };
    } else {
      // Materialized path: [Hq,T,T] probability buffer — faster at small T
      // where flash's one-thread-per-row layout underoccupies the GPU.
      const probs = this.acquireTransient(Hq * T * T * 4);
      this.dispatch(srcAttnProbs(a), [eq.data, ek.data, probs], ceilDiv(Hq * T, 64));
      this.dispatch(srcAttnOut(a), [probs, ev.data, eo.data], ceilDiv(T * Hq * hd, 64));
      out._backward = () => {
        this.ensureBackwardBegun();
        const dScore = this.acquireTransient(Hq * T * T * 4);
        this.dispatch(srcAttnDScore(a), [eo.grad, ev.data, probs, dScore], ceilDiv(Hq * T, 64));
        this.dispatch(srcAttnDq(a), [dScore, ek.data, eq.grad], ceilDiv(T * Hq * hd, 64));
        this.dispatch(srcAttnDkv(a, false), [dScore, eq.data, ek.grad], ceilDiv(T * Hkv * hd, 64));
        this.dispatch(srcAttnDkv(a, true), [probs, eo.grad, ev.grad], ceilDiv(T * Hkv * hd, 64));
      };
    }
    return out;
  }

  crossEntropy(logits: Tensor, targets: number[]): Tensor {
    this.beginForwardOp();
    const [T, V] = logits.shape;
    const el = this.entryFor(logits);
    const tgtBuf = this.uploadU32(targets);
    const probs = this.acquireTransient(T * V * 4);
    const perRow = this.acquireTransient(T * 4);
    const { t: loss, e: eo } = this.makeOut([1], [logits]);
    this.dispatch(srcCeFwd(T, V), [el.data, tgtBuf, probs, perRow], ceilDiv(T, 64));
    this.dispatch(srcCeReduce(T), [perRow, eo.data], 1);
    loss._backward = () => {
      // backward(loss, seed) already wrote the seed into the HOST grad array;
      // push it into the GPU-side loss grad after the grad clears are flushed.
      this.ensureBackwardBegun();
      this.queue.writeBuffer(eo.grad, 0, loss.grad);
      this.dispatch(srcCeBwd(T, V), [probs, tgtBuf, eo.grad, el.grad], ceilDiv(T * V, 256));
    };
    return loss;
  }

  // --- shared machinery -----------------------------------------------------------

  /**
   * A forward op after a backward means a new micro-batch graph: arm the
   * clears-then-seed flush for the next backward pass.
   */
  private beginForwardOp() {
    this.backwardBegun = false;
    if (this.deviceError) throw new Error(this.deviceError);
  }

  /**
   * First backward closure of a graph: flush the pending gradient-buffer
   * clears and submit, so a subsequent queue.writeBuffer of the loss seed
   * lands AFTER the clears in queue order.
   */
  private ensureBackwardBegun() {
    if (this.backwardBegun) return;
    this.backwardBegun = true;
    this.endPass();
    if (this.pendingClears.length > 0) {
      if (!this.enc) this.enc = this.device.createCommandEncoder();
      for (const b of this.pendingClears) this.enc.clearBuffer(b);
      this.pendingClears = [];
    }
    this.submit();
  }

  /** Seed an output tensor's GPU gradient from its host grad array (tests). */
  seedGradFromHost(t: Tensor) {
    this.ensureBackwardBegun();
    const e = this.entries.get(t);
    if (!e) throw new Error("seedGradFromHost: tensor has no GPU buffers");
    this.queue.writeBuffer(e.grad, 0, t.grad);
  }

  // --- raw-buffer access for GPU-side optimizers (see muon_gpu.ts) ---------------

  /**
   * The device buffers backing an external tensor, creating them (and
   * uploading the current host data) on first use. The buffers are stable for
   * the tensor's lifetime, so an optimizer can capture them once.
   */
  buffersFor(t: Tensor): { data: GpuBuffer; grad: GpuBuffer } {
    const e = this.entryFor(t);
    return { data: e.data, grad: e.grad };
  }

  /**
   * Stop sync() from reading this tensor's gradient back to the host: a GPU
   * optimizer consumes it in place. Per-step grad clearing is unaffected.
   */
  keepGradOnDevice(t: Tensor) {
    this.gradKeptOnDevice.add(t);
  }

  /**
   * A fresh zero-initialized storage buffer OUTSIDE the pool. Optimizer state
   * (momentum) must survive sync()'s transient recycling and must start at
   * exactly zero — pooled buffers come back dirty, freshly created ones are
   * guaranteed zeroed by the WebGPU spec.
   */
  createStateBuffer(bytes: number): GpuBuffer {
    return this.device.createBuffer({
      size: Math.max(4, Math.ceil(bytes / 4) * 4),
      usage: USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST,
    });
  }

  /**
   * Resolve pipeline + bind group once and return a closure that records the
   * dispatch. Only valid for PERSISTENT buffers (a pooled transient may be
   * recycled under the cached bind group). An optimizer re-records identical
   * dispatches every step, so this removes the dominant per-step encode cost
   * (measured: bind-group/source rebuilding was ~85% of the optimizer step).
   */
  prepareDispatch(code: string, buffers: GpuBuffer[], x: number, y = 1): () => void {
    const p = this.pipeline(code);
    const group = this.device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: buffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } })),
    });
    return () => {
      if (!this.enc) this.enc = this.device.createCommandEncoder();
      if (!this.pass) this.pass = this.enc.beginComputePass();
      this.pass.setPipeline(p);
      this.pass.setBindGroup(0, group);
      this.pass.dispatchWorkgroups(x, y, 1);
    };
  }

  /** prepareDispatch for the tiled GEMM (flavors as in srcGemm). */
  prepareGemm(
    kind: "NT" | "NN" | "TN",
    accum: boolean,
    M: number,
    N: number,
    K: number,
    a: GpuBuffer,
    b: GpuBuffer,
    c: GpuBuffer,
  ): () => void {
    return this.prepareDispatch(
      srcGemm(kind, accum, M, N, K),
      [a, b, c],
      ceilDiv(N, 16),
      ceilDiv(M, 16),
    );
  }

  private entryFor(t: Tensor): Entry {
    let e = this.entries.get(t);
    if (!e) {
      // A tensor first seen as an op input was created outside the backend
      // (parameter or test input): upload its host data and give it a
      // persistent gradient accumulator.
      const bytes = t.size * 4;
      e = {
        data: this.acquirePersistent(bytes),
        grad: this.acquirePersistent(bytes),
        bytes,
        external: true,
        gradNeedsClear: true, // pool buffers arrive dirty
      };
      this.entries.set(t, e);
      this.queue.writeBuffer(e.data, 0, t.data);
    }
    if (e.external && !this.touchedExternals.has(t)) {
      this.touchedExternals.add(t);
      if (e.gradNeedsClear) {
        this.pendingClears.push(e.grad);
        e.gradNeedsClear = false;
      }
    }
    return e;
  }

  private makeOut(shape: number[], prev: Tensor[]): { t: Tensor; e: Entry } {
    const t = Tensor.zeros(shape);
    t._prev = prev;
    const bytes = t.size * 4;
    const e: Entry = {
      data: this.acquireTransient(bytes),
      grad: this.acquireTransient(bytes),
      bytes,
      external: false,
      gradNeedsClear: false,
    };
    // Gradients accumulate with +=, so the (possibly recycled) buffer must be
    // zeroed before this graph's backward pass runs.
    this.pendingClears.push(e.grad);
    this.entries.set(t, e);
    return { t, e };
  }

  private acquirePersistent(bytes: number): GpuBuffer {
    return this.pool.acquire(bytes).buf;
  }

  private acquireTransient(bytes: number): GpuBuffer {
    const { buf, size } = this.pool.acquire(bytes);
    this.transients.push({ buf, size });
    return buf;
  }

  private uploadU32(values: number[]): GpuBuffer {
    const buf = this.acquireTransient(values.length * 4);
    // queue.writeBuffer executes before any later-submitted encoder, and this
    // buffer can't appear in already-encoded (unsubmitted) passes: it was free
    // in the pool until this call.
    this.queue.writeBuffer(buf, 0, Uint32Array.from(values));
    return buf;
  }

  private gemm(
    kind: "NT" | "NN" | "TN",
    accum: boolean,
    M: number,
    N: number,
    K: number,
    a: GpuBuffer,
    b: GpuBuffer,
    c: GpuBuffer,
  ) {
    this.dispatch(srcGemm(kind, accum, M, N, K), [a, b, c], ceilDiv(N, 16), ceilDiv(M, 16));
  }

  private rmsNormRows(
    x: Tensor,
    weight: Tensor,
    rows: number,
    d: number,
    eps: number,
    outShape: number[],
  ): Tensor {
    this.beginForwardOp();
    const ex = this.entryFor(x);
    const ew = this.entryFor(weight);
    const { t: out, e: eo } = this.makeOut(outShape, [x, weight]);
    const rInv = this.acquireTransient(rows * 4);
    this.dispatch(srcRmsNormFwd(d, eps), [ex.data, ew.data, eo.data, rInv], rows);
    out._backward = () => {
      this.ensureBackwardBegun();
      this.dispatch(srcRmsNormBwdX(d), [ex.data, ew.data, eo.grad, rInv, ex.grad], rows);
      this.dispatch(
        srcRmsNormBwdW(rows, d),
        [ex.data, eo.grad, rInv, ew.grad],
        ceilDiv(d, 64),
      );
    };
    return out;
  }

  private pipeline(code: string): GpuPipeline {
    let p = this.pipelines.get(code);
    if (!p) {
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
      });
      this.pipelines.set(code, p);
    }
    return p;
  }

  private dispatch(code: string, buffers: GpuBuffer[], x: number, y = 1, z = 1) {
    const p = this.pipeline(code);
    if (!this.enc) this.enc = this.device.createCommandEncoder();
    if (!this.pass) this.pass = this.enc.beginComputePass();
    const group = this.device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries: buffers.map((buf, i) => ({ binding: i, resource: { buffer: buf } })),
    });
    this.pass.setPipeline(p);
    this.pass.setBindGroup(0, group);
    this.pass.dispatchWorkgroups(x, y, z);
  }

  private endPass() {
    if (this.pass) {
      this.pass.end();
      this.pass = null;
    }
  }

  private submit() {
    this.endPass();
    if (this.enc) {
      this.queue.submit([this.enc.finish()]);
      this.enc = null;
    }
  }

  private copyToStaging(src: GpuBuffer, bytes: number): GpuBuffer {
    const stage = this.device.createBuffer({
      size: Math.ceil(bytes / 4) * 4,
      usage: USAGE.MAP_READ | USAGE.COPY_DST,
    });
    this.enc!.copyBufferToBuffer(src, 0, stage, 0, bytes);
    return stage;
  }
}

/**
 * Probe for a WebGPU device. Returns null when the runtime has no WebGPU
 * (Node and Bun today — run GPU work under Deno, or provide a navigator.gpu
 * polyfill); training then stays on the CPU reference backend.
 */
export async function initWebGPU(): Promise<WebGPUBackend | null> {
  // deno-lint-ignore no-explicit-any
  const nav: any = (globalThis as any).navigator;
  if (!nav?.gpu) return null;
  const adapter = await nav.gpu.requestAdapter();
  if (!adapter) return null;
  // Request the adapter's own maximum buffer limits instead of the WebGPU
  // spec's conservative default (128 MiB per storage buffer binding).
  // Attention is flash-tiled and no longer binds a [heads,T,T] buffer, but
  // other single buffers still grow with model/context size (logits [T,V]
  // and their gradients, embedding-weight gradients), so headroom stays
  // useful. Asking for exactly the adapter's reported ceiling can never
  // exceed what it supports; fall back to the default-limits device on the
  // rare adapter that still rejects it, so WebGPU availability itself never
  // regresses.
  let device: GpuDevice;
  try {
    device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
  } catch {
    device = await adapter.requestDevice();
  }
  const name = adapter.info?.description || adapter.info?.vendor || "unknown adapter";
  return new WebGPUBackend(device, String(name));
}
