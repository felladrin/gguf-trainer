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
 * A flat ceilDiv(n,256) grid overflows past ~16.7M elements: reached by SwiGLU
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
// (or TN) MACs instead of one: the arithmetic-intensity lever that lifts this
// off the "SMEM-tiled, 1 output/thread" rung. Single-sourced with the dispatch
// grid (gemm/prepareGemm below) so block size and workgroup count never drift.
//
// BK=16 halves the barrier count and the per-K-step loop overhead versus the
// original 8 at 8 KiB of workgroup storage, half the 16 KiB WebGPU floor (BK=32
// measured no faster and spends the whole floor). TM and TN must stay multiples
// of 4: the micro-tile fragments are read as vec4 (see srcGemm).
export const GEMM_BM = 64, GEMM_BN = 64, GEMM_BK = 16, GEMM_TM = 4, GEMM_TN = 4;
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
 *
 * Both staged tiles are held as vec4 and both fragments are read as vec4, so a
 * K-step costs 2 workgroup loads instead of 8. That is why the A tile is staged
 * TRANSPOSED (As[k][m], not As[m][k]): in the original layout a thread's TM
 * rows are BK apart, and a stride-BK fragment cannot be one load. The staging
 * loops walk vec4 units and still guard every element, so ragged edges are
 * unchanged.
 */
const XYZW = ["x", "y", "z", "w"];

export function srcGemm(
  kind: "NT" | "NN" | "TN",
  accum: boolean,
  M: number,
  N: number,
  K: number,
): string {
  const [BM, BN, BK, TM, TN, WG] = [GEMM_BM, GEMM_BN, GEMM_BK, GEMM_TM, GEMM_TN, GEMM_WG];
  if (TM % 4 !== 0 || TN % 4 !== 0 || BM % 4 !== 0 || BN % 4 !== 0) {
    throw new Error(`srcGemm: BM/BN/TM/TN must be multiples of 4 (vec4 fragments)`);
  }
  // gr = global row (m), gc = global col (n), gk = global k index.
  const aLoad = kind === "TN" ? "AB[gk * M + gr]" : "AB[gr * K + gk]";
  const bLoad = kind === "NT" ? "BB[gc * K + gk]" : "BB[gk * N + gc]";
  // Everything is f32: buffers, the shared tiles, the multiply, and the
  // K-length accumulation. (An f16-operand variant was removed: it gave no
  // wall-clock gain here since attention, not GEMM, dominates runtime, and
  // rounding operands to f16 overflowed on longer runs.) sh = tile scalar type.
  const sh = "f32";
  // Four guarded scalar loads per staged vec4: the four A rows are K apart (or
  // the four B cols K apart under NT), so only the workgroup-side store is wide.
  // Each guard here is individually redundant, against the store guards below and
  // against the other tile's K guard (a leaked A value at the ragged tail meets a
  // zeroed B at the same kc). They are kept anyway: without them each staging loop
  // would read out of bounds and its correctness would depend on the other loop.
  let aStage = "", bStage = "";
  for (let q = 0; q < 4; q++) {
    aStage += `      { let gr = blockRow + r + ${q}u; if (gr < M && gk < K) ` +
      `{ v.${XYZW[q]} = ${aLoad}; } }\n`;
    bStage += `      { let gc = gc0 + ${q}u; if (gk < K && gc < N) ` +
      `{ v.${XYZW[q]} = ${bLoad}; } }\n`;
  }
  // Unroll the tile so every accumulator/fragment is a compile-time-named
  // scalar (stays in registers) rather than a dynamically-indexed array.
  let decl = "", fragA = "", fragB = "", macs = "", stores = "";
  for (let i = 0; i < TM; i += 4) {
    fragA += `      let av${i} = As[kc * ${BM / 4}u + tRow4 + ${i / 4}u];\n`;
    for (let q = 0; q < 4; q++) fragA += `      let a${i + q} = av${i}.${XYZW[q]};\n`;
  }
  for (let j = 0; j < TN; j += 4) {
    fragB += `      let bv${j} = Bs[kc * ${BN / 4}u + tCol4 + ${j / 4}u];\n`;
    for (let q = 0; q < 4; q++) fragB += `      let b${j + q} = bv${j}.${XYZW[q]};\n`;
  }
  for (let i = 0; i < TM; i++) {
    for (let j = 0; j < TN; j++) {
      decl += `  var acc${i}_${j} = 0.0;\n`;
      macs += `      acc${i}_${j} = acc${i}_${j} + a${i} * b${j};\n`;
      const idx = `(blockRow + tRow + ${i}u) * N + (blockCol + tCol + ${j}u)`;
      const rhs = accum ? `CB[ci] + acc${i}_${j}` : `acc${i}_${j}`;
      stores += `  if (blockRow + tRow + ${i}u < M && blockCol + tCol + ${j}u < N) ` +
        `{ let ci = ${idx}; CB[ci] = ${rhs}; }\n`;
    }
  }
  return `
${bindF32(0, "AB", "read")}
${bindF32(1, "BB", "read")}
${bindF32(2, "CB", "read_write")}
const M: u32 = ${M}u; const N: u32 = ${N}u; const K: u32 = ${K}u;
var<workgroup> As: array<vec4<${sh}>, ${(BM * BK) / 4}>;  // [BK][BM/4]: transposed
var<workgroup> Bs: array<vec4<${sh}>, ${(BK * BN) / 4}>;  // [BK][BN/4]
@compute @workgroup_size(${WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lidx: u32) {
  let blockRow = wg.y * ${BM}u;
  let blockCol = wg.x * ${BN}u;
  let tRow = (lidx / ${BN / TN}u) * ${TM}u; // this thread's first row in the tile
  let tCol = (lidx % ${BN / TN}u) * ${TN}u; // this thread's first col in the tile
  let tRow4 = tRow / 4u;
  let tCol4 = tCol / 4u;
${decl}  var kk = 0u;
  loop {
    if (kk >= K) { break; }
    // Cooperatively stage the A and B tiles, one vec4 per step (grid-strided).
    for (var t = lidx; t < ${(BM * BK) / 4}u; t += ${WG}u) {
      let kc = t / ${BM / 4}u;
      let r = (t % ${BM / 4}u) * 4u;
      let gk = kk + kc;
      var v = vec4<${sh}>(0.0);
${aStage}      As[kc * ${BM / 4}u + r / 4u] = v;
    }
    for (var t = lidx; t < ${(BK * BN) / 4}u; t += ${WG}u) {
      let kc = t / ${BN / 4}u;
      let c = (t % ${BN / 4}u) * 4u;
      let gk = kk + kc;
      let gc0 = blockCol + c;
      var v = vec4<${sh}>(0.0);
${bStage}      Bs[kc * ${BN / 4}u + c / 4u] = v;
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
 * Deterministic and race-free (repeated ids just extend the scan) at the
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
  window?: number; // sliding-window size (keys [t-window+1, t]); 0/undefined = full causal
}

/**
 * How the flash kernels step along the head dimension. When the head size is a
 * multiple of 4 every Q/K/V/output row is addressed as `vec4<f32>`, which turns
 * each 4 scalar loads into one 16-byte load AND splits the head-deep dot product
 * into 4 independent accumulation chains (the two levers that lifted `srcGemm`,
 * applied to the loop where the loads actually live). Head sizes that are not a
 * multiple of 4 keep the scalar form: the emitted source is otherwise identical,
 * so there is one kernel body, not two. `tests/gpu-parity.ts` covers hd=6 (scalar)
 * and hd=24/32 (vectorized).
 */
interface AttnLanes {
  ty: string; // element type of a lane
  hdv: number; // lanes per head
  qsv: number; // lanes per query row
  ksv: number; // lanes per key row
  zero: string; // additive identity of `ty`
  /** Reduce a lane-wide accumulator to the scalar it stands for. */
  sum: (v: string) => string;
}

function attnLanes(a: AttnDims): AttnLanes {
  const w = a.hd % 4 === 0 ? 4 : 1;
  return {
    ty: w === 4 ? "vec4<f32>" : "f32",
    hdv: a.hd / w,
    qsv: (a.Hq * a.hd) / w,
    ksv: (a.Hkv * a.hd) / w,
    zero: w === 4 ? "vec4<f32>(0.0)" : "0.0",
    sum: w === 4 ? (v: string) => `${v}.x + ${v}.y + ${v}.z + ${v}.w` : (v: string) => v,
  };
}

/**
 * Query rows srcAttnBwdDkv stages per tile. Its workgroup footprint is
 * 8·BT·(hd+1) bytes (Q and dOut tiles plus the LSE/D statistics), so a fixed 32
 * overflows WebGPU's 16 KiB portable floor from hd=64 up: the head size every
 * published checkpoint uses. Sized down to the largest power of two that fits,
 * which keeps every device on one kernel rather than making long-context
 * training depend on the adapter granting more than the spec default.
 */
function attnBwdTile(a: AttnDims): number {
  let bt = 32;
  while (bt > 1 && 8 * bt * (a.hd + 1) > 16384) bt >>= 1;
  if (8 * bt * (a.hd + 1) > 16384) {
    throw new Error(
      `attention: head dim ${a.hd} cannot fit the 16 KiB portable workgroup-storage floor ` +
        `even at one query row per tile`,
    );
  }
  return bt;
}

/** Storage binding whose element type follows the lane width (see AttnLanes). */
function bindLane(i: number, name: string, access: "read" | "read_write", L: AttnLanes): string {
  return `@group(0) @binding(${i}) var<storage, ${access}> ${name}: array<${L.ty}>;`;
}

function attnConsts(a: AttnDims): string {
  const scale = 1 / Math.sqrt(a.hd);
  const L = attnLanes(a);
  return `const T: u32 = ${a.T}u; const HQ: u32 = ${a.Hq}u; const HKV: u32 = ${a.Hkv}u;
const HD: u32 = ${a.hd}u; const GROUP: u32 = ${a.Hq / a.Hkv}u;
const QS: u32 = ${a.Hq * a.hd}u; const KS: u32 = ${a.Hkv * a.hd}u;
// Lane-indexed strides: identical to HD/QS/KS in the scalar fallback.
const HDV: u32 = ${L.hdv}u; const QSV: u32 = ${L.qsv}u; const KSV: u32 = ${L.ksv}u;
const SCALE: f32 = ${f32lit(scale)}; const WINDOW: u32 = ${a.window ?? 0}u;
// Score scale folded into the log2 domain, so the online softmax can use the
// exp2 builtin directly: exp(x) costs an extra multiply by log2(e) on hardware
// whose only exponential instruction is base-2 (all of it). LSE is stored in
// the same domain, so both backward kernels recompute p with the same exp2.
const SCALE2: f32 = ${f32lit(scale * Math.LOG2E)};
// First key attended by query row t under the sliding window (0 = full causal).
fn winStart(t: u32) -> u32 { if (WINDOW == 0u || t + 1u <= WINDOW) { return 0u; } return t + 1u - WINDOW; }
// Block-aligned window start for the flash kernels: the lower key bound is made
// uniform across a 64-query block so every thread in a wave reads the same K[s]
// in lockstep (the free broadcast the full-causal kernel relies on: a per-thread
// lower bound offset by the query index destroys it, making SWA slower than full;
// keys before a row's own window [t-WINDOW+1,t] are read but masked out below).
fn winStartBlock(t: u32) -> u32 {
  if (WINDOW == 0u) { return 0u; }
  let blk = (t / 64u) * 64u;
  if (blk + 1u <= WINDOW) { return 0u; }
  return blk + 1u - WINDOW;
}`;
}

// --- Attention, small-T regime: materialized [Hq,T,T] probabilities ----------
// Two attention implementations coexist on purpose. These five kernels write
// the full probability (and dScore) matrix but parallelize over hd as well, giving
// T·Hq·hd threads with O(T)-deep loops. The flash kernels below never touch a
// [Hq,T,T] buffer but run one thread per (head, query row): Hq·T threads with
// O(T·hd)-deep loops, which underoccupies the GPU and lengthens the serial
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
  for (var si = winStart(t); si <= t; si++) {
    var dot = 0.0;
    for (var d = 0u; d < HD; d++) { dot = dot + QB[t * QS + h * HD + d] * KB[si * KS + kv * HD + d]; }
    maxS = max(maxS, dot * SCALE);
  }
  var sum = 0.0;
  for (var si = winStart(t); si <= t; si++) {
    var dot = 0.0;
    for (var d = 0u; d < HD; d++) { dot = dot + QB[t * QS + h * HD + d] * KB[si * KS + kv * HD + d]; }
    let e = exp(dot * SCALE - maxS);
    PB[(h * T + t) * T + si] = e;
    sum = sum + e;
  }
  for (var si = winStart(t); si <= t; si++) {
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
  for (var si = winStart(t); si <= t; si++) {
    a = a + PB[(h * T + t) * T + si] * VB[si * KS + kv * HD + d];
  }
  YB[t * QS + h * HD + d] = a;
}`;
}

/**
 * Softmax backward to scores: dS[h,t,s] = P·(dP − Σ P·dP)·scale, where
 * dP[s] = Σ_d dOut[t,h,d]·V[s,kv,d]. dP is recomputed in the second loop
 * instead of stored: avoids per-thread arrays, and the extra T·hd multiplies
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
  for (var si = winStart(t); si <= t; si++) {
    var dp = 0.0;
    for (var d = 0u; d < HD; d++) { dp = dp + GB[t * QS + h * HD + d] * VB[si * KS + kv * HD + d]; }
    dot = dot + PB[(h * T + t) * T + si] * dp;
  }
  for (var si = winStart(t); si <= t; si++) {
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
  for (var si = winStart(t); si <= t; si++) {
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
  let tEnd = select(T, min(T, si + WINDOW), WINDOW != 0u);
  for (var h = kv * GROUP; h < kv * GROUP + GROUP; h++) {
    for (var t = si; t < tEnd; t++) {
      a = a + ${lhs}[(h * T + t) * T + si] * SRC[t * QS + h * HD + d];
    }
  }
  DST[si * KS + kv * HD + d] = DST[si * KS + kv * HD + d] + a;
}`;
}

// --- Attention, large-T regime: flash-style, no [Hq,T,T] buffer ---------------

/**
 * Fused causal attention forward, flash-style: one thread per (head, query
 * row) runs an online softmax: running max `m`, running denominator `l`,
 * and an HD-wide output accumulator rescaled by exp(m−mNew) whenever the max
 * moves, so the [Hq,T,T] probability matrix is never materialized. The only
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
  const L = attnLanes(a);
  return `
${bindLane(0, "QB", "read", L)}
${bindLane(1, "KB", "read", L)}
${bindLane(2, "VB", "read", L)}
${bindLane(3, "YB", "read_write", L)}
${bindF32(4, "LSE", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= HQ * T) { return; }
  let h = i / T;
  let t = i % T;
  let kv = h / GROUP;
  var qr: array<${L.ty}, HDV>;
  var acc: array<${L.ty}, HDV>;
  for (var d = 0u; d < HDV; d++) {
    qr[d] = QB[t * QSV + h * HDV + d];
    acc[d] = ${L.zero};
  }
  var m = -3.0e38;
  var l = 0.0;
  for (var s = winStartBlock(t); s <= t; s++) {
    var dotv = ${L.zero};
    for (var d = 0u; d < HDV; d++) { dotv = dotv + qr[d] * KB[s * KSV + kv * HDV + d]; }
    if (WINDOW != 0u && s + WINDOW <= t) { continue; } // key before this row's window
    let sc = (${L.sum("dotv")}) * SCALE2;
    // The running max is monotone, so the accumulator only needs rescaling on
    // the handful of keys that actually raise it; every other key is a plain
    // multiply-add. On the branch that does raise it, p is exp2(sc-sc) = 1, so
    // either way exactly one exp2 is evaluated.
    if (sc > m) {
      let corr = exp2(m - sc);
      l = l * corr + 1.0;
      for (var d = 0u; d < HDV; d++) { acc[d] = acc[d] * corr + VB[s * KSV + kv * HDV + d]; }
      m = sc;
    } else {
      let p = exp2(sc - m);
      l = l + p;
      for (var d = 0u; d < HDV; d++) { acc[d] = acc[d] + p * VB[s * KSV + kv * HDV + d]; }
    }
  }
  for (var d = 0u; d < HDV; d++) { YB[t * QSV + h * HDV + d] = acc[d] / l; }
  LSE[h * T + t] = m + log2(l);
}`;
}

/**
 * D[h,t] = Σ_d dOut[t,h,d]·out[t,h,d]: the softmax-backward row constant
 * (equals Σ_s P·dP), precomputed once so neither backward kernel re-derives
 * it inside its O(T) scan.
 */
export function srcAttnBwdD(a: AttnDims): string {
  const L = attnLanes(a);
  return `
${bindLane(0, "GB", "read", L)}
${bindLane(1, "YB", "read", L)}
${bindF32(2, "DB", "read_write")}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= HQ * T) { return; }
  let h = i / T;
  let t = i % T;
  var sv = ${L.zero};
  for (var d = 0u; d < HDV; d++) {
    sv = sv + GB[t * QSV + h * HDV + d] * YB[t * QSV + h * HDV + d];
  }
  DB[h * T + t] = ${L.sum("sv")};
}`;
}

/**
 * dQ[t,h,:] += Σ_{s<=t} dS[t,s]·K[s,kv,:] with dS recomputed on the fly:
 * p = exp(q·k·scale − LSE[h,t]), dP = Σ_d dOut·V[s], dS = p·(dP − D[h,t])·scale.
 * One thread per (head, query row) owns the whole dQ row, each (t,s) score
 * is computed once and reused across HD via the private accumulator, and no
 * atomics are needed. Direct loads for the same reason as srcAttnFwd: the
 * lockstep key scan is already uniform per wavefront (shared-tile variant
 * measured slower).
 */
export function srcAttnBwdDq(a: AttnDims): string {
  const L = attnLanes(a);
  return `
${bindLane(0, "QB", "read", L)}
${bindLane(1, "KB", "read", L)}
${bindLane(2, "VB", "read", L)}
${bindLane(3, "GB", "read", L)}
${bindF32(4, "LSE", "read")}
${bindF32(5, "DB", "read")}
${bindLane(6, "DQ", "read_write", L)}
${attnConsts(a)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= HQ * T) { return; }
  let h = i / T;
  let t = i % T;
  let kv = h / GROUP;
  var qr: array<${L.ty}, HDV>;
  var go: array<${L.ty}, HDV>;
  var dq: array<${L.ty}, HDV>;
  for (var d = 0u; d < HDV; d++) {
    qr[d] = QB[t * QSV + h * HDV + d];
    go[d] = GB[t * QSV + h * HDV + d];
    dq[d] = ${L.zero};
  }
  let lse = LSE[h * T + t];
  let dRow = DB[h * T + t];
  for (var s = winStartBlock(t); s <= t; s++) {
    var dotv = ${L.zero};
    var dpv = ${L.zero};
    for (var d = 0u; d < HDV; d++) {
      dotv = dotv + qr[d] * KB[s * KSV + kv * HDV + d];
      dpv = dpv + go[d] * VB[s * KSV + kv * HDV + d];
    }
    if (WINDOW != 0u && s + WINDOW <= t) { continue; } // key before this row's window
    let ds = exp2((${L.sum("dotv")}) * SCALE2 - lse) * ((${L.sum("dpv")}) - dRow) * SCALE;
    for (var d = 0u; d < HDV; d++) { dq[d] = dq[d] + ds * KB[s * KSV + kv * HDV + d]; }
  }
  for (var d = 0u; d < HDV; d++) {
    DQ[t * QSV + h * HDV + d] = DQ[t * QSV + h * HDV + d] + dq[d];
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
 * tiles per group head. Query tiles start at the block's first key: the
 * causal t>=s means earlier tiles can't contribute to any key in the block.
 * Staging pays off here (unlike srcAttnFwd/srcAttnBwdDq, ~1.8x at T=4096):
 * each thread's scan starts at its own key t=s, so direct loads would be
 * wavefront-divergent; the shared tile re-aligns the whole block to one
 * uniform t range. The tile depth BT is sized so the shared footprint always
 * fits the 16 KiB spec-default maxComputeWorkgroupStorageSize (attnBwdTile);
 * at HD=32 that is 32 rows and 8.3 KiB, at HD=64 it is 16 rows and 8.1 KiB.
 * Tail threads (s >= T) still run the cooperative loads: barriers must stay
 * uniform.
 */
export function srcAttnBwdDkv(a: AttnDims): string {
  const L = attnLanes(a);
  // Two independent accumulation chains through the head dimension. Unlike the
  // forward and dQ kernels (where it measured neutral), this one reads both dot
  // operands from workgroup memory, and splitting the dependency chain lets the
  // second load issue while the first is still in flight: 1.18-1.20x on M1 Max.
  // Emitted only for an even lane count; the odd case keeps the single chain
  // rather than carrying a tail term.
  const BT = attnBwdTile(a);
  const split = L.hdv % 2 === 0;
  const dots = split
    ? `          var dotv = ${L.zero};
          var dpv = ${L.zero};
          var dotv1 = ${L.zero};
          var dpv1 = ${L.zero};
          for (var d = 0u; d < HDV; d += 2u) {
            dotv = dotv + Qs[b + d] * kr[d];
            dpv = dpv + Gs[b + d] * vr[d];
            dotv1 = dotv1 + Qs[b + d + 1u] * kr[d + 1u];
            dpv1 = dpv1 + Gs[b + d + 1u] * vr[d + 1u];
          }
          dotv = dotv + dotv1;
          dpv = dpv + dpv1;`
    : `          var dotv = ${L.zero};
          var dpv = ${L.zero};
          for (var d = 0u; d < HDV; d++) {
            dotv = dotv + Qs[b + d] * kr[d];
            dpv = dpv + Gs[b + d] * vr[d];
          }`;
  return `
${bindLane(0, "QB", "read", L)}
${bindLane(1, "KB", "read", L)}
${bindLane(2, "VB", "read", L)}
${bindLane(3, "GB", "read", L)}
${bindF32(4, "LSE", "read")}
${bindF32(5, "DB", "read")}
${bindLane(6, "DK", "read_write", L)}
${bindLane(7, "DV", "read_write", L)}
${attnConsts(a)}
const BT: u32 = ${BT}u;
var<workgroup> Qs: array<${L.ty}, ${BT * L.hdv}>;
var<workgroup> Gs: array<${L.ty}, ${BT * L.hdv}>;
var<workgroup> Ls: array<f32, ${BT}>;
var<workgroup> Ds: array<f32, ${BT}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let kv = wg.y;
  let s = wg.x * 64u + li.x;
  let live = s < T;
  var kr: array<${L.ty}, HDV>;
  var vr: array<${L.ty}, HDV>;
  var dk: array<${L.ty}, HDV>;
  var dv: array<${L.ty}, HDV>;
  if (live) {
    for (var d = 0u; d < HDV; d++) {
      kr[d] = KB[s * KSV + kv * HDV + d];
      vr[d] = VB[s * KSV + kv * HDV + d];
      dk[d] = ${L.zero};
      dv[d] = ${L.zero};
    }
  }
  // Under a sliding window, keys in this 64-block [wg.x*64, wg.x*64+63] are only
  // attended by queries up to (lastKey)+WINDOW-1, so stop the tile scan there.
  let t0End = select(T, min(T, wg.x * 64u + 64u + WINDOW), WINDOW != 0u);
  for (var h = kv * GROUP; h < kv * GROUP + GROUP; h++) {
    for (var t0 = wg.x * 64u; t0 < t0End; t0 += BT) {
      for (var j = li.x; j < BT * HDV; j += 64u) {
        let t = t0 + j / HDV;
        let d = j % HDV;
        var qval = ${L.zero};
        var gval = ${L.zero};
        if (t < T) {
          qval = QB[t * QSV + h * HDV + d];
          gval = GB[t * QSV + h * HDV + d];
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
        var tEnd = min(t0 + BT, T);
        if (WINDOW != 0u) { tEnd = min(tEnd, s + WINDOW); }
        for (var t = max(t0, s); t < tEnd; t++) {
          let b = (t - t0) * HDV;
${dots}
          let p = exp2((${L.sum("dotv")}) * SCALE2 - Ls[t - t0]);
          let ds = p * ((${L.sum("dpv")}) - Ds[t - t0]) * SCALE;
          for (var d = 0u; d < HDV; d++) {
            dk[d] = dk[d] + ds * Qs[b + d];
            dv[d] = dv[d] + p * Gs[b + d];
          }
        }
      }
      workgroupBarrier();
    }
  }
  if (live) {
    for (var d = 0u; d < HDV; d++) {
      DK[s * KSV + kv * HDV + d] = DK[s * KSV + kv * HDV + d] + dk[d];
      DV[s * KSV + kv * HDV + d] = DV[s * KSV + kv * HDV + d] + dv[d];
    }
  }
}`;
}

/** Threads per row in the cross-entropy kernels; also the reduction width. */
const CE_WG = 256;

/**
 * Per-row softmax + NLL, one WORKGROUP per row (not one thread): every row is a
 * V-long reduction, and at one thread per row the whole kernel ran on T threads
 * with each lane striding V floats apart, so nothing coalesced and the GPU sat
 * mostly idle. A workgroup per row makes consecutive lanes read consecutive
 * logits and gives T*CE_WG-way parallelism.
 *
 * PROBS holds UNNORMALIZED exp(z - max) and INV holds each row's 1/Σ: the
 * normalizing pass was a third full [T,V] read-modify-write whose only job was a
 * divide the backward can apply from a per-row scalar. The loss comes straight
 * from the row statistics, log(Σ) - (z_target - max), which also drops the
 * epsilon the old form needed to survive log(0).
 */
export function srcCeFwd(T: number, V: number): string {
  const g = gridRows(T);
  return `
${bindF32(0, "LOG", "read")}
${bindU32(1, "TGT")}
${bindF32(2, "PROBS", "read_write")}
${bindF32(3, "LT", "read_write")}
${bindF32(4, "INV", "read_write")}
const T: u32 = ${T}u; const V: u32 = ${V}u; const GX: u32 = ${g.x}u;
var<workgroup> red: array<f32, ${CE_WG}>;
@compute @workgroup_size(${CE_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.y * GX + wg.x;
  if (t >= T) { return; }                              // uniform: barriers stay legal
  let base = t * V;
  var mx = -3.0e38;
  for (var v = li; v < V; v += ${CE_WG}u) { mx = max(mx, LOG[base + v]); }
  red[li] = mx;
  workgroupBarrier();
  for (var stride = ${CE_WG / 2}u; stride > 0u; stride = stride >> 1u) {
    if (li < stride) { red[li] = max(red[li], red[li + stride]); }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();  // red is reused for the sum below
  var sum = 0.0;
  for (var v = li; v < V; v += ${CE_WG}u) {
    let e = exp(LOG[base + v] - m);
    PROBS[base + v] = e;
    sum = sum + e;
  }
  red[li] = sum;
  workgroupBarrier();
  for (var stride = ${CE_WG / 2}u; stride > 0u; stride = stride >> 1u) {
    if (li < stride) { red[li] = red[li] + red[li + stride]; }
    workgroupBarrier();
  }
  let s = red[0];
  if (li == 0u) {
    INV[t] = 1.0 / s;
    let tgt = TGT[t];
    if (tgt == 0xffffffffu) { LT[t] = 0.0; }           // ignore-index: no loss
    else { LT[t] = log(s) - (LOG[base + tgt] - m); }
  }
}`;
}

/** Single-thread mean of the per-row losses: T is tiny; simplicity wins. */
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

/**
 * Soft-target (sparse teacher) forward: per-row softmax, then the k-term
 * cross-entropy -Σ_j q·log p. Also stashes each row's teacher mass S = Σ_j q,
 * which the backward needs to scale p by. Mirrors srcCeFwd; the reduce stage is
 * shared (srcCeReduce).
 */
export function srcSoftCeFwd(T: number, V: number, K: number): string {
  const g = gridRows(T);
  return `
${bindF32(0, "LOG", "read")}
${bindU32(1, "TID")}
${bindF32(2, "TQ", "read")}
${bindF32(3, "PROBS", "read_write")}
${bindF32(4, "LT", "read_write")}
${bindF32(5, "SMASS", "read_write")}
${bindF32(6, "INV", "read_write")}
const T: u32 = ${T}u; const V: u32 = ${V}u; const K: u32 = ${K}u; const GX: u32 = ${g.x}u;
var<workgroup> red: array<f32, ${CE_WG}>;
@compute @workgroup_size(${CE_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.y * GX + wg.x;
  if (t >= T) { return; }                              // uniform: barriers stay legal
  let base = t * V;
  var mx = -3.0e38;
  for (var v = li; v < V; v += ${CE_WG}u) { mx = max(mx, LOG[base + v]); }
  red[li] = mx;
  workgroupBarrier();
  for (var stride = ${CE_WG / 2}u; stride > 0u; stride = stride >> 1u) {
    if (li < stride) { red[li] = max(red[li], red[li + stride]); }
    workgroupBarrier();
  }
  let m = red[0];
  workgroupBarrier();  // red is reused for the sum below
  var sum = 0.0;
  for (var v = li; v < V; v += ${CE_WG}u) {
    let e = exp(LOG[base + v] - m);
    PROBS[base + v] = e;
    sum = sum + e;
  }
  red[li] = sum;
  workgroupBarrier();
  for (var stride = ${CE_WG / 2}u; stride > 0u; stride = stride >> 1u) {
    if (li < stride) { red[li] = red[li] + red[li + stride]; }
    workgroupBarrier();
  }
  let s = red[0];
  if (li == 0u) {
    INV[t] = 1.0 / s;
    if (TID[t * K] == 0xffffffffu) { LT[t] = 0.0; SMASS[t] = 0.0; return; }  // ignored row
    // -Σ q·log(p) with p = e/Σ, expanded so no probability is ever read back.
    var acc = 0.0;
    var mass = 0.0;
    for (var j = 0u; j < K; j++) {
      let q = TQ[t * K + j];
      acc = acc - q * (LOG[base + TID[t * K + j]] - m);
      mass = mass + q;
    }
    LT[t] = acc + mass * log(s);
    SMASS[t] = mass;
  }
}`;
}

/** Soft-target backward, dense half: dLogits += scale · S[t] · p. */
export function srcSoftCeBwdP(T: number, V: number, K: number): string {
  return `
${bindF32(0, "PROBS", "read")}
${bindU32(1, "TID")}
${bindF32(2, "SMASS", "read")}
${bindF32(3, "LG", "read")}
${bindF32(4, "DIV", "read")}
${bindF32(5, "DLOG", "read_write")}
${bindF32(6, "INV", "read")}
const N: u32 = ${T * V}u; const V: u32 = ${V}u; const K: u32 = ${K}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.y * ${grid2D(T * V).roww}u + gid.x;
  if (i >= N) { return; }
  let t = i / V;
  if (TID[t * K] == 0xffffffffu) { return; }           // ignored row: no gradient
  DLOG[i] = DLOG[i] + (LG[0] / DIV[0]) * SMASS[t] * PROBS[i] * INV[t];
}`;
}

/**
 * Soft-target backward, sparse half: dLogits[ids] -= scale · q. One thread per
 * ROW (not per teacher entry) so the k read-modify-writes into that row are
 * serialized by construction: duplicate ids in a row cannot race.
 */
export function srcSoftCeBwdQ(T: number, V: number, K: number): string {
  return `
${bindU32(0, "TID")}
${bindF32(1, "TQ", "read")}
${bindF32(2, "LG", "read")}
${bindF32(3, "DIV", "read")}
${bindF32(4, "DLOG", "read_write")}
const T: u32 = ${T}u; const V: u32 = ${V}u; const K: u32 = ${K}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  if (t >= T) { return; }
  if (TID[t * K] == 0xffffffffu) { return; }           // ignored row: no gradient
  let scale = LG[0] / DIV[0];
  for (var j = 0u; j < K; j++) {
    let i = t * V + TID[t * K + j];
    DLOG[i] = DLOG[i] - scale * TQ[t * K + j];
  }
}`;
}

export function srcCeBwd(T: number, V: number): string {
  return `
${bindF32(0, "PROBS", "read")}
${bindU32(1, "TGT")}
${bindF32(2, "LG", "read")}
${bindF32(3, "DIV", "read")}
${bindF32(4, "DLOG", "read_write")}
${bindF32(5, "INV", "read")}
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
  DLOG[i] = DLOG[i] + (LG[0] / DIV[0]) * (PROBS[i] * INV[t] - ind);
}`;
}
