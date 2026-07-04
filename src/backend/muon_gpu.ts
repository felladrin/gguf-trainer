// GPU-resident Muon: the exact math of ../train/muon.ts (momentum, optional
// Nesterov, quintic Newton–Schulz, sqrt(rows/cols) scaling) expressed as GPU
// dispatches, so Muon-group weights, momentum, and gradients never cross the
// PCIe/host boundary during training. The CPU implementation is the spec; the
// only differences here are f32 accumulation order (GEMM tiles, tree
// reductions), which the parity gates in tests/gpu_parity.ts bound.
//
// Why not implement Optimizer from ../train/optimizer.ts: step() there is
// synchronous and host-side. This class instead RECORDS its dispatches into
// the backend's shared encoder — call recordStep() after all backward passes
// of a step and before the next forward, so in queue order the optimizer
// consumes gradients after backward completes and lands weight updates before
// the next forward reads them, and both before the next step's deferred grad
// clears (flushed when the next backward begins). The aux group (embeddings,
// head, norms) keeps the host AdamW path from ../train/adam.ts unchanged,
// including its global grad-norm clipping over aux params only.
//
// State buffers come from createStateBuffer(): outside the pool, so nothing
// here is recycled at sync(), and momentum starts at guaranteed zero.

import { Tensor } from "../model/autograd.ts";
import { AdamW } from "../train/adam.ts";
import type { AdamOpts } from "../train/adam.ts";
import { bindF32, ceilDiv, f32lit } from "./webgpu.ts";
import type { GpuBuffer, WebGPUBackend } from "./webgpu.ts";

export interface MuonGpuOpts {
  lr: number;
  momentum?: number; // default 0.95
  nesterov?: boolean; // default true
  nsSteps?: number; // Newton-Schulz iterations, default 5
  aux: AdamOpts; // AdamW settings for the non-Muon param group (host-side)
}

// Same quintic coefficients as ../train/muon.ts.
const NS_A = 3.4445;
const NS_B = -4.7750;
const NS_C = 2.0315;

// Frobenius-norm reduction sizing: stage 1 writes one partial per workgroup,
// stage 2 folds the partials in a single workgroup, so cap the partial count
// at one workgroup's width.
const RED_WG = 256;
function reduceGroups(n: number): number {
  return Math.min(RED_WG, ceilDiv(n, RED_WG));
}

// --- WGSL sources (shapes/hyperparams baked, matching the backend's style) ------

/** buf = m·buf + grad; U = nesterov ? grad + m·buf : buf (buf already updated). */
function srcMomentum(n: number, momentum: number, nesterov: boolean): string {
  return `
${bindF32(0, "GB", "read")}
${bindF32(1, "BUF", "read_write")}
${bindF32(2, "UB", "read_write")}
const N: u32 = ${n}u; const M: f32 = ${f32lit(momentum)};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  BUF[i] = M * BUF[i] + GB[i];
  UB[i] = ${nesterov ? "GB[i] + M * BUF[i]" : "BUF[i]"};
}`;
}

/** Stage 1 of sum-of-squares: one grid-strided partial sum per workgroup. */
function srcSumSqPartial(n: number, groups: number): string {
  return `
${bindF32(0, "UB", "read")}
${bindF32(1, "PART", "read_write")}
const N: u32 = ${n}u; const STRIDE: u32 = ${groups * RED_WG}u;
var<workgroup> red: array<f32, ${RED_WG}>;
@compute @workgroup_size(${RED_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  var s = 0.0;
  for (var i = wg.x * ${RED_WG}u + li.x; i < N; i += STRIDE) { let v = UB[i]; s = s + v * v; }
  red[li.x] = s;
  workgroupBarrier();
  var off = ${RED_WG / 2}u;
  loop {
    if (li.x < off) { red[li.x] = red[li.x] + red[li.x + off]; }
    workgroupBarrier();
    if (off == 1u) { break; }
    off = off / 2u;
  }
  if (li.x == 0u) { PART[wg.x] = red[0]; }
}`;
}

/** Stage 2: fold the partials and store the Frobenius norm (sqrt of total). */
function srcSumSqFinal(groups: number): string {
  return `
${bindF32(0, "PART", "read")}
${bindF32(1, "NORM", "read_write")}
const G: u32 = ${groups}u;
var<workgroup> red: array<f32, ${RED_WG}>;
@compute @workgroup_size(${RED_WG})
fn main(@builtin(local_invocation_id) li: vec3<u32>) {
  var s = 0.0;
  for (var i = li.x; i < G; i += ${RED_WG}u) { s = s + PART[i]; }
  red[li.x] = s;
  workgroupBarrier();
  var off = ${RED_WG / 2}u;
  loop {
    if (li.x < off) { red[li.x] = red[li.x] + red[li.x + off]; }
    workgroupBarrier();
    if (off == 1u) { break; }
    off = off / 2u;
  }
  if (li.x == 0u) { NORM[0] = sqrt(red[0]); }
}`;
}

/**
 * X = (flip ? Uᵀ : U) / (frob + 1e-7): move U ([m,n]) into the Newton–Schulz
 * working orientation (smaller first dim) and Frobenius-normalize, exactly as
 * the CPU newtonSchulz() prologue.
 */
function srcNormalize(m: number, n: number, flip: boolean): string {
  // When flipped X is [n,m], so X[row,col] = U[col,row] = UB[col*n + row].
  const load = flip ? "UB[col * NCOLS + row]" : "UB[i]";
  return `
${bindF32(0, "UB", "read")}
${bindF32(1, "NORM", "read")}
${bindF32(2, "XB", "read_write")}
const N: u32 = ${m * n}u; const C: u32 = ${flip ? m : n}u; const NCOLS: u32 = ${n}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let row = i / C;
  let col = i % C;
  XB[i] = ${load} / (NORM[0] + 1e-7);
}`;
}

/** A ← b·A + c·A² (A² precomputed into AA; in place — thread i only touches i). */
function srcCombineA(n: number): string {
  return `
${bindF32(0, "AB", "read_write")}
${bindF32(1, "AAB", "read")}
const N: u32 = ${n}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  AB[i] = ${f32lit(NS_B)} * AB[i] + ${f32lit(NS_C)} * AAB[i];
}`;
}

/** X ← a·X + B·X (B·X precomputed into BX). */
function srcCombineX(n: number): string {
  return `
${bindF32(0, "XB", "read_write")}
${bindF32(1, "BXB", "read")}
const N: u32 = ${n}u;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  XB[i] = ${f32lit(NS_A)} * XB[i] + BXB[i];
}`;
}

/**
 * W ← W − factor·ortho, reading ortho straight out of X (transposing back on
 * the fly when the NS ran flipped) — the CPU path's final transpose + update
 * fused into one pass. factor bakes lr·sqrt(max(1, rows/cols)).
 */
function srcApply(m: number, n: number, flip: boolean, factor: number): string {
  // When flipped X is [n,m], so ortho[row,col] = X[col,row] = XB[col*m + row].
  const load = flip ? "XB[col * NROWS + row]" : "XB[i]";
  return `
${bindF32(0, "WB", "read_write")}
${bindF32(1, "XB", "read")}
const N: u32 = ${m * n}u; const C: u32 = ${n}u; const NROWS: u32 = ${m}u;
const FACTOR: f32 = ${f32lit(factor)};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let row = i / C;
  let col = i % C;
  WB[i] = WB[i] - FACTOR * ${load};
}`;
}

// --- Newton–Schulz recording (shared by the optimizer and the parity test) ------

interface NsBuffers {
  U: GpuBuffer; // NS input, [m,n] in the param's orientation
  X: GpuBuffer; // working matrix, [r,c] with r = min(m,n)
  A: GpuBuffer; // X·Xᵀ, then b·A + c·A² in place
  AA: GpuBuffer; // A²
  BX: GpuBuffer; // (b·A + c·A²)·X
  PART: GpuBuffer; // reduction partials
  NORM: GpuBuffer; // Frobenius norm, 1 element
}

function allocNs(gpu: WebGPUBackend, m: number, n: number, U: GpuBuffer): NsBuffers {
  const size = m * n;
  const r = Math.min(m, n);
  return {
    U,
    X: gpu.createStateBuffer(size * 4),
    A: gpu.createStateBuffer(r * r * 4),
    AA: gpu.createStateBuffer(r * r * 4),
    BX: gpu.createStateBuffer(size * 4),
    PART: gpu.createStateBuffer(reduceGroups(size) * 4),
    NORM: gpu.createStateBuffer(4),
  };
}

/**
 * Build the dispatch closures for the quintic Newton–Schulz on ns.U ([m,n]);
 * running them leaves the result in ns.X. Closures instead of immediate
 * recording because the optimizer replays the identical dispatch sequence
 * every step — see prepareDispatch in webgpu.ts.
 */
function buildNewtonSchulz(
  gpu: WebGPUBackend,
  ns: NsBuffers,
  m: number,
  n: number,
  steps: number,
): (() => void)[] {
  const size = m * n;
  const flip = m > n;
  const r = flip ? n : m;
  const c = flip ? m : n;
  const groups = reduceGroups(size);
  const ops = [
    gpu.prepareDispatch(srcSumSqPartial(size, groups), [ns.U, ns.PART], groups),
    gpu.prepareDispatch(srcSumSqFinal(groups), [ns.PART, ns.NORM], 1),
    gpu.prepareDispatch(srcNormalize(m, n, flip), [ns.U, ns.NORM, ns.X], ceilDiv(size, 256)),
  ];
  for (let s = 0; s < steps; s++) {
    ops.push(
      gpu.prepareGemm("NT", false, r, r, c, ns.X, ns.X, ns.A), // A = X·Xᵀ
      gpu.prepareGemm("NN", false, r, r, r, ns.A, ns.A, ns.AA), // AA = A²
      gpu.prepareDispatch(srcCombineA(r * r), [ns.A, ns.AA], ceilDiv(r * r, 256)),
      gpu.prepareGemm("NN", false, r, c, r, ns.A, ns.X, ns.BX), // BX = B·X
      gpu.prepareDispatch(srcCombineX(size), [ns.X, ns.BX], ceilDiv(size, 256)),
    );
  }
  return ops;
}

/**
 * Test support: GPU newtonSchulz() on raw row-major data, through the same
 * kernels and the same transpose-back store path the optimizer uses (apply
 * with factor −1 into a zeroed tensor yields +ortho).
 */
export async function newtonSchulzGpu(
  gpu: WebGPUBackend,
  g: Float32Array,
  m: number,
  n: number,
  steps: number,
): Promise<Float32Array> {
  const src = new Tensor(Float32Array.from(g), [m, n]);
  const out = Tensor.zeros([m, n]);
  const ns = allocNs(gpu, m, n, gpu.buffersFor(src).data);
  for (const op of buildNewtonSchulz(gpu, ns, m, n, steps)) op();
  gpu.prepareDispatch(
    srcApply(m, n, m > n, -1),
    [gpu.buffersFor(out).data, ns.X],
    ceilDiv(m * n, 256),
  )();
  await gpu.sync([out]);
  return out.data;
}

// --- The optimizer ---------------------------------------------------------------

export class MuonGpu {
  /** Host-stepped params (embeddings, head, norms); the trainer uploads these. */
  readonly auxParams: Tensor[];
  private gpu: WebGPUBackend;
  private muonParams: Tensor[];
  private aux: AdamW;
  private ops: (() => void)[]; // one step's dispatches, prepared once

  constructor(
    gpu: WebGPUBackend,
    muonParams: Tensor[],
    auxParams: Tensor[],
    opts: MuonGpuOpts,
  ) {
    this.gpu = gpu;
    this.muonParams = muonParams;
    this.auxParams = auxParams;
    this.aux = new AdamW(auxParams, opts.aux);
    const momentum = opts.momentum ?? 0.95;
    const nesterov = opts.nesterov ?? true;
    const nsSteps = opts.nsSteps ?? 5;
    this.ops = [];
    for (const p of muonParams) {
      if (p.shape.length !== 2) throw new Error("Muon param must be 2-D");
      const [m, n] = p.shape;
      // buffersFor uploads the current host weights once — the single warm-up
      // upload; from here on the device copy is authoritative.
      const bufs = gpu.buffersFor(p);
      gpu.keepGradOnDevice(p);
      const buf = gpu.createStateBuffer(p.size * 4); // momentum, zero at start
      const ns = allocNs(gpu, m, n, gpu.createStateBuffer(p.size * 4));
      const factor = opts.lr * Math.sqrt(Math.max(1, m / n));
      this.ops.push(
        gpu.prepareDispatch(
          srcMomentum(p.size, momentum, nesterov),
          [bufs.grad, buf, ns.U],
          ceilDiv(p.size, 256),
        ),
        ...buildNewtonSchulz(gpu, ns, m, n, nsSteps),
        gpu.prepareDispatch(
          srcApply(m, n, m > n, factor),
          [bufs.data, ns.X],
          ceilDiv(p.size, 256),
        ),
      );
    }
  }

  /** Aux host grads only: Muon-group grads live on device and the backend
   * clears them before each backward pass. */
  zeroGrad() {
    this.aux.zeroGrad();
  }

  /**
   * Record this step's Muon dispatches. Call after every backward() of the
   * step (grads complete in queue order) and before the next forward.
   */
  recordStep() {
    for (const op of this.ops) op();
  }

  /** Host AdamW over the aux group. Call after sync() has read aux grads back. */
  stepAux() {
    this.aux.step();
  }

  /** Copy device-resident Muon weights back to host arrays (sampling/export). */
  async syncWeightsToHost(): Promise<void> {
    await this.gpu.sync(this.muonParams);
  }
}
