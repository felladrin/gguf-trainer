// GPU-resident AdamW for the aux param group (embeddings, output head, norms):
// the exact math of ../train/adam.ts (global grad-norm clip, bias-corrected
// moments, coupled weight decay) expressed as GPU dispatches, so aux weights,
// moment buffers, and gradients never cross the host boundary during training.
//
// Why this matters: with the aux group host-side, every step read back its
// gradients (dominated by the embedding grad — ~143 KiB at tinyConfig, but
// vocab·hidden·4 grows to tens of MB at real scale) and re-uploaded its
// weights. Moving it on-device removes that per-step traffic entirely; after
// warm-up only the loss scalars come back.
//
// Like MuonGpu this RECORDS dispatches into the backend's shared encoder
// (recordStep()), replaying closures prepared once at construction. Static
// hyperparameters (betas, eps, weight decay, clip) are baked into WGSL; the
// per-step-varying ones — learning rate (WSD) and the two bias-correction
// denominators (1-beta^t) — live in a 3-element device buffer rewritten each
// step. The global grad-norm clip is a two-stage reduction across ALL aux
// params (per-param sum-of-squares -> a global norm -> a shared scale) that
// every param's update then reads, matching AdamW's clip-then-step order.

import type { Tensor } from "../model/autograd.ts";
import { bindF32, ceilDiv, f32lit } from "./webgpu.ts";
import type { GpuBuffer, WebGPUBackend } from "./webgpu.ts";
import type { AdamOpts } from "../train/adam.ts";

const RED_WG = 256;
function reduceGroups(n: number): number {
  return Math.min(RED_WG, ceilDiv(n, RED_WG));
}

/** Stage 1: per-workgroup grid-strided partial sum of grad². */
function srcSsqPartial(n: number, groups: number): string {
  return `
${bindF32(0, "GB", "read")}
${bindF32(1, "PART", "read_write")}
const N: u32 = ${n}u; const STRIDE: u32 = ${groups * RED_WG}u;
var<workgroup> red: array<f32, ${RED_WG}>;
@compute @workgroup_size(${RED_WG})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  var s = 0.0;
  for (var i = wg.x * ${RED_WG}u + li.x; i < N; i += STRIDE) { let v = GB[i]; s = s + v * v; }
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

/** Stage 2: fold this param's partials and store its sum-of-squares at SSQ[idx]. */
function srcSsqStore(groups: number, idx: number): string {
  return `
${bindF32(0, "PART", "read")}
${bindF32(1, "SSQ", "read_write")}
const G: u32 = ${groups}u; const IDX: u32 = ${idx}u;
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
  if (li.x == 0u) { SSQ[IDX] = red[0]; }
}`;
}

/**
 * Fold per-param SSQ into the global grad norm and store the clip scale:
 * scale = norm > clip ? clip / (norm + 1e-12) : 1, exactly as ../train/adam.ts.
 */
function srcClipScale(nParams: number, clip: number): string {
  return `
${bindF32(0, "SSQ", "read")}
${bindF32(1, "SCALE", "read_write")}
const N: u32 = ${nParams}u; const CLIP: f32 = ${f32lit(clip)};
@compute @workgroup_size(1)
fn main() {
  var s = 0.0;
  for (var i = 0u; i < N; i++) { s = s + SSQ[i]; }
  let norm = sqrt(s);
  if (norm > CLIP) { SCALE[0] = CLIP / (norm + 1e-12); } else { SCALE[0] = 1.0; }
}`;
}

/**
 * One AdamW element update. HYP = [lr, biasCorr1, biasCorr2] (per step); SCALE
 * is the shared grad-norm-clip factor (1.0 when clip disabled). Betas, eps and
 * weight decay are baked. Mirrors adam.ts: clip g, update m/v, bias-correct,
 * add coupled weight decay to the update, descend.
 */
function srcAdam(n: number, beta1: number, beta2: number, eps: number, wd: number): string {
  const decay = wd > 0 ? `upd = upd + ${f32lit(wd)} * WB[i];` : "";
  return `
${bindF32(0, "GB", "read")}
${bindF32(1, "MB", "read_write")}
${bindF32(2, "VB", "read_write")}
${bindF32(3, "WB", "read_write")}
${bindF32(4, "HYP", "read")}
${bindF32(5, "SCALE", "read")}
const N: u32 = ${n}u;
const B1: f32 = ${f32lit(beta1)}; const B2: f32 = ${f32lit(beta2)}; const EPS: f32 = ${f32lit(eps)};
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let g = GB[i] * SCALE[0];
  let m = B1 * MB[i] + (1.0 - B1) * g;
  let v = B2 * VB[i] + (1.0 - B2) * g * g;
  MB[i] = m;
  VB[i] = v;
  let mHat = m / HYP[1];
  let vHat = v / HYP[2];
  var upd = mHat / (sqrt(vHat) + EPS);
  ${decay}
  WB[i] = WB[i] - HYP[0] * upd;
}`;
}

export class AdamWGpu {
  readonly params: Tensor[];
  private gpu: WebGPUBackend;
  private opts: Required<AdamOpts>;
  private baseLr: number;
  private lrScale = 1;
  private t = 0;
  private hyp: GpuBuffer; // [lr, bc1, bc2], rewritten each step
  private ops: (() => void)[]; // reductions (if clip) + finalize + per-param adam

  constructor(gpu: WebGPUBackend, params: Tensor[], opts: AdamOpts) {
    this.gpu = gpu;
    this.params = params;
    this.opts = {
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0.0,
      clip: 1.0,
      ...opts,
    };
    this.baseLr = opts.lr;
    this.hyp = gpu.createStateBuffer(3 * 4);

    const o = this.opts;
    const clipOn = o.clip > 0;
    // Shared clip scale: stays 1.0 when clip is off (createStateBuffer zeroes,
    // so seed it); the finalize kernel overwrites it each step when clip is on.
    const scale = gpu.createStateBuffer(4);
    gpu.writeStateBuffer(scale, Float32Array.of(1));

    const reductions: (() => void)[] = [];
    let ssq: GpuBuffer | null = null;
    if (clipOn) {
      ssq = gpu.createStateBuffer(params.length * 4);
      params.forEach((p, i) => {
        const groups = reduceGroups(p.size);
        const part = gpu.createStateBuffer(groups * 4);
        const bufs = gpu.buffersFor(p); // uploads host weights once; grad on device
        reductions.push(
          gpu.prepareDispatch(srcSsqPartial(p.size, groups), [bufs.grad, part], groups, 1, "adamw"),
          gpu.prepareDispatch(srcSsqStore(groups, i), [part, ssq!], 1, 1, "adamw"),
        );
      });
      reductions.push(
        gpu.prepareDispatch(srcClipScale(params.length, o.clip), [ssq, scale], 1, 1, "adamw"),
      );
    }

    const adam: (() => void)[] = params.map((p) => {
      const bufs = gpu.buffersFor(p); // no-op re-fetch if already created above
      gpu.keepGradOnDevice(p);
      const m = gpu.createStateBuffer(p.size * 4);
      const v = gpu.createStateBuffer(p.size * 4);
      return gpu.prepareDispatch(
        srcAdam(p.size, o.beta1, o.beta2, o.eps, o.weightDecay),
        [bufs.grad, m, v, bufs.data, this.hyp, scale],
        ceilDiv(p.size, 256),
        1,
        "adamw",
      );
    });

    this.ops = [...reductions, ...adam];
  }

  /** Set the effective lr to `scale` × the constructed base lr (WSD schedule). */
  setLrScale(scale: number) {
    this.lrScale = scale;
  }

  /**
   * Record this step's AdamW dispatches. Writes the per-step hyperparameters
   * (lr, bias corrections) before replaying the prepared closures — the write
   * is queue-ordered before the optimizer submit, like uploadParams.
   */
  recordStep() {
    this.t += 1;
    const o = this.opts;
    const bc1 = 1 - Math.pow(o.beta1, this.t);
    const bc2 = 1 - Math.pow(o.beta2, this.t);
    this.gpu.writeStateBuffer(this.hyp, Float32Array.of(this.baseLr * this.lrScale, bc1, bc2));
    for (const op of this.ops) op();
  }

  /** No-op: grads are device-resident and the backend clears them per step. */
  zeroGrad() {}

  /** Copy device-resident aux weights back to host (sampling/export). */
  async syncWeightsToHost(): Promise<void> {
    await this.gpu.sync(this.params);
  }
}
