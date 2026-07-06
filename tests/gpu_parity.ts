// GPU-vs-CPU parity: the validation gate for the WebGPU backend.
//
// For every op, in the bring-up order from docs/HANDOFF.md: run the CPU
// reference forward + backward, then the identical call routed through the
// WebGPU backend, and compare outputs and input gradients element-wise. The
// CPU side is already finite-difference-validated by tests/gradcheck.ts, so
// agreement here transfers that trust to the kernels. Additionally: a
// finite-difference check run directly against the GPU matmul (the harness
// composes on-device), a two-micro-batch gradient accumulation check, and a
// whole-model forward/backward parity check.
//
// Run:  deno run tests/gpu_parity.ts     (Node/Bun have no WebGPU: prints SKIP)

import {
  add,
  attention,
  backward,
  crossEntropy,
  embedding,
  gelu,
  linear,
  mul,
  mulberry32,
  randn,
  rmsNorm,
  rmsNormHeads,
  rope,
  scale,
  silu,
  Tensor,
} from "../src/model/autograd.ts";
import { Qwen3Model } from "../src/model/qwen3.ts";
import { Gemma3Model } from "../src/model/gemma3.ts";
import type { Gemma3Config, Qwen3Config } from "../src/model/config.ts";
import { Muon, newtonSchulz } from "../src/train/muon.ts";
import { trainLM } from "../src/train/trainer.ts";
import { wsdSchedule } from "../src/train/schedule.ts";
import { qkLogitScale } from "../src/train/qk_clip.ts";
import { AdamW } from "../src/train/adam.ts";
import { initWebGPU, WebGPUBackend } from "../src/backend/webgpu.ts";
import { MAX_WG } from "../src/backend/wgsl.ts";
import { MuonGpu, newtonSchulzGpu } from "../src/backend/muon_gpu.ts";
import { AdamWGpu } from "../src/backend/adamw_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";

// Same math, different summation order: f32 accumulation differences grow with
// reduction depth, so backward (which chains more reductions) gets more slack.
const FWD = { atol: 2e-4, rtol: 2e-3 };
const BWD = { atol: 1e-3, rtol: 1e-2 };

let failures = 0;

function randTensor(shape: number[], rng: () => number, scale = 0.8): Tensor {
  const t = Tensor.zeros(shape, true);
  for (let i = 0; i < t.data.length; i++) t.data[i] = randn(rng) * scale;
  return t;
}

function compare(
  label: string,
  got: Float32Array,
  want: Float32Array,
  tol: { atol: number; rtol: number },
): boolean {
  let maxAbs = 0;
  let worst = -1;
  for (let i = 0; i < want.length; i++) {
    const abs = Math.abs(got[i] - want[i]);
    if (abs > maxAbs) {
      maxAbs = abs;
      worst = i;
    }
    if (abs > tol.atol + tol.rtol * Math.max(Math.abs(got[i]), Math.abs(want[i]))) {
      console.log(`    MISMATCH ${label}[${i}]: gpu=${got[i]} cpu=${want[i]}`);
      return false;
    }
  }
  void worst;
  return true;
}

/** CPU reference pass, then the same call on the GPU, then element compare. */
async function opCase(
  gpu: WebGPUBackend,
  name: string,
  inputs: Tensor[],
  fwd: () => Tensor,
) {
  // CPU reference.
  for (const t of inputs) t.zeroGrad();
  const cpuOut = fwd();
  const rngR = mulberry32(0xbeef);
  const r = new Float32Array(cpuOut.data.length);
  for (let i = 0; i < r.length; i++) r[i] = cpuOut.data.length === 1 ? 1 : rngR() * 2 - 1;
  cpuOut.grad.set(r);
  cpuOut._backward();
  const cpuData = cpuOut.data.slice();
  const cpuGrads = inputs.map((t) => t.grad.slice());

  // GPU, identical call routed through the backend.
  for (const t of inputs) t.zeroGrad();
  gpu.install();
  let ok = true;
  try {
    const gpuOut = fwd();
    gpuOut.grad.set(r);
    gpu.seedGradFromHost(gpuOut);
    gpuOut._backward();
    await gpu.sync([gpuOut]);
    ok = compare(`${name}.out`, gpuOut.data, cpuData, FWD) && ok;
    for (let k = 0; k < inputs.length; k++) {
      ok = compare(`${name}.dInput${k}`, inputs[k].grad, cpuGrads[k], BWD) && ok;
    }
  } finally {
    gpu.uninstall();
  }
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} ${name}`);
}

/**
 * timestamp-query profiler smoke check: a profiled forward must attribute time
 * to the right op labels, with sane counts and no slot overflow. Numerics are
 * covered by every other case (profiling is a no-op on results); this only
 * guards the profiling plumbing. Skips cleanly when the device lacks the feature.
 */
async function profilerSmoke(gpu: WebGPUBackend) {
  if (!gpu.timestampSupported) {
    console.log("  skip timestamp-query profiler (feature unavailable)");
    return;
  }
  const x = randTensor([40, 48], mulberry32(5));
  const w = randTensor([40, 48], mulberry32(6));
  gpu.install();
  let ok = true;
  try {
    gpu.startProfile(64);
    const y = silu(linear(x, w));
    await gpu.sync([y]);
    const { kernels, overflow } = gpu.stopProfile();
    const labels = new Set(kernels.map((k) => k.label));
    ok = kernels.length > 0 && labels.has("linear") && labels.has("silu") && !overflow &&
      kernels.every((k) => k.ms >= 0 && k.count > 0);
  } finally {
    gpu.uninstall();
  }
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} timestamp-query profiler (labels + times)`);
}

/**
 * Force the 2-D workgroup-grid fold (flat per-element path): a dispatch with
 * more than MAX_WG workgroups in x overflows WebGPU's per-dimension cap, so the
 * backend splits it across (x, y) and each kernel rebuilds its flat index. Sized
 * just past the cap so the tail spills into a second grid row; every element
 * must be summed exactly once (no skipped tail, no double-count).
 */
async function flatOverflowGate(gpu: WebGPUBackend) {
  const n = MAX_WG * 256 + 777; // 65535*256 + 777 -> ceilDiv(n,256) = 65538 > cap
  const a = Tensor.zeros([n]);
  const b = Tensor.zeros([n]);
  for (let i = 0; i < n; i++) {
    a.data[i] = 1;
    b.data[i] = 2;
  }
  gpu.install();
  let ok = true;
  try {
    const y = add(a, b);
    await gpu.sync([y]);
    let bad = 0;
    for (let i = 0; i < n; i++) if (y.data[i] !== 3) bad++;
    ok = bad === 0;
    if (!ok) failures++;
    console.log(`  ${ok ? "ok " : "FAIL"} dispatch 2-D fold, flat (add n=${n}, ${bad} wrong)`);
  } finally {
    gpu.uninstall();
  }
}

/**
 * Mixed-precision GEMM: with precision "f16" the matmul rounds operands to f16
 * and accumulates in f32. Validate against the f32 CPU reference at a loose
 * tolerance (f16 keeps ~10 mantissa bits; the K reduction stays f32). Guards the
 * correctness of the f16 path; the throughput win itself is on packed-f16 GPUs.
 */
async function f16GemmCheck(gpu: WebGPUBackend) {
  if (!gpu.f16Supported) {
    console.log("  skip f16 GEMM (shader-f16 unavailable)");
    return;
  }
  const rng = mulberry32(0xf16);
  const x = randTensor([40, 48], rng);
  const w = randTensor([32, 48], rng);
  x.zeroGrad();
  w.zeroGrad();
  const cpu = linear(x, w); // f32 reference
  const r = new Float32Array(cpu.data.length);
  const rr = mulberry32(1);
  for (let i = 0; i < r.length; i++) r[i] = rr() * 2 - 1;
  cpu.grad.set(r);
  cpu._backward();
  const cData = cpu.data.slice(), cdx = x.grad.slice(), cdw = w.grad.slice();

  x.zeroGrad();
  w.zeroGrad();
  gpu.setPrecision("f16");
  gpu.install();
  let ok = true;
  try {
    const y = linear(x, w);
    y.grad.set(r);
    gpu.seedGradFromHost(y);
    y._backward();
    await gpu.sync([y]);
    const tol = { atol: 1e-2, rtol: 2e-2 };
    ok = compare("f16 linear.out", y.data, cData, tol) &&
      compare("f16 linear.dX", x.grad, cdx, tol) &&
      compare("f16 linear.dW", w.grad, cdw, tol);
  } finally {
    gpu.uninstall();
    gpu.setPrecision("f32");
  }
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} f16 GEMM vs f32 CPU (linear, loose tol)`);
}

/** Finite differences straight against GPU forwards (samples a few elements). */
async function gpuMatmulFdCheck(gpu: WebGPUBackend) {
  const rng = mulberry32(31337);
  const x = randTensor([5, 7], rng);
  const w = randTensor([6, 7], rng);

  gpu.install();
  let bad = 0;
  try {
    const fwd = () => linear(x, w);
    const out0 = fwd();
    const rngR = mulberry32(0xbeef);
    const r = new Float32Array(out0.data.length);
    for (let i = 0; i < r.length; i++) r[i] = rngR() * 2 - 1;
    x.zeroGrad();
    w.zeroGrad();
    out0.grad.set(r);
    gpu.seedGradFromHost(out0);
    out0._backward();
    await gpu.sync([out0]);
    const analytic = [x.grad.slice(), w.grad.slice()];

    const evalLoss = async (): Promise<number> => {
      const out = fwd();
      await gpu.sync([out]);
      let s = 0;
      for (let i = 0; i < out.data.length; i++) s += out.data[i] * r[i];
      return s;
    };

    const eps = 1e-2;
    const tensors = [x, w];
    const pick = mulberry32(0xcafe);
    for (let k = 0; k < tensors.length; k++) {
      const t = tensors[k];
      for (let n = 0; n < 6; n++) {
        const i = Math.floor(pick() * t.data.length);
        const orig = t.data[i];
        const xp = Math.fround(orig + eps);
        const xm = Math.fround(orig - eps);
        t.data[i] = xp;
        gpu.uploadParams([t]);
        const lp = await evalLoss();
        t.data[i] = xm;
        gpu.uploadParams([t]);
        const lm = await evalLoss();
        t.data[i] = orig;
        gpu.uploadParams([t]);
        const fd = (lp - lm) / (xp - xm);
        const g = analytic[k][i];
        if (Math.abs(fd - g) > 2e-3 + 1.5e-2 * Math.max(Math.abs(fd), Math.abs(g))) {
          bad++;
          console.log(`    FD MISMATCH tensor#${k}[${i}]: analytic=${g} fd=${fd}`);
        }
      }
    }
  } finally {
    gpu.uninstall();
  }
  if (bad > 0) failures++;
  console.log(`  ${bad === 0 ? "ok " : "FAIL"} matmul finite-difference on GPU (12 sampled elems)`);
}

function microConfig(): Qwen3Config {
  return {
    vocabSize: 50,
    hiddenSize: 32,
    nLayers: 2,
    nHeads: 4,
    nKVHeads: 2,
    headDim: 8,
    ffnDim: 64,
    ropeBase: 10000,
    rmsEps: 1e-6,
    maxSeq: 32,
    tieEmbeddings: true,
  };
}

async function modelParity(gpu: WebGPUBackend) {
  const cfg = microConfig();
  const model = new Qwen3Model(cfg, mulberry32(5));
  const rng = mulberry32(11);
  const T = 12;
  const ids = Array.from({ length: T }, () => Math.floor(rng() * cfg.vocabSize));
  const targets = Array.from({ length: T }, () => Math.floor(rng() * cfg.vocabSize));
  const params = model.params();

  // CPU reference.
  for (const p of params) p.zeroGrad();
  const cpuLogits = model.forward(ids);
  const cpuLoss = crossEntropy(cpuLogits, targets);
  backward(cpuLoss, 1);
  const cpuLogitsData = cpuLogits.data.slice();
  const cpuLossVal = cpuLoss.data[0];
  const cpuGrads = params.map((p) => p.grad.slice());

  // GPU.
  for (const p of params) p.zeroGrad();
  gpu.install();
  let ok = true;
  try {
    const logits = model.forward(ids);
    const loss = crossEntropy(logits, targets);
    backward(loss, 1);
    await gpu.sync([logits, loss]);

    ok = compare("model.logits", logits.data, cpuLogitsData, { atol: 5e-4, rtol: 1e-2 }) && ok;
    const lossAbs = Math.abs(loss.data[0] - cpuLossVal);
    if (lossAbs > 1e-3 + 1e-3 * Math.abs(cpuLossVal)) {
      console.log(`    MISMATCH loss: gpu=${loss.data[0]} cpu=${cpuLossVal}`);
      ok = false;
    }
    for (let i = 0; i < params.length; i++) {
      ok = compare(`model.dParam${i}`, params[i].grad, cpuGrads[i], BWD) && ok;
    }
  } finally {
    gpu.uninstall();
  }
  if (!ok) failures++;
  console.log(
    `  ${ok ? "ok " : "FAIL"} qwen3 model forward + full backward (${params.length} param tensors)`,
  );
}

/**
 * Gemma3 whole-model parity: exercises the arch's distinctive path (sqrt(hidden)
 * embed scale, sandwich norms, GeGLU, per-layer SWA + local/global RoPE). The
 * config mixes SWA layers (0,1,3) with a global layer (2) and T > slidingWindow
 * so the window genuinely restricts, matching the CPU windowed reference.
 */
async function gemma3ModelParity(gpu: WebGPUBackend) {
  const cfg: Gemma3Config = {
    vocabSize: 50,
    hiddenSize: 32,
    nLayers: 4,
    nHeads: 4,
    nKVHeads: 2,
    headDim: 8,
    ffnDim: 64,
    ropeBase: 1_000_000,
    ropeBaseLocal: 10_000,
    rmsEps: 1e-6,
    maxSeq: 32,
    tieEmbeddings: true,
    slidingWindow: 5,
    swaPattern: 3,
  };
  const model = new Gemma3Model(cfg, mulberry32(5));
  const rng = mulberry32(11);
  const T = 14;
  const ids = Array.from({ length: T }, () => Math.floor(rng() * cfg.vocabSize));
  const targets = Array.from({ length: T }, () => Math.floor(rng() * cfg.vocabSize));
  const params = model.params();

  for (const p of params) p.zeroGrad();
  const cpuLogits = model.forward(ids);
  const cpuLoss = crossEntropy(cpuLogits, targets);
  backward(cpuLoss, 1);
  const cpuLogitsData = cpuLogits.data.slice();
  const cpuLossVal = cpuLoss.data[0];
  const cpuGrads = params.map((p) => p.grad.slice());

  for (const p of params) p.zeroGrad();
  gpu.install();
  let ok = true;
  try {
    const logits = model.forward(ids);
    const loss = crossEntropy(logits, targets);
    backward(loss, 1);
    await gpu.sync([logits, loss]);
    ok = compare("gemma3.logits", logits.data, cpuLogitsData, { atol: 5e-4, rtol: 1e-2 }) && ok;
    const lossAbs = Math.abs(loss.data[0] - cpuLossVal);
    if (lossAbs > 1e-3 + 1e-3 * Math.abs(cpuLossVal)) {
      console.log(`    MISMATCH loss: gpu=${loss.data[0]} cpu=${cpuLossVal}`);
      ok = false;
    }
    for (let i = 0; i < params.length; i++) {
      ok = compare(`gemma3.dParam${i}`, params[i].grad, cpuGrads[i], BWD) && ok;
    }
  } finally {
    gpu.uninstall();
  }
  if (!ok) failures++;
  console.log(
    `  ${
      ok ? "ok " : "FAIL"
    } gemma3 model forward + full backward (${params.length} param tensors)`,
  );
}

/** Two micro-batches accumulated before one sync must match CPU accumulation. */
async function accumulationParity(gpu: WebGPUBackend) {
  const cfg = microConfig();
  const model = new Qwen3Model(cfg, mulberry32(5));
  const rng = mulberry32(23);
  const T = 9;
  const batch = () => ({
    ids: Array.from({ length: T }, () => Math.floor(rng() * cfg.vocabSize)),
    targets: Array.from({ length: T }, () => Math.floor(rng() * cfg.vocabSize)),
  });
  const b0 = batch();
  const b1 = batch();
  const params = model.params();

  for (const p of params) p.zeroGrad();
  backward(crossEntropy(model.forward(b0.ids), b0.targets), 0.5);
  backward(crossEntropy(model.forward(b1.ids), b1.targets), 0.5);
  const cpuGrads = params.map((p) => p.grad.slice());

  for (const p of params) p.zeroGrad();
  gpu.install();
  let ok = true;
  try {
    backward(crossEntropy(model.forward(b0.ids), b0.targets), 0.5);
    backward(crossEntropy(model.forward(b1.ids), b1.targets), 0.5);
    await gpu.sync();
    for (let i = 0; i < params.length; i++) {
      ok = compare(`accum.dParam${i}`, params[i].grad, cpuGrads[i], BWD) && ok;
    }
  } finally {
    gpu.uninstall();
  }
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} gradient accumulation across 2 micro-batches`);
}

/**
 * Functional large-T check for flash attention (appended by the tiling task):
 * on a device requested with SPEC-DEFAULT limits, run forward+backward at
 * T=3584 (Hq=4, hd=32). The pre-flash kernels needed a ~205 MB [Hq,T,T] probs
 * binding here — over the default 128 MiB maxStorageBufferBindingSize — and
 * failed bind-group validation. CPU comparison is far too slow at this T, so
 * assert completion with no device/validation error and finite, non-zero
 * output and input gradients instead.
 */
async function flashLargeTCheck() {
  // deno-lint-ignore no-explicit-any
  const nav: any = (globalThis as any).navigator;
  const adapter = await nav?.gpu?.requestAdapter?.();
  if (!adapter) {
    console.log("  SKIP flash large-T check: no WebGPU adapter");
    return;
  }
  const device = await adapter.requestDevice(); // no requiredLimits: spec defaults
  const gpu2 = new WebGPUBackend(device, "spec-default-limits");
  const T = 3584, Hq = 4, Hkv = 2, hd = 32;
  const rng = mulberry32(0xf1a5);
  const q = randTensor([T, Hq * hd], rng, 0.5);
  const k = randTensor([T, Hkv * hd], rng, 0.5);
  const v = randTensor([T, Hkv * hd], rng, 0.5);
  let ok = true;
  gpu2.install();
  try {
    device.pushErrorScope?.("validation");
    const out = attention(q, k, v, T, Hq, Hkv, hd);
    const rngR = mulberry32(0xbeef);
    for (let i = 0; i < out.grad.length; i++) out.grad[i] = rngR() * 2 - 1;
    gpu2.seedGradFromHost(out);
    out._backward();
    await gpu2.sync([out]);
    const err = await device.popErrorScope?.();
    if (err) {
      console.log(`    validation error: ${err.message}`);
      ok = false;
    }
    const arrays: [string, Float32Array][] = [
      ["out", out.data],
      ["dQ", q.grad],
      ["dK", k.grad],
      ["dV", v.grad],
    ];
    for (const [label, arr] of arrays) {
      let maxAbs = 0;
      let finite = true;
      for (let i = 0; i < arr.length; i++) {
        if (!Number.isFinite(arr[i])) {
          finite = false;
          break;
        }
        const a = Math.abs(arr[i]);
        if (a > maxAbs) maxAbs = a;
      }
      if (!finite || maxAbs === 0) {
        console.log(`    ${label}: finite=${finite} maxAbs=${maxAbs}`);
        ok = false;
      }
    }
  } finally {
    gpu2.destroy(); // uninstalls and frees its pooled buffers
  }
  if (!ok) failures++;
  console.log(
    `  ${ok ? "ok " : "FAIL"} flash attention fwd+bwd @ T=3584 under spec-default limits`,
  );
}

async function main() {
  const gpu = await initWebGPU();
  if (!gpu) {
    console.log("SKIP: no WebGPU in this runtime — run under Deno (or provide navigator.gpu).");
    return;
  }
  console.log(`=== GPU-vs-CPU parity checks (adapter: ${gpu.adapterName}) ===\n`);
  const rng = mulberry32(1234);

  // 1. matmul / linear — including a multi-tile case with non-multiple-of-16 dims.
  {
    const x = randTensor([5, 7], rng);
    const w = randTensor([6, 7], rng);
    await opCase(gpu, "linear (5x7 · 6x7ᵀ)", [x, w], () => linear(x, w));
  }
  {
    const x = randTensor([33, 48], rng);
    const w = randTensor([37, 48], rng);
    await opCase(gpu, "linear (33x48 · 37x48ᵀ, tiled)", [x, w], () => linear(x, w));
  }
  {
    // Straddles the register-tiled block dims (BM=BN=64, BK=8) in all three
    // axes with remainders: 2 row-blocks + tail, 2 col-blocks + tail, 10 K-steps.
    const x = randTensor([70, 80], rng);
    const w = randTensor([75, 80], rng);
    await opCase(gpu, "linear (70x80 · 75x80ᵀ, multi-block)", [x, w], () => linear(x, w));
  }
  await gpuMatmulFdCheck(gpu);
  await f16GemmCheck(gpu);
  await profilerSmoke(gpu);
  await flatOverflowGate(gpu);
  {
    // Row-per-workgroup 2-D fold: rmsNormHeads with rows = T*H past the cap.
    const T = 8200, H = 8, hd = 4; // rows = 65600 > MAX_WG (65535)
    const x = randTensor([T, H * hd], rng);
    const w = Tensor.zeros([hd], true);
    for (let i = 0; i < hd; i++) w.data[i] = 1 + 0.3 * randn(rng);
    await opCase(
      gpu,
      "dispatch 2-D fold, rows (rmsNormHeads rows=65600)",
      [x, w],
      () => rmsNormHeads(x, w, T, H, hd, 1e-6),
    );
  }

  // 2. elementwise
  {
    const a = randTensor([4, 6], rng);
    const b = randTensor([4, 6], rng);
    await opCase(gpu, "add", [a, b], () => add(a, b));
  }
  {
    const a = randTensor([4, 6], rng);
    const b = randTensor([4, 6], rng);
    await opCase(gpu, "mul", [a, b], () => mul(a, b));
  }
  {
    const x = randTensor([5, 7], rng, 1.5);
    await opCase(gpu, "silu", [x], () => silu(x));
  }
  {
    const x = randTensor([5, 7], rng, 1.5);
    await opCase(gpu, "gelu", [x], () => gelu(x));
  }
  {
    const x = randTensor([4, 6], rng);
    await opCase(gpu, "scale", [x], () => scale(x, 2.5));
  }

  // 3. reductions
  {
    const x = randTensor([4, 40], rng);
    const w = Tensor.zeros([40], true);
    for (let i = 0; i < 40; i++) w.data[i] = 1 + 0.3 * randn(rng);
    await opCase(gpu, "rmsNorm", [x, w], () => rmsNorm(x, w, 1e-6));
  }
  {
    const T = 3, H = 2, hd = 8;
    const x = randTensor([T, H * hd], rng);
    const w = Tensor.zeros([hd], true);
    for (let i = 0; i < hd; i++) w.data[i] = 1 + 0.3 * randn(rng);
    await opCase(gpu, "rmsNormHeads", [x, w], () => rmsNormHeads(x, w, T, H, hd, 1e-6));
  }

  // 4. embedding, rope, attention, crossEntropy
  {
    const w = randTensor([7, 4], rng);
    const ids = [0, 3, 3, 6, 1, 3];
    await opCase(gpu, "embedding", [w], () => embedding(w, ids));
  }
  {
    const T = 4, H = 2, hd = 6;
    const x = randTensor([T, H * hd], rng);
    await opCase(gpu, "rope", [x], () => rope(x, T, H, hd, 10000));
    await opCase(gpu, "rope(posOffset=5)", [x], () => rope(x, T, H, hd, 10000, 5));
  }
  {
    const T = 6, Hq = 4, Hkv = 2, hd = 8;
    const q = randTensor([T, Hq * hd], rng);
    const k = randTensor([T, Hkv * hd], rng);
    const v = randTensor([T, Hkv * hd], rng);
    await opCase(gpu, "attention(GQA)", [q, k, v], () => attention(q, k, v, T, Hq, Hkv, hd));
  }
  {
    const T = 5, V = 17;
    const logits = randTensor([T, V], rng);
    const targets = [2, 7, 2, 0, 16];
    await opCase(gpu, "crossEntropy", [logits], () => crossEntropy(logits, targets));
  }
  {
    // Ignore-index (-1) = assistant-only loss masking: masked rows contribute
    // no loss and no gradient; the mean is over kept rows. GPU must match CPU.
    const T = 8, V = 17;
    const logits = randTensor([T, V], rng);
    const targets = [-1, 3, -1, 11, 0, -1, 16, 5];
    await opCase(gpu, "crossEntropy (ignore-index)", [logits], () => crossEntropy(logits, targets));
  }

  // 5. graph-level
  await modelParity(gpu);
  await gemma3ModelParity(gpu);
  await accumulationParity(gpu);

  // 6. Attention kernels (appended by the tiling task):
  //    (a) Materialized path at small T (T < attnFlashMinT): non-multiples of
  //        tile size, both head dims, both GQA group sizes.
  //    (b) Flash path forced at the same T by temporarily lowering attnFlashMinT
  //        to 1 — proves the flash kernels correct at small T without changing
  //        the production threshold.
  //    (c) Backward-heavy case that exercises srcAttnBwdDkv's GQA head-group loop.
  //    (d) Large-T functional check under spec-default device limits.
  const flashCases: [number, number, number, number, string][] = [
    [67, 4, 2, 6, "T=67, hd=6, group=2"],
    [67, 2, 2, 32, "T=67, hd=32, group=1"],
    [130, 4, 2, 32, "T=130, hd=32, group=2"],
    [130, 3, 3, 6, "T=130, hd=6, group=1"],
    [193, 4, 1, 24, "T=193, hd=24, group=4"],
  ];
  for (const [T, Hq, Hkv, hd, label] of flashCases) {
    const q = randTensor([T, Hq * hd], rng);
    const k = randTensor([T, Hkv * hd], rng);
    const v = randTensor([T, Hkv * hd], rng);
    // Materialized path (default threshold keeps T < 2048 on the old kernels).
    await opCase(
      gpu,
      `attention(${label}) mat`,
      [q, k, v],
      () => attention(q, k, v, T, Hq, Hkv, hd),
    );
    // Flash path forced: same inputs, just the kernel path changes.
    gpu.attnFlashMinT = 1;
    await opCase(
      gpu,
      `attention(${label}) flash`,
      [q, k, v],
      () => attention(q, k, v, T, Hq, Hkv, hd),
    );
    gpu.attnFlashMinT = 2048;
  }

  //    (e) Sliding-window attention (Gemma3 SWA layers): each query t attends
  //        keys [t-W+1, t]. Window chosen < T (and not tile-aligned) so it
  //        genuinely restricts. Both paths must match the CPU windowed ref.
  const windowCases: [number, number, number, number, number, string][] = [
    [193, 4, 2, 24, 48, "T=193, hd=24, group=2, W=48"],
    [130, 3, 3, 6, 40, "T=130, hd=6, group=1, W=40"],
    [67, 4, 2, 32, 20, "T=67, hd=32, group=2, W=20"],
  ];
  for (const [T, Hq, Hkv, hd, W, label] of windowCases) {
    const q = randTensor([T, Hq * hd], rng);
    const k = randTensor([T, Hkv * hd], rng);
    const v = randTensor([T, Hkv * hd], rng);
    await opCase(
      gpu,
      `attention(${label}) mat`,
      [q, k, v],
      () => attention(q, k, v, T, Hq, Hkv, hd, W),
    );
    gpu.attnFlashMinT = 1;
    await opCase(
      gpu,
      `attention(${label}) flash`,
      [q, k, v],
      () => attention(q, k, v, T, Hq, Hkv, hd, W),
    );
    gpu.attnFlashMinT = 2048;
  }
  await flashLargeTCheck();

  // 7. GPU-resident Muon optimizer (src/backend/muon_gpu.ts): Newton–Schulz
  //    kernel parity, momentum-buffer persistence across steps, and the
  //    HANDOFF-mandated whole-trajectory parity against the CPU Muon.
  await newtonSchulzParity(gpu);
  await muonMomentumPersistence(gpu);
  await adamwGpuParity(gpu);
  await muonTrajectoryParity(gpu);
  await wsdScheduleParity(gpu);
  await qkClipTrajectoryParity(gpu);

  // 8. sync() must fence GPU completion even when it reads nothing back.
  await syncFenceGate(gpu);

  console.log(
    failures === 0 ? "\n=== all parity checks passed ===" : `\n=== ${failures} FAILURES ===`,
  );
  if (failures > 0) {
    // deno-lint-ignore no-explicit-any
    const proc = (globalThis as any).process;
    if (proc?.exit) proc.exit(1);
  }
}

main().catch((e) => {
  console.error("PARITY FAILED:", e);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
});

// --- GPU-resident Muon cases (called at the end of main; declarations hoist) -----

/**
 * GPU vs CPU newtonSchulz() on random matrices covering m<n, m>n (transpose
 * path), m=n, and non-multiple-of-16 dims (GEMM edge tiles). Tolerance: BWD.
 * Five quintic iterations chain ~15 order-dependent f32 reductions — deeper
 * than any single backward kernel — but NS is contractive toward the
 * orthogonal manifold, so divergence stays small (measured max |Δ| ≈ 7e-7
 * on these cases); BWD holds with >1000x margin.
 */
async function newtonSchulzParity(gpu: WebGPUBackend) {
  const rng = mulberry32(0x5eed);
  const cases: [number, number][] = [[5, 9], [24, 17], [33, 33], [16, 64]];
  let ok = true;
  for (const [m, n] of cases) {
    const g = new Float32Array(m * n);
    for (let i = 0; i < g.length; i++) g[i] = randn(rng) * 0.8;
    const want = newtonSchulz(g, m, n, 5);
    const got = await newtonSchulzGpu(gpu, g, m, n, 5);
    ok = compare(`ns(${m}x${n})`, got, want, BWD) && ok;
  }
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} newtonSchulz GPU parity (m<n, m>n, m=n, odd dims)`);
}

/**
 * Two consecutive optimizer steps with different grads must match the CPU
 * two-step result: catches momentum buffers that are zeroed, recycled, or
 * left dirty between steps (a fresh buf in step 2 shifts the result far
 * beyond tolerance). BWD tolerance for the same reasons as newtonSchulzParity
 * (measured max |Δ| ≈ 6e-8 — the lr·ortho update is small next to the weights).
 */
async function muonMomentumPersistence(gpu: WebGPUBackend) {
  const rng = mulberry32(0xabcd);
  const shape = [24, 17]; // flip path + non-multiple-of-16 dims
  const size = 24 * 17;
  const base = new Float32Array(size);
  const g1 = new Float32Array(size);
  const g2 = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    base[i] = randn(rng) * 0.5;
    g1[i] = randn(rng) * 0.1;
    g2[i] = randn(rng) * 0.1;
  }
  const hyper = { lr: 0.02, momentum: 0.95, aux: { lr: 1e-3 } };

  const pc = new Tensor(base.slice(), shape, true);
  const cpuOpt = new Muon([pc], [], hyper);
  pc.grad.set(g1);
  cpuOpt.step();
  pc.grad.set(g2);
  cpuOpt.step();

  const pg = new Tensor(base.slice(), shape, true);
  const gpuOpt = new MuonGpu(gpu, [pg], [], hyper);
  for (const g of [g1, g2]) {
    // seedGradFromHost stands in for a backward pass: it flushes the pending
    // grad clears first, so the write lands after them in queue order.
    pg.grad.set(g);
    gpu.seedGradFromHost(pg);
    gpuOpt.recordStep();
    await gpu.sync();
  }
  await gpuOpt.syncWeightsToHost();

  const ok = compare("muon2step.w", pg.data, pc.data, BWD);
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} Muon momentum persistence across 2 optimizer steps`);
}

/**
 * GPU AdamW (adamw_gpu.ts) vs CPU AdamW over 3 steps on a 2-D param and a 1-D
 * param, with grads sized so the global grad-norm clip TRIGGERS on step 1
 * (norm > clip, scale < 1) and relaxes below clip by step 3 — exercising both
 * branches of the on-device reduction plus moment persistence and bias
 * correction. Moments live in device state buffers; grads are seeded per step
 * (seedGradFromHost stands in for a backward, overwriting the device grad).
 * BWD tolerance: the clip reduction sums in tree order vs the CPU's sequential
 * order, and 3 Adam steps compound that, but it stays well under 1e-3.
 */
async function adamwGpuParity(gpu: WebGPUBackend) {
  const rng = mulberry32(0x4d4d);
  const shapes = [[12, 8], [5]];
  const opts = { lr: 5e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8, weightDecay: 0.01, clip: 1.0 };
  // Grad magnitudes: step 0 large (norm >> clip), then shrinking past the clip.
  const gradScale = [0.5, 0.05, 0.01];

  const bases = shapes.map((s) => {
    const n = s.reduce((a, b) => a * b, 1);
    const b = new Float32Array(n);
    for (let i = 0; i < n; i++) b[i] = randn(rng) * 0.3;
    return b;
  });
  const grads = gradScale.map((gs) =>
    shapes.map((s) => {
      const n = s.reduce((a, b) => a * b, 1);
      const g = new Float32Array(n);
      for (let i = 0; i < n; i++) g[i] = randn(rng) * gs;
      return g;
    })
  );

  // CPU reference.
  const cpuParams = shapes.map((s, i) => new Tensor(bases[i].slice(), s, true));
  const cpuOpt = new AdamW(cpuParams, opts);
  for (let step = 0; step < gradScale.length; step++) {
    for (let i = 0; i < cpuParams.length; i++) cpuParams[i].grad.set(grads[step][i]);
    cpuOpt.step();
  }

  // GPU.
  const gpuParams = shapes.map((s, i) => new Tensor(bases[i].slice(), s, true));
  const gpuOpt = new AdamWGpu(gpu, gpuParams, opts);
  for (let step = 0; step < gradScale.length; step++) {
    for (let i = 0; i < gpuParams.length; i++) {
      gpuParams[i].grad.set(grads[step][i]);
      gpu.seedGradFromHost(gpuParams[i]);
    }
    gpuOpt.recordStep();
    await gpu.sync();
  }
  await gpuOpt.syncWeightsToHost();

  let ok = true;
  for (let i = 0; i < cpuParams.length; i++) {
    ok = compare(`adamwGpu.p${i}`, gpuParams[i].data, cpuParams[i].data, BWD) && ok;
  }
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} GPU AdamW vs CPU (3 steps, grad-norm clip triggers)`);
}

/**
 * The HANDOFF gate: same seeds, same batches (trainLM and trainLMGpuResident
 * make identical rng calls), 4 full optimizer steps — CPU Muon trajectory vs
 * the GPU-resident optimizer. Losses per step and every final weight tensor
 * must agree. Tolerances: each step feeds fwd/bwd f32 divergence (~BWD-sized)
 * through Newton–Schulz into the weights, compounding per step; measured over
 * 4 steps: max loss |Δ| ≈ 4e-7, max weight |Δ| ≈ 4e-7, so the BWD-scale
 * bounds hold with orders of magnitude to spare.
 */
async function muonTrajectoryParity(gpu: WebGPUBackend) {
  const cfg = microConfig();
  const steps = 4, seqLen = 8, batchPerStep = 2;
  const rngTok = mulberry32(0x70cc);
  const tokens = Array.from({ length: 160 }, () => Math.floor(rngTok() * cfg.vocabSize));
  const hyper = { lr: 0.02, momentum: 0.95, aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 } };

  const cpuModel = new Qwen3Model(cfg, mulberry32(5));
  const cg = cpuModel.paramGroups();
  const cpuHist = trainLM(cpuModel, {
    tokens,
    seqLen,
    steps,
    batchPerStep,
    optimizer: new Muon(cg.muon, cg.aux, hyper),
    logEvery: 1,
    rng: mulberry32(7),
  });

  const gpuModel = new Qwen3Model(cfg, mulberry32(5));
  const gg = gpuModel.paramGroups();
  const gpuHist = await trainLMGpuResident(gpuModel, gpu, {
    tokens,
    seqLen,
    steps,
    batchPerStep,
    optimizer: new MuonGpu(gpu, gg.muon, gg.aux, hyper),
    logEvery: 1,
    rng: mulberry32(7),
  });

  let ok = true;
  if (gpuHist.length !== cpuHist.length) {
    console.log(`    history length ${gpuHist.length} != ${cpuHist.length}`);
    ok = false;
  }
  for (let i = 0; i < Math.min(cpuHist.length, gpuHist.length); i++) {
    const dl = Math.abs(gpuHist[i].loss - cpuHist[i].loss);
    if (dl > 1e-3 + 1e-3 * Math.abs(cpuHist[i].loss)) {
      console.log(
        `    MISMATCH loss@step${cpuHist[i].step}: gpu=${gpuHist[i].loss} cpu=${cpuHist[i].loss}`,
      );
      ok = false;
    }
  }
  const cpuParams = cpuModel.params();
  const gpuParams = gpuModel.params();
  for (let i = 0; i < cpuParams.length; i++) {
    ok = compare(`muonTraj.param${i}`, gpuParams[i].data, cpuParams[i].data, BWD) && ok;
  }
  if (!ok) failures++;
  console.log(
    `  ${ok ? "ok " : "FAIL"} Muon GPU training trajectory (${steps} steps, ` +
      `${cpuParams.length} weight tensors)`,
  );
}

/**
 * Same as muonTrajectoryParity but with a WSD schedule driving a DISTINCT lr
 * every step (warmup 2 → cooldown 2, floor 0.1: multipliers 0.5, 1, 0.55, 0.1).
 * This is the gate for the dynamic-lr path: MuonGpu now reads lr from a device
 * buffer that setLrScale() rewrites each step, and the CPU Muon scales its base
 * lr in host arrays — the two must still track to BWD tolerance. A regression
 * where the GPU lr write is mis-ordered relative to the apply dispatch, or the
 * buffer isn't actually read, shows up here as trajectory divergence.
 */
async function wsdScheduleParity(gpu: WebGPUBackend) {
  const cfg = microConfig();
  const steps = 4, seqLen = 8, batchPerStep = 2;
  const rngTok = mulberry32(0x70cc);
  const tokens = Array.from({ length: 160 }, () => Math.floor(rngTok() * cfg.vocabSize));
  const hyper = { lr: 0.02, momentum: 0.95, aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 } };
  const schedule = wsdSchedule({ warmupSteps: 2, stableSteps: 0, cooldownSteps: 2, minScale: 0.1 });

  const cpuModel = new Qwen3Model(cfg, mulberry32(5));
  const cg = cpuModel.paramGroups();
  const cpuHist = trainLM(cpuModel, {
    tokens,
    seqLen,
    steps,
    batchPerStep,
    optimizer: new Muon(cg.muon, cg.aux, hyper),
    schedule,
    logEvery: 1,
    rng: mulberry32(7),
  });

  const gpuModel = new Qwen3Model(cfg, mulberry32(5));
  const gg = gpuModel.paramGroups();
  const gpuHist = await trainLMGpuResident(gpuModel, gpu, {
    tokens,
    seqLen,
    steps,
    batchPerStep,
    optimizer: new MuonGpu(gpu, gg.muon, gg.aux, hyper),
    schedule,
    logEvery: 1,
    rng: mulberry32(7),
  });

  let ok = true;
  for (let i = 0; i < Math.min(cpuHist.length, gpuHist.length); i++) {
    const dl = Math.abs(gpuHist[i].loss - cpuHist[i].loss);
    if (dl > 1e-3 + 1e-3 * Math.abs(cpuHist[i].loss)) {
      console.log(
        `    MISMATCH loss@step${cpuHist[i].step}: gpu=${gpuHist[i].loss} cpu=${cpuHist[i].loss}`,
      );
      ok = false;
    }
  }
  const cpuParams = cpuModel.params();
  const gpuParams = gpuModel.params();
  for (let i = 0; i < cpuParams.length; i++) {
    ok = compare(`wsd.param${i}`, gpuParams[i].data, cpuParams[i].data, BWD) && ok;
  }
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} WSD-scheduled Muon trajectory (GPU lr buffer vs CPU)`);
}

/**
 * MuonClip / QK-logit clip active during training: the clip is host-side weight
 * math on the aux qNorm/kNorm, so CPU trainLM and GPU trainLMGpuResident must
 * apply it identically and stay on the same trajectory. tau=0.5 triggers from
 * init (qNorm=kNorm=ones gives proxy 1.0), so every step clips on both paths;
 * a path that skipped or misordered the clip would diverge. Also asserts the
 * clip actually held every layer's proxy at/under tau on the GPU-trained model.
 */
async function qkClipTrajectoryParity(gpu: WebGPUBackend) {
  const cfg = microConfig();
  const steps = 4, seqLen = 8, batchPerStep = 2, tau = 0.5;
  const rngTok = mulberry32(0x70cc);
  const tokens = Array.from({ length: 160 }, () => Math.floor(rngTok() * cfg.vocabSize));
  const hyper = { lr: 0.02, momentum: 0.95, aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 } };

  const cpuModel = new Qwen3Model(cfg, mulberry32(5));
  const cg = cpuModel.paramGroups();
  const cpuHist = trainLM(cpuModel, {
    tokens,
    seqLen,
    steps,
    batchPerStep,
    optimizer: new Muon(cg.muon, cg.aux, hyper),
    qkClipTau: tau,
    logEvery: 1,
    rng: mulberry32(7),
  });

  const gpuModel = new Qwen3Model(cfg, mulberry32(5));
  const gg = gpuModel.paramGroups();
  const gpuHist = await trainLMGpuResident(gpuModel, gpu, {
    tokens,
    seqLen,
    steps,
    batchPerStep,
    optimizer: new MuonGpu(gpu, gg.muon, gg.aux, hyper),
    qkClipTau: tau,
    logEvery: 1,
    rng: mulberry32(7),
  });

  let ok = true;
  for (let i = 0; i < Math.min(cpuHist.length, gpuHist.length); i++) {
    const dl = Math.abs(gpuHist[i].loss - cpuHist[i].loss);
    if (dl > 1e-3 + 1e-3 * Math.abs(cpuHist[i].loss)) {
      console.log(
        `    MISMATCH loss@step${cpuHist[i].step}: gpu=${gpuHist[i].loss} cpu=${cpuHist[i].loss}`,
      );
      ok = false;
    }
  }
  const cpuParams = cpuModel.params();
  const gpuParams = gpuModel.params();
  for (let i = 0; i < cpuParams.length; i++) {
    ok = compare(`qkClip.param${i}`, gpuParams[i].data, cpuParams[i].data, BWD) && ok;
  }
  // The clip must actually have bounded the logit scale on the trained model.
  for (const L of gpuModel.layers) {
    if (qkLogitScale(L.qNorm.data, L.kNorm.data, cfg.headDim) > tau + 1e-4) ok = false;
  }
  if (!ok) failures++;
  console.log(
    `  ${ok ? "ok " : "FAIL"} MuonClip trajectory parity + logit-scale bounded (GPU vs CPU)`,
  );
}

/**
 * Verify that sync() fences GPU completion even when it reads nothing back.
 * The resident training loop calls sync() twice per step: once to read losses
 * and aux grads, once to flush the optimizer dispatches. The second sync has
 * nothing to stage, so without an explicit fence (a staging copy of a 4-byte
 * sentinel) it would resolve at submit rather than at GPU completion — making
 * the optimizer-step timing dishonest and potentially recycling transients
 * the GPU is still writing. This test encodes a GPU linear op, calls sync()
 * with no reads, then reads the output in a second sync and checks correctness.
 * If the first sync didn't actually fence, the second sync's copy would race
 * the linear dispatch and either deadlock (invalid pipeline) or read zeros.
 */
async function syncFenceGate(gpu: WebGPUBackend) {
  const rng = mulberry32(0xfeed);
  const x = randTensor([8, 16], rng);
  const w = randTensor([12, 16], rng);

  // CPU reference for correctness check (linear is already imported at top).
  for (const t of [x, w]) t.zeroGrad();
  const cpuOut = linear(x, w);
  const cpuData = cpuOut.data.slice();

  // GPU: encode the linear dispatch, fence with empty sync(), then read back.
  for (const t of [x, w]) t.zeroGrad();
  gpu.install();
  let ok = true;
  try {
    const gpuOut = linear(x, w);
    // Empty sync: should fence GPU work even though it stages nothing.
    await gpu.sync();
    // Now read back — if the fence worked the values are those of the linear op.
    await gpu.sync([gpuOut]);
    ok = compare("syncFence.out", gpuOut.data, cpuData, FWD) && ok;
  } finally {
    gpu.uninstall();
  }
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} sync() fences GPU even with no readback`);
}
