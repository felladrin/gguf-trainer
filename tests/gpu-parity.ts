// GPU-vs-CPU parity: the validation gate for the WebGPU backend.
//
// For every op, in the bring-up order from docs/notes/journal.md: run the CPU
// reference forward + backward, then the identical call routed through the
// WebGPU backend, and compare outputs and input gradients element-wise. The
// CPU side is already finite-difference-validated by tests/gradcheck.ts, so
// agreement here transfers that trust to the kernels. Additionally: a
// finite-difference check run directly against the GPU matmul (the harness
// composes on-device), a two-micro-batch gradient accumulation check, and a
// whole-model forward/backward parity check.
//
// Run:  deno run tests/gpu-parity.ts     (Node/Bun have no WebGPU: prints SKIP)

import {
  add,
  attention,
  backward,
  crossEntropy,
  embedding,
  fusedCrossEntropy,
  gelu,
  linear,
  mul,
  mulberry32,
  randn,
  rmsNorm,
  rmsNormHeads,
  rope,
  scale,
  setCheckpointing,
  silu,
  softCrossEntropy,
  Tensor,
} from "../src/model/autograd.ts";
import { gemma3Config, Gemma3Model } from "../src/arch/gemma3.ts";
import { getArch } from "../src/model/registry.ts";
import { applyLora, clearLora } from "../src/train/lora.ts";
import { freezeForScoring, sequenceLoss } from "../src/train/loss.ts";
import { greedyComplete } from "../src/eval/generate.ts";
import type { Gemma3Config } from "../src/arch/gemma3.ts";
import { Muon, newtonSchulz } from "../src/train/muon.ts";
import { trainLM } from "../src/train/trainer.ts";
import { wsdSchedule } from "../src/train/schedule.ts";
import { qkLogitScale } from "../src/train/qk-clip.ts";
import { AdamW } from "../src/train/adam.ts";
import { initWebGPU, WebGPUBackend } from "../src/backend/webgpu.ts";
import { MAX_WG } from "../src/backend/wgsl.ts";
import { MuonGpu, newtonSchulzGpu } from "../src/backend/muon-gpu.ts";
import { AdamWGpu } from "../src/backend/adamw-gpu.ts";
import { trainLMGpuResident } from "../src/backend/train-gpu.ts";

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
    gpu.startProfile(); // no argument on purpose: the default must be a legal query-set size
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
 * Fused readout + chunked cross-entropy, against the dense path it replaces
 * (`crossEntropy(linear(...))`) rather than against its own CPU twin: the whole
 * point of the op is that the two agree, and the dense side is already
 * finite-difference-validated. Widths cover a ragged final chunk, an exact
 * split and a single chunk wider than the vocab; the offset GEMM variants are
 * only exercised when there is more than one chunk. Row 1 is ignore-index.
 */
async function fusedCeParity(gpu: WebGPUBackend) {
  // The last shape is the one that exercises the offset GEMM the way a real run
  // does. Below it every span sits inside a single 64x64 block with a K loop of
  // one BK=16 step, so `blockRow`, `blockCol` and the K stride are all pinned at
  // their first value and a wrong offset cannot show. At T=130, V=200, chunk=70
  // there are three spans, NT spans 2 column blocks over 3 K-steps, NN runs 5
  // K-steps so `(gk + off)` crosses BK boundaries, and TN reaches `blockRow=64`
  // with a nonzero offset: the exact term whose parenthesization broke once.
  for (
    const [T, H, V, chunk] of [[4, 3, 9, 4], [8, 16, 40, 10], [6, 12, 32, 64], [130, 40, 200, 70]]
  ) {
    const targets = Array.from({ length: T }, (_, i) => (i === 1 ? -1 : (i * 7 + 3) % V));
    const mk = () => {
      const r = mulberry32(0xf00d);
      return { h: randTensor([T, H], r), w: randTensor([V, H], r) };
    };
    const c = mk();
    const cpuLoss = crossEntropy(linear(c.h, c.w), targets);
    backward(cpuLoss, 1);

    const g = mk();
    gpu.install();
    let ok = true;
    try {
      const loss = fusedCrossEntropy(g.h, g.w, targets, chunk);
      backward(loss, 1);
      await gpu.sync([loss]);
      const dl = Math.abs(loss.data[0] - cpuLoss.data[0]);
      if (dl > 1e-3 + 1e-3 * Math.abs(cpuLoss.data[0])) {
        console.log(`    MISMATCH fusedCE loss: gpu=${loss.data[0]} cpu=${cpuLoss.data[0]}`);
        ok = false;
      }
      ok = compare(`fusedCE.dHidden(chunk=${chunk})`, g.h.grad, c.h.grad, BWD) && ok;
      ok = compare(`fusedCE.dReadout(chunk=${chunk})`, g.w.grad, c.w.grad, BWD) && ok;
    } finally {
      gpu.uninstall();
    }
    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok " : "FAIL"} fused chunked CE vs dense (T=${T} V=${V} chunk=${chunk})`,
    );
  }
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

// A tiny mixed SWA/global config for the optimizer/schedule/clip parity tests.
// swaPattern 2 -> layer 0 sliding-window (local RoPE), layer 1 global (global
// RoPE), so both paths are exercised; slidingWindow == maxSeq so it never
// restricts here (gemma3ModelParity covers a genuinely-restricting window).
function microConfig(): Gemma3Config {
  return {
    arch: "gemma3",
    vocabSize: 50,
    hiddenSize: 32,
    nLayers: 2,
    nHeads: 4,
    nKVHeads: 2,
    headDim: 8,
    ffnDim: 64,
    ropeBase: 1_000_000,
    ropeBaseLocal: 10_000,
    rmsEps: 1e-6,
    maxSeq: 32,
    tieEmbeddings: true,
    slidingWindow: 32,
    swaPattern: 2,
  };
}

/**
 * Gemma3 whole-model parity: exercises the arch's distinctive path (sqrt(hidden)
 * embed scale, sandwich norms, GeGLU, per-layer SWA + local/global RoPE). The
 * config mixes SWA layers (0,1,3) with a global layer (2) and T > slidingWindow
 * so the window genuinely restricts, matching the CPU windowed reference.
 */
async function gemma3ModelParity(gpu: WebGPUBackend) {
  const cfg: Gemma3Config = {
    arch: "gemma3",
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
  const model = new Gemma3Model(cfg, mulberry32(5));
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
 * reclaimTransients must not change the math. Freeing each micro-batch's
 * transient buffers mid-step (to cut peak VRAM so batch>=2 fits at long ctx)
 * only recycles pool buffers earlier, so an identical run with it off must
 * produce the same loss trajectory and the same final weights. batchPerStep 3
 * gives two reclaim boundaries per step. A grad-accumulation regression (freeing
 * a live param grad) or a dropped loss buffer would diverge here.
 */
async function reclaimTransientsParity(gpu: WebGPUBackend) {
  const cfg = microConfig();
  const steps = 4, seqLen = 8, batchPerStep = 3;
  const rngTok = mulberry32(0x5ec1);
  const tokens = Array.from({ length: 160 }, () => Math.floor(rngTok() * cfg.vocabSize));
  const hyper = { lr: 0.02, momentum: 0.95, aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 } };

  const run = async (reclaimTransients: boolean, lossChunk = 0, recompute = false) => {
    setCheckpointing(recompute);
    const model = new Gemma3Model(cfg, mulberry32(5));
    const g = model.paramGroups();
    const hist = await trainLMGpuResident(model, gpu, {
      tokens,
      seqLen,
      steps,
      batchPerStep,
      optimizer: new MuonGpu(gpu, g.muon, g.aux, hyper),
      logEvery: 1,
      rng: mulberry32(7),
      reclaimTransients,
      lossChunk,
    });
    setCheckpointing(false);
    return { hist, params: model.params() };
  };

  const off = await run(false);
  const on = await run(true);

  let ok = true;
  for (let i = 0; i < off.hist.length; i++) {
    const dl = Math.abs(on.hist[i].loss - off.hist[i].loss);
    if (dl > 1e-4 + 1e-4 * Math.abs(off.hist[i].loss)) {
      console.log(
        `    MISMATCH loss@step${off.hist[i].step}: on=${on.hist[i].loss} off=${off.hist[i].loss}`,
      );
      ok = false;
    }
  }
  for (let i = 0; i < off.params.length; i++) {
    ok = compare(`reclaim.param${i}`, on.params[i].data, off.params[i].data, BWD) && ok;
  }
  if (!ok) failures++;
  console.log(
    `  ${ok ? "ok " : "FAIL"} reclaimTransients matches off ` +
      `(${steps} steps x ${batchPerStep} micro-batches)`,
  );
  // The fused loss holds its softmax statistics and chunk scratch as transients
  // from forward until backward reads them, and `--reclaim` returns transients
  // to the pool at every micro-batch boundary. That contract is a loop-level
  // property the op-level parity case above cannot reach, so drive it through
  // the same three-micro-batch loop: vocab 50 at chunk 16 gives four spans with
  // nonzero offsets and a ragged tail, and Gemma3's tied readout makes the
  // offset TN gemm accumulate into the same `tokenEmbd.grad` that `embedding`'s
  // backward writes.
  const chunkOff = await run(false, 16);
  const chunkOn = await run(true, 16);
  let cok = true;
  for (let i = 0; i < off.hist.length; i++) {
    for (const [tag, arm] of [["reclaim-off", chunkOff], ["reclaim-on", chunkOn]] as const) {
      const dl = Math.abs(arm.hist[i].loss - off.hist[i].loss);
      if (dl > 1e-3 + 1e-3 * Math.abs(off.hist[i].loss)) {
        console.log(
          `    MISMATCH chunked ${tag} loss@step${off.hist[i].step}: ` +
            `${arm.hist[i].loss} vs dense ${off.hist[i].loss}`,
        );
        cok = false;
      }
    }
  }
  // The pair that actually isolates reclaim: same kernels, same span order, same
  // dispatch order, only pool-recycling timing differs, so nothing reorders the
  // f32 reduction and this must hold as tightly as the dense pair above. The
  // comparisons against the dense baseline are the looser, separate claim that
  // chunking preserves the math at loop scale.
  for (let i = 0; i < chunkOff.hist.length; i++) {
    const dl = Math.abs(chunkOn.hist[i].loss - chunkOff.hist[i].loss);
    if (dl > 1e-4 + 1e-4 * Math.abs(chunkOff.hist[i].loss)) {
      console.log(
        `    MISMATCH chunked reclaim on/off loss@step${chunkOff.hist[i].step}: ` +
          `on=${chunkOn.hist[i].loss} off=${chunkOff.hist[i].loss}`,
      );
      cok = false;
    }
  }
  for (let i = 0; i < off.params.length; i++) {
    cok =
      compare(`chunkedReclaim.param${i}`, chunkOn.params[i].data, chunkOff.params[i].data, BWD) &&
      cok;
    cok = compare(`chunkedVsDense.param${i}`, chunkOff.params[i].data, off.params[i].data, BWD) &&
      cok;
  }
  if (!cok) failures++;
  console.log(
    `  ${cok ? "ok " : "FAIL"} chunked loss across reclaim boundaries ` +
      `(${steps} steps x ${batchPerStep} micro-batches, chunk 16 over vocab ${cfg.vocabSize})`,
  );

  // Recompute across the same boundaries. The loss must be BIT-identical, not
  // merely close: recompute is a deterministic replay of the same ops on the
  // same inputs, so unlike chunking it reorders nothing, and any drift at all
  // means a buffer moved under the replay. The memory side of the region
  // free-list is a separate claim and cannot be seen from here, since a
  // stranded buffer still produces correct numbers: recomputeMemoryGate.
  const rcOff = await run(false, 0, true);
  const rcOn = await run(true, 0, true);
  let rok = true;
  for (let i = 0; i < off.hist.length; i++) {
    for (const [tag, arm] of [["reclaim-off", rcOff], ["reclaim-on", rcOn]] as const) {
      if (arm.hist[i].loss !== off.hist[i].loss) {
        console.log(
          `    MISMATCH recompute ${tag} loss@step${off.hist[i].step}: ` +
            `${arm.hist[i].loss} vs dense ${off.hist[i].loss}`,
        );
        rok = false;
      }
    }
  }
  for (let i = 0; i < off.params.length; i++) {
    rok = compare(`recomputeReclaim.param${i}`, rcOn.params[i].data, off.params[i].data, BWD) &&
      rok;
  }
  if (!rok) failures++;
  console.log(
    `  ${rok ? "ok " : "FAIL"} recompute across reclaim boundaries ` +
      `(${steps} steps x ${batchPerStep} micro-batches, bit-identical loss)`,
  );
}

/**
 * The flash kernels on a device that grants only the WebGPU spec defaults, which
 * is what a browser or a plain `requestDevice()` hands out. `initWebGPU` asks for
 * the adapter's maxima, so every other check in this file runs on a device that
 * may be far more generous, and the limit that bites at long context is
 * invisible there: at T=3584 the pre-flash kernels needed a ~205 MB [Hq,T,T]
 * probs binding, over the default 128 MiB maxStorageBufferBindingSize, and
 * failed bind-group validation. Deno's wgpu DOES validate that one, so this is
 * a real gate for it. It is not a gate for workgroup storage, which wgpu does
 * not validate: tests/kernel-limits.ts covers that from the source instead.
 *
 * CPU comparison is far too slow at these shapes, so assert completion with no
 * device/validation error and finite, non-zero outputs and input gradients.
 */
async function specDefaultLimitsCheck(T: number, hd: number, why: string) {
  // deno-lint-ignore no-explicit-any
  const nav: any = (globalThis as any).navigator;
  const adapter = await nav?.gpu?.requestAdapter?.();
  if (!adapter) {
    console.log(`  SKIP spec-default-limits check (${why}): no WebGPU adapter`);
    return;
  }
  const device = await adapter.requestDevice(); // no requiredLimits: spec defaults
  const gpu2 = new WebGPUBackend(device, "spec-default-limits");
  gpu2.attnFlashMinT = 1; // force the flash path even at the small shape
  const Hq = 4, Hkv = 2;
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
    `  ${ok ? "ok " : "FAIL"} flash attention fwd+bwd @ T=${T}, hd=${hd} ` +
      `under spec-default limits (${why})`,
  );
}

async function main() {
  const gpu = await initWebGPU();
  if (!gpu) {
    console.log("SKIP: no WebGPU in this runtime. Run under Deno (or provide navigator.gpu).");
    return;
  }
  console.log(`=== GPU-vs-CPU parity checks (adapter: ${gpu.adapterName}) ===\n`);
  const rng = mulberry32(1234);

  // 1. matmul / linear: including a multi-tile case with non-multiple-of-16 dims.
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
    // Straddles the register-tiled block dims (BM=BN=64, BK=16) in all three
    // axes with remainders: 2 row-blocks + tail, 2 col-blocks + tail, and a K
    // that is 5 full BK steps plus a 3-deep tail, so the ragged K arrives after
    // several steps have already accumulated into the register tile.
    const x = randTensor([70, 83], rng);
    const w = randTensor([75, 83], rng);
    await opCase(gpu, "linear (70x83 · 75x83ᵀ, multi-block, ragged K)", [x, w], () => linear(x, w));
  }
  await gpuMatmulFdCheck(gpu);
  await profilerSmoke(gpu);
  await aliasedBinaryOpParity(gpu);
  await targetRangeGate(gpu);
  await fusedCeParity(gpu);
  await recomputeModelParity(gpu);
  await loraModelParity(gpu);
  await recomputeMemoryGate();
  await evalFreezeGate();
  await generateFreezeGate();
  await frozenClearGate();
  await clearRearmPredicateGate();
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
    // A target ~90 logits behind the maximum. This is the shape lever 23 says the
    // suite was blind to: the CPU used to clamp such a row at -log(1e-12) = 27.63
    // while the GPU computed it exactly, and every existing case produces losses
    // of 2 to 10, nowhere near the clamp. Both sides now agree at the true value.
    // This pins MAGNITUDE, not precision: `compare`'s budget at a loss of 76.5 is
    // 0.153, while the clamp it guards sits 48.87 away. Its gradient half is
    // degenerate too: the target probabilities are 4.1e-41 and 8.8e-27, so dInput
    // is +-0.5 and zeros on both sides. Do not count this as gradient coverage.
    const V = 4;
    const logits = new Tensor(Float32Array.from([90, 0, 0, -3, 60, 1, 0, 0]), [2, V], true);
    const targets = [3, 2];
    await opCase(
      gpu,
      "crossEntropy (target far behind)",
      [logits],
      () => crossEntropy(logits, targets),
    );
  }
  {
    // A KEPT row that pads. Every other zero-weight teacher entry in this suite
    // sits in an IGNORED row, where both implementations return before the loop,
    // so the zero-weight skip on either side was never executed by anything.
    // Finite rather than -Infinity on purpose: this kernel seeds its running
    // maximum with -3.0e38 rather than -inf precisely to avoid depending on
    // infinity semantics on the device. The limit that follows is worth stating:
    // for a finite pad the skip is behaviour-preserving, so removing it does NOT
    // fail this case. What this pins is that the two implementations agree on a
    // padded row at all; the skip's actual purpose is pinned on the CPU side, in
    // `softCE skips a zero pad` in tests/gradcheck.ts.
    const V = 5, K = 3;
    const logits = randTensor([2, V], mulberry32(77));
    const ids = [3, 0, 2, 1, 4, 0];
    const probs = [0.7, 0.3, 0.0, 0.5, 0.5, 0.0];
    await opCase(
      gpu,
      "softCrossEntropy (zero-weight pad)",
      [logits],
      () => softCrossEntropy(logits, ids, probs, K),
    );
  }
  {
    // Ignore-index (-1) = assistant-only loss masking: masked rows contribute
    // no loss and no gradient; the mean is over kept rows. GPU must match CPU.
    const T = 8, V = 17;
    const logits = randTensor([T, V], rng);
    const targets = [-1, 3, -1, 11, 0, -1, 16, 5];
    await opCase(gpu, "crossEntropy (ignore-index)", [logits], () => crossEntropy(logits, targets));
  }
  {
    // V wider than the cross-entropy workgroup (256): every lane strides its row
    // several times, which is the loop the one-thread-per-row kernel never had.
    // The V=17 cases above enter it exactly once, so they cannot see a stride bug.
    const T = 3, V = 1000;
    const logits = randTensor([T, V], rng);
    const targets = [617, 0, 999];
    await opCase(
      gpu,
      "crossEntropy (V > workgroup)",
      [logits],
      () => crossEntropy(logits, targets),
    );
  }
  {
    // Soft-target CE (Phase B KL anchor): normalized row, truncated top-k row
    // (mass < 1), ignored row, and a row with a duplicate teacher id: the case
    // that would race if the sparse backward ran one thread per teacher entry.
    const T = 4, V = 17, K = 3;
    const logits = randTensor([T, V], rng);
    const ids = [3, 9, 16, 0, 5, 11, -1, 0, 0, 7, 7, 2];
    const q = [0.5, 0.3, 0.2, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.25, 0.25, 0.5];
    await opCase(gpu, "softCrossEntropy", [logits], () => softCrossEntropy(logits, ids, q, K));
  }
  {
    // The same wide-row case for the soft-target path, which has its own strided
    // loops and its own mass·log(Σ) loss form.
    const T = 3, V = 1000, K = 2;
    const logits = randTensor([T, V], rng);
    const ids = [617, 0, 999, 12, -1, 0];
    const q = [0.6, 0.4, 0.3, 0.2, 0.0, 0.0];
    await opCase(
      gpu,
      "softCrossEntropy (V > workgroup)",
      [logits],
      () => softCrossEntropy(logits, ids, q, K),
    );
  }

  // 5. graph-level
  await gemma3ModelParity(gpu);
  await accumulationParity(gpu);
  await reclaimTransientsParity(gpu);
  await checkpointCadence(gpu);

  // 6. Attention kernels (appended by the tiling task):
  //    (a) Materialized path at small T (T < attnFlashMinT): non-multiples of
  //        tile size, both head dims, both GQA group sizes.
  //    (b) Flash path forced at the same T by temporarily lowering attnFlashMinT
  //        to 1: proves the flash kernels correct at small T without changing
  //        the production threshold.
  //    (c) Backward-heavy case that exercises srcAttnBwdDkv's GQA head-group loop.
  //    (d) Large-T functional check under spec-default device limits.
  // Head dims cover all three lanes of the flash kernels' vec4 codegen: hd=6/12
  // are not multiples of 4 or leave an odd vec4 count (scalar fallback and the
  // single-chain dQ/dV loop), hd=64 is the shape every published checkpoint uses.
  const flashCases: [number, number, number, number, string][] = [
    [67, 4, 2, 6, "T=67, hd=6, group=2"],
    [67, 2, 2, 32, "T=67, hd=32, group=1"],
    [130, 4, 2, 32, "T=130, hd=32, group=2"],
    [130, 3, 3, 6, "T=130, hd=6, group=1"],
    [193, 4, 1, 24, "T=193, hd=24, group=4"],
    [130, 4, 2, 12, "T=130, hd=12, group=2"],
    [130, 4, 2, 64, "T=130, hd=64, group=2"],
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
    [193, 4, 2, 64, 48, "T=193, hd=64, group=2, W=48"],
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
  // hd=64 is the head size every published checkpoint trains at and had no
  // coverage at all. Note this pair checks that the kernels RUN at these shapes,
  // not that they respect the limits: Deno's wgpu does not enforce
  // maxComputeWorkgroupStorageSize at pipeline creation (probed: it accepted a
  // 16640-byte shader on a device granting 16384). tests/kernel-limits.ts is
  // what actually gates the footprint, by reading the emitted WGSL.
  await specDefaultLimitsCheck(3584, 32, "long-context storage binding");
  await specDefaultLimitsCheck(130, 64, "the published head size");

  // 7. GPU-resident Muon optimizer (src/backend/muon-gpu.ts): Newton–Schulz
  //    kernel parity, momentum-buffer persistence across steps, and the
  //    whole-trajectory parity against the CPU Muon.
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
 * Five quintic iterations chain ~15 order-dependent f32 reductions: deeper
 * than any single backward kernel, but NS is contractive toward the
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
 * (measured max |Δ| ≈ 6e-8: the lr·ortho update is small next to the weights).
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
 * GPU AdamW (adamw-gpu.ts) vs CPU AdamW over 3 steps on a 2-D param and a 1-D
 * param, with grads sized so the global grad-norm clip TRIGGERS on step 1
 * (norm > clip, scale < 1) and relaxes below clip by step 3: exercising both
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
 * The trajectory gate: same seeds, same batches (trainLM and trainLMGpuResident
 * make identical rng calls), 4 full optimizer steps: CPU Muon trajectory vs
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

  const cpuModel = new Gemma3Model(cfg, mulberry32(5));
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

  const gpuModel = new Gemma3Model(cfg, mulberry32(5));
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
 * lr in host arrays: the two must still track to BWD tolerance. A regression
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

  const cpuModel = new Gemma3Model(cfg, mulberry32(5));
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

  const gpuModel = new Gemma3Model(cfg, mulberry32(5));
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

  const cpuModel = new Gemma3Model(cfg, mulberry32(5));
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

  const gpuModel = new Gemma3Model(cfg, mulberry32(5));
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
 * sentinel) it would resolve at submit rather than at GPU completion: making
 * the optimizer-step timing dishonest and potentially recycling transients
 * the GPU is still writing. This test encodes a GPU linear op, calls sync()
 * with no reads, then reads the output in a second sync and checks correctness.
 * If the first sync didn't actually fence, the second sync's copy would race
 * the linear dispatch and either deadlock (invalid pipeline) or read zeros.
 */
/**
 * Activation recomputation on the device, over a whole model, against the CPU
 * reference WITHOUT it. Two things this reaches that the CPU gradcheck cannot:
 * the region free-list actually handing a released buffer to a later makeOut,
 * and the device-side gradient seed that carries the accumulated gradient into
 * the replayed subgraph. Run for all three architectures, since each wraps its
 * own layer body.
 */
async function recomputeModelParity(gpu: WebGPUBackend) {
  for (const name of ["gemma3", "llama", "qwen3"]) {
    const arch = getArch(name)!;
    // deno-lint-ignore no-explicit-any
    const cfg = arch.tinyConfig(23) as any;
    const ids = [3, 9, 1, 14, 7, 2], targets = [9, 1, 14, 7, 2, 5];

    const cpu = arch.build(cfg, mulberry32(5));
    const cpuLoss = crossEntropy(cpu.forward(ids), targets);
    backward(cpuLoss, 1);
    const cpuGrads = cpu.params().map((p) => p.grad.slice());

    const model = arch.build(cfg, mulberry32(5));
    gpu.install();
    setCheckpointing(true);
    let ok = true;
    try {
      const loss = crossEntropy(model.forward(ids), targets);
      backward(loss, 1);
      await gpu.sync([loss]);
      const dl = Math.abs(loss.data[0] - cpuLoss.data[0]);
      if (dl > 1e-3 + 1e-3 * Math.abs(cpuLoss.data[0])) {
        console.log(`    MISMATCH ${name} recompute loss: ${loss.data[0]} vs ${cpuLoss.data[0]}`);
        ok = false;
      }
      const ps = model.params();
      for (let i = 0; i < ps.length; i++) {
        ok = compare(`recompute.${name}.dParam${i}`, ps[i].grad, cpuGrads[i], BWD) && ok;
      }
    } finally {
      setCheckpointing(false);
      gpu.uninstall();
    }
    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok " : "FAIL"} ${name} recompute vs dense reference ` +
        `(${model.params().length} param tensors)`,
    );
  }
}

/**
 * The memory claims, which no correctness test can make: a region buffer that
 * never returns to the pool leaves every number correct and quietly allocates
 * around it. That is not hypothetical, it was the first version of this change.
 *
 * Two of the three assertions are constant-free, because each compares the same
 * quantity under two configs rather than against a fitted threshold:
 *
 *   sync() drain    - with reclaim OFF it is the only drain, so if it is gone
 *                     nothing ever returns and the pool grows with step count.
 *                     Healthy, it plateaus after the first step.
 *   reclaim drain   - with reclaim ON, region buffers must not accumulate per
 *                     micro-batch, which is reclaim's own documented contract.
 *                     If that drain is gone the pool grows with batchPerStep.
 *
 * The third is the headline claim (recompute shrinks the pool at all) and is
 * kept deliberately loose, since the exact ratio is a property of this tiny
 * shape: most of its pool is parameters, which recompute does not touch.
 *
 * `residentBytes().pool` only grows, so it is a high-water mark. Staging buffers
 * are created outside the pool and optimizer state is counted separately, so
 * nothing else moves these numbers.
 */
async function recomputeMemoryGate() {
  const cfg = gemma3Config(64, 64, 4, 256, 16);
  const tokens = Array.from({ length: 8192 }, (_, i) => (i * 7 + 3) % cfg.vocabSize);
  const poolFor = async (recompute: boolean, reclaim: boolean, steps: number, batch: number) => {
    const gpu = (await initWebGPU())!;
    setCheckpointing(recompute);
    const m = new Gemma3Model(cfg, mulberry32(5));
    const g = m.paramGroups();
    try {
      await trainLMGpuResident(m, gpu, {
        tokens,
        seqLen: 128,
        steps,
        batchPerStep: batch,
        optimizer: new MuonGpu(gpu, g.muon, g.aux, {
          lr: 0.01,
          momentum: 0.95,
          aux: { lr: 3e-3, weightDecay: 0, clip: 1 },
        }),
        logEvery: 100,
        rng: mulberry32(7),
        reclaimTransients: reclaim,
      });
      return gpu.residentBytes().pool;
    } finally {
      setCheckpointing(false);
      // destroy() runs setOpsBackend(null), so this leaves no backend installed.
      // Every caller here installs its own, but say so rather than rely on it.
      gpu.destroy();
    }
  };
  const mb = (n: number) => (n / 1e6).toFixed(1);

  // 1. sync() drain: pool must not grow with step count when reclaim is off.
  const s2 = await poolFor(true, false, 2, 2);
  const s6 = await poolFor(true, false, 6, 2);
  // 1% for bucket jitter. Healthy this is 0%; with that drain deleted the pool
  // grows 44% (10.9 -> 10.9 MB against 14.4 -> 20.8 MB), so the slack is ample.
  const steadySteps = s6 <= s2 * 1.01;

  // 2. reclaim drain: pool must not grow with micro-batch count when on.
  const b2 = await poolFor(true, true, 3, 2);
  const b4 = await poolFor(true, true, 3, 4);
  // 5% for the same reason. Healthy this is 0.013% (one 1 KB bucket); with that
  // drain deleted the pool grows 22% (10.0 -> 12.2 MB).
  const steadyBatch = b4 <= b2 * 1.05;

  // 3. the headline claim, loose on purpose.
  const dense = await poolFor(false, true, 3, 2);
  const shrinks = b2 < 0.9 * dense;

  const ok = steadySteps && steadyBatch && shrinks;
  if (!ok) failures++;
  console.log(
    `  ${ok ? "ok " : "FAIL"} recompute memory: flat in steps ` +
      `(${mb(s2)}->${mb(s6)} MB, reclaim off), flat in micro-batches ` +
      `(${mb(b2)}->${mb(b4)} MB, reclaim on), under dense (${mb(dense)} MB)`,
  );
}

/**
 * The eval readback gate. Scoring never runs backward, but every parameter is an
 * external with `requiresGrad`, so `entryFor` gave each one a full-size gradient
 * accumulator and `sync()` staged all of them back to the host on every window.
 * `freezeForScoring` is what stops both, and neither is visible in the score, so
 * only a measurement can tell whether it is still working.
 */
async function evalFreezeGate() {
  const cfg = gemma3Config(64, 64, 4, 256, 16);
  const ids = Array.from({ length: 65 }, (_, i) => (i * 11 + 5) % cfg.vocabSize);
  const arm = async (freeze: boolean, lossChunk: number) => {
    const gpu = (await initWebGPU())!;
    const m = new Gemma3Model(cfg, mulberry32(5));
    try {
      // Before uploadParams on purpose: entryFor sizes the buffer on first use.
      if (freeze) freezeForScoring(m);
      gpu.install();
      gpu.uploadParams(m.params());
      const loss = sequenceLoss(m, ids.slice(0, -1), ids.slice(1), lossChunk);
      await gpu.sync([loss]);
      return {
        readback: gpu.lastSyncReadbackBytes,
        pool: gpu.residentBytes().pool,
        loss: loss.data[0],
        paramBytes: m.params().reduce((a, t) => a + t.size * 4, 0),
      };
    } finally {
      gpu.destroy();
    }
  };
  const mb = (n: number) => (n / 1e6).toFixed(2);
  // Both readout paths. fusedCrossEntropy calls entryFor on the hidden state and
  // the readout weight itself rather than going through linear, so its touched
  // externals, and therefore what sync() stages back, are its own. At vocab 64
  // this is one span: multi-span numerics and the dW gemm are fusedCeParity's
  // job, and neither runs here, since the gate never calls backward.
  for (const lossChunk of [0, 64]) {
    const hot = await arm(false, lossChunk);
    const cold = await arm(true, lossChunk);
    const paramBytes = hot.paramBytes;

    // 1. The unfrozen arm reads back a whole model of gradients, plus the
    //    scalar. Pinning the baseline is what keeps assertion 2 from passing on
    //    a model so small the copies never mattered.
    const wasCopying = hot.readback >= paramBytes;
    // 2. Frozen, the only thing crossing the bus is the loss scalar itself.
    const stopped = cold.readback === 4;
    // 3. And the accumulators are not allocated either, so the pool drops by
    //    roughly the model. Both arms allocate the same data buffers and the
    //    same intermediates, so the gap is the parameter gradients less one
    //    256-byte stub. The 0.9 leaves a tenth of the model as margin, which is
    //    slack for bucket rounding, not for a partial regression.
    const smaller = cold.pool < hot.pool - 0.9 * paramBytes;
    // 4. The score is the point: freezing must not move it at all.
    const same = hot.loss === cold.loss;

    const ok = wasCopying && stopped && smaller && same;
    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok " : "FAIL"} eval freeze (chunk ${lossChunk}): readback ` +
        `${mb(hot.readback)} MB -> ${cold.readback} B (params ${mb(paramBytes)} MB), ` +
        `pool ${mb(hot.pool)} -> ${mb(cold.pool)} MB, loss ${hot.loss.toFixed(6)} both arms`,
    );
  }
}

/**
 * The same readback gate for `generate`, which pays it per TOKEN rather than per
 * window: `greedyComplete` syncs once per decoded token, so an unfrozen
 * parameter is a whole model of gradients crossing the bus on every one. Worth
 * its own arm rather than an extra case in evalFreezeGate, because the workload
 * is a decode loop with no loss in it and the thing that must not move is the
 * generated text.
 */
async function generateFreezeGate() {
  const cfg = gemma3Config(64, 64, 4, 256, 16);
  const prompt = [3, 11, 29, 5];
  const maxNew = 3;
  const arm = async (freeze: boolean) => {
    const gpu = (await initWebGPU())!;
    const m = new Gemma3Model(cfg, mulberry32(5));
    try {
      // Before uploadParams on purpose: entryFor sizes the buffer on first use,
      // so freezing after it stops the copies but keeps the allocation.
      if (freeze) freezeForScoring(m);
      gpu.install();
      gpu.uploadParams(m.params());
      const ids = await greedyComplete(m, gpu, prompt, maxNew);
      return {
        readback: gpu.lastSyncReadbackBytes,
        pool: gpu.residentBytes().pool,
        ids: ids.join(","),
        paramBytes: m.params().reduce((a, t) => a + t.size * 4, 0),
      };
    } finally {
      gpu.destroy();
    }
  };
  const hot = await arm(false);
  const cold = await arm(true);
  const paramBytes = hot.paramBytes;
  // The last sync reads back the last step's logits, [ctx, vocab] f32: the
  // context has grown by one token per step already taken, and greedyComplete
  // trims it to maxSeq. No stop token is passed, so the loop always runs to
  // maxNew and the count is exact rather than a bound.
  const lastLogits = Math.min(prompt.length + maxNew - 1, cfg.maxSeq) * cfg.vocabSize * 4;

  const wasCopying = hot.readback >= paramBytes;
  const stopped = cold.readback === lastLogits;
  const smaller = cold.pool < hot.pool - 0.9 * paramBytes;
  // A canary, not a guard. No read of requiresGrad feeds an output value: each
  // is entryFor's buffer choice, sync()'s staging decision, or a gate on a dW
  // accumulation (read in the closure on the GPU path, captured as wantsDW at
  // forward time on the CPU one). So no regression in the freeze can move the
  // text and this cannot be mutation-proved. It is here for a future forward
  // read of the flag.
  const same = hot.ids === cold.ids;

  const ok = wasCopying && stopped && smaller && same;
  if (!ok) failures++;
  const mb = (n: number) => (n / 1e6).toFixed(2);
  console.log(
    `  ${ok ? "ok " : "FAIL"} generate freeze: readback ${mb(hot.readback)} MB -> ` +
      `${cold.readback} B (logits alone ${lastLogits} B, params ${mb(paramBytes)} MB), ` +
      `pool ${mb(hot.pool)} -> ${mb(cold.pool)} MB, ids ${cold.ids}`,
  );
}

/**
 * The clear queue. `sync()` re-arms `gradNeedsClear` for every touched external,
 * and `entryFor` then queues that buffer for a `clearBuffer` on the next window.
 * A frozen external shares one 256-byte stub that nothing ever writes, so every
 * one of those clears was a no-op: a few hundred per window on a 293M model
 * under eval, and per step under LoRA, where the base weights are frozen for the
 * whole run.
 *
 * Waste has no symptom in a number, so the only way to see it is to count. The
 * second window is what matters: on the first, a frozen parameter is not queued
 * at all, because `entryFor` starts `gradNeedsClear` at `requiresGrad`. The
 * count does not go to zero, and should not: `makeOut` queues every
 * intermediate's gradient buffer unconditionally, which is a separate waste in a
 * forward-only run (#67).
 */
async function frozenClearGate() {
  const cfg = gemma3Config(64, 64, 4, 256, 16);
  const ids = Array.from({ length: 33 }, (_, i) => (i * 11 + 5) % cfg.vocabSize);
  const arm = async (freeze: boolean) => {
    const gpu = (await initWebGPU())!;
    const m = new Gemma3Model(cfg, mulberry32(5));
    try {
      if (freeze) freezeForScoring(m);
      gpu.install();
      gpu.uploadParams(m.params());
      // params().length stands in for "externals this forward touches", which
      // holds because this forward touches every parameter. A parameter a future
      // forward skipped would move the count for a reason unrelated to freezing.
      const window = async () => {
        const loss = sequenceLoss(m, ids.slice(0, -1), ids.slice(1), 0);
        await gpu.sync([loss]);
        return loss.data[0];
      };
      await window();
      const before = gpu.gradClearsIssued;
      const loss = await window();
      return { second: gpu.gradClearsIssued - before, params: m.params().length, loss };
    } finally {
      gpu.destroy();
    }
  };
  const hot = await arm(false);
  const cold = await arm(true);

  // Exactly one clear per parameter goes away, and nothing else does. The rest
  // of the count is the per-window intermediates, which makeOut queues
  // unconditionally at creation; those are a separate waste in a forward-only
  // run and are #67, not this.
  const savedTheParams = hot.second - cold.second === hot.params;
  // Weak on its own, since the intermediates alone satisfy it: what it catches
  // is makeOut's queue disappearing, which would make the assertion above pass
  // for the wrong reason. That a TRAINABLE accumulator still gets zeroed is
  // proved elsewhere, by the 16 parity checks that fail if the re-arm is dropped
  // rather than narrowed.
  const stillClears = hot.second > hot.params;
  // And the loss does not move, which is the claim that the removed clears were
  // doing nothing in the first place.
  const same = hot.loss === cold.loss;

  const ok = savedTheParams && stillClears && same;
  if (!ok) failures++;
  console.log(
    `  ${ok ? "ok " : "FAIL"} frozen clears: second window issues ${hot.second} unfrozen ` +
      `and ${cold.second} frozen, a saving of exactly the ${hot.params} parameters, ` +
      `loss ${hot.loss.toFixed(6)} both`,
  );
}

/**
 * Which predicate re-arms the clear queue, pinned by the one ordering the two
 * candidates disagree on.
 *
 * `e.grad !== frozenStub` and `t.requiresGrad` agree wherever the freeze came
 * before the first `entryFor`, which is every caller today, so the counting gate
 * above cannot tell them apart. They differ when a parameter is frozen
 * mid-window, after that window's `entryFor` and before its `sync()`: keying on
 * the flag records "no clear needed" while a full-size accumulator still holds
 * that window's gradients, and the next backward after a thaw accumulates on top
 * of them. Measured at exactly 2x when this was written.
 *
 * Nothing in the tree freezes mid-window. This is here so that simplifying the
 * predicate to the flag, which reads like the same thing and passes every other
 * check, fails something.
 *
 * It reads host `p.grad`, which `sync()` refreshes only while the tensor
 * requires grad and is not in `gradKeptOnDevice`. Construct an optimizer here,
 * or call `keepGradOnDevice`, and the staging stops: `third` would be a copy of
 * the step-1 host array and the gate would pass against any device state.
 */
async function clearRearmPredicateGate() {
  const cfg = gemma3Config(64, 64, 4, 256, 16);
  const ids = Array.from({ length: 33 }, (_, i) => (i * 11 + 5) % cfg.vocabSize);
  const gpu = (await initWebGPU())!;
  const m = new Gemma3Model(cfg, mulberry32(5));
  try {
    gpu.install();
    gpu.uploadParams(m.params());
    const step = async () => {
      const loss = sequenceLoss(m, ids.slice(0, -1), ids.slice(1), 0);
      backward(loss, 1);
      await gpu.sync([loss]);
      return m.params().map((p) => Float32Array.from(p.grad));
    };
    const first = await step();

    // The freeze lands between this window's backward and its sync, which is the
    // only moment the two predicates disagree about.
    const loss = sequenceLoss(m, ids.slice(0, -1), ids.slice(1), 0);
    backward(loss, 1);
    freezeForScoring(m);
    await gpu.sync([loss]);

    for (const p of m.params()) p.requiresGrad = true;
    const third = await step();

    // Same inputs, same weights (nothing stepped an optimizer), so an accumulator
    // that was properly zeroed gives the same gradients as the first step. A
    // stale one gives twice them.
    let worst = 0, n = 0;
    for (let i = 0; i < first.length; i++) {
      for (let j = 0; j < first[i].length; j++) {
        if (Math.abs(first[i][j]) < 1e-6) continue;
        worst = Math.max(worst, Math.abs(third[i][j] / first[i][j] - 1));
        n++;
      }
    }
    // n > 0 or this passes on an all-zero gradient: every element below the
    // threshold is skipped, so an empty comparison leaves `worst` at 0.
    const ok = n > 0 && worst < 1e-4;
    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok " : "FAIL"} clear re-arm survives a mid-window freeze: ` +
        `worst |g3/g1 - 1| = ${worst.toExponential(2)} over ${n} elems`,
    );
  } finally {
    gpu.destroy();
  }
}

/**
 * LoRA end to end on the device, over a whole model, against the CPU reference
 * running the same adapters. Reaches three things the CPU gradcheck cannot: the
 * adapted `linear` composing correctly through every architecture's projections,
 * the frozen-base path where the backend skips the dW gemm and binds a shared
 * stub for the gradient, and merge/unmerge round-tripping f32 weights.
 */
async function loraModelParity(gpu: WebGPUBackend) {
  for (const name of ["gemma3", "llama", "qwen3"]) {
    const arch = getArch(name)!;
    // deno-lint-ignore no-explicit-any
    const cfg = arch.tinyConfig(23) as any;
    const ids = [3, 9, 1, 14, 7, 2], targets = [9, 1, 14, 7, 2, 5];

    const cpu = arch.build(cfg, mulberry32(5));
    const hc = applyLora(cpu, 4, 8, mulberry32(99));
    // Move B off zero, or every adapter contributes nothing and this proves little.
    for (const t of hc.groups.aux) for (let i = 0; i < t.data.length; i++) t.data[i] += 0.05;
    const cpuLoss = crossEntropy(cpu.forward(ids), targets);
    backward(cpuLoss, 1);
    const cpuGrads = hc.groups.aux.map((p) => p.grad.slice());
    clearLora();

    const model = arch.build(cfg, mulberry32(5));
    const h = applyLora(model, 4, 8, mulberry32(99));
    for (const t of h.groups.aux) for (let i = 0; i < t.data.length; i++) t.data[i] += 0.05;
    gpu.install();
    // With recompute on, which is the combination the measured table uses and
    // which nothing else covers: the replay rebuilds the adapter subgraph inside
    // each block, so a double-counted adapter gradient would show up here as a
    // 2x against the CPU reference rather than as a plausible learning rate.
    setCheckpointing(true);
    const regionsBefore = gpu.regionCount();
    let ok = true;
    try {
      const loss = crossEntropy(model.forward(ids), targets);
      backward(loss, 1);
      await gpu.sync([loss]);
      const dl = Math.abs(loss.data[0] - cpuLoss.data[0]);
      if (dl > 1e-3 + 1e-3 * Math.abs(cpuLoss.data[0])) {
        console.log(`    MISMATCH ${name} lora loss: ${loss.data[0]} vs ${cpuLoss.data[0]}`);
        ok = false;
      }
      for (let i = 0; i < cpuGrads.length; i++) {
        ok = compare(`lora.${name}.dAdapter${i}`, h.groups.aux[i].grad, cpuGrads[i], BWD) && ok;
      }
      // "+ recompute" in the label has to be a claim, not a word: an arch that
      // stopped calling checkpoint() would otherwise keep this green.
      if (gpu.regionCount() === regionsBefore) {
        console.log(`    MISMATCH ${name} opened no recompute regions`);
        ok = false;
      }
    } finally {
      setCheckpointing(false);
      gpu.uninstall();
    }

    // The freeze, checked on the DEVICE. Reading `w.grad` on the host proves
    // nothing: sync() only stages gradients for requiresGrad tensors, so a
    // frozen base's host array stays zero whatever the kernels did. Two real
    // properties instead: every frozen base shares ONE buffer (so no full-size
    // accumulator was allocated for any of them), and zeroing that buffer and
    // running another backward leaves it zero (so nothing writes through it).
    // It has to be zeroed first, since a pooled buffer arrives dirty.
    gpu.install();
    let sharedStub = true;
    let stubDirty = 0;
    try {
      const bases = model.paramGroups().muon;
      const stubBuf = gpu.buffersFor(bases[0]).grad;
      for (const w of bases) if (gpu.buffersFor(w).grad !== stubBuf) sharedStub = false;
      gpu.writeStateBuffer(stubBuf, new Float32Array(64));
      const l2 = crossEntropy(model.forward(ids), targets);
      backward(l2, 1);
      await gpu.sync([l2]);
      const stub = await gpu.readStateBuffer(stubBuf, 64);
      for (const v of stub) stubDirty = Math.max(stubDirty, Math.abs(v));
    } finally {
      gpu.uninstall();
    }
    if (!sharedStub || stubDirty !== 0) {
      console.log(`    MISMATCH ${name} freeze: shared=${sharedStub} stub max=${stubDirty}`);
      ok = false;
    }

    // merge/unmerge must return the weights bit-close to where they started.
    const snapshot = model.paramGroups().muon.map((w) => w.data.slice());
    h.merge();
    h.unmerge();
    let drift = 0;
    const bases = model.paramGroups().muon;
    for (let i = 0; i < bases.length; i++) {
      for (let j = 0; j < snapshot[i].length; j++) {
        drift = Math.max(drift, Math.abs(bases[i].data[j] - snapshot[i][j]));
      }
    }
    if (drift > 1e-5) {
      console.log(`    MISMATCH ${name} merge/unmerge drift ${drift}`);
      ok = false;
    }
    clearLora();

    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok " : "FAIL"} ${name} lora + recompute vs CPU (${h.adapted} adapters, ` +
        `stub ${stubDirty}, merge drift ${drift.toExponential(1)})`,
    );
  }
}

/**
 * Element-wise ops given the SAME tensor twice.
 *
 * Issue #48 predicted that `add(t, t)` loses one of its two gradient
 * accumulations on the GPU, because the backward binds one buffer to two
 * `read_write` slots and does two read-modify-writes. It does not, and both
 * halves of why are worth pinning rather than rediscovering.
 *
 * The binding is legal. WebGPU's compatible-usage-list rule grants an explicit
 * "usage scope storage exception": multiple `storage` usages of one buffer in a
 * usage scope are allowed *even though they are writable*. So this is not a
 * validation error waiting to fire on a stricter backend.
 *
 * The arithmetic is defined too. These kernels run one invocation per element,
 * so both writes come from the same thread to the same address in program
 * order. There is no cross-thread race for the exception to expose.
 *
 * What the exception does NOT cover is a mixed list. Binding one buffer as
 * `read` and as `read_write` in the same dispatch is neither all-read-only nor
 * all-`storage`, and would be a real validation error. Nothing in this tree does
 * that: `mul`'s backward aliases `AB`/`BB` (both `read`) and `DA`/`DB` (both
 * `storage`), never one of each.
 *
 * All of those are properties of the current kernels that a future change could
 * break, which is what this is here for.
 */
async function aliasedBinaryOpParity(gpu: WebGPUBackend) {
  for (const n of [1, 6, 257, 5000]) {
    const t = randTensor([n], mulberry32(4));
    await opCase(gpu, `add(t, t) [n=${n}]`, [t], () => add(t, t));
    const u = randTensor([n], mulberry32(9));
    await opCase(gpu, `mul(t, t) [n=${n}]`, [u], () => mul(u, u));
  }
}

/**
 * Out-of-range indices, refused on the host, with an installed backend.
 *
 * The two losses need one arm each because they validate BELOW the backend
 * dispatch, so each implementation carries its own call and this is what proves
 * the device ones are still there. The kernel could not catch it for them: the
 * logits buffer is bound whole, so `LOG[t * V + tgt]` with `tgt >= V` is an
 * in-bounds read of the next row, measured returning exactly the CPU's wrong
 * value.
 *
 * The embedding arm is here for the opposite reason. Its guard sits ABOVE the
 * dispatch, so no backend can skip it, and what this pins is that placement:
 * move the call below and the CPU cases in gradcheck still pass while this
 * fails.
 */
async function targetRangeGate(gpu: WebGPUBackend) {
  const T = 3, H = 4, V = 6;
  const hid = randTensor([T, H], mulberry32(31));
  const w = randTensor([V, H], mulberry32(37));
  gpu.install();
  try {
    const refused = (
      fn: () => unknown,
      pattern = /is not -1 \(ignore\) or an integer in \[0,/,
    ) => {
      try {
        fn();
        return false;
      } catch (e) {
        return pattern.test((e as Error).message);
      }
    };
    // The legal arm first, and read back: on the device `loss.data` holds zeros
    // until sync, so checking it before would pass on an unwritten buffer.
    const good = crossEntropy(linear(hid, w), [0, V - 1, 1]);
    await gpu.sync([good]);
    const scores = Number.isFinite(good.data[0]) && good.data[0] > 0;

    const logits = linear(hid, w);
    const dense = refused(() => crossEntropy(logits, [0, V, 1]));
    const fused = refused(() => fusedCrossEntropy(hid, w, [0, V, 1], 2));
    // The input side too, reusing w as the table since it is already [V, H].
    // Its guard sits above the backend dispatch, so this fails if someone moves
    // it below, where an installed backend would skip it.
    const embed = refused(() => embedding(w, [0, V, 1]), /^embedding: id \d+ at position \d+ /);
    // softCrossEntropy's guard is above the dispatch too, and the GPU path never
    // had one of its own: that was #61.
    const soft = refused(
      () => softCrossEntropy(logits, [0, V, 1], [0.5, 0.5, 1], 1),
      /^softCrossEntropy: teacher id \d+ at slot \d+ of row \d+ /,
    );
    const ok = dense && fused && embed && soft && scores;
    if (!ok) failures++;
    console.log(
      `  ${ok ? "ok " : "FAIL"} GPU refuses an index outside its table ` +
        `(dense ${dense}, fused ${fused}, embedding ${embed}, softCE ${soft}, ` +
        `V-1 scores ${good.data[0].toFixed(4)})`,
    );
  } finally {
    // The legal arm above recorded work; draining here keeps the gate
    // independent of what runs after it.
    await gpu.sync([]);
    gpu.uninstall();
  }
}

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
    // Now read back, if the fence worked the values are those of the linear op.
    await gpu.sync([gpuOut]);
    ok = compare("syncFence.out", gpuOut.data, cpuData, FWD) && ok;
  } finally {
    gpu.uninstall();
  }
  if (!ok) failures++;
  console.log(`  ${ok ? "ok " : "FAIL"} sync() fences GPU even with no readback`);
}

/**
 * The two checkpoint triggers, and the fact that either one fires a write.
 * The wall-clock trigger exists because a step count is a proxy for time that
 * stops holding the moment step time changes, so a long run bounds what an
 * interruption costs in minutes instead. ANDing the triggers rather than ORing
 * them would quietly stretch that bound back out to the step cadence, which is
 * a silent failure everywhere except here.
 */
async function checkpointCadence(gpu: WebGPUBackend) {
  const cfg = microConfig();
  const rngTok = mulberry32(0x5ec2);
  const tokens = Array.from({ length: 160 }, () => Math.floor(rngTok() * cfg.vocabSize));
  const hyper = { lr: 0.02, momentum: 0.95, aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 } };

  const firedAt = async (cadence: { checkpointEvery?: number; checkpointEveryMs?: number }) => {
    const model = new Gemma3Model(cfg, mulberry32(5));
    const g = model.paramGroups();
    const hit: number[] = [];
    await trainLMGpuResident(model, gpu, {
      tokens,
      seqLen: 8,
      steps: 6,
      batchPerStep: 1,
      optimizer: new MuonGpu(gpu, g.muon, g.aux, hyper),
      logEvery: 100,
      rng: mulberry32(7),
      onCheckpoint: (step) => {
        hit.push(step);
      },
      ...cadence,
    });
    return hit;
  };

  // A step takes milliseconds, so 1e-3 ms is due at every step and 1e9 ms at none.
  const cases: [string, { checkpointEvery?: number; checkpointEveryMs?: number }, number[]][] = [
    ["steps only", { checkpointEvery: 2 }, [2, 4]],
    ["wall clock only", { checkpointEveryMs: 1e-3 }, [1, 2, 3, 4, 5]],
    ["both, only steps due", { checkpointEvery: 2, checkpointEveryMs: 1e9 }, [2, 4]],
    ["both, only the clock due", { checkpointEvery: 1e9, checkpointEveryMs: 1e-3 }, [
      1,
      2,
      3,
      4,
      5,
    ]],
    ["neither set", {}, []],
    ["clock set but never due", { checkpointEveryMs: 1e9 }, []],
  ];

  for (const [why, cadence, want] of cases) {
    const got = await firedAt(cadence);
    const ok = JSON.stringify(got) === JSON.stringify(want);
    if (!ok) {
      failures++;
      console.log(`    MISMATCH ${why}: fired at [${got}], expected [${want}]`);
    }
    console.log(`  ${ok ? "ok " : "FAIL"} checkpoint cadence (${why})`);
  }
}
