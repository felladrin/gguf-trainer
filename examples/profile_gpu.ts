// Per-kernel GPU-time profile of one training step, via the WebGPU
// timestamp-query feature. Answers "where does the step actually go?" so kernel
// work is aimed at the real bottleneck instead of a guess.
//
// How it works: WebGPUBackend.startProfile() puts each dispatch in its own
// compute pass with a begin/end timestamp pair, attributed to its op label
// (linear, attention, rmsnorm, muon, …); the deltas accumulate per label across
// sync() windows and stopProfile() returns them sorted. This is numerically a
// no-op (same kernels, buffers, order — only pass batching changes), but it
// SERIALIZES dispatches, so absolute ms are inflated and biased toward ops with
// many small dispatches (Muon's Newton–Schulz especially). Read the RELATIVE
// split, and use examples/… wall-clock runs for true step time.
//
//   deno run -A --unstable-webgpu examples/profile_gpu.ts [hidden] [layers] [seqLen] [steps]
//   deno run -A --unstable-webgpu examples/profile_gpu.ts 512 4 1024 4   # defaults
//
// A dispatch grid is capped at 65535 workgroups per dim, so very large
// seqLen×vocab can exceed it in the cross-entropy kernels; keep the probe shape
// modest (the split, not the absolute size, is the point).

import { initWebGPU } from "../src/backend/webgpu.ts";
import { MuonGpu } from "../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";
import { paramCount, scaleConfig } from "../src/model/config.ts";
import { Qwen3Model } from "../src/model/qwen3.ts";
import { mulberry32 } from "../src/model/autograd.ts";

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  return (globalThis as any).Deno?.args ?? [];
}

async function main() {
  const a = args();
  const hidden = a[0] ? Number(a[0]) : 512;
  const layers = a[1] ? Number(a[1]) : 4;
  const seqLen = a[2] ? Number(a[2]) : 1024;
  const profSteps = a[3] ? Number(a[3]) : 4;
  const vocab = 4096;
  const batch = 1;

  const gpu = await initWebGPU();
  if (!gpu) {
    console.error("No WebGPU (run under Deno: deno run -A --unstable-webgpu ...).");
    return;
  }
  console.log(`adapter: ${gpu.adapterName}`);
  if (!gpu.timestampSupported) {
    console.error("timestamp-query not granted by this device; cannot profile.");
    return;
  }

  const cfg = scaleConfig(vocab, hidden, layers, Math.max(512, seqLen));
  const model = new Qwen3Model(cfg, mulberry32(1234), { baseWidth: 128 });
  console.log(
    `model: hidden=${hidden} layers=${layers} seqLen=${seqLen} batch=${batch} ` +
      `~${(paramCount(cfg) / 1e6).toFixed(1)}M params`,
  );

  // Synthetic tokens — profiling depends on shapes, not corpus content.
  const rng = mulberry32(7);
  const need = seqLen * batch * (profSteps + 6) + 2;
  const toks = new Array(need);
  for (let i = 0; i < need; i++) toks[i] = Math.floor(rng() * vocab);

  const groups = model.paramGroups();
  const opt = new MuonGpu(gpu, groups.muon, groups.aux, {
    lr: 0.02,
    aux: { lr: 0.003, weightDecay: 0, clip: 1.0 },
  });

  // Warm-up (pipeline compile + steady state), then a profiled window. The
  // resident trainer install/uninstalls the backend; we bracket the profiled
  // steps with start/stopProfile around a second short run on the same model.
  await trainLMGpuResident(model, gpu, {
    tokens: toks,
    seqLen,
    steps: 3,
    batchPerStep: batch,
    optimizer: opt,
    rng: mulberry32(99),
    logEvery: 1000,
  });

  gpu.startProfile(2048);
  await trainLMGpuResident(model, gpu, {
    tokens: toks,
    seqLen,
    steps: profSteps,
    batchPerStep: batch,
    optimizer: opt,
    rng: mulberry32(99),
    logEvery: 1000,
  });
  const { kernels, overflow } = gpu.stopProfile();

  const total = kernels.reduce((acc, k) => acc + k.ms, 0);
  console.log(
    `\nprofiled ${profSteps} steps — GPU kernel time ${(total / profSteps).toFixed(2)} ms/step`,
  );
  console.log("(serialized: each dispatch in its own pass — read the split, not absolute ms)");
  if (overflow) {
    console.log("WARNING: query-slot overflow — some dispatches untimed (raise the cap)");
  }
  console.log("\n  kernel            ms/step     %    dispatches/step");
  console.log("  ----------------  --------  -----  ---------------");
  for (const k of kernels) {
    const ms = (k.ms / profSteps).toFixed(3).padStart(8);
    const pct = ((k.ms / total) * 100).toFixed(1).padStart(5);
    const cnt = (k.count / profSteps).toFixed(0).padStart(6);
    console.log(`  ${k.label.padEnd(16)}  ${ms}  ${pct}  ${cnt}`);
  }
  gpu.destroy();
}

main().catch((e) => {
  console.error(e?.stack ?? String(e));
  // deno-lint-ignore no-explicit-any
  (globalThis as any).Deno?.exit?.(1);
});
