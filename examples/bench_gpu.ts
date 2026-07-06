// Wall-clock training benchmark: run the same model+data in f32 and f16 and
// report steps/s and the loss trajectory for each, so the mixed-precision win
// (and stability) is measured on the actual target GPU rather than guessed.
//
//   deno run -A --unstable-webgpu examples/bench_gpu.ts [hidden] [layers] [seqLen] [steps] [batch]
//
// f16 = f16 GEMM multiply, f32 accumulate (setPrecision("f16")); needs shader-f16.

import { initWebGPU } from "../src/backend/webgpu.ts";
import { MuonGpu } from "../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";
import { gemma3Config, gemma3ParamCount } from "../src/model/config.ts";
import { Gemma3Model } from "../src/model/gemma3.ts";
import { mulberry32 } from "../src/model/autograd.ts";

// deno-lint-ignore no-explicit-any
const A: string[] = (globalThis as any).Deno?.args ?? [];
const hidden = A[0] ? Number(A[0]) : 512;
const layers = A[1] ? Number(A[1]) : 6;
const seqLen = A[2] ? Number(A[2]) : 512;
const steps = A[3] ? Number(A[3]) : 40;
const batch = A[4] ? Number(A[4]) : 8;
const vocab = 4096;

const maybeGpu = await initWebGPU();
if (!maybeGpu) throw new Error("no WebGPU (run under Deno with --unstable-webgpu)");
const gpu = maybeGpu; // non-null for use inside run()
console.log(`adapter: ${gpu.adapterName} | f16Supported: ${gpu.f16Supported}`);
const cfg = gemma3Config(vocab, hidden, layers, Math.max(512, seqLen));
console.log(
  `model: hidden=${hidden} layers=${layers} seqLen=${seqLen} batch=${batch} steps=${steps} ` +
    `~${(gemma3ParamCount(cfg) / 1e6).toFixed(1)}M params\n`,
);

// A small repeating pattern so loss visibly drops (learnable), shared by both runs.
const rng = mulberry32(3);
const need = seqLen * batch * (steps + 4) + 2;
const base = Array.from({ length: 96 }, () => Math.floor(rng() * vocab));
const toks = Array.from({ length: need }, (_, i) => base[i % base.length]);

async function run(precision: "f32" | "f16") {
  const model = new Gemma3Model(cfg, mulberry32(1234), { baseWidth: 128 });
  const g = model.paramGroups();
  const opt = new MuonGpu(gpu, g.muon, g.aux, {
    lr: 0.02,
    aux: { lr: 0.003, weightDecay: 0, clip: 1 },
  });
  const losses: number[] = [];
  // Warm-up step 0 compiles pipelines; time only the steady-state steps.
  let t0 = 0;
  await trainLMGpuResident(model, gpu, {
    tokens: toks,
    seqLen,
    steps,
    batchPerStep: batch,
    optimizer: opt,
    precision,
    rng: mulberry32(7),
    logEvery: Math.max(1, Math.round(steps / 6)),
    onLog: (_s, l) => losses.push(l),
    onStepTime: () => {
      if (t0 === 0) t0 = performance.now(); // start clock after step 0
    },
  });
  return { losses, ms: (performance.now() - t0) / (steps - 1) };
}

const f32 = await run("f32");
const f16 = await run("f16");
const tps = (ms: number) => (batch * seqLen) / (ms / 1000);
const fmt = (a: number[]) => a.map((l) => l.toFixed(2)).join("→");
console.log(
  `f32: ${f32.ms.toFixed(1)} ms/step  ${tps(f32.ms).toFixed(0)} tok/s   loss ${fmt(f32.losses)}`,
);
console.log(
  `f16: ${f16.ms.toFixed(1)} ms/step  ${tps(f16.ms).toFixed(0)} tok/s   loss ${fmt(f16.losses)}`,
);
console.log(`\nf16 speedup: ${(f32.ms / f16.ms).toFixed(2)}x`);
gpu.destroy();
