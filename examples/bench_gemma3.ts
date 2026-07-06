// Step-time probe for a real ~100M Gemma3 (SWA) config, so the training run's
// seqLen/batch are chosen from measurement, not guesswork.
//
//   deno run -A --unstable-webgpu examples/bench_gemma3.ts [hidden] [layers] [seqLen] [steps] [batch] [window]

import { initWebGPU } from "../src/backend/webgpu.ts";
import { MuonGpu } from "../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";
import { gemma3Config, gemma3ParamCount } from "../src/model/config.ts";
import { Gemma3Model } from "../src/model/gemma3.ts";
import { mulberry32 } from "../src/model/autograd.ts";

// deno-lint-ignore no-explicit-any
const A: string[] = (globalThis as any).Deno?.args ?? [];
const hidden = A[0] ? Number(A[0]) : 640;
const layers = A[1] ? Number(A[1]) : 16;
const seqLen = A[2] ? Number(A[2]) : 4096;
const steps = A[3] ? Number(A[3]) : 6;
const batch = A[4] ? Number(A[4]) : 1;
const window = A[5] ? Number(A[5]) : 1024;
const vocab = 32768;

const gpu = await initWebGPU();
if (!gpu) throw new Error("no WebGPU");
const cfg = gemma3Config(vocab, hidden, layers, Math.max(seqLen, 512), 64, window);
console.log(`adapter: ${gpu.adapterName} | f16: ${gpu.f16Supported}`);
console.log(
  `gemma3: hidden=${hidden} layers=${layers} heads=${cfg.nHeads}/${cfg.nKVHeads} ffn=${cfg.ffnDim} ` +
    `window=${window} pattern=${cfg.swaPattern} seqLen=${seqLen} batch=${batch} ` +
    `~${(gemma3ParamCount(cfg) / 1e6).toFixed(1)}M params`,
);

const rng = mulberry32(3);
const need = seqLen * batch * (steps + 2) + 2;
const base = Array.from({ length: 128 }, () => Math.floor(rng() * vocab));
const toks = Array.from({ length: need }, (_, i) => base[i % base.length]);

const model = new Gemma3Model(cfg, mulberry32(1234), { baseWidth: 128 });
const g = model.paramGroups();
const opt = new MuonGpu(gpu, g.muon, g.aux, {
  lr: 0.02,
  aux: { lr: 0.003, weightDecay: 0, clip: 1 },
});

let t0 = 0;
const losses: number[] = [];
await trainLMGpuResident(model, gpu, {
  tokens: toks,
  seqLen,
  steps,
  batchPerStep: batch,
  optimizer: opt,
  precision: gpu.f16Supported ? "f16" : "f32",
  rng: mulberry32(7),
  logEvery: 1,
  onLog: (_s, l) => losses.push(l),
  onStepTime: () => {
    if (t0 === 0) t0 = performance.now(); // start clock after step 0 (pipeline compiles)
  },
});
const ms = (performance.now() - t0) / (steps - 1);
const tps = (batch * seqLen) / (ms / 1000);
console.log(
  `\n${ms.toFixed(0)} ms/step   ${tps.toFixed(0)} tok/s   loss ${
    losses.map((l) => l.toFixed(2)).join("→")
  }`,
);
console.log(`(tokens/hour ≈ ${(tps * 3600 / 1e6).toFixed(1)}M)`);
gpu.destroy();
