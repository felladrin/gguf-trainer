// Throughput probe for sizing a real pretrain run on a given machine. Builds a
// Gemma3 model at the requested shape, feeds SYNTHETIC random tokens through the
// exact device-resident training path (trainLMGpuResident + MuonGpu), times a
// handful of steps, and projects tokens/day so a multi-week run can be planned
// from a measured st/s instead of a guess. No corpus or tokenizer needed.
//
//   deno run -A --unstable-webgpu examples/bench_pretrain.ts [hidden] [layers] [seqLen] [batch] [vocab] [steps] [window]
// Defaults: hidden 640, layers 12, seqLen 2048, batch 8, vocab 32768, steps 12, window 1024.

import { mulberry32 } from "../src/model/autograd.ts";
import { gemma3Config, gemma3ParamCount } from "../src/model/config.ts";
import { Gemma3Model } from "../src/model/gemma3.ts";
import { memTokenSource } from "../src/data/tokens.ts";
import { initWebGPU } from "../src/backend/webgpu.ts";
import { MuonGpu } from "../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  return g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
}

async function main() {
  const a = args();
  const hidden = a[0] ? Number(a[0]) : 640;
  const layers = a[1] ? Number(a[1]) : 12;
  const seqLen = a[2] ? Number(a[2]) : 2048;
  const batch = a[3] ? Number(a[3]) : 8;
  const vocab = a[4] ? Number(a[4]) : 32768;
  const steps = a[5] ? Number(a[5]) : 12;
  const window = a[6] ? Number(a[6]) : 1024;

  console.log("=== bench_pretrain: throughput probe ===\n");
  const gpu = await initWebGPU();
  if (!gpu) throw new Error("no WebGPU (run under Deno with --unstable-webgpu)");
  console.log(`adapter: ${gpu.adapterName}`);

  const cfg = gemma3Config(vocab, hidden, layers, Math.max(8192, seqLen), 64, window);
  const model = new Gemma3Model(
    cfg,
    mulberry32(1234),
    hidden === 128 ? undefined : { baseWidth: 128 },
  );
  const params = (gemma3ParamCount(cfg) / 1e6).toFixed(1);
  console.log(
    `model: hidden ${hidden} x ${layers} layers, heads ${cfg.nHeads}/${cfg.nKVHeads}, ` +
      `ffn ${cfg.ffnDim}, window ${window}, vocab ${vocab}, ~${params}M params`,
  );
  console.log(`config: seqLen ${seqLen}, batch ${batch}, ${steps} timed steps\n`);

  // Synthetic corpus: enough random ids for many distinct windows.
  const n = seqLen * batch * 4 + seqLen + 1;
  const rng = mulberry32(42);
  const toks = new Uint16Array(n);
  for (let i = 0; i < n; i++) toks[i] = Math.floor(rng() * vocab);
  const src = memTokenSource(toks);

  const groups = model.paramGroups();
  const opt = new MuonGpu(gpu, groups.muon, groups.aux, {
    lr: 0.01,
    momentum: 0.95,
    aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 },
  });

  const times: number[] = [];
  let t0 = 0;
  await trainLMGpuResident(model, gpu, {
    tokens: src,
    seqLen,
    steps,
    batchPerStep: batch,
    optimizer: opt,
    logEvery: 1,
    rng: mulberry32(7),
    onLog: (step) => {
      const now = performance.now();
      // Skip the first 2 steps (shader compile + warm-up); time the steady state.
      if (step === 2) t0 = now;
      else if (step > 2) times.push(now);
      console.log(`  step ${step}${step < 3 ? " (warmup)" : ""}`);
    },
  });

  const tEnd = performance.now();
  const steady = steps - 3; // steps 3..steps-1
  const secPerStep = (tEnd - t0) / 1000 / steady;
  const stPerSec = 1 / secPerStep;
  const tokPerStep = batch * seqLen;
  const tokPerDay = stPerSec * 86400 * tokPerStep;
  console.log(
    `\n=== ${params}M params @ seqLen ${seqLen} batch ${batch}: ${secPerStep.toFixed(2)} s/step, ` +
      `${stPerSec.toFixed(3)} st/s ===`,
  );
  console.log(
    `throughput: ${(tokPerStep / 1e3).toFixed(1)}K tok/step, ${
      (tokPerDay / 1e6).toFixed(0)
    }M tok/day`,
  );
  console.log(
    `to see 714M tokens (1 epoch of the blend): ~${(714e6 / tokPerDay).toFixed(1)} days ` +
      `(${Math.round(714e6 / tokPerStep)} steps)`,
  );
}

main().catch((e) => {
  console.error(String(e?.stack ?? e));
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
});
