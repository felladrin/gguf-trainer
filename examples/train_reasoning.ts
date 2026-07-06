// Real run: train a Gemma3 (SWA) reasoning model from scratch on the
// pretokenized OpenThoughts corpus (examples/prepare_reasoning.ts) and write a
// llama.cpp-loadable GGUF (arch "gemma3") with a chat template + <think>
// special tokens. Device-resident Muon+AdamW, WSD schedule, mid-run GGUF
// checkpoints so an interrupted long run still leaves a loadable model.
//
//   deno run -A --unstable-webgpu examples/train_reasoning.ts [hidden] [layers] [steps] [seqLen] [batch] [window]
// Defaults: hidden 512, layers 12 (~64M), 20000 steps, seqLen 2048, batch 2, window 1024.
// GGUF_F16=1 enables f16-compute GEMM (Strix Halo win).

import { readGGUF } from "../src/gguf/gguf.ts";
import { readFileText, writeFileBytes } from "../src/io.ts";
import { crossEntropy, mulberry32 } from "../src/model/autograd.ts";
import { gemma3Config, gemma3ParamCount } from "../src/model/config.ts";
import { Gemma3Model } from "../src/model/gemma3.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import type { TokenizerData } from "../src/tokenizer/bpe.ts";
import { buildGemma3GGUF } from "../src/export/export_gguf.ts";
import { wsdSchedule } from "../src/train/schedule.ts";
import { diskTokenSource, tokenBytes } from "../src/data/tokens.ts";
import type { TokenSource } from "../src/data/tokens.ts";
import { initWebGPU } from "../src/backend/webgpu.ts";
import type { WebGPUBackend } from "../src/backend/webgpu.ts";
import { MuonGpu } from "../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  return g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
}
function die(msg: string): never {
  console.error("train_reasoning: " + msg);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
  throw new Error(msg);
}
function argmax(row: Float32Array): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

/** Greedy chat completion (GPU forward, one sync/token), stops at <|im_end|>. */
async function chatGreedy(
  gpu: WebGPUBackend,
  model: Gemma3Model,
  tok: BPETokenizer,
  question: string,
  n: number,
): Promise<string> {
  const prompt = `<|im_start|>user\n${question}<|im_end|>\n<|im_start|>assistant\n`;
  const ids = tok.encode(prompt);
  const eos = tok.eosId;
  gpu.install();
  try {
    gpu.uploadParams(model.params());
    for (let i = 0; i < n; i++) {
      const ctx = ids.slice(-model.cfg.maxSeq);
      const logits = model.forward(ctx);
      await gpu.sync([logits]);
      const V = model.cfg.vocabSize;
      const base = (ctx.length - 1) * V;
      const next = argmax(logits.data.subarray(base, base + V));
      ids.push(next);
      if (next === eos) break;
    }
  } finally {
    gpu.uninstall();
  }
  return tok.decode(ids);
}

async function main() {
  const a = args();
  const hidden = a[0] ? Number(a[0]) : 512;
  const layers = a[1] ? Number(a[1]) : 12;
  const steps = a[2] ? Number(a[2]) : 20000;
  const seqLen = a[3] ? Number(a[3]) : 2048;
  const batch = a[4] ? Number(a[4]) : 2;
  const window = a[5] ? Number(a[5]) : 1024;
  const muonLr = 0.02, auxLr = 3e-3, baseWidth = 128;

  console.log("=== gemma3 reasoning model: train-from-scratch -> GGUF ===\n");
  const gpu = await initWebGPU();
  if (!gpu) die("no WebGPU (run under Deno with --unstable-webgpu)");
  console.log(`WebGPU adapter: ${gpu.adapterName}`);
  // deno-lint-ignore no-explicit-any
  const wantF16 = (globalThis as any).Deno?.env?.get?.("GGUF_F16") === "1";
  const precision: "f16" | "f32" = wantF16 && gpu.f16Supported ? "f16" : "f32";
  console.log(`Precision: ${precision}`);

  const dir = new URL(".", import.meta.url).pathname;
  const prefix = `${dir}reasoning`;
  let tokData: TokenizerData;
  try {
    tokData = JSON.parse(await readFileText(`${prefix}.tokenizer.json`));
  } catch {
    return die(`cannot read ${prefix}.tokenizer.json — run examples/prepare_reasoning.ts first`);
  }
  const tok = BPETokenizer.fromData(tokData);
  const chatTemplate = await readFileText(`${prefix}.template.txt`).catch(() => undefined);
  const bpt = tokenBytes(tok.vocabSize);
  let src: TokenSource;
  try {
    src = await diskTokenSource(`${prefix}.tokens`, bpt);
  } catch {
    return die(`cannot open ${prefix}.tokens — run examples/prepare_reasoning.ts first`);
  }
  const tokensPerStep = batch * seqLen;
  console.log(
    `Corpus: ${(src.length / 1e6).toFixed(1)}M tokens (${bpt}B), vocab ${tok.vocabSize}, ` +
      `eos=${tok.eosId}; ${steps} x ${batch} x ${seqLen} = ${
        (steps * tokensPerStep / 1e6).toFixed(0)
      }M tokens (${(steps * tokensPerStep / src.length).toFixed(1)} epochs)`,
  );

  const cfg = gemma3Config(tok.vocabSize, hidden, layers, Math.max(4096, seqLen), 64, window);
  if (seqLen > cfg.maxSeq) die(`seqLen ${seqLen} exceeds context_length ${cfg.maxSeq}`);
  const model = new Gemma3Model(
    cfg,
    mulberry32(1234),
    baseWidth === hidden ? undefined : { baseWidth },
  );
  console.log(
    `Model: gemma3, ${cfg.nLayers} layers, hidden=${cfg.hiddenSize}, heads=${cfg.nHeads}/` +
      `${cfg.nKVHeads}, headDim=${cfg.headDim}, ffn=${cfg.ffnDim}, window=${cfg.slidingWindow}, ` +
      `pattern=${cfg.swaPattern} (5:1 SWA:global), ~${
        (gemma3ParamCount(cfg) / 1e6).toFixed(1)
      }M params`,
  );

  // Trust gate: GPU forward+loss must match the CPU reference at init.
  const probeIn = src.window(0, 16), probeTgt = src.window(1, 16);
  const cpuLoss = crossEntropy(model.forward(probeIn), probeTgt).data[0];
  gpu.install();
  let gpuLoss: number;
  try {
    const l = crossEntropy(model.forward(probeIn), probeTgt);
    await gpu.sync([l]);
    gpuLoss = l.data[0];
  } finally {
    gpu.uninstall();
  }
  const drift = Math.abs(gpuLoss - cpuLoss);
  console.log(
    `Parity probe: CPU ${cpuLoss.toFixed(4)} vs GPU ${gpuLoss.toFixed(4)} (|Δ|=${
      drift.toExponential(1)
    })`,
  );
  if (drift > 1e-3 + 1e-3 * Math.abs(cpuLoss)) die("GPU/CPU parity probe failed");

  const groups = model.paramGroups();
  const opt = new MuonGpu(gpu, groups.muon, groups.aux, {
    lr: muonLr,
    momentum: 0.95,
    aux: { lr: auxLr, weightDecay: 0.0, clip: 1.0 },
  });
  const schedule = wsdSchedule({
    warmupSteps: Math.max(1, Math.round(steps * 0.05)),
    stableSteps: Math.max(0, steps - Math.round(steps * 0.05) - Math.round(steps * 0.2)),
    cooldownSteps: Math.max(1, Math.round(steps * 0.2)),
    minScale: 0.1,
  });

  const outPath = `${dir}reasoning-gemma3-f16.gguf`;
  const exportGGUF = async (): Promise<Uint8Array> => {
    const b = buildGemma3GGUF(model, tok.export(), cfg, {
      quant: "f16",
      name: "reasoning-gemma3",
      chatTemplate,
    });
    await writeFileBytes(outPath, b);
    return b;
  };
  const checkpointEvery = Math.min(500, Math.max(1, Math.round(steps / 40)));

  let firstLoss = 0, lastLoss = 0, stepIdx = 0, fwdMs = 0, optMs = 0;
  const t0 = Date.now();
  await trainLMGpuResident(model, gpu, {
    tokens: src,
    seqLen,
    steps,
    batchPerStep: batch,
    optimizer: opt,
    schedule,
    precision,
    logEvery: Math.max(1, Math.round(steps / 100)),
    rng: mulberry32(7),
    checkpointEvery,
    onCheckpoint: async (step) => {
      const b = await exportGGUF();
      const el = (Date.now() - t0) / 1000;
      console.log(
        `  [ckpt @ ${step}] ${outPath.split("/").pop()} ${(b.length / 1e6).toFixed(0)}MB, ` +
          `loss ${lastLoss.toFixed(3)}, ${(step / el).toFixed(2)} steps/s`,
      );
    },
    onLog: (step, loss) => {
      if (step === 0) firstLoss = loss;
      lastLoss = loss;
      const el = (Date.now() - t0) / 1000;
      console.log(
        `  step ${String(step).padStart(6)}  loss ${loss.toFixed(4)}  (${
          (step / Math.max(1, el)).toFixed(2)
        } st/s)`,
      );
    },
    onStepTime: (fwd, o) => {
      if (stepIdx++ === 0) return;
      fwdMs += fwd;
      optMs += o;
    },
  });
  src.close();
  const secs = (Date.now() - t0) / 1000;
  const warm = Math.max(1, steps - 1);
  console.log(
    `\nTraining: loss ${firstLoss.toFixed(3)} -> ${lastLoss.toFixed(3)} in ${
      (secs / 60).toFixed(1)
    }min ` +
      `(${(steps / secs).toFixed(2)} steps/s; fwd+bwd ${(fwdMs / warm).toFixed(0)}ms, opt ${
        (optMs / warm).toFixed(0)
      }ms/step)`,
  );

  console.log(`\nSample:\n${await chatGreedy(gpu, model, tok, "What is 12 + 30?", 200)}`);

  const bytes = await exportGGUF();
  const g = readGGUF(bytes);
  if (g.metadata.get("general.architecture") !== "gemma3") die("exported arch != gemma3");
  console.log(
    `\nWrote ${outPath} (${
      (bytes.length / 1e6).toFixed(0)
    } MB, ${g.tensors.length} tensors, gemma3 ✓)`,
  );
  console.log("=== run complete ===");
}

main().catch((e) => die(String(e?.stack ?? e)));
