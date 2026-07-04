// Real (non-toy) run: train a Qwen3 from scratch on the pretokenized TinyStories
// corpus and write a llama.cpp-loadable GGUF. This is the turnkey entry point the
// roadmap hands off — everything it needs (streaming loader, scaleConfig, muP
// init, GPU-resident Muon + AdamW, WSD schedule, GGUF export) is already in the
// library; this wires them to a real corpus with sensible defaults.
//
// Prerequisite: run the pretokenizer once to produce the corpus + vocab:
//   deno run -A examples/pretokenize.ts corpus/tinystories-valid.txt examples/tinystories 8192 10
//
// Then train (Deno only — WebGPU):
//   deno run -A examples/train_tinystories.ts                     # defaults below (~14M params)
//   deno run -A examples/train_tinystories.ts 512 8 6000          # ~33M, 6000 steps
//   deno run -A examples/train_tinystories.ts 128 2 12 64 4       # fast smoke test
// Positional args: [hidden=384] [layers=6] [steps=3000] [seqLen=256] [batch=16]
//                  [muonLr=0.02] [auxLr=0.003] [baseWidth=128]
//
// baseWidth is the muP proxy width the lrs were tuned at: init scales the tied
// embedding by sqrt(baseWidth/hidden) so the readout logits stay O(1) as you
// widen, and the same lrs transfer (DESIGN.md item 2). Set baseWidth==hidden to
// disable muP (standard 0.02 init).

import { readGGUF } from "../src/gguf/gguf.ts";
import { dequantize } from "../src/gguf/quantize.ts";
import { readFileText, writeFileBytes } from "../src/io.ts";
import { crossEntropy, mulberry32 } from "../src/model/autograd.ts";
import { paramCount, scaleConfig } from "../src/model/config.ts";
import { Qwen3Model } from "../src/model/qwen3.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import type { TokenizerData } from "../src/tokenizer/bpe.ts";
import { buildGGUF } from "../src/export/export_gguf.ts";
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
  console.error("train_tinystories: " + msg);
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

/** Greedy sampling with the forward pass on the GPU (one sync per token). */
async function generateGpu(
  gpu: WebGPUBackend,
  model: Qwen3Model,
  tok: BPETokenizer,
  prompt: string,
  n: number,
): Promise<string> {
  const ids = tok.encode(prompt);
  gpu.install();
  try {
    gpu.uploadParams(model.params());
    for (let i = 0; i < n; i++) {
      const ctx = ids.slice(-model.cfg.maxSeq);
      const logits = model.forward(ctx);
      await gpu.sync([logits]);
      const V = model.cfg.vocabSize;
      const base = (ctx.length - 1) * V;
      ids.push(argmax(logits.data.subarray(base, base + V)));
    }
  } finally {
    gpu.uninstall();
  }
  return tok.decode(ids);
}

async function main() {
  const a = args();
  const hidden = a[0] ? Number(a[0]) : 384;
  const layers = a[1] ? Number(a[1]) : 6;
  const steps = a[2] ? Number(a[2]) : 3000;
  const seqLen = a[3] ? Number(a[3]) : 256;
  const batch = a[4] ? Number(a[4]) : 16;
  const muonLr = a[5] ? Number(a[5]) : 0.02;
  const auxLr = a[6] ? Number(a[6]) : 3e-3;
  const baseWidth = a[7] ? Number(a[7]) : 128;
  for (const [k, v] of Object.entries({ hidden, layers, steps, seqLen, batch, baseWidth })) {
    if (!Number.isFinite(v) || v <= 0) die(`${k} must be a positive number, got ${v}`);
  }

  console.log("=== Felladrin's GGUF Trainer +∞ :: TinyStories train-from-scratch -> GGUF ===\n");

  const gpu = await initWebGPU();
  if (!gpu) {
    die(
      "WebGPU is not available in this runtime. Run with Deno " +
        "(deno run -A examples/train_tinystories.ts); Node/Bun have no WebGPU.",
    );
  }
  console.log(`WebGPU adapter: ${gpu.adapterName}`);

  // 1. Reuse the exact vocab the corpus was tokenized with (pretokenize.ts).
  const dir = new URL(".", import.meta.url).pathname;
  const tokenizerPath = `${dir}tinystories.tokenizer.json`;
  const tokensPath = `${dir}tinystories.tokens`;
  let tokData: TokenizerData;
  try {
    tokData = JSON.parse(await readFileText(tokenizerPath));
  } catch {
    die(
      `cannot read ${tokenizerPath} — run the pretokenizer first:\n` +
        "  deno run -A examples/pretokenize.ts corpus/tinystories-valid.txt examples/tinystories 8192 10",
    );
  }
  const tok = BPETokenizer.fromData(tokData);
  const bpt = tokenBytes(tok.vocabSize);
  let src: TokenSource;
  try {
    src = await diskTokenSource(tokensPath, bpt);
  } catch {
    return die(`cannot open ${tokensPath} — run the pretokenizer first (see above)`);
  }
  const tokensPerStep = batch * seqLen;
  const epochs = (steps * tokensPerStep) / src.length;
  console.log(
    `Corpus: ${src.length} tokens (${bpt} B/token), vocab ${tok.vocabSize}; ` +
      `${steps} steps x ${batch} x ${seqLen} = ${
        (steps * tokensPerStep / 1e6).toFixed(1)
      }M tokens ` +
      `(${epochs.toFixed(1)} epochs)`,
  );

  // 2. Real-shape model via scaleConfig with muP init (embedding scaled by
  //    sqrt(baseWidth/hidden); hidden matmuls and attention are already
  //    width-correct, so the forward pass and the GGUF contract are unchanged).
  const cfg = scaleConfig(tok.vocabSize, hidden, layers, 512);
  if (seqLen > cfg.maxSeq) die(`seqLen ${seqLen} exceeds context_length ${cfg.maxSeq}`);
  const mup = baseWidth === hidden ? undefined : { baseWidth };
  const model = new Qwen3Model(cfg, mulberry32(1234), mup);
  console.log(
    `Model: qwen3, ${cfg.nLayers} layers, hidden=${cfg.hiddenSize}, ` +
      `heads=${cfg.nHeads}/${cfg.nKVHeads}, headDim=${cfg.headDim}, ffn=${cfg.ffnDim}, ` +
      `~${(paramCount(cfg) / 1e6).toFixed(1)}M params` +
      (mup ? `, muP init @ baseWidth ${baseWidth}` : ", standard init"),
  );

  // 3. Trust gate: GPU forward+loss must match the CPU reference at init before
  //    we spend the run on this backend (tests/gpu_parity.ts is the full check).
  const probeIn = src.window(0, 16);
  const probeTgt = src.window(1, 16);
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
    `Parity probe: CPU ${cpuLoss.toFixed(5)} vs GPU ${gpuLoss.toFixed(5)} ` +
      `(|Δ|=${drift.toExponential(1)})`,
  );
  if (drift > 1e-3 + 1e-3 * Math.abs(cpuLoss)) {
    die("GPU/CPU parity probe failed — do not train on this backend");
  }

  // 4. Device-resident GPU training with a WSD schedule (warmup 10%, cooldown
  //    20%). Muon on the hidden matmuls, AdamW on the aux group; muP keeps the
  //    tuned lrs valid across widths, so no width-lr scaling here.
  const groups = model.paramGroups();
  const opt = new MuonGpu(gpu, groups.muon, groups.aux, {
    lr: muonLr,
    momentum: 0.95,
    aux: { lr: auxLr, weightDecay: 0.0, clip: 1.0 },
  });
  const schedule = wsdSchedule({
    warmupSteps: Math.max(1, Math.round(steps * 0.1)),
    stableSteps: Math.max(0, steps - Math.round(steps * 0.1) - Math.round(steps * 0.2)),
    cooldownSteps: Math.max(1, Math.round(steps * 0.2)),
    minScale: 0.1,
  });
  console.log(
    `\nGPU Muon (lr ${muonLr}) on ${groups.muon.length} matrices; ` +
      `GPU AdamW (lr ${auxLr}) on ${groups.aux.length} tensors; WSD schedule\n`,
  );

  let firstLoss = 0, lastLoss = 0, stepIdx = 0, fwdMs = 0, optMs = 0, readback = 0;
  const t0 = Date.now();
  await trainLMGpuResident(model, gpu, {
    tokens: src,
    seqLen,
    steps,
    batchPerStep: batch,
    optimizer: opt,
    schedule,
    logEvery: Math.max(1, Math.round(steps / 30)),
    rng: mulberry32(7),
    onLog: (step, loss) => {
      if (step === 0) firstLoss = loss;
      lastLoss = loss;
      console.log(`  step ${String(step).padStart(5)}  loss ${loss.toFixed(4)}`);
    },
    onStepTime: (fwd, o, bytes) => {
      if (stepIdx++ === 0) return; // skip step 0 (pipeline compiles)
      fwdMs += fwd;
      optMs += o;
      readback = bytes;
    },
  });
  src.close();
  const secs = (Date.now() - t0) / 1000;
  const warm = Math.max(1, steps - 1);
  console.log(
    `\nTraining: loss ${firstLoss.toFixed(3)} -> ${lastLoss.toFixed(3)} in ${secs.toFixed(1)}s ` +
      `(${(steps / secs).toFixed(1)} steps/s); per-step fwd+bwd+sync ${
        (fwdMs / warm).toFixed(1)
      } ms, ` +
      `opt ${(optMs / warm).toFixed(1)} ms, readback ${(readback / 1024).toFixed(1)} KiB`,
  );
  if (!(lastLoss < firstLoss)) die("Loss did not decrease");

  // 5. Sample a short continuation so the run's quality is visible at a glance.
  console.log(`\nGreedy sample: "${await generateGpu(gpu, model, tok, "Once upon a time", 40)}"`);

  // 6. Export GGUF (f16 for a faithful resume) and verify it parses.
  const bytes = buildGGUF(model, tok.export(), cfg, { quant: "f16", name: "tinystories-qwen3" });
  const outPath = `${dir}tinystories-qwen3-f16.gguf`;
  await writeFileBytes(outPath, bytes);
  const g = readGGUF(bytes);
  if (g.metadata.get("general.architecture") !== "qwen3") die("exported arch != qwen3");
  const expected = 2 + (cfg.tieEmbeddings ? 0 : 1) + cfg.nLayers * 11;
  if (g.tensors.length !== expected) die(`tensor count ${g.tensors.length} != ${expected}`);
  const emb = g.tensors.find((t) => t.name === "token_embd.weight");
  if (!emb) die("token_embd.weight missing from export");
  const de = dequantize(emb.type, emb.data, emb.dims[0] * emb.dims[1]);
  for (let i = 0; i < de.length; i++) if (!Number.isFinite(de[i])) die("non-finite embedding");
  console.log(
    `\nWrote ${outPath} (${
      (bytes.length / 1e6).toFixed(1)
    } MB, ${g.tensors.length} tensors, arch=qwen3 ✓)`,
  );
  console.log(
    `Try it:  llama-cli -m ${outPath} -p "Once upon a time" -n 64 -st --simple-io --temp 0 </dev/null`,
  );
  console.log("\n=== training run complete ===");
}

main().catch((e) => die(String(e?.stack ?? e)));
