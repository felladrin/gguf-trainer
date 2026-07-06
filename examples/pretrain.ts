// Curriculum stage 1 (+ stage 0): pretrain a Gemma3 BASE model from scratch on
// unlabeled English, and produce the ONE shared tokenizer the whole curriculum
// reuses. This is where language/coherence is learned; instruct → reasoning →
// tool-calling later fine-tune from this checkpoint (each via loadGemma3FromGGUF).
//
// Stage 0 — shared tokenizer (built once, then reused): a byte-level BPE with
// ALL curriculum specials reserved up front (CURRICULUM_SPECIALS: ChatML turns +
// <think>/</think> + the tool tags). The vocab and embedding matrix freeze here,
// so a later stage cannot add a special without discarding trained embeddings —
// reserving now is the only chance. Pretraining never emits them (it sees raw
// text); their rows sit at init until their stage's data first uses them.
//
// Stage 1 — pretrain: full-sequence next-token LM loss (NOT assistant-masked —
// there are no turns yet), device-resident Muon + AdamW, WSD schedule, mid-run
// GGUF checkpoints. The export is a BASE model: no chat template, eos is the
// document boundary. Coherence emerges well before a full epoch on TinyStories.
//
//   deno run -A --unstable-webgpu examples/pretrain.ts [corpus.txt] [hidden] [layers] [steps] [seqLen] [batch]
// Defaults: corpus=corpus/tinystories-valid.txt, hidden 512, layers 6 (~28M),
//   steps 2000, seqLen 1024, batch 8. GGUF_F16=1 enables f16-compute GEMM.
//
// NOTE: the corpus is read whole (readFileText), so keep a single file under
// ~500 MB (V8's ~512 MB string cap). For the full multi-GB TinyStories/FineWeb
// run, point this at a few-hundred-MB slice, or add chunked file reading first.

import { readGGUF } from "../src/gguf/gguf.ts";
import { readFileText, writeFileBytes } from "../src/io.ts";
import { crossEntropy, mulberry32 } from "../src/model/autograd.ts";
import { gemma3Config, gemma3ParamCount } from "../src/model/config.ts";
import { Gemma3Model } from "../src/model/gemma3.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import type { TokenizerData } from "../src/tokenizer/bpe.ts";
import { CURRICULUM_SPECIALS } from "../src/data/chat.ts";
import { buildGemma3GGUF } from "../src/export/export_gguf.ts";
import { wsdSchedule } from "../src/train/schedule.ts";
import { diskTokenSource, tokenBytes, writeTokenFile } from "../src/data/tokens.ts";
import type { TokenSource } from "../src/data/tokens.ts";
import { initWebGPU } from "../src/backend/webgpu.ts";
import type { WebGPUBackend } from "../src/backend/webgpu.ts";
import { MuonGpu } from "../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";

const DOC_SEP = "<|endoftext|>"; // TinyStories and most raw dumps mark doc boundaries with this
const VOCAB = 16384; // shared curriculum vocab (caps below this on a small sample)
const TRAIN_SAMPLE_MB = 12; // BPE converges on a few MB; no need to scan the whole corpus

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  return g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
}
function die(msg: string): never {
  console.error("pretrain: " + msg);
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
async function fileExists(path: string): Promise<boolean> {
  try {
    await readFileText(path);
    return true;
  } catch {
    return false;
  }
}

/** Greedy continuation with the GPU forward (one sync/token); stops at eos. */
async function generateGpu(
  gpu: WebGPUBackend,
  model: Gemma3Model,
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
      const next = argmax(logits.data.subarray(base, base + V));
      ids.push(next);
      if (next === tok.eosId) break;
    }
  } finally {
    gpu.uninstall();
  }
  return tok.decode(ids);
}

/** Train the shared tokenizer (stage 0) if absent, else load it. Reserves the
 * full curriculum special set so every later stage can use its tokens. */
async function sharedTokenizer(path: string, corpus: string): Promise<BPETokenizer> {
  if (await fileExists(path)) {
    const data = JSON.parse(await readFileText(path)) as TokenizerData;
    const tok = BPETokenizer.fromData(data);
    console.log(`Tokenizer: reused ${path} (vocab ${tok.vocabSize}, specials ${tok.specials})`);
    return tok;
  }
  // Strip doc markers from the sample so "<|endoftext|>" pieces don't pollute merges.
  const sample = corpus.slice(0, TRAIN_SAMPLE_MB * 1024 * 1024).split(DOC_SEP).join("\n");
  const tok = new BPETokenizer();
  tok.train(sample, VOCAB, CURRICULUM_SPECIALS);
  if (tokenBytes(tok.vocabSize) !== 2) die(`vocab ${tok.vocabSize} exceeds u16; widen the loader`);
  await writeFileBytes(path, new TextEncoder().encode(JSON.stringify(tok.export())));
  console.log(
    `Tokenizer: trained to ${tok.vocabSize} tokens on ${(sample.length / 1e6).toFixed(1)}M-char ` +
      `sample, ${CURRICULUM_SPECIALS.length} curriculum specials reserved -> ${path}`,
  );
  return tok;
}

/** Pretokenize the whole corpus into a growable Uint16Array (a number[] would
 * hit V8's max-array-length near 10^8), doc-by-doc with eos between docs. */
function encodeCorpus(tok: BPETokenizer, corpus: string): Uint16Array {
  const docs = corpus.split(DOC_SEP).map((d) => d.trim()).filter((d) => d.length > 0);
  let cap = 1 << 20, n = 0;
  let ids = new Uint16Array(cap);
  const push = (id: number) => {
    if (n >= cap) {
      cap *= 2;
      const grown = new Uint16Array(cap);
      grown.set(ids);
      ids = grown;
    }
    ids[n++] = id;
  };
  for (const doc of docs) {
    for (const id of tok.encode(doc)) push(id);
    push(tok.eosId);
  }
  console.log(`Pretokenized: ${docs.length} docs -> ${n} tokens`);
  return ids.subarray(0, n);
}

async function main() {
  const a = args();
  const dir = new URL(".", import.meta.url).pathname;
  const corpusPath = a[0] ?? `${dir}../corpus/tinystories-valid.txt`;
  const hidden = a[1] ? Number(a[1]) : 512;
  const layers = a[2] ? Number(a[2]) : 6;
  const steps = a[3] ? Number(a[3]) : 2000;
  const seqLen = a[4] ? Number(a[4]) : 1024;
  const batch = a[5] ? Number(a[5]) : 8;
  const muonLr = 0.02, auxLr = 3e-3, baseWidth = 128;

  console.log("=== curriculum stage 1: pretrain gemma3 base -> GGUF ===\n");
  const gpu = await initWebGPU();
  if (!gpu) die("no WebGPU (run under Deno with --unstable-webgpu)");
  console.log(`WebGPU adapter: ${gpu.adapterName}`);
  // deno-lint-ignore no-explicit-any
  const wantF16 = (globalThis as any).Deno?.env?.get?.("GGUF_F16") === "1";
  const precision: "f16" | "f32" = wantF16 && gpu.f16Supported ? "f16" : "f32";
  console.log(`Precision: ${precision}`);

  const corpus = await readFileText(corpusPath).catch(() =>
    die(`cannot read corpus ${corpusPath}`)
  );
  if (corpus.length === 0) die(`${corpusPath} is empty`);
  console.log(`Corpus: ${corpusPath} (${(corpus.length / 1e6).toFixed(1)}M chars)`);

  // Stage 0: the shared tokenizer (reused across all curriculum stages).
  const tokPath = `${dir}curriculum.tokenizer.json`;
  const tok = await sharedTokenizer(tokPath, corpus);
  const bpt = tokenBytes(tok.vocabSize);

  // Pretokenize (skip if a .tokens for this run already exists).
  const tokensPath = `${dir}pretrain.tokens`;
  if (!(await fileExists(tokensPath))) {
    await writeTokenFile(tokensPath, encodeCorpus(tok, corpus), bpt);
  }
  const src: TokenSource = await diskTokenSource(tokensPath, bpt);
  const tokensPerStep = batch * seqLen;
  console.log(
    `Tokens: ${
      (src.length / 1e6).toFixed(1)
    }M (${bpt}B), eos=${tok.eosId}; ${steps} x ${batch} x ` +
      `${seqLen} = ${(steps * tokensPerStep / 1e6).toFixed(0)}M tokens ` +
      `(${(steps * tokensPerStep / src.length).toFixed(1)} epochs)`,
  );

  const cfg = gemma3Config(tok.vocabSize, hidden, layers, Math.max(4096, seqLen));
  if (seqLen > cfg.maxSeq) die(`seqLen ${seqLen} exceeds context_length ${cfg.maxSeq}`);
  const model = new Gemma3Model(
    cfg,
    mulberry32(1234),
    baseWidth === hidden ? undefined : { baseWidth },
  );
  console.log(
    `Model: gemma3 base, ${cfg.nLayers} layers, hidden=${cfg.hiddenSize}, heads=${cfg.nHeads}/` +
      `${cfg.nKVHeads}, ffn=${cfg.ffnDim}, window=${cfg.slidingWindow}, ~${
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
    warmupSteps: Math.max(1, Math.round(steps * 0.1)),
    stableSteps: Math.max(0, steps - Math.round(steps * 0.1) - Math.round(steps * 0.2)),
    cooldownSteps: Math.max(1, Math.round(steps * 0.2)),
    minScale: 0.1,
  });

  // BASE model export: no chat template (there are no turns yet). The tokenizer
  // still carries the reserved specials, so the instruct stage inherits them.
  const outPath = `${dir}pretrain-base-f16.gguf`;
  const exportGGUF = async (): Promise<Uint8Array> => {
    const b = buildGemma3GGUF(model, tok.export(), cfg, { quant: "f16", name: "pretrain-base" });
    await writeFileBytes(outPath, b);
    return b;
  };
  const checkpointEvery = Math.min(500, Math.max(1, Math.round(steps / 20)));

  let firstLoss = 0, lastLoss = 0;
  const t0 = Date.now();
  await trainLMGpuResident(model, gpu, {
    tokens: src,
    seqLen,
    steps,
    batchPerStep: batch,
    optimizer: opt,
    schedule,
    precision,
    logEvery: Math.max(1, Math.round(steps / 50)),
    rng: mulberry32(7),
    checkpointEvery,
    onCheckpoint: async (step) => {
      const b = await exportGGUF();
      const el = (Date.now() - t0) / 1000;
      console.log(
        `  [ckpt @ ${step}] ${outPath.split("/").pop()} ${(b.length / 1e6).toFixed(0)}MB, ` +
          `loss ${lastLoss.toFixed(3)}, ${(step / Math.max(1, el)).toFixed(2)} st/s`,
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
  });
  src.close();
  console.log(
    `\nTraining: loss ${firstLoss.toFixed(3)} -> ${lastLoss.toFixed(3)} in ${
      ((Date.now() - t0) / 60000).toFixed(1)
    }min`,
  );

  console.log(`\nSample:\n${await generateGpu(gpu, model, tok, "Once upon a time", 60)}`);

  const bytes = await exportGGUF();
  const g = readGGUF(bytes);
  if (g.metadata.get("general.architecture") !== "gemma3") die("exported arch != gemma3");
  console.log(
    `\nWrote ${outPath} (${(bytes.length / 1e6).toFixed(0)} MB, ${g.tensors.length} tensors, ` +
      `gemma3 base ✓). Next stage resumes from this via loadGemma3FromGGUF.`,
  );
  console.log("=== stage 1 complete ===");
}

main().catch((e) => die(String(e?.stack ?? e)));
