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
// document boundary.
//
// TWO input modes:
//   1. A raw .txt corpus (≤~480 MB): trains the tokenizer + pretokenizes inline.
//      This is the TinyStories / single-file path.
//   2. A pretokenized .tokens file (from pretokenize.ts): the tokenizer is the
//      sibling <prefix>.tokenizer.json. This is the path for the multi-GB blend
//      corpus, which cannot be read whole (V8's ~512 MB single-string cap).
//
//   deno run -A --unstable-webgpu examples/pretrain.ts <corpus.txt|tokens.tokens> \
//       [hidden] [layers] [steps] [seqLen] [batch] [muonLr] [--flags]
//
//   flags (all optional):
//     --maxSeq=N       declared context_length (default max(8192, seqLen))
//     --window=N       SWA sliding-window size (default 1024, Gemma3's shape)
//     --headDim=N      attention head dim (default 64)
//     --resume=PATH    load weights from a prior GGUF (long-context phase B, or
//                      crash recovery); config must match the built model
//     --startStep=N    resume the WSD schedule + step counter at N (crash
//                      recovery mid-phase; steps stays the FULL run length)
//     --ckpt=N         checkpoint every N steps (default min(1000, steps/20))
//     --quant=f32|f16  checkpoint storage precision (default f32 — lossless
//                      resume; f16 halves disk at a tiny rounding cost)
//     --auxLr=F        AdamW lr for the aux (norm/embed) group (default 3e-3)
//     --out=PATH       output GGUF path (default examples/pretrain-base.gguf)
//     --name=STR       general.name in the GGUF (default "pretrain-base")
//
// Defaults: hidden 512, layers 6 (~28M), steps 2000, seqLen 1024, batch 8,
//   muonLr 0.01. Compute is f32 throughout.
//
// muonLr 0.01 is the proven stable peak (the 20k-step f32 run trained clean to
// loss 1.18 at 0.01; 0.02 diverged). Keep 0.01 for unattended long runs.
//
// NOTE: in .txt mode the corpus is read whole (readFileText), so keep a single
// file under ~480 MB. For a bigger corpus, build it with prepare_pretrain.ts as
// parts, run pretokenize.ts once, and feed the resulting .tokens file here.

import { readGGUF } from "../src/gguf/gguf.ts";
import { readFileBytes, readFileText, writeFileBytes } from "../src/io.ts";
import { crossEntropy, mulberry32 } from "../src/model/autograd.ts";
import { gemma3Config, gemma3ParamCount } from "../src/model/config.ts";
import type { Gemma3Config } from "../src/model/config.ts";
import { Gemma3Model } from "../src/model/gemma3.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import type { TokenizerData } from "../src/tokenizer/bpe.ts";
import { CURRICULUM_SPECIALS } from "../src/data/chat.ts";
import { buildGemma3GGUF } from "../src/export/export_gguf.ts";
import { configFromGGUF, loadWeightsFromGGUF } from "../src/export/load_gguf.ts";
import { wsdSchedule } from "../src/train/schedule.ts";
import { diskTokenSource, tokenBytes, writeTokenFile } from "../src/data/tokens.ts";
import type { TokenSource } from "../src/data/tokens.ts";
import type { QuantName } from "../src/gguf/quantize.ts";
import { initWebGPU } from "../src/backend/webgpu.ts";
import type { WebGPUBackend } from "../src/backend/webgpu.ts";
import { MuonGpu } from "../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";

const DOC_SEP = "<|endoftext|>"; // TinyStories and most raw dumps mark doc boundaries with this
const VOCAB = 16384; // .txt-mode shared vocab (caps below this on a small sample)
const TRAIN_SAMPLE_MB = 12; // BPE converges on a few MB; no need to scan the whole corpus

interface Flags {
  positional: string[];
  get(name: string): string | undefined;
  has(name: string): boolean;
}
function parseArgs(): Flags {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  const raw: string[] = g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
  const positional: string[] = [];
  const flags = new Map<string, string>();
  for (const a of raw) {
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      if (eq < 0) flags.set(a.slice(2), "");
      else flags.set(a.slice(2, eq), a.slice(eq + 1));
    } else positional.push(a);
  }
  return {
    positional,
    get: (n) => flags.get(n),
    has: (n) => flags.has(n),
  };
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

/** Load the tokenizer that pretokenize.ts wrote next to a .tokens file. */
async function siblingTokenizer(tokensPath: string): Promise<BPETokenizer> {
  const jsonPath = tokensPath.replace(/\.tokens$/, ".tokenizer.json");
  if (!(await fileExists(jsonPath))) {
    die(`no sibling tokenizer ${jsonPath} for ${tokensPath} (run pretokenize.ts first)`);
  }
  const data = JSON.parse(await readFileText(jsonPath)) as TokenizerData;
  const tok = BPETokenizer.fromData(data);
  console.log(`Tokenizer: loaded ${jsonPath} (vocab ${tok.vocabSize}, specials ${tok.specials})`);
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

/** True if two configs describe the same architecture (so a GGUF resume is valid). */
function configMatches(a: Gemma3Config, b: Gemma3Config): string | null {
  const keys: (keyof Gemma3Config)[] = [
    "vocabSize",
    "hiddenSize",
    "nLayers",
    "nHeads",
    "nKVHeads",
    "headDim",
    "ffnDim",
    "slidingWindow",
    "swaPattern",
  ];
  for (const k of keys) if (a[k] !== b[k]) return `${k}: built ${a[k]} vs checkpoint ${b[k]}`;
  return null;
}

async function main() {
  const flags = parseArgs();
  const a = flags.positional;
  const dir = new URL(".", import.meta.url).pathname;
  const inputPath = a[0] ?? `${dir}../corpus/tinystories-valid.txt`;
  const hidden = a[1] ? Number(a[1]) : 512;
  const layers = a[2] ? Number(a[2]) : 6;
  const steps = a[3] ? Number(a[3]) : 2000;
  const seqLen = a[4] ? Number(a[4]) : 1024;
  const batch = a[5] ? Number(a[5]) : 8;
  const muonLr = a[6] ? Number(a[6]) : 0.01;
  const auxLr = flags.get("auxLr") ? Number(flags.get("auxLr")) : 3e-3;
  const baseWidth = 128;
  const maxSeq = flags.get("maxSeq") ? Number(flags.get("maxSeq")) : Math.max(8192, seqLen);
  const window = flags.get("window") ? Number(flags.get("window")) : 1024;
  const headDim = flags.get("headDim") ? Number(flags.get("headDim")) : 64;
  const startStep = flags.get("startStep") ? Number(flags.get("startStep")) : 0;
  const quant = (flags.get("quant") ?? "f32") as QuantName;
  const resumePath = flags.get("resume");
  const outPath = flags.get("out") ?? `${dir}pretrain-base.gguf`;
  const name = flags.get("name") ?? "pretrain-base";
  if (quant !== "f32" && quant !== "f16") die(`--quant must be f32 or f16, got ${quant}`);
  if (startStep < 0 || startStep >= steps) die(`--startStep must be in [0, ${steps})`);

  console.log("=== curriculum stage 1: pretrain gemma3 base -> GGUF ===\n");
  const gpu = await initWebGPU();
  if (!gpu) die("no WebGPU (run under Deno with --unstable-webgpu)");
  console.log(`WebGPU adapter: ${gpu.adapterName}`);

  // --- Tokens + tokenizer: pretokenized (.tokens) or raw (.txt) input ---
  let tok: BPETokenizer;
  let src: TokenSource;
  if (inputPath.endsWith(".tokens")) {
    tok = await siblingTokenizer(inputPath);
    src = await diskTokenSource(inputPath, tokenBytes(tok.vocabSize));
    console.log(`Tokens: ${inputPath} (${(src.length / 1e6).toFixed(1)}M, pretokenized)`);
  } else {
    const corpus = await readFileText(inputPath).catch(() =>
      die(`cannot read corpus ${inputPath}`)
    );
    if (corpus.length === 0) die(`${inputPath} is empty`);
    console.log(`Corpus: ${inputPath} (${(corpus.length / 1e6).toFixed(1)}M chars)`);
    tok = await sharedTokenizer(`${dir}curriculum.tokenizer.json`, corpus);
    const bpt = tokenBytes(tok.vocabSize);
    const tokensPath = `${dir}pretrain.tokens`;
    if (!(await fileExists(tokensPath))) {
      await writeTokenFile(tokensPath, encodeCorpus(tok, corpus), bpt);
    }
    src = await diskTokenSource(tokensPath, bpt);
  }
  const tokensPerStep = batch * seqLen;
  console.log(
    `Run: ${steps} x ${batch} x ${seqLen} = ${(steps * tokensPerStep / 1e6).toFixed(0)}M tokens ` +
      `(${(steps * tokensPerStep / src.length).toFixed(1)} epochs of ${
        (src.length / 1e6).toFixed(0)
      }M), eos=${tok.eosId}` + (startStep ? `, resuming @ step ${startStep}` : ""),
  );

  const cfg = gemma3Config(tok.vocabSize, hidden, layers, maxSeq, headDim, window);
  if (seqLen > cfg.maxSeq) die(`seqLen ${seqLen} exceeds context_length ${cfg.maxSeq}`);
  const model = new Gemma3Model(
    cfg,
    mulberry32(1234),
    baseWidth === hidden ? undefined : { baseWidth },
  );
  console.log(
    `Model: gemma3 base, ${cfg.nLayers} layers, hidden=${cfg.hiddenSize}, heads=${cfg.nHeads}/` +
      `${cfg.nKVHeads}, headDim=${cfg.headDim}, ffn=${cfg.ffnDim}, ctx=${cfg.maxSeq}, ` +
      `window=${cfg.slidingWindow}, ~${(gemma3ParamCount(cfg) / 1e6).toFixed(1)}M params`,
  );

  // Resume weights from a prior GGUF (long-context phase, or crash recovery).
  if (resumePath) {
    const g = readGGUF(await readFileBytes(resumePath));
    const mismatch = configMatches(cfg, configFromGGUF(g));
    if (mismatch) die(`--resume config mismatch (${mismatch}); build the same shape as the ckpt`);
    loadWeightsFromGGUF(model, g);
    console.log(`Resumed weights from ${resumePath} (${g.tensors.length} tensors)`);
  }

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
  // WSD over the FULL run (0..steps); on resume we offset into it so the LR
  // continues rather than re-warming. 10% warmup / 20% cooldown, floor 0.1.
  const fullSchedule = wsdSchedule({
    warmupSteps: Math.max(1, Math.round(steps * 0.1)),
    stableSteps: Math.max(0, steps - Math.round(steps * 0.1) - Math.round(steps * 0.2)),
    cooldownSteps: Math.max(1, Math.round(steps * 0.2)),
    minScale: 0.1,
  });
  const schedule = (localStep: number) => fullSchedule(startStep + localStep);
  console.log(
    `Schedule: muon lr ${muonLr}, aux lr ${auxLr}, WSD warmup ${
      Math.round(steps * 0.1)
    } / cooldown ${Math.round(steps * 0.2)} steps, quant ${quant}`,
  );

  // WSD decay-phase instruct injection (MiniCPM/Xmodel-2 trick): from the cooldown
  // start, draw injectFrac of micro-batches from a second .tokens stream (instruct
  // data encoded with THIS tokenizer). Off unless --inject is given.
  const injectPath = flags.get("inject");
  let injectSource: TokenSource | undefined;
  const injectFrac = flags.get("injectFrac") ? Number(flags.get("injectFrac")) : 0;
  const cooldownStart = steps - Math.round(steps * 0.2);
  const injectFrom = flags.get("injectFrom") ? Number(flags.get("injectFrom")) : cooldownStart;
  if (injectPath) {
    injectSource = await diskTokenSource(injectPath, tokenBytes(tok.vocabSize));
    console.log(
      `Inject: ${injectPath} (${(injectSource.length / 1e6).toFixed(1)}M tokens), ` +
        `frac ${injectFrac} from step ${injectFrom}`,
    );
  }

  // BASE model export: no chat template (there are no turns yet). The tokenizer
  // still carries the reserved specials, so the instruct stage inherits them.
  // Write atomically (tmp + rename) so a crash mid-write can't corrupt the
  // checkpoint a multi-week run resumes from.
  const exportGGUF = async (): Promise<Uint8Array> => {
    const b = buildGemma3GGUF(model, tok.export(), cfg, { quant, name });
    await writeFileBytes(`${outPath}.tmp`, b);
    const fs = await import("node:fs");
    fs.renameSync(`${outPath}.tmp`, outPath);
    return b;
  };
  const ckptEvery = flags.get("ckpt")
    ? Number(flags.get("ckpt"))
    : Math.min(1000, Math.max(1, Math.round(steps / 20)));

  let firstLoss = 0, lastLoss = 0;
  const t0 = Date.now();
  await trainLMGpuResident(model, gpu, {
    tokens: src,
    seqLen,
    steps: steps - startStep,
    batchPerStep: batch,
    optimizer: opt,
    schedule,
    injectSource,
    injectFrac,
    injectFromStep: Math.max(0, injectFrom - startStep), // trainer step is local to this segment
    logEvery: Math.max(1, Math.round(steps / 100)),
    rng: mulberry32(7 + startStep), // vary batches across resume segments
    checkpointEvery: ckptEvery,
    onCheckpoint: async (localStep) => {
      const b = await exportGGUF();
      const step = startStep + localStep;
      const el = (Date.now() - t0) / 1000;
      console.log(
        `  [ckpt @ ${step}] ${outPath.split("/").pop()} ${(b.length / 1e6).toFixed(0)}MB, ` +
          `loss ${lastLoss.toFixed(3)}, ${(localStep / Math.max(1, el)).toFixed(3)} st/s`,
      );
    },
    onLog: (localStep, loss) => {
      const step = startStep + localStep;
      if (localStep === 0) firstLoss = loss;
      lastLoss = loss;
      const el = (Date.now() - t0) / 1000;
      console.log(
        `  step ${String(step).padStart(7)}  loss ${loss.toFixed(4)}  (${
          (localStep / Math.max(1, el)).toFixed(3)
        } st/s, ${(el / 60).toFixed(1)}min)`,
      );
    },
  });
  src.close();
  injectSource?.close();
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
      `gemma3 base ✓, ctx ${cfg.maxSeq}). Next stage resumes via loadGemma3FromGGUF.`,
  );
  console.log("=== stage 1 complete ===");
}

main().catch((e) => die(String(e?.stack ?? e)));
