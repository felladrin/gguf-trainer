// `pretrain` and `finetune`: the training loop, device-resident on WebGPU.
//
// One implementation, two commands, because they are the same run with a
// different loss mask:
//
//   pretrain  full-sequence next-token loss on unlabeled text. Trains a base
//             model from scratch, or continues one with --resume.
//   finetune  the same loop with --mask, so only the assistant turns are
//             supervised, plus a chat template embedded in the export.
//
// Two input modes, picked from the file extension:
//   *.tokens  pretokenized (see `tokenize`), with its sibling <name>.tokenizer.json
//   *.txt     a raw corpus: the tokenizer is trained inline (one file, under
//             ~480 MB, because V8 caps a single string at ~512 MB)
//
// The vocab and embedding matrix freeze the moment pretraining starts, so every
// special token a later stage needs is reserved up front (CURRICULUM_SPECIALS:
// ChatML turns, <think>, the tool tags). Pretraining never emits them; their
// rows sit at init until a fine-tune first uses them.

import { readGGUF } from "../gguf/gguf.ts";
import { greedyComplete, SAMPLE_PRESET } from "../eval/generate.ts";
import { lossTrend } from "../loss-trend.ts";
import { readFileBytes, readFileText, writeFileBytes } from "../io.ts";
import { fmtEta } from "../eta.ts";
import { crossEntropy, mulberry32 } from "../model/autograd.ts";
import type { Architecture, LanguageModel } from "../model/arch.ts";
import {
  archFromGGUF,
  archNames,
  assertFlagsBelongTo,
  DEFAULT_ARCH,
  getArch,
  mergedArchFlags,
} from "../model/registry.ts";
import { BPETokenizer } from "../tokenizer/bpe.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";
import { CURRICULUM_SPECIALS } from "../data/chat.ts";
import { llamaRunScript } from "../export/export-gguf.ts";
import { wsdSchedule } from "../train/schedule.ts";
import { diskTokenSource, idArrayFor, tokenBytes, writeTokenFile } from "../data/tokens.ts";
import type { IdArray } from "../data/tokens.ts";
import type { TokenSource } from "../data/tokens.ts";
import { parseQuantList } from "../gguf/quantize.ts";
import type { QuantName } from "../gguf/quantize.ts";
import { initWebGPU } from "../backend/webgpu.ts";
import type { WebGPUBackend } from "../backend/webgpu.ts";
import { deserializeOptState, MuonGpu, serializeOptState } from "../backend/muon-gpu.ts";
import { trainLMGpuResident } from "../backend/train-gpu.ts";
import type { Command, Flag, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";

const DOC_SEP = "<|endoftext|>"; // TinyStories and most raw dumps mark doc boundaries with this
const VOCAB = 16384; // .txt-mode shared vocab (caps below this on a small sample)
const TRAIN_SAMPLE_MB = 12; // BPE converges on a few MB; no need to scan the whole corpus

function die(msg: string): never {
  throw new UsageError(msg);
}
async function fileExists(path: string): Promise<boolean> {
  // Stat, don't read: the optimizer-state sidecar is a multi-hundred-MB binary
  // blob, and decoding it as UTF-8 to test existence overflows V8's max string
  // length and throws, which silently reported "no optstate" and cold-started
  // the optimizer on every resume.
  const fs = await import("node:fs");
  return fs.existsSync(path);
}

/**
 * The first two prompts of scripts/eval-completions.sh, so the glance at the end
 * of a run and the offline battery cannot disagree.
 *
 * Two rather than one, because a single greedy prompt is a coin flip: SAMPLE_PRESET
 * cannot break an alternating loop (each turn of the cycle refreshes the penalty
 * window), and the roleplay run printed "The world was in turmoil. The world was in
 * peace." on "Once upon a time" while all ten prompts of the battery were clean.
 */
const SAMPLE_PROMPTS = [
  "Once upon a time, there was a little",
  "The old man walked slowly toward the",
];

/**
 * End-of-run samples with the GPU forward (one sync/token); each stops at eos.
 *
 * Uses SAMPLE_PRESET rather than bare greedy: these lines are read as a quality
 * signal, and an unpenalized base model loops on one sentence for all 60 tokens.
 */
async function generateGpu(
  gpu: WebGPUBackend,
  model: LanguageModel,
  tok: BPETokenizer,
  prompts: string[],
  n: number,
): Promise<string> {
  gpu.install();
  try {
    gpu.uploadParams(model.params());
    const out: string[] = [];
    for (const prompt of prompts) {
      const ids = await greedyComplete(
        model,
        gpu,
        tok.encode(prompt),
        n,
        [tok.eosId],
        SAMPLE_PRESET,
      );
      out.push(tok.decode(ids));
    }
    return out.join("\n\n");
  } finally {
    gpu.uninstall();
  }
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
  // We train this vocab ourselves, so this can only trip on a bad VOCAB constant.
  if (tokenBytes(tok.vocabSize) !== 2) die(`vocab ${tok.vocabSize} exceeds u16; lower VOCAB`);
  await writeFileBytes(path, new TextEncoder().encode(JSON.stringify(tok.export())));
  console.log(
    `Tokenizer: trained to ${tok.vocabSize} tokens on ${(sample.length / 1e6).toFixed(1)}M-char ` +
      `sample, ${CURRICULUM_SPECIALS.length} curriculum specials reserved -> ${path}`,
  );
  return tok;
}

/** Load the tokenizer that `tokenize` wrote next to a .tokens file. */
async function siblingTokenizer(tokensPath: string): Promise<BPETokenizer> {
  const jsonPath = tokensPath.replace(/\.tokens$/, ".tokenizer.json");
  if (!(await fileExists(jsonPath))) {
    die(`no sibling tokenizer ${jsonPath} for ${tokensPath} (run \`tokenize\` first)`);
  }
  const data = JSON.parse(await readFileText(jsonPath)) as TokenizerData;
  const tok = BPETokenizer.fromData(data);
  console.log(`Tokenizer: loaded ${jsonPath} (vocab ${tok.vocabSize}, specials ${tok.specials})`);
  return tok;
}

/** Pretokenize the whole corpus into a growable typed array (a number[] would
 * hit V8's max-array-length near 10^8), doc-by-doc with eos between docs.
 *
 * The width follows the vocab: u16 for a vocab trained here, u32 for a resumed
 * foreign one. Qwen3 is 151,936 tokens and Llama-3 128,256, so a fixed u16 buffer
 * would wrap their ids silently rather than fail. */
export function encodeCorpus(tok: BPETokenizer, corpus: string): IdArray {
  const docs = corpus.split(DOC_SEP).map((d) => d.trim()).filter((d) => d.length > 0);
  const IdBuffer = idArrayFor(tok.vocabSize);
  let cap = 1 << 20, n = 0;
  let ids: IdArray = new IdBuffer(cap);
  const push = (id: number) => {
    if (n >= cap) {
      cap *= 2;
      const grown = new IdBuffer(cap);
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

async function run(v: Values, mode: "pretrain" | "finetune") {
  const inputPath = v.str("data");
  const resumeFlag = v.opt("resume");
  // A checkpoint knows its own architecture, so --resume decides and --arch only
  // has to agree. Without a checkpoint, --arch decides.
  // deno-lint-ignore no-explicit-any
  let arch: Architecture<any>;
  if (resumeFlag) {
    const head = readGGUF(
      await readFileBytes(resumeFlag).catch(() => {
        throw new UsageError(`cannot read --resume ${resumeFlag}`);
      }),
    );
    arch = archFromGGUF(head);
    if (v.given("arch") && v.str("arch") !== arch.name) {
      die(`--arch ${v.str("arch")} but ${resumeFlag} is a ${arch.name} checkpoint`);
    }
  } else {
    arch = getArch(v.str("arch"));
  }
  assertFlagsBelongTo(arch.name, v);
  const hidden = v.num("hidden");
  const layers = v.num("layers");
  const steps = v.num("steps");
  const seqLen = v.num("seq-len");
  const batch = v.num("batch");
  const muonLr = v.num("lr");
  const auxLr = v.num("aux-lr");
  const baseWidth = 128;
  const maxSeq = v.has("max-seq") ? v.num("max-seq") : Math.max(8192, seqLen);
  const headDim = v.num("head-dim");
  const startStep = v.num("start-step");
  const quant = v.str("checkpoint-precision") as QuantName;
  const resumePath = v.opt("resume");
  const outPath = v.str("out");
  const name = v.opt("name") ?? (mode === "finetune" ? "finetune" : "pretrain-base");
  const maskPath = v.opt("mask");
  const templatePath = v.opt("template");
  // A fine-tune inherits weights and starts a fresh optimizer: momentum from the
  // previous objective is not a warm start, it is a pull back toward it.
  const coldOpt = v.has("cold-optimizer") ? v.bool("cold-optimizer") : mode === "finetune";
  if (mode === "finetune" && !resumePath) {
    die("finetune needs --resume <base.gguf>: it fine-tunes an existing checkpoint");
  }
  if (mode === "finetune" && !maskPath) {
    die("finetune needs --mask <corpus.mask> from `chat-corpus` (assistant-only loss)");
  }
  if (startStep < 0 || startStep >= steps) die(`--start-step must be in [0, ${steps})`);
  const flags = {
    get: (k: string): string | undefined =>
      ({
        exportQuants: v.opt("export-quants"),
        inject: v.opt("inject"),
        injectFrac: v.has("inject-fraction") ? String(v.num("inject-fraction")) : undefined,
        injectFrom: v.has("inject-from-step") ? String(v.num("inject-from-step")) : undefined,
        ckpt: v.has("checkpoint-every") ? String(v.num("checkpoint-every")) : undefined,
        mask: maskPath,
        template: templatePath,
      })[k],
    has: (k: string): boolean => ({ reclaim: v.bool("reclaim"), coldOpt })[k] ?? false,
  };
  const exportQuants = flags.get("exportQuants")
    ? (() => {
      try {
        return parseQuantList(flags.get("exportQuants")!);
      } catch (e) {
        return die(String((e as Error).message));
      }
    })()
    : [];

  console.log(`=== ${mode}: ${arch.name} -> GGUF ===\n`);
  const gpu = await initWebGPU();
  if (!gpu) die("no WebGPU: training needs Deno (Node and Bun have no GPU backend here)");
  console.log("Device:");
  for (const line of gpu.describeDevice()) console.log(line);

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
    const stem = inputPath.replace(/\.[a-z]+$/i, "");
    tok = await sharedTokenizer(`${stem}.tokenizer.json`, corpus);
    const bpt = tokenBytes(tok.vocabSize);
    const tokensPath = `${stem}.tokens`;
    if (!(await fileExists(tokensPath))) {
      await writeTokenFile(tokensPath, encodeCorpus(tok, corpus), bpt);
    }
    src = await diskTokenSource(tokensPath, bpt);
  }
  // Instruct/SFT stage (--mask): supervise only the assistant turns, using the
  // mask `chat-corpus` wrote beside the .tokens file. Without it every
  // token is a target, which is right for pretraining and wrong for chat.
  let supervised: TokenSource | undefined;
  if (maskPath) {
    supervised = await diskTokenSource(maskPath, tokenBytes(tok.vocabSize));
    if (supervised.length !== src.length) {
      die(`mask ${maskPath} has ${supervised.length} tokens, corpus has ${src.length}`);
    }
    console.log(`Mask: ${maskPath} (assistant-only loss over ${src.length} tokens)`);
  }

  const tokensPerStep = batch * seqLen;
  console.log(
    `Run: ${steps} x ${batch} x ${seqLen} = ${(steps * tokensPerStep / 1e6).toFixed(0)}M tokens ` +
      `(${(steps * tokensPerStep / src.length).toFixed(1)} epochs of ${
        (src.length / 1e6).toFixed(0)
      }M), eos=${tok.eosId}` + (startStep ? `, resuming @ step ${startStep}` : ""),
  );

  const cfg = arch.configFromFlags(
    { vocabSize: tok.vocabSize, hiddenSize: hidden, nLayers: layers, maxSeq, headDim },
    v,
  );
  if (seqLen > cfg.maxSeq) die(`seqLen ${seqLen} exceeds context_length ${cfg.maxSeq}`);
  const model = arch.build(cfg, mulberry32(1234), baseWidth === hidden ? undefined : { baseWidth });
  console.log(
    `Model: ${arch.describe(cfg)}, ~${(arch.paramCount(cfg) / 1e6).toFixed(1)}M params`,
  );

  // Resume weights from a prior GGUF (long-context phase, or crash recovery).
  if (resumePath) {
    const g = readGGUF(await readFileBytes(resumePath));
    const mismatch = arch.configMatches(cfg, arch.configFromGGUF(g));
    if (mismatch) die(`--resume config mismatch (${mismatch}); build the same shape as the ckpt`);
    arch.loadWeights(model, g);
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
  // Restore optimizer state (Muon momentum + Adam moments + step) if a sidecar
  // sits next to the resume checkpoint: a warm resume instead of a cold restart.
  // Absent (e.g. a weights-only checkpoint) -> cold optimizer, exactly as before.
  if (resumePath && !flags.has("coldOpt")) {
    const optPath = `${resumePath}.optstate`;
    if (await fileExists(optPath)) {
      opt.importState(deserializeOptState(await readFileBytes(optPath)));
      console.log(`Resumed optimizer state from ${optPath}`);
    } else {
      console.log(`No optimizer state at ${optPath}; optimizer cold-starts (momentum re-warms)`);
    }
  } else if (resumePath) {
    console.log(`--coldOpt: ignoring any optimizer state (momentum from another objective)`);
  }
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
  // An SFT stage passes --template (the .template.txt `chat-corpus` wrote
  // beside its corpus), which embeds that Jinja template in the GGUF so
  // llama.cpp renders turns exactly the way the model was trained on them.
  // Write atomically (tmp + rename) so a crash mid-write can't corrupt the
  // checkpoint a multi-week run resumes from.
  const chatTemplate = templatePath
    ? await readFileText(templatePath).catch(() => die(`cannot read --template ${templatePath}`))
    : undefined;
  if (chatTemplate) console.log(`Chat template: ${templatePath} (embedded in every export)`);
  const exportGGUF = async (): Promise<Uint8Array> => {
    const b = arch.exportGGUF(model, tok.export(), cfg, { quant, name, chatTemplate });
    await writeFileBytes(`${outPath}.tmp`, b);
    const fs = await import("node:fs");
    fs.renameSync(`${outPath}.tmp`, outPath);
    return b;
  };
  // Optimizer-state sidecar next to the GGUF, so a resume continues with a warm
  // optimizer (Muon momentum + Adam moments) instead of cold-starting. Atomic.
  const optStatePath = `${outPath}.optstate`;
  const writeOptState = async (): Promise<number> => {
    const bytes = serializeOptState(await opt.exportState());
    await writeFileBytes(`${optStatePath}.tmp`, bytes);
    (await import("node:fs")).renameSync(`${optStatePath}.tmp`, optStatePath);
    return bytes.length;
  };
  const ckptEvery = flags.get("ckpt")
    ? Number(flags.get("ckpt"))
    : Math.min(1000, Math.max(1, Math.round(steps / 20)));
  // Opt-in wall-clock cadence, so what an interruption costs is bounded in
  // minutes rather than in steps. Both can be on; either one fires a write.
  const ckptEveryMs = v.has("checkpoint-every-minutes")
    ? v.num("checkpoint-every-minutes") * 60_000
    : undefined;
  if (ckptEveryMs !== undefined && !(ckptEveryMs > 0)) {
    die("--checkpoint-every-minutes must be greater than 0");
  }

  // Peak resident GPU storage-buffer memory (pool + optimizer state). Both only
  // grow, so this is the peak, and the headroom-to-ceiling signal for sizing the
  // next run on a fixed-memory box.
  const mem = () => {
    const r = gpu.residentBytes();
    return `${(r.total / 1e6).toFixed(0)}MB gpu (pool ${(r.pool / 1e6).toFixed(0)} + state ${
      (r.state / 1e6).toFixed(0)
    })`;
  };

  const losses: number[] = [];
  let lastLoss = 0;
  const t0 = Date.now();
  await trainLMGpuResident(model, gpu, {
    tokens: src,
    supervised,
    seqLen,
    steps: steps - startStep,
    batchPerStep: batch,
    optimizer: opt,
    schedule,
    injectSource,
    injectFrac,
    injectFromStep: Math.max(0, injectFrom - startStep), // trainer step is local to this segment
    // --reclaim: recycle each micro-batch's activations at the micro-batch
    // boundary so batch>=2 fits at long context (e.g. Phase B seqLen 8192, where
    // it otherwise OOMs). Off unless asked; one GPU fence/micro-batch of overhead.
    reclaimTransients: flags.has("reclaim"),
    logEvery: Math.max(1, Math.round(steps / 100)),
    rng: mulberry32(7 + startStep), // vary batches across resume segments
    checkpointEvery: ckptEvery,
    checkpointEveryMs: ckptEveryMs,
    onCheckpoint: async (localStep) => {
      const b = await exportGGUF();
      const optBytes = await writeOptState();
      const step = startStep + localStep;
      const el = (Date.now() - t0) / 1000;
      const rate = localStep / Math.max(1, el);
      console.log(
        `  [ckpt @ ${step}] ${outPath.split("/").pop()} ${(b.length / 1e6).toFixed(0)}MB ` +
          `+ optstate ${(optBytes / 1e6).toFixed(0)}MB, ` +
          `loss ${lastLoss.toFixed(3)}, ${rate.toFixed(3)} st/s, eta ${
            fmtEta((steps - step) / rate)
          }, mem ${mem()}`,
      );
    },
    onLog: (localStep, loss) => {
      const step = startStep + localStep;
      losses.push(loss);
      lastLoss = loss;
      const el = (Date.now() - t0) / 1000;
      const rate = localStep / Math.max(1, el);
      console.log(
        `  step ${String(step).padStart(7)}  loss ${loss.toFixed(4)}  (${rate.toFixed(3)} st/s, ${
          (el / 60).toFixed(1)
        }min, eta ${fmtEta((steps - step) / rate)})`,
      );
    },
  });
  src.close();
  injectSource?.close();
  {
    const el = (Date.now() - t0) / 1000;
    const localSteps = steps - startStep;
    const tokPerSec = (localSteps * tokensPerStep) / Math.max(1, el);
    const trend = lossTrend(losses);
    const lossPart = trend
      ? `loss ${trend.first.toFixed(3)} -> ${trend.last.toFixed(3)} ` +
        `(mean of first/last ${trend.window})`
      : "no logged losses";
    console.log(
      `\nTraining: ${lossPart} in ${(el / 60).toFixed(1)}min, ` +
        `${tokPerSec.toFixed(0)} tok/s, peak ${mem()}`,
    );
  }

  console.log(`\nSample:\n${await generateGpu(gpu, model, tok, SAMPLE_PROMPTS, 60)}`);

  const bytes = await exportGGUF();
  const optBytes = await writeOptState();
  const g = readGGUF(bytes);
  const wroteArch = g.metadata.get("general.architecture");
  if (wroteArch !== arch.name) die(`exported arch ${wroteArch} != ${arch.name}`);
  console.log(
    `\nWrote ${outPath} (${(bytes.length / 1e6).toFixed(0)} MB, ${g.tensors.length} tensors, ` +
      `${arch.name} ✓, ctx ${cfg.maxSeq}) + ${optStatePath.split("/").pop()} ` +
      `(${(optBytes / 1e6).toFixed(0)} MB). Next stage resumes with --resume.`,
  );

  // Companion run script + optional deployment-quant copies, so the step after
  // a run is copy-paste and the model can be tried at deploy precision without
  // re-training. Enc is reused for the small text write.
  const enc = new TextEncoder();
  const base = outPath.split("/").pop()!;
  const runScriptPath = `${outPath}.run.sh`;
  await writeFileBytes(runScriptPath, enc.encode(llamaRunScript(base, cfg)));
  console.log(`Wrote ${runScriptPath.split("/").pop()} (run with: bash ${runScriptPath})`);
  for (const eq of exportQuants) {
    if (eq === quant) continue; // already the main artifact
    const variantPath = outPath.replace(/\.gguf$/i, `.${eq.toUpperCase()}.gguf`);
    const vb = arch.exportGGUF(model, tok.export(), cfg, { quant: eq, name, chatTemplate });
    await writeFileBytes(variantPath, vb);
    console.log(
      `Wrote ${variantPath.split("/").pop()} (${(vb.length / 1e6).toFixed(0)} MB, ${eq})`,
    );
  }
  console.log(`=== ${mode} complete ===`);
}

const SHARED_FLAGS: Flag[] = [
  {
    name: "arch",
    type: "string",
    default: DEFAULT_ARCH,
    choices: archNames(),
    describe: "model architecture; ignored when --resume gives one. See `archs`",
  },
  {
    name: "data",
    type: "string",
    placeholder: "PATH",
    required: true,
    describe: "training data: a .tokens file from `tokenize`/`chat-corpus`, or a raw .txt corpus",
  },
  {
    name: "out",
    type: "string",
    placeholder: "PATH",
    required: true,
    describe: "where to write the GGUF (checkpoints overwrite it in place)",
  },
  {
    name: "steps",
    type: "number",
    required: true,
    describe: "optimizer steps to run; tokens seen = steps x batch x seq-len",
  },
  { name: "hidden", type: "number", default: 512, describe: "embedding width" },
  { name: "layers", type: "number", default: 6, describe: "transformer blocks" },
  { name: "head-dim", type: "number", default: 64, describe: "attention head dimension" },
  {
    name: "seq-len",
    type: "number",
    default: 1024,
    describe: "tokens per sequence during training",
  },
  {
    name: "max-seq",
    type: "number",
    placeholder: "N",
    describe: "context length declared in the GGUF (default: max(8192, seq-len))",
  },
  {
    name: "batch",
    type: "number",
    default: 8,
    describe: "sequences per step (gradient accumulation)",
  },
  {
    name: "lr",
    type: "number",
    default: 0.01,
    describe: "Muon learning rate for the matmuls; 0.01 is the proven stable peak, 0.02 diverged",
  },
  {
    name: "aux-lr",
    type: "number",
    default: 3e-3,
    describe: "AdamW learning rate for norms and embeddings",
  },
  {
    name: "resume",
    type: "string",
    placeholder: "PATH",
    describe:
      "continue from an existing GGUF checkpoint; its architecture must match the flags above",
  },
  {
    name: "cold-optimizer",
    type: "boolean",
    describe: "ignore the resumed checkpoint's .optstate sidecar and start the optimizer fresh",
  },
  {
    name: "start-step",
    type: "number",
    default: 0,
    describe: "resume the step counter and LR schedule here (crash recovery mid-run)",
  },
  {
    name: "checkpoint-every",
    type: "number",
    placeholder: "N",
    describe: "write the GGUF every N steps (default: min(1000, steps/20))",
  },
  {
    name: "checkpoint-every-minutes",
    type: "number",
    placeholder: "MIN",
    describe: "also write it whenever this many minutes have passed since the last one",
  },
  {
    name: "checkpoint-precision",
    type: "string",
    placeholder: "f32|f16",
    default: "f32",
    choices: ["f32", "f16"],
    describe: "checkpoint storage precision; f32 resumes losslessly",
  },
  {
    name: "export-quants",
    type: "string",
    placeholder: "LIST",
    describe: "also write deployment copies at the final export, e.g. q8_0,q4_0",
  },
  {
    name: "name",
    type: "string",
    placeholder: "STR",
    describe: "general.name in the exported GGUF",
  },
  {
    name: "reclaim",
    type: "boolean",
    describe:
      "free each micro-batch's activations at the micro-batch boundary: 5.6x less peak GPU memory for 23% less throughput (measured), and the way to fit batch>=2 at long context on a small GPU",
  },
];

const INJECT_FLAGS: Flag[] = [
  {
    name: "inject",
    type: "string",
    placeholder: "PATH",
    describe:
      "second .tokens stream to mix in during the LR cooldown (the MiniCPM decay-phase trick)",
  },
  {
    name: "inject-fraction",
    type: "number",
    placeholder: "F",
    describe: "fraction of micro-batches drawn from --inject (0 to 1)",
  },
  {
    name: "inject-from-step",
    type: "number",
    placeholder: "N",
    describe: "step to start injecting at (default: the cooldown start)",
  },
];

export const pretrainCommand: Command = {
  name: "pretrain",
  summary: "Train a base model from scratch, or continue one with --resume.",
  details: `Full-sequence next-token loss on unlabeled text. Every token is a target.

To CONTINUE an existing model (for example the published Minueza-3-95M-Base), pass --resume
and repeat that checkpoint's architecture flags exactly; a mismatch aborts before any
compute with the field that differs. Put the checkpoint's .optstate sidecar next to it and
the optimizer resumes warm.

The run writes its GGUF every --checkpoint-every steps, atomically, so an interrupted run
always leaves a loadable model plus an .optstate to resume from. Add
--checkpoint-every-minutes to write on wall clock as well, which is the cadence to set when
you care about how much work an interruption costs rather than how many steps it costs.`,
  examples: [
    "pretrain --data corpus.tokens --out model.gguf --steps 20000 --hidden 640 --layers 12",
    "pretrain --data more.tokens --out continued.gguf --steps 5000 --hidden 640 --layers 12 --resume Minueza-3-95M-Base.F32.gguf",
  ],
  flags: [...SHARED_FLAGS, ...mergedArchFlags().flags, ...INJECT_FLAGS],
  run: (v) => run(v, "pretrain"),
};

export const finetuneCommand: Command = {
  name: "finetune",
  summary: "Fine-tune a checkpoint on chat data, supervising only the assistant turns.",
  details: `The same training loop as \`pretrain\`, with two differences that make it SFT:
--mask restricts the loss to assistant turns, and --template embeds the chat template in
the exported GGUF so llama.cpp renders turns the way the model was trained on them. Both
files come from \`chat-corpus\`.

--resume is required: fine-tuning starts from a base model. The optimizer cold-starts by
default (--no-cold-optimizer to inherit the base run's momentum instead).

Use a learning rate around a tenth of the pretraining peak: SFT nudges a base model, it
does not reshape it.`,
  examples: [
    "finetune --data chat.tokens --mask chat.mask --template chat.template.txt --resume base.gguf --out instruct.gguf --steps 300 --lr 0.001 --hidden 640 --layers 12",
  ],
  flags: [
    ...SHARED_FLAGS,
    ...mergedArchFlags().flags,
    {
      name: "mask",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "assistant-only supervision mask from `chat-corpus` (this is what makes it SFT)",
    },
    {
      name: "template",
      type: "string",
      placeholder: "PATH",
      describe:
        "chat template to embed in the exported GGUF (the .template.txt from `chat-corpus`)",
    },
  ],
  run: (v) => run(v, "finetune"),
};
