// Held-out validation loss for a pretrain checkpoint: the trend signal the
// per-step training loss can't give you (that's a single noisy batch) and the
// parity probe was never meant to be (it's a 16-token GPU-vs-CPU trust gate).
//
// Scores a GGUF checkpoint on a FIXED, seeded set of windows drawn from a token
// stream: same seed -> same windows every run, so the mean loss is directly
// comparable across checkpoints. Forward-only, on the GPU, using the trainer's
// own model: the exact proven pattern of the pretrain parity probe, so it never
// touches the running trainer. It is safe to run against a checkpoint a live run
// is still writing, because every checkpoint is written atomically.
//
// --holdout scores only the LAST fraction of the stream. Against the training
// corpus that is an in-distribution slice: the trend is honest, the absolute
// number is optimistic, because the model has seen those tokens. Pass a SEPARATE
// .tokens file with --holdout 1 for a true generalization number.
//
// Keep --seed fixed across checkpoints. That is what makes the curve comparable.
//
// Watch a live run's curve (a new number as each checkpoint lands):
//   while true; do
//     deno run -A cli.ts eval-loss --model out/base.gguf --data data/blend.tokens --windows 64
//     sleep 600
//   done

import { readFileBytes } from "../io.ts";
import { loadModelFromGGUF } from "../export/load-gguf.ts";
import { crossEntropy, mulberry32 } from "../model/autograd.ts";
import { diskTokenSource, tokenBytes } from "../data/tokens.ts";
import type { Command, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";
import { initWebGPU } from "../backend/webgpu.ts";
import type { WebGPUBackend } from "../backend/webgpu.ts";

function die(msg: string): never {
  throw new UsageError(msg);
}

async function run(v: Values) {
  const modelPath = v.str("model");
  const tokensPath = v.str("data");
  const windows = v.num("windows");
  const seqLen = v.num("seq-len");
  const holdout = v.num("holdout");
  const seed = v.num("seed");
  const useCpu = v.bool("cpu");
  if (!(holdout > 0 && holdout <= 1)) die(`--holdout must be in (0, 1], got ${holdout}`);

  const { model, cfg } = loadModelFromGGUF(await readFileBytes(modelPath));
  if (seqLen > cfg.maxSeq) die(`--seqLen ${seqLen} exceeds model ctx ${cfg.maxSeq}`);

  const src = await diskTokenSource(tokensPath, tokenBytes(cfg.vocabSize));
  // Held-out region: the last `holdout` fraction of the stream. maxStart leaves
  // room for the input window plus its +1-shifted target.
  const regionStart = Math.floor(src.length * (1 - holdout));
  const lo = regionStart;
  const hi = src.length - seqLen - 1; // last valid window start
  if (hi <= lo) {
    die(`holdout region too small: need > ${seqLen + 1} tokens, have ${src.length - lo}`);
  }

  // FIXED windows: seeded once, so every checkpoint is scored on the same tokens.
  const rng = mulberry32(seed);
  const starts = Array.from({ length: windows }, () => lo + Math.floor(rng() * (hi - lo)));

  let gpu: WebGPUBackend | null = null;
  if (!useCpu) {
    gpu = await initWebGPU();
    if (gpu) {
      gpu.install();
      gpu.uploadParams(model.params());
    } else {
      console.log("(no WebGPU; falling back to CPU forward)");
    }
  }

  let sum = 0;
  try {
    for (const start of starts) {
      const inputs = src.window(start, seqLen);
      const targets = src.window(start + 1, seqLen);
      const loss = crossEntropy(model.forward(inputs), targets);
      if (gpu) await gpu.sync([loss]); // recycles this window's transients too
      sum += loss.data[0];
    }
  } finally {
    if (gpu) gpu.uninstall();
    src.close();
  }

  const mean = sum / windows;
  const kind = holdout >= 1
    ? "held-out file"
    : `in-distribution tail ${(holdout * 100).toFixed(1)}%`;
  console.log(
    `val loss ${mean.toFixed(4)}  ppl ${Math.exp(mean).toFixed(2)}  ` +
      `(${windows} x ${seqLen} tok, ${kind}, seed ${seed})  ${modelPath.split("/").pop()}`,
  );
}

export const evalLossCommand: Command = {
  name: "eval-loss",
  summary: "Held-out validation loss for a checkpoint, on a fixed set of windows.",
  details: `The trend signal the training loss cannot give you: per-step loss is one noisy batch,
this is a fixed, seeded sample of held-out windows, so the number is directly comparable
across checkpoints of the same run.

Keep --seed and --windows constant across checkpoints, or the curve is not a curve.

By default it scores the last 1% of the token stream. Against the training corpus that
region was still seen during training, so the trend is honest but the absolute value is
optimistic; pass a separate token file with --holdout 1 for a true generalization number.`,
  examples: [
    "eval-loss --model model.gguf --data data/blend.tokens",
    "eval-loss --model model.gguf --data data/heldout.tokens --holdout 1 --windows 128",
  ],
  flags: [
    {
      name: "model",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "the GGUF to score",
    },
    {
      name: "data",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: ".tokens file to sample windows from",
    },
    { name: "windows", type: "number", default: 64, describe: "how many windows to average" },
    {
      name: "seq-len",
      type: "number",
      default: 512,
      describe: "tokens per window; must fit the model's context",
    },
    {
      name: "holdout",
      type: "number",
      default: 0.01,
      describe: "evaluate only the last fraction of the stream (1 = the whole file)",
    },
    {
      name: "seed",
      type: "number",
      default: 1234,
      describe: "window-sampling seed; keep it fixed across checkpoints",
    },
    { name: "cpu", type: "boolean", describe: "force the CPU forward pass instead of WebGPU" },
  ],
  run: run,
};
