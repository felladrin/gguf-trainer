// Held-out validation loss for a pretrain checkpoint — the trend signal the
// per-step training loss can't give you (that's a single noisy batch) and the
// parity probe was never meant to be (it's a 16-token GPU-vs-CPU trust gate).
//
// Scores a GGUF checkpoint on a FIXED, seeded set of windows drawn from a token
// stream: same seed -> same windows every run, so the mean loss is directly
// comparable across checkpoints. Forward-only, on the GPU, using the trainer's
// own model — the exact proven pattern of the pretrain parity probe, so it never
// touches the running trainer. Safe to run against pretrain-blend-base.gguf while
// stage 1 is still training (each checkpoint is written atomically).
//
//   deno run -A --unstable-webgpu examples/eval_val.ts <model.gguf> <tokens.tokens> \
//       [--windows=N] [--seqLen=L] [--holdout=F] [--seed=S] [--cpu]
//
//     --windows=N   number of windows to average (default 64)
//     --seqLen=L    tokens per window (default 512; must be <= model ctx)
//     --holdout=F   evaluate only the LAST fraction F of the stream (default 0.01).
//                   For the training corpus this is an in-distribution slice: the
//                   trend is honest, the absolute number is optimistic (the model
//                   has seen these tokens). Pass a SEPARATE .tokens file with
//                   --holdout=1 for a true generalization number.
//     --seed=S      window-sampling seed (default 1234). Keep it FIXED across
//                   checkpoints — that's what makes the curve comparable.
//     --cpu         force CPU forward (default: WebGPU when available)
//
// Watch a live run's curve (new number as each checkpoint lands):
//   while true; do
//     deno run -A --unstable-webgpu examples/eval_val.ts \
//       examples/pretrain-blend-base.gguf examples/blend.tokens --windows=64
//     sleep 600
//   done

import { readFileBytes } from "../src/io.ts";
import { loadGemma3FromGGUF } from "../src/export/load_gguf.ts";
import { crossEntropy, mulberry32 } from "../src/model/autograd.ts";
import { diskTokenSource, tokenBytes } from "../src/data/tokens.ts";
import { initWebGPU } from "../src/backend/webgpu.ts";
import type { WebGPUBackend } from "../src/backend/webgpu.ts";

function parseArgs(): {
  pos: string[];
  get(n: string): string | undefined;
  has(n: string): boolean;
} {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  const raw: string[] = g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
  const pos: string[] = [];
  const flags = new Map<string, string>();
  for (const a of raw) {
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      if (eq < 0) flags.set(a.slice(2), "");
      else flags.set(a.slice(2, eq), a.slice(eq + 1));
    } else pos.push(a);
  }
  return { pos, get: (n) => flags.get(n), has: (n) => flags.has(n) };
}
function die(msg: string): never {
  console.error("eval_val: " + msg);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
  throw new Error(msg);
}

async function main() {
  const flags = parseArgs();
  const modelPath = flags.pos[0];
  const tokensPath = flags.pos[1];
  if (!modelPath || !tokensPath) {
    die(
      "usage: eval_val <model.gguf> <tokens.tokens> [--windows=N] [--seqLen=L] [--holdout=F] [--seed=S] [--cpu]",
    );
  }
  const windows = flags.get("windows") ? Number(flags.get("windows")) : 64;
  const seqLen = flags.get("seqLen") ? Number(flags.get("seqLen")) : 512;
  const holdout = flags.get("holdout") ? Number(flags.get("holdout")) : 0.01;
  const seed = flags.get("seed") ? Number(flags.get("seed")) : 1234;
  if (!(holdout > 0 && holdout <= 1)) die(`--holdout must be in (0, 1], got ${holdout}`);

  const { model, cfg } = loadGemma3FromGGUF(await readFileBytes(modelPath));
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
  if (!flags.has("cpu")) {
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

main().catch((e) => die(String(e?.stack ?? e)));
