// Streaming-corpus training demo — the entry point for a real (non-toy) run.
//
// Instead of holding the whole token stream in memory, this pretokenizes the
// corpus to a compact binary file once and streams fixed-length windows off
// disk during training (src/data/tokens.ts). Peak memory is O(window), not
// O(corpus), so the same loop scales from this synthetic corpus to a
// TinyStories / FineWeb-Edu slice — swap `CORPUS` for the contents of a real
// text file (readFileText) and bump gemma3Config().
//
// Run (any runtime):  deno run -A examples/train_streaming.ts
//                     node --experimental-strip-types examples/train_streaming.ts
// This runs on the CPU reference backend so it works everywhere; for the GPU
// path use MuonGpu + trainLMGpuResident as in demo_gpu.ts (both accept the same
// TokenSource).

import { mulberry32 } from "../src/model/autograd.ts";
import { gemma3Config, gemma3ParamCount } from "../src/model/config.ts";
import { Gemma3Model } from "../src/model/gemma3.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { Muon } from "../src/train/muon.ts";
import { trainLM } from "../src/train/trainer.ts";
import { wsdSchedule } from "../src/train/schedule.ts";
import { diskTokenSource, tokenBytes, writeTokenFile } from "../src/data/tokens.ts";
import { memTokenSource } from "../src/data/tokens.ts";

const CORPUS = `
the cat sat on the mat. the cat saw a red ball. the dog ran to the cat.
the dog and the cat played with the red ball on the mat. the sun was warm.
a bird sang in the tree. the cat looked at the bird. the bird flew away.
the dog ran after the ball. the cat sat on the mat and slept in the sun.
`.trim().replace(/\n/g, " ").repeat(60);

function argmax(row: number[]): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

function generate(model: Gemma3Model, tok: BPETokenizer, prompt: string, n: number): string {
  const ids = tok.encode(prompt);
  for (let i = 0; i < n; i++) {
    const ctx = ids.slice(-model.cfg.maxSeq);
    const logits = model.forward(ctx);
    const V = model.cfg.vocabSize;
    const base = (ctx.length - 1) * V;
    ids.push(argmax(Array.from(logits.data.subarray(base, base + V))));
  }
  return tok.decode(ids);
}

async function main() {
  console.log("=== Felladrin's GGUF Trainer +∞ :: streaming-corpus training ===\n");

  // 1. Tokenizer + pretokenize the corpus to a compact binary on disk.
  const tok = new BPETokenizer();
  tok.train(CORPUS, 512);
  const ids = tok.encode(CORPUS);
  const bpt = tokenBytes(tok.vocabSize);
  const outDir = new URL(".", import.meta.url).pathname;
  const tokenPath = `${outDir}stream-corpus.tokens`;
  await writeTokenFile(tokenPath, ids, bpt);
  console.log(
    `Tokenizer: vocab=${tok.vocabSize}; corpus ${ids.length} tokens -> ${tokenPath} ` +
      `(${bpt} B/token, ${((ids.length * bpt) / 1024).toFixed(1)} KiB on disk)`,
  );

  // 2. Stream windows off disk. Sanity: the disk source must return byte-for-byte
  //    the same windows as an in-memory source over the same tokens.
  const src = await diskTokenSource(tokenPath, bpt);
  const mem = memTokenSource(ids);
  if (src.length !== mem.length) throw new Error("disk/mem length mismatch");
  const probe = mulberry32(1);
  for (let t = 0; t < 200; t++) {
    const start = Math.floor(probe() * (src.length - 40));
    const a = src.window(start, 40), b = mem.window(start, 40);
    for (let i = 0; i < 40; i++) {
      if (a[i] !== b[i]) throw new Error(`disk window mismatch at ${start}+${i}`);
    }
  }
  console.log("Disk vs in-memory windows: identical over 200 random probes ✓");

  // 3. A real-shape (if small) model via scaleConfig, trained from the disk
  //    stream with a WSD schedule. Kept CPU-small here so the demo is quick;
  //    the same call scales up by widening gemma3Config and moving to the GPU loop.
  const cfg = gemma3Config(tok.vocabSize, 128, 2, 64);
  const model = new Gemma3Model(cfg, mulberry32(1234));
  console.log(
    `Model: hidden=${cfg.hiddenSize}, ${cfg.nLayers} layers, heads=${cfg.nHeads}/${cfg.nKVHeads}, ` +
      `ffn=${cfg.ffnDim}, ~${(gemma3ParamCount(cfg) / 1e3).toFixed(1)}K params\n`,
  );

  const groups = model.paramGroups();
  const opt = new Muon(groups.muon, groups.aux, {
    lr: 0.02,
    momentum: 0.95,
    aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 },
  });
  const steps = 40;
  const schedule = wsdSchedule({
    warmupSteps: 4,
    stableSteps: 28,
    cooldownSteps: 8,
    minScale: 0.1,
  });

  let first = 0, last = 0;
  const t0 = Date.now();
  trainLM(model, {
    tokens: src, // <-- streamed off disk
    seqLen: 32,
    steps,
    batchPerStep: 2,
    optimizer: opt,
    schedule,
    logEvery: 15,
    rng: mulberry32(7),
    onLog: (step, loss) => {
      if (step === 0) first = loss;
      last = loss;
      console.log(`  step ${String(step).padStart(3)}  loss ${loss.toFixed(4)}`);
    },
  });
  src.close();
  const secs = (Date.now() - t0) / 1000;
  console.log(`\nTraining: loss ${first.toFixed(3)} -> ${last.toFixed(3)} in ${secs.toFixed(1)}s`);
  if (!(last < first)) throw new Error("Loss did not decrease");

  console.log(`\nGreedy sample: "${generate(model, tok, "the cat", 16)}"`);
  console.log("\n=== streaming pipeline OK ===");
}

main().catch((e) => {
  console.error("STREAMING DEMO FAILED:", e);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
});
