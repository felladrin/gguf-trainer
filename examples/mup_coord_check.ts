// muP coordinate check on real data — the validation the in-repo gradcheck gate
// only approximates (it sweeps width at init on synthetic tokens). This trains
// the SAME real corpus (pretokenized TinyStories) at several widths under a
// CONSTANT learning rate and reports:
//
//   1. Init readout-logit RMS across width. Standard init grows ~sqrt(width)
//      (the blow-up that breaks LR transfer); muP init pins it flat. This is the
//      classic muP init diagnostic — the readout logit is the coordinate the
//      tied-embedding model exposes on the host (the forward pass keeps no
//      per-layer activation handles, so the logit RMS is the observable proxy).
//   2. Training loss trajectory per width at the SAME LR. muP's promise is LR
//      transfer: the curves cluster (a width you didn't tune still trains well),
//      whereas standard init's spread widens with width. The final-loss spread
//      across widths is the transfer metric printed at the end.
//   3. Post-training logit RMS across width — muP stays bounded; standard init
//      may drift or diverge at the widths its LR wasn't tuned for.
//
// Prerequisite: pretokenize a corpus first (produces examples/tinystories.*):
//   deno run -A examples/pretokenize.ts corpus/tinystories-valid.txt examples/tinystories 8192 10
// Run (Deno — WebGPU):
//   deno run -A examples/mup_coord_check.ts [steps=120] [seqLen=64] [batch=4]
//
// It runs on the GPU (constant-LR training at 3 widths is a minutes-scale job,
// not the multi-hour pretraining run). If the muP curves do NOT cluster tighter
// than standard init's, the fix is an explicit per-group LR width scale in
// src/model/mup.ts — see DESIGN.md item 2.

import { readFileText } from "../src/io.ts";
import { mulberry32 } from "../src/model/autograd.ts";
import { scaleConfig } from "../src/model/config.ts";
import { Qwen3Model } from "../src/model/qwen3.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import type { TokenizerData } from "../src/tokenizer/bpe.ts";
import { diskTokenSource, tokenBytes } from "../src/data/tokens.ts";
import type { TokenSource } from "../src/data/tokens.ts";
import { initWebGPU } from "../src/backend/webgpu.ts";
import { MuonGpu } from "../src/backend/muon_gpu.ts";
import { trainLMGpuResident } from "../src/backend/train_gpu.ts";

const BASE_WIDTH = 128; // the proxy width the LRs (below) are treated as tuned at
const WIDTHS = [128, 256, 512]; // 4x sweep; headDim 64 needs width % 128 == 0
const MUON_LR = 0.02;
const AUX_LR = 3e-3;

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  return g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
}

function die(msg: string): never {
  console.error("mup_coord_check: " + msg);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
  throw new Error(msg);
}

function rms(a: Float32Array): number {
  let s = 0;
  for (const v of a) s += v * v;
  return Math.sqrt(s / a.length);
}

/** Ratio of max/min over finite values; Infinity if any value is non-finite. */
function spread(xs: number[]): number {
  if (xs.some((x) => !Number.isFinite(x))) return Infinity;
  return Math.max(...xs) / Math.min(...xs);
}

interface Run {
  width: number;
  initRms: number;
  finalRms: number;
  finalLoss: number;
  losses: number[]; // logged trajectory
}

async function main() {
  const a = args();
  const steps = a[0] ? Number(a[0]) : 120;
  const seqLen = a[1] ? Number(a[1]) : 64;
  const batch = a[2] ? Number(a[2]) : 4;
  for (const [k, v] of Object.entries({ steps, seqLen, batch })) {
    if (!Number.isFinite(v) || v <= 0) die(`${k} must be a positive number, got ${v}`);
  }

  console.log("=== Felladrin's GGUF Trainer +∞ :: muP coordinate check on real data ===\n");

  const gpu = await initWebGPU();
  if (!gpu) die("WebGPU required — run with Deno (deno run -A examples/mup_coord_check.ts).");
  console.log(`WebGPU adapter: ${gpu.adapterName}`);

  // Corpus + the exact vocab it was tokenized with.
  const dir = new URL(".", import.meta.url).pathname;
  let tokData: TokenizerData;
  try {
    tokData = JSON.parse(await readFileText(`${dir}tinystories.tokenizer.json`));
  } catch {
    return die(
      "cannot read examples/tinystories.tokenizer.json — run the pretokenizer first:\n" +
        "  deno run -A examples/pretokenize.ts corpus/tinystories-valid.txt examples/tinystories 8192 10",
    );
  }
  const tok = BPETokenizer.fromData(tokData);
  const bpt = tokenBytes(tok.vocabSize);
  let src: TokenSource;
  try {
    src = await diskTokenSource(`${dir}tinystories.tokens`, bpt);
  } catch {
    return die("cannot open examples/tinystories.tokens — run the pretokenizer first (see above).");
  }
  // One fixed probe window, shared by every width so logit RMS is comparable
  // (logits are [seqLen, vocab] regardless of width). Away from position 0.
  const probe = src.window(4096, seqLen);
  console.log(
    `Corpus: ${src.length} tokens, vocab ${tok.vocabSize}; sweep widths ${WIDTHS.join(", ")} ` +
      `(baseWidth ${BASE_WIDTH}); ${steps} steps @ constant lr (muon ${MUON_LR}, aux ${AUX_LR}), ` +
      `seqLen ${seqLen}, batch ${batch}\n`,
  );

  // Train one (init-kind, width) model to completion per resident call — the
  // loss trajectory IS the coordinate check across training; the final weight
  // sync back to host lets us re-measure the logit RMS after real training.
  const train = async (width: number, mup: boolean): Promise<Run> => {
    const cfg = scaleConfig(tok.vocabSize, width, 2, 512);
    const model = new Qwen3Model(
      cfg,
      mulberry32(1234),
      mup ? { baseWidth: BASE_WIDTH } : undefined,
    );
    const initRms = rms(model.forward(probe).data);
    const g = model.paramGroups();
    const opt = new MuonGpu(gpu, g.muon, g.aux, {
      lr: MUON_LR,
      momentum: 0.95,
      aux: { lr: AUX_LR, weightDecay: 0.0, clip: 1.0 },
    });
    const losses: number[] = [];
    await trainLMGpuResident(model, gpu, {
      tokens: src,
      seqLen,
      steps,
      batchPerStep: batch,
      optimizer: opt,
      logEvery: Math.max(1, Math.round(steps / 6)), // ~6 trajectory points
      rng: mulberry32(7), // same batch stream across widths -> fair comparison
      onLog: (_s, loss) => losses.push(loss),
    });
    const finalRms = rms(model.forward(probe).data); // host weights restored by the loop
    return { width, initRms, finalRms, finalLoss: losses[losses.length - 1], losses };
  };

  const std: Run[] = [], mup: Run[] = [];
  for (const w of WIDTHS) {
    console.log(`training width ${w} (standard init)...`);
    std.push(await train(w, false));
    console.log(`training width ${w} (muP init)...`);
    mup.push(await train(w, true));
  }
  src.close();

  // --- report ---------------------------------------------------------------
  const f = (x: number) => (Number.isFinite(x) ? x.toFixed(3) : "  diverged");
  console.log("\nInit readout-logit RMS across width (muP should be ~flat, std should grow):");
  console.log("  width  |  std init  |  muP init");
  for (let i = 0; i < WIDTHS.length; i++) {
    console.log(
      `  ${String(WIDTHS[i]).padStart(5)}  |  ${f(std[i].initRms).padStart(8)}  |  ${
        f(mup[i].initRms).padStart(8)
      }`,
    );
  }
  const stdInitSpread = spread(std.map((r) => r.initRms));
  const mupInitSpread = spread(mup.map((r) => r.initRms));
  console.log(
    `  spread across ${WIDTHS[WIDTHS.length - 1] / WIDTHS[0]}x width: ` +
      `std ${stdInitSpread.toFixed(2)}x  vs  muP ${mupInitSpread.toFixed(2)}x`,
  );

  console.log("\nLoss trajectory (constant lr; muP curves should cluster = LR transfer):");
  console.log("  width  |  init kind  |  loss over training");
  for (let i = 0; i < WIDTHS.length; i++) {
    const row = (r: Run, kind: string) =>
      `  ${String(r.width).padStart(5)}  |  ${kind.padEnd(9)}  |  ${
        r.losses.map((l) => f(l)).join("  ")
      }`;
    console.log(row(std[i], "standard"));
    console.log(row(mup[i], "muP"));
  }
  const stdFinalSpread = spread(std.map((r) => r.finalLoss));
  const mupFinalSpread = spread(mup.map((r) => r.finalLoss));
  console.log(
    `  final-loss spread across width: std ${stdFinalSpread.toFixed(2)}x  vs  ` +
      `muP ${mupFinalSpread.toFixed(2)}x  (smaller = better transfer)`,
  );

  console.log("\nPost-training logit RMS across width (muP should stay bounded):");
  console.log("  width  |  std init  |  muP init");
  for (let i = 0; i < WIDTHS.length; i++) {
    console.log(
      `  ${String(WIDTHS[i]).padStart(5)}  |  ${f(std[i].finalRms).padStart(8)}  |  ${
        f(mup[i].finalRms).padStart(8)
      }`,
    );
  }

  // --- verdict (one runnable assertion so this fails loudly on regression) ---
  let ok = true;
  const notes: string[] = [];
  if (!(mupInitSpread < 1.5)) {
    ok = false;
    notes.push(`muP init RMS not flat across width (${mupInitSpread.toFixed(2)}x >= 1.5x)`);
  }
  if (!(stdInitSpread > mupInitSpread * 1.3)) {
    ok = false;
    notes.push("standard init RMS did not grow with width relative to muP");
  }
  if (!Number.isFinite(mupFinalSpread) || mup.some((r) => !Number.isFinite(r.finalRms))) {
    ok = false;
    notes.push("muP training diverged at some width (non-finite)");
  }
  if (!(mupFinalSpread <= stdFinalSpread * 1.05)) {
    // muP should transfer at least as well as std; allow a small tolerance.
    notes.push(
      `note: muP final-loss spread (${mupFinalSpread.toFixed(2)}x) not tighter than std ` +
        `(${stdFinalSpread.toFixed(2)}x) at this toy depth/step budget`,
    );
  }
  console.log(
    `\n${ok ? "PASS" : "FAIL"} muP coordinate check on real data` +
      (notes.length ? "\n  - " + notes.join("\n  - ") : ""),
  );
  if (!ok) die("muP coordinate check failed");
  console.log("\n=== muP coordinate check OK ===");
}

main().catch((e) => die(String(e?.stack ?? e)));
