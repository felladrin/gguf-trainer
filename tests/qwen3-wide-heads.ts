// Qwen3 with an attention block wider than the model: nHeads * headDim != hiddenSize.
// Run:  deno run -A tests/qwen3-wide-heads.ts
//
// Published Qwen3 checkpoints do this routinely. LittleLamb-0.3B is 16 heads of
// 128 over a width of 544, Qwen3-0.6B is 16 x 128 over 1024. `finetune --resume`
// against either used to abort in `qwen3Config` before it ever read the file,
// because the helper derived the head count from hiddenSize / headDim and
// insisted the division come out whole.
//
// arch-roundtrip covers the square case through `tinyConfig`. This covers the
// wide one, on the two things that break: the config the flags produce, and
// whether the wide oProj survives an export and reload.

import { qwen3, type Qwen3Config, Qwen3Model } from "../src/arch/qwen3.ts";
import { Values } from "../src/cli/args.ts";
import { mulberry32 } from "../src/model/autograd.ts";
import { loadModelFromGGUF } from "../src/export/load-gguf.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";

let failures = 0;
function check(name: string, cond: boolean, detail = "") {
  console.log(`  ${cond ? "ok  " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
  if (!cond) failures++;
}

const tok = new BPETokenizer();
tok.train("the quick brown fox jumps over the lazy dog. ".repeat(60), 300);

// LittleLamb's real shape, which is what `inspect` prints as its resume flags.
const shape = {
  vocabSize: tok.vocabSize,
  hiddenSize: 544,
  nLayers: 28,
  maxSeq: 40960,
  headDim: 128,
};
const flags = new Map<string, string | number | boolean>([
  ["heads", 16],
  ["kv-heads", 8],
  ["ffn-dim", 2560],
]);
const cfg = qwen3.configFromFlags(shape, new Values(flags));
check(
  "--heads survives a width that headDim does not divide",
  cfg.nHeads === 16,
  `nHeads=${cfg.nHeads}`,
);
check("explicit --kv-heads is kept", cfg.nKVHeads === 8, `nKVHeads=${cfg.nKVHeads}`);
check(
  "headDim is the flag, not hiddenSize / nHeads",
  cfg.headDim === 128,
  `headDim=${cfg.headDim}`,
);

// Without --heads there is nothing to derive the count from, and guessing is worse
// than the error: the message has to say which flag unblocks it.
let message = "";
try {
  qwen3.configFromFlags(shape, new Values(new Map()));
} catch (e) {
  message = (e as Error).message;
}
check(
  "a derived head count still refuses an indivisible width",
  message.includes("--heads"),
  message || "did not throw",
);

// The same shape, small enough to export in a test: 3 heads of 8 over a width of 12.
const tiny = {
  ...cfg,
  hiddenSize: 12,
  nLayers: 2,
  nHeads: 3,
  nKVHeads: 1,
  headDim: 8,
  ffnDim: 32,
  maxSeq: 64,
};
const model = new Qwen3Model(tiny, mulberry32(7));
const qProj = model.layers[0].qProj.shape;
const oProj = model.layers[0].oProj.shape;
check("qProj is [nHeads * headDim, hidden]", qProj[0] === 24 && qProj[1] === 12, `[${qProj}]`);
check("oProj is [hidden, nHeads * headDim]", oProj[0] === 12 && oProj[1] === 24, `[${oProj}]`);

const real = model.params().reduce((n, p) => n + p.size, 0);
check(
  "paramCount matches the built model",
  qwen3.paramCount(tiny) === real,
  `declared ${qwen3.paramCount(tiny)}, built ${real}`,
);

const ids = [1, 5, 2, 9, 5, 3, 8];
const before = Float32Array.from(model.forward(ids).data);
const loaded = loadModelFromGGUF(qwen3.exportGGUF(model, tok.export(), tiny, { quant: "f32" }));
const back = loaded.cfg as Qwen3Config;
check(
  "the reloaded config keeps the wide head block",
  back.nHeads === 3 && back.headDim === 8,
  JSON.stringify(back),
);
const after = loaded.model.forward(ids).data;
let maxDiff = 0;
for (let i = 0; i < before.length; i++) maxDiff = Math.max(maxDiff, Math.abs(before[i] - after[i]));
check("logits survive export and reload", maxDiff === 0, `maxDiff=${maxDiff.toExponential(2)}`);

console.log(failures === 0 ? "\n=== qwen3 wide heads ok ===" : `\n=== ${failures} failures ===`);
if (failures > 0) throw new Error(`${failures} qwen3 wide-head failures`);
