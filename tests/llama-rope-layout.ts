// Standalone assert check for the llama Q/K row order.
//
// llama.cpp rotates a `llama` checkpoint with LLAMA_ROPE_TYPE_NORM, pairing
// dimensions (2j, 2j+1); this engine's rope() pairs (j, j+headDim/2). The two are
// the same rotation over a different row order, so the llama arch reorders Q and K
// on load and on export.
//
// This check exists because the defect it guards is silent. Greedy generation
// still reads fine with the wrong order, and the CPU and GPU backends agree with
// each other because they share the path, so neither gpu-parity nor gradcheck sees
// it. The way it showed up was a base checkpoint scoring ppl 5.73 on "The cat sat
// on the mat." repeated 400 times, against llama.cpp's 1.006 on the same file.
//
// Run:  deno run -A tests/llama-rope-layout.ts
import { llama, llamaConfig, LlamaModel } from "../src/arch/llama.ts";
import { readGGUF } from "../src/gguf/gguf.ts";
import type { TokenizerData } from "../src/tokenizer/bpe.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

// Tiny but not degenerate: 2 KV heads under 4 query heads, so the Q and K
// reorders cover a different number of head blocks and a swap between them shows.
const HEAD_DIM = 8;
const cfg = { ...llamaConfig(64, 32, 1, 128, HEAD_DIM), nHeads: 4, nKVHeads: 2, ffnDim: 64 };

let seed = 1;
const rng = () => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
};

const model = new LlamaModel(cfg, rng);
const before = {
  q: Float32Array.from(model.layers[0].qProj.data),
  k: Float32Array.from(model.layers[0].kProj.data),
  v: Float32Array.from(model.layers[0].vProj.data),
};

const tok: TokenizerData = {
  tokens: Array.from({ length: cfg.vocabSize }, (_, i) => `t${i}`),
  merges: [],
  bosId: 0,
  eosId: 0,
  specials: [],
};

const bytes = llama.exportGGUF(model, tok, cfg, { quant: "f32" });
const g = readGGUF(bytes);

// --- The export must NOT be a copy of what is in memory ----------------------

const qOut = g.tensors.find((t) => t.name === "blk.0.attn_q.weight")!;
const vOut = g.tensors.find((t) => t.name === "blk.0.attn_v.weight")!;
const asF32 = (t: { data: Uint8Array }) =>
  new Float32Array(t.data.buffer, t.data.byteOffset, t.data.byteLength / 4);

ok(
  !asF32(qOut).every((x, i) => x === before.q[i]),
  "Q must be reordered on export, or llama.cpp rotates the wrong dimensions",
);
ok(
  asF32(vOut).every((x, i) => x === before.v[i]),
  "V carries no rotation and must be written untouched",
);

// The reorder must be a permutation of whole rows, not a reshuffle of values: the
// multiset of rows is unchanged, and each row's contents stay together.
const inDim = cfg.hiddenSize;
const rowKey = (a: Float32Array, r: number) => a.slice(r * inDim, (r + 1) * inDim).join(",");
const rowsBefore = new Set<string>();
const rowsAfter = new Set<string>();
for (let r = 0; r < cfg.nHeads * HEAD_DIM; r++) {
  rowsBefore.add(rowKey(before.q, r));
  rowsAfter.add(rowKey(asF32(qOut), r));
}
ok(rowsBefore.size === rowsAfter.size, "the reorder must not merge rows");
ok([...rowsBefore].every((r) => rowsAfter.has(r)), "every row must survive the reorder intact");

// The permutation stays inside its own head: row 0 of head 1 must not land in head 0.
const headOf = (a: Float32Array, key: string) => {
  for (let r = 0; r < cfg.nHeads * HEAD_DIM; r++) {
    if (rowKey(a, r) === key) return Math.floor(r / HEAD_DIM);
  }
  return -1;
};
for (let r = 0; r < cfg.nHeads * HEAD_DIM; r++) {
  const key = rowKey(asF32(qOut), r);
  ok(
    headOf(before.q, key) === Math.floor(r / HEAD_DIM),
    `row ${r} crossed a head boundary: heads must be reordered independently`,
  );
}

// --- Round trip --------------------------------------------------------------

const reloaded = new LlamaModel(cfg, () => 0.5);
llama.loadWeights(reloaded, g);
const q2 = reloaded.layers[0].qProj.data;
const k2 = reloaded.layers[0].kProj.data;
ok(q2.every((x, i) => x === before.q[i]), "export then load must return the original Q");
ok(k2.every((x, i) => x === before.k[i]), "export then load must return the original K");

// --- The exact convention ----------------------------------------------------

// Our row j and row j+half are llama.cpp's rows 2j and 2j+1 of the same head.
// Pinned explicitly, because a reorder that is merely self-consistent round-trips
// perfectly and still disagrees with llama.cpp.
const half = HEAD_DIM / 2;
const kOut = g.tensors.find((t) => t.name === "blk.0.attn_k.weight")!;

// K gets the same treatment as Q and over a different head count, so checking it
// only through the round trip above would leave exactly the self-consistent-but-
// wrong case this test exists to catch.
for (
  const [what, exported, original, heads] of [
    ["Q", asF32(qOut), before.q, cfg.nHeads],
    ["K", asF32(kOut), before.k, cfg.nKVHeads],
  ] as [string, Float32Array, Float32Array, number][]
) {
  for (let h = 0; h < heads; h++) {
    for (let j = 0; j < half; j++) {
      ok(
        rowKey(exported, h * HEAD_DIM + 2 * j) === rowKey(original, h * HEAD_DIM + j),
        `${what} head ${h}: our row ${j} must export as llama.cpp's row ${2 * j}`,
      );
      ok(
        rowKey(exported, h * HEAD_DIM + 2 * j + 1) === rowKey(original, h * HEAD_DIM + j + half),
        `${what} head ${h}: our row ${j + half} must export as llama.cpp's row ${2 * j + 1}`,
      );
    }
  }
}

// The guard that turns a shape mismatch into an error instead of a zero-filled
// tensor: 3 heads do not tile 4 heads' worth of rows.
let threw = false;
try {
  llama.exportGGUF(model, tok, { ...cfg, nHeads: 3 }, { quant: "f32" });
} catch {
  threw = true;
}
ok(threw, "a head count that does not tile the rows must throw, not zero-fill");

console.log("=== llama rope layout checks passed ===");
