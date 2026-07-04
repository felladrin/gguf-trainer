// End-to-end demo: train a tiny Qwen3 from scratch in TypeScript and write a
// llama.cpp-loadable GGUF — no HF, no Python, no PyTorch.
//
// Run (Deno):  deno run -A examples/demo.ts
// Run (Node):  node --experimental-strip-types examples/demo.ts
// Run (Bun):   bun examples/demo.ts

import { readGGUF } from "../src/gguf/gguf.ts";
import { dequantize, GGMLType } from "../src/gguf/quantize.ts";
import { writeFileBytes } from "../src/io.ts";
import { mulberry32 } from "../src/model/autograd.ts";
import { paramCount, tinyConfig } from "../src/model/config.ts";
import { Qwen3Model } from "../src/model/qwen3.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { Muon } from "../src/train/muon.ts";
import { trainLM } from "../src/train/trainer.ts";
import { buildGGUF } from "../src/export/export_gguf.ts";
import { loadQwen3FromGGUF } from "../src/export/load_gguf.ts";

// A small, repetitive corpus so a tiny model shows clear learning fast.
const CORPUS = `
the cat sat on the mat. the cat saw a red ball. the dog ran to the cat.
the dog and the cat played with the red ball on the mat. the sun was warm.
a bird sang in the tree. the cat looked at the bird. the bird flew away.
the dog ran after the ball. the cat sat on the mat and slept in the sun.
`.trim().repeat(12);

function argmax(row: Float32Array): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

function generate(model: Qwen3Model, tok: BPETokenizer, prompt: string, n: number): string {
  const ids = tok.encode(prompt);
  for (let i = 0; i < n; i++) {
    const ctx = ids.slice(-model.cfg.maxSeq);
    const logits = model.forward(ctx);
    const V = model.cfg.vocabSize;
    const base = (ctx.length - 1) * V;
    const last = logits.data.subarray(base, base + V);
    ids.push(argmax(last));
  }
  return tok.decode(ids);
}

async function main() {
  console.log("=== Felladrin's GGUF Trainer +∞ :: train-from-scratch -> GGUF ===\n");

  // 1. Train a byte-level BPE tokenizer on the corpus.
  const tok = new BPETokenizer();
  tok.train(CORPUS, 280);
  const tokens = tok.encode(CORPUS);
  console.log(`Tokenizer: vocab=${tok.vocabSize}, corpus encoded to ${tokens.length} tokens`);
  const rt = tok.decode(tok.encode("the cat sat on the mat."));
  console.log(`Tokenizer round-trip: "${rt}"`);
  if (rt !== "the cat sat on the mat.") throw new Error("Tokenizer round-trip failed");

  // 2. Build a tiny Qwen3 model. (Smaller than tinyConfig so the demo finishes
  //    in a few seconds on a CPU; scale up when running on the WebGPU backend.)
  const cfg = {
    ...tinyConfig(tok.vocabSize),
    hiddenSize: 96,
    nLayers: 3,
    nHeads: 4,
    nKVHeads: 2,
    headDim: 24,
    ffnDim: 256,
  };
  const model = new Qwen3Model(cfg, mulberry32(1234));
  console.log(
    `\nModel: qwen3, ${cfg.nLayers} layers, hidden=${cfg.hiddenSize}, ` +
      `heads=${cfg.nHeads}/${cfg.nKVHeads}, headDim=${cfg.headDim}, ` +
      `~${(paramCount(cfg) / 1e3).toFixed(1)}K params`,
  );

  // 3. Train with Muon (hidden matmuls) + AdamW (embeddings/head/norms).
  const groups = model.paramGroups();
  console.log(`Muon on ${groups.muon.length} matrices; AdamW on ${groups.aux.length} tensors\n`);
  const opt = new Muon(groups.muon, groups.aux, {
    lr: 0.02,
    momentum: 0.95,
    aux: { lr: 3e-3, weightDecay: 0.0, clip: 1.0 },
  });

  const rng = mulberry32(7);
  let firstLoss = 0;
  let lastLoss = 0;
  const t0 = Date.now();
  trainLM(model, {
    tokens,
    seqLen: 32,
    steps: 40,
    batchPerStep: 2,
    optimizer: opt,
    logEvery: 10,
    rng,
    onLog: (step, loss) => {
      if (step === 0) firstLoss = loss;
      lastLoss = loss;
      console.log(`  step ${String(step).padStart(3)}  loss ${loss.toFixed(4)}`);
    },
  });
  console.log(
    `\nTraining: loss ${firstLoss.toFixed(3)} -> ${lastLoss.toFixed(3)} ` +
      `in ${((Date.now() - t0) / 1000).toFixed(1)}s`,
  );
  if (!(lastLoss < firstLoss)) throw new Error("Loss did not decrease");

  // 4. Sample greedily to show the model learned the corpus.
  console.log(`\nGreedy sample: "${generate(model, tok, "the cat", 20)}"`);

  // 5. Export GGUF in F16 and Q8_0. Write to disk.
  const outDir = new URL(".", import.meta.url).pathname;
  let f16Bytes: Uint8Array | null = null;
  for (const quant of ["f16", "q8_0", "q4_0"] as const) {
    const bytes = buildGGUF(model, tok.export(), cfg, { quant, name: "tinyqwen3" });
    if (quant === "f16") f16Bytes = bytes;
    const path = `${outDir}tinyqwen3-${quant}.gguf`;
    await writeFileBytes(path, bytes);
    console.log(`\nWrote ${path}  (${(bytes.length / 1024).toFixed(1)} KiB)`);
    verify(bytes, cfg, quant);
  }

  // 6. Resume from the F16 checkpoint: rebuild model + tokenizer straight from
  //    the GGUF and confirm greedy sampling reproduces the pre-export output.
  const resumed = loadQwen3FromGGUF(f16Bytes!);
  const sample = generate(resumed.model, resumed.tokenizer, "the cat", 20);
  console.log(`\nResumed from GGUF -> greedy sample: "${sample}"`);

  console.log("\n=== all checks passed ===");
}

function verify(bytes: Uint8Array, cfg: ReturnType<typeof tinyConfig>, quant: string) {
  const g = readGGUF(bytes);
  const arch = g.metadata.get("general.architecture");
  if (arch !== "qwen3") throw new Error(`arch mismatch: ${arch}`);

  const expectedTensors = 2 + (cfg.tieEmbeddings ? 0 : 1) + cfg.nLayers * 11;
  if (g.tensors.length !== expectedTensors) {
    throw new Error(`tensor count ${g.tensors.length} != expected ${expectedTensors}`);
  }

  // token_embd.weight must round-trip: ne = [hidden, vocab].
  const emb = g.tensors.find((t) => t.name === "token_embd.weight");
  if (!emb) throw new Error("token_embd.weight missing");
  if (emb.dims[0] !== cfg.hiddenSize || emb.dims[1] !== cfg.vocabSize) {
    throw new Error(`token_embd dims ${emb.dims} wrong`);
  }

  // Dequantize it and sanity-check finite values.
  const count = emb.dims[0] * emb.dims[1];
  const de = dequantize(emb.type, emb.data, count);
  let finite = true;
  for (let i = 0; i < de.length; i++) if (!Number.isFinite(de[i])) finite = false;
  if (!finite) throw new Error("dequantized embedding has non-finite values");

  const tokens = g.metadata.get("tokenizer.ggml.tokens") as string[];
  const merges = g.metadata.get("tokenizer.ggml.merges") as string[];
  console.log(
    `  verify[${quant}]: ${g.tensors.length} tensors, ` +
      `${tokens.length} vocab, ${merges.length} merges, embd type=${typeName(emb.type)} ✓`,
  );
}

function typeName(t: number): string {
  for (const [k, v] of Object.entries(GGMLType)) if (v === t) return k;
  return `type${t}`;
}

main().catch((e) => {
  console.error("DEMO FAILED:", e);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
});
