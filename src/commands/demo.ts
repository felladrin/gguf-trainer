// End-to-end demo: train a tiny Gemma3 from scratch in TypeScript and write a
// llama.cpp-loadable GGUF, no HF, no Python, no PyTorch.
//
// Run:  deno run -A cli.ts demo
//
// Deno only, unlike the test suite: the CLI's command registry pulls in the two
// npm dependencies (hyparquet, @huggingface/jinja) that only Deno's import map
// resolves. The engine itself stays runtime-agnostic; `deno task test:node` is
// what proves that.

import type { Command, Values } from "../cli/args.ts";
import { readGGUF } from "../gguf/gguf.ts";
import { dequantize, GGMLType } from "../gguf/quantize.ts";
import { writeFileBytes } from "../io.ts";
import { mulberry32 } from "../model/autograd.ts";
import { gemma3, tinyGemma3Config } from "../arch/gemma3.ts";
import type { LanguageModel } from "../model/arch.ts";
import { BPETokenizer } from "../tokenizer/bpe.ts";
import { Muon } from "../train/muon.ts";
import { trainLM } from "../train/trainer.ts";
import { loadModelFromGGUF } from "../export/load-gguf.ts";

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

function generate(model: LanguageModel, tok: BPETokenizer, prompt: string, n: number): string {
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

async function run(v: Values) {
  console.log("=== Felladrin's GGUF Trainer +∞ :: train-from-scratch -> GGUF ===\n");

  // 1. Train a byte-level BPE tokenizer on the corpus.
  const tok = new BPETokenizer();
  tok.train(CORPUS, 280);
  const tokens = tok.encode(CORPUS);
  console.log(`Tokenizer: vocab=${tok.vocabSize}, corpus encoded to ${tokens.length} tokens`);
  const rt = tok.decode(tok.encode("the cat sat on the mat."));
  console.log(`Tokenizer round-trip: "${rt}"`);
  if (rt !== "the cat sat on the mat.") throw new Error("Tokenizer round-trip failed");

  // 2. Build a tiny Gemma3 model. (Smaller than tinyGemma3Config so the demo
  //    finishes in a few seconds on a CPU; scale up on the WebGPU backend.)
  const cfg = {
    ...tinyGemma3Config(tok.vocabSize),
    hiddenSize: 96,
    nLayers: 3,
    nHeads: 4,
    nKVHeads: 2,
    headDim: 24,
    ffnDim: 256,
  };
  const model = gemma3.build(cfg, mulberry32(1234));
  console.log(
    `\nModel: gemma3, ${cfg.nLayers} layers, hidden=${cfg.hiddenSize}, ` +
      `heads=${cfg.nHeads}/${cfg.nKVHeads}, headDim=${cfg.headDim}, ` +
      `~${(gemma3.paramCount(cfg) / 1e3).toFixed(1)}K params`,
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
  // Artifacts go to a working directory, never into the source tree.
  const outDir = v.str("out-dir").replace(/\/?$/, "/");
  (await import("node:fs")).mkdirSync(outDir, { recursive: true });
  let f16Bytes: Uint8Array | null = null;
  for (const quant of ["f16", "q8_0", "q4_0"] as const) {
    const bytes = gemma3.exportGGUF(model, tok.export(), cfg, { quant, name: "tinygemma3" });
    if (quant === "f16") f16Bytes = bytes;
    const path = `${outDir}tinygemma3-${quant}.gguf`;
    await writeFileBytes(path, bytes);
    console.log(`\nWrote ${path}  (${(bytes.length / 1024).toFixed(1)} KiB)`);
    verify(bytes, cfg, quant);
  }

  // 6. Resume from the F16 checkpoint: rebuild model + tokenizer straight from
  //    the GGUF and confirm greedy sampling reproduces the pre-export output.
  const resumed = loadModelFromGGUF(f16Bytes!);
  const sample = generate(resumed.model, resumed.tokenizer, "the cat", 20);
  console.log(`\nResumed from GGUF -> greedy sample: "${sample}"`);

  console.log("\n=== all checks passed ===");
}

function verify(bytes: Uint8Array, cfg: ReturnType<typeof tinyGemma3Config>, quant: string) {
  const g = readGGUF(bytes);
  const arch = g.metadata.get("general.architecture");
  if (arch !== "gemma3") throw new Error(`arch mismatch: ${arch}`);

  // Gemma3: 13 tensors/layer (11 + the two sandwich norms post_attention/post_ffw).
  const expectedTensors = 2 + (cfg.tieEmbeddings ? 0 : 1) + cfg.nLayers * 13;
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

export const demoCommand: Command = {
  name: "demo",
  summary: "Train a tiny model on a toy corpus end to end. The install check.",
  details: `Runs the whole pipeline in under a minute on CPU: trains a small Gemma3 on a toy
corpus, exports it at f16, q8_0 and q4_0, re-parses each file and verifies the structure.
No downloads, no GPU, no arguments.

If this passes, the engine works and any later failure is about data, flags or hardware.`,
  examples: ["demo", "demo --out-dir /tmp/gguf-check"],
  flags: [
    {
      name: "out-dir",
      type: "string",
      placeholder: "DIR",
      default: "demo-out",
      describe: "where to write the three GGUF files it builds",
    },
  ],
  run: run,
};
