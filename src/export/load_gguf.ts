// Load a Qwen3 checkpoint back from a GGUF this framework produced — the
// inverse of export_gguf.ts. Reconstructs the config from metadata, the model
// weights via the existing dequantizer, and the tokenizer from the embedded
// vocab/merges, so a run can resume (continue training or sample) from a saved
// GGUF instead of re-initializing from scratch.
//
// Scope: GGUFs written by this project's buildGGUF(). Tensor names, metadata
// keys, and the [out,in]->ne[in,out] weight layout match by construction, so
// loading is name lookup + dequantize + copy. Fidelity follows the export
// quant: f32/f16 round-trip cleanly; a q4_0/q8_0 checkpoint resumes from its
// (lossy) dequantized weights — export in f16 or f32 for a faithful resume.
// Loading an arbitrary external Qwen3 GGUF is a larger job (its tokenizer would
// need importing beyond our own gpt2/BPE data); see docs/HANDOFF.md.

import { readGGUF } from "../gguf/gguf.ts";
import type { GGUFFile } from "../gguf/gguf.ts";
import { dequantize } from "../gguf/quantize.ts";
import { mulberry32 } from "../model/autograd.ts";
import type { Tensor } from "../model/autograd.ts";
import type { Qwen3Config } from "../model/config.ts";
import { Qwen3Model } from "../model/qwen3.ts";
import { BPETokenizer } from "../tokenizer/bpe.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";

const ARCH = "qwen3";

function metaNum(g: GGUFFile, key: string): number {
  const v = g.metadata.get(key);
  if (typeof v !== "number") throw new Error(`GGUF missing numeric metadata "${key}"`);
  return v;
}

/** Reconstruct the Qwen3Config from the GGUF's qwen3.* metadata. */
export function configFromGGUF(g: GGUFFile): Qwen3Config {
  const arch = g.metadata.get("general.architecture");
  if (arch !== ARCH) throw new Error(`expected general.architecture "${ARCH}", got "${arch}"`);
  // Tie is implicit: buildGGUF omits output.weight when embeddings are tied.
  const tieEmbeddings = !g.tensors.some((t) => t.name === "output.weight");
  return {
    vocabSize: metaNum(g, `${ARCH}.vocab_size`),
    hiddenSize: metaNum(g, `${ARCH}.embedding_length`),
    nLayers: metaNum(g, `${ARCH}.block_count`),
    nHeads: metaNum(g, `${ARCH}.attention.head_count`),
    nKVHeads: metaNum(g, `${ARCH}.attention.head_count_kv`),
    headDim: metaNum(g, `${ARCH}.attention.key_length`),
    ffnDim: metaNum(g, `${ARCH}.feed_forward_length`),
    ropeBase: metaNum(g, `${ARCH}.rope.freq_base`),
    rmsEps: metaNum(g, `${ARCH}.attention.layer_norm_rms_epsilon`),
    maxSeq: metaNum(g, `${ARCH}.context_length`),
    tieEmbeddings,
  };
}

/** Reconstruct the tokenizer data from the GGUF's tokenizer.ggml.* metadata. */
export function tokenizerFromGGUF(g: GGUFFile): TokenizerData {
  const tokens = g.metadata.get("tokenizer.ggml.tokens");
  const merges = g.metadata.get("tokenizer.ggml.merges");
  if (!Array.isArray(tokens) || !Array.isArray(merges)) {
    throw new Error("GGUF missing tokenizer.ggml.tokens/merges");
  }
  return {
    tokens: tokens as string[],
    merges: merges as string[],
    bosId: metaNum(g, "tokenizer.ggml.bos_token_id"),
    eosId: metaNum(g, "tokenizer.ggml.eos_token_id"),
  };
}

/**
 * Copy weights from the GGUF into an existing model's params by tensor name.
 * The model's shapes must match the GGUF (build it from configFromGGUF). Each
 * tensor is dequantized per its own stored type — export may fall back to f16
 * for a matrix whose inner dim isn't a multiple of 32, so per-tensor type read
 * (not a global assumption) is required.
 */
export function loadWeightsFromGGUF(model: Qwen3Model, g: GGUFFile): void {
  const byName = new Map(g.tensors.map((t) => [t.name, t]));
  const load = (name: string, dst: Tensor) => {
    const t = byName.get(name);
    if (!t) throw new Error(`GGUF missing tensor "${name}"`);
    const de = dequantize(t.type, t.data, dst.size);
    // Serialized element order is our row-major [out,in] flattened, so the
    // dequantized array drops straight back into dst.data.
    dst.data.set(de);
  };

  load("token_embd.weight", model.tokenEmbd);
  load("output_norm.weight", model.outputNorm);
  if (model.output) load("output.weight", model.output);

  model.layers.forEach((L, i) => {
    const p = `blk.${i}`;
    load(`${p}.attn_norm.weight`, L.attnNorm);
    load(`${p}.attn_q.weight`, L.qProj);
    load(`${p}.attn_k.weight`, L.kProj);
    load(`${p}.attn_v.weight`, L.vProj);
    load(`${p}.attn_output.weight`, L.oProj);
    load(`${p}.attn_q_norm.weight`, L.qNorm);
    load(`${p}.attn_k_norm.weight`, L.kNorm);
    load(`${p}.ffn_norm.weight`, L.ffnNorm);
    load(`${p}.ffn_gate.weight`, L.gate);
    load(`${p}.ffn_up.weight`, L.up);
    load(`${p}.ffn_down.weight`, L.down);
  });
}

/**
 * Full checkpoint resume: bytes -> { model with trained weights, config,
 * tokenizer }. The model is built from GGUF metadata then has every weight
 * overwritten from the file (the init rng is irrelevant).
 */
export function loadQwen3FromGGUF(
  bytes: Uint8Array,
): { model: Qwen3Model; cfg: Qwen3Config; tokenizer: BPETokenizer } {
  const g = readGGUF(bytes);
  const cfg = configFromGGUF(g);
  const model = new Qwen3Model(cfg, mulberry32(0)); // weights overwritten below
  loadWeightsFromGGUF(model, g);
  const tokenizer = BPETokenizer.fromData(tokenizerFromGGUF(g));
  return { model, cfg, tokenizer };
}
