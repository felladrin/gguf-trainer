// Load a Qwen3 checkpoint back from a GGUF this framework produced — the
// inverse of export_gguf.ts. Reconstructs the config from metadata, the model
// weights via the existing dequantizer, and the tokenizer from the embedded
// vocab/merges, so a run can resume (continue training or sample) from a saved
// GGUF instead of re-initializing from scratch.
//
// Primary scope: GGUFs written by this project's buildGGUF(). Tensor names,
// metadata keys, and the [out,in]->ne[in,out] weight layout match by
// construction, so loading is name lookup + dequantize + copy. Fidelity follows
// the export quant: f32/f16 round-trip cleanly; a q4_0/q8_0 checkpoint resumes
// from its (lossy) dequantized weights — export in f16 or f32 for a faithful
// resume.
//
// External Qwen3 GGUFs (e.g. real Alibaba releases) also load as far as the
// pieces line up: configFromGGUF reads the standard qwen3.* metadata,
// tokenizerFromGGUF imports the gpt2/BPE vocab + merges + control tokens (via
// tokenizer.ggml.token_type), and loadWeightsFromGGUF dequantizes per tensor.
// Two known gaps remain for real quantized releases: (1) dequantize() decodes
// F32/F16/Q8_0/Q4_0 only — a k-quant file (Q4_K/Q6_K/…) now throws a clear
// error rather than loading garbage; (2) tokenization uses our GPT-2 split
// regex, so token boundaries may differ slightly from Qwen's own pre-tokenizer.
// Fine-tuning an external F16/Q8_0/Q4_0 Qwen3 works today; see docs/HANDOFF.md.

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

const TOKEN_TYPE_CONTROL = 3; // llama.cpp: NORMAL=1, UNKNOWN=2, CONTROL=3, USER_DEFINED=4

/** Reconstruct the tokenizer data from the GGUF's tokenizer.ggml.* metadata. */
export function tokenizerFromGGUF(g: GGUFFile): TokenizerData {
  const tokens = g.metadata.get("tokenizer.ggml.tokens");
  const merges = g.metadata.get("tokenizer.ggml.merges");
  if (!Array.isArray(tokens) || !Array.isArray(merges)) {
    throw new Error("GGUF missing tokenizer.ggml.tokens/merges");
  }
  const toks = tokens as string[];
  // Recover the control/special tokens so they re-encode atomically (ChatML,
  // <|endoftext|>, …). Prefer the token_type array (how llama.cpp and our own
  // exporter flag control tokens); fall back to the "<|…|>" shape when a GGUF
  // omits token_type. Without this, a resumed chat model — or an imported
  // external Qwen3 GGUF — would shred its special tokens back into bytes.
  const types = g.metadata.get("tokenizer.ggml.token_type");
  let specials: string[] = Array.isArray(types)
    ? toks.filter((_, i) => (types as number[])[i] === TOKEN_TYPE_CONTROL)
    : [];
  if (specials.length === 0) specials = toks.filter((t) => /^<\|.*\|>$/.test(t));
  return {
    tokens: toks,
    merges: merges as string[],
    bosId: metaNum(g, "tokenizer.ggml.bos_token_id"),
    eosId: metaNum(g, "tokenizer.ggml.eos_token_id"),
    specials: specials.length ? specials : undefined,
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
