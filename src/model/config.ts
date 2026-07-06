// Qwen3ForCausalLM configuration. Field names mirror the concepts used in the
// HF config and llama.cpp's qwen3 arch, but nothing here reads HF — this is
// the single source of truth for a model built by this project.

export interface Qwen3Config {
  vocabSize: number;
  hiddenSize: number; // d_model / embedding_length
  nLayers: number; // block_count
  nHeads: number; // query heads
  nKVHeads: number; // key/value heads (GQA)
  headDim: number; // attention.key_length / value_length (explicit in qwen3)
  ffnDim: number; // feed_forward_length (SwiGLU intermediate)
  ropeBase: number; // rope.freq_base
  rmsEps: number; // attention.layer_norm_rms_epsilon
  maxSeq: number; // context_length
  tieEmbeddings: boolean; // qwen3 small models tie token_embd <-> output
}

/** A deliberately tiny config so the CPU backend trains in seconds. */
export function tinyConfig(vocabSize: number): Qwen3Config {
  return {
    vocabSize,
    hiddenSize: 128,
    nLayers: 4,
    nHeads: 4,
    nKVHeads: 2,
    headDim: 32,
    ffnDim: 320,
    ropeBase: 10000,
    rmsEps: 1e-6,
    maxSeq: 128,
    tieEmbeddings: true,
  };
}

/**
 * A config for real (non-toy) runs: derive the attention/FFN dims from a chosen
 * width and depth with Qwen3-like ratios — headDim 64, GQA 2:1, SwiGLU ffn 3×
 * rounded to a multiple of 32 (so q8_0/q4_0 export never falls back to f16 on
 * the FFN's inner dim). `hiddenSize` must be a multiple of headDim×2 (integer,
 * even head count for the 2:1 GQA split). Example sizes (vocab 8k, tied):
 * hidden 384 × 6 layers ≈ 14M params; hidden 512 × 8 ≈ 33M.
 */
export function scaleConfig(
  vocabSize: number,
  hiddenSize: number,
  nLayers: number,
  maxSeq = 512,
  headDim = 64,
): Qwen3Config {
  if (hiddenSize % (headDim * 2) !== 0) {
    throw new Error(`hiddenSize ${hiddenSize} must be a multiple of headDim*2 (${headDim * 2})`);
  }
  const nHeads = hiddenSize / headDim;
  return {
    vocabSize,
    hiddenSize,
    nLayers,
    nHeads,
    nKVHeads: nHeads / 2,
    headDim,
    ffnDim: Math.round((hiddenSize * 3) / 32) * 32,
    ropeBase: 10000,
    rmsEps: 1e-6,
    maxSeq,
    tieEmbeddings: true,
  };
}

/**
 * Gemma3 (text) configuration. Same transformer backbone as Qwen3 (GQA,
 * QK-RMSNorm, NEOX RoPE, tied embeddings) plus Gemma3's distinctive pieces:
 * sandwich norms (post-attention + post-FFN), GeGLU (tanh-gelu) FFN, a
 * sqrt(hidden) embedding scale, per-layer sliding-window attention (the SWA
 * speed lever), and two RoPE bases — a large global base on the dense layers
 * and a small local base on the SWA layers. Layer `il` is GLOBAL (full
 * attention, global RoPE base) iff `il % swaPattern == swaPattern - 1`; every
 * other layer is SWA. Mirrors llama.cpp's `gemma3` arch so the export loads and
 * runs there (verified against llama.cpp master / build 9850).
 */
export interface Gemma3Config {
  vocabSize: number;
  hiddenSize: number;
  nLayers: number;
  nHeads: number;
  nKVHeads: number;
  headDim: number;
  ffnDim: number;
  ropeBase: number; // global (dense) layers, e.g. 1e6
  ropeBaseLocal: number; // SWA layers, e.g. 1e4
  rmsEps: number;
  maxSeq: number;
  tieEmbeddings: boolean;
  slidingWindow: number; // SWA window: query t attends keys [t-W+1, t]
  swaPattern: number; // period; layer il is global iff il % swaPattern == swaPattern-1
}

/** True if layer `il` is a full-attention (global) layer under the SWA pattern. */
export function isGlobalLayer(c: Gemma3Config, il: number): boolean {
  return c.swaPattern > 0 && il % c.swaPattern === c.swaPattern - 1;
}

/**
 * Derive a Gemma3 config from width/depth with sensible ratios: GQA 2:1,
 * headDim 64, GeGLU ffn ~4× rounded to a multiple of 32. `slidingWindow` and
 * the 5:1 SWA:global `swaPattern` default to Gemma3's shape.
 */
export function gemma3Config(
  vocabSize: number,
  hiddenSize: number,
  nLayers: number,
  maxSeq = 8192,
  headDim = 64,
  slidingWindow = 1024,
): Gemma3Config {
  if (hiddenSize % (headDim * 2) !== 0) {
    throw new Error(`hiddenSize ${hiddenSize} must be a multiple of headDim*2 (${headDim * 2})`);
  }
  const nHeads = hiddenSize / headDim;
  return {
    vocabSize,
    hiddenSize,
    nLayers,
    nHeads,
    nKVHeads: nHeads / 2,
    headDim,
    ffnDim: Math.round((hiddenSize * 4) / 32) * 32,
    ropeBase: 1_000_000,
    ropeBaseLocal: 10_000,
    rmsEps: 1e-6,
    maxSeq,
    tieEmbeddings: true,
    slidingWindow,
    swaPattern: 6,
  };
}

/** Rough parameter count for a Gemma3 model (2 extra norms/layer vs Qwen3). */
export function gemma3ParamCount(c: Gemma3Config): number {
  const attnQ = c.nHeads * c.headDim * c.hiddenSize;
  const attnKV = 2 * c.nKVHeads * c.headDim * c.hiddenSize;
  const attnO = c.hiddenSize * c.nHeads * c.headDim;
  const qkNorm = 2 * c.headDim;
  const norms = 4 * c.hiddenSize; // attn, post-attn, ffn, post-ffn
  const ffn = 3 * c.ffnDim * c.hiddenSize;
  const perLayer = attnQ + attnKV + attnO + qkNorm + norms + ffn;
  const embed = c.vocabSize * c.hiddenSize;
  const outHead = c.tieEmbeddings ? 0 : c.vocabSize * c.hiddenSize;
  return embed + c.nLayers * perLayer + c.hiddenSize + outHead;
}

/** Rough parameter count for reporting. */
export function paramCount(c: Qwen3Config): number {
  const attnQ = c.nHeads * c.headDim * c.hiddenSize;
  const attnK = c.nKVHeads * c.headDim * c.hiddenSize;
  const attnV = attnK;
  const attnO = c.hiddenSize * c.nHeads * c.headDim;
  const qkNorm = 2 * c.headDim;
  const norms = 2 * c.hiddenSize;
  const ffn = 3 * c.ffnDim * c.hiddenSize;
  const perLayer = attnQ + attnK + attnV + attnO + qkNorm + norms + ffn;
  const embed = c.vocabSize * c.hiddenSize;
  const outHead = c.tieEmbeddings ? 0 : c.vocabSize * c.hiddenSize;
  return embed + c.nLayers * perLayer + c.hiddenSize + outHead;
}
