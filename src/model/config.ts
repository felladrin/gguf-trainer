// Gemma3ForCausalLM (text) configuration — the single source of truth for a
// model built by this project. Field names mirror the concepts used in the HF
// config and llama.cpp's `gemma3` arch, but nothing here reads HF.
//
// Gemma3's distinctive pieces over a vanilla GQA transformer: sandwich norms
// (post-attention + post-FFN), GeGLU (tanh-gelu) FFN, a sqrt(hidden) embedding
// scale, per-layer sliding-window attention (the SWA speed lever), and two RoPE
// bases — a large global base on the dense layers and a small local base on the
// SWA layers. Layer `il` is GLOBAL (full attention, global RoPE base) iff
// `il % swaPattern == swaPattern - 1`; every other layer is SWA. Mirrors
// llama.cpp's `gemma3` arch so the export loads and runs there (verified against
// llama.cpp master / build 9850).

export interface Gemma3Config {
  vocabSize: number;
  hiddenSize: number; // d_model / embedding_length
  nLayers: number; // block_count
  nHeads: number; // query heads
  nKVHeads: number; // key/value heads (GQA)
  headDim: number; // attention.key_length / value_length
  ffnDim: number; // feed_forward_length (GeGLU intermediate)
  ropeBase: number; // rope.freq_base — global (dense) layers, e.g. 1e6
  ropeBaseLocal: number; // rope.freq_base_swa — SWA layers, e.g. 1e4
  rmsEps: number; // attention.layer_norm_rms_epsilon
  maxSeq: number; // context_length
  tieEmbeddings: boolean; // gemma3 ties token_embd <-> output
  slidingWindow: number; // SWA window: query t attends keys [t-W+1, t]
  swaPattern: number; // period; layer il is global iff il % swaPattern == swaPattern-1
}

/** True if layer `il` is a full-attention (global) layer under the SWA pattern. */
export function isGlobalLayer(c: Gemma3Config, il: number): boolean {
  return c.swaPattern > 0 && il % c.swaPattern === c.swaPattern - 1;
}

/**
 * Derive a Gemma3 config from width/depth with sensible ratios: GQA 2:1,
 * headDim 64, GeGLU ffn ~4× rounded to a multiple of 32 (so q8_0/q4_0 export
 * never falls back to f16 on the FFN's inner dim). `slidingWindow` and the 5:1
 * SWA:global `swaPattern` default to Gemma3's shape. `hiddenSize` must be a
 * multiple of headDim×2 (integer, even head count for the 2:1 GQA split).
 * Example sizes (vocab 8k, tied): hidden 512 × 12 layers ≈ 58M params.
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

/** A deliberately tiny Gemma3 config so the CPU backend trains in seconds. */
export function tinyGemma3Config(vocabSize: number): Gemma3Config {
  return gemma3Config(vocabSize, 128, 4, 128, 32, 64);
}

/** Rough parameter count for a Gemma3 model (2 extra norms/layer vs a vanilla GQA block). */
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
