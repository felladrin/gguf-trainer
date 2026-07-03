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
