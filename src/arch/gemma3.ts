// Gemma3, and the reference for what an architecture file looks like.
//
// Everything about Gemma3 lives here: its config fields, its forward pass, its
// GGUF metadata and tensor names, and the flags it adds to the CLI. Nothing
// outside this file knows that sliding windows or sandwich norms exist.
//
// What Gemma3 does beyond a vanilla GQA transformer, all of it visible in
// `forward` below and verified against llama.cpp's `gemma3` arch:
//
//   - a sqrt(hidden) scale on the token embeddings at input (llama.cpp applies
//     this at runtime, so the exported embeddings stay raw),
//   - sandwich norms: an extra RMSNorm on the attention output and on the FFN
//     output, each before its residual add,
//   - GeGLU: gelu(gate) * up, where a llama-style model would use silu,
//   - QK-RMSNorm on the per-head query and key vectors,
//   - per-layer sliding-window attention with a small local RoPE base, every
//     swaPattern-th layer being dense with a large global base.
//
// Norm weights train in gain-frame (init 1, forward = normalize(x) * w), which
// matches llama.cpp's rms_norm * w, so they export directly. The HF "+1"
// convention is a storage artifact of init-0 weights and does not apply here.

import {
  add,
  attention,
  embedding,
  gelu,
  linear,
  mul,
  param,
  rmsNorm,
  rmsNormHeads,
  rope,
  scale,
} from "../model/autograd.ts";
import type { Tensor } from "../model/autograd.ts";
import { muPEmbedStd, type MuPOpts } from "../model/mup.ts";
import type {
  Architecture,
  BaseShape,
  ExportOpts,
  LanguageModel,
  ModelConfig,
} from "../model/arch.ts";
import type { GGUFFile } from "../gguf/gguf.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";
import type { Flag, Values } from "../cli/args.ts";
import {
  addMatrix,
  addVector,
  assertWholeGQA,
  defaultKVHeads,
  diffFields,
  metaNum,
  ones,
  startGGUF,
  tensorLoader,
} from "./common.ts";

export interface Gemma3Config extends ModelConfig {
  arch: "gemma3";
  nHeads: number;
  nKVHeads: number; // GQA
  headDim: number;
  ffnDim: number; // GeGLU intermediate
  ropeBase: number; // dense layers
  ropeBaseLocal: number; // sliding-window layers
  rmsEps: number;
  slidingWindow: number; // query t attends keys [t-W+1, t]
  swaPattern: number; // layer il is dense iff il % swaPattern == swaPattern - 1
}

/** True when layer `il` is a full-attention (global) layer under the SWA pattern. */
export function isGlobalLayer(c: Gemma3Config, il: number): boolean {
  return c.swaPattern > 0 && il % c.swaPattern === c.swaPattern - 1;
}

interface Layer {
  attnNorm: Tensor;
  qProj: Tensor;
  kProj: Tensor;
  vProj: Tensor;
  oProj: Tensor;
  qNorm: Tensor;
  kNorm: Tensor;
  postAttnNorm: Tensor;
  ffnNorm: Tensor;
  gate: Tensor;
  up: Tensor;
  down: Tensor;
  postFfnNorm: Tensor;
}

export class Gemma3Model implements LanguageModel {
  cfg: Gemma3Config;
  tokenEmbd: Tensor;
  outputNorm: Tensor;
  output: Tensor | null; // null when embeddings are tied
  layers: Layer[] = [];

  constructor(cfg: Gemma3Config, rng: () => number, mup?: MuPOpts) {
    this.cfg = cfg;
    const h = cfg.hiddenSize;
    const qDim = cfg.nHeads * cfg.headDim;
    const kvDim = cfg.nKVHeads * cfg.headDim;
    const embStd = mup ? muPEmbedStd(mup.baseEmbedStd ?? 0.02, h, mup.baseWidth) : 0.02;
    this.tokenEmbd = param([cfg.vocabSize, h], embStd, rng);
    this.outputNorm = ones([h]);
    this.output = cfg.tieEmbeddings ? null : param([cfg.vocabSize, h], embStd, rng);

    for (let l = 0; l < cfg.nLayers; l++) {
      this.layers.push({
        attnNorm: ones([h]),
        qProj: param([qDim, h], 1 / Math.sqrt(h), rng),
        kProj: param([kvDim, h], 1 / Math.sqrt(h), rng),
        vProj: param([kvDim, h], 1 / Math.sqrt(h), rng),
        oProj: param([h, qDim], 1 / Math.sqrt(qDim), rng),
        qNorm: ones([cfg.headDim]),
        kNorm: ones([cfg.headDim]),
        postAttnNorm: ones([h]),
        ffnNorm: ones([h]),
        gate: param([cfg.ffnDim, h], 1 / Math.sqrt(h), rng),
        up: param([cfg.ffnDim, h], 1 / Math.sqrt(h), rng),
        down: param([h, cfg.ffnDim], 1 / Math.sqrt(cfg.ffnDim), rng),
        postFfnNorm: ones([h]),
      });
    }
  }

  params(): Tensor[] {
    const ps: Tensor[] = [this.tokenEmbd, this.outputNorm];
    if (this.output) ps.push(this.output);
    for (const L of this.layers) {
      ps.push(
        L.attnNorm,
        L.qProj,
        L.kProj,
        L.vProj,
        L.oProj,
        L.qNorm,
        L.kNorm,
        L.postAttnNorm,
        L.ffnNorm,
        L.gate,
        L.up,
        L.down,
        L.postFfnNorm,
      );
    }
    return ps;
  }

  paramGroups(): { muon: Tensor[]; aux: Tensor[] } {
    const muon: Tensor[] = [];
    const aux: Tensor[] = [this.tokenEmbd, this.outputNorm];
    if (this.output) aux.push(this.output);
    for (const L of this.layers) {
      muon.push(L.qProj, L.kProj, L.vProj, L.oProj, L.gate, L.up, L.down);
      aux.push(L.attnNorm, L.qNorm, L.kNorm, L.postAttnNorm, L.ffnNorm, L.postFfnNorm);
    }
    return { muon, aux };
  }

  /** Gemma3 has QK-RMSNorm, so MuonClip can bound its attention logits. */
  qkNorms(): { headDim: number; pairs: { qNorm: Tensor; kNorm: Tensor }[] } {
    return {
      headDim: this.cfg.headDim,
      pairs: this.layers.map((L) => ({ qNorm: L.qNorm, kNorm: L.kNorm })),
    };
  }

  forward(ids: number[]): Tensor {
    const c = this.cfg;
    const T = ids.length;
    // The input scale is runtime-only; the tied readout below uses raw embeddings.
    let h = scale(embedding(this.tokenEmbd, ids), Math.sqrt(c.hiddenSize));

    for (let il = 0; il < this.layers.length; il++) {
      const L = this.layers[il];
      const global = isGlobalLayer(c, il);
      const window = global ? 0 : c.slidingWindow;
      const ropeB = global ? c.ropeBase : c.ropeBaseLocal;

      const n1 = rmsNorm(h, L.attnNorm, c.rmsEps);
      let q = linear(n1, L.qProj);
      let k = linear(n1, L.kProj);
      const v = linear(n1, L.vProj);
      q = rmsNormHeads(q, L.qNorm, T, c.nHeads, c.headDim, c.rmsEps);
      k = rmsNormHeads(k, L.kNorm, T, c.nKVHeads, c.headDim, c.rmsEps);
      q = rope(q, T, c.nHeads, c.headDim, ropeB);
      k = rope(k, T, c.nKVHeads, c.headDim, ropeB);
      const a = attention(q, k, v, T, c.nHeads, c.nKVHeads, c.headDim, window);
      let attnOut = linear(a, L.oProj);
      attnOut = rmsNorm(attnOut, L.postAttnNorm, c.rmsEps);
      h = add(h, attnOut);

      const n2 = rmsNorm(h, L.ffnNorm, c.rmsEps);
      const g = gelu(linear(n2, L.gate));
      const u = linear(n2, L.up);
      let down = linear(mul(g, u), L.down);
      down = rmsNorm(down, L.postFfnNorm, c.rmsEps);
      h = add(h, down);
    }

    const normed = rmsNorm(h, this.outputNorm, c.rmsEps);
    return linear(normed, this.output ?? this.tokenEmbd);
  }
}

/**
 * Derive a full config from width and depth with Gemma3's ratios: GQA 2:1, FFN
 * ~4x rounded to a multiple of 32 (so q8_0/q4_0 never falls back to f16 on the
 * FFN's inner dimension), 5:1 sliding-window to dense.
 */
export function gemma3Config(
  vocabSize: number,
  hiddenSize: number,
  nLayers: number,
  maxSeq = 8192,
  headDim = 64,
  slidingWindow = 1024,
  heads?: number,
): Gemma3Config {
  // Only the DERIVED head count needs the width to divide: with an explicit
  // --heads the attention block is nHeads * headDim wide and the oProj maps it
  // back to hiddenSize, so the division is not required.
  if (heads === undefined && hiddenSize % (headDim * 2) !== 0) {
    throw new Error(
      `hiddenSize ${hiddenSize} must be a multiple of headDim*2 (${headDim * 2}), ` +
        `or pass --heads to set the query-head count directly`,
    );
  }
  const nHeads = heads ?? hiddenSize / headDim;
  return {
    arch: "gemma3",
    vocabSize,
    hiddenSize,
    nLayers,
    nHeads,
    nKVHeads: defaultKVHeads(nHeads, 2),
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

/** A deliberately tiny config, so the CPU backend trains it in seconds. */
export function tinyGemma3Config(vocabSize: number): Gemma3Config {
  return gemma3Config(vocabSize, 128, 4, 128, 32, 64);
}

const FLAGS: Flag[] = [
  {
    name: "window",
    type: "number",
    default: 1024,
    describe: "gemma3: sliding-window size on the local-attention layers",
  },
  {
    name: "swa-pattern",
    type: "number",
    default: 6,
    describe: "gemma3: one dense layer every N layers, the rest sliding-window",
  },
  {
    name: "rope-base",
    type: "number",
    placeholder: "F",
    describe: "RoPE frequency base on the dense layers (gemma3 default: 1e6)",
  },
  {
    name: "rms-eps",
    type: "number",
    placeholder: "F",
    describe: "RMSNorm epsilon (gemma3 default: 1e-6)",
  },
  {
    name: "rope-base-local",
    type: "number",
    default: 10_000,
    describe: "gemma3: RoPE frequency base on the sliding-window layers",
  },
  {
    name: "heads",
    type: "number",
    placeholder: "N",
    describe: "query heads (default: hidden / head-dim)",
  },
  {
    name: "kv-heads",
    type: "number",
    placeholder: "N",
    describe: "key/value heads for GQA (default: half the query heads)",
  },
  {
    name: "ffn-dim",
    type: "number",
    placeholder: "N",
    describe: "FFN intermediate width (default: 4x hidden, rounded to a multiple of 32)",
  },
  {
    name: "untied-embeddings",
    type: "boolean",
    describe: "give the output head its own weights instead of tying them to the embeddings",
  },
];

export const gemma3: Architecture<Gemma3Config> = {
  name: "gemma3",
  summary: "Gemma3: GQA + QK-norm, sandwich norms, GeGLU, sliding-window attention.",
  reference: "google/gemma-3-*, and Felladrin/Minueza-3-95M-Base",
  flags: FLAGS,

  configFromFlags(shape: BaseShape, v: Values): Gemma3Config {
    const base = gemma3Config(
      shape.vocabSize,
      shape.hiddenSize,
      shape.nLayers,
      shape.maxSeq,
      shape.headDim,
      v.num("window"),
      v.has("heads") ? v.num("heads") : undefined,
    );
    const nKVHeads = v.has("kv-heads") ? v.num("kv-heads") : base.nKVHeads;
    assertWholeGQA(base.nHeads, nKVHeads);
    return {
      ...base,
      swaPattern: v.num("swa-pattern"),
      ropeBase: v.has("rope-base") ? v.num("rope-base") : base.ropeBase,
      rmsEps: v.has("rms-eps") ? v.num("rms-eps") : base.rmsEps,
      ropeBaseLocal: v.num("rope-base-local"),
      nKVHeads,
      ffnDim: v.has("ffn-dim") ? v.num("ffn-dim") : base.ffnDim,
      tieEmbeddings: !v.bool("untied-embeddings"),
    };
  },

  configFromGGUF(g: GGUFFile): Gemma3Config {
    const a = "gemma3";
    // Tying is implicit: the exporter omits output.weight when embeddings are tied.
    const tieEmbeddings = !g.tensors.some((t) => t.name === "output.weight");
    return {
      arch: "gemma3",
      vocabSize: metaNum(g, `${a}.vocab_size`),
      hiddenSize: metaNum(g, `${a}.embedding_length`),
      nLayers: metaNum(g, `${a}.block_count`),
      nHeads: metaNum(g, `${a}.attention.head_count`),
      nKVHeads: metaNum(g, `${a}.attention.head_count_kv`),
      headDim: metaNum(g, `${a}.attention.key_length`),
      ffnDim: metaNum(g, `${a}.feed_forward_length`),
      ropeBase: metaNum(g, `${a}.rope.freq_base`),
      ropeBaseLocal: metaNum(g, `${a}.rope.freq_base_swa`),
      rmsEps: metaNum(g, `${a}.attention.layer_norm_rms_epsilon`),
      maxSeq: metaNum(g, `${a}.context_length`),
      tieEmbeddings,
      slidingWindow: metaNum(g, `${a}.attention.sliding_window`),
      swaPattern: metaNum(g, `${a}.attention.sliding_window_pattern`),
    };
  },

  tinyConfig(vocabSize: number): Gemma3Config {
    // swaPattern 2 with 2 layers means layer 0 is sliding-window and layer 1 is
    // dense, so a test sequence longer than the window exercises both attention
    // paths and both RoPE bases.
    return {
      arch: "gemma3",
      vocabSize,
      hiddenSize: 8,
      nLayers: 2,
      nHeads: 2,
      nKVHeads: 1,
      headDim: 4,
      ffnDim: 16,
      ropeBase: 1_000_000,
      ropeBaseLocal: 10_000,
      rmsEps: 1e-6,
      maxSeq: 16,
      tieEmbeddings: true,
      slidingWindow: 4,
      swaPattern: 2,
    };
  },

  build: (cfg, rng, mup) => new Gemma3Model(cfg, rng, mup),

  paramCount(c: Gemma3Config): number {
    const attn = c.nHeads * c.headDim * c.hiddenSize + // q
      2 * c.nKVHeads * c.headDim * c.hiddenSize + // k, v
      c.hiddenSize * c.nHeads * c.headDim; // o
    const qkNorm = 2 * c.headDim;
    const norms = 4 * c.hiddenSize; // attn, post-attn, ffn, post-ffn
    const ffn = 3 * c.ffnDim * c.hiddenSize;
    const perLayer = attn + qkNorm + norms + ffn;
    const embed = c.vocabSize * c.hiddenSize;
    return embed + c.nLayers * perLayer + c.hiddenSize + (c.tieEmbeddings ? 0 : embed);
  },

  describe(c: Gemma3Config): string {
    return `gemma3, ${c.nLayers} layers, hidden=${c.hiddenSize}, heads=${c.nHeads}/${c.nKVHeads}, ` +
      `headDim=${c.headDim}, ffn=${c.ffnDim}, ctx=${c.maxSeq}, window=${c.slidingWindow}`;
  },

  exportGGUF(model: LanguageModel, tok: TokenizerData, c: Gemma3Config, opts: ExportOpts) {
    const m = model as Gemma3Model;
    const a = "gemma3";
    const w = startGGUF(a, tok, opts);
    w.meta_u32(`${a}.context_length`, c.maxSeq);
    w.meta_u32(`${a}.embedding_length`, c.hiddenSize);
    w.meta_u32(`${a}.block_count`, c.nLayers);
    w.meta_u32(`${a}.feed_forward_length`, c.ffnDim);
    w.meta_u32(`${a}.attention.head_count`, c.nHeads);
    w.meta_u32(`${a}.attention.head_count_kv`, c.nKVHeads);
    w.meta_u32(`${a}.attention.key_length`, c.headDim);
    w.meta_u32(`${a}.attention.value_length`, c.headDim);
    w.meta_f32(`${a}.attention.layer_norm_rms_epsilon`, c.rmsEps);
    w.meta_u32(`${a}.attention.sliding_window`, c.slidingWindow);
    w.meta_u32(`${a}.attention.sliding_window_pattern`, c.swaPattern);
    w.meta_f32(`${a}.rope.freq_base`, c.ropeBase);
    w.meta_f32(`${a}.rope.freq_base_swa`, c.ropeBaseLocal);
    w.meta_u32(`${a}.vocab_size`, c.vocabSize);

    const q = opts.quant;
    addMatrix(w, "token_embd.weight", m.tokenEmbd, q);
    addVector(w, "output_norm.weight", m.outputNorm);
    if (m.output) addMatrix(w, "output.weight", m.output, q);

    m.layers.forEach((L, i) => {
      const p = `blk.${i}`;
      addVector(w, `${p}.attn_norm.weight`, L.attnNorm);
      addMatrix(w, `${p}.attn_q.weight`, L.qProj, q);
      addMatrix(w, `${p}.attn_k.weight`, L.kProj, q);
      addMatrix(w, `${p}.attn_v.weight`, L.vProj, q);
      addVector(w, `${p}.attn_q_norm.weight`, L.qNorm);
      addVector(w, `${p}.attn_k_norm.weight`, L.kNorm);
      addMatrix(w, `${p}.attn_output.weight`, L.oProj, q);
      addVector(w, `${p}.post_attention_norm.weight`, L.postAttnNorm);
      addVector(w, `${p}.ffn_norm.weight`, L.ffnNorm);
      addMatrix(w, `${p}.ffn_gate.weight`, L.gate, q);
      addMatrix(w, `${p}.ffn_up.weight`, L.up, q);
      addMatrix(w, `${p}.ffn_down.weight`, L.down, q);
      addVector(w, `${p}.post_ffw_norm.weight`, L.postFfnNorm);
    });
    return w.build();
  },

  loadWeights(model: LanguageModel, g: GGUFFile) {
    const m = model as Gemma3Model;
    const load = tensorLoader(g);
    load("token_embd.weight", m.tokenEmbd);
    load("output_norm.weight", m.outputNorm);
    if (m.output) load("output.weight", m.output);
    m.layers.forEach((L, i) => {
      const p = `blk.${i}`;
      load(`${p}.attn_norm.weight`, L.attnNorm);
      load(`${p}.attn_q.weight`, L.qProj);
      load(`${p}.attn_k.weight`, L.kProj);
      load(`${p}.attn_v.weight`, L.vProj);
      load(`${p}.attn_output.weight`, L.oProj);
      load(`${p}.attn_q_norm.weight`, L.qNorm);
      load(`${p}.attn_k_norm.weight`, L.kNorm);
      load(`${p}.post_attention_norm.weight`, L.postAttnNorm);
      load(`${p}.ffn_norm.weight`, L.ffnNorm);
      load(`${p}.ffn_gate.weight`, L.gate);
      load(`${p}.ffn_up.weight`, L.up);
      load(`${p}.ffn_down.weight`, L.down);
      load(`${p}.post_ffw_norm.weight`, L.postFfnNorm);
    });
  },

  configMatches(built: Gemma3Config, ckpt: Gemma3Config): string | null {
    return diffFields(built, ckpt, [
      { key: "vocabSize", flag: "vocab" },
      { key: "hiddenSize", flag: "hidden" },
      { key: "nLayers", flag: "layers" },
      // head-dim before heads: when the head count is derived from
      // hidden / head-dim, a wrong --head-dim is the cause, so report the
      // flag the user actually set rather than the derived --heads.
      { key: "headDim", flag: "head-dim" },
      { key: "nHeads", flag: "heads" },
      { key: "nKVHeads", flag: "kv-heads" },
      { key: "ffnDim", flag: "ffn-dim" },
      { key: "maxSeq", flag: "max-seq" },
      { key: "slidingWindow", flag: "window" },
      { key: "swaPattern", flag: "swa-pattern" },
      { key: "rmsEps", flag: "rms-eps" },
      { key: "ropeBase", flag: "rope-base" },
      { key: "ropeBaseLocal", flag: "rope-base-local" },
    ]);
  },
};
