// Qwen3: llama's shape with QK-RMSNorm, and an attention head size that is not
// tied to the model width.
//
// What it does differently from `llama`: each head's query and key vectors are
// RMS-normalized (per head, learned gain of headDim) BEFORE RoPE. That is the
// whole architectural difference, and it is the reason this file exists rather
// than a flag on llama.cpp's llama path: llama.cpp dispatches on the presence of
// `attn_q_norm`/`attn_k_norm`, so the tensors have to go out under the `qwen3`
// architecture to be read back.
//
// QK-norm also makes MuonClip usable here (see `qkNorms()` below), which plain
// llama cannot have.
//
// Defaults follow Qwen3-0.6B: rope base 1e6, eps 1e-6, GQA 2:1, tied embeddings.
// Its exact shape is hidden 1024 over 28 layers with 16 query heads of 128 and 8
// KV heads, which needs --head-dim 128 --heads 16 --kv-heads 8 --ffn-dim 3072.

import {
  add,
  attention,
  embedding,
  linear,
  mul,
  param,
  rmsNorm,
  rmsNormHeads,
  rope,
  silu,
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
  diffFields,
  metaNum,
  metaNumOr,
  ones,
  startGGUF,
  tensorLoader,
} from "./common.ts";

export interface Qwen3Config extends ModelConfig {
  arch: "qwen3";
  nHeads: number;
  nKVHeads: number;
  headDim: number;
  ffnDim: number;
  ropeBase: number;
  rmsEps: number;
}

interface Layer {
  attnNorm: Tensor;
  qProj: Tensor;
  kProj: Tensor;
  vProj: Tensor;
  oProj: Tensor;
  qNorm: Tensor;
  kNorm: Tensor;
  ffnNorm: Tensor;
  gate: Tensor;
  up: Tensor;
  down: Tensor;
}

export class Qwen3Model implements LanguageModel {
  cfg: Qwen3Config;
  tokenEmbd: Tensor;
  outputNorm: Tensor;
  output: Tensor | null; // null when embeddings are tied
  layers: Layer[] = [];

  constructor(cfg: Qwen3Config, rng: () => number, mup?: MuPOpts) {
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
        ffnNorm: ones([h]),
        gate: param([cfg.ffnDim, h], 1 / Math.sqrt(h), rng),
        up: param([cfg.ffnDim, h], 1 / Math.sqrt(h), rng),
        down: param([h, cfg.ffnDim], 1 / Math.sqrt(cfg.ffnDim), rng),
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
        L.ffnNorm,
        L.gate,
        L.up,
        L.down,
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
      aux.push(L.attnNorm, L.qNorm, L.kNorm, L.ffnNorm);
    }
    return { muon, aux };
  }

  qkNorms(): { headDim: number; pairs: { qNorm: Tensor; kNorm: Tensor }[] } {
    return {
      headDim: this.cfg.headDim,
      pairs: this.layers.map((L) => ({ qNorm: L.qNorm, kNorm: L.kNorm })),
    };
  }

  forward(ids: number[]): Tensor {
    const c = this.cfg;
    const T = ids.length;
    let h = embedding(this.tokenEmbd, ids); // no input scale, as in llama

    for (const L of this.layers) {
      const n1 = rmsNorm(h, L.attnNorm, c.rmsEps);
      let q = linear(n1, L.qProj);
      let k = linear(n1, L.kProj);
      const v = linear(n1, L.vProj);
      // The Qwen3 difference: per-head QK-RMSNorm, before RoPE.
      q = rmsNormHeads(q, L.qNorm, T, c.nHeads, c.headDim, c.rmsEps);
      k = rmsNormHeads(k, L.kNorm, T, c.nKVHeads, c.headDim, c.rmsEps);
      q = rope(q, T, c.nHeads, c.headDim, c.ropeBase);
      k = rope(k, T, c.nKVHeads, c.headDim, c.ropeBase);
      // window 0: full causal attention on every layer.
      const a = attention(q, k, v, T, c.nHeads, c.nKVHeads, c.headDim, 0);
      h = add(h, linear(a, L.oProj));

      const n2 = rmsNorm(h, L.ffnNorm, c.rmsEps);
      const g = silu(linear(n2, L.gate));
      const u = linear(n2, L.up);
      h = add(h, linear(mul(g, u), L.down));
    }

    const normed = rmsNorm(h, this.outputNorm, c.rmsEps);
    return linear(normed, this.output ?? this.tokenEmbd);
  }
}

/** Qwen3 ratios: GQA 2:1, SwiGLU FFN 3x hidden rounded to a multiple of 32. */
export function qwen3Config(
  vocabSize: number,
  hiddenSize: number,
  nLayers: number,
  maxSeq = 8192,
  headDim = 64,
  heads?: number,
): Qwen3Config {
  // Only the DERIVED head count needs the width to divide: published Qwen3
  // checkpoints routinely set nHeads * headDim wider than hiddenSize (LittleLamb
  // is 16 x 128 over a width of 544), and the model builds those fine.
  if (heads === undefined && hiddenSize % headDim !== 0) {
    throw new Error(
      `hiddenSize ${hiddenSize} must be a multiple of headDim (${headDim}), ` +
        `or pass --heads to set the query-head count directly`,
    );
  }
  const nHeads = heads ?? hiddenSize / headDim;
  return {
    arch: "qwen3",
    vocabSize,
    hiddenSize,
    nLayers,
    nHeads,
    nKVHeads: Math.max(1, Math.round(nHeads / 2)),
    headDim,
    // Rounded to a multiple of 32 so q8_0/q4_0 never falls back to f16 here.
    ffnDim: Math.round((hiddenSize * 3) / 32) * 32,
    ropeBase: 1_000_000,
    rmsEps: 1e-6,
    maxSeq,
    tieEmbeddings: true,
  };
}

const FLAGS: Flag[] = [
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
    describe: "key/value heads for GQA (qwen3 default: half the query heads)",
  },
  {
    name: "ffn-dim",
    type: "number",
    placeholder: "N",
    describe: "FFN intermediate width (qwen3 default: 3 x hidden, rounded to a multiple of 32)",
  },
  {
    name: "rope-base",
    type: "number",
    placeholder: "F",
    describe: "RoPE frequency base (qwen3 default: 1e6)",
  },
  {
    name: "rms-eps",
    type: "number",
    placeholder: "F",
    describe: "RMSNorm epsilon (qwen3 default: 1e-6)",
  },
  {
    name: "untied-embeddings",
    type: "boolean",
    describe: "give the output head its own weights instead of tying them to the embeddings",
  },
];

export const qwen3: Architecture<Qwen3Config> = {
  name: "qwen3",
  summary: "Qwen3: llama plus per-head QK-RMSNorm, head size independent of width.",
  reference: "Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B",
  flags: FLAGS,

  configFromFlags(shape: BaseShape, v: Values): Qwen3Config {
    const base = qwen3Config(
      shape.vocabSize,
      shape.hiddenSize,
      shape.nLayers,
      shape.maxSeq,
      shape.headDim,
      v.has("heads") ? v.num("heads") : undefined,
    );
    return {
      ...base,
      nKVHeads: v.has("kv-heads") ? v.num("kv-heads") : base.nKVHeads,
      ffnDim: v.has("ffn-dim") ? v.num("ffn-dim") : base.ffnDim,
      ropeBase: v.has("rope-base") ? v.num("rope-base") : base.ropeBase,
      rmsEps: v.has("rms-eps") ? v.num("rms-eps") : base.rmsEps,
      tieEmbeddings: !v.bool("untied-embeddings"),
    };
  },

  configFromGGUF(g: GGUFFile): Qwen3Config {
    const a = "qwen3";
    const hiddenSize = metaNum(g, `${a}.embedding_length`);
    const nHeads = metaNum(g, `${a}.attention.head_count`);
    return {
      arch: "qwen3",
      vocabSize: metaNumOr(
        g,
        `${a}.vocab_size`,
        (g.metadata.get("tokenizer.ggml.tokens") as string[])?.length ?? 0,
      ),
      hiddenSize,
      nLayers: metaNum(g, `${a}.block_count`),
      nHeads,
      nKVHeads: metaNum(g, `${a}.attention.head_count_kv`),
      // Always written by this exporter; qwen3 head size is independent of width.
      headDim: metaNumOr(g, `${a}.attention.key_length`, hiddenSize / nHeads),
      ffnDim: metaNum(g, `${a}.feed_forward_length`),
      ropeBase: metaNumOr(g, `${a}.rope.freq_base`, 1_000_000),
      rmsEps: metaNum(g, `${a}.attention.layer_norm_rms_epsilon`),
      maxSeq: metaNum(g, `${a}.context_length`),
      tieEmbeddings: !g.tensors.some((t) => t.name === "output.weight"),
    };
  },

  tinyConfig(vocabSize: number): Qwen3Config {
    return {
      arch: "qwen3",
      vocabSize,
      hiddenSize: 8,
      nLayers: 2,
      nHeads: 2,
      nKVHeads: 1,
      headDim: 4,
      ffnDim: 16,
      ropeBase: 1_000_000,
      rmsEps: 1e-6,
      maxSeq: 16,
      tieEmbeddings: true,
    };
  },

  build: (cfg, rng, mup) => new Qwen3Model(cfg, rng, mup),

  paramCount(c: Qwen3Config): number {
    const attn = c.nHeads * c.headDim * c.hiddenSize +
      2 * c.nKVHeads * c.headDim * c.hiddenSize +
      c.hiddenSize * c.nHeads * c.headDim;
    const norms = 2 * c.hiddenSize + 2 * c.headDim; // attn, ffn, q-norm, k-norm
    const ffn = 3 * c.ffnDim * c.hiddenSize;
    const embed = c.vocabSize * c.hiddenSize;
    return embed + c.nLayers * (attn + norms + ffn) + c.hiddenSize +
      (c.tieEmbeddings ? 0 : embed);
  },

  describe(c: Qwen3Config): string {
    return `qwen3, ${c.nLayers} layers, hidden=${c.hiddenSize}, heads=${c.nHeads}/${c.nKVHeads}, ` +
      `headDim=${c.headDim}, ffn=${c.ffnDim}, ctx=${c.maxSeq}, rope=${c.ropeBase}, QK-norm`;
  },

  exportGGUF(model: LanguageModel, tok: TokenizerData, c: Qwen3Config, opts: ExportOpts) {
    const m = model as Qwen3Model;
    const a = "qwen3";
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
    w.meta_u32(`${a}.rope.dimension_count`, c.headDim);
    w.meta_f32(`${a}.rope.freq_base`, c.ropeBase);
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
      addMatrix(w, `${p}.attn_output.weight`, L.oProj, q);
      addVector(w, `${p}.attn_q_norm.weight`, L.qNorm);
      addVector(w, `${p}.attn_k_norm.weight`, L.kNorm);
      addVector(w, `${p}.ffn_norm.weight`, L.ffnNorm);
      addMatrix(w, `${p}.ffn_gate.weight`, L.gate, q);
      addMatrix(w, `${p}.ffn_up.weight`, L.up, q);
      addMatrix(w, `${p}.ffn_down.weight`, L.down, q);
    });
    return w.build();
  },

  loadWeights(model: LanguageModel, g: GGUFFile) {
    const m = model as Qwen3Model;
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
      load(`${p}.ffn_norm.weight`, L.ffnNorm);
      load(`${p}.ffn_gate.weight`, L.gate);
      load(`${p}.ffn_up.weight`, L.up);
      load(`${p}.ffn_down.weight`, L.down);
    });
  },

  configMatches(built: Qwen3Config, ckpt: Qwen3Config): string | null {
    return diffFields(built, ckpt, [
      { key: "vocabSize", flag: "vocab" },
      { key: "hiddenSize", flag: "hidden" },
      { key: "nLayers", flag: "layers" },
      { key: "nHeads", flag: "heads" },
      { key: "nKVHeads", flag: "kv-heads" },
      { key: "headDim", flag: "head-dim" },
      { key: "ffnDim", flag: "ffn-dim" },
      { key: "maxSeq", flag: "max-seq" },
      { key: "rmsEps", flag: "rms-eps" },
      { key: "ropeBase", flag: "rope-base" },
    ]);
  },
};
