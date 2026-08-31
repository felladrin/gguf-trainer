// Llama: the plain pre-norm GQA transformer that most open models are.
//
// SmolLM2, TinyLlama and Mistral all fit this shape, so a model trained here
// loads in llama.cpp under the `llama` arch and can be compared directly against
// them.
//
// NOT Qwen2: llama.cpp registers `Qwen2ForCausalLM` on its own class and writes
// `general.architecture = "qwen2"` (conversion/qwen.py, MODEL_ARCH_NAMES in
// gguf-py/gguf/constants.py), which this registry does not know, so a converted
// Qwen2 or Qwen2.5 checkpoint cannot be resumed here. Qwen3 can: it has its own
// file. SmolLM3 is the same trap in the other direction, converting to `smollm3`
// despite the name.
//
// Everything Gemma3 adds, this one leaves out: no QK-norm, no sandwich norms, no
// embedding scale, one RoPE base instead of two, full attention on every layer.
// The FFN is SwiGLU, silu(gate) * up, where Gemma3 uses gelu. That difference
// list IS the diff between the two files, which is the point of this layout.
//
// Shapes are not derived from a single "hidden" number the way Gemma3's are:
// SmolLM2-135M is hidden 576 with 9 heads of 64 and 3 KV heads, which no ratio
// rule produces. Pass --heads, --kv-heads and --ffn-dim to reproduce a specific
// model exactly.

import {
  add,
  attention,
  embedding,
  linear,
  mul,
  param,
  rmsNorm,
  rope,
  silu,
} from "../model/autograd.ts";
import { Tensor } from "../model/autograd.ts";
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
  metaNumOr,
  ones,
  startGGUF,
  tensorLoader,
} from "./common.ts";

export interface LlamaConfig extends ModelConfig {
  arch: "llama";
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
  ffnNorm: Tensor;
  gate: Tensor;
  up: Tensor;
  down: Tensor;
}

/**
 * llama.cpp rotates a `llama` checkpoint with LLAMA_ROPE_TYPE_NORM, which pairs
 * dimensions (2j, 2j+1); this engine's rope() pairs (j, j+headDim/2), the NeoX
 * convention every other architecture here uses. Same rotation, different row
 * order, so the Q and K projections are reordered on the way in and back on the
 * way out. Nothing else in the model is affected: the rotation is the only place
 * a head's dimension order carries meaning.
 *
 * Getting this wrong is quiet rather than loud. Greedy generation still reads
 * fine because the argmax survives, and the CPU and GPU backends agree with each
 * other because they share the bug. What breaks is anything that needs an exact
 * match to earlier context: the base checkpoint scored ppl 5.73 on "The cat sat
 * on the mat." repeated 400 times, where llama.cpp scored 1.006 on the same file.
 * That gap is what this function closes.
 *
 * The other half of the same conversion lives in llama.cpp's converter, which does
 * the equivalent reorder when importing from Hugging Face, whose LlamaAttention uses
 * the half-split convention too: `permute()` in `conversion/llama.py` on builds that
 * split the converter up, `LlamaModel.modify_tensors` in `convert_hf_to_gguf.py`
 * before that.
 */
function reorderQK(t: Tensor, heads: number, headDim: number, toGGUF: boolean): Tensor {
  const inDim = t.shape[1];
  const half = headDim / 2;
  // The loop below writes only rows [0, heads*headDim). Anything else would be
  // left as zeros, which is a silent wrong answer rather than a failure, and
  // headDim can be derived from metadata rather than stated (configFromGGUF falls
  // back to hidden/nHeads when a checkpoint omits attention.key_length).
  if (heads * headDim !== t.shape[0] || headDim % 2 !== 0) {
    throw new Error(
      `reorderQK: ${heads} heads x ${headDim} dims does not tile ${t.shape[0]} rows evenly`,
    );
  }
  const out = Tensor.zeros(t.shape);
  for (let h = 0; h < heads; h++) {
    for (let j = 0; j < half; j++) {
      const lo = (h * headDim + j) * inDim; // our row j
      const hi = (h * headDim + j + half) * inDim; // our row j + half
      const even = (h * headDim + 2 * j) * inDim; // llama.cpp's row 2j
      const odd = (h * headDim + 2 * j + 1) * inDim; // llama.cpp's row 2j+1
      const [from0, to0, from1, to1] = toGGUF ? [lo, even, hi, odd] : [even, lo, odd, hi];
      out.data.set(t.data.subarray(from0, from0 + inDim), to0);
      out.data.set(t.data.subarray(from1, from1 + inDim), to1);
    }
  }
  return out;
}

export class LlamaModel implements LanguageModel {
  cfg: LlamaConfig;
  tokenEmbd: Tensor;
  outputNorm: Tensor;
  output: Tensor | null; // null when embeddings are tied
  layers: Layer[] = [];

  constructor(cfg: LlamaConfig, rng: () => number, mup?: MuPOpts) {
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
      ps.push(L.attnNorm, L.qProj, L.kProj, L.vProj, L.oProj, L.ffnNorm, L.gate, L.up, L.down);
    }
    return ps;
  }

  paramGroups(): { muon: Tensor[]; aux: Tensor[] } {
    const muon: Tensor[] = [];
    const aux: Tensor[] = [this.tokenEmbd, this.outputNorm];
    if (this.output) aux.push(this.output);
    for (const L of this.layers) {
      muon.push(L.qProj, L.kProj, L.vProj, L.oProj, L.gate, L.up, L.down);
      aux.push(L.attnNorm, L.ffnNorm);
    }
    return { muon, aux };
  }

  forward(ids: number[]): Tensor {
    const c = this.cfg;
    const T = ids.length;
    let h = embedding(this.tokenEmbd, ids); // no input scale, unlike Gemma3

    for (const L of this.layers) {
      const n1 = rmsNorm(h, L.attnNorm, c.rmsEps);
      let q = linear(n1, L.qProj);
      let k = linear(n1, L.kProj);
      const v = linear(n1, L.vProj);
      q = rope(q, T, c.nHeads, c.headDim, c.ropeBase);
      k = rope(k, T, c.nKVHeads, c.headDim, c.ropeBase);
      // window 0: full causal attention on every layer.
      const a = attention(q, k, v, T, c.nHeads, c.nKVHeads, c.headDim, 0);
      h = add(h, linear(a, L.oProj));

      const n2 = rmsNorm(h, L.ffnNorm, c.rmsEps);
      const g = silu(linear(n2, L.gate)); // SwiGLU, where Gemma3 uses gelu
      const u = linear(n2, L.up);
      h = add(h, linear(mul(g, u), L.down));
    }

    const normed = rmsNorm(h, this.outputNorm, c.rmsEps);
    return linear(normed, this.output ?? this.tokenEmbd);
  }
}

/** Defaults follow the common llama ratios: GQA 3:1, SwiGLU FFN ~8/3 x hidden. */
export function llamaConfig(
  vocabSize: number,
  hiddenSize: number,
  nLayers: number,
  maxSeq = 8192,
  headDim = 64,
): LlamaConfig {
  if (hiddenSize % headDim !== 0) {
    throw new Error(`hiddenSize ${hiddenSize} must be a multiple of headDim (${headDim})`);
  }
  const nHeads = hiddenSize / headDim;
  return {
    arch: "llama",
    vocabSize,
    hiddenSize,
    nLayers,
    nHeads,
    nKVHeads: defaultKVHeads(nHeads, 3),
    headDim,
    // Rounded to a multiple of 32 so q8_0/q4_0 never falls back to f16 here.
    ffnDim: Math.round((hiddenSize * 8 / 3) / 32) * 32,
    ropeBase: 100_000,
    rmsEps: 1e-5,
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
    describe:
      "key/value heads for GQA (default: a third of the query heads, rounded down to a divisor)",
  },
  {
    name: "ffn-dim",
    type: "number",
    placeholder: "N",
    describe: "FFN intermediate width (default: 8/3 x hidden, rounded to a multiple of 32)",
  },
  {
    name: "rope-base",
    type: "number",
    placeholder: "F",
    describe: "RoPE frequency base (llama default: 1e5)",
  },
  {
    name: "rms-eps",
    type: "number",
    placeholder: "F",
    describe: "RMSNorm epsilon (llama default: 1e-5)",
  },
  {
    name: "untied-embeddings",
    type: "boolean",
    describe: "give the output head its own weights instead of tying them to the embeddings",
  },
];

export const llama: Architecture<LlamaConfig> = {
  name: "llama",
  summary: "Llama: pre-norm GQA, SwiGLU, single RoPE base, full attention.",
  reference: "HuggingFaceTB/SmolLM2-135M, TinyLlama, Mistral",
  flags: FLAGS,

  configFromFlags(shape: BaseShape, v: Values): LlamaConfig {
    const base = llamaConfig(
      shape.vocabSize,
      shape.hiddenSize,
      shape.nLayers,
      shape.maxSeq,
      shape.headDim,
    );
    const nHeads = v.has("heads") ? v.num("heads") : base.nHeads;
    const nKVHeads = v.has("kv-heads") ? v.num("kv-heads") : defaultKVHeads(nHeads, 3);
    assertWholeGQA(nHeads, nKVHeads);
    return {
      ...base,
      nHeads,
      nKVHeads,
      ffnDim: v.has("ffn-dim") ? v.num("ffn-dim") : base.ffnDim,
      ropeBase: v.has("rope-base") ? v.num("rope-base") : base.ropeBase,
      rmsEps: v.has("rms-eps") ? v.num("rms-eps") : base.rmsEps,
      tieEmbeddings: !v.bool("untied-embeddings"),
    };
  },

  configFromGGUF(g: GGUFFile): LlamaConfig {
    const a = "llama";
    const hiddenSize = metaNum(g, `${a}.embedding_length`);
    const nHeads = metaNum(g, `${a}.attention.head_count`);
    return {
      arch: "llama",
      // llama.cpp derives vocab from the token list when the key is absent.
      vocabSize: metaNumOr(
        g,
        `${a}.vocab_size`,
        (g.metadata.get("tokenizer.ggml.tokens") as string[])?.length ?? 0,
      ),
      hiddenSize,
      nLayers: metaNum(g, `${a}.block_count`),
      nHeads,
      nKVHeads: metaNum(g, `${a}.attention.head_count_kv`),
      // key_length is optional in the llama arch; without it, heads split hidden.
      headDim: metaNumOr(g, `${a}.attention.key_length`, hiddenSize / nHeads),
      ffnDim: metaNum(g, `${a}.feed_forward_length`),
      ropeBase: metaNumOr(g, `${a}.rope.freq_base`, 10_000),
      rmsEps: metaNum(g, `${a}.attention.layer_norm_rms_epsilon`),
      maxSeq: metaNum(g, `${a}.context_length`),
      tieEmbeddings: !g.tensors.some((t) => t.name === "output.weight"),
    };
  },

  tinyConfig(vocabSize: number): LlamaConfig {
    // GQA with 2 query heads over 1 KV head, so the head-grouping path is real.
    return {
      arch: "llama",
      vocabSize,
      hiddenSize: 8,
      nLayers: 2,
      nHeads: 2,
      nKVHeads: 1,
      headDim: 4,
      ffnDim: 16,
      ropeBase: 100_000,
      rmsEps: 1e-5,
      maxSeq: 16,
      tieEmbeddings: true,
    };
  },

  build: (cfg, rng, mup) => new LlamaModel(cfg, rng, mup),

  paramCount(c: LlamaConfig): number {
    const attn = c.nHeads * c.headDim * c.hiddenSize +
      2 * c.nKVHeads * c.headDim * c.hiddenSize +
      c.hiddenSize * c.nHeads * c.headDim;
    const norms = 2 * c.hiddenSize; // attn, ffn
    const ffn = 3 * c.ffnDim * c.hiddenSize;
    const embed = c.vocabSize * c.hiddenSize;
    return embed + c.nLayers * (attn + norms + ffn) + c.hiddenSize +
      (c.tieEmbeddings ? 0 : embed);
  },

  describe(c: LlamaConfig): string {
    return `llama, ${c.nLayers} layers, hidden=${c.hiddenSize}, heads=${c.nHeads}/${c.nKVHeads}, ` +
      `headDim=${c.headDim}, ffn=${c.ffnDim}, ctx=${c.maxSeq}, rope=${c.ropeBase}`;
  },

  exportGGUF(model: LanguageModel, tok: TokenizerData, c: LlamaConfig, opts: ExportOpts) {
    const m = model as LlamaModel;
    const a = "llama";
    const w = startGGUF(a, tok, opts);
    w.meta_u32(`${a}.context_length`, c.maxSeq);
    w.meta_u32(`${a}.embedding_length`, c.hiddenSize);
    w.meta_u32(`${a}.block_count`, c.nLayers);
    w.meta_u32(`${a}.feed_forward_length`, c.ffnDim);
    w.meta_u32(`${a}.attention.head_count`, c.nHeads);
    w.meta_u32(`${a}.attention.head_count_kv`, c.nKVHeads);
    // Written explicitly: llama.cpp would otherwise assume hidden / heads, which
    // is wrong for shapes like SmolLM2's 576 hidden with 9 heads of 64.
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
      // Copies, not the live tensors: checkpoints are written mid-run.
      addMatrix(w, `${p}.attn_q.weight`, reorderQK(L.qProj, c.nHeads, c.headDim, true), q);
      addMatrix(w, `${p}.attn_k.weight`, reorderQK(L.kProj, c.nKVHeads, c.headDim, true), q);
      addMatrix(w, `${p}.attn_v.weight`, L.vProj, q);
      addMatrix(w, `${p}.attn_output.weight`, L.oProj, q);
      addVector(w, `${p}.ffn_norm.weight`, L.ffnNorm);
      addMatrix(w, `${p}.ffn_gate.weight`, L.gate, q);
      addMatrix(w, `${p}.ffn_up.weight`, L.up, q);
      addMatrix(w, `${p}.ffn_down.weight`, L.down, q);
    });
    return w.build();
  },

  loadWeights(model: LanguageModel, g: GGUFFile) {
    const m = model as LlamaModel;
    const c = m.cfg;
    const load = tensorLoader(g);
    load("token_embd.weight", m.tokenEmbd);
    load("output_norm.weight", m.outputNorm);
    if (m.output) load("output.weight", m.output);
    m.layers.forEach((L, i) => {
      const p = `blk.${i}`;
      load(`${p}.attn_norm.weight`, L.attnNorm);
      load(`${p}.attn_q.weight`, L.qProj);
      load(`${p}.attn_k.weight`, L.kProj);
      // NOT idempotent, and it has to run after these two loads and before nothing:
      // applying it twice returns a third row order, silently. Both callers build
      // the model on the line above, so it runs exactly once per model.
      L.qProj.data.set(reorderQK(L.qProj, c.nHeads, c.headDim, false).data);
      L.kProj.data.set(reorderQK(L.kProj, c.nKVHeads, c.headDim, false).data);
      load(`${p}.attn_v.weight`, L.vProj);
      load(`${p}.attn_output.weight`, L.oProj);
      load(`${p}.ffn_norm.weight`, L.ffnNorm);
      load(`${p}.ffn_gate.weight`, L.gate);
      load(`${p}.ffn_up.weight`, L.up);
      load(`${p}.ffn_down.weight`, L.down);
    });
  },

  configMatches(built: LlamaConfig, ckpt: LlamaConfig): string | null {
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
      { key: "tieEmbeddings", flag: "untied-embeddings", invert: true },
    ]);
  },
};
