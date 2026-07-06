// Gemma3ForCausalLM forward pass, built from the autograd ops. Uses a standard
// GQA transformer backbone (GQA, QK-RMSNorm, NEOX RoPE, tied embeddings, Muon
// param split) plus Gemma3's distinctive pieces so the export loads and runs in
// llama.cpp's `gemma3` arch (verified against llama.cpp master):
//
//   - sqrt(hidden) embedding scale on the token input (llama.cpp applies this
//     at runtime; the raw embeddings are exported unscaled),
//   - sandwich norms: an extra RMSNorm on the attention output and on the FFN
//     output, each applied BEFORE its residual add,
//   - GeGLU FFN: gelu(gate)·up (tanh-approx gelu) instead of SwiGLU,
//   - per-layer sliding-window attention (SWA layers) with a small local RoPE
//     base; the dense/global layers (every swaPattern-th) use full attention
//     and a large global RoPE base.
//
// Norm weights train in gain-frame (init 1, forward = normalize(x)·w), matching
// llama.cpp's plain rms_norm·w, so they export directly — the HF "+1" convention
// is a storage artifact of converting init-0 weights and does not apply here.

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
  Tensor,
} from "./autograd.ts";
import { type Gemma3Config, isGlobalLayer } from "./config.ts";
import { muPEmbedStd, type MuPOpts } from "./mup.ts";

interface Layer {
  attnNorm: Tensor;
  qProj: Tensor;
  kProj: Tensor;
  vProj: Tensor;
  oProj: Tensor;
  qNorm: Tensor;
  kNorm: Tensor;
  postAttnNorm: Tensor; // Gemma3: RMSNorm on attn output before the residual
  ffnNorm: Tensor;
  gate: Tensor;
  up: Tensor;
  down: Tensor;
  postFfnNorm: Tensor; // Gemma3: RMSNorm on FFN output before the residual
}

export class Gemma3Model {
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

  /** 2-D hidden matmul weights -> Muon; embeddings, head, and all norms -> AdamW aux. */
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

  /** Forward on a single token sequence -> logits [T, vocab]. */
  forward(ids: number[]): Tensor {
    const c = this.cfg;
    const T = ids.length;
    // Gemma3 scales the token embeddings by sqrt(hidden) at input (runtime-only;
    // exported embeddings stay raw). The tied readout uses the raw embeddings.
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
      attnOut = rmsNorm(attnOut, L.postAttnNorm, c.rmsEps); // post-attn norm before residual
      h = add(h, attnOut);

      const n2 = rmsNorm(h, L.ffnNorm, c.rmsEps);
      const g = gelu(linear(n2, L.gate));
      const u = linear(n2, L.up);
      let down = linear(mul(g, u), L.down);
      down = rmsNorm(down, L.postFfnNorm, c.rmsEps); // post-ffn norm before residual
      h = add(h, down);
    }

    const normed = rmsNorm(h, this.outputNorm, c.rmsEps);
    const wOut = this.output ?? this.tokenEmbd; // tied embeddings (not scaled)
    return linear(normed, wOut);
  }
}

function ones(shape: number[]): Tensor {
  const t = Tensor.zeros(shape, true);
  t.data.fill(1);
  return t;
}
