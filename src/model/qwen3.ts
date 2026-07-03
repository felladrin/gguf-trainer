// Qwen3ForCausalLM forward pass, built from the autograd ops.
// Per block: pre-attn RMSNorm -> GQA attention with QK-RMSNorm + RoPE ->
// residual -> pre-MLP RMSNorm -> SwiGLU -> residual. Then final RMSNorm and
// (optionally tied) output projection.

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
  Tensor,
} from "./autograd.ts";
import type { Qwen3Config } from "./config.ts";

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

export class Qwen3Model {
  cfg: Qwen3Config;
  tokenEmbd: Tensor;
  outputNorm: Tensor;
  output: Tensor | null; // null when embeddings are tied
  layers: Layer[] = [];

  constructor(cfg: Qwen3Config, rng: () => number) {
    this.cfg = cfg;
    const h = cfg.hiddenSize;
    const qDim = cfg.nHeads * cfg.headDim;
    const kvDim = cfg.nKVHeads * cfg.headDim;

    // Standard transformer inits: embeddings ~N(0, 0.02); linear layers scaled
    // by 1/sqrt(fan_in); norm weights start at 1.
    this.tokenEmbd = param([cfg.vocabSize, h], 0.02, rng);
    this.outputNorm = ones([h]);
    this.output = cfg.tieEmbeddings ? null : param([cfg.vocabSize, h], 0.02, rng);

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

  /**
   * Split params for Muon: 2-D hidden matmul weights go to `muon`; embeddings,
   * output head, and all norms go to `aux` (AdamW). This is the standard Muon
   * recipe — embeddings/heads misbehave under orthogonalized updates.
   */
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

  /** Forward on a single token sequence -> logits [T, vocab]. */
  forward(ids: number[]): Tensor {
    const c = this.cfg;
    const T = ids.length;
    let h = embedding(this.tokenEmbd, ids); // [T, hidden]

    for (const L of this.layers) {
      const n1 = rmsNorm(h, L.attnNorm, c.rmsEps);
      let q = linear(n1, L.qProj); // [T, nHeads*headDim]
      let k = linear(n1, L.kProj); // [T, nKVHeads*headDim]
      const v = linear(n1, L.vProj);
      // Qwen3: QK-RMSNorm per head, then RoPE.
      q = rmsNormHeads(q, L.qNorm, T, c.nHeads, c.headDim, c.rmsEps);
      k = rmsNormHeads(k, L.kNorm, T, c.nKVHeads, c.headDim, c.rmsEps);
      q = rope(q, T, c.nHeads, c.headDim, c.ropeBase);
      k = rope(k, T, c.nKVHeads, c.headDim, c.ropeBase);
      const a = attention(q, k, v, T, c.nHeads, c.nKVHeads, c.headDim);
      const attnOut = linear(a, L.oProj); // [T, hidden]
      h = add(h, attnOut);

      const n2 = rmsNorm(h, L.ffnNorm, c.rmsEps);
      const g = silu(linear(n2, L.gate));
      const u = linear(n2, L.up);
      const down = linear(mul(g, u), L.down);
      h = add(h, down);
    }

    const normed = rmsNorm(h, this.outputNorm, c.rmsEps);
    const wOut = this.output ?? this.tokenEmbd; // tied embeddings
    return linear(normed, wOut); // [T, vocab]
  }
}

function ones(shape: number[]): Tensor {
  const t = Tensor.zeros(shape, true);
  t.data.fill(1);
  return t;
}
