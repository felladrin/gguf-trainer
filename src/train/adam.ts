// AdamW optimizer with per-parameter moment buffers and gradient clipping.

import type { Tensor } from "../model/autograd.ts";

export interface AdamOpts {
  lr: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  weightDecay?: number;
  clip?: number; // global grad-norm clip; 0 disables
}

export class AdamW {
  private params: Tensor[];
  private m: Float32Array[];
  private v: Float32Array[];
  private t = 0;
  private opts: Required<AdamOpts>;
  private baseLr: number;

  constructor(params: Tensor[], opts: AdamOpts) {
    this.params = params;
    this.opts = {
      beta1: 0.9,
      beta2: 0.999,
      eps: 1e-8,
      weightDecay: 0.0,
      clip: 1.0,
      ...opts,
    };
    this.baseLr = opts.lr;
    this.m = params.map((p) => new Float32Array(p.size));
    this.v = params.map((p) => new Float32Array(p.size));
  }

  zeroGrad() {
    for (const p of this.params) p.grad.fill(0);
  }

  /** Set the effective lr to `scale` × the constructed base lr (WSD schedule). */
  setLrScale(scale: number) {
    this.opts.lr = this.baseLr * scale;
  }

  step() {
    const o = this.opts;
    this.t += 1;

    if (o.clip > 0) {
      let sq = 0;
      for (const p of this.params) {
        for (let i = 0; i < p.grad.length; i++) sq += p.grad[i] * p.grad[i];
      }
      const norm = Math.sqrt(sq);
      if (norm > o.clip) {
        const scale = o.clip / (norm + 1e-12);
        for (const p of this.params) {
          for (let i = 0; i < p.grad.length; i++) p.grad[i] *= scale;
        }
      }
    }

    const bc1 = 1 - Math.pow(o.beta1, this.t);
    const bc2 = 1 - Math.pow(o.beta2, this.t);

    for (let pi = 0; pi < this.params.length; pi++) {
      const p = this.params[pi];
      const m = this.m[pi];
      const v = this.v[pi];
      for (let i = 0; i < p.size; i++) {
        const g = p.grad[i];
        m[i] = o.beta1 * m[i] + (1 - o.beta1) * g;
        v[i] = o.beta2 * v[i] + (1 - o.beta2) * g * g;
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        let update = mHat / (Math.sqrt(vHat) + o.eps);
        if (o.weightDecay > 0) update += o.weightDecay * p.data[i];
        p.data[i] -= o.lr * update;
      }
    }
  }
}
