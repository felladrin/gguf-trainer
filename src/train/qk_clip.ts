// QK-logit control ("MuonClip"), adapted for a QK-normed model.
//
// Kimi K2's MuonClip caps attention-logit growth by rescaling the query/key
// PROJECTION weights after each optimizer step. That lever is inert here: Qwen3
// applies QK-RMSNorm after the q/k projections, so the projection scale is
// renormalized away and the logit magnitude is set entirely by the per-head-dim
// norm weights (qNorm/kNorm, shape [headDim], shared across heads). So we
// control the logit scale directly through them — a per-layer clip, since the
// norm weights are shared across a layer's heads.
//
// Per-layer logit-scale proxy:
//   s = (1/sqrt(headDim)) * sqrt( sum_d (qNorm[d] * kNorm[d])^2 )
// This is the std of a QK-normed attention logit: RMS-normed q,k directions make
// the logit sum_d u_q[d] u_k[d] qNorm[d] kNorm[d], whose scale is that formula.
// Measured (scratch, T=128): the observed causal max logit tracks ~3.3-4.4x this
// proxy and is monotone in it across a 16x range of norm scales, so bounding s
// bounds the max. If s > tau, scale qNorm and kNorm each by sqrt(tau/s): the
// product qNorm*kNorm then scales by tau/s, so s -> tau exactly (symmetric q/k
// scaling, matching QK-Clip's sqrt(gamma) on each side).
//
// This is intentionally NOT the verbatim observed-max algorithm: with QK-norm
// the projection-weight version is a no-op, and reading the true per-step max
// would mean instrumenting the (parity-delicate) attention kernels on both
// backends. The norm-based control is data-independent, host-side, identical on
// CPU and GPU, and off by default — Qwen3's QK-norm is already the primary
// explosion guard; this is the belt-and-suspenders for scaling up.

import type { Qwen3Model } from "../model/qwen3.ts";

/** Std-scale of a QK-normed attention logit for one layer's norm weights. */
export function qkLogitScale(qNorm: Float32Array, kNorm: Float32Array, headDim: number): number {
  let s = 0;
  for (let d = 0; d < headDim; d++) {
    const p = qNorm[d] * kNorm[d];
    s += p * p;
  }
  return Math.sqrt(s) / Math.sqrt(headDim);
}

/**
 * Cap every layer's logit-scale proxy at `tau` by rescaling qNorm/kNorm in
 * place. Returns the count of layers actually clipped (for logging); a layer
 * already under `tau` is untouched. Host-side weight math — the training loop
 * uploads the updated norms on the next step (they are aux/AdamW params).
 */
export function applyQKClip(model: Qwen3Model, tau: number): number {
  if (!(tau > 0)) throw new Error("qk-clip tau must be > 0");
  const hd = model.cfg.headDim;
  let clipped = 0;
  for (const L of model.layers) {
    const s = qkLogitScale(L.qNorm.data, L.kNorm.data, hd);
    if (s > tau) {
      const f = Math.sqrt(tau / s);
      for (let d = 0; d < hd; d++) {
        L.qNorm.data[d] *= f;
        L.kNorm.data[d] *= f;
      }
      clipped++;
    }
  }
  return clipped;
}
