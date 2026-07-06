// muP (maximal-update parametrization), contract-safe subset for this repo.
//
// Goal: tune the learning rate on a small "base width" proxy model and transfer
// it to a wider target without re-tuning, by parametrizing init (and, in general
// muP, LR) as functions of width so activations stay O(1) as width grows.
//
// Why only init here, and why it still works — the derivation for THIS model
// (RMSNorm everywhere, tied embeddings, 1/sqrt(headDim) attention, all kept for
// llama.cpp loadability):
//   - Input side: token_embd rows feed straight into an RMSNorm, which divides
//     out their scale — so the embedding's magnitude is irrelevant on the way
//     IN. We're free to choose it for the output side.
//   - Hidden matmuls: init 1/sqrt(fan_in) already makes each linear's output
//     O(1) at any width (variance fan_in * (1/fan_in) * 1). No change needed.
//   - Attention: 1/sqrt(headDim) keeps scores O(1). Unchanged (it's part of the
//     llama.cpp-loadable contract this repo keeps).
//   - Output (tied) side: logits = normed (RMS ~1) @ token_embd^T, so their
//     scale is sqrt(hidden) * embed_std. Standard init (const 0.02) makes logits
//     grow like sqrt(hidden) with width — the thing that breaks LR transfer.
//     Setting embed_std = base_std * sqrt(baseWidth/hidden) pins the logit RMS
//     to base_std * sqrt(baseWidth) at every width. That is the muP init here.
//
// LR transfer: the aux group is width-insensitive (embeddings are O(1); norms
// are 1-D), so its Adam LR needs no width scaling. The Muon group carries the
// width-dependent matmuls, but Muon's update is spectrally normalized (an
// orthogonal matrix times sqrt(max(1,rows/cols))), so its natural LR scaling
// with width is ~flat — unlike Adam's 1/width. tests/gpu_parity or the coord
// check (tests/gradcheck muPCoordinateCheck) verifies activations stay bounded
// across widths after a few steps, i.e. constant LR transfers. So the transfer
// recipe for this repo is: muP init + keep the tuned LRs. If a future change
// (e.g. non-width-scaling ffnDim, or an untied head) breaks that, add an
// explicit per-group LR scale here and re-run the coord check.

/** muP transfer settings: init embeddings/head relative to a base width. */
export interface MuPOpts {
  baseWidth: number; // the proxy width the hyperparameters were tuned at
  baseEmbedStd?: number; // embedding init std at baseWidth (default 0.02)
}

/**
 * muP embedding/readout init std at `width`, given the value `baseStd` that was
 * chosen at `baseWidth`. Scales as 1/sqrt(width) so the tied-readout logit RMS
 * is width-independent. At width == baseWidth this returns baseStd unchanged.
 */
export function muPEmbedStd(baseStd: number, width: number, baseWidth: number): number {
  if (!(width > 0 && baseWidth > 0)) throw new Error("muP widths must be > 0");
  return baseStd * Math.sqrt(baseWidth / width);
}
