// LoRA adapters, attached without any architecture knowing about them.
//
// `paramGroups().muon` is already exactly the set of 2-D hidden projections, so
// it is the seam: adapt those, freeze them, and register the pairs with
// `linear`. Nothing in src/arch/ changes, the trainer keeps calling the same
// three methods, and the exporter keeps writing one dense weight per tensor
// because the adapters are merged in before it runs.

import { type LoraAdapter, param, setLoraAdapters, Tensor } from "../model/autograd.ts";
import type { LanguageModel } from "../model/arch.ts";

export interface LoraHandle {
  /** The trainable tensors, in the split the optimizer expects. */
  groups: { muon: Tensor[]; aux: Tensor[] };
  rank: number;
  /** Adapted matrices, and the trainable parameter count. */
  adapted: number;
  trainable: number;
  /** Fold `B*A*scale` into each frozen base. Idempotent with `unmerge`. */
  merge(): void;
  /** Undo `merge`. Export merges, writes, and unmerges so training continues. */
  unmerge(): void;
}

/**
 * Attach rank-`rank` adapters to every matrix in the model's Muon group.
 *
 * The base weights are frozen in place (`requiresGrad = false`), which is what
 * lets the backend skip their gradient buffers and their dW products entirely.
 * `B` starts at zero, so the adapted model is exactly the base model at step 0
 * and a resumed checkpoint does not jump.
 *
 * Every adapter goes in the `aux` group, i.e. AdamW, not Muon. Muon
 * orthogonalizes the update matrix via Newton-Schulz, and running that on `A`
 * and `B` separately is not orthogonalizing `B*A`: it is a different algorithm
 * with no evidence behind it here. Choosing the optimizer that is known to work
 * on these shapes is the conservative call, and the one to revisit with a
 * measurement rather than an assumption.
 */
export function applyLora(
  model: LanguageModel,
  rank: number,
  alpha: number,
  rng: () => number,
): LoraHandle {
  if (rank <= 0) throw new Error(`applyLora: rank must be positive, got ${rank}`);
  const base = model.paramGroups().muon;
  const adapters = new Map<Tensor, LoraAdapter>();
  const aux: Tensor[] = [];
  const scale = alpha / rank;

  for (const w of base) {
    if (w.shape.length !== 2) continue; // Muon holds only matrices, but do not assume it
    const [outDim, inDim] = w.shape;
    // Standard LoRA init: A ~ N(0, 1/in) so its output is unit-ish, B = 0 so the
    // adapter contributes nothing until it has learned something.
    const a = param([rank, inDim], 1 / Math.sqrt(inDim), rng);
    const b = Tensor.zeros([outDim, rank], true);
    w.requiresGrad = false;
    adapters.set(w, { a, b, scale });
    aux.push(a, b);
  }
  setLoraAdapters(adapters);

  const fold = (sign: number) => {
    for (const [w, ad] of adapters) {
      const [outDim, inDim] = w.shape;
      for (let o = 0; o < outDim; o++) {
        for (let i = 0; i < inDim; i++) {
          let acc = 0;
          for (let r = 0; r < rank; r++) {
            acc += ad.b.data[o * rank + r] * ad.a.data[r * inDim + i];
          }
          w.data[o * inDim + i] += sign * scale * acc;
        }
      }
    }
  };

  return {
    groups: { muon: [], aux },
    rank,
    adapted: adapters.size,
    trainable: aux.reduce((n, t) => n + t.size, 0),
    merge: () => fold(1),
    unmerge: () => fold(-1),
  };
}

/** Detach every adapter, so `linear` goes back to the plain product. */
export function clearLora() {
  setLoraAdapters(new Map());
}
