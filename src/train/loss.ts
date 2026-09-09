// One sequence to one scalar loss, choosing between the dense and the chunked
// readout. Kept in one place so the three training loops (CPU reference, the
// two GPU-resident ones) cannot drift on which path they take.

import { crossEntropy, fusedCrossEntropy, type Tensor } from "../model/autograd.ts";
import type { LanguageModel } from "../model/arch.ts";

/**
 * `lossChunk` 0 selects the dense path: `forward` to [T, vocab] logits, then
 * cross-entropy over them. Any positive value streams the vocab in chunks of
 * that width and fuses the readout matmul into the loss, which never
 * materializes [T, vocab] at the cost of recomputing that matmul in backward.
 *
 * Falls back to dense for an architecture that does not expose
 * `forwardToReadout`, so a new arch works before it opts in.
 */
export function sequenceLoss(
  model: LanguageModel,
  inputIds: number[],
  targetIds: number[],
  lossChunk = 0,
): Tensor {
  if (lossChunk > 0 && model.forwardToReadout) {
    const { hidden, readout } = model.forwardToReadout(inputIds);
    return fusedCrossEntropy(hidden, readout, targetIds, lossChunk);
  }
  return crossEntropy(model.forward(inputIds), targetIds);
}
