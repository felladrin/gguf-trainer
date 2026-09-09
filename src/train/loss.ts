// One sequence to one scalar loss, choosing between the dense and the chunked
// readout, plus everything `--loss-chunk` has to be validated for.
//
// Kept in one place so the three training loops (CPU reference, the two
// GPU-resident ones) cannot drift on which path they take, and so the four
// commands that accept the flag cannot drift on how they check it. The checks
// live here rather than in src/cli/ because both encode facts about
// `fusedCrossEntropy`, not about argument parsing: the span ceiling exists
// because the chunk offset is baked into the kernel source, and the
// forwardToReadout check exists to override `sequenceLoss`'s own fallback.

import { crossEntropy, fusedCrossEntropy, type Tensor } from "../model/autograd.ts";
import type { LanguageModel } from "../model/arch.ts";

/**
 * Span ceiling for `--loss-chunk`. Each vocab span bakes its own offset into the
 * kernels it dispatches, so every span costs pipelines compiled before the first
 * step. No width this refuses buys memory worth having: at vocab 32768 the floor
 * is 328, and `[2048, 328]` is already 2.7 MB.
 */
export const MAX_LOSS_SPANS = 100;

/**
 * Freeze every parameter for a forward-only run.
 *
 * Nothing in eval calls backward, but the backend gives each external tensor a
 * persistent gradient accumulator on first use and `sync()` stages every one of
 * them back to the host afterwards, so each scored window copies a whole model
 * of zeros nobody reads: ~1.17 GB per window on a 293M f32 checkpoint, times 64
 * windows for an `eval-loss` default and times one per choice per item for
 * `eval-choice`. A frozen external shares one small stub instead (`entryFor` in
 * src/backend/webgpu.ts), so this drops the allocation as well as the copy.
 *
 * Call it before `uploadParams`: `entryFor` sizes the buffer on first use, and a
 * parameter frozen after that keeps the accumulator it already has.
 *
 * One-way, and deliberately without a thaw, though the failure is narrower than
 * it sounds. Measured: freeze a trained model, run any number of frozen windows,
 * thaw it and train again, and the gradients come back correct (ratio 1.0000
 * over 250757 elements). The last sync before the freeze already armed the
 * accumulator's clear while the parameter was still trainable, and that armed
 * clear still fires.
 *
 * What does not survive is a THAW on the same backend after a freeze that came
 * before the first `entryFor`. Every parameter then holds the shared 256-byte
 * stub rather than an accumulator, and `sync()` stages `t.size * 4` bytes out of
 * it: a device validation error for anything wider than 64 floats, and for a
 * narrower tensor a silent read of a buffer shared with every other frozen one.
 * So a trainable use afterwards wants a fresh backend, not just the flag back.
 * No caller evals and trains the same model in one process today.
 */
export function freezeForScoring(model: LanguageModel): void {
  for (const p of model.params()) p.requiresGrad = false;
}

/**
 * Everything `--loss-chunk` has to be checked for, in one place because three
 * commands take the flag and a fourth will forget half of it otherwise.
 *
 * Split in two so the cheap half can run before a caller reads a multi-GB
 * checkpoint: `lossChunkValueError` needs only the flag, `lossChunkModelError`
 * needs the model and the vocab.
 */
export function lossChunkValueError(lossChunk: number): string | null {
  if (!Number.isInteger(lossChunk) || lossChunk < 0) {
    return `--loss-chunk must be a whole number, 0 (dense) or positive, got ${lossChunk}`;
  }
  return null;
}

export function lossChunkModelError(
  lossChunk: number,
  vocabSize: number,
  archName: string,
  model: LanguageModel,
): string | null {
  if (lossChunk <= 0) return null;
  if (!model.forwardToReadout) {
    // The only reason to pass the flag is to get past the binding limit, and a
    // silent dense fallback walks straight back into it.
    return `--loss-chunk needs an architecture with forwardToReadout; ${archName} has none`;
  }
  const spans = Math.ceil(vocabSize / lossChunk);
  if (spans > MAX_LOSS_SPANS) {
    return `--loss-chunk ${lossChunk} splits a ${vocabSize}-token vocab into ${spans} spans, ` +
      `each compiling its own kernels before the first forward. ` +
      `Raise it to at least ${Math.ceil(vocabSize / MAX_LOSS_SPANS)}.`;
  }
  return null;
}

/**
 * `lossChunk` 0 selects the dense path: `forward` to [T, vocab] logits, then
 * cross-entropy over them. Any positive value streams the vocab in chunks of
 * that width and fuses the readout matmul into the loss, which never
 * materializes [T, vocab] at the cost of recomputing that matmul in backward.
 *
 * Falls back to dense for an architecture that does not expose
 * `forwardToReadout`, so a new arch works before it opts in when called
 * directly. The CLI deliberately makes that fallback unreachable: see
 * `lossChunkModelError`, which exists to turn it into an error, because a run
 * that passed the flag wanted the binding limit gone, not a quiet downgrade.
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
