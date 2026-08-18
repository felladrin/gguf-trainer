// Greedy (argmax) autoregressive completion using the model's OWN forward,
// optionally with llama.cpp's repetition penalty.
//
// There is deliberately NO KV cache: each step re-forwards the trailing
// maxSeq-token window, so a run of N new tokens costs O(N^2) forwards. That is
// fine for the short completions this project needs in-process (a training
// "Sample:" line, the generative eval), and it keeps a single portable forward
// path. Real generation is done by exporting to GGUF and running llama.cpp.
//
// The backend must already be installed (gpu.install) with params uploaded
// (gpu.uploadParams) before calling; pass gpu=null to run on the CPU backend.

import type { LanguageModel } from "../model/arch.ts";
import type { WebGPUBackend } from "../backend/webgpu.ts";

/** Index of the largest value in a logits row. */
export function argmax(row: Float32Array): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

/**
 * The preset `scripts/eval-completions.sh` documents and every checkpoint of
 * this project is judged with. A base model collapses into repetition loops
 * under bare greedy decoding, which reads far worse than the model is; a gentle
 * penalty kills the loops without the topic drift a heavy one causes, and temp 0
 * keeps it reproducible.
 */
export const SAMPLE_PRESET = { repeatPenalty: 1.15, repeatLastN: 128 } as const;

export type SampleOpts = { repeatPenalty?: number; repeatLastN?: number };

/**
 * argmax after llama.cpp's repetition penalty: a token seen in `recent` has its
 * logit divided by `penalty` when positive and multiplied when negative, so the
 * nudge is downward on either side of zero. `penalty` of 1 is plain argmax.
 */
export function penalizedArgmax(
  row: Float32Array,
  recent: Iterable<number>,
  penalty: number,
): number {
  if (penalty === 1) return argmax(row);
  const seen = new Set(recent);
  let best = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < row.length; i++) {
    const v = !seen.has(i) ? row[i] : row[i] > 0 ? row[i] / penalty : row[i] * penalty;
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

/**
 * Greedy-decode up to `maxNew` tokens after `promptIds`, stopping early when a
 * token in `stop` is produced. Returns the FULL id sequence (prompt +
 * generated); slice off `promptIds.length` for just the completion.
 *
 * Defaults to bare greedy, which is what a reproducible smoke test wants. Pass
 * SAMPLE_PRESET for a quality read of a mid-training checkpoint.
 */
export async function greedyComplete(
  model: LanguageModel,
  gpu: WebGPUBackend | null,
  promptIds: number[],
  maxNew: number,
  stop: Iterable<number> = [],
  { repeatPenalty = 1, repeatLastN = 128 }: SampleOpts = {},
): Promise<number[]> {
  const ids = promptIds.slice();
  const stopSet = new Set(stop);
  const V = model.cfg.vocabSize;
  for (let i = 0; i < maxNew; i++) {
    const ctx = ids.slice(-model.cfg.maxSeq);
    const logits = model.forward(ctx);
    if (gpu) await gpu.sync([logits]);
    const base = (ctx.length - 1) * V;
    // The penalty window spans the prompt too, as it does in llama.cpp.
    const next = penalizedArgmax(
      logits.data.subarray(base, base + V),
      ids.slice(-repeatLastN),
      repeatPenalty,
    );
    ids.push(next);
    if (stopSet.has(next)) break;
  }
  return ids;
}
