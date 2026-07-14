// Greedy (argmax) autoregressive completion using the model's OWN forward.
//
// There is deliberately NO KV cache: each step re-forwards the trailing
// maxSeq-token window, so a run of N new tokens costs O(N^2) forwards. That is
// fine for the short completions this project needs in-process (a training
// "Sample:" line, the generative eval), and it keeps a single portable forward
// path. Real generation is done by exporting to GGUF and running llama.cpp.
//
// The backend must already be installed (gpu.install) with params uploaded
// (gpu.uploadParams) before calling; pass gpu=null to run on the CPU backend.

import type { Gemma3Model } from "../model/gemma3.ts";
import type { WebGPUBackend } from "../backend/webgpu.ts";

/** Index of the largest value in a logits row. */
export function argmax(row: Float32Array): number {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
  return best;
}

/**
 * Greedy-decode up to `maxNew` tokens after `promptIds`, stopping early when a
 * token in `stop` is produced. Returns the FULL id sequence (prompt +
 * generated); slice off `promptIds.length` for just the completion.
 */
export async function greedyComplete(
  model: Gemma3Model,
  gpu: WebGPUBackend | null,
  promptIds: number[],
  maxNew: number,
  stop: Iterable<number> = [],
): Promise<number[]> {
  const ids = promptIds.slice();
  const stopSet = new Set(stop);
  const V = model.cfg.vocabSize;
  for (let i = 0; i < maxNew; i++) {
    const ctx = ids.slice(-model.cfg.maxSeq);
    const logits = model.forward(ctx);
    if (gpu) await gpu.sync([logits]);
    const base = (ctx.length - 1) * V;
    const next = argmax(logits.data.subarray(base, base + V));
    ids.push(next);
    if (stopSet.has(next)) break;
  }
  return ids;
}
