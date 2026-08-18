// Language-model training loop: sample fixed-length windows from a token
// stream, compute next-token cross-entropy, backprop, optimizer step.
// Optimizer-agnostic: pass an AdamW or Muon instance.

import { backward, crossEntropy } from "../model/autograd.ts";
import type { LanguageModel } from "../model/arch.ts";
import type { Optimizer } from "./optimizer.ts";
import { applyQKClip } from "./qk-clip.ts";
import { toTokenSource } from "../data/tokens.ts";
import type { TokenSource } from "../data/tokens.ts";

export interface TrainOpts {
  tokens: number[] | TokenSource; // full training token stream (in-memory or disk-backed)
  seqLen: number;
  steps: number;
  batchPerStep: number; // sequences accumulated per optimizer step
  optimizer: Optimizer;
  logEvery?: number;
  rng?: () => number;
  onLog?: (step: number, loss: number) => void;
  /** WSD (or any) lr-multiplier by step; applied before each optimizer step. */
  schedule?: (step: number) => number;
  /** MuonClip/QK-logit clip threshold; when set, clip qNorm/kNorm after each step. */
  qkClipTau?: number;
  /**
   * Optional supervision mask aligned to `tokens` (1 = train on this position,
   * 0 = ignore). When present, target positions with mask 0 become -1
   * (crossEntropy's ignore-index): this is assistant-only loss for chat models.
   */
  supervised?: TokenSource;
}

/** In place: set target to -1 (ignore-index) where the supervision mask is 0. */
export function maskWindow(targetIds: number[], supervised: TokenSource, start: number): number[] {
  const m = supervised.window(start, targetIds.length);
  for (let k = 0; k < targetIds.length; k++) if (!m[k]) targetIds[k] = -1;
  return targetIds;
}

export function trainLM(model: LanguageModel, opts: TrainOpts): { step: number; loss: number }[] {
  const opt = opts.optimizer;
  const rng = opts.rng ?? Math.random;
  const logEvery = opts.logEvery ?? 20;
  const tokens = toTokenSource(opts.tokens);
  const maxStart = tokens.length - opts.seqLen - 1;
  if (maxStart <= 0) throw new Error("Not enough tokens for one training window");

  const history: { step: number; loss: number }[] = [];

  for (let step = 0; step < opts.steps; step++) {
    if (opts.schedule) opt.setLrScale?.(opts.schedule(step));
    opt.zeroGrad();
    let lossSum = 0;

    for (let b = 0; b < opts.batchPerStep; b++) {
      const start = Math.floor(rng() * maxStart);
      const inputIds = tokens.window(start, opts.seqLen);
      const targetIds = tokens.window(start + 1, opts.seqLen);
      if (opts.supervised) maskWindow(targetIds, opts.supervised, start + 1);

      const logits = model.forward(inputIds);
      const loss = crossEntropy(logits, targetIds);
      backward(loss, 1 / opts.batchPerStep); // average grads over the batch
      lossSum += loss.data[0];
    }

    opt.step();
    if (opts.qkClipTau) applyQKClip(model, opts.qkClipTau);
    const avg = lossSum / opts.batchPerStep;
    if (step % logEvery === 0 || step === opts.steps - 1) {
      history.push({ step, loss: avg });
      opts.onLog?.(step, avg);
    }
  }

  return history;
}
