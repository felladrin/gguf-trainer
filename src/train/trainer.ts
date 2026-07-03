// Language-model training loop: sample fixed-length windows from a token
// stream, compute next-token cross-entropy, backprop, optimizer step.
// Optimizer-agnostic — pass an AdamW or Muon instance.

import { backward, crossEntropy } from "../model/autograd.ts";
import type { Qwen3Model } from "../model/qwen3.ts";
import type { Optimizer } from "./optimizer.ts";

export interface TrainOpts {
  tokens: number[]; // full training token stream
  seqLen: number;
  steps: number;
  batchPerStep: number; // sequences accumulated per optimizer step
  optimizer: Optimizer;
  logEvery?: number;
  rng?: () => number;
  onLog?: (step: number, loss: number) => void;
}

export function trainLM(model: Qwen3Model, opts: TrainOpts): { step: number; loss: number }[] {
  const opt = opts.optimizer;
  const rng = opts.rng ?? Math.random;
  const logEvery = opts.logEvery ?? 20;
  const maxStart = opts.tokens.length - opts.seqLen - 1;
  if (maxStart <= 0) throw new Error("Not enough tokens for one training window");

  const history: { step: number; loss: number }[] = [];

  for (let step = 0; step < opts.steps; step++) {
    opt.zeroGrad();
    let lossSum = 0;

    for (let b = 0; b < opts.batchPerStep; b++) {
      const start = Math.floor(rng() * maxStart);
      const inputIds = opts.tokens.slice(start, start + opts.seqLen);
      const targetIds = opts.tokens.slice(start + 1, start + opts.seqLen + 1);

      const logits = model.forward(inputIds);
      const loss = crossEntropy(logits, targetIds);
      backward(loss, 1 / opts.batchPerStep); // average grads over the batch
      lossSum += loss.data[0];
    }

    opt.step();
    const avg = lossSum / opts.batchPerStep;
    if (step % logEvery === 0 || step === opts.steps - 1) {
      history.push({ step, loss: avg });
      opts.onLog?.(step, avg);
    }
  }

  return history;
}
