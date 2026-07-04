// GPU training loop: the async twin of ../train/trainer.ts.
//
// The loop body is identical to trainLM — same window sampling, same rng call
// order (so a given seed produces the same batch sequence on both backends),
// same loss averaging. The one difference is structural: WebGPU readback is
// async-only, so reading the loss values and parameter gradients requires an
// `await gpu.sync()` between the backward passes and the optimizer step. That
// is why this lives in backend/ instead of complicating the reference trainer.
//
// Parameters stay authoritative on the host: the optimizer steps host arrays,
// and uploadParams() pushes them to the GPU at the top of each step.

import { backward, crossEntropy } from "../model/autograd.ts";
import type { Tensor } from "../model/autograd.ts";
import type { Qwen3Model } from "../model/qwen3.ts";
import type { TrainOpts } from "../train/trainer.ts";
import type { MuonGpu } from "./muon_gpu.ts";
import type { WebGPUBackend } from "./webgpu.ts";

export async function trainLMGpu(
  model: Qwen3Model,
  gpu: WebGPUBackend,
  opts: TrainOpts,
): Promise<{ step: number; loss: number }[]> {
  const opt = opts.optimizer;
  const rng = opts.rng ?? Math.random;
  const logEvery = opts.logEvery ?? 20;
  const maxStart = opts.tokens.length - opts.seqLen - 1;
  if (maxStart <= 0) throw new Error("Not enough tokens for one training window");

  const params = model.params();
  const history: { step: number; loss: number }[] = [];

  gpu.install();
  try {
    for (let step = 0; step < opts.steps; step++) {
      if (opts.schedule) opt.setLrScale?.(opts.schedule(step));
      opt.zeroGrad();
      gpu.uploadParams(params); // host params changed in the previous opt.step()

      const losses: Tensor[] = [];
      for (let b = 0; b < opts.batchPerStep; b++) {
        const start = Math.floor(rng() * maxStart);
        const inputIds = opts.tokens.slice(start, start + opts.seqLen);
        const targetIds = opts.tokens.slice(start + 1, start + opts.seqLen + 1);
        const loss = crossEntropy(model.forward(inputIds), targetIds);
        backward(loss, 1 / opts.batchPerStep); // average grads over the batch
        losses.push(loss);
      }

      // One readback per step: loss scalars plus accumulated parameter grads.
      await gpu.sync(losses);
      opt.step();

      let lossSum = 0;
      for (const l of losses) lossSum += l.data[0];
      const avg = lossSum / opts.batchPerStep;
      if (step % logEvery === 0 || step === opts.steps - 1) {
        history.push({ step, loss: avg });
        opts.onLog?.(step, avg);
      }
    }
  } finally {
    gpu.uninstall();
  }

  return history;
}

export interface TrainGpuResidentOpts {
  tokens: number[];
  seqLen: number;
  steps: number;
  batchPerStep: number;
  optimizer: MuonGpu;
  logEvery?: number;
  rng?: () => number;
  onLog?: (step: number, loss: number) => void;
  /** WSD (or any) lr-multiplier by step; applied before each optimizer step. */
  schedule?: (step: number) => number;
  /** Per-step wall-time split and sync readback volume, for profiling. */
  onStepTime?: (fwdBwdSyncMs: number, optimizerMs: number, readbackBytes: number) => void;
}

/**
 * The device-resident twin of trainLMGpu for the GPU Muon optimizer
 * (muon_gpu.ts): Muon-group weights, momentum, and grads stay on the GPU, so
 * per step only the aux group is uploaded and only aux grads (plus the loss
 * scalars) are read back. rng call order is identical to trainLM/trainLMGpu —
 * a given seed produces the same batch sequence on every path.
 *
 * Two syncs per step, on purpose: the first flushes forward+backward and reads
 * losses + aux grads; the optimizer dispatches are recorded after it, while
 * grads are still intact (their deferred clears only run when the NEXT
 * backward begins), and flushed by the second sync. That also makes the
 * fwd/opt wall-time split honest.
 */
export async function trainLMGpuResident(
  model: Qwen3Model,
  gpu: WebGPUBackend,
  opts: TrainGpuResidentOpts,
): Promise<{ step: number; loss: number }[]> {
  const opt = opts.optimizer;
  const rng = opts.rng ?? Math.random;
  const logEvery = opts.logEvery ?? 20;
  const maxStart = opts.tokens.length - opts.seqLen - 1;
  if (maxStart <= 0) throw new Error("Not enough tokens for one training window");

  const history: { step: number; loss: number }[] = [];

  gpu.install();
  try {
    for (let step = 0; step < opts.steps; step++) {
      const t0 = performance.now();
      if (opts.schedule) opt.setLrScale(opts.schedule(step));
      opt.zeroGrad();
      gpu.uploadParams(opt.auxParams); // Muon weights are device-resident: no upload

      const losses: Tensor[] = [];
      for (let b = 0; b < opts.batchPerStep; b++) {
        const start = Math.floor(rng() * maxStart);
        const inputIds = opts.tokens.slice(start, start + opts.seqLen);
        const targetIds = opts.tokens.slice(start + 1, start + opts.seqLen + 1);
        const loss = crossEntropy(model.forward(inputIds), targetIds);
        backward(loss, 1 / opts.batchPerStep); // average grads over the batch
        losses.push(loss);
      }

      await gpu.sync(losses); // loss scalars + aux grads; Muon grads stay on device
      const readback = gpu.lastSyncReadbackBytes;
      const t1 = performance.now();

      opt.recordStep();
      await gpu.sync(); // flush the optimizer dispatches
      opt.stepAux();
      opts.onStepTime?.(t1 - t0, performance.now() - t1, readback);

      let lossSum = 0;
      for (const l of losses) lossSum += l.data[0];
      const avg = lossSum / opts.batchPerStep;
      if (step % logEvery === 0 || step === opts.steps - 1) {
        history.push({ step, loss: avg });
        opts.onLog?.(step, avg);
      }
    }
  } finally {
    gpu.uninstall();
  }

  // One readback at the end so host arrays are again authoritative for
  // sampling and GGUF export.
  await opt.syncWeightsToHost();
  return history;
}
