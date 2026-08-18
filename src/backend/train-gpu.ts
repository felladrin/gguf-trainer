// GPU training loop: the async twin of ../train/trainer.ts.
//
// The loop body is identical to trainLM: same window sampling, same rng call
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
import type { LanguageModel } from "../model/arch.ts";
import { maskWindow, type TrainOpts } from "../train/trainer.ts";
import { applyQKClip } from "../train/qk-clip.ts";
import { toTokenSource } from "../data/tokens.ts";
import type { TokenSource } from "../data/tokens.ts";
import type { MuonGpu } from "./muon-gpu.ts";
import type { WebGPUBackend } from "./webgpu.ts";

export async function trainLMGpu(
  model: LanguageModel,
  gpu: WebGPUBackend,
  opts: TrainOpts,
): Promise<{ step: number; loss: number }[]> {
  const opt = opts.optimizer;
  const rng = opts.rng ?? Math.random;
  const logEvery = opts.logEvery ?? 20;
  const params = model.params();
  const tokens = toTokenSource(opts.tokens);
  const maxStart = tokens.length - opts.seqLen - 1;
  if (maxStart <= 0) throw new Error("Not enough tokens for one training window");

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
        const inputIds = tokens.window(start, opts.seqLen);
        const targetIds = tokens.window(start + 1, opts.seqLen);
        if (opts.supervised) maskWindow(targetIds, opts.supervised, start + 1);
        const loss = crossEntropy(model.forward(inputIds), targetIds);
        backward(loss, 1 / opts.batchPerStep); // average grads over the batch
        losses.push(loss);
      }

      // One readback per step: loss scalars plus accumulated parameter grads.
      await gpu.sync(losses);
      opt.step();
      if (opts.qkClipTau) applyQKClip(model, opts.qkClipTau);

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
  tokens: number[] | TokenSource;
  seqLen: number;
  steps: number;
  batchPerStep: number;
  optimizer: MuonGpu;
  logEvery?: number;
  rng?: () => number;
  onLog?: (step: number, loss: number) => void;
  /** WSD (or any) lr-multiplier by step; applied before each optimizer step. */
  schedule?: (step: number) => number;
  /** MuonClip/QK-logit clip threshold; when set, clip qNorm/kNorm after each step. */
  qkClipTau?: number;
  /** Per-step wall-time split and sync readback volume, for profiling. */
  onStepTime?: (fwdBwdSyncMs: number, optimizerMs: number, readbackBytes: number) => void;
  /**
   * Optional supervision mask aligned to `tokens` (1 = train, 0 = ignore).
   * Masked target positions become -1 (crossEntropy ignore-index): this is
   * assistant-only loss for chat models. Same semantics as TrainOpts.supervised.
   */
  supervised?: TokenSource;
  /**
   * Mid-run checkpoint: every `checkpointEvery` steps, sync the device-resident
   * weights back to the host and call `onCheckpoint` (e.g. to export a GGUF), so
   * a long run survives interruption. Skipped when either is unset.
   */
  checkpointEvery?: number;
  onCheckpoint?: (step: number) => void | Promise<void>;
  /**
   * WSD decay-phase data injection (MiniCPM / Xmodel-2 trick): from `injectFromStep`
   * on, each micro-batch draws from `injectSource` instead of the main tokens with
   * probability `injectFrac`. Used to fold a little instruct/ChatML data into the
   * cooldown so the base emerges more instructable. All unset -> no injection and
   * the rng stream is byte-identical to before (the decision rng() is only drawn
   * when injection is actually active), so CPU/GPU parity is unaffected.
   */
  injectSource?: TokenSource;
  injectFrac?: number; // 0..1 probability per micro-batch during the window
  injectFromStep?: number; // first step injection is eligible (e.g. cooldown start)
  /**
   * Recycle each micro-batch's transient GPU buffers at the micro-batch boundary
   * instead of holding all `batchPerStep` micro-batches' activations until the
   * end-of-step sync. Cuts peak VRAM to ~one micro-batch, which is what lets
   * batch>=2 fit at long context (e.g. seqLen 8192, where the crossEntropy
   * probs buffer alone is seqLen*vocab*4 ~= 1GB per micro-batch). Costs one GPU
   * fence per micro-batch (loses some CPU/GPU overlap), so it is OFF by default
   * and worth enabling only when memory-bound. Numerically identical to off:
   * only buffer-recycling timing changes (gated by tests/gpu-parity.ts).
   */
  reclaimTransients?: boolean;
}

/**
 * The device-resident twin of trainLMGpu for the GPU optimizer (muon-gpu.ts +
 * adamw-gpu.ts): BOTH param groups, Muon on the hidden matmuls, AdamW on the
 * aux group: keep their weights, moments, and grads on the GPU, so after
 * warm-up only the loss scalars are read back (no per-step aux upload or grad
 * readback). rng call order is identical to trainLM/trainLMGpu: a given seed
 * produces the same batch sequence on every path.
 *
 * Two syncs per step, on purpose: the first flushes forward+backward and reads
 * the loss scalars; the optimizer dispatches are recorded after it, while grads
 * are still intact (their deferred clears only run when the NEXT backward
 * begins), and flushed by the second sync. That also makes the fwd/opt
 * wall-time split honest.
 */
export async function trainLMGpuResident(
  model: LanguageModel,
  gpu: WebGPUBackend,
  opts: TrainGpuResidentOpts,
): Promise<{ step: number; loss: number }[]> {
  const opt = opts.optimizer;
  const rng = opts.rng ?? Math.random;
  const logEvery = opts.logEvery ?? 20;
  const tokens = toTokenSource(opts.tokens);
  const maxStart = tokens.length - opts.seqLen - 1;
  if (maxStart <= 0) throw new Error("Not enough tokens for one training window");

  // WSD decay-phase injection (all optional; off -> no behavior/rng change).
  const inject = opts.injectSource;
  const injectFrac = opts.injectFrac ?? 0;
  const injectFrom = opts.injectFromStep ?? 0;
  const injectMaxStart = inject ? inject.length - opts.seqLen - 1 : 0;

  const history: { step: number; loss: number }[] = [];

  // MuonClip reads/rewrites qNorm/kNorm on the host, but they are device-resident
  // now (AdamWGpu), so when clipping is on we read just those tiny norm tensors
  // back in the optimizer-flush sync, clip on the host, and re-upload, no extra
  // sync, and the round-trip is a few KB regardless of model size.
  const normTensors: Tensor[] = [];
  if (opts.qkClipTau) {
    for (const L of model.qkNorms?.().pairs ?? []) normTensors.push(L.qNorm, L.kNorm);
  }

  gpu.install();
  try {
    for (let step = 0; step < opts.steps; step++) {
      const t0 = performance.now();
      if (opts.schedule) opt.setLrScale(opts.schedule(step));
      opt.zeroGrad(); // both groups device-resident: no per-step upload

      const losses: Tensor[] = [];
      for (let b = 0; b < opts.batchPerStep; b++) {
        // Draw the inject decision (extra rng()) ONLY while injection is active,
        // so with injection off the rng stream is unchanged (parity preserved).
        const useInject = inject !== undefined && step >= injectFrom && injectFrac > 0 &&
          injectMaxStart > 0 && rng() < injectFrac;
        const src = useInject ? inject! : tokens;
        const start = Math.floor(rng() * (useInject ? injectMaxStart : maxStart));
        const inputIds = src.window(start, opts.seqLen);
        const targetIds = src.window(start + 1, opts.seqLen);
        if (opts.supervised && !useInject) maskWindow(targetIds, opts.supervised, start + 1);
        const loss = crossEntropy(model.forward(inputIds), targetIds);
        backward(loss, 1 / opts.batchPerStep); // average grads over the batch
        losses.push(loss);
        // Free this micro-batch's activations before the next one so peak VRAM
        // stays at ~one micro-batch. Skipped on the last (the end-of-step sync
        // reclaims it anyway). `losses` are kept: their data is read below.
        if (opts.reclaimTransients && b < opts.batchPerStep - 1) {
          await gpu.reclaimStepTransients(losses);
        }
      }

      await gpu.sync(losses); // loss scalars only; both groups' grads stay on device
      const readback = gpu.lastSyncReadbackBytes;
      const t1 = performance.now();

      opt.recordStep();
      await gpu.sync(normTensors); // flush the optimizer; read norms iff clipping
      if (opts.qkClipTau) {
        applyQKClip(model, opts.qkClipTau);
        gpu.uploadParams(normTensors); // capped norms land before the next forward
      }
      opts.onStepTime?.(t1 - t0, performance.now() - t1, readback);

      let lossSum = 0;
      for (const l of losses) lossSum += l.data[0];
      const avg = lossSum / opts.batchPerStep;
      if (step % logEvery === 0 || step === opts.steps - 1) {
        history.push({ step, loss: avg });
        opts.onLog?.(step, avg);
      }

      if (
        opts.onCheckpoint && opts.checkpointEvery && step > 0 &&
        step % opts.checkpointEvery === 0
      ) {
        await opt.syncWeightsToHost(); // pull device-resident weights before export
        await opts.onCheckpoint(step);
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
