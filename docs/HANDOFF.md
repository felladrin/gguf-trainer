# Handoff — instructions for the next development team

## What this project is

A TypeScript framework that trains a **Qwen3ForCausalLM** model **from scratch** and writes it
**directly to GGUF** — no Python, no Hugging Face, no PyTorch. It runs on **Deno, Bun, and Node**,
and trains on **any GPU — AMD, Apple Silicon, NVIDIA — via WebGPU** (Deno ships WebGPU natively;
Node/Bun fall back to CPU).

## Current state (working and verified)

Both backends are complete and pass end-to-end:

- **CPU reference path** (`examples/demo.ts`): trains a tiny Qwen3, loss drops 5.65 → 0.98, greedy
  sampling reproduces the corpus, writes + re-verifies GGUF in F16, Q8_0, Q4_0. Runs on Deno, Bun,
  and Node.
- **WebGPU path** (`examples/demo_gpu.ts`): the full op set from `src/model/autograd.ts` implemented
  as WGSL compute shaders — forward AND backward — in `src/backend/webgpu.ts`. Trains the full
  `tinyConfig` (725K params), loss 5.67 → 0.56, GPU greedy sampling reproduces the corpus, exports
  GGUF that loads and runs in `llama-cli`. Measured on an M1 Max (serial, no GPU contention): **~42
  ms/step** forward+backward+sync, **~3 ms/step** GPU Muon optimizer — **~21 steps/s** total.
  Baseline before these changes was 0.8 steps/s (1276 ms CPU Muon dominated). Both demos run
  unchanged.
- **Validation harnesses** (the gate for all future op/kernel work):
  - `tests/gradcheck.ts` — per-element finite-difference gradient checks for every CPU op plus
    whole-model checks, with a negative control proving the harness rejects wrong gradients. Runs on
    all three runtimes.
  - `tests/gpu_parity.ts` — GPU-vs-CPU forward and backward parity per op, finite differences run
    directly against GPU forwards, whole-model gradient parity, cross-micro-batch gradient
    accumulation, GPU Muon trajectory parity vs CPU, and sync() fence verification. 36 cases total.
    Skips cleanly where WebGPU is unavailable.

Implemented overall: GGUF v3 writer/reader, F16/Q8_0/Q4_0 (de)quantizers, byte-level BPE tokenizer,
reverse-mode CPU autograd, WGSL kernels for the whole op set, the Qwen3 forward pass (GQA,
QK-RMSNorm, RoPE, SwiGLU, tied embeddings), AdamW + Muon optimizers (CPU), GPU-resident Muon
optimizer, flash-style tiled causal attention, training loops for both backends.

## How the WebGPU backend plugs in

`WebGPUBackend` implements the `OpsBackend` interface from `src/model/autograd.ts` and registers via
`install()`; the model and `backward()` walk run unchanged. Ops encode GPU dispatches synchronously;
host `data`/`grad` arrays are stale until `await gpu.sync()` — the one unavoidable async point
(WebGPU readback is async-only). `sync()` always fences GPU completion, even with nothing to read
back (a 4-byte staging copy acts as the fence; see the fence comment in `webgpu.ts`).

The **device-resident training loop** (`trainLMGpuResident` in `src/backend/train_gpu.ts`) uses two
syncs per step: the first flushes forward+backward and reads loss scalars plus aux-group gradients;
the optimizer dispatches are recorded after it (grads still intact — deferred clears fire at the
next backward), and flushed by the second sync. Muon-group weights and momentum never leave the GPU
during training; a final `syncWeightsToHost()` call restores host authority for sampling and GGUF
export. Per-step host↔GPU traffic per step after warm-up: aux grads + losses only (~145 KiB at
tinyConfig vs ~2833 KiB before).

## What was just completed (both primary tasks from the previous HANDOFF)

### Task 1: GPU-resident Muon optimizer — done

Newton–Schulz now runs entirely on the GPU. New file `src/backend/muon_gpu.ts` holds `MuonGpu`:

- Muon-group 2-D weights, momentum buffers, and gradients are device-resident in persistent GPU
  buffers (created fresh via `createStateBuffer()`, never pooled/recycled, guaranteed
  zero-initialized per the WebGPU spec).
- Newton–Schulz: two-stage Frobenius-norm reduction on-device (sum-of-squares workgroup reduction →
  normalize kernel), five quintic iterations using the existing tiled GEMM (`NT` for X·Xᵀ, `NN` for
  A·A and B·X), two elementwise combine kernels (`A ← b·A + c·A²`, `X ← a·X + B·X`), and a fused
  apply kernel (`W -= lr · sqrt(max(1, rows/cols)) · ortho`) with transpose-back indexing for the
  `rows > cols` orientation flip.
- Dispatch closures are resolved once at construction (`prepareDispatch`/`prepareGemm` on
  `WebGPUBackend`), removing repeated bind-group/source-rebuild overhead.
- Aux group (embeddings, output head, norms) stays host-side AdamW with its existing global
  grad-norm clip over aux params only; Muon params are never clipped.

Measured on M1 Max, serial (no GPU contention):

| Params | GPU fwd+bwd+sync | old CPU Muon.step() | new GPU Muon.step() | speedup |
| ------ | ---------------- | ------------------- | ------------------- | ------- |
| 725K   | 42 ms            | 1,276 ms            | ~3 ms               | ~425×   |
| 5.1M   | ~210 ms          | ~10,026 ms          | ~2 ms               | ~5,000× |

Pipeline compile on step 0: ~59 ms (one-time, pipelines cache per shape thereafter).

Gate: trajectory-parity test in `tests/gpu_parity.ts` (`muonTrajectoryParity`) runs 4 steps of CPU
`trainLM + Muon` vs GPU `trainLMGpuResident + MuonGpu` with the same seed; max loss delta 3.6e-7,
max weight delta 3.6e-7 — within BWD tolerance by >1000×.

### Task 2: Hybrid flash-style causal attention — done

The attention op now uses a hybrid dispatch based on `WebGPUBackend.attnFlashMinT` (default 2048):

- **T < 2048 (materialized path):** the original five kernels writing the `[Hq,T,T]` probability
  matrix. Faster at small T where the flash kernel's one-thread-per-row layout underoccupies the
  GPU. Demo training (seqLen=32) and the existing parity cases use this path.
- **T ≥ 2048 (flash path):** online-softmax forward (one thread per query row, running max +
  rescaled accumulator, O(Hq·T) logsumexp buffer only), followed by three backward kernels that
  recompute probabilities from Q, K, and the logsumexp. No `[Hq,T,T]` buffer is ever allocated. The
  dK/dV kernel stages Q/dOut rows through 32-query workgroup tiles for ~1.8× backward speedup;
  forward and dQ use direct loads (a shared-tile variant measured 17% slower there — bandwidth
  already served by the wavefront cache).

The `[Hq,T,T]` single-buffer ceiling **no longer exists**: T=3584 with spec-default 128 MiB binding
limits now completes without error (old code needed ~205 MB for the probs buffer alone).

Measured on M1 Max, single attention op fwd+bwd+sync (Hq=4, Hkv=2, hd=32):

| T    | old (materialized) | new (mat path) | new (flash path) |
| ---- | ------------------ | -------------- | ---------------- |
| 32   | ~18 ms             | ~18 ms         | —                |
| 1024 | ~45 ms             | ~45 ms         | —                |
| 2048 | ~168 ms            | ~168 ms        | ~158 ms          |
| 4096 | ~478 ms            | —              | ~335 ms (1.4×)   |
| 8192 | ~1679 ms           | —              | ~773 ms (2.2×)   |

Gate: `tests/gpu_parity.ts` runs both paths explicitly at T=67/130/193 (each shape tested with
`attnFlashMinT=2048` for materialized and `attnFlashMinT=1` for flash), plus the `flashLargeTCheck`
functional gate at T=3584 under spec-default limits.

## Non-negotiable invariants (do not break these)

1. **GGUF loadability is a contract.** The `qwen3` tensor names and metadata keys in
   `src/export/export_gguf.ts` are what `llama.cpp` expects. If you touch them, re-validate the
   output loads and runs in `llama-cli`.
2. **Reference backend stays dependency-free and runtime-agnostic.** Everything in `src/` outside
   `backend/` must run on Deno, Bun, and Node with no install. File I/O goes through `src/io.ts`.
3. **No TS syntax that breaks Node `--experimental-strip-types`**: no `enum`, `namespace`, or
   constructor parameter properties. Use `as const` + explicit field assignment (existing code
   follows this).
4. **Correctness before speed** — optimize in the GPU backend, not by complicating the reference
   ops.

## Validation gate (required for every op/kernel change)

- `deno task test` (and `deno task test:node`) — finite-difference gradient checks + GPU parity. A
  new CPU op gets a case in `tests/gradcheck.ts`; a new or changed kernel gets a case in
  `tests/gpu_parity.ts`. A GPU op is not trusted until it matches CPU within tolerance.
- **GGUF check**: after any export change, load the file in `llama-cli` and confirm it runs.

## Secondary roadmap (after the primary tasks above, priority order)

1. **muP parametrization** — tune hyperparameters on a tiny proxy model, transfer to full size
   (touches init + per-tensor LR). The old `lr`-baking caveat is resolved: `MuonGpu` reads `lr` from
   a device buffer now (see WSD), so per-tensor/per-step lr changes cost only a buffer write.
   **Deferred to the real-data run (item 6)** on purpose: muP's payoff is LR transfer across widths,
   which only pays off once multi-width tuning is actually happening, and the Muon-group LR rule
   needs validating on real loss curves (see caveat below). Design worked out and contract-safe:
   - **Readout scale vs the llama.cpp contract.** Textbook muP controls readout scale with a `1/d`
     forward multiplier or a width-scaled readout init. Neither is available here: `token_embd` is
     the readout (tied), and an unfolded forward multiplier makes `llama-cli` inference diverge from
     training (can't fold into a tied weight). The contract-safe formulation: init `token_embd` std
     ∝ `1/√hidden` — the post-embedding RMSNorm erases the input-side effect while the tied readout
     logits stay O(1) — keep `1/√headDim` attention, add no forward multipliers. Hidden matmuls keep
     `1/√fan_in` init; per-tensor LR scales ∝ `base_width/width` for width-dependent layers.
   - **Muon-group LR caveat.** muP's `1/width` LR rule is derived for Adam/SGD. Muon's
     orthogonalized update already carries a `sqrt(max(1, rows/cols))` spectral factor and scales
     differently, so its width rule must be picked from a coordinate check (logit/activation RMS
     flat across widths), not assumed. Validate empirically during item 6.
   - The alternative (untie embeddings for a textbook readout) works and stays llama.cpp-compatible,
     but changes the model (more params, drops the tie convention Qwen3 small models use).
2. **WSD learning-rate schedule** (warmup → stable → linear cooldown) — done.
   `src/train/schedule.ts` `wsdSchedule()` returns a per-step lr multiplier; `Muon`, `MuonGpu`, and
   `AdamW` gained `setLrScale()`, and all three training loops (`trainLM`, `trainLMGpu`,
   `trainLMGpuResident`) apply an optional `schedule` before each step. `MuonGpu` holds `lr` in a
   1-element device buffer that `setLrScale()` rewrites in place (the apply pipelines bind it once
   via `prepareDispatch`, so no recompile). Gated by `wsdScheduleParity` (GPU-buffer lr vs CPU
   trajectory) in `tests/gpu_parity.ts` and a shape self-check in `tests/gradcheck.ts`. Left off the
   40-step demos on purpose — at that length the cooldown only costs final loss (the model is still
   descending steeply); WSD's payoff is on longer runs, so wire it in at roadmap item 6.
3. **MuonClip / attention-logit clipping** — done, adapted for QK-norm. `src/train/qk_clip.ts`
   `applyQKClip(model, tau)` caps each layer's logit-scale proxy
   `(1/√headDim)·√Σ_d(qNorm[d]·kNorm[d])²` at `tau` by rescaling `qNorm`/`kNorm` (each by
   `√(tau/s)`). Moonshot clips the q/k _projection_ weights, but Qwen3's QK-RMSNorm renormalizes
   those away — the norm weights are the actual lever (per-layer; they're shared across a layer's
   heads). Opt-in via `qkClipTau` on `trainLM`/`trainLMGpu`/`trainLMGpuResident`; host-side weight
   math so CPU and GPU apply it identically (gated by `qkClipTrajectoryParity` in `gpu_parity.ts`
   plus a unit check in `gradcheck.ts`). Off by default — QK-norm is the primary guard. Not a
   verbatim MuonClip: the observed-max version would mean instrumenting the parity-delicate
   attention kernels; the norm-based proxy is data-independent and tracks the observed max ~3.3–4.4×
   (T=128).
4. **Aux-group GPU residency** — `AdamW` still runs host-side on every step for embeddings/norms; at
   large scale the host↔GPU transfer for aux params becomes a secondary bottleneck. Lower priority
   than the items above.
5. **GGUF checkpoint loader** — dequant already exists in `src/gguf/quantize.ts`; wire it back into
   a `Qwen3Model` for resume/round-trip. Two tiers, different difficulty:
   - **Resuming a GGUF this framework produced**: straightforward. Architecture and tokenizer
     already match by construction — it's tensor-name matching plus the existing `dequantize()` to
     repopulate `model.params()`, then the current trainer runs unchanged.
   - **Fine-tuning an arbitrary external Qwen3 GGUF** (e.g. the real Alibaba releases): a bigger
     lift on top of that. Needs the config read back out of GGUF metadata instead of a hand-written
     `Qwen3Config`, and the actual vocab/merges loaded from the GGUF's tokenizer metadata since
     `src/tokenizer/bpe.ts` only trains fresh vocabularies, it doesn't import one.
6. **Scale + real data** — larger `Qwen3Config`, curated corpus, longer runs. Reminder: at small
   scale, data quality beats architecture tweaks. Dataset picks + links in `docs/DESIGN.md` ("Data
   quality > everything at small scale"): TinyStories to validate the pipeline, FineWeb-Edu to scale
   past storytelling.

## Hard limits and known nuances (not bugs)

- You **cannot train in Q4_0**; keep float master weights and quantize at export. Rationale in
  `docs/DESIGN.md`.
- Realistic scale on JS/WebGPU is small models (single-digit to low-tens of millions of params), not
  a CUDA-cluster 100M+ run.
- WGSL kernels bake shapes as constants (pipelines cache per shape). Training reuses shapes, so
  everything compiles once; generation compiles one variant per context length. Fine at demo scale.
- **Context length.** The old `[Hq,T,T]` single-buffer ceiling is gone. The flash path is used for T
  ≥ 2048 and keeps peak attention-side memory at O(Hq·T) — ~131 KB at T=8192/Hq=4 vs the old 1 GB.
  The `initWebGPU()` elevated-limits request remains (useful for other large buffers like logits and
  gradients), but attention no longer depends on it. The materialized path (T < 2048) still binds a
  `[Hq,T,T]` buffer — with the elevated limits that works to at least T=2048; below the crossover,
  the buffer is small anyway (67 MB at T=2048/Hq=4).
- `llama-cli` sampling from the 40-step toy models produces corpus-vocabulary text but not the exact
  in-framework greedy continuation — identical behavior for CPU- and GPU-trained artifacts
  (pre-tokenization of the prompt differs from our training windows at toy scale). Revisit under
  roadmap item 6.

## Where things live

```
src/gguf/      f16, quantize (+dequant), gguf writer/reader
src/tokenizer/ byte-level BPE
src/model/     config, autograd (CPU op set + OpsBackend hook), qwen3 forward
src/train/     optimizer iface, adamw, muon, trainer (CPU)
src/backend/   webgpu.ts (WGSL kernels, buffers, sync), train_gpu.ts,
               muon_gpu.ts (GPU-resident Muon optimizer)
src/export/    model -> GGUF (the llama.cpp contract)
tests/         gradcheck.ts (FD gradient gate), gpu_parity.ts (GPU-vs-CPU gate)
examples/      demo.ts (CPU end-to-end), demo_gpu.ts (WebGPU end-to-end)
docs/          DESIGN.md (rationale + technique research), HANDOFF.md (this file)
```

## How to run

```
deno task demo        # CPU end-to-end (Deno);  bun examples/demo.ts for Bun
deno task demo:node   # CPU end-to-end (Node)
deno task demo:gpu    # WebGPU end-to-end (Deno only today)
deno task test        # gradcheck + GPU parity
deno task test:node   # gradcheck on Node (parity prints SKIP: no WebGPU)
```

Read `docs/DESIGN.md` for the reasoning behind the optimizer choice (Muon) and the technique
roadmap, and `CONTRIBUTING.md` for the working rules.
