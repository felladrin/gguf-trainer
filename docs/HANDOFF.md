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
  `tinyConfig` (725K params, larger than the CPU demo's model), loss 5.67 → 0.56, GPU greedy
  sampling reproduces the corpus, exports GGUF that loads and runs in `llama-cli`. Measured on an M1
  Max: **36 ms/step** forward+backward+readback vs **143 ms/step** for the same model on CPU.
- **Validation harnesses** (the gate for all future op/kernel work):
  - `tests/gradcheck.ts` — per-element finite-difference gradient checks for every CPU op plus
    whole-model checks, with a negative control proving the harness rejects wrong gradients. Runs on
    all three runtimes.
  - `tests/gpu_parity.ts` — GPU-vs-CPU forward and backward parity per op, finite differences run
    directly against GPU forwards, whole-model gradient parity, and cross-micro-batch gradient
    accumulation. Skips cleanly where WebGPU is unavailable.

Implemented overall: GGUF v3 writer/reader, F16/Q8_0/Q4_0 (de)quantizers, byte-level BPE tokenizer,
reverse-mode CPU autograd, WGSL kernels for the whole op set, the Qwen3 forward pass (GQA,
QK-RMSNorm, RoPE, SwiGLU, tied embeddings), AdamW + Muon optimizers, training loops for both
backends.

## How the WebGPU backend plugs in

`WebGPUBackend` implements the `OpsBackend` interface from `src/model/autograd.ts` and registers via
`install()`; the model and `backward()` walk run unchanged. Ops encode GPU dispatches synchronously;
host `data`/`grad` arrays are stale until `await gpu.sync()` — the one unavoidable async point
(WebGPU readback is async-only). That is why the GPU training loop is the async twin
`src/backend/train_gpu.ts` rather than a change to the reference trainer. Parameters stay
authoritative on the host: `uploadParams()` each step, optimizer steps host arrays. A backend must
implement ALL ops — mixing CPU and GPU ops in one graph would silently read stale host mirrors.

## Your primary task: keep the optimizer on the GPU

Measured step-time split on the GPU demo (M1 Max, tinyConfig): GPU forward+backward+sync **36 ms**,
AdamW **5 ms**, **Muon `step()` 1276 ms** — the CPU-side Newton–Schulz iteration is ~95% of wall
time. This was item 5 ("optional") of the original kernel plan; it is now the whole ballgame:

1. Port Muon's Newton–Schulz to the existing GEMM kernels (it is a handful of small matmuls —
   `newtonSchulz()` in `src/train/muon.ts` is the spec) and keep momentum buffers device-resident.
2. This forces an optimizer-interface decision: `Optimizer.step()` is sync and host-side today.
   Cleanest path: a GPU-aware optimizer in `backend/` that consumes gradients before they're read
   back, syncing weights to host only for export/sampling. Do not complicate `src/train/*` — same
   rule as the trainer split.
3. Gate: identical training trajectory to the CPU Muon (same seeds) within float tolerance, verified
   in `tests/gpu_parity.ts`.

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

## Secondary roadmap (after optimizer kernels, priority order)

1. **muP parametrization** — tune hyperparameters on a tiny proxy model, transfer to full size
   (touches init + per-tensor LR).
2. **WSD learning-rate schedule** (warmup → stable → linear cooldown) — cheap win, pure trainer
   change.
3. **MuonClip / attention-logit clipping** — stability when scaling up; pairs with the existing
   QK-norm.
4. **GGUF checkpoint loader** — dequant already exists in `src/gguf/quantize.ts`; wire it back into
   a `Qwen3Model` for resume/round-trip.
5. **Scale + real data** — larger `Qwen3Config`, curated corpus, longer runs. Reminder: at small
   scale, data quality beats architecture tweaks.
6. **Flash-style attention + f16 buffers** — the current attention kernels materialize the [Hq,T,T]
   probability tensor (fine at T≤128); tile it when scaling context.

## Hard limits and known nuances (not bugs)

- You **cannot train in Q4_0**; keep float master weights and quantize at export. Rationale in
  `docs/DESIGN.md`.
- Realistic scale on JS/WebGPU is small models (single-digit to low-tens of millions of params), not
  a CUDA-cluster 100M+ run.
- WGSL kernels bake shapes as constants (pipelines cache per shape). Training reuses shapes, so
  everything compiles once; generation compiles one variant per context length. Fine at demo scale.
- `llama-cli` sampling from the 40-step toy models produces corpus-vocabulary text but not the exact
  in-framework greedy continuation — identical behavior for CPU- and GPU-trained artifacts
  (pre-tokenization of the prompt differs from our training windows at toy scale). Revisit under
  roadmap item 5.

## Where things live

```
src/gguf/      f16, quantize (+dequant), gguf writer/reader
src/tokenizer/ byte-level BPE
src/model/     config, autograd (CPU op set + OpsBackend hook), qwen3 forward
src/train/     optimizer iface, adamw, muon, trainer (CPU)
src/backend/   webgpu.ts (WGSL kernels, buffers, sync), train_gpu.ts
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
