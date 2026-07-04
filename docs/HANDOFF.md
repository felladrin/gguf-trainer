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

## Primary tasks (both high priority, independent — either can go first)

### 1. Keep the optimizer on the GPU

Measured step-time split on the GPU demo (M1 Max, tinyConfig, 725K params): GPU
forward+backward+sync **36 ms**, AdamW **5 ms**, **Muon `step()` 1276 ms** — the CPU-side
Newton–Schulz iteration is ~95% of wall time. This was item 5 ("optional") of the original kernel
plan; it is now the whole ballgame, and it gets worse fast, not gracefully, as params scale up:

| Params | GPU fwd+bwd+sync | CPU `Muon.step()`                          |
| ------ | ---------------- | ------------------------------------------ |
| 0.9M   | 183 ms           | 412 ms (56 matrices)                       |
| 4.9M   | 210 ms           | 10,026 ms (28 matrices)                    |
| 20.5M  | 201 ms           | not measured — already impractical at 4.9M |
| 100.4M | 387 ms           | not measured — already impractical at 4.9M |

GPU compute barely moves (111× more params costs ~2× more GPU time — the tiled GEMMs are nowhere
near saturated at these sizes). `Muon.step()` on CPU went from 412 ms to over 10 seconds for only
5.4× more parameters, and with _fewer_ matrices (28 vs. 56) — the cost is cubic in matrix dimension
(the unvectorized Newton–Schulz matmuls in `src/train/muon.ts`), not linear in matrix count. A real
pretraining run needs thousands of steps; at 20M+ params this is hours to days spent only in the
optimizer.

**Immediate fallback, no GPU work required:** drop Muon and optimize everything with `AdamW` instead
(already implemented, same `Optimizer` interface). Checked at the 100M-param config: a full
`AdamW.step()` over every parameter takes **530 ms**, in the same range as the GPU forward+backward,
because it's pure elementwise math with no matrix ops. This trades away Muon's ~2× compute
efficiency but unblocks training tens of millions of params _today_, before the work below lands.

The real fix:

1. Port Muon's Newton–Schulz to the existing GEMM kernels (it is a handful of small matmuls —
   `newtonSchulz()` in `src/train/muon.ts` is the spec) and keep momentum buffers device-resident.
2. This forces an optimizer-interface decision: `Optimizer.step()` is sync and host-side today.
   Cleanest path: a GPU-aware optimizer in `backend/` that consumes gradients before they're read
   back, syncing weights to host only for export/sampling. Do not complicate `src/train/*` — same
   rule as the trainer split.
3. Gate: identical training trajectory to the CPU Muon (same seeds) within float tolerance, verified
   in `tests/gpu_parity.ts`.

### 2. Tile causal attention (flash-style)

The attention kernels (`src/backend/webgpu.ts`) bind the whole `[heads,T,T]` causal-probability
matrix as one storage buffer. That single buffer, not compute, is what actually caps trainable
context length — measured directly:

| T    | Attention buffer (4 heads) | WebGPU spec-default limits (128 MiB/buffer)                              |
| ---- | -------------------------- | ------------------------------------------------------------------------ |
| 2560 | 105 MB                     | OK                                                                       |
| 3072 | 151 MB                     | **fails** — real `BindGroup ... is invalid` error, not silent corruption |

**8192 tokens is not trainable under the WebGPU spec's default device limits at any realistic head
count.** `initWebGPU()` now requests the adapter's own maximum buffer limits instead of that default
(already shipped — see "Context length" under Hard limits below), which raises the ceiling far past
3072 on adapters that grant it. That fix moves the wall; it doesn't remove it, and it doesn't touch
compute cost:

| T (elevated limits, M1 Max) | Attention buffer | fwd+bwd+sync |
| --------------------------- | ---------------- | ------------ |
| 2048                        | 67 MB            | 414 ms       |
| 4096                        | 268 MB           | 1.5 s        |
| 8192                        | 1.07 GB          | 4.36 s       |

Two independent reasons this still needs the real fix:

1. **Compute grows with T regardless of memory.** Full causal attention is inherently O(T²) work;
   the ~3.5× time jump per doubling above is that showing up directly, since the kernel isn't tiled
   or blocked. Flash-style tiling doesn't change that asymptotic — it's exact attention, not an
   approximation — but a blocked/online-softmax implementation uses memory bandwidth far better than
   the current one-shot `[T,T]` materialization, which is most of why real flash-attention
   implementations are faster in practice at the same T.
2. **The elevated-limits fix is adapter-dependent.** Not every GPU/browser grants
   `maxStorageBufferBindingSize` anywhere near the ~4 GiB this M1 Max allows; on one that doesn't,
   `initWebGPU()` silently falls back to the spec default and you're back to the ~2560–3072 ceiling
   with no warning. Flash tiling removes the single-buffer dependency entirely, so the ceiling stops
   being a function of which adapter happened to run the code.

Cheapest incremental step before the full rewrite: switch the probability buffer to f16 (halves
memory, roughly doubles the T ceiling for a given memory budget, on top of whatever the elevated
device limits already allow). The full fix is blocked, online-softmax attention over key/value tiles
so the `[T,T]` matrix is never materialized — standard flash-attention. Gate: same
finite-difference + GPU-parity checks as every other kernel (`tests/gpu_parity.ts`).

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
   (touches init + per-tensor LR).
2. **WSD learning-rate schedule** (warmup → stable → linear cooldown) — cheap win, pure trainer
   change.
3. **MuonClip / attention-logit clipping** — stability when scaling up; pairs with the existing
   QK-norm.
4. **GGUF checkpoint loader** — dequant already exists in `src/gguf/quantize.ts`; wire it back into
   a `Qwen3Model` for resume/round-trip. Two tiers, different difficulty:
   - **Resuming a GGUF this framework produced**: straightforward. Architecture and tokenizer
     already match by construction — it's tensor-name matching plus the existing `dequantize()` to
     repopulate `model.params()`, then the current trainer runs unchanged.
   - **Fine-tuning an arbitrary external Qwen3 GGUF** (e.g. the real Alibaba releases): a bigger
     lift on top of that. Needs the config read back out of GGUF metadata instead of a hand-written
     `Qwen3Config`, and the actual vocab/merges loaded from the GGUF's tokenizer metadata since
     `src/tokenizer/bpe.ts` only trains fresh vocabularies, it doesn't import one. Real Qwen3
     checkpoints also start at hundreds of millions of parameters — past where the optimizer
     bottleneck above stays practical, so this is gated by that fix regardless.
5. **Scale + real data** — larger `Qwen3Config`, curated corpus, longer runs. Reminder: at small
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
- **Context length ceiling.** The attention kernels bind the whole `[Hq,T,T]` causal-probability
  matrix as one storage buffer, which caps trainable T well before compute does. `initWebGPU()` now
  requests the adapter's own maximum buffer limits at device creation (falling back to the WebGPU
  spec default, `maxStorageBufferBindingSize` = 128 MiB, only if an adapter rejects that request) —
  on this M1 Max that raised the ceiling from ~2560–3072 tokens to a confirmed-working 8192. Not
  every adapter grants that much, and even where it's granted, cost still grows ~quadratically with
  T since attention isn't tiled. See "Primary tasks → 2. Tile causal attention" above for the
  measured numbers and the real (portable) fix.
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
