# Training-speed research (2026-08-18)

> A research pass, kept separate from `optimization.md` (the measured ledger) and
> `docs/notes/journal.md` (what was tried). Question: what can still make training faster,
> on any WebGPU machine, without leaving the portability constraint (plain WGSL, no vendor
> intrinsics, no hardware-specific paths). Nothing below is implemented; every claim is either
> derived from the kernels in `src/backend/wgsl.ts`, from the repo's own measurements, or sourced.

## Status after measuring (2026-08-18, M1 Max)

Everything from "## Verdict" down is the original pre-measurement analysis, unedited, because the gap
between what it predicted and what was measured is the useful part. Its numbers are the pre-rewrite
kernels and several of its conclusions are wrong; this section says which. `optimization.md` lever 1
is the current ledger.

| Lever                         | Predicted             | Measured (M1 Max)                                | Outcome                                                                       |
| :---------------------------- | :-------------------- | :----------------------------------------------- | :---------------------------------------------------------------------------- |
| A1 vec4 inner loop            | 1.3-1.5x on the slice | 1.9-2.5x on the slice                            | done, under-called                                                            |
| A2 exp2 + conditional rescale | 1.01-1.03x            | 1.16-1.36x per kernel                            | done, badly under-called: `exp` is not one instruction                        |
| A3 partial accumulators       | neutral or 1.2x       | 1.18-1.20x in `srcAttnBwdDkv`, neutral elsewhere | done in that one kernel only                                                  |
| B1 GEMM BK + vec4 fragments   | 1.1-1.3x              | 1.85x on the tied readout                        | done; needed the A tile staged transposed, which section 3 did not anticipate |
| C1 CE, drop the divide pass   | ~1.02x of the step    | 15.7x on CE                                      | done, but the divide pass was not the problem: one thread per row was         |
| D1 bind-group cache           | removes 10-50 ms/step | 71.5 ms/step = 0.4%                              | measured, NOT done                                                            |
| A4, A5, B2, C2, D2            | -                     | -                                                | untouched, still open                                                         |

Three things the plan got wrong, all of which the very first measurement would have caught, which is
why plan step 1 (build the benchmark first) was the right call:

1. **The target.** Section 2 does instruction accounting for `srcAttnFwd`. Profiled per kernel,
   `srcAttnBwdDkv` was ~70% of the attention slice and `srcAttnFwd` ~15%. The repo's 78%-attention
   figure had never been split per kernel, so the whole discussion aimed at the wrong kernel.
2. **The SFU argument.** "The SFU has wide headroom, so exp-count is second-order" led to A2 being
   ranked as a 1-2% afterthought. Swapping `exp` for `exp2` alone gave 1.36x on `srcAttnBwdDkv`,
   which has no conditional rescale at all. The reasoning assumed `exp` lowers to one multiply plus
   one hardware instruction; measured, it does not.
3. **Cross-entropy.** C1 was scoped as saving one pass out of four. The actual defect was that the
   kernel ran on T threads total with no coalescing; a workgroup per row plus C1 gave 15.7x.

A fourth thing nobody predicted, found by review rather than by measurement: `srcAttnBwdDkv`'s query
tile needed 16640 bytes of workgroup storage at head-dim 64, over WebGPU's 16 KiB portable floor, so
the shape every published checkpoint trains at was over budget on any implementation that validates
that limit. Nothing here does: Deno's wgpu was probed accepting a 16640-byte shader on a device
reporting a granted limit of 16384. Sizing the tile to the floor fixed it and was 1.18x faster as
well.

Two places below propose sizing workgroup memory against `caps.maxComputeWorkgroupStorageSize`: the
last sentence of section 2 and section 5. That is not what shipped and should not be. Sizing to the
granted limit emits a kernel that runs on the machine that built it and not on a spec-default one,
and since no runtime here validates the difference, the failure would surface only for someone else.
`attnBwdTile` sizes against the 16 KiB floor unconditionally, and `tests/kernel-limits.ts` holds it
there.

Everything below this line is the pre-measurement analysis, unedited. Read it as a record of the
reasoning, not as a description of the current kernels.

## Verdict

The measured step (Strix Halo, 94.7M, seq 2048, batch 8; `optimization.md` 2026-07-08) is
~78% attention, ~9% GEMM/linear, the rest optimizer/norm/CE. Re-deriving the useful FLOPs from
the kernels:

| Slice          | Runtime (measured) | Useful FLOPs (derived)  | FLOP efficiency (derived) |
| :------------- | :----------------- | :---------------------- | :------------------------ |
| attention      | ~15.9 s (78%)      | ~1.8 TFLOP/step         | ~115 GFLOP/s              |
| linear GEMM    | ~1.8 s (9%)        | ~7.2 TFLOP/step         | ~4 TFLOP/s                |
| Muon NS + rest | ~2.7 s (13%)       | ~1.2 TFLOP/step + norms | ~2 TFLOP/s (NS part)      |

The same silicon runs the `srcGemm` register-tiled kernel at 17-35x the FLOP efficiency of the
attention kernels: the hardware ceiling is not the wall, the kernel structure is. The attention inner loop
lacks exactly the three tricks that lifted GEMM 1.9-3.3x end-to-end (`notes/journal.md`,
"Performance work"): **vectorized loads, multi-accumulator ILP, and fewer instructions per useful
FLOP**. The three rewrites the journal reverted (f16, split-K, QT register tiling) do not touch
those three sub-levers, so the journal's "attention is at its practical floor" conclusion covers
what was tried, not what is untried.

Ranked untried levers (details in section 3; effort: S = one session with the parity gate,
M = a few, L = week-plus rewrite):

| #  | Lever                                                 | Slice touched         | Expected on slice          | Effort            |
| :- | :---------------------------------------------------- | :-------------------- | :------------------------- | :---------------- |
| A1 | vec4-vectorize the attention inner loop               | ~78% (attention)      | 1.3-1.5x                   | M                 |
| A2 | exp2 domain + conditional rescale in online softmax   | ~78% (attention)      | 1.01-1.03x                 | S                 |
| A3 | 4-way partial accumulators in the dot loops           | ~78% (attention)      | 1.0-1.2x if latency-bound  | S                 |
| B1 | GEMM: BK 8→16/32 + vec4 fragment loads                | ~9-15%                | 1.1-1.3x                   | S                 |
| C1 | CE: drop the divide pass (unnormalized probs)         | ~1-2% (+8K memory)    | ~1.02x step                | S                 |
| D1 | Cache bind groups in the main loop                    | CPU (hidden today)    | removes ~10-50 ms CPU/step | S                 |
| A5 | Window warmup (designed in `design.md` #7, not built) | warmup only           | ~2-3% of run at seq 2048   | S                 |
| B2 | Newton-Schulz 5→4 iterations                          | ~5-6%                 | ~1% of step                | S (quality check) |
| D2 | Packed micro-batches (true batch dimension)           | GEMM + fixed overhead | unmeasured, spike first    | L                 |

Realistic combined outcome at 95M/seq-2048: **1.25-1.45x end-to-end**; ceiling ~2x if attention
reaches a quarter of the GEMM kernel's FLOP efficiency. Section 6 orders the work so the first
half-day of measurement decides which sub-levers matter before any rewrite is committed.

## 1. Where the step goes, re-derived

FLOP accounting (2 FLOPs/MAC, per (head, t, s) attention pair at head-dim 64):

- forward: QK dot 2d + PV 2d = 256 FLOPs
- `srcAttnBwdDq`: QK 2d + dP 2d + dQ update 2d = 384 FLOPs
- `srcAttnBwdDkv`: QK 2d + dP 2d + dK 2d + dV 2d = 512 FLOPs
- `srcAttnBwdD`: 2d per row (negligible)

Pairs per micro-batch at T=2048, window 1024, 10 heads: SWA layer
Σ_t min(t+1, W) = 1.57M/head → 15.7M; global layer T(T+1)/2 = 2.1M/head → 21M. With 10 SWA +
2 global layers × 8 micro-batches: **~1.6G pairs/step → ~1.8 TFLOP of useful attention FLOPs**.
Note T=2048 with W=1024 makes a SWA layer cost ~73% of a global layer; the 5:1 SWA ratio only
pays off above T=2W (at T=8192 a global layer costs 4x an SWA layer, which is the measured 28%
throughput drop at long context).

Linear GEMMs: ~6 FLOPs/token/param on the 73.7M non-embedding params → ~7.2 TFLOP/step
(includes the tied-readout logits GEMM, ~206 GFLOP/step of it). Muon Newton-Schulz: 5 quintic
iterations × 3 [640³]-class GEMMs per 2-D param → ~1.2 TFLOP/step.

## 2. Why the attention kernel sits 17-35x below the GEMM kernel

Instruction accounting for the forward kernel, per wave per key-step (one key-step processes 32
query rows, one per lane; the two backward kernels repeat the shape, heavier):

- **128 scalar load instructions** (64 K-row + 64 V-row), wave-uniform and cache-served; as
  vec4 they would be 32;
- **128 FMA instructions**, but the QK dot is a **64-deep serial accumulation chain** (one
  `dot` accumulator), which is the only latency hazard at full occupancy;
- **2 MUFU instructions** (`corr` + `p`), plus the 2 base-conversion multiplies `exp()` implies:
  WGSL `exp` lowers to multiply-by-log2(e) + `v_exp2_f32`, and the SFU runs **quarter-rate** on
  AMD silicon (one `v_exp2_f32` per 4 clocks per SIMD32 on RDNA2, same SFU class on
  RDNA3/3.5). At 2 MUFU instructions per 32-pair key-step that is ~8 SFU cycles per ~274 issue
  slots: the SFU has wide headroom today, so exp-count is a second-order lever and **issue slots
  are the first-order constraint**;
- for reference the FP32 FMA ceiling is 128 FLOPs/CU/cycle single-issue (256 with the RDNA3.5
  VOPD dual-issue path, which a scalar loop does not reliably trigger).

That is ~8.6 issue slots per (head, t, s) pair, of which 4 are loads. Contrast `srcGemm`,
measured at ~2-4 TFLOP/s in the same stack: 4x4 unrolled accumulators (16 independent chains,
4x ILP), cooperative coalesced staging through workgroup memory, and a K-loop whose per-step
instruction count is amortized over 16 MACs per fragment load. The attention kernel uses none of
the three: scalar loads, a 1-wide accumulator chain, and an exp per rescale. That is the target;
it is a property of the kernel source, not of the silicon, which is why the fix is
machine-independent.

Hardware references used above: per-CU VALU rates and VOPD for gfx1151 from the ROCm profiler
speed-of-light docs ([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/conceptual/rdna/system-speed-of-light.html)),
SFU/TFU behavior and `v_exp_f32` from the ROCm HIP hardware-implementation chapter
([rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)),
the quarter-rate SFU figure from the RDNA2 instruction analysis at
[nelcit.github.io](https://nelcit.github.io/shader-clippy/blog/pow-const-squared). WebGPU's
portable workgroup-storage floor is 16 KiB ([MDN](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits));
the Strix device grants 64 KiB (journal probe), and the backend already captures the granted cap
in `DeviceCaps.maxComputeWorkgroupStorageSize`, so any tile choice can be sized against it at
runtime.

## 3. The levers

### A. Attention slice (~78%)

**A1. Vectorize the inner loop (vec4).** Load K/V rows as `vec4<f32>` (16 loads per row instead
of 64), keep `qr`/`acc` as 16 vec4 chunks, and dot with vec4 FMAs. Pure WGSL, no intrinsics;
naga lowers it to a single 16-byte load on every backend. Cuts 96 load instructions per key-step
in the forward (~36% of its issue slots) and a similar fraction in `srcAttnBwdDq`; in
`srcAttnBwdDkv` it applies to the staged Q/dOut loads. This is the repo's own roadmap item
1 (`journal.md` "Remaining perf roadmap"), scoped there to the small elementwise kernels; the
attention inner loop is where the scalar loads actually live. Risk: low; parity is tolerance-based
(`tests/gpu-parity.ts` BWD atol 1e-3 / rtol 1e-2), so load re-grouping cannot break the gate.

**A2. exp2 domain + conditional rescale.** Two independent changes in the online-softmax
kernels, both rounding-only:

1. Move the whole softmax to the log2 domain: bake `SCALE * LOG2E` into the kernel constant,
   use the WGSL `exp2` builtin, and store `LSE` in the log2 domain (the backward kernels
   already recompute `p = f(score - LSE)`, so they pick up the same representation). This
   removes the base-conversion multiply that `exp()` adds on AMD.
2. Conditional rescale: the running max is monotone, so `corr = exp(m - mNew)` is needed only
   when the max actually moves (a handful of times per row, not per key). The current code pays
   it unconditionally: the `corr` MUFU, the `m - mNew` subtract, and the `acc[d] * corr`
   multiplies (fused into the PV update) on every key-step.

Combined: the forward key-step drops from 2 MUFU + 2 conversion multiplies + the always-on
rescale bookkeeping to 1 MUFU and ~274 to ~266 issue slots. The SFU has wide headroom on the
current kernel (section 2), so this is a ~1-2% attention-slice gain (~3% on the forward
kernel), not the main play; it is included because it is small, safe, and it removes work that
grows with context. Safe for every architecture (llama, gemma3, qwen3) because no bound on
scores is assumed.

**A3. Partial accumulators in the dot loops.** Split each 64-deep serial dot into 4 independent
16-deep chains (unrolled, named scalars, exactly as `srcGemm` does for its micro-tile) and sum
at the end. If the kernel is FMA-latency-bound rather than issue-bound, this is the single
biggest constant factor available; if it is issue-bound it is neutral. The section 6
microbenchmark decides. Cost: ~1e-6 relative rounding change, inside the existing parity
tolerances.

**A4. Revisiting 2D workgroup tiling (only after A1-A3 data).** The reverted attempts do not
cover the standard configuration: split-K split the _key_ axis across lanes (destroying the
wave-uniform K/V broadcast, measured 0.4-0.7x); QT tiling put multiple query rows in one
thread's registers (occupancy collapse at QT≥2, measured 0.80-0.94x). Neither staged a K/V tile
through workgroup memory. But the evidence still points against a 2D-tiled forward: the base
kernel's K/V reads are already wave-uniform and cache-served, the journal's 32-key staged
forward measured 17% slower (barrier overhead with no bandwidth to save), and staging only
paid off in `srcAttnBwdDkv` where thread loop bounds diverge. Treat a 2D-tiled forward as a
bet on barriers being cheaper than the measurements suggest, and gate it on the A1-A3
results showing the kernel is still far from the GEMM kernel's efficiency.

**A5. Window warmup.** `design.md` #7 documents it as designed-not-built: train early steps with
a shrunk `--window`, then open it. At T=2048 the win is modest (SWA already costs ~73% of a
global layer; halving the window during the 10% warmup slice buys ~2-3% of the run); at
T≥4096 the global layers dominate and the same lever scales up. It also changes training
dynamics (early layers see a narrower context), so it is a model-quality decision as much as a
speed one.

### B. GEMM + Muon slice (~9-16%)

**B1. GEMM tiling sweep.** `srcGemm` bakes BM=BN=64, BK=8, TM=TN=4. Two untried knobs:

- **BK 8→16/32.** The K-loop pays 2 barriers per 8-K step (160 barriers per output tile at
  K=640) and 8 fragment loads per K-step; BK=16 halves both. Workgroup memory at BK=16 is 8 KiB
  (fits the 16 KiB portable floor; at BK=32 it is 16 KiB, which needs a cap check on the
  16-KiB-default devices).
- **vec4 fragment loads** from the staged tiles: `a0..a3` and `b0..b3` are 4-contiguous in the
  tile, so 8 scalar loads become 2 vec4 loads per K-step.

The logits GEMM (N=32768) is the largest single GEMM in the step; BN=64 tiles it 512 ways, no
edge waste, so the win is the K-loop constant factor, not tiling shape. Expected 1.1-1.3x on
the slice → ~1-4% of the step. This is a sweep, not a rewrite: the tile constants are the
single source shared with the dispatch grid.

**B2. Newton-Schulz 5→4 iterations.** NS is ~5-6% of the step (profile at 768/4L) and the
quintic polynomial's marginal gain past 4 iterations is a second-order conditioning term. Cut one
iteration, measure the orthogonality residual (`‖XXᵀ − I‖_F`) on a fixed gradient in the parity
suite, and the loss curve on a short seeded run, before accepting. ~1% of the step for a
quality check.

### C. Cross-entropy slice (~1-2% of time; the big memory consumer at 8K)

**C1. Unnormalized-probability CE (drop the divide pass).** `srcCeFwd` does three full passes
over [T,V] (max; exp+sum; divide) and `srcCeBwd` a fourth. The divide is redundant: store
`e = exp(z - max)` directly (one fewer pass, one fewer full [T,V] read-modify-write), keep
`invSum = 1/Σe` per row, compute the loss as `-log(Σe) + (z_t - max)` (no read of the
probability), and apply `invSum` in the backward (`dLogits = (LG/DIV)·(e·invSum - onehot)`).
This is the same unnormalized-trick the flash attention kernel already uses for `p`. Removes one
full [T,V] pass per micro-batch (256 MB at seq 2048/vocab 32768, 1 GiB at seq 8192) and the
divide rounding.

**C2. (Deferred, memory play) chunked online CE over the vocab axis**, no [T,V] materialization
at all: stream the readout weights in vocab chunks with running max/sum, second pass for the
gradient. Prior art is all CUDA/Triton:
[Liger-Kernel fused_linear_cross_entropy](https://github.com/linkedin/Liger-Kernel),
[mgmalek/efficient_cross_entropy](https://github.com/mgmalek/efficient_cross_entropy), and
fla's fused linear+CE. On Strix the [T,V] buffer fits (the 1 GiB at 8192 is why `--reclaim`
exists), so this only becomes a speed item if CE ever shows up in the profiler; keep it as the
memory lever for bigger-batch-at-8K.

### D. Loop structure (CPU side, currently hidden)

**D1. Bind-group caching for the training graph.** The optimizer already proved the pattern:
`prepareDispatch` existed because bind-group/source rebuilding was ~85% of the optimizer encode
(journal). The main loop does not use it: every op of every micro-batch calls
`this.pipeline(code)` + `this.bindGroup(p, buffers)`, i.e. 2-4K `createBindGroup` calls per step
at batch 8. With the GPU pinned at 100% this is hidden in wall-clock today, but it caps
CPU/GPU overlap and becomes a visible bubble the moment per-step GPU work shrinks (which is the
point of the attention levers). Cache key: (pipeline, buffer tuple), exactly as the optimizer
does; buffers are stable per tensor, so the cache is valid for the run. Standard WebGPU advice:
build bind groups once, reuse
([toji.dev](https://toji.dev/webgpu-best-practices)).

**D2. Packed micro-batches, spike first.** `optimization.md` #2 already scoped it and the number
is still missing: per-sequence overhead at batch 1 vs inside batch 8. The packed form is one
[B·T] tensor per op with block-diagonal attention: each sequence's rows attend only within their
own block, which is a one-lookup change in the flash kernels (`lower = max(seqStart(t),
winStartBlock(t))`) plus per-sequence loss bookkeeping. It enlarges every GEMM by B and deletes
7/8 of the per-micro-batch fixed work (submits, CE buffers, norm dispatches). This is the only
lever that is structurally larger than a constant factor, and it is a real rewrite of the forward
plus parity cases, so it is gated on the batch 1 vs 8 measurement showing >~10% per-sequence
overhead.

### E. Small items (profile first, fix only if the profiler agrees)

- **RoPE table.** `srcRope` computes `pow(BASE, -2j/HD)` + `sin` + `cos` per (t, h, j) element:
  three SFU-class ops per half-head-element, and `freq` depends only on `j` (32 distinct
  values) while training always starts at position 0. Precompute a `[half]` frequency buffer and
  a `[T, half]` sin/cos buffer per micro-batch: 3 SFU ops per element becomes a load.
- **`srcRmsNormBwdW` strided loads.** One thread per column scanning T rows at stride D f32:
  each 32-lane load instruction touches 32 distinct 64 B cache lines and uses 4 B of each
  (~16x overfetch at L1/L2). Derived estimate ~1% of the step at 95M: 73 such dispatches per
  micro-batch (6 norms/layer × 12 + output norm) × ~168 MB of line traffic each. Fix is the
  standard row-tile-through-workgroup-memory reduction. Verify with `startProfile` before
  touching: at other shapes it may be noise.
- **`srcEmbeddingBwd` O(V·d·T) scan.** Deterministic by design (no atomics); at 32768×640 it is
  ~0.3% of the step. It scales with vocab, so at 2x vocab it becomes a multi-percent item; the
  atomicAdd scatter alternative is available (parity is tolerance-based, so non-deterministic
  accumulation order passes the gate). Defer.

## 4. Ruled out (measured, do not re-tread)

| Idea                                | Outcome / reason                                                                                                                                                                  |
| :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| f16 compute (f16 mul, f32 accum)    | 0.98x on attention at seq 4096-8192 + overflow-to-NaN at step 2400 without clamps (journal)                                                                                       |
| f16 storage for Q/K/V               | 1.02-1.06x, shrinks with context (journal)                                                                                                                                        |
| Split-K attention (32 threads/row)  | 0.4-0.7x: destroys the wave-uniform K/V broadcast (journal)                                                                                                                       |
| QT query-register tiling            | 0.80-0.94x at QT=2, 0.48-0.68x at QT=3: register pressure halves occupancy faster than reuse pays (journal)                                                                       |
| WMMA / subgroup-matrix              | Not exposed by Deno's wgpu on gfx1151 (15 features probed, no `subgroups`); no WGSL matrix ops in the spec                                                                        |
| bf16 of any kind                    | No `bf16` WGSL type, no `shader-bf16` feature, no bf16 in naga (design.md)                                                                                                        |
| PyTorch + ROCm                      | The attention wall is the GPU, not the framework; and it is a different engine, off the portability constraint                                                                    |
| Fixed-max softmax via QK-norm bound | No safe static bound: q/k norm weights are trained, and under `qkClip` the observed max is 3.3-4.4x the proxy (heuristic, not a bound). A2 gets the exp savings without the bound |
| Per-lane SWA window start           | Destroys the wave broadcast; the block-aligned start (`winStartBlock`) is the documented reason SWA is not slower than full attention                                             |

## 5. Machine independence

Every lever above is plain WGSL: no intrinsics, no vendor paths, workgroup-memory sized against
`caps.maxComputeWorkgroupStorageSize` (16 KiB spec floor, 64 KiB on the Strix device), and no
assumption about SFU or load-pipeline rates. Where the levers differ across machines is the
_share_ of the bottleneck they address: A2 matters most where the SFU is narrow (AMD quarter-rate;
Apple's rate differs), A1 where load-issue is the constraint, A3 where the FMA latency chain
dominates. The plan below measures the split on the target device before committing to any
rewrite; `startProfile`/`stopProfile` (timestamp-query) plus the existing parity gate make each
step a ten-minute measurement, and the gate is already tolerance-based, so rounding-order
changes are admissible.

## 6. Plan (sequenced, with stop conditions)

1. **Permanent benchmark (prerequisite).** The journal's `bench_attn.ts` was reverted with the
   QT work; the main tree has no bench at all. Rebuild it as a registered `bench` subcommand
   (repo rule: a workflow not in `help` does not exist): fixed attention/GEMM/CE shapes, fixed
   seed, per-kernel ms via `startProfile`, printed as a table. Without it every lever below is
   argued instead of measured.
2. **Microbenchmark campaign (~half a day).** Four stripped variants of the forward kernel:
   full / loads-only / FMAs-only / SFU-only (exp2 loop) at T=2048 and 8192, plus a standalone
   `exp2` throughput loop. Output: which of {issue, MUFU, load latency, FMA latency} bounds the
   current kernel on this device.
3. **PR 1: A2** (exp2 + conditional rescale). Smallest diff, safe on all three architectures,
   parity-gated. Measure against step 1's baseline.
4. **PR 2: A1 + A3**, selected by step 2's verdict (A3 only if latency-bound).
5. **PR 3: B1 + C1** (GEMM BK/vec4 sweep; CE divide-pass removal).
6. **D1** bind-group cache (CPU-side, no parity exposure).
7. **D2 spike**: seq 2048 at batch 1 vs the per-sequence cost inside batch 8; if overhead is
   > ~10% of the step, schedule the packed-micro-batch rewrite; otherwise close `optimization.md`
   > item 2 with the number.
8. **A5/B2** are decisions, not code paths: window warmup is a run-policy question, NS-4 needs
   the orthogonality/loss check first.

**Stop condition:** if step 2 shows the attention kernel already at a large fraction of the SFU
or issue ceiling (A1-A3 can buy nothing), stop attention work; the remaining levers are D2 and
the algorithmic ones, and the honest answer becomes "the O(T²) pair count, not the kernel".

## 7. What it is worth

The documented pace is ~70M tokens/day (903 tok/s measured, overheads included). At 1.25-1.45x
that is 88-101M tokens/day: a 100M-token "quick experiment" drops from ~1.5 days to a day, and
the 1.44B-token run behind the published model from ~20 days to ~14-16. At the 2x ceiling the
same run is under two weeks. Nothing above changes the "sub-100M models are the honest target"
ceiling; all of it buys wall-clock back within it.

## References

- Repo measurements: `docs/optimization.md` (2026-07-08 baseline, attention 78%),
  `docs/notes/journal.md` (kernel rewrites, reverted attempts, remaining roadmap),
  `docs/design.md` (precision and backend bring-up).
- RDNA3.5 (gfx1151) speed-of-light rates, VOPD: [rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/conceptual/rdna/system-speed-of-light.html)
- RDNA SFU/TFU, `v_exp_f32`, LDS bandwidth, Wave32: [rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- SFU quarter-rate (RDNA2, applied by class): [nelcit.github.io](https://nelcit.github.io/shader-clippy/blog/pow-const-squared)
- WebGPU limit defaults (16 KiB workgroup storage floor): [developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits)
- WGSL spec (`exp2`, atomics, workgroup memory): [w3.org](https://www.w3.org/TR/WGSL)
- Bind-group reuse guidance: [toji.dev](https://toji.dev/webgpu-best-practices)
- Prior art, browser WGSL training (forward+backward+AdamW, online-softmax attention; small
  scale, no published throughput at 95M): [github.com](https://github.com/toprakdeviren/webgpu-llm)
- Fused linear-cross-entropy prior art (CUDA/Triton; concept reference for C2):
  [github.com](https://github.com/linkedin/Liger-Kernel), [github.com](https://github.com/mgmalek/efficient_cross_entropy)
- FlashAttention-3 (what full 2D tiling + tensor cores buys on NVIDIA; context for why the same
  structure is not automatically fast on a no-TC, no-subgroup WebGPU path):
  [arxiv.org](https://arxiv.org/abs/2407.08608)
