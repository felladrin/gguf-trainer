# Optimization notes

> Measured levers, not speculation. Maintained: every number here was taken on real hardware, and
> the commands are the current CLI.

What is left to make training faster, larger, more robust, or higher quality: grounded in
measurements taken on the Strix Halo (AMD Radeon, RADV GFX1151, 128 GB unified) during the
94.7M-param / 8192-context pretraining run, not on speculation. Read `docs/design.md` for
architecture and `readme.md` "Honest limits" for the ceiling this project accepts.

## Superseded baseline (2026-07-08)

Before the 2026-08-18 kernel rewrite: 0.049 st/s (20.4 s/step) at 94.7M / seq 2048 / batch 8 (that
step time implies ~803 tok/s, while every other pre-rewrite figure in this file uses 903 tok/s at
~18 s/step; the two were taken weeks apart and are not reconciled), GPU
busy ~100%, attention ~78% of runtime, f16 and f32 compute the same speed. Two things that
followed from it no longer hold. The GPU is NOT saturated (lever 1). And the reason once given for
f16, that it "only speeds the ~9% GEMM slice", was never the reason: measured, f16 compute is 0.98x
on attention itself and overflows to NaN without clamps, so that conclusion survives on its own
evidence (see the ruled-out table). The numbers are kept only so a re-measurement can be compared
against them; read the block below instead.

## Measured baseline (2026-08-19, post-rewrite, Strix)

| Metric                  | Value                     | How measured                                                          |
| :---------------------- | :------------------------ | :-------------------------------------------------------------------- |
| Throughput              | 0.0969 st/s (10.6 s/step) | 94.7M, seq 2048, batch 8, plateau rate                                |
| Throughput              | 1588 tok/s (1.76x)        | against 903 tok/s on the old kernels                                  |
| GPU busy                | **~42%** / 52.5%          | `gpu_busy_percent`; 42% during the run, 52.5% re-measured uncontended |
| Host CPU                | ~400% of 32 cores         | `top` on the trainer process                                          |
| Host RSS                | 1.06 GB steady            | sampled every 10s over 150 steps; flat from 300s on                   |
| Peak GPU (pool + state) | 39.3 GB                   | trainer's own readout, unchanged                                      |
| Profiled kernel time    | ~330 ms of a 10.6 s step  | `bench`, idle GPU, summed over a step's dispatches                    |

The two facts that now drive everything below:

1. **The step is host-bound, not GPU-bound.** The GPU idles ~58% of the time while four host cores
   stay busy. Cheaper kernels can no longer raise tokens/second on this box; only cheaper host work
   can. See lever 1c, which is where the time actually goes.
2. **Batch is sequential gradient accumulation, not a real batch dimension.** The training loop runs
   one sequence per forward/backward and sums the gradients (`train-gpu.ts`), so batch size trades
   step count for per-step time at a fixed tokens/second. It changes gradient noise, not throughput.

## Where the step goes: the arithmetic

Folded in from the retired `speed-research.md`. The FLOP accounting does not change when kernels are
rewritten. The instruction analysis that follows it is the PRE-MEASUREMENT reasoning, kept because
the gap between what it predicted and what was measured is the useful part, and it is wrong in
places that are called out inline. Lever 1 is the current ledger.

| Lever                         | Predicted             | Measured (M1 Max)                                | Outcome                                                                       |
| :---------------------------- | :-------------------- | :----------------------------------------------- | :---------------------------------------------------------------------------- |
| A1 vec4 inner loop            | 1.3-1.5x on the slice | 1.9-2.5x on the slice                            | done, under-called                                                            |
| A2 exp2 + conditional rescale | 1.01-1.03x            | 1.16-1.36x per kernel                            | done, badly under-called: `exp` is not one instruction                        |
| A3 partial accumulators       | neutral or 1.2x       | 1.18-1.20x in `srcAttnBwdDkv`, neutral elsewhere | done in that one kernel only                                                  |
| B1 GEMM BK + vec4 fragments   | 1.1-1.3x              | 1.85x on the tied readout                        | done; needed the A tile staged transposed, which the analysis did not foresee |
| C1 CE, drop the divide pass   | ~1.02x of the step    | 15.7x on CE                                      | done, but the divide pass was not the problem: one thread per row was         |
| D1 bind-group cache           | removes 10-50 ms/step | 71.5 ms/step, 0.4% of the pre-rewrite ~18 s step | measured, NOT done                                                            |

Three things the plan got wrong, all of which the first measurement caught, which is why building
the benchmark first was the right call:

1. **The target.** The instruction accounting under "Why the attention kernel sat 17-35x below the
   GEMM kernel" is for `srcAttnFwd`. Profiled per kernel, `srcAttnBwdDkv` was ~70% of the attention
   slice and `srcAttnFwd` ~15%: the whole discussion aimed at the wrong kernel.
2. **The SFU argument.** That same section concludes "issue slots are the first-order constraint,
   exp-count is second-order". Refuted: `exp` -> `exp2` alone gave 1.16-1.36x per kernel, including in `srcAttnBwdDkv`,
   which has no conditional rescale at all, so `exp` does not lower to one multiply plus one
   hardware instruction.
3. **Cross-entropy.** Scoped as saving one pass out of four. The actual defect was a kernel running
   on T threads total with no coalescing; a workgroup per row plus the fused pass gave 15.7x.

Parked, measured but not acted on while the step is host-bound (kernels are ~330 ms of a 10.6 s
step): `srcEmbeddingBwd` scaling (~0.3% of the step at 32768x640, multi-percent at 2x vocab, so
revisit if the vocab grows), `srcRmsNormBwdW`'s ~16x overfetch (~1% of the step), a RoPE table
precompute, and sliding-window warmup (~2-3% of the run at T=2048, worse at T>=4096). Chunked online
cross-entropy over the vocab axis, which would avoid materializing the `[T,V]` logits and buy
headroom for bigger batches at 8K, is deferred rather than executed: see lever 3 for the 1 GiB
logits tensor it targets. Two more stay open: 2D workgroup tiling for attention (a staged-forward
variant measured 17% SLOWER, `docs/notes/journal.md`), and cutting Newton-Schulz from five
iterations to four, which needs an orthogonality-residual check to gate it.

### FLOP accounting

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

### Why the attention kernel sat 17-35x below the GEMM kernel

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
the Strix device grants 64 KiB (journal probe), and the backend captures the granted cap in
`DeviceCaps.maxComputeWorkgroupStorageSize`. Sizing a tile against that GRANTED cap is exactly what
must not happen, and this sentence originally proposed it: it emits a kernel that runs on the
machine that built it and fails only for someone else, and no runtime here validates the difference.
`attnBwdTile` sizes against the 16 KiB floor unconditionally, and `tests/kernel-limits.ts` holds it
there.

## Throughput levers (the binding constraint)

### 1. Attention kernel: the ~78% slice: PARTLY RESOLVED (2026-08-18), 2.4x end-to-end on M1 Max

The 2026-07-08 reading of this line was that attention sat at a hardware floor. It did not; it sat
at a _codegen_ floor. Three restructurings had been tried and reverted (f16, split-K, query-register
tiling), and their failure was read as "the kernel is done". None of them touched the thing that was
actually costing the time: **every load in the head-dimension loops was scalar**, and the private
arrays holding a thread's Q row and output accumulator were f32 arrays indexed by a loop variable,
exactly the pattern that made the rolled GEMM 3x slower than the unrolled one.

Measured on an M1 Max with the new `bench` subcommand, min of 4 runs x 8 iterations, 95M geometry
(T=2048, 10 query heads over 5 KV heads, head-dim 64), GPU time per kernel:

| kernel                       |   before |   after | change |
| :--------------------------- | -------: | ------: | -----: |
| `srcAttnBwdDkv`, window 1024 | 225.7 ms | 55.2 ms |  4.09x |
| `srcAttnBwdDkv`, dense       | 376.2 ms | 89.7 ms |  4.19x |
| `srcAttnFwd`, window 1024    |  44.2 ms | 22.6 ms |  1.96x |
| `srcAttnFwd`, dense          |  81.3 ms | 35.6 ms |  2.28x |
| `srcAttnBwdDq`, window 1024  |  46.0 ms | 30.8 ms |  1.49x |
| `srcAttnBwdDq`, dense        |  84.8 ms | 50.3 ms |  1.69x |

The first correction the measurement made was to the target itself: **`srcAttnBwdDkv` was ~70% of
the attention slice**, not the forward kernel. The 78%-attention figure above was never broken out
per kernel, so the optimization effort had been aimed at the wrong one of the four.

Three changes, all plain portable WGSL, no intrinsics:

1. **vec4 lanes.** Q/K/V/output rows are addressed as `vec4<f32>` whenever the head dim is a
   multiple of 4, so a head-deep step is 1 load instead of 4 and the dot product carries 4
   independent accumulation chains instead of 1: the same two levers that lifted `srcGemm`. Head
   dims that are not a multiple of 4 keep the scalar form from the same codegen.
2. **exp2 domain.** The online softmax runs in log2 with the score scale pre-folded, and LSE is
   stored in the same domain so both backward kernels pick it up. `exp` is not the hardware
   instruction anywhere; `exp2` is.
3. **Conditional rescale** in the forward: the running max is monotone, so the accumulator is only
   rescaled on the keys that actually raise it, and on that branch `p` is exactly 1.

Plus two changes to `srcAttnBwdDkv` alone: a two-chain unroll of the head loop (1.18-1.20x there,
neutral in the other two, which is why it is not applied to them), and a query tile sized to fit the
16 KiB portable workgroup-storage floor. The old fixed 32-row tile needed 16640 B at head-dim 64,
over the floor, so the head size every published checkpoint uses was over budget on any
implementation that validates the limit. Halving it to 16 rows fits in 8320 B and is also 1.18x
FASTER on M1 Max: smaller tiles buy occupancy. (Two separate 1.18x factors compound into the
table's 4.09x, this one and the two-chain unroll above.)

That the bug had never been hit is not luck, and not a missing parity shape either: an hd=64 parity
case would have passed. What kept it working was the absence of any check. Probed on an M1 Max,
Deno's wgpu accepted a 16640-byte shader on a device reporting a granted
`maxComputeWorkgroupStorageSize` of **16384**, with an empty validation scope. The kernel was over
budget on every device it ever ran on, and was simply never told.

That is also why the gate is `tests/kernel-limits.ts` and not a GPU test. It parses the emitted WGSL
and asserts no kernel `wgsl.ts` builds declares more workgroup storage than the 16 KiB floor, across
every head dim the trainer accepts. A GPU test would have passed on the runtime the trainer actually
uses while the shape still failed for anyone on a stack that does validate. The check is also
independent of the fix by construction: `attnBwdTile` sizes the tile from a formula, the test sums
the array declarations, and neither consults the other.

End-to-end, 8 real pretrain steps at the 95M geometry, seq 2048, batch 2, on the M1 Max:
**204 -> 490 tokens/s (2.40x)**, with the per-step loss identical to four decimals at every step and
identical peak memory.

MEASURED ON STRIX (2026-08-19, idle GPU, min of 4 runs x 8 iterations), which the M1 Max table above
had been missing:

| kernel                            | GPU ms |
| :-------------------------------- | -----: |
| `linear`, tied readout, fwd + bwd |  5.322 |
| `attention.bwdDkv`, dense         |  1.177 |
| `crossEntropy`, T=2048 V=32768    |  0.769 |
| `attention.bwdDkv`, window 1024   |  0.730 |
| `attention.bwdDq`, dense          |  0.469 |
| `attention.fwd`, dense            |  0.440 |
| `linear`, FFN up, fwd + bwd       |  0.363 |
| `attention.bwdDq`, window 1024    |  0.301 |
| `attention.fwd`, window 1024      |  0.263 |
| `linear`, QKV, fwd + bwd          |  0.182 |
| `rmsnorm`                         |  0.038 |

End to end the rewrite is **1.76x on Strix** (903 -> 1588 tok/s), against 2.40x on the M1 Max. The
startup parity probe also tightened from |Δ|=6.0e-5 to |Δ|=2.4e-7, which is the exp2-domain softmax
and the independent vec4 accumulation chains being more accurate, not only faster.

The ranking inverted in the process: the tied readout GEMM is now 4.5x the largest attention kernel,
so "attention is ~78% of runtime" no longer describes this trainer. It is also moot, because the
whole kernel column sums to ~330 ms of a 10.6 s step (lever 1c).

Still standing, and still a research effort: the O(T^2) pair count itself. CONSTRAINT unchanged:
only approaches that stay portable, plain WGSL that runs cross-vendor (AMD/Apple/NVIDIA), no vendor
intrinsics or hardware-specific paths. A kernel that only helps gfx1151 at the cost of the "runs
anywhere" story is out of scope.

### 1b. GEMM and cross-entropy kernels (2026-08-18)

The same vec4 audit applied to the other two kernel families, measured the same way:

| kernel                                          |   before |    after | change |
| :---------------------------------------------- | -------: | -------: | -----: |
| `srcGemm`, tied readout [2048,640]x[32768,640]T | 337.7 ms | 183.0 ms |  1.85x |
| `srcGemm`, FFN up [2048,640]x[2560,640]T        |  19.2 ms |  15.8 ms |  1.22x |
| `srcGemm`, QKV [2048,640]x[1280,640]T           |  10.2 ms |   9.8 ms |  1.04x |
| cross-entropy fwd+bwd, T=2048 V=32768           |  73.7 ms |   4.7 ms |  15.7x |

GEMM: both staged tiles are held as vec4 and both micro-tile fragments are read as vec4, which
needs the A tile staged transposed (`As[k][m]`), because in the old `As[m][k]` layout a thread's
four rows were BK apart and a strided fragment cannot be one load. BK went 8 -> 16 (measured 1.09x
on top of vec4 at half the 16 KiB portable workgroup-storage floor; BK=32 was no faster and spends
the whole floor).

Cross-entropy: the old kernel ran **one thread per row**, so a [2048, 32768] softmax executed on
2048 threads with each lane striding a full row apart, coalescing nothing. It is now one workgroup
of 256 per row with a workgroup reduction, which is both 128x the parallelism and coalesced. On top
of that, `PROBS` now holds unnormalized `exp(z - max)` with a per-row `1/sum` beside it: the old
third pass over `[T,V]` existed only to apply a divide that the backward can apply from a scalar.
That also removes the `+1e-12` the loss needed to survive `log(0)`, since the loss is now
`log(sum) - (z_target - max)` straight from the row statistics.

A LARGER GEMM TILE WAS MEASURED AND REJECTED (2026-08-19). Six configurations, on Strix, comparing
the `linear` kernel column:

| BM/BN/BK/TM/TN         |    LDS | readout | ffn-up |   qkv |
| :--------------------- | -----: | ------: | -----: | ----: |
| 64/64/16/4/4 (current) |  8 KiB |   5.328 |  0.381 | 0.182 |
| 128/128/16/8/8         | 16 KiB |   5.724 |  0.324 | 0.187 |
| 128/128/8/8/8          |  8 KiB |   5.385 |  0.331 | 0.183 |
| 64/64/8/4/4            |  4 KiB |    6.41 |   0.49 |  0.20 |
| 128/64/16/8/4          | 12 KiB |    5.91 |   0.39 |  0.20 |
| 64/128/16/4/8          | 12 KiB |    6.44 |   0.47 |  0.21 |

A 128x128 tile with an 8x8 micro-tile fits the 16 KiB portable floor at 256 threads, so it was the
obvious candidate. It makes the readout GEMM WORSE and buys ~0.06 ms on the FFN GEMM, which is ~1.4
ms per micro-batch out of ~1500. Not applied.

Worth recording is how close this came to landing. An identical A/B run while a training run had the
GPU showed the readout going 5.60 -> 5.24 and looked like a clean 1.07x across all three shapes; the
direction reversed once the GPU was idle. An A/B taken while the box is doing something else
measures the something else. Kernel numbers in this file are min-of-4 on an idle GPU for that
reason.

Also note the device is running at the WebGPU portable defaults, not at what the hardware offers:
`initWebGPU` requests only the buffer limits, so the granted `maxComputeWorkgroupStorageSize` is
16 KiB against an adapter maximum of 64 KiB, `maxComputeInvocationsPerWorkgroup` is 256 against
1024, and `shader-f16` is supported but never requested. The startup banner reports the granted
values, which reads as a hardware statement but is really a statement about what was asked for.
Raising them is only worth doing if a kernel is ever the constraint again, and the 16 KiB floor is a
deliberate portability invariant that `tests/kernel-limits.ts` guards, so any use of more would have
to be adaptive rather than a raised floor.

Bind-group caching for the main loop (the `prepareDispatch` pattern the optimizers use) was measured
and NOT done: instrumenting `createBindGroup` over a real run gives 2346 calls and 71.5 ms per step
against an ~18 s step, i.e. 0.4%. It is also the exact pattern `prepareDispatch` warns about, since
the main loop's buffers are pooled transients that get recycled. Revisit only if the per-step GPU
work drops by an order of magnitude.

### 1c. The step is host-bound: where the 10.6 s actually goes (2026-08-19)

Wiring up the `onStepTime` hook the trainer already exposes, then splitting the phase inside the
micro-batch loop:

| phase                                        |           time |
| :------------------------------------------- | -------------: |
| optimizer (Muon + AdamW, all 28 tensors)     |           5 ms |
| `model.forward()`, per micro-batch, x8       | 1400 - 1650 ms |
| `backward()`, per micro-batch, x8            |     11 - 20 ms |
| end-of-step `gpu.sync()` (all GPU execution) |        1150 ms |

`model.forward()` is the whole step, and it is host time spent recording dispatches, not GPU time.
Proxying the ops backend shows it is not one op either: the cost tracks tensor size across all of
them (`linear` 235 ms over 85 calls, `rmsNorm` 36 ms over 49, `gelu` 34 ms over 12).

That points at allocation, and the `Tensor` constructor is why:

```ts
this.data = data;
this.grad = new Float32Array(data.length);
```

Every tensor gets TWO full-size host `Float32Array`s, including every intermediate activation under
the GPU backend, where both live on the device and are never read. Measured: **one forward allocates
4.34 GB of host array across 244 tensors.** Eight micro-batches are retained for the accumulation
step, so a step churns ~35 GB, which matched the 34.9 GB `smaps_rollup` reading taken during the
live run. (That reading did not reproduce later; see the negative result below.)

Honest accounting: warm-page allocation of 4.34 GB costs 2-4 ms, cold costs 126 ms, and an isolated
forward's op overhead is ~400 ms. That explains ~500 ms of the ~1500 ms per micro-batch. The
remainder is most likely GC against ~35 GB of live arrays, but that is NOT proven and should not be
quoted as if it were.

#### It was built, and it bought nothing (2026-08-19, later the same day)

Lazy `data`/`grad` did exactly what it was supposed to. `Tensor` allocates host storage on first
read, `size` comes from the shape so asking for it never allocates, and the CPU ops bind their
arrays to a local once per op instead of going through the getter per element (without that hoist
the CPU gradcheck ran 14% slower, 3.74 s to 4.27 s; with it, 3.75 s, level with baseline).

Instrumented over 2 steps at the real shape, 16 micro-batch forwards, 4,568 tensors:

|                  | per forward |        over 2 steps |
| :--------------- | ----------: | ------------------: |
| eager allocation |    4,395 MB |             70.3 GB |
| lazy allocation  |       48 MB |             0.77 GB |
| avoided          |             | **69.5 GB (98.9%)** |

Only 420 of 4,568 tensors ever touch host `data` and 174 ever touch host `grad`, which confirms the
4.34 GB figure above almost exactly. And it changed nothing:

| 150 steps, seq 2048, batch 8, interleaved arms, idle GPU |       eager |        lazy |
| :------------------------------------------------------- | ----------: | ----------: |
| throughput                                               | 2,651 tok/s | 2,627 tok/s |
| steady-state RSS                                         |    1,057 MB |    1,058 MB |
| final loss                                               |       5.045 |       5.045 |

So two claims above this line were wrong, and both were mine. The allocation does NOT explain the
34.9 GB resident: steady-state RSS is 1.06 GB on both arms and flat from 300 s on, because V8's
young-generation collector absorbs 4.4 GB per forward of short-lived `Float32Array` for free at this
rate. And allocation was NOT why the step is host-bound: removing 98.9% of it moved throughput by
less than 1%, inside the noise.

Where the 34.9 GB came from is unresolved. It was read once with `smaps_rollup` during the live run;
150 steps of the identical configuration will not reproduce it.

What survives: the step really is host-bound. Re-measured uncontended at the same shape,
`gpu_busy_percent` averages 52.5% over 40 samples (max 86%), so the GPU still idles about half the
step. The host time is in the dispatch path itself, not in allocating host arrays. Anyone taking
this on next should profile bind-group and pipeline setup per dispatch, not memory.

One loose end worth naming: this configuration reaches 0.161 st/s where the roleplay run logged
0.093 as its average (lever 5b: every resume runs ~750 steps slow, which is why the average sits
below the 0.0969 plateau), with byte-identical GPU allocation (39278 MB, pool 37714 + state 1564). A 1.7x gap that is
not yet attributed. The likeliest explanation is that the ten-hour run shared the GPU with the
benchmarking in this document, which is the same contention that reversed lever 1b's GEMM tile
result. Treat the 10.6 s/step baseline at the top of this file as an upper bound until that is
settled.

### 1d. Ruled out at the kernel level (measured, do not re-tread)

Measurements that closed a door; each cost real time to get.

| Idea                                | Outcome / reason                                                                                                                                                                |
| :---------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| f16 compute (f16 mul, f32 accum)    | 0.98x on attention at seq 4096-8192, plus overflow-to-NaN at step 2400 without clamps                                                                                           |
| f16 storage for Q/K/V               | 1.02-1.06x, and the gain shrinks as context grows                                                                                                                               |
| Split-K attention (32 threads/row)  | 0.4-0.7x: destroys the wave-uniform K/V broadcast                                                                                                                               |
| QT query-register tiling            | 0.80-0.94x at QT=2, 0.48-0.68x at QT=3: register pressure halves occupancy faster than reuse pays                                                                               |
| GEMM tile 128/128/16/8/8            | 1.07x under GPU contention, 0.93x on an idle GPU (min-of-4). The contended read was the artifact; see lever 1b                                                                  |
| WMMA / subgroup-matrix              | Not exposed by Deno's wgpu on gfx1151 (15 features probed, no `subgroups`); no WGSL matrix ops in the spec                                                                      |
| bf16 of any kind                    | No `bf16` WGSL type, no `shader-bf16` feature, no bf16 in naga                                                                                                                  |
| Fixed-max softmax via QK-norm bound | No safe static bound: q/k norm weights are trained, and under `qkClip` the observed max is 3.3-4.4x the proxy. The exp2-domain rescale gets the savings without needing a bound |
| Per-lane SWA window start           | Destroys the wave broadcast; the block-aligned start (`winStartBlock`) is why SWA is not slower than full attention                                                             |
| Lazy host `Tensor` storage          | Removes 98.9% of host allocation (4,395 MB/forward to 48 MB) and moves throughput <1%; see lever 1c                                                                             |

#### Machine independence

Every lever in this file, taken and rejected alike, is plain WGSL: no intrinsics, no vendor paths, workgroup memory sized against the
16 KiB spec floor rather than against the granted cap (see the retraction under "Where the step goes: the arithmetic": sizing to a
device's granted 64 KiB ships a kernel that only runs on that device), and no assumption about SFU
or load-pipeline rates. What differs across machines is the _share_ of the
bottleneck each lever addresses: the exp2 rescale matters most where the SFU is narrow (AMD
quarter-rate), vectorized loads where load-issue is the constraint, register tiling where the FMA
latency chain dominates. Measure the split on the target device before committing to a rewrite:
`startProfile`/`stopProfile` plus the parity gate make each step a ten-minute experiment, and the
gate is tolerance-based, so rounding-order changes are admissible.

### 2. True micro-batching (superseded premise, 2026-08-19)

Packing the `batchPerStep` sequences into one real batch dimension would enlarge the GEMMs and cut
per-launch + sync overhead. The old reasoning against it was that "GEMM is only ~9% of runtime and
the GPU is already saturated at batch 1": the GPU is NOT saturated, it idles ~58%. But the
conclusion survives for a different reason, which is that GPU work is only ~330 ms of the step at
all, so enlarging the GEMMs cannot buy much either. Lever 1c's proposed fix turned out to be a dead
end, so there is no longer a "fix that first"; the open question is where the dispatch-path host
time actually goes.

## Memory / scale levers

### 3. seq-8192 batch>1: RESOLVED, it was never a per-buffer limit

Re-probed 2026-08-04 on the idle GPU right after Phase A. **seq 8192 batch 2 and batch 4 both train
fine** (2 steps each, loss descending, gemma3 export OK). The old "RADV context lost at batch 2"
report does not reproduce; the flash-attention path plus the buffer pool removed whatever tripped
it. What remains is a plain memory wall, linear in tokens/step:

| config                       | tokens/step | peak GPU (pool + state) | throughput |
| ---------------------------- | ----------- | ----------------------- | ---------- |
| seq 2048 × batch 8 (Phase A) | 16384       | 39.3 GB (37.7 + 1.6)    | 903 tok/s  |
| seq 8192 × batch 2           | 16384       | 39.3 GB (37.7 + 1.6)    | 635 tok/s  |
| seq 8192 × batch 4           | 32768       | 76.2 GB (74.6 + 1.6)    | 653 tok/s  |

≈ 2.3 MB of activation pool per token in flight (micro-batch activations are all held for the
grad-accumulation step), so the 124 GB unified pool caps a step at ~50k tokens: **seq 8192 tops out
around batch 4-5**, not batch 1. No single buffer is near the 2048 MiB binding limit either: the
biggest is the per-micro-batch logits tensor, `T×V×4` = 1 GiB at T=8192/V=32768, and it does not
grow with batch. Phase B can budget batch 4.

The cost of long context is throughput, not memory: 8192 runs ~28% fewer tok/s than 2048 at equal
tokens/step (SWA covers 5 of 6 layers; the global layers still pay O(T²)).

### 4. More unique data: the binding constraint on quality (revised 2026-08-19)

The corpus is 722M unique tokens; the run does 2 epochs (~1.44B). `corpus` can emit more parts for
near-zero cost, but training them is the expense. Stated in tokens per parameter, the axis the
small-model literature argues on, against the models in the head-to-head table under lever 9:

| model                      | params | train tokens | tokens/param |
| :------------------------- | -----: | -----------: | -----------: |
| ours, `phaseA-final-88000` |  94.7M |        1.44B |           15 |
| Minueza-2-96M              |    96M |         185B |        1,927 |
| SmolLM2-135M               |   135M |          ~2T |      ~14,800 |

15 tokens per parameter is Chinchilla-optimal (~20:1), and therefore optimal for nothing this
project wants: Chinchilla minimizes loss for a fixed _training_ budget, not quality per parameter at
a fixed _model size_. Every model that beats us above trained two to three orders of magnitude
longer per parameter.

Throughput is 1588 tok/s sustained after the 2026-08-18 kernel rewrite (Strix, seq 2048 batch 8,
the plateau rate of the roleplay run), i.e. **137M tokens/day**, up from the 70M this section
used to assume. The arithmetic is still discouraging:

The "still to train" column is against the published model's 1.95B, not phaseA-final's 1.44B.

| target                         | total tokens | still to train | days at 137M/day |
| :----------------------------- | -----------: | -------------: | ---------------: |
| 100 tokens/param               |         9.5B |           7.5B |               55 |
| Minueza-2's 1,927 tokens/param |         185B |           183B |    1,335 (3.7 y) |
| SmolLM2's ~14,800 tokens/param |         1.4T |           1.4T |        ~28 years |

So closing the gap to Minueza-2 is not "a compute-time decision" as this section previously called
it: on one box it is out of reach. Reaching ~100 tokens/param is two months and is the only rung on
this ladder actually available. That, not architecture and not hyperparameters, is the ceiling on
quality here.

There is also a far end to this, which is worth knowing about even though it is not the end we are
near. [Extreme overtraining in tiny language models](https://huggingface.co/blog/Banaxi-Tech/ovdadadadd)
(2026-08-12) reports a 0.9M-param model whose scores peak around 22,000 tokens/param and decline
from there out to 222,000. That measurement is at 0.9M params, so the ratio itself does not carry to
94.7M (it would imply 2.1T tokens), and it is a single external run rather than something measured
here. The direction is what matters, and it agrees with SmolLM2: the useful ratio for small models
sits orders of magnitude above Chinchilla, with an eventual point of diminishing returns. We are at
the opposite end of that range by more than a factor of a thousand.

## Correctness / robustness

### 5. Checkpoint optimizer state (medium, real gap): DONE

Checkpoints now write a `<ckpt>.optstate` sidecar (Muon momentum + Adam moments + step) beside the
weights and restore it on `--resume`; absent -> cold start as before. `readStateBuffer` in the
backend does the readback. Validated: bit-exact round-trip, GPU parity unchanged, end-to-end resume.
(Phase A's early checkpoints predate this and have no sidecar, so they resume cold.)

### 5b. A resumed segment runs 15% slow for its first ~750 steps (2026-08-19)

Two independent resumes of the same run, from different checkpoints, both spent ~750 local steps at
0.084 st/s and then stepped to 0.0969 within one 75-step logging interval, landing on an identical
plateau:

| local step in segment | segment from 1000 | segment from 4350 |
| --------------------: | ----------------: | ----------------: |
|                   750 |            0.0850 |            0.0839 |
|                   825 |            0.0962 |            0.0954 |
|                   900 |            0.0969 |            0.0969 |

Reproducible to the same local step across two segments, so it is step-keyed and deterministic, not
thermal or environmental. **The operational consequence: every restart costs ~21 minutes of
throughput on top of the steps it loses**, which is far more than the 30-minute checkpoint window
suggests. Do not stop a run casually.

Cause NOT identified. Ruled out: the LR schedule (correctly offset, `schedule = (localStep) =>
fullSchedule(startStep + localStep)`, so warmup ends at a global step both segments were long past),
memory pressure (PSI zero across cpu/memory/io, no direct reclaim, no swap), thermals, competing
processes, and transparent hugepages (madvise mode, the trainer holds zero). V8 heap growth is also
unlikely: RSS is byte-identical before and after the jump.

Confounded and untested: checkpoints land every ~152 steps at this cadence, so the fifth one falls
at local step ~750 in both segments. "Fifth checkpoint" and "local step 750" are the same event in
this data. A segment run with a different `--checkpoint-every-minutes` would separate them for free.

### 6. rsync hygiene (cheap)

Plain rsync without `--delete` leaves stale files on the Strix copy when a source file is removed,
a real dual-machine drift bug. A small sync script with `--delete` (excluding `corpus/`, `*.gguf`,
`*.tokens`) removes the class. Moot if we run directly on Strix.

### 7. `--logEvery` flag (cheap)

`logEvery` is `steps/100` (880 steps ≈ 5 h between log lines here), so the first post-step-0 loss
reading is far out; checkpoints every 500 steps are the only early signal. A `--logEvery` flag would
give a tighter early-training view without touching the checkpoint cadence.

## Quality levers

### 8. WSD decay-phase instruct injection (medium): MECHANISM DONE

The MiniCPM / Xmodel-2 trick: blend a small fraction of ChatML instruct data into the WSD cooldown
so the base emerges more instructable. Implemented:
`pretrain --inject <tokens> --inject-fraction F --inject-from-step N` draws that fraction of micro-batches from
a second token source during the cooldown (default window = the WSD cooldown); off by default, and
the extra rng() is drawn only while active so parity is intact. Remaining to USE it: prepare an
instruct `.tokens` encoded with the shared tokenizer (smol-smoltalk → ChatML): tracked with the
instruct curriculum stage.

### 9. Eval harness (medium, validates the whole effort): DONE, first full run recorded

`eval-choice` scores a GGUF on ARC-Challenge / HellaSwag via length- normalized log-likelihood
(matches lm-eval-harness, comparable to the Minueza leaderboard numbers), plus acc/acc_norm;
datasets via the HF parquet tooling.

**Phase A base, `phaseA-final-88000.gguf` (step 88000, 1.44B tokens), run 2026-08-04:**

| eval                                                          | result                                   |
| ------------------------------------------------------------- | ---------------------------------------- |
| ARC-Challenge, full 1172 items, 0-shot                        | acc_norm **22.78%**, acc 19.28% (648 s)  |
| HellaSwag, 2000 of 10042 items, 0-shot                        | acc_norm **32.20%**, acc 31.50% (1179 s) |
| `eval-loss`, 64 × 512 tok, in-distribution tail 1%, seed 1234 | loss **3.0915**, ppl 22.01               |

HellaSwag clears chance (25%) by 7 points; ARC-Challenge sits just under it, which is what a 94.7M
base at 1.44B tokens looks like: ARC needs knowledge this token budget cannot buy. Only 2000
HellaSwag items were scored, so that figure carries ~±1% sampling noise; the ARC number is the full
set.

**The cross-model comparison is CLOSED (2026-08-05).** At the time, the loader accepted only the
one architecture it had, so `eval-choice` could not score the reference models. (It is
architecture-agnostic today, but it still only reads GGUFs this project produced.) Everything below
was scored by llama.cpp's `llama-perplexity --multiple-choice` instead, same binary, same task
order, on CPU.
`scripts/minueza-3-run/head-to-head.sh` runs it. ARC-Challenge = all 299 validation tasks, HellaSwag
= all 10042, from llama.cpp's canonical sets (`ikawrakow/validation-datasets-for-llama.cpp`, in
`evaldata/`).

| model                                | train tokens | ARC-Challenge | HellaSwag       |
| ------------------------------------ | ------------ | ------------- | --------------- |
| SmolLM2-135M                         | ~2T          | 31.44 ±2.69   | 42.81 ±0.49     |
| Supra2-100M-Base                     | 30B          | 27.42 ±2.58   | 35.31 ±0.48     |
| llama-160m                           | ?            | 22.74 ±2.43   | 33.94 ±0.47     |
| **ours, phaseA-final-88000 (94.7M)** | **1.44B**    | 21.74 ±2.39   | **28.46 ±0.45** |
| Minueza-2-96M                        | 185B         | 23.08 ±2.44   | 27.03 ±0.44     |
| Minueza-32M-Base                     | ?            | 22.74 ±2.43   | 25.75 ±0.44     |
| Qwen2-96M                            | ?            | 23.41 ±2.45   | 24.85 ±0.43     |

**The "Minueza-3" naming now has data behind it, on HellaSwag.** We clear Minueza-2-96M by 1.43
points (28.46 vs 27.03; combined uncertainty ~0.63, so ~2.3σ) and Minueza-32M-Base by 2.7. On
ARC-Challenge only SmolLM2 (31.44 ±2.69) is clearly above chance. Supra2 (27.42 ±2.58) is within one
error bar of chance, and the remaining five sit between 21.74 and 23.41, so no ranking among
them is meaningful, ours included. Two consistency checks: HellaSwag scored 28.4704 and
28.4605 on two independent runs of our model (task order is deterministic), and Supra2's 35.31 here
is close to the 0.36 acc_norm its card reports under the EleutherAI LM-Eval Harness, which suggests
the two rulers agree at this scale even though they normalize differently. Do NOT mix our
`eval-choice` numbers with these: the same checkpoint scores 32.20 acc_norm there (on 2000 items)
and 28.46 here.

**Gotcha that silently voids an ARC run:** `llama-perplexity` decodes a task's answers in parallel
and ABORTS the whole run when a task has more options than `-np` allows. ARC-Challenge contains
5-option questions, so the default 4 dies at task 210 of 299 and prints no result. Pass `-np 8`.

Scoring the reference models needed one GGUF that did not exist, so it now does:
[Felladrin/gguf-f16-Supra2-100M-Base](https://huggingface.co/Felladrin/gguf-f16-Supra2-100M-Base)
(F16 conversion of `SupraLabs/Supra2-100M-Base`, apache-2.0). llama.cpp does not recognize that
model's BPE pre-tokenizer, and its `Sequence[Digits(individual_digits), ByteLevel]` is exactly what
llama.cpp's `qwen2` pre-type implements (bare `\p{N}` plus byte-level), so the conversion asserts
that mapping; token ids were verified identical to the source `transformers` tokenizer.

SupraLabs already ship a correct F16 GGUF of their Instruct model (`Supra2-100M-SFT-F16.gguf`, same
`qwen2` pre-type, chat template embedded), so a faithful re-conversion would add nothing. What was
published instead is a one-metadata-change variant,
[Felladrin/gguf-f16-Supra2-100M-Instruct](https://huggingface.co/Felladrin/gguf-f16-Supra2-100M-Instruct):
the source declares `</s>` as EOS while its chat template ends every turn with `<|im_end|>`, so
there `<|im_end|>` (id 6) is registered as both EOS and EOT. llama.cpp stops correctly either way
(it treats the `<|im_end|>` NAME as end-of-generation whatever the EOS metadata says, verified by
generating past a turn boundary on both files), so the variant only helps runtimes without that
heuristic. All 134 tensors are byte-identical (sha256 per tensor) between the two files, which is
also the cleanest proof that our conversion path reproduces theirs exactly.

### 9b. The four-task score, finally measured (2026-08-20)

`eval-choice` gained ARC-Easy and PIQA, so the Open SLM Leaderboard's Intelligence Index is now
computable for our own checkpoints instead of estimated. Full sets, 0-shot, on the roleplay
continued-pretrain (`rp-full`, 94.7M):

| Task          |  Items | acc_norm | chance | normalized |
| :------------ | -----: | -------: | -----: | ---------: |
| PIQA          |  1,838 |   61.32% |     50 |      22.64 |
| ARC-Easy      |  2,376 |   40.03% |     25 |          - |
| ARC-Challenge |  1,172 |   22.61% |     25 |          - |
| ARC (mean)    |      - |   31.32% |     25 |       8.43 |
| HellaSwag     | 10,042 |   28.16% |     25 |       4.21 |

The board's formula normalizes each task against its chance floor, `N = 100 x (score - chance) /
(100 - chance)`, averages ARC-Easy and ARC-Challenge into ONE ARC term before normalizing, and
weights ArithMark-3 at 0.65:

    Index = (HellaSwag + ARC + PIQA + 0.65 x ArithMark) / 3.65
          = (4.21 + 8.43 + 22.64 + 0) / 3.65 = 9.67

**Intelligence Index 9.67.** ArithMark-3 is not implemented here, so it is assumed at chance;
omitting the term entirely gives 35.28 / 3 = 11.76, making the honest range 9.7-11.8. Against the
board, 9.67 places 48th of 130 counting ours: it ranks 131 models, of which 129 carry the complete
task data the index needs.

Three things to carry forward. ARC-Challenge at 22.61% is below its 25% chance floor, but lever 9's
head-to-head shows five of seven models between 21.74 and 23.41 under length-normalized scoring, so
this is a property of the ruler at this scale, not a defect of ours. PIQA carries the whole index
(22.64 against HellaSwag's 4.21) partly because two-option normalization divides by 50 rather than
75, which inflates any edge over chance. And lever 9's 32.20 on a 2000-item HellaSwag subset is
not reconciled by this run: 28.16 here is a different checkpoint and lever 9's 28.46 is a different
harness, so no pair isolates the subset. The clean test, `eval-choice` on phaseA-final over the full
10,042, has not been run.

### 10. Phase B KL anchor against the base checkpoint (medium): OP DONE

The continual-learning use of distillation (HF post `sergiopaniego/distillation-2026`): during Phase
B SFT, add a per-token KL term against the frozen Phase A base so its fluency survives fine-tuning;
models this small forget catastrophically under plain SFT. The teacher is this model, so the u16
custom-BPE vocab is no obstacle (external teachers are ruled out by it; see below). One new op: a
soft-target cross-entropy whose backward is `(p - q)`, where the existing `crossEntropy` backward is
`(p - onehot)` (`src/model/autograd.ts`), reusing its ignore-index masking. Phase B data is fixed,
so teacher logits are precomputed once over the SFT `.tokens` with the Phase A checkpoint and stored
top-k (~100 B/token at top-16, single-digit GB): no second model in GPU memory, zero per-step
teacher cost. CPU reference + WGSL port + gradient check follow the existing op pattern and need no
GPU, so it is buildable while Phase A trains. The same op later enables Cursor-style hint
self-distillation (teacher = same weights with an instruct prefix, student = bare, KL on the
response tokens); an online teacher forward adds roughly a third of a step (forward-only,
attention-bound), affordable at this scale.

**The op shipped (2026-08-04): `softCrossEntropy(logits, teacherIds, teacherProbs, k)`.** Sparse
teacher, `[T*k]` ids + probs per row; a row is ignored when its first id is negative, the same
convention `crossEntropy` uses, so assistant-only masking carries over unchanged. The teacher mass
need not be normalized: with top-k truncation it sums to `S <= 1` and the exact gradient is
`S*p - q`, which is the documented `(p - q)` when `S = 1`. The reported value is a cross-entropy in
nats (it differs from `KL(q||p)` by the teacher's entropy, a constant in the student's parameters),
so it is directly comparable to the hard-target loss. WGSL: `srcSoftCeFwd` reuses the `crossEntropy`
row-softmax and shares `srcCeReduce`; the backward is two ordered dispatches, dense `+S*p` over
`[T,V]` then sparse `-q` over the k entries, the sparse one **one thread per row** so duplicate
teacher ids in a row cannot race. Verified by `tests/gradcheck.ts` (finite differences, covering a
normalized row, a truncated row, an ignored row and a duplicate id) and `tests/gpu-parity.ts` (GPU
vs CPU, same four cases).

Not built yet: the teacher-logit precompute pass over the SFT `.tokens` and the SFT loop that sums
`crossEntropy + lambda * softCrossEntropy`. Note the composed form costs two row-softmaxes and two
`[T,V]` probability buffers (268 MB each at seq 2048, 1.07 GB at seq 8192); if that ever binds, fuse
both targets into a single op sharing one softmax.

### 11. What the sub-150M field does differently (2026-08-19)

The [Open SLM Leaderboard](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard) ranks 131
models under 150M on an Intelligence Index over HellaSwag, a combined ARC term (Easy and Challenge
averaged BEFORE normalizing), PIQA and ArithMark-3, each normalized so chance maps to 0, with
ArithMark-3 weighted 0.65. Lever 9b spells out the arithmetic. Three things separate the top of the
80-155M cohort from this project, and not one of them is an optimizer or a kernel. Two models is not
a controlled study, so read these as where to look, not as proven causes.

**Token budget, restated against real competitors.** Lever 4 makes this argument from Minueza-2 and
SmolLM2; the board says the same thing with models that are not outliers:

| model              | params |       tokens | tokens/param | Int Index |
| :----------------- | -----: | -----------: | -----------: | --------: |
| SmolLM2-135M       |   135M |          ~2T |       14,815 |     27.13 |
| GPT-X2.5-135M      |   135M |          75B |          556 |     25.17 |
| BananaMind-2-Pro   |   139M |         100B |          719 |     24.96 |
| Supra2-100M-Base   |   101M |          30B |          298 |     19.41 |
| ours, phaseA-final |  94.7M |        1.44B |       **15** |  unscored |
| ours, rp-full      |  94.7M | 1.95B + 123M |     **21.9** |      9.67 |

**Depth over width, and a 3x FFN rather than 4x.** Both top non-HuggingFace models spend parameters
on layers instead of on a wide FFN:

| model            | layers | hidden |   FFN | ratio | heads (Q/KV) | trained ctx |
| :--------------- | -----: | -----: | ----: | ----: | :----------- | ----------: |
| GPT-X2.5-135M    |     30 |    576 | 1,728 |  3.0x | 9 / 3        |       8,192 |
| BananaMind-2-Pro |     24 |    640 | 1,920 |  3.0x | 8 / 4        |       3,072 |
| ours             | **12** |    640 | 2,560 |  4.0x | 10 / 5       |       2,048 |

`gemma3Config` derives the FFN as ~4x hidden, so trying 3x and spending the savings on depth is a
one-config experiment, not a code change. Worth an A/B before the next from-scratch run.

**A data mixture we do not have.** Both report nearly the same blend: FineWeb-Edu ~50%, DCLM ~26%,
Cosmopedia-v2 ~13.5%, FineMath-4+ ~8%, Python ~2%. Ours is FineWeb-Edu and nothing else from that
list. No FineMath is the most likely reason arithmetic and ARC sit at chance for us, and it is the
cheapest of the three gaps to close.

Both also use plain AdamW at peak lr 1.5e-3 with a 2,000-step warmup, so Muon is not what separates
them from us. GPT-X2.5 uses WSD with the decay confined to the last 10%, against our 20% cooldown.

Not transferable, despite ranking 7th at 90M: `palmer-006` discloses no token count, no datasets and
no optimizer, and describes a merge plus a light finetune of an unnamed base.

## Explicitly not worth doing

- **Guarded/clamped f16 compute**: 0.98x measured on attention at seq 4096-8192, plus
  overflow-to-NaN at step 2400 without clamps. f32 also learns better. (Confirmed; f16-compute path
  removed.)
- **Using all 128 GB**, we are compute-bound; extra RAM buys nothing at this model size. It only
  matters as headroom for a much larger model, which the throughput ceiling makes impractical
  anyway.
- **Switching to PyTorch + ROCm for speed**: still not doing it, but the old reason was wrong and is
  worth retiring rather than repeating. It said "the attention wall is the GPU, not the framework".
  GPT-X2.5-135M trained 75B tokens on a single RTX 3080 Ti in ~800 hours, which is ~26,000 tokens/s
  on a LARGER model than ours against our 1588 on Strix. A mature stack on lesser hardware being 16x
  faster is exactly the framework being implicated, so that clause does not survive. What still
  holds, and is the whole reason, is the project direction: portable plain WGSL that runs
  cross-vendor is the goal, not chasing throughput on this box. ROCm on gfx1151 also lacks
  flash-attention.
- **Soft-label KD from off-the-shelf teachers** (e.g. Gemma-3-1B as teacher): vocab mismatch (our
  custom u16 BPE vs their 262k tokenizer), and cross-tokenizer KD (GOLD/ULD) is research-grade
  machinery. Data-level KD is already the corpus strategy (TinyStories and smol-smoltalk are
  teacher-generated); same-vocab distillation stays available via the checkpoint anchor (#10).

## References

- Repo measurements: `docs/notes/journal.md` (kernel rewrites, reverted attempts, remaining
  roadmap), `docs/design.md` (precision and backend bring-up).
- RDNA3.5 (gfx1151) speed-of-light rates, VOPD: [rocm.docs.amd.com](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/conceptual/rdna/system-speed-of-light.html)
- RDNA SFU/TFU, `v_exp_f32`, LDS bandwidth, Wave32: [rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- SFU quarter-rate (RDNA2, applied by class): [nelcit.github.io](https://nelcit.github.io/shader-clippy/blog/pow-const-squared)
- WebGPU limit defaults (16 KiB workgroup storage floor): [developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Web/API/GPUSupportedLimits)
- WGSL spec (`exp2`, atomics, workgroup memory): [w3.org](https://www.w3.org/TR/WGSL)
- Bind-group reuse guidance: [toji.dev](https://toji.dev/webgpu-best-practices)
- Prior art, browser WGSL training (forward+backward+AdamW, online-softmax attention; small
  scale, no published throughput at 95M): [github.com](https://github.com/toprakdeviren/webgpu-llm)
- Fused linear-cross-entropy prior art (CUDA/Triton; concept reference for the deferred chunked-CE idea above):
  [github.com](https://github.com/linkedin/Liger-Kernel), [github.com](https://github.com/mgmalek/efficient_cross_entropy)
- FlashAttention-3 (what full 2D tiling + tensor cores buys on NVIDIA; context for why the same
  structure is not automatically fast on a no-TC, no-subgroup WebGPU path):
  [arxiv.org](https://arxiv.org/abs/2407.08608)
