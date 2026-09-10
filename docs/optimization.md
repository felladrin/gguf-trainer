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
busy ~100% (a whole-device reading with its other users unrecorded, the signature lever 47 puts
under suspicion; the 1.76x below rests on tok/s, not on it), attention ~78% of runtime, f16 and f32
compute the same speed. Two things that
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
| GPU busy                | ~42% / 52.5% [^busy]      | `gpu_busy_percent`; 42% during the run, 52.5% re-measured uncontended |
| Host CPU                | ~400% of 32 cores         | `top` on the trainer process                                          |
| Host RSS                | 1.06 GB steady            | sampled every 10s over 150 steps; flat from 300s on                   |
| Peak GPU (pool + state) | 39.3 GB                   | trainer's own readout, unchanged                                      |
| Profiled kernel time    | ~330 ms of a 10.6 s step  | `bench`, idle GPU, summed over a step's dispatches                    |

Both GPU-busy figures are whole-device readings whose other users were not recorded, which lever 47
calibrated and sets aside; read them as ceilings on this trainer's share rather than as its
utilization. The profiled kernel time is the per-process number.

The two facts that now drive everything below:

1. **The step is host-bound, not GPU-bound.** Profiled kernel time is ~330 ms against a step of
   seconds: a reconstruction over the kernel families `bench` times, not a profile of a real step,
   since `bench` builds no model and has no layer loop, and its per-dispatch profiling inflates
   totals. The gap it has to close is roughly 30x, so it closes comfortably either way.
   Cheaper kernels can no longer raise tokens/second on this box; only cheaper host work can. See
   lever 1c, which is where the time actually goes.

   `gpu_busy_percent` corroborates rather than establishes this, and lever 47 says why: it is a
   whole-device counter, so its 52.5% is a ceiling on our trainer's share rather than a measure of
   it, and the `~42%` beside it in the table was taken during a run whose other device users are
   unrecorded.
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
cross-entropy over the vocab axis is no longer deferred: it shipped as `--loss-chunk`, and it had to
fuse the readout matmul rather than only chunk the loss, because the `[T,V]` logits and their
gradient belong to that matmul and outweigh the softmax scratch 2:1 (lever 19). Two more stay
open: 2D workgroup tiling for attention (a staged-forward
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
step. (Lever 47 later calibrated that counter and narrowed what this sentence can claim: it is
whole-device, "uncontended" here was not checked against `fuser -v /dev/dri/renderD128`, and the
counter reads 43.2% mean with nothing training at all. Read 52.5% as a ceiling on this trainer's
share, not as its utilization. Lever 47 also found an unrecorded tenant on this box's GPU,
which does not corroborate the loose end below about the ten-hour run, a different month and a
different tenant, but does make it more plausible that this box has them.)

The host time is in the dispatch path itself, not in allocating host arrays. Anyone taking
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
the GPU is already saturated at batch 1": the GPU is NOT saturated, it idles about half the step. But the
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

### 3b. `--reclaim`: 82% of the memory for 23% of the throughput (2026-08-20)

Lever 3 measured the wall; this measures the way through it. `--reclaim` frees each micro-batch's
activations at the micro-batch boundary instead of holding all of them for the grad-accumulation
step, so peak memory tracks one micro-batch rather than `batch`.

A/B at identical shape and seed, 20 steps, seq 2048 x batch 8, resumed from the same checkpoint:

| `--reclaim` | throughput | peak GPU (pool + state) | loss (first 10 -> last 10) |
| ----------- | ---------- | ----------------------- | -------------------------- |
| off         | 1341 tok/s | 39.3 GB (37.7 + 1.6)    | 2.708 -> 2.631             |
| on          | 1031 tok/s | 7.0 GB (5.4 + 1.6)      | 2.708 -> 2.631             |

The loss matches to four digits, which is the runtime confirmation of the `reclaimTransients`
parity test in `tests/gpu-parity.ts`: reclaiming changes the allocator, not the math.

The 23% is the cost of the drain. `reclaimStepTransients` ends the pass, submits, and awaits
`onSubmittedWorkDone` at every micro-batch boundary, so a step that used to hand the GPU one queue
submission now hands it `batch` of them, and lever 1c already established the step is host-bound.

**The flag stays off by default.** On a machine with memory to spare the trade is bad: throughput is
the scarce resource, and 23% turns an 18-day phase A into 23 days. On a machine without it the trade
is not a trade, because the alternative is an OOM. Reach for it when a run does not fit, and
remember that at `--batch 1` it is a no-op (there is no boundary to reclaim at).

Not yet measured: whether reclaim-on at a larger batch beats reclaim-off inside the same memory
budget. A larger batch amortizes the per-step host overhead, so it might; until someone runs it,
the table above is the only claim this file makes.

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

### 19. Chunked fused cross-entropy: 3.7 GB, and the 4K context wall (2026-09-09)

Lever 3 named the per-micro-batch logits tensor as the largest single buffer and left chunking it as
future work. Executed here, and the scoping in that note was one third of the problem: chunking the
cross-entropy _op_ removes only `probs`. The `[T,V]` data and gradient buffers belong to the readout
`linear`, so the readout matmul has to be fused into the loss to reach them. Three buffers, not one.

`--loss-chunk N` streams the vocab N columns at a time: for each chunk, `hidden @ Wchunkᵀ` into one
reused `[T,N]` scratch, an online-softmax update of the running (max, sum) with rescaling, and in
backward the same matmul recomputed before the chunk's gradient is turned into `dHidden` and `dW`.
The widest live buffer stops scaling with vocab. `srcGemm` grew a baked row offset so a chunk can
address rows of a `[vocab, hidden]` weight without binding a sub-range (a pooled buffer is always
bound whole); at offset 0 it emits byte-identical source, so no existing pipeline changed.

Measured on qwen3 293M (vocab 151936, hidden 544, 28 layers), `--seq-len 2048 --batch 2 --reclaim`,
6 steps, same seed:

| `--loss-chunk` | throughput | peak GPU (pool + state) | loss (first 3 -> last 3) |
| -------------- | ---------- | ----------------------- | ------------------------ |
| off (dense)    | 73 tok/s   | 22548 MB (18053 + 4495) | 5.140 -> 6.884           |
| 8192           | 73 tok/s   | 18881 MB (14386 + 4495) | 5.139 -> 6.881           |

The pool drops 3667 MB. Predicted: three `[T,V]` buffers at 2048x151936x4 = 3735 MB, less the 67 MB
`[T,8192]` scratch that replaces them, so 3668 MB. The loss agreeing to three decimals rather than
four is the expected cost of reordering an f32 reduction over the vocab axis; the gpu-parity case
compares the fused path against `crossEntropy(linear(...))` on the same inputs and holds to `BWD`
tolerance.

**The throughput column is the surprising one.** An extra full readout matmul per step should cost
something, and it costs nothing measurable, because lever 1c already found the step host-bound
(`gpu_busy_percent` ~52.5%, a whole-device ceiling; lever 47): the added GPU work lands in a gap
that was already idle. Do not
generalize that to a GPU-bound shape.

What it actually unlocks is context, not memory. At vocab 151936 the logits buffer is 1.16 GiB at
T=2048 and 2.32 GiB at T=4096, against the 2048 MiB `maxStorageBufferBindingSize` Strix grants, so
Qwen3-0.6B-shaped models at 4K did not train at any amount of free memory. Confirmed both ways:

```
--seq-len 4096            -> error: GPU storage buffer of 2374 MiB exceeds this device's limit of 2048 MiB
--seq-len 4096 --loss-chunk 8192 -> trains, 74 tok/s, peak 30820 MB (pool 26325 + state 4495)
```

`--loss-chunk` is capped at 100 spans because `voff` is baked into the kernel source, so each span
costs its own pipelines. In the two CE kernels the offset is read in exactly two places, so a
1-element u32 uniform would collapse 19 stats pipelines and 19 grad pipelines to 2 each and make
the ceiling nearly free to raise. The GEMM offsets are the harder half, and their byte-identity
property at `off = 0` is worth keeping, so this is the lever that removes the cap, not a defect.

`eval-loss` and `eval-choice` took the flag later, in the same shape: both build the same
`crossEntropy(model.forward(...), targets)` and both now call `sequenceLoss`. `eval-choice` needed
no other change, because it recovers a summed NLL as mean-times-kept-count and the chunked path uses
the same denominator; `tests/gradcheck.ts` pins that against an independently computed sum of
per-position `-log softmax`, not against either implementation, so a change moving BOTH to a
different denominator fails it rather than cancelling out. Measured:
`eval-loss --seq-len 4096` on a 151936-vocab checkpoint aborts dense and scores with
`--loss-chunk 8192`, and dense and chunked return identical numbers at seq 512 (val loss 3.6200 from
`eval-loss`, piqa acc_norm 60.00% over 30 items from `eval-choice`).

`generate` is the remaining dense forward: it builds `[T, vocab]` logits and reads only the last
row, so it hits the same limit at long context and `--loss-chunk` cannot help. It needs a
last-row-only readout, which is a different change.

Not done here: `softCrossEntropy` (the Phase B KL anchor) still materializes `[T,V]` through the
dense readout, so `--loss-chunk` does not apply to it. Chunking it means fusing the same readout
into `srcSoftCeFwd`, and the sparse teacher makes the gradient `S·p − q` rather than `p − q`.

### 20. Activation recomputation: 4.6x less pool, and 2.3x FASTER (2026-09-09)

`--recompute` replays each layer in backward instead of keeping its interior. Textbook gradient
checkpointing, named `--recompute` here because a checkpoint in this repo is a saved GGUF.

Measured on qwen3 293M (vocab 151936, hidden 544, 28 layers), `--seq-len 2048 --batch 2 --reclaim
--loss-chunk 8192`, 6 steps, same seed, and re-run once to confirm (identical to the digit). The
`--heads` this run used is not recorded here, which lever 47 needed and had to guess at; the ratio
reproduces there at 2.25x on an idle machine, with both absolutes about 1.5x higher:

| `--recompute` | throughput | peak GPU (pool + state) | loss (first 3 -> last 3) |
| ------------- | ---------- | ----------------------- | ------------------------ |
| off           | 72 tok/s   | 18974 MB (14480 + 4495) | 5.139 -> 6.881           |
| on            | 168 tok/s  | 7658 MB (3163 + 4495)   | 5.139 -> 6.881           |

The pool falls 4.6x. Weights and gradients are 2346 MB of what remains, so the activation term
itself goes from ~12.1 GB to ~0.8 GB.

**The throughput is the part to be suspicious of, so here is the evidence it is real.** The loss
matches to four digits over six steps and the parity suite is bit-exact under recompute (a replay
of the same ops reorders nothing, unlike chunking), so no work is being skipped. The baseline arm
also reproduces lever 19's numbers (72 vs 73 tok/s, pool 14480 vs 14386), so it is not a slow
control.

The mechanism is inferred, not proven. Lever 1c found the step host-bound at `gpu_busy_percent`
~52.5%, a whole-device ceiling rather than this trainer's share (lever 47), and `endRegion` ends
the pass and submits at every layer boundary. Before this, a whole
micro-batch was recorded into one compute pass and submitted once, so the GPU sat idle while the
host recorded 28 layers and then raced to catch up. Now layer 1 executes while the host records
layer 5. On that reading the extra forward pass is free because it lands in time the GPU was
already spending idle, and the win is overlap rather than arithmetic. **A GPU-bound shape should
expect the textbook ~30% slowdown instead.** `bench` cannot answer that question: it times kernel
families at fixed shapes, builds no model and has no layer loop, so measure it the way this table
was measured, with two short `pretrain` runs differing only in the flag.

If that reading is right, the same overlap is available without recomputing anything, by submitting
at layer boundaries on the dense path too. That is the obvious follow-up and it is not done here.
**Measured in lever 47, on an idle machine: no effect.** The earlier numbers in that lever's drafts
were background-load artifacts worth 2x, which is the trap it now exists to name.

The corollary is a trap worth naming: the `submit()` in `endRegion` is not needed for correctness.
It is what produces the overlap. Deleting it as redundant keeps the whole suite green and silently
returns the throughput to the dense number, so the comment there says so. **Still unmeasured:**
lever 47 tried and had to discard the arm, because it was run while the machine was busy.

The reason it is not needed for correctness is not the one this lever gave. It said no region
buffer ever reaches a `queue.writeBuffer` call site, and that is false: the loss backwards write
their seed into an `eo.grad` that `makeOut` does draw from `regionFree` in a `--recompute` run
(measured: 44 buffers off that list on a tiny gemma3 forward, the loss's grad among them). What
saves it is that `ensureBackwardBegun` submits immediately before that write, so it cannot run ahead
of a reader. The conclusion stands, the reason has been replaced, in `endRegion`'s
docstring and here. Lever 41 has the general form of the argument.

**This looks like it contradicts lever 3b, and does not.** 3b costs 23% by submitting once per
micro-batch on the same host-bound step; 20 gains 2.3x by submitting once per layer on it. The
difference is not the frequency, it is the wait. `reclaimStepTransients` ends the pass, submits,
and then AWAITS `onSubmittedWorkDone`, which drains the pipeline and stalls the host until the GPU
catches up. `endRegion` submits and returns. Submitting is the overlap; waiting for the submission
is the stall. That also sharpened the follow-up above, while it was still
open: a dense-path version had to submit without a fence, or it would reproduce 3b's 23% rather than
this lever's 2.3x. Lever 47 built it that way and measured no effect, so this is a record of the
reasoning rather than a decision anyone still has to take.

Correctness is gated three ways rather than by the loss curve: `checkpoint == off` in
`tests/gradcheck.ts` requires bit-identical gradients, `recomputeModelParity` runs all three
architectures against the CPU reference, and `recompute across reclaim boundaries` drives
`trainLMGpuResident` with reclaim on and requires a bit-identical loss.

The memory claim has its own gate, because no numeric test can see it: a region buffer that never
returns to the pool leaves every number correct and quietly allocates around it. That is not
hypothetical, it was the first version of this change. `recomputeMemoryGate` asserts it structurally, one
claim per drain, so no fitted constant carries the weight: with reclaim off the pool must not grow
with step count (10.9 -> 10.9 MB healthy, 14.4 -> 20.8 MB with the `sync()` drain gone), and with
reclaim on it must not grow with micro-batch count (7.6 -> 7.6 MB healthy, 10.0 -> 12.2 MB with the
`reclaimStepTransients` drain gone). Each deletion was verified to fail its own arm.

### 21. LoRA: the optimizer state goes 71x, the throughput goes down 14% (2026-09-09)

`--lora-rank N` freezes every parameter the model already had and trains rank-N adapters on the
matrices in the Muon group. Attached in `src/train/lora.ts` off `paramGroups().muon`, which is
already exactly the set of 2-D hidden projections, and registered with `linear` rather than with
the architectures: nothing in `src/arch/` knows adapters exist, and a new architecture gets them
for free. Adapters train under AdamW at `--aux-lr`; `--muon-lr` is inert, because the Muon group is
empty.

Measured on qwen3 293M (vocab 151936), `--seq-len 1024 --batch 1 --reclaim --loss-chunk 8192
--recompute`, 3 steps, same seed:

| mode             | throughput | peak GPU (pool + state) | trainable     |
| ---------------- | ---------- | ----------------------- | ------------- |
| full             | 43 tok/s   | 12838 MB (8343 + 4495)  | 293.3M        |
| `--lora-rank 16` | 37 tok/s   | 7554 MB (7490 + 63)     | 7.90M (2.69%) |

**Optimizer state is the whole story: 4495 MB to 63 MB, 71x.** That term is `16 B/param` for Muon
matrices and `8 B/param` for aux (lever 19's arithmetic), so it is the one cost that scales purely
with the trainable count, and the one that dominates at any model worth adapting.

**Freeze everything, not just what you adapt.** The first version froze only the matrices being
adapted and dropped the rest of the parameters from the optimizer without freezing them. They were
untrained either way, but they kept their gradient accumulators, kept dispatching their backward
kernels, and, for a tied 151936x1024 readout, kept `sync()` staging a 622 MB device-to-host copy of
an embedding gradient nothing read, every step, into a staging buffer `residentBytes()` does not
even count. Freezing the whole non-adapter set moved the measured numbers from 30 to 37 tok/s and
7884 to 7554 MB, and it is the honest statement of what LoRA trains here: adapters only, with the
embeddings, norms and readout held fixed.

**Throughput still drops 14%, and that is the real cost.** 196 projections become 196 x 3 matmuls
plus an add and a scale, so a forward records several hundred extra dispatches into lever 1c's
host-bound step. LoRA here buys memory, not speed. Fusing the adapter into the base GEMM would
change that, and is not done.

Three design choices worth recording, because each had an alternative:

**Adapters go on AdamW, not Muon.** Muon orthogonalizes the update matrix through Newton-Schulz,
and running that on `A` and `B` separately is not orthogonalizing `B*A`. It is a different
algorithm with no evidence behind it at these shapes, so the conservative optimizer wins until
someone measures the other one. This is why `LoraHandle.groups.muon` is empty.

**Merged into the base on every export.** `agents.md` says a checkpoint is already a file
llama.cpp can load, and invariant 2 gates resume on an exact architecture match with no field for
adapter-ness. Writing adapter tensors would break both. Every caller that exports folds
`B*A*scale` in first and folds it back out after, so 310 ordinary tensors go to the file and
training continues where it was. That fold is the CALLER's job at each site, not the exporter's,
which is exactly how the deployment-quant variants shipped unmerged in the first version of this
change. Measured merge/unmerge round-trip drift: 1.5e-8 on a tiny CPU model, 3.0e-8 through the
device parity case. Only the host copy accumulates it, since nothing on the device writes a frozen
base: at most half an ulp per checkpoint on a random walk, so ~3e-6 relative after a thousand
checkpoints, against a q8_0 export error four orders of magnitude larger. Noted, not fixed.

**`B` starts at zero.** The adapted model is exactly the checkpoint at step 0, so a resume does not
jump. Verified: an adapted forward and the base forward agree to the digit before the first step.

`--lora-rank` requires `--resume`. Adapters on a frozen random init are learning against noise, and
the CLI refuses rather than letting the run discover that overnight.

**A LoRA run neither reads nor writes the optimizer sidecar.** The sidecar holds moments for the
full parameter set; a LoRA run trains 392 adapters instead, so reading one throws on the count
mismatch before step 0, which is what `pretrain --resume --lora-rank` did at first (`finetune`
escaped it only because that mode defaults to a cold optimizer). Beyond the count, a resumed LoRA
run re-initializes `A` from the seed and `B` to zero while their learned product is already folded
into the base, so those moments describe a parameterization that no longer exists. It does not write
one either: a sidecar holding adapter moments beside a merged dense GGUF would break the next full
fine-tune resuming from that checkpoint. And it REMOVES one: a LoRA run rewrites `--out` with
merged weights, so any sidecar already sitting beside that file is invalidated by the rewrite, and
its parameter count still matches the full set well enough for `importState` to accept it. The
deletion is coupled to the rename rather than done at startup, so a run that dies before its first
checkpoint leaves an existing sidecar alone.

Chaining LoRA runs is the normal merge-and-restart pattern: run two takes the merged weights as its
new frozen base and costs only a few hundred steps of AdamW re-warm. Each cycle redraws `A` from
the same hardcoded seed, so repeated cycles reuse one random subspace.

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

### 22. `writeFileSync` wrote 1 TB for a 2.39 GB checkpoint (2026-09-09)

Not a performance lever, recorded here because it is a measured runtime limit that silently caps
what this repo can export. Exporting a Qwen3-0.6B-shaped checkpoint filled a 1.9 TB disk twice: the
`.tmp` reached 1.19 TB and then 855 GB, still growing when killed.

`exportGGUF` was innocent, and instrumenting it proved that: it returned exactly 2,390,146,560
bytes. The inflation was entirely in `writeFileBytes`, a bare `fs.writeFileSync(path, data)`. On
Deno 2.9.1 that call does not survive a buffer longer than 2^31 bytes, and rather than failing it
writes without bound. Isolated:

| buffer            | result                                         |
| ----------------- | ---------------------------------------------- |
| 1.90 GiB          | file matches                                   |
| 2^31 - 1024 bytes | file matches                                   |
| 2^31 + 1024 bytes | unbounded write, process killed by `ulimit -f` |

Reads are unaffected on Deno: `readFileSync` returns a 2 GiB+ file correctly, so `--resume` was
never at risk. Under Node it throws `ERR_FS_FILE_TOO_LARGE` past 2^31 - 1 (measured at both 2.2 GB
and 4.4 GB on v26.8.1), which no shipped path reaches because the CLI is Deno-only, and which is why
the large case below skips on Node rather than reporting a failure that is not this defect. `writeFileBytes` now writes through an open handle in 1 GiB spans.

**What it was capping.** An f32 GGUF crosses 2^31 bytes at ~537M parameters, so the two largest
rows of the readme's own base-model table could not be exported at all: Qwen3-0.6B-Base (2.22 GiB)
and TinyLlama_v1.1 (4.10 GiB). Nothing had hit it because every model taken end to end here is
smaller: LittleLamb-293M exports at 1.09 GiB.

Every large writer in the tree goes through `writeFileBytes`, so one change covers all of them. The
optimizer sidecar would have hit this before the weights did: it is 1504 MB for a 293M model, so a
0.6B run's sidecar is ~3 GB. `chat-corpus` reaches it too, through `writeTokenFile`, past ~537M
tokens.

Reported upstream as [denoland/deno#36810](https://github.com/denoland/deno/issues/36810); the
native `Deno.writeFileSync` handles the same buffer correctly, so it is specific to the `node:fs`
layer. The regression test is `tests/large-file-write.ts`. Its cheap half checks the span arithmetic and
byte-exact round trips at chunk sizes small enough to cross several spans in milliseconds, which is
the part an edit is likely to break; a 5000-byte write at the production 1 GiB chunk is one span and
would exercise none of it. The real 4.10 GiB write is behind `GGUF_TRAINER_BIG_IO=1` (~4.6 GB of RAM
and disk) and runs in a child process under `ulimit -f`, because a regression there does not fail an
assertion, it runs away: bounded, it dies on SIGXFSZ and the parent reports an ordinary FAIL and
removes the temp directory the killed child could not. What
no test here can reach is the partial-write drain, since `writeSync` never returns short for a
regular file.

### 23. The CPU cross-entropy clamped every confident-wrong loss at 27.63 (2026-09-09)

Found while trying to reproduce issue #48, which turned out not to be a bug: `add(t, t)` and
`mul(t, t)` bind one buffer to two `read_write` slots, and that is explicitly legal. WebGPU's
compatible-usage-list rule grants a "usage scope storage exception": multiple `storage` usages of
one buffer in a usage scope are allowed even though they are writable. The arithmetic is defined
too, because these kernels run one invocation per element, so both writes come from the same thread
to the same address in program order. Measured at four sizes on both ops, maxdiff 0.00e+0, and
`aliasedBinaryOpParity` now pins it.

The reproduction that first seemed to confirm #48 was the harness, not the code: `backward()` seeds
only the host scalar, so a non-scalar output leaves the device gradient unseeded and both sides
compare zeros. `seedGradFromHost` is what the parity harness uses for exactly that.

The real find was elsewhere. Both CPU losses computed the row's loss by reading a
normalized probability back and adding an epsilon, `-log(p_target + 1e-12)`. Once the target falls
about 88 logits behind the row maximum, `p_target` underflows f32 to zero and the epsilon takes
over, so the reported loss saturates at `-log(1e-12) = 27.63` no matter how wrong the prediction
is. Measured on a single row before the change:

| gap from the row maximum to the target logit | reported  | exact     |
| -------------------------------------------- | --------- | --------- |
| 10                                           | 10.000136 | 10.000136 |
| 30                                           | 27.541569 | 30        |
| 60                                           | 27.631021 | 60        |
| 90                                           | 27.631021 | 90        |

Both GPU kernels already used the numerically stable form (`srcCeFwd` computes
`log(s) - (z_target - m)`, and `srcSoftCeFwd`'s comment says it is "expanded so no probability is
ever read back"), and `fusedCrossEntropy` from lever 19 was written that way too. So the two dense
CPU losses were the only ones that clamped, and the chunked path this repo added was strictly more
accurate than the dense one it replaced.

`crossEntropy` and `softCrossEntropy` now use `log(Σ exp(z - m)) + m - z_target`. Gradients were
never affected: the backward uses the normalized probabilities, where underflow to zero is the
correct limit.

**What it was hiding.** The CPU reference is the correctness oracle for the GPU kernels, so a
divergence that only shows at extreme logits is exactly the kind that survives a parity suite: the
suite's shapes produce losses of 2 to 10, nowhere near the clamp. It has a case now,
`crossEntropy (target far behind)`, and restoring the clamp fails it at gpu 76.5
against cpu 27.63. It also capped `eval-loss --cpu`
at a perplexity of `e^27.63` for a badly mismatched model or tokenizer, which reads as a plausible
number rather than a saturated one.

### 24. `eval-choice` scored the wrong span once the prompt outgrew the context (2026-09-09)

Found while reviewing #51 and filed as #52. `choiceNLL` truncated the token window to `maxSeq` but
took the mask boundary from the _untruncated_ context length, so the two disagreed exactly when
truncation happened:

| ctx | choice | maxSeq | targets | boundary | choice tokens scored |
| --: | -----: | -----: | ------: | -------: | :------------------- |
|  20 |     10 |    512 |      29 |       19 | 10 of 10             |
| 100 |     10 |    105 |     104 |       99 | **5 of 10**          |
| 600 |     10 |    512 |     511 |      599 | **0 of 10**          |

Partial truncation scored only the tail of the choice. `kept` and the count the mean is multiplied
back by still agreed, so the summed NLL was self-consistent and merely too small, while `acc_norm`
kept dividing by the whole choice's character count: a truncated option won on a discount it did
not earn.

Full truncation was worse. The boundary (599) ran past `targets.length` (511), so the mask loop
_extended_ the array rather than writing into it, nothing was kept, and `choiceNLL` returned exactly
0. A summed NLL of zero beats every real one, so that option was always the prediction.

The boundary now comes from the window the model actually sees, `min(nCtx + nChoice, maxSeq) -
nChoice - 1`, which scores the whole choice at every context length: a prompt that outgrows the
context is trimmed from the left, so the choice always survives intact and only the oldest context
goes. Solving `choiceMaskStart < 0` gives the truncation refusals exactly, and they are narrow: a single
candidate answer at least as long as the model's whole declared context, or a stem that rendered to
nothing. A choice that itself rendered to nothing is refused by its own branch instead, since it
never reaches that solution set: it has no tokens to score at all. None of the three is scoreable,
and a shortened score is not comparable to a full one. `choiceMaskStart` and
`choiceWindowError` are exported and swept in `tests/eval-tasks.ts` over every window shape the
command accepts; restoring the old boundary fails them.

**The published numbers are unaffected.** `maxSeq` is the GGUF's declared `context_length`, which
this repo's exporter sets to `max(8192, seq-len)`, and the Intelligence Index table in lever 9c was
measured 0-shot, where the longest of these four prompts is a few hundred tokens. The bug needed
either a foreign base with a short declared context or a large `--shots`: `--shots 10` against a
512-context checkpoint reaches it.

### 25. Eval copied a whole model of gradients back per window, 15% of its wall clock (2026-09-09)

Found while reviewing #51 and filed as #53. `entryFor` gives every external tensor a persistent
gradient accumulator on first use, and `sync()` stages the gradient of every touched external back
to the host afterwards. Both eval commands forward over every parameter, and neither ever calls
backward, so each scored window allocated a full model of gradient buffers, never wrote them, and
copied them to the host anyway. On a 293M f32 checkpoint that is 1.17 GB per window, and
`eval-loss` defaults to 64 of them.

The lever already existed, from LoRA: a frozen external shares one small stub instead of a
full-size accumulator (lever 21), and `sync()` skips a gradient it is not keeping. So the fix is to
freeze the parameters before the first `entryFor` call, which `freezeForScoring` does in one line
per command. Ordering is load-bearing: `entryFor` sizes the buffer on first use, so freezing after
`uploadParams` stops the copies but keeps the allocation.

Measured on `littlelamb-base.f32.gguf` (293M f32), `eval-loss --windows 16 --seq-len 512`, three
runs each on the Strix Halo APU:

|        | wall clock            | val loss |
| ------ | --------------------- | -------- |
| before | 30.75, 30.64, 30.64 s | 3.4202   |
| after  | 26.23, 26.12, 26.07 s | 3.4202   |

Median 30.64 s to 26.12 s, **14.8% faster**, with 18.8 GB of copies removed and the score unchanged
to four decimals. That is a floor rather than the whole win: it was measured with #60 still in
place, so every window still issued one redundant `clearBuffer` per frozen parameter. The device
pool also drops by the size of the model, which is what `evalFreezeGate` in `tests/gpu-parity.ts`
pins: it measures `lastSyncReadbackBytes` and `residentBytes().pool` on a frozen and an unfrozen arm
and requires the two losses to be bit-equal. Making `freezeForScoring` a no-op leaves the readback
at the full model; moving it after `uploadParams` drops the readback but not the pool, and the pool
assertion is what pins the ordering.

`generate` has the same defect and pays it per token rather than per window: 40.1% of its wall
clock on the same checkpoint, over two runs. It was #58, fixed in lever 27, where a three-run
measurement put it at 39.0%. Freezing also leaves `sync()`
re-queueing the shared stub for clearing once per frozen parameter per window, which is #60 and
applies to LoRA training as much as to eval.

Under `--cpu` this is a no-op: `Tensor`'s constructor allocates a gradient array for every tensor
regardless, so there is nothing for a freeze to skip. Both arms measured 3.1600 there, as expected.

### 26. A target outside the vocab read the next row's logits, on BOTH backends (2026-09-09)

Found while reviewing #54 and filed as #55. `crossEntropy` never checked that a target indexes a
real logit row:

```ts
total += Math.log(sum) + maxL - logits.data[b + targets[t]];
```

With `targets[t] >= V` that lands in the NEXT row's logits, so the loss comes back finite and
plausible; only on the last row does it read past the array and give NaN. `fusedCrossEntropy` fails
differently and more quietly: an out-of-range target falls inside no vocab span, so `tgtLogit` stays
at 0 and the row's loss is simply wrong. `softCrossEntropy` validated its ids already, so the two
were inconsistent and the one without a check was the one on every training path.

**The GPU was not safer, which is what the issue assumed and what this lever said first.** The claim
was that WGSL's robust buffer access clamps or discards an out-of-range read. It does not apply:
`bindGroup` passes no offset and no size, so the logits buffer is bound whole and `LOG[t * V + tgt]`
with `tgt >= V` is a perfectly in-bounds read of the next row. Measured at `T=3, V=6`, one kept row
per configuration so the reported mean IS that row's term, with the out-of-range target on the row
named:

|                                                                      | CPU               | GPU               |
| -------------------------------------------------------------------- | ----------------- | ----------------- |
| target `V` on an interior row                                        | 2.038443088531494 | 2.038443088531494 |
| the same, predicted as "row 1's logsumexp minus row 2's first logit" | 2.038443096517044 |                   |
| target `V` on the LAST row                                           | NaN               | 2.1300957         |

Identical on an interior row, to every digit. On the last row the GPU is the worse of the two: the
CPU reads past its array and announces itself with NaN, while the GPU returns a finite, plausible
number. Where that number comes from depends on the size, and neither source is stable: at `V=6` the
overrun stays inside the 256-byte bucket `BufferPool` rounds every allocation up to, and pooled
buffers come back dirty, so it is whatever the last tenant left; at a real vocab the read lands well
past the end of the buffer, where WGSL promises memory safety and some in-bounds value of its
choosing, not a particular one. Either way
the digits move with pool state, which is the argument for checking on the host rather than quoting
2.1300957 as if it were a constant.

That shows in the end-to-end symptom too. Scoring `smolrp.gguf` (vocab 49152) against a corpus
tokenized with the 151936-entry Qwen3 vocab, `eval-loss --windows 1 --seq-len 128`:

|         | before                           | after this lever                                                                          | after lever 29                           |
| ------- | -------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------- |
| `--cpu` | `val loss NaN  ppl NaN`          | `crossEntropy: target 49751 at position 19 is not -1 (ignore) or an integer in [0,49152)` | `embedding: id 49751 at position 20 ...` |
| GPU     | `val loss 10.9754  ppl 58420.99` | the same message                                                                          | the same message                         |

The third column is what you see today. Lever 29 guards the input side, which the same corpus
reaches one step earlier, so this check now fires only when the inputs are in range and a target is
not.

A perplexity of 58421 from a completely mismatched pairing is a believable-looking number, and it is
the reason the check runs on the host rather than being left to the device. Read the CPU's NaN there
as a measurement, not as this mechanism: the input stream carries the same out-of-range ids as the
target stream, so `embedding` poisons the CPU forward before the loss runs (it did then; lever 29
stops it first now), and the loss's own last-row overrun only fires when the last target happens to
be out of range.

`keptRowsInVocab` does the range check and returns the kept count, so it replaces the counting loop
each of the four losses already ran and costs no extra pass. All four call it: the CPU and GPU
`crossEntropy` and the CPU and GPU `fusedCrossEntropy`. It refuses three things, and each is pinned
by its own case in `tests/gradcheck.ts`:

- a target that is not an integer in `[0, V)`. Weakening `>= V` to `> V` fails on `target == V`
  alone, which is the boundary the whole check turns on.
- a negative other than `-1`. `uploadU32` maps `-1` to `0xffffffff`, the marker the kernels test
  for, but `-2` becomes `0xfffffffe`, a huge target the GPU scores while the CPU skips the row.
- `targets.length !== T`. The losses sum `T` rows and divide by the count the helper returns, so a
  longer array inflates the denominator. The CPU used to count only `t < T` while the GPU counted
  the whole array, which was itself a divergence; it was reachable only through the `eval-choice`
  mask bug fixed in #56.

The accepting cases carry an oracle computed in the test rather than a comparison between the two
losses, because both now draw their denominator from the same helper and would agree on a wrong
count. Making the helper count ignored rows fails `-1 still means ignore`.

Not covered here: the refusal lands mid-run rather than at start-up (#64), which matters for
`pretrain`, whose trust gate only reads the first 16 tokens. The two neighbouring gaps are closed:
`embedding` by lever 29 and the GPU `softCrossEntropy`'s teacher ids by lever 30.
`embedding`, the input-side twin, was #63 and is now lever 29.

### 27. `generate` paid lever 25's cost per token, 39% of its wall clock (2026-09-09)

Filed as #58 while fixing #53, which named the two eval commands only. `greedyComplete` syncs once
per decoded token, and `sync()` stages the gradient of every touched external back to the host, so
an unfrozen parameter was a whole model of gradients crossing the bus per token rather than per
window. Nothing in generation runs backward, so those buffers were allocated, never written, and
copied anyway.

Measured on `littlelamb-base.f32.gguf` (293M f32), 40 tokens from a four-word prompt, three runs
each on the Strix Halo APU:

|        | wall clock            | completion                                              |
| ------ | --------------------- | ------------------------------------------------------- |
| before | 29.17, 30.34, 29.81 s | `, there was a small, red, and fluffy dog named Max...` |
| after  | 18.19, 18.17, 18.28 s | identical, to the character                             |

Median 29.81 s to 18.19 s, **39.0% faster**, and 46.9 GB of gradient copies that never happen. It is
the same `freezeForScoring` call lever 25 added, in the same place, and it lands in
`src/commands/generate.ts` rather than in `greedyComplete`. That helper is shared, and `pretrain`
hands it the model it just trained on the backend it trained through; an irreversible mutation of a
caller-owned model does not belong in a forward helper, whether or not it happens to be safe for
today's callers.

`generateFreezeGate` in `tests/gpu-parity.ts` pins it the way lever 25's gate does. The frozen arm's
readback is exactly the last step's logits, `[ctx, vocab]` f32 and nothing else. That is safe to
assert exactly, rather than as a bound, because no stop token is passed, so the loop always runs to
`maxNew`; the context length carries `greedyComplete`'s own `maxSeq` clamp rather than assuming it
does not bind. Making `freezeForScoring` a no-op leaves the readback at the full model;
moving the call after `uploadParams` drops the readback but not the pool.

The arm that compares the generated ids is a **canary, not a guard**. No read of `requiresGrad`
feeds an output value: each one is `entryFor`'s buffer choice, `sync()`'s staging decision, or a
gate on a dW accumulation, and that last kind is read in the closure on the GPU path but captured at
forward time on the CPU one (`const wantsDW = w.requiresGrad`). None of them writes to `out.data`.
So no regression in the freeze can move the text, and that assertion cannot be mutation-proved. It is there to catch a future forward-path read of the flag,
which is a different thing from evidence that this change is safe.

`generate` was the last workload paying this per token. `pretrain` samples once when training
is over, two prompts at 60 tokens through the same `greedyComplete`, and it looked like the same
bug: that was #66, and it is not one. Every GPU optimizer calls `keepGradOnDevice` on the parameters
it owns while building its state (`muon-gpu.ts`, `adamw-gpu.ts`), and `paramGroups()` covers every
parameter, so by the time the sample runs each one is already exempt from staging. Measured on the
real sequence, a resident training step with `MuonGpu` followed by a sample: the last sync reads back
the logits and nothing else (1280 bytes on the gate's micro model), with or without a
`keepGradOnDevice` call of our own. A LoRA run
reaches the same place by both routes: its base weights are frozen, and its adapters go through the
aux group like any other trainable tensor. The narrow version of #66 is nil too: freezing never
frees an accumulator, and these already exist from training, so there was nothing to reclaim on
either half.

`pretrain` did still stage a whole model of gradients in one place, its trust gate, which forwards
before the optimizer exists to keep anything on device. A LoRA run staged only its adapters there,
since `applyLora`'s freeze is deliberately placed before the gate. Once per run rather than per
token either way, which is why this paragraph read as a note rather than a finding. Lever 37 closed
it anyway.

### 28. The clear queue re-armed itself for frozen parameters, and it buys no time (2026-09-09)

Filed as #60 while reviewing lever 25. `sync()` set `e.gradNeedsClear = true` for every touched
external without asking whether that external has a real gradient accumulator. A frozen one does
not: `entryFor` hands it the shared 256-byte stub precisely because nothing ever writes to it. So
from the second window onward, every frozen parameter re-queued that one stub for a `clearBuffer`,
and each of those cleared the same 256 bytes to no purpose. Two callers hit it: eval since lever 25,
where every parameter is frozen, and LoRA training, where the base weights are frozen for the whole
run. On a 310-tensor checkpoint at `--windows 16` that is 4650 no-op commands: the first window
does not queue them, because `entryFor` starts `gradNeedsClear` at `requiresGrad`.

**Lever 32 has since subsumed the eval half of this.** A forward-only window issues no clears at
all now, frozen or not, so those 4650 are gone whether or not this landed. What remains here is the
LoRA and finetune half, where the window does run a backward and a frozen base weight's stub would
still be re-armed, and that is the configuration `frozenClearGate` measures.

`e.gradNeedsClear = e.grad !== this.frozenStub` is the whole fix, and which side of that predicate
it sits on is the interesting part. `t.requiresGrad` is the obvious spelling and it is wrong in one
ordering: freeze a parameter mid-window, after that window's `entryFor` and before its `sync()`, and
the flag records "no clear needed" while a full-size accumulator still holds that window's
gradients. A later thaw then accumulates on top of them. Measured at exactly 2x, against 1.0000 on
`main`, so that spelling would have been a real regression rather than a theoretical one. Keying on
the buffer cannot go stale in any ordering and drops exactly the same clears.

**It is worth no measurable time, and saying so is the point of this entry.** `eval-loss --windows 16
--seq-len 512` on `littlelamb-base.f32.gguf`, three runs each:

|        | median  | spread |
| ------ | ------- | ------ |
| before | 26.69 s | 0.32   |
| after  | 26.77 s | 0.07   |

That is +0.3%, well inside the run-to-run spread, and the val loss is 3.4202 either way. A
256-byte `clearBuffer` really is nearly free; what was wrong was the bookkeeping, not the clock.
It is still worth having, though the second reason this entry first gave was backwards. #66 does not
depend on it: those parameters already carry full accumulators from training, so `e.grad !==
frozenStub` is true for every one and their clears are still re-armed. #66 is the case this does
NOT cover, and it is not blocked by it either, since its copy saving comes from the `requiresGrad`
guard in the staging branch rather than from here. What remains is reason enough: `sync()` stated
something untrue about frozen externals, and the eval and LoRA no-ops leave the command stream at no
measurable cost.

**It costs nothing, which took two wrong readings to establish.** I first wrote that a parameter
frozen after it had a full-size accumulator would keep stale gradients forever. It does not: the
last sync before the freeze already armed that clear while the parameter was trainable, and the
armed clear still fires. A probe over three orderings on both branches settled it, and it also found
the one ordering that does break, which is the mid-window freeze above. `clearRearmPredicateGate`
pins that, because it is the only check that can tell the two candidate predicates apart: the
counting gate passes for both.

While measuring it, `freezeForScoring`'s docstring turned out to overstate its own hazard in a
different way. "A device validation error rather than a wrong number" is true only for a tensor
wider than 64 floats; the stub is 256 bytes, so a narrower one stages out of it without complaint.

Waste has no symptom in a number, so `frozenClearGate` in `tests/gpu-parity.ts` counts instead:
`gradClearsIssued` is a cumulative counter and the gate takes a delta around the second window,
which is the one that matters, since `entryFor` starts `gradNeedsClear` at `requiresGrad` and a
frozen parameter is not queued on the first. The frozen arm issues exactly `params` fewer clears
than the unfrozen one, and the loss is bit-equal.

The count does not go to zero, and the residual is the more interesting half: `makeOut` queues every
intermediate's gradient buffer unconditionally, 85 per window against 54 parameters on the toy model
in the gate, and those are full-size rather than a shared stub. Nothing reads them in a forward-only
run either. That is #67, and it measured as noise too (26.53 s against 26.69 s), so it is filed with
the measurement attached and no speed claim.

### 29. An embedding id outside the table read the next row, or returned zeros (2026-09-09)

Filed as #63 while fixing #55, which guarded the loss side only. `embedding` never checked that an
id indexes a row of the table, so `weight.data[id * d + j]` with `id >= V` read into the next row,
or past the array on the last one. The two backends then failed differently, which is the divergence
the parity suite exists to prevent. Measured at `V=4, d=3` with an id of `V + 2`:

|     | row returned for the bad id |
| --- | --------------------------- |
| CPU | `[NaN, NaN, NaN]`           |
| GPU | `[0, 0, 0]`                 |

Neither stops. The CPU poisons the forward with NaN, which at least announces itself; the GPU's
bound buffer discards the read and substitutes a zero row, so the run continues on a number that
looks fine.

`assertIdsInTable` refuses anything that is not an integer in `[0, V)`. There is no ignore marker
here, unlike a loss target: every position of a batch is a real token, so a negative is refused too.

It sits ABOVE the backend dispatch in `embedding` rather than in each backend, which is this file's
existing convention for a guard both paths need and neither can fold into work it already does
(`fusedCrossEntropy`'s dimension and chunk checks are there for the same reason). That placement is
the thing worth pinning, and it is: moving the call below the dispatch still passes the CPU cases in
`tests/gradcheck.ts` and fails `targetRangeGate`, because an installed backend then skips it.

**It fires before lever 26's check, which is what the ordering predicts.** The inputs go through the
table before the targets reach the loss. Scoring `smolrp.gguf` (vocab 49152) against a corpus
tokenized with the 151936-entry Qwen3 vocab now stops at `embedding: id 49751 at position 20`, where
before this it ran on to `crossEntropy: target 49751 at position 19`. Same token, one position
apart, because the targets are the inputs shifted by one.

### 30. The GPU never checked its teacher ids, and both backends carried the same shape guards (2026-09-09)

Filed as #61 while fixing #55. `softCrossEntropy` validated each teacher id against the vocab on the
CPU and not at all on the GPU, so an id built against a different vocab reached the kernel and
indexed whatever the bound buffer held. Same class as levers 26 and 29, and the same corpus mistake
reaches it: the KL anchor's teacher file is built once over the SFT corpus, so a teacher file and a
checkpoint can disagree exactly the way a `.tokens` file and a checkpoint can.

The fix is a placement, not a fourth call. `assertTeacherRows` runs ABOVE the backend dispatch, the
way lever 29's guard does and the way `fusedCrossEntropy`'s dimension, chunk and LoRA guards already
did. Validating below the dispatch is what created this gap in the first place: every implementation
then needs its own call, and one of them will be the one nobody remembers.

Hoisting paid for itself twice over, because the `k >= 1` and `[T*k]` length guards were **duplicated
verbatim** in the two backends. Both copies collapsed into the one call site, along with the CPU's
per-element id check inside its inner loop.

Three things the validator has to get right, each with its own case in `tests/gradcheck.ts`:

- the ignore marker is exactly `-1` in a row's first slot, not any negative. Loosening it to `< 0`
  skips the row unchecked, and `uploadU32` turns `-2` into a huge id the GPU scores while the CPU
  drops the row, which is precisely the input this guard exists to catch. Same rule, same reason, as
  `keptRowsInVocab`.
- an ignored row's remaining ids are never read, so junk there must be accepted.
- inside a kept row all `k` ids are checked, including the slots a short row pads at probability 0.
  Only the FORWARDS skip a pad, both on `q == 0`. Both backwards index by its id unconditionally,
  and the GPU's is a non-atomic read-modify-write (`DLOG[i] = DLOG[i] - scale * TQ[...]`), so an
  out-of-range pad id lands in another row and can lose that row's real update. That is the race
  the one-thread-per-row design exists to prevent, which makes this check load-bearing rather than
  contract-keeping. The consequence for whoever writes a teacher file: pad short rows with an
  in-range id, never with `-1`.

One instance of the shape this lever argues against is still in the tree, pre-existing and left
alone: `crossEntropy` dispatches above its own `keptRowsInVocab` call, so that guard lives in each
backend. All four call sites are present today, so there is no live gap, but it is the next place
one could open. Its rank check is above the dispatch, though: lever 31.

Nine mutations, each applied alone. Removing the id check fails eight cases and the GPU arm. Moving
the call below the dispatch passes every CPU case and fails only the GPU arm, which is the whole
point of that arm. Loosening the marker to `< 0`, skipping zero-probability slots, dropping the
probability-length clause, checking only the first slot, only the first row, checking ignored rows,
and dropping the `k` guard each fail their own cases and nothing else.

### 31. A guard that switched itself off on the input it was there to catch (2026-09-09)

Filed as #71 while reviewing lever 30. All three losses read `const [T, V] = logits.shape` and then
compare every id against `V`. Hand one a 1-D tensor and `V` is `undefined`, so `id >= V` is false for
every id: `keptRowsInVocab` and `assertTeacherRows` accept everything, and the loop then indexes past
the buffer. The range guards levers 26, 29 and 30 added turn themselves off on exactly the malformed
input they exist to catch.

`assertMatrix` refuses a `logits` that is not 2-D, a `hidden` or `w` in `fusedCrossEntropy` the same,
and `embedding`'s table, which reads a vocab size out of `shape[0]`. It runs above the backend
dispatch and before the id checks, in that order, because a guard reporting on a shape it has already
destructured is reporting nonsense.

Pinning the rank at exactly 2 is what makes the vocab size trustworthy again, and the reason is the
`Tensor` constructor: it throws when `data.length` does not equal the product of the shape, and
`shape` is never reassigned afterwards. So a 2-D shape guarantees `shape[1]` is consistent with the
buffer behind it, and there is no residual case of a `[3, 6]` holding four floats.

Unreachable today: every producer builds logits from `linear(hidden, w)` or `forwardToReadout` with a
2-D weight, and a GGUF's own dims never become a `Tensor.shape`. It is a hole in a trust boundary
rather than a live bug, which is why it is worth naming: the failure mode is silent acceptance.

**Measured with the guard disabled, five of the eight refusals were silent and three already threw
something unhelpful.** The test block records both lists, because the deliverable differs:

| input                               | before                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------- |
| `crossEntropy` on `[24]`            | NaN                                                                          |
| `crossEntropy` on `[2,3,4]`         | 0.8006, scored as `T=2, V=3` out of a 24-float buffer                        |
| `softCrossEntropy` on `[24]`        | NaN                                                                          |
| `fusedCrossEntropy` on a 1-D pair   | 3.1781, i.e. `log(24)`, since `undefined !== undefined` passes the dim check |
| `embedding` on `[2,3,4]`            | a row read with V from `shape[0]` and the stride from `shape[1]`             |
| `fusedCrossEntropy` on a 1-D hidden | `dim mismatch undefined vs 4`                                                |
| `fusedCrossEntropy` on a 1-D w      | `dim mismatch 4 vs undefined`                                                |
| `embedding` on `[24]`               | `data length 0 != shape 2,`                                                  |

Ten cases in `tests/gradcheck.ts` and an arm on `targetRangeGate`. Dropping any one of the five calls
fails that op's cases, and for `crossEntropy` the GPU arm as well, since that is the only rank guard
the gate exercises; loosening the rank test to `>= 1` fails all eight refusals, since the
point is the exact rank and not merely "has a shape"; and moving the `crossEntropy` call below the
dispatch passes every CPU case and fails only the GPU arm, which is by now the recognisable signature
of that mistake.

### 32. A forward-only window asked the device to zero buffers nobody reads (2026-09-09)

Filed as #67 while fixing #60, and filed with its own null measurement so nobody chases it as a
speed-up. `makeOut` queues every intermediate's gradient buffer for a `clearBuffer` at creation,
because a backward accumulates into it with `+=` and a pooled buffer arrives dirty. Its comment says
"before this graph's backward pass runs". Eval, `generate` and `pretrain`'s trust gate never run
one, so every one of those clears was zeroing a buffer nobody would read. On the toy model in
`forwardOnlyClearGate` a forward-only window issues 139 clears, 85 intermediates and 54 parameter
accumulators, and all 139 go: the 54 are the ones lever 28 attacked in the frozen case, and unlike
those the intermediates are full-size buffers rather than a shared 256-byte stub.

The 54 carry a semantic change worth stating, because it is the one thing here that is not pure
waste removal. Their clear ran BEFORE the same sync's gradient staging, so a forward-only window used
to leave zeros in the host `grad` arrays and now leaves whatever the last backward put there. No
caller reads them, eval and `generate` freezing and training keeping gradients on device, and the
gate pins it so that going back to zeros means someone reintroduced the accumulator clears.

`sync()` drained the queue whether or not a backward had begun, which is what forced them. It now
asks. **Dropping rather than deferring is what the recycling forces:** those buffers return to the
pool at the end of the same `sync()`, so a clear held over would land on whatever reacquires them.
The invariant that makes dropping safe is the one recycling already needs, that no backward may
begin for a graph built before the sync, and `ensureBackwardBegun` drains the queue itself when one
does.

**It buys no measurable time, the second such entry in a row.** `eval-loss --windows 16 --seq-len
512` on `littlelamb-base.f32.gguf` measures 26.62 s median against `main`'s 26.69 s, three runs each,
inside the run-to-run spread, and a one-step `pretrain` on a 306M model moved 36.58 s against 37.25 s
when probed. Both are the size of the noise. What the change buys is a command stream that says what it means.

`forwardOnlyClearGate` counts rather than times: a window with a backward still issues 139 clears,
one without issues 0, and the two losses are bit-equal. Removing the guard or inverting it makes the
forward-only arm issue all 139.

**It also invalidated a gate written four levers ago, and half of that lever's justification.**
`frozenClearGate` compared clear counts across two forward-only windows, which now issue none at
all, so it went red on a correct change. It runs a backward now, since a parameter's clear is only
observable in a window that has one. Lever 28's headline saving, 4650 no-op commands per eval run,
is likewise gone whether or not lever 28 had landed: what it still buys is the LoRA and finetune
case, and that entry now says so. Worth noting as the failure mode of counting gates and of the
levers that quote them: both pin a number a later, unrelated improvement is entitled to move.

The drain in `sync()` is a safety net rather than a live path, measured by deleting it: every clear
today is issued by `ensureBackwardBegun`, and no count moves. What it covers is a clear queued after
the backward began, an external first materialized mid-backward. Not a second backward over one
graph, which queues nothing at all: `ensureBackwardBegun` early-returns and `makeOut` is not called
again.

**The price is an invariant that used to be forgiving.** Dropping a clear is safe only while no
backward runs over a graph built before a sync. That ordering was previously wasteful but survivable,
since the buffers went back to the pool zeroed; now they go back dirty, so the same mistake would
accumulate into pool garbage and report a believable number. `ensureBackwardBegun` throws on it, and
the gate has an arm that builds a graph, syncs it, asks for a backward and expects the refusal.

The throw is armed only by a dropped `makeOut` clear, not by any dropped clear. `entryFor`'s
accumulators are persistent, never return to the pool, and are re-armed by the same `sync()`, so
dropping theirs is free: an optimizer constructor queues one per parameter, and flagging the whole
queue refused `new MuonGpu(...)` followed by a sync and a `seedGradFromHost` with no graph anywhere
in the flow. The gate has an arm for that too, and for the re-arm itself, which `pretrain` depends
on: its trust gate is a forward-only sync that drops every parameter's clear, and the training loop
then accumulates into those same accumulators.

The reset in `beginForwardOp` bounds what the throw catches to a backward with no forward op in
between. Forward A, sync, forward B, backward A still slips through, as it did before this change:
the flag narrows the window rather than closing it. Under `--recompute` that ordering is routine
rather than exotic, since a checkpoint replay's own forward is that forward B, and it still throws on
a real model only because the loss and readout backwards run before any checkpoint block.

### 33. The GGUF tensor boundary checked shapes in neither direction (2026-09-09)

Filed as #73 and #74 while reviewing lever 31, and fixed together because they are one boundary seen
from its two sides. Neither is reachable today, which is why both are worth naming: the failure mode
on each side is a plausible artifact rather than an error.

**Writing.** `addMatrix` destructures `const [outDim, inDim] = t.shape`. A 1-D tensor leaves `inDim`
undefined, so `inDim % 32 !== 0` is `NaN !== 0` and every such tensor silently becomes f16 whatever
the requested quant, and the ggml `ne` goes out as `[undefined, outDim]`. That is a corrupt file
produced without a word. `addVector` had the mirror hole.

**Reading.** `tensorLoader` never looked at `t.dims` at all:

```ts
const de = dequantize(t.type, t.data, dst.size);
dst.data.set(de);
```

The destination shape comes from `arch.build(cfg, ...)`, allocated from the metadata scalars before
any bytes are read, so the file's own dims were pure decoration. Measured, with the direction the
other way round from what this lever said first:

| the destination against the file's tensor | before                                                                           |
| ----------------------------------------- | -------------------------------------------------------------------------------- |
| same element count, transposed            | a scrambled weight and a model that generates noise                              |
| smaller                                   | a silent prefix load: a `[3,4]` tensor into a `[4]` destination gives 1, 2, 3, 4 |
| larger                                    | `RangeError: Offset is outside the bounds of the DataView`, no tensor named      |

`dequantize` always returns exactly `count` floats and a `Tensor`'s buffer always matches its shape,
so the "fills a prefix and leaves the rest" case cannot happen; the prefix is taken from the FILE,
not left in the destination. The only guard anywhere was the `token_embd.weight` round-trip check in
`demo.ts`, one tensor in one command.

The loader now compares `t.dims` against the destination shape reversed, ggml writing `ne`
fastest-moving first, which is exactly what `addMatrix` does when it sends `[outDim, inDim]` out as
`[inDim, outDim]`.

The rank half of that comparison is not decoration either. `[4]` reversed is a PREFIX of `[4, 3]`,
so an element-wise check alone accepts a 1-D destination against a 2-D tensor and `dequantize`
returns the first four values quite happily. That case is what the length clause is for, and it is
the one mutation that survived the first version of the test.

Two more things landed with it, both the same shape one axis over.

`dequantize` now checks that the buffer holds the bytes the type and count require, and that a block
type gets a whole number of blocks. The `RangeError` above is the polite failure; **q4_0 has a silent
one**. Its nibble read is a plain array index, and in JS `undefined & 0x0f` is 0, so every byte past
the end of a truncated block decodes as `(0 - 8) * scale`: finite, plausible, no NaN. Measured on a
block with half its nibbles missing, the last eight values came back as -8.

And the dims comparison trims trailing 1s from both sides. GGUF itself stores an explicit `n_dims`
and exactly that many values, so a file is never padded; llama.cpp pads `ne` to four with 1s **on
read**, which is why `[n]` and `[n, 1]` are the same tensor to it and why a writer emitting the
redundant trailing 1 produces a file it accepts and an exact comparison here would refuse. This repo
reads foreign GGUFs on purpose, BF16 existing only as an import path.

Trimming cannot admit anything it should refuse, because it preserves the product: if the two
trimmed lists are equal then the element counts are equal. The `> 1` floor has no distinguishing
input either, and that is a property rather than a gap. The two floors differ only for an all-1s
list, `[]` against `[1]`, and a list of positive integers is all-1s exactly when its product is 1; so
if one side is all-1s the other either is too, in which case both floors collapse both sides
identically, or has a different product, in which case both refuse.

**This brings the repo into line with llama.cpp rather than away from it,** which is the opposite of
what this entry said first. `check_tensor_dims` in `llama-model-loader.cpp` throws
`tensor '%s' has wrong shape` on exactly this mismatch, and it compares over `GGML_MAX_DIMS`
requiring `cur->ne[i] == 1` past the expected shape's length, which is the trailing-1 rule
implemented here. So the outlier was this repo, in not checking at all. I had written that llama.cpp
tolerates a padded embedding because it sizes the model from the tensor; that was recall, the local
checkout says otherwise, and the paragraph is gone.

What a foreign file is likely to trip on instead is a defaulted metadata key. `attention.key_length`
falls back to `embedding_length / head_count` and `vocab_size` to the token list's length, and a
wrong default changes the shape of every `attn_q` and `attn_k`. Turning that from a scrambled weight
into an error is the guard's best outcome, so the message names those two keys. All eight local
checkpoints across the three architectures load unchanged under it.

`assertRank` is `assertMatrix` from lever 31 with the rank as an argument, since the writer needs 2
for a matrix and 1 for a vector. Eleven mutations, ten in `tests/export-extras.ts` and one in
`tests/gradcheck.ts`, one per clause on each side. One survives and is left alone: loosening the
trim's floor from `> 1` to `> 0`, for the reason above. One is worth knowing about because it looks
like a tightening: `bytes.length < need` cannot become `!==`, since the writer pads every tensor up
to the file's alignment and the reader slices each one to the next tensor's offset, so `t.data`
legitimately carries that padding.

### 34. The corpus/vocab mismatch is caught when the file opens, not on the window that hits it (2026-09-09)

Filed as #64 while fixing #55. Levers 26, 29 and 30 made an out-of-range id an explicit error
instead of a silently wrong loss, which is the right outcome, but they raise it on whichever window
happens to contain the id. For eval that is fine, a run being minutes. For `pretrain` it is not: the
trust gate reads only the first 16 tokens, so a `.tokens` file built with the wrong tokenizer passes
it and the run can be tens of thousands of steps in before some later window holds a high id.
Everything written up to that point trained on whatever the guards were catching.

`assertCorpusFitsVocab` walks the source once when it opens, and `pretrain` calls it before the trust
gate rather than after.

`eval-loss` scans only `[lo, length)`, the region it actually scores, which at the default
`--holdout 0.01` is a hundredth of the file. That is not a micro-optimization: this command's own
header describes a watch loop re-running it every ten minutes against a live run's corpus, and a full
sequential pass over a FineWeb-scale file each time would evict more page cache than it warms, while
the eval itself touches 65,536 tokens.

**It costs almost nothing.** Measured at **310M tokens/s** on this machine, so the 17.4M-token
`lambrp.tokens` scans in 0.06 s and a FineWeb-scale 10B-token corpus would cost about 32 seconds,
once, against a run of hours. End to end on `eval-loss --windows 4` over that corpus the difference
is inside the noise: 17.90 and 17.93 s against `main`'s 17.89 and 17.99 s, and that is with the scan
narrowed to the scored region.

The mismatched pair from lever 26 now stops in 5.9 s instead of after loading a model and running a
forward:

```
data/lambrp-hold.tokens: token 52897 at position 36 is outside [0,49152). The corpus was
tokenized with a different vocab than the checkpoint; retokenize it with the checkpoint's
own tokenizer.
```

It catches a width mismatch in one direction, for free: a 2-byte file read as 4-byte yields ids in
the hundreds of millions, and one of this repo's own corpora reports `token 205291510 at position 0`
when read against the wrong vocab. **Not the other direction.** A 4-byte file read as 2-byte passes
the size check, which is only `% 2`, and every id becomes a half-word, so the count silently doubles
and every second one reads as 0. Nothing here catches that, and it belongs with the stale-`.tokens`
class below rather than with what this closes.

`memTokenSource.window` also gained the bounds check `diskTokenSource` always had. Without it a
window past the end returned `undefined` per token, which the losses refuse as "not an integer" by an
odd route.

The scan itself is `chunkSpans` from `src/io.ts`, which is `writeSpans` renamed and generalized: it
already computed exactly this loop and already refused a bad chunk, and reusing it deletes a second
copy of both. The rename also tightened it to a positive **integer**, since a fractional chunk
terminates but hands the caller a fractional length. The circularity that removed is worth naming:
the guard I had written existed only to protect the parameter I had just added, and the version in
`io.ts` has its own test.

Nine mutations in `tests/large-vocab.ts`, one in `tests/large-file-write.ts`. Two are worth naming:
the scan stopping after its first chunk is caught only because the chunk size is a parameter, which
is why it is one; and dropping `from` from the reported position is caught because the `from` case
asserts an absolute position, which is what a user needs to seek to.

**Still open, and not closed by a range check.** In the `.txt` branch `pretrain` reuses an existing
`${stem}.tokens` without rewriting it, while `sharedTokenizer` retrains the vocab whenever
`${stem}.tokenizer.json` is missing. Delete that json and you train on a stale token file built from
a vocab that no longer exists; this catches it only if the stale ids exceed the new vocab, so a
same-size or larger vocab passes, and so does any narrower vocab that flips the file's id width, per
the half-word case above. Closing the class needs tokenizer identity, a hash beside the
`.tokens`, not a range check.

### 35. `eval-choice` refuses an unscoreable item before the first forward (2026-09-09)

Filed as #57 while fixing #52. `choiceNLL` refuses a window it cannot score, which is right, but it
does it on the item that holds one: after the GGUF is loaded and after however many items sit ahead
of it. On a full HellaSwag set that is hours of forwards before the command exits with a usage error.
The new pass runs after the GPU comes up too, which is a few seconds on a run that was going to fail;
moving it earlier would put the dataset's network fetch in front of the backend install for no
proportionate gain.

The check does not need a tokenizer, and that is what makes it free rather than cheap. This repo's
BPE is byte-level, so **every token covers at least one UTF-8 byte**: a rendered choice under
`maxSeq` bytes cannot reach `maxSeq` tokens, and a non-empty stem encodes to at least one. The pass
walks strings the scoring loop was going to render anyway, and only a choice whose byte count reaches the model's
whole declared context gets encoded. On the shipped tasks that is none.

**Bytes, and not characters, and I shipped characters first.** `String.length` counts UTF-16 units
and does not bound the token count at all: a byte-level BPE falls back to one token per byte for
anything its merges do not cover, so a single three-byte BMP character can be three tokens, which is
the worst case by construction rather than a figure anyone measured. Measured against the 151936-entry Qwen3 vocab in
`data/lambrp-hold.tokenizer.json`, `⸻` is one character and two tokens and `ᚠᚢᚦ` is three characters
and six. The character version was unsound in the dangerous direction, quietly passing an item it was
meant to catch, and both ARC and HellaSwag carry non-ASCII text.

The byte bound holds for every vocab by construction, not by an argument about any one of them:
`GPT2_SPLIT` partitions the text without overlap, each pre-token starts at one symbol per byte,
`bpeWord` only ever shortens, and an unknown id only drops.

**And a broken premise would cost earliness, never correctness**, which is the better argument for
merging this than the bound itself. The pass refuses only an empty string, zero tokens under every
vocab, so anything it rejects `choiceNLL` would reject too; and every way the bound could fail makes
it pass a pair the scorer still checks with real token counts. That retires the "but a user's vocab
could differ" objection rather than answering it case by case.

Encoding every pair up front instead, which is what #57 asked for, costs 6.7 s on a full set: 40168
pairs at 0.17 ms each, measured. That is worth recording but it is not why this design won. 6.7 s
against hours of forwards is a rounding error, and the honest reason is that the length pass is
already free and exact, so paying anything for the same answer would be the worse trade.

The bound is tight at its boundary, which is where the `>=` earns its place: a choice of exactly
`maxSeq` bytes could encode to `maxSeq` tokens, one byte fewer could not. Weakening it to `>` fails
the case that pins it, and swapping bytes back for characters fails a different one.

`preflightByBytes` returns the pairs that still need encoding rather than doing the encoding itself,
so the arithmetic is testable without a tokenizer or a model, which is the same split
`choiceMaskStart` and `choiceWindowError` already use.

The pass runs per item rather than over the whole grid. At 10-shot the preamble repeats in every
pair's stem and `choiceText` is a `slice` that retains its parent string, so materializing all 40168
at once would hold, by estimate rather than measurement, a few hundred MB the streaming loop below
never does.

`renderPair` is the other half, and it is the one that keeps the pass honest. The two lines that
define where the stem ends and the choice begins used to exist twice, in the preflight and in the
scoring loop. If they drifted, the preflight would print its tick for strings that are not the ones
being scored, which is worse than not having the pass at all. Both sites call the same helper now,
and a test reassembles its two halves into exactly what the model sees.

The premise itself has a test, which matters more than the arithmetic does: the byte bound rests on
`encode` never emitting more tokens than the input has UTF-8 bytes, and the cases above take that
from a docblock. One check encodes an ASCII, an accented, an untrained multi-byte, a CJK and an
astral string against a real trained tokenizer and requires `tokens <= bytes` for each. Making
`encode` prepend a BOS, an ordinary thing for a tokenizer to grow, fails it at `café`, 6 tokens
against 5 bytes.

It also requires the other direction, `tokens >= 1` for a non-empty string, and that is the half
that genuinely varies by vocab: `encodeOrdinary` drops a symbol whose id is missing, and its comment
that "every byte is in the base vocab" is true of `train()` and not of `fromData()` on someone
else's GGUF, which is the path `eval-choice` takes. A vocab missing its byte tokens fails it at
`café`. Without that assertion the empty check, which is a string-emptiness proxy for zero tokens,
would quietly stop meaning what it says.

`withPreamble` joins the few-shot preamble to a stem, and is hoisted for the same reason
`renderPair` is: both loops have to agree about the separator.

### 36. Why there is no CPU training, measured (2026-09-10)

Asked directly after #41, and worth writing down because the honest answer is not the one the code
suggests. `pretrain` refuses to start without a GPU adapter, and the natural reading of that is that
a CPU training path does not exist. It does. `trainLM` in `src/train/trainer.ts` is a complete loop:
window sampling, backward, optimizer step, LR schedule, QK-clip, supervision masks, disk-backed
token sources. `demo` trains with it, and `gpu-parity` runs it step-for-step against the GPU trainer
to prove their trajectories match.

**Wiring it into `pretrain` is not the modest change a `gpu.` grep suggests**, and this entry said
it was until review counted properly. The grep returns 13 lines, but four are imports: nine calls
over seven methods, of which only `describeDevice` and `residentBytes` are trivially replaceable.
What the grep cannot see is the rest:

- the two optimizers are not interchangeable. `MuonOpts` and `MuonGpuOpts` are structurally
  identical, so the constructor ports, but `MuonGpu` has `recordStep()` where `Muon` has `step()`,
  so it does not satisfy `Optimizer` and `trainLM` cannot take it, while `trainLMGpuResident` types
  its parameter as `MuonGpu` so `Muon` cannot go the other way.
- `syncWeightsToHost()`, `exportState()` and `importState()` exist only on the GPU optimizers, so
  the `--resume` optimizer sidecar, its size in the checkpoint log, and the LoRA stale-sidecar
  removal have no CPU path at all.
- the `--recompute` guard tests `gpu.regionCount()`, and there is no backend-agnostic count to
  substitute, so that is a new API rather than a swap. `checkpoint()` itself is genuinely
  backend-agnostic, degrading to plain recompute, so the flag would work; the guard around it would
  throw.
- the trust gate would compare the CPU against itself, `cpuLoss` and the probe both coming from the
  same `model.forward`, so the difference is zero by construction and it prints a green line for a
  comparison it never made. That is the exact failure its own comment warns about for
  `--loss-chunk`.

None of that is a big project. It is more than a banner and a memory line, and the argument below
reads stronger for conceding it.

**The reason is throughput.** Same machine, same shape, same steps, CPU `trainLM` against GPU
`trainLMGpuResident`:

| shape                                                            |       CPU |       GPU |
| ---------------------------------------------------------------- | --------: | --------: |
| 6.0M params, vocab 8192, hidden 256, 4 layers, seq 256, 2 steps  |  16 tok/s | 786 tok/s |
| 32.0M params, vocab 16384, hidden 512, 6 layers, seq 256, 1 step | 1.6 tok/s |           |

49x at 6M, and CPU throughput falls faster than the parameter count rises: 5.3x the parameters cost
10x the time over that range. That pair says nothing about the GPU side, which is measured
separately below. What widens the gap is how much better the GPU holds up: it loses less than an
order of magnitude between the 6M and 596M shapes, while the CPU loses a full one between 6M and
32M. Two short timings at shapes differing in four variables are not a curve, so what follows is an
order-of-magnitude argument and nothing finer. Carrying the 32M figure LINEARLY to 596M, generous
against a trend worse than linear, puts a 596M CPU step in the range of seconds per token: a
100k-token fine-tune runs into weeks, and 10M tokens into years.

For the GPU at that size no extrapolation is needed, since it was measured directly:

```sh
deno run -A cli.ts pretrain --data data/lambrp.tokens --out /tmp/tps.gguf --steps 2 --batch 1 \
  --seq-len 2048 --arch qwen3 --hidden 1024 --layers 28 --heads 16 --head-dim 128 \
  --recompute --loss-chunk 8192
# Training: ... 108 tok/s, peak 15839MB gpu (pool 5903 + state 9936)
```

108 and 109 tok/s over two runs, so 10M tokens is about a day. The point is the ratio of orders of
magnitude, not the third digit on either side.

Threads do not rescue it either. The loop is scalar single-threaded JS with no worker pool and no
`--threads` flag, and even a perfect 10x from ten cores leaves that 596M estimate months short of
useful.

So shipping `--cpu` for training would be a small change that produces a trap: a flag that accepts
the run and then never finishes. Making CPU training genuinely useful is the several-new-files
project, a threaded SIMD or WASM/BLAS backend, which is a different undertaking from exposing the
reference loop. The loop's job is to be the oracle every GPU kernel is checked against, and it is
good at that.

**For anyone who has only a CPU**, the recommendation is `transformers` with `peft`, measured on
Qwen3-0.6B-Base at seq 512, batch 1, 10 threads: LoRA 118 tok/s at 6.1 GB peak, a full fine-tune
82 tok/s at 12.8 GB. Both fit 32 GB. The readme carries the same note and the install lines.

**The error message was the place a user would learn this, and it said the wrong thing.**
`initWebGPU` returns null for two reasons, no WebGPU in the runtime and no adapter on the machine,
and the die collapsed both into "training needs Deno". Someone running Deno on a GPU-less box was
told to use Deno, which is exactly #41's situation. `webgpuRuntime()` tells them apart, and
`initWebGPU` does not consult it: the `try`/`catch` around its adapter request already covers every
runtime without WebGPU, and a first-line guard duplicating the predicate proved unobservable across
six shapes of broken navigator. The error message was #84.

### 37. The trust gate staged a whole model of gradients for a backward it never runs (2026-09-10)

Filed as #80 while closing lever 27, which named this case and let it stand. `pretrain`'s parity
probe forwards, syncs and stops. It runs before the optimizer is built, so nothing had called
`keepGradOnDevice` yet, and `sync()` stages the gradient of every touched external. Nothing reads
them: no code in that file reads a host-side gradient, and both GPU optimizers do their clipping
and their step on device. Nor is the exemption wider than the one the optimizer is about to make.
Dense, it is `model.params()`, and both GPU optimizers keep everything in `paramGroups()`, which
`tests/arch-roundtrip.ts` asserts is the same set for every registered arch, tied head or not. Under
LoRA, `applyLora` freezes the base and hands the optimizer the adapters, which is what this keeps.

Measured on a Qwen3-0.6B shape (596M params, `--seq-len 512 --recompute --loss-chunk 8192`, one
step, `--reclaim`), two runs each:

|              | wall clock       |
| ------------ | ---------------- |
| as it stood  | 73.55 s, 74.23 s |
| with the fix | 71.72 s, 73.11 s |

Worth a second or so: 1.48 s between the means, against 1.39 s of spread within the after arm
itself, which with two runs a side is not a number to lean on. The finding is the 2,384,199,688 bytes of staging that never happens, printed by
the probe's own `lastSyncReadbackBytes` before the fix. Once per run, and honest to describe as
tidiness rather than a speed-up. What it is worth more than the
second is that it removes the last instance of the pattern levers 25, 27 and 32 exist to close.

The fix is one loop, and the interesting part is where it lives. Under LoRA the base is frozen
before the gate and stages nothing, so the tensors to keep are the adapters, which `model.params()`
does not return. Putting that choice at the call site made it untestable: the gate sits inside a
500-line command that reads files and parses flags, so a test can reach the probe only by
duplicating it, and a duplicate goes green no matter what the real caller picks. The probe's GPU
half is now `probeGpuLosses`, taking the LoRA handle rather than a tensor list, so both branches of
the decision run inside the function the test calls.

`probeReadbackGate` in `tests/gpu-parity.ts` pins it, controls included. The dense control passes an
empty adapter list, exactly what the helper keeps with its loop deleted; the LoRA one passes `null`,
so the helper keeps the frozen base, which is a no-op twice over and leaves the adapters staging.
What the gate does not reach is the call site: `pretrain` passes `lora` to the helper and the test
passes its own argument, so the branch is pinned and the decision to hand it the handle is not. Deleting the
loop, collapsing the LoRA branch to `model.params()`, and collapsing the non-LoRA branch to the
adapters each fail it. Moving the loop to just before the `sync()` does not, and should not:
`keepGradOnDevice` is read at sync time, so anywhere before it is the same call.

### 38. A token file now carries the tokenizer that produced it (2026-09-10)

Filed as #81, recorded in lever 34 first, and the reason it needed its own entry is that lever 34's
check cannot reach it. `assertCorpusFitsVocab` catches a stale `.tokens` only when its ids exceed
the new vocab. A stale file's ids are all perfectly legal: a same-size or larger vocab passes every
range check, and so does a narrower one that flips the id width, since `diskTokenSource`'s size
check is only `% bytesPerToken`, so a 4-byte file read as 2-byte passes it, doubles the token count
and splits each id into a low and a high half.

The reuse that makes it reachable is in `pretrain`'s `.txt` branch, which keeps an existing
`${stem}.tokens` rather than rewriting it, while `sharedTokenizer` retrains the vocab whenever
`${stem}.tokenizer.json` is missing. Losing the json is the trigger, and then any difference in the
retrain lands a different vocab, whether from `VOCAB`, from `CURRICULUM_SPECIALS`, or from the
corpus sample itself. Changing `VOCAB` with the json in place does nothing, since `sharedTokenizer`
reuses the json whenever it exists. The run then trains on a token file built from a vocab that no
longer exists, with every gate green.

Nothing about an id says which vocab produced it, so the vocab is stored instead:
`<file>.tokens.id`, a small JSON holding a SHA-256 over the exported tokenizer plus the vocab size
and the id width. `tokenize`, `chat-corpus` and `pretrain`'s own writer stamp it; `pretrain` checks
it in both input branches and on `--inject`, which opens a second `.tokens` with the same tokenizer
in hand and lands on the cooldown phase. `tokenize` and `chat-corpus` both stamp after their
round-trip self-checks rather than before, so a file that fails one cannot ship with a valid
identity beside it. The byte size cannot substitute there: such a file is complete on disk and
merely wrong, so its size agrees, and stamping first would turn what used to be an unstamped file
into a false ok. The rule has a second half: every writer drops any stamp it finds before writing.
That is what makes "a stamp never describes a file it did not see" unconditional rather than
length-dependent, and it is what closes the crash window, since a run that dies between the write
and the stamp leaves an unstamped file rather than the previous run's stamp looking valid.

The stamp also carries the file's byte size, which after the drop is doing a different job than it
first appeared to. No writer here can leave a stale stamp any more, so the size is not the detection
for that. What it catches is a change that did not come from these writers at all: an external
truncation, an interrupted copy, a `.tokens` restored from backup while its `.id` stayed, or an
`.id` moved beside a different file. Hashing the whole export rather than the size catches the case
a size check misses entirely, a vocab of the same size whose merges retokenize the corpus
differently, and the specials too. The id width is compared on its own rather than folded into the
hash, because a width mismatch corrupts the read whatever the tokenizer says.

Reproduced end to end rather than argued. A 144 KB corpus, one step, then the json deleted and one
word changed so the retrain lands on a different vocab:

```
Tokenizer: trained to 337 tokens on 0.1M-char sample, 11 curriculum specials reserved
error: mini.tokens was tokenized with a different tokenizer: vocab 336 against the current 337.
```

Exit 1, where before it trained.

**An unstamped file is neither an error nor a pass.** Files written before this existed cannot be
checked, and both `pretrain` and `eval-loss` say so in a line of their own rather than staying quiet
and implying a check happened. It is the shape of decision invariant 3 already makes for a missing
`.optstate`, which re-warms rather than failing.

Backfilling the 23 `.tokens` on this machine would be cheap, not the obstacle: 22 have a sibling
`.tokenizer.json`, and `stampTokenFile` needs only that json and the vocab size, never the corpus.
The reason not to is that such a stamp would assert an identity nobody verified. It is computed from
the very json `siblingTokenizer` will later load, so the check could only ever return ok; all it
would buy is turning a future edit of that json into an error. Which means the scope limit has to be
said out loud: for those 23 files, deleting the tokenizer json is exactly as unguarded today as it
was before this change. What is closed is every file written from here on.

The gap that remains for a stamped file is deleting the stamp by hand, which no flow does.

`eval-loss` gets the width half and not the fingerprint, and the reason is not that it lacks a
tokenizer. `loadModelFromGGUF` returns one; the command discards it. The reason is that a
GGUF-derived `export()` is not guaranteed equal to the json-derived one the stamp was written from.
On export `token_type` is assigned by the shape of the token text (`<|...|>` becomes CONTROL)
regardless of what was declared, and on import `specials` is recovered from `token_type` in vocab
order rather than from a declared list. For a locally trained tokenizer the two coincide, because
`train()` appends specials in declaration order at the tail; for a foreign base they do not.
A fingerprint gate there would refuse correct corpora.

That is also what makes the gate sound in `pretrain`: it takes its tokenizer from the sibling json
or from `sharedTokenizer`, never from `--resume`, so the comparison is json against json and never
crosses the GGUF boundary. Someone simplifying `pretrain` to read the tokenizer off the resumed
checkpoint would start refusing correct corpora.

The width half is worth having on its own there, and cannot false-refuse: `bytesPerToken` is a pure
function of the vocab size both sides already agree on. Lever 34's scan catches a width mismatch
only by luck, when some low half of a 4-byte id happens to exceed the model's vocab, and `eval-loss`
is the one command that answers with a number rather than an error, which `scripts/score-*.sh`
publish.

The cases live in `tests/large-vocab.ts`, beside the preflight's. Reporting an unstamped file as ok,
dropping the width comparison, hashing only the vocab size, and treating an unparseable stamp as
absent, dropping the byte-size comparison, dropping the object guard that keeps a stamp of `null`
from throwing a TypeError instead of a message, and widening the read to treat every failure as
absent rather than only ENOENT, and the width path reporting a malformed stamp as absent or skipping
the byte size, the write no longer dropping a stale stamp, and the width path hashing the tokenizer
after all. Eleven mutations, eleven distinct assertions. Two more pin what the rest of this entry
asserts rather than leaving it as prose: that the fingerprint survives a json round trip through
`fromData`, which is the failure that would refuse correct corpora rather than admit wrong ones; and
the width flip itself, through `writeTokenFile` and `diskTokenSource`. That second one corrected
this entry: reading a 4-byte file as 2-byte splits each id into its low and high halves, and the
high halves are 0 only for ids under 65,536. "Every second id reads as 0" was the common case
written up as the rule, and a corpus that crossed the u16 ceiling is precisely the one where it is
false.

What no test reaches is the call sites. Deleting `assertTokenFileId` from any of the three leaves
the suite green, and so does moving either producer's stamp back above its round-trip check. The
end-to-end runs above are the evidence, and they are not repeatable in CI.

### 39. The parquet reader is imported when a parquet file is read, not before (2026-09-10)

Filed as #83. `deno task test:node` exists to catch Deno-only API use in code that has to run under
Node too, and it could not reach any of `eval-choice`'s logic. The reason was a dependency that
logic does not use:

```
$ node --experimental-strip-types tests/eval-tasks.ts
Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'hyparquet' imported from src/data/parse.ts
```

`tests/eval-tasks.ts` imports `src/commands/eval-choice.ts`, which imports `src/data/parse.ts` for
the HF parquet reader, which imported `hyparquet` at the top. None of what the test exercises touches
any of it: `choiceMaskStart`, `choiceWindowError`, `preflightByBytes`, `renderPair`, `withPreamble`,
`argminPerChar`, `hellaswagPreprocess` and the row parsers are pure string and integer work. So 79
assertions over the eval scoring arithmetic never saw the Node runtime, because of an import none of
them reach. `hyparquet` resolves through Deno's import map and there is no
`node_modules`, so the failure is at load time, before a single assertion runs.

Moving it inside `parseParquet` fixes the coupling rather than routing around it, which is why it
beat the alternative of splitting the pure helpers into their own module. The graph pulls the
package when a parquet file is actually read, `deno check` still resolves the types through the
dynamic specifier, and `eval-choice --task piqa`, which takes the JSON loader, stops paying for a
reader it never calls.

The issue named one file. Restoring the static import and running each candidate says the truth:
three were blocked by it, `eval-tasks.ts`, `style-pipeline.ts` (through `style-seed.ts`) and
`rp-chats.ts` (through `scripts/build-rp-chats.ts`). Four were not blocked by anything.
`generate-penalty.ts` had simply never been added, and three had exemptions written for them in the
first version of the guard: `endpoint-score.ts`, `rp-battery-score.ts` and `rp-chats.ts` were each
recorded as blocked by a Deno API in the script they test, when every one of those sits behind an
`import.meta.main` guard and never runs on import. Two of the three passed under Node the moment
anyone tried.

`rp-chats.ts` is the interesting one, because it was on both lists: its exemption named the wrong
cause while the file really was unloadable. A reason that is wrong about why is invisible even when
it happens to be right about whether, which is worth more as a warning than either half alone.
`test:node` went from 14 files to 21.

The check this leaves behind is the tests themselves: seven files join `test:node`, and putting the
static import back fails three of them with the same `ERR_MODULE_NOT_FOUND` while the other four
pass. Verified by doing exactly that, file by file. Parquet reading is unchanged, checked against `tests/fixtures/tiny.parquet` under Deno,
including the `subarray` path that exercises the `byteOffset` slice.

**The static import is only half of why those assertions were invisible.** It made `eval-tasks.ts`
fail under Node; what made that go unnoticed is that both task lists are hand-maintained strings in
`deno.json` with nothing comparing them to `tests/`. `generate-penalty.ts` proves the point on its
own, since nothing was ever blocking it and it was missing anyway. `tests/task-coverage.ts` compares
both lists against the directory, recursively, and against an exemption list carrying a stated load
failure per file. It matches the runner and not just the path, so pasting `deno run tests/foo.ts`
into the `test:node` string cannot report Node coverage that does not exist. One exemption survives:
`npm-deps.ts` statically imports `@huggingface/jinja`, and exercising the npm dependencies is the
point of that file, so the import cannot move inside a function.

Two static npm imports remain outside `parse.ts`, in `corpus.ts` and `chat-corpus.ts`. Both are
reachable only through `src/cli/registry.ts` and so only from `cli.ts`, which is Deno-only anyway,
and no test imports either. They are not a coverage problem today, and the rule for the day one is:
move the import, do not earn an exemption.

The exemption list is where this bug can come back, which is why the bar written into it is a load
failure someone has seen rather than a guess. Its own first version is the cautionary case: five
entries, three of them wrong, each reading authoritative.

**What this does not fix is that CI never runs `test:node` at all.** `.github/workflows/test.yml`
runs `deno fmt --check`, `deno lint`, `deno check` and `deno task test`, with no Node step, so the
21 files in that task are checked against Node only on a developer's machine. Filed as #89 rather
than folded in here, because adding a Node job is a change to CI's risk profile and not to this
module graph.

The list check does run in CI, since `task-coverage.ts` is in `test`. So the half of this that
survives #89 is the half that catches a file going missing; what waits on #89 is catching a file
that is listed and broken. Worth noting that a Node job would not have caught #83 either:
`test:node` was green then, and would have been green in CI, because `eval-tasks.ts` was not in the
list.

### 40. The loss guards came up above the backend dispatch (2026-09-10)

Filed as #82, noted in lever 30 while fixing #61, and the last instance of that pattern in the loss
path. `crossEntropy` dispatched before it validated:

```ts
export function crossEntropy(logits: Tensor, targets: number[]): Tensor {
  assertMatrix(logits, "logits", "crossEntropy");
  if (opsBackend) return opsBackend.crossEntropy(logits, targets);
  const [T, V] = logits.shape;
  const kept = keptRowsInVocab(targets, T, V, "crossEntropy");
```

The rank check was above, from #72. The id check was not: it ran in the CPU body, and the GPU path
was covered only because `webgpu.ts` called it separately. Same for `fusedCrossEntropy`, which
called it in both places. There was no live gap, and that is the point: a guard under a dispatch has
to be repeated in every implementation, and the one nobody remembers is the one that ships
unvalidated. That is what #61 was.

**What kept it down there was that `keptRowsInVocab` returns something.** The count is both
backends' loss denominator, so hoisting the call means either recomputing it on the GPU path or
widening the interface. This takes the second: `OpsBackend.crossEntropy` and `.fusedCrossEntropy`
gain a `kept` parameter.

The reason is not the recount's cost. T is thousands, in a step whose readout alone does T x V x H
multiply-adds, so an integer pass over T is not measurable and calling it a real cost would be
overstating it. The reason is that a recount reopens the hole by another route: nothing would force
a future backend to get its number from the _validator_, since it only needs a number, and a bare
counting loop satisfies the compiler while skipping validation entirely. That is #61 wearing a
different hat. Passing `kept` down leaves the backend nothing to compute, so nothing to compute
wrongly. Widening an interface for a guard is the objection, and the answer is that `kept` is not a
guard: it is a value both implementations already needed and both were separately computing.

**`softCrossEntropy` is now on the other side of that argument, and this change did not move it.**
#70 hoisted its `assertTeacherRows` above the dispatch, but that validator returns nothing, so each
backend still counts its own kept rows: `webgpu.ts` runs `teacherIds[t * k] >= 0` over T while the
CPU body counts inside its own loop. Two implementations of one quantity, and the repo now answers
the same question two ways. Filed as #93 rather than folded in, and closed in lever 43.

Two things improve on the way. A malformed target now refuses before `beginForwardOp` and before
any `entryFor`, where it used to refuse after both, so nothing is left half-recorded and no pooled
buffer is taken. And `webgpu.ts` drops its import of `keptRowsInVocab` entirely.

`targetRangeGate` already had the arms, exactly as #82 predicted. The mutation that reproduces the
finding is no longer "move the call below the dispatch", which will not compile now that the
dispatch line reads `kept`. It is the loophole above: give the dispatch a bare counting loop instead
of the validator's return.

```ts
if (opsBackend) {
  let n = 0;
  for (let t = 0; t < T; t++) if (targets[t] >= 0) n++;
  return opsBackend.crossEntropy(logits, targets, n);
}
```

That turns `dense true` into `dense false`, and the same shape on the fused path turns `fused true`
into `fused false`, while every CPU case passes. Measured both ways, and the detail worth keeping is
that the legal arm's loss does not move: 1.6022 either way. The loophole computes the right
denominator and skips the validation, which is why nothing but a placement gate catches it.

**One instance of the gap shape remained, and it was not this one** (closed in lever 42).
`linearRaw` checked `inDim !== inDim2` below its dispatch and `webgpu.ts` repeated the identical
check with the identical message. No gap at the time, same as here, and hoisting it was a different
change to a different function. Filed as #91 rather than folded in, since `linear` is the hottest op
in the graph and "the check is free" wanted measuring rather than asserting.

Two duplicates in `fusedCrossEntropy`'s GPU path were a different leftover: `H !== H2` and
`chunk <= 0` were repeated there with identical messages, and the wrapper had checked both above the
dispatch since before this change, so they were already unreachable. Dead rather than gap-shaped,
and listed with #91; both went in lever 42.

### 41. What makes buffer recycling safe is queue ordering, not the fence in the comment (2026-09-10)

Filed as #87 while reviewing #86. `sync()` returns every transient buffer to the pool at the end of
the window, and the comment that justified it read:

```ts
// The mapAsync completions above prove all submitted work finished, so
// every transient buffer is idle and safe to recycle.
```

Those completions are the `await`s over `stagings`. When nothing stages there are none, so the
sentence proved nothing: the loop released buffers on the strength of a fence that did not happen.
`train-gpu.ts` calls `gpu.sync(normTensors)` at every optimizer flush with both parameter groups
kept on device, and `normTensors` is empty unless MuonClip is on, so the ordinary training step
takes the empty-await path once per step.

**How far behind the GPU is at that point: a median of 783 ms.** Measured by inserting an
`onSubmittedWorkDone()` on the empty path and timing it, on a 6-layer 512-hidden gemma3 step at
batch 2 and sequence 512.

**But no shipped path releases a live transient that way, which the issue got wrong and so did the
first draft of this lever.** `this.transients` is already empty at that call site: the loss sync five lines
earlier drained the list, and `opt.recordStep()` allocates only optimizer state, which does
not come from the pool. Measured over four steps: nine syncs, four with nothing staged, and all four
released zero transients. The release loop there is a no-op, so the 783 ms measures how far behind
the GPU runs, not how long a live buffer sits exposed. Every in-tree release of a live transient
today still happens behind a real fence, `mapAsync` when something stages or `onSubmittedWorkDone`
in `reclaimStepTransients`. What reaches the fence-free path is the public bare `sync()`, which is
what the gate below exercises, and the trainer joins it the day a caller keeps gradients on device
and syncs with transients outstanding.

That also gives a sharper answer than "the ordering holds": putting a fence on the empty path would
cost about 783 ms per step at a call site that releases nothing at all.

**It is sound, on three legs that have nothing to do with waiting.** The premise the other two rest
on is that `submit()` runs first: it finishes the encoder and nulls `this.enc`, so by the release
loop every command that could touch a released buffer is already on the queue. That is what a later
edit would break silently, by adding a release path that does not submit first or by hoisting the
loop, and it is why a `queue.writeBuffer` issued after the release cannot run ahead of a reader that
had not reached the queue yet. Then: a pooled buffer can only ever be touched on the queue timeline,
because `BufferPool.acquire` creates them `STORAGE | COPY_SRC | COPY_DST` with no `MAP_READ` or
`MAP_WRITE`, so no host mapping can reach one behind the queue's back, and every access is a
dispatch, a copy or `clearBuffer` command, or a `queue.writeBuffer`. Readback goes through
`copyToStaging`, which makes its own mappable buffer. And queue operations run in issue order, all
of those being queue operations. A dispatch, copy or clear is recorded into
a later encoder and arrives in a later submit, whose queue-timeline steps are "for each
commandBuffer in commandBuffers: execute each command in commandBuffer.[[command_list]]"
([spec](https://www.w3.org/TR/webgpu/#dom-gpuqueue-submit)). A host write is a `queue.writeBuffer`
issued after that `submit()`, which is the live case rather than a hypothetical one: `uploadU32` and
`uploadF32` acquire a transient and write it immediately, so `embedding` and `fusedCrossEntropy` do
it on every micro-batch. Neither can overtake work submitted before it.

Leg 2 is also why this is the ordinary way to drive WebGPU rather than a trick: every renderer
rewrites its vertex and uniform buffers each frame and fences nothing. gpuweb#3809 asks for the
ordering to be stated outright; what the spec writes down today is the ordered execution quoted
above.

**So the fence in `reclaimStepTransients` is not what makes that path safe either, and removing it
measured no win.** That path does drain, with `onSubmittedWorkDone()` at each micro-batch boundary,
and the same argument says it does not have to. Measured at batch 4 on the same shape, 12 steps and
three boundaries each: 36 waits, median 1113 ms and mean 1281 ms, 46.1 s of the run's 60.4 s spent
blocked in it. Deleting it moved the median step from 5111 ms to 4942 ms, a 169 ms shift inside a
within-arm spread of 1007 ms, itself 20% of the median. The wait is not the saving: the CPU is
waiting for work the GPU has to do either way, and the only thing removing the drain recovers is the
command recording that could have overlapped it.

**It is kept because deleting it buys nothing, and for no reason beyond that.** An earlier draft of
this lever claimed a second one, that `reclaimStepTransients` is the only place in shipped code
releasing a live in-flight transient, so deleting the drain would put the ordering argument on the
critical path for the first time. That is false, and the review that caught it produced a better
sentence than the one it replaced. `endRegion` pushes still-in-flight transients onto `regionFree`
after a `submit()` and nothing else, `acquireRecycled` hands them straight to the next `makeOut`,
and its own docstring says so outright: "No fence, and none is needed." Every `--recompute` run does
this at every layer boundary, and `recomputeModelParity` gates it by name, "the region free-list
actually handing a released buffer to a later makeOut". Instrumented on a tiny gemma3 forward with
checkpointing on: 44 buffers came off `regionFree`, the loss's own gradient buffer among them. Unfenced reuse of live buffers is already
shipped, already exercised per layer, and already covered by a whole-model parity test.
`sync()`'s release is the same mechanism, not a new bet.

Which leaves the counter-argument standing on its own, and it deserves stating, because this file
sets the opposite precedent elsewhere ("dead code that reads like a safeguard is worse than neither,
so it went"): a fence whose docstring now says it is not the reason is that shape. It stays because
the alternative measured no faster, and the docstring says that rather than pretending the drain is
load-bearing.

**The gate was asserting the wrong thing too.** `syncFenceGate` ran one `linear`, called `sync()`,
then read the output back, under the name "sync() fences GPU even with no readback". It proved
neither half: nothing was recycled and reused in between, and there is no fence to prove. It is now
`recycleReuseGate`, which runs a forward and a backward, empties the pool with a bare `sync()`, and
then reuses exactly the buffers that were released. Three checks, only one of them mutable from
inside the repo:

- The bare `sync()` stages nothing. Without it the gate is not on the path under test at all, which
  is what the first draft got wrong: it staged 32768 bytes of gradients and this check caught it.
- The pool does not grow across the second chain. Delete the release loop in `sync()` and this
  fires: `pool grew 299763968 -> 299788544 on reuse`. This is what proves the reuse happens at all.
- The first chain's gradients still match the CPU. No mutation inside this repo forces it, and the
  gate says so: it fails only if an implementation lets a later submit overtake an earlier one.

**The backward is what makes the second arm mean anything, and the first draft did not have it.**
Forward-only, the four recycled `[T, HID]` buffers split into two disjoint sets: `makeOut` takes a
data buffer then a gradient buffer, the pool pops LIFO, so the second chain's writes land on the
first chain's gradient buffers, which a forward-only graph never touches, while the first chain
reads only its data buffers. Nothing overlaps, so no ordering violation could have perturbed the
result, and the arm passed for a reason unrelated to what it tested. That inversion is structural
rather than incidental: the data-then-grad acquire order puts every pop on the wrong side. With the
backward recorded, the second chain's two data buffers are both read by work still in flight, gelu's
backward reading one and linear's backward reading the other twice, and both feed the gradients the
comparison checks. Backward allocates nothing, so the pool assertion is unchanged: the mutation
number confirms it independently, 299788544 - 299763968 = 24576 = 4 x 6144, exactly the four
buffers and nothing else.

Two smaller consequences of the same fix. The chain ends in a `crossEntropy` scalar, because
`backward` seeds `loss.grad[0]` on the host and a non-scalar output never gets its seed to the
device. And the gradients are read from the persistent accumulators rather than from a graph
output, whose buffer the bare `sync()` has already returned to the pool: reading that would have
been a second way for the arm to pass without proving anything.

### 42. `linear`'s dim check came up above the dispatch, and the dead copies went (2026-09-10)

Filed as #91 in lever 40, as the last instance of the shape that produced #61. `linearRaw`
dispatched before it validated, and `webgpu.ts` repeated the check with the identical message, so
there was no live gap and no way to tell which copy threw:

```ts
function linearRaw(x: Tensor, w: Tensor): Tensor {
  if (opsBackend) return opsBackend.linear(x, w);
  const [T, inDim] = x.shape;
  const [outDim, inDim2] = w.shape;
  if (inDim !== inDim2) throw new Error(`linear dim mismatch ${inDim} vs ${inDim2}`);
```

**The cost question the issue raised does not survive contact with the code.** `linear` is the
hottest op in the graph and the worry was that hoisting puts a guard on that path. It was already
on that path: the GPU copy ran on every call. Hoisting moves the comparison, it does not add one.
What the GPU path genuinely gains is the two shape destructures above the dispatch, which used to
happen only inside the backend, and that is small enough to put a number on rather than wave at.
Counted through the real training loop, `linearRaw` runs 86 times per step at 6 layers and batch 2
(43 per micro-batch: seven projections across six blocks, plus the readout), and 170 with
`--recompute`, since 42 of the 43 sit inside a `checkpoint()` whose replay re-runs them in backward.
Two destructures and a compare measure 4.10 ns, so 0.0007 ms per step at the higher count, against a
step of seconds. A before/after step timing would have measured this machine's variance and nothing
else: medians for the same configuration ranged 1222 ms to 2928 ms across runs.

**It is a small improvement rather than break-even, by the argument the repo already makes.** The
GPU refusal used to fire after `beginForwardOp()` and `curLabel = "linear"`; it now fires before the
backend is entered at all, so a refusal leaves no profiler state behind. That is the property lever
40 claims for `crossEntropy`.

Unlike #82 there is no interface to widen: the check returns nothing, so the dispatch line is
untouched.

The message now names both shapes, since after the hoist there is one copy and it is the only thing
the caller gets:

```
linear dim mismatch: x is [24, 64] and w is [40, 65], so the contracted dimension is 64 on one side
and 65 on the other. A projection wired to the wrong config field is the usual cause.
```

The cause it names is the one that can actually reach it. A LoRA adapter of the wrong width was the
first draft's guess and cannot happen: `applyLora` derives both factors from `w.shape`, there is no
adapter deserializer, and `--lora-rank` requires `--resume`, so the base is loaded and its config
checked first. What reaches this throw is a projection wired to the wrong config field, or a
hand-built graph in a test. The shapes print through `shape.join(", ")` to match `assertRank`
rather than rendering `[3,4]` where the neighbouring guard renders `[3, 4]`.

**Two dead checks went with it.** `WebGPUBackend.fusedCrossEntropy` repeated `H !== H2` and
`chunk <= 0` with the wrapper's exact messages, and the wrapper has validated both above the
dispatch since before #82. Nothing reaches that method except the wrapper: `sequenceLoss` is the
only caller of `fusedCrossEntropy` in `src/`, and it goes through `autograd.ts`.

`targetRangeGate` gains three arms and a name that fits them (`GPU refuses malformed op inputs`,
since `linear` is not a loss). What each one is worth is not the same:

- `linear`: putting the check back below `if (opsBackend)` turns `linear true` into `linear false`
  while every CPU caller keeps passing. That is the signature of this mistake, and the third time
  the gate has caught it.
- `fusedDim`: deleting the wrapper's `H !== H2` turns `fusedDim true` into `fusedDim false`. Clean.
- `fusedChunk`: deleting the wrapper's `chunk <= 0` does NOT turn this arm false. The suite dies
  with `Fatal JavaScript out of memory` inside the arm, 13 checks in. With the backend installed the
  wrapper dispatches, so what spins is `webgpu.ts`'s span builder,
  `for (v0 = 0; v0 < V; v0 += chunk) spans.push(...)`, which never advances on a zero chunk and
  allocates without bound. Inside the very method whose duplicate this change deleted. Exit 133
  rather than exit 1: the regression is caught, but by exhaustion rather than by the assertion, and
  that is worth knowing before someone reads a red suite and looks for a mismatch line.

**What the deletion gives up, stated plainly.** With the copies in place, a wrapper whose guard was
removed still threw cleanly from the backend. With them gone, the wrapper is the only thing between
a mismatched shape and the heap exhaustion above. That is the standing trade of #61 and #82: a
duplicate is also what lets the real guard rot unnoticed, and the arms above are the replacement.

### 43. `softCrossEntropy` counts its kept rows once, above the dispatch (2026-09-10)

Filed as #93 in lever 40, as the function left on the wrong side of the argument that lever made.
#70 hoisted `assertTeacherRows` above the backend dispatch, but that validator returned nothing, so
each backend went on computing the kept-row count itself:

```ts
// src/backend/webgpu.ts
let kept = 0;
for (let t = 0; t < T; t++) if (teacherIds[t * k] >= 0) kept++;
```

```ts
// src/model/autograd.ts, inside the CPU forward loop
kept++;
```

One quantity, two implementations, nothing comparing them. They agreed, and nothing made them keep
agreeing. A divergence would not throw either: the count is the loss denominator, so a wrong one
reports a plausible number. `crossEntropy` and `fusedCrossEntropy` had exactly this shape until
lever 40 and now do not, so the repo was answering the same question two ways in adjacent
functions, and the older-looking answer was the one a new backend would copy.

**Taken the way #82 took it.** `assertTeacherRows` becomes `keptTeacherRows`, returns the count, and
`OpsBackend.softCrossEntropy` gains a `kept` parameter. The rename is the point rather than a
tidy-up: the old name promised validation only, and a function that also returns the denominator
should say so, the way `keptRowsInVocab` does. The count moves into the loop that was already
walking the rows, so it costs nothing, and the CPU forward loses its own `kept++`.

The alternative the issue listed, leaving the recount and pinning the agreement with a gate, is
cheaper and tests the property rather than removing the possibility. It was not taken for the reason
lever 40 gives: a backend that recounts only needs a number, and a bare counting loop satisfies the
compiler while skipping the validation the same pass does. Passing `kept` down leaves nothing for a
backend to get wrong.

**The mutation is lever 40's, and it reproduces.** Give the dispatch its own counting loop:

```ts
let kept = 0;
if (opsBackend) {
  for (let t = 0; t < T; t++) if (teacherIds[t * k] >= 0) kept++;
  return opsBackend.softCrossEntropy(logits, teacherIds, teacherProbs, k, kept);
}
kept = keptTeacherRows(teacherIds, teacherProbs, T, k, V);
```

That turns `softCE true` into `softCE false` on `targetRangeGate` while every CPU case passes,
which is the signature of a guard that stopped covering the GPU path. The loophole computes the
right denominator; what it drops is the validation.

**What this costs, and the test that pays it back.** The old arrangement had a weak cross-check
nobody designed: two independently written counts, compared through a parity case that carries
ignored rows, are unlikely to be identically wrong. One shared count scales the CPU and GPU losses
together, so the parity cases go on passing. Measured rather than assumed: with `keptTeacherRows`
changed to `return T`, and before the case below existed, the entire suite exited 0.

The repo already carries the warning, over the oracle that answers it. `chunked summed NLL` came in
with #51 and the sentence above it with #62: "The oracle is computed here rather than taken from
either loss: both draw their denominator from the helper, so comparing them to each other would
agree on a wrong kept count." (#82 gets the credit for this in a first draft of this lever and
deserves none of it; it never touched `gradcheck.ts`.) This ships `softCE summed NLL`, the same
shape for the soft-target loss: a CPU-only case whose expected value is computed in the test and
whose `kept` is written out as the literal 3 rather than derived. `return T` fails it and only it,
at `maxAbs=1.67e+0`, and so does moving the `kept++` above the ignore-marker `continue`, which for
this input is the same mutation: one ignored row means both return `T`. Separating them needs a case
with two.

The case takes its own seed rather than the shared `rng`, so it does not shift every case below it
onto different inputs.

The `T`-divisor mutation still has its own job, and it is a different one. Replacing the GPU divisor
with `T` fails the parity cases at `gpu=2.3589 cpu=3.1453`, a ratio of exactly 3/4: that catches a
consumer of the count diverging from the other consumer. The new case catches the shared producer
being wrong, which nothing could see. Three mutations, three different failures.

### 44. CI runs the Node suite, which it never did (2026-09-10)

Filed as #89 while closing #83. `.github/workflows/test.yml` ran four things, `deno fmt --check`,
`deno lint`, `deno check`, and `deno task test`. No Node step. So `deno task test:node`, whose whole
purpose is principle 1, that everything the model itself needs runs on Deno, Bun and Node with no
npm install, was checked only when someone happened to run it on their own machine.

#83 is what that costs. A static `hyparquet` import in `src/data/parse.ts` kept 79 assertions over
`eval-choice`'s scoring arithmetic out of `test:node` entirely, and four more files were missing for
the other reason, that nothing compared the two task lists to `tests/`. Both fixes landed with
guards that no CI could enforce.

**What made it its own issue rather than four lines in #83 was the risk profile, and the answer is
that all three worries were unfounded.** Measured by running it:

| worry                                                                         | measurement                                                                     |
| :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------ |
| The strip-types flag has moved across 22.x and 23.x, so the pin is a decision | `22` resolved to v22.23.2 and `lts/*` to v24.20.0 on `ubuntu-latest`; both pass |
| The 21 files pass on one developer's Node, not necessarily the runner's       | Both matrix entries green on the first run                                      |
| `test` and `test:node` overlap, so the job roughly doubles the suite          | 21s and 20s beside the Deno job's 20s, in parallel: no wall-clock change        |

Locally for comparison, `test:node` is 6.9s against `test`'s 14.8s, because the GPU parity file
skips without a WebGPU adapter under Node.

**Both entries are pinned, and that first run is why.** `node-version: "22"` resolves to the newest
22.x, so the entry called "the floor" tested everything except the floor: `readme.md` promises Node
22.6+, and 22.6 through 22.17 carry an older stripper than the v22.23.2 that ran. It is now
`22.6.0`, which makes that entry test the promise, and if it ever fails the promise was wrong and
belongs raised rather than worked around. `lts/*` went the same way for a different reason: it
resolved to v24.20.0, and the LTS line rolls to the next major on a calendar, so keeping it would
eventually turn CI red on a date rather than on a change, in a repo where red blocks every PR. `24`
moves when a PR moves it. `fail-fast: false` so one version failing still reports the other.

**What the job does and does not prove, since a green matrix entry is easy to over-read.** Of the 21
files, 20 run their checks under Node. `gpu-parity.ts` prints `SKIP: no WebGPU in this runtime` and
exits 0, so none of its assertions run; what it still proves is that its whole static import graph,
the WebGPU backend, the trainer and `src/commands/pretrain.ts`, loads under Node with no npm
install. That is exactly the #83 class of defect, which makes it one of the more valuable entries
rather than dead weight. `large-file-write.ts` runs its round trips but leaves its big-IO case
behind `GGUF_TRAINER_BIG_IO=1`, skipped under both runners. The exit code does propagate: the task
chains with `&&`, and an injected `process.exit(3)` in the first file surfaced as `task exit=3`
rather than being swallowed.

**Principle 1 names three runtimes and this enforces two.** Bun has no task and no job. Filed
separately rather than folded in, because a `test:bun` task is a decision about a third runtime
rather than a line of YAML.

It is a separate job rather than a step on the existing one, so it runs beside the Deno suite
instead of adding to its wall clock. `denoland/setup-deno` appears in it only to read the task
string out of `deno.json`: spelling the file list into the workflow would make it a third copy of a
list `tests/task-coverage.ts` exists to keep at two. The cost of that is narrow and worth naming: a
Deno-side breakage reddens the Node job for a non-Node reason.

The workflow also gained a `concurrency` group, because it went from one job to three and a
superseded push now wastes three runners rather than one. It cancels on `pull_request` only: a
cancelled main build leaves no record of whether that commit was ever green.

### 45. `bench` fenced its timed pass by accident (2026-09-10)

Filed as #99 while reviewing #98. `bench` times one pass per case like this:

```ts
/** One timed pass: forward (+ backward), then a fence. */
async function once(gpu: WebGPUBackend, c: Case) {
  ...
  await gpu.sync([]);
}
```

The comment says "then a fence", and there is one, but not for the reason the call looks like.
`sync([])` passes no reads, so the only thing that can stage is `touchedExternals`, and every case's
inputs come from `randTensor`, which builds them `requiresGrad`. Their gradients stage, the
`mapAsync` awaits are real, and the pass waits for the GPU. Lever 41 is what makes that load-bearing
rather than incidental trivia: a `sync()` with nothing staged submits and returns without awaiting
anything.

**The two things that would break it are both things a plausible new case would do.** Frozen inputs,
which is what a forward-only or LoRA-shaped case naturally uses, or a call to `keepGradOnDevice`,
which is what a case modelling the resident training loop would copy. Either empties `stagings`, and
the pass is then timed to the submit rather than to completion. Everywhere else in the repo that
failure produces a wrong result you can see. Here it produces a plausible number, in the one command
whose entire output is numbers.

**Fixed by asserting rather than by fencing, and the reason is what the number means.** An explicit
`onSubmittedWorkDone()` would restore the wait and cost nothing today, but it would let such a case
exist: this file's header defines the wall number as including the host-side graph build and the
gradient readback, so a case that stages nothing has already stopped reporting what the column
claims, fence or no fence. A fence keeps one half of the header's promise true while the other goes
quietly false for one row of a column whose other rows keep it, which is a worse artifact than a
hard failure and the same class of bug as #99 itself.

**Byte-exact, not `> 0`, and the first draft got that wrong.** `lastSyncReadbackBytes` sums every
touched external, so one live input keeps it non-zero. A case that freezes only SOME of its inputs
therefore keeps the fence and loses part of the readback, and a zero check waves it through. That
partial shape is the likelier mistake, not the rarer one: the natural additions here are
LoRA-shaped or inference-shaped, where some tensor is frozen. So the check compares against what
`Case.inputs` declares, summed as `t.size * 4` and deduped, computed once per case in `timeCase` so
the measured window is untouched.

Both modes reproduce, on the `rmsnorm` case whose inputs are 5245440 bytes of gradient:

| mutation                                   | reported                    | caught by `> 0`? |
| :----------------------------------------- | :-------------------------- | :--------------- |
| `keepGradOnDevice` on every input          | `read back 0 of 5245440`    | yes              |
| `keepGradOnDevice` on the first input only | `read back 2560 of 5245440` | **no**           |

The published numbers do not move: the assertion is one integer compare on a path that allocates
nothing, and `bench --suite all` passes every case with the exact equality, which is also the
measurement that confirms `sync()` stages exactly `t.size * 4` per live external.

The assertion is two-sided by construction, and the message says so: fewer bytes than declared means
an input is frozen or kept on device, more means `run` touched an external that `inputs` does not
list, which also missed its `zeroGrad`. `Case.inputs` now documents that contract, since it is the
declaration being checked rather than a note about what to keep alive.

**One review suggestion declined.** The error used to offer "or read something back explicitly",
which `Case` has no way to express, so the advice was unreachable; the suggestion was to add a
`reads?: Tensor[]` field and forward it to `sync()`. Declined on caller count, and the second round
gave the better reason: those bytes would have to count toward the expected total, and then a
gradient-free case satisfies the assertion with a DATA readback while the header promises the wall
number carries the gradient one. The field would buy the fence back at the cost of the property the
assertion exists to protect. The message now names only the option that exists.

No test file, deliberately. `bench` needs a GPU and is not in `deno task test`, and the check runs on
every real invocation, which is where it belongs. The property it depends on, that
`keepGradOnDevice` empties the staging list, is already pinned in `recycleReuseGate`.

### 46. Bun, the runtime principle 1 named and nothing checked (2026-09-10)

Filed as #97 while reviewing #94. Principle 1 says everything the model itself needs must run on
Deno, Bun and Node with no npm install. There was a `test` task and a `test:node` task. Nothing for
Bun, so #94 left the claim enforced for two runtimes out of the three it names.

**The probe answered the first question immediately and the interesting one second.** All 21 files
run clean under Bun 1.4.2 on `ubuntu-latest`, in 12s beside the Deno job's 19s, with the same skip
profile as Node: `gpu-parity.ts` prints its no-WebGPU SKIP, `large-file-write.ts` leaves its big-IO
case behind `GGUF_TRAINER_BIG_IO=1`. So a job was warranted rather than an edit to the principle.

**Then `bun run tests/npm-deps.ts` passed, and that is the finding.** That file is the one Node
cannot load: it statically imports `@huggingface/jinja`, a bare specifier that resolves through
Deno's import map, and with no `node_modules` it is `ERR_MODULE_NOT_FOUND`. Under Bun it works. Not
because Bun resolves it, but because Bun downloads the package from the registry at runtime:

```
bun run --no-install tests/npm-deps.ts
  error: Cannot find module '@huggingface/jinja' from '.../tests/npm-deps.ts'
bun run tests/npm-deps.ts
  passes
```

A Bun job that ran the files plainly would therefore satisfy "no npm install" by installing, and
would go green while proving the opposite of what it claims. That is the #83 shape again, one layer
down: a check whose green means something other than what its name says.

So `--no-install` is in the task rather than in the workflow, since it is a property of what the
task asserts and not of where it runs: a contributor running `deno task test:bun` gets the promise
CI gets. All 21 files still pass with it, which is what makes the flag free to take.

**And the flag is checked, because a comment saying "do not delete this" is what already failed
here.** Review's first finding on this change was that `--no-install` was load-bearing and
unenforced: strip it from all 21 invocations and every check in the repo stayed green while the job
went back to satisfying "no npm install" by installing. Note the asymmetry with Node, which does not
need this: dropping `--experimental-strip-types` fails loudly on the 22.6.0 leg, so that task
enforces its own flag. The Bun one cannot, so `tests/task-coverage.ts` carries a `needs` list per
runtime and asserts every invocation still has it, and CI runs
`bun run --no-install tests/npm-deps.ts` as a positive control that FAILS, so if auto-install ever
comes back by another route (a warmed cache, a `node_modules` some future config materializes) the
job says so instead of quietly proving nothing.

`npm-deps.ts` is exempt under Bun for the same reason it is under Node, and that is structural
rather than luck: Bun does not read `deno.json`'s `imports` and there is no tsconfig `paths` here,
so under Bun a bare specifier can only resolve through `node_modules`. Any future test that imports
one is exempt from both by construction.

**`tests/task-coverage.ts` went from two lists to three, as a table rather than a third copy of the
loop.** Its whole reason for existing is that two hand-maintained strings in `deno.json` drifted
from `tests/` and nothing compared them, so growing it by copy-paste would have been the joke
telling itself. Dropping a file from `test:bun` now fails with the file, the task and the command to
run:

```
tests/eval-tasks.ts is not run by `deno task test:bun` and has no exemption entry. Run
`bun run --no-install tests/eval-tasks.ts`: if it passes, add it to the task; if it cannot load,
add it to this file's exemption map with what fails
```

Pasting `node tests/eta-fmt.ts` into the Bun string fails the same way, because the check matches
the runner and not just the path.

That last one had a hole the review found: the runner match was per FILE, so a wrong-runner path was
invisible whenever the file was also listed correctly. Appending `&& node tests/eta-fmt.ts` to a
complete `test:bun` fired nothing, because `eta-fmt.ts` was already listed by the invocation above
it. A count of the `tests/` paths the string mentions against the count the runner claimed closes
it, and the message names the three ways it can happen:

```
deno.json's test:bun task mentions 22 tests/ paths but 21 are invoked by "bun run --no-install";
one is handed to another runtime, written ./tests/, or passed as an argument rather than as the
entry point
```

It counts occurrences on both sides rather than against the `Set` of files, so a task that runs one
file twice on purpose is not a false failure blaming a cause that is not there. `test` gets a row of
its own for this check, since the hole was never Bun-specific.

The Bun version is pinned to 1.4.2 rather than `latest`, for the reason lever 44 gives for pinning
the Node ceiling: Bun has no LTS line, so `latest` turns CI red on a release date rather than on a
change, and a moving version makes the numbers in this lever unreproducible. The pin buys a second
thing, which is that the positive control can assert bun's actual wording. It checks the message
rather than the exit code, because `if bun run ...; then fail` treats every nonzero exit as the
expected one, and a broken import or a crash inside `npm-deps.ts` would then pass the control while
auto-install was quietly back on. Unlike Node, no floor is claimed: `readme.md` says CI pins 1.4.2
and that no older Bun is promised, because none was tested.

**And one check closes the shape behind both #89 and #97.** Each was the same story: a task existed,
was correct, and nothing on a runner invoked it, so its guarantee held only for whoever ran it by
hand. `tests/task-coverage.ts` now reads `.github/workflows/test.yml` and asserts each of the three
tasks appears as a `run:` step. Replace `deno task test:bun` in the workflow with anything else and
it fails with "that task is checked only by whoever runs it by hand. That is what #89 and #97 were".
It couples a test to a CI file, which is the objection; comparing two hand-maintained lists is what
this file already exists to do, which is the answer.

**What the Bun job catches that the Node one cannot.** Not stricter syntax: Bun runs TypeScript
natively rather than through Node's stripper, so it accepts things Node refuses. What it covers is a
Web API or a `node:` builtin behaving differently there, which is the half of principle 1 that had
no check at all.

### 47. A 2x measurement artifact, and what it cost (2026-09-10)

Lever 20 named a follow-up: "the same overlap is available without recomputing anything, by
submitting at layer boundaries on the dense path too." Measuring it turned into a lesson about
measuring, which is the more useful half and is why this lever leads with it.

**Whatever else is running on this box is worth 2x on this step.** Same binary, same shape, same
flags, same seed:

| arm           | nothing else known running | with other work running |
| :------------ | -------------------------: | ----------------------: |
| dense         |                  113 tok/s |         54 and 58 tok/s |
| `--recompute` |                  248 tok/s |               117 tok/s |

That is 1.95x to 2.09x on the dense arm and 2.12x on the other. **What the other work was is not
established, and a draft of this lever said CPU with more confidence than the evidence carries.**
Review agents were running, which is CPU. So, it turned out later, was a second agent session on
this machine driving an unrelated ROCm job in bursts, which is the GPU. Which of the two moves the
number, or whether both do, is unmeasured; the run set up to settle it timed out when the other
session's job restarted mid-run, which is the lesson rather than an inconvenience. Nothing in the
output says any of it: the run prints a plausible tok/s whatever else is on the machine.

**The factor is not constant, and that is what actually corrupted the drafts.** If it were, ratios
would survive it and three loaded A/Bs of one unchanged line would have agreed with each other. The
54-versus-58 on one arm is the visible edge of the same variation, and an intermittent GPU tenant is
one thing that would produce exactly that. What the controls do establish is
that the arms were the same work at different speeds: peak GPU reads 16227 MB in the quiet dense
runs and in the loaded ones alike, and three separate builds (yesterday's HEAD, current main, the
patched tree with the flag off) give 114, 110 and 113 quiet. So it is neither a config difference
nor a regression from the day's merges, both of which were checked first and were wrong.
Three earlier drafts of this lever reported +16%, -32% and +57% for the same one-line change,
because the arms were interleaved with other work. **Any `pretrain` A/B on this box is void unless
the machine is otherwise idle**, and the file's existing bar, lever 20's "re-run once to confirm",
does not catch it: a repeat under the same load reproduces the same wrong number.

**Measured quietly, the follow-up does nothing.** qwen3, 28 layers, hidden 544, 16 heads of 128, seq
2048, batch 2, `--reclaim --loss-chunk 8192`, vocab 151936, 6 steps:

| arm                                                                                      |         tok/s |
| :--------------------------------------------------------------------------------------- | ------------: |
| dense, three builds (yesterday's HEAD, current main, the patched tree with the flag off) | 114, 110, 113 |
| dense + `submit()` at every layer boundary                                               |           117 |

117 is 2.6% above the fastest of those three, against a 3.6% spread among the three themselves, so
the gap is the size of the noise rather than inside it. Either way the honest reading is no
measurable effect rather than a small win. Lever 20's follow-up is closed as "measured, nothing
there" rather than as refuted, and nothing ships.

**Lever 20's headline reproduces.** Quiet, on current main, the same shape gives 110 tok/s dense and
248 with `--recompute`, a ratio of 2.25x against its 2.3x. Both absolutes are about 1.5x above its
72 and 168, which is either a faster stack since 2026-09-09 or its own numbers having been taken
with something else running; this lever cannot tell which and does not claim to. The peaks differ
too, 16227 MB against its 18974 MB, so the shape is close rather than identical: lever 20 never
recorded its `--heads`, and hidden 544 is not a multiple of the default head dim, so it must have
passed one. That 16227 is also the control above, which is why it is worth carrying twice.

**What this lever explicitly does not establish.** Deleting `endRegion`'s `submit()`, which lever
20's corollary says returns the throughput to the dense number, was measured only under load and
those numbers are discarded; it remains unmeasured. So does any account of what `--recompute` buys
beyond the 2.25x itself. `gpu_busy_percent` was tried as a discriminator and its readings discarded by
this lever's own rule: the conditions they were taken under were not recorded. A draft blamed the
98-100% they showed on the counter being pinned. It is not pinned. Sampled at 10 Hz for 30 s with
no heavy workload on the device, it reads **mean 43.2%, median 17%, max 99%, min 0%** over 300
samples, so it spans its range and is not clipped at the top. The arm that would show it is
LINEAR, back-to-back dispatches against an otherwise idle device, was not run: the other tenant took
the GPU back first. "Idles about half the step" is a linearity claim and inherits that gap.

**But that same arm is what stops 52.5% meaning what it looks like.** The counter reads 43.2% mean
with NO training on the device at all, against lever 1c's 52.5% during a run, over 40 samples, on a
distribution that runs 0 to 99 with a median of 17. Those two are not distinguishable at that
sample size. What survives is the weaker and still sufficient reading: 52.5% is whole-device busy,
so it is a CEILING on our trainer's share and the device really was idle for a good part of the
wall clock. That carries "not GPU-bound" and it carries lever 19's "the added work lands in a gap
that was already idle". It does not carry 52.5% as the trainer's utilization, and the rule below
bites it as hard as it bites the 42%: neither reading records what else held the device.

**What it does not do is attribute.** It is a whole-device counter, and this box is shared. The
first calibration attempt read **mean 99.8% over 200 samples with our trainer not running at all**,
and `fuser` showed why: a second agent session on this machine had the render node open for an
unrelated ROCm job. That is the one reading here with a verified device record, and it is what the
attribution rests on.

The rest is weaker and is offered as a hypothesis rather than a finding: the 98-100% readings taken
during training look like that same tenant, and the one 76.6% reading looks like its absence. Only
that one arm's device state was ever checked with `fuser`, so those are retro-inferences from the
counter, which is the thing under suspicion. The check that would have settled it is one command:

```
fuser -v /dev/dri/renderD128
```

which on this machine routinely lists a compositor, an idle `llama-server`, and, during this work,
that second session's job. Nothing in `gpu_busy_percent` tells our trainer's share apart from
theirs. The 43.2% calibration above was taken with `fuser` listing only the compositor and the idle
`llama-server`, checked before and after; that is the standard the rest of the readings here do not
meet.

**This is where the retraction at the top of the lever came from.** A draft named CPU contention as
the cause because that is what was known to be running; the tenant above is a second candidate, and
neither was checked per run. A later attempt to settle it timed out because the other session's job
restarted mid-run, which is itself the point: on a shared box the control has to be verified rather
than assumed, before and after.

So the rule this lever leaves behind is not "run it quiet", which is unfalsifiable, but: record what
else held the CPU and the render node, check both at the start and the end of the run, and treat any
number without that record as unusable.

Read literally that voids this lever's own tok/s figures too, which have no such record. They
survive on something the counter readings cannot offer: they come from the trainer's own output
rather than a shared counter, and three independent builds agree within 3.6% (114, 110, 113). A
single unlabelled reading of a whole-device counter has neither property. That includes the `~42%` in this file's summary table, whose
device conditions are not recorded. Principle 1 does not rest on either reading: it rests on the
profiled kernel time, and the counter appears there only as a ceiling.

To re-run the follow-up: `checkpoint()`'s passthrough at `if (!checkpointing) return fn()` needs the
backend's `submit()`, which is private on `WebGPUBackend` and absent from the `RegionBackend`
interface `checkpoint` holds, so it takes a cast or an interface member. And run it on an idle
machine, which is the whole point of this entry.

### 48. An all-ignored batch, on the device (2026-09-10)

Surfaced while reviewing #96. `kept` is the loss denominator, and every path clamps it the same way:

```ts
const denom = kept > 0 ? kept : 1;
```

Six times. Three in `autograd.ts` and three in `webgpu.ts`, one per loss per side. Without the clamp
the mean is 0/0; with one missing on one side, that path reports NaN while the other reports 0, and
neither throws.

`tests/gradcheck.ts` drives `every row ignored` through the two hard-target losses, and it runs with
no backend installed, so it only ever exercised the CPU copies. The three device copies had nothing
driving them, and `softCrossEntropy` had nothing on either side. A masked batch is not exotic:
assistant-only loss masking produces one whenever a training window lands entirely inside a prompt.

**The sixth clamp needed its own case, in gradcheck rather than here.** `gpu-parity.ts` returns
`SKIP` with no adapter, and CI has no GPU, so the device gate does not run there. The CPU clamps for
the two hard-target losses are covered anyway by gradcheck's existing `every row ignored`, but the
CPU `softCrossEntropy` clamp was covered by nothing that runs GPU-less: the `teacher id range`
block's smallest kept count is 2. One case in that list closes it, and the whole point is that it
runs where the suite actually runs.

`allIgnoredGate` runs all three losses over a fully masked batch with the backend installed, and
requires both the device answer and the CPU reference to be exactly 0, with a scored batch beside
them in the same readback as the control. Six mutations, one per clamp, each failing in exactly its
own arm:

| clamp removed       | what the gate prints                                                                                                      |
| :------------------ | :------------------------------------------------------------------------------------------------------------------------ |
| `webgpu.ts` dense   | `dense NaN, fused 0, softCE 0`                                                                                            |
| `webgpu.ts` fused   | `dense 0, fused NaN, softCE 0`                                                                                            |
| `webgpu.ts` soft    | `dense 0, fused 0, softCE NaN`                                                                                            |
| `autograd.ts` dense | `MISMATCH allIgnored.dense: gpu=0 cpu=NaN`                                                                                |
| `autograd.ts` fused | `MISMATCH allIgnored.fused: gpu=0 cpu=NaN`                                                                                |
| `autograd.ts` soft  | `MISMATCH allIgnored.softCE: gpu=0 cpu=NaN`, and GPU-less, `teacher id range` reports `failed: every teacher row ignored` |

Each device row also prints the summary line and each CPU row also prints its `MISMATCH`; the table
shows whichever is the more useful of the two.

**The bottom three needed a fix to the gate before they failed at all, and it is the interesting
part.** The first draft checked `!Number.isFinite(got[i]) || Math.abs(got[i] - cpu[i]) > 1e-6`,
which looks symmetric and is not: with a NaN reference, `Math.abs(0 - NaN)` is NaN and `NaN > 1e-6`
is false, so a missing CPU clamp passed while the arm claimed to cover it. A comparison against a
NaN oracle is not a weak test, it is a test that cannot fail.

The fix went further than adding a second finiteness check. The arm now asserts exactly
`got !== 0 || cpu !== 0`, which is stronger and shorter: exact zero is guaranteed rather than hoped
for, since every kernel writes 0 for an ignored row and the CPU totals are sums over no terms, so
nothing accumulates and there is no float noise to absorb; and `NaN !== 0` is true, so one
comparison covers both sides with no finiteness test at all. Written the other way round,
`Math.abs(got[i]) > 0` is false for NaN, which is the same trap again.

Three limits worth stating. At `kept == 0` the numerator is 0 too, so no arm here can grip the
count's VALUE, only the clamp; the value is pinned by the partial-mask parity cases and by
`chunked summed NLL` and `softCE summed NLL`.

WGSL does not promise that `0.0 / 0.0` is NaN, and a draft of this paragraph said the CPU comparison
made the device arms adapter-independent. It does not, and the truth is the reverse: the check is
`got !== 0` and `cpu !== 0` independently, so an adapter that yields 0 there makes all three device
mutations invisible, and the CPU oracle cannot rescue them because it is 0 too. The three device
rows are what THIS adapter did.

And the arm's pass condition is "everything is 0", which an unread host buffer also satisfies, since
`makeOut` hands back a `Tensor.zeros`. Deleting the readback made the first version pass on
unwritten memory. Two things close that. A scored batch rides in the same `sync` and must be nonzero
and match the CPU, which is the control `targetRangeGate` carries for the same reason; with the
readback deleted it reports `scored control 0.0000, 0.0000, 0.0000`. And every one of the six host
scalars is seeded to `NaN` first, so each proves individually that the readback wrote it: `NaN !== 0`
fails an ignored arm and `!(Math.abs(NaN) > 1e-6)` fails a control arm. The control alone would not
have caught an op that never reached the device at all, since it takes the real path; the seed does.
With the sync deleted, all six now report.

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

`eval-choice` scores a GGUF on ARC-Challenge / HellaSwag via length-normalized log-likelihood,
plus acc/acc_norm; datasets via the HF parquet tooling. Everything in this lever predates the
parity fix in 9c, so its acc_norm figures were computed with a different normalizer than the ones
below and do not compare to them.

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

### 9c. Two divergences from lm-eval-harness, and what fixing them moved (2026-08-21)

Lever 9 claimed `eval-choice` matched lm-eval-harness. It did not, in two places, both found by
reading our scoring against the reference implementation rather than by any test failing:

1. **The HellaSwag query.** lm-eval scores
   `preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize())`. We used the bare `ctx`
   field: no activity label, and none of the reference preprocessing (`" [title]"` to `". "`,
   bracketed spans dropped, the resulting double spaces collapsed). The endings get that same
   preprocessing and did not. Note `str.capitalize()` lowercases the tail, which the obvious
   JavaScript one-liner does not.
2. **The `acc_norm` normalizer.** lm-eval divides the summed log-likelihood by the CHARACTER length
   of the choice (`completion_len = np.array([float(len(i)) for i in choices])`). We divided by
   token count. This one touches every task, not just HellaSwag, and per-token normalization is
   tokenizer-dependent, which is the thing character length exists to avoid.

Full sets, 0-shot, before and after, on the roleplay SFT (`rp-chat3`, the published
Minueza-3-95M-RP):

| Task          | acc_norm before | acc_norm after | acc before | acc after |
| :------------ | --------------: | -------------: | ---------: | --------: |
| PIQA          |           61.04 |          60.88 |      60.17 |     60.17 |
| ARC-Easy      |           39.90 |          41.04 |      45.03 |     45.03 |
| ARC-Challenge |           23.89 |          25.51 |      20.39 |     20.39 |
| HellaSwag     |           28.40 |          29.90 |      28.05 |     28.14 |
| **Index**     |        **9.81** |      **10.77** |          - |         - |

**The raw `acc` column is the check that the change did only what it should.** It is identical on
all three ARC/PIQA rows: sum-NLL ranking cannot be touched by swapping a normalizer, and those
items never changed. Only HellaSwag's `acc` moves (28.05 to 28.14), which is the query rebuild.

Scored against the published base with the fixed harness, which lever 9b could not do because the
base had never been run on all four tasks:

| Model                          |  PIQA | ARC-Easy | ARC-C | HellaSwag |     Index |
| :----------------------------- | ----: | -------: | ----: | --------: | --------: |
| Minueza-3-95M-Base             | 61.26 |    40.53 | 23.81 |     30.14 |     10.67 |
| stage 2, style SFT (`rp-full`) | 60.34 |    41.12 | 24.49 |     30.16 |     10.40 |
| Minueza-3-95M-RP (`rp-chat3`)  | 60.88 |    41.04 | 25.51 |     29.90 | **10.77** |

Two roleplay training stages moved the index by 0.10, which is the flat result the model card
claims. Omitting the ArithMark-3 term rather than assuming it at chance gives 12.98, 12.66 and
13.11.

**Do not read the dip.** The middle row is 0.27 below the base and 0.37 below the finished model,
and the battery does not resolve that. A 1-sigma binomial error on a single Index value is 0.71 on
this basis (PIQA +-1.14 pp on 1,838 items, ARC-E +-1.01 on 2,376, ARC-C +-1.26 on 1,172, HellaSwag
+-0.46 on 10,042), so the whole three-row spread fits inside one standard error of any one row. The
three checkpoints are scored on identical items, which makes this a paired comparison and the true
resolution better than 0.71, but recovering how much better needs per-item results the logs do not
keep. **The defensible claim is the one the card makes: the curriculum did not move general
capability. Ordering these three by index is not supported.**

**The original plan for this lever did not work, and the reason is worth keeping.** It was to score
`Felladrin/Minueza-32M-UltraChat` and compare against its real Open LLM Leaderboard v1 entry
(ARC-C 25-shot acc_norm 21.08 ±1.19, HellaSwag 10-shot 26.95 ±0.44, run 2024-03-01). All four rows
failed with `GGUF missing tokenizer.ggml.tokens/merges`: that GGUF carries a SentencePiece unigram
vocab, and `eval-choice` reads BPE only. An external validation still needs a model that is both on
a leaderboard we can read and shipped with a BPE GGUF in an architecture this loader supports.

**A process note that cost a wasted battery.** `eval-choice` runs out of the working tree, and a
`git checkout` during a multi-hour run swaps the code under the next task in the loop: a re-measure
of the stage 2 checkpoint reproduced the old numbers exactly, because the fix had been committed and
the branch switched away six seconds later. Long benchmarks run from a `git worktree` pinned to the
branch being measured, with absolute model paths, and the script greps the source for the change it
is supposed to be measuring before it starts. The stage 2 row above is that re-measure, run from a
pinned worktree with the grep guard; its raw `acc` matches the discarded battery on PIQA, ARC-Easy
and ARC-Challenge and differs only on HellaSwag, which is the signature the fix should leave.

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
teacher, `[T*k]` ids + probs per row; a row is ignored when its first id is exactly -1, the same
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

### 12. Picking between checkpoints: the training loss selects the wrong one (2026-08-21)

The roleplay SFT ran as two halves of one WSD schedule over the same 18.3M-token corpus: `rp-chat3`
stopped at step 550 (0.5 epochs), `rp-chat4` continued the same schedule to step 1100 (1.0 epoch),
same seed, same shapes. The training loss says the longer run is the better model. Everything else
says the opposite.

| signal                                | v3 (550 steps) | v4 (1100 steps)           |
| :------------------------------------ | :------------- | :------------------------ |
| training loss, mean of the last 10    | 2.726          | **2.681**                 |
| name given at turn 2, recalled turn 6 | yes            | no                        |
| persona held across 6 turns           | yes            | collapses to one template |
| greedy decode                         | no loop        | loops on 2 of 4 prompts   |

The four-task Intelligence Index is deliberately not in that table. Both checkpoints were scored on
it before the lm-eval parity fix and came out a fifth of a point apart, which is inside what a
benchmark this size resolves; re-scoring v4 on the current harness would cost hours of GPU to decide
nothing. The behavioral rows come from `--jinja` through the OpenAI chat endpoint with the full
history resent each turn, which is the only honest way to probe this (agents.md, the ChatML gotcha).

**The rule: at this scale a checkpoint is chosen by a behavioral probe, and the training loss is a
divergence alarm, not a selector.** The extra half epoch bought 0.045 nats on the training
distribution and paid for it in the behavior the model exists for. Nothing about that is surprising
in hindsight, which is the point: the loss curve gave no hint, and reading two transcripts settled
it in five minutes.

**What we could not do, and why.** The obvious instrument is held-out loss, and `eval-loss` already
supports it (`--holdout 1` against a separate `.tokens` file; the default 1% tail is corpus the run
has seen, which its help says outright). It cannot be applied retroactively here: v4 consumed the
whole corpus, so every token held out from v3 is training data for v4, and any split of `rpchat3`
is rigged toward v4. **A held-out slice has to be carved before training starts, or it does not
exist.**

For the next run the split is a jsonl step, not a code change: hold back the last few hundred
conversations before `chat-corpus`, tokenize them as a second corpus with the same `--tokenizer`,
and score every checkpoint with fixed `--windows` and `--seed`. `chat-corpus` has no split flag
(only `--max-rows`), so the split happens upstream of it.

**Two things this lever assumed and should not have.** First, that "every checkpoint" exists:
`finetune` overwrites its `--out` path on every write, so a finished run leaves exactly one file and
nothing to rank. `--keep-checkpoints` now also writes each one as `<out>-step<N>.gguf`, reusing the
bytes the export already produced. Without that flag this lever's whole prescription has no input.
Second, that carving the split before training is sufficient to make it held-out. It is not, if the
source corpus contains duplicates: the LittleLamb split was carved cleanly at document boundaries
and one of its 500 documents still appeared in the training half, because that conversation was in
the source twice and the boundary cannot see that. **Dedup against the training side after
splitting, not just split carefully** — one hash set over both sides is the whole check, and it
found a 2,720-token document worth 0.49% of that holdout.

Adopt the pairing, not either half alone: held-out loss to rank checkpoints cheaply and catch the
turn upward, a behavioral probe to confirm the one it picks. Neither the loss curve nor the
four-task benchmark separated these two by more than noise, and the loss curve pointed the wrong
way; six turns of transcript settled it in minutes.

### 13. Tightening the sampler makes a 95M model worse, not better (2026-08-20)

Prior going in: a tiny model has a garbage tail, so truncate it hard. The measurement says the
opposite, and the two settings on the RP model card come out of this table rather than out of
folklore.

14 presets, 2 scenarios (Iris the librarian, Thorn the cat), 2 seeds, all through
`llama-server --jinja` with the full history resent each turn, scored on `rp-chat3`. Metrics:
distinct-trigram ratio, longest repeated n-gram inside a reply, mean reply length, and whether the
model still knows the user's name at turn 6.

| preset                                    | distinct-3 | worst loop | mean len | name recall |
| :---------------------------------------- | ---------: | ---------: | -------: | ----------: |
| mirostat2 (t5, n0.1)                      |      0.992 |          0 |     33.6 |        0.75 |
| st-2026 (t1.0, min-p .05, DRY .8, XTC)    |      0.985 |          2 |     46.3 |        0.50 |
| dry-strong (t0.85, min-p .08, DRY 1.0)    |      0.973 |          1 |     34.2 |        0.50 |
| nsigma-dry (t1.0, ns1.0, DRY .8)          |      0.971 |          0 |     29.4 |        0.00 |
| topk-classic (t0.7, k40, p0.9, rep 1.1)   |      0.966 |          0 |     36.0 |        0.50 |
| card-current (t0.8, p0.9, min-p .05, 1.1) |      0.912 |        2.5 |     52.5 |        0.25 |
| minp-tight (t0.8, min-p .10)              |      0.589 |        6.5 |     52.9 |        0.75 |
| low-temp (t0.6, min-p .05, rep 1.05)      |      0.560 |        5.5 |     44.7 |        0.50 |
| minp-tighter (t0.7, min-p .15)            |      0.412 |        6.0 |     30.4 |        0.25 |

`min-p 0.15` at temp 0.7 produced 8-gram loops and a distinct-3 of 0.41: squeeze the distribution of
a 95M model and there is nothing left but its favourite phrase. **Every preset that avoided loops did
it with DRY, XTC, mirostat or top-n-sigma, not with truncation.** Zero presets leaked a user turn, so
that failure mode is gone from the chat stages.

**The metric leaderboard is a trap, and that is the more useful half of this.** Mirostat tops every
column and reads worst by eye ("Junius me!", "I love working here, and it gets me **working**!").
Highest-diversity and least-coherent are the same thing at this size, because distinct-3 rewards
exactly the invented-token garble a small model produces under pressure. A garble metric (the
tokenizer's own pieces-per-word, ~2.0 on ordinary English and ~4.0 on invented names like
"Sagittariroh") catches what distinct-3 pays for, and any future sweep needs it in the table.

By eye the two survivors are `dry-strong` and `topk-classic`, which is what the model card ships as
its recommended and alternative presets. A round-2 grid of 11 presets around those was written but
never run; it is not what the card's numbers rest on.

### 14. Four scenarios the RP battery was missing, and whether `-ngl` changes its answers (2026-08-22)

`eval-rp-completions.sh` has existed since the roleplay corpus tools landed, with five prompts shaped
like a real horde request and `-r "You:"` as the stop, so a model that writes the human's turn shows up
as truncation. What it did not have was a scenario per failure mode, which is what a checkpoint
comparison needs: without one, three LittleLamb checkpoints all "look fine" and nothing names what
differs.

Four added, each isolating one thing: holding a stated refusal, recalling a fact given three turns up,
a card dense with proper nouns, and two characters in one scene. Their personas are invented rather
than lifted from PIPPA, so a model that memorized a popular card cannot score on recall instead of
adherence. On step 881 the two new hard ones fail visibly and differently: the gate sergeant emits
three consecutive `Sergeant Idris Vale:` labels instead of one turn, and the barge scene switches from
`Captain Rook:` to `Rook:` and then writes Pell's line. Neither is visible in the original five.

**Whether `-ngl` changes the answer: measured, and on this hardware it does not.** The concern was that
GPU offload produces different completions from CPU at temp 0. On Strix Halo (gfx1151, Vulkan backend,
llama.cpp b7682), across 26 deterministic completions (the 10-prompt base battery on checkpoints 481
and 881, plus a 6-scenario RP set run at the temp-0 override):

| Comparison                                          | Result                                |
| --------------------------------------------------- | ------------------------------------- |
| `-ngl 0` vs `-ngl 1` vs `-ngl 10`                   | byte-identical                        |
| `-ngl 0` vs `-ngl 99`, base battery, 20 completions | byte-identical                        |
| `-ngl 0` vs `-ngl 99`, RP set, 6 completions        | 1 token differs ("That's" vs "Good.") |
| `-ngl 99` run twice                                 | byte-identical                        |

Device memory scales with the flag (472 / 731 / 1295 MiB), so the partial offloads are real and not
silently falling back to CPU. Each setting is deterministic; the single divergence is a float-order
argmax flip on a near-tie, not degradation, and neither completion is worse than the other. So the
backend is not a correctness problem here, but a checkpoint comparison decided by eye can turn on one
token: record which backend and which build a result came from. `NGL` is now a tunable and the script
prints it in the header for that reason.

Untested, and the reason the flag is exposed rather than pinned: the original report of incoherent
offloaded output was on an M1 Max, which is the Metal backend, not Vulkan. Nothing above transfers to
it.

### 15. The card's own sampler preset was the worst of four (2026-08-22)

Two questions, one measurement. First, whether `llama-server` mishandles the model: it does not.
With `-ngl 999` on the Q8_0 file, `llama-completion` from the same build at the same seed and
parameters returns byte-identical text. The suspicion was reasonable and it is ruled out.

Second, whether the preset the card recommended was the right one. It was not. Four presets, four
personas by twenty seeds each, through `/completion` with `You:` as the stop:

| Preset                                                    | Mean tokens | Replies under 15 tokens |
| --------------------------------------------------------- | ----------- | ----------------------- |
| card-dry: `temp 0.85 min-p 0.08 top-k 0` + DRY 1.0/1.75/2 | 35.2        | 28% +/- 5%              |
| card-topk: `temp 0.7 top-k 40 top-p 0.9 rep 1.1/128`      | 58.5        | 16% +/- 4%              |
| warm-minp: `temp 1.0 min-p 0.10 rep 1.05/64`              | 50.7        | 12% +/- 4%              |
| cool-topk: `temp 0.6 top-k 30 top-p 0.9 rep 1.1/128`      | 61.5        | **8% +/- 3%**           |

The card led with card-dry, which collapses most often: a reply that runs eleven tokens and hands the
turn straight back is a real roleplay failure, not a metric artifact. Every DRY variant tried landed
at 28-34%; every non-DRY one at 8-16%. The gap from worst to best is 20 points against a combined
error of about 6. The card now leads with cool-topk.

**This does not overturn lever 13, it measures a different axis.** That sweep asked which presets
avoid loops and garble, and DRY genuinely does. This one asks how often a preset produces a collapsed
reply on transcript-shaped prompts with a stop sequence, which the earlier sweep never posed.

**What no preset fixed.** Reading the text rather than the counts: a persona written as a male cat
comes back as "her" under every setting, and a persona written as reticent and uncomfortable with
praise gushes under every setting. One preset had the model address the user as "my dear apprentice"
when the card makes the character the apprentice. Persona adherence at 95M is a capacity limit and
the sampler does not reach it; the card's Limitations say so now.

Unprompted adult drift from these SFW prompts was rare, 0 to 3 hits per 80 depending on preset, which
at that sample size does not separate the presets. It is not zero: one sample turned "That sword you
made is beautiful" into "just a tool for your pleasure".

### 16. Held-out loss rose during the cooldown, and a slope through a random walk predicted nothing (2026-08-23; the cause is settled in 16b)

The LittleLamb RP fine-tune is the first run with the instrument lever 12 asked for: a held-out
slice carved before training, `--keep-checkpoints` writing a snapshot series, and 34 rows in
`out/holdout-200.log` (base, 32 checkpoints, the final file) all scored on the same 64 windows of
1024 tokens at seed 1234. Every comparison below is therefore paired, which is what makes
checkpoint-to-checkpoint jitter the right error scale rather than the standard error of a
64-window estimate.

**Held-out loss rose during the cooldown.** The schedule is 220 warmup, stable to 1760, cooldown
over the last 440 steps.

| Phase                            | Held-out loss     |
| -------------------------------- | ----------------- |
| plateau, steps 762-1400 (n=11)   | 2.8888, sd 0.0062 |
| plateau, steps 1600-1680 (n=2)   | 2.8594, sd 0.0001 |
| cooldown, steps 1767-2200 (n=10) | 2.8730, sd 0.0072 |
| final checkpoint, step 2200      | 2.8683            |

The release is step 1680 at 2.8594. Note the phase boundary: step 1767 is 7 steps _inside_ the
cooldown, so it belongs in the third row and not, as an earlier draft of this lever had it, in the
plateau above it.

**That heading is deliberately weaker than "the cooldown made it worse", because the data does not
support the stronger claim.** Phase is perfectly confounded with time and there is no constant-LR
continuation to compare against. Worse, this same curve takes an unexplained 0.025 excursion
elsewhere (below), so the honest noise scale for a phase-sized claim is the size of the excursions
the series makes for no reason anyone can name, not the 0.0066 mean gap between the 32
checkpoints' 31 adjacent pairs. By that standard the cooldown's 0.0135 is _smaller_ than something already conceded as
unexplained. There is even a reading in which the cooldown helped: driving the LR toward its floor
freezes the trajectory, and what looks like degradation is the walk being stopped somewhere
mediocre instead of wandering back. **The counterfactual is one cheap experiment, with one trap in it.** The seed is
`mulberry32(7 + startStep)`, so a resume draws a different data stream than the original tail did:
comparing a fresh flat-LR resume against the original cooldown would confound the schedule with the
batch order. Run _both_ arms as resumes from the same step so both get the same stream and only the
schedule differs. There is also no flat-LR flag, since warmup and cooldown are hardcoded at 10% and
20% of `--steps`, so the flat arm has to be built by choosing a `--steps` whose cooldown falls
outside the segment. Until that runs, this is an association.

**It ran the next day, and the association was spurious: see 16b.** Both hedges above were the
right ones to keep. The reading in which the cooldown helped is the reading that survived.

One thing that is free and confound-independent, though, and worth stating because the paragraph
above understates the evidence: all ten cooldown snapshots score worse than both pre-cooldown flat
points. Ten for ten is not nothing, even if it cannot separate the schedule from time. It turned
out to be nothing: ten for ten against a baseline that does not exist yet is ten for ten against
time, and 16b built the baseline.

**The curve is a staircase, and two extrapolations through it were both wrong.** Mid-run, an OLS
tail fit projected roughly 2.86 at step 2200 and the argument was that the remaining hours bought
almost nothing. Then the curve stepped down across 1400-1600, a refit projected roughly 2.83, and
the argument reversed to "let it run". Both were wrong, and the projections are quoted to two
figures here because neither reproduces exactly from the checked-in log: they were computed against
partial series while the run was still going. **Scoring two more snapshots settled what no fit
could**: 1600, 1680 and 1767 came in at 2.8595, 2.8594 and 2.8600, three flat points spanning the
cooldown boundary, which says the descent had already stopped.

**The step is unexplained, and the most likely cause is that there is no cause.** The move across
1400-1600 is 0.0247 endpoint to endpoint, 0.0294 between the two plateau means. It is not the LR
schedule: it happened ~300 steps inside the stable phase. It is not a data-mix change: the run
passed no injection flags, and `injectFromStep` in `train-gpu.ts` is the only mechanism that would
alter the mix. The trainer's own banner says "0.5 epochs of 17M", which is the token ratio and not
a claim that the sampler walks the corpus in order. It
is **not an epoch boundary, and that ruling-out was originally argued the wrong way**: the trainer
draws every window as `Math.floor(rng() * maxStart)`, i.i.d. with replacement per sequence
(`train-gpu.ts`, which is the path this run took, and identically in `trainer.ts`), so there are no
epochs at any step count, and the arithmetic about 4101 steps per
epoch was answering a question the sampler never poses. What that sampler does imply is the fourth
candidate: with 2 sequences per step under assistant-only masking, the weights random-walk inside a
low-loss basin, and plateau-step-plateau is what a random walk looks like sampled at 32 points.

**The behavioural battery was scored by a broken metric twice, and both breaks flattered the
conclusion in front of them.** This is the part of the run worth carrying forward, more than any
number above it.

The scorer began with two buckets, the character and everyone else. The battery's two-character
prompt names a deckhand called Pell in the persona and asks for both voices, so a `Pell:` line
counted as a stray label. That scored a checkpoint higher for **ignoring the character it was told
to write**, and the released step 1680 is exactly such a checkpoint, so the battery appeared to
endorse it. Splitting staged speakers out of "everyone else" reversed that: 1680 became the worst
of 18.

The split was also wrong. It filed `You:` under staged-and-therefore-correct, when the model
writing the human's turn is the truncation the whole battery exists to expose, and `You:` was 76%
of that bucket, 825 of 1,092 labels across 216 runs. So the second reading reversed again. Each
time, the bucket that merged a defect with correct behaviour was the one carrying the conclusion.

Four buckets survive scrutiny: `handback` (`You:`), `self` (a consecutive re-label, not an
alternation), `costar` (a speaker the transcript staged), `invented`. Over 18 checkpoint means:

| Count    | t (18) | t without step 1680 | within-ckpt SD | between-ckpt SD |
| -------- | ------ | ------------------- | -------------- | --------------- |
| handback | -2.87  | -2.30               | 1.20           | 0.69            |
| self     | +1.46  | +1.43               | 0.97           | 0.30            |
| costar   | -0.71  | **+0.89**           | 1.17           | 0.46            |
| invented | +1.34  | +1.74               | 1.09           | 0.45            |

**Read the last two columns first.** Within-checkpoint spread exceeds between-checkpoint spread on
every count, which is this instrument saying it cannot separate these checkpoints. `costar` also
flips sign when one point is dropped: the battery has nothing between step 1281 and step 1680, so
1680 carries 35% of the leverage in any regression across it. The one count that survives both
checks is `handback`, and it declines, meaning later checkpoints stop before writing the human's
turn. That is an improvement, and it is the opposite of what the first two scorers said.

**None of this ranks checkpoints here, and an earlier claim of t=-3.02 on a combined other-speaker
count is retracted rather than re-derived.** That number cannot be checked against anything: it came
from a scorer that lived in the session scratchpad and is gone. The battery also has nothing past
step 1680, so it could not speak to the cooldown even if the metric were sound.

`scripts/score-rp-battery.ts` and `scripts/dump-rp-prompts.sh` are in the repo now, with
`tests/rp-battery-score.ts` pinning all four distinctions so neither merge can come back silently,
and `docs/measurements/` carries the logs and per-run counts every number above is read from.

**What survived all of this.** Training loss picked differently again: over the 101 logged steps
the per-batch trace ranges 1.52 to 3.22, with its minimum at step 1782 and its maximum at 1914,
both inside the cooldown and neither related to the held-out curve. The ranking half of lever 12's
prescription did the work; the behavioural half needed its instrument repaired before it was worth
reading.

### 16b. The counterfactual: the cooldown was not the cause, and the rise did not reproduce (2026-08-24)

Lever 16 left the cooldown as an association and named the experiment that would settle it. The
experiment is `scripts/counterfactual-cooldown.sh`, it cost about ten and a half GPU-hours, and it
came back against the hypothesis twice over.

**The design, and why the flat arm looks strange.** Both arms resume `out/lamb-ckpt-1767.gguf`. The
window sampler is seeded `mulberry32(7 + startStep)`, so resuming both at the same step is what buys
an identical data stream; only the schedule then differs. Warmup and cooldown are hardcoded at 10%
and 20% of `--steps`, so there is no flat-LR flag and the flat arm has to be constructed:

|                             | arm A, cooldown       | arm B, flat          |
| :-------------------------- | :-------------------- | :------------------- |
| `--steps`                   | 2200                  | 2760                 |
| WSD cooldown per the header | 440 steps, from 1760  | 552 steps, from 2208 |
| LR scale across 1767-2167   | 0.9836 down to 0.1000 | flat at 1.0          |
| stopped at                  | 2200, ran out         | 2167, killed         |

Arm B's cooldown is real but starts 41 steps past where the arm is killed, which is the whole trick.
Both headers are in `docs/measurements/`, and both logged the same parity probe (CPU 3.3022 vs GPU
3.3022), which is the check that they started from identical weights.

**The trap that nearly cost a night.** `checkpointEvery` keys on the step local to the segment
(`train-gpu.ts`), while the filename carries the global `startStep + localStep`. The obvious stop
condition, waiting for `cf-flat-step2200.gguf`, waits on local step 433, which is prime and
therefore never a multiple of any cadence. That file is never written and the loop never exits.
The arms stop at 2167, local step 400, for this reason and no other.

**Result: the cooldown arm is better at every matched step.**

| Global step | arm A, cooldown | arm B, flat |   A - B |
| ----------: | --------------: | ----------: | ------: |
|        1817 |          2.8591 |      2.8595 | -0.0004 |
|        1867 |          2.8527 |      2.8545 | -0.0018 |
|        1917 |          2.8556 |      2.8595 | -0.0039 |
|        1967 |          2.8531 |      2.8611 | -0.0080 |
|        2017 |          2.8519 |      2.8552 | -0.0033 |
|        2067 |          2.8629 |      2.8744 | -0.0115 |
|        2117 |          2.8553 |      2.8581 | -0.0028 |
|        2167 |          2.8566 |      2.8595 | -0.0029 |

Eight of eight negative. Mean -0.0043, paired t = -3.37 on 7 df, and the sign test alone gives
p = 0.0078 without assuming anything about the distribution. Lever 16 was right to refuse the
stronger claim, and the direction it floated as a possibility ("driving the LR toward its floor
freezes the trajectory") is the direction the data took.

Read the magnitude with the same suspicion lever 16 applied to its own: -0.0043 is smaller than the
0.0066 adjacent-checkpoint jitter. Only the pairing makes it visible, and nothing about checkpoint
selection changes because of it.

**The finding that actually settles lever 16: neither arm reproduced the rise.**

| Series, steps 1817-2167      | Mean held-out loss |
| :--------------------------- | -----------------: |
| the original run             |             2.8752 |
| arm A, the same schedule     |             2.8559 |
| arm B, flat                  |             2.8602 |
| the shared origin, step 1767 |             2.8600 |

Arm A ran the identical schedule from identical weights and stayed flat across a stretch where the
original climbed 0.015 nats. A cause that does not reproduce when you re-run it is not the cause. So
the fourth candidate in lever 16 is the one left standing: the weights random-walk in a low-loss
basin, and the excursion the original took during steps 1800-2200 is the same kind of event as the
unexplained 0.025 excursion across 1400-1600, which also sits nowhere near a phase boundary.

**What arm A does not share with the original tail**, and therefore what a follow-up would have to
separate: no optimizer state was kept at step 1767, so both arms cold-start Muon where the original
carried momentum through, and `mulberry32(7 + 1767)` draws different windows than the original run's
`mulberry32(7 + 0)` reached by that point. Either is enough to move a random walk this far. The
cold start is symmetric across the arms, so it does not touch the A-versus-B comparison; it only
means arm A is not a byte reproduction of the original tail, which is exactly why it is evidence
about schedules and not about that specific trajectory.

**One observation, recorded rather than acted on.** Arm A's snapshots run 2.8519 to 2.8629 on the
same holdout at the same knobs, and several of them beat the released step-1680 checkpoint at
2.8594; the best is 0.0075 lower, about one jitter unit, from a lineage with a cold-started
optimizer. That is not a reason to re-release anything. It is a reason to suspect that restarting
the optimizer mid-run is worth its own experiment, which this one cannot answer because both arms
did it.

**What this changes about how to read a loss curve here.** Lever 12 established that training loss
picks the wrong checkpoint. This adds the weaker-looking but more expensive lesson: a held-out curve
that turns at a phase boundary is not evidence about the phase either. Anything on this setup that
moves less than the excursions the series makes for no reason needs a shared-checkpoint
counterfactual before it becomes a claim, and running one costs a night.

### 17. The `llama` architecture rotated the wrong dimension pairs, and nothing in the suite noticed (2026-08-25)

Fine-tuning SmolLM2-135M-Instruct-heretic opened with a masked training loss of 6.44 and a held-out
loss of 6.18. For a 2T-token instruct model on ordinary chat data those numbers are not plausible,
and the corpus was the obvious suspect. It was not the corpus.

**The measurement that named it.** A file of `The cat sat on the mat.` repeated 400 times, scored
by both engines on the same GGUF:

| Engine                       | Perplexity |
| ---------------------------- | ---------- |
| llama.cpp `llama-perplexity` | 1.006      |
| this trainer, `eval-loss`    | 5.73       |

A model that cannot predict the 399th repetition of a sentence it has already seen 398 times is not
reading its own context. Greedy continuation says the same thing in one line: from
`The cat sat on the ...`, llama.cpp continues `mat. The cat sat on the mat.` and this engine
continued `floor. The cat was sitting on the floor.` Locally fluent, no memory.

**The cause.** llama.cpp maps `LLM_ARCH_LLAMA` to `LLAMA_ROPE_TYPE_NORM`, which rotates dimension
pairs (2j, 2j+1). `rope()` here rotates (j, j+headDim/2), the NeoX convention. Both are the same
rotation over a different row order, and `conversion/llama.py` in llama.cpp does exactly this
reorder when it imports from Hugging Face, whose `LlamaAttention` uses the half-split form. So a
converted checkpoint arrives in the interleaved order and this engine rotated it as though it were
half-split. `src/arch/llama.ts` now reorders Q and K on load and back on export;
`tests/llama-rope-layout.ts` pins the convention. The reorder lives in llama.cpp's converter too, as
`permute()` in `conversion/llama.py`, or as `LlamaModel.modify_tensors` in `convert_hf_to_gguf.py`
on builds from before that file was split up.

**Any `llama` checkpoint written here before the fix is now misread.** It is stored in the old row
order and the loader applies the inverse permutation to it, so a resume runs and the loss jumps
rather than failing. To carry one over, load it with the pre-fix code and export it again with this
one. Nothing published is affected: `--arch llama` never carried a released checkpoint.

Confirmation, same holdout and same knobs, before and after:

| Measurement                              | Before | After |
| ---------------------------------------- | ------ | ----- |
| repeated-sentence perplexity             | 5.73   | 1.09  |
| held-out loss, base checkpoint           | 6.18   | 2.73  |
| training loss, first steps of the RP run | ~6.4   | ~2.4  |

Greedy continuation now matches llama.cpp token for token on the repetition prompt.

**Only `llama` was affected, and that is why it survived.** `gemma3` and `qwen3` are both NeoX in
llama.cpp, so the shared `rope()` was already right for them, and every published model from this
repo is one of those two: Minueza-3 is gemma3, LittleLamb is qwen3. `--arch llama` had never carried
a released checkpoint.

**What no test could see, and this is the part worth carrying forward.** The suite is built out of
self-consistency checks, and this defect is self-consistent. `gpu-parity` compares the WebGPU
backend against the CPU reference, and both share `rope()`, so they agreed exactly (the run banner
printed |Δ|=0.0e+0 while the model was reading its context wrong). `gradcheck` compares analytic
gradients against finite differences of the same forward pass, so a wrong-but-differentiable
rotation checks out. `arch-roundtrip` exports and re-imports through this engine, and a reorder
that is missing on both sides round-trips perfectly. Even the export contract held: llama.cpp
loaded the file, reported the right shape, and generated readable text.

Every one of those instruments compares the project against itself. None of them compares it
against llama.cpp, which is the actual contract for a GGUF trainer, and the defect lived exactly in
the gap. The new test pins the byte order against llama.cpp's convention explicitly rather than
asserting a round trip, because a reorder that is merely self-consistent passes a round trip and
still disagrees with the reference.

The cheap version of that check is the one that found it: take a published checkpoint, score the
same file with both engines, and compare. It costs a minute per architecture and it is now the
thing to run first when a loss looks wrong.

### 18. One corpus for both llama.cpp endpoints, and the mask that stops a model writing the human's turn (2026-08-25)

Every roleplay model in this series has been used two ways and trained for one. SillyTavern and
Kobold Lite send a raw persona transcript to `/completions`; an OpenAI-shaped client sends ChatML to
`/v1/chat/completions`. Minueza-3-95M-RP and LittleLamb-293M-RP were both ChatML fine-tunes, and
LittleLamb's card says so outright: "that format is not what this model was trained on".

**The measurement that motivated it.** `SmolLM2-135M-Instruct-heretic`, unmodified, on six raw
transcript scenarios with no stop string, at temp 0.6 / top-k 30 / repeat-penalty 1.1:

| Count over six raw scenarios         | base |
| ------------------------------------ | ---- |
| wrote the human's turn               | 5    |
| prefixed its reply with its own name | 6    |
| stopped on EOS                       | 1    |

A base instruct model handed a transcript plays every part in it. The chat endpoint was already
clean on the same six personas, so the whole problem is one format.

**The corpus is one file with two renders, and the difference is where the mask goes.** The ChatML
half is `assistantLossMask` as before. The transcript half (`src/data/transcript.ts`) renders a
`[Character: X]` header, a persona line, `<START>`, then labelled turns, and supervises **exactly one
reply**: everything up to the final `Name:` is context, the reply after it is the target, and an
appended `<|im_end|>` closes it. The human's turns are mask 0 at every offset, so no gradient ever
teaches the model to produce them.

Three details that are the whole design, and each one was a choice with an alternative:

- **One supervised reply per document, not all of them.** Supervising every character turn is more
  token-efficient, but then the stop signal appears once per document against six replies, and the
  model learns to continue rather than to stop. One reply per document makes the ratio 1:1.
- **The terminator is `<|im_end|>`, not a `You:` handback.** Training the model to emit the next
  speaker label is the horde convention and it works, with a client that sets a stop string. It also
  trains the exact behaviour being removed. Ending on EOS stops generation with any client and with
  no configuration, and it means the token `You:` never appears as a target anywhere in the corpus.
- **The human's label varies.** Seven of twelve slots are `You`, the rest are other names. A model
  trained on one literal string learns the string; the point is the shape "a name, then a colon", so
  a SillyTavern persona called anything else still works.

Byte offsets rather than token search: `BPETokenizer.byteLengths` gives each token's contribution to
`decode()`, so the reply's byte span maps onto token indices in one pass, and a token straddling the
boundary drops from the mask rather than half-supervising the model's own speaker label.

**Result, same six scenarios and the same sampler, 1300 steps and 10.6M tokens later:**

| Count over six raw scenarios         | base | fine-tune |
| ------------------------------------ | ---- | --------- |
| wrote the human's turn               | 5    | 1         |
| prefixed its reply with its own name | 6    | 0         |
| stopped on EOS                       | 1    | 6         |

Chat stayed at 0 and 0 on both. Six scenarios at one sampled seed is a coarse instrument and the
right reading is "gone from these six", not a rate.

**The scorer was wrong first, in the way lever 16 warned about.** The "stopped on EOS" column read
`res.stopped_eos`, a field this llama.cpp build does not send; it reports `stop_type: "eos"`. An
absent field is `undefined`, `undefined` is falsy, so the column read 0 for every model. **It read 0
for the base too, and that was true**, which is exactly why nobody looked: the broken metric agreed
with the control. It was caught by hand-reading a completion that had obviously terminated while the
table said it had not. `stoppedOnEos` now throws when neither field is present, because "the field is
missing" and "the model never stopped" are different facts and only one of them is a zero.

**What the battery still cannot see, and it is the thing users will notice first.** At temp 0 with no
repetition penalty, six of the twelve completions were repetition loops running to the token cap; the
gate sergeant said a variation of "I'm not ready for that." fourteen times in one 120-token reply.
Under the card's preset all twelve terminated. So the deterministic battery is right for comparing
checkpoints and actively misleading about how the model reads, which is why `eval-endpoints.ts` grew
a `--sampler` flag and the measurements directory carries both runs. Do not put them in one table.

**Checkpoint selection had nothing to select.** Held-out loss fell monotonically from 2.7647 to
2.6416 with no turn, and the last four snapshots span 0.0045 against a mean adjacent gap of 0.0060.
The battery broke the tie (three EOS stops at step 1300 against two, one and two) and that is also
coarse. Both pointed at the last checkpoint, which is also the one whose optimizer state exists on
disk, so it shipped. Worth being plain that this is two weak instruments agreeing, not one strong one
deciding.

**Q4_0 is worse than the model it was fine-tuned from.** F32 2.6416, Q8_0 2.6439, Q4_0 2.9063,
against a base of 2.7647. The 293M sibling paid 0.1161 for Q4_0; this paid 0.2647 and gave back more
than the fine-tune bought. Embeddings are tied at this size, so the embedding matrix doubles as the
output projection and 4-bit lands directly on the logits, which is the likely mechanism and is not
tested here. The file is not published. A 78 MB download whose only distinguishing property is being
worse than its own base is a trap for whoever sorts by size.

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

- Held-out eval per epoch, early stopping on it, and "loss != correctness" as standard SFT practice
  (a QLoRA/PEFT pipeline, so only the evaluation discipline transfers to this project):
  [towardsdatascience.com](https://towardsdatascience.com/how-to-fine-tune-an-llm-an-end-to-end-guide/)
- lm-evaluation-harness, the reference for the query construction and the acc_norm normalizer
  `eval-choice` matches: [github.com](https://github.com/EleutherAI/lm-evaluation-harness)
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
- Fused linear-cross-entropy prior art (CUDA/Triton; the shape lever 19 follows):
  [github.com](https://github.com/linkedin/Liger-Kernel), [github.com](https://github.com/mgmalek/efficient_cross_entropy)
- FlashAttention-3 (what full 2D tiling + tensor cores buys on NVIDIA; context for why the same
  structure is not automatically fast on a no-TC, no-subgroup WebGPU path):
  [arxiv.org](https://arxiv.org/abs/2407.08608)
