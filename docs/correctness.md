# Correctness

What this trainer checks, what each check was written after, and where to put a new one. Every
guard below exists because something got past its absence, and most of those failures produced a
plausible number rather than a crash.

Read [performance.md](performance.md) for throughput and memory, [design.md](design.md) for why the
engine is shaped this way.

## The four instruments

`deno task test` runs all of them. None takes longer than a minute.

| Instrument                | What it proves                                                                                                                                                                             |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/gradcheck.ts`      | every autograd op's analytic gradient matches a central finite difference, with a negative control that has a deliberately wrong backward, so the harness is known to reject bad gradients |
| `tests/gpu-parity.ts`     | every WGSL kernel's forward and backward match the reference implementation within tolerance, plus the device-behaviour gates below                                                        |
| `tests/kernel-limits.ts`  | no emitted kernel declares more workgroup storage than the 16 KiB WebGPU floor, at any head dim the trainer accepts                                                                        |
| `tests/arch-roundtrip.ts` | every registered architecture exports to GGUF and re-imports to the same model                                                                                                             |

The reference implementation in `src/model/autograd.ts` exists for the first two. It is not a
training path and is not exposed as one: it is the oracle the kernels are held against, and
`gpu-parity` runs it step for step against the GPU trainer to prove their trajectories match.

**Every one of those instruments compares the project against itself.** That is the gap the `llama` RoPE defect lived in: `src/arch/llama.ts` rotated the wrong RoPE dimension pairs, and parity agreed exactly
(|Δ|=0.0e+0 while the model was reading its context wrong), gradcheck passed a
wrong-but-differentiable rotation, and a round trip that is missing a reorder on both sides is
perfect. What caught it was scoring one file with both engines:

| Engine                       | Perplexity on a 400x repeated sentence |
| :--------------------------- | -------------------------------------: |
| llama.cpp `llama-perplexity` |                                  1.006 |
| this trainer, `eval-loss`    |                                   5.73 |

A model that cannot predict the 399th repetition of a sentence it has seen 398 times is not reading
its own context. **Run that check first when a loss looks wrong**; it costs a minute per
architecture. `scripts/cross-engine-check.sh` is the scripted form, and `tests/llama-rope-layout.ts`
now pins the byte order against llama.cpp's convention explicitly rather than asserting a round
trip.

## Prove a test can fail

Break the thing a new test guards, watch it go red, put it back. Three tests written in one day
passed for reasons unrelated to what they claimed:

- **A NaN oracle.** `!isFinite(got) || Math.abs(got - want) > tol` reads as symmetric and is not:
  with `want` NaN, `Math.abs(0 - NaN)` is NaN and `NaN > tol` is false, so a broken reference side
  passes silently.
- **A pass condition the absence of work also satisfies.** "The loss is 0 on a fully masked batch"
  passes just as well when the readback never happened, because an unread host tensor is zeros. It
  needs a scored control in the same `sync`, and a NaN seeded into every scalar first.
- **Two sets that never intersect.** A gate meant to catch a later pass overwriting an earlier
  one's buffer allocated them so the writes and reads landed on different halves of the pool.
  Running the backward is what made the buffers overlap.

The question is not "does this pass" but "what would have to be true for this to fail, and have I
made that happen". If the answer is "nothing I can do from inside this repo", say so in the test.

## Validate above the backend dispatch

A guard that lives inside a backend is a guard the other backend does not have. Every input check in
`src/model/autograd.ts` runs before the dispatch, and moving one below it is a change the tests
catch. Four defects came from that placement, all of the same shape: **an out-of-range index is not
a crash, it is a believable number.**

| Defect                                       | Reference                                | GPU                                    |
| :------------------------------------------- | :--------------------------------------- | :------------------------------------- |
| embedding id past the table                  | `[NaN, NaN, NaN]`                        | `[0, 0, 0]`, run continues             |
| cross-entropy target past the vocab          | next row's logit, or NaN on the last row | next row's logit, finite and plausible |
| `softCrossEntropy` teacher id past the vocab | checked                                  | unchecked, indexed whatever was bound  |
| `-2` as an ignore marker                     | row skipped                              | `0xfffffffe` scored as a huge target   |

Scoring a vocab-49152 checkpoint against a corpus tokenized with a 151936-entry vocab used to return
`val loss 10.9754  ppl 58420.99` on the GPU. It now stops at
`embedding: id 49751 at position 20 is not an integer in [0,49152)`. That message arrives one
position before the loss's own `target 49751 at position 19`, and the order is the point: the inputs
reach the embedding table before the targets reach the loss, so the same bad corpus is caught a step
earlier. `tests/gradcheck.ts` pins both the position and the `embedding:` prefix for that reason.

The GPU is the worse of the two on the last row, and WGSL's robust buffer access is not the safety
net it looks like: the logits buffer is bound whole, so reading the next row is perfectly in bounds.
Past the end of the buffer, WGSL promises some in-bounds value of its choosing, not a particular
one, and pooled buffers come back dirty, so the digits move with pool state.

Three rules the validators follow, each pinned by its own case:

- The ignore marker is exactly `-1`, never `< 0`. `uploadU32` maps `-1` to `0xffffffff`, which the
  kernels test for; `-2` becomes a huge id.
- Lengths are checked, not assumed. The losses sum `T` rows and divide by a count, so a longer
  targets array inflates the denominator.
- Accepting cases carry an oracle computed in the test, never a comparison between the two backends:
  both draw their denominator from the same helper and would agree on a wrong count.

## Catch it when the file opens

`assertCorpusFitsVocab` scans the region a command will actually read, at 310M tokens/s, so a
corpus/vocab mismatch fails at startup rather than on whichever window happens to hold the bad id.
A range check is not enough on its own, though: a stale `.tokens` file whose ids are all legal
passes every range check, and a 4-byte file read as 2-byte passes the size check too, doubling the
token count and splitting every id in half. So **a token file carries a fingerprint of the tokenizer
that produced it**, and a file that predates the stamp says so rather than claiming to be checked.

`eval-choice` refuses an unscoreable item before the first forward for the same reason: a candidate
at least as long as the model's whole declared context, or a stem that rendered to nothing, cannot
produce a comparable score, and a shortened score silently wins.

## Numerical forms that hide a defect

- **Never read a probability back to compute a loss.** `-log(p_target + 1e-12)` saturates at 27.63
  once the target falls ~88 logits behind the row maximum, so every confident-wrong loss reported
  the same number. All four losses now use `log(Σ exp(z - m)) + m - z_target`. Gradients were never
  affected; the backward uses normalized probabilities, where underflow to zero is the correct limit.
- **A mask boundary comes from the window the model sees**, not from the untruncated input.
  `eval-choice` took it from the pre-truncation context length, so a partially truncated choice
  scored only its tail while `acc_norm` kept dividing by the whole choice's character count, and a
  fully truncated one returned exactly 0, which beats every real summed NLL.
- **Shapes are checked in both directions at the GGUF boundary.** The writer destructured a 1-D
  tensor into `[outDim, inDim]` and got `undefined` for the second; the reader never looked at
  `t.dims` at all.

## Device behaviour the suite pins

These are gates in `tests/gpu-parity.ts` that count or measure rather than compare numbers, because
the defects they cover leave every number correct:

- **Forward-only work stages nothing.** Both eval commands and `generate` used to allocate a full
  model of gradient accumulators per window and copy them to the host, never having run a backward:
  1.17 GB per window on a 293M checkpoint. `freezeForScoring` before the first `entryFor` call
  removed 18.8 GB of copies from a 64-window eval (14.8% faster) and 46.9 GB from a 40-token
  generation (39.0% faster). Ordering is load-bearing, so the gate asserts both the readback and the
  pool: freezing after `uploadParams` stops the copies but keeps the allocation.
- **Recompute does not leak pool.** A region buffer that never returns to the pool leaves every
  number right and quietly allocates around it. The gate asserts structurally, one claim per drain:
  with reclaim off the pool must not grow with step count, with reclaim on it must not grow with
  micro-batch count. Each drain was deleted to confirm its arm fails.
- **Buffer recycling is safe because of queue ordering**, not because of any fence. The comment that
  claimed otherwise was wrong, and the gate that checked it was asserting something else entirely
  until a backward was added to make the buffers actually overlap.
- **An all-ignored batch runs through the device losses** without producing NaN.

## Writing a large file

`fs.writeFileSync(path, data)` does not survive a buffer longer than 2^31 bytes on Deno 2.9.1:
instead of failing it writes without bound. A 2.39 GB export filled a 1.9 TB disk twice, reaching
1.19 TB before it was killed. Every large writer goes through `writeFileBytes`, which writes through
an open handle in 1 GiB spans.

This is not exotic: an f32 GGUF passes 2^31 bytes at ~537M parameters, so two rows of the readme's
own base-model table could not be exported at all, and the optimizer sidecar hits it first (1504 MB
for a 293M model). Reported upstream as
[denoland/deno#36810](https://github.com/denoland/deno/issues/36810); native `Deno.writeFileSync`
handles the same buffer correctly, so it is specific to the `node:fs` layer.

`tests/large-file-write.ts` checks the span arithmetic exhaustively and does a real multi-chunk
round trip at a small size. The genuine 4.10 GiB write is behind `GGUF_TRAINER_BIG_IO=1`, in a child
process under `ulimit -f`, because a regression there does not fail an assertion, it runs away.

## Known limits of the suite

- **A resumed segment runs ~15% slow for its first ~750 steps**, reproducibly and to the same local
  step across two independent resumes, then steps onto the same plateau. Cause not identified;
  ruled out are the LR schedule, memory pressure, thermals, competing processes, transparent
  hugepages and V8 heap growth. The operational consequence is that every restart costs ~21 minutes
  of throughput on top of the steps it loses.
- **A `llama` checkpoint written before the RoPE fix is misread today.** It is stored in the old row
  order and the loader applies the inverse permutation, so a resume runs and the loss jumps rather
  than failing. Nothing published is affected; `--arch llama` never carried a released checkpoint.
- **The parity probe at startup reads the first 16 tokens only**, so a corpus defect deeper in the
  file surfaces mid-run rather than at step 0.
