# Correctness and robustness notes

> Measured levers, not speculation. Maintained: every number here was taken on real hardware, and
> the commands are the current CLI.

The half of the lever list that is about being right rather than being fast: guards, gates, file
formats, and the failures each one was written after. Split out of
[optimization.md](optimization.md) on 2026-09-10, when it had grown past half that file.

**Lever numbers are stable ids, not an ordering.** They are never reused and never renumbered, they
are not sequential within either file, and a number tells you nothing about which file holds it. If
a lever you are looking for is not here, it is in the other one:

```
grep -rn "^### 39\." docs/
```

Everything else about this file follows optimization.md's conventions: the measurement is the point,
a lever that turned out to be wrong is corrected in place with the number that produced it kept, and
a null result is worth as much as a win.

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
