# Adding an architecture

One file in `src/arch/`, one line in `src/model/registry.ts`. Nothing else in the repo needs to
change: the CLI flags, the `archs` listing, the tests, the checkpoint loader and the resume logic
all read the registry.

Read `src/arch/llama.ts` first. It is the smallest complete example, under 400 lines, and most of
that is the config and the GGUF names.

Then diff it against `src/arch/qwen3.ts`, which was written by following the recipe below. Qwen3 is
llama plus one thing (per-head QK-RMSNorm before RoPE), and the diff between the two files is
exactly that one thing plus the names it goes out under. That is the shape a good arch file has.

## The 20-minute version

```sh
cp src/arch/llama.ts src/arch/myarch.ts     # start from the closest existing one
# edit the config fields, the forward pass, and the GGUF names
# add `import { myarch } from "../arch/myarch.ts";` and put it in ARCHITECTURES
deno run -A cli.ts archs                    # it should be listed
deno run -A tests/arch-roundtrip.ts         # it should pass
deno run -A tests/gradcheck.ts              # your forward pass is now gradient-checked
```

Then train something tiny and load it in llama.cpp:

```sh
deno run -A cli.ts tokenize --text corpus/any.txt --out data/tiny --vocab 2048
deno run -A cli.ts pretrain --arch myarch --data data/tiny.tokens \
  --out /tmp/t.gguf --hidden 128 --layers 2 --head-dim 32 --steps 20 --seq-len 128 --batch 2
llama-completion -m /tmp/t.gguf -c 256 -n 20 -p "Once upon a time"
```

If llama.cpp loads it and the loss went down, you are done.

## What you implement

The contract is `Architecture<C>` in `src/model/arch.ts`. Every method exists because something
outside the arch file needs it:

| Member                 | Who calls it                                   | Get it wrong and                                      |
| ---------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| `name`                 | the registry, `--arch`, `general.architecture` | nothing resolves                                      |
| `summary`, `reference` | `archs`                                        | your architecture looks undocumented                  |
| `flags`                | `pretrain`, `finetune`                         | your knobs are unreachable                            |
| `configFromFlags`      | `pretrain`                                     | flags do not become a model                           |
| `configFromGGUF`       | resume, `inspect`, `generate`, evals           | checkpoints cannot be read back                       |
| `tinyConfig`           | the test suite                                 | you get no automatic coverage                         |
| `build`                | everything that instantiates a model           | -                                                     |
| `paramCount`           | the startup log                                | sizes are reported wrong                              |
| `describe`             | the startup log, `inspect`                     | runs are unidentifiable in a log                      |
| `exportGGUF`           | every checkpoint write                         | llama.cpp will not load it                            |
| `loadWeights`          | every resume                                   | weights land in the wrong tensors                     |
| `configMatches`        | resume                                         | a mismatched resume trains garbage instead of failing |

And the model you return from `build` implements `LanguageModel`: `params()`, `paramGroups()`,
`forward(ids)`. Optionally `qkNorms()` if the architecture has QK-RMSNorm and should support
MuonClip.

## The parts that are easy to get wrong

**`paramGroups` decides what trains how.** 2-D hidden matmuls go to `muon`; embeddings, the output
head and every norm go to `aux` (AdamW). A tensor in neither group silently never trains. The
round-trip test checks the split covers every parameter, not that you split it correctly.

**Weight layout is `[out, in]` row-major.** `addMatrix` writes ggml's `ne = [in, out]` for you, and
`tensorLoader` reads it back. Do not transpose by hand.

**Norm weights train in gain-frame**, initialized to 1 with `forward = normalize(x) * w`. That
matches llama.cpp's `rms_norm * w` and exports directly. The HF "+1" convention is a storage
artifact of init-0 weights; adding it here would double-count.

**Tied embeddings are implicit in the file.** Omit `output.weight` when tied, and read tying back as
"does the file have an output.weight". Do not add a metadata key for it.

**Match llama.cpp's key names exactly.** They are `<arch>.embedding_length`, `<arch>.block_count`,
`<arch>.attention.head_count`, and so on. Copy them from llama.cpp's loader for your architecture
rather than guessing, and write `attention.key_length` whenever `head_dim * n_heads != hidden`, or
llama.cpp will infer the wrong head size.

**Every field `configMatches` gates needs a flag the user can pass and a place in `inspect`'s
resume line.** For an arch-specific field that means its flag in your `flags` list and a place in
`resumeFlags` (`src/commands/inspect.ts`): a value flag goes in the `optional` table, a boolean
that sets a side (like `--untied-embeddings`) gets a check that prints it only when the
checkpoint needs it; the shape flags (`--hidden`, `--layers`, `--head-dim`, `--max-seq`) come from
`pretrain` and are already printed. A gated field missing either is an abort the user cannot
satisfy (#36's gemma3 `--rms-eps` flag), and a flag without a gated field is a silent mismatch
that trains on the wrong value (#35's RoPE base, #37's untied-embeddings). `vocab` is the one
exception: it comes from the tokenizer, not from a flag.

**Shared flag names must mean the same thing.** `--kv-heads` and `--ffn-dim` already exist; reuse
them if they mean what you think. Do not give a shared flag a `default` in your arch file: the CLI
merges flags across architectures, so a default would leak into everyone else's. Put your default in
`configFromFlags` instead, and name it in the flag's `describe` (`"(qwen3 default: 1e6)"`). Only the
first-registered owner's `describe` fits on the merged help line, so when your default differs the
merged text gets a "differs per --arch" note automatically and `archs --json` carries your wording;
you do not have to edit another architecture's file. Passing a flag that belongs only to another
architecture is already an error, so you do not have to defend against that either.

## What you get for free

Registering an architecture enrolls it in:

- **`tests/arch-roundtrip.ts`**: parameter count honesty, optimizer-group coverage, config
  round-trip through GGUF, bit-exact f32 logits after export and reload, quantized reload within
  tolerance, and that `configMatches` rejects a wrong shape while accepting the right one.
- **`tests/gradcheck.ts`**: your `forward` gets finite-difference checked per element and along
  random directions at the real init scale. This is the test that catches a wrong backward, and you
  did not have to write it.
- The whole CLI: `pretrain`, `finetune`, `eval-loss`, `eval-choice`, `generate`, `inspect`,
  `export`, and the resume machinery including the optimizer sidecar.

What you do NOT get automatically is a GPU parity check of a _new op_. The existing ops are already
covered in `tests/gpu-parity.ts`; if your architecture needs an op that does not exist yet, that op
needs a CPU implementation, a WGSL kernel, and a parity entry before your architecture will train on
the GPU.

## When your architecture needs a new operation

Composing existing ops (`linear`, `attention`, `rmsNorm`, `rope`, `silu`, `gelu`, `mul`, `add`,
`scale`, `embedding`) covers most transformers. If yours needs something new, that is a bigger job
than the arch file:

1. Implement it in `src/model/autograd.ts`, forward and backward, and add it to `OpsBackend`.
2. Add a finite-difference check in `tests/gradcheck.ts`.
3. Write the WGSL kernel in `src/backend/`, forward and backward.
4. Add a GPU-vs-CPU parity entry in `tests/gpu-parity.ts`.
5. Then write the arch file.

**Worked example, the hybrid conv models** (LFM2 and friends): they interleave short causal
depthwise convolutions with attention. The attention half is already here; the conv half needs a
`causalConv1d(x, weight, kernelSize)` op with the four steps above. Its backward is the fiddly part:
gradients flow to the input, to the kernel weights, and across the time axis with the same causal
shift the forward uses. Budget the op, not the architecture.

## Checklist before you send it

```
- [ ] `deno run -A cli.ts archs` lists it with a useful summary and a reference model
- [ ] `deno run -A tests/arch-roundtrip.ts` passes
- [ ] `deno run -A tests/gradcheck.ts` passes (it now includes your forward pass)
- [ ] `deno task test` passes as a whole
- [ ] a tiny model trains for 20 steps and the loss goes down
- [ ] llama.cpp loads the export and generates text
- [ ] `inspect` on that checkpoint prints flags that actually resume it
- [ ] the arch file's header comment says what this architecture does DIFFERENTLY
```

That last one matters more than it looks. The value of one-file-per-architecture is that the diff
between two files is the diff between two models; a header that just says "transformer" throws that
away.
