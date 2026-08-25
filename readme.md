```
 ______     ______     __  __     ______      ______   ______     ______     __     __   __     ______     ______
/\  ___\   /\  ___\   /\ \/\ \   /\  ___\    /\__  _\ /\  == \   /\  __ \   /\ \   /\ "-.\ \   /\  ___\   /\  == \
\ \ \__ \  \ \ \__ \  \ \ \_\ \  \ \  __\    \/_/\ \/ \ \  __<   \ \  __ \  \ \ \  \ \ \-.  \  \ \  __\   \ \  __<
 \ \_____\  \ \_____\  \ \_____\  \ \_\         \ \_\  \ \_\ \_\  \ \_\ \_\  \ \_\  \ \_\\"\_\  \ \_____\  \ \_\ \_\
  \/_____/   \/_____/   \/_____/   \/_/          \/_/   \/_/ /_/   \/_/\/_/   \/_/   \/_/ \/_/   \/_____/   \/_/ /_/

           F E L L A D R I N ' S   G G U F   T R A I N E R   +∞   ::   train-from-scratch -> GGUF, no PyTorch
```

# Felladrin's GGUF Trainer

Train a language model from scratch, in TypeScript, straight to GGUF. No Python, no PyTorch. The weights live in a GGUF file from the first step to the last, so every checkpoint is already something llama.cpp can load.

Training runs on WebGPU (WGSL compute shaders, forward and backward), so AMD, Apple Silicon and NVIDIA all work. The CPU backend is pure TypeScript, and it's there as the correctness reference, not as a fast path.

I trained [Minueza-3-95M-Base](https://huggingface.co/Felladrin/Minueza-3-95M-Base) with it (94.7M parameters, 1.95B tokens, one APU).

## Requirements

Deno 2.x, which ships WebGPU natively (CI tests on it). That's it: no build step, no npm install.

Node 22.6+ and Bun run the CPU reference path too, which is enough for the test suite (`deno task test:node`), though the CLI itself is Deno-only.

Memory scales with `seq-len x batch`, and on this trainer that means tens of GB at the shapes a real run uses. `--reclaim` frees each micro-batch's activations as it goes, which cut peak GPU memory 5.6x in the measured A/B (39.3 GB to 7.0 GB) for 23% less throughput: it is off by default, and it is what makes a long-context run fit on a small GPU. Measured both ways in [docs/optimization.md](docs/optimization.md) lever 3b.

The continue-training recipe below also uses [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli), Hugging Face's CLI, to fetch a published model. Nothing else needs it.

## Quickstart

```sh
deno run -A cli.ts demo     # trains a tiny model, exports 3 GGUFs, verifies them. Under a minute.
deno run -A cli.ts help     # every command
```

To continue training a published model:

```sh
hf download Felladrin/Minueza-3-95M-Base --local-dir base/
deno run -A cli.ts inspect --model base/Minueza-3-95M-Base.F32.gguf
deno run -A cli.ts pretrain --data your.tokens --out out/continued.gguf \
  --resume base/Minueza-3-95M-Base.F32.gguf \
  --hidden 640 --layers 12 --steps 5000 --seq-len 2048 --batch 8
```

## Base models to fine-tune

Any published checkpoint works as a starting point if it clears two gates.

**It has to convert to one of the three architectures.** `inspect` reads `general.architecture` out
of the GGUF and looks it up in the registry, so `gemma3`, `llama` and `qwen3` load and everything
else does not. The name on the model card is not a reliable guide: Qwen2 and Qwen2.5 convert to a
`qwen2` arch, and SmolLM3 converts to `smollm3`, so neither loads here despite Qwen3 and SmolLM2
both being fine.

**For chat or roleplay fine-tuning it also needs ChatML in its vocab.** `chat-corpus` aborts unless
`<|im_start|>`, `<|im_end|>` and `<|endoftext|>` each encode as a single token. A base without them
is still perfectly good for continued pretraining with `pretrain --resume`; it just cannot go
through the SFT path.

Verified against the Hugging Face API on 2026-08-25, smallest first:

| Model                                                                               | `--arch` | Params | Chat fine-tune | Good for                                                                       |
| :---------------------------------------------------------------------------------- | :------- | -----: | :------------- | :----------------------------------------------------------------------------- |
| [Minueza-3-95M-Base](https://huggingface.co/Felladrin/Minueza-3-95M-Base)           | `gemma3` |  94.7M | yes            | the fastest loop, and the one trained by this repo                             |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)                   | `llama`  |   135M | yes            | the best-trained tiny base; start here if you want a real result               |
| [SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | `llama`  |   135M | yes            | same, already instruction-tuned, so SFT adds a register rather than the format |
| [LittleLamb](https://huggingface.co/MultiverseComputingCAI/LittleLamb)              | `qwen3`  |   293M | yes            | the qwen3 option, and heavily trained for its size                             |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M)                   | `llama`  |   362M | yes            | a step up while still finishing overnight                                      |
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)                      | `qwen3`  |   596M | yes            | about as large as this trainer is practical at                                 |
| [TinyLlama_v1.1](https://huggingface.co/TinyLlama/TinyLlama_v1.1)                   | `llama`  |   1.1B | no             | continued pretraining only, and slow here; the no-ChatML case                  |

All Apache-2.0 and none gated. Google's Gemma 3 checkpoints are `gemma3` and would otherwise fit,
but they are access-gated and carry the Gemma licence rather than a permissive one, which makes
them a poor first suggestion.

Three of these have been taken end to end with this repo, so the recipe is not theoretical:
[Minueza-3-95M-RP](https://huggingface.co/Felladrin/Minueza-3-95M-RP),
[LittleLamb-293M-RP](https://huggingface.co/Felladrin/LittleLamb-293M-RP) and
[SmolLM2-135M-Heretic-RP](https://huggingface.co/Felladrin/SmolLM2-135M-Heretic-RP).

Evaluating one that is not listed takes a single command. Convert it with llama.cpp's
`convert_hf_to_gguf.py`, then:

```sh
deno run -A cli.ts inspect --model your-base.gguf --dump-tokenizer data/your-base.tokenizer.json
```

That prints the architecture, the exact `--resume` flags the checkpoint needs, and whether the
vocab can drive `chat-corpus`. If the architecture is not one of the three, nothing else matters.
The full recipe is in [agents.md](agents.md) under "Fine-tune a downloaded checkpoint".

[agents.md](agents.md) is the full manual: recipes for every workflow, the invariants that waste a run when broken, what each failure message means, and the measured throughput. It's written for a coding agent, which happens to make it the fastest read for a person too.

## Commands

|                               |                                                                                                                  |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `corpus`                      | download and shuffle a pretraining corpus from Hugging Face                                                      |
| `tokenize`                    | text corpus to binary token stream plus a BPE vocab                                                              |
| `chat-corpus`                 | chat dataset to SFT tokens plus an assistant-only loss mask                                                      |
| `pretrain`                    | train a base model, or continue one with `--resume`                                                              |
| `finetune`                    | fine-tune on chat data, supervising only the assistant turns                                                     |
| `eval-loss`                   | held-out validation loss on fixed windows                                                                        |
| `eval-choice`                 | ARC-Challenge / ARC-Easy / HellaSwag / PIQA / perplexity                                                         |
| `generate`                    | greedy completion without llama.cpp                                                                              |
| `inspect`                     | a GGUF's metadata, shape, and the flags needed to resume it                                                      |
| `export`                      | re-export under a release name, with quants                                                                      |
| `archs`                       | list the architectures this build can train                                                                      |
| `bench`                       | time the WebGPU kernels at fixed shapes, before and after a change                                               |
| `demo`                        | the install check                                                                                                |
| `style-seed`, `style-restyle` | optional: build a chat corpus rewritten in one author's voice (needs the [pi](https://pi.dev/) coding agent CLI) |

Every command has `--help`, and `deno run -A cli.ts help --json` dumps the whole CLI as a schema.

## Architectures

Three, for now: `gemma3` (GQA + QK-norm, sandwich norms, GeGLU, sliding-window attention), `llama` (pre-norm GQA, SwiGLU, full attention, the SmolLM2 and TinyLlama shape) and `qwen3` (llama plus per-head QK-RMSNorm).

Pick one with `--arch`. A checkpoint records its own, so resuming never needs it, and
[Base models to fine-tune](#base-models-to-fine-tune) lists published checkpoints that load.

Each architecture is a single file in `src/arch/` plus one line in the registry, and it gets the gradient checks and the export round-trip test for free. If you want to add one, it's documented in [docs/adding-an-architecture.md](docs/adding-an-architecture.md).

## What works

- GGUF v3 writer and reader, spec-faithful metadata and tensor layout, with F16/Q8_0/Q4_0 quantizers matching the ggml block layout.
- Muon (Newton-Schulz orthogonalized momentum) and AdamW, both GPU-resident, with MuonClip, muP init transfer and a WSD schedule.
- The whole op set as WGSL, forward and backward, including flash-style and sliding-window attention and a fused cross-entropy.
- Checkpoint resume through GGUF plus an optimizer-state sidecar, so a long run survives an interruption.
- `deno task test` type-checks the tree, then runs finite-difference gradient checks on every op (with a negative control, so we know it can catch a wrong backward) and GPU-vs-CPU parity on every kernel.

## Results

Scored with `eval-choice` on the Open SLM Leaderboard's four tasks, full sets, 0-shot, acc_norm. [Minueza-3-95M-Base](https://huggingface.co/Felladrin/Minueza-3-95M-Base), the published 94.7M model trained with this repo, gets PIQA 61.26, ARC-Easy 40.53, ARC-Challenge 23.81, HellaSwag 30.14. Combined the way that board combines them, that is an **Intelligence Index of 10.67**, against 25-27 at the top of the board. The formula and the arithmetic are in [docs/optimization.md](docs/optimization.md) lever 9c.

Two qualifiers that matter. This is a self-computed index, not a submitted entry: the scoring matches lm-eval-harness on the query construction and on normalizing acc_norm by character length, which is what makes the numbers comparable at all, but nobody else ran them. And the board's fifth task (ArithMark-3) is not implemented here, so it is assumed at chance; omitting the term instead gives 12.98, making the honest range 10.7-13.0.

At 94.7M parameters on one consumer APU, the cohort above it on that board is trained on 14 to 700 times more tokens per parameter. `docs/optimization.md` lever 11 covers what they do differently, and lever 4 covers why tokens-per-parameter, not anything in this codebase, is the ceiling on quality.

## Honest limits

- **A model this size is a demonstration, not a product.** At 94.7M and ~21 tokens per parameter it produces locally fluent text and cannot hold a conversation: it does not carry facts across turns, and it will confabulate an answer to anything factual. The index above measures the trainer working, not a model you would deploy.
- Single-digit to low-hundreds of millions of parameters. At 94.7M on one APU it does 1588 tokens/s, up from the ~900 the published model was trained at, so the published model's 1.95B tokens is about 14 days rather than 25. JS and WebGPU still won't match a CUDA cluster, and no flag changes that.
- Training keeps float master weights and quantizes at export, so you can't really train in Q4_0.
- Context length is capped by a WebGPU buffer limit, before compute becomes the problem: 8192 on adapters that grant their full buffer size, 2500-3000 on the ones that don't.
- The GGUF output is structurally verified here, but please check it against `llama-cli` before trusting a specific build.

## Contributing

See [contributing.md](.github/contributing.md). The short version: keep the engine dependency-free and runtime-agnostic, cover any new autograd op with a finite-difference gradient check, and don't break GGUF loadability in llama.cpp (the tensor names and metadata keys are a contract).

## License

MIT, see [license.txt](license.txt).
