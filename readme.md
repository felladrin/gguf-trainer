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

Pick one with `--arch`. A checkpoint records its own, so resuming never needs it.

Each architecture is a single file in `src/arch/` plus one line in the registry, and it gets the gradient checks and the export round-trip test for free. If you want to add one, it's documented in [docs/adding-an-architecture.md](docs/adding-an-architecture.md).

## What works

- GGUF v3 writer and reader, spec-faithful metadata and tensor layout, with F16/Q8_0/Q4_0 quantizers matching the ggml block layout.
- Muon (Newton-Schulz orthogonalized momentum) and AdamW, both GPU-resident, with MuonClip, muP init transfer and a WSD schedule.
- The whole op set as WGSL, forward and backward, including flash-style and sliding-window attention and a fused cross-entropy.
- Checkpoint resume through GGUF plus an optimizer-state sidecar, so a long run survives an interruption.
- `deno task test` type-checks the tree, then runs finite-difference gradient checks on every op (with a negative control, so we know it can catch a wrong backward) and GPU-vs-CPU parity on every kernel.

## Results

Scored with `eval-choice` on the Open SLM Leaderboard's four tasks, full sets, 0-shot: PIQA 61.32, ARC-Easy 40.03, ARC-Challenge 22.61, HellaSwag 28.16 (acc_norm). Combined the way that board combines them, those give an **Intelligence Index of 9.67**, against 25-27 at the top of the board; the formula and the arithmetic are in [docs/optimization.md](docs/optimization.md) lever 9b.

Two qualifiers that matter. This is a self-computed index, not a submitted entry, and the board's fifth task (ArithMark-3) is not implemented here, so it is assumed at chance; omitting the term instead gives 11.76. And it was measured on a roleplay continuation of the published base, not on the published [Minueza-3-95M-Base](https://huggingface.co/Felladrin/Minueza-3-95M-Base) file itself, which has not been scored on all four tasks yet.

For placement, 9.67 would sit above 82 of the 129 entries on that board with complete task data, at 94.7M parameters trained on one consumer APU. The cohort above it is trained on 14 to 700 times more tokens per parameter; `docs/optimization.md` lever 11 covers what they do differently, and lever 4 covers why tokens-per-parameter, not anything in this codebase, is the ceiling on quality.

## Honest limits

- **A model this size is a demonstration, not a product.** At 94.7M and ~21 tokens per parameter it produces locally fluent text and cannot hold a conversation: it does not carry facts across turns, and it will confabulate an answer to anything factual. The index above measures the trainer working, not a model you would deploy.
- Single-digit to low-hundreds of millions of parameters. At 94.7M on one APU it does 1588 tokens/s, up from the ~900 the published model was trained at, so 2B tokens is about 14 days rather than 25. JS and WebGPU still won't match a CUDA cluster, and no flag changes that.
- Training keeps float master weights and quantizes at export, so you can't really train in Q4_0.
- Context length is capped by a WebGPU buffer limit, before compute becomes the problem: 8192 on adapters that grant their full buffer size, 2500-3000 on the ones that don't.
- The GGUF output is structurally verified here, but please check it against `llama-cli` before trusting a specific build.

## Contributing

See [contributing.md](.github/contributing.md). The short version: keep the engine dependency-free and runtime-agnostic, cover any new autograd op with a finite-difference gradient check, and don't break GGUF loadability in llama.cpp (the tensor names and metadata keys are a contract).

## License

MIT, see [license.txt](license.txt).
