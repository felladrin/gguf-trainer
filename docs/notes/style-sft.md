# Style SFT: teaching a small base model one consistent voice (2026-08-17)

> Written during the Minueza-3 run. Some paths name that run's artifacts; the equivalent commands are in [agents.md](../../agents.md).

Instruct SFT on a small, hand-filtered chat set whose **answers have been rewritten in a single author's voice**, on top of a 94.7M base (hidden 640 x 12 layers, ctx 8192). A model this size cannot be a knowledge assistant, so the target is what it can actually carry: turn-taking, register, and a voice that stays the same from answer to answer.

The numbers below come from one 1200-conversation run. They are recorded because the failure modes were expensive to find, not because that particular model matters.

Four steps, each resumable, with the intermediate corpora under `corpus/style/` (gitignored):

| Step     | Command         | Output                                                   |
| -------- | --------------- | -------------------------------------------------------- |
| Seed     | `style-seed`    | `corpus/style/seed-N.jsonl`                              |
| Restyle  | `style-restyle` | `corpus/style/restyled-N.jsonl`                          |
| Tokenize | `chat-corpus`   | `data/style-N.{tokens,mask,tokenizer.json,template.txt}` |
| Train    | `finetune`      | `out/sft-style-N.gguf`                                   |

`deno run tests/style-pipeline.ts` covers the pure logic (seed filters, reply parsing, style validation).

## 1. Seed

A weighted mix of `HuggingFaceTB/smoltalk` configs, the SFT set built for SmolLM2 (135M-1.7B), which is the closest thing to this size class:

| Config                   | Share | Why                                             |
| ------------------------ | ----- | ----------------------------------------------- |
| `everyday-conversations` | 30%   | short multi-turn chit-chat: teaches turn-taking |
| `smol-magpie-ultra`      | 30%   | the general-assistant bulk                      |
| `systemchats-30k`        | 15%   | makes the system prompt mean something          |
| `smol-summarize`         | 12%   | the answer is derivable from the prompt         |
| `smol-rewrite`           | 8%    | same                                            |
| `smol-constraints`       | 5%    | instruction-following formats                   |

Left out on purpose: `numina-cot-100k`, `metamathqa-50k`, `self-oss-instruct`, `openhermes-100k` (math, code and knowledge recall are all above a 94.7M base, and they teach confident wrong answers), `apigen-80k` (that is the later tool-calling stage) and `longalign` (blows the sequence budget).

Filters: 2-12 messages, strict user/assistant alternation ending on the assistant, assistant turns 20-900 chars, user turns <= 700, conversation <= 2500 chars, near-ASCII, no code fences or LaTeX.

Dedup is on the **whole** conversation, not a prefix: thousands of distinct `everyday-conversations` chats share the opening exchange ("Hi" / "Hello! How can I help you today?"), so any prefix key throws almost all of them away (2078 of 2260 in the first attempt).

## 2. Restyle

`style-restyle` shells out to the `pi` CLI in print mode against a local llama-swap endpoint (this run used `KAT-Coder-V2.5-Dev`), with a style guide as the **system prompt** and the conversation as stdin. Only assistant turns are rewritten; user and system turns pass through verbatim, because the model must learn to _answer_ in that voice, not to be _asked_ in it.

Decisions that came out of measurement, not taste:

- **The style guide is injected, not discovered.** Pointing the agent at a file leaves it to read it with its own tools; the run uses `-nt` (no tools) and `--system-prompt`, so the guide is guaranteed present and the prefix is byte-identical on every call.
- **`--jobs=1`.** Three parallel jobs measured 0.048 conversations/s against 0.0625 sequential: a 13k-token system prompt is a prompt-cache hit only while one stream owns the slot.
- **`### REPLY k` markers, not JSON.** A JSON array of strings fails on any reply containing a paragraph break (raw newlines inside a JSON string). Switching the reply format took acceptance from 2/6 to 7/8. A JSON array is still accepted as a fallback.
- **Every rewrite is validated** (count, per-reply length ratio 0.3-2.5x, no leaked author name), retried once, then dropped to `<out>.failed.jsonl` with a reason. Rejects never reach training.
- **Em dashes are normalized, not rejected** (`a—b` -> `a, b`, `$500–$2,000` -> `$500-$2,000`): the source data is full of them and the fix is mechanical.

Acceptance over the 1200-conversation run: 1055 kept, 145 rejected (96 length ratio, 18 too short, 11 count mismatch, 1 unparseable). Throughput 0.07 conversations/s (~14s each; multi-turn rows cost more than single-answer ones), so budget ~4.5 hours per 1200.

## 3. Tokenize

`chat-corpus` takes the restyled `.jsonl` directly. It reuses the base model's `tokenizer.json` **verbatim** (the frozen-vocab rule), renders through the exporter's own Jinja template so training text matches inference text, and writes the assistant-only supervision mask beside the tokens (~63% of tokens supervised on this corpus).

## 4. Train

`finetune` is the pretraining loop with three additions:

- `--mask`: assistant-only loss. This is what makes it SFT rather than more pretraining.
- `--template`: embeds the chat template in every exported GGUF, so llama.cpp renders turns the way the model was trained on them.
- `--cold-optimizer`: ignore the resumed checkpoint's optimizer sidecar, because a new stage should not inherit momentum from another objective.

LR is 0.001, a tenth of the pretraining peak: SFT nudges an existing base, it does not reshape it.

## Honest expectations

At HellaSwag 29.25 the base is barely above chance, so this stage buys _format and voice_, not correctness. Judge the result on whether it holds ChatML turns, stops at `<|im_end|>`, and answers in short plain sentences without filler openers. Do not judge it on whether the answers are true.

Two things worth knowing before repeating this:

- An 11-conversation smoke corpus (2940 tokens, 62.9% supervised) trained 20 steps from the base and reached loss 3.374 -> 1.281, which is memorization at 28 epochs and not learning. It is still the right first run: it proves the export carries the ChatML template and declares `<|im_end|>` as EOS, which `llama-server`'s `/v1/chat/completions` then renders from.
- ~360k tokens is thin for style transfer. The pipeline is resumable and additive, so the answer is to raise `--total`, restyle only the new ids, and retrain rather than start over.
