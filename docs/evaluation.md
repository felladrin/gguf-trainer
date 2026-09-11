# Evaluation

How a checkpoint from this repo is scored, what the scores are worth, and which instruments have
turned out to be misleading. Throughput lives in [performance.md](performance.md); this file is
about model quality, which is a different axis and has a different ceiling.

## The ceiling is tokens per parameter

Not the optimizer, not the kernels, not the hyperparameters. Stated on the axis the small-model
literature argues on:

| model              | params | train tokens | tokens/param | Intelligence Index |
| :----------------- | -----: | -----------: | -----------: | -----------------: |
| SmolLM2-135M       |   135M |          ~2T |       14,815 |              27.13 |
| GPT-X2.5-135M      |   135M |          75B |          556 |              25.17 |
| BananaMind-2-Pro   |   139M |         100B |          719 |              24.96 |
| Supra2-100M-Base   |   101M |          30B |          298 |              19.41 |
| Minueza-3-95M-Base |  94.7M |        1.95B |         20.6 |              10.67 |

20 tokens per parameter is Chinchilla-optimal, and therefore optimal for nothing this project wants:
Chinchilla minimizes loss for a fixed training budget, not quality per parameter at a fixed model
size. At 1588 tok/s, which is 137M tokens/day, reaching 100 tokens/param on one box is about two
months and is the only rung on that ladder actually available. Minueza-2's 1,927 tokens/param is
3.7 years and SmolLM2's ~14,800 is decades.

Two things the top of that cohort does differently, both cheap to try and neither yet measured here:

- **Depth over width, and a 3x FFN rather than 4x.** GPT-X2.5 is 30 layers at hidden 576 with a
  1,728 FFN; BananaMind-2-Pro is 24 at 640 with 1,920. Ours is 12 layers at 640 with a 2,560 FFN.
  `gemma3Config` derives the FFN as ~4x hidden, so this is a one-config experiment.
- **A data mixture we do not have.** Both report roughly FineWeb-Edu 50%, DCLM 26%, Cosmopedia-v2
  13.5%, FineMath-4+ 8%, Python 2%. Ours is FineWeb-Edu and nothing else. No FineMath is the most
  likely reason arithmetic and ARC sit at chance, and it is the cheapest of the gaps to close.

Both also use plain AdamW at peak lr 1.5e-3 with a 2,000-step warmup, so Muon is not what separates
them from us.

## The four-task score

`eval-choice` scores the Open SLM Leaderboard's tasks by length-normalized log-likelihood: for each
candidate it forwards `[context + completion]` once and reads the negative log-likelihood over the
completion tokens only. `acc_norm` ranks by summed NLL divided by the choice's CHARACTER length,
`acc` by the raw sum.

**Character length, not token count, is what makes the numbers comparable.** lm-evaluation-harness
uses `completion_len = np.array([float(len(i)) for i in choices])`; per-token normalization is
tokenizer-dependent, which is the thing character length exists to avoid. This repo divided by token
count until 2026-08-21, and HellaSwag's query was the bare `ctx` field rather than
`preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize())`. Fixing both moved the index
by about a point:

| Task          | acc_norm before | acc_norm after | acc before | acc after |
| :------------ | --------------: | -------------: | ---------: | --------: |
| PIQA          |           61.04 |          60.88 |      60.17 |     60.17 |
| ARC-Easy      |           39.90 |          41.04 |      45.03 |     45.03 |
| ARC-Challenge |           23.89 |          25.51 |      20.39 |     20.39 |
| HellaSwag     |           28.40 |          29.90 |      28.05 |     28.14 |
| **Index**     |        **9.81** |      **10.77** |          - |         - |

The raw `acc` column is the check that the change did only what it should: swapping a normalizer
cannot touch a sum-NLL ranking, and it did not, on every row except the one whose query was rebuilt.

The board's formula normalizes each task against its chance floor,
`N = 100 x (score - chance) / (100 - chance)`, averages ARC-Easy and ARC-Challenge into ONE ARC term
before normalizing, and weights ArithMark-3 at 0.65:

    Index = (HellaSwag + ARC + PIQA + 0.65 x ArithMark) / 3.65

**Minueza-3-95M-Base scores 10.67**: PIQA 61.26, ARC-Easy 40.53, ARC-Challenge 23.81,
HellaSwag 30.14. ArithMark-3 is not implemented here and is assumed at chance; omitting the term
instead gives 12.98, so the honest range is 10.7 to 13.0.

Four qualifiers to carry with any number here:

- **It is self-computed, not a submitted entry.** The scoring matches lm-eval-harness on query
  construction and normalization, which is what makes it comparable at all, but nobody else ran it.
- **A single Index value has a 1-sigma binomial error of about 0.71** on these item counts. Three
  checkpoints spanning 0.37 are not ordered by it. The defensible claim from the roleplay curriculum
  is that it did not move general capability, not that any stage won.
- **ARC-Challenge below its 25% chance floor is a property of the ruler at this scale**, not a
  defect: five of seven models in a head-to-head sat between 21.74 and 23.41 under length-normalized
  scoring.
- **PIQA carries the whole index.** Normalizing the row above gives PIQA 22.52, ARC 9.56 and
  HellaSwag 6.85, which recombine to the 10.67. Two-option normalization divides by 50 rather than
  75, so any edge over chance on PIQA counts for twice as much.

Do not mix these numbers with llama.cpp's `llama-perplexity --multiple-choice`: the same checkpoint
scores 32.20 acc_norm here, on a 2,000-item HellaSwag subset, and 28.46 there on the full 10,042.
Harness and subset are confounded in that pair, so it measures neither one. If you do run that
harness, pass `-np 8`, because ARC-Challenge contains 5-option questions and the default 4 aborts
the run at task 210 of 299 with no result.

## Picking a checkpoint

**The training loss is a divergence alarm, not a selector.** Two halves of one roleplay SFT schedule,
same seed, same shapes:

| signal                                | v3 (550 steps) | v4 (1100 steps)           |
| :------------------------------------ | :------------- | :------------------------ |
| training loss, mean of the last 10    | 2.726          | **2.681**                 |
| name given at turn 2, recalled turn 6 | yes            | no                        |
| persona held across 6 turns           | yes            | collapses to one template |
| greedy decode                         | no loop        | loops on 2 of 4 prompts   |

The extra half epoch bought 0.045 nats on the training distribution and paid for it in the behaviour
the model exists for. The loss curve gave no hint; reading two transcripts settled it in five
minutes.

Use the pairing, not either half: **held-out loss to rank checkpoints cheaply, a behavioural probe
to confirm the one it picks.** Both need setting up before the run, not after:

- **Carve the held-out slice before training starts**, or it does not exist. A split of a corpus the
  run has already consumed is rigged.
- **Dedup across the split, not just at document boundaries.** One LittleLamb holdout document
  appeared in the training half because the source carried that conversation twice. One hash set
  over both sides is the whole check.
- **`finetune` overwrites `--out` on every write**, so without `--keep-checkpoints` a finished run
  leaves exactly one file and nothing to rank.

**A held-out curve that turns at a phase boundary is not evidence about the phase.** The LittleLamb
held-out loss rose during the WSD cooldown, which looked like the cooldown hurting. A
counterfactual, both arms resumed from the same checkpoint so they share a data stream and differ
only in schedule, said otherwise: the cooldown arm was better at all eight matched steps
(mean -0.0043, sign test p = 0.0078), and neither arm reproduced the rise at all. The original
excursion was the same kind of event as an unexplained 0.025 excursion elsewhere in the same series:
the weights random-walk in a low-loss basin, and plateau-step-plateau is what a random walk looks
like sampled at 32 points. Anything moving less than the excursions the series makes for no reason
needs a shared-checkpoint counterfactual before it becomes a claim, and running one costs a night.

## Behavioural probes, and how they break

Three scorers in this repo were wrong before they were right, and **each break flattered the
conclusion in front of it**:

- A roleplay battery bucketed a staged co-star's line as a stray label, so it scored a checkpoint
  higher for ignoring the character it was told to write. Splitting staged speakers out reversed the
  ranking. The split then filed `You:` under "staged and therefore correct", when the model writing
  the human's turn is the exact failure the battery exists to expose, and `You:` was 76% of that
  bucket (825 of 1,092 labels across 216 runs). That reversed it again.
- An endpoint scorer read `res.stopped_eos`, a field the llama.cpp build does not send (it reports
  `stop_type: "eos"`). An absent field is `undefined`, `undefined` is falsy, so the column read 0 for
  every model, **including the control, which is why nobody looked.** `stoppedOnEos` throws now when
  neither field is present, because "the field is missing" and "the model never stopped" are
  different facts and only one of them is a zero.
- A combined other-speaker count reported t = -3.02 from a scorer that lived in a scratchpad and is
  gone. It is retracted rather than re-derived.

The per-run counts and the holdout logs every number in this section is read from are in
[measurements/](measurements/). `scripts/score-rp-battery.ts` carries the four distinctions that
survived (`handback`, `self`, `costar`, `invented`) and `tests/rp-battery-score.ts` pins them so no
merge comes back silently.
Read a battery's within-checkpoint spread against its between-checkpoint spread before reading its
ranking: on 18 checkpoints the former exceeded the latter on every count, which is an instrument
saying it cannot separate them.

## Samplers

Two sweeps, and they measure different axes. Together they are why the model cards recommend what
they do.

**Truncating hard makes a 95M model worse.** 14 presets, 2 scenarios, 2 seeds: `min-p 0.15` at
temp 0.7 produced 8-gram loops and a distinct-trigram ratio of 0.41. Every preset that avoided loops
did it with DRY, XTC, mirostat or top-n-sigma, not with truncation. Squeeze the distribution of a
model this size and there is nothing left but its favourite phrase.

**But the diversity leaderboard is a trap.** Mirostat topped every column and reads worst by eye.
Highest-diversity and least-coherent are the same thing at this size, because distinct-3 rewards
exactly the invented-token garble a small model produces under pressure. Any future sweep needs a
garble metric in the table: the tokenizer's own pieces-per-word, ~2.0 on ordinary English and ~4.0
on invented names.

**And a preset that avoids loops can still collapse.** Four presets, four personas, twenty seeds,
scored on how often a reply runs under 15 tokens and hands the turn straight back:

| Preset                                          | Mean tokens | Replies under 15 tokens |
| :---------------------------------------------- | ----------: | ----------------------: |
| `temp 0.85 min-p 0.08 top-k 0` + DRY 1.0/1.75/2 |        35.2 |              28% +/- 5% |
| `temp 0.7 top-k 40 top-p 0.9 rep 1.1/128`       |        58.5 |              16% +/- 4% |
| `temp 1.0 min-p 0.10 rep 1.05/64`               |        50.7 |              12% +/- 4% |
| `temp 0.6 top-k 30 top-p 0.9 rep 1.1/128`       |        61.5 |           **8% +/- 3%** |

Every DRY variant landed at 28-34%, every non-DRY one at 8-16%, against a combined error of about 6
points.

**What no preset fixes.** A persona written as a male cat comes back as "her" under every setting; a
persona written as reticent gushes under every setting. Persona adherence at 95M is a capacity limit
and the sampler does not reach it.

## Two operational notes

- **Long benchmarks run from a pinned `git worktree`, with absolute model paths.** `eval-choice`
  runs out of the working tree, and a `git checkout` during a multi-hour run swaps the code under the
  next task in the loop. A re-measure once reproduced the old numbers exactly because the branch had
  switched away six seconds after the fix landed. The scripts grep the source for the change they are
  supposed to be measuring before they start.
- **Record which llama.cpp backend and build a result came from.** On Strix Halo (Vulkan, b7682),
  `-ngl 0` through `-ngl 99` are byte-identical across 26 deterministic completions except for one
  near-tie argmax flip. That is reassuring rather than general: the original report of incoherent
  offloaded output was on Metal, and nothing here transfers to it.

## References

- lm-evaluation-harness, the reference for the query construction and the `acc_norm` normalizer
  `eval-choice` matches: [github.com](https://github.com/EleutherAI/lm-evaluation-harness)
- Open SLM Leaderboard: [huggingface.co](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard)
- Extreme overtraining in tiny language models, for the far end of the tokens/param curve:
  [huggingface.co](https://huggingface.co/blog/Banaxi-Tech/ovdadadadd)
- Held-out eval per epoch and "loss != correctness" as standard SFT practice:
  [towardsdatascience.com](https://towardsdatascience.com/how-to-fine-tune-an-llm-an-end-to-end-guide/)
