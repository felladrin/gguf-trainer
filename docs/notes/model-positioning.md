# What a sub-100M model is actually used for (2026-08-12)

> Written during the Minueza-3 run, before the CLI existed. Paths like `examples/…` name that run's
> artifacts and scripts; the equivalent commands are in [agents.md](../../agents.md).

Phase B ended at HellaSwag 29.25, and the question that follows is not "how do we get to 35" but
"what is this model _for_". This is a demand survey of the sub-100M population on Hugging Face, run
first-hand against their API, to answer that from evidence instead of taste. It changes the premise
of `post-training.md`, which assumes the target is a general chat assistant.

## Method (reproducible)

```sh
# top-N most-downloaded, with param counts, then filter total < 100M locally
curl -sG https://huggingface.co/api/models \
  --data-urlencode sort=downloads --data-urlencode direction=-1 \
  --data-urlencode limit=1000 --data-urlencode skip=0 \
  --data-urlencode 'expand[]=safetensors' --data-urlencode 'expand[]=downloads' \
  --data-urlencode 'expand[]=pipeline_tag' --data-urlencode 'expand[]=tags'
```

`downloads` is the trailing 30 days. `skip` caps at 4000, so each sweep is the top 4000 of its
ordering: one sweep unfiltered, one with `pipeline_tag=text-generation`, plus `summarization` /
`text2text-generation` / `translation`. Param counts come from `safetensors.total`; 2682 of the top
4000 report one.

**Two measurement traps.** The API undercounts params for repos whose safetensors index is sharded
or partial: `ornith-ai/Ornith-1.0-35B` reports 664,944 params across 16 shards, and it is a real 35B
MoE, not a fake. And `mlx-community/*-4bit` entries report _packed_ sizes, so a 0.6B model lands at
93.2M. Both classes must be excluded by hand. Likewise, most of the sub-100M `text-generation`
download volume is CI fixtures (`trl-internal-testing/*`, `*/tiny-random-*`,
`peft-internal-testing/*`); dropping them takes 151 models down to 53.

## Demand, by task

| Category                         | Models <100M | Downloads/mo | Typical arch                |
| -------------------------------- | ------------ | ------------ | --------------------------- |
| Embeddings (sentence-similarity) | 37           | **260.4M**   | BERT, 22-33M par            |
| Reranking (text-ranking)         | 10           | **96.0M**    | BERT cross-encoder, 22M par |
| Feature-extraction               | 27           | 82.7M        | BERT                        |
| Translation                      | 49           | 22.6M        | T5 / OPUS-MT, 60-75M par    |
| Fill-mask                        | 10           | 11.9M        | DistilBERT                  |
| Text-classification              | 16           | 9.5M         | DistilBERT, 67M par         |
| Token-classification (NER)       | 48           | 4.4M         | DistilBERT                  |
| Text-generation (decoder-only)   | 61           | 37.9M        | ~all CI fixtures            |

The single most-downloaded model in the sweep is `sentence-transformers/all-MiniLM-L6-v2`: **22.7M
params, 240.3M downloads/mo**. Four times smaller than ours.

## Decoder-only under 100M has no production users

The 53 real ones, ranked:

| Model                           | Par.  | Downloads/mo | Role                        |
| ------------------------------- | ----- | ------------ | --------------------------- |
| `distilbert/distilgpt2`         | 88.2M | 2.35M        | research baseline           |
| `EleutherAI/pythia-70m-deduped` | 95.6M | 0.77M        | interpretability suite      |
| `EleutherAI/pythia-14m`         | 14.1M | 0.27M        | interpretability suite      |
| `Maykeye/TinyLLama-v0`          | 4.6M  | 0.23M        | demo                        |
| `ggml-org/stories15M_MOE`       | 36.4M | 0.065M       | llama.cpp test asset        |
| `SupraLabs/Supra-50M-Reasoning` | 51.8M | 0.035M       | experiment (the Supra2 lab) |
| `Xenova/llama2.c-stories15M`    | 15.2M | 0.015M       | transformers.js demo        |

Baselines, interpretability suites, fixtures and demos. Not one product. Whatever this model
becomes, there is no existing audience for it _as a small causal LM_; the audience attaches to a
task.

Generation at this size does have a market, but not through a causal LM: `google-t5/t5-small`
(60.5M) pulls **21.8M/mo** and the `Helsinki-NLP/opus-mt-*` family (60-75M) serves translation.
Encoder-decoder, task-specific.

And the plausible-sounding generative ideas fail on demand, not on capability.
`JulesBelveze/t5-small-headline-generator` exists at 60.5M params with **~0 downloads**, and the
whole sub-100M `summarization` category (54 models) is near-zero. Title and summary generation at
this size is buildable and unwanted.

## Priority for the fine-tunes

1. **Classification / routing.** Cheapest fit: a decoder does it natively, demand is proven across
   many niches (sentiment, toxicity, phishing, PII, gibberish detection), and the fine-tune costs
   hours. The concrete target is MiniSearch's "does this query need a web search?".
2. **Reranking.** The widest supply gap in the data: 10 models serving 96.0M downloads/mo is **9.6M
   per model**, the highest ratio of any category. Also feeds MiniSearch.
3. **Translation** stays out for now despite t5-small proving the size, because our 32768 BPE vocab
   is English-centric: a new pair needs a new tokenizer and a fresh pretrain.
4. **Embeddings are deliberately skipped** despite the largest raw demand. Bidirectional attention
   wins the task, and `all-MiniLM-L6-v2` at 240M/mo is not displaceable.

**No new architecture is needed for 1 and 2.** Do both generatively: constrain the output with
llama.cpp `--grammar` and score by the logprob of the label tokens. For reranking that is the monoT5
recipe ("is this document relevant to this query?", read the `yes` logprob). Both run on today's
GGUF export path, through the same `sft.ts` that was already the next step. Only the `.tokens`
changes.
