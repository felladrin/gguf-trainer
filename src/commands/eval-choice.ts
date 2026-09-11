// Evaluate a trained GGUF on multiple-choice benchmarks (ARC-Challenge,
// ARC-Easy, HellaSwag, PIQA) and word/token perplexity, using the trainer's OWN forward, no
// external eval framework, so it stays as portable as the rest of the project.
//
// Scoring is the standard length-normalized log-likelihood: for each candidate
// completion we forward [context + completion] once and read the negative
// log-likelihood over ONLY the completion tokens (context targets masked to -1,
// which crossEntropy ignores). The lowest-NLL choice is the prediction:
//   - acc_norm  ranks by summed NLL divided by the choice's CHARACTER length
//   - acc       ranks by summed NLL (raw log-likelihood)
// Character length, not token count, is what lm-eval-harness normalizes by
// (`completion_len = np.array([float(len(i)) for i in choices])`), and it is the
// tokenizer-independent choice: per-token normalization makes the metric depend
// on how a given vocab happens to split the ending. Matching it is what lets
// these numbers sit next to the Open LLM Leaderboard figures.
//
// --shots defaults to 0; the Open LLM Leaderboard uses 25 for ARC and 10 for
// HellaSwag, so compare like with like before reading anything into a gap.
//
// A full run takes thousands of forwards, so run it when the trainer is idle; a
// small --limit is enough to smoke the pipeline.

import { readFileBytes } from "../io.ts";
import { loadModelFromGGUF } from "../export/load-gguf.ts";
import {
  freezeForScoring,
  lossChunkModelError,
  lossChunkValueError,
  sequenceLoss,
} from "../train/loss.ts";
import type { LanguageModel } from "../model/arch.ts";
import type { BPETokenizer } from "../tokenizer/bpe.ts";
import { requireGPU } from "../backend/webgpu.ts";
import type { WebGPUBackend } from "../backend/webgpu.ts";
import { fetchParquetUrls } from "../data/hf.ts";
import { parseDataFile, type Row } from "../data/parse.ts";
import type { Command, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";

function die(msg: string): never {
  throw new UsageError(msg);
}

interface MCItem {
  context: string; // the stem/prompt
  choices: string[]; // candidate completions
  gold: number; // index of the correct choice
}

/** One dataset: where to fetch it and how to turn a parsed row into an MCItem. */
interface Task {
  id: string;
  config: string;
  split: string;
  toItem(row: Row): MCItem | null;
  /** Render the "prompt + answer" text for a choice (few-shot exemplars reuse it). */
  render(ctx: string, choice: string): string;
  /** Chance accuracy, for reading a result: 4-option tasks 25, 2-option PIQA 50. */
  chance: number;
  /** Replaces the HF-parquet fetch for a dataset the datasets-server cannot convert. */
  load?(limit: number): Promise<Row[]>;
}

/** ARC choices arrive as a { text: [...], label: [...] } struct; answerKey is a label. */
function arcItem(row: Row): MCItem | null {
  const q = row["question"];
  // deno-lint-ignore no-explicit-any
  const ch = row["choices"] as any;
  const key = row["answerKey"];
  if (typeof q !== "string" || !ch || typeof key !== "string") return null;
  const texts: string[] = Array.isArray(ch.text) ? ch.text : [];
  const labels: string[] = Array.isArray(ch.label) ? ch.label : [];
  const gold = labels.indexOf(key);
  if (texts.length < 2 || gold < 0) return null;
  return { context: q, choices: texts, gold };
}

/**
 * HellaSwag ships raw WikiHow/ActivityNet markup, and every published number
 * for it is measured after lm-eval-harness's cleanup, not before: " [title]"
 * becomes a sentence break, any other bracketed span is dropped, and the double
 * spaces that leaves collapse.
 */
export function hellaswagPreprocess(text: string): string {
  return text.trim()
    .replaceAll(" [title]", ". ")
    .replace(/\[.*?\]/g, "")
    .replaceAll("  ", " ");
}

/**
 * The query is the activity label plus the two context halves, not the bare
 * `ctx` field: without the label the model loses the topic the ending has to be
 * plausible for, and the score is no longer the one the leaderboards report.
 */
export function hellaswagItem(row: Row): MCItem | null {
  const label = row["activity_label"];
  const ctxA = row["ctx_a"] ?? row["ctx"];
  const ctxB = row["ctx_b"] ?? "";
  // deno-lint-ignore no-explicit-any
  const endings = row["endings"] as any;
  const gold = typeof row["label"] === "number" ? row["label"] : parseInt(String(row["label"]), 10);
  if (typeof label !== "string" || typeof ctxA !== "string" || typeof ctxB !== "string") {
    return null;
  }
  if (!Array.isArray(endings)) return null;
  if (!Number.isInteger(gold) || gold < 0 || gold >= endings.length) return null;
  // Python's str.capitalize() also lowercases the tail, and the reference
  // implementation's output is what the published numbers were measured on.
  const capB = ctxB.charAt(0).toUpperCase() + ctxB.slice(1).toLowerCase();
  const query = hellaswagPreprocess(`${label}: ${ctxA} ${capB}`);
  return { context: query, choices: endings.map((e) => hellaswagPreprocess(String(e))), gold };
}

/** PIQA: a goal and two candidate solutions, gold given by a 0/1 label. */
export function piqaItem(row: Row): MCItem | null {
  const goal = row["goal"], sol1 = row["sol1"], sol2 = row["sol2"];
  const gold = Number(row["label"]);
  if (typeof goal !== "string" || typeof sol1 !== "string" || typeof sol2 !== "string") return null;
  if (gold !== 0 && gold !== 1) return null;
  return { context: goal, choices: [sol1, sol2], gold };
}

/**
 * Join PIQA's questions to its gold labels, which ship as a parallel file of one
 * integer per line aligned by position. A length mismatch means every label is
 * against the wrong question, which would still score and would look like a
 * plausible near-chance result, so it aborts instead.
 */
export function attachPiqaLabels(rows: Row[], labelsText: string): Row[] {
  const labels = labelsText.split("\n").map((l) => l.trim()).filter((l) => l.length > 0);
  if (labels.length !== rows.length) {
    die(
      `piqa: ${rows.length} questions but ${labels.length} labels, refusing to score misaligned data`,
    );
  }
  return rows.map((r, i) => ({ ...r, label: Number(labels[i]) }));
}

// PIQA's canonical dataset ships a Python loading script, so the HF
// datasets-server never converts it to parquet and fetchParquetUrls finds
// nothing for it. Read the authors' own distribution, which is the same thing
// that script downloads, rather than a third-party mirror whose contents we
// would have to take on trust.
const PIQA_BASE = "https://yonatanbisk.com/piqa/data";

async function loadPiqa(limit: number): Promise<Row[]> {
  const get = async (name: string): Promise<Uint8Array> => {
    const resp = await fetch(`${PIQA_BASE}/${name}`);
    if (!resp.ok) die(`fetch ${name} -> ${resp.status} ${resp.statusText}`);
    return new Uint8Array(await resp.arrayBuffer());
  };
  const rows = await parseDataFile("valid.jsonl", await get("valid.jsonl"));
  const labelled = attachPiqaLabels(rows, new TextDecoder().decode(await get("valid-labels.lst")));
  return limit ? labelled.slice(0, limit) : labelled;
}

const ARC_RENDER = (c: string, a: string) => `Question: ${c}\nAnswer: ${a}`;

export const TASKS: Record<string, Task> = {
  arc: {
    id: "allenai/ai2_arc",
    config: "ARC-Challenge",
    split: "test",
    toItem: arcItem,
    render: ARC_RENDER,
    chance: 25,
  },
  "arc-easy": {
    id: "allenai/ai2_arc",
    config: "ARC-Easy",
    split: "test",
    toItem: arcItem,
    render: ARC_RENDER,
    chance: 25,
  },
  hellaswag: {
    id: "Rowan/hellaswag",
    config: "default",
    split: "validation",
    toItem: hellaswagItem,
    render: (c, a) => `${c} ${a}`,
    chance: 25,
  },
  piqa: {
    id: "ybisk/piqa",
    config: "plain_text",
    split: "validation",
    toItem: piqaItem,
    render: ARC_RENDER,
    chance: 50,
    load: loadPiqa,
  },
};

async function loadRows(task: Task, limit: number): Promise<Row[]> {
  if (task.load) return task.load(limit);
  const urls = await fetchParquetUrls(task.id, task.config, task.split);
  if (!urls.length) die(`no parquet for ${task.id} [${task.config}/${task.split}]`);
  const rows: Row[] = [];
  for (const url of urls) {
    const resp = await fetch(url);
    if (!resp.ok) die(`fetch ${url} -> ${resp.status}`);
    const part = await parseDataFile(url, new Uint8Array(await resp.arrayBuffer()));
    for (const r of part) {
      rows.push(r);
      if (limit && rows.length >= limit) return rows;
    }
  }
  return rows;
}

/**
 * The target index the choice span starts at, measured on the window AFTER
 * truncation. Deriving it from the untruncated context length instead masks the
 * wrong span once context plus choice passes maxSeq: the score then covers only
 * the tail of the choice, or none of it.
 */
export function choiceMaskStart(nCtx: number, nChoice: number, maxSeq: number): number {
  return Math.min(nCtx + nChoice, maxSeq) - nChoice - 1;
}

/** The few-shot preamble joined to an item's stem, defined once so the preflight
 * and the scoring loop cannot disagree about the separator. */
export function withPreamble(preamble: string, it: MCItem): string {
  return preamble ? `${preamble}\n\n${it.context}` : it.context;
}

/** The exact strings `choiceNLL` will score, so the preflight cannot check others. */
export function renderPair(
  render: (ctx: string, choice: string) => string,
  ctx: string,
  choice: string,
): { ctxOnly: string; choiceText: string } {
  const ctxOnly = render(ctx, "").replace(/\s+$/, "");
  return { ctxOnly, choiceText: render(ctx, choice).slice(ctxOnly.length) };
}

const UTF8 = new TextEncoder();

/**
 * Refuse an unscoreable item before the first forward, without a tokenizer.
 *
 * `choiceWindowError` needs token counts, and encoding every pair up front costs
 * 6.7 s on a full HellaSwag set, measured. It is not needed for almost any of
 * them, because this repo's BPE is byte-level: every token covers at least one
 * UTF-8 byte, so a rendered choice under `maxSeq` BYTES cannot reach `maxSeq`
 * tokens, and a non-empty stem encodes to at least one token.
 *
 * Bytes, not characters. `String.length` counts UTF-16 units and does not bound
 * the token count: a character whose bytes no merge covers decomposes to one
 * token per byte, so a single three-byte character can be three tokens. Measured
 * against the 151936-entry Qwen3 vocab in `data/lambrp-hold.tokenizer.json`, one
 * `\u2e3b` is one character and two tokens. Only the byte count holds, and it
 * holds for every vocab by construction: GPT2_SPLIT partitions without overlap,
 * each pre-token starts at one symbol per byte, bpeWord only shortens, and an
 * unknown id only drops.
 *
 * That last clause is also why a broken premise costs earliness rather than
 * correctness. This refuses only an empty string, which is zero tokens under
 * every vocab, so anything it rejects `choiceNLL` would reject too; and every
 * way the bound could fail makes it pass a pair the scorer still checks with
 * real token counts.
 *
 * Returns the pairs that still have to be checked exactly, usually none.
 */
export function preflightByBytes(
  pairs: { ctxOnly: string; choiceText: string }[],
  maxSeq: number,
): { needExactCheck: { ctxOnly: string; choiceText: string }[]; empty: "stem" | "choice" | null } {
  const needExactCheck: { ctxOnly: string; choiceText: string }[] = [];
  let empty: "stem" | "choice" | null = null;
  for (const p of pairs) {
    if (empty === null && p.ctxOnly.length === 0) empty = "stem";
    if (empty === null && p.choiceText.length === 0) empty = "choice";
    if (UTF8.encode(p.choiceText).length >= maxSeq) needExactCheck.push(p);
  }
  return { needExactCheck, empty };
}

/**
 * Null when the window can score every choice token with at least one token in
 * front of it to predict the first one from. A negative start means truncation
 * would eat choice tokens, and a short score is not comparable to a full one:
 * the summed NLL drops while acc_norm still divides by the whole choice, so the
 * truncated option wins on a discount it did not earn.
 */
export function choiceWindowError(nCtx: number, nChoice: number, maxSeq: number): string | null {
  if (nChoice < 1) return "a choice rendered to 0 tokens, so there is nothing to score";
  if (nCtx < 1) return "the stem rendered to 0 tokens, so nothing predicts the first choice token";
  if (choiceMaskStart(nCtx, nChoice, maxSeq) < 0) {
    return `a choice takes ${nChoice} tokens and needs one more of context, but this model's ` +
      `context is ${maxSeq}: scoring it would drop choice tokens and flatter that option`;
  }
  return null;
}

/** Summed negative log-likelihood of `choiceText` given `ctxText`, scored over
 * ONLY the choice tokens. */
async function choiceNLL(
  model: LanguageModel,
  tok: BPETokenizer,
  gpu: WebGPUBackend,
  ctxText: string,
  choiceText: string,
  lossChunk: number,
): Promise<number> {
  const ctxIds = tok.encode(ctxText);
  const chIds = tok.encode(choiceText);
  const bad = choiceWindowError(ctxIds.length, chIds.length, model.cfg.maxSeq);
  if (bad) die(`${bad}. The choice reads ${JSON.stringify(choiceText.slice(0, 60))}`);
  const full = [...ctxIds, ...chIds].slice(-model.cfg.maxSeq);
  const inputs = full.slice(0, -1);
  const targets = full.slice(1);
  // Keep only the choice tokens as targets; -1 (ignored) everywhere else. The
  // first choice token is predicted from the token before it, so the boundary
  // sits one index earlier than the choice does.
  const firstChoiceTgt = choiceMaskStart(ctxIds.length, chIds.length, model.cfg.maxSeq);
  for (let i = 0; i < firstChoiceTgt; i++) targets[i] = -1;
  const nChoice = targets.length - firstChoiceTgt;
  // Two invariants the summed NLL rests on, and each catches an edit the other
  // waves through. Both losses now refuse a targets array longer than the logit
  // rows, so this one's value is the message: it names the choice and the count
  // rather than the array length. And nChoice is what the mean is multiplied
  // back by, so it has to be the whole choice, which nothing else pins.
  if (targets.length !== inputs.length || nChoice !== chIds.length) {
    throw new Error(
      `mask does not line up: inputs ${inputs.length}, targets ${targets.length}, ` +
        `scored ${nChoice} of ${chIds.length} choice tokens`,
    );
  }
  // sequenceLoss returns the mean over kept rows, exactly as crossEntropy did,
  // so multiplying by the kept count still recovers the summed NLL whichever
  // path it took.
  const loss = sequenceLoss(model, inputs, targets, lossChunk);
  await gpu.sync([loss]);
  return loss.data[0] * nChoice;
}

/**
 * acc_norm's prediction: the choice with the lowest NLL per character of its own
 * text, measured before the render adds its delimiter. Normalizing per token
 * instead would make the metric depend on how the vocab splits each ending, so a
 * retokenized model would score differently on identical predictions.
 */
export function argminPerChar(sums: number[], choices: string[]): number {
  let best = 0;
  const per = (i: number) => sums[i] / Math.max(1, choices[i].length);
  for (let i = 1; i < sums.length; i++) if (per(i) < per(best)) best = i;
  return best;
}

async function run(v: Values) {
  const modelPath = v.str("model");
  const taskName = v.str("task");
  const limit = v.num("limit");
  const shots = v.num("shots");

  console.log(`=== eval-choice: ${taskName} on ${modelPath.split("/").pop()} ===`);
  const lossChunk = v.num("loss-chunk");
  // Before the checkpoint read: a typo'd width should not cost a multi-GB load.
  const badChunk = lossChunkValueError(lossChunk);
  if (badChunk) die(badChunk);
  const { model, tokenizer: tok, cfg } = loadModelFromGGUF(await readFileBytes(modelPath));
  // Scoring never runs backward, and a gradient buffer per parameter would be
  // allocated once and staged back to the host on every scored choice.
  freezeForScoring(model);
  const badForModel = lossChunkModelError(lossChunk, cfg.vocabSize, cfg.arch, model);
  if (badForModel) die(badForModel);

  const gpu = await requireGPU("eval-choice");
  console.log(`WebGPU adapter: ${gpu.adapterName}`);
  gpu.install();
  gpu.uploadParams(model.params());

  try {
    if (taskName === "ppl") {
      // Token perplexity over a text sample: reuse arc questions as generic text
      // if no corpus given, but ppl is most useful on held-out corpus; here we
      // just expose the hook. (Full ppl harness: feed a .txt via --limit lines.)
      die("ppl task: run against a held-out corpus; not wired to a default source yet");
    }
    const task = TASKS[taskName] ??
      die(`unknown task "${taskName}" (${Object.keys(TASKS).join("|")})`);
    const rows = await loadRows(task, limit ? limit + shots : 0);
    const items = rows.map((r) => task.toItem(r)).filter((x): x is MCItem => x !== null);
    if (items.length < shots + 1) die(`too few parsed items (${items.length}) for ${shots}-shot`);

    // Few-shot preamble: the first `shots` items rendered with their gold answer.
    const shotItems = items.slice(0, shots);
    const evalItems = items.slice(shots, limit ? shots + limit : undefined);
    const preamble = shotItems.map((s) => task.render(s.context, s.choices[s.gold])).join("\n\n");

    // Before any forward. choiceNLL refuses an unscoreable window, but it does
    // it on the item that holds one, which on a full set is hours in. The BYTE
    // bound settles almost every pair without a tokenizer; only a choice whose
    // bytes reach the context length needs encoding.
    //
    // Per item rather than over the whole grid: at 10-shot the preamble repeats
    // in every pair's ctxOnly, and choiceText is a slice that retains its parent
    // string, so materializing 40168 of them at once holds hundreds of MB the
    // streaming loop below never does.
    let total = 0, encoded = 0;
    for (const it of evalItems) {
      const ctx = withPreamble(preamble, it);
      const pairs = it.choices.map((ch) => renderPair(task.render, ctx, ch));
      const pre = preflightByBytes(pairs, cfg.maxSeq);
      if (pre.empty) {
        const which = pre.empty;
        const p = pairs.find((q) => (which === "stem" ? q.ctxOnly : q.choiceText).length === 0)!;
        die(
          `${taskName}: an item's ${which} rendered to nothing, so it cannot be scored. ` +
            // The stem's TAIL: at any --shots its head is the preamble's opening,
            // identical for every item, so the head would identify nothing.
            `The pair reads ${
              JSON.stringify(`${p.ctxOnly.slice(-60)}|${p.choiceText.slice(0, 40)}`)
            }`,
        );
      }
      for (const p of pre.needExactCheck) {
        const bad = choiceWindowError(
          tok.encode(p.ctxOnly).length,
          tok.encode(p.choiceText).length,
          cfg.maxSeq,
        );
        if (bad) {
          die(`${taskName}: ${bad}. The choice reads ${JSON.stringify(p.choiceText.slice(0, 60))}`);
        }
      }
      total += pairs.length;
      encoded += pre.needExactCheck.length;
    }
    console.log(
      `Windows: none of ${total} choices exceeds the ${cfg.maxSeq}-token context ` +
        `(${encoded} needed encoding) \u2713`,
    );

    let correctNorm = 0, correctRaw = 0, done = 0;
    const t0 = Date.now();
    for (const it of evalItems) {
      const ctx = withPreamble(preamble, it);
      const sums: number[] = [];
      for (const ch of it.choices) {
        // Through renderPair, the same helper the preflight used: if the two
        // sites drifted, the preflight would be vouching for strings that are
        // not the ones scored.
        const { ctxOnly, choiceText } = renderPair(task.render, ctx, ch);
        sums.push(await choiceNLL(model, tok, gpu, ctxOnly, choiceText, lossChunk));
      }
      const bestNorm = argminPerChar(sums, it.choices);
      let bestRaw = 0;
      for (let i = 1; i < sums.length; i++) if (sums[i] < sums[bestRaw]) bestRaw = i;
      if (bestNorm === it.gold) correctNorm++;
      if (bestRaw === it.gold) correctRaw++;
      done++;
      if (done % 50 === 0) console.log(`  ${done}/${evalItems.length} …`);
    }
    const pct = (n: number) => (100 * n / Math.max(1, done)).toFixed(2);
    console.log(
      `\n${taskName} (${done} items, ${shots}-shot, chance ${task.chance}): ` +
        `acc_norm ${pct(correctNorm)}%  acc ${pct(correctRaw)}%  ` +
        `(${((Date.now() - t0) / 1000).toFixed(0)}s)`,
    );
  } finally {
    gpu.uninstall();
  }
}

export const evalChoiceCommand: Command = {
  name: "eval-choice",
  summary: "Score a model on a multiple-choice benchmark (ARC, HellaSwag, PIQA) or perplexity.",
  details: `Runs the trainer's own forward pass, so it scores a GGUF without llama.cpp. Each
option is scored by its length-normalized log-likelihood and the highest wins, which is the
standard multiple-choice protocol.

Expect low absolute numbers from a small model. Chance is 25.0 on arc, arc-easy and hellaswag,
and 50.0 on piqa, which has two options. Use it to compare checkpoints of the same model, where
the trend is the signal.

The arc task is ARC-Challenge and arc-easy is ARC-Easy. Those four tasks are the components of the
Open SLM Leaderboard's Intelligence Index, so running all four makes a checkpoint directly
comparable to that board.`,
  examples: [
    "eval-choice --model model.gguf --task hellaswag --limit 500",
    "eval-choice --model model.gguf --task arc",
    "eval-choice --model model.gguf --task piqa",
  ],
  flags: [
    {
      name: "model",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "the GGUF to score",
    },
    {
      name: "task",
      type: "string",
      default: "arc",
      choices: ["arc", "arc-easy", "hellaswag", "piqa", "ppl"],
      describe: "benchmark to run",
    },
    {
      name: "limit",
      type: "number",
      default: 0,
      describe: "stop after N examples (0 = the whole set); use a few hundred for a quick read",
    },
    {
      name: "shots",
      type: "number",
      default: 0,
      describe: "few-shot examples prepended to each question",
    },
    {
      name: "loss-chunk",
      type: "number",
      default: 0,
      placeholder: "N",
      describe:
        "stream the readout and the loss in vocab chunks of N instead of materializing [seq-len, vocab] logits: the way to score a large-vocab model at long context (0 = dense)",
    },
  ],
  run: run,
};
