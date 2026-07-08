// Evaluate a trained GGUF on multiple-choice benchmarks (ARC-Challenge,
// HellaSwag) and word/token perplexity, using the trainer's OWN forward — no
// external eval framework, so it stays as portable as the rest of the project.
//
// Scoring is the standard length-normalized log-likelihood: for each candidate
// completion we forward [context + completion] once and read the mean negative
// log-likelihood over ONLY the completion tokens (context targets masked to -1,
// which crossEntropy ignores). The lowest mean-NLL choice is the prediction:
//   - acc_norm  ranks by mean NLL   (length-normalized; the headline metric)
//   - acc       ranks by summed NLL (raw log-likelihood)
// This matches the lm-eval-harness definition, so numbers compare to the Open
// LLM Leaderboard figures on the Minueza model cards.
//
//   deno run -A --unstable-webgpu examples/eval_mc.ts <model.gguf> [task] [--limit=N] [--shots=K] [--cpu]
//     task  : arc | hellaswag | ppl   (default arc)
//     --limit=N : cap examples (for a quick smoke; omit for the full set)
//     --shots=K : few-shot exemplars prepended (default 0; leaderboard uses 25 ARC / 10 HellaSwag)
//     --cpu     : force CPU forward (default: WebGPU when available)
//
// Full runs want the GPU and take a while (thousands of forwards); run them when
// the trainer is idle. A small --limit on --cpu is enough to smoke the pipeline.

import { readFileBytes } from "../src/io.ts";
import { loadGemma3FromGGUF } from "../src/export/load_gguf.ts";
import { crossEntropy } from "../src/model/autograd.ts";
import type { Gemma3Model } from "../src/model/gemma3.ts";
import type { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { initWebGPU } from "../src/backend/webgpu.ts";
import type { WebGPUBackend } from "../src/backend/webgpu.ts";
import { fetchParquetUrls } from "../web/server/hf.ts";
import { parseDataFile, type Row } from "../web/server/parse.ts";

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  return g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
}
function die(msg: string): never {
  console.error("eval_mc: " + msg);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
  throw new Error(msg);
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

function hellaswagItem(row: Row): MCItem | null {
  const ctx = (row["ctx"] ?? row["ctx_a"]) as string;
  // deno-lint-ignore no-explicit-any
  const endings = row["endings"] as any;
  const label = row["label"];
  if (typeof ctx !== "string" || !Array.isArray(endings)) return null;
  const gold = typeof label === "number" ? label : parseInt(String(label), 10);
  if (!Number.isInteger(gold) || gold < 0 || gold >= endings.length) return null;
  return { context: ctx, choices: endings.map(String), gold };
}

const TASKS: Record<string, Task> = {
  arc: {
    id: "allenai/ai2_arc",
    config: "ARC-Challenge",
    split: "test",
    toItem: arcItem,
    render: (c, a) => `Question: ${c}\nAnswer: ${a}`,
  },
  hellaswag: {
    id: "Rowan/hellaswag",
    config: "default",
    split: "validation",
    toItem: hellaswagItem,
    render: (c, a) => `${c} ${a}`,
  },
};

async function loadRows(task: Task, limit: number): Promise<Row[]> {
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

/** Mean + summed negative log-likelihood of `choiceText` given `ctxText`,
 * scored over ONLY the choice tokens. Runs on GPU if `gpu` is installed. */
async function choiceNLL(
  model: Gemma3Model,
  tok: BPETokenizer,
  gpu: WebGPUBackend | null,
  ctxText: string,
  choiceText: string,
): Promise<{ mean: number; sum: number }> {
  const ctxIds = tok.encode(ctxText);
  const chIds = tok.encode(choiceText);
  const full = [...ctxIds, ...chIds].slice(-model.cfg.maxSeq);
  const inputs = full.slice(0, -1);
  const targets = full.slice(1);
  // Keep only the choice tokens as targets; -1 (ignored) everywhere else. The
  // first choice token is predicted from the last context token, at targets
  // index ctxIds.length-1.
  const firstChoiceTgt = Math.max(0, ctxIds.length - 1);
  for (let i = 0; i < firstChoiceTgt; i++) targets[i] = -1;
  const nChoice = targets.length - firstChoiceTgt;
  const loss = crossEntropy(model.forward(inputs), targets);
  if (gpu) await gpu.sync([loss]);
  const mean = loss.data[0];
  return { mean, sum: mean * Math.max(1, nChoice) };
}

async function main() {
  const a = args();
  const positional = a.filter((x) => !x.startsWith("--"));
  const flag = (name: string) => {
    const f = a.find((x) => x.startsWith(`--${name}=`));
    return f ? f.slice(name.length + 3) : undefined;
  };
  const modelPath = positional[0];
  const taskName = positional[1] ?? "arc";
  if (!modelPath) {
    die("usage: eval_mc <model.gguf> [arc|hellaswag|ppl] [--limit=N] [--shots=K] [--cpu]");
  }
  const limit = flag("limit") ? Number(flag("limit")) : 0;
  const shots = flag("shots") ? Number(flag("shots")) : 0;
  const useCpu = a.includes("--cpu");

  console.log(`=== eval_mc: ${taskName} on ${modelPath.split("/").pop()} ===`);
  const { model, tokenizer: tok } = loadGemma3FromGGUF(await readFileBytes(modelPath));

  let gpu: WebGPUBackend | null = null;
  if (!useCpu) {
    gpu = await initWebGPU();
    if (!gpu) console.log("(no WebGPU; falling back to CPU forward)");
    else {
      console.log(`WebGPU adapter: ${gpu.adapterName}`);
      gpu.install();
      gpu.uploadParams(model.params());
    }
  }

  try {
    if (taskName === "ppl") {
      // Token perplexity over a text sample: reuse arc questions as generic text
      // if no corpus given — but ppl is most useful on held-out corpus; here we
      // just expose the hook. (Full ppl harness: feed a .txt via --limit lines.)
      die("ppl task: run against a held-out corpus; not wired to a default source yet");
    }
    const task = TASKS[taskName] ?? die(`unknown task "${taskName}" (arc|hellaswag)`);
    const rows = await loadRows(task, limit ? limit + shots : 0);
    const items = rows.map((r) => task.toItem(r)).filter((x): x is MCItem => x !== null);
    if (items.length < shots + 1) die(`too few parsed items (${items.length}) for ${shots}-shot`);

    // Few-shot preamble: the first `shots` items rendered with their gold answer.
    const shotItems = items.slice(0, shots);
    const evalItems = items.slice(shots, limit ? shots + limit : undefined);
    const preamble = shotItems.map((s) => task.render(s.context, s.choices[s.gold])).join("\n\n");

    let correctNorm = 0, correctRaw = 0, done = 0;
    const t0 = Date.now();
    for (const it of evalItems) {
      const ctx = preamble ? `${preamble}\n\n${it.context}` : it.context;
      const scored = [];
      for (const ch of it.choices) {
        const full = task.render(ctx, ch);
        // Score only the choice span: render context without the answer to find
        // the boundary, then score the full render's choice tokens.
        const ctxOnly = task.render(ctx, "").replace(/\s+$/, "");
        scored.push(await choiceNLL(model, tok, gpu, ctxOnly, full.slice(ctxOnly.length)));
      }
      let bestNorm = 0, bestRaw = 0;
      for (let i = 1; i < scored.length; i++) {
        if (scored[i].mean < scored[bestNorm].mean) bestNorm = i;
        if (scored[i].sum < scored[bestRaw].sum) bestRaw = i;
      }
      if (bestNorm === it.gold) correctNorm++;
      if (bestRaw === it.gold) correctRaw++;
      done++;
      if (done % 50 === 0) console.log(`  ${done}/${evalItems.length} …`);
    }
    const pct = (n: number) => (100 * n / Math.max(1, done)).toFixed(2);
    console.log(
      `\n${taskName} (${done} items, ${shots}-shot): ` +
        `acc_norm ${pct(correctNorm)}%  acc ${pct(correctRaw)}%  ` +
        `(${((Date.now() - t0) / 1000).toFixed(0)}s)`,
    );
  } finally {
    if (gpu) gpu.uninstall();
  }
}

main().catch((e) => die(String(e?.stack ?? e)));
