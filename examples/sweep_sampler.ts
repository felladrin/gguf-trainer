// Sampler-preset sweep for a pretraining checkpoint: the milestone tool that (re)derives the
// SAMPLER preset used by examples/eval_completions.sh. It automates the manual sweep that produced
// preset "D": generate the fixed prompt battery under a pool of llama.cpp sampler configs, score
// every completion locally for repetition collapse (THE failure mode of a mid-training base
// model), optionally score coherence with an LLM judge, run one perturbation round around the
// winner (explore-then-exploit in miniature), and print a ranked table plus a paste-ready
// SAMPLER=(...) line.
//
// This is a MILESTONE tool, not a per-checkpoint step: eval_completions.sh keeps its preset
// FROZEN between milestones so successive checkpoints stay comparable ("judge the model, not the
// sampler"). Re-run at milestones (end of a phase, after each curriculum stage), paste the winner
// into eval_completions.sh, and note the step it was tuned at.
//
//   deno run -A examples/sweep_sampler.ts <model.gguf> [--n=128] [--seeds=1] [--only=D,U]
//                                         [--port=8137]
//     --n=N      new tokens per completion (default 128 — longer than the battery's 60 on
//                purpose: loops mostly show late in a generation)
//     --seeds=N  seeds per prompt, 42..42+N-1 (default 1; raise to average out sampling luck)
//     --only=..  comma-separated config names to restrict the starting pool (the perturbation
//                round still runs around whatever wins)
//     --port=N   port for the spawned llama-server (default 8137)
//
// Scoring
//   rep   : mean of seq-rep-2 and seq-rep-3 over words (1 - unique n-grams / n-grams); 0 = no
//           repeated n-grams, 1 = a total loop. Catches "fresh fresh fresh..." collapse; it
//           CANNOT see topic drift.
//   judge : optional 0-10 continuation quality from any OpenAI-compatible chat endpoint:
//             JUDGE_URL=http://host:port/v1  [JUDGE_MODEL=name]  [JUDGE_API_KEY=...]
//           e.g. a bigger local model under llama-server (use a plain instruct model, not a
//           reasoning one — the reply must start with the number). With a judge, ranking is by
//           judge score (rep as tie-break). Without one, ranking is fewest-loops with a
//           tie-break toward the gentlest sampler (least logit distortion) among near-ties, and
//           topic drift is NOT measured — eyeball the printed samples before adopting a winner.
//
// Requires llama-server (llama.cpp) on PATH; override with LLAMA_SERVER=/path/to/llama-server.
// The model is loaded ONCE by one spawned server; configs are swept via per-request sampling
// params, so a full sweep costs minutes, not hours.

export interface SamplerCfg {
  name: string;
  temp: number;
  topP: number;
  topK: number;
  minP: number;
  presencePenalty: number;
  repeatPenalty: number;
  repeatLastN: number;
}

/** Aggregate scores rank() needs; Result (below) carries the rest. */
export interface CfgScore {
  cfg: SamplerCfg;
  /** mean of (seq-rep-2 + seq-rep-3)/2 across generations; lower is better */
  rep: number;
  /** mean judge score 0-10 across judged generations; null when no judge ran */
  judge: number | null;
  /** how many generations the judge scored successfully */
  judged: number;
}

function mk(
  name: string,
  temp: number,
  topP: number,
  topK: number,
  minP: number,
  presencePenalty: number,
  repeatPenalty: number,
  repeatLastN: number,
): SamplerCfg {
  return { name, temp, topP, topK, minP, presencePenalty, repeatPenalty, repeatLastN };
}

// Starting pool. Anchored by the two presets eval_completions.sh documents (D incumbent, U
// high-variety alternative) plus baselines that expose loop collapse (G greedy, N naive),
// repeat-penalty-only variants (RP3 was the pre-sweep default), gentler/heavier and
// cooler/warmer takes on D, and the pure-min-p school (M, MP). Keep D/U in sync with
// eval_completions.sh.
//                       name    temp  top-p top-k min-p  pres  rep-p  rln
export const POOL: SamplerCfg[] = [
  mk("G", 0, 1, 0, 0, 0, 1, 64),
  mk("N", 0.7, 0.95, 40, 0, 0, 1, 64),
  mk("D", 0.7, 0.85, 30, 0.02, 0.4, 1.15, 128),
  mk("U", 0.7, 0.8, 20, 0, 1.5, 1, 64),
  mk("RP3", 0.7, 0.85, 30, 0.02, 0, 1.3, 128),
  mk("RP15", 0.7, 0.85, 30, 0.02, 0, 1.15, 128),
  mk("DL", 0.7, 0.85, 30, 0.02, 0.2, 1.1, 128),
  mk("DH", 0.7, 0.85, 30, 0.02, 0.6, 1.2, 128),
  mk("DC", 0.5, 0.85, 30, 0.02, 0.4, 1.15, 128),
  mk("DW", 0.9, 0.85, 30, 0.02, 0.4, 1.15, 128),
  mk("M", 0.8, 1, 0, 0.1, 0, 1, 64),
  mk("MP", 0.8, 1, 0, 0.1, 0.3, 1.1, 128),
];

// Same battery as eval_completions.sh (keep in sync): narrative, descriptive, everyday, factual
// recall, science, abstraction/list, dialogue, arithmetic.
const PROMPTS: string[] = [
  "Once upon a time, there was a little",
  "The old man walked slowly toward the",
  "In the morning, she went to the store to buy",
  "The sun rose over the mountains and",
  "The capital of France is",
  "Water is made of",
  "The three most important things in life are",
  "Scientists have recently discovered that",
  '"Hello," said Tom, "I want to',
  "2 + 2 =",
];

/** seq-rep-n over words: 1 - unique n-grams / n-grams. Empty output counts as fully degenerate. */
export function seqRepN(text: string, n: number): number {
  const words = text.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? [];
  if (words.length === 0) return 1;
  if (words.length < n + 1) return 0;
  const grams = new Set<string>();
  let total = 0;
  for (let i = 0; i + n <= words.length; i++) {
    grams.add(words.slice(i, i + n).join(" "));
    total++;
  }
  return 1 - grams.size / total;
}

export function repScore(text: string): number {
  return (seqRepN(text, 2) + seqRepN(text, 3)) / 2;
}

/**
 * How hard the config bends the model's own distribution. Penalties are the drift-causers (per
 * the sweep that picked D); truncation (top-p/k, min-p) and moderate temperature are not. Used
 * only as a tie-break: among configs whose rep scores are within noise of each other, prefer the
 * one that distorts least, so we keep judging the model, not the sampler.
 */
export function distortion(c: SamplerCfg): number {
  return (c.repeatPenalty - 1) + 0.5 * c.presencePenalty;
}

/** Identity of a config up to its name (for deduping perturbations against configs already run). */
export function cfgKey(c: SamplerCfg): string {
  return [c.temp, c.topP, c.topK, c.minP, c.presencePenalty, c.repeatPenalty, c.repeatLastN]
    .join("|");
}

/** First number in the judge's reply, accepted only if it is a plausible 0-10 score. */
export function parseJudgeScore(reply: string): number | null {
  const m = reply.match(/\d+(\.\d+)?/);
  if (!m) return null;
  const v = Number(m[0]);
  return v >= 0 && v <= 10 ? v : null;
}

/**
 * Judged: rank by judge score, rep as tie-break. Unjudged: rank by rep, but treat rep gaps
 * within `eps` of the best as noise and prefer the gentlest sampler among those near-ties.
 */
export function rank<T extends CfgScore>(rs: T[], eps = 0.02): T[] {
  if (rs.length === 0) return [...rs];
  const sorted = [...rs];
  if (sorted.some((r) => r.judge !== null)) {
    sorted.sort((a, b) =>
      ((b.judge ?? -1) - (a.judge ?? -1)) || (a.rep - b.rep) ||
      (distortion(a.cfg) - distortion(b.cfg))
    );
    return sorted;
  }
  sorted.sort((a, b) => (a.rep - b.rep) || (distortion(a.cfg) - distortion(b.cfg)));
  const best = sorted[0].rep;
  const near = sorted.filter((r) => r.rep - best <= eps);
  near.sort((a, b) => (distortion(a.cfg) - distortion(b.cfg)) || (a.rep - b.rep));
  return [...near, ...sorted.filter((r) => r.rep - best > eps)];
}

function r2(x: number): number {
  return Math.round(x * 100) / 100;
}
function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** One exploit round: small steps on the axes that matter, clamped, deduped against `seen`. */
export function neighbors(c: SamplerCfg, seen: Set<string>): SamplerCfg[] {
  const out: SamplerCfg[] = [];
  const add = (tag: string, patch: Partial<SamplerCfg>): void => {
    const n: SamplerCfg = { ...c, ...patch, name: `${c.name}~${tag}` };
    const key = cfgKey(n);
    if (seen.has(key)) return;
    seen.add(key);
    out.push(n);
  };
  add("t-", { temp: r2(clamp(c.temp - 0.1, 0.2, 1.2)) });
  add("t+", { temp: r2(clamp(c.temp + 0.1, 0.2, 1.2)) });
  add("pp-", { presencePenalty: r2(clamp(c.presencePenalty - 0.2, 0, 1.5)) });
  add("pp+", { presencePenalty: r2(clamp(c.presencePenalty + 0.2, 0, 1.5)) });
  add("rp-", { repeatPenalty: r2(clamp(c.repeatPenalty - 0.05, 1, 1.4)) });
  add("rp+", { repeatPenalty: r2(clamp(c.repeatPenalty + 0.05, 1, 1.4)) });
  return out;
}

/** The line eval_completions.sh wants, verbatim. */
export function samplerLine(c: SamplerCfg): string {
  return `SAMPLER=(--temp ${c.temp} --top-p ${c.topP} --top-k ${c.topK} --min-p ${c.minP} ` +
    `--presence-penalty ${c.presencePenalty} --repeat-penalty ${c.repeatPenalty} ` +
    `--repeat-last-n ${c.repeatLastN})`;
}

// ───────────────────────── runtime below (Deno-only; spawns llama-server) ─────────────────────

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  return g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
}
function die(msg: string): never {
  console.error("sweep_sampler: " + msg);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
  throw new Error(msg);
}
function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

interface Gen {
  prompt: string;
  seed: number;
  text: string;
  rep: number;
  judge: number | null;
}
interface Result extends CfgScore {
  rep2: number;
  rep3: number;
  gens: Gen[];
}
interface JudgeEnv {
  url: string;
  model: string;
  key: string | null;
}

function judgeEnvFromEnv(): JudgeEnv | null {
  const url = Deno.env.get("JUDGE_URL");
  if (!url) return null;
  return {
    url: url.replace(/\/+$/, ""),
    model: Deno.env.get("JUDGE_MODEL") ?? "default",
    key: Deno.env.get("JUDGE_API_KEY") ?? null,
  };
}

const RUBRIC = "You grade completions from a very small (~100M parameter) BASE language model " +
  "that is mid-pretraining. Score 0-10 how well COMPLETION continues PROMPT: reward fluent " +
  "grammar, staying on the prompt's topic, and natural progression; punish repetition loops, " +
  "word salad, and abrupt topic drift. Do NOT punish missing factual knowledge or simplistic " +
  "content — judge only text quality. Reply with ONLY the number.";

async function judgeOne(env: JudgeEnv, prompt: string, completion: string): Promise<number | null> {
  const headers: Record<string, string> = { "content-type": "application/json" };
  if (env.key) headers["authorization"] = `Bearer ${env.key}`;
  const res = await fetch(`${env.url}/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: env.model,
      temperature: 0,
      max_tokens: 8,
      messages: [
        { role: "system", content: RUBRIC },
        { role: "user", content: `PROMPT:\n${prompt}\n\nCOMPLETION:\n${completion}` },
      ],
    }),
    signal: AbortSignal.timeout(60_000),
  });
  if (!res.ok) die(`judge HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const j: unknown = await res.json();
  const content = (j as { choices?: { message?: { content?: unknown } }[] })
    .choices?.[0]?.message?.content;
  return typeof content === "string" ? parseJudgeScore(content) : null;
}

async function complete(
  port: number,
  cfg: SamplerCfg,
  prompt: string,
  seed: number,
  n: number,
): Promise<string> {
  const res = await fetch(`http://127.0.0.1:${port}/completion`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      prompt,
      n_predict: n,
      seed,
      temperature: cfg.temp,
      top_p: cfg.topP,
      top_k: cfg.topK,
      min_p: cfg.minP,
      presence_penalty: cfg.presencePenalty,
      repeat_penalty: cfg.repeatPenalty,
      repeat_last_n: cfg.repeatLastN,
      cache_prompt: false,
    }),
    signal: AbortSignal.timeout(180_000),
  });
  if (!res.ok) die(`/completion HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const j: unknown = await res.json();
  const content = (j as { content?: unknown }).content;
  if (typeof content !== "string") die("/completion response has no string .content");
  return content;
}

async function evalCfg(
  port: number,
  cfg: SamplerCfg,
  seedList: number[],
  n: number,
  judge: JudgeEnv | null,
): Promise<Result> {
  const gens: Gen[] = [];
  for (const prompt of PROMPTS) {
    for (const seed of seedList) {
      const text = await complete(port, cfg, prompt, seed, n);
      gens.push({
        prompt,
        seed,
        text,
        rep: repScore(text),
        judge: judge ? await judgeOne(judge, prompt, text) : null,
      });
    }
  }
  const judged = gens.filter((g): g is Gen & { judge: number } => g.judge !== null);
  return {
    cfg,
    rep: mean(gens.map((g) => g.rep)),
    rep2: mean(gens.map((g) => seqRepN(g.text, 2))),
    rep3: mean(gens.map((g) => seqRepN(g.text, 3))),
    judge: judged.length ? mean(judged.map((g) => g.judge)) : null,
    judged: judged.length,
    gens,
  };
}

interface Server {
  proc: Deno.ChildProcess;
  logPath: string;
}

async function startServer(model: string, port: number): Promise<Server> {
  const bin = Deno.env.get("LLAMA_SERVER") ?? "llama-server";
  const logPath = await Deno.makeTempFile({ prefix: "sweep_sampler_server_", suffix: ".log" });
  const logFile = await Deno.open(logPath, { write: true, truncate: true });
  let proc: Deno.ChildProcess;
  try {
    proc = new Deno.Command(bin, {
      args: ["-m", model, "-c", "1024", "--host", "127.0.0.1", "--port", String(port)],
      stdin: "null",
      stdout: "null",
      stderr: "piped",
    }).spawn();
  } catch (e) {
    logFile.close();
    if (e instanceof Deno.errors.NotFound) {
      die(`'${bin}' not found on PATH — install llama.cpp or set LLAMA_SERVER=/path/to/it`);
    }
    throw e;
  }
  proc.stderr.pipeTo(logFile.writable).catch(() => {});
  let exited: Deno.CommandStatus | null = null;
  proc.status.then((s) => {
    exited = s;
  });
  const deadline = Date.now() + 120_000;
  while (Date.now() < deadline && exited === null) {
    try {
      const r = await fetch(`http://127.0.0.1:${port}/health`, {
        signal: AbortSignal.timeout(1_000),
      });
      await r.body?.cancel();
      if (r.ok) return { proc, logPath };
    } catch {
      // not listening yet
    }
    await new Promise((res) => setTimeout(res, 250));
  }
  const tail = (await Deno.readTextFile(logPath)).split("\n").slice(-15).join("\n");
  die(
    `llama-server ${
      exited !== null
        ? `exited early (code ${(exited as Deno.CommandStatus).code})`
        : "not healthy after 120s"
    } — log tail:\n${tail}`,
  );
}

function fmtRow(r: Result, rankNo: number): string {
  const c = r.cfg;
  return [
    String(rankNo).padStart(4),
    "  ",
    c.name.padEnd(9),
    String(c.temp).padStart(5),
    String(c.topP).padStart(7),
    String(c.topK).padStart(7),
    String(c.minP).padStart(7),
    String(c.presencePenalty).padStart(6),
    String(c.repeatPenalty).padStart(7),
    String(c.repeatLastN).padStart(5),
    "  ",
    r.rep2.toFixed(3).padStart(6),
    r.rep3.toFixed(3).padStart(6),
    r.rep.toFixed(3).padStart(6),
    (r.judge === null ? "-" : r.judge.toFixed(1)).padStart(7),
  ].join("");
}

function printTable(rs: Result[]): void {
  console.log(
    [
      "rank".padStart(4),
      "  ",
      "config".padEnd(9),
      "temp".padStart(5),
      "top-p".padStart(7),
      "top-k".padStart(7),
      "min-p".padStart(7),
      "pres".padStart(6),
      "rep-p".padStart(7),
      "rln".padStart(5),
      "  ",
      "rep2".padStart(6),
      "rep3".padStart(6),
      "rep".padStart(6),
      "judge".padStart(7),
    ].join(""),
  );
  rs.forEach((r, i) => console.log(fmtRow(r, i + 1)));
}

async function dumpAll(rs: Result[], model: string, n: number): Promise<string> {
  const path = await Deno.makeTempFile({ prefix: "sweep_sampler_", suffix: ".txt" });
  const parts: string[] = [`sweep_sampler transcripts  model=${model}  n=${n}\n`];
  for (const r of rs) {
    const js = r.judge === null ? "" : `  judge=${r.judge.toFixed(1)}`;
    parts.push(`\n════ ${r.cfg.name}  rep=${r.rep.toFixed(3)}${js}\n${samplerLine(r.cfg)}\n`);
    for (const g of r.gens) {
      const gj = g.judge === null ? "" : `, judge=${g.judge}`;
      parts.push(`\n[seed ${g.seed}, rep=${g.rep.toFixed(3)}${gj}] ${g.prompt}\n${g.text}\n`);
    }
  }
  await Deno.writeTextFile(path, parts.join(""));
  return path;
}

async function main(): Promise<void> {
  let model = "";
  let n = 128;
  let seeds = 1;
  let port = 8137;
  let only: string[] | null = null;
  for (const a of args()) {
    if (a.startsWith("--n=")) n = Number(a.slice(4));
    else if (a.startsWith("--seeds=")) seeds = Number(a.slice(8));
    else if (a.startsWith("--port=")) port = Number(a.slice(7));
    else if (a.startsWith("--only=")) {
      only = a.slice(7).split(",").map((s) => s.trim()).filter(Boolean);
    } else if (a.startsWith("--")) die(`unknown flag '${a}' (see header for usage)`);
    else model = a;
  }
  if (!model) {
    die(
      "usage: deno run -A examples/sweep_sampler.ts <model.gguf> " +
        "[--n=128] [--seeds=1] [--only=D,U] [--port=8137]",
    );
  }
  if (!Number.isFinite(n) || n < 8) die("--n must be a number >= 8");
  if (!Number.isFinite(seeds) || seeds < 1) die("--seeds must be a number >= 1");
  if (!Number.isFinite(port) || port < 1 || port > 65535) die("--port must be a valid port");
  try {
    Deno.statSync(model);
  } catch {
    die(`no model at '${model}' — pass a checkpoint path (rsync one first, see header)`);
  }

  let pool = POOL;
  if (only) {
    const want = new Set(only.map((s) => s.toUpperCase()));
    pool = POOL.filter((c) => want.has(c.name.toUpperCase()));
    if (pool.length === 0) {
      die(`--only matched nothing; known configs: ${POOL.map((c) => c.name).join(",")}`);
    }
  }
  const judge = judgeEnvFromEnv();
  const seedList = Array.from({ length: seeds }, (_, i) => 42 + i);

  console.log(`sweep_sampler: model=${model}`);
  console.log(
    `  ${pool.length} configs x ${PROMPTS.length} prompts x ${seeds} seed(s), ` +
      `n=${n} tokens, judge=${judge ? judge.url : "off"}`,
  );

  const srv = await startServer(model, port);
  const onInt = (): void => {
    try {
      srv.proc.kill();
    } catch {
      // already gone
    }
    Deno.exit(130);
  };
  Deno.addSignalListener("SIGINT", onInt);
  try {
    const results: Result[] = [];
    const seen = new Set(pool.map(cfgKey));
    const run = async (cfg: SamplerCfg): Promise<void> => {
      const t0 = Date.now();
      const r = await evalCfg(port, cfg, seedList, n, judge);
      results.push(r);
      const js = r.judge === null
        ? ""
        : `  judge=${r.judge.toFixed(1)} (${r.judged}/${r.gens.length})`;
      const dt = ((Date.now() - t0) / 1000).toFixed(1);
      console.log(`  ${cfg.name.padEnd(8)} rep=${r.rep.toFixed(3)}${js}  ${dt}s`);
    };

    console.log("round 1: starting pool");
    for (const cfg of pool) await run(cfg);

    const winner1 = rank(results)[0].cfg;
    const extra = neighbors(winner1, seen);
    if (extra.length) {
      console.log(`round 2: perturbations around ${winner1.name}`);
      for (const cfg of extra) await run(cfg);
    }

    const final = rank(results);
    console.log("");
    printTable(final);
    if (judge) {
      const partial = final.filter((r) => r.judged < r.gens.length);
      if (partial.length) {
        console.log(
          `\nwarning: judge failed to score some generations (${
            partial.map((r) => `${r.cfg.name} ${r.judged}/${r.gens.length}`).join(", ")
          }) — means cover the judged subset only.`,
        );
      }
    } else {
      console.log(
        "\nno judge: ranking = fewest repetition loops, tie-break toward the gentlest sampler;" +
          "\ntopic drift is NOT measured. Eyeball the samples below (or set JUDGE_URL) before" +
          "\nadopting the winner.",
      );
    }
    const w = final[0];
    console.log(
      `\nwinner: ${w.cfg.name} — paste into examples/eval_completions.sh and note the step ` +
        "it was tuned at:",
    );
    console.log(samplerLine(w.cfg));
    console.log("\nsamples (top 3 configs, first 2 prompts, whitespace squeezed):");
    for (const r of final.slice(0, 3)) {
      for (const g of r.gens.filter((g) => g.seed === seedList[0]).slice(0, 2)) {
        const txt = g.text.replace(/\s+/g, " ").trim().slice(0, 160);
        console.log(`  [${r.cfg.name}] ${g.prompt} -> ${txt}`);
      }
    }
    console.log(`\nfull transcripts: ${await dumpAll(final, model, n)}`);
  } finally {
    Deno.removeSignalListener("SIGINT", onInt);
    try {
      srv.proc.kill();
    } catch {
      // already exited
    }
    await srv.proc.status;
  }
}

if (import.meta.main) await main();
