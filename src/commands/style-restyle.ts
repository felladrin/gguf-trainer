// `style-restyle`: rewrite a chat corpus's answers in a target voice.
//
// Drives the pi coding agent in print mode against whatever model pi is
// configured with, using the given style guide as the system prompt, and asks it to
// rewrite ONLY the assistant turns of each seeded conversation. User and system
// turns are passed through verbatim: the model must learn to ANSWER in that
// voice, not to be asked in it.
//
//   style-restyle --in corpus/style/seed.jsonl --out corpus/style/restyled.jsonl
//
// Resumable: every finished conversation is appended to --out as one JSON line,
// and a rerun skips ids already there. Rejects go to <out>.failed.jsonl with the
// reason, never silently into the training set.
//
// --jobs defaults to 1 on purpose. The style guide is a large system prompt and
// it is identical on every call, so a single sequential stream hits
// llama.cpp's prompt cache on all of it and only pays for the conversation.
// Parallel calls evict each other's cache and end up slower.

import { readFileText } from "../io.ts";
import type { ChatMessage } from "../data/chat.ts";
import type { Command, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";

const RULES = `
---

# Your job right now

You are rewriting the assistant replies of a training dataset so they read as if
the author of the style guide above wrote them. The guide is what changes; the
content is not yours to touch.

Hard rules:
- Preserve every fact, number, name, step, and the meaning of the answer. Add
  nothing: no opinions, anecdotes, hardware, employers, or project names.
- You ARE the assistant answering the user. Never mention the author, the style
  guide, or this task.
- Same language as the original (English). Lists stay lists. Refusals stay
  refusals.
- Keep roughly the original length (0.6x to 1.4x). No greeting or sign-off that
  was not already there. No em dashes.
- Output format: one block per input reply, in order, each opened by a line
  reading "### REPLY k" (k counting from 1) and followed by the rewritten text.
  Nothing before the first marker, nothing after the last block, no commentary,
  no JSON, no code fences.`;

const message = (count: number): string =>
  `Rewrite each of the ${count} entries of \`replies\` in the style described. ` +
  `\`conversation\` is context only, do not rewrite it. Emit exactly ${count} ` +
  `blocks, "### REPLY 1" through "### REPLY ${count}". Input follows.`;

function die(msg: string): never {
  throw new UsageError(msg);
}

export interface SeedRecord {
  id: string;
  config: string;
  messages: ChatMessage[];
}

/** The JSON payload handed to the rewriter: context plus the turns to rewrite. */
export function buildPayload(rec: SeedRecord): string {
  return JSON.stringify({
    conversation: rec.messages,
    replies: rec.messages.filter((m) => m.role === "assistant")
      .map((m, i) => ({ n: i + 1, text: m.content })),
  });
}

/**
 * Split a model reply into the rewritten turns. The "### REPLY k" markers exist
 * because a JSON array does not survive contact with a local model: replies
 * contain paragraph breaks, which come back as raw newlines inside the strings
 * and make the whole array unparseable. A JSON array is still accepted as a
 * fallback, for the calls where the model answers in it anyway.
 */
export function parseReplies(text: string): string[] | null {
  const marks = [...text.matchAll(/^[ \t]*#{1,6}[ \t]*REPLY[ \t]*(\d+)[ \t]*:?[ \t]*$/gim)];
  if (marks.length) {
    const out: string[] = [];
    for (let i = 0; i < marks.length; i++) {
      const from = marks[i].index! + marks[i][0].length;
      const to = i + 1 < marks.length ? marks[i + 1].index! : text.length;
      out.push(text.slice(from, to).replace(/^```[a-z]*\n?|```\s*$/gim, "").trim());
    }
    return out;
  }
  const start = text.indexOf("[");
  const end = text.lastIndexOf("]");
  if (start < 0 || end <= start) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(text.slice(start, end + 1));
  } catch {
    return null;
  }
  if (!Array.isArray(parsed)) return null;
  if (!parsed.every((v) => typeof v === "string")) return null;
  return parsed as string[];
}

/**
 * Em dashes are the one style rule the rewriter keeps breaking (they are all
 * over the source data), and they are mechanical to fix, so fix them instead of
 * throwing the sample away: "a, b" reads as "a, b", a numeric range as a hyphen.
 */
export function stripDashes(s: string): string {
  return s
    .replace(/(\d)\s*[—–]\s*([$€£]?\d)/g, "$1-$2")
    .replace(/\s*[—–]\s*/g, ", ")
    .replace(/,\s*,/g, ",");
}

/** Why a rewrite is unusable, or "" when it is fine. Length drift is the tell. */
export function rejectReason(
  originals: string[],
  rewritten: string[],
  authorNames: string[] = [],
): string {
  if (rewritten.length !== originals.length) {
    return `got ${rewritten.length} replies, wanted ${originals.length}`;
  }
  for (let i = 0; i < originals.length; i++) {
    const r = rewritten[i].trim();
    if (r.length < 15) return `reply ${i} too short (${r.length} chars)`;
    const ratio = r.length / originals[i].length;
    if (ratio < 0.3 || ratio > 2.5) return `reply ${i} length ratio ${ratio.toFixed(2)}`;
    // A rewrite that addresses or names the author has slipped out of character:
    // it is writing ABOUT the voice instead of IN it.
    if (authorNames.some((n) => new RegExp(`\\b${n}\\b`, "i").test(r))) {
      return `reply ${i} names the author`;
    }
  }
  return "";
}

/** Assistant turns replaced in order; every other turn untouched. */
export function applyReplies(rec: SeedRecord, rewritten: string[]): SeedRecord {
  let k = 0;
  return {
    ...rec,
    messages: rec.messages.map((m) =>
      m.role === "assistant" ? { role: m.role, content: stripDashes(rewritten[k++].trim()) } : m
    ),
  };
}

async function runPi(
  systemPrompt: string,
  payload: string,
  count: number,
  timeoutMs: number,
  provider: string,
  model: string,
): Promise<string> {
  // deno-lint-ignore no-explicit-any
  const D = (globalThis as any).Deno;
  if (!D?.Command) die("this script needs Deno (it spawns the pi CLI)");
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const child = new D.Command("pi", {
      args: [
        "-p",
        ...(provider ? ["--provider", provider] : []),
        ...(model ? ["--model", model] : []),
        "--thinking",
        "off",
        "--no-session",
        "-nt",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "-nc",
        "--system-prompt",
        systemPrompt,
        message(count),
      ],
      stdin: "piped",
      stdout: "piped",
      stderr: "piped",
      signal: ctl.signal,
    }).spawn();
    const w = child.stdin.getWriter();
    await w.write(new TextEncoder().encode(payload));
    await w.close();
    const { code, stdout, stderr } = await child.output();
    const out = new TextDecoder().decode(stdout);
    if (code !== 0) throw new Error(`pi exited ${code}: ${new TextDecoder().decode(stderr)}`);
    return out;
  } finally {
    clearTimeout(timer);
  }
}

async function run(v: Values) {
  const inPath = v.str("in");
  const outPath = v.str("out");
  const failPath = `${outPath}.failed.jsonl`;
  const limit = v.num("limit");
  const jobs = Math.max(1, v.num("jobs"));
  const timeoutMs = v.num("timeout") * 1000;
  const skillPath = v.str("style-guide");
  // Empty means "whatever pi is configured with", which is the portable default.
  const authorNames = (v.opt("author-name") ?? "").split(",").map((x) => x.trim()).filter(Boolean);
  const provider = v.opt("provider") ?? "";
  const model = v.opt("model") ?? "";

  const skill = await readFileText(skillPath).catch(() =>
    die(`cannot read style guide ${skillPath}`)
  );
  const systemPrompt = skill + RULES;

  const seedText = await readFileText(inPath).catch(() => die(`cannot read ${inPath}`));
  const seed: SeedRecord[] = seedText.trim().split("\n").filter(Boolean).map((l) => JSON.parse(l));

  const fs = await import("node:fs");
  const dir = outPath.slice(0, outPath.lastIndexOf("/"));
  if (dir) fs.mkdirSync(dir, { recursive: true });
  const done = new Set<string>();
  if (fs.existsSync(outPath)) {
    for (const line of fs.readFileSync(outPath, "utf8").split("\n")) {
      if (line.trim()) done.add(JSON.parse(line).id);
    }
  }
  let todo = seed.filter((r) => !done.has(r.id));
  if (limit > 0) todo = todo.slice(0, limit);

  console.log(
    `=== restyle: ${todo.length} conversations (${done.size} already done) ` +
      `via ${model || "pi's default model"}, ${jobs} job(s) ===`,
  );
  console.log(`style guide: ${skillPath} (${(skill.length / 1024).toFixed(0)} KB)\n`);

  let ok = 0, failed = 0, index = 0;
  const t0 = Date.now();
  const worker = async () => {
    while (index < todo.length) {
      const rec = todo[index++];
      const originals = rec.messages.filter((m) => m.role === "assistant").map((m) => m.content);
      let reason = "";
      for (let attempt = 1; attempt <= 2 && reason !== "ok"; attempt++) {
        try {
          const text = await runPi(
            systemPrompt,
            buildPayload(rec),
            originals.length,
            timeoutMs,
            provider,
            model,
          );
          const replies = parseReplies(text);
          if (!replies) {
            reason = `unparseable reply: ${text.slice(0, 160).replace(/\n/g, " ")}`;
            continue;
          }
          reason = rejectReason(originals, replies, authorNames);
          if (reason) continue;
          fs.appendFileSync(outPath, JSON.stringify(applyReplies(rec, replies)) + "\n");
          reason = "ok";
          ok++;
        } catch (e) {
          reason = `pi failed: ${(e as Error).message.slice(0, 160)}`;
        }
      }
      if (reason !== "ok") {
        failed++;
        fs.appendFileSync(failPath, JSON.stringify({ id: rec.id, reason }) + "\n");
      }
      const n = ok + failed;
      if (n % 10 === 0 || n === todo.length) {
        const rate = n / ((Date.now() - t0) / 1000);
        console.log(
          `  ${n}/${todo.length} (${ok} ok, ${failed} rejected, ${rate.toFixed(2)}/s, ` +
            `eta ${Math.round((todo.length - n) / Math.max(rate, 1e-6) / 60)}m)`,
        );
      }
    }
  };
  await Promise.all(Array.from({ length: jobs }, worker));

  console.log(`\nwrote ${outPath}: ${ok} restyled conversations`);
  if (failed) console.log(`${failed} rejected; reasons in ${failPath}`);
  console.log(
    `next: chat-corpus --data ${outPath} --tokenizer <base>.tokenizer.json --out data/style`,
  );
}

export const styleRestyleCommand: Command = {
  name: "style-restyle",
  summary: "Rewrite a chat corpus's assistant turns in one author's voice, via the pi CLI.",
  details: `Optional workflow, and the only command here that shells out to something else: it
drives the \`pi\` CLI against whatever model pi is configured with, using --style-guide as the
system prompt.

Only assistant turns are rewritten. User and system turns pass through untouched, because
the model should learn to ANSWER in that voice, not to be asked in it.

Every rewrite is validated (reply count, per-reply length ratio 0.3 to 2.5x, no leaked
author name) and retried once; whatever still fails is written to <out>.failed.jsonl with a
reason and never reaches training. The output is appended per conversation, so a re-run
skips what is already done.

Keep --jobs at 1 unless you have measured otherwise: the style guide is a large, constant
system prompt, and one sequential stream keeps hitting the server's prompt cache while
parallel calls evict each other.`,
  examples: [
    "style-restyle --in corpus/style/seed.jsonl --out corpus/style/restyled.jsonl --style-guide ~/my-voice.md",
    "style-restyle --in seed.jsonl --out restyled.jsonl --style-guide voice.md --limit 10",
  ],
  flags: [
    {
      name: "in",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "seed .jsonl from `style-seed`",
    },
    {
      name: "out",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "output .jsonl; rejects go to <out>.failed.jsonl",
    },
    {
      name: "style-guide",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "markdown file describing the target voice, used as the system prompt",
    },
    {
      name: "author-name",
      type: "string",
      placeholder: "NAME[,NAME]",
      describe:
        "reject any rewrite that mentions these names: the model has stepped out of character",
    },
    {
      name: "model",
      type: "string",
      placeholder: "ID",
      describe: "model id passed to pi (default: whatever pi is configured with)",
    },
    {
      name: "provider",
      type: "string",
      placeholder: "NAME",
      describe: "provider passed to pi (default: pi's own default)",
    },
    { name: "limit", type: "number", default: 0, describe: "stop after N conversations (0 = all)" },
    { name: "jobs", type: "number", default: 1, describe: "parallel pi calls" },
    {
      name: "timeout",
      type: "number",
      default: 300,
      describe: "seconds before a single pi call is killed",
    },
  ],
  run: run,
};
