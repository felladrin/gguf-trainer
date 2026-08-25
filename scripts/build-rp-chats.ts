// Build the dual-format roleplay SFT corpus: one .jsonl that `chat-corpus` turns
// into tokens plus a reply-only mask.
//
//   deno run -A scripts/build-rp-chats.ts --out data/rpx
//
// Writes <out>.jsonl and <out>-hold.jsonl. The holdout is split HERE, before
// tokenizing, because a slice carved afterwards is corpus the run has already
// seen and cannot rank two checkpoints (agents.md, "Evaluate").
//
// Two formats, because llama.cpp has two endpoints and a roleplay model is used
// through both. `/v1/chat/completions` sends ChatML; `/completions` sends a
// persona block and "Name:" turns. Rows tagged `format: "transcript"` train the
// second, with the human's turns masked out (src/data/transcript.ts). A PIPPA
// conversation goes into ONE arm or the other, never both, so neither arm is
// scored on text the model saw in the other.
//
// Sources, and what each is here for:
//
//   PygmalionAI/PIPPA            the roleplay register itself; human-written logs
//   Gryphe/Opus-WritingPrompts   long-form prose, so replies are not all dialogue
//   kalomaze/Opus_Instruct_3k    multi-turn instruction following at quality
//   jondurbin/gutenberg-dpo-v0.1 public-domain novel prose (the `chosen` side only)
//   HuggingFaceTB/smol-smoltalk  what the base was instruction-tuned on: this is
//                                what keeps the chat endpoint from collapsing into
//                                roleplay-only after the fine-tune
//
//   deno run tests/rp-chats.ts   the pure row-building logic, nothing downloaded

import { fetchParquetUrls } from "../src/data/hf.ts";
import { parseDataFile } from "../src/data/parse.ts";
import { normalizeRole } from "../src/data/chat.ts";
import { normalizeDefinitions, type PippaRow, substitute } from "./build-rp-corpus.ts";

const PIPPA_CACHE = "corpus/cache/pippa_deduped.jsonl";

export interface ChatRow {
  messages: { role: string; content: string }[];
  format?: "transcript";
  character?: string;
  user_label?: string;
  persona?: string;
  examples?: string;
}

// ---------------------------------------------------------------------------
// Deterministic randomness: the corpus must rebuild byte-identically.
// ---------------------------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Labels for the human side of a transcript. "You" is the SillyTavern and Kobold
 * default and stays the majority, but it is not the only one: a model trained on
 * a single literal label learns that string rather than the shape "a name, then a
 * colon", and then emits "You:" in contexts where no such label exists.
 */
export const USER_LABELS = [
  "You",
  "You",
  "You",
  "You",
  "You",
  "You",
  "You",
  "User",
  "Anon",
  "Alex",
  "Sam",
  "Morgan",
];

// ---------------------------------------------------------------------------
// Identity scrubbing
// ---------------------------------------------------------------------------

/** System prompts that assert someone else's product identity. Left in, a 135M
 * model happily tells users it is Claude, made by Anthropic. */
const FOREIGN_IDENTITY = /\b(claude|anthropic|smollm|openai|chatgpt|gpt-4|hugging\s*face)\b/i;

export function scrubSystem(content: string): string | null {
  if (FOREIGN_IDENTITY.test(content)) return null;
  return content.trim() || null;
}

// ---------------------------------------------------------------------------
// PIPPA -> both arms
// ---------------------------------------------------------------------------

export interface PippaOpts {
  /** Cap on a document's characters. 6000 is ~1500 tokens, comfortably inside a
   * 2048-token training window with the persona block. */
  maxChars: number;
  minAsciiRatio: number;
  minTurns: number;
}

export const DEFAULT_PIPPA: PippaOpts = { maxChars: 6000, minAsciiRatio: 0.9, minTurns: 3 };

function asciiRatio(s: string): number {
  let ascii = 0;
  for (let i = 0; i < s.length; i++) if (s.charCodeAt(i) < 128) ascii++;
  return s.length === 0 ? 0 : ascii / s.length;
}

interface Parsed {
  character: string;
  persona: string;
  examples: string;
  turns: { human: boolean; text: string }[];
}

/** PIPPA's card fields and log, with Character.AI's placeholder syntax resolved.
 * Returns null when the row cannot make a usable conversation. */
export function parsePippa(row: PippaRow, opts: PippaOpts = DEFAULT_PIPPA): Parsed | null {
  const character = (row.bot_name ?? "").trim();
  if (!character || character.length > 40) return null;

  const turns = (row.conversation ?? [])
    .map((t) => ({
      human: t.is_human === true,
      text: substitute((t.message ?? "").trim(), character),
    }))
    .filter((t) => t.text.length > 0);
  if (turns.length < opts.minTurns) return null;

  const joined = turns.map((t) => t.text).join(" ");
  if (joined.length < 200 || asciiRatio(joined) < opts.minAsciiRatio) return null;

  return {
    character,
    persona: substitute((row.bot_description ?? "").trim(), character),
    examples: usefulExamples(normalizeDefinitions(row.bot_definitions ?? "", character), character),
    turns,
  };
}

/**
 * Example dialogue, or "" when the card's definitions are not dialogue. Many
 * PIPPA cards leave `bot_definitions` as a bare colon, a URL, or a W++ attribute
 * dump; rendered under an "Example dialogue:" heading those teach the model that
 * the block is filler. Require at least one line that actually looks like the
 * character speaking.
 */
export function usefulExamples(defs: string, character: string): string {
  const speaks = new RegExp(`^\\s*${character.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\s*:\\s*\\S`);
  const lines = defs.split("\n");
  return lines.some((l) => speaks.test(l) && l.trim().length > character.length + 12) ? defs : "";
}

/** Keep the newest turns that fit, so the target reply always survives and the
 * context that reaches it is the most recent. */
function tailWithin(
  turns: { human: boolean; text: string }[],
  budget: number,
): { human: boolean; text: string }[] {
  const kept: { human: boolean; text: string }[] = [];
  let used = 0;
  for (let i = turns.length - 1; i >= 0; i--) {
    used += turns[i].text.length + 12; // label, colon, newline
    if (used > budget && kept.length) break;
    kept.unshift(turns[i]);
  }
  return kept;
}

/** The ChatML arm: the whole conversation, every character turn supervised. The
 * system prompt is written the way a person writes one, because that is what the
 * chat endpoint will actually receive. */
export function pippaToChatML(p: Parsed, opts: PippaOpts = DEFAULT_PIPPA): ChatRow | null {
  const header = [`You are ${p.character}. Stay in character and reply only as ${p.character}.`];
  if (p.persona) header.push(`\n${p.character}'s Persona: ${p.persona}`);
  if (p.examples) header.push(`\nExample dialogue:\n${p.examples}`);
  const system = header.join("\n").slice(0, opts.maxChars);

  // A ChatML turn list must alternate: PIPPA logs sometimes repeat a side, and
  // the template renders those as two consecutive user blocks, which trains the
  // model to expect its own turn to be someone else's.
  const merged: { human: boolean; text: string }[] = [];
  for (const t of p.turns) {
    const last = merged[merged.length - 1];
    if (last && last.human === t.human) last.text += "\n\n" + t.text;
    else merged.push({ ...t });
  }
  // The first turn must be the human's for a chat exchange; PIPPA opens with the
  // character's greeting, which belongs in the system block, not as a reply to
  // nothing.
  if (merged.length && !merged[0].human) merged.shift();
  const turns = tailWithin(merged, opts.maxChars - system.length);
  if (turns.length < 2 || !turns[0].human || turns[turns.length - 1].human) return null;

  return {
    messages: [
      { role: "system", content: system },
      ...turns.map((t) => ({ role: t.human ? "user" : "assistant", content: t.text })),
    ],
  };
}

/** The transcript arm: context up to a chosen character turn, that turn as the
 * single target. `cut` picks which character turn (0..1 across the eligible ones). */
export function pippaToTranscript(
  p: Parsed,
  cut: number,
  userLabel: string,
  opts: PippaOpts = DEFAULT_PIPPA,
): ChatRow | null {
  const targets: number[] = [];
  for (let i = 1; i < p.turns.length; i++) if (!p.turns[i].human) targets.push(i);
  if (!targets.length) return null;
  const end = targets[Math.min(targets.length - 1, Math.floor(cut * targets.length))];

  const head = `[Character: ${p.character}]`.length +
    (p.persona ? p.character.length + 12 + p.persona.length : 0) +
    (p.examples ? p.examples.length + 16 : 0);
  const turns = tailWithin(p.turns.slice(0, end + 1), Math.max(600, opts.maxChars - head));
  if (turns.length < 2 || turns[turns.length - 1].human) return null;

  return {
    format: "transcript",
    character: p.character,
    user_label: userLabel,
    persona: p.persona.slice(0, 2000),
    examples: p.examples.slice(0, 2000),
    messages: turns.map((t) => ({ role: t.human ? "user" : "assistant", content: t.text })),
  };
}

// ---------------------------------------------------------------------------
// Hugging Face sources -> ChatML rows
// ---------------------------------------------------------------------------

/** Normalize any of the shapes these five datasets use into a ChatRow, dropping
 * rows that do not end on an assistant turn (nothing to supervise). */
export function toChatRow(row: Record<string, unknown>): ChatRow | null {
  const list = row.conversations ?? row.messages;
  const messages: { role: string; content: string }[] = [];

  if (Array.isArray(list)) {
    for (const item of list) {
      if (typeof item !== "object" || item === null) continue;
      const r = item as Record<string, unknown>;
      const role = normalizeRole(String(r.from ?? r.role ?? ""));
      const content = String(r.value ?? r.content ?? "").trim();
      if (!content) continue;
      if (role === "system") {
        const kept = scrubSystem(content);
        if (kept) messages.push({ role, content: kept });
        continue;
      }
      if (role !== "user" && role !== "assistant") continue;
      messages.push({ role, content });
    }
  } else if (typeof row.prompt === "string" && typeof row.chosen === "string") {
    messages.push({ role: "user", content: row.prompt.trim() });
    messages.push({ role: "assistant", content: row.chosen.trim() });
  }

  if (messages.length < 2) return null;
  if (messages[messages.length - 1].role !== "assistant") return null;
  const firstNonSystem = messages.find((m) => m.role !== "system");
  if (!firstNonSystem || firstNonSystem.role !== "user") return null;
  return { messages };
}

async function fromHub(id: string, limit: number): Promise<ChatRow[]> {
  const out: ChatRow[] = [];
  const urls = await fetchParquetUrls(id, "default", "train");
  for (const url of urls) {
    if (out.length >= limit) break;
    console.log(`  ${id}: ${url.split("/").pop()}`);
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    for (const row of await parseDataFile(url, bytes)) {
      if (out.length >= limit) break;
      const r = toChatRow(row as Record<string, unknown>);
      if (r) out.push(r);
    }
  }
  console.log(`  ${id}: ${out.length} rows`);
  return out;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

async function* lines(path: string): AsyncGenerator<string> {
  const fs = await import("node:fs");
  const decoder = new TextDecoder();
  let carry = "";
  for await (const chunk of fs.createReadStream(path, { highWaterMark: 1 << 22 })) {
    carry += decoder.decode(chunk as Uint8Array, { stream: true });
    const parts = carry.split("\n");
    carry = parts.pop() ?? "";
    for (const p of parts) if (p.trim()) yield p;
  }
  if (carry.trim()) yield carry;
}

function arg(name: string, fallback: string): string {
  const hit = Deno.args.find((a) => a.startsWith(`--${name}=`));
  if (hit) return hit.slice(name.length + 3);
  const i = Deno.args.indexOf(`--${name}`);
  return i >= 0 && Deno.args[i + 1] ? Deno.args[i + 1] : fallback;
}

async function main(): Promise<void> {
  const out = arg("out", "data/rpx");
  const holdout = Number(arg("holdout", "400"));
  const pippaChat = Number(arg("pippa-chat", "8000"));
  const pippaTx = Number(arg("pippa-transcript", "12000"));
  const rng = mulberry32(Number(arg("seed", "11")));

  console.log("=== Felladrin's GGUF Trainer +∞ :: dual-format roleplay corpus ===\n");

  const fs = await import("node:fs");
  if (!fs.existsSync(PIPPA_CACHE)) {
    throw new Error(`${PIPPA_CACHE} missing: run scripts/build-rp-corpus.ts once to cache it`);
  }

  const chatArm: ChatRow[] = [];
  const txArm: ChatRow[] = [];
  let read = 0, unusable = 0;
  for await (const line of lines(PIPPA_CACHE)) {
    read++;
    if (chatArm.length >= pippaChat && txArm.length >= pippaTx) break;
    let parsed: Parsed | null = null;
    try {
      parsed = parsePippa(JSON.parse(line) as PippaRow);
    } catch {
      parsed = null;
    }
    if (!parsed) {
      unusable++;
      continue;
    }
    // One arm or the other, never both.
    const wantsTx = txArm.length < pippaTx &&
      (chatArm.length >= pippaChat || rng() < pippaTx / (pippaChat + pippaTx));
    const row = wantsTx
      ? pippaToTranscript(parsed, rng(), USER_LABELS[Math.floor(rng() * USER_LABELS.length)])
      : (chatArm.length < pippaChat ? pippaToChatML(parsed) : null);
    if (!row) {
      unusable++;
      continue;
    }
    (row.format === "transcript" ? txArm : chatArm).push(row);
  }
  console.log(
    `PIPPA: ${read} rows read -> ${chatArm.length} ChatML + ${txArm.length} transcripts ` +
      `(${unusable} unusable)`,
  );

  const extras: ChatRow[] = [];
  for (
    const [id, limit] of [
      ["Gryphe/Opus-WritingPrompts", 2000],
      ["kalomaze/Opus_Instruct_3k", 3000],
      ["jondurbin/gutenberg-dpo-v0.1", 1000],
      ["HuggingFaceTB/smol-smoltalk", 6000],
    ] as [string, number][]
  ) {
    try {
      extras.push(...await fromHub(id, limit));
    } catch (e) {
      console.log(`  ${id}: SKIPPED (${e})`); // a source being down must not sink the run
    }
  }

  const all = [...chatArm, ...txArm, ...extras];
  for (let i = all.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [all[i], all[j]] = [all[j], all[i]];
  }

  const enc = new TextEncoder();
  const write = (path: string, rows: ChatRow[]) => {
    const fd = fs.openSync(path, "w");
    try {
      for (const r of rows) fs.writeSync(fd, enc.encode(JSON.stringify(r) + "\n"));
    } finally {
      fs.closeSync(fd);
    }
    console.log(`  ${rows.length} rows -> ${path}`);
  };
  write(`${out}-hold.jsonl`, all.slice(0, holdout));
  write(`${out}.jsonl`, all.slice(holdout));

  const tx = all.filter((r) => r.format === "transcript").length;
  console.log(
    `\nTotal ${all.length} rows: ${tx} transcripts, ${all.length - tx} ChatML ` +
      `(${(100 * tx / all.length).toFixed(0)}% raw-completion format)`,
  );
  console.log("\n=== dual-format roleplay corpus OK ===");
}

if (import.meta.main) await main();
