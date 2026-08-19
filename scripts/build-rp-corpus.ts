// Render PIPPA's roleplay logs into the plain-text transcript format that AI
// Horde scribes actually receive, so a base model can be continued-pretrained on
// its deployment distribution rather than on chat-formatted data.
//
//   deno run -A scripts/build-rp-corpus.ts --out corpus/rp-pippa.txt
//
// Horde text workers serve raw prompt continuation (KoboldCpp's /api/v1/generate,
// llama.cpp's /completions), never a chat template. The prompt a SillyTavern or
// Kobold Lite user sends is a persona block followed by "Name:" / "You:" turns,
// which is what this writes. Documents are separated by <|endoftext|>, so
// `tokenize` reads the output unchanged.
//
// PIPPA's `bot_definitions` carries Character.AI's own formatting (the {{char}}
// and {{user}} placeholders and their many hand-typed variants, END_OF_DIALOG
// between example chats); the paper's Appendix A says to preprocess it, and
// substitute/normalizeDefinitions below are that step.
//
//   deno run tests/rp-corpus.ts     # the pure rendering logic, no download

const DOC_SEP = "<|endoftext|>";
const USER_NAME = "You"; // the horde/SillyTavern default for the human side
const PIPPA_URL =
  "https://huggingface.co/datasets/PygmalionAI/PIPPA/resolve/main/pippa_deduped.jsonl";

/** One PIPPA row; only the fields the transcript needs. */
export interface PippaRow {
  bot_name?: string;
  bot_greeting?: string;
  bot_definitions?: string;
  bot_description?: string;
  conversation?: { message?: string; is_human?: boolean }[];
}

export interface RenderOpts {
  /** Truncate at the last turn boundary under this many characters. 40k is ~9.5k
   * tokens, just over the model's declared 8192 context, so every kept document
   * can still fill a full-context window; the cap only exists to stop a handful
   * of outliers (the longest log in PIPPA is 2.65M characters, p99 is 140k) from
   * dominating the sampler. */
  maxChars: number;
  /** Reject a transcript below this share of ASCII: at 94.7M parameters, tokens
   * spent on CJK and emoji are tokens not spent on the target register. */
  minAsciiRatio: number;
  /** Minimum non-empty utterances. 3 matches pippa_deduped's own floor and keeps
   * the "persona block, greeting, first exchange" shape, which is exactly what a
   * fresh horde chat looks like. */
  minTurns: number;
}

export const DEFAULT_RENDER: RenderOpts = { maxChars: 40000, minAsciiRatio: 0.9, minTurns: 3 };

export type Rendered = { text: string } | { reject: string };

/** Macro bodies that name the character. PIPPA has {{c}} and {{char_1}} beside
 * the documented {{char}}. */
const BOT_ALIAS = /^(?:char[_ ]?\d*|c|bot|character)$/i;
/** Macro bodies that name the human: {{u}}, {{u01}}, {{user_3}}, and four
 * misspellings of {{random_user_N}}. */
const USER_ALIAS = /^(?:you|u\d*|users?|usser|(?:random[_ ]?)?user[_ ]?\d*)$/i;
/** A double-delimiter macro, tolerating the mixed pairs people typed by accident
 * ({{char]}, {[char}}, {{char]]). The opening brace is required, so the [[...]]
 * some cards use for something else is left alone. */
const MACRO = /\{[{[]([^{}[\]\n]{1,60})[}\]][}\]]/g;
/** A single-brace macro. Only substituted when it names someone, because braces
 * in this corpus also carry prose ({smiles}) and the W++ persona notation. */
const SINGLE_MACRO = /\{([^{}[\]\n]{1,40})\}/g;
/** A pasted reference link (wikis, YouTube, Reddit), with its wrapping bracket
 * when it has one. */
const URL_REF = /\s*[([]?\s*https?:\/\/[^\s)\]]+[)\]]?/gi;

/** Who a macro names, or null when it names no one. A trailing colon belongs to
 * the speaker label, not to the name, so it survives the lookup. */
function macroName(inner: string, botName: string): string | null {
  const trimmed = inner.trim();
  const colon = trimmed.endsWith(":") ? ":" : "";
  const key = (colon ? trimmed.slice(0, -1) : trimmed).trim();
  if (BOT_ALIAS.test(key) || key.toLowerCase() === botName.toLowerCase()) return botName + colon;
  if (USER_ALIAS.test(key)) return USER_NAME + colon;
  return null;
}

/**
 * Character.AI's placeholders, pasted reference links, and the doc marker (an
 * embedded one would forge a false document boundary downstream).
 *
 * Resolving only the documented {{char}} and {{user}} is not enough. PIPPA's
 * cards carry 60 other spellings, and the render left 4,719 macros standing
 * across 5.6% of documents; a 94.7M model trained on that emitted "{{u01}}:" as
 * a speaker name, which breaks the "Name:" turn structure a horde client parses.
 * So the rule for a double-delimiter macro is: resolve the aliases we can name,
 * unwrap everything else (another character's name, or a stage direction someone
 * put in braces), and let no brace through. Single braces get the narrower
 * treatment, since in this corpus they are usually not macros at all.
 */
export function substitute(text: string, botName: string): string {
  return text
    .replaceAll(MACRO, (_, inner: string) => macroName(inner, botName) ?? inner.trim())
    .replaceAll(SINGLE_MACRO, (whole, inner: string) => macroName(inner, botName) ?? whole)
    .replaceAll(/<BOT>/g, botName)
    .replaceAll(/<USER>/g, USER_NAME)
    .replaceAll(URL_REF, "")
    // Only the renderer gets to emit <START> and the document separator, so a
    // log that happens to contain either cannot move a boundary.
    .replaceAll(/<START>/gi, " ")
    .replaceAll(DOC_SEP, " ");
}

/**
 * Turn a `bot_definitions` blob into example dialogue. Character.AI separates
 * example chats with END_OF_DIALOG; they become one example block with a blank
 * line between chats.
 *
 * They deliberately do NOT each become a <START>. Doing that gave a mean of 4.96
 * markers per document (median 4, max 30, 70.6% above two), and a probe trained
 * on it emitted <START> mid-conversation, having learned the marker as something
 * that recurs every few turns. A prompt has one, or two with examples.
 */
export function normalizeDefinitions(defs: string, botName: string): string {
  return substitute(defs, botName)
    .replaceAll(/[ \t]*\n?\s*END_OF_DIALOG\s*\n?[ \t]*/g, "\n\n")
    .split("\n")
    .map((l) => l.trimEnd())
    .filter((l, i, all) => l.length > 0 || (i > 0 && all[i - 1].length > 0)) // collapse blank runs
    .join("\n")
    .trim();
}

function asciiRatio(s: string): number {
  let ascii = 0;
  for (let i = 0; i < s.length; i++) if (s.charCodeAt(i) < 128) ascii++;
  return s.length === 0 ? 0 : ascii / s.length;
}

/** Cut at the last line boundary at or under `limit`, so a document never ends
 * mid-utterance. */
function truncateAtTurn(text: string, limit: number): string {
  if (text.length <= limit) return text;
  const cut = text.lastIndexOf("\n", limit);
  return (cut > 0 ? text.slice(0, cut) : text.slice(0, limit)).trimEnd();
}

/** One conversation as a horde-shaped transcript, or why it was dropped. */
export function renderRow(row: PippaRow, opts: RenderOpts = DEFAULT_RENDER): Rendered {
  const botName = (row.bot_name ?? "").trim();
  if (!botName || botName.length > 40) return { reject: "no-name" };

  const turns = (row.conversation ?? [])
    .map((t) => ({
      human: t.is_human === true,
      text: substitute((t.message ?? "").trim(), botName),
    }))
    .filter((t) => t.text.length > 0);
  if (turns.length < opts.minTurns) return { reject: "short" };

  const lines: string[] = [`[Character: ${botName}]`];

  const description = substitute((row.bot_description ?? "").trim(), botName);
  if (description) lines.push(`${botName}'s Persona: ${description}`);

  const definitions = normalizeDefinitions(row.bot_definitions ?? "", botName);
  if (definitions) lines.push("<START>", definitions);

  lines.push("<START>");

  // The greeting is the character's first utterance, and PIPPA already repeats it
  // as conversation[0]; emit it only when the log does not.
  const greeting = substitute((row.bot_greeting ?? "").trim(), botName);
  if (greeting && turns[0].text !== greeting) lines.push(`${botName}: ${greeting}`);

  for (const t of turns) lines.push(`${t.human ? USER_NAME : botName}: ${t.text}`);

  const text = truncateAtTurn(lines.join("\n").trim(), opts.maxChars);
  if (text.length < 400) return { reject: "tiny" };
  if (asciiRatio(text) < opts.minAsciiRatio) return { reject: "non-ascii" };
  return { text };
}

// ---------------------------------------------------------------------------
// Driver: cache the source file, stream it, write the corpus.
// ---------------------------------------------------------------------------

async function ensureCached(path: string): Promise<void> {
  const fs = await import("node:fs");
  if (fs.existsSync(path) && fs.statSync(path).size > 0) {
    console.log(`Source: ${path} (cached, ${(fs.statSync(path).size / 1e6).toFixed(1)} MB)`);
    return;
  }
  console.log(`Source: downloading ${PIPPA_URL}`);
  const resp = await fetch(PIPPA_URL);
  if (!resp.ok || !resp.body) throw new Error(`fetch PIPPA -> ${resp.status} ${resp.statusText}`);
  fs.mkdirSync(path.slice(0, path.lastIndexOf("/")) || ".", { recursive: true });
  const tmp = `${path}.part`;
  const fd = fs.openSync(tmp, "w");
  let bytes = 0;
  try {
    for await (const chunk of resp.body) {
      fs.writeSync(fd, chunk);
      bytes += chunk.length;
    }
  } finally {
    fs.closeSync(fd);
  }
  fs.renameSync(tmp, path); // rename last, so a killed download never looks cached
  console.log(`  ${(bytes / 1e6).toFixed(1)} MB -> ${path}`);
}

/** Yield whole lines from a large file without holding it in memory (the source
 * is ~257 MB, half of V8's single-string ceiling). */
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
  const out = arg("out", "corpus/rp-pippa.txt");
  const cache = arg("cache", "corpus/cache/pippa_deduped.jsonl");
  const maxChars = Number(arg("max-chars", String(DEFAULT_RENDER.maxChars)));
  const opts: RenderOpts = { ...DEFAULT_RENDER, maxChars };

  console.log("=== Felladrin's GGUF Trainer +∞ :: roleplay transcript corpus ===\n");
  await ensureCached(cache);

  const fs = await import("node:fs");
  fs.mkdirSync(out.slice(0, out.lastIndexOf("/")) || ".", { recursive: true });
  const enc = new TextEncoder();
  const sep = enc.encode(`\n${DOC_SEP}\n`);
  const fd = fs.openSync(out, "w");

  const rejects = new Map<string, number>();
  let rows = 0, kept = 0, chars = 0;
  try {
    for await (const line of lines(cache)) {
      rows++;
      let row: PippaRow;
      try {
        row = JSON.parse(line) as PippaRow;
      } catch {
        rejects.set("unparseable", (rejects.get("unparseable") ?? 0) + 1);
        continue;
      }
      const r = renderRow(row, opts);
      if ("reject" in r) {
        rejects.set(r.reject, (rejects.get(r.reject) ?? 0) + 1);
        continue;
      }
      fs.writeSync(fd, enc.encode(r.text));
      fs.writeSync(fd, sep);
      kept++;
      chars += r.text.length;
    }
  } finally {
    fs.closeSync(fd);
  }

  if (kept === 0) throw new Error(`no transcripts survived the filters (${rows} rows read)`);
  const mb = fs.statSync(out).size / 1e6;
  console.log(
    `\nKept ${kept}/${rows} transcripts, ${mb.toFixed(1)} MB, ` +
      `~${(chars / 4.2 / 1e6).toFixed(0)}M tokens est. -> ${out}`,
  );
  for (const [reason, n] of [...rejects].sort((a, b) => b[1] - a[1])) {
    console.log(`  dropped ${String(n).padStart(6)}  ${reason}`);
  }
  console.log("\n=== roleplay corpus OK ===");
}

if (import.meta.main) await main();
