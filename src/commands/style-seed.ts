// Curriculum stage 2, part 1: the SEED corpus for the style SFT.
//
// Samples a filtered mix of HuggingFaceTB/smoltalk configs into one JSONL of
// short English conversations. This is the raw material: `style-restyle`
// rewrites the assistant turns in a target voice, and `chat-corpus` then
// tokenizes the result against the base model's frozen vocab.
//
//   style-seed --total 200 --out corpus/style/seed.jsonl
//
// The mix is weighted, so --total is the only knob between a prototype and the
// full run. Configs were chosen for what a ~95M model can actually learn:
// turn-taking, system-prompt following, and text transformation (summarize /
// rewrite), where the answer is derivable from the prompt. The math, code and
// long-context configs (numina-cot, metamathqa, self-oss-instruct, apigen,
// longalign) are deliberately absent; they teach confident wrong answers at
// this scale.
//
// Filters (all of them exist to keep the pi rewrite pass cheap and the SFT
// sequences short): 2-12 messages, assistant turns 20-900 chars, user turns
// <=700, whole conversation <=2500, near-ASCII, no code fences or LaTeX.

import { fetchParquetUrls } from "../data/hf.ts";
import { parseDataFile } from "../data/parse.ts";
import { detectMapping, rowToMessages } from "../data/chat.ts";
import type { ChatMessage } from "../data/chat.ts";
import { mulberry32 } from "../model/autograd.ts";
import type { Command, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";

function die(msg: string): never {
  throw new UsageError(msg);
}

const DATASET = "HuggingFaceTB/smoltalk";
const DEFAULT_MIX = "everyday-conversations:30,smol-magpie-ultra:30,systemchats-30k:15," +
  "smol-summarize:12,smol-rewrite:8,smol-constraints:5";

/** Every reason a row is unusable, so the skip counts say something. */
export type Reject =
  | "ok"
  | "turns"
  | "roles"
  | "length"
  | "nonascii"
  | "markup";

export function classify(messages: ChatMessage[] | null): Reject {
  if (!messages) return "turns";
  const body = messages.filter((m) => m.role !== "system");
  if (body.length < 2 || body.length > 12) return "turns";
  // Must be a clean user/assistant alternation ending on the assistant.
  for (let i = 0; i < body.length; i++) {
    if (body[i].role !== (i % 2 === 0 ? "user" : "assistant")) return "roles";
  }
  if (body[body.length - 1].role !== "assistant") return "roles";

  let chars = 0;
  for (const m of messages) {
    const n = m.content.length;
    chars += n;
    if (m.role === "assistant" && (n < 20 || n > 900)) return "length";
    if (m.role === "user" && n > 700) return "length";
  }
  if (chars > 2500) return "length";

  const all = messages.map((m) => m.content).join("\n");
  if (/```|\\\(|\\\[|\$\$/.test(all)) return "markup";
  let exotic = 0;
  for (let i = 0; i < all.length; i++) if (all.charCodeAt(i) > 127) exotic++;
  if (exotic / all.length > 0.02) return "nonascii";
  return "ok";
}

/** Deterministic in-place Fisher-Yates against the repo's seeded RNG. */
export function shuffle<T>(items: T[], rng: () => number): T[] {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
  return items;
}

interface Seeded {
  id: string;
  config: string;
  messages: ChatMessage[];
}

async function sampleConfig(config: string, quota: number, seed: number): Promise<Seeded[]> {
  const urls = await fetchParquetUrls(DATASET, config, "train");
  if (!urls.length) die(`no parquet shards for ${DATASET} ${config}/train`);

  const want = quota * 5; // over-collect, then shuffle: shard order is topic-ordered
  const candidates: Seeded[] = [];
  const rejects = new Map<Reject, number>();
  const seenFirstUser = new Set<string>();
  let dupes = 0;
  // deno-lint-ignore no-explicit-any
  let mapping: any = null;

  for (const url of urls) {
    if (candidates.length >= want) break;
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    const rows = await parseDataFile(url, bytes);
    if (!mapping) mapping = detectMapping(rows[0]);
    if (!mapping) die(`could not detect a conversational mapping for ${config}`);
    for (let i = 0; i < rows.length && candidates.length < want; i++) {
      const messages = rowToMessages(rows[i], mapping);
      const verdict = classify(messages);
      if (verdict !== "ok") {
        rejects.set(verdict, (rejects.get(verdict) ?? 0) + 1);
        continue;
      }
      // Dedup on the whole conversation: in everyday-conversations thousands of
      // genuinely different chats share the opening exchange ("Hi" / "Hello! How
      // can I help you today?"), so any prefix key throws almost all of them away.
      const key = messages!.map((m) => m.content).join(" ");
      if (seenFirstUser.has(key)) {
        dupes++;
        continue;
      }
      seenFirstUser.add(key);
      candidates.push({ id: `${config}:${candidates.length}`, config, messages: messages! });
    }
  }

  const kept = shuffle(candidates, mulberry32(seed + config.length)).slice(0, quota);
  const skips = [...rejects].map(([k, v]) => `${k} ${v}`).join(", ");
  console.log(
    `  ${config}: ${kept.length}/${quota} kept from ${candidates.length} candidates ` +
      `(skipped ${skips || "none"}, dupes ${dupes})`,
  );
  return kept;
}

async function run(v: Values) {
  const total = v.num("total");
  const outPath = v.str("out");
  const seed = v.num("seed");
  const mix = v.str("mix").split(",").map((part) => {
    const [config, weight] = part.split(":");
    if (!config || !weight) throw new UsageError(`bad --mix entry "${part}", want config:weight`);
    return { config, weight: Number(weight) };
  });
  const weightSum = mix.reduce((s, m) => s + m.weight, 0);

  console.log(`=== style seed: ${total} conversations from ${DATASET} ===\n`);
  const out: Seeded[] = [];
  for (const { config, weight } of mix) {
    const quota = Math.max(1, Math.round((total * weight) / weightSum));
    out.push(...await sampleConfig(config, quota, seed));
  }
  shuffle(out, mulberry32(seed));

  const fs = await import("node:fs");
  const dir = outPath.slice(0, outPath.lastIndexOf("/"));
  if (dir) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(outPath, out.map((r) => JSON.stringify(r)).join("\n") + "\n");

  const assistantChars = out.reduce(
    (s, r) =>
      s + r.messages.filter((m) => m.role === "assistant")
        .reduce((t, m) => t + m.content.length, 0),
    0,
  );
  console.log(
    `\nwrote ${outPath}: ${out.length} conversations, ` +
      `${(assistantChars / 1000).toFixed(1)}k assistant chars to restyle`,
  );
  console.log(`next: style-restyle --in ${outPath}`);
}

export const styleSeedCommand: Command = {
  name: "style-seed",
  summary: "Sample a filtered chat corpus from smoltalk, ready to be rewritten.",
  details: `Optional workflow. Builds the input for \`style-restyle\`: a weighted, filtered,
deduplicated sample of HuggingFaceTB/smoltalk, the SFT set built for models in the 135M to
1.7B range.

The default mix covers turn-taking, system-prompt following and text transformation, and
leaves out the math, code and long-context configs: those are above what a sub-100M model
can learn and they teach it to be confidently wrong.

Filters: 2-12 messages, strict user/assistant alternation ending on the assistant, answers
20-900 chars, conversation under 2500 chars, near-ASCII, no code fences or LaTeX.`,
  examples: [
    "style-seed --total 200 --out corpus/style/seed.jsonl",
    "style-seed --total 2000 --out corpus/style/seed.jsonl --mix everyday-conversations:50,smol-summarize:50",
  ],
  flags: [
    {
      name: "out",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "output .jsonl, one conversation per line",
    },
    {
      name: "total",
      type: "number",
      default: 200,
      describe: "conversations to sample, split across the mix by weight",
    },
    {
      name: "mix",
      type: "string",
      placeholder: "CFG:W,...",
      default: DEFAULT_MIX,
      describe: "smoltalk configs and their weights",
    },
    {
      name: "seed",
      type: "number",
      default: 1234,
      describe: "sampling seed; the same seed reproduces the same corpus",
    },
  ],
  run: run,
};
