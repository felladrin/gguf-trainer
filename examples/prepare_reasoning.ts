// Build a reasoning SFT corpus for a from-scratch Gemma3 model: download a
// recent reasoning dataset (default open-thoughts/OpenThoughts-114k — diverse
// math/code/science/logic with long chain-of-thought), normalize each example
// to ChatML with the thinking wrapped in <think>…</think> special tokens, train
// a byte-level BPE with those specials, and pretokenize to a .tokens stream the
// trainer streams off disk.
//
//   deno run -A examples/prepare_reasoning.ts [maxRows] [vocab] [outPrefix] [dataset]
//
// Writes <outPrefix>.tokens, <outPrefix>.tokenizer.json, <outPrefix>.template.txt.

import { fetchParquetUrls } from "../web/server/hf.ts";
import { parseDataFile } from "../web/server/parse.ts";
import {
  type ChatMessage,
  CHATML_SPECIALS,
  detectMapping,
  rowToMessages,
} from "../src/data/chat.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { tokenBytes, writeTokenFile } from "../src/data/tokens.ts";
import { writeFileBytes } from "../src/io.ts";

// deno-lint-ignore no-explicit-any
const A: string[] = (globalThis as any).Deno?.args ?? [];
const maxRows = A[0] ? Number(A[0]) : 40000;
const vocab = A[1] ? Number(A[1]) : 32768;
const outPrefix = A[2] ?? "examples/reasoning";
const dataset = A[3] ?? "open-thoughts/OpenThoughts-114k";

// Thinking special tokens (the user's requirement) + ChatML turn markers.
const THINK = ["<think>", "</think>"];
const SPECIALS = [...CHATML_SPECIALS, ...THINK];

// A minimal ChatML template: training rendering == inference rendering, so the
// model sees exactly what llama-server will feed it. Assistant content already
// carries <think>…</think>, so nothing arch-specific is needed here.
const CHAT_TEMPLATE =
  `{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}`;

/** Remap OpenThoughts / R1-style thinking markers to our <think> specials. */
function remapThinking(s: string): string {
  return s
    .replaceAll("<|begin_of_thought|>", "<think>")
    .replaceAll("<|end_of_thought|>", "</think>")
    .replaceAll("<|begin_of_solution|>", "")
    .replaceAll("<|end_of_solution|>", "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function renderChatML(msgs: ChatMessage[]): string {
  let out = "";
  for (const m of msgs) {
    const content = m.role === "assistant" ? remapThinking(m.content) : m.content.trim();
    out += `<|im_start|>${m.role}\n${content}<|im_end|>\n`;
  }
  return out;
}

console.log(`dataset: ${dataset} | target ${maxRows} rows | vocab ${vocab}`);
const urls = await fetchParquetUrls(dataset, "default", "train");
if (!urls.length) throw new Error("no parquet shards found for default/train");
console.log(`${urls.length} parquet shards available`);

// Keep conversations as separate strings (never join into one — V8 caps a
// single string at ~512MB, and a full corpus is larger). A bounded prefix is
// concatenated only for BPE training.
const convos: string[] = [];
let mapping: ReturnType<typeof detectMapping> = null;
let withThink = 0;
let totalChars = 0;
for (const url of urls) {
  if (convos.length >= maxRows) break;
  console.log(`downloading ${url.split("/").pop()} …`);
  const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
  const rows = await parseDataFile(url, bytes);
  if (!mapping) mapping = detectMapping(rows[0]);
  if (!mapping) throw new Error("could not detect a conversational mapping");
  for (const row of rows) {
    if (convos.length >= maxRows) break;
    const msgs = rowToMessages(row, mapping);
    if (!msgs || msgs.length < 2) continue;
    const text = renderChatML(msgs);
    if (text.includes("<think>")) withThink++;
    convos.push(text);
    totalChars += text.length;
  }
  console.log(`  ${convos.length}/${maxRows} conversations (${withThink} with <think>)`);
}
console.log(`corpus: ${(totalChars / 1e6).toFixed(1)}M chars from ${convos.length} conversations`);

// Train BPE on a bounded prefix (specials are atomic).
const tok = new BPETokenizer();
const sampleParts: string[] = [];
let sampleLen = 0;
for (const c of convos) {
  if (sampleLen >= 12_000_000) break;
  sampleParts.push(c);
  sampleLen += c.length;
}
const sample = sampleParts.join("");
console.log(`training BPE (vocab ${vocab}) on ${(sample.length / 1e6).toFixed(1)}M-char sample …`);
tok.train(sample, vocab, SPECIALS);
// Chat model: stop on <|im_end|>.
const imEnd = tok.idOf("<|im_end|>");
if (imEnd !== undefined) tok.eosId = imEnd;
console.log(
  `tokenizer: vocab=${tok.vocabSize}, eos=${tok.eosId} (<|im_end|>), ` +
    `think ids=${THINK.map((t) => tok.idOf(t)).join(",")}`,
);

// Encode conversation-by-conversation into one id stream (avoids the giant
// string; specials stay atomic within each conversation).
console.log("encoding corpus (per conversation) …");
const t0 = performance.now();
const ids: number[] = [];
for (let i = 0; i < convos.length; i++) {
  for (const x of tok.encode(convos[i])) ids.push(x);
  if (i > 0 && i % 4000 === 0) {
    console.log(`  ${i}/${convos.length} convos, ${(ids.length / 1e6).toFixed(1)}M tokens`);
  }
}
console.log(
  `encoded ${(ids.length / 1e6).toFixed(2)}M tokens in ${
    ((performance.now() - t0) / 1000).toFixed(0)
  }s ` +
    `(${(ids.length / convos.length).toFixed(0)} tok/convo)`,
);

await writeTokenFile(`${outPrefix}.tokens`, ids, tokenBytes(vocab));
await writeFileBytes(
  `${outPrefix}.tokenizer.json`,
  new TextEncoder().encode(JSON.stringify(tok.export())),
);
await writeFileBytes(`${outPrefix}.template.txt`, new TextEncoder().encode(CHAT_TEMPLATE));
console.log(
  `\nwrote ${outPrefix}.tokens (${tokenBytes(vocab)}B/token), .tokenizer.json, .template.txt`,
);
console.log(`sample decode:\n${tok.decode(ids.slice(0, 80))}`);
