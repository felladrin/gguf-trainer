// Build an instruct SFT corpus for a base model: download a chat dataset
// (default HuggingFaceTB/smol-smoltalk: short, teacher-generated multi-turn
// chat, the natural first curriculum stage after pretraining), render each
// conversation through the SAME Jinja chat template the exporter embeds, encode
// it with the BASE MODEL'S EXISTING VOCAB, and write the token stream plus an
// assistant-only supervision mask.
//
// --data takes a Hugging Face dataset id, or a local .jsonl of {messages: [...]}
// rows: that second form is how a `style-seed` -> `style-restyle` corpus reaches
// the trainer.
//
// Writes <out>.tokens, <out>.mask, <out>.tokenizer.json, <out>.template.txt.
//
// This NEVER trains a vocab: the embedding matrix
// froze at pretraining, so a new vocab would be incompatible with the
// checkpoint. The base tokenizer already reserves every curriculum special
// (CURRICULUM_SPECIALS), which is checked below before anything is encoded.

import { Template } from "@huggingface/jinja";
import { fetchParquetUrls } from "../data/hf.ts";
import { parseDataFile } from "../data/parse.ts";
import {
  assistantLossMask,
  CURRICULUM_SPECIALS,
  DEFAULT_CHAT_TEMPLATE,
  detectMapping,
  rowToMessages,
} from "../data/chat.ts";
import { BPETokenizer } from "../tokenizer/bpe.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";
import { diskTokenSource, tokenBytes, writeTokenFile } from "../data/tokens.ts";
import { readFileBytes, readFileText, writeFileBytes } from "../io.ts";
import type { Command, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";

function die(msg: string): never {
  throw new UsageError(msg);
}

async function run(v: Values) {
  const maxRows = v.num("max-rows");
  const outPrefix = v.str("out");
  const dataset = v.str("data");
  const tokenizerPath = v.str("tokenizer");

  // --- 1. The base model's vocab, reused verbatim -----------------------------

  let tokData: TokenizerData;
  try {
    tokData = JSON.parse(await readFileText(tokenizerPath)) as TokenizerData;
  } catch (e) {
    die(`cannot read ${tokenizerPath} (${e}): the SFT corpus must reuse the base vocab`);
  }
  const tok = BPETokenizer.fromData(tokData);
  const missing = CURRICULUM_SPECIALS.filter((s) => tok.idOf(s) === undefined);
  if (missing.length) {
    die(
      `${tokenizerPath} lacks curriculum specials [${missing.join(", ")}]: it was not built with ` +
        `the "curriculum" specials set, so no later stage can use them`,
    );
  }
  if (tokenBytes(tok.vocabSize) !== 2) {
    die(`vocab ${tok.vocabSize} needs u32 tokens; widen this path`);
  }
  const imStart = tok.idOf("<|im_start|>")!;
  const imEnd = tok.idOf("<|im_end|>")!;
  const sepId = tok.idOf("<|endoftext|>")!; // hard boundary between conversations
  // A chat model stops at <|im_end|>; the base stopped at <|endoftext|>. Vocab and
  // merges are untouched, so this only changes the exported EOS metadata.
  tok.eosId = imEnd;
  console.log(
    `vocab: reused ${tokenizerPath} (${tok.vocabSize} tokens, all ${CURRICULUM_SPECIALS.length} ` +
      `curriculum specials present), eos=${tok.eosId} (<|im_end|>)`,
  );

  // --- 2. Rows -> ChatML strings ----------------------------------------------

  // The exporter's template, rendered here so training text == inference text.
  const render = new Template(DEFAULT_CHAT_TEMPLATE);

  console.log(`dataset: ${dataset} | target ${maxRows} rows`);
  // A local .jsonl / .json (the restyled style corpus from restyle_with_pi.ts) is
  // read from disk; anything else is a Hugging Face dataset id, streamed shard by
  // shard. Both end up as rows for the same mapping/render path below.
  const local = /\.(jsonl|json)$/.test(dataset);
  const urls = local ? [dataset] : await fetchParquetUrls(dataset, "default", "train");
  if (!urls.length) die(`no parquet shards found for ${dataset} default/train`);
  console.log(local ? `local file` : `${urls.length} parquet shards available`);

  const convos: string[] = [];
  let mapping: ReturnType<typeof detectMapping> = null;
  let skipped = 0, totalChars = 0;
  for (const url of urls) {
    if (convos.length >= maxRows) break;
    console.log(`${local ? "reading" : "downloading"} ${url.split("/").pop()} …`);
    const bytes = local
      ? await readFileBytes(url)
      : new Uint8Array(await (await fetch(url)).arrayBuffer());
    const rows = await parseDataFile(url, bytes);
    if (!mapping) mapping = detectMapping(rows[0]);
    if (!mapping) die("could not detect a conversational mapping");
    for (const row of rows) {
      if (convos.length >= maxRows) break;
      const messages = rowToMessages(row, mapping);
      if (!messages || messages.length < 2) {
        skipped++;
        continue;
      }
      let text: string;
      try {
        text = render.render({ messages, add_generation_prompt: false });
      } catch {
        skipped++; // a row the template cannot render
        continue;
      }
      if (!text.trim()) {
        skipped++;
        continue;
      }
      convos.push(text);
      totalChars += text.length;
    }
    console.log(`  ${convos.length}/${maxRows} conversations (${skipped} skipped)`);
  }
  if (!convos.length) die("no usable conversations");
  console.log(
    `corpus: ${(totalChars / 1e6).toFixed(1)}M chars from ${convos.length} conversations`,
  );

  // --- 3. Encode + assistant-only mask ----------------------------------------

  // Growable typed arrays: a plain number[] hits V8's max fast-array length past
  // ~10^8 tokens, and a single joined string would exceed the ~512MB string cap.
  console.log("encoding corpus (per conversation, with assistant-only mask) …");
  const t0 = performance.now();
  let ids = new Uint16Array(1 << 24);
  let sup = new Uint8Array(1 << 24);
  let len = 0;
  const decode = (x: number[]) => tok.decode(x);
  for (let i = 0; i < convos.length; i++) {
    const e = tok.encode(convos[i]);
    const m = assistantLossMask(e, imStart, imEnd, decode);
    if (len + e.length + 1 > ids.length) {
      const cap = Math.max(ids.length * 2, len + e.length + 1);
      const grownIds = new Uint16Array(cap);
      grownIds.set(ids.subarray(0, len));
      ids = grownIds;
      const grownSup = new Uint8Array(cap);
      grownSup.set(sup.subarray(0, len));
      sup = grownSup;
    }
    ids.set(e, len);
    sup.set(m, len);
    len += e.length;
    ids[len] = sepId; // separator is scaffolding, never supervised
    sup[len] = 0;
    len++;
    if (i > 0 && i % 10000 === 0) {
      console.log(`  ${i}/${convos.length} convos, ${(len / 1e6).toFixed(1)}M tokens`);
    }
  }
  const stream = ids.subarray(0, len);
  const mask = sup.subarray(0, len);
  let kept = 0;
  for (let i = 0; i < len; i++) kept += mask[i];
  console.log(
    `encoded ${(len / 1e6).toFixed(2)}M tokens in ${
      ((performance.now() - t0) / 1000).toFixed(0)
    }s ` +
      `(${(len / convos.length).toFixed(0)} tok/convo, ${
        (totalChars / len).toFixed(2)
      } chars/token)`,
  );
  console.log(
    `assistant-only loss mask: ${(kept / 1e6).toFixed(2)}M/${(len / 1e6).toFixed(2)}M tokens ` +
      `supervised (${(100 * kept / len).toFixed(1)}%)`,
  );
  if (kept === 0) die("mask supervises nothing: check the ChatML specials and the template");

  // --- 4. Write + round-trip check --------------------------------------------

  const bpt = tokenBytes(tok.vocabSize);
  await writeTokenFile(`${outPrefix}.tokens`, stream, bpt);
  await writeTokenFile(`${outPrefix}.mask`, mask, bpt);
  await writeFileBytes(
    `${outPrefix}.tokenizer.json`,
    new TextEncoder().encode(JSON.stringify(tok.export())),
  );
  await writeFileBytes(
    `${outPrefix}.template.txt`,
    new TextEncoder().encode(DEFAULT_CHAT_TEMPLATE),
  );

  const src = await diskTokenSource(`${outPrefix}.tokens`, bpt);
  const maskSrc = await diskTokenSource(`${outPrefix}.mask`, bpt);
  if (src.length !== len) die(`round-trip length mismatch: wrote ${len}, read ${src.length}`);
  if (maskSrc.length !== len) die(`mask length ${maskSrc.length} != tokens ${len}`);
  const head = src.window(0, 64);
  for (let i = 0; i < head.length; i++) {
    if (head[i] !== stream[i]) die(`round-trip mismatch at token ${i}`);
  }
  src.close();
  maskSrc.close();

  console.log(
    `\nwrote ${outPrefix}.tokens (${bpt}B/token, ${((len * bpt) / 1e6).toFixed(1)} MB), ` +
      `${outPrefix}.mask, .tokenizer.json, .template.txt`,
  );
  console.log(`round-trip: disk token file matches the encoded head ✓`);
  console.log(
    `\nsample decode (first 120 tokens):\n${tok.decode(Array.from(stream.subarray(0, 120)))}`,
  );
}

export const chatCorpusCommand: Command = {
  name: "chat-corpus",
  summary: "Turn a chat dataset into SFT tokens plus an assistant-only loss mask.",
  details: `Renders every conversation through the exporter's own chat template, encodes it with
the BASE MODEL'S tokenizer, and writes four files:

  <out>.tokens         the token stream
  <out>.mask           1 where a token is part of an assistant turn, 0 elsewhere
  <out>.tokenizer.json the same vocab, with EOS moved to <|im_end|>
  <out>.template.txt   the chat template to embed in the fine-tuned GGUF

Feed all four to \`finetune\`. The tokenizer is REUSED, never retrained: the embedding matrix
froze when pretraining started, so a new vocab would not match the checkpoint. The command
aborts if the tokenizer lacks the ChatML specials.

--data takes a Hugging Face dataset id, or a local .jsonl of {"messages": [...]} rows. The
usual shapes (messages, ShareGPT conversations, Alpaca instruction/output) are detected
automatically.`,
  examples: [
    "chat-corpus --data HuggingFaceTB/smol-smoltalk --tokenizer data/blend.tokenizer.json --out data/chat",
    "chat-corpus --data ./my-conversations.jsonl --tokenizer data/blend.tokenizer.json --out data/chat --max-rows 5000",
  ],
  flags: [
    {
      name: "data",
      type: "string",
      placeholder: "ID|PATH",
      required: true,
      describe: "Hugging Face dataset id, or a local .jsonl of conversations",
    },
    {
      name: "out",
      type: "string",
      placeholder: "PREFIX",
      required: true,
      describe: "output prefix for the .tokens, .mask, .tokenizer.json and .template.txt",
    },
    {
      name: "tokenizer",
      type: "string",
      placeholder: "PATH",
      required: true,
      describe: "the base model's tokenizer.json, reused verbatim",
    },
    {
      name: "max-rows",
      type: "number",
      default: 100000,
      describe: "stop after this many usable conversations",
    },
  ],
  run: run,
};
