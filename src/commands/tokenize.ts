// Pretokenize a raw UTF-8 text corpus into the trainer's binary token format.
//
// Trains a byte-level BPE vocab on a bounded sample of the corpus (BPE converges
// on a few MB; encoding the whole thing to learn the vocab is wasteful), then
// encodes the entire corpus document-by-document and writes two files:
//
//   <out-prefix>.tokens          bare little-endian tokens (tokenBytes-wide)
//   <out-prefix>.tokenizer.json  exported vocab + merges (BPETokenizer.export),
//                                so the training run and inference reuse the exact
//                                vocab the corpus was tokenized with.
//
// The corpus may span SEVERAL part files (comma-separated), and each part must stay
// under V8's ~512 MB string cap, but the token output is streamed shard-by-shard,
// so the total corpus can be multi-GB. The vocab is trained on a sample of the
// FIRST part (`corpus` shuffles its output, so that prefix is representative).
//
// Documents are split on the "<|endoftext|>" marker (TinyStories and most raw
// dumps use it) and rejoined with the single EOS token id, so the model sees a
// clean boundary token between documents instead of the literal characters.
//
// --curriculum-specials reserves CURRICULUM_SPECIALS (ChatML turns, think and
// tool tags) in the vocab. It is REQUIRED when the model will later be
// fine-tuned for chat: the vocab and embeddings freeze at pretraining, so a
// later stage cannot add a special token without discarding trained embeddings.

import { BPETokenizer } from "../tokenizer/bpe.ts";
import type { TokenizerData } from "../tokenizer/bpe.ts";
import { readFileText, writeFileBytes } from "../io.ts";
import { diskTokenSource, tokenBytes } from "../data/tokens.ts";
import { CURRICULUM_SPECIALS } from "../data/chat.ts";
import type { Command, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";

const DOC_SEP = "<|endoftext|>";

function fail(msg: string): never {
  throw new UsageError(msg);
}

/** Encode docs into a growable Uint32Array (a number[] hits V8's max-array-length
 * near 10^8 tokens), one document at a time with EOS between documents. */
function encodeDocs(tok: BPETokenizer, docs: string[]): Uint32Array {
  let cap = 1 << 20, n = 0;
  let ids = new Uint32Array(cap);
  const push = (id: number) => {
    if (n >= cap) {
      cap *= 2;
      const grown = new Uint32Array(cap);
      grown.set(ids);
      ids = grown;
    }
    ids[n++] = id;
  };
  for (let i = 0; i < docs.length; i++) {
    for (const id of tok.encode(docs[i])) push(id);
    push(tok.eosId);
    if (i > 0 && i % 20000 === 0) {
      console.log(`  encoded ${i}/${docs.length} docs (${(i / docs.length * 100).toFixed(0)}%)`);
    }
  }
  return ids.subarray(0, n);
}

async function run(v: Values) {
  const inArg = v.str("text");
  const outPrefix = v.str("out");
  const vocabSize = v.num("vocab");
  const trainSampleMB = v.num("sample-mb");
  const specials = v.bool("curriculum-specials") ? CURRICULUM_SPECIALS : undefined;
  if (vocabSize < 257) {
    fail(`--vocab must be >= 257 (256 byte tokens + specials), got ${vocabSize}`);
  }
  if (trainSampleMB <= 0) fail(`--sample-mb must be > 0, got ${trainSampleMB}`);
  const inPaths = inArg.split(",").map((p) => p.trim()).filter((p) => p.length > 0);

  console.log("=== Felladrin's GGUF Trainer +∞ :: tokenize ===\n");

  // 1. Train BPE on a bounded sample of the FIRST part (or reuse an existing
  //    tokenizer). Cutting mid-document is fine for a vocab; strip doc markers
  //    so "<|endoftext|>" pieces don't pollute the merges. The tokenizer is
  //    saved BEFORE encoding, so a failure in the (long) encode never loses the
  //    (long) vocab training: a re-run reuses it.
  const t0 = Date.now();
  const fs = await import("node:fs");
  const tokenizerPath = `${outPrefix}.tokenizer.json`;
  let text = await readFileText(inPaths[0]);
  if (text.length === 0) fail(`${inPaths[0]} is empty`);
  let tok: BPETokenizer;
  if (fs.existsSync(tokenizerPath)) {
    tok = BPETokenizer.fromData(JSON.parse(await readFileText(tokenizerPath)) as TokenizerData);
    console.log(`Vocab: reused ${tokenizerPath} (${tok.vocabSize} tokens)`);
  } else {
    const sampleChars = Math.floor(trainSampleMB * 1024 * 1024);
    const sample = text.slice(0, sampleChars).split(DOC_SEP).join("\n");
    const tTrain = Date.now();
    tok = new BPETokenizer();
    tok.train(sample, vocabSize, specials);
    await writeFileBytes(tokenizerPath, new TextEncoder().encode(JSON.stringify(tok.export())));
    console.log(
      `Vocab: trained to ${tok.vocabSize} tokens on ${(sample.length / 1e6).toFixed(1)}M-char ` +
        `sample (${tok.merges.length} merges${
          specials ? `, ${specials.length} curriculum specials reserved` : ""
        }, ${((Date.now() - tTrain) / 1000).toFixed(1)}s) -> ${tokenizerPath}`,
    );
  }
  const bpt = tokenBytes(tok.vocabSize);

  // 2. Encode part by part, appending each part's tokens to the output as we go
  //    so peak memory stays O(one part) however large the whole corpus is.
  const tokensPath = `${outPrefix}.tokens`;
  const fd = fs.openSync(tokensPath, "w");
  let totalTokens = 0, totalChars = 0, totalDocs = 0;
  const probes: { offset: number; ids: number[] }[] = [];
  try {
    for (let p = 0; p < inPaths.length; p++) {
      if (p > 0) text = await readFileText(inPaths[p]);
      if (text.length === 0) fail(`${inPaths[p]} is empty`);
      const docs = text.split(DOC_SEP).map((d) => d.trim()).filter((d) => d.length > 0);
      if (docs.length === 0) fail(`${inPaths[p]} has no non-empty documents`);
      console.log(
        `[${p + 1}/${inPaths.length}] ${inPaths[p]}: ${(text.length / 1e6).toFixed(1)}M chars, ` +
          `${docs.length} docs`,
      );
      const tEnc = Date.now();
      const ids = encodeDocs(tok, docs);
      probes.push({ offset: totalTokens, ids: Array.from(ids.subarray(0, 32)) });
      const bytes = new Uint8Array(ids.length * bpt);
      const dv = new DataView(bytes.buffer);
      for (let i = 0; i < ids.length; i++) {
        if (bpt === 2) dv.setUint16(i * 2, ids[i], true);
        else dv.setUint32(i * 4, ids[i], true);
      }
      fs.writeSync(fd, bytes);
      totalTokens += ids.length;
      totalChars += docs.reduce((s, d) => s + d.length, 0);
      totalDocs += docs.length;
      console.log(
        `  ${ids.length} tokens in ${((Date.now() - tEnc) / 1000).toFixed(1)}s ` +
          `(running total ${totalTokens}, ${(totalChars / totalTokens).toFixed(2)} chars/token)`,
      );
    }
  } finally {
    fs.closeSync(fd);
  }

  console.log(
    `\nWrote ${tokensPath} (${bpt} B/token, ${((totalTokens * bpt) / 1e6).toFixed(1)} MB, ` +
      `${totalDocs} docs); tokenizer already at ${tokenizerPath}`,
  );

  // 4. Self-check: reopen the token file and confirm each part's head round-trips.
  const src = await diskTokenSource(tokensPath, bpt);
  if (src.length !== totalTokens) {
    fail(`round-trip length mismatch: wrote ${totalTokens}, read ${src.length}`);
  }
  for (const probe of probes) {
    const w = src.window(probe.offset, probe.ids.length);
    for (let i = 0; i < probe.ids.length; i++) {
      if (w[i] !== probe.ids[i]) fail(`round-trip mismatch at ${probe.offset + i}`);
    }
  }
  src.close();
  console.log(`Round-trip: disk token file matches encoded ids at ${probes.length} part head(s) ✓`);
  console.log(`\n=== tokenize OK (${((Date.now() - t0) / 1000).toFixed(1)}s total) ===`);
}

export const tokenizeCommand: Command = {
  name: "tokenize",
  summary: "Turn a text corpus into the binary token stream the trainer reads.",
  details: `Trains a byte-level BPE vocab on a bounded sample of the corpus, then encodes the
whole thing and writes two files:

  <out>.tokens           the token stream the trainer memory-maps
  <out>.tokenizer.json   the vocab and merges, which every later stage MUST reuse verbatim

Pass --curriculum-specials when the model will later be fine-tuned for chat, reasoning or
tool calls. The vocab and embedding matrix freeze when pretraining starts, so those special
tokens have to exist before the first step or no later stage can use them.

--text accepts several comma-separated part files. Each part must stay under ~480 MB (V8
caps a single string), but the token output is streamed, so the corpus itself can be
multi-GB. The vocab is trained on a sample of the first part.`,
  examples: [
    "tokenize --text corpus/blend.txt --out data/blend --vocab 32768 --curriculum-specials",
    "tokenize --text corpus/p1.txt,corpus/p2.txt --out data/big --vocab 32768",
  ],
  flags: [
    {
      name: "text",
      type: "string",
      placeholder: "PATH[,PATH]",
      required: true,
      describe: "raw UTF-8 corpus file(s), comma-separated; documents split on <|endoftext|>",
    },
    {
      name: "out",
      type: "string",
      placeholder: "PREFIX",
      required: true,
      describe: "output prefix; writes <prefix>.tokens and <prefix>.tokenizer.json",
    },
    {
      name: "vocab",
      type: "number",
      default: 8192,
      describe: "BPE vocab size, including the 256 byte tokens and any specials",
    },
    {
      name: "sample-mb",
      type: "number",
      default: 10,
      describe: "megabytes of the corpus used to train the vocab",
    },
    {
      name: "curriculum-specials",
      type: "boolean",
      describe:
        "reserve the ChatML, <think> and tool-call tokens now, so a later fine-tune can use them",
    },
  ],
  run: run,
};
