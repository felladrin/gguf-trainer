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
// The corpus may span SEVERAL part files (comma-separated) — each part must stay
// under V8's ~512 MB string cap, but the token output is streamed shard-by-shard,
// so the total corpus can be multi-GB. The vocab is trained on a sample of the
// FIRST part (prepare_pretrain.ts shuffles, so that prefix is representative).
//
// Documents are split on the "<|endoftext|>" marker (TinyStories and most raw
// dumps use it) and rejoined with the single EOS token id, so the model sees a
// clean boundary token between documents instead of the literal characters.
//
// Usage (Deno has the file APIs everywhere via node:fs):
//   deno run -A examples/pretokenize.ts <in.txt[,in2.txt,…]> <out-prefix> [vocabSize] [trainSampleMB] [curriculum]
// e.g.
//   deno run -A examples/pretokenize.ts corpus/tinystories-valid.txt examples/tinystories 8192 10
//   deno run -A examples/pretokenize.ts corpus/blend-p1.txt,corpus/blend-p2.txt examples/blend 32768 32 curriculum
//
// Pass `curriculum` as the 5th arg to reserve CURRICULUM_SPECIALS (ChatML turns,
// think/tool tags) in the vocab — REQUIRED when the tokens feed pretrain.ts for
// a curriculum base: the vocab and embeddings freeze at pretrain, so a later
// stage cannot add a special token without discarding trained embeddings.

import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import type { TokenizerData } from "../src/tokenizer/bpe.ts";
import { readFileText, writeFileBytes } from "../src/io.ts";
import { diskTokenSource, tokenBytes } from "../src/data/tokens.ts";
import { CURRICULUM_SPECIALS } from "../src/data/chat.ts";

const DOC_SEP = "<|endoftext|>";

function args(): string[] {
  // deno-lint-ignore no-explicit-any
  const g = globalThis as any;
  return g.Deno?.args ?? g.process?.argv?.slice(2) ?? [];
}

function fail(msg: string): never {
  console.error("pretokenize: " + msg);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(1);
  throw new Error(msg);
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

async function main() {
  const [inArg, outPrefix, vocabArg, sampleArg, specialsArg] = args();
  if (!inArg || !outPrefix) {
    fail(
      "usage: pretokenize <in.txt[,in2.txt,…]> <out-prefix> [vocabSize=8192] [trainSampleMB=10] [curriculum]",
    );
  }
  const inPaths = inArg.split(",").map((p) => p.trim()).filter((p) => p.length > 0);
  const vocabSize = vocabArg ? Number(vocabArg) : 8192;
  const trainSampleMB = sampleArg ? Number(sampleArg) : 10;
  const specials = specialsArg === "curriculum" ? CURRICULUM_SPECIALS : undefined;
  if (specialsArg && !specials) {
    fail(`unknown specials set "${specialsArg}" (expected: curriculum)`);
  }
  if (!Number.isFinite(vocabSize) || vocabSize < 257) {
    fail(`vocabSize must be >= 257 (256 bytes + specials), got ${vocabArg}`);
  }
  if (!Number.isFinite(trainSampleMB) || trainSampleMB <= 0) {
    fail(`trainSampleMB must be > 0, got ${sampleArg}`);
  }

  console.log("=== Felladrin's GGUF Trainer +∞ :: pretokenize ===\n");

  // 1. Train BPE on a bounded sample of the FIRST part (or reuse an existing
  //    tokenizer). Cutting mid-document is fine for a vocab; strip doc markers
  //    so "<|endoftext|>" pieces don't pollute the merges. The tokenizer is
  //    saved BEFORE encoding, so a failure in the (long) encode never loses the
  //    (long) vocab training — a re-run reuses it.
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
  console.log(`\n=== pretokenize OK (${((Date.now() - t0) / 1000).toFixed(1)}s total) ===`);
}

main().catch((e) => fail(String(e?.stack ?? e)));
