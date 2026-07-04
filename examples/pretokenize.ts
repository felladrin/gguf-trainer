// Pretokenize a raw UTF-8 text corpus into the trainer's binary token format.
//
// Trains a byte-level BPE vocab on a bounded sample of the corpus (BPE converges
// on a few MB; encoding the whole thing to learn the vocab is wasteful), then
// encodes the entire corpus document-by-document and writes two files:
//
//   <out-prefix>.tokens          bare little-endian tokens (writeTokenFile)
//   <out-prefix>.tokenizer.json  exported vocab + merges (BPETokenizer.export),
//                                so the training run and inference reuse the exact
//                                vocab the corpus was tokenized with.
//
// Documents are split on the "<|endoftext|>" marker (TinyStories and most raw
// dumps use it) and rejoined with the single EOS token id, so the model sees a
// clean boundary token between stories instead of the literal characters.
//
// This is the input to the real training run (examples/train_streaming.ts and
// examples/demo_gpu.ts both accept a diskTokenSource over the .tokens file).
//
// Usage (Deno has the file APIs everywhere via node:fs):
//   deno run -A examples/pretokenize.ts <input.txt> <out-prefix> [vocabSize] [trainSampleMB]
// e.g.
//   deno run -A examples/pretokenize.ts corpus/tinystories-valid.txt examples/tinystories 8192 10

import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { readFileText, writeFileBytes } from "../src/io.ts";
import { diskTokenSource, tokenBytes, writeTokenFile } from "../src/data/tokens.ts";

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

async function main() {
  const [inPath, outPrefix, vocabArg, sampleArg] = args();
  if (!inPath || !outPrefix) {
    fail(
      "usage: pretokenize <input.txt> <out-prefix> [vocabSize=8192] [trainSampleMB=10]",
    );
  }
  const vocabSize = vocabArg ? Number(vocabArg) : 8192;
  const trainSampleMB = sampleArg ? Number(sampleArg) : 10;
  if (!Number.isFinite(vocabSize) || vocabSize < 257) {
    fail(`vocabSize must be >= 257 (256 bytes + specials), got ${vocabArg}`);
  }
  if (!Number.isFinite(trainSampleMB) || trainSampleMB <= 0) {
    fail(`trainSampleMB must be > 0, got ${sampleArg}`);
  }

  console.log("=== Felladrin's GGUF Trainer +∞ :: pretokenize ===\n");

  // 1. Read the corpus and split it into documents.
  const t0 = Date.now();
  const text = await readFileText(inPath);
  if (text.length === 0) fail(`${inPath} is empty`);
  const docs = text.split(DOC_SEP).map((d) => d.trim()).filter((d) => d.length > 0);
  if (docs.length === 0) fail(`${inPath} has no non-empty documents`);
  console.log(
    `Corpus: ${inPath}  ${(text.length / 1e6).toFixed(1)}M chars, ${docs.length} documents ` +
      `(read in ${((Date.now() - t0) / 1000).toFixed(1)}s)`,
  );

  // 2. Train BPE on a bounded sample. Cutting mid-document is fine for a vocab;
  //    strip the doc markers so "<|endoftext|>" pieces don't pollute the merges.
  const sampleChars = Math.floor(trainSampleMB * 1024 * 1024);
  const sample = text.slice(0, sampleChars).split(DOC_SEP).join("\n");
  const tTrain = Date.now();
  const tok = new BPETokenizer();
  tok.train(sample, vocabSize);
  console.log(
    `Vocab: trained to ${tok.vocabSize} tokens on ${(sample.length / 1e6).toFixed(1)}M-char ` +
      `sample (${tok.merges.length} merges, ${((Date.now() - tTrain) / 1000).toFixed(1)}s)`,
  );

  // 3. Encode the whole corpus, one document at a time, EOS between documents.
  const tEnc = Date.now();
  const ids: number[] = [];
  let chars = 0;
  for (let i = 0; i < docs.length; i++) {
    for (const id of tok.encode(docs[i])) ids.push(id);
    ids.push(tok.eosId);
    chars += docs[i].length;
    if (i > 0 && i % 20000 === 0) {
      const pct = ((i / docs.length) * 100).toFixed(0);
      console.log(`  encoded ${i}/${docs.length} docs (${pct}%), ${ids.length} tokens so far`);
    }
  }
  const encSecs = (Date.now() - tEnc) / 1000;
  console.log(
    `Encoded: ${ids.length} tokens from ${docs.length} docs in ${encSecs.toFixed(1)}s ` +
      `(${(chars / ids.length).toFixed(2)} chars/token)`,
  );

  // 4. Write the token file + the tokenizer, so the run reuses this exact vocab.
  const bpt = tokenBytes(tok.vocabSize);
  const tokensPath = `${outPrefix}.tokens`;
  const tokenizerPath = `${outPrefix}.tokenizer.json`;
  await writeTokenFile(tokensPath, ids, bpt);
  await writeFileBytes(tokenizerPath, new TextEncoder().encode(JSON.stringify(tok.export())));
  console.log(
    `\nWrote ${tokensPath} (${bpt} B/token, ${((ids.length * bpt) / 1e6).toFixed(1)} MB on disk)` +
      `\nWrote ${tokenizerPath} (vocab + merges for reuse)`,
  );

  // 5. Self-check: reopen the token file and confirm it round-trips the ids.
  const src = await diskTokenSource(tokensPath, bpt);
  if (src.length !== ids.length) {
    fail(`round-trip length mismatch: wrote ${ids.length}, read ${src.length}`);
  }
  for (let probe = 0; probe < 500; probe++) {
    // Deterministic stride across the file — no RNG needed for a spot check.
    const start = Math.floor((probe / 500) * (src.length - 32));
    const w = src.window(start, 32);
    for (let i = 0; i < 32; i++) {
      if (w[i] !== ids[start + i]) fail(`round-trip mismatch at ${start + i}`);
    }
  }
  src.close();
  console.log("Round-trip: disk token file matches encoded ids over 500 probes ✓");
  console.log(`\n=== pretokenize OK (${((Date.now() - t0) / 1000).toFixed(1)}s total) ===`);
}

main().catch((e) => fail(String(e?.stack ?? e)));
