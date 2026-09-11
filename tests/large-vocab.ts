// Standalone assert check for the u16 ceiling on token ids.
//
// Qwen3 is 151,936 tokens, Llama-3 128,256, Gemma 262,144. A fixed-width u16
// accumulator does not fail on those: it wraps each id into a smaller one that is
// itself perfectly legal, so a corrupted SFT corpus looks healthy and trains to
// garbage. That silence is the reason this check exists.
//
// The on-disk round-trip at both widths is covered by gradcheck.ts; this file is
// about picking the width and about what goes wrong when it is picked wrong.
// Run:  deno run tests/large-vocab.ts
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  assertCorpusFitsVocab,
  checkTokenFileId,
  checkTokenFileWidth,
  diskTokenSource,
  idArrayFor,
  memTokenSource,
  stampTokenFile,
  tokenBytes,
  tokenIdPath,
  tokenizerFingerprint,
  writeTokenFile,
} from "../src/data/tokens.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { encodeCorpus } from "../src/commands/pretrain.ts";
import { GGUFWriter, readGGUF } from "../src/gguf/gguf.ts";
import { tokenizerFromGGUF } from "../src/export/load-gguf.ts";
import { CHATML_SPECIALS } from "../src/data/chat.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

function throws(fn: () => unknown, needle: string, msg: string): void {
  try {
    fn();
  } catch (e) {
    const m = String((e as Error).message);
    if (!m.includes(needle)) throw new Error(`${msg}: threw "${m}", expected "${needle}"`);
    return;
  }
  throw new Error(`${msg}: did not throw`);
}

// The threshold, including both sides of the boundary.
ok(tokenBytes(32768) === 2, "the vocab trained here stays u16");
ok(tokenBytes(49152) === 2, "SmolLM2's 49,152 still fits u16");
ok(tokenBytes(65536) === 2, "65,536 is the last vocab that fits u16");
ok(tokenBytes(65537) === 4, "one past the ceiling widens to u32");
ok(tokenBytes(128256) === 4, "Llama-3's 128,256 needs u32");
ok(tokenBytes(151936) === 4, "Qwen3's 151,936 needs u32");
ok(tokenBytes(262144) === 4, "Gemma's 262,144 needs u32");

// The failure the width guards against. Every wrapped value below is a legal id,
// which is why nothing downstream can notice the damage.
const ids = [151935, 151643, 128255, 100000, 70000, 65536, 42];
const u16 = new Uint16Array(ids.length);
u16.set(ids);
ok(u16[0] !== ids[0], "u16 wraps a Qwen3-range id instead of failing");
ok(u16[0] === (151935 & 0xffff), "and lands on 20863, a plausible small id");
ok(u16[5] === 0, "65,536 wraps to 0, the padding id");
ok(u16[6] === 42, "ids under the ceiling are untouched, so the corruption is partial");

const u32 = new Uint32Array(ids.length);
u32.set(ids);
ok(Array.from(u32).join(",") === ids.join(","), "u32 preserves every id exactly");

// idArrayFor is the fix: every encoder asks it for the width instead of assuming
// u16. This is the part that fails on a tree without the change.
ok(idArrayFor(32768) === Uint16Array, "a vocab that fits gets a u16 buffer");
ok(idArrayFor(49152) === Uint16Array, "SmolLM2's vocab gets a u16 buffer");
ok(idArrayFor(65537) === Uint32Array, "one past the ceiling gets a u32 buffer");
ok(idArrayFor(151936) === Uint32Array, "Qwen3's vocab gets a u32 buffer");

// The contract that matters at the call site: the largest id a vocab can produce
// must survive a round-trip through the buffer idArrayFor hands back.
for (const v of [32768, 49152, 128256, 151936, 262144]) {
  const buf = new (idArrayFor(v))(1);
  buf[0] = v - 1;
  ok(buf[0] === v - 1, `vocab ${v}: top id ${v - 1} survives its own buffer width`);
}

// And the same loop against a hard-coded u16 buffer, which is what the encoders
// used to do: the two large vocabs must be the ones that break.
const broken = [32768, 49152, 128256, 151936, 262144].filter((v) => {
  const buf = new Uint16Array(1);
  buf[0] = v - 1;
  return buf[0] !== v - 1;
});
ok(
  broken.join(",") === "128256,151936,262144",
  `a u16 buffer must corrupt exactly the large vocabs, got [${broken.join(",")}]`,
);

// The real encoder, on a vocab past the ceiling. This is the assertion that fails
// on a tree without the fix, and it fails with a WRONG VALUE rather than an import
// error: 69999 & 0xffff is 4463, which is itself a legal id.
{
  // One real byte token so the `for (const id of tok.encode(doc)) push(id)` path
  // is fed a genuine id too, not only the eos that `push(tok.eosId)` supplies.
  const tokens = Array.from({ length: 70000 }, (_, i) => `t${i}`);
  tokens[0] = "a";
  tokens[69999] = "<|endoftext|>";
  const tok = BPETokenizer.fromData({
    tokens,
    merges: [],
    bosId: 69999,
    eosId: 69999,
    specials: ["<|endoftext|>"],
  });
  ok(tok.vocabSize === 70000, `fixture vocab is past the u16 ceiling, got ${tok.vocabSize}`);
  const ids = Array.from(encodeCorpus(tok, "a<|endoftext|>b"));
  ok(ids.includes(69999), `eos 69999 survives the encode buffer, got [${ids.join(",")}]`);
  ok(!ids.includes(4463), "and is not truncated to 4463, the u16 wrap of that id");
  ok(ids.includes(0), `the ordinary-encode path also reached the buffer, got [${ids.join(",")}]`);
}

// `inspect --dump-tokenizer` is how a downloaded checkpoint's vocab reaches the
// corpus commands, and tokenizerFromGGUF is the whole of its logic. Round-trip a
// synthetic large-vocab GGUF rather than a real 1.1 GB download.
{
  const tokens = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    ...Array.from(
      { length: 99997 },
      (_, i) => `w${i}`,
    ),
  ];
  const w = new GGUFWriter();
  w.meta_string("general.architecture", "llama");
  w.meta_arr_str("tokenizer.ggml.tokens", tokens);
  w.meta_arr_str("tokenizer.ggml.merges", ["w0 w1"]);
  // llama.cpp marks turn tokens CONTROL (3); tokenizerFromGGUF recovers specials
  // from that, and falls back to the <|...|> shape only when no types are present.
  w.meta_arr_i32("tokenizer.ggml.token_type", tokens.map((_t, i) => (i < 3 ? 3 : 1)));
  w.meta_u32("tokenizer.ggml.bos_token_id", 0);
  w.meta_u32("tokenizer.ggml.eos_token_id", 2);
  const t = tokenizerFromGGUF(readGGUF(w.build()));

  ok(t.tokens.length === 100000, `vocab survives the round-trip, got ${t.tokens.length}`);
  ok(tokenBytes(t.tokens.length) === 4, "and it is a vocab that needs u32");
  ok(t.eosId === 2, `eos comes back, got ${t.eosId}`);
  for (const sp of CHATML_SPECIALS) {
    ok(t.specials?.includes(sp) === true, `${sp} is recovered as an atomic special`);
  }
  // The verdict inspect prints, and the condition chat-corpus enforces.
  const atomic = CHATML_SPECIALS.filter((x) => t.specials?.includes(x));
  ok(atomic.length === CHATML_SPECIALS.length, "so this vocab can drive chat-corpus");
}

// The corpus preflight. The losses and the embedding refuse an out-of-range id
// themselves, but they do it on whichever window happens to hold it, and
// pretrain's trust gate only reads the first 16 tokens: a .tokens file built
// against a different vocab passes that and the run can be tens of thousands of
// steps in before some later window trips a guard.
{
  const V = 100;
  const clean = memTokenSource(Array.from({ length: 4096 }, (_, i) => i % V));
  assertCorpusFitsVocab(clean, V, "clean");

  const at = (pos: number, id: number) => {
    const a = Array.from({ length: 4096 }, (_, i) => i % V);
    a[pos] = id;
    return memTokenSource(a);
  };
  // Positions chosen to straddle the scan's chunking: first, last, and one past
  // a chunk boundary if the chunk size ever drops below the corpus length.
  for (
    const [pos, id, why] of [
      [0, V, "the first token"],
      [4095, V, "the last token"],
      [2000, V * 10, "a middle token, far out of range"],
      [7, -1, "a negative id"],
      [9, 1.5, "a non-integer id"],
    ] as [number, number, string][]
  ) {
    throws(
      () => assertCorpusFitsVocab(at(pos, id), V, "corpus.tokens"),
      `token ${id} at position ${pos} is outside [0,${V})`,
      `${why} is refused, and named by position`,
    );
  }
  // V-1 is a legal id and must not be refused; V is the boundary.
  assertCorpusFitsVocab(at(50, V - 1), V, "clean");

  // The chunking, at a size a test can build. Without a chunk argument the
  // whole 4096-token corpus fits in one 1M-token read and a scan that stopped
  // after the first chunk would pass everything above.
  for (const pos of [0, 63, 64, 65, 4095]) {
    throws(
      () => assertCorpusFitsVocab(at(pos, V), V, "corpus.tokens", { chunk: 64 }),
      `at position ${pos} is outside`,
      `a bad id at ${pos} is found with a 64-token chunk`,
    );
  }
  assertCorpusFitsVocab(clean, V, "clean", { chunk: 64 });
  assertCorpusFitsVocab(clean, V, "clean", { chunk: 4096 });
  assertCorpusFitsVocab(clean, V, "clean", { chunk: 9999 });
  throws(
    () => assertCorpusFitsVocab(clean, V, "clean", { chunk: 0 }),
    "chunk must be a positive whole number",
    "a non-positive chunk is refused rather than looping forever",
  );
  // `from` is what keeps eval-loss from reading the part of the file it never
  // scores: a bad id before it must not be reported.
  assertCorpusFitsVocab(at(10, V), V, "clean", { from: 11 });
  throws(
    () => assertCorpusFitsVocab(at(3000, V), V, "corpus.tokens", { from: 2000 }),
    "at position 3000 is outside",
    "a bad id after `from` is still found, and still at its absolute position",
  );

  // memTokenSource had no bounds check, so a window past the end returned
  // `undefined` per token, which the losses now refuse by an odd route.
  throws(
    () => memTokenSource([1, 2, 3]).window(2, 3),
    "out of range",
    "a window past the end is refused rather than padded with undefined",
  );
  throws(() => memTokenSource([1, 2, 3]).window(-1, 2), "out of range", "a negative start too");
}

// The other half of the same class: a stale .tokens whose ids are all legal.
//
// assertCorpusFitsVocab above catches a stale file only when its ids exceed the
// new vocab. A same-size or larger vocab passes every range check, and so does a
// narrower one that flips the file's id width, since the size check is only
// `% bytesPerToken`. Nothing about an id says which vocab produced it, so the
// tokenizer is stamped beside the file instead.
{
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "tokenid-"));
  const tokensPath = path.join(dir, "corpus.tokens");
  fs.writeFileSync(tokensPath, new Uint8Array([1, 0, 2, 0, 3, 0]));

  const train = (text: string, vocab: number) => {
    const t = new BPETokenizer();
    t.train(text, vocab, []);
    return t;
  };
  // 280 is below what either corpus saturates at, so both land on exactly 280:
  // same size, different merges, which is the case a vocab-size check cannot see.
  const a = train("the quick brown fox jumps over the lazy dog ".repeat(40), 280);
  const b = train("lorem ipsum dolor sit amet consectetur adipiscing elit ".repeat(40), 280);

  const verdict = async (t: BPETokenizer) => {
    const v = await checkTokenFileId(tokensPath, t.export(), t.vocabSize, tokenBytes(t.vocabSize));
    return v.status === "mismatch" ? v.message : v.status;
  };

  // Stability, which matters more than discrimination: if a save/load round trip
  // moved the fingerprint, every correct corpus would start being refused after
  // the tokenizer was reloaded from its json, which is how pretrain always gets
  // it. `specials` is the only computed field in export(), and fromData filters
  // the declared list by what train() put in the vocab, which is all of it.
  const reloaded = BPETokenizer.fromData(JSON.parse(JSON.stringify(a.export())));
  ok(
    await tokenizerFingerprint(reloaded.export()) === await tokenizerFingerprint(a.export()),
    "the fingerprint survives a json round trip through fromData",
  );

  ok(await verdict(a) === "unstamped", "an unstamped file reports that, rather than passing");

  await stampTokenFile(tokensPath, a.export(), a.vocabSize, tokenBytes(a.vocabSize));
  ok(await verdict(a) === "ok", "the tokenizer that stamped it matches");

  ok(a.vocabSize === b.vocabSize, "the two tokenizers really are the same size");
  const merges = await verdict(b);
  ok(
    typeof merges === "string" && merges.includes("different merges or specials"),
    `same size, different merges is refused and named as such: got ${merges}`,
  );

  // The width flip, which is checked on its own: the same tokenizer with a
  // 4-byte stamp read as 2-byte still doubles the token count, and
  // diskTokenSource's `% 2` cannot see it. What the halves hold is at the
  // bottom of this block.
  await stampTokenFile(tokensPath, a.export(), a.vocabSize, 4);
  const flipped = await verdict(a);
  ok(
    typeof flipped === "string" && flipped.includes("4 bytes per token") &&
      flipped.includes("read as 2"),
    `a width flip is refused even when the tokenizer matches: got ${flipped}`,
  );

  // The byte size, for a file changed by something that is not one of this
  // repo's writers, which all drop the stamp first: an external truncation, an
  // interrupted copy, an `.id` moved beside a different file.
  await stampTokenFile(tokensPath, a.export(), a.vocabSize, tokenBytes(a.vocabSize));
  fs.writeFileSync(tokensPath, new Uint8Array([1, 0, 2, 0]));
  const resized = await verdict(a);
  ok(
    typeof resized === "string" && resized.includes("is 4 bytes and its stamp was written for 6"),
    `a file rewritten under its own stamp is refused: got ${resized}`,
  );
  fs.writeFileSync(tokensPath, new Uint8Array([1, 0, 2, 0, 3, 0]));

  // JSON.parse("null") succeeds, so without a shape guard the field reads throw
  // a TypeError out of the function instead of saying what is wrong.
  fs.writeFileSync(tokenIdPath(tokensPath), "null");
  const nullStamp = await verdict(a);
  ok(
    typeof nullStamp === "string" && nullStamp.includes("not readable JSON"),
    `a stamp of "null" gets the message, not a stack trace: got ${nullStamp}`,
  );

  fs.writeFileSync(tokenIdPath(tokensPath), "{not json");
  const broken2 = await verdict(a);
  ok(
    typeof broken2 === "string" && broken2.includes("not readable JSON"),
    `an unparseable stamp is an error, not a silent pass: got ${broken2}`,
  );

  // The fingerprint covers specials, which are neither vocab size nor merges.
  const sample = "the quick brown fox jumps over the lazy dog ".repeat(40);
  const plain = new BPETokenizer();
  plain.train(sample, 280, []);
  const special = new BPETokenizer();
  special.train(sample, 280, ["<|extra|>"]);
  ok(
    await tokenizerFingerprint(plain.export()) !== await tokenizerFingerprint(special.export()),
    "a reserved special changes the fingerprint, which neither merges nor a size check would show",
  );

  // The claim the width message makes, demonstrated rather than asserted. A
  // 4-byte file read as 2-byte passes diskTokenSource's `% bytesPerToken` and
  // doubles the token count, because each id splits into its low and high
  // halves. The odd slots are the high halves, which are 0 for every id under
  // 65,536 and not otherwise: "every second id reads as 0" is the common case,
  // not the rule, and a corpus that crossed the u16 ceiling is exactly the one
  // whose high halves are non-zero.
  const wide = path.join(dir, "wide.tokens");
  await writeTokenFile(wide, [42, 70000, 7], 4);
  const asWritten = await diskTokenSource(wide, 4);
  const asNarrow = await diskTokenSource(wide, 2);
  ok(asWritten.length === 3, "3 tokens at the width it was written");
  ok(asNarrow.length === 6, "6 at half the width, so the count doubles rather than failing");
  const halves = asNarrow.window(0, 6);
  ok(
    halves[0] === 42 && halves[1] === 0 && halves[4] === 7 && halves[5] === 0,
    `an id under 65,536 becomes itself followed by a 0: got ${halves.join(",")}`,
  );
  ok(
    halves[2] === 70000 % 65536 && halves[3] === 1,
    `and one above it becomes two non-zero halves: got ${halves.join(",")}`,
  );
  asWritten.close();
  asNarrow.close();

  // The other half of the byte-size rule: a rewrite of the same length would
  // leave the stamp agreeing on bytes and reporting a false ok, so the write
  // drops it. The invariant is unconditional, not size-dependent.
  const rewritten = path.join(dir, "rewritten.tokens");
  await writeTokenFile(rewritten, [1, 2, 3], 2);
  await stampTokenFile(rewritten, a.export(), a.vocabSize, 2);
  ok(fs.existsSync(tokenIdPath(rewritten)), "the stamp is there to begin with");
  await writeTokenFile(rewritten, [4, 5, 6], 2);
  const after = await checkTokenFileId(rewritten, a.export(), a.vocabSize, 2);
  ok(
    after.status === "unstamped",
    `a same-length rewrite leaves no stamp rather than a matching one: got ${after.status}`,
  );

  // Only ENOENT means absent. Treating every read failure as "no stamp" is the
  // bug pretrain's optstate probe already paid for once, where decoding a
  // multi-GB sidecar as UTF-8 overflowed and reported no optstate.
  const blocked = path.join(dir, "blocked.tokens");
  fs.writeFileSync(blocked, new Uint8Array([1, 0]));
  fs.mkdirSync(tokenIdPath(blocked));
  let threw = "";
  try {
    await checkTokenFileId(blocked, a.export(), a.vocabSize, 2);
  } catch (e) {
    threw = String((e as Error).message);
  }
  ok(
    threw !== "",
    "a stamp path that cannot be read propagates rather than reporting an unstamped file",
  );

  // The width-only path, which is all eval-loss can check. It must not go quiet
  // on a file its sibling refuses, which is the shape this whole entry exists to
  // close.
  const wOnly = path.join(dir, "wonly.tokens");
  fs.writeFileSync(wOnly, new Uint8Array([1, 0, 2, 0]));
  ok(
    (await checkTokenFileWidth(wOnly, 2)).status === "unstamped",
    "no stamp means the width cannot be checked either",
  );
  await stampTokenFile(wOnly, a.export(), a.vocabSize, 2);
  ok((await checkTokenFileWidth(wOnly, 2)).status === "ok", "a matching width passes");
  const wWrong = await checkTokenFileWidth(wOnly, 4);
  ok(
    wWrong.status === "mismatch" && wWrong.message.includes("read as 4"),
    "a mismatched width is refused without a tokenizer in hand",
  );
  fs.writeFileSync(wOnly, new Uint8Array([1, 0]));
  const wResized = await checkTokenFileWidth(wOnly, 2);
  ok(
    wResized.status === "mismatch" && wResized.message.includes("is 2 bytes"),
    "and so is a file rewritten under its own stamp, which needs no tokenizer either",
  );
  // Back to a file its stamp describes, so the byte size is not what is being
  // measured below.
  await stampTokenFile(wOnly, a.export(), a.vocabSize, 2);

  // The asymmetry the tokenizer stamp exists for: a different tokenizer
  // of the same width is refused by the full check and passed by the width one.
  // Without this, "improving" the width path to hash the tokenizer would refuse
  // correct corpora in eval-loss with the suite green.
  const sameWidth = await checkTokenFileId(wOnly, b.export(), b.vocabSize, 2);
  ok(
    sameWidth.status === "mismatch" && (await checkTokenFileWidth(wOnly, 2)).status === "ok",
    "a different tokenizer of the same width: refused by the full check, passed by the width one",
  );

  fs.writeFileSync(tokenIdPath(wOnly), "null");
  const wNull = await checkTokenFileWidth(wOnly, 2);
  ok(
    wNull.status === "mismatch" && wNull.message.includes("not readable JSON"),
    "a malformed stamp is refused here too, rather than reported as absent",
  );

  fs.rmSync(dir, { recursive: true, force: true });
}

console.log("large-vocab: all checks passed");
