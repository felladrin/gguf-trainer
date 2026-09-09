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
import {
  assertCorpusFitsVocab,
  idArrayFor,
  memTokenSource,
  tokenBytes,
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

console.log("large-vocab: all checks passed");
