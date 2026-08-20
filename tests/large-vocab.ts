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
import { idArrayFor, tokenBytes } from "../src/data/tokens.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { encodeCorpus } from "../src/commands/pretrain.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
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
  const tokens = Array.from({ length: 70000 }, (_, i) => `t${i}`);
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
}

console.log("large-vocab: all checks passed");
