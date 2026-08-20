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
import { tokenBytes } from "../src/data/tokens.ts";

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

// The width a caller must pick for each vocab, which is the whole contract.
for (const v of [32768, 49152, 151936]) {
  const A = tokenBytes(v) === 2 ? Uint16Array : Uint32Array;
  const buf = new A(1);
  buf[0] = v - 1; // the largest id that vocab can produce
  ok(buf[0] === v - 1, `vocab ${v}: the top id survives its own width`);
}

console.log("large-vocab: all checks passed");
