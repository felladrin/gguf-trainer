// Standalone assert check for the style-SFT data path: the seed filters and the
// restyle driver's parsing/validation (no dataset downloaded, no pi spawned;
// importing either module must not run main).
// Run:  deno run tests/style-pipeline.ts
import { classify, shuffle } from "../src/commands/style-seed.ts";
import {
  applyReplies,
  parseReplies,
  rejectReason,
  type SeedRecord,
  stripDashes,
} from "../src/commands/style-restyle.ts";
import type { ChatMessage } from "../src/data/chat.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}
const msg = (role: string, content: string): ChatMessage => ({ role, content });
const chat = (...pairs: string[]): ChatMessage[] =>
  pairs.map((c, i) => msg(i % 2 === 0 ? "user" : "assistant", c));

const A = "x".repeat(200); // a plausible assistant answer
const U = "how does this work?";

// --- seed filters ----------------------------------------------------------
ok(classify(chat(U, A)) === "ok", "a plain user/assistant pair is usable");
ok(classify([msg("system", "be brief"), ...chat(U, A)]) === "ok", "a leading system turn is fine");
ok(classify(null) === "turns", "a row with no conversation is rejected");
ok(classify(chat(U)) === "turns", "an unanswered question is rejected");
ok(classify(Array.from({ length: 14 }, (_, i) => chat(U, A)[i % 2])) === "turns", "13+ turns out");
ok(classify([msg("assistant", A), msg("user", U)]) === "roles", "assistant-first is rejected");
ok(classify(chat(U, A, U, A).slice(0, 3)) === "roles", "a trailing user turn is rejected");
ok(classify(chat(U, "too short")) === "length", "a stub answer is rejected");
ok(classify(chat(U, "y".repeat(950))) === "length", "a 950-char answer is rejected");
ok(classify(chat("u".repeat(750), A)) === "length", "a 750-char question is rejected");
const long = "z".repeat(880); // under the per-answer cap, over the total once repeated
ok(classify(chat(U, long, U, long, U, long)) === "length", "2500 chars total is the cap");
ok(classify(chat(U, "```ts\ncode\n```" + A)) === "markup", "code fences are rejected");
ok(classify(chat(U, "\\(x^2\\)" + A)) === "markup", "LaTeX is rejected");
ok(classify(chat(U, "日本語のテキストです" + A.slice(0, 60))) === "nonascii", "non-English out");

// Same seed, same order; a different seed moves at least one element.
const ids = () => Array.from({ length: 40 }, (_, i) => i);
const rng = (s: number) => {
  let a = s;
  return () => ((a = (a * 1664525 + 1013904223) >>> 0) / 4294967296);
};
ok(
  shuffle(ids(), rng(7)).join() === shuffle(ids(), rng(7)).join(),
  "shuffle is deterministic per seed",
);
ok(shuffle(ids(), rng(7)).join() !== shuffle(ids(), rng(8)).join(), "a new seed reshuffles");
ok(shuffle(ids(), rng(7)).slice().sort((a, b) => a - b).join() === ids().join(), "nothing is lost");

// --- rewriter output parsing ----------------------------------------------
ok(
  parseReplies("### REPLY 1\nfirst\n\nline two\n### REPLY 2\nsecond")?.join("|") ===
    "first\n\nline two|second",
  "marker blocks keep their paragraph breaks",
);
ok(
  parseReplies("## Reply 1:\n```\nfenced\n```\n## Reply 2\nplain")?.join("|") === "fenced|plain",
  "marker matching is loose and a stray fence is dropped",
);
ok(parseReplies('### REPLY 1\n["not json"]')?.join() === '["not json"]', "markers win over JSON");
ok(parseReplies('```json\n["a", "b"]\n```')?.join() === "a,b", "a fenced array still parses");
ok(
  parseReplies('Here you go: ["a"] hope it helps')?.join() === "a",
  "prose around the array is ok",
);
ok(parseReplies("not json at all") === null, "a non-answer is rejected");
ok(parseReplies('{"replies": ["a"]}')?.join() === "a", "an array wrapped in an object is salvaged");
ok(parseReplies('["a", 2]') === null, "a non-string element is rejected");

ok(rejectReason([A], ["y".repeat(180)]) === "", "a same-length rewrite passes");
ok(rejectReason([A, A], [A]).includes("wanted 2"), "a dropped reply is caught");
ok(rejectReason([A], ["y".repeat(40)]) !== "", "an over-compressed rewrite is caught");
ok(rejectReason([A], ["y".repeat(600)]) !== "", "a padded rewrite is caught");
ok(rejectReason([A], ["short"]) !== "", "a stub rewrite is caught");
ok(
  rejectReason([A], ["y".repeat(150) + " as Ada would say"], ["Ada"]).includes("names the author"),
  "a rewrite that names the author is caught",
);
ok(
  rejectReason([A], ["y".repeat(150) + " as Ada would say"]) === "",
  "with no --author-name, nothing is treated as a name leak",
);

ok(stripDashes("a placebo—so the baseline") === "a placebo, so the baseline", "em dash -> comma");
ok(stripDashes("pay $500–$2,000") === "pay $500-$2,000", "a numeric range keeps a hyphen");
ok(!stripDashes("one — two – three").includes("—"), "no em dash survives");

const rec: SeedRecord = { id: "t:1", config: "t", messages: chat(U, A, U, A) };
const applied = applyReplies(rec, ["one—two", "three"]);
ok(applied.messages[0].content === U, "user turns pass through untouched");
ok(applied.messages[1].content === "one, two", "assistant turns are replaced and normalized");
ok(applied.messages[3].content === "three", "replies are applied in order");
ok(rec.messages[1].content === A, "the seed record is not mutated");

console.log("style_pipeline: all assertions passed");
