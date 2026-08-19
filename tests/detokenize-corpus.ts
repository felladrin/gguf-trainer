// Standalone assert check for the PTB detokenizer (importing the module must
// not run main).
// Run:  deno run tests/detokenize-corpus.ts
import { detokenize, detokenizeLine } from "../scripts/detokenize-corpus.ts";

function eq(got: string, want: string, msg: string): void {
  if (got !== want) {
    throw new Error(`${msg}\n  got  ${JSON.stringify(got)}\n  want ${JSON.stringify(want)}`);
  }
}

// The real shape from euclaise/writingprompts.
eq(
  detokenizeLine(" `` Is it just me, or am I GREAT at entrances?!'' "),
  '"Is it just me, or am I GREAT at entrances?!"',
  "PTB quotes become real ones",
);
eq(detokenizeLine("you could n't blame it"), "you couldn't blame it", "split contraction rejoins");
eq(detokenizeLine("I ai n't gon na do it"), "I ain't gonna do it", "n't and gon na together");
eq(detokenizeLine("he wan na and got ta"), "he wanna and gotta", "the other na-splits");
eq(detokenizeLine("music ( still not clear )"), "music (still not clear)", "bracket padding");
eq(detokenizeLine("a [ note ] here"), "a [note] here", "square brackets too");
eq(detokenizeLine("trailing space   "), "trailing space", "trailing whitespace goes");
eq(detokenizeLine("  leading space"), "leading space", "the PTB join's leading space goes");

// Idempotence: running the fixer twice must not keep eating characters.
const once = detokenizeLine(" `` Hello, '' he said. ");
eq(detokenizeLine(once), once, "rules are idempotent");

// Text that was never damaged must survive untouched.
for (const clean of ['"Already fine," he said.', "isn't and wasn't", "a (parenthetical) aside"]) {
  eq(detokenizeLine(clean), clean, `undamaged text is left alone: ${clean}`);
}

// Whole-text form: a space-only line becomes blank, and the run collapses to one.
eq(
  detokenize("para one. \n \n para two. \n \n \n para three."),
  "para one.\n\npara two.\n\npara three.",
  "paragraph breaks normalize",
);

// The document separator must survive, or tokenize loses every boundary.
eq(
  detokenize("story one. \n<|endoftext|>\nstory two."),
  "story one.\n<|endoftext|>\nstory two.",
  "document markers are untouched",
);

console.log("detokenize-corpus: all checks passed");
