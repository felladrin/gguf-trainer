// Standalone assert check for the eval-choice task table: the PIQA row parser,
// the label join that PIQA's split label file needs, and the dataset coordinates
// each task is registered under (nothing downloaded).
// Run:  deno run tests/eval-tasks.ts
import {
  argminPerChar,
  attachPiqaLabels,
  choiceMaskStart,
  choiceWindowError,
  hellaswagItem,
  hellaswagPreprocess,
  piqaItem,
  preflightByBytes,
  renderPair,
  TASKS,
} from "../src/commands/eval-choice.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

function throws(fn: () => unknown, msg: string): void {
  try {
    fn();
  } catch {
    return;
  }
  throw new Error(msg);
}

const row = (over: Record<string, unknown> = {}) => ({
  goal: "How do I dry a wet phone?",
  sol1: "Put it in a bowl of rice overnight.",
  sol2: "Put it in the oven on a high heat.",
  label: 0,
  ...over,
});

// piqaItem: two options, in sol1/sol2 order, gold straight from the 0/1 label.
const a = piqaItem(row())!;
ok(a !== null, "a well-formed PIQA row parses");
ok(a.context === "How do I dry a wet phone?", "goal becomes the context");
ok(a.choices.length === 2, "two candidate solutions");
ok(a.choices[0].startsWith("Put it in a bowl"), "sol1 is choice 0");
ok(a.choices[1].startsWith("Put it in the oven"), "sol2 is choice 1");
ok(a.gold === 0, "label 0 selects sol1");
ok(piqaItem(row({ label: 1 }))!.gold === 1, "label 1 selects sol2");
ok(piqaItem(row({ label: "1" }))!.gold === 1, "a stringified label still parses");

// The label is what decides right from wrong, so anything unusable must drop the
// row rather than default to 0, which would silently score it against sol1.
ok(piqaItem(row({ label: undefined })) === null, "a missing label drops the row");
ok(piqaItem(row({ label: 2 })) === null, "an out-of-range label drops the row");
ok(piqaItem(row({ label: "x" })) === null, "a non-numeric label drops the row");
ok(piqaItem(row({ sol2: undefined })) === null, "a missing solution drops the row");
ok(piqaItem(row({ goal: 42 })) === null, "a non-string goal drops the row");

// attachPiqaLabels: PIQA ships gold labels in a file parallel to the questions,
// joined by position. Getting the join wrong still produces a scoreable set.
const rows = [row({ label: undefined }), row({ label: undefined }), row({ label: undefined })];
const joined = attachPiqaLabels(rows, "0\n1\n1\n");
ok(joined.map((r) => r.label).join(",") === "0,1,1", "labels attach in file order");
ok(rows.every((r) => r.label === undefined), "the input rows are not mutated");
ok(
  attachPiqaLabels(rows, "0\n1\n1").length === 3,
  "a file without a trailing newline is not one label short",
);
ok(
  attachPiqaLabels(rows, "0\n\n1\n \n1\n").map((r) => r.label).join(",") === "0,1,1",
  "blank lines are skipped",
);
throws(
  () => attachPiqaLabels(rows, "0\n1\n"),
  "too few labels must abort, not score every question against a shifted answer",
);
throws(() => attachPiqaLabels(rows, "0\n1\n1\n0\n"), "too many labels must abort");

// Task registration: the CLI advertises these names, and ARC-Easy differs from
// ARC-Challenge only by its config, which is exactly the kind of copy-paste that
// silently scores the wrong split.
for (const name of ["arc", "arc-easy", "hellaswag", "piqa"]) {
  ok(TASKS[name] !== undefined, `task ${name} is registered`);
}
ok(TASKS["arc"].config === "ARC-Challenge", "arc is the Challenge split");
ok(TASKS["arc-easy"].config === "ARC-Easy", "arc-easy is the Easy split");
ok(TASKS["arc"].id === TASKS["arc-easy"].id, "both ARC tasks read the same dataset");
ok(
  TASKS["arc"].chance === 25 && TASKS["piqa"].chance === 50,
  "chance floors match the option count",
);
ok(TASKS["piqa"].load !== undefined, "piqa bypasses the parquet path it has no parquet for");
ok(TASKS["arc"].load === undefined, "arc still uses the parquet path");

// The render is the prompt the model actually sees, and it has to match
// lm-eval-harness or the numbers are not comparable to the leaderboard's.
ok(
  TASKS["piqa"].render("G", "S") === "Question: G\nAnswer: S",
  `piqa renders like ARC, got ${JSON.stringify(TASKS["piqa"].render("G", "S"))}`,
);
ok(TASKS["hellaswag"].render("C", "E") === "C E", "hellaswag renders as a plain continuation");

// HellaSwag: the query is not the bare `ctx` field. lm-eval-harness scores
// preprocess(activity_label + ": " + ctx_a + " " + ctx_b.capitalize()), and every
// published HellaSwag number is measured on that string, so a shortcut here makes
// the result incomparable while still looking like a plausible score.
const hs = (over: Record<string, unknown> = {}) => ({
  activity_label: "Roof shingle removal",
  ctx_a: "A man is sitting on a roof.",
  ctx_b: "he",
  ctx: "A man is sitting on a roof. he",
  endings: ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "c", "d"],
  label: 1,
  ...over,
});

const h = hellaswagItem(hs())!;
ok(h !== null, "a well-formed HellaSwag row parses");
ok(
  h.context === "Roof shingle removal: A man is sitting on a roof. He",
  `activity label prefixes the query and ctx_b is capitalized, got ${JSON.stringify(h.context)}`,
);
ok(!h.context.includes(" he"), "the lowercase ctx_b must not survive uncapitalized");
ok(h.gold === 1, "label selects the gold ending");
ok(hellaswagItem(hs({ label: "1" }))!.gold === 1, "a stringified label still parses");
ok(hellaswagItem(hs({ activity_label: undefined })) === null, "no activity label drops the row");
ok(hellaswagItem(hs({ label: 9 })) === null, "an out-of-range label drops the row");

// Python's str.capitalize() lowercases the tail; copying only the uppercase half
// of it silently changes the query on every row whose ctx_b has inner capitals.
ok(
  hellaswagItem(hs({ ctx_b: "the MAN then" }))!.context.endsWith("The man then"),
  "ctx_b's tail is lowercased, the way str.capitalize() does it",
);

// The same cleanup runs over the endings, not just the query.
ok(
  hellaswagItem(hs({ endings: ["a [header] b", "x", "y", "z"] }))!.choices[0] === "a b",
  "endings are preprocessed too",
);

ok(hellaswagPreprocess("  padded  ") === "padded", "outer whitespace goes");
ok(
  hellaswagPreprocess("Do it [title] Then rest") === "Do it. Then rest",
  "[title] becomes a break",
);
ok(
  hellaswagPreprocess("keep [substeps] this") === "keep this",
  "other bracketed spans are dropped",
);
ok(hellaswagPreprocess("a  b") === "a b", "the double space a drop leaves is collapsed");

// acc_norm normalizes by the choice's character length, not by its token count.
// The two disagree exactly when a longer choice is cheaper per token, which is
// the case the metric exists to handle.
ok(argminPerChar([10, 12], ["ab", "abcd"]) === 1, "the longer choice wins on cost per character");
ok(argminPerChar([10, 12], ["ab", "ab"]) === 0, "equal lengths fall back to the raw sum");
ok(argminPerChar([5], ["only"]) === 0, "a single choice is the prediction");
ok(argminPerChar([1, 1], ["", "abcd"]) === 1, "an empty choice does not divide by zero");

// The choice mask. The scored span is the last nChoice targets of the window the
// model actually sees, so it has to be measured after truncation. Measuring it
// from the untruncated context length instead scores only the tail of the choice
// once context plus choice passes maxSeq, and scores none of it once the context
// alone fills maxSeq, which returns a summed NLL of 0 that wins both metrics.
ok(choiceMaskStart(20, 10, 512) === 19, "an untruncated window masks the whole context");
ok(
  choiceMaskStart(100, 10, 105) === 94,
  `a partly truncated window keeps all 10 choice tokens, got ${choiceMaskStart(100, 10, 105)}`,
);
ok(
  choiceMaskStart(600, 10, 512) === 501,
  `a fully truncated context keeps all 10 choice tokens, got ${choiceMaskStart(600, 10, 512)}`,
);

// The two invariants choiceNLL relies on, over every window shape it accepts:
// the boundary indexes a real target, and the count it multiplies the mean back
// by is the whole choice.
for (const maxSeq of [8, 105, 512]) {
  for (let nCtx = 1; nCtx <= maxSeq + 40; nCtx += 7) {
    for (let nChoice = 1; nChoice < maxSeq; nChoice += 3) {
      if (choiceWindowError(nCtx, nChoice, maxSeq) !== null) continue;
      const nTargets = Math.min(nCtx + nChoice, maxSeq) - 1; // targets = full.slice(1)
      const start = choiceMaskStart(nCtx, nChoice, maxSeq);
      const shape = `ctx ${nCtx}, choice ${nChoice}, maxSeq ${maxSeq}`;
      ok(
        start >= 0 && start < nTargets,
        `the mask boundary stays inside the targets (${shape}): ${start} of ${nTargets}`,
      );
      ok(
        nTargets - start === nChoice,
        `every choice token is scored (${shape}): ${nTargets - start} of ${nChoice}`,
      );
    }
  }
}

// Refusing beats scoring a shortened choice: the summed NLL falls while acc_norm
// keeps dividing by the full character count, so the truncated option wins.
ok(choiceWindowError(20, 10, 512) === null, "a window that fits is scoreable");
ok(choiceWindowError(1, 511, 512) === null, "a choice with one token of context to spare fits");
ok(
  choiceWindowError(1, 512, 512) !== null,
  "a choice that fills the context leaves nothing to predict it from",
);
ok(choiceWindowError(600, 600, 512) !== null, "a choice longer than the context is refused");
// Each refusal has to name its own cause. An empty stem reaching the
// context-length branch reads as "this model's context is 512" when the context
// is fine and the stem is the problem.
ok(
  choiceWindowError(20, 0, 512)?.includes("a choice rendered to 0 tokens") === true,
  `an empty choice is refused as an empty choice, got ${choiceWindowError(20, 0, 512)}`,
);
ok(
  choiceWindowError(0, 5, 512)?.includes("the stem rendered to 0 tokens") === true,
  `an empty stem is refused as an empty stem, got ${choiceWindowError(0, 5, 512)}`,
);
ok(
  choiceWindowError(1, 512, 512)?.includes("context is 512") === true,
  `a choice that fills the context names the context, got ${choiceWindowError(1, 512, 512)}`,
);

// The composition the two formulas describe: slice the window the way choiceNLL
// does, run its mask loop, and look at what is left scoreable. The slicing is
// re-typed here rather than called, so this pins the boundary against the shape
// it is meant for, not choiceNLL's own three slice lines. Those are pinned at
// runtime instead, by the invariant throw before sequenceLoss.
for (const [nCtx, nChoice, maxSeq] of [[20, 10, 512], [100, 10, 105], [600, 10, 512], [1, 7, 8]]) {
  const shape = `ctx ${nCtx}, choice ${nChoice}, maxSeq ${maxSeq}`;
  ok(choiceWindowError(nCtx, nChoice, maxSeq) === null, `${shape} is scoreable`);
  const ids = Array.from({ length: nCtx + nChoice }, (_, i) => i + 1);
  const full = ids.slice(-maxSeq);
  const inputs = full.slice(0, -1);
  const targets = full.slice(1);
  const start = choiceMaskStart(nCtx, nChoice, maxSeq);
  for (let i = 0; i < start; i++) targets[i] = -1;
  ok(
    targets.length === inputs.length,
    `the mask does not extend the targets (${shape}): ${targets.length} vs ${inputs.length}`,
  );
  ok(
    targets.filter((t) => t >= 0).length === nChoice,
    `the kept count is the whole choice (${shape})`,
  );
  ok(
    targets.slice(start).join(",") === ids.slice(-nChoice).join(","),
    `the scored targets are the choice tokens themselves (${shape})`,
  );
}

// The preflight. choiceNLL refuses an unscoreable window on the item that holds
// one, which on a full set is hours in. Encoding every pair up front to find
// them costs 6.7 s on HellaSwag, measured, so a length bound does the work
// instead.
//
// BYTES, not characters. This repo's BPE is byte-level, so every token covers at
// least one UTF-8 byte and the byte count bounds the token count by
// construction. `String.length` does not: measured with the repo's own
// tokenizer, "\u2e3b" is one UTF-16 unit and two tokens.
{
  const pair = (ctxOnly: string, choiceText: string) => ({ ctxOnly, choiceText });
  const maxSeq = 16;

  const none = preflightByBytes([pair("Question: x\nAnswer:", " yes"), pair("q", "a")], maxSeq);
  ok(none.needExactCheck.length === 0, "short choices need no tokenizer at all");
  ok(none.empty === null, "and a non-empty pair is settled by being non-empty");

  // At exactly maxSeq bytes the bound stops proving anything, so that pair has
  // to be encoded. One byte less and it cannot reach maxSeq tokens.
  ok(
    preflightByBytes([pair("q", "x".repeat(maxSeq))], maxSeq).needExactCheck.length === 1,
    "a choice of maxSeq bytes is not settled",
  );
  ok(
    preflightByBytes([pair("q", "x".repeat(maxSeq - 1))], maxSeq).needExactCheck.length === 0,
    "one byte under, it is",
  );

  // The case characters get wrong. Six of these are 6 UTF-16 units, well under
  // maxSeq, but 18 UTF-8 bytes, so they could encode to more tokens than the
  // context holds and the pair has to be handed on.
  ok(
    preflightByBytes([pair("q", "\u2e3b".repeat(6))], maxSeq).needExactCheck.length === 1,
    "characters do not bound the token count; bytes do",
  );
  ok(
    "\u2e3b".repeat(6).length < maxSeq,
    "and that case really is under maxSeq by the character count",
  );

  ok(preflightByBytes([pair("", " yes")], maxSeq).empty === "stem", "an empty stem is caught");
  ok(preflightByBytes([pair("q", "")], maxSeq).empty === "choice", "an empty choice too");

  // Only the long ones are handed on, not the whole batch.
  const mixed = preflightByBytes(
    [pair("q", "short"), pair("q", "y".repeat(99)), pair("q", "also short")],
    maxSeq,
  );
  ok(mixed.needExactCheck.length === 1, "only the pair that could reach the ceiling is returned");
  ok(mixed.needExactCheck[0].choiceText.length === 99, "and it is the right one");

  // renderPair is what makes the preflight vouch for the strings actually
  // scored: both sites call it, so they cannot drift.
  const rp = renderPair(TASKS["piqa"].render, "G", "S");
  ok(rp.ctxOnly === "Question: G\nAnswer:", "the stem is the render with an empty choice, trimmed");
  ok(rp.choiceText === " S", "and the choice is what the full render adds after it");
  ok(
    TASKS["piqa"].render("G", "S") === rp.ctxOnly + rp.choiceText,
    "the two halves reassemble into exactly what the model sees",
  );
}

console.log("eval-tasks: all checks passed");
