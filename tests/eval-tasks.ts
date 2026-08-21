// Standalone assert check for the eval-choice task table: the PIQA row parser,
// the label join that PIQA's split label file needs, and the dataset coordinates
// each task is registered under (nothing downloaded).
// Run:  deno run tests/eval-tasks.ts
import {
  argminPerChar,
  attachPiqaLabels,
  hellaswagItem,
  hellaswagPreprocess,
  piqaItem,
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

console.log("eval-tasks: all checks passed");
