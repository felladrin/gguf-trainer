// Standalone assert check for the eval-choice task table: the PIQA row parser,
// the label join that PIQA's split label file needs, and the dataset coordinates
// each task is registered under (nothing downloaded).
// Run:  deno run tests/eval-tasks.ts
import { attachPiqaLabels, piqaItem, TASKS } from "../src/commands/eval-choice.ts";

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

console.log("eval-tasks: all checks passed");
