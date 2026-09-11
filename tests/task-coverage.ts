// Every test file on disk is run by `deno task test`, and CI runs that task.
//
// tests/generate-penalty.ts is what this is for: nothing ever blocked it from
// running, it was simply never added to the task, and it sat in tests/ unrun
// for weeks looking exactly like a file that ran. The task string is
// hand-maintained, so a check is what keeps it honest rather than a doc rule.
// The CI half is the same shape one level up: a correct task that no runner
// invokes guarantees nothing (#89, #97).
//
// One capability went with the Node and Bun tasks and is worth naming, since
// this is the file that would otherwise carry it: there is no per-invocation
// flag check any more. `--no-install` was load-bearing on the Bun task and
// asserted here. No flag on `deno run` carries that weight today, so the check
// was dropped rather than kept empty.
//
// Run:  deno run -A tests/task-coverage.ts
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const cfg = JSON.parse(fs.readFileSync(path.join(root, "deno.json"), "utf8")) as {
  tasks?: Record<string, unknown>;
};
const body = cfg.tasks?.test;
ok(typeof body === "string", "deno.json has no `test` task, or it is not a string");
const task = body as string;

/** Every .ts under tests/, at any depth, minus the fixture data. */
function testFiles(dir: string, prefix = ""): string[] {
  const out: string[] = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const rel = prefix ? `${prefix}/${e.name}` : e.name;
    if (e.isDirectory()) {
      if (e.name === "fixtures") continue; // data the tests read, not tests
      out.push(...testFiles(path.join(dir, e.name), rel));
    } else if (e.name.endsWith(".ts")) out.push(rel);
  }
  return out.sort();
}
const onDisk = testFiles(path.join(root, "tests"));

// Any flags, not just today's: `deno run --allow-read` would otherwise make a
// listed file look missing, and the error would point at the wrong thing.
const invoked = new Set(
  Array.from(
    task.matchAll(/deno run(?:\s+--?[\w=./-]+)*\s+tests\/([A-Za-z0-9._/-]+\.ts)/g),
    (m) => m[1],
  ),
);

for (const f of onDisk) {
  ok(
    invoked.has(f),
    `tests/${f} is not run by \`deno task test\`. Add it; a shared helper rather than a runnable ` +
      `test belongs in src/ or under tests/fixtures/, not as a bare tests/*.ts`,
  );
}
for (const f of invoked) {
  ok(onDisk.includes(f), `\`deno task test\` runs tests/${f}, which is not on disk`);
}
// Per PATH, not per file: `&& node tests/eta-fmt.ts` appended to the task would
// otherwise hide behind the correct invocation on the line before it.
// Occurrences against occurrences, not against `invoked`, which is a Set: a
// file invoked twice on purpose would otherwise fail here, blaming a cause that
// is not there.
const mentioned = task.match(/tests\/[A-Za-z0-9._/-]+\.ts/g) ?? [];
const claimed = task.match(/deno run(?:\s+--?[\w=./-]+)*\s+tests\/[A-Za-z0-9._/-]+\.ts/g) ?? [];
ok(
  mentioned.length === claimed.length,
  `deno.json's test task mentions ${mentioned.length} tests/ paths but ${claimed.length} are ` +
    `invoked by \`deno run\`; one is handed to another runtime, written ./tests/, or passed as ` +
    `an argument rather than as the entry point`,
);

const workflow = fs.readFileSync(path.join(root, ".github/workflows/test.yml"), "utf8");
ok(
  /run:\s+deno task test\s*$/m.test(workflow),
  ".github/workflows/test.yml has no `run: deno task test` step, so the suite is checked only " +
    "by whoever runs it by hand. That is what #89 and #97 were",
);

console.log(`task-coverage: ${onDisk.length} test files, all run by \`deno task test\` ✓`);
