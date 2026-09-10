// Every test file is in `deno task test`, and in `deno task test:node` unless
// something in it genuinely cannot load there.
//
// The static hyparquet import that produced #83 is only half of why 79
// assertions over eval-choice's scoring arithmetic went uncovered under Node.
// The import made `tests/eval-tasks.ts` fail there; what made that invisible is
// that both task lists are hand-maintained strings in deno.json with nothing
// comparing them to what is in tests/. `tests/generate-penalty.ts` proves the
// second half on its own: nothing was ever blocking it, and it was missing
// anyway. A doc rule is what already failed here, so this is a check instead.
//
// Run:  deno run -A tests/task-coverage.ts
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Files deliberately outside `test:node`, each with what actually stops them.
 *
 * The bar is a load-time failure you have seen, not a guess. The first version
 * of this list had five entries and three were wrong: they described a Deno API
 * in the script under test rather than anything on the test's load path, and all
 * three ran under Node the moment anyone tried. That is the same invisible gap
 * this file exists to close, wearing a reason that reads authoritative.
 */
const NODE_EXEMPT: Record<string, string> = {
  "npm-deps.ts":
    "statically imports @huggingface/jinja, a bare specifier that resolves through Deno's " +
    "import map; with no node_modules it is ERR_MODULE_NOT_FOUND at load. Exercising the npm " +
    "dependencies is the point of the file, so this one cannot be moved inside a function.",
};

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const testsDir = path.join(root, "tests");

const cfg = JSON.parse(fs.readFileSync(path.join(root, "deno.json"), "utf8")) as {
  tasks?: Record<string, unknown>;
};
// Renaming the task is covered below; renaming or dropping the block around it
// would otherwise be a bare TypeError, and Deno reads deno.json as JSONC, so a
// comment someone adds is a bare SyntaxError from the parse above.
ok(cfg.tasks !== undefined, "deno.json has no `tasks` block; this check reads both task strings");
/**
 * Filenames a task actually hands to a given runner.
 *
 * Matching the runner and not just the path matters: pasting `deno run
 * tests/foo.ts` into the test:node string would otherwise report Node coverage
 * that does not exist, which is the false signal this file was written to stop.
 */
function invoked(task: string, runner: RegExp): Set<string> {
  const body = cfg.tasks![task];
  ok(
    typeof body === "string",
    `deno.json's "${task}" task is ${
      body === undefined ? "missing" : "not a string"
    }; this check reads both task strings and cannot see an object form`,
  );
  const re = new RegExp(`${runner.source}\\s+tests/([A-Za-z0-9._/-]+\\.ts)`, "g");
  return new Set(Array.from((body as string).matchAll(re), (m) => m[1]));
}

// Any flags, not just today's: `deno run --allow-read` or `node
// --experimental-strip-types --no-warnings` would otherwise make a listed file
// look missing, and the error would point at the wrong thing.
const FLAGS = "(?:\\s+--?[\\w=./-]+)*";
const inTest = invoked("test", new RegExp(`deno run${FLAGS}`));
const inNode = invoked("test:node", new RegExp(`node${FLAGS}`));

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
const onDisk = testFiles(testsDir);

for (const f of onDisk) {
  ok(
    inTest.has(f),
    `tests/${f} is not run by \`deno task test\`. Add it; a shared helper rather than a runnable ` +
      `test belongs in src/ or under tests/fixtures/, not as a bare tests/*.ts`,
  );
  const reason = NODE_EXEMPT[f];
  ok(
    inNode.has(f) || reason !== undefined,
    `tests/${f} is not run by \`deno task test:node\` and has no NODE_EXEMPT entry. Run ` +
      `\`node --experimental-strip-types tests/${f}\`: if it passes, add it to the task; if it ` +
      `cannot load, add it here with what fails`,
  );
  ok(
    !(inNode.has(f) && reason !== undefined),
    `tests/${f} is run by \`deno task test:node\` and also listed as exempt from it; drop the ` +
      `exemption`,
  );
}

for (const [task, listed] of [["test", inTest], ["test:node", inNode]] as const) {
  for (const f of listed) {
    ok(onDisk.includes(f), `\`deno task ${task}\` runs tests/${f}, which is not on disk`);
  }
}
for (const [f, reason] of Object.entries(NODE_EXEMPT)) {
  ok(onDisk.includes(f), `NODE_EXEMPT names tests/${f}, which is not on disk`);
  // An empty string satisfies a presence check, and an exemption without a
  // reason is what the three wrong ones would have collapsed to.
  ok(reason.trim().length > 20, `NODE_EXEMPT["${f}"] needs a reason, not a placeholder`);
}

console.log(
  `task-coverage: ${onDisk.length} test files, ${inTest.size} run under deno, ${inNode.size} ` +
    `under node, ${Object.keys(NODE_EXEMPT).length} exempt with a stated load failure ✓`,
);
