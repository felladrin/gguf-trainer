// Every test file is in `deno task test`, and in `deno task test:node` and
// `deno task test:bun` unless something in it genuinely cannot load there.
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

/**
 * The same list for Bun. It is a separate map because the reasons differ: Bun
 * runs TypeScript natively rather than through Node's stripper, so a file the
 * stripper refuses could still run there.
 *
 * That it holds the same one entry is structural rather than a coincidence.
 * Bun does not read `deno.json`'s `imports`, and there is no tsconfig `paths`
 * here, so under Bun a bare specifier can only resolve through `node_modules`.
 * Any future test importing one is therefore exempt from both by construction.
 *
 * `bun run --no-install tests/npm-deps.ts` fails with `Cannot find module
 * '@huggingface/jinja'`, the same shape as Node's. WITHOUT the flag it passes,
 * because Bun downloads the package from the registry at runtime, which is the
 * thing principle 1 exists to forbid. That is why `test:bun` passes
 * `--no-install`, why `needs` below asserts the flag is still there, and why CI
 * runs the same command as a positive control expecting it to fail.
 */
const BUN_EXEMPT: Record<string, string> = {
  "npm-deps.ts":
    "same bare specifier as under Node: `bun run --no-install` is Cannot find module. Without " +
    "--no-install it passes only because Bun auto-installs @huggingface/jinja from the registry " +
    "at runtime, which is an npm install by another name and not a pass this file should claim.",
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
// Written once each: the flag check below re-derives its regex from `runner`,
// and the two have to be the same pattern for that derivation to cover exactly
// the invocations `invoked` counted.
const DENO_RUNNER = new RegExp(`deno run${FLAGS}`);
const NODE_RUNNER = new RegExp(`node${FLAGS}`);
const BUN_RUNNER = new RegExp(`bun(?:\\s+run)?${FLAGS}`);
const inTest = invoked("test", DENO_RUNNER);
/**
 * The non-Deno runtimes, each with the list it is checked against. Adding a
 * fourth means adding a row, not another copy of the loop below: the reason
 * this file exists is that two hand-maintained lists drifted from `tests/` and
 * nothing compared them.
 */
const RUNTIMES: {
  task: string;
  runner: RegExp;
  listed: Set<string>;
  exempt: Record<string, string>;
  map: string;
  how: string;
  needs: string[];
}[] = [
  {
    task: "test:node",
    runner: NODE_RUNNER,
    listed: invoked("test:node", NODE_RUNNER),
    exempt: NODE_EXEMPT,
    map: "NODE_EXEMPT",
    how: "node --experimental-strip-types",
    // Not listed: dropping it fails loudly on the 22.6.0 leg, which is the
    // oldest Node whose stripper this repo claims to run on.
    needs: [],
  },
  {
    // `bun run x.ts` and `bun x.ts` are both valid, so the `run` is optional
    // here; the flag is what has to be there, and `needs` is what says so.
    task: "test:bun",
    runner: BUN_RUNNER,
    listed: invoked("test:bun", BUN_RUNNER),
    exempt: BUN_EXEMPT,
    map: "BUN_EXEMPT",
    how: "bun run --no-install",
    // Load-bearing and otherwise unenforced: strip it and every check here
    // still passes while the job satisfies "no npm install" by installing.
    needs: ["--no-install"],
  },
];

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
  for (const rt of RUNTIMES) {
    const reason = rt.exempt[f];
    ok(
      rt.listed.has(f) || reason !== undefined,
      `tests/${f} is not run by \`deno task ${rt.task}\` and has no ${rt.map} entry. Run ` +
        `\`${rt.how} tests/${f}\`: if it passes, add it to the task; if it cannot load, add it ` +
        `to ${rt.map} with what fails`,
    );
    ok(
      !(rt.listed.has(f) && reason !== undefined),
      `tests/${f} is run by \`deno task ${rt.task}\` and also listed as exempt from it; drop ` +
        `the exemption`,
    );
  }
}

/** `test` under the same checks, minus the exemptions it does not have. */
const DENO_ROW = {
  task: "test",
  runner: DENO_RUNNER,
  listed: inTest,
  exempt: {} as Record<string, string>,
  map: "",
  how: "deno run",
  needs: [] as string[],
};
const ALL = [DENO_ROW, ...RUNTIMES];
for (const { task, listed } of ALL) {
  for (const f of listed) {
    ok(onDisk.includes(f), `\`deno task ${task}\` runs tests/${f}, which is not on disk`);
  }
}
for (const rt of ALL) {
  const body = cfg.tasks![rt.task] as string;
  // Per PATH, not per file. The per-file check above passes as soon as a file
  // is listed once, so `&& node tests/eta-fmt.ts` appended to test:bun hides
  // behind the correct invocation on the line before it.
  const mentioned = body.match(/tests\/[A-Za-z0-9._/-]+\.ts/g) ?? [];
  const claimed = body.match(
    new RegExp(`${rt.runner.source}\\s+tests/[A-Za-z0-9._/-]+\\.ts`, "g"),
  ) ?? [];
  // Occurrences against occurrences, not against `listed`, which is a Set: a
  // file invoked twice on purpose would otherwise be a false failure blaming a
  // cause that is not there.
  ok(
    mentioned.length === claimed.length,
    `deno.json's ${rt.task} task mentions ${mentioned.length} tests/ paths but ` +
      `${claimed.length} are invoked by "${rt.how}"; one is handed to another runtime, ` +
      `written ./tests/, or passed as an argument rather than as the entry point`,
  );
  // Every invocation carries the flags the runtime's promise depends on. A
  // flag in the task string is a claim, and nothing else here checks it.
  for (const flag of rt.needs) {
    for (const call of body.matchAll(new RegExp(`${rt.runner.source}\\s+tests/`, "g"))) {
      ok(
        call[0].includes(flag),
        `deno.json's ${rt.task} invokes \`${call[0].trim()}\` without ${flag}, which is what ` +
          `makes the task's promise real rather than assumed`,
      );
    }
  }
  for (const [f, reason] of Object.entries(rt.exempt)) {
    ok(onDisk.includes(f), `${rt.map} names tests/${f}, which is not on disk`);
    // An empty string satisfies a presence check, and an exemption without a
    // reason is what the three wrong ones would have collapsed to.
    ok(reason.trim().length > 20, `${rt.map}["${f}"] needs a reason, not a placeholder`);
  }
}

/**
 * Every task here is run by CI. This is the check #89 and #97 both needed and
 * neither had: in both, the task existed, was correct, and nothing on a runner
 * invoked it, so its guarantee held only for whoever ran it by hand. It couples
 * a test to a workflow file, which is the objection; comparing two
 * hand-maintained lists is what this file already does, which is the answer.
 */
const workflow = fs.readFileSync(path.join(root, ".github/workflows/test.yml"), "utf8");
for (const rt of ALL) {
  ok(
    new RegExp(`run:\\s+deno task ${rt.task}\\s*$`, "m").test(workflow),
    `.github/workflows/test.yml has no \`run: deno task ${rt.task}\` step, so that task is ` +
      `checked only by whoever runs it by hand. That is what #89 and #97 were`,
  );
}

console.log(
  `task-coverage: ${onDisk.length} test files, ${inTest.size} under deno, ` +
    RUNTIMES.map((r) =>
      `${r.listed.size} under ${r.task.replace(/^test:/, "")} ` +
      `(${Object.keys(r.exempt).length} exempt)`
    ).join(", ") + " ✓",
);
