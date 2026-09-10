// Every test file is in `deno task test`, and in `deno task test:node` unless
// it has a stated reason not to be.
//
// The static hyparquet import that produced #83 is only half of why 79
// assertions over eval-choice's scoring arithmetic went uncovered under Node.
// The import made `tests/eval-tasks.ts` fail there; what made that invisible is
// that both task lists are hand-maintained strings in deno.json with nothing
// comparing them to what is actually in tests/. Adding a file to one and
// forgetting the other is silent, and so is the reverse. A doc rule is what
// already failed here, so this is a check instead.
//
// Run:  deno run -A tests/task-coverage.ts
import * as fs from "node:fs";

/**
 * Files deliberately outside `test:node`, each with the reason.
 *
 * Add to this only for a real runtime constraint. "It has not been tried" is
 * what this check exists to surface.
 */
const NODE_EXEMPT: Record<string, string> = {
  "npm-deps.ts": "exercises the npm dependencies themselves, so it needs Deno's import map",
  "rp-chats.ts": "drives scripts/build-rp-chats.ts, which reads Deno.args",
  "rp-battery-score.ts":
    "drives scripts/score-rp-battery.ts, which uses Deno.Command and Deno.readDir",
  "endpoint-score.ts": "drives scripts/eval-endpoints.ts, same Deno-only tooling",
  "task-coverage.ts": "reads deno.json to check the two task lists; nothing for Node to add",
};

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

const cfg = JSON.parse(fs.readFileSync("deno.json", "utf8")) as {
  tasks: Record<string, string>;
};
const listed = (task: string) =>
  new Set(
    Array.from(cfg.tasks[task].matchAll(/tests\/([A-Za-z0-9._-]+\.ts)/g), (m) => m[1]),
  );

const onDisk = fs.readdirSync("tests")
  .filter((f) => f.endsWith(".ts"))
  .sort();
const inTest = listed("test");
const inNode = listed("test:node");

for (const f of onDisk) {
  ok(inTest.has(f), `tests/${f} is not in \`deno task test\`; add it, or delete the file`);
  const exempt = f in NODE_EXEMPT;
  ok(
    inNode.has(f) || exempt,
    `tests/${f} is not in \`deno task test:node\` and has no entry in NODE_EXEMPT. ` +
      `Try \`node --experimental-strip-types tests/${f}\`: if it runs, add it to the task; ` +
      `if it cannot, add it here with the reason.`,
  );
  ok(
    !(inNode.has(f) && exempt),
    `tests/${f} is in \`deno task test:node\` and also listed as exempt from it; drop the exemption`,
  );
}

for (const f of inTest) {
  ok(onDisk.includes(f), `\`deno task test\` runs tests/${f}, which is not on disk`);
}
for (const f of inNode) {
  ok(onDisk.includes(f), `\`deno task test:node\` runs tests/${f}, which is not on disk`);
}
for (const f of Object.keys(NODE_EXEMPT)) {
  ok(onDisk.includes(f), `NODE_EXEMPT names tests/${f}, which is not on disk`);
}

console.log(
  `task-coverage: ${onDisk.length} test files, ${inTest.size} in test, ${inNode.size} in ` +
    `test:node, ${Object.keys(NODE_EXEMPT).length} exempt with a reason ✓`,
);
