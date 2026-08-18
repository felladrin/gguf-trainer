// `archs`: what this trainer can build.
//
// The list comes from the registry, so an architecture that has been added
// correctly shows up here with no further work. If yours is missing from this
// output, it is not registered.

import type { Command, Values } from "../cli/args.ts";
import { ARCHITECTURES, DEFAULT_ARCH } from "../model/registry.ts";

function run(v: Values) {
  if (v.bool("json")) {
    console.log(JSON.stringify(
      {
        default: DEFAULT_ARCH,
        architectures: ARCHITECTURES.map((a) => ({
          name: a.name,
          summary: a.summary,
          reference: a.reference ?? null,
          flags: a.flags.map((f) => ({
            name: f.name,
            type: f.type,
            describe: f.describe,
            default: f.default ?? null,
          })),
        })),
      },
      null,
      2,
    ));
    return;
  }

  for (const a of ARCHITECTURES) {
    console.log(`${a.name}${a.name === DEFAULT_ARCH ? "  (default)" : ""}`);
    console.log(`  ${a.summary}`);
    if (a.reference) console.log(`  shape-compatible with: ${a.reference}`);
    console.log(`  flags: ${a.flags.map((f) => `--${f.name}`).join(" ")}`);
    console.log("");
  }
  console.log(
    "Pass one with `pretrain --arch <name>`. A checkpoint records its own architecture,\n" +
      "so --resume does not need it. To add one, see docs/adding-an-architecture.md.",
  );
}

export const archsCommand: Command = {
  name: "archs",
  summary: "List the model architectures this build can train.",
  details: `Every architecture is one file in src/arch/ registered in src/model/registry.ts. The
CLI flags, the tests and the checkpoint loader all read that registry, so this list is
the ground truth for what \`--arch\` accepts.`,
  examples: ["archs", "archs --json"],
  flags: [
    { name: "json", type: "boolean", describe: "machine-readable output" },
  ],
  run: run,
};
