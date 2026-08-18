// The one entry point. Every workflow in this repo is a subcommand here.
//
//   deno run -A cli.ts <command> [--flags]
//   deno run -A cli.ts help            # the command list
//   deno run -A cli.ts help --json     # the whole CLI as a machine-readable schema
//   deno run -A cli.ts <command> --help
//
// Exit codes are part of the contract: 0 success, 1 usage error (bad or missing
// flag, unreadable input), 2 runtime failure (the work started and could not
// finish). Diagnostics go to stderr prefixed "error:", everything else to stdout.

import { type Command, helpFor, parse, schema, UsageError } from "./src/cli/args.ts";
import { COMMANDS } from "./src/cli/registry.ts";

const BIN = "deno run -A cli.ts";

function commandList(): string {
  const groups = new Map<string, Command[]>();
  for (const c of COMMANDS) {
    const group = c.name.startsWith("eval") ? "evaluate" : GROUP[c.name] ?? "other";
    groups.set(group, [...(groups.get(group) ?? []), c]);
  }
  const lines = [
    "gguf-trainer - train a language model from scratch, in TypeScript, straight to GGUF.",
    "",
    `Usage: ${BIN} <command> [--flags]`,
    "",
  ];
  for (const [group, cmds] of groups) {
    lines.push(`${group}:`);
    for (const c of cmds) lines.push(`  ${c.name.padEnd(14)}${c.summary}`);
    lines.push("");
  }
  lines.push(
    `Run \`${BIN} <command> --help\` for a command's flags,`,
    `or \`${BIN} help --json\` for every command and flag as JSON.`,
    "",
    "Reading this as an agent? agents.md is the operating manual: recipes, invariants,",
    "and what each failure message means.",
  );
  return lines.join("\n");
}

const GROUP: Record<string, string> = {
  corpus: "prepare data",
  tokenize: "prepare data",
  "chat-corpus": "prepare data",
  "style-seed": "prepare data",
  "style-restyle": "prepare data",
  pretrain: "train",
  finetune: "train",
  export: "ship a model",
  inspect: "ship a model",
  generate: "ship a model",
  demo: "check the install",
  bench: "check the install",
  archs: "check the install",
};

function fail(message: string, code: number): never {
  console.error(`error: ${message}`);
  // deno-lint-ignore no-explicit-any
  const proc = (globalThis as any).process;
  if (proc?.exit) proc.exit(code);
  // deno-lint-ignore no-explicit-any
  (globalThis as any).Deno?.exit(code);
  throw new Error(message);
}

if (import.meta.main) {
  // deno-lint-ignore no-explicit-any
  const argv: string[] = (globalThis as any).Deno?.args ??
    // deno-lint-ignore no-explicit-any
    (globalThis as any).process?.argv?.slice(2) ?? [];
  const [name, ...rest] = argv;

  if (!name || name === "help" || name === "--help" || name === "-h") {
    if (rest.includes("--json") || argv.includes("--json")) {
      console.log(JSON.stringify(schema(COMMANDS, BIN), null, 2));
    } else if (rest[0] && !rest[0].startsWith("-")) {
      const target = COMMANDS.find((c) => c.name === rest[0]);
      if (!target) fail(`unknown command "${rest[0]}". Run \`${BIN} help\`.`, 1);
      console.log(helpFor(target, BIN));
    } else {
      console.log(commandList());
    }
  } else {
    const cmd = COMMANDS.find((c) => c.name === name);
    if (!cmd) {
      const names = COMMANDS.map((c) => c.name).join(", ");
      fail(`unknown command "${name}". Available: ${names}.`, 1);
    }
    if (rest.includes("--help") || rest.includes("-h")) {
      console.log(helpFor(cmd, BIN));
    } else {
      try {
        await cmd.run(parse(cmd, rest));
      } catch (e) {
        if (e instanceof UsageError) fail(`${e.message}`, 1);
        fail(`${cmd.name} failed: ${(e as Error).message}`, 2);
      }
    }
  }
}
