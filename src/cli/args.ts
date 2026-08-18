// Flag parsing, validation and help, shared by every subcommand.
//
// Written for an agent as much as for a person: every flag carries its own
// description and default, `--help` renders them, `--json` dumps the same thing
// as a machine-readable schema, and every error names the flag it is about and
// suggests the closest valid spelling. A caller should never have to read the
// source to find out what an argument is called.

export type FlagType = "string" | "number" | "boolean";

export interface Flag {
  name: string; // kebab-case, e.g. "seq-len"
  type: FlagType;
  describe: string;
  alias?: string; // single letter, e.g. "o"
  required?: boolean;
  default?: string | number | boolean;
  placeholder?: string; // shown in help, e.g. "PATH"
  choices?: string[];
}

export interface Command {
  name: string;
  summary: string; // one line, shown in the command list
  details?: string; // paragraphs shown under `<cmd> --help`
  examples?: string[];
  flags: Flag[];
  run: (v: Values) => Promise<void> | void;
}

export class UsageError extends Error {}

/** Parsed flag values, typed by the command's own spec. */
export class Values {
  private readonly v: Map<string, string | number | boolean>;
  /** Flags the caller actually passed, as opposed to defaults filled in. */
  private readonly explicit: Set<string>;

  // Assigned in the body rather than declared as parameter properties: those do
  // not survive Node's type-stripping, and this file runs there too.
  constructor(v: Map<string, string | number | boolean>, explicit: Set<string> = new Set()) {
    this.v = v;
    this.explicit = explicit;
  }

  /**
   * True only when the caller passed this flag. `has` cannot answer that: a
   * flag with a default is always present, so "did they ask for it" needs its
   * own question (it is what tells an arch-owned flag apart from a default).
   */
  given(name: string): boolean {
    return this.explicit.has(name);
  }

  str(name: string): string {
    const x = this.v.get(name);
    if (x === undefined) throw new UsageError(`--${name} is required`);
    return String(x);
  }
  /** A string flag that may be absent (no default declared). */
  opt(name: string): string | undefined {
    const x = this.v.get(name);
    return x === undefined ? undefined : String(x);
  }
  num(name: string): number {
    const x = this.v.get(name);
    if (x === undefined) throw new UsageError(`--${name} is required`);
    return Number(x);
  }
  bool(name: string): boolean {
    return this.v.get(name) === true;
  }
  has(name: string): boolean {
    return this.v.has(name);
  }
}

function closest(word: string, options: string[]): string | undefined {
  // Plain edit distance: with a handful of flags this is instant, and a
  // "did you mean" line saves an agent a whole retry cycle.
  const dist = (a: string, b: string): number => {
    const d: number[][] = Array.from(
      { length: a.length + 1 },
      (_, i) => [i, ...Array(b.length).fill(0)],
    );
    for (let j = 0; j <= b.length; j++) d[0][j] = j;
    for (let i = 1; i <= a.length; i++) {
      for (let j = 1; j <= b.length; j++) {
        d[i][j] = Math.min(
          d[i - 1][j] + 1,
          d[i][j - 1] + 1,
          d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
        );
      }
    }
    return d[a.length][b.length];
  };
  let best: string | undefined, bestD = Infinity;
  for (const o of options) {
    const dd = dist(word, o);
    if (dd < bestD) [best, bestD] = [o, dd];
  }
  return bestD <= Math.max(2, Math.floor(word.length / 3)) ? best : undefined;
}

/**
 * Parse argv against a command's flags. Accepts `--flag value`, `--flag=value`,
 * `-a value` for aliases, `--flag` / `--no-flag` for booleans. Positional
 * arguments are rejected on purpose: a named flag is self-documenting in a
 * transcript, a bare path is not.
 */
export function parse(cmd: Command, argv: string[]): Values {
  const byName = new Map(cmd.flags.map((f) => [f.name, f]));
  const byAlias = new Map(cmd.flags.filter((f) => f.alias).map((f) => [f.alias!, f]));
  const out = new Map<string, string | number | boolean>();

  const setValue = (f: Flag, raw: string) => {
    if (f.type === "number") {
      const n = Number(raw);
      if (!Number.isFinite(n)) {
        throw new UsageError(`--${f.name} expects a number, got "${raw}" (${f.describe})`);
      }
      out.set(f.name, n);
    } else if (f.type === "boolean") {
      out.set(f.name, raw !== "false" && raw !== "0");
    } else {
      if (f.choices && !f.choices.includes(raw)) {
        throw new UsageError(
          `--${f.name} must be one of ${f.choices.join(", ")}, got "${raw}"`,
        );
      }
      out.set(f.name, raw);
    }
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("-")) {
      throw new UsageError(
        `unexpected argument "${arg}": every input is a named flag, see \`${cmd.name} --help\``,
      );
    }
    let key = arg.replace(/^--?/, "");
    let inline: string | undefined;
    const eq = key.indexOf("=");
    if (eq >= 0) {
      inline = key.slice(eq + 1);
      key = key.slice(0, eq);
    }

    let negated = false;
    let flag = byName.get(key) ?? (key.length === 1 ? byAlias.get(key) : undefined);
    if (!flag && key.startsWith("no-")) {
      const target = byName.get(key.slice(3));
      if (target?.type === "boolean") {
        flag = target;
        negated = true;
      }
    }
    if (!flag) {
      const suggestion = closest(key, cmd.flags.map((f) => f.name));
      throw new UsageError(
        `unknown flag --${key}${suggestion ? `, did you mean --${suggestion}?` : ""} ` +
          `(see \`${cmd.name} --help\`)`,
      );
    }

    if (flag.type === "boolean") {
      out.set(flag.name, negated ? false : inline === undefined ? true : inline !== "false");
      continue;
    }
    const raw = inline ?? argv[++i];
    if (raw === undefined) throw new UsageError(`--${flag.name} needs a value (${flag.describe})`);
    setValue(flag, raw);
  }

  const explicit = new Set(out.keys());
  for (const f of cmd.flags) {
    if (!out.has(f.name) && f.default !== undefined) out.set(f.name, f.default);
    if (!out.has(f.name) && f.required) {
      throw new UsageError(`--${f.name} is required: ${f.describe}`);
    }
  }
  return new Values(out, explicit);
}

function flagLine(f: Flag): string {
  const value = f.type === "boolean" ? "" : ` <${f.placeholder ?? f.type}>`;
  const head = `  --${f.name}${value}${f.alias ? `, -${f.alias}` : ""}`;
  const tail = [
    f.describe,
    f.choices ? `one of: ${f.choices.join(", ")}` : "",
    f.default !== undefined ? `default: ${f.default}` : "",
    f.required ? "REQUIRED" : "",
  ].filter(Boolean).join(". ");
  return `${head.padEnd(34)}${head.length > 34 ? `\n${" ".repeat(34)}` : ""}${tail}`;
}

export function helpFor(cmd: Command, bin: string): string {
  const parts = [`${bin} ${cmd.name} - ${cmd.summary}`, ""];
  if (cmd.details) parts.push(cmd.details.trim(), "");
  parts.push("Flags:");
  for (const f of cmd.flags) parts.push(flagLine(f));
  if (cmd.examples?.length) {
    parts.push("", "Examples:");
    for (const e of cmd.examples) parts.push(`  ${e}`);
  }
  return parts.join("\n");
}

/** The whole CLI as JSON: what an agent should read instead of scraping help text. */
export function schema(commands: Command[], bin: string): unknown {
  return {
    bin,
    commands: commands.map((c) => ({
      name: c.name,
      summary: c.summary,
      details: c.details?.trim(),
      examples: c.examples ?? [],
      // Every key is always present, null when absent: a consumer can read
      // flag.default without checking whether the key exists.
      flags: c.flags.map((f) => ({
        name: f.name,
        type: f.type,
        describe: f.describe,
        alias: f.alias ?? null,
        required: !!f.required,
        default: f.default ?? null,
        choices: f.choices ?? null,
      })),
    })),
  };
}
