// The architecture registry.
//
// Adding an architecture is: write src/arch/<name>.ts, import it here, add it to
// the list. Nothing else in the repo needs to change. The CLI builds its flags
// from this list, `archs` prints it, the tests loop over it, and a checkpoint
// finds its own architecture through it.

import type { Architecture, ModelConfig } from "./arch.ts";
import type { Flag, Values } from "../cli/args.ts";
import { UsageError } from "../cli/args.ts";
import type { GGUFFile } from "../gguf/gguf.ts";
import { gemma3 } from "../arch/gemma3.ts";
import { llama } from "../arch/llama.ts";
import { qwen3 } from "../arch/qwen3.ts";

// deno-lint-ignore no-explicit-any
export const ARCHITECTURES: Architecture<any>[] = [gemma3, llama, qwen3];

/** The default when no --arch is given: the one this project trained first. */
export const DEFAULT_ARCH = "gemma3";

export function archNames(): string[] {
  return ARCHITECTURES.map((a) => a.name);
}

/** Look up an architecture, failing with the list of valid names. */
// deno-lint-ignore no-explicit-any
export function getArch(name: string): Architecture<any> {
  const a = ARCHITECTURES.find((x) => x.name === name);
  if (!a) throw new Error(`unknown architecture "${name}". Available: ${archNames().join(", ")}`);
  return a;
}

/**
 * The architecture a GGUF was written by. A checkpoint always knows its own
 * architecture, so resuming never needs the user to repeat it.
 */
// deno-lint-ignore no-explicit-any
export function archFromGGUF(g: GGUFFile): Architecture<any> {
  const name = g.metadata.get("general.architecture");
  if (typeof name !== "string") throw new Error("GGUF has no general.architecture");
  const a = ARCHITECTURES.find((x) => x.name === name);
  if (!a) {
    throw new Error(
      `GGUF architecture "${name}" is not implemented here (have: ${archNames().join(", ")}). ` +
        `See docs/adding-an-architecture.md.`,
    );
  }
  return a;
}

/** Read a checkpoint's config without building the model. */
export function configFromGGUF(g: GGUFFile): ModelConfig {
  return archFromGGUF(g).configFromGGUF(g);
}

/**
 * Every architecture's flags, merged for a command that accepts `--arch`.
 * Shared names (--kv-heads, --ffn-dim, --rope-base) appear once; the owning
 * architectures are tracked so passing a flag that belongs to a DIFFERENT
 * architecture is an error rather than a silent no-op.
 *
 * Only the first owner's help text can be shown for a shared flag, so when the
 * owners disagree about what the flag defaults to, the merged text says so
 * instead of quietly presenting one architecture's default as everyone's.
 */
export function mergedArchFlags(): { flags: Flag[]; owners: Map<string, string[]> } {
  const flags: Flag[] = [];
  const owners = new Map<string, string[]>();
  const contested = new Set<string>();
  for (const a of ARCHITECTURES) {
    for (const f of a.flags) {
      const seen = owners.get(f.name);
      if (seen) {
        seen.push(a.name);
        const first = flags.find((x) => x.name === f.name);
        if (first && first.describe !== f.describe) contested.add(f.name);
        continue;
      }
      owners.set(f.name, [a.name]);
      flags.push(f);
    }
  }
  return {
    flags: flags.map((f) =>
      contested.has(f.name)
        ? { ...f, describe: `${f.describe}; differs per --arch, see \`archs --json\`` }
        : f
    ),
    owners,
  };
}

/** Fail when a flag the user passed belongs only to some other architecture. */
export function assertFlagsBelongTo(archName: string, v: Values): void {
  const { owners } = mergedArchFlags();
  for (const [flag, archs] of owners) {
    if (v.given(flag) && !archs.includes(archName)) {
      throw new UsageError(
        `--${flag} belongs to the ${archs.join("/")} architecture, but --arch is ${archName}`,
      );
    }
  }
}
