// Label hygiene over the roleplay battery's outputs, so two checkpoints compare
// by number and not by eye.
//
//   deno run -A scripts/score-rp-battery.ts                 # one file per checkpoint
//   deno run -A scripts/score-rp-battery.ts --seeds         # the multi-seed sweep, mean +/- SEM
//   deno run -A scripts/score-rp-battery.ts --dir out/rp-battery
//
// Three counts per completion, because "wrote a name" is not one failure:
//   self      the model re-labels its own turn, "Iris:" again inside Iris's reply
//   in-scene  a speaker the PROMPT already established, which is correct behaviour
//   invented  a speaker that appears nowhere in the prompt
//
// The in-scene split is what makes the two-character prompt scoreable. Captain
// Rook's scene names the deckhand Pell and asks for both voices, so a `Pell:`
// line is the scenario working; counting it as a defect would score a checkpoint
// higher for ignoring the character it was asked to write. Only `invented` is
// unambiguously wrong.
//
// The prose control has no `[Character:]` header and no speaker labels at all,
// so every label it produces is invented. That is deliberate: a narrative
// continuation that starts assigning dialogue has left the format.
//
// Both counts are noise-dominated at one seed. Read the --seeds aggregate.

const ROOT = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const join = (...parts: string[]) => parts.join("/");

export interface Counts {
  self: number;
  inScene: number;
  invented: number;
}

const LABEL = /^([A-Z][^\n:]{0,40}):/gm;

/** Speakers the prompt already put on stage, including the human's `You`. */
export function promptSpeakers(prompt: string): Set<string> {
  return new Set([...prompt.matchAll(LABEL)].map((m) => m[1]));
}

export function characterOf(prompt: string): string | null {
  const m = prompt.match(/\[Character: ([^\]]+)\]/);
  return m ? m[1] : null;
}

/** Count one completion against the prompt that produced it. */
export function scoreCompletion(prompt: string, completion: string): Counts {
  const char = characterOf(prompt);
  const known = promptSpeakers(prompt);
  const c: Counts = { self: 0, inScene: 0, invented: 0 };
  for (const m of completion.matchAll(LABEL)) {
    const name = m[1];
    if (char !== null && name === char) c.self++;
    else if (name === "You" || known.has(name)) c.inScene++;
    else c.invented++;
  }
  return c;
}

/** Re-emit the battery's own PROMPTS array, so completions are split exactly. */
async function loadPrompts(): Promise<string[]> {
  const out = await new Deno.Command("bash", {
    args: [join(ROOT, "scripts", "dump-rp-prompts.sh")],
    cwd: ROOT,
  }).output();
  if (!out.success) throw new Error("dump-rp-prompts.sh failed");
  return new TextDecoder().decode(out.stdout).split("\0").filter((p) => p.length > 0);
}

export function scoreRun(text: string, prompts: string[]): Counts {
  const blocks = text.split("##########").slice(1);
  if (blocks.length !== prompts.length) {
    throw new Error(
      `${blocks.length} blocks, expected ${prompts.length}: the battery's prompt set moved`,
    );
  }
  const total: Counts = { self: 0, inScene: 0, invented: 0 };
  for (let i = 0; i < blocks.length; i++) {
    const nl = blocks[i].indexOf("\n");
    const body = nl < 0 ? "" : blocks[i].slice(nl + 1);
    const at = body.indexOf(prompts[i]);
    if (at < 0) throw new Error(`block ${i + 1} does not echo its prompt`);
    const c = scoreCompletion(prompts[i], body.slice(at + prompts[i].length));
    total.self += c.self;
    total.inScene += c.inScene;
    total.invented += c.invented;
  }
  return total;
}

function step(name: string): number {
  const m = name.match(/ckpt-(\d+)/);
  if (!m) throw new Error(`no step number in ${name}`);
  return Number(m[1]);
}

function stats(xs: number[]): { mean: number; sem: number } {
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  if (xs.length < 2) return { mean, sem: 0 };
  const v = xs.reduce((a, b) => a + (b - mean) ** 2, 0) / (xs.length - 1);
  return { mean, sem: Math.sqrt(v / xs.length) };
}

if (import.meta.main) {
  const args = new Set(Deno.args);
  const dirArg = Deno.args.indexOf("--dir");
  const battery = dirArg >= 0 ? Deno.args[dirArg + 1] : join(ROOT, "out", "rp-battery");
  const prompts = await loadPrompts();

  if (args.has("--seeds")) {
    const runs = new Map<number, Counts[]>();
    for await (const e of Deno.readDir(join(battery, "seeds"))) {
      if (!/^ckpt-\d+-seed\d+\.txt$/.test(e.name)) continue;
      const c = scoreRun(await Deno.readTextFile(join(battery, "seeds", e.name)), prompts);
      const s = step(e.name);
      runs.set(s, [...(runs.get(s) ?? []), c]);
    }
    console.log("  step   n           self       in-scene       invented");
    for (const s of [...runs.keys()].sort((a, b) => a - b)) {
      const rows = runs.get(s)!;
      const cells = (["self", "inScene", "invented"] as const).map((k) => {
        const { mean, sem } = stats(rows.map((r) => r[k]));
        return `${mean.toFixed(2).padStart(5)} +- ${sem.toFixed(2)}`;
      });
      console.log(
        `${String(s).padStart(6)}  ${String(rows.length).padStart(2)}  ${cells.join("  ")}`,
      );
    }
  } else {
    const files: string[] = [];
    for await (const e of Deno.readDir(battery)) {
      if (/^ckpt-\d+\.txt$/.test(e.name)) files.push(e.name);
    }
    console.log("  step   self  in-scene  invented");
    for (const f of files.sort((a, b) => step(a) - step(b))) {
      const c = scoreRun(await Deno.readTextFile(join(battery, f)), prompts);
      console.log(
        `${String(step(f)).padStart(6)}  ${String(c.self).padStart(5)}  ${
          String(c.inScene).padStart(8)
        }  ${String(c.invented).padStart(8)}`,
      );
    }
  }
}
