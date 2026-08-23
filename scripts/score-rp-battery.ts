// Label hygiene over the roleplay battery's outputs, so two checkpoints compare
// by number and not by eye.
//
//   deno run -A scripts/score-rp-battery.ts                 # one file per checkpoint
//   deno run -A scripts/score-rp-battery.ts --seeds         # the multi-seed sweep, mean +/- SEM
//   deno run -A scripts/score-rp-battery.ts --dir out/rp-battery
//
// Four counts per completion. Two earlier versions of this scorer had two and
// three, and both lumped a defect together with correct behaviour in a way that
// reversed the conclusion drawn from them, so the buckets are spelled out:
//
//   handback  `You:`, the model writing the human's turn. `-r "You:"` stops
//             generation there, so this is the truncation the battery is built
//             to expose, not a neutral event.
//   self      the character re-labelling its own turn with no other speaker in
//             between. An alternation like `Pell:` then `Captain Rook:` is a
//             scene taking turns, so only a consecutive repeat counts.
//   costar    a speaker the PROMPT staged, `Pell:` in the two-character scene.
//             That scenario asks for both voices, so this is the scenario
//             working; counting it as a defect scores a checkpoint higher for
//             ignoring the character it was told to write.
//   invented  a speaker that appears nowhere in the prompt. The only bucket
//             that is unambiguously wrong.
//
// The prose control has no `[Character:]` header, so any label at all is either
// handback or invented: a narrative continuation writing dialogue attributions
// has left the format.
//
// All four are noise-dominated at one seed. Read the --seeds aggregate, and
// note that between-checkpoint spread is about the size of within-checkpoint
// spread, so treat a trend across checkpoints with suspicion.

const ROOT = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const join = (...parts: string[]) => parts.join("/");

export interface Counts {
  handback: number;
  self: number;
  costar: number;
  invented: number;
}

const LABEL = /^([A-Z][^\n:]{0,40}):/gm;

/**
 * Speakers the prompt staged in the transcript itself. Anchored after `<START>`
 * so the persona block's own `Iris's Persona:` line does not become a name the
 * model can regurgitate for free.
 */
export function promptSpeakers(prompt: string): Set<string> {
  const at = prompt.indexOf("<START>");
  const scene = at < 0 ? prompt : prompt.slice(at);
  return new Set([...scene.matchAll(LABEL)].map((m) => m[1]));
}

export function characterOf(prompt: string): string | null {
  const m = prompt.match(/\[Character: ([^\]]+)\]/);
  return m ? m[1] : null;
}

/** Count one completion against the prompt that produced it. */
export function scoreCompletion(prompt: string, completion: string): Counts {
  const char = characterOf(prompt);
  const known = promptSpeakers(prompt);
  const c: Counts = { handback: 0, self: 0, costar: 0, invented: 0 };
  let previous: string | null = char;
  for (const m of completion.matchAll(LABEL)) {
    const name = m[1];
    if (name === "You") c.handback++;
    else if (char !== null && name === char) {
      if (previous === char) c.self++;
      else c.costar++; // taking the turn back after someone else spoke
    } else if (known.has(name)) c.costar++;
    else c.invented++;
    previous = name;
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
  const total: Counts = { handback: 0, self: 0, costar: 0, invented: 0 };
  for (let i = 0; i < blocks.length; i++) {
    const nl = blocks[i].indexOf("\n");
    const body = nl < 0 ? "" : blocks[i].slice(nl + 1);
    const at = body.indexOf(prompts[i]);
    if (at < 0) throw new Error(`block ${i + 1} does not echo its prompt`);
    const c = scoreCompletion(prompts[i], body.slice(at + prompts[i].length));
    total.handback += c.handback;
    total.self += c.self;
    total.costar += c.costar;
    total.invented += c.invented;
  }
  return total;
}

function die(msg: string): never {
  console.error(msg);
  Deno.exit(2);
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
  const battery = dirArg >= 0
    ? (Deno.args[dirArg + 1] ?? die("--dir needs a path"))
    : join(ROOT, "out", "rp-battery");
  const prompts = await loadPrompts();

  if (args.has("--seeds")) {
    const runs = new Map<number, Counts[]>();
    for await (const e of Deno.readDir(join(battery, "seeds"))) {
      if (!/^ckpt-\d+-seed\d+\.txt$/.test(e.name)) continue;
      const c = scoreRun(await Deno.readTextFile(join(battery, "seeds", e.name)), prompts);
      const s = step(e.name);
      runs.set(s, [...(runs.get(s) ?? []), c]);
    }
    console.log("  step   n       handback           self         costar       invented");
    for (const s of [...runs.keys()].sort((a, b) => a - b)) {
      const rows = runs.get(s)!;
      const cells = (["handback", "self", "costar", "invented"] as const).map((k) => {
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
    console.log("  step  handback   self  costar  invented");
    for (const f of files.sort((a, b) => step(a) - step(b))) {
      const c = scoreRun(await Deno.readTextFile(join(battery, f)), prompts);
      console.log(
        `${String(step(f)).padStart(6)}  ${String(c.handback).padStart(8)}  ${
          String(c.self).padStart(5)
        }  ${String(c.costar).padStart(6)}  ${String(c.invented).padStart(8)}`,
      );
    }
  }
}
