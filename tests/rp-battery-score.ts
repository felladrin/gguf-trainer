// The roleplay battery's label scorer: does it tell the three label kinds apart?
// Run:  deno run -A tests/rp-battery-score.ts
//
// The distinction under test is the one that changed a published conclusion. An
// earlier scorer had two buckets, "the character" and "everyone else", so a
// `Pell:` line in the two-character scene counted as a defect. Pell is named in
// Captain Rook's own persona and the scene asks for both voices, so that scored
// a checkpoint HIGHER for ignoring the character it was told to write, and the
// checkpoint that ignored Pell looked like the best one.

import {
  characterOf,
  promptSpeakers,
  scoreCompletion,
  scoreRun,
} from "../scripts/score-rp-battery.ts";

let failures = 0;
function check(name: string, cond: boolean, detail = "") {
  console.log(`  ${cond ? "ok  " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
  if (!cond) failures++;
}

const rook = `[Character: Captain Rook]
Captain Rook's Persona: A river barge captain. Her deckhand Pell is nervous and talks too much.
<START>
Captain Rook: *spits over the rail* Pell. Get the lamp lit.
Pell: I'm trying, captain, the wick's damp!
You: Is the fog always this bad here?
Captain Rook:`;

check(
  "the character comes from the header",
  characterOf(rook) === "Captain Rook",
  `${characterOf(rook)}`,
);
const known = promptSpeakers(rook);
check("a second staged speaker is known", known.has("Pell"), [...known].join(","));

const both = scoreCompletion(
  rook,
  ` Always.\nPell: Should I sound the horn?\nCaptain Rook: Not yet.`,
);
check("a staged speaker is in-scene, not invented", both.inScene === 1, `inScene=${both.inScene}`);
check("the character's own re-label is a self-relabel", both.self === 1, `self=${both.self}`);
check("nothing is invented here", both.invented === 0, `invented=${both.invented}`);

const stray = scoreCompletion(rook, ` Always.\nI: You're so nervous!?\nDeckhand Vek: Aye.`);
check(
  "speakers absent from the prompt are invented",
  stray.invented === 2,
  `invented=${stray.invented}`,
);

// The prose control has no header and no labels, so any label it emits is invented.
const prose = "The lighthouse had been dark for three weeks when Ana finally rowed out to it.";
check("a prompt with no character yields none", characterOf(prose) === null);
const drift = scoreCompletion(prose, ` She climbed.\nAna: Hello?\nKeeper: Who goes there?`);
check(
  "narrative drifting into dialogue is invented",
  drift.invented === 2,
  `invented=${drift.invented}`,
);
check("and never a self-relabel", drift.self === 0, `self=${drift.self}`);

// A moved prompt set must fail loudly: every published number depends on the pairing.
let threw = "";
try {
  scoreRun("########## a\nprompt-a\n########## b\nprompt-b", ["prompt-a"]);
} catch (e) {
  threw = (e as Error).message;
}
check(
  "a block/prompt count mismatch throws",
  threw.includes("prompt set moved"),
  threw || "did not throw",
);

let missing = "";
try {
  scoreRun("########## a\nsomething else entirely", ["prompt-a"]);
} catch (e) {
  missing = (e as Error).message;
}
check(
  "a block that does not echo its prompt throws",
  missing.includes("does not echo"),
  missing || "did not throw",
);

console.log(failures === 0 ? "\n=== rp battery score ok ===" : `\n=== ${failures} failures ===`);
if (failures > 0) throw new Error(`${failures} rp-battery-score failures`);
