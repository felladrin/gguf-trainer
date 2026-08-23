// The roleplay battery's label scorer: does it tell the three label kinds apart?
// Run:  deno run -A tests/rp-battery-score.ts
//
// Two earlier scorers each merged a defect with correct behaviour and each
// reversed a conclusion drawn from them. First, "everyone else" counted `Pell:`
// as a stray label, though the two-character scene asks for both voices, which
// scored a checkpoint HIGHER for ignoring the character it was told to write.
// Then the fix filed `You:` under correct behaviour, when the model writing the
// human's turn is the truncation the battery exists to expose, and `You:` was
// 76% of that bucket. Every one of those four distinctions is pinned below.

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

const alt = scoreCompletion(
  rook,
  ` Always.\nPell: Should I sound the horn?\nCaptain Rook: Not yet.`,
);
check("a staged co-star is not a defect", alt.costar === 2, `costar=${alt.costar}`);
check(
  "taking the turn back after Pell is alternation, not a re-label",
  alt.self === 0,
  `self=${alt.self}`,
);
check("nothing is invented here", alt.invented === 0, `invented=${alt.invented}`);

const repeat = scoreCompletion(rook, ` Always.\nCaptain Rook: And another thing.`);
check("a consecutive repeat IS a self-relabel", repeat.self === 1, `self=${repeat.self}`);

const handback = scoreCompletion(rook, ` Always.\nYou: Thanks.`);
check(
  "the human's turn is a handback, never correct behaviour",
  handback.handback === 1,
  `handback=${handback.handback}`,
);
check("and is not counted as a co-star", handback.costar === 0, `costar=${handback.costar}`);

const stray = scoreCompletion(rook, ` Always.\nI: You're so nervous!?\nDeckhand Vek: Aye.`);
check(
  "speakers absent from the prompt are invented",
  stray.invented === 2,
  `invented=${stray.invented}`,
);

// The persona block names the character too; only the transcript stages speakers.
check(
  "a persona-block label is not a staged speaker",
  !known.has("Captain Rook's Persona"),
  [...known].join(","),
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
const proseYou = scoreCompletion(prose, ` She climbed.\nYou: Who's there?`);
check(
  "even with no character, You: is a handback",
  proseYou.handback === 1,
  `handback=${proseYou.handback}`,
);

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
