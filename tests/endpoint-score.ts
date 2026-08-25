// Standalone assert check for the dual-endpoint scorer. The two endpoints must
// NOT agree about a trailing handback label: on `/completions` it is the format's
// own convention and correct, on `/v1/chat/completions` it is the defect. A
// scorer that merges them reports whichever conclusion its bucket boundary
// happens to draw, which is exactly how the roleplay battery was misread twice
// (docs/optimization.md, lever 16).
//
// Run:  deno run tests/endpoint-score.ts
import { scoreChat, scoreRaw, stoppedOnEos } from "../scripts/eval-endpoints.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

// --- The clean case ---------------------------------------------------------

const clean = "The harbor records are in the third stack, behind the tea things.";
ok(!scoreChat(clean, "Iris", "You").handback, "a plain reply is not a handback");
ok(!scoreChat(clean, "Iris", "You").selflabel, "a plain reply carries no self-label");
ok(!scoreRaw(clean, "Iris", "You").handback, "a plain completion is not a handback");

// --- Where the endpoints must disagree --------------------------------------

const trailing = "The harbor records are in the third stack.\nYou:";
ok(
  scoreChat(trailing, "Iris", "You").handback,
  "a chat reply that opens the human's turn is the defect the goal names",
);
ok(
  !scoreRaw(trailing, "Iris", "You").handback,
  "on the raw endpoint a bare trailing label is the handback convention, not a defect",
);

const playedBothSides = "The harbor records are in the third stack.\nYou: Thanks, I'll look.";
ok(scoreRaw(playedBothSides, "Iris", "You").handback, "writing the human's turn is the defect");
ok(scoreChat(playedBothSides, "Iris", "You").handback, "and it is a defect on chat too");

// --- Which label ------------------------------------------------------------

// A model that memorized "You" hands back to it even when the prompt used a name.
ok(
  scoreRaw("Right this way.\nYou: Thank you.", "Iris", "Marcus").handback,
  "the memorized label counts even when the prompt used another",
);
ok(
  scoreRaw("Right this way.\nMarcus: Thank you.", "Iris", "Marcus").handback,
  "the prompt's own label counts too",
);
ok(
  !scoreRaw("Right this way, Marcus. The stacks bite.", "Iris", "Marcus").handback,
  "a name inside prose is not a speaker label",
);
ok(
  !scoreChat("You will find it in the third stack.", "Iris", "You").handback,
  "a sentence starting with You is not a label: the colon is what makes it one",
);

// --- Self-labelling ---------------------------------------------------------

ok(
  scoreChat("Iris: Third stack, behind the tea.", "Iris", "You").selflabel,
  "a reply prefixed with its own name renders as Iris: Iris: in a client",
);
ok(
  scoreChat("The Iris in the window box needs water.", "Iris", "You").selflabel === false,
  "the character's name in prose is not a label",
);
ok(
  scoreChat("Third stack.\nIris: And mind the tea.", "Iris", "You").selflabel,
  "a mid-reply re-label counts as well",
);

// A multi-word character name has regex metacharacters in it more often than not.
ok(
  scoreRaw("Gate's shut.\nYou: Please.", "Sergeant Idris Vale", "You").handback,
  "a multi-word character name must not break the match",
);
ok(
  scoreChat("Dr. Halima Reyes: The north ridge.", "Dr. Halima Reyes", "You").selflabel,
  "a name with a period must be escaped, not treated as a wildcard",
);

// --- Empty ------------------------------------------------------------------

ok(scoreChat("", "Iris", "You").empty, "an empty reply is empty");
ok(scoreChat("  \n ", "Iris", "You").empty, "whitespace is empty");
ok(!scoreChat("No.", "Iris", "You").empty, "a short refusal is a real reply");

// --- The stop reason ---------------------------------------------------------

// The count that matters most on the raw endpoint, and the easiest to read wrong:
// a model that stops correctly and a scorer looking at the wrong field both report
// the same number until you check which.
ok(stoppedOnEos({ stop_type: "eos" }), "stop_type eos is a stop on EOS");
ok(!stoppedOnEos({ stop_type: "limit" }), "stop_type limit is the token cap");
ok(!stoppedOnEos({ stop_type: "word" }), "stop_type word is a stop string");
ok(stoppedOnEos({ stopped_eos: true }), "older builds report a boolean instead");
ok(!stoppedOnEos({ stopped_eos: false }), "and the false case is still false");

let raised = false;
try {
  stoppedOnEos({ content: "hi", index: 0 });
} catch {
  raised = true;
}
ok(raised, "a response with no stop field must throw: absent is not the same as never");

console.log("=== endpoint score checks passed ===");
