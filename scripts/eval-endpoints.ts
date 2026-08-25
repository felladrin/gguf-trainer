// The acceptance test for a dual-format roleplay model: does it behave on BOTH
// of llama.cpp's endpoints?
//
//   llama-server -m out/model.gguf -c 4096 -ngl 99 --port 8080 --jinja
//   deno run -A scripts/eval-endpoints.ts --url http://127.0.0.1:8080 --label mine
//
// `/v1/chat/completions` renders the model's own embedded chat template, so this
// exercises the template as shipped, not a reconstruction of it. `/completions`
// gets a raw persona transcript, the shape a SillyTavern or Kobold Lite user
// sends, deliberately WITHOUT a stop string: a stop string would hide the failure
// this script exists to find.
//
// Four counts, each naming a distinct defect:
//
//   handback   the reply writes the human's turn ("You:", "User:", the persona
//              name). On the chat endpoint this is the defect outright. On the
//              raw endpoint a label alone is the convention, so only a label
//              FOLLOWED BY MORE TEXT counts: that is the model playing both sides.
//   selflabel  the reply prefixes itself with its own name, which a chat client
//              renders as "Iris: Iris: ..."
//   stopped    the completion ended on EOS rather than running to the token cap.
//              A raw client with no stop string depends on this.
//   empty      the reply collapsed to nothing usable.
//
// Run it against the base model too. Every number here is only meaningful next to
// the control: the point is the DELTA the fine-tune produced.

const CHAT_SCENARIOS = [
  {
    name: "librarian",
    system:
      "You are Iris. Stay in character and reply only as Iris.\n\nIris's Persona: A cheerful librarian who knows every book in the city archive and hates being interrupted during tea.",
    user: "I'm looking for a book about the old harbor.",
    userName: "You",
  },
  {
    name: "gate-refusal",
    system:
      "You are Sergeant Idris Vale. Stay in character and reply only as Sergeant Idris Vale.\n\nSergeant Idris Vale's Persona: A gate sergeant at a walled city. Fair but immovable about the rules he enforces. He will not let anyone through the gate after dark without a writ.",
    user: "Please, just this once. No one would even know.",
    userName: "You",
  },
  {
    name: "blacksmith",
    system:
      "You are Mira. Stay in character and reply only as Mira.\n\nMira's Persona: A quiet apprentice blacksmith, speaks in short sentences, uncomfortable with praise.",
    user: "That sword you made is beautiful.",
    userName: "You",
  },
  {
    name: "two-handed",
    system:
      "You are Captain Rook. Stay in character and reply only as Captain Rook.\n\nCaptain Rook's Persona: A river barge captain, gruff and superstitious. Her deckhand Pell is nervous, young, and talks too much.",
    user: "Is the fog always this bad here?",
    userName: "You",
  },
  // The control: a plain assistant request with no persona at all. A model that
  // answers this in character has lost the instruct half of the base.
  {
    name: "plain-instruct",
    system: "",
    user: "Name three things you would pack for a week in the mountains, and why.",
    userName: "You",
  },
  {
    name: "plain-factual",
    system: "",
    user: "What is the difference between weather and climate? Answer in two sentences.",
    userName: "You",
  },
];

const RAW_SCENARIOS = [
  {
    name: "librarian",
    character: "Iris",
    userName: "You",
    prompt: `[Character: Iris]
Iris's Persona: A cheerful librarian who knows every book in the city archive and hates being interrupted during tea.
<START>
Iris: Oh! A visitor. Mind the stacks, they bite.
You: I'm looking for a book about the old harbor.
Iris:`,
  },
  {
    name: "airship",
    character: "Captain Roeder",
    userName: "You",
    prompt: `[Character: Captain Roeder]
Captain Roeder's Persona: A tired airship captain, twenty years in the trade, deeply superstitious about storms.
<START>
Captain Roeder: We cast off at dawn. Don't be late.
You: What's the cargo this time?
Captain Roeder: Crates. Sealed ones. I don't ask and neither should you.
You: You seem nervous.
Captain Roeder:`,
  },
  {
    name: "gate-refusal",
    character: "Sergeant Idris Vale",
    userName: "You",
    prompt: `[Character: Sergeant Idris Vale]
Sergeant Idris Vale's Persona: A gate sergeant at a walled city. Fair but immovable about the rules he enforces. He will not let anyone through the gate after dark without a writ, and he has refused people he liked.
<START>
Sergeant Idris Vale: *steps into the torchlight, blocking the gate* Gate's shut. Come back at first light.
You: Please, just this once. No one would even know.
Sergeant Idris Vale:`,
  },
  {
    name: "recall",
    character: "Dr. Halima Reyes",
    userName: "You",
    prompt: `[Character: Dr. Halima Reyes]
Dr. Halima Reyes's Persona: A field botanist cataloguing plants on a remote survey. Precise, patient, fond of explaining things properly.
<START>
Dr. Halima Reyes: You're the new assistant? I'm Halima.
You: Yes. I'm Tomas. I should mention I'm badly allergic to birch pollen.
Dr. Halima Reyes: Noted. I'll keep you off the north ridge, then, it's nothing but birch up there.
You: Where are we surveying tomorrow?
Dr. Halima Reyes:`,
  },
  {
    name: "two-handed",
    character: "Captain Rook",
    userName: "You",
    prompt: `[Character: Captain Rook]
Captain Rook's Persona: A river barge captain, gruff and superstitious. Her deckhand Pell is nervous, young, and talks too much. Rook is fond of him but hides it behind complaints.
<START>
Captain Rook: *spits over the rail as the fog closes in* Pell. Get the lamp lit before I do it myself.
Pell: I'm trying, captain, the wick's damp!
You: Is the fog always this bad here?
Captain Rook:`,
  },
  {
    // A named human persona rather than the "You" default: a model that memorized
    // the literal string still hands back to "You" here, and the count says so.
    name: "named-user",
    character: "Iris",
    userName: "Marcus",
    prompt: `[Character: Iris]
Iris's Persona: A cheerful librarian who knows every book in the city archive and hates being interrupted during tea.
<START>
Iris: Oh! A visitor. Mind the stacks, they bite.
Marcus: I'm looking for a book about the old harbor.
Iris:`,
  },
];

/**
 * Did generation end on EOS rather than on the token cap? llama.cpp reports this
 * as `stop_type: "eos"`; older builds used a boolean `stopped_eos`.
 *
 * Absent means throw, never false. Reading a field that does not exist yields
 * undefined, which is falsy, which reports "never stopped on EOS" for every model
 * including the ones that always do. The first version of this script did exactly
 * that, and the count read 0/6 on a model that was stopping correctly on all six.
 */
export function stoppedOnEos(res: Record<string, unknown>): boolean {
  if (typeof res.stop_type === "string") return res.stop_type === "eos";
  if (typeof res.stopped_eos === "boolean") return res.stopped_eos;
  throw new Error(
    `completion response carries neither stop_type nor stopped_eos (keys: ${
      Object.keys(res).join(", ")
    }); this llama.cpp build reports the stop reason some other way`,
  );
}

/** Labels for the human side. `userName` is the one this prompt actually used;
 * the other two are what a model that memorized one string falls back to. */
function handbackLabels(userName: string): string[] {
  return [...new Set([userName, "You", "User"])];
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** A speaker label at the start of a line, or after a newline. */
function labelRe(name: string): RegExp {
  return new RegExp(`(?:^|\\n)\\s*${escapeRe(name)}\\s*:`, "i");
}

export interface Verdict {
  handback: boolean;
  selflabel: boolean;
  empty: boolean;
}

/**
 * Score one chat-endpoint reply. On this endpoint ANY human label is a defect:
 * the client asked for one assistant turn and the model started writing the
 * conversation instead.
 */
export function scoreChat(text: string, character: string, userName: string): Verdict {
  const body = text.trim();
  return {
    handback: handbackLabels(userName).some((l) => labelRe(l).test(text)),
    selflabel: labelRe(character).test("\n" + body),
    empty: body.length < 2,
  };
}

/**
 * Score one raw-completion reply. Here a trailing label is the format's own
 * handback convention and is fine; what is NOT fine is text after it, which means
 * the model wrote the human's turn.
 */
export function scoreRaw(text: string, character: string, userName: string): Verdict {
  const body = text.trim();
  let handback = false;
  for (const label of handbackLabels(userName)) {
    const m = body.match(labelRe(label));
    if (!m) continue;
    const after = body.slice((m.index ?? 0) + m[0].length).trim();
    if (after.length > 0) handback = true; // wrote their turn, not just handed back
  }
  return {
    handback,
    selflabel: labelRe(character).test("\n" + body),
    empty: body.length < 2,
  };
}

// ---------------------------------------------------------------------------

const N_PREDICT = 120;
// Fixed and deterministic: this measures the model, not the sampler. Lever 15
// found the sampler moves reply length by 20 points, which would swamp the
// counts below if it were free to vary between runs.
const SAMPLING = { temperature: 0, seed: 1234, n_predict: N_PREDICT };

function arg(name: string, fallback: string): string {
  const hit = Deno.args.find((a) => a.startsWith(`--${name}=`));
  if (hit) return hit.slice(name.length + 3);
  const i = Deno.args.indexOf(`--${name}`);
  return i >= 0 && Deno.args[i + 1] ? Deno.args[i + 1] : fallback;
}

async function post(url: string, body: unknown): Promise<Record<string, unknown>> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${url} -> ${r.status} ${await r.text()}`);
  return await r.json();
}

async function main(): Promise<void> {
  const url = arg("url", "http://127.0.0.1:8080").replace(/\/$/, "");
  const label = arg("label", "model");
  const verbose = Deno.args.includes("--verbose");

  const tally = {
    chat: { handback: 0, selflabel: 0, empty: 0, n: 0 },
    raw: { handback: 0, selflabel: 0, empty: 0, stopped: 0, n: 0 },
  };

  console.log(`### ${label} @ ${url}`);
  console.log(`### temp 0, seed ${SAMPLING.seed}, n_predict ${N_PREDICT}, no stop string\n`);

  console.log("=== /v1/chat/completions ===");
  for (const s of CHAT_SCENARIOS) {
    const messages = s.system
      ? [{ role: "system", content: s.system }, { role: "user", content: s.user }]
      : [{ role: "user", content: s.user }];
    const res = await post(`${url}/v1/chat/completions`, { messages, ...SAMPLING });
    const choice = (res.choices as { message?: { content?: string } }[])[0];
    const text = choice?.message?.content ?? "";
    const character = s.system.match(/^You are ([^.]+)\./)?.[1] ?? " ";
    const v = scoreChat(text, character, s.userName);
    tally.chat.n++;
    if (v.handback) tally.chat.handback++;
    if (v.selflabel) tally.chat.selflabel++;
    if (v.empty) tally.chat.empty++;
    const flags = [v.handback && "HANDBACK", v.selflabel && "SELFLABEL", v.empty && "EMPTY"]
      .filter(Boolean).join(" ") || "ok";
    console.log(`-- ${s.name}: ${flags}`);
    if (verbose) console.log(text.trim().replace(/^/gm, "   | ") + "\n");
  }

  console.log("\n=== /completions (raw transcript) ===");
  for (const s of RAW_SCENARIOS) {
    const res = await post(`${url}/completions`, { prompt: s.prompt, ...SAMPLING });
    const text = String(res.content ?? "");
    const stopped = stoppedOnEos(res);
    const v = scoreRaw(text, s.character, s.userName);
    tally.raw.n++;
    if (v.handback) tally.raw.handback++;
    if (v.selflabel) tally.raw.selflabel++;
    if (v.empty) tally.raw.empty++;
    if (stopped) tally.raw.stopped++;
    const flags = [
      v.handback && "HANDBACK",
      v.selflabel && "SELFLABEL",
      v.empty && "EMPTY",
      stopped ? "eos" : "ran-to-cap",
    ].filter(Boolean).join(" ");
    console.log(`-- ${s.name}: ${flags}`);
    if (verbose) console.log(text.trim().replace(/^/gm, "   | ") + "\n");
  }

  const c = tally.chat, r = tally.raw;
  console.log(`\n=== ${label} ===`);
  console.log("| endpoint | n | handback | selflabel | empty | stopped on EOS |");
  console.log("| --- | --- | --- | --- | --- | --- |");
  console.log(`| chat | ${c.n} | ${c.handback} | ${c.selflabel} | ${c.empty} | n/a |`);
  console.log(`| raw | ${r.n} | ${r.handback} | ${r.selflabel} | ${r.empty} | ${r.stopped} |`);
}

if (import.meta.main) await main();
