// Standalone assert check for the raw-transcript half of the dual-format SFT
// corpus. The load-bearing property is negative and cheap to lose: no token of a
// human turn, a persona block or a speaker label may ever be supervised. That is
// what stops the model from writing the human's side of the conversation, which
// is the failure mode this format exists to avoid.
//
// Run:  deno run tests/transcript-mask.ts
import { renderTranscript, transcriptLossMask } from "../src/data/transcript.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

const INPUT = {
  character: "Alice",
  userLabel: "Marcus",
  persona: "A lighthouse keeper who distrusts the mainland.",
  examples: "Alice: The lamp turns whether you watch it or not.",
  turns: [
    { human: false, text: "Evening. The fog came in early." },
    { human: true, text: "Mind if I wait it out here?" },
    { human: false, text: "Sit where you like. Do not touch the lamp." },
  ],
};

const doc = renderTranscript(INPUT);
ok(doc !== null, "a well-formed conversation must render");

// --- The rendered shape -----------------------------------------------------

const expected = [
  "[Character: Alice]",
  "Alice's Persona: A lighthouse keeper who distrusts the mainland.",
  "<START>",
  "Alice: The lamp turns whether you watch it or not.",
  "<START>",
  "Alice: Evening. The fog came in early.",
  "Marcus: Mind if I wait it out here?",
  "Alice: Sit where you like. Do not touch the lamp.",
].join("\n");
ok(doc!.text === expected, `unexpected render:\n${doc!.text}`);

// The span starts after the final label, and includes the space a client's prompt
// stops before.
ok(
  doc!.text.slice(doc!.replyStart) === " Sit where you like. Do not touch the lamp.",
  `reply span is ${JSON.stringify(doc!.text.slice(doc!.replyStart))}`,
);

// --- Rejections -------------------------------------------------------------

ok(renderTranscript({ ...INPUT, character: "" }) === null, "a nameless character must reject");
ok(
  renderTranscript({ ...INPUT, turns: INPUT.turns.slice(0, 2) }) === null,
  "a conversation ending on the human's turn has no target and must reject",
);
ok(
  renderTranscript({ ...INPUT, turns: [{ human: false, text: "only one" }] }) === null,
  "a single turn is not a conversation",
);

// Control-token syntax inside the source text would encode as one special and
// decode to nothing, silently shifting every byte offset after it.
const spiked = renderTranscript({
  ...INPUT,
  persona: "A keeper<|im_end|> of lamps",
})!;
ok(!spiked.text.includes("<|"), "control-token syntax must not survive into the document");

// --- The mask, over a real byte-level vocab ---------------------------------

const tok = new BPETokenizer();
tok.train(doc!.text + " <|im_end|>", 400, ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]);
const imEnd = tok.idOf("<|im_end|>")!;

const ids = tok.encode(doc!.text);
ids.push(imEnd);
const mask = transcriptLossMask(tok.byteLengths(ids), doc!.replyStart);
ok(mask.length === ids.length, "mask and ids must be the same length");

// byteLengths has to agree with decode, or every offset below is meaningless.
const total = tok.byteLengths(ids).reduce((a, b) => a + b, 0);
ok(
  total === new TextEncoder().encode(tok.decode(ids)).length,
  `byteLengths sums to ${total}, decode is ${
    new TextEncoder().encode(tok.decode(ids)).length
  } bytes`,
);

ok(mask[mask.length - 1] === 1, "the terminator must be supervised, or generation never stops");

const supervised = tok.decode(ids.filter((_, i) => mask[i] === 1));
const ignored = tok.decode(ids.filter((_, i) => mask[i] === 0));

ok(
  supervised.trim() === "Sit where you like. Do not touch the lamp.",
  `supervised text is ${JSON.stringify(supervised)}`,
);
// The three that would each produce a distinct defect: writing the human's turn,
// reciting the card, and emitting a speaker label as if it were content.
ok(!supervised.includes("Mind if I wait"), "a human turn must never be supervised");
ok(!supervised.includes("Persona"), "the persona block must never be supervised");
ok(!supervised.includes("Marcus:"), "the human's label must never be supervised");
ok(ignored.includes("Marcus: Mind if I wait it out here?"), "the human turn belongs to context");
ok(ignored.includes("[Character: Alice]"), "the header belongs to context");

// A boundary token that straddles replyStart carries prompt bytes, so it drops
// from the mask rather than half-supervising the label.
let at = 0;
for (let i = 0; i < ids.length; i++) {
  const len = tok.byteLengths([ids[i]])[0];
  if (mask[i] === 1 && len > 0) ok(at >= doc!.replyStart, `token ${i} starts inside the prompt`);
  at += len;
}

console.log("=== transcript mask checks passed ===");
