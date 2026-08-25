// Standalone assert check for the dual-format corpus builder: the two arms a
// PIPPA row can become, and the identity scrubbing. Nothing is downloaded;
// importing the module must not run main.
//
// Run:  deno run tests/rp-chats.ts
import {
  type ChatRow,
  mulberry32,
  parsePippa,
  pippaToChatML,
  pippaToTranscript,
  scrubSystem,
  toChatRow,
  usefulExamples,
  USER_LABELS,
} from "../scripts/build-rp-chats.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

const ROW = {
  bot_name: "Iris",
  bot_description: "A cartographer who has never left {{char}}'s island.",
  bot_definitions: "{{char}}: The map is not the shore.\nEND_OF_DIALOG\n{{user}}: Show me.",
  conversation: [
    {
      message: "The tide chart says you are late, and the tide does not keep notes for anyone.",
      is_human: false,
    },
    {
      message: "I brought the ink you asked for, all four jars of it, wrapped in oilcloth.",
      is_human: true,
    },
    {
      message: "Set it by the window where the light is even, and do not spill a drop of it.",
      is_human: false,
    },
    { message: "What are you drawing there, at the edge of the big sheet?", is_human: true },
    { message: "The reef, before it moves again. It always moves again.", is_human: false },
  ],
};

const parsed = parsePippa(ROW)!;
ok(parsed !== null, "a well-formed PIPPA row must parse");
ok(parsed.character === "Iris", "the character name comes from bot_name");
ok(
  parsed.persona === "A cartographer who has never left Iris's island.",
  `placeholders must resolve: ${parsed.persona}`,
);
ok(!parsed.examples.includes("END_OF_DIALOG"), "example separators must be normalized away");
ok(parsed.turns.length === 5, `expected 5 turns, got ${parsed.turns.length}`);

// --- Rejections -------------------------------------------------------------

ok(parsePippa({ ...ROW, bot_name: "" }) === null, "a nameless card must reject");
ok(
  parsePippa({ ...ROW, conversation: ROW.conversation.slice(0, 2) }) === null,
  "a two-turn log is below the floor",
);

// --- The ChatML arm ---------------------------------------------------------

const chat = pippaToChatML(parsed)!;
ok(chat.format === undefined, "the ChatML arm carries no format tag");
ok(chat.messages[0].role === "system", "the persona goes in a system turn");
ok(chat.messages[0].content.startsWith("You are Iris."), "the system turn addresses the model");
ok(chat.messages[1].role === "user", "a chat exchange opens on the human");
ok(
  chat.messages[chat.messages.length - 1].role === "assistant",
  "it must close on the character, or there is nothing to supervise",
);
for (let i = 2; i < chat.messages.length; i++) {
  ok(chat.messages[i].role !== chat.messages[i - 1].role, `roles repeat at ${i}`);
}

// Consecutive same-side turns merge rather than rendering as two blocks in a row.
const doubled = parsePippa({
  ...ROW,
  conversation: [
    {
      message: "First half of a thought that got sent before it was finished, as they do.",
      is_human: true,
    },
    {
      message: "Second half, same speaker, arriving a moment later on its own line.",
      is_human: true,
    },
    {
      message: "A reply that waited for both halves before saying anything at all.",
      is_human: false,
    },
  ],
})!;
const merged = pippaToChatML(doubled)!;
ok(merged.messages.length === 3, `expected system+user+assistant, got ${merged.messages.length}`);
ok(merged.messages[1].content.includes("First half"), "the merged turn keeps the first half");
ok(merged.messages[1].content.includes("Second half"), "the merged turn keeps the second half");

// --- The transcript arm -----------------------------------------------------

const tx = pippaToTranscript(parsed, 0.99, "Marcus")!;
ok(tx.format === "transcript", "the transcript arm is tagged");
ok(tx.user_label === "Marcus", "the human label is the caller's");
ok(tx.character === "Iris", "the character name carries through");
ok(
  tx.messages[tx.messages.length - 1].role === "assistant",
  "the document must end on the target reply",
);

// The cut point actually moves: an early cut must not carry the late turns.
const early = pippaToTranscript(parsed, 0, "You")!;
const lateText = early.messages.map((m) => m.content).join(" ");
ok(!lateText.includes("The reef, before it moves again."), "an early cut must stop early");
ok(
  tx.messages.map((m) => m.content).join(" ").includes("The reef, before it moves again."),
  "a late cut must reach the last turn",
);

ok(
  USER_LABELS.filter((l) => l === "You").length * 2 > USER_LABELS.length,
  "You stays the majority",
);
ok(new Set(USER_LABELS).size > 1, "one literal label is what taught the old models to say You:");

// --- Foreign identity -------------------------------------------------------

ok(
  scrubSystem("You are an AI assistant named Claude created by Anthropic.") === null,
  "a system prompt claiming another product's identity must be dropped",
);
ok(scrubSystem("You are a helpful assistant.") === "You are a helpful assistant.", "plain kept");

const scrubbed = toChatRow({
  conversations: [
    { from: "system", value: "You are Claude, made by Anthropic." },
    { from: "human", value: "Hi." },
    { from: "gpt", value: "Hello." },
  ],
})!;
ok(scrubbed.messages.length === 2, "the scrubbed system turn leaves the exchange intact");
ok(scrubbed.messages[0].role === "user", "the exchange still opens on the human");

// --- Row shapes -------------------------------------------------------------

const pairs = toChatRow({ prompt: "Write a chapter.", chosen: "It was a dark night." })!;
ok(pairs.messages.length === 2, "prompt/chosen pairs become a two-turn exchange");
ok(pairs.messages[1].content === "It was a dark night.", "the chosen side is the target");
ok(toChatRow({ prompt: "x", rejected: "y" }) === null, "a row with no chosen side is unusable");
ok(
  toChatRow({ messages: [{ role: "user", content: "unanswered" }] }) === null,
  "a row that never reaches an assistant turn has nothing to supervise",
);

// --- Determinism ------------------------------------------------------------

const draw = (seed: number) => {
  const r = mulberry32(seed);
  return [r(), r(), r()].join(",");
};
ok(draw(11) === draw(11), "the corpus must rebuild byte-identically");
ok(draw(11) !== draw(12), "different seeds must actually differ");

const rows: ChatRow[] = [chat, tx];
ok(rows.filter((r) => r.format === "transcript").length === 1, "both arms are represented");

// --- Degenerate example blocks ----------------------------------------------

ok(
  usefulExamples("Iris: The map is not the shore, and never was.", "Iris") !== "",
  "a real example line must survive",
);
ok(usefulExamples(":\n\nexample.com", "Iris") === "", "a bare colon and a URL are not dialogue");
ok(usefulExamples("Iris: hi", "Iris") === "", "a two-word stub is not an example");
ok(usefulExamples("Personality: brave, tall, quiet", "Iris") === "", "a W++ dump is not dialogue");

console.log("=== rp chat corpus checks passed ===");
