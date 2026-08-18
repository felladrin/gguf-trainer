// Standalone assert check for the roleplay-corpus renderer: placeholder
// substitution, Character.AI definition normalization, and the drop filters
// (nothing downloaded; importing the module must not run main).
// Run:  deno run tests/rp-corpus.ts
import {
  DEFAULT_RENDER,
  normalizeDefinitions,
  type PippaRow,
  renderRow,
  substitute,
} from "../scripts/build-rp-corpus.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

const turns = (n: number) =>
  Array.from({ length: n }, (_, i) => ({
    message: `${"utterance ".repeat(12)}${i}`,
    is_human: i % 2 === 1,
  }));

const row = (over: Partial<PippaRow> = {}): PippaRow => ({
  bot_name: "Ayaka",
  bot_greeting: "Hello there.",
  bot_description: "A maid who swapped bodies with her mistress.",
  bot_definitions: "{{char}}: Hi.\n{{user}}: Who?\nEND_OF_DIALOG\n{{random_user_1}}: Again?",
  conversation: turns(6),
  ...over,
});

// substitute
ok(substitute("{{char}} met {{user}}", "Ayaka") === "Ayaka met You", "char/user placeholders");
ok(
  substitute("{{CHAR}} and {{User}}", "Ayaka") === "Ayaka and You",
  "placeholders are case-insensitive",
);
ok(
  substitute("{{random_user_12}}: hi", "Ayaka") === "You: hi",
  "random_user_N collapses to the user",
);
ok(substitute("<BOT> saw <USER>", "Ayaka") === "Ayaka saw You", "angle-bracket placeholders");
ok(!substitute("a <|endoftext|> b", "Ayaka").includes("<|endoftext|>"), "doc marker stripped");

// normalizeDefinitions
const defs = normalizeDefinitions("{{char}}: Hi.\nEND_OF_DIALOG\n{{user}}: Yo.", "Ayaka");
ok(defs.includes("Ayaka: Hi."), "definitions get substituted");
ok(
  defs === "Ayaka: Hi.\n\nYou: Yo.",
  `END_OF_DIALOG becomes a blank line, got ${JSON.stringify(defs)}`,
);
ok(!defs.includes("<START>"), "example chats do not each get their own <START>");
ok(!defs.includes("END_OF_DIALOG"), "no raw END_OF_DIALOG survives");
ok(
  !normalizeDefinitions("{{char}}: Hi.\n<START>\n{{user}}: Yo.", "Ayaka").includes("<START>"),
  "a <START> already in the source is stripped, so only the renderer emits them",
);
ok(!/\n\n\n/.test(normalizeDefinitions("a\n\n\n\nb", "X")), "blank runs collapse");

// renderRow: the happy path
const r = renderRow(row());
ok("text" in r, "a well-formed row renders");
const text = (r as { text: string }).text;
ok(text.startsWith("[Character: Ayaka]"), "opens with the character header");
ok(text.includes("Ayaka's Persona: A maid"), "carries the persona line");
ok(text.includes("\nYou: "), "human turns are labeled You");
ok(text.includes("\nAyaka: "), "bot turns are labeled with the character name");
// Two markers, whatever the number of example chats: one opening the example
// block, one opening the real log. That is the shape a prompt arrives in.
ok(text.split("<START>").length - 1 === 2, "one <START> for the examples, one for the log");
const bare = renderRow(row({ bot_definitions: "" })) as { text: string };
ok(bare.text.split("<START>").length - 1 === 1, "no definitions means exactly one <START>");

// The greeting is emitted only when the log does not already open with it.
const dup = renderRow(
  row({ conversation: [{ message: "Hello there.", is_human: false }, ...turns(5)] }),
);
ok(
  (dup as { text: string }).text.split("Ayaka: Hello there.").length === 2,
  "greeting is not duplicated",
);

// Filters
ok("reject" in renderRow(row({ bot_name: "" })), "unnamed rows drop");
ok("reject" in renderRow(row({ bot_name: "x".repeat(41) })), "absurd names drop");
ok(
  "text" in renderRow(row({ conversation: turns(3) })),
  "3-turn logs are kept (pippa_deduped's own floor)",
);
ok("reject" in renderRow(row({ conversation: turns(2) })), "sub-3-turn logs drop");
ok("reject" in renderRow(row({ conversation: [] })), "empty logs drop");
ok(
  "reject" in
    renderRow(
      row({
        bot_description: "",
        bot_definitions: "",
        conversation: [{ message: "hi", is_human: false }, { message: "yo", is_human: true }, {
          message: "ok",
          is_human: false,
        }, { message: "np", is_human: true }],
      }),
    ),
  "too-short transcripts drop",
);
ok(
  "reject" in renderRow(row({ bot_description: "の".repeat(3000), bot_definitions: "" })),
  "mostly non-ASCII transcripts drop",
);

// Truncation lands on a turn boundary, never mid-utterance.
const long = renderRow(row({ conversation: turns(400) }), { ...DEFAULT_RENDER, maxChars: 2000 });
const cut = (long as { text: string }).text;
ok(cut.length <= 2000, `truncated to ${cut.length} <= 2000`);
ok(/(^|\n)(You|Ayaka): /.test(cut.slice(cut.lastIndexOf("\n"))), "last line is a whole turn");

console.log("rp-corpus: all checks passed");
