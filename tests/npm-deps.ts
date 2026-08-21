// The two npm dependencies, exercised rather than only type-checked: hyparquet
// behind `parseDataFile` and @huggingface/jinja behind the exported chat
// template. `deno check` proves their signatures still resolve; nothing else in
// the suite proves they still behave, which is why renovate holds them back.
// Run:  deno run -A tests/npm-deps.ts
import { parseDataFile } from "../src/data/parse.ts";
import { DEFAULT_CHAT_TEMPLATE, detectMapping } from "../src/data/chat.ts";
import { readFileBytes } from "../src/io.ts";
import { Template } from "@huggingface/jinja";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

// --- hyparquet -------------------------------------------------------------

const rows = await parseDataFile(
  "tests/fixtures/tiny.parquet",
  await readFileBytes("tests/fixtures/tiny.parquet"),
);
ok(rows.length === 2, `expected 2 rows, got ${rows.length}`);

// Non-ASCII has to survive the round trip as a string, not as bytes.
ok(rows[0].bot_name === "Íris", `bot_name: ${JSON.stringify(rows[0].bot_name)}`);
ok(rows[0].turns === 2, `turns: ${JSON.stringify(rows[0].turns)}`);
ok(rows[0].score === 0.5 && rows[1].score === null, `score: ${JSON.stringify(rows[1].score)}`);

// The list-of-structs shape is the one `detectMapping` reads a dataset through,
// so a flattened or stringified column would break every conversational corpus.
const convo = rows[0].conversation as { from: string; value: string }[];
ok(Array.isArray(convo) && convo.length === 2, `conversation: ${JSON.stringify(convo)}`);
ok(convo[0].from === "human" && convo[1].from === "gpt", JSON.stringify(convo.map((m) => m.from)));
ok(convo[1].value === "Sou a Íris — a bibliotecária.", JSON.stringify(convo[1].value));

const mapping = detectMapping(rows[0]);
ok(mapping?.kind === "conversations", `mapping: ${JSON.stringify(mapping)}`);

// hyparquet reads through an AsyncBuffer, so parseParquet has to slice the
// backing ArrayBuffer at the view's own offset. A Uint8Array that does not start
// at 0 is the case that catches a slice written as (0, byteLength): every caller
// today hands over a fresh whole-file array, so nothing else would notice.
const whole = await readFileBytes("tests/fixtures/tiny.parquet");
const padded = new Uint8Array(whole.byteLength + 8);
padded.set(whole, 8);
const offsetRows = await parseDataFile("x.parquet", padded.subarray(8));
ok(offsetRows.length === 2, `offset view: ${offsetRows.length} rows`);
ok(offsetRows[0].bot_name === "Íris", `offset view: ${JSON.stringify(offsetRows[0].bot_name)}`);

// --- @huggingface/jinja ----------------------------------------------------

const render = new Template(DEFAULT_CHAT_TEMPLATE);
const messages = [
  { role: "system", content: "You are Iris." },
  { role: "user", content: "Hello?" },
  { role: "assistant", content: "Mind the stacks." },
];

const full = render.render({ messages, add_generation_prompt: false });
ok(
  full === "<|im_start|>system\nYou are Iris.<|im_end|>\n" +
      "<|im_start|>user\nHello?<|im_end|>\n" +
      "<|im_start|>assistant\nMind the stacks.<|im_end|>\n",
  JSON.stringify(full),
);

// The trailing newline is load-bearing, not cosmetic: strip it and the model's
// first sampled token is ":" completing the header, which reads as a broken
// model rather than a broken prompt.
const prompt = render.render({ messages: messages.slice(0, 2), add_generation_prompt: true });
ok(prompt.endsWith("<|im_start|>assistant\n"), JSON.stringify(prompt.slice(-40)));

// The tools branch takes a different path through the template: it opens its own
// system turn, so the leading system message must not also be emitted by the loop.
const withTools = render.render({
  messages: [{ role: "user", content: "weather?" }],
  add_generation_prompt: false,
  tools: [{ type: "function", function: { name: "get_weather" } }],
});
ok(withTools.startsWith("<|im_start|>system\n# Tools\n"), JSON.stringify(withTools.slice(0, 40)));
ok(
  withTools.includes('{"type": "function", "function": {"name": "get_weather"}}'),
  JSON.stringify(withTools),
);
ok(
  withTools.endsWith("<|im_start|>user\nweather?<|im_end|>\n"),
  JSON.stringify(withTools.slice(-40)),
);

// A tool result is folded into a user turn, and consecutive tool messages share
// one turn rather than opening a header each.
const toolTurn = render.render({
  messages: [
    { role: "user", content: "weather?" },
    { role: "tool", content: "sunny" },
    { role: "tool", content: "18C" },
  ],
  add_generation_prompt: false,
});
ok(
  toolTurn.endsWith(
    "<|im_start|>user\n<tool_response>\nsunny\n</tool_response>\n<tool_response>\n18C\n</tool_response><|im_end|>\n",
  ),
  JSON.stringify(toolTurn),
);

console.log("npm-deps: all checks passed");
