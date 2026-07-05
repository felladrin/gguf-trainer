// Conversational-dataset normalization + the default Qwen3 chat template.
//
// Hugging Face conversational datasets come in several shapes (OpenAI `messages`,
// ShareGPT `conversations`, Alpaca `instruction/input/output`, plain
// `prompt/response`). This module turns any of them into a normalized
// ChatMessage[] so the rest of the pipeline is schema-agnostic. It is pure logic
// with no dependency — the Jinja *rendering* of these messages into a training
// string lives in the web layer (server + client), which is where a Jinja engine
// is available; here we only supply the template text and the normalization.

export interface ChatMessage {
  role: string; // "system" | "user" | "assistant" | "tool"
  content: string;
}

/** The four things a model can be trained to be; drives data + template use. */
export type ModelType = "base" | "instruct" | "reasoning" | "tools";

// The Qwen3-4B-Instruct-2507 chat template, verbatim. String.raw keeps the
// Jinja source's `\n` and `\"` as literal backslash sequences (a normal string
// literal would collapse `\n` into a real newline, corrupting the template).
export const DEFAULT_QWEN3_CHAT_TEMPLATE = String.raw`{%- if tools -%}
    {{- "<|im_start|>system\n" -}}
    {%- if messages[0].role == "system" -%}
        {{- messages[0].content + "\n\n" -}}
    {%- endif -%}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" -}}
    {%- for tool in tools -%}
        {{- "\n" -}}
        {{- tool | tojson -}}
    {%- endfor -%}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" -}}
{%- elif messages[0].role == "system" -%}
    {{- "<|im_start|>system\n" + messages[0].content + "<|im_end|>\n" -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message.content is string -%}
        {%- set content = message.content -%}
    {%- else -%}
        {%- set content = "" -%}
    {%- endif -%}
    {%- if message.role == "user" or message.role == "system" and not loop.first -%}
        {{- "<|im_start|>" + message.role + "\n" + content + "<|im_end|>" + "\n" -}}
    {%- elif message.role == "assistant" -%}
        {{- "<|im_start|>" + message.role + "\n" + content -}}
        {%- if message.tool_calls -%}
            {%- for tool_call in message.tool_calls -%}
                {%- if loop.first and content or not loop.first -%}
                    {{- "\n" -}}
                {%- endif -%}
                {%- if tool_call.function -%}
                    {%- set tool_call = tool_call.function -%}
                {%- endif -%}
                {{- "<tool_call>\n{\"name\": \"" -}}
                {{- tool_call.name -}}
                {{- "\", \"arguments\": " -}}
                {%- if tool_call.arguments is string -%}
                    {{- tool_call.arguments -}}
                {%- else -%}
                    {{- tool_call.arguments | tojson -}}
                {%- endif -%}
                {{- "}\n</tool_call>" -}}
            {%- endfor -%}
        {%- endif -%}
        {{- "<|im_end|>\n" -}}
    {%- elif message.role == "tool" -%}
        {%- if loop.first or messages[loop.index0 - 1].role != "tool" -%}
            {{- "<|im_start|>user" -}}
        {%- endif -%}
        {{- "\n<tool_response>\n" -}}
        {{- content -}}
        {{- "\n</tool_response>" -}}
        {%- if loop.last or messages[loop.index0 + 1].role != "tool" -%}
            {{- "<|im_end|>\n" -}}
        {%- endif -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "<|im_start|>assistant\n" -}}
{%- endif -%}`;

/** ChatML control tokens a chat/tool model needs as atomic specials. */
export const CHATML_SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"];

/**
 * How to read a dataset's columns. `kind` picks the shape; the field names say
 * which columns carry the data (auto-detected, overridable in the wizard).
 */
export interface FieldMapping {
  kind: "text" | "messages" | "conversations" | "instruction" | "prompt";
  textField?: string; // kind=text
  messagesField?: string; // kind=messages/conversations (list column)
  roleKey?: string; // per-item role key (messages: "role"; conversations: "from")
  contentKey?: string; // per-item content key (messages: "content"; conversations: "value")
  systemField?: string; // kind=instruction: optional system column
  instructionField?: string; // kind=instruction
  inputField?: string; // kind=instruction: optional extra input
  outputField?: string; // kind=instruction
  promptField?: string; // kind=prompt
  responseField?: string; // kind=prompt
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function firstKey(row: Record<string, unknown>, candidates: string[]): string | undefined {
  for (const c of candidates) if (c in row) return c;
  return undefined;
}

/** Map a ShareGPT/other role token onto our canonical set. */
export function normalizeRole(role: string): string {
  const r = role.toLowerCase();
  if (r === "human" || r === "user") return "user";
  if (r === "gpt" || r === "assistant" || r === "bot" || r === "chatgpt") return "assistant";
  if (r === "system") return "system";
  if (r === "tool" || r === "observation" || r === "function") return "tool";
  return r;
}

/**
 * Inspect a sample row and guess how to read it. Returns null if nothing
 * recognizable is found (the wizard then asks the user to map fields).
 */
export function detectMapping(row: unknown): FieldMapping | null {
  if (!isRecord(row)) return null;

  const listField = firstKey(row, ["messages", "conversation", "conversations", "chat"]);
  if (listField && Array.isArray(row[listField])) {
    const item = (row[listField] as unknown[])[0];
    if (isRecord(item)) {
      const roleKey = firstKey(item, ["role", "from"]) ?? "role";
      const contentKey = firstKey(item, ["content", "value", "text"]) ?? "content";
      const kind = roleKey === "from" ? "conversations" : "messages";
      return { kind, messagesField: listField, roleKey, contentKey };
    }
  }

  const instructionField = firstKey(row, ["instruction", "prompt", "question"]);
  const outputField = firstKey(row, ["output", "response", "answer", "completion", "chosen"]);
  if (instructionField && outputField) {
    return {
      kind: "instruction",
      instructionField,
      inputField: firstKey(row, ["input", "context"]),
      outputField,
      systemField: firstKey(row, ["system", "system_prompt"]),
    };
  }

  const textField = firstKey(row, ["text", "content", "raw", "document"]);
  if (textField && typeof row[textField] === "string") return { kind: "text", textField };

  return null;
}

/** Extract a conversation from a row per the mapping; null if not conversational. */
export function rowToMessages(row: unknown, mapping: FieldMapping): ChatMessage[] | null {
  if (!isRecord(row)) return null;
  const push = (out: ChatMessage[], role: string, content: unknown) => {
    if (typeof content === "string" && content.length > 0) out.push({ role, content });
  };

  if (mapping.kind === "messages" || mapping.kind === "conversations") {
    const list = row[mapping.messagesField ?? "messages"];
    if (!Array.isArray(list)) return null;
    const roleKey = mapping.roleKey ?? (mapping.kind === "conversations" ? "from" : "role");
    const contentKey = mapping.contentKey ??
      (mapping.kind === "conversations" ? "value" : "content");
    const out: ChatMessage[] = [];
    for (const item of list) {
      if (!isRecord(item)) continue;
      const role = item[roleKey];
      if (typeof role !== "string") continue;
      push(out, normalizeRole(role), item[contentKey]);
    }
    return out.length ? out : null;
  }

  if (mapping.kind === "instruction") {
    const out: ChatMessage[] = [];
    if (mapping.systemField) push(out, "system", row[mapping.systemField]);
    const instr = row[mapping.instructionField ?? "instruction"];
    const extra = mapping.inputField ? row[mapping.inputField] : "";
    const user = typeof instr === "string"
      ? instr + (typeof extra === "string" && extra ? "\n\n" + extra : "")
      : "";
    push(out, "user", user);
    push(out, "assistant", row[mapping.outputField ?? "output"]);
    return out.length >= 2 ? out : null;
  }

  if (mapping.kind === "prompt") {
    const out: ChatMessage[] = [];
    push(out, "user", row[mapping.promptField ?? "prompt"]);
    push(out, "assistant", row[mapping.responseField ?? "response"]);
    return out.length >= 2 ? out : null;
  }

  return null;
}

/** Extract plain pretraining text from a row per the mapping; null if empty. */
export function rowToText(row: unknown, mapping: FieldMapping): string | null {
  if (!isRecord(row)) return null;
  const v = row[mapping.textField ?? "text"];
  return typeof v === "string" && v.length > 0 ? v : null;
}
