// The raw-transcript half of a dual-format SFT corpus.
//
// A chat model reached through `/v1/chat/completions` sees ChatML; the same model
// reached through llama.cpp's `/completions` sees a plain persona block followed
// by "Name:" turns, which is what SillyTavern, Kobold Lite and AI Horde scribes
// actually send. Training only the first format leaves the second to whatever the
// base model happened to learn, and training the second WITHOUT a mask is how a
// model learns to write the human's turn as well as its own.
//
// So a transcript document here supervises exactly one reply: everything up to
// the final "Name:" is context, the reply that follows is the target, and an
// <|im_end|> terminator closes it so generation stops on EOS even when the client
// sets no stop string. The human's turns are never a target, at any offset, which
// is the whole point.
//
// Offsets are UTF-8 BYTE offsets, because that is the unit a byte-level BPE token
// maps onto (see BPETokenizer.byteLengths). Callers map them onto tokens with
// transcriptLossMask.

/** A rendered transcript plus the byte offset where its supervised reply starts. */
export interface TranscriptDoc {
  /** The document as the model sees it. The supervised reply is its tail. */
  text: string;
  /** UTF-8 byte offset of the first supervised byte; the span runs to the end. */
  replyStart: number;
}

export interface TranscriptTurn {
  human: boolean;
  text: string;
}

export interface TranscriptInput {
  /** The persona the model speaks as; becomes the label on its own turns. */
  character: string;
  /** The label on the human's turns. Varied by the caller so the format reads as
   * "some name, then a colon" rather than as the literal string "You:". */
  userLabel: string;
  persona?: string;
  /** Example dialogue from a character card, already normalized. */
  examples?: string;
  /** Oldest first. The last turn must be the character's: it is the target. */
  turns: TranscriptTurn[];
}

/** Control-token syntax must never reach the text: `<|im_end|>` inside a document
 * would encode as one special and decode to nothing, desynchronizing every byte
 * offset after it. */
const CONTROL_SYNTAX = /<\|[^>]*\|>/g;

function clean(s: string): string {
  return s.replaceAll(CONTROL_SYNTAX, " ").trim();
}

function utf8Len(s: string): number {
  return new TextEncoder().encode(s).length;
}

/**
 * Render one conversation as a horde-shaped transcript whose last character turn
 * is the target. Returns null when the input cannot form one (no name, or the
 * last turn is not the character's).
 *
 * The reply span deliberately starts AFTER the final "Name:" and includes the
 * leading space, because that is exactly where inference starts: a client's
 * prompt ends with the label and the model continues from there. It ends at the
 * end of the reply, with no trailing newline, so `/completions` returns the reply
 * text and nothing else.
 */
export function renderTranscript(input: TranscriptInput): TranscriptDoc | null {
  const character = clean(input.character);
  const userLabel = clean(input.userLabel) || "You";
  if (!character || character.length > 40) return null;

  const turns = input.turns
    .map((t) => ({ human: t.human, text: clean(t.text) }))
    .filter((t) => t.text.length > 0);
  if (turns.length < 2 || turns[turns.length - 1].human) return null;

  const head: string[] = [`[Character: ${character}]`];
  const persona = clean(input.persona ?? "");
  if (persona) head.push(`${character}'s Persona: ${persona}`);
  const examples = clean(input.examples ?? "");
  if (examples) head.push("<START>", examples);
  head.push("<START>");

  const lines = turns.map((t) => `${t.human ? userLabel : character}: ${t.text}`);
  const reply = turns[turns.length - 1].text;

  // Build the prefix and the reply separately, then join: the offset has to be
  // measured on the exact bytes that end up in `text`, not re-derived from a
  // search, which would find the wrong occurrence when a reply repeats earlier text.
  const prefix = [...head, ...lines.slice(0, -1), `${character}:`].join("\n");
  const text = `${prefix} ${reply}`;
  return { text, replyStart: utf8Len(prefix) };
}

/**
 * Loss mask over an encoded transcript: 1 for the tokens of the supervised reply
 * and for the appended terminator, 0 for the persona block, the example dialogue
 * and every human turn.
 *
 * `byteLens[i]` is how many bytes token i contributes to the decoded text, so the
 * running sum gives each token's byte range. A token is supervised only when it
 * STARTS at or after `replyStart`: one that straddles the boundary carries prompt
 * bytes, and supervising it would train the model on part of its own label. That
 * costs at most one token per document, and only when BPE fuses the colon to the
 * word after it.
 *
 * `ids` and `byteLens` must cover the terminator too; it is always supervised,
 * because "stop here" is the behaviour that keeps a raw completion from running
 * on into the human's turn.
 */
export function transcriptLossMask(
  byteLens: number[],
  replyStart: number,
  terminatorCount = 1,
): number[] {
  const sup = new Array<number>(byteLens.length).fill(0);
  const bodyEnd = byteLens.length - terminatorCount;
  let at = 0;
  for (let i = 0; i < bodyEnd; i++) {
    if (at >= replyStart) sup[i] = 1;
    at += byteLens[i];
  }
  for (let i = Math.max(0, bodyEnd); i < byteLens.length; i++) sup[i] = 1;
  return sup;
}
