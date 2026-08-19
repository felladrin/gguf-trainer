// Undo Penn-Treebank tokenization damage in a plain-text corpus.
//
//   deno run -A scripts/detokenize-corpus.ts --in corpus/x.txt --out corpus/x.clean.txt
//
// Several public fiction corpora (euclaise/writingprompts among them) were
// tokenized and then imperfectly detokenized, so they ship `` and '' where the
// quotes belong, "could n't" for "couldn't", and a stray space on every line.
// Measured on a 30 MB slice of writingprompts: 12.9 PTB quotes and 5.9 split
// contractions per 1000 words, and a trailing space on 53 lines per 1000 words.
// A 94.7M model fine-tuned on that learns to emit it, so this runs first.
//
// Prose only: it strips leading indentation, which is damage in a detokenized
// story corpus and meaning in a code one.
//
//   deno run tests/detokenize-corpus.ts     # the rules, on fixtures

/** One line, damage undone. Every rule is idempotent, so a re-run is a no-op. */
export function detokenizeLine(line: string): string {
  return line
    .replace(/``\s*/g, '"') // PTB open quote, plus the space it always trails
    .replace(/\s*''/g, '"') // PTB close quote
    .replace(/ n't\b/g, "n't") // could n't -> couldn't
    .replace(/\b(gon|wan) na\b/g, "$1na")
    .replace(/\bgot ta\b/g, "gotta")
    .replace(/\(\s+/g, "(")
    .replace(/\s+\)/g, ")")
    .replace(/\[\s+/g, "[")
    .replace(/\s+\]/g, "]")
    .replace(/^[ \t]+|[ \t]+$/g, "");
}

/** Whole-text form: per-line rules, then collapse the blank runs that stripping
 * a space-only line leaves behind. */
export function detokenize(text: string): string {
  return text
    .split("\n")
    .map(detokenizeLine)
    .join("\n")
    .replace(/\n{3,}/g, "\n\n");
}

async function main(): Promise<void> {
  const argOf = (name: string, fallback = "") => {
    const eq = Deno.args.find((a) => a.startsWith(`--${name}=`));
    if (eq) return eq.slice(name.length + 3);
    const i = Deno.args.indexOf(`--${name}`);
    return i >= 0 && Deno.args[i + 1] ? Deno.args[i + 1] : fallback;
  };
  const inPath = argOf("in");
  const outPath = argOf("out");
  if (!inPath || !outPath) throw new Error("usage: --in <corpus.txt> --out <clean.txt>");

  const fs = await import("node:fs");
  const decoder = new TextDecoder();
  const enc = new TextEncoder();
  const fd = fs.openSync(outPath, "w");
  let carry = "", inBytes = 0, outBytes = 0, blankRun = 0, changed = 0;
  try {
    for await (const chunk of fs.createReadStream(inPath, { highWaterMark: 1 << 22 })) {
      inBytes += (chunk as Uint8Array).length;
      carry += decoder.decode(chunk as Uint8Array, { stream: true });
      const parts = carry.split("\n");
      carry = parts.pop() ?? ""; // a partial last line waits for the next chunk
      for (const raw of parts) {
        const line = detokenizeLine(raw);
        if (line !== raw) changed++;
        // Collapse the blank runs that stripping a space-only line leaves.
        if (line.length === 0) {
          if (++blankRun > 1) continue;
        } else {
          blankRun = 0;
        }
        const b = enc.encode(`${line}\n`);
        fs.writeSync(fd, b);
        outBytes += b.length;
      }
    }
    if (carry) {
      const b = enc.encode(detokenizeLine(carry));
      fs.writeSync(fd, b);
      outBytes += b.length;
    }
  } finally {
    fs.closeSync(fd);
  }

  console.log(
    `detokenize: ${(inBytes / 1e6).toFixed(1)} MB -> ${(outBytes / 1e6).toFixed(1)} MB, ` +
      `${changed} lines changed -> ${outPath}`,
  );
}

if (import.meta.main) await main();
