// Parse downloaded dataset bytes into row objects. Supports the formats HF
// datasets actually ship in: Parquet (the auto-converted default), JSONL/JSON,
// plain text, and CSV/TSV. Parquet uses hyparquet (pure JS, runs under Deno).

import { parquetReadObjects } from "hyparquet";

export type Row = Record<string, unknown>;

function parseJSONL(text: string): Row[] {
  const out: Row[] = [];
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s) continue;
    try {
      const v = JSON.parse(s);
      if (v && typeof v === "object") out.push(v as Row);
    } catch { /* skip malformed line */ }
  }
  return out;
}

function parseJSON(text: string): Row[] {
  const v = JSON.parse(text);
  if (Array.isArray(v)) return v.filter((x) => x && typeof x === "object") as Row[];
  if (v && typeof v === "object") {
    // {data:[...]} or {rows:[...]} wrappers are common.
    for (const k of ["data", "rows", "examples"]) {
      if (Array.isArray((v as Row)[k])) return (v as Row)[k] as Row[];
    }
    return [v as Row];
  }
  return [];
}

function splitCSVLine(line: string, delim: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQ) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else inQ = false;
      } else cur += c;
    } else if (c === '"') inQ = true;
    else if (c === delim) {
      out.push(cur);
      cur = "";
    } else cur += c;
  }
  out.push(cur);
  return out;
}

function parseCSV(text: string, delim: string): Row[] {
  const lines = text.split(/\r?\n/).filter((l) => l.length > 0);
  if (lines.length < 2) return [];
  const header = splitCSVLine(lines[0], delim);
  const out: Row[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = splitCSVLine(lines[i], delim);
    const row: Row = {};
    header.forEach((h, j) => (row[h] = cells[j] ?? ""));
    out.push(row);
  }
  return out;
}

async function parseParquet(bytes: Uint8Array): Promise<Row[]> {
  // hyparquet reads via an AsyncBuffer ({ byteLength, slice }); wrap the in-memory
  // bytes. ArrayBuffer.slice is synchronous but a sync return satisfies the await.
  const ab = bytes.buffer.slice(
    bytes.byteOffset,
    bytes.byteOffset + bytes.byteLength,
  ) as ArrayBuffer;
  const rows = await parquetReadObjects({
    file: { byteLength: ab.byteLength, slice: (s: number, e?: number) => ab.slice(s, e) },
    utf8: true,
  });
  return rows as Row[];
}

/** Parse bytes by file extension. `.txt` becomes a single {text} row. */
export async function parseDataFile(name: string, bytes: Uint8Array): Promise<Row[]> {
  const ext = (name.match(/\.([a-z0-9]+)(?:\?|$)/i)?.[1] ?? "").toLowerCase();
  if (ext === "parquet") return await parseParquet(bytes);
  const text = new TextDecoder().decode(bytes);
  switch (ext) {
    case "jsonl":
      return parseJSONL(text);
    case "json":
      return parseJSON(text);
    case "csv":
      return parseCSV(text, ",");
    case "tsv":
      return parseCSV(text, "\t");
    case "txt":
      return [{ text }];
    default:
      // Best-effort: JSONL if it looks line-delimited, else one text row.
      return text.includes("\n") && text.trimStart().startsWith("{")
        ? parseJSONL(text)
        : [{ text }];
  }
}
