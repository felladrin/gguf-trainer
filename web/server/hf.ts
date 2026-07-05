// Hugging Face dataset resolution. Preview uses the Datasets Server JSON API
// (cheap, uniform across Parquet/JSONL/CSV); training downloads the actual data
// files (the auto-converted Parquet the Datasets Server exposes, or a direct
// file URL) and parses them locally — see parse.ts.

const DS = "https://datasets-server.huggingface.co";
const HUB = "https://huggingface.co";
const DATA_EXT = /\.(jsonl|json|txt|csv|parquet|tsv)(\?|$)/i;

export interface Resolved {
  kind: "repo" | "file";
  id: string; // "owner/name" for repo; the file URL for file
  url: string; // original
}

/** Parse a HF dataset URL (or "owner/name", or a direct file URL) into a target. */
export function resolveDataset(input: string): Resolved {
  const url = input.trim();
  if (DATA_EXT.test(url) && /^https?:\/\//i.test(url)) return { kind: "file", id: url, url };

  // https://huggingface.co/datasets/owner/name[/...]
  const m = url.match(/huggingface\.co\/datasets\/([^/\s]+\/[^/\s?#]+)/i);
  if (m) return { kind: "repo", id: decodeURIComponent(m[1]), url };

  // Bare "owner/name".
  if (/^[^/\s]+\/[^/\s]+$/.test(url)) return { kind: "repo", id: url, url };

  if (/^https?:\/\//i.test(url)) return { kind: "file", id: url, url };
  throw new Error(`Not a recognizable Hugging Face dataset URL: ${input}`);
}

export function hfHeaders(token?: string): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function getJSON(url: string, token?: string): Promise<unknown> {
  const r = await fetch(url, { headers: hfHeaders(token) });
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(
      `HF request failed ${r.status} ${r.statusText} for ${url}: ${body.slice(0, 300)}`,
    );
  }
  return r.json();
}

export interface SplitInfo {
  configs: string[];
  splits: string[]; // splits available for the default/first config
  byConfig: Record<string, string[]>;
}

/** List configs (subsets) and splits for a dataset. */
export async function fetchSplits(id: string, token?: string): Promise<SplitInfo> {
  const data = await getJSON(`${DS}/splits?dataset=${encodeURIComponent(id)}`, token) as {
    splits: { config: string; split: string }[];
  };
  const byConfig: Record<string, string[]> = {};
  for (const s of data.splits) (byConfig[s.config] ??= []).push(s.split);
  const configs = Object.keys(byConfig);
  return { configs, splits: byConfig[configs[0]] ?? [], byConfig };
}

export interface FirstRows {
  features: string[];
  rows: unknown[];
}

/** Up to 100 sample rows + column names, for the wizard preview. */
export async function fetchFirstRows(
  id: string,
  config: string,
  split: string,
  token?: string,
): Promise<FirstRows> {
  const q = `dataset=${encodeURIComponent(id)}&config=${encodeURIComponent(config)}&split=${
    encodeURIComponent(split)
  }`;
  const data = await getJSON(`${DS}/first-rows?${q}`, token) as {
    features: { name: string }[];
    rows: { row: unknown }[];
  };
  return {
    features: data.features.map((f) => f.name),
    rows: data.rows.map((r) => r.row),
  };
}

/** Direct Parquet file URLs for a config/split (HF's canonical raw data). */
export async function fetchParquetUrls(
  id: string,
  config: string,
  split: string,
  token?: string,
): Promise<string[]> {
  const data = await getJSON(`${DS}/parquet?dataset=${encodeURIComponent(id)}`, token) as {
    parquet_files: { config: string; split: string; url: string }[];
  };
  return data.parquet_files
    .filter((f) => f.config === config && f.split === split)
    .map((f) => f.url);
}

export interface RepoFile {
  path: string;
  size: number;
  url: string; // resolve URL to download the raw file
}

/**
 * Enumerate a dataset repo's file tree and return the original data files
 * (jsonl/json/csv/tsv/txt/parquet). This is the fallback for datasets the
 * Datasets Server never auto-converted to Parquet (unconvertible layouts, or
 * gated/private repos): list the repo, then download the source files directly.
 * When `split` is given, prefer files whose path names that split (train/…);
 * otherwise return every data file found. Order is smallest-first so a bounded
 * run pulls whole files rather than a fraction of one giant shard.
 */
export async function fetchRepoDataFiles(
  id: string,
  split?: string,
  token?: string,
): Promise<RepoFile[]> {
  const tree = await getJSON(
    `${HUB}/api/datasets/${id}/tree/main?recursive=true`,
    token,
  ) as { type: string; path: string; size?: number }[];
  const files = tree
    .filter((e) => e.type === "file" && DATA_EXT.test(e.path))
    .map((e) => ({
      path: e.path,
      size: e.size ?? 0,
      url: `${HUB}/datasets/${id}/resolve/main/${
        e.path.split("/").map(encodeURIComponent).join("/")
      }`,
    }))
    .sort((a, b) => a.size - b.size);
  if (split) {
    const named = files.filter((f) => new RegExp(`(^|[/_.-])${split}([/_.-]|$)`, "i").test(f.path));
    if (named.length) return named;
  }
  return files;
}

/** Optional total row count for a config/split (nice-to-have; may be absent). */
export async function fetchNumRows(
  id: string,
  config: string,
  split: string,
  token?: string,
): Promise<number | undefined> {
  try {
    const data = await getJSON(`${DS}/size?dataset=${encodeURIComponent(id)}`, token) as {
      size: { splits: { config: string; split: string; num_rows: number }[] };
    };
    return data.size.splits.find((s) => s.config === config && s.split === split)?.num_rows;
  } catch {
    return undefined;
  }
}
