// Thin client for the Deno engine server (same origin in prod, Vite-proxied in dev).

import type { DatasetPreview, TrainConfig, TrainEvent } from "../../shared/types.ts";

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data?.error ?? `${r.status} ${r.statusText}`);
  return data as T;
}

export function previewDataset(
  url: string,
  opts?: { config?: string; split?: string; hfToken?: string },
): Promise<DatasetPreview> {
  return postJSON<DatasetPreview>("/api/dataset/preview", { url, ...opts });
}

export function startTraining(config: TrainConfig): Promise<{ jobId: string }> {
  return postJSON<{ jobId: string }>("/api/train/start", config);
}

export function stopTraining(jobId: string): Promise<{ stopped: boolean }> {
  return postJSON<{ stopped: boolean }>("/api/train/stop", { jobId });
}

export async function getConfig(): Promise<{ defaultChatTemplate: string }> {
  const r = await fetch("/api/config");
  return r.json();
}

export async function listModels(): Promise<{ file: string; sizeMB: number }[]> {
  const r = await fetch("/api/models");
  const data = await r.json();
  return data.models ?? [];
}

export function modelUrl(file: string): string {
  return `/api/models/${encodeURIComponent(file)}`;
}

/** Subscribe to a job's SSE stream. Returns a close function. */
export function subscribeEvents(
  jobId: string,
  onEvent: (e: TrainEvent) => void,
  onClose?: () => void,
): () => void {
  const es = new EventSource(`/api/train/events?job=${encodeURIComponent(jobId)}`);
  es.onmessage = (m) => {
    try {
      onEvent(JSON.parse(m.data) as TrainEvent);
    } catch { /* ignore malformed */ }
  };
  es.onerror = () => {
    es.close();
    onClose?.();
  };
  return () => es.close();
}
