// In-process job registry + SSE fan-out. One training job runs at a time (a
// single GPU), so this stays deliberately small: a map of jobs, each buffering
// its event history for late subscribers and holding a cooperative stop flag.

import type { TrainConfig, TrainEvent } from "../shared/types.ts";

export class StopSignal extends Error {
  constructor() {
    super("stopped by user");
    this.name = "StopSignal";
  }
}

export interface Job {
  id: string;
  config: TrainConfig;
  status: "running" | "done" | "error" | "stopped";
  events: TrainEvent[];
  subscribers: Set<(e: TrainEvent) => void>;
  stopRequested: boolean;
  file?: string;
  createdAt: number;
}

const jobs = new Map<string, Job>();

let counter = 0;
function newId(): string {
  counter += 1;
  // Time-free id (Date.now is fine here — not inside a workflow), still unique.
  return `job-${Date.now().toString(36)}-${counter}`;
}

export function createJob(config: TrainConfig): Job {
  const job: Job = {
    id: newId(),
    config,
    status: "running",
    events: [],
    subscribers: new Set(),
    stopRequested: false,
    createdAt: Date.now(),
  };
  jobs.set(job.id, job);
  return job;
}

export function getJob(id: string): Job | undefined {
  return jobs.get(id);
}

export function activeJob(): Job | undefined {
  for (const j of jobs.values()) if (j.status === "running") return j;
  return undefined;
}

export function emit(job: Job, event: TrainEvent): void {
  job.events.push(event);
  for (const cb of job.subscribers) {
    try {
      cb(event);
    } catch { /* a dead subscriber must not break the run */ }
  }
}

export function requestStop(id: string): boolean {
  const job = jobs.get(id);
  if (!job || job.status !== "running") return false;
  job.stopRequested = true;
  return true;
}

/** Subscribe to a job's events; replays history first. Returns an unsubscribe fn. */
export function subscribe(job: Job, cb: (e: TrainEvent) => void): () => void {
  for (const e of job.events) cb(e);
  if (job.status === "running") {
    job.subscribers.add(cb);
    return () => job.subscribers.delete(cb);
  }
  return () => {};
}
