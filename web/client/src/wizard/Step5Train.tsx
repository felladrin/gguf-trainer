import { useEffect, useRef, useState } from "react";
import { listModels, startTraining, stopTraining, subscribeEvents } from "../api.ts";
import { buildConfig, type StepProps } from "./state.ts";
import { LossChart } from "../components/LossChart.tsx";
import type { TrainEvent } from "../../../shared/types.ts";

interface Live {
  status: string;
  points: { step: number; loss: number }[];
  step: number;
  steps: number;
  stepsPerSec: number;
  log: string[];
  sample?: string;
  terminal?: "done" | "error" | "stopped";
  message?: string;
}

const empty: Live = { status: "", points: [], step: 0, steps: 0, stepsPerSec: 0, log: [] };

export function Step5Train({ state, set, onDone }: StepProps & { onDone: () => void }) {
  const [live, setLive] = useState<Live>(empty);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState("");
  const closeRef = useRef<null | (() => void)>(null);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // If we return to this step with a job already running, re-attach: the server
    // replays event history, so the chart and log rebuild and live updates resume.
    if (state.jobId && !closeRef.current) {
      closeRef.current = subscribeEvents(state.jobId, onEvent);
    }
    return () => closeRef.current?.();
  }, []);
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [live.log]);

  async function start() {
    setStarting(true);
    setError("");
    setLive(empty);
    try {
      const { jobId } = await startTraining(buildConfig(state));
      set({ jobId });
      closeRef.current = subscribeEvents(jobId, onEvent);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }

  function onEvent(e: TrainEvent) {
    setLive((prev) => reduce(prev, e));
    // A checkpoint, a stop, or completion all leave a loadable GGUF: make it the
    // default to test/download and refresh the model list (for resume in step 1).
    const file = e.type === "checkpoint" || e.type === "done" || e.type === "stopped"
      ? e.file
      : undefined;
    if (file) {
      set({ doneFile: file });
      listModels().then((models) => set({ models })).catch(() => {});
    }
  }

  async function stop() {
    if (state.jobId) await stopTraining(state.jobId).catch(() => {});
  }

  const running = !!state.jobId && !live.terminal;
  const pct = live.steps ? Math.round((live.step / live.steps) * 100) : 0;

  return (
    <div className="card">
      <h2>Train</h2>
      <p className="hint">
        Runs on the local WebGPU engine. Keep this tab open; progress streams live.
      </p>

      {!state.jobId && (
        <button className="primary" onClick={start} disabled={starting}>
          {starting ? <span className="spinner" /> : "Start training"}
        </button>
      )}
      {error && <div className="err">{error}</div>}

      {state.jobId && (
        <>
          <div className="stats">
            <span className="stat">Status: <b>{live.terminal ?? live.status ?? "…"}</b></span>
            {live.steps > 0 && <span className="stat">Step: <b>{live.step}/{live.steps} ({pct}%)</b></span>}
            {live.stepsPerSec > 0 && <span className="stat">Speed: <b>{live.stepsPerSec.toFixed(2)} steps/s</b></span>}
            {live.points.length > 0 && (
              <span className="stat">Loss: <b>{live.points[live.points.length - 1].loss.toFixed(3)}</b></span>
            )}
          </div>

          <div style={{ marginTop: 12 }}>
            <LossChart points={live.points} />
          </div>

          <label>Log</label>
          <div className="log" ref={logRef}>{live.log.join("\n")}</div>

          {live.sample && (
            <>
              <label>Sample generation</label>
              <div className="preview-table"><pre>{live.sample}</pre></div>
            </>
          )}

          <div className="actions">
            {running
              ? <button className="danger" onClick={stop}>Stop &amp; save</button>
              : <span className="muted">
                {live.terminal === "done"
                  ? <span className="ok">Training complete.</span>
                  : live.terminal === "stopped"
                  ? <span className="ok">Stopped — latest checkpoint saved.</span>
                  : live.terminal === "error"
                  ? <span className="err">{live.message}</span>
                  : ""}
              </span>}
            <button
              className="primary"
              disabled={!state.doneFile}
              title={state.doneFile ? "" : "Available once the first checkpoint is saved"}
              onClick={onDone}
            >
              Test model →
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function reduce(s: Live, e: TrainEvent): Live {
  switch (e.type) {
    case "status":
      return { ...s, status: e.message, log: [...s.log, `• ${e.message}`] };
    case "corpus":
      return {
        ...s,
        log: [
          ...s.log,
          `corpus: ${e.tokens.toLocaleString()} tokens, vocab ${e.vocab}, ${e.docs} docs, ${
            e.epochs.toFixed(1)
          } epochs`,
        ],
      };
    case "model":
      return { ...s, log: [...s.log, `model: ${(e.params / 1e6).toFixed(1)}M params — ${e.summary}`] };
    case "step":
      return {
        ...s,
        step: e.step,
        steps: e.steps,
        stepsPerSec: e.stepsPerSec,
        points: [...s.points, { step: e.step, loss: e.loss }],
      };
    case "sample":
      return { ...s, sample: e.text };
    case "checkpoint":
      return {
        ...s,
        log: [...s.log, `checkpoint @ step ${e.step}: saved ${e.file} (${e.sizeMB.toFixed(1)} MB)`],
      };
    case "done":
      return {
        ...s,
        terminal: "done",
        status: "done",
        log: [...s.log, `done: ${e.file} (${e.sizeMB.toFixed(1)} MB, ${e.tensors} tensors)`],
      };
    case "stopped":
      return {
        ...s,
        terminal: "stopped",
        status: "stopped",
        log: [
          ...s.log,
          e.file
            ? `stopped: saved ${e.file} (${(e.sizeMB ?? 0).toFixed(1)} MB) — test or resume it`
            : "stopped (no checkpoint saved yet)",
        ],
      };
    case "error":
      return { ...s, terminal: "error", message: e.message, log: [...s.log, `! ${e.message}`] };
  }
}
