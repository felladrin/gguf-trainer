import { useState } from "react";
import {
  fmtCount,
  fmtDuration,
  paramsFor,
  PRESETS,
  residentMB,
  stepSeconds,
} from "../lib/estimate.ts";
import type { QuantName } from "../../../shared/types.ts";
import type { StepProps } from "./state.ts";

export function Step4Model({ state, set }: StepProps) {
  const [advanced, setAdvanced] = useState(false);
  const params = paramsFor(state.vocabSize, state.hidden, state.layers, state.maxSeq);
  const totalTokens = state.steps * state.batch * state.seqLen;
  const totalTime = params ? state.steps * stepSeconds(params, state.seqLen, state.batch) : NaN;
  const hiddenValid = state.hidden % 128 === 0;

  return (
    <div className="card">
      <h2>Model size &amp; training</h2>
      <p className="hint">Pick a size, then tune if you like. Bigger = slower but more capable.</p>

      {state.goal === "continue" && (
        <div className="warnbox">
          Resuming: architecture comes from the checkpoint. Only the training settings below apply.
        </div>
      )}

      <label>Size preset</label>
      <div className="cards">
        {PRESETS.map((pr) => {
          const pc = paramsFor(state.vocabSize, pr.hidden, pr.layers, state.maxSeq);
          const sel = state.hidden === pr.hidden && state.layers === pr.layers;
          return (
            <div
              key={pr.key}
              className={`selectable ${sel ? "sel" : ""}`}
              onClick={() => set({ hidden: pr.hidden, layers: pr.layers })}
            >
              <div className="t">{pr.label}</div>
              <div className="d">
                hidden {pr.hidden} · {pr.layers} layers · ~{pc ? fmtCount(pc) : "?"} params
              </div>
            </div>
          );
        })}
      </div>

      <div className="row" style={{ marginTop: 12 }}>
        <Num label="Steps" value={state.steps} onChange={(v) => set({ steps: v })} />
        <Num label="Sequence length" value={state.seqLen} onChange={(v) => set({ seqLen: v })} />
        <Num label="Batch" value={state.batch} onChange={(v) => set({ batch: v })} />
      </div>

      {state.modelType !== "base" && (
        <label className="checkrow" style={{ marginTop: 12 }}>
          <input
            type="checkbox"
            checked={state.maskPromptLoss}
            onChange={(e) => set({ maskPromptLoss: e.target.checked })}
          />
          <span>
            Train on assistant turns only{" "}
            <span className="muted">
              (mask the system/user prompt from the loss — the standard way to fine-tune instruct
              models)
            </span>
          </span>
        </label>
      )}

      <div style={{ marginTop: 12 }}>
        <button className="ghost" onClick={() => setAdvanced(!advanced)}>
          {advanced ? "Hide" : "Show"} advanced settings
        </button>
      </div>

      {advanced && (
        <>
          <div className="row" style={{ marginTop: 12 }}>
            <Num label="Hidden size (×128)" value={state.hidden} onChange={(v) => set({ hidden: v })} />
            <Num label="Layers" value={state.layers} onChange={(v) => set({ layers: v })} />
            <Num label="Context length" value={state.maxSeq} onChange={(v) => set({ maxSeq: v })} />
          </div>
          <div className="row">
            <Num label="Muon LR" value={state.muonLr} step={0.001} onChange={(v) => set({ muonLr: v })} />
            <Num label="Aux LR (AdamW)" value={state.auxLr} step={0.0001} onChange={(v) => set({ auxLr: v })} />
            <Num label="muP base width" value={state.baseWidth} onChange={(v) => set({ baseWidth: v })} />
          </div>
          <div className="row">
            <div>
              <label>Export quantization</label>
              <select value={state.quant} onChange={(e) => set({ quant: e.target.value as QuantName })}>
                <option value="f16">F16 (best for resuming)</option>
                <option value="q8_0">Q8_0</option>
                <option value="q4_0">Q4_0 (smallest)</option>
              </select>
            </div>
            <Num label="Max rows (0 = auto)" value={state.maxRows} onChange={(v) => set({ maxRows: v })} />
            <div />
          </div>
        </>
      )}

      {!hiddenValid && <div className="err">Hidden size must be a multiple of 128.</div>}

      <div className="stats">
        <span className="stat">Params: <b>{params ? fmtCount(params) : "invalid"}</b></span>
        <span className="stat">Resident state: <b>{params ? `${residentMB(params).toFixed(0)} MB` : "—"}</b></span>
        <span className="stat">Tokens/run: <b>{fmtCount(totalTokens)}</b></span>
        <span className="stat">Est. time: <b>{fmtDuration(totalTime)}</b></span>
        <div className="muted" style={{ marginTop: 6, fontSize: 12 }}>
          Time is a rough M1-Max ballpark; your GPU will differ. Memory is far from the limit — the
          32 GB unified pool holds these easily.
        </div>
      </div>
    </div>
  );
}

function Num(
  { label, value, onChange, step }: {
    label: string;
    value: number;
    onChange: (v: number) => void;
    step?: number;
  },
) {
  return (
    <div>
      <label>{label}</label>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  );
}
