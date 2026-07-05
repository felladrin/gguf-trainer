import { MODEL_TYPES } from "../lib/datasets.ts";
import type { ModelType } from "../../../shared/types.ts";
import type { StepProps } from "./state.ts";

export function Step1Goal({ state, set }: StepProps) {
  return (
    <div className="card">
      <h2>What are we building?</h2>
      <p className="hint">Start a new model, or continue training one you exported earlier.</p>

      <div className="cards">
        <div
          className={`selectable ${state.goal === "new" ? "sel" : ""}`}
          onClick={() => set({ goal: "new" })}
        >
          <div className="t">New model</div>
          <div className="d">Train from scratch on a dataset you choose.</div>
        </div>
        <div
          className={`selectable ${state.goal === "continue" ? "sel" : ""}`}
          onClick={() => set({ goal: "continue" })}
        >
          <div className="t">Continue training</div>
          <div className="d">Resume from a GGUF checkpoint (weights + vocab reused).</div>
        </div>
      </div>

      {state.goal === "continue" && (
        <>
          <label>Checkpoint</label>
          {state.models.length === 0
            ? <p className="muted">No models found yet. Train one first, then it appears here.</p>
            : (
              <select
                value={state.resumeFrom}
                onChange={(e) => set({ resumeFrom: e.target.value })}
              >
                <option value="">Select a checkpoint…</option>
                {state.models.map((m) => (
                  <option key={m.file} value={m.file}>
                    {m.file} ({m.sizeMB.toFixed(1)} MB)
                  </option>
                ))}
              </select>
            )}
          <div className="warnbox">
            On resume the model size and vocabulary come from the checkpoint; the Tokenizer and Model
            steps are informational. Optimizer momentum restarts fresh.
          </div>
        </>
      )}

      <label>Model type</label>
      <p className="hint" style={{ margin: "0 0 10px" }}>
        Type sets the training format and the recommended datasets. It does not change the export
        format — every model exports as a llama.cpp-loadable Qwen3 GGUF.
      </p>
      <div className="cards">
        {MODEL_TYPES.map((t) => (
          <div
            key={t.key}
            className={`selectable ${state.modelType === t.key ? "sel" : ""}`}
            onClick={() => set({ modelType: t.key as ModelType })}
          >
            <div className="t">{t.label}</div>
            <div className="d">{t.blurb}</div>
          </div>
        ))}
      </div>

      <label>Model name</label>
      <input
        type="text"
        placeholder={`qwen3-${state.modelType}`}
        value={state.name}
        onChange={(e) => set({ name: e.target.value })}
      />

      <div className="warnbox">
        Reality check: at the sizes WebGPU-from-scratch reaches (a few million to ~50M params), a
        chat/reasoning/tool model learns the <em>format</em> but not strong capability. Great for
        learning the pipeline; not a substitute for a large pretrained model.
      </div>
    </div>
  );
}
