import { useState } from "react";
import { Template } from "@huggingface/jinja";
import { previewDataset } from "../api.ts";
import { RECOMMENDED } from "../lib/datasets.ts";
import { rowToMessages, rowToText } from "../../../../src/data/chat.ts";
import type { FieldMapping } from "../../../shared/types.ts";
import type { StepProps } from "./state.ts";

export function Step2Dataset({ state, set }: StepProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function loadPreview(url: string, config?: string, split?: string) {
    setLoading(true);
    setError("");
    try {
      const preview = await previewDataset(url, { config, split, hfToken: state.hfToken || undefined });
      set({
        datasetUrl: url,
        preview,
        mapping: preview.detected ?? { kind: "text", textField: "text" },
        dsConfig: preview.config || undefined,
        dsSplit: preview.split || undefined,
      });
    } catch (e) {
      set({ preview: undefined });
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  const p = state.preview;
  const suggestions = RECOMMENDED[state.modelType];

  return (
    <div className="card">
      <h2>Choose a dataset</h2>
      <p className="hint">
        Paste any Hugging Face dataset URL (or a direct .jsonl/.parquet/.txt file URL). Preview is
        fetched via the HF Datasets Server; training downloads the raw data.
      </p>

      <div className="row">
        <input
          type="text"
          placeholder="e.g. roneneldan/TinyStories"
          value={state.datasetUrl}
          onChange={(e) => set({ datasetUrl: e.target.value })}
          onKeyDown={(e) => e.key === "Enter" && state.datasetUrl && loadPreview(state.datasetUrl)}
          style={{ flex: 3 }}
        />
        <button
          className="primary"
          style={{ flex: 0, minWidth: 110 }}
          disabled={!state.datasetUrl || loading}
          onClick={() => loadPreview(state.datasetUrl)}
        >
          {loading ? <span className="spinner" /> : "Preview"}
        </button>
      </div>

      <div className="chips">
        {suggestions.map((s) => (
          <div key={s.url} className="chip" onClick={() => loadPreview(s.url)} title={s.note}>
            {s.label}
            <small>{s.url}</small>
          </div>
        ))}
      </div>

      <label>HF token (optional, for gated/private datasets)</label>
      <input
        type="password"
        placeholder="hf_..."
        value={state.hfToken}
        onChange={(e) => set({ hfToken: e.target.value })}
      />

      {error && <div className="err">{error}</div>}

      {p && (
        <>
          {p.configs.length > 0 && (
            <div className="row" style={{ marginTop: 16 }}>
              <div>
                <label>Subset (config)</label>
                <select
                  value={p.config}
                  onChange={(e) => loadPreview(state.datasetUrl, e.target.value, undefined)}
                >
                  {p.configs.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label>Split</label>
                <select
                  value={p.split}
                  onChange={(e) => loadPreview(state.datasetUrl, p.config, e.target.value)}
                >
                  {p.splits.map((s) => <option key={s} value={s}>{s}</option>)}
                </select>
              </div>
            </div>
          )}

          <div className="stats">
            <span className="stat">Columns: <b>{p.features.join(", ") || "—"}</b></span>
            {p.numRowsTotal != null && (
              <span className="stat">Rows: <b>{p.numRowsTotal.toLocaleString()}</b></span>
            )}
            {p.directFile && <span className="stat">(direct file)</span>}
          </div>

          <MappingEditor
            mapping={state.mapping!}
            features={p.features}
            modelType={state.modelType}
            onChange={(m) => set({ mapping: m })}
          />

          <label>Sample (as the model will see it)</label>
          <div className="preview-table">
            <pre>{renderSample(state)}</pre>
          </div>
        </>
      )}
    </div>
  );
}

function renderSample(state: StepProps["state"]): string {
  const p = state.preview;
  if (!p || !state.mapping || p.rows.length === 0) return "(no preview)";
  const row = p.rows[0];
  if (state.modelType === "base") {
    return (rowToText(row, state.mapping) ?? "(no text found with this mapping)").slice(0, 2000);
  }
  const msgs = rowToMessages(row, state.mapping);
  if (!msgs || msgs.length === 0) return "(could not read a conversation — adjust the mapping)";
  try {
    return new Template(state.chatTemplate).render({ messages: msgs, add_generation_prompt: false })
      .slice(0, 2000);
  } catch (e) {
    return "Template error: " + (e instanceof Error ? e.message : String(e));
  }
}

function MappingEditor(
  { mapping, features, modelType, onChange }: {
    mapping: FieldMapping;
    features: string[];
    modelType: string;
    onChange: (m: FieldMapping) => void;
  },
) {
  const upd = (p: Partial<FieldMapping>) => onChange({ ...mapping, ...p });
  const kinds = modelType === "base"
    ? ["text"]
    : ["messages", "conversations", "instruction", "prompt", "text"];

  return (
    <div style={{ marginTop: 12 }}>
      <label>How to read the columns</label>
      <div className="row">
        <div>
          <select value={mapping.kind} onChange={(e) => upd({ kind: e.target.value as FieldMapping["kind"] })}>
            {kinds.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </div>
        {mapping.kind === "text" && (
          <FieldSelect label="text field" value={mapping.textField} features={features} onChange={(v) => upd({ textField: v })} />
        )}
        {(mapping.kind === "messages" || mapping.kind === "conversations") && (
          <>
            <FieldSelect label="list field" value={mapping.messagesField} features={features} onChange={(v) => upd({ messagesField: v })} />
            <FieldText label="role key" value={mapping.roleKey ?? ""} onChange={(v) => upd({ roleKey: v })} />
            <FieldText label="content key" value={mapping.contentKey ?? ""} onChange={(v) => upd({ contentKey: v })} />
          </>
        )}
        {mapping.kind === "instruction" && (
          <>
            <FieldSelect label="instruction" value={mapping.instructionField} features={features} onChange={(v) => upd({ instructionField: v })} />
            <FieldSelect label="output" value={mapping.outputField} features={features} onChange={(v) => upd({ outputField: v })} />
          </>
        )}
        {mapping.kind === "prompt" && (
          <>
            <FieldSelect label="prompt" value={mapping.promptField} features={features} onChange={(v) => upd({ promptField: v })} />
            <FieldSelect label="response" value={mapping.responseField} features={features} onChange={(v) => upd({ responseField: v })} />
          </>
        )}
      </div>
    </div>
  );
}

function FieldSelect(
  { label, value, features, onChange }: {
    label: string;
    value?: string;
    features: string[];
    onChange: (v: string) => void;
  },
) {
  return (
    <div>
      <label>{label}</label>
      <select value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
        <option value="">—</option>
        {features.map((f) => <option key={f} value={f}>{f}</option>)}
      </select>
    </div>
  );
}

function FieldText(
  { label, value, onChange }: { label: string; value: string; onChange: (v: string) => void },
) {
  return (
    <div>
      <label>{label}</label>
      <input type="text" value={value} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}
