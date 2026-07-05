import { useEffect, useRef, useState } from "react";
import type { Wllama } from "@wllama/wllama";
import { listModels, modelUrl } from "../api.ts";
import { chat, complete, loadModel } from "../lib/wllama.ts";
import type { StepProps } from "./state.ts";

interface Msg {
  role: "user" | "assistant";
  content: string;
}

export function Step6Test({ state, set }: StepProps) {
  const [file, setFile] = useState(state.doneFile ?? state.resumeFrom ?? "");
  const [loadPct, setLoadPct] = useState(-1);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Msg[]>([]);
  const [completion, setCompletion] = useState("");
  const wllamaRef = useRef<Wllama | null>(null);
  const isChat = state.modelType !== "base";

  useEffect(() => {
    listModels().then((models) => set({ models })).catch(() => {});
    return () => {
      wllamaRef.current?.exit().catch(() => {});
    };
  }, []);

  async function load() {
    if (!file) return;
    setError("");
    setReady(false);
    setLoadPct(0);
    try {
      await wllamaRef.current?.exit().catch(() => {});
      wllamaRef.current = await loadModel(modelUrl(file), (f) => setLoadPct(Math.round(f * 100)));
      setReady(true);
      setLoadPct(100);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setLoadPct(-1);
    }
  }

  async function send() {
    const w = wllamaRef.current;
    if (!w || !input.trim() || busy) return;
    setBusy(true);
    try {
      if (isChat) {
        const next: Msg[] = [...messages, { role: "user", content: input }];
        setMessages([...next, { role: "assistant", content: "" }]);
        setInput("");
        await chat(w, next, (partial) => {
          setMessages([...next, { role: "assistant", content: partial }]);
        });
      } else {
        setCompletion(input);
        await complete(w, input, (partial) => setCompletion(partial));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Test your model</h2>
      <p className="hint">
        Runs entirely in your browser via wllama (llama.cpp in WASM). No server inference.
      </p>

      <div className="row">
        <div style={{ flex: 3 }}>
          <label>Model file</label>
          <select value={file} onChange={(e) => setFile(e.target.value)}>
            <option value="">Select a model…</option>
            {state.models.map((m) => (
              <option key={m.file} value={m.file}>{m.file} ({m.sizeMB.toFixed(1)} MB)</option>
            ))}
          </select>
        </div>
        <div style={{ flex: 1, display: "flex", alignItems: "flex-end" }}>
          <button className="primary" onClick={load} disabled={!file || (loadPct >= 0 && !ready)}>
            {loadPct >= 0 && !ready ? `Loading ${loadPct}%` : "Load"}
          </button>
        </div>
      </div>

      {state.doneFile && (
        <p style={{ marginTop: 10 }}>
          <a href={modelUrl(state.doneFile)} download>Download {state.doneFile}</a>
        </p>
      )}

      {error && <div className="err">{error}</div>}

      {ready && (
        <div style={{ marginTop: 18 }}>
          {isChat
            ? (
              <>
                <div className="chat">
                  {messages.map((m, i) => (
                    <div key={i} className={`msg ${m.role}`}>{m.content || (busy ? "…" : "")}</div>
                  ))}
                </div>
                <div className="chat-input">
                  <input
                    type="text"
                    placeholder="Say something…"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && send()}
                  />
                  <button className="primary" onClick={send} disabled={busy || !input.trim()}>
                    {busy ? <span className="spinner" /> : "Send"}
                  </button>
                </div>
              </>
            )
            : (
              <>
                <label>Prompt</label>
                <input
                  type="text"
                  placeholder="Once upon a time"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && send()}
                />
                <button className="primary" style={{ marginTop: 10 }} onClick={send} disabled={busy}>
                  {busy ? <span className="spinner" /> : "Generate"}
                </button>
                {completion && <div className="preview-table" style={{ marginTop: 12 }}><pre>{completion}</pre></div>}
              </>
            )}
        </div>
      )}

      <div className="actions">
        <button className="ghost" onClick={() => set({ step: 1, jobId: undefined, doneFile: undefined })}>
          Train another
        </button>
        <span />
      </div>
    </div>
  );
}
