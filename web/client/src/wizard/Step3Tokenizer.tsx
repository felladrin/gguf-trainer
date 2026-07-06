import { DEFAULT_CHAT_TEMPLATE } from "../../../../src/data/chat.ts";
import type { StepProps } from "./state.ts";

export function Step3Tokenizer({ state, set }: StepProps) {
  const isChat = state.modelType !== "base";
  return (
    <div className="card">
      <h2>Tokenizer &amp; chat template</h2>
      <p className="hint">
        A byte-level BPE vocabulary is trained on your corpus.{" "}
        {isChat && "ChatML control tokens are added automatically so turns tokenize atomically."}
      </p>

      {state.goal === "continue" && (
        <div className="warnbox">
          Resuming a checkpoint: its vocabulary is reused as-is. These settings are ignored.
        </div>
      )}

      <label>Vocabulary size</label>
      <div className="row">
        <select value={state.vocabSize} onChange={(e) => set({ vocabSize: Number(e.target.value) })}>
          <option value={4096}>4,096 (small, fastest)</option>
          <option value={8192}>8,192 (recommended)</option>
          <option value={16384}>16,384</option>
          <option value={32000}>32,000 (large)</option>
        </select>
      </div>

      {isChat && (
        <>
          <label>Chat template (Jinja)</label>
          <p className="hint" style={{ margin: "0 0 6px" }}>
            Embedded into the GGUF as <code>tokenizer.chat_template</code>, and used to format
            training data. Default is a ChatML template (from Qwen3-Instruct).
          </p>
          <textarea
            value={state.chatTemplate}
            onChange={(e) => set({ chatTemplate: e.target.value })}
            spellCheck={false}
            style={{ minHeight: 260 }}
          />
          <div style={{ marginTop: 8 }}>
            <button
              className="ghost"
              onClick={() => set({ chatTemplate: DEFAULT_CHAT_TEMPLATE })}
            >
              Reset to default
            </button>
          </div>
          {state.modelType === "reasoning" && (
            <div className="warnbox">
              For a reasoning model, your dataset's assistant turns should contain the
              &lt;think&gt;…&lt;/think&gt; traces — the template renders them as-is.
            </div>
          )}
          {state.modelType === "tools" && (
            <div className="warnbox">
              For tool-calling, use a dataset whose messages include tool_calls / tool responses; the
              template renders &lt;tool_call&gt; and &lt;tool_response&gt; blocks.
            </div>
          )}
        </>
      )}
    </div>
  );
}
