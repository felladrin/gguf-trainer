import { useEffect, useMemo, useState } from "react";
import { getConfig, listModels } from "./api.ts";
import { DEFAULT_QWEN3_CHAT_TEMPLATE } from "../../../src/data/chat.ts";
import { defaultState, type State } from "./wizard/state.ts";
import { Step1Goal } from "./wizard/Step1Goal.tsx";
import { Step2Dataset } from "./wizard/Step2Dataset.tsx";
import { Step3Tokenizer } from "./wizard/Step3Tokenizer.tsx";
import { Step4Model } from "./wizard/Step4Model.tsx";
import { Step5Train } from "./wizard/Step5Train.tsx";
import { Step6Test } from "./wizard/Step6Test.tsx";

const STEPS = ["Goal", "Dataset", "Tokenizer", "Model", "Train", "Test"];

export function App() {
  const [state, setState] = useState<State>(() => defaultState(DEFAULT_QWEN3_CHAT_TEMPLATE));
  const set = (p: Partial<State>) => setState((s) => ({ ...s, ...p }));

  // Pull the canonical template + any existing models from the server on load.
  useEffect(() => {
    getConfig()
      .then((c) => c.defaultChatTemplate && set({ chatTemplate: c.defaultChatTemplate }))
      .catch(() => {/* keep the bundled default */});
    listModels().then((models) => set({ models })).catch(() => {});
  }, []);

  const canNext = useMemo(() => gate(state), [state]);

  const goto = (step: number) => set({ step });
  const next = () => set({ step: Math.min(6, state.step + 1) });
  const back = () => set({ step: Math.max(1, state.step - 1) });

  return (
    <div className="app">
      <h1>Felladrin's GGUF Trainer</h1>
      <p className="subtitle">
        Train a Qwen3 model from scratch, straight to GGUF, on your own GPU — then test it in the
        browser.
      </p>

      <div className="stepper">
        {STEPS.map((label, i) => {
          const n = i + 1;
          const cls = n === state.step ? "active" : n < state.step ? "done" : "";
          return <div key={label} className={`pill ${cls}`}>{n}. {label}</div>;
        })}
      </div>

      {state.step === 1 && <Step1Goal state={state} set={set} />}
      {state.step === 2 && <Step2Dataset state={state} set={set} />}
      {state.step === 3 && <Step3Tokenizer state={state} set={set} />}
      {state.step === 4 && <Step4Model state={state} set={set} />}
      {state.step === 5 && <Step5Train state={state} set={set} onDone={() => goto(6)} />}
      {state.step === 6 && <Step6Test state={state} set={set} />}

      {state.step < 5 && (
        <div className="actions">
          <button className="ghost" onClick={back} disabled={state.step === 1}>Back</button>
          <button className="primary" onClick={next} disabled={!canNext}>Continue</button>
        </div>
      )}
    </div>
  );
}

/** Whether the Continue button is enabled for the current step. */
function gate(s: State): boolean {
  switch (s.step) {
    case 1:
      return s.goal === "new" || (s.goal === "continue" && !!s.resumeFrom);
    case 2:
      return !!s.preview && !!s.mapping;
    case 3:
      return s.vocabSize >= 512 && (s.modelType === "base" || s.chatTemplate.trim().length > 0);
    case 4:
      return s.hidden % 128 === 0 && s.layers > 0 && s.steps > 0 && s.seqLen > 0 && s.batch > 0;
    default:
      return true;
  }
}
