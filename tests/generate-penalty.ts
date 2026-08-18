// Standalone assert check for the repetition penalty on the trainer's own
// greedy decoder: the sign-dependent nudge, the penalty=1 identity, and the
// loop it exists to break.
// Run:  deno run tests/generate-penalty.ts
import { argmax, greedyComplete, penalizedArgmax, SAMPLE_PRESET } from "../src/eval/generate.ts";
import type { LanguageModel } from "../src/model/arch.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}

const row = (...v: number[]) => Float32Array.from(v);

ok(penalizedArgmax(row(3, 2.9, 1), [0], 1) === 0, "penalty 1 leaves argmax alone");
ok(argmax(row(3, 2.9, 1)) === 0, "unpenalized argmax picks the top logit");

// Positive logits are divided: 3 / 1.15 = 2.61 loses to an untouched 2.9.
ok(penalizedArgmax(row(3, 2.9, 1), [0], 1.15) === 1, "positive logit is divided down");

// Negative logits are multiplied, so the push is downward on that side too:
// -1 * 2 = -2 loses to an untouched -1.5.
ok(penalizedArgmax(row(-1, -1.5), [0], 2) === 1, "negative logit is multiplied down");

// A token outside the window keeps its logit.
ok(penalizedArgmax(row(3, 2.9, 1), [1], 1.15) === 0, "unseen token is untouched");

// A model that always prefers the token it just emitted: bare greedy repeats it
// forever, which is the "The world was in a state of chaos." x6 failure.
const V = 4;
const loopy = {
  cfg: { vocabSize: V, maxSeq: 64 },
  forward(ctx: number[]) {
    const data = new Float32Array(ctx.length * V);
    for (let t = 0; t < ctx.length; t++) data.set([0, 3, 2.9, 0.1], t * V);
    return { data };
  },
} as unknown as LanguageModel;

const bare = await greedyComplete(loopy, null, [0], 12);
ok(new Set(bare.slice(1)).size === 1, `bare greedy loops on one token, got ${bare.slice(1)}`);

const penalized = await greedyComplete(loopy, null, [0], 12, [], SAMPLE_PRESET);
ok(
  new Set(penalized.slice(1)).size > 1,
  `the preset breaks the loop, got ${penalized.slice(1)}`,
);

console.log("generate-penalty: all checks passed");
