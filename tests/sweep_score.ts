// Standalone assert check for sweep_sampler's pure scoring/ranking helpers (no server spawned;
// importing the module must not run main).
// Run:  deno run tests/sweep_score.ts
import {
  cfgKey,
  type CfgScore,
  distortion,
  neighbors,
  parseJudgeScore,
  POOL,
  rank,
  repScore,
  type SamplerCfg,
  samplerLine,
  seqRepN,
} from "../examples/sweep_sampler.ts";

function ok(cond: boolean, msg: string): void {
  if (!cond) throw new Error(msg);
}
function near(got: number, want: number, msg: string, tol = 1e-9): void {
  if (Math.abs(got - want) > tol) throw new Error(`${msg}: got ${got}, want ${want}`);
}

// seq-rep-n
near(seqRepN("the cat sat on the mat", 2), 0, "all bigrams unique");
near(seqRepN("fresh fresh fresh fresh fresh", 2), 0.75, "pure loop rep2");
near(seqRepN("Fresh, fresh! FRESH fresh.", 2), 1 - 1 / 3, "case/punct folded");
near(seqRepN("", 2), 1, "empty output is fully degenerate");
near(seqRepN("one two", 3), 0, "too short to have n-grams");
ok(
  repScore("fresh fresh fresh fresh fresh") > repScore("the cat sat on the mat"),
  "loops score worse than varied text",
);

// judge reply parsing
near(parseJudgeScore("7") ?? -1, 7, "bare number");
near(parseJudgeScore("Score: 8.5/10") ?? -1, 8.5, "number with prose");
near(parseJudgeScore("0") ?? -1, 0, "zero is a valid score");
ok(parseJudgeScore("ten") === null, "words are not scores");
ok(parseJudgeScore("42") === null, "out-of-range rejected");

// distortion
function cfg(name: string, rp: number, pp: number): SamplerCfg {
  return {
    name,
    temp: 0.7,
    topP: 1,
    topK: 0,
    minP: 0,
    presencePenalty: pp,
    repeatPenalty: rp,
    repeatLastN: 64,
  };
}
near(distortion(cfg("d", 1.15, 0.4)), 0.35, "distortion of the incumbent");
near(distortion(cfg("n", 1, 0)), 0, "no penalties, no distortion");

// rank, no judge: near-ties (<= eps) resolve toward the gentlest sampler
function score(c: SamplerCfg, rep: number, judge: number | null = null): CfgScore {
  return { cfg: c, rep, judge, judged: judge === null ? 0 : 1 };
}
const harsh = score(cfg("harsh", 1.3, 1.0), 0.285);
const gentle = score(cfg("gentle", 1, 0), 0.3);
const far = score(cfg("far", 1, 0), 0.4);
ok(
  rank([harsh, gentle, far]).map((r) => r.cfg.name).join(",") === "gentle,harsh,far",
  "near-tie prefers gentle; clear gaps rank by rep",
);

// rank, judged: judge score dominates, rep breaks ties
const jA = score(cfg("jA", 1, 0), 0.3, 8);
const jB = score(cfg("jB", 1, 0), 0.1, 5);
const jC = score(cfg("jC", 1, 0), 0.05, 8);
ok(
  rank([jA, jB, jC]).map((r) => r.cfg.name).join(",") === "jC,jA,jB",
  "judged ranking: judge desc, then rep",
);

// neighbors: clamped at the edges and deduped against configs already run
const g = cfg("g", 1, 0);
g.temp = 0;
const seenG = new Set([cfgKey(g)]);
ok(
  neighbors(g, seenG).map((c) => c.name).join(",") === "g~t-,g~pp+,g~rp+",
  "edge config keeps only the distinct, in-range neighbors",
);
const d = POOL.find((c) => c.name === "D");
ok(d !== undefined, "pool has the incumbent D");
const seenPool = new Set(POOL.map(cfgKey));
ok(neighbors(d as SamplerCfg, seenPool).length === 6, "D has 6 novel neighbors vs the pool");

// pool sanity + the incumbent line stays in sync with eval_completions.sh
ok(new Set(POOL.map((c) => c.name)).size === POOL.length, "pool names unique");
ok(new Set(POOL.map(cfgKey)).size === POOL.length, "pool configs unique");
ok(POOL.some((c) => c.name === "U"), "pool has the variety preset U");
ok(
  samplerLine(d as SamplerCfg) ===
    "SAMPLER=(--temp 0.7 --top-p 0.85 --top-k 30 --min-p 0.02 --presence-penalty 0.4 " +
      "--repeat-penalty 1.15 --repeat-last-n 128)",
  "D renders exactly the line eval_completions.sh documents",
);

console.log("sweep_score: all assertions passed");
