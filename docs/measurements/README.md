# Measurements

Logs that a `docs/optimization.md` lever quotes numbers from. Everything the trainer writes lands
in `out/`, which is gitignored, so a lever citing `out/whatever.log` is unverifiable to anyone who
was not at the machine. These are the copies that make the levers checkable.

Small and append-only. A file here should be the raw tool output, not a summary: the point is that
a reader can recompute the lever's arithmetic rather than trust the lever's arithmetic.

| File                        | What produced it                                                                                                                       |
| :-------------------------- | :------------------------------------------------------------------------------------------------------------------------------------- |
| `lambrp-holdout.log`        | `eval-loss --holdout 1 --windows 64 --seq-len 1024 --seed 1234` over every LittleLamb-293M-RP snapshot, the base, and the final export |
| `lambrp-holdout-quants.log` | the same command against the published Q8_0 and Q4_0 files                                                                             |
| `lambrp-train.log`          | the run's own header and per-step loss lines, from `finetune`                                                                          |
| `lambrp-battery.txt`        | `scripts/score-rp-battery.ts`, one battery run per checkpoint                                                                          |
| `lambrp-battery-seeds.txt`  | `scripts/score-rp-battery.ts --seeds`, 12 seeds per checkpoint over 216 runs                                                           |

The battery's raw completions are not here: 216 files of model output is too much to carry in a
repo, and the two tables above are what the lever reads. Regenerate them with
`scripts/eval-rp-completions.sh` if you need the text.
