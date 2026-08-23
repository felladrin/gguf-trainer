"""Label hygiene over the RP battery outputs in out/rp-battery/.

Two counts per completion, both about turn structure rather than prose quality:
  self     the model re-labels its own turn ("Iris:" again inside its reply)
  other    the model writes a speaker that is neither the character nor "You"

Both are noise-dominated at one seed (pooled within-checkpoint SD ~1.7 against a
between-checkpoint range of ~1.7), so read the seeds/ aggregate, not a single file.

  python3 scripts/score-rp-battery.py            # single-seed files
  python3 scripts/score-rp-battery.py --seeds    # the 12-seed sweep, mean +- SEM
"""
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "out" / "rp-battery"

raw = subprocess.run(
    ["bash", str(ROOT / "scripts" / "dump-rp-prompts.sh")],
    cwd=ROOT, capture_output=True, text=True, check=True,
).stdout
PROMPTS = [p for p in raw.split("\0") if p]

LABEL = re.compile(r"^([A-Z][^\n:]{0,40}):", re.M)


def counts(path):
    """(self_relabel, other_speaker) summed over the nine prompts of one run."""
    blocks = path.read_text().split("##########")[1:]
    if len(blocks) != len(PROMPTS):
        raise SystemExit(f"{path.name}: {len(blocks)} blocks, expected {len(PROMPTS)}")
    self_n = other_n = 0
    for block, prompt in zip(blocks, PROMPTS):
        body = block.split("\n", 1)[1]
        i = body.find(prompt)
        if i < 0:
            raise SystemExit(f"{path.name}: prompt not echoed in block")
        completion = body[i + len(prompt):]
        char = prompt.partition("[Character: ")[2].partition("]")[0]
        for name in LABEL.findall(completion):
            if name == char:
                self_n += 1
            elif name != "You":
                other_n += 1
    return self_n, other_n


def step(path):
    return int(re.search(r"ckpt-(\d+)", path.name).group(1))


if "--seeds" in sys.argv:
    runs = {}
    for f in (BATTERY / "seeds").glob("ckpt-*-seed*.txt"):
        runs.setdefault(step(f), []).append(counts(f))
    print(f"{'step':>6}  {'n':>2}  {'self':>13}  {'other':>13}")
    for s in sorted(runs):
        rows = runs[s]
        cells = []
        for col in (0, 1):
            vals = [r[col] for r in rows]
            sem = stdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
            cells.append(f"{mean(vals):5.2f} +- {sem:4.2f}")
        print(f"{s:>6}  {len(rows):>2}  {cells[0]}  {cells[1]}")
else:
    print(f"{'step':>6}  {'self':>5}  {'other':>5}")
    for f in sorted(BATTERY.glob("ckpt-*.txt"), key=step):
        a, b = counts(f)
        print(f"{step(f):>6}  {a:>5}  {b:>5}")
