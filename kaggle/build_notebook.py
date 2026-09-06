#!/usr/bin/env python3
r"""Generate ``notebook/toysvg_kaggle.ipynb``.

The notebook is a build artifact rather than a hand-edited file: keeping the cells
in source form means it can be regenerated, reviewed in a diff, and syntax-checked.
``--check`` compiles every code cell (with magics stripped), which catches the
classic failure of a notebook that only breaks once Kaggle runs it.

This notebook is the *reader*: it explains the score, verifies the harness, and
replays a stored run with no model call. The Kaggle **Benchmarks** tasks live in
``toysvg_task_file.py`` and ``toysvg_control_task_file.py``, which are pushed
with ``kaggle b t push``; keeping the two separate matters because the CLI
converts a task file into the notebook that backs the task, and a task file must
call ``.run()``/``.evaluate()`` to record anything.

    python -m kaggle.build_notebook            # writes the notebook
    python -m kaggle.build_notebook --check    # check only, write nothing
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

Cell = Tuple[str, str]

TITLE = r'''# ToySVG: a deterministic figure-copying benchmark

Give a vision model a black-and-white outline it has never seen, hide the original
geometry and every name that could be recalled, and ask for SVG path data that
reproduces it. Score the result geometrically.

There are two ways to run this, and they are deliberately separate:

| artifact | what it is for | how it runs |
|---|---|---|
| `toysvg_task_file.py`, `toysvg_control_task_file.py` | the Kaggle **Benchmarks** tasks | `kaggle b t push` / `kaggle b t run`: models come from Kaggle's panel, no API key of your own |
| this notebook | reading, checking, and replaying results | nothing to push; runs anywhere, including with no network |

This notebook has two modes, set in the first cell:

- **`replay`**: no model calls. Scores a stored run through the same code path a
  live run uses, and redraws the specification's figures. If you want to check the
  benchmark rather than run it, this is the mode that matters.
- **`own-key`**: calls your own provider (Gemini, say) with a key from Kaggle
  Secrets, using the harness's own runner.

Whatever mode you use, two properties decide how you read any number below:

- **The score is deterministic.** Region overlap, boundary agreement and topology
  are computed against a hidden ground-truth path, after fitting translation and
  scale away. Identical inputs yield identical outputs via deterministic arithmetic without subjective evaluation.
- **Every score ships with the range it came from**: a floor from shape-blind
  answers, and a ceiling from a pixel tracer on the same image at the same budget.
  Without those two a score has no scale.'''

CONFIG = r'''# ==== switch ==============================================================
# "replay"   offline verification mode: score the stored run, redraw the figures
# "own-key"  call your own provider with a key from Kaggle Secrets
MODE = "replay"

# Reasoning models spend the output budget on hidden tokens and come back with a
# path cut off mid-command, which measures the token cap rather than perception.
REASONING = "none"

# Limit the item set for a quick pass, e.g. ["01001100afd9128f"]. None = all items.
ITEMS = None

# Where per-item decompositions are appended, so a score that looks wrong can be
# taken apart afterwards without re-running anything.
FINDINGS = "toysvg-decompositions.jsonl"

print(f"MODE = {MODE}")'''

SETUP = r'''import json
import os
import sys
from pathlib import Path

def find_root():
    """Locate the harness: attached dataset, extracted bundle, or this checkout."""
    override = os.environ.get("TOYSVG_ROOT")
    if override and (Path(override) / "code" / "toysvg_task.py").exists():
        return Path(override)
    for base in (Path.cwd(), Path("/kaggle/input"), Path(".")):
        if not base.exists():
            continue
        if (base / "code" / "toysvg_task.py").exists():
            return base
        for hit in base.glob("*/code/toysvg_task.py"):
            return hit.parents[2]
        for hit in base.glob("*/*/code/toysvg_task.py"):
            return hit.parents[3]
    for hit in Path(".").glob("kaggle/toysvg_task.py"):
        return hit.parents[1]
    raise SystemExit("attach the ToySVG dataset, or set TOYSVG_ROOT")

ROOT = find_root()
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "kaggle"))
PACK_DIR = ROOT / "data" / "pack"
GT_DIR = ROOT / "data" / "gt"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

import numpy, scipy, skimage, PIL
print("harness  ", ROOT)
print("pack     ", PACK_DIR)
print("versions ", f"numpy {numpy.__version__}, scipy {scipy.__version__}, "
                  f"scikit-image {skimage.__version__}, Pillow {PIL.__version__}")

# The rasteriser ships with the harness, so scores cannot drift with a rendering
# library's version. The boundary term and the tracer do call into scipy and
# scikit-image, where a version change can move a score by a hair: report any
# mismatch rather than discovering it in a re-run.
pins = {}
req = ROOT / "code" / "requirements.txt"
if not req.exists():
    req = Path("requirements.txt")
if req.exists():
    for line in req.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and ">=" in line:
            name, _, want = line.partition(">=")
            pins[name.strip()] = want.strip()
have = {"numpy": numpy.__version__, "scipy": scipy.__version__,
        "scikit-image": skimage.__version__, "Pillow": PIL.__version__}
for name, want in pins.items():
    got = have.get(name)
    if got and got.split(".")[0] != want.split(".")[0]:
        print(f"  version drift: {name} {got} vs pinned >={want}")'''

PACK = r'''import toysvg_task as ft

# ---- the blind half stays blind -------------------------------------------
# The corpus ships in two halves: pack/ is what a model may see, gt/ is the hidden
# geometry. Re-assert that split here, before spending a model call.
FORBIDDEN = {"gt_d", "gt_fill_rule", "source_sha256", "stroke_median_px",
             "ink_area_px", "topology", "n_commands_gt", "n_subpaths_gt"}
ALLOWED = {"item_id", "budget", "k", "canvas", "input_png", "input_sha256"}

def keys_in(blob):
    out = set()
    if isinstance(blob, dict):
        for k, v in blob.items():
            out.add(str(k)); out |= keys_in(v)
    elif isinstance(blob, list):
        for v in blob:
            out |= keys_in(v)
    return out

leaks = []
for path in sorted(PACK_DIR.rglob("*.json")):
    leaked = keys_in(json.loads(path.read_text())) & FORBIDDEN
    if leaked:
        leaks.append((str(path), sorted(leaked)))
tasks = ft.public_tasks(PACK_DIR)
for iid, rec in tasks.items():
    extra = set(rec) - ALLOWED
    if extra:
        leaks.append((iid, sorted(extra)))
assert not leaks, f"model-facing half carries hidden fields: {leaks}"

rows = []
for iid in sorted(tasks):
    rec = tasks[iid]
    hidden = json.loads((GT_DIR / f"{iid}.json").read_text())
    rows.append({"item_id": iid, "stratum": hidden["stratum"], "budget": rec["budget"],
                 "k": rec["k"], "png_path": str(PACK_DIR / rec["input_png"])})
if ITEMS:
    rows = [r for r in rows if r["item_id"] in set(ITEMS)]

print(f"{len(rows)} items, blind half verified: no name and no geometry in pack/")'''

SELFTEST = r'''# ---- instrument self-test, before any model call --------------------------
# Two things must hold before any model number means anything:
#   1. reflexivity: the hidden path scored against itself is exactly 1.000
#   2. a blank answer scores 0, through the gate, not by accident
# Reflexivity is a property of the geometry, so it is read from the identity
# floor. Feeding the hidden path in as if it were an answer asks a second
# question: does the ground truth fit the command budget? And some items
# were captured from source icons that need more commands than their tier allows.
ft.configure(pack_dir=PACK_DIR, findings_path=None, reasoning=REASONING)

checks = ft.budget_report(PACK_DIR)
assert all(c["identity_floor"] > 0.999 for c in checks), \
    "the scorer is not reflexive: fix the harness before scoring models"
print(f"reflexivity: {len(checks)}/{len(checks)} items return 1.000000 "
      f"against their own ground truth")

blank = ft.score_answer("", rows[0]["item_id"])
assert blank["score"] == 0.0 and blank["gate"] == "empty"
print(f"blank answer: score 0.0000, gate {blank['gate']!r}")

over = [c for c in checks if c["gt_over_budget"]]
print(f"\nground truth over budget in {len(over)}/{len(checks)} items")
for c in over:
    print(f"  {c['item_id'][:8]}: source path {c['gt_commands']} commands > k={c['k']}, "
          f"so its geometry scores {c['gt_partial_score']} but the gate is "
          f"{c['gt_gate']!r}; its reference is the ceiling C(k)={c['ceiling_score']:.3f}, "
          f"not the identity")
for c in checks:
    print(f"  {c['item_id'][:8]}  k={c['k']:<3} source {c['gt_commands']:>3} cmds  "
          f"ceiling {c['ceiling_score']:.3f}")'''

MD_KBENCH = r'''## Running it on Kaggle Benchmarks

The benchmark runs on Kaggle's infrastructure, against Kaggle's model panel. No key
of your own is involved, which is what makes the comparison cheap and identical
across models.

```bash
kaggle b init -y                      # once per environment: writes MODEL_PROXY_* to .env

# validate locally before pushing; a task file that never calls
# .run()/.evaluate() is accepted and then records nothing
python toysvg_task_file.py

kaggle b t push toysvg-redraw-from-png \
    -f toysvg_task_file.py -d <you>/toysvg-benchmark-harness --wait
kaggle b t push toysvg-structure-questions \
    -f toysvg_control_task_file.py -d <you>/toysvg-benchmark-harness --wait

kaggle b t models                     # valid model slugs
kaggle b t run toysvg-redraw-from-png -m gemini-2.5-pro --wait
kaggle b t run toysvg-redraw-from-png -m gemini-2.5-pro -m claude-sonnet-4 --wait
kaggle b t status toysvg-redraw-from-png
kaggle b t log toysvg-redraw-from-png -m gemini-2.5-pro
kaggle b t download toysvg-redraw-from-png -o ./results
kaggle b t publish toysvg-redraw-from-png
```

Four details decide whether this works:

- **`-d` is per dataset and must be repeated on every push.** Re-pushing without it
  silently detaches the data the task needs.
- **The push slug must match the `@task(name=...)` in the file.** Otherwise the push
  fails before anything runs.
- **Two tasks, one benchmark.** The drawing task and the structure control are
  pushed separately, then added to the same benchmark in the Kaggle web UI: the CLI
  manages tasks, and a *benchmark* is a curated collection of them.
- **Cost is measurable, not asserted:** `kaggle b quota` reports inference spend
  through the model proxy, and `kaggle b leaderboard <owner>/<benchmark> --show`
  prints the per-model, per-task table.'''

MD_REPLAY = r'''## Mode `replay`: Offline Verification Mode

Scores a stored run through the same code path a live run uses. If you are reading
this to check the benchmark rather than to run it, this is the mode that matters:
it reproduces the specification's numbers from the shipped answers.'''

REPLAY = r'''if MODE == "replay":
    run_dir = ROOT / "runs" / "gemini-2.5-flash-demo"
    stored = [json.loads(line) for line in
              (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    scored = []
    for rec in stored:
        iid = rec["item_id"]
        if ITEMS and iid not in set(ITEMS):
            continue
        raw = run_dir / rec.get("raw_file", f"raw/{iid}.txt")
        if not raw.exists():
            continue
        scored.append(ft.score_answer(raw.read_text(), iid, model="gemini-2.5-flash",
                                      finish_reason=rec.get("finish_reason")))
    print(ft.decompose(scored))
    print("\nThe three headline numbers here are the ones the specification reports: "
          "0.262, 0.132 and 0.176, against ceilings 0.899, 0.889 and 0.482.")'''

REPLAY_FIGS = r'''# ---- the same figures the specification carries ---------------------------
if MODE == "replay":
    figs = WORK / "figures"
    figs.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        from tools import make_figures
        make_figures._main(["--pack", str(PACK_DIR),
                            "--run", str(ROOT / "runs" / "gemini-2.5-flash-demo"),
                            "--out", str(figs),
                            "--table", str(WORK / "demo-table.json"),
                            "--budget-figure"])
        print("wrote", sorted(p.name for p in figs.glob("*.pdf")))
    except Exception as exc:  # noqa: BLE001 - figures are a convenience, not the measurement
        print(f"figure step skipped: {type(exc).__name__}: {exc}")'''

MD_OWN = r'''## Mode `own-key`: your own provider

Useful for a model that is not in Kaggle's panel, or when you want the harness's
own runner. The key comes from Kaggle Secrets; nothing but an 8-character
fingerprint of it is ever written to a run directory.'''

OWN_KEY = r'''if MODE == "own-key":
    from kaggle_secrets import UserSecretsClient
    from svgbench.runner import _main as run_models
    from svgbench.score import _main as score_run

    key = UserSecretsClient().get_secret("GEMINI_API_KEY")   # Add-ons -> Secrets
    (WORK / "keys").mkdir(exist_ok=True)
    key_file = WORK / "keys" / "gemini.txt"
    key_file.write_text(key)
    os.chmod(key_file, 0o600)
    print("key loaded; only its fingerprint will be recorded")

    run_models(["--pack", str(PACK_DIR), "--out", str(WORK / "runs" / "kaggle-own"),
                "--provider", "gemini", "--model", "gemini-2.5-flash",
                "--key-file", str(key_file), "--thinking-budget", "0",
                "--max-output-tokens", "8192", "--min-interval", "20"])
    score_run(["--pack", str(PACK_DIR), "--run", str(WORK / "runs" / "kaggle-own")])'''

RESULTS = r'''# ---- score against the range it came from ---------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_findings():
    paths = [WORK / FINDINGS, Path(FINDINGS)]
    for p in paths:
        if p.exists():
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return []

scored = load_findings()
if not scored and MODE == "replay":
    scored = []
    run_dir = ROOT / "runs" / "gemini-2.5-flash-demo"
    for rec in [json.loads(l) for l in (run_dir / "results.jsonl").read_text().splitlines()
                if l.strip()]:
        raw = run_dir / rec.get("raw_file", f"raw/{rec['item_id']}.txt")
        if raw.exists():
            scored.append(ft.score_answer(raw.read_text(), rec["item_id"],
                                          model="gemini-2.5-flash"))

assert scored, "nothing scored yet: run mode replay, own-key, or a kbench run first"
fig, ax = plt.subplots(figsize=(9, 3.6))
xs = np.arange(len(scored))
ax.bar(xs, [r["score"] for r in scored], width=0.55, color="#2b6cb0", label="model $S$")
ax.bar(xs, [r["s_region"] for r in scored], width=0.22, color="#90cdf4", label="region term")
ax.plot(xs, [r["ceiling_score"] or np.nan for r in scored], "k*", ms=10,
        label="tracer ceiling $C(k)$")
if scored[0].get("floors"):
    ax.plot(xs, [r["floors"].get("bbox", np.nan) for r in scored], "v",
            color="#c53030", ms=7, label="bounding-box floor")
ax.set_xticks(xs)
ax.set_xticklabels([r["item_id"][:8] for r in scored], fontsize=8, rotation=90)
ax.set_ylabel("score")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25, axis="y")
ax.legend(fontsize=8)
ax.set_title("score against the range it came from", fontsize=10)
fig.tight_layout()
out = WORK / "toysvg-scores.png"
fig.savefig(out, dpi=140)
print("wrote", out)'''

MD_END = r'''## How the score is computed

For each item the model sees one PNG and must return path data inside a command
budget. The harness parses the answer strictly, fits translation and uniform scale
(never rotation or reflection, because an upside-down arrow is a different figure),
then measures three terms against the hidden path:

- **region**: soft intersection-over-union over coverage rather than a thresholded
  mask, so a half-pixel shift costs a little instead of stepping off a cliff;
- **edge**: the symmetric mean boundary distance mapped through `exp(-d / 3 px)`,
  with the 95th percentile beside it, because one stray part moves the tail
  enormously and the mean only slightly;
- **topology**: `1 / (1 + delta)` on the difference in component and hole counts,
  which is what stops a filled-in silhouette from passing on overlap alone.

The composite is `0.55 * region + 0.35 * edge + 0.10 * topology`. Every item also
carries two reference answers computed from the same pixels: a tracer ceiling
`C(k)` and shape-blind floors, so `rho = S / C(k)` says how much of what was
achievable from those pixels at that budget the model actually got.

## What this measures, and what it does not

It measures fidelity to an **exact hidden path**, which is why it needs no judge and
why the same answer always yields the same number. It does not claim that geometric
fidelity equals what a viewer would call a good drawing: a pleasing copy that is
geometrically wrong scores low here, and an uninspiring but exact copy scores 1.0.
Read a score against its item's floor and ceiling, never on its own.

Two leakage controls are structural rather than declared. The model-facing half
cannot reach the ground truth (`pack/` is what a runner reads and `gt/` is
scorer-only, and this notebook re-asserts the split before spending a call) and
prompts omit asset names, categories, and provenance cues to prevent textual recall. A
model that recognises an icon from memory can still score well, which is why the
split design exists and why the control task is run alongside: it separates "cannot
see the structure" from "cannot produce coordinates".'''

CELLS: List[Cell] = [
    ("markdown", TITLE),
    ("code", CONFIG),
    ("code", SETUP),
    ("code", PACK),
    ("code", SELFTEST),
    ("markdown", MD_KBENCH),
    ("markdown", MD_REPLAY),
    ("code", REPLAY),
    ("code", REPLAY_FIGS),
    ("markdown", MD_OWN),
    ("code", OWN_KEY),
    ("code", RESULTS),
    ("markdown", MD_END),
]


def _strip_magics(source: str) -> str:
    """Remove IPython magics and shell escapes so the cell can be compiled."""
    out = []
    for line in source.splitlines():
        out.append("pass" if line.lstrip().startswith(("%", "!")) else line)
    return "\n".join(out)


def check(cells: Sequence[Cell]) -> List[str]:
    problems = []
    for i, (kind, src) in enumerate(cells):
        if kind != "code":
            continue
        try:
            ast.parse(_strip_magics(src))
        except SyntaxError as exc:
            problems.append(f"cell {i}: {exc}")
    return problems


def notebook(cells: Sequence[Cell]) -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": src.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
            }
            for kind, src in cells
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("notebook")))
    ap.add_argument("--check", action="store_true", help="compile the cells, write nothing")
    args = ap.parse_args(argv)

    problems = check(CELLS)
    if problems:
        for p in problems:
            print(p)
        return 1
    if args.check:
        print(f"{sum(1 for k, _ in CELLS if k == 'code')} code cells compile; "
              f"{sum(1 for k, _ in CELLS if k == 'markdown')} markdown cells")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "toysvg_kaggle.ipynb"
    nb_json = json.dumps(notebook(CELLS), indent=1) + "\n"
    out.write_text(nb_json, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KiB, {len(CELLS)} cells)")
    print("run it on Kaggle by attaching the dataset as a normal notebook, or "
          "locally with TOYSVG_ROOT=<bundle>")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
