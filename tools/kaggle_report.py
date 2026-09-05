#!/usr/bin/env python3
"""Turn retrieved Kaggle runs into the report's tables and figures.

Input is what ``kaggle b t download`` wrote under ``data/kaggle-runs/``: one
directory per model per run, each holding the decomposition log the task appended
to as it scored (per-item components, gate, ceiling, and the raw answer) and the
three-panel preview strips. Nothing here re-scores anything: the numbers are the
ones the runs produced, so a figure cannot disagree with the log it came from.

Three pieces of output, because the questions a reader has are different:

* **per model** -- what did each model score, and out of what kind of failure;
* **per item** -- where did each model lose it, which a ranking cannot show;
* **plates** -- what each model actually drew for each input, beside that input and the
  hidden truth, which no table of numbers can show at all.

Also writes ``summary.json`` beside the runs, so the specification's numbers can be
reproduced without re-running anything.

    python -m tools.kaggle_report --runs data/kaggle-runs --figures docs/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

#: A preview strip is three panels wide with two 2\,px dividers: this is the panel
#: width it implies, and the offset of each panel inside it.
_PANEL_GAP = 2


def load_runs(runs: Path, task: str, log_name: str) -> List[Dict[str, Any]]:
    """Every retrieved run of one task: ``{model, version, rows}``."""
    found: List[Dict[str, Any]] = []
    for log in sorted(runs.glob(f"{task}/*/*/*/{log_name}")):
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if not rows:
            continue
        found.append({
            "model": log.parents[1].name,
            "version": int(log.parents[2].name),
            "run_id": log.parent.name,
            "rows": rows,
        })
    return found


def _ceilings(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    return {str(r["item_id"]): float(r["ceiling_score"]) for r in rows
            if r.get("ceiling_score") is not None}


def summarise(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-model means, spread, gate composition, and mean rho over scored items."""
    out: List[Dict[str, Any]] = []
    for run in runs:
        rows = run["rows"]
        scores = [float(r["score"]) for r in rows]
        gates: Dict[str, int] = {}
        for row in rows:
            gates[str(row["gate"])] = gates.get(str(row["gate"]), 0) + 1
        ceilings = _ceilings(rows)
        rhos = [float(r["score"]) / ceilings[str(r["item_id"])] for r in rows
                if str(r["item_id"]) in ceilings and ceilings[str(r["item_id"])] > 0]
        out.append({
            "model": run["model"],
            "version": run["version"],
            "n_items": len(rows),
            "mean_S": float(np.mean(scores)),
            "best_S": float(np.max(scores)),
            "worst_S": float(np.min(scores)),
            "mean_rho": float(np.mean(rhos)) if rhos else 0.0,
            "max_rho": float(np.max(rhos)) if rhos else 0.0,
            "items_beating_ceiling": int(sum(1 for x in rhos if x > 1.0)),
            "gates": dict(sorted(gates.items())),
            "mean_commands": float(np.mean([int(r["n_commands"]) for r in rows])),
        })
    return sorted(out, key=lambda d: -d["mean_S"])


def latest_per_model(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One run per model: the newest version, and the newest run within it.

    A task's history holds every push, so mixing versions would put two code paths in
    one plate. The version is part of the run's path, which is why it is read at all.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        key = run["model"]
        rank = (run["version"], run["run_id"])
        if key not in best or rank > (best[key]["version"], best[key]["run_id"]):
            best[key] = run
    return sorted(best.values(), key=lambda r: -float(np.mean([x["score"] for x in r["rows"]])))


def _panels(strip_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three panels of a retrieved strip: what was seen, drawn, hidden."""
    img = np.asarray(Image.open(strip_path).convert("L"))
    width = (img.shape[1] - 2 * _PANEL_GAP) // 3
    seen = img[:, 0:width]
    drawn = img[:, width + _PANEL_GAP:2 * width + _PANEL_GAP]
    truth = img[:, 2 * width + 2 * _PANEL_GAP:3 * width + 2 * _PANEL_GAP]
    return seen, drawn, truth


def figure_plates(runs: Sequence[Dict[str, Any]], dest: Path, runs_root: Path,
                  task: str, items_per_plate: int = 6) -> List[Path]:
    """One plate per group of items: input, every model's drawing, hidden truth.

    This is the figure that answers the question a score cannot: *what did each model
    actually draw for this image*. Rows are the input, one row per model in the order
    of the summary table, and the hidden truth; columns are items, ordered by their
    tracer ceiling so the hardest item in the pack is leftmost.

    The drawn panel is the answer rendered through the scorer's own aligned path (the
    geometry the number measures) rather than the raw path data, so a plate and the
    score beside it cannot disagree. Panels come from the strips the run itself wrote,
    so nothing here is re-rendered and no image library can change a line.
    """
    latest = latest_per_model(runs)
    models = [run["model"] for run in latest]
    by_model = {run["model"]: run for run in latest}
    ceilings = _ceilings(latest[0]["rows"])
    # The split is a property of the item, not of the answer, so it is read once and
    # printed in the column header: a plate that shows which regime an item belongs to
    # does not need the prose to list item ids.
    splits = {str(r["item_id"]): str(r.get("split", "")) for r in latest[0]["rows"]}
    order = sorted({str(r["item_id"]) for r in latest[0]["rows"]},
                   key=lambda item: ceilings.get(item, 0.0))

    def strip_of(model: str, item: str) -> Path:
        run = by_model[model]
        return (runs_root / task / str(run["version"]) / model / run["run_id"]
                / "previews" / f"{item}.png")

    cell = 256
    gut = 3
    written: List[Path] = []
    for page, start in enumerate(range(0, len(order), items_per_plate), start=1):
        chunk = order[start:start + items_per_plate]
        sources = ["input"] + models + ["truth"]
        height = len(sources) * (cell + gut) + gut
        width = len(chunk) * (cell + gut) + gut
        canvas = np.full((height, width), 255, dtype=np.uint8)

        for col, item in enumerate(chunk):
            for row, source in enumerate(sources):
                strip = strip_of(models[0], item)
                if not strip.exists():
                    continue
                seen, drawn, truth = _panels(strip)
                if source == "input":
                    panel = seen
                elif source == "truth":
                    panel = truth
                else:
                    other = strip_of(source, item)
                    panel = _panels(other)[1] if other.exists() else drawn
                y0 = gut + row * (cell + gut)
                x0 = gut + col * (cell + gut)
                canvas[y0:y0 + cell, x0:x0 + cell] = panel

        fig, ax = plt.subplots(figsize=(7.4, 7.4 * height / width))
        ax.imshow(canvas, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([gut + i * (cell + gut) + cell / 2 for i in range(len(sources))])
        ax.set_yticklabels(
            ["the image\nthe model saw"] + [_display(m) for m in models] + ["hidden\ntruth"],
            fontsize=6.5)
        for col, item in enumerate(chunk):
            xc = gut + col * (cell + gut) + cell / 2
            ax.text(xc, -6, f"{item[:8]}\n{splits.get(item, '')}\nceiling "
                             f"{ceilings.get(item, 0):.2f}",
                    ha="center", va="bottom", fontsize=6.0, linespacing=1.35)
        # Per-cell score, so the plate can be read without the table beside it.
        for col, item in enumerate(chunk):
            for row, model in enumerate(models, start=1):
                rec = next(r for r in by_model[model]["rows"] if str(r["item_id"]) == item)
                y0 = gut + row * (cell + gut)
                x0 = gut + col * (cell + gut)
                mark = _gate_marker(str(rec["gate"]))
                label = f"{float(rec['score']):.2f}" + (f" {mark}" if mark else "")
                ax.text(x0 + 5, y0 + 14, label, fontsize=6.0, color="#111111",
                        va="top", ha="left",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8,
                                  pad=0.8))
        ax.set_xlabel(f"items {start + 1}-{start + len(chunk)} of {len(order)}, "
                      "ordered by tracer ceiling (hardest for a tracer first); "
                      "marks: x unparsable, o over budget", fontsize=6.5)
        fig.tight_layout()
        out = dest / f"kaggle-plate-{page}.pdf"
        fig.savefig(out)
        fig.savefig(dest / f"kaggle-plate-{page}.png", dpi=200)
        plt.close(fig)
        written.append(out)
    return written


def _gate_marker(gate: str) -> str:
    return {"ok": "", "unparsable": "x", "over_budget": "o", "empty": ".", "no_svg": "s",
            "truncated": "t", "degenerate": "d", "off_canvas": "^"}.get(gate, "?")


def figure_models(runs: Sequence[Dict[str, Any]], summary: Sequence[Dict[str, Any]],
                  ceiling_mean: float, dest: Path) -> None:
    """Mean score per model, the individual items behind it, and the ceiling reference.

    The item marks matter more than the bars: two models whose means are close can
    differ in whether they fail everywhere or answer most items and gate a few to
    zero, and a ranking alone hides that. Bars carry the gate tally for the same
    reason -- a zero from a gate is not the same zero as a zero from geometry.
    """
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    models = [s["model"] for s in summary]
    x = np.arange(len(models))
    means = [s["mean_S"] for s in summary]
    n_items = summary[0]["n_items"] if summary else 32
    ax.bar(x, means, width=0.55, color="#2b6cb0", alpha=0.85, label=f"mean $S$ over {n_items} items")

    rng = np.random.default_rng(0)  # fixed jitter: the figure must not move between runs
    for i, run in enumerate(runs):
        scores = np.array([float(r["score"]) for r in run["rows"]])
        jitter = rng.uniform(-0.16, 0.16, size=scores.size)
        ax.scatter(np.full(scores.size, i) + jitter, scores, s=14, marker="o",
                   facecolor="white", edgecolor="#1a365d", linewidth=0.7, zorder=3,
                   label="one item" if i == 0 else None)

    ax.axhline(ceiling_mean, color="#888888", ls="--", lw=1.0)
    ax.text(len(models) - 0.5, ceiling_mean + 0.015, "mean tracer ceiling (per item)",
            ha="right", va="bottom", fontsize=8, color="#555555")
    for i, s in enumerate(summary):
        # White backing: an item marker can land on the same spot when a model's
        # best item is close to its mean, and an unreadable number helps nobody.
        ax.text(i, s["mean_S"] + 0.012, f"{s['mean_S']:.3f}", ha="center", fontsize=8.5,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.9))
    # The gate tally goes under the model name rather than inside the bar: a bar is
    # only as wide as its category, and "3 over budget" is wider than the bar the
    # last models draw, so inside-the-bar text collides with the bar edge.
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_display(s['model'])}\n{_gate_text(s['gates'])}" for s in summary],
                       fontsize=8)
    ax.set_ylabel("$S$ per item; bar is the mean", fontsize=9)
    ax.set_ylim(0.0, max(1.0, max(means) + 0.12))
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(dest / "kaggle-models.pdf")
    fig.savefig(dest / "kaggle-models.png", dpi=200)
    plt.close(fig)


def figure_items(runs: Sequence[Dict[str, Any]], dest: Path) -> None:
    """Score per model per item, items ordered by ceiling, gate failures marked."""
    by_model = {run["model"]: {str(r["item_id"]): r for r in run["rows"]} for run in runs}
    ceilings = _ceilings(runs[0]["rows"])
    # Ordered by the ceiling, so the heatmap reads left-to-right as hardest-to-easiest
    # from the pixels' point of view rather than by item id.
    order = sorted({str(r["item_id"]) for r in runs[0]["rows"]},
                   key=lambda item: ceilings.get(item, 0.0))
    models = [s["model"] for s in summarise(runs)]

    grid = np.full((len(models), len(order)), np.nan)
    gates: List[List[str]] = [[str(by_model[m][i]["gate"]) for i in order] for m in models]
    for r, model in enumerate(models):
        for c, item in enumerate(order):
            grid[r, c] = float(by_model[model][item]["score"])

    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    im = ax.imshow(grid, cmap="magma_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{i[:4]}\n{ceilings[i]:.2f}" for i in order], fontsize=6.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.set_xlabel("item (first four id characters, tracer ceiling beneath)", fontsize=8)
    for r in range(len(models)):
        for c in range(len(order)):
            mark = _gate_marker(gates[r][c])
            txt = f"{grid[r, c]:.2f}"
            ax.text(c, r, txt + (f" {mark}" if mark else ""), ha="center", va="center",
                    fontsize=6.5,
                    color="white" if grid[r, c] < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, label="$S$", format="%.1f")
    fig.tight_layout()
    fig.savefig(dest / "kaggle-items.pdf")
    fig.savefig(dest / "kaggle-items.png", dpi=200)
    plt.close(fig)


def _display(model: str) -> str:
    """A model slug without its date stamp: ``claude-sonnet-4-5-20250929`` -> ``claude-sonnet-4-5``."""
    return re.sub(r"-20\d{6}$", "", model)


def _gate_text(gates: Dict[str, int]) -> str:
    """The gate column, in the words the table uses: ``ok`` alone means no gate fired."""
    fired = {k: v for k, v in gates.items() if k != "ok"}
    if not fired:
        return "none"
    return ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(fired.items()))


#: Rank suffixes for the generated macros, spelled because a LaTeX control word
#: may contain letters only.
_ORDINALS = ("One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight")


def _ordinal(rank: int) -> str:
    return _ORDINALS[rank - 1] if 0 < rank <= len(_ORDINALS) else f"R{rank}"


def write_tex(summary: Sequence[Dict[str, Any]], ceilings: Dict[str, float], dest: Path) -> List[Path]:
    """Emit the specification's table and the numbers its captions quote.

    Two fragments, because LaTeX needs them in two places: macros go in the preamble,
    the table body goes inside the ``tabularx``. Emitting them rather than typing them
    is the point: a number in the specification that was copied from a run by hand
    is a number that can disagree with the run it cites.
    """
    dest.mkdir(parents=True, exist_ok=True)
    values = list(ceilings.values())
    macros = dest / "kaggle-numbers.tex"
    macros.write_text(
        "% Generated by tools/kaggle_report.py -- do not edit; regenerate instead.\n"
        f"\\newcommand{{\\kgNItems}}{{{len(values)}}}\n"
        f"\\newcommand{{\\kgNModels}}{{{len(summary)}}}\n"
        f"\\newcommand{{\\kgCeilMean}}{{{float(np.mean(values)):.4f}}}\n"
        f"\\newcommand{{\\kgCeilMin}}{{{float(np.min(values)):.4f}}}\n"
        f"\\newcommand{{\\kgCeilMax}}{{{float(np.max(values)):.4f}}}\n"
        # Letters, not digits, in the macro names: a control word ends at the first
        # non-letter, so \kgMean1 would define \kgMean and typeset a stray `1`.
        + "".join(
            f"\\newcommand{{\\kgMean{_ordinal(i)}}}{{{s['mean_S']:.4f}}}"
            f"\\newcommand{{\\kgBest{_ordinal(i)}}}{{{s['best_S']:.3f}}}"
            f"\\newcommand{{\\kgWorst{_ordinal(i)}}}{{{s['worst_S']:.3f}}}"
            f"\\newcommand{{\\kgRho{_ordinal(i)}}}{{{s['mean_rho']:.3f}}}\n"
            for i, s in enumerate(summary, start=1)
        ),
        encoding="utf-8",
    )

    # The whole alignment, not only its rows: a tabularx cannot be opened in the
    # specification and closed here: a row break at the end of an \input leaves
    # \bottomrule outside the alignment, which TeX reports as a misplaced \noalign.
    rows = dest / "kaggle-model-table.tex"
    body = [
        "\\begin{tabularx}{\\textwidth}{@{}lrrrrrX@{}}",
        "\\toprule",
        "Model & mean $\\score$ & best & worst & mean $\\rho$ & above ceiling "
        "& Gates that fired \\\\",
        "\\midrule",
    ]
    for s in summary:
        body.append(
            f"\\texttt{{{_display(s['model'])}}} & {s['mean_S']:.4f} & {s['best_S']:.3f} & "
            f"{s['worst_S']:.3f} & {s['mean_rho']:.3f} & {s['items_beating_ceiling']}/"
            f"{s['n_items']} & {_gate_text(s['gates'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabularx}"]
    rows.write_text(
        "% Generated by tools/kaggle_report.py -- do not edit; regenerate instead.\n"
        + "\n".join(body)
        + "\n",
        encoding="utf-8",
    )
    return [macros, rows]


def _print(summary: Sequence[Dict[str, Any]], ceiling_mean: float) -> None:
    head = (f"{'model':<28} {'mean S':>7} {'mean rho':>9} {'best':>6} {'worst':>6} "
            f"{'>1 ceil':>8}  gates")
    print(head)
    print("-" * len(head))
    for s in summary:
        gates = ", ".join(f"{k} {v}" for k, v in s["gates"].items())
        print(f"{s['model']:<28} {s['mean_S']:>7.4f} {s['mean_rho']:>9.3f} "
              f"{s['best_S']:>6.3f} {s['worst_S']:>6.3f} "
              f"{s['items_beating_ceiling']:>8}  {gates}")
    print(f"\nmean tracer ceiling over the pack: {ceiling_mean:.4f}")
    print("'mean rho' is S/C per item averaged; '>1 ceil' counts items where the model "
          "scored above its own item's tracer ceiling, which is a reference rather than "
          "a cap.")


def _main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default="data/kaggle-runs")
    ap.add_argument("--figures", default="docs/figures")
    ap.add_argument("--task", default="toysvg-redraw-from-png")
    ap.add_argument("--log", default="toysvg-decompositions.jsonl")
    args = ap.parse_args(argv)

    runs = load_runs(Path(args.runs), args.task, args.log)
    if not runs and args.task == "toysvg-redraw-from-png":
        runs = load_runs(Path(args.runs), "figcopy-redraw-from-png", "figcopy-decompositions.jsonl")
    if not runs:
        raise SystemExit(f"no runs under {args.runs}/{args.task}; "
                         "run `kaggle b t download <task> -o data/kaggle-runs` first")
    runs = latest_per_model(runs)
    summary = summarise(runs)
    ceilings = _ceilings(runs[0]["rows"])
    ceiling_mean = float(np.mean(list(ceilings.values())))
    _print(summary, ceiling_mean)

    dest = Path(args.figures)
    dest.mkdir(parents=True, exist_ok=True)
    figure_models(latest_per_model(runs), summary, ceiling_mean, dest)
    figure_items(runs, dest)
    plates = figure_plates(runs, dest, Path(args.runs), args.task)
    print(f"plates -> {', '.join(p.name for p in plates)}")
    tex = write_tex(summary, ceilings, dest.parent / "generated")
    print(f"latex  -> {', '.join(str(p.relative_to(dest.parent)) for p in tex)}")
    (Path(args.runs) / "summary.json").write_text(
        json.dumps({"task": args.task, "ceiling_mean": ceiling_mean,
                    "models": summary}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nfigures -> {dest}/kaggle-models.pdf, {dest}/kaggle-items.pdf")
    print(f"summary -> {Path(args.runs) / 'summary.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
