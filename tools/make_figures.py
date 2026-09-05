#!/usr/bin/env python3
"""Build the specification's figures and worked-example table from real run data.

Nothing here is drawn by hand or invented. Every panel is the artefact the harness
actually used: the PNG the model was shown, the hidden ground truth, the model's
own geometry, and their colour-coded disagreement; and every number in the table
is measured by the same scorer that produced them. If a number in the document
cannot be traced back to one of these files, it does not belong in the document.

Usage::

    python -m tools.make_figures --pack data/pack --run data/runs/gemini-2.5-flash-demo \\
        --out docs/figures --table data/analysis/demo-table.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

from svgbench.metrics import compare, topology_counts
from svgbench.pathdata import parse_path
from svgbench.raster import render_coverage
from svgbench.score import (
    _merged_parsed,
    _render_aligned,
    boundary_parsed,
    gt_coverage,
    score_item,
    trace_ceiling,
)
from svgbench.align import align_similarity
from svgbench.tasks import SS, load_hidden, load_public

INK = "#1a1a1a"
GT_COLOR = np.array([0.85, 0.20, 0.20])   # ground truth only
PRED_COLOR = np.array([0.13, 0.36, 0.80])  # prediction only
BOTH_COLOR = np.array([0.10, 0.10, 0.10])


def overlay_rgb(gt: np.ndarray, pred: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Colour-coded disagreement: red = missed, blue = spurious, black = agreed."""
    g = gt >= thresh
    p = pred >= thresh
    img = np.ones(gt.shape + (3,), dtype=np.float64)
    img[g & ~p] = GT_COLOR
    img[~g & p] = PRED_COLOR
    img[g & p] = BOTH_COLOR
    return img


def _render_answer(text: str, item, gt_cov: np.ndarray):
    """Parse, align and render a stored answer exactly as the scorer does."""
    parsed, doc = _merged_parsed(text.strip(), item.gt_fill_rule)
    if parsed is None:
        return None, None, None
    raw = _render_aligned(parsed, doc, None, item.canvas, item.gt_fill_rule)
    fit = boundary_parsed(raw) or parsed
    al = align_similarity(fit, gt_cov, float(raw.sum()), mode="similarity")
    aligned = _render_aligned(parsed, doc, al.matrix, item.canvas, item.gt_fill_rule)
    return raw, aligned, al


def collect(pack_dir: Path, run_dir: Path) -> List[Dict[str, object]]:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows: Dict[str, Dict[str, object]] = {r["item_id"]: r for r in manifest.get("results", [])}
    out: List[Dict[str, object]] = []
    for task in load_public(pack_dir):
        item_id = str(task["item_id"])
        if item_id not in rows:
            continue
        item = load_hidden(pack_dir, item_id)
        text = (run_dir / rows[item_id]["raw_file"]).read_text(encoding="utf-8")
        gt = gt_coverage(item)
        png = np.asarray(Image.open(pack_dir / str(task["input_png"])).convert("L"))
        raw, aligned, al = _render_answer(text, item, gt)
        sc = score_item(text, item, "demo", finish_reason=rows[item_id].get("finish_reason"),
                        want_ceiling=True, png_path=pack_dir / str(task["input_png"]))
        out.append({
            "item_id": item_id, "item": item, "text": text.strip(), "gt": gt, "png": png,
            "raw": raw, "aligned": aligned, "align": al, "score": sc,
            "stratum": item.stratum, "split": item.split,
            "finish_reason": rows[item_id].get("finish_reason"),
        })
    return out


# ---------------------------------------------------------------- figure 1: panels


def figure_panels(records: Sequence[Dict[str, object]], out_dir: Path) -> Path:
    n = len(records)
    fig = plt.figure(figsize=(11.4, 2.95 * n))
    gs = GridSpec(n, 4, figure=fig, hspace=0.28, wspace=0.06)
    titles = ["(a) input the model sees", "(b) hidden ground truth",
              "(c) the model's answer", "(d) disagreement"]

    for i, r in enumerate(records):
        item = r["item"]
        gt, png, aligned = r["gt"], r["png"], r["aligned"]
        sc = r["score"]
        panels = [
            (png, None),
            (gt, None),
            (aligned if aligned is not None else np.zeros_like(gt), None),
            (overlay_rgb(gt, aligned if aligned is not None else np.zeros_like(gt)), "rgb"),
        ]
        for j, (data, kind) in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            if kind == "rgb":
                ax.imshow(data, interpolation="nearest")
            else:
                ax.imshow(1.0 - data, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#bbbbbb")
            if i == 0:
                ax.set_title(titles[j], fontsize=9.5, pad=6)
        # row label
        fig.text(0.012, 1 - (i + 0.5) / n, f"item {item.item_id[:8]}\nstratum {item.stratum}\n"
                 f"score {sc.score:.3f}", fontsize=8.2, va="center", ha="left", family="monospace")

    fig.subplots_adjust(left=0.115, right=0.995, top=0.955, bottom=0.012)
    path = out_dir / "demo-panels.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "demo-panels.png", dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- figure 2: budget


def figure_budget(records: Sequence[Dict[str, object]], out_dir: Path, budgets=(16, 48, 160, None)) -> Path:
    fig, axes = plt.subplots(1, len(records), figsize=(11.4, 3.0), sharey=True)
    if len(records) == 1:
        axes = [axes]
    xs = np.array([16, 48, 160, 400], dtype=float)

    for ax, r in zip(axes, records):
        item, sc = r["item"], r["score"]
        png = r["png"]
        gt = r["gt"]
        ceiling, cmds = [], []
        for k in budgets:
            cd, cn = trace_ceiling(png, k, canvas=item.canvas)
            try:
                cov, _, _, _ = _merged_parsed(cd, "nonzero")
                parsed, doc = cov, None
                cov0 = _render_aligned(parsed, doc, None, item.canvas, "nonzero")
                al = align_similarity(boundary_parsed(cov0) or parsed, gt, float(cov0.sum()), mode="similarity")
                c = float(compare(gt, _render_aligned(parsed, doc, al.matrix, item.canvas, "nonzero")).score)
            except Exception:
                c = np.nan
            ceiling.append(c)
            cmds.append(cn)

        ax.plot(xs, ceiling, "-o", color="#444444", ms=3.4, lw=1.4, label="tracer ceiling $C(k)$")
        ax.axhline(sc.floors.get("bbox", np.nan), color="#999999", ls="--", lw=1.1, label="bbox floor")
        ax.plot([48], [sc.score], "*", color=PRED_COLOR, ms=13, label="model score at $k{=}48$")
        ax.set_xscale("log")
        ax.set_xticks([16, 48, 160, 400])
        ax.set_xticklabels(["16", "48", "160", "$\\infty$"], fontsize=8)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_title(f"{item.item_id[:8]}  ({item.stratum})", fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("score $S$", fontsize=9)
    axes[0].legend(fontsize=7.4, loc="lower right", frameon=True, framealpha=0.92)
    fig.subplots_adjust(left=0.062, right=0.99, top=0.9, bottom=0.11, wspace=0.08)
    path = out_dir / "demo-budget.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "demo-budget.png", dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- table


def build_table(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    rows = []
    for r in records:
        sc = r["score"]
        rows.append({
            "item_id": sc.item_id,
            "stratum": sc.stratum,
            "split": sc.split,
            "k": sc.k,
            "commands": sc.n_commands,
            "style": "stroke" if "stroke" in r["text"] else "fill",
            "s_region": sc.s_region,
            "s_edge": sc.s_edge,
            "s_topology": sc.s_topology,
            "score": sc.score,
            "ceiling": sc.ceiling_score,
            "ceiling_commands": sc.ceiling_commands,
            "rho": sc.rho,
            "chamfer_px": sc.chamfer_px,
            "hausdorff95_px": sc.hausdorff95_px,
            "area_ratio": sc.area_ratio,
            "gt_components": sc.gt_components,
            "pred_components": sc.pred_components,
            "gt_holes": sc.gt_holes,
            "pred_holes": sc.pred_holes,
            "floors": sc.floors,
            "align_scale": sc.align_scale,
            "align_tx": sc.align_tx,
            "align_ty": sc.align_ty,
            "answer": r["text"],
            "finish_reason": r["finish_reason"],
        })
    mean = float(np.mean([r["score"] for r in rows])) if rows else 0.0
    return {"model": "google gemini-2.5-flash (via Google AI)", "rows": rows, "mean_score": mean}


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--run", default="data/runs/gemini-2.5-flash-demo")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--table", default="data/analysis/demo-table.json")
    ap.add_argument("--budget-figure", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = collect(Path(args.pack), Path(args.run))
    records.sort(key=lambda r: ["S", "M", "C"].index(r["stratum"]))
    if not records:
        raise SystemExit("no scored items found for this run")

    p1 = figure_panels(records, out_dir)
    Path(args.table).write_text(json.dumps(build_table(records), indent=2), encoding="utf-8")

    print(f"wrote {p1}")
    if args.budget_figure:
        p2 = figure_budget(records, out_dir)
        print(f"wrote {p2}")
    print(f"wrote {args.table}")
    t = build_table(records)
    print(f"\nmean score across {len(t['rows'])} demo items: {t['mean_score']:.4f}")
    for r in t["rows"]:
        print(f"  {r['item_id'][:8]} {r['stratum']} cmds={r['commands']:3d} style={r['style']:6s} "
              f"S={r['score']:.3f} ceiling={r['ceiling']:.3f} rho={r['rho']:.3f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
