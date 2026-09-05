#!/usr/bin/env python3
"""Visual demonstrations of the metric terms *on real model answers*.

The model-free primers live in :mod:`tools.make_metric_primers` and come first in
the specification; these two figures come later, where the three-item demo is
reported, and they exist to show what the terms do to answers that were actually
generated:

* ``demo-edge``: the distance field, the two directed means and the
  exponential, on the first demo item;
* ``demo-topology``: components and holes, truth above answer, on all three
  demo items, where one answer read a thin outline as a solid silhouette.

    python -m tools.make_metric_figures --pack data/pack \\
        --run data/runs/gemini-2.5-flash-demo --out docs/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage

from svgbench.align import align_similarity
from svgbench.metrics import region_boundary
from svgbench.score import _merged_parsed, _render_aligned, boundary_parsed, gt_coverage
from svgbench.tasks import load_hidden, load_public
from tools.figstyle import COMPONENT_COLORS, HOLE_COLOR, bare, overlay, paint_components


def figure_edge(out_dir: Path, records: Sequence[Dict[str, object]]) -> Tuple[Path, Dict[str, object]]:
    r = records[0]
    item, gt, pred = r["item"], r["gt"], r["aligned"]
    bg, bp = region_boundary(gt >= 0.5), region_boundary(pred >= 0.5)
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    d_pred_to_gt = dt_g[bp]
    d_gt_to_pred = dt_p[bg]
    mean_pg, mean_gp = float(d_pred_to_gt.mean()), float(d_gt_to_pred.mean())
    cd = 0.5 * (mean_pg + mean_gp)

    fig = plt.figure(figsize=(11.4, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.35], wspace=0.12)

    base = np.ones(gt.shape + (3,))
    base[gt >= 0.5] = 0.90
    base[pred >= 0.5] = 0.78
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(base, interpolation="nearest")
    ax.imshow(np.where(bg, 1.0, np.nan), cmap="autumn_r", interpolation="nearest", vmin=0, vmax=1)
    ax.imshow(np.where(bp, 1.0, np.nan), cmap="winter", interpolation="nearest", vmin=0, vmax=1)
    bare(ax, "where the distances are measured\nred = truth outline, blue = answer outline")
    ax.legend(handles=[Patch(color="#d62728", label=f"truth boundary ({int(bg.sum())} px)"),
                       Patch(color="#1f77b4", label=f"answer boundary ({int(bp.sum())} px)")],
              fontsize=7.2, loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    shown = np.where(dt_g <= 24, dt_g, np.nan)
    im = ax.imshow(shown, cmap="viridis_r", interpolation="nearest")
    ax.imshow(np.where(bp, 1.0, np.nan), cmap="gray", interpolation="nearest", vmin=0, vmax=1.6, alpha=0.55)
    bare(ax, "the field itself: distance from any point\nto the nearest truth-outline pixel")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("distance (px)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    ax = fig.add_subplot(gs[0, 2])
    xs = np.linspace(0, 24, 400)
    for tau, style, lw, col in ((1.0, ":", 1.1, "#999999"), (3.0, "-", 2.0, "#2b6cb0"),
                                (10.0, "--", 1.2, "#999999")):
        ax.plot(xs, np.exp(-xs / tau), style, lw=lw, color=col,
                label=rf"$\tau = {tau:g}$ px" + ("  (used)" if tau == 3.0 else ""))
    for rr in records:
        c, sc = rr["chamfer_px"], rr["s_edge"]
        ax.plot([c], [sc], "*", ms=13, color="#333333")
        ax.annotate(f"{rr['item'].item_id[:8]}\ncd {c:.1f}px $\\to$ {sc:.3f}", (c, sc),
                    textcoords="offset points", xytext=(7, 6), fontsize=7)
    ax.set_xlabel("Chamfer distance (px)", fontsize=9)
    ax.set_ylabel("$s_E = \\exp(-\\mathrm{cd}/\\tau)$", fontsize=9)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.set_title("why $\\tau = 3$ px makes this term sharp", fontsize=9.5, pad=6)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.20)
    path = out_dir / "demo-edge.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "demo-edge.png", dpi=140)
    plt.close(fig)

    return path, {
        "item_id": item.item_id,
        "gt_boundary_px": int(bg.sum()),
        "pred_boundary_px": int(bp.sum()),
        "mean_pred_to_gt_px": mean_pg,
        "mean_gt_to_pred_px": mean_gp,
        "chamfer_px": cd,
        "s_edge": float(np.exp(-cd / 3.0)),
        "hausdorff95_px": max(float(np.percentile(d_pred_to_gt, 95)), float(np.percentile(d_gt_to_pred, 95))),
    }


def figure_topology(out_dir: Path, records: Sequence[Dict[str, object]]) -> Tuple[Path, Dict[str, object]]:
    n = len(records)
    fig, axes = plt.subplots(2, n, figsize=(3.7 * n, 7.4))
    if n == 1:
        axes = axes.reshape(2, 1)

    detail = []
    for j, r in enumerate(records):
        item, gt, pred = r["item"], r["gt"], r["aligned"]
        gc, gh = paint_components(axes[0][j], gt >= 0.5, "hidden ground truth")
        pc, ph = paint_components(axes[1][j], pred >= 0.5, f"model answer, item {item.item_id[:8]}")
        delta = abs(gc - pc) + abs(gh - ph)
        s_t = 1.0 / (1.0 + delta)
        axes[1][j].set_xlabel(f"$\\Delta = |{gc}-{pc}| + |{gh}-{ph}| = {delta}$"
                              f"   $\\Rightarrow$   $s_T = 1/(1+{delta}) = {s_t:.4f}$", fontsize=8.2)

        raw = r["text"]
        style = "stroke" if "stroke" in raw else "fill"
        detail.append({
            "item_id": item.item_id, "stratum": item.stratum, "answer_style": style,
            "gt_components": gc, "gt_holes": gh, "pred_components": pc, "pred_holes": ph,
            "delta": delta, "s_topology": s_t,
        })

    fig.legend(handles=[
        Patch(color=COMPONENT_COLORS[0], label="separate ink regions (components)"),
        Patch(color=HOLE_COLOR, label="enclosed holes"),
        Patch(color="white", label="background"),
    ], fontsize=8.2, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))

    fig.subplots_adjust(left=0.02, right=0.985, top=0.93, bottom=0.105, hspace=0.16, wspace=0.05)
    path = out_dir / "demo-topology.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "demo-topology.png", dpi=140)
    plt.close(fig)
    return path, {"items": detail}


# ---------------------------------------------------------------- driver


def collect(pack_dir: Path, run_dir: Path) -> List[Dict[str, object]]:
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = {r["item_id"]: r for r in manifest.get("results", [])}
    out = []
    for task in load_public(pack_dir):
        iid = str(task["item_id"])
        if iid not in rows:
            continue
        item = load_hidden(pack_dir, iid)
        text = (run_dir / rows[iid]["raw_file"]).read_text(encoding="utf-8")
        gt = gt_coverage(item)
        parsed, doc = _merged_parsed(text.strip(), item.gt_fill_rule)
        if parsed is None:
            continue
        raw_cov = _render_aligned(parsed, doc, None, item.canvas, item.gt_fill_rule)
        al = align_similarity(boundary_parsed(raw_cov) or parsed, gt, float(raw_cov.sum()),
                              mode="similarity")
        pred = _render_aligned(parsed, doc, al.matrix, item.canvas, item.gt_fill_rule)
        gt_b, pr_b = gt >= 0.5, pred >= 0.5
        bg, bp = region_boundary(gt_b), region_boundary(pr_b)
        dt_g = ndimage.distance_transform_edt(~bg)
        dt_p = ndimage.distance_transform_edt(~bp)
        cd = (0.5 * (float(dt_g[bp].mean()) + float(dt_p[bg].mean()))
              if bg.any() and bp.any() else float("nan"))
        out.append({
            "item_id": iid, "item": item, "text": text.strip(), "gt": gt, "aligned": pred,
            "chamfer_px": cd, "s_edge": float(np.exp(-cd / 3.0)) if cd == cd else 0.0,
        })
    out.sort(key=lambda r: ["S", "M", "C"].index(r["item"].stratum))
    return out


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--run", default="data/runs/gemini-2.5-flash-demo")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--json-out", default="data/analysis/metric-demos.json")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = collect(Path(args.pack), Path(args.run))
    if not records:
        raise SystemExit("no scored items found")

    p1, d1 = figure_edge(out_dir, records)
    p2, d2 = figure_topology(out_dir, records)
    blob = {"edge": d1, "topology": d2}
    Path(args.json_out).write_text(json.dumps(blob, indent=2), encoding="utf-8")

    for p in (p1, p2):
        print(f"wrote {p}")
    print(f"wrote {args.json_out}\n")
    print(f"edge: item {d1['item_id'][:8]}  chamfer={d1['chamfer_px']:.3f}px  "
          f"s_E={d1['s_edge']:.4f}  hausdorff95={d1['hausdorff95_px']:.2f}px")
    print("topology:")
    for t in d2["items"]:
        print(f"  {t['item_id'][:8]} {t['stratum']} style={t['answer_style']:6s} "
              f"gt {t['gt_components']}c/{t['gt_holes']}h vs pred "
              f"{t['pred_components']}c/{t['pred_holes']}h -> delta={t['delta']} "
              f"s_T={t['s_topology']:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
