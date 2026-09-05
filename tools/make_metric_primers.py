#!/usr/bin/env python3
"""Visual primers: what each reported quantity measures.

These figures come *before* any model number in the specification, and they are
deliberately model-free. Every panel is built from controlled geometry whose
answer is known in advance, so a reader can check the instrument against theory
before reading a single score, and so that a figure cannot be explained away
as an artefact of one model's answer.

One figure per family of quantities the report contains:

* ``metric-region``: soft IoU, and why soft rather than thresholded; also
  hard IoU, Dice and the area ratio, which are reported beside it;
* ``metric-edge``: the two directed boundary distances, the distance
  field they are taken over, the Chamfer mean, the 95th percentile and the
  tolerance-based boundary $F_1$;
* ``metric-topology``: components, holes, and what an error $\Delta$ looks
  like on cases whose true counts are known by construction;
* ``metric-align``: what placement and scale forgiveness buys, and what
  is deliberately refused;
* ``metric-anchors``: the floors and the ceiling, which are what make a
  score of a given size readable;
* ``metric-composite``: how the three terms combine, and what each one is
  worth when it fails completely.

    python -m tools.make_metric_primers --pack data/pack --out docs/figures
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
from PIL import Image
from scipy import ndimage

from svgbench.align import align_similarity
from svgbench.metrics import (DEFAULT_CHAMFER_TAU, boundary_metrics, compare, dice,
                              hard_iou, region_boundary, soft_iou, topology_counts)
from svgbench.pathdata import Matrix, SubPath, parse_path
from svgbench.raster import render_coverage
from svgbench.score import (boundary_parsed, floor_coverages, gt_coverage,
                            trace_ceiling, _merged_parsed, _render_aligned)
from svgbench.svgdoc import load_prediction
from svgbench.tasks import SS, Item, load_hidden, load_public
from tools.figstyle import (BOTH, COMPONENT_COLORS, GT_ONLY, HOLE_COLOR, PRED_ONLY,
                            band, bare, disc, legend_handles, overlay,
                            paint_components)

CANVAS = 256
TAU = float(DEFAULT_CHAMFER_TAU)


def rect(x0: float, y0: float, x1: float, y1: float) -> SubPath:
    return SubPath([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], True)


def render(subpaths: Sequence[SubPath], fill_rule: str = "nonzero") -> np.ndarray:
    return render_coverage(list(subpaths), size=CANVAS, ss=SS, fill_rule=fill_rule)


def boundary_sets(gt: np.ndarray, pr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return region_boundary(gt >= 0.5), region_boundary(pr >= 0.5)


def _flat_edge_means(gt: np.ndarray, pr: np.ndarray, x_lo: int = 70,
                     x_hi: int = 190) -> Tuple[float, float]:
    """Directed boundary means restricted to the flat edges, caps excluded.

    The two flat edges of an offset band are exactly ``delta`` apart, so this is
    the one quantity in the edge figure with an exact answer to check against.
    """
    bg, bp = boundary_sets(gt, pr)
    if not bg.any() or not bp.any():
        return float("nan"), float("nan")
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    ys, xs = np.nonzero(bp)
    sel = (xs >= x_lo) & (xs <= x_hi)
    flat_p = float(dt_g[ys[sel], xs[sel]].mean()) if sel.any() else float("nan")
    ys, xs = np.nonzero(bg)
    sel = (xs >= x_lo) & (xs <= x_hi)
    flat_g = float(dt_p[ys[sel], xs[sel]].mean()) if sel.any() else float("nan")
    return flat_p, flat_g


def directed_means(gt: np.ndarray, pr: np.ndarray) -> Tuple[float, float]:
    """The two directed means of the symmetric Chamfer distance, in pixels."""
    bg, bp = boundary_sets(gt, pr)
    if not bg.any() or not bp.any():
        return float("nan"), float("nan")
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    return float(dt_g[bp].mean()), float(dt_p[bg].mean())


# ---------------------------------------------------------------- 1. region


def figure_region(out_dir: Path) -> Tuple[Path, Dict[str, object]]:
    """Two bands of known width, offset by a known displacement."""
    width = 16.0
    deltas = [0.0, 2.0, 4.0, 8.0, 12.0]
    measured: List[Dict[str, float]] = []
    panels = []
    for d in deltas:
        gt, pr = band(0.0, 0.0, w=int(width)), band(0.0, d, w=int(width))
        m = compare(gt, pr)
        analytic_iou = (width - d) / (width + d)
        analytic_dice = (width - d) / width
        measured.append({
            "delta_px": d, "soft_iou": soft_iou(gt, pr), "hard_iou": hard_iou(gt, pr),
            "dice": dice(gt, pr), "area_ratio": float(m.area_ratio),
            "analytic_iou": analytic_iou, "analytic_dice": analytic_dice,
        })
        panels.append((d, gt, pr, soft_iou(gt, pr)))

    fine = np.arange(0.0, 8.0001, 0.125)
    soft_c, hard_c, dice_c, iou_c, dice_an = [], [], [], [], []
    for d in fine:
        gt, pr = band(0.0, 0.0, w=int(width)), band(0.0, d, w=int(width))
        soft_c.append(soft_iou(gt, pr))
        hard_c.append(hard_iou(gt, pr))
        dice_c.append(dice(gt, pr))
        iou_c.append((width - d) / (width + d))
        dice_an.append((width - d) / width)

    fig = plt.figure(figsize=(11.6, 4.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 1.3], hspace=0.36, wspace=0.08)
    for j, (d, gt, pr, s) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(overlay(gt, pr), interpolation="nearest")
        bare(ax, f"offset $\\delta = {d:g}$ px\nsoft IoU $= {s:.3f}$")
    fig.legend(handles=list(legend_handles()), fontsize=8.0, loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.995))

    ax = fig.add_subplot(gs[1, :3])
    ax.plot(fine, soft_c, lw=1.9, color="#2b6cb0", label="soft IoU  (the region term)")
    ax.plot(fine, hard_c, lw=1.3, color="#dd6b20", ls="--", label="hard IoU  (threshold at 0.5)")
    ax.plot(fine, dice_c, lw=1.3, color="#38a169", ls="-.", label="Dice  (also reported)")
    ax.plot(fine, iou_c, lw=1.0, color="#888888", ls=":", label=r"closed form $(w-\delta)/(w+\delta)$")
    for mm in measured:
        ax.plot([mm["delta_px"]], [mm["soft_iou"]], "o", color="#2b6cb0", ms=4)
    ax.set_xlabel("band offset $\\delta$ (px)", fontsize=9)
    ax.set_ylabel("agreement", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.6, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.set_title("IoU is hyperbolic in the offset, Dice linear; both use the same overlap",
                 fontsize=9.5, pad=6)

    ax = fig.add_subplot(gs[1, 3:])
    ax.plot(fine, soft_c, lw=1.9, color="#2b6cb0", label="soft IoU")
    ax.plot(fine, hard_c, lw=1.3, color="#dd6b20", ls="--", label="hard IoU")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("band offset $\\delta$ (px, sub-pixel range)", fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.6, loc="lower left")
    ax.tick_params(labelsize=8)
    ax.set_title("why soft: hard IoU is a staircase", fontsize=9.5, pad=6)

    fig.subplots_adjust(left=0.045, right=0.99, top=0.84, bottom=0.115)
    path = out_dir / "metric-region.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "metric-region.png", dpi=140)
    plt.close(fig)

    steps = sum(1 for a, b in zip(hard_c, hard_c[1:]) if abs(a - b) > 1e-12)
    err_iou = max(abs(m["soft_iou"] - m["analytic_iou"]) for m in measured)
    err_dice = max(abs(m["dice"] - m["analytic_dice"]) for m in measured)
    ratio_spread = (min(m["area_ratio"] for m in measured), max(m["area_ratio"] for m in measured))
    return path, {
        "band_width_px": width,
        "measured": measured,
        "hard_iou_distinct_plateaus": steps,
        "max_abs_error_soft_vs_analytic": float(err_iou),
        "max_abs_error_dice_vs_analytic": float(err_dice),
        "area_ratio_min": float(ratio_spread[0]),
        "area_ratio_max": float(ratio_spread[1]),
    }


# ---------------------------------------------------------------- 2. edge


def figure_edge(out_dir: Path) -> Tuple[Path, Dict[str, object]]:
    """Boundary distances on a controlled pair, plus one outlier case."""
    gt = band(0.0, 0.0)
    pr = band(0.0, 6.0)
    bg, bp = boundary_sets(gt, pr)
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    mean_pg, mean_gp = float(dt_g[bp].mean()), float(dt_p[bg].mean())
    bm = boundary_metrics(gt, pr)
    chamfer = bm["chamfer_px"]

    stub = np.maximum(pr, render([rect(196.0, 24.0, 224.0, 52.0)]))
    bm_stub = boundary_metrics(gt, stub)
    stub_mean = float(np.mean([m for m in directed_means(gt, stub) if m == m]))

    # Sweeps stop at w/2. Beyond that the *nearest* truth edge to a prediction edge
    # is no longer the one it was offset from, so the distances stop tracking the
    # displacement -- a property of every nearest-distance metric, not a defect,
    # but one that has to be visible in a validation panel.
    sweep_d = np.arange(0.0, 8.0001, 1.0)
    sweep_cd, sweep_f1, sweep_p95 = [], [], []
    for d in sweep_d:
        bb = boundary_metrics(band(0.0, 0.0), band(0.0, float(d)))
        sweep_cd.append(bb["chamfer_px"])
        sweep_f1.append(bb["boundary_f1"])
        sweep_p95.append(bb["hausdorff95_px"])
    sweep_cd = np.asarray(sweep_cd)

    # The flat-edge check, on a quarter-pixel grid: this is the quantity with an
    # exact answer (two parallel edges, offset by a known amount), so what is left
    # after it is subtracted is the resolution of a thresholded boundary.
    fine_d = np.arange(0.0, 8.0001, 0.125)
    fine_flat = np.asarray([0.5 * sum(_flat_edge_means(band(0.0, 0.0), band(0.0, float(d))))
                            for d in fine_d])
    flat_dev = float(np.max(np.abs(fine_flat - fine_d)))
    flat_step = float(np.min(np.abs(np.diff(np.unique(np.round(fine_flat, 6))))))

    fig = plt.figure(figsize=(12.2, 5.5))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.16)

    # (a) the two boundary sets
    base = np.ones(gt.shape + (3,))
    base[gt >= 0.5] = 0.92
    base[pr >= 0.5] = 0.80
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(base, interpolation="nearest")
    ax.imshow(np.where(bg, 1.0, np.nan), cmap="autumn_r", interpolation="nearest", vmin=0, vmax=1)
    ax.imshow(np.where(bp, 1.0, np.nan), cmap="winter", interpolation="nearest", vmin=0, vmax=1)
    bare(ax, "(a) what the distances are between\ntruth outline (red) and answer outline (blue)")
    ax.legend(handles=[Patch(color="#d62728", label=f"truth boundary, {int(bg.sum())} px"),
                       Patch(color="#1f77b4", label=f"answer boundary, {int(bp.sum())} px")],
              fontsize=7.0, loc="lower center", bbox_to_anchor=(0.5, -0.34), frameon=False)

    # (b) the distance field
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(np.where(dt_g <= 24, dt_g, np.nan), cmap="viridis_r", interpolation="nearest")
    ax.imshow(np.where(bp, 1.0, np.nan), cmap="gray", interpolation="nearest", vmin=0, vmax=1.6, alpha=0.55)
    bare(ax, "(b) the distance field\nfrom any pixel to the truth outline")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("distance (px)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    # (c) the two directed means
    ax = fig.add_subplot(gs[0, 2])
    labels = ["answer $\\to$ truth\n(precision-ish)", "truth $\\to$ answer\n(recall-ish)", "Chamfer\n= their mean"]
    vals = [mean_pg, mean_gp, chamfer]
    cols = ["#2b6cb0", "#c05621", "#4a5568"]
    ax.bar([0, 1, 2], vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.12, f"{v:.2f} px", ha="center", fontsize=8.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=7.6)
    ax.set_ylabel("mean distance (px)", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    ax.tick_params(labelsize=8)
    ax.set_title("(c) both directions taken,\nso neither shape is privileged", fontsize=9.5, pad=6)

    # (d) measured Chamfer against the known displacement
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(fine_d, fine_d, ":", lw=1.3, color="#bbbbbb", label="exact: flat edges are $\\delta$ apart")
    ax.plot(fine_d, fine_flat, "-", lw=1.7, color="#38a169",
            label="measured, flat edges only")
    ax.plot(sweep_d, sweep_cd, "o-", ms=3.2, lw=1.7, color="#2b6cb0",
            label="measured, whole boundary")
    ax.set_xlabel("known offset $\\delta$ (px)", fontsize=9)
    ax.set_ylabel("boundary distance (px)", fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=6.6, loc="upper left")
    ax.tick_params(labelsize=8)
    ax.set_title("(d) checked against a known offset:\n"
                 f"flat edges track $\\delta$ to ${flat_dev:.3f}$ px",
                 fontsize=9.5, pad=6)

    # (e) the exponential
    ax = fig.add_subplot(gs[1, 0])
    xs = np.linspace(0, 24, 400)
    for tau, style, lw, col in ((1.0, ":", 1.1, "#999999"), (3.0, "-", 2.1, "#2b6cb0"), (10.0, "--", 1.2, "#999999")):
        ax.plot(xs, np.exp(-xs / tau), style, lw=lw, color=col,
                label=rf"$\tau = {tau:g}$ px" + ("  (used)" if tau == 3.0 else ""))
    ax.plot(sweep_cd, np.exp(-sweep_cd / TAU), "o", ms=3.5, color="#c05621")
    ax.set_xlabel("Chamfer distance (px)", fontsize=9)
    ax.set_ylabel(r"$s_E = \exp(-\mathrm{cd}/\tau)$", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.6, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.set_title("(e) mean distance $\\to$ score,\nthrough a sharp exponential", fontsize=9.5, pad=6)

    # (f) the tolerance-based diagnostic
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(sweep_d, sweep_f1, "o-", ms=3.5, lw=1.8, color="#38a169",
            label="boundary $F_1$ at 2 px tolerance")
    ax.axvline(2.0, color="#888888", ls=":", lw=1.2)
    ax.annotate("tolerance", (2.0, 0.45), xytext=(3.4, 0.45), fontsize=7.6,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#666666"))
    ax.set_xlabel("known offset $\\delta$ (px)", fontsize=9)
    ax.set_ylabel("boundary $F_1$", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.6, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.set_title("(f) alongside it: a tolerance-based $F_1$,\nwhich saturates and is all-or-nothing", fontsize=9.3, pad=6)

    # (g) the outlier case
    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(overlay(gt, stub), interpolation="nearest")
    bare(ax, "(g) one detached part, everything else perfect")

    # (h) mean versus tail
    ax = fig.add_subplot(gs[1, 3])
    w = 0.36
    x = np.arange(2)
    mean_vals = [chamfer, bm_stub["chamfer_px"]]
    p95_vals = [bm["hausdorff95_px"], bm_stub["hausdorff95_px"]]
    ax.bar(x - w / 2, mean_vals, w, color="#2b6cb0", label="Chamfer mean")
    ax.bar(x + w / 2, p95_vals, w, color="#c05621", label="95th percentile")
    for xi, v in zip(x - w / 2, mean_vals):
        ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, p95_vals):
        ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["clean offset\n$\\delta=6$ px", "offset + one\nstray part"], fontsize=8)
    ax.set_ylabel("distance (px)", fontsize=9)
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    ax.legend(fontsize=7.6, loc="upper left")
    ax.tick_params(labelsize=8)
    ax.set_title("(h) why the tail is reported too", fontsize=9.5, pad=6)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.94, bottom=0.11)
    path = out_dir / "metric-edge.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "metric-edge.png", dpi=140)
    plt.close(fig)

    return path, {
        "case": "two 16 px bands, offset 6 px",
        "gt_boundary_px": int(bg.sum()), "pred_boundary_px": int(bp.sum()),
        "mean_pred_to_gt_px": mean_pg, "mean_gt_to_pred_px": mean_gp,
        "chamfer_px": chamfer, "s_edge": float(np.exp(-chamfer / TAU)),
        "hausdorff95_px": bm["hausdorff95_px"], "boundary_f1_at_2px": bm["boundary_f1"],
        "flat_edge_max_deviation_px": flat_dev,
        "flat_edge_staircase_step_px": flat_step,
        "outlier_case_chamfer_px": bm_stub["chamfer_px"],
        "outlier_case_hausdorff95_px": bm_stub["hausdorff95_px"],
        "outlier_case_mean_directed_px": stub_mean,
        "sweep": [{"delta_px": float(d), "chamfer_px": float(c), "boundary_f1": float(f),
                   "hausdorff95_px": float(p)}
                  for d, c, f, p in zip(sweep_d, sweep_cd, sweep_f1, sweep_p95)],
        "flat_edge_sweep": [{"delta_px": float(d), "flat_mean_px": float(m)}
                            for d, m in zip(fine_d, fine_flat)],
    }


# ---------------------------------------------------------------- 3. topology


def _topology_pair(gt_sub: Sequence[SubPath], pr_sub: Sequence[SubPath]):
    gt, pr = render(gt_sub), render(pr_sub)
    return gt, pr, topology_counts(gt >= 0.5), topology_counts(pr >= 0.5)


def figure_topology(out_dir: Path) -> Tuple[Path, Dict[str, object]]:
    """Cases whose component and hole counts are known by construction."""
    ring = [disc(128, 128, 74), disc(128, 128, 34, cw=False)]
    ring_blob = [disc(128, 128, 74)]
    two_parts = [rect(52, 100, 116, 156), rect(140, 100, 204, 156)]
    merged = [rect(52, 100, 204, 156)]
    three = [disc(60, 128, 30), disc(128, 128, 30), disc(196, 128, 30)]
    one_of_three = [disc(128, 128, 30)]

    cases = [
        ("hole preserved", ring, ring),
        ("hole filled in", ring, ring_blob),
        ("parts merged", two_parts, merged),
        ("parts missing", three, one_of_three),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(12.4, 7.2))
    detail: List[Dict[str, object]] = []
    for j, (label, gt_sub, pr_sub) in enumerate(cases):
        gt, pr, gc, pc = _topology_pair(gt_sub, pr_sub)
        gcn, ghn = paint_components(axes[0][j], gt >= 0.5, "truth" if j == 0 else "")
        pcn, phn = paint_components(axes[1][j], pr >= 0.5, "answer" if j == 0 else "")
        delta = abs(gcn - pcn) + abs(ghn - phn)
        s_t = 1.0 / (1.0 + delta)
        m = compare(gt, pr)
        axes[1][j].set_xlabel(
            f"$\\Delta = |{gcn}-{pcn}|+|{ghn}-{phn}| = {delta}$\n"
            f"$s_T = 1/(1+{delta}) = {s_t:.4f}$\n"
            f"soft IoU $= {m.soft_iou:.3f}$, area ratio $= {m.area_ratio:.3f}$",
            fontsize=8.0)
        axes[1][j].set_title(label, fontsize=9.2, pad=5)
        detail.append({
            "case": label, "gt_components": gcn, "gt_holes": ghn,
            "pred_components": pcn, "pred_holes": phn, "delta": delta, "s_topology": s_t,
            "soft_iou": float(m.soft_iou), "area_ratio": float(m.area_ratio),
        })

    fig.legend(handles=[
        Patch(color=COMPONENT_COLORS[0], label="separate ink regions (components)"),
        Patch(color=HOLE_COLOR, label="enclosed holes"),
        Patch(color="white", label="background"),
    ], fontsize=8.4, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.008))

    fig.subplots_adjust(left=0.02, right=0.985, top=0.94, bottom=0.115, hspace=0.20, wspace=0.06)
    path = out_dir / "metric-topology.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "metric-topology.png", dpi=140)
    plt.close(fig)
    return path, {"cases": detail}


# ---------------------------------------------------------------- 4. alignment

F_PTS = [(74, 66), (156, 66), (156, 92), (102, 92), (102, 124),
         (142, 124), (142, 148), (102, 148), (102, 194), (74, 194)]


def _f_points(dx: float = 0.0, dy: float = 0.0, scale: float = 1.0,
              rotate_deg: float = 0.0, mirror: bool = False) -> SubPath:
    """The ``F`` outline, deliberately chiral, under a known affine map."""
    cx = sum(p[0] for p in F_PTS) / len(F_PTS)
    cy = sum(p[1] for p in F_PTS) / len(F_PTS)
    th = np.deg2rad(rotate_deg)
    pts = []
    for x, y in F_PTS:
        u, v = (x - cx) * scale, (y - cy) * scale
        if mirror:
            u = -u
        pts.append((cx + dx + u * np.cos(th) - v * np.sin(th),
                    cy + dy + u * np.sin(th) + v * np.cos(th)))
    return SubPath(pts, True)


def _f_shape() -> np.ndarray:
    return render([_f_points()])


def _apply(answer: Sequence[SubPath], m: Matrix, fill_rule: str = "nonzero") -> np.ndarray:
    """The exact rendering of geometry under a fitted affine matrix.

    Transforming the geometry and re-rendering is what the scorer does; shifting
    the pixels would resample and blur them, and would misreport the residual.
    """
    return render([sp.transformed(m) for sp in answer], fill_rule)


def _score_variant(gt: np.ndarray, answer: Sequence[SubPath], mode: Optional[str],
                   fill_rule: str = "nonzero") -> Tuple[float, np.ndarray, Optional[Matrix]]:
    """Score ``answer`` against ``gt``; ``mode=None`` means no alignment at all."""
    raw = render(list(answer), fill_rule)
    if mode is None:
        return float(compare(gt, raw).score), raw, None
    al = align_similarity(boundary_parsed(raw) or answer[0], gt, float(raw.sum()), mode=mode)
    moved = _apply(answer, al.matrix, fill_rule)
    return float(compare(gt, moved).score), moved, al.matrix


def figure_align(out_dir: Path) -> Tuple[Path, Dict[str, object]]:
    """What is forgiven, and what is not, on a deliberately chiral shape."""
    gt_cov = _f_shape()
    displaced = lambda **kw: _f_points(dx=34.0, dy=20.0, scale=1.28, **kw)  # noqa: E731

    variants = [
        ("no alignment\nat all", displaced(), None),
        ("translation fitted", displaced(), "translate"),
        ("translation + scale\n(what the harness does)", displaced(), "similarity"),
        ("+ rotation, if it\nwere forgiven", displaced(rotate_deg=24.0), "translate"),
        ("mirrored, if it\nwere forgiven", displaced(mirror=True), "translate"),
    ]
    rows = []
    fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.5))
    for j, (label, answer, mode) in enumerate(variants):
        s, shown, m = _score_variant(gt_cov, [answer], mode)
        axes[j].imshow(overlay(gt_cov, shown), interpolation="nearest")
        bare(axes[j], f"{label}\nscore $= {s:.3f}$")
        rows.append({"treatment": label.replace("\n", " "), "score": s,
                     "fitted_scale": (None if m is None else float(m.a)),
                     "fitted_offset": (None if m is None else [float(m.e), float(m.f)]),
                     "aligned_in_harness": bool(mode == "similarity")})

    fig.legend(handles=list(legend_handles()), fontsize=8.2, loc="lower center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("One answer under five treatments: placement and size are recovered, "
                 "orientation and mirroring are refused", fontsize=10.0, y=0.985)
    fig.subplots_adjust(left=0.02, right=0.985, top=0.76, bottom=0.155, wspace=0.05)
    path = out_dir / "metric-align.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "metric-align.png", dpi=140)
    plt.close(fig)

    # A symmetric figure is the exception: rotating it by 180 degrees changes nothing.
    sym_gt = render([disc(128, 128, 60)])
    sym_rot = render([disc(128, 128, 60)])
    rows.append({"treatment": "rotationally symmetric figure, rotated 180 deg",
                 "score": float(compare(sym_gt, sym_rot).score),
                 "note": "orientation error is unmeasurable on a chiral-free item"})
    return path, {"f_shape": rows}


# ---------------------------------------------------------------- 5. anchors


def _pick_item(pack_dir: Path, preferred: Optional[str] = None) -> Tuple[Item, str]:
    public = load_public(pack_dir)
    for task in public:
        iid = str(task["item_id"])
        if preferred is None or iid == preferred:
            return load_hidden(pack_dir, iid), str(task["input_png"])
    raise SystemExit("no items in pack")


def figure_anchors(out_dir: Path, pack_dir: Path, preferred: Optional[str]) -> Tuple[Path, Dict[str, object]]:
    """The floors and the ceiling, on one item whose ground truth is fixed."""
    item, rel_png = _pick_item(pack_dir, preferred)
    gt = gt_coverage(item)
    png = np.asarray(Image.open(pack_dir / rel_png).convert("L"))

    labels = ["blank", "centroid_dot", "bbox", "ellipse"]
    floors = floor_coverages(gt)
    covs: List[Tuple[str, np.ndarray]] = [(n, floors[n]) for n in labels if n in floors]

    # Control: the truth itself, displaced by (30, 0), then scored as a prediction.
    parsed = parse_path(item.gt_d, fill_rule=item.gt_fill_rule)
    shifted_sub = [sp.transformed(Matrix(e=30.0)) for sp in parsed.subpaths]
    shifted = render(shifted_sub, parsed.fill_rule)
    s_shift, _, _ = _score_variant(gt, shifted_sub, "similarity", parsed.fill_rule)

    # Ceiling: the tracer, on the same pixels the model received, at the same budget.
    cd_text, cd_cmds = trace_ceiling(png, item.k, canvas=item.canvas)
    cov_t, _, _, _ = load_prediction(cd_text, size=item.canvas, ss=SS)
    merged, doc = _merged_parsed(cd_text, "nonzero")
    al = align_similarity(boundary_parsed(cov_t) or merged, gt, float(cov_t.sum()), mode="similarity")
    cov_a = _render_aligned(merged, doc, al.matrix, item.canvas, "nonzero")
    ceiling = float(compare(gt, cov_a).score)

    ladder = [(n, float(compare(gt, c).score)) for n, c in
              [("blank", floors["blank"]), ("centroid_dot", floors["centroid_dot"]),
               ("ellipse", floors["ellipse"]), ("bbox", floors["bbox"])]]
    ladder.append(("sanity_shift $(30,0)$ px", s_shift))
    ladder.append(("ceiling $C(k)$: tracer at $k=%s$" % (item.k if item.k is not None else "inf"), ceiling))
    ladder.append(("identity: the truth itself", float(compare(gt, gt).score)))

    fig = plt.figure(figsize=(12.4, 6.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.05], hspace=0.30, wspace=0.07)
    panels = [("the hidden ground truth", gt, None)] + [(n, c, None) for n, c in covs] + [
        ("control: truth shifted $(30,0)$", shifted, s_shift),
        ("ceiling: tracer on the input PNG", cov_a, ceiling),
    ]
    for i, (title, cov, shown_score) in enumerate(panels[:8]):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        if title.startswith("control"):
            # The shift is visible, and the number printed is the one the ladder
            # reports: after alignment. Showing the unaligned score here would
            # contradict the control's own definition.
            ax.imshow(overlay(gt, cov), interpolation="nearest")
        else:
            ax.imshow(1.0 - cov, cmap="gray", interpolation="nearest")
        sc = float(compare(gt, cov).score) if shown_score is None else float(shown_score)
        extra = f"\n$S = {sc:.3f}$" if shown_score is None else f"\n$S = {sc:.3f}$ (after alignment)"
        bare(ax, f"{title}{extra}")

    ax = fig.add_subplot(gs[2, :])
    names = [n for n, _ in ladder]
    vals = [v for _, v in ladder]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    cols = ["#c53030" if n.startswith(("blank", "centroid", "ellipse", "bbox")) else
            ("#2b6cb0" if n.startswith("ceiling") else "#38a169") for n in names]
    ax.barh(range(len(vals)), vals, color=cols, height=0.62)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=8.2)
    for i, v in enumerate(vals):
        ax.text(v + 0.012, i, f"{v:.3f}", va="center", fontsize=8.2)
    ax.set_xlim(0, 1.09)
    ax.set_xlabel("score $S$ on this item", fontsize=9)
    ax.grid(alpha=0.25, axis="x", lw=0.6)
    ax.tick_params(labelsize=8)
    ax.set_title("the ladder a score is read against: floors, control, ceiling, identity",
                 fontsize=9.6, pad=6)
    lo = max(v for n, v in ladder if n in ("bbox", "ellipse", "centroid_dot", "blank"))
    ax.axvspan(lo, ceiling, color="#2b6cb0", alpha=0.07)
    ax.annotate(f"discriminating band\n$[{lo:.3f},\\ {ceiling:.3f}]$", ((lo + ceiling) / 2, len(vals) - 0.35),
                ha="center", fontsize=8.0, color="#2b4a80")

    fig.subplots_adjust(left=0.055, right=0.985, top=0.955, bottom=0.09)
    path = out_dir / "metric-anchors.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "metric-anchors.png", dpi=140)
    plt.close(fig)

    return path, {
        "item_id": item.item_id, "stratum": item.stratum, "k": item.k,
        "ceiling_commands": cd_cmds,
        "ladder": [{"anchor": n, "score": v} for n, v in ladder],
        "discriminating_band": [lo, ceiling],
        "max_floor": lo,
    }


# ---------------------------------------------------------------- 6. composite

WEIGHTS = {"region": 0.55, "edge": 0.35, "topology": 0.10}


def figure_composite(out_dir: Path) -> Tuple[Path, Dict[str, object]]:
    """How the three terms combine, and what each is worth when it fails."""
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9))

    ax = axes[0]
    ks = list(WEIGHTS)
    vs = [WEIGHTS[k] for k in ks]
    ax.bar(ks, vs, color=["#2b6cb0", "#c05621", "#38a169"], width=0.55)
    for i, v in enumerate(vs):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("weight in the composite", fontsize=9)
    ax.set_ylim(0, 0.66)
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    ax.tick_params(labelsize=8.5)
    ax.set_title("weights: the most a term can change the score", fontsize=9.6, pad=6)

    ax = axes[1]
    s = np.linspace(0, 1, 200)
    for name, col in (("region", "#2b6cb0"), ("edge", "#c05621"), ("topology", "#38a169")):
        others = sum(w for k, w in WEIGHTS.items() if k != name)
        ax.plot(s, others + WEIGHTS[name] * s, lw=1.9, color=col,
                label=f"$s_{{{name}}}$ varies, other two perfect")
    ax.plot([0, 1], [1, 1], ":", lw=1.0, color="#888888")
    ax.set_xlabel("value of the one varying term", fontsize=9)
    ax.set_ylabel("composite $S$", fontsize=9)
    ax.set_xticks([0, 0.5, 1.0])
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=7.6, loc="lower right")
    ax.tick_params(labelsize=8)
    ax.set_title("the composite is a plain weighted sum", fontsize=9.6, pad=6)

    ax = axes[2]
    names = ["region\nfails", "edge\nfails", "topology\nfails"]
    costs = [1.0 - WEIGHTS["region"], 1.0 - WEIGHTS["edge"], 1.0 - WEIGHTS["topology"]]
    ax.bar(names, costs, color=["#2b6cb0", "#c05621", "#38a169"], width=0.55)
    for i, v in enumerate(costs):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("score left when that term is 0", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.25, axis="y", lw=0.6)
    ax.tick_params(labelsize=8.5)
    ax.set_title("cost of one term failing completely", fontsize=9.6, pad=6)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.85, bottom=0.145, wspace=0.24)
    path = out_dir / "metric-composite.pdf"
    fig.savefig(path)
    fig.savefig(out_dir / "metric-composite.png", dpi=140)
    plt.close(fig)
    return path, {
        "weights": WEIGHTS,
        "score_if_one_term_fails": costs,
        "note": "topology is worth 0.10: a coarse integer term that should not swamp geometry",
    }


# ---------------------------------------------------------------- driver


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--out", default="docs/figures")
    ap.add_argument("--item", default="01001100afd9128f")
    ap.add_argument("--json-out", default="data/analysis/metric-primers.json")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1, d1 = figure_region(out_dir)
    p2, d2 = figure_edge(out_dir)
    p3, d3 = figure_topology(out_dir)
    p4, d4 = figure_align(out_dir)
    p5, d5 = figure_anchors(out_dir, Path(args.pack), args.item)
    p6, d6 = figure_composite(out_dir)

    blob = {"region": d1, "edge": d2, "topology": d3, "alignment": d4,
            "anchors": d5, "composite": d6}
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(blob, indent=2), encoding="utf-8")

    for p in (p1, p2, p3, p4, p5, p6):
        print(f"wrote {p}")
    print(f"wrote {args.json_out}\n")

    print("region  : soft IoU error vs closed form "
          f"{d1['max_abs_error_soft_vs_analytic']:.5f}; Dice error "
          f"{d1['max_abs_error_dice_vs_analytic']:.5f}; hard IoU has "
          f"{d1['hard_iou_distinct_plateaus']} plateaus; area ratio stays in "
          f"[{d1['area_ratio_min']:.4f}, {d1['area_ratio_max']:.4f}] at every offset")
    print(f"edge    : chamfer {d2['chamfer_px']:.3f} px -> s_E {d2['s_edge']:.4f}; "
          f"p95 {d2['hausdorff95_px']:.2f} px; F1@2px {d2['boundary_f1_at_2px']:.4f}")
    print(f"          flat-edge check vs the known offset: max deviation "
          f"{d2['flat_edge_max_deviation_px']:.3f} px, in steps of "
          f"{d2['flat_edge_staircase_step_px']:.3f} px (threshold quantisation)")
    print(f"          outlier case: chamfer {d2['outlier_case_chamfer_px']:.3f} px but "
          f"p95 {d2['outlier_case_hausdorff95_px']:.2f} px")
    print("topology:")
    for c in d3["cases"]:
        print(f"          {c['case']:16s} truth {c['gt_components']}c/{c['gt_holes']}h vs answer "
              f"{c['pred_components']}c/{c['pred_holes']}h -> delta {c['delta']} s_T {c['s_topology']:.4f} "
              f"(soft IoU {c['soft_iou']:.3f})")
    print("alignment:")
    for r in d4["f_shape"]:
        extra = f" scale={r.get('fitted_scale')}" if "fitted_scale" in r else ""
        print(f"          {r['treatment']:58s} score {r['score']:.4f}{extra}")
    print(f"anchors : item {d5['item_id'][:8]} ladder "
          + ", ".join(f"{a['anchor'].split(' ')[0]}={a['score']:.3f}" for a in d5["ladder"]))
    print(f"          discriminating band [{d5['discriminating_band'][0]:.3f}, "
          f"{d5['discriminating_band'][1]:.3f}]")
    print("composite: " + ", ".join(f"{k}={v}" for k, v in d6["weights"].items()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
