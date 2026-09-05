#!/usr/bin/env python3
"""Print every intermediate quantity between a stored answer and its final score.

A score is only trustworthy if a reader can follow it. This tool runs the exact
scoring path and prints each step with its numbers: command count, ink areas, the
soft intersection and union behind the region term, both directed boundary means
behind the edge term, the topology counts, the three weighted contributions, and
the ceiling they are compared against.

    python -m tools.explain_score --pack data/pack \\
        --run data/runs/gemini-2.5-flash-demo --item 01001100afd9128f
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

from svgbench.align import align_similarity, signed_distance_field
from svgbench.metrics import DEFAULT_CHAMFER_TAU, DEFAULT_EDGE_TOLERANCE, DEFAULT_WEIGHTS, region_boundary, topology_counts
from svgbench.pathdata import parse_path
from svgbench.score import (
    _merged_parsed,
    _render_aligned,
    boundary_parsed,
    gt_coverage,
    score_item,
    trace_ceiling,
)
from svgbench.tasks import SS, load_hidden, load_public


def explain(pack_dir: Path, run_dir: Path, item_id: str) -> Dict[str, object]:
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rec = next(r for r in run["results"] if r["item_id"] == item_id)
    task = next(t for t in load_public(pack_dir) if t["item_id"] == item_id)
    item = load_hidden(pack_dir, item_id)
    text = (run_dir / rec["raw_file"]).read_text(encoding="utf-8").strip()
    png_path = pack_dir / str(task["input_png"])

    # ---- stage 1: parse ----
    parsed, doc = _merged_parsed(text, item.gt_fill_rule)
    n_commands = doc.n_commands if doc is not None else parsed.n_commands
    n_subpaths = doc.n_subpaths if doc is not None else parsed.n_subpaths
    strokes = sorted({l.stroke_width for l in doc.layers if l.stroke_width > 0}) if doc is not None else []
    fills = sorted({l.fill for l in doc.layers}) if doc is not None else []

    # ---- stage 2: render ----
    gt = gt_coverage(item)
    before = _render_aligned(parsed, doc, None, item.canvas, item.gt_fill_rule)

    # ---- stage 3: align ----
    fit = boundary_parsed(before) or parsed
    al = align_similarity(fit, gt, float(before.sum()), mode="similarity")
    pred = _render_aligned(parsed, doc, al.matrix, item.canvas, item.gt_fill_rule)

    # ---- stage 4: the three terms, with their raw inputs ----
    inter = float(np.minimum(gt, pred).sum())
    union = float(np.maximum(gt, pred).sum())
    s_region = inter / union if union > 0 else 1.0

    g = gt >= 0.5
    p = pred >= 0.5
    hard_inter = int((g & p).sum())
    hard_union = int((g | p).sum())
    dice_num, dice_den = 2.0 * inter, float(gt.sum() + pred.sum())

    bg, bp = region_boundary(g), region_boundary(p)
    dt_g = ndimage.distance_transform_edt(~bg)
    dt_p = ndimage.distance_transform_edt(~bp)
    d_pred_to_gt = dt_g[bp]
    d_gt_to_pred = dt_p[bg]
    mean_pred_to_gt = float(d_pred_to_gt.mean())
    mean_gt_to_pred = float(d_gt_to_pred.mean())
    chamfer = 0.5 * (mean_pred_to_gt + mean_gt_to_pred)
    hausdorff95 = max(float(np.percentile(d_pred_to_gt, 95)), float(np.percentile(d_gt_to_pred, 95)))
    s_edge = float(np.exp(-chamfer / DEFAULT_CHAMFER_TAU))

    gt_top, pred_top = topology_counts(g), topology_counts(p)
    topo_error = abs(gt_top["components"] - pred_top["components"]) + abs(gt_top["holes"] - pred_top["holes"])
    s_topology = 1.0 / (1.0 + topo_error)

    w = DEFAULT_WEIGHTS
    contributions = {
        "region": w["region"] * s_region,
        "edge": w["edge"] * s_edge,
        "topology": w["topology"] * s_topology,
    }
    score = sum(contributions.values())

    # ---- stage 5: anchors ----
    png = np.asarray(Image.open(png_path).convert("L"))
    _, cn = trace_ceiling(png, item.k, canvas=item.canvas)
    ceiling = score_item(text, item, "explain", finish_reason=rec.get("finish_reason"), want_ceiling=True, png_path=png_path)

    return {
        "item_id": item_id,
        "stratum": item.stratum,
        "split": item.split,
        "budget": item.budget,
        "k": item.k,
        "model": f"{run['provider']}:{run['model']}",
        "finish_reason": rec.get("finish_reason"),
        "answer": text,
        "stage1_parse": {
            "is_svg_document": doc is not None,
            "n_commands": n_commands,
            "n_subpaths": n_subpaths,
            "stroke_widths": strokes,
            "fill_flags": fills,
            "under_budget": None if item.k is None else n_commands <= item.k,
        },
        "stage2_render": {
            "canvas": item.canvas,
            "supersampling": SS,
            "ink_area_before_alignment_px": float(before.sum()),
            "gt_ink_area_px": float(gt.sum()),
        },
        "stage3_align": {
            "scale": float(al.scale),
            "offset": [float(al.offset[0]), float(al.offset[1])],
            "loss": float(al.loss),
            "ink_area_after_alignment_px": float(pred.sum()),
            "area_ratio": float(pred.sum() / gt.sum()) if gt.sum() > 0 else None,
            "score_if_no_alignment": float(np.minimum(gt, before).sum() / max(1e-9, np.maximum(gt, before).sum())),
        },
        "stage4_terms": {
            "region": {
                "formula": "sum(min(gt,pred)) / sum(max(gt,pred))",
                "soft_intersection": inter,
                "soft_union": union,
                "value": s_region,
                "hard_intersection_px": hard_inter,
                "hard_union_px": hard_union,
                "hard_iou": hard_inter / hard_union if hard_union else 1.0,
                "dice": dice_num / dice_den if dice_den else 1.0,
            },
            "edge": {
                "formula": "exp(-chamfer / tau), tau = %g px" % DEFAULT_CHAMFER_TAU,
                "gt_boundary_pixels": int(bg.sum()),
                "pred_boundary_pixels": int(bp.sum()),
                "mean_pred_boundary_to_gt_px": mean_pred_to_gt,
                "mean_gt_boundary_to_pred_px": mean_gt_to_pred,
                "chamfer_px": chamfer,
                "hausdorff95_px": hausdorff95,
                "boundary_f1_at_%gpx" % DEFAULT_EDGE_TOLERANCE: float(
                    (2 * (d_pred_to_gt <= DEFAULT_EDGE_TOLERANCE).mean() * (d_gt_to_pred <= DEFAULT_EDGE_TOLERANCE).mean())
                    / max(1e-9, (d_pred_to_gt <= DEFAULT_EDGE_TOLERANCE).mean() + (d_gt_to_pred <= DEFAULT_EDGE_TOLERANCE).mean())
                ),
                "value": s_edge,
            },
            "topology": {
                "formula": "1 / (1 + |d_components| + |d_holes|)",
                "gt_components": gt_top["components"],
                "gt_holes": gt_top["holes"],
                "pred_components": pred_top["components"],
                "pred_holes": pred_top["holes"],
                "topology_error": topo_error,
                "value": s_topology,
            },
        },
        "stage5_composite": {
            "weights": w,
            "contributions": contributions,
            "sum_before_rounding": score,
            "score": round(score, 6),
        },
        "anchors": {
            "identity": ceiling.floors.get("identity"),
            "blank": ceiling.floors.get("blank"),
            "bbox": ceiling.floors.get("bbox"),
            "ellipse": ceiling.floors.get("ellipse"),
            "centroid_dot": ceiling.floors.get("centroid_dot"),
            "tracer_ceiling_at_k": ceiling.ceiling_score,
            "tracer_commands_used": cn,
            "parsimony_ratio": ceiling.rho,
        },
        "cross_check": {
            "score_from_score_item": ceiling.score,
            "match": abs(score - ceiling.score) < 1e-9,
        },
    }


def _fmt(x: Optional[float], n: int = 4) -> str:
    return "-" if x is None else f"{x:.{n}f}"


def report(d: Dict[str, object]) -> str:
    s1, s2, s3, s4 = d["stage1_parse"], d["stage2_render"], d["stage3_align"], d["stage4_terms"]
    s5, an = d["stage5_composite"], d["anchors"]
    r, e, t = s4["region"], s4["edge"], s4["topology"]
    out = []
    out.append(f"item {d['item_id']}   {d['split']}  stratum {d['stratum']}  budget {d['budget']} (k={d['k']})")
    out.append(f"model {d['model']}   finish_reason={d['finish_reason']}")
    out.append("")
    out.append("STAGE 1  parse")
    out.append(f"  svg document={s1['is_svg_document']}  commands={s1['n_commands']}  subpaths={s1['n_subpaths']}"
               f"  stroke_widths={s1['stroke_widths']}  fill={s1['fill_flags']}  under budget={s1['under_budget']}")
    out.append("")
    out.append("STAGE 2  render (pinned rasteriser, ss=4)")
    out.append(f"  gt ink area      = {s2['gt_ink_area_px']:.1f} px   (sum of coverage)")
    out.append(f"  answer ink area  = {s2['ink_area_before_alignment_px']:.1f} px  (before alignment)")
    out.append("")
    out.append("STAGE 3  align (translation + uniform scale only)")
    out.append(f"  fitted scale={s3['scale']:.4f}  offset=({s3['offset'][0]:.2f}, {s3['offset'][1]:.2f})  loss={s3['loss']:.4f}")
    out.append(f"  ink after alignment = {s3['ink_area_after_alignment_px']:.1f} px   area ratio = {_fmt(s3['area_ratio'],3)}")
    out.append("")
    out.append("STAGE 4  the three terms")
    out.append(f"  REGION   sum(min)={r['soft_intersection']:.1f}  sum(max)={r['soft_union']:.1f}"
               f"  -> s_R = {r['value']:.4f}")
    out.append(f"           (hard IoU for reference: {r['hard_intersection_px']}/{r['hard_union_px']} = {_fmt(r['hard_iou'],4)}; dice = {_fmt(r['dice'],4)})")
    out.append(f"  EDGE     pred->gt mean = {e['mean_pred_boundary_to_gt_px']:.3f} px   gt->pred mean = {e['mean_gt_boundary_to_pred_px']:.3f} px")
    out.append(f"           chamfer = mean of the two = {e['chamfer_px']:.4f} px   hausdorff95 = {e['hausdorff95_px']:.2f} px")
    out.append(f"           s_E = exp(-{e['chamfer_px']:.4f}/3) = {e['value']:.4f}")
    out.append(f"  TOPOLOGY gt {t['gt_components']} comp / {t['gt_holes']} holes   pred {t['pred_components']} comp / {t['pred_holes']} holes")
    out.append(f"           error = {t['topology_error']}  -> s_T = 1/(1+{t['topology_error']}) = {t['value']:.4f}")
    out.append("")
    out.append("STAGE 5  composite")
    c = s5["contributions"]
    out.append(f"  {s5['weights']['region']:.2f} x {r['value']:.4f} = {c['region']:.6f}   (region)")
    out.append(f"  {s5['weights']['edge']:.2f} x {e['value']:.4f} = {c['edge']:.6f}   (edge)")
    out.append(f"  {s5['weights']['topology']:.2f} x {t['value']:.4f} = {c['topology']:.6f}   (topology)")
    out.append(f"  S = {s5['sum_before_rounding']:.6f}   -> reported {s5['score']:.3f}")
    out.append(f"  cross-check against the scorer's own path: {d['cross_check']['match']}")
    out.append("")
    out.append("ANCHORS  (same pipeline, so the score has a scale)")
    out.append(f"  identity={_fmt(an['identity'],3)}  bbox={_fmt(an['bbox'],3)}  ellipse={_fmt(an['ellipse'],3)}"
               f"  blank={_fmt(an['blank'],3)}  centroid_dot={_fmt(an['centroid_dot'],3)}")
    out.append(f"  tracer ceiling C({d['k']}) = {_fmt(an['tracer_ceiling_at_k'],3)} using {an['tracer_commands_used']} commands")
    out.append(f"  parsimony ratio rho = S / C = {_fmt(an['parsimony_ratio'],3)}")
    return "\n".join(out)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--run", required=True)
    ap.add_argument("--item", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    d = explain(Path(args.pack), Path(args.run), args.item)
    print(report(d))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(d, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
