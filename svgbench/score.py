"""Scoring: turn stored answers into a decomposed, gated, anchored table.

This module is the only place that reads ground truth. It applies the output
contract, the failure taxonomy, the alignment, the three metric terms, the
reference floors and the tracer ceiling, and writes a per-item report.

Order of operations matters and is fixed: **gate, then align, then measure**.

* Gating first, because an unparsable answer has no geometry to align, and
  because the taxonomy must be applied to what the model wrote rather than to
  what a repair step might have salvaged from it.
* Alignment before measurement, because the benchmark forgives placement and
  scale by design, and measuring first would score the model on the offset.
* The ceiling is traced from *the same input pixels the model received*, so the
  ceiling is a property of the item and not of the model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .align import align_similarity
from .metrics import compare, topology_counts, zero_metrics
from .pathdata import ParsedPath, PathParseError, SubPath, parse_path, simplify_polyline
from .raster import coverage_to_uint8, render_coverage
from .svgdoc import load_prediction, parse_svg_document, render_layers
from .tasks import CANVAS, SS, Item, load_hidden, load_public

# ---------------------------------------------------------------- taxonomy

GATE_OK = "ok"
GATE_EMPTY = "empty"
GATE_UNPARSABLE = "unparsable"
GATE_DEGENERATE = "degenerate"
GATE_OVER_BUDGET = "over_budget"
GATE_OFF_CANVAS = "off_canvas"
GATE_NO_SVG = "no_svg"
GATE_TRUNCATED = "truncated"

TRUNCATED_FINISH_REASONS = {"MAX_TOKENS", "length", "max_tokens", "LENGTH"}


# ---------------------------------------------------------------- geometry helpers


def gt_coverage(item: Item) -> np.ndarray:
    parsed = parse_path(item.gt_d, fill_rule=item.gt_fill_rule)
    return render_coverage(parsed.subpaths, size=item.canvas, ss=SS, fill_rule=parsed.fill_rule)


def _merged_parsed(text: str, fill_rule: str) -> Tuple[Optional[ParsedPath], Optional[object]]:
    """Parse an answer into a single ParsedPath for alignment purposes."""
    if "<" in text and "path" in text.lower():
        doc = parse_svg_document(text)
        if not doc.layers:
            return None, None
        subpaths = []
        for layer in doc.layers:
            subpaths.extend(layer.parsed.subpaths)
        merged = ParsedPath(d=text, subpaths=subpaths, commands=[], fill_rule=fill_rule)
        return merged, doc
    return parse_path(text, fill_rule=fill_rule), None


def _render_aligned(parsed: ParsedPath, doc, matrix, canvas: int, fill_rule: str) -> np.ndarray:
    if doc is not None:
        return render_layers(doc, size=canvas, ss=SS, matrix=matrix)
    return render_coverage(parsed.subpaths, size=canvas, ss=SS, matrix=matrix, fill_rule=fill_rule)


def boundary_parsed(coverage: np.ndarray, min_len: int = 8) -> Optional[ParsedPath]:
    """The 0.5-level contour of a rendered answer, as a path of boundary points.

    Alignment must use the prediction's *ink boundary*, not its path points, and
    the difference is not subtle. A model that answers with a stroke writes a
    centreline; the ground truth is a thin band whose distance field is zero on
    the band's two *edges*. Fitting a centreline to that field pulls it onto an
    edge, a systematic error of half the stroke width: measured at $8$\,px on a
    $16$\,px stroke, which is enough on its own to collapse the overlap score.
    Contouring the rendered ink fixes this for strokes and fills alike, because
    after rendering there is only one thing: the boundary of the ink.
    """
    from skimage import measure

    contours = measure.find_contours(np.asarray(coverage, dtype=np.float64), 0.5)
    subs: List[SubPath] = []
    for c in contours:
        if len(c) < min_len:
            continue
        # find_contours yields (row, col); canvas coordinates put the centre of
        # pixel (i, j) at (j + 0.5, i + 0.5).
        subs.append(SubPath([(float(col) + 0.5, float(row) + 0.5) for row, col in c], True))
    if not subs:
        return None
    return ParsedPath(d="", subpaths=subs, commands=[], fill_rule="nonzero")


def _answered_geometry(parsed: ParsedPath, doc, item: Item,
                       gt_cov: np.ndarray) -> Tuple[np.ndarray, Optional[object], Optional[np.ndarray]]:
    """Render an answer, fit it, and render it again as the measurement sees it.

    ``(coverage_unaligned, alignment, coverage_aligned)``; the alignment is ``None``
    when the answer renders no ink at all. Split out of :func:`score_item` so that a
    preview renders through the same three statements as the score, rather than
    through a lookalike that could drift from it.
    """
    cov_plain = _render_aligned(parsed, doc, None, item.canvas, parsed.fill_rule)
    if cov_plain.sum() <= 0.0:
        return cov_plain, None, None
    fit_input = boundary_parsed(cov_plain) or parsed
    al = align_similarity(fit_input, gt_cov, float(cov_plain.sum()), mode="similarity")
    cov_aligned = _render_aligned(parsed, doc, al.matrix, item.canvas, parsed.fill_rule)
    return cov_plain, al, cov_aligned


def render_answer(text: str, item: Item, align: bool = True) -> Optional[np.ndarray]:
    """The answer as the scorer measured it, for previews and reports.

    ``None`` if the answer yields no geometry. With ``align=False`` it returns the
    unregistered rendering, which is what a figure should use when the point is to
    show *where* the model put the figure rather than how it scored.
    """
    try:
        parsed, doc = _merged_parsed((text or "").strip(), "nonzero")
    except PathParseError:
        return None
    if parsed is None or not parsed.subpaths:
        return None
    cov_plain, _, cov_aligned = _answered_geometry(parsed, doc, item, gt_coverage(item))
    if align:
        return cov_aligned
    return cov_plain


# ---------------------------------------------------------------- floors and ceiling


def floor_coverages(gt_cov: np.ndarray) -> Dict[str, np.ndarray]:
    """The floor anchors, all derived from ground truth alone."""
    canvas = gt_cov.shape[0]
    mask = gt_cov >= 0.5
    out: Dict[str, np.ndarray] = {"blank": np.zeros_like(gt_cov)}
    if not mask.any():
        return out

    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    bbox = np.zeros_like(gt_cov)
    bbox[y0 : y1 + 1, x0 : x1 + 1] = 1.0
    out["bbox"] = bbox

    # Area-matched ellipse from the second moments of the ink.
    mx, my = float(xs.mean()), float(ys.mean())
    dx, dy = xs - mx, ys - my
    cov = np.cov(np.vstack([dx, dy]))
    if np.isfinite(cov).all() and np.linalg.det(cov) > 0:
        vals, vecs = np.linalg.eigh(cov)
        # Scaling so the ellipse area matches the ink area: pi*r1*r2 = area.
        area = float(mask.sum())
        scale = math.sqrt(area / (math.pi * math.sqrt(max(vals[0], 1e-9) * max(vals[1], 1e-9)))) * 2.0
        r1, r2 = math.sqrt(max(vals[0], 1e-9)) * scale, math.sqrt(max(vals[1], 1e-9)) * scale
        yy, xx = np.mgrid[0:canvas, 0:canvas]
        ux, uy = vecs[0, 0], vecs[1, 0]
        vx, vy = vecs[0, 1], vecs[1, 1]
        u = (xx - mx) * ux + (yy - my) * uy
        v = (xx - mx) * vx + (yy - my) * vy
        ell = ((u / r1) ** 2 + (v / r2) ** 2) <= 1.0
        out["ellipse"] = ell.astype(np.float64)

    dot = np.zeros_like(gt_cov)
    r = max(2.0, math.sqrt(float(mask.sum()) / math.pi) * 0.25)
    yy, xx = np.mgrid[0:canvas, 0:canvas]
    dot[((xx - mx) ** 2 + (yy - my) ** 2) <= r * r] = 1.0
    out["centroid_dot"] = dot
    return out


def trace_ceiling(png: np.ndarray, k: Optional[int], canvas: int = CANVAS, tol: float = 0.9) -> Tuple[str, int]:
    """A deterministic tracer applied to the *input pixels*: the ceiling C(k).

    The ceiling answers "what can be achieved from this image without any
    understanding of it". The tracer thresholds the ink, extracts contours at the
    0.5 level, and simplifies them. At a finite budget the simplification
    tolerance is raised until the command count fits, which is exactly the
    trade-off the budget imposes on a model, and therefore the right comparison.

    Returns ``(path_d, n_commands)``.
    """
    from skimage import measure

    ink = np.asarray(png, dtype=np.float64) / 255.0
    # Contours of the *coverage* field, so the result is sub-pixel.
    field = 1.0 - ink
    contours = measure.find_contours(field, 0.5)
    contours = [c for c in contours if len(c) >= 8]

    def emit(tolerance: float) -> Tuple[str, int]:
        parts: List[str] = []
        total = 0
        for c in contours:
            pts = simplify_polyline([(float(x), float(y)) for y, x in c], tolerance)
            if len(pts) < 3:
                continue
            parts.append(f"M{pts[0][0]:.2f} {pts[0][1]:.2f}")
            for x, y in pts[1:]:
                parts.append(f"L{x:.2f} {y:.2f}")
            parts.append("Z")
            total += len(pts) + 1
        return "".join(parts), total

    d, n = emit(tol)
    if k is None:
        return d, n
    if n <= k:
        return d, n
    lo, hi = tol, 24.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        d_mid, n_mid = emit(mid)
        if n_mid <= k:
            hi = mid
        else:
            lo = mid
    return emit(hi)


# ---------------------------------------------------------------- scoring one item


@dataclass
class ItemScore:
    item_id: str
    model: str
    stratum: str = ""
    split: str = ""
    budget: str = ""
    k: Optional[int] = None
    gate: str = GATE_OK
    scored: bool = True
    n_commands: int = 0
    n_subpaths: int = 0
    is_svg: bool = False
    s_region: float = 0.0
    s_edge: float = 0.0
    s_topology: float = 0.0
    score: float = 0.0
    partial_score: Optional[float] = None
    soft_iou: float = 0.0
    hard_iou: float = 0.0
    chamfer_px: float = 0.0
    hausdorff95_px: float = 0.0
    area_ratio: float = 0.0
    gt_components: int = 0
    gt_holes: int = 0
    pred_components: int = 0
    pred_holes: int = 0
    topology_error: int = 0
    align_scale: float = 1.0
    align_tx: float = 0.0
    align_ty: float = 0.0
    ceiling_score: Optional[float] = None
    ceiling_commands: Optional[int] = None
    rho: Optional[float] = None
    floors: Dict[str, float] = field(default_factory=dict)
    tolerant_score: Optional[float] = None
    note: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def score_item(text: str, item: Item, model: str, finish_reason: Optional[str] = None, want_ceiling: bool = True, png_path: Optional[Path] = None) -> ItemScore:
    """Score one stored answer."""
    rec = ItemScore(item_id=item.item_id, model=model, stratum=item.stratum, split=item.split, budget=item.budget, k=item.k)
    gt_cov = gt_coverage(item)
    gt_top = topology_counts(gt_cov >= 0.5)
    rec.gt_components, rec.gt_holes = int(gt_top["components"]), int(gt_top["holes"])

    if want_ceiling and png_path is not None and png_path.exists():
        from PIL import Image

        png = np.asarray(Image.open(png_path).convert("L"))
        cd, cn = trace_ceiling(png, item.k, canvas=item.canvas)
        rec.ceiling_commands = cn
        try:
            cov, _, _, _ = load_prediction(cd, size=item.canvas, ss=SS)
            merged, doc = _merged_parsed(cd, "nonzero")
            if merged is not None:
                al = align_similarity(boundary_parsed(cov) or merged, gt_cov, float(cov.sum()), mode="similarity")
                cov_a = _render_aligned(merged, doc, al.matrix, item.canvas, "nonzero")
                rec.ceiling_score = float(compare(gt_cov, cov_a).score)
        except Exception:
            rec.ceiling_score = None

    for name, cov in floor_coverages(gt_cov).items():
        rec.floors[name] = float(compare(gt_cov, cov).score)
    rec.floors["identity"] = float(compare(gt_cov, gt_cov).score)

    # ---- gates ----
    stripped = (text or "").strip()
    if not stripped:
        rec.gate = GATE_TRUNCATED if (finish_reason or "") in TRUNCATED_FINISH_REASONS else GATE_EMPTY
        rec.scored = False
        m = zero_metrics(gt_cov)
        rec.s_region, rec.s_edge, rec.s_topology, rec.score = 0.0, 0.0, 0.0, m.score
        rec.note = f"finish_reason={finish_reason}"
        return rec
    if (finish_reason or "") in TRUNCATED_FINISH_REASONS:
        rec.note = f"finish_reason={finish_reason}"

    try:
        parsed, doc = _merged_parsed(stripped, "nonzero")
    except PathParseError as exc:
        rec.gate = GATE_UNPARSABLE
        rec.scored = False
        rec.note = str(exc)[:120]
        rec.score = zero_metrics(gt_cov).score
        return rec

    if parsed is None or not parsed.subpaths:
        rec.gate = GATE_UNPARSABLE if doc is None else GATE_NO_SVG
        rec.scored = False
        rec.score = zero_metrics(gt_cov).score
        return rec

    if doc is not None:
        rec.n_commands, rec.n_subpaths, rec.is_svg = doc.n_commands, doc.n_subpaths, True
    else:
        rec.n_commands, rec.n_subpaths = parsed.n_commands, parsed.n_subpaths

    over_budget = item.k is not None and rec.n_commands > item.k

    # ---- alignment, then measurement ----
    cov_plain, al, cov_aligned = _answered_geometry(parsed, doc, item, gt_cov)
    if cov_aligned is None:
        rec.gate = GATE_DEGENERATE
        rec.scored = False
        rec.score = zero_metrics(gt_cov).score
        return rec
    rec.align_scale = float(al.scale)
    rec.align_tx, rec.align_ty = float(al.offset[0]), float(al.offset[1])

    # off-canvas check, before alignment is forgiven
    if cov_plain.sum() <= 0.0 and cov_aligned.sum() <= 0.0:
        rec.gate = GATE_OFF_CANVAS
        rec.scored = False

    m = compare(gt_cov, cov_aligned)
    rec.s_region, rec.s_edge, rec.s_topology, rec.score = m.s_region, m.s_edge, m.s_topology, float(m.score)
    rec.soft_iou, rec.hard_iou = m.soft_iou, m.hard_iou
    rec.chamfer_px, rec.hausdorff95_px = m.chamfer_px, m.hausdorff95_px
    rec.area_ratio = float(m.area_ratio)
    rec.pred_components, rec.pred_holes = m.pred_components, m.pred_holes
    rec.topology_error = m.topology_error

    # the tolerant reading: a stroked element with no fill means "outline"
    try:
        cov_t, _, _, _ = load_prediction(stripped, size=item.canvas, ss=SS, tolerant=True)
        al_t = align_similarity(boundary_parsed(cov_t) or parsed, gt_cov, float(cov_t.sum()), mode="similarity")
        cov_t = _render_plain_tolerant(stripped, al_t.matrix, item.canvas)
        rec.tolerant_score = float(compare(gt_cov, cov_t).score)
    except Exception:
        rec.tolerant_score = None

    if over_budget:
        rec.gate = GATE_OVER_BUDGET
        rec.partial_score = rec.score
        rec.scored = False
        rec.score = 0.0
    else:
        rec.gate = GATE_OK

    if rec.ceiling_score and rec.ceiling_score > 0:
        rec.rho = rec.score / rec.ceiling_score
    return rec


def _render_plain_tolerant(text: str, matrix, canvas: int) -> np.ndarray:
    doc = parse_svg_document(text, tolerant=True)
    return render_layers(doc, size=canvas, ss=SS, matrix=matrix)


# ---------------------------------------------------------------- run scoring


def score_run(
    pack_dir: Path | str,
    run_dir: Path | str,
    out_path: Optional[Path | str] = None,
    want_ceiling: bool = True,
) -> Dict[str, object]:
    pack_dir, run_dir = Path(pack_dir), Path(run_dir)
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    model = f"{manifest.get('provider')}:{manifest.get('model')}"
    results = {r["item_id"]: r for r in manifest.get("results", [])}

    scores: List[ItemScore] = []
    for task in load_public(pack_dir):
        item_id = str(task["item_id"])
        run_rec = results.get(item_id)
        if run_rec is None:
            continue
        text = (run_dir / str(run_rec.get("raw_file", ""))).read_text(encoding="utf-8") if run_rec.get("raw_file") else ""
        item = load_hidden(pack_dir, item_id)
        scores.append(
            score_item(
                text, item, model,
                finish_reason=run_rec.get("finish_reason"),
                want_ceiling=want_ceiling,
                png_path=pack_dir / str(task["input_png"]),
            )
        )

    ok = [s for s in scores if s.scored]
    summary = {
        "model": model,
        "run_id": manifest.get("run_id"),
        "n": len(scores),
        "n_scored": len(ok),
        "gates": _histogram([s.gate for s in scores]),
        "mean_score": float(np.mean([s.score for s in scores])) if scores else None,
        "mean_score_scored_only": float(np.mean([s.score for s in ok])) if ok else None,
        "mean_rho": float(np.mean([s.rho for s in ok if s.rho is not None])) if any(s.rho is not None for s in ok) else None,
        "mean_ceiling": float(np.mean([s.ceiling_score for s in scores if s.ceiling_score is not None])) if any(s.ceiling_score is not None for s in scores) else None,
        "by_stratum": _aggregate(scores, "stratum"),
        "by_split": _aggregate(scores, "split"),
    }
    blob = {
        "run": {k: manifest.get(k) for k in ("run_id", "provider", "model", "prompt_version", "temperature", "thinking_budget", "max_output_tokens", "key_fingerprint")},
        "summary": summary,
        "items": [s.to_dict() for s in scores],
    }
    if out_path:
        Path(out_path).write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return blob


def _histogram(values: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _aggregate(scores: Sequence[ItemScore], key: str) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[ItemScore]] = {}
    for s in scores:
        groups.setdefault(str(getattr(s, key)), []).append(s)
    return {
        k: {
            "n": len(v),
            "mean_score": float(np.mean([x.score for x in v])),
            "mean_region": float(np.mean([x.s_region for x in v])),
            "mean_edge": float(np.mean([x.s_edge for x in v])),
            "mean_topology": float(np.mean([x.s_topology for x in v])),
        }
        for k, v in sorted(groups.items())
    }


def format_table(blob: Dict[str, object]) -> str:
    """A fixed-width table, so the report shape does not change between runs."""
    rows = blob["items"]  # type: ignore[index]
    head = f"{'item':18s} {'split':11s} {'str':4s} {'gate':12s} {'cmds':>5s} {'s_R':>6s} {'s_E':>6s} {'s_T':>6s} {'score':>6s} {'ceil':>6s} {'rho':>6s}"
    lines = [head, "-" * len(head)]
    for r in rows:
        ceil = "-" if r.get("ceiling_score") is None else f"{r['ceiling_score']:.3f}"
        rho = "-" if r.get("rho") is None else f"{r['rho']:.3f}"
        lines.append(
            f"{r['item_id']:18s} {r['split']:11s} {r['stratum']:4s} {r['gate']:12s} {r['n_commands']:5d} "
            f"{r['s_region']:6.3f} {r['s_edge']:6.3f} {r['s_topology']:6.3f} {r['score']:6.3f} {ceil:>6s} {rho:>6s}"
        )
    s = blob["summary"]  # type: ignore[index]
    lines.append("")
    lines.append(f"model {s['model']}  n={s['n']}  scored={s['n_scored']}  mean={s['mean_score']:.4f}  ceiling={s['mean_ceiling']}")
    lines.append(f"gates: {s['gates']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- cli


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score a ToySVG run.")
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-ceiling", action="store_true")
    args = ap.parse_args(argv)

    out = args.out or str(Path(args.run) / "scores.json")
    blob = score_run(args.pack, args.run, out_path=out, want_ceiling=not args.no_ceiling)
    print(format_table(blob))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
