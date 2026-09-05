"""The response matrix: does the instrument respond the way the specification says it must?

This is the replacement for a human rater panel (\S\\ref{sec:validation}). The
target here is a hidden path the harness holds, so fidelity is computable and no
opinion is required. What still has to be *demonstrated* is that the scorer
behaves correctly on known inputs. Each row below has a pre-declared expected
response, and a row that disagrees is a bug in the instrument rather than a
finding about a model:

===================  ==========================  ===============================
perturbation         expected response           why
===================  ==========================  ===============================
identity             exactly 1.000               reflexivity
translate ±(30,0)    ~1.000                      placement is forgiven
translate ±(0,30)    ~1.000                      placement is forgiven
scale 0.80 / 1.25    ~1.000                      scale is forgiven
rotate 90 / 180      < 0.60                      orientation is *not* forgiven
reflect              < 0.80                      mirroring is *not* forgiven
===================  ==========================  ===============================

The asymmetry is deliberate and is a design commitment: forgiving rotation would
let a model that recalled a mirrored copy score as having drawn the figure, and
for an icon family up and down are different objects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .align import align_similarity
from .metrics import compare
from .pathdata import Matrix, parse_path
from .raster import render_coverage
from .score import gt_coverage
from .tasks import SS, load_hidden, load_public


def _score_under(gt_cov: np.ndarray, parsed, matrix: Matrix, canvas: int, fit: bool = True) -> float:
    """Render the prediction under ``matrix``, optionally forgive placement, then score.

    The subtlety that matters: the aligner estimates the prediction's centroid
    from the point cloud it is given, so it must be handed the *already
    transformed* geometry. Passing the untransformed path while the coverage
    comes from the transformed one makes the fit meaningless: the aligner
    recentres on the wrong place and converges to the identity, which is exactly
    the failure this function originally had. The two must describe the same
    shape.
    """
    moved = parsed.transformed(matrix)
    cov = render_coverage(moved.subpaths, size=canvas, ss=SS, matrix=None, fill_rule=moved.fill_rule)
    if cov.sum() <= 0.0:
        return 0.0
    if fit:
        al = align_similarity(moved, gt_cov, float(cov.sum()), mode="similarity")
        cov = render_coverage(moved.subpaths, size=canvas, ss=SS, matrix=al.matrix, fill_rule=moved.fill_rule)
    return float(compare(gt_cov, cov).score)


def response_matrix(pack_dir: Path | str, item_ids: Optional[Sequence[str]] = None, canvas_probe: int = 512) -> List[Dict[str, object]]:
    """Run the full battery over the selected items."""
    pack_dir = Path(pack_dir)
    tasks = load_public(pack_dir)
    if item_ids:
        wanted = set(item_ids)
        tasks = [t for t in tasks if t["item_id"] in wanted]

    rows: List[Dict[str, object]] = []
    for task in tasks:
        item = load_hidden(pack_dir, str(task["item_id"]))
        gt_cov = gt_coverage(item)
        parsed = parse_path(item.gt_d, fill_rule=item.gt_fill_rule)
        c = item.canvas

        cases = [
            ("identity", Matrix(), 0.999, 1.0001),
            ("translate +30x", Matrix.translate(30.0, 0.0), 0.97, 1.0001),
            ("translate -30x", Matrix.translate(-30.0, 0.0), 0.97, 1.0001),
            ("translate +30y", Matrix.translate(0.0, 30.0), 0.97, 1.0001),
            ("translate +17x -23y", Matrix.translate(17.0, -23.0), 0.97, 1.0001),
            ("scale 0.80", Matrix.scale(0.80, cx=c / 2.0, cy=c / 2.0), 0.95, 1.0001),
            ("scale 1.25", Matrix.scale(1.25, cx=c / 2.0, cy=c / 2.0), 0.95, 1.0001),
            ("rotate 90", Matrix.rotate(90.0, c / 2.0, c / 2.0), 0.0, 0.60),
            ("rotate 180", Matrix.rotate(180.0, c / 2.0, c / 2.0), 0.0, 0.60),
            ("reflect x", Matrix(-1.0, 0.0, 0.0, 1.0, float(c), 0.0), 0.0, 0.80),
            ("reflect y", Matrix(1.0, 0.0, 0.0, -1.0, 0.0, float(c)), 0.0, 0.80),
        ]
        for name, mat, lo, hi in cases:
            raw = _score_under(gt_cov, parsed, mat, c, fit=False)
            fitted = _score_under(gt_cov, parsed, mat, c, fit=True)
            # Symmetry guard. "Rotation must be refused" is a claim about the
            # *scorer*, but it can only be tested on a figure that rotation
            # actually changes. A rotationally symmetric item (a circular plug,
            # a plus sign) scores $\approx 1$ under rotation because the two
            # figures are the same object, and reading that as a failure would be
            # wrong. The item's own symmetry group is measured here from the
            # ground truth, and cases that the item is invariant under are marked
            # not applicable rather than passed or failed.
            self_score = raw if name.startswith(("rotate", "reflect")) else None
            applicable = True
            if self_score is not None and self_score >= 0.95:
                applicable = False
            rows.append({
                "item_id": item.item_id,
                "case": name,
                "score_no_fit": raw,
                "score_fitted": fitted,
                "expected_lo": lo,
                "expected_hi": hi,
                "self_score": self_score,
                "applicable": applicable,
                "pass": bool((lo <= fitted <= hi) if applicable else True),
            })
    return rows


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Aggregate the battery into pass/fail per case, averaged over items."""
    by_case: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_case.setdefault(str(r["case"]), []).append(r)
    out: Dict[str, object] = {}
    for case, rs in by_case.items():
        used = [r for r in rs if r.get("applicable", True)]
        fitted = np.array([float(r["score_fitted"]) for r in used]) if used else np.array([np.nan])
        raw = np.array([float(r["score_no_fit"]) for r in used]) if used else np.array([np.nan])
        out[case] = {
            "n": len(used),
            "n_skipped_symmetric": len(rs) - len(used),
            "mean_fitted": float(fitted.mean()),
            "min_fitted": float(np.nanmin(fitted)),
            "max_fitted": float(np.nanmax(fitted)),
            "mean_no_fit": float(raw.mean()),
            "n_pass": int(sum(1 for r in used if r["pass"])),
            "all_pass": bool(all(r["pass"] for r in used)),
        }
    return out


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Response-matrix diagnostic for the ToySVG scorer.")
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    ids = [s for s in (args.only or "").split(",") if s.strip()] or None
    rows = response_matrix(args.pack, ids)
    summary = summarize(rows)

    w = 20
    print(f"{'case':22s} {'mean(no fit)':>12s} {'mean(fitted)':>13s} {'min':>7s} {'max':>7s}  {'verdict'}")
    print("-" * 78)
    for case, s in summary.items():
        if not s["n"]:
            print(f"{case:22s} {'-':>12s} {'-':>13s} {'-':>7s} {'-':>7s}  N/A (all items symmetric under it)")
            continue
        verdict = "PASS" if s["all_pass"] else f"FAIL ({s['n_pass']}/{s['n']})"
        if s.get("n_skipped_symmetric"):
            verdict += f"  [{s['n_skipped_symmetric']} symmetric item(s) skipped]"
        print(f"{case:22s} {s['mean_no_fit']:12.4f} {s['mean_fitted']:13.4f} {s['min_fitted']:7.4f} {s['max_fitted']:7.4f}  {verdict}")

    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0 if all(v["all_pass"] for v in summary.values() if v["n"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
