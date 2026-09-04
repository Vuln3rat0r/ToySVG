"""Shape metrics.

Design notes that matter for fairness:

* **Region** agreement uses *soft* Intersection-over-Union on coverage, not a
  hard threshold. A hard threshold turns a half-pixel shift into a cliff edge;
  soft IoU degrades smoothly, so a model is not punished for subpixel rounding.
* **Edge** agreement uses symmetric Chamfer distance on region boundaries. Two
  drawings can overlap heavily and still be structurally different, so the
  boundary term is not redundant with IoU.
* **Topology** compares connected components and hole counts. Without it, a
  drawing that fills in every hole of an outline icon can score well on region
  overlap while having clearly failed to see the shape.

Every component is reported separately. A single blended number hides which
kind of failure occurred, and this benchmark exists to diagnose seeing.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy import ndimage
from skimage import measure as skmeasure

#: Boundary tolerance in pixels for the boundary-F1 term.
DEFAULT_EDGE_TOLERANCE = 2.0
#: Chamfer length scale in pixels: exp(-chamfer / TAU) is the edge score.
DEFAULT_CHAMFER_TAU = 3.0

#: Blend of the three components in the headline score.
DEFAULT_WEIGHTS = {"region": 0.55, "edge": 0.35, "topology": 0.10}

_STRUCTURE = np.ones((3, 3), dtype=bool)


def soft_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Soft Jaccard on coverage in [0, 1]."""
    inter = float(np.minimum(a, b).sum())
    union = float(np.maximum(a, b).sum())
    if union <= 0.0:
        return 1.0
    return inter / union


def hard_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> float:
    ma, mb = a >= threshold, b >= threshold
    union = int((ma | mb).sum())
    if union == 0:
        return 1.0
    return float((ma & mb).sum()) / union


def dice(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(a.sum() + b.sum())
    if denom <= 0.0:
        return 1.0
    return 2.0 * float(np.minimum(a, b).sum()) / denom


def region_boundary(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a binary region (region minus its erosion)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=_STRUCTURE, border_value=0)
    return mask & ~eroded


def boundary_metrics(a: np.ndarray, b: np.ndarray, tolerance: float = DEFAULT_EDGE_TOLERANCE) -> Dict[str, float]:
    """Symmetric boundary distances and tolerance-based boundary F1.

    Returns distances in pixels plus a normalised Chamfer (fraction of the image
    diagonal) so that numbers are comparable across canvas sizes.
    """
    mas, mbs = a >= 0.5, b >= 0.5
    ba, bb = region_boundary(mas), region_boundary(mbs)
    h, w = a.shape
    diagonal = math.hypot(h, w)

    if not ba.any() and not bb.any():
        return {"chamfer_px": 0.0, "chamfer_norm": 0.0, "hausdorff95_px": 0.0, "boundary_f1": 1.0}
    if not ba.any() or not bb.any():
        # One side drew nothing: maximum disagreement, bounded by the diagonal.
        return {
            "chamfer_px": float(diagonal),
            "chamfer_norm": 1.0,
            "hausdorff95_px": float(diagonal),
            "boundary_f1": 0.0,
        }

    dt_to_a = ndimage.distance_transform_edt(~ba)
    dt_to_b = ndimage.distance_transform_edt(~bb)

    d_b_to_a = dt_to_a[bb]  # prediction boundary -> ground truth boundary
    d_a_to_b = dt_to_b[ba]

    chamfer = 0.5 * (float(d_b_to_a.mean()) + float(d_a_to_b.mean()))
    hausdorff95 = max(float(np.percentile(d_b_to_a, 95)), float(np.percentile(d_a_to_b, 95)))

    precision = float((d_b_to_a <= tolerance).mean())
    recall = float((d_a_to_b <= tolerance).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "chamfer_px": chamfer,
        "chamfer_norm": chamfer / diagonal,
        "hausdorff95_px": hausdorff95,
        "boundary_f1": f1,
    }


def topology_counts(mask: np.ndarray) -> Dict[str, int]:
    """Connected components and holes of a binary region."""
    if not mask.any():
        return {"components": 0, "holes": 0}
    labels = skmeasure.label(mask, connectivity=2)
    components = int(labels.max())
    euler = int(skmeasure.euler_number(mask, connectivity=2))
    return {"components": components, "holes": components - euler}


@dataclass
class ShapeMetrics:
    """Every component of the score, plus the context needed to read it."""

    # region
    soft_iou: float = 0.0
    hard_iou: float = 0.0
    dice: float = 0.0
    area_ratio: float = 1.0
    # edge
    chamfer_px: float = 0.0
    chamfer_norm: float = 0.0
    hausdorff95_px: float = 0.0
    boundary_f1: float = 0.0
    # topology
    gt_components: int = 0
    gt_holes: int = 0
    pred_components: int = 0
    pred_holes: int = 0
    topology_error: int = 0
    # composed
    s_region: float = 0.0
    s_edge: float = 0.0
    s_topology: float = 0.0
    score: float = 0.0
    weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def compare(
    gt_coverage: np.ndarray,
    pred_coverage: np.ndarray,
    edge_tolerance: float = DEFAULT_EDGE_TOLERANCE,
    chamfer_tau: float = DEFAULT_CHAMFER_TAU,
    weights: Optional[Dict[str, float]] = None,
) -> ShapeMetrics:
    """Compare an aligned prediction against ground truth."""
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)

    gt_area = float(gt_coverage.sum())
    pred_area = float(pred_coverage.sum())
    area_ratio = (pred_area / gt_area) if gt_area > 0 else (1.0 if pred_area == 0 else float("inf"))

    bm = boundary_metrics(gt_coverage, pred_coverage, tolerance=edge_tolerance)
    gt_top = topology_counts(gt_coverage >= 0.5)
    pred_top = topology_counts(pred_coverage >= 0.5)

    topo_error = abs(gt_top["components"] - pred_top["components"]) + abs(gt_top["holes"] - pred_top["holes"])

    s_region = soft_iou(gt_coverage, pred_coverage)
    s_edge = math.exp(-bm["chamfer_px"] / chamfer_tau)
    s_topology = 1.0 / (1.0 + float(topo_error))

    score = w["region"] * s_region + w["edge"] * s_edge + w["topology"] * s_topology

    return ShapeMetrics(
        soft_iou=s_region,
        hard_iou=hard_iou(gt_coverage, pred_coverage),
        dice=dice(gt_coverage, pred_coverage),
        area_ratio=float(area_ratio),
        chamfer_px=bm["chamfer_px"],
        chamfer_norm=bm["chamfer_norm"],
        hausdorff95_px=bm["hausdorff95_px"],
        boundary_f1=bm["boundary_f1"],
        gt_components=gt_top["components"],
        gt_holes=gt_top["holes"],
        pred_components=pred_top["components"],
        pred_holes=pred_top["holes"],
        topology_error=int(topo_error),
        s_region=s_region,
        s_edge=s_edge,
        s_topology=s_topology,
        score=score,
        weights=w,
    )


def zero_metrics(gt_coverage: np.ndarray) -> ShapeMetrics:
    """Metrics for an unusable prediction (parse failure, empty path, over budget)."""
    pred = np.zeros_like(gt_coverage)
    m = compare(gt_coverage, pred)
    m.s_region = 0.0
    m.s_edge = 0.0
    m.s_topology = 0.0
    m.score = 0.0
    return m
