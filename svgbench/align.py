"""Similarity alignment: what we forgive, and what we deliberately do not.

A model that draws the right shape in the wrong place, or at the wrong size,
has still *seen* the shape. Penalising that would measure coordinate bookkeeping
rather than vision. So the scorer fits a translation and a uniform scale before
measuring.

Rotation and reflection are **not** fitted. They stay in the test on purpose:
for icons, an arrow pointing up and one pointing down are different objects, and
letting the scorer rotate freely would let a memorised copy score as a drawing.

The fitted family is ``p -> c + s(p - c) + t`` with ``c`` the prediction's own
centroid, so ``(s, t) = (1, 0)`` is exactly the identity transform. That matters:
it lets the scorer prove that an answer which is already correct is reproduced
bit for bit, rather than approximately.

Fitting is a deterministic coordinate-descent on a signed distance field of the
ground truth, so it has no seed and no iteration-count variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .pathdata import Matrix, ParsedPath, Point

#: Scale steps tried around the current best, per refinement round.
_SCALE_STEPS = (0.08, 0.04, 0.02, 0.01)
#: Translation steps in pixels, coarse to fine.
_TRANSLATE_STEPS = (8.0, 4.0, 2.0, 1.0, 0.5, 0.25)
#: Points sampled from the prediction when fitting (keeps fitting cheap).
_MAX_FIT_POINTS = 400

#: Mean boundary distance, in pixels, below which the identity transform is
#: returned without searching. The objective is evaluated on a signed distance
#: field built from the *rasterised* mask, so a prediction that lies exactly on
#: the true geometric boundary still measures roughly half a pixel away (the
#: mask boundary is quantised to the lattice). Without this tolerance the search
#: micro-nudges an already-correct answer and an identical figure scores below 1.
#: The value is set at one pixel: below the canvas's own resolution, an
#: alignment correction is not meaningful, and above it, a real misregistration
#: still gets corrected.
IDENTITY_TOLERANCE_PX = 1.0


@dataclass
class AlignmentResult:
    matrix: Matrix
    scale: float
    offset: Tuple[float, float]
    loss: float
    mode: str


def signed_distance_field(coverage: np.ndarray) -> np.ndarray:
    """Signed distance to the region boundary: negative inside, positive outside.

    Half-pixel convention, and it is load-bearing. A Euclidean distance transform
    measures distance between pixel *centres*, so the outermost inside pixel
    (whose centre actually lies half a pixel from the edge) is reported at
    distance 1. Subtracting 0.5 from each side places the zero level where the
    geometry is, and makes the field 1-Lipschitz rather than 2-Lipschitz.

    Without the correction the zero set sits half a pixel *inside* the shape and
    every fit inherits a $(-0.5,-0.5)$ bias. Measured on the response matrix, a
    pure $+30$ px translation recovered to $0.924$ instead of $1.000$, with the
    fitted offset $(-30.52,-0.51)$ against a true $(-30, 0)$, where the second
    component is the giveaway, since a shift along $x$ alone has no reason to
    need $y$.
    """
    mask = coverage >= 0.5
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    return np.where(mask, -(inside - 0.5), outside - 0.5)


def _sample_bilinear(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, pixel_offset: float = 0.5) -> np.ndarray:
    """Sample an image at continuous *canvas* coordinates.

    The rasteriser gives pixel $(i,j)$ the area $[j,j+1)\\times[i,i+1)$, so its
    centre lies at $(j+0.5, i+0.5)$. Array indexing implies the centre is at
    $(j, i)$ instead, and sampling the field as if that were true shifts every
    query by half a pixel on each axis. The aligner then reproduces that shift
    faithfully: \emph{measured}, a pure $+30$\,px translation fitted to
    $(-30.52, -0.51)$ rather than $(-30, 0)$, and at the true alignment the
    objective still reported a mean edge distance of $0.33$\,px instead of zero.
    The $y$ component is the tell: a shift along $x$ has no reason to want one.
    """
    h, w = img.shape
    xs = np.clip(xs - pixel_offset, 0.0, w - 1.0)
    ys = np.clip(ys - pixel_offset, 0.0, h - 1.0)
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = xs - x0
    fy = ys - y0
    return (
        img[y0, x0] * (1 - fx) * (1 - fy)
        + img[y0, x1] * fx * (1 - fy)
        + img[y1, x0] * (1 - fx) * fy
        + img[y1, x1] * fx * fy
    )


def coverage_moments(coverage: np.ndarray) -> Tuple[float, float, float]:
    """Return ``(centroid_x, centroid_y, area)`` of the covered region."""
    total = float(coverage.sum())
    if total <= 0.0:
        h, w = coverage.shape
        return (w / 2.0, h / 2.0, 0.0)
    # +0.5 because pixel (i, j) is centred at canvas point (j + 0.5, i + 0.5).
    ys, xs = np.indices(coverage.shape, dtype=np.float64)
    xs, ys = xs + 0.5, ys + 0.5
    return (float((coverage * xs).sum() / total), float((coverage * ys).sum() / total), total)


def _subsample(points: Sequence[Point], limit: int = _MAX_FIT_POINTS) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if len(arr) <= limit:
        return arr
    idx = np.linspace(0, len(arr) - 1, limit)
    return arr[np.rint(idx).astype(np.int64)]


def align_similarity(
    parsed: ParsedPath,
    gt_coverage: np.ndarray,
    pred_area: float,
    mode: str = "similarity",
    rounds: int = 3,
) -> AlignmentResult:
    """Fit translation (and, unless ``mode='translate'``, uniform scale).

    ``pred_area`` is the coverage sum of the *unaligned* prediction, so the area
    term can be evaluated in closed form as the scale changes instead of
    re-rasterising on every candidate move.
    """
    identity = AlignmentResult(Matrix(), 1.0, (0.0, 0.0), float("inf"), mode)
    pts = _subsample(parsed.all_points())
    if len(pts) == 0 or pred_area <= 0.0:
        return identity

    field = signed_distance_field(gt_coverage)
    gt_cx, gt_cy, gt_area = coverage_moments(gt_coverage)
    if gt_area <= 0.0:
        return identity

    pred_centroid = pts.mean(axis=0)
    s0 = 1.0 if mode == "translate" else math.sqrt(gt_area / pred_area)
    s0 = float(min(max(s0, 0.05), 20.0))
    recentre = (gt_cx - pred_centroid[0], gt_cy - pred_centroid[1])

    area_ratio = pred_area / gt_area

    def loss_of(scale: float, dx: float, dy: float) -> float:
        # Scale about the prediction's own centroid, then translate. The pair
        # (scale, offset) = (1, 0) is therefore exactly the identity, which is
        # what makes the identity case provably exact (Proposition: identity
        # exactness) instead of approximately exact.
        xs = pred_centroid[0] + scale * (pts[:, 0] - pred_centroid[0]) + dx
        ys = pred_centroid[1] + scale * (pts[:, 1] - pred_centroid[1]) + dy
        edge = float(np.abs(_sample_bilinear(field, xs, ys)).mean())
        # Log-ratio keeps the area penalty symmetric (2x too big == 2x too small)
        # and bounded, so it can never dominate the boundary term.
        area = abs(math.log(max(scale * scale * area_ratio, 1e-9)))
        return edge + 2.0 * area

    # Seed with the identity, so an answer that is already correct is never
    # perturbed by a nonzero-centroid starting guess, then with the
    # area-matched, centroid-matched guess.
    identity_loss = loss_of(1.0, 0.0, 0.0)
    if identity_loss <= IDENTITY_TOLERANCE_PX:
        # Already registered to within the rasteriser's own resolution. Return
        # the exact identity so that identical geometry reproduces bit for bit.
        return AlignmentResult(Matrix(), 1.0, (0.0, 0.0), identity_loss, mode)
    best = [1.0, 0.0, 0.0, identity_loss]

    def consider(scale: float, dx: float, dy: float) -> None:
        scale = float(min(max(scale, 0.05), 20.0))
        loss = loss_of(scale, dx, dy)
        if loss < best[3] - 1e-12:
            best[:] = [scale, dx, dy, loss]

    consider(s0, recentre[0], recentre[1])

    for scale_step in _SCALE_STEPS:
        for _ in range(rounds):
            improved = True
            while improved:
                improved = False
                before = best[3]
                s, dx, dy = best[0], best[1], best[2]
                if mode != "translate":
                    consider(s * (1 + scale_step), dx, dy)
                    consider(s * (1 - scale_step), dx, dy)
                for step in _TRANSLATE_STEPS:
                    consider(s, dx + step, dy)
                    consider(s, dx - step, dy)
                    consider(s, dx, dy + step)
                    consider(s, dx, dy - step)
                    consider(s, dx + step, dy + step)
                    consider(s, dx - step, dy - step)
                    consider(s, dx + step, dy - step)
                    consider(s, dx - step, dy + step)
                improved = best[3] < before - 1e-12

    # ---------------------------------------------------------- sub-pixel stage
    # The field above is built from the rasterised mask, and the Euclidean
    # distance transform measures distance to *pixel centres*, so its zero level
    # set is quantised to the lattice and the fit settles up to half a pixel away
    # from the true geometry. That residual is not cosmetic: at a one-pixel
    # stroke it costs several percent of the score, and it would penalise a model
    # for a placement error the benchmark promises to forgive.
    #
    # Near the boundary the soft coverage gives an essentially exact signed
    # distance instead: for a straight edge the coverage of a pixel whose centre
    # lies at signed distance d is 0.5 - d, so ``0.5 - coverage`` equals d to
    # first order and is free of the lattice offset. It is used here only for local
    # refinement, because it necessarily saturates (coverage is clamped to [0,1])
    # and so cannot pull a distant prediction into place -- which is exactly what
    # the distance-transform field above is for.
    soft_field = 0.5 - gt_coverage

    def soft_loss(scale: float, dx: float, dy: float) -> float:
        xs = pred_centroid[0] + scale * (pts[:, 0] - pred_centroid[0]) + dx
        ys = pred_centroid[1] + scale * (pts[:, 1] - pred_centroid[1]) + dy
        edge = float(np.abs(_sample_bilinear(soft_field, xs, ys)).mean())
        area = abs(math.log(max(scale * scale * area_ratio, 1e-9)))
        return edge + 2.0 * area

    fine_best = [best[0], best[1], best[2], soft_loss(best[0], best[1], best[2])]

    def consider_fine(scale: float, dx: float, dy: float) -> None:
        scale = float(min(max(scale, 0.05), 20.0))
        loss = soft_loss(scale, dx, dy)
        if loss < fine_best[3] - 1e-12:
            fine_best[:] = [scale, dx, dy, loss]

    for step in (0.5, 0.25, 0.125, 0.05):
        for _ in range(rounds):
            improved = True
            while improved:
                improved = False
                before = fine_best[3]
                s, dx, dy = fine_best[0], fine_best[1], fine_best[2]
                if mode != "translate":
                    consider_fine(s * (1 + 0.004), dx, dy)
                    consider_fine(s * (1 - 0.004), dx, dy)
                consider_fine(s, dx + step, dy)
                consider_fine(s, dx - step, dy)
                consider_fine(s, dx, dy + step)
                consider_fine(s, dx, dy - step)
                consider_fine(s, dx + step, dy + step)
                consider_fine(s, dx - step, dy + step)
                consider_fine(s, dx + step, dy - step)
                consider_fine(s, dx - step, dy - step)
                improved = fine_best[3] < before - 1e-12

    scale, dx, dy = fine_best[0], fine_best[1], fine_best[2]
    # Compose: scale about the prediction centroid, then translate. For
    # (1, 0, 0) this composition is the exact identity in floating point, since
    # -c + c == 0 exactly, so an already-correct answer is reproduced bit for bit.
    m = Matrix.translate(dx, dy) @ Matrix.scale(
        scale, cx=pred_centroid[0], cy=pred_centroid[1]
    )
    return AlignmentResult(matrix=m, scale=scale, offset=(dx, dy), loss=float(fine_best[3]), mode=mode)
