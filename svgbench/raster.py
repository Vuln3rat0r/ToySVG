"""A small, pinned, dependency-free SVG-path rasteriser.

Why not use an existing SVG renderer? Because a benchmark score must not move
when a library version changes. resvg / cairosvg / librsvg each have their own
antialiasing and fill-rule quirks, so "the same" SVG can score differently after
a routine upgrade. Here the rasteriser is ~150 lines we control:

* edges come from :mod:`svgbench.pathdata`, so curves are flattened at a fixed
  sample count;
* coverage is estimated by point sampling on an ``ss x ss`` supersampled grid
  using deterministic integer arithmetic;
* the fill rule is applied per scanline span, which gives correct holes for both
  ``nonzero`` and ``evenodd``.

Coverage is returned as a float array in ``[0, 1]``: 1.0 fully inside, 0.0 fully
outside. Working with soft coverage rather than a hard threshold is what makes
the region metrics stable under sub-pixel translation.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .pathdata import Matrix, ParsedPath, SubPath

DEFAULT_SIZE = 256
DEFAULT_SS = 4


def _edge_arrays(subpaths: Sequence[SubPath], matrix: Matrix | None):
    """Flatten subpaths into edge endpoint arrays in device coordinates."""
    starts = []
    ends = []
    for sp in subpaths:
        pts = sp.points
        if len(pts) < 2:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        if matrix is not None:
            x = arr[:, 0]
            y = arr[:, 1]
            arr = np.stack(
                [matrix.a * x + matrix.c * y + matrix.e, matrix.b * x + matrix.d * y + matrix.f],
                axis=1,
            )
        # Filling always closes the subpath implicitly.
        closed = np.concatenate([arr, arr[:1]], axis=0)
        starts.append(closed[:-1])
        ends.append(closed[1:])
    if not starts:
        return None
    p0 = np.concatenate(starts, axis=0)
    p1 = np.concatenate(ends, axis=0)
    return p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1]


def render_coverage(
    subpaths: Sequence[SubPath],
    size: int = DEFAULT_SIZE,
    ss: int = DEFAULT_SS,
    matrix: Matrix | None = None,
    fill_rule: str = "nonzero",
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    """Rasterise ``subpaths`` into a ``(height, width)`` coverage array.

    ``matrix`` maps user coordinates to device pixels; the default identity
    means one user unit equals one pixel, which is what the 0..256 canvas
    contract in the prompt specifies.
    """
    if width is None:
        width = size
    if height is None:
        height = size
    out = np.zeros((height, width), dtype=np.float64)

    if fill_rule not in ("nonzero", "evenodd"):
        raise ValueError(f"unsupported fill rule {fill_rule!r}")

    edges = _edge_arrays(subpaths, matrix)
    if edges is None:
        return out
    x0, y0, x1, y1 = edges

    # Clip the scanline work to the shape's bounding box (plus a 2px margin so
    # antialiased edges are fully covered). Rows are aligned to the supersample
    # grid so the box filter below stays exact.
    ymin = float(min(y0.min(), y1.min())) - 2.0
    ymax = float(max(y0.max(), y1.max())) + 2.0
    # Clip to the canvas, then snap the range to whole pixels so it is aligned to
    # the supersample grid. Without the snap the box filter below would blend
    # samples from two different pixel rows: silently whenever the range happens
    # to divide evenly by ``ss``, and with a reshape error when it does not.
    # Fractional path coordinates (most real icons) hit this, so it matters.
    r0 = ss * int(math.floor(max(0.0, ymin)))
    r1 = ss * int(math.ceil(min(float(height), ymax)))
    if r1 <= r0:
        return out

    cols = width * ss
    rows = r1 - r0
    yc = (np.arange(r0, r1, dtype=np.float64) + 0.5) / ss

    # Half-open crossing test: a vertex shared by two edges counts once.
    below0 = y0[None, :] <= yc[:, None]
    below1 = y1[None, :] <= yc[:, None]
    cross = below0 != below1

    dy = (y1 - y0)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (yc[:, None] - y0[None, :]) / dy
        # Horizontal edges give 0/0 here; they never cross a scanline, so the
        # NaN is discarded by the ``cross`` mask -- silence it explicitly.
        xat = x0[None, :] + t * (x1[None, :] - x0[None, :])
    xat = np.where(cross, xat, np.inf)

    direction = np.where((y1 > y0)[None, :], 1.0, -1.0)
    direction = np.where(cross, direction, 0.0)

    # Sort crossings left to right, then walk the span list.
    order = np.argsort(xat, axis=1, kind="stable")
    xs = np.take_along_axis(xat, order, axis=1)
    ds = np.take_along_axis(direction, order, axis=1)

    if fill_rule == "nonzero":
        winding = np.cumsum(ds, axis=1)
        inside = winding != 0.0
    else:
        # Span i lies after i+1 crossings, so odd-even fills where the crossing
        # *count* is odd, i.e. at even span indices.
        parity = np.arange(xs.shape[1], dtype=np.int64) % 2 == 0
        inside = np.broadcast_to(parity[None, :], xs.shape).copy()

    span_a = xs[:, :-1]
    span_b = xs[:, 1:]
    valid = np.isfinite(span_a) & np.isfinite(span_b) & inside[:, :-1]
    if not valid.any():
        return out

    # Span [a, b) in device units maps to supersample columns c where the
    # sample centre (c + 0.5)/ss lies inside the span.
    ca = np.ceil(span_a[valid] * ss - 0.5).astype(np.int64)
    cb = np.ceil(span_b[valid] * ss - 0.5).astype(np.int64)
    ca = np.clip(ca, 0, cols)
    cb = np.clip(cb, 0, cols)
    keep = cb > ca
    if not keep.any():
        return out
    ca = ca[keep]
    cb = cb[keep]
    row_idx = np.broadcast_to(np.arange(rows)[:, None], valid.shape)[valid][keep]

    # Slab trick: +1 where a filled span opens, -1 where it closes.
    diff = np.zeros((rows, cols + 1), dtype=np.float64)
    np.add.at(diff, (row_idx, ca), 1.0)
    np.add.at(diff, (row_idx, cb), -1.0)
    cov_ss = np.cumsum(diff, axis=1)[:, :cols] > 0.0

    # Box-downsample the supersampled grid to pixel coverage.
    cov = cov_ss.reshape(rows // ss, ss, width, ss).mean(axis=(1, 3))
    top = r0 // ss
    out[top : top + cov.shape[0], :] = cov
    return out


def render_path(
    parsed: ParsedPath,
    size: int = DEFAULT_SIZE,
    ss: int = DEFAULT_SS,
    matrix: Matrix | None = None,
) -> np.ndarray:
    """Convenience wrapper for a :class:`~svgbench.pathdata.ParsedPath`."""
    return render_coverage(parsed.subpaths, size=size, ss=ss, matrix=matrix, fill_rule=parsed.fill_rule)


def hard_mask(coverage: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarise a coverage map at ``threshold``."""
    return coverage >= threshold


def coverage_to_uint8(coverage: np.ndarray) -> np.ndarray:
    """Coverage -> dark-on-light grayscale, the input PNG convention.

    Ground truth and prediction are rendered through this same path, so the
    benchmark never depends on an image library's own antialiasing.
    """
    return np.clip(np.rint((1.0 - coverage) * 255.0), 0, 255).astype(np.uint8)
