"""Shared drawing helpers for the specification's figures.

Kept separate from the two figure tools so that the primer figures (synthetic,
model-free) and the demo figures (real model answers) render identically: the
same colours must mean the same thing in both, or a reader comparing the primer
against the results will misread one of them.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib
import numpy as np
from matplotlib.patches import Patch
from scipy import ndimage
from skimage import measure as skmeasure

from svgbench.pathdata import SubPath
from svgbench.raster import render_coverage

#: Ink present in the ground truth only.
GT_ONLY = np.array([0.85, 0.20, 0.20])
#: Ink present in the answer only.
PRED_ONLY = np.array([0.13, 0.36, 0.80])
#: Ink present in both: this is the numerator of the region term.
BOTH = np.array([0.10, 0.10, 0.10])

COMPONENT_COLORS = [
    "#2b6cb0", "#dd6b20", "#38a169", "#805ad5", "#b83280",
    "#00838f", "#c53030", "#7b6f00", "#4a5568", "#2c7a7b",
]
HOLE_COLOR = "#e53e3e"


def overlay(gt: np.ndarray, pred: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Three-colour agreement map: truth only / answer only / both."""
    g, p = gt >= thresh, pred >= thresh
    img = np.ones(gt.shape + (3,), dtype=np.float64)
    img[g & ~p] = GT_ONLY
    img[~g & p] = PRED_ONLY
    img[g & p] = BOTH
    return img


def bare(ax, title: Optional[str] = None, fs: float = 9.0) -> None:
    """An image panel: no ticks, light frame, optional title."""
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c8c8c8")
    if title:
        ax.set_title(title, fontsize=fs, pad=4)


def band(dx: float = 0.0, dy: float = 0.0, w: int = 16, x0: int = 34, x1: int = 222,
         y0: int = 96, ss: int = 4, size: int = 256) -> np.ndarray:
    """A horizontal band of width ``w``, displaced by a known amount.

    The controlled subject of the region and edge primers: the displacement is
    known exactly, so a measured metric can be checked against a closed form.
    """
    sp = SubPath([(x0 + dx, y0 + dy), (x1 + dx, y0 + dy),
                  (x1 + dx, y0 + dy + w), (x0 + dx, y0 + dy + w)], True)
    return render_coverage([sp], size=size, ss=ss)


def disc(cx: float, cy: float, r: float, ss: int = 4, size: int = 256,
         cw: bool = True) -> SubPath:
    """A polygonal circle; ``cw=False`` reverses the winding so it cuts a hole."""
    n = 180
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a in ang]
    if not cw:
        pts = pts[::-1]
    return SubPath(pts, True)


def paint_components(ax, mask: np.ndarray, title: str, fs: float = 9.0) -> Tuple[int, int]:
    """Colour each connected component distinctly, enclosed holes in red.

    Returns the counts the topology term uses: ``(components, holes)``.
    """
    labels = skmeasure.label(mask, connectivity=2)
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    n_comp = int(labels.max())
    n_holes = int(skmeasure.label(holes, connectivity=2).max()) if holes.any() else 0

    img = np.ones(mask.shape + (3,), dtype=np.float64)
    for i in range(1, n_comp + 1):
        col = np.array(matplotlib.colors.to_rgb(COMPONENT_COLORS[(i - 1) % len(COMPONENT_COLORS)]))
        img[labels == i] = 0.35 + 0.65 * col
    if holes.any():
        img[holes] = np.array(matplotlib.colors.to_rgb(HOLE_COLOR))
    ax.imshow(img, interpolation="nearest")
    bare(ax, f"{title}\n{n_comp} component(s), {n_holes} enclosed hole(s)", fs=fs)
    return n_comp, n_holes


def legend_handles(include_holes: bool = False) -> Sequence[Patch]:
    h = [
        Patch(color=GT_ONLY, label="truth ink only"),
        Patch(color=PRED_ONLY, label="answer ink only"),
        Patch(color=BOTH, label="both (counts as agreement)"),
    ]
    if include_holes:
        h.append(Patch(color=HOLE_COLOR, label="enclosed hole"))
    return h
