"""Corpus construction: source fetching, outline filtering, and shape statistics.

The benchmark needs figures that are *drawn as outlines* -- thin strokes -- rather
than solid silhouettes. This is a measurement decision, not a matter of taste:

* A solid silhouette can be approximated crudely and still overlap well, so it
  scores generously. A thin band cannot: the union of a blob and a band is
  roughly the silhouette, while the intersection is only the band, so a filled
  blob is punished by the region term.
* Outline artwork is a thin structure, so the boundary term carries signal about
  precision rather than about gross placement, and the tracer ceiling is
  informative rather than trivially high.

**How outline-ness is decided.** By measuring the stroke width of the rendered
figure: the median distance transform on the medial axis, doubled. On the pinned
Phosphor release this recovers the design's nominal stroke widths almost exactly
(8.1 / 12.2 / 16.1 / 24.2 px for thin / light / regular / bold in a 256 canvas),
while the ``fill`` weight measures 40.3 px and contains maximal inscribed radii up
to 176 px. The threshold therefore sits in a genuine gap rather than being tuned.
The p95 statistic is used rather than the mean so that a single junction -- where
inscribed discs of two strokes overlap and inflate the local radius -- cannot
decide the verdict.

**What is deliberately *not* a filter.** Whether the figure encloses an interior.
Open strokes (a barcode, an equals sign, an arrow) are just as much outlines as a
closed house, and requiring a hole would discard a fifth to a third of each
weight. Enclosure is therefore a *stratum*: ``closed`` figures exercise the
topology term, ``open`` figures exercise the boundary term, and results are
reported per stratum.

Nothing here is random. Thresholds are named constants and every per-item
measurement is written out, so the filter can be audited and the rejects seen.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from skimage import morphology as skmorph

from .metrics import region_boundary, topology_counts
from .svgdoc import parse_svg_document, render_layers

# ---------------------------------------------------------------- source pinning

PHOSPHOR_VERSION = "2.1.1"
PHOSPHOR_URL = (
    f"https://registry.npmjs.org/@phosphor-icons/core/-/core-{PHOSPHOR_VERSION}.tgz"
)
PHOSPHOR_LICENCE = "MIT"

#: Weights whose artwork is a thin stroke. ``fill`` is a solid silhouette.
#: ``duotone`` is excluded for a different reason: it is a two-layer composite
#: whose base plate is solid *and* carries opacity 0.2, so its rendered mask is a
#: grey region rather than ink, and a faithful copy would have to reproduce two
#: layers -- a different task from drawing one outline.
OUTLINE_WEIGHTS: Tuple[str, ...] = ("thin", "light", "regular", "bold")
EXCLUDED_WEIGHTS: Tuple[str, ...] = ("fill", "duotone")

#: Design stroke widths of the outline weights, in a 256 canvas. Recorded so the
#: measurement can be checked against intent rather than merely trusted.
EXPECTED_STROKE_PX: Dict[str, float] = {"thin": 8.0, "light": 12.0, "regular": 16.0, "bold": 24.0}

# ------------------------------------------------------------- outline criterion

CANVAS = 256
SS = 4

#: Measured stroke width must not exceed this, in pixels. Separates every outline
#: weight (measured p90 <= 24.7 px for bold) from ``fill`` (median 40.3 px).
MAX_STROKE_PX = 32.0

#: A figure too small to draw meaningfully is excluded whatever its stroke.
MIN_INK_AREA_PX = 150.0

# ---------------------------------------------------------------- stratification

#: Enclosure classes. Both are kept; this is a stratum, not a filter.
CLOSED = "closed"   # encloses at least one interior region
OPEN = "open"       # a stroke that encloses nothing (barcode, arrow, equals)

#: Complexity strata by command count, as terciles of the accepted corpus. The
#: cut points are filled in by ``analyze`` from the data and recorded in the
#: manifest, so a result can cite the exact stratification it used.
STRATUM_SIMPLE = "S"
STRATUM_MEDIUM = "M"
STRATUM_COMPLEX = "C"


@dataclass
class ShapeStats:
    """Everything the filter, the strata, and the report need, measured from pixels."""

    name: str
    weight: str
    n_commands: int
    n_subpaths: int
    # geometry
    area_px: float
    area_fraction: float
    bbox: Tuple[float, float, float, float]
    # topology and enclosure
    components: int
    holes: int
    enclosed_area_px: float
    enclosed_fraction: float
    # stroke measurement
    stroke_median_px: float
    stroke_p95_px: float
    max_inscribed_radius_px: float
    skeleton_length_px: float
    boundary_pixels: float
    # verdict
    is_outline: bool
    reject_reason: str = ""

    @property
    def enclosure(self) -> str:
        return CLOSED if self.holes >= 1 else OPEN

    def to_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["enclosure"] = self.enclosure
        return d


def _stroke_widths(mask: np.ndarray, edt: np.ndarray) -> Tuple[float, float, float]:
    """Estimate stroke width from the medial axis.

    On the medial axis of a band of width ``w`` the distance transform equals
    ``w/2``, so doubling gives the stroke width. Returns
    ``(median_px, p95_px, skeleton_length_px)``; zeros for an empty mask.

    The median describes the stroke; the p95 is used for the verdict because a
    junction inflates the local radius, and the max is reported only as a
    diagnostic (it reaches 52 px on ``bold`` purely from junctions).
    """
    if not mask.any():
        return 0.0, 0.0, 0.0
    skel = skmorph.skeletonize(mask)
    n = int(skel.sum())
    if n == 0:
        return 0.0, 0.0, 0.0
    values = edt[skel]
    return (
        float(2.0 * np.median(values)),
        float(2.0 * np.percentile(values, 95)),
        float(n),
    )


def shape_stats(
    name: str, weight: str, svg_text: str, size: int = CANVAS, ss: int = SS
) -> ShapeStats:
    """Measure one asset and decide whether it is outline artwork."""
    doc = parse_svg_document(svg_text)
    if not doc.layers:
        return ShapeStats(
            name=name, weight=weight, n_commands=0, n_subpaths=0, area_px=0.0,
            area_fraction=0.0, bbox=(0.0, 0.0, 0.0, 0.0), components=0, holes=0,
            enclosed_area_px=0.0, enclosed_fraction=0.0, stroke_median_px=0.0,
            stroke_p95_px=0.0, max_inscribed_radius_px=0.0, skeleton_length_px=0.0,
            boundary_pixels=0.0, is_outline=False, reject_reason="no path data",
        )

    coverage = render_layers(doc, size=size, ss=ss)
    mask = coverage >= 0.5

    topo = topology_counts(mask)
    area = float(mask.sum())
    filled = ndimage.binary_fill_holes(mask) if mask.any() else mask
    enclosed_area = float((filled & ~mask).sum())
    silhouette = area + enclosed_area
    enclosed_fraction = (enclosed_area / silhouette) if silhouette > 0 else 0.0

    edt = ndimage.distance_transform_edt(mask) if mask.any() else np.zeros_like(mask, dtype=float)
    max_radius = float(edt.max()) if mask.any() else 0.0
    stroke_med, stroke_p95, skel_len = _stroke_widths(mask, edt)

    ys, xs = np.nonzero(mask)
    bbox = (
        (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        if len(xs)
        else (0.0, 0.0, 0.0, 0.0)
    )

    stats = ShapeStats(
        name=name,
        weight=weight,
        n_commands=doc.n_commands,
        n_subpaths=doc.n_subpaths,
        area_px=area,
        area_fraction=area / float(size * size),
        bbox=bbox,
        components=int(topo["components"]),
        holes=int(topo["holes"]),
        enclosed_area_px=enclosed_area,
        enclosed_fraction=enclosed_fraction,
        stroke_median_px=stroke_med,
        stroke_p95_px=stroke_p95,
        max_inscribed_radius_px=max_radius,
        skeleton_length_px=skel_len,
        boundary_pixels=float(region_boundary(mask).sum()),
        is_outline=False,
    )
    reason = outline_reject_reason(stats)
    stats.is_outline = reason == ""
    stats.reject_reason = reason
    return stats


def outline_reject_reason(stats: ShapeStats) -> str:
    """Return ``""`` if the figure is thin-stroke outline artwork, else why not."""
    if stats.area_px < MIN_INK_AREA_PX or stats.components == 0:
        return f"too small or empty (area {stats.area_px:.0f} px)"
    if stats.skeleton_length_px <= 0.0:
        return "no medial axis"
    if stats.stroke_p95_px > MAX_STROKE_PX:
        return (
            f"not a thin stroke (stroke p95 {stats.stroke_p95_px:.1f} px "
            f"> {MAX_STROKE_PX})"
        )
    return ""


# ------------------------------------------------------------------ asset access

def fetch_phosphor(cache_dir: Path | str = "data/cache") -> Path:
    """Download and extract the pinned Phosphor release, caching the tarball."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tarball = cache / f"phosphor-core-{PHOSPHOR_VERSION}.tgz"
    if not tarball.exists():
        with urllib.request.urlopen(PHOSPHOR_URL, timeout=120) as resp:
            tarball.write_bytes(resp.read())
    extracted = cache / f"phosphor-{PHOSPHOR_VERSION}"
    if not (extracted / "assets").is_dir() and not (extracted / "package" / "assets").is_dir():
        with tarfile.open(tarball) as tf:
            tf.extractall(extracted)
    for candidate in (extracted / "package" / "assets", extracted / "assets"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"no assets directory inside {extracted}")


def asset_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def iter_assets(
    assets_dir: Path | str, weights: Sequence[str] = OUTLINE_WEIGHTS
) -> Iterable[Tuple[str, str, str]]:
    """Yield ``(name, weight, svg_text)`` for each asset in the given weights."""
    root = Path(assets_dir)
    for weight in weights:
        folder = root / weight
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.svg")):
            stem = path.stem
            suffix = f"-{weight}"
            name = stem[: -len(suffix)] if stem.endswith(suffix) else stem
            yield name, weight, path.read_text(encoding="utf-8")


# ------------------------------------------------------------------- the analysis

def _block(items: Sequence[ShapeStats]) -> Dict[str, object]:
    if not items:
        return {}
    stroke = np.array([s.stroke_median_px for s in items], dtype=float)
    p95 = np.array([s.stroke_p95_px for s in items], dtype=float)
    enc = np.array([s.enclosed_fraction for s in items], dtype=float)
    holes = np.array([s.holes for s in items], dtype=float)
    return {
        "n": len(items),
        "accepted": int(sum(s.is_outline for s in items)),
        "accepted_fraction": float(np.mean([s.is_outline for s in items])),
        "stroke_median_px": {
            "median": float(np.median(stroke)),
            "p10": float(np.percentile(stroke, 10)),
            "p90": float(np.percentile(stroke, 90)),
        },
        "stroke_p95_px": {"median": float(np.median(p95)), "p90": float(np.percentile(p95, 90)), "max": float(p95.max())},
        "enclosed_fraction": {"median": float(np.median(enc))},
        "holes": {"median": float(np.median(holes)), "min": float(holes.min())},
        "closed_fraction": float(np.mean(holes >= 1)),
        "reject_reasons": _reason_histogram(items),
    }


def _reason_histogram(items: Sequence[ShapeStats]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in items:
        if s.is_outline:
            continue
        key = re.sub(r"\(.*", "", s.reject_reason).strip() or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def stratum_cuts(stats: Sequence[ShapeStats]) -> Tuple[float, float]:
    """Tercile cut points of command count over the accepted corpus."""
    cmds = np.array([s.n_commands for s in stats if s.is_outline], dtype=float)
    if len(cmds) < 3:
        return (0.0, 0.0)
    return float(np.percentile(cmds, 100.0 / 3.0)), float(np.percentile(cmds, 200.0 / 3.0))


def complexity_stratum(n_commands: float, cuts: Tuple[float, float]) -> str:
    lo, hi = cuts
    if n_commands <= lo:
        return STRATUM_SIMPLE
    if n_commands <= hi:
        return STRATUM_MEDIUM
    return STRATUM_COMPLEX


def summarize(stats: Sequence[ShapeStats]) -> Dict[str, object]:
    """Aggregate measurements per weight and overall, with the strata."""
    accepted = [s for s in stats if s.is_outline]
    cuts = stratum_cuts(stats)
    by_weight = {w: _block([s for s in stats if s.weight == w]) for w in sorted({s.weight for s in stats})}

    # Does the measurement recover the design's nominal stroke widths?
    stroke_check = {}
    for weight, expected in EXPECTED_STROKE_PX.items():
        measured = [s.stroke_median_px for s in accepted if s.weight == weight]
        if measured:
            med = float(np.median(measured))
            stroke_check[weight] = {
                "expected_px": expected,
                "measured_median_px": med,
                "abs_error_px": abs(med - expected),
                "n": len(measured),
            }

    return {
        "overall": _block(stats),
        "by_weight": by_weight,
        "accepted_total": len(accepted),
        "complexity_cuts_commands": {"p33": cuts[0], "p67": cuts[1]},
        "enclosure_counts": {
            CLOSED: int(sum(s.enclosure == CLOSED for s in accepted)),
            OPEN: int(sum(s.enclosure == OPEN for s in accepted)),
        },
        "stratum_counts": {
            name: int(sum(complexity_stratum(s.n_commands, cuts) == name for s in accepted))
            for name in (STRATUM_SIMPLE, STRATUM_MEDIUM, STRATUM_COMPLEX)
        },
        "stroke_width_recovered": stroke_check,
    }


def analyze(
    assets_dir: Path | str,
    weights: Sequence[str] = tuple(OUTLINE_WEIGHTS) + tuple(EXCLUDED_WEIGHTS),
    sample: Optional[int] = None,
    seed: int = 20260919,
) -> Tuple[List[ShapeStats], Dict[str, object]]:
    """Measure a corpus slice. Deterministic given ``(sample, seed)``."""
    rng = np.random.default_rng(seed)
    stats: List[ShapeStats] = []
    for weight in weights:
        items = list(iter_assets(assets_dir, [weight]))
        if sample is not None and len(items) > sample:
            idx = np.sort(rng.choice(len(items), size=sample, replace=False))
            items = [items[i] for i in idx]
        for name, w, text in items:
            stats.append(shape_stats(name, w, text))
    return stats, summarize(stats)


# ---------------------------------------------------------------------------- CLI

def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="svgbench.corpus", description="corpus construction")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="download and cache the pinned release")
    p_fetch.add_argument("--cache", default="data/cache")

    p_an = sub.add_parser("analyze", help="measure assets and report the outline filter")
    p_an.add_argument("--assets", default=None, help="assets dir (defaults to the cache)")
    p_an.add_argument("--cache", default="data/cache")
    p_an.add_argument("--weights", default=",".join(tuple(OUTLINE_WEIGHTS) + tuple(EXCLUDED_WEIGHTS)))
    p_an.add_argument("--sample", type=int, default=None, help="items per weight")
    p_an.add_argument("--seed", type=int, default=20260919)
    p_an.add_argument("--json", default=None)
    p_an.add_argument("--print-rejects", type=int, default=0)

    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        print(f"assets at {fetch_phosphor(args.cache)}")
        return 0

    assets = Path(args.assets) if args.assets else fetch_phosphor(args.cache)
    weights = [w for w in args.weights.split(",") if w]
    stats, summary = analyze(assets, weights, sample=args.sample, seed=args.seed)

    print(json.dumps(summary, indent=2))
    for s in stats:
        if not s.is_outline and args.print_rejects > 0:
            print(f"  reject {s.weight}/{s.name}: {s.reject_reason}")
            args.print_rejects -= 1
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(
                {
                    "phosphor_version": PHOSPHOR_VERSION,
                    "licence": PHOSPHOR_LICENCE,
                    "criteria": {
                        "max_stroke_px": MAX_STROKE_PX,
                        "min_ink_area_px": MIN_INK_AREA_PX,
                        "canvas": CANVAS,
                        "supersampling": SS,
                    },
                    "summary": summary,
                    "items": [s.to_dict() for s in stats],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
