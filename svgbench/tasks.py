"""Task-pack construction.

Turns outline-filtered corpus assets into \emph{items}: a PNG the model may see,
and a hidden ground-truth path it may not.

The blind-input commitment is enforced \emph{structurally} rather than by
convention. Model-facing files live under ``<pack>/``, withholding asset names,
license metadata, source hashes, and vector geometry beyond the rendered PNG.
Ground truth lives under a separate held-out tree, which a model runner cannot access. If the two were interchangeable (as they are in most
benchmarks, where the task file simply *is* dataset row $i$), then blinding
would be a promise; here it is a directory boundary.

Item identity is a salted hash, so the file names carry no retrieval key.

Cost note. Deciding ``is_outline`` requires rasterising and taking a medial axis,
which measured at $\\approx 0.4$\\,s per asset. Measuring all 1{,}512 assets of a
weight serially is therefore a ten-minute operation, which is unacceptable in a
build step. The pool is instead drawn from a deterministic hash ordering of the
asset names and halts as soon as enough assets have \emph{passed} the filter, and
the measurement is cached to JSON so rebuilds are free. Hash ordering is
uncorrelated with shape, so the pool is a sample rather than a biased head of the
directory listing, but it is a sample, and the strata cuts are computed on it,
which is why the pool size is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from . import corpus as _corpus
from .pathdata import ParsedPath, SubPath, parse_path, simplify_polyline
from .raster import coverage_to_uint8, render_path
from .svgdoc import parse_svg_document, render_layers

CANVAS = _corpus.CANVAS
SS = _corpus.SS

# ---------------------------------------------------------------- constants

SPLIT_CLEAN = "ID-CLEAN"
SPLIT_DEGRADED = "ID-DEGRADED"
SPLIT_PERTURB = "OOD-PERTURB"
SPLIT_DISJOINT = "OOD-DISJOINT"
SPLITS = (SPLIT_CLEAN, SPLIT_DEGRADED, SPLIT_PERTURB, SPLIT_DISJOINT)

BUDGETS = {"B0": 16, "B1": 48, "B2": 160, "B3": None}
PRIMARY_BUDGET = "B1"

DEFAULT_SALT = "toysvg-p0-2026"
DEFAULT_SEED = 20260919
DEFAULT_POOL = 180
DEFAULT_JOBS = 8

# Perturbation parameters, frozen, and **not used by version 1**. Version 1
# stores the vendor's own path for every item (see PERTURB_ENABLED), so a model
# is never measured against geometry this benchmark invented. The deformation
# below is retained because it is the designed anti-memorisation probe for the
# deferred OOD-PERTURB split, and because a published construction should be
# readable without a git checkout of the version that shipped it.
#
# The deformation is a smooth displacement
# field with amplitude in canvas units; it is *not* a random warp, because a
# corpus that changes shape between two builds is not a benchmark.
#
# What protects the item is the field's *gradient*, not its amplitude. At these
# frequencies the wavelengths are 256 px and 183 px, so the gradient is at most
# 0.27 px/px vertically and 0.38 px/px horizontally: across one design stroke
# (16 px on the regular weight) the horizontal component of the displacement
# moves by at most 4.3 px and the vertical by at most 6.1 px, measured as 5.94 px
# worst case over all vertex pairs within a stroke width in the three perturbed
# items (p95 4.1-4.6 px). Slow enough not to open an intended gap or sever a
# stroke, but not negligible next to a 16 px stroke, so the invariants it must
# preserve are *measured* rather than inherited from the bound: on the three
# perturbed items the stroke median is 16.00 px before and after, components and
# holes are unchanged, and ink area moves by at most 2.8%.
#: With the perturbation off, the items that would have been deformed receive
#: the vendor path sharp and unmodified: which is exactly the clean treatment,
#: so that is what they are labelled. Nothing else about the partition moves.
V1_SPLIT_LABEL = {SPLIT_PERTURB: SPLIT_CLEAN}

PERTURB_ENABLED = False
PERTURB_AMPLITUDE_PX = 11.0
PERTURB_FREQ = (1.0, 1.4)
PERTURB_PHASE = (0.7, 2.1)
#: How far the emitted polyline may deviate from the exact deformed curve, in
#: pixels: 7% of a design stroke, so the re-expression is finer than the drawing.
PERTURB_TOLERANCE_PX = 1.1

# Degradation parameters, frozen, applied to the *input PNG only*. Ground truth
# geometry is untouched, so the item stays exactly gradeable.
DEGRADE_BLUR_SIGMA = 1.1
DEGRADE_NOISE_SIGMA = 6.0
DEGRADE_CONTRAST = 0.78
DEGRADE_MARGIN = 0.06


# ---------------------------------------------------------------- item model


@dataclass
class Item:
    """One benchmark item, split into a public half and a hidden half."""

    item_id: str
    stratum: str
    split: str
    budget: str
    canvas: int
    seed: int
    # --- hidden side: never written into the model-facing directory ---
    name: str = ""
    weight: str = ""
    gt_d: str = ""
    gt_fill_rule: str = "nonzero"
    n_commands_gt: int = 0
    n_subpaths_gt: int = 0
    source_sha256: str = ""
    topology: Dict[str, int] = field(default_factory=dict)
    ink_area_px: float = 0.0
    stroke_median_px: float = 0.0

    @property
    def k(self) -> Optional[int]:
        return BUDGETS[self.budget]

    def public(self) -> Dict[str, object]:
        """The model-facing record. Contains nothing that identifies the shape."""
        return {
            "item_id": self.item_id,
            "budget": self.budget,
            "k": self.k,
            "canvas": self.canvas,
            "input_png": f"input/{self.item_id}.png",
        }

    def hidden(self) -> Dict[str, object]:
        return asdict(self)


def item_id_for(name: str, weight: str, salt: str) -> str:
    """A stable, non-identifying identifier.

    Deliberately not reversible and deliberately not a prefix of the name: a
    model that could recover ``ph-house`` from the file name would be handed the
    retrieval key that the blind-input commitment exists to withhold.
    """
    digest = hashlib.sha256(f"{salt}|{weight}|{name}".encode("utf-8")).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------- deformation


def _displacement(x: float, y: float) -> Tuple[float, float]:
    """Frozen smooth displacement field, in canvas units."""
    ax = PERTURB_AMPLITUDE_PX
    (f1, f2), (p1, p2) = PERTURB_FREQ, PERTURB_PHASE
    dx = ax * math.sin(2.0 * math.pi * f1 * y / CANVAS + p1)
    dy = ax * math.sin(2.0 * math.pi * f2 * x / CANVAS + p2)
    return dx, dy


def _dedupe(points: Sequence[Tuple[float, float]], eps: float = 1e-6) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in points:
        if not out or abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps:
            out.append((float(p[0]), float(p[1])))
    return out


def _emit_polylines(subpaths: Sequence[SubPath]) -> str:
    """Serialise flattened subpaths as an absolute ``M``/``L``/``Z`` expression."""
    parts: List[str] = []
    for sp in subpaths:
        pts = _dedupe(sp.points)
        if len(pts) < 3:
            continue
        x0, y0 = pts[0]
        parts.append(f"M{x0:.3f} {y0:.3f}")
        for x, y in pts[1:]:
            parts.append(f"L{x:.3f} {y:.3f}")
        if sp.closed:
            parts.append("Z")
    return "".join(parts)


def perturb_path(parsed: ParsedPath, tolerance: float = PERTURB_TOLERANCE_PX) -> str:
    """Deform geometry smoothly, preserving topology and command economy.

    Flatten, displace, simplify. The deformation is smooth and
    topology-preserving by construction (the field is continuous and slow, so it
    cannot open a gap or sever a stroke), so a perturbed item remains a
    legitimate outline figure rather than becoming noise. Its purpose is
    anti-memorisation: exact recall of the original icon yields the *wrong*
    answer by a measurable margin, so the gap between the clean and perturbed
    splits estimates how much of a model's score is memory.

    The result is still SVG path data, in the same language the model answers in
    (absolute ``M``/``L``/``Z`` at a 3-decimal precision, arcs and curves
    resolved) and never a bitmap. The re-expression is longer than the
    vendor's (2.2x to 5.1x in characters here, because a smooth deformation on a
    fine polyline has more vertices than the original arcs and straights), but
    it is scored by rendering, so its verbosity costs the model nothing.
    """
    deformed: List[SubPath] = []
    for sp in parsed.subpaths:
        moved = []
        for x, y in sp.points:
            dx, dy = _displacement(x, y)
            moved.append((x + dx, y + dy))
        moved = _dedupe(simplify_polyline(moved, tolerance))
        if len(moved) >= 3:
            deformed.append(SubPath(moved, sp.closed))
    return _emit_polylines(deformed)


# ---------------------------------------------------------------- degradation


def degrade_png(img: np.ndarray, seed: int) -> np.ndarray:
    """Reduce the input image's quality without touching the ground truth."""
    from scipy import ndimage

    x = img.astype(np.float64) / 255.0
    x = ndimage.gaussian_filter(x, sigma=DEGRADE_BLUR_SIGMA, truncate=2.0, mode="constant", cval=1.0)
    x = x * DEGRADE_CONTRAST + (1.0 - DEGRADE_CONTRAST) * 0.5
    rng = np.random.default_rng(seed)
    x = np.clip(x + rng.normal(0.0, DEGRADE_NOISE_SIGMA / 255.0, size=x.shape), 0.0, 1.0)
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def _apply_margin(img: np.ndarray, margin: float = DEGRADE_MARGIN) -> np.ndarray:
    """Inset the figure slightly, so a degraded item is not pixel-aligned by luck."""
    if margin <= 0.0:
        return img
    h, w = img.shape
    out = np.full_like(img, 255)
    dy, dx = int(round(h * margin)), int(round(w * margin))
    out[dy : h - dy, dx : w - dx] = img[dy : h - dy, dx : w - dx]
    return out


# ---------------------------------------------------------------- measurement


def _candidates(assets_dir: Path, weights: Sequence[str]) -> List[Tuple[str, str, Path]]:
    """Enumerate ``(name, weight, path)`` for the given weights."""
    out: List[Tuple[str, str, Path]] = []
    for weight in weights:
        folder = assets_dir / weight
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.svg")):
            stem, suffix = path.stem, f"-{weight}"
            name = stem[: -len(suffix)] if stem.endswith(suffix) else stem
            out.append((name, weight, path))
    return out


def _measure_one(args: Tuple[str, str, str]) -> Dict[str, object]:
    """Measure a single asset. Top-level so it can be pickled to a worker."""
    name, weight, path_str = args
    text = ""
    try:
        text = Path(path_str).read_text(encoding="utf-8")
        stats = _corpus.shape_stats(name, weight, text)
    except Exception as exc:  # a broken asset must not abort the build
        return {
            "name": name, "weight": weight, "path": path_str,
            "is_outline": False, "reject_reason": f"error: {type(exc).__name__}",
            "n_commands": 0, "n_subpaths": 0, "area_px": 0.0,
            "components": 0, "holes": 0, "stroke_median_px": 0.0, "stroke_p95_px": 0.0,
            "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    return {
        "name": name, "weight": weight, "path": path_str,
        "is_outline": bool(stats.is_outline),
        "reject_reason": stats.reject_reason,
        "n_commands": int(stats.n_commands),
        "n_subpaths": int(stats.n_subpaths),
        "area_px": float(stats.area_px),
        "components": int(stats.components),
        "holes": int(stats.holes),
        "stroke_median_px": float(stats.stroke_median_px),
        "stroke_p95_px": float(stats.stroke_p95_px),
        "source_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def measure_pool(
    assets_dir: Path | str,
    weights: Sequence[str],
    want_accepted: int = DEFAULT_POOL,
    salt: str = DEFAULT_SALT,
    jobs: int = DEFAULT_JOBS,
    cache_dir: Optional[Path | str] = "data/cache",
    batch: int = 96,
    verbose: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Measure assets until ``want_accepted`` pass the outline filter.

    Returns ``(accepted_records, counters)``. The asset order is a deterministic
    salted-hash ordering, so the pool does not depend on directory listing order
    and is reproducible across machines.
    """
    assets_dir = Path(assets_dir)
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        key = hashlib.sha256(
            f"{_corpus.PHOSPHOR_VERSION}|{','.join(weights)}|{want_accepted}|{salt}".encode()
        ).hexdigest()[:16]
        cache_path = Path(cache_dir) / f"pool-{key}.json"
        if cache_path.exists():
            blob = json.loads(cache_path.read_text(encoding="utf-8"))
            return blob["accepted"], blob["counters"]

    cands = _candidates(assets_dir, weights)
    cands.sort(key=lambda t: hashlib.sha256(f"{salt}|{t[1]}|{t[0]}".encode()).hexdigest())

    accepted: List[Dict[str, object]] = []
    scanned = 0
    rejected: Dict[str, int] = {}
    cursor = 0

    def drain(chunk: List[Tuple[str, str, Path]]) -> None:
        nonlocal scanned
        payload = [(n, w, str(p)) for n, w, p in chunk]
        if jobs and jobs > 1 and len(payload) > 1:
            import multiprocessing as mp

            with mp.Pool(processes=min(jobs, len(payload))) as pool:
                results = pool.map(_measure_one, payload)
        else:
            results = [_measure_one(a) for a in payload]
        for rec in results:
            scanned += 1
            if rec["is_outline"]:
                accepted.append(rec)
            else:
                key = str(rec["reject_reason"]).split("(")[0].strip() or "unknown"
                rejected[key] = rejected.get(key, 0) + 1

    while len(accepted) < want_accepted and cursor < len(cands):
        chunk = cands[cursor : cursor + batch]
        cursor += len(chunk)
        drain(chunk)
        if verbose:
            print(f"  measured {scanned} assets, {len(accepted)} accepted", flush=True)

    counters = {
        "assets_available": len(cands),
        "assets_scanned": scanned,
        "accepted": len(accepted),
        "acceptance_rate": (len(accepted) / scanned) if scanned else 0.0,
        "reject_reasons": dict(sorted(rejected.items(), key=lambda kv: -kv[1])),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"accepted": accepted, "counters": counters}), encoding="utf-8")
    return accepted, counters


# ---------------------------------------------------------------- split logic


def _split_for(item_id: str, split_fractions: Dict[str, float]) -> str:
    """Assign a split deterministically from the item id.

    Hashing the id rather than the name keeps the assignment reproducible while
    still uncorrelated with the shape.
    """
    h = int(hashlib.sha256(f"split|{item_id}".encode()).hexdigest(), 16)
    u = (h % 10_000) / 10_000.0
    acc = 0.0
    for name in SPLITS:
        acc += split_fractions.get(name, 0.0)
        if u < acc:
            return name
    return SPLITS[-1]


def _stratum_cuts(records: Sequence[Dict[str, object]]) -> Tuple[float, float]:
    """Tercile cut points of command count, over accepted assets."""
    cmds = np.array([float(r["n_commands"]) for r in records], dtype=float)
    if len(cmds) < 3:
        return (0.0, 0.0)
    return float(np.percentile(cmds, 100.0 / 3.0)), float(np.percentile(cmds, 200.0 / 3.0))


# ---------------------------------------------------------------- pack build


def build_pack(
    assets_dir: Path | str,
    out_dir: Path | str,
    salt: str = DEFAULT_SALT,
    seed: int = DEFAULT_SEED,
    per_stratum: int | Dict[str, int] = 8,
    total_items: Optional[int] = None,
    weights: Sequence[str] = ("regular",),
    budgets: Sequence[str] = (PRIMARY_BUDGET,),
    split_fractions: Optional[Dict[str, float]] = None,
    pool_want_accepted: int = DEFAULT_POOL,
    jobs: int = DEFAULT_JOBS,
    cache_dir: Optional[Path | str] = "data/cache",
    stratify_by_weight: bool = False,
    verbose: bool = False,
) -> Dict[str, object]:
    """Build a task pack. Returns the manifest."""
    assets_dir = Path(assets_dir)
    out_dir = Path(out_dir)
    gt_dir = out_dir.parent / "gt"
    input_dir = out_dir / "input"
    for d in (out_dir / "tasks", input_dir, gt_dir):
        d.mkdir(parents=True, exist_ok=True)

    # The bins keep their P0 fractions and their P0 order, so no item changes
    # bin when the perturbation is switched off: the perturb bin is *relabelled*
    # (see V1_SPLIT_LABEL) rather than removed, because an interval cannot be
    # deleted from the middle of a partition without moving items between the
    # bins that remain.
    split_fractions = split_fractions or {
        SPLIT_CLEAN: 0.50,
        SPLIT_DEGRADED: 0.20,
        SPLIT_PERTURB: 0.20,
        SPLIT_DISJOINT: 0.10,
    }

    accepted, counters = measure_pool(
        assets_dir, weights, want_accepted=pool_want_accepted, salt=salt, jobs=jobs,
        cache_dir=cache_dir, verbose=verbose,
    )
    if not accepted:
        raise SystemExit(f"no outline assets found under {assets_dir} for weights {list(weights)}")
    cuts = _stratum_cuts(accepted)

    # deterministic selection inside each stratum
    buckets: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for rec in accepted:
        stratum = _corpus.complexity_stratum(float(rec["n_commands"]), cuts)
        key = (stratum, str(rec["weight"]) if stratify_by_weight else "")
        buckets.setdefault(key, []).append(rec)
    for key in buckets:
        buckets[key].sort(key=lambda r: hashlib.sha256(f"{salt}|{r['name']}|{r['weight']}".encode()).hexdigest())

    if total_items is not None:
        base = total_items // 3
        rem = total_items % 3
        quotas = {
            _corpus.STRATUM_SIMPLE: base + (1 if rem > 0 else 0),
            _corpus.STRATUM_MEDIUM: base + (1 if rem > 1 else 0),
            _corpus.STRATUM_COMPLEX: base,
        }
    elif isinstance(per_stratum, dict):
        quotas = per_stratum
    else:
        quotas = {s: int(per_stratum) for s in (_corpus.STRATUM_SIMPLE, _corpus.STRATUM_MEDIUM, _corpus.STRATUM_COMPLEX)}

    items: List[Item] = []
    for (stratum, _wk), recs in sorted(buckets.items()):
        quota = quotas.get(stratum, 0)
        for rec in recs[:quota]:
            name, weight = str(rec["name"]), str(rec["weight"])
            svg_text = Path(str(rec["path"])).read_text(encoding="utf-8")
            doc = parse_svg_document(svg_text)
            if not doc.layers:
                continue
            parsed = doc.layers[0].parsed
            base_coverage = render_layers(doc, size=CANVAS, ss=SS)
            if base_coverage.sum() <= 0.0:
                continue
            item_id = item_id_for(name, weight, salt)
            split = _split_for(item_id, split_fractions)
            if not PERTURB_ENABLED:
                split = V1_SPLIT_LABEL.get(split, split)

            for budget in budgets:
                it = Item(
                    item_id=item_id,
                    stratum=stratum,
                    split=split,
                    budget=budget,
                    canvas=CANVAS,
                    seed=seed,
                    name=name,
                    weight=weight,
                    gt_fill_rule=parsed.fill_rule,
                    n_commands_gt=int(rec["n_commands"]),
                    n_subpaths_gt=int(rec["n_subpaths"]),
                    source_sha256=str(rec["source_sha256"]),
                    topology={"components": int(rec["components"]), "holes": int(rec["holes"])},
                    ink_area_px=float(rec["area_px"]),
                    stroke_median_px=float(rec["stroke_median_px"]),
                )
                if split == SPLIT_PERTURB and PERTURB_ENABLED:
                    # Deformed geometry: the ground truth *is* the deformation,
                    # so both its expression and its command count change.
                    it.gt_d = perturb_path(parsed)
                    coverage = render_path(parse_path(it.gt_d, fill_rule=parsed.fill_rule), size=CANVAS, ss=SS)
                    rep = parse_path(it.gt_d, fill_rule=parsed.fill_rule)
                    it.n_commands_gt, it.n_subpaths_gt = rep.n_commands, rep.n_subpaths
                else:
                    it.gt_d = parsed.d
                    coverage = base_coverage

                # ---- hidden ground truth, in its own tree ----
                (gt_dir / f"{item_id}.json").write_text(json.dumps(it.hidden(), indent=2), encoding="utf-8")

                # ---- the image the model sees ----
                png = coverage_to_uint8(coverage)
                if split == SPLIT_DEGRADED:
                    png = _apply_margin(degrade_png(png, seed + int(item_id[:6], 16)))
                Image.fromarray(png, mode="L").save(input_dir / f"{item_id}.png")

                pub = it.public()
                pub["input_sha256"] = hashlib.sha256((input_dir / f"{item_id}.png").read_bytes()).hexdigest()
                (out_dir / "tasks" / f"{item_id}.json").write_text(json.dumps(pub, indent=2), encoding="utf-8")
                items.append(it)

    manifest = {
        "name": "toysvg",
        "pack_version": "P0",
        "salt": salt,
        "seed": seed,
        "canvas": CANVAS,
        "ss": SS,
        "source": {
            "family": "phosphor",
            "version": _corpus.PHOSPHOR_VERSION,
            "licence": _corpus.PHOSPHOR_LICENCE,
            "url": _corpus.PHOSPHOR_URL,
            "weights": list(weights),
        },
        "filter": {"max_stroke_p95_px": _corpus.MAX_STROKE_PX, "min_ink_area_px": _corpus.MIN_INK_AREA_PX},
        "measurement": counters | {"pool_want_accepted": pool_want_accepted, "jobs": jobs},
        "perturbation": (
            {"enabled": True, "amplitude_px": PERTURB_AMPLITUDE_PX, "freq": list(PERTURB_FREQ),
             "phase": list(PERTURB_PHASE), "tolerance_px": PERTURB_TOLERANCE_PX}
            if PERTURB_ENABLED else
            {"enabled": False, "split_deferred": SPLIT_PERTURB,
             "note": "every item's gt_d is the vendor path verbatim; no geometric transformation is applied"}
        ),
        "degradation": {
            "blur_sigma": DEGRADE_BLUR_SIGMA, "noise_sigma": DEGRADE_NOISE_SIGMA,
            "contrast": DEGRADE_CONTRAST, "margin": DEGRADE_MARGIN,
        },
        "budgets": {k: v for k, v in BUDGETS.items() if k in budgets},
        "stratum_cuts_commands": list(cuts),
        "counts": _counts(items),
        "n_items": len(items),
        "gt_dir": str(gt_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _counts(items: Sequence[Item]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for it in items:
        out.setdefault(it.split, {})
        out[it.split][it.stratum] = out[it.split].get(it.stratum, 0) + 1
    return out


# ---------------------------------------------------------------- loading


def load_manifest(pack_dir: Path | str) -> Dict[str, object]:
    return json.loads((Path(pack_dir) / "manifest.json").read_text(encoding="utf-8"))


def load_public(pack_dir: Path | str) -> List[Dict[str, object]]:
    """Model-facing task records, sorted for reproducibility."""
    tasks = Path(pack_dir) / "tasks"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(tasks.glob("*.json"))]


def load_hidden(pack_dir: Path | str, item_id: str) -> Item:
    """Ground truth. Runners must not call this; only the scorer may."""
    pack_dir = Path(pack_dir)
    return Item(**json.loads((pack_dir.parent / "gt" / f"{item_id}.json").read_text(encoding="utf-8")))


def fixture_svg() -> str:
    """A shape that is *not* in the corpus, for teaching syntax in the prompt."""
    return (
        '<svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg">'
        '<path d="M40 200 L128 56 L216 200 Z" fill="none" stroke="currentColor" stroke-width="16"/></svg>'
    )


# ---------------------------------------------------------------- cli


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build a ToySVG task pack from Phosphor assets.")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", default="data/pack")
    ap.add_argument("--per-stratum", type=int, default=8)
    ap.add_argument("--total-items", type=int, default=None, help="total items across strata (e.g. 32)")
    ap.add_argument("--weights", default="regular")
    ap.add_argument("--budgets", default=PRIMARY_BUDGET)
    ap.add_argument("--salt", default=DEFAULT_SALT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL, help="how many accepted assets to measure")
    ap.add_argument("--jobs", type=int, default=min(DEFAULT_JOBS, os.cpu_count() or 1))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    manifest = build_pack(
        args.assets, args.out, salt=args.salt, seed=args.seed,
        per_stratum=args.per_stratum, total_items=args.total_items,
        weights=tuple(w.strip() for w in args.weights.split(",") if w.strip()),
        budgets=tuple(b.strip() for b in args.budgets.split(",") if b.strip()),
        pool_want_accepted=args.pool, jobs=args.jobs,
        cache_dir=None if args.no_cache else "data/cache", verbose=args.verbose,
    )
    print(json.dumps({
        "n_items": manifest["n_items"], "counts": manifest["counts"],
        "cuts": manifest["stratum_cuts_commands"], "measurement": manifest["measurement"],
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
