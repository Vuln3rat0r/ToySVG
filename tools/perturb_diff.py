#!/usr/bin/env python3
"""Side-by-side of the vendor SVG and the perturbed truth, per OOD-PERTURB item.

The pack has two halves: the PNG a model receives, and the hidden ``gt_d`` its
answer is measured against. For the three ``OOD-PERTURB`` items the two halves
are deliberately different files, so the fairest way to report the warp is to
export both halves as ordinary SVGs and let a reader open them.

This tool writes, per item, into ``--out``:

* ``<name>.original.svg``  -- the vendor's own document, byte-for-byte, straight
  out of the cached ``@phosphor-icons/core`` tarball;
* ``<name>.perturbed.svg`` -- the hidden truth, i.e. the same path after flatten
  / displace / simplify, expressed as absolute ``M``/``L``/``Z``;
* ``<name>.overlay.svg``   -- both at once, original in blue, perturbed in red;
* ``<name>.compare.png``   -- original | the input PNG the model received |
  perturbed | overlay, so the whole item is legible in one image;
* ``compare.html``         -- the three items stacked, for a browser.

It also re-derives the perturbation from the vendor path and checks it against
the stored ``gt_d`` string, so the export cannot silently disagree with the
scoring target.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from svgbench.pathdata import parse_path
from svgbench.raster import coverage_to_uint8, render_coverage
from svgbench.tasks import (
    CANVAS,
    PERTURB_AMPLITUDE_PX,
    PERTURB_FREQ,
    PERTURB_PHASE,
    PERTURB_TOLERANCE_PX,
    SS,
    _displacement,
    perturb_path,
)

ROOT = Path(__file__).resolve().parents[1]
TARBALL = ROOT / "data" / "cache" / "phosphor-core-2.1.1.tgz"
ORANGE = "#c8501e"
BLUE = "#1f4f9c"
RED = "#b3202c"
INK = "#101418"


def vendor_d(tarball: Path, weight: str, name: str) -> str:
    """Return the single ``d`` attribute of the vendor's own asset, verbatim."""
    member = f"package/assets/{weight}/{name}.svg"
    with tarfile.open(tarball, "r:gz") as tf:
        raw = tf.extractfile(member)
        if raw is None:
            raise FileNotFoundError(member)
        text = raw.read().decode("utf-8")
    d = re.search(r'd="([^"]+)"', text)
    if d is None:
        raise ValueError(f"no path data in {member}")
    return d.group(1)


def cmd_count(d: str) -> int:
    """Number of path commands in an expression (what the budget charges)."""
    return len(re.findall(r"[MmLlHhVvCcSsQqTtAaZz]", d))


def code_pair(name: str, d_orig: str, d_pert: str) -> str:
    """Both halves as pastable documents, plus their counts."""
    return (
        f"# {name}: vendor original (verbatim from @phosphor-icons/core 2.1.1, regular)\n"
        f"# chars {len(d_orig)}   commands {cmd_count(d_orig)}\n"
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 256 256\" fill=\"#101418\">\n"
        f"  <path d=\"{d_orig}\"/>\n"
        "</svg>\n\n"
        f"# {name}: perturbed truth (the hidden gt_d this item is scored against)\n"
        f"# chars {len(d_pert)}   commands {cmd_count(d_pert)}\n"
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 256 256\" fill=\"#101418\">\n"
        f"  <path d=\"{d_pert}\"/>\n"
        "</svg>\n"
    )


def document(d: str, fill: str = INK) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" '
        f'fill="{fill}"><path fill-rule="nonzero" d="{d}"/></svg>\n'
    )


def overlay_document(d_orig: str, d_pert: str) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 256 256" fill="none">'
        f'<path d="{d_orig}" fill="{BLUE}" opacity="0.55"/>'
        f'<path d="{d_pert}" fill="{RED}" opacity="0.55"/>'
        "</svg>\n"
    )


def raster(d: str, size: int = CANVAS, ss: int = 4) -> np.ndarray:
    """Coverage of ``d`` as dark-on-light uint8, the input-PNG convention."""
    parsed = parse_path(d)
    cov = render_coverage(parsed.subpaths, size=size, ss=ss, fill_rule=parsed.fill_rule)
    return coverage_to_uint8(cov)


def bbox(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def displacement_stats(d_orig: str) -> Dict[str, float]:
    """What the field actually does to the vendor's own vertices."""
    pts = [p for sp in parse_path(d_orig).subpaths for p in sp.points]
    mags = [float(np.hypot(*_displacement(x, y))) for x, y in pts]
    x0, y0, x1, y1 = bbox(pts)
    moved = [(x + _displacement(x, y)[0], y + _displacement(x, y)[1]) for x, y in pts]
    mx0, my0, mx1, my1 = bbox(moved)
    outside = sum(1 for x, y in moved if x < 0 or y < 0 or x > CANVAS or y > CANVAS)
    return {
        "vertices": len(pts),
        "disp_max_px": max(mags),
        "disp_mean_px": float(np.mean(mags)),
        "bbox_orig": (x0, y0, x1, y1),
        "bbox_moved": (mx0, my0, mx1, my1),
        "vertices_off_canvas": outside,
    }


def fmt_box(b: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v:.1f}" for v in b) + "]"


def panel(img: np.ndarray, scale: int = 2) -> Image.Image:
    return Image.fromarray(img).resize((CANVAS * scale, CANVAS * scale), Image.NEAREST)


def strip(images: Sequence[Image.Image], gap: int = 8) -> Image.Image:
    w = sum(i.width for i in images) + gap * (len(images) - 1)
    h = max(i.height for i in images)
    out = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for img in images:
        out.paste(img.convert("RGB"), (x, 0))
        x += img.width + gap
    return out


def build(out_dir: Path, pack_dir: Path, tarball: Path, weight: str) -> List[Dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((pack_dir / "manifest.json").read_text())
    items: List[Dict[str, object]] = []
    panels: List[Tuple[str, Image.Image]] = []

    for gtp in sorted((ROOT / "data" / "gt").glob("*.json")):
        hidden = json.loads(gtp.read_text())
        if hidden["split"] != "OOD-PERTURB":
            continue
        item_id = str(hidden["item_id"])
        name = str(hidden["name"])
        d_orig = vendor_d(tarball, weight, name)
        d_pert = str(hidden["gt_d"])

        # The export must agree with the scoring target, so re-derive it.
        derived = perturb_path(parse_path(d_orig))
        agrees = derived == d_pert

        (out_dir / f"{name}.original.svg").write_text(document(d_orig))
        (out_dir / f"{name}.perturbed.svg").write_text(document(d_pert))
        (out_dir / f"{name}.overlay.svg").write_text(overlay_document(d_orig, d_pert))
        (out_dir / f"{name}.d.txt").write_text(code_pair(name, d_orig, d_pert))

        img_orig = raster(d_orig)
        img_pert = raster(d_pert)
        input_png = np.asarray(Image.open(pack_dir / "input" / f"{item_id}.png").convert("L"))
        # Whose geometry is in the image the model sees? Measure, do not assume.
        err_pert = float(np.mean(np.abs(input_png.astype(float) - img_pert.astype(float))))
        err_orig = float(np.mean(np.abs(input_png.astype(float) - img_orig.astype(float))))

        composed = strip([panel(img_orig), panel(input_png), panel(img_pert)])
        composed.save(out_dir / f"{name}.compare.png")
        panels.append((f"{name}  ({item_id[:8]})", composed))

        stats = displacement_stats(d_orig)
        items.append(
            {
                "item_id": item_id,
                "name": name,
                "split": hidden["split"],
                "d_original": d_orig,
                "d_perturbed": d_pert,
                "derivation_matches_gt": agrees,
                "chars": (len(d_orig), len(d_pert)),
                "commands": (cmd_count(d_orig), int(hidden["n_commands_gt"])),
                "subpaths": (len(parse_path(d_orig).subpaths), len(parse_path(d_pert).subpaths)),
                "input_is_perturbed": err_pert < err_orig,
                "mean_abs_err_to_input": (err_orig, err_pert),
                **stats,
            }
        )

    page = ["<!doctype html><meta charset='utf-8'><title>OOD-PERTURB: vendor vs truth</title>",
            "<style>body{font:14px/1.5 system-ui,sans-serif;background:#fff;color:#101418;margin:24px}"
            "img{display:block;border:1px solid #d0d5dd;margin:6px 0 22px}"
            "h2{font-size:15px;margin:22px 0 4px}"
            "code{font-family:ui-monospace,Menlo,monospace;font-size:12px}"
            "caption,figcaption{color:#475467}</style>",
            "<h1>OOD-PERTURB: what the vendor drew vs what the truth is</h1>",
            "<p>Left: <b>original</b>. Middle: <b>the PNG the model receives</b>. "
            "Right: <b>perturbed truth</b> that the answer is scored against. "
            "The overlay file puts both paths on one canvas (original blue, truth red).</p>"]
    for label, composed in panels:
        buf = io.BytesIO()
        composed.save(buf, format="PNG")
        import base64

        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        page.append(f"<h2>{html.escape(label)}</h2>")
        page.append(f"<img width='{composed.width}' src='data:image/png;base64,{b64}'>")
    (out_dir / "compare.html").write_text("\n".join(page) + "\n")
    return items


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "compare"))
    ap.add_argument("--pack", default=str(ROOT / "data" / "pack"))
    ap.add_argument("--tarball", default=str(TARBALL))
    ap.add_argument("--weight", default="regular")
    ap.add_argument("--show", type=int, default=0, help="print full d strings for N items")
    args = ap.parse_args(argv)

    items = build(Path(args.out), Path(args.pack), Path(args.tarball), args.weight)
    print(f"field: dx = {PERTURB_AMPLITUDE_PX}*sin(2pi*{PERTURB_FREQ[0]}*y/{CANVAS} + {PERTURB_PHASE[0]})"
          f"   dy = {PERTURB_AMPLITUDE_PX}*sin(2pi*{PERTURB_FREQ[1]}*x/{CANVAS} + {PERTURB_PHASE[1]})"
          f"   simplify {PERTURB_TOLERANCE_PX} px")
    for it in items:
        eo, ep = it["mean_abs_err_to_input"]  # type: ignore[misc]
        print(
            f"\n{it['name']}  {it['item_id'][:8]}  {it['split']}\n"
            f"  chars    original {it['chars'][0]}  -> perturbed {it['chars'][1]}\n"
            f"  path cmds original {it['commands'][0]}"
            f"  -> perturbed {it['commands'][1]}\n"
            f"  subpaths original {it['subpaths'][0]}"
            f"  -> perturbed {it['subpaths'][1]}\n"
            f"  vertices {it['vertices']}  displacement max {it['disp_max_px']:.2f} px"
            f"  mean {it['disp_mean_px']:.2f} px\n"
            f"  bbox original {fmt_box(it['bbox_orig'])}"  # type: ignore[arg-type]
            f"  -> moved {fmt_box(it['bbox_moved'])}\n"  # type: ignore[arg-type]
            f"  vertices pushed off canvas: {it['vertices_off_canvas']}\n"
            f"  input PNG vs original render mean|err| {eo:.2f}"
            f"   vs perturbed render {ep:.2f}  -> input is the perturbed drawing:"
            f" {it['input_is_perturbed']}\n"
            f"  re-derived truth == stored gt_d: {it['derivation_matches_gt']}"
        )
    if args.show:
        for it in items[: args.show]:
            print(f"\n--- {it['name']} original d ---\n{it['d_original']}")
            print(f"\n--- {it['name']} perturbed d ---\n{it['d_perturbed']}")
    print(f"\nwrote {len(items)} items to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
