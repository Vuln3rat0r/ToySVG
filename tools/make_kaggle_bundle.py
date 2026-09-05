#!/usr/bin/env python3
"""Export the harness and a task pack as an uploadable Kaggle Dataset.

A Kaggle notebook cannot see this machine, and the repository is not published, so
everything the notebook needs travels as one dataset: the harness code, the
evaluation pack (model-facing half), the ground truth (hidden half), and a
stored run for the no-key replay mode.

Two checks run at export time, because both are cheap here and expensive later:

* **No ground truth in the model-facing half.** Every file under ``pack/`` is
  scanned for the hidden fields (``gt_d``, ``name``, ``source_sha256``). If one
  appears, the export stops: a label leaking into the directory a runner reads is
  the single mistake that would invalidate every number.
* **Every item is complete.** Each public task has its PNG and its ground-truth
  record, and every PNG hash matches the recorded one.

    python -m tools.make_kaggle_bundle --out dist/kaggle-toysvg

The dataset id is ``<owner>/toysvg-benchmark-harness``, with the owner read from
the authenticated Kaggle account, because an id naming the wrong account is
rejected by the server only at upload time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: Keys that identify the shape, its source, or its geometry. Checked as JSON keys
#: rather than as substrings, because the pack manifest legitimately records the
#: ground-truth *directory* and a substring test confuses that with the path itself.
FORBIDDEN_KEYS = frozenset({
    "gt_d", "gt_fill_rule", "source_sha256", "stroke_median_px", "ink_area_px",
    "topology", "n_commands_gt", "n_subpaths_gt",
})

#: The complete schema of a model-facing task record: an id, a budget, a canvas,
#: and a PNG with its digest. Nothing else may appear.
ALLOWED_TASK_KEYS = frozenset({
    "item_id", "budget", "k", "canvas", "input_png", "input_sha256",
})

CODE = [
    ("svgbench", ("pathdata.py", "raster.py", "svgdoc.py", "metrics.py", "align.py",
                  "corpus.py", "tasks.py", "prompts.py", "runner.py", "score.py",
                  "diag.py", "__init__.py")),
    ("tools", ("__init__.py", "figstyle.py", "make_figures.py", "make_metric_figures.py",
               "make_metric_primers.py", "explain_score.py")),
    ("kaggle", ("__init__.py", "toysvg_task.py", "toysvg_task_file.py",
                "toysvg_control_task_file.py", "build_notebook.py",
                "test_task_files.py")),
]


def _copy_code(root: Path, dest: Path) -> List[str]:
    """Copy the harness into ``dest`` (which is the bundle's ``code/`` directory)."""
    written: List[str] = []
    for package, files in CODE:
        (dest / package).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = root / package / name
            if not src.exists():
                continue
            shutil.copy2(src, dest / package / name)
            written.append(f"{package}/{name}")
    # Also expose the bridge at the top level
    for bridge_name in ("toysvg_task.py",):
        bridge = root / "kaggle" / bridge_name
        if bridge.exists():
            shutil.copy2(bridge, dest / bridge_name)
            written.append(bridge_name)
    # The notebook reads the pins to report version drift rather than discovering
    # it in a re-run, so the file has to travel with the bundle.
    reqs = root / "requirements.txt"
    if reqs.exists():
        shutil.copy2(reqs, dest / "requirements.txt")
        shutil.copy2(reqs, dest.parent / "requirements.txt")
        written.append("requirements.txt")
    return written


def _keys_in(blob: object) -> set:
    """Every key appearing anywhere in a JSON document."""
    found = set()
    if isinstance(blob, dict):
        for key, value in blob.items():
            found.add(str(key))
            found |= _keys_in(value)
    elif isinstance(blob, list):
        for value in blob:
            found |= _keys_in(value)
    return found


def _check_pack(pack_dir: Path, gt_dir: Path) -> List[Dict[str, object]]:
    """Verify the blind half is blind, and that every item is complete."""
    tasks = sorted((pack_dir / "tasks").glob("*.json"))
    if not tasks:
        raise SystemExit(f"no task records in {pack_dir / 'tasks'}")

    for path in sorted(pack_dir.rglob("*.json")):
        keys = _keys_in(json.loads(path.read_text(encoding="utf-8")))
        leaked = keys & FORBIDDEN_KEYS
        if leaked:
            raise SystemExit(f"leak: {sorted(leaked)} in model-facing file {path}")

    items: List[Dict[str, object]] = []
    for path in tasks:
        rec = json.loads(path.read_text(encoding="utf-8"))
        extra = set(rec) - ALLOWED_TASK_KEYS
        if extra:
            raise SystemExit(f"leak: {path.name} carries fields outside the public schema: "
                             f"{sorted(extra)}")
        iid = str(rec["item_id"])
        png = pack_dir / str(rec["input_png"])
        gt = gt_dir / f"{iid}.json"
        if not png.exists():
            raise SystemExit(f"{iid}: missing {png}")
        if not gt.exists():
            raise SystemExit(f"{iid}: missing ground truth {gt}")
        digest = hashlib.sha256(png.read_bytes()).hexdigest()
        if digest != rec["input_sha256"]:
            raise SystemExit(f"{iid}: PNG hash does not match the manifest")
        hidden = json.loads(gt.read_text(encoding="utf-8"))
        if not hidden.get("gt_d"):
            raise SystemExit(f"{iid}: ground-truth record carries no path data")
        items.append({"item_id": iid, "stratum": hidden.get("stratum", ""),
                      "budget": rec.get("budget"), "k": rec.get("k")})
    return items


def _dataset_metadata(slug: str, title: str, items: Sequence[Dict[str, object]]) -> Dict[str, object]:
    strata: Dict[str, int] = {}
    for it in items:
        strata[str(it["stratum"])] = strata.get(str(it["stratum"]), 0) + 1
    spread = ", ".join(f"{k}: {v}" for k, v in sorted(strata.items()))
    return {
        "title": title,
        "id": slug,
        "licenses": [{"name": "other"}],
        "subtitle": f"Deterministic outline redraw benchmark harness with a {len(items)}-item evaluation pack",
        "description": (
            "ToySVG is a deterministic benchmark evaluating how well vision models perceive monochrome "
            "outline figures and redraw them as compact SVG paths. The harness includes a strict path parser, "
            "pinned coverage rasteriser, similarity alignment optimization, geometric metrics (soft IoU, "
            "boundary distance, topology), floor anchors, and procedural tracer ceilings.\n\n"
            "The task pack is organized in two halves: `data/pack/` is the model-facing half (blind PNGs and "
            "manifests, with names and geometry removed) and `data/gt/` is the hidden ground-truth half (exact "
            "target paths evaluated solely by the offline scorer).\n\n"
            f"Items: {len(items)} ({spread}).\n\n"
            "All scores are deterministic and bit-for-bit reproducible: identical inputs yield identical floating-point "
            "metrics without external rendering dependencies. The figures derive from Phosphor Icons (MIT license)."
        ),
        # Kaggle validates these against its own tag vocabulary and silently drops
        # what it does not recognise, so this list holds only the one label the server
        # is known to accept (it uses display labels, not slugs: "computer vision").
        "keywords": ["computer vision"],
    }


def _write_readme(dest: Path, items: Sequence[Dict[str, object]], slug: str) -> None:
    (dest / "README.md").write_text(
        "# ToySVG: harness and evaluation pack\n\n"
        "Attach this dataset to a notebook, then add `code/` to `sys.path`:\n\n"
        "```python\n"
        "import sys\n"
        f"ROOT = '/kaggle/input/{slug.split('/')[-1]}'\n"
        "sys.path.insert(0, f'{ROOT}/code')\n"
        "```\n\n"
        "## Layout\n\n"
        "| path | role |\n|---|---|\n"
        "| `code/svgbench/` | the harness: parser, rasteriser, metrics, alignment, scorer |\n"
        "| `code/tools/` | the figure and trace tools used by the specification |\n"
        "| `code/kaggle/toysvg_task.py` | the bridge: defines the Kaggle task |\n"
        "| `data/pack/input/` | the blind PNGs a model receives |\n"
        "| `data/pack/tasks/` | the model-facing task records |\n"
        "| `data/gt/` | the hidden ground truth, read only by the scorer |\n"
        "| `runs/` | a stored three-item run, for replay without an API key |\n\n"
        "The split between `data/pack/` and `data/gt/` is enforced by construction: "
        "`svgbench.tasks.load_public` reads the first and cannot reach the second, and "
        "`load_hidden` is documented as scorer-only. The exporter refuses to build a "
        "bundle in which a hidden field appears anywhere under `data/pack/`.\n\n"
        f"Items in this pack: {len(items)}.\n",
        encoding="utf-8")


def _default_slug(dataset: str = "toysvg-benchmark-harness") -> str:
    """``<owner>/<dataset>``, owner taken from the authenticated Kaggle account.

    Reads only the ``username`` field of the CLI's config file. The API key lives
    in the same file and is never read, and nothing from the file is printed.
    """
    config_dirs = [os.environ.get("KAGGLE_CONFIG_DIR"), Path.home() / ".kaggle"]
    for conf_dir in config_dirs:
        if not conf_dir:
            continue
        try:
            cfg = json.loads((Path(conf_dir) / "kaggle.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        owner = cfg.get("username")
        if owner:
            return f"{owner}/{dataset}"
    return f"CHANGEME/{dataset}"


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".")
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--gt", default="data/gt")
    ap.add_argument("--run", default="data/runs/gemini-2.5-flash-demo")
    ap.add_argument("--out", default="dist/kaggle-toysvg")
    ap.add_argument("--slug", default=None,
                    help="<kaggle-username>/<dataset-slug> "
                         "(default: the authenticated Kaggle account)")
    ap.add_argument("--title", default="ToySVG benchmark harness and evaluation pack")
    ap.add_argument("--force", action="store_true", help="overwrite an existing bundle")
    args = ap.parse_args(argv)

    if args.slug is None:
        args.slug = _default_slug()
    if args.slug.startswith("CHANGEME/"):
        raise SystemExit(
            "no Kaggle username found; pass --slug <your-username>/toysvg-benchmark-harness "
            "(or sign in with `kaggle config`)"
        )
    root = Path(args.root).resolve()
    dest = Path(args.out)
    if dest.exists():
        if not args.force:
            raise SystemExit(f"{dest} exists; pass --force to rebuild")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    pack_dir = root / args.pack
    gt_dir = root / args.gt
    items = _check_pack(pack_dir, gt_dir)

    code_files = _copy_code(root, dest / "code")
    shutil.copytree(pack_dir, dest / "data" / "pack")
    shutil.copytree(gt_dir, dest / "data" / "gt")
    run_dir = root / args.run
    if run_dir.exists():
        shutil.copytree(run_dir, dest / "runs" / run_dir.name)

    for junk in dest.rglob("__pycache__"):
        shutil.rmtree(junk)
    for junk in dest.rglob("*.pyc"):
        junk.unlink()

    _write_readme(dest, items, args.slug)
    (dest / "dataset-metadata.json").write_text(
        json.dumps(_dataset_metadata(args.slug, args.title, items), indent=2), encoding="utf-8")

    manifest = {
        "slug": args.slug,
        "n_items": len(items),
        "items": items,
        "code_files": code_files,
        "replay_run": run_dir.name if run_dir.exists() else None,
        "requirements": (root / "requirements.txt").read_text(encoding="utf-8").strip().splitlines()
        if (root / "requirements.txt").exists() else [],
    }
    (dest / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"bundle: {dest}  ({len(items)} items, {len(code_files)} code files, "
          f"{size / 1024:.0f} KiB)")
    print("  " + "\n  ".join(sorted(str(p.relative_to(dest)) for p in dest.rglob("*")
                                    if p.is_file() and p.parent == dest)))
    print("\nblind half checked: no " + ", ".join(sorted(FORBIDDEN_KEYS))
          + " anywhere under data/pack/, and every task record matches the public schema")
    n_gt = sum(1 for _ in (dest / "data" / "gt").rglob("*"))
    print(f"\nthis bundle carries the hidden half ({n_gt} files under data/gt/): a copy that is "
          "made public exposes the ground truth for these items, and any item whose answers "
          "are visible no longer measures copying. Keep this bundle private, or export the "
          "blind half alone before sharing.")
    print(f"\nupload (needs the Kaggle CLI and your credentials, not run here):\n"
          f"  kaggle datasets create -p {dest} --dir-mode zip\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
