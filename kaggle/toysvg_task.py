"""ToySVG as a Kaggle Benchmarks task.

The harness and the Kaggle Benchmarks SDK agree on the essentials: a model
receives an image, returns text, and the task returns one number. So the bridge is
thin: it feeds the PNG in as an attachment, hands the answer to the same
``svgbench.score`` code path the specification describes, and returns the
composite $\score$ as a float, which is what a Kaggle leaderboard aggregates.

Three things about this bridge are deliberate:

* **Direct geometric measurement.** The task returns continuous measured fidelity
  without subjective judge models or arbitrary discretization thresholds.
  The leaderboard aggregates closed-form geometric quantities, and the gates
  that score zero (unparsable, over budget, off canvas) reflect the task's failure
  taxonomy rather than harness errors.
* **``reasoning="none"`` by default.** A reasoning model spends the output budget
  on hidden tokens and returns a path cut off mid-command; the visible answer then
  measures the token cap rather than perception. The setting is recorded with
  every run.
* **The decomposition travels with the score.** Components and gates are appended
  to a JSONL file as each item completes, so a leaderboard number that looks wrong
  can be taken apart afterwards without re-running anything.

Importable without Kaggle installed: ``kaggle_benchmarks`` is imported lazily, so
``python -m kaggle.toysvg_task --dry-run`` works on a laptop and proves the
scoring path before a single model call is spent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

#: Where the notebook sets these once, before tasks are defined.
CONFIG: Dict[str, Any] = {
    "pack_dir": None,
    "findings_path": None,
    "reasoning": "none",
    "model_label": "kaggle",
}


def configure(pack_dir: Path | str, findings_path: Optional[Path | str] = None,
              reasoning: str = "none", model_label: str = "kaggle") -> None:
    """Point the bridge at a pack and at a writable place for decompositions."""
    CONFIG["pack_dir"] = Path(pack_dir)
    CONFIG["findings_path"] = None if findings_path is None else Path(findings_path)
    CONFIG["reasoning"] = reasoning
    CONFIG["model_label"] = model_label


def _pack() -> Path:
    pack_dir = CONFIG["pack_dir"]
    if pack_dir is None:
        raise RuntimeError("call toysvg_task.configure(pack_dir=...) first")
    return Path(pack_dir)


def public_tasks(pack_dir: Path | str) -> Dict[str, Dict[str, object]]:
    """The model-facing records, keyed by item id."""
    from svgbench.tasks import load_public

    return {str(t["item_id"]): t for t in load_public(pack_dir)}


def prompt_for(k: Optional[int]) -> str:
    """The frozen prompt. Identical to the one the local runner uses."""
    from svgbench.prompts import build_prompt

    return build_prompt(k)


def score_answer(answer: str, item_id: str, pack_dir: Optional[Path | str] = None,
                 model: Optional[str] = None, finish_reason: Optional[str] = None) -> Dict[str, Any]:
    """Score one answer with the full harness: gates, alignment, three terms.

    Returns the composite plus every component, so a caller can report the
    number and diagnose it from the same call.
    """
    from svgbench.score import score_item
    from svgbench.tasks import load_hidden

    pack_dir = Path(pack_dir or _pack())
    tasks = public_tasks(pack_dir)
    if item_id not in tasks:
        raise KeyError(f"item {item_id!r} is not in pack {pack_dir}")
    task = tasks[item_id]
    png_path = pack_dir / str(task["input_png"])

    item = load_hidden(pack_dir, item_id)
    rec = score_item(answer, item, model or str(CONFIG["model_label"]),
                     finish_reason=finish_reason, want_ceiling=True, png_path=png_path)
    out = {
        "item_id": item_id,
        # The raw answer travels with its score. A report quotes it and a preview
        # renders it; neither can be reconstructed from the numbers afterwards.
        "answer": answer,
        "stratum": item.stratum,
        "split": item.split,
        "k": item.k,
        "gate": rec.gate,
        "scored": bool(rec.scored),
        "score": float(rec.score),
        "partial_score": rec.partial_score,
        "s_region": float(rec.s_region),
        "s_edge": float(rec.s_edge),
        "s_topology": float(rec.s_topology),
        "soft_iou": float(rec.soft_iou),
        "hard_iou": float(rec.hard_iou),
        "chamfer_px": float(rec.chamfer_px),
        "hausdorff95_px": float(rec.hausdorff95_px),
        "area_ratio": float(rec.area_ratio),
        "gt_components": int(rec.gt_components),
        "gt_holes": int(rec.gt_holes),
        "pred_components": int(rec.pred_components),
        "pred_holes": int(rec.pred_holes),
        "topology_error": int(rec.topology_error),
        "align_scale": float(rec.align_scale),
        "align_tx": float(rec.align_tx),
        "align_ty": float(rec.align_ty),
        "n_commands": int(rec.n_commands),
        "n_subpaths": int(rec.n_subpaths),
        "ceiling_score": rec.ceiling_score,
        "ceiling_commands": rec.ceiling_commands,
        "rho": rec.rho,
        "floors": dict(rec.floors),
        "note": getattr(rec, "note", ""),
    }
    _record(out)
    return out


def _record(row: Dict[str, Any]) -> None:
    """Append the decomposition, so nothing depends on the notebook staying alive."""
    path = CONFIG["findings_path"]
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def identity_check(pack_dir: Path | str, item_id: str) -> Dict[str, Any]:
    """Test reflexivity, and report whether the ground truth fits its own budget.

    Reflexivity is a property of the *geometry*: the hidden path scored against
    itself must give exactly 1. That is ``floors['identity']``. Feeding the hidden
    path in as if it were a model answer is a different question, because the
    budget gate applies to answers: some ground truths were captured from
    source icons that need more commands than the tier allows. Such an item is not
    broken: the budget tier asks what is achievable at ``k``, which is what the
    tracer ceiling measures. But it has to be visible, or a harness check will look
    like a failure the day it meets one.
    """
    from svgbench.tasks import load_hidden

    pack_dir = Path(pack_dir)
    item = load_hidden(pack_dir, item_id)
    rec = score_item_for(item, pack_dir, item_id)
    return {
        "item_id": item_id,
        "identity_floor": float((rec.get("floors") or {}).get("identity", 0.0)),
        "gt_commands": int(rec["n_commands"]),
        "k": rec["k"],
        "gt_over_budget": bool(rec["k"] is not None and rec["n_commands"] > rec["k"]),
        "gt_gate": rec["gate"],
        "gt_partial_score": rec["partial_score"],
        "ceiling_score": rec["ceiling_score"],
    }


def score_item_for(item, pack_dir: Path, item_id: str) -> Dict[str, Any]:
    """Score the item's own ground truth through the answer path."""
    return score_answer(item.gt_d, item_id, pack_dir=pack_dir)


def budget_report(pack_dir: Path | str) -> List[Dict[str, Any]]:
    """How many items have a ground truth that exceeds its own command budget."""
    pack_dir = Path(pack_dir)
    return [identity_check(pack_dir, iid) for iid in sorted(public_tasks(pack_dir))]


def decompose(rows: List[Dict[str, Any]]) -> str:
    """A fixed-width table of the decomposition, for a notebook cell."""
    head = ("item      stratum  gate        S      region edge   topo   rho    "
            "chamfer  cmds")
    lines = [head, "-" * len(head)]
    for r in rows:
        rho = "-" if not r.get("rho") else f"{r['rho']:.2f}"
        ceil = "-" if not r.get("ceiling_score") else f"{r['ceiling_score']:.3f}"
        lines.append(
            f"{r['item_id'][:8]:<9} {r.get('stratum', ''):<8} {r['gate']:<11} "
            f"{r['score']:.3f}  {r['s_region']:.3f}  {r['s_edge']:.3f}  "
            f"{r['s_topology']:.3f}  {rho:<6} {r['chamfer_px']:>7.2f}  "
            f"{r['n_commands']:>4}  (ceiling {ceil})")
    return "\n".join(lines)


def mean_score(rows: List[Dict[str, Any]]) -> float:
    """The unweighted mean of the scored items.

    Unweighted on purpose: every item counts once, whatever its stratum. Weighting
    by ceiling or by ground-truth complexity would make the headline number depend
    on the pack's mix rather than on the model.
    """
    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    return sum(scores) / len(scores) if scores else 0.0


def gate_tally(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """How many items ended in each gate, so a mean can be read with its failures."""
    counts: Dict[str, int] = {}
    for row in rows:
        gate = str(row.get("gate", ""))
        counts[gate] = counts.get(gate, 0) + 1
    return dict(sorted(counts.items()))


def preview_png(answer: str, item_id: str, out_dir: Path | str,
                pack_dir: Optional[Path | str] = None) -> Optional[Path]:
    """Write a three-panel strip: what the model saw, what it drew, what was hidden.

    The middle panel is rendered from the *aligned* coverage the scorer measured, so
    the picture shows exactly the geometry the number is about. Files are written for
    every item, because they are the record a report is assembled from; the notebook
    displays only a couple of them, since composing and displaying all of them spends
    benchmark time without adding evidence.
    """
    import numpy as np
    from PIL import Image

    from svgbench.raster import coverage_to_uint8
    from svgbench.score import gt_coverage, render_answer
    from svgbench.tasks import load_hidden

    pack_dir = Path(pack_dir or _pack())
    item = load_hidden(pack_dir, item_id)
    png = pack_dir / str(public_tasks(pack_dir)[item_id]["input_png"])
    seen = np.asarray(Image.open(png).convert("L"))
    cov = render_answer(answer, item)
    drawn = seen.copy() if cov is None else coverage_to_uint8(cov)
    if cov is None:
        drawn[:] = 255  # unparsable: an empty panel, not a copy of the input
    truth = coverage_to_uint8(gt_coverage(item))
    divider = np.full((seen.shape[0], 2), 128, dtype=np.uint8)
    strip = np.concatenate([seen, divider, drawn, divider, truth], axis=1)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{item_id}.png"
    Image.fromarray(strip).save(path)
    return path


def score_control_answer(answer: str, item_id: str, png_path: str,
                         pack_dir: Optional[Path | str] = None) -> Dict[str, Any]:
    """Score one structure answer field by field, and record it.

    The drawing task keeps its evidence in the decomposition log; the control needs
    the same, or its per-item answers exist only inside the notebook that produced
    them and a report has nothing to quote.
    """
    payload = json_payload(answer)
    verdicts = control_field_verdicts(answer, png_path, payload=payload)
    out = {
        "item_id": item_id,
        "task": "structure",
        "answer": answer,
        "correct": sum(1 for ok in verdicts.values() if ok),
        "asked": len(CONTROL_SCORED),
        "fields": verdicts,
        # Whether an answer yielded a JSON object at all, and whether it arrived in a
        # code fence. Without these two flags a format refusal and a wrong reading
        # are the same row of the report.
        "parsed": payload is not None,
        "fenced": "```" in str(answer),
        "truth": structure_truth(png_path),
    }
    _record(out)
    return out


# The task definitions themselves live in the pushable task files
# (`toysvg_task_file.py`, `toysvg_control_task_file.py`) rather than here: the
# Kaggle CLI validates the file it is handed for a `@task` decorator, so the
# decorated function has to be at module level in the file that gets pushed.
# This module is the library those files call into.


#: The prompt for the perception-only control. Same image, structural questions,
#: no drawing: a different construct, which is why it is a separate task.
CONTROL_PROMPT = (
    "Answer with a single JSON object and nothing else:\n"
    '{"regions": <int>, "holes": <int>, "heavier_side": "left"|"right"|"even",\n'
    ' "left_right_symmetric": true|false, "dominant_axis": "<angle in degrees>"}\n'
    "Count separate ink regions and enclosed holes; judge the rest from the image only."
)


#: Fields of the control prompt scored against pixel ground truth.
CONTROL_SCORED = ("regions", "holes", "heavier_side", "left_right_symmetric",
                  "dominant_axis")
#: An axis is a direction, so it is compared modulo $180^\circ$.
AXIS_TOLERANCE_DEG = 15.0


def structure_truth(png_path: str) -> Dict[str, object]:
    """The structure of an item, read from the pixels the model was shown.

    Computed from the PNG rather than from the hidden path, so every field is
    answerable by looking at the image and every field has a single correct value.
    """
    import numpy as np
    from PIL import Image
    from svgbench.metrics import soft_iou, topology_counts

    gray = np.asarray(Image.open(png_path).convert("L"), dtype=float) / 255.0
    coverage = 1.0 - gray
    mask = coverage >= 0.5
    top = topology_counts(mask)

    bits = gray < 0.9  # antialiasing included, so the mass is not quantised
    ys, xs = np.nonzero(bits)
    height, width = gray.shape
    left = float(coverage[:, : width // 2].sum())
    right = float(coverage[:, width // 2:].sum())
    total = max(left + right, 1e-9)
    balance = (left - right) / total
    side = "even" if abs(balance) < 0.05 else ("left" if balance > 0 else "right")

    symmetric = bool(soft_iou(mask.astype(float), mask[:, ::-1].astype(float)) >= 0.90)

    axis = 0.0
    if len(xs) > 2:
        cov = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
        axis = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)

    return {"regions": int(top["components"]), "holes": int(top["holes"]),
            "heavier_side": side, "left_right_symmetric": symmetric,
            "dominant_axis": axis}


def json_payload(answer: str) -> Optional[Dict[str, Any]]:
    """The JSON object an answer contains, tolerating a markdown code fence.

    A fence is transport, not content. The prompt asks for one JSON object, and a
    model that returns it inside ```json has still answered the question -- so
    rejecting it would score a formatting habit as a perception failure, which is
    precisely the confusion this control exists to resolve. Measured: one model
    fenced every answer and read 0/60 as a result, against 28/60 for a model that
    did not, with no evidence in either direction about what it saw.

    ``None`` when there is no JSON object in the answer at all.
    """
    text = str(answer).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        got = json.loads(text.strip().strip("`").strip())
    except Exception:  # noqa: BLE001 - absence of an object is the answer here
        return None
    return got if isinstance(got, dict) else None


def control_field_verdicts(answer: str, png_path: str,
                           payload: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Which scored fields an answer got right, by name.

    A count cannot distinguish a model that misread the image from one that answered
    in a shape the parser could not read; the names can, and an answer with no JSON
    object is reported as wrong on every field, beside the ``parsed`` flag that says
    so.
    """
    got = json_payload(answer) if payload is None else payload
    if got is None:
        return {field: False for field in CONTROL_SCORED}
    truth = structure_truth(png_path)
    verdicts: Dict[str, bool] = {}
    for field in CONTROL_SCORED:
        want = truth[field]
        try:
            if field == "dominant_axis":
                diff = abs(float(got[field]) - float(want)) % 180.0
                ok = min(diff, 180.0 - diff) <= AXIS_TOLERANCE_DEG
            elif field == "left_right_symmetric":
                ok = bool(got[field]) == bool(want)
            elif field in ("regions", "holes"):
                ok = int(got[field]) == int(want)
            else:
                ok = str(got[field]).strip().lower() == str(want)
        except (KeyError, TypeError, ValueError):
            ok = False
        verdicts[field] = bool(ok)
    return verdicts


def control_answer_is_correct(answer: str, png_path: str) -> tuple:
    """Score one control answer as ``(correct, asked)`` over the pixels' verdict."""
    verdicts = control_field_verdicts(answer, png_path)
    return (sum(1 for ok in verdicts.values() if ok), len(CONTROL_SCORED))


# ---------------------------------------------------------------- dry run


def _dry_run(pack_dir: Path, run_dir: Optional[Path]) -> int:
    """Score the stored answers through this module, with no model and no Kaggle.

    This is the check that the bridge is wired correctly: the numbers it prints
    must equal the ones the specification reports for the same run.
    """
    configure(pack_dir=pack_dir, findings_path=None)
    tasks = public_tasks(pack_dir)
    rows: List[Dict[str, Any]] = []

    if run_dir is not None and (run_dir / "results.jsonl").exists():
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            iid = str(rec.get("item_id"))
            raw = rec.get("raw_file") or f"raw/{iid}.txt"
            path = run_dir / str(raw)
            if iid not in tasks or not path.exists():
                continue
            rows.append(score_answer(path.read_text(encoding="utf-8"), iid,
                                     finish_reason=rec.get("finish_reason")))
    print(decompose(rows))

    # Reflexivity, and the corpus fact that goes with it.
    checks = budget_report(pack_dir)
    bad = [c for c in checks if c["identity_floor"] < 0.999]
    over = [c for c in checks if c["gt_over_budget"]]
    print(f"\n{len(rows)} answers scored through the Kaggle bridge.")
    print(f"reflexivity: {len(checks) - len(bad)}/{len(checks)} items score 1.000000 "
          f"against their own ground truth")
    if bad:
        print(f"  not reflexive: {[c['item_id'][:8] for c in bad]}")
    if over:
        detail = ", ".join(f"{c['item_id'][:8]} {c['gt_commands']}>{c['k']}" for c in over)
        print(f"ground truth over budget in {len(over)}/{len(checks)} items ({detail}):")
        print("  for those the reference is the tracer ceiling, not the identity, "
              "because no answer that fits the budget can equal the source path")
    return 0 if not bad else 1


def _main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="data/pack")
    ap.add_argument("--run", default="data/runs/gemini-2.5-flash-demo")
    ap.add_argument("--dry-run", action="store_true",
                    help="score stored answers (or the ground truth) with no model call")
    args = ap.parse_args(argv)
    if not args.dry_run:
        ap.error("nothing to do: pass --dry-run, or import define_task() from a notebook")
    run_dir = Path(args.run) if args.run else None
    return _dry_run(Path(args.pack), run_dir)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
