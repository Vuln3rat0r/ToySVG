# %% [markdown]
# # ToySVG: structure questions (perception-only control)
#
# The same images, no drawing. The model is asked what is *in* the picture:
# how many separate ink regions, how many enclosed holes, which side carries more
# ink, whether the figure is left-right symmetric, and the angle of its dominant
# axis. Every field is scored against the pixels the model was shown, so the
# control is answerable by looking and needs no ground-truth path.
#
# It exists to separate two failures the drawing task alone confounds: a model that
# cannot see the structure will fail both tasks, while a model that sees it and
# cannot produce coordinates fails only the drawing. A large gap between the two
# is the signature of a drawing failure; agreement at the floor is the signature of
# a perception failure.
#
# Push it as its own task, then add both tasks to the same benchmark in the Kaggle
# web UI (the CLI manages tasks; a *benchmark* is a curated collection of them):
#
# ```bash
# kaggle b init -y
# python toysvg_control_task_file.py
# kaggle b t push toysvg-structure-questions \
#     -f toysvg_control_task_file.py -d <you>/toysvg-benchmark-harness --wait
# kaggle b t run toysvg-structure-questions -m gemini-2.5-pro --wait
# kaggle b t download toysvg-structure-questions -o ./results
# ```

# %%
import contextlib
import json
import os
import sys
from pathlib import Path

import kaggle_benchmarks as kbench
import pandas as pd
from kaggle_benchmarks.content_types import images


def _chat_context(name: str):
    """Return an isolated chat context when supported by the SDK."""
    if hasattr(kbench, "chats") and hasattr(kbench.chats, "new"):
        try:
            return kbench.chats.new(name, orphan=True)
        except TypeError:
            return kbench.chats.new(name)
    return contextlib.nullcontext()


def _prompt_item(llm, prompt_text: str, png_path: str, context_label: str) -> str:
    """Prompt the LLM with an image, gracefully handling reasoning parameter rejections.

    Tries configured REASONING first. If rejected by the provider/proxy (e.g. gpt-6-astra
    or reasoning models where reasoning cannot be disabled with 'none'), falls back to
    'low', then default/None. Each attempt runs in its own isolated chat context so
    failed calls never pollute the conversation history with dangling turns or duplicate images.
    """
    levels = [REASONING]
    for lvl in ("low", None):
        if lvl not in levels:
            levels.append(lvl)

    last_exc = None
    for idx, r_level in enumerate(levels):
        with _chat_context(f"{context_label}_r{idx}"):
            img = images.from_path(str(png_path))
            kw = {"image": img}
            if r_level is not None:
                kw["reasoning"] = r_level
            try:
                return str(llm.prompt(prompt_text, **kw))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue

    if last_exc is not None:
        raise last_exc
    return ""


REASONING = "none"

#: Where the harness may be mounted: a notebook sees /kaggle/input/<slug>, a
#: benchmark task sees /kaggle/input/datasets/<owner>/<slug>. Shallowest-first, so the
#: shortest match wins. Duplicated from the drawing task file, because the root has to
#: be known before the harness can be imported.
MOUNT_GLOBS = (
    "*/code/toysvg_task.py",
    "*/*/code/toysvg_task.py",
    "*/*/*/code/toysvg_task.py",
)

#: The mount root, overridable because the search below is otherwise untestable
#: anywhere except on Kaggle.
INPUT_ROOT = Path(os.environ.get("TOYSVG_INPUT_ROOT", "/kaggle/input"))


def _diagnose(inputs: Path) -> str:
    """Dump the environment for the one failure that is purely environmental.

    An empty or unexpected /kaggle/input is not something the task code can reason
    its way out of, and the creation log is the only place it can be seen. Variable
    *names* are listed, never values: this runs on Kaggle's infrastructure, where the
    process environment holds credentials.
    """
    lines = ["attach the ToySVG dataset with -d <you>/toysvg-benchmark-harness",
             f"cwd: {Path.cwd()}"]
    for var in sorted(os.environ):
        if var.startswith("TOYSVG"):
            lines.append(f"{var}={os.environ[var]}")
        elif var.startswith(("KAGGLE", "MODEL_PROXY")):
            lines.append(f"{var} is set (value withheld)")
    if not inputs.exists():
        lines.append(f"{inputs} does not exist")
        return "\n".join(lines)
    entries = sorted(path for path in inputs.rglob("*")
                     if len(path.relative_to(inputs).parts) <= 4)
    lines.append(f"{inputs} ({len(entries)} entries to depth 4):")
    shown = entries[:60]
    lines += [f"  {path.relative_to(inputs)}" + ("/" if path.is_dir() else "")
              for path in shown]
    if len(entries) > len(shown):
        lines.append(f"  ... {len(entries) - len(shown)} more")
    if not entries:
        lines.append("  (empty)")
    return "\n".join(lines)


def _module_report() -> str:
    """Which of the harness's dependencies this environment actually has.

    The task environment is not the notebook image: it is a bare virtualenv, and an
    ImportError from inside the harness says only that one module is absent. Naming
    all of them turns that into a decision about what to depend on.
    """
    lines = [f"python {sys.version.split()[0]}"]
    for name in ("numpy", "PIL", "scipy", "skimage", "pandas", "matplotlib"):
        try:
            module = __import__(name)
        except Exception as exc:  # noqa: BLE001  (absence is the answer)
            lines.append(f"  {name}: MISSING ({type(exc).__name__})")
        else:
            lines.append(f"  {name}: {getattr(module, '__version__', 'unknown')}")
    return "\n".join(lines)


def _root() -> Path:
    override = os.environ.get("TOYSVG_ROOT")
    if override and (Path(override) / "code" / "toysvg_task.py").exists():
        return Path(override)
    if (Path.cwd() / "code" / "toysvg_task.py").exists():
        return Path.cwd()
    if (Path.cwd() / "kaggle" / "toysvg_task.py").exists():
        return Path.cwd().parent
    inputs = INPUT_ROOT
    if inputs.exists():
        for pattern in MOUNT_GLOBS:
            hit = next(iter(sorted(inputs.glob(pattern))), None)
            if hit is not None:
                return hit.parent.parent  # <mount>/code/toysvg_task.py -> <mount>
    raise SystemExit(_diagnose(inputs))


ROOT = _root()
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "kaggle"))

#: The harness measures with scipy's exact Euclidean distance transform and with
#: scikit-image's region labelling and marching squares. The task environment is a
#: bare virtualenv that provides numpy, pandas and Pillow but neither of these, so
#: they are installed on first use. Measured environment: python 3.11.15, numpy
#: 2.3.5, Pillow 11.3.0, pandas 2.3.3 (excluding scipy, scikit-image, matplotlib).
#: Distribution name and the module it provides, kept as a pair because
#: ``find_spec`` takes an *import* name and ``scikit-image`` is not one -- testing the
#: distribution name reports it missing forever, and the guard then runs pip on every
#: single run.
NUMERIC_DEPS = (("scipy", "scipy"), ("scikit-image", "skimage"))


def _ensure_deps() -> None:
    """Install the harness's numeric dependencies if this environment lacks them."""
    import importlib.util
    import subprocess

    if os.environ.get("TOYSVG_SKIP_INSTALL"):
        return  # a local validation run must not install into the machine running it
    missing = [dist for dist, module in NUMERIC_DEPS
               if importlib.util.find_spec(module) is None]
    if not missing:
        return
    print(f"installing {missing} with {sys.executable}")
    done = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet",
         "--disable-pip-version-check", *missing],
        capture_output=True, text=True)
    print(f"pip exited {done.returncode}")
    if done.returncode != 0:
        print((done.stdout or "")[-600:], (done.stderr or "")[-600:])


_ensure_deps()

try:
    import toysvg_task as ft  # noqa: E402
    from svgbench.tasks import load_public as _load_public  # noqa: E402,F401
except Exception as exc:  # noqa: BLE001  (the report is the point)
    raise SystemExit(f"the harness failed to import: {type(exc).__name__}: {exc}\n\n"
                     + _module_report()) from None

print("task environment:")
print(_module_report())

PACK_DIR = ROOT / "data" / "pack"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
print(f"harness  {ROOT}\npack     {PACK_DIR}")

# %%
tasks = ft.public_tasks(PACK_DIR)
evaluation_data = pd.DataFrame([
    {"item_id": item_id, "png_path": str(PACK_DIR / tasks[item_id]["input_png"])}
    for item_id in sorted(tasks)
])
print(f"{len(evaluation_data)} items")

# %%
# %% [markdown]
# ## The task
#
# The same two-task split as the drawing task, for the same reason: the outer task's
# return value is what Kaggle reports, so the outer one answers every item and
# returns the normalized accuracy rate rather than one item's count. Per-item
# answers and field-by-field verdicts go to a JSONL file, which is what a report
# quotes from.

# %%
FINDINGS = Path(os.environ.get("TOYSVG_FINDINGS", str(WORK / "toysvg-control.jsonl")))
ft.configure(pack_dir=PACK_DIR, findings_path=FINDINGS,
             reasoning=REASONING, model_label="kbench-control")


def _load_findings(path: Path) -> list:
    """Rows the log holds, in the order they were scored."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def structure_one(llm, item_id: str, png_path: str, n: int = 1, total: int = 32) -> tuple[int, int]:
    """Report the structure of one image: regions, holes, mass, symmetry, axis."""
    answer = _prompt_item(llm, ft.CONTROL_PROMPT, png_path, f"ctrl_{item_id[:8]}")
    rec = ft.score_control_answer(str(answer), item_id, str(png_path))
    print(f"{item_id[:8]}  {n}/{total}  {rec['correct']}/{rec['asked']}  "
          f"score={float(rec['correct']) / max(int(rec['asked']), 1):.3f}")
    return (int(rec["correct"]), int(rec["asked"]))


# `description` is explicit because it defaults to the docstring, and the server caps
# it at 255 characters; passing it keeps the docstring free to be documentation.
@kbench.task(name="toysvg-structure-questions",
             description="Report the structure of every figure in the pack: regions, "
                         "holes, ink mass, left-right symmetry, dominant axis. "
                         "Reported score: accuracy rate across all evaluated fields.")
def structure_questions(llm) -> float:
    """Report each figure's structure; the reported value is the accuracy rate.

    Every field is scored against the pixels the model was shown, so the task is
    answerable by looking at the image and needs no ground-truth path. It separates
    a model that cannot see the structure from one that sees it and cannot produce
    coordinates: the first fails both tasks, the second only the drawing.
    """
    logged_before = len(_load_findings(FINDINGS))
    rows_data = evaluation_data.to_dict("records")
    got_total = 0
    asked_total = 0
    for n, row in enumerate(rows_data, start=1):
        item_id = str(row["item_id"])
        png_path = str(row["png_path"])
        correct, asked = structure_one(llm, item_id, png_path, n=n, total=len(rows_data))
        got_total += correct
        asked_total += asked

    print(f"completed {len(rows_data)} / {len(rows_data)}")
    rows = _load_findings(FINDINGS)[logged_before:]
    for row in rows:
        missed = [name for name, ok in (row.get("fields") or {}).items() if not ok]
        print(f"  {str(row.get('item_id'))[:8]}: missed {missed or 'nothing'}")
    if rows:
        print(f"mean item accuracy "
              f"{sum(r['correct'] / max(r['asked'], 1) for r in rows) / len(rows):.3f}"
              "  (per-item, unweighted)")
    rate = got_total / asked_total if asked_total else 0.0
    print(f"structure accuracy {got_total}/{asked_total} = {rate:.3f}  <- the reported value")
    return float(rate)


# %%
# %% [markdown]
# ## Running it
#
# Top level, as above: a file that defines a task and never runs it pushes cleanly and
# records nothing.

# %%
structure_questions.run(kbench.llm)
