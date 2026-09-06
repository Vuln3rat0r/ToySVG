# %% [markdown]
# # ToySVG: copy a hidden outline figure
#
# One task. A model is shown one monochrome PNG of an outline figure, with no name,
# no category and no provenance, and must return SVG path data that reproduces it
# inside a command budget. The returned value is a deterministic geometric score
# against the hidden ground-truth path.
#
# Push it and run it against models on Kaggle's infrastructure:
#
# ```bash
# kaggle b init -y                     # once; writes MODEL_PROXY_* to .env
# python toysvg_task_file.py          # validate locally first (see the note below)
# kaggle b t push toysvg-redraw-from-png \
#     -f toysvg_task_file.py -d <you>/toysvg-benchmark-harness --wait
# kaggle b t run toysvg-redraw-from-png -m gemini-2.5-pro --wait
# kaggle b t status toysvg-redraw-from-png
# kaggle b t download toysvg-redraw-from-png -o ./results
# kaggle b t publish toysvg-redraw-from-png
# ```
#
# `-d` attaches the dataset carrying the harness and the pack. Repeat the flag per
# dataset, and repeat it on every push: re-pushing without it silently detaches
# what was attached before. Creating the *benchmark* itself (the curated collection
# of tasks) is done in the Kaggle web UI; the CLI manages tasks only.
#
# The score has no judge and no threshold in it. It is region overlap, boundary
# agreement and topology against the exact hidden path, with translation and scale
# fitted away first, so the same answer always yields the same number.

# %%
# %% [markdown]
# ## What the model sees
#
# The pack ships in two halves and this file only ever touches the first:
# `data/pack/` holds the blind PNGs and a manifest with no names; `data/gt/` holds
# the ground truth and is read by the scorer alone. The check below re-asserts that
# split before a single model call is spent.

# %%
import contextlib
import json
import os
import random
import sys
import time
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

#: Reasoning models spend the output budget on hidden tokens and come back with a
#: path cut off mid-command, which measures the token cap rather than perception.
#: Valid values are "none", "low", "medium", "high".
REASONING = "none"

#: Where the harness may be mounted. A notebook finds an attached dataset at
#: /kaggle/input/<slug>, while a benchmark task mounts it under
#: /kaggle/input/datasets/<owner>/<slug> -- a layout that is not documented and was
#: only visible in the creation log, which is why the failure below dumps what it
#: found. Patterns are tried shallowest-first so the shortest match wins. Locally the
#: same file is validated with `python toysvg_task_file.py`, so the working
#: directory and TOYSVG_ROOT are searched too. Duplicated in the control task file
#: because the root has to be known before the harness can be imported.
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
GT_DIR = ROOT / "data" / "gt"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

print(f"harness  {ROOT}")
print(f"pack     {PACK_DIR}")

# %%
# %% [markdown]
# ## The evaluation set
#
# One row per item: the PNG the model receives, its id, and the budget. Nothing
# else crosses into the task: stratum, split, and asset names are withheld.

# %%
tasks = ft.public_tasks(PACK_DIR)
items, strata = [], []
for item_id in sorted(tasks):
    rec = tasks[item_id]
    items.append((item_id, str(PACK_DIR / rec["input_png"]), int(rec["k"])))
    hidden = json.loads((GT_DIR / f"{item_id}.json").read_text())
    strata.append({"item_id": item_id, "stratum": hidden["stratum"]})

# The list is walked by the task itself, not handed to the scheduler as
# `evaluation_data`, and the difference is the whole point of the cell below. A
# scheduled row is one *element*, and an element is a record in its own right:
# twelve elements per model are twelve numbers beside the mean, and a display that
# picks one of them shows a single item's score under the rubric's name. Stratum
# and ground-truth path never enter the list at all: both are hidden fields, and
# the scorer locates the ground truth from the item id itself.
print(pd.DataFrame(strata).to_string(index=False))

# %%
# %% [markdown]
# ## Artifacts: what a run leaves behind
#
# Every item's raw answer is logged beside its components, and a three-panel strip
# is written for every item: the image the model saw, what it drew, and the hidden
# truth. Those files are the record a report is assembled from. Two of the strips
# are displayed in this cell; the rest are written but not shown, because image
# payload is carried on every model run and the evidence a number is wrong lives in
# the file, not in the cell.
#
# Both paths are overridable so that a local validation run does not write into the
# dataset it is validating.

# %%
FINDINGS = Path(os.environ.get("TOYSVG_FINDINGS", str(WORK / "toysvg-decompositions.jsonl")))
PREVIEW_DIR = Path(os.environ.get("TOYSVG_PREVIEWS", str(WORK / "previews")))
#: Chosen in advance, not picked from the results: one easy stratum and the hardest
#: item in the pack, so the two strips show a clean copy and a failed one.
SHOWN_INLINE = ("01001100afd9128f", "16f8e20c45edadc1")

ft.configure(pack_dir=PACK_DIR, findings_path=FINDINGS,
             reasoning=REASONING, model_label="kbench")


def _load_findings(path: Path) -> list:
    """Rows the log holds, in the order they were scored."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _write_previews(rows: list) -> None:
    """Write a strip per item, then display the two chosen in advance."""
    written = 0
    for row in rows:
        if row.get("task") == "structure" or "answer" not in row:
            continue
        if ft.preview_png(str(row["answer"]), str(row["item_id"]), PREVIEW_DIR):
            written += 1
    print(f"previews written {written} -> {PREVIEW_DIR}")
    try:
        from IPython.display import Image, display
    except Exception:  # noqa: BLE001  (not a notebook: the files are the point)
        return
    for item_id in SHOWN_INLINE:
        path = PREVIEW_DIR / f"{item_id}.png"
        row = next((r for r in rows if r.get("item_id") == item_id), None)
        if not path.exists() or row is None:
            continue
        print(f"{item_id[:8]}  input | model (as measured) | hidden truth   "
              f"S={float(row.get('score', 0.0)):.3f}  gate={row.get('gate', '?')}  "
              f"ceiling={row.get('ceiling_score')}")
        display(Image(filename=str(path)))


# %%
# %% [markdown]
# ## The task
#
# One task, one recorded element, and that element's value is the mean.
#
# The loop over the pack lives inside the task rather than in `evaluation_data`,
# and the difference is not stylistic. A scheduled row becomes an **element**, and
# an element is a record in its own right: twelve elements per model are twelve
# numbers beside the mean, and a display that picks one of them shows a single
# item's score under the rubric's name: which is exactly what happened to the
# first version of this file, whose inner task left twelve such records behind.
# Handing the platform one number removes the question: whichever level a view
# reads, the value is the mean over the pack.
#
# The cost of doing the loop here is that the SDK's per-element retry and its
# answer cache no longer apply, so both are written out below: every item is
# attempted, every attempt is logged, and an item that exhausts its attempts is
# scored through the ordinary gate path: an empty answer scores zero, is named
# as such in the log, and stays in the mean.

# %%
#: Attempts per item, and the backoff between them: 10s, 20s, 40s, then 80s at the
#: cap. A shorter schedule was measured and does not work. One push of this file came
#: back with `429 heavy load` on every prompt -- 36 attempts in 3.5 minutes -- and
#: scored a mean of 0.000: the provider was busy and the retry policy answered for
#: it. A model's own failures are what this task measures, a supplier's are not, so
#: the waiting is long enough to outlast a busy window rather than to decorate a log.
ITEM_ATTEMPTS = 5
RETRY_BACKOFF_S = 10.0
RETRY_BACKOFF_CAP_S = 80.0

#: If this share of the pack never receives an answer, the run raises instead of
#: reporting a mean. A number that is mostly "the provider would not talk to us" is
#: not a measurement of the model, and a leaderboard has no column for the
#: difference. Two unanswered items in a row abort the run early, so an outage costs
#: minutes rather than the whole schedule.
MAX_UNANSWERED_SHARE = 0.25
EARLY_ABORT_AFTER = 2

#: Seeded jitter: the retry schedule is part of the run's behaviour, and a run whose
#: timing changes between executions cannot be re-read.
_JITTER = random.Random(20260919)


# `description` is explicit because it defaults to the docstring and the server caps
# it at 255 characters -- a longer one fails validation *after* the notebook has run,
# so a whole run is spent to learn it.
@kbench.task(name="toysvg-redraw-from-png",
             description="Copy a pack of hidden outline figures as SVG path data "
                         "within a command budget. Reported score: the mean of the "
                         "per-item geometric scores against the hidden paths.")
def toysvg_redraw(llm) -> float:
    """Draw every figure in the pack; the reported value is the mean item score.

    The mean is unweighted -- each item counts once, whatever its stratum -- and it
    includes the gates that score zero. A mean computed over the items a model
    managed to answer measures something other than what the pack asks for.
    """
    logged_before = len(_load_findings(FINDINGS))
    unanswered: list = []
    for n, (item_id, png_path, k) in enumerate(items, start=1):
        answer, failure = "", None
        prompt_text = ft.prompt_for(int(k))
        for attempt in range(1, ITEM_ATTEMPTS + 1):
            try:
                answer = _prompt_item(llm, prompt_text, png_path, f"item_{n}_att_{attempt}")
                failure = None
                break
            except Exception as exc:  # noqa: BLE001  (provider and transport alike)
                failure = exc
                print(f"{item_id[:8]}  {n}/{len(items)}  attempt {attempt}/"
                      f"{ITEM_ATTEMPTS} failed: {type(failure).__name__}: {str(failure)[:160]}")
                if attempt < ITEM_ATTEMPTS:
                    delay = min(RETRY_BACKOFF_S * 2 ** (attempt - 1), RETRY_BACKOFF_CAP_S)
                    delay *= 0.75 + 0.5 * _JITTER.random()
                    print(f"{item_id[:8]}  waiting {delay:.0f}s before attempt "
                          f"{attempt + 1}/{ITEM_ATTEMPTS}")
                    time.sleep(delay)
        if failure is not None:
            unanswered.append(item_id)
            print(f"{item_id[:8]}  every attempt raised {type(failure).__name__}; "
                  "scoring an empty answer, and the item stays in the mean")
            # An outage that begins at item one is not going to end at item two.
            if len(unanswered) == n and n >= EARLY_ABORT_AFTER:
                raise RuntimeError(
                    f"the first {n} items got no answer from the provider "
                    f"({type(failure).__name__}: {str(failure)[:120]}); stopping rather "
                    "than measuring the supply of a model rather than the model")
        rec = ft.score_answer(str(answer), item_id)
        print(f"{item_id[:8]}  {n}/{len(items)}  gate={rec['gate']:<11} "
              f"S={rec['score']:.4f}  region={rec['s_region']:.3f} "
              f"edge={rec['s_edge']:.3f} topo={rec['s_topology']:.3f}  "
              f"cmds={rec['n_commands']}/{rec['k']}  ceiling={rec['ceiling_score']}")

    # The mean comes from the log, which is now the complete record: every item is
    # logged exactly once, because each one is scored here rather than cached by the
    # SDK. The check below is that the log holds one row per item -- the arithmetic
    # cannot then be an average over a subset of the pack.
    rows = _load_findings(FINDINGS)[logged_before:]
    if len(unanswered) > MAX_UNANSWERED_SHARE * len(items):
        raise RuntimeError(
            f"{len(unanswered)} of {len(items)} items never reached the model "
            f"({', '.join(i[:8] for i in unanswered)}); this run measured the "
            "provider's availability, and a mean over it would be published as a "
            "score")
    if not rows:
        print("no item was scored; reporting 0.0")
        return 0.0
    if len(rows) != len(items):
        raise RuntimeError(f"{len(rows)} items logged for a pack of {len(items)}; "
                           "the mean would not be over the whole pack")
    mean = ft.mean_score(rows)
    print(ft.decompose(rows))
    print(f"gates: {ft.gate_tally(rows)}")
    print(f"mean score {mean:.4f} over {len(rows)} items  <- the reported value")
    _write_previews(rows)
    return float(mean)


# %%
# %% [markdown]
# ## Running it
#
# `.run()` has to be called at module level. A pushed file that defines a task but
# never runs it is accepted by the validator and then produces nothing: the silent
# failure that turns a benchmark into an empty leaderboard. `kbench.llm` is the
# placeholder the platform substitutes per model, which is what lets "Add Models"
# schedule this same task across the whole panel without touching the code.

# %%
toysvg_redraw.run(kbench.llm)
