#!/usr/bin/env python3
"""Run both Kaggle task files against a stub SDK.

The Kaggle SDK is not installed here and the Model Proxy is not reachable from this
machine, so this test substitutes a fake ``kaggle_benchmarks`` module that
implements exactly the surface the task files use (``task``, ``llm``,
``client.enable_cache``, ``content_types.images``) and drives them with answers
that are already stored on disk.

What it proves: the task files find the harness, the prompt reaches the model with
the image attached, an answer comes back through the real scorer, the return value
is the right type for a leaderboard, and the decomposition is written.

The task files are also checked against the SDK's own source, which is where three
of these rules come from: the return annotation must be a type Kaggle registers for
automatic inference (a bare ``tuple`` raises ``TypeError`` at import time), a
``from __future__ import annotations`` import would stringify that annotation and
break the same lookup, and every column of ``evaluation_data`` is passed to the task
as a keyword argument and bound with a strict ``signature.bind`` (a column the task
does not declare fails once per item). What remains unproven until a real push is
the server-side conversion and the Model Proxy call itself.

    python -m kaggle.test_task_files
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import json
import os
import runpy
import shutil
import sys
import tempfile
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]

#: A stored answer, reused so the score in the test is a known number.
DEMO_RUN = ROOT / "data" / "runs" / "gemini-2.5-flash-demo"
PROMPTS: List[str] = []

#: The return annotations the SDK registers for automatic inference
#: (``_infer_result_type`` looks the annotation up in ``results.types``). Anything
#: else is a ``TypeError`` from ``Task.__post_init__`` at import time.
REGISTERED_RETURNS = frozenset({
    "None", "bool", "int", "float", "dict", "tuple[int, int]", "tuple[float, float]",
})


class _FakeImage:
    def __init__(self, path: str) -> None:
        self.path = path


class _FakeClient:
    @contextlib.contextmanager
    def enable_cache(self):
        yield


class _FakeChats:
    @contextlib.contextmanager
    def new(self, name: str = ""):
        yield


class _FakeLLM:
    """Returns the stored answer for whichever item was just described."""

    def __init__(self, answers: Dict[str, str], control: bool = False) -> None:
        self.answers = answers
        self.control = control
        self.item_hint: Optional[str] = None

    def prompt(self, message, image=None, reasoning=None, **kwargs):  # noqa: ANN001
        assert image is not None, "the task must attach the image"
        assert reasoning in ("none", "low", None), f"reasoning should be none/low/None, got {reasoning!r}"
        PROMPTS.append(message)
        name = Path(image.path).stem if isinstance(image, _FakeImage) else ""
        if name in self.answers:
            return _FakeStructured(self.answers[name])
        if self.control:
            return _FakeStructured(json.dumps({
                "regions": 1, "holes": 1, "heavier_side": "even",
                "left_right_symmetric": True, "dominant_axis": 90.0}))
        # No stored answer for this item: reply with the prompt's own syntax
        # example, which is valid path data and deliberately the wrong figure. The
        # point is to exercise parse, align and score rather than the empty gate.
        from svgbench.prompts import FIXTURE_PATH

        return _FakeStructured(
            '<svg viewBox="0 0 256 256"><path d="' + FIXTURE_PATH +
            '" fill="none" stroke="currentColor" stroke-width="16"/></svg>')


class _FakeStructured(str):
    """A string that also supports .strip() and comparison, like a Message."""


class _FakeRunResult:
    def __init__(self, results: List[object]) -> None:
        self.completed_runs = _Runs(results)
        self.errored_runs: List[object] = []
        self._n = len(results)

    def __len__(self) -> int:
        return self._n


class _Runs:
    def __init__(self, results: Sequence[object]) -> None:
        self._results = list(results)
        self._frame = None

    def __len__(self) -> int:
        return len(self._results)

    def as_dataframe(self):
        import pandas as pd

        return pd.DataFrame({"result": self._results})


def _build_fake_sdk(answers: Dict[str, str], control: bool = False) -> types.ModuleType:
    kbench = types.ModuleType("kaggle_benchmarks")
    recorded: Dict[str, object] = {}

    def task(name=None, **kwargs):  # noqa: ANN001
        def wrap(fn):
            fn.task_name = name

            def evaluate(llm=None, evaluation_data=None, **kw):  # noqa: ANN001
                model = llm[0] if isinstance(llm, (list, tuple)) else llm
                out = []
                # Emulate the real scheduler, which builds {**grid, **row.to_dict()}
                # and hands it to Task.run, whose signature.bind accepts nothing the
                # task does not declare. This stub used to filter columns down to a
                # whitelist, and that leniency is precisely why a stray column
                # survived it and would have failed every item on Kaggle.
                declared = set(inspect.signature(fn).parameters)
                for row in evaluation_data.to_dict("records"):
                    unexpected = sorted(set(row) - declared - {"_id"})
                    assert not unexpected, (
                        f"evaluation_data carries {unexpected}, which the task does "
                        "not declare; the real scheduler passes every column as a "
                        "keyword argument, and signature.bind then raises TypeError "
                        "once per item")
                    model.item_hint = row.get("item_id")
                    out.append(fn(model, **{k: v for k, v in row.items()
                                            if k in declared}))
                recorded["results"] = out
                return _FakeRunResult(out)

            def run(llm=None, **kw):  # noqa: ANN001
                # The return value is what a leaderboard shows, so it is captured.
                value = fn(llm or _FakeLLM(answers))
                recorded["run_result"] = value
                return value

            fn.evaluate = evaluate
            fn.run = run
            return fn

        return wrap

    kbench.task = task
    kbench.benchmark = task
    kbench.llm = _FakeLLM(answers, control=control)
    kbench.llms = {}
    kbench.client = _FakeClient()
    kbench.chats = _FakeChats()
    kbench.last_reasoning_traces = lambda: None
    content_types = types.ModuleType("kaggle_benchmarks.content_types")
    images = types.ModuleType("kaggle_benchmarks.content_types.images")
    images.from_path = lambda p: _FakeImage(str(p))
    images.from_url = lambda u: _FakeImage(str(u))
    images.from_base64 = lambda b, format="jpeg": _FakeImage("<base64>")
    content_types.images = images
    kbench.content_types = content_types
    kbench._recorded = recorded
    return kbench


def _stored_answers() -> Dict[str, str]:
    """Map each item id to the answer the demo run produced for it."""
    answers: Dict[str, str] = {}
    if not (DEMO_RUN / "results.jsonl").exists():
        return answers
    for line in (DEMO_RUN / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        raw = DEMO_RUN / rec.get("raw_file", f"raw/{rec['item_id']}.txt")
        if raw.exists():
            answers[str(rec["item_id"])] = raw.read_text(encoding="utf-8")
    return answers


def check_task_file(path: Path, slug: str) -> List[str]:
    """The pre-push checks Kaggle's own guidance implies.

    Two failures are undetectable downstream: a file whose ``@task`` name does not
    match the slug being pushed, and a file that defines a task but never runs it:
    the push succeeds and the task then records nothing.
    """
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    problems: List[str] = []
    names: List[str] = []
    if any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
           and any(a.name == "annotations" for a in n.names) for n in tree.body):
        problems.append("`from __future__ import annotations` turns the return "
                        "annotation into a string, so Kaggle's result-type lookup "
                        "fails at import time")
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            call = dec if isinstance(dec, ast.Call) else None
            if call is None:
                continue
            func = call.func
            attr = func.attr if isinstance(func, ast.Attribute) else ""
            if attr not in ("task", "benchmark"):
                continue
            given = None
            for kw in call.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    given = str(kw.value.value)
            names.append(given or node.name)
            # The server caps both at 255 characters and enforces it *after* the
            # notebook has run, so an over-long docstring costs a whole run to find
            # out. description defaults to the docstring when not given.
            description = None
            for kw in call.keywords:
                if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                    description = str(kw.value.value)
            if description is None:
                description = ast.get_docstring(node) or ""
            label = given or node.name
            for what, value in (("name", label), ("description", description)):
                if len(value) > 255:
                    problems.append(
                        f"{node.name}: the {what} is {len(value)} characters; the "
                        "server rejects anything over 255")
            if node.returns is not None:
                ann = ast.unparse(node.returns)
                if ann not in REGISTERED_RETURNS:
                    problems.append(
                        f"{node.name} is annotated `-> {ann}`, which Kaggle does not "
                        "register for automatic inference; Task.__post_init__ raises "
                        "TypeError on it at import time")
    if not names:
        problems.append("no @task-decorated function at module level")
    elif slug not in [n.replace("_", "-") for n in names]:
        problems.append(f"@task names {names} do not include the push slug {slug!r}")
    if "evaluate(" not in src and ".run(" not in src:
        problems.append("defines a task but never calls .run()/.evaluate(): silent no-op")
    return problems


def check_mount_layout(path: Path, tmp: Path) -> None:
    """The harness must be found in both layouts Kaggle uses.

    A notebook mounts an attached dataset at ``/kaggle/input/<slug>``; a benchmark
    task mounts it at ``/kaggle/input/datasets/<owner>/<slug>``. The second layout is
    undocumented and was found only in a creation log, after two pushes had already
    failed on it, so it is exercised here rather than trusted.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    keep = []
    for node in tree.body:
        # Only the imports _root and _diagnose need. The rest of the module's
        # top-level code imports the harness and the SDK, which is the very thing
        # under test and must not run here.
        if isinstance(node, ast.Import) and any(a.name == "os" for a in node.names):
            keep.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == "pathlib":
            keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in ("_root", "_diagnose"):
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("MOUNT_GLOBS", "INPUT_ROOT")
                for t in node.targets):
            keep.append(node)
    code = compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec")

    here = Path.cwd()
    saved = {k: os.environ.pop(k, None) for k in ("TOYSVG_ROOT", "TOYSVG_INPUT_ROOT")}
    try:
        os.chdir(tmp)
        os.environ["TOYSVG_INPUT_ROOT"] = str(tmp / "input")
        # Executed only now, because INPUT_ROOT is read once at import time.
        namespace: Dict[str, object] = {}
        exec(code, namespace)
        find_root = namespace["_root"]
        for layout in (("toysvg-benchmark-harness",),
                       ("datasets", "someowner", "toysvg-benchmark-harness")):
            shutil.rmtree(tmp / "input", ignore_errors=True)
            mount = tmp / "input" / Path(*layout)
            (mount / "code").mkdir(parents=True)
            (mount / "code" / "toysvg_task.py").touch()
            try:
                found = find_root()  # type: ignore[operator]
            except SystemExit as exc:
                # _root() reports a missing mount by exiting; as a test failure it
                # should read as one, keeping the diagnostic it printed.
                raise AssertionError(
                    f"{path.name}: the {'/'.join(layout)} layout was not found. "
                    f"The task file reported:\n{exc}") from None
            assert found == mount, (
                f"{path.name}: the {'/'.join(layout)} layout resolved to {found}, "
                f"not {mount}")
    finally:
        os.chdir(here)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_task_file(path: Path) -> None:
    """Execute a task file with the stub SDK in place.

    The file is executed whole rather than cell by cell: every markdown cell in it
    is already a Python comment, so the source is valid Python as written.
    """
    namespace: Dict[str, object] = {"__name__": "__main__", "__file__": str(path)}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)


def _main() -> int:
    answers = _stored_answers()
    if not answers:
        raise SystemExit("no stored answers found; run the demo model first")
    kbench = _build_fake_sdk(answers)
    control_sdk = _build_fake_sdk(answers, control=True)
    sys.modules["kaggle_benchmarks"] = kbench
    sys.modules["kaggle_benchmarks.content_types"] = kbench.content_types
    sys.modules["kaggle_benchmarks.content_types.images"] = kbench.content_types.images

    # The task files install the harness's numeric dependencies when an environment
    # lacks them. That is right on Kaggle and wrong here: a validation run must not
    # install packages into the machine running it.
    os.environ["TOYSVG_SKIP_INSTALL"] = "1"
    bundle_path = ROOT / "dist" / "kaggle-toysvg"
    os.environ.setdefault("TOYSVG_ROOT", str(bundle_path))
    root = Path(os.environ["TOYSVG_ROOT"])
    if not (root / "code" / "toysvg_task.py").exists():
        raise SystemExit(f"build the bundle first: {root} is missing")

    os.chdir(root)  # the task files locate the harness from the working directory
    out_root = Path(tempfile.mkdtemp())
    targets = (("toysvg_task_file.py", "toysvg-redraw-from-png", kbench),
               ("toysvg_control_task_file.py", "toysvg-structure-questions", control_sdk))
    for name, slug, sdk in targets:
        path = ROOT / "kaggle" / name
        problems = check_task_file(path, slug)
        assert not problems, f"{name}: {problems}"
        check_mount_layout(path, Path(tempfile.mkdtemp()))
        print(f"  {name}: mount search finds both notebook and task layouts")
        sys.modules["kaggle_benchmarks"] = sdk
        sys.modules["kaggle_benchmarks.content_types"] = sdk.content_types
        sys.modules["kaggle_benchmarks.content_types.images"] = sdk.content_types.images
        for mod in [m for m in sys.modules if m == "toysvg_task"]:
            del sys.modules[mod]
        # The artifacts go to a temporary directory, so a validation run cannot leave
        # a decomposition log or preview strips inside the bundle it is validating.
        findings = out_root / f"{slug}.jsonl"
        previews = out_root / f"{slug}-previews"
        os.environ["TOYSVG_FINDINGS"] = str(findings)
        os.environ["TOYSVG_PREVIEWS"] = str(previews)
        run_task_file(path)
        print(f"\n=== {name} (slug {slug}) ===")
        recorded = sdk._recorded  # type: ignore[attr-defined]
        reported = recorded["run_result"]
        logged = [json.loads(line) for line in findings.read_text(encoding="utf-8").splitlines()
                  if line.strip()]

        if name.startswith("toysvg_task_file"):
            # One element, and the reason is on the record: a scheduled row is a
            # result the platform can display on its own, so twelve of them put a
            # single item's score one click away from being read as the score. The
            # assertion is absence rather than a count, because a count that is
            # right today is a count that silently grows tomorrow.
            assert "results" not in recorded, (
                "the drawing task scheduled elements; each element is a separate "
                "result, and a view that shows one of them shows an item score")
            scored = [float(row["score"]) for row in logged]
            assert len(scored) > 1, "the pack should have several items"
            mean = sum(scored) / len(scored)
            best = max(scored)
            # The reported value is what a leaderboard shows. "The best item" and "the
            # average over the pack" are different claims, and only the mean is the one
            # this benchmark makes.
            assert isinstance(reported, float), \
                f"the task reported {type(reported).__name__}, not a float"
            assert abs(reported - mean) < 1e-9, f"reported {reported}, which is not the mean {mean}"
            assert abs(reported - best) > 1e-6, \
                f"reported {reported} is the best item's score, not a mean over items"
            assert len(logged) == len(scored), f"{len(logged)} logged vs {len(scored)} scored"
            assert all(row.get("answer") for row in logged), \
                "the raw answer is not in the decomposition log"
            assert all("score" in row for row in logged), "no score logged"
            strips = sorted(previews.glob("*.png"))
            assert len(strips) == len(scored), \
                f"{len(strips)} preview strips for {len(scored)} items"
            from PIL import Image as _Image
            with _Image.open(strips[0]) as strip:
                width, height = strip.size
            assert width == 3 * height + 4, \
                f"a strip is {width}x{height}: not three panels and two dividers"
            print(f"evaluated {len(scored)} items, best {best:.4f}")
            print(f"  reported {reported:.4f} == mean over items (best differs, so the "
                  "leaderboard cannot be showing a best-of)")
            print(f"  {len(logged)} answers logged with raw text, {len(strips)} strips "
                  f"{width}x{height} written")
        else:
            assert "results" not in recorded, (
                "the control task scheduled elements; each element is a separate "
                "result, and a view that shows one of them shows an item score")
            rate = float(reported)
            assert logged and all("fields" in row for row in logged), \
                "per-field verdicts are not logged"
            assert len(logged) > 1, "the pack should have several items"
            total_asked = 5 * len(logged)
            total_correct = sum(r["correct"] for r in logged)
            expected_rate = total_correct / total_asked
            assert abs(rate - expected_rate) < 1e-6, f"rate mismatch: {rate} vs {expected_rate}"
            # A fenced JSON answer is the same answer. One model wrapped every answer
            # in ```json and read 0/60 while a model that did not read 28/60 -- with no
            # evidence about what either saw. The control exists to separate a model
            # that cannot perceive the structure from one that cannot produce the
            # format, so the fence must not decide the score.
            ctl = sys.modules["toysvg_task"]
            root_bundle = Path(os.environ["TOYSVG_ROOT"])
            png = sorted((root_bundle / "data" / "pack" / "input").glob("*.png"))[0]
            bare = json.dumps(ctl.structure_truth(str(png)))
            perfect = ctl.control_field_verdicts(bare, str(png))
            assert all(perfect.values()), \
                f"an answer copied from the pixels must be fully correct, got {perfect}"
            fenced = ctl.control_field_verdicts("```json\n" + bare + "\n```", str(png))
            assert fenced == perfect, "a fenced JSON answer scored differently"
            assert ctl.json_payload("no json here") is None, "prose parsed as JSON"
            print(f"evaluated {len(logged)} items")
            print(f"  reported {rate:.4f} == structure accuracy rate ({total_correct}/{total_asked}); "
                  f"{len(logged)} answers logged with per-field verdicts")
            print("  a fenced JSON answer scores identically to a bare one")

    # The drawing task's evidence is its log, not a results object: that is the
    # point of the restructure, so the log is what this last check reads.
    drawing = out_root / "toysvg-redraw-from-png.jsonl"
    scored = [float(row["score"]) for row in
              (json.loads(line) for line in drawing.read_text(encoding="utf-8").splitlines()
               if line.strip())]
    assert max(scored) > 0.2, "the stored answer should score about 0.26"

    assert PROMPTS, "no prompt reached the model"
    assert "budget" in PROMPTS[0].lower(), "the budget is missing from the prompt"
    assert all(len(p) > 200 for p in PROMPTS), "the prompt looks truncated"
    for forbidden in ("phosphor", "icon name", ".svg"):
        assert forbidden not in PROMPTS[0].lower(), f"prompt leaks {forbidden!r}"

    print(f"\n{len(PROMPTS)} prompts sent, none naming the figure, budget stated, "
          f"image attached, reasoning disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
