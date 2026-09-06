# Running ToySVG on Kaggle

Kaggle Benchmarks is a good home for this benchmark for one reason above the rest:
the models come from Kaggle's own panel, through their model proxy, so every model
sees the same prompt, the same image and the same budget; nobody has to hand
over an API key to reproduce a number. The scoring is deterministic, so
the number on the leaderboard is the same one this repository computes.

## Three artifacts, three jobs

| file | job |
|---|---|
| `../tools/make_kaggle_bundle.py` | exports the harness, the pack, the ground truth and a stored run as one Kaggle Dataset |
| `toysvg_task_file.py` | the **drawing** task: PNG in, path data out, geometric score returned |
| `toysvg_control_task_file.py` | the **structure** task: identical input images, structural perception queries |
| `notebook/toysvg_kaggle.ipynb` | the reader: verifies the harness, replays the stored run, redraws the figures |

Task files are pushed with the CLI, which converts them to the notebook that backs
the task. That is why the task definition is at module level in those files rather
than imported from a library: the CLI validates the file it is handed for a
`@task` decorator, and a file that defines a task but never calls
`.run()`/`.evaluate()` is accepted and then records nothing.

## Workflow

```bash
# 1. build the dataset bundle (adds a leak check: the model-facing half is
#    scanned for hidden fields, and every item's PNG hash is verified)
python -m tools.make_kaggle_bundle --out dist/kaggle-toysvg \
    --slug <you>/toysvg-benchmark-harness --force
kaggle datasets create -p dist/kaggle-toysvg --dir-mode zip

# 2. credentials for local validation and for running tasks
kaggle b init -y

# 3. validate locally, then push. `-d` is per dataset and must be repeated on
#    every push: re-pushing without it silently detaches the data.
python toysvg_task_file.py
kaggle b t push toysvg-redraw-from-png \
    -f toysvg_task_file.py -d <you>/toysvg-benchmark-harness --wait
kaggle b t push toysvg-structure-questions \
    -f toysvg_control_task_file.py -d <you>/toysvg-benchmark-harness --wait

# 4. run, inspect, collect
kaggle b t models
kaggle b t run toysvg-redraw-from-png -m gemini-2.5-pro --wait
kaggle b t status toysvg-redraw-from-png
kaggle b t log toysvg-redraw-from-png -m gemini-2.5-pro
kaggle b t download toysvg-redraw-from-png -o ./results
kaggle b t publish toysvg-redraw-from-png

# 5. cost and the combined table
kaggle b quota
kaggle b leaderboard <you>/<benchmark> --show
```

Adding both tasks to one *benchmark* is done in the Kaggle web UI: the CLI manages
tasks, and a benchmark is a curated collection of them.

## What runs where

| | runs on | needs |
|---|---|---|
| `replay` mode in the notebook | offline or air-gapped environments | the dataset |
| `own-key` mode in the notebook | anywhere with outbound HTTPS | a key in Kaggle Secrets |
| the pushed tasks | Kaggle's infrastructure | nothing from you but the CLI |

## Verifying without Kaggle

The SDK is not needed to check any of this. Two local checks run the real code
paths:

```bash
python -m kaggle.toysvg_task --dry-run        # score stored answers through the bridge
python -m kaggle.test_task_files              # run both task files against a stub SDK
```

`test_task_files` substitutes a fake `kaggle_benchmarks` module, drives both task
files with answers already on disk, and asserts the two failures Kaggle's own
guidance warns about: an `@task` name that does not match the push slug, and a task
file that never runs its task. It also checks that the prompt states the budget,
that the image is attached, that reasoning is disabled, and that prompts omit
asset identifiers.

What it cannot check is whether the real SDK's argument names match this reading of
its documentation. That is what `kaggle b init -y && python toysvg_task_file.py`
is for, and it is worth doing before the first push.

## Prompt Formulation
 
The frozen prompt of `svgbench/prompts.py` (version `p1`), unchanged, because a
score computed with a different prompt is not comparable with one computed here.
It specifies the command budget, illustrates output syntax with an open zigzag,
and omits asset names, categories, or provenance cues. Reasoning is disabled:
a reasoning model spends output tokens on hidden scratchpads and returns paths
truncated mid-command, measuring token limits rather than spatial perception.

## Reading a leaderboard number

Each score is paired with the range it came from: shape-blind floors, and a tracer
ceiling computed from the same PNG at the same budget. Two consequences worth
knowing before comparing models:

- **Raw scores are not comparable across items.** The third demo item's ground
  truth needs 66 commands against a budget of 48, so no budget-fitting answer can
  equal it; its reference is the ceiling `C(k) = 0.482`, and `rho = S / C(k)` is
  the number to compare.
- **The control separates seeing from drawing.** A model that fails both tasks
  cannot see the structure; a model that passes the control and fails the drawing
  task can see it and cannot produce coordinates. That distinction is the reason
  the benchmark has two tasks rather than one.
