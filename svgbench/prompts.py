"""The prompt protocol.

One string, identical for every model, every item, every split. The only field
that varies is the budget ``k``, because the budget is part of the task rather
than a hidden penalty.

Three properties are load-bearing and are the reason this lives in its own
module rather than being assembled at the call site:

* **Metadata exclusion.** Omits icon names, categories, libraries, and licenses
  to eliminate textual cues. The prompt is a function of ``k`` and the image alone.
* **Syntax is taught, the task is not.** One worked example of the output format,
  using a shape deliberately unlike anything in the corpus, so that malformed
  SVG does not masquerade as a perceptual failure.
* **Frozen.** Changing this string invalidates every score computed with it, so
  it carries a version and the version is recorded in every run manifest.

``PROMPT_VERSION`` must be bumped whenever the text changes. See
\\S\\ref{sec:prompt} of the specification.
"""

from __future__ import annotations

from typing import Optional

PROMPT_VERSION = "p1"

# The teaching example: an open zigzag with one cubic. It is a stroke-only,
# asymmetric figure that does not resemble any corpus item, and it exercises
# ``M``, ``L``, ``C`` and ``stroke-width``: the syntax a model actually needs.
FIXTURE_PATH = "M24 192 L64 96 L104 160 C120 184 152 88 176 64 L232 96"

_INSTRUCTIONS = """You are shown one image: a black-and-white line figure on a white background. It is an icon drawn as an outline: a thin black stroke, not a solid silhouette.

Reproduce that figure as SVG path data.

Rules:
1. Reply with SVG only. No explanation, no commentary, no markdown code fences.
2. Use the canvas 0 0 256 256. Either return a bare path expression, or a full <svg> element with viewBox="0 0 256 256"; both are accepted.
3. Prefer <path> elements. Fill and stroke are both treated as ink.
4. Preserve the figure's structure: the number of separate shapes, and any enclosed hole inside a shape.
5. Match the figure's position, size, and proportions within the canvas.
6. Do not add text, labels, shadows, or background rectangles.
"""

_BUDGET_LIMITED = """Budget: at most {k} drawing commands in total, counting each M, L, H, V, C, S, Q, T, A and Z as one command. A command may carry several coordinate pairs; for example "L10 10 20 20" is one command, not two. Going over the budget scores zero, so stay strictly within it. With this few commands you must choose them to capture the figure's overall form and its most important features.
"""

_BUDGET_FREE = """Budget: none. There is no limit on the number of commands. Use as many as you need to reproduce the figure accurately, and no more than that.
"""

_EXAMPLE = """The output format is exactly this: note that the shape below is a syntax example and is NOT the figure you must draw:

<svg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg"><path d="{fixture}" fill="none" stroke="currentColor" stroke-width="16"/></svg>
"""


def build_prompt(k: Optional[int], canvas: int = 256, version: str = PROMPT_VERSION) -> str:
    """The full prompt for one item. A function of the budget alone."""
    if version != PROMPT_VERSION:  # pragma: no cover - defensive
        raise ValueError(f"unknown prompt version {version!r}")
    budget = _BUDGET_FREE if k is None else _BUDGET_LIMITED.format(k=k)
    return (
        _INSTRUCTIONS
        + "\n"
        + budget
        + "\n"
        + _EXAMPLE.format(fixture=FIXTURE_PATH)
        + f"\nReproduce the attached figure now, using the {canvas}x{canvas} canvas."
    )


def prompt_hash() -> str:
    """Stable digest of the prompt text, for the run manifest."""
    import hashlib

    return hashlib.sha256(build_prompt(48).encode("utf-8")).hexdigest()[:16]
