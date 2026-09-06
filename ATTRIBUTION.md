# Attribution and third-party notices

## Ground-truth corpus: Phosphor Icons

The benchmark corpus is designed to be derived from the published **Phosphor
Icons** asset release.

| | |
|---|---|
| Project | Phosphor Icons (`@phosphor-icons/core`) |
| Version | 2.1.1 (pinned) |
| Licence | MIT |
| Source | https://github.com/phosphor-icons/core |
| Contents | 1,512 icons in 6 weights; ~6.5 MB unpacked |

MIT permits use, modification, and redistribution, including commercially,
provided the copyright notice and permission notice are retained. Accordingly:

- the upstream `LICENSE` file is redistributed verbatim alongside the corpus;
- every published result names the project and version;
- the corpus manifest records the pinned version, so a build can be reproduced.

The icons serve as hidden ground-truth references: prompts omit asset identifiers entirely.

## Licences of alternative sources (considered, not used)

Recorded because a benchmark corpus inherits the strictest upstream term, and
because a second icon family is the planned generalisation split.

| Project | Licence | Note |
|---|---|---|
| Tabler Icons 3.47 | MIT | stroke-based by default, so hidden ground truth would be a stroke |
| Lucide 1.47 | ISC | stroke-based, 24-unit viewBox, non-integer coordinates |
| Material Symbols 0.47 | Apache-2.0 | variable-font derived, redundant across sizes |
| Simple Icons 16.31 | CC0-1.0 | brand logos, trademark considerations in some jurisdictions |
| Font Awesome 7.3 | CC-BY-4.0 AND OFL-1.1 AND MIT | attribution obligations spread across three licences |
| Iconify JSON 2.2 | MIT | aggregator, ~484 MB unpacked; candidate for a later split |

Licences above were read from each project's published package metadata
(npm registry) in September 2026. Verify before redistributing anything derived.

## Software dependencies

`numpy`, `scipy`, `scikit-image`, and `Pillow` are runtime dependencies for the
harness and are used under their own licences (BSD-3-Clause / BSD-3-Clause /
BSD-3-Clause / MIT-CC0 respectively). The benchmark uses a self-contained
rasteriser without external SVG library dependencies, ensuring deterministic scores
independent of third-party renderer updates.

## Model outputs

Evaluation artefacts (raw model answers, per-item scores, reports) are results,
not corpus data. If they are ever published, they should be accompanied by the
model version strings, harness version, corpus hash, and the prompts used, so the
run can be rerun rather than believed.

## Repository code and benchmark data

The evaluation harness code is provided under the MIT Licence. Ground truth assets are derived from Phosphor Icons under the MIT Licence; see above for the upstream copyright notice.
