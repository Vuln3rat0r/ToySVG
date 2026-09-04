"""ToySVG: a deterministic benchmark for measuring how well a vision
model can *see* a shape and *re-draw* it as an SVG path.

The package is deliberately self-contained so that a score is reproducible:
nothing here depends on an external SVG rendering library, on wall-clock time,
or on random state that is not explicitly seeded and recorded.
"""

__version__ = "0.1.0"

from .pathdata import PathParseError, ParsedPath, parse_path  # noqa: F401
from .raster import render_coverage  # noqa: F401

__all__ = ["PathParseError", "ParsedPath", "parse_path", "render_coverage", "__version__"]
