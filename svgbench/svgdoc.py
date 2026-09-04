"""Reading SVG documents: layers, transforms, and stroke expansion.

Two fairness decisions live here and both are recorded in the output.

**1. Strokes count.** A model that draws an outline as
``<path stroke="#000" stroke-width="4" fill="none">`` has drawn the shape. If the
scorer only understood fills, that answer would score zero for a formatting
reason rather than a perceptual one. So strokes are expanded into fillable
geometry (a polyline swept by a disc), and the reported command count is taken
from the model's own ``d`` -- the expansion never inflates a budget.

**2. Spec vs. intent.** SVG says an element with a stroke and no ``fill``
attribute is *both* filled black and stroked. Taken literally, that turns many
well-meant outline answers into blobs. We therefore score both readings and
report both: ``spec`` (literal SVG) and ``tolerant`` (stroke-without-fill is
treated as an outline). Neither silently replaces the other.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pathdata import Matrix, ParsedPath, PathParseError, SubPath, parse_path
from .raster import render_coverage

VIEWBOX_RE = re.compile(r"viewBox\s*=\s*[\"']([^\"']+)[\"']", re.I)
_ATTR_RE = re.compile(r"([\w:-]+)\s*=\s*[\"']([^\"']*)[\"']", re.S)
_TAG_RE = re.compile(r"<(/?)(\w[\w:-]*)((?:\s*[\w:-]+\s*=\s*[\"'][^\"']*[\"'])*)\s*(/?)>", re.S)
_TRANSFORM_RE = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)", re.I)

#: Segments used to approximate the round cap/join discs of a stroke.
_STROKE_DISC_SEGMENTS = 16


def _numbers(text: str) -> List[float]:
    return [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?", text or "")]


def parse_transform(text: Optional[str]) -> Matrix:
    """Parse an SVG ``transform`` attribute into a single matrix."""
    m = Matrix()
    if not text:
        return m
    for name, args in _TRANSFORM_RE.findall(text):
        v = _numbers(args)
        name = name.lower()
        if name == "matrix" and len(v) >= 6:
            local = Matrix(v[0], v[1], v[2], v[3], v[4], v[5])
        elif name == "translate" and v:
            local = Matrix.translate(v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale" and v:
            local = Matrix.scale(v[0], v[1] if len(v) > 1 else v[0])
        elif name == "rotate" and v:
            cx, cy = (v[1], v[2]) if len(v) >= 3 else (0.0, 0.0)
            local = Matrix.rotate(v[0], cx, cy)
        elif name == "skewx" and v:
            local = Matrix(a=1.0, c=math.tan(math.radians(v[0])), d=1.0)
        elif name == "skewy" and v:
            local = Matrix(a=1.0, b=math.tan(math.radians(v[0])), d=1.0)
        else:
            continue
        m = m @ local
    return m


def stroke_polygons(subpaths: Sequence[SubPath], width: float) -> List[SubPath]:
    """Sweep a polyline by a disc of diameter ``width`` (round joins and caps).

    Every generated polygon is wound counter-clockwise, so the union of all of
    them is exactly the ``nonzero`` fill of the whole set -- no boolean geometry
    library required.
    """
    if width <= 0:
        return []
    r = width / 2.0
    out: List[SubPath] = []

    def disc(cx: float, cy: float) -> SubPath:
        pts = [
            (
                cx + r * math.cos(2 * math.pi * k / _STROKE_DISC_SEGMENTS),
                cy + r * math.sin(2 * math.pi * k / _STROKE_DISC_SEGMENTS),
            )
            for k in range(_STROKE_DISC_SEGMENTS)
        ]
        return SubPath(pts, closed=True)

    for sp in subpaths:
        pts = list(sp.points)
        if not pts:
            continue
        if len(pts) == 1:
            out.append(disc(pts[0][0], pts[0][1]))
            continue
        if sp.closed and pts[0] != pts[-1]:
            pts.append(pts[0])
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length <= 1e-12:
                continue
            nx, ny = -dy / length * r, dx / length * r
            # Orientation matters, and it is not a style choice. The union of
            # these polygons is taken with the ``nonzero`` rule, which counts
            # winding numbers, so a quad wound *against* the discs it overlaps
            # cancels them to zero and punches a hole at every vertex. Measured
            # before this fix: a one-line stroke rendered as a component with a
            # hole in it, and a 12-subpath answer rendered as 31 components with
            # 44 holes. The order below is the reverse of the naive one so that
            # quads and discs share an orientation, which is what the docstring
            # above promises.
            out.append(SubPath([(x0 + nx, y0 + ny), (x0 - nx, y0 - ny), (x1 - nx, y1 - ny), (x1 + nx, y1 + ny)], True))
        for (x, y) in pts:
            out.append(disc(x, y))
    return out


@dataclass
class Layer:
    parsed: ParsedPath
    fill: bool = True
    opacity: float = 1.0
    stroke_width: float = 0.0


@dataclass
class SvgDoc:
    layers: List[Layer] = field(default_factory=list)
    viewbox: Optional[Tuple[float, float, float, float]] = None

    @property
    def n_commands(self) -> int:
        return sum(l.parsed.n_commands for l in self.layers)

    @property
    def n_subpaths(self) -> int:
        return sum(l.parsed.n_subpaths for l in self.layers)

    def viewbox_matrix(self, size: int) -> Matrix:
        """Standard SVG viewBox -> viewport mapping onto a ``size`` canvas."""
        if not self.viewbox:
            return Matrix()
        vx, vy, vw, vh = self.viewbox
        if vw <= 0 or vh <= 0:
            return Matrix()
        s = min(size / vw, size / vh)
        return Matrix.translate((size - vw * s) / 2.0, (size - vh * s) / 2.0) @ Matrix.scale(s) @ Matrix.translate(-vx, -vy)


def parse_svg_document(text: str, tolerant: bool = False) -> SvgDoc:
    """Parse an SVG string into layers, honouring groups and transforms."""
    doc = SvgDoc()
    vb = VIEWBOX_RE.search(text or "")
    if vb:
        vals = _numbers(vb.group(1))
        if len(vals) == 4:
            doc.viewbox = (vals[0], vals[1], vals[2], vals[3])

    stack: List[Matrix] = [Matrix()]
    for m in _TAG_RE.finditer(text or ""):
        closing, name, attr_text, self_closing = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
        if closing:
            if name == "g" and len(stack) > 1:
                stack.pop()
            continue
        attrs: Dict[str, str] = {k.lower(): v for k, v in _ATTR_RE.findall(attr_text)}
        local = parse_transform(attrs.get("transform"))
        current = stack[-1] @ local
        if name == "g":
            if not self_closing:
                stack.append(current)
            continue
        if name != "path":
            continue

        d = attrs.get("d")
        if not d:
            continue
        fill_rule = attrs.get("fill-rule", attrs.get("fillrule", "nonzero")).lower()
        if fill_rule not in ("nonzero", "evenodd"):
            fill_rule = "nonzero"
        try:
            parsed = parse_path(d, fill_rule=fill_rule)
        except PathParseError:
            continue

        fill_attr = attrs.get("fill", None)
        stroke_attr = attrs.get("stroke", None)
        has_stroke = stroke_attr is not None and stroke_attr.strip().lower() != "none"
        stroke_width = 1.0
        if has_stroke:
            try:
                stroke_width = float(attrs.get("stroke-width", "1"))
            except ValueError:
                stroke_width = 1.0

        if fill_attr is None:
            # SVG default is a black fill. In tolerant mode a stroked element
            # with no explicit fill means "I meant an outline".
            fill = not (tolerant and has_stroke)
        else:
            fill = fill_attr.strip().lower() != "none"

        try:
            opacity = float(attrs.get("opacity", "1"))
        except ValueError:
            opacity = 1.0
        try:
            opacity *= float(attrs.get("fill-opacity", "1"))
        except ValueError:
            pass

        doc.layers.append(Layer(parsed.transformed(current), fill=fill, opacity=max(0.0, min(1.0, opacity)), stroke_width=stroke_width if has_stroke else 0.0))
    return doc


def render_layers(
    doc: SvgDoc,
    size: int = 256,
    ss: int = 4,
    matrix: Optional[Matrix] = None,
    use_viewbox: bool = True,
) -> np.ndarray:
    """Composite layers into a single coverage map.

    ``matrix`` is applied on top of the viewBox mapping, which is how the
    scorer injects the fitted similarity transform.
    """
    base = doc.viewbox_matrix(size) if use_viewbox else Matrix()
    extra = base if matrix is None else (matrix @ base)
    total = np.zeros((size, size), dtype=np.float64)
    for layer in doc.layers:
        if layer.opacity <= 0.0:
            continue
        ink = np.zeros((size, size), dtype=np.float64)
        if layer.fill:
            ink = render_coverage(layer.parsed.subpaths, size, ss, extra, layer.parsed.fill_rule)
        if layer.stroke_width > 0.0:
            polys = stroke_polygons(layer.parsed.subpaths, layer.stroke_width)
            stroked = render_coverage(polys, size, ss, extra, "nonzero")
            # Union of fill and stroke ink for the same element.
            ink = 1.0 - (1.0 - ink) * (1.0 - stroked)
        alpha = ink * layer.opacity
        total = 1.0 - (1.0 - total) * (1.0 - alpha)
    return total


def load_prediction(text: str, size: int = 256, ss: int = 4, tolerant: bool = False, fill_rule: str = "nonzero"):
    """Interpret a model answer, whether it is a bare ``d`` string or an SVG doc.

    Returns ``(coverage, n_commands, n_subpaths, is_svg)``. Raises
    :class:`PathParseError` when nothing usable could be read.
    """
    text = (text or "").strip()
    if not text:
        raise PathParseError("empty prediction")
    if "<" in text and "path" in text.lower():
        doc = parse_svg_document(text, tolerant=tolerant)
        if not doc.layers:
            raise PathParseError("no usable <path> element in SVG")
        coverage = render_layers(doc, size=size, ss=ss)
        return coverage, doc.n_commands, doc.n_subpaths, True
    parsed = parse_path(text, fill_rule=fill_rule)
    coverage = render_coverage(parsed.subpaths, size, ss, None, parsed.fill_rule)
    return coverage, parsed.n_commands, parsed.n_subpaths, False
