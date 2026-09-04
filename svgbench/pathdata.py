"""SVG path-data parsing, deterministic curve flattening, and affine transforms.

Determinism rules enforced here (these are what make scores reproducible):

* curves are flattened at a *fixed* sample count -- never an adaptive tolerance,
  because tolerance-based flattening silently changes with library versions;
* arcs are split into a fixed number of cubics, same reasoning;
* the parser is strict: anything it cannot understand raises, so a malformed
  model answer is a *recorded failure*, never a silently different drawing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

Point = Tuple[float, float]

#: Samples per cubic segment. 24 keeps the chord error of a 256-unit icon well
#: under a quarter pixel while staying cheap for thousands of tasks.
CURVE_SAMPLES = 24

#: An arc is approximated by at most this many cubics (each <= 90 degrees).
ARC_MAX_SEGMENTS = 8

_NUM = r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?"
_TOKEN_RE = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|(" + _NUM + r")")
_SEPARATORS = frozenset(" \t\r\n,")
_ARGC = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


class PathParseError(ValueError):
    """Raised when a `d` attribute cannot be parsed."""


@dataclass
class SubPath:
    """A single subpath, already flattened to a polyline."""

    points: List[Point]
    closed: bool = True

    def transformed(self, m: "Matrix") -> "SubPath":
        return SubPath([m.apply(p) for p in self.points], self.closed)

    def __len__(self) -> int:
        return len(self.points)


@dataclass
class Matrix:
    """2D affine transform, SVG convention: x' = a*x + c*y + e."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 1.0
    e: float = 0.0
    f: float = 0.0

    def apply(self, p: Point) -> Point:
        x, y = p
        return (self.a * x + self.c * y + self.e, self.b * x + self.d * y + self.f)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """Return the transform equivalent to applying ``other`` then ``self``."""
        return Matrix(
            a=self.a * other.a + self.c * other.b,
            b=self.b * other.a + self.d * other.b,
            c=self.a * other.c + self.c * other.d,
            d=self.b * other.c + self.d * other.d,
            e=self.a * other.e + self.c * other.f + self.e,
            f=self.b * other.e + self.d * other.f + self.f,
        )

    @staticmethod
    def translate(tx: float, ty: float) -> "Matrix":
        return Matrix(e=tx, f=ty)

    @staticmethod
    def scale(sx: float, sy: float | None = None, cx: float = 0.0, cy: float = 0.0) -> "Matrix":
        if sy is None:
            sy = sx
        return (
            Matrix.translate(cx, cy)
            @ Matrix(a=sx, d=sy)
            @ Matrix.translate(-cx, -cy)
        )

    @staticmethod
    def rotate(degrees: float, cx: float = 0.0, cy: float = 0.0) -> "Matrix":
        r = math.radians(degrees)
        cos, sin = math.cos(r), math.sin(r)
        return (
            Matrix.translate(cx, cy)
            @ Matrix(a=cos, b=sin, c=-sin, d=cos)
            @ Matrix.translate(-cx, -cy)
        )


@dataclass
class ParsedPath:
    """The result of parsing a `d` attribute."""

    d: str
    subpaths: List[SubPath]
    commands: List[Tuple[str, List[float]]]
    fill_rule: str = "nonzero"

    # ---- cheap descriptive statistics, all used by the benchmark report ----

    @property
    def n_commands(self) -> int:
        """Number of drawing commands, used as the *budget unit*.

        This is the honest measure of how much vector information a model
        emitted: a tracer needs hundreds, a model that understood the shape
        needs a handful.
        """
        return len(self.commands)

    @property
    def n_subpaths(self) -> int:
        return len(self.subpaths)

    @property
    def n_points(self) -> int:
        return sum(len(sp) for sp in self.subpaths)

    def all_points(self) -> List[Point]:
        out: List[Point] = []
        for sp in self.subpaths:
            out.extend(sp.points)
        return out

    def bbox(self) -> Tuple[float, float, float, float]:
        pts = self.all_points()
        if not pts:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def transformed(self, m: Matrix) -> "ParsedPath":
        return ParsedPath(
            d=self.d,
            subpaths=[sp.transformed(m) for sp in self.subpaths],
            commands=self.commands,
            fill_rule=self.fill_rule,
        )

    def to_d(self, precision: int = 2) -> str:
        """Re-emit a normalised `d` string (used by baselines and tracing)."""
        fmt = "{:." + str(precision) + "g}"
        parts: List[str] = []
        for sp in self.subpaths:
            if len(sp.points) < 2:
                continue
            pts = list(sp.points)
            if pts[0] == pts[-1]:
                pts = pts[:-1]
            parts.append("M" + " ".join(fmt.format(v) for v in pts[0]))
            for p in pts[1:]:
                parts.append("L" + " ".join(fmt.format(v) for v in p))
            parts.append("Z")
        return "".join(parts)

    def to_svg(self, size: int = 256, precision: int = 2, fill: str = "#000") -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
            f'width="{size}" height="{size}">'
            f'<path fill="{fill}" fill-rule="{self.fill_rule}" d="{self.to_d(precision)}"/>'
            "</svg>"
        )


def _tokenize(d: str) -> List[Tuple[str | None, float | None]]:
    tokens: List[Tuple[str | None, float | None]] = []
    pos = 0
    for m in _TOKEN_RE.finditer(d):
        gap = d[pos : m.start()].strip()
        if gap.strip(" \t\r\n,"):
            raise PathParseError(f"unexpected characters {gap!r} in path data")
        pos = m.end()
        letter, num = m.group(1), m.group(2)
        tokens.append((letter, None if num is None else float(num)))
    if d[pos:].strip():
        raise PathParseError(f"unexpected trailing characters {d[pos:]!r} in path data")
    return tokens


def _take_numbers(tokens, i: int, n: int, cmd: str) -> List[float]:
    args: List[float] = []
    for _ in range(n):
        if i >= len(tokens) or tokens[i][1] is None:
            raise PathParseError(f"command '{cmd}' expects {n} numbers, got {len(args)}")
        args.append(float(tokens[i][1]))
        i += 1
    return args


def parse_commands(d: str) -> List[Tuple[str, List[float]]]:
    """Parse a `d` string into ``(command_letter, arguments)`` pairs."""
    if d is None:
        raise PathParseError("path data is None")
    tokens = _tokenize(d)
    if not tokens:
        raise PathParseError("path data is empty")
    cmds: List[Tuple[str, List[float]]] = []
    i = 0
    while i < len(tokens):
        letter = tokens[i][0]
        if letter is None:
            raise PathParseError("path data starts with a number, not a command")
        i += 1
        # Argument counts are keyed by the upper-case command; relative commands
        # (l, a, c, ...) take the same number of arguments as their absolute
        # counterparts.
        n = _ARGC[letter.upper()]
        if n == 0:
            cmds.append((letter, []))
            continue
        args = _take_numbers(tokens, i, n, letter)
        i += n
        cmds.append((letter, args))
        # Implicit repetition: further bare number groups repeat this command.
        # Per the SVG grammar an initial moveto degrades to a lineto.
        repeat = {"M": "L", "m": "l"}.get(letter, letter)
        while i < len(tokens) and tokens[i][0] is None:
            args = _take_numbers(tokens, i, n, repeat)
            i += n
            cmds.append((repeat, args))
    return cmds


def _cubic_points(p0: Point, p1: Point, p2: Point, p3: Point, n: int = CURVE_SAMPLES) -> List[Point]:
    pts: List[Point] = []
    for k in range(1, n + 1):
        t = k / n
        mt = 1.0 - t
        a = mt * mt * mt
        b = 3.0 * mt * mt * t
        c = 3.0 * mt * t * t
        dd = t * t * t
        pts.append(
            (
                a * p0[0] + b * p1[0] + c * p2[0] + dd * p3[0],
                a * p0[1] + b * p1[1] + c * p2[1] + dd * p3[1],
            )
        )
    return pts


def _arc_to_cubics(
    p0: Point, rx: float, ry: float, phi_deg: float, large_arc: bool, sweep: bool, p1: Point
) -> List[Tuple[Point, Point, Point]]:
    """Endpoint -> centre parameterisation (SVG 1.1 F.6.5), then cubic approximants.

    Returns a list of ``(control1, control2, endpoint)`` triples with the start
    point implied by the caller.
    """
    if p0 == p1:
        return []
    rx, ry = abs(rx), abs(ry)
    if rx == 0.0 or ry == 0.0:
        return []  # degenerate arc is a straight line; caller handles it
    phi = math.radians(phi_deg)
    cos_p, sin_p = math.cos(phi), math.sin(phi)

    dx2 = (p0[0] - p1[0]) / 2.0
    dy2 = (p0[1] - p1[1]) / 2.0
    x1p = cos_p * dx2 + sin_p * dy2
    y1p = -sin_p * dx2 + cos_p * dy2

    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1.0:
        s = math.sqrt(lam)
        rx *= s
        ry *= s

    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(0.0, num / den))
    if large_arc == sweep:
        coef = -coef
    cxp = coef * rx * y1p / ry
    cyp = -coef * ry * x1p / rx
    cx = cos_p * cxp - sin_p * cyp + (p0[0] + p1[0]) / 2.0
    cy = sin_p * cxp + cos_p * cyp + (p0[1] + p1[1]) / 2.0

    def angle(ux: float, uy: float, vx: float, vy: float) -> float:
        dot = ux * vx + uy * vy
        norm = math.hypot(ux, uy) * math.hypot(vx, vy)
        if norm == 0.0:
            return 0.0
        a = math.acos(max(-1.0, min(1.0, dot / norm)))
        return -a if (ux * vy - uy * vx) < 0 else a

    theta1 = angle(1.0, 0.0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    elif sweep and dtheta < 0:
        dtheta += 2 * math.pi

    n_segs = max(1, min(ARC_MAX_SEGMENTS, int(math.ceil(abs(dtheta) / (math.pi / 2.0) - 1e-9))))
    delta = dtheta / n_segs
    t = 4.0 / 3.0 * math.tan(delta / 4.0)

    out: List[Tuple[Point, Point, Point]] = []
    theta = theta1
    for _ in range(n_segs):
        cos1, sin1 = math.cos(theta), math.sin(theta)
        theta2 = theta + delta
        cos2, sin2 = math.cos(theta2), math.sin(theta2)

        def pt(cth: float, sth: float) -> Point:
            return (
                cx + rx * cth * cos_p - ry * sth * sin_p,
                cy + rx * cth * sin_p + ry * sth * cos_p,
            )

        def deriv(cth: float, sth: float) -> Point:
            return (-rx * sth * cos_p - ry * cth * sin_p, -rx * sth * sin_p + ry * cth * cos_p)

        e1x, e1y = deriv(cos1, sin1)
        e2x, e2y = deriv(cos2, sin2)
        end = pt(cos2, sin2)
        c1 = (pt(cos1, sin1)[0] + t * e1x, pt(cos1, sin1)[1] + t * e1y)
        c2 = (end[0] - t * e2x, end[1] - t * e2y)
        out.append((c1, c2, end))
        theta = theta2
    return out


def parse_path(d: str, fill_rule: str = "nonzero") -> ParsedPath:
    """Parse `d` into flattened subpaths. Raises :class:`PathParseError`."""
    cmds = parse_commands(d)
    subpaths: List[SubPath] = []
    current: List[Point] = []
    cx = cy = 0.0
    sx = sy = 0.0
    prev_cubic_ctrl: Point | None = None
    prev_quad_ctrl: Point | None = None

    def flush(closed: bool) -> None:
        nonlocal current
        if len(current) >= 2:
            subpaths.append(SubPath(current, closed))
        current = []

    for letter, args in cmds:
        rel = letter.islower()
        u = letter.upper()

        if u == "M":
            flush(False)
            x, y = args
            cx, cy = (cx + x, cy + y) if rel else (x, y)
            sx, sy = cx, cy
            current = [(cx, cy)]
            prev_cubic_ctrl = prev_quad_ctrl = None

        elif u == "L":
            x, y = args
            cx, cy = (cx + x, cy + y) if rel else (x, y)
            current.append((cx, cy))
            prev_cubic_ctrl = prev_quad_ctrl = None

        elif u == "H":
            x = args[0]
            cx = cx + x if rel else x
            current.append((cx, cy))
            prev_cubic_ctrl = prev_quad_ctrl = None

        elif u == "V":
            y = args[0]
            cy = cy + y if rel else y
            current.append((cx, cy))
            prev_cubic_ctrl = prev_quad_ctrl = None

        elif u == "C":
            x1, y1, x2, y2, x, y = args
            if rel:
                p1 = (cx + x1, cy + y1)
                p2 = (cx + x2, cy + y2)
                end = (cx + x, cy + y)
            else:
                p1, p2, end = (x1, y1), (x2, y2), (x, y)
            current.extend(_cubic_points((cx, cy), p1, p2, end))
            prev_cubic_ctrl, prev_quad_ctrl = p2, None
            cx, cy = end

        elif u == "S":
            x2, y2, x, y = args
            if prev_cubic_ctrl is not None:
                p1 = (2 * cx - prev_cubic_ctrl[0], 2 * cy - prev_cubic_ctrl[1])
            else:
                p1 = (cx, cy)
            if rel:
                p2 = (cx + x2, cy + y2)
                end = (cx + x, cy + y)
            else:
                p2, end = (x2, y2), (x, y)
            current.extend(_cubic_points((cx, cy), p1, p2, end))
            prev_cubic_ctrl, prev_quad_ctrl = p2, None
            cx, cy = end

        elif u == "Q":
            x1, y1, x, y = args
            if rel:
                q = (cx + x1, cy + y1)
                end = (cx + x, cy + y)
            else:
                q, end = (x1, y1), (x, y)
            p1 = (cx + 2.0 / 3.0 * (q[0] - cx), cy + 2.0 / 3.0 * (q[1] - cy))
            p2 = (end[0] + 2.0 / 3.0 * (q[0] - end[0]), end[1] + 2.0 / 3.0 * (q[1] - end[1]))
            current.extend(_cubic_points((cx, cy), p1, p2, end))
            prev_cubic_ctrl, prev_quad_ctrl = None, q
            cx, cy = end

        elif u == "T":
            x, y = args
            if prev_quad_ctrl is not None:
                q = (2 * cx - prev_quad_ctrl[0], 2 * cy - prev_quad_ctrl[1])
            else:
                q = (cx, cy)
            end = (cx + x, cy + y) if rel else (x, y)
            p1 = (cx + 2.0 / 3.0 * (q[0] - cx), cy + 2.0 / 3.0 * (q[1] - cy))
            p2 = (end[0] + 2.0 / 3.0 * (q[0] - end[0]), end[1] + 2.0 / 3.0 * (q[1] - end[1]))
            current.extend(_cubic_points((cx, cy), p1, p2, end))
            prev_cubic_ctrl, prev_quad_ctrl = None, q
            cx, cy = end

        elif u == "A":
            rx, ry, rot, large, sweep, x, y = args
            end = (cx + x, cy + y) if rel else (x, y)
            segs = _arc_to_cubics((cx, cy), rx, ry, rot, bool(large), bool(sweep), end)
            if not segs:
                current.append(end)
            else:
                for c1, c2, e in segs:
                    current.extend(_cubic_points((cx, cy), c1, c2, e))
                    cx, cy = e
                cx, cy = end
            prev_cubic_ctrl = prev_quad_ctrl = None
            cx, cy = end

        elif u == "Z":
            current.append((sx, sy))
            flush(True)
            cx, cy = sx, sy
            prev_cubic_ctrl = prev_quad_ctrl = None

    flush(False)
    return ParsedPath(d=d, subpaths=subpaths, commands=cmds, fill_rule=fill_rule)


def simplify_polyline(points: Sequence[Point], tolerance: float) -> List[Point]:
    """Ramer-Douglas-Peucker, iterative so long paths cannot blow the stack."""
    pts = list(points)
    if len(pts) < 3 or tolerance <= 0:
        return pts
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        x0, y0 = pts[i0]
        x1, y1 = pts[i1]
        dx, dy = x1 - x0, y1 - y0
        norm = math.hypot(dx, dy)
        best, best_i = -1.0, -1
        for i in range(i0 + 1, i1):
            px, py = pts[i]
            if norm == 0.0:
                dist = math.hypot(px - x0, py - y0)
            else:
                dist = abs(dy * px - dx * py + x1 * y0 - y1 * x0) / norm
            if dist > best:
                best, best_i = dist, i
        if best > tolerance:
            keep[best_i] = True
            stack.append((i0, best_i))
            stack.append((best_i, i1))
    return [p for p, k in zip(pts, keep) if k]
