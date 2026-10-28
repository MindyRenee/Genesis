"""Trigonometry — the circle's vocabulary.

Where ``arithmetic.py`` hears numbers and ``algebra.py`` keeps the
unknown, this module hears angles: "sin of pi over six", "the
triangle with sides 3 4 5", "what angle is that". Special values
speak their exact names — sqrt(3)/2, not 0.866 — because the
faculty's job is the words, and the words for the famous angles
are exact.

The machine half lives in ``src/math/trig.rs``: normalize, the
triangle laws, the ratios. Here: the parse, the names, the answer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .algebra import ParseError, evaluate, parse


@dataclass(frozen=True)
class TrigResult:
    """The outcome of a trigonometric question — the operation,
    the input as given, the output rendered, the ok flag, and the
    failure wording when ok is false."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# The famous angles and their exact ratios — the words the circle
# already knows. Keys are radians as fractions of pi.
_EXACT_SIN = {
    0: "0",
    (1, 6): "1/2",
    (1, 4): "√2/2",
    (1, 3): "√3/2",
    (1, 2): "1",
    (2, 3): "√3/2",
    (3, 4): "√2/2",
    (5, 6): "1/2",
}
_EXACT_COS = {
    0: "1",
    (1, 6): "√3/2",
    (1, 4): "√2/2",
    (1, 3): "1/2",
    (1, 2): "0",
    (2, 3): "-1/2",
    (3, 4): "-√2/2",
    (5, 6): "-√3/2",
}


def _pi_fraction(angle: float) -> tuple[int, int] | int | None:
    """Recognize angle = k·pi/d for small denominators."""
    for d in (2, 3, 4, 6, 12):
        k = round(angle * d / math.pi)
        if abs(angle - k * math.pi / d) < 1e-9:
            if k >= 0:
                k %= 2 * d
            g = math.gcd(k, d) or 1
            k, d = k // g, d // g
            return k if d == 1 else (k, d)
    return None


def _eval_angle(text: str) -> float:
    """Evaluate angle text — 'pi/6', '30 degrees', '0.5'."""
    m = re.fullmatch(r"(.+?)\s*(?:degrees?|deg|°)", text.strip(), re.IGNORECASE)
    if m:
        return math.radians(evaluate(parse(m.group(1))))
    return evaluate(parse(text))


_TRIG_WORD = re.compile(
    r"\b(?:sin|cos|tan|sec|csc|cot|asin|acos|atan|angle|triangle|"
    r"degrees?|radians?|hypotenuse|opposite|adjacent)\b",
    re.IGNORECASE,
)


def looks_like_trig(text: str) -> bool:
    """Whether the text asks about angles, ratios, or triangles.

    True for "sin of pi/6", "30 degrees in radians", "triangle
    with sides 3 4 5" — the circle's own vocabulary.
    """
    return bool(_TRIG_WORD.search(text))


def ratio_text(func: str, angle_text: str) -> TrigResult:
    """A trigonometric ratio at an angle — exact name when the
    angle is a famous one, else the decimal."""
    try:
        angle = _eval_angle(angle_text)
    except (ParseError, ValueError, OverflowError) as e:
        return TrigResult(func, angle_text, "", False, str(e))
    name = func.lower()
    if name in {"asin", "acos", "atan"}:
        inverse = {"asin": math.asin, "acos": math.acos, "atan": math.atan}[name]
        try:
            radians = inverse(angle)
        except ValueError as e:
            return TrigResult(func, angle_text, "", False, str(e))
        frac = _pi_fraction(radians)
        text_out = (
            f"pi/{frac[1]}"
            if isinstance(frac, tuple)
            else (f"{frac}·pi" if isinstance(frac, int) and frac != 0 else f"{radians:.6g}")
        )
        if isinstance(frac, tuple) and frac[0] != 1:
            text_out = f"{frac[0]}·pi/{frac[1]}"
        return TrigResult(func, angle_text, text_out, True, data={"radians": radians})
    ratio = {"sin": math.sin, "cos": math.cos, "tan": math.tan}.get(name)
    if ratio is None:
        return TrigResult(func, angle_text, "", False, f"unknown ratio '{func}'")
    val = ratio(angle)
    frac = _pi_fraction(angle)
    key = frac if isinstance(frac, int) else (frac if frac else None)
    if name == "sin" and key in _EXACT_SIN and val >= 0:
        out = _EXACT_SIN[key]
    elif name == "cos" and key in _EXACT_COS:
        out = _EXACT_COS[key]
    else:
        out = f"{val:.6g}"
    return TrigResult(func, angle_text, out, True, data={"value": val})


def convert_text(text: str, to: str) -> TrigResult:
    """Convert angle units — 'pi/6 to degrees', '30 deg to rad'."""
    try:
        angle = _eval_angle(text)
    except (ParseError, ValueError, OverflowError) as e:
        return TrigResult("convert", text, "", False, str(e))
    if to.startswith("deg"):
        return TrigResult("convert", text, f"{math.degrees(angle):.6g}°", True)
    return TrigResult("convert", text, f"{angle:.6g} rad", True)


def triangle_text(sides: list[float]) -> TrigResult:
    """Solve a triangle from three sides — angles in degrees and
    Heron's area, or the honest failure when the sides can't close."""
    if len(sides) != 3:
        return TrigResult("triangle", str(sides), "", False, "three sides, please")
    a, b, c = sorted(sides)
    if a <= 0 or a + b <= c:
        return TrigResult(
            "triangle", str(sides), "", False, "those sides cannot close into a triangle"
        )

    def ang(adj1: float, adj2: float, opp: float) -> float:
        return math.degrees(math.acos((adj1**2 + adj2**2 - opp**2) / (2 * adj1 * adj2)))

    angles = (ang(a, b, c), ang(a, c, b), ang(b, c, a))
    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    out = f"angles {angles[0]:.4g}°, {angles[1]:.4g}°, {angles[2]:.4g}°; area {area:.6g}"
    return TrigResult("triangle", str(sides), out, True, data={"angles": angles, "area": area})
