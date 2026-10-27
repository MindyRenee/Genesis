"""Complex analysis — the words for the plane of numbers.

A complex number arrives as "3+4i", "(1, -1)", "cis pi/3" and
leaves as magnitude, argument, polar form, powers, roots. The
roots of unity answer in words — "the 5th roots of 1" — and the
plane's geometry is just the unit circle walked n ways.

The machine half is ``src/math/complex_analysis.rs``: Complex,
polar/rect, nth roots, De Moivre, winding number. Here: the
notation and the reply.
"""

from __future__ import annotations

import cmath
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .algebra import ParseError, evaluate, parse


@dataclass(frozen=True)
class ComplexResult:
    """The outcome of a complex question."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


_COMPLEX_LIT = re.compile(
    r"^\s*\(?\s*(-?[\d.]+)\s*([+-])\s*([\d.]+)\s*i\s*\)?\s*$",
    re.IGNORECASE,
)
_PAIR_LIT = re.compile(r"^\s*\(?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\)?\s*$")
_CIS_LIT = re.compile(r"^\s*cis\s+(.+)$", re.IGNORECASE)
_I_LIT = re.compile(r"^\s*(-?)i\s*$", re.IGNORECASE)
_REAL_LIT = re.compile(r"^\s*(-?[\d.]+)\s*$")


def parse_complex(text: str) -> complex | None:
    """Read "3+4i", "(1,-1)", "cis pi/3", "-i", "2" as a complex."""
    t = text.strip()
    if m := _COMPLEX_LIT.match(t):
        re_, sgn, im = m.groups()
        return complex(float(re_), (1 if sgn == "+" else -1) * float(im))
    if m := _PAIR_LIT.match(t):
        return complex(float(m.group(1)), float(m.group(2)))
    if m := _CIS_LIT.match(t):
        try:
            angle = evaluate(parse(m.group(1)))
        except ParseError:
            return None
        return cmath.exp(1j * angle)
    if m := _I_LIT.match(t):
        return complex(0.0, -1.0 if m.group(1) else 1.0)
    if m := _REAL_LIT.match(t):
        return complex(float(m.group(1)))
    return None


def _fmt(z: complex, digits: int = 6) -> str:
    r, i = z.real, z.imag
    if abs(i) < 1e-12:
        return f"{r:.{digits}g}"
    if abs(r) < 1e-12:
        return f"{i:.{digits}g}i"
    sign = "+" if i >= 0 else "-"
    return f"{r:.{digits}g} {sign} {abs(i):.{digits}g}i"


def polar(z: complex) -> tuple[float, float]:
    """(magnitude, argument) — the number's own coordinates."""
    return abs(z), cmath.phase(z)


def nth_roots(n: int) -> list[complex]:
    """The n-th roots of unity — the circle walked n equal ways."""
    return [cmath.exp(2j * math.pi * k / n) for k in range(n)]


def roots_of(z: complex, n: int) -> list[complex]:
    """All n n-th roots of z — the principal root rotated around."""
    r, th = polar(z)
    rn = r ** (1.0 / n)
    return [cmath.rect(rn, (th + 2 * math.pi * k) / n) for k in range(n)]


def winding_number(path: list[complex], center: complex = 0j) -> int:
    """How many times a closed path wraps a point — the discrete
    argument principle."""
    turns = 0.0
    for i in range(len(path)):
        a = path[i] - center
        b = path[(i + 1) % len(path)] - center
        turns += cmath.phase(b / a) if a != 0 and b != 0 else 0.0
    return round(turns / (2 * math.pi))


_CX_WORD = re.compile(
    r"\b(?:complex|imaginary|modulus|argument|conjugate|polar|"
    r"rectangular|euler's|de moivre|roots of unity|winding)\b|"
    r"\d\s*[+-]\s*[\d.]*i\b|\bcis\b",
    re.IGNORECASE,
)


def looks_like_complex(text: str) -> bool:
    """Whether the text asks about complex numbers — '3+4i',
    'roots of unity', 'polar form'."""
    return bool(_CX_WORD.search(text))


def describe_text(text: str) -> ComplexResult:
    """Portrait of a complex number: rectangular, polar, magnitude,
    argument."""
    z = parse_complex(text)
    if z is None:
        return ComplexResult("describe", text, "", False,
                             "could not read a complex number")
    r, th = polar(z)
    deg = math.degrees(th)
    out = (f"{_fmt(z)}: |z| = {r:.6g}, arg z = {th:.6g} rad "
           f"({deg:.4g}°), polar {r:.6g}·cis({th:.6g})")
    return ComplexResult("describe", text, out, True,
                         data={"value": z, "r": r, "theta": th})


def power_text(text: str, n: int) -> ComplexResult:
    """"(1+i)^8" — De Moivre: scale modulus, multiply angle."""
    z = parse_complex(text)
    if z is None:
        return ComplexResult("power", text, "", False,
                             "could not read a complex number")
    out = z ** n
    return ComplexResult("power", f"{text}^{n}", _fmt(out), True,
                         data={"value": out})


def roots_unity_text(n: int) -> ComplexResult:
    """"the 5th roots of unity" — all of them, in rectangular."""
    if n < 1:
        return ComplexResult("roots", str(n), "", False, "n must be positive")
    roots = nth_roots(n)
    out = ", ".join(_fmt(z) for z in roots)
    return ComplexResult("roots", f"{n}th roots of unity", out, True,
                         data={"roots": roots})


def roots_text(text: str, n: int) -> ComplexResult:
    """"the cube roots of -8" — all n roots, none forgotten."""
    z = parse_complex(text)
    if z is None:
        return ComplexResult("roots", text, "", False,
                             "could not read a complex number")
    if n < 1:
        return ComplexResult("roots", text, "", False, "n must be positive")
    roots = roots_of(z, n)
    out = ", ".join(_fmt(w) for w in roots)
    return ComplexResult("roots", f"{n}th roots of {text}", out, True,
                         data={"roots": roots})
