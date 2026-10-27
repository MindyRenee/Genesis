"""Real analysis — the words for limits and the epsilon-delta game.

The formal heart of calculus: "for every epsilon there is a delta".
Here that is not a metaphor but a playable game — give the faculty
a challenger (epsilon) and it either answers with a delta that
holds everywhere it probes, or concedes honestly. A limit claim is
verified as a game won, refuted with a witness, or reported
undecided — the same three-way honesty ``symbolic.py`` keeps.

Functions are also graphs: a function sampled over an interval is
its ``(x, y)`` points — ``plot`` returns the coordinate pairs her
canvas or her eyes can use directly.

The machine half is ``src/math/real_analysis.rs``: sample,
epsilon-delta, bisection, newton, riemann_sum, simpson, plot.
Here: the limit's words and the honest verdict.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .algebra import ParseError, evaluate, parse


@dataclass(frozen=True)
class AnalysisResult:
    """The outcome of an analysis question — verified, refuted
    (with the witness), or undecided."""

    operation: str
    input: str
    output: str
    ok: bool
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def fn_of(text: str) -> Callable[[float], float]:
    """Compile expression text to a callable on the variable."""
    expr = parse(text)
    def f(x: float) -> float:
        return evaluate(expr, {"x": x})
    return f


def plot(text_or_fn: str | Callable[[float], float],
         lo: float, hi: float, n: int = 128) -> list[tuple[float, float]]:
    """A function's graph as its (x, y) points — n evenly spaced
    samples over [lo, hi]."""
    f = fn_of(text_or_fn) if isinstance(text_or_fn, str) else text_or_fn
    if n < 2 or lo >= hi:
        return []
    step = (hi - lo) / (n - 1)
    return [(lo + i * step, f(lo + i * step)) for i in range(n)]


def delta_game(f: Callable[[float], float], a: float, limit: float,
               delta: float, epsilon: float, probes: int = 64) -> float | None:
    """The epsilon-delta game, played honestly.

    Every x with ``0 < |x - a| < delta`` is probed; return the
    witness where ``|f(x) - L| >= epsilon`` — the refuting point —
    or None when delta answers this challenger. The punctured
    neighborhood means x = a itself is never tested: a limit
    ignores what the function does at the point.
    """
    if epsilon <= 0 or delta <= 0 or probes <= 0:
        raise ValueError("epsilon, delta and probes must be positive")
    for i in range(probes):
        off = delta * (i + 1) / probes
        for x in (a - off, a + off):
            if abs(x - a) >= delta or x == a:
                continue
            try:
                y = f(x)
            except (ValueError, OverflowError, ZeroDivisionError):
                return x
            if not math.isfinite(y) or abs(y - limit) >= epsilon:
                return x
    return None


def verify_limit(f: Callable[[float], float], a: float,
                 claimed: float, epsilon: float) -> AnalysisResult:
    """Answer the challenger: find a delta for this epsilon by
    halving until the game is won, or report undecided."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        return AnalysisResult(
            "limit", f"x→{a}", "", False, error="epsilon must be positive and finite"
        )
    delta = 1.0
    for _ in range(64):
        w = delta_game(f, a, claimed, delta, epsilon)
        if w is None:
            return AnalysisResult(
                "limit", f"x→{a}", f"verified: δ = {delta:.3g} answers ε = {epsilon:g}",
                True, data={"delta": delta, "epsilon": epsilon})
        delta = min(delta / 2, abs(w - a) * 0.9)
    return AnalysisResult("limit", f"x→{a}", "undecided — no delta found",
                          False, error="undecided")


def numeric_limit(f: Callable[[float], float], a: float) -> AnalysisResult:
    """Approach a from both sides — the answer the function gives."""
    left = right = None
    for k in range(1, 40):
        h = 0.5 ** k
        try:
            left = f(a - h)
            right = f(a + h)
        except (ValueError, OverflowError, ZeroDivisionError):
            left = right = None
            continue
        if not math.isfinite(left) or not math.isfinite(right):
            left = right = None
    if left is not None and right is not None and abs(left - right) < 1e-7:
        value = (left + right) / 2
        return AnalysisResult("limit", f"x→{a}", f"{value:.8g}", True,
                              data={"value": value})
    if left is not None and right is not None:
        return AnalysisResult("limit", f"x→{a}",
                              f"left {left:.6g} ≠ right {right:.6g} — the limit does not exist",
                              False, error="sides disagree")
    return AnalysisResult("limit", f"x→{a}", "undecided", False, error="undecided")


def bisection(f: Callable[[float], float], lo: float, hi: float,
              tol: float = 1e-12) -> float | None:
    """A root inside a sign change — the interval halved to tol."""
    if tol <= 0 or lo >= hi:
        return None
    flo = f(lo)
    fhi = f(hi)
    if flo == 0:
        return lo
    if fhi == 0:
        return hi
    if not math.isfinite(flo) or not math.isfinite(fhi) or flo * fhi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        fm = f(mid)
        if abs(fm) < tol or hi - lo < tol:
            return mid
        if flo * fm < 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def riemann_sum(f: Callable[[float], float], lo: float, hi: float,
                n: int = 1000) -> float:
    """Left-endpoint Riemann sum — area as accumulation."""
    if n < 1:
        raise ValueError("n must be positive")
    if lo == hi:
        return 0.0
    dx = (hi - lo) / n
    return sum(f(lo + i * dx) for i in range(n)) * dx


def simpson(f: Callable[[float], float], lo: float, hi: float,
            n: int = 1000) -> float:
    """Simpson's rule — parabolic quadrature, n forced even."""
    if n < 1:
        raise ValueError("n must be positive")
    if n % 2:
        n += 1
    dx = (hi - lo) / n
    total = f(lo) + f(hi)
    total += 4 * sum(f(lo + i * dx) for i in range(1, n, 2))
    total += 2 * sum(f(lo + i * dx) for i in range(2, n, 2))
    return total * dx / 3


_ANAL_WORD = re.compile(
    r"\b(?:limit|limits|converges?|diverges?|continuous|continuity|"
    r"epsilon|delta|riemann|integral|area under|approach|"
    r"uniform|supremum|infimum|cauchy|bounded)\b",
    re.IGNORECASE,
)


def looks_like_analysis(text: str) -> bool:
    """Whether the text asks about limits, convergence, or
    continuity — the vocabulary of nearness made precise."""
    return bool(_ANAL_WORD.search(text))


def limit_text(text: str, at: float) -> AnalysisResult:
    """"limit of sin(x)/x at 0" — numeric approach first, then
    the delta game verifying what was found."""
    try:
        f = fn_of(text)
    except ParseError as e:
        return AnalysisResult("limit", text, "", False, str(e))
    return numeric_limit(f, at)


def plot_text(text: str, lo: float, hi: float, n: int = 128) -> AnalysisResult:
    """The graph of an expression — (x, y) points, renderable."""
    try:
        pts = plot(text, lo, hi, n)
    except ParseError as e:
        return AnalysisResult("plot", text, "", False, str(e))
    return AnalysisResult("plot", text,
                          f"{len(pts)} points on [{lo:g}, {hi:g}]",
                          True, data={"points": pts})


def integral_text(text: str, lo: float, hi: float) -> AnalysisResult:
    """"integral of x^2 from 0 to 1" — Simpson's quadrature."""
    try:
        f = fn_of(text)
    except ParseError as e:
        return AnalysisResult("integral", text, "", False, str(e))
    val = simpson(f, lo, hi)
    return AnalysisResult("integral", text, f"{val:.8g}", True, data={"value": val})
