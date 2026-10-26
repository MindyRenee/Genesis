"""Parietal math — number sense through calculus (IPS).

Five layers, matching the anatomy of the intraparietal sulcus:

    number_sense.py   — the approximate number system: subitizing,
                        Weber-Fechner magnitude comparison, estimation,
                        plausibility checking. The intuition.
    arithmetic.py     — exact calculation: a safe recursive-descent
                        evaluator over digits, symbols, and
                        number/operator words. The procedure.
    algebra.py        — formal manipulation: expression trees that
                        simplify, expand, factor, and solve. The
                        unknown kept symbolic.
    calculus.py       — the mathematics of change: differentiation
                        and integration over the algebra trees.
    statistics.py     — descriptive statistics over collections:
                        the named means, dispersion, and moments.
                        The population picture.

``symbolic.py`` sits beside them as the oracle — the sympy-backed
checker ``frontal_lobe/research.py`` verifies claims against. The
faculty here is endogenous: it manipulates structure itself, and
its results are verified, never asserted.

``ParietalMath`` is the facade the rest of the mind talks to.
"""

from __future__ import annotations

import re

from .algebra import (
    AlgebraResult,
    expand_text,
    factor_text,
    parse,
    render,
    simplify_text,
    solve,
)
from .arithmetic import CalcResult, evaluate
from .calculus import (
    derivative_at_text,
    derivative_text,
    integral_between_text,
    integral_text,
    taylor,
)
from .number_sense import MagnitudeJudgment, NumberSense
from .statistics import MEAN_KINDS, StatSummary, mean, summarize

__all__ = [
    "MEAN_KINDS",
    "AlgebraResult",
    "CalcResult",
    "MagnitudeJudgment",
    "NumberSense",
    "ParietalMath",
    "StatSummary",
    "looks_like_algebra",
    "looks_like_calculus",
    "looks_like_math",
    "looks_like_statistics",
    "mean",
    "summarize",
]

# A question is math-shaped when it contains a digit expression with
# an operator, or explicit operator words between quantities.
_DIGIT_EXPR = re.compile(r"\d\s*[-+*/x×÷^%]\s*\d|\d\s*\(")
_WORD_EXPR = re.compile(
    r"\b(?:plus|minus|times|multiplied|divided|squared|cubed|sqrt|"
    r"modulo)\b"
    r"|\bto the power\b|\bpower of\b|\braised to\b|\bsquare root\b",
    re.IGNORECASE,
)


def looks_like_math(text: str) -> bool:
    """Whether the text asks for a computation.

    True for "what is 7 + 5", "nine times eight", "sqrt 144" —
    anything carrying an explicit operation. A bare number alone
    ("what is 5") is not a computation request.
    """
    return bool(_DIGIT_EXPR.search(text) or _WORD_EXPR.search(text))


# A question is statistics-shaped when it names a descriptive
# measure — the named means, the dispersion family, or the
# shape moments.
_STATS_EXPR = re.compile(
    r"\b(?:mean|average|median|mode|midrange|variance|deviation|"
    r"percentile|quartile|kurtosis|skewness|moment|dispersion|"
    r"coefficient of variation|standard error|standard score|"
    r"z-?score)\b",
    re.IGNORECASE,
)


def looks_like_statistics(text: str) -> bool:
    """Whether the text asks for a descriptive statistic.

    True for "what's the mean of these", "the variance of the
    sample", "harmonic mean of 4 and 6" — anything naming a
    descriptive measure.
    """
    return bool(_STATS_EXPR.search(text))


# Algebra-shape: an equation with an unknown, or the manipulation
# verbs ("solve", "simplify", "expand", "factor") over an expression
# that carries a variable. A variable alone — "2x + 3" — is already
# algebra; a number alone is arithmetic's job.
_ALGEBRA_VERB = re.compile(r"\b(?:solve|simplify|expand|factor|factorize)\b", re.IGNORECASE)
_HAS_VAR = re.compile(r"\b[a-zA-Z](?!\w)\b|\b[a-zA-Z]\d|\d[a-zA-Z]\b")


def looks_like_algebra(text: str) -> bool:
    """Whether the text asks for symbolic manipulation.

    True for "solve x^2 - 4 = 0", "expand (x+1)^3", "2x + 3x" —
    an unknown in play or an explicit manipulation verb.
    """
    return bool(
        _ALGEBRA_VERB.search(text)
        or ("=" in text and _HAS_VAR.search(text))
        or (_HAS_VAR.search(text) and re.search(r"[a-zA-Z0-9)]\s*[-+*/^=]", text))
    )


# Calculus-shape: the named operations of change — derivatives,
# integrals, series expansion, limits, slope at a point.
_CALCULUS_VERB = re.compile(
    r"\b(?:derivative|differentiate|d/d[a-z]|integral|integrate|"
    r"antiderivative|taylor|maclaurin|limit of|slope of)\b",
    re.IGNORECASE,
)


def looks_like_calculus(text: str) -> bool:
    """Whether the text asks for the mathematics of change."""
    return bool(_CALCULUS_VERB.search(text))


# Natural-language wrappers ahead of the math — the extraction pass
# strips these so "what is the derivative of x^2" reaches the parser
# as "x^2".
_LEAD_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:what\s+is\s+|what's\s+|whats\s+|compute\s+|calculate\s+|"
    r"evaluate\s+|find\s+)?",
    re.IGNORECASE,
)
_OP_RE = re.compile(
    r"^\s*(?:the\s+)?"
    r"(?:solve|simplify|expand|factor(?:ize)?|differentiate|"
    r"derivative\s+of|derivative|d/d[a-z]\s*|integrate|"
    r"integral\s+of|integral|antiderivative\s+of|antiderivative|"
    r"limit\s+of|slope\s+of)\s*",
    re.IGNORECASE,
)
_TAIL_RE = re.compile(r"\s*(?:for\s+[a-zA-Z]\s*)?[.?!]?\s*$", re.IGNORECASE)
_BETWEEN_RE = re.compile(r"\bfrom\s+(-?\d+\.?\d*)\s+to\s+(-?\d+\.?\d*)", re.IGNORECASE)
_AT_RE = re.compile(r"\bat\s+[a-zA-Z]\s*=\s*(-?\d+\.?\d*)", re.IGNORECASE)


def _extract(text: str) -> str:
    """Strip the natural-language wrapper around the math —
    the question words, the operation verb, the trailing clause."""
    s = _LEAD_RE.sub("", text)
    s = _OP_RE.sub("", s)
    s = _BETWEEN_RE.sub("", s)
    s = _AT_RE.sub("", s)
    s = _TAIL_RE.sub("", s)
    return s.strip()


class ParietalMath:
    """Her calculation faculty — the parietal lobe's number cortex."""

    def __init__(self) -> None:
        """Initialize the facade with an approximate number sense."""
        self.sense = NumberSense()

    def looks_like_math(self, text: str) -> bool:
        """Whether the text asks for a computation."""
        return looks_like_math(text)

    def evaluate(self, text: str) -> CalcResult:
        """Compute the arithmetic in the text, if any."""
        return evaluate(text)

    def compare(self, a: float, b: float) -> MagnitudeJudgment:
        """Approximate magnitude comparison (the ANS, not exact math)."""
        return self.sense.compare(a, b)

    def subitizes(self, n: float) -> bool:
        """Whether *n* is small enough to be perceived instantly."""
        return self.sense.subitizes(n)

    def plausible(self, claimed: float, expected: float) -> bool:
        """Sanity check: is *claimed* a believable value for *expected*?"""
        return self.sense.plausible(claimed, expected)

    def looks_like_statistics(self, text: str) -> bool:
        """Whether the text asks for a descriptive statistic."""
        return looks_like_statistics(text)

    def mean(self, values: list[float], kind: str = "arithmetic", p: float = 1.0) -> float:
        """A named mean over a collection (see ``statistics.MEAN_KINDS``)."""
        return mean(values, kind=kind, p=p)

    def summarize(self, values: list[float]) -> StatSummary:
        """The full descriptive profile of a collection."""
        return summarize(values)

    # ── Algebra — the unknown kept symbolic ──────────────────────

    def looks_like_algebra(self, text: str) -> bool:
        """Whether the text asks for symbolic manipulation."""
        return looks_like_algebra(text)

    def simplify(self, text: str) -> AlgebraResult:
        """Reduce the expression in the text to canonical form."""
        return simplify_text(_extract(text))

    def expand(self, text: str) -> AlgebraResult:
        """Distribute the expression in the text fully."""
        return expand_text(_extract(text))

    def factor(self, text: str) -> AlgebraResult:
        """Factor the polynomial in the text."""
        return factor_text(_extract(text))

    def solve(self, text: str) -> AlgebraResult:
        """Solve the equation in the text — closed forms first."""
        return solve(_extract(text))

    # ── Calculus — the mathematics of change ─────────────────────

    def looks_like_calculus(self, text: str) -> bool:
        """Whether the text asks for the mathematics of change."""
        return looks_like_calculus(text)

    def differentiate(self, text: str, var: str = "x") -> AlgebraResult:
        """The derivative of the expression in the text."""
        return derivative_text(_extract(text), var)

    def slope_at(self, text: str, at: float, var: str = "x") -> AlgebraResult:
        """The derivative evaluated at a point — the slope there."""
        return derivative_at_text(_extract(text), var, at)

    def integrate(self, text: str, var: str = "x") -> AlgebraResult:
        """The antiderivative of the expression in the text."""
        return integral_text(_extract(text), var)

    def integrate_between(self, text: str, lo: float, hi: float, var: str = "x") -> AlgebraResult:
        """The definite integral — closed form, else Simpson."""
        return integral_between_text(_extract(text), var, lo, hi)

    def taylor_series(
        self, text: str, at: float = 0.0, order: int = 5, var: str = "x"
    ) -> AlgebraResult:
        """The Taylor polynomial of the expression around ``at``."""
        try:
            out = taylor(parse(_extract(text)), var, at, order)
            return AlgebraResult("taylor", text, render(out), True)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as e:
            return AlgebraResult("taylor", text, "", False, str(e))
