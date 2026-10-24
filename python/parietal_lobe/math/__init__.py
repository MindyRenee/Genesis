"""Parietal math — number sense and exact calculation (IPS).

Three layers, matching the anatomy of the intraparietal sulcus:

    number_sense.py   — the approximate number system: subitizing,
                        Weber-Fechner magnitude comparison, estimation,
                        plausibility checking. The intuition.
    arithmetic.py     — exact symbolic calculation: a safe
                        recursive-descent evaluator over digits,
                        symbols, and number/operator words. The
                        procedure.
    statistics.py     — descriptive statistics over collections:
                        the named means, dispersion, and moments.
                        The population picture.

``ParietalMath`` is the facade the rest of the mind talks to.
"""

from __future__ import annotations

import re

from .arithmetic import CalcResult, evaluate
from .number_sense import MagnitudeJudgment, NumberSense
from .statistics import MEAN_KINDS, StatSummary, mean, summarize

__all__ = [
    "MEAN_KINDS",
    "CalcResult",
    "MagnitudeJudgment",
    "NumberSense",
    "ParietalMath",
    "StatSummary",
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
