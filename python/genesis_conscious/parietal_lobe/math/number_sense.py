"""Number sense — the approximate number system of the IPS.

The intraparietal sulcus (IPS) encodes *quantity* before any symbolic
arithmetic: infants and animals discriminate numerosities without
counting. Two mechanisms live here:

- **Subitizing** — instant exact recognition of small quantities
  (up to ~4). No counting; the quantity is perceived, not computed.
- **The approximate number system (ANS)** — magnitude estimation on
  a log-compressed mental number line (Dehaene, 2003; Feigenson,
  Dehaene & Spelke, 2004). Large quantities blur together:
  discriminability follows a Weber ratio, so telling 8 from 9 is
  easy but 80 from 90 is hard at a glance.

This module is the faculty underneath ``arithmetic.py``: arithmetic
computes exact answers; the number sense judges whether a quantity
*feels* bigger, estimates magnitudes it can't count, and knows when
an answer is implausible (sanity checking — a wrong answer that
violates magnitude intuition gets flagged, the way a person notices
"that can't be right" before checking the math).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Largest quantity recognized instantly and exactly (subitizing range).
SUBITIZING_LIMIT = 4

# Weber fraction — the ratio that governs approximate discriminability.
# Adults: ~0.15–0.17. Two magnitudes are reliably told apart when
# |a - b| / max(a, b) exceeds this fraction.
WEBER_FRACTION = 0.15


@dataclass(frozen=True)
class MagnitudeJudgment:
    """An approximate comparison between two quantities."""

    a: float
    b: float
    relation: str           # "less" | "equal" | "greater"
    confidence: float       # 0..1 — low near the Weber threshold
    exact: bool             # True when both within subitizing range


class NumberSense:
    """Approximate quantity faculty — subitizing and the ANS."""

    # ── Subitizing ────────────────────────────────────────────────

    def subitizes(self, n: float) -> bool:
        """Whether *n* is recognized instantly, without counting."""
        return 0 < n <= SUBITIZING_LIMIT and float(n).is_integer()

    # ── Magnitude representation ──────────────────────────────────

    def represent(self, n: float) -> float:
        """Position on the mental number line (log-compressed).

        Perceived magnitude grows logarithmically — the ANS encodes
        ln(n), which is why 1→2 feels like a bigger step than 8→9.
        """
        if n <= 0:
            return 0.0
        return math.log(n)

    def estimate(self, n: float) -> tuple[float, float]:
        """Approximate magnitude and its noise (mean, spread).

        Estimation error grows with magnitude: the Gaussian noise
        width is proportional to the true value (scalar variability,
        Gibbon 1977). Exact within the subitizing range.
        """
        if self.subitizes(n):
            return n, 0.0
        noise = max(n * WEBER_FRACTION, 0.5)
        return n, noise

    # ── Comparison ────────────────────────────────────────────────

    def compare(self, a: float, b: float) -> MagnitudeJudgment:
        """Judge which of two quantities is larger, approximately.

        Exact when both magnitudes are within the subitizing range.
        Otherwise confidence falls off as the ratio approaches 1
        (the Weber law: discrimination depends on ratio, not
        absolute difference).
        """
        if self.subitizes(a) and self.subitizes(b):
            relation = "less" if a < b else "greater" if a > b else "equal"
            return MagnitudeJudgment(a, b, relation, 1.0, exact=True)

        if a == b:
            return MagnitudeJudgment(a, b, "equal", 1.0, exact=False)

        bigger, smaller = (a, b) if a > b else (b, a)
        ratio_gap = (bigger - smaller) / bigger if bigger > 0 else 0.0
        # Confidence rises with the Weber-ratio gap; saturates once
        # the ratio is clearly discriminable. Clamped to [0, 1] —
        # non-positive magnitudes are outside the ANS domain and must
        # not produce a negative confidence.
        confidence = min(1.0, max(0.0, ratio_gap / WEBER_FRACTION))
        relation = "greater" if a > b else "less"
        return MagnitudeJudgment(a, b, relation, confidence, exact=False)

    # ── Plausibility ──────────────────────────────────────────────

    def plausible(self, claimed: float, expected: float) -> bool:
        """Whether *claimed* is a believable value for *expected*.

        The sanity check arithmetic can't do for itself: an answer
        off by an order of magnitude feels wrong even without
        recomputing. Returns False when the claimed magnitude lies
        outside the discriminable neighborhood of the expected one.
        """
        if expected == 0:
            return claimed == 0
        return abs(self.represent(claimed) - self.represent(expected)) < math.log(
            1.0 + WEBER_FRACTION
        )
