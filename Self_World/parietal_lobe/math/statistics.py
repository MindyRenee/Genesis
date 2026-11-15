"""Statistics — central tendency, spread, and shape, parietal lobe.

Where ``number_sense`` estimates a single magnitude and
``arithmetic`` evaluates an expression, this module describes a
*collection*: where its center is, how widely the values scatter
around it, and what shape the distribution takes. Descriptive
statistics is the number sense applied to a population — the same
IPS machinery, asked about a sample rather than a quantity.

Three families live here, matching the classical taxonomy:

- **Central tendency** — the named means (Pythagorean: arithmetic,
  geometric, harmonic; the power-mean ladder: quadratic, cubic,
  contraharmonic; the interpolating family: Heronian, Heinz,
  Lehmer) plus median and mode. Every named mean is a member of the
  generalized/power mean or a proven special case — the hierarchy
  is real, not a list.
- **Dispersion** — variance and standard deviation, absolute
  deviation and the robust MAD, quartile/percentile ranges, the
  coefficient of variation, and the plain range.
- **Shape** — raw and central moments, skewness, kurtosis, and the
  standard error; plus ``sampling_means`` which *demonstrates* the
  central limit theorem rather than asserting it.

Every formula is written out — a small, inspectable faculty whose
answers can be traced, not a stolen ``numpy.mean``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass


def _as_list(xs: Iterable[float]) -> list[float]:
    vals = [float(x) for x in xs]
    if not vals:
        raise ValueError("empty data — no statistics on nothing")
    return vals


def _positive(xs: list[float], name: str) -> None:
    if any(x <= 0 for x in xs):
        raise ValueError(f"{name} is defined on positive data only")


# ── Central tendency ─────────────────────────────────────────────


def arithmetic_mean(xs: Iterable[float]) -> float:
    """The plain average — Σx / n. Minimizes squared error."""
    vals = _as_list(xs)
    return sum(vals) / len(vals)


def geometric_mean(xs: Iterable[float]) -> float:
    """The multiplicative center — (Πx)^(1/n). Positive data only;
    the right mean for ratios and growth factors."""
    vals = _as_list(xs)
    _positive(vals, "geometric_mean")
    return math.exp(sum(math.log(x) for x in vals) / len(vals))


def harmonic_mean(xs: Iterable[float]) -> float:
    """n / Σ(1/x) — the mean for rates, where the denominators
    carry the unit (speeds, densities). Positive data only."""
    vals = _as_list(xs)
    _positive(vals, "harmonic_mean")
    return len(vals) / sum(1.0 / x for x in vals)


def power_mean(xs: Iterable[float], p: float) -> float:
    """The generalized mean — ((Σx^p)/n)^(1/p).

    The ladder the named means sit on: p=1 arithmetic, p=2
    quadratic, p=3 cubic, p=−1 harmonic, p→0 geometric, p→+∞ max,
    p→−∞ min. Positive data only.
    """
    vals = _as_list(xs)
    _positive(vals, "power_mean")
    if p == 0:
        return geometric_mean(vals)
    return (sum(x**p for x in vals) / len(vals)) ** (1.0 / p)


def quadratic_mean(xs: Iterable[float]) -> float:
    """Root mean square — the power mean at p=2. Large values
    dominate; the right average for magnitudes that add in
    quadrature (energies, amplitudes)."""
    return power_mean(xs, 2.0)


def cubic_mean(xs: Iterable[float]) -> float:
    """The power mean at p=3 — heavier weight on the large values
    still; used in elasticity and size-distribution work."""
    return power_mean(xs, 3.0)


def contraharmonic_mean(xs: Iterable[float]) -> float:
    """Σx² / Σx — the ratio of the second raw moment to the first.
    Sits just above the arithmetic mean; positive data only."""
    vals = _as_list(xs)
    _positive(vals, "contraharmonic_mean")
    return sum(x * x for x in vals) / sum(vals)


def heronian_mean(a: float, b: float) -> float:
    """(a + √(ab) + b) / 3 — the two-value mean Heron used for
    frustum volumes. Between the arithmetic and geometric means;
    non-negative data only."""
    if a < 0 or b < 0:
        raise ValueError("heronian_mean is defined on non-negative data")
    return (a + math.sqrt(a * b) + b) / 3.0


def heinz_mean(a: float, b: float, t: float = 0.5) -> float:
    """The symmetric interpolating mean (aᵗb¹⁻ᵗ + a¹⁻ᵗbᵗ)/2.

    t=1/2 gives the geometric mean; t→0 or 1 gives the arithmetic
    mean; intermediate t interpolates between them. Positive data,
    0 ≤ t ≤ 1.
    """
    if a <= 0 or b <= 0:
        raise ValueError("heinz_mean is defined on positive data")
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must lie in [0, 1]")
    return (a**t * b ** (1.0 - t) + a ** (1.0 - t) * b**t) / 2.0


def lehmer_mean(a: float, b: float, p: float) -> float:
    """(aᵖ + bᵖ) / (aᵖ⁻¹ + bᵖ⁻¹) — the Lehmer family.

    Another ladder the named means sit on for two values: p=1
    arithmetic, p=2 contraharmonic, p=1/2 geometric, p=0 harmonic;
    the extremes recover max and min. Positive data."""
    if a <= 0 or b <= 0:
        raise ValueError("lehmer_mean is defined on positive data")
    num = a**p + b**p
    den = a ** (p - 1.0) + b ** (p - 1.0)
    if den == 0:
        raise ValueError("lehmer_mean: denominators vanish at this p")
    return num / den


def median(xs: Iterable[float]) -> float:
    """The middle value — robust where the arithmetic mean is not.
    Even counts average the two middle values."""
    vals = sorted(_as_list(xs))
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def mode(xs: Iterable[float]) -> float:
    """The most frequent value — the only mean defined on
    non-numeric data. Ties return the smallest mode."""
    vals = _as_list(xs)
    counts = Counter(vals)
    top = max(counts.values())
    return min(v for v, c in counts.items() if c == top)


def midrange(xs: Iterable[float]) -> float:
    """(min + max) / 2 — the midpoint of the range itself."""
    vals = _as_list(xs)
    return (min(vals) + max(vals)) / 2.0


def trimmed_mean(xs: Iterable[float], proportion: float = 0.1) -> float:
    """Arithmetic mean of the middle (1 − 2·proportion) of the
    data — the mean with the tails cut off. Robustness through
    exclusion rather than weighting."""
    vals = sorted(_as_list(xs))
    if not 0.0 <= proportion < 0.5:
        raise ValueError("trim proportion must lie in [0, 0.5)")
    k = int(len(vals) * proportion)
    inner = vals[k : len(vals) - k]
    return sum(inner) / len(inner)


# ── Dispersion ───────────────────────────────────────────────────


def variance(xs: Iterable[float], sample: bool = False) -> float:
    """Mean squared deviation from the mean — the second central
    moment. ``sample=True`` divides by n−1 (Bessel's correction)."""
    vals = _as_list(xs)
    if sample and len(vals) < 2:
        raise ValueError("sample variance needs at least two values")
    mu = arithmetic_mean(vals)
    return sum((x - mu) ** 2 for x in vals) / (len(vals) - (1 if sample else 0))


def standard_deviation(xs: Iterable[float], sample: bool = False) -> float:
    """√variance — spread in the data's own units."""
    return math.sqrt(variance(xs, sample=sample))


def mean_absolute_deviation(xs: Iterable[float]) -> float:
    """Mean |x − mean| — dispersion in plain units; less sensitive
    to tails than the standard deviation."""
    vals = _as_list(xs)
    mu = arithmetic_mean(vals)
    return sum(abs(x - mu) for x in vals) / len(vals)


def median_absolute_deviation(xs: Iterable[float]) -> float:
    """Median |x − median| — the fully robust spread: it survives
    up to half the data being outliers."""
    vals = _as_list(xs)
    med = median(vals)
    return median(abs(x - med) for x in vals)


def percentile(xs: Iterable[float], q: float) -> float:
    """The q-th percentile — linear interpolation between ranks
    (the type-7 convention, numpy/R default)."""
    vals = sorted(_as_list(xs))
    if not 0.0 <= q <= 100.0:
        raise ValueError("percentile q must lie in [0, 100]")
    pos = (q / 100.0) * (len(vals) - 1)
    lo = math.floor(pos)
    frac = pos - lo
    if lo + 1 < len(vals):
        return vals[lo] + frac * (vals[lo + 1] - vals[lo])
    return vals[lo]


def percentile_range(xs: Iterable[float], lo: float = 25.0, hi: float = 75.0) -> float:
    """hi-percentile − lo-percentile — the width of any chosen
    middle band of the distribution."""
    return percentile(xs, hi) - percentile(xs, lo)


def interquartile_range(xs: Iterable[float]) -> float:
    """Q3 − Q1 — the width of the middle half; the spread that
    ignores both tails entirely."""
    return percentile_range(xs, 25.0, 75.0)


def data_range(xs: Iterable[float]) -> float:
    """max − min — the whole width, tails and all."""
    vals = _as_list(xs)
    return max(vals) - min(vals)


def coefficient_of_variation(xs: Iterable[float]) -> float:
    """σ / μ — dispersion relative to the mean. Unit-free, so it
    compares spreads across differently-scaled data."""
    vals = _as_list(xs)
    mu = arithmetic_mean(vals)
    if mu == 0:
        raise ValueError("coefficient of variation undefined at mean 0")
    return standard_deviation(vals) / abs(mu)


# ── Shape — moments and the sampling theorem ─────────────────────


def raw_moment(xs: Iterable[float], k: int) -> float:
    """E[xᵏ] — the k-th raw moment. k=1 is the arithmetic mean."""
    vals = _as_list(xs)
    return sum(x**k for x in vals) / len(vals)


def central_moment(xs: Iterable[float], k: int) -> float:
    """E[(x − μ)ᵏ] — the k-th moment about the mean. k=2 is the
    population variance; higher k describe shape."""
    vals = _as_list(xs)
    mu = arithmetic_mean(vals)
    return sum((x - mu) ** k for x in vals) / len(vals)


def skewness(xs: Iterable[float]) -> float:
    """m₃ / m₂^1.5 — the direction of the tail. Negative leans
    left, positive leans right, zero is symmetric."""
    m2 = central_moment(xs, 2)
    if m2 == 0:
        raise ValueError("skewness undefined on constant data")
    return central_moment(xs, 3) / m2**1.5


def kurtosis(xs: Iterable[float], excess: bool = False) -> float:
    """m₄ / m₂² — the weight of the tails relative to the
    shoulders. ``excess=True`` subtracts 3 so the normal
    distribution reads 0."""
    m2 = central_moment(xs, 2)
    if m2 == 0:
        raise ValueError("kurtosis undefined on constant data")
    k = central_moment(xs, 4) / (m2 * m2)
    return k - 3.0 if excess else k


def standard_error(xs: Iterable[float]) -> float:
    """σ / √n — the expected spread of the *mean itself* if the
    sampling were repeated. The quantity the CLT predicts."""
    vals = _as_list(xs)
    return standard_deviation(vals) / math.sqrt(len(vals))


def sampling_means(xs: Iterable[float], n: int, draws: int, rng: object) -> list[float]:
    """Draw ``draws`` samples of size ``n`` with replacement and
    return their means — the empirical sampling distribution of
    the mean.

    This is the central limit theorem made concrete rather than
    stated: whatever the shape of ``xs``, the returned means
    cluster around ``arithmetic_mean(xs)`` with spread ≈ σ/√n —
    ``standard_error`` is the prediction, this list is the
    observation. ``rng`` is any object with ``.choices(pop, k=n)``;
    pass ``random.Random(seed)`` for reproducibility.
    """
    vals = _as_list(xs)
    if n < 1 or draws < 1:
        raise ValueError("n and draws must be positive")
    return [
        arithmetic_mean(rng.choices(vals, k=n))  # type: ignore[attr-defined]
        for _ in range(draws)
    ]


# ── The dispatcher and the summary ───────────────────────────────

#: The named means reachable through :func:`mean`. Power-mean
#: members take ``p``; Heronian/Heinz/Lehmer are two-value means
#: and take ``a``/``b`` rather than a data list.
MEAN_KINDS = (
    "arithmetic",
    "geometric",
    "harmonic",
    "quadratic",
    "cubic",
    "contraharmonic",
    "power",
    "median",
    "mode",
    "midrange",
)


def mean(xs: Iterable[float], kind: str = "arithmetic", p: float = 1.0) -> float:
    """Compute a named mean over *xs*.

    ``kind`` is one of :data:`MEAN_KINDS`; ``p`` is the order for
    ``kind="power"`` and ignored otherwise.
    """
    vals = _as_list(xs)
    if kind == "arithmetic":
        return arithmetic_mean(vals)
    if kind == "geometric":
        return geometric_mean(vals)
    if kind == "harmonic":
        return harmonic_mean(vals)
    if kind == "quadratic":
        return quadratic_mean(vals)
    if kind == "cubic":
        return cubic_mean(vals)
    if kind == "contraharmonic":
        return contraharmonic_mean(vals)
    if kind == "power":
        return power_mean(vals, p)
    if kind == "median":
        return median(vals)
    if kind == "mode":
        return mode(vals)
    if kind == "midrange":
        return midrange(vals)
    raise ValueError(f"unknown mean kind: {kind!r}")


@dataclass(frozen=True)
class StatSummary:
    """A full descriptive profile of a sample."""

    n: int
    minimum: float
    maximum: float
    arithmetic_mean: float
    median: float
    mode: float
    variance: float
    standard_deviation: float
    mean_absolute_deviation: float
    interquartile_range: float
    skewness: float | None  # None on constant data
    kurtosis: float | None  # raw; None on constant data
    standard_error: float


def summarize(xs: Iterable[float]) -> StatSummary:
    """The whole picture at once — center, spread, and shape."""
    vals = _as_list(xs)
    m2 = central_moment(vals, 2)
    return StatSummary(
        n=len(vals),
        minimum=min(vals),
        maximum=max(vals),
        arithmetic_mean=arithmetic_mean(vals),
        median=median(vals),
        mode=mode(vals),
        variance=m2,
        standard_deviation=math.sqrt(m2),
        mean_absolute_deviation=mean_absolute_deviation(vals),
        interquartile_range=interquartile_range(vals),
        skewness=None if m2 == 0 else central_moment(vals, 3) / m2**1.5,
        kurtosis=None if m2 == 0 else central_moment(vals, 4) / (m2 * m2),
        standard_error=math.sqrt(m2) / math.sqrt(len(vals)),
    )
