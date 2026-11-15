"""Tests for parietal_lobe.math.statistics.

Covers the named means and their proven relationships (the mean
inequality chain, the power-mean and Lehmer ladders recovering the
classical cases), the dispersion family, the shape moments, and the
empirical central-limit demonstration.
"""

from __future__ import annotations

import math
import random

import pytest
from parietal_lobe.math import looks_like_statistics, summarize
from parietal_lobe.math import statistics as st

# ── Central tendency ─────────────────────────────────────────────


def test_arithmetic_mean():
    assert st.arithmetic_mean([1, 2, 3, 4]) == 2.5


def test_geometric_mean():
    assert st.geometric_mean([1, 4]) == 2.0
    assert st.geometric_mean([2, 8]) == 4.0


def test_harmonic_mean():
    assert st.harmonic_mean([1, 2, 4]) == pytest.approx(12 / 7)
    # The classic: equal-speed round trip averages the harmonic way.
    assert st.harmonic_mean([30, 60]) == 40.0


def test_mean_inequality_holds():
    # For positive non-constant data:
    #   harmonic ≤ geometric ≤ arithmetic ≤ contraharmonic
    xs = [1.0, 3.0, 5.0, 9.0]
    h, g, a, c = (
        st.harmonic_mean(xs),
        st.geometric_mean(xs),
        st.arithmetic_mean(xs),
        st.contraharmonic_mean(xs),
    )
    assert h < g < a < c


def test_power_mean_ladder_recovers_named_means():
    xs = [2.0, 4.0, 8.0]
    assert st.power_mean(xs, 1) == pytest.approx(st.arithmetic_mean(xs))
    assert st.power_mean(xs, 2) == pytest.approx(st.quadratic_mean(xs))
    assert st.power_mean(xs, 3) == pytest.approx(st.cubic_mean(xs))
    assert st.power_mean(xs, -1) == pytest.approx(st.harmonic_mean(xs))
    assert st.power_mean(xs, 0) == pytest.approx(st.geometric_mean(xs))


def test_quadratic_and_cubic_means():
    assert st.quadratic_mean([1, 7]) == 5.0  # sqrt((1+49)/2)
    assert st.cubic_mean([1, 2]) == pytest.approx((9 / 2) ** (1 / 3))


def test_contraharmonic_mean():
    assert st.contraharmonic_mean([1, 2, 3]) == pytest.approx(14 / 6)


def test_heronian_mean_between_geometric_and_arithmetic():
    a, b = 4.0, 9.0
    h = st.heronian_mean(a, b)  # (4 + 6 + 9)/3
    assert h == pytest.approx(19 / 3)
    assert st.geometric_mean([a, b]) < h < st.arithmetic_mean([a, b])


def test_heinz_mean_endpoints():
    a, b = 2.0, 8.0
    assert st.heinz_mean(a, b, 0.5) == pytest.approx(st.geometric_mean([a, b]))
    assert st.heinz_mean(a, b, 0.0) == pytest.approx(st.arithmetic_mean([a, b]))
    assert st.heinz_mean(a, b, 1.0) == pytest.approx(st.arithmetic_mean([a, b]))


def test_lehmer_mean_ladder():
    a, b = 2.0, 8.0
    assert st.lehmer_mean(a, b, 1.0) == pytest.approx(st.arithmetic_mean([a, b]))
    assert st.lehmer_mean(a, b, 0.5) == pytest.approx(st.geometric_mean([a, b]))
    assert st.lehmer_mean(a, b, 0.0) == pytest.approx(st.harmonic_mean([a, b]))
    assert st.lehmer_mean(a, b, 2.0) == pytest.approx(st.contraharmonic_mean([a, b]))
    # The extremes recover max and min.
    assert st.lehmer_mean(a, b, 100.0) == pytest.approx(max(a, b), rel=1e-3)


def test_median():
    assert st.median([3, 1, 2]) == 2.0
    assert st.median([4, 1, 3, 2]) == 2.5


def test_mode():
    assert st.mode([1, 2, 2, 3]) == 2.0


def test_midrange_and_trimmed_mean():
    xs = list(range(1, 11))
    assert st.midrange(xs) == 5.5
    assert st.trimmed_mean(xs, 0.2) == pytest.approx(5.5)
    # Trimming kills the outlier; the plain mean does not.
    skewed = [1, 2, 3, 4, 100]
    assert st.trimmed_mean(skewed, 0.2) < st.arithmetic_mean(skewed)


def test_positive_only_means_reject_nonpositive():
    with pytest.raises(ValueError):
        st.geometric_mean([1, -2, 3])
    with pytest.raises(ValueError):
        st.harmonic_mean([0, 1])
    with pytest.raises(ValueError):
        st.lehmer_mean(-1, 2, 1)


def test_empty_data_rejected():
    with pytest.raises(ValueError):
        st.arithmetic_mean([])


# ── Dispersion ───────────────────────────────────────────────────


def test_variance_and_sd():
    xs = [2, 4, 4, 4, 5, 5, 7, 9]  # the textbook example: var = 4
    assert st.variance(xs) == pytest.approx(4.0)
    assert st.standard_deviation(xs) == pytest.approx(2.0)
    # Bessel's correction inflates the sample estimate.
    assert st.variance(xs, sample=True) == pytest.approx(32 / 7)


def test_absolute_deviations():
    xs = [2, 4, 6, 8]
    assert st.mean_absolute_deviation(xs) == 2.0
    assert st.median_absolute_deviation(xs) == 2.0


def test_percentile_and_ranges():
    xs = list(range(1, 101))
    assert st.percentile(xs, 50) == pytest.approx(50.5)
    assert st.percentile(xs, 0) == 1.0
    assert st.percentile(xs, 100) == 100.0
    assert st.interquartile_range(xs) == pytest.approx(49.5)
    assert st.percentile_range(xs, 10, 90) == pytest.approx(79.2)
    assert st.data_range(xs) == 99.0


def test_coefficient_of_variation():
    xs = [10, 10, 10, 10]
    assert st.coefficient_of_variation(xs) == 0.0
    with pytest.raises(ValueError):
        st.coefficient_of_variation([-1, 1])  # mean 0


# ── Shape ────────────────────────────────────────────────────────


def test_moments():
    xs = [1, 2, 3, 4]
    assert st.raw_moment(xs, 1) == 2.5  # the arithmetic mean
    assert st.raw_moment(xs, 2) == pytest.approx(7.5)
    assert st.central_moment(xs, 1) == pytest.approx(0.0)
    assert st.central_moment(xs, 2) == pytest.approx(st.variance(xs))


def test_skewness_sign():
    symmetric = [1, 2, 3, 4, 5]
    assert st.skewness(symmetric) == pytest.approx(0.0)
    right_tailed = [1, 1, 1, 2, 2, 3, 9]
    assert st.skewness(right_tailed) > 0


def test_kurtosis():
    # A peaked distribution: excess kurtosis > 0.
    peaked = [5] * 20 + [1, 9]
    assert st.kurtosis(peaked, excess=True) > 0
    with pytest.raises(ValueError):
        st.kurtosis([3, 3, 3])  # constant data


def test_standard_error_shrinks_with_n():
    base = [1, 2, 3, 4, 5, 6, 7, 8]
    assert st.standard_error(base) == pytest.approx(st.standard_deviation(base) / math.sqrt(8))


def test_sampling_means_demonstrates_clt():
    # Wildly non-normal source: bimodal extremes.
    pop = [0.0] * 500 + [10.0] * 500
    rng = random.Random(7)
    means = st.sampling_means(pop, n=40, draws=300, rng=rng)
    assert len(means) == 300
    # CLT: means cluster at the population mean with spread ≈ σ/√n,
    # where n is the sample size drawn each time (40).
    assert st.arithmetic_mean(means) == pytest.approx(st.arithmetic_mean(pop), abs=0.3)
    assert st.standard_deviation(means) == pytest.approx(
        st.standard_deviation(pop) / math.sqrt(40), rel=0.15
    )


# ── Dispatcher and summary ───────────────────────────────────────


def test_mean_dispatcher():
    xs = [2.0, 4.0, 8.0]
    assert st.mean(xs) == pytest.approx(14 / 3)
    assert st.mean(xs, "harmonic") == pytest.approx(st.harmonic_mean(xs))
    assert st.mean(xs, "power", p=3) == pytest.approx(st.cubic_mean(xs))
    assert st.mean(xs, "median") == 4.0
    with pytest.raises(ValueError):
        st.mean(xs, "fibonacci")


def test_summarize():
    s = summarize([2, 4, 4, 4, 5, 5, 7, 9])
    assert s.n == 8
    assert s.minimum == 2.0 and s.maximum == 9.0
    assert s.arithmetic_mean == 5.0
    assert s.median == 4.5
    assert s.mode == 4.0
    assert s.variance == pytest.approx(4.0)
    assert s.skewness is not None and s.kurtosis is not None


def test_summarize_constant_data():
    s = summarize([7.0] * 5)
    assert s.variance == 0.0
    assert s.skewness is None and s.kurtosis is None


def test_looks_like_statistics():
    assert looks_like_statistics("what is the mean of these numbers")
    assert looks_like_statistics("harmonic mean of 4 and 6")
    assert looks_like_statistics("give me the variance")
    assert not looks_like_statistics("what is 7 + 5")
    assert not looks_like_statistics("hello there")
