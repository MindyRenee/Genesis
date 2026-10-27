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

And the higher faculties — one file per domain, each mirrored by a
pure-machine twin under ``src/math/`` in the body:

    trig.py             — the circle's vocabulary: angles, ratios,
                          triangles.
    linear_algebra.py   — spaces and maps: matrices, determinants,
                          systems.
    abstract_algebra.py — structure itself: permutations, cycles,
                          orbits, groups. The cube's home.
    real_analysis.py    — limits made honest: the epsilon-delta
                          game, graphs as (x, y) points.
    complex_analysis.py — the plane of numbers: polar form, roots
                          of unity, De Moivre.
    number_theory.py    — the integers' inner life: primes,
                          factors, congruences.
    topology.py         — shape without distance: opens, closure,
                          the Euler characteristic.
    geometry.py         — shape with distance: areas, hulls,
                          point-in-polygon.
    discrete.py         — relations, sets, graphs — the decidable
                          finite.
    combinatorics.py    — counting without listing: choose,
                          arrange, derange.

And the three pillars mathematical thinking stands on:

    axioms.py           — the ladder of laws: closure through
                          abelian group, each rung witnessed.
    proof.py            — valid inference: truth tables, entailment,
                          the named rules.

``symbolic.py`` sits beside them as the oracle — the sympy-backed
checker ``frontal_lobe/research.py`` verifies claims against. The
faculty here is endogenous: it manipulates structure itself, and
its results are verified, never asserted.

``ParietalMath`` is the facade the rest of the mind talks to.
"""

from __future__ import annotations

import re

from .abstract_algebra import (
    GroupResult,
    compose_text,
    looks_like_group,
    order_text,
)
from .abstract_algebra import (
    describe_text as group_describe_text,
)
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
from .axioms import AxiomResult, classify_text, looks_like_axioms
from .calculus import (
    derivative_at_text,
    derivative_text,
    integral_between_text,
    integral_text,
    taylor,
)
from .combinatorics import (
    CountResult,
    arrange_text,
    choose_text,
    derange_text,
    looks_like_counting,
)
from .complex_analysis import (
    ComplexResult,
    looks_like_complex,
    power_text,
    roots_text,
    roots_unity_text,
)
from .complex_analysis import (
    describe_text as complex_describe_text,
)
from .discrete import (
    DiscreteResult,
    connected_text,
    looks_like_discrete,
    relation_text,
)
from .geometry import (
    GeoResult,
    area_text,
    distance_text,
    hull_text,
    inside_text,
    looks_like_geometry,
)
from .linear_algebra import (
    LinearResult,
    determinant_text,
    looks_like_linear_algebra,
    matmul_text,
    rank_text,
)
from .linear_algebra import (
    solve_text as solve_system_text,
)
from .number_sense import MagnitudeJudgment, NumberSense
from .number_theory import (
    NumberResult,
    crt_text,
    gcd_text,
    looks_like_number_theory,
    prime_text,
    solve_congruence_text,
)
from .number_theory import (
    factor_text as factor_number_text,
)
from .proof import (
    ProofResult,
    entail_text,
    looks_like_proof,
    satisfiable_text,
    tautology_text,
)
from .real_analysis import (
    AnalysisResult,
    limit_text,
    looks_like_analysis,
    plot_text,
)
from .real_analysis import (
    integral_text as numeric_integral_text,
)
from .statistics import MEAN_KINDS, StatSummary, mean, summarize
from .topology import (
    TopoResult,
    describe_space_text,
    euler_surface,
    looks_like_topology,
)
from .trig import (
    TrigResult,
    convert_text,
    looks_like_trig,
    ratio_text,
    triangle_text,
)

__all__ = [
    "MEAN_KINDS",
    "AlgebraResult",
    "AnalysisResult",
    "AxiomResult",
    "CalcResult",
    "ComplexResult",
    "CountResult",
    "DiscreteResult",
    "GeoResult",
    "GroupResult",
    "LinearResult",
    "MagnitudeJudgment",
    "NumberResult",
    "NumberSense",
    "ParietalMath",
    "ProofResult",
    "StatSummary",
    "TopoResult",
    "TrigResult",
    "euler_surface",
    "looks_like_algebra",
    "looks_like_analysis",
    "looks_like_axioms",
    "looks_like_calculus",
    "looks_like_complex",
    "looks_like_counting",
    "looks_like_discrete",
    "looks_like_geometry",
    "looks_like_group",
    "looks_like_linear_algebra",
    "looks_like_math",
    "looks_like_number_theory",
    "looks_like_proof",
    "looks_like_statistics",
    "looks_like_topology",
    "looks_like_trig",
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

    # ── Trigonometry — the circle's vocabulary ──────────────────

    def looks_like_trig(self, text: str) -> bool:
        """Whether the text asks about angles, ratios, or triangles."""
        return looks_like_trig(text)

    def trig_ratio(self, func: str, angle_text: str) -> TrigResult:
        """A ratio at an angle — exact names for the famous angles."""
        return ratio_text(func, _extract(angle_text))

    def convert_angle(self, text: str, to: str) -> TrigResult:
        """Degrees to radians or back."""
        return convert_text(_extract(text), to)

    def triangle(self, sides: list[float]) -> TrigResult:
        """Solve a triangle from three sides."""
        return triangle_text(sides)

    # ── Linear algebra — spaces and maps ─────────────────────────

    def looks_like_linear_algebra(self, text: str) -> bool:
        """Whether the text asks a matrix question."""
        return looks_like_linear_algebra(text)

    def determinant(self, text: str) -> LinearResult:
        """The determinant of a bracketed matrix — exact."""
        return determinant_text(text)

    def matrix_rank(self, text: str) -> LinearResult:
        """The rank of a bracketed matrix."""
        return rank_text(text)

    def matmul(self, a_text: str, b_text: str) -> LinearResult:
        """The product of two bracketed matrices."""
        return matmul_text(a_text, b_text)

    def solve_system(self, a_text: str, b_text: str) -> LinearResult:
        """Solve Ax = b — exact fractions."""
        return solve_system_text(a_text, b_text)

    # ── Abstract algebra — structure itself ─────────────────────

    def looks_like_group(self, text: str) -> bool:
        """Whether the text asks about permutations or groups."""
        return looks_like_group(text)

    def permutation_order(self, text: str) -> GroupResult:
        """The order of a cycle-notation permutation."""
        return order_text(text)

    def permutation_compose(self, left: str, right: str) -> GroupResult:
        """Compose two permutations, left then right."""
        return compose_text(left, right)

    def permutation_describe(self, text: str) -> GroupResult:
        """Cycles, order, parity — the move's portrait."""
        return group_describe_text(text)

    # ── Real analysis — limits made honest ──────────────────────

    def looks_like_analysis(self, text: str) -> bool:
        """Whether the text asks about limits or convergence."""
        return looks_like_analysis(text)

    def limit(self, text: str, at: float) -> AnalysisResult:
        """The limit of an expression as x approaches a point."""
        return limit_text(_extract(text), at)

    def plot(self, text: str, lo: float, hi: float, n: int = 128) -> AnalysisResult:
        """The graph of an expression — its (x, y) points."""
        return plot_text(_extract(text), lo, hi, n)

    def numeric_integral(self, text: str, lo: float, hi: float) -> AnalysisResult:
        """The definite integral by Simpson's quadrature."""
        return numeric_integral_text(_extract(text), lo, hi)

    # ── Complex analysis — the plane of numbers ─────────────────

    def looks_like_complex(self, text: str) -> bool:
        """Whether the text asks about complex numbers."""
        return looks_like_complex(text)

    def complex_describe(self, text: str) -> ComplexResult:
        """Magnitude, argument, polar form — the number's portrait."""
        return complex_describe_text(text)

    def complex_power(self, text: str, n: int) -> ComplexResult:
        """A complex number raised to an integer power."""
        return power_text(text, n)

    def roots_of_unity(self, n: int) -> ComplexResult:
        """All n nth roots of unity."""
        return roots_unity_text(n)

    def complex_roots(self, text: str, n: int) -> ComplexResult:
        """All n nth roots of a complex number."""
        return roots_text(text, n)

    # ── Number theory — the integers' inner life ────────────────

    def looks_like_number_theory(self, text: str) -> bool:
        """Whether the text asks about primes, factors, congruences."""
        return looks_like_number_theory(text)

    def is_prime(self, n: int) -> NumberResult:
        """Primality with the factorization when composite."""
        return prime_text(n)

    def factor_number(self, n: int) -> NumberResult:
        """Prime factorization, with its check."""
        return factor_number_text(n)

    def gcd(self, a: int, b: int) -> NumberResult:
        """The greatest common divisor, with the Bezout witness."""
        return gcd_text(a, b)

    def solve_congruence(self, a: int, b: int, m: int) -> NumberResult:
        """Solve ax ≡ b (mod m)."""
        return solve_congruence_text(a, b, m)

    def chinese_remainder(self, pairs: list[tuple[int, int]]) -> NumberResult:
        """CRT over (residue, modulus) pairs."""
        return crt_text(pairs)

    # ── Topology — shape without distance ───────────────────────

    def looks_like_topology(self, text: str) -> bool:
        """Whether the text asks about opens, surfaces, connectedness."""
        return looks_like_topology(text)

    def euler_characteristic(self, v: int, e: int, f: int) -> TopoResult:
        """V − E + F — the count that knows the surface."""
        return euler_surface(v, e, f)

    def is_topology(self, n: int, opens: list[list[int]]) -> TopoResult:
        """Whether a collection of subsets is a topology on n points."""
        return describe_space_text(n, opens)

    # ── Geometry — shape with distance ──────────────────────────

    def looks_like_geometry(self, text: str) -> bool:
        """Whether the text asks about points, areas, hulls."""
        return looks_like_geometry(text)

    def distance(self, text: str) -> GeoResult:
        """Distance between two "(x, y)" points."""
        return distance_text(text)

    def polygon_area(self, text: str) -> GeoResult:
        """Area of a point list — the shoelace."""
        return area_text(text)

    def convex_hull(self, text: str) -> GeoResult:
        """The convex hull of a point list."""
        return hull_text(text)

    def point_inside(self, point_text: str, poly_text: str) -> GeoResult:
        """Whether a point lies inside a polygon — ray casting."""
        return inside_text(point_text, poly_text)

    # ── Discrete — relations, sets, graphs ──────────────────────

    def looks_like_discrete(self, text: str) -> bool:
        """Whether the text asks about relations, sets, or graphs."""
        return looks_like_discrete(text)

    def relation_properties(self, text: str, universe_text: str | None = None) -> DiscreteResult:
        """Which relation properties hold — reflexive through
        transitive, with the equivalence/order verdict."""
        return relation_text(text, universe_text)

    def graph_connected(self, text: str) -> DiscreteResult:
        """Whether an edge list is one connected component."""
        return connected_text(text)

    # ── Combinatorics — counting without listing ────────────────

    def looks_like_counting(self, text: str) -> bool:
        """Whether the text asks a counting question."""
        return looks_like_counting(text)

    def choose(self, n: int, k: int) -> CountResult:
        """C(n,k) — subsets, not orders."""
        return choose_text(n, k)

    def arrange(self, n: int, k: int | None = None) -> CountResult:
        """Orders: n! or P(n,k) — the multiplication rule."""
        return arrange_text(n, k)

    def derangements(self, n: int) -> CountResult:
        """!n — permutations leaving nothing home."""
        return derange_text(n)

    # ── Axioms — the ladder of laws ─────────────────────────────

    def looks_like_axioms(self, text: str) -> bool:
        """Whether the text asks about algebraic structure."""
        return looks_like_axioms(text)

    def classify_structure(self, table: list[list[int]]) -> AxiomResult:
        """Which rung a Cayley table reaches, with the axiom it
        fails and the witness."""
        return classify_text(table)

    # ── Proof — valid inference ─────────────────────────────────

    def looks_like_proof(self, text: str) -> bool:
        """Whether the text asks about proof or validity."""
        return looks_like_proof(text)

    def is_tautology(self, prop: tuple, words: str = "") -> ProofResult:
        """Whether a proposition tree is true under every world."""
        return tautology_text(prop, words)

    def is_satisfiable(self, prop: tuple, words: str = "") -> ProofResult:
        """Whether some world makes a proposition true — with one."""
        return satisfiable_text(prop, words)

    def entails(self, premises: list[tuple], goal: tuple) -> ProofResult:
        """Whether premises force a goal — or the countermodel."""
        return entail_text(premises, goal)
