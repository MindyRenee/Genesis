"""Mid-DLPFC rule induction — the Rational Rules model.

Pins the model's math: the grammar prior favors short derivations
and production reuse, the outlier likelihood penalizes misclassified
examples exponentially, the posterior converges on the true rule,
and confidence crossing threshold exposes the acquired signal —
the mid-DLPFC disengagement of Crescentini et al. (2011).
"""

from __future__ import annotations

import math

from frontal_lobe.dlpfc import (
    DorsolateralPFC,
    InducedRule,
    Literal,
)


def _rule(*senses: tuple[tuple[str, bool], ...]) -> InducedRule:
    """Build a rule from ((feature, present), ...) senses."""
    return InducedRule(tuple(frozenset(Literal(f, s) for f, s in sense) for sense in senses))


# ── Induction ───────────────────────────────────────────────


def test_conjunctive_rule_is_map() -> None:
    """Examples of 'a ∧ b' induce the conjunctive rule."""
    d = DorsolateralPFC()
    d.observe({"a", "b"}, True)
    d.observe({"a"}, False)
    d.observe({"b"}, False)
    d.observe(set(), False)
    result = d.induce()
    assert result.map_rule == _rule((("a", True), ("b", True)))
    assert result.confidence > 0.5
    assert d.predict({"a", "b"}) > 0.9
    assert d.predict({"a"}) < 0.5


def test_disjunctive_rule_is_map() -> None:
    """Examples of 'a ∨ b' induce a two-sense formula."""
    d = DorsolateralPFC()
    d.observe({"a"}, True)
    d.observe({"b"}, True)
    d.observe({"a", "b"}, True)
    d.observe(set(), False)
    d.observe({"c"}, False)
    result = d.induce()
    assert result.map_rule == _rule((("a", True),), (("b", True),))


def test_occam_simplest_consistent_rule_wins() -> None:
    """The syntactic-simplicity prior prefers the fewest literals."""
    d = DorsolateralPFC()
    d.observe({"a", "b"}, True)
    d.observe({"a"}, False)
    assert d.induce().map_rule == _rule((("b", True),))


def test_negated_literal_induced() -> None:
    """A membership rule on an *absent* feature is learnable."""
    d = DorsolateralPFC()
    d.observe(set(), True)
    d.observe({"b"}, True)
    d.observe({"a"}, False)
    d.observe({"a", "b"}, False)
    assert d.induce().map_rule == _rule((("a", False),))


def test_outlier_tolerance() -> None:
    """A single mislabeled example doesn't destroy the rule."""
    d = DorsolateralPFC()
    for _ in range(3):
        d.observe({"a", "b"}, True)
    d.observe({"a"}, False)
    d.observe({"a"}, False)
    d.observe({"b"}, False)
    d.observe({"b"}, False)
    d.observe(set(), False)
    d.observe({"a", "b"}, False)  # the outlier — mislabeled
    assert d.induce().map_rule == _rule((("a", True), ("b", True)))
    assert d.predict({"a", "b"}) > 0.5


def test_acquired_signal() -> None:
    """Confidence crosses threshold → the rule is learned."""
    d = DorsolateralPFC()
    for _ in range(2):
        d.observe({"a", "b"}, True)
        d.observe({"a"}, False)
        d.observe({"b"}, False)
    d.observe(set(), False)
    result = d.induce()
    assert result.acquired
    assert result.confidence >= d.acquire_threshold


# ── Prior structure ─────────────────────────────────────────


def test_shorter_derivation_has_higher_prior() -> None:
    d = DorsolateralPFC()
    d.observe({"a", "b"}, True)
    simple = _rule((("a", True),))
    complex_ = _rule((("a", True), ("b", True)))
    assert d._log_prior(simple) > d._log_prior(complex_)


def test_literal_reuse_is_cheaper_than_new_literals() -> None:
    """Reusing a literal across senses costs less prior mass —
    the Dirichlet-marginalized production reuse of the paper."""
    d = DorsolateralPFC()
    d.observe({"a", "b", "c"}, True)
    reuse = _rule((("a", True),), (("a", True), ("b", True)))
    fresh = _rule((("a", True),), (("b", True), ("c", True)))
    assert d._log_prior(reuse) > d._log_prior(fresh)


def test_dm_prior_formula() -> None:
    """Γ(k)/Γ(k+n) · ∏ n_j! for symmetric Dirichlet(1)."""
    from frontal_lobe.dlpfc import _dm_logp

    # k=2, counts (2,1): Γ(2)·2!·1!/Γ(5) = 2/24 = 1/12
    assert math.isclose(_dm_logp(2, (2, 1)), math.log(1 / 12))
    # k=4, counts (1,1): Γ(4)·1·1/Γ(6) = 6/120 = 1/20
    assert math.isclose(_dm_logp(4, (1, 1)), math.log(1 / 20))


# ── Evidence and inference mechanics ────────────────────────


def test_posterior_predictive_weights_uncertain_rules() -> None:
    """One positive example: matching item > 0.5, non-matching < 0.5."""
    d = DorsolateralPFC()
    d.observe({"a"}, True)
    assert d.predict({"a"}) > 0.5
    assert d.predict(set()) < 0.5


def test_keyed_observe_replaces_label() -> None:
    d = DorsolateralPFC()
    d.observe({"a"}, True, key="item-1")
    d.observe({"a"}, False, key="item-1")
    assert d.example_count == 1
    d.observe({"a", "b"}, True, key="item-2")
    assert d.example_count == 2


def test_deterministic() -> None:
    """Same evidence → same posterior (no sampling)."""
    kwargs = [({"a", "b"}, True), ({"a"}, False), ({"b"}, False)]
    d1, d2 = DorsolateralPFC(), DorsolateralPFC()
    for feats, label in kwargs:
        d1.observe(feats, label)
        d2.observe(feats, label)
    r1, r2 = d1.induce(), d2.induce()
    assert r1.map_rule == r2.map_rule
    assert r1.confidence == r2.confidence


def test_empty_space_predicts_chance() -> None:
    d = DorsolateralPFC()
    d.observe(set(), True)  # empty features → empty vocabulary
    assert d.predict({"a"}) == 0.5
    assert d.induce().map_rule is None


def test_reset() -> None:
    d = DorsolateralPFC()
    d.observe({"a"}, True)
    d.reset()
    assert not d.has_examples
    assert d.induce().map_rule is None


def test_persistence_roundtrip() -> None:
    d = DorsolateralPFC(exception_cost=5.0)
    d.observe({"a", "b"}, True)
    d.observe({"a"}, False)
    restored = DorsolateralPFC.from_dict(d.to_dict())
    assert restored.exception_cost == 5.0
    assert restored.induce().map_rule == d.induce().map_rule
    assert restored.induce().confidence == d.induce().confidence


def test_describe_is_formal_not_prose() -> None:
    rule = _rule((("a", True), ("b", False)), (("c", True),))
    assert rule.describe() == "(a ∧ ¬b) ∨ c"


def test_canonical_order() -> None:
    """Syntactically identical formulae are the same object."""
    a = _rule(
        (("b", True), ("a", True)),
    )
    b = _rule(
        (("a", True), ("b", True)),
    )
    assert a == b and hash(a) == hash(b)
