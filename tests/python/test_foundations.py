"""Tests for foundationalism — first principles, demonstration,
indispensability, and the Agrippan audit."""

from frontal_lobe.reasoning.foundations import Foundations, audit
from frontal_lobe.reasoning.logic import atom, parse


def test_first_principles_derive_theorems():
    """The archai entail the rest — episteme is demonstration from
    indemonstrable starting points."""
    f = Foundations(frozenset({atom("rain"), parse("rain -> wet"), parse("wet -> slippery")}))
    assert f.proves(atom("slippery"))
    assert f.is_basic(atom("rain"))
    assert f.is_derived(atom("slippery"))


def test_demonstrate_returns_the_warrant_or_countermodel():
    f = Foundations(frozenset({atom("p"), parse("p -> q")}))
    assert f.demonstrate(atom("q")).valid
    bad = f.demonstrate(atom("r"))
    assert not bad.valid
    assert bad.countermodel is not None


def test_support_finds_the_minimal_foundation():
    """Which principles carry the claim — the smallest sufficient set."""
    axioms = frozenset({atom("a"), parse("a -> b"), atom("c"), parse("c -> d")})
    f = Foundations(axioms)
    assert f.support(atom("b")) == frozenset({atom("a"), parse("a -> b")})
    # Nothing reaches an unconnected claim.
    assert f.support(atom("z")) is None


def test_indispensable_axioms_are_load_bearing():
    """An axiom is indispensable iff removing it loses the proof."""
    axioms = frozenset({atom("p"), parse("p -> q")})
    f = Foundations(axioms)
    ind = f.indispensable(atom("q"))
    assert ind == axioms  # both are load-bearing
    # But an atom with a redundant route has no indispensable member.
    axioms2 = frozenset({atom("p"), atom("q"), parse("p -> q")})
    assert Foundations(axioms2).indispensable(atom("q")) == frozenset()


def test_audit_grounds_the_edifice():
    """Every belief bottoms out at a first principle."""
    f = Foundations(frozenset({atom("a"), parse("a -> b")}))
    result = audit(f, {atom("a"), atom("b"), parse("a -> b")})
    assert result.grounded
    assert atom("a") in result.basic
    assert atom("b") in result.derived


def test_audit_flags_ungrounded_beliefs():
    """A belief no chain reaches from the basics is Agrippa's regress."""
    f = Foundations(frozenset({atom("a"), parse("a -> b")}))
    result = audit(f, {atom("b"), atom("z")})
    assert atom("z") in result.ungrounded
    assert not result.grounded


def test_audit_detects_circular_justification():
    """b justified by c, c justified by b — Agrippa's circle: the
    beliefs support each other but nothing supports either."""
    f = Foundations(frozenset())  # no archai at all
    # b follows from c, c follows from b: a pure circle.
    result = audit(
        f,
        {
            parse("c -> b"),
            parse("b -> c"),
            atom("b"),
            atom("c"),
        },
    )
    assert not result.grounded
    assert atom("b") in {x for circle in result.circular for x in circle}
