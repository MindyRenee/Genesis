"""Exclusion — reasoning by elimination, the first logic.

Pins the organ's math: elimination transfers warrant to the residue
(the disjunction's yoke — what separates inference from avoidance),
the space resolves on a single survivor and falsifies when the
disjunction itself fails, and observation scores surprisal as
−log₂ P(outcome) — the LC phasic-burst drive, maximal for an
excluded or unentertained outcome (Cesana-Arlotti et al. 2018 —
the infant's pupil dilation).
"""

from __future__ import annotations

import math

import pytest
from frontal_lobe.exclusion import OptionSpace

# ── The infant task ───────────────────────────────────────────


def test_disjunctive_syllogism_resolves() -> None:
    """snake ∨ ball ; ¬ball ; ∴ snake — the Cesana-Arlotti scene."""
    space = OptionSpace(["snake", "ball"])
    result = space.eliminate("ball")
    assert result.resolved == "snake"
    assert space.posterior() == {"snake": 1.0}


def test_expected_outcome_is_no_surprise() -> None:
    """A resolved residue, confirmed, costs zero bits."""
    space = OptionSpace(["snake", "ball"])
    space.eliminate("ball")
    obs = space.observe("snake")
    assert obs.surprisal == 0.0
    assert not obs.impossible


def test_impossible_outcome_falsifies() -> None:
    """The scooped object still there: the disjunction was wrong."""
    space = OptionSpace(["snake", "ball"])
    space.eliminate("ball")
    obs = space.observe("ball")
    assert obs.impossible
    assert obs.surprisal == math.inf
    assert obs.falsified and space.falsified
    # Reality wins regardless — she saw what she saw.
    assert space.posterior() == {"ball": 1.0}


# ── Warrant transfer ──────────────────────────────────────────


def test_elimination_transfers_mass() -> None:
    """Negating one option *raises* the residue — the inference."""
    space = OptionSpace(["a", "b", "c"])
    result = space.eliminate("c")
    assert space.posterior() == pytest.approx({"a": 0.5, "b": 0.5})
    assert result.transferred["a"] == pytest.approx(1 / 6)
    assert result.transferred["b"] == pytest.approx(1 / 6)


def test_priors_bias_the_transfer() -> None:
    space = OptionSpace(["a", "b"], priors={"a": 3.0, "b": 1.0})
    space.eliminate("b")
    assert space.posterior() == {"a": 1.0}


def test_nonexclusive_space_transfers_nothing() -> None:
    """Negation without disjunction is avoidance, not inference."""
    space = OptionSpace(["a", "b"], priors={"a": 0.7, "b": 0.4}, exclusive=False)
    result = space.eliminate("b")
    assert result.transferred == {}
    assert space.posterior() == {"a": 0.7}


def test_unentertained_outcome_is_impossible() -> None:
    space = OptionSpace(["a", "b"])
    obs = space.observe("c")
    assert obs.impossible and obs.falsified


# ── Terminal states ───────────────────────────────────────────


def test_eliminating_all_falsifies_the_disjunction() -> None:
    space = OptionSpace(["a", "b"])
    result = space.eliminate("a", "b")
    assert result.falsified
    assert space.falsified


def test_eliminating_absent_option_is_noop() -> None:
    space = OptionSpace(["a", "b"])
    result = space.eliminate("zebra")
    assert result.eliminated == ()
    assert not result.falsified


def test_reset_opens_a_new_ambiguity() -> None:
    space = OptionSpace(["a", "b"])
    space.eliminate("a", "b")
    assert space.falsified
    space.reset()
    space.entertain("x")
    assert not space.falsified
    assert space.posterior() == {"x": 1.0}


# ── The LC drive ──────────────────────────────────────────────


def test_surprise_callback_fires() -> None:
    """The on_surprise hook receives (surprisal, impossible)."""
    seen: list[tuple[float, bool]] = []
    space = OptionSpace(["a", "b"], on_surprise=lambda s, i: seen.append((s, i)))
    space.observe("a")
    space.reset()
    space.entertain("x")
    space.entertain("y")
    space.eliminate("x")
    space.observe("x")
    assert seen == [(1.0, False), (math.inf, True)]


# ── The referent consumer ─────────────────────────────────────


class _Exec:
    """Minimal central-executive stub: records focus bindings."""

    def __init__(self) -> None:
        self.focus: str | None = None

    def direct_attention(self, items, relevant):
        self.focus = relevant[0]


class _WM:
    def __init__(self) -> None:
        self.central_executive = _Exec()


def _lobe() -> object:
    """A FrontalLobe carrying only the organs the resolver needs."""
    from frontal_lobe.prefrontal import FrontalLobe

    lobe = FrontalLobe.__new__(FrontalLobe)
    lobe.exclusion = OptionSpace()
    lobe.working_memory = _WM()
    return lobe


def test_clarification_observes_and_binds() -> None:
    """'Alice did X to Bob' — 'I meant Bob' resolves and binds."""
    lobe = _lobe()
    for rank, e in enumerate(["Alice", "Bob"]):
        lobe.exclusion.entertain(e, 0.5**rank)
    lobe._resolve_referent_space("I meant Bob")
    assert lobe.exclusion.resolved == "Bob"
    assert lobe.working_memory.central_executive.focus == "Bob"


def test_unrelated_turn_abandons_the_space() -> None:
    lobe = _lobe()
    lobe.exclusion.entertain("Alice")
    lobe.exclusion.entertain("Bob")
    lobe._resolve_referent_space("what's the weather like")
    assert len(lobe.exclusion) == 0


def test_contending_mentions_keep_the_space_open() -> None:
    lobe = _lobe()
    lobe.exclusion.entertain("Alice")
    lobe.exclusion.entertain("Bob")
    lobe._resolve_referent_space("Alice or Bob, either way")
    assert len(lobe.exclusion) == 2
