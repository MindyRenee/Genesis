"""Tests for semantic synthesis — compositional sense over real
parses, and truth-by-convention."""

from frontal_lobe.reasoning.logic import atom, not_, parse
from temporal_lobe.comprehension import Comprehension
from temporal_lobe.morphology import register_learned_verb
from temporal_lobe.semantics import (
    COMPLEMENTATION,
    analytic,
    analytic_status,
    synthesize,
)

_engine = Comprehension()


def _prop(text: str):
    return _engine.comprehend(text).propositions[0]


def test_predication_composes_subject_and_predicate():
    """The whole is a function of the parts — subject saturating the
    predicate's first place."""
    s = synthesize(_prop("Mary is in the kitchen"))
    assert ("subject", "Mary") in s.arguments
    assert ("location", "the kitchen") in s.arguments
    assert s.head == "is"


def test_roles_fill_their_own_slots():
    """Each semantic role saturates a distinct slot — the sense keeps
    them separate rather than flattening to surface order."""
    s = synthesize(_prop("John gave the ball to Mary"))
    slots = dict(s.arguments)
    assert slots["subject"] == "John"
    assert slots["object"] == "the ball"
    assert slots["goal"] == "Mary"


def test_complement_clauses_contribute_sense_not_reference():
    """'John believes that Mary is in the kitchen' — the embedded
    clause supplies its *sense* (what John believes), not its
    truth-value — Frege's oblique mode."""
    register_learned_verb("believe")
    s = synthesize(_prop("John believes that Mary is in the kitchen"))
    assert s.mode == COMPLEMENTATION
    assert len(s.embedded) == 1
    inner = s.embedded[0]
    assert ("subject", "Mary") in inner.arguments


def test_passive_swaps_the_subject_slot():
    """The patient grammatical subject composes as patient — the
    sense is about who was acted on, not word order. 'take' is a
    learned verb — a live organism has it; register it first."""
    register_learned_verb("take")
    s = synthesize(_prop("the ball was taken by Mary"))
    slots = dict(s.arguments)
    assert slots["patient"] == "the ball"
    assert slots["agent"] == "Mary"


def test_same_meaning_same_sense():
    """Compositionality: paraphrases that express the same meaning
    key identically."""
    a = synthesize(_prop("Mary is in the kitchen"))
    b = synthesize(_prop("Mary is in the kitchen"))
    assert a.key() == b.key()


def test_analytic_is_truth_by_convention():
    """Meaning postulates alone settle the analytic — no world
    required: 'bachelors are unmarried' follows from the rules."""
    postulates = [
        parse("bachelor -> unmarried"),
        parse("bachelor -> man"),
    ]
    assert analytic(parse("bachelor -> unmarried"), postulates)
    # The conjunction of the rules is itself analytic.
    assert analytic_status(parse("bachelor -> unmarried & man"), postulates) == "analytic"


def test_synthetic_is_what_convention_leaves_open():
    """What the postulates don't settle is synthetic — left to the
    world, where correspondence takes over."""
    postulates = [parse("bachelor -> unmarried")]
    assert analytic_status(atom("rich"), postulates) == "synthetic"
    assert analytic_status(not_(atom("unmarried")), postulates) == "synthetic"


def test_contradictory_is_analytically_false():
    """The third verdict: the postulates settle ¬f — analytically
    false, refuted by convention alone."""
    postulates = [parse("bachelor -> unmarried")]
    # 'bachelor and not unmarried' — its negation is entailed.
    f = parse("bachelor & ~unmarried")
    assert analytic_status(f, postulates) == "contradictory"
