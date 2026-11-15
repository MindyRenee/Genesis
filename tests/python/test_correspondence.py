"""Tests for parietal correspondence — truth as relation between
proposition and state of affairs."""

from itertools import pairwise

from parietal_lobe.correspondence import (
    CorrespondenceLedger,
    Verdict,
    _claim_of,
)
from parietal_lobe.situation_model import SituationModel
from temporal_lobe.comprehension import Comprehension, Proposition
from temporal_lobe.morphology import register_learned_verb

_engine = Comprehension()


def _say(model: SituationModel, text: str) -> None:
    """Assert a sentence into the model — what was said becomes the
    described world."""
    model.update(_engine.comprehend(text))


def _prop(text: str) -> Proposition:
    """The first proposition of a real parse."""
    return _engine.comprehend(text).propositions[0]


def _check(model: SituationModel, text: str):
    ledger = CorrespondenceLedger(model)
    return ledger.check(_prop(text))


def test_supported_location() -> None:
    """The model affirms the state of affairs — the statement
    corresponds to the described world."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    c = _check(model, "Mary is in the kitchen")
    assert c.verdict is Verdict.SUPPORTED
    assert c.true is True


def test_contradicted_location() -> None:
    """The model affirms the negation — the state of affairs does
    not obtain."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    c = _check(model, "Mary is in the bathroom")
    assert c.verdict is Verdict.CONTRADICTED
    assert c.true is False


def test_negated_claim_supported() -> None:
    """'Mary is not in the bathroom' corresponds when the model has
    her settled elsewhere."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    c = _check(model, "Mary is not in the bathroom")
    assert c.verdict is Verdict.SUPPORTED


def test_negated_claim_contradicted() -> None:
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    c = _check(model, "Mary is not in the kitchen")
    assert c.verdict is Verdict.CONTRADICTED


def test_unknown_leaves_the_prior_untouched() -> None:
    """The model's silence is not evidence — an unchecked claim
    honestly holds the Beta(1,1) prior at 0.5."""
    model = SituationModel()
    ledger = CorrespondenceLedger(model)
    c = ledger.check(_prop("Mary is in the kitchen"))
    assert c.verdict is Verdict.UNKNOWN
    assert c.degree == 0.5
    assert c.evidence == 0.0


def test_unsettled_is_equivocal_evidence() -> None:
    """A 'maybe' answer speaks but cannot decide — evidence grows
    while the degree holds mid."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen or the bathroom")
    ledger = CorrespondenceLedger(model)
    c = ledger.check(_prop("Mary is in the kitchen"))
    assert c.verdict is Verdict.UNSETTLED
    assert c.evidence > 0.0
    assert c.degree == 0.5


def test_degree_approaches_convention_t_limit() -> None:
    """Repeated confirmation drives τ → 1 — 'p is true' approaches
    'p obtains' — without ever asserting it outright."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    ledger = CorrespondenceLedger(model)
    degrees = [ledger.check(_prop("Mary is in the kitchen")).degree for _ in range(5)]
    assert all(a < b for a, b in pairwise(degrees))
    assert degrees[-1] > 0.8
    assert degrees[-1] < 1.0


def test_contradiction_pulls_degree_down() -> None:
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    ledger = CorrespondenceLedger(model)
    for _ in range(3):
        ledger.check(_prop("Mary is in the bathroom"))
    assert ledger.check(_prop("Mary is in the bathroom")).degree < 0.2


def test_carries_supported() -> None:
    model = SituationModel()
    _say(model, "John took the ball")
    c = _check(model, "John took the ball")
    assert c.verdict is Verdict.SUPPORTED


def test_carries_contradicted_by_other_holder() -> None:
    """An object recorded under another holder is a real
    contradiction — two people cannot carry the same thing."""
    model = SituationModel()
    _say(model, "John took the ball")
    c = _check(model, "Mary took the ball")
    assert c.verdict is Verdict.CONTRADICTED
    assert "john" in c.detail


def test_give_transfers_possession() -> None:
    model = SituationModel()
    _say(model, "John gave the ball to Mary")
    assert _check(model, "Mary took the ball").verdict is Verdict.SUPPORTED


def test_drop_is_a_negated_carries() -> None:
    """'John dropped the ball' claims his hands end up empty —
    supported when the ball is elsewhere."""
    model = SituationModel()
    _say(model, "John took the ball")
    _say(model, "John dropped the ball")
    c = _check(model, "John dropped the ball")
    assert c.verdict is Verdict.SUPPORTED


def test_motion_is_a_location_claim() -> None:
    model = SituationModel()
    _say(model, "John went to the kitchen")
    c = _check(model, "John is in the kitchen")
    assert c.verdict is Verdict.SUPPORTED


def test_attribute_supported() -> None:
    model = SituationModel()
    _say(model, "Mary is hungry")
    c = _check(model, "Mary is hungry")
    assert c.verdict is Verdict.SUPPORTED


def test_positional_relation_supported() -> None:
    model = SituationModel()
    _say(model, "the circle is left of the square")
    c = _check(model, "the circle is left of the square")
    assert c.verdict is Verdict.SUPPORTED


def test_positional_relation_contradicted() -> None:
    model = SituationModel()
    _say(model, "the circle is left of the square")
    c = _check(model, "the circle is right of the square")
    assert c.verdict is Verdict.CONTRADICTED


def test_direction_supported_and_converse() -> None:
    """A direction fact bears on the claim *and* its converse —
    'kitchen north of hallway' also decides 'hallway south of
    kitchen'."""
    model = SituationModel()
    _say(model, "the kitchen is north of the hallway")
    assert _check(model, "the kitchen is north of the hallway").verdict is Verdict.SUPPORTED
    assert _check(model, "the hallway is south of the kitchen").verdict is Verdict.SUPPORTED


def test_size_order_supported() -> None:
    model = SituationModel()
    _say(model, "the box is bigger than the ball")
    assert _check(model, "the box is bigger than the ball").verdict is Verdict.SUPPORTED


def test_size_order_inverted_claim_contradicted() -> None:
    model = SituationModel()
    _say(model, "the box is bigger than the ball")
    assert _check(model, "the ball is bigger than the box").verdict is Verdict.CONTRADICTED


def test_fear_supported() -> None:
    model = SituationModel()
    _say(model, "a mouse is afraid of cats")
    c = _check(model, "a mouse is afraid of cats")
    assert c.verdict is Verdict.SUPPORTED


def test_belief_claim_checks_the_holder_model() -> None:
    """'John believes that Mary is in the kitchen' is about John's
    mind — it corresponds to his stated belief set, not to where
    Mary actually is. 'believe' is a learned verb — a live organism
    acquires it through exposure, so the test registers it first."""
    register_learned_verb("believe")
    model = SituationModel()
    _say(model, "Mary is in the bathroom")
    _say(model, "John believes that Mary is in the kitchen")
    c = _check(model, "John believes that Mary is in the kitchen")
    assert c.verdict is Verdict.SUPPORTED


def test_event_only_claims_have_no_bearing() -> None:
    """A predicate with no settled state residue gets the honest
    verdict — no claim is minted for what the model cannot judge."""
    model = SituationModel()
    _say(model, "John went to the kitchen")
    ledger = CorrespondenceLedger(model)
    c = ledger.check(_prop("the sun shines brightly over the hills"))
    assert c.claim is None or c.verdict is Verdict.UNKNOWN


def test_assess_is_read_only() -> None:
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    ledger = CorrespondenceLedger(model)
    c = ledger.assess(_prop("Mary is in the kitchen"))
    assert c.verdict is Verdict.SUPPORTED
    assert c.evidence == 0.0
    assert ledger.tracked == 0


def test_paraphrases_pool_evidence() -> None:
    """The ledger keys on the claim, not the string — 'john went to
    the kitchen' and 'john is in the kitchen' assert the same state
    of affairs and share one posterior."""
    model = SituationModel()
    _say(model, "John is in the kitchen")
    ledger = CorrespondenceLedger(model)
    ledger.check(_prop("John went to the kitchen"))
    ledger.check(_prop("John is in the kitchen"))
    assert ledger.tracked == 1


def test_ledger_evicts_least_evidenced() -> None:
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    ledger = CorrespondenceLedger(model, max_propositions=2)
    ledger.check(_prop("Mary is in the kitchen"))
    ledger.check(_prop("Mary is in the kitchen"))
    ledger.check(_prop("Mary is in the bathroom"))
    ledger.check(_prop("Mary is in the hallway"))
    assert ledger.tracked == 2


def test_claim_key_canonicalizes() -> None:
    a = _claim_of(_prop("John went to the kitchen"))
    b = _claim_of(_prop("John is in the kitchen"))
    assert a is not None and b is not None and a.key() == b.key()
