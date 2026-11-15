"""Tests for frontal judgment — the assertion act: propositions and
assessments in, stances out."""

from frontal_lobe.reasoning.critical_thinking import (
    AssessmentRecommendation,
    CriticalAssessment,
)
from frontal_lobe.reasoning.judgment import Judge, Stance
from parietal_lobe.correspondence import CorrespondenceLedger
from parietal_lobe.situation_model import SituationModel
from temporal_lobe.comprehension import Comprehension, Proposition

_engine = Comprehension()


def _say(model: SituationModel, text: str) -> None:
    model.update(_engine.comprehend(text))


def _prop(text: str) -> Proposition:
    return _engine.comprehend(text).propositions[0]


def _judge(model: SituationModel, text: str, checks: int = 1):
    judge = Judge(CorrespondenceLedger(model))
    prop = _prop(text)
    out = judge.judge(prop)
    for _ in range(checks - 1):
        out = judge.judge(prop)
    return out


def test_supported_claim_hedges_on_first_evidence() -> None:
    """One confirming check is Beta(1,1) → τ = 2/3 — the judgment
    leans true but doesn't yet assert."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    j = _judge(model, "Mary is in the kitchen")
    assert j.stance is Stance.HEDGE
    assert j.affirms is None
    assert not j.committed


def test_accumulated_support_asserts() -> None:
    """Evidence pooling carries the stance over the assert threshold —
    the second confirming check turns inclination into assertion."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    j = _judge(model, "Mary is in the kitchen", checks=2)
    assert j.stance is Stance.ASSERT
    assert j.affirms is True
    assert j.committed


def test_contradicted_claim_denies_on_evidence() -> None:
    """A claim the world denies is hedged on one check, denied once
    the contradiction repeats — confident negation, not just doubt."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    j = _judge(model, "Mary is in the bathroom")
    assert j.stance is Stance.HEDGE
    j = _judge(model, "Mary is in the bathroom", checks=3)
    assert j.stance is Stance.DENY
    assert j.affirms is False


def test_equivocal_model_suspends() -> None:
    """'Either/or' in the discourse is an equivocal bearing — the
    honest judgment is suspension, not a weak lean."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen or the bathroom")
    j = _judge(model, "Mary is in the kitchen")
    assert j.stance is Stance.SUSPEND
    assert not j.committed


def test_no_bearing_suspends() -> None:
    """A claim about an untracked state of affairs has nothing to
    judge — suspension, not a coin flip."""
    model = SituationModel()
    j = _judge(model, "Mary is in the kitchen")
    assert j.stance is Stance.SUSPEND
    assert j.degree == 0.5
    assert j.evidence == 0.0


def test_without_ledger_everything_suspends() -> None:
    """No world to check against — judgment refuses to pretend."""
    j = Judge().judge(_prop("Mary is in the kitchen"))
    assert j.stance is Stance.SUSPEND


def test_judgment_carries_grounds() -> None:
    """The judgment cites the model state that bore on it — a stance
    without grounds would be assertion without reason."""
    model = SituationModel()
    _say(model, "Mary is in the kitchen")
    j = _judge(model, "Mary is in the kitchen")
    assert j.grounds, "a judged claim must carry what bore on it"
    assert j.content


def _assessment(rec: AssessmentRecommendation, conf: float = 0.8) -> CriticalAssessment:
    return CriticalAssessment(
        original_confidence=conf,
        revised_confidence=conf,
        evidence_quality=0.6,
        recommendation=rec,
        fallacies=["hasty generalization"] if rec is AssessmentRecommendation.REJECT else [],
    )


def test_assessment_stances_map_into_assertoric_force() -> None:
    """The epistemic recommendations render in the same vocabulary:
    accept asserts, hedge hedges, reject denies, investigate suspends."""
    judge = Judge()
    cases = [
        (AssessmentRecommendation.ACCEPT, Stance.ASSERT),
        (AssessmentRecommendation.HEDGE, Stance.HEDGE),
        (AssessmentRecommendation.REJECT, Stance.DENY),
        (AssessmentRecommendation.INVESTIGATE, Stance.SUSPEND),
    ]
    for rec, want in cases:
        j = judge.of_assessment("the claim", _assessment(rec))
        assert j.stance is want, f"{rec} → {j.stance}, wanted {want}"


def test_assessment_grounds_carry_fallacies() -> None:
    """A rejected reasoning result cites what failed it."""
    j = Judge().of_assessment("the claim", _assessment(AssessmentRecommendation.REJECT))
    assert j.stance is Stance.DENY
    assert "hasty generalization" in j.grounds
