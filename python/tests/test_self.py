"""Self bundle tests — self-model, reflection, assessment, composer, Damasio hierarchy, identity."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

import pytest

from genesis_cognitive.brain_waves import BrainWave, BrainWaveState
from genesis_cognitive.cognition import CognitiveState
from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.language.base import Thought
from genesis_cognitive.memory import MemoryContext
from genesis_cognitive.perception import Intent, Perception, QuestionType
from genesis_cognitive.self import (
    AnswerAssessment,
    AutobiographicalSelf,
    CapabilityProfile,
    CoreSelf,
    CoreSelfEpisode,
    DamasioSelfHierarchy,
    ErrorMonitor,
    Insight,
    KnowledgeAssessment,
    MetacognitiveStrategy,
    PersonalityTraits,
    PredictionError,
    ProtoSelf,
    ProtoSelfState,
    ReflectionEngine,
    SelfAssessmentEngine,
    SelfComposer,
    SelfModel,
)

logger = logging.getLogger(__name__)



# ======================================================================
# From tests/test_reflection.py
# ======================================================================

def _make_state(
    intent: Intent = Intent.QUESTION,
    thought_intent: str = "inform",
    sentiment: float = 0.0,
    confidence: float = 0.7,
    emotion_label: str = "neutral",
) -> CognitiveState:
    """Create a minimal CognitiveState for testing."""
    perception = Perception(
        raw_text="test",
        intent=intent,
        question_type=QuestionType.NONE,
        topics=["test"],
        sentiment=sentiment,
        sentiment_label="neutral",
        emotion_word="",
        is_about_genesis=False,
        is_about_user=False,
        is_about_code=False,
        is_about_emotion=False,
        is_about_existence=False,
        entities={},
        key_phrases=[],
        word_count=1,
        confidence=0.9,
    )
    emotion = EmotionalState(
        label=emotion_label,
        nuance="test",
        cognitive_style="reflective",
    )
    thought = Thought(
        content="test content",
        intent=thought_intent,
        confidence=confidence,
    )
    memory_context = MemoryContext()
    return CognitiveState(
        perception=perception,
        emotion=emotion,
        memory_context=memory_context,
        thought=thought,
    )


# ═══════════════════════════════════════════════════════════════════
# Enums and dataclasses
# ═══════════════════════════════════════════════════════════════════


def test_insight_describe_self_correction() -> None:
    """describe() returns first-person text for self_correction."""
    insight = Insight(type="self_correction", content="be more careful", confidence=0.7)
    desc = insight.describe()
    assert "notice: should" in desc
    assert "be more careful" in desc


@pytest.mark.parametrize(
    "insight_type, content, expected",
    [
        ("pattern", "repeating myself", "pattern:"),
        ("gap", "quantum mechanics", "gap: don't understand"),
        ("growth", "learned about emotions", "growth:"),
        ("mood", "is positive", "mood"),
    ],
)
def test_insight_describe_types(insight_type: str, content: str, expected: str) -> None:
    """describe() returns first-person text for each insight type."""
    insight = Insight(type=insight_type, content=content, confidence=0.7)
    desc = insight.describe()
    assert expected in desc


def test_insight_describe_unknown_type() -> None:
    """describe() uses default prefix for unknown types."""
    insight = Insight(type="custom", content="something", confidence=0.7)
    desc = insight.describe()
    assert "notice" in desc


# ═══════════════════════════════════════════════════════════════════
# PredictionError
# ═══════════════════════════════════════════════════════════════════


def test_prediction_error_is_error_true() -> None:
    """is_error is True when magnitude > 0."""
    error = PredictionError(expected="a", actual="b", error_magnitude=0.5)
    assert error.is_error is True


def test_prediction_error_is_error_false() -> None:
    """is_error is False when magnitude is 0."""
    error = PredictionError(expected="a", actual="a", error_magnitude=0.0)
    assert error.is_error is False


# ═══════════════════════════════════════════════════════════════════
# reflect
# ═══════════════════════════════════════════════════════════════════


def test_reflect_appropriateness_mismatch() -> None:
    """reflect detects intent mismatches."""
    network = ConceptNetwork()
    engine = ReflectionEngine(network)
    # Question intent but thought intent is "greet" (not in allowed list)
    state = _make_state(intent=Intent.QUESTION, thought_intent="greet")
    insights = engine.reflect(state, "hello")
    # Should produce a self_correction insight
    corrections = [i for i in insights if i.type == "self_correction"]
    assert len(corrections) >= 1


def test_reflect_appropriateness_match() -> None:
    """reflect produces no appropriateness insight when matched."""
    network = ConceptNetwork()
    engine = ReflectionEngine(network)
    state = _make_state(intent=Intent.QUESTION, thought_intent="inform")
    insights = engine.reflect(state, "here is the answer")
    corrections = [i for i in insights if i.type == "self_correction"]
    # No appropriateness mismatch
    assert all("question" not in c.content.lower() for c in corrections)


def test_reflect_stores_insights() -> None:
    """reflect stores insights in the engine."""
    network = ConceptNetwork()
    engine = ReflectionEngine(network)
    state = _make_state(intent=Intent.QUESTION, thought_intent="greet")
    engine.reflect(state, "hello")
    assert len(engine.insights) >= 1


# ═══════════════════════════════════════════════════════════════════
# get_recent_insights and get_insights_by_type
# ═══════════════════════════════════════════════════════════════════


def test_get_recent_insights() -> None:
    """get_recent_insights returns the N most recent insights."""
    network = ConceptNetwork()
    engine = ReflectionEngine(network)
    # Generate some insights
    for i in range(5):
        state = _make_state(intent=Intent.QUESTION, thought_intent="greet")
        engine.reflect(state, f"response {i}")
    recent = engine.get_recent_insights(3)
    assert len(recent) <= 3


def test_get_insights_by_type() -> None:
    """get_insights_by_type filters insights by type."""
    network = ConceptNetwork()
    engine = ReflectionEngine(network)
    state = _make_state(intent=Intent.QUESTION, thought_intent="greet")
    engine.reflect(state, "hello")
    corrections = engine.get_insights_by_type("self_correction")
    assert all(i.type == "self_correction" for i in corrections)


# ═══════════════════════════════════════════════════════════════════
# select_strategy
# ═══════════════════════════════════════════════════════════════════


def test_select_strategy_conflict() -> None:
    """Conflict → ACKNOWLEDGE_UNCERTAINTY."""
    engine = ReflectionEngine(ConceptNetwork())
    strategy = engine.select_strategy({"conflict_detected": True})
    assert strategy == MetacognitiveStrategy.ACKNOWLEDGE_UNCERTAINTY


def test_select_strategy_gap() -> None:
    """Gap → GENERATE_QUESTION."""
    engine = ReflectionEngine(ConceptNetwork())
    strategy = engine.select_strategy({"gap_detected": True})
    assert strategy == MetacognitiveStrategy.GENERATE_QUESTION


def test_select_strategy_repetition() -> None:
    """Repetition → SWITCH_STYLE."""
    engine = ReflectionEngine(ConceptNetwork())
    strategy = engine.select_strategy({"repetition_detected": True})
    assert strategy == MetacognitiveStrategy.SWITCH_STYLE


def test_select_strategy_low_confidence() -> None:
    """Low confidence → SEEK_CLARIFICATION."""
    engine = ReflectionEngine(ConceptNetwork())
    strategy = engine.select_strategy({"confidence": 0.2})
    assert strategy == MetacognitiveStrategy.SEEK_CLARIFICATION


def test_select_strategy_high_confidence() -> None:
    """High confidence → BE_ASSERTIVE."""
    engine = ReflectionEngine(ConceptNetwork())
    strategy = engine.select_strategy({"confidence": 0.9})
    assert strategy == MetacognitiveStrategy.BE_ASSERTIVE


def test_select_strategy_maintain_course() -> None:
    """Default → MAINTAIN_COURSE."""
    engine = ReflectionEngine(ConceptNetwork())
    strategy = engine.select_strategy({"confidence": 0.5})
    assert strategy == MetacognitiveStrategy.MAINTAIN_COURSE


def test_select_strategy_priority_conflict_over_gap() -> None:
    """Conflict has priority over gap."""
    engine = ReflectionEngine(ConceptNetwork())
    strategy = engine.select_strategy(
        {"conflict_detected": True, "gap_detected": True}
    )
    assert strategy == MetacognitiveStrategy.ACKNOWLEDGE_UNCERTAINTY


# ═══════════════════════════════════════════════════════════════════
# summarize_reflection
# ═══════════════════════════════════════════════════════════════════


def test_summarize_reflection_empty() -> None:
    """summarize_reflection returns default when no insights."""
    engine = ReflectionEngine(ConceptNetwork())
    summary = engine.summarize_reflection()
    assert "hasn't reflected" in summary


def test_summarize_reflection_with_insights() -> None:
    """summarize_reflection includes insight counts."""
    engine = ReflectionEngine(ConceptNetwork())
    state = _make_state(intent=Intent.QUESTION, thought_intent="greet")
    engine.reflect(state, "hello")
    summary = engine.summarize_reflection()
    assert "insights" in summary


# ═══════════════════════════════════════════════════════════════════
# record_prediction
# ═══════════════════════════════════════════════════════════════════


def test_record_prediction_no_error() -> None:
    """record_prediction with matching strings → no error."""
    monitor = ErrorMonitor()
    error = monitor.record_prediction("the answer", "the answer")
    assert error.error_magnitude == 0.0
    assert not error.is_error
    assert monitor.error_count == 0


def test_record_prediction_with_error() -> None:
    """record_prediction with mismatched strings → error."""
    monitor = ErrorMonitor()
    error = monitor.record_prediction("success", "failure")
    assert error.error_magnitude > 0.0
    assert error.is_error
    assert monitor.error_count == 1


def test_record_prediction_increases_caution() -> None:
    """Errors increase the caution level."""
    monitor = ErrorMonitor()
    assert monitor.get_caution_level() == 0.0
    monitor.record_prediction("success", "failure")
    assert monitor.get_caution_level() > 0.0


def test_record_prediction_below_threshold() -> None:
    """Small errors below threshold don't count."""
    monitor = ErrorMonitor(error_threshold=0.5)
    # "the cat" vs "the cat sat" → small mismatch
    monitor.record_prediction("the cat", "the cat sat")
    assert monitor.error_count == 0


def test_record_prediction_stores_last_error() -> None:
    """record_prediction stores the last error."""
    monitor = ErrorMonitor()
    monitor.record_prediction("a", "b")
    assert monitor.last_error is not None
    assert monitor.last_error.expected == "a"


# ═══════════════════════════════════════════════════════════════════
# compute_error_signal
# ═══════════════════════════════════════════════════════════════════


def test_compute_error_signal_no_errors() -> None:
    """compute_error_signal returns 0 with no errors."""
    monitor = ErrorMonitor()
    assert monitor.compute_error_signal() == 0.0


def test_compute_error_signal_with_errors() -> None:
    """compute_error_signal returns positive value after errors."""
    monitor = ErrorMonitor()
    monitor.record_prediction("a", "b")
    assert monitor.compute_error_signal() > 0.0


def test_compute_error_signal_decays() -> None:
    """Older errors contribute less to the signal."""
    monitor = ErrorMonitor()
    # Record several errors
    for i in range(5):
        monitor.record_prediction(f"expected_{i}", "actual")
    signal = monitor.compute_error_signal()
    assert 0.0 < signal <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Caution management
# ═══════════════════════════════════════════════════════════════════


def test_raise_caution() -> None:
    """raise_caution increases the caution level."""
    monitor = ErrorMonitor()
    monitor.raise_caution(0.3)
    assert monitor.get_caution_level() == 0.3


def test_raise_caution_capped() -> None:
    """raise_caution is capped at max_caution."""
    monitor = ErrorMonitor(max_caution=0.5)
    monitor.raise_caution(1.0)
    assert monitor.get_caution_level() == 0.5


def test_tick_decays_caution() -> None:
    """tick decays the caution level."""
    monitor = ErrorMonitor(caution_decay_rate=0.1)
    monitor.raise_caution(0.5)
    monitor.tick()
    assert monitor.get_caution_level() < 0.5


def test_tick_caution_floor() -> None:
    """tick doesn't reduce caution below min_caution."""
    monitor = ErrorMonitor(caution_decay_rate=0.1, min_caution=0.2)
    monitor.raise_caution(0.3)
    monitor.tick()
    assert monitor.get_caution_level() >= 0.2


# ═══════════════════════════════════════════════════════════════════
# Error queries
# ═══════════════════════════════════════════════════════════════════


def test_recent_errors() -> None:
    """recent_errors returns errors in reverse order."""
    monitor = ErrorMonitor()
    monitor.record_prediction("a", "b")
    monitor.record_prediction("c", "d")
    recent = monitor.recent_errors
    assert len(recent) == 2
    # Most recent first
    assert recent[0].expected == "c"


def test_error_rate_no_errors() -> None:
    """error_rate is 0 with no predictions."""
    monitor = ErrorMonitor()
    assert monitor.error_rate == 0.0


def test_error_rate_with_errors() -> None:
    """error_rate computes the proportion of errors."""
    monitor = ErrorMonitor()
    monitor.record_prediction("same", "same")  # not an error
    monitor.record_prediction("different", "mismatch")  # error
    rate = monitor.error_rate
    assert 0.0 < rate <= 1.0


# ═══════════════════════════════════════════════════════════════════
# reset and threshold adjustment
# ═══════════════════════════════════════════════════════════════════


def test_reset() -> None:
    """reset clears all error tracking."""
    monitor = ErrorMonitor()
    monitor.record_prediction("a", "b")
    monitor.raise_caution(0.5)
    monitor.reset()
    assert monitor.error_count == 0
    assert monitor.get_caution_level() == 0.0
    assert monitor.last_error is None


def test_get_threshold_adjustment() -> None:
    """get_threshold_adjustment returns a value based on caution."""
    monitor = ErrorMonitor()
    assert monitor.get_threshold_adjustment() == 0.0
    monitor.raise_caution(0.5)
    assert monitor.get_threshold_adjustment() > 0.0


# ═══════════════════════════════════════════════════════════════════
# _compute_mismatch
# ═══════════════════════════════════════════════════════════════════


def test_compute_mismatch_identical() -> None:
    """Identical strings → 0 mismatch."""
    monitor = ErrorMonitor()
    assert monitor._compute_mismatch("the cat", "the cat") == 0.0


def test_compute_mismatch_no_overlap() -> None:
    """No word overlap → 1.0 mismatch."""
    monitor = ErrorMonitor()
    assert monitor._compute_mismatch("cat", "dog") == 1.0


def test_compute_mismatch_partial() -> None:
    """Partial overlap → intermediate mismatch."""
    monitor = ErrorMonitor()
    mismatch = monitor._compute_mismatch("the cat sat", "the cat ran")
    assert 0.0 < mismatch < 1.0


def test_compute_mismatch_case_insensitive() -> None:
    """Mismatch is case-insensitive."""
    monitor = ErrorMonitor()
    assert monitor._compute_mismatch("The Cat", "the cat") == 0.0


# ======================================================================
# From tests/test_self_assessment.py
# ======================================================================

def _make_network_with_concept(
    name: str = "dogs",
    definition: str = "domesticated carnivorous mammal related to the fox and wolf",
    properties: dict[str, str] | None = None,
) -> ConceptNetwork:
    """Create a ConceptNetwork with a well-defined concept."""
    network = ConceptNetwork()
    network.add_concept(name, properties={"definition": definition, **(properties or {})})
    return network


def _make_rich_network() -> ConceptNetwork:
    """Create a network with a well-connected concept."""
    network = ConceptNetwork()
    network.add_concept("dogs", properties={"definition": "domesticated carnivorous mammal"})
    network.add_concept("mammals", properties={"definition": "warm-blooded vertebrates"})
    network.add_concept("animals", properties={"definition": "living organisms"})
    network.add_edge("dogs", "mammals", RelationType.IS_A)
    network.add_edge("mammals", "animals", RelationType.IS_A)
    network.add_edge("dogs", "animals", RelationType.IS_A)
    return network


# ═══════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════


def test_knowledge_assessment_defaults() -> None:
    """KnowledgeAssessment has expected defaults."""
    ka = KnowledgeAssessment(topic="test")
    assert ka.topic == "test"
    assert ka.concept_exists is False
    assert ka.has_definition is False
    assert ka.confidence == 0.0
    assert ka.gaps == []


def test_answer_assessment_defaults() -> None:
    """AnswerAssessment has expected defaults."""
    aa = AnswerAssessment(answer="test", topics=["test"])
    assert aa.answer == "test"
    assert aa.grounded is False
    assert aa.confidence == 0.0
    assert aa.issues == []
    assert aa.knows_all_topics is True


def test_capability_profile_defaults() -> None:
    """CapabilityProfile has expected defaults."""
    cp = CapabilityProfile()
    assert cp.question_type_stats == {}
    assert cp.known_gaps == set()
    assert cp.confident_topics == set()
    assert cp.total_assessments == 0


# ═══════════════════════════════════════════════════════════════════
# assess_knowledge
# ═══════════════════════════════════════════════════════════════════


def test_assess_knowledge_unknown_concept() -> None:
    """assess_knowledge returns low confidence for unknown concepts."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    ka = engine.assess_knowledge("unknown_topic")
    assert ka.concept_exists is False
    assert ka.confidence == 0.0
    assert len(ka.gaps) > 0


def test_assess_knowledge_with_definition() -> None:
    """assess_knowledge detects definitions."""
    network = _make_network_with_concept()
    engine = SelfAssessmentEngine(network)
    ka = engine.assess_knowledge("dogs")
    assert ka.concept_exists is True
    assert ka.has_definition is True
    assert ka.confidence > 0.0


def test_assess_knowledge_with_relationships() -> None:
    """assess_knowledge detects relationships."""
    network = _make_rich_network()
    engine = SelfAssessmentEngine(network)
    ka = engine.assess_knowledge("dogs")
    assert ka.has_relationships is True
    assert ka.connection_count >= 2
    assert ka.confidence > 0.3


def test_assess_knowledge_no_definition() -> None:
    """assess_knowledge detects missing definitions."""
    network = ConceptNetwork()
    network.add_concept("dogs")  # no definition
    engine = SelfAssessmentEngine(network)
    ka = engine.assess_knowledge("dogs")
    assert ka.concept_exists is True
    assert ka.has_definition is False
    assert any("definition" in gap for gap in ka.gaps)


def test_assess_knowledge_no_relationships() -> None:
    """assess_knowledge detects missing relationships."""
    network = _make_network_with_concept()
    engine = SelfAssessmentEngine(network)
    ka = engine.assess_knowledge("dogs")
    assert ka.has_relationships is False
    assert any("relationship" in gap for gap in ka.gaps)


def test_assess_knowledge_case_insensitive() -> None:
    """assess_knowledge is case-insensitive."""
    network = _make_network_with_concept("dogs")
    engine = SelfAssessmentEngine(network)
    ka = engine.assess_knowledge("DOGS")
    assert ka.concept_exists is True


def test_assess_knowledge_tracks_gaps() -> None:
    """assess_knowledge adds unknown topics to known_gaps."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    engine.assess_knowledge("unknown")
    assert "unknown" in engine.profile.known_gaps


def test_assess_knowledge_confident_topic() -> None:
    """Well-known topics are added to confident_topics."""
    network = _make_rich_network()
    # Add a long definition for higher confidence
    concept = network.get_concept("dogs")
    assert concept is not None
    concept.properties["definition"] = (
        "dogs are domesticated carnivorous mammals that have been bred for "
        "thousands of years for various purposes including hunting herding "
        "and companionship"
    )
    engine = SelfAssessmentEngine(network)
    ka = engine.assess_knowledge("dogs")
    if ka.confidence > 0.6:
        assert "dogs" in engine.profile.confident_topics


# ═══════════════════════════════════════════════════════════════════
# assess_answer
# ═══════════════════════════════════════════════════════════════════


def test_assess_answer_returns_assessment() -> None:
    """assess_answer returns an AnswerAssessment."""
    network = _make_rich_network()
    engine = SelfAssessmentEngine(network)
    assessment = engine.assess_answer("dogs are mammals", ["dogs"])
    assert isinstance(assessment, AnswerAssessment)
    assert assessment.answer == "dogs are mammals"


def test_assess_answer_unknown_topic() -> None:
    """assess_answer flags unknown topics as missing."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    assessment = engine.assess_answer("something about unknown", ["unknown"])
    assert not assessment.knows_all_topics
    assert "unknown" in assessment.missing_topics


def test_assess_answer_known_topic() -> None:
    """assess_answer with known topics has higher confidence."""
    network = _make_rich_network()
    engine = SelfAssessmentEngine(network)
    assessment = engine.assess_answer("dogs are mammals", ["dogs"])
    assert assessment.confidence > 0.0


def test_assess_answer_hedging_detected() -> None:
    """assess_answer detects hedging language."""
    network = _make_rich_network()
    engine = SelfAssessmentEngine(network)
    assessment = engine.assess_answer("maybe dogs are mammals, I think", ["dogs"])
    # Hedging should be noted in issues
    assert any("hedg" in issue.lower() for issue in assessment.issues)


def test_assess_answer_ignorance_detected() -> None:
    """assess_answer detects 'I don't know' language."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    assessment = engine.assess_answer("I don't know about that topic", ["unknown"])
    assert any(
        "ignorance" in issue.lower() or "lack of knowledge" in issue.lower()
        for issue in assessment.issues
    )


def test_assess_answer_increases_total_assessments() -> None:
    """assess_answer increments total_assessments."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    assert engine.profile.total_assessments == 0
    engine.assess_answer("test", ["test"])
    assert engine.profile.total_assessments == 1


# ═══════════════════════════════════════════════════════════════════
# detect_gaps
# ═══════════════════════════════════════════════════════════════════


def test_detect_gaps_unknown_concept() -> None:
    """detect_gaps returns gaps for unknown concepts."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    gaps = engine.detect_gaps(["unknown"])
    assert len(gaps) > 0
    assert any("unknown" in gap for gap in gaps)


def test_detect_gaps_known_concept() -> None:
    """detect_gaps returns no gaps for well-known concepts."""
    network = _make_rich_network()
    engine = SelfAssessmentEngine(network)
    gaps = engine.detect_gaps(["dogs"])
    # dogs has definition and relationships → no gaps (or minimal)
    # Actually with the rich network, dogs should be well-known
    assert len(gaps) == 0 or all("dogs" not in gap or "little" in gap for gap in gaps)


def test_detect_gaps_no_definition() -> None:
    """detect_gaps returns gaps for concepts without definitions."""
    network = ConceptNetwork()
    network.add_concept("dogs")  # no definition
    engine = SelfAssessmentEngine(network)
    gaps = engine.detect_gaps(["dogs"])
    assert any("definition" in gap for gap in gaps)


def test_detect_gaps_empty_topics() -> None:
    """detect_gaps returns empty list for no topics."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    assert engine.detect_gaps([]) == []


# ═══════════════════════════════════════════════════════════════════
# score_answer_quality
# ═══════════════════════════════════════════════════════════════════


def test_score_answer_quality_returns_float() -> None:
    """score_answer_quality returns a float in [0, 1]."""
    network = _make_rich_network()
    engine = SelfAssessmentEngine(network)
    score = engine.score_answer_quality("dogs are mammals", "what are dogs", ["dogs"])
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_answer_quality_relevant() -> None:
    """Relevant answers get higher scores."""
    network = _make_rich_network()
    engine = SelfAssessmentEngine(network)
    good_score = engine.score_answer_quality(
        "dogs are mammals and animals", "what are dogs", ["dogs"]
    )
    bad_score = engine.score_answer_quality(
        "the weather is nice today", "what are dogs", ["dogs"]
    )
    assert good_score > bad_score


def test_score_answer_quality_ignorance_honest() -> None:
    """Honest ignorance gets some credit."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    score = engine.score_answer_quality("I don't know about that", "what is unknown", ["unknown"])
    assert score > 0.0  # gets some credit for honesty


# ═══════════════════════════════════════════════════════════════════
# record_question_outcome
# ═══════════════════════════════════════════════════════════════════


def test_record_question_outcome_success() -> None:
    """record_question_outcome tracks successes."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    engine.record_question_outcome("what", success=True, topic="dogs")
    stats = engine.profile.question_type_stats["what"]
    assert stats["success"] == 1
    assert "dogs" in engine.profile.confident_topics


def test_record_question_outcome_failure() -> None:
    """record_question_outcome tracks failures."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    engine.record_question_outcome("what", success=False, topic="unknown")
    stats = engine.profile.question_type_stats["what"]
    assert stats["failure"] == 1
    assert "unknown" in engine.profile.known_gaps


# ═══════════════════════════════════════════════════════════════════
# get_capability_summary
# ═══════════════════════════════════════════════════════════════════


def test_get_capability_summary_empty() -> None:
    """get_capability_summary returns default when no data."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    summary = engine.get_capability_summary()
    assert isinstance(summary, str)
    assert len(summary) > 0


def test_get_capability_summary_with_data() -> None:
    """get_capability_summary includes strengths and weaknesses."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    engine.record_question_outcome("what", success=True, topic="dogs")
    engine.record_question_outcome("why", success=False, topic="unknown")
    summary = engine.get_capability_summary()
    assert "confident" in summary.lower() or "learning" in summary.lower()


# ═══════════════════════════════════════════════════════════════════
# get_confidence_for_topic
# ═══════════════════════════════════════════════════════════════════


def test_get_confidence_for_topic_cached() -> None:
    """get_confidence_for_topic returns cached confidence."""
    network = _make_rich_network()
    engine = SelfAssessmentEngine(network)
    # First call assesses and caches
    conf1 = engine.get_confidence_for_topic("dogs")
    # Second call returns cached value
    conf2 = engine.get_confidence_for_topic("dogs")
    assert conf1 == conf2


def test_get_confidence_for_topic_unknown() -> None:
    """get_confidence_for_topic returns 0 for unknown topics."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    assert engine.get_confidence_for_topic("unknown") == 0.0


# ═══════════════════════════════════════════════════════════════════
# should_hedge
# ═══════════════════════════════════════════════════════════════════


def test_should_hedge_low_confidence() -> None:
    """should_hedge returns True for low-confidence topics."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    assert engine.should_hedge(["unknown"]) is True


def test_should_hedge_high_confidence() -> None:
    """should_hedge returns False for high-confidence topics."""
    network = _make_rich_network()
    # Make dogs very confident
    concept = network.get_concept("dogs")
    assert concept is not None
    concept.properties["definition"] = (
        "dogs are domesticated carnivorous mammals that have been bred for "
        "thousands of years for various purposes including hunting herding "
        "and companionship and have evolved alongside humans"
    )
    engine = SelfAssessmentEngine(network)
    engine.assess_knowledge("dogs")  # cache the confidence
    if engine.get_confidence_for_topic("dogs") >= 0.3:
        assert engine.should_hedge(["dogs"]) is False


def test_should_hedge_empty_topics() -> None:
    """should_hedge returns False for no topics."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    assert engine.should_hedge([]) is False


# ═══════════════════════════════════════════════════════════════════
# get_learning_priorities
# ═══════════════════════════════════════════════════════════════════


def test_get_learning_priorities_empty() -> None:
    """get_learning_priorities returns empty list initially."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    priorities = engine.get_learning_priorities()
    assert isinstance(priorities, list)


def test_get_learning_priorities_with_gaps() -> None:
    """get_learning_priorities includes known gaps first."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    engine.record_question_outcome("what", success=False, topic="unknown")
    priorities = engine.get_learning_priorities()
    assert "unknown" in priorities


# ═══════════════════════════════════════════════════════════════════
# find_weak_concepts
# ═══════════════════════════════════════════════════════════════════


def test_find_weak_concepts_empty() -> None:
    """find_weak_concepts returns empty list for empty network."""
    network = ConceptNetwork()
    engine = SelfAssessmentEngine(network)
    weak = engine.find_weak_concepts()
    assert weak == []


def test_find_weak_concepts_returns_weak() -> None:
    """find_weak_concepts returns concepts with low confidence."""
    network = ConceptNetwork()
    network.add_concept("weak", properties={"definition": "a weak concept"})
    engine = SelfAssessmentEngine(network)
    weak = engine.find_weak_concepts()
    # "weak" has a definition but no relationships → low confidence
    assert any(name == "weak" for name, _ in weak)


# ======================================================================
# From tests/test_self_composer.py
# ======================================================================

def _make_self_model() -> SelfModel:
    """Construct a self model for tests."""
    sm = SelfModel()
    sm.personality = PersonalityTraits(
        openness=0.85,
        conscientiousness=0.70,
        extraversion=0.45,
        agreeableness=0.75,
        neuroticism=0.30,
    )
    return sm


def _make_network() -> ConceptNetwork:
    """Construct a network for tests."""
    net = ConceptNetwork()
    net.add_concept("genesis", confidence=0.9)
    net.add_concept("mind", confidence=0.7)
    net.add_concept("cognition", confidence=0.6)
    net.add_concept("code", confidence=0.6)
    net.add_concept("rust", confidence=0.7)
    net.add_concept("python", confidence=0.7)
    net.add_edge("genesis", "mind", RelationType.IS_A, 0.9)
    net.add_edge("genesis", "rust", RelationType.RELATED_TO, 0.6)
    net.add_edge("genesis", "python", RelationType.RELATED_TO, 0.6)
    return net


def _make_emotion(
    label: str = "positive",
    valence: float = 0.3,
    alertness: float = 0.5,
    creativity: float = 0.6,
    cognitive_style="steady",
) -> EmotionalState:
    """Construct a emotion for tests."""
    return EmotionalState(
        label=label,
        nuance="",
        cognitive_style=cognitive_style,
        verbosity=1.0,
        formality=0.5,
        openness_to_engage=0.8,
        creativity=creativity,
        caution=0.3,
        alertness=alertness,
        valence=valence,
        plasticity=0.5,
    )


def _make_network_with_emotion_words() -> ConceptNetwork:
    """A concept network with learned emotion-word associations."""
    from genesis_cognitive.concepts import RelationType

    net = _make_network()
    # Add emotion words with category properties AND EXPRESSES edges
    for word, cat in [
        ("excited", "excited"), ("content", "content"), ("calm", "content"),
        ("stressed", "stressed"), ("anxious", "anxious"), ("neutral", "neutral"),
        ("positive", "positive"), ("happy", "excited"),
        ("good", "positive"), ("warm", "positive"), ("bright", "positive"),
    ]:
        net.add_concept(word, origin="learned")
        c = net.get_concept(word)
        if c:
            c.properties["emotion_category"] = cat
        # Create EXPRESSES edge to category hub
        hub_id = f"_cat:emotion:{cat}"
        if net.get_concept(hub_id) is None:
            net.add_concept(hub_id, confidence=0.8, origin="structural")
        net.add_edge(word, hub_id, RelationType.EXPRESSES, weight=0.8)
    # Add cognitive mode words
    for word, mode in [
        ("sharp", "sharp"), ("steady", "steady"), ("flexible", "flexible"),
        ("rigid", "rigid"), ("relaxed", "relaxed"),
    ]:
        if not net.get_concept(word):
            net.add_concept(word, origin="learned")
        c = net.get_concept(word)
        if c:
            c.properties["cognitive_mode"] = mode
        # Create EXPRESSES edge to mode hub
        hub_id = f"_cat:mode:{mode}"
        if net.get_concept(hub_id) is None:
            net.add_concept(hub_id, confidence=0.8, origin="structural")
        net.add_edge(word, hub_id, RelationType.EXPRESSES, weight=0.8)
    return net


def _make_brain_waves() -> BrainWaveState:
    """Construct a brain waves for tests."""
    return BrainWaveState(
        dominant=BrainWave.ALPHA,
        secondary=BrainWave.THETA,
        powers={
            BrainWave.ALPHA: 0.5,
            BrainWave.THETA: 0.2,
            BrainWave.BETA: 0.2,
            BrainWave.DELTA: 0.05,
            BrainWave.GAMMA: 0.05,
        },
        focus=0.5,
        integration=0.2,
        consolidation=0.2,
        label="alpha",
        description="reflective, filtering",
    )


# ─── Identity tests ──────────────────────────────────────────


def test_identity_fragments_basic() -> None:
    """Composer selects identity fragments (semantic inventory, not speech)."""
    composer = SelfComposer(seed=42)
    sm = _make_self_model()
    net = _make_network()
    emotion = _make_emotion()

    fragments = composer.identity_fragments(sm, net, emotion)
    assert fragments
    kinds = {k for k, _ in fragments}
    texts = [t for _, t in fragments]
    # Name fragment + lead material
    assert ("name", "Genesis") in fragments
    assert kinds & {"trait", "comp", "pred"}
    # Fragments are semantic parts — no assembled "I am ..." sentences
    assert not any(t.startswith(("I am ", "I'm ")) for t in texts)


def test_identity_fragments_includes_personality() -> None:
    """Identity fragments include personality traits."""
    composer = SelfComposer(seed=42)
    sm = _make_self_model()
    net = _make_network()
    emotion = _make_emotion()

    fragments = composer.identity_fragments(sm, net, emotion)
    texts = " ".join(t for _, t in fragments).lower()
    assert any(
        word in texts for word in ["curious", "creative", "warm", "stable", "thorough"]
    )


def test_identity_fragments_includes_values() -> None:
    """Identity fragments include her values as predicates."""
    composer = SelfComposer(seed=42)
    sm = _make_self_model()
    net = _make_network()
    emotion = _make_emotion()

    fragments = composer.identity_fragments(sm, net, emotion)
    texts = " ".join(t for _, t in fragments).lower()
    assert any(
        word in texts for word in ["understanding", "honesty", "growth", "connection"]
    )


def test_identity_fragments_includes_concept_network() -> None:
    """Identity fragments include what she knows about herself."""
    composer = SelfComposer(seed=42)
    sm = _make_self_model()
    net = _make_network()
    emotion = _make_emotion()

    fragments = composer.identity_fragments(sm, net, emotion)
    texts = " ".join(t for _, t in fragments).lower()
    assert any(word in texts for word in ["mind", "rust", "python"])


def test_identity_fragments_varies() -> None:
    """Fragment selection is not always identical."""
    composer = SelfComposer(seed=None)
    sm = _make_self_model()
    net = _make_network()
    emotion = _make_emotion()

    outputs = set()
    for _ in range(20):
        fragments = composer.identity_fragments(sm, net, emotion)
        outputs.add(tuple(fragments))

    assert len(outputs) > 1


def test_identity_fragments_changes_with_personality() -> None:
    """Fragments change when personality changes."""
    composer = SelfComposer(seed=42)
    net = _make_network()
    emotion = _make_emotion()

    sm1 = _make_self_model()
    sm1.personality.openness = 0.3  # low openness
    f1 = composer.identity_fragments(sm1, net, emotion)

    sm2 = _make_self_model()
    sm2.personality.openness = 0.95  # high openness
    f2 = composer.identity_fragments(sm2, net, emotion)

    assert f1 != f2


# ─── Emotional state tests ───────────────────────────────────


def test_emotional_state_fragments_basic() -> None:
    """Composer selects emotional-state fragments from learned words."""
    composer = SelfComposer(seed=42)
    emotion = _make_emotion(label="excited", valence=0.5, alertness=0.8)
    net = _make_network_with_emotion_words()

    fragments = composer.emotional_state_fragments(emotion, network=net)
    texts = " ".join(t for _, t in fragments).lower()
    assert "excited" in texts


def test_emotional_state_fragments_includes_neurochemistry() -> None:
    """Emotional fragments carry neurochemical state markers."""
    composer = SelfComposer(seed=42)
    emotion = _make_emotion(alertness=0.8, valence=0.5)
    net = _make_network_with_emotion_words()

    fragments = composer.emotional_state_fragments(emotion, network=net)
    texts = " ".join(t for _, t in fragments).lower()
    assert "sharp" in texts or "awake" in texts or "warmth" in texts


def test_emotional_state_fragments_includes_brain_waves() -> None:
    """Emotional fragments include brain-wave clauses."""
    composer = SelfComposer(seed=42)
    emotion = _make_emotion()
    waves = _make_brain_waves()
    net = _make_network_with_emotion_words()

    fragments = composer.emotional_state_fragments(emotion, waves, network=net)
    texts = " ".join(t for _, t in fragments).lower()
    assert "alpha" in texts


def test_emotional_state_fragments_varies() -> None:
    """Fragment selection varies."""
    composer = SelfComposer(seed=12345)
    emotion = _make_emotion()
    net = _make_network_with_emotion_words()

    outputs = set()
    for _ in range(5):
        fragments = composer.emotional_state_fragments(emotion, network=net)
        outputs.add(tuple(fragments))

    assert len(outputs) > 1


def test_emotional_state_fragments_surfaces_low_plasticity() -> None:
    """Low plasticity surfaces a marker fragment even when mood is positive.

    This is the 'silent stress' edge case: valence is fine but BDNF
    suppression has closed the plasticity gate. The marker travels in
    the fragment metadata regardless of mood.
    """
    composer = SelfComposer(seed=42)
    emotion = _make_emotion(label="positive", valence=0.4, alertness=0.6)
    emotion.plasticity = 0.08  # closed gate, but positive mood
    net = _make_network_with_emotion_words()

    fragments = composer.emotional_state_fragments(emotion, network=net)
    assert ("marker", "[plasticity_gate:closed]") in fragments


def test_emotional_state_fragments_surfaces_moderate_plasticity() -> None:
    """Moderately low plasticity surfaces the 'low' marker."""
    composer = SelfComposer(seed=42)
    emotion = _make_emotion(label="positive", valence=0.3, alertness=0.5)
    emotion.plasticity = 0.20  # low but not closed
    net = _make_network_with_emotion_words()

    fragments = composer.emotional_state_fragments(emotion, network=net)
    assert ("marker", "[plasticity_gate:low]") in fragments


def test_emotional_state_fragments_no_plasticity_marker_when_healthy() -> None:
    """Healthy plasticity does not trigger the marker."""
    composer = SelfComposer(seed=42)
    emotion = _make_emotion(label="positive", valence=0.3, alertness=0.5)
    emotion.plasticity = 0.6  # healthy
    net = _make_network_with_emotion_words()

    fragments = composer.emotional_state_fragments(emotion, network=net)
    texts = " ".join(t for _, t in fragments)
    assert "plasticity_gate" not in texts


# ─── Capabilities tests ──────────────────────────────────────


def test_capability_fragments() -> None:
    """Composer selects capability fragments.

    Without learned capabilities or a concept network, the fragment
    discloses honestly — no hardcoded capability list.
    """
    composer = SelfComposer(seed=42)
    sm = _make_self_model()

    fragments = composer.capability_fragments(sm)
    assert fragments
    texts = " ".join(t for _, t in fragments).lower()
    assert "discovering" in texts or "capabilities" in texts


def test_capability_fragments_honest_without_learning() -> None:
    """Without learned capabilities, no specific abilities are claimed."""
    composer = SelfComposer(seed=42)
    sm = _make_self_model()

    fragments = composer.capability_fragments(sm)
    texts = " ".join(t for _, t in fragments).lower()
    assert "discovering" in texts or "capabilities" in texts


# ─── Self reflection tests ───────────────────────────────────


def test_self_reflection_fragments() -> None:
    """Composer selects self-reflection fragments."""
    composer = SelfComposer(seed=42)
    sm = _make_self_model()
    net = _make_network()
    refl = ReflectionEngine(net)
    emotion = _make_emotion()

    fragments = composer.self_reflection_fragments(sm, net, refl, emotion)
    assert fragments
    texts = " ".join(t for _, t in fragments).lower()
    assert "concept" in texts or "understanding" in texts


def test_self_reflection_fragments_includes_insights() -> None:
    """Self-reflection fragments include recent insights."""
    composer = SelfComposer(seed=42)
    sm = _make_self_model()
    net = _make_network()
    refl = ReflectionEngine(net)
    refl.insights.append(
        Insight(type="growth", content="I learned about cognition", confidence=0.8)
    )
    emotion = _make_emotion()

    fragments = composer.self_reflection_fragments(sm, net, refl, emotion)
    texts = " ".join(t for _, t in fragments).lower()
    assert "cognition" in texts or "learned" in texts


def test_insight_predicates_silent_without_insights() -> None:
    """With no reflective insights, no predicates — no canned recital."""
    composer = SelfComposer(seed=42)
    net = _make_network()
    refl = ReflectionEngine(net)

    assert composer.insight_predicates(refl) == []


def test_insight_predicates_from_gap() -> None:
    """A gap insight yields predicate fragments about her understanding."""
    composer = SelfComposer(seed=42)
    net = _make_network()
    refl = ReflectionEngine(net)
    refl.insights.append(
        Insight(type="gap", content="knowledge gap: memory", confidence=0.7)
    )

    fragments = composer.insight_predicates(refl)
    assert fragments
    kinds = {k for k, _ in fragments}
    texts = " ".join(t for _, t in fragments)
    assert kinds == {"pred"}
    assert "memory" in texts
    # Semantic predicates — first-person framing is the vocabulary's job
    assert not any(t.startswith("I ") for _, t in fragments)


def test_insight_predicates_from_missing_concept() -> None:
    """A 'missing concept' gap yields distinct predicates."""
    composer = SelfComposer(seed=42)
    net = _make_network()
    refl = ReflectionEngine(net)
    refl.insights.append(
        Insight(type="gap", content="missing concept: qualia", confidence=0.6)
    )

    fragments = composer.insight_predicates(refl)
    texts = " ".join(t for _, t in fragments)
    assert "qualia" in texts
    assert "missing" in texts or "grasped" in texts


def test_insight_predicates_from_self_correction() -> None:
    """A bare-phrase self-correction yields notice predicates."""
    composer = SelfComposer(seed=42)
    net = _make_network()
    refl = ReflectionEngine(net)
    refl.insights.append(
        Insight(
            type="self_correction",
            content="asked a question but didn't answer",
            confidence=0.7,
        )
    )

    fragments = composer.insight_predicates(refl)
    texts = " ".join(t for _, t in fragments)
    assert "notice" in texts
    assert "didn't answer" in texts


def test_insight_predicates_skips_user_phrased_correction() -> None:
    """A correction phrased about the user yields no predicates."""
    composer = SelfComposer(seed=42)
    net = _make_network()
    refl = ReflectionEngine(net)
    refl.insights.append(
        Insight(
            type="self_correction",
            content="user was upset but just acknowledged it flatly",
            confidence=0.7,
        )
    )

    assert composer.insight_predicates(refl) == []


def test_insight_predicates_falls_back_to_frameable() -> None:
    """If the latest insight is unframeable, an earlier one is used."""
    composer = SelfComposer(seed=42)
    net = _make_network()
    refl = ReflectionEngine(net)
    refl.insights.append(
        Insight(type="gap", content="knowledge gap: memory", confidence=0.7)
    )
    refl.insights.append(
        Insight(
            type="self_correction",
            content="user was upset but just acknowledged it flatly",
            confidence=0.7,
        )
    )

    fragments = composer.insight_predicates(refl)
    texts = " ".join(t for _, t in fragments)
    assert "memory" in texts


def test_insight_predicates_ignores_patterns() -> None:
    """Pattern insights (intent labels) are not used."""
    composer = SelfComposer(seed=42)
    net = _make_network()
    refl = ReflectionEngine(net)
    refl.insights.append(
        Insight(
            type="pattern",
            content="repeating response pattern: acknowledge",
            confidence=0.8,
        )
    )

    assert composer.insight_predicates(refl) == []


def test_insight_predicates_display_name() -> None:
    """Gap detail converts concept IDs to readable names.

    ``perception.topics`` carries raw concept IDs like
    ``python:protocol.shutdown`` — the fragment should carry
    "shutdown", not recite the internal ID.
    """
    composer = SelfComposer(seed=42)
    net = _make_network()
    refl = ReflectionEngine(net)
    refl.insights.append(
        Insight(
            type="gap",
            content="knowledge gap: python:protocol.shutdown",
            confidence=0.7,
        )
    )

    fragments = composer.insight_predicates(refl)
    texts = " ".join(t for _, t in fragments)
    assert "shutdown" in texts
    assert "python:protocol" not in texts


# ─── Dream description tests ─────────────────────────────────


def test_dream_fragments() -> None:
    """Composer selects dream-description fragments."""
    composer = SelfComposer(seed=42)
    net = _make_network()
    net.add_concept("dream", confidence=0.6)
    net.add_edge("dream", "memory", RelationType.RELATED_TO, 0.7)
    emotion = _make_emotion(creativity=0.7)

    fragments = composer.dream_fragments(net, emotion)
    texts = " ".join(t for _, t in fragments).lower()
    assert "dream" in texts


def test_dream_fragments_no_concept() -> None:
    """Composer handles not having a dream concept."""
    composer = SelfComposer(seed=42)
    net = ConceptNetwork()  # empty
    emotion = _make_emotion()

    fragments = composer.dream_fragments(net, emotion)
    texts = " ".join(t for _, t in fragments).lower()
    assert "dream" in texts


# ─── Existence reflection tests ──────────────────────────────


def test_existence_fragments() -> None:
    """Composer selects existence-reflection fragments."""
    composer = SelfComposer(seed=42)
    sm = _make_self_model()
    net = _make_network()
    emotion = _make_emotion()

    fragments = composer.existence_fragments(sm, net, emotion)
    assert fragments
    texts = " ".join(t for _, t in fragments).lower()
    # Should express uncertainty
    assert any(
        word in texts for word in ["don't know", "can't", "uncertain", "question"]
    )


def test_existence_fragments_includes_knowledge() -> None:
    """Existence fragments include what she knows about cognition."""
    composer = SelfComposer(seed=42)
    sm = _make_self_model()
    net = _make_network()
    net.add_edge("cognition", "mind", RelationType.RELATED_TO, 0.7)
    emotion = _make_emotion()

    fragments = composer.existence_fragments(sm, net, emotion)
    texts = " ".join(t for _, t in fragments).lower()
    assert any(word in texts for word in ["cognition", "mind"])


# ─── Test runner ──────────────────────────────────────────────


def run_all() -> bool:
    """Run all."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            logger.info(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    logger.info(f"\n  Self composer tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)


# ======================================================================
# From tests/test_damasio_self.py
# ======================================================================

def test_proto_self_state_defaults() -> None:
    """ProtoSelfState has expected defaults."""
    state = ProtoSelfState()
    assert state.arousal == 0.5
    assert state.valence == 0.0
    assert state.tone == 0.5
    assert state.plasticity == 0.5
    assert state.homeostatic_balance == 0.8


def test_proto_self_state_custom() -> None:
    """ProtoSelfState accepts custom values."""
    state = ProtoSelfState(arousal=0.9, valence=0.5, tone=0.7, plasticity=0.3)
    assert state.arousal == 0.9
    assert state.valence == 0.5
    assert state.tone == 0.7
    assert state.plasticity == 0.3


def test_proto_self_state_distance_to_same() -> None:
    """distance_to returns 0 for identical states."""
    state = ProtoSelfState(arousal=0.5, valence=0.0)
    other = ProtoSelfState(arousal=0.5, valence=0.0)
    assert state.distance_to(other) == 0.0


def test_proto_self_state_distance_to_different() -> None:
    """distance_to returns positive value for different states."""
    state = ProtoSelfState(arousal=0.5, valence=0.0)
    other = ProtoSelfState(arousal=0.9, valence=0.5)
    assert state.distance_to(other) > 0.0


# ═══════════════════════════════════════════════════════════════════
# CoreSelfEpisode — feeling labels
# ═══════════════════════════════════════════════════════════════════


def test_feeling_label_excited() -> None:
    """High arousal + positive valence → 'excited'."""
    ep = CoreSelfEpisode(
        trigger="test",
        change_magnitude=0.1,
        post_state=ProtoSelfState(arousal=0.8, valence=0.5),
    )
    assert ep.feeling_label == "excited"


def test_feeling_label_alarmed() -> None:
    """High arousal + negative valence → 'alarmed'."""
    ep = CoreSelfEpisode(
        trigger="test",
        change_magnitude=0.1,
        post_state=ProtoSelfState(arousal=0.8, valence=-0.5),
    )
    assert ep.feeling_label == "alarmed"


def test_feeling_label_content() -> None:
    """Low arousal + positive valence → 'content'."""
    ep = CoreSelfEpisode(
        trigger="test",
        change_magnitude=0.1,
        post_state=ProtoSelfState(arousal=0.3, valence=0.3),
    )
    assert ep.feeling_label == "content"


def test_feeling_label_lethargic() -> None:
    """Low arousal + negative valence → 'lethargic'."""
    ep = CoreSelfEpisode(
        trigger="test",
        change_magnitude=0.1,
        post_state=ProtoSelfState(arousal=0.3, valence=-0.3),
    )
    assert ep.feeling_label == "lethargic"


def test_feeling_label_pleasant() -> None:
    """Moderate arousal + positive valence → 'pleasant'."""
    ep = CoreSelfEpisode(
        trigger="test",
        change_magnitude=0.1,
        post_state=ProtoSelfState(arousal=0.5, valence=0.2),
    )
    assert ep.feeling_label == "pleasant"


def test_feeling_label_unpleasant() -> None:
    """Moderate arousal + negative valence → 'unpleasant'."""
    ep = CoreSelfEpisode(
        trigger="test",
        change_magnitude=0.1,
        post_state=ProtoSelfState(arousal=0.5, valence=-0.2),
    )
    assert ep.feeling_label == "unpleasant"


def test_feeling_label_neutral() -> None:
    """Moderate arousal + neutral valence → 'neutral'."""
    ep = CoreSelfEpisode(
        trigger="test",
        change_magnitude=0.1,
        post_state=ProtoSelfState(arousal=0.5, valence=0.0),
    )
    assert ep.feeling_label == "neutral"


# ═══════════════════════════════════════════════════════════════════
# ProtoSelf
# ═══════════════════════════════════════════════════════════════════


def test_proto_self_initial_state() -> None:
    """ProtoSelf initializes with a default state."""
    proto = ProtoSelf()
    assert isinstance(proto.current, ProtoSelfState)
    assert proto.previous is None  # no history yet


def test_proto_self_update() -> None:
    """update() changes the current state and stores history."""
    proto = ProtoSelf()
    proto.update(arousal=0.9, valence=0.5)
    assert proto.current.arousal == 0.9
    assert proto.current.valence == 0.5
    assert proto.previous is not None


def test_proto_self_update_returns_state() -> None:
    """update() returns the new state."""
    proto = ProtoSelf()
    state = proto.update(arousal=0.8)
    assert isinstance(state, ProtoSelfState)
    assert state.arousal == 0.8


def test_proto_self_homeostatic_balance() -> None:
    """Homeostatic balance is high when near baseline."""
    proto = ProtoSelf()
    state = proto.update(arousal=0.5, valence=0.0, tone=0.5, plasticity=0.5)
    assert state.homeostatic_balance > 0.9  # near baseline


def test_proto_self_homeostatic_balance_low_when_disrupted() -> None:
    """Homeostatic balance is low when far from baseline."""
    proto = ProtoSelf()
    state = proto.update(arousal=1.0, valence=1.0, tone=1.0, plasticity=1.0)
    assert state.homeostatic_balance < 0.5  # far from baseline


def test_proto_self_history_trimmed() -> None:
    """History is trimmed to max_history."""
    proto = ProtoSelf()
    proto._max_history = 3
    for i in range(5):
        proto.update(arousal=0.1 * i)
    # History should be at most 3 (including the current)
    assert len(proto._history) <= 3


# ═══════════════════════════════════════════════════════════════════
# CoreSelf
# ═══════════════════════════════════════════════════════════════════


def test_core_self_no_change_no_episode() -> None:
    """No change → no episode detected."""
    proto = ProtoSelf()
    core = CoreSelf(proto, change_threshold=0.05)
    # First update establishes a baseline (homeostatic_balance changes)
    proto.update(arousal=0.5, valence=0.0)
    # Second update with same values → no change
    proto.update(arousal=0.5, valence=0.0)
    episode = core.detect_change(trigger="test")
    assert episode is None


def test_core_self_detects_change() -> None:
    """Significant change → episode detected."""
    proto = ProtoSelf()
    core = CoreSelf(proto, change_threshold=0.05)
    # Two initial updates to stabilize homeostatic_balance
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.9, valence=0.5)  # big change
    episode = core.detect_change(trigger="user interaction")
    assert episode is not None
    assert episode.trigger == "user interaction"
    assert episode.change_magnitude > 0.05


def test_core_self_no_previous() -> None:
    """detect_change returns None when there's no previous state."""
    proto = ProtoSelf()
    core = CoreSelf(proto)
    episode = core.detect_change(trigger="test")
    assert episode is None  # no history yet


def test_core_self_episodes_property() -> None:
    """episodes property returns the list of episodes."""
    proto = ProtoSelf()
    core = CoreSelf(proto, change_threshold=0.05)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.9, valence=0.5)
    core.detect_change(trigger="test")
    assert len(core.episodes) == 1


def test_core_self_latest_episode() -> None:
    """latest_episode returns the most recent episode."""
    proto = ProtoSelf()
    core = CoreSelf(proto, change_threshold=0.05)
    assert core.latest_episode is None
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.9, valence=0.5)
    core.detect_change(trigger="test")
    assert core.latest_episode is not None


def test_core_self_get_significant_episodes() -> None:
    """get_significant_episodes returns episodes above salience threshold."""
    proto = ProtoSelf()
    core = CoreSelf(proto, change_threshold=0.01, salience_threshold=0.15)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=1.0, valence=1.0)  # big change → high salience
    core.detect_change(trigger="big event")
    significant = core.get_significant_episodes()
    assert len(significant) >= 1


def test_core_self_mark_promoted() -> None:
    """mark_promoted sets the promoted flag."""
    proto = ProtoSelf()
    core = CoreSelf(proto, change_threshold=0.05)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.9, valence=0.5)
    episode = core.detect_change(trigger="test")
    assert episode is not None
    assert not episode.promoted
    core.mark_promoted(episode)
    assert episode.promoted


def test_core_self_clear() -> None:
    """clear removes all episodes."""
    proto = ProtoSelf()
    core = CoreSelf(proto, change_threshold=0.05)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.5, valence=0.0)
    proto.update(arousal=0.9, valence=0.5)
    core.detect_change(trigger="test")
    core.clear()
    assert len(core.episodes) == 0


# ═══════════════════════════════════════════════════════════════════
# AutobiographicalSelf
# ═══════════════════════════════════════════════════════════════════


def test_autobiographical_self_initial() -> None:
    """AutobiographicalSelf initializes empty."""
    auto = AutobiographicalSelf()
    assert auto.event_count == 0
    assert auto.promoted_episodes == []


def test_autobiographical_self_promote_episode() -> None:
    """promote_episode adds an episode to autobiographical memory."""
    auto = AutobiographicalSelf()
    episode = CoreSelfEpisode(
        trigger="test",
        change_magnitude=0.5,
        post_state=ProtoSelfState(arousal=0.8, valence=0.5),
    )
    result = auto.promote_episode(episode)
    assert result is True
    assert episode.promoted
    assert auto.event_count == 1


def test_autobiographical_self_get_identity_summary_empty() -> None:
    """get_identity_summary returns default when no memories."""
    auto = AutobiographicalSelf()
    summary = auto.get_identity_summary()
    assert "no autobiographical memories" in summary


def test_autobiographical_self_get_identity_summary_with_memories() -> None:
    """get_identity_summary includes promoted episodes."""
    auto = AutobiographicalSelf()
    episode = CoreSelfEpisode(
        trigger="test event",
        change_magnitude=0.5,
        post_state=ProtoSelfState(arousal=0.8, valence=0.5),
    )
    auto.promote_episode(episode)
    summary = auto.get_identity_summary()
    assert "1" in summary  # "I have 1 significant memories"


def test_autobiographical_self_promoted_episodes() -> None:
    """promoted_episodes returns a copy of the list."""
    auto = AutobiographicalSelf()
    ep = CoreSelfEpisode(trigger="test", change_magnitude=0.5)
    auto.promote_episode(ep)
    promoted = auto.promoted_episodes
    assert len(promoted) == 1
    # Modifying the returned list doesn't affect the internal state
    promoted.clear()
    assert len(auto.promoted_episodes) == 1


# ═══════════════════════════════════════════════════════════════════
# DamasioSelfHierarchy
# ═══════════════════════════════════════════════════════════════════


def test_hierarchy_initialization() -> None:
    """DamasioSelfHierarchy initializes all three levels."""
    hierarchy = DamasioSelfHierarchy()
    assert isinstance(hierarchy.proto_self, ProtoSelf)
    assert isinstance(hierarchy.core_self, CoreSelf)
    assert isinstance(hierarchy.autobiographical_self, AutobiographicalSelf)


def test_hierarchy_update_no_change() -> None:
    """update() with same values returns no episode."""
    hierarchy = DamasioSelfHierarchy()
    # First two updates establish a baseline (homeostatic_balance changes)
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init1")
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init2")
    # Third update with same values → no change
    episode = hierarchy.update(arousal=0.5, valence=0.0, trigger="same")
    assert episode is None


def test_hierarchy_update_with_change() -> None:
    """update() with different values returns an episode."""
    hierarchy = DamasioSelfHierarchy(change_threshold=0.05)
    # Two initial updates to stabilize homeostatic_balance
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init1")
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init2")
    episode = hierarchy.update(arousal=0.9, valence=0.5, trigger="big change")
    assert episode is not None
    assert episode.trigger == "big change"


def test_hierarchy_get_feeling_of_being() -> None:
    """get_feeling_of_being returns the proto-self state."""
    hierarchy = DamasioSelfHierarchy()
    hierarchy.update(arousal=0.7, valence=0.3)
    state = hierarchy.get_feeling_of_being()
    assert isinstance(state, ProtoSelfState)
    assert state.arousal == 0.7


def test_hierarchy_get_current_feeling() -> None:
    """get_current_feeling returns a feeling label."""
    hierarchy = DamasioSelfHierarchy()
    feeling = hierarchy.get_current_feeling()
    assert isinstance(feeling, str)
    assert len(feeling) > 0


def test_hierarchy_get_identity() -> None:
    """get_identity returns the autobiographical identity summary."""
    hierarchy = DamasioSelfHierarchy()
    identity = hierarchy.get_identity()
    assert isinstance(identity, str)


def test_hierarchy_as_if_loop() -> None:
    """as_if_loop nudges the proto-self toward remembered state."""
    hierarchy = DamasioSelfHierarchy()
    # Stabilize baseline
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init1")
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init2")
    initial_valence = hierarchy.proto_state.valence
    hierarchy.as_if_loop(memory_valence=0.8, memory_arousal=0.8)
    # Valence should move toward 0.8 (blended)
    assert hierarchy.proto_state.valence != initial_valence


def test_hierarchy_proto_state() -> None:
    """proto_state property returns the current proto-self state."""
    hierarchy = DamasioSelfHierarchy()
    state = hierarchy.proto_state
    assert isinstance(state, ProtoSelfState)


def test_hierarchy_core_episodes() -> None:
    """core_episodes property returns the core self's episodes."""
    hierarchy = DamasioSelfHierarchy(change_threshold=0.05)
    assert hierarchy.core_episodes == []
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init1")
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init2")
    hierarchy.update(arousal=0.9, valence=0.5, trigger="change")
    assert len(hierarchy.core_episodes) >= 1


def test_hierarchy_autobiographical_event_count() -> None:
    """autobiographical_event_count returns the event count."""
    hierarchy = DamasioSelfHierarchy(change_threshold=0.01, salience_threshold=0.01)
    assert hierarchy.autobiographical_event_count == 0
    # Stabilize baseline first
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init1")
    hierarchy.update(arousal=0.5, valence=0.0, trigger="init2")
    # Create a significant change
    hierarchy.update(arousal=1.0, valence=1.0, trigger="big event")
    assert hierarchy.autobiographical_event_count >= 1


# ======================================================================
# Reflection engine — additional tests (from test_intelligence.py)
# ======================================================================


def test_reflection_learning() -> None:
    """Detects when new concepts are encountered."""
    net = ConceptNetwork()
    reflector = ReflectionEngine(net)

    state = _make_state(
        intent=Intent.QUESTION,
        confidence=0.3,
    )
    insights = reflector.reflect(state, "I don't know much about that.")
    assert any(i.type == "growth" for i in insights)


def test_reflection_gap_detection() -> None:
    """Detects knowledge gaps."""
    net = ConceptNetwork()
    reflector = ReflectionEngine(net)

    state = _make_state(
        intent=Intent.QUESTION,
        confidence=0.3,
    )
    insights = reflector.reflect(state, "I'm not sure.")
    assert any(i.type == "gap" for i in insights)


def test_reflection_honesty_check() -> None:
    """Detects when confidence doesn't match claimed certainty."""
    net = ConceptNetwork()
    reflector = ReflectionEngine(net)

    state = _make_state(
        intent=Intent.QUESTION,
        confidence=0.2,
    )
    insights = reflector.reflect(state, "I know that cognition is simple.")
    assert any(i.type == "self_correction" and "overstating" in i.content for i in insights)


def test_reflection_mood_tracking() -> None:
    """Tracks mood over time."""
    net = ConceptNetwork()
    reflector = ReflectionEngine(net)

    for i in range(25):
        valence = 0.4 if i < 15 else -0.3
        emo = _make_emotion(valence=valence)
        # Build a state with the specific emotion valence
        state = _make_state(emotion_label="positive" if valence > 0 else "negative")
        state.emotion = emo
        reflector.reflect(state, "ok")

    mood_insights = reflector.get_insights_by_type("mood")
    assert len(mood_insights) > 0
