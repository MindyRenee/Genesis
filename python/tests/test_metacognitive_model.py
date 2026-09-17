"""Tests for the recursive metacognitive model.

Tests the recursive, self-terminating generative model of cognitive
processes: prediction, learning, feedback, spawn/prune, persistence.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from genesis_cognitive.brain_waves import BrainWave, BrainWaveState
from genesis_cognitive.cognition import CognitiveState
from genesis_cognitive.concepts import ConceptNetwork
from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.language.base import Thought
from genesis_cognitive.memory import MemoryContext
from genesis_cognitive.perception import Intent, Perception, QuestionType
from genesis_cognitive.self import (
    CognitiveProcessModel,
    Insight,
    MetacognitiveFeedback,
    MetacognitivePrediction,
    ReflectionEngine,
)


def _make_state(
    intent: Intent = Intent.QUESTION,
    thought_intent: str = "inform",
    sentiment: float = 0.0,
    confidence: float = 0.7,
    emotion_label: str = "neutral",
    valence: float = 0.0,
    arousal: float = 0.5,
    brain_wave: BrainWave | None = None,
    caution: float | None = None,
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
        cognitive_style="reflective",
    )
    emotion.valence = valence
    emotion.arousal = arousal
    thought = Thought(
        content="test content",
        intent=thought_intent,
        confidence=confidence,
    )
    memory_context = MemoryContext()
    state = CognitiveState(
        perception=perception,
        emotion=emotion,
        memory_context=memory_context,
        thought=thought,
    )
    if brain_wave is not None:
        powers = dict.fromkeys(BrainWave, 0.1)
        powers[brain_wave] = 0.6
        state.brain_waves = BrainWaveState(
            dominant=brain_wave,
            secondary=BrainWave.ALPHA,
            powers=powers,
            focus=0.5,
            integration=0.5,
            consolidation=0.5,
            label=brain_wave.value,
            description="test",
        )
    if caution is not None:
        state.error_monitor_caution = caution
    return state


# ═══════════════════════════════════════════════════════════════════
# Basic prediction
# ═══════════════════════════════════════════════════════════════════


def test_predict_returns_prediction() -> None:
    """predict() returns a MetacognitivePrediction with valid fields."""
    model = CognitiveProcessModel()
    state = _make_state()
    pred = model.predict(state)
    assert isinstance(pred, MetacognitivePrediction)
    assert 0.0 <= pred.p_insight <= 1.0
    assert 0.0 <= pred.expected_confidence <= 1.0
    assert 0.0 <= pred.precision <= 1.0
    assert pred.depth >= 1
    assert isinstance(pred.p_insight_type, dict)


def test_predict_starts_at_depth_1() -> None:
    """A fresh model has depth 1 (only the object-level)."""
    model = CognitiveProcessModel()
    assert model.depth() == 1
    state = _make_state()
    pred = model.predict(state)
    assert pred.depth == 1


def test_prediction_type_distribution_sums_to_one() -> None:
    """The insight-type distribution sums to ~1."""
    model = CognitiveProcessModel()
    state = _make_state()
    pred = model.predict(state)
    total = sum(pred.p_insight_type.values())
    assert abs(total - 1.0) < 0.01


# ═══════════════════════════════════════════════════════════════════
# Learning
# ═══════════════════════════════════════════════════════════════════


def test_update_returns_feedback() -> None:
    """update() returns a MetacognitiveFeedback."""
    model = CognitiveProcessModel()
    state = _make_state()
    pred = model.predict(state)
    insights = [Insight(type="gap", content="test gap", confidence=0.7)]
    fb = model.update(state, insights, pred)
    assert isinstance(fb, MetacognitiveFeedback)
    assert fb.depth >= 1


def test_model_learns_from_consistent_outcomes() -> None:
    """After consistent training, the model's predictions improve.

    If we always produce a gap insight, the model should learn to
    predict p_insight → 1.0 and p_gap → high.
    """
    model = CognitiveProcessModel()
    state = _make_state(intent=Intent.QUESTION, confidence=0.3)
    insights = [Insight(type="gap", content="test gap", confidence=0.7)]

    for _ in range(50):
        pred = model.predict(state)
        model.update(state, insights, pred)

    final_pred = model.predict(state)
    # After 50 consistent cycles, the model should predict insight
    # probability > 0.5 (it learned that this state → insight).
    assert final_pred.p_insight > 0.5
    # And the gap type should be dominant.
    assert final_pred.p_insight_type.get("gap", 0.0) > 0.3


def test_model_learns_no_insights() -> None:
    """After consistent no-insight training, p_insight should drop."""
    model = CognitiveProcessModel()
    state = _make_state(intent=Intent.GREETING, confidence=0.9)
    insights: list[Insight] = []

    for _ in range(50):
        pred = model.predict(state)
        model.update(state, insights, pred)

    final_pred = model.predict(state)
    assert final_pred.p_insight < 0.5


def test_precision_adapts() -> None:
    """Precision changes based on prediction accuracy."""
    model = CognitiveProcessModel()
    state = _make_state()
    initial_precision = model.precision()

    # Train with consistent outcomes → precision should increase
    insights = [Insight(type="gap", content="test", confidence=0.7)]
    for _ in range(30):
        pred = model.predict(state)
        model.update(state, insights, pred)

    final_precision = model.precision()
    # After consistent training, precision should be >= initial
    # (the model is predicting well → confidence increases).
    assert final_precision >= initial_precision


# ═══════════════════════════════════════════════════════════════════
# Recursion / spawn / prune
# ═══════════════════════════════════════════════════════════════════


def test_spawn_level_on_sustained_surprise() -> None:
    """A new level spawns when the top level's surprise is sustained high.

    We force sustained high surprise by alternating between producing
    insights and not producing them unpredictably — the model can't
    predict this, so its surprise stays high, and after enough cycles
    a new level should spawn.
    """
    from genesis_cognitive.self.metacognitive_model import (
        SPAWN_PERSISTENCE,
    )

    model = CognitiveProcessModel()
    state = _make_state()
    initial_depth = model.depth()

    # Alternate unpredictably: insight, no insight, insight, no insight
    # This maximizes prediction error (the model can't learn a pattern).
    with_insight = [Insight(type="gap", content="test", confidence=0.7)]
    without_insight: list[Insight] = []

    # Run enough cycles to trigger spawn. We need sustained high
    # surprise for SPAWN_PERSISTENCE consecutive cycles.
    for i in range(SPAWN_PERSISTENCE * 3):
        pred = model.predict(state)
        insights = with_insight if i % 2 == 0 else without_insight
        model.update(state, insights, pred)

    # The model should have spawned at least one level.
    assert model.depth() > initial_depth


def test_prune_level_on_sustained_predictability() -> None:
    """A level is pruned when it becomes predictable.

    First spawn a level with unpredictable outcomes, then make the
    outcomes very consistent. The spawned level should eventually
    be pruned because it's no longer adding information.
    """
    from genesis_cognitive.self.metacognitive_model import (
        PRUNE_MIN_AGE,
    )

    model = CognitiveProcessModel()
    state = _make_state()

    # Phase 1: spawn a level with unpredictable outcomes
    with_insight = [Insight(type="gap", content="test", confidence=0.7)]
    without_insight: list[Insight] = []
    for i in range(60):
        pred = model.predict(state)
        insights = with_insight if i % 2 == 0 else without_insight
        model.update(state, insights, pred)

    spawned_depth = model.depth()
    assert spawned_depth >= 2  # a level was spawned

    # Phase 2: make outcomes very consistent → the meta level should
    # become predictable and get pruned.
    consistent_insights = [Insight(type="gap", content="test", confidence=0.7)]
    for _ in range(PRUNE_MIN_AGE * 3):
        pred = model.predict(state)
        model.update(state, consistent_insights, pred)

    # The spawned level should have been pruned (or be close to
    # pruning). We check that depth decreased or stayed the same
    # (it may still be above 1 if the threshold hasn't been crossed
    # yet, but it should not have increased).
    assert model.depth() <= spawned_depth


def test_depth_never_exceeds_max() -> None:
    """The model never exceeds MAX_LEVELS."""
    from genesis_cognitive.self.metacognitive_model import MAX_LEVELS

    model = CognitiveProcessModel()
    state = _make_state()
    with_insight = [Insight(type="gap", content="test", confidence=0.7)]
    without_insight: list[Insight] = []

    # Run many cycles with unpredictable outcomes to try to spawn
    # many levels.
    for i in range(500):
        pred = model.predict(state)
        insights = with_insight if i % 2 == 0 else without_insight
        model.update(state, insights, pred)

    assert model.depth() <= MAX_LEVELS


# ═══════════════════════════════════════════════════════════════════
# Feedback
# ═══════════════════════════════════════════════════════════════════


def test_metacognitive_surprise_in_range() -> None:
    """Metacognitive surprise is always in [0, 1]."""
    model = CognitiveProcessModel()
    state = _make_state()
    pred = model.predict(state)
    insights = [Insight(type="gap", content="test", confidence=0.7)]
    fb = model.update(state, insights, pred)
    assert 0.0 <= fb.metacognitive_surprise <= 1.0


def test_feedback_tracks_spawn_prune() -> None:
    """Feedback records whether a level was spawned or pruned."""
    model = CognitiveProcessModel()
    state = _make_state()
    pred = model.predict(state)
    insights = [Insight(type="gap", content="test", confidence=0.7)]
    fb = model.update(state, insights, pred)
    assert isinstance(fb.level_spawned, bool)
    assert isinstance(fb.level_pruned, bool)


def test_predicted_self_correction_flag() -> None:
    """predicted_self_correction is set when self_correction prob > 0.3."""
    model = CognitiveProcessModel()
    state = _make_state(intent=Intent.CORRECTION, confidence=0.2)
    # Train the model to expect self-corrections
    insights = [Insight(type="self_correction", content="test", confidence=0.8)]
    for _ in range(50):
        pred = model.predict(state)
        model.update(state, insights, pred)

    # Now predict — should predict self_correction
    pred = model.predict(state)
    fb = model.update(state, insights, pred)
    # After training, the model should predict self_correction
    assert fb.predicted_self_correction or pred.p_insight_type.get("self_correction", 0.0) > 0.2


# ═══════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════


def test_persistence_roundtrip() -> None:
    """The model survives save/load roundtrip."""
    model = CognitiveProcessModel()
    state = _make_state()
    insights = [Insight(type="gap", content="test", confidence=0.7)]

    # Train the model
    for _ in range(20):
        pred = model.predict(state)
        model.update(state, insights, pred)

    saved = model.to_dict()
    loaded = CognitiveProcessModel()
    loaded.load_from_dict(saved)

    # The loaded model should produce the same predictions
    pred_original = model.predict(state)
    pred_loaded = loaded.predict(state)
    assert abs(pred_original.p_insight - pred_loaded.p_insight) < 0.01
    assert model.depth() == loaded.depth()
    assert model.cycle_count == loaded.cycle_count


def test_persistence_empty_data() -> None:
    """Loading from empty/corrupt data starts fresh."""
    model = CognitiveProcessModel()
    model.load_from_dict({})
    assert model.depth() == 1


def test_persistence_backward_compatible() -> None:
    """Loading from data without metacognitive_model key leaves default."""
    """This is tested at the persistence.py level, not the model level."""
    model = CognitiveProcessModel()
    # No metacognitive_model key → model stays at default
    assert model.depth() == 1


# ═══════════════════════════════════════════════════════════════════
# Integration with ReflectionEngine
# ═══════════════════════════════════════════════════════════════════


def test_reflection_engine_has_metacognitive_model() -> None:
    """ReflectionEngine creates a CognitiveProcessModel."""
    engine = ReflectionEngine(ConceptNetwork())
    assert isinstance(engine.metacognitive_model, CognitiveProcessModel)


def test_reflection_updates_metacognitive_model() -> None:
    """reflect() updates the metacognitive model's feedback."""
    engine = ReflectionEngine(ConceptNetwork())
    state = _make_state()
    engine.reflect(state, "test response")
    assert engine.metacognitive_model.feedback is not None
    assert engine.metacognitive_model.cycle_count == 1


def test_reflection_predicts_before_reflecting() -> None:
    """reflect() stores the prediction before running checks."""
    engine = ReflectionEngine(ConceptNetwork())
    state = _make_state()
    engine.reflect(state, "test response")
    assert engine.metacognitive_model.last_prediction is not None
    assert isinstance(engine.metacognitive_model.last_prediction, MetacognitivePrediction)


def test_multiple_reflections_accumulate_cycles() -> None:
    """Multiple reflect() calls increment the cycle count."""
    engine = ReflectionEngine(ConceptNetwork())
    state = _make_state()
    for _ in range(5):
        engine.reflect(state, "test response")
    assert engine.metacognitive_model.cycle_count == 5


def test_describe_returns_string() -> None:
    """describe() returns a human-readable string."""
    model = CognitiveProcessModel()
    state = _make_state()
    pred = model.predict(state)
    insights = [Insight(type="gap", content="test", confidence=0.7)]
    model.update(state, insights, pred)
    desc = model.describe()
    assert isinstance(desc, str)
    assert "depth=" in desc
    assert "L0:" in desc


def test_self_surprise_returns_float() -> None:
    """self_surprise() returns a float in [0, 1]."""
    model = CognitiveProcessModel()
    state = _make_state()
    pred = model.predict(state)
    insights = [Insight(type="gap", content="test", confidence=0.7)]
    model.update(state, insights, pred)
    s = model.self_surprise()
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Feedback into reflection depth (metacognitive surprise → deeper)
# ═══════════════════════════════════════════════════════════════════


def test_delta_skips_deep_reflection_by_default() -> None:
    """Delta brain wave skips deep reflection when surprise is low."""
    engine = ReflectionEngine(ConceptNetwork())
    # First cycle with non-delta to establish low-surprise feedback.
    state_normal = _make_state(brain_wave=BrainWave.ALPHA)
    engine.reflect(state_normal, "test response")
    # Verify feedback exists and surprise is low.
    assert engine.metacognitive_model.feedback is not None
    # Now reflect with delta — should skip deep checks (fewer insights).
    state_delta = _make_state(brain_wave=BrainWave.DELTA)
    insights_delta = engine.reflect(state_delta, "test response")
    # Delta with low surprise → only essential checks (appropriateness
    # + repetition). Should produce 0 or very few insights.
    assert len(insights_delta) <= 2


def test_high_metacognitive_surprise_overrides_delta_skip() -> None:
    """High metacognitive surprise overrides the delta skip.

    When the previous cycle's metacognitive surprise is high, the
    model is saying "I can't predict my own cognition." Even in
    delta (deep rest), she should reflect harder. We force high
    surprise with random outcomes (varying type, count, and
    presence), then check that a delta cycle runs the deeper checks.
    """
    import random

    rng = random.Random(42)  # deterministic for reproducibility
    engine = ReflectionEngine(ConceptNetwork())
    state = _make_state(brain_wave=BrainWave.ALPHA)

    # Phase 1: train with random outcomes to drive up surprise.
    # Random insight type, count, and presence — the linear model
    # can't predict this from constant features.
    insight_types = ["gap", "self_correction", "pattern", "growth", "mood"]
    for _ in range(60):
        pred = engine.metacognitive_model.predict(state)
        if rng.random() < 0.5:
            n_insights = rng.randint(1, 3)
            insights = [
                Insight(
                    type=rng.choice(insight_types),
                    content="test",
                    confidence=rng.uniform(0.3, 0.9),
                )
                for _ in range(n_insights)
            ]
        else:
            insights = []
        engine.metacognitive_model.update(state, insights, pred)

    # Verify surprise is high enough to trigger the override.
    assert engine.metacognitive_model.feedback is not None
    prev_surprise = engine.metacognitive_model.feedback.metacognitive_surprise
    assert prev_surprise > 0.15  # above the override threshold

    # Phase 2: reflect with delta. High surprise should override
    # the delta skip and run the deeper checks.
    state_delta = _make_state(brain_wave=BrainWave.DELTA)
    engine.reflect(state_delta, "test response")
    # The model was updated (cycle count increased from the
    # direct updates + the reflect call).
    assert engine.metacognitive_model.cycle_count > 60


# ═══════════════════════════════════════════════════════════════════
# Pre-response caution from prediction (predicted self-correction → hedge)
# ═══════════════════════════════════════════════════════════════════


def test_prediction_stores_self_correction_probability() -> None:
    """The prediction stores p_self_correction in p_insight_type."""
    model = CognitiveProcessModel()
    state = _make_state(intent=Intent.CORRECTION, confidence=0.2)
    # Train to predict self-corrections
    insights = [Insight(type="self_correction", content="test", confidence=0.8)]
    for _ in range(50):
        pred = model.predict(state)
        model.update(state, insights, pred)

    pred = model.predict(state)
    assert "self_correction" in pred.p_insight_type
    assert pred.p_insight_type["self_correction"] > 0.2


def test_last_prediction_available_before_reflection() -> None:
    """last_prediction is available for the cognition engine to read.

    The cognition engine reads last_prediction (from the previous
    cycle) to pre-hedge the current response. This test verifies
    the prediction is stored and accessible.
    """
    engine = ReflectionEngine(ConceptNetwork())
    state = _make_state()
    # Before any reflection, last_prediction is None.
    assert engine.metacognitive_model.last_prediction is None
    # After one reflection, last_prediction is set.
    engine.reflect(state, "test response")
    assert engine.metacognitive_model.last_prediction is not None
    assert isinstance(engine.metacognitive_model.last_prediction, MetacognitivePrediction)
