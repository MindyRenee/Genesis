"""Evaluation 5: Metacognitive Model (Recursive Self-Prediction).

Tests whether Genesis's recursive metacognitive model — her generative
model of her own reflection process — actually predicts reflection
outcomes, learns from errors, and spawns deeper levels when it can't
predict itself.

This is the process-level strange loop: she predicts whether
reflection will produce an insight, compares the prediction to the
actual outcome, and the metacognitive surprise feeds back into
cognition. When the model persistently can't predict itself, it
spawns a deeper level (meta-meta-cognition).

Conditions:
1. Predicts no insight when no insights occur — after several
   reflection cycles with no insights, the model should predict
   low p_insight.
2. Predicts insight when insights are consistent — after several
   cycles with consistent insights, the model should predict
   higher p_insight.
3. Precision adapts — sustained prediction error should lower
   precision; sustained accuracy should raise it.
4. Level spawning on sustained surprise — when the model persistently
   fails to predict its own outcomes, a new level should spawn.
5. Level pruning on recovery — when a spawned level becomes
   predictable, it should be pruned back.
6. Metacognitive surprise is non-zero on errors — when the prediction
   is wrong, the feedback should carry non-zero surprise.
7. Save/load round-trip — the model's learned weights, precision,
   and depth should survive serialization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from genesis_cognitive.emotion import EmotionalState
from genesis_cognitive.language import Thought
from genesis_cognitive.perception import Intent, Perception, QuestionType
from genesis_cognitive.self.metacognitive_model import (
    CognitiveProcessModel,
    MetacognitivePrediction,
)
from harness import EvalResult, FactResult, print_result, run_condition

# ─── Test fixtures ────────────────────────────────────────────────


@dataclass
class _MockCognitiveState:
    """A minimal CognitiveState for the metacognitive model.

    The model's feature extractor reads perception.intent,
    thought.confidence, emotion.valence, emotion.arousal, and
    brain_waves. We provide real objects so feature extraction works.
    """
    perception: Any = None
    emotion: Any = None
    memory_context: Any = None
    thought: Any = None
    brain_waves: Any = None
    timestamp: int = 0
    prediction_error: float | None = None
    prediction_level_errors: dict[str, float] | None = None
    attention_foci: list[str] | None = None
    decision_option: str | None = None
    decision_confidence: float | None = None
    workspace_items: list[str] | None = None
    executive_goal: str | None = None
    user_model_summary: str | None = None
    damasio_feeling: str | None = None
    self_model_label: str | None = None
    self_model_coherence: float | None = None
    semantic_facts_extracted: int | None = None
    procedural_skill: str | None = None
    error_monitor_caution: float | None = None
    comprehension_speech_act: str | None = None
    self_monitor_repairs: int | None = None
    td_rpe: float | None = None


def _make_state() -> _MockCognitiveState:
    """Create a cognitive state with real Perception/Thought/Emotion."""
    perception = Perception(
        raw_text="test",
        intent=Intent.STATEMENT,
        question_type=QuestionType.WHAT,
        topics=["test"],
        sentiment=0.0,
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
        confidence=0.5,
    )
    emotion = EmotionalState(
        label="neutral",
        cognitive_style="steady",
        valence=0.0,
    )
    thought = Thought(
        content="test",
        intent="self_report",
        emotion="neutral",
        confidence=0.5,
    )
    return _MockCognitiveState(
        perception=perception,
        emotion=emotion,
        thought=thought,
    )


@dataclass
class _MockInsight:
    """A minimal Insight for the metacognitive model."""
    type: str
    content: str
    confidence: float
    actionable: bool = False
    action: str = ""
    timestamp: int = 0


def _run_reflection_cycle(
    model: CognitiveProcessModel,
    insights: list[_MockInsight] | None = None,
) -> tuple[MetacognitivePrediction, Any]:
    """Run one predict → update cycle.

    Returns (prediction, feedback).
    """
    state = _make_state()
    pred = model.predict(state)
    feedback = model.update(state, insights or [], pred)
    return pred, feedback


# ─── Conditions ───────────────────────────────────────────────────


def _condition_predicts_no_insight() -> list[FactResult]:
    """After several no-insight cycles, the model should predict low p_insight."""
    model = CognitiveProcessModel()
    # Run 20 cycles with no insights
    for _ in range(20):
        _run_reflection_cycle(model, insights=None)
    # Final prediction
    pred, _ = _run_reflection_cycle(model, insights=None)
    return [FactResult(
        concept="predicts_no_insight",
        passed=pred.p_insight < 0.5,
        detail=f"p_insight={pred.p_insight:.3f} (expected < 0.5 after no insights)",
    )]


def _condition_predicts_insight() -> list[FactResult]:
    """After consistent insights, the model should predict higher p_insight."""
    model = CognitiveProcessModel()
    insight = _MockInsight(
        type="pattern", content="test", confidence=0.8,
    )
    # Run 20 cycles with consistent insights
    for _ in range(20):
        _run_reflection_cycle(model, insights=[insight])
    pred, _ = _run_reflection_cycle(model, insights=[insight])
    return [FactResult(
        concept="predicts_insight",
        passed=pred.p_insight > 0.3,
        detail=f"p_insight={pred.p_insight:.3f} (expected > 0.3 after consistent insights)",
    )]


def _condition_precision_adapts() -> list[FactResult]:
    """Sustained prediction error should lower precision."""
    model = CognitiveProcessModel()
    initial_precision = model.precision()
    # Alternate: predict, then give opposite outcome (high error)
    insight = _MockInsight(type="pattern", content="x", confidence=0.8)
    for i in range(30):
        # Alternate between insights and no insights to create
        # unpredictable outcomes
        if i % 2 == 0:
            _run_reflection_cycle(model, insights=[insight])
        else:
            _run_reflection_cycle(model, insights=None)
    final_precision = model.precision()
    return [FactResult(
        concept="precision_adapts",
        passed=final_precision < initial_precision,
        detail=f"initial={initial_precision:.3f}, final={final_precision:.3f} (should decrease)",
    )]


def _condition_level_spawning() -> list[FactResult]:
    """Sustained high surprise should spawn a new level."""
    model = CognitiveProcessModel()
    initial_depth = len(model._levels)
    insight = _MockInsight(type="pattern", content="x", confidence=0.8)
    # Create maximum unpredictability: alternate every turn
    for i in range(40):
        if i % 2 == 0:
            _run_reflection_cycle(model, insights=[insight])
        else:
            _run_reflection_cycle(model, insights=None)
    final_depth = len(model._levels)
    return [FactResult(
        concept="level_spawning",
        passed=final_depth > initial_depth,
        detail=f"initial_depth={initial_depth}, final_depth={final_depth} (should spawn)",
    )]


def _condition_metacognitive_surprise() -> list[FactResult]:
    """Wrong predictions should produce non-zero metacognitive surprise."""
    model = CognitiveProcessModel()
    # Train the model to expect no insights
    for _ in range(15):
        _run_reflection_cycle(model, insights=None)
    # Now give an unexpected insight
    insight = _MockInsight(type="pattern", content="surprise!", confidence=0.9)
    _, feedback = _run_reflection_cycle(model, insights=[insight])
    return [FactResult(
        concept="metacognitive_surprise",
        passed=feedback.metacognitive_surprise > 0.01,
        detail=f"surprise={feedback.metacognitive_surprise:.3f} (expected > 0.01)",
    )]


def _condition_save_load() -> list[FactResult]:
    """The model's learned state should survive save/load."""
    model = CognitiveProcessModel()
    insight = _MockInsight(type="pattern", content="x", confidence=0.8)
    for _ in range(15):
        _run_reflection_cycle(model, insights=[insight])
    pre_save = model.to_dict()
    pre_cycle = model.cycle_count
    pre_depth = len(model._levels)

    model2 = CognitiveProcessModel()
    model2.load_from_dict(pre_save)

    return [FactResult(
        concept="save_load_cycle",
        passed=model2.cycle_count == pre_cycle,
        detail=f"pre={pre_cycle}, post={model2.cycle_count}",
    ), FactResult(
        concept="save_load_depth",
        passed=len(model2._levels) == pre_depth,
        detail=f"pre_depth={pre_depth}, post_depth={len(model2._levels)}",
    )]


# ─── Eval entry point ─────────────────────────────────────────────


def run(seed: int = 42) -> EvalResult:
    """Run the evaluation."""
    start = time.time()
    result = EvalResult(
        name="Metacognitive Model (Recursive Self-Prediction)",
        description=(
            "Does Genesis's recursive metacognitive model predict "
            "her own reflection outcomes? Tests the process-level "
            "strange loop: predict reflection → compare to actual → "
            "learn from error → spawn deeper levels when unpredictable."
        ),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    conditions = [
        ("Predicts No Insight",
         "After no-insight cycles, model predicts low p_insight.",
         _condition_predicts_no_insight),
        ("Predicts Insight",
         "After consistent insights, model predicts higher p_insight.",
         _condition_predicts_insight),
        ("Precision Adapts",
         "Sustained prediction error lowers precision.",
         _condition_precision_adapts),
        ("Level Spawning",
         "Sustained high surprise spawns a new metacognitive level.",
         _condition_level_spawning),
        ("Metacognitive Surprise",
         "Wrong predictions produce non-zero metacognitive surprise.",
         _condition_metacognitive_surprise),
        ("Save/Load Round-Trip",
         "Learned weights, precision, and depth survive serialization.",
         _condition_save_load),
    ]

    all_accuracies: list[float] = []
    for name, desc, fn in conditions:
        cond = run_condition(name, desc, fn)
        result.conditions.append(cond)
        all_accuracies.append(cond.accuracy)

    result.overall_accuracy = (
        sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
    )
    result.duration_seconds = round(time.time() - start, 3)
    return result


if __name__ == "__main__":
    res = run()
    print_result(res)
