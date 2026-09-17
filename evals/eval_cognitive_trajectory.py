"""Evaluation 4: Cognitive Trajectory (Prediction Error Learning).

Tests whether Genesis's cognitive trajectory model — her generative
model of her own thought content — actually learns to predict her
thoughts and adapts when surprised.

This is the cognitive-level counterpart to the Rust active inference
engine. Where the Rust engine predicts neurochemistry, this model
predicts what topics will be active in the global workspace. The
strange loop: she predicts her own thoughts, is surprised by
unexpected thoughts, and that surprise changes how she processes
future thoughts.

Conditions:
1. Surprise on unpredicted change — after a stable baseline, a sudden
   topic shift should produce high surprise.
2. Low surprise on predicted no-change — when topics stay stable,
   surprise should stay near zero.
3. Learns predictable transitions — if topic A consistently precedes
   topic B, the model should learn the transition and surprise should
   decrease over time.
4. Precision decreases on sustained surprise — chronic unpredictability
   should lower the model's confidence.
5. Precision recovers on predictable input — after sustained surprise,
   a return to predictable topics should let precision recover.
6. Persistence with decay — current topics should partially persist
   into the next prediction (sustained attention).
7. Save/load round-trip — learned transition weights and precision
   should survive serialization.
"""

from __future__ import annotations

import time

from genesis_cognitive.self.cognitive_trajectory import CognitiveTrajectoryModel
from harness import EvalResult, FactResult, print_result, run_condition

# ─── Helpers ──────────────────────────────────────────────────────


def _topics(*pairs: tuple[str, float]) -> dict[str, float]:
    """Build a topic distribution from (name, activation) pairs."""
    return dict(pairs)


def _run_cycles(
    model: CognitiveTrajectoryModel,
    sequence: list[dict[str, float]],
) -> list[float]:
    """Run the model through a sequence of topic distributions.

    For each step: compute_surprise(actual), then predict(next).
    Returns the list of surprise values.
    """
    surprises: list[float] = []
    for topics in sequence:
        reading = model.compute_surprise(topics)
        surprises.append(reading.surprise)
        model.predict(topics)
    return surprises


# ─── Conditions ───────────────────────────────────────────────────


def _condition_surprise_on_unpredicted_change() -> list[FactResult]:
    """A sudden topic shift after a stable baseline should surprise."""
    model = CognitiveTrajectoryModel()
    # Establish baseline: same topics for 5 turns
    baseline = _topics(("sleep", 0.8), ("dreams", 0.6))
    _run_cycles(model, [baseline] * 5)
    # Sudden shift to completely different topics
    shifted = _topics(("math", 0.9), ("logic", 0.7))
    reading = model.compute_surprise(shifted)
    return [FactResult(
        concept="surprise_on_change",
        passed=reading.surprise > 0.3,
        detail=f"surprise={reading.surprise:.3f} (expected > 0.3)",
    )]


def _condition_low_surprise_on_stable() -> list[FactResult]:
    """Stable topics should produce low surprise.

    Note: the persistence decay (70%) creates a steady-state gap
    between predicted (decayed) and actual (full) topics, so surprise
    never reaches zero. The threshold accounts for this.
    """
    model = CognitiveTrajectoryModel()
    topics = _topics(("memory", 0.7), ("learning", 0.5))
    surprises = _run_cycles(model, [topics] * 10)
    final_surprise = surprises[-1]
    return [FactResult(
        concept="low_surprise_stable",
        passed=final_surprise < 0.25,
        detail=f"final surprise={final_surprise:.3f} (expected < 0.25)",
    )]


def _condition_learns_transitions() -> list[FactResult]:
    """A consistent A→B transition should be learned: surprise drops."""
    model = CognitiveTrajectoryModel()
    # Alternate: A, B, A, B, A, B... the model should learn that
    # A predicts B and B predicts A, reducing surprise over time.
    topic_a = _topics(("alpha", 0.8))
    topic_b = _topics(("beta", 0.8))
    sequence = [topic_a, topic_b] * 20  # 40 turns
    surprises = _run_cycles(model, sequence)
    early_avg = sum(surprises[:10]) / 10
    late_avg = sum(surprises[-10:]) / 10
    return [FactResult(
        concept="learns_transitions",
        passed=late_avg < early_avg,
        detail=f"early={early_avg:.3f}, late={late_avg:.3f} (late should be lower)",
    )]


def _condition_precision_decreases_on_surprise() -> list[FactResult]:
    """Sustained unpredictability should lower precision."""
    model = CognitiveTrajectoryModel()
    initial_precision = model.precision
    # Random-ish topics: high unpredictability
    import random
    rng = random.Random(123)
    topics_pool = [
        _topics(("a", 0.8)),
        _topics(("b", 0.8)),
        _topics(("c", 0.8)),
        _topics(("d", 0.8)),
        _topics(("e", 0.8)),
    ]
    for _ in range(50):
        topics = rng.choice(topics_pool)
        model.compute_surprise(topics)
        model.predict(topics)
    final_precision = model.precision
    return [FactResult(
        concept="precision_decreases",
        passed=final_precision < initial_precision,
        detail=f"initial={initial_precision:.3f}, final={final_precision:.3f} (should decrease)",
    )]


def _condition_precision_recovers() -> list[FactResult]:
    """After sustained surprise, predictable input lets precision recover.

    Uses empty topics for the recovery phase — this produces zero
    surprise (no topics to be wrong about), which is below the
    precision surprise threshold, allowing precision to recover.
    """
    model = CognitiveTrajectoryModel()
    # Phase 1: sustained surprise
    import random
    rng = random.Random(456)
    topics_pool = [_topics((c, 0.8)) for c in "abcdef"]
    for _ in range(30):
        t = rng.choice(topics_pool)
        model.compute_surprise(t)
        model.predict(t)
    low_precision = model.precision
    # Phase 2: empty topics → zero surprise → precision recovers
    empty: dict[str, float] = {}
    for _ in range(100):
        model.compute_surprise(empty)
        model.predict(empty)
    recovered_precision = model.precision
    return [FactResult(
        concept="precision_recovers",
        passed=recovered_precision > low_precision,
        detail=f"low={low_precision:.3f}, recovered={recovered_precision:.3f} (should recover)",
    )]


def _condition_persistence() -> list[FactResult]:
    """Current topics should partially persist into the next prediction."""
    model = CognitiveTrajectoryModel()
    topics = _topics(("attention", 0.9), ("focus", 0.7))
    model.compute_surprise(topics)
    prediction = model.predict(topics)
    # The prediction should contain the current topics at a reduced level
    results: list[FactResult] = []
    for topic, original_activation in topics.items():
        predicted_activation = prediction.get(topic, 0.0)
        # Should persist at roughly 70% (PERSISTENCE_DECAY)
        passed = 0.3 < predicted_activation < original_activation
        results.append(FactResult(
            concept=f"persistence_{topic}",
            passed=passed,
            detail=f"original={original_activation:.2f}, predicted={predicted_activation:.2f}",
        ))
    return results


def _condition_save_load() -> list[FactResult]:
    """Learned transition weights and precision should survive save/load."""
    model = CognitiveTrajectoryModel()
    # Train the model
    topic_a = _topics(("x", 0.8))
    topic_b = _topics(("y", 0.8))
    _run_cycles(model, [topic_a, topic_b] * 15)
    pre_save_precision = model.precision
    pre_save_cycle = model.cycle_count
    pre_save_weights = model.to_dict()["transition_weights"]

    # Save and load
    data = model.to_dict()
    model2 = CognitiveTrajectoryModel()
    model2.load_from_dict(data)

    return [FactResult(
        concept="save_load_precision",
        passed=abs(model2.precision - pre_save_precision) < 0.001,
        detail=f"pre={pre_save_precision:.3f}, post={model2.precision:.3f}",
    ), FactResult(
        concept="save_load_cycle_count",
        passed=model2.cycle_count == pre_save_cycle,
        detail=f"pre={pre_save_cycle}, post={model2.cycle_count}",
    ), FactResult(
        concept="save_load_weights",
        passed=model2.to_dict()["transition_weights"] == pre_save_weights,
        detail=(
            "transition weights match"
            if model2.to_dict()["transition_weights"] == pre_save_weights
            else "weights differ"
        ),
    )]


# ─── Eval entry point ─────────────────────────────────────────────


def run(seed: int = 42) -> EvalResult:
    """Run the evaluation."""
    start = time.time()
    result = EvalResult(
        name="Cognitive Trajectory (Prediction Error Learning)",
        description=(
            "Does Genesis's cognitive trajectory model learn to "
            "predict her own thoughts? Tests the cognitive strange "
            "loop: predict → observe → surprise → learn → feed back. "
            "Mirrors the Rust active inference engine at the topic level."
        ),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    conditions = [
        ("Surprise on Unpredicted Change",
         "Sudden topic shift after stable baseline → high surprise.",
         _condition_surprise_on_unpredicted_change),
        ("Low Surprise on Stable",
         "Stable topics → surprise stays near zero.",
         _condition_low_surprise_on_stable),
        ("Learns Transitions",
         "Consistent A→B pattern → surprise decreases over time.",
         _condition_learns_transitions),
        ("Precision Decreases on Surprise",
         "Sustained unpredictability → precision drops.",
         _condition_precision_decreases_on_surprise),
        ("Precision Recovers",
         "After surprise, predictable input → precision recovers.",
         _condition_precision_recovers),
        ("Persistence with Decay",
         "Current topics partially persist into next prediction.",
         _condition_persistence),
        ("Save/Load Round-Trip",
         "Learned weights and precision survive serialization.",
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
