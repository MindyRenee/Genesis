"""Tests for the cognitive trajectory model.

Tests the generative model of thought content trajectory:
- Prediction (persistence + learned transitions)
- Surprise computation (L1 distance)
- Learning (delta rule)
- Precision adaptation (sustained surprise → decay)
- Boundedness (no NaN/inf, values in [0, 1])
"""

from __future__ import annotations

import math

import pytest

from genesis_cognitive.self.cognitive_trajectory import CognitiveTrajectoryModel

# Persistence decay: predict(x) → x * 0.7
_DECAY = 0.7


# ═══════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════


def test_trajectory_init_defaults() -> None:
    """Model starts with neutral state."""
    model = CognitiveTrajectoryModel()
    assert model.precision == 0.5
    assert model.surprise_ema == 0.0
    assert model.cycle_count == 0
    assert model.last_reading is None


# ═══════════════════════════════════════════════════════════════════
# compute_surprise — first cycle
# ═══════════════════════════════════════════════════════════════════


def test_surprise_first_cycle_empty_actual() -> None:
    """First cycle with empty actual and empty prediction → surprise 0."""
    model = CognitiveTrajectoryModel()
    reading = model.compute_surprise({})
    assert reading.surprise == 0.0
    assert reading.cycle_count == 1


def test_surprise_first_cycle_unpredicted_content() -> None:
    """First cycle: no prediction but actual content → surprise is the
    actual activation (everything is unexpected)."""
    model = CognitiveTrajectoryModel()
    reading = model.compute_surprise({"cats": 0.8})
    # No prediction → all actual content is surprising
    # L1 = |0.8 - 0| = 0.8, normalized by 1 topic = 0.8
    assert reading.surprise == pytest.approx(0.8)
    assert reading.cycle_count == 1
    assert reading.actual_topics == {"cats": 0.8}
    assert reading.predicted_topics == {}


# ═══════════════════════════════════════════════════════════════════
# predict — persistence and transitions
# ═══════════════════════════════════════════════════════════════════


def test_predict_persistence_decay() -> None:
    """Predicted topics persist at 70% of current activation."""
    model = CognitiveTrajectoryModel()
    predicted = model.predict({"cats": 1.0})
    assert predicted["cats"] == pytest.approx(_DECAY)


def test_predict_prunes_near_zero() -> None:
    """Topics below 0.01 activation are pruned from prediction."""
    model = CognitiveTrajectoryModel()
    predicted = model.predict({"weak": 0.01})
    # 0.01 * 0.7 = 0.007 < 0.01 → pruned
    assert "weak" not in predicted


def test_predict_empty_input() -> None:
    """Empty current topics → empty prediction."""
    model = CognitiveTrajectoryModel()
    predicted = model.predict({})
    assert predicted == {}


def test_predict_clamps_to_01() -> None:
    """Predicted activations are clamped to [0, 1]."""
    model = CognitiveTrajectoryModel()
    # Force a high prediction through transitions
    model._transition_weights = {
        "a": {"b": 1.0},
    }
    predicted = model.predict({"a": 1.0})
    # a persists at 0.7, b gets 1.0 * 1.0 * 0.3 = 0.3
    assert predicted["a"] <= 1.0
    assert predicted["b"] <= 1.0
    assert predicted["b"] >= 0.0


# ═══════════════════════════════════════════════════════════════════
# compute_surprise — with prediction
# ═══════════════════════════════════════════════════════════════════


def test_surprise_perfect_match() -> None:
    """Surprise is 0 when actual matches the decayed prediction."""
    model = CognitiveTrajectoryModel()
    # predict(1.0) → predicted = 0.7. compute_surprise(0.7) → match.
    model.predict({"cats": 1.0})
    reading = model.compute_surprise({"cats": _DECAY})
    assert reading.surprise == pytest.approx(0.0)


def test_surprise_total_mismatch() -> None:
    """Surprise is high when actual is completely different from prediction."""
    model = CognitiveTrajectoryModel()
    model.predict({"cats": 1.0})  # predicted = {"cats": 0.7}
    reading = model.compute_surprise({"dogs": 0.7})
    # L1 = |0.7 - 0| + |0 - 0.7| = 1.4, normalized by 2 topics = 0.7
    assert reading.surprise > 0.5
    assert reading.surprise == pytest.approx(0.7)


def test_surprise_partial_match() -> None:
    """Surprise is moderate when actual partially matches prediction."""
    model = CognitiveTrajectoryModel()
    model.predict({"cats": 1.0, "dogs": 0.5})
    # predicted = {"cats": 0.7, "dogs": 0.35}
    reading = model.compute_surprise({"cats": 0.7, "birds": 0.5})
    # cats matches (0.7 vs 0.7), dogs missing (0.35 vs 0), birds new (0 vs 0.5)
    # L1 = 0 + 0.35 + 0.5 = 0.85, normalized by 3 topics ≈ 0.283
    assert 0.2 < reading.surprise < 0.4


def test_surprise_bounded_01() -> None:
    """Surprise is always in [0, 1]."""
    model = CognitiveTrajectoryModel()
    model.predict({"a": 1.0, "b": 1.0})
    reading = model.compute_surprise({"c": 1.0, "d": 1.0, "e": 1.0, "f": 1.0})
    assert 0.0 <= reading.surprise <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Learning — transition weights
# ═══════════════════════════════════════════════════════════════════


def test_learning_updates_transitions() -> None:
    """compute_surprise updates transition weights via delta rule."""
    model = CognitiveTrajectoryModel()
    model.predict({"cats": 0.7})
    model.compute_surprise({"cats": 0.7, "dogs": 0.5})
    # After learning, "cats" → "dogs" transition should exist
    assert "cats" in model._transition_weights
    assert "dogs" in model._transition_weights["cats"]


def test_learning_transition_weight_bounded() -> None:
    """Transition weights stay in [-1, 1]."""
    model = CognitiveTrajectoryModel()
    model._precision = 1.0  # maximize learning rate
    for _ in range(100):
        model.predict({"cats": 1.0})
        model.compute_surprise({"dogs": 1.0})
    weight = model._transition_weights["cats"]["dogs"]
    assert -1.0 <= weight <= 1.0


def test_learning_repeated_transitions_strengthen() -> None:
    """Repeated same transition strengthens the weight over time."""
    model = CognitiveTrajectoryModel()
    model._precision = 1.0
    for _ in range(10):
        model.predict({"cats": 0.8})
        model.compute_surprise({"cats": 0.8, "dogs": 0.6})

    weight_after = model._transition_weights["cats"]["dogs"]
    # Should be positive (cats predicts dogs)
    assert weight_after > 0.0


# ═══════════════════════════════════════════════════════════════════
# Precision adaptation
# ═══════════════════════════════════════════════════════════════════


def test_precision_decays_on_sustained_surprise() -> None:
    """Precision decreases when surprise EMA exceeds threshold."""
    model = CognitiveTrajectoryModel()
    initial_precision = model.precision
    # Force high surprise for many cycles (total mismatch)
    for _ in range(20):
        model.predict({"a": 1.0})
        model.compute_surprise({"z": 0.7})
    assert model.precision < initial_precision


def test_precision_recovers_on_sustained_low_surprise() -> None:
    """Precision increases when surprise EMA is below threshold."""
    model = CognitiveTrajectoryModel()
    # First drop precision with high surprise
    for _ in range(20):
        model.predict({"a": 1.0})
        model.compute_surprise({"z": 0.7})
    low_precision = model.precision
    # Then recover with perfect predictions (actual = decayed prediction)
    for _ in range(100):
        model.predict({"a": 1.0})
        model.compute_surprise({"a": _DECAY})  # matches predicted 0.7
    assert model.precision > low_precision


def test_precision_bounded() -> None:
    """Precision stays in [0.1, 1.0]."""
    model = CognitiveTrajectoryModel()
    # Force maximum surprise
    for _ in range(1000):
        model.predict({"a": 1.0})
        model.compute_surprise({"z": 1.0})
    assert model.precision >= 0.1
    # Force minimum surprise
    for _ in range(1000):
        model.predict({"a": 1.0})
        model.compute_surprise({"a": _DECAY})
    assert model.precision <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Surprise EMA
# ═══════════════════════════════════════════════════════════════════


def test_surprise_ema_smooths() -> None:
    """Surprise EMA smooths over cycles — a low-surprise cycle after a
    high-surprise cycle has lower EMA than the high cycle alone."""
    model = CognitiveTrajectoryModel()
    # High surprise cycle
    model.predict({"a": 1.0})
    reading1 = model.compute_surprise({"z": 0.7})
    # Low surprise cycle (actual matches decayed prediction)
    model.predict({"a": 1.0})
    reading2 = model.compute_surprise({"a": _DECAY})
    # The EMA should have moved toward the lower surprise
    assert reading2.surprise < reading1.surprise
    assert reading2.surprise_ema < reading1.surprise


def test_surprise_ema_bounded_01() -> None:
    """Surprise EMA stays in [0, 1]."""
    model = CognitiveTrajectoryModel()
    for _ in range(100):
        model.predict({"a": 1.0})
        model.compute_surprise({"z": 1.0})
    assert 0.0 <= model.surprise_ema <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Reading properties
# ═══════════════════════════════════════════════════════════════════


def test_reading_is_surprised() -> None:
    """is_surprised returns True when surprise_ema exceeds threshold."""
    model = CognitiveTrajectoryModel()
    for _ in range(20):
        model.predict({"a": 1.0})
        model.compute_surprise({"z": 0.7})  # total mismatch
    assert model.last_reading is not None
    assert model.last_reading.is_surprised


def test_reading_not_surprised_when_stable() -> None:
    """is_surprised returns False when surprise_ema is low."""
    model = CognitiveTrajectoryModel()
    # Perfect predictions: actual = decayed prediction → surprise = 0
    for _ in range(50):
        model.predict({"a": 1.0})
        model.compute_surprise({"a": _DECAY})
    assert model.last_reading is not None
    assert not model.last_reading.is_surprised


# ═══════════════════════════════════════════════════════════════════
# No NaN/inf
# ═══════════════════════════════════════════════════════════════════


def test_no_nan_after_many_cycles() -> None:
    """No NaN/inf after many cycles of mixed surprise."""
    model = CognitiveTrajectoryModel()
    topics_a = {"cats": 0.7, "dogs": 0.5}
    topics_b = {"quantum": 0.8, "physics": 0.6}
    for i in range(200):
        topics = topics_a if i % 2 == 0 else topics_b
        model.predict(topics)
        model.compute_surprise(topics)
    assert math.isfinite(model.surprise_ema)
    assert math.isfinite(model.precision)
    assert math.isfinite(model.cycle_count)


# ═══════════════════════════════════════════════════════════════════
# Full cycle: predict → surprise → predict
# ═══════════════════════════════════════════════════════════════════


def test_full_cycle_predict_surprise_predict() -> None:
    """A full predict→surprise→predict cycle produces consistent state."""
    model = CognitiveTrajectoryModel()
    # Turn 1: predict (empty), observe content → surprise (unpredicted)
    model.predict({})
    r1 = model.compute_surprise({"cats": 0.8})
    assert r1.surprise == pytest.approx(0.8)  # everything is unexpected
    # Turn 1 end: predict next from current topics
    next_pred = model.predict({"cats": 0.8})
    assert "cats" in next_pred
    assert next_pred["cats"] == pytest.approx(0.8 * _DECAY)
    # Turn 2: observe (matches decayed prediction) → low surprise
    r2 = model.compute_surprise({"cats": 0.8 * _DECAY})
    assert r2.surprise == pytest.approx(0.0)
    assert r2.cycle_count == 2


# ═══════════════════════════════════════════════════════════════════
# Persistence — to_dict / load_from_dict
# ═══════════════════════════════════════════════════════════════════


def test_persistence_round_trip() -> None:
    """to_dict → load_from_dict preserves learned state."""
    model = CognitiveTrajectoryModel()
    # Learn some transitions
    for _ in range(10):
        model.predict({"cats": 0.8})
        model.compute_surprise({"cats": 0.8, "dogs": 0.6})
    assert model.cycle_count == 10
    assert model.surprise_ema > 0.0
    original_precision = model.precision
    original_weights = {
        src: dict(targets)
        for src, targets in model._transition_weights.items()
    }

    # Serialize
    data = model.to_dict()
    assert data["version"] == 1
    assert data["cycle_count"] == 10

    # Deserialize into a new model
    restored = CognitiveTrajectoryModel()
    restored.load_from_dict(data)
    assert restored.cycle_count == 10
    assert restored.surprise_ema == model.surprise_ema
    assert restored.precision == original_precision
    assert restored._transition_weights == original_weights


def test_persistence_backward_compatible() -> None:
    """load_from_dict with empty dict leaves model at defaults."""
    model = CognitiveTrajectoryModel()
    model.load_from_dict({})
    assert model.cycle_count == 0
    assert model.surprise_ema == 0.0
    assert model.precision == 0.5
    assert model._transition_weights == {}


def test_persistence_resets_predicted_topics() -> None:
    """load_from_dict resets predicted_topics (transient, not persisted)."""
    model = CognitiveTrajectoryModel()
    model.predict({"cats": 0.7})
    assert model._predicted_topics != {}
    data = model.to_dict()
    # to_dict doesn't include predicted_topics
    assert "predicted_topics" not in data
    # load_from_dict resets them
    model.load_from_dict(data)
    assert model._predicted_topics == {}
    assert model.last_reading is None
