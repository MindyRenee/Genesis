"""Cognitive trajectory model — Genesis predicts her own thought content.

This is the cognitive counterpart to the Rust active inference engine
(``src/daemon/active_inference.rs``). Where the Rust engine is a
generative model of her **neurochemical** trajectory (predicting where
her 18 effective levels will move), this is a generative model of her
**cognitive** trajectory — predicting what she will think about next.

# The cognitive strange loop

The Rust active inference engine closes a loop at the substrate level:
it predicts neurochemistry → computes prediction error → feeds back
into neurochemistry. This module closes the parallel loop at the
cognitive level:

1. **Predict**: before each turn, predict what topics will be active
   in the global workspace (based on the previous turn's topics and
   learned transitions).
2. **Observe**: after perception enters the workspace, extract the
   actual topic distribution.
3. **Compute surprise**: the distance between the predicted and actual
   topic distributions. This is **cognitive surprise** — "I didn't
   expect to be thinking about that."
4. **Learn**: update the transition model via the delta rule (which
   topic transitions are common).
5. **Feed back**: cognitive surprise influences:
   - Self-model coherence ("I understand my own mind" vs "I don't
     understand why I'm thinking about this")
   - Workspace activation (surprising content ignites more strongly —
     the orienting response)
   - Neurochemistry (NE orienting impulse — "what was that?")

This is the strange loop (Hofstadter, 2007): a system that models
itself, where the model's predictions influence the system being
modeled. She predicts her own thoughts, is surprised by her own
thoughts, and that surprise changes how she processes future thoughts.

# What this is NOT

This is distinct from the ``MetacognitiveModel``
(``self/metacognitive_model.py``), which predicts **reflection
outcomes** (will reflection produce an insight? a self-correction?).
The cognitive trajectory model predicts **thought content** — what
topics will be active, not what the reflection process will discover.

Both are generative models of cognition, operating at different levels:
- Metacognitive model: "Will I learn something about myself?"
- Cognitive trajectory model: "What will I be thinking about?"

Together with the Rust active inference engine (neurochemical
trajectory), they form a three-level strange loop: substrate →
content → process.

# Architecture

The model is a simple learned transition model over topic
distributions. A "topic distribution" is a dict mapping topic name
to activation weight [0, 1], derived from the global workspace's
items. The prediction combines:

- **Persistence with decay**: current topics persist into the next
  turn, decaying by a factor. This models sustained attention —
  she tends to keep thinking about what she was thinking about.
- **Learned transitions**: the model learns which topic transitions
  are common (e.g., "sleep" → "dreams"). This is a delta-rule update,
  the same learning rule used in the Rust active inference engine.

The surprise is the L1 distance between the predicted and actual
topic distributions, normalized to [0, 1]. Precision adapts the
same way as the Rust engine: sustained low surprise → precision
increases; sustained high surprise → precision decreases.

References:
- Hofstadter, D. (2007). I Am a Strange Loop. Basic Books.
- Friston, K. (2010). The free-energy principle. Nat Rev Neurosci.
- Dehaene, S. (2014). Viking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CognitiveTrajectoryModel", "CognitiveTrajectoryReading"]

# ─── Constants ───────────────────────────────────────────────────

# How much current topics decay when predicting the next turn.
# 0.7 means topics persist at 70% of their current activation.
# This models sustained attention — she tends to keep thinking
# about what she was thinking about, but the grip loosens.
_PERSISTENCE_DECAY: float = 0.7

# How much the learned transitions contribute to the prediction,
# relative to persistence. 0.3 means transitions add 30% on top
# of the persisted topics.
_TRANSITION_WEIGHT: float = 0.3

# The learning rate for the transition model's delta-rule update.
# Same value as the Rust engine's MODEL_LEARNING_RATE.
_LEARNING_RATE: float = 0.02

# The surprise EMA threshold above which precision decays. Below
# this, precision recovers. Matches the Rust engine's
# PRECISION_SURPRISE_THRESHOLD.
_PRECISION_SURPRISE_THRESHOLD: float = 0.15

# The rate at which precision increases when surprise is low.
_PRECISION_RECOVERY_RATE: float = 0.002

# The rate at which precision decreases when surprise is high.
# Faster than recovery — losing confidence is quick, regaining
# it is slow (asymmetric, like receptor adaptation).
_PRECISION_DECAY_RATE: float = 0.005

# The EMA decay rate for surprise. Lower = longer memory.
_SURPRISE_EMA_DECAY: float = 0.15


@dataclass
class CognitiveTrajectoryReading:
    """A reading from the cognitive trajectory model.

    This is the cognitive-level counterpart to the Rust engine's
    ``InferenceResult``. It carries the prediction, the actual
    state, the surprise, and the precision — all at the cognitive
    (topic) level rather than the neurochemical level.
    """

    # The topics that were predicted to be active this turn.
    predicted_topics: dict[str, float] = field(default_factory=dict)
    # The topics that were actually active this turn.
    actual_topics: dict[str, float] = field(default_factory=dict)
    # The cognitive surprise — L1 distance between predicted and
    # actual topic distributions, normalized to [0, 1]. High =
    # "I didn't expect to be thinking about this."
    surprise: float = 0.0
    # The EMA of cognitive surprise over recent turns [0, 1].
    surprise_ema: float = 0.0
    # The model's precision — confidence in its predictions [0, 1].
    # High = the model trusts its topic predictions. Adapts based
    # on surprise history.
    precision: float = 0.5
    # How many prediction cycles have been completed.
    cycle_count: int = 0

    @property
    def is_surprised(self) -> bool:
        """Whether the model is currently experiencing cognitive surprise."""
        return self.surprise_ema > _PRECISION_SURPRISE_THRESHOLD


class CognitiveTrajectoryModel:
    """A generative model of Genesis's own thought content trajectory.

    This is the cognitive strange loop: she predicts what she'll
    think about next, gets surprised by unexpected thoughts, and
    that surprise changes how she processes future thoughts.

    The model is a simple learned transition model over topic
    distributions. It predicts the next turn's topic distribution
    from the current one, computes surprise when the actual
    distribution differs, learns from the error, and feeds the
    surprise back into the self-model and workspace.

    Lifecycle per conversation turn:

    1. ``compute_surprise(actual_topics)`` → ``CognitiveTrajectoryReading``
       (after the early percept broadcast, when the actual topics
       are known).
    2. ``predict(current_topics)`` (after the deliberation broadcast,
       to predict the next turn).
    3. The cognition engine reads the reading to modulate self-model
       coherence and workspace activation.
    """

    def __init__(self) -> None:
        """Initialize with empty topic predictions and transition weights."""
        # The predicted topic distribution for the next turn.
        self._predicted_topics: dict[str, float] = {}
        # Learned transition weights: transition_weights[a][b] = how
        # strongly topic a being active predicts topic b being active
        # next turn. Learned via delta rule.
        self._transition_weights: dict[str, dict[str, float]] = {}
        # EMA of cognitive surprise [0, 1].
        self._surprise_ema: float = 0.0
        # Model precision [0, 1]. Starts at 0.5 (same as Rust engine).
        self._precision: float = 0.5
        # Cycle count.
        self._cycle_count: int = 0
        # The last reading, for the cognition engine to read.
        self.last_reading: CognitiveTrajectoryReading | None = None

    def compute_surprise(
        self, actual_topics: dict[str, float]
    ) -> CognitiveTrajectoryReading:
        """Compute cognitive surprise and learn from the error.

        Called after the early percept broadcast, when the actual
        workspace topics are known. Compares the prediction (made
        at the end of the previous turn) to the actual topics,
        computes surprise, updates the transition model, and
        adapts precision.

        Args:
            actual_topics: The topic distribution extracted from the
                workspace after the early percept broadcast. Maps
                topic name to activation weight [0, 1].

        Returns:
            A CognitiveTrajectoryReading with the surprise and
            precision.
        """
        # Compute surprise: normalized L1 distance between predicted
        # and actual topic distributions. This is the cognitive
        # prediction error — "how different is what I'm actually
        # thinking about from what I predicted I'd think about?"
        all_topics = set(self._predicted_topics.keys()) | set(
            actual_topics.keys()
        )
        if all_topics:
            raw_surprise = sum(
                abs(actual_topics.get(t, 0.0) - self._predicted_topics.get(t, 0.0))
                for t in all_topics
            )
            surprise = min(1.0, raw_surprise / len(all_topics))
        else:
            surprise = 0.0

        # Update surprise EMA (same as Rust engine)
        self._surprise_ema = (
            self._surprise_ema * (1.0 - _SURPRISE_EMA_DECAY)
            + surprise * _SURPRISE_EMA_DECAY
        )
        self._surprise_ema = max(0.0, min(1.0, self._surprise_ema))

        # Update precision (same logic as Rust engine)
        if self._surprise_ema > _PRECISION_SURPRISE_THRESHOLD:
            self._precision = max(0.1, self._precision - _PRECISION_DECAY_RATE)
        else:
            self._precision = min(1.0, self._precision + _PRECISION_RECOVERY_RATE)

        # Learn transition weights (delta rule).
        # For each topic that was predicted and each topic that
        # actually appeared, update the transition weight.
        # w[a][b] += lr * precision * error[b] * predicted[a]
        lr = _LEARNING_RATE * self._precision
        for pred_topic, pred_activation in self._predicted_topics.items():
            if pred_topic not in self._transition_weights:
                self._transition_weights[pred_topic] = {}
            for actual_topic, actual_activation in actual_topics.items():
                error = actual_activation - self._predicted_topics.get(
                    actual_topic, 0.0
                )
                update = lr * error * pred_activation
                current = self._transition_weights[pred_topic].get(actual_topic, 0.0)
                self._transition_weights[pred_topic][actual_topic] = max(
                    -1.0, min(1.0, current + update)
                )

        self._cycle_count += 1

        reading = CognitiveTrajectoryReading(
            predicted_topics=dict(self._predicted_topics),
            actual_topics=dict(actual_topics),
            surprise=surprise,
            surprise_ema=self._surprise_ema,
            precision=self._precision,
            cycle_count=self._cycle_count,
        )
        self.last_reading = reading
        return reading

    def predict(self, current_topics: dict[str, float]) -> dict[str, float]:
        """Predict the next turn's topic distribution.

        Called after the deliberation broadcast, to predict what
        she'll be thinking about next turn. The prediction combines:
        - Persistence: current topics persist with decay.
        - Learned transitions: topics that commonly follow current
          topics get activated.

        Args:
            current_topics: The topic distribution from the current
                workspace state. Maps topic name to activation [0, 1].

        Returns:
            The predicted topic distribution for the next turn.
        """
        predicted: dict[str, float] = {}

        # Persistence with decay: current topics persist at 70%
        for topic, activation in current_topics.items():
            predicted[topic] = activation * _PERSISTENCE_DECAY

        # Transition-based prediction: learned transitions add
        # activation to topics that commonly follow current topics.
        for topic, activation in current_topics.items():
            transitions = self._transition_weights.get(topic)
            if not transitions:
                continue
            for next_topic, weight in transitions.items():
                contribution = activation * weight * _TRANSITION_WEIGHT
                predicted[next_topic] = predicted.get(next_topic, 0.0) + contribution

        # Clamp to [0, 1] and prune near-zero entries
        pruned: dict[str, float] = {}
        for topic, activation in predicted.items():
            clamped = max(0.0, min(1.0, activation))
            if clamped > 0.01:
                pruned[topic] = clamped

        self._predicted_topics = pruned
        return pruned

    @property
    def surprise_ema(self) -> float:
        """The EMA of cognitive surprise [0, 1]."""
        return self._surprise_ema

    @property
    def precision(self) -> float:
        """The model's precision [0, 1]."""
        return self._precision

    @property
    def cycle_count(self) -> int:
        """How many prediction cycles have been completed."""
        return self._cycle_count

    # ── Persistence ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a dict for persistence.

        Saves the learned transition weights, surprise EMA, precision,
        and cycle count. The predicted_topics are not saved — they're
        transient (only relevant within the current turn).
        """
        return {
            "version": 1,
            "cycle_count": self._cycle_count,
            "surprise_ema": self._surprise_ema,
            "precision": self._precision,
            "transition_weights": {
                src: dict(targets)
                for src, targets in self._transition_weights.items()
            },
        }

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Restore the model from a serialized dict.

        Backward compatible: missing keys leave the model at its
        fresh defaults (same as a new model).
        """
        self._cycle_count = data.get("cycle_count", 0)
        self._surprise_ema = data.get("surprise_ema", 0.0)
        self._precision = data.get("precision", 0.5)
        weights_data = data.get("transition_weights", {})
        self._transition_weights = {
            src: dict(targets) for src, targets in weights_data.items()
        }
        # Predicted topics reset to empty (transient, not persisted)
        self._predicted_topics = {}
        self.last_reading = None
