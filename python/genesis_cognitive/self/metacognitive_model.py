"""Recursive metacognitive model — Genesis models its own cognitive processes.

This is the recursive self-modeling layer that sits on top of the
reflection engine. Where the Rust active-inference engine
(``src/daemon/active_inference.rs``) is a generative model of its
neurochemical trajectory, this is a generative model of its
cognitive trajectory: it predicts what its own reflection will
discover before it runs, compares the prediction to what actually
happens, learns from the error, and feeds the *metacognitive
surprise* back into cognition.

# Why this is recursive self-modeling

The model is a stack of levels, each predicting the level below:

- **Level 0** (object): the actual cognitive-process outcomes — did
  reflection produce an insight? what type? what confidence? This is
  observed, not predicted.
- **Level 1** (meta): a generative model predicting level-0 outcomes
  from pre-reflection features. Its prediction error is
  *metacognitive surprise* — "I didn't expect to learn that about
  myself."
- **Level 2** (meta-meta): a generative model predicting level-1's
  metacognitive surprise. "I didn't expect to be surprised about my
  own cognition."
- **Level N**: predicts level N-1's prediction error.

The model models itself modeling the system. The recursion is real,
not faked with fixed multiplicative discounts: each level is a
learned linear predictor that adapts via the delta rule, and each
level's prediction error is the training signal for the level above.

# Self-termination (as deep as useful)

The tower is not fixed at 3 levels and is not unbounded. It
self-terminates:

- A new level is **spawned** when the current top level's EMA
  prediction error exceeds ``SPAWN_THRESHOLD`` for a sustained period
  (``SPAWN_PERSISTENCE`` observations). The system can't predict its
  own metacognition well → it builds a model of that unpredictability.
- A level is **pruned** when its own prediction error EMA drops below
  ``PRUNE_THRESHOLD`` (it has become predictable → the level above it
  adds no information).
- Each level's learning rate is smaller and its precision decays
  faster, so higher levels track slower and are more uncertain.
  Diminishing returns are structural: precision floors prevent
  infinite stacking, and the finite feature space means a higher
  level eventually can't extract more signal.

In practice the tower settles at 2–4 levels: level 1 is almost always
useful (predicting reflection outcomes), level 2 is often useful
(predicting when it'll be surprised by itself), and higher levels
appear only during periods of genuine cognitive unpredictability.

# The feedback loop

This is what makes it recursive self-modeling rather than passive
prediction:

1. Before reflection: predict the reflection outcome.
2. Reflection runs → actual outcomes.
3. Compare → metacognitive feedback (per-target errors + aggregate
   surprise).
4. Update each level's model via delta rule.
5. Possibly spawn/prune a level.
6. Feed metacognitive surprise back into cognition:
   - High surprise → deeper reflection next time (it is unpredictable
     to itself → pay attention).
   - Sustained low surprise → lighter reflection (it understands
     itself → don't waste effort).
   - Pre-reflection prediction modulates response caution: if a
     self-correction is predicted, pre-hedge.

# Neuroscience grounding

This implements the Nelson & Narens (1990) meta-level/object-level
model as a *learned* system, not a static annotation. The
meta-level monitors and controls the object-level; here the
meta-level is a generative model whose prediction errors drive
control (deeper vs. lighter reflection, response caution). The
recursive stacking follows the hierarchical predictive-coding
principle (Friston, 2008): each level predicts the level below, and
prediction errors propagate upward to train the next level.

The self-termination mechanism is grounded in the observation that
human metacognition is not a fixed tower — the depth of
metacognitive scrutiny adapts to the situation. We don't normally
think about whether we know that we know that we know our name; we
do engage deeper metacognitive scrutiny when our first-order
cognition is failing in ways we didn't expect. The spawn/prune
mechanism captures this adaptive depth.

References:
- Nelson, T. O. & Narens, L. (1990). Metamemory: a theoretical
  framework and new findings. Perspectives on Psychological Science.
- Friston, K. (2008). Hierarchical models in the brain. PLoS
  Computational Biology.
- Koriat, A. (2007). Metacognition and cognition. Higher-Order
  Thinking, Metacognition, and Self-Regulated Learning.
- Flavell, J. H. (1979). Metacognition and cognitive monitoring.
  American Psychologist.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..cognition import CognitiveState
    from .reflection import Insight

__all__ = [
    "CognitiveProcessModel",
    "MetacognitiveFeedback",
    "MetacognitivePrediction",
]

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────

# The number of pre-reflection features. See ``_extract_features``.
# 19 intent one-hot + 1 confidence + 1 valence + 1 arousal + 5 brain
# waves + 3 recent rates + 1 maturity + 1 caution = 32.
NUM_FEATURES: int = 32

# The number of predicted targets. See ``MetacognitivePrediction``.
NUM_TARGETS: int = 5

# Learning rate for level 1 (the object-level predictor). Higher =
# faster adaptation, noisier. Lower = slower, more stable.
BASE_LEARNING_RATE: float = 0.05

# Per-level learning-rate decay. Each higher level learns slower,
# because it models a slower signal (the prediction error EMA of the
# level below) and because higher-order metacognition should be more
# conservative. ``level_lr(n) = BASE_LEARNING_RATE * (DECAY ** n)``.
LEVEL_LR_DECAY: float = 0.6

# Base precision for level 1. Precision is the model's confidence in
# its own predictions, in [0, 1]. Higher = trusts predictions more.
BASE_PRECISION: float = 0.5

# Per-level precision decay. Higher levels start less confident and
# recover confidence slower, modeling the compounding uncertainty of
# recursion ("uncertainty about uncertainty").
LEVEL_PRECISION_DECAY: float = 0.7

# Rate at which a level's precision increases when its surprise is
# low (it's predicting well). Per observation. Slow — regaining
# confidence takes time.
PRECISION_RECOVERY_RATE: float = 0.008

# Rate at which a level's precision decreases when its surprise is
# high. Faster than recovery — losing confidence is quick, regaining
# it is slow (asymmetric, like receptor adaptation). But not so fast
# that the model traps itself: if precision drops before the model
# can learn, it never recovers.
PRECISION_DECAY_RATE: float = 0.015

# Surprise threshold above which precision decays, below which it
# recovers. Separates "predicting myself well" from "failing to
# predict myself."
PRECISION_SURPRISE_THRESHOLD: float = 0.2

# EMA decay for per-level surprise. Lower = longer memory of past
# metacognitive surprise; higher = more reactive to recent surprise.
SURPRISE_EMA_DECAY: float = 0.2

# ─── Self-termination thresholds ──────────────────────────────────

# A new level is spawned when the top level's surprise EMA exceeds
# this for ``SPAWN_PERSISTENCE`` consecutive observations. The system
# is failing to predict its own metacognition → build a model of that
# unpredictability.
SPAWN_THRESHOLD: float = 0.25

# How many consecutive high-surprise observations are required before
# spawning a level. Prevents spawning from transient spikes.
SPAWN_PERSISTENCE: int = 10

# A level is pruned when its surprise EMA drops below this. It has
# become predictable → the level above it adds no information.
PRUNE_THRESHOLD: float = 0.08

# Minimum number of observations a level must accumulate before it's
# eligible for pruning. Prevents pruning a freshly-spawned level
# before it has a chance to learn.
PRUNE_MIN_AGE: int = 30

# Hard cap on the number of levels. A safety bound against
# pathological growth; in practice the spawn/prune logic keeps the
# tower at 2–4 levels.
MAX_LEVELS: int = 6

# Minimum precision. Prevents a level from losing all confidence,
# which would make its predictions uninformative and its error
# signal useless to the level above.
MIN_PRECISION: float = 0.1


# ─── Insight-type vocabulary ──────────────────────────────────────

# The insight types the model predicts, in a fixed order. The order
# is arbitrary but must be stable across save/load.
INSIGHT_TYPES: tuple[str, ...] = (
    "self_correction",
    "pattern",
    "gap",
    "growth",
    "mood",
)


# ─── Data classes ─────────────────────────────────────────────────


@dataclass(slots=True)
class MetacognitivePrediction:
    """A prediction about the outcome of the next reflection.

    All fields are in [0, 1] unless noted. These are the model's
    *expectations* before reflection runs — what level 1 of the
    recursive model predicts will happen.
    """

    # Probability that reflection will produce at least one insight.
    p_insight: float = 0.5
    # Probability distribution over insight types (sums to ~1).
    p_insight_type: dict[str, float] = field(default_factory=dict)
    # Expected mean confidence of produced insights.
    expected_confidence: float = 0.5
    # Expected number of insights (can exceed 1).
    expected_count: float = 0.5
    # The model's confidence in its own prediction (level-1 precision).
    precision: float = 0.5
    # How many levels the recursive model currently has.
    depth: int = 1
    # The level-1 surprise EMA at prediction time — how unpredictable
    # its own cognition has been recently.
    self_surprise: float = 0.0


@dataclass(slots=True)
class MetacognitiveFeedback:
    """The result of comparing a prediction to the actual outcome.

    This is the training signal for the recursive model and the
    feedback signal for cognition. ``metacognitive_surprise`` is the
    key field: it's how surprised the model was by its own cognitive
    process, aggregated across all levels. High → it is unpredictable
    to itself → pay more attention.
    """

    # Per-target prediction errors, in [-1, 1].
    error_insight: float = 0.0
    error_confidence: float = 0.0
    error_count: float = 0.0
    # Distribution-level error (mean absolute error over insight-type
    # probabilities), in [0, 1].
    error_type_distribution: float = 0.0
    # Aggregate level-1 surprise (RMS of per-target errors), [0, 1].
    level1_surprise: float = 0.0
    # Aggregate metacognitive surprise across all levels, [0, 1].
    # This is the feedback signal: high → deeper reflection next time.
    metacognitive_surprise: float = 0.0
    # Whether a level was spawned or pruned this cycle.
    level_spawned: bool = False
    level_pruned: bool = False
    # Current depth after any spawn/prune.
    depth: int = 1
    # Whether the model predicted a self-correction was likely. The
    # cognition engine can use this to pre-hedge.
    predicted_self_correction: bool = False
    # Timestamp for logging.
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


# ─── A single level of the recursive model ───────────────────────


@dataclass(slots=True)
class _MetacognitiveLevel:
    """One level of the recursive metacognitive model.

    A level is a linear predictor: ``predicted = weights @ features + bias``.
    Level 0 predicts the cognitive-process outcomes from pre-reflection
    features. Level N>0 predicts level N-1's surprise from the same
    features. Each level learns via the delta rule and maintains its
    own precision and surprise EMA.
    """

    # The level index: 0 = object-level (predicts outcomes), 1 = meta
    # (predicts level-0 surprise), 2 = meta-meta, etc.
    level: int
    # Weight matrix: weights[target][feature]. Shape [NUM_TARGETS][NUM_FEATURES].
    # For level 0, targets are the cognitive outcomes. For level N>0,
    # target 0 is the level-(N-1) surprise and the rest are unused
    # (kept for uniform shape).
    weights: list[list[float]]
    # Bias vector: bias[target]. Shape [NUM_TARGETS].
    bias: list[float]
    # Precision: confidence in this level's predictions, [0, 1].
    precision: float
    # EMA of this level's surprise (prediction error), [0, 1].
    surprise_ema: float
    # Number of observations this level has seen. Used for pruning
    # eligibility and maturity.
    observations: int
    # Consecutive observations where surprise_ema > SPAWN_THRESHOLD.
    # When this reaches SPAWN_PERSISTENCE, a new level is spawned.
    consecutive_high_surprise: int

    def predict(self, features: list[float]) -> list[float]:
        """Predict the targets for this level from features.

        Returns a list of NUM_TARGETS floats. For level 0, these are
        the predicted cognitive outcomes. For level N>0, target 0 is
        the predicted level-(N-1) surprise; the rest are unused.
        """
        out = [0.0] * NUM_TARGETS
        for t in range(NUM_TARGETS):
            s = self.bias[t]
            row = self.weights[t]
            for f in range(NUM_FEATURES):
                s += row[f] * features[f]
            out[t] = s
        return out

    def update(
        self,
        features: list[float],
        targets: list[float],
        lr: float,
    ) -> float:
        """Update this level's model via the delta rule.

        Returns the RMS prediction error (surprise) for this level,
        in [0, 1].
        """
        predicted = self.predict(features)
        errors = [targets[t] - predicted[t] for t in range(NUM_TARGETS)]
        # Update weights and bias: w += lr * error * feature
        for t in range(NUM_TARGETS):
            e = errors[t]
            row = self.weights[t]
            for f in range(NUM_FEATURES):
                row[f] += lr * e * features[f]
            self.bias[t] += lr * e * 0.5
        # Clamp weights/bias to prevent runaway growth.
        for t in range(NUM_TARGETS):
            row = self.weights[t]
            for f in range(NUM_FEATURES):
                if row[f] > 2.0:
                    row[f] = 2.0
                elif row[f] < -2.0:
                    row[f] = -2.0
            if self.bias[t] > 0.5:
                self.bias[t] = 0.5
            elif self.bias[t] < -0.5:
                self.bias[t] = -0.5

        # RMS error, normalized to [0, 1].
        sq_sum = sum(e * e for e in errors)
        rms = math.sqrt(sq_sum / NUM_TARGETS)
        if rms < 0.0:
            rms = 0.0
        elif rms > 1.0:
            rms = 1.0
        return rms

    def update_precision(self, surprise: float, dt_scale: float = 1.0) -> None:
        """Adapt precision based on sustained surprise.

        Sustained low surprise → precision increases (trust predictions).
        Sustained high surprise → precision decreases (lose confidence).
        """
        if self.surprise_ema > PRECISION_SURPRISE_THRESHOLD:
            self.precision -= PRECISION_DECAY_RATE * dt_scale
        else:
            self.precision += PRECISION_RECOVERY_RATE * dt_scale
        if self.precision < MIN_PRECISION:
            self.precision = MIN_PRECISION
        elif self.precision > 1.0:
            self.precision = 1.0

    def update_surprise_ema(self, surprise: float) -> None:
        """Update the EMA of this level's surprise."""
        alpha = SURPRISE_EMA_DECAY
        self.surprise_ema = self.surprise_ema * (1.0 - alpha) + surprise * alpha
        if self.surprise_ema < 0.0:
            self.surprise_ema = 0.0
        elif self.surprise_ema > 1.0:
            self.surprise_ema = 1.0


def _new_level(level: int) -> _MetacognitiveLevel:
    """Create a fresh level with identity-ish initialization.

    Weights start at zero (predict the mean) rather than identity,
    because the targets are bounded [0,1] outcomes, not state vectors.
    Bias starts at the prior mean of each target (0.5 for outcomes,
    0.0 for surprise).
    """
    weights = [[0.0] * NUM_FEATURES for _ in range(NUM_TARGETS)]
    # Level 0 predicts outcomes centered around 0.5; higher levels
    # predict surprise centered around 0.
    bias = [0.5 if level == 0 else 0.0] * NUM_TARGETS
    precision = max(
        MIN_PRECISION,
        BASE_PRECISION * (LEVEL_PRECISION_DECAY ** level),
    )
    return _MetacognitiveLevel(
        level=level,
        weights=weights,
        bias=bias,
        precision=precision,
        surprise_ema=0.0,
        observations=0,
        consecutive_high_surprise=0,
    )


# ─── Feature extraction ───────────────────────────────────────────


def _outcome_rate(
    outcome_history: deque[tuple[bool, bool, bool]] | None,
    field_index: int,
    default: float,
) -> float:
    """Compute the rate of a given outcome field from history.

    Returns the fraction of recent interactions that produced the
    given outcome, or a neutral prior if no history exists.
    """
    if outcome_history and len(outcome_history) > 0:
        return sum(1 for o in outcome_history if o[field_index]) / len(outcome_history)
    return default


def _extract_features(
    state: CognitiveState,
    outcome_history: deque[tuple[bool, bool, bool]] | None = None,
    cycle_count: int = 0,
) -> list[float]:
    """Extract pre-reflection features from the cognitive state.

    Returns a fixed-length list of NUM_FEATURES floats, all in [0, 1]
    or [-1, 1]. The features are chosen to be predictive of
    reflection outcomes:

    - Intent one-hot (19 dims): what kind of speech act is this?
      Questions more likely to reveal gaps; corrections more likely
      to produce self-corrections.
    - Thought confidence: low confidence → more likely to produce
      gap insights.
    - Emotion valence/arousal: negative valence → more self-corrections;
      high arousal → more mood insights.
    - Brain-wave dominant band one-hot (5 dims): gamma → deeper
      reflection; delta → minimal reflection.
    - Recent insight rate, self-correction rate, gap rate: computed
      from the model's rolling outcome history (base rates).
    - Interaction count (maturity): log-scaled; early interactions
      are noisier.
    - Error-monitor caution: high caution → more self-monitoring.

    The feature vector is dense and bounded, suitable for a linear
    predictor with delta-rule learning.
    """
    features: list[float] = [0.0] * NUM_FEATURES
    idx = 0

    # Intent one-hot (19 dims). We use the Intent enum values in a
    # fixed order. The Perception.intent is an Intent enum.
    from ..perception import Intent

    intent_order = list(Intent)
    perception = state.perception
    for i, intent in enumerate(intent_order):
        if idx + i < NUM_FEATURES:
            features[idx + i] = 1.0 if perception.intent == intent else 0.0
    idx += len(intent_order)

    # Thought confidence [0, 1].
    if idx < NUM_FEATURES:
        features[idx] = max(0.0, min(1.0, state.thought.confidence))
        idx += 1

    # Emotion valence [-1, 1] → [0, 1].
    if idx < NUM_FEATURES:
        features[idx] = (state.emotion.valence + 1.0) / 2.0
        idx += 1

    # Emotion arousal [0, 1].
    if idx < NUM_FEATURES:
        features[idx] = max(0.0, min(1.0, state.emotion.arousal))
        idx += 1

    # Brain-wave dominant band one-hot (5 dims).
    from ..brain_waves import BrainWave

    wave = state.brain_waves
    bands = [BrainWave.DELTA, BrainWave.THETA, BrainWave.ALPHA, BrainWave.BETA, BrainWave.GAMMA]
    for i, band in enumerate(bands):
        if idx + i < NUM_FEATURES:
            features[idx + i] = 1.0 if (wave is not None and wave.dominant == band) else 0.0
    idx += len(bands)

    # Recent insight rate [0, 1] — fraction of recent interactions
    # that produced at least one insight. Computed from the model's
    # rolling outcome history. Before any history exists, use a
    # neutral prior.
    if idx < NUM_FEATURES:
        features[idx] = _outcome_rate(outcome_history, 0, 0.3)
        idx += 1

    # Recent self-correction rate [0, 1].
    if idx < NUM_FEATURES:
        features[idx] = _outcome_rate(outcome_history, 1, 0.1)
        idx += 1

    # Recent gap rate [0, 1].
    if idx < NUM_FEATURES:
        features[idx] = _outcome_rate(outcome_history, 2, 0.1)
        idx += 1

    # Interaction count (maturity) — log-scaled to [0, 1]. Early
    # interactions are noisier; the model can learn to discount its
    # own predictions when it's young.
    if idx < NUM_FEATURES:
        features[idx] = min(1.0, math.log1p(cycle_count) / math.log(100.0))
        idx += 1

    # Error-monitor caution [0, 1].
    caution = state.error_monitor_caution
    if idx < NUM_FEATURES:
        features[idx] = max(0.0, min(1.0, caution)) if caution is not None else 0.0
        idx += 1

    # Pad remaining slots with 0 (reserved for future features).
    return features


def _insight_targets(insights: list[Insight]) -> list[float]:
    """Convert actual reflection outcomes into target values.

    Returns a list of NUM_TARGETS floats:
    - target 0: p_insight (1.0 if any insight, 0.0 otherwise)
    - target 1: p_self_correction (fraction of insights that are
      self_corrections, or 0 if no insights)
    - target 2: mean confidence of insights (or 0.5 if none)
    - target 3: normalized insight count (count / 5, clamped to [0,1])
    - target 4: p_gap (fraction of insights that are gaps, or 0)
    """
    if not insights:
        return [0.0, 0.0, 0.5, 0.0, 0.0]

    count = len(insights)
    p_insight = 1.0
    n_self_correction = sum(1 for i in insights if i.type == "self_correction")
    n_gap = sum(1 for i in insights if i.type == "gap")
    p_self_correction = n_self_correction / count
    p_gap = n_gap / count
    mean_conf = sum(i.confidence for i in insights) / count
    norm_count = min(1.0, count / 5.0)
    return [
        p_insight,
        p_self_correction,
        mean_conf,
        norm_count,
        p_gap,
    ]


def _type_distribution(insights: list[Insight]) -> dict[str, float]:
    """Compute the empirical distribution over insight types."""
    if not insights:
        return dict.fromkeys(INSIGHT_TYPES, 0.0)
    counts: dict[str, int] = dict.fromkeys(INSIGHT_TYPES, 0)
    for i in insights:
        if i.type in counts:
            counts[i.type] += 1
    total = len(insights)
    return {t: counts[t] / total for t in INSIGHT_TYPES}


def _distribution_error(
    predicted: dict[str, float],
    actual: dict[str, float],
) -> float:
    """Mean absolute error between two distributions over insight types."""
    total = 0.0
    for t in INSIGHT_TYPES:
        p = predicted.get(t, 0.0)
        a = actual.get(t, 0.0)
        total += abs(p - a)
    return total / len(INSIGHT_TYPES)


# ─── The recursive model ──────────────────────────────────────────


class CognitiveProcessModel:
    """The recursive, self-terminating generative model of cognition.

    This is the recursive self-modeling layer. It predicts the
    outcome of Genesis's own reflection process before it runs,
    learns from the prediction error, and feeds the metacognitive
    surprise back into cognition. The model is a stack of levels,
    each predicting the level below, that grows and shrinks based on
    whether higher-order modeling is useful.

    Lifecycle per reflection cycle:

    1. ``predict(state)`` → ``MetacognitivePrediction`` (before
       reflection runs).
    2. Reflection runs, producing ``insights``.
    3. ``update(state, insights, prediction)`` → ``MetacognitiveFeedback``
       (after reflection runs).
    4. The cognition engine reads ``feedback`` to modulate the next
       cycle (caution, reflection depth).
    """

    def __init__(self) -> None:
        """Initialize with a single object-level metacognitive predictor."""
        # The stack of levels. Index 0 is the object-level predictor.
        # Always has at least one level.
        self._levels: list[_MetacognitiveLevel] = [_new_level(0)]
        # The most recent feedback, for the cognition engine to read.
        self.feedback: MetacognitiveFeedback | None = None
        # The most recent prediction, kept for logging/introspection.
        self.last_prediction: MetacognitivePrediction | None = None
        # Rolling history of level-1 surprise for spawn/prune decisions.
        self._surprise_history: deque[float] = deque(maxlen=50)
        # Rolling history of recent reflection outcomes, for feature
        # extraction. Each entry is (had_insight, had_self_correction,
        # had_gap). This lets the model compute recent rates without
        # needing a reference to the reflection engine.
        self._outcome_history: deque[tuple[bool, bool, bool]] = deque(maxlen=20)
        # Total number of reflection cycles processed.
        self.cycle_count: int = 0

    # ── Public API ───────────────────────────────────────────────

    def predict(self, state: CognitiveState) -> MetacognitivePrediction:
        """Predict the outcome of the next reflection.

        Called before ``ReflectionEngine.reflect()`` runs. Uses the
        level-0 (object-level) model to predict cognitive outcomes
        from pre-reflection features.
        """
        features = _extract_features(state, self._outcome_history, self.cycle_count)
        level0 = self._levels[0]
        raw = level0.predict(features)

        # Map raw predictions to the MetacognitivePrediction fields.
        # raw[0] = p_insight, raw[1] = p_self_correction,
        # raw[2] = mean_confidence, raw[3] = norm_count, raw[4] = p_gap
        # Map raw predictions to [0, 1] via clamping. We use clamping
        # (not sigmoid) so the prediction space matches the target
        # space: the model learns to predict raw=0.0 for "no insight"
        # and raw=1.0 for "insight", and clamping preserves those
        # values. Sigmoid would map raw=0.0 → 0.5, making it impossible
        # to predict values near 0 or 1.
        p_insight = _clamp01(raw[0])
        p_self_correction = _clamp01(raw[1])
        p_gap = _clamp01(raw[4])
        expected_confidence = _clamp01(raw[2])
        expected_count = max(0.0, raw[3]) * 5.0  # un-normalize

        # Build the insight-type distribution from the per-type
        # probabilities. We have p_self_correction and p_gap directly;
        # the remaining mass is split among pattern/growth/mood using
        # a learned prior (uniform for now; the model learns via the
        # bias terms).
        remaining = max(0.0, 1.0 - p_self_correction - p_gap)
        p_type = {
            "self_correction": p_self_correction,
            "gap": p_gap,
            "pattern": remaining * 0.4,
            "growth": remaining * 0.4,
            "mood": remaining * 0.2,
        }
        # Normalize to sum to 1.
        total = sum(p_type.values())
        if total > 0:
            p_type = {k: v / total for k, v in p_type.items()}

        # Aggregate self-surprise across all levels above 0.
        self_surprise = 0.0
        if len(self._levels) > 1:
            self_surprise = self._levels[1].surprise_ema

        pred = MetacognitivePrediction(
            p_insight=p_insight,
            p_insight_type=p_type,
            expected_confidence=expected_confidence,
            expected_count=expected_count,
            precision=level0.precision,
            depth=len(self._levels),
            self_surprise=self_surprise,
        )
        self.last_prediction = pred
        return pred

    def _update_levels(self, features: list[float], targets: list[float]) -> None:
        """Update level 0 and all higher levels via the delta rule.

        Learning rate is NOT multiplied by precision. Precision is a
        confidence signal (how much to trust predictions), not a
        learning gate. Coupling them creates a trap: when the model
        starts wrong, precision drops, which slows learning, which
        keeps the model wrong. Decoupling them lets the model learn
        at a constant rate while precision tracks confidence
        independently.

        Each level N>0 predicts the surprise of level N-1. The
        "target" for level N is the surprise of level N-1, placed
        in target slot 0.
        """
        # ── Update level 0 (object-level) ──
        level0 = self._levels[0]
        lr0 = BASE_LEARNING_RATE
        surprise0 = level0.update(features, targets, lr0)
        level0.observations += 1
        level0.update_surprise_ema(surprise0)
        level0.update_precision(surprise0)

        # ── Update higher levels ──
        for n in range(1, len(self._levels)):
            level = self._levels[n]
            below = self._levels[n - 1]
            # The target for this level is the surprise of the level
            # below, in slot 0. Other slots are unused (set to the
            # level's own prediction so error is 0 — no learning on
            # unused slots).
            level_targets = level.predict(features)
            level_targets[0] = below.surprise_ema
            lr_n = BASE_LEARNING_RATE * (LEVEL_LR_DECAY ** n)
            surprise_n = level.update(features, level_targets, lr_n)
            level.observations += 1
            level.update_surprise_ema(surprise_n)
            level.update_precision(surprise_n)

    def _spawn_or_prune_levels(self) -> tuple[bool, bool]:
        """Spawn or prune metacognitive levels based on surprise.

        Returns (spawned, pruned).
        """
        spawned = False
        pruned = False
        top = self._levels[-1]
        if top.surprise_ema > SPAWN_THRESHOLD:
            top.consecutive_high_surprise += 1
        else:
            top.consecutive_high_surprise = 0

        if (
            top.consecutive_high_surprise >= SPAWN_PERSISTENCE
            and len(self._levels) < MAX_LEVELS
        ):
            new_lvl = _new_level(len(self._levels))
            self._levels.append(new_lvl)
            top.consecutive_high_surprise = 0
            spawned = True
            logger.debug(
                "metacognitive model spawned level %d (surprise %.3f)",
                len(self._levels) - 1,
                top.surprise_ema,
            )

        # Prune: if a non-top, non-base level has become predictable,
        # remove it. We prune the highest such level (closest to top)
        # to keep the tower coherent.
        if len(self._levels) > 1:
            for n in range(len(self._levels) - 1, 0, -1):
                lvl = self._levels[n]
                if (
                    lvl.observations >= PRUNE_MIN_AGE
                    and lvl.surprise_ema < PRUNE_THRESHOLD
                ):
                    self._levels.pop(n)
                    pruned = True
                    logger.debug(
                        "metacognitive model pruned level %d (surprise %.3f)",
                        n,
                        lvl.surprise_ema,
                    )
                    break
        return spawned, pruned

    def _compute_feedback(
        self,
        insights: list[Insight],
        prediction: MetacognitivePrediction,
        spawned: bool,
        pruned: bool,
    ) -> MetacognitiveFeedback:
        """Compute the metacognitive feedback signal for the cognition engine."""
        actual_p_insight = 1.0 if insights else 0.0
        actual_mean_conf = (
            sum(i.confidence for i in insights) / len(insights)
            if insights
            else 0.5
        )
        actual_count = len(insights)
        actual_type_dist = _type_distribution(insights)

        error_insight = actual_p_insight - prediction.p_insight
        error_confidence = actual_mean_conf - prediction.expected_confidence
        error_count = (actual_count / 5.0) - (prediction.expected_count / 5.0)
        error_type = _distribution_error(
            prediction.p_insight_type, actual_type_dist
        )

        # Level-1 surprise = RMS of per-target errors.
        level1_surprise = math.sqrt(
            (error_insight ** 2 + error_confidence ** 2 + error_count ** 2) / 3.0
        )
        level1_surprise = max(0.0, min(1.0, level1_surprise))

        # Aggregate metacognitive surprise across all levels: weighted
        # sum of each level's surprise EMA, with higher levels weighted
        # less (they model slower signals). This is the feedback
        # signal for cognition.
        if self._levels:
            total_weight = 0.0
            weighted_sum = 0.0
            for n, lvl in enumerate(self._levels):
                w = LEVEL_PRECISION_DECAY ** n
                weighted_sum += w * lvl.surprise_ema
                total_weight += w
            meta_surprise = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            meta_surprise = 0.0
        meta_surprise = max(0.0, min(1.0, meta_surprise))

        predicted_self_correction = (
            prediction.p_insight_type.get("self_correction", 0.0) > 0.3
        )

        return MetacognitiveFeedback(
            error_insight=error_insight,
            error_confidence=error_confidence,
            error_count=error_count,
            error_type_distribution=error_type,
            level1_surprise=level1_surprise,
            metacognitive_surprise=meta_surprise,
            level_spawned=spawned,
            level_pruned=pruned,
            depth=len(self._levels),
            predicted_self_correction=predicted_self_correction,
        )

    def update(
        self,
        state: CognitiveState,
        insights: list[Insight],
        prediction: MetacognitivePrediction,
    ) -> MetacognitiveFeedback:
        """Update the model after reflection completes.

        Compares the prediction to the actual outcomes, updates each
        level via the delta rule, adapts precision, and possibly
        spawns or prunes a level. Returns the feedback signal for the
        cognition engine.
        """
        self.cycle_count += 1
        features = _extract_features(state, self._outcome_history, self.cycle_count)
        targets = _insight_targets(insights)

        # Record the outcome for future feature extraction (recent
        # insight/self-correction/gap rates).
        had_insight = len(insights) > 0
        had_self_correction = any(i.type == "self_correction" for i in insights)
        had_gap = any(i.type == "gap" for i in insights)
        self._outcome_history.append((had_insight, had_self_correction, had_gap))

        self._update_levels(features, targets)
        spawned, pruned = self._spawn_or_prune_levels()
        fb = self._compute_feedback(insights, prediction, spawned, pruned)
        self.feedback = fb
        self._surprise_history.append(fb.level1_surprise)
        return fb

    # ── Introspection ────────────────────────────────────────────

    def depth(self) -> int:
        """Current recursion depth (number of levels)."""
        return len(self._levels)

    def precision(self) -> float:
        """Level-0 precision — confidence in cognitive predictions."""
        return self._levels[0].precision if self._levels else 0.0

    def self_surprise(self) -> float:
        """Aggregate metacognitive surprise (how unpredictable it is to itself)."""
        if self.feedback is not None:
            return self.feedback.metacognitive_surprise
        return 0.0

    def describe(self) -> str:
        """Human-readable description of the model state."""
        lines = [f"depth={len(self._levels)} cycles={self.cycle_count}"]
        for n, lvl in enumerate(self._levels):
            lines.append(
                f"  L{n}: precision={lvl.precision:.2f} "
                f"surprise_ema={lvl.surprise_ema:.3f} "
                f"obs={lvl.observations}"
            )
        if self.feedback is not None:
            lines.append(
                f"  feedback: meta_surprise={self.feedback.metacognitive_surprise:.3f}"
            )
        return "\n".join(lines)

    # ── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a dict for persistence."""
        return {
            "version": 1,
            "cycle_count": self.cycle_count,
            "outcome_history": list(self._outcome_history),
            "levels": [
                {
                    "level": lvl.level,
                    "weights": lvl.weights,
                    "bias": lvl.bias,
                    "precision": lvl.precision,
                    "surprise_ema": lvl.surprise_ema,
                    "observations": lvl.observations,
                    "consecutive_high_surprise": lvl.consecutive_high_surprise,
                }
                for lvl in self._levels
            ],
        }

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Restore the model from a serialized dict."""
        self.cycle_count = data.get("cycle_count", 0)
        # Restore outcome history (backward compatible: missing key → empty).
        history = data.get("outcome_history", [])
        self._outcome_history = deque(
            ((bool(o[0]), bool(o[1]), bool(o[2])) for o in history if len(o) >= 3),
            maxlen=20,
        )
        levels_data = data.get("levels", [])
        if not levels_data:
            # Corrupt or empty — start fresh.
            self._levels = [_new_level(0)]
            return
        self._levels = []
        for ld in levels_data:
            lvl = _MetacognitiveLevel(
                level=ld.get("level", len(self._levels)),
                weights=ld.get("weights", [[0.0] * NUM_FEATURES] * NUM_TARGETS),
                bias=ld.get("bias", [0.5] * NUM_TARGETS),
                precision=max(MIN_PRECISION, min(1.0, ld.get("precision", BASE_PRECISION))),
                surprise_ema=max(0.0, min(1.0, ld.get("surprise_ema", 0.0))),
                observations=ld.get("observations", 0),
                consecutive_high_surprise=ld.get("consecutive_high_surprise", 0),
            )
            self._levels.append(lvl)
        if not self._levels:
            self._levels = [_new_level(0)]


# ─── Helpers ──────────────────────────────────────────────────────


def _clamp01(x: float) -> float:
    """Clamp a value to [0, 1]. NaN-safe."""
    if x != x:  # NaN check
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x
