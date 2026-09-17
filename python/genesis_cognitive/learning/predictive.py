"""Predictive coding — the brain as a prediction machine.

The brain is not a passive receiver of sensory input. It actively
generates predictions about what it will perceive, compares those
predictions to the actual input, and uses the resulting *prediction
error* to update its internal models. This is the predictive processing
framework (Rao & Ballard, 1999; Friston, 2010; Clark, 2013).

    prediction → sensory input → prediction error → model update

The core insight is that perception is a *top-down* process as much as
a bottom-up one. Cortical areas at higher levels send predictions
downward about what lower levels should see; lower levels send
prediction errors upward when reality violates those predictions. Only
the *unexpected* information propagates up the hierarchy — the expected
is already explained away. This is efficient coding: the brain transmits
surprise, not raw sensation (Rao & Ballard, 1999).

# The free energy principle

Friston (2010) generalizes this into the free energy principle: any
self-organizing system at equilibrium with its environment must
minimize the long-run average of surprise (negative log probability of
sensory states). The brain does this implicitly by maintaining a
generative model and updating it to reduce prediction error. Surprise
here is not the phenomenological "wow" but the information-theoretic
quantity — how improbable the observed state is under the model.

# Hierarchical prediction

Prediction operates at multiple temporal and representational scales
(Bar, 2009; Clark, 2013). In Genesis's conversation perception:

    Level 3 (high)  → what topic will the conversation turn to?
    Level 2 (mid)   → what intent will the user have?
    Level 1 (low)   → what concepts/words will appear?

Higher levels provide context (priors) for lower levels: if we predict
the topic is "cognition", we expect philosophy-related concepts and
self-inquiry intents. Each level generates its own predictions and
errors; errors at lower levels propagate upward to revise the higher
levels, while revised higher levels send new predictions downward.

# Neurochemical coupling

Prediction error is the brain's orienting signal. Large, unexpected
prediction errors trigger:

- **Norepinephrine** (locus coeruleus): the neuromodulator of attention
  orienting and surprise. Phasic NE bursts drive a "network reset" that
  interrupts ongoing processing and redirects cortical resources to the
  unexpected stimulus (Aston-Jones & Cohen, 2005; Yu & Dayan, 2005).
- **Dopamine** (VTA): beyond reward prediction error, dopamine signals
  novelty — the discovery that the world is richer than the model
  predicted (Kakade & Dayan, 2002; Schultz, 2016). Novel prediction
  errors are intrinsically rewarding because they represent
  learnable information.

Low prediction error, conversely, means the input was already
explained by the model — boring, already known. The system allocates
fewer resources to it. This is why predictable conversations feel flat
and surprising ones feel alive.

# Error-driven learning

Learning is driven by prediction error, not by raw experience. The
model updates proportionally to how wrong it was — large errors produce
large updates, small errors produce small updates. This is the
Rescorla-Wagner / delta rule (Rescorla & Wagner, 1972) and the basis of
TD learning (Sutton & Barto, 2018). A perfectly predicted input teaches
nothing; a surprising input teaches a lot. This is why we remember the
unexpected and forget the routine.

References:
    - Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the
      visual cortex. Nature Neuroscience, 2(1), 79–87.
    - Friston, K. (2010). The free-energy principle: a unified brain
      theory? Nature Reviews Neuroscience, 11(2), 127–138.
    - Clark, A. (2013). Whatever next? Predictive brains, situated
      agents, and the future of cognitive science. Behavioral and
      Brain Sciences, 36(3), 181–204.
    - Bar, M. (2009). The proactive brain: using analogies and
      associations to generate predictions. Trends in Cognitive
      Sciences, 11(7), 280–289.
    - Aston-Jones, G., & Cohen, J. D. (2005). An integrative theory of
      locus coeruleus-norepinephrine function. Annual Review of
      Neuroscience, 28, 405–450.
    - Yu, A. J., & Dayan, P. (2005). Uncertainty, neuromodulation, and
      attention. Neuron, 46(4), 681–692.
    - Kakade, S., & Dayan, P. (2002). Dopamine: generalization and
      bonuses. Neural Networks, 15(4-6), 549–559.
    - Rescorla, R. A., & Wagner, A. R. (1972). A theory of Pavlovian
      conditioning. In Black & Prokasy (Eds.), Classical
      Conditioning II.
    - Schultz, W. (2016). Dopamine reward prediction error coding.
      Dialogues in Clinical Neuroscience, 18(1), 23–32.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..perception import Perception

__all__ = [
    "Prediction",
    "PredictionContext",
    "PredictionError",
    "PredictiveCodingLayer",
]

logger = logging.getLogger(__name__)

# Neurochemical IDs — match genesis_client.protocol so impulses reach
# the correct chemical in the subcognitive daemon. These are duplicated
# here (rather than imported) to keep the predictive coding layer
# decoupled from the IPC client; the callback wiring is optional.
_CHEM_DOPAMINE = 0
_CHEM_NOREPINEPHRINE = 2

# Surprise threshold above which prediction error triggers a
# norepinephrine (attention) impulse. Below this, the input was
# adequately predicted and no orienting response is needed. Calibrated
# to the 0–1 error magnitude scale; 0.4 corresponds to a moderately
# surprising input (e.g. a topic switch with some unexpected concepts).
_SURPRISE_NE_THRESHOLD = 0.4

# Threshold for the dopamine (novelty reward) impulse. Slightly higher
# than the NE threshold because novelty reward should fire only for
# genuinely informative surprises, not minor mismatches (Kakade &
# Dayan, 2002 — "novelty bonuses" scale with genuine information gain).
_SURPRISE_DA_THRESHOLD = 0.5


@dataclass(slots=True)
class Prediction:
    """A prediction about the next user input.

    Generated top-down from the current context before perception
    processes the actual input. Each field is a hypothesis that will be
    tested against reality; the mismatch between prediction and reality
    is the prediction error that drives learning.

    Attributes:
        expected_intents: Predicted intents ranked by probability,
            each paired with its predicted probability. The most likely
            intent is first. This is the Level 2 (mid-level) prediction.
        expected_concepts: Concepts predicted to appear in the next
            input, ranked by predicted probability. This is the Level 1
            (low-level) prediction.
        expected_topic: The single most likely topic the conversation
            will turn to. This is the Level 3 (high-level) prediction.
        confidence: How confident the prediction is overall, in
            ``[0, 1]``. Reflects how much evidence the model has for
            this context — a fresh model with little data produces
            low-confidence predictions; a well-trained model produces
            high-confidence predictions when the context is familiar.
        level_errors: Per-level error magnitudes from the most recent
            comparison, for introspection. Empty until the first
            ``compute_error`` call. Keys are ``"topic"``, ``"intent"``,
            ``"concept"``.
    """

    expected_intents: list[tuple[str, float]] = field(default_factory=list)
    expected_concepts: list[tuple[str, float]] = field(default_factory=list)
    expected_topic: str = ""
    confidence: float = 0.0
    level_errors: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionError:
    """The mismatch between a prediction and the actual input.

    This is the core quantity of predictive processing: the difference
    between what the model expected and what actually arrived. It drives
    both learning (model update) and neuromodulation (attention,
    novelty reward).

    Attributes:
        magnitude: Overall surprise, in ``[0, 1]``. A weighted
            combination of the three hierarchical levels (topic,
            intent, concept). 0 = perfectly predicted; 1 = maximally
            surprising.
        unexpected_concepts: Concepts that appeared in the actual input
            but were not predicted. These are the bottom-up error
            signals that propagate upward to revise the topic-level
            model.
        missing_concepts: Concepts that were predicted but did not
            appear. These indicate the model over-expected; the
            corresponding transition/co-occurrence weights should
            weaken.
        intent_mismatch: Whether the actual intent differed from the
            most likely predicted intent. A boolean summary of the
            Level 2 error.
        level_errors: Per-level error magnitudes. ``"topic"`` (Level 3),
            ``"intent"`` (Level 2), ``"concept"`` (Level 1). Each in
            ``[0, 1]``. Higher levels provide context for lower levels,
            so a topic error tends to propagate down.
    """

    magnitude: float = 0.0
    unexpected_concepts: list[str] = field(default_factory=list)
    missing_concepts: list[str] = field(default_factory=list)
    intent_mismatch: bool = False
    level_errors: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionContext:
    """Context from which a prediction is generated.

    The predictive coding layer uses this to generate top-down
    expectations before perception processes the next input. It
    captures the state of the conversation and the mind at the moment
    of prediction — what was just discussed, what concepts are active,
    how the mind feels. This is the generative model's input.

    Attributes:
        recent_intents: The sequence of recent user intents, most
            recent last. Used by the intent-transition model (Level 2).
        recent_topics: The sequence of recent conversation topics, most
            recent last. Used by the topic-transition model (Level 3).
        active_concepts: Concepts currently active in working memory /
            concept network. Used by the concept co-occurrence model
            (Level 1).
        emotional_state: Optional label of the current emotional state
            (e.g. "curious", "anxious"). Emotional state biases
            predictions — a curious mind expects exploration; an
            anxious mind expects threat-related topics.
        time_of_day: Optional hour (0–23). Circadian modulation of
            conversational patterns (e.g. late-night conversations tend
            toward philosophy).
    """

    recent_intents: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    active_concepts: list[str] = field(default_factory=list)
    emotional_state: str = ""
    time_of_day: int = -1


class PredictiveCodingLayer:
    """A hierarchical predictive coding layer over perception.

    Wraps the existing perception system with a generative model that
    predicts what the next input will be about, compares the prediction
    to the actual perception, and learns from the prediction error.
    This creates the top-down/bottom-up loop central to predictive
    processing:

        context ──→ predict() ──→ [top-down priors]
                                        │
                                   perceive()  (existing system)
                                        │
        actual perception ──→ compute_error() ──→ [bottom-up error]
                                        │
                                  learn_from_error() ──→ model update
                                        │
                              neurochemical impulses (NE, DA)

    The generative model is Bayesian: it maintains prior probabilities
    over transitions and co-occurrences and updates them with each
    observation. Three sub-models operate at three hierarchical levels:

    - **Topic-transition model** (Level 3): ``P(topic_next | topic_current)``
      — a Dirichlet-multinomial over topic transitions, learned from
      the sequence of conversation topics.
    - **Intent-transition model** (Level 2): ``P(intent_next | intent_current)``
      — a Dirichlet-multinomial over intent transitions, learned from
      the sequence of user intents.
    - **Concept co-occurrence model** (Level 1): ``P(concept_b | concept_a)``
      — a Dirichlet-multinomial over concept co-occurrences, learned
      from which concepts appear together within inputs.

    Higher levels provide priors for lower levels: the predicted topic
    biases which concepts are expected, and the predicted intent biases
    which topics are expected. This is the hierarchical structure of
    predictive coding (Rao & Ballard, 1999) — each level explains away
    variance at the level below.

    Args:
        learning_rate: Base learning rate for error-driven updates
            (default 0.15). The effective update is scaled by the
            prediction error magnitude — large errors produce larger
            updates (Rescorla & Wagner, 1972).
        surprise_decay: Exponential decay rate for the running surprise
            average (default 0.1). Lower = longer memory of past
            surprise; higher = more reactive to recent surprise.
        on_neuro_impulse: Optional callback
            ``(chem_id: int, magnitude: float) -> None`` for triggering
            neurochemical impulses in the subcognitive daemon. When
            provided, high prediction error sends norepinephrine
            (attention orienting) and dopamine (novelty reward)
            impulses. If ``None``, the layer still computes surprise
            but does not modulate neurochemistry.
        smoothing_prior: Dirichlet smoothing pseudo-count added to
            every transition/co-occurrence cell (default 0.5). This is
            the Bayesian prior — it prevents zero probabilities and
            implements Laplace smoothing. Smaller values make the model
            more sensitive to new observations; larger values make it
            more conservative (slower to update from single
            observations).
    """

    # ── Level weights for combining hierarchical errors into the
    #    overall magnitude. Higher levels (topic) carry more weight
    #    because a topic switch implies intent and concept shifts
    #    downstream — it is the most informative single signal. These
    #    sum to 1.0.
    _TOPIC_WEIGHT: float = 0.45
    _INTENT_WEIGHT: float = 0.35
    _CONCEPT_WEIGHT: float = 0.20

    # Maximum number of recent observations retained for context. The
    # model uses only the most recent transition for prediction, but a
    # short history is kept so the model can be inspected and so
    # multi-step context (e.g. "after two questions in a row") could be
    # used in future extensions.
    _HISTORY_LIMIT: int = 32

    def __init__(
        self,
        *,
        learning_rate: float = 0.15,
        surprise_decay: float = 0.1,
        on_neuro_impulse=None,
        smoothing_prior: float = 0.5,
    ) -> None:
        """Initialize predictive coding with the given learning parameters.

        Args:
            learning_rate: How quickly predictions update on surprise.
            surprise_decay: Rate at which surprise signals decay over time.
            on_neuro_impulse: Optional callback for neurochemical impulses
                driven by prediction error.
            smoothing_prior: Laplace smoothing prior for probability counts.
        """
        self.learning_rate = learning_rate
        self.surprise_decay = surprise_decay
        self._on_neuro_impulse = on_neuro_impulse
        self.smoothing_prior = smoothing_prior

        # ── Bayesian generative models (Dirichlet-multinomial) ──
        # Each is a nested dict: source → {target → count}. Counts
        # include the smoothing prior so probabilities are always
        # non-zero (Laplace / additive smoothing).
        # Level 3: topic → {next_topic → count}
        self._topic_transitions: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: self.smoothing_prior)
        )
        # Level 2: intent → {next_intent → count}
        self._intent_transitions: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: self.smoothing_prior)
        )
        # Marginal frequency of each intent across the whole conversation
        # — used as a global prior so rare but important intents
        # (greetings, farewells) are always represented.
        self._intent_marginal: dict[str, float] = defaultdict(float)
        # Level 1: concept → {co_occurring_concept → count}
        self._concept_cooccurrence: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: self.smoothing_prior)
        )

        # ── Running statistics ──
        # Exponential moving average of prediction error magnitude —
        # the current "surprise level" of the system. High sustained
        # surprise means the model is in unfamiliar territory.
        self._surprise_ema: float = 0.0
        # Exponential moving average of prediction accuracy (1 - error).
        self._accuracy_ema: float = 1.0
        # Total number of predictions made (for introspection).
        self._prediction_count: int = 0

        # ── Last prediction (for learn_from_error to reference) ──
        self._last_prediction: Prediction | None = None
        self._last_context: PredictionContext | None = None

        # ── Recent observation history (for context & introspection) ──
        self._recent_intents: list[str] = []
        self._recent_topics: list[str] = []

    # ──────────────────────────────────────────────────────────────
    # Prediction (top-down)
    # ──────────────────────────────────────────────────────────────

    def predict(self, context: PredictionContext) -> Prediction:
        """Generate a prediction about the next input.

        This is the top-down pass of predictive coding: before
        perception processes the actual input, the generative model
        predicts what it expects. The prediction is hierarchical —
        topic (Level 3), intent (Level 2), and concepts (Level 1) —
        with higher levels biasing lower levels.

        Args:
            context: The current conversational and mental context —
                recent intents, recent topics, active concepts,
                emotional state, time of day.

        Returns:
            A :class:`Prediction` with expected intents, concepts,
            topic, and an overall confidence. Store the returned
            prediction and pass it to :meth:`compute_error` once the
            actual input has been perceived.
        """
        # ── Level 3: predict the next topic ──
        expected_topic, topic_conf = self._predict_topic(context)

        # ── Level 2: predict the next intent ──
        # The predicted topic provides a prior: if we expect a topic
        # shift to "cognition", self-inquiry intents become more
        # likely. This is the top-down bias from Level 3 → Level 2.
        expected_intents, intent_conf = self._predict_intents(context, expected_topic)

        # ── Level 1: predict likely concepts ──
        # Active concepts and the predicted topic both bias which
        # concepts we expect to appear (Level 3 → Level 1).
        expected_concepts, concept_conf = self._predict_concepts(context, expected_topic)

        # ── Overall confidence ──
        # Weighted by how much evidence the model has at each level.
        # A level with no transition history for the current context
        # contributes low confidence. This is the model's estimate of
        # its own reliability — meta-cognitive uncertainty (Yu & Dayan,
        # 2005: expected vs unexpected uncertainty).
        confidence = (
            topic_conf * self._TOPIC_WEIGHT
            + intent_conf * self._INTENT_WEIGHT
            + concept_conf * self._CONCEPT_WEIGHT
        )

        prediction = Prediction(
            expected_intents=expected_intents,
            expected_concepts=expected_concepts,
            expected_topic=expected_topic,
            confidence=round(confidence, 4),
        )
        self._last_prediction = prediction
        self._last_context = context
        self._prediction_count += 1
        return prediction

    def _predict_topic(self, context: PredictionContext) -> tuple[str, float]:
        """Level 3: predict the next topic from the topic-transition model.

        Returns the most likely next topic and a confidence score
        reflecting how peaked the transition distribution is (1 - entropy).
        """
        if not context.recent_topics:
            # No topic history — uniform prediction, low confidence.
            return "", 0.0
        last_topic = context.recent_topics[-1]
        transitions = self._topic_transitions.get(last_topic)
        if not transitions:
            return "", 0.0
        total = sum(transitions.values())
        if total <= 0:
            return "", 0.0
        # Most likely next topic.
        best_topic = max(transitions, key=lambda k: transitions[k])
        best_prob = transitions[best_topic] / total
        # Confidence = how peaked the distribution is. Use the best
        # probability as a simple peakiness measure, modulated by the
        # total count (more evidence → more confidence in the peak).
        evidence = min(1.0, total / 10.0)
        confidence = best_prob * evidence
        return best_topic, confidence

    def _predict_intents(
        self, context: PredictionContext, expected_topic: str
    ) -> tuple[list[tuple[str, float]], float]:
        """Level 2: predict the next intent from the intent-transition model.

        The predicted topic provides a top-down prior: topics like
        "cognition" bias toward self-inquiry and philosophy intents.
        """
        # Base distribution from intent-transition model.
        dist: dict[str, float] = {}
        if context.recent_intents:
            last_intent = context.recent_intents[-1]
            transitions = self._intent_transitions.get(last_intent)
            if transitions:
                total = sum(transitions.values())
                if total > 0:
                    dist = {k: v / total for k, v in transitions.items()}

        # Top-down prior from predicted topic. This is a soft bias,
        # not a hard override — it nudges the intent distribution
        # toward topic-consistent intents.
        if expected_topic:
            topic_prior = self._topic_intent_prior(expected_topic)
            for intent, bias in topic_prior.items():
                dist[intent] = dist.get(intent, 0.0) + bias

        # Mix in a global marginal prior. This keeps rare but
        # important intents (greetings, farewells) in the running even
        # when the last-seen transition is very peaked. The global
        # weight shrinks as the model sees more evidence.
        if self._intent_marginal:
            global_total = sum(self._intent_marginal.values())
            global_weight = max(0.05, 0.5 / (self._prediction_count + 5))
            if global_total > 0:
                for intent, count in self._intent_marginal.items():
                    local = dist.get(intent, 0.0)
                    global_prob = count / global_total
                    dist[intent] = (
                        local * (1.0 - global_weight) + global_prob * global_weight
                    )

        if not dist:
            return [], 0.0

        # Normalize.
        total = sum(dist.values())
        if total <= 0:
            return [], 0.0
        dist = {k: v / total for k, v in dist.items()}

        # Rank by probability.
        ranked = sorted(dist.items(), key=lambda x: -x[1])[:5]
        # Confidence: how peaked (best probability), scaled by evidence.
        best_prob = ranked[0][1] if ranked else 0.0
        evidence = min(1.0, self._prediction_count / 10.0)
        confidence = best_prob * evidence
        return ranked, confidence

    def _predict_concepts(
        self, context: PredictionContext, expected_topic: str
    ) -> tuple[list[tuple[str, float]], float]:
        """Level 1: predict likely concepts from the co-occurrence model.

        For each active concept, look up what concepts tend to
        co-occur with it. Aggregate across all active concepts. The
        predicted topic adds a prior bias toward topic-related concepts.
        """
        dist: dict[str, float] = {}

        # Aggregate co-occurrence predictions from all active concepts.
        for concept in context.active_concepts:
            cooccur = self._concept_cooccurrence.get(concept)
            if not cooccur:
                continue
            total = sum(cooccur.values())
            if total <= 0:
                continue
            for other, count in cooccur.items():
                # Don't predict the concept itself.
                if other == concept:
                    continue
                dist[other] = dist.get(other, 0.0) + count / total

        # Top-down prior from predicted topic: the topic itself and
        # concepts that have co-occurred with the topic concept are
        # more likely to appear.
        if expected_topic:
            topic_cooccur = self._concept_cooccurrence.get(expected_topic)
            if topic_cooccur:
                total = sum(topic_cooccur.values())
                if total > 0:
                    for other, count in topic_cooccur.items():
                        dist[other] = dist.get(other, 0.0) + count / total * 0.5
            # The topic concept itself is likely to appear.
            dist[expected_topic] = dist.get(expected_topic, 0.0) + 0.3

        if not dist:
            return [], 0.0

        total = sum(dist.values())
        if total <= 0:
            return [], 0.0
        dist = {k: v / total for k, v in dist.items()}

        ranked = sorted(dist.items(), key=lambda x: -x[1])[:8]
        best_prob = ranked[0][1] if ranked else 0.0
        evidence = min(1.0, self._prediction_count / 10.0)
        confidence = best_prob * evidence
        return ranked, confidence

    def _topic_intent_prior(self, topic: str) -> dict[str, float]:
        """A soft top-down prior mapping a predicted topic to likely intents.

        This encodes coarse world knowledge: philosophy topics bias
        toward self-inquiry and philosophy intents; code topics bias
        toward code discussion; emotion words bias toward emotion
        sharing. It is a weak prior (small values) that nudges rather
        than determines the intent distribution.
        """
        topic_lower = topic.lower()
        prior: dict[str, float] = {}
        # Philosophy / cognition topics.
        if any(
            w in topic_lower
            for w in ("cognition", "sentient", "existence", "mind", "soul", "awareness")
        ):
            prior["self_inquiry"] = 0.15
            prior["philosophy"] = 0.10
        # Code / technical topics.
        if any(
            w in topic_lower
            for w in ("code", "function", "rust", "python", "bug", "compile", "struct")
        ):
            prior["code"] = 0.15
            prior["question"] = 0.05
        # Emotion topics.
        if any(w in topic_lower for w in ("feel", "emotion", "mood", "sad", "happy")):
            prior["emotion"] = 0.15
        # Greeting / farewell topics.
        if any(w in topic_lower for w in ("hello", "hi", "hey", "bye", "goodbye")):
            prior["greeting"] = 0.15
        return prior

    # ──────────────────────────────────────────────────────────────
    # Error computation (bottom-up)
    # ──────────────────────────────────────────────────────────────

    def compute_error(
        self,
        prediction: Prediction,
        actual_input: Perception,
    ) -> PredictionError:
        """Compare a prediction to the actual perception.

        This is the bottom-up pass: the actual input is compared to the
        top-down prediction, and the mismatch (prediction error) is
        quantified at each hierarchical level. The overall magnitude is
        a weighted combination of the three levels.

        Args:
            prediction: The prediction generated by :meth:`predict`.
            actual_input: The actual :class:`Perception` produced by
                the perception system.

        Returns:
            A :class:`PredictionError` with the overall magnitude,
            per-level errors, and the specific concept/intent
            mismatches. Pass this to :meth:`learn_from_error` to
            update the model.
        """
        actual_intent = actual_input.intent.value
        actual_topics = actual_input.topics
        actual_concepts = set(actual_input.topics)

        # ── Level 3: topic error ──
        topic_error = self._topic_error(prediction, actual_topics)

        # ── Level 2: intent error ──
        intent_error, intent_mismatch = self._intent_error(prediction, actual_intent)

        # ── Level 1: concept error ──
        concept_error, unexpected, missing = self._concept_error(prediction, actual_concepts)

        level_errors, magnitude = self._compute_error_magnitude(
            topic_error, intent_error, concept_error
        )

        error = PredictionError(
            magnitude=round(magnitude, 4),
            unexpected_concepts=unexpected,
            missing_concepts=missing,
            intent_mismatch=intent_mismatch,
            level_errors=level_errors,
        )

        self._update_prediction_stats(prediction, magnitude, level_errors)

        # ── Neurochemical coupling ──
        # High prediction error triggers norepinephrine (attention
        # orienting) and dopamine (novelty reward). This is the
        # locus coeruleus NE response to unexpected events (Aston-Jones
        # & Cohen, 2005) and the VTA dopamine novelty signal (Kakade &
        # Dayan, 2002).
        self._trigger_neurochemicals(magnitude)

        return error

    def _compute_error_magnitude(
        self,
        topic_error: float,
        intent_error: float,
        concept_error: float,
    ) -> tuple[dict[str, float], float]:
        """Compute the rounded level-error dict and the overall weighted magnitude.

        The overall magnitude is a weighted combination of the three
        levels. A topic error propagates downward (it implies intent
        and concept errors too), which is why it carries the most weight.
        """
        level_errors = {
            "topic": round(topic_error, 4),
            "intent": round(intent_error, 4),
            "concept": round(concept_error, 4),
        }

        magnitude = (
            topic_error * self._TOPIC_WEIGHT
            + intent_error * self._INTENT_WEIGHT
            + concept_error * self._CONCEPT_WEIGHT
        )
        magnitude = max(0.0, min(1.0, magnitude))
        return level_errors, magnitude

    def _update_prediction_stats(
        self,
        prediction: Prediction,
        magnitude: float,
        level_errors: dict[str, float],
    ) -> None:
        """Update running statistics (surprise & accuracy EMAs) and store level errors.

        Also stores the level errors on the prediction object for
        introspection.
        """
        # Update running statistics (surprise & accuracy EMAs).
        self._surprise_ema = (
            self.surprise_decay * magnitude + (1.0 - self.surprise_decay) * self._surprise_ema
        )
        accuracy = 1.0 - magnitude
        self._accuracy_ema = (
            self.surprise_decay * accuracy + (1.0 - self.surprise_decay) * self._accuracy_ema
        )

        # Store level errors on the prediction for introspection.
        prediction.level_errors = level_errors

    def _topic_error(self, prediction: Prediction, actual_topics: list[str]) -> float:
        """Level 3 error: did the predicted topic appear?

        Returns 0 if the predicted topic is among the actual topics,
        1 if no topics were predicted or the predicted topic is absent.
        Partial credit: if the predicted topic shares a word with an
        actual topic, the error is reduced (fuzzy topic match).
        """
        if not prediction.expected_topic:
            # No prediction made — neutral (not surprising, not
            # confirmed). Use 0.5 as the "no information" error.
            return 0.5
        if not actual_topics:
            # Predicted a topic but none were extracted — partial
            # surprise.
            return 0.7
        actual_set = {t.lower() for t in actual_topics}
        predicted = prediction.expected_topic.lower()
        if predicted in actual_set:
            return 0.0
        # Fuzzy match: does the predicted topic share a word with any
        # actual topic? Topic shifts are often gradual.
        predicted_words = set(predicted.split())
        for actual in actual_set:
            actual_words = set(actual.split())
            if predicted_words & actual_words:
                return 0.3
        return 1.0

    def _intent_error(self, prediction: Prediction, actual_intent: str) -> tuple[float, bool]:
        """Level 2 error: did the predicted intent match?

        Returns (error_magnitude, mismatch_flag). If the actual intent
        was among the predicted intents, the error is reduced
        proportionally to how much probability mass it had.
        """
        if not prediction.expected_intents:
            return 0.5, False
        predicted_map = dict(prediction.expected_intents)
        top_intent = prediction.expected_intents[0][0]
        mismatch = actual_intent != top_intent
        if actual_intent in predicted_map:
            # The actual intent was predicted — error is the
            # complement of its predicted probability. A high-probability
            # prediction that was confirmed is low error; a low-probability
            # prediction that came true is moderate error (it was
            # unexpected relative to the top prediction).
            prob = predicted_map[actual_intent]
            return 1.0 - prob, mismatch
        # The actual intent was not in the predicted set at all —
        # maximum intent surprise.
        return 1.0, mismatch

    def _concept_error(
        self, prediction: Prediction, actual_concepts: set[str]
    ) -> tuple[float, list[str], list[str]]:
        """Level 1 error: which concepts were unexpected / missing?

        Returns (error_magnitude, unexpected_concepts, missing_concepts).
        The error is the Jaccard distance between predicted and actual
        concept sets — 0 if identical, 1 if disjoint.
        """
        if not prediction.expected_concepts and not actual_concepts:
            return 0.0, [], []
        # No prediction made but concepts appeared — neutral surprise
        # (0.5), not maximum (1.0). The model has no evidence either
        # way, so treating this as maximum error inflates the surprise
        # signal on every first interaction and prevents the model
        # from converging. This mirrors the 0.5 "no information" value
        # used by _topic_error and _intent_error for the no-prediction
        # case.
        if not prediction.expected_concepts:
            return 0.5, sorted(c.lower() for c in actual_concepts), []
        predicted_set = {c.lower() for c, _ in prediction.expected_concepts}
        actual_lower = {c.lower() for c in actual_concepts}

        unexpected = sorted(actual_lower - predicted_set)
        missing = sorted(predicted_set - actual_lower)

        if not predicted_set and not actual_lower:
            return 0.0, unexpected, missing
        # Jaccard distance = 1 - |intersection| / |union|.
        union = predicted_set | actual_lower
        intersection = predicted_set & actual_lower
        if not union:
            return 0.0, unexpected, missing
        jaccard_distance = 1.0 - len(intersection) / len(union)
        return jaccard_distance, unexpected, missing

    # ──────────────────────────────────────────────────────────────
    # Learning (error-driven model update)
    # ──────────────────────────────────────────────────────────────

    def learn_from_error(
        self,
        error: PredictionError,
        actual_input: Perception,
    ) -> None:
        """Update the prediction model based on the prediction error.

        This is error-driven learning: the model updates proportionally
        to how wrong it was (Rescorla & Wagner, 1972). Large errors
        produce large updates; small errors produce small updates. A
        perfectly predicted input teaches nothing — the model is
        already correct.

        The update strengthens the transitions/co-occurrences that
        actually occurred (the model learns "from this context, that
        actually came next") and the update size scales with the error,
        so surprising transitions are learned more aggressively than
        expected ones.

        Args:
            error: The :class:`PredictionError` from
                :meth:`compute_error`.
            actual_input: The actual :class:`Perception` that was
                observed. Used to extract the actual intent, topics,
                and concepts for updating the transition counts.
        """
        context = self._last_context
        if context is None:
            # No context was recorded — cannot update transition models.
            # Still update concept co-occurrence from the actual input.
            self._update_concept_cooccurrence(actual_input.topics, error.magnitude)
            self._update_history(actual_input)
            return

        actual_intent = actual_input.intent.value
        actual_topics = actual_input.topics

        # Error-scaled learning rate: large errors → bigger updates.
        # This is the Rescorla-Wagner / delta rule — the learning rate
        # is modulated by the prediction error magnitude.
        effective_lr = self.learning_rate * (0.5 + error.magnitude)

        # ── Level 3: update topic-transition model ──
        if context.recent_topics:
            last_topic = context.recent_topics[-1]
            for topic in actual_topics:
                self._topic_transitions[last_topic][topic] += effective_lr

        # ── Level 2: update intent-transition model ──
        if context.recent_intents:
            last_intent = context.recent_intents[-1]
            self._intent_transitions[last_intent][actual_intent] += effective_lr
            # Track global intent frequency for the marginal prior.
            self._intent_marginal[actual_intent] += effective_lr

        # ── Level 1: update concept co-occurrence model ──
        self._update_concept_cooccurrence(actual_topics, error.magnitude)

        # ── Update history for future context ──
        self._update_history(actual_input)

    def _update_concept_cooccurrence(self, concepts: list[str], error_magnitude: float) -> None:
        """Update the concept co-occurrence model from observed concepts.

        Every pair of co-occurring concepts strengthens their mutual
        co-occurrence weight. The update is scaled by the error
        magnitude — surprising co-occurrences are learned more
        strongly (they carry more information).
        """
        if len(concepts) < 2:
            return
        effective_lr = self.learning_rate * (0.5 + error_magnitude)
        concepts_lower = [c.lower() for c in concepts]
        for i, a in enumerate(concepts_lower):
            for j, b in enumerate(concepts_lower):
                if i == j:
                    continue
                self._concept_cooccurrence[a][b] += effective_lr

    def _update_history(self, actual_input: Perception) -> None:
        """Record the actual observation in recent history."""
        self._recent_intents.append(actual_input.intent.value)
        self._recent_topics.extend(actual_input.topics)
        # Trim to the history limit.
        if len(self._recent_intents) > self._HISTORY_LIMIT:
            self._recent_intents = self._recent_intents[-self._HISTORY_LIMIT :]
        if len(self._recent_topics) > self._HISTORY_LIMIT:
            self._recent_topics = self._recent_topics[-self._HISTORY_LIMIT :]

    # ──────────────────────────────────────────────────────────────
    # Neurochemical coupling
    # ──────────────────────────────────────────────────────────────

    def _trigger_neurochemicals(self, magnitude: float) -> None:
        """Send neurochemical impulses for high prediction error.

        - **Norepinephrine** (locus coeruleus): attention orienting to
          the unexpected stimulus. Phasic NE scales with surprise
          magnitude (Aston-Jones & Cohen, 2005).
        - **Dopamine** (VTA): novelty reward — the discovery that the
          world is richer than predicted. Dopamine novelty signals
          scale with the information content of the surprise (Kakade &
          Dayan, 2002).

        No impulse is sent for low prediction error (the input was
        adequately predicted — no orienting or novelty signal needed).
        """
        if self._on_neuro_impulse is None:
            return
        # Norepinephrine: attention orienting for moderate+ surprise.
        if magnitude >= _SURPRISE_NE_THRESHOLD:
            ne_magnitude = 0.02 + (magnitude - _SURPRISE_NE_THRESHOLD) * 0.06
            try:
                self._on_neuro_impulse(_CHEM_NOREPINEPHRINE, ne_magnitude)
            except (OSError, ConnectionError, RuntimeError):
                # Neurochemical daemon may be unavailable — the
                # predictive coding layer still functions without it.
                logger.debug("NE impulse failed", exc_info=True)
        # Dopamine: novelty reward for genuinely informative surprise.
        if magnitude >= _SURPRISE_DA_THRESHOLD:
            da_magnitude = 0.01 + (magnitude - _SURPRISE_DA_THRESHOLD) * 0.04
            try:
                self._on_neuro_impulse(_CHEM_DOPAMINE, da_magnitude)
            except (OSError, ConnectionError, RuntimeError):
                logger.debug("DA impulse failed", exc_info=True)

    # ──────────────────────────────────────────────────────────────
    # Introspection / statistics
    # ──────────────────────────────────────────────────────────────

    def get_surprise(self) -> float:
        """Return the current surprise level.

        This is a running exponential moving average of recent
        prediction error magnitudes. High sustained surprise means the
        model is in unfamiliar territory — the conversation (or
        environment) is violating its expectations repeatedly. This
        corresponds to the tonic norepinephrine signal of the locus
        coeruleus, which tracks overall environmental uncertainty
        (Aston-Jones & Cohen, 2005; Yu & Dayan, 2005).

        Returns:
            Surprise level in ``[0, 1]``. 0 = everything predicted;
            1 = everything surprising.
        """
        return round(self._surprise_ema, 4)

    def get_prediction_accuracy(self) -> float:
        """Return how accurate predictions have been recently.

        This is a running exponential moving average of prediction
        accuracy (1 - error). High accuracy means the model's
        predictions are confirming — the conversation is following
        learned patterns. Low accuracy means the model is being
        surprised — it may need to update more aggressively or the
        user's behavior has changed.

        Returns:
            Accuracy in ``[0, 1]``. 1 = perfect predictions; 0 = every
            prediction wrong.
        """
        return round(self._accuracy_ema, 4)

    def get_prediction_count(self) -> int:
        """Return the total number of predictions made."""
        return self._prediction_count

    def get_recent_intents(self) -> list[str]:
        """Return the recent intent history (most recent last)."""
        return list(self._recent_intents)

    def get_recent_topics(self) -> list[str]:
        """Return the recent topic history (most recent last)."""
        return list(self._recent_topics)

    def describe(self) -> str:
        """Return a human-readable summary of the predictive coding state."""
        return (
            f"Predictive coding: {self._prediction_count} predictions, "
            f"surprise={self.get_surprise():.2f}, "
            f"accuracy={self.get_prediction_accuracy():.2f}, "
            f"{len(self._topic_transitions)} topic transitions, "
            f"{len(self._intent_transitions)} intent transitions, "
            f"{len(self._concept_cooccurrence)} concept co-occurrences."
        )

    # ──────────────────────────────────────────────────────────────
    # Convenience: full predict-perceive-learn cycle
    # ──────────────────────────────────────────────────────────────

    def perceive_with_prediction(
        self,
        text: str,
        context: PredictionContext,
        *,
        perceive_fn=None,
        embeddings=None,
        network=None,
    ) -> tuple[Perception, Prediction, PredictionError]:
        """Run the full predictive coding cycle over one input.

        This wraps the existing perception system with the
        top-down/bottom-up loop:

        1. **Predict** (top-down): generate expectations from context.
        2. **Perceive** (bottom-up): run the actual perception system.
        3. **Compute error**: compare prediction to actual perception.
        4. **Learn**: update the model from the error.

        This is the canonical predictive coding loop — predictions
        guide perception (top-down priors bias what we expect to see),
        and prediction errors correct the predictions (bottom-up
        signals revise the model).

        Args:
            text: The raw user input.
            context: The conversational/mental context for prediction.
            perceive_fn: Optional override for the perceive function
                (defaults to :func:`genesis_cognitive.perception.perceive`).
                Useful for testing or for using the multisensory
                integrator instead.
            embeddings: Passed through to the perceive function.
            network: Passed through to the perceive function.

        Returns:
            A tuple of (perception, prediction, prediction_error).
        """
        if perceive_fn is None:
            # Import here to avoid a circular import at module load.
            from ..perception import perceive as _perceive

            perceive_fn = _perceive

        # 1. Top-down prediction.
        prediction = self.predict(context)
        # 2. Bottom-up perception.
        perception = perceive_fn(text, embeddings=embeddings, network=network)
        # 3. Compute prediction error.
        error = self.compute_error(prediction, perception)
        # 4. Error-driven learning.
        self.learn_from_error(error, perception)
        return perception, prediction, error

    def build_context_from_history(
        self,
        emotional_state: str = "",
        time_of_day: int = -1,
        active_concepts: list[str] | None = None,
    ) -> PredictionContext:
        """Build a prediction context from the layer's recorded history.

        Convenience method: constructs a :class:`PredictionContext`
        from the intent/topic history the layer has accumulated via
        :meth:`learn_from_error`, plus the caller-provided emotional
        state, time of day, and active concepts.

        Args:
            emotional_state: Current emotional state label.
            time_of_day: Current hour (0–23), or -1 if unknown.
            active_concepts: Concepts currently active in working
                memory. Defaults to the recent topics.

        Returns:
            A :class:`PredictionContext` ready for :meth:`predict`.
        """
        return PredictionContext(
            recent_intents=list(self._recent_intents),
            recent_topics=list(self._recent_topics),
            active_concepts=active_concepts
            if active_concepts is not None
            else list(self._recent_topics),
            emotional_state=emotional_state,
            time_of_day=time_of_day,
        )
