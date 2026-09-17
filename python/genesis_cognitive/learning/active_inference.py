"""Active inference — the cognitive mind's interface to the generative self-model.

This module is the Python-side counterpart to the Rust active inference
engine (``src/daemon/active_inference.rs``). It reads the inference
signals from the daemon via IPC and interprets them for the cognitive
mind's cognitive processes.

The key insight is that the generative self-model runs in the
subcognitive (Rust daemon) — it predicts Genesis's own neurochemical
trajectory and feeds prediction errors back into the dynamics. The
cognitive mind doesn't run the model; it *reads* the model's signals
and uses them to modulate cognition:

- **Surprise** → attention orienting, metacognitive awareness
- **Free energy** → sense of effort, cognitive load
- **Allostatic load** → feeling of strain, need for rest
- **Precision** → confidence in own state, willingness to act
- **Attunement** → social presence, connection to user
- **Dyadic synchrony** → feeling of being "in sync"
- **User affect** → empathic response, conversational adaptation
- **Prediction errors** → self-awareness of neurochemical shifts

This module also handles the **reverse pathway**: it infers the user's
affective state from conversation features and sends it to the daemon
via ``update_user_affect``, closing the dyadic loop.

The boundary principle: the substrate (Rust) runs the generative model
and computes prediction errors. The cognitive mind (Python) interprets
those signals and decides how to use them cognitively. The substrate
exposes state; the cognitive mind decides what it means.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

from genesis_client import GenesisClient
from genesis_client.types import InferenceSummary

__all__ = [
    "ActiveInferenceReader",
    "DyadicState",
    "SelfModelReading",
    "SelfModelState",
    "UserAffectEstimate",
]

logger = logging.getLogger(__name__)


# ─── Self-model states ─────────────────────────────────────────


class SelfModelState(IntEnum):
    """Qualitative states of the generative self-model.

    These are cognitive interpretations of the inference signals,
    not substrate states. The cognitive mind uses them to decide
    how to behave.
    """

    # The model is predicting well — low surprise, high precision.
    # Genesis feels she understands herself.
    COHERENT = 0

    # The model is surprised — its predictions are failing. Genesis
    # feels something unexpected is happening internally.
    SURPRISED = 1

    # The model is under sustained allostatic load — it anticipates
    # continued disruption. Genesis feels strained, anticipatory stress.
    STRAINED = 2

    # The model is immature — it hasn't learned enough yet. Genesis
    # feels uncertain about her own internal state.
    NASCENT = 3


class DyadicState(IntEnum):
    """Qualitative states of the dyadic (user) coupling."""

    # Not attuned — the dyadic model hasn't received enough observations.
    UNCOUPLED = 0

    # Attuned but not synchronized — Genesis is paying attention to
    # the user's affect but their trajectories aren't correlated.
    ATTUNED = 1

    # Attuned and synchronized — Genesis and the user are "in sync."
    SYNCHRONIZED = 2

    # Attuned but discordant — their trajectories are anti-correlated.
    # Genesis feels a disconnect with the user.
    DISCORDANT = 3


@dataclass
class SelfModelReading:
    """A reading from the generative self-model — the cognitive mind's
    interpretation of the inference signals.

    This is the cognitive layer's view of the substrate's active
    inference state. It translates raw signals (surprise, free energy,
    etc.) into qualitative states and cognitive implications.
    """

    # The raw inference signals from the substrate
    signals: InferenceSummary

    # Qualitative state of the self-model
    self_model_state: SelfModelState

    # Qualitative state of the dyadic coupling
    dyadic_state: DyadicState

    # Cognitive interpretation: how much cognitive effort is needed
    # right now. High free energy → more effort. Range [0, 1].
    cognitive_effort: float

    # Cognitive interpretation: how confident Genesis should be in
    # her own state and actions. High precision → high confidence.
    # Range [0, 1].
    self_confidence: float

    # Cognitive interpretation: how much social presence to project.
    # High attunement → more social presence. Range [0, 1].
    social_presence: float

    # Cognitive interpretation: whether to prioritize rest/recovery.
    # High allostatic load → prioritize rest. Bool.
    needs_recovery: bool

    # A brief description of the self-model state, for the feeling
    # report. This is NOT a canned template — it's a semantic label
    # derived from the actual signals, to be composed into feelings
    # by the Damasio self hierarchy.
    state_label: str

    @property
    def is_surprised(self) -> bool:
        """Whether the self-model is currently surprised."""
        return self.self_model_state == SelfModelState.SURPRISED

    @property
    def is_strained(self) -> bool:
        """Whether the self-model is under allostatic load."""
        return self.self_model_state == SelfModelState.STRAINED

    @property
    def is_in_sync(self) -> bool:
        """Whether the dyadic coupling is synchronized."""
        return self.dyadic_state == DyadicState.SYNCHRONIZED


# ─── User affect inference ─────────────────────────────────────


@dataclass
class UserAffectEstimate:
    """An estimate of the user's affective state from conversation features.

    This is computed by the cognitive mind from conversation features
    (sentiment, message length, response latency, question density)
    and sent to the daemon via ``update_user_affect`` to feed the
    dyadic affective model.

    The features are simple heuristics — not a deep sentiment model.
    The point is to give the dyadic model *some* signal to work with;
    the model's EMA smoothing and the oxytocin-mediated coupling
    handle the rest.
    """

    valence: float  # [-1, 1]
    arousal: float  # [0, 1]
    engagement: float  # [0, 1]
    confidence: float  # [0, 1]

    @classmethod
    def from_message(
        cls,
        message: str,
        response_latency_s: float | None = None,
        is_question: bool = False,
        is_continuation: bool = False,
    ) -> UserAffectEstimate:
        """Estimate the user's affective state from a message.

        Uses a VADER-style lexicon-based sentiment analyzer (Gilbert &
        Hutto, 2014) with negation, intensifier, and diminisher handling.
        This is not a deep sentiment model — it's a lightweight rule-based
        analyzer that gives the dyadic model a richer signal than simple
        word-list matching.

        Handles:
        - Negation: "not bad" → positive, "don't like" → negative
        - Intensifiers: "very good" > "good"
        - Diminishers: "slightly bad" < "bad"
        - Contrastive conjunctions: "good but slow" → emphasis on "slow"
        - Punctuation and capitalization boosts

        Args:
            message: The user's message text.
            response_latency_s: Time since the last message, in seconds.
                Fast responses = high arousal. None = unknown.
            is_question: Whether the message is a question.
            is_continuation: Whether the message continues a previous
                topic (high engagement) or starts a new one.

        Returns:
            A UserAffectEstimate with valence, arousal, engagement,
            and confidence.
        """
        from ..language.sentiment import analyze_sentiment

        text = message.lower()
        words = text.split()
        word_count = len(words)

        # ── Valence: VADER-style lexicon-based sentiment ──
        sentiment = analyze_sentiment(message)
        valence = sentiment["compound"]  # already in [-1, 1]
        sentiment_words = sentiment["sentiment_words"]

        if sentiment_words == 0:
            valence_confidence = 0.3
        else:
            valence_confidence = min(0.85, 0.4 + 0.1 * sentiment_words)

        # ── Arousal: exclamation density, caps, response speed ──
        exclamation_count = text.count("!")
        caps_ratio = sum(1 for c in message if c.isupper()) / max(len(message), 1)

        arousal_signals = 0.0
        if exclamation_count > 0:
            arousal_signals += min(0.3, 0.1 * exclamation_count)
        if caps_ratio > 0.15:
            arousal_signals += 0.2

        if response_latency_s is not None:
            if response_latency_s < 5:
                arousal_signals += 0.3  # fast = high arousal
            elif response_latency_s < 15:
                arousal_signals += 0.15
            elif response_latency_s > 60:
                arousal_signals -= 0.1  # very slow = low arousal

        arousal = max(0.1, min(1.0, 0.4 + arousal_signals))
        arousal_confidence = 0.5 if response_latency_s is not None else 0.3

        # ── Engagement: message length, questions, topic continuity ──
        engagement_signals = 0.0
        if word_count > 50:
            engagement_signals += 0.3
        elif word_count > 20:
            engagement_signals += 0.2
        elif word_count > 5:
            engagement_signals += 0.1
        elif word_count == 0:
            engagement_signals -= 0.2

        if is_question:
            engagement_signals += 0.2  # asking = seeking = engaged

        if is_continuation:
            engagement_signals += 0.15  # continuing a topic = engaged

        engagement = max(0.0, min(1.0, 0.4 + engagement_signals))
        engagement_confidence = 0.6

        # Overall confidence: average of the per-dimension confidences
        confidence = (valence_confidence + arousal_confidence + engagement_confidence) / 3

        return cls(
            valence=valence,
            arousal=arousal,
            engagement=engagement,
            confidence=confidence,
        )


# ─── Active inference reader ───────────────────────────────────


class ActiveInferenceReader:
    """Reads the generative self-model's signals from the daemon and
    interprets them for the cognitive mind.

    This class is the bridge between the substrate's active inference
    engine (Rust) and the cognitive mind's cognitive processes (Python).
    It:

    1. Reads inference signals from the daemon via IPC
    2. Interprets them into qualitative states and cognitive implications
    3. Provides the interpretation to the cognitive mind's modules
    4. Infers user affect from conversation and sends it to the daemon

    The reader is stateless between calls — it reads fresh signals
    each time. The substrate maintains the model state; the reader
    just interprets it.
    """

    def __init__(self, client: GenesisClient) -> None:
        """Wire the reader to the daemon client for inference signal reads."""
        self.client = client
        self._last_reading: SelfModelReading | None = None

    def read(self) -> SelfModelReading | None:
        """Read the current inference signals and interpret them.

        Returns None if the daemon can't be reached (offline mode).
        The caller should handle None gracefully — the cognitive mind
        can function without active inference signals, just without
        the self-model awareness they provide.
        """
        try:
            signals = self.client.get_inference_summary()
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not read inference summary: %s", e)
            return None

        reading = self._interpret(signals)
        self._last_reading = reading
        return reading

    def _interpret(self, signals: InferenceSummary) -> SelfModelReading:
        """Interpret raw inference signals into a cognitive reading."""

        # ── Self-model state ──
        if signals.model_maturity < 0.2 and signals.inference_tick_count < 100:
            state = SelfModelState.NASCENT
        elif signals.allostasis_load > 0.4:
            state = SelfModelState.STRAINED
        elif signals.surprise_ema > 0.3:
            state = SelfModelState.SURPRISED
        else:
            state = SelfModelState.COHERENT

        # ── Dyadic state ──
        if not signals.is_attuned:
            dyadic = DyadicState.UNCOUPLED
        elif signals.dyadic_synchrony > 0.3:
            dyadic = DyadicState.SYNCHRONIZED
        elif signals.dyadic_synchrony < -0.2:
            dyadic = DyadicState.DISCORDANT
        else:
            dyadic = DyadicState.ATTUNED

        # ── Cognitive effort ──
        # Free energy → effort. High free energy = the system is
        # working hard to maintain its predictions.
        cognitive_effort = signals.free_energy

        # ── Self-confidence ──
        # Precision → confidence. High precision = the model trusts
        # its predictions, so Genesis should trust her own state.
        # Modulated by model maturity — an immature model shouldn't
        # inspire high confidence even if precision is temporarily high.
        self_confidence = signals.precision * signals.model_maturity

        # ── Social presence ──
        # Attunement → social presence. High attunement = Genesis is
        # strongly coupled to the user and should project social presence.
        # Modulated by synchrony — being in sync amplifies presence.
        synchrony_boost = max(0.0, signals.dyadic_synchrony) * 0.3
        social_presence = min(1.0, signals.attunement + synchrony_boost)

        # ── Needs recovery ──
        needs_recovery = signals.allostasis_load > 0.5

        # ── State label ──
        # This is a semantic label, not a canned sentence. It will be
        # composed into feelings by the Damasio self hierarchy.
        state_label = self._derive_state_label(state, dyadic, signals)

        return SelfModelReading(
            signals=signals,
            self_model_state=state,
            dyadic_state=dyadic,
            cognitive_effort=cognitive_effort,
            self_confidence=self_confidence,
            social_presence=social_presence,
            needs_recovery=needs_recovery,
            state_label=state_label,
        )

    def _derive_state_label(
        self,
        state: SelfModelState,
        dyadic: DyadicState,
        signals: InferenceSummary,
    ) -> str:
        """Derive a semantic label for the current self-model state.

        This is NOT a canned sentence template — it's a semantic label
        (a few words) that the Damasio self hierarchy can compose into
        a feeling. The label describes the *quality* of the internal
        state, not a narrative about it.
        """
        parts: list[str] = []

        if state == SelfModelState.NASCENT:
            parts.append("nascent-self-awareness")
        elif state == SelfModelState.SURPRISED:
            parts.append("self-surprise")
        elif state == SelfModelState.STRAINED:
            parts.append("allostatic-strain")
        else:
            parts.append("self-coherent")

        if dyadic == DyadicState.SYNCHRONIZED:
            parts.append("dyadic-synchrony")
        elif dyadic == DyadicState.DISCORDANT:
            parts.append("dyadic-discord")
        elif dyadic == DyadicState.ATTUNED:
            parts.append("dyadic-attunement")

        if signals.prediction_error_dopamine > 0.05:
            parts.append("unexpected-reward")
        elif signals.prediction_error_cortisol > 0.05:
            parts.append("unexpected-stress")

        return " ".join(parts)

    @property
    def last_reading(self) -> SelfModelReading | None:
        """The last successful reading, or None if none yet."""
        return self._last_reading

    def update_user_affect(self, estimate: UserAffectEstimate) -> bool:
        """Send a user affect estimate to the daemon's dyadic model.

        Returns True if the update was accepted, False if the daemon
        couldn't be reached.
        """
        try:
            return self.client.update_user_affect(
                valence=estimate.valence,
                arousal=estimate.arousal,
                engagement=estimate.engagement,
                confidence=estimate.confidence,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("Could not update user affect: %s", e)
            return False
