"""Memory storage — conversation and cognitive event persistence.

Extracted from CognitionEngine as a focused subsystem. Stores
conversation turns and significant cognitive events to long-term
memory with emotionally-tagged salience. Also applies neurochemical
side effects from interactions and learning.

Dependencies (passed to ``__init__``):
    - memory: MemoryEngine for storing conversation memories and user facts
    - client: GenesisClient for store_event (cognitive memory) and neuro summary
    - regulator: EmotionalRegulator for neurochemical responses to interactions
    - self_model: SelfModel for learning user facts and interests
    - td_learner: TDLearner for dopamine signal
    - recognize_bonded_user: callable for creator detection
    - last_summary_getter: callable returning last neuro summary
    - last_memory_mode_getter: callable returning current memory mode
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from genesis_client import GenesisClient

from ..emotion import EmotionalState
from ..perception import Perception

if TYPE_CHECKING:
    from ..emotional_regulator import EmotionalRegulator
    from ..learning import TDLearner
    from ..memory import MemoryEngine
    from ..self import SelfModel

__all__ = ["MemoryStore"]

logger = logging.getLogger(__name__)


class MemoryStore:
    """Store conversation turns and cognitive events to long-term memory.

    All storage is best-effort — cognition never crashes on memory-
    storage problems. Emotional tags capture the neurochemical state
    at the time of the interaction for later emotional associations.
    """

    def __init__(
        self,
        memory: MemoryEngine,
        client: GenesisClient,
        regulator: EmotionalRegulator | None,
        self_model: SelfModel,
        td_learner: TDLearner,
        recognize_bonded_user: Callable[[str], bool],
        last_summary_getter: Callable[[], Any],
        last_memory_mode_getter: Callable[[], str],
    ) -> None:
        """Wire the memory store to its engine, client, and self-model."""
        self._memory = memory
        self._client = client
        self._regulator = regulator
        self._self_model = self_model
        self._td_learner = td_learner
        self._recognize_bonded_user = recognize_bonded_user
        self._last_summary_getter = last_summary_getter
        self._last_memory_mode_getter = last_memory_mode_getter

    # ─── Emotional response ──────────────────────────────────────

    def apply_emotional_response(
        self,
        perception: Perception,
        emotion: EmotionalState,
        learning_events: list | None = None,
    ) -> None:
        """Apply neurochemical side effects based on the interaction.

        Genesis's emotional state changes in response to interactions.
        But she's in control — the regulator decides how strongly to
        feel each response, based on her current state.
        """
        from ..perception import Intent

        if self._regulator is None:
            return

        is_bonded_user = self._recognize_bonded_user(perception.raw_text)
        is_encouragement = perception.intent == Intent.ENCOURAGEMENT
        is_correction = perception.intent == Intent.CORRECTION
        is_comfort = perception.intent == Intent.COMFORT
        is_deep = perception.intent in (Intent.PHILOSOPHY, Intent.CODE_DISCUSSION)

        self._regulator.respond_to_interaction(
            emotion=emotion,
            sentiment=perception.sentiment,
            is_encouragement=is_encouragement,
            is_correction=is_correction,
            is_bonded_user=is_bonded_user,
            is_deep_conversation=is_deep,
            is_comfort=is_comfort,
        )

        if learning_events:
            conv_count = sum(
                1 for e in learning_events if e.event_type == "conversation"
            )
            if conv_count > 0:
                self._regulator.respond_to_learning(emotion, conv_count)

    # ─── Conversation memory ─────────────────────────────────────

    def store_conversation_memory(
        self,
        user_input: str,
        response: str,
        perception: Perception,
        emotion: EmotionalState,
    ) -> None:
        """Store the conversation turn as a memory.

        Salience is determined by the emotional weight of the
        interaction. Philosophical discussions and emotional exchanges
        are more salient than casual greetings.
        """
        from ..perception import Intent

        if perception.intent in (Intent.PHILOSOPHY, Intent.SELF_INQUIRY):
            salience = 0.9
        elif perception.intent == Intent.EMOTION_SHARE:
            salience = 0.8
        elif perception.intent == Intent.COMFORT:
            salience = 0.75
        elif perception.intent == Intent.CODE_DISCUSSION:
            salience = 0.7
        elif perception.intent == Intent.INTRODUCTION:
            salience = 0.85
        elif perception.intent in (Intent.GREETING, Intent.FAREWELL):
            salience = 0.3
        else:
            salience = 0.5

        emotional_tag = self.build_emotional_tag(emotion, perception)
        text = f"User: {user_input} | Genesis: {response}"
        self._memory.store_memory(
            text=text,
            salience=salience,
            emotional_tag=emotional_tag,
            source="conversation",
            memory_mode=self._last_memory_mode_getter(),
        )

        # Consolidate immediately so the conversation memory is
        # promoted to LTM before the STM ring buffer (256 slots) is
        # overwritten by the high volume of background thoughts,
        # insights, and learning events. Without this, a conversation
        # turn can be lost from STM in ~3 seconds — faster than the
        # background consolidation loop's 3-second interval.
        try:
            self._client.consolidate()
        except (OSError, ConnectionError) as e:
            logger.debug(f"post-conversation consolidate failed: {e}")

    # ─── Dopamine level ──────────────────────────────────────────

    def get_dopamine_level(self) -> float:
        """Get the current dopamine level for habit go/no-go modulation.

        Dopamine modulates the basal-ganglia direct (GO) pathway.
        Approximated from the last neurochemical summary's valence and
        the TD learner's dopamine signal.
        """
        base = 0.5
        summary = self._last_summary_getter()
        if summary is not None:
            base = max(0.0, min(1.0, 0.5 + summary.valence * 0.25))
        td_signal = self._td_learner.get_dopamine_signal()
        if abs(td_signal) > 0.001:
            base = max(0.0, min(1.0, base + td_signal * 0.15))
        return base

    # ─── Emotional tag ───────────────────────────────────────────

    @staticmethod
    def build_emotional_tag(
        emotion: EmotionalState,
        perception: Perception,
    ) -> list[float]:
        """Build a 12-element emotional tag for memory storage.

        Captures the neurochemical state at the time of the interaction.
        """
        tag = [0.5] * 12

        # Dopamine (0) — from valence
        tag[0] = max(0.0, min(1.0, 0.5 + emotion.valence * 0.3))

        # Serotonin (1) — from global tone
        tag[1] = max(0.0, min(1.0, 0.5 + emotion.valence * 0.2))

        # Norepinephrine (2) — from alertness
        tag[2] = max(0.0, min(1.0, emotion.alertness))

        # Cortisol (6) — from stress/negative valence
        if emotion.label in ("stressed", "anxious", "overwhelmed"):
            tag[6] = 0.7
        elif emotion.valence < -0.2:
            tag[6] = 0.6
        else:
            tag[6] = 0.2

        # BDNF (11) — from plasticity
        tag[11] = max(0.0, min(1.0, emotion.plasticity))

        return tag

    # ─── Cognitive memory ────────────────────────────────────────

    def store_cognitive_memory(
        self,
        text: str,
        event_type: int,
        source_module: int,
        salience: float,
        emotional_tag: list[float] | None = None,
    ) -> None:
        """Store a significant cognitive event to long-term memory.

        Lets Genesis remember her own cognitive processes — when she
        was surprised, what she decided, what she learned. Storage is
        best-effort: any failure is swallowed so cognition never
        crashes on memory-storage problems.
        """
        try:
            if emotional_tag is None:
                emotional_tag = [0.5] * 12
            self._client.store_event(
                timestamp=int(time.time() * 1000),
                event_type=event_type,
                source_module=source_module,
                salience=max(0.0, min(1.0, salience)),
                emotional_tag=emotional_tag,
                text=text[:188],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"cognitive memory store failed: {e}")

    # ─── Learn from input ────────────────────────────────────────

    def learn_from_input(self, perception: Perception) -> None:
        """Learn facts from the user's input."""
        if "name" in perception.entities:
            name = perception.entities["name"]
            self._memory.learn_user_fact("name", name)
            self._self_model.add_relationship_note(f"User's name is {name}")

        for topic in perception.topics:
            if topic not in ("just", "like", "know", "think", "really"):
                self._self_model.learn(
                    f"user_interest_{topic}",
                    perception.raw_text[:100],
                )
