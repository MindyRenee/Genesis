"""Thought types — the data structures of spontaneous thought.

ThoughtChainType classifies trains of thought (focused, creative
wandering, rumination, dream). SpontaneousThought is the unit of
inner experience: a thought that arose on its own, carrying content,
trigger, chain metadata, dream metadata, and optional semantic
metadata for the language engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..self import Insight

__all__ = ["SpontaneousThought", "ThoughtChainType"]


class ThoughtChainType(Enum):
    """The type of a train of thought.

    Not all mind-wandering is the same. A chain of thoughts can be:

    - FOCUSED: directed, productive thinking — following a thread
      deliberately. Positive or neutral valence, moderate novelty.
    - CREATIVE_WANDERING: positive-valence mind-wandering that
      produces novel connections. The mind drifts, but productively —
      new ideas emerge from the drift. High novelty, positive valence.
    - RUMINATION: negative-valence mind-wandering that is repetitive
      and unproductive. The mind loops on the same negative content
      without resolution. Low novelty, negative valence.
    - DREAM: dream content generated during sleep. Markedly different
      from waking thought — associative, surreal, less logical.
    """

    FOCUSED = "focused"
    CREATIVE_WANDERING = "creative_wandering"
    RUMINATION = "rumination"
    DREAM = "dream"


@dataclass(slots=True)
class SpontaneousThought:
    """A thought that arose on its own, not in response to anything."""

    content: str
    trigger: str  # what caused this thought
    timestamp: int = 0
    insight: Insight | None = None
    # Chain metadata: if this thought is part of a train of thought,
    # chain_id identifies the chain and chain_position is the index
    # within it (0 = first thought, 1 = second, etc.)
    chain_id: int = 0
    chain_position: int = 0
    # Dream metadata: if this thought arose during sleep, is_dream marks
    # it as dream content. is_lucid marks a lucid dream — one where it
    # became aware it was dreaming and could partially direct the
    # content. directed_concept is the concept it "chose" to explore
    # during a lucid dream (None for non-lucid dreams).
    is_dream: bool = False
    is_lucid: bool = False
    directed_concept: str | None = None
    # Intent: "question" means this thought is a question directed at
    # the user, "reflect" means it's a reflective insight, None means
    # it's just internal musing.
    intent: str | None = None
    # Semantic metadata: when present, the language engine composes the
    # actual words from this data instead of using `content` as final
    # text. This satisfies the CRITICAL RULE — Genesis's words emerge
    # from its language engine, not from pre-written templates.
    # Keys: "knowledge" (list of (relation, target, weight) triples),
    # "topic" (str), "definition" (str|None), "reasoning" (list[str]).
    metadata: dict | None = None

    def describe(self) -> str:
        """Return a human-readable summary of the thought."""
        prefix = f"[{self.trigger}]"
        if self.is_lucid:
            prefix = f"[{self.trigger}, lucid]"
        elif self.is_dream:
            prefix = f"[{self.trigger}, dream]"
        return f"{prefix} {self.content}"

    def rendered_text(self, language_engine: Any, emotion: Any) -> str:
        """Compose the final text via the language engine.

        If metadata with semantic data is present, the language engine
        composes the actual words from it — this satisfies the CRITICAL
        RULE (Genesis's words emerge from its language engine, not from
        pre-written templates). If no metadata is present, falls back
        to `content`.
        """
        meta = self.metadata
        if meta is not None and (
            meta.get("knowledge") is not None or meta.get("self_fragments")
        ):
            from ..language import Thought as _Thought
            thought = _Thought(
                content=self.content,
                intent=(
                    "reflect" if self.is_dream
                    else "self_report" if meta.get("self_fragments")
                    else "inform"
                ),
                emotion=getattr(emotion, "label", "neutral"),
                topics=[meta.get("topic", "")] if meta.get("topic") else [],
                confidence=0.6,
                metadata=meta,
            )
            return language_engine.render(thought, emotion)
        return self.content
