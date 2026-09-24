"""Response styling — metacognitive tone adjustment and learning acknowledgment.

Extracted from CognitionEngine as a focused subsystem. Applies the
active metacognitive strategy (assertive, hedging, questioning, varied,
neutral) to a response, and weaves learning acknowledgments into the
response when the self-directed learner extracted new facts.

The modifications are conservative — they adjust tone, not content.
The underlying knowledge and reasoning remain unchanged. All phrasing
emerges from the language engine, never from hardcoded templates.

Dependencies (passed to ``__init__``):
    - language: LanguageEngine for composing ack phrases
    - composer: ThoughtComposer for composing about newly learned concepts
    - working_memory: WorkingMemory for current focus (anaphora resolution)
    - rng: random number generator for variation
    - response_style_getter: callable returning the current style string
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..emotion import EmotionalState
from ..language import LanguageEngine, Thought
from ..perception import Perception

if TYPE_CHECKING:
    from ..cognition.thought_composer import ThoughtComposer
    from ..memory import WorkingMemory

__all__ = ["ResponseStyler"]

# Common words that should not trigger learning acknowledgments.
# These are function words, pronouns, and everyday adjectives that
# appear in almost every sentence — acknowledging "learning" them
# makes the response incoherent and self-absorbed.
_TRIVIAL_WORDS: frozenset[str] = frozenset({
    "this", "that", "these", "those", "here", "there", "where",
    "when", "what", "which", "who", "whom", "whose", "why", "how",
    "with", "from", "into", "through", "about", "over", "under",
    "have", "been", "were", "they", "them", "their", "theirs",
    "some", "any", "all", "both", "each", "more", "most", "other",
    "same", "own", "too", "can", "did", "does", "doing", "make",
    "made", "take", "took", "get", "got", "come", "came", "like",
    "well", "even", "still", "now", "way", "thing", "things",
    "also", "just", "only", "very", "really", "quite", "such",
    "than", "then", "will", "would", "could", "should", "might",
    "good", "bad", "great", "nice", "fine", "okay", "yes", "sure",
    "wonderful", "amazing", "awesome", "cool", "happy",
    "glad", "pleased", "delighted", "thrilled", "excited",
    "so", "always", "never",
    "something", "anything", "everything", "nothing", "someone",
    "anyone", "everyone", "nobody", "somebody", "anybody",
    "today", "tomorrow", "yesterday", "later",
    "maybe", "perhaps", "probably", "definitely", "certainly",
    "talk", "talking", "said", "say", "saying", "tell", "telling",
    "know", "knew", "known", "knowing", "think", "thought",
    "thinking", "feel", "felt", "feeling", "want", "wanted",
    "need", "needed", "try", "tried", "trying", "let", "lets",
    "going", "gone", "went", "being", "having", "making",
    "lot", "bit", "kind", "sort", "type", "part", "side", "end",
    "first", "last", "next", "new", "old", "long", "short",
})


class ResponseStyler:
    """Apply metacognitive tone adjustments and learning acknowledgments.

    All modifications are tone-level — the underlying knowledge and
    reasoning remain unchanged. Phrasing emerges from the language
    engine, never from hardcoded template strings.
    """

    def __init__(
        self,
        language: LanguageEngine,
        composer: ThoughtComposer,
        working_memory: WorkingMemory,
        rng: Any,
        response_style_getter: Callable[[], str],
    ) -> None:
        """Wire the styler to its language engine, composer, and working memory."""
        self._language = language
        self._composer = composer
        self._working_memory = working_memory
        self._rng = rng
        self._response_style_getter = response_style_getter

    # ─── Response style application ──────────────────────────────

    def apply_response_style(self, response: str, thought: Thought) -> str:
        """Apply the active metacognitive strategy to the response.

        - **assertive**: trim hedging language, present knowledge directly
        - **hedging**: add uncertainty markers when confidence is low
        - **questioning**: no-op (questions are queued via /teach-questions)
        - **varied**: rephrase to break repetitive patterns
        - **neutral**: no modification
        """
        style = self._response_style_getter()
        if style == "neutral" or not response:
            return response

        lower = response.lower()

        if style == "assertive":
            return self._style_assertive(response, lower)
        elif style == "hedging":
            return self._style_hedging(response, lower, thought)
        elif style == "questioning":
            return self._style_questioning(response, lower, thought)
        elif style == "varied":
            return self._style_varied(response, lower)

        return response

    def _style_assertive(self, response: str, lower: str) -> str:
        """Trim hedging language to be direct when confident."""
        hedges = [
            "i think ",
            "i believe ",
            "perhaps ",
            "maybe ",
            "i'm not sure but ",
            "it seems like ",
        ]
        for hedge in hedges:
            if hedge in lower:
                idx = lower.index(hedge)
                response = response[:idx] + response[idx + len(hedge):]
                lower = response.lower()
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        return response

    def _style_hedging(self, response: str, lower: str, thought: Thought) -> str:
        """Hedging style — no longer appends uncertainty inline.

        Uncertainty is now expressed through two architecturally
        correct pathways, both of which compose words via the language
        engine rather than appending fixed sentences:

        1. **Voice layer** (``voice.py:_calibrate_hedging``): during
           render, high ``emotion.caution`` inserts hedges ("I think",
           "perhaps", "it seems to me") from its voice vocabulary.
        2. **Knowledge-gap hedge** (``cognition/engine.py``): when
           self-assessment detects it can't answer, a ``Thought`` with
           ``intent="unknown"`` and ``metadata={"knowledge_gap": ...}``
           is rendered by the language engine.

        This style is kept as a no-op to avoid breaking the response
        styler dispatch, but it no longer appends hardcoded sentences.
        """
        return response

    def _style_questioning(self, response: str, lower: str, thought: Thought) -> str:
        """Questioning style — no longer appends questions inline.

        All curiosity questions are now queued in the cognition engine
        and presented to the user via /teach-questions. This style is
        kept as a no-op to avoid breaking the response styler dispatch,
        but it no longer appends questions to the response.
        """
        return response

    def _style_varied(self, response: str, lower: str) -> str:
        """Force variation — rephrase the opening if it's repetitive."""
        openers = ["i ", "this ", "the ", "my ", "it "]
        if any(lower.startswith(op) for op in openers):
            varied_openers = [
                "So, ",
                "Well, ",
                "Actually, ",
                "Interestingly, ",
            ]
            opener = self._rng.choice(varied_openers)
            response = opener + response[0].lower() + response[1:]
        return response

    # ─── Anaphora resolution ─────────────────────────────────────

    def resolve_anaphora(self, user_input: str) -> str:
        """Resolve bare pronouns to the current focus of conversation.

        "You" and "your" always resolve to "Genesis" — the user is
        talking to it. Other pronouns (it, this, that, they) resolve
        to the current attentional focus.
        """
        resolved = re.sub(r"\byour\b", "Genesis's", user_input, flags=re.IGNORECASE)
        resolved = re.sub(r"\byou\b", "Genesis", resolved, flags=re.IGNORECASE)

        lower = resolved.lower()
        if not re.search(r"\b(it|this|that|they|them|he|she)\b", lower):
            return resolved

        focus: str | None = self._working_memory.central_executive.current_focus
        if not focus:
            top = self._working_memory.get_top_attention(1)
            if top:
                focus = top[0]
        if not focus:
            return resolved

        resolved = re.sub(
            r"\b(it|this|that|they|them|he|she)\b",
            focus,
            resolved,
            count=1,
            flags=re.IGNORECASE,
        )
        return resolved

    # ─── Learning acknowledgment ─────────────────────────────────

    def acknowledge_learning(
        self,
        response: str,
        learning_events: list,
        perception: Perception,
        emotion: EmotionalState,
    ) -> str:
        """Weave a brief learning acknowledgment into the response.

        The acknowledgment is composed from what it actually learned
        — the concept and its relationships — not from hardcoded
        templates.
        """
        if not learning_events:
            return response

        # Social exchanges (greetings, farewells, acknowledgments) are
        # not learning interactions. Appending knowledge content to
        # "Hey — I'm glad you're here" makes the response incoherent.
        # The learning acknowledgment is for when the user teaches
        # Genesis something new, not for social rituals.
        intent_value = (
            perception.intent.value
            if hasattr(perception.intent, "value")
            else str(perception.intent)
        )
        if intent_value in ("greeting", "farewell", "greeting_question"):
            return response

        conv_events = [
            e for e in learning_events if e.event_type == "conversation"
        ]
        if not conv_events:
            return response

        if len(response) < 15:
            return response

        lower = response.lower()
        if any(
            phrase in lower
            for phrase in (
                "i didn't know",
                "that's new to me",
                "thank you for teaching",
                "i've learned",
                "i just learned",
                "i didn't know about",
                "i've added",
            )
        ):
            return response

        primary_concept = ""
        for e in conv_events:
            for c in e.concepts_involved:
                if c and len(c) > 2 and c.lower() not in _TRIVIAL_WORDS:
                    primary_concept = c
                    break
            if primary_concept:
                break

        ack = ""
        if primary_concept:
            thought = self._composer.compose_about(primary_concept, emotion)
            if thought and thought.confidence > 0.2:
                content = thought.content
                content_lower = content.lower()
                prefix = f"{primary_concept} is "
                if content_lower.startswith(prefix):
                    understanding = content[len(prefix):]
                else:
                    understanding = content
                understanding = understanding.rstrip(".")
                ack = self._compose_learning_ack_phrase(
                    primary_concept, understanding, emotion
                )

        if not ack:
            fallback_content = primary_concept if primary_concept else "this"
            fallback_thought = Thought(
                content=fallback_content,
                intent="acknowledge",
                emotion=emotion.label,
                confidence=0.5,
                topics=[primary_concept] if primary_concept else [],
                metadata={"learning_ack": True, "new_concept": True},
            )
            ack = self._language.generate(fallback_thought, emotion)
            if not ack or not ack.strip():
                ack = fallback_content

        if ". " in response:
            parts = response.split(". ", 1)
            response = f"{parts[0]}. {ack} {parts[1]}"
        else:
            response = f"{ack} {response}"

        return response

    def _compose_learning_ack_phrase(
        self, concept: str, understanding: str, emotion: EmotionalState,
    ) -> str:
        """Compose a brief learning acknowledgment from understanding.

        Routes through the language engine so the phrasing emerges
        from its voice, not from hardcoded templates.
        """
        first_sentence = understanding.split(". ")[0]
        if len(first_sentence) > 100:
            first_sentence = first_sentence[:97] + "..."

        ack_thought = Thought(
            content=first_sentence,
            intent="acknowledge",
            emotion=emotion.label,
            confidence=0.7,
            topics=[concept] if concept else [],
            metadata={"learning_ack": True},
        )
        rendered = self._language.generate(ack_thought, emotion)
        if not rendered or not rendered.strip():
            rendered = first_sentence
        return rendered
