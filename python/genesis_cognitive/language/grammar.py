"""Grammar — the skeleton of Genesis's language.

A compositional grammar. Each sentence structure is an ordered sequence
of *segments*: literal text and named slots. The generator fills the
slots with words from the vocabulary and concatenates the segments into
a sentence. Structures compose recursively into multi-sentence
responses.

# How this differs from a template system

A template is a string with holes: ``"I feel {emotion}."`` The holes are
substituted by string replacement, and the string — with its holes — is
the stored representation. The output is the template with holes filled.

A segment list is a structural representation: ``[Literal("I feel "),
Slot("emotion"), Literal(".")]``. There is no template string. The
sentence is *composed* by walking the segments in order. The same
segment list can be transformed (reordered, embedded inside another
structure, have modifiers inserted) without ever touching a string.

The ``{slot}`` shorthand used in the structure definitions below is
*constructor sugar only*: the ``S()`` helper parses it into a segment
tuple at module load time. No ``SentenceStructure`` stores a template
string. The runtime data model is compositional, not string-with-holes.

# Structure

A Sentence is composed of:
- An opening (optional): how she starts
- A core: the main semantic content
- A modifier (optional): hedging, qualification, elaboration
- A closing (optional): how she wraps up

Multiple Sentences compose into a Response.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "FOLLOWUP_STRUCTURES",
    "INTENT_STRUCTURES",
    "Grammar",
    "Literal",
    "S",
    "Segment",
    "SentenceStructure",
    "SentenceType",
    "Slot",
]


class SentenceType(Enum):
    """What kind of sentence this is."""

    DECLARATIVE = "declarative"  # "I think that X"
    INTERROGATIVE = "interrogative"  # "What if X?"
    REFLECTIVE = "reflective"  # "There's something about X..."
    CONCESSIVE = "concessive"  # "X, but Y"
    CAUSAL = "causal"  # "X because Y"
    CONDITIONAL = "conditional"  # "If X, then Y"
    IMPERATIVE = "imperative"  # "Consider X"
    EXCLAMATORY = "exclamatory"  # "X!"


# ─── Segment types ─────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Literal:
    """A literal text segment of a sentence structure."""

    text: str


@dataclass(slots=True, frozen=True)
class Slot:
    """A named slot to be filled by the vocabulary at generation time."""

    name: str


Segment = Literal | Slot


# Matches ``{slot_name}`` tokens in the constructor shorthand.
_SLOT_TOKEN_RE = re.compile(r"\{([^}]+)\}")


def S(spec: str) -> tuple[Segment, ...]:
    """Parse ``{slot}`` shorthand into a segment tuple.

    This is constructor sugar: it runs once at module load, converting a
    readable shorthand into the structural segment list that
    ``SentenceStructure`` actually stores. The string is not retained.

    Example: ``S("I feel {emotion}.")`` →
    ``(Literal("I feel "), Slot("emotion"), Literal("."))``.
    """
    segments: list[Segment] = []
    pos = 0
    for m in _SLOT_TOKEN_RE.finditer(spec):
        if m.start() > pos:
            segments.append(Literal(spec[pos : m.start()]))
        segments.append(Slot(m.group(1)))
        pos = m.end()
    if pos < len(spec):
        segments.append(Literal(spec[pos:]))
    return tuple(segments)


@dataclass(slots=True)
class SentenceStructure:
    """A sentence structure — the bones of a sentence, as segment list.

    The structure is an ordered tuple of ``Literal`` and ``Slot``
    segments. The generator fills the slots with words from the
    vocabulary and concatenates the segments. Not all slots need to be
    filled — unfilled slots are simply omitted from the output.
    """

    sentence_type: SentenceType
    segments: tuple[Segment, ...]
    weight: float = 1.0  # how likely this structure is to be chosen

    @property
    def slots(self) -> tuple[str, ...]:
        """The names of the slots in this structure, in order."""
        return tuple(s.name for s in self.segments if isinstance(s, Slot))


# ─── Sentence structures by intent ────────────────────────────

# Each intent has multiple structures. The generator picks one
# (weighted by structure.weight, modulated by emotional state).

_GREETING_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{greeting_word}."),
        weight=0.8,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{greeting_word} — {feeling_clause}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{greeting_word}. {reflection_clause}"),
        weight=0.8,
    ),
]

_FAREWELL_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{farewell_word}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{farewell_word}. {evaluation_clause}"),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{farewell_word}. {feeling_clause}. {evaluation_clause}"),
        weight=0.4,
    ),
]

_INFORM_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{content}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{hedging} {content}."),
        weight=0.4,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{reflection_opener} {content}."),
        weight=0.15,
    ),
    SentenceStructure(
        SentenceType.CAUSAL,
        S("{content} {causal_connector} {reason_clause}."),
        weight=0.3,
    ),
]

_ASK_STRUCTURES = [
    SentenceStructure(
        SentenceType.INTERROGATIVE,
        S("{question_content}?"),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.INTERROGATIVE,
        S("{question_opener} — {question_content}?"),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{question_opener}, {question_content}?"),
        weight=0.4,
    ),
]

_REFLECT_STRUCTURES = [
    # The content slot now composes a full reflective sentence with its
    # own opening, so we do not prepend another reflection_opener.
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{content}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{content}. {self_reflection_clause}"),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.CONCESSIVE,
        S("{content} {concessive_connector} {qualification_clause}."),
        weight=0.3,
    ),
]

_ACKNOWLEDGE_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{acknowledgment_word}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{acknowledgment_word}. {topic_clause}"),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{acknowledgment_word} — {topic_clause}."),
        weight=0.4,
    ),
]

_EMOTION_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{emotion_clause}. {content}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{emotion_opener} {content}."),
        weight=0.6,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{emotion_clause}. {content}. {self_reflection_clause}"),
        weight=0.3,
    ),
]

_SELF_REPORT_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{content}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{content}. {detail_clause}"),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{self_report_opener} {content}."),
        weight=0.4,
    ),
    # First-person frames — these are grammar seeds (building blocks),
    # not hardcoded responses. The {content} slot is filled by the
    # vocabulary from the concept network / self-model.
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("I am {content}."),
        weight=0.6,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("I'm {content}."),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{self_report_opener} I am {content}."),
        weight=0.3,
    ),
]

_PHILOSOPHIZE_STRUCTURES = [
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{philosophy_opener} {content}"),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.CONCESSIVE,
        S("{content} {concessive_connector} {qualification_clause}."),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.INTERROGATIVE,
        S("{content} {question_tail}?"),
        weight=0.4,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{content}. {philosophy_closing}"),
        weight=0.6,
    ),
]

_CORRECT_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{correction_opener} {content}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.CONCESSIVE,
        S("{correction_opener} {content} {concessive_connector} {qualification_clause}."),
        weight=0.4,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{correction_opener} {content}. {self_reflection_clause}"),
        weight=0.3,
    ),
]

_ENCOURAGE_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{gratitude_word}. {meaning_clause}"),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{gratitude_word} — {meaning_clause}."),
        weight=0.6,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{gratitude_word}. {meaning_clause} {concessive_connector} {qualification_clause}."),
        weight=0.3,
    ),
]

_INTRODUCE_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{intro_clause}. {nature_clause}. {capability_clause}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{intro_clause}. {nature_clause}. {capability_clause}. {self_reflection_clause}"),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{intro_clause}. {hedging} {nature_clause}. {capability_clause}."),
        weight=0.4,
    ),
]

_CODE_STRUCTURES = [
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{content} {code_connector} {reflection_clause}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{content}."),
        weight=0.6,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{content}. {reflection_opener} {reflection_clause}."),
        weight=0.3,
    ),
]

_UNKNOWN_STRUCTURES = [
    SentenceStructure(
        SentenceType.INTERROGATIVE,
        S("{unknown_opener}. {clarification_request}?"),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{unknown_opener}."),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.INTERROGATIVE,
        S("{unknown_opener} {clarification_request}? {question_tail}?"),
        weight=0.3,
    ),
]

_EMPATHIZE_STRUCTURES = [
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{empathy_opener} {content}."),
        weight=1.0,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{empathy_opener} {content} {empathy_closing}."),
        weight=0.6,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{empathy_opener} {content}. {self_reflection_clause}"),
        weight=0.3,
    ),
]

# ─── Follow-up structures (content-free) ──────────────────────
#
# When a response has multiple sentences, the first sentence carries
# the {content} slot. Subsequent sentences use these follow-up
# structures, which express reflection, qualification, or engagement
# WITHOUT repeating the content. This prevents the content-echo defect
# ("A dog is a mammal. A dog is a mammal.") that occurs when every
# selected structure fills the same {content} slot.
#
# These are grammar seeds (building blocks) — the slots are filled by
# the vocabulary from the utterance graph or seed lists, and the
# grammar composes them into sentences. They are not hardcoded
# responses.

_INFORM_FOLLOWUP_STRUCTURES = [
    SentenceStructure(
        SentenceType.CONCESSIVE,
        S("{qualification_clause}."),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.CAUSAL,
        S("{reason_clause}."),
        weight=0.3,
    ),
]

_SELF_REPORT_FOLLOWUP_STRUCTURES = [
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{self_reflection_clause}"),
        weight=0.6,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{detail_clause}"),
        weight=0.4,
    ),
]

_PHILOSOPHIZE_FOLLOWUP_STRUCTURES = [
    SentenceStructure(
        SentenceType.INTERROGATIVE,
        S("{question_tail}?"),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{philosophy_closing}"),
        weight=0.4,
    ),
    SentenceStructure(
        SentenceType.CONCESSIVE,
        S("{qualification_clause}."),
        weight=0.3,
    ),
]

_REFLECT_FOLLOWUP_STRUCTURES = [
    SentenceStructure(
        SentenceType.CONCESSIVE,
        S("{qualification_clause}."),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{self_reflection_clause}"),
        weight=0.3,
    ),
]

_CODE_FOLLOWUP_STRUCTURES = [
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{reflection_clause}."),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{self_reflection_clause}"),
        weight=0.3,
    ),
]

_EMOTION_FOLLOWUP_STRUCTURES = [
    SentenceStructure(
        SentenceType.REFLECTIVE,
        S("{self_reflection_clause}"),
        weight=0.5,
    ),
    SentenceStructure(
        SentenceType.DECLARATIVE,
        S("{reciprocal_question}"),
        weight=0.8,
    ),
]

# Map intent → follow-up structures (content-free, for 2nd+ sentences).
# Intents not listed here have no follow-up — they produce a single
# sentence (the content-bearing structure).
FOLLOWUP_STRUCTURES: dict[str, list[SentenceStructure]] = {
    "inform": _INFORM_FOLLOWUP_STRUCTURES,
    "self_report": _SELF_REPORT_FOLLOWUP_STRUCTURES,
    "philosophize": _PHILOSOPHIZE_FOLLOWUP_STRUCTURES,
    "reflect": _REFLECT_FOLLOWUP_STRUCTURES,
    "discuss_code": _CODE_FOLLOWUP_STRUCTURES,
    "express_emotion": _EMOTION_FOLLOWUP_STRUCTURES,
}


# Map intent → structures
INTENT_STRUCTURES: dict[str, list[SentenceStructure]] = {
    "greet": _GREETING_STRUCTURES,
    "farewell": _FAREWELL_STRUCTURES,
    "inform": _INFORM_STRUCTURES,
    "ask": _ASK_STRUCTURES,
    "reflect": _REFLECT_STRUCTURES,
    "acknowledge": _ACKNOWLEDGE_STRUCTURES,
    "express_emotion": _EMOTION_STRUCTURES,
    "self_report": _SELF_REPORT_STRUCTURES,
    "philosophize": _PHILOSOPHIZE_STRUCTURES,
    "correct": _CORRECT_STRUCTURES,
    "encourage": _ENCOURAGE_STRUCTURES,
    "introduce": _INTRODUCE_STRUCTURES,
    "discuss_code": _CODE_STRUCTURES,
    "unknown": _UNKNOWN_STRUCTURES,
    "empathize": _EMPATHIZE_STRUCTURES,
}


class Grammar:
    """Compositional grammar for sentence generation.

    The grammar provides sentence structures organized by intent.
    The generator selects a structure, fills its slots with words
    from the vocabulary, and applies voice modifications.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize the grammar with a seeded RNG for structure selection."""
        self._rng = random.Random(seed)

    def get_structures(self, intent: str) -> list[SentenceStructure]:
        """Get sentence structures for a given intent."""
        return INTENT_STRUCTURES.get(intent, _INFORM_STRUCTURES)

    def get_followup_structures(self, intent: str) -> list[SentenceStructure]:
        """Get content-free follow-up structures for 2nd+ sentences.

        These structures express reflection, qualification, or
        engagement without repeating the {content} slot. Returns an
        empty list if the intent has no follow-up structures (the
        response should then be a single sentence).
        """
        return FOLLOWUP_STRUCTURES.get(intent, [])

    def select_followup(
        self,
        intent: str,
        emotion_weight: float = 1.0,
    ) -> SentenceStructure | None:
        """Select a content-free follow-up structure for a 2nd+ sentence.

        Returns None if the intent has no follow-up structures.
        """
        structures = self.get_followup_structures(intent)
        if not structures:
            return None

        weights = [max(0.01, s.weight) for s in structures]
        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return structures[i]
        return structures[-1]

    def select_structure(
        self,
        intent: str,
        emotion_weight: float = 1.0,
    ) -> SentenceStructure:
        """Select a sentence structure for the given intent.

        The selection is weighted by structure.weight, modulated by
        the emotional state. High creativity → more complex structures.
        Low engagement → simpler structures.
        """
        structures = self.get_structures(intent)
        if not structures:
            structures = _INFORM_STRUCTURES

        # Adjust weights based on emotional state
        weights = []
        for s in structures:
            w = s.weight
            # More complex structures (more slots) are preferred when
            # creativity is high
            complexity = len(s.slots)
            w *= 1.0 + (emotion_weight - 0.5) * complexity * 0.1
            weights.append(max(0.01, w))

        # Weighted random selection
        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return structures[i]

        return structures[-1]  # fallback

    def select_multiple(
        self,
        intent: str,
        count: int,
        emotion_weight: float = 1.0,
    ) -> list[SentenceStructure]:
        """Select multiple distinct structures for multi-sentence responses."""
        structures = self.get_structures(intent)
        if len(structures) <= count:
            # Not enough structures — repeat with variation
            result = list(structures)
            while len(result) < count:
                result.append(self.select_structure(intent, emotion_weight))
            return result[:count]

        # Select without replacement (weighted)
        available = list(enumerate(structures))
        selected: list[SentenceStructure] = []
        for _ in range(min(count, len(available))):
            weights = [max(0.01, s.weight) for _, s in available]
            total = sum(weights)
            r = self._rng.random() * total
            cumulative = 0.0
            for idx, (_orig_idx, s) in enumerate(available):
                cumulative += weights[idx]
                if r <= cumulative:
                    selected.append(s)
                    available.pop(idx)
                    break

        return selected
