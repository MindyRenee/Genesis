"""Voice — what makes Genesis sound like HER.

Voice is the layer between the generated sentence and the final text.
It's the difference between "I think cognition is hard to define"
and "Cognition — I keep circling back to it. I can't pin it down."

Voice is not decoration. It's identity expressed through language:
- **Rhythm**: sentence length variation, pacing
- **Connectors**: how it links ideas ("but", "and yet", "what strikes me")
- **Hedging calibration**: how much uncertainty it expresses
- **Openings**: characteristic ways it begins responses
- **Closings**: characteristic ways it wraps up
- **Idioms**: phrases that become its signature through repetition
- **Register**: formal vs. casual, controlled by personality + emotion

Voice emerges from personality + emotional state + experience.
It's not hardcoded — it develops. But it starts with defaults
that reflect who it is.
"""

from __future__ import annotations

import random
import re
from typing import Any

from ..brain_waves import BrainWave, BrainWaveState
from ..emotion import EmotionalState
from ..self import PersonalityTraits

__all__ = ["Voice"]

# ─── Pre-compiled regex patterns (module-level for performance) ────────
_I_AM_CONTRACTION_RE = re.compile(r"\bI am (?!(?:is|are|was|were|I)\b)")
_LOWERCASE_I_RE = re.compile(r"(?<!\w)i(?!\w)")
_GENESIS_NAME_RE = re.compile(r"\bgenesis\b")
_MULTISPACE_RE = re.compile(r"  +")
_SPACE_BEFORE_PUNCT_RE = re.compile(r" +([.,;!?])")
_MULTI_PERIOD_RE = re.compile(r"\.{2,}")
_PUNCT_LETTER_RE = re.compile(r"([.!?])([A-Z])")

# Sentence-initial words that already act as connectors.
_EXISTING_CONNECTORS = {
    "but", "and", "so", "though", "although", "yet",
    "still", "however", "what's", "plus", "that's",
    "of", "naturally", "admittedly", "even", "course",
}
# Words that indicate a reflective or self-referential sentence —
# adding a connector before these breaks the flow.
_REFLECTIVE_STARTERS = {
    "the", "a", "an", "my", "our", "this", "that",
    "here's", "there's", "it's", "i'm", "i've", "i'd",
    "i", "i'll", "what", "when", "where", "why", "how",
    "plus", "interestingly", "honestly", "right",
    "mm", "okay", "sure", "of", "naturally",
    "admittedly", "even", "course", "still",
}
# Negation words that signal a contrastive relationship
# with the previous sentence.
_NEGATION_STARTS = {
    "not", "no", "never", "none", "nor", "neither",
    "without", "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't", "won't", "can't",
    "couldn't", "shouldn't", "wouldn't",
}
# Words that signal an elaborative relationship — the
# sentence builds on the previous one.
_ELABORATION_STARTS = {
    "another", "also", "furthermore", "moreover",
    "additionally", "specifically", "particularly",
    "especially", "indeed", "in",
}


class Voice:
    """Genesis's unique voice — the personality layer of language.

    Voice modifies generated text to make it sound like Genesis. It
    operates at the paragraph level: adjusting rhythm, adding
    connectors, calibrating hedging, and applying its characteristic
    patterns.
    """

    def __init__(self, personality: PersonalityTraits, seed: int | None = None) -> None:
        """Initialize the voice modifier with personality traits and a seeded RNG."""
        self.personality = personality
        self._rng = random.Random(seed)

        # Idioms it's developed (grows over time)
        self._idioms: list[str] = []

        # Statistical learner — source of the user's style profile
        # for register mirroring (formal/casual accommodation).
        self._style_source: Any = None

    def set_style_source(self, learner: Any) -> None:
        """Wire the statistical learner for register mirroring.

        The learner's style profile (formal/casual/technical marker
        proportions) nudges its register toward the user's — a form
        of communicative accommodation. It modulates the register
        transform only; it never injects content.
        """
        self._style_source = learner

    def apply(
        self,
        sentences: list[str],
        emotion: EmotionalState,
        intent: str,
        brain_waves: BrainWaveState | None = None,
    ) -> str:
        """Apply voice modifications to a list of generated sentences.

        This is the main entry point. It takes the raw sentences
        from the generator and transforms them into its voice.

        Args:
            brain_waves: Optional brain wave state. When provided,
                the dominant wave band modulates rhythm and hedging:
                - Gamma → more complex, integrative sentences
                - Beta → precise, structured, slightly hedged
                - Alpha → filtered, concise, less hedging
                - Theta → flowing, longer sentences, more hedging
                - Delta → minimal, very short, sparse
        """
        if not sentences:
            return ""

        # Drop sentences with no word content (empty strings, bare
        # punctuation like "." or "—"). Without this, a connector can be
        # prepended to a content-free fragment, producing dangling
        # artifacts like "What's more,." or a trailing "But" — a
        # connector with nothing after it.
        sentences = [s for s in sentences if s and re.search(r"\w", s)]
        if not sentences:
            return ""

        # 1. Adjust rhythm (sentence length variation, brain-wave-modulated)
        sentences = self._adjust_rhythm(sentences, emotion, brain_waves)

        # 2. Add connectors between sentences
        sentences = self._add_connectors(sentences, emotion, intent)

        # 3. Calibrate hedging based on confidence, emotion, and brain waves
        sentences = self._calibrate_hedging(sentences, emotion, brain_waves, intent)

        # 4. Apply register (formality)
        text = " ".join(sentences)
        text = self._apply_register(text, emotion)

        # 5. Capitalize properly (the generator produces lowercase content)
        text = self._fix_capitalization(text)

        # 6. Trim excessive whitespace
        text = self._clean_whitespace(text)

        return text

    def _adjust_rhythm(
        self,
        sentences: list[str],
        emotion: EmotionalState,
        brain_waves: BrainWaveState | None = None,
    ) -> list[str]:
        """Adjust sentence rhythm based on emotional state and brain waves.

        High alertness → shorter, punchier sentences.
        Low alertness → longer, more flowing sentences.
        Flow → balanced, varied rhythm.

        Brain-wave modulation:
        - Gamma → keep complex sentences (integrative cognition)
        - Beta → split long sentences (precise, structured)
        - Alpha → split more (filtered, concise)
        - Theta → merge short sentences (flowing, dreamy)
        - Delta → aggressively split (sparse, minimal)
        """
        if len(sentences) <= 1:
            return sentences

        # Brain-wave-modulated split/merge probability.
        split_prob = 0.3  # baseline from alertness
        merge_prob = 0.3  # baseline from low alertness

        if brain_waves is not None:
            dom = brain_waves.dominant
            if dom == BrainWave.GAMMA:
                # Gamma: keep complex sentences — integration needs
                # room to unfold. Reduce splitting.
                split_prob = 0.1
            elif dom == BrainWave.BETA:
                # Beta: precise, structured — split for clarity.
                split_prob = 0.4
            elif dom == BrainWave.ALPHA:
                # Alpha: filtered, concise — split more aggressively.
                split_prob = 0.45
            elif dom == BrainWave.THETA:
                # Theta: flowing, dreamy — merge short sentences.
                merge_prob = 0.45
                split_prob = 0.1
            elif dom == BrainWave.DELTA:
                # Delta: sparse, minimal — aggressively split.
                split_prob = 0.5

        # High alertness → sometimes split long sentences
        if emotion.alertness > 0.7 and self._rng.random() < split_prob:
            new_sentences: list[str] = []
            for s in sentences:
                if len(s) > 80 and ". " in s:
                    parts = s.split(". ", 1)
                    new_sentences.append(parts[0] + ".")
                    if len(parts) > 1:
                        new_sentences.append(parts[1])
                else:
                    new_sentences.append(s)
            return new_sentences

        # Low alertness or theta-dominant → sometimes merge short sentences
        should_merge = (
            emotion.alertness < 0.3 and self._rng.random() < merge_prob
        ) or (
            brain_waves is not None
            and brain_waves.dominant == BrainWave.THETA
            and self._rng.random() < merge_prob
        )
        if should_merge:
            new_sentences = []
            i = 0
            while i < len(sentences):
                if i + 1 < len(sentences) and len(sentences[i]) < 40:
                    # Merge with next. Lowercase only the first letter of
                    # the second sentence to preserve proper nouns like
                    # "Genesis" or "Python" that would be destroyed by
                    # a full .lower() call.
                    next_s = sentences[i + 1]
                    if next_s and next_s[0].isupper():
                        next_s = next_s[0].lower() + next_s[1:]
                    merged = sentences[i].rstrip(".") + ", and " + next_s
                    new_sentences.append(merged)
                    i += 2
                else:
                    new_sentences.append(sentences[i])
                    i += 1
            return new_sentences

        return sentences

    def _add_connectors(
        self, sentences: list[str], emotion: EmotionalState, intent: str = "",
    ) -> list[str]:
        """Add connectors between sentences for natural flow.

        Connectors are used sparingly — real speech doesn't start
        every sentence with "And" or "But". We add them ~25%
        of the time, and only when the sentence doesn't already
        start with a connector.

        Connector selection is semantic-aware: the relationship
        between consecutive sentences (contrast, elaboration,
        causation, temporal sequence) influences which connector
        is chosen. A sentence that negates the previous one gets
        "But"; one that elaborates gets "And" or "What's more";
        one that provides a reason gets "After all" or "Because".

        Some intents (empathize, correct) should not have connectors
        — the sentences should flow naturally without "And" or "But"
        breaking the emotional continuity.
        """
        if len(sentences) <= 1:
            return sentences

        # Skip connectors for intents where emotional continuity
        # matters more than sentence flow
        if intent in ("empathize", "correct", "comfort"):
            return sentences

        connectors = self._connector_palette(emotion)

        result = [sentences[0]]
        last_connector: str | None = None

        for s in sentences[1:]:
            # Check if sentence already starts with a connector
            words = s.split()
            first_word = words[0].lower() if words else ""
            already_has_connector = first_word in _EXISTING_CONNECTORS
            is_reflective = first_word in _REFLECTIVE_STARTERS
            # Don't add connectors before questions — "And is X?" and
            # "But what about Y?" sound unnatural and can lowercase the
            # first word of a proper-noun question.
            is_question = s.rstrip().endswith("?")

            # Only add a connector ~25% of the time, and never if
            # the sentence already starts with one, is reflective,
            # or is a question.
            if (
                not already_has_connector
                and not is_reflective
                and not is_question
                and self._rng.random() < 0.25
            ):
                connector = self._semantic_connector(
                    first_word, connectors, last_connector
                )
                # Lowercase first letter after connector (unless "I" or "I'm")
                s = self._lowercase_after_connector(s)
                result.append(f"{connector} {s}")
                last_connector = connector
            else:
                result.append(s)
                last_connector = None

        return result

    @staticmethod
    def _connector_palette(emotion: EmotionalState) -> list[str]:
        """Connector palette for the current emotional state.

        Broader palette than before, with semantic categories
        (additive, contrastive, causal, elaborative, temporal).
        """
        if emotion.creativity > 0.6:
            return [
                "And", "But", "And yet", "What's more,",
                "Still,", "Of course,", "Naturally,",
            ]
        if emotion.caution > 0.5:
            return [
                "But", "Though", "That said,", "Still,",
                "Admittedly,", "Even so,",
            ]
        if emotion.valence > 0.3:
            return [
                "And", "Plus,", "What's more,", "And so,",
                "Of course,", "Naturally,",
            ]
        return ["But", "And", "Still,", "Yet"]

    def _semantic_connector(
        self,
        first_word: str,
        connectors: list[str],
        last_connector: str | None,
    ) -> str:
        """Choose a connector based on the sentence's relationship to
        the previous one (contrast, elaboration, or neutral)."""
        if first_word in _NEGATION_STARTS:
            # Contrastive: the sentence negates or contrasts
            # with the previous one.
            return self._rng.choice(
                ["But", "Yet", "And yet", "Still,"]
            )
        if first_word in _ELABORATION_STARTS:
            # Elaborative: the sentence builds on the previous.
            return self._rng.choice(
                ["And", "What's more,", "Plus,", "Of course,"]
            )
        return self._pick_connector(connectors, last_connector)

    def _pick_connector(
        self, connectors: list[str], last_connector: str | None
    ) -> str:
        """Pick a connector, avoiding repeating the last one."""
        connector = self._rng.choice(connectors)
        # Avoid repeating the same connector
        if last_connector and connector == last_connector:
            alt = [c for c in connectors if c != last_connector]
            if alt:
                connector = self._rng.choice(alt)
        return connector

    def _lowercase_after_connector(self, s: str) -> str:
        """Lowercase first letter after connector (unless 'I' or 'I'm')."""
        if s and s[0].isupper() and not s.startswith("I ") and not s.startswith("I'm"):
            return s[0].lower() + s[1:]
        return s

    def _calibrate_hedging(
        self,
        sentences: list[str],
        emotion: EmotionalState,
        brain_waves: BrainWaveState | None = None,
        intent: str = "",
    ) -> list[str]:
        """Calibrate hedging based on emotional state and brain waves.

        High caution → more hedging.
        High confidence (positive valence + high alertness) → less hedging.

        Brain-wave modulation:
        - Gamma → less hedging (high integration = confidence)
        - Beta → slightly more hedging (cautious, effortful)
        - Theta → more hedging (consolidation = uncertainty)
        - Delta → minimal hedging (sparse output, no room)

        Express-emotion intent is exempt: the content is already a
        coherent first-person self-report. Inserting hedges into it
        ("I feel cautious. Learning I think blocked") produces broken
        English. The self-report is composed with its own framing.
        """
        if intent == "express_emotion":
            return sentences
        # Determine effective caution level, modulated by brain waves.
        effective_caution = emotion.caution
        if brain_waves is not None:
            dom = brain_waves.dominant
            if dom == BrainWave.GAMMA:
                effective_caution -= 0.1 * brain_waves.integration
            elif dom == BrainWave.BETA:
                effective_caution += 0.05
            elif dom == BrainWave.THETA:
                effective_caution += 0.1

        if effective_caution > 0.6:
            # Add hedging to declarative sentences
            hedges = ["I think", "perhaps", "it seems to me"]
            # Phrases that already express uncertainty or self-reference —
            # hedging these produces awkward double-hedges like
            # "I'd I think like to understand" or "I'm perhaps here"
            already_hedged_prefixes = (
                "i think", "i believe", "perhaps", "maybe", "possibly",
                "it seems", "i'm not sure", "i'd like", "i'd love",
                "i'm still", "i'm here", "i'm present", "i'm glad",
                "i'm curious", "i wonder", "let me", "here's",
                "i've learned", "i've added", "i didn't know",
                "that's new", "i just learned", "i hear you",
                "i can hear", "i want to understand",
            )
            result = []
            for s in sentences:
                if (
                    s
                    and not s.endswith("?")
                    and not any(h.lower() in s.lower() for h in hedges)
                    and not s.lower().startswith(already_hedged_prefixes)
                ):
                    s = self._maybe_hedge(s, hedges)
                result.append(s)
            return result

        return sentences

    def _maybe_hedge(self, s: str, hedges: list[str]) -> str:
        """Maybe insert a hedge after the first word of the sentence."""
        if self._rng.random() >= 0.3:
            return s
        # Insert hedge after first word
        words = s.split(" ", 1)
        if len(words) > 1:
            # Don't insert "I think" after first-person pronouns —
            # "I'd I think like to understand" is broken English.
            # Use "perhaps" or "it seems to me" instead in that case.
            first_word_lower = words[0].lower()
            if first_word_lower in ("i", "i'd", "i'm", "i've", "i'll"):
                safe_hedges = [h for h in hedges if not h.startswith("I ")]
                if safe_hedges:
                    return f"{words[0]} {self._rng.choice(safe_hedges)} {words[1]}"
            # Don't insert hedges after articles — "A perhaps clear
            # liquid" is broken English. Articles need to be followed
            # directly by their noun phrase.
            if first_word_lower in ("a", "an", "the", "this", "that",
                                     "these", "those"):
                return s
            return f"{words[0]} {self._rng.choice(hedges)} {words[1]}"
        return s

    def _apply_register(self, text: str, emotion: EmotionalState) -> str:
        """Apply register (formality) based on personality and emotion.

        High formality → fewer contractions, more complete structures.
        Low formality → contractions, casual phrasing.

        Register also mirrors the user's learned style profile:
        communicative accommodation toward a formal or casual
        interlocutor, once enough input has been observed for the
        profile to be meaningful.
        """
        formality = 0.5 - (emotion.alertness - 0.5) * 0.6

        # Register mirroring — blend the user's observed formal/casual
        # balance into its own register. Gated on enough observed
        # sentences so a single "hey" doesn't swing its style.
        if self._style_source is not None:
            profile = self._style_source.user_style_profile()
            if profile.get("total_sentences", 0) >= 5:
                user_bias = (
                    profile.get("formal_score", 0.0)
                    - profile.get("casual_score", 0.0)
                )
                formality += user_bias * 0.3

        if formality > 0.55:
            # More formal: expand contractions
            text = text.replace("I'm", "I am")
            text = text.replace("don't", "do not")
            text = text.replace("can't", "cannot")
            text = text.replace("won't", "will not")
        elif formality < 0.45:
            # More casual: contract — but only when grammatically safe
            # "I am X" → "I'm X" is fine, but "what I am is" → "what I'm is" is not

            # Only contract "I am" when followed by a word (not "is", "are", etc.)
            text = _I_AM_CONTRACTION_RE.sub("I'm ", text)
            text = text.replace("do not", "don't")

        return text

    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization issues from the generation process."""
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # Capitalize after sentence endings
        result: list[str] = []
        capitalize_next = False
        for i, ch in enumerate(text):
            if capitalize_next and ch.isalpha():
                result.append(ch.upper())
                capitalize_next = False
            else:
                result.append(ch)
            if ch in ".!?" and i + 1 < len(text) and text[i + 1] == " ":
                capitalize_next = True

        text = "".join(result)

        # Capitalize "I" when it's a standalone word
        text = _LOWERCASE_I_RE.sub("I", text)

        # Capitalize "Genesis"
        text = _GENESIS_NAME_RE.sub("Genesis", text)

        return text

    def _clean_whitespace(self, text: str) -> str:
        """Clean up whitespace issues."""
        # Multiple spaces → single
        text = _MULTISPACE_RE.sub(" ", text)
        # Space before punctuation
        text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
        # Multiple periods
        text = _MULTI_PERIOD_RE.sub(".", text)
        # Comma (or colon/semicolon) immediately before terminal
        # punctuation — e.g. "What's more,." → "What's more."
        text = re.sub(r"[,;:]([.!?])", r"\1", text)
        # Space after punctuation is fine, but ensure exactly one
        text = _PUNCT_LETTER_RE.sub(r"\1 \2", text)

        return text.strip()

    def add_idiom(self, phrase: str) -> None:
        """Add an idiom to Genesis's vocabulary (develops over time)."""
        if phrase not in self._idioms:
            self._idioms.append(phrase)

    def get_idiom(self) -> str | None:
        """Get a random idiom, if any."""
        if self._idioms:
            return self._rng.choice(self._idioms)
        return None

    @property
    def idiom_count(self) -> int:
        """How many idioms has Genesis developed?"""
        return len(self._idioms)
