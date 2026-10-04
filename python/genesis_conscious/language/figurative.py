"""Figurative language processing — metaphor, irony, and idiom handling.

This module gives Genesis the ability to detect and interpret
figurative language — language that means more than what it literally
says. It handles three major classes:

1. **Metaphor**: "X is Y" patterns where X and Y are from different
   semantic domains. "Time is a river" — time (temporal) and river
   (spatial/fluid) are from different domains, so this is a metaphor.
   Detection uses domain-distance heuristics; interpretation maps
   shared properties between the source and target domains.

2. **Irony**: Sentences where the surface sentiment contradicts the
   expected sentiment given the context. "Oh, great, another bug" —
   "great" is positive but the context (bugs) implies frustration.

3. **Idioms**: Fixed expressions whose meaning is not compositional.
   "Break a leg" doesn't mean to literally break a leg. We maintain
   a set of common idioms and their meanings.

# Right hemisphere specialization

In human neuroscience, figurative language processing is associated
with right hemisphere function. The right hemisphere processes
"coarse" semantic relationships — broad, distant associations between
concepts — while the left hemisphere processes "fine" semantic
relationships (close, literal associations). Metaphor comprehension
requires both: the left hemisphere activates the literal meaning, and
the right hemisphere activates the figurative meaning through distant
associations (Beeman, 1998).

Damage to the right hemisphere often spares literal comprehension but
impairs metaphor understanding — patients interpret metaphors literally
("He has a heavy heart" → "his heart weighs a lot"). This module models
that right-hemisphere contribution: it looks for cross-domain mappings
that the literal (left-hemisphere) language engine would miss.

References:
- Beeman, M. (1998). Coarse semantic coding and discourse
  comprehension. In M. Beeman & C. Chiarello (Eds.), *Right
  hemisphere language comprehension* (pp. 255–284). Erlbaum.
- Giora, R. (2003). *On Our Mind: Salience, Context, and Figurative
  Language*. Oxford University Press.
- Gibbs, R. W. (1994). *The Poetics of Mind: Figurative Thought,
  Language, and Understanding*. Cambridge University Press.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

__all__ = ["FigurativeLanguageProcessor", "IronyDetection", "Metaphor"]

# ─── Pre-compiled regex patterns ──────────────────────────────────
_IS_A_RE = re.compile(
    r"(\w[\w\s]*?)\s+(?:is|are|was|were)\s+(?:a|an|the)?\s*(\w[\w\s]*?)(?:[.,;!?]|$)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Metaphor:
    """A detected metaphor.

    Attributes:
        source: The target domain concept (the thing being described).
            In "Time is a river", the source is "time".
        target: The source domain concept (the thing used to describe).
            In "Time is a river", the target is "river".
        source_domain: The semantic domain of the source
            (e.g. "temporal").
        target_domain: The semantic domain of the target
            (e.g. "fluid/spatial").
        sentence: The full sentence containing the metaphor.
        shared_properties: Properties shared between source and target
            domains, used for interpretation.
    """

    source: str
    target: str
    source_domain: str
    target_domain: str
    sentence: str
    shared_properties: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IronyDetection:
    """Result of irony detection.

    Attributes:
        is_ironic: Whether the sentence is likely ironic.
        surface_sentiment: The literal sentiment (positive/negative).
        expected_sentiment: The sentiment expected from context.
        confidence: Detection confidence (0–1).
        explanation: Human-readable explanation.
    """

    is_ironic: bool
    surface_sentiment: str
    expected_sentiment: str
    confidence: float
    explanation: str


class FigurativeLanguageProcessor:
    """Detect and interpret figurative language.

    This class models right-hemisphere language processing: it looks
    for cross-domain mappings (metaphors), sentiment-context
    contradictions (irony), and non-compositional expressions (idioms).

    The processor works without external NLP libraries — it uses
    domain heuristics and a curated idiom list. It's a simplified
    model of figurative comprehension, not a full NLP pipeline, but
    it captures the core mechanisms.
    """

    # Semantic domains — maps words to their domain.
    # This is a simplified ontology; a full system would use
    # WordNet domains or a distributed semantic representation.
    _DOMAIN_MAP: ClassVar[dict[str, frozenset[str]]] = {
        "temporal": frozenset(
            {
                "time",
                "hour",
                "minute",
                "second",
                "day",
                "night",
                "year",
                "past",
                "future",
                "present",
                "moment",
                "eternity",
                "history",
                "memory",
                "yesterday",
                "tomorrow",
                "season",
            }
        ),
        "fluid": frozenset(
            {
                "river",
                "ocean",
                "sea",
                "stream",
                "water",
                "wave",
                "flood",
                "tide",
                "current",
                "pool",
                "lake",
                "rain",
                "drop",
                "flow",
                "drift",
            }
        ),
        "fire": frozenset(
            {
                "fire",
                "flame",
                "burn",
                "blaze",
                "spark",
                "ember",
                "ash",
                "smoke",
                "heat",
                "inferno",
                "kindle",
                "ignite",
            }
        ),
        "light": frozenset(
            {
                "light",
                "sun",
                "star",
                "moon",
                "dawn",
                "dusk",
                "shine",
                "glow",
                "beam",
                "ray",
                "shadow",
                "dark",
                "bright",
                "dim",
                "illuminate",
            }
        ),
        "journey": frozenset(
            {
                "road",
                "path",
                "way",
                "journey",
                "travel",
                "walk",
                "trail",
                "destination",
                "crossroad",
                "map",
                "compass",
                "wander",
                "arrive",
                "depart",
            }
        ),
        "body": frozenset(
            {
                "heart",
                "mind",
                "soul",
                "hand",
                "eye",
                "head",
                "bone",
                "blood",
                "brain",
                "skin",
                "arm",
                "foot",
                "shoulder",
                "chest",
                "spine",
            }
        ),
        "plant": frozenset(
            {
                "tree",
                "flower",
                "root",
                "branch",
                "leaf",
                "seed",
                "blossom",
                "garden",
                "vine",
                "fruit",
                "thorn",
                "bloom",
            }
        ),
        "war": frozenset(
            {
                "battle",
                "war",
                "fight",
                "sword",
                "shield",
                "army",
                "soldier",
                "weapon",
                "attack",
                "defend",
                "conquer",
                "surrender",
                "fortress",
                "siege",
            }
        ),
        "weather": frozenset(
            {
                "storm",
                "cloud",
                "wind",
                "thunder",
                "lightning",
                "rain",
                "snow",
                "fog",
                "hurricane",
                "calm",
                "breeze",
            }
        ),
        "machine": frozenset(
            {
                "engine",
                "gear",
                "wheel",
                "machine",
                "clock",
                "lever",
                "spring",
                "mechanism",
                "cog",
                "pulley",
                "circuit",
            }
        ),
        "emotion": frozenset(
            {
                "love",
                "joy",
                "fear",
                "anger",
                "sadness",
                "hope",
                "despair",
                "passion",
                "grief",
                "happiness",
                "rage",
            }
        ),
        "construction": frozenset(
            {
                "wall",
                "foundation",
                "tower",
                "bridge",
                "building",
                "arch",
                "pillar",
                "roof",
                "frame",
                "structure",
            }
        ),
        "life": frozenset(
            {
                "life",
                "birth",
                "death",
                "living",
                "alive",
                "existence",
                "mortal",
            }
        ),
        "knowledge": frozenset(
            {
                "knowledge",
                "understanding",
                "wisdom",
                "learning",
                "know",
                "truth",
                "idea",
                "insight",
            }
        ),
        "argument": frozenset(
            {
                "argument",
                "debate",
                "discussion",
                "claim",
                "position",
                "reason",
                "logic",
                "case",
            }
        ),
    }

    # Reverse lookup: word → domain
    _WORD_TO_DOMAIN: ClassVar[dict[str, str]] = {}
    for _domain, _words in _DOMAIN_MAP.items():
        for _w in _words:
            _WORD_TO_DOMAIN[_w] = _domain
    del _domain, _words, _w

    # Common idioms and their meanings
    _IDIOMS: ClassVar[dict[str, str]] = {
        "break a leg": "good luck (theatrical origin)",
        "piece of cake": "something very easy",
        "hit the books": "to study hard",
        "spill the beans": "to reveal a secret",
        "under the weather": "feeling ill",
        "cost an arm and a leg": "very expensive",
        "once in a blue moon": "very rarely",
        "a dime a dozen": "very common, nothing special",
        "jump the gun": "to act prematurely",
        "bite the bullet": "to endure something unpleasant",
        "cut to the chase": "get to the point directly",
        "on the same page": "in agreement",
        "throw in the towel": "to give up",
        "burn the midnight oil": "to work late into the night",
        "let the cat out of the bag": "to reveal a secret",
        "barking up the wrong tree": "pursuing a wrong lead",
        "beat around the bush": "to avoid the main topic",
        "a blessing in disguise": "something bad that turns out good",
        "caught between two stools": "torn between two choices",
        "see eye to eye": "to agree completely",
        "the ball is in your court": "it's your decision now",
        "cold shoulder": "to deliberately ignore someone",
        "cross your fingers": "to hope for good luck",
        "fit as a fiddle": "in excellent health",
        "hang in there": "don't give up",
        "ignorance is bliss": "not knowing is better",
        "kill two birds with one stone": "accomplish two things at once",
        "let sleeping dogs lie": "don't disturb a stable situation",
        "no pain no gain": "effort is required for results",
        "out of the blue": "unexpectedly",
        "read between the lines": "find hidden meaning",
        "the whole nine yards": "everything, completely",
        "time flies": "time passes quickly",
        "tip of the iceberg": "a small visible part of a larger problem",
        "when pigs fly": "something that will never happen",
        "words fail me": "I'm so surprised I can't speak",
    }

    # Positive and negative sentiment words for irony detection
    _POSITIVE_WORDS = frozenset(
        {
            "great",
            "wonderful",
            "fantastic",
            "amazing",
            "perfect",
            "excellent",
            "brilliant",
            "lovely",
            "superb",
            "delightful",
            "good",
            "nice",
            "happy",
            "glad",
            "joy",
        }
    )
    _NEGATIVE_WORDS = frozenset(
        {
            "terrible",
            "awful",
            "horrible",
            "dreadful",
            "disaster",
            "nightmare",
            "catastrophe",
            "fiasco",
            "debacle",
            "tragedy",
            "bad",
            "wrong",
            "broken",
            "fail",
            "failure",
            "bug",
            "error",
            "crash",
            "problem",
            "issue",
        }
    )

    def detect_metaphor(self, text: str) -> Metaphor | None:
        """Detect a metaphor in the given text.

        Looks for "X is Y" patterns where X and Y belong to different
        semantic domains. If the domains differ, the expression is
        likely metaphorical.

        Args:
            text: The input text to analyze.

        Returns:
            A Metaphor object if a metaphor is detected, None otherwise.
        """
        if not text:
            return None

        # Search for "X is/are/was/were Y" patterns
        for match in _IS_A_RE.finditer(text):
            source_raw = match.group(1).strip().lower()
            target_raw = match.group(2).strip().lower()

            # Skip very short or very long phrases
            if len(source_raw) < 2 or len(target_raw) < 2:
                continue
            if len(source_raw) > 50 or len(target_raw) > 50:
                continue

            # Look up domains
            source_domain = self._lookup_domain(source_raw)
            target_domain = self._lookup_domain(target_raw)

            # If both domains are known and different → metaphor
            if (
                source_domain is not None
                and target_domain is not None
                and source_domain != target_domain
            ):
                shared = self._find_shared_properties(
                    source_raw, target_raw, source_domain, target_domain
                )
                return Metaphor(
                    source=source_raw,
                    target=target_raw,
                    source_domain=source_domain,
                    target_domain=target_domain,
                    sentence=text,
                    shared_properties=shared,
                )

        return None

    def interpret_metaphor(self, metaphor: Metaphor) -> str:
        """Interpret what a metaphor means.

        Maps the shared properties between the source and target
        domains to produce a plain-language explanation.

        Args:
            metaphor: The metaphor to interpret.

        Returns:
            A string explaining the metaphor's meaning.
        """
        source = metaphor.source
        target = metaphor.target
        shared = metaphor.shared_properties

        if shared:
            properties_text = ", ".join(shared)
            return (
                f"The metaphor '{source} is {target}' maps the "
                f"properties of {target} ({metaphor.target_domain}) "
                f"onto {source} ({metaphor.source_domain}). "
                f"The shared properties are: {properties_text}."
            )
        else:
            return (
                f"The metaphor '{source} is {target}' draws on the "
                f"{metaphor.target_domain} domain to describe "
                f"{source} in the {metaphor.source_domain} domain. "
                f"The mapping suggests that {source} shares qualities "
                f"with {target}."
            )

    def _surface_sentiment(self, text_lower: str) -> str:
        """Determine the surface sentiment of *text_lower*.

        We check for positive/negative sentiment words. When both
        are present, we look at which type appears first — sentiment
        expressions typically come before the topic in ironic
        utterances ("Oh great, another bug" → "great" is the
        sentiment, "bug" is the topic).
        """
        surface_positive = sum(1 for w in self._POSITIVE_WORDS if w in text_lower)
        surface_negative = sum(1 for w in self._NEGATIVE_WORDS if w in text_lower)

        if surface_positive > surface_negative:
            return "positive"
        if surface_negative > surface_positive:
            return "negative"
        if surface_positive > 0 and surface_negative > 0:
            # Tie — check which sentiment word appears first
            first_pos = min(
                (text_lower.find(w) for w in self._POSITIVE_WORDS if w in text_lower),
                default=len(text_lower),
            )
            first_neg = min(
                (text_lower.find(w) for w in self._NEGATIVE_WORDS if w in text_lower),
                default=len(text_lower),
            )
            if first_pos < first_neg:
                return "positive"
            if first_neg < first_pos:
                return "negative"
            return "neutral"
        return "neutral"

    def _expected_sentiment(
        self, context_lower: str, text_lower: str, surface_sentiment: str
    ) -> str:
        """Determine the expected sentiment given context (or text alone)."""
        if context_lower:
            ctx_positive = sum(1 for w in self._POSITIVE_WORDS if w in context_lower)
            ctx_negative = sum(1 for w in self._NEGATIVE_WORDS if w in context_lower)
            if ctx_negative > ctx_positive:
                return "negative"
            if ctx_positive > ctx_negative:
                return "positive"
            return "neutral"
        # Without context, check if the text itself has
        # contradictory signals (e.g., positive word + negative topic)
        has_negative_topic = any(w in text_lower for w in self._NEGATIVE_WORDS)
        if surface_sentiment == "positive" and has_negative_topic:
            return "negative"
        return surface_sentiment

    def _irony_explanation(
        self, is_ironic: bool, surface_sentiment: str, expected_sentiment: str
    ) -> tuple[float, str]:
        """Build the confidence and explanation for an irony result."""
        if is_ironic:
            confidence = 0.7
            explanation = (
                f"The text expresses {surface_sentiment} sentiment, "
                f"but the context suggests {expected_sentiment} — "
                f"this is likely ironic."
            )
        else:
            confidence = 0.8
            explanation = (
                f"No irony detected — surface sentiment ({surface_sentiment}) "
                f"aligns with expected sentiment ({expected_sentiment})."
            )
        return confidence, explanation

    def detect_irony(self, text: str, context: str = "") -> IronyDetection:
        """Detect irony in the given text.

        Irony is detected when the surface sentiment of the text
        contradicts the expected sentiment given the context.

        Args:
            text: The text to analyze.
            context: Optional surrounding context for comparison.

        Returns:
            An IronyDetection result.
        """
        if not text:
            return IronyDetection(
                is_ironic=False,
                surface_sentiment="neutral",
                expected_sentiment="neutral",
                confidence=0.0,
                explanation="No text to analyze.",
            )

        text_lower = text.lower()
        context_lower = context.lower() if context else ""

        surface_sentiment = self._surface_sentiment(text_lower)
        expected_sentiment = self._expected_sentiment(
            context_lower, text_lower, surface_sentiment
        )

        # Irony = surface contradicts expected
        is_ironic = (
            surface_sentiment != "neutral"
            and expected_sentiment != "neutral"
            and surface_sentiment != expected_sentiment
        )

        confidence, explanation = self._irony_explanation(
            is_ironic, surface_sentiment, expected_sentiment
        )

        return IronyDetection(
            is_ironic=is_ironic,
            surface_sentiment=surface_sentiment,
            expected_sentiment=expected_sentiment,
            confidence=confidence,
            explanation=explanation,
        )

    def detect_idiom(self, text: str) -> tuple[str, str] | None:
        """Detect an idiom in the given text.

        Args:
            text: The input text to check.

        Returns:
            A tuple of (idiom, meaning) if an idiom is found,
            None otherwise.
        """
        if not text:
            return None
        text_lower = text.lower()
        for idiom, meaning in self._IDIOMS.items():
            if idiom in text_lower:
                return (idiom, meaning)
        return None

    def get_idiom_meaning(self, idiom: str) -> str | None:
        """Get the meaning of a known idiom.

        Args:
            idiom: The idiom phrase to look up.

        Returns:
            The meaning of the idiom, or None if unknown.
        """
        return self._IDIOMS.get(idiom.lower())

    @property
    def idiom_count(self) -> int:
        """Number of idioms in the processor's knowledge base."""
        return len(self._IDIOMS)

    # ─── Internal helpers ─────────────────────────────────────────

    def _lookup_domain(self, phrase: str) -> str | None:
        """Look up the semantic domain for a word or phrase.

        Checks each word in the phrase against the domain map.
        Returns the first matching domain.
        """
        words = phrase.split()
        for word in words:
            # Strip trailing 's' for simple plural handling
            domain = self._WORD_TO_DOMAIN.get(word)
            if domain is None and word.endswith("s"):
                domain = self._WORD_TO_DOMAIN.get(word[:-1])
            if domain is not None:
                return domain
        return None

    def _find_shared_properties(
        self,
        source: str,
        target: str,
        source_domain: str,
        target_domain: str,
    ) -> list[str]:
        """Find shared properties between source and target domains.

        This is a simplified mapping of common cross-domain metaphors.
        A full system would use a property ontology.
        """
        # Common metaphorical mappings. Keys must use actual _DOMAIN_MAP
        # domain names, not the concept words themselves — _lookup_domain
        # resolves concept words to domains (e.g., "mind" → "body",
        # "memory" → "temporal", "love" → "emotion", "time" → "temporal").
        mappings: dict[tuple[str, str], list[str]] = {
            ("temporal", "fluid"): ["flow", "movement", "continuity", "current"],
            ("temporal", "journey"): ["progress", "direction", "milestones"],
            ("temporal", "construction"): ["foundation", "structure", "building"],
            ("temporal", "fire"): ["consuming", "fleeting", "irreversible"],
            ("emotion", "fire"): ["intensity", "warmth", "consuming", "spreading"],
            ("emotion", "weather"): ["intensity", "changeability", "atmosphere"],
            ("emotion", "fluid"): ["depth", "overflow", "currents", "waves"],
            ("emotion", "light"): ["brightness", "warmth", "illumination"],
            ("emotion", "plant"): ["growth", "nurturing", "blooming", "roots"],
            ("life", "journey"): ["path", "destination", "obstacles", "progress"],
            ("life", "plant"): ["growth", "seasons", "roots", "blooming"],
            ("body", "machine"): ["processing", "mechanism", "efficiency"],
            ("body", "light"): ["illumination", "clarity", "brightness"],
            ("knowledge", "light"): ["illumination", "clarity", "visibility"],
            ("knowledge", "journey"): ["path", "discovery", "exploration"],
            ("argument", "war"): ["conflict", "strategy", "attack", "defense"],
        }

        key = (source_domain, target_domain)
        return mappings.get(key, [])
