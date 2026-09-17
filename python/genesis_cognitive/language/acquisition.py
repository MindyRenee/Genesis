"""Language acquisition mechanism — how Genesis learns language from input.

This module implements a computational model of language acquisition
based on statistical learning. It follows the Saffran model of
statistical learning (Saffran, Aslin & Newport, 1996), which showed
that infants can segment words from continuous speech by tracking
transitional probabilities between syllables.

The mechanism has three components:

1. **Statistical learning** (Saffran model): Track transitional
   probabilities between syllables/words. Word boundaries occur where
   transitional probabilities drop — within-word transitions have high
   probabilities, between-word transitions have low probabilities.

2. **Chunking**: Identify common multi-word units that appear
   frequently together. "New York" or "good morning" are chunks that
   should be treated as single units rather than two separate words.

3. **Word segmentation**: Identify word boundaries from input patterns
   using the transitional probability drops detected by the
   statistical learning component.

# Background

Infants learn language not through explicit instruction but through
statistical regularities in the input. The Saffran et al. (1996)
experiment demonstrated that 8-month-old infants can segment
continuous speech into words after just 2 minutes of exposure, using
transitional probabilities between syllables. This module applies the
same principle: Genesis segments and chunks language from the user's
input, building up a repertoire of learned units.

References:
- Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical
  learning by 8-month-old infants. *Science*, 274(5294), 1926–1928.
- Saffran, J. R. (2003). Statistical language learning: Mechanisms
  and constraints. *Current Directions in Psychological Science*,
  12(4), 110–114.
- Bod, R. (2009). From exemplar to grammar: A probabilistic
  analogy-based model of language learning. *Cognitive Science*,
  33(5), 752–793.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

__all__ = ["LanguageAcquisition", "LearnedUnit"]


@dataclass(slots=True)
class LearnedUnit:
    """A unit of language learned from input.

    A learned unit can be a word, a chunk (multi-word unit), or a
    segmented word boundary pattern.

    Attributes:
        text: The text of the learned unit (e.g. "hello" or
            "good morning").
        kind: The type of unit — "word", "chunk", or "boundary".
        frequency: How many times this unit has been observed.
        confidence: Learning confidence (0–1), based on frequency
            and statistical strength.
        transitional_prob: For words, the average transitional
            probability within the unit (high for within-word
            transitions, low for between-word transitions).
    """

    text: str
    kind: str  # "word", "chunk", or "boundary"
    frequency: int
    confidence: float
    transitional_prob: float = 0.0


class LanguageAcquisition:
    """Learn language from input through statistical learning.

    Implements the Saffran model of statistical learning: tracks
    transitional probabilities between syllables/words, identifies
    word boundaries where probabilities drop, and chunks common
    multi-word units.

    Usage::

        acquirer = LanguageAcquisition()
        units = acquirer.acquire_from_input("good morning everyone how are you")
        # units might include: word("good"), chunk("good morning"),
        # word("everyone"), word("how"), word("are"), word("you")

    Attributes:
        syllable_transitions: Transitional probabilities between
            syllables — P(syllable_b | syllable_a).
        word_frequencies: Frequency counts for observed words.
        chunk_frequencies: Frequency counts for multi-word chunks.
        learned_words: Set of words identified through segmentation.
        learned_chunks: Set of chunks identified through chunking.
    """

    # Minimum frequency for a chunk to be considered learned
    _CHUNK_MIN_FREQ = 2

    # Minimum chunk length (in words)
    _CHUNK_MIN_WORDS = 2

    # Maximum chunk length (in words)
    _CHUNK_MAX_WORDS = 4

    # Transitional probability threshold for word boundaries.
    # Below this, a transition is considered a word boundary.
    _BOUNDARY_THRESHOLD = 0.3

    # Minimum frequency for a word to be considered learned
    _WORD_MIN_FREQ = 1

    def __init__(self) -> None:
        """Initialize empty transition counts and word/chunk frequency tables."""
        # Transitional probabilities: P(b | a) = count(a,b) / count(a)
        self._transition_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._syllable_counts: Counter[str] = Counter()

        # Word and chunk frequencies
        self._word_freq: Counter[str] = Counter()
        self._chunk_freq: Counter[tuple[str, ...]] = Counter()

        # Learned units
        self._learned_words: set[str] = set()
        self._learned_chunks: set[str] = set()

        # Total inputs processed
        self._total_inputs: int = 0

    def acquire_from_input(self, text: str) -> list[LearnedUnit]:
        """Learn language units from input text.

        Processes the input through three stages:
        1. Statistical learning: updates transitional probabilities
        2. Word segmentation: identifies word boundaries
        3. Chunking: identifies common multi-word units

        Args:
            text: The input text to learn from.

        Returns:
            A list of LearnedUnit objects discovered or reinforced
            in this input.
        """
        if not text or not text.strip():
            return []

        self._total_inputs += 1

        # Tokenize into syllables (simplified: use words as syllables
        # for the transitional probability model)
        syllables = self._tokenize_to_syllables(text)
        if not syllables:
            return []

        # Stage 1: Word segmentation using transitional probabilities.
        # Segment BEFORE updating transitions so the current input is
        # segmented using statistics from PREVIOUS inputs. On the
        # first input, there are no transition counts, so every TP
        # is 0.0 (below threshold) → each syllable is its own word,
        # matching the documented first-input behavior. Previously,
        # _update_transitions ran first, making every TP = 1.0 on
        # the first input (each pair seen once: 1/1), which merged
        # the entire input into one giant "word."
        segmented_words = self._segment_words(syllables)

        # Stage 2: Update transitional probabilities with this input
        # for future segmentation.
        self._update_transitions(syllables)
        for word in segmented_words:
            self._word_freq[word] += 1
            if self._word_freq[word] >= self._WORD_MIN_FREQ:
                self._learned_words.add(word)

        # Stage 3: Chunking — find common multi-word units
        new_chunks = self._find_chunks(segmented_words)
        for chunk in new_chunks:
            self._learned_chunks.add(chunk)

        # Build the list of learned units from this input
        units: list[LearnedUnit] = []

        # Add segmented words
        for word in segmented_words:
            freq = self._word_freq.get(word, 0)
            tp = self._avg_transitional_prob(word)
            confidence = min(1.0, freq / 10.0)  # saturates at 10 observations
            units.append(
                LearnedUnit(
                    text=word,
                    kind="word",
                    frequency=freq,
                    confidence=confidence,
                    transitional_prob=tp,
                )
            )

        # Add chunks
        for chunk in new_chunks:
            chunk_tuple = tuple(chunk.split())
            freq = self._chunk_freq.get(chunk_tuple, 0)
            confidence = min(1.0, freq / 5.0)
            units.append(
                LearnedUnit(
                    text=chunk,
                    kind="chunk",
                    frequency=freq,
                    confidence=confidence,
                )
            )

        return units

    def get_transitional_probability(self, syllable_a: str, syllable_b: str) -> float:
        """Get the transitional probability P(b | a).

        This is the core of the Saffran model: within-word transitions
        have high probabilities, between-word transitions have low
        probabilities. Word boundaries occur where TP drops.

        Args:
            syllable_a: The preceding syllable/word.
            syllable_b: The following syllable/word.

        Returns:
            The transitional probability P(b | a), in [0, 1].
        """
        count_a = self._syllable_counts.get(syllable_a, 0)
        if count_a == 0:
            return 0.0
        count_ab = self._transition_counts.get(syllable_a, {}).get(syllable_b, 0)
        return count_ab / count_a

    @property
    def learned_words(self) -> set[str]:
        """The set of words learned through segmentation."""
        return set(self._learned_words)

    @property
    def learned_chunks(self) -> set[str]:
        """The set of multi-word chunks learned through chunking."""
        return set(self._learned_chunks)

    @property
    def vocabulary_size(self) -> int:
        """Total number of distinct words observed."""
        return len(self._word_freq)

    @property
    def total_inputs(self) -> int:
        """Total number of inputs processed."""
        return self._total_inputs

    # ─── Internal methods ─────────────────────────────────────────

    def _update_transitions(self, syllables: list[str]) -> None:
        """Update transitional probability counts from a syllable sequence."""
        for syl in syllables:
            self._syllable_counts[syl] += 1
        for i in range(len(syllables) - 1):
            a = syllables[i]
            b = syllables[i + 1]
            self._transition_counts[a][b] += 1

    def _segment_words(self, syllables: list[str]) -> list[str]:
        """Segment a syllable sequence into words using transitional
        probability drops.

        Following the Saffran model: a word boundary is placed between
        syllables a and b when P(b | a) is below the boundary threshold.
        Within-word transitions have high TP; between-word transitions
        have low TP.

        If no transition data is available yet (first input), each
        syllable is treated as a separate word.
        """
        if not syllables:
            return []

        if len(syllables) == 1:
            return syllables

        words: list[str] = []
        current_word: list[str] = [syllables[0]]

        for i in range(1, len(syllables)):
            prev = syllables[i - 1]
            curr = syllables[i]
            tp = self.get_transitional_probability(prev, curr)

            if tp < self._BOUNDARY_THRESHOLD:
                # Low TP → word boundary
                words.append(" ".join(current_word))
                current_word = [curr]
            else:
                # High TP → same word
                current_word.append(curr)

        # Don't forget the last word
        words.append(" ".join(current_word))

        return words

    def _find_chunks(self, words: list[str]) -> list[str]:
        """Find common multi-word chunks in a word sequence.

        Looks for sequences of 2–4 words that appear frequently across
        inputs. Chunks are stored as space-joined strings.
        """
        new_chunks: list[str] = []

        for n in range(self._CHUNK_MIN_WORDS, self._CHUNK_MAX_WORDS + 1):
            for i in range(len(words) - n + 1):
                chunk = tuple(words[i : i + n])
                self._chunk_freq[chunk] += 1
                freq = self._chunk_freq[chunk]
                if freq >= self._CHUNK_MIN_FREQ:
                    chunk_str = " ".join(chunk)
                    if chunk_str not in self._learned_chunks:
                        new_chunks.append(chunk_str)

        return new_chunks

    def _avg_transitional_prob(self, word: str) -> float:
        """Compute the average transitional probability within a word.

        For multi-syllable words, this is the average TP between
        consecutive syllables. For single-syllable words, returns 1.0
        (no internal transitions to evaluate).
        """
        syllables = word.split()
        if len(syllables) <= 1:
            return 1.0

        tps: list[float] = []
        for i in range(len(syllables) - 1):
            tp = self.get_transitional_probability(syllables[i], syllables[i + 1])
            tps.append(tp)

        return sum(tps) / len(tps) if tps else 1.0

    @staticmethod
    def _tokenize_to_syllables(text: str) -> list[str]:
        """Tokenize text into syllable-like units.

        For this simplified model, we use individual words as syllables.
        A more sophisticated implementation would split words into
        phonological syllables, but word-level segmentation captures
        the same statistical learning principle.
        """
        # Remove punctuation and split
        cleaned = re.sub(r"[^\w\s]", " ", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if not cleaned:
            return []
        return [w.lower() for w in cleaned.split() if len(w) > 0]
