"""Statistical language learning from user input.

This module implements statistical learning of language patterns from
the user's input. Genesis tracks word frequencies, n-gram patterns
(bigrams and trigrams), and builds a simple probabilistic grammar from
observed patterns. It also adapts to the user's communication style
(formal, casual, technical).

The generative engine can use the learned statistics to match the
user's style — producing language that mirrors their vocabulary,
sentence length, and register.

# Background

Statistical learning is a foundational mechanism in human language
acquisition. Children are sensitive to the statistical structure of
their input — transitional probabilities between syllables (Saffran,
Aslin & Newport, 1996), word frequencies, and distributional patterns.
This module applies the same principle: Genesis learns the statistical
regularities of the user's language and uses them to adapt its own.

References:
- Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical
  learning by 8-month-old infants. *Science*, 274(5294), 1926–1928.
- Jurafsky, D., & Martin, J. H. (2024). *Speech and Language
  Processing* (3rd ed.). Chapter 3: N-gram Language Models.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

__all__ = ["NGramModel", "StatisticalLanguageLearner"]

# ─── Pre-compiled regex patterns ──────────────────────────────────
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass(slots=True)
class NGramModel:
    """A simple n-gram model with frequency counts.

    Attributes:
        n: The n in n-gram (2 for bigrams, 3 for trigrams).
        counts: Mapping from n-gram tuple to observed count.
        context_counts: Mapping from the (n-1)-gram prefix to
            the total count of all n-grams sharing that prefix.
            Used for computing conditional probabilities.
        prefix_index: Mapping from prefix tuple to
            {completion_word: count} for every n-gram with that
            prefix. Maintained alongside ``counts`` so prefix-based
            queries (e.g. next-word prediction) are O(K) in the
            number of completions for the prefix rather than O(N)
            in the total number of observed n-grams.
    """

    n: int
    counts: dict[tuple[str, ...], int] = field(default_factory=dict)
    context_counts: dict[tuple[str, ...], int] = field(default_factory=dict)
    prefix_index: dict[tuple[str, ...], dict[str, int]] = field(default_factory=dict)

    def observe(self, tokens: list[str]) -> None:
        """Record n-grams from a token list."""
        if len(tokens) < self.n:
            return
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i : i + self.n])
            prefix = ngram[:-1]
            completion = ngram[-1]
            self.counts[ngram] = self.counts.get(ngram, 0) + 1
            self.context_counts[prefix] = self.context_counts.get(prefix, 0) + 1
            bucket = self.prefix_index.setdefault(prefix, {})
            bucket[completion] = bucket.get(completion, 0) + 1

    def completions_for(self, prefix: tuple[str, ...]) -> dict[str, int]:
        """Return ``{completion_word: count}`` for all n-grams with ``prefix``.

        Lookup is O(1) average; iterating the result is O(K) where K
        is the number of distinct completions observed for the prefix.
        """
        return self.prefix_index.get(prefix, {})

    def probability(self, ngram: tuple[str, ...]) -> float:
        """Conditional probability P(word_n | prefix).

        Uses simple maximum-likelihood estimation with add-one
        (Laplace) smoothing to avoid zero probabilities for unseen
        n-grams.
        """
        prefix = ngram[:-1]
        context_count = self.context_counts.get(prefix, 0)
        ngram_count = self.counts.get(ngram, 0)
        # Laplace smoothing — vocabulary size is approximated by
        # the number of distinct n-grams observed
        vocab = len(self.counts) if self.counts else 1
        return (ngram_count + 1.0) / (context_count + vocab)

    def most_common(self, k: int = 10) -> list[tuple[tuple[str, ...], int]]:
        """Return the k most frequent n-grams."""
        return Counter(self.counts).most_common(k)


class StatisticalLanguageLearner:
    """Learn the user's language patterns from input text.

    Tracks word frequencies, bigram/trigram patterns, and adapts to
    the user's communication style. The generative engine can query
    this learner to match the user's vocabulary and register.

    Usage::

        learner = StatisticalLanguageLearner()
        learner.learn_from_input("Hello, could you help me with this?")
        learner.learn_from_input("Sure, I can assist with that.")
        profile = learner.user_style_profile()
        # profile = {"style": "formal", "avg_sentence_length": 7.5, ...}
    """

    # Words that signal a formal register
    _FORMAL_MARKERS = frozenset(
        {
            "therefore",
            "however",
            "furthermore",
            "nevertheless",
            "consequently",
            "accordingly",
            "hence",
            "thus",
            "would",
            "could",
            "shall",
            "may",
            "might",
            "regarding",
            "concerning",
            "pursuant",
            "hereby",
            "request",
            "inquire",
            "assist",
            "provide",
        }
    )

    # Words that signal a casual register
    _CASUAL_MARKERS = frozenset(
        {
            "yeah",
            "yep",
            "nope",
            "ok",
            "okay",
            "cool",
            "gonna",
            "wanna",
            "gotta",
            "kinda",
            "sorta",
            "hey",
            "hi",
            "bye",
            "lol",
            "haha",
            "stuff",
            "thing",
            "things",
            "guy",
            "guys",
        }
    )

    # Words that signal a technical register
    _TECHNICAL_MARKERS = frozenset(
        {
            "function",
            "variable",
            "parameter",
            "algorithm",
            "implementation",
            "interface",
            "module",
            "class",
            "method",
            "compile",
            "runtime",
            "debug",
            "syntax",
            "semantic",
            "asynchronous",
            "concurrent",
            "optimization",
            "refactor",
            "architecture",
            "protocol",
        }
    )

    def __init__(self) -> None:
        """Initialize empty word frequency and n-gram models."""
        # Word frequency counts
        self._word_freq: Counter[str] = Counter()

        # N-gram models
        self._bigrams: NGramModel = NGramModel(n=2)
        self._trigrams: NGramModel = NGramModel(n=3)

        # Sentence length tracking (in words)
        self._sentence_lengths: list[int] = []

        # Total tokens observed
        self._total_tokens: int = 0

        # Total sentences observed
        self._total_sentences: int = 0

        # Style marker counts
        self._formal_count: int = 0
        self._casual_count: int = 0
        self._technical_count: int = 0

    def learn_from_input(self, text: str) -> None:
        """Learn language patterns from a user's input text.

        Updates word frequencies, n-gram models, sentence length
        statistics, and style marker counts.

        Args:
            text: The user's input text to learn from.
        """
        if not text or not text.strip():
            return

        # Split into sentences
        sentences = self._split_sentences(text)
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            if not tokens:
                continue

            # Update word frequencies
            self._word_freq.update(tokens)
            self._total_tokens += len(tokens)
            self._total_sentences += 1
            self._sentence_lengths.append(len(tokens))

            # Update n-gram models
            self._bigrams.observe(tokens)
            self._trigrams.observe(tokens)

            # Update style markers
            lower_tokens = {t.lower() for t in tokens}
            self._formal_count += len(lower_tokens & self._FORMAL_MARKERS)
            self._casual_count += len(lower_tokens & self._CASUAL_MARKERS)
            self._technical_count += len(lower_tokens & self._TECHNICAL_MARKERS)

    def user_style_profile(self) -> dict:
        """Return a profile of the user's communication style.

        The profile includes:
        - ``style``: the dominant register ("formal", "casual",
          "technical", or "neutral")
        - ``avg_sentence_length``: average words per sentence
        - ``vocabulary_size``: number of distinct words observed
        - ``top_words``: the 10 most frequent words
        - ``top_bigrams``: the 5 most frequent bigrams
        - ``top_trigrams``: the 5 most frequent trigrams
        - ``formal_score``: proportion of formal markers (0–1)
        - ``casual_score``: proportion of casual markers (0–1)
        - ``technical_score``: proportion of technical markers (0–1)

        Returns:
            A dictionary describing the user's style profile.
        """
        avg_len = (
            sum(self._sentence_lengths) / len(self._sentence_lengths)
            if self._sentence_lengths
            else 0.0
        )

        # Determine dominant style
        total_markers = self._formal_count + self._casual_count + self._technical_count
        if total_markers == 0:
            style = "neutral"
            formal_score = 0.0
            casual_score = 0.0
            technical_score = 0.0
        else:
            formal_score = self._formal_count / total_markers
            casual_score = self._casual_count / total_markers
            technical_score = self._technical_count / total_markers
            scores = {
                "formal": formal_score,
                "casual": casual_score,
                "technical": technical_score,
            }
            style = max(scores, key=lambda k: scores[k])
            # If no clear winner (>40%), call it neutral
            if scores[style] < 0.4:
                style = "neutral"

        return {
            "style": style,
            "avg_sentence_length": round(avg_len, 2),
            "vocabulary_size": len(self._word_freq),
            "total_tokens": self._total_tokens,
            "total_sentences": self._total_sentences,
            "top_words": self._word_freq.most_common(10),
            "top_bigrams": [(list(ng), count) for ng, count in self._bigrams.most_common(5)],
            "top_trigrams": [(list(ng), count) for ng, count in self._trigrams.most_common(5)],
            "formal_score": round(formal_score, 3),
            "casual_score": round(casual_score, 3),
            "technical_score": round(technical_score, 3),
        }

    def word_frequency(self, word: str) -> int:
        """Get the observed frequency of a word."""
        return self._word_freq.get(word.lower(), 0)

    def bigram_probability(self, word_a: str, word_b: str) -> float:
        """Get P(word_b | word_a) from the bigram model."""
        return self._bigrams.probability((word_a.lower(), word_b.lower()))

    def trigram_probability(self, w1: str, w2: str, w3: str) -> float:
        """Get P(w3 | w1, w2) from the trigram model."""
        return self._trigrams.probability((w1.lower(), w2.lower(), w3.lower()))

    def suggest_next_word(self, context: list[str]) -> str | None:
        """Suggest the most likely next word given context.

        Uses the trigram model if the context is long enough,
        otherwise falls back to the bigram model.

        Lookup is O(K) in the number of completions observed for the
        prefix, via the prefix index — it does not scan the full
        n-gram table. Ranking by raw count is equivalent to ranking
        by Laplace-smoothed probability for a fixed prefix, since
        the smoothing denominator (context_count + V) is constant
        across completions sharing that prefix.

        Args:
            context: The preceding words (most recent last).

        Returns:
            The most likely next word, or None if no data.
        """
        context_lower = [w.lower() for w in context]
        if len(context_lower) >= 2:
            bucket = self._trigrams.completions_for(tuple(context_lower[-2:]))
            if bucket:
                return max(bucket, key=lambda w: bucket[w])
        if len(context_lower) >= 1:
            bucket = self._bigrams.completions_for(tuple(context_lower[-1:]))
            if bucket:
                return max(bucket, key=lambda w: bucket[w])
        return None

    def sequence_fluency(self, text: str) -> float:
        """Score local transitions against learned language experience.

        The score remains neutral for contexts the learner has never seen,
        so sparse experience cannot reject novel wording. Evidence only
        moves the score where an observed prefix supports or disfavors the
        candidate's continuation.
        """
        tokens = self._tokenize(text)
        if len(tokens) < 2 or not self._bigrams.context_counts:
            return 0.5

        supported: list[float] = []
        for first, second in pairwise(tokens):
            prefix = (first,)
            context_count = self._bigrams.context_counts.get(prefix, 0)
            if context_count:
                supported.append(
                    self._bigrams.counts.get((first, second), 0) / context_count
                )

        if not supported:
            return 0.5
        evidence = len(supported) / (len(tokens) - 1)
        observed_score = sum(supported) / len(supported)
        return 0.5 + evidence * (observed_score - 0.5)

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct words observed."""
        return len(self._word_freq)

    @property
    def total_inputs(self) -> int:
        """Total number of sentences observed."""
        return self._total_sentences

    @property
    def total_tokens(self) -> int:
        """Total number of word tokens observed across all input."""
        return self._total_tokens

    # ─── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences on sentence-ending punctuation."""
        # Simple sentence splitter — good enough for learning
        parts = re.split(r"[.!?]+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into lowercase words, stripping punctuation."""
        # Remove punctuation and split on whitespace
        cleaned = _PUNCT_RE.sub(" ", text)
        cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
        if not cleaned:
            return []
        return [w.lower() for w in cleaned.split() if len(w) > 0]
