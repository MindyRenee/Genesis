"""Topic resolution — extract and refine topic concepts from user input.

Extracted from CognitionEngine as a focused subsystem. The resolver
combines adjacent topic words into multi-word concepts, finds
plural/singular variants in the concept network, merges concepts
found directly in the raw text, and filters out function words.

Dependencies (passed to ``__init__``):
    - network: ConceptNetwork for concept lookup and neighbor queries
"""

from __future__ import annotations

import re

from genesis_cognitive.concepts import _FUNCTION_WORDS, ConceptNetwork

__all__ = ["TopicResolver"]


class TopicResolver:
    """Resolve raw topic words into concept-network-compatible topics.

    Takes a list of single-word topics (from the perception layer) and
    the raw input text, then:
    1. Combines adjacent words into multi-word concepts (e.g. "utah"
       + "law" → "utah law" if that concept exists).
    2. Finds plural/singular variants (e.g. "felony" ↔ "felonies").
    3. Merges multi-word concepts found directly in the raw text.
    4. Filters out English function words.
    """

    # Extra conversational words that aren't in the concept_network
    # stoplist but still aren't domain concepts.
    _EXTRA_STOP = frozenset(
        {"tell", "about", "actually", "no", "yes", "really", "very"}
    )

    def __init__(self, network: ConceptNetwork) -> None:
        """Wire the resolver to its concept network."""
        self._network = network

    def resolve_topics(self, topics: list[str], raw_text: str) -> list[str]:
        """Combine adjacent topic words into multi-word concepts.

        The perception module extracts single-word topics (e.g., "utah",
        "law"), but the concept network stores multi-word concepts (e.g.,
        "utah law"). This method tries to find multi-word combinations
        of the topics that exist in the concept network, and replaces
        the individual words with the combined concept.

        This also tries to find concepts from the raw text that may not
        have been extracted as topics (e.g., "stare decisis" where both
        words are too short/common to be topics).

        It also handles plural/singular mismatches (e.g., the question
        uses "felony" but the concept network has "felonies").

        Topics already resolved by the perception layer's context-aware
        resolution (code concepts with python:/rust: prefix) are passed
        through unchanged — they don't need further combination.
        """
        # If topics are already resolved to code/identity concepts
        # (have python: or rust: prefix), pass them through — they've
        # already been contextually resolved and shouldn't be recombined
        if topics and any(
            t.startswith(("python:", "rust:", "identity:")) for t in topics
        ):
            return topics

        lower = raw_text.lower()
        resolved = self._combine_adjacent_topics(topics, lower)
        resolved = self._merge_text_concepts(raw_text, resolved)
        return self._filter_function_words(resolved, topics)

    def find_concept_variant(self, word: str) -> str | None:
        """Find a concept by trying the word as-is, then pluralized/singularized.

        Prefers concepts that actually have relationships (neighbors).
        """
        net = self._network
        candidates = [word]
        # Try simple plural (add 's')
        if net.get_concept(word + "s"):
            candidates.append(word + "s")
        # Try plural (y → ies)
        if word.endswith("y") and net.get_concept(word[:-1] + "ies"):
            candidates.append(word[:-1] + "ies")
        # Try singular (strip trailing 's')
        if word.endswith("s") and net.get_concept(word[:-1]):
            candidates.append(word[:-1])
        # Try singular (ies → y)
        if word.endswith("ies") and net.get_concept(word[:-3] + "y"):
            candidates.append(word[:-3] + "y")
        # Try singular (es → s or strip s)
        if word.endswith("es"):
            if net.get_concept(word[:-1]):
                candidates.append(word[:-1])
            if net.get_concept(word[:-2]):
                candidates.append(word[:-2])
        # Try verb inflections (-ing)
        if word.endswith("ing"):
            if net.get_concept(word[:-3]):
                candidates.append(word[:-3])
            if net.get_concept(word[:-3] + "e"):
                candidates.append(word[:-3] + "e")
        # Try past tense (-ed)
        if word.endswith("ed"):
            if net.get_concept(word[:-2]):
                candidates.append(word[:-2])
            if net.get_concept(word[:-1]):
                candidates.append(word[:-1])

        # Prefer the candidate that has relationships
        for c in candidates:
            if net.has_neighbors(c):
                return c
        # Fall back to any that exists
        for c in candidates:
            if net.get_concept(c):
                return c
        return None

    def _combine_adjacent_topics(
        self, topics: list[str], lower: str
    ) -> list[str]:
        """Try pairs of adjacent topics to find multi-word concepts in the network."""
        resolved: list[str] = []
        used: set[int] = set()

        # Try pairs of adjacent topics (only if the combined phrase
        # actually appears in the text)
        for i in range(len(topics)):
            if i in used:
                continue
            combined = None
            for j in range(i + 1, min(i + 3, len(topics))):
                candidate = " ".join(topics[i : j + 1])
                if candidate not in lower:
                    continue
                found = self.find_concept_variant(candidate)
                if found:
                    combined = found
                    used.update(range(i, j + 1))
                    break
            if combined:
                resolved.append(combined)
            elif i not in used:
                # Try singular/plural for single words
                found = self.find_concept_variant(topics[i])
                resolved.append(found if found else topics[i])
        return resolved

    def _merge_text_concepts(
        self, raw_text: str, resolved: list[str]
    ) -> list[str]:
        """Find multi-word concepts from raw text and merge with resolved topics."""
        words = re.findall(r"[^\W_]+", raw_text.casefold())
        text_concepts: list[str] = []
        max_width = min(5, len(words))
        for width in range(max_width, 1, -1):
            for start in range(len(words) - width + 1):
                candidate = " ".join(words[start:start + width])
                found = self.find_concept_variant(candidate)
                if (
                    found
                    and found not in resolved
                    and found not in text_concepts
                ):
                    text_concepts.append(found)

        # Prioritize multi-word concepts from the text (they're more
        # specific) over single-word topics that might be too generic
        if text_concepts:
            # Sort by length (longer = more specific) then deduplicate
            text_concepts.sort(key=len, reverse=True)
            # Remove any single-word topics that are substrings of the
            # multi-word concepts we found
            covered = set()
            for tc in text_concepts:
                for w in tc.split():
                    covered.add(w)
            resolved = text_concepts + [
                t for t in resolved if t not in covered
            ]
        return resolved

    def _filter_function_words(
        self, resolved: list[str], topics: list[str]
    ) -> list[str]:
        """Filter out English function words from resolved topics."""
        # Filter out English function words — these are grammatical
        # structure (question words, articles, auxiliary verbs), not
        # domain concepts. This is English language processing, not
        # domain-specific knowledge. Domain-specific filtering (e.g.,
        # "relationship" in a question about ice and water) is handled
        # by graph connectivity in the thought composer's
        # _find_best_connected_pair method.
        stop = _FUNCTION_WORDS | self._EXTRA_STOP

        domain_topics = [t for t in resolved if t.lower() not in stop]
        if domain_topics:
            resolved = domain_topics

        return resolved if resolved else topics
