"""WordNet dictionary — Genesis's primary word reference.

It looks up words in WordNet (via NLTK) to find definitions,
hypernyms (IS_A), meronyms (PART_OF), similar-to, and antonyms.
This is its dictionary — it uses it to build its own concept
network from what it finds.

No web access required. WordNet is a local database.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WordNetEntry:
    """A single WordNet sense for a word."""
    word: str
    definition: str
    part_of_speech: str
    examples: list[str]
    hypernyms: list[str]      # IS_A — what this is a kind of
    meronyms: list[str]       # PART_OF — parts of this
    similar_to: list[str]     # SIMILAR_TO — related concepts
    antonyms: list[str]       # OPPOSITE_OF — opposites


def lookup_word(word: str, max_senses: int = 2) -> list[WordNetEntry]:
    """Look up a word in WordNet and return its senses.

    Returns up to max_senses entries, prioritizing nouns then verbs
    then adjectives. Each entry has the definition, part of speech,
    and semantic relations (hypernyms, meronyms, similar, antonyms).

    Returns empty list if the word is not in WordNet.
    """
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        logger.warning("NLTK WordNet not available")
        return []

    clean = word.lower().strip().replace(" ", "_")
    try:
        synsets = wn.synsets(clean)
    except LookupError:
        logger.warning("NLTK WordNet corpus not downloaded; returning no senses")
        return []
    if not synsets:
        return []

    # Prioritize: nouns first, then verbs, then others
    prioritized = _prioritize_synsets(synsets, max_senses)

    pos_map = {
        "n": "noun",
        "v": "verb",
        "a": "adjective",
        "s": "adjective",
        "r": "adverb",
    }

    entries: list[WordNetEntry] = []
    for synset in prioritized:
        entries.append(_build_wordnet_entry(synset, word, pos_map))

    return entries


def _prioritize_synsets(synsets, max_senses: int):
    """Prioritize synsets by part of speech: nouns, then verbs, then others.

    Returns up to ``max_senses`` synsets in priority order.
    """
    noun_synsets = [s for s in synsets if s.pos() == "n"]
    verb_synsets = [s for s in synsets if s.pos() == "v"]
    other_synsets = [s for s in synsets if s.pos() not in ("n", "v")]
    return (noun_synsets + verb_synsets + other_synsets)[:max_senses]


def _build_wordnet_entry(synset, word: str, pos_map: dict[str, str]) -> WordNetEntry:
    """Build a WordNetEntry from a synset, extracting all semantic relations.

    Extracts hypernyms (IS_A), meronyms (PART_OF), similar-to terms,
    and antonyms, filtering out the query word itself and very short terms.
    """
    # Hypernyms (IS_A)
    hypernyms = []
    for h in synset.hypernyms()[:2]:
        lemmas = h.lemma_names()
        if lemmas:
            h_word = lemmas[0].replace("_", " ").lower()
            if h_word != word.lower() and len(h_word) > 2:
                hypernyms.append(h_word)

    # Meronyms (PART_OF)
    meronyms = []
    for m in (synset.part_meronyms() + synset.substance_meronyms())[:2]:
        lemmas = m.lemma_names()
        if lemmas:
            m_word = lemmas[0].replace("_", " ").lower()
            if m_word != word.lower() and len(m_word) > 2:
                meronyms.append(m_word)

    # Similar to
    similar = []
    for s in synset.similar_tos()[:2]:
        lemmas = s.lemma_names()
        if lemmas:
            s_word = lemmas[0].replace("_", " ").lower()
            if s_word != word.lower() and len(s_word) > 2:
                similar.append(s_word)

    # Antonyms
    antonyms = []
    for lemma in synset.lemmas():
        for ant in lemma.antonyms():
            a_word = ant.name().replace("_", " ").lower()
            if a_word != word.lower() and len(a_word) > 2:
                antonyms.append(a_word)

    return WordNetEntry(
        word=word.lower(),
        definition=synset.definition(),
        part_of_speech=pos_map.get(synset.pos(), "concept"),
        examples=synset.examples()[:1],
        hypernyms=hypernyms,
        meronyms=meronyms,
        similar_to=similar,
        antonyms=antonyms,
    )


def lookup_definition(word: str) -> str | None:
    """Look up just the primary definition for a word.

    Returns the first noun sense definition, or the first sense
    if no noun sense exists. Returns None if not in WordNet.
    """
    entries = lookup_word(word, max_senses=1)
    if entries:
        return entries[0].definition
    return None
