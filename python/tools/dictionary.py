"""Dictionary — WordNet through NLTK, her own lexicon.

Definitions, synonyms, antonyms, hypernyms, hyponyms, example
sentences — the same corpus her language machinery already ships.
Data lookups only: what she *does* with a definition is her own.

Docs: https://www.nltk.org/howto/wordnet.html ·
https://wordnet.princeton.edu/
"""

from __future__ import annotations

from typing import Any

from tools.framework import ToolResult

from infrastructure.text import is_blank


def _wordnet() -> Any | None:
    try:
        from nltk.corpus import wordnet

        return wordnet
    except Exception:  # noqa: BLE001 — corpus missing or nltk absent
        return None


def define(word: str, limit: int = 5) -> ToolResult:
    """Look a word up in WordNet — senses, relations, examples.

    Per synset: part of speech, gloss (definition), lemmas (synonym
    set), antonyms, hypernyms, an example sentence when the corpus
    carries one. https://www.nltk.org/howto/wordnet.html
    """
    wn = _wordnet()
    if wn is None:
        return ToolResult(success=False, error="wordnet corpus unavailable (nltk data)")
    if is_blank(word):
        return ToolResult(success=False, error="empty word")
    synsets = wn.synsets(word.strip().lower().replace(" ", "_"))
    if not synsets:
        return ToolResult(success=False, error=f"no WordNet entry for {word!r}")
    senses = []
    for ss in synsets[: max(1, min(limit, 15))]:
        senses.append(
            {
                "name": ss.name(),
                "pos": ss.pos(),
                "definition": ss.definition(),
                "synonyms": sorted({lemma.name().replace("_", " ") for lemma in ss.lemmas()}),
                "antonyms": sorted(
                    {a.name().replace("_", " ") for lemma in ss.lemmas() for a in lemma.antonyms()}
                ),
                "hypernyms": sorted({h.name().split(".")[0] for h in ss.hypernyms()}),
                "examples": ss.examples(),
            }
        )
    output = "\n".join(
        f"{s['name']} ({s['pos']}): {s['definition']}"
        + (f" — e.g. {s['examples'][0]}" if s["examples"] else "")
        for s in senses
    )
    return ToolResult(success=True, output=output, data={"senses": senses})


def thesaurus(word: str) -> ToolResult:
    """Synonym and antonym sets for a word, grouped by part of speech."""
    wn = _wordnet()
    if wn is None:
        return ToolResult(success=False, error="wordnet corpus unavailable")
    if is_blank(word):
        return ToolResult(success=False, error="empty word")
    syn: set[str] = set()
    ant: set[str] = set()
    for ss in wn.synsets(word.strip().lower().replace(" ", "_")):
        for lemma in ss.lemmas():
            name = lemma.name().replace("_", " ")
            if name.lower() != word.lower():
                syn.add(name)
            ant.update(a.name().replace("_", " ") for a in lemma.antonyms())
    if not syn and not ant:
        return ToolResult(success=False, error=f"no WordNet entry for {word!r}")
    return ToolResult(
        success=True,
        output=(
            f"synonyms: {', '.join(sorted(syn)) or '—'}\nantonyms: {', '.join(sorted(ant)) or '—'}"
        ),
        data={"synonyms": sorted(syn), "antonyms": sorted(ant)},
    )


DICTIONARY_TOOLS = (
    (
        "define",
        "WordNet lookup: senses with part of speech, gloss, synonyms, "
        "antonyms, hypernyms, corpus example sentences. limit= senses "
        "(default 5). The Princeton lexicon, on her desk.",
        define,
    ),
    (
        "thesaurus",
        "Synonym and antonym sets for a word across every sense, "
        "deduplicated and sorted — the thesaurus half of WordNet.",
        thesaurus,
    ),
)
