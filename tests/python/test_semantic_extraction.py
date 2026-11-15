r"""Regression tests for article-boundary handling in fact extraction.

The optional-article groups in ``_IS_A_PLURAL_RE``/``_HAS_RE`` once read
``(?:a|an)?\s*`` — the ``a`` alternative matched the leading character of
words like "also", "among", "attached", and the ``an`` in "an excellent",
producing corrupted objects ("lso created", "mong the most", "n excellent
somatosensory system") that were stored as concept-network edges. The
article must be a complete word: ``(?:(?:a|an|the)\s+)?``.
"""

from memory.semantic import SemanticMemory


def _objects(memory: SemanticMemory, text: str, relation: str) -> list[str]:
    return [f.object for f in memory.extract_facts(text) if f.relation == relation]


def test_plural_is_a_does_not_eat_leading_a() -> None:
    memory = SemanticMemory()
    objects = _objects(memory, "Lists are also created purely for entertainment.", "is_a")
    assert objects == ["also created purely for entertainment"]
    assert all(not o.startswith("lso") for o in objects)


def test_plural_is_a_preserves_among() -> None:
    memory = SemanticMemory()
    objects = _objects(memory, "Ferrets are among the most intelligent pets.", "is_a")
    assert objects == ["among the most intelligent pets"]


def test_has_does_not_eat_an_of_an() -> None:
    memory = SemanticMemory()
    objects = _objects(
        memory,
        "The ferret has an excellent somatosensory system.",
        "has_property",
    )
    assert objects == ["excellent somatosensory system"]


def test_legitimate_articles_still_strip() -> None:
    memory = SemanticMemory()
    facts = {(f.subject, f.relation, f.object) for f in memory.extract_facts("Cats are mammals.")}
    assert ("cats", "is_a", "mammals") in facts

    memory = SemanticMemory()
    facts = {(f.subject, f.relation, f.object) for f in memory.extract_facts("A cat has a tail.")}
    assert ("cat", "has_property", "tail") in facts

    memory = SemanticMemory()
    facts = {
        (f.subject, f.relation, f.object) for f in memory.extract_facts("The brain has the cortex.")
    }
    assert ("brain", "has_property", "cortex") in facts
