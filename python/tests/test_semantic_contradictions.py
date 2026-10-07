from genesis_conscious.memory.semantic import SemanticMemory


def test_explicit_contradiction_is_stored_as_semantic_fact() -> None:
    memory = SemanticMemory()

    facts = memory.extract_facts("Evidence contradicts theory.")

    assert any(
        f.key == ("evidence", "contradicts", "theory")
        and f.confidence == 0.7
        for f in facts
    )


def test_explicit_opposition_is_stored_as_semantic_fact() -> None:
    memory = SemanticMemory()

    facts = memory.extract_facts("Free will is the opposite of determinism.")

    assert any(
        f.key == ("free will", "opposite_of", "determinism")
        for f in facts
    )
