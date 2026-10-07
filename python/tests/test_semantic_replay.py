from genesis_conscious.memory.semantic import Fact, SemanticMemory


def test_sleep_replay_does_not_inflate_existing_fact_confidence() -> None:
    memory = SemanticMemory()
    memory.store_fact(
        Fact(
            subject="cats",
            relation="is_a",
            object="mammals",
            confidence=0.5,
        )
    )

    memory.extract_facts(
        "Cats are mammals.",
        reinforce_existing=False,
    )

    fact = memory._facts.get(("cats", "is_a", "mammals"))
    assert fact is not None
    assert fact.source_count == 1
    assert fact.confidence == 0.5


def test_normal_observation_still_reinforces_existing_fact() -> None:
    memory = SemanticMemory()
    memory.store_fact(
        Fact(
            subject="cats",
            relation="is_a",
            object="mammals",
            confidence=0.5,
        )
    )

    memory.extract_facts("Cats are mammals.")

    fact = memory._facts.get(("cats", "is_a", "mammals"))
    assert fact is not None
    assert fact.source_count == 2
    assert fact.confidence == 0.65
