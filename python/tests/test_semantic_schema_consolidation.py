"""Regression tests for semantic schema consolidation."""
from genesis_conscious.memory.semantic import Fact, SemanticMemory


def test_schema_counts_observation_once_not_once_per_fact():
    memory = SemanticMemory()
    facts = [
        Fact("dog", "is_a", "mammal"),
        Fact("dog", "has_property", "fur"),
        Fact("dog", "relates_to", "pet"),
    ]
    memory.form_schemas(facts)
    schema = memory.get_schema("dog")
    assert schema is not None
    assert schema.instances == 1
    assert schema.abstraction_level < 0.5

def test_schema_instances_accumulate_across_observations():
    memory = SemanticMemory()
    memory.form_schemas([Fact("dog", "is_a", "mammal")])
    memory.form_schemas([Fact("dog", "has_property", "fur")])
    schema = memory.get_schema("dog")
    assert schema is not None
    assert schema.instances == 2
