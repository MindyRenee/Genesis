"""Regression tests for semantic persistence validation."""
from genesis_cognitive.memory.semantic import SemanticMemory

def test_restore_rejects_nonfinite_fact_confidence():
    memory = SemanticMemory()
    memory.restore({"facts": [
        {"subject": "bad", "relation": "is_a", "object": "thing", "confidence": "nan"},
        {"subject": "good", "relation": "is_a", "object": "thing", "confidence": 2.0},
    ]})
    assert memory.get_all_facts()[0].subject == "good"
    assert memory.get_all_facts()[0].confidence == 1.0

def test_restore_rejects_invalid_schema_numbers():
    memory = SemanticMemory()
    memory.restore({"schemas": [
        {"concept": "bad", "instances": -1},
        {"concept": "nan", "abstraction_level": "nan"},
        {"concept": "good", "instances": 2, "abstraction_level": 3.0},
    ]})
    assert memory.get_schema("bad") is None
    assert memory.get_schema("nan") is None
    assert memory.get_schema("good").abstraction_level == 1.0
