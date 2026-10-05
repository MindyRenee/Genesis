"""Regression tests for semantic persistence replacement."""
from genesis_cognitive.memory.semantic import Fact, SemanticMemory

def test_restore_replaces_existing_semantic_state():
    memory = SemanticMemory()
    memory.extract_facts("Cats are mammals.")
    assert memory.fact_count
    memory.restore({
        "facts": [{
            "subject": "dogs",
            "relation": "is_a",
            "object": "mammals",
            "confidence": 0.8,
            "source_count": 2,
            "extracted_at": 10,
            "last_reinforced": 20,
        }],
        "schemas": [{
            "concept": "dogs",
            "parts": [],
            "properties": [],
            "functions": [],
            "relations": {"is_a": ["mammals"]},
            "instances": 2,
            "abstraction_level": 0.5,
            "created_at": 10,
            "last_updated": 20,
        }],
        "facts_extracted": 1,
        "schemas_formed": 1,
        "consolidations": 0,
    })
    assert memory.fact_count == 1
    assert memory.schema_count == 1
    assert memory.get_all_facts()[0].key == ("dogs", "is_a", "mammals")
    assert memory.get_schema("cats") is None
    assert memory.get_schema("dogs").instances == 2
