"""Regression tests for concept-network restore caches."""
from genesis_cognitive.concepts import ConceptNetwork


def test_replace_contents_invalidates_sparse_activation_index():
    old = ConceptNetwork()
    old.add_concept("old")
    old.get_concept("old").activation = 0.8
    old.tick()
    assert old._active_ids is not None

    fresh = ConceptNetwork()
    fresh.add_concept("new")
    fresh.get_concept("new").activation = 0.8

    old.replace_contents(fresh)

    assert old._active_ids is None
    assert old._ticks_since_sweep == 0
    assert old.get_concept("old") is None
    assert old.get_concept("new") is not None
