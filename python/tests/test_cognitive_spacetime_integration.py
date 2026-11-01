"""Integration tests: cognitive-map observations feed the shared causal workspace."""

from temporal_lobe.cognitive_map import CognitiveMap
from temporal_lobe.light_cones import WorldDomain


def test_cognitive_map_publishes_self_and_outer_events():
    cognitive_map = CognitiveMap()
    cognitive_map.observe(position=(2.0, 3.0), entities=[], bounds=(10, 10))
    snapshot = cognitive_map.spacetime.snapshot()
    assert snapshot.apex.domain is WorldDomain.SELF
    assert snapshot.apex.position == (2.0, 3.0, 0.0)
    assert snapshot.outer_view
    assert snapshot.outer_view[0].domain is WorldDomain.OUTER


def test_cognitive_map_context_change_is_recorded_as_inner_event():
    cognitive_map = CognitiveMap()
    events = cognitive_map.observe(position=(1.0, 1.0), entities=[], bounds=(8, 8))
    assert any(event.kind == "context" for event in events)
    snapshot = cognitive_map.spacetime.snapshot()
    assert snapshot.inner_view
    assert snapshot.inner_view[0].domain is WorldDomain.INNER
