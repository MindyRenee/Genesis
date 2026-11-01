"""Integration tests: cognitive-map observations feed the shared causal workspace."""

from temporal_lobe.cognitive_map import CognitiveMap
import time

from occipital_lobe.epipolar import TwoViewReport
from occipital_lobe.visual_spacetime import VisualSpacetimeBridge
from temporal_lobe.light_cones import WorldDomain
from temporal_lobe.spacetime_workspace import SpacetimeWorkspace


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


def test_visual_and_map_events_share_one_workspace():
    workspace = SpacetimeWorkspace()
    cognitive_map = CognitiveMap(spacetime=workspace)
    cognitive_map.observe(position=(2.0, 3.0), entities=[], bounds=(10, 10))
    bridge = VisualSpacetimeBridge(
        workspace,
        camera_to_workspace=lambda report: (4.0, 5.0, 0.0),
    )
    result = bridge.publish(
        TwoViewReport(kind="travel", inlier_ratio=0.75),
        time_s=time.monotonic(),
        event_id="camera:integration",
    )
    assert result.accepted
    snapshot = workspace.snapshot()
    assert snapshot.apex.domain is WorldDomain.SELF
    assert any(item.event_id == "camera:integration" for item in snapshot.outer_view)
    assert any(item.event_id.startswith("map-observation:") for item in snapshot.outer_view)
