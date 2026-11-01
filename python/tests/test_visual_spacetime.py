"""Tests for the calibrated visual causal bridge."""

from occipital_lobe.epipolar import TwoViewReport
from occipital_lobe.visual_spacetime import VisualSpacetimeBridge
from temporal_lobe.light_cones import WorldDomain
from temporal_lobe.spacetime_workspace import SpacetimeWorkspace


def test_bridge_refuses_uncalibrated_visual_coordinates():
    bridge = VisualSpacetimeBridge(SpacetimeWorkspace())
    result = bridge.publish(TwoViewReport(kind="travel", inlier_ratio=0.9), time_s=1.0)
    assert not result.accepted
    assert result.event is None
    assert "calibrated" in result.reason


def test_bridge_publishes_only_through_explicit_transform():
    workspace = SpacetimeWorkspace()
    workspace.record_self((0.0, 0.0, 0.0), time_s=1.0, event_id="self")
    bridge = VisualSpacetimeBridge(
        workspace,
        camera_to_workspace=lambda report: (1.0, 2.0, 0.0),
    )
    result = bridge.publish(
        TwoViewReport(kind="travel", inlier_ratio=0.8),
        time_s=2.0,
        event_id="camera-2",
    )
    assert result.accepted
    assert result.event is not None
    assert result.event.domain is WorldDomain.OUTER
    assert result.event.position == (1.0, 2.0, 0.0)
    assert workspace.snapshot().outer_view[0].event_id == "camera-2"


def test_bridge_rejects_unresolved_visual_geometry():
    bridge = VisualSpacetimeBridge(
        SpacetimeWorkspace(),
        camera_to_workspace=lambda report: (0.0, 0.0, 0.0),
    )
    result = bridge.publish(TwoViewReport(kind="unresolved"), time_s=1.0)
    assert not result.accepted
