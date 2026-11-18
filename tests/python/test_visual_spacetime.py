"""Tests for the calibrated visual causal bridge."""

import pytest
from occipital_lobe.epipolar import TwoViewReport
from occipital_lobe.visual_spacetime import VisualSpacetimeBridge, egocentric_projection
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


def test_egocentric_projection_travel_uses_baseline_depth():
    report = TwoViewReport(kind="travel", bearing_deg=90.0, elevation_deg=0.0, median_depth=2.0)
    x, y, z = egocentric_projection(report)
    assert x == pytest.approx(2.0)  # 90° right of forward lands on +X
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(0.0)


def test_egocentric_projection_forward_and_elevation():
    report = TwoViewReport(kind="travel", bearing_deg=0.0, elevation_deg=90.0, median_depth=1.5)
    x, y, z = egocentric_projection(report)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(1.5)  # straight up is +Z


def test_egocentric_projection_non_travel_lands_on_unit_sphere():
    for kind in ("still", "turn"):
        report = TwoViewReport(kind=kind, bearing_deg=0.0, elevation_deg=0.0, median_depth=0.0)
        x, y, z = egocentric_projection(report)
        assert (x, y, z) == pytest.approx((0.0, 1.0, 0.0))  # direction forward, range nominal


def test_egocentric_projection_zero_depth_travel_is_nominal():
    report = TwoViewReport(kind="travel", bearing_deg=180.0, elevation_deg=0.0, median_depth=0.0)
    x, y, z = egocentric_projection(report)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(-1.0)  # 180° = behind
    assert z == pytest.approx(0.0)


def test_egocentric_projection_within_flips_horizontal_axis():
    # A within (mirror) look reads directions with facing handedness —
    # a structure at her right reports bearing 90° in mirror view. The
    # canonical workspace frame un-flips it to her actual left.
    report = TwoViewReport(
        kind="travel", bearing_deg=90.0, elevation_deg=0.0, median_depth=2.0, facing="within"
    )
    x, y, z = egocentric_projection(report)
    assert x == pytest.approx(-2.0)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(0.0)


def test_egocentric_projection_up_tips_forward_to_the_pole():
    # Looking up, the optical axis is +Z: image-forward lands overhead.
    report = TwoViewReport(
        kind="travel", bearing_deg=0.0, elevation_deg=0.0, median_depth=1.0, facing="up"
    )
    x, y, z = egocentric_projection(report)
    assert x == pytest.approx(0.0, abs=1e-9)
    assert y == pytest.approx(0.0, abs=1e-9)
    assert z == pytest.approx(1.0)


def test_egocentric_projection_down_is_up_opposite():
    # up and down are the pair — the same look tipped −90° lands below.
    report = TwoViewReport(
        kind="travel", bearing_deg=0.0, elevation_deg=0.0, median_depth=1.0, facing="down"
    )
    x, y, z = egocentric_projection(report)
    assert (x, y, z) == pytest.approx((0.0, 0.0, -1.0))


def test_egocentric_projection_up_keeps_lateral_axis():
    # A look-up's right stays her right: bearing 90° under "up" is +X.
    report = TwoViewReport(
        kind="travel", bearing_deg=90.0, elevation_deg=0.0, median_depth=1.0, facing="up"
    )
    x, y, z = egocentric_projection(report)
    assert (x, y, z) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
