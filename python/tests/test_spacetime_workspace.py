"""Tests for the shared three-domain spacetime workspace."""

import pytest

from temporal_lobe.light_cones import WorldDomain
from temporal_lobe.spacetime_workspace import SpacetimeWorkspace


def test_workspace_exposes_three_views_against_one_self_apex():
    workspace = SpacetimeWorkspace(signal_speed=1.0)
    workspace.record(time_s=8.0, position=(1.0, 0.0, 0.0), domain=WorldDomain.INNER,
                     event_id="memory")
    workspace.record(time_s=9.0, position=(2.0, 0.0, 0.0), domain=WorldDomain.OUTER,
                     event_id="camera")
    apex = workspace.record_self((0.0, 0.0, 0.0), time_s=10.0, event_id="self-now")
    snapshot = workspace.snapshot()
    assert snapshot.apex is apex
    assert [r.event_id for r in snapshot.inner_view] == ["memory"]
    assert [r.event_id for r in snapshot.outer_view] == ["camera"]
    assert snapshot.inner_view[0].in_past_cone
    assert snapshot.outer_view[0].in_past_cone


def test_only_self_events_can_advance_apex():
    workspace = SpacetimeWorkspace()
    with pytest.raises(ValueError, match="record_self"):
        workspace.record(WorldDomain.SELF, (0.0, 0.0, 0.0), time_s=1.0)


def test_snapshot_requires_apex():
    with pytest.raises(ValueError, match="record_self"):
        SpacetimeWorkspace().snapshot()


def test_bulk_registration_rejects_self_events():
    workspace = SpacetimeWorkspace()
    event = workspace.record_self((0.0, 0.0, 0.0), time_s=1.0)
    with pytest.raises(ValueError, match="cannot advance"):
        workspace.register_events([event])
