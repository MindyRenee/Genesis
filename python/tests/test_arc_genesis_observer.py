"""Tests for bridge-to-Genesis event routing."""
from __future__ import annotations

from parietal_lobe.arc_genesis_observer import GenesisArcObserver


class FakeWorld:
    def __init__(self):
        self.events = []

    def perceive(self, content, **kwargs):
        self.events.append(("observation", content, kwargs))
        return object()

    def it_acted(self, description, *, detail=""):
        self.events.append(("action", description, detail))
        return object()


class FakeMind:
    def __init__(self):
        self.world = FakeWorld()


def test_observer_routes_observation_into_world():
    mind = FakeMind()
    observer = GenesisArcObserver(mind)
    observer("observation", {
        "game_id": "demo", "step": 0, "phase": "initial",
        "observation": {"state": "PLAY", "grid": [[1, 0]]},
        "legal_actions": ["ACTION1"],
    })
    kind, content, kwargs = mind.world.events[0]
    assert kind == "observation"
    assert "demo" in content
    assert kwargs["source"] == "arc3:demo"
    assert kwargs["metadata"]["legal_actions"] == ["ACTION1"]
    assert kwargs["percept"]["grid"] == [[1, 0]]


def test_observer_routes_action_with_transition_telemetry():
    mind = FakeMind()
    observer = GenesisArcObserver(mind)
    observer("action", {
        "game_id": "demo", "step": 1, "action": "ACTION1", "data": None,
        "reasoning": {"hypothesis": "test"},
        "observation_before": {"state": "PLAY"},
        "observation_after": {"state": "WIN"},
    })
    kind, content, detail = mind.world.events[0]
    assert kind == "action"
    assert "ACTION1" in content
    assert '"state_after":"WIN"' in detail
    assert observer.snapshot() == {"events_seen": 1, "events_failed": 0}
