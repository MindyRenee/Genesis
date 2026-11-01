"""Tests for bridge-to-Genesis event and latent-state routing."""
from __future__ import annotations

from types import SimpleNamespace

from parietal_lobe.arc_genesis_observer import GenesisArcObserver
from temporal_lobe.comprehension import Comprehension
from temporal_lobe.language_latent_space import LanguageLatentSpace


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
        self.language_latent_space = LanguageLatentSpace()
        self.brain = SimpleNamespace(
            temporal=SimpleNamespace(
                comprehension=Comprehension(),
                observe_language=self.language_latent_space.observe,
            )
        )


def test_observer_routes_observation_into_world_and_language_latent():
    mind = FakeMind()
    observer = GenesisArcObserver(mind)
    observer(
        "observation",
        {
            "game_id": "demo",
            "step": 0,
            "phase": "initial",
            "observation": {"state": "PLAY", "grid": [[1, 0]]},
            "legal_actions": ["ACTION1"],
        },
    )
    kind, content, kwargs = mind.world.events[0]
    assert kind == "observation"
    assert "demo" in content
    assert kwargs["source"] == "arc3:demo"
    assert kwargs["metadata"]["legal_actions"] == ["ACTION1"]
    assert kwargs["percept"]["grid"] == [[1, 0]]
    assert mind.language_latent_space.observations == 1
    assert observer.snapshot()["latent_observations"] == 1


def test_observer_routes_action_and_updates_transition_geometry():
    mind = FakeMind()
    observer = GenesisArcObserver(mind)
    observer(
        "observation",
        {
            "game_id": "demo",
            "step": 0,
            "phase": "initial",
            "observation": {"state": "PLAY"},
            "legal_actions": ["ACTION1"],
        },
    )
    observer(
        "action",
        {
            "game_id": "demo",
            "step": 1,
            "action": "ACTION1",
            "data": None,
            "reasoning": {"hypothesis": "test"},
            "observation_before": {"state": "PLAY"},
            "observation_after": {"state": "WIN"},
        },
    )
    kind, content, detail = mind.world.events[1]
    assert kind == "action"
    assert "ACTION1" in content
    assert '"state_after":"WIN"' in detail
    assert mind.language_latent_space.observations == 2
    snapshot = observer.snapshot()
    assert snapshot["events_seen"] == 2
    assert snapshot["events_failed"] == 0
    assert snapshot["latent_observations"] == 2
    assert snapshot["last_transition_residual"] > 0.0


def test_observer_rejects_unknown_event():
    mind = FakeMind()
    observer = GenesisArcObserver(mind)
    try:
        observer("unknown", {"game_id": "demo"})
    except ValueError as exc:
        assert "unknown ARC observer event" in str(exc)
    else:
        raise AssertionError("unknown event should fail")
    assert observer.snapshot()["events_failed"] == 1
