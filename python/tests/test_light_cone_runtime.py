"""Tests for the live world-event to light-cone bridge."""

from __future__ import annotations

import numpy as np

from temporal_lobe.events import EventKind, ExternalEvent
from temporal_lobe.light_cone_runtime import LightConeExperienceBridge
from temporal_lobe.light_cones import WorldDomain


class FakeEmbeddings:
    def __init__(self, vectors):
        self.vectors = vectors

    def get_concept_vector(self, name):
        return self.vectors.get(name)


def test_bridge_uses_existing_topic_vectors_and_updates_self_apex():
    bridge = LightConeExperienceBridge()
    embeddings = FakeEmbeddings({"grid-pattern": np.array([1.0, 2.0, 3.0])})
    event = ExternalEvent(
        kind=EventKind.PERCEPTION,
        source="arc",
        content="grid observed",
        salience=0.7,
        topics=["grid-pattern"],
        timestamp=10.0,
    )

    assert bridge.process(event, embeddings) is True
    apex = bridge.manifold.apex
    assert apex is not None
    assert apex.domain is WorldDomain.SELF
    assert apex.time_s == 0.0
    assert apex.causes[0].startswith("world:arc:")
    assert bridge.manifold.hologram.plastic_state == (
        complex(0.05, 0.0),
        complex(0.1, 0.0),
        complex(0.15, 0.0),
    )


def test_bridge_skips_events_without_real_vectors():
    bridge = LightConeExperienceBridge()
    event = ExternalEvent(
        kind=EventKind.PERCEPTION,
        source="arc",
        content="unmapped observation",
        topics=["missing"],
        timestamp=10.0,
    )

    assert bridge.process(event, FakeEmbeddings({})) is False
    assert bridge.manifold.apex is None


def test_bridge_clamps_delayed_event_time_without_reversing_apex():
    bridge = LightConeExperienceBridge()
    embeddings = FakeEmbeddings({"concept": np.array([1.0])})
    for timestamp in (20.0, 15.0):
        event = ExternalEvent(
            kind=EventKind.PERCEPTION,
            source="sensor",
            content="observation",
            topics=["concept"],
            timestamp=timestamp,
        )
        assert bridge.process(event, embeddings) is True
    assert bridge.manifold.apex.time_s == 0.0
    assert bridge.manifold.summary()["experience_count"] == 2
