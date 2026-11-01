"""Tests for the outer world's numerical experiential geometry."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from temporal_lobe.events import EventKind, ExternalEvent  # noqa: E402
from worlds.experiential_space import ExperientialLatentSpace  # noqa: E402
from worlds.world import OuterWorld  # noqa: E402


def _event(kind=EventKind.PERCEPTION, timestamp=1_700_000_000.0):
    return ExternalEvent(
        kind=kind,
        source="sensor",
        content="a structured observation",
        salience=0.7,
        addressed=False,
        topics=["observation"],
        timestamp=timestamp,
    )


def test_experiential_space_uses_observed_events_and_returns_finite_geometry():
    space = ExperientialLatentSpace()
    result = space.observe(_event())
    assert result["observations"] == 1
    assert len(result["state"]) == 22
    assert np.asarray(result["base_metric"]).shape == (4, 4)
    assert np.asarray(result["fisher_information"]).shape == (22, 22)
    assert len(result["global_geometry"]["hyperbolic_state"]) == 7
    assert len(result["global_geometry"]["spherical_state"]) == 7
    assert len(result["global_geometry"]["euclidean_state"]) == 8
    assert len(result["toroidal_state"]) == 8
    assert np.all(np.isfinite(np.asarray(result["fisher_information"])))
    assert np.all(np.linalg.eigvalsh(np.asarray(result["fisher_information"])) > 0)
    assert np.all(np.linalg.eigvalsh(np.asarray(result["base_metric"])) > 0)
    geometry = result["global_geometry"]
    assert np.linalg.norm(geometry["hyperbolic_state"]) < 1.0
    np.testing.assert_allclose(np.linalg.norm(geometry["spherical_state"]), 1.0)
    assert all(
        np.isfinite(value)
        for value in geometry["distances_from_reference"].values()
    )
    assert result["fisher_volume"] > 0


def test_toroidal_factors_lie_on_unit_circles():
    result = ExperientialLatentSpace().observe(_event())
    torus = np.asarray(result["toroidal_state"]).reshape(4, 2)
    np.testing.assert_allclose(np.sum(torus * torus, axis=1), 1.0, atol=1e-12)


def test_hamiltonian_flow_preserves_phase_space_norm():
    space = ExperientialLatentSpace()
    result = space.observe(_event())
    before = np.asarray(result["state"])
    after = np.asarray(result["symplectic_state"])
    np.testing.assert_allclose(
        np.linalg.norm(before), np.linalg.norm(after), rtol=1e-12, atol=1e-12
    )


def test_experiential_state_roundtrip_preserves_online_statistics():
    original = ExperientialLatentSpace()
    original.observe(_event(EventKind.PERCEPTION))
    original.observe(_event(EventKind.USER_SPEECH, 1_700_000_010.0))
    restored = ExperientialLatentSpace()
    restored.restore(original.to_dict())
    assert restored.observations == original.observations
    np.testing.assert_allclose(restored.snapshot()["state"], original.snapshot()["state"])
    np.testing.assert_allclose(
        restored.snapshot()["fisher_information"],
        original.snapshot()["fisher_information"],
    )


def test_outer_world_records_each_event_into_experiential_space_and_persists_it():
    world = OuterWorld()
    world.perceive(
        "a visual observation", source="camera", percept=np.array([[0, 1], [2, 3]])
    )
    assert world.experiential_space.observations == 1
    saved = world.to_dict()
    restored = OuterWorld()
    restored.restore_from_dict(saved)
    assert restored.experiential_space.observations == 1
    assert len(restored.experiential_space.snapshot()["state"]) == 22


def test_structured_payload_changes_sensory_coordinates():
    first = ExperientialLatentSpace().observe(_event())
    second = ExperientialLatentSpace().observe(
        ExternalEvent(
            kind=EventKind.PERCEPTION,
            source="sensor",
            content="a structured observation",
            salience=0.7,
            percept=np.array([[0.0, 1.0], [2.0, 3.0]]),
            timestamp=1_700_000_000.0,
        )
    )
    assert first["state"][16:20] != second["state"][16:20]
    assert first["toroidal_state"] != second["toroidal_state"]


def test_valence_sign_is_positive_negative_or_neutral():
    for value, expected in ((0.4, "positive"), (-0.4, "negative"), (0.0, "neutral")):
        event = _event()
        event.metadata["valence"] = value
        result = ExperientialLatentSpace().observe(event)
        assert result["valence"] == value
        assert result["valence_sign"] == expected


def test_user_speech_sentiment_reaches_outer_world_valence():
    from limbic_system.sentiment import (
        LEARNED_LEXICON,
        clear_learned_sentiment,
        learn_sentiment,
    )

    prior_lexicon = LEARNED_LEXICON.copy()
    clear_learned_sentiment()
    learn_sentiment("wonderful", 3.0)
    learn_sentiment("excellent", 3.0)
    try:
        world = OuterWorld()
        world.hear_user("This is wonderful and excellent.")
        result = world.experiential_space.snapshot()
        assert result["valence"] > 0.0
        assert result["valence_sign"] == "positive"
    finally:
        clear_learned_sentiment()
        LEARNED_LEXICON.update(prior_lexicon)


def test_nested_mapping_payload_extracts_numeric_grid():
    event = _event()
    event.percept = {"grid": [[0, 0, 1], [0, 1, 0]], "label": "grid"}
    result = ExperientialLatentSpace().observe(event)
    assert result["state"][21] == 1.0
    assert result["state"][17] > 0.0
