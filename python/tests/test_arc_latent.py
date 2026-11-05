"""Tests for ARC grid decoding, topology, and action-conditioned latent dynamics."""

import pytest

from worlds.games.latent import (
    ArcEnvironmentDecoder,
    FEATURE_DIMENSIONS,
    GoalEvidence,
    GridState,
    SceneGraph,
)


def test_grid_state_rejects_ragged_and_invalid_arc_grids():
    with pytest.raises(ValueError, match="equal width"):
        GridState.from_rows([[0, 1], [2]])
    with pytest.raises(ValueError, match="palette"):
        GridState.from_rows([[0, 16]])
    with pytest.raises(ValueError, match="positive"):
        GridState(0, 1, ())


def test_scene_graph_extracts_components_and_exact_enclosed_holes():
    grid = [
        [9, 9, 9, 9, 9],
        [9, 0, 0, 0, 9],
        [9, 0, 0, 0, 9],
        [9, 0, 0, 0, 9],
        [9, 9, 9, 9, 9],
    ]
    graph = SceneGraph.extract(GridState.from_rows(grid))
    ring = next(node for node in graph.nodes if node.color == 9)
    assert ring.pixel_count == 16
    assert ring.betti_0 == 1
    assert ring.betti_1 == 1
    assert ring.bbox == (0, 0, 4, 4)


def test_scene_graph_uses_stable_object_ids_and_spatial_edges():
    grid = [
        [0, 9, 8, 0, 9],
        [0, 9, 8, 0, 9],
    ]
    first = SceneGraph.extract(GridState.from_rows(grid))
    second = SceneGraph.extract(GridState.from_rows(grid))
    assert first.nodes == second.nodes
    assert [node.id for node in first.nodes] == list(range(len(first.nodes)))
    assert len(first.edges) == 2
    assert all(edge.adjacency for edge in first.edges)
    assert all(edge.source != edge.target for edge in first.edges)


def test_latent_encoding_is_fixed_width_and_keeps_graph_evidence():
    decoder = ArcEnvironmentDecoder()
    state = decoder.encode([[0, 9, 0], [0, 9, 0], [8, 0, 8]])
    assert len(state.vector) == FEATURE_DIMENSIONS
    assert len(state.epistemic_variance) == FEATURE_DIMENSIONS
    assert state.graph.width == 3
    assert len(state.graph.nodes) == 3
    assert state.to_dict()["object_count"] == 3
    assert state.to_dict()["objects"]
    typed = state.to_experiential_latent()
    assert typed.domain.value == "experiential_outer"
    assert typed.category == "arc_observation"
    assert typed.values == state.vector


def test_action_dynamics_are_learned_from_observed_transitions_only():
    decoder = ArcEnvironmentDecoder()
    before = decoder.encode([[0, 9], [0, 0]])
    after = decoder.encode([[0, 0], [0, 9]])
    assert decoder.predict("ACTION1", before) is None

    decoder.observe_transition("ACTION1", before, after)
    prediction = decoder.predict("ACTION1", before)
    assert prediction is not None
    assert prediction.vector == pytest.approx(after.vector)
    assert prediction.distance_to(after) == pytest.approx(0.0)
    assert min(prediction.epistemic_variance) >= 1.0
    assert decoder.action_evidence()["ACTION1"]["samples"] == 1


def test_repeated_consistent_transitions_reduce_epistemic_uncertainty():
    decoder = ArcEnvironmentDecoder()
    before = decoder.encode([[0, 9], [0, 0]])
    after = decoder.encode([[0, 0], [0, 9]])
    decoder.observe_transition("ACTION1", before, after)
    decoder.observe_transition("ACTION1", before, after)
    prediction = decoder.predict("ACTION1", before)
    assert prediction is not None
    assert prediction.epistemic_variance == before.epistemic_variance
    assert decoder.action_evidence()["ACTION1"]["samples"] == 2


def test_action_model_does_not_extrapolate_to_a_distant_latent_context():
    decoder = ArcEnvironmentDecoder()
    before = decoder.encode([[0, 9], [0, 0]])
    after = decoder.encode([[0, 0], [0, 9]])
    distant = decoder.encode([[1] * 16 for _ in range(16)])
    decoder.observe_transition("ACTION1", before, after)
    assert decoder.predict("ACTION1", distant) is None


def test_transition_model_roundtrips_without_fabricating_samples():
    decoder = ArcEnvironmentDecoder()
    before = decoder.encode([[0, 9], [0, 0]])
    after = decoder.encode([[0, 0], [0, 9]])
    decoder.observe_transition("ACTION2", before, after)
    restored = ArcEnvironmentDecoder(decoder.to_dict())
    assert restored.action_evidence() == decoder.action_evidence()
    assert restored.predict("ACTION2", before) is not None
    assert restored.predict("ACTION7", before) is None


def test_goal_evidence_uses_observed_level_and_terminal_state():
    progress = GoalEvidence("NOT_FINISHED", 1, 2, 4)
    win = GoalEvidence("WIN", 2, 2, 0)
    assert progress.level_progress
    assert not progress.terminal_success
    assert not win.level_progress
    assert win.terminal_success
