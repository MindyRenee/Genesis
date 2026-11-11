"""Tests for ARC grid decoding, topology, and action-conditioned latent dynamics."""

import pytest
from worlds.games.latent import (
    FEATURE_DIMENSIONS,
    ArcEnvironmentDecoder,
    DifferentiableOperatorSpace,
    GoalAttractorManifold,
    GoalEvidence,
    GridState,
    LatentState,
    SceneGraph,
    TransitionOperator,
    compute_local_betti,
    validate_lie_bracket_closure,
)


def test_grid_state_rejects_ragged_and_invalid_arc_grids():
    with pytest.raises(ValueError, match="equal width"):
        GridState.from_rows([[0, 1], [2]])
    with pytest.raises(ValueError, match="palette"):
        GridState.from_rows([[0, 16]])
    with pytest.raises(ValueError, match="positive"):
        GridState(0, 1, ())
    with pytest.raises(ValueError, match="integer"):
        GridState(1, 1, (True,))
    assert GridState.from_rows([[2, 3]]).get(1, 0) == 3
    assert GridState(2, 1, [2, 3]).data == (2, 3)


def test_scene_graph_extracts_components_and_exact_enclosed_holes():
    grid = [
        [9, 9, 9, 9, 9],
        [9, 0, 0, 0, 9],
        [9, 0, 0, 0, 9],
        [9, 0, 0, 0, 9],
        [9, 9, 9, 9, 9],
    ]
    # The enclosed zero-valued region is the background for this fixture;
    # specify it explicitly because the ring color is the grid's mode.
    graph = SceneGraph.extract(GridState.from_rows(grid), background=0)
    ring = next(node for node in graph.nodes if node.color == 9)
    assert ring.pixel_count == 16
    assert ring.betti_0 == 1
    assert ring.betti_1 == 1
    assert graph.betti_0 == 1
    assert graph.betti_1 == 1
    ring_cells = frozenset(
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value == 9
    )
    assert compute_local_betti(ring_cells, 0, 0, 4, 4) == (1, 1)
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
    state = decoder.encode([[0, 9, 8], [0, 9, 0], [8, 0, 8]])
    assert len(state.vector) == FEATURE_DIMENSIONS
    assert len(state.epistemic_variance) == FEATURE_DIMENSIONS
    assert state.graph.width == 3
    assert len(state.graph.nodes) == 4
    assert state.to_dict()["object_count"] == 4
    assert state.to_dict()["objects"]
    assert state.to_dict()["edges"]
    typed = state.to_experiential_latent()
    assert typed.domain == "outer"
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
    assert prediction.epistemic_variance == pytest.approx((0.0,) * FEATURE_DIMENSIONS)
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


def test_empty_board_has_a_valid_zero_object_embedding():
    state = ArcEnvironmentDecoder().encode([[0, 0], [0, 0]])
    assert state.graph.nodes == ()
    assert state.graph.edges == ()
    assert len(state.vector) == FEATURE_DIMENSIONS
    assert all(value == 0.0 for value in state.vector[3:])


def test_malformed_saved_transition_context_is_ignored():
    decoder = ArcEnvironmentDecoder(
        {"ACTION1": [{"samples": "not-a-count", "reference_state": []}]}
    )
    assert decoder.action_evidence() == {}


def test_scene_graph_infers_nonzero_background_using_parietal_rule():
    grid = [[7, 7, 7], [7, 9, 7], [7, 7, 7]]
    graph = SceneGraph.extract(GridState.from_rows(grid))
    assert graph.background == 7
    assert len(graph.nodes) == 1
    assert graph.nodes[0].color == 9


def test_constant_metric_geodesic_distance_and_goal_attractor():
    graph = SceneGraph(1, 1, 0, (), ())
    first = LatentState((0.0, 0.0), (1.0, 1.0), graph)
    second = LatentState((3.0, 4.0), (1.0, 1.0), graph)
    metric = ((2.0, 0.0), (0.0, 1.0))
    assert first.geodesic_distance(second, metric) == pytest.approx((34.0) ** 0.5)
    goal = GoalAttractorManifold((3.0, 4.0), metric, 0.01)
    assert goal.is_satisfied(second)
    assert not goal.is_satisfied(first)
    with pytest.raises(ValueError, match="positive definite"):
        GoalAttractorManifold((0.0, 0.0), ((1.0, 2.0), (2.0, 1.0)), 1.0)


def test_affine_transition_operator_applies_and_computes_lie_brackets():
    scale = TransitionOperator(((2.0, 0.0), (0.0, 1.0)), (1.0, -1.0))
    assert scale.apply((3.0, 4.0)) == pytest.approx((7.0, 3.0))
    with pytest.raises(ValueError, match="dimensions"):
        scale.apply((1.0,))
    e = TransitionOperator(((0.0, 1.0), (0.0, 0.0)), (0.0, 0.0))
    f = TransitionOperator(((0.0, 0.0), (1.0, 0.0)), (0.0, 0.0))
    h = TransitionOperator(((1.0, 0.0), (0.0, -1.0)), (0.0, 0.0))
    closed = DifferentiableOperatorSpace({"e": e, "f": f, "h": h})
    assert closed.bracket("e", "f").matrix == ((-1.0, 0.0), (0.0, 1.0))
    assert validate_lie_bracket_closure(closed)
    assert not DifferentiableOperatorSpace({"e": e, "f": f}).validate_lie_bracket_closure()


def test_operator_space_does_not_infer_operators_for_opaque_actions():
    space = DifferentiableOperatorSpace()
    with pytest.raises(KeyError, match="unknown transition operator"):
        space.apply("ACTION1", (0.0,))
