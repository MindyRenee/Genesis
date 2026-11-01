"""Tests for the NumPy-native geometric self latent space."""

from __future__ import annotations

import pytest
import numpy as np

from self.latent_geometry import (
    AgencyOwnershipFiberBundle,
    AutopoieticSymplecticFlow,
    GrassmannianSubspaceModule,
    NarrativeAgencyBaseManifold,
    SelfConsistencyInformationGeometry,
    SelfLatentSpaceArchitecture,
)


def test_narrative_metric_is_symmetric_positive_definite() -> None:
    coords = np.array([[1.0, -2.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])
    metric, weights = NarrativeAgencyBaseManifold().forward(coords, [[0.5], [-1.0]])
    assert metric.shape == (2, 4, 4)
    assert weights.shape == (2,)
    for matrix in metric:
        assert np.allclose(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() > 0


def test_narrative_metric_rejects_wrong_dimensions() -> None:
    with pytest.raises(ValueError, match="columns"):
        NarrativeAgencyBaseManifold().forward(np.zeros((2, 3)), np.zeros((2, 1)))


def test_fiber_has_expected_shape_and_uses_input() -> None:
    encoder = AgencyOwnershipFiberBundle(input_dim=8, toroidal_factors=4, fiber_dim=32)
    first = encoder.forward(np.zeros((3, 8)))
    second = encoder.forward(np.ones((3, 8)))
    assert first.shape == (3, 40)
    assert np.isfinite(first).all()
    assert not np.allclose(first, second)


def test_grassmannian_projector_is_symmetric_idempotent_and_rank_k() -> None:
    module = GrassmannianSubspaceModule(latent_dim=10, subspace_dim=4)
    states = np.arange(30, dtype=float).reshape(3, 10)
    projected, projector = module.forward(states)
    assert projected.shape == states.shape
    assert np.allclose(projector, projector.T, atol=1e-12)
    assert np.allclose(projector @ projector, projector, atol=1e-12)
    assert np.linalg.matrix_rank(projector, tol=1e-10) == 4


def test_consistency_metric_is_spd_and_loss_is_squared_residual() -> None:
    states = np.array([[1.0, 2.0], [3.0, 4.0]])
    actions = np.zeros_like(states)
    metric, loss, predicted = SelfConsistencyInformationGeometry(2).forward(states, actions)
    assert np.allclose(predicted, states)
    assert np.allclose(loss, [5.0, 25.0])
    for matrix in metric:
        assert np.linalg.eigvalsh(matrix).min() > 0


def test_hamiltonian_midpoint_flow_preserves_quadratic_energy() -> None:
    flow = AutopoieticSymplecticFlow(dim=4, stiffness=1.0)
    state = np.array([[1.0, 2.0, 3.0, 4.0]])
    evolved = flow.forward(state, dt=0.1)
    assert evolved.shape == state.shape
    assert np.allclose(np.sum(evolved**2), np.sum(state**2), rtol=1e-12, atol=1e-12)


def test_hamiltonian_flow_rejects_odd_dimension() -> None:
    with pytest.raises(ValueError, match="even"):
        AutopoieticSymplecticFlow(dim=3)


def test_integrated_architecture_shapes_and_finite_outputs() -> None:
    model = SelfLatentSpaceArchitecture()
    batch = 4
    output = model.forward(
        np.zeros((batch, 4)),
        np.zeros((batch, 1)),
        np.ones((batch, 8)),
        np.zeros((batch, 40)),
    )
    assert output["narrative_base_metric"].shape == (batch, 4, 4)
    assert output["grassmannian_subspace_states"].shape == (batch, 40)
    assert output["grassmannian_projection_operator"].shape == (40, 40)
    assert output["consistency_information_metric"].shape == (batch, 40, 40)
    assert output["autopoietic_evolved_states"].shape == (batch, 40)
    assert all(np.isfinite(value).all() for value in output.values())


def test_integrated_architecture_rejects_batch_mismatch() -> None:
    model = SelfLatentSpaceArchitecture()
    with pytest.raises(ValueError, match="batch size"):
        model.forward(
            np.zeros((2, 4)),
            np.zeros((2, 1)),
            np.zeros((3, 8)),
            np.zeros((3, 40)),
        )



def test_real_signal_observation_updates_shared_transition_state() -> None:
    model = SelfLatentSpaceArchitecture()
    first = model.observe_signal(
        "Genesis receives an observation",
        kind="world:perception",
        salience=0.7,
        inbound=True,
        metadata={"source": "test"},
    )
    second = model.observe_signal(
        "Genesis chooses an action",
        kind="world:action",
        salience=0.5,
        inbound=False,
        metadata={"success": True},
    )
    assert first["observations"] == 1
    assert second["observations"] == 2
    assert first["kind"] == "world:perception"
    assert second["kind"] == "world:action"
    assert len(second["agency_fiber_states"]) == 40
    assert second["coherence_loss"] > 0.0
    assert model.snapshot()["observations"] == 2


def test_real_signal_observation_is_deterministic_and_validates_inputs() -> None:
    model = SelfLatentSpaceArchitecture()
    first = model.observe_signal("same input", kind="thought:reflection")
    other = SelfLatentSpaceArchitecture().observe_signal(
        "same input", kind="thought:reflection"
    )
    assert first["agency_fiber_states"] == other["agency_fiber_states"]
    with pytest.raises(ValueError, match="salience"):
        model.observe_signal("bad", kind="test", salience=2.0)



def test_self_latent_state_round_trips_with_transition_context() -> None:
    original = SelfLatentSpaceArchitecture()
    original.observe_signal("perception", kind="world:perception")
    original.observe_signal("action", kind="world:action", inbound=False)
    payload = original.to_dict()

    restored = SelfLatentSpaceArchitecture()
    restored.restore(payload)
    assert restored.observations == 2
    assert restored.snapshot()["kind"] == "world:action"

    restored.observe_signal("next event", kind="world:notification")
    assert restored.observations == 3


def test_self_latent_restore_rejects_malformed_state_without_mutation() -> None:
    model = SelfLatentSpaceArchitecture()
    model.observe_signal("existing", kind="thought:reflection")
    before = model.to_dict()
    with pytest.raises(ValueError, match="previous vector"):
        model.restore({"version": 1, "observations": 2, "last_state": None, "last_snapshot": {}})
    assert model.to_dict() == before



def test_self_latent_snapshot_cannot_mutate_internal_state() -> None:
    model = SelfLatentSpaceArchitecture()
    model.observe_signal("immutable snapshot", kind="thought:reflection")
    snapshot = model.snapshot()
    snapshot["agency_fiber_states"][0] = 999.0
    assert model.snapshot()["agency_fiber_states"][0] != 999.0
