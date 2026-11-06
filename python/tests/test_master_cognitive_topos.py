"""Tests for the unified cognitive-state transport engine."""

from __future__ import annotations

import numpy as np
import pytest
from self.master_cognitive_topos import (
    EhresmannConnectionTransport,
    KuramotoRiemannianSynchronizer,
    MasterCognitiveToposEngine,
    MasterFreeEnergyOptimizer,
)


def test_transport_changes_dimension_and_normalizes_nonzero_rows() -> None:
    transport = EhresmannConnectionTransport(3, 2, np.array([[1., 0., 0.], [0., 1., 0.]]))
    result = transport.forward([[3., 4., 0.], [0., 0., 0.]])
    assert result.shape == (2, 2)
    assert np.allclose(np.linalg.norm(result[0]), 1.0)
    assert np.allclose(result[1], 0.0)


def test_transport_rejects_incompatible_connection() -> None:
    with pytest.raises(ValueError, match="shape"):
        EhresmannConnectionTransport(3, 2, np.zeros((3, 2)))


def test_free_energy_is_finite_nonnegative_and_coupled() -> None:
    objective = MasterFreeEnergyOptimizer()
    total, penalty = objective.forward([1., 2.], [2., 2.], [1., 1.], [1., 1.])
    assert total > 0
    assert penalty >= 0
    assert np.isfinite(total)
    with pytest.raises(ValueError, match="non-negative"):
        objective.forward([-1.], [1.], [1.], [1.])



def test_free_energy_alignment_penalty_does_not_cancel_across_batch() -> None:
    objective = MasterFreeEnergyOptimizer(lambda_1=1.0, lambda_2=0.0)
    # Equal and opposite per-sample errors have zero signed mean but nonzero
    # mismatch; the objective must not misreport them as aligned.
    total, penalty = objective.forward(
        [1.0, 3.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [3.0, 1.0],
    )
    assert penalty == pytest.approx(2.0)
    assert total > 0

def test_kuramoto_uses_theta_j_minus_theta_i_sign() -> None:
    sync = KuramotoRiemannianSynchronizer(2, np.zeros(2))
    phases = np.array([[0.0, np.pi / 2]])
    weights = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    result = sync.forward(phases, weights, K=1.0, dt=0.1)
    assert result[0, 0] > phases[0, 0]
    assert result[0, 1] < phases[0, 1]


def test_engine_integrates_four_domains_and_phases() -> None:
    rng = np.random.default_rng(7)
    batch, dim = 4, 8
    engine = MasterCognitiveToposEngine(dim)
    states = [rng.normal(size=(batch, dim)) for _ in range(4)]
    phases = rng.normal(size=(batch, dim))
    weights = rng.uniform(size=(batch, dim, dim))
    output = engine.forward(*states, phases, weights)
    assert output["transported_macro_state"].shape == (batch, dim)
    assert output["synchronized_phases"].shape == (batch, dim)
    assert np.isfinite(output["master_free_energy"])
    assert np.isfinite(output["coupling_constraint_penalty"])
    for key in (
        "transported_macro_state",
        "transported_outer_to_inner",
        "transported_inner_to_self",
        "transported_self_to_language",
        "synchronized_phases",
    ):
        assert np.isfinite(output[key]).all()


def test_engine_rejects_batch_mismatch_and_bad_metric_shape() -> None:
    engine = MasterCognitiveToposEngine(3)
    states = [np.zeros((2, 3)) for _ in range(4)]
    with pytest.raises(ValueError, match="batch size"):
        engine.forward(
            states[0], states[1], states[2], states[3],
            np.zeros((1, 3)), np.zeros((2, 3, 3)),
        )
    with pytest.raises(ValueError, match="metric_weights"):
        engine.forward(
            *states, np.zeros((2, 3)), np.zeros((2, 3, 2)),
        )
