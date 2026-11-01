"""Tests for the NumPy inner-world experiential state model."""
import numpy as np
import pytest
from sleep.inner_world import (
    AffectiveFiberQualiaSpace,
    InnerWorldExperientialLatentSpace,
    InteroceptiveBaseManifold,
    RecursiveMetaCognitiveSymplecticFlow,
    UltrametricEpisodicMemorySpace,
    VariationalFreeEnergyGeometry,
)


def _signals(seed: float = 0.0) -> np.ndarray:
    return np.linspace(-0.8 + seed, 0.8 + seed, 8)


def test_homeostatic_metric_is_positive_definite_and_deviation_sensitive():
    manifold = InteroceptiveBaseManifold()
    baseline, baseline_scale = manifold.forward(np.zeros(4))
    drifted, drifted_scale = manifold.forward(np.array([0.5, 0.0, 0.0, 0.0]))
    assert baseline_scale == 1.0
    assert drifted_scale > baseline_scale
    assert np.all(np.linalg.eigvalsh(baseline) > 0.0)
    assert np.all(np.linalg.eigvalsh(drifted) > 0.0)


def test_affective_fiber_has_40_finite_coordinates_and_unit_circle_factors():
    fiber = AffectiveFiberQualiaSpace().forward(_signals())
    assert fiber.shape == (40,)
    assert np.all(np.isfinite(fiber))
    torus = fiber[32:].reshape(4, 2)
    np.testing.assert_allclose(np.sum(torus * torus, axis=1), 1.0, atol=1e-12)


def test_ultrametric_memory_satisfies_strong_triangle_inequality():
    memory = UltrametricEpisodicMemorySpace(levels=8)
    vectors = [
        AffectiveFiberQualiaSpace().forward(_signals(seed))
        for seed in (0.0, 0.01, 0.2, 0.6)
    ]
    for x in vectors:
        for y in vectors:
            for z in vectors:
                d_xz = memory.distance(x, z)
                d_xy = memory.distance(x, y)
                d_yz = memory.distance(y, z)
                assert d_xz <= max(d_xy, d_yz) + 1e-15


def test_memory_nearest_and_serialization_round_trip():
    encoder = AffectiveFiberQualiaSpace()
    memory = UltrametricEpisodicMemorySpace(capacity=3)
    a = encoder.forward(_signals())
    b = encoder.forward(_signals(0.1))
    memory.add(a, "first")
    memory.add(b, "second")
    assert memory.nearest(a)[0]["label"] == "first"
    restored = UltrametricEpisodicMemorySpace()
    restored.restore(memory.to_dict())
    assert restored.size == 2
    assert restored.nearest(a)[0]["label"] == "first"


def test_kl_geometry_is_finite_positive_definite_and_nonnegative():
    fiber = AffectiveFiberQualiaSpace().forward(_signals())
    metric, free_energy, mean, variance = VariationalFreeEnergyGeometry().forward(fiber)
    assert metric.shape == (40, 40)
    assert np.all(np.linalg.eigvalsh(metric) > 0.0)
    assert np.all(np.isfinite(mean))
    assert np.all(variance > 0.0)
    assert np.isfinite(free_energy)
    assert free_energy >= 0.0


def test_hamiltonian_flow_preserves_phase_space_norm():
    state = AffectiveFiberQualiaSpace().forward(_signals())
    evolved = RecursiveMetaCognitiveSymplecticFlow().forward(state)
    np.testing.assert_allclose(
        np.linalg.norm(evolved),
        np.linalg.norm(state),
        rtol=1e-12,
        atol=1e-12,
    )


def test_integrated_inner_world_updates_memory_and_restores_state():
    world = InnerWorldExperientialLatentSpace(memory_capacity=4)
    first = world.observe([0.0, 0.5, 0.0, 0.5], _signals(), "thought-a")
    second = world.observe([0.2, 0.6, -0.1, 0.4], _signals(0.1), "thought-b")
    assert first["observations"] == 1
    assert second["observations"] == 2
    assert second["ultrametric_memory_nodes"]["size"] == 2
    assert len(second["affective_fiber_states"]) == 40
    assert len(second["recursive_meta_states"]) == 40
    restored = InnerWorldExperientialLatentSpace()
    restored.restore(world.to_dict())
    assert restored.observations == 2
    assert restored.ultrametric_memory.size == 2


def test_invalid_shapes_and_nonfinite_inputs_fail_closed():
    with pytest.raises(ValueError):
        InteroceptiveBaseManifold().forward([0.0, 0.0])
    with pytest.raises(ValueError):
        AffectiveFiberQualiaSpace().forward([0.0] * 7)
    with pytest.raises(ValueError):
        AffectiveFiberQualiaSpace().forward([0.0] * 7 + [float("nan")])
