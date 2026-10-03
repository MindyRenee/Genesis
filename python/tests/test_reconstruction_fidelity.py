"""Regression tests for reconstruction fidelity across cognitive transforms."""

import numpy as np

from genesis_cognitive.infrastructure.vq_codebook import VQCodebook
from genesis_cognitive.temporal_lobe.mtl_bridge import MemoryBridge


def test_vq_reports_actual_quantized_reconstruction_and_saturation() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(0.0, 0.03, (24, 6)).astype(np.float32)
    names = [f"c{i}" for i in range(len(vectors))]

    codebook = VQCodebook(dim=6, k=4, residual_scale=0.01)
    stats = codebook.fit(vectors, names, iters=10)

    reconstructed = codebook.reconstruct_all()
    mse = float(np.mean((vectors - reconstructed) ** 2))
    assert codebook.reconstruction_error == mse
    assert stats["final_error"] == mse
    assert 0.0 <= codebook.residual_saturation_fraction <= 1.0
    assert stats["relative_error"] == codebook.relative_reconstruction_error
    assert stats["cosine"] == codebook.reconstruction_cosine


def test_vq_rejects_invalid_training_inputs() -> None:
    codebook = VQCodebook(dim=4, k=2)
    vectors = np.zeros((2, 4), dtype=np.float32)

    try:
        codebook.fit(vectors, ["only_one"])
    except ValueError as exc:
        assert "one entry per vector" in str(exc)
    else:
        raise AssertionError("mismatched concept names must be rejected")


def test_memory_bridge_reports_both_reconstruction_directions() -> None:
    rng = np.random.default_rng(11)
    bridge = MemoryBridge(vtc_dim=4, embedding_dim=6, ridge_lambda=1e-8)

    for i in range(12):
        v = rng.normal(size=4).astype(np.float64)
        e = np.concatenate([v, v[:2]])
        bridge.learn_association(v, f"concept_{i}", e)

    assert bridge.W is not None
    assert bridge.embedding_reconstruction_mse >= 0.0
    assert bridge.embedding_reconstruction_relative_error >= 0.0
    assert -1.0 <= bridge.embedding_reconstruction_cosine <= 1.0
    assert bridge.vtc_cycle_reconstruction_mse >= 0.0
    assert bridge.vtc_cycle_reconstruction_relative_error >= 0.0

    # The diagnostic must describe the same learned projection used by
    # recognize/generate, not an independent approximation.
    V = np.array([ex.vtc_vector for ex in bridge._examples], dtype=np.float64)
    E = np.array(
        [bridge._concept_embeddings[ex.concept_name] for ex in bridge._examples],
        dtype=np.float64,
    )
    predicted = V @ bridge.W.T
    expected_mse = float(np.mean((E - predicted) ** 2))
    assert bridge.embedding_reconstruction_mse == expected_mse


def test_vq_reconstruction_metrics_survive_persistence(tmp_path) -> None:
    rng = np.random.default_rng(19)
    vectors = rng.normal(0.0, 0.05, (12, 5)).astype(np.float32)
    names = [f"p{i}" for i in range(len(vectors))]

    original = VQCodebook(dim=5, k=3, residual_scale=0.02)
    original.fit(vectors, names, iters=5)
    path = tmp_path / "vq.npz"
    original.save(str(path))

    restored = VQCodebook(dim=5, k=3)
    assert restored.load(str(path))
    assert restored.reconstruction_error == original.reconstruction_error
    assert restored.relative_reconstruction_error == original.relative_reconstruction_error
    assert restored.reconstruction_cosine == original.reconstruction_cosine
    assert restored.residual_saturation_fraction == original.residual_saturation_fraction


def test_vq_empty_fit_resets_all_reconstruction_metrics() -> None:
    codebook = VQCodebook(dim=3, k=2)
    vectors = np.ones((4, 3), dtype=np.float32)
    codebook.fit(vectors, ["a", "b", "c", "d"], iters=2)
    codebook.fit(np.empty((0, 3), dtype=np.float32), [])

    assert codebook.reconstruction_error == 0.0
    assert codebook.relative_reconstruction_error == 0.0
    assert codebook.reconstruction_cosine == 0.0
    assert codebook.residual_saturation_fraction == 0.0
