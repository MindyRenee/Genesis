"""Tests for the VQ codebook module."""

import os

import numpy as np
import pytest

from genesis_cognitive.vq_codebook import VQCodebook


@pytest.fixture
def sample_vectors():
    """Generate a small set of test vectors with known cluster structure."""
    rng = np.random.default_rng(42)
    # 3 clusters of 10 points each, 8 dimensions
    centers = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    vectors = []
    names = []
    for ci in range(3):
        for i in range(10):
            v = centers[ci] + rng.normal(0, 0.05, 8).astype(np.float32)
            vectors.append(v)
            names.append(f"concept_{ci}_{i}")
    return np.array(vectors, dtype=np.float32), names


class TestVQCodebook:
    """Tests for v q codebook."""
    def test_fit_and_reconstruct(self, sample_vectors):
        """Codebook should train and reconstruct vectors with low error."""
        vectors, names = sample_vectors
        cb = VQCodebook(dim=8, k=3)
        stats = cb.fit(vectors, names, iters=20)

        assert cb.is_trained
        assert cb.n_concepts == 30
        assert stats["k"] == 3
        assert stats["n"] == 30
        # Reconstruction error should be low (vectors are tightly clustered)
        assert cb.reconstruction_error < 0.01

    def test_reconstruct_individual(self, sample_vectors):
        """Reconstructing a specific concept should return a close vector."""
        vectors, names = sample_vectors
        cb = VQCodebook(dim=8, k=3)
        cb.fit(vectors, names, iters=20)

        recon = cb.reconstruct("concept_0_0")
        assert recon is not None
        original = vectors[0]
        # Should be close to the original
        dist = np.linalg.norm(original - recon)
        assert dist < 0.15

    def test_reconstruct_unknown_concept(self, sample_vectors):
        """Reconstructing an unknown concept should return None."""
        vectors, names = sample_vectors
        cb = VQCodebook(dim=8, k=3)
        cb.fit(vectors, names, iters=20)
        assert cb.reconstruct("nonexistent") is None

    def test_encode_decode_roundtrip(self, sample_vectors):
        """Encoding then decoding should approximately recover the vector."""
        vectors, names = sample_vectors
        cb = VQCodebook(dim=8, k=3)
        cb.fit(vectors, names, iters=20)

        test_vec = vectors[5]
        pid, residual = cb.encode(test_vec)
        recon = cb.decode(pid, residual)
        dist = np.linalg.norm(test_vec - recon)
        assert dist < 0.15

    def test_compression_ratio(self, sample_vectors):
        """Compression ratio should be > 1 (smaller than raw storage)."""
        vectors, names = sample_vectors
        cb = VQCodebook(dim=8, k=3)
        cb.fit(vectors, names, iters=20)
        ratio = cb.compression_ratio()
        assert ratio > 1.0

    def test_merge_candidates(self, sample_vectors):
        """Should find merge candidates within the same cluster."""
        vectors, names = sample_vectors
        cb = VQCodebook(dim=8, k=3, residual_scale=0.01)
        cb.fit(vectors, names, iters=20)

        # With a very low threshold, should find many merge candidates
        candidates = cb.find_merge_candidates(threshold=0.1)
        # Concepts in the same cluster should be merge candidates
        assert len(candidates) > 0
        # All pairs should have distance < threshold
        for _a, _b, dist in candidates:
            assert dist < 0.1

    def test_save_load_roundtrip(self, sample_vectors, tmp_path):
        """Save and load should preserve the codebook."""
        vectors, names = sample_vectors
        cb = VQCodebook(dim=8, k=3)
        cb.fit(vectors, names, iters=20)

        path = str(tmp_path / "vq_test.npz")
        cb.save(path)
        assert os.path.exists(path)

        cb2 = VQCodebook(dim=8, k=3)
        loaded = cb2.load(path)
        assert loaded
        assert cb2.is_trained
        assert cb2.n_concepts == 30
        assert cb2.reconstruction_error == pytest.approx(cb.reconstruction_error, rel=1e-5)

        # Reconstruction should match
        r1 = cb.reconstruct("concept_0_0")
        r2 = cb2.reconstruct("concept_0_0")
        np.testing.assert_array_almost_equal(r1, r2)

    def test_load_nonexistent(self):
        """Loading a nonexistent file should return False."""
        cb = VQCodebook(dim=8, k=3)
        assert not cb.load("/nonexistent/path/file.npz")

    def test_stats(self, sample_vectors):
        """Stats should return a dict with expected keys."""
        vectors, names = sample_vectors
        cb = VQCodebook(dim=8, k=3)
        cb.fit(vectors, names, iters=20)
        stats = cb.stats()
        assert "k" in stats
        assert "n_concepts" in stats
        assert "dim" in stats
        assert "compression_ratio" in stats
        assert "reconstruction_error" in stats

    def test_empty_vectors(self):
        """Fitting with empty vectors should not crash."""
        cb = VQCodebook(dim=4, k=8)
        empty = np.zeros((0, 4), dtype=np.float32)
        stats = cb.fit(empty, [], iters=5)
        # k is clamped to min(k, n) = 0, but shouldn't crash
        assert stats["n"] == 0

    def test_k_larger_than_n(self, sample_vectors):
        """K should be clamped to the number of data points."""
        vectors, names = sample_vectors
        cb = VQCodebook(dim=8, k=100)
        cb.fit(vectors, names, iters=5)
        # K should be clamped to 30 (number of vectors)
        assert cb.prototypes.shape[0] == 30
