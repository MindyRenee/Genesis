"""Tests for V4 sparse coding and reconstruction fidelity."""

import numpy as np

from genesis_cognitive.occipital_lobe.v4 import V4Model


def test_gram_cache_is_immediately_consistent_after_dictionary_update():
    model = V4Model(
        n_v1_features=2,
        pool_size=1,
        n_features=3,
        n_ista_iters=5,
        seed=1,
    )
    X = np.ones((4, model.input_dim), dtype=np.float64)
    Z = np.full((4, model.n_features), 0.1, dtype=np.float64)

    model._update_dictionary(X, Z)

    np.testing.assert_allclose(model._phi4_gram, model.phi4.T @ model.phi4)
    assert model._effective_ista_step() <= 0.99 / model._ista_lipschitz + 1e-15


def test_reconstruction_metrics_match_returned_code_and_dictionary():
    model = V4Model(
        n_v1_features=4,
        pool_size=1,
        n_features=6,
        sparsity=0.01,
        n_ista_iters=50,
        seed=2,
    )
    v1 = np.random.default_rng(3).random((4, 4))

    z = model.process(v1, (2, 2), learn=False)

    X, recon = model._pool_v1(v1, 2, 2)[0], None
    # process() includes color padding, so reconstruct against the actual
    # input dimensionality used by V4.
    X = np.hstack([X, np.zeros((X.shape[0], model.color_dims))])
    recon = z @ model.phi4.T
    error = X - recon
    expected_mse = np.mean(error * error)
    expected_relative = np.sum(error * error) / max(np.sum(X * X), 1e-12)

    np.testing.assert_allclose(model.reconstruction_mse, expected_mse)
    np.testing.assert_allclose(
        model.relative_reconstruction_error, expected_relative
    )
    assert np.isfinite(model.reconstruction_cosine)
    assert 0.0 <= model.sparsity_fraction <= 1.0


def test_learning_returns_code_for_current_dictionary():
    model = V4Model(
        n_v1_features=2,
        pool_size=1,
        n_features=3,
        sparsity=0.01,
        n_ista_iters=20,
        seed=4,
    )
    v1 = np.random.default_rng(5).random((4, 2))

    z = model.process(v1, (2, 2), learn=True)

    pooled, _ = model._pool_v1(v1, 2, 2)
    X = np.hstack([pooled, np.zeros((pooled.shape[0], model.color_dims))])
    np.testing.assert_allclose(z, model._infer_latents(X))
    np.testing.assert_allclose(model._phi4_gram, model.phi4.T @ model.phi4)
