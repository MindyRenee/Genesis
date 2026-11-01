"""Tests for Genesis's machine-native language latent geometry."""
from __future__ import annotations

import numpy as np
import pytest

from temporal_lobe.comprehension import Comprehension
from temporal_lobe.language_latent_space import (
    LanguageLatentSpace,
    decode_text_cells,
    encode_text_cells,
)


def test_six_dot_text_encoding_is_lossless_for_unicode() -> None:
    text = "Genesis: braille math → language; 你好"
    cells, bit_length = encode_text_cells(text)
    assert cells
    assert all(0 <= cell.value < 64 for cell in cells)
    assert decode_text_cells(cells, bit_length) == text


def test_six_dot_decoder_rejects_bad_bit_lengths() -> None:
    cells, _ = encode_text_cells("hello")
    with pytest.raises(ValueError):
        decode_text_cells(cells, 7)


def test_language_latent_observes_real_parse_and_ot_geometry() -> None:
    parser = Comprehension()
    latent = LanguageLatentSpace()
    first_text = "The cat sits on the mat."
    first = latent.observe(first_text, parser.comprehend(first_text))
    second_text = "Where did the cat go?"
    second = latent.observe(second_text, parser.comprehend(second_text))

    assert first["observations"] == 1
    assert second["observations"] == 2
    assert len(second["latent_state"]) == 32
    assert len(second["cell_distribution"]) == 64
    assert second["transport_plan_shape"][1] == 32
    assert np.isfinite(second["transport_cost"])
    assert second["transport_cost"] >= 0.0
    assert second["transition_residual"] > 0.0
    fisher = np.asarray(second["fisher_information"])
    assert fisher.shape == (32, 32)
    assert np.allclose(fisher, fisher.T, atol=1e-10)
    assert np.linalg.eigvalsh(fisher).min() >= -1e-9
    assert second["regularized_fisher_volume"] > 0.0


def test_language_latent_checkpoint_round_trip_and_isolation() -> None:
    parser = Comprehension()
    latent = LanguageLatentSpace()
    text = "A robot senses light."
    latent.observe(text, parser.comprehend(text))
    checkpoint = latent.to_dict()
    checkpoint["mean"][0] = 99.0
    assert latent.to_dict()["mean"][0] != 99.0

    saved = latent.to_dict()
    restored = LanguageLatentSpace()
    restored.restore(saved)
    assert restored.observations == 1
    assert restored.snapshot() == latent.snapshot()


def test_bad_language_latent_restore_is_atomic() -> None:
    latent = LanguageLatentSpace()
    parser = Comprehension()
    text = "A signal changed."
    latent.observe(text, parser.comprehend(text))
    before = latent.to_dict()
    bad = latent.to_dict()
    bad["m2"] = [[float("nan")]]
    with pytest.raises(ValueError):
        latent.restore(bad)
    assert latent.to_dict() == before


def test_sinkhorn_rejects_invalid_marginals() -> None:
    from temporal_lobe.language_latent_space import _sinkhorn

    with pytest.raises(ValueError):
        _sinkhorn(np.array([0.4, 0.4]), np.array([1.0]), np.zeros((2, 1)))
