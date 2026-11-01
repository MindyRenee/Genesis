"""Tests for opt-in Android camera frame decoding."""
from __future__ import annotations

from PIL import Image

import numpy as np

from brainstem.android_media import capture_android_frame, decode_float32_audio, load_rgb_frame


def test_camera_jpeg_decodes_to_detached_rgb_uint8(tmp_path) -> None:
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (3, 2), (20, 80, 140)).save(path, format="JPEG")
    frame = load_rgb_frame(path)
    assert frame.shape == (2, 3, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.owndata
    assert np.allclose(frame[0, 0], [20, 80, 140], atol=3)



def test_microphone_pcm_decoder_rejects_partial_and_nonfinite_samples() -> None:
    samples = decode_float32_audio(np.array([0.25, -0.5], dtype="<f4").tobytes())
    assert np.allclose(samples, [0.25, -0.5])
    import pytest

    with pytest.raises(ValueError):
        decode_float32_audio(b"abc")
    with pytest.raises(ValueError):
        decode_float32_audio(np.array([np.nan], dtype="<f4").tobytes())


def test_float32_pcm_decoder_rejects_partial_and_nonfinite_samples() -> None:
    samples = decode_float32_audio(np.array([0.25, -0.5], dtype="<f4").tobytes())
    assert np.allclose(samples, [0.25, -0.5])
    import pytest

    with pytest.raises(ValueError):
        decode_float32_audio(b"abc")
    with pytest.raises(ValueError):
        decode_float32_audio(np.array([np.nan], dtype="<f4").tobytes())



def test_on_demand_camera_capture_is_absent_outside_android(monkeypatch) -> None:
    monkeypatch.delenv("ANDROID_ROOT", raising=False)
    monkeypatch.delenv("ANDROID_DATA", raising=False)
    assert capture_android_frame() is None
