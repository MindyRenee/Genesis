"""Tests for opt-in Android camera frame decoding."""
from __future__ import annotations

import numpy as np
from PIL import Image

from brainstem.android_media import load_rgb_frame


def test_camera_jpeg_decodes_to_detached_rgb_uint8(tmp_path) -> None:
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (3, 2), (20, 80, 140)).save(path, format="JPEG")
    frame = load_rgb_frame(path)
    assert frame.shape == (2, 3, 3)
    assert frame.dtype == np.uint8
    assert frame.flags.owndata
    assert np.allclose(frame[0, 0], [20, 80, 140], atol=3)
