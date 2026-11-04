"""Tests for the native Android camera stream receiver."""

from __future__ import annotations

import time

import numpy as np
from occipital_lobe.perception import android_stream


def test_latest_stream_frame_returns_copy_of_fresh_frame() -> None:
    android_stream._reset_stream_state_for_tests()
    frame = np.zeros((3, 4, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    with android_stream._frame_lock:
        android_stream._latest_frame = frame
        android_stream._latest_at = time.monotonic()

    received = android_stream.latest_stream_frame()
    assert received is not None
    assert received.shape == (3, 4, 3)
    assert received.dtype == np.uint8
    received[0, 0] = (9, 9, 9)
    assert android_stream.latest_stream_frame()[0, 0].tolist() == [1, 2, 3]


def test_latest_stream_frame_rejects_stale_frame() -> None:
    android_stream._reset_stream_state_for_tests()
    with android_stream._frame_lock:
        android_stream._latest_frame = np.zeros((1, 1, 3), dtype=np.uint8)
        android_stream._latest_at = time.monotonic() - 10

    assert android_stream.latest_stream_frame(max_age_seconds=1.0) is None


def test_latest_stream_frame_is_unavailable_before_first_frame() -> None:
    android_stream._reset_stream_state_for_tests()
    assert android_stream.latest_stream_frame() is None
