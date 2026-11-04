"""Tests for the shared optic-nerve frame contract."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from genesis_client.optic_nerve import HEADER, OpticNerve


@pytest.mark.parametrize(
    ("fourcc", "payload", "expected"),
    [
        (b"RGB3", bytes((255, 0, 0)), (255, 0, 0)),
        (b"BGR3", bytes((0, 0, 255)), (255, 0, 0)),
    ],
)
def test_latest_normalizes_rgb_fourcc_channel_order(
    tmp_path,
    monkeypatch,
    fourcc: bytes,
    payload: bytes,
    expected: tuple[int, int, int],
) -> None:
    """Both producer formats reach perception as RGB, never swapped."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    name = "genesis_optic_nerve_test_channels"
    frame_path = tmp_path / f"{name}.frame"
    frame_path.write_bytes(struct.pack("<QII4sII", 2, 1, 1, fourcc, len(payload), 3) + payload)

    nerve = OpticNerve(name)
    try:
        frame = nerve.latest(copy=True)
    finally:
        nerve.close()

    assert frame is not None
    assert frame.shape == (1, 1, 3)
    assert tuple(int(channel) for channel in frame[0, 0]) == expected
    assert frame.dtype == np.uint8
    assert frame.nbytes == 3
    assert HEADER == 28
