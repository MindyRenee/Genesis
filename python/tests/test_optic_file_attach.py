"""The Python optic reader attaches to the Termux frame file when no
POSIX shared-memory segment exists (Android has no /dev/shm namespace)."""

from __future__ import annotations

import struct

import numpy as np

from genesis_client import optic_nerve


def _write_frame(path, width, height, pixels: bytes) -> None:
    # Same 28-byte header the Rust producer writes: seqlock counter (even =
    # complete), width, height, fourcc, size, stride.
    header = struct.pack("<QII4sII", 2, width, height, b"RGB3", len(pixels), width * 3)
    path.write_bytes(header + pixels)


def test_reader_attaches_to_frame_file_when_no_posix_segment(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    name = "genesis_optic_file_attach_test"
    pixels = bytes(range(2 * 2 * 3))
    _write_frame(tmp_path / f"{name}.frame", 2, 2, pixels)

    # No POSIX segment with this name exists, so the reader must fall back.
    nerve = optic_nerve.OpticNerve(name=name)
    try:
        frame = nerve.latest(copy=True)
    finally:
        nerve.close()

    assert frame is not None
    assert frame.shape == (2, 2, 3)
    # RGB3 is packed R,G,B; no channel reversal is needed.
    expected = np.frombuffer(pixels, dtype=np.uint8).reshape(2, 2, 3)
    assert np.array_equal(frame, expected)


def test_missing_segment_and_file_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    try:
        optic_nerve.OpticNerve(name="genesis_optic_no_such_segment")
    except (OSError, FileNotFoundError):
        return
    raise AssertionError("attaching to nothing should fail")
