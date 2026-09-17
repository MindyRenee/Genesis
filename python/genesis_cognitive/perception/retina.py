"""Retina — zero-copy access to the Rust retina shared-memory camera feed.

This module attaches to the POSIX shared memory segment ``/genesis_retina``
created by the Rust ``retina`` binary and wraps the live frame directly into a
NumPy array without any serialization, JSON, or socket round-trips.

The frame counter uses a seqlock protocol: the writer increments it to an
odd value before writing the frame, then to an even value when the write
is complete. The reader retries if it sees an odd counter or if the
counter changes between the two reads surrounding the frame copy.
"""

from __future__ import annotations

import struct
from multiprocessing import resource_tracker, shared_memory

import numpy as np

NAME = "genesis_retina"
HEADER = 28

# Layout, same as src/bin/retina.rs:
#   u64 counter (seqlock: even = frame complete, odd = write in progress)
#   u32 width
#   u32 height
#   [u8; 4] fourcc
#   u32 size   (total image bytes, may include row padding)
#   u32 stride (bytes per line, may be > width * bytes_per_pixel)
_HEADER_FMT = "<QII4sII"
# Maximum retries before giving up on a consistent read.
_MAX_RETRIES = 64


class Retina:
    """Attach to the live retina shared memory and expose frames as numpy arrays."""

    def __init__(self, name: str = NAME) -> None:
        """Attach to the named retina shared-memory segment without creating it."""
        self._shmem = shared_memory.SharedMemory(name=name, create=False)
        # Prevent the multiprocessing resource tracker from unlinking the
        # persistent shared memory when this Python process exits. The
        # tracker was registered with the POSIX name (leading slash);
        # the public `.name` property strips it on some versions, so
        # try both spellings and tolerate lookup failures — leaking the
        # registration is safe (segment survives), unlinking it is not.
        for candidate in (
            getattr(self._shmem, "_name", None),
            getattr(self._shmem, "name", None),
        ):
            if not candidate:
                continue
            try:
                resource_tracker.unregister(candidate, "shared_memory")
                break
            except (KeyError, AttributeError):
                continue
        self._buf: memoryview | None = self._shmem.buf

    def close(self) -> None:
        """Detach from the shared memory. The underlying segment survives."""
        self._shmem.close()

    def read_header(self) -> tuple[int, int, int, bytes, int, int]:
        """Return counter, width, height, fourcc, size, stride."""
        assert self._buf is not None
        raw = self._buf[:HEADER]
        return struct.unpack(_HEADER_FMT, raw)

    def _read_counter(self) -> int:
        """Read just the frame counter (first 8 bytes)."""
        assert self._buf is not None
        return struct.unpack_from("<Q", self._buf, 0)[0]

    def latest(self, copy: bool = False) -> np.ndarray | None:
        """Return the latest frame as an HWC uint8 numpy array.

        Uses the seqlock protocol to guarantee a consistent read: the
        counter must be even and stable across two reads surrounding the
        frame copy. Returns ``None`` if no consistent frame could be
        read after ``_MAX_RETRIES`` attempts (e.g. the writer is
        producing frames faster than we can read them).

        The array is a zero-copy view of the shared memory unless ``copy=True``.
        The shape always matches the exact byte allocation advertised by the
        Rust retina header.
        """
        assert self._buf is not None
        offset = HEADER

        for _ in range(_MAX_RETRIES):
            # Seqlock read: read counter, check even, read frame, re-read counter.
            c1 = self._read_counter()
            if c1 & 1:
                # Write in progress — retry.
                continue

            # Read the full header (width, height, fourcc, size, stride).
            _counter, width, height, fourcc, size, stride = self.read_header()

            # Snapshot the frame data.
            frame = np.frombuffer(self._buf, dtype=np.uint8, offset=offset, count=size)

            # Re-read the counter. If it changed, the frame was overwritten
            # during our read — retry.
            c2 = self._read_counter()
            if c1 != c2:
                continue

            # Consistent read — shape and convert the frame.
            # Handle row padding: if stride > width * bytes_per_pixel,
            # the driver added padding bytes at the end of each row.
            # We reshape to (height, stride) then slice the valid pixels.
            if fourcc in (b"RGB3", b"BGR3"):
                bpp = 3
                row_bytes = width * bpp
                if stride == row_bytes:
                    frame = frame[:row_bytes * height].reshape((height, width, bpp))
                else:
                    frame = frame[:stride * height].reshape((height, stride))
                    frame = frame[:, :row_bytes].reshape((height, width, bpp))
                if fourcc == b"BGR3":
                    # Convert BGR → RGB in place for downstream color logic.
                    frame = frame[..., ::-1]
            elif fourcc == b"YUYV":
                # YUYV is 2 bytes per pixel, uncompressed.
                bpp = 2
                row_bytes = width * bpp
                if stride == row_bytes:
                    frame = frame[:row_bytes * height].reshape((height, width, bpp))
                else:
                    frame = frame[:stride * height].reshape((height, stride))
                    frame = frame[:, :row_bytes].reshape((height, width, bpp))
            else:
                # For other uncompressed formats, return the raw 1D buffer as-is.
                frame = frame[:size]
            if copy:
                result = np.copy(frame)
                del frame
                return result
            return frame

        return None


def latest_frame(copy: bool = True) -> np.ndarray | None:
    """One-shot: return the latest frame.

    Always returns a copy. The ``copy`` parameter is accepted for
    backward compatibility but a zero-copy view is unsafe in this
    one-shot helper: the ``Retina`` object is local and goes out of
    scope when the function returns, which unmaps the shared memory
    backing the view and leaves the caller with a dangling pointer
    (use-after-unmap). Callers that need a zero-copy view should
    construct a ``Retina`` directly and keep it alive for the
    lifetime of the view.

    Returns ``None`` if no consistent frame could be read (e.g. the
    writer is producing frames faster than we can read them).
    """
    r = Retina()
    try:
        return r.latest(copy=True)
    finally:
        r.close()
