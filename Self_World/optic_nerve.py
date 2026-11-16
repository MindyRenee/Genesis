"""Optic nerve — the mind's end of the afferent path.

The body's optic nerve (``src/daemon/optic_nerve.rs`` — the daemon owns
the camera now, a capture thread inside the body rather than a separate
process) writes raw frames into the POSIX shared-memory segment
``/genesis_optic_nerve``. This module attaches to that segment and wraps
the live frame directly into a NumPy array without any serialization,
JSON, or socket round-trips.

The frame counter is a plain sequence number: the writer bumps it once
per published frame, after the frame bytes land. A read that races a
writer may observe a torn frame — the consumers pool statistics over
subsampled pixels, so a transient artifact averages out.

This is the afferent wire only — it delivers pixels. What the pixels
*mean* lives in the visual cortex (``occipital_lobe/perception/vision.py``),
which is the words side.
"""

from __future__ import annotations

import struct
from multiprocessing import resource_tracker, shared_memory

import numpy as np

NAME = "genesis_optic_nerve"
# The screen-capture nerve — a second afferent segment fed by
# ``scripts/optic_relay.py`` (an x11grab of the display, or any relayed
# source), carrying the same 28-byte header + frame counter. The
# body's subcortical channels stay on the primary nerve; this one is
# cortical only — it is how cortex sees what the world shows her.
SCREEN_NAME = "genesis_screen_nerve"
HEADER = 28

# Layout, same as src/daemon/optic_nerve.rs:
#   u64 counter (frame sequence number: 0 = no frame published yet)
#   u32 width
#   u32 height
#   [u8; 4] fourcc
#   u32 size   (total image bytes, may include row padding)
#   u32 stride (bytes per line, may be > width * bytes_per_pixel)
_HEADER_FMT = "<QII4sII"


class OpticNerve:
    """Attach to the live optic-nerve segment and expose frames as numpy arrays."""

    def __init__(self, name: str = NAME) -> None:
        """Attach to the named optic-nerve shared-memory segment without creating it."""
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
        # Release our own memoryview first: any slice or np.frombuffer
        # view still alive holds an exported pointer on the segment and
        # SharedMemory.close() raises BufferError while exports exist.
        if self._buf is not None:
            self._buf.release()
            self._buf = None
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

        A single snapshot: read the frame counter (0 means no frame has
        ever been published → ``None``), then the header and payload in
        one pass. There is no write-in-progress marker, so a read that
        races a writer may return a frame composited from two captures —
        the consumers treat pixels as statistics, not truth.

        The array is a zero-copy view of the shared memory unless ``copy=True``.
        The shape always matches the exact byte allocation advertised by the
        nerve's header.
        """
        assert self._buf is not None
        offset = HEADER

        c1 = self._read_counter()
        if c1 == 0:
            # The segment exists but no frame has ever been published.
            return None

        # Read the full header (width, height, fourcc, size, stride).
        _counter, width, height, fourcc, size, stride = self.read_header()

        # Snapshot the frame data.
        frame = np.frombuffer(self._buf, dtype=np.uint8, offset=offset, count=size)

        # Shape and convert the frame.
        # Handle row padding: if stride > width * bytes_per_pixel,
        # the driver added padding bytes at the end of each row.
        # We reshape to (height, stride) then slice the valid pixels.
        if fourcc in (b"RGB3", b"BGR3"):
            bpp = 3
            row_bytes = width * bpp
            if stride == row_bytes:
                frame = frame[: row_bytes * height].reshape((height, width, bpp))
            else:
                frame = frame[: stride * height].reshape((height, stride))
                frame = frame[:, :row_bytes].reshape((height, width, bpp))
            if fourcc == b"RGB3":
                # V4L2's RGB3 stores B,G,R in memory (the fourcc
                # names word order, not byte order); BGR3 is the
                # one already in R,G,B byte order. Flip RGB3's
                # memory order so downstream sees RGB.
                frame = frame[..., ::-1]
        elif fourcc == b"YUYV":
            # YUYV is 2 bytes per pixel, uncompressed.
            bpp = 2
            row_bytes = width * bpp
            if stride == row_bytes:
                frame = frame[: row_bytes * height].reshape((height, width, bpp))
            else:
                frame = frame[: stride * height].reshape((height, stride))
                frame = frame[:, :row_bytes].reshape((height, width, bpp))
        else:
            # For other uncompressed formats, return the raw 1D buffer as-is.
            frame = frame[:size]
        if copy:
            result = np.copy(frame)
            del frame
            return result
        return frame


def latest_frame(copy: bool = True, name: str = NAME) -> np.ndarray | None:
    """One-shot: return the latest frame.

    Always returns a copy. The ``copy`` parameter is accepted for
    backward compatibility but a zero-copy view is unsafe in this
    one-shot helper: the ``OpticNerve`` object is local and goes out of
    scope when the function returns, which unmaps the shared memory
    backing the view and leaves the caller with a dangling pointer
    (use-after-unmap). Callers that need a zero-copy view should
    construct an ``OpticNerve`` directly and keep it alive for the
    lifetime of the view.

    ``name`` selects the nerve segment — the default is the body's own
    eye; ``SCREEN_NAME`` reads the screen-capture nerve when a relay
    feeds it.

    Returns ``None`` if the nerve segment is absent or no frame has
    ever been published.
    """
    try:
        nerve = OpticNerve(name)
    except FileNotFoundError:
        # No nerve segment — the body's eye is dark (camera absent or
        # the capture limb between re-open attempts). That is the
        # documented text-only degradation: no frame, not a fault.
        return None
    try:
        return nerve.latest(copy=True)
    finally:
        nerve.close()
