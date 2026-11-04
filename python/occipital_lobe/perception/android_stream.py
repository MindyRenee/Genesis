"""Receive continuous RGB frames from the native Android camera bridge.

The companion app sends length-prefixed JPEG frames over loopback TCP.
This module keeps only the newest frame: perception consumes current visual
input rather than building an unbounded queue of stale images.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)
_HOST = "127.0.0.1"
_PORT = 8765
_MAX_FRAME_BYTES = 8 * 1024 * 1024
_server_lock = threading.Lock()
_server_started = False
_frame_lock = threading.Lock()
_latest_frame: np.ndarray | None = None
_latest_at = 0.0


def _read_exact(connection: socket.socket, size: int) -> bytes | None:
    """Read exactly size bytes, returning None if the peer disconnects."""
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _serve() -> None:
    """Accept the companion app and retain only its most recent JPEG frame."""
    global _latest_frame, _latest_at
    try:
        import io

        from PIL import Image
    except ImportError:
        logger.warning("Android camera stream requires Pillow")
        return

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((_HOST, _PORT))
            server.listen(1)
            server.settimeout(1.0)
            logger.info("Android camera frame receiver listening on %s:%d", _HOST, _PORT)
            while True:
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    continue
                with connection:
                    connection.settimeout(5.0)
                    while True:
                        try:
                            header = _read_exact(connection, 4)
                            if header is None:
                                break
                            size = struct.unpack(">I", header)[0]
                            if size <= 0 or size > _MAX_FRAME_BYTES:
                                logger.warning("Rejected Android camera frame size %d", size)
                                break
                            payload = _read_exact(connection, size)
                            if payload is None:
                                break
                            with Image.open(io.BytesIO(payload)) as image:
                                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
                            with _frame_lock:
                                _latest_frame = rgb
                                _latest_at = time.monotonic()
                        except (OSError, ValueError, RuntimeError) as exc:
                            logger.debug("Android camera stream disconnected: %s", exc)
                            break
    except OSError as exc:
        logger.warning("Could not start Android camera receiver: %s", exc)


def ensure_stream_server() -> bool:
    """Start the receiver once; returns whether its listener thread was started."""
    global _server_started
    with _server_lock:
        if _server_started:
            return True
        thread = threading.Thread(
            target=_serve,
            name="genesis-android-camera",
            daemon=True,
        )
        thread.start()
        _server_started = True
        return True


def latest_stream_frame(max_age_seconds: float = 2.0) -> np.ndarray | None:
    """Return a copy of the newest fresh frame, or None if the stream is stale."""
    with _frame_lock:
        if _latest_frame is None or time.monotonic() - _latest_at > max_age_seconds:
            return None
        return _latest_frame.copy()


def _reset_stream_state_for_tests() -> None:
    """Reset process-local stream state for isolated tests."""
    global _latest_frame, _latest_at
    with _frame_lock:
        _latest_frame = None
        _latest_at = 0.0
