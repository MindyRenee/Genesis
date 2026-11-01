"""Opt-in Android camera capture through Termux:API.

The camera is sampled only when GENESIS_ANDROID_CAMERA=1. Every frame is a
real JPEG returned by Termux:API and is passed to Genesis's existing visual
pipeline. Missing commands, permissions, or hardware mean no frame is emitted.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def is_android_termux_camera() -> bool:
    """Detect Android plus the installed Termux camera command."""
    return bool(os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA")) and (
        shutil.which("termux-camera-photo") is not None
    )


def load_rgb_frame(path: str | Path) -> np.ndarray:
    """Decode a captured image into a detached uint8 RGB array."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


class AndroidCameraAdapter:
    """Low-rate still-frame producer for Genesis's shared vision pipeline."""

    def __init__(
        self,
        on_frame: Callable[[np.ndarray], None],
        *,
        interval_seconds: float = 5.0,
        command_timeout: float = 15.0,
        camera_id: int = 0,
    ) -> None:
        if not callable(on_frame):
            raise TypeError("on_frame must be callable")
        if interval_seconds < 2.0:
            raise ValueError("camera interval must be at least 2 seconds")
        if command_timeout < 2.0:
            raise ValueError("camera command timeout must be at least 2 seconds")
        if isinstance(camera_id, bool) or not isinstance(camera_id, int) or camera_id < 0:
            raise ValueError("camera_id must be a non-negative integer")
        self._on_frame = on_frame
        self._interval = float(interval_seconds)
        self._timeout = float(command_timeout)
        self._camera_id = camera_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error_log = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start only when the device and Termux camera command are present."""
        if self.running:
            return True
        if not is_android_termux_camera():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="genesis-android-camera",
            daemon=True,
        )
        self._thread.start()
        return True

    def close(self, timeout: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, timeout))
        self._thread = None

    def _run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="genesis-camera-") as directory:
            image_path = Path(directory) / "frame.jpg"
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    result = subprocess.run(
                        ["termux-camera-photo", "-c", str(self._camera_id), str(image_path)],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=self._timeout,
                    )
                    if result.returncode == 0 and image_path.is_file():
                        frame = load_rgb_frame(image_path)
                        if frame.size:
                            try:
                                self._on_frame(frame)
                            except Exception:  # noqa: BLE001 — camera callback is isolated
                                logger.exception("Android camera callback failed")
                        image_path.unlink(missing_ok=True)
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    now = time.monotonic()
                    if now - self._last_error_log >= 60.0:
                        logger.warning("Termux camera capture unavailable; retrying: %s", error)
                        self._last_error_log = now
                remaining = self._interval - (time.monotonic() - started)
                self._stop.wait(max(0.0, remaining))
