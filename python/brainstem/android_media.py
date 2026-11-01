"""Opt-in Android camera capture through Termux:API.

The camera is sampled only when GENESIS_ANDROID_CAMERA=1. Every frame is a
real JPEG returned by Termux:API and is passed to Genesis's existing visual
pipeline. Missing commands, permissions, or hardware mean no frame is emitted.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

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


def capture_android_frame(camera_id: int = 0, timeout: float = 15.0) -> np.ndarray | None:
    """Capture one on-demand still frame through Termux:API, or return None."""
    if isinstance(camera_id, bool) or not isinstance(camera_id, int) or camera_id < 0:
        raise ValueError("camera_id must be a non-negative integer")
    if timeout < 2.0:
        raise ValueError("camera timeout must be at least 2 seconds")
    if not is_android_termux_camera():
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="genesis-camera-once-") as directory:
            image_path = Path(directory) / "frame.jpg"
            result = subprocess.run(
                ["termux-camera-photo", "-c", str(camera_id), str(image_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode != 0 or not image_path.is_file():
                return None
            return load_rgb_frame(image_path)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class AndroidCameraAdapter:
    """Low-rate still-frame producer for Genesis's shared vision pipeline."""

    def __init__(
        self,
        on_frame: Callable[[np.ndarray], None],
        *,
        interval_seconds: float = 10.0,
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
                    if not self._stop.is_set() and result.returncode == 0 and image_path.is_file():
                        frame = load_rgb_frame(image_path)
                        if frame.size:
                            try:
                                self._on_frame(frame)
                            except Exception:
                                logger.exception("Android camera callback failed")
                        image_path.unlink(missing_ok=True)
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    now = time.monotonic()
                    if now - self._last_error_log >= 60.0:
                        logger.warning("Termux camera capture unavailable; retrying: %s", error)
                        self._last_error_log = now
                remaining = self._interval - (time.monotonic() - started)
                self._stop.wait(max(0.0, remaining))



def decode_float32_audio(data: bytes) -> np.ndarray:
    """Decode mono little-endian float32 PCM emitted by ffmpeg."""
    if not data or len(data) % 4:
        raise ValueError("audio PCM must contain complete float32 samples")
    samples = np.frombuffer(data, dtype="<f4").copy()
    if samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("audio PCM must contain finite samples")
    return np.clip(samples, -1.0, 1.0)


def is_android_termux_microphone() -> bool:
    """Detect Android plus the recording and decoding commands."""
    return (
        bool(os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"))
        and shutil.which("termux-microphone-record") is not None
        and shutil.which("ffmpeg") is not None
    )


class AndroidMicrophoneAdapter:
    """Opt-in short-clip microphone input using Termux:API and ffmpeg."""

    def __init__(
        self,
        on_clip: Callable[[np.ndarray, int], None],
        *,
        interval_seconds: float = 5.0,
        clip_seconds: int = 1,
        sample_rate: int = 16000,
        command_timeout: float = 8.0,
    ) -> None:
        if not callable(on_clip):
            raise TypeError("on_clip must be callable")
        if interval_seconds < 2.0:
            raise ValueError("microphone interval must be at least 2 seconds")
        if clip_seconds not in {1, 2, 3, 4, 5}:
            raise ValueError("clip_seconds must be between 1 and 5")
        if sample_rate < 8000:
            raise ValueError("sample_rate must be at least 8000 Hz")
        self._on_clip = on_clip
        self._interval = float(interval_seconds)
        self._clip_seconds = clip_seconds
        self._sample_rate = sample_rate
        self._timeout = float(command_timeout)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_error_log = 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start only when both recording and decoding tools are available."""
        if self.running:
            return True
        if not is_android_termux_microphone():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="genesis-android-microphone",
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
        if is_android_termux_microphone():
            try:
                subprocess.run(
                    ["termux-microphone-record", "-q"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=min(self._timeout, 3.0),
                )
            except (OSError, subprocess.SubprocessError):
                pass

    def _run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="genesis-microphone-") as directory:
            audio_path = Path(directory) / "clip.m4a"
            while not self._stop.is_set():
                started = time.monotonic()
                try:
                    subprocess.run(
                        [
                            "termux-microphone-record",
                            "-f",
                            str(audio_path),
                            "-l",
                            str(self._clip_seconds),
                            "-e",
                            "aac",
                            "-r",
                            str(self._sample_rate),
                            "-c",
                            "1",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=self._timeout,
                    )
                    deadline = time.monotonic() + self._timeout
                    while not self._stop.is_set() and time.monotonic() < deadline:
                        info = subprocess.run(
                            ["termux-microphone-record", "-i"],
                            check=True,
                            capture_output=True,
                            text=True,
                            timeout=min(self._timeout, 3.0),
                        )
                        try:
                            status = json.loads(info.stdout)
                        except json.JSONDecodeError:
                            status = {}
                        if isinstance(status, dict) and status.get("isRecording") is False:
                            break
                        self._stop.wait(0.2)
                    else:
                        raise TimeoutError("Termux microphone recording did not finish")
                    if not audio_path.is_file() or audio_path.stat().st_size == 0:
                        raise OSError("Termux microphone produced no audio file")
                    decoded = subprocess.run(
                        [
                            "ffmpeg",
                            "-nostdin",
                            "-v",
                            "error",
                            "-i",
                            str(audio_path),
                            "-f",
                            "f32le",
                            "-acodec",
                            "pcm_f32le",
                            "-ac",
                            "1",
                            "-ar",
                            str(self._sample_rate),
                            "pipe:1",
                        ],
                        check=True,
                        capture_output=True,
                        timeout=self._timeout,
                    )
                    samples = decode_float32_audio(decoded.stdout)
                    if self._stop.is_set():
                        break
                    try:
                        self._on_clip(samples, self._sample_rate)
                    except Exception:
                        logger.exception("Android microphone callback failed")
                    audio_path.unlink(missing_ok=True)
                except (OSError, subprocess.SubprocessError, ValueError, TimeoutError) as error:
                    now = time.monotonic()
                    if now - self._last_error_log >= 60.0:
                        logger.warning("Termux microphone capture unavailable; retrying: %s", error)
                        self._last_error_log = now
                remaining = self._interval - (time.monotonic() - started)
                self._stop.wait(max(0.0, remaining))
