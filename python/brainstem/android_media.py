"""Android microphone capture through Termux:API.

Visual input is owned by the daemon's optic nerve; this module contains only
the separate audio-device adapter.
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

logger = logging.getLogger(__name__)


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
        if not is_android_termux_microphone() or _shared_download_dir() is None:
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
        shared_dir = _shared_download_dir()
        if shared_dir is None:
            logger.warning(
                "Termux microphone requires termux-setup-storage and shared Download access"
            )
            return
        while not self._stop.is_set():
            started = time.monotonic()
            descriptor, filename = tempfile.mkstemp(
                prefix="genesis-audio-", suffix=".m4a", dir=shared_dir
            )
            os.close(descriptor)
            audio_path = Path(filename)
            audio_path.unlink(missing_ok=True)
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
            finally:
                audio_path.unlink(missing_ok=True)
            remaining = self._interval - (time.monotonic() - started)
            self._stop.wait(max(0.0, remaining))
