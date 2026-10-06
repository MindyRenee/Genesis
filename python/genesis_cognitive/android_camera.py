"""Android camera adapter."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from genesis_cognitive.android_tools import find_termux_command

_COMMAND = "termux-camera-photo"
_DEFAULT_CAMERA = "0"
_TIMEOUT_SECONDS = 20


def _camera_command() -> str | None:
    return find_termux_command(_COMMAND)


def capture_android_camera(data_dir: str | os.PathLike[str]) -> str | None:
    """Capture one Android camera frame through Termux:API."""
    command = _camera_command()
    if command is None:
        return None
    camera_id = os.environ.get("GENESIS_ANDROID_CAMERA_ID", _DEFAULT_CAMERA).strip()
    if not camera_id.isdigit():
        raise ValueError("GENESIS_ANDROID_CAMERA_ID must be a numeric camera id")
    output_dir = Path(data_dir) / "android_camera"
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = output_dir / "latest.jpg"
    temporary = output_dir / f".capture-{os.getpid()}-{time.monotonic_ns()}.jpg"
    try:
        completed = subprocess.run(
            [command, "-c", camera_id, str(temporary)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0 or not temporary.is_file():
            return None
        os.replace(temporary, output)
        return str(output)
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def android_camera_available() -> bool:
    return _camera_command() is not None
