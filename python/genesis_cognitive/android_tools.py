"""Android/Termux command discovery helpers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

_TERMUX_BIN_CANDIDATES = (
    Path("/data/data/com.termux/files/usr/bin"),
    Path("/data/data/com.termux/files/home/.termux/bin"),
)


def find_termux_command(name: str) -> str | None:
    """Find a Termux:API command from native Termux or a proot Linux userland."""
    command = shutil.which(name)
    if command:
        return command

    prefix = os.environ.get("TERMUX_PREFIX", "").strip()
    if prefix:
        candidate = Path(prefix) / "bin" / name
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return str(candidate)

    for directory in _TERMUX_BIN_CANDIDATES:
        candidate = directory / name
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return str(candidate)

    return None
