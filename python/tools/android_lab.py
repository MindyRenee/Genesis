"""Android — the phone-side reach, through adb when it exists.

Her relay limbs (device.vibrate etc.) fire through the daemon's
actuate; this is the *inspection* side: device list, shell, package
census. Absent adb reports absence — never a fabricated device.

Docs: https://developer.android.com/tools/adb
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Any

from tools.framework import ToolResult


def adb(
    action: str = "devices",
    command: str = "",
    timeout: int = 30,
    **_kwargs: Any,
) -> ToolResult:
    """Android Debug Bridge reach — devices, shell, packages.

    ``action``: devices (connected devices + states), shell
    (command= runs on-device via `adb shell`), packages (installed
    package names, n= limit), getprop (system properties).
    Requires adb installed and a device authorized; absence reports
    honestly. https://developer.android.com/tools/adb
    """
    exe = shutil.which("adb")
    if exe is None:
        return ToolResult(
            success=False,
            error="adb not installed on this host (android-tools package)",
        )
    timeout = max(1, min(timeout, 300))
    if action == "devices":
        args = [exe, "devices", "-l"]
    elif action == "shell":
        if not command:
            return ToolResult(success=False, error="shell needs command=")
        args = [exe, "shell", command]
    elif action == "packages":
        args = [exe, "shell", "pm", "list", "packages"]
    elif action == "getprop":
        args = [exe, "shell", "getprop"]
    else:
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: devices, shell, packages, getprop",
        )
    try:
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
        return ToolResult(
            success=r.returncode == 0,
            output=r.stdout,
            error=r.stderr,
            data={"returncode": r.returncode},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error=f"adb timed out after {timeout}s")
    except OSError as e:
        return ToolResult(success=False, error=str(e))


ANDROID_TOOLS = (
    (
        "adb",
        "Android Debug Bridge: action= devices|shell|packages|getprop. "
        "shell needs command=. Honest absence when adb is not "
        "installed — her relay limbs go through `actuate` instead.",
        adb,
    ),
)
