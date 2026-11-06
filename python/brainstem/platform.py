"""Canonical runtime platform detection.

One source tree runs on three Android realities that are not the same
platform:

- ``linux`` — ordinary desktop/server Linux (systemd, full sysfs, X/Wayland).
- ``termux`` — Android userspace: ``termux-*`` commands and the Termux:API
  bridge are the hardware boundary.
- ``android-proot`` — a full Linux distro (Ubuntu, Debian…) inside proot on
  an Android kernel, e.g. Andronix. glibc userland; ``/proc`` and parts of
  ``/sys`` are visible; no ``termux-*`` commands, no Android service access,
  typically no root. Compile-time ``target_os = "android"`` never fires here —
  the build target is plain Linux, so detection must be runtime.

Detection is runtime and cached. Capability checks answer "can we reach it"
rather than "what is the kernel", and an absent capability must stay absent —
never fabricate a reading.
"""
from __future__ import annotations

import os
import platform as _platform
import shutil
from functools import lru_cache

LINUX = "linux"
TERMUX = "termux"
ANDROID_PROOT = "android-proot"


@lru_cache(maxsize=1)
def _kernel_release() -> str:
    return _platform.release().lower()


@lru_cache(maxsize=1)
def is_android_kernel() -> bool:
    """True when the running kernel is Android's.

    Termux and proot distros both see the host kernel release, which on
    Android carries an android/vendor signature (``-android``, ``-andronix``,
    ``-aospa``…) or an Android-specific ``/proc/version`` builder.
    """
    release = _kernel_release()
    if any(tag in release for tag in ("android", "aospa", "andronix")):
        return True
    try:
        with open("/proc/version", encoding="utf-8", errors="replace") as fh:
            return "android" in fh.read().lower()
    except OSError:
        return False


def is_termux() -> bool:
    """True inside native Termux (the Android terminal environment)."""
    if os.environ.get("TERMUX_VERSION"):
        return True
    prefix = os.environ.get("PREFIX", "")
    return "/com.termux/" in prefix or os.path.isdir("/data/data/com.termux")


def is_android_proot() -> bool:
    """True on a Linux distro running under proot on an Android kernel.

    Android kernel, but not Termux, and no Android userspace tools
    (``getprop``/``am``) on PATH — the signature of Andronix-style proot.
    """
    if not is_android_kernel() or is_termux():
        return False
    if os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"):
        return True
    # Kernel says Android but no Android userspace tools are reachable:
    # that is exactly what a proot distro looks like.
    return shutil.which("getprop") is None


@lru_cache(maxsize=1)
def platform_name() -> str:
    """The canonical platform tag: ``termux``, ``android-proot``, or ``linux``."""
    if is_termux():
        return TERMUX
    if is_android_proot():
        return ANDROID_PROOT
    return LINUX


def pulse_audio_available() -> bool:
    """True when PulseAudio/PipeWire answers — the proot audio bridge."""
    if shutil.which("pactl") is None:
        return False
    import subprocess

    try:
        result = subprocess.run(
            ["pactl", "info"], capture_output=True, timeout=3, check=False
        )
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
