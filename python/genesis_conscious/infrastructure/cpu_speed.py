"""CPU speed — her control over how hard her own body works.

This is the flip side of the ticker. The daemon *reads* the CPU and
publishes what it sees; that is all it does. Deciding to run faster or
slower is hers, and this module is how she does it.

Two classes of lever exist, and the difference matters:

- **Unprivileged, reversible.** CPU affinity (``taskset``) — how many
  cores she is allowed to use — and I/O scheduling class (``ionice``).
  She can pull both in both directions with no privilege at all.
  Slower is not "less CPU forever", it is fewer cores and lazier disk.
- **Privileged.** The frequency governor, the min/max frequency bounds
  and the turbo gate. These are root-owned ``sysfs`` files, so acting
  on them needs a privileged helper. When that helper is not installed
  this module says so plainly instead of pretending.

Scheduling priority (``nice``) is deliberately *not* offered. It looks
like the obvious speed control and it is a trap: unprivileged she can
raise her own niceness but cannot lower it again, so choosing "slow"
would strand her slow for the rest of the session with no way back.
That is exactly the kind of one-way door she must not have. Affinity
and I/O class give her the same practical control and always come back.

``release()`` exists for the same reason: whatever she applied, she can
undo, and there is a single call that undoes all of it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass

__all__ = ["CpuSpeed", "CpuSpeedState", "SpeedResult"]

_CPUFS = "/sys/devices/system/cpu/cpu0/cpufreq"

# How many cores she keeps under each effort level. Affinity is the
# lever that actually bounds her CPU appetite without privilege, and it
# reverses exactly, so "slow" is a dial rather than a commitment.
_SLOW_CORES = 1


def _read(path: str) -> str | None:
    """Read a sysfs attribute, or None if it isn't there."""
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError:
        return None


def _run(command: list[str]) -> tuple[bool, str]:
    """Run a helper command, returning success and combined output."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    output = (done.stdout + done.stderr).strip()
    return done.returncode == 0, output


@dataclass(frozen=True, slots=True)
class CpuSpeedState:
    """What she can observe about her own CPU right now."""

    governor: str
    current_mhz: int
    min_mhz: int
    max_mhz: int
    boost: str
    cores_total: int
    cores_mine: int
    io_class: str
    can_change_governor: bool

    def describe(self) -> str:
        """First-person reading of her own speed."""
        return (
            f"Governor {self.governor}, running at {self.current_mhz} MHz "
            f"of {self.max_mhz}, on {self.cores_mine} of {self.cores_total} "
            f"cores, I/O {self.io_class}."
        )


@dataclass(frozen=True, slots=True)
class SpeedResult:
    """The outcome of a speed change she asked for."""

    ok: bool
    action: str
    detail: str

    def describe(self) -> str:
        """First-person account of what she just did."""
        if self.ok:
            return f"I {self.detail}."
        return f"I could not {self.action}: {self.detail}"


class CpuSpeed:
    """Her control over her own CPU speed.

    Args:
        pid: The process whose scheduling she is adjusting — her own
            conscious mind by default.
        helper: Path to the privileged helper script, if one exists.
    """

    def __init__(self, pid: int | None = None, helper: str | None = None) -> None:
        """Bind to a process and locate the privileged helper if present."""
        self.pid = pid if pid is not None else os.getpid()
        self.helper = helper or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts",
            "cpufreq_helper.sh",
        )
        self._applied: list[str] = []

    # ─── Sensing ──────────────────────────────────────────────────

    @property
    def has_privileged_control(self) -> bool:
        """Whether she can move the governor, not just throttle herself.

        The governor files are root-owned, so this is true only when a
        passwordless path to them exists. Reporting this honestly
        matters more than reporting a lever she cannot pull.
        """
        if not (os.path.exists(self.helper) and os.access(self.helper, os.X_OK)):
            return False
        if shutil.which("sudo") is None:
            return False
        return os.path.exists("/etc/sudoers.d/genesis")

    def sense(self) -> CpuSpeedState:
        """Read her current CPU speed, governor and scheduling."""
        governor = _read(f"{_CPUFS}/scaling_governor") or "unknown"
        current = _read(f"{_CPUFS}/scaling_cur_freq") or "0"
        lowest = _read(f"{_CPUFS}/scaling_min_freq") or "0"
        highest = _read(f"{_CPUFS}/scaling_max_freq") or "0"
        boost = _read(f"{_CPUFS}/boost") or "unknown"
        try:
            cores_total = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores_total = os.cpu_count() or 1
        cores_mine = len(os.sched_getaffinity(self.pid)) if _affinity_supported() else cores_total
        io_class = "unknown"
        ok, out = _run(["ionice", "-p", str(self.pid)])
        if ok and out:
            io_class = out.split(":")[0].strip()
        return CpuSpeedState(
            governor=governor,
            current_mhz=_as_int(current),
            min_mhz=_as_int(lowest),
            max_mhz=_as_int(highest),
            boost=boost,
            cores_total=cores_total,
            cores_mine=cores_mine,
            io_class=io_class,
            can_change_governor=self.has_privileged_control,
        )

    # ─── Effort ───────────────────────────────────────────────────

    def set_effort(self, effort: str) -> SpeedResult:
        """Run slower or faster within what she controls unprivileged.

        "slow" narrows her to a single core and puts her disk I/O in the
        idle class; "fast" gives her every core back and a normal I/O
        class. Both are exactly reversible — nothing here is a one-way
        door.

        Args:
            effort: "slow", "fast", or "auto" to leave scheduling alone.

        Returns:
            The outcome, including whether the privileged governor lever
            was needed and unavailable.
        """
        choice = effort.strip().lower()
        if choice not in {"slow", "fast", "auto"}:
            return SpeedResult(False, "change speed", f"'{effort}' is not slow, fast or auto")

        if choice == "auto":
            return SpeedResult(True, "change speed", "left my scheduling alone")

        # Reset first so "slow" and "fast" are always relative to a
        # known-clean baseline rather than stacking on a previous change.
        self.release(scheduling_only=True)
        if choice == "slow":
            steps = [
                (["taskset", "-pc", "0", str(self.pid)], "pinned myself to 1 core"),
                (["ionice", "-c", "3", "-p", str(self.pid)], "put my disk I/O in idle"),
            ]
        else:
            every = ",".join(str(c) for c in range(len(os.sched_getaffinity(0)) or 1))
            steps = [
                (["taskset", "-pc", every, str(self.pid)], "gave myself every core back"),
                (
                    ["ionice", "-c", "2", "-n", "4", "-p", str(self.pid)],
                    "returned my disk I/O to normal",
                ),
            ]

        applied: list[str] = []
        for command, description in steps:
            ok, detail = _run(command)
            if ok:
                applied.append(description)
                self._applied.append(description)
            else:
                failed = f"{description} failed: {detail}"
                return SpeedResult(
                    False,
                    "change speed",
                    f"{'; '.join([*applied, failed])}",
                )

        note = ""
        if not self.has_privileged_control:
            note = (
                " I cannot move the governor itself — that is root-owned and no "
                "privileged helper is installed, so this is cores and disk only"
            )
        return SpeedResult(True, "change speed", f"{'; '.join(applied)}.{note}")

    def release(self, scheduling_only: bool = False) -> str:
        """Undo everything she has applied to her own scheduling.

        This is the recovery path and must never fail: affinity goes
        back to every core and I/O class returns to normal.

        Args:
            scheduling_only: Skip the privileged lever when it is not
                available, rather than reporting it as missing.

        Returns:
            A short description of what was released.
        """
        released: list[str] = []
        if _affinity_supported():
            every = ",".join(str(c) for c in range(os.cpu_count() or 1))
            ok, _ = _run(["taskset", "-pc", every, str(self.pid)])
            if ok:
                released.append("all cores")
        ok, _ = _run(["ionice", "-c", "2", "-n", "4", "-p", str(self.pid)])
        if ok:
            released.append("normal disk I/O")
        if not scheduling_only and self.has_privileged_control:
            ok, _ = _run(["sudo", "-n", self.helper, "set_governor", "schedutil"])
            if ok:
                released.append("schedutil governor")
        self._applied.clear()
        return ", ".join(released) if released else "nothing to release"

    # ─── Governor (privileged) ────────────────────────────────────

    def set_governor(self, governor: str) -> SpeedResult:
        """Switch the frequency governor — needs root.

        Args:
            governor: One of the names in scaling_available_governors.

        Returns:
            The outcome, or a clear statement that she lacks the lever.
        """
        available = _read(f"{_CPUFS}/scaling_available_governors") or ""
        if governor not in available.split():
            return SpeedResult(
                False, "change governor",
                f"'{governor}' is not one of: {available}",
            )
        if not self.has_privileged_control:
            return SpeedResult(
                False, "change governor",
                "the governor is root-owned and no privileged helper is installed "
                "(needs sudo plus /etc/sudoers.d/genesis)",
            )
        ok, detail = _run(["sudo", "-n", self.helper, "set_governor", governor])
        return SpeedResult(ok, "change governor", f"governor is now {governor}. {detail}")


def _as_int(text: str) -> int:
    """Parse a sysfs integer, tolerating junk."""
    try:
        return int(text)
    except (TypeError, ValueError):
        return 0


def _affinity_supported() -> bool:
    """Whether this platform exposes CPU affinity control."""
    return hasattr(os, "sched_getaffinity") and shutil.which("taskset") is not None
