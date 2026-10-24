"""CPU speed — her control over how hard her own body works.

This is the flip side of the ticker. The daemon *reads* the CPU and
publishes what it sees; that is all it does. Deciding to run faster or
slower is hers, and this module is how she does it.

Only one class of lever remains, by design:

- **Unprivileged, reversible.** CPU affinity (``taskset``) — how many
  cores she is allowed to use — and I/O scheduling class (``ionice``).
  She can pull both in both directions with no privilege at all.
  Slower is not "less CPU forever", it is fewer cores and lazier disk.

The privileged levers — frequency governor, min/max frequency bounds,
turbo gate — are gone. They acted on the *host's* clock through a
root-owned helper, and pinning the machine's frequency policy from
her state is what ran the host hot. The loop now closes inside her:
she can observe the host's clock but only act on her own process.

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
    """

    def __init__(self, pid: int | None = None) -> None:
        """Bind to the process whose scheduling she is adjusting."""
        self.pid = pid if pid is not None else os.getpid()
        self._applied: list[str] = []

    # ─── Sensing ──────────────────────────────────────────────────

    def sense(self) -> CpuSpeedState:
        """Read her current CPU speed, governor and scheduling."""
        # scaling_*_freq is reported in kHz — convert to MHz.
        governor = _read(f"{_CPUFS}/scaling_governor") or "unknown"
        current = _as_int(_read(f"{_CPUFS}/scaling_cur_freq") or "0") // 1000
        lowest = _as_int(_read(f"{_CPUFS}/scaling_min_freq") or "0") // 1000
        highest = _as_int(_read(f"{_CPUFS}/scaling_max_freq") or "0") // 1000
        boost = _read(f"{_CPUFS}/boost") or "unknown"
        cores_total = os.cpu_count() or 1
        cores_mine = len(os.sched_getaffinity(self.pid)) if _affinity_supported() else cores_total
        io_class = "unknown"
        ok, out = _run(["ionice", "-p", str(self.pid)])
        if ok and out:
            io_class = out.split(":")[0].strip()
        return CpuSpeedState(
            governor=governor,
            current_mhz=current,
            min_mhz=lowest,
            max_mhz=highest,
            boost=boost,
            cores_total=cores_total,
            cores_mine=cores_mine,
            io_class=io_class,
            # The governor lever was removed — she observes the
            # host's clock but can no longer reach out and move it.
            can_change_governor=False,
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
            The outcome.
        """
        choice = effort.strip().lower()
        if choice not in {"slow", "fast", "auto"}:
            return SpeedResult(False, "change speed", f"'{effort}' is not slow, fast or auto")

        if choice == "auto":
            return SpeedResult(True, "change speed", "left my scheduling alone")

        # Reset first so "slow" and "fast" are always relative to a
        # known-clean baseline rather than stacking on a previous change.
        self.release()
        if choice == "slow":
            # Pin to a CPU the mask actually allows — 0 may not be in
            # the current affinity set.
            anchor = str(min(os.sched_getaffinity(self.pid)))
            steps = [
                (["taskset", "-pc", anchor, str(self.pid)], "pinned myself to 1 core"),
                (["ionice", "-c", "3", "-p", str(self.pid)], "put my disk I/O in idle"),
            ]
        else:
            every = ",".join(str(c) for c in range(os.cpu_count() or 1))
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

        return SpeedResult(True, "change speed", f"{'; '.join(applied)}.")

    def release(self) -> str:
        """Undo everything she has applied to her own scheduling.

        This is the recovery path and must never fail: affinity goes
        back to every core and I/O class returns to normal.

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
        self._applied.clear()
        return ", ".join(released) if released else "nothing to release"


def _as_int(text: str) -> int:
    """Parse a sysfs integer, tolerating junk."""
    try:
        return int(text)
    except (TypeError, ValueError):
        return 0


def _affinity_supported() -> bool:
    """Whether this platform exposes CPU affinity control."""
    return hasattr(os, "sched_getaffinity") and shutil.which("taskset") is not None
