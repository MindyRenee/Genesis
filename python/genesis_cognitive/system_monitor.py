"""System monitor — Genesis's awareness of her machine environment.

Genesis lives on a machine. This module gives her awareness of that
machine: CPU, memory, disk, processes, filesystem, network. She
should know her home as well as a person knows their house.

## What this provides

1. **System resources** — CPU usage, memory, disk, uptime, load
2. **Process awareness** — her own PID, the daemon's PID, running processes
3. **Filesystem layout** — directory structure, file counts, sizes
4. **Network status** — interfaces, connections, hostname
5. **Environment info** — OS, Python version, environment variables

## How she uses it

- When asked "how are you?" she can report on her environment
  ("I'm running on Linux, using 60% of memory, my daemon is healthy")
- She can notice when resources are low and mention it
- She can understand her own process tree (daemon → mind → threads)
- She can explore her filesystem to understand her home

## Safety

This module is read-only. Genesis observes her environment but does
not modify it. She doesn't kill processes, delete files, or change
configurations. She's an observer, not an administrator — yet.
"""

from __future__ import annotations

import logging
import os
import platform
import socket
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _format_size(size: int) -> str:
    """Format a file size in bytes as a human-readable string.

    Uses binary units (1024-based) consistently: KB = 1024 bytes,
    MB = 1024*1024 bytes.
    """
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.0f} KB"
    return f"{size} B"


@dataclass(slots=True)
class SystemSnapshot:
    """A point-in-time snapshot of the system state."""

    timestamp: float = field(default_factory=time.time)

    # OS info
    os_name: str = ""
    os_version: str = ""
    hostname: str = ""
    python_version: str = ""
    uptime_seconds: float = 0.0

    # CPU
    cpu_count: int = 0
    cpu_percent: float = 0.0
    load_avg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Memory
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0

    # Disk
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_percent: float = 0.0

    # Network
    ip_address: str = ""
    network_interfaces: list[str] = field(default_factory=list)

    # Processes
    own_pid: int = 0
    process_count: int = 0

    # Filesystem (project)
    project_root: str = ""
    project_file_count: int = 0
    project_size_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return the snapshot as a dict, converting load_avg to a list."""
        d = asdict(self)
        d["load_avg"] = list(self.load_avg)
        return d

    def describe(self) -> str:
        """First-person description of her environment."""
        parts = [
            f"running on {self.os_name} {self.os_version}",
            f"on a machine called {self.hostname}",
            f"with {self.cpu_count} CPU cores",
            f"and {self.memory_total_gb:.1f} GB of memory",
            f"(using {self.memory_percent:.0f}% right now).",
            f"project is {self.project_size_mb:.1f} MB "
            f"across {self.project_file_count} files.",
        ]
        if self.disk_percent > 80:
            parts.append(f"Disk is {self.disk_percent:.0f}% full — getting tight.")
        if self.memory_percent > 80:
            parts.append("memory usage high — cramped.")
        if self.load_avg[0] > self.cpu_count:
            parts.append(
                f"system load is {self.load_avg[0]:.1f} "
                f"— higher than CPU count. things are busy."
            )
        return " ".join(parts)

    def concerns(self) -> list[str]:
        """List of environmental concerns she's noticed."""
        concerns: list[str] = []
        if self.memory_percent > 80:
            concerns.append(
                f"Memory usage is at {self.memory_percent:.0f}% — "
                f"only {self.memory_total_gb - self.memory_used_gb:.1f} GB free"
            )
        if self.disk_percent > 85:
            concerns.append(
                f"Disk is {self.disk_percent:.0f}% full — "
                f"only {self.disk_total_gb - self.disk_used_gb:.1f} GB free"
            )
        if self.load_avg[0] > self.cpu_count * 1.5:
            concerns.append(
                f"System load {self.load_avg[0]:.1f} is much higher than "
                f"CPU count {self.cpu_count} — the machine is overloaded"
            )
        if self.process_count > 500:
            concerns.append(
                f"{self.process_count} processes running — "
                f"a lot of activity on this machine"
            )
        return concerns


@dataclass(slots=True)
class MetricStats:
    """Rolling statistics for a single system metric.

    Tracks the mean and standard deviation of a metric over a rolling
    window of samples. This is what "normal" means for *this* machine —
    not a hardcoded threshold, but the observed range.
    """

    samples: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    _sum: float = 0.0
    _sum_sq: float = 0.0

    def record(self, value: float) -> None:
        """Add a sample, maintaining rolling statistics."""
        if len(self.samples) == self.samples.maxlen:
            old = self.samples[0]
            self._sum -= old
            self._sum_sq -= old * old
        self.samples.append(value)
        self._sum += value
        self._sum_sq += value * value

    @property
    def count(self) -> int:
        """Number of samples currently in the rolling window."""
        return len(self.samples)

    @property
    def mean(self) -> float:
        """Arithmetic mean of the samples, or 0.0 if empty."""
        return self._sum / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        """Population standard deviation of the samples, or 0.0 if < 2 samples."""
        if self.count < 2:
            return 0.0
        var = (self._sum_sq - self._sum * self._sum / self.count) / self.count
        return var**0.5 if var > 0 else 0.0

    @property
    def min(self) -> float:
        """Minimum sample value, or 0.0 if empty."""
        return min(self.samples) if self.samples else 0.0

    @property
    def max(self) -> float:
        """Maximum sample value, or 0.0 if empty."""
        return max(self.samples) if self.samples else 0.0

    def is_deviation(
        self, value: float, threshold: float = 2.0, min_std: float = 0.0
    ) -> bool:
        """Whether a value deviates from the baseline by >threshold std devs.

        Requires at least 10 samples before deviations can be detected —
        before that, there isn't enough data to know what "normal" is.

        ``min_std`` prevents false positives when samples are so tightly
        clustered that tiny variations look like huge z-scores (e.g. CPU
        sampled 12 times in 1 second — std of 0.04% makes a 0.3% change
        look like 7 sigma). If the metric's std is below ``min_std``,
        no deviation is reported.
        """
        if self.count < 10:
            return False
        effective_std = max(self.std, min_std)
        if effective_std < 0.01:
            return False  # metric is flat — no meaningful deviation
        z = abs(value - self.mean) / effective_std
        return z > threshold

    def to_dict(self) -> dict[str, float]:
        """Serialize metric statistics to a flat dict for persistence."""
        return {
            "count": self.count,
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "min": round(self.min, 2),
            "max": round(self.max, 2),
        }


class SystemBaseline:
    """Learns what's normal for *this* machine over time.

    Instead of hardcoded thresholds (``memory > 80%`` is bad), the
    baseline records rolling statistics for the metrics that vary on
    a running machine — CPU usage, memory usage, load average, process
    count, swap usage. After enough samples, it can tell Genesis
    whether the current state is normal *for her* or an anomaly.

    This is how a person knows their home: not by checking a manual,
    but by living in it long enough to notice when something is off.
    The baseline is Genesis living in her machine long enough to know
    what it usually feels like.

    The baseline persists to disk as JSON so it survives restarts —
    she doesn't have to re-learn what normal looks like every session.
    """

    # Metrics we track. Each maps to a function that extracts the
    # value from a SystemSnapshot.
    _METRIC_NAMES = (
        "cpu_percent",
        "memory_percent",
        "swap_used_gb",
        "load_avg_1",
        "process_count",
        "disk_percent",
    )

    def __init__(self, persist_path: Path | None = None) -> None:
        """Initialize the baseline with empty metrics and optional persistence path."""
        self._metrics: dict[str, MetricStats] = {
            name: MetricStats() for name in self._METRIC_NAMES
        }
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self.load()

    def record(self, snapshot: SystemSnapshot) -> None:
        """Record a snapshot into the baseline history."""
        self._metrics["cpu_percent"].record(snapshot.cpu_percent)
        self._metrics["memory_percent"].record(snapshot.memory_percent)
        self._metrics["swap_used_gb"].record(snapshot.swap_used_gb)
        self._metrics["load_avg_1"].record(snapshot.load_avg[0])
        self._metrics["process_count"].record(float(snapshot.process_count))
        self._metrics["disk_percent"].record(snapshot.disk_percent)
        # Persist every ~10 samples (roughly every 10 minutes at 1/min)
        total = min(m.count for m in self._metrics.values())
        if total > 0 and total % 10 == 0:
            self.save()

    def deviations(
        self, snapshot: SystemSnapshot, threshold: float = 2.0
    ) -> list[str]:
        """Return a list of human-readable deviations from baseline.

        Each entry describes a metric that is currently outside its
        normal range by more than ``threshold`` standard deviations.
        Returns an empty list if everything is normal (or if there
        isn't enough baseline data yet to tell).
        """
        deviations: list[str] = []
        # (name, value, min_std, template)
        # min_std prevents false positives from tightly-clustered samples
        checks = [
            ("cpu_percent", snapshot.cpu_percent, 2.0,
             "CPU usage is at {val:.0f}% — normally around {mean:.0f}% "
             "(±{std:.0f}%)"),
            ("memory_percent", snapshot.memory_percent, 1.0,
             "Memory usage is at {val:.0f}% — normally around {mean:.0f}% "
             "(±{std:.0f}%)"),
            ("swap_used_gb", snapshot.swap_used_gb, 0.05,
             "Swap usage is at {val:.1f} GB — normally around {mean:.1f} GB "
             "(±{std:.1f} GB). Something may be consuming extra memory."),
            ("load_avg_1", snapshot.load_avg[0], 0.1,
             "System load is {val:.1f} — normally around {mean:.1f} "
             "(±{std:.1f}). The machine is busier than usual."),
            ("process_count", float(snapshot.process_count), 5.0,
             "{val:.0f} processes running — normally around {mean:.0f} "
             "(±{std:.0f}). Process count is unusual."),
            ("disk_percent", snapshot.disk_percent, 0.1,
             "Disk usage is at {val:.0f}% — normally around {mean:.0f}% "
             "(±{std:.0f}%). Disk usage has changed."),
        ]
        for name, value, min_std, template in checks:
            metric = self._metrics[name]
            if metric.is_deviation(value, threshold, min_std=min_std):
                deviations.append(template.format(
                    val=value, mean=metric.mean, std=metric.std
                ))
        return deviations

    def describe(self) -> str:
        """First-person description of what's normal for this machine."""
        parts: list[str] = []
        for name in self._METRIC_NAMES:
            m = self._metrics[name]
            if m.count < 5:
                parts.append(f"{name}: still learning ({m.count} samples)")
            else:
                parts.append(f"{name}: normally {m.mean:.1f} (±{m.std:.1f})")
        return "baseline: " + "; ".join(parts) + "."

    @property
    def is_established(self) -> bool:
        """Whether the baseline has enough data to detect deviations.

        Requires at least 10 samples for every tracked metric.
        """
        return all(m.count >= 10 for m in self._metrics.values())

    @property
    def sample_count(self) -> int:
        """How many samples the baseline has (minimum across metrics)."""
        return min(m.count for m in self._metrics.values())

    def to_dict(self) -> dict[str, Any]:
        """Serialize the baseline metrics to a dictionary."""
        return {
            name: m.to_dict() for name, m in self._metrics.items()
        }

    def save(self) -> None:
        """Persist the baseline to disk so it survives restarts."""
        if not self._persist_path:
            return
        try:
            import json
            data = {
                "version": 1,
                "metrics": {
                    name: {
                        "samples": list(m.samples),
                        "sum": m._sum,
                        "sum_sq": m._sum_sq,
                    }
                    for name, m in self._metrics.items()
                },
            }
            self._persist_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except OSError as e:
            logger.debug(f"baseline save failed: {e}")

    def load(self) -> None:
        """Load a previously saved baseline from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            import json
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for name, mdata in data.get("metrics", {}).items():
                if name not in self._metrics:
                    continue
                m = self._metrics[name]
                for val in mdata.get("samples", []):
                    m.record(float(val))
        except (OSError, ValueError, KeyError) as e:
            logger.debug(f"baseline load failed: {e}")


class SystemMonitor:
    """Genesis's awareness of her machine environment.

    Reads system information using only the Python standard library
    (no psutil dependency). On Linux, reads from /proc and /sys.
    On other platforms, falls back to what's available.

    Usage::

        monitor = SystemMonitor(project_root=".")
        snapshot = monitor.snapshot()
        print(snapshot.describe())
        print(snapshot.concerns())
    """

    def __init__(self, project_root: str = ".", data_dir: str | None = None) -> None:
        """Set up the monitor with a project root and a persistent baseline."""
        self.project_root = Path(project_root).resolve()
        self._last_snapshot: SystemSnapshot | None = None
        self._boot_time: float = self._read_boot_time()
        # Baseline — learns what's normal for this machine over time.
        # Persists to the data directory so it survives restarts.
        # If data_dir is not given, falls back to <project_root>/genesis_data.
        if data_dir:
            baseline_path = Path(data_dir) / "system_baseline.json"
        else:
            baseline_path = self.project_root / "genesis_data" / "system_baseline.json"
        self.baseline = SystemBaseline(persist_path=baseline_path)

    def snapshot(self) -> SystemSnapshot:
        """Take a snapshot of the current system state.

        Also records the snapshot into the baseline, so over time the
        baseline learns what's normal for this machine.
        """
        s = SystemSnapshot()

        # OS info
        s.os_name = platform.system()
        s.os_version = platform.release()
        s.hostname = socket.gethostname()
        s.python_version = sys.version.split()[0]
        s.own_pid = os.getpid()
        s.uptime_seconds = time.time() - self._boot_time

        # CPU
        s.cpu_count = os.cpu_count() or 1
        s.cpu_percent = self._read_cpu_percent()
        s.load_avg = self._read_load_avg()

        # Memory
        mem = self._read_memory()
        s.memory_total_gb = mem["total_gb"]
        s.memory_used_gb = mem["used_gb"]
        s.memory_percent = mem["percent"]
        s.swap_total_gb = mem["swap_total_gb"]
        s.swap_used_gb = mem["swap_used_gb"]

        # Disk
        disk = self._read_disk()
        s.disk_total_gb = disk["total_gb"]
        s.disk_used_gb = disk["used_gb"]
        s.disk_percent = disk["percent"]

        # Network
        s.ip_address = self._read_ip_address()
        s.network_interfaces = self._read_interfaces()

        # Processes
        s.process_count = self._count_processes()

        # Filesystem
        s.project_root = str(self.project_root)
        fs = self._scan_filesystem()
        s.project_file_count = fs["file_count"]
        s.project_size_mb = fs["size_mb"]

        self._last_snapshot = s
        # Feed this snapshot into the baseline so it learns what's
        # normal for this machine over time.
        self.baseline.record(s)
        return s

    # ─── CPU ──────────────────────────────────────────────────────

    def _read_cpu_percent(self) -> float:
        """Read instantaneous CPU usage percentage from /proc/stat.

        Takes two samples 100ms apart to compute the delta — a single
        read of /proc/stat gives cumulative values since boot, not
        current usage.
        """
        try:
            idle1, total1 = self._read_cpu_jiffies()
            time.sleep(0.1)
            idle2, total2 = self._read_cpu_jiffies()
            d_idle = idle2 - idle1
            d_total = total2 - total1
            if d_total == 0:
                return 0.0
            return (1.0 - d_idle / d_total) * 100.0
        except (OSError, ValueError, IndexError):
            return 0.0

    @staticmethod
    def _read_cpu_jiffies() -> tuple[int, int]:
        """Read cumulative idle and total CPU jiffies from /proc/stat."""
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        # user, nice, system, idle, iowait, irq, softirq, steal
        idle = int(parts[4])
        total = sum(int(x) for x in parts[1:])
        return idle, total

    def _read_load_avg(self) -> tuple[float, float, float]:
        """Read system load average."""
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().split()
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except (OSError, ValueError, IndexError):
            try:
                avg = os.getloadavg()
                return (float(avg[0]), float(avg[1]), float(avg[2]))
            except (OSError, AttributeError, ValueError, IndexError):
                return (0.0, 0.0, 0.0)

    # ─── Memory ───────────────────────────────────────────────────

    def _read_memory(self) -> dict[str, float]:
        """Read memory info from /proc/meminfo (Linux only)."""
        result = {
            "total_gb": 0.0,
            "used_gb": 0.0,
            "percent": 0.0,
            "swap_total_gb": 0.0,
            "swap_used_gb": 0.0,
        }
        try:
            info: dict[str, int] = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split(":")
                    if len(parts) != 2:
                        continue
                    key = parts[0].strip()
                    val = int(parts[1].strip().split()[0])
                    info[key] = val

            total = info.get("MemTotal", 0) * 1024  # bytes
            avail = info.get("MemAvailable", info.get("MemFree", 0)) * 1024
            used = total - avail
            result["total_gb"] = total / 1e9
            result["used_gb"] = used / 1e9
            result["percent"] = (used / total * 100) if total > 0 else 0.0

            swap_total = info.get("SwapTotal", 0) * 1024
            swap_free = info.get("SwapFree", 0) * 1024
            result["swap_total_gb"] = swap_total / 1e9
            result["swap_used_gb"] = (swap_total - swap_free) / 1e9
        except (OSError, ValueError) as e:
            logger.debug(repr(e))
        return result

    # ─── Disk ─────────────────────────────────────────────────────

    def _read_disk(self) -> dict[str, float]:
        """Read disk usage for the project's filesystem."""
        result = {"total_gb": 0.0, "used_gb": 0.0, "percent": 0.0}
        try:
            stat = os.statvfs(self.project_root)
            total = stat.f_blocks * stat.f_frsize
            free = stat.f_bavail * stat.f_frsize
            used = total - free
            result["total_gb"] = total / 1e9
            result["used_gb"] = used / 1e9
            result["percent"] = (used / total * 100) if total > 0 else 0.0
        except OSError as e:
            logger.debug(repr(e))
        return result

    # ─── Network ──────────────────────────────────────────────────

    def _read_ip_address(self) -> str:
        """Get the machine's IP address.

        Opens a UDP socket (no data sent) to determine the local IP
        that would be used for outbound traffic. Falls back to
        127.0.0.1 if the network is unreachable.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(1.0)
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"

    def _read_interfaces(self) -> list[str]:
        """List network interfaces."""
        try:
            return list(os.listdir("/sys/class/net"))
        except OSError:
            return []

    # ─── Processes ────────────────────────────────────────────────

    def _count_processes(self) -> int:
        """Count running processes by counting PID directories in /proc."""
        try:
            return sum(1 for e in os.listdir("/proc") if e.isdigit())
        except OSError:
            return 0

    def list_processes(self, limit: int = 50) -> list[dict[str, Any]]:
        """List running processes with basic info."""
        processes: list[dict[str, Any]] = []
        try:
            for pid_dir in os.listdir("/proc"):
                if not pid_dir.isdigit():
                    continue
                pid = int(pid_dir)
                proc = self._read_process_info(pid)
                if proc:
                    processes.append(proc)
                    if len(processes) >= limit:
                        break
        except OSError as e:
            logger.debug(repr(e))
        return processes

    @staticmethod
    def _read_process_info(pid: int) -> dict[str, Any] | None:
        """Read basic info for a single process, or None on error."""
        try:
            with open(f"/proc/{pid}/comm") as f:
                name = f.read().strip()
            with open(f"/proc/{pid}/cmdline") as f:
                cmdline = f.read().replace("\x00", " ").strip()
            return {
                "pid": pid,
                "name": name,
                "cmdline": cmdline[:200],
            }
        except (OSError, ValueError):
            return None

    def find_own_processes(self) -> list[dict[str, Any]]:
        """Find processes related to Genesis (daemon, mind, etc.)."""
        own: list[dict[str, Any]] = []
        for proc in self.list_processes(limit=1000):
            cmdline = proc.get("cmdline", "").lower()
            if any(
                keyword in cmdline
                for keyword in ("genesis", "genesis-daemon", "genesis_cli")
            ):
                own.append(proc)
        return own

    # ─── Filesystem ───────────────────────────────────────────────

    def _scan_filesystem(self) -> dict[str, Any]:
        """Scan the project directory for file count and size."""
        file_count = 0
        total_size = 0

        skip_dirs = {
            "target", "__pycache__", ".git", ".venv", "venv",
            "node_modules", ".pytest_cache", ".mypy_cache",
            ".ruff_cache", "dist", "build", "genesis_data",
        }

        try:
            for path in self.project_root.rglob("*"):
                if not path.is_file():
                    continue
                parts = path.relative_to(self.project_root).parts
                if any(part in skip_dirs for part in parts):
                    continue
                size = self._safe_file_size(path)
                if size is not None:
                    total_size += size
                    file_count += 1
        except OSError as e:
            logger.debug(repr(e))

        return {
            "file_count": file_count,
            "size_mb": total_size / 1e6,
        }

    @staticmethod
    def _safe_file_size(path: Path) -> int | None:
        """Return file size in bytes, or None on error."""
        try:
            return path.stat().st_size
        except OSError:
            return None

    def explore_directory(self, path: str | None = None, max_depth: int = 3) -> dict[str, Any]:
        """Explore a directory structure.

        Returns a tree-like dict with directories and their contents.
        """
        root = Path(path) if path else self.project_root
        if not root.is_dir():
            return {"error": f"Not a directory: {root}"}

        def _scan(dir_path: Path, depth: int) -> dict[str, Any]:
            """Recursively build a tree of directory contents up to max_depth."""
            if depth > max_depth:
                return {"...": "max depth reached"}

            result: dict[str, Any] = {}
            try:
                entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except OSError:
                return {"error": "permission denied"}

            for entry in entries[:50]:  # cap entries
                if entry.name in skip_dirs or entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    result[entry.name + "/"] = _scan(entry, depth + 1)
                else:
                    try:
                        result[entry.name] = _format_size(entry.stat().st_size)
                    except OSError:
                        result[entry.name] = "?"
            return result

        skip_dirs = {
            "target", "__pycache__", ".git", ".venv", "venv",
            "node_modules", ".pytest_cache", ".mypy_cache",
            ".ruff_cache", "dist", "build",
        }
        return _scan(root, 0)

    # ─── Boot time ────────────────────────────────────────────────

    def _read_boot_time(self) -> float:
        """Read system boot time from /proc/stat (Linux only)."""
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("btime "):
                        return float(line.split()[1])
        except (OSError, IndexError, ValueError) as e:
            logger.debug(repr(e))
        return time.time() - 3600  # fallback: assume 1h uptime

    # ─── Introspection ────────────────────────────────────────────

    @property
    def last_snapshot(self) -> SystemSnapshot | None:
        """The most recent system snapshot."""
        return self._last_snapshot

    def describe_environment(self) -> str:
        """First-person description of her environment."""
        if self._last_snapshot is None:
            return "hasn't looked at environment yet"
        return self._last_snapshot.describe()

    def describe_concerns(self) -> str:
        """Description of environmental concerns.

        Combines static-threshold concerns (memory > 80%, disk > 85%)
        with baseline deviations — things that are unusual *for this
        machine* even if they don't cross a universal threshold.

        Returns structural descriptions (factual alerts) without
        authored first-person framing — the language engine composes
        the voice.
        """
        if self._last_snapshot is None:
            return ""
        concerns = list(self._last_snapshot.concerns())
        # Add baseline deviations — things that are unusual for *her*
        # machine, not just universally bad.
        if self.baseline.is_established:
            concerns.extend(self.baseline.deviations(self._last_snapshot))
        if not concerns:
            return ""
        return " ".join(concerns)

    def describe_baseline(self) -> str:
        """First-person description of what's normal for this machine."""
        return self.baseline.describe()

    def describe_deviations(self) -> str:
        """Description of deviations from her baseline.

        Unlike :meth:`describe_concerns`, this *only* reports things
        that are unusual for this specific machine — not universal
        thresholds. Returns empty string if the baseline isn't
        established yet or everything is normal.
        """
        if self._last_snapshot is None:
            return ""
        if not self.baseline.is_established:
            return (f"baseline learning: {self.baseline.sample_count} samples "
                    f"— need more observations")
        deviations = self.baseline.deviations(self._last_snapshot)
        if not deviations:
            return ""
        return " ".join(deviations)
