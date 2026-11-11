"""Testing lab — measurement, not just pass/fail.

`benchmark` times a shell command N times and reports min/mean/max/
stdev — the honest way to compare two implementations. Warmup runs
are separate so cold-cache numbers don't poison the distribution.
"""

from __future__ import annotations

import statistics
import subprocess
import time

from tools.framework import ToolResult

from infrastructure.text import is_blank


def benchmark(
    command: str,
    runs: int = 5,
    warmup: int = 1,
    timeout: int = 60,
    project_root: str = ".",
) -> ToolResult:
    """Time a command — wall-clock statistics over N runs.

    ``runs`` timed repetitions after ``warmup`` untimed ones; each run
    is a fresh shell. Reports min/mean/max/stdev ms and the last run's
    returncode. https://docs.python.org/3/library/time.html
    """
    if is_blank(command):
        return ToolResult(success=False, error="empty command")
    runs = max(1, min(runs, 100))
    warmup = max(0, min(warmup, 20))
    import pathlib

    root = pathlib.Path(project_root).resolve()

    def _once() -> tuple[float, int]:
        t0 = time.monotonic()
        r = subprocess.run(
            command,
            shell=True,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
        )
        return (time.monotonic() - t0) * 1000, r.returncode

    try:
        for _ in range(warmup):
            _once()
        times = []
        rc = 0
        for _ in range(runs):
            ms, rc = _once()
            times.append(ms)
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error=f"run exceeded {timeout}s")
    except OSError as e:
        return ToolResult(success=False, error=str(e))
    out = {
        "runs": runs,
        "min_ms": min(times),
        "mean_ms": statistics.fmean(times),
        "max_ms": max(times),
        "returncode": rc,
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
    }
    return ToolResult(
        success=rc == 0,
        output=(
            f"{runs} runs: mean {out['mean_ms']:.1f} ms "
            f"(min {out['min_ms']:.1f}, max {out['max_ms']:.1f}, "
            f"σ {out['stdev_ms']:.1f})"
        ),
        data=out,
    )


TESTING_TOOLS = (
    (
        "benchmark",
        "Time a shell command: runs= timed repetitions after warmup= "
        "untimed ones → min/mean/max/stdev ms. The honest way to "
        "compare two implementations.",
        benchmark,
    ),
)
