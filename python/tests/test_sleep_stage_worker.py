"""Sleep-stage consolidation must not block the inner-life thread.

`InnerLife._determine_dream_stage` fires a callback when the ultradian
cycle enters a new stage, and that callback used to run the whole N3 pass
inline on the inner-life thread. N3 consolidation is heavy and
uninterruptible, so a single pass:

- made shutdown unable to stop the inner life (every stop reported
  "InnerLife thread did not exit"), and
- stopped dream generation for its duration, because dreams are generated
  on that same thread.

`SleepStageWorker` moves the heavy half onto its own thread. These tests
pin the properties that makes safe: nothing is dropped, nothing overlaps,
and anything already accepted is drained before the worker exits — because
shutdown closes the archive underneath it.
"""

from __future__ import annotations

import threading
import time

from genesis_conscious.mind.sleep import SleepStageWorker
from genesis_conscious.sleep.architecture import SleepStage


class _Recorder:
    """Records the stages a worker runs, and when they overlap."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.seen: list[SleepStage] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def __call__(self, stage: SleepStage, dt: float) -> None:
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.delay:
                time.sleep(self.delay)
            with self._lock:
                self.seen.append(stage)
        finally:
            with self._lock:
                self.concurrent -= 1

    def wait_for(self, n: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.seen) >= n:
                    return True
            time.sleep(0.01)
        return False


def test_submit_returns_immediately_instead_of_running_inline() -> None:
    """The whole point: the caller's thread is not held.

    A slow pass must not delay the submitter. Before the worker existed,
    a 0.4 s consolidation pass blocked the inner-life thread for 0.4 s.
    """
    rec = _Recorder(delay=0.4)
    worker = SleepStageWorker(rec)
    worker.start()
    try:
        start = time.monotonic()
        worker.submit(SleepStage.N3, 0.5)
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, (
            f"submit blocked the caller for {elapsed:.3f}s; consolidation is "
            f"still running on the caller's thread"
        )
        assert rec.wait_for(1), "the pass never ran"
    finally:
        worker.stop(timeout=5)


def test_no_stage_is_dropped() -> None:
    """Every accepted stage runs, in order.

    Coalescing to the newest request would be cheaper but would silently
    skip an N3 pass when a later stage arrived while it was still queued.
    """
    rec = _Recorder()
    worker = SleepStageWorker(rec)
    worker.start()
    try:
        for stage in (
            SleepStage.N2, SleepStage.N3, SleepStage.REM, SleepStage.N3,
        ):
            worker.submit(stage, 0.5)
        assert rec.wait_for(4), f"only ran {rec.seen}"
    finally:
        worker.stop(timeout=5)

    assert rec.seen == [SleepStage.N2, SleepStage.N3, SleepStage.REM, SleepStage.N3]


def test_passes_never_overlap() -> None:
    """Consolidation mutates the network, edge log and archive at once.

    Two concurrent passes would interleave writes to all three, so the
    worker must serialize even though it is a thread.
    """
    rec = _Recorder(delay=0.05)
    worker = SleepStageWorker(rec)
    worker.start()
    try:
        for _ in range(6):
            worker.submit(SleepStage.N3, 0.5)
        assert rec.wait_for(6, timeout=10), f"only ran {len(rec.seen)}"
    finally:
        worker.stop(timeout=5)

    assert rec.max_concurrent == 1, (
        f"{rec.max_concurrent} consolidation passes ran concurrently"
    )


def test_stop_drains_what_was_already_accepted() -> None:
    """Shutdown closes the archive, so accepted work must still run.

    A stage the worker has already accepted and not yet started cannot be
    dropped: the pass would be lost, and stopping early is precisely when
    the archive is about to close.
    """
    rec = _Recorder(delay=0.2)
    worker = SleepStageWorker(rec)
    worker.start()
    worker.submit(SleepStage.N3, 0.5)
    worker.submit(SleepStage.N3, 0.5)

    assert worker.stop(timeout=10), "worker should exit within the timeout"
    assert len(rec.seen) == 2, (
        f"stop dropped accepted work: ran {len(rec.seen)} of 2"
    )
    assert worker.pending == 0


def test_queue_is_bounded_and_says_when_it_drops() -> None:
    """A backlog must not grow without limit, and must not fail silently."""
    rec = _Recorder()
    worker = SleepStageWorker(rec)
    # Deliberately do not start it, so nothing drains.
    depth = worker.submit(SleepStage.N3, 0.5)
    assert depth == 1
    for _ in range(SleepStageWorker.MAX_PENDING + 5):
        depth = worker.submit(SleepStage.N3, 0.5)
    assert depth <= SleepStageWorker.MAX_PENDING, (
        f"queue grew to {depth}, past the bound of "
        f"{SleepStageWorker.MAX_PENDING}"
    )


def test_target_failure_does_not_kill_the_worker() -> None:
    """One bad pass must not stop later stages from consolidating."""
    calls: list[SleepStage] = []

    def flaky(stage: SleepStage, dt: float) -> None:
        calls.append(stage)
        if len(calls) == 1:
            raise RuntimeError("consolidation exploded")

    worker = SleepStageWorker(flaky)
    worker.start()
    try:
        worker.submit(SleepStage.N3, 0.5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(calls) < 1:
            time.sleep(0.01)
        worker.submit(SleepStage.N3, 0.5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(calls) < 2:
            time.sleep(0.01)
        assert len(calls) == 2, (
            f"worker died on the first failure: ran {len(calls)} of 2"
        )
    finally:
        worker.stop(timeout=5)


def test_stop_is_idempotent_and_safe_before_start() -> None:
    worker = SleepStageWorker(_Recorder())
    assert worker.stop(timeout=1), "stopping an unstarted worker is fine"
    worker.start()
    assert worker.stop(timeout=5)
    assert worker.stop(timeout=1), "stopping twice is fine"
