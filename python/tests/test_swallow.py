"""Tests for the swallowed-exception tally.

Every ``except`` in Genesis discards its error, which is correct — a
mind must not die because a ``/proc`` read failed. But the sites logged
at DEBUG and the default log level is INFO, so the failures were
invisible: a subsystem could fail on every cycle indefinitely with
nothing anywhere saying so.

The fix is a *count* rather than a transcript. These tests pin the
properties that make that trustworthy: durable, logarithmically
bounded, and honest about the current total.
"""

from __future__ import annotations

import json

import pytest

from genesis_client.swallow import (
    failing_sites,
    note_swallowed,
    reset_swallow_tallies,
    set_swallow_sink,
    swallow_count,
    swallow_report,
    total_swallowed,
)
from genesis_cognitive.infrastructure.journal import CognitiveJournal


@pytest.fixture(autouse=True)
def _clean_tallies():
    reset_swallow_tallies()
    set_swallow_sink(None)
    yield
    reset_swallow_tallies()
    set_swallow_sink(None)


def test_first_failure_is_always_recorded(tmp_path):
    """The first failure of a site is never suppressed."""
    journal = CognitiveJournal(tmp_path)
    set_swallow_sink(lambda site, count: journal.record(
        "error", site, site=site, count=count
    ))
    try:
        note_swallowed("sensor.read", OSError("boom"))
    finally:
        set_swallow_sink(None)
        journal.close()

    (event,) = [
        json.loads(line)
        for line in (tmp_path / "cognitive_journal.jsonl").read_text().splitlines()
    ]
    assert event["site"] == "sensor.read"
    assert event["count"] == 1


def test_repeat_failures_are_bounded_not_transcribed(tmp_path):
    """A site failing 10,000 times must not write 10,000 lines.

    This is the property that makes journalling every catch site
    acceptable at all: a per-second failure would otherwise write
    86,400 lines a day and bury everything worth reading.
    """
    journal = CognitiveJournal(tmp_path)
    set_swallow_sink(lambda site, count: journal.record(
        "error", site, site=site, count=count
    ))
    try:
        for _ in range(10_000):
            note_swallowed("chatty", RuntimeError("again"))
    finally:
        set_swallow_sink(None)
        journal.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "cognitive_journal.jsonl").read_text().splitlines()
    ]
    assert len(events) < 20, f"{len(events)} records is not bounded"
    # Counts double, so the record is logarithmic in the failure count.
    counts = [e["count"] for e in events]
    assert counts[0] == 1
    assert counts == sorted(counts)


def test_count_is_exact_even_when_no_record_is_written():
    """The tally is exact; only the journalled record is throttled.

    Reporting a stale number would defeat the purpose — the count is the
    telemetry, and a reader must be able to trust the live figure.
    """
    for _ in range(5_000):
        note_swallowed("s", ValueError("x"))
    assert swallow_count("s") == 5_000
    assert total_swallowed() == 5_000


def test_sites_are_counted_independently():
    """Two failing sites are two entries, not one merged total."""
    for _ in range(10):
        note_swallowed("a.one", OSError("x"))
    note_swallowed("b.two", OSError("y"))
    assert swallow_count("a.one") == 10
    assert swallow_count("b.two") == 1
    assert swallow_count("never.seen") == 0


def test_failing_sites_sorted_by_count():
    """The busiest sites come first, so the worst is visible."""
    for _ in range(3):
        note_swallowed("quiet", OSError("x"))
    for _ in range(50):
        note_swallowed("loud", OSError("x"))
    sites = failing_sites()
    assert sites[0] == ("loud", 50)
    assert failing_sites(limit=1) == [("loud", 50)]


def test_report_shape_is_display_ready():
    """Introspection gets totals and a top-sites breakdown."""
    for _ in range(7):
        note_swallowed("x", OSError("x"))
    note_swallowed("y", OSError("y"))
    report = swallow_report()
    assert report["total"] == 8
    assert report["distinct_sites"] == 2
    assert report["busiest"][0] == ("x", 7)


def test_failing_sink_does_not_raise():
    """A broken sink must not turn a swallowed error into a raised one.

    This is the critical safety property: the whole point of a catch
    site is that the failure stops there. If reporting the failure could
    itself raise, the caller's error handling would be worse than doing
    nothing.
    """

    def exploding_sink(site: str, count: int) -> None:
        raise RuntimeError("journal is on fire")

    set_swallow_sink(exploding_sink)
    note_swallowed("a", OSError("original failure"))  # must not raise
    set_swallow_sink(None)
    assert swallow_count("a") == 1


def test_no_sink_still_counts():
    """With no journal attached the tally works; only the record is absent."""
    set_swallow_sink(None)
    for _ in range(4):
        note_swallowed("detached", OSError("x"))
    assert swallow_count("detached") == 4


def test_time_based_report_keeps_the_record_current(tmp_path, monkeypatch):
    """A site failing steadily is re-reported when time passes.

    Count-based backoff alone can leave the recorded count far below the
    truth between thresholds, so a reader sees a stale number with no
    indication the site is still failing. The time bound is what makes
    the record trustworthy rather than merely bounded.
    """
    journal = CognitiveJournal(tmp_path)
    set_swallow_sink(lambda site, count: journal.record(
        "error", site, site=site, count=count
    ))
    try:
        for _ in range(9):  # up to the last doubling step
            note_swallowed("slow", OSError("x"))
        # Pretend the time bound has elapsed.
        import genesis_client.swallow as sw

        monkeypatch.setattr(sw, "_TAIL_SECONDS", -1.0)
        note_swallowed("slow", OSError("x"))
    finally:
        set_swallow_sink(None)
        journal.close()

    events = [
        json.loads(line)
        for line in (tmp_path / "cognitive_journal.jsonl").read_text().splitlines()
    ]
    assert events[-1]["count"] == 10, "the final record must carry the true count"
