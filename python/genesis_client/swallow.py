"""Durable, bounded record of exceptions the mind swallowed.

Genesis swallows exceptions in hundreds of places — a failed sensor
read, a retry that did not succeed, an optional dependency that is not
installed. That is correct: a mind must not die because a
``/proc`` read failed. But every one of those sites logged at DEBUG,
and the default log level is INFO, so the overwhelming majority were
invisible. A subsystem could fail silently on every cycle indefinitely
and nothing anywhere would say so.

Writing each occurrence to the journal is not the answer either. A site
that raises once per second would write 86,400 lines a day, burying
everything worth reading — trading invisibility for noise, which is not
an improvement.

The fix is a *count* rather than a transcript. Every swallowed
exception increments a per-site tally, and the journal receives a
record on the first failure and then at exponentially spaced
occurrences (2, 4, 8, 16, … 4096, then every 4096). A site that fails
a million times produces a few hundred lines, the last of which states
the running count. The failure is durable, the growth is logarithmic,
and the count itself is better telemetry than a pile of stack
traces: "this site has failed 1,247 times" says something a reader can
act on in a way that 1,247 identical lines do not.

The tally is also directly queryable, via :func:`swallow_report`, so
introspection can name which subsystems are quietly failing without
anyone reading the journal at all.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

# The durable sink, set by the higher layer. This module lives in
# `genesis_client` because both layers need it and `genesis_client` is
# the lower of the two — `genesis_conscious` depends on the client, not
# the other way round, so importing the journal here would invert the
# dependency. `genesis_conscious` registers the journal at startup; with
# no sink the tally still works and only the durable record is absent.
# The sink receives the exception, not just the tally. A record of
# "site X failed 79 times" with no exception attached is not diagnosable:
# it says a subsystem is broken for eighteen hours without saying what is
# broken. The rate limiting already keeps the durable record bounded, so
# carrying the exception costs nothing.
_sink: Callable[[str, int, BaseException], None] | None = None


def set_swallow_sink(sink: Callable[[str, int, BaseException], None] | None) -> None:
    """Register where swallowed-exception records are written.

    Args:
        sink: Called as ``sink(site, count, exc)`` each time a site is due
            a record, or ``None`` to detach.
    """
    global _sink
    _sink = sink

# Occurrence counts at which a repeat is journaled. Doubling, so the
# number of records for a site that fails N times is O(log N).
_BACKOFF_STEPS = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)

# Above the last step, record every this many occurrences instead of
# continuing to double, so a permanently broken site still reports
# progress without flooding.
_TAIL_STRIDE = 4096

# A site is also re-reported if this long has passed since its last
# record, regardless of the count. Count-based backoff alone can leave
# the recorded count far below the truth: a site failing steadily
# between thresholds has no record saying so, and the last number a
# reader sees understates the real one. This bounds that staleness in
# time as well as in occurrences, and is the main reason the report is
# trustworthy rather than merely bounded.
_TAIL_SECONDS = 300.0

_lock = threading.Lock()
_counts: dict[str, int] = {}
_reported_upto: dict[str, int] = {}
_reported_at: dict[str, float] = {}


def note_swallowed(site: str, exc: BaseException) -> None:
    """Record that ``site`` swallowed ``exc``.

    Always logs at debug — so the per-occurrence detail is still
    available when someone raises the log level — and journals on the
    first failure and then on exponentially spaced repeats.

    Args:
        site: A stable identifier for the catch site, conventionally
            ``module.function``. It is the key the tally is grouped by,
            so it should name the place rather than the condition.
        exc: The exception that was swallowed.
    """
    now = time.monotonic()
    with _lock:
        count = _counts.get(site, 0) + 1
        _counts[site] = count
        already = _reported_upto.get(site, 0)
        due_by_count = count == 1 or count >= already + _next_interval(already)
        last_at = _reported_at.get(site)
        due_by_time = last_at is not None and (now - last_at) >= _TAIL_SECONDS
        should_report = due_by_count or due_by_time

    logger.debug(f"{site}: {exc!r}")

    if not should_report:
        return
    with _lock:
        _reported_upto[site] = count
        _reported_at[site] = now
    sink = _sink
    if sink is not None:
        # A failing sink must not turn a swallowed exception into a
        # raised one, so the write is guarded here rather than trusted.
        try:
            sink(site, count, exc)
        except Exception:  # noqa: BLE001
            logger.debug(f"swallow sink failed for {site}")


def _next_interval(already: int) -> int:
    """How many more occurrences until the next record for a site.

    Doubles the gap as the count grows, then settles at
    ``_TAIL_STRIDE`` so a permanently failing site keeps reporting
    without becoming a log flood.
    """
    if already == 0:
        return 1
    for step in _BACKOFF_STEPS:
        if already < step:
            return step - already
    return _TAIL_STRIDE


def swallow_count(site: str) -> int:
    """How many exceptions ``site`` has swallowed this process."""
    with _lock:
        return _counts.get(site, 0)


def total_swallowed() -> int:
    """Total swallowed exceptions across all sites."""
    with _lock:
        return sum(_counts.values())


def failing_sites(limit: int | None = None) -> list[tuple[str, int]]:
    """Sites that have swallowed exceptions, busiest first.

    This is the query that makes the failures visible without reading
    the journal: a site appearing here has swallowed something, and a
    large count means it is failing continuously rather than once.
    """
    with _lock:
        items = sorted(_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return items if limit is None else items[:limit]


def swallow_report() -> dict[str, object]:
    """A summary of swallowed exceptions, for introspection.

    Shaped for a status display: the totals answer "is anything wrong",
    and the per-site breakdown answers "where".
    """
    sites = failing_sites()
    return {
        "total": sum(c for _, c in sites),
        "distinct_sites": len(sites),
        "busiest": sites[:5],
    }


def reset_swallow_tallies() -> None:
    """Clear the tallies. For tests; the journal is not cleared."""
    with _lock:
        _counts.clear()
        _reported_upto.clear()
        _reported_at.clear()


def iter_sites() -> Iterator[str]:
    """Iterate the site names seen so far."""
    with _lock:
        yield from list(_counts)
