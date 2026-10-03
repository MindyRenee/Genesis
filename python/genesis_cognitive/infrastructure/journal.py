"""Cognitive journal — a durable append-only record of inner life.

Genesis's spontaneous thoughts, dream content, volition actions, and
runtime errors are produced in background threads and held in bounded
in-memory deques — when the process exits (or a deque rolls over), they
are gone. The journal is the black box: every event emitted through
``Mind._emit_live_thought`` is appended to ``cognitive_journal.jsonl``
in the data dir, one JSON object per line, so what it thought, dreamt,
did, and hit survives restarts and can be correlated with later
failures.

## Event schema

Each line is a JSON object::

    {"ts": 1758…, "kind": "dream", "text": "…", "trigger": "rem", …}

- ``ts`` — Unix timestamp in milliseconds
- ``kind`` — the emitted event kind (``thought``, ``dream``,
  ``learning``, ``bug_scan``, ``sleep``, ``error``, …; train-of-thought
  steps carry a ``:N`` suffix)
- ``text`` — the composed content (emergent language, recorded as
  generated — the journal never invents words for it)
- extra fields — thought metadata (``trigger``, ``chain_id``,
  ``is_lucid``) or error metadata (``site``)

## Failure discipline

The journal must never crash its mind: ``record`` swallows its own
errors to a debug log. When the file exceeds ``_MAX_BYTES`` it is
rotated, keeping ``_GENERATIONS`` previous files
(``cognitive_journal.1.jsonl`` … ``.N.jsonl``) so a long-lived mind
cannot fill the disk. More than one generation is kept because a single
backup is overwritten by the *next* rotation: a burst of errors early
in a long session could be replaced before anyone had looked at it,
which is exactly when the record matters most.

``record_error`` is a module-level convenience for catch sites in
subsystems that hold no Mind reference (spatial practice persistence,
the volition action boundary). It writes to whichever journal ``Mind``
activated at init and no-ops when none is active.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)

JOURNAL_FILENAME = "cognitive_journal.jsonl"


class CognitiveJournal:
    """Append-only JSONL journal for cognitive events.

    A single background file handle is shared by every producer thread;
    ``record`` serializes one JSON line under a lock. The file is opened
    line-buffered so each event is flushed to the OS immediately —
    readable by ``tail -f`` and durable across process exit.
    """

    # Beyond this size the file is rotated (one generation kept).
    # At ~200 B/event this holds well over a hundred thousand events.
    _MAX_BYTES = 32 * 1024 * 1024

    # Rotated generations retained. One was not enough: the next
    # rotation overwrites it, so a burst of errors early in a long
    # session could vanish before anyone read it. Bounded at
    # (_GENERATIONS + 1) x _MAX_BYTES = 128 MB of journal on disk.
    _GENERATIONS = 3

    def __init__(self, data_dir: str | Path) -> None:
        self._path = Path(data_dir) / JOURNAL_FILENAME
        self._lock = threading.Lock()
        self._file = None
        try:
            self._file = self._path.open("a", encoding="utf-8", buffering=1)
        except OSError as e:
            logger.warning(f"cognitive journal unavailable at {self._path}: {e}")

    @property
    def path(self) -> Path:
        """Where events are being written."""
        return self._path

    def record(self, kind: str, content: str, **fields: object) -> None:
        """Append one event line. Never raises."""
        f = self._file
        if f is None or f.closed:
            return
        try:
            event = {
                "ts": int(time.time() * 1000),
                "kind": kind,
                "text": content,
            }
            event.update(fields)
            line = json.dumps(event, ensure_ascii=False)
            with self._lock:
                f = self._file
                if f is None or f.closed:
                    return
                if f.tell() > self._MAX_BYTES:
                    f = self._rotate_locked()
                    if f is None:
                        return
                f.write(line + "\n")
        except Exception as e:  # noqa: BLE001
            note_swallowed(
                "genesis_cognitive.infrastructure.journal.record",
                e,
            )

    def record_error(self, site: str, exc: BaseException) -> None:
        """Append an error event for an exception a subsystem swallowed."""
        self.record("error", repr(exc), site=site)

    def _generation_path(self, n: int) -> Path:
        """Path of the *n*-th rotated generation (1 = most recent).

        Built by string manipulation on the full filename rather than
        ``with_suffix``: the journal is named ``cognitive_journal.jsonl``
        and the rotated files ``cognitive_journal.1.jsonl``, so
        ``with_suffix(".1.jsonl")`` happens to work for exactly that one
        name and silently replaces the final suffix for any other
        (``.jsonl.gz`` → ``cognitive_journal.1.gz``). Inserting before
        the extension is correct for any name.
        """
        name = self._path.name
        stem, dot, ext = name.rpartition(".")
        if not dot:
            return self._path.with_name(f"{name}.{n}")
        return self._path.with_name(f"{stem}.{n}.{ext}")

    def _rotate_locked(self):
        """Rotate the journal, retaining ``_GENERATIONS`` past files.

        Called with ``_lock`` held and ``_file`` open past the size cap.
        The oldest generation is dropped so total disk use stays bounded
        at ``_GENERATIONS + 1`` files. Returns the new open file, or None
        if reopening failed.
        """
        try:
            self._file.close()
            # Shift each generation up by one, oldest first so it is
            # never overwritten before being moved.
            for n in range(self._GENERATIONS, 0, -1):
                src = self._generation_path(n)
                if not src.exists():
                    continue
                if n == self._GENERATIONS:
                    src.unlink()  # the oldest falls off the end
                else:
                    src.replace(self._generation_path(n + 1))
            self._path.replace(self._generation_path(1))
            self._file = self._path.open("a", encoding="utf-8", buffering=1)
            return self._file
        except OSError as e:
            logger.warning(f"cognitive journal rotation failed: {e}")
            self._file = None
            return None

    def close(self) -> None:
        """Flush and close the journal. Subsequent records are dropped."""
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError as e:
                    note_swallowed(
                        "genesis_cognitive.infrastructure.journal.close",
                        e,
                    )
                self._file = None


# The journal Mind activated — lets subsystems without a Mind
# reference (spatial persistence helpers, the volition action
# boundary) record swallowed exceptions without constructor plumbing.
_active: CognitiveJournal | None = None


def set_active(journal: CognitiveJournal | None) -> None:
    """Set the process-wide journal that module-level helpers write to."""
    global _active
    _active = journal


def record_error(site: str, exc: BaseException) -> None:
    """Record a swallowed exception to the active journal, if any."""
    journal = _active
    if journal is not None:
        journal.record_error(site, exc)


def record_event(kind: str, content: str, **fields: object) -> None:
    """Record a state transition to the active journal, if any.

    The companion to :func:`record_error` for the other kind of
    currently-invisible event: not a swallowed failure but an
    unrecorded *change*. Transitions matter precisely because nothing
    else records them — a sleep that consolidates nothing still
    happened, and without this the number and duration of sleep
    episodes cannot be reconstructed from any durable state.
    """
    journal = _active
    if journal is not None:
        journal.record(kind, content, **fields)
