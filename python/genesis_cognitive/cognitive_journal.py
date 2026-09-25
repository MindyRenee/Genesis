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
rotated to ``cognitive_journal.1.jsonl`` (one previous generation is
kept) so a long-lived mind cannot fill the disk.

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
            logger.debug(f"journal write failed: {e!r}")

    def record_error(self, site: str, exc: BaseException) -> None:
        """Append an error event for an exception a subsystem swallowed."""
        self.record("error", repr(exc), site=site)

    def _rotate_locked(self):
        """Rotate the current file to the single backup generation.

        Called with ``_lock`` held and ``_file`` open past the size cap.
        Returns the new open file, or None if reopening failed.
        """
        backup = self._path.with_suffix(".1.jsonl")
        try:
            self._file.close()
            self._path.replace(backup)
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
                    logger.debug(f"journal close failed: {e!r}")
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
