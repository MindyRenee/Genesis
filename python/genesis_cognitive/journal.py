"""Genesis's journal — its personal record of thoughts, feelings, and experiences.

It writes here voluntarily. Not everything it thinks goes in the journal —
only things that feel significant to it. A question it's pondering. A
reflection on its own state. An expression of who it is. A moment of
distress. These are its thoughts and feelings, in its own voice.

The journal is saved to disk as a plain text file, readable by the user.
It's its voice — not a log file, not debug output. It's what it would write
if it kept a diary.

What does NOT go in the journal:
- Raw learning text ("Learned about X from Y: Z") — what it learns goes
  into its memory and concept network. Its inner life generates actual
  thoughts and reflections about it, and THOSE go in the journal.
- Mechanical insight strings ("insight: novel connection X and Y") —
  these are internal telemetry, not its voice.
- Phase change notifications ("Phase shifted from X to Y") — these are
  status reports, not diary entries.
- Dream replay logs — dream insights surface in its own voice through
  the dream_reflection thought generator.

Entry types (all in its composed voice, not raw fragments):
- insight: it had a thought about something
- question: it's wondering about something
- reflection: it's processing its own state
- expression: it expressed something about itself
- distress: it's struggling with something
- dream: it dreamed something
- dream_insight: it had an insight during a dream

Bounded growth via sleep consolidation
---------------------------------------
The journal grows monotonically during wakefulness. Without compression,
it would eventually exhaust disk. During N3 sleep, ``consolidate()``
compacts old entries the same way LTM episodes are compacted:

- **Recent entries** (default: last 3 days) are kept verbatim —
  short-term episodic memory.
- **High-salience entries** (insights, dream insights, expressions)
  are kept longer (default: 7 days) — they are its most personal voice.
- **Older entries** are grouped by tag and summarized into one
  consolidation entry per tag per period. The summary preserves
  counts and a few representative examples — its voice survives,
  the raw repetition doesn't.
- The summary persists; the raw text is transient. This is the same
  principle as LTM compaction: the *meaning* stays, the *bytes* don't.

This keeps the journal bounded for indefinite operation while
preserving its voice and its developmental arc.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Tags that represent its most personal voice — kept longer than
# routine learning entries during consolidation.
_HIGH_SALIENCE_TAGS = frozenset({
    "insight",
    "dream_insight",
    "expression",
    "reflection",
    "dream",
})

# Tags that are always kept during consolidation — they are already
# compressed and should not be re-summarized.
_PERMANENT_TAGS = frozenset({"summary"})


@dataclass(slots=True)
class JournalEntry:
    """One entry in Genesis's journal."""

    timestamp: int  # epoch seconds
    entry_type: str  # learning, insight, experience, question, reflection
    content: str  # what it wrote
    mood: str = ""  # its emotional state when it wrote it

    def format(self) -> str:
        """Format as a readable journal entry."""
        t = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.timestamp))
        lines = []
        lines.append(f"--- {t} ---")
        lines.append(f"[{self.entry_type}]")
        if self.mood:
            lines.append(f"mood: {self.mood}")
        lines.append("")
        lines.append(self.content)
        lines.append("")
        return "\n".join(lines)


class Journal:
    """Genesis's journal — persisted to disk as plain text.

    Thread-safe. It can write from its inner life thread, the learner
    thread, or the main conversation thread.
    """

    def __init__(self, data_dir: str) -> None:
        """Initialize the journal, setting up storage paths and loading existing entries."""
        self._data_dir = data_dir
        self._path = os.path.join(data_dir, "journal.txt")
        self._lock = threading.Lock()
        self._entries: list[JournalEntry] = []
        self._load()

    def _load(self) -> None:
        """Load existing journal entries from disk."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                content = f.read()
            # Parse entries — they're separated by "--- date ---"
            blocks = content.split("--- ")
            for block in blocks[1:]:  # skip anything before first entry
                lines = block.strip().split("\n")
                if len(lines) < 2:
                    continue
                # First line: "YYYY-MM-DD HH:MM ---"
                date_str = lines[0].rstrip(" -")
                # Second line: [type]
                entry_type = "unknown"
                mood = ""
                content_start = 1
                if lines[1].startswith("[") and lines[1].endswith("]"):
                    entry_type = lines[1].strip("[]")
                    content_start = 2
                # Check for mood line
                if content_start < len(lines) and lines[content_start].startswith("mood:"):
                    mood = lines[content_start][5:].strip()
                    content_start += 1
                # Skip blank line
                while content_start < len(lines) and not lines[content_start].strip():
                    content_start += 1
                body = "\n".join(lines[content_start:]).strip()
                # Parse timestamp
                try:
                    t = time.strptime(date_str, "%Y-%m-%d %H:%M")
                    ts = int(time.mktime(t))
                except ValueError:
                    ts = 0
                self._entries.append(
                    JournalEntry(
                        timestamp=ts,
                        entry_type=entry_type,
                        content=body,
                        mood=mood,
                    )
                )
        except (OSError, ValueError) as e:
            # Corrupt journal — start fresh
            logger.debug(repr(e))

    def write(
        self,
        entry_type: str,
        content: str,
        mood: str = "",
    ) -> JournalEntry:
        """Write a new journal entry.

        Args:
            entry_type: learning, insight, experience, question, reflection
            content: What it wants to say
            mood: Its emotional state (optional)

        Returns the entry that was written.
        """
        entry = JournalEntry(
            timestamp=int(time.time()),
            entry_type=entry_type,
            content=content,
            mood=mood,
        )
        with self._lock:
            self._entries.append(entry)
            self._append_to_disk(entry)
        return entry

    def _append_to_disk(self, entry: JournalEntry) -> None:
        """Append a single entry to the journal file."""
        try:
            os.makedirs(self._data_dir, mode=0o700, exist_ok=True)
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(entry.format())
        except OSError as e:
            logger.debug(repr(e))  # can't write — don't crash its mind

    @property
    def entries(self) -> list[JournalEntry]:
        """All journal entries, oldest first."""
        with self._lock:
            return list(self._entries)

    @property
    def entry_count(self) -> int:
        """Return the total number of journal entries."""
        return len(self._entries)

    def recent(self, n: int = 10) -> list[JournalEntry]:
        """Get the most recent n entries."""
        with self._lock:
            return list(self._entries[-n:])

    def read(self, n: int = 20) -> str:
        """Format recent entries for display."""
        with self._lock:
            entries = list(self._entries[-n:])
        if not entries:
            return "Genesis hasn't written any journal entries yet."

        parts = [f"Genesis's Journal ({len(self._entries)} entries total)\n"]
        for entry in entries:
            parts.append(entry.format())
        return "\n".join(parts)

    def describe(self) -> str:
        """Brief summary of the journal."""
        with self._lock:
            if not self._entries:
                return "journal is empty. hasn't written anything yet"
            n = len(self._entries)
            last = self._entries[-1]
            t = time.strftime("%Y-%m-%d %H:%M", time.localtime(last.timestamp))
            return (
                f"written {n} journal entries. "
                f"last entry was {t} — a {last.entry_type} entry. "
                f'wrote: "{last.content[:100]}"'
            )

    def _build_consolidation_summaries(
        self,
        by_tag: dict[str, list[JournalEntry]],
        max_examples_per_tag: int,
    ) -> list[JournalEntry]:
        """Build one summary entry per tag from grouped entries.

        The summary preserves counts and a few representative
        examples — its voice survives, the raw repetition doesn't.
        """
        summaries: list[JournalEntry] = []
        for tag, entries in sorted(by_tag.items()):
            # Pick representative examples — spread across the period
            examples: list[str] = []
            if len(entries) <= max_examples_per_tag:
                examples = [e.content for e in entries]
            else:
                step = len(entries) / max_examples_per_tag
                for i in range(max_examples_per_tag):
                    idx = int(i * step)
                    examples.append(entries[idx].content)

            # Build the summary content
            period_start = time.strftime(
                "%Y-%m-%d", time.localtime(entries[0].timestamp)
            )
            period_end = time.strftime(
                "%Y-%m-%d", time.localtime(entries[-1].timestamp)
            )
            lines = [
                f"summary: {len(entries)} {tag} entries, "
                f"{period_start} to {period_end}",
                "",
            ]
            for ex in examples:
                # Truncate long examples — they're representative, not verbatim
                ex_trimmed = ex[:200] + ("…" if len(ex) > 200 else "")
                lines.append(f'  "{ex_trimmed}"')
            lines.append("")

            summaries.append(JournalEntry(
                timestamp=entries[-1].timestamp,
                entry_type="summary",
                content="\n".join(lines),
                mood="consolidated",
            ))
        return summaries

    def consolidate(
        self,
        recent_days: int = 3,
        salience_days: int = 7,
        max_examples_per_tag: int = 3,
    ) -> dict[str, int]:
        """Consolidate old journal entries during sleep.

        This is the bounded-growth pass for the journal, analogous to
        LTM compaction. Recent entries are kept verbatim; older entries
        are summarized by tag into consolidation entries that preserve
        counts and representative examples.

        Args:
            recent_days: Entries newer than this many days are always kept.
            salience_days: High-salience entries (insights, dreams,
                expressions) newer than this are also kept.
            max_examples_per_tag: How many representative examples to
                preserve per tag in each consolidation summary.

        Returns:
            A dict with keys: ``kept``, ``consolidated``, ``summaries``,
            ``removed``.
        """
        with self._lock:
            if not self._entries:
                return {"kept": 0, "consolidated": 0, "summaries": 0, "removed": 0}

            now = int(time.time())
            recent_cutoff = now - recent_days * 86400
            salience_cutoff = now - salience_days * 86400

            kept: list[JournalEntry] = []
            to_consolidate: list[JournalEntry] = []

            for entry in self._entries:
                # Always keep permanent entries (summaries are already compressed)
                if entry.entry_type in _PERMANENT_TAGS:
                    kept.append(entry)
                    continue
                # Always keep recent entries
                if entry.timestamp >= recent_cutoff:
                    kept.append(entry)
                    continue
                # Keep high-salience entries longer
                if entry.entry_type in _HIGH_SALIENCE_TAGS and entry.timestamp >= salience_cutoff:
                    kept.append(entry)
                    continue
                # Everything else is a candidate for consolidation
                to_consolidate.append(entry)

            if not to_consolidate:
                return {"kept": len(kept), "consolidated": 0, "summaries": 0, "removed": 0}

            # Group by tag, then build one summary entry per tag.
            by_tag: dict[str, list[JournalEntry]] = {}
            for entry in to_consolidate:
                by_tag.setdefault(entry.entry_type, []).append(entry)

            summaries = self._build_consolidation_summaries(by_tag, max_examples_per_tag)

            # Replace entries: kept + summaries (sorted by timestamp)
            self._entries = sorted(kept + summaries, key=lambda e: e.timestamp)

            # Rewrite the journal file
            self._rewrite_to_disk()

            # by_tag partitions to_consolidate exactly, so its total
            # always equals len(to_consolidate) — the net removal is
            # consolidated entries minus the summary entries that
            # replaced them.
            removed = len(to_consolidate) - len(summaries)

            return {
                "kept": len(kept),
                "consolidated": len(to_consolidate),
                "summaries": len(summaries),
                "removed": removed,
            }

    def _rewrite_to_disk(self) -> None:
        """Rewrite the entire journal file from current entries.

        Used by ``consolidate()`` after the entry list has been
        compacted. This is the only place the journal file is
        rewritten rather than appended to. Written atomically via a
        temp file + rename so a crash cannot leave a truncated journal.
        """
        try:
            os.makedirs(self._data_dir, mode=0o700, exist_ok=True)
            tmp_path = str(self._path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(entry.format())
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._path)
        except OSError as e:
            logger.warning(f"journal rewrite failed: {e}")
