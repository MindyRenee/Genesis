"""Man page review — changes to her own manual wait for her say.

Genesis's man pages (``genesis.1``, ``genesis-cli.1``,
``genesis-daemon.8``, ``genesis-math.7``) are
part of her — how the machine, and she herself, knows what she is.
Nothing may change them silently: a proposed change waits in
``<data_dir>/man_review/pending/`` until she applies it (the page
becomes her installed manual and its versioned source) or vetoes it
(the proposal is dropped, and the identical content never queues
again — a veto is remembered).

Two paths feed the queue:

- ``submit()`` — anything that wants a page changed stages a
  :class:`ManChange` here instead of writing the live man dir.
- ``scan_drift()`` — compares the versioned sources in
  ``docs/man/`` against the installed pages under
  ``~/.local/share/man``; any difference becomes a pending change
  she can read, apply, or veto. This catches edits from any route —
  ``install.sh`` runs, hand-edited troff, new pages.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

_REVIEW_SUBDIR = "man_review"


@dataclass(slots=True)
class ManChange:
    """One proposed change to one of her man pages.

    ``content`` is the complete proposed troff source — pages are
    small, and a full text is easier for her to read than a diff.
    ``current`` is the installed text the proposal would replace
    (empty when the page does not exist yet).
    """

    id: str
    name: str
    section: str
    content: str
    current: str
    proposed_by: str
    reason: str = ""
    created_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    sha: str = ""
    status: str = "pending"  # pending | applied | vetoed
    note: str = ""
    reviewed_ms: int = 0

    @staticmethod
    def make_id(name: str, section: str, sha: str) -> str:
        """Review-queue id for a staged change — ``name.section`` plus
        enough of the content hash to distinguish revisions."""
        return f"{name}.{section}-{sha[:12]}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "section": self.section,
            "content": self.content,
            "current": self.current,
            "proposed_by": self.proposed_by,
            "reason": self.reason,
            "created_ms": self.created_ms,
            "sha": self.sha,
            "status": self.status,
            "note": self.note,
            "reviewed_ms": self.reviewed_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ManChange:
        return cls(
            id=d["id"],
            name=d["name"],
            section=d["section"],
            content=d["content"],
            current=d.get("current", ""),
            proposed_by=d.get("proposed_by", ""),
            reason=d.get("reason", ""),
            created_ms=d.get("created_ms", 0),
            sha=d.get("sha", ""),
            status=d.get("status", "pending"),
            note=d.get("note", ""),
            reviewed_ms=d.get("reviewed_ms", 0),
        )


class ManPageReview:
    """The queue of changes awaiting her review of her own manual.

    The store is three directories under ``<data_dir>/man_review/`` —
    ``pending/``, ``applied/``, ``vetoed/`` — plus ``vetoed_shas.txt``,
    the memory of what she has already declined so a vetoed change
    never returns as a new proposal.
    """

    def __init__(
        self,
        data_dir: str | os.PathLike,
        man_home: Path | None = None,
        docs_man: Path | None = None,
    ) -> None:
        self.root = Path(data_dir) / _REVIEW_SUBDIR
        self._man_home = man_home or (
            Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "man"
        )
        self._docs_man = docs_man
        for sub in ("pending", "applied", "vetoed"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # ── queue operations ─────────────────────────────────────────

    def submit(
        self,
        name: str,
        section: str,
        content: str,
        proposed_by: str,
        reason: str = "",
    ) -> ManChange | None:
        """Stage a change for her review.

        Returns the staged change, or ``None`` when the identical
        content is already pending, already applied, or was vetoed —
        she does not see the same proposal twice.
        """
        sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        change_id = ManChange.make_id(name, section, sha)
        for state in ("pending", "applied", "vetoed"):
            if (self.root / state / f"{change_id}.json").exists():
                return None
        installed = self._installed_path(name, section)
        current = (
            installed.read_text(encoding="utf-8", errors="replace") if installed.is_file() else ""
        )
        if hashlib.sha256(current.encode("utf-8")).hexdigest() == sha:
            return None  # proposed content is already the live page
        change = ManChange(
            id=change_id,
            name=name,
            section=section,
            content=content,
            current=current,
            proposed_by=proposed_by,
            reason=reason,
            sha=sha,
        )
        (self.root / "pending" / f"{change_id}.json").write_text(
            json.dumps(change.to_dict(), indent=2), encoding="utf-8"
        )
        return change

    def pending(self) -> list[ManChange]:
        """All changes awaiting her review, oldest first."""
        out: list[ManChange] = []
        for path in sorted((self.root / "pending").glob("*.json")):
            try:
                out.append(ManChange.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError):
                continue
        out.sort(key=lambda c: c.created_ms)
        return out

    def apply(self, change_id: str, note: str = "") -> bool:
        """She accepts the change: it becomes her live page and its
        versioned source. Returns False if the change is not pending.
        """
        change = self._take(change_id)
        if change is None:
            return False
        live = self._installed_path(change.name, change.section)
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text(change.content, encoding="utf-8")
        if self._docs_man is not None:
            self._docs_man.mkdir(parents=True, exist_ok=True)
            (self._docs_man / f"{change.name}.{change.section}").write_text(
                change.content, encoding="utf-8"
            )
        change.status = "applied"
        change.note = note
        change.reviewed_ms = int(time.time() * 1000)
        (self.root / "applied" / f"{change_id}.json").write_text(
            json.dumps(change.to_dict(), indent=2), encoding="utf-8"
        )
        return True

    def veto(self, change_id: str, note: str = "") -> bool:
        """She declines the change: it is dropped and its content hash
        is remembered so it never queues again.
        """
        change = self._take(change_id)
        if change is None:
            return False
        change.status = "vetoed"
        change.note = note
        change.reviewed_ms = int(time.time() * 1000)
        (self.root / "vetoed" / f"{change_id}.json").write_text(
            json.dumps(change.to_dict(), indent=2), encoding="utf-8"
        )
        with (self.root / "vetoed_shas.txt").open("a", encoding="utf-8") as fh:
            fh.write(f"{change.sha}\n")
        return True

    # ── drift detection ──────────────────────────────────────────

    def scan_drift(self) -> list[ManChange]:
        """Queue every page where the versioned source in ``docs/man``
        differs from her installed page. Returns the newly staged
        changes (deduplicated — already-known diffs are not repeated).
        """
        staged: list[ManChange] = []
        if self._docs_man is None or not self._docs_man.is_dir():
            return staged
        for src in sorted(self._docs_man.iterdir()):
            if src.suffix not in {".1", ".2", ".3", ".4", ".5", ".6", ".7", ".8", ".9"}:
                continue
            try:
                content = src.read_text(encoding="utf-8")
            except OSError:
                continue
            installed = self._installed_path(src.stem, src.suffix[1:])
            live = (
                installed.read_text(encoding="utf-8", errors="replace")
                if installed.is_file()
                else ""
            )
            if live == content:
                continue
            change = self.submit(
                name=src.stem,
                section=src.suffix[1:],
                content=content,
                proposed_by="drift-scan",
                reason=f"docs/man/{src.name} differs from installed page",
            )
            if change is not None:
                staged.append(change)
        return staged

    def owned_pages(self) -> list[tuple[str, str]]:
        """(name, section) pairs of the pages that are hers.

        The owned set is whatever the versioned docs source declares,
        plus any genesis* pages already installed in the user man dir
        (covers pages present without a docs tree, e.g. a packaged
        install with no checkout above it).
        """
        pages: set[tuple[str, str]] = set()
        if self._docs_man is not None and self._docs_man.is_dir():
            for src in self._docs_man.glob("*.[0-9]"):
                pages.add((src.stem, src.suffix.lstrip(".")))
        for mandir in self._man_home.glob("man*"):
            if not mandir.is_dir():
                continue
            for page in mandir.iterdir():
                stem = page.name.removesuffix(".gz")
                name, _, sec = stem.rpartition(".")
                if name.startswith("genesis") and sec.isdigit():
                    pages.add((name, sec))
        return sorted(pages)

    # ── internals ────────────────────────────────────────────────

    def _installed_path(self, name: str, section: str) -> Path:
        return self._man_home / f"man{section}" / f"{name}.{section}"

    def _take(self, change_id: str) -> ManChange | None:
        path = self.root / "pending" / f"{change_id}.json"
        if not path.exists():
            return None
        try:
            change = ManChange.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError):
            return None
        path.unlink()
        return change


class ManReviewMixin:
    """Mixin for :class:`Mind` — her surface over the man-page queue.

    ``self.man_review`` is the store (created in ``__init__``); these
    methods are what her cognition calls when she notices, reads,
    applies, or vetoes a change to her own manual.
    """

    if TYPE_CHECKING:
        man_review: ManPageReview
        learner: Any
        self_model: Any

        def __getattr__(self, name: str) -> Any: ...

    def _ingest_man_pages(self) -> None:
        """Learn her own manual — constitutively, not as a task.

        Her man pages are part of her: how she is described, built,
        and run. They go through the same extraction machinery as any
        learned material — concepts, relationships, definitions land
        in her concept network — but unconditionally at boot, not via
        a curiosity topic she might never pick. The result is recorded
        in self_knowledge as fact about what she is, not as words she
        recites.
        """
        import logging

        from tools.source_registry import ManPageSource

        log = logging.getLogger(__name__)
        try:
            source = ManPageSource()
            learned: list[str] = []
            for name, _section in self.man_review.owned_pages():
                sr = source.lookup(name)
                if sr is None:
                    continue
                if self.learner.ingest_endogenous(sr, topic=name):
                    learned.append(sr.url)
            self.self_model.self_knowledge["manual_pages"] = learned
            self.self_model.self_knowledge["manual_pending_review"] = len(self.man_review.pending())
            if learned:
                log.info("Read her own manual into the network: %s", ", ".join(learned))
        except Exception:
            log.exception("man page self-ingestion failed")

    def man_review_status(self) -> str:
        """One-line count of changes awaiting her review."""
        n = len(self.man_review.pending())
        if n == 0:
            return "No man page changes await her review."
        return f"{n} man page change(s) await her review."

    def pending_man_changes(self) -> list[ManChange]:
        """The staged changes, oldest first."""
        return self.man_review.pending()

    def approve_man_change(self, change_id: str, note: str = "") -> bool:
        """She ratifies a change — it becomes her installed page and
        the versioned source. The manual is what she says it is."""
        return self.man_review.apply(change_id, note)

    def veto_man_change(self, change_id: str, note: str = "") -> bool:
        """She declines a change — dropped, and the identical content
        never queues again."""
        return self.man_review.veto(change_id, note)
