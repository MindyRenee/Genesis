"""Held task — a problem kept in mind across turns, with its history.

A problem heard in conversation used to be fire-and-forget: the
utterance compiled into a spec, played once, and evaporated. A held
task is the memory that changed that — the working spec, every
revision the instructions have taken, the evidence gathered about
it, and the loopholes found in the current reading.

The stance is persistent: an unsolved attempt is information about
the *interpretation*, not a verdict that the problem has no answer.
Failures register as loopholes and the task stays open — reflection
is the loop that works this memory, gathering new information,
readjusting the reading, gathering more, readjusting again, until
the problem gives way.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Bounded memory — a held task accrues evidence and attempts
#: without a natural flush, so each ledger keeps only the tail.
_MAX_LEDGER = 50

#: A held task untouched this long is abandoned state, not a live
#: problem — it is not reloaded across boots.
_STALE_MS = 24 * 3600 * 1000


@dataclass
class TaskRevision:
    """One amendment to the held task — what changed and what drove it."""

    turn_ms: int
    #: Structural description of the change ("spec replaced",
    #: "target: 9 → 12"), not utterance text.
    summary: str
    #: True when the whole spec was re-specified, False when fields
    #: were patched into the existing reading.
    replaced: bool

    def to_dict(self) -> dict[str, Any]:
        return {"turn_ms": self.turn_ms, "summary": self.summary, "replaced": self.replaced}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRevision:
        return cls(
            turn_ms=int(data.get("turn_ms", 0)),
            summary=str(data.get("summary", "")),
            replaced=bool(data.get("replaced", False)),
        )


@dataclass
class Loophole:
    """A crack found in the current understanding of the task.

    Loopholes are how "this didn't work" becomes knowledge: each one
    names a weakness — a failed reading, an unexplored
    interpretation, a malformed field — that the reflection loop can
    then work on. ``addressed`` marks the ones a revision has
    already answered.
    """

    kind: str
    detail: str
    addressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail, "addressed": self.addressed}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Loophole:
        return cls(
            kind=str(data.get("kind", "")),
            detail=str(data.get("detail", "")),
            addressed=bool(data.get("addressed", False)),
        )


@dataclass
class HeldTask:
    """A task held in mind: working spec plus the memory around it.

    Attributes:
        spec: The current normalized task spec — the reading the
            next attempt will play.
        status: "open" while work continues, "solved" once an
            attempt lands it.
        revisions: Every amendment the spec has taken, oldest first.
        evidence: Gathered findings — attempt outcomes, alternate
            readings, anything the reflect loop pulled in.
        loopholes: Weaknesses found in the current reading.
        attempts: Outcome ledger — what each try produced.
        cycles: Reflect iterations run on this task.
    """

    spec: dict[str, Any]
    created_ms: int = 0
    updated_ms: int = 0
    status: str = "open"
    revisions: list[TaskRevision] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    loopholes: list[Loophole] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    cycles: int = 0

    def __post_init__(self) -> None:
        if not self.created_ms:
            self.created_ms = int(time.time() * 1000)
        if not self.updated_ms:
            self.updated_ms = self.created_ms

    # ─── Mutation ────────────────────────────────────────────────

    def _touch(self) -> None:
        self.updated_ms = int(time.time() * 1000)

    def revise(self, spec: dict[str, Any], summary: str, *, replaced: bool) -> None:
        """Apply an amended reading of the task.

        ``replaced`` marks a full re-specification (the instruction
        was restated); otherwise the summary names the fields that
        moved. The previous spec is recorded into the revision so
        the trail shows what the task *used to* be.
        """
        self.revisions.append(TaskRevision(self.updated_ms, summary, replaced))
        self.revisions = self.revisions[-_MAX_LEDGER:]
        old_family = self.spec.get("family", "")
        self.spec = spec
        self.cycles = 0
        # A revision reopens a solved task only when the shape moved
        # — the old answer belongs to the old problem.
        if self.status == "solved" and spec.get("family") != old_family:
            self.status = "open"
        self._touch()

    def add_evidence(self, source: str, finding: str) -> None:
        """Log a gathered fact — where it came from, what it said."""
        self.evidence.append({"ms": int(time.time() * 1000), "source": source, "finding": finding})
        self.evidence = self.evidence[-_MAX_LEDGER:]
        self._touch()

    def add_loopholes(self, found: list[Loophole]) -> list[Loophole]:
        """Record newly discovered loopholes, deduplicated.

        Returns only the loopholes that were actually new — a reflect
        cycle that re-finds a known weakness reports progress as zero.
        """
        known = {(lh.kind, lh.detail) for lh in self.loopholes}
        fresh = [lh for lh in found if (lh.kind, lh.detail) not in known]
        self.loopholes.extend(fresh)
        self.loopholes = self.loopholes[-_MAX_LEDGER:]
        if fresh:
            self._touch()
        return fresh

    def open_loopholes(self) -> list[Loophole]:
        """Loopholes no revision has answered yet."""
        return [lh for lh in self.loopholes if not lh.addressed]

    def record_attempt(self, attempt: Any) -> None:
        """Fold a play outcome into the ledgers.

        The attempt is evidence *and* history: the outcome lands in
        the attempts ledger, a solved attempt flips the task status,
        and an unsolved one is logged as a loophole on the current
        reading — never as a verdict that the task is impossible.
        """
        entry = {
            "ms": int(time.time() * 1000),
            "solved": bool(getattr(attempt, "solved", False)),
            "score": float(getattr(attempt, "score", 0.0) or 0.0),
            "rule": str(getattr(attempt, "rule", "") or ""),
            "failure": str(getattr(attempt, "failure", "") or ""),
            "total_attempts": int(getattr(attempt, "total_attempts", 0) or 0),
        }
        self.attempts.append(entry)
        self.attempts = self.attempts[-_MAX_LEDGER:]
        if entry["solved"]:
            self.status = "solved"
        else:
            self.add_loopholes(
                [
                    Loophole(
                        "unsolved_attempt",
                        str(entry["failure"])
                        or f"reading scored {entry['score']:.2f} without solving",
                    )
                ]
            )
        self._touch()

    # ─── Views ───────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Compact status for introspection (the /task surface)."""
        return {
            "name": self.spec.get("name", ""),
            "family": self.spec.get("family", ""),
            "status": self.status,
            "revisions": len(self.revisions),
            "attempts": len(self.attempts),
            "cycles": self.cycles,
            "open_loopholes": len(self.open_loopholes()),
            "age_s": max(0, int(time.time() * 1000) - self.created_ms) // 1000,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec,
            "created_ms": self.created_ms,
            "updated_ms": self.updated_ms,
            "status": self.status,
            "revisions": [r.to_dict() for r in self.revisions],
            "evidence": self.evidence,
            "loopholes": [lh.to_dict() for lh in self.loopholes],
            "attempts": self.attempts,
            "cycles": self.cycles,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeldTask | None:
        spec = data.get("spec")
        if not isinstance(spec, dict) or not spec:
            return None
        task = cls(
            spec=spec,
            created_ms=int(data.get("created_ms", 0)),
            updated_ms=int(data.get("updated_ms", 0)),
            status=str(data.get("status", "open")),
            revisions=[TaskRevision.from_dict(r) for r in data.get("revisions", [])],
            evidence=list(data.get("evidence", []))[-_MAX_LEDGER:],
            loopholes=[Loophole.from_dict(lh) for lh in data.get("loopholes", [])],
            attempts=list(data.get("attempts", []))[-_MAX_LEDGER:],
            cycles=int(data.get("cycles", 0)),
        )
        return task

    # ─── Persistence ─────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist the held task — the memory of held information."""
        try:
            path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path) -> HeldTask | None:
        """Reload a held task across restarts.

        Anything unreadable or older than ``_STALE_MS`` is abandoned
        state, not a live problem — None, never fatal.
        """
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        task = cls.from_dict(data) if isinstance(data, dict) else None
        if task is None:
            return None
        if int(time.time() * 1000) - task.updated_ms > _STALE_MS:
            return None
        return task
