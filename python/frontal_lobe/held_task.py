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
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Bounded memory — a held task accrues evidence and attempts
#: without a natural flush, so each ledger keeps only the tail.
_MAX_LEDGER = 50

#: Value learning rate — how much of each prediction error the
#: held expectation absorbs per outcome (α in V ← V + α·δ).
_VALUE_LR = 0.3

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
        """Rehydrate a revision from its serialized dict.

        Absent fields decode to defaults rather than failing — an
        older task file is a memory, not a schema.
        """
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
        """Rehydrate a loophole from its serialized dict; absent
        fields default (empty kind/detail, unaddressed)."""
        return cls(
            kind=str(data.get("kind", "")),
            detail=str(data.get("detail", "")),
            addressed=bool(data.get("addressed", False)),
        )


def goal_of(spec: dict[str, Any]) -> dict[str, Any]:
    """The goal of a task — the end calculation, not the task's name.

    A goal is the state that closes the computation: the 4 in
    2+2=?. Each family carries its target in different fields —
    the number the counts must sum to, the continuation the pattern
    implies, the arrangement the relations require — so this
    extracts the end-state itself: ∃x closing the spec.

    Nothing here *satisfies* anything — satisfaction is the body's
    response to the state being reached (the reward signal, not the
    math object).
    """
    family = str(spec.get("family", ""))
    if family == "quantities":
        return {"equals": spec.get("target")}
    if family == "sequence":
        return {
            "extends": list(spec.get("pattern") or []),
            "to": spec.get("length"),
        }
    if family == "relations":
        return {"arrangement": list(spec.get("goals") or [])}
    if family == "classification":
        return {
            "partition": spec.get("predicate"),
            "items": len(spec.get("items") or []),
        }
    if family == "sorter":
        return {
            "seating": {
                "blocks": len(spec.get("blocks") or []),
                "slots": len(spec.get("slots") or []),
            }
        }
    if family in ("assembly", "grid"):
        return {"structure": len(spec.get("pieces") or [])}
    return {"result": True}


def goal_text(goal: dict[str, Any]) -> str:
    """The end state as a reason seed — data for the composer,
    not a sentence she recites."""
    if "equals" in goal:
        return f"reach {goal['equals']}"
    if "extends" in goal:
        return "continue the pattern"
    if "arrangement" in goal:
        return "the arrangement the relations require"
    if "partition" in goal:
        return "the rule that sorts them"
    if "seating" in goal:
        return "every block seated"
    if "structure" in goal:
        return "the finished structure"
    return "the answer"


@dataclass
class HeldTask:
    """A task held in mind: working spec plus the memory around it.

    Attributes:
        spec: The current normalized task spec — the reading the
            next attempt will play.
        goal: The end calculation — the binding that closes the
            computation (extracted from the spec, refreshed on
            revision).
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
    #: Second-order state — the distribution over readings.
    #: log-weight per hypothesis (spec name → w); ``_NULL`` is the
    #: residual "some reading I haven't formed" hypothesis, held at
    #: weight 0 so failed readings push mass onto it. The posterior
    #: over readings is softmax(w); confidence is its top mass —
    #: metacognition as a function, not a stored number.
    beliefs: dict[str, float] = field(default_factory=dict)
    goal: dict[str, Any] = field(default_factory=dict)
    #: The held prediction — V(s), the expected value of the current
    #: reading. Not a number recomputed on demand: a maintained
    #: memory representation (OFC/ventral striatum carry expected
    #: value) that the prediction error itself rewrites —
    #: V ← V + α·δ, so the same δ the body broadcasts is what
    #: updates it. ``last_delta`` is the most recent error signal;
    #: transient, but carried here so the felt response reads the
    #: same δ the memory learned from.
    expected_value: float = 0.5
    last_delta: float = 0.0

    def __post_init__(self) -> None:
        if not self.created_ms:
            self.created_ms = int(time.time() * 1000)
        if not self.updated_ms:
            self.updated_ms = self.created_ms
        if not self.goal:
            self.goal = goal_of(self.spec)

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
        self.goal = goal_of(spec)
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

    def distance(self) -> float:
        """Distance to the goal — the means-ends difference.

        The best score across attempts is how far the closest reading
        got toward the end calculation; the distance is what remains:
        1.0 untouched, 0.0 reached.
        """
        if self.status == "solved":
            return 0.0
        if not self.attempts:
            return 1.0
        return max(0.0, 1.0 - max(e["score"] for e in self.attempts))

    # ─── Beliefs — the second-order function ─────────────────────

    _NULL = "__unread__"

    def learn_reading(self, name: str) -> None:
        """Register a hypothesis — a candidate reading of the task.

        Every reading enters at the null prior (weight 0); attempts
        move the weights, never the list order.
        """
        self.beliefs.setdefault(name, 0.0)
        self.beliefs.setdefault(self._NULL, 0.0)

    def posterior(self) -> dict[str, float]:
        """softmax over log-weights — the distribution over readings."""
        if not self.beliefs:
            return {}
        w_max = max(self.beliefs.values())
        exps = {n: math.exp(w - w_max) for n, w in self.beliefs.items()}
        total = sum(exps.values()) or 1.0
        return {n: v / total for n, v in exps.items()}

    def confidence(self, name: str | None = None) -> float:
        """Posterior mass on a reading — how sure she is *of it*.

        ``name`` defaults to the current spec's: "how sure am I that
        this is the right way to read the task" — a computed
        quantity, not a dial.
        """
        post = self.posterior()
        if not post:
            return 0.5
        key = name or str(self.spec.get("name", ""))
        return post.get(key, 0.0)

    def top_reading(self) -> tuple[str, float]:
        """The most probable hypothesis and its mass."""
        post = self.posterior()
        if not post:
            return "", 0.0
        top = max(post.items(), key=lambda kv: kv[1])
        return top

    def record_attempt(self, attempt: Any) -> None:
        """Fold a play outcome into the ledgers and the beliefs.

        The attempt is evidence *and* history: the outcome lands in
        the attempts ledger, a solved attempt flips the task status,
        and an unsolved one is logged as a loophole on the current
        reading — never as a verdict that the task is impossible.
        The reading's log-weight takes the update: solved is
        absorbing certainty, unsolved adds ln(score) of negative
        evidence — Bayesian in spirit, bounded in form.
        """
        entry: dict[str, Any] = {
            "ms": int(time.time() * 1000),
            "solved": bool(getattr(attempt, "solved", False)),
            "score": float(getattr(attempt, "score", 0.0) or 0.0),
        }
        # δ = r − V(s) is computed here, where V lives: the
        # outcome arrives, is compared against the held
        # prediction, and the error signal rewrites the prediction
        # itself — V ← V + α·δ (TD learning; the phasic dopamine
        # signal IS the teaching signal). The same δ is what the
        # body broadcasts.
        r = 1.0 if entry["solved"] else float(entry["score"])
        self.last_delta = r - self.expected_value
        self.expected_value += _VALUE_LR * self.last_delta
        entry.update(
            {
                "rule": str(getattr(attempt, "rule", "") or ""),
                "failure": str(getattr(attempt, "failure", "") or ""),
                "total_attempts": int(getattr(attempt, "total_attempts", 0) or 0),
                "name": str(getattr(attempt, "task", "") or self.spec.get("name", "")),
            }
        )
        self.attempts.append(entry)
        self.attempts = self.attempts[-_MAX_LEDGER:]
        name: str = entry["name"] or str(self.spec.get("name", ""))
        self.beliefs.setdefault(name, 0.0)
        self.beliefs.setdefault(self._NULL, 0.0)
        if entry["solved"]:
            self.status = "solved"
            # Absorbing certainty — the oracle verified this reading.
            self.beliefs[name] = 10.0
        else:
            self.beliefs[name] += math.log(max(float(entry["score"]), 0.05))
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
            "confidence": round(self.confidence(), 3),
            "goal": self.goal,
            "distance": round(self.distance(), 3),
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
            "beliefs": dict(self.beliefs),
            "goal": self.goal,
            "expected_value": self.expected_value,
            "last_delta": self.last_delta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeldTask | None:
        """Rebuild a held task from its serialized form.

        Returns ``None`` when the payload carries no usable spec —
        a specless task is nothing to resume, so absence is the
        honest decode. The rolling ledgers clamp to
        ``_MAX_LEDGER`` on the way in.
        """
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
            beliefs={str(k): float(v) for k, v in dict(data.get("beliefs", {})).items()},
            goal=dict(data.get("goal", {})),
            expected_value=float(data.get("expected_value", 0.5)),
            last_delta=float(data.get("last_delta", 0.0)),
        )
        return task

    # ─── Persistence ─────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Persist the held task — the memory of held information."""
        try:
            path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        except OSError as e:
            from genesis_client.swallow import note_swallowed

            note_swallowed("frontal_lobe.held_task.save", e)

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
