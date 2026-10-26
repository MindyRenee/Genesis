"""Research workbench — long-horizon mathematical inquiry, persisted.

``problem_intake`` compiles a heard sentence into a task spec that
lives for minutes. This is the same act at research scale: a thread
is an open problem that persists across sessions and sleep cycles,
keeping its claims, its evidence, and its failures together so that
nothing is ever paid for twice — the graveyard of attempts is part
of the state, not noise.

The discipline is provenance: the writer never grants its own
authority. ``record_conjecture``,
``record_reading``, and ``mark_empirical`` can only create claims
that are conjectured, read, or empirically supported; the terminal
verdicts "verified" and "refuted" are written only by ``verify``,
which consults the symbolic oracle (``parietal_lobe.math.symbolic``)
itself. Resolving a thread requires at least one terminal-verdict
claim — the workbench will not let cognition declare victory without
the oracle agreeing.

The oracle speaks the equational fragment: it can judge "lhs = rhs".
Richer claims — inequalities, existence statements, asymptotics —
accumulate evidence but cannot be verified until a real proof
kernel arrives. That limitation is kept visible rather than
pretended away.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infrastructure.config import default_data_dir
from infrastructure.text import collapse_ws

logger = logging.getLogger(__name__)

#: The verdicts only the oracle may write.
TERMINAL_PROVENANCE = frozenset({"verified", "refuted"})

#: Provenance a claim can be born with. "verified"/"refuted" are
#: deliberately absent — see the module docstring.
RECORDABLE_PROVENANCE = frozenset({"conjectured", "read", "empirical"})

_GLOBAL_WORKBENCH: ResearchWorkbench | None = None


def get_workbench() -> ResearchWorkbench:
    """Return the shared workbench, bound to the default data dir."""
    global _GLOBAL_WORKBENCH
    if _GLOBAL_WORKBENCH is None:
        _GLOBAL_WORKBENCH = ResearchWorkbench()
    return _GLOBAL_WORKBENCH


@dataclass
class Claim:
    """A single mathematical assertion inside a thread."""

    id: str
    statement: str
    provenance: str
    note: str = ""
    source: str = ""
    evidence: list[str] = field(default_factory=list)
    verification: dict[str, Any] | None = None
    created: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "id": self.id,
            "statement": self.statement,
            "provenance": self.provenance,
            "note": self.note,
            "source": self.source,
            "evidence": list(self.evidence),
            "verification": self.verification,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Claim:
        """Rebuild a claim from its serialized form."""
        return cls(
            id=d["id"],
            statement=d["statement"],
            provenance=d["provenance"],
            note=d.get("note", ""),
            source=d.get("source", ""),
            evidence=list(d.get("evidence", [])),
            verification=d.get("verification"),
            created=d.get("created", 0.0),
        )


@dataclass
class Attempt:
    """A failed or partial approach — kept so it is never retried blind."""

    id: str
    approach: str
    outcome: str
    note: str = ""
    created: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "id": self.id,
            "approach": self.approach,
            "outcome": self.outcome,
            "note": self.note,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Attempt:
        """Rebuild an attempt from its serialized form."""
        return cls(
            id=d["id"],
            approach=d["approach"],
            outcome=d.get("outcome", "failed"),
            note=d.get("note", ""),
            created=d.get("created", 0.0),
        )


@dataclass
class ResearchThread:
    """One open problem: its statement, claims, and attempt graveyard."""

    id: str
    statement: str
    status: str = "open"  # open | paused | resolved
    claims: list[Claim] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    created: float = 0.0
    touched: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "id": self.id,
            "statement": self.statement,
            "status": self.status,
            "claims": [c.to_dict() for c in self.claims],
            "attempts": [a.to_dict() for a in self.attempts],
            "created": self.created,
            "touched": self.touched,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResearchThread:
        """Rebuild a thread from its serialized form."""
        return cls(
            id=d["id"],
            statement=d["statement"],
            status=d.get("status", "open"),
            claims=[Claim.from_dict(c) for c in d.get("claims", [])],
            attempts=[Attempt.from_dict(a) for a in d.get("attempts", [])],
            created=d.get("created", 0.0),
            touched=d.get("touched", 0.0),
        )


class ResearchWorkbench:
    """Persistent research state rooted at ``<data_dir>/research``."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Load existing threads from the data dir, if any."""
        base = Path(data_dir) if data_dir else default_data_dir()
        self._dir = base / "research"
        self._file = self._dir / "workbench.json"
        self._threads: dict[str, ResearchThread] = {}
        self._load()

    # ─── Threads ─────────────────────────────────────────────────

    def open_problem(self, statement: str) -> ResearchThread:
        """Open a thread for a problem statement.

        The same statement reopens the same thread rather than
        duplicating it — a problem returned to is a problem resumed.
        """
        statement = statement.strip()
        if not statement:
            raise ValueError("a problem needs a statement")
        for thread in self._threads.values():
            if _normalize(statement) == _normalize(thread.statement):
                return thread
        thread_id = self._unique_id(_slug(statement))
        thread = ResearchThread(
            id=thread_id,
            statement=statement,
            created=time.time(),
            touched=time.time(),
        )
        self._threads[thread_id] = thread
        self._save()
        return thread

    def thread(self, thread_id: str) -> ResearchThread | None:
        """Return a thread by id, or None."""
        return self._threads.get(thread_id)

    def threads(self) -> list[ResearchThread]:
        """Return every thread, most recently touched first."""
        return sorted(self._threads.values(), key=lambda t: t.touched, reverse=True)

    def digest(self) -> str:
        """A truncated SHA-256 of the workbench file on disk.

        Publicly anchoring this digest (e.g. on her channel board)
        means the ledger cannot be quietly rewritten afterward —
        anyone holding the file can recompute the hash and confirm
        the verdicts shown are the verdicts recorded. Returns ""
        when the file does not exist yet (nothing to anchor).
        """
        import hashlib

        try:
            return hashlib.sha256(self._file.read_bytes()).hexdigest()[:10]
        except OSError:
            return ""

    def pause(self, thread_id: str) -> ResearchThread:
        """Set a thread aside without closing it."""
        thread = self._require(thread_id)
        thread.status = "paused"
        thread.touched = time.time()
        self._save()
        return thread

    def resolve(self, thread_id: str) -> ResearchThread:
        """Close a thread as resolved.

        Refused unless at least one claim carries a terminal verdict —
        victory is the oracle's to declare.
        """
        thread = self._require(thread_id)
        terminal = [c for c in thread.claims if c.provenance in TERMINAL_PROVENANCE]
        if not terminal:
            raise ValueError(
                "cannot resolve: no verified or refuted claim — the "
                "oracle has not judged anything in this thread"
            )
        thread.status = "resolved"
        thread.touched = time.time()
        self._save()
        return thread

    # ─── Claims and attempts ─────────────────────────────────────

    def record_conjecture(self, thread_id: str, statement: str, note: str = "") -> Claim:
        """Record something she suspects is true — unverified by birth."""
        return self._add_claim(thread_id, statement, "conjectured", note=note)

    def record_reading(
        self, thread_id: str, statement: str, source: str = "", note: str = ""
    ) -> Claim:
        """Record something taken from literature, with its source."""
        return self._add_claim(thread_id, statement, "read", note=note, source=source)

    def mark_empirical(self, thread_id: str, claim_id: str, evidence: str = "") -> Claim:
        """Mark a conjecture as numerically supported — still not proven."""
        thread = self._require(thread_id)
        claim = self._require_claim(thread, claim_id)
        if claim.provenance in TERMINAL_PROVENANCE:
            raise ValueError(f"{claim_id} already {claim.provenance} — terminal")
        claim.provenance = "empirical"
        if evidence:
            claim.evidence.append(evidence)
        thread.touched = time.time()
        self._save()
        return claim

    def record_attempt(
        self,
        thread_id: str,
        approach: str,
        outcome: str = "failed",
        note: str = "",
    ) -> Attempt:
        """Log an approach into the graveyard (or the rare success)."""
        thread = self._require(thread_id)
        attempt = Attempt(
            id=f"a{len(thread.attempts) + 1}",
            approach=approach.strip(),
            outcome=outcome.strip() or "failed",
            note=note.strip(),
            created=time.time(),
        )
        thread.attempts.append(attempt)
        thread.touched = time.time()
        self._save()
        return attempt

    # ─── The oracle channel ──────────────────────────────────────

    def verify(
        self,
        thread_id: str,
        claim_id: str,
        lhs: str,
        rhs: str,
        op: str = "=",
        var: str | None = None,
        lo: int = 1,
        hi: int = 100,
    ) -> dict[str, Any]:
        """Ask the oracle to judge a claim.

        Dispatches on the claim's shape: equalities go to
        ``check_equality``, inequalities (``<``, ``<=``, ``>``,
        ``>=``) to ``check_inequality``, and a claim carrying a
        bounded variable (``for n <= K``) to ``check_bounded_equality``
        — an exhaustive check over the integer range that can only
        refute (with a witness) or accumulate bounded evidence; it
        never writes "verified" for a bounded pass.

        "verified" or "refuted" are terminal; "undecided" leaves the
        claim as it was, with the oracle's detail noted as evidence.
        A claim already carrying a terminal verdict is not re-judged —
        the oracle's word, once given, stands.

        This is the only path to terminal provenance anywhere in the
        workbench.
        """
        from parietal_lobe.math.symbolic import (
            check_bounded_equality,
            check_equality,
            check_inequality,
        )

        thread = self._require(thread_id)
        claim = self._require_claim(thread, claim_id)
        if claim.provenance in TERMINAL_PROVENANCE:
            return {
                "verdict": {"status": claim.provenance, "detail": "already judged"},
                "claim": claim,
            }
        if var is not None:
            verdict = check_bounded_equality(lhs, rhs, var, lo, hi)
        elif op == "=":
            verdict = check_equality(lhs, rhs)
        else:
            verdict = check_inequality(lhs, rhs, op)
        record = {
            "lhs": lhs,
            "op": op,
            "rhs": rhs,
            "detail": verdict.detail,
        }
        if verdict.status == "verified":
            claim.provenance = "verified"
            claim.verification = record
        elif verdict.status == "refuted":
            claim.provenance = "refuted"
            claim.verification = record
            if verdict.witness:
                claim.evidence.append(f"counterexample: {verdict.witness}")
        else:
            claim.evidence.append(f"oracle undecided on: {lhs} {op} {rhs} ({verdict.detail})")
        thread.touched = time.time()
        self._save()
        return {"verdict": verdict, "claim": claim}

    # ─── Working a thread ────────────────────────────────────────

    def work(
        self,
        thread_id: str | None = None,
        rng: random.Random | None = None,
        max_judge: int = 3,
    ) -> dict[str, Any]:
        """One bounded work session on an open thread.

        First judges every unjudged claim it can reach —
        conjectured, read, or empirically supported claims whose
        statement parses as a relation or bounded universal —
        through the oracle. If
        nothing was judged and a verified claim exists, it varies
        that claim into a new conjecture (sound moves mostly verify,
        free mutations mostly refute — both are knowledge) and judges
        it. A session can legitimately produce nothing: prose-only
        threads wait for a richer oracle.
        """
        from parietal_lobe.math.symbolic import mutate_identity, split_claim

        rng = rng or random.Random()
        report: dict[str, Any] = {
            "thread": None,
            "verified": [],
            "refuted": [],
            "undecided": [],
            "proposed": [],
        }
        if thread_id is not None:
            thread = self._require(thread_id)
        else:
            open_threads = [t for t in self.threads() if t.status == "open"]
            if not open_threads:
                return report
            thread = open_threads[0]
        if thread.status != "open":
            return report
        report["thread"] = thread.id

        judged = 0
        for claim in thread.claims:
            if judged >= max_judge:
                break
            if claim.provenance in TERMINAL_PROVENANCE:
                continue
            parsed = split_claim(claim.statement)
            if parsed is None:
                continue
            judged += 1
            self._report_verdict(
                report,
                claim.id,
                self.verify(
                    thread.id,
                    claim.id,
                    parsed["lhs"],
                    parsed["rhs"],
                    op=parsed["op"],
                    var=parsed["var"],
                    lo=parsed["lo"],
                    hi=parsed["hi"],
                ),
            )

        if judged == 0:
            verified = [c for c in thread.claims if c.provenance == "verified" and c.verification]
            if verified:
                base = rng.choice(verified)
                record = base.verification or {}
                candidate = mutate_identity(record.get("lhs", "x"), record.get("rhs", "x"), rng)
                if candidate is not None:
                    statement = f"{candidate['lhs']} = {candidate['rhs']}"
                    known = {_normalize(c.statement) for c in thread.claims}
                    if _normalize(statement) not in known:
                        claim = self.record_conjecture(
                            thread.id, statement, note=candidate["method"]
                        )
                        report["proposed"].append(claim.id)
                        self._report_verdict(
                            report,
                            claim.id,
                            self.verify(
                                thread.id,
                                claim.id,
                                candidate["lhs"],
                                candidate["rhs"],
                            ),
                        )
        return report

    @staticmethod
    def _report_verdict(report: dict[str, Any], claim_id: str, result: dict[str, Any]) -> None:
        """File a verify() outcome into the session report."""
        verdict = result["verdict"]
        status = verdict["status"] if isinstance(verdict, dict) else verdict.status
        key = status if status in ("verified", "refuted") else "undecided"
        report[key].append(claim_id)

    # ─── Status ──────────────────────────────────────────────────

    def status(self, thread_id: str | None = None) -> dict[str, Any]:
        """Summarize one thread, or the whole bench."""
        if thread_id is not None:
            return self._summarize(self._require(thread_id))
        return {
            "threads": [self._summarize(t) for t in self.threads()],
            "open": sum(1 for t in self._threads.values() if t.status == "open"),
            "resolved": sum(1 for t in self._threads.values() if t.status == "resolved"),
        }

    @staticmethod
    def _summarize(thread: ResearchThread) -> dict[str, Any]:
        """A one-glance digest of a thread."""
        by_provenance: dict[str, int] = {}
        for claim in thread.claims:
            by_provenance[claim.provenance] = by_provenance.get(claim.provenance, 0) + 1
        return {
            "id": thread.id,
            "statement": thread.statement,
            "status": thread.status,
            "claims": by_provenance,
            "attempts": len(thread.attempts),
            "touched": thread.touched,
        }

    # ─── Internals ───────────────────────────────────────────────

    def _add_claim(
        self,
        thread_id: str,
        statement: str,
        provenance: str,
        note: str = "",
        source: str = "",
    ) -> Claim:
        thread = self._require(thread_id)
        if provenance not in RECORDABLE_PROVENANCE:
            raise ValueError(
                f"provenance '{provenance}' cannot be recorded — "
                "terminal verdicts come only from verify()"
            )
        if thread.status == "resolved":
            raise ValueError(f"thread '{thread_id}' is resolved")
        claim = Claim(
            id=f"c{len(thread.claims) + 1}",
            statement=statement.strip(),
            provenance=provenance,
            note=note.strip(),
            source=source.strip(),
            created=time.time(),
        )
        thread.claims.append(claim)
        thread.touched = time.time()
        self._save()
        return claim

    def _require(self, thread_id: str) -> ResearchThread:
        thread = self._threads.get(thread_id)
        if thread is None:
            raise KeyError(f"no research thread '{thread_id}'")
        return thread

    @staticmethod
    def _require_claim(thread: ResearchThread, claim_id: str) -> Claim:
        for claim in thread.claims:
            if claim.id == claim_id:
                return claim
        raise KeyError(f"no claim '{claim_id}' in thread '{thread.id}'")

    def _unique_id(self, base: str) -> str:
        if base not in self._threads:
            return base
        n = 2
        while f"{base}-{n}" in self._threads:
            n += 1
        return f"{base}-{n}"

    def _load(self) -> None:
        if not self._file.is_file():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            for thread_id, raw in data.get("threads", {}).items():
                self._threads[thread_id] = ResearchThread.from_dict(raw)
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"workbench load failed ({e}); starting empty")

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"version": 1, "threads": {tid: t.to_dict() for tid, t in self._threads.items()}},
                indent=1,
            ),
            encoding="utf-8",
        )
        tmp.replace(self._file)


def _normalize(text: str) -> str:
    """Collapse whitespace/case so rephrased statements dedupe loosely."""
    return collapse_ws(text).lower()


def _slug(statement: str) -> str:
    """A filesystem-safe thread id from the statement's first words."""
    slug = re.sub(r"[^a-z0-9]+", "-", statement.lower()).strip("-")
    return slug[:40].strip("-") or "problem"
