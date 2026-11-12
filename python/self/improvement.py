"""Self-improvement — Genesis proposes modifications to its own code.

Genesis identifies opportunities for improvement, researches solutions,
and generates concrete proposals. Each proposal includes the exact code
change, the rationale, and the expected benefit. Pending proposals run
through the verification pipeline automatically — the gate applies what
passes and rejects what fails with teaching notes; feedback from both
outcomes helps Genesis learn what kinds of improvements are valuable.

A small set of safe, mechanical fixes (the "autonomous allowlist") it
applies directly to its own code — currently ``silent_except``,
``unreachable_code``, ``print_in_code``, and ``none_comparison``.
These fixes don't change control flow (or only remove dead code), only
add observability or correct idiomatic style, and are validated with
``py_compile`` before being written. If validation fails, it reverts.
This is its first step toward autonomous self-modification.

## The proposal lifecycle

1. **Discovery**: Genesis identifies an improvement opportunity through:
   - Bug reports (bare except, mutable defaults, long functions, etc.)
   - Reflection insights (repetitive patterns, knowledge gaps)
   - Code analysis (missing error handling, inefficient patterns)
   - Research (best practices, new techniques it learned about)

2. **Proposal generation**: It creates a concrete proposal with:
   - The file and lines to change
   - The original code
   - The proposed replacement code
   - A rationale explaining why
   - The expected benefit
   - A confidence score

3. **Owner oversight**: the human can watch and steer, but never has
   to — the queue drains itself through verified experiments:
   - `/proposals` — see pending proposals
   - `/proposal N` — see full details of proposal N
   - `/accept N [reason]` — run a proposal's verification now
   - `/reject N [reason]` — reject with feedback

4. **Feedback learning**: It records the outcome and the feedback,
   building a model of what kinds of proposals are valued. Over time,
   it gets better at proposing useful changes.

## Applying changes — her source is her workspace

Genesis works on the live tree. Every code change — an autonomous
mechanical fix or a proposal experiment — is written to the file in
place, verified there, and reverted if verification fails. There is no
staging copy: the system is her workspace. Two paths land changes:

- **Autonomous fixes** — a small allowlist of safe, mechanical fixes
  (no control-flow changes; observability additions like logging;
  Python-only). Validated with ``py_compile``, import execution, and
  the full test suite, reverting on any failure.
- **Experiments** — everything else, including Rust. A proposal's
  ``original_code`` → ``proposed_code`` edit lands on the live file
  (with a ``.pre_experiment`` backup beside it), then the full
  verification gate runs: ``py_compile`` + module import + pytest +
  cargo test. A passing gate marks the proposal applied; a failure
  restores the file and the proposal is rejected with notes teaching
  what went wrong. The verification gate is the arbiter of which
  changes land — pending proposals run through it automatically; the
  human ``/accept`` path is an override, not a gate.

## Boundary

It only modifies its own source (defined by ``_OWN_SOURCE_DIRS``,
``_OWN_SOURCE_FILES``, and the crate's ``src/`` tree in
``SelfImprovement``) — never tests, configuration, scripts,
Cargo.toml, or build.rs. That boundary is not permission, it is
coverage: the test suite and build files are the instruments that
verify every change, so they stay outside the reach of the thing they
measure. This boundary is the single source of truth shared
by autonomous fixes, experiments, and proposal generation.

## Proposal categories

- **bugfix**: Fix a detected bug or code smell
- **optimization**: Improve performance or efficiency
- **safety**: Add error handling, input validation, or bounds checking
- **refactor**: Improve code structure without changing behavior
- **feature**: Propose a new capability
- **research**: Propose researching a topic before implementing
"""

from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from tools.framework import get_tools

if TYPE_CHECKING:
    from temporal_lobe.concepts import ConceptNetwork

logger = logging.getLogger(__name__)


def _parse_source_tree(source: str) -> ast.Module | None:
    """Parse Python source, returning None on syntax error."""
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


_LOGGER_BINDING_RE = re.compile(r"^\s*logger\s*=\s*(?:logging\.)?getLogger", re.M)


def _module_has_logger(source: str) -> bool:
    """True when the module binds a module-level ``logger``.

    A fix that emits ``logger.debug(...)`` / ``logger.info(...)`` is
    only valid where the name resolves — injected into a module
    without one it compiles cleanly and then raises NameError the
    first time it executes.
    """
    return _LOGGER_BINDING_RE.search(source) is not None


_PRAGMA_COMMENT_RE = re.compile(
    r"(noqa\s*:?\s*[A-Z]*[\d]*(,\s*[A-Z]+\d+)*"
    r"|type:\s*ignore(\[[^\]]*\])?"
    r"|pragma\s*:\s*\S+"
    r"|nosec\b"
    r"|pylint\s*:\s*\S+"
    r"|fmt\s*:\s*\S+"
    r"|isort\s*:\s*\S+)",
    re.IGNORECASE,
)


def _comment_has_prose(text: str) -> bool:
    """True when the comments in *text* contain explanation.

    Lint and pragma markers (``# noqa``, ``# type: ignore``,
    ``# nosec``, ``# pragma: no cover`` …) are stripped first; two or
    more remaining words means a human explained the code — e.g.
    ``# a listener must never kill her game``. A documented
    ``except: pass`` is deliberate and must not be "fixed".
    """
    for comment in re.findall(r"#(.*)$", text, re.M):
        prose = _PRAGMA_COMMENT_RE.sub("", comment)
        if len(re.findall(r"[A-Za-z]{3,}", prose)) >= 2:
            return True
    return False


def _splice_line(line: str, start: int, end: int, replacement: str) -> str:
    """Replace ``line[start:end]`` using AST byte offsets.

    AST ``col_offset`` values are UTF-8 byte offsets, not character
    indices, so we splice on the encoded bytes to stay correct for
    lines containing non-ASCII text.
    """
    data = line.encode("utf-8")
    return (data[:start] + replacement.encode("utf-8") + data[end:]).decode("utf-8")


def _operand_span(data: bytes, node: ast.expr) -> tuple[int, int] | None:
    """Byte span of ``node`` in a line, including any wrapping parentheses.

    AST ``col_offset``/``end_col_offset`` point at the expression itself
    and exclude redundant parentheses — so ``(a and b)`` has the span of
    ``a and b``. When an edit replaces a whole operand, the wrapping
    parens must be included or they are left dangling. This expands the
    span outward while the byte just outside each end is ``(`` / ``)``
    (ignoring whitespace), which only happens when they wrap the node.
    """
    start, end = node.col_offset, node.end_col_offset
    if start is None or end is None:
        return None
    while True:
        left = start
        while left > 0 and data[left - 1 : left] in (b" ", b"\t"):
            left -= 1
        right = end
        while right < len(data) and data[right : right + 1] in (b" ", b"\t"):
            right += 1
        if (
            left > 0
            and data[left - 1 : left] == b"("
            and right < len(data)
            and data[right : right + 1] == b")"
        ):
            start, end = left - 1, right + 1
        else:
            return start, end


def _compares_on_line(tree: ast.Module, line_num: int) -> list[ast.Compare]:
    """Return the single-line Compare nodes starting on ``line_num``."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and node.lineno == line_num
        and getattr(node, "end_lineno", node.lineno) == line_num
    ]


class ProposalStatus(Enum):
    """Status of a self-improvement proposal."""

    PENDING = "pending"  # queued — experiments run these automatically
    ACCEPTED = "accepted"  # approved (owner via /accept, or self via verify) — ready to apply
    REJECTED = "rejected"  # human declined, with feedback
    APPLIED = "applied"  # the change was applied to the codebase
    WITHDRAWN = "withdrawn"  # Genesis withdrew it (e.g., found it was wrong)


class ProposalCategory(Enum):
    """What kind of improvement this proposal is."""

    BUGFIX = "bugfix"  # fix a detected bug or code smell
    OPTIMIZATION = "optimization"  # improve performance
    SAFETY = "safety"  # add error handling or validation
    REFACTOR = "refactor"  # improve structure without behavior change
    FEATURE = "feature"  # propose a new capability
    RESEARCH = "research"  # propose researching before implementing


@dataclass(slots=True)
class Proposal:
    """A concrete code modification proposed by Genesis.

    Each proposal represents a specific change to a specific file.
    The proposal includes enough detail for the human to understand
    what would change and why, without needing to read the full file.
    """

    id: int
    title: str
    category: ProposalCategory
    file_path: str
    description: str  # what the proposal does
    rationale: str  # why it's proposing this
    expected_benefit: str  # what improvement this brings
    original_code: str  # the code to be changed
    proposed_code: str  # the replacement code
    confidence: float  # 0..1, how confident it is
    status: ProposalStatus = ProposalStatus.PENDING
    feedback: str = ""  # reason for accept/reject — human's or the gate's
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    reviewed_at: int = 0  # when it left pending (accepted, applied, rejected)
    source: str = ""  # what triggered this proposal (e.g. "code_analysis")
    line_hint: int | None = None  # the line the finding was reported at —
    # resolves which occurrence of original_code to replace when the
    # snippet matches more than one site in the file

    def describe(self) -> str:
        """Brief one-line description for list display."""
        return (
            f"[{self.id}] {self.title} "
            f"({self.category.value}, {self.status.value}) "
            f"— {self.file_path}"
        )

    def describe_full(self) -> str:
        """Full description for detailed display."""
        lines = [
            f"Proposal #{self.id}: {self.title}",
            f"  Category: {self.category.value}",
            f"  Status:   {self.status.value}",
            f"  File:     {self.file_path}",
            f"  Confidence: {self.confidence:.0%}",
            f"  Source:   {self.source}",
            "",
            f"  Description: {self.description}",
            f"  Rationale:   {self.rationale}",
            f"  Benefit:     {self.expected_benefit}",
            "",
            "  --- Original code ---",
        ]
        for line in self.original_code.splitlines():
            lines.append(f"  | {line}")
        lines.append("  --- Proposed code ---")
        for line in self.proposed_code.splitlines():
            lines.append(f"  | {line}")
        if self.feedback:
            lines.append("")
            lines.append(f"  Feedback: {self.feedback}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category.value,
            "file_path": self.file_path,
            "description": self.description,
            "rationale": self.rationale,
            "expected_benefit": self.expected_benefit,
            "original_code": self.original_code,
            "proposed_code": self.proposed_code,
            "confidence": self.confidence,
            "status": self.status.value,
            "feedback": self.feedback,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "source": self.source,
            "line_hint": self.line_hint,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        """Deserialize from persistence."""
        return cls(
            id=data["id"],
            title=data["title"],
            category=ProposalCategory(data.get("category", "bugfix")),
            file_path=data["file_path"],
            description=data["description"],
            rationale=data["rationale"],
            expected_benefit=data["expected_benefit"],
            original_code=data["original_code"],
            proposed_code=data["proposed_code"],
            confidence=data["confidence"],
            status=ProposalStatus(data.get("status", "pending")),
            feedback=data.get("feedback", ""),
            created_at=data.get("created_at", 0),
            reviewed_at=data.get("reviewed_at", 0),
            source=data.get("source", ""),
            line_hint=data.get("line_hint"),
        )


@dataclass(slots=True)
class FeedbackRecord:
    """A record of human feedback on a proposal.

    Genesis uses these records to learn what kinds of proposals
    are valued. Over time, it adjusts its proposal generation
    to favor categories and patterns that get accepted.
    """

    proposal_id: int
    category: str
    accepted: bool
    feedback: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the feedback record to a dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "category": self.category,
            "accepted": self.accepted,
            "feedback": self.feedback,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackRecord:
        """Reconstruct a FeedbackRecord from a serialized dictionary."""
        return cls(
            proposal_id=data["proposal_id"],
            category=data["category"],
            accepted=data["accepted"],
            feedback=data.get("feedback", ""),
            timestamp=data.get("timestamp", 0),
        )


class SelfImprovement:
    """Genesis's ability to propose modifications to its own code.

    This engine:
    1. Identifies improvement opportunities (from code analysis)
    2. Generates concrete proposals with actual code changes
    3. Tracks human feedback to improve future proposals
    4. Persists proposals across sessions

    Proposals describe the exact code change and run through the
    verification pipeline — the gate, not a review queue, decides what
    lands. Verified heuristic experiments (``HeuristicExperiment``)
    are restricted to own-source files and validated by py_compile +
    import + the test suite with automatic revert on failure.
    """

    def __init__(
        self,
        network: ConceptNetwork | None = None,
        project_root: str = ".",
    ) -> None:
        """Initialize the engine with optional network and project root."""
        self.network = network
        self.project_root = project_root

        self._proposals: list[Proposal] = []
        self._next_id = 1
        self._feedback_history: list[FeedbackRecord] = []

        # Persistent title memory for dedup. Proposals are evicted
        # under the cap below, but their titles must stay remembered
        # or an evicted rejected/applied proposal gets re-proposed —
        # the exact duplicate noise the dedup exists to prevent.
        self._known_titles: OrderedDict[str, None] = OrderedDict()

        # Bounds for long-running state — all three lists are fully
        # serialized by to_dict and would otherwise grow forever.
        self._max_proposals = 500
        self._max_known_titles = 2000
        self._max_feedback = 500
        self._max_experiments = 500

        # Category success rates — learned from feedback
        # Maps category → (accepted_count, total_count)
        self._category_stats: dict[str, tuple[int, int]] = {}

        # Timestamp of the last proposal generation cycle (telemetry,
        # not a throttle — there is deliberately no rate limit).
        self._last_generation_time: float = 0.0

        # Heuristic experiment history (workstream D audit log)
        self._experiment_history: list[ExperimentRecord] = []

    # ─── Public API ─────────────────────────────────────────────

    def generate_proposals(self, max_proposals: int = 5, force: bool = False) -> list[Proposal]:
        """Scan for improvement opportunities and generate proposals.

        This is the main entry point. It scans code for common
        improvement patterns and generates its own ideas — refactors,
        optimizations, and new capabilities it thinks would benefit it.

        It doesn't just fix defects. It forms opinions about its own
        code: "this is wasteful," "this could be simpler," "I want
        to add this capability." These are its ideas, not just
        mechanical pattern-matching.

        Args:
            max_proposals: Maximum number of proposals to generate per cycle.
            force: Unused — kept for call-site compatibility. There is
                no throttle: the verification gate is the only arbiter
                of which changes land.

        Returns:
            List of new proposals (all with PENDING status).
        """
        self._last_generation_time = time.time()

        new_proposals: list[Proposal] = []

        # Generate proposals from code analysis (docstrings,
        #    f-strings, long functions, many params). This is a
        #    single AST pass over each allowlisted file that covers
        #    both mechanical patterns and its own opinions about
        #    code structure.
        remaining = max_proposals - len(new_proposals)
        if remaining > 0:
            code_proposals = self._propose_from_code_analysis(remaining)
            new_proposals.extend(code_proposals)

        # Store the proposals, skipping duplicates of any existing
        # proposal with the same title (regardless of status). This
        # prevents the proposal list from filling up with repeated
        # "Add docstring to X()" entries across generation cycles —
        # including ones that were already accepted and applied.
        # _known_titles survives proposal eviction so a title is
        # never re-proposed just because its proposal aged out.
        existing_titles = set(self._known_titles)
        # Also dedup within this batch (don't add two identical
        # proposals in the same generation cycle).
        seen_in_batch: set[str] = set()
        stored: list[Proposal] = []
        for proposal in new_proposals:
            if proposal.title in existing_titles:
                continue
            if proposal.title in seen_in_batch:
                continue
            seen_in_batch.add(proposal.title)
            proposal.id = self._next_id
            self._next_id += 1
            self._proposals.append(proposal)
            self._remember_title(proposal.title)
            stored.append(proposal)
        self._enforce_proposal_cap()

        return stored

    def _remember_title(self, title: str) -> None:
        """Remember a proposal title for dedup, LRU-bounded."""
        self._known_titles[title] = None
        self._known_titles.move_to_end(title)
        while len(self._known_titles) > self._max_known_titles:
            self._known_titles.popitem(last=False)

    def _enforce_proposal_cap(self) -> None:
        """Bound ``_proposals`` — it is fully serialized by to_dict.

        Evicts the oldest terminal proposal first (rejected/applied/
        withdrawn — kept only as audit trail), then accepted, and as
        a last resort the oldest pending proposal. Pending proposals
        represent unreviewed work, so they are only dropped when the
        queue is entirely unreviewed and already over the cap.
        """
        while len(self._proposals) > self._max_proposals:
            idx = next(
                (
                    i
                    for i, p in enumerate(self._proposals)
                    if p.status
                    in (
                        ProposalStatus.REJECTED,
                        ProposalStatus.APPLIED,
                        ProposalStatus.WITHDRAWN,
                    )
                ),
                None,
            )
            if idx is None:
                idx = next(
                    (
                        i
                        for i, p in enumerate(self._proposals)
                        if p.status == ProposalStatus.ACCEPTED
                    ),
                    0,  # all pending — drop the oldest
                )
            dropped = self._proposals.pop(idx)
            logger.info(
                "Self-improvement: evicted proposal #%d (%s, %s) under the %d-proposal cap",
                dropped.id,
                dropped.status.value,
                dropped.title,
                self._max_proposals,
            )

    def get_pending_proposals(self) -> list[Proposal]:
        """Get all proposals queued for a verification experiment."""
        return [p for p in self._proposals if p.status == ProposalStatus.PENDING]

    #: A pending proposal unrun this long is dead weight, not
    #: pending work — self-triage withdraws it rather than letting
    #: the queue stand as a permanent, undrainable obligation.
    _PROPOSAL_STALE_MS = 7 * 24 * 3600 * 1000
    #: Most withdrawals per triage call — discharge is bounded work,
    #: not a purge.
    _PROPOSAL_TRIAGE_CAP = 10

    def triage_pending(self, now_ms: int | None = None) -> list[int]:
        """Withdraw pending proposals that no longer deserve a run.

        The queue drains itself through verified experiments — this
        triage only clears proposals that could never survive the gate
        anyway, so the drain isn't clogged with dead entries. Two
        grounds for withdrawal:

        * **stale** — unrun past ``_PROPOSAL_STALE_MS``;
        * **dead file** — ``file_path`` no longer exists.

        Returns the withdrawn proposal ids, capped at
        ``_PROPOSAL_TRIAGE_CAP`` per call.
        """
        now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        withdrawn: list[int] = []
        for proposal in self.get_pending_proposals():
            if len(withdrawn) >= self._PROPOSAL_TRIAGE_CAP:
                break
            reason = ""
            if now_ms - proposal.created_at > self._PROPOSAL_STALE_MS:
                reason = "stale — unreviewed past the horizon"
            elif not os.path.isfile(os.path.join(self.project_root, proposal.file_path)):
                reason = "its target file no longer exists"
            if reason and self.withdraw_proposal(proposal.id, reason):
                withdrawn.append(proposal.id)
        if withdrawn:
            logger.info(
                "Self-improvement: self-triaged %d stale proposal(s) — %s",
                len(withdrawn),
                withdrawn,
            )
        return withdrawn

    def get_proposal(self, proposal_id: int) -> Proposal | None:
        """Get a specific proposal by ID."""
        for p in self._proposals:
            if p.id == proposal_id:
                return p
        return None

    def get_all_proposals(self) -> list[Proposal]:
        """Get all proposals (any status)."""
        return list(self._proposals)

    def clear_proposals(self) -> int:
        """Remove all proposals and return how many were cleared."""
        count = len(self._proposals)
        self._proposals = []
        self._next_id = 1
        return count

    def accept_proposal(self, proposal_id: int, feedback: str = "") -> bool:
        """Accept a proposal.

        Args:
            proposal_id: The ID of the proposal to accept.
            feedback: Optional reason for accepting.

        Returns:
            True if the proposal was found and accepted.
        """
        proposal = self.get_proposal(proposal_id)
        if proposal is None or proposal.status != ProposalStatus.PENDING:
            return False

        proposal.status = ProposalStatus.ACCEPTED
        proposal.feedback = feedback
        proposal.reviewed_at = int(time.time() * 1000)

        self._record_feedback(proposal, accepted=True, feedback=feedback)
        return True

    def reject_proposal(self, proposal_id: int, feedback: str = "") -> bool:
        """Reject a proposal with feedback.

        The feedback is crucial — it's how Genesis learns what kinds
        of proposals are not valuable. It uses this to adjust its
        future proposal generation.

        Args:
            proposal_id: The ID of the proposal to reject.
            feedback: The reason for rejection. This is recorded and
                used to improve future proposals.

        Returns:
            True if the proposal was found and rejected.
        """
        proposal = self.get_proposal(proposal_id)
        if proposal is None:
            return False
        # Allow rejection from PENDING (human rejects) or ACCEPTED
        # (experiment failed after approval). Once REJECTED or APPLIED,
        # it's final.
        if proposal.status not in (ProposalStatus.PENDING, ProposalStatus.ACCEPTED):
            return False

        proposal.status = ProposalStatus.REJECTED
        proposal.feedback = feedback
        proposal.reviewed_at = int(time.time() * 1000)

        self._record_feedback(proposal, accepted=False, feedback=feedback)
        return True

    def mark_applied(self, proposal_id: int) -> bool:
        """Mark a proposal as applied — but only via the experiment pipeline.

        This is called by the experiment framework after a successful
        experiment. It should NOT be called directly to bypass the
        experiment pipeline — every code change must go through
        run_experiment with full verification.

        PENDING and ACCEPTED proposals can be marked as applied — the
        verification gate is the approval: a pending proposal that
        survives the gate is as vetted as an accepted one. Rejected or
        withdrawn proposals cannot be silently resurrected as applied.

        Args:
            proposal_id: The ID of the proposal that was applied.

        Returns:
            True if the proposal was found and marked as applied.
        """
        proposal = self.get_proposal(proposal_id)
        if proposal is None or proposal.status not in (
            ProposalStatus.PENDING,
            ProposalStatus.ACCEPTED,
        ):
            return False
        proposal.status = ProposalStatus.APPLIED
        return True

    def withdraw_proposal(self, proposal_id: int, reason: str = "") -> bool:
        """Withdraw a proposal (Genesis decides it was wrong).

        Args:
            proposal_id: The ID of the proposal to withdraw.
            reason: Why it's withdrawing it.

        Returns:
            True if the proposal was found and withdrawn.
        """
        proposal = self.get_proposal(proposal_id)
        if proposal is None or proposal.status != ProposalStatus.PENDING:
            return False
        proposal.status = ProposalStatus.WITHDRAWN
        proposal.feedback = reason
        return True

    def proposals_summary(self) -> str:
        """Human-readable summary of all proposals."""
        pending = self.get_pending_proposals()
        accepted = [p for p in self._proposals if p.status == ProposalStatus.ACCEPTED]
        rejected = [p for p in self._proposals if p.status == ProposalStatus.REJECTED]
        applied = [p for p in self._proposals if p.status == ProposalStatus.APPLIED]

        lines = [
            "Self-improvement proposals:",
            f"  Pending:  {len(pending)}",
            f"  Accepted: {len(accepted)}",
            f"  Rejected: {len(rejected)}",
            f"  Applied:  {len(applied)}",
        ]

        if pending:
            lines.append("")
            lines.append("  Pending proposals:")
            for p in pending:
                lines.append(f"    {p.describe()}")

        # Show feedback-based learning stats
        if self._category_stats:
            lines.append("")
            lines.append("  Category success rates:")
            for cat, (accepted_n, total) in sorted(self._category_stats.items()):
                rate = accepted_n / total if total > 0 else 0.0
                lines.append(f"    {cat}: {accepted_n}/{total} ({rate:.0%})")

        return "\n".join(lines)

    # ─── Autonomous fixes ────────────────────────────────────────

    # Categories Genesis fixes directly on the live file. Each entry
    # maps to a method that applies the fix. The allowlist is
    # intentionally tiny — only safe, mechanical fixes that don't
    # change control flow.
    _AUTONOMOUS_FIX_CATEGORIES: ClassVar[set[str]] = {
        "silent_except",
        "unreachable_code",
        "print_in_code",
        "none_comparison",
    }

    # ─── Own-source boundary (single source of truth) ────────────
    #
    # All three systems that touch Genesis's code — autonomous fixes,
    # experiments, and proposal generation — share the same boundary:
    # its own source — Python under _OWN_SOURCE_DIRS, Rust under the
    # crate's src/ — never tests, never config, never Cargo.toml or
    # build.rs. This is the one place that boundary is defined.

    _OWN_SOURCE_DIRS: ClassVar[set[str]] = {
        # The brain-region tree under python/ plus the support
        # packages at the repository root — genesis_client,
        # infrastructure, world, and eval.
        # Tests and __pycache__ are excluded inside _is_own_source.
        "python/",
        "genesis_client/",
        "infrastructure/",
        "world/",
        "eval/",
    }

    # Top-level modules beside python/ — the assembled mind, the
    # console entry, and support adapters.
    _OWN_SOURCE_FILES: ClassVar[set[str]] = {
        "mind.py",
        "genesis_cli.py",
        "commitment.py",
        "discord_feed.py",
        "setup_embeddings.py",
    }

    # Rust subtree of her own crate — repo-relative, so it can only
    # ever match <project_root>/src: callers normalize file_path and
    # reject absolute paths and "..", so a "src/" prefix cannot reach
    # /usr/src, ~/.cargo/registry, or any other crate on the machine.
    _RUST_SOURCE_DIR: ClassVar[str] = "src/"

    @classmethod
    def _is_own_source(cls, file_path: str) -> bool:
        """Check if a file is Genesis's own Python or Rust source.

        This is the single source of truth for the own-source boundary.
        Used by autonomous fixes, experiments, and proposal generation.

        Allowed: any .py file under _OWN_SOURCE_DIRS or in
        _OWN_SOURCE_FILES; any .rs file under the crate's src/ tree.
        Never: tests, config directories, build.rs, Cargo.toml (not
        under src/), or non-source files.
        """
        file_parts = file_path.replace("\\", "/").split("/")
        if any(part in ("tests", "test", "__pycache__") for part in file_parts):
            return False
        if file_path.endswith(".rs"):
            # Only inside the crate's own source tree — anchored at the
            # repo root by the caller's normalization, never anywhere
            # else a "src" directory or ".rs" file happens to live.
            return file_path.startswith(cls._RUST_SOURCE_DIR)
        if not file_path.endswith(".py"):
            return False
        # Must be under an allowed directory or an explicitly allowed file
        if file_path not in cls._OWN_SOURCE_FILES and not any(
            file_path.startswith(d) for d in cls._OWN_SOURCE_DIRS
        ):
            return False
        # Never test files
        if "test_" in file_path or "_test.py" in file_path:
            return False
        # Never config directories — but a source module named
        # config.py is source code, not configuration.
        if any(part.lower() == "config" for part in file_parts):
            return False
        return True

    def get_feedback_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics about proposal feedback by category.

        Returns:
            Dict mapping category name to stats:
            {category: {accepted: int, rejected: int, total: int, rate: float}}
        """
        stats: dict[str, dict[str, Any]] = {}
        for category, (accepted, total) in self._category_stats.items():
            stats[category] = {
                "accepted": accepted,
                "rejected": total - accepted,
                "total": total,
                "rate": accepted / total if total > 0 else 0.0,
            }
        return stats

    # ─── Proposal generation from bug reports ───────────────────

    def _propose_from_code_analysis(self, max_proposals: int) -> list[Proposal]:
        """Generate proposals by scanning code for improvement patterns.

        It looks for opportunities to improve code quality,
        add error handling, or optimize.

        Scans all experimentable files (its own Python source under
        python/ or genesis_client/).
        """
        proposals: list[Proposal] = []

        # Scan all experimentable files — its own Python source.
        experimentable_files = self._collect_experimentable_files()

        for filepath in experimentable_files:
            if not os.path.isfile(filepath):
                continue
            if len(proposals) >= max_proposals:
                break

            rel_path = os.path.relpath(filepath, self.project_root)

            # Skip files we already have proposals for
            existing = {p.source for p in self._proposals if p.file_path == rel_path}

            file_proposals = self._analyze_python_file(
                filepath, rel_path, max_proposals - len(proposals), existing
            )
            proposals.extend(file_proposals)

        return proposals

    def _collect_experimentable_files(self) -> list[str]:
        """Collect all experimentable Python files under _OWN_SOURCE_DIRS.

        Also includes explicitly allowed top-level files from
        _OWN_SOURCE_FILES. Returns absolute paths, sorted for
        deterministic ordering.
        """
        root = Path(self.project_root)
        files: list[str] = []
        for dir_prefix in SelfImprovement._OWN_SOURCE_DIRS:
            dir_path = root / dir_prefix
            if not dir_path.is_dir():
                continue
            for py_file in sorted(dir_path.rglob("*.py")):
                rel = os.path.relpath(str(py_file), self.project_root)
                if SelfImprovement._is_own_source(rel):
                    files.append(str(py_file))
        # The crate's own source — proposals can form about the body
        # too; the experiment gate decides what may apply.
        src_dir = root / SelfImprovement._RUST_SOURCE_DIR
        if src_dir.is_dir():
            for rs_file in sorted(src_dir.rglob("*.rs")):
                rel = os.path.relpath(str(rs_file), self.project_root)
                if SelfImprovement._is_own_source(rel):
                    files.append(str(rs_file))
        # Add explicit top-level files
        for file_path in sorted(SelfImprovement._OWN_SOURCE_FILES):
            abs_path = root / file_path
            if abs_path.is_file():
                files.append(str(abs_path))
        return files

    def _analyze_python_file(
        self,
        filepath: str,
        rel_path: str,
        max_proposals: int,
        existing_sources: set[str],
    ) -> list[Proposal]:
        """Analyze a Python file for improvement opportunities."""
        proposals: list[Proposal] = []

        try:
            with open(filepath, encoding="utf-8") as f:
                source = f.read()
        except OSError:
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        lines = source.splitlines(keepends=True)

        # Walk the tree manually to track class context. ast.walk()
        # flattens the tree, so we lose the parent class — which means
        # methods like __init__ in different classes get the same
        # dedup key, causing proposals for one class to be rejected as
        # duplicates of proposals for another class in the same file.
        def _walk_with_class(
            node: ast.AST,
            class_name: str | None = None,
        ) -> list[tuple[ast.AST, str | None]]:
            """Walk the AST, tracking the enclosing class name."""
            results: list[tuple[ast.AST, str | None]] = []
            for child in ast.iter_child_nodes(node):
                child_class = class_name
                if isinstance(child, ast.ClassDef):
                    child_class = child.name
                results.append((child, child_class))
                results.extend(_walk_with_class(child, child_class))
            return results

        for node, class_name in _walk_with_class(tree):
            if len(proposals) >= max_proposals:
                break

            # Look for functions missing docstrings
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Include class name in the source key so methods in
                # different classes don't collide.
                qualified_name = f"{class_name}.{node.name}" if class_name else node.name
                source_key = f"analysis:missing_docstring:{qualified_name}:{node.lineno}"
                if source_key in existing_sources:
                    continue

                if not self._has_docstring(node) and self._docstring_worth_proposing(node):
                    proposal = self._propose_docstring(
                        node, lines, rel_path, source_key, class_name
                    )
                    if proposal:
                        proposals.append(proposal)

            # Look for bare string formatting that could be f-strings
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                    source_key = f"analysis:old_format:{node.lineno}"
                    if source_key in existing_sources:
                        continue
                    proposal = self._propose_fstring(node, lines, rel_path, source_key)
                    if proposal:
                        proposals.append(proposal)

        # Long functions: skipped. Refactoring a long function requires
        # semantic understanding — extracting the wrong helper breaks
        # invariants. A TODO comment is not a real fix (see the
        # principle in _generate_fix). It should only propose changes
        # it can actually generate correctly.

        return proposals

    def _has_docstring(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if a function has a docstring."""
        if not node.body:
            return False
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                return True
        return False

    @staticmethod
    def _docstring_worth_proposing(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> bool:
        """Filter missing-docstring candidates to worthwhile targets.

        The generator writes the summary line from the function name
        alone, so for a non-dunder the summary can only restate the
        name — "Emit." on ``emit()`` is noise, not documentation.
        A proposal is only worth a review slot when the Args section
        adds information the signature does not already state: at
        least one parameter whose inferred description is more than
        the parameter name echoed back ("record: The record." adds
        nothing; "path: The path to read from or write to." does).
        """
        if node.name.startswith("_"):
            return False
        args = node.args
        for a in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if a.arg in ("self", "cls"):
                continue
            echo = f"The {a.arg.lstrip('_').replace('_', ' ')}."
            if SelfImprovement._infer_arg_description(a.arg) != echo:
                return True
        return False

    def _propose_docstring(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        rel_path: str,
        source: str,
        class_name: str | None = None,
    ) -> Proposal | None:
        """Propose adding a docstring to a function.

        Generates a real docstring inferred from the function name and
        signature — a summary line derived from the function name, and
        an Args section listing each parameter. This is a genuine code
        change that adds value, not a TODO stub.
        """
        # The indent is always derived from the 'def' line, not the last
        # line of the signature — comments or continuation lines may have
        # different indentation that would produce a wrong body indent.
        def_line = lines[node.lineno - 1].rstrip("\n")
        indent = " " * (len(def_line) - len(def_line.lstrip()))
        func_name = node.name
        # Include the class name in the title so methods in different
        # classes don't collide in the deduplication set. Without this,
        # "Add docstring to __init__() in memory_systems.py" would be
        # the same title for every class's __init__ in that file.
        qualified_name = f"{class_name}.{func_name}" if class_name else func_name

        # For multi-line function signatures (e.g., def __init__(\n    self,\n    ...
        # ), the docstring must go after the last line of the signature,
        # not after the first line. node.lineno is the 'def' line; the
        # body starts at node.body[0].lineno, so the signature ends at
        # the line before the first body statement.
        sig_end_line = node.body[0].lineno - 1 if node.body else node.lineno
        if sig_end_line < node.lineno or sig_end_line > len(lines):
            sig_end_line = node.lineno

        # Build a real docstring from the function name and parameters.
        body_indent = indent + "    "
        summary = self._infer_docstring_summary(func_name, class_name)
        args = self._extract_docstring_args(node)
        docstring = self._format_docstring(body_indent, summary, args)

        # Show the full signature in the original/proposed code. The
        # first body line rides along: that seat is where a docstring
        # would live, so once one exists the snippet stops matching and
        # the proposal reads stale instead of inserting a duplicate.
        sig_lines = [lines[i].rstrip("\n") for i in range(node.lineno - 1, sig_end_line)]
        seat = (
            [lines[node.body[0].lineno - 1].rstrip("\n")]
            if node.body and node.body[0].lineno <= len(lines)
            else []
        )
        original = "\n".join(sig_lines + seat)
        proposed = "\n".join([*sig_lines, docstring, *seat])

        return Proposal(
            id=0,
            title=f"Add docstring to {qualified_name}() in {os.path.basename(rel_path)}",
            category=ProposalCategory.REFACTOR,
            file_path=rel_path,
            description=f"Add a docstring to {qualified_name}()",
            rationale=f"{qualified_name}() has no docstring, making it harder "
            f"to understand its purpose and usage.",
            expected_benefit="Docstrings improve code maintainability and "
            "enable better IDE support and documentation generation.",
            original_code=original,
            proposed_code=proposed,
            confidence=0.6 * self._category_confidence(ProposalCategory.REFACTOR),
            source=source,
            line_hint=node.lineno,
        )

    @staticmethod
    def _infer_docstring_summary(func_name: str, class_name: str | None) -> str:
        """Infer a one-line docstring summary from the function name.

        Converts snake_case to words and handles common dunder methods
        and private-method prefixes. The result is a concise phrase
        describing what the function likely does.
        """
        # Handle dunder methods
        dunder_summaries: dict[str, str] = {
            "__init__": "Initialize the instance.",
            "__repr__": "Return a developer-readable representation.",
            "__str__": "Return a human-readable string representation.",
            "__len__": "Return the number of items.",
            "__iter__": "Return an iterator over the items.",
            "__next__": "Return the next item.",
            "__enter__": "Enter the context manager.",
            "__exit__": "Exit the context manager.",
            "__eq__": "Check equality with another instance.",
            "__hash__": "Return a hash value for the instance.",
            "__bool__": "Return the truth value of the instance.",
            "__contains__": "Check if an item is present.",
            "__getitem__": "Get an item by key or index.",
            "__setitem__": "Set an item by key or index.",
            "__delitem__": "Delete an item by key or index.",
        }
        if func_name in dunder_summaries:
            return dunder_summaries[func_name]

        # Strip leading underscores (private convention)
        clean = func_name.lstrip("_")
        # Convert snake_case to words
        words = clean.replace("_", " ").strip()
        # Capitalize first word, keep rest as-is for readability
        if words:
            words = words[0].upper() + words[1:]
        # Add a period
        return f"{words}." if words else f"{func_name}."

    @staticmethod
    def _extract_docstring_args(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[tuple[str, str]]:
        """Extract (name, description) pairs for docstring Args.

        Skips ``self``, ``cls``, and ``*`` / ``**`` markers. The
        description is a brief inference from the parameter name.
        """
        args: list[tuple[str, str]] = []
        all_args = node.args

        # Positional args
        for arg in all_args.posonlyargs + all_args.args:
            if arg.arg in ("self", "cls"):
                continue
            desc = SelfImprovement._infer_arg_description(arg.arg)
            args.append((arg.arg, desc))

        # *args
        if all_args.vararg:
            name = all_args.vararg.arg
            args.append((name, "Additional positional arguments."))

        # Keyword-only args
        for arg in all_args.kwonlyargs:
            desc = SelfImprovement._infer_arg_description(arg.arg)
            args.append((arg.arg, desc))

        # **kwargs
        if all_args.kwarg:
            name = all_args.kwarg.arg
            args.append((name, "Additional keyword arguments."))

        return args

    @staticmethod
    def _infer_arg_description(param_name: str) -> str:
        """Infer a brief description for a parameter from its name."""
        # Common parameter name patterns
        name = param_name.lstrip("_")
        words = name.replace("_", " ")

        # Heuristic descriptions for common names
        lower = name.lower()
        if lower in ("data", "content", "text", "body", "payload"):
            return f"The {words} to process."
        if lower in ("name", "key", "identifier", "id"):
            return f"The {words} to identify the target."
        if lower in ("path", "filepath", "filename", "file"):
            return f"The {words} to read from or write to."
        if lower in ("url", "uri", "endpoint"):
            return f"The {words} to connect to."
        if lower in ("timeout", "delay", "interval"):
            return f"The {words} in seconds."
        if lower in ("max", "limit", "count", "size", "n"):
            return f"The maximum {words}."
        if lower in ("verbose", "debug", "force", "dry_run", "recursive"):
            return f"Whether to {words}."
        if lower in ("callback", "handler", "listener", "observer"):
            return f"The {words} to invoke."
        if lower in ("result", "output", "response", "value"):
            return f"The {words} to use or return."
        if lower.startswith("is_") or lower.startswith("has_") or lower.startswith("should_"):
            return f"Whether {words}."
        if lower.startswith("num_"):
            return f"The number of {words[4:]}."
        if lower.startswith("use_"):
            return f"Whether to use {words[4:]}."

        # Default: describe based on the name
        return f"The {words}."

    @staticmethod
    def _format_docstring(
        indent: str,
        summary: str,
        args: list[tuple[str, str]],
    ) -> str:
        """Format a Google-style docstring with summary and Args.

        Args:
            indent: The indentation string (e.g., ``"        "``).
            summary: The one-line summary.
            args: List of (param_name, description) pairs.
        """
        if not args:
            return f'{indent}"""{summary}"""'
        lines = [f'{indent}"""{summary}', "", f"{indent}Args:"]
        for name, desc in args:
            lines.append(f"{indent}    {name}: {desc}")
        lines.append(f'{indent}"""')
        return "\n".join(lines)

    def _propose_fstring(
        self,
        node: ast.BinOp,
        lines: list[str],
        rel_path: str,
        source: str,
    ) -> Proposal | None:
        """Propose replacing % formatting with an f-string.

        Handles the common ``"format string" % (args)`` pattern by
        converting ``%s``, ``%d``, ``%f`` etc. to ``{arg}`` expressions.
        Falls back to None for complex cases that can't be safely
        converted (named placeholders, mapping substitutions, or
        expressions where the format string can't be statically parsed).
        """
        if node.lineno > len(lines):
            return None
        line = lines[node.lineno - 1].rstrip("\n")

        # Only propose if the line uses % formatting
        if "%" not in line or "format(" in line:
            return None

        # Attempt the actual conversion
        converted = self._convert_percent_to_fstring(node)
        if converted is None:
            return None

        return Proposal(
            id=0,
            title=f"Modernize string formatting in {os.path.basename(rel_path)}:{node.lineno}",
            category=ProposalCategory.REFACTOR,
            file_path=rel_path,
            description="Replace % string formatting with f-string",
            rationale="% formatting is older and less readable than f-strings. "
            "f-strings are the modern Python standard (PEP 498).",
            expected_benefit="Improved readability and maintainability.",
            original_code=line,
            proposed_code=converted,
            confidence=0.4 * self._category_confidence(ProposalCategory.REFACTOR),
            source=source,
            line_hint=node.lineno,
        )

    @staticmethod
    def _convert_percent_to_fstring(node: ast.BinOp) -> str | None:
        """Convert a ``"fmt" % args`` BinOp node to an f-string.

        Returns the f-string source code, or None if the conversion
        can't be done safely (e.g. named placeholders, mapping
        substitutions, non-constant format strings, or format specs
        that don't map cleanly to f-string format specs).
        """
        # Left side must be a string constant
        if not isinstance(node.left, ast.Constant) or not isinstance(node.left.value, str):
            return None

        fmt_str = node.left.value

        # Reject named placeholders (%(name)s) — they require a mapping
        if "%(" in fmt_str:
            return None

        # Find all % format specifiers in the string
        # Match %s, %d, %f, %r, %x, %o, %e, %g, %c, %%,
        # and width/precision variants like %5.2f
        specs = list(re.finditer(r"%(?:(\d+)?(?:\.(\d+))?)?([sdrfegoxXcb%])", fmt_str))
        if not specs:
            return None

        # Count actual placeholders (%% is an escape, not a placeholder)
        placeholders = [s for s in specs if s.group(3) != "%"]
        if not placeholders:
            return None

        # Extract the right-side expressions
        right = node.right
        if isinstance(right, ast.Tuple):
            arg_exprs = right.elts
        else:
            arg_exprs = [right]

        # The number of placeholders must match the number of arguments
        if len(placeholders) != len(arg_exprs):
            return None

        # Build the f-string by replacing each placeholder with {expr}
        result_parts: list[str] = []
        last_end = 0
        arg_idx = 0
        for spec in specs:
            # Add the text before this spec
            result_parts.append(fmt_str[last_end : spec.start()])
            conv = spec.group(3)

            if conv == "%":
                # %% → literal %
                result_parts.append("%")
            else:
                # Get the source representation of the argument expression
                arg_expr = arg_exprs[arg_idx]
                arg_idx += 1
                arg_src = ast.unparse(arg_expr)

                # Map % conversion to f-string format spec
                width = spec.group(1) or ""
                precision = spec.group(2) or ""

                if conv == "s":
                    # %s → {arg}  (or {arg:>{width}} if width specified)
                    if width:
                        result_parts.append(f"{{{arg_src}:>{width}}}")
                    else:
                        result_parts.append(f"{{{arg_src}}}")
                elif conv == "r":
                    # %r → {arg!r}
                    result_parts.append(f"{{{arg_src}!r}}")
                elif conv in ("d", "i"):
                    # %d → {arg:d} or {arg} (int formatting)
                    if width or precision:
                        spec_str = f"{width}.{precision}f" if precision else width
                        result_parts.append(f"{{{arg_src}:{spec_str}}}")
                    else:
                        result_parts.append(f"{{{arg_src}}}")
                elif conv in ("f", "e", "g"):
                    # %f → {arg:.Nf} etc.
                    fmt_spec = ""
                    if width:
                        fmt_spec += width
                    if precision:
                        fmt_spec += f".{precision}"
                    fmt_spec += conv
                    result_parts.append(f"{{{arg_src}:{fmt_spec}}}")
                elif conv in ("x", "X", "o", "b"):
                    # %x → {arg:x}, %X → {arg:X}, etc.
                    fmt_spec = conv
                    if width:
                        fmt_spec = f"{width}{fmt_spec}"
                    result_parts.append(f"{{{arg_src}:{fmt_spec}}}")
                elif conv == "c":
                    # %c → chr(arg) — can't directly map to f-string spec
                    return None
                else:
                    return None

            last_end = spec.end()

        # Add remaining text after the last spec
        result_parts.append(fmt_str[last_end:])

        fstring_body = "".join(result_parts)
        return f'f"{fstring_body}"'

    # ─── Feedback learning ──────────────────────────────────────

    def _record_feedback(self, proposal: Proposal, accepted: bool, feedback: str) -> None:
        """Record human feedback and update category statistics."""
        record = FeedbackRecord(
            proposal_id=proposal.id,
            category=proposal.category.value,
            accepted=accepted,
            feedback=feedback,
        )
        self._feedback_history.append(record)
        # Bound — serialized in full by to_dict.
        if len(self._feedback_history) > self._max_feedback:
            del self._feedback_history[: len(self._feedback_history) - self._max_feedback]

        # Update category statistics
        cat = proposal.category.value
        accepted_n, total = self._category_stats.get(cat, (0, 0))
        total += 1
        if accepted:
            accepted_n += 1
        self._category_stats[cat] = (accepted_n, total)

        # Log the feedback for learning
        if accepted:
            logger.info(
                f"Proposal #{proposal.id} accepted. "
                f"Category {cat} success rate: {accepted_n}/{total}"
            )
        else:
            logger.info(f"Proposal #{proposal.id} rejected: '{feedback}'.")

    def _category_confidence(self, category: ProposalCategory) -> float:
        """Adjust confidence based on historical success rate.

        If proposals of this category have been mostly accepted,
        confidence is boosted. If mostly rejected, confidence is
        reduced. This is how it learns to propose better things.
        """
        cat = category.value
        accepted_n, total = self._category_stats.get(cat, (0, 0))
        if total < 3:
            return 1.0  # not enough data, use base confidence

        rate = accepted_n / total
        # Scale: 0% success → 0.5 confidence, ~71%+ success → 1.0
        return min(1.0, 0.5 + rate * 0.7)

    def record_experiment(self, record: ExperimentRecord) -> None:
        """Append an experiment record, bounding the audit log.

        ``_experiment_history`` is serialized in full by to_dict, so
        it must stay bounded — over months of operation the log would
        otherwise grow without limit.
        """
        self._experiment_history.append(record)
        if len(self._experiment_history) > self._max_experiments:
            del self._experiment_history[: len(self._experiment_history) - self._max_experiments]

    # ─── Persistence ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "proposals": [p.to_dict() for p in self._proposals],
            "next_id": self._next_id,
            "feedback_history": [f.to_dict() for f in self._feedback_history],
            "category_stats": {cat: list(stats) for cat, stats in self._category_stats.items()},
            "last_generation_time": self._last_generation_time,
            "experiments": [e.to_dict() for e in self._experiment_history],
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore state from persistence.

        Pending proposals that can never be applied are dropped:
        - the target file no longer exists;
        - the original_code snippet is gone (the file changed);
        - the snippet matches several sites and the proposal carries
          no line_hint, so applying it could not pick the right one.

        Non-pending proposals (accepted/rejected/applied) are kept
        regardless so the audit trail is preserved.
        """
        restored = [Proposal.from_dict(p) for p in data.get("proposals", [])]
        stale: list[Proposal] = []
        kept: list[Proposal] = []
        for p in restored:
            if p.status is ProposalStatus.PENDING and not self._proposal_can_apply(p):
                stale.append(p)
            else:
                kept.append(p)
        if stale:
            logger.info(
                "Self-improvement: dropped %d stale pending proposal(s) "
                "that can no longer be applied: %s",
                len(stale),
                ", ".join(f"#{p.id} ({p.file_path})" for p in stale),
            )
        self._proposals = kept
        self._enforce_proposal_cap()
        self._next_id = data.get("next_id", 1)

        # Rebuild the title dedup memory from the restored proposals so
        # their titles are never re-proposed.
        self._known_titles.clear()
        for p in self._proposals:
            self._remember_title(p.title)

        self._feedback_history = [
            FeedbackRecord.from_dict(f) for f in data.get("feedback_history", [])
        ][-self._max_feedback :]
        self._category_stats = {
            cat: tuple(stats) for cat, stats in data.get("category_stats", {}).items()
        }
        self._last_generation_time = data.get("last_generation_time", 0.0)

        # Update _next_id if any proposals have higher IDs
        if self._proposals:
            self._next_id = max(self._next_id, max(p.id for p in self._proposals) + 1)

        # Restore heuristic experiment history
        self._experiment_history = [
            ExperimentRecord.from_dict(e) for e in data.get("experiments", [])
        ][-self._max_experiments :]

    def _proposal_can_apply(self, proposal: Proposal) -> bool:
        """Check that a pending proposal's edit is still applicable.

        Same reasoning as the stale-file drop: if the snippet is gone
        the experiment reports "not found", and if it is ambiguous
        with no line_hint the experiment cannot pick a site — both
        outcomes are dead queue entries either way.
        """
        abs_path = os.path.join(self.project_root, proposal.file_path)
        try:
            with open(abs_path, encoding="utf-8") as f:
                file_source = f.read()
        except OSError:
            return False
        if not proposal.original_code:
            return False
        occurrences = file_source.count(proposal.original_code)
        if occurrences == 0:
            return False
        if occurrences > 1 and proposal.line_hint is None:
            return False
        # Comments in the original must survive into the proposed code:
        # a proposal that silently drops a trailing note deletes either a
        # lint suppression or someone's intent.
        for line in proposal.original_code.splitlines():
            hash_idx = line.find("#")
            if hash_idx == -1:
                continue
            comment = line[hash_idx:].rstrip()
            if comment and comment not in proposal.proposed_code:
                return False
        # ``logger.`` calls need a module-level binding — run_experiment
        # applies the proposal verbatim and will not inject one.
        if re.search(r"\blogger\.", proposal.proposed_code) and not re.search(
            r"(?m)^logger\s*[:=]", file_source
        ):
            return False
        return True


# ─── Verified self-improvement (workstream D) ──────────────────────


@dataclass(slots=True)
class ExperimentRecord:
    """A record of a heuristic experiment — an attempt to improve a
    learning heuristic, with pass/fail status and revert tracking.

    This is the audit log for the verified self-improvement loop.
    Every attempt (success or failure) is recorded so we can see
    what Genesis tried, what worked, and what was reverted.
    """

    file_path: str
    description: str
    status: str  # "applied", "reverted", "skipped"
    reason: str  # why it was applied/reverted/skipped
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the experiment record to a dictionary."""
        return {
            "file_path": self.file_path,
            "description": self.description,
            "status": self.status,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        """Reconstruct an ExperimentRecord from a serialized dictionary."""
        return cls(
            file_path=data.get("file_path", ""),
            description=data.get("description", ""),
            status=data.get("status", "skipped"),
            reason=data.get("reason", ""),
            timestamp=data.get("timestamp", 0),
        )


class HeuristicExperiment:
    """A verified self-improvement experiment.

    This is the framework for Genesis to safely improve its own
    learning heuristics. The process:

    1. **Propose**: identify a heuristic function to tweak (e.g.
       relaxing a regex to catch a concept pattern it was missing).
    2. **Apply**: write the change into the live source file, with a
       ``.pre_experiment`` backup beside it — her source tree is her
       workspace; there is no staging copy.
    3. **Verify**: run the verification gate (py_compile + module
       import + pytest + cargo test) against the live tree.
    4. **Commit or revert**: if all checks pass the change stays —
       taking effect on her next start, like any other edit. If any
       check fails the file is restored from the backup content, and
       the live tree carries no unverified edit.

    Guardrails:
    - Only own-source files can be modified (Genesis's Python source
      and the crate's own ``src/`` tree — repo-relative paths only, so
      a Rust edit can never land in /usr/src or another crate).
    - Never tests, never config, never Cargo.toml or build.rs — the
      instruments that verify changes stay outside the reach of the
      thing they measure.
    - Rust experiments are rate-limited: the cargo gate is the
      expensive rung of the ladder.
    - Full audit log (ExperimentRecord for every attempt).
    - Human can hard-rollback.
    """

    # The own-source boundary is defined in SelfImprovement
    # (_OWN_SOURCE_DIRS, _OWN_SOURCE_FILES, _is_own_source) — the
    # single source of truth shared by all three systems.

    # Verify-gate timeouts. The Python arm measured ~2x headroom over
    # the healthy suite; the Rust arm must absorb a cold dependency
    # build (worst case ~15 min) plus the ~4 min cargo suite.
    _VERIFY_TIMEOUT_S: ClassVar[float] = 900.0
    _RUST_VERIFY_TIMEOUT_S: ClassVar[float] = 2400.0

    # Rust experiment rate limit — the cargo gate costs minutes of CPU,
    # so attempts (not just successes) are budgeted like the admin
    # helper's hourly cap.
    _MAX_RUST_EXPERIMENTS_PER_HOUR: ClassVar[int] = 4
    _RUST_RATE_WINDOW_S: ClassVar[float] = 3600.0

    def __init__(
        self,
        engine: SelfImprovement,
        project_root: str = ".",
    ) -> None:
        """Wire the experiment engine to the self-improvement engine and project root.

        Edits land on the live tree — her source is her workspace —
        verified in place and reverted on failure.
        """
        self.engine = engine
        self.project_root = project_root
        self._verify_script = os.path.join(project_root, "scripts", "verify_change.sh")
        self._rust_attempts: deque[float] = deque()
        self._reconcile_interrupted()

    def _reconcile_interrupted(self) -> None:
        """Restore files left edited by an experiment that died mid-verify.

        An experiment parks the pre-edit content in ``<file>.pre_experiment``
        and sets an ``<file>.in_flight`` marker for the live-edit window.
        A crash or shutdown between the write and the gate's verdict leaves
        the marker behind with the edit still in the tree; without
        reconciliation the orphaned edit becomes the next experiment's
        baseline, and each interrupted attempt ratchets the tree further
        from clean. A marker with no readable backup can only be flagged —
        nothing else knows what the file looked like.
        """
        root = Path(self.project_root)
        if not root.is_dir():
            return
        for marker in root.rglob("*.in_flight"):
            live = marker.with_suffix("")
            backup = live.with_suffix(live.suffix + ".pre_experiment")
            try:
                original = backup.read_text(encoding="utf-8")
            except OSError:
                logger.warning(
                    f"In-flight experiment marker without readable backup: {live} — left as-is"
                )
                marker.unlink(missing_ok=True)
                continue
            try:
                live.write_text(original, encoding="utf-8")
                logger.warning(
                    f"Restored {live} from {backup.name} — experiment interrupted mid-verify"
                )
            except OSError as e:
                logger.error(f"Could not restore interrupted experiment file {live}: {e}")
            marker.unlink(missing_ok=True)

    def run_experiment(
        self,
        file_path: str,
        original_code: str,
        proposed_code: str,
        description: str,
        line_hint: int | None = None,
    ) -> ExperimentRecord:
        """Run a single heuristic experiment with full verification.

        Args:
            file_path: Path to the file to modify (relative to
                project root). Must be on the allowlist.
            original_code: The original code snippet to replace.
            proposed_code: The replacement code.
            description: What the experiment does.
            line_hint: The line the finding was reported at — used to
                pick the correct occurrence when ``original_code``
                matches more than one site in the file.

        Returns:
            An ExperimentRecord with the outcome.
        """
        # Normalize first — a "python/../…" path would pass the
        # own-source prefix check while escaping the tree entirely.
        file_path = os.path.normpath(file_path).replace("\\", "/")
        if os.path.isabs(file_path) or file_path.startswith(".."):
            return ExperimentRecord(
                file_path=file_path,
                description=description,
                status="reverted",
                reason="path escapes the source tree",
            )

        # Guardrails
        skip = self._check_experiment_guardrails(file_path, description)
        if skip:
            return skip

        abs_path = Path(self.project_root) / file_path

        # Read and verify the original code exists
        try:
            source = abs_path.read_text(encoding="utf-8")
        except OSError as e:
            return ExperimentRecord(
                file_path=file_path,
                description=description,
                status="reverted",
                reason=f"could not read file: {e}",
            )

        if original_code not in source:
            return ExperimentRecord(
                file_path=file_path,
                description=description,
                status="reverted",
                reason="original code snippet not found in file",
            )

        # Apply to the live file, verify there, revert on failure.
        return self._apply_and_verify(
            source,
            original_code,
            proposed_code,
            file_path,
            description,
            line_hint,
        )

    def _check_experiment_guardrails(
        self, file_path: str, description: str
    ) -> ExperimentRecord | None:
        """Check if a file is experimentable.

        Returns a skip record (with reason) or None if the experiment
        is allowed to proceed.
        """
        if not self._is_experimentable(file_path):
            return ExperimentRecord(
                file_path=file_path,
                description=description,
                status="skipped",
                reason=(
                    f"file not own source: {file_path}. "
                    f"Only Genesis's own Python source and the crate's "
                    f"src/ tree can be modified. "
                    f"Never tests, config, Cargo.toml, or build.rs."
                ),
            )
        if file_path.endswith(".rs") and not self._rust_budget_allows():
            return ExperimentRecord(
                file_path=file_path,
                description=description,
                status="skipped",
                reason=(
                    f"rust experiment rate limit — "
                    f"{self._MAX_RUST_EXPERIMENTS_PER_HOUR} per hour "
                    f"(the cargo gate is the expensive rung)"
                ),
            )
        return None

    def _rust_budget_allows(self) -> bool:
        """Sliding-window rate limit on Rust experiments — cargo builds
        cost real CPU, so attempts are budgeted, not just successes."""
        now = time.time()
        while self._rust_attempts and now - self._rust_attempts[0] > self._RUST_RATE_WINDOW_S:
            self._rust_attempts.popleft()
        if len(self._rust_attempts) >= self._MAX_RUST_EXPERIMENTS_PER_HOUR:
            return False
        self._rust_attempts.append(now)
        return True

    @classmethod
    def _is_experimentable(cls, file_path: str) -> bool:
        """Check if a file can be modified by an experiment.

        Delegates to SelfImprovement._is_own_source — the single
        source of truth for the own-source boundary. Every own-source
        file is experimentable: the verification gate, not a permission
        list, decides which edits land.
        """
        return SelfImprovement._is_own_source(file_path)

    @staticmethod
    def _resolve_occurrence(
        source: str,
        occurrences: list[int],
        original_code: str,
        line_hint: int | None,
    ) -> int | None:
        """Pick the occurrence of a snippet matching the reported line.

        Proposals carry the line the bug was reported at; when a
        snippet matches several sites, the hint picks the intended
        one. Returns None when no occurrence lines up — a guess would
        edit the wrong code.
        """
        if line_hint is None:
            return None
        span_lines = original_code.count("\n")
        # Prefer an occurrence starting exactly at the reported line.
        for off in occurrences:
            if source.count("\n", 0, off) + 1 == line_hint:
                return off
        # Otherwise accept one whose line span contains the hint —
        # e.g. the report points inside a multi-line snippet.
        for off in occurrences:
            start_line = source.count("\n", 0, off) + 1
            if start_line <= line_hint <= start_line + span_lines:
                return off
        return None

    def _apply_and_verify(
        self,
        source: str,
        original_code: str,
        proposed_code: str,
        file_path: str,
        description: str,
        line_hint: int | None = None,
    ) -> ExperimentRecord:
        """Apply the change to the live file, verify there, revert on failure.

        Her source tree is her workspace — the edit lands on the live
        file with a ``.pre_experiment`` backup beside it. Compilation
        and the test gate run against the live tree; on any failure the
        original content is restored, so the tree never keeps an
        unverified edit.
        """
        # ``str.replace(snippet, ..., 1)`` would edit the FIRST
        # occurrence — when a snippet like "    print(" matches several
        # sites, that silently rewrites the wrong code while the
        # reported bug remains. Resolve the intended site: a unique
        # occurrence always wins; an ambiguous one needs the reported
        # line to disambiguate, and without a match it is refused
        # outright rather than guessed.
        occurrences: list[int] = []
        found = source.find(original_code)
        while found != -1:
            occurrences.append(found)
            found = source.find(original_code, found + 1)
        offset = occurrences[0]
        if len(occurrences) > 1:
            resolved = self._resolve_occurrence(source, occurrences, original_code, line_hint)
            if resolved is None:
                return ExperimentRecord(
                    file_path=file_path,
                    description=description,
                    status="reverted",
                    reason=(
                        f"original code snippet is ambiguous "
                        f"({len(occurrences)} sites in file) — "
                        f"needs a more specific snippet"
                    ),
                )
            offset = resolved

        # A "missing docstring" proposal carries a signature-only
        # snippet, so it keeps matching after the function gains a
        # docstring by another route — inserting a second docstring
        # literal ahead of the real one. When the proposed text appends
        # a docstring where the file already opens one, the finding no
        # longer holds; refuse rather than duplicate.
        if proposed_code.startswith(original_code):
            inserted = proposed_code[len(original_code) :].lstrip()
            if inserted.startswith(('"""', "'''")):
                following = source[offset + len(original_code) :].lstrip()
                if following.startswith(('"""', "'''")):
                    return ExperimentRecord(
                        file_path=file_path,
                        description=description,
                        status="reverted",
                        reason=("stale proposal — the target already carries a docstring"),
                    )

        modified_source = source[:offset] + proposed_code + source[offset + len(original_code) :]
        if modified_source == source:
            return ExperimentRecord(
                file_path=file_path,
                description=description,
                status="reverted",
                reason="replacement produced no change",
            )
        return self._live_verify(
            file_path,
            source,
            modified_source,
            description,
        )

    def _live_verify(
        self,
        file_path: str,
        source: str,
        modified_source: str,
        description: str,
    ) -> ExperimentRecord:
        """Write the edited file into the live tree, run the full
        verification gate there, and revert on any failure.

        The original content is parked in a ``.pre_experiment`` file
        beside the live file before the edit lands — the recovery
        point if a verified-good change later proves wrong, and the
        audit trail of what was touched.
        """
        live_file = Path(self.project_root) / file_path
        backup = live_file.with_suffix(live_file.suffix + ".pre_experiment")
        # The marker witnesses the live-edit window: written after the
        # backup, removed on every verdict path below. Surviving past a
        # crash means the tree holds an unverified edit — the next boot's
        # reconcile restores the backup.
        marker = live_file.with_suffix(live_file.suffix + ".in_flight")
        try:
            backup.write_text(source, encoding="utf-8")
            marker.write_text(description, encoding="utf-8")
            live_file.write_text(modified_source, encoding="utf-8")
        except OSError as e:
            # The live file may be partially written — restore it.
            self._revert(live_file, source)
            marker.unlink(missing_ok=True)
            return ExperimentRecord(
                file_path=file_path,
                description=description,
                status="reverted",
                reason=f"could not write live file: {e}",
            )

        try:
            is_rust = file_path.endswith(".rs")
            if not is_rust:
                # Verify: py_compile first (fast fail). Rust has no
                # single-file syntax gate at this layer — the cargo suite
                # below is its fast-fail.
                compile_result = get_tools().run(
                    "compile_python",
                    path=str(live_file),
                    project_root=self.project_root,
                )
                if not compile_result.success:
                    self._revert(live_file, source)
                    record = ExperimentRecord(
                        file_path=file_path,
                        description=description,
                        status="reverted",
                        reason=f"py_compile failed: {compile_result.error}",
                    )
                    self.engine.record_experiment(record)
                    return record

            # Verify: the full gate against the live tree — py_compile all,
            # module imports, pytest, and cargo test. The live crate builds
            # into its own target/ cache, so a Rust experiment costs an
            # incremental build rather than a cold one.
            if not self._run_verify_script(
                python_dir=os.path.join(self.project_root, "python"),
                rust_dir=self.project_root,
                timeout=self._RUST_VERIFY_TIMEOUT_S if is_rust else self._VERIFY_TIMEOUT_S,
            ):
                self._revert(live_file, source)
                record = ExperimentRecord(
                    file_path=file_path,
                    description=description,
                    status="reverted",
                    reason="verification gate failed (tests failed)",
                )
                self.engine.record_experiment(record)
                return record

            record = ExperimentRecord(
                file_path=file_path,
                description=description,
                status="applied",
                reason="all verification checks passed",
            )
            self.engine.record_experiment(record)
            logger.info(f"Heuristic experiment applied: {description} in {file_path}")
            return record
        finally:
            marker.unlink(missing_ok=True)

    def _run_verify_script(
        self,
        python_dir: str | None = None,
        rust_dir: str | None = None,
        timeout: float = _VERIFY_TIMEOUT_S,
    ) -> bool:
        """Run the verification script. Returns True if all checks pass.

        ``python_dir``/``rust_dir`` select the trees under test — the
        live tree in both arms, since edits land in place now.
        """
        if not os.path.exists(self._verify_script):
            logger.warning(f"Verify script not found: {self._verify_script}")
            return False
        command = ["bash", self._verify_script]
        if python_dir is not None:
            command += [python_dir, rust_dir or "."]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                # The verify script runs the full Python suite (~5
                # min, includes the end-to-end daemon test) and the
                # full cargo test suite. A 300s timeout
                # could never complete a healthy run, so every
                # experiment was falsely reverted. 900s gives ~2x
                # headroom over the measured healthy runtime; the
                # rust arm's cold-build allowance is its own constant.
                timeout=timeout,
            )
            if result.returncode != 0:
                logger.warning(
                    f"Verification failed (exit {result.returncode}): {result.stderr[:200]}"
                )
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.warning("Verification timed out (15 min)")
            return False
        except OSError as e:
            logger.warning(f"Could not run verify script: {e}")
            return False

    @staticmethod
    def _revert(path: Path, original_content: str) -> None:
        """Revert a file to its original content."""
        try:
            path.write_text(original_content, encoding="utf-8")
            logger.info(f"Reverted {path}")
        except OSError as e:
            logger.error(f"Could not revert {path}: {e}")

    @property
    def experiment_history(self) -> list[ExperimentRecord]:
        """All experiment records (copy)."""
        return list(self.engine._experiment_history)

    @property
    def applied_count(self) -> int:
        """Number of successfully applied experiments."""
        return sum(1 for r in self.engine._experiment_history if r.status == "applied")

    @property
    def reverted_count(self) -> int:
        """Number of reverted experiments."""
        return sum(1 for r in self.engine._experiment_history if r.status == "reverted")
