"""Self-improvement — Genesis proposes modifications to her own code.

Genesis identifies opportunities for improvement, researches solutions,
and generates concrete proposals. Each proposal includes the exact code
change, the rationale, and the expected benefit. The human reviews most
proposals and can accept or reject them, providing feedback that helps
Genesis learn what kinds of improvements are valuable.

A small set of safe, mechanical fixes (the "autonomous allowlist") she
applies directly to her own code — currently ``silent_except``,
``unreachable_code``, ``print_in_code``, and ``none_comparison``.
These fixes don't change control flow (or only remove dead code), only
add observability or correct idiomatic style, and are validated with
``py_compile`` before being written. If validation fails, she reverts.
This is her first step toward autonomous self-modification.

## The proposal lifecycle

1. **Discovery**: Genesis identifies an improvement opportunity through:
   - Bug reports (bare except, mutable defaults, long functions, etc.)
   - Reflection insights (repetitive patterns, knowledge gaps)
   - Code analysis (missing error handling, inefficient patterns)
   - Research (best practices, new techniques she learned about)

2. **Proposal generation**: She creates a concrete proposal with:
   - The file and lines to change
   - The original code
   - The proposed replacement code
   - A rationale explaining why
   - The expected benefit
   - A confidence score

3. **Human review**: The human can:
   - `/proposals` — see pending proposals
   - `/proposal N` — see full details of proposal N
   - `/accept N [reason]` — accept a proposal
   - `/reject N [reason]` — reject with feedback

4. **Feedback learning**: She records the outcome and the feedback,
   building a model of what kinds of proposals are valued. Over time,
   she gets better at proposing useful changes.

## Autonomous fixes

For a small allowlist of safe, mechanical fixes, Genesis applies the
change directly without human review. The allowlist is intentionally
tiny — only fixes that:
- Don't change control flow
- Only add observability (logging)
- Can be validated with ``py_compile``
- Are in her own source code (not tests)

If validation fails, she reverts the change and logs the failure.

## Safety

Genesis proposes most changes for human review. For the small allowlist
of autonomous fixes, she validates every change with ``py_compile``,
import execution, and the full test suite, reverting on any failure.
She only modifies her own Python source (defined by
``_OWN_SOURCE_DIRS`` and ``_OWN_SOURCE_FILES`` in
``SelfImprovementEngine``) — never tests, configuration, scripts, or
the Rust substrate. This boundary is the single source of truth shared
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
import py_compile
import re
import subprocess
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..tools.framework import ToolResult, get_tools

if TYPE_CHECKING:
    from ..bug_reporter import BugReport, BugReporter
    from ..concepts import ConceptNetwork

logger = logging.getLogger(__name__)


def _parse_source_tree(source: str) -> ast.Module | None:
    """Parse Python source, returning None on syntax error."""
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


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
        while left > 0 and data[left - 1:left] in (b" ", b"\t"):
            left -= 1
        right = end
        while right < len(data) and data[right:right + 1] in (b" ", b"\t"):
            right += 1
        if (
            left > 0
            and data[left - 1:left] == b"("
            and right < len(data)
            and data[right:right + 1] == b")"
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

    PENDING = "pending"      # waiting for human review
    ACCEPTED = "accepted"     # human approved, ready to apply
    REJECTED = "rejected"     # human declined, with feedback
    APPLIED = "applied"       # the change was applied to the codebase
    WITHDRAWN = "withdrawn"   # Genesis withdrew it (e.g., found it was wrong)


class ProposalCategory(Enum):
    """What kind of improvement this proposal is."""

    BUGFIX = "bugfix"           # fix a detected bug or code smell
    OPTIMIZATION = "optimization"  # improve performance
    SAFETY = "safety"           # add error handling or validation
    REFACTOR = "refactor"       # improve structure without behavior change
    FEATURE = "feature"         # propose a new capability
    RESEARCH = "research"       # propose researching before implementing


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
    description: str               # what the proposal does
    rationale: str                 # why she's proposing this
    expected_benefit: str          # what improvement this brings
    original_code: str             # the code to be changed
    proposed_code: str             # the replacement code
    confidence: float              # 0..1, how confident she is
    status: ProposalStatus = ProposalStatus.PENDING
    feedback: str = ""             # human's reason for accept/reject
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    reviewed_at: int = 0           # when the human reviewed it
    source: str = ""               # what triggered this proposal (bug_report, reflection, etc.)

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
        )


@dataclass(slots=True)
class FeedbackRecord:
    """A record of human feedback on a proposal.

    Genesis uses these records to learn what kinds of proposals
    are valued. Over time, she adjusts her proposal generation
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


class SelfImprovementEngine:
    """Genesis's ability to propose modifications to her own code.

    This engine:
    1. Identifies improvement opportunities (from bug reports, reflection, code analysis)
    2. Generates concrete proposals with actual code changes
    3. Tracks human feedback to improve future proposals
    4. Persists proposals across sessions

    Most changes require human review: proposals describe the exact
    code change and wait for acceptance. Two narrow paths modify code
    without review: allowlisted mechanical fixes (``apply_autonomous_fixes``)
    and verified heuristic experiments (``HeuristicExperiment``) — both
    restricted to own-source files outside the protected core, and both
    validated by py_compile + import + the test suite with automatic
    revert on failure.
    """

    def __init__(
        self,
        network: ConceptNetwork | None = None,
        bug_reporter: BugReporter | None = None,
        project_root: str = ".",
    ) -> None:
        """Initialize the engine with optional network, bug reporter, and project root."""
        self.network = network
        self.bug_reporter = bug_reporter
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

        # Cooldown between proposal generation cycles
        self._last_generation_time: float = 0.0
        self._generation_cooldown: float = 1.0  # safety net only; urge is primary gate

        # Heuristic experiment history (workstream D audit log)
        self._experiment_history: list[ExperimentRecord] = []

    # ─── Public API ─────────────────────────────────────────────

    def generate_proposals(self, max_proposals: int = 5, force: bool = False) -> list[Proposal]:
        """Scan for improvement opportunities and generate proposals.

        This is the main entry point. It:
        1. Checks bug reports for fixable issues
        2. Scans code for common improvement patterns
        3. Generates her own ideas — refactors, optimizations, and
           new capabilities she thinks would benefit her

        She doesn't just fix bugs. She forms opinions about her own
        code: "this is wasteful," "this could be simpler," "I want
        to add this capability." These are her ideas, not just
        mechanical pattern-matching.

        Args:
            max_proposals: Maximum number of proposals to generate per cycle.
            force: If True, bypass the cooldown.

        Returns:
            List of new proposals (all with PENDING status).
        """
        now = time.time()
        if not force and now - self._last_generation_time < self._generation_cooldown:
            return []
        self._last_generation_time = now

        new_proposals: list[Proposal] = []

        # 1. Generate proposals from bug reports
        if self.bug_reporter:
            bug_proposals = self._propose_from_bugs(max_proposals)
            new_proposals.extend(bug_proposals)

        # 2. Generate proposals from code analysis (docstrings,
        #    f-strings, long functions, many params). This is a
        #    single AST pass over each allowlisted file that covers
        #    both mechanical patterns and her own opinions about
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
                "Self-improvement: evicted proposal #%d (%s, %s) "
                "under the %d-proposal cap",
                dropped.id, dropped.status.value, dropped.title,
                self._max_proposals,
            )

    def get_pending_proposals(self) -> list[Proposal]:
        """Get all proposals awaiting human review."""
        return [p for p in self._proposals if p.status == ProposalStatus.PENDING]

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
        of proposals are not valuable. She uses this to adjust her
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

        Only proposals in ACCEPTED status can be marked as applied.
        This prevents a rejected or withdrawn proposal from being
        silently resurrected as applied.

        Args:
            proposal_id: The ID of the proposal that was applied.

        Returns:
            True if the proposal was found and marked as applied.
        """
        proposal = self.get_proposal(proposal_id)
        if proposal is None or proposal.status != ProposalStatus.ACCEPTED:
            return False
        proposal.status = ProposalStatus.APPLIED
        return True

    def withdraw_proposal(self, proposal_id: int, reason: str = "") -> bool:
        """Withdraw a proposal (Genesis decides it was wrong).

        Args:
            proposal_id: The ID of the proposal to withdraw.
            reason: Why she's withdrawing it.

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

    # Categories that Genesis is allowed to fix autonomously (without
    # human review). Each entry maps to a method that applies the fix.
    # The allowlist is intentionally tiny — only safe, mechanical fixes
    # that don't change control flow.
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
    # her own Python source, never tests, never config, never Rust.
    # This is the one place that boundary is defined.

    _OWN_SOURCE_DIRS: ClassVar[set[str]] = {
        "python/genesis_cognitive/",
        "python/genesis_client/",
    }

    _OWN_SOURCE_FILES: ClassVar[set[str]] = {
        "python/genesis_cli.py",
        "python/setup_embeddings.py",
    }

    # ─── Protected core — never modified autonomously ───────────
    #
    # These are the files that define Genesis's cognition, language,
    # and cognition — the most critical code paths. Even a "safe
    # mechanical fix" (adding logging to a bare except) is dangerous
    # here: the running instance is mid-conversation when the fix is
    # applied, the test suite takes time to run, and a revert leaves
    # the file in a transient state. More importantly, these files are
    # the ones we (the developers) are actively iterating on — her
    # autonomous changes create a moving target and can mask or
    # introduce bugs in the exact systems that make her work.
    #
    # Proposals can still be generated for these files (she can
    # suggest improvements), but she cannot apply them autonomously
    # — they require human review via /accept.
    _PROTECTED_CORE_DIRS: ClassVar[set[str]] = {
        "python/genesis_cognitive/cognition/",
        "python/genesis_cognitive/language/",
    }
    _PROTECTED_CORE_FILES: ClassVar[set[str]] = {
        "python/genesis_cognitive/mind.py",
    }

    @classmethod
    def _is_protected_core(cls, file_path: str) -> bool:
        """Check if a file is protected core — never auto-modified.

        Protected files can still have proposals generated for them
        (she can suggest improvements), but autonomous fixes and
        experiments cannot touch them. This prevents her self-
        modification loop from changing the cognition/language engine
        while she's running, which creates a moving target and can
        break the very systems that make her work.
        """
        if any(file_path.startswith(d) for d in cls._PROTECTED_CORE_DIRS):
            return True
        return file_path in cls._PROTECTED_CORE_FILES

    @classmethod
    def _is_own_source(cls, file_path: str) -> bool:
        """Check if a file is Genesis's own Python source.

        This is the single source of truth for the own-source boundary.
        Used by autonomous fixes, experiments, and proposal generation.

        Allowed: any .py file under _OWN_SOURCE_DIRS or in
        _OWN_SOURCE_FILES.
        Never: tests, config directories, Rust source, non-Python files.
        """
        if not file_path.endswith(".py"):
            return False
        # Must be under an allowed directory or an explicitly allowed file
        if file_path not in cls._OWN_SOURCE_FILES and not any(
            file_path.startswith(d) for d in cls._OWN_SOURCE_DIRS
        ):
            return False
        # Never test files
        file_parts = file_path.replace("\\", "/").split("/")
        if any(part in ("tests", "test", "__pycache__") for part in file_parts):
            return False
        if "test_" in file_path or "_test.py" in file_path:
            return False
        # Never config directories — but a source module named
        # config.py is source code, not configuration.
        if any(part.lower() == "config" for part in file_parts):
            return False
        return True

    def apply_autonomous_fixes(self, max_fixes: int = 1) -> list[dict[str, Any]]:
        """Apply safe, mechanical fixes to her own code without human review.

        This is Genesis's first step toward self-modification. She
        only applies fixes on the autonomous allowlist (currently
        ``silent_except``, ``unreachable_code``, ``print_in_code``,
        and ``none_comparison``), only to her own Python source
        (gated by ``_is_own_source`` — never scripts, config, tests,
        or the Rust substrate), and only after validating the result
        with ``py_compile`` and an import execution gate. If
        validation fails, she reverts the change.

        Args:
            max_fixes: Maximum number of fixes to apply per call.
                Default 1 — baby steps.

        Returns:
            List of applied fixes, each a dict with:
            - ``file``: the file path
            - ``line``: the line number
            - ``category``: the bug category
            - ``description``: what she changed
        """
        if not self.bug_reporter:
            return []

        applied: list[dict[str, Any]] = []

        # Get current bugs
        try:
            scan_result = self.bug_reporter.scan()
        except (OSError, SyntaxError, ValueError) as e:
            logger.warning(f"Autonomous fix: bug scan failed: {e}")
            return []

        for bug in scan_result.bugs:
            if len(applied) >= max_fixes:
                break

            if bug.category not in self._AUTONOMOUS_FIX_CATEGORIES:
                continue

            # Enforce the own-source boundary — she can only touch her
            # own Python source, never tests, config, scripts, or the
            # Rust substrate.
            if not self._is_autonomous_allowed(bug.file):
                continue

            if bug.category == "silent_except":
                result = self._apply_silent_except_fix(bug)
            elif bug.category == "unreachable_code":
                result = self._apply_unreachable_code_fix(bug)
            elif bug.category == "print_in_code":
                result = self._apply_print_in_code_fix(bug)
            elif bug.category == "none_comparison":
                result = self._apply_none_comparison_fix(bug)
            else:
                result = None
            if result is not None:
                applied.append(result)

        if applied:
            logger.info(
                f"Autonomous fix: applied {len(applied)} fix(es) "
                f"without human review"
            )

        return applied

    @classmethod
    def _is_autonomous_allowed(cls, file_path: str) -> bool:
        """Check whether a file is in the autonomous-fix allowlist.

        Delegates to _is_own_source — the single source of truth for
        the own-source boundary — and additionally excludes protected
        core files (cognition, language, mind) that must never be
        auto-modified.
        """
        if not cls._is_own_source(file_path):
            return False
        if cls._is_protected_core(file_path):
            return False
        return True

    def _apply_silent_except_fix(self, bug: BugReport) -> dict[str, Any] | None:
        """Apply a silent_except fix to a file.

        Reads the file, finds the ``except ...: pass`` pattern at the
        reported line, replaces ``pass`` with ``logger.debug(...)``,
        adds ``as e`` to the except clause, then validates and writes
        via ``_validate_and_write_fix`` (py_compile, import execution,
        and the test suite). Reverts on any validation failure.
        """
        file_path = Path(self.project_root) / bug.file

        # Safety: only .py files in the project
        if not file_path.exists() or file_path.suffix != ".py":
            return None

        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"Autonomous fix: could not read {bug.file}: {e}")
            return None

        lines = source.split("\n")

        # The except line is at bug.line - 1, the pass line is the next
        # line. bug.line < 1 would silently index from the end of file.
        if bug.line < 1 or bug.line >= len(lines):
            return None

        except_line = lines[bug.line - 1]
        pass_line = lines[bug.line]

        # Verify this is still a silent except
        if "except" not in except_line:
            return None
        stripped_pass = pass_line.strip()
        if stripped_pass != "pass" and not stripped_pass.startswith("pass "):
            return None

        # Build the fix
        fixed_except = self._add_as_e_to_except(except_line)
        if fixed_except is None:
            return None

        indent = len(pass_line) - len(pass_line.lstrip())
        indent_str = " " * indent

        # Build a descriptive log message from the context
        context = self._infer_except_context(lines, bug.line)
        fixed_pass = f"{indent_str}logger.debug(f'{context}: {{e}}')"

        # Apply the fix
        lines[bug.line - 1] = fixed_except
        lines[bug.line] = fixed_pass
        fixed_source = "\n".join(lines)

        if not self._validate_and_write_fix(file_path, source, fixed_source, bug.file):
            return None

        logger.info(
            f"Autonomous fix: applied silent_except fix to "
            f"{bug.file}:{bug.line}"
        )

        return {
            "file": bug.file,
            "line": bug.line,
            "category": "silent_except",
            "description": f"Replaced 'except: pass' with logging at {bug.file}:{bug.line}",
        }

    def _apply_unreachable_code_fix(self, bug: BugReport) -> dict[str, Any] | None:
        """Apply an unreachable_code fix to a file.

        Deletes the dead statement at the reported line (the statement
        immediately after a return, raise, break, or continue) and
        validates via ``_validate_and_write_fix`` (py_compile, import
        execution, and the test suite). Reverts on any failure.
        """
        file_path = Path(self.project_root) / bug.file

        if not file_path.exists() or file_path.suffix != ".py":
            return None

        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"Autonomous fix: could not read {bug.file}: {e}")
            return None

        lines = source.split("\n")

        if bug.line < 1 or bug.line > len(lines):
            return None

        # Verify the previous non-blank line ends a control-flow path
        prev_idx = bug.line - 2
        while prev_idx >= 0 and not lines[prev_idx].strip():
            prev_idx -= 1
        if prev_idx < 0:
            return None

        prev_line = lines[prev_idx]
        terminators = ("return", "raise", "break", "continue")
        if not any(term in prev_line for term in terminators):
            return None

        # The reported line is the dead statement; remove it
        del lines[bug.line - 1]
        fixed_source = "\n".join(lines)

        if not self._validate_and_write_fix(file_path, source, fixed_source, bug.file):
            return None

        logger.info(
            f"Autonomous fix: removed unreachable code at {bug.file}:{bug.line}"
        )

        return {
            "file": bug.file,
            "line": bug.line,
            "category": "unreachable_code",
            "description": (
                f"Removed unreachable code after {prev_line.strip()} "
                f"at {bug.file}:{bug.line}"
            ),
        }

    def _apply_print_in_code_fix(
        self, bug: BugReport
    ) -> dict[str, Any] | None:
        """Replace print() with a logger call at the reported line.

        Also ensures a module-level ``logger`` exists so the fix is
        safe and self-contained. Validates and writes via
        ``_validate_and_write_fix`` (py_compile, import execution, and
        the test suite). Reverts on any validation failure.
        """
        file_path = Path(self.project_root) / bug.file

        if not file_path.exists() or file_path.suffix != ".py":
            return None

        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"Autonomous fix: could not read {bug.file}: {e}")
            return None

        lines = source.split("\n")

        if bug.line < 1 or bug.line > len(lines):
            return None

        fix = self._fix_print_in_code(lines, bug.line)
        if fix is None:
            return None

        _original, fixed_line, _desc, _benefit = fix

        # Make sure a module logger exists, otherwise logger.info() won't
        # resolve at runtime. If the file already has one, leave it alone.
        target_idx = bug.line - 1
        has_logger = re.search(r"^\s*logger\s*=\s*logging\.getLogger", source, re.M)
        if not has_logger:
            for idx, line in enumerate(lines):
                if line.startswith("import logging"):
                    lines.insert(idx + 1, "logger = logging.getLogger(__name__)")
                    if idx + 1 <= target_idx:
                        target_idx += 1
                    break
            else:
                # No logging import found; add both at the top.
                # Keep __future__ imports first if they exist.
                insert_at = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if i == 0 and stripped.startswith("#!"):
                        insert_at = i + 1
                        continue
                    if stripped.startswith("from __future__"):
                        insert_at = i + 1
                    elif stripped and not stripped.startswith(("#", '"', "'")):
                        # Reached real code; stop scanning
                        break
                    else:
                        insert_at = i + 1
                lines.insert(insert_at, "import logging")
                lines.insert(insert_at + 1, "logger = logging.getLogger(__name__)")
                if insert_at <= target_idx:
                    target_idx += 2

        lines[target_idx] = fixed_line
        fixed_source = "\n".join(lines)

        if not self._validate_and_write_fix(file_path, source, fixed_source, bug.file):
            return None

        logger.info(f"Autonomous fix: replaced print() in {bug.file}:{bug.line}")

        return {
            "file": bug.file,
            "line": bug.line,
            "category": "print_in_code",
            "description": f"Replaced print() with logger.info() at {bug.file}:{bug.line}",
        }

    def _apply_none_comparison_fix(self, bug: BugReport) -> dict[str, Any] | None:
        """Apply a none_comparison fix to a file.

        Replaces '== None' with 'is None' and '!= None' with 'is not None'
        at the reported line, then validates via ``_validate_and_write_fix``
        (py_compile, import execution, and the test suite). Reverts on any
        validation failure.
        """
        file_path = Path(self.project_root) / bug.file

        if not file_path.exists() or file_path.suffix != ".py":
            return None

        try:
            source = file_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning(f"Autonomous fix: could not read {bug.file}: {e}")
            return None

        lines = source.split("\n")

        if bug.line < 1 or bug.line > len(lines):
            return None

        line = lines[bug.line - 1]
        fixed = re.sub(r"==\s*None\b", "is None", line)
        fixed = re.sub(r"!=\s*None\b", "is not None", fixed)
        if fixed == line:
            return None

        lines[bug.line - 1] = fixed
        fixed_source = "\n".join(lines)

        if not self._validate_and_write_fix(file_path, source, fixed_source, bug.file):
            return None

        logger.info(
            f"Autonomous fix: replaced None comparison in {bug.file}:{bug.line}"
        )

        return {
            "file": bug.file,
            "line": bug.line,
            "category": "none_comparison",
            "description": (
                f"Replaced '== None'/'!= None' with 'is None'/'is not None' "
                f"at {bug.file}:{bug.line}"
            ),
        }

    def _validate_and_write_fix(
        self,
        file_path: Path,
        original_source: str,
        fixed_source: str,
        rel_path: str,
    ) -> bool:
        """Validate fixed source with py_compile and, if tests exist, pytest.

        If either validation step fails, the original file content is
        restored before returning False.
        """
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(fixed_source)
            tmp_path = tmp.name

        # Compile the temp file directly. It lives outside project_root,
        # so we do not pass it through the sandboxed tool.
        try:
            py_compile.compile(tmp_path, doraise=True)
            compile_result = ToolResult(success=True, output=f"Compiled {tmp_path}.")
        except py_compile.PyCompileError as e:
            compile_result = ToolResult(success=False, error=str(e))
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if not compile_result.success:
            logger.warning(
                f"Autonomous fix: validation failed for {rel_path}: "
                f"{compile_result.error}"
            )
            return False

        try:
            file_path.write_text(fixed_source, encoding="utf-8")
        except OSError as e:
            logger.warning(
                f"Autonomous fix: could not write {rel_path}: {e}"
            )
            return False

        # Import-execution gate: py_compile only validates syntax — it
        # does not execute module-level code. A file that raises at
        # import time (e.g. an invalid re.compile at module scope, a
        # missing module-level name) passes py_compile but breaks every
        # importer. Import the changed module in a subprocess to catch
        # this precisely, before the slow full-suite gate below.
        module_name = self._module_name_for(rel_path)
        if module_name:
            try:
                import_result = subprocess.run(
                    ["python3", "-c", f"import {module_name}"],
                    cwd=os.path.join(self.project_root, "python"),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.warning(
                    f"Autonomous fix: import check errored for {rel_path}: {e}"
                )
                self._revert_write(file_path, original_source, rel_path)
                return False
            if import_result.returncode != 0:
                logger.warning(
                    f"Autonomous fix: import check failed for {rel_path}: "
                    f"{import_result.stderr.strip()[:300]}"
                )
                self._revert_write(file_path, original_source, rel_path)
                return False

        # Run the Python test suite as a post-write gate.
        # Use python/tests (the actual Python test directory), not
        # tests/ (which contains Rust integration tests and causes
        # pytest to exit with code 5 "no tests collected").
        test_result = get_tools().run(
            "run_pytest",
            target="python/tests",
            project_root=self.project_root,
        )
        if not test_result.success:
            logger.warning(
                f"Autonomous fix: tests failed after {rel_path} change. "
                f"{test_result.error}"
            )
            # The fix compiled but broke tests. Revert before reporting failure.
            self._revert_write(file_path, original_source, rel_path)
            return False
        return True

    @staticmethod
    def _module_name_for(rel_path: str) -> str | None:
        """Map a repo-relative .py path to an importable module name.

        ``"python/genesis_cognitive/perception.py"`` →
        ``"genesis_cognitive.perception"``; ``"python/genesis_cli.py"``
        → ``"genesis_cli"``. Returns None for paths that are not
        importable project modules (tests, scripts, or files outside
        the packages) — the import gate is skipped for those.
        """
        if not rel_path.endswith(".py") or "__pycache__" in rel_path:
            return None
        parts = rel_path.replace("\\", "/").split("/")
        if parts and parts[0] == "python":
            parts = parts[1:]
        if not parts:
            return None
        parts[-1] = parts[-1][: -len(".py")]
        if len(parts) == 1:
            # Top-level script (e.g. genesis_cli.py) — importable
            # directly from the python/ directory.
            return parts[0] or None
        if parts[0] not in ("genesis_cognitive", "genesis_client"):
            return None
        return ".".join(parts)

    @staticmethod
    def _revert_write(file_path: Path, original_source: str, rel_path: str) -> None:
        """Restore a file's original content after a gate failure."""
        try:
            file_path.write_text(original_source, encoding="utf-8")
        except OSError as e:
            logger.error(
                f"Autonomous fix: could not revert {rel_path} after gate "
                f"failure: {e}"
            )

    @staticmethod
    def _infer_except_context(lines: list[str], except_line_num: int) -> str:
        """Infer a description for the except block from surrounding code.

        Looks at the method name or nearby code to generate a
        meaningful log message.
        """
        # Search backward for the enclosing def
        for i in range(except_line_num - 1, max(0, except_line_num - 20), -1):
            line = lines[i]
            m = re.match(r"\s*def\s+(\w+)", line)
            if m:
                return f"{m.group(1)} failed"
        return "silent except"

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

    def _propose_from_bugs(self, max_proposals: int) -> list[Proposal]:
        """Generate proposals from bug report findings.

        Each bug report that has a clear fix becomes a proposal
        with the actual code change.
        """
        if not self.bug_reporter:
            return []

        proposals: list[Proposal] = []

        # Get recent bug reports. Use a high max_files to ensure we
        # reach Genesis's own source — when scanning from the project
        # root, rglob returns files in filesystem order, and the first
        # 100 may be scripts/Rust files before reaching python/genesis_cognitive/.
        try:
            scan_result = self.bug_reporter.scan(
                max_files=500, include_tests=False
            )
            bugs = scan_result.bugs
            logger.info(
                f"Self-improvement: bug scan found {len(bugs)} bugs "
                f"({scan_result.new_bugs} new)"
            )
        except (OSError, SyntaxError, ValueError) as e:
            logger.warning(f"Self-improvement: bug scan failed: {e}")
            return []

        # Prioritize by severity: warnings before style issues.
        # A silent_except (warning) is more important than a print_in_code (style).
        severity_rank = {"error": 0, "warning": 1, "style": 2}
        bugs = sorted(bugs, key=lambda b: severity_rank.get(b.severity, 3))

        # Deduplicate — don't propose for bugs we already have proposals for
        existing_keys = {
            (p.file_path, p.source)
            for p in self._proposals
            if p.source.startswith("bug:")
        }

        # Diversify: cap proposals per category and per file so she
        # surfaces different kinds of issues rather than flooding the
        # review queue with 5 copies of the same problem.
        max_per_category = max(1, max_proposals // 3)
        max_per_file = max(1, max_proposals // 3)
        category_counts: dict[str, int] = {}
        file_counts: dict[str, int] = {}

        for bug in bugs:
            if len(proposals) >= max_proposals:
                break

            key = (bug.file, f"bug:{bug.category}:{bug.line}")
            if key in existing_keys:
                continue

            # Skip bugs in non-experimentable files — proposals for
            # files she can't experiment on (tests, config, Rust,
            # scripts) just accumulate as unreviewable noise. Use
            # _is_experimentable (not _is_allowlisted) so protected
            # core files (cognition/, language/, mind.py) still get
            # bug-fix proposals — they require human review via
            # /accept but are experimentable, matching the behaviour
            # of _propose_from_code_analysis.
            if not HeuristicExperiment._is_experimentable(bug.file):
                continue

            if category_counts.get(bug.category, 0) >= max_per_category:
                continue
            if file_counts.get(bug.file, 0) >= max_per_file:
                continue

            proposal = self._generate_bug_fix_proposal(bug)
            if proposal:
                proposals.append(proposal)
                existing_keys.add(key)
                category_counts[bug.category] = category_counts.get(bug.category, 0) + 1
                file_counts[bug.file] = file_counts.get(bug.file, 0) + 1

        return proposals

    def _generate_bug_fix_proposal(self, bug: BugReport) -> Proposal | None:
        """Generate a concrete fix proposal from a bug report.

        Reads the actual code at the bug location and generates
        the replacement code.
        """
        # Read the file and extract the relevant code
        file_path = os.path.join(self.project_root, bug.file)
        if not os.path.exists(file_path):
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                source = f.read()
        except OSError:
            return None

        # Generate the fix based on the bug category
        fix = self._generate_fix(bug, source)
        if fix is None:
            return None

        original_code, proposed_code, description, benefit = fix

        # Adjust confidence based on category success rate
        confidence = self._category_confidence(ProposalCategory.BUGFIX)

        return Proposal(
            id=0,
            title=f"Fix {bug.category} in {os.path.basename(bug.file)}:{bug.line}",
            category=ProposalCategory.BUGFIX,
            file_path=bug.file,
            description=description,
            rationale=f"Bug report: {bug.description}",
            expected_benefit=benefit,
            original_code=original_code,
            proposed_code=proposed_code,
            confidence=confidence,
            source=f"bug:{bug.category}:{bug.line}",
        )

    def _generate_fix(
        self, bug: BugReport, source: str
    ) -> tuple[str, str, str, str] | None:
        """Generate a concrete fix for a specific bug.

        Returns (original_code, proposed_code, description, benefit)
        or None if no fix can be generated.
        """
        lines = source.splitlines(keepends=True)

        if bug.line < 1 or bug.line > len(lines):
            return None

        category = bug.category

        if category == "bare_except":
            return self._fix_bare_except(lines, bug.line)
        elif category == "silent_except":
            return self._fix_silent_except(lines, bug.line)
        elif category == "mutable_default":
            return self._fix_mutable_default(lines, bug.line)
        elif category == "missing_type_hints":
            return self._fix_missing_type_hints(lines, bug.line)
        elif category == "print_in_code":
            return self._fix_print_in_code(lines, bug.line)
        elif category == "unwrap_in_production":
            return self._fix_unwrap(lines, bug.line)
        elif category == "import_star":
            return self._fix_import_star(lines, bug.line)
        elif category == "unreachable_code":
            return self._fix_unreachable_code(lines, bug.line)
        elif category == "none_comparison":
            return self._fix_none_comparison(lines, bug.line)
        elif category == "is_literal_comparison":
            return self._fix_is_literal_comparison(source, bug.line)
        elif category == "type_equality_check":
            return self._fix_type_equality_check(source, bug.line)
        elif category == "boolean_equality":
            return self._fix_boolean_equality(source, bug.line)
        elif category == "assert_tuple":
            return self._fix_assert_tuple(source, bug.line)
        elif category == "base_exception_catch":
            return self._fix_base_exception_catch(lines, bug.line)
        elif category == "fstring_no_placeholder":
            return self._fix_fstring_no_placeholder(source, bug.line)
        elif category == "long_function":
            # Long functions require semantic understanding to refactor
            # safely — extracting the wrong helper breaks invariants.
            # A TODO comment is not a real fix (see the principle above).
            # Skip this category rather than generate a useless proposal.
            return None

        # Unknown category — skip rather than generate a TODO comment.
        # A TODO comment is not a real fix and wastes human review time.
        # She should only propose changes she can actually generate.
        return None

    def _fix_bare_except(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Fix a bare 'except:' by replacing with 'except Exception:'."""
        line = lines[line_num - 1].rstrip("\n")
        fixed = line.replace("except:", "except Exception:")
        if fixed == line:
            return None
        return (
            line,
            fixed,
            "Replace bare 'except:' with 'except Exception:'",
            "Prevents catching SystemExit and KeyboardInterrupt, which "
            "could mask critical signals like Ctrl+C.",
        )

    def _fix_silent_except(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Fix a silent 'except: pass' by adding logging."""
        # Find the except block
        except_line = lines[line_num - 1].rstrip("\n")
        # Get the indentation of the pass line (usually next line)
        pass_line = ""
        if line_num < len(lines):
            pass_line = lines[line_num].rstrip("\n")

        if "pass" in pass_line:
            indent = len(pass_line) - len(pass_line.lstrip())
            indent_str = " " * indent
            fixed_pass = f"{indent_str}logger.debug(f'silent except: {{e}}')"
            # Add 'as e' to the except clause, preserving exception types
            fixed_except = self._add_as_e_to_except(except_line)
            if fixed_except is None:
                return None
            return (
                f"{except_line}\n{pass_line}",
                f"{fixed_except}\n{fixed_pass}",
                "Replace silent 'except: pass' with logging",
                "Silent exception handling hides bugs. Logging makes "
                "errors visible while still preventing crashes.",
            )
        return None

    @staticmethod
    def _add_as_e_to_except(except_line: str) -> str | None:
        """Add 'as e' to an except clause, preserving exception types.

        Handles:
        - ``except:`` → ``except Exception as e:``
        - ``except OSError:`` → ``except OSError as e:``
        - ``except (OSError, ConnectionError):`` → ``except (OSError, ConnectionError) as e:``
        - ``except ... as e:`` → unchanged (already has 'as e')
        """
        if " as e:" in except_line or " as e :" in except_line:
            return except_line  # already has 'as e'
        # Bare except: → except Exception as e:
        if re.match(r"^\s*except\s*:", except_line):
            return except_line.replace("except:", "except Exception as e:")
        # Typed except: add 'as e' before the colon
        m = re.match(r"^(\s*except\s+.+?)(\s*):", except_line)
        if m:
            return f"{m.group(1)} as e:{m.group(2)}"
        return None

    def _fix_mutable_default(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Fix mutable default argument by using None sentinel."""
        line = lines[line_num - 1].rstrip("\n")

        # Find mutable defaults: = [], = {}, = set()
        mutable_patterns = [
            (r"=\s*\[\]", "= None"),
            (r"=\s*\{\}", "= None"),
            (r"=\s*set\(\)", "= None"),
        ]

        fixed = line
        param_name = ""
        for pattern, replacement in mutable_patterns:
            match = re.search(pattern, fixed)
            if match:
                # Extract the parameter name
                before_match = fixed[: match.start()]
                param_match = re.search(r"(\w+)\s*$", before_match)
                if param_match:
                    param_name = param_match.group(1)
                fixed = re.sub(pattern, replacement, fixed)

        if fixed == line or not param_name:
            return None

        # Add the sentinel check as the first line of the function body
        # Find the indentation of the next line (function body)
        body_indent = "    "  # default
        if line_num < len(lines):
            next_line = lines[line_num]
            body_indent = " " * (len(next_line) - len(next_line.lstrip()))

        # Determine the original mutable to reconstruct
        if "[]" in line:
            sentinel_init = (
                f"{body_indent}{param_name} = []"
                f" if {param_name} is None else {param_name}"
            )
        elif "{}" in line:
            sentinel_init = (
                f"{body_indent}{param_name} = {{}}"
                f" if {param_name} is None else {param_name}"
            )
        else:
            sentinel_init = (
                f"{body_indent}{param_name} = set()"
                f" if {param_name} is None else {param_name}"
            )

        return (
            line,
            f"{fixed}\n{sentinel_init}",
            f"Replace mutable default for '{param_name}' with None sentinel",
            "Mutable default arguments share state across calls, causing "
            "subtle bugs. The None sentinel pattern is the Pythonic fix.",
        )

    def _fix_missing_type_hints(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Add type hints to a function signature.

        Skipped — adding type hints requires understanding what types
        each parameter actually accepts. The previous implementation
        assumed every parameter was ``str``, which is almost always
        wrong and would introduce incorrect type annotations. Proper
        type inference requires analyzing the function body and call
        sites, which is beyond mechanical pattern-matching.
        """
        return None

    def _fix_print_in_code(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Replace print() with logger call.

        Strips print-only keyword arguments (file=, sep=, end=, flush=)
        that logger.info() does not accept. Without this, the rewrite
        produces TypeError at runtime and the fix gets reverted by the
        test gate, creating an infinite scan→fix→fail→revert loop.
        """
        line = lines[line_num - 1].rstrip("\n")

        # Replace print with logger.info
        fixed = re.sub(r"\bprint\s*\(", "logger.info(", line)
        if fixed == line:
            return None

        # Strip print-only keyword arguments that logger.info() doesn't
        # accept: file=, sep=, end=, flush=
        fixed = re.sub(r",\s*file\s*=\s*[^,\)]+", "", fixed)
        fixed = re.sub(r",\s*sep\s*=\s*[^,\)]+", "", fixed)
        fixed = re.sub(r",\s*end\s*=\s*[^,\)]+", "", fixed)
        fixed = re.sub(r",\s*flush\s*=\s*[^,\)]+", "", fixed)

        return (
            line,
            fixed,
            "Replace print() with logger.info()",
            "Using the logging framework instead of print() allows "
            "log level control, formatting, and routing to files.",
        )

    def _fix_unwrap(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Replace .unwrap() with proper error handling.

        Skipped — ``.unwrap()`` is a Rust pattern, not a Python one.
        If the bug reporter flags this in Python files, it's a false
        positive. Generating a fix would produce incorrect code.
        """
        return None

    def _fix_import_star(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Replace 'from X import *' with explicit imports.

        Skipped — converting ``from X import *`` to ``import X``
        breaks all call sites that use the imported names directly,
        and we can't know which names are needed without parsing the
        entire module and all its re-exports. This requires human
        analysis, not a mechanical fix.
        """
        return None

    def _fix_unreachable_code(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Remove unreachable code after return/raise/break/continue.

        Finds the statement at bug.line (reported as unreachable) and
        removes it. Only removes a single line — multi-line unreachable
        blocks require deeper analysis and are left for human review.
        """
        if line_num < 1 or line_num > len(lines):
            return None
        line = lines[line_num - 1].rstrip("\n")
        # Don't remove blank lines or comments — they might be intentional
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return None
        # Don't remove docstrings or decorators
        if stripped.startswith('"""') or stripped.startswith("'''"):
            return None
        if stripped.startswith("@"):
            return None
        # Remove the unreachable line
        return (
            line,
            "",  # empty string = delete the line
            "Remove unreachable code after return/raise/break/continue",
            "Dead code after a terminal statement can never execute. "
            "Removing it reduces confusion and keeps the function clean.",
        )

    def _fix_none_comparison(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Replace '== None' with 'is None' and '!= None' with 'is not None'.

        PEP 8 requires comparisons to None use 'is' / 'is not', not
        equality operators. Equality can trigger __eq__ overrides that
        behave unexpectedly; 'is' tests object identity.
        """
        if line_num < 1 or line_num > len(lines):
            return None
        line = lines[line_num - 1].rstrip("\n")
        # Replace == None → is None and != None → is not None.
        # Use word boundaries to avoid matching '==None' inside larger
        # tokens (unlikely, but defensive).
        fixed = re.sub(r"==\s*None\b", "is None", line)
        fixed = re.sub(r"!=\s*None\b", "is not None", fixed)
        if fixed == line:
            return None
        return (
            line,
            fixed,
            "Replace '== None' / '!= None' with 'is None' / 'is not None'",
            "PEP 8: comparisons to None should use 'is' / 'is not', "
            "never equality operators. Equality can trigger __eq__ "
            "overrides; 'is' tests identity.",
        )

    def _fix_is_literal_comparison(
        self, source: str, line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Replace 'is'/'is not' with '=='/'!=' when comparing to a literal.

        The edit is AST-guided: the exact operator span is taken from
        the parsed comparison, so only the offending operator changes —
        a nearby valid ``is None`` on the same line is left alone.
        """
        from ..bug_reporter import is_literal_operand

        tree = _parse_source_tree(source)
        if tree is None:
            return None
        lines = source.splitlines()
        if line_num < 1 or line_num > len(lines):
            return None
        line = lines[line_num - 1]

        for node in _compares_on_line(tree, line_num):
            operands: list[ast.expr] = [node.left, *node.comparators]
            for i, op in enumerate(node.ops):
                if not isinstance(op, (ast.Is, ast.IsNot)):
                    continue
                left_operand = operands[i]
                right_operand = node.comparators[i]
                if not (
                    is_literal_operand(left_operand)
                    or is_literal_operand(right_operand)
                ):
                    continue
                start = left_operand.end_col_offset
                end = right_operand.col_offset
                if start is None or end is None:
                    continue
                gap = line.encode("utf-8")[start:end].decode("utf-8")
                if isinstance(op, ast.IsNot):
                    new_gap = re.sub(r"\bis\s+not\b", "!=", gap)
                else:
                    new_gap = re.sub(r"\bis\b", "==", gap)
                if new_gap == gap:
                    continue
                fixed = _splice_line(line, start, end, new_gap)
                return (
                    line,
                    fixed,
                    "Replace 'is'/'is not' with '=='/'!=' when comparing "
                    "to a literal",
                    "Comparing a value to a literal with 'is' tests object "
                    "identity, which is unreliable for freshly-created "
                    "literal objects. '==' compares values.",
                )
        return None

    def _fix_type_equality_check(
        self, source: str, line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Replace 'type(x) == T' with 'isinstance(x, T)'.

        The whole comparison (both operands and the operator) is
        replaced, AST-guided so the class expression and the value
        expression are copied verbatim from the source.
        """
        from ..bug_reporter import type_call_arg

        tree = _parse_source_tree(source)
        if tree is None:
            return None
        lines = source.splitlines()
        if line_num < 1 or line_num > len(lines):
            return None
        line = lines[line_num - 1]
        data = line.encode("utf-8")

        for node in _compares_on_line(tree, line_num):
            operands: list[ast.expr] = [node.left, *node.comparators]
            for i, op in enumerate(node.ops):
                if not isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                    continue
                left_operand = operands[i]
                right_operand = node.comparators[i]
                left_arg = type_call_arg(left_operand)
                right_arg = type_call_arg(right_operand)
                if (left_arg is None) == (right_arg is None):
                    continue
                value_arg: ast.expr
                class_side: ast.expr
                if left_arg is not None:
                    value_arg, class_side = left_arg, right_operand
                else:
                    assert right_arg is not None
                    value_arg, class_side = right_arg, left_operand
                if not isinstance(
                    class_side, (ast.Name, ast.Attribute, ast.Tuple)
                ):
                    continue
                # Expand only the comparison's own operands — the value
                # and class are sub-expressions (e.g. the x inside
                # type(x)), whose surrounding parens belong to the call,
                # not to them.
                left_span = _operand_span(data, left_operand)
                right_span = _operand_span(data, right_operand)
                if (
                    value_arg.col_offset is None
                    or value_arg.end_col_offset is None
                    or class_side.col_offset is None
                    or class_side.end_col_offset is None
                    or left_span is None
                    or right_span is None
                ):
                    continue
                arg_text = data[
                    value_arg.col_offset:value_arg.end_col_offset
                ].decode("utf-8")
                class_text = data[
                    class_side.col_offset:class_side.end_col_offset
                ].decode("utf-8")
                call = f"isinstance({arg_text}, {class_text})"
                if isinstance(op, (ast.NotEq, ast.IsNot)):
                    call = f"not {call}"
                fixed = _splice_line(line, left_span[0], right_span[1], call)
                return (
                    line,
                    fixed,
                    "Replace type(x) comparison with isinstance()",
                    "type(x) == T ignores subclasses — an instance of a "
                    "subclass has a different type object. "
                    "isinstance(x, T) respects inheritance.",
                )
        return None

    def _fix_boolean_equality(
        self, source: str, line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Replace 'x == True'/'x == False' with a truthiness test."""
        tree = _parse_source_tree(source)
        if tree is None:
            return None
        lines = source.splitlines()
        if line_num < 1 or line_num > len(lines):
            return None
        line = lines[line_num - 1]
        data = line.encode("utf-8")

        for node in _compares_on_line(tree, line_num):
            if len(node.ops) != 1 or not isinstance(
                node.ops[0], (ast.Eq, ast.NotEq)
            ):
                continue
            left, right = node.left, node.comparators[0]
            bool_val: bool | None = None
            if isinstance(right, ast.Constant) and isinstance(right.value, bool):
                bool_val, other = right.value, left
            elif isinstance(left, ast.Constant) and isinstance(left.value, bool):
                bool_val, other = left.value, right
            else:
                continue
            # x == True -> x, x == False -> not x, x != True -> not x,
            # x != False -> x.
            negate = (isinstance(node.ops[0], ast.Eq) and not bool_val) or (
                isinstance(node.ops[0], ast.NotEq) and bool_val
            )
            other_span = _operand_span(data, other)
            left_span = _operand_span(data, left)
            right_span = _operand_span(data, right)
            if other_span is None or left_span is None or right_span is None:
                continue
            other_text = data[other_span[0]:other_span[1]].decode("utf-8")
            if negate:
                already_parenthesised = (
                    other_text.startswith("(") and other_text.endswith(")")
                )
                if isinstance(
                    other,
                    (
                        ast.BoolOp,
                        ast.Compare,
                        ast.BinOp,
                        ast.UnaryOp,
                        ast.IfExp,
                        ast.Lambda,
                        ast.NamedExpr,
                    ),
                ) and not already_parenthesised:
                    other_text = f"not ({other_text})"
                else:
                    other_text = f"not {other_text}"
            fixed = _splice_line(line, left_span[0], right_span[1], other_text)
            return (
                line,
                fixed,
                "Replace comparison to True/False with a truthiness test",
                "Comparing to the bool literals with '==' is redundant and "
                "fragile — a truthy non-bool compares unequal. Test the "
                "value's truthiness directly.",
            )
        return None

    def _fix_assert_tuple(
        self, source: str, line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Rewrite 'assert (cond, msg)' to 'assert cond, msg'."""
        tree = _parse_source_tree(source)
        if tree is None:
            return None
        lines = source.splitlines()
        if line_num < 1 or line_num > len(lines):
            return None
        line = lines[line_num - 1]
        data = line.encode("utf-8")

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assert)
                and node.lineno == line_num
                and getattr(node, "end_lineno", line_num) == line_num
                and node.msg is None
                and isinstance(node.test, ast.Tuple)
                and len(node.test.elts) == 2
            ):
                continue
            cond, message = node.test.elts
            if (
                cond.col_offset is None
                or cond.end_col_offset is None
                or message.col_offset is None
                or message.end_col_offset is None
                or node.test.col_offset is None
                or node.test.end_col_offset is None
            ):
                continue
            cond_text = data[cond.col_offset:cond.end_col_offset].decode("utf-8")
            msg_text = data[message.col_offset:message.end_col_offset].decode("utf-8")
            fixed = _splice_line(
                line,
                node.test.col_offset,
                node.test.end_col_offset,
                f"{cond_text}, {msg_text}",
            )
            return (
                line,
                fixed,
                "Rewrite 'assert (cond, msg)' so the assertion checks cond",
                "A tuple is always truthy, so 'assert (cond, msg)' never "
                "fails. 'assert cond, msg' actually checks cond.",
            )
        return None

    def _fix_base_exception_catch(
        self, lines: list[str], line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Replace 'except BaseException' with 'except Exception'."""
        if line_num < 1 or line_num > len(lines):
            return None
        line = lines[line_num - 1].rstrip("\n")
        fixed = re.sub(r"\bBaseException\b", "Exception", line)
        if fixed == line:
            return None
        return (
            line,
            fixed,
            "Catch Exception instead of BaseException",
            "BaseException also catches SystemExit and KeyboardInterrupt, "
            "so the program can't be stopped with Ctrl+C or shut down "
            "cleanly.",
        )

    def _fix_fstring_no_placeholder(
        self, source: str, line_num: int
    ) -> tuple[str, str, str, str] | None:
        """Drop the redundant 'f' prefix from an f-string with no placeholders."""
        tree = _parse_source_tree(source)
        if tree is None:
            return None
        lines = source.splitlines()
        if line_num < 1 or line_num > len(lines):
            return None
        line = lines[line_num - 1]
        data = line.encode("utf-8")

        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.JoinedStr)
                and node.lineno == line_num
                and getattr(node, "end_lineno", line_num) == line_num
                and not any(
                    isinstance(v, ast.FormattedValue) for v in node.values
                )
            ):
                continue
            if node.col_offset is None or node.end_col_offset is None:
                continue
            text = data[node.col_offset:node.end_col_offset].decode("utf-8")
            # Literal braces would change meaning if the prefix is dropped.
            if "{" in text:
                continue
            if (
                text[:1] in ("f", "F")
                and len(text) > 1
                and text[1] in ("'", '"')
            ):
                fixed = _splice_line(line, node.col_offset, node.col_offset + 1, "")
                return (
                    line,
                    fixed,
                    "Remove the redundant 'f' prefix from an f-string "
                    "with no placeholders",
                    "An f-string with no {…} behaves like a plain string; "
                    "the prefix usually signals a forgotten interpolation.",
                )
        return None

    # ─── Proposal generation from code analysis ─────────────────

    def _propose_from_code_analysis(self, max_proposals: int) -> list[Proposal]:
        """Generate proposals by scanning code for improvement patterns.

        This goes beyond bug reports — it looks for opportunities
        to improve code quality, add error handling, or optimize.

        Scans all experimentable files (her own Python source under
        genesis_cognitive/ or genesis_client/).
        """
        proposals: list[Proposal] = []

        # Scan all experimentable files — her own Python source.
        experimentable_files = self._collect_experimentable_files()

        for filepath in experimentable_files:
            if not os.path.isfile(filepath):
                continue
            if len(proposals) >= max_proposals:
                break

            rel_path = os.path.relpath(filepath, self.project_root)

            # Skip files we already have proposals for
            existing = {
                p.source
                for p in self._proposals
                if p.file_path == rel_path
            }

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
        for dir_prefix in SelfImprovementEngine._OWN_SOURCE_DIRS:
            dir_path = root / dir_prefix
            if not dir_path.is_dir():
                continue
            for py_file in sorted(dir_path.rglob("*.py")):
                rel = os.path.relpath(str(py_file), self.project_root)
                if SelfImprovementEngine._is_own_source(rel):
                    files.append(str(py_file))
        # Add explicit top-level files
        for file_path in sorted(SelfImprovementEngine._OWN_SOURCE_FILES):
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
            node: ast.AST, class_name: str | None = None,
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
                qualified_name = (
                    f"{class_name}.{node.name}" if class_name else node.name
                )
                source_key = f"analysis:missing_docstring:{qualified_name}:{node.lineno}"
                if source_key in existing_sources:
                    continue

                if not self._has_docstring(node):
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
                    proposal = self._propose_fstring(
                        node, lines, rel_path, source_key
                    )
                    if proposal:
                        proposals.append(proposal)

        # Long functions: skipped. Refactoring a long function requires
        # semantic understanding — extracting the wrong helper breaks
        # invariants. A TODO comment is not a real fix (see the
        # principle in _generate_fix). She should only propose changes
        # she can actually generate correctly.

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
        qualified_name = (
            f"{class_name}.{func_name}" if class_name else func_name
        )

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

        # Show the full signature in the original/proposed code
        sig_lines = [lines[i].rstrip("\n") for i in range(node.lineno - 1, sig_end_line)]
        original = "\n".join(sig_lines)
        proposed = original + "\n" + docstring

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
            desc = SelfImprovementEngine._infer_arg_description(arg.arg)
            args.append((arg.arg, desc))

        # *args
        if all_args.vararg:
            name = all_args.vararg.arg
            args.append((name, "Additional positional arguments."))

        # Keyword-only args
        for arg in all_args.kwonlyargs:
            desc = SelfImprovementEngine._infer_arg_description(arg.arg)
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
        specs = list(re.finditer(
            r"%(?:(\d+)?(?:\.(\d+))?)?([sdrfegoxXcb%])", fmt_str
        ))
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
            result_parts.append(fmt_str[last_end:spec.start()])
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

    def _record_feedback(
        self, proposal: Proposal, accepted: bool, feedback: str
    ) -> None:
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
            logger.info(
                f"Proposal #{proposal.id} rejected: '{feedback}'."
            )

    def _category_confidence(self, category: ProposalCategory) -> float:
        """Adjust confidence based on historical success rate.

        If proposals of this category have been mostly accepted,
        confidence is boosted. If mostly rejected, confidence is
        reduced. This is how she learns to propose better things.
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
        it must stay bounded even though experiments are capped at
        ``_MAX_PER_DAY`` — over months of operation the log would
        otherwise grow without limit.
        """
        self._experiment_history.append(record)
        if len(self._experiment_history) > self._max_experiments:
            del self._experiment_history[
                : len(self._experiment_history) - self._max_experiments
            ]

    # ─── Persistence ────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "proposals": [p.to_dict() for p in self._proposals],
            "next_id": self._next_id,
            "feedback_history": [f.to_dict() for f in self._feedback_history],
            "category_stats": {
                cat: list(stats) for cat, stats in self._category_stats.items()
            },
            "last_generation_time": self._last_generation_time,
            "experiments": [e.to_dict() for e in self._experiment_history],
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore state from persistence.

        Pending proposals whose target file no longer exists (e.g. the
        file was deleted between sessions) are dropped — a proposal to
        modify a vanished file can never be applied and only clutters
        the review queue. Non-pending proposals (accepted/rejected/
        applied) are kept regardless so the audit trail is preserved.
        """
        restored = [Proposal.from_dict(p) for p in data.get("proposals", [])]
        stale: list[Proposal] = []
        kept: list[Proposal] = []
        for p in restored:
            if p.status is ProposalStatus.PENDING and not os.path.exists(
                os.path.join(self.project_root, p.file_path)
            ):
                stale.append(p)
            else:
                kept.append(p)
        if stale:
            logger.info(
                "Self-improvement: dropped %d stale pending proposal(s) "
                "whose target file no longer exists: %s",
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
        ][-self._max_feedback:]
        self._category_stats = {
            cat: tuple(stats)
            for cat, stats in data.get("category_stats", {}).items()
        }
        self._last_generation_time = data.get("last_generation_time", 0.0)

        # Update _next_id if any proposals have higher IDs
        if self._proposals:
            self._next_id = max(self._next_id, max(p.id for p in self._proposals) + 1)

        # Restore heuristic experiment history
        self._experiment_history = [
            ExperimentRecord.from_dict(e)
            for e in data.get("experiments", [])
        ][-self._max_experiments:]


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

    This is the framework for Genesis to safely improve her own
    learning heuristics. The process:

    1. **Propose**: identify a heuristic function to tweak (e.g.
       relaxing a regex to catch a concept pattern she was missing).
    2. **Apply**: write the change to a scratch copy of the file.
    3. **Verify**: run the verification gate (py_compile + tests, no
       regression).
    4. **Commit or revert**: if all checks pass, keep the change.
       If any check fails, revert to the original.

    Tight guardrails:
    - Only own-source files can be modified (Genesis's Python source).
    - Never tests, never config, never the Rust substrate.
    - Max N changes per day.
    - Full audit log (ExperimentRecord for every attempt).
    - Human can hard-rollback.
    """

    # The own-source boundary is defined in SelfImprovementEngine
    # (_OWN_SOURCE_DIRS, _OWN_SOURCE_FILES, _is_own_source) — the
    # single source of truth shared by all three systems.

    # Maximum experiments per day — safety valve.
    _MAX_PER_DAY: ClassVar[int] = 3

    def __init__(
        self,
        engine: SelfImprovementEngine,
        project_root: str = ".",
    ) -> None:
        """Wire the experiment engine to the self-improvement engine and project root."""
        self.engine = engine
        self.project_root = project_root
        self._verify_script = os.path.join(
            project_root, "scripts", "verify_change.sh"
        )

    def run_experiment(
        self,
        file_path: str,
        original_code: str,
        proposed_code: str,
        description: str,
    ) -> ExperimentRecord:
        """Run a single heuristic experiment with full verification.

        Args:
            file_path: Path to the file to modify (relative to
                project root). Must be on the allowlist.
            original_code: The original code snippet to replace.
            proposed_code: The replacement code.
            description: What the experiment does.

        Returns:
            An ExperimentRecord with the outcome.
        """
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
                file_path=file_path, description=description,
                status="reverted", reason=f"could not read file: {e}",
            )

        if original_code not in source:
            return ExperimentRecord(
                file_path=file_path, description=description,
                status="reverted", reason="original code snippet not found in file",
            )

        # Apply, verify, and record
        return self._apply_and_verify(
            abs_path, source, original_code, proposed_code, file_path, description
        )

    def _check_experiment_guardrails(
        self, file_path: str, description: str
    ) -> ExperimentRecord | None:
        """Check if a file is experimentable and daily limit.

        Returns a skip record (with reason) or None if the experiment
        is allowed to proceed.
        """
        if not self._is_experimentable(file_path):
            return ExperimentRecord(
                file_path=file_path, description=description,
                status="skipped",
                reason=(
                    f"file not own source: {file_path}. "
                    f"Only Genesis's own Python source can be modified. "
                    f"Never tests, config, or the Rust substrate."
                ),
            )
        today_experiments = self._count_today_experiments()
        if today_experiments >= self._MAX_PER_DAY:
            return ExperimentRecord(
                file_path=file_path, description=description,
                status="skipped",
                reason=f"daily limit reached ({self._MAX_PER_DAY}/day)",
            )
        return None

    @classmethod
    def _is_allowlisted(cls, file_path: str) -> bool:
        """Check if a file is on the experiment allowlist (can be changed).

        Delegates to SelfImprovementEngine._is_own_source — the single
        source of truth for the own-source boundary — and additionally
        excludes protected core files (cognition, language, mind) that
        must never be auto-modified.
        """
        if not SelfImprovementEngine._is_own_source(file_path):
            return False
        if SelfImprovementEngine._is_protected_core(file_path):
            return False
        return True

    @classmethod
    def _is_experimentable(cls, file_path: str) -> bool:
        """Check if a file is safe to look at and form proposals about.

        Delegates to SelfImprovementEngine._is_own_source — the single
        source of truth for the own-source boundary. Protected core
        files are still experimentable (she can form proposals for
        them) — the protection only blocks autonomous application.
        """
        return SelfImprovementEngine._is_own_source(file_path)

    def _apply_and_verify(
        self,
        abs_path: Path,
        source: str,
        original_code: str,
        proposed_code: str,
        file_path: str,
        description: str,
    ) -> ExperimentRecord:
        """Apply the change, run verification, and return the outcome."""
        modified_source = source.replace(original_code, proposed_code, 1)
        if modified_source == source:
            return ExperimentRecord(
                file_path=file_path, description=description,
                status="reverted", reason="replacement produced no change",
            )

        try:
            abs_path.write_text(modified_source, encoding="utf-8")
        except OSError as e:
            return ExperimentRecord(
                file_path=file_path, description=description,
                status="reverted", reason=f"could not write file: {e}",
            )

        # Verify: py_compile first (fast fail)
        compile_result = get_tools().run(
            "compile_python",
            path=str(abs_path),
            project_root=self.project_root,
        )
        if not compile_result.success:
            self._revert(abs_path, source)
            return ExperimentRecord(
                file_path=file_path, description=description,
                status="reverted", reason=f"py_compile failed: {compile_result.error}",
            )

        # Verify: full test gate
        if not self._run_verify_script():
            self._revert(abs_path, source)
            record = ExperimentRecord(
                file_path=file_path, description=description,
                status="reverted",
                reason="verification gate failed (tests failed)",
            )
            self.engine.record_experiment(record)
            return record

        # All checks passed — keep the change
        record = ExperimentRecord(
            file_path=file_path, description=description,
            status="applied", reason="all verification checks passed",
        )
        self.engine.record_experiment(record)
        logger.info(
            f"Heuristic experiment applied: {description} in {file_path}"
        )
        return record

    def _run_verify_script(self) -> bool:
        """Run the verification script. Returns True if all checks pass."""
        if not os.path.exists(self._verify_script):
            logger.warning(
                f"Verify script not found: {self._verify_script}"
            )
            return False
        try:
            result = subprocess.run(
                ["bash", self._verify_script],
                capture_output=True,
                text=True,
                # The verify script runs the full Python suite (~5
                # min, includes the end-to-end daemon test) and the
                # full cargo test suite. A 300s timeout
                # could never complete a healthy run, so every
                # experiment was falsely reverted. 900s gives ~2x
                # headroom over the measured healthy runtime.
                timeout=900,  # 15 minute timeout
            )
            if result.returncode != 0:
                logger.warning(
                    f"Verification failed (exit {result.returncode}): "
                    f"{result.stderr[:200]}"
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

    def _count_today_experiments(self) -> int:
        """Count experiments applied or reverted today."""
        today = time.strftime("%Y-%m-%d")
        count = 0
        for record in self.engine._experiment_history:
            record_date = time.strftime(
                "%Y-%m-%d", time.localtime(record.timestamp / 1000)
            )
            if record_date == today and record.status in ("applied", "reverted"):
                count += 1
        return count

    @property
    def experiment_history(self) -> list[ExperimentRecord]:
        """All experiment records (copy)."""
        return list(self.engine._experiment_history)

    @property
    def applied_count(self) -> int:
        """Number of successfully applied experiments."""
        return sum(
            1 for r in self.engine._experiment_history if r.status == "applied"
        )

    @property
    def reverted_count(self) -> int:
        """Number of reverted experiments."""
        return sum(
            1 for r in self.engine._experiment_history if r.status == "reverted"
        )
