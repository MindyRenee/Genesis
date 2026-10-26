"""Mind proposals — self-improvement experiment proposals."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from self import ExperimentRecord, Proposal

logger = logging.getLogger(__name__)


class ProposalsMixin:
    """Mixin for :class:`Mind` — see module docstring."""

    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see mind.py).
        def __getattr__(self, name: str) -> Any: ...

    def proposals_status(self) -> str:
        """Return a summary of self-improvement proposals."""
        return self.self_improvement.proposals_summary()

    def get_proposal_detail(self, proposal_id: int) -> str:
        """Get full details of a specific proposal."""
        proposal = self.self_improvement.get_proposal(proposal_id)
        if proposal is None:
            return self._render_self_report(
                f"no proposal with id {proposal_id}",
                confidence=0.3,
                metadata={"proposal_id": proposal_id, "missing": True},
            )
        return proposal.describe_full()

    def clear_proposals(self) -> str:
        """Clear all self-improvement proposals."""
        count = self.self_improvement.clear_proposals()
        return self._render_self_report(
            f"cleared {count} proposals",
            metadata={"proposals_cleared": count},
        )

    def accept_proposal(self, proposal_id: int, feedback: str = "") -> str:
        """Approve a self-improvement proposal and run it as an experiment.

        This is the manual override — pending proposals run through the
        same pipeline on their own via ``run_pending_experiments``; an
        explicit /accept just jumps this one to the front of the line.

        The flow:
        1. It writes a proposal (an idea for improving its code).
        2. The experiment runs: the change is written to the live file,
           verified with py_compile + import + the full test suites,
           and automatically reverted if anything fails.
        3. If the experiment passes, the proposal is marked APPLIED.
        4. If the experiment fails, the proposal is REJECTED with
           notes explaining what went wrong. It reads the notes,
           learns, and can rewrite and submit a new proposal.

        Rejection is not a dead end — it's a learning opportunity.
        The feedback notes teach it the correct way so it can fix
        its approach and try again.
        """
        proposal = self.self_improvement.get_proposal(proposal_id)
        if proposal is None:
            return self._render_self_report(
                f"could not accept proposal {proposal_id}, "
                "it may not exist or may already be reviewed",
                confidence=0.3,
                metadata={"proposal_id": proposal_id, "accept_failed": True},
            )
        if not self.self_improvement.accept_proposal(proposal_id, feedback):
            return self._render_self_report(
                f"could not accept proposal {proposal_id}, "
                "it may not exist or may already be reviewed",
                confidence=0.3,
                metadata={"proposal_id": proposal_id, "accept_failed": True},
            )

        result = self._run_proposal_experiment(proposal)
        if result["status"] == "applied":
            return self._render_self_report(
                f"approved proposal {proposal_id}, "
                f"experiment passed verification: {result['reason']}",
                intent="inform",
                confidence=0.8,
                metadata={"proposal_id": proposal_id, "experiment_status": "applied"},
            )
        return self._render_self_report(
            f"approved proposal {proposal_id}, "
            f"but the experiment {result['status']}. "
            f"Rejected with notes — it can learn and retry.",
            intent="inform",
            confidence=0.5 if result.get("error") else 0.6,
            metadata={
                "proposal_id": proposal_id,
                "experiment_status": result["status"],
                "rejection_notes": result["notes"],
            },
        )

    def run_pending_experiments(self, max_experiments: int = 1) -> list[dict[str, Any]]:
        """Run pending proposals through the verification pipeline.

        The queue drains itself — each pending proposal whose edit is
        still applicable is applied to the live tree and verified. A
        pass marks it APPLIED; a failure reverts the file and rejects
        the proposal with teaching notes so the next version is better.
        Stale proposals (their snippet no longer matches the file) are
        withdrawn instead — retrying them can only fail.

        Returns:
            One record per attempted proposal: ``id``, ``title``,
            ``status`` (``applied``/``reverted``/``skipped``), and
            ``reason``.
        """
        outcomes: list[dict[str, Any]] = []
        for proposal in self.self_improvement.get_pending_proposals():
            if len(outcomes) >= max_experiments:
                break
            if not self.self_improvement._proposal_can_apply(proposal):
                self.self_improvement.withdraw_proposal(
                    proposal.id,
                    "stale — the snippet no longer matches the file, "
                    "re-scan and re-propose",
                )
                continue
            result = self._run_proposal_experiment(proposal)
            outcomes.append(
                {
                    "id": proposal.id,
                    "title": proposal.title,
                    "status": result["status"],
                    "reason": result["reason"],
                }
            )
        return outcomes

    def _run_proposal_experiment(self, proposal: Proposal) -> dict[str, Any]:
        """Apply a proposal's edit to the live tree and verify it.

        Shared by the manual ``/accept`` path and the autonomous
        ``run_pending_experiments`` drain. An applied experiment marks
        the proposal APPLIED; anything else rejects it with teaching
        notes — rejection is feedback, not a dead end.

        Returns:
            ``status`` (the experiment record's status), ``reason``,
            and ``notes`` when the proposal was rejected.
        """
        try:
            record = self.heuristic_experiment.run_experiment(
                file_path=proposal.file_path,
                original_code=proposal.original_code,
                proposed_code=proposal.proposed_code,
                description=proposal.description,
                line_hint=proposal.line_hint,
            )
        except Exception as e:  # noqa: BLE001
            notes = f"experiment error: {e}. The experiment framework "
            notes += "hit an unexpected error. Check the logs and try "
            notes += "a different approach."
            self.self_improvement.reject_proposal(proposal.id, notes)
            logger.warning(
                f"heuristic experiment for proposal #{proposal.id} failed: {e}"
            )
            return {"status": "error", "reason": str(e), "notes": notes, "error": True}

        if record.status == "applied":
            self.self_improvement.mark_applied(proposal.id)
            return {"status": "applied", "reason": record.reason}

        notes = self._build_rejection_notes(proposal, record)
        self.self_improvement.reject_proposal(proposal.id, notes)
        return {"status": record.status, "reason": record.reason, "notes": notes}

    def _build_rejection_notes(
        self,
        proposal: Proposal,
        record: ExperimentRecord,
    ) -> str:
        """Build teaching notes for a rejected proposal.

        The notes explain:
        - What the proposal tried to do
        - Why the experiment failed
        - What it should do differently next time

        These notes are stored in the proposal's feedback field and
        recorded in the feedback history so it can learn from them.
        """
        notes = f"Proposal: {proposal.title}\n"
        notes += f"File: {proposal.file_path}\n"
        notes += f"Experiment status: {record.status}\n"
        notes += f"Reason: {record.reason}\n\n"
        notes += "What to do differently:\n"
        if "not experimentable" in record.reason:
            notes += (
                "- This file is not experimentable. Only Python source "
                "files under python/ can "
                "be modified. Never tests, config, or the Rust "
                "substrate. Propose changes to an experimentable "
                "file instead.\n"
            )
        elif "py_compile" in record.reason:
            notes += (
                "- The proposed code has a syntax error. Check the "
                "Python syntax carefully before submitting. Use "
                "py_compile to validate before proposing.\n"
            )
        elif "verification" in record.reason or "tests" in record.reason:
            notes += (
                "- The proposed code broke tests. The change is not "
                "safe. Reconsider the "
                "approach — maybe the original code was correct, or "
                "the fix needs to be more careful.\n"
            )
        elif "original code snippet not found" in record.reason:
            notes += (
                "- The original code to replace was not found in the "
                "file. The file may have changed since the proposal "
                "was generated. Re-scan the file and generate a new "
                "proposal with the current code.\n"
            )
        elif "no change" in record.reason:
            notes += (
                "- The proposed code is identical to the original. "
                "Make sure the proposal actually changes something.\n"
            )
        else:
            notes += (
                "- The experiment failed for an unexpected reason. "
                "Read the reason above carefully and adjust the "
                "approach accordingly.\n"
            )
        notes += "\nRewrite the proposal with the correct approach and submit it again."
        return notes

    def reject_proposal(self, proposal_id: int, feedback: str = "") -> str:
        """Reject a self-improvement proposal with teaching feedback.

        The feedback should explain why the proposal is rejected and
        teach the correct way. It reads the feedback, learns from it,
        and can rewrite and submit a new proposal with the correct
        approach. Rejection is not a dead end — it's a learning
        opportunity.
        """
        if self.self_improvement.reject_proposal(proposal_id, feedback):
            return self._render_self_report(
                f"rejected proposal {proposal_id}, notes: {feedback}",
                metadata={"proposal_id": proposal_id, "rejected": True, "feedback": feedback},
            )
        return self._render_self_report(
            f"could not reject proposal {proposal_id}, it may not exist or may already be reviewed",
            confidence=0.3,
            metadata={"proposal_id": proposal_id, "reject_failed": True},
        )

    def experiments_status(self) -> str:
        """Return a summary of heuristic experiments (workstream D).

        This is an admin display (like proposals_status), not a
        conversational response — the experiment history is structured
        data (status markers, file paths, reasons) that must be shown
        verbatim, not composed by the language engine.
        """
        exp = self.heuristic_experiment
        history = exp.experiment_history
        if not history:
            return "No heuristic experiments yet."
        lines = [
            f"Heuristic experiments: {exp.applied_count} applied, "
            f"{exp.reverted_count} reverted, {len(history)} total.",
            "",
            "Recent experiments:",
        ]
        for record in history[-10:]:
            status_marker = {"applied": "+", "reverted": "-", "skipped": "?"}.get(
                record.status, "?"
            )
            lines.append(
                f"  [{status_marker}] {record.description} "
                f"({record.file_path}) — {record.status}: {record.reason}"
            )
        return "\n".join(lines)
