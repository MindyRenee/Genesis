"""Mind proposals — self-improvement experiment proposals."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..self import ExperimentRecord, Proposal

logger = logging.getLogger(__name__)


class ProposalsMixin:
    """Mixin for :class:`Mind` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
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

        The flow:
        1. It writes a proposal (an idea for improving its code).
        2. You review it and approve it with /accept N.
        3. The experiment runs: code is applied, verified with
           py_compile + tests, and automatically reverted
           if anything fails.
        4. If the experiment passes, the proposal is marked APPLIED.
        5. If the experiment fails, the proposal is REJECTED with
           notes explaining what went wrong. It reads the notes,
           learns, and can rewrite and submit a new proposal.

        There is no bypass path. Every code change goes through the
        verification pipeline. The only exceptions are the autonomous
        bug fixes (silent_except, unreachable_code, print_in_code)
        which are safe mechanical changes that don't need the pipeline.

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

        # Run the experiment: apply the code change with full
        # verification. If it fails, reject with notes so it can
        # learn and rewrite.
        try:
            record = self.heuristic_experiment.run_experiment(
                file_path=proposal.file_path,
                original_code=proposal.original_code,
                proposed_code=proposal.proposed_code,
                description=proposal.description,
            )
            status = record.status
            if status == "applied":
                # Experiment passed — mark as applied
                self.self_improvement.mark_applied(proposal_id)
                return self._render_self_report(
                    f"approved proposal {proposal_id}, "
                    f"experiment passed verification: {record.reason}",
                    intent="inform",
                    confidence=0.8,
                    metadata={"proposal_id": proposal_id, "experiment_status": status},
                )
            else:
                # Experiment failed or was skipped — reject with
                # notes explaining what went wrong. It reads the
                # notes, learns, and can rewrite and submit a new
                # proposal with the correct approach.
                notes = self._build_rejection_notes(proposal, record)
                self.self_improvement.reject_proposal(
                    proposal_id, notes,
                )
                return self._render_self_report(
                    f"approved proposal {proposal_id}, "
                    f"but the experiment {status}. "
                    f"Rejected with notes — it can learn and retry.",
                    intent="inform",
                    confidence=0.6,
                    metadata={
                        "proposal_id": proposal_id,
                        "experiment_status": status,
                        "rejection_notes": notes,
                    },
                )
        except Exception as e:  # noqa: BLE001
            # Experiment framework error — reject with notes
            notes = f"experiment error: {e}. The experiment framework "
            notes += "hit an unexpected error. Check the logs and try "
            notes += "a different approach."
            self.self_improvement.reject_proposal(proposal_id, notes)
            logger.warning(f"heuristic experiment for proposal #{proposal_id} failed: {e}")
            return self._render_self_report(
                f"approved proposal {proposal_id}, "
                f"but the experiment hit an error. "
                f"Rejected with notes — it can learn and retry.",
                confidence=0.5,
                metadata={
                    "proposal_id": proposal_id,
                    "experiment_error": True,
                    "rejection_notes": notes,
                },
            )
    def _build_rejection_notes(
        self, proposal: Proposal, record: ExperimentRecord,
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
                "files in genesis_cognitive/ and genesis_client/ can "
                "be modified. Never tests, config, or the Rust "
                "substrate. Propose changes to an experimentable "
                "file instead.\n"
            )
        elif "daily limit" in record.reason:
            notes += (
                "- The daily experiment limit was reached. Wait until "
                "tomorrow or ask the human to increase the limit.\n"
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
        notes += (
            "\nRewrite the proposal with the correct approach and "
            "submit it again."
        )
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
