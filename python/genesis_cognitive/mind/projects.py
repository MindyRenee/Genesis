"""Mind projects — project creation, management, and notes."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from genesis_client.protocol import CHEM_NAMES

from ..tools.project_creator import (
    archive_project as _archive_project,
)
from ..tools.project_creator import (
    clear_project_notes,
    create_project,
    leave_project_note,
    list_projects,
    manage_project_lifecycle,
    projects_with_notes,
    read_project_notes,
)
from ..tools.project_creator import (
    restore_project as _restore_project,
)

logger = logging.getLogger(__name__)


class ProjectsMixin:
    """Mixin for :class:`Mind` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        def __getattr__(self, name: str) -> Any: ...


    def create_project(self, description: str):
        """Create a Python project from a description.

        Composes a real, queryable knowledge-base module from its
        concept network — not a ``print()`` scaffold. It gathers
        what it knows about the topic (concepts, definitions, typed
        edges) and composes a working Python module that encodes that
        knowledge, with matching tests. Falls back to a minimal
        scaffold if it doesn't know enough about the topic.

        Args:
            description: A natural-language description / topic.

        Returns:
            A ProjectResult describing what was created.
        """
        return create_project(
            description=description,
            data_dir=self.data_dir,
            network=self.cognition.network,
        )
    def list_projects(self) -> list[dict]:
        """List all projects Genesis has created.

        Returns a list of dicts with name, path, file count, size in
        bytes, and status ("active" or "archived") for each project.
        Active projects are expanded directories; archived projects
        are compressed ``.tar.zst`` files.
        """
        return list_projects(self.data_dir)
    def archive_project(self, name: str) -> bool:
        """Archive a project to ``.tar.zst`` and reclaim its space.

        It has full autonomy over its project lifecycle. This
        compresses the project into the archive directory and removes
        the expanded files.
        """
        return _archive_project(name, self.data_dir)
    def restore_project(self, name: str) -> bool:
        """Restore an archived project from its ``.tar.zst`` archive."""
        return _restore_project(name, self.data_dir)
    def manage_projects(self) -> list[str]:
        """Autonomously manage project storage to stay within bounds.

        It reviews its own projects and archives ones that are too
        large or too numerous. Returns the list of project names it
        archived.
        """
        return manage_project_lifecycle(self.data_dir)
    def _projects_status_summary(self) -> dict[str, Any]:
        """Return a compact summary of its project portfolio for status().

        Exposes counts (active, archived, total) and total size so its
        creative output is visible through introspection without
        requiring a full ``list_projects()`` call.
        """
        try:
            projects = list_projects(self.data_dir)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"_projects_status_summary failed: {e}")
            return {"active": 0, "archived": 0, "total": 0, "total_size_bytes": 0}
        active = sum(1 for p in projects if p.get("status") == "active")
        archived = sum(1 for p in projects if p.get("status") == "archived")
        total_size = sum(p.get("size_bytes", 0) for p in projects)
        return {
            "active": active,
            "archived": archived,
            "total": len(projects),
            "total_size_bytes": total_size,
        }
    def note_project(self, project_name: str, note: str) -> bool:
        """Leave a mentor's note on one of Genesis's projects.

        The note is appended to ``NOTES.md`` in the project directory.
        Genesis absorbs notes via ``absorb_project_notes()`` — it
        reads them, stores them as long-term memory, and adds what it
        learned to its concept network.

        Args:
            project_name: The project to leave a note on.
            note: The feedback/explanation text.

        Returns:
            True if the note was written, False if the project doesn't exist.
        """
        return leave_project_note(self.data_dir, project_name, note)
    def read_project_notes(self, project_name: str) -> str | None:
        """Read the notes left on a project.

        Returns the raw notes text, or None if there are no notes.
        """
        return read_project_notes(self.data_dir, project_name)
    def absorb_project_notes(self, project_name: str | None = None) -> str:
        """Read and absorb mentor notes from project(s).

        This is how Genesis learns from feedback on its creative work.
        It reads the NOTES.md file(s) from its projects, stores each
        note as a long-term memory, adds what it learned to its
        concept network, and emits a thought about what it learned.
        The notes are cleared after absorption so it doesn't re-read
        the same feedback.

        Args:
            project_name: If given, absorb notes from just that project.
                If None, absorb notes from all projects that have them.

        Returns:
            A summary of what it absorbed, rendered through its
            language engine.
        """
        if project_name is not None:
            targets = [project_name]
        else:
            targets = projects_with_notes(self.data_dir)

        if not targets:
            return self._render_self_report(
                "no project notes to absorb",
                confidence=0.4,
                metadata={"notes_absorbed": 0},
            )

        absorbed_count = 0
        summaries: list[str] = []

        for pname in targets:
            notes = read_project_notes(self.data_dir, pname)
            if not notes:
                continue

            # Store the notes as a long-term memory. This is a
            # significant learning event, not a transient one.
            try:
                chem = self.client.get_state()
                tag = [
                    float(chem.chemicals.get(CHEM_NAMES[i], 0.0))
                    for i in range(12)
                ]
                self.memory.store_memory(
                    text=f"Mentor notes on project '{pname}': {notes[:500]}",
                    salience=0.9,
                    emotional_tag=tag,
                    # VALID_SOURCES has no "mentor" — absorbed notes are
                    # external-source learning (docs/feedback), not
                    # conversation or self-generation.
                    source="learning",
                    source_confidence=0.95,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"absorb_project_notes: memory store failed: {e}")

            # Add what it learned to its concept network. The project
            # concept already exists (added by _add_project_concept);
            # link the feedback to it.
            try:
                from ..concepts import RelationType

                feedback_concept = f"feedback_on_{pname}"
                self.cognition.network.add_concept(
                    feedback_concept,
                    confidence=0.85,
                    properties={
                        "definition": f"mentor feedback on project {pname}",
                        "type": "feedback",
                    },
                )
                self.cognition.network.add_edge(
                    pname, feedback_concept, RelationType.RELATED_TO, 0.8,
                )
                self.cognition.network.add_edge(
                    "genesis", feedback_concept, RelationType.RELATED_TO, 0.7,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"absorb_project_notes: concept add failed: {e}")

            # Clear the notes so it doesn't re-absorb them
            clear_project_notes(self.data_dir, pname)
            absorbed_count += 1
            summaries.append(pname)

        summary_text = (
            f"absorbed mentor notes on {absorbed_count} project(s): "
            f"{', '.join(summaries)}"
        )

        # Emit a live thought about what it learned
        self._emit_live_thought("code", summary_text)

        return self._render_self_report(
            summary_text,
            intent="inform",
            confidence=0.8,
            metadata={
                "notes_absorbed": absorbed_count,
                "projects": summaries,
            },
        )
