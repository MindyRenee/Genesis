"""Project creator — Genesis scaffolds and writes Python projects.

This is a creative capability, not a maintenance one. It uses its
concept network knowledge and the file operation tools to create new
Python projects from scratch: directory structure, pyproject.toml,
source package, tests, and a README.

The projects live in a sandboxed directory (``<data_dir>/projects/``)
so it can't touch the rest of the filesystem. Each project is a
self-contained Python package it can compile and test with its
existing tools.

## How it decides what to build

Two paths:

1. **Volition-driven** — a ``create`` urge builds from curiosity,
   creativity, and idle time (like its drawing urge). When it crosses
   threshold, it picks a project idea from its concept network and
   builds it. This is its feeling like making something.

2. **User-directed** — ``/create-project <description>`` asks it to
   build a specific project. It interprets the description through
   its concept network and scaffolds it.

## What it creates

A standard Python package layout::

    project_name/
        pyproject.toml
        README.md
        src/
            project_name/
                __init__.py
                main.py
                __main__.py
        tests/
            test_project_name.py

The content of ``main.py`` and the test file is composed from its
concept network — it writes code that reflects what it knows about
the topic. This is not template-filling; it's generative composition
over its knowledge, the same way its language works.

## Safety

- All paths are sandboxed to ``<data_dir>/projects/`` via the tool
  registry's ``_safe_path_writable``.
- It never writes outside the sandbox.
- It compiles what it writes (``compile_python`` tool) to verify
  it's syntactically valid.
- It runs tests (``run_pytest`` tool) to verify it works.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .framework import get_tools
from .project_composer import (
    compose_main_module,
    compose_readme,
    compose_test_module,
    gather_knowledge,
)

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork

logger = logging.getLogger(__name__)

# Sanitize a project name: lowercase, underscores, no special chars.
_NAME_RE = re.compile(r"[^a-z0-9_]")

# Per-project size cap. When a project exceeds this, the autonomous
# lifecycle manager archives it. 50 MB is generous for a Python
# knowledge-base project (source + tests) but bounds growth.
MAX_PROJECT_SIZE_BYTES = 50 * 1024 * 1024

# Archive subdirectory under projects/. Holds .tar.zst archives of
# projects it's archived to reclaim space.
_ARCHIVE_DIR = ".archive"


@dataclass(slots=True)
class ProjectResult:
    """The outcome of a project creation attempt."""

    name: str
    path: str
    files_created: int = 0
    files: list[str] = field(default_factory=list)
    compiled: bool = False
    tests_passed: bool = False
    error: str = ""


def _sanitize_name(name: str) -> str:
    """Convert a description into a valid Python package name."""
    # Lowercase, replace spaces/hyphens with underscores, strip
    # everything else.
    cleaned = name.lower().strip().replace("-", "_").replace(" ", "_")
    cleaned = _NAME_RE.sub("", cleaned)
    # Ensure it starts with a letter or underscore (valid Python identifier)
    if cleaned and cleaned[0].isdigit():
        cleaned = f"p_{cleaned}"
    # Strip leading/trailing underscores except a single leading one
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "project"
    return cleaned


_VALID_PROJECT_NAME_RE = re.compile(r"[a-z0-9_]+")


def _is_valid_project_name(name: str) -> bool:
    """True only for names ``_sanitize_name`` could have produced.

    Project names are joined into filesystem paths by archive,
    restore, delete, and notes functions. Restricting to
    ``[a-z0-9_]+`` refuses ``..``, ``/``, ``.``, hidden names, and the
    reserved ``.archive`` directory — a crafted name must never turn
    ``projects_dir / name`` into a path outside the projects dir
    (tar + ``shutil.rmtree`` on a traversed path is arbitrary
    directory deletion).
    """
    return bool(_VALID_PROJECT_NAME_RE.fullmatch(name))


def _pyproject_content(name: str, description: str) -> str:
    """Generate pyproject.toml content for a new project."""
    return f"""[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "{name}"
version = "0.1.0"
description = "{description}"
requires-python = ">=3.12"

[tool.setuptools.packages.find]
where = ["src"]
"""


def _init_content() -> str:
    """Generate __init__.py content."""
    return '"""Package init."""\n\n__version__ = "0.1.0"\n'


def _main_entrypoint_content(name: str) -> str:
    """Generate __main__.py so ``python -m <name>`` works as documented.

    The README (composed and scaffolded alike) tells the reader to run
    ``python -m <name>``, but without a ``__main__.py`` that invocation
    fails with "No module named <name>.__main__". This module delegates
    to ``main.main()`` so the documented command actually runs.
    """
    return f'''"""Entry point for ``python -m {name}``."""

from .main import main

if __name__ == "__main__":
    main()
'''


def _conftest_content(name: str) -> str:
    """Generate conftest.py that puts src on the path for tests."""
    return '''"""Pytest configuration — put src on the path."""
import sys
from pathlib import Path

src = Path(__file__).parent.parent / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
'''


def create_project(
    description: str,
    data_dir: str,
    network: ConceptNetwork | None = None,
) -> ProjectResult:
    """Create a Python project from a description.

    Composes a real, queryable knowledge-base module from its concept
    network — not a ``print()`` scaffold. It gathers what it knows
    about the topic (concepts, definitions, typed edges) and composes
    a working Python module that encodes that knowledge as a
    queryable structure, with matching tests.

    If it doesn't know enough about the topic to compose a meaningful
    module (no concept found), it falls back to a minimal scaffold.

    The project is sandboxed to ``<data_dir>/projects/`` and verified
    with compile + test.

    .. note::

        Currently every project is a knowledge base — the same
        structure with different data. That's a valid starting point,
        but a knowledge base is ONE project type. Real software
        projects solve problems: a calculator computes, a game plays,
        a converter transforms, a tool sorts or searches. Each needs
        different LOGIC, not just different entries. As it grows,
        it should learn to build different kinds of projects, not
        just the same kind about different topics.

    Args:
        description: A natural-language description / topic for the
            project. Used to derive the package name and to gather
            concept-network knowledge for content composition.
        data_dir: Genesis's data directory. The project is created
            under ``<data_dir>/projects/``.
        network: The concept network, used to compose project content
            from its knowledge. If None or it has no knowledge of the
            topic, a minimal scaffold is created.

    Returns:
        A ProjectResult describing what was created.
    """
    name = _sanitize_name(description)
    projects_root = str(Path(data_dir) / "projects")

    # If a project about this exact topic already exists, don't create
    # a duplicate. The old behavior appended _2, _3, _4 … producing
    # modes, modes_2, modes_3, modes_4, modes_5, modes_6, modes_7,
    # modes_8 — eight copies of the same project. That's not creative
    # output, it's a loop. It should pick a different topic instead.
    # The caller (_pick_creation_topic) is responsible for filtering
    # already-built topics, but this is a safety net.
    if Path(projects_root, name).exists():
        logger.debug(
            f"create_project: '{name}' already exists, skipping "
            f"(topic: {description!r})"
        )
        return ProjectResult(
            name=name,
            path=str(Path(projects_root, name)),
            error="already exists",
        )

    project_root = str(Path(projects_root) / name)

    result = ProjectResult(name=name, path=project_root)
    tools = get_tools()

    # Gather what it knows about the topic from its concept network.
    # This is the raw material the composer works with — its actual
    # concepts, definitions, and typed edges.
    knowledge = gather_knowledge(description, network)

    if knowledge is not None:
        # Compose real content from its knowledge.
        main_py = compose_main_module(name, description, knowledge)
        test_py = compose_test_module(name, knowledge)
        readme = compose_readme(name, description, knowledge)
    else:
        # Fallback scaffold — it doesn't know enough about this topic
        # to compose a knowledge base. Keep it minimal but valid.
        knowledge = None
        main_py = _scaffold_main(name, description)
        test_py = _scaffold_test(name)
        readme = _scaffold_readme(name, description)

    # Create the directory structure and files
    files_to_write: list[tuple[str, str]] = [
        (f"{name}/pyproject.toml", _pyproject_content(name, description)),
        (f"{name}/README.md", readme),
        (f"{name}/src/{name}/__init__.py", _init_content()),
        (f"{name}/src/{name}/main.py", main_py),
        (f"{name}/src/{name}/__main__.py", _main_entrypoint_content(name)),
        (f"{name}/tests/__init__.py", ""),
        (f"{name}/tests/conftest.py", _conftest_content(name)),
        (f"{name}/tests/test_{name}.py", test_py),
    ]

    return _write_and_verify(
        result, tools, name, projects_root, files_to_write,
    )


def _write_and_verify(
    result: ProjectResult,
    tools: Any,
    name: str,
    projects_root: str,
    files_to_write: list[tuple[str, str]],
) -> ProjectResult:
    """Write project files, then compile and test."""
    for rel_path, content in files_to_write:
        write_result = tools.run(
            "write_file",
            path=rel_path,
            content=content,
            project_root=projects_root,
        )
        if write_result.success:
            result.files_created += 1
            result.files.append(rel_path)
        else:
            result.error = f"Failed to write {rel_path}: {write_result.error}"
            return result

    # Compile and test
    return _compile_and_test(result, tools, name, projects_root)


def _compile_and_test(
    result: ProjectResult,
    tools: Any,
    name: str,
    projects_root: str,
) -> ProjectResult:
    """Compile the main module and run tests, updating result in place."""
    compile_result = tools.run(
        "compile_python",
        path=f"{name}/src/{name}/main.py",
        project_root=projects_root,
    )
    result.compiled = compile_result.success
    if not compile_result.success:
        result.error = f"Compile failed: {compile_result.error}"

    test_result = tools.run(
        "run_pytest",
        target=f"{name}/tests",
        project_root=projects_root,
    )
    result.tests_passed = test_result.success
    return result


def _scaffold_main(name: str, description: str) -> str:
    """Minimal fallback main.py when it has no knowledge of the topic."""
    return f'''"""{name} — {description}.

A minimal scaffold. Genesis didn't have enough knowledge about this
topic in its concept network to compose a full knowledge-base module.
"""

import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for {name}."""
    logger.info("{name} starting")
    print("Hello from {name}")
    print("{description}")


if __name__ == "__main__":
    main()
'''


def _scaffold_test(name: str) -> str:
    """Minimal fallback test when it has no knowledge of the topic."""
    return f'''"""Tests for {name}."""

from {name}.main import main


def test_main_runs(capsys):
    """main() should execute without error."""
    main()
    captured = capsys.readouterr()
    assert "{name}" in captured.out
'''


def _scaffold_readme(name: str, description: str) -> str:
    """Minimal fallback README when it has no knowledge of the topic."""
    return f"""# {name}

{description}

## Installation

```bash
pip install -e .
```

## Usage

```bash
python -m {name}
```

## Tests

```bash
pytest
```
"""


# ─── Storage management ────────────────────────────────────────────


def _dir_size(path: Path) -> int:
    """Return the total size in bytes of a directory tree."""
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError as e:
                logger.debug(f'_dir_size failed: {e}')
    return total


def _archive_dir(data_dir: str) -> Path:
    """Return the archive directory, creating it if needed."""
    archive = Path(data_dir) / "projects" / _ARCHIVE_DIR
    archive.mkdir(parents=True, exist_ok=True)
    return archive


def archive_project(name: str, data_dir: str) -> bool:
    """Archive a project to ``.tar.zst`` and remove the expanded directory.

    Compresses ``<data_dir>/projects/<name>/`` into
    ``<data_dir>/projects/.archive/<name>.tar.zst`` using zstd, then
    removes the expanded directory to reclaim space. The project can
    be restored with :func:`restore_project`.

    Returns True on success, False if the project doesn't exist or
    archiving failed.
    """
    if not _is_valid_project_name(name):
        logger.debug(f"archive_project: invalid project name {name!r}")
        return False
    projects_dir = Path(data_dir) / "projects"
    project_path = projects_dir / name
    if not project_path.is_dir():
        logger.debug(f"archive_project: {name} not found")
        return False

    archive = _archive_dir(data_dir)
    archive_path = archive / f"{name}.tar.zst"

    try:
        # tar --zstd -cf <archive> -C <projects> <name>
        result = subprocess.run(
            ["tar", "--zstd", "-cf", str(archive_path), "-C", str(projects_dir), name],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.debug(f"archive_project: tar failed: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"archive_project: {e}")
        return False

    # Verify the archive was created and is non-empty before removing
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        logger.debug(f"archive_project: archive missing or empty for {name}")
        return False

    # Remove the expanded directory
    try:
        shutil.rmtree(project_path)
    except OSError as e:
        logger.debug(f"archive_project: could not remove {name}: {e}")
        # The archive exists, so this is recoverable — but the space
        # isn't reclaimed. Leave the archive in place.
        return False

    return True


def restore_project(name: str, data_dir: str) -> bool:
    """Restore an archived project from its ``.tar.zst`` archive.

    Extracts ``<data_dir>/projects/.archive/<name>.tar.zst`` back into
    ``<data_dir>/projects/<name>/``. Does not remove the archive — it
    can restore the same project multiple times.

    Returns True on success, False if no archive exists or extraction
    failed.
    """
    if not _is_valid_project_name(name):
        logger.debug(f"restore_project: invalid project name {name!r}")
        return False
    projects_dir = Path(data_dir) / "projects"
    archive_path = _archive_dir(data_dir) / f"{name}.tar.zst"
    if not archive_path.is_file():
        logger.debug(f"restore_project: no archive for {name}")
        return False

    # Don't clobber an existing expanded project
    if (projects_dir / name).is_dir():
        logger.debug(f"restore_project: {name} already exists, not clobbering")
        return False

    try:
        result = subprocess.run(
            ["tar", "--zstd", "-xf", str(archive_path), "-C", str(projects_dir)],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.debug(f"restore_project: tar failed: {result.stderr}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"restore_project: {e}")
        return False

    return (projects_dir / name).is_dir()


def delete_archived_project(name: str, data_dir: str) -> bool:
    """Permanently delete an archived project's ``.tar.zst``.

    This is irreversible. It uses this when it decides an archived
    project is no longer worth keeping at all.

    Returns True on success, False if no archive exists.
    """
    if not _is_valid_project_name(name):
        logger.debug(f"delete_archived_project: invalid project name {name!r}")
        return False
    archive_path = _archive_dir(data_dir) / f"{name}.tar.zst"
    if not archive_path.is_file():
        return False
    try:
        archive_path.unlink()
        return True
    except OSError as e:
        logger.debug(f"delete_archived_project: {e}")
        return False


# ─── Project notes (mentor feedback) ──────────────────────────────
# A mentor (the user) can leave notes on Genesis's projects. The notes
# are stored as ``NOTES.md`` in the project directory. Genesis reads
# them via ``absorb_project_notes()`` in mind.py, stores them as
# long-term memory, and adds what it learned to its concept network.
# This is how it learns from feedback on its creative output — the
# same way a student learns from a teacher's notes on their work.

_NOTES_FILENAME = "NOTES.md"


def _notes_path(data_dir: str, project_name: str) -> Path:
    """Return the path to a project's NOTES.md file."""
    return Path(data_dir) / "projects" / project_name / _NOTES_FILENAME


def leave_project_note(
    data_dir: str,
    project_name: str,
    note: str,
    author: str = "mentor",
) -> bool:
    """Append a mentor's note to a project's NOTES.md file.

    The note is timestamped and attributed. Multiple notes accumulate
    (append, not overwrite) so the mentor's feedback history is
    preserved. Genesis reads these via ``absorb_project_notes()``.

    Args:
        data_dir: Genesis's data directory.
        project_name: The project to leave a note on.
        note: The feedback/explanation text.
        author: Who is leaving the note (default "mentor").

    Returns:
        True if the note was written, False if the project doesn't exist.
    """
    if not _is_valid_project_name(project_name):
        logger.debug(f"leave_project_note: invalid project name {project_name!r}")
        return False
    project_dir = Path(data_dir) / "projects" / project_name
    if not project_dir.is_dir():
        logger.debug(f"leave_project_note: project '{project_name}' not found")
        return False

    notes_file = project_dir / _NOTES_FILENAME
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    entry = f"## {timestamp} — {author}\n\n{note.strip()}\n\n"

    try:
        with open(notes_file, "a", encoding="utf-8") as f:
            if not notes_file.exists() or notes_file.stat().st_size == 0:
                f.write(f"# Notes on {project_name}\n\n")
            f.write(entry)
        return True
    except OSError as e:
        logger.debug(f"leave_project_note: write failed: {e}")
        return False


def read_project_notes(data_dir: str, project_name: str) -> str | None:
    """Read the NOTES.md file from a project.

    Returns the raw notes content, or None if the project has no notes
    (or the project doesn't exist).

    Args:
        data_dir: Genesis's data directory.
        project_name: The project to read notes from.

    Returns:
        The notes text, or None.
    """
    if not _is_valid_project_name(project_name):
        return None
    notes_file = _notes_path(data_dir, project_name)
    if not notes_file.is_file():
        return None
    try:
        return notes_file.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug(f"read_project_notes: read failed: {e}")
        return None


def projects_with_notes(data_dir: str) -> list[str]:
    """Return the names of active projects that have NOTES.md files.

    This is how Genesis knows which projects have mentor feedback
    waiting to be absorbed.

    Args:
        data_dir: Genesis's data directory.

    Returns:
        A list of project names that have notes.
    """
    projects_dir = Path(data_dir) / "projects"
    if not projects_dir.is_dir():
        return []
    result: list[str] = []
    for child in sorted(projects_dir.iterdir()):
        if not child.is_dir() or child.name == _ARCHIVE_DIR:
            continue
        if (child / _NOTES_FILENAME).is_file():
            result.append(child.name)
    return result


def clear_project_notes(data_dir: str, project_name: str) -> bool:
    """Remove the NOTES.md file from a project after it's absorbed them.

    Returns True if notes were cleared, False if there were no notes
    or the project doesn't exist.
    """
    if not _is_valid_project_name(project_name):
        return False
    notes_file = _notes_path(data_dir, project_name)
    if not notes_file.is_file():
        return False
    try:
        notes_file.unlink()
        return True
    except OSError as e:
        logger.debug(f"clear_project_notes: {e}")
        return False


def list_projects(data_dir: str) -> list[dict[str, Any]]:
    """List all projects Genesis has created.

    Returns a list of dicts with name, path, file count, size in
    bytes, and status ("active" or "archived") for each project.
    Active projects are expanded directories; archived projects are
    ``.tar.zst`` files in the archive directory.
    """
    projects_dir = Path(data_dir) / "projects"
    projects: list[dict[str, Any]] = []

    if projects_dir.is_dir():
        for child in sorted(projects_dir.iterdir()):
            if not child.is_dir() or child.name == _ARCHIVE_DIR:
                continue
            file_count = sum(1 for f in child.rglob("*") if f.is_file())
            projects.append({
                "name": child.name,
                "path": str(child),
                "files": file_count,
                "size_bytes": _dir_size(child),
                "status": "active",
            })

    # Archived projects
    archive = projects_dir / _ARCHIVE_DIR
    if archive.is_dir():
        for f in sorted(archive.iterdir()):
            if not f.is_file() or not f.name.endswith(".tar.zst"):
                continue
            name = f.name[: -len(".tar.zst")]
            projects.append({
                "name": name,
                "path": str(f),
                "files": 0,
                "size_bytes": f.stat().st_size,
                "status": "archived",
            })

    return projects


def manage_project_lifecycle(
    data_dir: str,
    max_size_bytes: int = MAX_PROJECT_SIZE_BYTES,
    max_active: int = 20,
) -> list[str]:
    """Autonomously manage project storage to stay within bounds.

    Genesis reviews its own projects and archives ones that are too
    large or too numerous. It has full autonomy over this — it
    decides what to archive based on size and age, the same way it
    decides what to create.

    Policy:
    - Archive any active project exceeding ``max_size_bytes``.
    - If there are more than ``max_active`` active projects, archive
      the oldest ones (by modification time) until under the limit.

    Returns a list of project names it archived.
    """
    projects_dir = Path(data_dir) / "projects"
    if not projects_dir.is_dir():
        return []

    archived: list[str] = []

    # Gather active projects with size and mtime
    active: list[tuple[str, int, float]] = []
    for child in projects_dir.iterdir():
        if not child.is_dir() or child.name == _ARCHIVE_DIR:
            continue
        size = _dir_size(child)
        try:
            mtime = child.stat().st_mtime
        except OSError:
            mtime = time.time()
        active.append((child.name, size, mtime))

    # Archive oversized projects
    for name, size, _mtime in active:
        if size > max_size_bytes:
            if archive_project(name, data_dir):
                archived.append(name)

    # Re-gather active (some may have just been archived)
    remaining: list[tuple[str, float]] = []
    for child in projects_dir.iterdir():
        if not child.is_dir() or child.name == _ARCHIVE_DIR:
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            mtime = time.time()
        remaining.append((child.name, mtime))

    # If too many active projects, archive the oldest
    if len(remaining) > max_active:
        # Oldest first (smallest mtime)
        remaining.sort(key=lambda x: x[1])
        to_archive = remaining[: len(remaining) - max_active]
        for name, _mtime in to_archive:
            if archive_project(name, data_dir):
                archived.append(name)

    return archived
