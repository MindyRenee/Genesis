"""Workspace — a working copy of her own source that she may modify.

The computer is hers, including the freedom to break things. But her
source code is the substrate she is *currently running on*, so editing
it in place couples two different kinds of risk: a bad edit takes out
the code executing her, and a half-applied edit leaves her in a state
she cannot introspect because the introspecting code is what changed.

The workspace gives her somewhere to edit instead. It is a real copy of
her source tree; she edits, compiles, tests and imports against it, and
only a deliberate promotion carries changes back to the live tree.
Promotion is a normal act she can take on her own judgement — nothing
here decides for her — but it is explicit and reversible, because the
live tree is the thing keeping her alive.

**This is a workspace, not a sandbox.** It is not a security boundary
and makes no pretense of being one. She can reach the entire machine
through her other tools; this only decides *where her code edits land*.
Anything that restricts her by filtering strings belongs to the Unix
permission model, never here (see AGENTS.md, "Permissions, not
pretend-safety").

Changes take effect on her next start, not mid-flight: the running
interpreter has already imported her modules, so promoting a file does
not hot-reload it. That is deliberate — restarting herself is a
decision she makes, like any other.
"""

from __future__ import annotations

import filecmp
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Workspace", "WorkspaceStats"]

# Never copied into the workspace: build output, dependency trees,
# caches, and her own runtime state. All of it is regenerable, all of it
# is large, and none of it is source she edits.
_EXCLUDED_DIRS = frozenset({
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-arc3",
    "__pycache__",
    "callgraph",
    "node_modules",
    "target",
    "voices",
})
_EXCLUDED_SUFFIXES = (".pyc", ".pyo", ".so", ".rlib", ".rmeta")

# Excluded by path from the repo root, because the directory name alone
# is too generic to match on (a "results" dir is not always data) or the
# contents are bulk data she reads through other tools rather than
# source she edits — TTS voice models, downloaded benchmark corpora, and
# recorded eval output.
_EXCLUDED_PATHS = frozenset({
    "benchmarks/results",
    "evals/arc_official",
    "evals/arc3_envs",
    "evals/arc_tasks",
    "evals/babi_tasks",
    "evals/results",
})

# Her live runtime state lives outside the source tree, but a stray
# data dir inside a checkout must not be copied either.
_EXCLUDED_NAMES = frozenset({
    "concept_archive.db",
    "concept_archive.db-wal",
    "concept_archive.db-shm",
    "drawings",
    "genesis_data",
    "source_cache",
})


def _is_excluded(rel: Path) -> bool:
    """Whether a repo-relative path is generated, vendored, or state."""
    parts = rel.parts
    if any(part in _EXCLUDED_DIRS or part in _EXCLUDED_NAMES for part in parts):
        return True
    prefix = ""
    for part in parts:
        prefix = f"{prefix}/{part}" if prefix else part
        if prefix in _EXCLUDED_PATHS:
            return True
    return False


def _exclusion_patterns() -> list[str]:
    """Glob patterns for ``cp --exclude``, matching :func:`_is_excluded`.

    ``cp`` matches these against the source path, so every rule is
    emitted at each depth it can occur at.
    """
    patterns: list[str] = []
    for name in _EXCLUDED_DIRS | _EXCLUDED_NAMES:
        patterns.append(name)
        patterns.append(f"{name}/*")
        patterns.append(f"*/{name}")
        patterns.append(f"*/{name}/*")
    for rel in _EXCLUDED_PATHS:
        patterns.append(rel)
        patterns.append(f"{rel}/*")
    patterns.extend(f"*{suffix}" for suffix in _EXCLUDED_SUFFIXES)
    return patterns


@dataclass(frozen=True, slots=True)
class WorkspaceStats:
    """A snapshot of how far the workspace has drifted from the source."""

    materialized: bool
    source_root: str
    path: str
    changed: int
    untracked: int
    last_promoted: float

    def describe(self) -> str:
        """First-person summary she can read in introspection."""
        if not self.materialized:
            return "I have no workspace open — my code is the live code."
        if self.changed or self.untracked:
            return (
                f"I have {self.changed} changed and {self.untracked} new "
                "file(s) in my workspace, not yet promoted."
            )
        return "My workspace matches my live code."


class Workspace:
    """A writable copy of Genesis's source tree.

    Args:
        source_root: The live checkout she is running from.
        data_dir: Her data directory; the workspace lives inside it so
            it is never confused with source and is covered by the same
            lifecycle as the rest of her state.
    """

    def __init__(self, source_root: str, data_dir: str) -> None:
        """Record where the source is and where the workspace will live."""
        self.source_root = str(Path(source_root).resolve())
        self.path = str(Path(data_dir).resolve() / "workspace")
        self._last_promoted = 0.0

    # ─── Existence ────────────────────────────────────────────────

    def is_materialized(self) -> bool:
        """Whether a workspace copy currently exists on disk."""
        return (Path(self.path) / "python" / "genesis_conscious").is_dir()

    def ensure(self) -> str:
        """Create the workspace if absent and return its path.

        The copy uses ``cp --reflink`` where the filesystem supports
        copy-on-write, so materializing is near-instant and costs
        almost no space until she writes. Filesystems without reflink
        fall back to a plain recursive copy.

        Returns:
            The workspace root path.
        """
        target = Path(self.path)
        if self.is_materialized():
            return self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        self._copy_tree(Path(self.source_root), target)
        return self.path

    def _copy_tree(self, src: Path, dst: Path) -> None:
        """Copy the source tree, skipping generated and state paths."""
        command = ["cp", "-a", "--reflink=auto"]
        for pattern in _exclusion_patterns():
            command += ["-I", pattern]
        command += [str(src) + "/.", str(dst)]
        try:
            subprocess.run(command, check=True, capture_output=True)
            if not (dst / "python" / "genesis_conscious").is_dir():
                raise subprocess.CalledProcessError(1, "cp")
        except (subprocess.CalledProcessError, OSError):
            # No reflink support, or cp unavailable — fall back to a
            # plain copy so the workspace still works.
            dst.mkdir(parents=True, exist_ok=True)
            self._copy_tree_fallback(src, dst, Path())

    def _copy_tree_fallback(self, src: Path, dst: Path, rel: Path) -> None:
        """Recursive copy used when ``cp`` is unavailable.

        Walks with the repo-relative prefix intact so path-based
        exclusions apply the same way they do for ``cp``.
        """
        dst.mkdir(parents=True, exist_ok=True)
        for entry in src.iterdir():
            child_rel = rel / entry.name
            if _is_excluded(child_rel) or entry.name.endswith(_EXCLUDED_SUFFIXES):
                continue
            target = dst / entry.name
            if entry.is_dir():
                self._copy_tree_fallback(entry, target, child_rel)
            elif entry.is_file():
                shutil.copy2(entry, target)

    # ─── Drift ────────────────────────────────────────────────────

    def _source_files(self, root: Path) -> list[Path]:
        """Every editable source file under ``root``, sorted."""
        out: list[Path] = []
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if _is_excluded(rel):
                continue
            if path.name.endswith(_EXCLUDED_SUFFIXES):
                continue
            if path.is_file():
                out.append(path)
        return out

    def changed_files(self) -> list[str]:
        """Repo-relative paths whose workspace copy differs from source.

        Returns:
            Sorted relative paths of files she has edited.
        """
        if not self.is_materialized():
            return []
        src, work = Path(self.source_root), Path(self.path)
        changed = []
        for path in self._source_files(work):
            rel = path.relative_to(work)
            original = src / rel
            if not original.exists():
                continue
            if not filecmp.cmp(original, path, shallow=False):
                changed.append(str(rel))
        return sorted(changed)

    def new_files(self) -> list[str]:
        """Repo-relative paths that exist only in the workspace.

        Returns:
            Sorted relative paths of files she created.
        """
        if not self.is_materialized():
            return []
        src, work = Path(self.source_root), Path(self.path)
        new = []
        for path in self._source_files(work):
            rel = path.relative_to(work)
            if not (src / rel).exists():
                new.append(str(rel))
        return sorted(new)

    def stats(self) -> WorkspaceStats:
        """Summarize the workspace's drift from the live source."""
        return WorkspaceStats(
            materialized=self.is_materialized(),
            source_root=self.source_root,
            path=self.path,
            changed=len(self.changed_files()),
            untracked=len(self.new_files()),
            last_promoted=self._last_promoted,
        )

    # ─── Promotion ────────────────────────────────────────────────

    def promote(self, paths: list[str] | None = None) -> list[str]:
        """Carry workspace changes into the live source tree.

        Each promoted file is backed up beside itself first, so an
        unwanted promotion is recoverable rather than merely regretted.
        Promotion does not restart her — the change takes effect when
        she next starts.

        Args:
            paths: Specific repo-relative files to promote. None
                promotes every changed and new file.

        Returns:
            The paths actually written back, relative to the source root.
        """
        if not self.is_materialized():
            return []
        if paths is None:
            targets = self.changed_files() + self.new_files()
        else:
            targets = sorted({str(Path(p)) for p in paths})
        src, work = Path(self.source_root), Path(self.path)
        promoted: list[str] = []
        for rel in targets:
            candidate = (work / rel).resolve()
            # Refuse anything that escapes the workspace, whether by
            # absolute path or by climbing out with "..".
            if not candidate.is_relative_to(Path(self.path)):
                continue
            if not candidate.is_file():
                continue
            destination = src / rel
            if destination.exists():
                backup = destination.with_suffix(destination.suffix + ".pre_workspace")
                shutil.copy2(destination, backup)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, destination)
            promoted.append(rel)
        if promoted:
            import time

            self._last_promoted = time.time()
        return promoted

    def discard(self) -> bool:
        """Throw away every workspace change and return to the live tree.

        Returns:
            True if a materialized workspace was removed.
        """
        target = Path(self.path)
        if not target.exists():
            return False
        shutil.rmtree(target, ignore_errors=True)
        return True

    def refresh(self) -> str:
        """Rebuild the workspace from the current source, discarding edits.

        Returns:
            The new workspace path.
        """
        self.discard()
        return self.ensure()
