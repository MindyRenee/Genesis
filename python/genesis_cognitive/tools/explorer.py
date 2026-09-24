"""Filesystem exploration — Genesis's ability to explore local files.

This module gives Genesis the ability to explore its own filesystem:
read files, traverse directories, and learn from local text sources
(README files, documentation, source code comments, etc.).

## Safety

Exploration is sandboxed to the project root directory. Genesis cannot
read files outside its project tree. File size is capped to prevent
memory issues.

## Integration

The Explorer works alongside the CodeLearner (which does deep AST
analysis of source code) by handling non-code text files: markdown,
documentation, configuration, and plain text.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..concepts import ConceptNetwork, RelationType

logger = logging.getLogger(__name__)

# Pre-compiled regex for text extraction
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^\s*[-*]\s+(.+)$", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# File extensions we can learn from (text files, not code — code is CodeLearner's job)
_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".cfg", ".toml", ".ini", ".yaml", ".yml", ".json"}

# Directories to skip during exploration
_SKIP_DIRS = {
    "target",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

# Maximum file size to read (256 KB)
MAX_FILE_SIZE = 256 * 1024

# Maximum concepts to extract per file
MAX_CONCEPTS_PER_FILE = 50


@dataclass(slots=True)
class FileExplorationResult:
    """Result of exploring a single file."""

    filepath: str
    file_type: str  # "markdown", "text", "config", "json"
    lines: int
    concepts_added: int
    relationships_added: int
    summary: str  # first sentence or description


@dataclass(slots=True)
class ExplorationResult:
    """Result of exploring a directory tree."""

    files_explored: int
    concepts_added: int
    relationships_added: int
    file_results: list[FileExplorationResult] = field(default_factory=list)
    directories_visited: int = 0


class Explorer:
    """Genesis's filesystem exploration system.

    Reads and learns from text files in its project directory:
    documentation, configuration, notes, and other non-code text.
    Code files are handled by CodeLearner; this module handles everything else.

    Args:
        network: The concept network to add concepts to.
        project_root: Root directory to explore (sandboxed to this tree).
        client: Optional GenesisClient for storing exploration memories.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        project_root: str = ".",
        client=None,
    ) -> None:
        """Bind to a concept network and prepare filesystem exploration state.

        Args:
            network: The concept network to add explored concepts to.
            project_root: Root directory to explore from.
            client: Optional GenesisClient for storing exploration memories.
        """
        self.network = network
        self.project_root = Path(project_root).resolve()
        self.client = client
        self._explored_files: set[str] = set()

    def _inside_root(self, path: Path) -> bool:
        """Return True if `path` resolves to a location inside project_root."""
        try:
            return path.resolve().is_relative_to(self.project_root)
        except (ValueError, OSError):
            return False

    def explore_directory(self, path: str | None = None, max_files: int = 50) -> ExplorationResult:
        """Explore a directory tree, learning from text files.

        Walks the directory looking for .md, .txt, .rst, .yaml, .toml
        and other text files. Reads each one and extracts concepts.

        Args:
            path: Subdirectory to explore (relative to project_root).
                  If None, explores the whole project root.
            max_files: Maximum number of files to explore.

        Returns:
            Summary of what was learned.
        """
        if path:
            root = self.project_root / path
            if not self._inside_root(root):
                logger.warning(f"Refusing to explore outside project root: {path}")
                return ExplorationResult(files_explored=0, concepts_added=0, relationships_added=0)
        else:
            root = self.project_root

        if not root.exists() or not root.is_dir():
            return ExplorationResult(files_explored=0, concepts_added=0, relationships_added=0)

        result = ExplorationResult(files_explored=0, concepts_added=0, relationships_added=0)
        dirs_visited = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # Filter skip dirs in-place
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            dirs_visited += 1

            for filename in filenames:
                if result.files_explored >= max_files:
                    break

                ext = os.path.splitext(filename)[1].lower()
                if ext not in _TEXT_EXTENSIONS:
                    continue

                filepath = os.path.join(dirpath, filename)
                if filepath in self._explored_files:
                    continue

                file_result = self.explore_file(filepath)
                if file_result:
                    result.files_explored += 1
                    result.concepts_added += file_result.concepts_added
                    result.relationships_added += file_result.relationships_added
                    result.file_results.append(file_result)

                    if result.files_explored % 10 == 0:
                        logger.info(
                            f"Explored {result.files_explored} files, "
                            f"+{result.concepts_added} concepts"
                        )

            if result.files_explored >= max_files:
                break

        result.directories_visited = dirs_visited

        # Store a memory of the exploration
        if self.client and result.files_explored > 0:
            try:
                self.client.store_event(
                    timestamp=int(time.time() * 1000),
                    event_type=3,  # Observation
                    source_module=6,  # Sensory
                    salience=0.4,
                    emotional_tag=[0.5] * 12,
                    text=(
                        f"Explored {result.files_explored} files in {path or 'project root'}: "
                        f"+{result.concepts_added} concepts"
                    )[:188],
                )
            except (OSError, ConnectionError) as e:
                logger.debug(repr(e))

        return result

    def explore_file(self, filepath: str) -> FileExplorationResult | None:
        """Read and learn from a single text file.

        Args:
            filepath: Path to the file to explore.

        Returns:
            What was learned, or None if the file was skipped.
        """
        try:
            path = Path(filepath)
            if not path.exists() or not path.is_file():
                return None

            # Security: ensure file is within project root
            resolved = path.resolve()
            try:
                if not resolved.is_relative_to(self.project_root):
                    logger.warning(f"Skipping file outside project root: {filepath}")
                    return None
            except ValueError:
                logger.warning(f"Skipping file outside project root: {filepath}")
                return None

            # Size check
            size = path.stat().st_size
            if size > MAX_FILE_SIZE:
                logger.debug(f"Skipping large file ({size} bytes): {filepath}")
                return None

            # Read
            text = path.read_text(encoding="utf-8", errors="replace")
            self._explored_files.add(str(resolved))

            ext = path.suffix.lower()
            file_type = _classify_file(ext, filepath)

            # Extract concepts based on file type
            concepts_added, rels_added, summary = self._learn_from_text(text, filepath, file_type)

            lines = text.count("\n") + 1

            return FileExplorationResult(
                filepath=str(resolved.relative_to(self.project_root)),
                file_type=file_type,
                lines=lines,
                concepts_added=concepts_added,
                relationships_added=rels_added,
                summary=summary[:200],
            )

        except (OSError, UnicodeDecodeError, ValueError) as e:
            logger.warning(f"Failed to explore {filepath}: {e}")
            return None

    def _learn_from_text(self, text: str, filepath: str, file_type: str) -> tuple[int, int, str]:
        """Extract concepts from text and add them to the network.

        Returns (concepts_added, relationships_added, summary).
        """
        concepts_added = 0
        rels_added = 0

        # Add a concept for the file itself
        filename = os.path.basename(filepath)
        self.network.add_concept(
            f"file:{filename}",
            confidence=0.8,
            origin="exploration",
            properties={
                "type": file_type,
                "path": filepath,
            },
        )
        concepts_added += 1

        if file_type == "markdown":
            ca, ra = self._learn_markdown(text, filename)
            concepts_added += ca
            rels_added += ra
        elif file_type == "json":
            ca, ra = self._learn_json(text, filename)
            concepts_added += ca
            rels_added += ra
        else:
            ca = self._learn_generic_text(text, filename)
            concepts_added += ca

        summary = self._extract_summary(text, file_type)
        return concepts_added, rels_added, summary

    def _learn_markdown(self, text: str, filename: str) -> tuple[int, int]:
        """Extract concepts from markdown headings and links."""
        concepts_added = 0
        rels_added = 0

        # Extract headings as concepts
        headings = _HEADING_RE.findall(text)
        for heading in headings[:MAX_CONCEPTS_PER_FILE]:
            heading_clean = heading.strip().lower()
            if len(heading_clean) < 3 or len(heading_clean) > 80:
                continue
            self.network.add_concept(
                heading_clean,
                confidence=0.6,
                origin="exploration",
                properties={"source": filename, "type": "heading"},
            )
            self.network.add_edge(
                heading_clean,
                f"file:{filename}",
                RelationType.PART_OF,
                weight=0.5,
                origin="exploration",
            )
            concepts_added += 1
            rels_added += 1

        # Extract linked terms
        links = _LINK_RE.findall(text)
        for link in links[:20]:
            link_clean = link.strip().lower()
            if len(link_clean) < 3 or len(link_clean) > 60:
                continue
            self.network.add_concept(
                link_clean,
                confidence=0.5,
                origin="exploration",
                properties={"source": filename, "type": "link"},
            )
            concepts_added += 1

        return concepts_added, rels_added

    def _learn_json(self, text: str, filename: str) -> tuple[int, int]:
        """Extract concepts from JSON keys."""
        concepts_added = 0
        rels_added = 0

        # For JSON files, try to extract keys as concepts
        try:
            import json

            data = json.loads(text)
            if isinstance(data, dict):
                for key in list(data.keys())[:MAX_CONCEPTS_PER_FILE]:
                    key_clean = key.lower().replace("_", " ").replace("-", " ")
                    if len(key_clean) < 3 or len(key_clean) > 60:
                        continue
                    self.network.add_concept(
                        key_clean,
                        confidence=0.5,
                        origin="exploration",
                        properties={"source": filename, "type": "config_key"},
                    )
                    self.network.add_edge(
                        key_clean,
                        f"file:{filename}",
                        RelationType.PART_OF,
                        origin="exploration",
                    )
                    concepts_added += 1
                    rels_added += 1
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(repr(e))

        return concepts_added, rels_added

    def _learn_generic_text(self, text: str, filename: str) -> int:
        """Extract significant words from generic text."""
        concepts_added = 0

        # Generic text: extract words
        words = _WORD_RE.findall(text.lower())
        word_counts: dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Add top frequent words as concepts
        top_words = sorted(word_counts.items(), key=lambda x: -x[1])[:MAX_CONCEPTS_PER_FILE]
        for word, count in top_words:
            if count < 2:  # only words that appear multiple times
                continue
            self.network.add_concept(
                word,
                confidence=0.4,
                origin="exploration",
                properties={"source": filename, "frequency": count},
            )
            concepts_added += 1

        return concepts_added

    def _extract_summary(self, text: str, file_type: str) -> str:
        """Generate a summary (first meaningful sentence)."""
        # Generate a summary (first meaningful sentence)
        sentences = _SENTENCE_SPLIT_RE.split(text)
        summary = ""
        for s in sentences:
            s = s.strip()
            if len(s) > 20 and not s.startswith("#") and not s.startswith("---"):
                summary = s[:200]
                break

        if not summary:
            summary = f"{file_type} file with {text.count(chr(10)) + 1} lines"

        return summary

    def explore_path(self, path: str) -> str:
        """Explore a specific file or directory and return a human-readable summary.

        This is the user-facing method for guided exploration.

        Args:
            path: Path to explore (relative to project_root).

        Returns:
            Human-readable summary of what was found.
        """
        full_path = self.project_root / path

        if not self._inside_root(full_path):
            return f"Refusing to explore outside project root: {path}"

        if not full_path.exists():
            return f"Path not found: {path}"

        if full_path.is_file():
            result = self.explore_file(str(full_path))
            if result:
                return (
                    f"Explored {result.filepath} ({result.file_type}, {result.lines} lines). "
                    f"Learned {result.concepts_added} concepts. "
                    f"Summary: {result.summary}"
                )
            return f"Could not explore {path}"

        # Directory
        dir_result = self.explore_directory(path)
        return (
            f"Explored {dir_result.files_explored} files in {path} "
            f"across {dir_result.directories_visited} directories. "
            f"Learned {dir_result.concepts_added} concepts, "
            f"{dir_result.relationships_added} relationships."
        )

    def get_exploration_summary(self) -> dict:
        """Return a summary of exploration state."""
        return {
            "files_explored": len(self._explored_files),
            "project_root": str(self.project_root),
        }


def _classify_file(ext: str, filepath: str) -> str:
    """Classify a file by its extension and name."""
    if ext == ".md":
        return "markdown"
    if ext == ".rst":
        return "rst"
    if ext == ".json":
        return "json"
    if ext in (".yaml", ".yml"):
        return "yaml"
    if ext in (".toml", ".ini", ".cfg"):
        return "config"
    if ext == ".txt":
        return "text"
    return "text"
