"""Filesystem exploration — Genesis's ability to explore local files.

This module gives Genesis the ability to explore its own filesystem:
read files, traverse directories, and learn from what it finds.

## Two kinds of access — looking vs. remembering

Genesis reads *everything* — no file type is skipped:

- **Text files** are read and learned: concepts are extracted into the
  concept network (markdown headings, JSON keys, frequent terms).
- **Binary files** are read and catalogued: presence, size, hash, and
  format are remembered — the content itself is not text-learnable.
- **Oversized files** are catalogued by presence and size; their first
  ``MAX_FILE_SIZE`` bytes remain readable via ``read_file``.

``read_file`` is the raw organ: it returns a file's actual content —
text or bytes — transiently, for anything inside the allowed roots,
including pseudo-fs files. It memorizes nothing: it is how she *looks*
at a file, as opposed to how she *remembers* one.

## Scope

Read scope is the whole filesystem by default (owner-granted), narrowed
by ``GENESIS_EXPLORE_ROOTS`` or an explicit ``allowed_roots``. Symlinks
are resolved before the scope check, so a link inside an allowed root
cannot reach outside it. Pseudo-filesystems (/proc, /sys, ...) are not
walked or memorized — they are kernel interfaces, not documents — but
remain readable through ``read_file``. File size is capped to prevent
memory issues.

## Integration

The Explorer works alongside the CodeLearner (which does deep AST
analysis of source code) by handling non-code files: markdown,
documentation, configuration, plain text, and binary artefacts.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from temporal_lobe.concepts import ConceptNetwork, RelationType

from genesis_client.swallow import note_swallowed
from infrastructure.text import sentences as _sentences

logger = logging.getLogger(__name__)

# Pre-compiled regex for text extraction
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^\s*[-*]\s+(.+)$", re.MULTILINE)
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

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

# Pseudo-filesystems: nothing under them is a document she can learn.
# They are excluded from walking and memorizing only — ``read_file``
# still opens them transiently.
_PSEUDO_ROOTS = {"/proc", "/sys", "/dev", "/run", "/boot", "/snap", "/lost+found"}
# Their *names* as they appear under "/" — pruned when walking root.
_ROOT_PRUNE = {p.lstrip("/") for p in _PSEUDO_ROOTS}


def default_allowed_roots() -> list[str]:
    """Read scope for the explorer.

    Whole filesystem by default (``.``-denylist rules still apply).
    ``GENESIS_EXPLORE_ROOTS`` (os.pathsep-separated) narrows it for
    deployments that want her confinement back.
    """
    raw = os.environ.get("GENESIS_EXPLORE_ROOTS", "").strip()
    if raw:
        return [r for r in raw.split(os.pathsep) if r]
    return ["/"]


@dataclass(slots=True)
class FileExplorationResult:
    """Result of exploring a single file."""

    filepath: str
    file_type: str  # "markdown", "text", "config", "json", "binary", "oversized"
    lines: int
    concepts_added: int
    relationships_added: int
    summary: str  # first sentence or description


@dataclass(slots=True)
class FileRead:
    """A raw, transient read of a file — the organ's look, not a memory.

    Nothing in a FileRead is committed to the concept network; it is
    what ``read_file`` hands back when Genesis looks at a file directly.

    Attributes:
        filepath: resolved path that was read.
        size: file size in bytes.
        is_binary: True when the content is not UTF-8 text.
        truncated: True when only the first ``MAX_FILE_SIZE`` bytes
            were read.
        content: decoded text (str) or raw bytes.
        sha256: hex digest of the bytes actually read.
    """

    filepath: str
    size: int
    is_binary: bool
    truncated: bool
    content: str | bytes
    sha256: str


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

    Reads and learns from text files anywhere in the allowed roots —
    documentation, configuration, notes, and other non-code text.
    Code files are handled by CodeLearner; this module handles everything else.

    Args:
        network: The concept network to add concepts to.
        project_root: Root directory that relative paths resolve against.
        client: Optional GenesisClient for storing exploration memories.
        allowed_roots: Read scope. Defaults to the project root; pass
            ``default_allowed_roots()`` for the deployment-wide scope
            (``GENESIS_EXPLORE_ROOTS``, else the whole filesystem).
    """

    def __init__(
        self,
        network: ConceptNetwork,
        project_root: str = ".",
        client=None,
        allowed_roots: list[str] | None = None,
    ) -> None:
        """Bind to a concept network and prepare filesystem exploration state.

        Args:
            network: The concept network to add explored concepts to.
            project_root: Root directory to explore from — also the
                default scope when ``allowed_roots`` is not given.
            client: Optional GenesisClient for storing exploration memories.
            allowed_roots: Everywhere she may read.
        """
        self.network = network
        self.project_root = Path(project_root).resolve()
        self.client = client
        roots = allowed_roots if allowed_roots is not None else [str(self.project_root)]
        self._allowed_roots = tuple(Path(r).expanduser().resolve() for r in roots)
        self._explored_files: set[str] = set()
        # Corpus statistics for term salience — document frequency of
        # each term over the files she has explored, and the document
        # count. The exploration stream IS her corpus: a term's
        # information content is its specificity, not its raw count.
        self._doc_freq: dict[str, int] = {}
        self._doc_count = 0

    def _inside_root(self, path: Path) -> bool:
        """True if `path` is inside an allowed root (memorizing scope).

        Pseudo-filesystems are excluded — they are kernel interfaces,
        not documents.
        """
        try:
            resolved = path.resolve()
        except OSError:
            return False
        if any(resolved.is_relative_to(Path(p)) for p in _PSEUDO_ROOTS):
            return False
        return any(resolved.is_relative_to(root) for root in self._allowed_roots)

    def _in_allowed_roots(self, path: Path) -> bool:
        """True if `path` resolves inside an allowed root (raw-read scope).

        The look-scope excludes nothing the deployment granted — even
        pseudo-fs entries are readable here, transiently.
        """
        try:
            resolved = path.resolve()
        except OSError:
            return False
        return any(resolved.is_relative_to(root) for root in self._allowed_roots)

    def explore_directory(self, path: str | None = None, max_files: int = 50) -> ExplorationResult:
        """Explore a directory tree, learning from every file it finds.

        Walks the directory and visits every file: text is learned,
        binaries and oversized files are catalogued. Nothing in scope
        is skipped.

        Args:
            path: Subdirectory to explore (relative to project_root).
                  If None, explores the whole project root.
            max_files: Maximum number of files to explore.

        Returns:
            Summary of what was learned.
        """
        if path:
            root = Path(path).expanduser()
            if not root.is_absolute():
                root = self.project_root / root
            if not self._inside_root(root):
                logger.warning(f"Refusing to explore outside allowed scope: {path}")
                return ExplorationResult(files_explored=0, concepts_added=0, relationships_added=0)
        else:
            root = self.project_root

        if not root.exists() or not root.is_dir():
            return ExplorationResult(files_explored=0, concepts_added=0, relationships_added=0)

        result = ExplorationResult(files_explored=0, concepts_added=0, relationships_added=0)
        dirs_visited = 0

        for dirpath, dirnames, filenames in os.walk(root):
            # Filter skip dirs in-place — plus, at the filesystem root,
            # the pseudo-filesystems.
            dirnames[:] = [
                d
                for d in dirnames
                if d not in _SKIP_DIRS
                and not d.startswith(".")
                and not (dirpath == "/" and d in _ROOT_PRUNE)
            ]
            dirs_visited += 1

            for filename in filenames:
                if result.files_explored >= max_files:
                    break

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
                note_swallowed(
                    "tools.explorer.explore_directory",
                    e,
                )

        return result

    def read_file(self, filepath: str) -> FileRead | None:
        """Read a file's actual content — text or bytes — without learning it.

        This is the raw organ: anything inside the allowed roots can be
        looked at — binaries, dotfiles, even pseudo-fs entries.
        Nothing read here is committed to the concept network
        or remembered; it is a transient look, not a memory.

        Args:
            filepath: Path to the file (absolute, or resolved relative
                to project_root).

        Returns:
            A FileRead with the content, or None if the path is out of
            scope, missing, not a regular file, or unreadable.
        """
        path = Path(filepath).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        try:
            resolved = path.resolve()
        except OSError:
            return None
        if not self._in_allowed_roots(resolved):
            logger.warning(f"read_file outside allowed scope: {filepath}")
            return None
        return self._read(resolved)

    def _read(self, path: Path) -> FileRead | None:
        """Read up to ``MAX_FILE_SIZE`` bytes of a resolved, in-scope path."""
        try:
            if not path.is_file():
                return None
            size = path.stat().st_size
            with path.open("rb") as fh:
                data = fh.read(MAX_FILE_SIZE + 1)
        except OSError as e:
            note_swallowed("tools.explorer._read", e)
            return None
        truncated = len(data) > MAX_FILE_SIZE
        if truncated:
            data = data[:MAX_FILE_SIZE]
        digest = hashlib.sha256(data).hexdigest()
        text = _decode_text(data)
        if text is None:
            return FileRead(str(path), size, True, truncated, data, digest)
        return FileRead(str(path), size, False, truncated, text, digest)

    def explore_file(self, filepath: str) -> FileExplorationResult | None:
        """Read and learn from a single file — no file type is skipped.

        Text is learned into the concept network; binaries and
        oversized files are catalogued by presence, size, hash, and
        format. The raw content of any file remains available via
        ``read_file``.

        Args:
            filepath: Path to the file to explore.

        Returns:
            What was learned, or None if the file is out of scope,
            missing, or unreadable.
        """
        try:
            gated = self._gate(filepath)
            if not isinstance(gated, tuple):
                return gated
            resolved, display, size = gated

            read = self._read(resolved)
            if read is None:
                return None
            self._explored_files.add(str(resolved))

            return self._content_result(resolved, display, read, size, filepath)
        except (OSError, UnicodeDecodeError, ValueError) as e:
            logger.warning(f"Failed to explore {filepath}: {e}")
            return None

    def _display(self, resolved: Path) -> str:
        try:
            return str(resolved.relative_to(self.project_root))
        except ValueError:
            return str(resolved)

    def _gate(
        self,
        filepath: str,
    ) -> tuple[Path, str, int] | FileExplorationResult | None:
        """The artefact gate — scope and size decide before any read.

        Returns ``(resolved, display, size)`` to continue into content,
        a catalogue result when the artefact is gated by its class, or
        None when the path is missing or out of scope.
        """
        path = Path(filepath)
        if not path.exists() or not path.is_file():
            return None

        resolved = path.resolve()
        if not self._inside_root(resolved):
            logger.warning(f"Skipping file outside allowed scope: {filepath}")
            return None

        size = path.stat().st_size
        display = self._display(resolved)

        # Beyond the read cap: catalogue the artefact by presence.
        if size > MAX_FILE_SIZE:
            self._learn_oversized(resolved, size)
            self._explored_files.add(str(resolved))
            return _catalog_result(
                display,
                "oversized",
                f"{size} bytes — beyond the read cap; readable via read_file",
            )

        return resolved, display, size

    def _content_result(
        self,
        resolved: Path,
        display: str,
        read: FileRead,
        size: int,
        filepath: str,
    ) -> FileExplorationResult:
        """Content dispatch: binaries catalogue, text learns."""
        if read.is_binary:
            concepts = self._learn_binary(resolved, read)
            fmt = _sniff_format(
                read.content if isinstance(read.content, bytes) else b"",
                resolved.suffix.lower(),
            )
            return _catalog_result(
                display,
                "binary",
                f"binary file ({fmt}, {size} bytes)",
                concepts_added=concepts,
            )

        text = read.content if isinstance(read.content, str) else ""

        file_type = _classify_file(resolved.suffix.lower(), filepath)
        concepts_added, rels_added, summary = self._learn_from_text(text, filepath, file_type)

        return FileExplorationResult(
            filepath=display,
            file_type=file_type,
            lines=text.count("\n") + 1,
            concepts_added=concepts_added,
            relationships_added=rels_added,
            summary=summary[:200],
        )

    def _learn_oversized(self, path: Path, size: int) -> None:
        """Catalogue a file beyond the read cap — presence and size."""
        self.network.add_concept(
            f"file:{path.name}",
            confidence=0.6,
            origin="exploration",
            properties={
                "type": "oversized",
                "path": str(path),
                "size": size,
                "format": _classify_file(path.suffix.lower(), str(path)),
            },
        )

    def _learn_binary(self, path: Path, read: FileRead) -> int:
        """Catalogue a binary file — presence, size, hash, format."""
        raw = read.content if isinstance(read.content, bytes) else b""
        self.network.add_concept(
            f"file:{path.name}",
            confidence=0.6,
            origin="exploration",
            properties={
                "type": "binary",
                "path": str(path),
                "size": read.size,
                "sha256": read.sha256,
                "format": _sniff_format(raw, path.suffix.lower()),
            },
        )
        return 1

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
            note_swallowed(
                "tools.explorer._learn_json",
                e,
            )

        return concepts_added, rels_added

    def _learn_generic_text(self, text: str, filename: str) -> int:
        """Extract informative terms via smoothed TF-IDF salience.

        Raw frequency is not information — a word repeated in every
        file tells her nothing. Salience is term frequency × inverse
        document frequency over her exploration corpus:

            tf(w, d)  = count(w, d) / |d|
            idf(w)    = ln((N + 1) / (df(w) + 1)) + 1
            score(w)  = tf(w, d) · idf(w)

        The +1 smoothing keeps unseen terms finite (idf → ln(N+1)+1)
        and ubiquitous terms nonzero but small (idf → 1). Concepts
        enter by score rank — specificity is what survives.

        The document's term histogram is folded into the corpus only
        after scoring, so a file's own words can't inflate their idf.
        """
        concepts_added = 0

        words = _WORD_RE.findall(text.lower())
        word_counts: dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        n_terms = len(words) or 1
        n_docs = self._doc_count
        scores = {
            w: (count / n_terms) * (math.log((n_docs + 1) / (self._doc_freq.get(w, 0) + 1)) + 1.0)
            for w, count in word_counts.items()
        }
        top_words = sorted(scores.items(), key=lambda x: -x[1])[:MAX_CONCEPTS_PER_FILE]
        for word, score in top_words:
            if word_counts[word] < 2 or score <= 0.0:
                continue
            self.network.add_concept(
                word,
                confidence=0.4,
                origin="exploration",
                properties={
                    "source": filename,
                    "frequency": word_counts[word],
                    "salience": score,
                },
            )
            concepts_added += 1

        # Fold this document into the corpus AFTER scoring — the
        # document may not vote on its own specificity.
        self._doc_count += 1
        for word in word_counts:
            self._doc_freq[word] = self._doc_freq.get(word, 0) + 1

        return concepts_added

    def _extract_summary(self, text: str, file_type: str) -> str:
        """Generate a summary (first meaningful sentence)."""
        # Generate a summary (first meaningful sentence)
        sentences = _sentences(text)
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
            path: Path to explore — relative to project_root or
                absolute within the allowed roots.

        Returns:
            Human-readable summary of what was found.
        """
        full_path = Path(path).expanduser()
        if not full_path.is_absolute():
            full_path = self.project_root / full_path

        if not self._inside_root(full_path):
            return f"Refusing to explore outside allowed scope: {path}"

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


# Bytes that count as printable for the legacy-encoding fallback:
# ASCII printable + whitespace + the Latin-1 supplement (everything
# below 0xA0 that isn't already covered is a control character).
_PRINTABLE_BYTES = (
    frozenset(range(0x20, 0x7F)) | frozenset((9, 10, 13)) | frozenset(range(0xA0, 0x100))
)


def _catalog_result(
    display: str,
    file_type: str,
    summary: str,
    *,
    concepts_added: int = 1,
) -> FileExplorationResult:
    """Result for a file remembered as an artefact, not read as text."""
    return FileExplorationResult(
        filepath=display,
        file_type=file_type,
        lines=0,
        concepts_added=concepts_added,
        relationships_added=0,
        summary=summary,
    )


def _decode_text(data: bytes) -> str | None:
    """Best-effort text decode; ``None`` means genuinely binary.

    Order: BOM-marked encodings → UTF-16 without a BOM (alternating
    NULs on one side, printable on the other) → a NUL gate → UTF-8 →
    mostly-printable Latin-1. Only when every plausible text encoding
    fails is the content binary.
    """
    if not data:
        return ""
    for bom, enc in (
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
        (b"\xef\xbb\xbf", "utf-8"),
    ):
        if data.startswith(bom):
            try:
                return data.decode(enc)
            except UnicodeDecodeError:
                break
    head = data[:8192]
    # UTF-16 without a BOM: NULs alternating on one side of mostly-ASCII
    # text. The text side must be genuinely printable — an all-zeros
    # binary has NULs on *both* sides and must not decode to NUL text.
    if len(head) >= 8:
        pairs = head[: len(head) - (len(head) % 2)]
        half = len(pairs) // 2
        for nul_side, text_side, enc in (
            (pairs[0::2], pairs[1::2], "utf-16-be"),
            (pairs[1::2], pairs[0::2], "utf-16-le"),
        ):
            nul_frac = sum(1 for b in nul_side if b == 0) / half
            text_frac = sum(1 for b in text_side if b in _PRINTABLE_BYTES) / half
            if nul_frac > 0.6 and text_frac > 0.5:
                try:
                    return data.decode(enc)
                except UnicodeDecodeError:
                    break
    # What remains NUL-dense is not utf-16-shaped text — it is binary.
    if b"\x00" in head:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # Legacy encodings (latin-1 & friends): decodable as text when
        # the content is overwhelmingly printable.
        if head and sum(1 for b in head if b in _PRINTABLE_BYTES) / len(head) >= 0.85:
            return data.decode("latin-1")
        return None


# Magic-byte prefixes → format labels for cataloguing binary files.
_MAGIC_FORMATS: tuple[tuple[bytes, str], ...] = (
    (b"\x7fELF", "elf"),
    (b"\x89PNG", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"PK\x03\x04", "zip"),
    (b"\x1f\x8b", "gzip"),
    (b"SQLite format 3", "sqlite"),
    (b"RIFF", "riff"),
    (b"\x89HDF", "hdf5"),
    (b"%PDF", "pdf"),
    (b"OggS", "ogg"),
    (b"fLaC", "flac"),
)


def _sniff_format(data: bytes, ext: str) -> str:
    """Best-effort format label for binary content (magic bytes, else extension)."""
    for magic, fmt in _MAGIC_FORMATS:
        if data.startswith(magic):
            return fmt
    if data[4:8] == b"ftyp":
        return "mp4-family"
    return ext.lstrip(".") or "binary"
