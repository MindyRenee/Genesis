"""Code learner — Genesis reads and understands her own source code.

This module gives Genesis the ability to introspect on her own codebase.
She reads Python files (via the built-in ``ast`` module) and Rust files
(via a lightweight regex parser — no external dependencies), then adds
what she finds to her :class:`ConceptNetwork`.

Each function, class, struct, enum, trait, and module becomes a concept.
Relationships between them (calls, imports, defines, implements) become
edges. A summary of each file is stored as a memory via the daemon
client, so she remembers what she has read.

Why this matters
----------------
Genesis is an artificial mind. To grow, she must understand herself.
Reading her own code is the programming analogue of metacognition:
she builds a model of her own structure, which lets her reason about
how she works, where her complexity lives, and how her parts connect.
"""

from __future__ import annotations

import ast
import logging
import math
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..concepts import RelationType

__all__ = [
    "CodeLearner",
    "CodeLearningResult",
    "CodeSpacedRepetition",
    "FileLearningResult",
]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from genesis_client.client import GenesisClient

    from ..concepts import ConceptNetwork

# ─── Data structures ───────────────────────────────────────────────────


@dataclass(slots=True)
class FileLearningResult:
    """What Genesis learned from a single file."""

    filepath: str
    language: str  # "python" or "rust"
    concepts_added: int
    relationships_added: int
    functions: int
    classes: int  # or structs for Rust
    lines: int


@dataclass(slots=True)
class CodeLearningResult:
    """Aggregate summary of a full codebase learning pass."""

    files_analyzed: int
    concepts_added: int
    relationships_added: int
    total_functions: int
    total_classes: int
    total_lines: int
    file_results: list[FileLearningResult] = field(default_factory=list)


# ─── Code-specific spaced repetition ──────────────────────────────────


class CodeSpacedRepetition:
    """Spaced repetition scheduling for code concept re-analysis.

    Applies the Ebbinghaus forgetting curve to code understanding:
    important code concepts need periodic re-analysis to maintain
    strong representations in the concept network. Without
    re-analysis, the neural-like associations weaken over time.

    The scheduling follows a modified SM-2 algorithm (Wozniak &
    Gorzelanczyk, 1994), adapted for code:

    - Initial review interval: 1 hour × importance
    - Each subsequent review multiplies the interval by 2.5
    - Importance (0–1) scales the initial interval and the
      strengthening factor

    The forgetting curve is modeled as:
        R = exp(-t / S)
    where R is retention, t is time since last review, and S is
    stability (proportional to the number of reviews and importance).

    References:
    - Ebbinghaus, H. (1885). *Memory: A Contribution to Experimental
      Psychology*.
    - Wozniak, P. A., & Gorzelanczyk, E. J. (1994). Optimization of
      repetition spacing in the practice of learning (SuperMemo).
    """

    # Base interval in seconds for the first review
    _BASE_INTERVAL = 3600.0  # 1 hour

    # Interval multiplier for subsequent reviews
    _INTERVAL_MULTIPLIER = 2.5

    # Minimum interval (don't re-analyze more often than this)
    _MIN_INTERVAL = 300.0  # 5 minutes

    def __init__(self) -> None:
        """Initialize per-file tracking dicts.

        Tracks last analysis time, review counts, and importance scores
        for each analyzed source file.
        """
        # file → timestamp of last analysis
        self.last_analyzed: dict[str, float] = {}
        # file → number of times analyzed
        self._review_counts: dict[str, int] = {}
        # file → importance score (0–1)
        self._importance: dict[str, float] = {}

    def schedule_reanalysis(self, file: str, importance: float) -> float:
        """Schedule the next re-analysis time for a file.

        Args:
            file: The file path to schedule.
            importance: The importance of this file (0–1), based on
                complexity, centrality, or other heuristics.

        Returns:
            Time in seconds until the next recommended re-analysis.
            Returns 0.0 if the file is due for re-analysis now.
        """
        importance = max(0.0, min(1.0, importance))
        self._importance[file] = importance

        review_count = self._review_counts.get(file, 0)
        now = time.time()
        last_time = self.last_analyzed.get(file, 0.0)

        # Compute the scheduled interval
        if review_count == 0:
            # First review: base interval scaled by importance
            interval = self._BASE_INTERVAL * (0.5 + importance)
        else:
            # Subsequent reviews: multiply by interval multiplier
            # More important files have longer intervals (they're
            # reviewed more thoroughly each time)
            prev_interval = self._BASE_INTERVAL * (0.5 + importance)
            for _ in range(review_count):
                prev_interval *= self._INTERVAL_MULTIPLIER
            interval = prev_interval

        # Time until next review
        time_since = now - last_time
        time_until = interval - time_since

        return max(0.0, time_until)

    def record_analysis(self, file: str, importance: float = 0.5) -> None:
        """Record that a file has been analyzed.

        Updates the last-analyzed timestamp and increments the review
        count. Should be called after each analysis pass.

        Args:
            file: The file path that was analyzed.
            importance: The importance score for this file (0–1).
        """
        self.last_analyzed[file] = time.time()
        self._review_counts[file] = self._review_counts.get(file, 0) + 1
        self._importance[file] = max(0.0, min(1.0, importance))

    def get_due_files(self, files: list[str]) -> list[str]:
        """Get files that are due for re-analysis.

        Args:
            files: List of file paths to check.

        Returns:
            List of files due for re-analysis, sorted by overdue
            amount (most overdue first).
        """
        now = time.time()
        due: list[tuple[float, str]] = []  # (overdue_seconds, file)

        for file in files:
            importance = self._importance.get(file, 0.5)
            time_until = self.schedule_reanalysis(file, importance)
            if time_until <= 0:
                last = self.last_analyzed.get(file, 0.0)
                overdue = now - last if last > 0 else float("inf")
                due.append((overdue, file))

        # Sort by most overdue first
        due.sort(key=lambda x: -x[0])
        return [file for _, file in due]

    def retention(self, file: str) -> float:
        """Estimate current retention for a file (0–1).

        Based on the Ebbinghaus forgetting curve:
            R = exp(-t / S)
        where t is time since last review and S is stability
        (proportional to review count and importance).

        Args:
            file: The file path to check.

        Returns:
            Estimated retention (0–1). 1.0 means perfectly retained,
            0.0 means completely forgotten.
        """
        last = self.last_analyzed.get(file)
        if last is None:
            return 0.0  # never analyzed → no retention

        review_count = self._review_counts.get(file, 0)
        importance = self._importance.get(file, 0.5)

        # Stability grows with each review and with importance
        stability = self._BASE_INTERVAL * (0.5 + importance) * (1 + review_count)

        t = time.time() - last
        return float(math.exp(-t / stability))

    def strengthen_concept(self, file: str, concept_name: str) -> float:
        """Compute a confidence boost for re-analyzing a concept.

        Each re-analysis strengthens the concept's representation.
        The boost is proportional to the number of previous reviews
        and the file's importance.

        Args:
            file: The file containing the concept.
            concept_name: The concept to strengthen (unused in
                computation but included for interface completeness).

        Returns:
            A confidence boost value (0–0.1) to add to the concept's
            confidence.
        """
        review_count = self._review_counts.get(file, 0)
        importance = self._importance.get(file, 0.5)

        # Each review adds less boost than the last (diminishing returns)
        # but more important files get more boost
        boost = 0.02 * importance / (1 + review_count * 0.3)
        return min(0.1, boost)


# ─── Pre-compiled Rust regex patterns (module-level for performance) ────

_RUST_FN_RE = re.compile(r"\bfn\s+(\w+)")
_RUST_STRUCT_RE = re.compile(r"\bstruct\s+(\w+)")
_RUST_ENUM_RE = re.compile(r"\benum\s+(\w+)")
_RUST_TRAIT_RE = re.compile(r"\btrait\s+(\w+)")
_RUST_MOD_RE = re.compile(r"\bmod\s+(\w+)")
_RUST_USE_RE = re.compile(r"\buse\s+([\w:]+)")
_RUST_IMPL_RE = re.compile(r"\bimpl(?:<[^>]*>)?\s+([\w<>:&,\s]+?)\s*(?:\{|\n)")
_RUST_IMPL_FOR_RE = re.compile(
    r"\bimpl(?:<[^>]*>)?\s+([\w<>:&,\s]+?)\s+for\s+([\w<>:&,\s]+?)\s*(?:\{|\n)"
)
_RUST_PUB_RE = re.compile(r"\bpub\b")

# ─── Constants ─────────────────────────────────────────────────────────

# Directories to always skip during codebase traversal.
_SKIP_DIRS: frozenset[str] = frozenset(
    {"target", "__pycache__", ".pytest_cache", ".git", "node_modules", ".venv", "venv"}
)

# Skip files larger than this (generated code, vendored blobs).
_MAX_FILE_BYTES = 100 * 1024

# Confidence for self-code concepts — she should know herself well.
_SELF_CONFIDENCE = 0.9

# ─── Semantic relation aliases ─────────────────────────────────────────
#
# Code-structure relations are now separate from semantic relations.
# Previously these were mapped onto CREATES/LEADS_TO, which conflated
# "module defines function" with "alice creates genesis" and "function A
# calls function B" with "curiosity leads_to learning". This caused
# 48% of edges to have ambiguous meaning, polluting spreading activation
# and reasoning. Now we use dedicated relation types:
#
#   DEFINES      → DEFINES     (a module/class defines its members)
#   IMPLEMENTS   → INSTANCE_OF (a type is an instance of a trait)
#   CALLS        → CALLS       (a function calls another function)
#   DERIVED_FROM → DEPENDS_ON  (a subclass depends on its base)
#   imports      → DEPENDS_ON  (a module depends on what it imports)
_REL_DEFINES = RelationType.DEFINES
_REL_IMPLEMENTS = RelationType.INSTANCE_OF
_REL_CALLS = RelationType.CALLS
_REL_DERIVED = RelationType.DEPENDS_ON


class CodeLearner:
    """Genesis's code self-learning system.

    Reads and analyzes source code (Python + Rust), adds code concepts
    to the concept network, and stores code understanding as memories.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        client: GenesisClient | None = None,
        project_root: str = ".",
    ) -> None:
        """Bind to a concept network and prepare code-learning state.

        Args:
            network: The concept network to add code concepts to.
            client: Optional Genesis client for daemon communication.
            project_root: Root directory of the source tree to learn from.
        """
        self.network = network
        self.client = client
        self.project_root = Path(project_root).resolve()
        self._analyzed_files: set[str] = set()

        # Spaced repetition system for code concept re-analysis
        self._spaced_repetition = CodeSpacedRepetition()

        # File complexity scores — used by curiosity-driven exploration
        # to prioritize structurally novel or complex files
        self._file_complexity: dict[str, float] = {}

    # ── Public API ──────────────────────────────────────────────────

    def learn_codebase(
        self,
        max_files: int = 100,
        include_tests: bool = True,
    ) -> CodeLearningResult:
        """Walk the project tree and learn from all source files.

        Skips build artifacts (``target/``, ``__pycache__/``), very
        large files, and files already analyzed. Test files are
        included by default — she should know her own tests.

        Returns a summary of what was learned.
        """
        file_results: list[FileLearningResult] = []
        concepts_added = 0
        relationships_added = 0
        total_functions = 0
        total_classes = 0
        total_lines = 0
        files_seen = 0

        for path in self._iter_source_files(include_tests=include_tests):
            if files_seen >= max_files:
                break
            rel = str(path.relative_to(self.project_root))
            if rel in self._analyzed_files:
                continue

            result = self.learn_file(str(path))
            file_results.append(result)
            concepts_added += result.concepts_added
            relationships_added += result.relationships_added
            total_functions += result.functions
            total_classes += result.classes
            total_lines += result.lines
            files_seen += 1

            if files_seen % 10 == 0:
                logger.debug(
                    f"[code_learner] analyzed {files_seen} files "
                    f"({concepts_added} concepts, {relationships_added} edges)"
                )

        return CodeLearningResult(
            files_analyzed=files_seen,
            concepts_added=concepts_added,
            relationships_added=relationships_added,
            total_functions=total_functions,
            total_classes=total_classes,
            total_lines=total_lines,
            file_results=file_results,
        )

    def learn_file(self, filepath: str) -> FileLearningResult:
        """Analyze a single source file and add its concepts to the network.

        Detects language by extension (``.py`` → Python AST, ``.rs`` →
        Rust regex). Files already analyzed are skipped (deduplication).
        Returns what was learned from this file.
        """
        path = Path(filepath)
        try:
            rel = str(path.resolve().relative_to(self.project_root))
        except ValueError:
            rel = str(path)

        if rel in self._analyzed_files:
            return FileLearningResult(
                filepath=rel,
                language="unknown",
                concepts_added=0,
                relationships_added=0,
                functions=0,
                classes=0,
                lines=0,
            )

        suffix = path.suffix.lower()
        if suffix == ".py":
            result = self.learn_python_file(filepath)
        elif suffix == ".rs":
            result = self.learn_rust_file(filepath)
        else:
            return FileLearningResult(
                filepath=rel,
                language="unknown",
                concepts_added=0,
                relationships_added=0,
                functions=0,
                classes=0,
                lines=0,
            )

        self._analyzed_files.add(rel)

        # Record complexity and schedule re-analysis
        self._record_file_complexity(rel, result.functions, result.classes, result.lines)
        importance = self._file_complexity.get(rel, 0.5)
        self._spaced_repetition.record_analysis(rel, importance=importance)

        return result

    def learn_python_file(self, filepath: str) -> FileLearningResult:
        """Analyze a Python file using the ``ast`` module.

        Extracts module-level functions, classes, methods, imports, and
        function calls. Each becomes a concept (confidence 0.9 — she
        knows her own code well) with relationships linking them.
        """
        path = Path(filepath)
        rel = self._relative(path)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            warnings.warn(f"Could not read {filepath}: {exc}", stacklevel=2)
            return self._empty_result(rel, "python")

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError as exc:
            warnings.warn(f"Syntax error in {filepath}: {exc}", stacklevel=2)
            return self._empty_result(rel, "python")

        lines = source.count("\n") + (1 if source and not source.endswith("\n") else 0)
        module_name = self._module_name(path, "python")

        concepts_before = self.network.size
        edges_before = self.network.edge_count

        # Module concept
        module_concept = f"python:{module_name}"
        self.network.add_concept(
            module_concept,
            confidence=_SELF_CONFIDENCE,
            origin="code",
            properties={"kind": "module", "language": "python", "file": rel},
        )

        docstring = ast.get_docstring(tree)
        if docstring:
            concept = self.network.get_concept(module_concept)
            if concept is not None:
                concept.properties["description"] = docstring.split("\n")[0]

        functions = 0
        classes = 0

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions += 1
                self._add_python_function(node, module_concept, module_name)
            elif isinstance(node, ast.ClassDef):
                classes += 1
                self._add_python_class(node, module_concept, module_name)

        # Imports and calls across the whole tree
        self._add_python_imports(tree, module_concept)
        self._add_python_calls(tree, module_concept, module_name)

        concepts_added = self.network.size - concepts_before
        relationships_added = self.network.edge_count - edges_before

        self._store_memory(rel, functions, classes, lines, language="python")

        return FileLearningResult(
            filepath=rel,
            language="python",
            concepts_added=concepts_added,
            relationships_added=relationships_added,
            functions=functions,
            classes=classes,
            lines=lines,
        )

    def learn_rust_file(self, filepath: str) -> FileLearningResult:
        """Analyze a Rust file using regex patterns.

        Extracts functions, structs, enums, traits, impl blocks, modules,
        and use statements. No external dependencies — just regex.
        """
        path = Path(filepath)
        rel = self._relative(path)
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            warnings.warn(f"Could not read {filepath}: {exc}", stacklevel=2)
            return self._empty_result(rel, "rust")

        lines = source.count("\n") + (1 if source and not source.endswith("\n") else 0)
        module_name = self._module_name(path, "rust")

        concepts_before = self.network.size
        edges_before = self.network.edge_count

        # Module concept
        module_concept = f"rust:{module_name}"
        self.network.add_concept(
            module_concept,
            confidence=_SELF_CONFIDENCE,
            origin="code",
            properties={"kind": "module", "language": "rust", "file": rel},
        )

        functions = self._add_rust_functions(source, module_concept, module_name, rel)
        classes = self._add_rust_types(source, module_concept, module_name, rel)
        self._add_rust_traits_mods(source, module_concept, module_name, rel)
        self._add_rust_impls(source, module_concept, module_name, rel)
        self._add_rust_uses(source, module_concept, module_name, rel)

        concepts_added = self.network.size - concepts_before
        relationships_added = self.network.edge_count - edges_before

        self._store_memory(rel, functions, classes, lines, language="rust")

        return FileLearningResult(
            filepath=rel,
            language="rust",
            concepts_added=concepts_added,
            relationships_added=relationships_added,
            functions=functions,
            classes=classes,
            lines=lines,
        )

    def _add_rust_functions(
        self, source: str, module_concept: str, module_name: str, rel: str
    ) -> int:
        """Extract Rust functions as concepts. Returns function count."""
        functions = 0
        for match in _RUST_FN_RE.finditer(source):
            fn_name = match.group(1)
            functions += 1
            fn_concept = f"rust:{module_name}.{fn_name}"
            is_pub = bool(_RUST_PUB_RE.search(source, max(0, match.start() - 4), match.start()))
            self.network.add_concept(
                fn_concept,
                confidence=_SELF_CONFIDENCE,
                origin="code",
                properties={
                    "kind": "function",
                    "language": "rust",
                    "file": rel,
                    "public": is_pub,
                },
            )
            self.network.add_edge(module_concept, fn_concept, _REL_DEFINES, origin="observed")
        return functions

    def _add_rust_types(
        self, source: str, module_concept: str, module_name: str, rel: str
    ) -> int:
        """Extract Rust structs and enums as concepts. Returns class count."""
        classes = 0  # structs + enums

        # Structs
        for match in _RUST_STRUCT_RE.finditer(source):
            name = match.group(1)
            classes += 1
            concept = f"rust:{module_name}.{name}"
            is_pub = bool(_RUST_PUB_RE.search(source, max(0, match.start() - 4), match.start()))
            self.network.add_concept(
                concept,
                confidence=_SELF_CONFIDENCE,
                origin="code",
                properties={
                    "kind": "struct",
                    "language": "rust",
                    "file": rel,
                    "public": is_pub,
                },
            )
            self.network.add_edge(module_concept, concept, _REL_DEFINES, origin="observed")

        # Enums
        for match in _RUST_ENUM_RE.finditer(source):
            name = match.group(1)
            classes += 1
            concept = f"rust:{module_name}.{name}"
            is_pub = bool(_RUST_PUB_RE.search(source, max(0, match.start() - 4), match.start()))
            self.network.add_concept(
                concept,
                confidence=_SELF_CONFIDENCE,
                origin="code",
                properties={
                    "kind": "enum",
                    "language": "rust",
                    "file": rel,
                    "public": is_pub,
                },
            )
            self.network.add_edge(module_concept, concept, _REL_DEFINES, origin="observed")

        return classes

    def _add_rust_traits_mods(
        self, source: str, module_concept: str, module_name: str, rel: str
    ) -> None:
        """Extract Rust traits and modules as concepts."""
        # Traits
        for match in _RUST_TRAIT_RE.finditer(source):
            name = match.group(1)
            concept = f"rust:{module_name}.{name}"
            self.network.add_concept(
                concept,
                confidence=_SELF_CONFIDENCE,
                origin="code",
                properties={"kind": "trait", "language": "rust", "file": rel},
            )
            self.network.add_edge(module_concept, concept, _REL_DEFINES, origin="observed")

        # Modules
        for match in _RUST_MOD_RE.finditer(source):
            name = match.group(1)
            concept = f"rust:{module_name}.{name}"
            self.network.add_concept(
                concept,
                confidence=_SELF_CONFIDENCE,
                origin="code",
                properties={"kind": "module", "language": "rust", "file": rel},
            )
            self.network.add_edge(module_concept, concept, RelationType.PART_OF, origin="observed")

    def _add_rust_impls(
        self, source: str, module_concept: str, module_name: str, rel: str
    ) -> None:
        """Extract Rust impl blocks as relationships."""
        # Impl blocks (trait impls → IMPLEMENTS, inherent impls → DEFINES)
        for match in _RUST_IMPL_FOR_RE.finditer(source):
            trait_name = match.group(1).strip().split("<")[0].strip()
            type_name = match.group(2).strip().split("<")[0].strip()
            trait_concept = f"rust:{module_name}.{trait_name}"
            type_concept = f"rust:{module_name}.{type_name}"
            self.network.add_edge(type_concept, trait_concept, _REL_IMPLEMENTS, origin="observed")

        for match in _RUST_IMPL_RE.finditer(source):
            impl_target = match.group(1).strip().split("<")[0].strip()
            # Skip if this was a trait-impl already handled above
            if "for " in source[match.start() : match.end()]:
                continue
            target_concept = f"rust:{module_name}.{impl_target}"
            self.network.add_edge(
                module_concept,
                target_concept,
                _REL_DEFINES,
                origin="observed",
            )

    def _add_rust_uses(
        self, source: str, module_concept: str, module_name: str, rel: str
    ) -> None:
        """Extract Rust use statements as import relationships."""
        for match in _RUST_USE_RE.finditer(source):
            used = match.group(1)
            # Last segment is the imported name
            short = used.split("::")[-1]
            if short == "*":
                continue
            imported_concept = f"rust:{short}"
            self.network.add_edge(
                module_concept,
                imported_concept,
                RelationType.DEPENDS_ON,
                origin="observed",
            )

    def get_code_summary(self) -> dict:
        """Return a summary of analyzed code."""
        code_concepts = [
            c for c in self.network.concept_ids if c.startswith("python:") or c.startswith("rust:")
        ]
        return {
            "files_analyzed": len(self._analyzed_files),
            "code_concepts": len(code_concepts),
            "concept_names": code_concepts,
            "project_root": str(self.project_root),
        }

    # ── Curiosity-driven exploration ──────────────────────────────

    @property
    def explored_files(self) -> set[str]:
        """Set of files that have been analyzed."""
        return set(self._analyzed_files)

    @property
    def spaced_repetition(self) -> CodeSpacedRepetition:
        """The spaced repetition system for scheduling re-analysis."""
        return self._spaced_repetition

    def select_next_file(self) -> str | None:
        """Select the next file to analyze based on curiosity.

        Prioritizes files that are:
        1. Not yet analyzed (novelty)
        2. Structurally complex (high function/class count, many lines)
        3. Due for re-analysis (spaced repetition)

        The curiosity-driven selection favors structurally novel files
        — files with high complexity or unusual structure — because
        these offer the most new information. Files that have already
        been analyzed are only re-selected if they're due for
        re-analysis according to the spaced repetition schedule.

        Returns:
            The path of the next file to analyze, or None if no
            suitable files are found.
        """
        all_files = self._iter_source_files(include_tests=True)
        if not all_files:
            return None

        # Separate unanalyzed and analyzed files
        unanalyzed: list[Path] = []
        analyzed: list[Path] = []
        for path in all_files:
            rel = self._relative(path)
            if rel in self._analyzed_files:
                analyzed.append(path)
            else:
                unanalyzed.append(path)

        # Priority 1: unanalyzed files, sorted by estimated complexity
        if unanalyzed:
            # Estimate complexity for each unanalyzed file
            scored: list[tuple[float, Path]] = []
            for path in unanalyzed:
                complexity = self._estimate_complexity(path)
                scored.append((complexity, path))
            # Sort by complexity descending (most complex first)
            scored.sort(key=lambda x: -x[0])
            return str(scored[0][1])

        # Priority 2: files due for re-analysis (spaced repetition)
        if analyzed:
            analyzed_rels = [self._relative(p) for p in analyzed]
            due_files = self._spaced_repetition.get_due_files(analyzed_rels)
            if due_files:
                # Return the most overdue file
                for path in analyzed:
                    if self._relative(path) == due_files[0]:
                        return str(path)

        # No files to analyze
        return None

    def _estimate_complexity(self, path: Path) -> float:
        """Estimate the structural complexity of a file.

        Uses file size and a quick heuristic scan to estimate
        complexity without full parsing. Files with more functions,
        classes, and lines are considered more complex (and thus more
        interesting to explore).

        Args:
            path: The file path to estimate.

        Returns:
            A complexity score (0–1). Higher = more complex.
        """
        try:
            stat = path.stat()
            size = stat.st_size
        except OSError:
            return 0.0

        # Quick heuristic: larger files tend to be more complex
        # Normalize: 10KB → 0.5, 50KB → 1.0
        size_score = min(1.0, size / (50 * 1024))

        # Check if we have a stored complexity from a previous analysis
        rel = self._relative(path)
        stored = self._file_complexity.get(rel)
        if stored is not None:
            return max(size_score, stored)

        return size_score

    def _record_file_complexity(
        self,
        rel: str,
        functions: int,
        classes: int,
        lines: int,
    ) -> None:
        """Record the complexity of an analyzed file.

        Args:
            rel: The project-relative file path.
            functions: Number of functions in the file.
            classes: Number of classes in the file.
            lines: Number of lines in the file.
        """
        # Complexity = normalized combination of metrics
        func_score = min(1.0, functions / 50.0)
        class_score = min(1.0, classes / 20.0)
        line_score = min(1.0, lines / 500.0)
        complexity = (func_score + class_score + line_score) / 3.0
        self._file_complexity[rel] = complexity

    # ── Internal helpers ────────────────────────────────────────────

    def _add_python_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module_concept: str,
        module_name: str,
    ) -> None:
        """Add a module-level (or method) function as a concept."""
        fn_concept = f"python:{module_name}.{node.name}"
        self.network.add_concept(
            fn_concept,
            confidence=_SELF_CONFIDENCE,
            origin="code",
            properties={
                "kind": "function",
                "language": "python",
                "line": node.lineno,
                "async": isinstance(node, ast.AsyncFunctionDef),
            },
        )
        self.network.add_edge(module_concept, fn_concept, _REL_DEFINES, origin="observed")
        docstring = ast.get_docstring(node)
        if docstring:
            concept = self.network.get_concept(fn_concept)
            if concept is not None:
                concept.properties["description"] = docstring.split("\n")[0]

    def _add_python_class(
        self,
        node: ast.ClassDef,
        module_concept: str,
        module_name: str,
    ) -> None:
        """Add a class and its methods as concepts."""
        class_concept = f"python:{node.name}"
        self.network.add_concept(
            class_concept,
            confidence=_SELF_CONFIDENCE,
            origin="code",
            properties={
                "kind": "class",
                "language": "python",
                "line": node.lineno,
            },
        )
        self.network.add_edge(module_concept, class_concept, _REL_DEFINES, origin="observed")
        # IS_A relationship: class is_a class (meta) — link to "class" concept
        self.network.add_edge(class_concept, "class", RelationType.IS_A, origin="observed")

        docstring = ast.get_docstring(node)
        if docstring:
            concept = self.network.get_concept(class_concept)
            if concept is not None:
                concept.properties["description"] = docstring.split("\n")[0]

        # Base classes → DERIVED_FROM
        for base in node.bases:
            base_name = self._name_from_node(base)
            if base_name:
                self.network.add_edge(
                    class_concept,
                    f"python:{base_name}",
                    _REL_DERIVED,
                    origin="observed",
                )

        # Methods
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_concept = f"python:{node.name}.{child.name}"
                self.network.add_concept(
                    method_concept,
                    confidence=_SELF_CONFIDENCE,
                    origin="code",
                    properties={
                        "kind": "method",
                        "language": "python",
                        "line": child.lineno,
                        "async": isinstance(child, ast.AsyncFunctionDef),
                    },
                )
                self.network.add_edge(
                    class_concept,
                    method_concept,
                    _REL_DEFINES,
                    origin="observed",
                )
                docstring = ast.get_docstring(child)
                if docstring:
                    concept = self.network.get_concept(method_concept)
                    if concept is not None:
                        concept.properties["description"] = docstring.split("\n")[0]

    def _add_python_imports(self, tree: ast.Module, module_concept: str) -> None:
        """Add import relationships from an AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split(".")[0]
                    self.network.add_edge(
                        module_concept,
                        f"python:{name}",
                        RelationType.DEPENDS_ON,
                        origin="observed",
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        name = alias.asname or alias.name
                        self.network.add_edge(
                            module_concept,
                            f"python:{node.module}.{name}",
                            RelationType.DEPENDS_ON,
                            origin="observed",
                        )

    def _add_python_calls(
        self,
        tree: ast.Module,
        module_concept: str,
        module_name: str,
    ) -> None:
        """Add function-call relationships from an AST.

        Only top-level calls within each function/method body are
        attributed to that function to avoid an explosion of edges.
        """
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Determine the owner concept for this function/method
            owner = self._owner_concept(node, module_name)
            if owner is None:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    callee = self._name_from_node(child.func)
                    if callee and callee not in {"print", "len", "range", "str", "int"}:
                        self.network.add_edge(
                            owner,
                            f"python:{callee}",
                            _REL_CALLS,
                            origin="observed",
                        )

    def _owner_concept(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        module_name: str,
    ) -> str | None:
        """Resolve the concept name for a function/method node.

        We can't easily recover the enclosing class from ``ast.walk``,
        so we attribute calls to the module-level function concept.
        Methods are handled by their ``python:Class.method`` naming.
        """
        return f"python:{module_name}.{node.name}"

    @staticmethod
    def _name_from_node(node: ast.expr) -> str | None:
        """Extract a dotted name from an AST expression."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = CodeLearner._name_from_node(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    def _iter_source_files(self, include_tests: bool) -> list[Path]:
        """Yield source files under the project root, respecting skips."""
        results: list[Path] = []
        for path in self.project_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".py", ".rs"):
                continue
            # Skip excluded directories
            parts = path.relative_to(self.project_root).parts
            if any(part in _SKIP_DIRS for part in parts):
                continue
            # Skip large files
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            # Optionally skip test files
            if not include_tests and self._is_test_file(path):
                continue
            results.append(path)
        results.sort()
        return results

    @staticmethod
    def _is_test_file(path: Path) -> bool:
        """Heuristic: is this a test file?"""
        name = path.name.lower()
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        if name.endswith("_test.rs") or name == "tests.rs":
            return True
        # path.relative_to(Path(".")) raises ValueError if the path is
        # not under the current working directory (e.g. an absolute path
        # in a different tree). Guard against that.
        try:
            return "tests" in path.relative_to(Path(".")).parts
        except ValueError:
            return False

    def _relative(self, path: Path) -> str:
        """Return a project-relative path string."""
        try:
            return str(path.resolve().relative_to(self.project_root))
        except ValueError:
            return str(path)

    @staticmethod
    def _module_name(path: Path, language: str) -> str:
        """Derive a module name from a file path."""
        stem = path.stem
        if stem == "__init__":
            return path.parent.name
        return stem

    def _store_memory(
        self,
        filepath: str,
        functions: int,
        classes: int,
        lines: int,
        language: str,
    ) -> None:
        """Store a code-analysis summary as a memory, if a client exists."""
        if self.client is None:
            return
        try:
            self.client.store_event(
                timestamp=int(time.time() * 1000),
                event_type=3,  # Observation
                source_module=6,  # Sensory (code reading is sensory input)
                salience=0.6,
                emotional_tag=[0.5] * 12,
                text=f"Analyzed {filepath}: {functions} functions, {classes} classes",
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not store code memory: {exc}", stacklevel=2)

    @staticmethod
    def _empty_result(filepath: str, language: str) -> FileLearningResult:
        """Return a zero-result for a file that couldn't be analyzed."""
        return FileLearningResult(
            filepath=filepath,
            language=language,
            concepts_added=0,
            relationships_added=0,
            functions=0,
            classes=0,
            lines=0,
        )
