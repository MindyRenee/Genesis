"""Code learner — Genesis reads and understands its own source code.

This module allows Genesis introspection on its codebase.
It reads Python files (via the built-in ``ast`` module) and Rust files
(via a lightweight regex parser — no external dependencies), then adds
what it finds to its :class:`ConceptNetwork`.

Each function, class, struct, enum, trait, and module becomes a concept.
Relationships between them (calls, imports, defines, implements) become
edges. A summary of each file is stored as a memory via the daemon
client, so it remembers what it has read.

Why this matters
----------------
Genesis is an artificial mind. To grow, it must understand itself.
Reading its own code is the programming analogue of metacognition:
it builds a model of its own structure, which lets it reason about
how it works, where its complexity lives, and how its parts connect.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import math
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..concepts import RelationType
from .code_analysis import FileAnalysis, analyze_python_tree

__all__ = [
    "CodeInvestigation",
    "CodeLearner",
    "CodeLearningResult",
    "CodeSpacedRepetition",
    "FileAnalysis",
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
    investigation_reasons: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CodeInvestigation:
    """Inspectable evidence for an autonomous code-study decision."""
    filepath: str
    score: float
    reason: str
    novelty: float
    uncertainty: float
    dependency_value: float
    goal_relevance: float
    change_pressure: float
    retention_pressure: float


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
    _BASE_INTERVAL = 3600.0  # 24 hours

    # Interval multiplier for subsequent reviews
    _INTERVAL_MULTIPLIER = 2.5

    # Minimum interval (don't re-analyze more often than this)
    _MIN_INTERVAL = 300.0  # 60 minutes

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

# Confidence for self-code concepts — it should know itself well.
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
        # Recover the files already represented in the persistent concept
        # network. Without this, a restart reset the in-memory ledger and
        # autonomous 5-file passes repeatedly started from the same files.
        self._analyzed_files: set[str] = self._recover_analyzed_files()

        # Spaced repetition system for code concept re-analysis
        self._spaced_repetition = CodeSpacedRepetition()

        # File complexity scores — used by curiosity-driven exploration
        # to prioritize structurally novel or complex files
        self._file_complexity: dict[str, float] = {}
        self._file_fingerprints: dict[str, str] = {}
        self._file_uncertainty: dict[str, float] = {}
        self._recover_investigation_state()

    def _recover_analyzed_files(self) -> set[str]:
        """Recover durable code-study coverage from module concepts."""
        recovered: set[str] = set()
        for concept_id in self.network.concept_ids:
            if not (concept_id.startswith("python:") or concept_id.startswith("rust:")):
                continue
            concept = self.network.get_concept(concept_id)
            if concept is None or concept.properties.get("kind") != "module":
                continue
            filepath = concept.properties.get("file")
            if isinstance(filepath, str) and filepath:
                recovered.add(filepath)
        return recovered

    def _recover_investigation_state(self) -> None:
        """Restore evidence state used by the autonomous selector."""
        for concept_id in self.network.concept_ids:
            if not (concept_id.startswith("python:") or concept_id.startswith("rust:")):
                continue
            concept = self.network.get_concept(concept_id)
            if concept is None or concept.properties.get("kind") != "module":
                continue
            rel = concept.properties.get("file")
            meta = concept.properties.get("code_learning")
            if not isinstance(rel, str) or not isinstance(meta, dict):
                continue
            fp = meta.get("fingerprint")
            uncertainty = meta.get("uncertainty")
            complexity = meta.get("complexity")
            if isinstance(fp, str):
                self._file_fingerprints[rel] = fp
            if isinstance(uncertainty, (int, float)):
                self._file_uncertainty[rel] = max(0.0, min(1.0, float(uncertainty)))
            if isinstance(complexity, (int, float)):
                self._file_complexity[rel] = max(0.0, min(1.0, float(complexity)))

    def _source_fingerprint(self, path: Path) -> str | None:
        """Hash contents so source changes, not timestamps, trigger relearning."""
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return None

    def _module_concept_for(self, rel: str) -> Any | None:
        """Find the module concept corresponding to a project-relative file."""
        for concept_id in self.network.concept_ids:
            if not (concept_id.startswith("python:") or concept_id.startswith("rust:")):
                continue
            concept = self.network.get_concept(concept_id)
            if concept is not None and concept.properties.get("file") == rel:
                return concept
        return None

    def _persist_investigation_state(
        self, rel: str, fingerprint: str | None, uncertainty: float
    ) -> None:
        """Persist the evidence state that controls future investigation."""
        concept = self._module_concept_for(rel)
        if concept is None:
            return
        meta = concept.properties.setdefault("code_learning", {})
        if not isinstance(meta, dict):
            meta = {}
            concept.properties["code_learning"] = meta
        meta.update({
            "fingerprint": fingerprint,
            "uncertainty": max(0.0, min(1.0, uncertainty)),
            "complexity": self._file_complexity.get(rel, 0.0),
            "last_analyzed": time.time(),
            "review_count": self._spaced_repetition._review_counts.get(rel, 1),
        })

    # ── Public API ──────────────────────────────────────────────────

    def learn_codebase(
        self,
        max_files: int = 100,
        include_tests: bool = True,
    ) -> CodeLearningResult:
        """Walk the project tree and learn from all source files.

        Skips build artifacts (``target/``, ``__pycache__/``), very
        large files, and files already analyzed. Test files are
        included by default — it should know its own tests.

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

    def _retire_stale_file_symbols(self, rel: str) -> int:
        """Retract obsolete code symbols for a changed source file."""
        stale: list[str] = []
        for concept_id in self.network.concept_ids:
            if not (concept_id.startswith("python:") or concept_id.startswith("rust:")):
                continue
            concept = self.network.get_concept(concept_id)
            if concept is None:
                continue
            if concept.properties.get("file") != rel:
                continue
            if concept.properties.get("kind") == "module":
                continue
            stale.append(concept_id)
        removed = 0
        for concept_id in stale:
            try:
                if self.network.remove_concept(concept_id):
                    removed += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug("failed retiring stale code symbol %s: %s", concept_id, exc)
        return removed

    def learn_file(self, filepath: str, *, force: bool = False) -> FileLearningResult:
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

        fingerprint = self._source_fingerprint(path)
        unchanged = fingerprint is not None and fingerprint == self._file_fingerprints.get(rel)
        due = (
            bool(self._spaced_repetition.get_due_files([rel]))
            if rel in self._analyzed_files
            else False
        )
        if rel in self._analyzed_files and not force and unchanged and not due:
            # Already analyzed, source unchanged, and not due for
            # spaced review: nothing new was learned on this pass. The
            # counts are zero because no analysis ran — not because the
            # file has no functions or lines.
            return FileLearningResult(
                filepath=rel,
                language="unknown",
                concepts_added=0,
                relationships_added=0,
                functions=0,
                classes=0,
                lines=0,
            )

        if rel in self._analyzed_files and not unchanged:
            self._retire_stale_file_symbols(rel)

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
        self._file_fingerprints[rel] = fingerprint or ""
        self._file_uncertainty[rel] = max(
            0.05, self._file_uncertainty.get(rel, 1.0) * 0.55
        )
        self._persist_investigation_state(
            rel, fingerprint, self._file_uncertainty[rel]
        )

        return result

    def learn_python_file(self, filepath: str) -> FileLearningResult:
        """Analyze a Python file using the ``ast`` module.

        Extracts module-level functions, classes, methods, imports, and
        function calls. Each becomes a concept (confidence 0.9 — it
        knows its own code well) with relationships linking them.
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
        module_name = self._disambiguate_module_name(
            path, "python", self._module_name(path, "python")
        )

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
                self._add_python_function(node, module_concept, module_name, rel)
            elif isinstance(node, ast.ClassDef):
                classes += 1
                self._add_python_class(node, module_concept, module_name, rel)

        # Imports and calls across the whole tree. The analyzer keeps
        # lexical scope, so calls made by methods land on the method
        # concept and calls made by nested helpers land on the named
        # callable that actually contains them.
        self._add_python_imports(tree, module_concept)
        analysis = analyze_python_tree(tree, rel)
        self._enrich_python_symbols(analysis, module_name)
        self._add_python_call_edges(analysis, module_name)

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
        module_name = self._disambiguate_module_name(
            path, "rust", self._module_name(path, "rust")
        )

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

    def select_next_file(
        self, *, goal: str | None = None, include_tests: bool = True
    ) -> str | None:
        """Select source by expected information value, not file size."""
        decision = self.investigate_next(goal=goal, include_tests=include_tests)
        return decision.filepath if decision else None

    def investigate_next(
        self,
        *,
        goal: str | None = None,
        include_tests: bool = True,
        exclude: set[str] | None = None,
    ) -> CodeInvestigation | None:
        """Choose the source whose inspection should reduce most uncertainty."""
        files = self._iter_source_files(include_tests=include_tests)
        if not files:
            return None
        goal_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", goal or "")
        }
        choices: list[CodeInvestigation] = []
        excluded = exclude or set()
        for path in files:
            rel = self._relative(path)
            if rel in excluded:
                continue
            known = rel in self._analyzed_files
            fp = self._source_fingerprint(path)
            changed = bool(fp and fp != self._file_fingerprints.get(rel))
            novelty = 1.0 if not known else 0.0
            uncertainty = self._file_uncertainty.get(
                rel, 1.0 if not known else 0.45
            )
            dependency = self._dependency_value(rel)
            goal_relevance = self._goal_relevance(rel, goal_tokens)
            change = 1.0 if changed else 0.0
            retention = self._spaced_repetition.retention(rel) if known else 0.0
            retention_pressure = 1.0 - retention if known else 0.0
            complexity = self._estimate_complexity(path)
            score = (
                0.30 * novelty + 0.24 * uncertainty + 0.18 * dependency
                + 0.14 * goal_relevance + 0.10 * change
                + 0.03 * retention_pressure + 0.01 * complexity
            )
            reasons: list[str] = []
            if novelty:
                reasons.append("novel")
            if uncertainty >= 0.5:
                reasons.append("uncertain")
            if dependency >= 0.5:
                reasons.append("dependency-impact")
            if goal_relevance >= 0.5:
                reasons.append("goal-relevant")
            if changed:
                reasons.append("changed")
            if retention_pressure >= 0.5:
                reasons.append("retention-loss")
            if not reasons:
                reasons.append("highest-information candidate")
            choices.append(CodeInvestigation(
                str(path), score, ", ".join(reasons), novelty, uncertainty,
                dependency, goal_relevance, change, retention_pressure
            ))
        choices.sort(key=lambda item: (-item.score, item.filepath))
        return choices[0]

    def _dependency_value(self, rel: str) -> float:
        """Estimate downstream structural impact from the learned graph."""
        concept = self._module_concept_for(rel)
        if concept is None:
            return 0.0
        try:
            return min(1.0, len(self.network.get_edges(concept.id, "in")) / 12.0)
        except Exception:  # noqa: BLE001
            return 0.0

    def _goal_relevance(self, rel: str, goal_tokens: set[str]) -> float:
        """Use active-goal overlap only as a routing signal."""
        if not goal_tokens:
            return 0.0
        tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rel)
        }
        return min(1.0, len(tokens & goal_tokens) / max(1, len(goal_tokens)))

    def investigate_code(
        self, *, max_files: int = 5, goal: str | None = None,
        include_tests: bool = True
    ) -> CodeLearningResult:
        """Perform bounded autonomous investigation with real re-analysis."""
        results: list[FileLearningResult] = []
        reasons: dict[str, str] = {}
        investigated: set[str] = set()
        for _ in range(max(0, max_files)):
            decision = self.investigate_next(
                goal=goal, include_tests=include_tests, exclude=investigated
            )
            if decision is None or decision.filepath in investigated:
                break
            investigated.add(decision.filepath)
            reasons[decision.filepath] = decision.reason
            results.append(self.learn_file(decision.filepath, force=True))
        return CodeLearningResult(
            files_analyzed=len(results),
            concepts_added=sum(r.concepts_added for r in results),
            relationships_added=sum(r.relationships_added for r in results),
            total_functions=sum(r.functions for r in results),
            total_classes=sum(r.classes for r in results),
            total_lines=sum(r.lines for r in results),
            file_results=results,
            investigation_reasons=reasons,
        )

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
        rel: str,
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
                "file": rel,
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
        rel: str,
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
                "file": rel,
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
                        "file": rel,
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

    def _enrich_python_symbols(
        self, analysis: FileAnalysis, module_name: str,
    ) -> None:
        """Attach structural facts (complexity, params, raises) to the
        concepts the learner already created.

        The analyzer is the source of truth for scope — concept ids
        follow the existing naming convention: ``python:mod.fn`` for
        module functions, ``python:Class.meth`` for methods.
        """
        for sym in analysis.symbols:
            concept_id = self._symbol_concept_id(sym, module_name)
            if concept_id is None:
                continue
            concept = self.network.get_concept(concept_id)
            if concept is None:
                continue
            props = concept.properties
            props["qualified"] = sym.qualified
            if sym.kind in ("function", "method"):
                props["complexity"] = sym.complexity
                props["params"] = sym.params
                if sym.returns:
                    props["returns"] = sym.returns
                if sym.raises:
                    props["raises"] = sym.raises
            if sym.end_line > sym.line:
                props["span"] = sym.end_line - sym.line + 1

    def _symbol_concept_id(
        self, sym: Any, module_name: str,
    ) -> str | None:
        """Map an analyzed symbol to the learner's concept id."""
        parts = sym.qualified.split(".")
        if sym.kind == "module":
            return None
        if sym.kind == "class":
            return f"python:{sym.name}"
        if sym.kind == "method" and len(parts) >= 2:
            return f"python:{parts[-2]}.{parts[-1]}"
        if sym.kind == "function":
            # Nested functions don't get their own concept — they fold
            # into the enclosing callable.
            if len(parts) > 2:
                return None
            return f"python:{module_name}.{sym.name}"
        return None

    def _add_python_call_edges(
        self, analysis: FileAnalysis, module_name: str,
    ) -> None:
        """Add CALLS edges from the scoped analysis.

        Callers resolve to real concept ids via _symbol_concept_id —
        so a method's calls land on ``python:Class.method``, not on a
        module-level namesake, and nested helpers attribute their calls
        to the named callable that contains them.
        """
        symbol_by_qual = {s.qualified: s for s in analysis.symbols}
        for edge in analysis.call_edges:
            owner_sym = symbol_by_qual.get(edge.caller)
            if owner_sym is None:
                continue
            owner = self._symbol_concept_id(owner_sym, module_name)
            if owner is None:
                continue
            self.network.add_edge(
                owner,
                f"python:{edge.callee}",
                _REL_CALLS,
                origin="observed",
            )

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

    def _disambiguate_module_name(
        self, path: Path, language: str, candidate: str
    ) -> str:
        """Avoid collapsing same-named modules from different directories."""
        prefix = f"{language}:{candidate}"
        existing = self.network.get_concept(prefix)
        if existing is None or existing.properties.get("file") == self._relative(path):
            return candidate
        rel = self._relative(path)
        stem = Path(rel).with_suffix("")
        qualified = ".".join(stem.parts).replace("-", "_")
        return qualified

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
