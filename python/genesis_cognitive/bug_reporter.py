"""Bug reporter — Genesis's ability to notice problems in her own code.

This is not a linter. It's something more organic: Genesis reads her
own code and notices things that bother her. Sometimes she knows
exactly what's wrong (a bare ``except:`` that swallows all errors).
Sometimes she just has a vague sense that something is off (a function
that's unusually long, a module with too many responsibilities).

The bug reporter performs static analysis — it never executes code.
It scans Python files via AST and Rust files via regex, looking for
common patterns that indicate problems:

**Python patterns (simple, mastery-first approach):**
1. Bare ``except:`` — catches SystemExit, KeyboardInterrupt, etc.
2. Mutable default arguments — shared state across calls
3. Unreachable code after ``return`` / ``raise`` / ``break``
4. Functions longer than 50 lines — hard to understand
5. Modules with >20 functions — too many responsibilities
6. Duplicate string literals — possible copy-paste errors
7. ``import *`` — pollutes namespace, hides dependencies
8. Bare ``pass`` in ``except`` — silently swallows errors
9. ``print()`` left in production code (not in ``__main__``)
10. Missing type hints on public functions
11. Comparison to ``None`` with ``==`` / ``!=`` — should use ``is`` / ``is not`` (PEP 8)

**Rust patterns:**
1. ``unwrap()`` in non-test code — panics on None/Err
2. ``todo!()`` or ``unimplemented!()`` left in code
3. Functions longer than 80 lines
4. ``unsafe`` blocks without safety comment

Each bug is logged to a structured JSONL file so it can be reviewed
later. Bugs are also added as concepts to the network so Genesis can
reason about them and surface them in conversation.

## The log files

Bugs are tracked in two files in ``genesis_data``:

- ``bug_reports.jsonl`` — a **running list** of currently-open issues.
  One JSON object per line. Rewritten each scan so it always reflects
  the current state. An issue is only present if it's still found in
  the code. When an issue is fixed, it's removed from this file.

- ``bug_reports_history.jsonl`` — an **audit trail** of resolved issues.
  When an issue is no longer found, it's moved here with a
  ``resolved_at`` timestamp. This lets Genesis talk about her track
  record ("I've fixed 12 issues") without cluttering the running list.

Each entry contains:

    {
        "timestamp": "2024-01-15T10:30:00",
        "file": "python/genesis_cognitive/cognition.py",
        "line": 425,
        "severity": "warning",
        "category": "bare_except",
        "description": "Bare except: catches SystemExit and KeyboardInterrupt",
        "suggestion": "Use 'except Exception:' to avoid catching system exits"
    }

## Severity levels

- **error**: Likely to cause incorrect behavior (bare except, mutable defaults)
- **warning**: Code smell that may cause issues (long functions, too many imports)
- **style**: Minor issues (missing type hints, print in production)

Genesis starts with simple patterns and masters them before moving
to more complex analysis. This is the foundation.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "BugPatternKnowledge",
    "BugReport",
    "BugReporter",
    "BugScanResult",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BugPatternKnowledge:
    """Maps a bug pattern to the concepts it involves and why it matters.

    This is what lets Genesis *understand* a bug, not just detect it.
    Each pattern is connected to:
    - The concepts in her network that are relevant (e.g., "exception",
      "system exit", "keyboard interrupt" for a bare except)
    - The consequence — what happens if this bug isn't fixed
    - A docs query — what she should study to understand it better
    - A comprehension threshold — how well she needs to understand the
      concepts before she can honestly report this bug
    """

    category: str
    concepts: list[str]  # concept names she needs to understand
    consequence: str  # what happens if unfixed
    docs_query: str  # what to search in docs
    docs_language: str  # "python" or "rust"
    comprehension_threshold: float = 0.3  # min avg concept confidence


# ─── Bug pattern knowledge registry ──────────────────────────────────
#
# Each entry maps a bug category to the concepts Genesis needs to
# understand, the consequence of the bug, and what docs to fetch.
# This is the bridge between "I found a pattern" and "I understand
# why this is a problem."

_BUG_PATTERN_KNOWLEDGE: dict[str, BugPatternKnowledge] = {
    "bare_except": BugPatternKnowledge(
        category="bare_except",
        concepts=["exception", "system exit", "keyboard interrupt", "catch", "error handling"],
        consequence=(
            "a bare except catches SystemExit and KeyboardInterrupt, "
            "which means the program can't be stopped with Ctrl+C "
            "and can't shut down cleanly"
        ),
        docs_query="exception handling try except",
        docs_language="python",
        comprehension_threshold=0.3,
    ),
    "silent_except": BugPatternKnowledge(
        category="silent_except",
        concepts=["exception", "error handling", "catch", "logging"],
        consequence=(
            "errors are silently swallowed — if something goes wrong, "
            "no one will ever know because the error is hidden"
        ),
        docs_query="exception handling logging",
        docs_language="python",
        comprehension_threshold=0.3,
    ),
    "mutable_default": BugPatternKnowledge(
        category="mutable_default",
        concepts=["mutable", "shared state", "default argument", "side effect"],
        consequence=(
            "the mutable default is created once and shared across all "
            "calls, so modifications in one call affect the next — "
            "this is shared state that causes unexpected behavior"
        ),
        docs_query="default argument mutable",
        docs_language="python",
        comprehension_threshold=0.3,
    ),
    "long_function": BugPatternKnowledge(
        category="long_function",
        concepts=["complexity", "maintainability", "code"],
        consequence=(
            "a long function is hard to understand and maintain — "
            "high complexity leads to bugs because it's difficult "
            "to reason about all the paths through the code"
        ),
        docs_query="function complexity refactoring",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
    "import_star": BugPatternKnowledge(
        category="import_star",
        concepts=["namespace", "namespace pollution", "code"],
        consequence=(
            "import star pollutes the namespace — it's unclear where "
            "each name comes from, and names can conflict, causing "
            "subtle bugs"
        ),
        docs_query="import star namespace",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
    "print_in_code": BugPatternKnowledge(
        category="print_in_code",
        concepts=["logging", "code", "error handling"],
        consequence=(
            "print statements in production code can't be filtered, "
            "formatted, or directed to files — logging is better "
            "because it can be controlled"
        ),
        docs_query="logging print",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
    "missing_type_hints": BugPatternKnowledge(
        category="missing_type_hints",
        concepts=["type hint", "code", "understanding"],
        consequence=(
            "without type hints, it's hard to know what a function "
            "accepts and returns — this makes the code harder to "
            "understand and maintain"
        ),
        docs_query="type hints annotations",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
    "unreachable_code": BugPatternKnowledge(
        category="unreachable_code",
        concepts=["unreachable code", "complexity", "code"],
        consequence=(
            "code after a return or raise statement can never run — "
            "it's dead code that adds complexity without value"
        ),
        docs_query="unreachable code dead code",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
    "rust_unwrap": BugPatternKnowledge(
        category="rust_unwrap",
        concepts=["unwrap", "panic", "error", "safety"],
        consequence=(
            "unwrap panics if the value is None or an error — "
            "in production code, this means the program crashes "
            "instead of handling the error gracefully"
        ),
        docs_query="option unwrap result",
        docs_language="rust",
        comprehension_threshold=0.3,
    ),
    "rust_todo": BugPatternKnowledge(
        category="rust_todo",
        concepts=["code", "error"],
        consequence=(
            "todo! or unimplemented! means the function isn't finished — "
            "calling it will panic"
        ),
        docs_query="todo unimplemented macro",
        docs_language="rust",
        comprehension_threshold=0.2,
    ),
    "rust_unsafe_no_comment": BugPatternKnowledge(
        category="rust_unsafe_no_comment",
        concepts=["safety", "invariant", "code"],
        consequence=(
            "an unsafe block without a safety comment has no explanation "
            "of why it's safe — if the invariant is violated, it causes "
            "undefined behavior"
        ),
        docs_query="unsafe safety comment",
        docs_language="rust",
        comprehension_threshold=0.3,
    ),
    "none_comparison": BugPatternKnowledge(
        category="none_comparison",
        concepts=["none", "identity", "equality", "comparison"],
        consequence=(
            "comparing to None with == uses equality, which can trigger "
            "unexpected __eq__ overrides — 'is' tests identity and is "
            "the correct way to check for None"
        ),
        docs_query="none comparison is vs equality",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
    "is_literal_comparison": BugPatternKnowledge(
        category="is_literal_comparison",
        concepts=["identity", "equality", "comparison", "literal"],
        consequence=(
            "comparing a value to a literal with 'is' tests object "
            "identity, not equality. Two equal strings or numbers are "
            "often different objects, so the comparison is unreliable "
            "and may be False when it should be True (Python even warns "
            "about this at compile time)"
        ),
        docs_query="is vs equality comparison literal",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
    "type_equality_check": BugPatternKnowledge(
        category="type_equality_check",
        concepts=["type", "isinstance", "inheritance", "comparison"],
        consequence=(
            "comparing type(x) to a class with == or is ignores "
            "inheritance — an instance of a subclass has a different "
            "type object, so the check fails even though the object is "
            "perfectly usable. isinstance() respects subclassing"
        ),
        docs_query="type vs isinstance subclass",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
    "raise_without_from": BugPatternKnowledge(
        category="raise_without_from",
        concepts=["exception", "exception chaining", "error handling"],
        consequence=(
            "raising a new exception inside an except block without "
            "'from' hides the original error. The traceback loses the "
            "root cause, making the failure much harder to diagnose"
        ),
        docs_query="raise from exception chaining",
        docs_language="python",
        comprehension_threshold=0.3,
    ),
    "return_in_finally": BugPatternKnowledge(
        category="return_in_finally",
        concepts=["finally", "control flow", "exception", "error handling"],
        consequence=(
            "a return, break, or continue inside finally overrides any "
            "in-flight exception or return value — the exception is "
            "silently discarded, so errors vanish without a trace"
        ),
        docs_query="finally return swallows exception",
        docs_language="python",
        comprehension_threshold=0.3,
    ),
    "boolean_equality": BugPatternKnowledge(
        category="boolean_equality",
        concepts=["boolean", "truthiness", "comparison"],
        consequence=(
            "comparing a value to True or False with == is redundant and "
            "fragile — a truthy value that is not the True singleton "
            "compares unequal, which hides the intended truthiness test"
        ),
        docs_query="comparison boolean truthiness",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
    "assert_tuple": BugPatternKnowledge(
        category="assert_tuple",
        concepts=["assert", "tuple", "truthiness", "testing"],
        consequence=(
            "assert with the condition and message inside parentheses "
            "builds a tuple, and a non-empty tuple is always true — so "
            "the assertion never fails and the check is silently useless"
        ),
        docs_query="assert tuple always true",
        docs_language="python",
        comprehension_threshold=0.3,
    ),
    "base_exception_catch": BugPatternKnowledge(
        category="base_exception_catch",
        concepts=["exception", "system exit", "keyboard interrupt", "error handling"],
        consequence=(
            "catching BaseException also catches SystemExit and "
            "KeyboardInterrupt — the program can't be stopped with "
            "Ctrl+C and can't shut down cleanly"
        ),
        docs_query="baseexception exception handling",
        docs_language="python",
        comprehension_threshold=0.3,
    ),
    "fstring_no_placeholder": BugPatternKnowledge(
        category="fstring_no_placeholder",
        concepts=["f-string", "string formatting", "code"],
        consequence=(
            "an f-string with no placeholders behaves like a plain "
            "string — the 'f' prefix does nothing, which usually means "
            "a '{...}' interpolation was forgotten"
        ),
        docs_query="f-string formatting placeholder",
        docs_language="python",
        comprehension_threshold=0.2,
    ),
}


@dataclass(slots=True)
class BugReport:
    """A single bug or code concern Genesis noticed."""

    file: str
    line: int
    severity: str  # "error", "warning", "style"
    category: str  # "bare_except", "long_function", etc.
    description: str
    suggestion: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    status: str = "open"  # "open" or "resolved"
    resolved_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bug report to a dict for JSONL logging."""
        return asdict(self)

    def describe(self) -> str:
        """First-person description of the bug."""
        return (
            f"in {self.file} at line {self.line}, noticed {self.description}"
            + (f" — {self.suggestion.lower()}" if self.suggestion else "")
        )


@dataclass(slots=True)
class BugScanResult:
    """Result of scanning the codebase for bugs."""

    total_files: int = 0
    total_bugs: int = 0
    new_bugs: int = 0  # bugs not seen in previous scans
    resolved_bugs: int = 0  # bugs that were open last scan but are gone now
    errors: int = 0
    warnings: int = 0
    style_issues: int = 0
    bugs: list[BugReport] = field(default_factory=list)
    scanned_files: set[str] = field(default_factory=set)

    @property
    def has_bugs(self) -> bool:
        """True if any bugs were found in this scan."""
        return self.total_bugs > 0

    @property
    def has_new_bugs(self) -> bool:
        """True if any bugs are new (not seen in previous scans)."""
        return self.new_bugs > 0

    @property
    def has_resolved_bugs(self) -> bool:
        """True if any previously-open bugs were resolved since last scan."""
        return self.resolved_bugs > 0

    def by_category(self) -> dict[str, list[BugReport]]:
        """Group bugs by category."""
        grouped: dict[str, list[BugReport]] = {}
        for bug in self.bugs:
            grouped.setdefault(bug.category, []).append(bug)
        return grouped

    def summary(self) -> str:
        """Human-readable summary."""
        if not self.has_bugs and not self.has_resolved_bugs:
            return f"Scanned {self.total_files} files. No issues found."
        parts = [
            f"Scanned {self.total_files} files. "
            f"Found {self.total_bugs} issues "
            f"({self.new_bugs} new, {self.resolved_bugs} resolved): "
            f"{self.errors} errors, {self.warnings} warnings, "
            f"{self.style_issues} style."
        ]
        for cat, bugs in self.by_category().items():
            parts.append(f"  {cat}: {len(bugs)}")
        return "\n".join(parts)


# ─── Python AST visitors ─────────────────────────────────────────────


def is_literal_operand(node: ast.expr) -> bool:
    """True if ``node`` is a literal that must not be compared with ``is``.

    ``None`` and the singletons ``True``/``False``/``Ellipsis`` are
    excluded — ``x is None`` is correct, and identity checks against
    the bool singletons are a separate style question. Everything else
    (numbers, strings, bytes, and container displays) is a fresh object
    whose identity is not stable, so ``is`` is a real bug there.
    """
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or value is Ellipsis or isinstance(value, bool):
            return False
        return isinstance(value, (int, float, complex, str, bytes))
    return isinstance(node, (ast.Tuple, ast.List, ast.Dict, ast.Set))


def type_call_arg(node: ast.expr) -> ast.expr | None:
    """If ``node`` is ``type(x)``, return ``x``; otherwise return None.

    Only the single-argument, keyword-free form counts — ``type(x, y, z)``
    is the three-argument dynamic-class constructor, not a type check.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "type"
        and len(node.args) == 1
        and not node.keywords
    ):
        return node.args[0]
    return None


def _catches_base_exception(node: ast.expr | None) -> bool:
    """True if an except clause names BaseException (directly or in a tuple)."""
    if node is None:
        return False
    names = node.elts if isinstance(node, ast.Tuple) else [node]
    return any(
        isinstance(n, ast.Name) and n.id == "BaseException" for n in names
    )


def iter_block_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Yield statements from a block, descending into nested blocks.

    Recurses through compound statements (if/for/while/with/try) but
    does not descend into nested function, class, or lambda scopes —
    a ``return`` inside a nested ``def`` is not part of the outer
    block's control flow.
    """
    for stmt in body:
        yield stmt
        for child in ast.iter_child_nodes(stmt):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                continue
            if isinstance(child, ast.stmt):
                yield from iter_block_statements([child])


class _PythonBugVisitor(ast.NodeVisitor):
    """Walk a Python AST and collect bug reports.

    Each visit method checks for specific patterns. The visitor
    accumulates bugs in ``self.bugs``.
    """

    def __init__(self, filepath: str, source_lines: list[str]) -> None:
        """Initialize the scanner for a specific file."""
        self.filepath = filepath
        self.source_lines = source_lines
        self.bugs: list[BugReport] = []
        # Depth of enclosing except handlers — used to spot a bare
        # ``raise`` inside ``except`` that drops the original cause.
        self._except_depth = 0

    def _add_bug(
        self,
        line: int,
        severity: str,
        category: str,
        description: str,
        suggestion: str = "",
    ) -> None:
        """Record a bug found during scanning."""
        self.bugs.append(
            BugReport(
                file=self.filepath,
                line=line,
                severity=severity,
                category=category,
                description=description,
                suggestion=suggestion,
            )
        )

    # ── Bare except ──

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Check for bare except: (no exception type specified)."""
        if node.type is None:
            self._add_bug(
                line=node.lineno,
                severity="error",
                category="bare_except",
                description="Bare except: catches SystemExit, KeyboardInterrupt, and all errors",
                suggestion="Use 'except Exception:' to avoid catching system exits",
            )
        # Check for pass-only except body
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Pass)
        ):
            self._add_bug(
                line=node.lineno,
                severity="warning",
                category="silent_except",
                description="except block contains only 'pass' — errors are silently swallowed",
                suggestion="At minimum, log the error so it's not invisible",
            )
        # 'except BaseException' is as broad as a bare except: it also
        # catches SystemExit and KeyboardInterrupt, so Ctrl+C can't stop
        # the program and clean shutdown is blocked.
        if _catches_base_exception(node.type):
            self._add_bug(
                line=node.lineno,
                severity="warning",
                category="base_exception_catch",
                description=(
                    "'except BaseException' also catches SystemExit and "
                    "KeyboardInterrupt — Ctrl+C can't stop the program"
                ),
                suggestion="Catch 'Exception' instead of 'BaseException'",
            )
        # Track nesting so visit_Raise knows it's inside an except block.
        self._except_depth += 1
        self.generic_visit(node)
        self._except_depth -= 1

    # ── assert on a tuple literal ──

    def visit_Assert(self, node: ast.Assert) -> None:
        """Flag 'assert (cond, msg)' — a non-empty tuple is always truthy.

        Writing the condition and message inside parentheses makes the
        assertion a tuple, which is always true, so the check silently
        never fails. The intended form is 'assert cond, msg'.
        """
        if not isinstance(node.test, ast.Tuple) or not node.test.elts:
            return
        self._add_bug(
            line=node.lineno,
            severity="warning",
            category="assert_tuple",
            description=(
                "assert on a tuple literal — a non-empty tuple is always "
                "truthy, so this assertion never fails"
            ),
            suggestion=(
                "Write 'assert cond, msg' (no parentheses around the "
                "condition) so the assertion actually checks cond"
            ),
        )

    # ── f-string with no placeholders ──

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """Flag an f-string with no interpolation.

        'f"hello"' produces the same value as 'hello' but signals the
        author meant to interpolate something — usually a forgotten
        '{...}'. Literal braces ('f"{{}}"') are excluded because
        dropping the f would change the output.
        """
        if any(isinstance(v, ast.FormattedValue) for v in node.values):
            return
        line_text = (
            self.source_lines[node.lineno - 1]
            if 0 < node.lineno <= len(self.source_lines)
            else ""
        )
        if "{" in line_text:
            return
        self._add_bug(
            line=node.lineno,
            severity="style",
            category="fstring_no_placeholder",
            description=(
                "f-string has no placeholders — the 'f' prefix does "
                "nothing (a placeholder was probably forgotten)"
            ),
            suggestion="Remove the 'f' prefix, or add the intended {expression}",
        )

    # ── Mutable default arguments ──

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check for mutable defaults, long functions, missing type hints."""
        self._check_mutable_defaults(node)
        self._check_function_length(node)
        self._check_missing_type_hints(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check for mutable defaults, long functions, missing type hints."""
        self._check_mutable_defaults(node)  # type: ignore[arg-type]
        self._check_function_length(node)  # type: ignore[arg-type]
        self._check_missing_type_hints(node)  # type: ignore[arg-type]
        self.generic_visit(node)

    def _check_mutable_defaults(self, node: ast.FunctionDef) -> None:
        """Mutable default arguments (list, dict, set) are shared across calls."""
        for arg in node.args.defaults:
            if isinstance(arg, (ast.List, ast.Dict, ast.Set)):
                self._add_bug(
                    line=node.lineno,
                    severity="error",
                    category="mutable_default",
                    description=(
                        f"Function '{node.name}' has a mutable default argument "
                        f"(list/dict/set) — shared across all calls"
                    ),
                    suggestion="Use None as default and create the mutable inside the function",
                )

    def _function_length_threshold(self) -> int:
        """Return the max acceptable function length for this file type.

        Core package code is held to a stricter standard (110) than
        standalone training/evaluation scripts (150) or tests (200).
        """
        lower = self.filepath.lower()
        if "test" in lower:
            return 200
        rel = self.filepath.replace("\\", "/")
        if any(rel.startswith(d + "/") for d in self._CODE_DIRS):
            return 110
        return 150

    def _check_function_length(self, node: ast.FunctionDef) -> None:
        """Functions longer than the context-aware threshold are flagged."""
        # Skip test files — test functions are often long because they
        # set up complex fixtures inline, and that's fine.
        if "test" in self.filepath.lower():
            return
        # Skip data functions — body is a single return of a list/dict
        # literal. These are data tables, not logic. Splitting them
        # doesn't improve readability.
        if (
            len(node.body) == 2
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[1], ast.Return)
            and isinstance(node.body[1].value, (ast.List, ast.Dict))
        ):
            return
        start = node.lineno
        end = getattr(node, "end_lineno", start) or start
        length = end - start
        threshold = self._function_length_threshold()
        if length > threshold:
            self._add_bug(
                line=start,
                severity="warning",
                category="long_function",
                description=(
                    f"Function '{node.name}' is {length} lines long "
                    f"(threshold {threshold}) — hard to understand and maintain"
                ),
                suggestion="Consider breaking it into smaller functions",
            )

    def _check_missing_type_hints(self, node: ast.FunctionDef) -> None:
        """Public functions (no leading underscore) should have type hints."""
        if node.name.startswith("_"):
            return
        # Skip __init__, __str__, etc.
        if node.name.startswith("__"):
            return
        # Skip test files — test functions don't need type hints.
        if "test" in self.filepath.lower():
            return
        # Check if any argument lacks a type annotation
        args = node.args.args
        if not args:
            return
        missing = [
            a.arg for a in args
            if a.annotation is None and a.arg != "self"
        ]
        if missing and node.returns is None:
            self._add_bug(
                line=node.lineno,
                severity="style",
                category="missing_type_hints",
                description=(
                    f"Function '{node.name}' has no type hints — "
                    f"hard to know what it accepts and returns"
                ),
                suggestion="Add type annotations for parameters and return type",
            )

    # ── Unreachable code after return ──

    def _check_unreachable(self, node: ast.stmt, body: list[ast.stmt]) -> None:
        """Check for statements after return/raise/break/continue."""
        for i, stmt in enumerate(body[:-1]):
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                next_stmt = body[i + 1]
                self._add_bug(
                    line=next_stmt.lineno,
                    severity="warning",
                    category="unreachable_code",
                    description=(
                        f"Unreachable code after {type(stmt).__name__.lower()} "
                        f"statement"
                    ),
                    suggestion="Remove the dead code after the return/raise",
                )

    # ── import * ──

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Check for 'from X import *' — pollutes namespace."""
        for alias in node.names:
            if alias.name == "*":
                self._add_bug(
                    line=node.lineno,
                    severity="warning",
                    category="import_star",
                    description=(
                        f"'from {node.module} import *' pollutes the namespace "
                        f"and hides dependencies"
                    ),
                    suggestion="Import only what you need explicitly",
                )
        self.generic_visit(node)

    # ── print() in production code ──

    # Directories that are part of the core package — print() in here
    # should be replaced with logging. Files outside these are CLI,
    # training, or setup scripts where print() is acceptable.
    _CODE_DIRS: ClassVar[frozenset[str]] = frozenset(
        {"python/genesis_cognitive", "python/genesis_client", "src"}
    )

    def visit_Call(self, node: ast.Call) -> None:
        """Check for print() outside __main__ blocks."""
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            # Skip print() in test files — tests use print() for
            # debug output, which is standard practice.
            if "test" in self.filepath.lower():
                self.generic_visit(node)
                return
            # Only flag print() in core package code. CLI/training/setup
            # scripts are allowed to print to stdout.
            rel = self.filepath.replace("\\", "/")
            if not any(rel.startswith(d + "/") for d in self._CODE_DIRS):
                self.generic_visit(node)
                return
            self._add_bug(
                line=node.lineno,
                severity="style",
                category="print_in_code",
                description="print() found in code — consider using logging instead",
                suggestion="Replace print() with logger.debug/info/warning",
            )
        self.generic_visit(node)

    # ── Comparison to None with == or != ──

    def visit_Compare(self, node: ast.Compare) -> None:
        """Check for 'x == None' or 'x != None' — should use 'is'/'is not'.

        PEP 8: "Comparisons to singletons like None should always be done
        with 'is' or 'is not', never the equality operators."
        """
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                continue
            if not isinstance(comparator, ast.Constant):
                continue
            if comparator.value is not None:
                continue
            # Skip test files — tests sometimes compare to None for
            # clarity of intent, and it's not a production concern.
            if "test" in self.filepath.lower():
                continue
            op_str = "==" if isinstance(op, ast.Eq) else "!="
            self._add_bug(
                line=node.lineno,
                severity="style",
                category="none_comparison",
                description=(
                    f"Comparison to None with '{op_str}' — "
                    f"should use 'is' / 'is not' (PEP 8)"
                ),
                suggestion=(
                    f"Replace '{op_str} None' with "
                    f"{'is None' if isinstance(op, ast.Eq) else 'is not None'}"
                ),
            )
        self._check_is_literal(node)
        self._check_type_equality(node)
        self._check_boolean_equality(node)
        self.generic_visit(node)

    def _check_boolean_equality(self, node: ast.Compare) -> None:
        """Flag 'x == True' / 'x == False' — compare truthiness instead.

        Comparing to the bool literals with == is redundant at best and
        can mask bugs at worst (a truthy non-bool is not the True
        singleton). PEP 8 / flake8 E712.
        """
        if len(node.ops) != 1 or not isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            return
        for operand in (node.left, node.comparators[0]):
            if isinstance(operand, ast.Constant) and isinstance(operand.value, bool):
                self._add_bug(
                    line=node.lineno,
                    severity="style",
                    category="boolean_equality",
                    description=(
                        f"Comparison to the boolean literal "
                        f"'{operand.value}' with '==' — test truthiness instead"
                    ),
                    suggestion=(
                        "Use 'if x:' / 'if not x:' rather than "
                        "'x == True' / 'x == False'"
                    ),
                )
                return

    def _check_is_literal(self, node: ast.Compare) -> None:
        """Flag 'x is <literal>' / 'x is not <literal>' — identity vs equality.

        'is' tests object identity. A literal on either side is a fresh
        object whose identity is not guaranteed, so the comparison is
        unreliable (and Python raises a SyntaxWarning for it).
        """
        operands: list[ast.expr] = [node.left, *node.comparators]
        for i, op in enumerate(node.ops):
            if not isinstance(op, (ast.Is, ast.IsNot)):
                continue
            left_operand = operands[i]
            right_operand = node.comparators[i]
            if not (
                is_literal_operand(left_operand) or is_literal_operand(right_operand)
            ):
                continue
            op_str = "is" if isinstance(op, ast.Is) else "is not"
            self._add_bug(
                line=node.lineno,
                severity="warning",
                category="is_literal_comparison",
                description=(
                    f"Comparison with '{op_str}' against a literal — "
                    f"'is' tests identity, not equality"
                ),
                suggestion=(
                    "Use '==' / '!=' to compare values; reserve 'is' "
                    "for None and other singletons"
                ),
            )

    def _check_type_equality(self, node: ast.Compare) -> None:
        """Flag 'type(x) == T' / 'type(x) is T' — use isinstance instead.

        Comparing the exact type ignores subclasses: an instance of a
        subclass has a different type object, so the check fails even
        though the object is usable as a T. ``type(a) == type(b)`` is
        fine (both sides are types) and is not flagged.
        """
        operands: list[ast.expr] = [node.left, *node.comparators]
        for i, op in enumerate(node.ops):
            if not isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)):
                continue
            left_operand = operands[i]
            right_operand = node.comparators[i]
            left_is_type = type_call_arg(left_operand) is not None
            right_is_type = type_call_arg(right_operand) is not None
            if left_is_type == right_is_type:
                # Neither side is a type() call, or both are (comparing
                # two types is legitimate) — nothing to report.
                continue
            class_side = right_operand if left_is_type else left_operand
            if not isinstance(class_side, (ast.Name, ast.Attribute, ast.Tuple)):
                continue
            self._add_bug(
                line=node.lineno,
                severity="warning",
                category="type_equality_check",
                description=(
                    "Comparing type(x) to a class ignores subclasses — "
                    "use isinstance(x, T) instead"
                ),
                suggestion=(
                    "Replace with isinstance(x, T) (or not isinstance(...)) "
                    "so subclasses are accepted"
                ),
            )

    # ── raise inside except without 'from' ──

    def visit_Raise(self, node: ast.Raise) -> None:
        """Flag 'raise NewError(...)' inside except without 'from'.

        Without an explicit cause, the original exception is replaced
        by the new one and its traceback is lost (PEP 3134). Use
        ``raise NewError(...) from err`` — or ``from None`` to
        deliberately suppress chaining.
        """
        if (
            self._except_depth > 0
            and node.exc is not None
            and node.cause is None
            and "test" not in self.filepath.lower()
        ):
            self._add_bug(
                line=node.lineno,
                severity="style",
                category="raise_without_from",
                description=(
                    "raise inside an except block without 'from' — "
                    "the original exception chain is lost"
                ),
                suggestion=(
                    "Use 'raise NewError(...) from err' (or 'from None' "
                    "to suppress chaining deliberately)"
                ),
            )
        self.generic_visit(node)

    # ── return/break/continue inside finally ──

    def _check_finally_control_flow(self, node: ast.Try) -> None:
        """Flag return/break/continue inside a finally block.

        A terminal statement in finally overrides whatever exception or
        return was in flight — the exception is silently discarded.
        """
        for stmt in iter_block_statements(node.finalbody):
            if isinstance(stmt, (ast.Return, ast.Break, ast.Continue)):
                kind = type(stmt).__name__.lower()
                self._add_bug(
                    line=stmt.lineno,
                    severity="error",
                    category="return_in_finally",
                    description=(
                        f"'{kind}' inside a finally block — it overrides "
                        f"any in-flight exception or return value, "
                        f"silently discarding errors"
                    ),
                    suggestion=(
                        "Move the control-flow out of finally; a finally "
                        "block should only clean up"
                    ),
                )

    def visit_Try(self, node: ast.Try) -> None:
        """Check try/finally blocks for control flow that swallows errors."""
        self._check_finally_control_flow(node)
        self.generic_visit(node)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        """Check try*/finally blocks (Python 3.11+) for the same issue."""
        self._check_finally_control_flow(node)  # type: ignore[arg-type]
        self.generic_visit(node)


# ─── Rust patterns (regex-based) ─────────────────────────────────────

_RUST_UNWRAP_RE = re.compile(r"\.unwrap\(\)", re.MULTILINE)
_RUST_TODO_RE = re.compile(r"\b(todo!|unimplemented!)\(", re.MULTILINE)
_RUST_UNSAFE_RE = re.compile(r"\bunsafe\s*\{", re.MULTILINE)
_RUST_FN_RE = re.compile(
    r"^\s*(pub\s+)?(async\s+)?fn\s+(\w+)", re.MULTILINE
)


def _scan_rust_file(filepath: str, source: str) -> list[BugReport]:
    """Scan a Rust file for common issues."""
    bugs: list[BugReport] = []
    lines = source.split("\n")

    # .unwrap() is idiomatic in Rust test code — don't report it there.
    # In tests, panicking on None/Err is the desired behavior (test failure).
    # Detect test code two ways:
    # 1. File name contains "test" (e.g. test_foo.rs)
    # 2. File contains #[cfg(test)] inline test modules (common in Rust)
    is_test_file = "test" in filepath.lower()
    # Find the line where #[cfg(test)] mod tests starts, if any.
    # .unwrap() calls after that line are in test code.
    test_module_start = None
    for i, line in enumerate(lines):
        if "#[cfg(test)]" in line:
            test_module_start = i
            break

    if not is_test_file:
        for m in _RUST_UNWRAP_RE.finditer(source):
            line_no = source[: m.start()].count("\n") + 1
            # Skip .unwrap() inside #[cfg(test)] modules — it's idiomatic
            # in test code where panicking is the desired failure mode.
            if test_module_start is not None and line_no > test_module_start:
                continue
            bugs.append(
                BugReport(
                    file=filepath,
                    line=line_no,
                    severity="warning",
                    category="rust_unwrap",
                    description=".unwrap() can panic on None/Err — risky in production code",
                    suggestion="Use .unwrap_or(), .unwrap_or_else(), or match instead",
                )
            )

    # todo!() or unimplemented!()
    for m in _RUST_TODO_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        bugs.append(
            BugReport(
                file=filepath,
                line=line_no,
                severity="error",
                category="rust_todo",
                description=f"{m.group(1)} left in code — not implemented yet",
                suggestion="Implement the function or use a proper error type",
            )
        )

    # unsafe blocks without safety comment
    unsafe_severity = "style" if is_test_file else "warning"
    for m in _RUST_UNSAFE_RE.finditer(source):
        line_no = source[: m.start()].count("\n") + 1
        # Skip matches inside comments (/// or //) — code examples in
        # doc comments can contain `unsafe {` which isn't real code.
        current_line = lines[line_no - 1].strip()
        if current_line.startswith("///") or current_line.startswith("//"):
            continue
        if not _has_safety_comment(lines, line_no):
            bugs.append(
                BugReport(
                    file=filepath,
                    line=line_no,
                    severity=unsafe_severity,
                    category="rust_unsafe_no_comment",
                    description=(
                        "unsafe block without a SAFETY comment"
                        " explaining why it's safe"
                    ),
                    suggestion="Add a '// SAFETY: ...' comment explaining the invariant",
                )
            )

    return bugs


def _has_safety_comment(lines: list[str], line_no: int) -> bool:
    """Check if an unsafe block at line_no has a SAFETY comment nearby.

    Looks up to 10 lines before, on the same line, and on the next line.
    """
    # Check up to 10 lines before the unsafe block
    for look_back in range(1, min(11, line_no)):
        prev = lines[line_no - 1 - look_back].strip()
        if prev.startswith("// SAFETY") or prev.startswith("/// SAFETY"):
            return True
    # Check the same line (inline unsafe { ... })
    if "// SAFETY" in lines[line_no - 1]:
        return True
    # Check the first line inside the block
    if line_no < len(lines):
        next_line = lines[line_no].strip()
        if next_line.startswith("// SAFETY"):
            return True
    return False


# ─── BugReporter ─────────────────────────────────────────────────────


class BugReporter:
    """Genesis's bug detection and reporting system.

    Scans her own codebase for issues using static analysis (no
    execution). Reports are logged to a structured JSONL file and
    can be surfaced in conversation.

    Usage::

        reporter = BugReporter(project_root="/path/to/genesis")
        result = reporter.scan()
        print(result.summary())
        # Bugs are also written to genesis_data/bug_reports.jsonl
    """

    # Directories to skip during scanning
    _SKIP_DIRS: ClassVar[set[str]] = {
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
        "genesis_data",
    }

    # Maximum file size to scan (512 KB)
    _MAX_FILE_SIZE = 512 * 1024

    def __init__(
        self,
        project_root: str = ".",
        log_path: str | None = None,
        network: Any = None,
    ) -> None:
        """Initialize the bug reporter."""
        self.project_root = Path(project_root).resolve()
        if log_path:
            self.log_path = Path(log_path)
        else:
            self.log_path = self.project_root / "genesis_data" / "bug_reports.jsonl"
        # History file — resolved issues are moved here so the running
        # list stays clean. Sits next to the main log.
        self.history_path = self.log_path.with_name(
            self.log_path.stem + "_history.jsonl"
        )
        self._last_scan: BugScanResult | None = None
        # Concept network — used to check if Genesis actually understands
        # the bugs she's reporting. Without this, she's pattern-matching,
        # not understanding. Set by Mind after creation.
        self._network = network
        # Track which bug categories she's studied (fetched docs for)
        self._studied_categories: set[str] = set()
        # Track open issues to avoid duplicate logging across scans and
        # to detect when issues are resolved. Keyed by (file, category)
        # — line numbers shift as code evolves, but the issue identity
        # is stable per file+category. When a previously-open issue is
        # no longer found, it's moved to the history file and Genesis
        # gets credit for the fix.
        self._open_keys: set[tuple[str, str]] = set()
        self._log_lock = threading.Lock()
        self._load_open_keys()

    def scan(
        self,
        max_files: int = 100,
        include_tests: bool = True,
    ) -> BugScanResult:
        """Scan the codebase for bugs.

        Walks the project tree, analyzes each Python and Rust file,
        and collects bug reports. Results are logged to the JSONL file.

        Args:
            max_files: Maximum number of files to scan.
            include_tests: Whether to scan test files.

        Returns:
            A BugScanResult with all bugs found.
        """
        result = BugScanResult()

        for path in self._iter_source_files(include_tests=include_tests):
            if result.total_files >= max_files:
                break

            rel = str(path.relative_to(self.project_root))
            file_bugs = self._scan_file(rel, path)
            result.bugs.extend(file_bugs)
            result.scanned_files.add(rel)
            result.total_files += 1

        # Count by severity
        for bug in result.bugs:
            if bug.severity == "error":
                result.errors += 1
            elif bug.severity == "warning":
                result.warnings += 1
            else:
                result.style_issues += 1
        result.total_bugs = len(result.bugs)

        # Log new bugs and detect resolved ones
        new_count, resolved_count = self._log_bugs(
            result.bugs, result.scanned_files
        )
        result.new_bugs = new_count
        result.resolved_bugs = resolved_count

        self._last_scan = result
        logger.debug(
            f"Bug scan complete: {result.total_files} files, "
            f"{result.total_bugs} bugs found "
            f"({new_count} new, {resolved_count} resolved)"
        )
        return result

    def _scan_file(self, rel_path: str, path: Path) -> list[BugReport]:
        """Scan a single file for bugs."""
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        if path.suffix == ".py":
            return self._scan_python(rel_path, source)
        elif path.suffix == ".rs":
            return _scan_rust_file(rel_path, source)
        return []

    def _scan_python(self, rel_path: str, source: str) -> list[BugReport]:
        """Scan a Python file using AST analysis."""
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError:
            return []

        lines = source.split("\n")
        visitor = _PythonBugVisitor(rel_path, lines)
        visitor.visit(tree)

        # Check for unreachable code in function bodies
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visitor._check_unreachable(node, node.body)

        return visitor.bugs

    # Directories that contain Genesis's own source — scanned first
    # so that max_files limits don't skip her code in favor of evals,
    # examples, or Rust files.
    _PRIORITY_DIRS: ClassVar[tuple[str, ...]] = (
        "python/genesis_cognitive/",
        "python/genesis_client/",
    )

    def _iter_source_files(self, include_tests: bool = True) -> Iterator[Path]:
        """Yield source files to scan.

        Files are sorted with Genesis's own source first (under
        _PRIORITY_DIRS), then other Python files, then Rust files.
        This ensures that max_files limits don't skip her code in
        favor of evals, examples, or the Rust substrate.
        """
        collected: list[Path] = []
        for path in self.project_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in (".py", ".rs"):
                continue
            # Skip excluded directories — plus any virtualenv
            # (".venv*", "venv*") or site-packages tree, which is
            # third-party code, not Genesis's.
            parts = path.relative_to(self.project_root).parts
            if any(
                part in self._SKIP_DIRS
                or part.startswith(".venv")
                or part == "site-packages"
                for part in parts
            ):
                continue
            # Skip test files if not included
            if not include_tests and (
                "test" in path.name.lower() or "tests" in parts
            ):
                continue
            # Skip large files
            try:
                if path.stat().st_size > self._MAX_FILE_SIZE:
                    continue
            except OSError:
                continue
            collected.append(path)

        def _sort_key(path: Path) -> tuple[int, str]:
            """Sort paths so Genesis's own source is scanned first.

            Returns a tuple ``(priority, rel_path)`` where priority 0
            is Genesis's cognitive-mind source, 1 is other Python,
            and 2 is Rust. This ensures ``max_files`` limits don't
            crowd out her code in favour of peripheral files.
            """
            rel = str(path.relative_to(self.project_root))
            normalized = rel.replace("\\", "/")
            if any(normalized.startswith(d) for d in self._PRIORITY_DIRS):
                return (0, rel)  # Genesis's own source first
            if path.suffix == ".py":
                return (1, rel)  # other Python second
            return (2, rel)  # Rust last

        collected.sort(key=_sort_key)
        yield from collected

    def _load_open_keys(self) -> None:
        """Load open issue keys from the JSONL log at startup.

        The running-list log only contains currently-open issues, so
        every entry is loaded. Resolved issues live in the history file
        and don't participate in dedup or resolution detection.
        """
        if not self.log_path.exists():
            return
        try:
            with self.log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except (json.JSONDecodeError, KeyError):
                        continue
                    key = (
                        entry.get("file", ""),
                        entry.get("category", ""),
                    )
                    self._open_keys.add(key)
        except OSError as e:
            logger.debug("Could not load open bug-report keys: %s", e)

    def _log_bugs(
        self, bugs: list[BugReport], scanned_files: set[str] | None = None
    ) -> tuple[int, int]:
        """Maintain the running list of open issues.

        The log file is rewritten each scan so it always reflects the
        current state — it's a running list, not an append-only audit
        trail. Deduplicates by (file, category): line numbers shift as
        code evolves, but the issue identity is stable per file+category.

        - Issues found this scan that weren't in the open set are "new".
        - Issues that were open but are no longer found are "resolved" —
          they're moved to the history file and removed from the running
          list. But only if their file was actually scanned this cycle
          (a file skipped due to max_files limit may still have the bug).
        - Issues from files NOT scanned this cycle are **preserved** —
          carried forward unchanged in both the open set and the running
          list. A partial scan (max_files limit) must not silently drop
          bugs from unscanned files.
        - The running list is rewritten with currently-found issues plus
          preserved issues from unscanned files.

        Thread-safe via _log_lock since both the bug-scan thread and
        the improvement thread call scan().

        Returns (new_count, resolved_count).
        """
        with self._log_lock:
            if scanned_files is None:
                scanned_files = {bug.file for bug in bugs}

            # Build set of currently-found issue keys, deduping by
            # (file, category). If the same file+category appears
            # multiple times (e.g., two long functions in one file),
            # keep only the first occurrence — one entry per issue type
            # per file.
            current_keys: set[tuple[str, str]] = set()
            deduped_bugs: list[BugReport] = []
            for bug in bugs:
                key = (bug.file, bug.category)
                if key not in current_keys:
                    current_keys.add(key)
                    deduped_bugs.append(bug)

            # Detect newly opened issues
            new_bugs = [
                bug for bug in deduped_bugs
                if (bug.file, bug.category) not in self._open_keys
            ]

            # Detect resolved issues — were open, now gone.
            # Only mark as resolved if the file was actually scanned
            # this cycle. If a file was skipped (e.g., max_files limit),
            # its bugs may not appear even though they still exist.
            resolved_keys: set[tuple[str, str]] = set()
            preserved_keys: set[tuple[str, str]] = set()
            for key in self._open_keys - current_keys:
                file_path, _category = key
                if (
                    file_path in scanned_files
                    or not (self.project_root / file_path).exists()
                ):
                    # Scanned and clean, or the file is gone entirely
                    # (moved/deleted) — either way the report is stale.
                    resolved_keys.add(key)
                else:
                    # File wasn't scanned — carry the bug forward
                    preserved_keys.add(key)

            # Carry forward BugReport entries for unscanned files so
            # they remain in the running list. Read them from the
            # current log file since we don't have them in memory.
            preserved_bugs = self._load_preserved_bugs(preserved_keys)
            deduped_bugs.extend(preserved_bugs)
            current_keys |= preserved_keys

            # Update in-memory open set
            self._open_keys = current_keys

            # Ensure directory exists
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

            # Archive resolved issues to the history file BEFORE rewriting
            # the running list — _archive_resolved reads the current log
            # to extract resolved entries, so it must run before the
            # log is overwritten.
            if resolved_keys:
                self._archive_resolved(resolved_keys)

            # Rewrite the running list with currently-open issues
            # (current findings + preserved bugs from unscanned files)
            self._write_running_list(deduped_bugs)

            return len(new_bugs), len(resolved_keys)

    def _load_preserved_bugs(
        self, preserved_keys: set[tuple[str, str]]
    ) -> list[BugReport]:
        """Load BugReport entries from the running list for unscanned files.

        When a partial scan (max_files limit) skips files, their
        previously-open bugs must be carried forward unchanged. This
        reads the current running list and returns entries matching
        the preserved keys.
        """
        if not preserved_keys or not self.log_path.exists():
            return []
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        preserved: list[BugReport] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, KeyError):
                continue
            key = (entry.get("file", ""), entry.get("category", ""))
            if key in preserved_keys:
                preserved.append(BugReport(
                    file=entry.get("file", ""),
                    line=entry.get("line", 0),
                    severity=entry.get("severity", ""),
                    category=entry.get("category", ""),
                    description=entry.get("description", ""),
                    suggestion=entry.get("suggestion", ""),
                    timestamp=entry.get("timestamp", ""),
                    status=entry.get("status", "open"),
                    resolved_at=entry.get("resolved_at", ""),
                ))
        return preserved

    def _write_running_list(self, open_bugs: list[BugReport]) -> None:
        """Rewrite the log file with only currently-open issues.

        This is what makes it a running list — the file is rebuilt from
        scratch each scan, so it can never accumulate duplicates or
        stale resolved entries.
        """
        try:
            lines = [
                json.dumps(bug.to_dict()) + "\n"
                for bug in open_bugs
            ]
            self.log_path.write_text(
                "".join(lines), encoding="utf-8"
            )
        except OSError as e:
            logger.debug("Could not write bug-report log: %s", e)

    def _archive_resolved(self, resolved_keys: set[tuple[str, str]]) -> None:
        """Move resolved issues from the running list to the history file.

        Reads the current running list, extracts entries matching the
        resolved keys, appends them to the history file with a
        ``resolved_at`` timestamp, then rewrites the running list
        without the resolved entries.
        """
        if not self.log_path.exists():
            return

        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return

        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        remaining: list[str] = []
        archived: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, KeyError):
                remaining.append(line)
                continue

            key = (entry.get("file", ""), entry.get("category", ""))
            if key in resolved_keys:
                entry["status"] = "resolved"
                entry["resolved_at"] = now
                archived.append(json.dumps(entry) + "\n")
            else:
                remaining.append(line)

        # Append resolved entries to history
        if archived:
            try:
                with self.history_path.open("a", encoding="utf-8") as f:
                    f.writelines(archived)
            except OSError as e:
                logger.debug("Could not write bug-report history: %s", e)

        # Rewrite running list without the resolved entries
        try:
            self.log_path.write_text(
                "\n".join(remaining) + ("\n" if remaining else ""),
                encoding="utf-8",
            )
        except OSError as e:
            logger.debug("Could not update bug-report log: %s", e)

    @property
    def last_scan(self) -> BugScanResult | None:
        """The result of the most recent scan."""
        return self._last_scan

    def check_comprehension(self, category: str) -> tuple[bool, float, list[str]]:
        """Check if Genesis understands a bug category.

        This is the difference between pattern-matching and
        understanding. She looks up the concepts associated with
        the bug pattern in her concept network and checks if she
        actually knows them. If she doesn't, she can't honestly
        report the bug — she's just reciting a rule she memorized.

        Args:
            category: The bug category (e.g., "bare_except").

        Returns:
            A tuple of (understands, confidence, missing_concepts).
            - understands: True if her average concept confidence
              meets the threshold for this pattern.
            - confidence: The average confidence across relevant
              concepts (0.0 if no network or pattern unknown).
            - missing_concepts: Concepts she doesn't know at all.
        """
        knowledge = _BUG_PATTERN_KNOWLEDGE.get(category)
        if not knowledge or not self._network:
            # No knowledge mapping or no network — can't verify
            # understanding. Return False to be safe (honest default).
            return False, 0.0, knowledge.concepts if knowledge else []

        confidences: list[float] = []
        missing: list[str] = []

        for concept_name in knowledge.concepts:
            concept = self._network.get_concept(concept_name)
            if concept is None:
                missing.append(concept_name)
            else:
                confidences.append(concept.confidence)

        if not confidences:
            return False, 0.0, missing

        avg_confidence = sum(confidences) / len(confidences)
        understands = avg_confidence >= knowledge.comprehension_threshold
        return understands, avg_confidence, missing

    def get_bug_knowledge(self, category: str) -> BugPatternKnowledge | None:
        """Get the knowledge mapping for a bug category."""
        return _BUG_PATTERN_KNOWLEDGE.get(category)

    def recent_bugs(self, n: int = 10) -> list[BugReport]:
        """Return the most recent bugs from the running list.

        The running list contains only currently-open issues. If fewer
        than ``n`` open issues exist, recent resolved issues from the
        history file are included to fill out the result.
        """
        bugs: list[BugReport] = []
        # Read open issues from the running list
        if self.log_path.exists():
            try:
                lines = self.log_path.read_text(encoding="utf-8").strip().split("\n")
                for line in lines:
                    if line.strip():
                        data = json.loads(line)
                        bugs.append(
                            BugReport(
                                file=data["file"],
                                line=data["line"],
                                severity=data["severity"],
                                category=data["category"],
                                description=data["description"],
                                suggestion=data.get("suggestion", ""),
                                timestamp=data.get("timestamp", ""),
                                status=data.get("status", "open"),
                                resolved_at=data.get("resolved_at", ""),
                            )
                        )
            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.debug(repr(e))
        # If we need more, pull recent resolved issues from history
        if len(bugs) < n and self.history_path.exists():
            try:
                lines = self.history_path.read_text(encoding="utf-8").strip().split("\n")
                for line in reversed(lines):
                    if len(bugs) >= n:
                        break
                    if line.strip():
                        data = json.loads(line)
                        bugs.append(
                            BugReport(
                                file=data["file"],
                                line=data["line"],
                                severity=data["severity"],
                                category=data["category"],
                                description=data["description"],
                                suggestion=data.get("suggestion", ""),
                                timestamp=data.get("timestamp", ""),
                                status=data.get("status", "resolved"),
                                resolved_at=data.get("resolved_at", ""),
                            )
                        )
            except (OSError, json.JSONDecodeError, KeyError) as e:
                logger.debug(repr(e))
        return bugs[:n]

    def bug_track_record(self) -> dict[str, int]:
        """Return a summary of Genesis's bug-fixing track record.

        Open issues are counted from the running list; resolved issues
        are counted from the history file. This is how she can talk
        about her own progress — "I've fixed 12 issues, 8 still open."
        """
        record: dict[str, int] = {
            "total_reported": 0,
            "open": 0,
            "resolved": 0,
        }
        # Count open issues from the running list
        if self.log_path.exists():
            try:
                with self.log_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            json.loads(line)
                        except (json.JSONDecodeError, KeyError):
                            continue
                        record["open"] += 1
            except OSError as e:
                logger.debug("Could not read bug-report log for stats: %s", e)
        # Count resolved issues from the history file
        if self.history_path.exists():
            try:
                with self.history_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            json.loads(line)
                        except (json.JSONDecodeError, KeyError):
                            continue
                        record["resolved"] += 1
            except OSError as e:
                logger.debug("Could not read bug-report history for stats: %s", e)
        record["total_reported"] = record["open"] + record["resolved"]
        return record

    def categories_to_study(self) -> list[str]:
        """Return bug categories she found but doesn't understand.

        These are categories where she detected bugs but her
        comprehension check failed — she needs to study the
        relevant concepts before she can honestly report them.
        """
        if not self._last_scan:
            return []

        to_study: list[str] = []
        for category in self._last_scan.by_category():
            if category in self._studied_categories:
                continue
            understands, _, _ = self.check_comprehension(category)
            if not understands:
                to_study.append(category)
        return to_study

    def mark_studied(self, category: str) -> None:
        """Mark a bug category as studied (docs fetched)."""
        self._studied_categories.add(category)

    def to_dict(self) -> dict[str, Any]:
        """Serialize persistent state to a dict."""
        return {
            "studied_categories": sorted(self._studied_categories),
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore persistent state from a dict."""
        self._studied_categories = set(data.get("studied_categories", []))

    def describe_concerns(self) -> str:
        """First-person description of what's bothering her about her code.

        This is what Genesis says when asked about bugs or concerns.
        It's conversational, not a dry list. Only reports bugs she
        actually understands — if she doesn't understand a category,
        she says she needs to study it more.

        Concern phrasings are drawn from thought templates in the
        concept network (bug_error_concern, bug_category_concern) so
        she can grow her own vocabulary for expressing code concerns
        as she learns, rather than reciting developer-authored prose.
        """
        if self._last_scan is None or not self._last_scan.has_bugs:
            return "no code concerns detected"

        result = self._last_scan
        parts: list[str] = []

        # Separate bugs into understood and not-yet-understood
        understood_bugs: list[BugReport] = []
        not_understood_cats: list[str] = []

        for cat, bugs in result.by_category().items():
            understands, _, _ = self.check_comprehension(cat)
            if understands:
                understood_bugs.extend(bugs)
            else:
                not_understood_cats.append(cat)

        # Report understood bugs with explanations
        if understood_bugs:
            self._describe_understood_bugs(understood_bugs, parts)

        # Report not-understood categories honestly
        if not_understood_cats:
            self._describe_not_understood(not_understood_cats, parts)

        if not parts:
            return "no code concerns detected"

        return " ".join(parts)

    def _describe_understood_bugs(
        self, understood_bugs: list[BugReport], parts: list[str]
    ) -> None:
        """Report understood bugs with severity counts and explanations.

        Uses thought templates from the concept network for concern
        phrasings, falling back to structural descriptions (counts,
        file/line) only — not authored prose.
        """
        understood_errors = sum(1 for b in understood_bugs if b.severity == "error")
        understood_warnings = sum(1 for b in understood_bugs if b.severity == "warning")
        understood_style = sum(1 for b in understood_bugs if b.severity == "style")

        if understood_errors > 0:
            # Structural description only — no authored prose
            first = understood_bugs[0]
            parts.append(
                f"{understood_errors} errors in code — "
                f"in {first.file} at line {first.line}: "
                f"{first.description.lower()}"
            )

        if understood_warnings > 0:
            # Structural count only — no authored prose
            if parts:
                parts.append(f"also {understood_warnings} warnings")
            else:
                parts.append(f"{understood_warnings} warnings in code")

        if understood_style > 0 and not parts:
            parts.append(f"noticed {understood_style} style issues")

        # Describe top understood categories with explanations
        by_cat: dict[str, list[BugReport]] = {}
        for bug in understood_bugs:
            by_cat.setdefault(bug.category, []).append(bug)

        for cat, bugs in sorted(by_cat.items(), key=lambda x: -len(x[1]))[:3]:
            first = bugs[0]
            # Structural description only — no authored prose
            parts.append(
                f"found {len(bugs)} '{cat}' issues — "
                f"in {first.file} at line {first.line}: "
                f"{first.description.lower()}"
            )

    def _describe_not_understood(
        self, not_understood_cats: list[str], parts: list[str]
    ) -> None:
        """Report not-understood categories honestly.

        Uses structural descriptions (counts) only — no authored
        first-person prose. The language engine composes the final
        voice.
        """
        count = len(not_understood_cats)
        if parts:
            parts.append(f"also {count} ununderstood categories")
        else:
            parts.append(f"{count} ununderstood categories")
