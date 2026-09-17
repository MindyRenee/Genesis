"""Tests for Genesis's bug reporter — static analysis detectors.

Covers the semantic correctness detectors that sit on top of the
original style/pattern checks:

- ``is_literal_comparison`` — ``x is 5`` / ``x is "s"`` (identity vs
  equality; a fresh literal is never the same object).
- ``type_equality_check`` — ``type(x) == T`` ignores subclasses.
- ``raise_without_from`` — ``raise`` inside ``except`` drops the cause.
- ``return_in_finally`` — ``return``/``break``/``continue`` in a
  ``finally`` block swallows in-flight exceptions.

The tests scan real files in a temp project root so they exercise the
same path the running system uses.
"""

from __future__ import annotations

import os
import tempfile
import textwrap

from genesis_cognitive.bug_reporter import BugReporter


def _scan(source: str) -> list:
    """Write ``source`` to a temp project and return the bugs found."""
    root = tempfile.mkdtemp()
    path = os.path.join(root, "sample.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(source))
    return BugReporter(project_root=root).scan(max_files=5).bugs


def _categories(source: str) -> set[str]:
    """Return the set of bug categories found in ``source``."""
    return {bug.category for bug in _scan(source)}


# ─── is_literal_comparison ────────────────────────────────────────


def test_is_literal_int_detected() -> None:
    """'x is 5' is flagged — identity is unreliable for literals."""
    assert "is_literal_comparison" in _categories(
        """
        def f(x):
            return x is 5
        """
    )


def test_is_literal_string_detected() -> None:
    """'x is "ready"' is flagged."""
    assert "is_literal_comparison" in _categories(
        """
        def f(x):
            return x is "ready"
        """
    )


def test_is_not_literal_detected() -> None:
    """'x is not 5' is flagged."""
    assert "is_literal_comparison" in _categories(
        """
        def f(x):
            return x is not 5
        """
    )


def test_is_none_not_flagged() -> None:
    """'x is None' is correct and must not be flagged."""
    assert "is_literal_comparison" not in _categories(
        """
        def f(x):
            return x is None
        """
    )


def test_is_true_not_flagged() -> None:
    """Identity against the bool singletons is a separate style choice."""
    assert "is_literal_comparison" not in _categories(
        """
        def f(x):
            return x is True
        """
    )


# ─── type_equality_check ──────────────────────────────────────────


def test_type_equality_detected() -> None:
    """'type(x) == int' is flagged — use isinstance."""
    assert "type_equality_check" in _categories(
        """
        def f(x):
            return type(x) == int
        """
    )


def test_type_is_not_detected() -> None:
    """'type(x) is not str' is flagged."""
    assert "type_equality_check" in _categories(
        """
        def f(x):
            return type(x) is not str
        """
    )


def test_type_vs_type_not_flagged() -> None:
    """'type(a) == type(b)' is legitimate and must not be flagged."""
    assert "type_equality_check" not in _categories(
        """
        def f(a, b):
            return type(a) == type(b)
        """
    )


def test_isinstance_not_flagged() -> None:
    """The correct form, isinstance, is not flagged."""
    assert "type_equality_check" not in _categories(
        """
        def f(x):
            return isinstance(x, int)
        """
    )


# ─── raise_without_from ───────────────────────────────────────────


def test_raise_without_from_detected() -> None:
    """'raise X' inside except without 'from' is flagged."""
    assert "raise_without_from" in _categories(
        """
        def f():
            try:
                pass
            except ValueError:
                raise RuntimeError("bad")
        """
    )


def test_raise_from_not_flagged() -> None:
    """'raise X from err' is correct and must not be flagged."""
    assert "raise_without_from" not in _categories(
        """
        def f():
            try:
                pass
            except ValueError as err:
                raise RuntimeError("bad") from err
        """
    )


def test_bare_reraise_not_flagged() -> None:
    """A bare 'raise' re-raises the original — no cause is needed."""
    assert "raise_without_from" not in _categories(
        """
        def f():
            try:
                pass
            except ValueError:
                raise
        """
    )


def test_raise_outside_except_not_flagged() -> None:
    """A raise with no enclosing except has no cause to chain from."""
    assert "raise_without_from" not in _categories(
        """
        def f():
            raise RuntimeError("bad")
        """
    )


# ─── return_in_finally ────────────────────────────────────────────


def test_return_in_finally_detected() -> None:
    """A return in finally is flagged as an error."""
    bugs = _scan(
        """
        def f():
            try:
                return 1
            finally:
                return 2
        """
    )
    matches = [b for b in bugs if b.category == "return_in_finally"]
    assert matches, "return in finally not detected"
    assert matches[0].severity == "error"


def test_clean_finally_not_flagged() -> None:
    """A finally block that only cleans up is fine."""
    assert "return_in_finally" not in _categories(
        """
        def f():
            try:
                pass
            finally:
                cleanup()
        """
    )


# ─── boolean_equality ─────────────────────────────────────────────


def test_boolean_equality_true_detected() -> None:
    """'x == True' is flagged (use truthiness)."""
    assert "boolean_equality" in _categories(
        """
        def f(x):
            return x == True
        """
    )


def test_boolean_equality_false_detected() -> None:
    """'x != False' is flagged."""
    assert "boolean_equality" in _categories(
        """
        def f(x):
            return x != False
        """
    )


def test_boolean_identity_not_flagged() -> None:
    """'x is True' is a different check and is not flagged here."""
    assert "boolean_equality" not in _categories(
        """
        def f(x):
            return x is True
        """
    )


def test_boolean_comparison_to_int_not_flagged() -> None:
    """'x == 1' is a legitimate value comparison."""
    assert "boolean_equality" not in _categories(
        """
        def f(x):
            return x == 1
        """
    )


# ─── assert_tuple ─────────────────────────────────────────────────


def test_assert_tuple_detected() -> None:
    """'assert (cond, msg)' is flagged — always truthy."""
    assert "assert_tuple" in _categories(
        """
        def f(x):
            assert (x > 0, "must be positive")
        """
    )


def test_assert_correct_form_not_flagged() -> None:
    """'assert cond, msg' is the correct form."""
    assert "assert_tuple" not in _categories(
        """
        def f(x):
            assert x > 0, "must be positive"
        """
    )


# ─── base_exception_catch ─────────────────────────────────────────


def test_base_exception_catch_detected() -> None:
    """'except BaseException' is flagged like a bare except."""
    assert "base_exception_catch" in _categories(
        """
        def f():
            try:
                pass
            except BaseException:
                pass
        """
    )


def test_base_exception_in_tuple_detected() -> None:
    """'except (BaseException, ValueError)' is flagged."""
    assert "base_exception_catch" in _categories(
        """
        def f():
            try:
                pass
            except (BaseException, ValueError):
                pass
        """
    )


def test_exception_catch_not_flagged() -> None:
    """'except Exception' is the recommended form."""
    assert "base_exception_catch" not in _categories(
        """
        def f():
            try:
                pass
            except Exception:
                pass
        """
    )


# ─── fstring_no_placeholder ───────────────────────────────────────


def test_fstring_without_placeholder_detected() -> None:
    """'f"hello"' is flagged — the prefix does nothing."""
    assert "fstring_no_placeholder" in _categories(
        """
        def f():
            return f"hello world"
        """
    )


def test_fstring_with_placeholder_not_flagged() -> None:
    """A real f-string is not flagged."""
    assert "fstring_no_placeholder" not in _categories(
        """
        def f(name):
            return f"hello {name}"
        """
    )


def test_fstring_literal_braces_not_flagged() -> None:
    """'f"{{}}"' uses literal braces — dropping the f would change output."""
    assert "fstring_no_placeholder" not in _categories(
        """
        def f():
            return f"{{}}"
        """
    )


# ─── no false positives on clean code ─────────────────────────────


def test_clean_code_has_no_new_categories() -> None:
    """Well-written code triggers none of the new detectors."""
    new_categories = {
        "is_literal_comparison",
        "type_equality_check",
        "raise_without_from",
        "return_in_finally",
        "boolean_equality",
        "assert_tuple",
        "base_exception_catch",
        "fstring_no_placeholder",
    }
    found = _categories(
        """
        def classify(value):
            if value is None:
                return "none"
            if isinstance(value, (int, float)):
                return "number"
            return "other"

        def parse(text):
            try:
                return int(text)
            except ValueError as err:
                raise ValueError(f"not a number: {text}") from err
            finally:
                log_attempt(text)
        """
    )
    assert not (found & new_categories), found & new_categories
