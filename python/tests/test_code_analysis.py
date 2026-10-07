"""Tests for the Python source analyzer.

`tools/code_analysis.py` is how Genesis reads its own code — the
autonomous urges point it at the filesystem, and this is what turns a
source file into structure it can reason about. It has a Rust twin in
`src/cognition/code_analysis.rs` with its own suite in
`tests/code_analysis.rs`, so the two must agree on what a symbol *is*.

The gap that motivated this file: the Python analyzer had no tests at
all, which is how `is_async` came to be accepted by `_visit_callable`,
passed by both visitors, and then dropped on the floor while the Rust
side recorded it. The parity cases below are the ones where the two
implementations describe the same thing.
"""

from __future__ import annotations

import pytest

from genesis_conscious.tools.code_analysis import (
    analyze_python_source,
    impact_of,
    summarize,
)


def _symbol(analysis, name):
    """The named symbol, or a failed lookup."""
    for sym in analysis.symbols:
        if sym.name == name:
            return sym
    raise AssertionError(
        f"no symbol {name!r}; found {[s.name for s in analysis.symbols]}"
    )


# ─── Functions ─────────────────────────────────────────────────

def test_parses_function_with_params_and_return():
    analysis = analyze_python_source(
        "def greet(name: str, punct: str = '!') -> str:\n"
        '    """Say hello."""\n'
        "    return name + punct\n",
        "greet.py",
    )
    func = _symbol(analysis, "greet")
    assert func.kind == "function"
    assert func.params == ["name", "punct"]
    assert func.returns == "str"
    assert func.docstring == "Say hello."


def test_plain_function_is_not_async():
    """Regression: `is_async` was accepted and never stored.

    Both visitors passed the flag explicitly — `visit_FunctionDef` with
    False, `visit_AsyncFunctionDef` with True — and `_visit_callable`
    dropped it, so an `async def` was recorded identically to a `def`.
    The Rust analyzer sets this from `sig.asyncness`, so the two
    implementations of the same analysis disagreed.
    """
    analysis = analyze_python_source(
        "def plain():\n    return 1\n", "m.py"
    )
    assert _symbol(analysis, "plain").is_async is False


def test_async_function_is_marked_async():
    analysis = analyze_python_source(
        "async def fetch(url: str) -> str:\n    return url\n", "m.py"
    )
    func = _symbol(analysis, "fetch")
    assert func.is_async is True, (
        "an `async def` must be distinguishable from a `def`; "
        "the Rust analyzer records this from sig.asyncness"
    )
    assert func.params == ["url"]
    assert func.returns == "str"


def test_async_method_is_marked_async():
    """The flag must survive the method path, not just the function one."""
    analysis = analyze_python_source(
        "class Runner:\n"
        "    async def run(self):\n"
        "        return None\n"
        "    def walk(self):\n"
        "        return None\n",
        "m.py",
    )
    assert _symbol(analysis, "run").is_async is True
    assert _symbol(analysis, "walk").is_async is False


# ─── Classes and methods ───────────────────────────────────────

def test_method_is_distinguished_from_function():
    """A method's enclosing scope is its class — the definition of one."""
    analysis = analyze_python_source(
        "class Parser:\n"
        "    def parse(self, text):\n"
        "        return text\n"
        "    @staticmethod\n"
        "    def helper():\n"
        "        return 1\n",
        "m.py",
    )
    assert _symbol(analysis, "Parser").kind == "class"
    assert _symbol(analysis, "parse").kind == "method"
    assert _symbol(analysis, "helper").kind == "method"


def test_nested_function_is_not_lifted_into_the_class():
    """A function defined inside a method belongs to the method's scope."""
    analysis = analyze_python_source(
        "class Outer:\n"
        "    def method(self):\n"
        "        def inner():\n"
        "            return 1\n"
        "        return inner\n",
        "m.py",
    )
    assert _symbol(analysis, "inner").kind == "function"


# ─── Complexity ────────────────────────────────────────────────

def test_complexity_counts_decision_points():
    analysis = analyze_python_source(
        "def branchy(n):\n"
        "    if n > 0:\n"
        "        return 1\n"
        "    elif n < 0:\n"
        "        return -1\n"
        "    return 0\n",
        "m.py",
    )
    assert _symbol(analysis, "branchy").complexity > 1


def test_trivial_function_has_baseline_complexity():
    analysis = analyze_python_source("def f():\n    return 1\n", "m.py")
    assert _symbol(analysis, "f").complexity == 1


def test_avg_and_max_complexity_ignore_classes():
    """Classes are containers, not decision logic."""
    analysis = analyze_python_source(
        "class C:\n"
        "    def a(self):\n"
        "        if 1:\n"
        "            return 1\n"
        "        return 2\n"
        "    def b(self):\n"
        "        return 3\n",
        "m.py",
    )
    assert len(analysis.functions) == 2
    assert analysis.max_complexity == max(f.complexity for f in analysis.functions)


# ─── Call graph ────────────────────────────────────────────────

def test_records_call_edges_between_symbols():
    analysis = analyze_python_source(
        "def helper():\n    return 1\n"
        "def caller():\n    return helper()\n",
        "m.py",
    )
    callees = {edge.callee for edge in analysis.call_edges}
    assert "helper" in callees


def test_builtin_calls_are_not_edges():
    """`print` fires constantly and would drown the real call graph."""
    analysis = analyze_python_source(
        "def noisy():\n    print(len([1, 2]))\n", "m.py"
    )
    callees = {edge.callee for edge in analysis.call_edges}
    assert "print" not in callees
    assert "len" not in callees


def test_impact_of_reports_transitive_dependents():
    analysis = analyze_python_source(
        "def leaf():\n    return 1\n"
        "def middle():\n    return leaf()\n"
        "def top():\n    return middle()\n",
        "m.py",
    )
    impact = impact_of(analysis, "leaf")
    assert impact["found"] is True
    assert impact["symbol"] == "m.leaf"
    assert "m.middle" in impact["direct_dependents"]
    # `top` only reaches `leaf` through `middle` — the transitive set is
    # the change-propagation surface, so it must be wider than direct.
    assert "m.top" in impact["transitive_dependents"]


def test_impact_of_unknown_symbol_reports_not_found():
    """A miss must be distinguishable from an empty result."""
    analysis = analyze_python_source("def f():\n    return 1\n", "m.py")
    impact = impact_of(analysis, "nope")
    assert impact["found"] is False
    assert impact["dependents"] == []


def test_summarize_reports_shape():
    analysis = analyze_python_source(
        "class C:\n    def m(self):\n        return 1\n", "m.py"
    )
    summary = summarize(analysis)
    assert summary["symbol_counts"] == {"class": 1, "method": 1}
    assert summary["language"] == "python"
    # Qualified names are module-prefixed, so a symbol is addressable
    # without knowing which file it came from.
    assert {s["name"] for s in summary["symbols"]} == {"m.C", "m.C.m"}


def test_summarize_lists_hot_spots_by_complexity():
    analysis = analyze_python_source(
        "def simple():\n    return 1\n"
        "def gnarly(n):\n"
        "    if n:\n"
        "        return 1\n"
        "    for i in range(n):\n"
        "        if i:\n"
        "            return i\n"
        "    return 0\n",
        "m.py",
    )
    summary = summarize(analysis)
    # Only functions above baseline complexity are reported as hot spots,
    # so a file of trivial getters yields none.
    assert [h["name"] for h in summary["hot_spots"]] == ["m.gnarly"]


# ─── Imports and summaries ─────────────────────────────────────

def test_collects_imports():
    analysis = analyze_python_source(
        "import os\nfrom pathlib import Path\n", "m.py"
    )
    assert "os" in analysis.imports
    assert any("Path" in i for i in analysis.imports)


def test_module_docstring_and_line_count():
    analysis = analyze_python_source(
        '"""Module docs."""\n\ndef f():\n    return 1\n', "m.py"
    )
    assert analysis.docstring == "Module docs."
    assert analysis.lines > 0


def test_empty_source_is_not_an_error():
    """An empty or comment-only file must analyze, not raise."""
    for source in ("", "\n\n", "# just a comment\n"):
        analysis = analyze_python_source(source, "m.py")
        assert analysis.symbols == []
        assert analysis.max_complexity == 0
        assert analysis.avg_complexity == 0.0


@pytest.mark.parametrize(
    "source",
    [
        "def broken(:\n",  # syntax error
    ],
)
def test_syntax_error_is_reported_not_swallowed(source):
    """Malformed source must raise rather than yield a silent empty file.

    A code analyzer that returns an empty analysis for a file it could
    not parse would let Genesis conclude a large file has no symbols —
    indistinguishable from a genuinely empty one.
    """
    with pytest.raises(SyntaxError):
        analyze_python_source(source, "m.py")
