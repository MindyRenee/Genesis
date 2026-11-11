"""Lint & format — the quality bench.

`lint_python` runs ruff and pyflakes; `format_python` runs ruff
format. They discover the tools the way the environment actually is:
the dev venv (~/.local/lib/genesis/venv/bin) first, then PATH —
never a hardcoded interpreter. `syntax_check` is the polyglot gate:
compile-only validation for python, c, c++, rust, java, bash.

Docs: https://docs.astral.sh/ruff/ ·
https://pypi.org/project/pyflakes/ ·
https://doc.rust-lang.org/rustc/
"""

from __future__ import annotations

import pathlib
import py_compile
import shutil
import subprocess
import tempfile
from typing import Any

from tools.framework import ToolResult


def _find_tool(name: str) -> str | None:
    """Discover a dev tool: project venv first, then PATH."""
    venv = pathlib.Path.home() / ".local/lib/genesis/venv/bin" / name
    if venv.is_file():
        return str(venv)
    return shutil.which(name)


def lint_python(path: str, project_root: str = ".") -> ToolResult:
    """Ruff + pyflakes over a file or directory.

    Returns each linter's own diagnostics verbatim in ``data``; a
    missing linter is reported in the output, not silently skipped.
    https://docs.astral.sh/ruff/linter/
    """
    root = pathlib.Path(project_root).resolve()
    target = (root / pathlib.Path(path)).resolve()
    if not target.exists():
        return ToolResult(success=False, error=f"not found: {path}")
    findings: list[str] = []
    reports: dict[str, str] = {}
    for name, args in (
        ("ruff", ["check", "--output-format=concise"]),
        ("pyflakes", []),
    ):
        exe = _find_tool(name)
        if exe is None:
            findings.append(f"{name}: not installed")
            continue
        try:
            r = subprocess.run(
                [exe, *args, str(target)],
                capture_output=True,
                text=True,
                timeout=120,
                stdin=subprocess.DEVNULL,
            )
            reports[name] = (r.stdout + r.stderr).strip()
            if reports[name]:
                findings.append(f"── {name} ──\n{reports[name]}")
        except (OSError, subprocess.TimeoutExpired) as e:
            findings.append(f"{name}: failed to run ({e})")
    issues = any(v for v in reports.values())
    return ToolResult(
        success=not issues,
        output="\n".join(findings) if findings else "clean",
        data={"reports": reports},
    )


def format_python(path: str, project_root: str = ".", check: bool = False) -> ToolResult:
    """Ruff format — rewrite a file/dir canonically (check=True just reports).

    https://docs.astral.sh/ruff/formatter/
    """
    exe = _find_tool("ruff")
    if exe is None:
        return ToolResult(success=False, error="ruff not installed")
    root = pathlib.Path(project_root).resolve()
    target = (root / pathlib.Path(path)).resolve()
    if not target.exists():
        return ToolResult(success=False, error=f"not found: {path}")
    args = [exe, "format"] + (["--check"] if check else []) + [str(target)]
    try:
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return ToolResult(success=False, error=str(e))
    return ToolResult(
        success=r.returncode == 0,
        output=r.stdout or "(formatted)",
        error=r.stderr,
        data={"returncode": r.returncode, "check_only": check},
    )


_SYNTAX: dict[str, Any] = {
    "python": lambda s: _check_python(s),
    "bash": lambda s: (["bash", "-n", str(s)],),
    "c": lambda s: (["gcc", "-fsyntax-only", str(s)],),
    "cpp": lambda s: (["g++", "-fsyntax-only", str(s)],),
    "rust": lambda s: (
        ["rustc", "--edition", "2021", "--emit", "metadata", "-o", "/dev/null", str(s)],
    ),
    "java": lambda s: _check_java(s),
    "javascript": lambda s: (["node", "--check", str(s)],),
}


def _check_python(src: pathlib.Path) -> tuple[None, str]:
    try:
        py_compile.compile(str(src), doraise=True)
        return (None, "")
    except py_compile.PyCompileError as e:
        return (None, str(e))


def _check_java(src: pathlib.Path) -> tuple[list[str], pathlib.Path]:
    outdir = pathlib.Path(tempfile.mkdtemp(prefix="genesis_javac_"))
    return (["javac", "-d", str(outdir), "-proc:none", str(src)], outdir)


def syntax_check(path: str, language: str = "", project_root: str = ".") -> ToolResult:
    """Compile-only validation — proves it parses, runs nothing.

    ``language`` is inferred from the extension when empty (py, sh,
    c, cpp/cc/cxx, rs, java, js). Absent toolchains report absence.
    """
    root = pathlib.Path(project_root).resolve()
    src = (root / pathlib.Path(path)).resolve()
    if not src.is_file():
        return ToolResult(success=False, error=f"not found: {path}")
    lang = language.strip().lower() or {
        ".py": "python",
        ".sh": "bash",
        ".c": "c",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".rs": "rust",
        ".java": "java",
        ".js": "javascript",
    }.get(src.suffix.lower(), "")
    checker = _SYNTAX.get(lang)
    if checker is None:
        return ToolResult(
            success=False,
            error=f"no checker for {lang or src.suffix!r}; known: {', '.join(sorted(_SYNTAX))}",
        )
    probe = {
        "bash": "bash",
        "c": "gcc",
        "cpp": "g++",
        "rust": "rustc",
        "java": "javac",
        "javascript": "node",
    }.get(lang)
    if probe and shutil.which(probe) is None:
        return ToolResult(success=False, error=f"{probe} not installed")
    try:
        spec = checker(src)
        if lang == "python":
            _, err = spec  # type: ignore[misc]
            if err:
                return ToolResult(success=False, error=err)
            return ToolResult(success=True, output=f"{path}: syntax ok")
        cmd, outdir = spec if lang == "java" else (spec[0], None)  # type: ignore[misc]
        r = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=120,
            stdin=subprocess.DEVNULL,
        )
        if outdir is not None:
            shutil.rmtree(outdir, ignore_errors=True)
        return ToolResult(
            success=r.returncode == 0,
            output=f"{path}: syntax ok" if r.returncode == 0 else r.stdout,
            error=r.stderr,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return ToolResult(success=False, error=str(e))


LINT_TOOLS = (
    (
        "lint_python",
        "Run ruff and pyflakes over a Python file or directory — "
        "verbatim diagnostics, each linter's presence reported. "
        "Docs: docs.astral.sh/ruff/linter",
        lint_python,
    ),
    (
        "format_python",
        "Canonical formatting via ruff format — rewrites the file/dir; "
        "check=True reports without writing. Docs: docs.astral.sh/ruff/formatter",
        format_python,
    ),
    (
        "syntax_check",
        "Polyglot compile-only check: python, bash, c, cpp, rust, "
        "java, javascript (inferred from extension or language=). "
        "Nothing executes — parse and typecheck only.",
        syntax_check,
    ),
)
