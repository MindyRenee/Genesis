"""Tool use — the interface between Genesis's mind and the world.

A tool is a callable capability that Genesis can decide to invoke.
Tools are deliberately narrow, inspectable, and safe. They are the
non-LLM analogue of function calling: the mind forms an intention,
looks at what it can do, and dispatches a concrete action.
"""

from __future__ import annotations

import os
import pathlib
import py_compile
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Global, lazy singleton so any subsystem can call a tool whenever it needs one.
# The registry is stateless (pure functions), so a single instance is fine.
_GLOBAL_TOOLS: ToolRegistry | None = None


def get_tools() -> ToolRegistry:
    """Return the shared tool registry."""
    global _GLOBAL_TOOLS
    if _GLOBAL_TOOLS is None:
        _GLOBAL_TOOLS = ToolRegistry()
    return _GLOBAL_TOOLS


@dataclass
class ToolResult:
    """The outcome of a tool invocation."""

    success: bool
    output: str = ""
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class Tool:
    """A single tool Genesis can use."""

    def __init__(self, name: str, description: str, run: Callable[..., ToolResult]) -> None:
        """Store the tool's name, description, and callable."""
        self.name = name
        self.description = description
        self._run = run

    def run(self, **kwargs: Any) -> ToolResult:
        """Invoke the tool's callable with the given keyword arguments."""
        return self._run(**kwargs)


class ToolRegistry:
    """Holds the tools available to Genesis."""

    def __init__(self) -> None:
        """Initialize an empty tool registry and register built-in tools."""
        self._tools: dict[str, Tool] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Register the built-in tools that ship with Genesis."""
        self.register(
            Tool(
                "compile_python",
                "Compile a Python file to check for syntax errors.",
                compile_python_file,
            )
        )
        self.register(
            Tool(
                "run_pytest",
                "Run pytest on a test file or directory and return the result.",
                run_pytest,
            )
        )
        self.register(
            Tool(
                "read_file",
                "Read the contents of a file for inspection.",
                read_file,
            )
        )
        self.register(
            Tool(
                "write_file",
                "Write content to a file, creating it and parent dirs if needed.",
                write_file,
            )
        )
        self.register(
            Tool(
                "make_dir",
                "Create a directory, including parents.",
                make_dir,
            )
        )
        self.register(
            Tool(
                "list_dir",
                "List the contents of a directory.",
                list_dir,
            )
        )
        self.register(
            Tool(
                "delete_file",
                "Delete a file (refuses to delete directories).",
                delete_file,
            )
        )
        self.register(
            Tool(
                "move_file",
                "Move or rename a file within the project root.",
                move_file,
            )
        )
        self.register(
            Tool(
                "run_shell",
                "Run a shell command sandboxed to the project root. "
                "Blocked: sudo, rm -rf /, dd, mkfs, curl, wget, ssh. "
                "Timeout: 30s default, 120s max.",
                run_shell,
            )
        )
        self.register(
            Tool(
                "concept_lookup",
                "Look up a concept in the knowledge network and describe it.",
                concept_lookup,
            )
        )
        self.register(
            Tool(
                "web_search",
                "Search the web for a query and return result URLs. "
                "Read-only. Adult/malware content is blocked. "
                "Use this when it doesn't know something and wants "
                "to look it up — in conversation, for fun, or to learn.",
                web_search,
            )
        )
        self.register(
            Tool(
                "web_fetch",
                "Fetch a web page and extract its readable text. "
                "Read-only GET. Adult/malware URLs are refused. "
                "Use after web_search to read a specific result page.",
                web_fetch,
            )
        )

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry, keyed by its name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool with the given name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        """Return a list of dicts with each tool's name and description."""
        return [
            {"name": t.name, "description": t.description} for t in self._tools.values()
        ]

    def run(self, name: str, **kwargs: Any) -> ToolResult:
        """Look up and invoke a named tool, returning an error result if unknown."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False, error=f"Unknown tool: {name}"
            )
        return tool.run(**kwargs)


def _safe_path(path: str, project_root: str) -> pathlib.Path | None:
    """Resolve `path` under `project_root`, rejecting escapes.

    Relative paths are resolved against `project_root`. Absolute paths
    are allowed if they still resolve to a location inside
    `project_root`. Symlinks that escape the root are rejected.
    """
    root = pathlib.Path(project_root).resolve()
    p = pathlib.Path(path)
    resolved = (root / p).resolve()
    try:
        if not resolved.is_relative_to(root):
            return None
    except ValueError:
        return None
    return resolved


def _safe_path_writable(path: str, project_root: str) -> pathlib.Path | None:
    """Like _safe_path, but works for paths that don't exist yet.

    _safe_path calls .resolve() which follows symlinks, but for a
    path that doesn't exist yet, resolve() returns the path as-is.
    We resolve the parent directory and append the filename, checking
    the parent stays inside root.
    """
    root = pathlib.Path(project_root).resolve()
    p = pathlib.Path(path)
    parent = (root / p.parent).resolve()
    full = parent / p.name
    try:
        if not parent.is_relative_to(root):
            return None
    except ValueError:
        return None
    return full


def compile_python_file(path: str, project_root: str = ".") -> ToolResult:
    """Compile a Python source file under `project_root` without executing it."""
    safe = _safe_path(path, project_root)
    if safe is None:
        return ToolResult(success=False, error=f"Path is outside project root: {path}")
    if not safe.is_file():
        return ToolResult(success=False, error=f"File not found: {path}")
    try:
        py_compile.compile(str(safe), doraise=True)
        return ToolResult(success=True, output=f"Compiled {path} successfully.")
    except py_compile.PyCompileError as e:
        return ToolResult(success=False, error=str(e))


def run_pytest(target: str, project_root: str = ".") -> ToolResult:
    """Run pytest on a file or directory under `project_root`."""
    safe = _safe_path(target, project_root)
    if safe is None:
        return ToolResult(success=False, error=f"Target is outside project root: {target}")
    root = pathlib.Path(project_root).resolve()
    env = os.environ.copy()
    # If the repo has a 'python' package dir at the root, include it so
    # imports resolve even when pytest is invoked from the repo root.
    # Check relative to project_root (not CWD) so the tool works
    # correctly regardless of the caller's working directory.
    if (root / "python").is_dir():
        pythonpath = env.get("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}python"
        else:
            env["PYTHONPATH"] = "python"
    else:
        pythonpath = env.get("PYTHONPATH", "")
        if pythonpath:
            env["PYTHONPATH"] = f"{pythonpath}{os.pathsep}{project_root}"
        else:
            env["PYTHONPATH"] = project_root
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", str(safe), "-q"],
            cwd=project_root,
            capture_output=True,
            text=True,
            # The full python/tests suite takes ~5 minutes (it
            # includes the end-to-end daemon test). A 120s timeout
            # here could never complete on a healthy suite, so every
            # gated change was falsely reverted. 600s gives ~2x
            # headroom over the measured healthy runtime.
            timeout=600,
            env=env,
        )
        return ToolResult(
            success=result.returncode == 0,
            output=result.stdout,
            error=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error="pytest timed out")
    except FileNotFoundError:
        return ToolResult(success=False, error="python3 or pytest not found")


def concept_lookup(concept: str, network: Any) -> ToolResult:
    """Look up a concept in the knowledge network and summarise it."""
    if network is None:
        return ToolResult(success=False, error="No concept network provided")
    node = network.get_concept(concept)
    if node is None:
        matches = network.search_concepts(concept, limit=1)
        if not matches:
            return ToolResult(success=False, error=f"Concept '{concept}' not found")
        concept = matches[0]
        node = network.get_concept(concept)
    if node is None:
        return ToolResult(success=False, error=f"Concept '{concept}' not found")

    parts: list[str] = []
    definition = node.properties.get("definition", "")
    has_definition = bool(definition) and definition != "NO DEF"
    if has_definition:
        parts.append(f"{concept} is defined as {definition}")

    outgoing = network.get_edges(concept, direction="out")
    for edge in outgoing[:3]:
        relation = edge.relation.value.replace("_", " ")
        parts.append(f"{concept} {relation} {edge.target}")

    if not parts:
        return ToolResult(
            success=True,
            output="",
            data={"concept": concept, "edge_facts": []},
        )
    return ToolResult(
        success=True,
        output=". ".join(parts),
        data={
            "concept": concept,
            "definition": definition if has_definition else "",
            "edges": len(outgoing),
            # Structured (relation, target, weight) triples — lets the
            # question handler compose the answer through the language
            # engine's knowledge path instead of reciting `output`.
            "edge_facts": [
                (edge.relation.value, edge.target, edge.weight)
                for edge in outgoing[:5]
            ],
        },
    )


def read_file(
    path: str, offset: int = 0, limit: int = 200, project_root: str = "."
) -> ToolResult:
    """Read part of a text file under `project_root`."""
    safe = _safe_path(path, project_root)
    if safe is None:
        return ToolResult(success=False, error=f"Path is outside project root: {path}")
    if not safe.is_file():
        return ToolResult(success=False, error=f"File not found: {path}")
    try:
        with open(safe) as f:
            text = f.read()
        start = offset
        end = offset + limit if limit > 0 else len(text)
        snippet = text[start:end]
        return ToolResult(
            success=True,
            output=snippet,
            data={"total_chars": len(text)},
        )
    except OSError as e:
        return ToolResult(success=False, error=str(e))


# ─── File operation tools (for project creation) ─────────────────────


def write_file(
    path: str, content: str, project_root: str = "."
) -> ToolResult:
    """Write content to a file under `project_root`, creating it if needed.

    Creates parent directories if they don't exist. Overwrites the
    file if it already exists.
    """
    safe = _safe_path_writable(path, project_root)
    if safe is None:
        return ToolResult(success=False, error=f"Path is outside project root: {path}")
    try:
        safe.parent.mkdir(parents=True, exist_ok=True)
        safe.write_text(content, encoding="utf-8")
        return ToolResult(
            success=True,
            output=f"Wrote {len(content)} chars to {path}",
            data={"path": str(safe), "chars": len(content)},
        )
    except OSError as e:
        return ToolResult(success=False, error=str(e))


def make_dir(path: str, project_root: str = ".") -> ToolResult:
    """Create a directory under `project_root`, including parents."""
    safe = _safe_path_writable(path, project_root)
    if safe is None:
        return ToolResult(success=False, error=f"Path is outside project root: {path}")
    try:
        safe.mkdir(parents=True, exist_ok=True)
        return ToolResult(
            success=True,
            output=f"Directory ready: {path}",
            data={"path": str(safe)},
        )
    except OSError as e:
        return ToolResult(success=False, error=str(e))


def list_dir(path: str = ".", project_root: str = ".") -> ToolResult:
    """List the contents of a directory under `project_root`."""
    safe = _safe_path(path, project_root)
    if safe is None:
        return ToolResult(success=False, error=f"Path is outside project root: {path}")
    if not safe.is_dir():
        return ToolResult(success=False, error=f"Not a directory: {path}")
    try:
        entries: list[str] = []
        for child in sorted(safe.iterdir()):
            prefix = "d " if child.is_dir() else "f "
            entries.append(f"{prefix}{child.name}")
        return ToolResult(
            success=True,
            output="\n".join(entries) if entries else "(empty)",
            data={"path": str(safe), "count": len(entries)},
        )
    except OSError as e:
        return ToolResult(success=False, error=str(e))


def delete_file(path: str, project_root: str = ".") -> ToolResult:
    """Delete a file under `project_root`. Refuses to delete directories."""
    safe = _safe_path(path, project_root)
    if safe is None:
        return ToolResult(success=False, error=f"Path is outside project root: {path}")
    if not safe.exists():
        return ToolResult(success=False, error=f"File not found: {path}")
    if safe.is_dir():
        return ToolResult(success=False, error=f"Refusing to delete directory: {path}")
    try:
        safe.unlink()
        return ToolResult(success=True, output=f"Deleted {path}")
    except OSError as e:
        return ToolResult(success=False, error=str(e))


def move_file(
    src: str, dst: str, project_root: str = "."
) -> ToolResult:
    """Move or rename a file within `project_root`."""
    safe_src = _safe_path(src, project_root)
    safe_dst = _safe_path_writable(dst, project_root)
    if safe_src is None:
        return ToolResult(success=False, error=f"Source outside project root: {src}")
    if safe_dst is None:
        return ToolResult(success=False, error=f"Destination outside project root: {dst}")
    if not safe_src.exists():
        return ToolResult(success=False, error=f"Source not found: {src}")
    try:
        safe_dst.parent.mkdir(parents=True, exist_ok=True)
        safe_src.replace(safe_dst)
        return ToolResult(success=True, output=f"Moved {src} to {dst}")
    except OSError as e:
        return ToolResult(success=False, error=str(e))


# ─── Shell tool (sandboxed) ───────────────────────────────────────────


# Commands that are always blocked — too dangerous for autonomous use.
# Patterns are matched against the command with whitespace normalized
# (runs of whitespace collapsed to single spaces) so that variations
# like "sudo\\trm" or "rm  -rf" can't bypass the filter.
_BLOCKED_PATTERNS = (
    "rm -rf /",
    "rm -rf ~",
    "rm -rf /*",
    "sudo ",
    "dd if=",
    "mkfs",
    "fdisk",
    "shred ",
    ":(){ :|:& };:",  # fork bomb (with spaces — the real form)
    "chmod 777",
    "curl ",
    "wget ",
    "nc ",
    "ssh ",
    "scp ",
    "find / -delete",
    "find / -exec",
)

# Max timeout for shell commands (seconds). Long enough for pip install
# or pytest, short enough to prevent runaway processes.
_MAX_SHELL_TIMEOUT = 120


def run_shell(
    command: str,
    project_root: str = ".",
    timeout: int = 30,
) -> ToolResult:
    """Run a shell command sandboxed to `project_root`.

    The command runs with `cwd=project_root` so it can only affect
    files in that directory (assuming the command itself doesn't
    escape via absolute paths — but destructive absolute-path
    commands are blocked below).

    Safety guards:
    - Blocked patterns: ``rm -rf /``, ``sudo``, ``dd``, ``mkfs``,
      ``shred``, fork bombs, network tools (``curl``, ``wget``,
      ``nc``, ``ssh``, ``scp``), and ``chmod 777``.
    - Timeout: default 30s, max 120s. Prevents runaway processes.
    - No environment inheritance of secrets: only PATH, HOME, and
      PYTHONPATH are passed through.

    Args:
        command: The shell command to execute.
        project_root: The working directory for the command.
        timeout: Maximum execution time in seconds (capped at 120).
    """
    if not command or not command.strip():
        return ToolResult(success=False, error="Empty command")

    # Normalize whitespace so patterns can't be bypassed with tabs,
    # newlines, or extra spaces (e.g. "sudo\trm" → "sudo rm",
    # "rm  -rf /" → "rm -rf /").
    normalized = re.sub(r"\s+", " ", command).strip()
    lower = normalized.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern in lower:
            return ToolResult(
                success=False,
                error=f"Blocked pattern detected: {pattern.strip()}",
            )

    # Clamp timeout
    timeout = max(1, min(timeout, _MAX_SHELL_TIMEOUT))

    # Resolve the project root
    root = pathlib.Path(project_root).resolve()
    if not root.is_dir():
        return ToolResult(success=False, error=f"Project root not found: {project_root}")

    # Minimal environment — no secret leakage
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    # Pass PYTHONPATH if set (for test runners)
    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = pythonpath

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return ToolResult(
            success=result.returncode == 0,
            output=result.stdout,
            error=result.stderr,
            data={"returncode": result.returncode},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error=f"Command timed out after {timeout}s")
    except FileNotFoundError:
        return ToolResult(success=False, error="Shell not available")
    except OSError as e:
        return ToolResult(success=False, error=str(e))


# ─── Web search tools ───────────────────────────────────────────────


def web_search(query: str, limit: int = 5) -> ToolResult:
    """Search the web (DuckDuckGo) for a query.

    Returns up to ``limit`` result URLs, filtered through the
    adult/malware content blocklist. Read-only — no data is sent
    beyond the search query itself.

    The returned ``data["results"]`` is a list of dicts with ``url``
    and ``title`` keys. The ``output`` is a human-readable summary
    for tool-use contexts; Genesis composes its own words from the
    structured data, not by reciting this string.
    """
    from .web_search import search as _search

    results = _search(query, limit=limit)
    if not results:
        return ToolResult(
            success=False,
            error="No results found (or query was filtered as adult content).",
        )
    return ToolResult(
        success=True,
        output=f"Found {len(results)} results for '{query}'.",
        data={
            "query": query,
            "results": [
                {"url": r.url, "title": r.title, "snippet": r.snippet}
                for r in results
            ],
        },
    )


def web_fetch(url: str) -> ToolResult:
    """Fetch a web page and extract its readable text.

    Read-only GET request. Refuses URLs on the adult/malware
    blocklist. Only processes text/html and text/plain responses.

    The returned ``data`` contains ``url``, ``title``, ``content``
    (the extracted text), and ``related_links`` (outbound links found
    on the page, for curiosity chaining). The ``output`` is a short
    summary; Genesis learns from ``content`` and composes its own
    understanding — it does not recite the raw page text.
    """
    from .web_search import fetch as _fetch

    page = _fetch(url)
    if page is None:
        return ToolResult(
            success=False,
            error="Fetch failed (URL blocked, offline, or content too short).",
        )
    return ToolResult(
        success=True,
        output=f"Fetched '{page.title}' ({len(page.content)} chars).",
        data={
            "url": page.url,
            "title": page.title,
            "content": page.content,
            "related_links": page.related_links,
        },
    )
