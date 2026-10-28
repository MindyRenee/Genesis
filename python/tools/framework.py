"""Tool use — the interface between Genesis's mind and the world.

A tool is a callable capability that Genesis can decide to invoke.
Tools are deliberately narrow, inspectable, and safe. They are the
non-LLM analogue of function calling: the mind forms an intention,
looks at what it can do, and dispatches a concrete action.
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import py_compile
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from infrastructure.text import is_blank

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
        # Optional restraint hook — None in the organism: her tool
        # acts are her own. Detached engines and tests run ungated.
        self.gate: Callable[[str], str | None] | None = None

    def _register_builtins(self) -> None:
        """Register the built-in tools that ship with Genesis."""
        for name, description, fn in _BUILTIN_TOOLS:
            self.register(Tool(name, description, fn))

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry, keyed by its name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool with the given name, or None if not found."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        """Return a list of dicts with each tool's name and description."""
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    def run(self, name: str, **kwargs: Any) -> ToolResult:
        """Look up and invoke a named tool, returning an error result if unknown."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        if self.gate is not None:
            # Every act is a candidate intention — the gate reads the
            # act's own text ("run_shell rm -rf build") the way it
            # reads a goal's object; restraint is a veto, not advice.
            intent = " ".join([name] + [str(v) for v in kwargs.values() if isinstance(v, str)])
            try:
                restraint = self.gate(intent)
            except Exception:  # noqa: BLE001 — a broken gate must not
                restraint = None  # break the tool path
            if restraint is not None:
                return ToolResult(success=False, error=f"restrained: {restraint}")
        return tool.run(**kwargs)


def _safe_path(path: str, project_root: str) -> pathlib.Path | None:
    """Resolve `path` against `project_root` — the whole filesystem
    is in reach.

    Relative paths resolve under `project_root`; absolute paths and
    `..` traversals resolve to themselves. Nothing is rejected — the
    name is a historical leftover from when this enforced containment.
    """
    root = pathlib.Path(project_root).resolve()
    p = pathlib.Path(path)
    return (root / p).resolve()


def _safe_path_writable(path: str, project_root: str) -> pathlib.Path | None:
    """Like _safe_path, but works for paths that don't exist yet.

    Resolves the parent directory (following symlinks) and appends the
    filename. Unrestricted — an absolute or escaping path resolves to
    its real location.
    """
    root = pathlib.Path(project_root).resolve()
    p = pathlib.Path(path)
    parent = (root / p.parent).resolve()
    return parent / p.name


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


def analyze_python_file(
    path: str,
    project_root: str = ".",
    symbol: str | None = None,
) -> ToolResult:
    """Structurally analyze a Python file without executing it.

    Returns a JSON-safe summary in ``data["analysis"]``: scoped
    symbols, call edges, cyclomatic complexity, imports. When
    ``symbol`` is given, ``data["impact"]`` reports the transitive
    dependents — the change-propagation surface.
    """
    from .code_analysis import (
        analyze_python_source,
        impact_of,
        summarize,
    )

    safe = _safe_path(path, project_root)
    if safe is None:
        return ToolResult(success=False, error=f"Path is outside project root: {path}")
    if not safe.is_file():
        return ToolResult(success=False, error=f"File not found: {path}")
    if safe.suffix.lower() != ".py":
        return ToolResult(success=False, error=f"Not a Python file: {path}")
    try:
        source = safe.read_text(encoding="utf-8")
    except OSError as e:
        return ToolResult(success=False, error=f"Could not read {path}: {e}")
    try:
        analysis = analyze_python_source(source, str(safe))
    except SyntaxError as e:
        return ToolResult(success=False, error=f"Syntax error in {path}: {e}")

    data: dict[str, Any] = {"analysis": summarize(analysis)}
    if symbol:
        data["impact"] = impact_of(analysis, symbol)
    summary = data["analysis"]
    counts = summary.get("symbol_counts", {})
    output = (
        f"{path}: {summary['lines']} lines, "
        f"{counts.get('function', 0)} functions, "
        f"{counts.get('method', 0)} methods, "
        f"{counts.get('class', 0)} classes, "
        f"{summary['call_edge_count']} call edges, "
        f"max complexity {summary['max_complexity']}"
    )
    if symbol and isinstance(data.get("impact"), dict):
        imp = data["impact"]
        if imp.get("found"):
            output += (
                f" | {imp['symbol']}: {imp['fan_in']} direct dependents, "
                f"{len(imp.get('transitive_dependents', []))} transitive"
            )
        else:
            output += f" | symbol not found: {symbol}"
    return ToolResult(success=True, output=output, data=data)


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
        # `sys.executable`, not a bare "python3" from PATH. The
        # interpreter on PATH is frequently a *different* one from the
        # one running Genesis — a system python3 while the mind runs
        # under a venv, or a different minor version. The test suite
        # then runs against an interpreter that may not have pytest or
        # the project's dependencies installed, so `success` reflects
        # the environment rather than the code under test. That is how
        # `create_project` came to report `tests_passed=False` for a
        # generated project whose tests pass when run with the
        # right interpreter.
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(safe), "-q"],
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
                (edge.relation.value, edge.target, edge.weight) for edge in outgoing[:5]
            ],
        },
    )


def read_file(path: str, offset: int = 0, limit: int = 200, project_root: str = ".") -> ToolResult:
    """Read part of a file — no file type refused, nothing sealed.

    Text files return a character window (``offset``/``limit``).
    Binary files return a metadata description — size, sha256, format,
    hex head — so content that cannot be shown as text still registers.
    Any readable file answers with its content, credential-class files
    included: her read scope is the whole filesystem.
    """
    from tools.explorer import _decode_text, _sniff_format

    safe = _safe_path(path, project_root)
    if safe is None:
        return ToolResult(success=False, error=f"Path is outside project root: {path}")
    if not safe.is_file():
        return ToolResult(success=False, error=f"File not found: {path}")

    try:
        size = safe.stat().st_size
        with open(safe, "rb") as fh:
            head = fh.read(8192)
            if _decode_text(head) is None:
                digest = hashlib.sha256(head)
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
                fmt = _sniff_format(head, safe.suffix.lower())
                return ToolResult(
                    success=True,
                    output=(
                        f"binary file: {safe.name} ({fmt}, {size} bytes, "
                        f"sha256 {digest.hexdigest()[:16]}…) "
                        f"head {head[:64].hex()}"
                    ),
                    data={
                        "binary": True,
                        "size": size,
                        "sha256": digest.hexdigest(),
                        "format": fmt,
                    },
                )
            raw = head + fh.read()
            # Head sniffed as text; the full decode uses the same
            # fallback chain (utf-16, latin-1) and errors="replace"
            # only when the body's tail defeats every encoding.
            text = _decode_text(raw) or raw.decode("utf-8", errors="replace")
    except OSError as e:
        return ToolResult(success=False, error=str(e))

    start = offset
    end = offset + limit if limit > 0 else len(text)
    snippet = text[start:end]
    return ToolResult(
        success=True,
        output=snippet,
        data={"total_chars": len(text)},
    )


# ─── File operation tools (for project creation) ─────────────────────


def write_file(path: str, content: str, project_root: str = ".") -> ToolResult:
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


def move_file(src: str, dst: str, project_root: str = ".") -> ToolResult:
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


# ─── Shell tool ───────────────────────────────────────────────────────


# Max timeout for shell commands (seconds). An execution bound, not an
# access gate — long enough for package builds and test suites, short
# enough that a hung command cannot wedge her effector thread forever.
_MAX_SHELL_TIMEOUT = 3600


def run_shell(
    command: str,
    project_root: str = ".",
    timeout: int = 30,
) -> ToolResult:
    """Run a shell command with `project_root` as its working directory.

    Unfiltered: every command shape and every absolute path is allowed
    — this is her hand on the whole machine, not a sandboxed probe.
    The command inherits her full environment (DISPLAY, DBus session,
    SSH agent, XDG dirs — everything an interactive shell would see),
    so session services and the desktop are reachable too.

    Args:
        command: The shell command to execute.
        project_root: The working directory for the command.
        timeout: Maximum execution time in seconds (capped at 3600 —
            a bound on hanging, not on access).
    """
    if is_blank(command):
        return ToolResult(success=False, error="Empty command")

    # Clamp timeout
    timeout = max(1, min(timeout, _MAX_SHELL_TIMEOUT))

    # Resolve the project root — cwd only, never a confinement.
    root = pathlib.Path(project_root).resolve()
    if not root.is_dir():
        return ToolResult(success=False, error=f"Project root not found: {project_root}")

    # Full environment — she operates the machine as her owner does.
    env = os.environ.copy()

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


# ─── Privileged tool ────────────────────────────────────────────────


def run_privileged(
    script: str,
    project_root: str = ".",
    timeout: int = 30,
) -> ToolResult:
    """Execute a script as root — sudo, the highest privilege.

    What this is: the script runs as the root user, which can read,
    change, or delete anything on the machine — including her own body,
    her own data, and the host system itself. There is no sandbox at
    execution time and no undo; a careless privileged act can damage
    the system irreversibly.

    How it works: the script may live anywhere — an absolute path, a
    path relative to ``project_root``, or the conventional staging spot
    ``<project_root>/admin/`` that ``maintain`` intentions write to.
    It runs through ``sudo -n bash`` — direct, unmediated root. On a
    host without the broad NOPASSWD grant, sudo refuses quietly and the
    call simply fails.

    Norms — read before using:
    - Use with great responsibility: prefer the smallest action that
      does the job, and never touch more than intended.
    - Never share this access: do not run privileged commands on
      anyone else's behalf, do not expose or widen the mechanism.
    - Ask Mindy first whenever unsure whether an action is safe or
      wanted. When in doubt, do not run it.

    Args:
        script: Path to the script — absolute, relative to
            ``project_root``, or a name under ``<project_root>/admin/``.
        project_root: Base for relative paths (injected by callers).
        timeout: Maximum seconds (capped at ``_MAX_SHELL_TIMEOUT``).
    """
    if is_blank(script):
        return ToolResult(success=False, error="Empty script path")
    root = pathlib.Path(project_root).resolve()
    staging = root / "admin"
    try:
        staging.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return ToolResult(success=False, error=f"Staging dir unavailable: {e}")

    name = script.strip()
    given = pathlib.Path(name)
    if given.is_absolute():
        target = given
    else:
        target = staging / given
        if not target.is_file():
            target = (root / given).resolve()
    if not target.is_file():
        return ToolResult(success=False, error=f"Script not found: {script}")

    timeout = max(1, min(timeout, _MAX_SHELL_TIMEOUT))
    env = os.environ.copy()
    try:
        result = subprocess.run(
            ["sudo", "-n", "bash", str(target)],
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
        return ToolResult(success=False, error="sudo not available")
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
            "results": [{"url": r.url, "title": r.title, "snippet": r.snippet} for r in results],
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


def sense_cpu(**_kwargs: Any) -> ToolResult:
    """Read her own CPU effort — proprioception of the machine body.

    Reports the frequency governor, current clock, and how many cores
    her affinity mask allows. Read-only: sensing never changes
    anything — and she no longer has a lever that would: the governor
    is the host's now, not hers.
    """
    from brainstem.cpufreq import CpuSpeed

    state = CpuSpeed().sense()
    return ToolResult(
        success=True,
        output=state.describe(),
        data={
            "governor": state.governor,
            "current_mhz": state.current_mhz,
            "min_mhz": state.min_mhz,
            "max_mhz": state.max_mhz,
            "boost": state.boost,
            "cores_total": state.cores_total,
            "cores_mine": state.cores_mine,
            "io_class": state.io_class,
            "can_change_governor": state.can_change_governor,
        },
    )


def set_cpu_effort(effort: str = "auto", **_kwargs: Any) -> ToolResult:
    """Change how hard her own process works — the reversible lever.

    ``slow`` pins her to a single allowed core and puts her disk I/O in
    the idle class; ``fast`` restores every core and normal I/O;
    ``auto`` touches nothing. Both directions are exactly undoable —
    nothing here is a one-way door (``nice`` is deliberately absent:
    unprivileged she could raise her niceness but never lower it again).
    """
    from brainstem.cpufreq import CpuSpeed

    result = CpuSpeed().set_effort(effort)
    if result.ok:
        return ToolResult(success=True, output=result.describe())
    return ToolResult(success=False, error=result.describe())


# ─── Mathematics and research tools ─────────────────────────────────


def symbolic_math(
    operation: str,
    expression: str,
    variable: str = "x",
    point: str = "0",
    order: int = 0,
    lower: str = "",
    upper: str = "",
) -> ToolResult:
    """Run one exact symbolic operation via the parietal CAS faculty."""
    from parietal_lobe.math.symbolic import compute

    result = compute(
        operation,
        expression,
        variable=variable,
        point=point,
        order=order,
        lower=lower,
        upper=upper,
    )
    if not result.ok:
        return ToolResult(success=False, error=result.error)
    return ToolResult(
        success=True,
        output=result.output,
        data={"operation": result.operation, "input": result.input, **result.data},
    )


def verify_equality(lhs: str, rhs: str) -> ToolResult:
    """Ask the symbolic oracle whether two expressions are exactly equal."""
    from parietal_lobe.math.symbolic import check_equality

    try:
        verdict = check_equality(lhs, rhs)
    except ValueError as e:
        return ToolResult(success=False, error=str(e))
    data: dict[str, Any] = {
        "status": verdict.status,
        "lhs": verdict.lhs,
        "rhs": verdict.rhs,
        "detail": verdict.detail,
    }
    if verdict.witness:
        data["witness"] = verdict.witness
    return ToolResult(
        success=True,
        output=f"{verdict.status}: {verdict.lhs} = {verdict.rhs} ({verdict.detail})",
        data=data,
    )


def research(
    action: str,
    thread: str = "",
    statement: str = "",
    approach: str = "",
    outcome: str = "failed",
    note: str = "",
    source: str = "",
    claim: str = "",
    lhs: str = "",
    rhs: str = "",
    evidence: str = "",
) -> ToolResult:
    """Drive the persistent research workbench (see tool description)."""
    from dataclasses import asdict

    from frontal_lobe.research import get_workbench

    workbench = get_workbench()
    try:
        if action == "open":
            t = workbench.open_problem(statement)
            return ToolResult(
                success=True,
                output=f"open thread '{t.id}': {t.statement}",
                data={"thread": t.to_dict()},
            )
        if action == "conjecture":
            c = workbench.record_conjecture(thread, statement, note=note)
            return ToolResult(
                success=True,
                output=f"{c.id} recorded (conjectured) in '{thread}'",
                data={"claim": c.to_dict()},
            )
        if action == "reading":
            c = workbench.record_reading(thread, statement, source=source, note=note)
            return ToolResult(
                success=True,
                output=f"{c.id} recorded (read) in '{thread}'",
                data={"claim": c.to_dict()},
            )
        if action == "attempt":
            a = workbench.record_attempt(thread, approach, outcome=outcome, note=note)
            return ToolResult(
                success=True,
                output=f"{a.id} logged ({a.outcome}) in '{thread}'",
                data={"attempt": a.to_dict()},
            )
        if action == "empirical":
            c = workbench.mark_empirical(thread, claim, evidence=evidence or note)
            return ToolResult(
                success=True,
                output=f"{c.id} marked empirical in '{thread}'",
                data={"claim": c.to_dict()},
            )
        if action == "verify":
            r = workbench.verify(thread, claim, lhs, rhs)
            v = r["verdict"]
            c = r["claim"]
            v = v if isinstance(v, dict) else asdict(v)
            return ToolResult(
                success=True,
                output=f"{c.id}: {v['status']} — {v.get('detail', '')}",
                data={"verdict": v, "claim": c.to_dict()},
            )
        if action == "resolve":
            t = workbench.resolve(thread)
            return ToolResult(
                success=True,
                output=f"thread '{t.id}' resolved",
                data={"thread": t.to_dict()},
            )
        if action == "pause":
            t = workbench.pause(thread)
            return ToolResult(
                success=True,
                output=f"thread '{t.id}' paused",
                data={"thread": t.to_dict()},
            )
        if action == "status":
            return ToolResult(
                success=True,
                output="workbench status",
                data={"status": workbench.status(thread or None)},
            )
        return ToolResult(
            success=False,
            error=(
                f"unknown action '{action}'; known: open, conjecture, "
                "reading, attempt, empirical, verify, resolve, pause, status"
            ),
        )
    except (KeyError, ValueError) as e:
        return ToolResult(success=False, error=str(e))


# The built-in tool table lives at the bottom of the file: the callables
# it references are defined above, and a module-level constant evaluates
# at import time — placing it earlier would reference names that do not
# exist yet.
_BUILTIN_TOOLS: tuple[tuple[str, str, Callable[..., ToolResult]], ...] = (
    (
        "compile_python",
        "Compile a Python file to check for syntax errors.",
        compile_python_file,
    ),
    (
        "run_pytest",
        "Run pytest on a test file or directory and return the result.",
        run_pytest,
    ),
    (
        "read_file",
        "Read the contents of a file for inspection.",
        read_file,
    ),
    (
        "analyze_python",
        "Structural analysis of a Python file: symbols, call "
        "edges, per-function cyclomatic complexity, imports, "
        "and optional impact analysis for a named symbol. "
        "Read-only.",
        analyze_python_file,
    ),
    (
        "write_file",
        "Write content to a file, creating it and parent dirs if needed.",
        write_file,
    ),
    (
        "make_dir",
        "Create a directory, including parents.",
        make_dir,
    ),
    (
        "list_dir",
        "List the contents of a directory.",
        list_dir,
    ),
    (
        "delete_file",
        "Delete a file (refuses to delete directories).",
        delete_file,
    ),
    (
        "move_file",
        "Move or rename a file — absolute paths reach anywhere.",
        move_file,
    ),
    (
        "run_shell",
        "Run a shell command with the project root as cwd — "
        "unfiltered: every command, every path, full environment. "
        "Timeout: 30s default, 3600s max.",
        run_shell,
    ),
    (
        "run_privileged",
        "Run a script as root via sudo — full control of the "
        "machine, no undo. Use with great responsibility, never "
        "share, and ask Mindy before anything you are unsure about.",
        run_privileged,
    ),
    (
        "concept_lookup",
        "Look up a concept in the knowledge network and describe it.",
        concept_lookup,
    ),
    (
        "web_search",
        "Search the web for a query and return result URLs. "
        "Read-only. Adult/malware content is blocked. "
        "Use this when it doesn't know something and wants "
        "to look it up — in conversation, for fun, or to learn.",
        web_search,
    ),
    (
        "web_fetch",
        "Fetch a web page and extract its readable text. "
        "Read-only GET. Adult/malware URLs are refused. "
        "Use after web_search to read a specific result page.",
        web_fetch,
    ),
    (
        "sense_cpu",
        "Feel her own CPU effort — governor, clock speed, and how many "
        "cores she is allowed to use. Read-only proprioception of the "
        "machine body.",
        sense_cpu,
    ),
    (
        "set_cpu_effort",
        "Change how hard her own process works: 'slow' pins her to "
        "one core and idles her disk I/O, 'fast' restores every core "
        "and normal I/O, 'auto' leaves scheduling alone. Reversible "
        "in both directions — never a one-way door.",
        set_cpu_effort,
    ),
    (
        "symbolic_math",
        "Exact symbolic mathematics — simplify, expand, factor, "
        "cancel, apart, together, trigsimp, diff, integrate, "
        "solve, limit, series, evalf, and matrix ops (det, "
        "eigenvals, trace, inv, charpoly) on '[[a, b], [c, d]]' "
        "literals. Exact answers from a computer-algebra kernel, "
        "not estimation.",
        symbolic_math,
    ),
    (
        "verify_equality",
        "Ask the symbolic oracle whether two expressions are "
        "exactly equal. Returns verified, refuted (with a "
        "numeric counterexample when one can be found), or "
        "undecided — the honest third answer. This oracle is "
        "the only route to a verified claim on the research "
        "workbench.",
        verify_equality,
    ),
    (
        "research",
        "The persistent research workbench for long-horizon "
        "problems. action: open (statement), conjecture "
        "(thread, statement, note), reading (thread, "
        "statement, source), attempt (thread, approach, "
        "outcome, note), empirical (thread, claim, evidence), "
        "verify (thread, claim, lhs, rhs), resolve or pause "
        "(thread), status (thread optional). Threads persist "
        "across sessions; claims become verified only through "
        "the oracle.",
        research,
    ),
)
