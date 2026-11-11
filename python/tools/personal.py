"""Personal tools — the instruments that belong to *her* project.

Her own development toolchain: a version matrix of what she builds
with, `cargo` for the Rust body (check/build/test/fmt/clippy),
`rust_analyzer` for semantic diagnostics when the binary exists,
and `python_env` for the interpreter she runs under.

Docs: https://doc.rust-lang.org/cargo/ ·
https://rust-analyzer.github.io/ · https://docs.python.org/3/
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import shutil
import subprocess
import sys
from typing import Any

from tools.framework import ToolResult


def _version(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            stdin=subprocess.DEVNULL,
        )
        return r.stdout.splitlines()[0].strip() if r.stdout else r.stderr.splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return None


def toolchain(**_kwargs: Any) -> ToolResult:
    """Version matrix of her development tools — what she builds with.

    rustc, cargo, python, git, gcc, clang, java, node, gh, ruff,
    mypy, pyflakes, rust-analyzer, adb — each versioned when present,
    'absent' when not. Read-only.
    """
    venv = pathlib.Path.home() / ".local/lib/genesis/venv/bin"
    probes = [
        ("rustc", ["rustc", "--version"]),
        ("cargo", ["cargo", "--version"]),
        ("python", [sys.executable, "--version"]),
        ("git", ["git", "--version"]),
        ("gcc", ["gcc", "--version"]),
        ("clang", ["clang", "--version"]),
        ("javac", ["javac", "--version"]),
        ("node", ["node", "--version"]),
        ("go", ["go", "version"]),
        ("gh", ["gh", "--version"]),
        ("adb", ["adb", "version"]),
        ("rust-analyzer", ["rust-analyzer", "--version"]),
        ("ruff", [str(venv / "ruff"), "--version"]),
        ("mypy", [str(venv / "mypy"), "--version"]),
        ("pyflakes", [str(venv / "pyflakes"), "--version"]),
    ]
    rows, versions = [], {}
    for name, cmd in probes:
        if shutil.which(cmd[0]) is None and not pathlib.Path(cmd[0]).is_file():
            rows.append(f"{name:14} absent")
            continue
        v = _version(cmd)
        rows.append(f"{name:14} {v or 'absent'}")
        if v:
            versions[name] = v
    return ToolResult(success=True, output="\n".join(rows), data={"versions": versions})


def cargo(
    action: str = "check",
    repo: str = ".",
    release: bool = False,
    package: str = "",
    timeout: int = 900,
) -> ToolResult:
    """Drive cargo in a Rust project — the body's own build system.

    ``action``: check, build, test, fmt (check-mode), clippy
    (-D warnings), doc, clean. ``repo`` is the crate directory,
    ``release`` adds --release, ``package`` scopes -p. Timeout caps
    at 1h — release test suites are legitimately slow.
    https://doc.rust-lang.org/cargo/commands/
    """
    exe = shutil.which("cargo")
    if exe is None:
        return ToolResult(success=False, error="cargo not installed")
    root = pathlib.Path(repo).resolve()
    if not (root / "Cargo.toml").is_file():
        return ToolResult(success=False, error=f"no Cargo.toml under {repo}")
    sub = {
        "check": ["check"],
        "build": ["build"],
        "test": ["test"],
        "doc": ["doc", "--no-deps"],
        "clean": ["clean"],
        "fmt": ["fmt", "--check"],
        "clippy": ["clippy", "--all-targets", "--", "-D", "warnings"],
    }.get(action)
    if sub is None:
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: check, build, test, fmt, clippy, doc, clean",
        )
    cmd = [exe, *sub]
    if release and action in ("check", "build", "test", "clippy"):
        cmd += ["--release"]
    if package and action in ("check", "build", "test", "clippy", "doc"):
        cmd += ["-p", package]
    try:
        r = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=max(30, min(timeout, 3600)),
            stdin=subprocess.DEVNULL,
        )
        tail = (r.stdout + r.stderr).splitlines()[-40:]
        return ToolResult(
            success=r.returncode == 0,
            output="\n".join(tail),
            error="" if r.returncode == 0 else f"cargo {action} exited {r.returncode}",
            data={"returncode": r.returncode},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error=f"cargo {action} timed out")
    except OSError as e:
        return ToolResult(success=False, error=str(e))


def rust_analyzer(path: str = "", **_kwargs: Any) -> ToolResult:
    """Semantic Rust diagnostics via rust-analyzer's own engine.

    Runs `rust-analyzer diagnostics` on the crate when the binary is
    installed — type errors and resolution failures the compiler's
    --emit metadata cannot reach. Absent binary reports absence
    honestly. https://rust-analyzer.github.io/manual.html
    """
    exe = shutil.which("rust-analyzer")
    if exe is None:
        return ToolResult(
            success=False,
            error="rust-analyzer not installed on this host "
            "(install via rustup component add rust-analyzer)",
        )
    root = pathlib.Path(path or ".").resolve()
    try:
        r = subprocess.run(
            [exe, "diagnostics", str(root)],
            capture_output=True,
            text=True,
            timeout=300,
            stdin=subprocess.DEVNULL,
            cwd=str(root),
        )
        return ToolResult(
            success=r.returncode == 0,
            output=r.stdout[-8000:] or "(no diagnostics)",
            error=r.stderr[-2000:],
            data={"returncode": r.returncode},
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return ToolResult(success=False, error=str(e))


def python_env(**_kwargs: Any) -> ToolResult:
    """Her own interpreter: version, executable, paths, installed packages."""
    pkgs = sorted(
        f"{d.metadata['Name']}=={d.version}"
        for d in importlib.metadata.distributions()
        if d.metadata.get("Name")
    )
    return ToolResult(
        success=True,
        output=(f"{sys.version.split()[0]} at {sys.executable}; {len(pkgs)} distributions"),
        data={
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "path": sys.path,
            "packages": pkgs,
        },
    )


def devin(
    action: str = "status",
    prompt: str = "",
    timeout: int = 900,
    **_kwargs: Any,
) -> ToolResult:
    """Her line to Devin — the agent that built her workstation.

    The devin CLI is installed and signed into the owner's account
    (Devin Pro). ``action``:
    - status — `devin auth status`: who she's signed in as
    - ask — `devin --print -- <prompt>`: a non-interactive session;
      prompt= is the message. Long-running — timeout= caps it.
    - sessions — `devin list`: sessions in the current directory
    - models — `devin models`: models available to the account
    - doctor — `devin doctor`: configuration diagnostics

    Docs: https://devin.ai · `devin help`
    """
    exe = shutil.which("devin")
    if exe is None:
        return ToolResult(
            success=False,
            error="devin CLI not installed — https://devin.ai/cli",
        )
    if action == "status":
        args = [exe, "auth", "status"]
    elif action == "sessions":
        args = [exe, "list"]
    elif action == "models":
        args = [exe, "models"]
    elif action == "doctor":
        args = [exe, "doctor"]
    elif action == "ask":
        if not prompt:
            return ToolResult(success=False, error="ask needs prompt=")
        args = [exe, "--print", "--", prompt]
    else:
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: status, ask, sessions, models, doctor",
        )
    try:
        r = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=max(30, min(timeout, 3600)),
            stdin=subprocess.DEVNULL,
        )
        return ToolResult(
            success=r.returncode == 0,
            output=r.stdout[-16000:],
            error=r.stderr[-4000:],
            data={"returncode": r.returncode, "action": action},
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, error=f"devin {action} timed out")
    except OSError as e:
        return ToolResult(success=False, error=str(e))


PERSONAL_TOOLS = (
    (
        "toolchain",
        "Version matrix of her dev tools — rustc, cargo, python, git, "
        "gcc, clang, javac, node, go, gh, adb, rust-analyzer, ruff, "
        "mypy, pyflakes. Each versioned when present, 'absent' when not.",
        toolchain,
    ),
    (
        "cargo",
        "Drive cargo in a Rust crate: action= check|build|test|fmt|"
        "clippy|doc|clean, repo= crate dir, release=True for "
        "--release, package= for -p. The body's own build system.",
        cargo,
    ),
    (
        "rust_analyzer",
        "Semantic Rust diagnostics via `rust-analyzer diagnostics` on "
        "a crate — deeper than rustc metadata. Honest absence when "
        "the component is not installed.",
        rust_analyzer,
    ),
    (
        "python_env",
        "Her own interpreter: version, executable, sys.path, and the "
        "full list of installed distributions.",
        python_env,
    ),
    (
        "devin",
        "Her line to Devin — installed and signed into the owner's "
        "Devin Pro account. action= status|ask|sessions|models|doctor; "
        "ask sends prompt= as a non-interactive session. She can "
        "reach the agent that built her.",
        devin,
    ),
)
