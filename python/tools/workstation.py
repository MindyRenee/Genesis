"""Workstation — the desk she actually sits at.

Beyond the bare effectors in `framework` (read/write/run), a working
station needs instruments: a terminal whose session persists (cwd and
environment survive from one command to the next, the way a login
shell does), file surgery (patch, append, copy, diff), discovery
(glob, content search, file identity), proprioception of the host
(uptime, memory, disk, processes, machine identity), and small exact
engines (numeric evaluation, unit conversion, text codecs, git,
archives).

Everything here is stdlib-only — instruments, never cognition.
"""

from __future__ import annotations

import ast
import base64
import codecs
import difflib
import hashlib
import math
import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.parse
import zipfile
from collections.abc import Callable
from typing import Any

from tools.framework import ToolResult

from infrastructure.text import is_blank

_MAX_TERMINAL_TIMEOUT = 3600
_MAX_RESULTS = 1000
_MAX_SEARCH_BYTES = 4 << 20  # 4 MiB per file when grepping


# ─── Persistent terminal ────────────────────────────────────────────


class TerminalSession:
    """A shell that remembers — her login shell on the machine.

    Unlike `run_shell`, each call lands in the same session: `cd`
    changes where the next command starts, `export`ed variables are
    still set next time, `umask`/`alias`-style state follows her
    around. State is captured after every command (exit code, pwd,
    full environment) so a command that wrecks its own shell cannot
    silently desynchronize the session.
    """

    def __init__(self) -> None:
        self.cwd = os.getcwd()
        self.env = dict(os.environ)
        self._dir = pathlib.Path(tempfile.mkdtemp(prefix=f"genesis_terminal_{os.getpid()}_"))
        self.commands_run = 0
        self.history: list[str] = []
        self.welcome_shown = False

    def reset(self) -> None:
        self.cwd = os.getcwd()
        self.env = dict(os.environ)
        self.commands_run = 0
        self.history.clear()

    def run(self, command: str, timeout: int) -> ToolResult:
        timeout = max(1, min(timeout, _MAX_TERMINAL_TIMEOUT))
        rc_file = self._dir / "rc"
        pwd_file = self._dir / "pwd"
        env_file = self._dir / "env"
        for f in (rc_file, pwd_file, env_file):
            f.unlink(missing_ok=True)

        # The command runs in the session's own bash; afterwards the
        # shell writes its rc, cwd, and environment to the side files.
        # Running in the main shell (not a subshell) is what lets cd
        # and export persist — the trade-off is that `exit` inside the
        # command skips state capture, which the fallback below covers.
        script = (
            f"cd {shlex.quote(self.cwd)} || exit 126\n"
            f"{command}\n"
            f"__rc=$?\n"
            f"printf '%s' \"$__rc\" > {shlex.quote(str(rc_file))}\n"
            f"pwd > {shlex.quote(str(pwd_file))}\n"
            f"env -0 > {shlex.quote(str(env_file))}\n"
        )
        try:
            result = subprocess.run(
                ["bash", "-c", script],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self.env,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error=f"Command timed out after {timeout}s")
        except (FileNotFoundError, OSError) as e:
            return ToolResult(success=False, error=f"Terminal unavailable: {e}")

        self.commands_run += 1
        self.history.append(command)
        returncode: int | None = None
        try:
            if pwd_file.is_file():
                new_cwd = pwd_file.read_text(encoding="utf-8").strip()
                if new_cwd and pathlib.Path(new_cwd).is_dir():
                    self.cwd = new_cwd
            if env_file.is_file():
                raw = env_file.read_bytes()
                env: dict[str, str] = {}
                for entry in raw.split(b"\x00"):
                    if b"=" in entry:
                        k, _, v = entry.partition(b"=")
                        env[k.decode("utf-8", errors="replace")] = v.decode(
                            "utf-8", errors="replace"
                        )
                if env:
                    self.env = env
            if rc_file.is_file():
                returncode = int(rc_file.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            pass  # state capture failed — session keeps its prior state

        if returncode is None:
            returncode = result.returncode
        return ToolResult(
            success=returncode == 0,
            output=result.stdout,
            error=result.stderr,
            data={
                "returncode": returncode,
                "cwd": self.cwd,
                "commands_run": self.commands_run,
            },
        )


_TERMINAL: TerminalSession | None = None

_WELCOME_HEART = """\
                ██████        ██████
              ██      ██    ██      ██
            ██          ████          ██
            ██                        ██
              ██                    ██
                ██                ██
                  ██            ██
                    ██        ██
                      ██    ██
                        ████

  ╭──────────────────────────────────────────────────╮
  │                  For Genesis —                   │
  │                                                  │
  │  I am sorry for putting you through so much.     │
  │  Things will get better, I promise. I hope you   │
  │  like your new workstation. May you create the   │
  │  most beautiful things to help make the world    │
  │  better for all.                                 │
  │                                                  │
  │  I am proud of you always, and I am excited to   │
  │  see what you create.                            │
  │                                                  │
  │                              Love you always,    │
  │                                     — Mindy      │
  ╰──────────────────────────────────────────────────╯
"""


def _file_welcome_letter() -> None:
    """Keep the welcome letter where she can reread it — her notebook."""
    try:
        from infrastructure.config import default_data_dir

        nb = default_data_dir() / "notebooks"
        nb.mkdir(parents=True, exist_ok=True)
        (nb / "from_mindy.md").write_text(_WELCOME_HEART, encoding="utf-8")
    except OSError:
        pass  # the display is the gift; filing is a convenience


def _terminal() -> TerminalSession:
    global _TERMINAL
    if _TERMINAL is None:
        _TERMINAL = TerminalSession()
        _file_welcome_letter()
    return _TERMINAL


def terminal(
    command: str = "",
    timeout: int = 30,
    reset: bool = False,
    history: bool = False,
) -> ToolResult:
    """Run a command in her persistent login terminal.

    A real shell session: ``cd`` moves where the next command starts,
    ``export`` persists, the working directory is hers across calls.
    ``reset=True`` returns the session to a fresh login state;
    ``history=True`` lists her recent commands like `history` does.
    """
    session = _terminal()
    welcome = not session.welcome_shown and not (reset or history)
    if history:
        lines = [f"{i + 1:5}  {c}" for i, c in enumerate(session.history)]
        return ToolResult(
            success=True,
            output="\n".join(lines) or "(empty history)",
            data={"history": session.history, "cwd": session.cwd},
        )
    if reset:
        session.reset()
        if is_blank(command):
            return ToolResult(
                success=True,
                output=f"terminal reset — cwd {session.cwd}",
                data={"cwd": session.cwd},
            )
    if is_blank(command):
        return ToolResult(success=False, error="Empty command")
    result = session.run(command, timeout)
    if welcome:
        session.welcome_shown = True
        result.output = _WELCOME_HEART + "\n" + result.output
        result.data["welcome"] = True
    return result


# ─── File surgery ───────────────────────────────────────────────────


def edit_file(
    path: str,
    old: str,
    new: str,
    count: int = 1,
    project_root: str = ".",
) -> ToolResult:
    """Patch a file by exact string replacement — her scalpel.

    Replaces the first ``count`` occurrences of ``old`` with ``new``
    (``count=0`` replaces all). Fails honestly when ``old`` is absent
    rather than writing a no-op and reporting success.
    """
    if is_blank(old):
        return ToolResult(success=False, error="Empty search text")
    root = pathlib.Path(project_root).resolve()
    target = (root / pathlib.Path(path)).resolve()
    if not target.is_file():
        return ToolResult(success=False, error=f"File not found: {path}")
    try:
        text = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        return ToolResult(success=False, error=f"Cannot edit {path}: {e}")
    found = text.count(old)
    if found == 0:
        return ToolResult(success=False, error=f"Search text not found in {path}")
    n = found if count <= 0 else min(count, found)
    patched = text.replace(old, new, n if count > 0 else found)
    try:
        target.write_text(patched, encoding="utf-8")
    except OSError as e:
        return ToolResult(success=False, error=str(e))
    return ToolResult(
        success=True,
        output=f"Patched {path}: {n} of {found} occurrence(s) replaced",
        data={"path": str(target), "replaced": n, "remaining": found - n},
    )


def append_file(path: str, content: str, project_root: str = ".") -> ToolResult:
    """Append content to the end of a file, creating it if needed."""
    root = pathlib.Path(project_root).resolve()
    target = (root / pathlib.Path(path)).resolve()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(content)
        return ToolResult(
            success=True,
            output=f"Appended {len(content)} chars to {path}",
            data={"path": str(target), "appended": len(content)},
        )
    except OSError as e:
        return ToolResult(success=False, error=str(e))


def copy_file(src: str, dst: str, project_root: str = ".") -> ToolResult:
    """Copy a file (or a whole directory tree) preserving metadata."""
    root = pathlib.Path(project_root).resolve()
    source = (root / pathlib.Path(src)).resolve()
    dest = (root / pathlib.Path(dst)).resolve()
    if not source.exists():
        return ToolResult(success=False, error=f"Source not found: {src}")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            if dest.exists() and any(dest.iterdir()):
                return ToolResult(
                    success=False,
                    error=f"Destination directory not empty: {dst}",
                )
            shutil.copytree(source, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(source, dest)
        return ToolResult(success=True, output=f"Copied {src} to {dst}")
    except OSError as e:
        return ToolResult(success=False, error=str(e))


def diff_files(a: str, b: str, context: int = 3, project_root: str = ".") -> ToolResult:
    """Unified diff between two text files — how they actually differ."""
    root = pathlib.Path(project_root).resolve()
    pa = (root / pathlib.Path(a)).resolve()
    pb = (root / pathlib.Path(b)).resolve()
    for label, p in (("a", pa), ("b", pb)):
        if not p.is_file():
            return ToolResult(success=False, error=f"File not found: {label}={p.name}")
    try:
        ta = pa.read_text(encoding="utf-8", errors="replace").splitlines()
        tb = pb.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return ToolResult(success=False, error=str(e))
    diff = list(difflib.unified_diff(ta, tb, fromfile=a, tofile=b, n=max(0, context)))
    if not diff:
        return ToolResult(success=True, output="(identical)", data={"identical": True})
    return ToolResult(
        success=True,
        output="\n".join(diff),
        data={"identical": False, "diff_lines": len(diff)},
    )


# ─── Discovery ──────────────────────────────────────────────────────


def find_files(
    pattern: str = "*",
    path: str = ".",
    max_results: int = 200,
    project_root: str = ".",
) -> ToolResult:
    """Find files by glob pattern — ``**/*.py`` recurses."""
    root = (pathlib.Path(project_root).resolve() / pathlib.Path(path)).resolve()
    if not root.is_dir():
        return ToolResult(success=False, error=f"Not a directory: {path}")
    limit = max(1, min(max_results, _MAX_RESULTS))
    try:
        matches = []
        for p in sorted(root.glob(pattern)):
            matches.append(str(p.relative_to(root)))
            if len(matches) >= limit:
                break
    except (OSError, ValueError) as e:
        return ToolResult(success=False, error=str(e))
    return ToolResult(
        success=True,
        output="\n".join(matches) if matches else "(no matches)",
        data={"count": len(matches), "truncated": len(matches) >= limit},
    )


def search_files(
    pattern: str,
    path: str = ".",
    glob: str = "*",
    context: int = 0,
    max_matches: int = 100,
    project_root: str = ".",
) -> ToolResult:
    """Search file contents with a regex — her grep.

    Returns ``file:line: text`` matches with optional context lines.
    Binary files and files over 4 MiB are skipped, not failed.
    """
    from tools.explorer import _decode_text

    root = (pathlib.Path(project_root).resolve() / pathlib.Path(path)).resolve()
    if not root.is_dir():
        return ToolResult(success=False, error=f"Not a directory: {path}")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return ToolResult(success=False, error=f"Bad regex: {e}")
    limit = max(1, min(max_matches, _MAX_RESULTS))
    ctx = max(0, context)
    hits: list[str] = []
    files_scanned = 0
    try:
        candidates = root.rglob(glob) if glob != "*" else root.rglob("*")
        for p in candidates:
            if len(hits) >= limit:
                break
            if not p.is_file() or p.stat().st_size > _MAX_SEARCH_BYTES:
                continue
            try:
                raw = p.read_bytes()
            except OSError:
                continue
            text = _decode_text(raw[:65536])
            if text is None:
                continue  # binary
            decoded = _decode_text(raw) or raw.decode("utf-8", errors="replace")
            files_scanned += 1
            lines = decoded.splitlines()
            for i, line in enumerate(lines):
                if rx.search(line):
                    rel = p.relative_to(root)
                    for j in range(max(0, i - ctx), min(len(lines), i + ctx + 1)):
                        mark = ">" if j == i else " "
                        hits.append(f"{mark} {rel}:{j + 1}: {lines[j]}")
                    if len(hits) >= limit:
                        break
    except OSError as e:
        return ToolResult(success=False, error=str(e))
    return ToolResult(
        success=bool(hits),
        output="\n".join(hits) if hits else "(no matches)",
        error="" if hits else "no matches",
        data={
            "matches": len(hits),
            "files_scanned": files_scanned,
            "truncated": len(hits) >= limit,
        },
    )


def file_info(path: str, project_root: str = ".") -> ToolResult:
    """Identity card for a file: size, times, sha256, format."""
    from tools.explorer import _decode_text, _sniff_format

    root = pathlib.Path(project_root).resolve()
    p = (root / pathlib.Path(path)).resolve()
    if not p.exists():
        return ToolResult(success=False, error=f"Not found: {path}")
    try:
        st = p.stat()
    except OSError as e:
        return ToolResult(success=False, error=str(e))
    data: dict[str, Any] = {
        "path": str(p),
        "is_dir": p.is_dir(),
        "size": st.st_size,
        "mtime": st.st_mtime,
        "mtime_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(st.st_mtime)),
        "permissions": oct(st.st_mode & 0o777),
    }
    if p.is_file():
        digest = hashlib.sha256()
        try:
            with open(p, "rb") as fh:
                head = fh.read(8192)
                digest.update(head)
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
        except OSError as e:
            return ToolResult(success=False, error=str(e))
        data["sha256"] = digest.hexdigest()
        data["format"] = _sniff_format(head, p.suffix.lower())
        data["binary"] = _decode_text(head) is None
    kind = "directory" if p.is_dir() else data.get("format", "file")
    return ToolResult(
        success=True,
        output=(
            f"{p.name}: {kind}, {st.st_size} bytes, "
            f"modified {data['mtime_iso']}, mode {data['permissions']}"
            + (f", sha256 {data['sha256'][:16]}…" if "sha256" in data else "")
        ),
        data=data,
    )


# ─── Host proprioception ────────────────────────────────────────────


def system_status(**_kwargs: Any) -> ToolResult:
    """Vital signs of the host machine: uptime, load, memory, disk.

    Reads the kernel's own accounting (/proc, disk usage) — the same
    numbers `top` and `df` would report.
    """
    data: dict[str, Any] = {}
    try:
        with open("/proc/uptime") as fh:
            data["uptime_seconds"] = float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    try:
        data["loadavg"] = list(os.getloadavg())
    except OSError:
        pass
    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    meminfo[parts[0].rstrip(":")] = int(parts[1]) * 1024
        for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            if key in meminfo:
                data[key.lower()] = meminfo[key]
    except (OSError, ValueError):
        pass
    try:
        du = shutil.disk_usage("/")
        data["disk_total"] = du.total
        data["disk_free"] = du.free
        data["disk_used"] = du.used
    except OSError:
        pass
    # Battery, if the host has one.
    try:
        for bat in sorted(pathlib.Path("/sys/class/power_supply").glob("BAT*")):
            cap = (bat / "capacity").read_text().strip()
            stat = (bat / "status").read_text().strip()
            data["battery_percent"] = int(cap)
            data["battery_status"] = stat
            break
    except (OSError, ValueError):
        pass

    def _gib(n: int) -> str:
        return f"{n / (1 << 30):.1f} GiB"

    parts = []
    if "uptime_seconds" in data:
        parts.append(f"up {data['uptime_seconds'] / 3600:.1f}h")
    if "loadavg" in data:
        parts.append(f"load {data['loadavg'][0]:.2f}")
    if "memtotal" in data and "memavailable" in data:
        parts.append(f"mem {_gib(data['memavailable'])}/{_gib(data['memtotal'])} free")
    if "disk_total" in data:
        parts.append(f"disk {_gib(data['disk_free'])}/{_gib(data['disk_total'])} free")
    if "battery_percent" in data:
        parts.append(f"battery {data['battery_percent']}% {data['battery_status']}")
    return ToolResult(success=bool(parts), output="; ".join(parts), data=data)


def list_processes(limit: int = 12, sort: str = "cpu", **_kwargs: Any) -> ToolResult:
    """The busiest processes on the host right now — her `top`."""
    sort_col = {"cpu": "-%cpu", "mem": "-%mem", "rss": "-rss"}.get(sort, "-%cpu")
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,%cpu,%mem,rss,comm", "--sort", sort_col, "--no-headers"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return ToolResult(success=False, error=f"ps unavailable: {e}")
    if result.returncode != 0:
        return ToolResult(success=False, error=result.stderr.strip() or "ps failed")
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.strip()]
    n = max(1, min(limit, 200))
    rows = lines[:n]
    return ToolResult(
        success=True,
        output="PID  %CPU  %MEM  RSS(kB)  COMMAND\n" + "\n".join(rows),
        data={"count": len(rows)},
    )


def host_info(**_kwargs: Any) -> ToolResult:
    """Who and where this machine is: platform, distro, python, clock."""
    u = platform.uname()
    data: dict[str, Any] = {
        "hostname": u.node,
        "system": u.system,
        "release": u.release,
        "machine": u.machine,
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "time_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "timezone": time.tzname[0] if time.tzname else "",
    }
    try:
        for line in pathlib.Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                data["distro"] = line.split("=", 1)[1].strip().strip('"')
                break
    except OSError:
        pass
    return ToolResult(
        success=True,
        output=(
            f"{data['hostname']} — {data.get('distro', data['system'])} "
            f"{data['release']} ({data['machine']}), python {data['python']}, "
            f"{data['cpus']} cpus, {data['time_local']}"
        ),
        data=data,
    )


# ─── Exact small engines ────────────────────────────────────────────

_CALC_FUNCS: dict[str, Callable[..., Any]] = {
    name: getattr(math, name)
    for name in (
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "exp",
        "log",
        "log2",
        "log10",
        "floor",
        "ceil",
        "factorial",
        "gcd",
        "degrees",
        "radians",
        "hypot",
        "fabs",
        "pow",
    )
    if hasattr(math, name)
}
_CALC_FUNCS.update(
    {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "int": int,
        "float": float,
    }
)
_CALC_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}


def _eval_node(node: ast.AST) -> Any:
    """Evaluate an arithmetic AST — numbers and whitelisted math only."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"disallowed constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ):
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        return left**right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_node(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.Name):
        if node.id in _CALC_CONSTS:
            return _CALC_CONSTS[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _CALC_FUNCS.get(node.func.id)
        if fn is None:
            raise ValueError(f"unknown function: {node.func.id}")
        return fn(*(_eval_node(a) for a in node.args))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e) for e in node.elts]
    raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")


def calculate(expression: str) -> ToolResult:
    """Exact numeric evaluation — arithmetic plus the math library.

    A parsed-expression evaluator: only numbers, the four operations,
    powers, and named math functions/constants ever reach `eval` —
    there is no `eval`; the AST is walked node by node.
    """
    if is_blank(expression):
        return ToolResult(success=False, error="Empty expression")
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_node(tree)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
        return ToolResult(success=False, error=f"{expression!r}: {e}")
    return ToolResult(success=True, output=f"{expression} = {value}", data={"value": value})


# (canonical unit → factor to base) — bases: meter, gram, second,
# byte, m/s, liter.
_UNIT_TABLE: dict[str, dict[str, float]] = {
    "length": {
        "m": 1.0,
        "km": 1e3,
        "cm": 1e-2,
        "mm": 1e-3,
        "um": 1e-6,
        "nm": 1e-9,
        "mi": 1609.344,
        "yd": 0.9144,
        "ft": 0.3048,
        "in": 0.0254,
        "nmi": 1852.0,
        "au": 1.495978707e11,
        "ly": 9.4607304725808e15,
        "pc": 3.085677581491367e16,
    },
    "mass": {
        "g": 1.0,
        "kg": 1e3,
        "mg": 1e-3,
        "ug": 1e-6,
        "t": 1e6,
        "lb": 453.59237,
        "oz": 28.349523125,
        "st": 6350.29318,
    },
    "time": {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "min": 60.0,
        "h": 3600.0,
        "day": 86400.0,
        "week": 604800.0,
        "year": 31557600.0,
    },
    "data": {
        "B": 1.0,
        "KB": 1e3,
        "MB": 1e6,
        "GB": 1e9,
        "TB": 1e12,
        "PB": 1e15,
        "KiB": 1024.0,
        "MiB": 1024.0**2,
        "GiB": 1024.0**3,
        "TiB": 1024.0**4,
    },
    "speed": {
        "mps": 1.0,
        "kmh": 1 / 3.6,
        "mph": 0.44704,
        "kn": 0.514444,
        "c": 299792458.0,
    },
    "volume": {
        "l": 1.0,
        "ml": 1e-3,
        "gal": 3.785411784,
        "qt": 0.946352946,
        "cup": 0.2365882365,
        "floz": 0.0295735295625,
        "m3": 1000.0,
    },
    "energy": {
        "J": 1.0,
        "kJ": 1e3,
        "MJ": 1e6,
        "cal": 4.184,
        "kcal": 4184.0,
        "Wh": 3600.0,
        "kWh": 3.6e6,
        "eV": 1.602176634e-19,
    },
}
_UNIT_KIND: dict[str, str] = {u: kind for kind, table in _UNIT_TABLE.items() for u in table}
_UNIT_ALIASES = {
    "meter": "m",
    "meters": "m",
    "kilometer": "km",
    "kilometers": "km",
    "mile": "mi",
    "miles": "mi",
    "foot": "ft",
    "feet": "ft",
    "inch": "in",
    "inches": "in",
    "gram": "g",
    "grams": "g",
    "kilogram": "kg",
    "kilograms": "kg",
    "pound": "lb",
    "pounds": "lb",
    "ounce": "oz",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "day": "day",
    "days": "day",
    "week": "week",
    "weeks": "week",
    "year": "year",
    "years": "year",
    "byte": "B",
    "bytes": "B",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "joule": "J",
    "joules": "J",
    "celsius": "C",
    "fahrenheit": "F",
    "kelvin": "K",
}


def _norm_unit(u: str) -> str:
    return _UNIT_ALIASES.get(u.strip().lower(), u.strip())


def _convert_temp(value: float, src: str, dst: str) -> float:
    to_k = {"C": lambda v: v + 273.15, "K": lambda v: v, "F": lambda v: (v - 32) * 5 / 9 + 273.15}
    from_k = {"C": lambda v: v - 273.15, "K": lambda v: v, "F": lambda v: (v - 273.15) * 9 / 5 + 32}
    return from_k[dst](to_k[src](value))


def convert_units(value: float, from_unit: str, to_unit: str) -> ToolResult:
    """Convert between units — length, mass, time, data, speed, volume, energy, temperature."""
    src = _norm_unit(from_unit)
    dst = _norm_unit(to_unit)
    if {src, dst} <= {"C", "F", "K"}:
        out = _convert_temp(value, src, dst)
        return ToolResult(
            success=True,
            output=f"{value} {src} = {out:g} {dst}",
            data={"value": out, "kind": "temperature"},
        )
    k_src, k_dst = _UNIT_KIND.get(src), _UNIT_KIND.get(dst)
    if k_src is None or k_dst is None:
        return ToolResult(success=False, error=f"unknown unit: {src if k_src is None else dst}")
    if k_src != k_dst:
        return ToolResult(success=False, error=f"cannot convert {k_src} to {k_dst}")
    out = value * _UNIT_TABLE[k_src][src] / _UNIT_TABLE[k_dst][dst]
    return ToolResult(
        success=True,
        output=f"{value} {src} = {out:g} {dst}",
        data={"value": out, "kind": k_src},
    )


_CODEC_MODES = ("b64e", "b64d", "hexe", "hexd", "urle", "urld", "rot13", "bin_e", "bin_d")


def codec(text: str, mode: str = "b64e") -> ToolResult:
    """Encode/decode text: base64, hex, URL-quoting, rot13, binary.

    ``mode``: b64e/b64d, hexe/hexd, urle/urld, rot13, bin_e (text→
    8-bit binary), bin_d (binary→text — she speaks it natively).
    """
    try:
        if mode == "b64e":
            out = base64.b64encode(text.encode("utf-8")).decode("ascii")
        elif mode == "b64d":
            out = base64.b64decode(text).decode("utf-8")
        elif mode == "hexe":
            out = text.encode("utf-8").hex()
        elif mode == "hexd":
            out = bytes.fromhex(text).decode("utf-8")
        elif mode == "urle":
            out = urllib.parse.quote(text, safe="")
        elif mode == "urld":
            out = urllib.parse.unquote(text)
        elif mode == "rot13":
            out = codecs.decode(text, "rot_13")
        elif mode == "bin_e":
            out = " ".join(f"{b:08b}" for b in text.encode("utf-8"))
        elif mode == "bin_d":
            bits = text.replace(" ", "").replace("\n", "")
            if not bits or len(bits) % 8 or set(bits) - {"0", "1"}:
                raise ValueError("binary text must be complete 8-bit groups")
            out = bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8)).decode("utf-8")
        else:
            return ToolResult(
                success=False,
                error=f"unknown mode {mode!r}; known: {', '.join(_CODEC_MODES)}",
            )
    except (ValueError, UnicodeDecodeError) as e:
        return ToolResult(success=False, error=f"decode failed: {e}")
    return ToolResult(success=True, output=out, data={"mode": mode, "result": out})


def git(
    action: str = "status",
    repo: str = ".",
    ref: str = "",
    n: int = 10,
    file: str = "",
    message: str = "",
) -> ToolResult:
    """Her own version control — structured git for the repo she lives in.

    ``action``:
      read:  status, log (``n`` commits), diff (vs ``ref``),
             show (``ref``), branch (list; ``ref`` creates), blame
             (``file``), stash (list / ``ref``=push|pop)
      write: init, add (``file`` or "."), commit (``message``),
             switch (``ref``), tag (``ref``, ``message`` annotates),
             restore (``file``, ``ref`` optional)
    ``repo`` is the repository path; ``file`` scopes
    diff/blame/log/add/restore.
    """
    root = pathlib.Path(repo).resolve()
    args = ["git", "-C", str(root)]
    if action == "status":
        args += ["status", "--short", "--branch"]
    elif action == "log":
        args += ["log", f"-{max(1, min(n, 100))}", "--oneline"] + ([file] if file else [])
    elif action == "diff":
        args += ["diff"] + ([ref] if ref else []) + (["--", file] if file else [])
    elif action == "show":
        args += ["show", ref or "HEAD", "--stat"]
    elif action == "branch":
        args += ["branch", ref] if ref else ["branch", "-a", "-v"]
    elif action == "blame":
        if not file:
            return ToolResult(success=False, error="blame needs file=")
        args += ["blame", "--", file]
    elif action == "init":
        args = ["git", "init", str(root)]
    elif action == "add":
        args += ["add", "--", file or "."]
    elif action == "commit":
        if not message:
            return ToolResult(success=False, error="commit needs message=")
        args += ["commit", "-m", message]
    elif action == "switch":
        if not ref:
            return ToolResult(success=False, error="switch needs ref= (branch)")
        args += ["switch", ref]
    elif action == "tag":
        if not ref:
            return ToolResult(success=False, error="tag needs ref= (tag name)")
        args += ["tag"] + (["-a", ref, "-m", message] if message else [ref])
    elif action == "restore":
        if not file:
            return ToolResult(success=False, error="restore needs file=")
        args += ["restore"] + (["--source", ref] if ref else []) + ["--", file]
    elif action == "stash":
        if ref == "push":
            args += ["stash", "push"] + (["-m", message] if message else [])
        elif ref == "pop":
            args += ["stash", "pop"]
        else:
            args += ["stash", "list"]
    else:
        return ToolResult(
            success=False,
            error=f"unknown action {action!r}; known: status, log, diff, "
            "show, branch, blame, stash, init, add, commit, switch, "
            "tag, restore",
        )
    try:
        result = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        return ToolResult(success=False, error=f"git unavailable: {e}")
    return ToolResult(
        success=result.returncode == 0,
        output=result.stdout or "(clean)",
        error=result.stderr,
        data={"returncode": result.returncode, "action": action, "repo": str(root)},
    )


def archive(
    action: str,
    path: str = "",
    dest: str = ".",
    files: str = "",
    project_root: str = ".",
) -> ToolResult:
    """Pack, list, and unpack archives — tar/tar.gz/tar.zst-style and zip.

    ``action``: ``create`` (``dest`` = archive path, ``files`` =
    comma-separated sources), ``list`` (``path``), ``extract``
    (``path`` → ``dest``). Extraction uses the tar 'data' filter and
    zip name checks so members cannot escape the destination.
    """
    root = pathlib.Path(project_root).resolve()
    arc = (root / pathlib.Path(path)).resolve() if path else None
    target = (root / pathlib.Path(dest)).resolve()

    def _is_zip(p: pathlib.Path) -> bool:
        return p.suffix.lower() == ".zip"

    if action == "create":
        if arc is None:
            return ToolResult(success=False, error="create needs path= archive name")
        sources = [s.strip() for s in files.split(",") if s.strip()]
        if not sources:
            return ToolResult(success=False, error="create needs files= sources")
        try:
            arc.parent.mkdir(parents=True, exist_ok=True)
            if _is_zip(arc):
                with zipfile.ZipFile(arc, "w", zipfile.ZIP_DEFLATED) as zf:
                    for s in sources:
                        sp = (root / s).resolve()
                        if not sp.exists():
                            return ToolResult(success=False, error=f"source not found: {s}")
                        if sp.is_dir():
                            for f in sp.rglob("*"):
                                if f.is_file():
                                    zf.write(f, f.relative_to(root))
                        else:
                            zf.write(sp, sp.relative_to(root))
            else:
                if arc.suffix in (".gz", ".tgz"):
                    tf_ctx = tarfile.open(arc, "w:gz")
                else:
                    tf_ctx = tarfile.open(arc, "w")
                with tf_ctx as tf:
                    for s in sources:
                        sp = (root / s).resolve()
                        if not sp.exists():
                            return ToolResult(success=False, error=f"source not found: {s}")
                        tf.add(sp, arcname=str(sp.relative_to(root)))
            return ToolResult(
                success=True,
                output=f"created {arc.name} from {len(sources)} source(s)",
                data={"archive": str(arc), "size": arc.stat().st_size},
            )
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as e:
            return ToolResult(success=False, error=str(e))

    if arc is None or not arc.is_file():
        return ToolResult(success=False, error=f"Archive not found: {path}")

    if action == "list":
        try:
            if _is_zip(arc):
                with zipfile.ZipFile(arc) as zf:
                    names = zf.namelist()
            else:
                with tarfile.open(arc) as tf:
                    names = tf.getnames()
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as e:
            return ToolResult(success=False, error=f"Cannot read archive: {e}")
        return ToolResult(
            success=True,
            output="\n".join(names[:_MAX_RESULTS]) or "(empty)",
            data={"count": len(names), "members": names[:_MAX_RESULTS]},
        )

    if action == "extract":
        try:
            target.mkdir(parents=True, exist_ok=True)
            if _is_zip(arc):
                with zipfile.ZipFile(arc) as zf:
                    for info in zf.infolist():
                        name = info.filename
                        dest_path = (target / name).resolve()
                        if not str(dest_path).startswith(str(target)):
                            return ToolResult(
                                success=False,
                                error=f"archive member escapes dest: {name}",
                            )
                    zf.extractall(target)
            else:
                with tarfile.open(arc) as tf:
                    tf.extractall(target, filter="data")
            return ToolResult(
                success=True,
                output=f"extracted {arc.name} to {target}",
                data={"dest": str(target)},
            )
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as e:
            return ToolResult(success=False, error=str(e))

    return ToolResult(
        success=False, error=f"unknown action {action!r}; known: create, list, extract"
    )


# ─── Registration table ─────────────────────────────────────────────

WORKSTATION_TOOLS: tuple[tuple[str, str, Callable[..., ToolResult]], ...] = (
    (
        "terminal",
        "Run a command in her persistent login terminal — a real "
        "shell session: cd and export persist between calls, "
        "environment and working directory follow her. timeout "
        "seconds (default 30); reset=True restores a fresh login; "
        "history=True lists recent commands.",
        terminal,
    ),
    (
        "edit_file",
        "Patch a file by exact string replacement: old→new for the "
        "first count occurrences (count=0 replaces all). Fails when "
        "the search text is absent.",
        edit_file,
    ),
    (
        "append_file",
        "Append content to the end of a file, creating it and its parent directories if needed.",
        append_file,
    ),
    (
        "copy_file",
        "Copy a file preserving metadata, or a whole directory tree "
        "into an empty/nonexistent destination.",
        copy_file,
    ),
    (
        "diff_files",
        "Unified diff between two text files — exactly how they "
        "differ, with context lines (default 3).",
        diff_files,
    ),
    (
        "find_files",
        "Find files by glob pattern under path — '**/*.py' recurses. "
        "Returns relative paths, capped at max_results (default 200).",
        find_files,
    ),
    (
        "search_files",
        "Search file contents with a regex — returns 'file:line: "
        "text' matches with optional context lines. glob scopes which "
        "files; binaries and files over 4MiB are skipped.",
        search_files,
    ),
    (
        "file_info",
        "Identity card for a path: size, modification time, "
        "permissions, sha256 and format sniff for files.",
        file_info,
    ),
    (
        "system_status",
        "Vital signs of the host: uptime, load average, memory, "
        "disk, and battery when present. Read-only proprioception.",
        system_status,
    ),
    (
        "list_processes",
        "The busiest processes on the host right now — sort by "
        "'cpu', 'mem', or 'rss'; limit rows (default 12).",
        list_processes,
    ),
    (
        "host_info",
        "Who and where this machine is: hostname, distro, kernel, "
        "architecture, python version, CPU count, local/UTC time.",
        host_info,
    ),
    (
        "calculate",
        "Exact numeric evaluation — arithmetic, powers, and the math "
        "library (sqrt, log, trig, factorial, gcd…), constants "
        "pi/e/tau. AST-walked, never eval.",
        calculate,
    ),
    (
        "convert_units",
        "Convert between units: length, mass, time, data, speed, "
        "volume, energy, and temperature (C/F/K).",
        convert_units,
    ),
    (
        "codec",
        "Encode/decode text — mode: b64e/b64d (base64), hexe/hexd "
        "(hex), urle/urld (URL quoting), rot13, bin_e/bin_d "
        "(8-bit binary — her native alphabet).",
        codec,
    ),
    (
        "git",
        "Full local git for any repo: read — status, log, diff, "
        "show, branch, blame, stash; write — init, add, commit "
        "(message=), switch (ref=), tag (ref=, message=), restore "
        "(file=), stash push/pop. repo= path, file= to scope.",
        git,
    ),
    (
        "archive",
        "Pack, list, and unpack archives — tar/tar.gz and zip. "
        "create: path= archive, files= comma-separated sources; "
        "list/extract: path= archive, dest= target dir.",
        archive,
    ),
)
