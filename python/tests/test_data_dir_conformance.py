"""There is exactly one state directory, and every site lands on it.

The canonical location is ``~/.local/share/genesis``; the only override
is ``GENESIS_DATA_DIR`` (development and testing). There is no pin file
and no ``XDG_DATA_HOME`` step: ambient environment must never be able to
point one component at a second state tree while the rest read the
first — that is how this checkout once forked into two divergent
directories for six days, ~8.7k concepts and 8 projects landing in a
directory under ``~/.var/app/`` that nothing was monitoring.

This module asserts the invariant three ways:

1. Every resolution site — ``genesis``, ``config.py``, and every tool
   that computes a default — returns the same answer under a
   deliberately hostile environment (``XDG_DATA_HOME`` set to a foreign
   path must be *ignored* by all of them, identically).
2. A text-level sweep: nothing in the tree may read
   ``.genesis-data-dir`` or use ``XDG_DATA_HOME`` for the state dir —
   that is how a fork mechanism would be reintroduced.
3. The pin file itself must not exist in this checkout.

The Rust side has the mirrored unit tests in ``src/data_dir.rs``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PIN_NAME = ".genesis-data-dir"
HOSTILE_XDG = "/sandbox/private/data"
CANONICAL = "/home/tester/.local/share/genesis"


def _run(cmd: list[str], env: dict[str, str], cwd: Path | None = None) -> str:
    """Run a command, returning stdout, and fail loudly on error."""
    proc = subprocess.run(
        cmd,
        env=env,
        cwd=cwd or REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"{cmd} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def _base_env(**overrides: str | None) -> dict[str, str]:
    """A clean environment plus the given overrides (``None`` unsets)."""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"GENESIS_DATA_DIR", "XDG_DATA_HOME", "HOME"}
    }
    env["HOME"] = "/home/tester"
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


# ── Individual resolvers ──────────────────────────────────────────
# Each returns the data directory it would use, given the environment.


def resolve_launcher(env: dict[str, str]) -> str:
    """genesis — the interactive launcher."""
    source = (REPO / "genesis").read_text()
    # The resolution lives before argument parsing; everything after it
    # starts or stops processes.
    header = source.split("# ─── Parse arguments", 1)[0]
    out = _run(
        ["bash", "-c", f'{header}\nprintf "%s" "$DATA_DIR"'],
        env,
    )
    return out.strip()


def resolve_config_py(env: dict[str, str]) -> str:
    """infrastructure/config.py — the in-process default."""
    out = _run(
        [
            "python3",
            "-c",
            "import sys; sys.path[:0] = ['python', '.'];"
            "from infrastructure.config import default_data_dir;"
            "print(default_data_dir())",
        ],
        env,
    )
    return out.strip()


def resolve_setup_embeddings(env: dict[str, str]) -> str:
    """setup_embeddings.py — builds an index over the concept set."""
    out = _run(
        [
            "python3",
            "-c",
            "import importlib.util, sys;"
            "sys.path[:0] = ['python', '.'];"
            "spec = importlib.util.spec_from_file_location("
            "    'se', 'setup_embeddings.py');"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
            "sys.argv = ['setup_embeddings.py'];"
            "print(m._parse_setup_args()[0])",
        ],
        env,
    )
    return out.strip()


def resolve_vosk_dir_shell(env: dict[str, str]) -> str:
    """scripts/install_optional_deps.sh — where it installs the model."""
    out = _run([str(REPO / "scripts/install_optional_deps.sh"), "--data-dir"], env)
    return out.strip()


# The two real implementations of the rule — the shell launcher and
# the in-process Python resolver. Everything else must delegate to one
# of these; it is checked for delegation below, not re-tested.
DIR_RESOLVERS = {
    "genesis": resolve_launcher,
    "config.py": resolve_config_py,
    "setup_embeddings.py": resolve_setup_embeddings,
}

# Tools that must reach the data dir through the shared resolver —
# a private reimplementation is how forks start.
DELEGATORS = {
    "scripts/discord_pulse.py": "default_data_dir",
    "scripts/restore_archived_dreams.py": "default_data_dir",
    "scripts/recover_pruned_concepts.py": "default_data_dir",
    "eval/abstraction_gradient.py": "default_data_dir",
    "setup_embeddings.py": "default_data_dir",
}

# Asset installers place content inside the resolved data dir — checked
# as data dir plus subpath.
ASSET_RESOLVERS = {
    "install_optional_deps.sh (vosk)": (resolve_vosk_dir_shell, "vosk-models"),
}


# ── Cases ─────────────────────────────────────────────────────────


def _cases() -> list[tuple[str, dict[str, str | None], str]]:
    """(name, env overrides, expected data dir).

    The hostile ``XDG_DATA_HOME`` must be ignored identically by every
    resolver — a process launched by a sandboxed host must land on the
    canonical directory like everything else.
    """
    return [
        (
            "hostile XDG_DATA_HOME is ignored — canonical wins",
            {"XDG_DATA_HOME": HOSTILE_XDG},
            CANONICAL,
        ),
        (
            "explicit GENESIS_DATA_DIR is the only override",
            {"GENESIS_DATA_DIR": "/state/explicit", "XDG_DATA_HOME": HOSTILE_XDG},
            "/state/explicit",
        ),
        (
            "blank GENESIS_DATA_DIR is treated as unset",
            {"GENESIS_DATA_DIR": "   ", "XDG_DATA_HOME": HOSTILE_XDG},
            CANONICAL,
        ),
        (
            "no XDG_DATA_HOME at all still resolves canonical",
            {"XDG_DATA_HOME": None},
            CANONICAL,
        ),
    ]


@pytest.mark.parametrize(
    ("case", "overrides", "expected"),
    _cases(),
    ids=[c[0] for c in _cases()],
)
def test_resolvers_agree(
    case: str,
    overrides: dict[str, str | None],
    expected: str,
) -> None:
    """Every resolver returns the same answer under every environment."""
    env = _base_env(**dict(overrides))

    for name, resolver in DIR_RESOLVERS.items():
        got = resolver(env)
        assert got == expected, (
            f"{name} disagrees under '{case}'.\n"
            f"  expected: {expected}\n"
            f"  got:      {got}\n"
            f"  env:      GENESIS_DATA_DIR={env.get('GENESIS_DATA_DIR')!r} "
            f"XDG_DATA_HOME={env.get('XDG_DATA_HOME')!r}\n"
            f"  There is one rule — GENESIS_DATA_DIR else "
            f"~/.local/share/genesis — and every site implements it."
        )


@pytest.mark.parametrize(
    ("case", "overrides", "expected"),
    _cases(),
    ids=[c[0] for c in _cases()],
)
def test_asset_resolvers_live_under_the_data_dir(
    case: str,
    overrides: dict[str, str | None],
    expected: str,
) -> None:
    """Asset install targets sit inside the resolved data dir."""
    env = _base_env(**dict(overrides))

    for name, (resolver, subdir) in ASSET_RESOLVERS.items():
        got = resolver(env)
        assert got == f"{expected}/{subdir}", (
            f"{name} points outside the data dir under '{case}'.\n"
            f"  expected: {expected}/{subdir}\n"
            f"  got:      {got}"
        )


def test_tools_delegate_rather_than_resolve() -> None:
    """Every tool that needs the data dir imports the shared resolver.

    A private reimplementation — however small — is how the resolvers
    drifted into a fork in the first place. Each delegator must import
    ``default_data_dir`` and define no ``_default_data_dir`` of its own.
    """
    for rel, symbol in DELEGATORS.items():
        text = (REPO / rel).read_text()
        assert f"import {symbol}" in text or f"import {symbol} " in text, (
            f"{rel} does not import {symbol} — it must delegate to the "
            "shared resolver, not carry its own."
        )
        assert "def _default_data_dir" not in text, (
            f"{rel} defines a private _default_data_dir — delegate to "
            "infrastructure.config.default_data_dir instead."
        )


def test_no_pin_file_exists() -> None:
    """The pin mechanism is gone — the file must not exist either.

    A leftover or reintroduced `.genesis-data-dir` is dead configuration
    that looks live; its absence is part of the one-directory invariant.
    """
    assert not (REPO / PIN_NAME).exists(), (
        f"{PIN_NAME} exists — the pin mechanism was removed; delete the "
        "file. State lives in the canonical directory, only overridable "
        "by GENESIS_DATA_DIR."
    )


def test_nothing_reads_pin_or_xdg_for_state() -> None:
    """No source file may reintroduce the fork mechanism.

    Sweeps .py/.sh/.rs for the two things that split her state: the
    pin-file name and XDG_DATA_HOME used for the *data dir*. XDG for
    non-state assets (man pages, model candidates) is allowed and
    allowlisted by file below.
    """
    # Files whose XDG_DATA_HOME use is for non-state assets only —
    # each was verified by hand when added here.
    allowed = {
        "config.py",  # vosk_model_dir asset candidates, not the data dir
        "man_review.py",  # resolves $XDG_DATA_HOME/man for man pages
        "learning.py",  # same — man-page location, never the data dir
        "install.sh",  # MAN_HOME only — the user man dir genuinely is XDG
        "data_dir.rs",  # docstring documents the mechanism's absence
        "test_data_dir_conformance.py",  # this file
    }

    suspicious: list[str] = []
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".rs"}:
            continue
        rel = path.relative_to(REPO)
        parts = rel.parts
        if parts[0] in {"target", "build", ".git", "tests"} or parts[0].startswith(".venv"):
            continue
        if parts[0] == "python" and parts[1:2] == ("tests",) and rel.name in allowed:
            continue
        if rel.name in allowed:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if PIN_NAME in text:
            suspicious.append(f"{rel} (reads {PIN_NAME})")
        elif "XDG_DATA_HOME" in text:
            suspicious.append(f"{rel} (uses XDG_DATA_HOME)")

    assert not suspicious, (
        "These files reference the pin file or XDG_DATA_HOME — the fork "
        "mechanism must not be reintroduced. If the reference is for a "
        "non-state asset, verify it by hand and allowlist the file: "
        f"{suspicious}"
    )


def test_only_one_inbound_listener() -> None:
    """The daemon's Unix socket is the only outside entrance — ever.

    Everything that reaches the organism — mind, CLI, tools — connects
    to the one UnixListener bound at <data_dir>/genesis.sock. A second
    inbound listener (Unix or TCP, any layer) would be a second door
    into the body and must never appear. Outbound sockets (the local
    IP probe, websocket clients) are fine — they cannot accept a
    connection. The optic nerve's shared memory is afferent data, not an
    entrance: no command can travel it, and the gate on it arrives
    through the socket.
    """
    binds: list[str] = []
    listens: list[str] = []
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in {".rs", ".py"}:
            continue
        rel = path.relative_to(REPO)
        parts = rel.parts
        if (
            parts[0] in {"target", "build", ".git", "tests"}
            or parts[0].startswith(".venv")
            or rel.name == "test_data_dir_conformance.py"
        ):
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        # Any listener bind anywhere in the body crate or the mind —
        # there may only ever be the one in daemon/ipc.rs.
        for line_no, line in enumerate(text.splitlines(), 1):
            if "UnixListener::bind" in line or "TcpListener" in line:
                binds.append(str(rel))
            # Python-side socket .listen( — a bare word followed by
            # the call, so recognizer self.listen(...) can't match.
            if ".listen(" in line and "socket" in line.lower():
                listens.append(f"{rel}:{line_no}")

    assert binds == ["src/daemon/ipc.rs"], (
        "Inbound listener binds moved or multiplied — the organism has "
        f"exactly one entrance, <data_dir>/genesis.sock: {binds}"
    )
    assert not listens, (
        "A Python file opens a listening socket — inbound connections "
        f"only ever arrive through the daemon's Unix socket: {listens}"
    )
