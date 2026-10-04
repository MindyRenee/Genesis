"""Every data-dir resolver must return the same answer.

Genesis resolves its state directory in a dozen places: the launcher, the
maintenance scripts, the installer, the Python side, and the Rust
examples. Each was written independently, and they drifted. That drift is
not cosmetic — when a launcher (Flatpak, a sandboxed agent host) sets
``XDG_DATA_HOME`` to its own private directory, a resolver that honors it
attaches the checkout to a *second*, empty state directory while the rest
of the tooling reads the first. Two instances then accumulate divergent
concepts, edges, and journals.

That is not hypothetical: this checkout forked for six days, ~8.7k
concepts and 8 projects landing in a directory under ``~/.var/app/``
that nothing was monitoring, while the original looked (and reported) as
though it had simply never archived anything.

So the rule is one ordered list, and this module is what enforces it:

1. ``GENESIS_DATA_DIR`` — explicit override, always wins.
2. ``.genesis-data-dir`` — the checkout's uncommitted pin. Its contents
   are the data dir itself, not a parent of it.
3. ``$XDG_DATA_HOME/genesis``.
4. ``~/.local/share/genesis``.

The pin outranks ``XDG_DATA_HOME`` precisely because ``XDG_DATA_HOME`` is
ambient environment, not Genesis configuration.

If this file starts failing after an unrelated change, a resolver drifted
— fix the resolver, not this test. The Rust side has the mirrored test
suite in ``src/data_dir.rs``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PIN_NAME = ".genesis-data-dir"

# A checkout that is not this one. Each case below combines a pin, an
# explicit override, and a hostile ambient XDG so that any resolver which
# honors the wrong input disagrees with the others.
PINNED = "/state/pinned-checkout"
HOSTILE_XDG = "/sandbox/private/data"


def _run(cmd: list[str], env: dict[str, str], cwd: Path | None = None) -> str:
    """Run a command, returning stdout, and fail loudly on error."""
    proc = subprocess.run(
        cmd, env=env, cwd=cwd or REPO, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"{cmd} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}"
    )
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


def resolve_run_sh(env: dict[str, str]) -> str:
    """run.sh — the launcher, and the authority for the order."""
    source = (REPO / "run.sh").read_text()
    # The resolution lives before argument parsing; everything after it
    # starts or stops processes.
    header = source.split("# ─── Parse arguments", 1)[0]
    out = _run(
        ["bash", "-c", f'{header}\nprintf "%s" "$DATA_DIR"'],
        env,
    )
    return out.strip()


def resolve_hygiene_sh(env: dict[str, str]) -> str:
    """scripts/hygiene.sh — decides what it cleans."""
    out = _run([str(REPO / "scripts/hygiene.sh"), "--data-dir"], env)
    return out.strip()


def resolve_prune_dead_concepts(env: dict[str, str]) -> str:
    """scripts/prune_dead_concepts.py — deletes concepts."""
    return _python_default(REPO / "scripts/prune_dead_concepts.py", env, "_default_data_dir")


def resolve_migrate_edges(env: dict[str, str]) -> str:
    """scripts/migrate_edges_to_log.py — rewrites the canonical edge log."""
    return _python_default(
        REPO / "scripts/migrate_edges_to_log.py", env, "_default_data_dir",
    )


def resolve_config_py(env: dict[str, str]) -> str:
    """infrastructure/config.py — the in-process default."""
    out = _run(
        [
            "python3", "-c",
            "import sys; sys.path.insert(0, 'python');"
            "from genesis_conscious.infrastructure.config import default_data_dir;"
            "print(default_data_dir())",
        ],
        env,
    )
    return out.strip()


def resolve_setup_embeddings(env: dict[str, str]) -> str:
    """python/setup_embeddings.py — builds an index over the concept set."""
    out = _run(
        [
            "python3", "-c",
            "import importlib.util, sys;"
            "sys.path.insert(0, 'python');"
            "spec = importlib.util.spec_from_file_location("
            "    'se', 'python/setup_embeddings.py');"
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


def _python_default(script: Path, env: dict[str, str], attr: str) -> str:
    """Read a module-level default resolver without running its main()."""
    out = _run(
        [
            "python3", "-c",
            f"import importlib.util, sys;"
            f"sys.path.insert(0, 'python');"
            f"spec = importlib.util.spec_from_file_location('m', {str(script)!r});"
            f"m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m);"
            f"print(getattr(m, {attr!r})())",
        ],
        env,
    )
    return out.strip()


# The resolvers that answer "which data directory?", i.e. those that must
# agree exactly with run.sh.
DIR_RESOLVERS = {
    "run.sh": resolve_run_sh,
    "hygiene.sh": resolve_hygiene_sh,
    "prune_dead_concepts.py": resolve_prune_dead_concepts,
    "migrate_edges_to_log.py": resolve_migrate_edges,
    "config.py": resolve_config_py,
    "setup_embeddings.py": resolve_setup_embeddings,
}

# Asset resolvers legitimately answer "which directory holds <asset>?",
# so they are the data dir plus a subpath — checked separately.
ASSET_RESOLVERS = {
    "install_optional_deps.sh (vosk)": (resolve_vosk_dir_shell, "vosk-models"),
}


# ── Cases ─────────────────────────────────────────────────────────


def _cases() -> list[tuple[str, dict[str, str | None], str]]:
    """(name, env overrides, expected data dir).

    Only pin-present configurations are exercised against the real
    resolvers, because a pin cannot be relocated for them: the shell ones
    resolve ``.genesis-data-dir`` against the repository root and the
    Python ones against ``__file__``. That is also the configuration that
    actually broke — the fork only ever happened to a pinned checkout.

    The unpinned fallbacks are covered exhaustively by the pure-function
    tests in ``src/data_dir.rs``, which need no repository at all.
    """
    return [
        (
            "hostile XDG_DATA_HOME must not beat the pin",
            {"XDG_DATA_HOME": HOSTILE_XDG},
            PINNED,
        ),
        (
            "explicit GENESIS_DATA_DIR beats the pin",
            {"GENESIS_DATA_DIR": "/state/explicit", "XDG_DATA_HOME": HOSTILE_XDG},
            "/state/explicit",
        ),
        (
            "blank GENESIS_DATA_DIR is treated as unset, so the pin wins",
            {"GENESIS_DATA_DIR": "   ", "XDG_DATA_HOME": HOSTILE_XDG},
            PINNED,
        ),
        (
            "no XDG_DATA_HOME at all still resolves to the pin",
            {"XDG_DATA_HOME": None},
            PINNED,
        ),
    ]


def _real_pin() -> str:
    """The value this checkout's pin file holds."""
    text = (REPO / PIN_NAME).read_text().splitlines()
    assert text, f"{PIN_NAME} is empty; it must name the data directory"
    return text[0].strip()


# Every case below assumes this checkout is pinned, so read the real value
# once and substitute it, rather than hard-coding a path that goes stale
# when the pin changes.
PINNED = _real_pin()


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
    """Every resolver returns run.sh's answer, under every environment."""
    env = _base_env(**dict(overrides))

    results: dict[str, str] = {}
    for name, resolver in DIR_RESOLVERS.items():
        results[name] = resolver(env)
        assert results[name] == expected, (
            f"{name} disagrees under '{case}'.\n"
            f"  expected: {expected}\n"
            f"  got:      {results[name]}\n"
            f"  env:      GENESIS_DATA_DIR={env.get('GENESIS_DATA_DIR')!r} "
            f"XDG_DATA_HOME={env.get('XDG_DATA_HOME')!r}\n"
            f"  Every resolver must implement run.sh's order; see the module "
            f"docstring for what that order is and why the pin outranks "
            f"XDG_DATA_HOME."
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
    """Asset directories sit inside the resolved data dir, not beside it."""
    env = _base_env(**dict(overrides))

    for name, (resolver, subdir) in ASSET_RESOLVERS.items():
        got = resolver(env)
        assert got == f"{expected}/{subdir}", (
            f"{name} points outside the data dir under '{case}'.\n"
            f"  expected: {expected}/{subdir}\n"
            f"  got:      {got}\n"
            f"  An asset read from another instance's directory is the bug "
            f"that made this checkout's microphone depend on a sibling "
            f"instance having a model."
        )


def test_every_resolver_in_the_repo_is_covered() -> None:
    """A new resolver must be added here, or this test is incomplete.

    Greps the tree for anything that mentions the state directory and
    fails if it is not one of the resolvers exercised above. Without this,
    someone could add a tenth resolver, get the precedence wrong, and
    the other nine would still agree — so the suite would stay green
    while the bug came back.
    """
    known = {
        "run.sh",
        "hygiene.sh",
        "prune_dead_concepts.py",
        "migrate_edges_to_log.py",
        "install_optional_deps.sh",
        "config.py",
        "setup_embeddings.py",
        "read_state.rs",
        "recover.rs",
        "merge_data_dirs.py",
        "data_dir.rs",
        "test_data_dir_conformance.py",
        "genesis_cli.py",
        "ltm_index.py",
    }
    # Files that reference the location without resolving it: this test,
    # the Rust resolver (mirrored test suite), and a client docstring
    # whose example hardcodes a socket path in a usage example.
    allowed = {
        "test_data_dir_conformance.py",
        "data_dir.rs",
        "__init__.py",  # genesis_client usage example in the module docstring
    }

    suspicious: list[str] = []
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".sh", ".rs"}:
            continue
        rel = path.relative_to(REPO)
        parts = rel.parts
        if parts[0] in {"target", ".git", "tests", "python"} and parts[0] != "python":
            continue
        if parts[0] == "python" and parts[1:2] == ("tests",):
            continue
        if rel.name in allowed or rel.name in known:
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if ".local/share/genesis" in text or "XDG_DATA_HOME" in text:
            suspicious.append(str(rel))

    assert not suspicious, (
        "These files reference the state directory but are not covered by "
        "this conformance test. If one of them resolves the directory, add "
        f"it to DIR_RESOLVERS or ASSET_RESOLVERS: {suspicious}"
    )


def test_pin_file_contents_are_the_dir_not_a_parent() -> None:
    """A pin naming a parent directory is a config error worth catching.

    run.sh assigns the pin straight to GENESIS_DATA_DIR, so a trailing
    slash or a `.../share` value silently becomes the data dir. Cheap to
    assert the shape the rest of the system assumes.
    """
    value = _real_pin()
    assert value == PINNED
    assert not value.endswith("/"), "a trailing slash is a parent, not a dir"
    assert os.path.isabs(value), "the pin must be an absolute path"
