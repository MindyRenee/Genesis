"""Isolated-daemon harness for baseline and assay work.

Every capture runs against a throwaway daemon in a temporary data
directory — never the production instance — so settling, impulses,
and washouts cannot corrupt real developmental state. Mirrors the
spawn/wait/teardown pattern of ``scripts/test_integration.py``.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket as _socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager

from genesis_client import GenesisClient
from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)

_DAEMON_CANDIDATES = ("target/release/genesis-daemon", "target/debug/genesis-daemon")


def find_daemon_binary(project_root: str | None = None) -> str | None:
    """Locate the genesis-daemon binary, or None if not built.

    Checks ``GENESIS_DAEMON`` first, then the release/debug build
    outputs relative to the project root.
    """
    env = os.environ.get("GENESIS_DAEMON")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    root = project_root or _project_root()
    for rel in _DAEMON_CANDIDATES:
        path = os.path.abspath(os.path.join(root, rel))
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _project_root() -> str:
    here = os.path.abspath(__file__)
    # .../python/genesis_cognitive/eval/harness.py -> .../ (project root)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))


def wait_for_socket(socket_path: str, timeout: float = 10.0) -> bool:
    """Wait until the daemon's socket accepts connections."""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(socket_path):
            try:
                s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(socket_path)
                s.close()
                return True
            except OSError as e:
                # Socket file exists but daemon isn't accepting yet — keep polling.
                note_swallowed(
                    "genesis_cognitive.eval.harness.wait_for_socket",
                    e,
                )
        time.sleep(0.1)
    return os.path.exists(socket_path)


def settle(client: GenesisClient, *, ticks: int = 60, dt: float = 1.0) -> None:
    """Advance neurochemistry until transients decay to resting state."""
    for _ in range(ticks):
        client.advance_physics(dt)


@contextmanager
def isolated_daemon(
    *,
    stm_capacity: int = 64,
    ltm_capacity: int = 256,
    project_root: str | None = None,
) -> Iterator[GenesisClient]:
    """Spawn a throwaway daemon and yield a connected client.

    Uses a fresh temporary data directory. On exit, shuts the daemon
    down via IPC and removes the directory. The force-kill fallback
    applies only to this disposable test daemon — never to a
    production instance carrying developmental state.
    """
    binary = find_daemon_binary(project_root)
    if binary is None:
        raise FileNotFoundError(
            "genesis-daemon binary not found (run: cargo build --release)"
        )
    data_dir = tempfile.mkdtemp(prefix="genesis_eval_")
    socket_path = os.path.join(data_dir, "genesis.sock")
    proc = subprocess.Popen(
        [binary, "--data-dir", data_dir, "--stm-capacity", str(stm_capacity),
         "--ltm-capacity", str(ltm_capacity)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    client = GenesisClient(socket_path)
    try:
        if not wait_for_socket(socket_path):
            proc.kill()
            proc.wait()
            raise TimeoutError("daemon did not create its socket in time")
        time.sleep(0.3)
        client.connect()
        yield client
    finally:
        try:
            client.disconnect()
        except OSError as e:
            note_swallowed(
                "genesis_cognitive.eval.harness.isolated_daemon",
                e,
            )
        try:
            killer = GenesisClient(socket_path)
            killer.connect()
            killer.shutdown()
            killer.disconnect()
            proc.wait(timeout=5.0)
        except Exception:  # noqa: BLE001
            # Disposable temp-dir daemon only: avoid hanging the harness
            # forever when graceful shutdown fails.
            proc.kill()
            proc.wait()
        shutil.rmtree(data_dir, ignore_errors=True)
