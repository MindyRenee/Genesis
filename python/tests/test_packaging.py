"""Packaging and console-entry tests.

The installed ``genesis`` command is the packaged equivalent of
run.sh: it sets GENESIS_RUN itself, loads the deployed env file, and
owns ``--headless``/``--stop``. These tests pin that contract plus the
install-layout path resolution in infrastructure/paths.py.
"""

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import genesis_cli
from infrastructure.paths import code_root, package_root


def test_package_root_is_the_repo_root() -> None:
    # The package root is the directory holding the top-level packages:
    # the repo root in a checkout (python/ plus the support packages),
    # site-packages in an install.
    assert package_root() == Path(__file__).resolve().parents[2]


def test_code_root_is_the_repo_root() -> None:
    assert code_root() == Path(__file__).resolve().parents[2]


def test_code_root_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GENESIS_ROOT", str(tmp_path))
    assert code_root() == tmp_path


def test_parse_headless_and_stop(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["genesis", "--headless", "--stop"])
    args = genesis_cli._parse_cli_args()
    assert args.headless and args.stop


def test_console_main_sets_genesis_run(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GENESIS_RUN", raising=False)
    monkeypatch.setenv("GENESIS_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setattr(genesis_cli, "main", lambda: 7)
    assert genesis_cli._console_main() == 7
    assert os.environ["GENESIS_RUN"] == "1"


def test_agent_env_loads_and_real_env_wins(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / "agent.env"
    env_file.write_text(
        '# comment\nGENESIS_TEST_KEY = from_file\nGENESIS_QUOTED = "quoted value"\nnot a line\n'
    )
    monkeypatch.setenv("GENESIS_ENV_FILE", str(env_file))
    monkeypatch.setenv("GENESIS_TEST_KEY", "from_env")
    monkeypatch.delenv("GENESIS_QUOTED", raising=False)
    genesis_cli._load_agent_env()
    assert os.environ["GENESIS_TEST_KEY"] == "from_env"
    assert os.environ["GENESIS_QUOTED"] == "quoted value"


def test_find_binary_env_dir(monkeypatch, tmp_path) -> None:
    fake = tmp_path / "genesis-daemon"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("GENESIS_BIN_DIR", str(tmp_path))
    assert genesis_cli._find_binary("genesis-daemon") == str(fake)


def test_find_binary_env_dir_missing_falls_back(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GENESIS_BIN_DIR", str(tmp_path / "empty"))
    # Falls through to the checkout's target/release, which exists.
    assert genesis_cli._find_binary("genesis-daemon") is not None


def test_pid_from_file_dead_pid(tmp_path) -> None:
    pidfile = tmp_path / "x.pid"
    # One below the common pid_max ceiling — legal, never assigned
    # in practice on a test host.
    pidfile.write_text("4194303")
    assert genesis_cli._pid_from_file(str(pidfile)) is None


def test_stop_all_never_signals_process_failing_identity(tmp_path) -> None:
    """A pidfile naming a foreign process is stale data, not a target.

    PID reuse means the number alone proves nothing — a ``sleep`` that
    inherited our dead CLI's pid must never be signaled. The file is
    swept as stale; the process is untouched.
    """
    proc = subprocess.Popen(["sleep", "60"])
    try:
        (tmp_path / "genesis_cli.pid").write_text(str(proc.pid))
        (tmp_path / "genesis.sock").touch()
        (tmp_path / "genesis.lock").touch()
        assert not genesis_cli._any_running(str(tmp_path))
        assert genesis_cli._stop_all(str(tmp_path)) == 0
        assert proc.poll() is None  # never signaled — still sleeping
        assert not (tmp_path / "genesis.sock").exists()
        assert not (tmp_path / "genesis_cli.pid").exists()
    finally:
        proc.kill()
        proc.wait()


def _fake_cli(data_dir: Path, sigint_disposition: str) -> subprocess.Popen:
    """A real process whose argv passes the CLI identity check.

    Writes a ready file only after its signal handler is installed —
    without the barrier a fast _stop_all could SIGINT the child during
    interpreter startup, before the disposition exists.
    """
    cli = data_dir / "genesis_cli.py"
    ready = data_dir / "ready"
    cli.write_text(
        "import signal, sys, time\n"
        f"{sigint_disposition}\n"
        f"open({str(ready)!r}, 'w').close()\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, str(cli), "--data-dir", str(data_dir)])
    (data_dir / "genesis_cli.pid").write_text(str(proc.pid))
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready.exists(), "fake CLI never installed its handler"
    return proc


def test_stop_all_sigints_validated_cli_and_waits(tmp_path) -> None:
    """The graceful path: SIGINT a verified CLI and wait out its save.

    No grace-window shortcut, no SIGKILL — the process exits on its own
    and its pidfile is swept by the owner that no longer exists.
    """
    proc = _fake_cli(
        tmp_path,
        "signal.signal(signal.SIGINT, lambda *a: sys.exit(0))",
    )
    try:
        assert genesis_cli._stop_all(str(tmp_path)) == 0
        assert proc.wait(timeout=10) == 0
        assert not (tmp_path / "genesis_cli.pid").exists()
    finally:
        proc.kill()
        proc.wait()


def test_stop_all_timeout_preserves_files_and_never_kills(tmp_path, monkeypatch) -> None:
    """A CLI still saving at the deadline is a 'retry later', not a kill.

    run.sh's contract, which the old 5s+SIGKILL path violated: files and
    locks stay in place so the in-progress save can finish on its own,
    and the process is left running — interrupting a save is the one
    thing the teardown must never do.
    """
    monkeypatch.setattr(genesis_cli, "_STOP_STAGE_GRACE_S", 0.2)
    lock = tmp_path / "genesis.lock"
    lock.write_text("held")
    proc = _fake_cli(
        tmp_path,
        "signal.signal(signal.SIGINT, signal.SIG_IGN)",
    )
    try:
        assert genesis_cli._stop_all(str(tmp_path)) == 1
        # It ignored SIGINT and was never escalated — still running.
        assert proc.poll() is None
        assert (tmp_path / "genesis_cli.pid").exists()
        assert lock.read_text() == "held"
    finally:
        proc.kill()
        proc.wait()


def test_stop_all_nothing_running(tmp_path) -> None:
    assert not genesis_cli._any_running(str(tmp_path))
    assert genesis_cli._stop_all(str(tmp_path)) == 0
