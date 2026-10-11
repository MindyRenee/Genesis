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


def test_package_root_is_the_python_dir() -> None:
    # This test file lives in python/tests/, so parents[1] is python/.
    assert package_root() == Path(__file__).resolve().parents[1]


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
        "# comment\n"
        "GENESIS_TEST_KEY = from_file\n"
        'GENESIS_QUOTED = "quoted value"\n'
        "not a line\n"
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


def test_stop_all_kills_tracked_process(tmp_path) -> None:
    proc = subprocess.Popen(["sleep", "60"])
    try:
        (tmp_path / "genesis_cli.pid").write_text(str(proc.pid))
        (tmp_path / "genesis.sock").touch()
        (tmp_path / "genesis.lock").touch()
        assert genesis_cli._any_running(str(tmp_path))
        assert genesis_cli._stop_all(str(tmp_path)) == 0
        deadline = time.monotonic() + 5
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert proc.poll() is not None
        assert not (tmp_path / "genesis.sock").exists()
        assert not (tmp_path / "genesis_cli.pid").exists()
        assert not genesis_cli._any_running(str(tmp_path))
    finally:
        proc.kill()
        proc.wait()


def test_stop_all_nothing_running(tmp_path) -> None:
    assert not genesis_cli._any_running(str(tmp_path))
    assert genesis_cli._stop_all(str(tmp_path)) == 0
