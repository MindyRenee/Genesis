import fcntl
import os
import signal
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

import genesis_cli
from genesis_cognitive.mind.lifecycle import LifecycleMixin


def test_shutdown_does_not_force_kill_slow_daemon(tmp_path):
    mind = Mock()
    daemon = Mock()
    daemon.wait.side_effect = [subprocess.TimeoutExpired("daemon", 5), 0]
    daemon.returncode = 0
    with patch.object(genesis_cli, "_remove_pid_file"):
        assert genesis_cli._shutdown(None, mind, daemon, data_dir=str(tmp_path))
    mind.stop.assert_called_once()
    daemon.terminate.assert_called_once()
    daemon.kill.assert_not_called()
    assert daemon.wait.call_count == 2


def test_shutdown_does_not_force_kill_slow_retina():
    retina = Mock()
    retina.wait.side_effect = [subprocess.TimeoutExpired("retina", 3), 0]
    retina.returncode = 0
    assert genesis_cli._shutdown(None, Mock(), None, retina)
    retina.kill.assert_not_called()
    assert retina.wait.call_count == 2


def test_shutdown_reports_failed_daemon_exit():
    daemon = Mock()
    daemon.returncode = 1
    assert not genesis_cli._shutdown(None, Mock(), daemon)
    daemon.terminate.assert_called_once()
    daemon.wait.assert_called_once()


def test_repeated_shutdown_signal_does_not_interrupt_save():
    shutting_down = threading.Event()
    with patch.object(genesis_cli.signal, "signal") as install:
        genesis_cli._setup_signal_handlers(shutting_down)
    handler = install.call_args_list[0].args[1]
    with pytest.raises(KeyboardInterrupt):
        handler(signal.SIGINT, None)
    assert shutting_down.is_set()
    try:
        handler(signal.SIGINT, None)
        handler(signal.SIGTERM, None)
    except KeyboardInterrupt:
        pytest.fail("Repeated signals interrupted shutdown")


def test_sigterm_interrupts_blocking_input():
    with patch.object(genesis_cli.signal, "signal") as install:
        genesis_cli._setup_signal_handlers(threading.Event())
    with pytest.raises(KeyboardInterrupt):
        install.call_args_list[0].args[1](signal.SIGTERM, None)


def test_failed_cli_lock_closes_descriptor(tmp_path):
    with (
        patch.object(genesis_cli.os, "open", return_value=123),
        patch.object(genesis_cli.fcntl, "flock", side_effect=BlockingIOError),
        patch.object(genesis_cli.os, "close") as close,
        patch.object(genesis_cli, "_CLI_LOCK_FD", None),
    ):
        assert genesis_cli._acquire_cli_lock(str(tmp_path)) is None
        close.assert_called_once_with(123)
        assert genesis_cli._CLI_LOCK_FD is None


def test_failed_restore_cannot_be_saved_during_shutdown():
    mind = Mock(spec=LifecycleMixin)
    mind._running = False
    mind.client = Mock()

    def connect():
        mind._running = True

    mind._start_connect_and_register.side_effect = connect
    mind._start_restore_state.side_effect = OSError("invalid cognitive_state")
    with pytest.raises(OSError):
        LifecycleMixin.start(mind)
    assert not mind._running
    mind._start_autonomous_subsystems.assert_not_called()


def _run_shell_helpers(code, data_dir):
    source = (Path(__file__).resolve().parents[2] / "run.sh").read_text()
    helpers = source.split("# ─── Parse arguments", 1)[0]
    return subprocess.run(
        ["bash", "-c", helpers + "\n" + code],
        env={**os.environ, "XDG_DATA_HOME": str(data_dir)},
        capture_output=True, text=True, timeout=5,
    )


@pytest.mark.parametrize("pid", ["0", "-1", "1", "not-a-pid"])
def test_stop_rejects_invalid_pid_files(tmp_path, pid):
    data_dir = tmp_path / "genesis-public"
    data_dir.mkdir()
    (data_dir / "genesis_cli.pid").write_text(pid)
    result = _run_shell_helpers(
        'kill() { return 0; }; _read_pid "$DATA_DIR/genesis_cli.pid"', tmp_path,
    )
    assert result.returncode != 0
    assert not result.stdout.strip()


def test_stop_timeout_preserves_locks_and_never_sigkills(tmp_path):
    data_dir = tmp_path / "genesis-public"
    data_dir.mkdir()
    lock = data_dir / "genesis.lock"
    lock.write_text("held")
    with lock.open() as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run_shell_helpers(
            '''
_read_pid() { echo 4242; }
_matches_pid() { return 0; }
kill() { echo "signal $*"; return 0; }
pgrep() { return 1; }
sleep() { SECONDS=$((SECONDS + 200)); }
_stop_all
''', tmp_path,
        )
    assert result.returncode != 0
    assert "KILL" not in result.stdout
    assert lock.read_text() == "held"


def test_stop_rejects_reused_pid_from_unrelated_process(tmp_path):
    data_dir = tmp_path / "genesis-public"
    data_dir.mkdir()
    (data_dir / "genesis_cli.pid").write_text(str(os.getpid()))
    result = _run_shell_helpers('_read_pid "$DATA_DIR/genesis_cli.pid"', tmp_path)
    assert result.returncode != 0


@pytest.mark.parametrize("name", ["data with spaces", "data.[test]+"])
@pytest.mark.parametrize("suffix", ["", "-other"])
def test_process_identity_requires_exact_data_directory(tmp_path, name, suffix):
    result = _run_shell_helpers(
        '''
mapfile() { args=(python3 python/genesis_cli.py --data-dir "$DATA_DIR"''' + suffix + '''); }
_matches_pid "$$" "$DATA_DIR/genesis_cli.pid"
''', tmp_path / name,
    )
    assert (result.returncode == 0) == (suffix == "")


def test_shell_shutdown_orders_cli_before_children(tmp_path):
    result = _run_shell_helpers(
        '''
_find_pid() {
    case "$1" in
        *cli.pid) echo 100 ;;
        *daemon.pid) echo 200 ;;
        *retina.pid) echo 300 ;;
    esac
}
_matches_pid() { return 1; }
kill() { echo "$*"; }
_stop_all
''', tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["-INT 100", "-TERM 200", "-TERM 300"]


@pytest.mark.parametrize("raw", [b"{", b'{"version":2,"reflection":null}'])
def test_failed_mind_start_preserves_saved_state(tmp_path, raw):
    from genesis_cognitive.mind import Mind

    path = tmp_path / "cognitive_state.json"
    path.write_bytes(raw)
    mind = Mind(str(tmp_path / "genesis.sock"), offline=True)
    mind.client = Mock()
    with pytest.raises(OSError, match="cognitive_state"):
        mind.start()
    mind.stop()
    assert not mind._running
    assert path.read_bytes() == raw
    mind.client.disconnect.assert_called_once()


def test_daemon_start_timeout_reaps_child(tmp_path):
    proc = Mock()
    proc.poll.return_value = None
    proc.pid = 4242
    with (
        patch.object(genesis_cli.subprocess, "Popen", return_value=proc),
        patch.object(genesis_cli, "DAEMON_START_TIMEOUT", 0),
    ):
        with pytest.raises(RuntimeError, match="didn't start"):
            genesis_cli._start_daemon("/unused/daemon", str(tmp_path), str(tmp_path / "sock"))
    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()
    proc.kill.assert_not_called()


def test_interactive_failure_still_shuts_down(tmp_path):
    cleanup = Mock()
    with (
        patch.dict(os.environ, {"GENESIS_RUN": "1"}),
        patch.multiple(
            genesis_cli,
            _parse_cli_args=Mock(return_value=Mock(
                data_dir=str(tmp_path), tail=False, daemon_socket=None, offline=True,
            )),
            _acquire_cli_lock=Mock(return_value=1),
            _start_daemon_process=Mock(),
            _start_retina_process=Mock(),
            Mind=Mock(),
            _wire_voice=Mock(return_value=(Mock(), Mock())),
            _setup_ambient_listener=Mock(),
            _setup_auditory_cortex=Mock(),
            _print_welcome=Mock(),
            _setup_signal_handlers=Mock(),
            _start_sleep_watcher=Mock(),
            _run_interactive_loop=Mock(side_effect=RuntimeError("input failure")),
            _shutdown=cleanup,
        ),
    ):
        with pytest.raises(RuntimeError, match="input failure"):
            genesis_cli.main()
    cleanup.assert_called_once()


@pytest.mark.parametrize("failing_step", [
    "_wire_voice",
    "_setup_ambient_listener",
    "_setup_auditory_cortex",
    "_print_welcome",
    "_start_sleep_watcher",
])
def test_startup_failure_still_shuts_down(tmp_path, failing_step):
    cleanup = Mock(return_value=True)
    mind = Mock()
    patches = {
        "_wire_voice": Mock(return_value=(Mock(), Mock())),
        "_setup_ambient_listener": Mock(),
        "_setup_auditory_cortex": Mock(),
        "_print_welcome": Mock(),
        "_setup_signal_handlers": Mock(),
        "_start_sleep_watcher": Mock(),
        "_run_interactive_loop": Mock(),
    }
    patches[failing_step] = Mock(side_effect=RuntimeError("setup failed"))
    with (
        patch.dict(os.environ, {"GENESIS_RUN": "1"}),
        patch.multiple(
            genesis_cli,
            _parse_cli_args=Mock(return_value=Mock(
                data_dir=str(tmp_path), tail=False, daemon_socket=None,
                offline=True,
            )),
            _acquire_cli_lock=Mock(return_value=1),
            _start_daemon_process=Mock(),
            _start_retina_process=Mock(),
            Mind=Mock(return_value=mind),
            _shutdown=cleanup,
            **patches,
        ),
    ):
        with pytest.raises(RuntimeError, match="setup failed"):
            genesis_cli.main()
    cleanup.assert_called_once()
    assert cleanup.call_args.args[1] is mind


def test_shutdown_reports_failed_final_save(tmp_path):
    from genesis_cognitive.mind import Mind

    mind = Mind(str(tmp_path / "genesis.sock"), offline=True)
    mind._running = True
    mind.client = Mock()
    mind._save_state = Mock(return_value=False)

    assert mind.stop() is False
    mind.client.disconnect.assert_called_once()


def test_shutdown_waits_for_autosave_before_saving(tmp_path):
    from genesis_cognitive.mind import Mind

    mind = Mind(str(tmp_path / "genesis.sock"), offline=True)
    mind._running = True
    mind.client = Mock()
    worker_can_exit = threading.Event()
    saved = threading.Event()

    def worker():
        worker_can_exit.wait(timeout=10)

    mind._autosave_thread = threading.Thread(target=worker, daemon=True)
    mind._autosave_thread.start()
    mind._save_state = Mock(side_effect=lambda: saved.set() or True)

    stopper = threading.Thread(target=mind.stop, daemon=True)
    stopper.start()
    assert not saved.wait(timeout=0.2)
    worker_can_exit.set()
    stopper.join(timeout=10)
    assert not stopper.is_alive()
    assert saved.is_set()
