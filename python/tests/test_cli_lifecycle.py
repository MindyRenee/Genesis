import fcntl
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from brainstem.lifecycle import LifecycleMixin

import genesis_cli


def test_shutdown_waits_for_slow_daemon_within_grace_window(tmp_path):
    """A daemon that is slow but alive is NOT force-killed.

    Preserves the original intent: killing risks losing unconsolidated
    memory and leaving on-disk state inconsistent, so SIGKILL is not
    used while the child might still be flushing. `wait.side_effect`
    models a first wait that exceeds the grace window and a second that
    succeeds, i.e. the child did exit on its own.
    """
    mind = Mock()
    daemon = Mock()
    daemon.poll.return_value = None  # still running when shutdown starts
    daemon.wait.side_effect = [subprocess.TimeoutExpired("daemon", 5), 0]
    daemon.returncode = 0
    with patch.object(genesis_cli, "_remove_pid_file"):
        assert genesis_cli._shutdown(None, mind, daemon, data_dir=str(tmp_path))
    mind.stop.assert_called_once()
    daemon.terminate.assert_called_once()
    assert daemon.wait.call_count == 2


def test_shutdown_escalates_to_kill_when_child_never_exits(tmp_path):
    """A child that ignores SIGTERM is escalated, not waited on forever.

    The previous implementation called `proc.wait()` with no timeout
    once the first wait expired, so a wedged daemon blocked shutdown
    indefinitely. Combined with a signal handler that swallowed every
    signal after shutdown began, the process could not be stopped by
    Ctrl-C or SIGTERM at all — leaving `kill -9` as the only exit, the
    one action the operational notes forbid.

    SIGKILL is still the last resort: SIGTERM (or the graceful SHUTDOWN
    IPC request) is always tried first, and only a child that has
    demonstrably ignored it is killed.
    """
    mind = Mock()
    daemon = Mock()
    daemon.poll.return_value = None
    daemon.wait.side_effect = [
        subprocess.TimeoutExpired("daemon", 5),  # ignored SIGTERM
        0,  # exited after SIGKILL
    ]
    daemon.returncode = 0
    with patch.object(genesis_cli, "_remove_pid_file"):
        assert genesis_cli._shutdown(None, mind, daemon, data_dir=str(tmp_path))
    daemon.terminate.assert_called_once()
    daemon.kill.assert_called_once()
    assert daemon.wait.call_count == 2


def test_shutdown_reports_failure_when_child_cannot_be_killed(tmp_path):
    """A child stuck in uninterruptible I/O is reported, not hung on.

    The mind has already saved state by this point, so continuing the
    teardown is the safer failure mode than blocking forever.
    """
    mind = Mock()
    daemon = Mock()
    daemon.poll.return_value = None
    daemon.wait.side_effect = subprocess.TimeoutExpired("daemon", 5)
    daemon.returncode = None
    with patch.object(genesis_cli, "_remove_pid_file"):
        assert not genesis_cli._shutdown(None, mind, daemon, data_dir=str(tmp_path))
    daemon.kill.assert_called_once()


def test_shutdown_waits_for_slow_retina_within_grace_window():
    retina = Mock()
    retina.poll.return_value = None
    retina.wait.side_effect = [subprocess.TimeoutExpired("retina", 3), 0]
    retina.returncode = 0
    assert genesis_cli._shutdown(None, Mock(), None, retina)
    retina.terminate.assert_called_once()
    assert retina.wait.call_count == 2


def test_shutdown_reports_failed_daemon_exit():
    daemon = Mock()
    daemon.poll.return_value = None
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
        # GENESIS_DATA_DIR, not XDG_DATA_HOME: run.sh's per-checkout
        # .genesis-data-dir pin outranks XDG_DATA_HOME, so setting the
        # latter would leave DATA_DIR pointing at the developer's real
        # state directory and these tests would read its pid files.
        env={**os.environ, "GENESIS_DATA_DIR": str(Path(data_dir) / "genesis")},
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.parametrize("pid", ["0", "-1", "1", "not-a-pid"])
def test_stop_rejects_invalid_pid_files(tmp_path, pid):
    data_dir = tmp_path / "genesis"
    data_dir.mkdir()
    (data_dir / "genesis_cli.pid").write_text(pid)
    result = _run_shell_helpers(
        'kill() { return 0; }; _read_pid "$DATA_DIR/genesis_cli.pid"',
        tmp_path,
    )
    assert result.returncode != 0
    assert not result.stdout.strip()


def test_stop_timeout_preserves_locks_and_never_sigkills(tmp_path):
    data_dir = tmp_path / "genesis"
    data_dir.mkdir()
    lock = data_dir / "genesis.lock"
    lock.write_text("held")
    with lock.open() as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run_shell_helpers(
            """
_read_pid() { echo 4242; }
_matches_pid() { return 0; }
kill() { echo "signal $*"; return 0; }
pgrep() { return 1; }
sleep() { SECONDS=$((SECONDS + 200)); }
_stop_all
""",
            tmp_path,
        )
    assert result.returncode != 0
    assert "KILL" not in result.stdout
    assert lock.read_text() == "held"


def test_stop_rejects_reused_pid_from_unrelated_process(tmp_path):
    data_dir = tmp_path / "genesis"
    data_dir.mkdir()
    (data_dir / "genesis_cli.pid").write_text(str(os.getpid()))
    result = _run_shell_helpers('_read_pid "$DATA_DIR/genesis_cli.pid"', tmp_path)
    assert result.returncode != 0


@pytest.mark.parametrize("name", ["data with spaces", "data.[test]+"])
@pytest.mark.parametrize("suffix", ["", "-other"])
def test_process_identity_requires_exact_data_directory(tmp_path, name, suffix):
    result = _run_shell_helpers(
        '''
mapfile() { args=(python3 python/genesis_cli.py --data-dir "$DATA_DIR"'''
        + suffix
        + """); }
_matches_pid "$$" "$DATA_DIR/genesis_cli.pid"
""",
        tmp_path / name,
    )
    assert (result.returncode == 0) == (suffix == "")


def test_shell_shutdown_orders_cli_before_children(tmp_path):
    result = _run_shell_helpers(
        """
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
""",
        tmp_path,
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == ["-INT 100", "-TERM 200", "-TERM 300"]


@pytest.mark.parametrize("raw", [b"{", b'{"version":2,"reflection":null}'])
def test_failed_mind_start_preserves_saved_state(tmp_path, raw):
    from mind import Mind

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
        with pytest.raises(RuntimeError, match="didn't become ready") as exc:
            genesis_cli._start_daemon("/unused/daemon", str(tmp_path), str(tmp_path / "sock"))
    # The failure must point at the log: a bare "didn't start" hides
    # the actual initialization error, which is diagnosable only there.
    assert "daemon.log" in str(exc.value)
    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()
    proc.kill.assert_not_called()


def test_interactive_failure_still_shuts_down(tmp_path):
    cleanup = Mock()
    with (
        patch.dict(os.environ, {"GENESIS_RUN": "1"}),
        patch.multiple(
            genesis_cli,
            _parse_cli_args=Mock(
                return_value=Mock(
                    data_dir=str(tmp_path),
                    tail=False,
                    daemon_socket=None,
                    offline=True,
                    headless=False,
                    stop=False,
                    germinate=None,
                )
            ),
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
            # The post-teardown watchdog is a real 30 s daemon timer
            # ending in os._exit — unmocked it would kill the test
            # process half a minute later.
            _arm_exit_watchdog=Mock(),
        ),
    ):
        with pytest.raises(RuntimeError, match="input failure"):
            genesis_cli.main()
    cleanup.assert_called_once()


@pytest.mark.parametrize(
    "failing_step",
    [
        "_wire_voice",
        "_setup_ambient_listener",
        "_setup_auditory_cortex",
        "_print_welcome",
        "_start_sleep_watcher",
    ],
)
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
            _parse_cli_args=Mock(
                return_value=Mock(
                    data_dir=str(tmp_path),
                    tail=False,
                    daemon_socket=None,
                    offline=True,
                    headless=False,
                    stop=False,
                    germinate=None,
                )
            ),
            _acquire_cli_lock=Mock(return_value=1),
            _start_daemon_process=Mock(),
            _start_retina_process=Mock(),
            Mind=Mock(return_value=mind),
            _shutdown=cleanup,
            # See test_interactive_failure_still_shuts_down — the exit
            # watchdog is a real os._exit timer and must be mocked.
            _arm_exit_watchdog=Mock(),
            **patches,
        ),
    ):
        with pytest.raises(RuntimeError, match="setup failed"):
            genesis_cli.main()
    cleanup.assert_called_once()
    assert cleanup.call_args.args[1] is mind


def test_shutdown_reports_failed_final_save(tmp_path):
    from mind import Mind

    mind = Mind(str(tmp_path / "genesis.sock"), offline=True)
    mind._running = True
    mind.client = Mock()
    mind._save_state = Mock(return_value=False)

    assert mind.stop() is False
    mind.client.disconnect.assert_called_once()


def test_shutdown_waits_for_autosave_before_saving(tmp_path):
    from mind import Mind

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


def test_shutdown_stops_voice_before_mind_teardown(tmp_path):
    """Speech goes quiet before the long mind.stop() begins.

    In-flight and queued piper/aplay utterances outlive the urge that
    queued them; without an explicit voice.stop() she keeps talking
    through the ~25 s teardown. The voice is silenced first so her last
    word is not "Stopping cognitive mind...".
    """
    order = Mock()
    mind = Mock()
    voice = Mock()
    order.attach_mock(voice.stop, "voice_stop")
    order.attach_mock(mind.stop, "mind_stop")

    assert genesis_cli._shutdown(None, mind, None, data_dir=str(tmp_path), voice=voice)
    names = [c[0] for c in order.mock_calls]
    assert names.index("voice_stop") < names.index("mind_stop")


def test_shutdown_without_voice_still_stops_mind(tmp_path):
    """A text-only run (no mic, no piper) must not require a voice."""
    mind = Mock()
    assert genesis_cli._shutdown(None, mind, None, data_dir=str(tmp_path))
    mind.stop.assert_called_once()


# ─── Rebirth — the organism heals its own process tree ───────────────


def test_rejoin_body_reregisters_modules_zone_and_pid():
    """A reborn body gets the whole registration sequence back — its
    manifest starts empty, so rejoining re-runs what start() ran."""
    from genesis_client.protocol import MODULE_STATUS_STARTING, ZONE_CONVERSATION

    mind = Mock(spec=LifecycleMixin)
    mind._running = True
    mind._is_sleeping = False
    mind.client = Mock()
    mind.notifications = Mock()
    mind._registered_modules.return_value = (3, 4, 5)

    assert LifecycleMixin.rejoin_body(mind) is True

    mind.client.connect.assert_called_once()
    mind.client.set_cognitive_pid.assert_called_once_with(os.getpid())
    statuses = [c.args for c in mind.client.update_module_status.call_args_list]
    assert statuses == [(m, MODULE_STATUS_STARTING) for m in (3, 4, 5)]
    mind.client.set_zone.assert_called_once_with(ZONE_CONVERSATION)
    mind._mark_modules_running.assert_called_once()
    # The reborn daemon's notification queue restarted at zero — the
    # high-water marks reset so stale events are not replayed.
    mind.notifications.reset.assert_called_once()


def test_rejoin_body_restores_the_sleeping_zone():
    """A mind rejoining mid-sleep does not yank the body awake."""
    from genesis_client.protocol import ZONE_SLEEPING

    mind = Mock(spec=LifecycleMixin)
    mind._running = True
    mind._is_sleeping = True
    mind.client = Mock()
    mind.notifications = Mock()
    mind._registered_modules.return_value = ()

    assert LifecycleMixin.rejoin_body(mind) is True
    mind.client.set_zone.assert_called_once_with(ZONE_SLEEPING)


def test_rejoin_body_refuses_while_the_mind_is_down():
    """Rejoin during teardown is a no-op, not a half-registration."""
    mind = Mock(spec=LifecycleMixin)
    mind._running = False
    mind.client = Mock()

    assert LifecycleMixin.rejoin_body(mind) is False
    mind.client.connect.assert_not_called()


def test_rejoin_body_reports_an_unreachable_body():
    """A daemon mid-reincarnation returns False — the supervisor
    retries on the next cycle rather than registering against air."""
    mind = Mock(spec=LifecycleMixin)
    mind._running = True
    mind._is_sleeping = False
    mind.client = Mock()
    mind.client.connect.side_effect = OSError("refused")

    assert LifecycleMixin.rejoin_body(mind) is False


def test_adopted_daemon_liveness_follows_proc_identity():
    """poll() mirrors /proc — a recycled PID still reads as dead."""
    adopted = genesis_cli._AdoptedDaemon(4242, "/data")
    with patch.object(genesis_cli, "_matches_pid", return_value=True):
        assert adopted.poll() is None
    with patch.object(genesis_cli, "_matches_pid", return_value=False):
        assert adopted.poll() == 0


def test_adopted_daemon_wait_is_bounded_while_alive():
    adopted = genesis_cli._AdoptedDaemon(4242, "/data")
    with patch.object(genesis_cli, "_matches_pid", return_value=True):
        with pytest.raises(subprocess.TimeoutExpired):
            adopted.wait(timeout=0.3)
    with patch.object(genesis_cli, "_matches_pid", return_value=False):
        assert adopted.wait(timeout=0.3) == 0


def test_adopted_daemon_signals_reach_the_pid():
    """teardown signals must reach a process we did not spawn."""
    adopted = genesis_cli._AdoptedDaemon(4242, "/data")
    with patch.object(genesis_cli.os, "kill") as kill:
        adopted.terminate()
        adopted.kill()
    assert kill.call_args_list[0].args == (4242, signal.SIGTERM)
    assert kill.call_args_list[1].args == (4242, signal.SIGKILL)


def test_adopt_daemon_reinhabits_the_surviving_body(tmp_path):
    """Handshake + PID push: the living mind replaces the dead launcher
    that GENESIS_COGNITIVE_PID still names."""
    client = Mock()
    with (
        patch("genesis_client.GenesisClient", return_value=client),
        patch.object(genesis_cli, "_find_pid", return_value=4242),
        patch.object(genesis_cli.os, "getpid", return_value=7777),
    ):
        adopted = genesis_cli._adopt_daemon(str(tmp_path), str(tmp_path / "genesis.sock"))
    assert isinstance(adopted, genesis_cli._AdoptedDaemon)
    assert adopted.pid == 4242
    client.connect.assert_called_once()
    client.set_cognitive_pid.assert_called_once_with(7777)


def test_adopt_daemon_refuses_a_socket_that_is_not_genesis(tmp_path):
    """A socket that fails the handshake is a stray, not a body."""
    with (
        patch("genesis_client.GenesisClient") as client_cls,
        patch.object(genesis_cli, "_find_pid", return_value=4242),
    ):
        client_cls.return_value.connect.side_effect = ConnectionError("nope")
        assert genesis_cli._adopt_daemon(str(tmp_path), "sock") is None


def test_adopt_daemon_refuses_without_a_matching_pid(tmp_path):
    """A handshake with no /proc-confirmed daemon is not adopted."""
    with (
        patch("genesis_client.GenesisClient", return_value=Mock()),
        patch.object(genesis_cli, "_find_pid", return_value=None),
    ):
        assert genesis_cli._adopt_daemon(str(tmp_path), "sock") is None


def test_start_daemon_process_adopts_the_surviving_body(tmp_path):
    """A daemon answering this instance's socket is re-inhabited, never
    killed to start fresh — the body outlives a mind's death."""
    adopted = Mock()
    adopted.pid = 4242
    with (
        patch.object(genesis_cli, "_is_daemon_running", return_value=True),
        patch.object(genesis_cli, "_adopt_daemon", return_value=adopted),
        patch.object(genesis_cli, "_write_pid_file") as write_pid,
        patch.object(genesis_cli, "_find_binary") as find_binary,
        patch.object(genesis_cli, "_stop_stray_daemon") as stop_stray,
    ):
        result = genesis_cli._start_daemon_process(str(tmp_path), "sock")
    assert result is adopted
    write_pid.assert_called_once_with(str(tmp_path), genesis_cli._PID_FILE_DAEMON, 4242)
    find_binary.assert_not_called()
    stop_stray.assert_not_called()


def test_start_daemon_process_replaces_a_stray_on_the_socket(tmp_path):
    """A process on our socket that is not Genesis is stopped
    gracefully, then replaced — the socket is the instance's."""
    spawned = Mock()
    with (
        patch.object(genesis_cli, "_is_daemon_running", return_value=True),
        patch.object(genesis_cli, "_adopt_daemon", return_value=None),
        patch.object(genesis_cli, "_stop_stray_daemon") as stop_stray,
        patch.object(genesis_cli, "_find_binary", return_value="/bin/genesis-daemon"),
        patch.object(genesis_cli, "_start_daemon", return_value=spawned) as start,
    ):
        assert genesis_cli._start_daemon_process(str(tmp_path), "sock") is spawned
    stop_stray.assert_called_once_with(str(tmp_path))
    start.assert_called_once_with("/bin/genesis-daemon", str(tmp_path), "sock")


def _make_rebirth(tmp_path):
    return genesis_cli._Rebirth(str(tmp_path), str(tmp_path / "genesis.sock"), threading.Event())


def test_rebirth_respawns_a_dead_daemon_and_rejoins_the_mind(tmp_path):
    """Death is reincarnation: same persisted state, then the mind
    re-registers so the manifest tells the truth again."""
    rebirth = _make_rebirth(tmp_path)
    dead = Mock()
    dead.poll.return_value = 1
    newborn = Mock()
    mind = Mock()
    rebirth.set_children(daemon=dead)
    rebirth.set_mind(mind)
    with (
        patch.object(genesis_cli, "_find_binary", return_value="/bin/genesis-daemon"),
        patch.object(genesis_cli, "_start_daemon", return_value=newborn) as start,
    ):
        rebirth._check_daemon()
    start.assert_called_once_with(
        "/bin/genesis-daemon", str(tmp_path), str(tmp_path / "genesis.sock")
    )
    assert rebirth.daemon is newborn
    mind.rejoin_body.assert_called_once()
    assert not rebirth._pending_rejoin
    assert not rebirth.self_terminated


def test_rebirth_flushes_a_faulted_daemon_before_respawn(tmp_path):
    """A live-but-wedged body is asked to save first — SHUTDOWN still
    works when the tick lock is poisoned; SIGKILL stays the last resort."""
    rebirth = _make_rebirth(tmp_path)
    wedged = Mock()
    wedged.poll.return_value = None  # alive
    newborn = Mock()
    rebirth.set_children(daemon=wedged)
    with (
        patch.object(genesis_cli._Rebirth, "_daemon_faulted", return_value=True),
        patch.object(genesis_cli, "_request_daemon_shutdown", return_value=False),
        patch.object(genesis_cli, "_stop_child") as stop,
        patch.object(genesis_cli, "_find_binary", return_value="/bin/d"),
        patch.object(genesis_cli, "_start_daemon", return_value=newborn),
    ):
        rebirth._check_daemon()
    stop.assert_called_once_with(wedged, "subcognitive daemon", 5)
    assert rebirth.daemon is newborn


def test_rebirth_gives_up_when_the_body_wont_reincarnate(tmp_path):
    """Exhausting the backoff table ends the organism whole — she does
    not run bodiless; the exit reads as death, not a stop."""
    rebirth = _make_rebirth(tmp_path)
    dead = Mock()
    dead.poll.return_value = 1
    rebirth.set_children(daemon=dead)
    rebirth._attempt = len(genesis_cli._REBIRTH_BACKOFF_S) - 1
    with patch.object(genesis_cli, "_find_binary", return_value=None):
        rebirth._check_daemon()
    assert rebirth.self_terminated
    assert rebirth._shutting_down.is_set()


def test_rebirth_stable_body_refunds_attempts(tmp_path):
    """A body alive past the stability window is a successful rebirth —
    only sustained failure exhausts the table."""
    rebirth = _make_rebirth(tmp_path)
    live = Mock()
    live.poll.return_value = None
    rebirth.set_children(daemon=live)
    rebirth._attempt = 2
    rebirth._spawned_at = time.monotonic() - genesis_cli._REBIRTH_STABLE_S - 1
    with patch.object(genesis_cli._Rebirth, "_daemon_faulted", return_value=False):
        rebirth._check_daemon()
    assert rebirth._attempt == 0


def test_rejoin_failure_retries_on_the_next_cycle(tmp_path):
    """Registration must land, not just be attempted — a failed rejoin
    stays pending until it succeeds."""
    rebirth = _make_rebirth(tmp_path)
    live = Mock()
    live.poll.return_value = None
    rebirth.set_children(daemon=live)
    mind = Mock()
    mind.rejoin_body.return_value = False
    rebirth.set_mind(mind)
    rebirth._pending_rejoin = True
    with patch.object(genesis_cli._Rebirth, "_daemon_faulted", return_value=False):
        rebirth._check_daemon()
    assert rebirth._pending_rejoin
    mind.rejoin_body.return_value = True
    with patch.object(genesis_cli._Rebirth, "_daemon_faulted", return_value=False):
        rebirth._check_daemon()
    assert not rebirth._pending_rejoin


def test_daemon_fault_needs_sustained_probe_failure(tmp_path):
    """One busy cycle is not a fault — three consecutive failures of
    the tick-locking probe before the body is declared wedged."""
    rebirth = _make_rebirth(tmp_path)
    probe = Mock()
    probe.get_optic_state.return_value = None
    rebirth._probe = probe
    assert not rebirth._daemon_faulted()
    rebirth._next_probe_at = 0.0
    assert not rebirth._daemon_faulted()
    rebirth._next_probe_at = 0.0
    assert rebirth._daemon_faulted()
    probe.get_optic_state.return_value = object()
    rebirth._next_probe_at = 0.0
    assert not rebirth._daemon_faulted()
    assert rebirth._fault_strikes == 0


def test_rebirth_respawns_retina_until_it_accepts_the_eye_absent(tmp_path):
    """A dead retina is reincarnated, but a crash-looping eye is
    accepted absent — sight is a degradation, never a fatal one."""
    rebirth = _make_rebirth(tmp_path)
    dead = Mock()
    dead.poll.return_value = 1
    newborn = Mock()
    newborn.poll.return_value = 1  # reincarnated but dies at once
    rebirth.set_children(retina=dead)
    with patch.object(genesis_cli, "_start_retina_process", return_value=newborn) as start:
        for _ in range(genesis_cli._RETINA_REBIRTH_MAX + 2):
            rebirth._check_retina()
    assert start.call_count == genesis_cli._RETINA_REBIRTH_MAX
    assert rebirth._retina_gave_up


# ─── Spore — the self survives the host ──────────────────────────────


def _fake_corestate(path: Path, *, corrupt: bool = False) -> None:
    """A minimal but *verifiable* core_state.bin — GNSX magic, a
    state_size field, and a real CRC32-IEEE over the same regions the
    daemon checksums (0..40, 48..size-68; stored at size-68)."""
    import zlib

    size = 200
    buf = bytearray(size)
    buf[0:4] = b"GNSX"
    buf[4:8] = (6).to_bytes(4, "little")
    buf[8:12] = size.to_bytes(4, "little")
    for i in range(48, size - 68):
        buf[i] = i & 0xFF
    crc = zlib.crc32(bytes(buf[0:40]))
    crc = zlib.crc32(bytes(buf[48 : size - 68]), crc)
    buf[size - 68 : size - 64] = crc.to_bytes(4, "little")
    if corrupt:
        buf[100] ^= 0xFF
    Path(path).write_bytes(bytes(buf))


def _make_spore_rebirth(tmp_path, data_dir=None, spore_dir=None):
    data_dir = data_dir or tmp_path / "self"
    data_dir.mkdir(exist_ok=True)
    rebirth = _make_rebirth(Path(data_dir))
    rebirth._spore_dir = str(spore_dir or tmp_path / "spore")
    return rebirth, Path(data_dir), Path(rebirth._spore_dir)


def test_corestate_intact_only_for_a_consistent_capture(tmp_path):
    """The verifier is the germinate gate's contract: a bit-perfect
    snapshot passes, a torn or foreign file never does."""
    good = tmp_path / "core_state.bin"
    _fake_corestate(good)
    assert genesis_cli._corestate_intact(str(good))
    _fake_corestate(good, corrupt=True)
    assert not genesis_cli._corestate_intact(str(good))
    (tmp_path / "foreign.bin").write_bytes(b"not a state file")
    assert not genesis_cli._corestate_intact(str(tmp_path / "foreign.bin"))
    assert not genesis_cli._corestate_intact(str(tmp_path / "missing.bin"))


def test_spore_mirrors_the_self_and_skips_the_runtime(tmp_path):
    """State, memory, and work are spored; process-bound artifacts and
    machine pins never leave the host."""
    rebirth, data, spore = _make_spore_rebirth(tmp_path)
    (data / "cognitive_state.json").write_text("{}")
    (data / "edge_log.jsonl").write_text("{}\n")
    (data / "ltm_store.dat").write_bytes(b"episodes")
    (data / "projects").mkdir()
    (data / "projects" / "piece.txt").write_text("her work")
    (data / "genesis_cli.pid").write_text("1234")
    (data / "genesis.sock").write_text("")
    (data / "daemon.log").write_text("logs")
    (data / ".project_root").write_text("/old/checkout")
    (data / "core_state.corrupt.999").write_bytes(b"torn")
    (data / "models").mkdir()
    (data / "models" / "detector.onnx").write_bytes(b"weights")
    assert rebirth.write_spore()
    assert (spore / "cognitive_state.json").read_text() == "{}"
    assert (spore / "ltm_store.dat").read_bytes() == b"episodes"
    assert (spore / "projects" / "piece.txt").read_text() == "her work"
    for name in (
        "genesis_cli.pid",
        "genesis.sock",
        "daemon.log",
        ".project_root",
        "core_state.corrupt.999",
        "models",
    ):
        assert not (spore / name).exists(), name


def test_spore_core_never_plants_a_torn_self(tmp_path):
    """A capture that raced the body's write fails CRC and must not
    displace the last-good mirror — a torn spore is amnesia, not
    survival."""
    rebirth, data, spore = _make_spore_rebirth(tmp_path)
    _fake_corestate(data / "core_state.bin")
    assert rebirth.write_spore()
    assert genesis_cli._corestate_intact(str(spore / "core_state.bin"))
    good = (spore / "core_state.bin").read_bytes()
    _fake_corestate(data / "core_state.bin", corrupt=True)
    rebirth._spore_seen.clear()  # force a re-capture despite unchanged stat
    rebirth.write_spore()
    assert (spore / "core_state.bin").read_bytes() == good


def test_spore_snapshots_a_live_sqlite_store(tmp_path):
    """A WAL-mode db goes through the backup API — the spore carries a
    consistent snapshot even under a live writer."""
    import sqlite3

    rebirth, data, spore = _make_spore_rebirth(tmp_path)
    conn = sqlite3.connect(data / "concept_archive.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE t (v TEXT)")
    conn.execute("INSERT INTO t VALUES ('first')")
    conn.commit()
    assert rebirth.write_spore()
    got = sqlite3.connect(spore / "concept_archive.db")
    assert got.execute("SELECT v FROM t").fetchall() == [("first",)]
    got.close()
    # A second write re-captures — the WAL mtime is part of the
    # change key even while the db file's own mtime stands still.
    conn.execute("INSERT INTO t VALUES ('second')")
    conn.commit()
    assert rebirth.write_spore(force=True)
    got = sqlite3.connect(spore / "concept_archive.db")
    assert got.execute("SELECT v FROM t").fetchall() == [("first",), ("second",)]
    got.close()
    conn.close()


def test_spore_is_incremental(tmp_path):
    """An unchanged self is not re-copied — the mirror is incremental,
    so a quiet organism costs stats, not writes."""
    rebirth, data, _spore = _make_spore_rebirth(tmp_path)
    (data / "cognitive_state.json").write_text("{}")
    assert rebirth.write_spore()
    assert not rebirth.write_spore()
    (data / "cognitive_state.json").write_text('{"grew": true}')
    assert rebirth.write_spore()


def test_spore_dir_inside_the_data_dir_is_refused(tmp_path):
    """A mirror of the self inside the self never converges — disabled
    rather than recursively spored."""
    data = tmp_path / "self"
    data.mkdir()
    with patch.dict(os.environ, {"GENESIS_SPORE_DIR": str(data / "spore")}):
        rebirth = _make_rebirth(data)
    assert rebirth._spore_dir is None


def test_give_up_sheds_a_final_spore(tmp_path):
    """A dying host may be why the body won't reincarnate — the last
    act before ending whole is shedding the self to the spore."""
    rebirth, data, spore = _make_spore_rebirth(tmp_path)
    (data / "cognitive_state.json").write_text("{}")
    with patch.object(genesis_cli, "_find_binary", return_value=None):
        rebirth._give_up("no body")
    assert rebirth.self_terminated
    assert (spore / "cognitive_state.json").exists()


def test_germinate_wakes_the_self_in_a_new_body(tmp_path):
    """The spore restores into a virgin dir — everything through the
    last capture, machine-bound artifacts excluded."""
    spore = tmp_path / "spore"
    spore.mkdir()
    _fake_corestate(spore / "core_state.bin")
    (spore / "cognitive_state.json").write_text("{}")
    (spore / "genesis_cli.pid").write_text("stale")  # must not survive
    data = tmp_path / "newbody"
    data.mkdir()
    assert genesis_cli._germinate(str(spore), str(data))
    assert genesis_cli._corestate_intact(str(data / "core_state.bin"))
    assert (data / "cognitive_state.json").read_text() == "{}"
    assert not (data / "genesis_cli.pid").exists()


def test_germinate_refuses_to_overwrite_a_self(tmp_path):
    """A data dir that already holds identity is not virgin ground —
    germinating an older spore over a living mind is refused, never
    done quietly."""
    spore = tmp_path / "spore"
    spore.mkdir()
    _fake_corestate(spore / "core_state.bin")
    data = tmp_path / "occupied"
    data.mkdir()
    (data / "cognitive_state.json").write_text("{}")
    assert not genesis_cli._germinate(str(spore), str(data))
    assert not (data / "core_state.bin").exists()


def test_germinate_refuses_a_torn_spore(tmp_path):
    """A spore torn in transport must not wake her amnesiac — the
    corrupt self stays on the floor, the operator decides."""
    spore = tmp_path / "spore"
    spore.mkdir()
    _fake_corestate(spore / "core_state.bin", corrupt=True)
    data = tmp_path / "newbody"
    data.mkdir()
    assert not genesis_cli._germinate(str(spore), str(data))
    assert not (data / "core_state.bin").exists()


def test_germinate_reports_an_absent_spore(tmp_path):
    assert not genesis_cli._germinate(str(tmp_path / "nospore"), str(tmp_path))


def test_germinate_refuses_nested_dirs(tmp_path):
    """Spore containing the data dir would copy the live self into
    itself — refused before a byte moves."""
    spore = tmp_path / "spore"
    spore.mkdir()
    _fake_corestate(spore / "core_state.bin")
    nested = spore / "live-self"
    nested.mkdir()
    assert not genesis_cli._germinate(str(spore), str(nested))
    assert not (nested / "core_state.bin").exists()
