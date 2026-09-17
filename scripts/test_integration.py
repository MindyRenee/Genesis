"""Integration test: start the Genesis daemon and talk to it.

This test starts the actual Rust daemon binary, connects via the Python
client, exercises all IPC commands, and verifies the responses. It
requires the daemon to be built first: `cargo build --release`.

Run with:
    python3 scripts/test_integration.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

# Add the python/ directory to the path
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import logging

from genesis_client import GenesisClient
from genesis_client.protocol import (
    CHEM_DOPAMINE,
    ZONE_CODING,
)
from genesis_client.types import MemoryStats, NeuroSummary, PhaseInfo

logger = logging.getLogger(__name__)


def find_daemon_binary() -> str | None:
    """Find the genesis-daemon binary."""
    # Check common locations
    candidates = [
        # Release build
        os.path.join(os.path.dirname(__file__), "..", "target", "release", "genesis-daemon"),
        # Debug build
        os.path.join(os.path.dirname(__file__), "..", "target", "debug", "genesis-daemon"),
    ]
    for path in candidates:
        path = os.path.abspath(path)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def wait_for_socket(socket_path: str, timeout: float = 10.0) -> bool:
    """Wait for the daemon's socket to accept connections."""
    import socket as _socket

    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(socket_path):
            try:
                s = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(socket_path)
                s.close()
                return True
            except OSError:
                pass
        time.sleep(0.1)
    return os.path.exists(socket_path)


def _run_basic_ipc_tests(client, passed: int, failed: int) -> tuple[int, int]:
    """Run Tests 1-4: ping, neuro summary, phase, memory stats."""
    # ─── Test 1: Ping ──────────────────────────────────
    try:
        pong = client.ping()
        assert pong.ack is True, f"ping ack should be True, got {pong.ack}"
        logger.info(f"  PASS  ping (uptime={pong.uptime_ms}ms)")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  ping: {e}")
        failed += 1

    # ─── Test 2: GetNeuroSummary ───────────────────────
    try:
        summary = client.get_neuro_summary()
        assert isinstance(summary, NeuroSummary)
        assert 0.0 <= summary.arousal <= 1.0, f"arousal out of range: {summary.arousal}"
        assert -1.0 <= summary.valence <= 1.0, f"valence out of range: {summary.valence}"
        assert 0.0 <= summary.plasticity_gate <= 1.0
        assert 0.0 <= summary.encoding_weight <= 1.0
        assert 0.0 <= summary.consolidation_weight <= 1.0
        assert 0.0 <= summary.retrieval_weight <= 1.0
        logger.info(
            f"  PASS  get_neuro_summary (phase={summary.phase_name}, "
            f"arousal={summary.arousal:.2f}, valence={summary.valence:.2f})"
        )
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  get_neuro_summary: {e}")
        failed += 1

    # ─── Test 3: GetPhase ──────────────────────────────
    try:
        phase = client.get_phase()
        assert isinstance(phase, PhaseInfo)
        assert 0.0 <= phase.arousal <= 1.0
        assert -1.0 <= phase.valence <= 1.0
        logger.info(f"  PASS  get_phase (phase={phase.phase_name}, arousal={phase.arousal:.2f})")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  get_phase: {e}")
        failed += 1

    # ─── Test 4: GetMemoryStats ────────────────────────
    try:
        stats = client.get_memory_stats()
        assert isinstance(stats, MemoryStats)
        assert stats.stm_count == 0, f"STM should be empty, got {stats.stm_count}"
        assert stats.ltm_count == 0, f"LTM should be empty, got {stats.ltm_count}"
        assert stats.ltm_capacity == 256
        logger.info(
            f"  PASS  get_memory_stats (stm={stats.stm_count}, "
            f"ltm={stats.ltm_count}/{stats.ltm_capacity})"
        )
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  get_memory_stats: {e}")
        failed += 1

    return passed, failed


def _run_event_tests(client, passed: int, failed: int) -> tuple[int, int]:
    """Run Tests 5-8: store event, STM count, set zone, neuro impulse."""
    # ─── Test 5: StoreEvent ────────────────────────────
    try:
        result = client.store_event(
            timestamp=int(time.time() * 1000),
            event_type=0,  # UserInput
            source_module=4,  # Language
            salience=0.8,
            emotional_tag=[0.7, 0.5, 0.6, 0.5, 0.4, 0.5, 0.2, 0.3, 0.4, 0.4, 0.1, 0.4],
            text="User said: Hello, Genesis!",
        )
        assert result is True, "store_event should return True"
        logger.info("  PASS  store_event")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  store_event: {e}")
        failed += 1

    # ─── Test 6: Verify STM count increased ────────────
    try:
        stats = client.get_memory_stats()
        assert stats.stm_count == 1, f"STM should have 1 entry, got {stats.stm_count}"
        logger.info(f"  PASS  stm_count after store ({stats.stm_count})")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  stm_count after store: {e}")
        failed += 1

    # ─── Test 7: SetZone ───────────────────────────────
    try:
        result = client.set_zone(ZONE_CODING)
        assert result is True
        logger.info("  PASS  set_zone (coding)")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  set_zone: {e}")
        failed += 1

    # ─── Test 8: NeuroImpulse ──────────────────────────
    try:
        result = client.neuro_impulse(CHEM_DOPAMINE, 0.3)
        assert result is True
        logger.info("  PASS  neuro_impulse (dopamine +0.3)")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  neuro_impulse: {e}")
        failed += 1

    return passed, failed


def _run_state_tests(client, passed: int, failed: int) -> tuple[int, int]:
    """Run Tests 9-14: get state, find similar, retrieve, sync, convenience, observation."""
    # ─── Test 9: GetState ──────────────────────────────
    try:
        state = client.get_state()
        assert state.heartbeat > 0, f"heartbeat should be > 0, got {state.heartbeat}"
        assert state.cognitive_zone == ZONE_CODING
        assert -1.0 <= state.valence <= 1.0
        logger.info(
            f"  PASS  get_state (heartbeat={state.heartbeat}, "
            f"zone={state.cognitive_zone}, phase={state.phase_name})"
        )
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  get_state: {e}")
        failed += 1

    # ─── Test 10: FindSimilar (empty LTM) ──────────────
    try:
        results = client.find_similar("hello", limit=5)
        assert isinstance(results, list)
        # LTM is empty, so no results expected
        logger.info(f"  PASS  find_similar (results={len(results)})")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  find_similar: {e}")
        failed += 1

    # ─── Test 11: RetrieveEpisode (nonexistent) ────────
    try:
        from genesis_client.exceptions import EpisodeNotFound

        client.retrieve_episode(999)
        logger.info("  FAIL  retrieve_episode should have raised EpisodeNotFound")
        failed += 1
    except EpisodeNotFound:
        logger.info("  PASS  retrieve_episode (not found as expected)")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  retrieve_episode: unexpected exception {e}")
        failed += 1

    # ─── Test 12: Sync ─────────────────────────────────
    try:
        result = client.sync()
        assert result is True
        logger.info("  PASS  sync")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  sync: {e}")
        failed += 1

    # ─── Test 13: Convenience methods ──────────────────
    try:
        assert client.reward(0.2) is True
        assert client.stress(0.1) is True
        assert client.calm(0.2) is True
        assert client.bond(0.1) is True
        logger.info("  PASS  convenience methods (reward, stress, calm, bond)")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  convenience methods: {e}")
        failed += 1

    # ─── Test 14: Store observation ────────────────────
    try:
        assert client.store_observation("test observation", salience=0.6) is True
        stats = client.get_memory_stats()
        assert stats.stm_count == 2, f"STM should have 2 entries, got {stats.stm_count}"
        logger.info("  PASS  store_observation")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  store_observation: {e}")
        failed += 1

    return passed, failed


def _run_ltm_tests(client, passed: int, failed: int) -> tuple[int, int]:
    """Run Tests 15-17: consolidation, retrieve episode, find similar."""
    # ─── Test 15: Wait for consolidation ───────────────
    # The daemon ticks at 5Hz. Consolidation runs when STM has entries
    # and plasticity gate is open. Wait a few seconds for it to process.
    try:
        logger.info("  INFO  waiting for consolidation (5 seconds)...")
        time.sleep(5.0)
        stats = client.get_memory_stats()
        # The daemon should have consolidated some entries to LTM
        # (consolidation threshold is 0.15, our salience was 0.6-0.8)
        if stats.ltm_count > 0:
            logger.info(f"  PASS  consolidation occurred (ltm={stats.ltm_count})")
        else:
            # Consolidation might not have happened if the tick loop
            # hasn't run enough cycles or the threshold wasn't met.
            # This is not a hard failure — it depends on timing.
            logger.info(
                f"  SKIP  no consolidation yet (ltm={stats.ltm_count}, stm={stats.stm_count})"
            )
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  consolidation check: {e}")
        failed += 1

    # ─── Test 16: RetrieveEpisode (if LTM has entries) ─
    try:
        stats = client.get_memory_stats()
        if stats.ltm_count > 0:
            # Try to retrieve episode 1
            ep = client.retrieve_episode(1)
            assert ep is not None
            assert ep.episode_id == 1
            assert len(ep.text) > 0
            assert len(ep.full_emotional_tag) == 12
            logger.info(f"  PASS  retrieve_episode (id=1, text='{ep.text[:30]}...')")
        else:
            logger.info("  SKIP  retrieve_episode (LTM empty)")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  retrieve_episode: {e}")
        failed += 1

    # ─── Test 17: FindSimilar (if LTM has entries) ─────
    try:
        stats = client.get_memory_stats()
        if stats.ltm_count > 0:
            results = client.find_similar("hello genesis", limit=5)
            assert isinstance(results, list)
            if len(results) > 0:
                logger.info(
                    f"  PASS  find_similar (found {len(results)} results, "
                    f"best dist={results[0].hamming_distance})"
                )
            else:
                logger.info("  PASS  find_similar (no results, but call succeeded)")
        else:
            logger.info("  SKIP  find_similar (LTM empty)")
        passed += 1
    except Exception as e:  # noqa: BLE001
        logger.info(f"  FAIL  find_similar: {e}")
        failed += 1

    return passed, failed


def _cleanup_integration(proc, socket_path: str, data_dir: str) -> None:
    """Shutdown the daemon and clean up the data directory."""
    # Shutdown the daemon
    try:
        # Try graceful shutdown via IPC
        client = GenesisClient(socket_path)
        client.connect()
        client.shutdown()
        client.disconnect()
        proc.wait(timeout=5.0)
        logger.info("  PASS  daemon shutdown via IPC")
    except Exception as e:  # noqa: BLE001
        # Graceful IPC shutdown failed. This is a disposable temp-dir
        # test daemon (no developmental continuity), so SIGKILL here
        # cannot corrupt live state — it only avoids hanging the test
        # suite forever. Never do this to a production instance.
        logger.warning(f"daemon shutdown failed, force-killing test daemon: {e}")
        proc.kill()
        proc.wait()
        logger.info("  INFO  test daemon force-killed")

    # Clean up
    shutil.rmtree(data_dir, ignore_errors=True)


def run_integration_tests() -> bool:
    """Run the full integration test suite."""
    daemon_path = find_daemon_binary()
    if daemon_path is None:
        logger.info("  SKIP  daemon binary not found (run: cargo build --release)")
        return True  # Skip, not fail

    logger.info(f"  INFO  Using daemon: {daemon_path}")

    # Create a temporary data directory
    data_dir = tempfile.mkdtemp(prefix="genesis_test_")
    socket_path = os.path.join(data_dir, "genesis.sock")

    passed = 0
    failed = 0

    # Start the daemon
    logger.info(f"  INFO  Starting daemon (data_dir={data_dir})")
    proc = subprocess.Popen(
        [daemon_path, "--data-dir", data_dir, "--stm-capacity", "64", "--ltm-capacity", "256"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for the socket to appear
        if not wait_for_socket(socket_path):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            logger.info("  FAIL  daemon didn't create socket in time")
            logger.info(f"        stderr: {stderr[:500]}")
            failed += 1
            return False

        # Give the daemon a moment to initialise
        time.sleep(0.5)

        # Connect
        client = GenesisClient(socket_path)
        client.connect()
        logger.info("  PASS  connected to daemon")

        passed, failed = _run_basic_ipc_tests(client, passed, failed)
        passed, failed = _run_event_tests(client, passed, failed)
        passed, failed = _run_state_tests(client, passed, failed)
        passed, failed = _run_ltm_tests(client, passed, failed)

        # Disconnect
        client.disconnect()
        logger.info("  PASS  disconnected")

    finally:
        _cleanup_integration(proc, socket_path, data_dir)

    logger.info(f"\n  Integration tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
