"""Unit tests for the Genesis client protocol layer.

These tests verify the binary framing, packing, and unpacking logic
without needing a running daemon. Uses only the standard library.
"""

import logging
import socket
import struct
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from genesis_client.client import GenesisClient
from genesis_client.exceptions import ConnectionError, ProtocolError
from genesis_client.protocol import (
    ADVANCE_NEURO,
    CHEM_BDNF,
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_NAMES,
    CHEM_SEROTONIN,
    GET_NEURO_SUMMARY,
    GET_STATE,
    HANDSHAKE,
    MODULE_ATTENTION,
    MODULE_DREAMING,
    MODULE_EMOTION,
    MODULE_GUARDRAILS,
    MODULE_INTENTION,
    MODULE_LANGUAGE,
    MODULE_MEMORY,
    MODULE_METACOGNITION,
    MODULE_MOTOR,
    MODULE_REASONING,
    MODULE_SENSORY,
    MODULE_STATUS_ERROR,
    MODULE_STATUS_IDLE,
    MODULE_STATUS_RUNNING,
    MODULE_STATUS_STARTING,
    MODULE_STATUS_STOPPED,
    MODULE_STATUS_STOPPING,
    MODULE_SUBCOGNITIVE,
    PHASE_FLOW,
    PHASE_NAMES,
    PHASE_STRESS,
    PING,
    PROTOCOL_VERSION,
    SHUTDOWN,
    STORE_EPISODE,
    STORE_EVENT,
    UPDATE_MODULE_STATUS,
    ZONE_CODING,
    ZONE_CONVERSATION,
    ZONE_NAMES,
    pack_message,
    read_message,
    unpack_message,
    write_message,
)

logger = logging.getLogger(__name__)


def test_pack_empty_payload():
    """A command with no payload packs as [len=1][cmd_id]."""
    msg = pack_message(PING)
    assert msg == struct.pack("<I", 1) + bytes([PING])


def test_pack_with_payload():
    """A command with payload packs as [len][cmd_id][payload]."""
    payload = b"\x01\x02\x03"
    msg = pack_message(GET_STATE, payload)
    assert msg == struct.pack("<I", 4) + bytes([GET_STATE]) + payload


def test_unpack_roundtrip():
    """pack -> unpack roundtrips correctly."""
    for cmd_id in range(1, 14):
        payload = bytes(range(cmd_id))
        msg = pack_message(cmd_id, payload)
        unpacked_cmd, unpacked_data = unpack_message(msg)
        assert unpacked_cmd == cmd_id, f"cmd {cmd_id}: got {unpacked_cmd}"
        assert unpacked_data == payload, f"cmd {cmd_id}: data mismatch"


def test_unpack_zero_length():
    """A zero-length message is a protocol violation and raises ValueError.

    The length must include at least the command_id byte, so zero is
    invalid. The Rust daemon rejects zero-length frames with an error;
    the Python client matches this.
    """
    msg = struct.pack("<I", 0)
    with pytest.raises(ValueError, match="zero"):
        unpack_message(msg)


def test_unpack_too_short():
    """A message shorter than the length prefix raises ValueError."""
    try:
        unpack_message(b"\x01\x02")
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        logger.debug(repr(e))


def test_unpack_declared_longer_than_actual():
    """A message whose declared length exceeds actual raises ValueError."""
    msg = struct.pack("<I", 100) + bytes([PING])
    try:
        unpack_message(msg)
        raise AssertionError("should have raised ValueError")
    except ValueError as e:
        logger.debug(repr(e))


def test_unpack_exceeds_max_length():
    """A declared length exceeding MAX_MESSAGE_LEN raises ValueError."""
    from genesis_client.protocol import MAX_MESSAGE_LEN

    msg = struct.pack("<I", MAX_MESSAGE_LEN + 1) + bytes([PING])
    with pytest.raises(ValueError, match="exceeds maximum"):
        unpack_message(msg)


def test_pack_invalid_cmd_id():
    """pack_message rejects cmd_id outside u8 range."""
    with pytest.raises(ValueError, match="cmd_id must be 0-255"):
        pack_message(-1)
    with pytest.raises(ValueError, match="cmd_id must be 0-255"):
        pack_message(256)


def test_pack_oversize_payload():
    """pack_message rejects payloads that would exceed MAX_MESSAGE_LEN."""
    from genesis_client.protocol import MAX_MESSAGE_LEN

    # MAX_MESSAGE_LEN includes the 1-byte cmd_id, so the payload limit
    # is MAX_MESSAGE_LEN - 1.
    oversize = b"\x00" * MAX_MESSAGE_LEN
    with pytest.raises(ValueError, match="exceeds maximum"):
        pack_message(PING, oversize)


def test_read_message_zero_length():
    """read_message rejects zero-length frames (protocol violation).

    The Rust daemon rejects zero-length frames; the client must match.
    """
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Send a raw zero-length frame (4 bytes of zeros)
        sock_a.sendall(struct.pack("<I", 0))
        with pytest.raises(ConnectionError, match="zero"):
            read_message(sock_b)
    finally:
        sock_a.close()
        sock_b.close()


def test_write_and_read_roundtrip():
    """write_message -> read_message roundtrips correctly."""
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        payload = b"hello genesis"
        write_message(sock_a, GET_NEURO_SUMMARY, payload)
        cmd, data = read_message(sock_b)
        assert cmd == GET_NEURO_SUMMARY
        assert data == payload
    finally:
        sock_a.close()
        sock_b.close()


def test_write_and_read_empty_payload():
    """Empty payload roundtrips correctly."""
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        write_message(sock_a, PING)
        cmd, data = read_message(sock_b)
        assert cmd == PING
        assert data == b""
    finally:
        sock_a.close()
        sock_b.close()


def test_multiple_messages_in_sequence():
    """Multiple messages can be sent and received in sequence."""
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        for i in range(10):
            payload = struct.pack("<I", i)
            write_message(sock_a, GET_STATE, payload)

        for i in range(10):
            cmd, data = read_message(sock_b)
            assert cmd == GET_STATE
            assert struct.unpack("<I", data)[0] == i
    finally:
        sock_a.close()
        sock_b.close()


def test_read_closed_socket():
    """Reading from a closed socket raises ConnectionError."""
    sock_a, sock_b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sock_a.close()
    sock_b.close()
    try:
        read_message(sock_b)
        raise AssertionError("should have raised ConnectionError")
    except ConnectionError as e:
        logger.debug(repr(e))


def test_command_ids():
    """Command IDs match the Rust protocol table."""
    assert GET_STATE == 1
    assert GET_NEURO_SUMMARY == 2
    assert STORE_EVENT == 3
    assert STORE_EPISODE == 32
    assert PING == 8
    assert SHUTDOWN == 12


def test_chemical_ids():
    """Chemical IDs match NeurochemicalId discriminants."""
    assert CHEM_DOPAMINE == 0
    assert CHEM_SEROTONIN == 1
    assert CHEM_CORTISOL == 6
    assert CHEM_BDNF == 11


def test_phase_ids():
    """Phase IDs match MentalPhase discriminants."""
    assert PHASE_FLOW == 2
    assert PHASE_STRESS == 3


def test_zone_ids():
    """Zone IDs match CognitiveZone discriminants."""
    assert ZONE_CODING == 2
    assert ZONE_CONVERSATION == 1


def test_name_maps():
    """Name maps have entries for all known IDs."""
    assert CHEM_NAMES[CHEM_DOPAMINE] == "dopamine"
    assert CHEM_NAMES[CHEM_CORTISOL] == "cortisol"
    assert PHASE_NAMES[PHASE_FLOW] == "flow"
    assert ZONE_NAMES[ZONE_CODING] == "coding"


def test_module_ids():
    """Module IDs match ModuleId discriminants from the Rust manifest."""
    assert MODULE_SUBCOGNITIVE == 0
    assert MODULE_ATTENTION == 1
    assert MODULE_MEMORY == 2
    assert MODULE_EMOTION == 3
    assert MODULE_LANGUAGE == 4
    assert MODULE_REASONING == 5
    assert MODULE_SENSORY == 6
    assert MODULE_MOTOR == 7
    assert MODULE_METACOGNITION == 8
    assert MODULE_DREAMING == 9
    assert MODULE_INTENTION == 10
    assert MODULE_GUARDRAILS == 11


def test_module_status_values():
    """Module status values match ModuleStatus discriminants."""
    assert MODULE_STATUS_STOPPED == 0
    assert MODULE_STATUS_STARTING == 1
    assert MODULE_STATUS_RUNNING == 2
    assert MODULE_STATUS_IDLE == 3
    assert MODULE_STATUS_STOPPING == 4
    assert MODULE_STATUS_ERROR == 5


def test_update_module_status_command_id():
    """UPDATE_MODULE_STATUS command ID is 13."""
    assert UPDATE_MODULE_STATUS == 13


def test_handshake_command_id():
    """HANDSHAKE command ID is 15."""
    assert HANDSHAKE == 15


def test_protocol_version():
    """Protocol version is 2 (must match Rust PROTOCOL_VERSION).

    v2 added the trailing `cpu_boost` byte to the body-control response.
    """
    assert PROTOCOL_VERSION == 2


def _client_with_socket():
    """Helper: with socket."""
    client = GenesisClient("/unused/genesis.sock", request_timeout=1.5)
    sock = Mock(spec=socket.socket)
    sock.gettimeout.return_value = 1.5
    client._sock = sock
    return client, sock


@pytest.mark.parametrize("command", [STORE_EVENT, ADVANCE_NEURO, SHUTDOWN])
def test_client_does_not_replay_mutation_after_lost_response(command):
    """Test client does not replay mutation after lost response."""
    client, sock = _client_with_socket()
    replacement = Mock(spec=socket.socket)

    def reconnect():
        """Swap the client's socket for the replacement mock."""
        client._sock = replacement

    with (
        patch("genesis_client.client.write_message") as send,
        patch("genesis_client.client.read_message", side_effect=[
            ConnectionError("response lost"), (command, b"\x01"),
        ]),
        patch.object(client, "connect", side_effect=reconnect),
    ):
        with pytest.raises(ConnectionError):
            client._request(command, b"payload")

    send.assert_called_once_with(sock, command, b"payload")
    sock.close.assert_called_once()


def test_client_read_retries_once_and_closes_broken_socket():
    """Test client read retries once and closes broken socket."""
    client, sock = _client_with_socket()
    replacement = Mock(spec=socket.socket)

    def reconnect():
        """Swap the client's socket for the replacement mock."""
        client._sock = replacement

    with (
        patch("genesis_client.client.write_message") as send,
        patch("genesis_client.client.read_message", side_effect=[
            ConnectionError("response lost"), (GET_STATE, b"fresh"),
        ]),
        patch.object(client, "connect", side_effect=reconnect),
    ):
        assert client._request(GET_STATE) == b"fresh"

    assert send.call_count == 2
    sock.close.assert_called_once()
    assert client.is_connected


def test_client_failed_retry_closes_both_sockets():
    """Test client failed retry closes both sockets."""
    client, sock = _client_with_socket()
    replacement = Mock(spec=socket.socket)

    def reconnect():
        """Swap the client's socket for the replacement mock."""
        client._sock = replacement

    with (
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", side_effect=ConnectionError("lost")),
        patch.object(client, "connect", side_effect=reconnect),
    ):
        with pytest.raises(ConnectionError):
            client._request(GET_STATE)

    sock.close.assert_called_once()
    replacement.close.assert_called_once()
    assert not client.is_connected
    assert not client._cache


def test_client_disconnect_invalidates_cached_state():
    """Test client disconnect invalidates cached state."""
    client, sock = _client_with_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", return_value=(GET_STATE, b"old")),
    ):
        assert client._request(GET_STATE) == b"old"
        client.disconnect()
        assert not client._cache
        with pytest.raises(ConnectionError):
            client._request(GET_STATE)

    sock.close.assert_called_once()


def test_client_connect_replaces_socket_and_respects_timeout():
    """Test client connect replaces socket and respects timeout."""
    client, sock = _client_with_socket()
    client._cache[GET_STATE] = (0.0, b"old")
    replacement = Mock(spec=socket.socket)
    with (
        patch("genesis_client.client.os.path.exists", return_value=True),
        patch("genesis_client.client.socket.socket", return_value=replacement),
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", return_value=(
            HANDSHAKE, bytes([PROTOCOL_VERSION]),
        )),
    ):
        client.connect()

    sock.close.assert_called_once()
    replacement.settimeout.assert_called_with(1.5)
    assert not client._cache


def test_client_write_invalidates_cache_after_acquiring_lock():
    """Test client write invalidates cache after acquiring lock."""
    client, _ = _client_with_socket()
    guard = MagicMock()
    guard.__enter__.side_effect = lambda: client._cache.update({GET_STATE: (0.0, b"old")})
    with (
        patch.object(client, "_lock", guard),
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", return_value=(STORE_EVENT, b"\x01")),
    ):
        client._request(STORE_EVENT, b"payload")

    assert GET_STATE not in client._cache


def test_client_cached_read_is_serialized_with_writes():
    """Test client cached read is serialized with writes."""
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", return_value=(GET_STATE, b"old")),
    ):
        client._request(GET_STATE)
        guard = MagicMock()
        with patch.object(client, "_lock", guard):
            assert client._request(GET_STATE) == b"old"
        guard.__enter__.assert_called_once()


def test_client_cache_expires_despite_wall_clock_rollback():
    """Test client cache expires despite wall clock rollback."""
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.time") as clock,
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", side_effect=[
            (GET_STATE, b"old"), (GET_STATE, b"fresh"),
        ]),
    ):
        clock.time.side_effect = [200.0, 100.0]
        clock.monotonic.side_effect = [10.0, 10.2, 10.2]
        assert client._request(GET_STATE) == b"old"
        assert client._request(GET_STATE) == b"fresh"


def test_client_discards_connection_after_response_command_mismatch():
    """Test client discards connection after response command mismatch."""
    client, sock = _client_with_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", return_value=(STORE_EVENT, b"\x01")),
    ):
        with pytest.raises(ProtocolError):
            client._request(GET_STATE)

    sock.close.assert_called_once()
    assert not client.is_connected


def test_client_detects_multibyte_error_for_ack_commands():
    """Ack commands (STORE_EVENT, ADVANCE_NEURO, etc.) pad error
    responses to the expected size. The client must detect these
    multi-byte errors, not just single-byte ones.
    """
    from genesis_client.protocol import ERROR_INTERNAL_ERROR

    client, _ = _client_with_socket()
    # STORE_EVENT error: daemon returns [0x84, 0, 0, 0, 0, 0, 0, 0] (8 bytes)
    error_resp = bytes([ERROR_INTERNAL_ERROR]) + b"\x00" * 7
    with (
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", return_value=(STORE_EVENT, error_resp)),
    ):
        with pytest.raises(ProtocolError) as exc_info:
            client._request(STORE_EVENT, b"payload")

    assert "internal error" in str(exc_info.value).lower()


def test_client_detects_single_byte_error_for_data_commands():
    """Data commands (GET_*) return a single-byte error on failure."""
    from genesis_client.protocol import ERROR_READ_FAILED, GET_NEURO_SUMMARY

    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(GET_NEURO_SUMMARY, bytes([ERROR_READ_FAILED])),
        ),
    ):
        with pytest.raises(ProtocolError) as exc_info:
            client._request(GET_NEURO_SUMMARY)

    assert "read failed" in str(exc_info.value).lower()


def test_client_does_not_false_positive_on_data_first_byte():
    """A raw-data response whose first byte happens to be >= 0x80
    (e.g. an f32 mantissa byte) must NOT be treated as an error.
    """
    from genesis_client.protocol import GET_INFERENCE_SUMMARY

    client, _ = _client_with_socket()
    # 60 bytes of data — first byte is 0x9A (part of an f32 mantissa),
    # which has the high bit set but is NOT an error response.
    data = bytes([0x9A]) + b"\x99\x99\x3E" + b"\x00" * 56
    assert len(data) == 60
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(GET_INFERENCE_SUMMARY, data),
        ),
    ):
        # Should NOT raise — the 60-byte response is data, not an error.
        result = client._request(GET_INFERENCE_SUMMARY)
        assert result == data


@pytest.mark.parametrize(
    ("dt_sent", "dt_expected"),
    [
        (0.5, 0.5),      # in range — passed through unchanged
        (100.0, 10.0),   # stall longer than the daemon's max — truncated
        (0.0, 0.001),    # degenerate zero elapsed — clamped to min
        (-5.0, 0.001),   # negative (clock anomaly) — clamped to min
    ],
)
def test_client_advance_neuro_clamps_dt(dt_sent, dt_expected):
    """advance_neuro must clamp dt to the daemon's [0.001, 10.0] range.

    The mind passes raw measured elapsed time; the client enforces the
    daemon's dt contract at the boundary so longer stalls are
    truncated rather than rejected. Without this, its neurochemical
    clock would lag the wall clock whenever the loop stalls.
    """
    import struct

    client, _ = _client_with_socket()
    # Valid ADVANCE_NEURO response: ack + 4×f32 + u32 = 21 bytes.
    ok_resp = b"\x01" + b"\x00\x00\x80\x3F" * 4 + b"\x00" * 4
    assert len(ok_resp) == 21
    with (
        patch("genesis_client.client.write_message") as send,
        patch("genesis_client.client.read_message", return_value=(ADVANCE_NEURO, ok_resp)),
    ):
        result = client.advance_neuro(dt=dt_sent)

    assert result is not None
    # The payload sent must be the clamped dt, packed as f32 LE.
    sent_payload = send.call_args[0][2]
    (packed_dt,) = struct.unpack("<f", sent_payload)
    assert abs(packed_dt - dt_expected) < 1e-6


def test_protocol_error_is_os_error():
    """ProtocolError must inherit from OSError so that existing
    `except (OSError, ConnectionError)` clauses in calling code
    catch protocol errors correctly.
    """
    assert issubclass(ProtocolError, OSError)


def test_exception_hierarchy():
    """Verify the full exception hierarchy matches the documented design.

    Hierarchy:
        GenesisError(Exception)
        ├── ConnectionError(GenesisError, OSError)
        │   └── DaemonNotRunning(ConnectionError)
        ├── ProtocolError(GenesisError, OSError)
        └── EpisodeNotFound(GenesisError, LookupError)
    """
    from genesis_client.exceptions import (
        ConnectionError as GenesisConnectionError,
    )
    from genesis_client.exceptions import (
        DaemonNotRunning,
        EpisodeNotFound,
        GenesisError,
    )

    # GenesisError is the root, inherits from Exception.
    assert issubclass(GenesisError, Exception)

    # ConnectionError inherits from both GenesisError and OSError.
    assert issubclass(GenesisConnectionError, GenesisError)
    assert issubclass(GenesisConnectionError, OSError)

    # ProtocolError inherits from both GenesisError and OSError.
    assert issubclass(ProtocolError, GenesisError)
    assert issubclass(ProtocolError, OSError)

    # DaemonNotRunning inherits from ConnectionError (and transitively
    # from OSError). This is critical: genesis_cli.py catches
    # mind.start() errors with `except (OSError, ConnectionError)`.
    # Without this inheritance, DaemonNotRunning would escape as an
    # unhandled traceback when the daemon isn't running.
    assert issubclass(DaemonNotRunning, GenesisConnectionError)
    assert issubclass(DaemonNotRunning, OSError)
    assert issubclass(DaemonNotRunning, GenesisError)

    # EpisodeNotFound inherits from LookupError (the Python convention
    # for "not found" errors, shared with KeyError and IndexError).
    assert issubclass(EpisodeNotFound, GenesisError)
    assert issubclass(EpisodeNotFound, LookupError)


def test_daemon_not_running_caught_by_os_error():
    """DaemonNotRunning must be catchable by `except OSError`.

    This is the exact pattern genesis_cli.py uses at startup:
        try:
            mind.start()
        except (OSError, ConnectionError) as e:
            print(f"Error: cannot connect to daemon: {e}")
    """
    from genesis_client.exceptions import DaemonNotRunning

    try:
        raise DaemonNotRunning("test")
    except OSError as exc:
        assert isinstance(exc, DaemonNotRunning)
    else:
        raise AssertionError("DaemonNotRunning should be caught by except OSError")


def test_episode_not_found_caught_by_lookup_error():
    """EpisodeNotFound must be catchable by `except LookupError`."""
    from genesis_client.exceptions import EpisodeNotFound

    try:
        raise EpisodeNotFound("test")
    except LookupError as exc:
        assert isinstance(exc, EpisodeNotFound)
    else:
        raise AssertionError("EpisodeNotFound should be caught by except LookupError")


# ─── Test runner ──────────────────────────────────────────────


def run_all():
    """Run all tests and report results."""
    tests = [
        test_pack_empty_payload,
        test_pack_with_payload,
        test_unpack_roundtrip,
        test_unpack_zero_length,
        test_unpack_too_short,
        test_unpack_declared_longer_than_actual,
        test_write_and_read_roundtrip,
        test_write_and_read_empty_payload,
        test_multiple_messages_in_sequence,
        test_read_closed_socket,
        test_command_ids,
        test_chemical_ids,
        test_phase_ids,
        test_zone_ids,
        test_name_maps,
        test_module_ids,
        test_module_status_values,
        test_update_module_status_command_id,
        test_handshake_command_id,
        test_protocol_version,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            logger.info(f"  PASS  {test.__name__}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            logger.info(f"  FAIL  {test.__name__}: {e}")
            failed += 1
    logger.info(f"\n  Protocol tests: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
