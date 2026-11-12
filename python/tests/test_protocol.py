"""Unit tests for the Genesis client protocol layer.

These tests verify the binary framing, packing, and unpacking logic
without needing a running daemon. Uses only the standard library.
"""

import logging
import socket
import struct
import sys
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from genesis_client.client import GenesisClient
from genesis_client.exceptions import ConnectionError, ProtocolError
from genesis_client.protocol import (
    ADVANCE_NEURO,
    ADVANCE_PHYSICS,
    ANALYZE_CODE,
    CHEM_BDNF,
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_NAMES,
    CHEM_SEROTONIN,
    DT_NOMINAL_STEP,
    GET_FIELD_PROBE,
    GET_FOLD_MARGINS,
    GET_ION_SUMMARY,
    GET_NEURO_SUMMARY,
    GET_OPTIC_STATE,
    GET_PHASE,
    GET_SENSOR_PRESENCE,
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
    SET_FIELD_CARRIER,
    SET_FIELD_PHASE,
    SET_FOCUS,
    SET_LID,
    SET_OPTIC_GAIN,
    SET_WAKE_ALARM,
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
    # The ion layer is additive and continues past ADVANCE_PHYSICS.
    # genesis2 used 36, but this tree already spent 36 on
    # GET_SENSOR_PRESENCE, so GET_ION_SUMMARY must not collide.
    assert GET_ION_SUMMARY == 40
    assert GET_SENSOR_PRESENCE == 36
    # The field probe is additive and continues past SET_VOLITION;
    # opcode 48 is retired (pinned, never reused) and the carrier
    # write control used by the ring-down experiment stays live.
    assert GET_FIELD_PROBE == 47
    assert SET_FIELD_PHASE == 48
    assert SET_FIELD_CARRIER == 49
    # The optic pair is additive and continues past the field
    # commands: the optic-state read, the corticothalamic gain, and
    # the accommodation request must not collide with them.
    assert GET_OPTIC_STATE == 52
    assert SET_OPTIC_GAIN == 53
    assert SET_FOCUS == 54
    # The fold-margins read is additive past DELETE_EPISODE — the
    # highest core command id — and must not collide with the
    # notification range (100+) or the tensor slab (41–45).
    assert GET_FOLD_MARGINS == 62
    # The voluntary lid is additive past SET_COGNITIVE_PID.
    assert SET_LID == 57


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
    """Protocol version is 3 (must match Rust PROTOCOL_VERSION).

    v3 extended BodyState with the timing/involuntary/senescence
    layer (pulse, throttle, PSI, battery cycles, entropy,
    clocksource, suspend caps) and added SET_WAKE_ALARM.
    """
    # PROTOCOL_VERSION is the *wire protocol* version and is unaffected
    # by the schema bump that added the ion layer (schema v4 lives in
    # the state file, not the IPC framing).
    assert PROTOCOL_VERSION == 3


def _client_with_socket():
    """Helper: client with a mock control-channel socket."""
    client = GenesisClient("/unused/genesis.sock", request_timeout=1.5)
    sock = Mock(spec=socket.socket)
    sock.gettimeout.return_value = 1.5
    client._sock = sock
    client._ever_connected = True
    return client, sock


def _client_with_read_socket():
    """Helper: client with a mock read-channel socket.

    Read-only commands (``_READ_CHANNEL_CMDS``) travel on a second
    daemon connection, so tests exercising GET_*/PING/episode reads
    drive ``_read_sock`` and ``_connect_read`` rather than the control
    channel.
    """
    # The socket path must exist on disk: the client treats a missing
    # socket file as "daemon gone" and fails quietly instead of
    # retrying — retry coverage needs a path that is present.
    tmp = tempfile.NamedTemporaryFile(prefix="genesis-test-", suffix=".sock")
    client = GenesisClient(tmp.name, request_timeout=1.5)
    client._socket_file = tmp  # keep the file alive for the client's life
    sock = Mock(spec=socket.socket)
    sock.gettimeout.return_value = 1.5
    client._read_sock = sock
    client._ever_connected = True
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
        patch(
            "genesis_client.client.read_message",
            side_effect=[
                ConnectionError("response lost"),
                (command, b"\x01"),
            ],
        ),
        patch.object(client, "connect", side_effect=reconnect),
    ):
        with pytest.raises(ConnectionError):
            client._request(command, b"payload")

    send.assert_called_once_with(sock, command, b"payload")
    sock.close.assert_called_once()


def test_client_read_retries_once_and_closes_broken_socket():
    """Test client read retries once and closes broken socket."""
    client, sock = _client_with_read_socket()
    replacement = Mock(spec=socket.socket)

    def reconnect():
        """Swap the client's read socket for the replacement mock."""
        client._read_sock = replacement

    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            side_effect=[
                ConnectionError("response lost"),
                (GET_STATE, b"fresh"),
            ],
        ),
        patch.object(client, "_connect_read", side_effect=reconnect),
    ):
        assert client._request(GET_STATE) == b"fresh"

    assert send.call_count == 2
    sock.close.assert_called_once()
    assert client._read_sock is replacement


def test_client_missing_socket_fails_quietly_without_reconnect(caplog):
    """A lost response with no socket file means teardown — fail, don't retry."""
    client, _ = _client_with_read_socket()
    client.socket_path = "/nonexistent/genesis-test.sock"

    with (
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", side_effect=ConnectionError("lost")),
        patch.object(client, "_connect_read") as connect_read,
        caplog.at_level(logging.WARNING, "genesis_client.client"),
    ):
        with pytest.raises(ConnectionError):
            client._request(GET_STATE)

    connect_read.assert_not_called()
    assert not any("attempting reconnect" in r.message for r in caplog.records)


def test_client_failed_retry_closes_both_sockets():
    """Test client failed retry closes both sockets."""
    client, sock = _client_with_read_socket()
    replacement = Mock(spec=socket.socket)

    def reconnect():
        """Swap the client's read socket for the replacement mock."""
        client._read_sock = replacement

    with (
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", side_effect=ConnectionError("lost")),
        patch.object(client, "_connect_read", side_effect=reconnect),
    ):
        with pytest.raises(ConnectionError):
            client._request(GET_STATE)

    sock.close.assert_called_once()
    replacement.close.assert_called_once()
    assert client._read_sock is None
    assert not client._cache


def test_client_disconnect_invalidates_cached_state():
    """Test client disconnect invalidates cached state."""
    client, sock = _client_with_read_socket()
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


def test_client_reconnects_on_demand_after_connection_lost():
    """A previously-connected client re-establishes the link itself.

    After a daemon restart the next request reopens the socket rather
    than every caller failing until some slower path notices. A client
    that never connected still refuses immediately.
    """
    client = GenesisClient("/unused/genesis.sock", request_timeout=1.5)
    # Never connected: the request must raise, not attempt a connect.
    with pytest.raises(ConnectionError):
        client._request(GET_STATE)

    client._ever_connected = True
    replacement = Mock(spec=socket.socket)
    replacement.gettimeout.return_value = 1.5
    with (
        patch("genesis_client.client.os.path.exists", return_value=True),
        patch("genesis_client.client.socket.socket", return_value=replacement),
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            side_effect=[
                (HANDSHAKE, bytes([PROTOCOL_VERSION])),
                (GET_STATE, b"fresh"),
            ],
        ),
    ):
        assert client._request(GET_STATE) == b"fresh"

    # GET_STATE travels the read channel — the reconnect happens there.
    assert client._read_sock is replacement


def test_client_reads_bypass_held_control_channel():
    """A stalled control-channel request must not starve reads.

    Regression for the head-of-line block observed live: every IPC
    call serialized through one socket lock, so one slow control
    request (think burst, timed-out write) parked heartbeat,
    regulator, and volition callers on ``_lock`` — including the
    ``should_stop`` reads that let workers clear at shutdown. Pure
    reads now travel a second connection under ``_read_lock``.
    """
    client, read_sock = _client_with_read_socket()
    control_sock = Mock(spec=socket.socket)
    client._sock = control_sock

    # Simulate an in-flight stalled control request holding _lock.
    acquired = client._lock.acquire(timeout=1.0)
    assert acquired
    try:
        with (
            patch("genesis_client.client.write_message") as send,
            patch(
                "genesis_client.client.read_message",
                return_value=(GET_PHASE, b"\x00"),
            ),
        ):
            # Must complete without ever touching the held _lock.
            assert client._request(GET_PHASE) == b"\x00"
        send.assert_called_once_with(read_sock, GET_PHASE, b"")
    finally:
        client._lock.release()


def test_client_control_request_uses_control_channel():
    """Writes stay on the control channel even with a live read socket."""
    client, read_sock = _client_with_read_socket()
    client._sock = Mock(spec=socket.socket)
    client._sock.gettimeout.return_value = 1.5

    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(STORE_EVENT, b"\x01"),
        ),
    ):
        client._request(STORE_EVENT, b"payload")

    send.assert_called_once_with(client._sock, STORE_EVENT, b"payload")
    read_sock.sendall.assert_not_called()
    read_sock.shutdown.assert_not_called()
    read_sock.close.assert_not_called()


def test_client_disconnect_closes_both_channels():
    """disconnect() tears down the read channel too."""
    client, read_sock = _client_with_read_socket()
    control_sock = Mock(spec=socket.socket)
    client._sock = control_sock

    client.disconnect()

    control_sock.close.assert_called_once()
    read_sock.close.assert_called_once()
    assert client._sock is None
    assert client._read_sock is None


def test_client_connect_replaces_socket_and_respects_timeout():
    """Test client connect replaces socket and respects timeout."""
    client, sock = _client_with_socket()
    client._cache[GET_STATE] = (0.0, b"old")
    replacement = Mock(spec=socket.socket)
    with (
        patch("genesis_client.client.os.path.exists", return_value=True),
        patch("genesis_client.client.socket.socket", return_value=replacement),
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(
                HANDSHAKE,
                bytes([PROTOCOL_VERSION]),
            ),
        ),
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


def test_client_cached_read_uses_read_channel_lock():
    """Reads serialize on the read-channel lock, not the control lock."""
    client, _ = _client_with_read_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", return_value=(GET_STATE, b"old")),
    ):
        client._request(GET_STATE)
        guard = MagicMock()
        guard.__enter__.return_value = None
        guard.__exit__.return_value = False
        with patch.object(client, "_read_lock", guard):
            assert client._request(GET_STATE) == b"old"
        guard.__enter__.assert_called_once()


def test_client_cache_expires_despite_wall_clock_rollback():
    """Test client cache expires despite wall clock rollback."""
    client, _ = _client_with_read_socket()
    with (
        patch("genesis_client.client.time") as clock,
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            side_effect=[
                (GET_STATE, b"old"),
                (GET_STATE, b"fresh"),
            ],
        ),
    ):
        clock.time.side_effect = [200.0, 100.0]
        # Req 1: t0, elapsed, cache-store. Req 2: freshness check
        # (stale → falls through), t0, elapsed, cache-store.
        clock.monotonic.side_effect = [10.0, 10.2, 10.2, 11.0, 11.0, 11.0, 11.0]
        assert client._request(GET_STATE) == b"old"
        assert client._request(GET_STATE) == b"fresh"


def test_client_discards_connection_after_response_command_mismatch():
    """Test client discards connection after response command mismatch."""
    client, sock = _client_with_read_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch("genesis_client.client.read_message", return_value=(STORE_EVENT, b"\x01")),
    ):
        with pytest.raises(ProtocolError):
            client._request(GET_STATE)

    sock.close.assert_called_once()
    assert client._read_sock is None


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

    client, _ = _client_with_read_socket()
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

    client, _ = _client_with_read_socket()
    # 60 bytes of data — first byte is 0x9A (part of an f32 mantissa),
    # which has the high bit set but is NOT an error response.
    data = bytes([0x9A]) + b"\x99\x99\x3e" + b"\x00" * 56
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
        (0.5, 0.5),  # in range — passed through unchanged
        (100.0, 10.0),  # stall longer than the daemon's max — truncated
        (0.0, 0.001),  # degenerate zero elapsed — clamped to min
        (-5.0, 0.001),  # negative (clock anomaly) — clamped to min
    ],
)
def test_advance_physics_clamps_dt(dt_sent, dt_expected):
    """advance_physics must clamp dt to the body's [0.001, 10.0] range.

    This is the evaluation-only command, whose whole purpose is to let
    an offline assay vary dt as its independent variable. It enforces
    the same bounds the daemon applies to its own measured elapsed time,
    so an experiment cannot request a step the production integrator
    would refuse.
    """
    import struct

    client, _ = _client_with_socket()
    # Valid response: ack + 4x f32 + u32 = 21 bytes (same shape as
    # ADVANCE_NEURO, so the decode path is shared).
    ok_resp = b"\x01" + b"\x00\x00\x80\x3f" * 4 + b"\x00" * 4
    assert len(ok_resp) == 21
    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(ADVANCE_PHYSICS, ok_resp),
        ),
    ):
        result = client.advance_physics(dt_sent)

    assert result is not None
    sent_payload = send.call_args[0][2]
    (packed_dt,) = struct.unpack("<f", sent_payload)
    assert abs(packed_dt - dt_expected) < 1e-6


def test_advance_neuro_takes_no_dt_from_the_caller():
    """The production command must not accept a caller-chosen step.

    Time ownership is the daemon's. `advance_neuro` takes no `dt` and
    sends the nominal step as a placeholder the daemon ignores, so the
    coupling that let the mind dictate the body's integration rate cannot
    be reintroduced from this layer. Pinning the signature means a
    future caller passing `dt=` fails loudly instead of quietly
    reintroducing two clocks on one body.
    """
    import inspect

    params = inspect.signature(GenesisClient.advance_neuro).parameters
    assert "dt" not in params, "advance_neuro must not accept dt; the daemon owns time"

    client, _ = _client_with_socket()
    ok_resp = b"\x01" + b"\x00\x00\x80\x3f" * 4 + b"\x00" * 4
    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(ADVANCE_NEURO, ok_resp),
        ),
    ):
        assert client.advance_neuro() is not None

    # Whatever is on the wire is the nominal step, not a caller's choice.
    import struct

    (packed,) = struct.unpack("<f", send.call_args[0][2])
    assert abs(packed - DT_NOMINAL_STEP) < 1e-6


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


def test_set_wake_alarm_packs_u64_and_returns_armed_epoch():
    """set_wake_alarm sends [u64 epoch] and returns the armed epoch."""
    epoch = 1_800_000_000
    armed_resp = b"\x01" + struct.pack("<Q", epoch)
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_WAKE_ALARM, armed_resp),
        ),
    ):
        result = client.set_wake_alarm(epoch)

    assert result == epoch
    # The request went out as SET_WAKE_ALARM with a u64 LE payload.
    assert send.call_args[0][1] == SET_WAKE_ALARM
    assert send.call_args[0][2] == struct.pack("<Q", epoch)


def test_set_wake_alarm_disarm_sends_zero():
    """Disarming sends epoch 0 and returns the cleared alarm state."""
    ok_resp = b"\x01" + struct.pack("<Q", 0)
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_WAKE_ALARM, ok_resp),
        ),
    ):
        result = client.set_wake_alarm(0)

    assert result == 0
    assert send.call_args[0][2] == struct.pack("<Q", 0)


def test_set_wake_alarm_returns_none_on_helper_failure():
    """A [0][current] response means the helper failed → None."""
    fail_resp = b"\x00" + struct.pack("<Q", 0)
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_WAKE_ALARM, fail_resp),
        ),
    ):
        assert client.set_wake_alarm(1_800_000_000) is None


def test_set_wake_alarm_returns_none_on_malformed_response():
    """A truncated response can't carry an armed epoch → None."""
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_WAKE_ALARM, b"\x01\x02"),
        ),
    ):
        assert client.set_wake_alarm(1_800_000_000) is None


def test_set_focus_packs_f32_meters_and_acks():
    """set_focus sends [f32 meters] on the control channel and maps
    the ack byte to True."""
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_FOCUS, b"\x01"),
        ),
    ):
        assert client.set_focus(0.5) is True

    assert send.call_args[0][1] == SET_FOCUS
    assert send.call_args[0][2] == struct.pack("<f", 0.5)


def test_set_focus_zero_relaxes_the_lens():
    """A non-positive distance is the body's relax-to-infinity — the
    client passes it through; the daemon clamps it to 0 D."""
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_FOCUS, b"\x01"),
        ),
    ):
        assert client.set_focus(0.0) is True

    assert send.call_args[0][2] == struct.pack("<f", 0.0)


def test_set_focus_returns_false_when_the_daemon_predates_it():
    """An unacked or missing response is an absent channel, not a
    silent success."""
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_FOCUS, b"\x00"),
        ),
    ):
        assert client.set_focus(2.0) is False

    # A daemon that closes mid-request reads as unsupported too.
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            side_effect=ConnectionError("gone"),
        ),
        patch.object(client, "connect"),
    ):
        assert client.set_focus(2.0) is False


def test_set_lid_packs_f32_openness_and_acks():
    """set_lid sends [f32 openness] on the control channel and maps
    the ack byte to True — the voluntary lid is a held position."""
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_LID, b"\x01"),
        ),
    ):
        assert client.set_lid(0.05) is True

    assert send.call_args[0][1] == SET_LID
    assert send.call_args[0][2] == struct.pack("<f", 0.05)


def test_set_lid_returns_false_when_the_daemon_predates_it():
    """A body without the command cannot hold the lid — False marks
    the absent channel rather than a silent success."""
    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_LID, b"\x00"),
        ),
    ):
        assert client.set_lid(0.05) is False

    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            side_effect=ConnectionError("gone"),
        ),
        patch.object(client, "connect"),
    ):
        assert client.set_lid(0.05) is False


def test_set_cognitive_pid_packs_u32_and_acks():
    """set_cognitive_pid sends [u32 pid] on the control channel so a
    re-inhabiting mind replaces the dead launcher's env-var PID."""
    from genesis_client.protocol import SET_COGNITIVE_PID

    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_COGNITIVE_PID, b"\x01"),
        ),
    ):
        assert client.set_cognitive_pid(4242) is True

    assert send.call_args[0][1] == SET_COGNITIVE_PID
    assert send.call_args[0][2] == struct.pack("<I", 4242)


def test_set_cognitive_pid_false_when_the_daemon_predates_it():
    """A body that predates the command cannot adopt the new PID —
    False marks the absent channel rather than a silent success."""
    from genesis_client.protocol import SET_COGNITIVE_PID

    client, _ = _client_with_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(SET_COGNITIVE_PID, b"\x00"),
        ),
    ):
        assert client.set_cognitive_pid(4242) is False

    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            side_effect=ConnectionError("gone"),
        ),
        patch.object(client, "connect"),
    ):
        assert client.set_cognitive_pid(4242) is False


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


def test_ion_summary_unpack_matches_wire_layout():
    """IonSummary decodes the daemon's fixed 108-byte block.

    The daemon serializes 27 little-endian f32 in a pinned field order
    (four IonId arrays, then eleven scalars). If either side reorders a
    field the payload still unpacks — it just reports the wrong ion —
    so the test pins the order rather than only the width.
    """
    import struct

    from genesis_client.types import IonSummary

    values = [float(i) for i in range(27)]
    raw = struct.pack("<27f", *values)
    ions = IonSummary.unpack(raw)

    assert len(raw) == 108
    assert ions.intracellular_mm == (0.0, 1.0, 2.0, 3.0)
    assert ions.extracellular_mm == (4.0, 5.0, 6.0, 7.0)
    assert ions.reversal_potential_mv == (8.0, 9.0, 10.0, 11.0)
    assert ions.conductance == (12.0, 13.0, 14.0, 15.0)
    assert ions.membrane_potential_mv == 16.0
    assert ions.nak_pump_rate == 17.0
    assert ions.kcl_cotransporter_flux == 18.0
    assert ions.ncx_flux == 19.0
    assert ions.atp_availability == 20.0
    assert ions.calcium_signal == 21.0
    assert ions.chloride_efficacy == 22.0
    assert ions.excitability == 23.0
    assert ions.gradient_integrity == 24.0
    assert ions.energy_load == 25.0
    assert ions.net_membrane_current == 26.0

    # IonId order is calcium, chloride, potassium, sodium.
    assert ions.ion_names == ("calcium", "chloride", "potassium", "sodium")
    assert ions.intracellular("potassium") == 2.0
    assert ions.extracellular("sodium") == 7.0
    assert ions.reversal_potential("calcium") == 8.0

    with pytest.raises(ValueError, match="108 bytes"):
        IonSummary.unpack(raw[:107])


def test_fold_margins_unpack_matches_wire_layout():
    """FoldMargins decodes the daemon's fixed 40-byte block.

    The daemon serializes four little-endian f64 — homeostat, the two
    fold edges, the normalized position — then a u8 bistable flag and
    7 padding bytes. The order is pinned so a field reorder on either
    side surfaces as a test failure rather than a silently-swapped
    edge.
    """
    import struct

    from genesis_client.types import FoldMargins

    raw = struct.pack("<4d", 214.0, 100.0, 300.0, 0.379) + bytes([1]) + bytes(7)
    m = FoldMargins.unpack(raw)

    assert len(raw) == 40
    assert m.h == 214.0
    assert m.h_sleep_fold == 100.0
    assert m.h_wake_fold == 300.0
    assert m.position == 0.379
    assert m.bistable is True

    raw_mono = struct.pack("<4d", 50.0, 100.0, 300.0, -0.25) + bytes([0]) + bytes(7)
    assert FoldMargins.unpack(raw_mono).bistable is False

    with pytest.raises(ValueError, match="40 bytes"):
        FoldMargins.unpack(raw[:39])


def test_set_volition_is_retired_opcode_46() -> None:
    """The SET_VOLITION wire number is retired — volitional pushes onto the
    sleep-wake network were removed in favour of chemical impulses — but
    the constant is kept pinned at 46 so the number is never reused."""
    from genesis_client import protocol

    assert protocol.SET_VOLITION == 46
    assert "SET_VOLITION" in protocol.__all__
    assert not hasattr(protocol, "set_volition")


def test_neuro_adjust_baseline_is_retired_opcode_23() -> None:
    """The NEURO_ADJUST_BASELINE wire number is retired — direct baseline
    edits bypassed the body's own dynamics; NEURO_IMPULSE's bounded,
    decaying impulses are the only sanctioned chemistry channel — but
    the constant is kept pinned at 23 so the number is never reused."""
    from genesis_client import protocol

    assert protocol.NEURO_ADJUST_BASELINE == 23
    assert "NEURO_ADJUST_BASELINE" in protocol.__all__
    assert not hasattr(protocol, "neuro_adjust_baseline")


def test_set_field_phase_is_retired_opcode_48() -> None:
    """The SET_FIELD_PHASE wire number is retired — modeled band phase is
    no longer pushed into the probe's physical generator; the phase lock
    runs physics → model only — but the constant is kept pinned at 48
    so the number is never reused."""
    from genesis_client import protocol

    assert protocol.SET_FIELD_PHASE == 48
    assert "SET_FIELD_PHASE" in protocol.__all__
    assert not hasattr(protocol, "set_field_phase")


def test_get_internals_is_opcode_51_and_decodes_lines() -> None:
    """Process-local daemon internals — pinned against the Rust daemon's
    `cmd::GET_INTERNALS` (51) and decoded from `[u8 ack]` + `key=value` lines."""
    from genesis_client import protocol

    assert protocol.GET_INTERNALS == 51
    assert "GET_INTERNALS" in protocol.__all__

    client, _ = _client_with_read_socket()
    body = b"\x01tick_count=42\ngate.intero.period_ms=6000\n"
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(protocol.GET_INTERNALS, body),
        ),
    ):
        internals = client.get_internals()

    assert internals == {
        "tick_count": "42",
        "gate.intero.period_ms": "6000",
    }


def test_analyze_code_request_payload_and_decode() -> None:
    """analyze_code packs [u16 path][path][u32 src][src], travels the
    read channel (pure analysis — never queues behind writes), and
    decodes the tab-separated line report."""
    from genesis_client import protocol

    assert protocol.ANALYZE_CODE == 55
    assert "ANALYZE_CODE" in protocol.__all__

    client, read_sock = _client_with_read_socket()
    report = (
        b"file\tdemo.rs\n"
        b"lines\t12\n"
        b"functions\t1\n"
        b"avg_complexity\t1.5\n"
        b"unit\tfunction\thelper\thelper\t1-4\tpub\ti32;i32\ti32"
        b"\tadd\tVec;HashMap\t1\t0\t3\tAdds things.\n"
        b"impl\tConfig\tHandler\n"
        b"impl\tConfig\t-\n"
        b"edge\tcaller\thelper\n"
    )
    with (
        patch("genesis_client.client.write_message") as send,
        patch(
            "genesis_client.client.read_message",
            return_value=(ANALYZE_CODE, b"\x01" + report),
        ),
    ):
        result = client.analyze_code("demo.rs", "pub fn helper() {}")

    assert result is not None
    assert result["file"] == "demo.rs"
    assert result["stats"]["functions"] == 1
    assert result["stats"]["avg_complexity"] == 1.5
    unit = result["units"][0]
    assert unit["kind"] == "function"
    assert unit["name"] == "helper"
    assert unit["span"] == (1, 4)
    assert unit["public"] is True
    assert unit["params"] == ["i32", "i32"]
    assert unit["return_type"] == "i32"
    assert unit["calls"] == ["add"]
    assert unit["refs"] == ["Vec", "HashMap"]
    assert unit["stmts"] == 3
    assert unit["docs"] == "Adds things."
    assert result["impls"] == [("Config", "Handler"), ("Config", None)]
    assert result["edges"] == [("caller", "helper")]

    # Payload shape: [u16 path_len][path][u32 src_len][source].
    sent_cmd, sent_payload = send.call_args[0][1], send.call_args[0][2]
    assert sent_cmd == ANALYZE_CODE
    path_len = struct.unpack("<H", sent_payload[:2])[0]
    assert sent_payload[2 : 2 + path_len] == b"demo.rs"
    src_off = 2 + path_len
    src_len = struct.unpack("<I", sent_payload[src_off : src_off + 4])[0]
    assert sent_payload[src_off + 4 : src_off + 4 + src_len] == b"pub fn helper() {}"
    # Pure analysis rides the read channel, not the control socket.
    assert send.call_args[0][0] is read_sock


def test_analyze_code_returns_none_on_daemon_rejection() -> None:
    """A parse failure (INVALID_VALUE) or dead daemon yields None —
    an absent channel, never a fabricated reading."""
    client, _ = _client_with_read_socket()
    with (
        patch("genesis_client.client.write_message"),
        patch(
            "genesis_client.client.read_message",
            return_value=(ANALYZE_CODE, b"\x05"),
        ),
    ):
        assert client.analyze_code("bad.rs", "fn broken( {") is None

    client2 = GenesisClient("/unused/genesis.sock", request_timeout=1.5)
    assert client2.analyze_code("x.rs", "fn x() {}") is None


def test_decode_code_analysis_tolerates_partial_reports() -> None:
    """Malformed numeric fields degrade to zero rather than poison
    the reading; comma-bearing type strings survive the ';' split."""
    from genesis_client.client import _decode_code_analysis

    report = (
        "file\tdemo.rs\n"
        "functions\tNaN\n"
        "unit\tfunction\tf\tf\t-\tpub,async"
        "\tHashMap<String, Vec<u8>>;(u8, u8)\t\t\t\tNaN\t0\t1\t\n"
    )
    decoded = _decode_code_analysis(report)
    assert decoded["stats"]["functions"] == 0
    unit = decoded["units"][0]
    assert unit["span"] == (0, 0)
    assert unit["async_"] is True
    assert unit["params"] == ["HashMap<String, Vec<u8>>", "(u8, u8)"]
    assert unit["branches"] == 0
