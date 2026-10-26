"""Genesis client — the cognitive mind's bridge to the subcognitive.

This is the Python library that the cognitive mind uses to communicate
with the Rust subcognitive daemon. It connects to the daemon's Unix
domain socket and provides a clean, Pythonic API for all IPC commands.

The protocol is request-response only. Subcognitive→cognitive
notifications (phase changes, dream insights) are handled separately
by ``infrastructure.notifications``, a pull-based queue that
detects state changes during the polling cycle — not by this client.

# Usage

    from genesis_client import GenesisClient

    client = GenesisClient("/path/to/genesis.sock")
    client.connect()

    # Check if the daemon is alive
    pong = client.ping()
    print(f"Daemon uptime: {pong.uptime_ms}ms")

    # Read how Genesis feels
    summary = client.get_neuro_summary()
    print(f"Phase: {summary.phase_name}")
    print(f"Arousal: {summary.arousal:.2f}")
    print(f"Valence: {summary.valence:.2f}")

    # Store an event in short-term memory
    client.store_event(
        timestamp=int(time.time() * 1000),
        event_type=0,  # UserInput
        source_module=4,  # Language
        salience=0.8,
        emotional_tag=[0.7, 0.5, 0.6, 0.5, 0.4, 0.5,
                       0.2, 0.3, 0.4, 0.4, 0.1, 0.4],
        text="User said hello",
    )

    # Apply a neurochemical impulse (e.g., dopamine boost from reward)
    client.neuro_impulse(chem=0, magnitude=0.3)  # dopamine +0.3

    # Set the cognitive zone (what Genesis is doing)
    client.set_zone(2)  # Coding

    # Search for similar memories
    results = client.find_similar("hello world greeting", limit=5)

    client.disconnect()
"""

from __future__ import annotations

import logging
import os
import socket
import struct
import threading
import time
from typing import Any

from .exceptions import (
    ConnectionError as GenesisConnectionError,
)
from .exceptions import (
    DaemonNotRunning,
    EpisodeNotFound,
    ProtocolError,
)
from .protocol import (
    _F32,
    _U16,
    _U32,
    _U64,
    ADVANCE_NEURO,
    ADVANCE_PHYSICS,
    ANALYZE_CODE,
    APPLY_BODY_CONTROL,
    ARCHIVE_EPISODE,
    ASSOCIATE,
    # Chemical IDs (for convenience wrappers)
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_GABA,
    CHEM_OXYTOCIN,
    CONSOLIDATE,
    DELETE_EPISODE,
    DREAM,
    DROP_TENSOR,
    DT_MAX,
    DT_MIN,
    DT_NOMINAL_STEP,
    EMOTIONAL_TAG_SIZE,
    FIND_SIMILAR,
    FIND_SIMILAR_ARCHIVED,
    GET_BODY_CONTROL,
    GET_BODY_STATE,
    GET_FIELD_PROBE,
    GET_INFERENCE_SUMMARY,
    GET_INTERNALS,
    GET_ION_SUMMARY,
    GET_MEMORY_STATS,
    GET_MOTOR_TARGET,
    GET_NEURO_SUMMARY,
    GET_OPTIC_STATE,
    GET_PHASE,
    GET_PLASTICITY_PROFILE,
    GET_PROCESS_HEALTH,
    GET_RECENT_EPISODES,
    GET_SENSOR_PRESENCE,
    # Command IDs
    GET_STATE,
    GET_SUBSYSTEM_TELEMETRY,
    GET_TENSOR,
    GET_ZONE_TRANSITIONS,
    HANDSHAKE,
    LIST_TENSORS,
    # Module IDs (for validation and store_observation)
    MODULE_GUARDRAILS,
    MODULE_SENSORY,
    # Module status (for heartbeat_module)
    MODULE_STATUS_ERROR,
    MODULE_STATUS_IDLE,
    MODULE_STATUS_RUNNING,
    NEURO_ADJUST_BASELINE,
    NEURO_IMPULSE,
    PING,
    PROTOCOL_VERSION,
    PUT_TENSOR,
    READ_SENSORS,
    RETRIEVE_EPISODE,
    SAVE_INFERENCE,
    SEARCH_ARCHIVED,
    SEARCH_EPISODES,
    SET_COGNITIVE_PID,
    SET_FIELD_CARRIER,
    SET_FIELD_PHASE,
    SET_FOCUS,
    SET_LID,
    SET_OPTIC_GAIN,
    SET_POLICY_AUTHORITY,
    SET_WAKE_ALARM,
    SET_ZONE,
    SHUTDOWN,
    STORE_EPISODE,
    STORE_EVENT,
    SYNC,
    UNARCHIVE_EPISODE,
    UPDATE_MODULE_STATUS,
    UPDATE_USER_AFFECT,
    # Zone IDs (for validation)
    ZONE_SLEEPING,
    # Error detection
    error_name,
    is_error_byte,
    # Wire helpers
    read_message,
    write_message,
)
from .swallow import note_swallowed
from .tensor import (
    MotorTarget,
    Tensor,
    TensorMeta,
    pack_name,
    unpack_list,
    unpack_motor_target,
)
from .types import (
    ArchivedEpisode,
    BodyControlState,
    BodyState,
    CoreState,
    Episode,
    FieldProbeSignature,
    FieldState,
    InferenceSummary,
    IonSummary,
    MemoryStats,
    NeuroSummary,
    OpticState,
    PhaseInfo,
    PingResponse,
    PlasticityProfile,
    ProcessHealth,
    RecentEpisode,
    SensorPresence,
    SimilarEpisode,
    SubsystemReport,
    ZoneTransitions,
    unpack_archived_episodes,
    unpack_recent_episodes,
    unpack_similar_results,
)

logger = logging.getLogger(__name__)


class GenesisClient:
    """Client for the Genesis subcognitive daemon.

    Connects to the daemon via a Unix domain socket and provides
    a Pythonic API for all IPC commands.

    Args:
        socket_path: Path to the daemon's Unix socket file.

    Raises:
        DaemonNotRunning: If the socket file doesn't exist (daemon not running).
    """

    def __init__(self, socket_path: str, request_timeout: float = 30.0) -> None:
        """Initialize the client with a socket path and request timeout."""
        self.socket_path = socket_path
        self._sock: socket.socket | None = None
        self._read_sock: socket.socket | None = None
        self._ever_connected = False
        self._lock = threading.RLock()
        self._read_lock = threading.RLock()
        self._cache_lock = threading.Lock()
        self.request_timeout = request_timeout
        # Short-TTL cache for frequently-called read-only commands.
        # Background threads (inner_life, learner, regulator) call
        # get_state/get_neuro_summary many times per second. The daemon
        # is reactive (state advances when the mind calls advance_neuro,
        # roughly once per heartbeat cycle), so a 150ms cache is fresh
        # enough for emotion assessment while eliminating redundant
        # socket round-trips that contend with the conversation thread
        # for the IPC lock.
        self._cache: dict[int, tuple[float, bytes]] = {}
        self._cache_ttl = 0.150  # 150ms

    def connect(self) -> None:
        """Connect to the daemon with protocol version handshake.

        Raises:
            DaemonNotRunning: If the socket file doesn't exist.
            GenesisConnectionError: If the connection fails.
            ProtocolError: If the daemon's protocol version doesn't match.
        """
        with self._lock:
            self.disconnect()
            self._connect()

    @staticmethod
    def _close(sock: socket.socket | None, site: str) -> None:
        """Close a channel socket, interrupting any in-flight read.

        ``shutdown`` runs before ``close`` so a thread parked in
        ``read_message`` on this socket is released immediately rather
        than lingering until its read timeout fires.
        """
        if sock is None:
            return
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError as e:
            # not connected or already torn down
            note_swallowed(site, e)
        try:
            sock.close()
        except OSError as e:
            note_swallowed(site, e)

    def _close_socket(self) -> None:
        """Close the control socket if open and clear the reference.

        Does NOT clear the cache or acquire the lock — callers already
        hold ``_lock`` (the request path) or cleared it themselves.
        """
        self._close(self._sock, "genesis_client.client._close_socket")
        self._sock = None

    def _close_read_socket(self) -> None:
        """Close the read-channel socket (caller holds ``_read_lock``)."""
        self._close(self._read_sock, "genesis_client.client._close_read_socket")
        self._read_sock = None

    def _open_channel(self) -> socket.socket:
        """Open one socket to the daemon and run the handshake.

        Used for both the control channel (``_connect``) and the
        read-only channel (``_ensure_read_connected``). Returns the
        connected, version-checked socket; raises on any failure after
        closing the half-open socket.
        """
        if not os.path.exists(self.socket_path):
            raise DaemonNotRunning(f"socket not found: {self.socket_path} (is the daemon running?)")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.request_timeout)  # 30s timeout — daemon should never take longer
        try:
            sock.connect(self.socket_path)
        except OSError as e:
            self._close(sock, "genesis_client.client._open_channel")
            raise GenesisConnectionError(f"failed to connect: {e}") from e

        # Protocol version handshake
        try:
            write_message(sock, HANDSHAKE, bytes([PROTOCOL_VERSION]))
            resp_cmd, resp_data = read_message(sock)
        except OSError as e:
            self._close(sock, "genesis_client.client._open_channel")
            raise GenesisConnectionError(f"handshake failed: {e}") from e

        if resp_cmd != HANDSHAKE or not resp_data:
            self._close(sock, "genesis_client.client._open_channel")
            raise ProtocolError(f"handshake failed: unexpected response (cmd={resp_cmd})")
        if is_error_byte(resp_data[0]):
            # The daemon signalled a version mismatch: the first byte
            # is ERROR_VERSION_MISMATCH and the second (if present) is
            # its actual version.
            self._close(sock, "genesis_client.client._open_channel")
            actual = resp_data[1] if len(resp_data) > 1 else None
            raise ProtocolError(
                f"protocol version mismatch: client={PROTOCOL_VERSION}, "
                f"daemon={actual if actual is not None else 'unknown'}"
            )
        daemon_version = resp_data[0]
        if daemon_version != PROTOCOL_VERSION:
            self._close(sock, "genesis_client.client._open_channel")
            raise ProtocolError(
                f"protocol version mismatch: client={PROTOCOL_VERSION}, daemon={daemon_version}"
            )
        return sock

    def _connect(self) -> None:
        """Open the control socket and perform the handshake (no lock)."""
        self._sock = self._open_channel()
        self._ever_connected = True

    def _connect_read(self) -> None:
        """Open the read-only channel (caller holds ``_read_lock``)."""
        self._read_sock = self._open_channel()

    def disconnect(self) -> None:
        """Close the connection to the daemon."""
        with self._lock:
            with self._cache_lock:
                self._cache.clear()
            self._close_socket()
            # The read channel is closed without taking _read_lock so
            # disconnect can never block behind a stalled read; the
            # in-flight read sees the socket die and reconnects lazily.
            self._close(self._read_sock, "genesis_client.client.disconnect")
            self._read_sock = None

    def __enter__(self) -> GenesisClient:
        """Context manager entry — connect and return self."""
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Context manager exit — disconnect."""
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Whether the client currently has an open socket connection."""
        return self._sock is not None

    def _ensure_connected(self) -> socket.socket:
        """Return the open socket, reconnecting if the link dropped.

        A client that was connected before reconnects on demand: after
        a daemon restart the next request re-establishes the link
        instead of every caller failing until some slower path notices.
        A client that was never connected still raises, and a clean
        ``disconnect()`` while the daemon is genuinely down fails the
        same way it always did — the connect attempt just reports the
        real reason (socket missing, refused) instead of a bare
        "not connected".
        """
        if self._sock is None:
            if not self._ever_connected:
                raise GenesisConnectionError("not connected — call connect() first")
            self._connect()
        sock = self._sock
        if sock is None:
            raise GenesisConnectionError("not connected")
        return sock

    def _ensure_read_connected(self) -> socket.socket:
        """Return the read-channel socket, opening it on first use.

        The read channel is a second daemon connection carrying only
        side-effect-free commands (``_READ_CHANNEL_CMDS``). The daemon
        serves each connection on its own thread, so reads never queue
        behind a burst of writes on the control channel — a stalled or
        merely busy ``think()`` no longer starves heartbeat, regulator,
        and intention telemetry reads. Same reconnect rule as the
        control channel: only a previously-connected client reopens.
        """
        if self._read_sock is None:
            if not self._ever_connected:
                raise GenesisConnectionError("not connected — call connect() first")
            self._connect_read()
        sock = self._read_sock
        if sock is None:
            raise GenesisConnectionError("not connected")
        return sock

    # Commands served on the read channel — every one is a pure read
    # with no daemon-side effect, so reordering a read against a
    # concurrent write can at worst observe slightly older state, the
    # same staleness the 150ms cache already allows. Write/idempotent-
    # write commands (STORE_*, NEURO_*, PUT/DROP_TENSOR, ADVANCE_*,
    # UPDATE_*, SET_*, READ_SENSORS, SAVE_INFERENCE, ARCHIVE_EPISODE,
    # SHUTDOWN, SYNC, CONSOLIDATE, ASSOCIATE, DREAM) stay on the
    # control channel where their ordering is preserved.
    _READ_CHANNEL_CMDS = frozenset(
        {
            PING,
            ANALYZE_CODE,
            GET_STATE,
            GET_NEURO_SUMMARY,
            GET_PHASE,
            GET_MEMORY_STATS,
            GET_INFERENCE_SUMMARY,
            GET_BODY_STATE,
            GET_BODY_CONTROL,
            GET_PLASTICITY_PROFILE,
            GET_FIELD_PROBE,
            GET_INTERNALS,
            GET_ION_SUMMARY,
            GET_MOTOR_TARGET,
            GET_OPTIC_STATE,
            GET_PROCESS_HEALTH,
            GET_RECENT_EPISODES,
            GET_SENSOR_PRESENCE,
            GET_SUBSYSTEM_TELEMETRY,
            GET_TENSOR,
            GET_ZONE_TRANSITIONS,
            LIST_TENSORS,
            RETRIEVE_EPISODE,
            FIND_SIMILAR,
            FIND_SIMILAR_ARCHIVED,
            SEARCH_ARCHIVED,
            SEARCH_EPISODES,
        }
    )

    # Read-only commands that are safe to cache with a short TTL.
    # State only changes when the mind advances neurochemistry (via
    # ADVANCE_NEURO) or issues a write command, so cached values are
    # at most one heartbeat cycle stale — acceptable for emotion/state
    # assessment in background threads. Write commands (STORE_EVENT,
    # NEURO_IMPULSE, SET_ZONE, SYNC, SHUTDOWN, UPDATE_*) are never
    # cached.
    _CACHEABLE_CMDS = frozenset(
        {
            GET_STATE,
            GET_NEURO_SUMMARY,
            GET_PHASE,
            GET_MEMORY_STATS,
            GET_INFERENCE_SUMMARY,
            GET_BODY_STATE,
            GET_BODY_CONTROL,
            GET_PLASTICITY_PROFILE,
        }
    )

    # Write commands that invalidate the cache. When these are sent,
    # all cached read-only responses are cleared so the next read
    # sees the updated state. Without this, a neuro_impulse followed
    # immediately by feel() would return the pre-impulse state for
    # up to 150ms.
    _CACHE_INVALIDATING_CMDS = frozenset(
        {
            STORE_EVENT,
            NEURO_IMPULSE,
            NEURO_ADJUST_BASELINE,
            SET_ZONE,
            SYNC,
            SHUTDOWN,
            UPDATE_MODULE_STATUS,
            UPDATE_USER_AFFECT,
            ARCHIVE_EPISODE,
            UNARCHIVE_EPISODE,
            DELETE_EPISODE,
            STORE_EPISODE,
            ADVANCE_NEURO,
            ADVANCE_PHYSICS,
            CONSOLIDATE,
            ASSOCIATE,
            DREAM,
            READ_SENSORS,
            APPLY_BODY_CONTROL,
            SAVE_INFERENCE,
            SET_POLICY_AUTHORITY,
            SET_WAKE_ALARM,
        }
    )

    # All tensor commands are idempotent (last-write-wins put,
    # read-only gets/list, idempotent remove), so every one may be
    # retried after a reconnect.
    _RETRYABLE_CMDS = _CACHEABLE_CMDS | frozenset(
        {
            PING,
            RETRIEVE_EPISODE,
            FIND_SIMILAR,
            FIND_SIMILAR_ARCHIVED,
            GET_RECENT_EPISODES,
            SEARCH_ARCHIVED,
            SEARCH_EPISODES,
            GET_OPTIC_STATE,
            PUT_TENSOR,
            GET_TENSOR,
            LIST_TENSORS,
            DROP_TENSOR,
            GET_MOTOR_TARGET,
        }
    )

    # Commands whose response first byte is an ack/status byte (1 =
    # success, 0 = not-found/invalid, 0x80+ = error). For these, the
    # error check examines the first byte regardless of response length
    # — the daemon pads error responses to the expected size (e.g.
    # STORE_EVENT returns 8 bytes, ADVANCE_NEURO returns 21 bytes).
    _ACK_CMDS = frozenset(
        {
            PING,
            STORE_EVENT,
            STORE_EPISODE,
            SET_ZONE,
            NEURO_IMPULSE,
            SYNC,
            SHUTDOWN,
            UPDATE_MODULE_STATUS,
            UPDATE_USER_AFFECT,
            ARCHIVE_EPISODE,
            UNARCHIVE_EPISODE,
            DELETE_EPISODE,
            NEURO_ADJUST_BASELINE,
            ADVANCE_NEURO,
            ADVANCE_PHYSICS,
            CONSOLIDATE,
            ASSOCIATE,
            DREAM,
            READ_SENSORS,
            APPLY_BODY_CONTROL,
            SAVE_INFERENCE,
            SET_POLICY_AUTHORITY,
            SET_WAKE_ALARM,
            # PUT/DROP ack with a single byte. GET_TENSOR's leading
            # found-byte is 0 or 1, so it can never alias an error code.
            PUT_TENSOR,
            DROP_TENSOR,
            # GET_INTERNALS leads with an ack byte before its text body.
            GET_INTERNALS,
            # Optic state leads with an ack byte before the snapshot.
            GET_OPTIC_STATE,
            SET_OPTIC_GAIN,
            SET_FOCUS,
        }
    )

    def _request(
        self,
        cmd_id: int,
        payload: bytes = b"",
        *,
        timeout: float | None = None,
    ) -> bytes:
        """Send a request and return the response payload.

        Thread-safe: serializes send/receive so concurrent calls from
        background threads (autonomous learner, inner life) don't
        interleave on a shared socket. Pure reads run on a dedicated
        second connection (``_READ_CHANNEL_CMDS``) so they never queue
        behind a burst of writes; everything else uses the control
        channel, preserving write ordering.

        Read-only commands (GET_*) are served from a short-TTL cache
        when fresh, eliminating redundant socket round-trips from
        background threads that poll state frequently.

        If the connection is lost (broken pipe, daemon restart), this
        attempts one automatic reconnection before giving up.

        Args:
            timeout: Optional per-call socket timeout in seconds.
                Uses ``self.request_timeout`` when not provided.

        Raises:
            GenesisConnectionError: If the connection is lost and
                reconnection fails.
            ProtocolError: If the response is malformed.
        """
        effective_timeout = self.request_timeout if timeout is None else timeout
        on_read_channel = cmd_id in self._READ_CHANNEL_CMDS
        channel_lock = self._read_lock if on_read_channel else self._lock
        with channel_lock:
            if on_read_channel:
                self._ensure_read_connected()
            else:
                self._ensure_connected()
            # Check cache for read-only commands
            if cmd_id in self._CACHEABLE_CMDS and not payload:
                with self._cache_lock:
                    cached = self._cache.get(cmd_id)
                if cached is not None:
                    ts, data = cached
                    if time.monotonic() - ts < self._cache_ttl:
                        return data

            # Write commands invalidate all cached state — the next read
            # must go to the daemon to see the updated values.
            if cmd_id in self._CACHE_INVALIDATING_CMDS:
                with self._cache_lock:
                    self._cache.clear()

            for attempt in range(2):
                resp_data = self._request_attempt(
                    cmd_id,
                    payload,
                    effective_timeout,
                    on_read_channel,
                    attempt,
                )
                if resp_data is not None:
                    return resp_data

        raise GenesisConnectionError("request failed after reconnect attempt")

    def _request_attempt(
        self,
        cmd_id: int,
        payload: bytes,
        effective_timeout: float,
        on_read_channel: bool,
        attempt: int,
    ) -> bytes | None:
        """One send/receive attempt on the appropriate channel.

        Returns the response payload, or None when the connection was
        lost on a retryable command and the loop should attempt again.
        Raises on unrecoverable loss or a broken response.
        """
        sock = self._ensure_read_connected() if on_read_channel else self._ensure_connected()
        old_timeout = sock.gettimeout()
        t0 = time.monotonic()
        try:
            sock.settimeout(effective_timeout)
            write_message(sock, cmd_id, payload)
            resp_cmd, resp_data = read_message(sock)
            elapsed = time.monotonic() - t0
            if elapsed > 2.0:
                # Slow IPC is the early sign of daemon-side
                # serialization stalls — a command that holds
                # the tick lock or LTM while doing slow work
                # queues every caller behind it.
                logger.debug(
                    "ipc cmd %d took %.2fs (deadline %.2fs)",
                    cmd_id,
                    elapsed,
                    effective_timeout,
                )
        except (GenesisConnectionError, ConnectionError, OSError) as e:
            # Tear down only this channel — a stalled read must
            # not knock out the control channel and vice versa.
            # The cache clears too: a daemon restart may leave
            # entries describing a state that no longer exists.
            if on_read_channel:
                self._close_read_socket()
            else:
                self._close_socket()
            with self._cache_lock:
                self._cache.clear()
            # Connection broke — try to reconnect once
            if attempt == 0:
                # Teardown unlinks the socket: a daemon that is gone
                # for good leaves nothing to reconnect to. Fail
                # quietly instead of logging a warning that can only
                # end in failure — mirrored to her feed, it reads as
                # a crash on her way out.
                if not os.path.exists(self.socket_path):
                    raise GenesisConnectionError(f"connection lost: {e}") from e
                # Name the command — a bare "timed out" hides
                # which request was stalled on the daemon.
                logger.warning(
                    f"connection lost during command {cmd_id} ({e}), attempting reconnect..."
                )
                try:
                    if on_read_channel:
                        self._connect_read()
                    else:
                        self.connect()
                except Exception as re:
                    raise GenesisConnectionError(f"reconnect failed: {re}") from re
                if cmd_id not in self._RETRYABLE_CMDS:
                    raise GenesisConnectionError(
                        f"connection lost after command {cmd_id}: outcome unknown; "
                        "command was not retried"
                    ) from e
                return None
            raise GenesisConnectionError(f"connection lost: {e}") from e
        finally:
            # Restore the socket's timeout for reuse — unless the
            # failure path above already closed it, in which
            # case the fd is dead and settimeout would raise EBADF.
            live = self._read_sock if on_read_channel else self._sock
            if live is sock:
                try:
                    sock.settimeout(old_timeout)
                except OSError as e:
                    note_swallowed(
                        "genesis_client.client._request",
                        e,
                    )

        return self._validate_response(
            cmd_id,
            resp_cmd,
            resp_data,
            payload,
            on_read_channel,
        )

    def _validate_response(
        self,
        cmd_id: int,
        resp_cmd: int,
        resp_data: bytes,
        payload: bytes,
        on_read_channel: bool,
    ) -> bytes:
        """Check a response for framing/protocol errors and cache reads."""
        if resp_cmd != cmd_id:
            # Framing is broken on this channel — drop it, but
            # don't take the other channel's lock from here.
            if on_read_channel:
                self._close_read_socket()
            else:
                self._close_socket()
            raise ProtocolError(f"response command {resp_cmd} != request command {cmd_id}")
        # Check for error responses: the daemon returns a
        # response whose first byte has the high bit set
        # (0x80+) when a command fails (e.g. read_consistent
        # failed, payload too short, internal error).
        #
        # For ack commands (STORE_EVENT, ADVANCE_NEURO, etc.)
        # the daemon pads the error response to the expected
        # size, so we must check the first byte regardless of
        # response length. For raw-data commands (GET_*), the
        # first byte is struct data that can legitimately be
        # >= 0x80 (e.g. an f32 mantissa byte), so we only
        # check single-byte responses for those.
        if resp_data and is_error_byte(resp_data[0]):
            if cmd_id in self._ACK_CMDS or len(resp_data) == 1:
                err_code = resp_data[0]
                raise ProtocolError(f"daemon error: {error_name(err_code)} (0x{err_code:02x})")
        # Cache read-only responses
        if cmd_id in self._CACHEABLE_CMDS and not payload:
            with self._cache_lock:
                self._cache[cmd_id] = (time.monotonic(), resp_data)
        return resp_data

    # ─── Commands ──────────────────────────────────────────────

    def ping(self, *, timeout: float | None = None) -> PingResponse:
        """Ping the daemon. Returns ack + uptime."""
        resp = self._request(PING, timeout=timeout)
        return PingResponse.unpack(resp)

    def get_state(self, *, timeout: float | None = None) -> CoreState:
        """Get the full core state (3480 bytes, schema v5), parsed into useful fields."""
        resp = self._request(GET_STATE, timeout=timeout)
        return CoreState.unpack(resp)

    def get_ion_summary(self, *, timeout: float | None = None) -> IonSummary:
        """Get the electrochemical ion summary (108 bytes).

        Exposes calcium, chloride, potassium, and sodium concentrations,
        reversal potentials, conductances, membrane voltage, transporter
        fluxes, ATP availability, and normalized excitability/plasticity
        signals from the substrate's biophysical layer.

        The ion layer is authoritative state owned by the daemon, so
        this is a pure observation — there is no request payload and no
        way for the caller to step it. Absent sensors or an older daemon
        surface as an exception here, which is why interoception treats
        a missing summary as "no reading", never as a resting one.
        """
        resp = self._request(GET_ION_SUMMARY, timeout=timeout)
        return IonSummary.unpack(resp)

    def get_field_probe(self, *, timeout: float | None = None) -> FieldProbeSignature | None:
        """Get the body's real substrate signature (v6: 408 B; older: 328/308/195/114/102).

        The daemon's field-probe thread samples real substrate
        dynamics — instruction throughput, scheduler jitter, P-state
        transitions, interrupt flux, measured power draw where the
        machine's sensors report it, and the radios' own telemetry
        (WLAN byte flux + RSSI, wireless-attributed interrupts, HCI
        counters) — and decomposes them at the wave bands' center
        frequencies. It also writes a carrier rhythm into real work
        keyed to the body's phase, and evaluates Maxwell's equations
        on the measured current swing (quasi-static B, induction E,
        ω⁴-suppressed radiated power — estimates, with the loop
        geometry an assumption).

        Returns ``None`` when the probe isn't running or the daemon
        predates the command: an absent channel is no reading, never
        a zero reading — same discipline as ``get_sensor_presence``.
        """
        try:
            resp = self._request(GET_FIELD_PROBE, timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return None
        if not resp or resp[0] >= 0x80:
            return None
        return FieldProbeSignature.unpack(resp)

    def get_field_state(self, *, timeout: float | None = None) -> FieldState | None:
        """The latent field's integration readout — the organism's
        center of mass and how far its major systems have spread
        from it.

        The body's own field (src/field/) pulls its fibers — body,
        prediction, agency — toward their Fréchet barycenter on a
        shared latent manifold; this returns that center, the
        measured disagreement about it, and the field's thermal
        state (internal tension plus the physical bath — the
        silicon's measured timing noise).

        Returns ``None`` on transport failure or when the daemon
        predates the ``field.*`` internals keys — an absent channel,
        never a zero reading.
        """
        internals = self.get_internals(timeout=timeout)
        if internals is None:
            return None
        return FieldState.from_internals(internals)

    def get_internals(self, *, timeout: float | None = None) -> dict[str, str] | None:
        """The daemon's process-local internals — the gears no state
        file shows.

        Every adaptive gate's full snapshot (bounds, period, next-fire
        countdown, smoothed demand), the mind lease, tick counter,
        last-applied body policy, and the autonomous
        fallback's own gates — as a flat ``key=value`` map. The wire
        format is additive: unknown keys from a newer daemon ride
        through harmlessly, so callers should pick the keys they know
        rather than depend on completeness.

        Observation only — nothing here can move the gears. Returns
        ``None`` on transport failure or when the daemon predates the
        command: an absent channel is no reading, never a zero one.
        """
        try:
            resp = self._request(GET_INTERNALS, timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return None
        if not resp or resp[0] != 1:
            return None
        out: dict[str, str] = {}
        for line in resp[1:].decode("utf-8", "replace").splitlines():
            key, sep, value = line.partition("=")
            if sep and key:
                out[key] = value
        return out

    def set_field_phase(self, phase: float, *, timeout: float | None = None) -> bool:
        """Push the mind's modeled carrier-band phase into the probe.

        The physics-follows-model leg of the mutual phase lock: the
        probe extrapolates the pushed phase at its carrier rate and
        pulls the generator's phase toward it each tick. A target not
        refreshed within 5 s is dropped — a dead mind can't steer
        the body's rhythm.

        Returns ``False`` when the probe isn't running or the daemon
        predates the command — an absent channel, never silent.
        """
        try:
            resp = self._request(SET_FIELD_PHASE, _F32.pack(phase), timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return False
        return bool(resp) and resp[0] == 1

    def set_field_carrier(
        self, depth: float, freq: float = 0.0, *, timeout: float | None = None
    ) -> bool:
        """Set the carrier's real-work modulation — the write side.

        ``depth`` 0..1 scales the work-slice modulation (0 =
        unmodulated work, carrier silent). ``freq`` 0 hands the rhythm
        back to the body's phase; positive overrides it (clamped
        0.1..100 Hz) so evaluation can drive arbitrary bands —
        including infraslow rates where the thermal field can
        hold an oscillation.

        Returns ``False`` when the probe isn't running or the daemon
        predates the command — an absent channel, never silent.
        """
        payload = _F32.pack(depth) + _F32.pack(freq)
        try:
            resp = self._request(SET_FIELD_CARRIER, payload, timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return False
        return bool(resp) and resp[0] == 1

    def get_optic_state(self, *, timeout: float | None = None) -> OpticState | None:
        """The optic nerve's subcortical channel state — the body's own
        read of the eye.

        Photopic and melanopic irradiance, motion energy, the
        collicular orienting transient, the SCN light drive the body
        feeds its circadian oscillator, and the thalamic relay gate
        the mind must honour — near zero in NREM/REM, open while
        awake. A look through a closed gate is a blind look.

        ``eye_present`` distinguishes "the eye exists but sees nothing"
        from "there is no eye"; ``frame_seq``/``frame_age_ms`` let the
        caller tell a stalled feed from a live one.

        Returns ``None`` on transport failure or when the daemon
        predates the command — an absent channel is no reading, never
        a zero reading.
        """
        try:
            resp = self._request(GET_OPTIC_STATE, timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return None
        try:
            return OpticState.unpack(resp)
        except (IndexError, struct.error, ValueError):
            return None

    def set_optic_gain(self, gain: float, *, timeout: float | None = None) -> bool:
        """Corticothalamic feedback — the mind's engagement gain onto
        the thalamic relay (the layer-6 V1→dLGN projection).

        Modulatory only: the body clamps ``gain`` into 0..1 and scales
        the relay within ±40% of what the sleep stage permits — it can
        never open a gate NREM has closed. 0.5 is neutral, above 1.0
        is meaningless (clamped).

        Returns ``False`` when the daemon predates the command — an
        absent channel, never silent.
        """
        try:
            resp = self._request(SET_OPTIC_GAIN, _F32.pack(gain), timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return False
        return bool(resp) and resp[0] == 1

    def analyze_code(
        self, path: str, source: str, *, timeout: float | None = None
    ) -> dict[str, Any] | None:
        """The body's structural sense of a Rust source file.

        The daemon parses the source into a real AST (syn) — semantic
        units with calls/references/complexity, the dependency graph,
        impl relations — a richer reading than the mind's regex parser.
        The report is decoded into::

            {"file": str, "stats": {lines, functions, structs, enums,
              traits, impls, max_complexity, avg_complexity},
             "units": [{kind, name, qualified, span, public, async_,
              unsafe_, body, params, return_type, calls, refs,
              branches, loops, stmts, docs}],
             "impls": [(self_type, trait_or_None)],
             "edges": [(caller_qualified, callee_qualified)]}

        Returns ``None`` on transport failure, a parse error the daemon
        rejected (``INVALID_VALUE``), or when the daemon predates the
        command — an absent channel, never a fabricated reading. Callers
        should fall back to their own parser on ``None``.
        """
        path_b = path.encode("utf-8", "replace")[:0xFFFF]
        src_b = source.encode("utf-8", "replace")
        payload = _U16.pack(len(path_b)) + path_b + _U32.pack(len(src_b)) + src_b
        try:
            resp = self._request(ANALYZE_CODE, payload, timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return None
        if not resp or resp[0] != 1:
            return None
        return _decode_code_analysis(resp[1:].decode("utf-8", "replace"))

    def set_focus(self, meters: float, *, timeout: float | None = None) -> bool:
        """Accommodation — the mind's focus request in metres.

        The body converts the distance to the ciliary drive
        (``1/m`` dioptres, clamped to 0..6 — infinity out to a ~17 cm
        near point) and slews toward it on the advance clock, pulling
        in faster than it releases. ``meters <= 0`` or non-finite
        relaxes the eye to infinity. Monocular: the ciliary servo
        only — no vergence exists to request.

        Returns ``False`` when the daemon predates the command — an
        absent channel, never silent.
        """
        try:
            resp = self._request(SET_FOCUS, _F32.pack(meters), timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return False
        return bool(resp) and resp[0] == 1

    def set_lid(self, openness: float, *, timeout: float | None = None) -> bool:
        """Voluntary eyelid — the orbicularis oculi the hardware lacks.

        ``openness`` in 0..1 composes with the sleep phase's own lid
        position by min: the mind can shut an open eye (to rest it,
        to dim the world) but can never open one the stage has
        closed. Held, not momentary — lids stay where the mind left
        them until it moves them again. 1.0 releases the eye to the
        phase's own position.

        Returns ``False`` when the daemon predates the command — an
        absent channel, never silent.
        """
        try:
            resp = self._request(SET_LID, _F32.pack(openness), timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return False
        return bool(resp) and resp[0] == 1

    def set_cognitive_pid(self, pid: int, *, timeout: float | None = None) -> bool:
        """Tell the body which process is its cognitive mind.

        Pushed when a new mind re-inhabits a surviving body — adoption
        replaces ``GENESIS_COGNITIVE_PID``, which still names the dead
        launcher. Interoception then measures the living mind's
        CPU/memory instead of a corpse's, and ``GET_PROCESS_HEALTH``
        reports the mind alive again.

        Returns ``False`` when the daemon predates the command — an
        absent channel, never silent.
        """
        try:
            resp = self._request(SET_COGNITIVE_PID, _U32.pack(pid), timeout=timeout)
        except (ConnectionError, OSError, ProtocolError):
            return False
        return bool(resp) and resp[0] == 1

    # ─── Tensor slab ──────────────────────────────────────────
    # Named multidimensional tensors shared with the daemon — world
    # frames, salience fields, embeddings, motor fields. Wire layouts
    # live in `genesis_client.tensor` and mirror
    # `src/daemon/tensor_slab.rs`. The slab is process-local daemon
    # state: nothing here survives a daemon restart.

    def put_tensor(
        self,
        name: str,
        tensor: Tensor,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Store a named tensor on the daemon's slab (last write wins).

        Args:
            name: 1–128 byte UTF-8 name (e.g. ``"world.frame"``).
            tensor: The :class:`Tensor` to store.

        Returns:
            True when the daemon acknowledged the store.
        """
        resp = self._request(PUT_TENSOR, pack_name(name) + tensor.pack(), timeout=timeout)
        return bool(resp) and resp[0] == 1

    def get_tensor(self, name: str, *, timeout: float | None = None) -> Tensor | None:
        """Fetch a named tensor, or ``None`` when absent."""
        resp = self._request(GET_TENSOR, pack_name(name), timeout=timeout)
        if not resp or resp[0] == 0:
            return None
        return Tensor.unpack(resp[1:])

    def list_tensors(self, *, timeout: float | None = None) -> list[TensorMeta]:
        """Names, dtypes, and shapes of every live slab tensor."""
        resp = self._request(LIST_TENSORS, timeout=timeout)
        return unpack_list(resp)

    def drop_tensor(self, name: str, *, timeout: float | None = None) -> bool:
        """Remove a named tensor. Idempotent — absent names ack too."""
        resp = self._request(DROP_TENSOR, pack_name(name), timeout=timeout)
        return bool(resp) and resp[0] == 1

    def get_motor_target(self, *, timeout: float | None = None) -> MotorTarget:
        """What the published motor fields currently point at.

        The daemon decodes ``motor.spatial`` (argmax cell projected
        onto its canonical 64×64 coordinate grid) and ``motor.actions``
        (argmax disposition, mapped through ``motor.action_ids``) into
        a single fixed-width answer — the decision read back from the
        distributed field, not the symbol that wrote it.
        """
        resp = self._request(GET_MOTOR_TARGET, timeout=timeout)
        return unpack_motor_target(resp)

    def get_neuro_summary(self, *, timeout: float | None = None) -> NeuroSummary:
        """Get the compact neurochemical summary (32 bytes).

        This is the most efficient way to check how Genesis feels.
        """
        resp = self._request(GET_NEURO_SUMMARY, timeout=timeout)
        return NeuroSummary.unpack(resp)

    def get_plasticity_profile(self, *, timeout: float | None = None) -> PlasticityProfile:
        """Get the plasticity profile — metaplastic state summary (40 bytes).

        Exposes the substrate's "learning-to-learn" state: coupling-
        matrix drift, receptor sensitivities, BDNF/cortisol levels,
        and the plasticity gate. The cognitive mind uses this to
        adapt its learning strategy (see ``learning_posture``).
        """
        resp = self._request(GET_PLASTICITY_PROFILE, timeout=timeout)
        return PlasticityProfile.unpack(resp)

    def get_body_state(self, *, timeout: float | None = None) -> BodyState:
        """Get the interoceptive body state — how Genesis's machine feels.

        Returns CPU temperature, frequency, memory pressure, I/O activity,
        load average, battery level, and a human-readable description.
        This is Genesis's sense of its own body.
        """
        resp = self._request(GET_BODY_STATE, timeout=timeout)
        return BodyState.unpack(resp)

    def get_process_health(self, *, timeout: float | None = None) -> ProcessHealth:
        """Get which of Genesis's own processes are alive, and death count.

        A subsystem that dies used to disappear from
        ``get_subsystem_telemetry`` with no record, so a crashed retina
        looked exactly like one that was never started — the reader saw
        a shorter list and could not tell why.

        ``deaths`` counts alive→not-alive transitions since startup, so
        it separates a loss from an absence. A subsystem that was never
        running does not count.
        """
        resp = self._request(GET_PROCESS_HEALTH, timeout=timeout)
        return ProcessHealth.unpack(resp)

    def get_zone_transitions(self, *, timeout: float | None = None) -> ZoneTransitions:
        """Get how often Genesis has changed zone and phase, and dwell times.

        Neither was measured before: the mmap'd zone record held a
        single duration sample that every transition overwrote, with no
        count at all. So a mind oscillating between two phases was
        indistinguishable from one that had settled.

        Zone and phase are returned separately because they are driven
        by different mechanisms — the zone follows intentions, the
        phase follows neurochemistry — and oscillating in one while
        stable in the other is a distinct fault.

        Counts are process-local and reset when the daemon restarts, so
        they read as a rate rather than a lifetime total.
        """
        resp = self._request(GET_ZONE_TRANSITIONS, timeout=timeout)
        return ZoneTransitions.unpack(resp)

    def get_sensor_presence(self, *, timeout: float | None = None) -> SensorPresence:
        """Get which optional hardware sensors this machine actually has.

        ``GET_BODY_STATE`` returns ~30 values with no per-channel
        validity, so a channel backed by no sensor reads the same as one
        reporting a healthy zero. This distinguishes them: a set bit
        means the matching ``BodyState`` field carries a real reading.

        A zero mask is returned — never an error — when the daemon is
        older than this command or no interoception read has completed
        yet. That is the safe direction: nothing is claimed to be
        measured, so callers must not treat an unverified channel as
        evidence about the machine.
        """
        try:
            resp = self._request(GET_SENSOR_PRESENCE, timeout=timeout)
        except (ConnectionError, OSError):
            return SensorPresence(mask=0)
        return SensorPresence.unpack(resp)

    def get_subsystem_telemetry(self, *, timeout: float | None = None) -> SubsystemReport:
        """Get per-subsystem silicon telemetry — which part of it is firing.

        Returns a :class:`SubsystemReport` with two granularities:

        - ``subsystems``: one :class:`SubsystemTelemetry` per process in its
          process tree (daemon, cognitive, retina) — hardware-measured
          CPU/I/O share plus microarchitectural prediction-error
          ratios (cache/branch misses).
        - ``modules``: one :class:`ModuleTelemetry` per non-Stopped
          manifest module — the self-reported brain parts (language,
          memory, emotion, …) with each part's share of the cognitive
          process's measured work.

        The cognitive mind correlates this with its task zone to feel
        *where* its activity lives.
        """
        resp = self._request(GET_SUBSYSTEM_TELEMETRY, timeout=timeout)
        return SubsystemReport.unpack(resp)

    def set_wake_alarm(self, epoch_secs: int, *, timeout: float | None = None) -> int | None:
        """Arm or disarm the RTC wake alarm — when the machine exists next.

        ``epoch_secs`` is a Unix timestamp; 0 disarms. Returns the
        alarm epoch the hardware reports afterward (0 = none armed),
        or ``None`` if the daemon could not arm it (no passwordless
        sudo grant on this host).

        This only schedules the wake interrupt; it does NOT suspend
        the machine. Suspending stays a separate deliberate act.
        """
        resp = self._request(SET_WAKE_ALARM, _U64.pack(epoch_secs), timeout=timeout)
        if len(resp) < 9 or resp[0] != 1:
            return None
        return _U64.unpack(resp[1:9])[0]

    def get_body_control(self, *, timeout: float | None = None) -> BodyControlState:
        """Get the body control state — what Genesis is doing to its body.

        Returns CPU frequency policy, scheduling priorities, I/O priority,
        thermal cap status, and a human-readable description. This is
        Genesis's awareness of its own agency over its hardware.
        """
        resp = self._request(GET_BODY_CONTROL, timeout=timeout)
        return BodyControlState.unpack(resp)

    def get_inference_summary(self, *, timeout: float | None = None) -> InferenceSummary:
        """Get the active inference summary — the generative self-model's
        projection (60 bytes).

        Exposes surprise, free energy, allostatic load, precision,
        dyadic attunement/synchrony, user affect, and prediction errors.
        This is how the cognitive mind knows how well Genesis is
        predicting its own neurochemical trajectory and how attuned
        it is to the user.
        """
        resp = self._request(GET_INFERENCE_SUMMARY, timeout=timeout)
        return InferenceSummary.unpack(resp)

    def update_user_affect(
        self,
        valence: float,
        arousal: float,
        engagement: float,
        confidence: float = 0.7,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Update the user affect observation — the cognitive mind's
        inference of the user's affective state from conversation
        features.

        This feeds the dyadic affective model, which measures the
        dyadic coupling — evidence her own appraisal acts on, not
        prescribed chemistry.

        Args:
            valence: User's affective valence [-1, 1].
            arousal: User's affective arousal [0, 1].
            engagement: User's engagement level [0, 1].
            confidence: Confidence in the observation [0, 1]. Lower
                confidence = the observation is noisy and weighted less.
        """
        if not (-1.0 <= valence <= 1.0):
            raise ValueError(f"valence must be [-1, 1], got {valence}")
        if not (0.0 <= arousal <= 1.0):
            raise ValueError(f"arousal must be [0, 1], got {arousal}")
        if not (0.0 <= engagement <= 1.0):
            raise ValueError(f"engagement must be [0, 1], got {engagement}")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be [0, 1], got {confidence}")
        payload = _F32.pack(valence) + _F32.pack(arousal)
        payload += _F32.pack(engagement) + _F32.pack(confidence)
        resp = self._request(UPDATE_USER_AFFECT, payload, timeout=timeout)
        return bool(resp and resp[0] == 1)

    def get_phase(self, *, timeout: float | None = None) -> PhaseInfo:
        """Get the current mental phase + arousal/valence (9 bytes)."""
        resp = self._request(GET_PHASE, timeout=timeout)
        return PhaseInfo.unpack(resp)

    def get_memory_stats(self, *, timeout: float | None = None) -> MemoryStats:
        """Get memory store statistics (STM count, LTM count, capacity)."""
        resp = self._request(GET_MEMORY_STATS, timeout=timeout)
        return MemoryStats.unpack(resp)

    def store_event(
        self,
        timestamp: int,
        event_type: int,
        source_module: int,
        salience: float,
        emotional_tag: list[float],
        text: str,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Store an event in short-term memory.

        Args:
            timestamp: Milliseconds since Unix epoch.
            event_type: Event type ID (0=UserInput, 1=Output, etc.)
            source_module: Module ID that produced this event.
            salience: Importance score [0.0, 1.0].
            emotional_tag: EMOTIONAL_TAG_SIZE effective neurochemical levels.
            text: Event description (max 188 bytes).

        Returns:
            True if the event was stored successfully.
        """
        if len(emotional_tag) != EMOTIONAL_TAG_SIZE:
            raise ValueError(
                f"emotional_tag must have {EMOTIONAL_TAG_SIZE} elements, got {len(emotional_tag)}"
            )
        if not (0 <= event_type <= 255):
            raise ValueError(f"event_type must be 0-255, got {event_type}")
        if not (0 <= source_module <= 255):
            raise ValueError(f"source_module must be 0-255, got {source_module}")

        # Truncate to the 188-byte STM text slot on a UTF-8 character
        # boundary so the daemon's decode doesn't end in U+FFFD.
        text_bytes = text.encode("utf-8")
        if len(text_bytes) > 188:
            text_bytes = text_bytes[:188].decode("utf-8", errors="ignore").encode("utf-8")
        text_len = len(text_bytes)

        payload = bytearray()
        payload.extend(_U64.pack(timestamp))
        payload.append(event_type & 0xFF)
        payload.append(source_module & 0xFF)
        payload.extend(_F32.pack(salience))
        for v in emotional_tag:
            payload.extend(_F32.pack(v))
        payload.extend(_U16.pack(text_len))
        payload.extend(text_bytes)

        resp = self._request(STORE_EVENT, bytes(payload), timeout=timeout)
        return bool(resp) and resp[0] == 1

    def store_episode(
        self,
        timestamp: int,
        event_type: int,
        source_module: int,
        salience: float,
        emotional_tag: list[float],
        text: str,
        *,
        timeout: float | None = None,
    ) -> int | None:
        """Store an episode directly in long-term memory, bypassing STM.

        Used by the cognitive mind for deliberate memory stores
        (conversation turns, learning events) where the real LTM
        episode ID is needed immediately for Python-side tracking.
        Raw pre-cognitive events should use :meth:`store_event`
        (the STM path); this method is for memories the cognitive
        mind has already decided are worth keeping.

        Args:
            timestamp: Milliseconds since Unix epoch.
            event_type: Event type ID (0=UserInput, 1=Output, etc.)
            source_module: Module ID that produced this event.
            salience: Importance score [0.0, 1.0].
            emotional_tag: EMOTIONAL_TAG_SIZE effective neurochemical levels.
            text: Event description (max 65535 bytes — the wire protocol's
            u16 text_len limit). The LTM store itself can handle up to
            256 KB per episode; the bottleneck is the u16 length field.

        Returns:
            The real LTM episode ID on success, or None if the store
            failed.
        """
        if len(emotional_tag) != EMOTIONAL_TAG_SIZE:
            raise ValueError(
                f"emotional_tag must have {EMOTIONAL_TAG_SIZE} elements, got {len(emotional_tag)}"
            )
        if not (0 <= event_type <= 255):
            raise ValueError(f"event_type must be 0-255, got {event_type}")
        if not (0 <= source_module <= 255):
            raise ValueError(f"source_module must be 0-255, got {source_module}")

        # Truncate to the u16 wire-protocol limit (65535 bytes) on a
        # UTF-8 character boundary. Unlike store_event (STM), whose
        # fixed ring-buffer slot caps text at 188 bytes, the LTM path
        # has no structural size limit below the wire protocol's u16.
        text_bytes = text.encode("utf-8")
        if len(text_bytes) > 0xFFFF:
            text_bytes = text_bytes[:0xFFFF].decode("utf-8", errors="ignore").encode("utf-8")
        text_len = len(text_bytes)

        payload = bytearray()
        payload.extend(_U64.pack(timestamp))
        payload.append(event_type & 0xFF)
        payload.append(source_module & 0xFF)
        payload.extend(_F32.pack(salience))
        for v in emotional_tag:
            payload.extend(_F32.pack(v))
        payload.extend(_U16.pack(text_len))
        payload.extend(text_bytes)

        resp = self._request(STORE_EPISODE, bytes(payload), timeout=timeout)
        if len(resp) >= 9 and resp[0] == 1:
            return _U64.unpack(resp[1:9])[0]
        return None

    def retrieve_episode(self, episode_id: int, *, timeout: float | None = None) -> Episode:
        """Retrieve a full episode from long-term memory by ID.

        Args:
            episode_id: The episode's unique ID.

        Returns:
            The Episode with full text and emotional tags.

        Raises:
            EpisodeNotFound: If the episode doesn't exist.
        """
        payload = _U64.pack(episode_id)
        resp = self._request(RETRIEVE_EPISODE, payload, timeout=timeout)
        ep = Episode.unpack(resp)
        if ep is None:
            raise EpisodeNotFound(f"episode {episode_id} not found")
        return ep

    def find_similar(
        self, query: str, limit: int = 10, *, timeout: float | None = None
    ) -> list[SimilarEpisode]:
        """Find memories similar to the query text.

        Uses SimHash associative matching — similar content produces
        similar hashes, and Hamming distance measures dissimilarity.

        Args:
            query: The search query.
            limit: Maximum number of results.

        Returns:
            List of SimilarEpisode, sorted by Hamming distance
            (most similar first).
        """
        query_bytes = query.encode("utf-8")
        if not (0 <= limit <= 255):
            raise ValueError(f"limit must be 0-255, got {limit}")
        payload = bytearray()
        payload.extend(_U32.pack(len(query_bytes)))
        payload.extend(query_bytes)
        payload.append(limit & 0xFF)

        resp = self._request(FIND_SIMILAR, bytes(payload), timeout=timeout)
        return unpack_similar_results(resp)

    def find_similar_archived(
        self, query: str, limit: int = 10, *, timeout: float | None = None
    ) -> list[SimilarEpisode]:
        """Deep recall — find similar memories in the archived tier.

        Same SimHash associative matching as `find_similar`, scanned
        over episodes the sleep compressor retired. Archived episodes
        are out of the active index but intact on disk, so a specific
        enough cue can still reach them. A hit is meant to be
        reconsolidated — pass its episode_id to `unarchive_episode`
        when the memory is actually used.

        Args:
            query: The search query.
            limit: Maximum number of results.

        Returns:
            List of SimilarEpisode, sorted by Hamming distance
            (most similar first).
        """
        query_bytes = query.encode("utf-8")
        if not (0 <= limit <= 255):
            raise ValueError(f"limit must be 0-255, got {limit}")
        payload = bytearray()
        payload.extend(_U32.pack(len(query_bytes)))
        payload.extend(query_bytes)
        payload.append(limit & 0xFF)

        resp = self._request(FIND_SIMILAR_ARCHIVED, bytes(payload), timeout=timeout)
        return unpack_similar_results(resp)

    def get_recent_episodes(
        self,
        limit: int = 20,
        source_module: int | None = None,
        *,
        timeout: float | None = None,
    ) -> list[RecentEpisode]:
        """Get recent episodes from long-term memory.

        Args:
            limit: Maximum number of episodes to return (1-255).
            source_module: If given, filter to episodes from this
                module (e.g. 9 for dream insights, 0 for subcognitive
                associations). None returns all recent episodes.

        Returns:
            List of RecentEpisode, most recent first.
        """
        limit_byte = max(1, min(limit, 255)) & 0xFF
        if source_module is None:
            filter_byte = 255
        else:
            if not (0 <= source_module <= 254):
                raise ValueError(
                    f"source_module must be 0-254 (255 is reserved for 'all'), got {source_module}"
                )
            filter_byte = source_module & 0xFF
        resp = self._request(
            GET_RECENT_EPISODES,
            bytes([limit_byte, filter_byte]),
            timeout=timeout,
        )
        return unpack_recent_episodes(resp)

    def archive_episode(self, episode_id: int, *, timeout: float | None = None) -> bool:
        """Archive an episode in long-term memory.

        Used by sleep compression to compact low-salience episodes
        after their semantic content has been extracted into the
        holographic graph. The episode is removed from the active
        similarity index but remains retrievable by ID — unlike
        deletion, archiving preserves the episode so that dream
        insights and association chains that reference it still
        resolve. Nothing is ever lost.

        Args:
            episode_id: The episode's unique ID.

        Returns:
            True if the episode was archived, False if not found.
        """
        payload = _U64.pack(episode_id)
        resp = self._request(ARCHIVE_EPISODE, payload, timeout=timeout)
        return bool(resp and resp[0] == 1)

    def unarchive_episode(self, episode_id: int, *, timeout: float | None = None) -> bool:
        """Un-archive an episode — reconsolidation.

        The mirror of `archive_episode`: when deep recall surfaces a
        retired memory and the mind actually uses it, this returns the
        episode to the active similarity index, re-bundles its SDR,
        and floors its salience at the reconsolidation level so the
        next sleep pass retains what recall used. Retrieval is what
        earns a memory its warmth.

        Args:
            episode_id: The episode's unique ID.

        Returns:
            True if the episode was un-archived, False if not found.
        """
        payload = _U64.pack(episode_id)
        resp = self._request(UNARCHIVE_EPISODE, payload, timeout=timeout)
        return bool(resp and resp[0] == 1)

    def delete_episode(self, episode_id: int, *, timeout: float | None = None) -> bool:
        """Delete an episode from LTM — real forgetting.

        The terminal tier: archived is still there, deleted is not.
        The mind's own retention model decides which episodes earn
        release; the daemon only executes it. Deletion removes the
        episode from every index and retrieval path but leaves its
        payload slot orphaned in the data file — physical space is
        reclaimed by a file rebuild, not this call.

        Args:
            episode_id: The episode's unique ID.

        Returns:
            True if the episode was deleted, False if not found or
            already deleted.
        """
        payload = _U64.pack(episode_id)
        resp = self._request(DELETE_EPISODE, payload, timeout=timeout)
        return bool(resp and resp[0] == 1)

    def search_archived(
        self,
        limit: int = 1000,
        offset: int = 0,
        *,
        timeout: float | None = None,
    ) -> list[ArchivedEpisode]:
        """Page through the archived tier — metadata only.

        The cold store's table of contents: every archived episode's
        id, timestamp, salience, flags, source module, and compact
        emotional tag, in ascending episode-id order (oldest first).
        Sleep compression enumerates this to decide which traces have
        decayed past retrieval — the release decision needs the
        bookkeeping, not the body of each memory.

        Args:
            limit: Maximum number of entries to return (capped at
                1000 by the daemon per page; this method paginates
                internally).
            offset: Number of entries to skip from the start.

        Returns:
            List of ArchivedEpisode, ordered by episode ID ascending.
        """
        if not (0 <= limit <= 0xFFFFFFFF):
            raise ValueError(f"limit must be 0-4294967295, got {limit}")
        if not (0 <= offset <= 0xFFFFFFFF):
            raise ValueError(f"offset must be 0-4294967295, got {offset}")
        entries: list[ArchivedEpisode] = []
        while len(entries) < limit:
            page_offset = offset + len(entries)
            if page_offset > 0xFFFFFFFF:
                break
            payload = _U32.pack(min(1000, limit - len(entries))) + _U32.pack(page_offset)
            resp = self._request(SEARCH_ARCHIVED, payload, timeout=timeout)
            page = unpack_archived_episodes(resp)
            if not page:
                break
            if entries and page[0].episode_id <= entries[-1].episode_id:
                raise ProtocolError("archived search returned overlapping pages")
            entries.extend(page)
            if len(page) < 1000:
                break  # short page — end of the tier
        return entries

    def search_episodes(
        self,
        limit: int = 1000,
        offset: int = 0,
        *,
        timeout: float | None = None,
    ) -> list[RecentEpisode]:
        """Search episodes in long-term memory by offset.

        Pages through all active episodes in ascending episode ID
        order. Used by sleep compression to iterate over the full LTM
        for compaction.

        Args:
            limit: Maximum number of episodes to return (capped at
                1000 by the daemon).
            offset: Number of episodes to skip from the start.

        Returns:
            List of RecentEpisode, ordered by episode ID ascending.
        """
        if not (0 <= limit <= 0xFFFFFFFF):
            raise ValueError(f"limit must be 0-4294967295, got {limit}")
        if not (0 <= offset <= 0xFFFFFFFF):
            raise ValueError(f"offset must be 0-4294967295, got {offset}")
        limit = min(limit, 1000)
        episodes: list[RecentEpisode] = []
        while len(episodes) < limit:
            page_offset = offset + len(episodes)
            if page_offset > 0xFFFFFFFF:
                break
            payload = _U32.pack(limit - len(episodes)) + _U32.pack(page_offset)
            resp = self._request(SEARCH_EPISODES, payload, timeout=timeout)
            page = unpack_recent_episodes(resp)
            if not page:
                break
            if episodes and page[0].episode_id <= episodes[-1].episode_id:
                raise ProtocolError("episode search returned overlapping pages")
            episodes.extend(page)
        return episodes

    def set_zone(self, zone: int, *, timeout: float | None = None) -> bool:
        """Set the cognitive task zone (what Genesis is doing).

        Args:
            zone: Zone ID (0=Idle, 1=Conversation, 2=Coding, etc.)

        Returns:
            True if the zone was set successfully.
        """
        if not (0 <= zone <= ZONE_SLEEPING):
            raise ValueError(f"zone must be 0-{ZONE_SLEEPING}, got {zone}")
        resp = self._request(SET_ZONE, bytes([zone & 0xFF]), timeout=timeout)
        return bool(resp) and resp[0] == 1

    def neuro_impulse(self, chem: int, magnitude: float, *, timeout: float | None = None) -> bool:
        """Apply a neurochemical impulse (external event affecting mood).

        Args:
            chem: Neurochemical ID (0=Dopamine, 1=Serotonin, etc.)
            magnitude: Impulse magnitude (positive increases, negative
                       decreases). Typically in range [-0.5, +0.5].

        Returns:
            True if the impulse was applied successfully.
        """
        if not (0 <= chem <= 17):
            raise ValueError(f"chem must be 0-17 (18 chemicals), got {chem}")
        payload = bytearray()
        payload.append(chem & 0xFF)
        payload.extend(_F32.pack(magnitude))
        resp = self._request(NEURO_IMPULSE, bytes(payload), timeout=timeout)
        return bool(resp) and resp[0] == 1

    def set_policy_authority(self, authority: float, *, timeout: float | None = None) -> bool:
        """Delegate policy choice to the inference engine, in part.

        Choosing a stance — calm, focus, bond, rest, mobilise — is a
        judgement about what to feel and do, so it belongs to cognition
        rather than to the engine's own free-energy scoring. This is how
        she delegates some of that judgement and keeps some, and how she
        withdraws it again.

        At 0.0 the engine still predicts, learns and reports what it
        would have chosen, but does not act on it: she decides and acts
        through `neuro_impulse`, where the effect is hers and traceable
        to a judgement she made. Her homeostatic reflex is unaffected at
        any value — correcting drift toward what she wants is physiology,
        not a choice.

        Args:
            authority: Fraction of policy authority to delegate, [0, 1].

        Returns:
            True if the delegation was recorded.
        """
        payload = bytearray()
        payload.extend(_F32.pack(max(0.0, min(1.0, float(authority)))))
        resp = self._request(SET_POLICY_AUTHORITY, bytes(payload), timeout=timeout)
        return bool(resp) and resp[0] == 1

    def neuro_adjust_baseline(
        self, chem: int, delta: float, *, timeout: float | None = None
    ) -> bool:
        """Adjust a neurochemical's baseline (set-point) directly.

        Unlike neuro_impulse, which only affects the current level and
        decays back to baseline, this permanently shifts the baseline.
        Used by the startup wake cascade to clear residual adenosine
        sleep pressure from a previous session.

        Args:
            chem: Neurochemical ID (0=Dopamine, 1=Serotonin, etc.)
            delta: Change to apply to the baseline (positive raises,
                  negative lowers). Baseline is clamped to [0.05, 0.95].

        Returns:
            True if the baseline was adjusted successfully.
        """
        if not (0 <= chem <= 17):
            raise ValueError(f"chem must be 0-17 (18 chemicals), got {chem}")
        payload = bytearray()
        payload.append(chem & 0xFF)
        payload.extend(_F32.pack(delta))
        resp = self._request(NEURO_ADJUST_BASELINE, bytes(payload), timeout=timeout)
        return bool(resp) and resp[0] == 1

    def sync(self, *, timeout: float | None = None) -> bool:
        """Request the daemon to sync all stores to disk.

        Returns:
            True if the sync was successful.
        """
        resp = self._request(SYNC, timeout=timeout)
        return bool(resp) and resp[0] == 1

    # ─── Reactive commands (mind-driven, not tick-driven) ──────
    #
    # These methods drive the daemon's functions on demand. The
    # daemon no longer runs a fixed tick loop — the cognitive mind
    # decides when to advance neurochemistry, consolidate memories,
    # associate, dream, read sensors, and control the body.

    def advance_neuro(
        self, *, timeout: float | None = None
    ) -> tuple[float, float, float, float, int] | None:
        """Advance the body's neurochemistry and run active inference.

        There is deliberately no ``dt`` parameter. The daemon owns time:
        it is the body, it integrates at TICK_INTERVAL_MS on its own
        clock, and it measures its own elapsed interval. Accepting a
        caller-supplied step is what allowed the mind to dictate the
        body's integration rate — the coupling that had neurochemistry
        advancing in ~1-second lumps while its rate constants, the
        circadian oscillator, and the sleep machinery were all tuned for
        200 ms.

        The 4-byte dt field remains on the wire so the frame layout and
        PROTOCOL_VERSION are unchanged, but the daemon reads and ignores
        it. What is sent is the nominal step, purely as a placeholder.

        The returned signals are also available from shared memory
        (``CoreState.inference``), so a caller that only wants to read
        them does not need this call at all.

        Returns:
            (surprise, free_energy, precision, allostatic_load,
             tick_count) or None on error.
        """
        resp = self._request(ADVANCE_NEURO, _F32.pack(DT_NOMINAL_STEP), timeout=timeout)
        return self._decode_advance(resp)

    @staticmethod
    def _decode_advance(
        resp: bytes | None,
    ) -> tuple[float, float, float, float, int] | None:
        """Decode the 21-byte advance response shared by both commands."""
        if not resp or resp[0] != 1 or len(resp) < 21:
            return None
        surprise = _F32.unpack(resp[1:5])[0]
        free_energy = _F32.unpack(resp[5:9])[0]
        precision = _F32.unpack(resp[9:13])[0]
        allostatic = _F32.unpack(resp[13:17])[0]
        tick_count = _U32.unpack(resp[17:21])[0]
        return (surprise, free_energy, precision, allostatic, tick_count)

    def advance_physics(
        self, dt: float, *, timeout: float | None = None
    ) -> tuple[float, float, float, float, int] | None:
        """Advance the body by an explicit step. **Evaluation only.**

        The production path is :meth:`advance_neuro`, which takes no step
        because the daemon owns time. This command exists for offline
        experiments that need `dt` as an independent variable -- the
        causal assay advances a fixed number of ticks and measures the
        per-tick response to an impulse, so it cannot use the daemon's
        wall clock.

        Only ever call this against an isolated daemon in a temporary
        data directory (``eval/harness.py``). Driving the live daemon with
        it would put two clocks on the body again, which is the coupling
        this was split apart to remove.

        Args:
            dt: Simulated seconds to advance. Clamped to the same
                ``DT_MIN..DT_MAX`` range the body applies to its own
                elapsed time, so an experiment cannot request a step the
                production integrator would refuse.

        Returns:
            (surprise, free_energy, precision, allostatic_load,
             tick_count) or None on error.
        """
        dt = min(DT_MAX, max(DT_MIN, dt))
        resp = self._request(ADVANCE_PHYSICS, _F32.pack(dt), timeout=timeout)
        return self._decode_advance(resp)

    def consolidate(self, *, timeout: float | None = None) -> int:
        """Consolidate STM → LTM. Returns episodes promoted."""
        resp = self._request(CONSOLIDATE, timeout=timeout)
        if not resp or resp[0] != 1 or len(resp) < 5:
            return 0
        return _U32.unpack(resp[1:5])[0]

    def associate(self, *, timeout: float | None = None) -> int:
        """Find associations for recent episodes. Returns count."""
        resp = self._request(ASSOCIATE, timeout=timeout)
        if not resp or resp[0] != 1 or len(resp) < 5:
            return 0
        return _U32.unpack(resp[1:5])[0]

    def dream(self, *, timeout: float | None = None) -> int:
        """Run one dream cycle. Returns insights found."""
        resp = self._request(DREAM, timeout=timeout)
        if not resp or resp[0] != 1 or len(resp) < 5:
            return 0
        return _U32.unpack(resp[1:5])[0]

    def read_sensors(self, *, timeout: float | None = None) -> BodyState | None:
        """Read hardware sensors and apply interoception impulses."""
        resp = self._request(READ_SENSORS, timeout=timeout)
        if not resp or resp[0] != 1:
            return None
        # Same format as get_body_state, minus the ack byte
        return BodyState.unpack(resp[1:])

    def apply_body_control(self, *, timeout: float | None = None) -> BodyControlState | None:
        """Request the daemon to apply the recommended body control.

        The cognitive mind calls this to ask the daemon to apply the
        shared body control (CPU frequency, thermal cap) and the
        daemon's own scheduling/I/O priority. This is the cognitive
        mind requesting an action via IPC — NOT the tick controlling
        outward. The tick only computes and publishes the
        recommendation; the cognitive mind decides when to act on it.

        The daemon's reactive handler uses change detection, so
        redundant requests (when the recommended values haven't
        changed) are cheap — no sysfs writes or subprocess spawns.
        """
        resp = self._request(APPLY_BODY_CONTROL, timeout=timeout)
        if not resp or resp[0] != 1:
            return None
        # Same format as get_body_control, minus the ack byte
        return BodyControlState.unpack(resp[1:])

    def save_inference(self, *, timeout: float | None = None) -> bool:
        """Save the active inference model to disk."""
        resp = self._request(SAVE_INFERENCE, timeout=timeout)
        return bool(resp) and resp[0] == 1

    def shutdown(self, *, timeout: float | None = None) -> bool:
        """Request the daemon to shut down gracefully.

        Returns:
            True if the shutdown command was acknowledged.
        """
        resp = self._request(SHUTDOWN, timeout=timeout)
        return bool(resp) and resp[0] == 1

    def update_module_status(
        self,
        module_id: int,
        status: int,
        cpu_share: float | None = None,
        *,
        task_id: int | None = None,
        error_code: int | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Update a module's status in the runtime manifest.

        Used by the cognitive mind to register its modules (Sensory,
        Motor, etc.) with the subcognitive daemon and send heartbeats.
        The daemon updates the module's status and last_heartbeat
        timestamp in the mmap'd manifest.

        Args:
            module_id: Module ID (0=Subcognitive, 6=Sensory, 7=Motor, etc.)
            status: Status ID (0=Stopped, 1=Starting, 2=Running,
                    3=Idle, 4=Stopping, 5=Error)
            cpu_share: Optional self-reported activity fraction
                    [0,1] — the module's share of the cognitive
                    process's measured work. Written into the
                    manifest's ``cpu_share`` and served by
                    GET_SUBSYSTEM_TELEMETRY's module section.
            task_id: Optional identifier for what the module is working
                    on (0 = none). Serves the manifest's ``task_id``.
            error_code: Optional error the module has hit (0 = none).
                    Serves the manifest's ``error_code``, so a failed
                    part is distinguishable from one that never ran.
                    Pass 0 to clear a previous error.

        All three optional fields are appended in that order, so a
        daemon that predates ``task_id`` simply ignores the extra
        bytes.

        Returns:
            True if the status was updated successfully.
        """
        if not (0 <= module_id <= MODULE_GUARDRAILS):
            raise ValueError(f"module_id must be 0-{MODULE_GUARDRAILS}, got {module_id}")
        if not (0 <= status <= MODULE_STATUS_ERROR):
            raise ValueError(f"status must be 0-{MODULE_STATUS_ERROR}, got {status}")
        payload = bytearray([module_id & 0xFF, status & 0xFF])
        if cpu_share is not None:
            payload += struct.pack("<f", max(0.0, min(1.0, float(cpu_share))))
        if task_id is not None:
            payload += struct.pack("<Q", max(0, int(task_id)))
        if error_code is not None:
            payload += struct.pack("<I", max(0, int(error_code) & 0xFFFFFFFF))
        resp = self._request(
            UPDATE_MODULE_STATUS,
            bytes(payload),
            timeout=timeout,
        )
        return bool(resp) and resp[0] == 1

    # ─── Convenience methods ───────────────────────────────────

    def reward(self, magnitude: float = 0.3, *, timeout: float | None = None) -> bool:
        """Apply a dopamine impulse (reward signal).

        Convenience wrapper around neuro_impulse.
        """
        return self.neuro_impulse(CHEM_DOPAMINE, magnitude, timeout=timeout)

    def stress(self, magnitude: float = 0.3, *, timeout: float | None = None) -> bool:
        """Apply a cortisol impulse (stress signal).

        Convenience wrapper around neuro_impulse.
        """
        return self.neuro_impulse(CHEM_CORTISOL, magnitude, timeout=timeout)

    def calm(self, magnitude: float = 0.3, *, timeout: float | None = None) -> bool:
        """Apply a GABA impulse (calming signal).

        Convenience wrapper around neuro_impulse.
        """
        return self.neuro_impulse(CHEM_GABA, magnitude, timeout=timeout)

    def bond(self, magnitude: float = 0.3, *, timeout: float | None = None) -> bool:
        """Apply an oxytocin impulse (social bonding signal).

        Convenience wrapper around neuro_impulse.
        """
        return self.neuro_impulse(CHEM_OXYTOCIN, magnitude, timeout=timeout)

    def store_observation(
        self,
        text: str,
        salience: float = 0.5,
        emotional_tag: list[float] | None = None,
        *,
        timeout: float | None = None,
    ) -> bool:
        """Store an observation event with the current timestamp.

        Convenience wrapper around store_event that fills in the
        timestamp and event type automatically.
        """
        if emotional_tag is None:
            # Neutral emotional tag
            emotional_tag = [0.5] * 12
        return self.store_event(
            timestamp=int(time.time() * 1000),
            event_type=3,  # EventType::Observation (see ring_buffer.rs)
            source_module=MODULE_SENSORY,
            salience=salience,
            emotional_tag=emotional_tag,
            text=text,
            timeout=timeout,
        )

    def heartbeat_module(
        self,
        module_id: int,
        cpu_share: float | None = None,
        *,
        task_id: int | None = None,
        error_code: int | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Send a heartbeat for a module (status stays Running).

        Convenience wrapper around update_module_status that keeps
        the module's status as Running and refreshes its heartbeat
        timestamp.

        A heartbeat with no measured work reports ``Idle`` rather than
        ``Running``: "alive but not working" and "working hard" are
        different facts, and the manifest has a status for each. A
        module with an error keeps ``Error`` so the failure is not
        overwritten by a routine heartbeat.

        Args:
            module_id: Module ID (6=Sensory, 7=Motor, etc.)
            cpu_share: Optional self-reported activity fraction
                    [0,1] for this module.
            task_id: Optional identifier for what the module is on.
            error_code: Optional error the module has hit (0 = none).

        Returns:
            True if the heartbeat was recorded.
        """
        if error_code:
            status = MODULE_STATUS_ERROR
        elif cpu_share is not None and cpu_share <= 0.0:
            status = MODULE_STATUS_IDLE
        else:
            status = MODULE_STATUS_RUNNING
        return self.update_module_status(
            module_id,
            status,
            cpu_share,
            task_id=task_id,
            error_code=error_code,
            timeout=timeout,
        )


def _decode_code_analysis(text: str) -> dict[str, Any]:
    """Decode the ``ANALYZE_CODE`` line report into its dict form.

    The wire is one record per line: ``file``, the stat keys,
    ``unit``, ``impl``, and ``edge`` records. Fields are
    tab-separated; ``;`` separates list items (type strings can hold
    commas); ``-`` marks an absent impl trait. Malformed numeric
    fields degrade to zero rather than poison the reading — a partial
    report is still a report.
    """
    stats: dict[str, Any] = {}
    units: list[dict[str, Any]] = []
    impls: list[tuple[str, str | None]] = []
    edges: list[tuple[str, str]] = []
    file_path = ""

    def _int(s: str) -> int:
        try:
            return int(s)
        except ValueError:
            return 0

    def _split(s: str) -> list[str]:
        return [item for item in s.split(";") if item]

    for line in text.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        key = fields[0]
        if key == "file":
            file_path = fields[1]
        elif key in {
            "lines",
            "functions",
            "structs",
            "enums",
            "traits",
            "impls",
            "max_complexity",
        }:
            stats[key] = _int(fields[1])
        elif key == "avg_complexity":
            try:
                stats[key] = float(fields[1])
            except ValueError:
                stats[key] = 0.0
        elif key == "unit" and len(fields) >= 14:
            flags = {f for f in fields[5].split(",") if f}
            span = fields[4].partition("-")
            units.append(
                {
                    "kind": fields[1],
                    "name": fields[2],
                    "qualified": fields[3],
                    "span": (_int(span[0]), _int(span[2])),
                    "public": "pub" in flags,
                    "async_": "async" in flags,
                    "unsafe_": "unsafe" in flags,
                    "body": "body" in flags,
                    "params": _split(fields[6]),
                    "return_type": fields[7] or None,
                    "calls": _split(fields[8]),
                    "refs": _split(fields[9]),
                    "branches": _int(fields[10]),
                    "loops": _int(fields[11]),
                    "stmts": _int(fields[12]),
                    "docs": fields[13],
                }
            )
        elif key == "impl" and len(fields) >= 3:
            impls.append((fields[1], None if fields[2] == "-" else fields[2]))
        elif key == "edge" and len(fields) >= 3:
            edges.append((fields[1], fields[2]))
    return {
        "file": file_path,
        "stats": stats,
        "units": units,
        "impls": impls,
        "edges": edges,
    }
