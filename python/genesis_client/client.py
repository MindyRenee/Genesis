"""Genesis client — the cognitive mind's bridge to the subcognitive.

This is the Python library that the cognitive mind uses to communicate
with the Rust subcognitive daemon. It connects to the daemon's Unix
domain socket and provides a clean, Pythonic API for all IPC commands.

The protocol is request-response only. Subcognitive→cognitive
notifications (phase changes, dream insights) are handled separately
by ``genesis_cognitive.notifications``, a pull-based queue that
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
    APPLY_BODY_CONTROL,
    ARCHIVE_EPISODE,
    ASSOCIATE,
    # Chemical IDs (for convenience wrappers)
    CHEM_CORTISOL,
    CHEM_DOPAMINE,
    CHEM_GABA,
    CHEM_OXYTOCIN,
    CONSOLIDATE,
    DREAM,
    FIND_SIMILAR,
    GET_BODY_CONTROL,
    GET_BODY_STATE,
    GET_INFERENCE_SUMMARY,
    GET_MEMORY_STATS,
    GET_NEURO_SUMMARY,
    GET_PHASE,
    GET_PLASTICITY_PROFILE,
    GET_RECENT_EPISODES,
    # Command IDs
    GET_STATE,
    GET_SUBSYSTEM_TELEMETRY,
    HANDSHAKE,
    # Module IDs (for validation and store_observation)
    MODULE_GUARDRAILS,
    MODULE_SENSORY,
    # Module status (for heartbeat_module)
    MODULE_STATUS_ERROR,
    MODULE_STATUS_RUNNING,
    NEURO_ADJUST_BASELINE,
    NEURO_IMPULSE,
    PING,
    PROTOCOL_VERSION,
    READ_SENSORS,
    RETRIEVE_EPISODE,
    SAVE_INFERENCE,
    SEARCH_EPISODES,
    SET_WAKE_ALARM,
    SET_ZONE,
    SHUTDOWN,
    STORE_EPISODE,
    STORE_EVENT,
    SYNC,
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
from .types import (
    BodyControlState,
    BodyState,
    CoreState,
    Episode,
    InferenceSummary,
    MemoryStats,
    NeuroSummary,
    PhaseInfo,
    PingResponse,
    PlasticityProfile,
    RecentEpisode,
    SimilarEpisode,
    SubsystemReport,
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
        self._lock = threading.RLock()
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

    def _close_socket(self) -> None:
        """Close the current socket if open and clear the reference.

        Does NOT clear the cache or acquire the lock — used internally
        by ``_connect`` on error paths where the caller (``connect``)
        has already cleared the cache and holds the lock.
        """
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError as e:
                logger.debug(f'_close_socket failed: {e}')
            self._sock = None

    def _connect(self) -> None:
        """Open the socket and perform the protocol handshake (no lock)."""
        if not os.path.exists(self.socket_path):
            raise DaemonNotRunning(f"socket not found: {self.socket_path} (is the daemon running?)")
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.settimeout(
                self.request_timeout
            )  # 30s timeout — daemon should never take longer
            self._sock.connect(self.socket_path)
        except OSError as e:
            # Clean up the dead socket so a later _ensure_connected()
            # doesn't reuse it — it would fail on a broken connection
            # instead of reconnecting.
            self._close_socket()
            raise GenesisConnectionError(f"failed to connect: {e}") from e

        # Protocol version handshake
        try:
            write_message(self._sock, HANDSHAKE, bytes([PROTOCOL_VERSION]))
            resp_cmd, resp_data = read_message(self._sock)
        except OSError as e:
            self._close_socket()
            raise GenesisConnectionError(f"handshake failed: {e}") from e

        if resp_cmd != HANDSHAKE or not resp_data:
            self._close_socket()
            raise ProtocolError(
                f"handshake failed: unexpected response (cmd={resp_cmd})"
            )
        if is_error_byte(resp_data[0]):
            # The daemon signalled a version mismatch: the first byte
            # is ERROR_VERSION_MISMATCH and the second (if present) is
            # its actual version.
            self._close_socket()
            actual = resp_data[1] if len(resp_data) > 1 else None
            raise ProtocolError(
                f"protocol version mismatch: client={PROTOCOL_VERSION}, "
                f"daemon={actual if actual is not None else 'unknown'}"
            )
        daemon_version = resp_data[0]
        if daemon_version != PROTOCOL_VERSION:
            self._close_socket()
            raise ProtocolError(
                f"protocol version mismatch: client={PROTOCOL_VERSION}, "
                f"daemon={daemon_version}"
            )

    def disconnect(self) -> None:
        """Close the connection to the daemon."""
        with self._lock:
            self._cache.clear()
            if self._sock is not None:
                try:
                    self._sock.close()
                except OSError as e:
                    logger.debug(repr(e))
                self._sock = None

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
        """Return the open socket or raise if not connected."""
        if self._sock is None:
            raise GenesisConnectionError("not connected — call connect() first")
        return self._sock

    # Read-only commands that are safe to cache with a short TTL.
    # State only changes when the mind advances neurochemistry (via
    # ADVANCE_NEURO) or issues a write command, so cached values are
    # at most one heartbeat cycle stale — acceptable for emotion/state
    # assessment in background threads. Write commands (STORE_EVENT,
    # NEURO_IMPULSE, SET_ZONE, SYNC, SHUTDOWN, UPDATE_*) are never
    # cached.
    _CACHEABLE_CMDS = frozenset({
        GET_STATE, GET_NEURO_SUMMARY, GET_PHASE, GET_MEMORY_STATS,
        GET_INFERENCE_SUMMARY, GET_BODY_STATE, GET_BODY_CONTROL,
        GET_PLASTICITY_PROFILE,
    })

    # Write commands that invalidate the cache. When these are sent,
    # all cached read-only responses are cleared so the next read
    # sees the updated state. Without this, a neuro_impulse followed
    # immediately by feel() would return the pre-impulse state for
    # up to 150ms.
    _CACHE_INVALIDATING_CMDS = frozenset({
        STORE_EVENT, NEURO_IMPULSE, NEURO_ADJUST_BASELINE, SET_ZONE, SYNC,
        SHUTDOWN, UPDATE_MODULE_STATUS, UPDATE_USER_AFFECT, ARCHIVE_EPISODE,
        STORE_EPISODE,
        ADVANCE_NEURO, CONSOLIDATE, ASSOCIATE, DREAM, READ_SENSORS,
        APPLY_BODY_CONTROL, SAVE_INFERENCE,
    })

    _RETRYABLE_CMDS = _CACHEABLE_CMDS | frozenset({
        PING, RETRIEVE_EPISODE, FIND_SIMILAR, GET_RECENT_EPISODES, SEARCH_EPISODES,
    })

    # Commands whose response first byte is an ack/status byte (1 =
    # success, 0 = not-found/invalid, 0x80+ = error). For these, the
    # error check examines the first byte regardless of response length
    # — the daemon pads error responses to the expected size (e.g.
    # STORE_EVENT returns 8 bytes, ADVANCE_NEURO returns 21 bytes).
    _ACK_CMDS = frozenset({
        PING, STORE_EVENT, STORE_EPISODE, SET_ZONE, NEURO_IMPULSE, SYNC,
        SHUTDOWN, UPDATE_MODULE_STATUS, UPDATE_USER_AFFECT,
        ARCHIVE_EPISODE,
        NEURO_ADJUST_BASELINE, ADVANCE_NEURO, CONSOLIDATE, ASSOCIATE,
        DREAM, READ_SENSORS, APPLY_BODY_CONTROL, SAVE_INFERENCE,
    })

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
        interleave on the shared socket.

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
        with self._lock:
            self._ensure_connected()
            # Check cache for read-only commands
            if cmd_id in self._CACHEABLE_CMDS and not payload:
                cached = self._cache.get(cmd_id)
                if cached is not None:
                    ts, data = cached
                    if time.monotonic() - ts < self._cache_ttl:
                        return data

            # Write commands invalidate all cached state — the next read
            # must go to the daemon to see the updated values.
            if cmd_id in self._CACHE_INVALIDATING_CMDS:
                self._cache.clear()

            for attempt in range(2):
                sock = self._ensure_connected()
                old_timeout = sock.gettimeout()
                try:
                    sock.settimeout(effective_timeout)
                    write_message(sock, cmd_id, payload)
                    resp_cmd, resp_data = read_message(sock)
                except (GenesisConnectionError, ConnectionError, OSError) as e:
                    self.disconnect()
                    # Connection broke — try to reconnect once
                    if attempt == 0:
                        logger.warning(f"connection lost ({e}), attempting reconnect...")
                        try:
                            self.connect()
                        except Exception as re:
                            raise GenesisConnectionError(
                                f"reconnect failed: {re}"
                            ) from re
                        if cmd_id not in self._RETRYABLE_CMDS:
                            raise GenesisConnectionError(
                                f"connection lost after command {cmd_id}: outcome unknown; "
                                "command was not retried"
                            ) from e
                        continue
                    raise GenesisConnectionError(f"connection lost: {e}") from e
                finally:
                    try:
                        sock.settimeout(old_timeout)
                    except OSError as e:
                        logger.debug(f"failed to restore socket timeout: {e}")

                if resp_cmd != cmd_id:
                    self.disconnect()
                    raise ProtocolError(
                        f"response command {resp_cmd} != request command {cmd_id}"
                    )
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
                        raise ProtocolError(
                            f"daemon error: {error_name(err_code)} (0x{err_code:02x})"
                        )
                # Cache read-only responses
                if cmd_id in self._CACHEABLE_CMDS and not payload:
                    self._cache[cmd_id] = (time.monotonic(), resp_data)
                return resp_data

        raise GenesisConnectionError("request failed after reconnect attempt")

    # ─── Commands ──────────────────────────────────────────────

    def ping(self, *, timeout: float | None = None) -> PingResponse:
        """Ping the daemon. Returns ack + uptime."""
        resp = self._request(PING, timeout=timeout)
        return PingResponse.unpack(resp)

    def get_state(self, *, timeout: float | None = None) -> CoreState:
        """Get the full core state (3288 bytes), parsed into useful fields."""
        resp = self._request(GET_STATE, timeout=timeout)
        return CoreState.unpack(resp)

    def get_neuro_summary(self, *, timeout: float | None = None) -> NeuroSummary:
        """Get the compact neurochemical summary (32 bytes).

        This is the most efficient way to check how Genesis feels.
        """
        resp = self._request(GET_NEURO_SUMMARY, timeout=timeout)
        return NeuroSummary.unpack(resp)

    def get_plasticity_profile(
        self, *, timeout: float | None = None
    ) -> PlasticityProfile:
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

    def set_wake_alarm(
        self, epoch_secs: int, *, timeout: float | None = None
    ) -> int | None:
        """Arm or disarm the RTC wake alarm — when the machine exists next.

        ``epoch_secs`` is a Unix timestamp; 0 disarms. Returns the
        alarm epoch the hardware reports afterward (0 = none armed),
        or ``None`` if the daemon could not arm it (helper missing or
        sudo not installed — see ``scripts/install_sudoers.sh``).

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

    def get_inference_summary(
        self, *, timeout: float | None = None
    ) -> InferenceSummary:
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

        This feeds the dyadic affective model, which couples Genesis's
        neurochemistry to the inferred user state via oxytocin-mediated
        attunement.

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
            emotional_tag: 12 effective neurochemical levels.
            text: Event description (max 188 bytes).

        Returns:
            True if the event was stored successfully.
        """
        if len(emotional_tag) != 12:
            raise ValueError(f"emotional_tag must have 12 elements, got {len(emotional_tag)}")
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
            emotional_tag: 12 effective neurochemical levels.
            text: Event description (max 65535 bytes — the wire protocol's
            u16 text_len limit). The LTM store itself can handle up to
            256 KB per episode; the bottleneck is the u16 length field.

        Returns:
            The real LTM episode ID on success, or None if the store
            failed.
        """
        if len(emotional_tag) != 12:
            raise ValueError(f"emotional_tag must have 12 elements, got {len(emotional_tag)}")
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

    def retrieve_episode(
        self, episode_id: int, *, timeout: float | None = None
    ) -> Episode:
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
                    f"source_module must be 0-254 (255 is reserved for 'all'), "
                    f"got {source_module}"
                )
            filter_byte = source_module & 0xFF
        resp = self._request(
            GET_RECENT_EPISODES,
            bytes([limit_byte, filter_byte]),
            timeout=timeout,
        )
        return unpack_recent_episodes(resp)

    def archive_episode(
        self, episode_id: int, *, timeout: float | None = None
    ) -> bool:
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
        payload = _U32.pack(limit) + _U32.pack(offset)
        resp = self._request(SEARCH_EPISODES, payload, timeout=timeout)
        return unpack_recent_episodes(resp)

    def set_zone(self, zone: int, *, timeout: float | None = None) -> bool:
        """Set the cognitive task zone (what Genesis is doing).

        Args:
            zone: Zone ID (0=Idle, 1=Conversation, 2=Coding, etc.)

        Returns:
            True if the zone was set successfully.
        """
        if not (0 <= zone <= ZONE_SLEEPING):
            raise ValueError(
                f"zone must be 0-{ZONE_SLEEPING}, got {zone}"
            )
        resp = self._request(SET_ZONE, bytes([zone & 0xFF]), timeout=timeout)
        return bool(resp) and resp[0] == 1

    def neuro_impulse(
        self, chem: int, magnitude: float, *, timeout: float | None = None
    ) -> bool:
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
        self, dt: float = 0.2, *, timeout: float | None = None
    ) -> tuple[float, float, float, float, int] | None:
        """Advance neurochemical dynamics by dt and run active inference.

        This is the core physics integration step. The mind calls
        this when brain waves say it's time to advance.

        ``dt`` is the simulated time to advance, in seconds. It is
        clamped to the daemon's accepted range [0.001, 10.0] (the
        same bounds the daemon enforces) so that elapsed-time callers
        can pass raw measured intervals: longer stalls are truncated
        rather than rejected.

        Returns:
            (surprise, free_energy, precision, allostatic_load,
             tick_count) or None on error.
        """
        # Mirror the daemon's dt validation (ipc.rs clamps to
        # [0.001, 10.0]) so callers can pass unclamped elapsed time.
        dt = min(10.0, max(0.001, dt))
        resp = self._request(ADVANCE_NEURO, _F32.pack(dt), timeout=timeout)
        if not resp or resp[0] != 1 or len(resp) < 21:
            return None
        surprise = _F32.unpack(resp[1:5])[0]
        free_energy = _F32.unpack(resp[5:9])[0]
        precision = _F32.unpack(resp[9:13])[0]
        allostatic = _F32.unpack(resp[13:17])[0]
        tick_count = _U32.unpack(resp[17:21])[0]
        return (surprise, free_energy, precision, allostatic, tick_count)

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

        Returns:
            True if the status was updated successfully.
        """
        if not (0 <= module_id <= MODULE_GUARDRAILS):
            raise ValueError(
                f"module_id must be 0-{MODULE_GUARDRAILS}, got {module_id}"
            )
        if not (0 <= status <= MODULE_STATUS_ERROR):
            raise ValueError(
                f"status must be 0-{MODULE_STATUS_ERROR}, got {status}"
            )
        payload = bytearray([module_id & 0xFF, status & 0xFF])
        if cpu_share is not None:
            payload += struct.pack(
                "<f", max(0.0, min(1.0, float(cpu_share)))
            )
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
        timeout: float | None = None,
    ) -> bool:
        """Send a heartbeat for a module (status stays Running).

        Convenience wrapper around update_module_status that keeps
        the module's status as Running and refreshes its heartbeat
        timestamp.

        Args:
            module_id: Module ID (6=Sensory, 7=Motor, etc.)
            cpu_share: Optional self-reported activity fraction
                    [0,1] for this module.

        Returns:
            True if the heartbeat was recorded.
        """
        return self.update_module_status(
            module_id, MODULE_STATUS_RUNNING, cpu_share, timeout=timeout
        )
