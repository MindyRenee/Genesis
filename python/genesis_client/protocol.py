"""Binary protocol constants and low-level framing.

This module implements the wire protocol used by the Genesis subcognitive
daemon. The protocol is length-prefixed binary (not JSON) for speed and
type safety.

Message format:
    [u32 LE: total_payload_len] [u8: command_id] [payload bytes]

The first 4 bytes are the total payload length (including the command_id
byte). The command_id determines how to interpret the rest of the payload.

All constants must match the Rust daemon's ``src/daemon/ipc.rs``.
The Rust daemon is the authoritative source; this module mirrors it.
"""

from __future__ import annotations

import socket
import struct

from .exceptions import ConnectionError

__all__ = [
    "ADVANCE_NEURO",
    "APPLY_BODY_CONTROL",
    "ARCHIVE_EPISODE",
    "ASSOCIATE",
    "CHEM_ACETYLCHOLINE",
    "CHEM_ADENOSINE",
    "CHEM_BDNF",
    "CHEM_CORTISOL",
    "CHEM_CRH",
    "CHEM_DOPAMINE",
    "CHEM_ENDOCANNABINOID",
    "CHEM_ENDORPHIN",
    "CHEM_EPINEPHRINE",
    "CHEM_GABA",
    "CHEM_GLUTAMATE",
    "CHEM_HISTAMINE",
    "CHEM_MELATONIN",
    "CHEM_NAMES",
    "CHEM_NOREPINEPHRINE",
    "CHEM_OREXIN",
    "CHEM_OXYTOCIN",
    "CHEM_SEROTONIN",
    "CHEM_VASOPRESSIN",
    "CONSOLIDATE",
    "DREAM",
    "ERROR_INTERNAL_ERROR",
    "ERROR_INVALID_VALUE",
    "ERROR_OK",
    "ERROR_PAYLOAD_TOO_LONG",
    "ERROR_PAYLOAD_TOO_SHORT",
    "ERROR_READ_FAILED",
    "ERROR_UNKNOWN_COMMAND",
    "ERROR_VERSION_MISMATCH",
    "FIND_SIMILAR",
    "GET_BODY_CONTROL",
    "GET_BODY_STATE",
    "GET_INFERENCE_SUMMARY",
    "GET_MEMORY_STATS",
    "GET_NEURO_SUMMARY",
    "GET_PHASE",
    "GET_PLASTICITY_PROFILE",
    "GET_RECENT_EPISODES",
    "GET_STATE",
    "GET_SUBSYSTEM_TELEMETRY",
    "HANDSHAKE",
    "MAX_MESSAGE_LEN",
    "MODULE_ATTENTION",
    "MODULE_DREAMING",
    "MODULE_EMOTION",
    "MODULE_GUARDRAILS",
    "MODULE_INTENTION",
    "MODULE_LANGUAGE",
    "MODULE_MEMORY",
    "MODULE_METACOGNITION",
    "MODULE_MOTOR",
    "MODULE_REASONING",
    "MODULE_SENSORY",
    "MODULE_STATUS_ERROR",
    "MODULE_STATUS_IDLE",
    "MODULE_STATUS_RUNNING",
    "MODULE_STATUS_STARTING",
    "MODULE_STATUS_STOPPED",
    "MODULE_STATUS_STOPPING",
    "MODULE_SUBCOGNITIVE",
    "NEURO_ADJUST_BASELINE",
    "NEURO_IMPULSE",
    "NOTIFY_DREAM_INSIGHT",
    "NOTIFY_EPISODE_CONSOLIDATED",
    "NOTIFY_ERROR_DETECTED",
    "NOTIFY_PHASE_CHANGED",
    "PHASE_ACTIVE",
    "PHASE_ALERT",
    "PHASE_DROWSY",
    "PHASE_FLOW",
    "PHASE_NAMES",
    "PHASE_NREM",
    "PHASE_OVERWHELMED",
    "PHASE_REM",
    "PHASE_SLEEPING",
    "PHASE_STRESS",
    "PING",
    "PROTOCOL_VERSION",
    "READ_SENSORS",
    "RETRIEVE_EPISODE",
    "SAVE_INFERENCE",
    "SEARCH_EPISODES",
    "SET_WAKE_ALARM",
    "SET_ZONE",
    "SHUTDOWN",
    "STORE_EPISODE",
    "STORE_EVENT",
    "SYNC",
    "UPDATE_MODULE_STATUS",
    "UPDATE_USER_AFFECT",
    "ZONE_ANALYSIS",
    "ZONE_CODING",
    "ZONE_CONVERSATION",
    "ZONE_ERROR_RECOVERY",
    "ZONE_IDLE",
    "ZONE_LEARNING",
    "ZONE_NAMES",
    "ZONE_PLANNING",
    "ZONE_REFLECTION",
    "ZONE_SLEEPING",
    "_F32",
    "_I32",
    "_U8",
    "_U16",
    "_U32",
    "_U64",
    "error_name",
    "is_error_byte",
    "pack_message",
    "read_exact",
    "read_message",
    "unpack_message",
    "write_message",
]

# ─── Message framing limits ───────────────────────────────────
# Largest message the client will accept. Must be >= the daemon's
# own limit (1 MiB) so legitimate responses are not rejected.
MAX_MESSAGE_LEN = 1 << 20  # 1 MiB

# ─── Command IDs ──────────────────────────────────────────────

GET_STATE = 1
GET_NEURO_SUMMARY = 2
STORE_EVENT = 3
RETRIEVE_EPISODE = 4
FIND_SIMILAR = 5
SET_ZONE = 6
GET_PHASE = 7
PING = 8
NEURO_IMPULSE = 9
GET_MEMORY_STATS = 10
SYNC = 11
SHUTDOWN = 12
UPDATE_MODULE_STATUS = 13
GET_RECENT_EPISODES = 14
HANDSHAKE = 15
GET_PLASTICITY_PROFILE = 16
GET_BODY_STATE = 17
GET_BODY_CONTROL = 18
GET_INFERENCE_SUMMARY = 19
UPDATE_USER_AFFECT = 20
SEARCH_EPISODES = 22
NEURO_ADJUST_BASELINE = 23

# Reactive commands (mind-driven, not tick-driven)
ADVANCE_NEURO = 24
CONSOLIDATE = 25
ASSOCIATE = 26
DREAM = 27
READ_SENSORS = 28
APPLY_BODY_CONTROL = 29
SAVE_INFERENCE = 30
ARCHIVE_EPISODE = 31
STORE_EPISODE = 32
# Per-subsystem silicon telemetry — which part of the mind's process
# tree is firing (per-process CPU/I/O share + miss ratios).
GET_SUBSYSTEM_TELEMETRY = 33
# RTC wake alarm — arm or disarm the hardware interrupt that resumes
# the machine from suspend. Payload: [u64 epoch_secs] (0 = disarm).
# Response: [u8 ok][u64 armed_epoch]. Arming the alarm does not itself
# suspend the machine — the two operations stay deliberately separate.
SET_WAKE_ALARM = 34

# ─── Error response codes ─────────────────────────────────────
# When a command fails, the daemon returns a single-byte response
# with one of these codes (high bit set). The client checks the
# first byte of every response: if the high bit is set, it's an
# error, not data. Must match the `error` module in
# src/daemon/ipc.rs.

ERROR_OK = 0
ERROR_PAYLOAD_TOO_SHORT = 0x80
ERROR_PAYLOAD_TOO_LONG = 0x81
ERROR_UNKNOWN_COMMAND = 0x82
ERROR_VERSION_MISMATCH = 0x83
ERROR_INTERNAL_ERROR = 0x84
ERROR_INVALID_VALUE = 0x85
ERROR_READ_FAILED = 0x86

_ERROR_NAMES = {
    ERROR_PAYLOAD_TOO_SHORT: "payload too short",
    ERROR_PAYLOAD_TOO_LONG: "payload too long",
    ERROR_UNKNOWN_COMMAND: "unknown command",
    ERROR_VERSION_MISMATCH: "version mismatch",
    ERROR_INTERNAL_ERROR: "internal error",
    ERROR_INVALID_VALUE: "invalid value",
    ERROR_READ_FAILED: "read failed (seqlock contention or corrupt state)",
}


def is_error_byte(b: int) -> bool:
    """Check if a response byte is an error code (high bit set)."""
    return b >= 0x80


def error_name(code: int) -> str:
    """Human-readable name for an error code."""
    return _ERROR_NAMES.get(code, f"unknown error 0x{code:02x}")


# ─── Protocol version ─────────────────────────────────────────
# Must match PROTOCOL_VERSION in src/daemon/ipc.rs. The client
# sends this on connect; the daemon rejects mismatches.
#
# v2: the body-control response gained a trailing `cpu_boost` byte
# after the EPP string.
#
# v3: BodyState grew the timing/involuntary/senescence layer — 9 f32
# fields (pulse_hz, pulse, throttle_state, top_freq_share, psi_cpu,
# psi_io, psi_mem, battery_cycles, entropy_level) plus 2 u8 fields
# (clocksource, suspend_caps) appended before desc_len; the fixed
# header is now 116 bytes (was 78). Also adds SET_WAKE_ALARM (34).

PROTOCOL_VERSION = 3

# ─── Notification IDs (server → client, unsolicited) ─────────
#
# Reserved / unwired: the IPC channel is strictly request-response,
# so the daemon never sends these today. The cognitive mind detects
# phase changes and dream insights by polling instead — see
# ``genesis_cognitive/notifications.py`` (a pull-based queue). These
# IDs reserve the 100+ opcode range for a future push mechanism and
# mirror the Rust ``notify`` module in ``src/daemon/ipc.rs``.

NOTIFY_PHASE_CHANGED = 100
NOTIFY_EPISODE_CONSOLIDATED = 101
NOTIFY_DREAM_INSIGHT = 102
NOTIFY_ERROR_DETECTED = 103

# ─── Cognitive zone IDs ───────────────────────────────────────

ZONE_IDLE = 0
ZONE_CONVERSATION = 1
ZONE_CODING = 2
ZONE_ANALYSIS = 3
ZONE_LEARNING = 4
ZONE_REFLECTION = 5
ZONE_ERROR_RECOVERY = 6
ZONE_PLANNING = 7
ZONE_SLEEPING = 8

# ─── Mental phase IDs ─────────────────────────────────────────

PHASE_ACTIVE = 0
PHASE_ALERT = 1
PHASE_FLOW = 2
PHASE_STRESS = 3
PHASE_DROWSY = 4
PHASE_NREM = 5
PHASE_REM = 6
PHASE_OVERWHELMED = 7

# Backward-compatible alias for code that still references the old
# single sleep phase. NREM is the default sleep state (most of the
# night is spent in NREM).
PHASE_SLEEPING = PHASE_NREM

# ─── Neurochemical IDs ────────────────────────────────────────
#
# Indices match the Rust NeurochemicalId enum order.
# v2 had 12 chemicals (indices 0-11).
# v3 adds 6 more (indices 12-17): endocannabinoid, vasopressin, CRH,
# orexin, epinephrine, melatonin.

CHEM_DOPAMINE = 0
CHEM_SEROTONIN = 1
CHEM_NOREPINEPHRINE = 2
CHEM_ACETYLCHOLINE = 3
CHEM_GABA = 4
CHEM_GLUTAMATE = 5
CHEM_CORTISOL = 6
CHEM_OXYTOCIN = 7
CHEM_ENDORPHIN = 8
CHEM_HISTAMINE = 9
CHEM_ADENOSINE = 10
CHEM_BDNF = 11
# v3 chemicals (schema v3, added in the 18-chemical expansion)
CHEM_ENDOCANNABINOID = 12
CHEM_VASOPRESSIN = 13
CHEM_CRH = 14
CHEM_OREXIN = 15
CHEM_EPINEPHRINE = 16
CHEM_MELATONIN = 17

CHEM_NAMES = {
    CHEM_DOPAMINE: "dopamine",
    CHEM_SEROTONIN: "serotonin",
    CHEM_NOREPINEPHRINE: "norepinephrine",
    CHEM_ACETYLCHOLINE: "acetylcholine",
    CHEM_GABA: "gaba",
    CHEM_GLUTAMATE: "glutamate",
    CHEM_CORTISOL: "cortisol",
    CHEM_OXYTOCIN: "oxytocin",
    CHEM_ENDORPHIN: "endorphin",
    CHEM_HISTAMINE: "histamine",
    CHEM_ADENOSINE: "adenosine",
    CHEM_BDNF: "bdnf",
    # v3 chemicals
    CHEM_ENDOCANNABINOID: "endocannabinoid",
    CHEM_VASOPRESSIN: "vasopressin",
    CHEM_CRH: "crh",
    CHEM_OREXIN: "orexin",
    CHEM_EPINEPHRINE: "epinephrine",
    CHEM_MELATONIN: "melatonin",
}

PHASE_NAMES = {
    PHASE_ACTIVE: "active",
    PHASE_ALERT: "alert",
    PHASE_FLOW: "flow",
    PHASE_STRESS: "stress",
    PHASE_DROWSY: "drowsy",
    PHASE_NREM: "nrem",
    PHASE_REM: "rem",
    PHASE_OVERWHELMED: "overwhelmed",
}

ZONE_NAMES = {
    ZONE_IDLE: "idle",
    ZONE_CONVERSATION: "conversation",
    ZONE_CODING: "coding",
    ZONE_ANALYSIS: "analysis",
    ZONE_LEARNING: "learning",
    ZONE_REFLECTION: "reflection",
    ZONE_ERROR_RECOVERY: "error-recovery",
    ZONE_PLANNING: "planning",
    ZONE_SLEEPING: "sleeping",
}

# ─── Module IDs (for UpdateModuleStatus) ──────────────────────

MODULE_SUBCOGNITIVE = 0
MODULE_ATTENTION = 1
MODULE_MEMORY = 2
MODULE_EMOTION = 3
MODULE_LANGUAGE = 4
MODULE_REASONING = 5
MODULE_SENSORY = 6
MODULE_MOTOR = 7
MODULE_METACOGNITION = 8
MODULE_DREAMING = 9
MODULE_INTENTION = 10
MODULE_GUARDRAILS = 11

# ─── Module status values ─────────────────────────────────────

MODULE_STATUS_STOPPED = 0
MODULE_STATUS_STARTING = 1
MODULE_STATUS_RUNNING = 2
MODULE_STATUS_IDLE = 3
MODULE_STATUS_STOPPING = 4
MODULE_STATUS_ERROR = 5

# ─── Wire format helpers ──────────────────────────────────────

# Struct formats for common wire types
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_F32 = struct.Struct("<f")
_I32 = struct.Struct("<i")
_U16 = struct.Struct("<H")
_U8 = struct.Struct("<B")


def pack_message(cmd_id: int, payload: bytes = b"") -> bytes:
    """Pack a command + payload into a framed message.

    Format: [u32 LE: total_len] [u8: cmd_id] [payload]
    where total_len = 1 + len(payload).

    Raises:
        ValueError: If cmd_id is not a valid u8 (0-255) or the
            total message length exceeds ``MAX_MESSAGE_LEN``.
    """
    if not 0 <= cmd_id <= 0xFF:
        raise ValueError(f"cmd_id must be 0-255, got {cmd_id}")
    total_len = 1 + len(payload)
    if total_len > MAX_MESSAGE_LEN:
        raise ValueError(
            f"message length {total_len} exceeds maximum {MAX_MESSAGE_LEN}"
        )
    return _U32.pack(total_len) + bytes([cmd_id]) + payload


def unpack_message(data: bytes) -> tuple[int, bytes]:
    """Unpack a framed message into (command_id, payload).

    The input must contain at least the 4-byte length prefix and the
    payload that follows. Returns (cmd_id, payload) where payload
    excludes the cmd_id byte.

    Raises:
        ValueError: If the data is too short for the length prefix,
            the declared length is zero (protocol violation — must
            include at least the command_id byte), the declared
            length exceeds ``MAX_MESSAGE_LEN``, or the data is
            shorter than the declared length.
    """
    if len(data) < 4:
        raise ValueError("message too short for length prefix")
    total_len = _U32.unpack(data[:4])[0]
    if total_len == 0:
        raise ValueError("message length is zero (must include at least the command_id byte)")
    if total_len > MAX_MESSAGE_LEN:
        raise ValueError(
            f"message length {total_len} exceeds maximum {MAX_MESSAGE_LEN}"
        )
    if len(data) < 4 + total_len:
        raise ValueError("message shorter than declared length")
    cmd_id = data[4]
    payload = data[5 : 4 + total_len]
    return (cmd_id, payload)


def read_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from a socket, blocking until all are read.

    Raises ConnectionError if the socket closes before n bytes are read
    or if a socket error occurs during the read.
    """
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError as e:
            raise ConnectionError(f"socket error during read: {e}") from e
        if not chunk:
            raise ConnectionError("socket closed during read")
        buf.extend(chunk)
    return bytes(buf)


def read_message(sock: socket.socket) -> tuple[int, bytes]:
    """Read a complete framed message from a socket.

    Returns (cmd_id, payload) where payload excludes the cmd_id byte.

    Raises ConnectionError if:
    - The length prefix is zero (protocol violation — must include
      at least the command_id byte). The Rust daemon rejects
      zero-length frames with an error; the client matches this.
    - The length prefix exceeds ``MAX_MESSAGE_LEN``, which prevents
      a malicious or buggy daemon from forcing the client to allocate
      an unbounded amount of memory.
    - The socket closes or errors during the read.
    """
    len_buf = read_exact(sock, 4)
    total_len = _U32.unpack(len_buf)[0]
    if total_len == 0:
        raise ConnectionError(
            "message length is zero (must include at least the command_id byte)"
        )
    if total_len > MAX_MESSAGE_LEN:
        raise ConnectionError(
            f"message length {total_len} exceeds maximum {MAX_MESSAGE_LEN}"
        )
    payload_buf = read_exact(sock, total_len)
    cmd_id = payload_buf[0]
    payload = payload_buf[1:]
    return (cmd_id, payload)


def write_message(sock: socket.socket, cmd_id: int, payload: bytes = b"") -> None:
    """Write a framed message to a socket.

    Raises ValueError if cmd_id or payload size is invalid (see
    ``pack_message`` for details).
    """
    sock.sendall(pack_message(cmd_id, payload))
