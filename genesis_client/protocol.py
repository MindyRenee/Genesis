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

# Provenance marker — one letter of the creator lattice (see
# PROVENANCE.md).
_MARKER_G09 = 0x4753094544B80616

__all__ = [
    "ADVANCE_NEURO",
    "ADVANCE_PHYSICS",
    "ANALYZE_CODE",
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
    "DELETE_EPISODE",
    "DREAM",
    "DROP_TENSOR",
    "ERROR_INTERNAL_ERROR",
    "ERROR_INVALID_VALUE",
    "ERROR_OK",
    "ERROR_PAYLOAD_TOO_LONG",
    "ERROR_PAYLOAD_TOO_SHORT",
    "ERROR_READ_FAILED",
    "ERROR_UNKNOWN_COMMAND",
    "ERROR_VERSION_MISMATCH",
    "FIND_SIMILAR",
    "FIND_SIMILAR_ARCHIVED",
    "GET_BODY_CONTROL",
    "GET_BODY_STATE",
    "GET_DYADIC_STATE",
    "GET_FIELD_PROBE",
    "GET_INFERENCE_SUMMARY",
    "GET_INTERNALS",
    "GET_ION_SUMMARY",
    "GET_MEMORY_STATS",
    "GET_MOTOR_TARGET",
    "GET_NEURO_SUMMARY",
    "GET_OPTIC_STATE",
    "GET_PHASE",
    "GET_PLASTICITY_PROFILE",
    "GET_PROCESS_HEALTH",
    "GET_RECENT_EPISODES",
    "GET_SENSOR_PRESENCE",
    "GET_STATE",
    "GET_SUBSYSTEM_TELEMETRY",
    "GET_TENSOR",
    "GET_ZONE_TRANSITIONS",
    "HANDSHAKE",
    "LIST_TENSORS",
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
    "PUT_TENSOR",
    "READ_SENSORS",
    "RETRIEVE_EPISODE",
    "SAVE_INFERENCE",
    "SEARCH_ARCHIVED",
    "SEARCH_EPISODES",
    "SENSOR_AC_ADAPTER",
    "SENSOR_BATTERY",
    "SENSOR_BATTERY_CYCLES",
    "SENSOR_CORE_VOLTAGE",
    "SENSOR_FAN",
    "SENSOR_FREQUENCY",
    "SENSOR_GPE",
    "SENSOR_LOCAL_TIMER",
    "SENSOR_MAX_FREQUENCY",
    "SENSOR_MEMORY",
    "SENSOR_NAMES",
    "SENSOR_PERF_COUNTERS",
    "SENSOR_POWER",
    "SENSOR_PSI",
    "SENSOR_RAPL",
    "SENSOR_SUPPLY_VOLTAGE",
    "SENSOR_TEMPERATURE",
    "SET_COGNITIVE_PID",
    "SET_FIELD_CARRIER",
    "SET_FIELD_PHASE",
    "SET_FOCUS",
    "SET_LID",
    "SET_OPTIC_GAIN",
    "SET_POLICY_AUTHORITY",
    "SET_VOLITION",
    "SET_WAKE_ALARM",
    "SET_ZONE",
    "SHUTDOWN",
    "STORE_EPISODE",
    "STORE_EVENT",
    "SUBSYSTEM_COGNITIVE",
    "SUBSYSTEM_DAEMON",
    "SUBSYSTEM_NAMES",
    "SUBSYSTEM_RETINA",
    "SYNC",
    "UNARCHIVE_EPISODE",
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

# The electrochemical ion layer: a compact 108-byte snapshot of
# calcium/chloride/potassium/sodium state. Additive — it adds a new
# opcode and touches no existing layout, so PROTOCOL_VERSION is
# unaffected.
#
# NOTE: this tree already spent command id 36 on GET_SENSOR_PRESENCE,
# so the ion layer continues past ADVANCE_PHYSICS at 40 rather than
# reusing genesis2's 36.
GET_ION_SUMMARY = 40

# Number of neurochemical levels in the ``emotional_tag`` vector sent
# with STORE_EVENT / STORE_EPISODE. Fixed by the wire layout: the
# daemon reads ``[f32; 12]`` (see ``STORE_EPISODE`` in src/daemon/ipc.rs)
# and passes the values through positionally, so sender and receiver
# must agree on the width or the tag is silently truncated.
EMOTIONAL_TAG_SIZE = 12

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
#
# ADVANCE_NEURO carries a 4-byte dt placeholder that the daemon reads and
# ignores: the daemon owns time and measures its own elapsed interval.
# Mind.advance_neuro takes no `dt`, so the coupling cannot be
# reintroduced from this layer. The bounds below are what the *body*
# applies to itself; the brain-wave oscillator uses the identical range so
# the two clocks accept the same intervals (see DT_MIN).
ADVANCE_NEURO = 24

# Advance the body by an explicit step. Evaluation only, against an
# isolated daemon -- the causal assay varies dt as its independent
# variable and reads per-tick response. Mirrors ipc.rs
# `cmd::ADVANCE_PHYSICS`.
ADVANCE_PHYSICS = 39
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

# Set how much authority the inference engine has to choose its own
# policy on Genesis's behalf. Request: [f32 authority] in [0,1].
# Response: [u8 ack] + [f32 applied].
SET_POLICY_AUTHORITY = 35

# Retired — never reuse this wire number. Volitional pushes onto the
# sleep-wake network were removed: /sleep and /wake act through
# neurochemical impulses only; the chemicals control, as in a real
# brain — nothing exogenous is held on the wire.
SET_VOLITION = 46

# The body's real measured substrate dynamics — the daemon's field
# probe decomposes instruction throughput, scheduler jitter, P-state
# transitions, and interrupt flux at the wave bands' center
# frequencies, and reports the carrier rhythm it writes into real
# work. Response: v2 is 114 bytes — [u8 version][10 f32 throughput
# bands][10 f32 jitter bands][f32 throughput_mean][f32 jitter_cv]
# [f32 pstate_rate][f32 irq_rate][f32 carrier_hz][u8 samples_ok]
# [f32 carrier_lag][f32 carrier_phase][f32 carrier_coherence].
# v1 is 102 bytes without the three phase fields.
# v3 is 195 bytes — the v2 body plus the electrical + Maxwell tail:
# [10 f32 power bands (real measured watts)][f32 power_watts]
# [f32 power_sensor_hz][f32 rail_volts][f32 supply_amps]
# [f32 power_lag][f32 power_coherence][u8 electrical_present mask]
# [f32 carrier_current_a][f32 b_est_ut][f32 e_ind_vm]
# [f32 radiated_pwr_w]. The Maxwell fields are estimates — the
# current is measured, the loop geometry an assumption
# (src/daemon/maxwell.rs). v4 appends the radio tail
# [10f rf flux bands][10f rf rssi bands][f32 rf_bytes_s]
# [f32 rf_rssi_dbm][f32 rf_noise_dbm][f32 rf_irq_rate]
# [f32 rf_sensor_hz][f32 bt_bytes_s][f32 bt_events_s]
# [f32 bt_rssi_dbm][u8 rf_present mask] → 308 bytes. v5 appends
# iw station telemetry [5×f32] → 328 bytes; v6 appends the flux
# phasors [10f rf_flux_phase][10f rf_flux_phase_coherence] →
# 408 bytes.
# Layout: `src/daemon/field_probe.rs`. Additive: no CoreState or
# BodyState layout changes, so PROTOCOL_VERSION stays 3. A daemon
# without a running probe returns a single error byte — an absent
# channel, never a healthy zero.
GET_FIELD_PROBE = 47

# Push the mind's modeled carrier-band phase into the probe's
# generator — the physics-follows-model leg of the mutual phase
# lock. Request: [f32 phase_rad]. The probe extrapolates it at the
# carrier rate and pulls its generator phase toward it each tick
# (bounded first-order pull); a target not refreshed within 5 s is
# dropped so a dead mind can't steer the body's rhythm. Reply:
# [u8 ack]. Additive: PROTOCOL_VERSION stays 3.
SET_FIELD_PHASE = 48

# Carrier write control — the modulation depth and frequency of the
# probe's real-work carrier. Request: [f32 depth][f32 freq_hz].
# Depth 0..1 (0 = unmodulated work); freq 0 hands the rhythm back to
# the body's phase, positive overrides it (clamped 0.1..100 Hz) so
# evaluation can drive arbitrary bands — including infraslow rates
# where the thermal field can hold an oscillation.
# Reply: [u8 ack]. Additive: PROTOCOL_VERSION stays 3.
SET_FIELD_CARRIER = 49

# RETIRED — pinned so the wire number is never reused. The per-person
# dyadic snapshot channel was removed with the person registry; the
# interlocutor model's coupling output is applied to neurochemistry
# inside the tick.
GET_DYADIC_STATE = 50

# The daemon's process-local internals — the gears no state file or
# CoreState field exposes: every adaptive gate's full snapshot
# (bounds, period, next-fire countdown, smoothed demand), the mind
# lease, tick counter, volition drive, last-applied body policy, and
# the autonomous fallback's own gates.
# Response: [u8 ack] (1) then UTF-8 `key=value` lines — self-describing
# and additive, so a reader written today still parses tomorrow's
# longer dump. Observation only.
# Additive: PROTOCOL_VERSION stays 3.
GET_INTERNALS = 51

# The optic nerve's subcortical channel state — the body's own read
# of the retinal segment, not the mind's: photopic and melanopic
# irradiance, motion energy, the collicular orienting transient, the
# SCN light drive it currently feeds the circadian oscillator, and
# the thalamic relay gate the mind must honour (it approaches 0 in
# NREM — sensory transmission is blocked at the thalamus, so a look
# during sleep is a blind look). This is the same /genesis_retina
# segment the mind reads frames from — the daemon consumes the
# nerve's non-image-forming branches, the mind the relayed image.
# Response: [u8 ack] then 46 bytes —
#   [u8 flags][u8 pad][7 × f32][u64 frame_seq][i64 frame_age_ms]
#   flags bit0 = eye ever produced a frame, bit1 = frames fresh (<5 s)
# A daemon that has never seen a frame still answers — an absent eye
# is a reading, not an error. A daemon that predates the command
# returns UNKNOWN_COMMAND, which the client turns into None.
# Additive: PROTOCOL_VERSION stays 3.
GET_OPTIC_STATE = 52

# Corticothalamic feedback — the mind's engagement gain onto the
# thalamic relay, the layer-6 V1→dLGN projection. Modulatory only:
# it scales the relay within ±40 % and can never open a gate the
# sleep stage has closed. Request: [f32 gain] in 0..1 (0.5 = neutral).
# Reply: [u8 ack]. Additive: PROTOCOL_VERSION stays 3.
SET_OPTIC_GAIN = 53

# Accommodation — the mind's focus request in metres. The body
# converts it to the ciliary drive (1/m dioptres, clamped to 0..6 —
# infinity to a ~17 cm near point) and slews toward it on the
# advance clock; non-positive/non-finite distances relax to
# infinity. Monocular: no vergence component. Request:
# [f32 meters]. Reply: [u8 ack]. Additive: PROTOCOL_VERSION stays 3.
SET_FOCUS = 54

# Structural code analysis — the daemon's code sense. The mind sends a
# source file it already read ([u16 path_len][path utf-8]
# [u32 source_len][source utf-8]); the body parses it into a real Rust
# AST (syn), extracts semantic units, the call graph, impl relations,
# and complexity metrics, and returns `[u8 ack]` + a UTF-8 line report
# (tab-separated fields, `;`-separated lists — type strings may contain
# commas). A source that fails to parse returns INVALID_VALUE so the
# mind can fall back to its own lightweight parser. Stateless, pure.
ANALYZE_CODE = 55

# The cognitive mind's PID, pushed over IPC when a new mind re-inhabits
# a surviving body. GENESIS_COGNITIVE_PID names the launcher alive at
# spawn; after a re-inhabitation it still holds the dead launcher's
# PID, and interoception would measure a corpse.
# Request: [u32 pid]. Response: [u8 ack] (1).
# Additive: PROTOCOL_VERSION unaffected.
SET_COGNITIVE_PID = 56

# Voluntary eyelid — the orbicularis oculi the camera hardware lacks.
# The mind's requested lid openness composes with the sleep phase's
# own lid position by min: the will can shut an open eye (to rest it,
# to dim the world) but cannot open one the stage has closed.
# Request: [f32 openness] in 0..1 (1.0 = released to phase). Held,
# not momentary — lids stay where the mind left them.
# Reply: [u8 ack]. Additive: PROTOCOL_VERSION stays 3.
SET_LID = 57

# Deep recall — SimHash similarity scan over the archived tier of
# LTM. Sleep compression retires low-salience episodes to the
# archive, where they stay retrievable by ID but leave every
# associative index; this reaches them when the active set has no
# answer. Request and response are identical to FIND_SIMILAR.
# Additive: PROTOCOL_VERSION stays 3.
FIND_SIMILAR_ARCHIVED = 58

# Un-archive an episode — reconsolidation. Clears the archived flag,
# returns the episode to the active similarity index, re-bundles its
# SDR, and floors its salience at the reconsolidation level so the
# next sleep pass retains what recall actually used. Request/response
# identical to ARCHIVE_EPISODE. Additive: PROTOCOL_VERSION stays 3.
UNARCHIVE_EPISODE = 59

# Page through the archived tier — the cold store's table of
# contents. Request [u32 limit][u32 offset]; response is [u32 count]
# then per entry [u64 id][u64 ts][f32 salience][u8 flags]
# [u8 source_module][f32×4 emotional_tag] — metadata only, no
# payload text. Additive: PROTOCOL_VERSION stays 3.
SEARCH_ARCHIVED = 60

# Delete an episode for real. Marks the deleted flag and removes
# every index; the payload slot stays orphaned in the data file —
# reclaiming physical space is a rebuild pass, not this call.
# Request/response identical to ARCHIVE_EPISODE. Additive:
# PROTOCOL_VERSION stays 3.
DELETE_EPISODE = 61

# Which optional hardware sensors this machine actually has.
# Response: [u16 mask] — a bitmask of SENSOR_* below. A set bit means
# the matching BodyState field carries a real reading; a clear bit
# means there is no sensor and the value is a placeholder. Zero means
# nothing has been discovered yet, which is the safe direction.
#
# Additive: no BodyState layout changed, so PROTOCOL_VERSION stays 3.
# A daemon that predates this rejects it as an unknown command, which
# the client turns into an empty mask.
GET_SENSOR_PRESENCE = 36

# How many times Genesis changed task zone and emergent phase, and how
# long it spent in each.
# Response: [u8 zone_count] then per zone [u32 count][u64 total_ms],
#           then [u8 phase_count] then per phase the same 12-byte record.
# Indices are the CognitiveZone and MentalPhase discriminants. Counts
# are process-local and reset on restart.
# Additive: no BodyState layout changed.
GET_ZONE_TRANSITIONS = 37

# Which of Genesis's own processes are alive, and how many have died.
# Response: [u16 alive_mask][u16 deaths][u8 last_death]
# Bit n of alive_mask is set while subsystem n runs (0=daemon,
# 1=cognitive, 2=retina). A subsystem that was never running does not
# count as a death. last_death is 0xff when nothing has been lost.
# A subsystem that dies used to vanish from the telemetry with no
# trace; this makes the loss countable.
# Additive: no BodyState layout changed.
GET_PROCESS_HEALTH = 38

# ─── Tensor slab ──────────────────────────────────────────────
# Named multidimensional tensors shared between the daemon and the
# mind — world frames, salience fields, embeddings, motor fields.
# Wire layouts live in `genesis_client/tensor.py` and mirror
# `src/daemon/tensor_slab.rs`; the slab is process-local daemon state.
# All five commands are additive — no BodyState/CoreState layout
# changes — so PROTOCOL_VERSION stays 3.

PUT_TENSOR = 41
GET_TENSOR = 42
LIST_TENSORS = 43
DROP_TENSOR = 44
# Decode the published motor fields (`motor.spatial` +
# `motor.actions`) into a concrete action/grid target on the daemon's
# canonical 64×64 coordinate grid (`src/field/motor.rs`). Response is
# a 14-byte base plus an additive 16-byte distribution tail when the
# daemon reports field shape — see `tensor.unpack_motor_target`.
GET_MOTOR_TARGET = 45

SUBSYSTEM_DAEMON = 0
SUBSYSTEM_COGNITIVE = 1
SUBSYSTEM_RETINA = 2

SUBSYSTEM_NAMES: dict[int, str] = {
    SUBSYSTEM_DAEMON: "daemon",
    SUBSYSTEM_COGNITIVE: "cognitive",
    SUBSYSTEM_RETINA: "retina",
}

# ─── Sensor presence bits (must match
#     src/daemon/interoception.rs `pub mod sensor`) ─────────────
SENSOR_TEMPERATURE = 1 << 0
SENSOR_FREQUENCY = 1 << 1
SENSOR_MAX_FREQUENCY = 1 << 2
SENSOR_BATTERY = 1 << 3
SENSOR_AC_ADAPTER = 1 << 4
SENSOR_FAN = 1 << 5
SENSOR_POWER = 1 << 6
SENSOR_CORE_VOLTAGE = 1 << 7
SENSOR_SUPPLY_VOLTAGE = 1 << 8
SENSOR_RAPL = 1 << 9
SENSOR_PERF_COUNTERS = 1 << 10
SENSOR_GPE = 1 << 11
SENSOR_LOCAL_TIMER = 1 << 12
SENSOR_MEMORY = 1 << 13
SENSOR_PSI = 1 << 14
SENSOR_BATTERY_CYCLES = 1 << 15
SENSOR_CPUIDLE = 1 << 16

# Bit position -> channel name, for reporting. Order matches the Rust
# `SENSOR_BITS` table so both sides list absences in the same order.
SENSOR_NAMES: dict[int, str] = {
    SENSOR_TEMPERATURE: "temperature",
    SENSOR_FREQUENCY: "frequency",
    SENSOR_MAX_FREQUENCY: "max frequency",
    SENSOR_BATTERY: "battery",
    SENSOR_AC_ADAPTER: "AC adapter",
    SENSOR_FAN: "fan PWM",
    SENSOR_POWER: "power draw",
    SENSOR_CORE_VOLTAGE: "core voltage",
    SENSOR_SUPPLY_VOLTAGE: "supply voltage",
    SENSOR_RAPL: "RAPL power domains",
    SENSOR_PERF_COUNTERS: "perf counters",
    SENSOR_GPE: "EC GPE counter",
    SENSOR_LOCAL_TIMER: "local timer counters",
    SENSOR_MEMORY: "total memory",
    SENSOR_PSI: "pressure stall information",
    SENSOR_BATTERY_CYCLES: "battery cycle count",
    SENSOR_CPUIDLE: "cpuidle state counters",
}

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

# ─── Simulation-step bounds ───────────────────────────────────
# The daemon clamps ADVANCE_NEURO's dt to this range (ipc.rs,
# ADVANCE_NEURO handler). The cognitive layer must use the *same*
# ceiling for every wall-clock-driven simulation step, or two
# clocks measuring the same elapsed time disagree and drift apart
# without limit.
#
# This was a live bug: brain_waves.py clamped its own dt to 1.0 s
# while chemistry accepted 10.0 s, so any heartbeat slower than 1 s
# silently advanced chemistry by the full interval but the waves by
# at most 1 s. Past that point the wave clock fell behind the body
# clock by up to 90% and never recovered, because the loss was
# discarded rather than deferred.
#
# A step longer than this is a genuine stall (suspend, GC pause),
# not a frame boundary. Sub-stepping covers it below.
DT_MIN = 0.001
DT_MAX = 10.0

# The body's nominal integration step: TICK_INTERVAL_MS (200 ms) on the
# Rust side. Sent as the placeholder for the ADVANCE_NEURO dt field,
# which the daemon reads and ignores — it measures its own elapsed time.
# Mirrored here only so the wire value matches what the body is tuned
# for; it does not select the step.
DT_NOMINAL_STEP = 0.2

# ─── Notification IDs (server → client, unsolicited) ─────────
#
# Reserved / unwired: the IPC channel is strictly request-response,
# so the daemon never sends these today. The cognitive mind detects
# phase changes and dream insights by polling instead — see
# ``infrastructure/notifications.py`` (a pull-based
# queue). These
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
        raise ValueError(f"message length {total_len} exceeds maximum {MAX_MESSAGE_LEN}")
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
        raise ValueError(f"message length {total_len} exceeds maximum {MAX_MESSAGE_LEN}")
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
        raise ConnectionError("message length is zero (must include at least the command_id byte)")
    if total_len > MAX_MESSAGE_LEN:
        raise ConnectionError(f"message length {total_len} exceeds maximum {MAX_MESSAGE_LEN}")
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
