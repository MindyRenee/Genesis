//! IPC layer — Unix domain socket protocol for the Python cognitive
//! mind to communicate with the Rust subcognitive daemon.
//!
//! # Protocol design
//!
//! The protocol is **length-prefixed binary**, not JSON. This is
//! deliberate:
//!
//! - **Speed**: no parsing overhead. A request is a few bytes.
//! - **Efficiency**: no string allocation for simple commands.
//! - **Type safety**: the protocol is defined in Rust types, not
//!   a schema language.
//!
//! ## Message format
//!
//! Every message (request and response) has the same framing:
//!
//! ```text
//! [u32 LE: payload_len] [u8: command_id] [payload: payload_len-1 bytes]
//! ```
//!
//! The first 4 bytes are the total payload length (including the
//! command_id byte). The command_id determines how to interpret the
//! rest of the payload.
//!
//! ## Commands
//!
//! | ID | Command          | Direction | Payload                    |
//! |----|------------------|-----------|----------------------------|
//! |  1 | GetState         | Req       | (none)                     |
//! |    |                  | Resp      | GenesisCoreState (3288 B)  |
//! |  2 | GetNeuroSummary  | Req       | (none)                     |
//! |    |                  | Resp      | NeuroSummary (32 B)        |
//! |  3 | StoreEvent       | Req       | StoreEventReq              |
//! |    |                  | Resp      | u8 ack (1=ok, 0x80+=error) |
//! |  4 | RetrieveEpisode  | Req       | u64 episode_id             |
//! |    |                  | Resp      | EpisodePayload             |
//! |  5 | FindSimilar      | Req       | FindSimilarReq             |
//! |    |                  | Resp      | FindSimilarResp            |
//! |  6 | SetZone          | Req       | u8 zone                    |
//! |    |                  | Resp      | u8 ack (1=ok)              |
//! |  7 | GetPhase         | Req       | (none)                     |
//! |    |                  | Resp      | u8 phase + f32 arousal +   |
//! |    |                  |           |      f32 valence           |
//! |  8 | Ping             | Req       | (none)                     |
//! |    |                  | Resp      | u8 ack (1=ok) + u64 uptime |
//! |  9 | NeuroImpulse     | Req       | u8 chem_id + f32 magnitude |
//! |    |                  | Resp      | u8 ack (1=ok)              |
//! | 10 | GetMemoryStats   | Req       | (none)                     |
//! |    |                  | Resp      | MemoryStats (24 B)         |
//! | 11 | Sync             | Req       | (none)                     |
//! |    |                  | Resp      | u8 ack (1=ok)              |
//! | 12 | Shutdown         | Req       | (none)                     |
//! |    |                  | Resp      | u8 ack (1=ok)              |
//! | 13 | UpdateModuleStatus| Req      | u8 module_id + u8 status   |
//! |    |                  | Resp      | u8 ack (1=ok)              |
//! | 14 | GetRecentEpisodes| Req       | u8 limit + u8 source_filter|
//! |    |                  | Resp      | RecentEpisodesResp         |
//! | 16 | GetPlasticityProfile| Req    | (none)                     |
//! |    |                  | Resp      | PlasticityProfile (40 B)  |
//! | 23 | NeuroAdjustBaseline| Req    | u8 chem_id + f32 delta     |
//! |    |                  | Resp      | u8 ack (1=ok)              |
//!
//! ## Notifications (server → client, unsolicited)
//!
//! | ID | Notification      | Payload                        |
//! |----|-------------------|--------------------------------|
//! | 100| PhaseChanged      | u8 old_phase + u8 new_phase    |
//! | 101| EpisodeConsolidated| u64 episode_id                |
//! | 102| DreamInsight      | u64 episode_id + u32 chain_len |
//! | 103| ErrorDetected     | u32 error_code + u16 msg_len   |
//! |    |                   | + msg bytes                    |

use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

use crate::cognition::SelfModel;
use crate::state::{CognitiveZone, GenesisCoreState, ModuleId, NeurochemicalId};
use crate::store::{EventType, LtmStore, MmapState, RingBuffer};

// ─────────────────────────────────────────────────────────────────
//  Protocol constants
// ─────────────────────────────────────────────────────────────────

/// Protocol version. Increment when the wire format changes.
/// The client sends its version on connect; the daemon rejects
/// mismatches with an explicit error code.
///
/// v2: the body-control response gained a trailing `cpu_boost` byte
/// after the EPP string (and the EPP string itself was previously
/// missing from the Rust client's parser). Bumping the version makes
/// a stale client fail loudly at handshake instead of silently
/// misparsing the body-control payload.
pub const PROTOCOL_VERSION: u8 = 2;

/// Maximum total payload length we will accept in a single message.
/// The largest legitimate request is `StoreEvent` carrying up to
/// `MAX_PAYLOAD_SIZE` (256 KiB) of text plus its fixed header. 1 MiB
/// gives ample headroom while bounding the allocation a malicious or
/// buggy client can force the daemon to make before we read the body.
/// A length prefix above this is rejected as `PAYLOAD_TOO_LONG`.
pub const MAX_MESSAGE_LEN: usize = 1 << 20; // 1 MiB

/// Command opcodes sent by the cognitive mind to the daemon.
pub mod cmd {
    /// Request the full core state snapshot.
    pub const GET_STATE: u8 = 1;
    /// Request the compact neurochemical summary (`NeuroSummary`).
    pub const GET_NEURO_SUMMARY: u8 = 2;
    /// Store a pre-cognitive event into short-term memory.
    pub const STORE_EVENT: u8 = 3;
    /// Retrieve a single episode by ID.
    pub const RETRIEVE_EPISODE: u8 = 4;
    /// Find episodes similar to a query text (SimHash/SDR match).
    pub const FIND_SIMILAR: u8 = 5;
    /// Set the active task zone.
    pub const SET_ZONE: u8 = 6;
    /// Get the current emergent mental phase.
    pub const GET_PHASE: u8 = 7;
    /// Liveness check (round-trip a byte).
    pub const PING: u8 = 8;
    /// Apply a transient neurochemical impulse.
    pub const NEURO_IMPULSE: u8 = 9;
    /// Request short/long-term memory statistics (`MemoryStats`).
    pub const GET_MEMORY_STATS: u8 = 10;
    /// Flush LTM pages to disk (`msync`).
    pub const SYNC: u8 = 11;
    /// Request graceful daemon shutdown.
    pub const SHUTDOWN: u8 = 12;
    /// Update a module's runtime manifest status.
    pub const UPDATE_MODULE_STATUS: u8 = 13;
    /// Request the most recent episodes.
    pub const GET_RECENT_EPISODES: u8 = 14;
    /// Handshake: client sends [u8 client_version], daemon responds
    /// [u8 daemon_version]. If versions differ, the daemon closes
    /// the connection after responding.
    pub const HANDSHAKE: u8 = 15;
    /// Get the plasticity profile — a compact summary of the
    /// metaplastic state (coupling-matrix drift, receptor
    /// sensitivities, BDNF/cortisol levels, plasticity gate) that
    /// the cognitive mind uses to adapt its learning strategy.
    /// Response: PlasticityProfile (40 bytes).
    pub const GET_PLASTICITY_PROFILE: u8 = 16;
    /// Get the interoceptive body state — CPU temperature, frequency,
    /// memory pressure, load, I/O activity, battery, thermal throttling,
    /// cognitive load, and distress level. This is how the cognitive
    /// mind feels her own body. Response: BodyState (64 bytes).
    pub const GET_BODY_STATE: u8 = 17;
    /// Get the body control state — what Genesis is doing to her body
    /// (CPU frequency policy, scheduling priorities, I/O priority,
    /// thermal cap status). This is how the cognitive mind knows what
    /// her neurochemistry is driving. Response: BodyControlState.
    pub const GET_BODY_CONTROL: u8 = 18;
    /// Get the active inference summary — the generative self-model's
    /// projection (surprise, free energy, allostatic load, precision,
    /// dyadic attunement/synchrony, user affect, prediction errors).
    /// Response: InferenceSignals (60 bytes).
    pub const GET_INFERENCE_SUMMARY: u8 = 19;
    /// Update the user affect observation — the cognitive mind's
    /// inference of the user's affective state from conversation
    /// features. This feeds the dyadic affective model. Request:
    /// [f32 valence][f32 arousal][f32 engagement][f32 confidence].
    /// Response: u8 ack (1=ok).
    pub const UPDATE_USER_AFFECT: u8 = 20;
    /// Search episodes by offset — page through all active episodes
    /// for sleep compression compaction. Request:
    /// [u32 limit][u32 offset]. Response: same format as
    /// GET_RECENT_EPISODES.
    pub const SEARCH_EPISODES: u8 = 22;
    /// Adjust a neurochemical's baseline (set-point) directly. Unlike
    /// NeuroImpulse, which only affects the current level and decays
    /// back to baseline, this permanently shifts the baseline. Used by
    /// the startup wake cascade to clear residual adenosine sleep
    /// pressure from a previous session — biologically justified as
    /// the glymphatic system having already cleared adenosine during
    /// the offline period (restart = sleep already happened).
    /// Request: [u8 chem_id][f32 delta]. Response: u8 ack (1=ok).
    pub const NEURO_ADJUST_BASELINE: u8 = 23;

    // ─── Reactive commands (mind-driven, not tick-driven) ───────
    // The daemon no longer runs a fixed tick loop. Instead, the
    // cognitive mind drives each function through IPC commands when
    // internal state (brain waves, thresholds, urges) says it's time.
    // The daemon is a bus — it carries information and executes
    // requests, but never initiates actions on its own.

    /// Advance neurochemical dynamics by dt, then run active inference
    /// and dyadic model update. The mind sends this when brain waves
    /// say it's time to advance. Request: [f32 dt]. Response:
    /// [u8 ack][f32 surprise][f32 free_energy][f32 precision]
    /// [f32 allostatic_load][u32 tick_count] = 21 bytes.
    pub const ADVANCE_NEURO: u8 = 24;
    /// Consolidate STM → LTM. The mind sends this when STM fill
    /// crosses a threshold weighted by consolidation_weight.
    /// Response: [u8 ack][u32 consolidated] = 5 bytes.
    pub const CONSOLIDATE: u8 = 25;
    /// Find associations for recent episodes. The mind sends this when
    /// new episodes have been stored. Response: [u8 ack][u32 associations]
    /// = 5 bytes.
    pub const ASSOCIATE: u8 = 26;
    /// Run one dream cycle (only meaningful when sleeping). The mind
    /// sends this when sleep state + dream pressure cross a threshold.
    /// Response: [u8 ack][u32 dream_insights] = 5 bytes.
    pub const DREAM: u8 = 27;
    /// Read hardware sensors and apply interoception impulses to
    /// neurochemistry. The mind sends this when it wants to feel its
    /// body. Response: [u8 ack] + BodyState (same as GET_BODY_STATE).
    pub const READ_SENSORS: u8 = 28;
    /// Read neurochemistry and apply CPU frequency, scheduling
    /// priority, and I/O priority. The mind sends this when
    /// neurochemistry has changed enough to warrant a different policy.
    /// Response: [u8 ack] + BodyControlState (same as GET_BODY_CONTROL).
    pub const APPLY_BODY_CONTROL: u8 = 29;
    /// Save the active inference model to disk. The mind sends this
    /// periodically or before shutdown. Response: [u8 ack].
    pub const SAVE_INFERENCE: u8 = 30;
    /// Archive an episode from LTM. Used by sleep compression to
    /// compact low-salience episodes after extracting their semantic
    /// content. Unlike a hard delete, the archived episode remains
    /// retrievable by ID — it is only removed from the active
    /// similarity index. Request: [u64 episode_id]. Response: u8 ack
    /// (1=ok, 0=not found).
    pub const ARCHIVE_EPISODE: u8 = 31;
    /// Store an episode directly in LTM, bypassing STM. Used by the
    /// cognitive mind for deliberate memory stores (conversation turns,
    /// learning events) where the real LTM episode ID is needed
    /// immediately for Python-side tracking (MemoryRecord, attractor
    /// network, RIF). Raw pre-cognitive events should still use
    /// STORE_EVENT (STM path); STORE_EPISODE is for memories the
    /// cognitive mind has already decided are worth keeping.
    /// Request:  [u64 timestamp][u8 event_type][u8 source_module]
    ///           [f32 salience][f32×12 emotional_tag]
    ///           [u16 text_len][text bytes]
    /// Response: [u8 ack (1=ok, 0=failed)][u64 episode_id] (on ack=1)
    pub const STORE_EPISODE: u8 = 32;
    /// Get per-subsystem silicon telemetry — which part of the mind's
    /// process tree is firing. Each subsystem (daemon, cognitive,
    /// retina) reports its own CPU share, I/O share, and
    /// microarchitectural prediction-error ratios (cache/branch
    /// misses). The cognitive mind correlates this with her task
    /// zone to feel *where* her activity lives. Response:
    /// [u8 count] then per subsystem:
    ///   [u8 subsystem][u32 pid][f32 cpu][f32 io]
    ///   [f32 cache_miss_rate][f32 branch_miss_rate] = 21 bytes each.
    pub const GET_SUBSYSTEM_TELEMETRY: u8 = 33;
}

/// Notification opcodes for daemon→cognitive push messages.
///
/// **Reserved / unwired.** The IPC channel is strictly request-response:
/// the daemon never writes an unsolicited message to a client, so none
/// of these are ever sent today. The cognitive mind instead detects
/// these events by polling (`python/genesis_cognitive/notifications.py`,
/// a pull-based queue), which is the active mechanism. These constants
/// reserve the 100+ opcode range for a future push implementation and
/// mirror the Python `NOTIFY_*` constants in
/// `python/genesis_client/protocol.py`; do not remove them without
/// removing that mirror and the accompanying documentation.
pub mod notify {
    /// The emergent mental phase changed.
    pub const PHASE_CHANGED: u8 = 100;
    /// An episode was consolidated from STM to LTM.
    pub const EPISODE_CONSOLIDATED: u8 = 101;
    /// A dream insight was generated during sleep.
    pub const DREAM_INSIGHT: u8 = 102;
    /// An internal error was detected.
    pub const ERROR_DETECTED: u8 = 103;
}

/// Error codes returned as the first byte of a response when a
/// request is malformed or rejected.
///
/// **Convention**: success acks use `1` (not `0`) for historical
/// reasons — the Python client checks `resp[0] == 1`. Error codes
/// are therefore renumbered to `0x80+` so they can never collide
/// with the success byte or any valid data response first byte.
/// A client can unambiguously distinguish success (`1`) from error
/// (`>= 0x80`) by checking the high bit.
pub mod error {
    /// Success (no error).
    pub const OK: u8 = 0;
    /// The request payload was shorter than required.
    pub const PAYLOAD_TOO_SHORT: u8 = 0x80;
    /// The request payload exceeded `MAX_MESSAGE_LEN`.
    pub const PAYLOAD_TOO_LONG: u8 = 0x81;
    /// The command opcode was not recognized.
    pub const UNKNOWN_COMMAND: u8 = 0x82;
    /// The client's protocol version does not match the daemon's.
    pub const VERSION_MISMATCH: u8 = 0x83;
    /// An unexpected internal error occurred.
    pub const INTERNAL_ERROR: u8 = 0x84;
    /// A request field held an invalid value.
    pub const INVALID_VALUE: u8 = 0x85;
    /// Consistent read failed after all retries (seqlock contention
    /// or corrupt state). Returned instead of zero-filled data so
    /// callers can distinguish failure from a valid zero state.
    pub const READ_FAILED: u8 = 0x86;
}

// ─────────────────────────────────────────────────────────────────
//  Wire types
// ─────────────────────────────────────────────────────────────────

/// Compact neurochemical summary (32 bytes) — sent in response to
/// GetNeuroSummary. This is what the Python side reads to know how
/// Genesis feels without transferring the full 3288-byte state.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct NeuroSummary {
    /// Arousal axis: high = alert, low = drowsy.
    pub arousal: f32,
    /// Valence axis: positive = pleasant, negative = unpleasant.
    pub valence: f32,
    /// Global neurochemical tone (overall activation level).
    pub global_tone: f32,
    /// Plasticity gate: how capable the system is of forming new memories.
    pub plasticity_gate: f32,
    /// Memory encoding weight derived from neurochemistry.
    pub encoding_weight: f32,
    /// Memory consolidation weight derived from neurochemistry.
    pub consolidation_weight: f32,
    /// Memory retrieval weight derived from neurochemistry.
    pub retrieval_weight: f32,
    /// Current emergent mental phase.
    pub phase: u8,
    /// Alignment padding.
    pub _pad: [u8; 3],
}

const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<NeuroSummary>() == 32);
    assert!(offset_of!(NeuroSummary, arousal) == 0);
    assert!(offset_of!(NeuroSummary, valence) == 4);
    assert!(offset_of!(NeuroSummary, global_tone) == 8);
    assert!(offset_of!(NeuroSummary, plasticity_gate) == 12);
    assert!(offset_of!(NeuroSummary, encoding_weight) == 16);
    assert!(offset_of!(NeuroSummary, consolidation_weight) == 20);
    assert!(offset_of!(NeuroSummary, retrieval_weight) == 24);
    assert!(offset_of!(NeuroSummary, phase) == 28);
};

/// Memory statistics (24 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct MemoryStats {
    /// Number of episodes currently in short-term memory.
    pub stm_count: u64,
    /// Number of episodes in long-term memory.
    pub ltm_count: u64,
    /// Legacy LTM capacity (always `u32::MAX` in v2).
    pub ltm_capacity: u32,
    /// Alignment padding.
    pub _pad: u32,
}

const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<MemoryStats>() == 24);
    assert!(offset_of!(MemoryStats, stm_count) == 0);
    assert!(offset_of!(MemoryStats, ltm_count) == 8);
    assert!(offset_of!(MemoryStats, ltm_capacity) == 16);
};

/// Plasticity profile (40 bytes) — a compact summary of the
/// metaplastic state, sent in response to GetPlasticityProfile.
///
/// This exposes the substrate's "learning-to-learn" state so the
/// cognitive mind can adapt its learning strategy. The coupling
/// matrix self-modifies under sustained emotion (metaplasticity);
/// this struct makes that drift observable without transferring the
/// full 1296-byte matrix.
///
/// ## Fields
///
/// | Field | Meaning |
/// |-------|---------|
/// | `plasticity_gate` | BDNF/cortisol-derived gate [0,1]. High = capable of new memory formation. |
/// | `bdnf_effective` | Effective BDNF level (level × receptor sensitivity). Drives plasticity. |
/// | `bdnf_tonic` | Sustained (tonic) BDNF. Slow signal that drives metaplasticity. |
/// | `cortisol_effective` | Effective cortisol. Acute stress. |
/// | `cortisol_tonic` | Sustained cortisol. Chronic stress signal — suppresses BDNF, blocks metaplasticity. |
/// | `dopamine_effective` | Effective dopamine. Reward/novelty. |
/// | `serotonin_effective` | Effective serotonin. Mood stability; enables BDNF recovery. |
/// | `coupling_drift` | Mean absolute deviation of the coupling matrix from the default. 0 = no metaplastic change. Grows under sustained stress or wellbeing. |
/// | `mean_receptor_sensitivity` | Mean receptor sensitivity across all 18 chemicals. <1 = downregulation (tolerance); >1 = upregulation (sensitization). |
/// | `emergent_phase` | Current emergent mental phase (matches NeuroSummary.phase). |
/// | `_pad` | Alignment padding. |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PlasticityProfile {
    /// BDNF/cortisol-derived gate. High = capable of new memory formation.
    pub plasticity_gate: f32,
    /// Effective BDNF level (level × receptor sensitivity). Drives plasticity.
    pub bdnf_effective: f32,
    /// Sustained (tonic) BDNF. Slow signal that drives metaplasticity.
    pub bdnf_tonic: f32,
    /// Effective cortisol. Acute stress.
    pub cortisol_effective: f32,
    /// Sustained cortisol. Chronic stress signal — suppresses BDNF, blocks metaplasticity.
    pub cortisol_tonic: f32,
    /// Effective dopamine. Reward/novelty.
    pub dopamine_effective: f32,
    /// Effective serotonin. Mood stability; enables BDNF recovery.
    pub serotonin_effective: f32,
    /// Mean absolute deviation of the coupling matrix from the default. 0 = no metaplastic change.
    pub coupling_drift: f32,
    /// Mean receptor sensitivity across all 18 chemicals. <1 = downregulation; >1 = upregulation.
    pub mean_receptor_sensitivity: f32,
    /// Current emergent mental phase (matches `NeuroSummary.phase`).
    pub emergent_phase: u8,
    /// Alignment padding.
    pub _pad: [u8; 3],
}

const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<PlasticityProfile>() == 40);
    assert!(offset_of!(PlasticityProfile, plasticity_gate) == 0);
    assert!(offset_of!(PlasticityProfile, bdnf_effective) == 4);
    assert!(offset_of!(PlasticityProfile, bdnf_tonic) == 8);
    assert!(offset_of!(PlasticityProfile, cortisol_effective) == 12);
    assert!(offset_of!(PlasticityProfile, cortisol_tonic) == 16);
    assert!(offset_of!(PlasticityProfile, dopamine_effective) == 20);
    assert!(offset_of!(PlasticityProfile, serotonin_effective) == 24);
    assert!(offset_of!(PlasticityProfile, coupling_drift) == 28);
    assert!(offset_of!(PlasticityProfile, mean_receptor_sensitivity) == 32);
    assert!(offset_of!(PlasticityProfile, emergent_phase) == 36);
};

/// Interoceptive body state — what Genesis feels about her machine.
///
/// This is a plain data struct (not `#[repr(C)]`) because it contains
/// a `String`. The IPC response is parsed field-by-field rather than
/// memcpy'd.
#[derive(Clone, Debug, Default)]
pub struct BodyStateSummary {
    /// CPU temperature in degrees Celsius (raw).
    pub cpu_temp_c: f32,
    /// Body temperature normalized [0,1], 0.5 = normal (50°C).
    pub temperature: f32,
    /// CPU frequency as fraction of max — her arousal level.
    pub arousal_freq: f32,
    /// Memory pressure [0,1] — cognitive load.
    pub cognitive_load: f32,
    /// I/O throughput fraction [0,1] — data exchange activity.
    /// This is active data processing (reading from and writing to
    /// external storage), not drowsiness.
    pub io_activity: f32,
    /// Self-process CPU usage as a fraction of CPU capacity [0,2].
    /// 0.0 = idle, 1.0 = all cores busy, >1.0 = overloaded. Values
    /// above 1.0 indicate the process tree is using more than its
    /// fair share (multiple threads on all cores). The interoceptor
    /// clamps to [0,2]; the distress threshold is >1.5.
    pub stress_load: f32,
    /// Battery level [0,1], 1.0 = full or on AC.
    pub energy_reserve: f32,
    /// On AC power (true) or battery (false).
    pub on_ac_power: bool,
    /// Number of CPU cores.
    pub num_cores: u32,
    /// Whether any distress condition is active.
    pub distressed: bool,
    /// Autonomic afferent signal rate — GPE interrupts/sec from
    /// the Embedded Controller. The EC is the machine's autonomic
    /// nervous system (a subordinate processor), not a heart.
    /// 0.0 = no EC detected. Informational only; regularity is
    /// normal and does not produce stress.
    pub autonomic_rate: f32,
    /// Thermoregulatory effort — fan speed normalized [0,1].
    /// How hard the cooling system is working to remove waste
    /// heat. This is thermoregulation, not cardiac output.
    pub thermoregulatory_effort: f32,
    /// Metabolic rate — power draw normalized [0,1]. How much
    /// energy the machine is consuming right now. 0.0 if no
    /// power sensor is available.
    pub metabolic_rate: f32,
    /// Core voltage (Vcore) in volts. Raw CPU electrical state,
    /// clamped to [0, 25]. 0.0 if no sensor is available.
    pub core_voltage: f32,
    /// Supply voltage (battery rail) in volts, clamped to [0, 25].
    /// The cognitive mind reads this for the energy-reserve health
    /// signal (a CRH impulse when critically low). 0.0 if no
    /// battery sensor is available.
    pub supply_voltage: f32,
    /// RAPL core-domain switching rate [0,1] — execution-unit
    /// firing at the electron level. 0.0 if no powercap driver.
    pub core_activity: f32,
    /// RAPL uncore-domain switching rate [0,1] — integration
    /// fabric (cache/memory-controller) activity.
    pub uncore_activity: f32,
    /// RAPL DRAM-domain switching rate [0,1] — memory-subsystem
    /// firing (encoding/retrieval traffic).
    pub dram_activity: f32,
    /// Cache miss ratio of her process tree [0,1] from perf
    /// counters — microarchitectural prediction errors. 0.0 if
    /// perf is unavailable (perf_event_paranoid ≥ 4).
    pub cache_miss_rate: f32,
    /// Branch misprediction ratio of her process tree [0,1] —
    /// the hardware branch predictor guessing wrong while
    /// running her.
    pub branch_miss_rate: f32,
    /// Human-readable first-person description.
    pub description: String,
}

/// Per-subsystem silicon telemetry — one process's activity signals.
/// `subsystem` is the wire tag: 0 = daemon, 1 = cognitive, 2 = retina.
#[derive(Clone, Debug, Default)]
pub struct SubsystemTelemetrySummary {
    /// Subsystem tag (0=daemon, 1=cognitive, 2=retina).
    pub subsystem: u8,
    /// Process ID.
    pub pid: u32,
    /// This process's CPU share [0,2] — its slice of stress_load.
    pub cpu: f32,
    /// This process's I/O share [0,1] — its slice of io_activity.
    pub io: f32,
    /// This process's cache-miss ratio [0,1].
    pub cache_miss_rate: f32,
    /// This process's branch-miss ratio [0,1].
    pub branch_miss_rate: f32,
}

/// Per-module telemetry — one brain part's self-reported activity.
#[derive(Clone, Debug, Default)]
pub struct ModuleTelemetrySummary {
    /// ModuleId discriminant (0=Subcognitive … 11=Guardrails).
    pub module_id: u8,
    /// ModuleStatus discriminant (0=Stopped … 5=Error).
    pub status: u8,
    /// Self-reported activity share of the cognitive process [0,1].
    pub cpu_share: f32,
}

/// Combined response of GET_SUBSYSTEM_TELEMETRY: process-level subsystems
/// (hardware-measured) plus module-level brain parts
/// (self-reported by the cognitive mind).
#[derive(Clone, Debug, Default)]
pub struct SubsystemTelemetryReport {
    /// Per-process subsystem telemetry.
    pub subsystems: Vec<SubsystemTelemetrySummary>,
    /// Per-module brain-part telemetry.
    pub modules: Vec<ModuleTelemetrySummary>,
}

/// Body control state summary — what Genesis is doing to her body.
///
/// This is the mirror of `BodyStateSummary` (interoception). While
/// `BodyStateSummary` is what she *feels*, `BodyControlSummary` is
/// what she's *doing* — the actions her neurochemistry has driven.
#[derive(Clone, Debug, Default)]
pub struct BodyControlSummary {
    /// CPU frequency floor (kHz).
    pub cpu_min_freq_khz: u32,
    /// CPU frequency ceiling (kHz).
    pub cpu_max_freq_khz: u32,
    /// CPU governor ("schedutil", "powersave", etc.).
    pub cpu_governor: String,
    /// Whether thermal cap is active.
    pub thermally_capped: bool,
    /// Temperature that triggered the cap (0 if no cap).
    pub thermal_cap_temp_c: f32,
    /// Daemon nice value.
    pub daemon_nice: i32,
    /// Cognitive mind nice value.
    pub cognitive_nice: i32,
    /// I/O scheduling class label.
    pub io_class: String,
    /// Plasticity gate [0,1] that drove I/O priority.
    pub plasticity_gate: f32,
    /// Whether she's controlling the cognitive mind's PID.
    pub controlling_cognitive: bool,
    /// Energy Performance Preference (EPP) profile she set — the
    /// hardware's voltage/frequency operating-envelope hint. Empty
    /// string when EPP is not supported (e.g. acpi-cpufreq).
    pub cpu_epp: String,
    /// Turbo (boost) gate state — the hardware permission to run
    /// above the base P-state. `Unavailable` when the platform
    /// exposes no gate.
    pub cpu_boost: super::cpufreq::BoostState,
    /// Human-readable first-person description.
    pub description: String,
}

/// Serialize a body-control state to the wire layout shared by the
/// GET_BODY_CONTROL and APPLY_BODY_CONTROL responses.
///
/// Layout (little-endian):
///   [u32 cpu_min_freq_khz][u32 cpu_max_freq_khz]
///   [u32 gov_len][gov bytes]
///   [u8 thermally_capped][f32 thermal_cap_temp_c]
///   [i32 daemon_nice][i32 cognitive_nice]
///   [u32 io_class_len][io_class bytes]
///   [f32 plasticity_gate][u8 controlling_cognitive]
///   [u32 epp_len][epp bytes]
///   [u8 cpu_boost]
///   [u32 desc_len][desc bytes]
///
/// APPLY_BODY_CONTROL prepends a one-byte ack. Keeping the layout in a
/// single function prevents the two handlers from drifting — the
/// client's `get_body_control` parser silently fell out of sync with
/// the daemon once already (the EPP string was omitted).
fn serialize_body_control(ctrl: &super::cpufreq::BodyControlState) -> Vec<u8> {
    let gov_bytes = ctrl.cpu_governor.as_bytes();
    let io_bytes = ctrl.io_class.as_bytes();
    let epp_bytes = ctrl.cpu_epp.as_bytes();
    let desc_bytes = ctrl.description.as_bytes();
    let mut resp = Vec::with_capacity(
        8 + 4
            + gov_bytes.len()
            + 1
            + 4
            + 8
            + 4
            + io_bytes.len()
            + 4
            + 1
            + 4
            + epp_bytes.len()
            + 1
            + 4
            + desc_bytes.len(),
    );
    resp.extend_from_slice(&ctrl.cpu_min_freq_khz.to_le_bytes());
    resp.extend_from_slice(&ctrl.cpu_max_freq_khz.to_le_bytes());
    resp.extend_from_slice(&(gov_bytes.len() as u32).to_le_bytes());
    resp.extend_from_slice(gov_bytes);
    resp.push(ctrl.thermally_capped as u8);
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(ctrl.thermal_cap_temp_c, 0.0, 150.0).to_le_bytes(),
    );
    resp.extend_from_slice(&ctrl.daemon_nice.to_le_bytes());
    resp.extend_from_slice(&ctrl.cognitive_nice.to_le_bytes());
    resp.extend_from_slice(&(io_bytes.len() as u32).to_le_bytes());
    resp.extend_from_slice(io_bytes);
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(ctrl.plasticity_gate, 0.0, 1.0).to_le_bytes(),
    );
    resp.push(ctrl.controlling_cognitive as u8);
    resp.extend_from_slice(&(epp_bytes.len() as u32).to_le_bytes());
    resp.extend_from_slice(epp_bytes);
    resp.push(ctrl.cpu_boost.to_wire());
    resp.extend_from_slice(&(desc_bytes.len() as u32).to_le_bytes());
    resp.extend_from_slice(desc_bytes);
    resp
}

/// Serialize an interoceptive `BodyState` for the wire.
///
/// Layout (little-endian):
///   [f32 cpu_temp_c][f32 temperature][f32 arousal_freq]
///   [f32 cognitive_load][f32 io_activity][f32 stress_load]
///   [f32 energy_reserve][u8 on_ac_power][u32 num_cores]
///   [u8 distressed][f32 autonomic_rate]
///   [f32 thermoregulatory_effort][f32 metabolic_rate]
///   [f32 core_voltage][f32 supply_voltage]
///   [f32 core_activity][f32 uncore_activity][f32 dram_activity]
///   [f32 cache_miss_rate][f32 branch_miss_rate]
///   [u32 desc_len][desc bytes]
///
/// Fixed header = 7×f32 (28) + 1 + 4 + 1 + 10×f32 (40) + 4 = 78 bytes.
///
/// GET_BODY_STATE and READ_SENSORS both return this layout (the latter
/// prefixed with a one-byte ack), so the serialization lives in one
/// place. The two previously diverged — READ_SENSORS silently omitted
/// `core_voltage`/`supply_voltage`, which shifted the description parse
/// and fed garbage voltages to the cognitive mind's interoception.
fn serialize_body_state(body: &super::interoception::BodyState) -> Vec<u8> {
    let desc_bytes = body.description.as_bytes();
    let mut resp = Vec::with_capacity(78 + desc_bytes.len());
    // Sanitize outgoing f32 fields — a NaN in the shared body state
    // would propagate to the Python client and corrupt its affect
    // inference.
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.cpu_temp_c, 0.0, 150.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.temperature, 0.0, 150.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.arousal_freq, 0.0, 1.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.cognitive_load, 0.0, 1.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.io_activity, 0.0, 1.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.stress_load, 0.0, 2.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.energy_reserve, 0.0, 1.0).to_le_bytes(),
    );
    resp.push(body.on_ac_power as u8);
    resp.extend_from_slice(&body.num_cores.to_le_bytes());
    resp.push(body.distressed as u8);
    // Machine-native body fields — autonomic monitoring,
    // thermoregulatory effort, and metabolic rate.
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.autonomic_rate, 0.0, 100.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.thermoregulatory_effort, 0.0, 1.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.metabolic_rate, 0.0, 1.0).to_le_bytes(),
    );
    // Voltage fields — core voltage (Vcore) and supply voltage
    // (battery rail). Both are raw volts, clamped to [0, 25].
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.core_voltage, 0.0, 25.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.supply_voltage, 0.0, 25.0).to_le_bytes(),
    );
    // Silicon switching fields — RAPL per-domain energy rates
    // (normalized [0, 1]) and perf-counter miss ratios. 0.0 on
    // systems without powercap or perf access.
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.core_activity, 0.0, 1.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.uncore_activity, 0.0, 1.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.dram_activity, 0.0, 1.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.cache_miss_rate, 0.0, 1.0).to_le_bytes(),
    );
    resp.extend_from_slice(
        &crate::state::sanitize::finite_clamp(body.branch_miss_rate, 0.0, 1.0).to_le_bytes(),
    );
    resp.extend_from_slice(&(desc_bytes.len() as u32).to_le_bytes());
    resp.extend_from_slice(desc_bytes);
    resp
}

/// Serialize per-subsystem silicon telemetry for GET_SUBSYSTEM_TELEMETRY.
///
/// Layout (little-endian):
///   [u8 count] then per subsystem:
///   [u8 subsystem][u32 pid][f32 cpu][f32 io]
///   [f32 cache_miss_rate][f32 branch_miss_rate]
///   then a module section — the mind's self-reported anatomy:
///   [u8 module_count] then per module:
///   [u8 module_id][u8 status][f32 cpu_share]
///
/// 1 + count × 21 + 1 + module_count × 6 bytes.
/// An empty report serializes as [0][0].
fn serialize_subsystem_telemetry(
    subsystems: &[super::interoception::SubsystemTelemetry],
    modules: &[(u8, u8, f32)],
) -> Vec<u8> {
    let mut resp = Vec::with_capacity(2 + subsystems.len() * 21 + modules.len() * 6);
    resp.push(subsystems.len().min(u8::MAX as usize) as u8);
    for subsystem in subsystems.iter().take(u8::MAX as usize) {
        resp.push(subsystem.subsystem.to_wire());
        resp.extend_from_slice(&subsystem.pid.to_le_bytes());
        resp.extend_from_slice(
            &crate::state::sanitize::finite_clamp(subsystem.cpu, 0.0, 2.0).to_le_bytes(),
        );
        resp.extend_from_slice(
            &crate::state::sanitize::finite_clamp(subsystem.io, 0.0, 1.0).to_le_bytes(),
        );
        resp.extend_from_slice(
            &crate::state::sanitize::finite_clamp(subsystem.cache_miss_rate, 0.0, 1.0).to_le_bytes(),
        );
        resp.extend_from_slice(
            &crate::state::sanitize::finite_clamp(subsystem.branch_miss_rate, 0.0, 1.0).to_le_bytes(),
        );
    }
    resp.push(modules.len().min(u8::MAX as usize) as u8);
    for (module_id, status, cpu_share) in modules.iter().take(u8::MAX as usize) {
        resp.push(*module_id);
        resp.push(*status);
        resp.extend_from_slice(
            &crate::state::sanitize::finite_clamp(*cpu_share, 0.0, 1.0).to_le_bytes(),
        );
    }
    resp
}

// ─────────────────────────────────────────────────────────────────
//  Message framing
// ─────────────────────────────────────────────────────────────────

/// Read a complete message from a stream.
///
/// Returns (command_id, payload) where payload excludes the command_id.
pub fn read_message(stream: &mut UnixStream) -> std::io::Result<(u8, Vec<u8>)> {
    // Read 4-byte length prefix
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let total_len = u32::from_le_bytes(len_buf) as usize;

    if total_len == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "message length is zero (must include at least the command_id byte)",
        ));
    }

    // Bound the allocation before reading. Without this, a u32 length
    // prefix of up to ~4 GiB would force the daemon to attempt a
    // multi-gigabyte allocation (and a long blocking read) on any
    // malformed or malicious message — a trivial denial of service
    // against the single-threaded IPC server.
    if total_len > MAX_MESSAGE_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("message length {total_len} exceeds maximum {MAX_MESSAGE_LEN}"),
        ));
    }

    // Read the payload (includes command_id as first byte)
    let mut payload = vec![0u8; total_len];
    stream.read_exact(&mut payload)?;

    // Split the command byte from the payload without an extra
    // allocation+copy. `payload.split_first` borrows; we need owned
    // data, so drain the first element instead. This reuses the
    // original Vec's allocation for the data — one alloc, not two.
    let cmd_id = payload.remove(0);
    Ok((cmd_id, payload))
}

/// Write a message to a stream.
///
/// Enforces the same `MAX_MESSAGE_LEN` bound as `read_message`: a
/// response that exceeds 1 MiB would be sent successfully by the daemon
/// but rejected by the client's `read_message`, causing a silent
/// protocol desync. The guard also prevents the `as u32` cast from
/// truncating on 64-bit targets if `payload.len()` exceeds `u32::MAX`.
pub fn write_message(stream: &mut UnixStream, cmd_id: u8, payload: &[u8]) -> std::io::Result<()> {
    let total_len = 1 + payload.len();
    if total_len > MAX_MESSAGE_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("response length {total_len} exceeds maximum {MAX_MESSAGE_LEN}"),
        ));
    }
    let total_len = total_len as u32;
    stream.write_all(&total_len.to_le_bytes())?;
    stream.write_all(&[cmd_id])?;
    stream.write_all(payload)?;
    stream.flush()
}

// ─────────────────────────────────────────────────────────────────
//  LTM access abstraction
// ─────────────────────────────────────────────────────────────────

/// A trait for obtaining mutable access to an `LtmStore`.
///
/// This abstraction lets the `IpcServer` work with both:
/// - **Shared ownership** (daemon): `Arc<Mutex<LtmStore>>` implements
///   this trait, locking the mutex and returning a `MutexGuard`.
///
/// Without this, the daemon binary had to reimplement the entire IPC
/// accept loop and connection handler — a 100-line duplication that
/// could drift from the library version.
pub trait LtmAccess {
    /// The borrow type returned by [`access`](Self::access), which
    /// yields mutable access to the underlying [`LtmStore`].
    type Guard<'a>: std::ops::DerefMut<Target = LtmStore>
    where
        Self: 'a;

    /// Acquire mutable access to the underlying [`LtmStore`].
    fn access(&mut self) -> Self::Guard<'_>;
}

impl LtmAccess for std::sync::Arc<std::sync::Mutex<LtmStore>> {
    type Guard<'a>
        = std::sync::MutexGuard<'a, LtmStore>
    where
        Self: 'a;

    fn access(&mut self) -> Self::Guard<'_> {
        // A poisoned LTM mutex means a prior writer panicked mid-write.
        // Recovering the inner data is safer than crashing the daemon
        // (which would kill every IPC connection thread). The data may
        // be inconsistent, but not undefined behavior — `LtmStore`
        // contains only owned Rust types (no raw pointers in the
        // metadata structures).
        self.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl LtmAccess for LtmStore {
    type Guard<'a>
        = &'a mut LtmStore
    where
        Self: 'a;

    fn access(&mut self) -> Self::Guard<'_> {
        self
    }
}

// ─────────────────────────────────────────────────────────────────
//  IPC server
// ─────────────────────────────────────────────────────────────────

/// The IPC server. Runs in a separate thread, accepting connections
/// and handling requests.
pub struct IpcServer {
    socket_path: PathBuf,
    shutdown_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl IpcServer {
    /// Create a new IPC server at the given socket path.
    pub fn new(socket_path: impl AsRef<Path>) -> Self {
        Self {
            socket_path: socket_path.as_ref().to_path_buf(),
            shutdown_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Get a clone of the shutdown flag (for testing).
    pub fn shutdown_flag(&self) -> std::sync::Arc<std::sync::atomic::AtomicBool> {
        self.shutdown_flag.clone()
    }

    /// Check if the server has been shut down.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown_flag
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Start the server (static method, owns the server).
    /// This blocks the calling thread — run it in a dedicated thread.
    ///
    /// The `ltm` parameter is generic over `LtmAccess`, so callers
    /// pass an `Arc<Mutex<LtmStore>>` (daemon and tests) without
    /// reimplementing the server loop.
    ///
    /// `mmap` and `stm` are wrapped in `Arc` because they cannot be
    /// cloned (they own raw mmap pointers and file descriptors). The
    /// daemon shares them between the reactive handler and IPC thread via
    /// `Arc`; tests wrap their owned values in `Arc` as well.
    ///
    /// Each accepted connection is served on its own thread, so a
    /// single slow request (e.g., a large LTM scan) cannot block other
    /// clients from connecting.
    pub fn run_with_owned<L, F>(
        server: &IpcServer,
        mmap: std::sync::Arc<MmapState>,
        stm: std::sync::Arc<RingBuffer>,
        ltm: L,
        handler: F,
    ) where
        L: LtmAccess + Clone + Send + 'static,
        F: Fn(&MmapState, &RingBuffer, &mut LtmStore, u8, &[u8]) -> Vec<u8>
            + Clone
            + Send
            + 'static,
    {
        // Remove stale socket
        let _ = std::fs::remove_file(&server.socket_path);

        let listener = match UnixListener::bind(&server.socket_path) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[ipc] failed to bind socket: {e}");
                return;
            }
        };

        // Set non-blocking so we can poll for shutdown
        let _ = listener.set_nonblocking(true);
        let mut client_threads = Vec::new();

        while !server
            .shutdown_flag
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            let stream = match listener.accept() {
                Ok((s, _)) => s,
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
                Err(e) => {
                    eprintln!("[ipc] accept error: {e}");
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
            };

            let shutdown_flag = server.shutdown_flag();
            let mmap = mmap.clone();
            let stm = stm.clone();
            let ltm = ltm.clone();
            let handler = handler.clone();
            client_threads.push(std::thread::spawn(move || {
                Self::serve_connection(shutdown_flag, stream, &mmap, &stm, ltm, &handler);
            }));
        }

        // Do not return while a connected client may still be inside a
        // handler that mutates mmap/STM/LTM. The caller syncs those
        // stores after joining this thread; a fixed grace period would
        // allow a slow handler to write after the final sync.
        for client in client_threads {
            if let Err(e) = client.join() {
                eprintln!("[ipc] client thread panicked during shutdown: {e:?}");
            }
        }

        let _ = std::fs::remove_file(&server.socket_path);
    }

    /// Start the server (instance method, convenience).
    pub fn run<L, F>(
        &self,
        mmap: std::sync::Arc<MmapState>,
        stm: std::sync::Arc<RingBuffer>,
        ltm: L,
        handler: F,
    ) where
        L: LtmAccess + Clone + Send + 'static,
        F: Fn(&MmapState, &RingBuffer, &mut LtmStore, u8, &[u8]) -> Vec<u8>
            + Clone
            + Send
            + 'static,
    {
        Self::run_with_owned(self, mmap, stm, ltm, handler);
    }

    /// Serve a single connection to completion.
    ///
    /// The first message must be a HANDSHAKE. If the client's version
    /// doesn't match, the daemon responds with the daemon version and
    /// closes the connection. This prevents silent miscommunication
    /// when the protocol changes.
    fn serve_connection<L, F>(
        shutdown_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
        mut stream: UnixStream,
        mmap: &std::sync::Arc<MmapState>,
        stm: &std::sync::Arc<RingBuffer>,
        mut ltm: L,
        handler: &F,
    ) where
        L: LtmAccess,
        F: Fn(&MmapState, &RingBuffer, &mut LtmStore, u8, &[u8]) -> Vec<u8>,
    {
        // ── Handshake ───────────────────────────────────────────
        let (cmd, payload) = match read_message(&mut stream) {
            Ok(msg) => msg,
            Err(_) => return,
        };

        let client_version = if cmd == cmd::HANDSHAKE && !payload.is_empty() {
            payload[0]
        } else {
            // Legacy client (no handshake). Assume version 1.
            // Process the message as a normal request below.
            1
        };

        if cmd == cmd::HANDSHAKE {
            if client_version != PROTOCOL_VERSION {
                // Version mismatch — reply with the documented error
                // code followed by our version, then close. A client
                // that speaks a matching version always sees a bare
                // version byte (below); only a mismatched client sees
                // the high-bit error byte, so this is unambiguous.
                let resp = [error::VERSION_MISMATCH, PROTOCOL_VERSION];
                let _ = write_message(&mut stream, cmd::HANDSHAKE, &resp);
                return;
            }
            let resp = [PROTOCOL_VERSION];
            let _ = write_message(&mut stream, cmd::HANDSHAKE, &resp);
        } else {
            // Legacy client: process the first message as a normal
            // request (backwards compatible with pre-handshake clients).
            // The LTM guard is scoped to the handler call — holding it
            // across write_message would serialize LTM access against
            // this client's socket latency.
            let response = {
                let mut guard = ltm.access();
                handler(mmap, stm, &mut guard, cmd, &payload)
            };
            let _ = write_message(&mut stream, cmd, &response);
        }

        // ── Main loop ───────────────────────────────────────────
        // Set a read timeout so the loop can periodically check the
        // shutdown flag even when a client is idle (connected but not
        // sending). Without this, a blocking read_exact would prevent
        // the daemon from shutting down until the client disconnects.
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_millis(500)));
        loop {
            if shutdown_flag.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }

            let (cmd, payload) = match read_message(&mut stream) {
                Ok(msg) => msg,
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    // Read timed out — check shutdown flag and retry.
                    continue;
                }
                Err(_) => break, // Connection closed or error
            };

            if cmd == cmd::SHUTDOWN {
                let _ = write_message(&mut stream, cmd::SHUTDOWN, &[1]);
                shutdown_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                break;
            }

            // The LTM guard is scoped to the handler call — holding it
            // across write_message would serialize LTM access against
            // this client's socket latency.
            let response = {
                let mut guard = ltm.access();
                handler(mmap, stm, &mut guard, cmd, &payload)
            };

            if write_message(&mut stream, cmd, &response).is_err() {
                break;
            }
        }
    }

    /// Signal the server to shut down.
    pub fn shutdown(&self) {
        self.shutdown_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for IpcServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

// ─────────────────────────────────────────────────────────────────
//  IPC client (for testing and for Python via FFI later)
// ─────────────────────────────────────────────────────────────────

/// A simple IPC client for connecting to the daemon.
pub struct IpcClient {
    stream: UnixStream,
}

impl IpcClient {
    /// Connect to the daemon at the given socket path.
    ///
    /// Performs a protocol version handshake. If the daemon's version
    /// doesn't match `PROTOCOL_VERSION`, returns an error.
    pub fn connect(socket_path: impl AsRef<Path>) -> std::io::Result<Self> {
        let stream = UnixStream::connect(socket_path)?;
        let mut client = Self { stream };

        // Handshake: send our version, expect the daemon's version back.
        write_message(&mut client.stream, cmd::HANDSHAKE, &[PROTOCOL_VERSION])?;
        let (resp_cmd, resp_data) = read_message(&mut client.stream)?;
        if resp_cmd != cmd::HANDSHAKE || resp_data.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "handshake failed: no response",
            ));
        }
        let daemon_version = resp_data[0];
        if daemon_version >= 0x80 {
            // The daemon signalled a version mismatch: the first byte
            // is `error::VERSION_MISMATCH` and the second (if present)
            // is its actual version.
            let actual = resp_data.get(1).copied();
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                match actual {
                    Some(v) => format!(
                        "protocol version mismatch: client={}, daemon={}",
                        PROTOCOL_VERSION, v
                    ),
                    None => format!(
                        "protocol version mismatch: client={}, daemon version unknown",
                        PROTOCOL_VERSION
                    ),
                },
            ));
        }
        if daemon_version != PROTOCOL_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "protocol version mismatch: client={}, daemon={}",
                    PROTOCOL_VERSION, daemon_version
                ),
            ));
        }

        Ok(client)
    }

    /// Send a request and receive the response.
    pub fn request(&mut self, cmd_id: u8, payload: &[u8]) -> std::io::Result<Vec<u8>> {
        write_message(&mut self.stream, cmd_id, payload)?;
        let (resp_cmd, resp_data) = read_message(&mut self.stream)?;
        // Verify the response command matches the request. If the
        // server ever sends an unsolicited notification, the client
        // would otherwise misinterpret it as a response.
        if resp_cmd != cmd_id {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("response command {resp_cmd} does not match request command {cmd_id}"),
            ));
        }
        Ok(resp_data)
    }

    /// Ping the daemon.
    pub fn ping(&mut self) -> std::io::Result<bool> {
        let resp = self.request(cmd::PING, &[])?;
        Ok(!resp.is_empty() && resp[0] == 1)
    }

    /// Get the full core state (3288 bytes).
    pub fn get_state(&mut self) -> std::io::Result<GenesisCoreState> {
        let resp = self.request(cmd::GET_STATE, &[])?;
        if resp.len() < core::mem::size_of::<GenesisCoreState>() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "state response too short",
            ));
        }
        // Copy the bytes into a GenesisCoreState
        let mut state = GenesisCoreState::new(0, 0);
        // SAFETY: `resp.len()` was checked to be ≥
        // `size_of::<GenesisCoreState>()` bytes, so `resp.as_ptr()` is
        // valid for a read of that size. `state` is a freshly-constructed
        // stack value, valid for a write of its own size. The two regions
        // cannot overlap (one is heap, the other stack). `GenesisCoreState`
        // is `#[repr(C)]` with no padding gaps, so the byte copy produces
        // a valid bit-pattern.
        unsafe {
            std::ptr::copy_nonoverlapping(
                resp.as_ptr(),
                &mut state as *mut GenesisCoreState as *mut u8,
                core::mem::size_of::<GenesisCoreState>(),
            );
        }
        Ok(state)
    }

    /// Get the neurochemical summary.
    pub fn get_neuro_summary(&mut self) -> std::io::Result<NeuroSummary> {
        let resp = self.request(cmd::GET_NEURO_SUMMARY, &[])?;
        if resp.len() < core::mem::size_of::<NeuroSummary>() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "neuro summary response too short",
            ));
        }
        let mut summary = NeuroSummary::default();
        // SAFETY: `resp.len()` was checked ≥ size_of::<NeuroSummary>().
        // `summary` is a stack value. Non-overlapping heap→stack copy.
        // NeuroSummary is #[repr(C)] with no padding.
        unsafe {
            std::ptr::copy_nonoverlapping(
                resp.as_ptr(),
                &mut summary as *mut NeuroSummary as *mut u8,
                core::mem::size_of::<NeuroSummary>(),
            );
        }
        Ok(summary)
    }

    /// Get the plasticity profile — metaplastic state summary.
    pub fn get_plasticity_profile(&mut self) -> std::io::Result<PlasticityProfile> {
        let resp = self.request(cmd::GET_PLASTICITY_PROFILE, &[])?;
        if resp.len() < core::mem::size_of::<PlasticityProfile>() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "plasticity profile response too short",
            ));
        }
        let mut profile = PlasticityProfile::default();
        // SAFETY: `resp.len()` was checked ≥ size_of::<PlasticityProfile>().
        // `profile` is a stack value. Non-overlapping heap→stack copy.
        // PlasticityProfile is #[repr(C)] with no padding.
        unsafe {
            std::ptr::copy_nonoverlapping(
                resp.as_ptr(),
                &mut profile as *mut PlasticityProfile as *mut u8,
                core::mem::size_of::<PlasticityProfile>(),
            );
        }
        Ok(profile)
    }

    /// Get the interoceptive body state — what Genesis feels about
    /// her machine (CPU temp, memory pressure, load, battery, etc.).
    pub fn get_body_state(&mut self) -> std::io::Result<BodyStateSummary> {
        let resp = self.request(cmd::GET_BODY_STATE, &[])?;
        // Minimum: 7×f32 (28) + 1 + 4 + 1 + 5×f32 (20) + 4 = 58 bytes
        // (empty description). The fixed header is 7 f32 (cpu_temp_c,
        // temperature, arousal_freq, cognitive_load, io_activity,
        // stress_load, energy_reserve) + on_ac_power (1) + num_cores
        // (4) + distressed (1) + autonomic_rate (4) +
        // thermoregulatory_effort (4) + metabolic_rate (4) +
        // core_voltage (4) + supply_voltage (4) + desc_len (4). A
        // shorter response would fail the take_f32 / u32 reads below.
        if resp.len() < 58 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "body state response too short",
            ));
        }
        let mut off = 0;
        // Bounds-checked readers: return InvalidData instead of panicking
        // if a truncated/malformed response is missing bytes. The daemon
        // always sends a full response, but a single corrupted packet
        // must not crash the client.
        let take_f32 = |slice: &[u8], offset: &mut usize| -> std::io::Result<f32> {
            let bytes: [u8; 4] = slice
                .get(*offset..*offset + 4)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body state response truncated at f32 field",
                    )
                })?
                .try_into()
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body state f32 slice not 4 bytes",
                    )
                })?;
            *offset += 4;
            Ok(f32::from_le_bytes(bytes))
        };
        let take_u32 = |slice: &[u8], offset: &mut usize| -> std::io::Result<u32> {
            let bytes: [u8; 4] = slice
                .get(*offset..*offset + 4)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body state response truncated at u32 field",
                    )
                })?
                .try_into()
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body state u32 slice not 4 bytes",
                    )
                })?;
            *offset += 4;
            Ok(u32::from_le_bytes(bytes))
        };
        let cpu_temp_c = take_f32(&resp, &mut off)?;
        let temperature = take_f32(&resp, &mut off)?;
        let arousal_freq = take_f32(&resp, &mut off)?;
        let cognitive_load = take_f32(&resp, &mut off)?;
        let io_activity = take_f32(&resp, &mut off)?;
        let stress_load = take_f32(&resp, &mut off)?;
        let energy_reserve = take_f32(&resp, &mut off)?;
        let on_ac_power = resp.get(off).map(|&b| b != 0).unwrap_or(false);
        off += 1;
        let num_cores = take_u32(&resp, &mut off)?;
        let distressed = resp.get(off).map(|&b| b != 0).unwrap_or(false);
        off += 1;
        // Machine-native body fields — autonomic monitoring,
        // thermoregulatory effort, and metabolic rate.
        let autonomic_rate = take_f32(&resp, &mut off)?;
        let thermoregulatory_effort = take_f32(&resp, &mut off)?;
        let metabolic_rate = take_f32(&resp, &mut off)?;
        // Voltage fields — core voltage (Vcore) and supply voltage
        // (battery rail). These precede desc_len on the wire; skipping
        // them misaligns the description parse.
        let core_voltage = take_f32(&resp, &mut off)?;
        let supply_voltage = take_f32(&resp, &mut off)?;
        // Layout detection: wire v2 packets carry five extra f32
        // fields (silicon switching + miss ratios) with desc_len at
        // offset 74; legacy v1 packets put desc_len at 54. The u32
        // at 54 is v1's desc_len only if it's consistent with the
        // packet size — on a v2 packet those bytes are the
        // core_activity f32, which can't plausibly be a small
        // desc_len.
        let at54 = u32::from_le_bytes(
            resp.get(54..58)
                .and_then(|s| s.try_into().ok())
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body state response truncated at layout probe",
                    )
                })?,
        ) as usize;
        let v1 = at54 <= resp.len() - 58;
        let (core_activity, uncore_activity, dram_activity, cache_miss_rate, branch_miss_rate) =
            if !v1 && resp.len() >= 78 {
                (
                    take_f32(&resp, &mut off)?,
                    take_f32(&resp, &mut off)?,
                    take_f32(&resp, &mut off)?,
                    take_f32(&resp, &mut off)?,
                    take_f32(&resp, &mut off)?,
                )
            } else {
                (0.0, 0.0, 0.0, 0.0, 0.0)
            };
        let desc_len = take_u32(&resp, &mut off)? as usize;
        let description = if let Some(end) = off.checked_add(desc_len) {
            if end <= resp.len() {
                String::from_utf8_lossy(&resp[off..end]).into_owned()
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        Ok(BodyStateSummary {
            cpu_temp_c,
            temperature,
            arousal_freq,
            cognitive_load,
            io_activity,
            stress_load,
            energy_reserve,
            on_ac_power,
            num_cores,
            distressed,
            autonomic_rate,
            thermoregulatory_effort,
            metabolic_rate,
            core_voltage,
            supply_voltage,
            core_activity,
            uncore_activity,
            dram_activity,
            cache_miss_rate,
            branch_miss_rate,
            description,
        })
    }

    /// Get per-subsystem silicon telemetry — which part of the mind's
    /// process tree is firing (per-process CPU/I/O share plus
    /// microarchitectural miss ratios) — plus the module section:
    /// which brain part is active, self-reported by the cognitive
    /// mind via UPDATE_MODULE_STATUS.
    pub fn get_subsystem_telemetry(&mut self) -> std::io::Result<SubsystemTelemetryReport> {
        let resp = self.request(cmd::GET_SUBSYSTEM_TELEMETRY, &[])?;
        let count = resp.first().copied().unwrap_or(0) as usize;
        const REC: usize = 21;
        const MOD_REC: usize = 6;
        if resp.len() < 1 + count * REC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "subsystem telemetry response too short",
            ));
        }
        let mut report = SubsystemTelemetryReport {
            subsystems: Vec::with_capacity(count),
            modules: Vec::new(),
        };
        for i in 0..count {
            let base = 1 + i * REC;
            let f32_at = |o: usize| {
                resp.get(base + o..base + o + 4)
                    .and_then(|s| s.try_into().ok())
                    .map(f32::from_le_bytes)
                    .unwrap_or(0.0)
            };
            report.subsystems.push(SubsystemTelemetrySummary {
                subsystem: resp[base],
                pid: resp
                    .get(base + 1..base + 5)
                    .and_then(|s| s.try_into().ok())
                    .map(u32::from_le_bytes)
                    .unwrap_or(0),
                cpu: f32_at(5),
                io: f32_at(9),
                cache_miss_rate: f32_at(13),
                branch_miss_rate: f32_at(17),
            });
        }
        // Module section (v2). Older daemons end after the subsystem
        // section — treat a missing/short module tail as empty.
        let tail = 1 + count * REC;
        if resp.len() > tail {
            let mcount = resp[tail] as usize;
            if resp.len() >= tail + 1 + mcount * MOD_REC {
                report.modules = Vec::with_capacity(mcount);
                for i in 0..mcount {
                    let base = tail + 1 + i * MOD_REC;
                    report.modules.push(ModuleTelemetrySummary {
                        module_id: resp[base],
                        status: resp[base + 1],
                        cpu_share: f32::from_le_bytes(
                            resp.get(base + 2..base + 6)
                                .and_then(|s| s.try_into().ok())
                                .unwrap_or([0u8; 4]),
                        ),
                    });
                }
            }
        }
        Ok(report)
    }

    /// Get the body control state — what Genesis is doing to her body.
    pub fn get_body_control(&mut self) -> std::io::Result<BodyControlSummary> {
        let resp = self.request(cmd::GET_BODY_CONTROL, &[])?;
        // Response layout (all little-endian), matching the daemon's
        // GET_BODY_CONTROL / APPLY_BODY_CONTROL serializers:
        //   [u32 cpu_min_freq_khz][u32 cpu_max_freq_khz]
        //   [u32 gov_len][gov bytes]
        //   [u8 thermally_capped][f32 thermal_cap_temp_c]
        //   [i32 daemon_nice][i32 cognitive_nice]
        //   [u32 io_class_len][io_class bytes]
        //   [f32 plasticity_gate][u8 controlling_cognitive]
        //   [u32 epp_len][epp bytes]
        //   [u8 cpu_boost]
        //   [u32 desc_len][desc bytes]
        //
        // Minimum fixed header (empty governor, io_class and epp
        // strings): 2×u32 (8) + u32 gov_len (4) + u8 (1) + f32 (4)
        // + 2×i32 (8) + u32 io_class_len (4) + f32 (4) + u8 (1)
        // + u32 epp_len (4) + u8 cpu_boost (1) + u32 desc_len (4)
        // = 43 bytes. A shorter response would fail the take_* reads
        // below.
        if resp.len() < 43 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "body control response too short",
            ));
        }
        let mut off = 0;
        // Bounds-checked readers: return InvalidData instead of panicking
        // if a truncated/malformed response is missing bytes. The daemon
        // always sends a full response, but a single corrupted packet
        // must not crash the client.
        let take_u32 = |slice: &[u8], offset: &mut usize| -> std::io::Result<u32> {
            let bytes: [u8; 4] = slice
                .get(*offset..*offset + 4)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body control response truncated at u32 field",
                    )
                })?
                .try_into()
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body control u32 slice not 4 bytes",
                    )
                })?;
            *offset += 4;
            Ok(u32::from_le_bytes(bytes))
        };
        let take_f32 = |slice: &[u8], offset: &mut usize| -> std::io::Result<f32> {
            let bytes: [u8; 4] = slice
                .get(*offset..*offset + 4)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body control response truncated at f32 field",
                    )
                })?
                .try_into()
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body control f32 slice not 4 bytes",
                    )
                })?;
            *offset += 4;
            Ok(f32::from_le_bytes(bytes))
        };
        let take_i32 = |slice: &[u8], offset: &mut usize| -> std::io::Result<i32> {
            let bytes: [u8; 4] = slice
                .get(*offset..*offset + 4)
                .ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body control response truncated at i32 field",
                    )
                })?
                .try_into()
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "body control i32 slice not 4 bytes",
                    )
                })?;
            *offset += 4;
            Ok(i32::from_le_bytes(bytes))
        };
        let take_string = |slice: &[u8], offset: &mut usize| -> std::io::Result<String> {
            let len = take_u32(slice, offset)? as usize;
            // Use saturating_add to prevent usize overflow before .min()
            // evaluates — a malicious or corrupted u32 length could
            // wrap on 32-bit targets.
            let end = (*offset).saturating_add(len).min(slice.len());
            let s = String::from_utf8_lossy(&slice[*offset..end]).into_owned();
            *offset = end;
            Ok(s)
        };

        let cpu_min_freq_khz = take_u32(&resp, &mut off)?;
        let cpu_max_freq_khz = take_u32(&resp, &mut off)?;
        let cpu_governor = take_string(&resp, &mut off)?;
        let thermally_capped = resp.get(off).map(|&b| b != 0).unwrap_or(false);
        off += 1;
        let thermal_cap_temp_c = take_f32(&resp, &mut off)?;
        let daemon_nice = take_i32(&resp, &mut off)?;
        let cognitive_nice = take_i32(&resp, &mut off)?;
        let io_class = take_string(&resp, &mut off)?;
        let plasticity_gate = take_f32(&resp, &mut off)?;
        let controlling_cognitive = resp.get(off).map(|&b| b != 0).unwrap_or(false);
        off += 1;
        let cpu_epp = take_string(&resp, &mut off)?;
        // Turbo gate — a single byte decoded to the tri-state. A
        // missing byte (truncated packet) decodes to `Unavailable`,
        // the safe default.
        let cpu_boost =
            super::cpufreq::BoostState::from_wire(resp.get(off).copied().unwrap_or(0));
        off += 1;
        let description = take_string(&resp, &mut off)?;

        Ok(BodyControlSummary {
            cpu_min_freq_khz,
            cpu_max_freq_khz,
            cpu_governor,
            thermally_capped,
            thermal_cap_temp_c,
            daemon_nice,
            cognitive_nice,
            io_class,
            plasticity_gate,
            controlling_cognitive,
            cpu_epp,
            cpu_boost,
            description,
        })
    }

    /// Get the active inference summary — the generative self-model's
    /// projection (60 bytes). This exposes surprise, free energy,
    /// allostatic load, precision, dyadic attunement/synchrony, user
    /// affect, and prediction errors to the cognitive mind.
    pub fn get_inference_summary(&mut self) -> std::io::Result<crate::state::InferenceSignals> {
        let resp = self.request(cmd::GET_INFERENCE_SUMMARY, &[])?;
        if resp.len() < core::mem::size_of::<crate::state::InferenceSignals>() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "inference summary response too short",
            ));
        }
        let mut signals = crate::state::InferenceSignals::new();
        // SAFETY: `resp.len()` was checked ≥ size_of::<InferenceSignals>().
        // `signals` is a stack value. Non-overlapping heap→stack copy.
        // InferenceSignals is #[repr(C)] with no padding (all f32/u32,
        // 4-byte aligned).
        unsafe {
            std::ptr::copy_nonoverlapping(
                resp.as_ptr(),
                &mut signals as *mut crate::state::InferenceSignals as *mut u8,
                core::mem::size_of::<crate::state::InferenceSignals>(),
            );
        }
        Ok(signals)
    }

    /// Update the user affect observation — the cognitive mind's
    /// inference of the user's affective state from conversation
    /// features. This feeds the dyadic affective model.
    pub fn update_user_affect(
        &mut self,
        valence: f32,
        arousal: f32,
        engagement: f32,
        confidence: f32,
    ) -> std::io::Result<bool> {
        let mut payload = Vec::with_capacity(16);
        payload.extend_from_slice(&valence.to_le_bytes());
        payload.extend_from_slice(&arousal.to_le_bytes());
        payload.extend_from_slice(&engagement.to_le_bytes());
        payload.extend_from_slice(&confidence.to_le_bytes());
        let resp = self.request(cmd::UPDATE_USER_AFFECT, &payload)?;
        Ok(!resp.is_empty() && resp[0] == 1)
    }

    /// Store an event in STM.
    /// Returns `true` if the daemon acknowledged the store.
    pub fn store_event(
        &mut self,
        timestamp: u64,
        event_type: u8,
        source_module: u8,
        salience: f32,
        emotional_tag: &[f32; 12],
        text: &str,
    ) -> std::io::Result<bool> {
        let text_bytes = text.as_bytes();
        // Truncate on a UTF-8 char boundary so the daemon's lossy decode
        // doesn't end with a replacement character.
        let mut text_len = text_bytes.len().min(188);
        while text_len > 0 && !text.is_char_boundary(text_len) {
            text_len -= 1;
        }
        let text_len = text_len as u16;
        let mut payload = Vec::with_capacity(32 + text_len as usize);
        payload.extend_from_slice(&timestamp.to_le_bytes());
        payload.push(event_type);
        payload.push(source_module);
        payload.extend_from_slice(&salience.to_le_bytes());
        for &v in emotional_tag {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        payload.extend_from_slice(&text_len.to_le_bytes());
        payload.extend_from_slice(&text_bytes[..text_len as usize]);
        let resp = self.request(cmd::STORE_EVENT, &payload)?;
        Ok(!resp.is_empty() && resp[0] == 1)
    }

    /// Store an episode directly in LTM, bypassing STM.
    ///
    /// Returns the real LTM episode ID on success, or `None` if the
    /// daemon rejected the store. Used by the cognitive mind for
    /// deliberate memory stores where the episode ID is needed
    /// immediately for Python-side tracking.
    pub fn store_episode(
        &mut self,
        timestamp: u64,
        event_type: u8,
        source_module: u8,
        salience: f32,
        emotional_tag: &[f32; 12],
        text: &str,
    ) -> std::io::Result<Option<u64>> {
        let text_bytes = text.as_bytes();
        // The wire protocol's u16 text_len field limits text to 65535
        // bytes. Unlike store_event (STM), whose fixed ring-buffer
        // slot caps text at 188 bytes, the LTM path has no structural
        // size limit below the wire protocol's u16.
        let mut text_len = text_bytes.len().min(0xFFFF);
        while text_len > 0 && !text.is_char_boundary(text_len) {
            text_len -= 1;
        }
        let text_len = text_len as u16;
        let mut payload = Vec::with_capacity(32 + text_len as usize);
        payload.extend_from_slice(&timestamp.to_le_bytes());
        payload.push(event_type);
        payload.push(source_module);
        payload.extend_from_slice(&salience.to_le_bytes());
        for &v in emotional_tag {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        payload.extend_from_slice(&text_len.to_le_bytes());
        payload.extend_from_slice(&text_bytes[..text_len as usize]);
        let resp = self.request(cmd::STORE_EPISODE, &payload)?;
        if resp.len() >= 9 && resp[0] == 1 {
            let id = u64::from_le_bytes(resp[1..9].try_into().unwrap_or_default());
            Ok(Some(id))
        } else {
            Ok(None)
        }
    }

    /// Set the cognitive zone.
    /// Returns `true` if the daemon acknowledged the zone change.
    pub fn set_zone(&mut self, zone: CognitiveZone) -> std::io::Result<bool> {
        let resp = self.request(cmd::SET_ZONE, &[zone as u8])?;
        Ok(!resp.is_empty() && resp[0] == 1)
    }

    /// Apply a neurochemical impulse.
    /// Returns `true` if the daemon acknowledged the impulse.
    pub fn neuro_impulse(
        &mut self,
        chem: NeurochemicalId,
        magnitude: f32,
    ) -> std::io::Result<bool> {
        let mut payload = Vec::with_capacity(5);
        payload.push(chem as u8);
        payload.extend_from_slice(&magnitude.to_le_bytes());
        let resp = self.request(cmd::NEURO_IMPULSE, &payload)?;
        Ok(!resp.is_empty() && resp[0] == 1)
    }

    /// Request a sync to disk.
    /// Returns `true` if the daemon acknowledged the sync.
    pub fn sync(&mut self) -> std::io::Result<bool> {
        let resp = self.request(cmd::SYNC, &[])?;
        Ok(!resp.is_empty() && resp[0] == 1)
    }

    /// Request shutdown.
    /// Returns `true` if the daemon acknowledged the shutdown.
    pub fn shutdown(&mut self) -> std::io::Result<bool> {
        let resp = self.request(cmd::SHUTDOWN, &[])?;
        Ok(!resp.is_empty() && resp[0] == 1)
    }

    /// Update a module's status in the manifest.
    /// Used by the Python cognitive mind to register Sensory/Motor
    /// modules and send heartbeats.
    /// Returns `true` if the daemon acknowledged the update.
    pub fn update_module_status(&mut self, module_id: u8, status: u8) -> std::io::Result<bool> {
        let resp = self.request(cmd::UPDATE_MODULE_STATUS, &[module_id, status])?;
        Ok(!resp.is_empty() && resp[0] == 1)
    }
}

// ─────────────────────────────────────────────────────────────────
//  Default request handler
// ─────────────────────────────────────────────────────────────────

/// The default request handler. This is what the daemon uses to
/// process IPC requests.
///
/// **Response convention**: Ack commands (STORE_EVENT, SET_ZONE,
/// NEURO_IMPULSE, SYNC, SHUTDOWN, UPDATE_MODULE_STATUS,
/// UPDATE_USER_AFFECT, PING) return `vec![1]` on success. On error
/// they return `vec![error::XXX]` where `error::XXX >= 0x80`, so the
/// client can distinguish success (`1`) from error (`>= 0x80`) by
/// checking the high bit of the first response byte.
///
/// Data commands (GET_STATE, GET_NEURO_SUMMARY, etc.) return raw
/// struct bytes on success, or a zeroed struct of the same size if
/// the mmap seqlock read fails (graceful degradation).
pub fn default_handler(
    mmap: &MmapState,
    stm: &RingBuffer,
    ltm: &mut LtmStore,
    cmd: u8,
    payload: &[u8],
) -> Vec<u8> {
    match cmd {
        cmd::PING => {
            // Respond with ack (1=ok) + uptime_ms, matching the
            // protocol table and the Python PingResponse.unpack.
            // Uptime is derived from the mmap header's created_at
            // timestamp vs the current wall clock.
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            let created_ms = mmap
                .read_consistent()
                .map(|s| s.header.created_at)
                .unwrap_or(now_ms);
            let uptime = now_ms.saturating_sub(created_ms);
            let mut resp = vec![1u8];
            resp.extend_from_slice(&uptime.to_le_bytes());
            resp
        }

        cmd::HANDSHAKE => {
            // Handshake is handled in handle_connection, but if it
            // reaches here (legacy path), respond with our version.
            vec![PROTOCOL_VERSION]
        }

        cmd::GET_STATE => {
            // Return a consistent snapshot of the full state.
            // `read_consistent` uses the seqlock protocol to ensure a
            // consistent read without creating a `&GenesisCoreState`
            // reference to the mmap'd memory. Retry a few times in
            // case a writer is mid-update.
            let mut snapshot = None;
            for _ in 0..8 {
                snapshot = mmap.read_consistent();
                if snapshot.is_some() {
                    break;
                }
                std::hint::spin_loop();
            }
            match snapshot {
                Some(state) => {
                    // SAFETY: `state` is a valid stack-owned
                    // GenesisCoreState. The slice covers exactly
                    // size_of::<GenesisCoreState>() bytes.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            &state as *const GenesisCoreState as *const u8,
                            core::mem::size_of::<GenesisCoreState>(),
                        )
                    };
                    bytes.to_vec()
                }
                None => {
                    eprintln!("[ipc] GET_STATE: read_consistent failed after 8 retries");
                    vec![error::READ_FAILED]
                }
            }
        }

        cmd::GET_NEURO_SUMMARY => {
            // `read_consistent` uses the seqlock protocol without
            // creating a reference to the mmap'd memory. The tick loop
            // may be writing concurrently, so retry a few times.
            let mut snapshot = None;
            for _ in 0..8 {
                snapshot = mmap.read_consistent();
                if snapshot.is_some() {
                    break;
                }
                std::hint::spin_loop();
            }
            match snapshot {
                Some(state) => {
                    let summary = NeuroSummary {
                        arousal: crate::state::sanitize::finite_clamp(
                            state.neurochemicals.arousal,
                            0.0,
                            1.0,
                        ),
                        valence: crate::state::sanitize::finite_clamp(
                            state.neurochemicals.valence,
                            -1.0,
                            1.0,
                        ),
                        global_tone: crate::state::sanitize::finite_clamp(
                            state.neurochemicals.global_tone,
                            0.0,
                            2.0,
                        ),
                        plasticity_gate: crate::state::sanitize::finite_clamp(
                            state.neurochemicals.plasticity_gate,
                            0.0,
                            1.0,
                        ),
                        encoding_weight: crate::state::sanitize::finite_clamp(
                            state.memory.encoding_weight,
                            0.0,
                            1.0,
                        ),
                        consolidation_weight: crate::state::sanitize::finite_clamp(
                            state.memory.consolidation_weight,
                            0.0,
                            1.0,
                        ),
                        retrieval_weight: crate::state::sanitize::finite_clamp(
                            state.memory.retrieval_weight,
                            0.0,
                            1.0,
                        ),
                        phase: state.zones.emergent_phase,
                        _pad: [0; 3],
                    };
                    // SAFETY: `summary` is a valid stack-owned value.
                    // Slice covers exactly size_of::<NeuroSummary>() bytes.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            &summary as *const NeuroSummary as *const u8,
                            core::mem::size_of::<NeuroSummary>(),
                        )
                    };
                    bytes.to_vec()
                }
                None => {
                    eprintln!("[ipc] GET_NEURO_SUMMARY: read_consistent failed after 8 retries");
                    vec![error::READ_FAILED]
                }
            }
        }

        cmd::STORE_EVENT => {
            // Minimum payload: 8 (timestamp) + 1 (event_type) + 1 (source)
            // + 4 (salience) + 48 (12×f32 emotional_tag) + 2 (text_len) = 64
            if payload.len() < 64 {
                return vec![error::PAYLOAD_TOO_SHORT, 0, 0, 0, 0, 0, 0, 0];
            }
            let timestamp = u64::from_le_bytes(payload[0..8].try_into().unwrap_or_default());
            let event_type = payload[8];
            let source_module = payload[9];
            // Sanitize salience: NaN/inf from the wire would corrupt
            // the STM ring buffer entry and later the consolidation
            // logic (salience-based promotion thresholds).
            let salience = crate::state::sanitize::finite_clamp(
                f32::from_le_bytes(payload[10..14].try_into().unwrap_or_default()),
                0.0,
                1.0,
            );
            let mut emotional_tag = [0.0f32; 12];
            for (i, chunk) in payload[14..62].chunks(4).enumerate() {
                // Sanitize each emotional tag value: NaN/inf would
                // corrupt the emotional tag and later the LTM
                // associative retrieval (SimHash is sensitive to NaN).
                emotional_tag[i] = crate::state::sanitize::finite_clamp(
                    f32::from_le_bytes(chunk.try_into().unwrap_or_default()),
                    0.0,
                    1.0,
                );
            }
            let text_len =
                u16::from_le_bytes(payload[62..64].try_into().unwrap_or_default()) as usize;
            let text_len = text_len.min(payload.len() - 64);
            let text = String::from_utf8_lossy(&payload[64..64 + text_len]).into_owned();

            // Store in STM
            let entry = crate::store::RingBufferEntry::new(
                timestamp,
                crate::store::EventType::from_u8(event_type),
                source_module,
                salience,
                emotional_tag,
                &text,
            );
            stm.push(entry);
            vec![1] // ack
        }

        cmd::RETRIEVE_EPISODE => {
            // Request: [u64 episode_id]
            // Response: [u8 found] [episode data...]
            // If found=0, no more data. If found=1:
            // [u64 episode_id][u64 timestamp][f32 salience][u64 assoc_hash]
            // [f32×4 compact_tag][f32×12 full_tag][u8 event_type][u8 source_module]
            // [u32 text_len][text bytes]
            if payload.len() < 8 {
                return vec![0];
            }
            let episode_id = u64::from_le_bytes(payload[0..8].try_into().unwrap_or_default());
            match ltm.retrieve(episode_id) {
                Ok(ep) => {
                    let text_bytes = ep.text.as_bytes();
                    let mut resp = Vec::with_capacity(
                        1 + 8 + 8 + 4 + 8 + 16 + 48 + 1 + 1 + 4 + text_bytes.len(),
                    );
                    resp.push(1); // found
                    resp.extend_from_slice(&ep.episode_id.to_le_bytes());
                    resp.extend_from_slice(&ep.timestamp.to_le_bytes());
                    resp.extend_from_slice(&ep.salience.to_le_bytes());
                    resp.extend_from_slice(&ep.association_hash.to_le_bytes());
                    for &v in &ep.emotional_tag {
                        resp.extend_from_slice(&v.to_le_bytes());
                    }
                    for &v in &ep.full_emotional_tag {
                        resp.extend_from_slice(&v.to_le_bytes());
                    }
                    resp.push(ep.event_type);
                    resp.push(ep.source_module);
                    resp.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
                    resp.extend_from_slice(text_bytes);
                    resp
                }
                Err(_) => vec![0], // not found
            }
        }

        cmd::FIND_SIMILAR => {
            // Request: [u32 query_len][query bytes][u8 limit]
            // Response: [u32 count] then for each:
            //   [u64 episode_id][u32 hamming_dist][u64 timestamp][f32 salience]
            if payload.len() < 5 {
                return vec![0; 4];
            }
            let query_len =
                u32::from_le_bytes(payload[0..4].try_into().unwrap_or_default()) as usize;
            if payload.len() < 4 + query_len + 1 {
                return vec![0; 4];
            }
            let query = String::from_utf8_lossy(&payload[4..4 + query_len]).into_owned();
            let limit = payload[4 + query_len] as usize;

            // Use SimHash-based search (exact Hamming distance, no
            // superposition noise) instead of the LogHD bundle store
            // approach. The bundle store's superposition saturates at
            // scale (~24K+ episodes), making all activations near-
            // uniform and destroying retrieval accuracy. SimHash is
            // O(n) but gives exact results regardless of episode count.
            let results = ltm.find_similar_hash(&query, limit);

            let mut resp = Vec::with_capacity(4 + results.len() * 24);
            resp.extend_from_slice(&(results.len() as u32).to_le_bytes());
            for (ep_id, dist, entry) in results {
                resp.extend_from_slice(&ep_id.to_le_bytes());
                resp.extend_from_slice(&dist.to_le_bytes());
                resp.extend_from_slice(&entry.timestamp.to_le_bytes());
                resp.extend_from_slice(&entry.salience.to_le_bytes());
            }
            resp
        }

        cmd::SET_ZONE => {
            if payload.is_empty() {
                return vec![error::PAYLOAD_TOO_SHORT];
            }
            let zone = payload[0];
            // Validate zone value before attempting the write.
            if zone > CognitiveZone::Sleeping as u8 {
                return vec![error::INVALID_VALUE];
            }
            let now = crate::daemon::current_ms();
            let new_zone = CognitiveZone::from_u8(zone);

            // Read the pre-transition state to capture the old zone
            // and build a self-model for the autobiographical trace.
            let pre_snapshot = mmap.read_consistent();
            let old_zone = pre_snapshot
                .as_ref()
                .map(|s| s.zones.zone())
                .unwrap_or(new_zone);

            // Track whether the zone actually changed inside the
            // modify closure.
            //
            // An explicit SET_ZONE is a *manual* request from the
            // cognitive mind — the authority on the task zone — so it
            // must always be honoured, even if a previous manual
            // override is still active. `transition_to` marks the zone
            // as a manual override so the daemon's automatic tick
            // selection cannot fight it.
            //
            // Previously this was gated on `zone_override_active == 0`
            // and immediately cleared the override. That latched the
            // zone: once the override byte became 1 (it persists in the
            // mmap'd state), every subsequent SET_ZONE was silently
            // ignored — freezing the zone at Idle and desyncing the
            // mind's sleep state (and everything else) from the daemon.
            let mut zone_changed = false;
            match mmap.modify(now, |state| {
                zone_changed = new_zone != state.zones.zone();
                if zone_changed {
                    state.zones.transition_to(new_zone, now);
                }
                state.manifest.recompute();
            }) {
                Ok(()) => {
                    // Autobiographical trace: if the zone actually
                    // changed, store a structured record of the
                    // transition context in LTM. This gives the
                    // daemon a memory of its own state transitions
                    // — the same trace the old tick() method wrote.
                    // The trace is a structured log of real state
                    // values, not a hardcoded response.
                    if zone_changed && old_zone != new_zone
                        && let Some(snap) = pre_snapshot.as_ref()
                    {
                        let model = SelfModel::from_state(snap, now);
                        let mut emotional_tag = [0.0f32; 12];
                        for (i, slot) in emotional_tag.iter_mut().enumerate() {
                            let id = NeurochemicalId::from_u8(i as u8);
                            *slot = snap.neurochemicals.effective(id);
                        }
                        let cort_eff = snap.neurochemicals.effective(NeurochemicalId::Cortisol);
                        let da_eff = snap.neurochemicals.effective(NeurochemicalId::Dopamine);
                        let compact_tag = [model.arousal, model.valence, cort_eff, da_eff];
                        let salience = crate::state::sanitize::finite_clamp(
                            model.arousal + model.valence.abs(),
                            0.0,
                            1.0,
                        );
                        let trace_text = format!(
                            "zone_transition: {} -> {} | phase: {} | valence: {:.2} | arousal: {:.2} | sleep_pressure: {:.2} | cause: {}",
                            old_zone.label(),
                            new_zone.label(),
                            model.phase,
                            model.valence,
                            model.arousal,
                            model.sleep_pressure,
                            model.emotional_cause,
                        );
                        if let Err(e) = ltm.store_meta(
                            now,
                            salience,
                            emotional_tag,
                            compact_tag,
                            EventType::Internal as u8,
                            ModuleId::Subcognitive as u8,
                            &trace_text,
                        ) {
                            eprintln!("[ipc] SET_ZONE autobiographical trace failed: {e}");
                        }
                    }
                    vec![1]
                }
                Err(e) => {
                    eprintln!("[ipc] SET_ZONE modify failed: {e}");
                    vec![error::INTERNAL_ERROR]
                }
            }
        }

        cmd::GET_PHASE => {
            // `read_consistent` uses the seqlock protocol without
            // creating a reference to the mmap'd memory.
            // Retry a few times in case a writer is mid-update.
            let mut snapshot = None;
            for _ in 0..8 {
                snapshot = mmap.read_consistent();
                if snapshot.is_some() {
                    break;
                }
                std::hint::spin_loop();
            }
            match snapshot {
                Some(state) => {
                    let mut resp = Vec::with_capacity(9);
                    resp.push(state.zones.emergent_phase);
                    resp.extend_from_slice(
                        &crate::state::sanitize::finite_clamp(
                            state.neurochemicals.arousal,
                            0.0,
                            1.0,
                        )
                        .to_le_bytes(),
                    );
                    resp.extend_from_slice(
                        &crate::state::sanitize::finite_clamp(
                            state.neurochemicals.valence,
                            -1.0,
                            1.0,
                        )
                        .to_le_bytes(),
                    );
                    resp
                }
                None => {
                    eprintln!("[ipc] GET_PHASE: read_consistent failed after 8 retries");
                    vec![error::READ_FAILED]
                }
            }
        }

        cmd::NEURO_IMPULSE => {
            if payload.len() < 5 {
                return vec![error::PAYLOAD_TOO_SHORT];
            }
            let chem_id = payload[0];
            let raw_magnitude = f32::from_le_bytes(payload[1..5].try_into().unwrap_or_default());
            // Validate chem_id — from_u8 has a catch-all that maps
            // invalid IDs to Dopamine, which would silently corrupt
            // the dopamine pathway. Reject explicitly instead.
            if chem_id >= 18 {
                return vec![error::INVALID_VALUE];
            }
            // Reject NaN/inf from the wire — a single NaN impulse
            // would propagate through the coupling matrix to all 18
            // chemicals within one tick and permanently corrupt the
            // mmap'd state. `finite_or` replaces non-finite values
            // with 0.0 (no-op impulse), which is safe because the
            // caller cannot distinguish "rejected" from "applied with
            // zero magnitude" — both return ack [1].
            //
            // Also clamp the magnitude to a physiologically plausible
            // range [-1, 1]. Without this, a buggy or malicious client
            // could send a huge finite magnitude (e.g. 1e20) that would
            // set the chemical's velocity to an enormous value. The
            // velocity decays geometrically but would keep the chemical
            // saturated at 1.0 for hundreds of ticks, distorting the
            // entire neurochemical state and the downstream body-control
            // model. Clamping at the IPC boundary is the right place —
            // it protects all downstream code regardless of caller.
            let magnitude = crate::state::sanitize::finite_clamp(
                crate::state::sanitize::finite_or(raw_magnitude, 0.0),
                -1.0,
                1.0,
            );
            let now = crate::daemon::current_ms();
            let chem_id_enum = crate::state::neurochemical::NeurochemicalId::from_u8(chem_id);
            match mmap.modify(now, |state| {
                state
                    .neurochemicals
                    .apply_impulse_capped(chem_id_enum, magnitude, now);
                state.neurochemicals.recompute_derived();
                state.sync_neurochemistry_to_state();
            }) {
                Ok(()) => vec![1],
                Err(e) => {
                    eprintln!("[ipc] NEURO_IMPULSE modify failed: {e}");
                    vec![error::INTERNAL_ERROR]
                }
            }
        }

        cmd::GET_MEMORY_STATS => {
            let stats = MemoryStats {
                stm_count: stm.count(),
                ltm_count: ltm.count() as u64,
                ltm_capacity: ltm.capacity(),
                _pad: 0,
            };
            // SAFETY: `stats` is a valid stack-owned value.
            // Slice covers exactly size_of::<MemoryStats>() bytes.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &stats as *const MemoryStats as *const u8,
                    core::mem::size_of::<MemoryStats>(),
                )
            };
            bytes.to_vec()
        }

        cmd::SYNC => {
            let mut ok = true;
            if let Err(e) = ltm.sync() {
                eprintln!("[ipc] SYNC ltm sync failed: {e}");
                ok = false;
            }
            if let Err(e) = stm.sync() {
                eprintln!("[ipc] SYNC stm sync failed: {e}");
                ok = false;
            }
            if let Err(e) = mmap.sync() {
                eprintln!("[ipc] SYNC mmap sync failed: {e}");
                ok = false;
            }
            if ok {
                vec![1]
            } else {
                vec![error::INTERNAL_ERROR]
            }
        }

        cmd::SHUTDOWN => vec![1],

        cmd::UPDATE_MODULE_STATUS => {
            // Request:  [u8 module_id] [u8 status] [optional f32 cpu_share]
            // Response: [u8 ack (1=ok, 0=invalid module/status)]
            //
            // The optional cpu_share is the module's self-reported
            // activity fraction [0,1] — the cognitive mind measures
            // how much of its cycle time each brain part consumed
            // and reports it here so GET_SUBSYSTEM_TELEMETRY can answer
            // "which brain part is firing" at module granularity.
            if payload.len() < 2 {
                return vec![error::PAYLOAD_TOO_SHORT];
            }
            let module_id = payload[0];
            let status = payload[1];
            let cpu_share = if payload.len() >= 6 {
                Some(crate::state::sanitize::finite_clamp(
                    payload
                        .get(2..6)
                        .and_then(|s| s.try_into().ok())
                        .map(f32::from_le_bytes)
                        .unwrap_or(0.0),
                    0.0,
                    1.0,
                ))
            } else {
                None
            };
            let now = crate::daemon::current_ms();

            // Validate module_id against known ModuleId values (0-11)
            if module_id > ModuleId::Guardrails as u8 {
                return vec![0];
            }
            // Validate status (0-5)
            if status > 5 {
                return vec![0];
            }

            match mmap.modify(now, |state| {
                if let Some(module) = state.manifest.get_mut_by_id(module_id) {
                    module.status = status;
                    module.heartbeat(now);
                    if let Some(share) = cpu_share {
                        module.cpu_share = share;
                    }
                }
                state.manifest.recompute();
            }) {
                Ok(()) => vec![1],
                Err(e) => {
                    eprintln!("[ipc] UPDATE_MODULE_STATUS modify failed: {e}");
                    vec![error::INTERNAL_ERROR]
                }
            }
        }

        cmd::GET_RECENT_EPISODES => {
            // Request:  [u8 limit] [u8 source_filter]
            //   limit: max episodes to return (capped at 255)
            //   source_filter: module ID to filter by, or 255 for all
            // Response: [u32 count] then for each episode:
            //   [u64 episode_id][u64 timestamp][f32 salience]
            //   [u8 event_type][u8 source_module][u32 text_len][text bytes]
            if payload.len() < 2 {
                return vec![0; 4];
            }
            let limit = payload[0] as usize;
            let source_filter = payload[1];
            let filter = if source_filter == 255 {
                None
            } else {
                Some(source_filter)
            };

            let episodes = ltm.recent_episodes(limit, filter);

            let mut resp = Vec::with_capacity(4 + episodes.len() * 64);
            resp.extend_from_slice(&(episodes.len() as u32).to_le_bytes());
            for ep in &episodes {
                let text_bytes = ep.text.as_bytes();
                resp.extend_from_slice(&ep.episode_id.to_le_bytes());
                resp.extend_from_slice(&ep.timestamp.to_le_bytes());
                resp.extend_from_slice(&ep.salience.to_le_bytes());
                resp.push(ep.event_type);
                resp.push(ep.source_module);
                resp.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
                resp.extend_from_slice(text_bytes);
            }
            resp
        }

        cmd::GET_PLASTICITY_PROFILE => {
            // `read_consistent` uses the seqlock protocol without
            // creating a reference to the mmap'd memory.
            // Retry a few times in case a writer is mid-update.
            let mut snapshot = None;
            for _ in 0..8 {
                snapshot = mmap.read_consistent();
                if snapshot.is_some() {
                    break;
                }
                std::hint::spin_loop();
            }
            match snapshot {
                Some(state) => {
                    let n = &state.neurochemicals;
                    let bdnf_idx = NeurochemicalId::BDNF as usize;
                    let cort_idx = NeurochemicalId::Cortisol as usize;
                    let da_idx = NeurochemicalId::Dopamine as usize;
                    let srt_idx = NeurochemicalId::Serotonin as usize;

                    // Coupling-matrix drift: mean absolute deviation of
                    // off-diagonal entries from the default matrix. This
                    // quantifies how much metaplasticity has reshaped
                    // the chemical interaction structure.
                    let mut drift_sum = 0.0f32;
                    let mut drift_count = 0u32;
                    for (i, (default_row, current_row)) in crate::state::DEFAULT_COUPLING_MATRIX
                        .iter()
                        .zip(n.coupling_matrix.iter())
                        .enumerate()
                    {
                        for (j, (&dv, &cv)) in
                            default_row.iter().zip(current_row.iter()).enumerate()
                        {
                            if i != j {
                                drift_sum += (cv - dv).abs();
                                drift_count += 1;
                            }
                        }
                    }
                    let coupling_drift = if drift_count > 0 {
                        crate::state::sanitize::finite_clamp(
                            drift_sum / drift_count as f32,
                            0.0,
                            1.0,
                        )
                    } else {
                        0.0
                    };

                    // Mean receptor sensitivity across all chemicals.
                    let mean_rs: f32 = crate::state::sanitize::finite_clamp(
                        n.chemicals
                            .iter()
                            .map(|c| c.receptor_sensitivity)
                            .sum::<f32>()
                            / n.chemicals.len() as f32,
                        0.0,
                        1.0,
                    );

                    let profile = PlasticityProfile {
                        plasticity_gate: crate::state::sanitize::finite_clamp(
                            n.plasticity_gate,
                            0.0,
                            1.0,
                        ),
                        bdnf_effective: crate::state::sanitize::finite_clamp(
                            n.effective_levels[bdnf_idx],
                            0.0,
                            1.0,
                        ),
                        bdnf_tonic: crate::state::sanitize::finite_clamp(
                            n.chemicals[bdnf_idx].tonic_level,
                            0.0,
                            1.0,
                        ),
                        cortisol_effective: crate::state::sanitize::finite_clamp(
                            n.effective_levels[cort_idx],
                            0.0,
                            1.0,
                        ),
                        cortisol_tonic: crate::state::sanitize::finite_clamp(
                            n.chemicals[cort_idx].tonic_level,
                            0.0,
                            1.0,
                        ),
                        dopamine_effective: crate::state::sanitize::finite_clamp(
                            n.effective_levels[da_idx],
                            0.0,
                            1.0,
                        ),
                        serotonin_effective: crate::state::sanitize::finite_clamp(
                            n.effective_levels[srt_idx],
                            0.0,
                            1.0,
                        ),
                        coupling_drift,
                        mean_receptor_sensitivity: mean_rs,
                        emergent_phase: n.emergent_phase,
                        _pad: [0; 3],
                    };
                    // SAFETY: `profile` is a valid stack-owned value.
                    // Slice covers exactly size_of::<PlasticityProfile>() bytes.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            &profile as *const PlasticityProfile as *const u8,
                            core::mem::size_of::<PlasticityProfile>(),
                        )
                    };
                    bytes.to_vec()
                }
                None => {
                    eprintln!(
                        "[ipc] GET_PLASTICITY_PROFILE: read_consistent failed after 8 retries"
                    );
                    vec![error::READ_FAILED]
                }
            }
        }

        cmd::GET_BODY_STATE => {
            // Read the latest body state published by the interoception
            // path and serialize it with the shared layout.
            let body = super::interoception::read_shared_body_state();
            serialize_body_state(&body)
        }

        cmd::GET_SUBSYSTEM_TELEMETRY => {
            // Per-subsystem telemetry published at the end of each
            // interoception read — which part of the mind's process
            // tree is firing — plus the module section from the
            // runtime manifest: which brain *part* is active.
            // Subsystems are hardware-measured per process; modules are
            // self-reported by the cognitive mind via
            // UPDATE_MODULE_STATUS's optional cpu_share.
            let subsystems = super::interoception::read_shared_subsystem_telemetry();
            let mut modules = Vec::new();
            if let Some(state) = mmap.read_consistent() {
                for m in &state.manifest.modules {
                    if m.status() != crate::state::ModuleStatus::Stopped {
                        modules.push((m.module_id, m.status, m.cpu_share));
                    }
                }
            }
            serialize_subsystem_telemetry(&subsystems, &modules)
        }

        cmd::GET_BODY_CONTROL => {
            // Read the latest body control state published by the tick
            // loop and serialize it. The layout lives in
            // `serialize_body_control` so this handler and
            // APPLY_BODY_CONTROL cannot drift apart.
            let ctrl = super::cpufreq::read_shared_control_state();
            serialize_body_control(&ctrl)
        }

        cmd::GET_INFERENCE_SUMMARY => {
            // Return the inference signals from the core state.
            // These are the active inference engine's projection:
            // surprise, free energy, allostatic load, precision,
            // dyadic attunement/synchrony, user affect, prediction
            // errors. 60 bytes.
            let mut snapshot = None;
            for _ in 0..8 {
                snapshot = mmap.read_consistent();
                if snapshot.is_some() {
                    break;
                }
                std::hint::spin_loop();
            }
            match snapshot {
                Some(state) => {
                    // SAFETY: `state` is a valid stack-owned
                    // GenesisCoreState. InferenceSignals is a
                    // #[repr(C)] struct with no padding (all f32/u32,
                    // 4-byte aligned). Slice covers exactly
                    // size_of::<InferenceSignals>() bytes.
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            &state.inference_signals as *const crate::state::InferenceSignals
                                as *const u8,
                            core::mem::size_of::<crate::state::InferenceSignals>(),
                        )
                    };
                    bytes.to_vec()
                }
                None => {
                    eprintln!(
                        "[ipc] GET_INFERENCE_SUMMARY: read_consistent failed after 8 retries"
                    );
                    vec![error::READ_FAILED]
                }
            }
        }

        cmd::UPDATE_USER_AFFECT => {
            // Parse the user affect observation from the payload:
            // [f32 valence][f32 arousal][f32 engagement][f32 confidence]
            // = 16 bytes.
            if payload.len() < 16 {
                return vec![error::PAYLOAD_TOO_SHORT];
            }
            // Sanitize all four f32 fields: NaN/inf from the wire
            // would propagate through the dyadic model into the
            // neurochemical system via oxytocin/cortisol impulses.
            // Rust's native `.clamp()` returns NaN for NaN input,
            // so we use `finite_clamp` which guarantees a finite
            // return value.
            let valence = crate::state::sanitize::finite_clamp(
                f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]),
                -1.0,
                1.0,
            );
            let arousal = crate::state::sanitize::finite_clamp(
                f32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]),
                0.0,
                1.0,
            );
            let engagement = crate::state::sanitize::finite_clamp(
                f32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]),
                0.0,
                1.0,
            );
            let confidence = crate::state::sanitize::finite_clamp(
                f32::from_le_bytes([payload[12], payload[13], payload[14], payload[15]]),
                0.0,
                1.0,
            );
            let obs = super::dyadic_model::UserAffectObservation {
                valence,
                arousal,
                engagement,
                confidence,
            };
            super::dyadic_model::publish_user_affect(obs);
            vec![1] // ack
        }

        cmd::ARCHIVE_EPISODE => {
            // Request:  [u64 episode_id]
            // Response: [u8 ack (1=ok, 0=not found)]
            if payload.len() < 8 {
                return vec![error::PAYLOAD_TOO_SHORT];
            }
            let episode_id = u64::from_le_bytes(payload[0..8].try_into().unwrap_or_default());
            match ltm.archive(episode_id) {
                Ok(()) => vec![1],
                Err(_) => vec![0], // not found
            }
        }

        cmd::STORE_EPISODE => {
            // Direct LTM store, bypassing STM. Used by the cognitive
            // mind for deliberate memory stores where the real LTM
            // episode ID must be returned synchronously.
            //
            // Request:  [u64 timestamp][u8 event_type][u8 source_module]
            //           [f32 salience][f32×12 emotional_tag]
            //           [u16 text_len][text bytes]
            // Response: [u8 ack (1=ok, 0=failed)][u64 episode_id]
            //
            // Minimum payload: 8 + 1 + 1 + 4 + 48 + 2 = 64 bytes
            // (same wire layout as STORE_EVENT).
            if payload.len() < 64 {
                return vec![error::PAYLOAD_TOO_SHORT, 0, 0, 0, 0, 0, 0, 0, 0];
            }
            let timestamp = u64::from_le_bytes(payload[0..8].try_into().unwrap_or_default());
            let event_type = payload[8];
            let source_module = payload[9];
            let salience = crate::state::sanitize::finite_clamp(
                f32::from_le_bytes(payload[10..14].try_into().unwrap_or_default()),
                0.0,
                1.0,
            );
            let mut emotional_tag = [0.0f32; 12];
            for (i, chunk) in payload[14..62].chunks(4).enumerate() {
                emotional_tag[i] = crate::state::sanitize::finite_clamp(
                    f32::from_le_bytes(chunk.try_into().unwrap_or_default()),
                    0.0,
                    1.0,
                );
            }
            let text_len =
                u16::from_le_bytes(payload[62..64].try_into().unwrap_or_default()) as usize;
            let text_len = text_len.min(payload.len() - 64);
            let text = String::from_utf8_lossy(&payload[64..64 + text_len]).into_owned();

            // Expand the 12-element wire tag into the full LTM tag and
            // compute the compact tag, using the same helpers as the
            // consolidation engine so LTM payloads are identical
            // regardless of which path stored them.
            let full_tag = crate::daemon::consolidation::expand_emotional_tag(&emotional_tag);
            let compact_tag = crate::daemon::consolidation::compact_tag_from_12(&emotional_tag);

            match ltm.store(
                timestamp,
                salience,
                full_tag,
                compact_tag,
                event_type,
                source_module,
                &text,
            ) {
                Ok(episode_id) => {
                    let mut resp = Vec::with_capacity(9);
                    resp.push(1); // ack
                    resp.extend_from_slice(&episode_id.to_le_bytes());
                    resp
                }
                Err(e) => {
                    eprintln!("[ipc] STORE_EPISODE failed: {e}");
                    vec![0, 0, 0, 0, 0, 0, 0, 0, 0]
                }
            }
        }

        cmd::SEARCH_EPISODES => {
            // Request:  [u32 limit][u32 offset]
            // Response: [u32 count] then for each episode:
            //   [u64 episode_id][u64 timestamp][f32 salience]
            //   [u8 event_type][u8 source_module][u32 text_len][text bytes]
            // Same per-episode format as GET_RECENT_EPISODES.
            if payload.len() < 8 {
                return vec![0; 4];
            }
            let limit = u32::from_le_bytes(payload[0..4].try_into().unwrap_or_default()) as usize;
            let offset = u32::from_le_bytes(payload[4..8].try_into().unwrap_or_default()) as usize;
            // Cap limit to prevent unbounded response sizes.
            let limit = limit.min(1000);

            let episodes = ltm.search_episodes(limit, offset);

            let mut resp = Vec::with_capacity(4 + episodes.len() * 64);
            resp.extend_from_slice(&(episodes.len() as u32).to_le_bytes());
            for ep in &episodes {
                let text_bytes = ep.text.as_bytes();
                resp.extend_from_slice(&ep.episode_id.to_le_bytes());
                resp.extend_from_slice(&ep.timestamp.to_le_bytes());
                resp.extend_from_slice(&ep.salience.to_le_bytes());
                resp.push(ep.event_type);
                resp.push(ep.source_module);
                resp.extend_from_slice(&(text_bytes.len() as u32).to_le_bytes());
                resp.extend_from_slice(text_bytes);
            }
            resp
        }

        cmd::NEURO_ADJUST_BASELINE => {
            // Request:  [u8 chem_id][f32 delta]
            // Response: [u8 ack (1=ok)]
            if payload.len() < 5 {
                return vec![error::PAYLOAD_TOO_SHORT];
            }
            let chem_id = payload[0];
            let raw_delta = f32::from_le_bytes(payload[1..5].try_into().unwrap_or_default());
            // Validate chem_id — same as NEURO_IMPULSE.
            if chem_id >= 18 {
                return vec![error::INVALID_VALUE];
            }
            // Reject NaN/inf — same safety as NeuroImpulse.
            let delta = crate::state::sanitize::finite_or(raw_delta, 0.0);
            let chem_id_enum = crate::state::neurochemical::NeurochemicalId::from_u8(chem_id);
            let now = crate::daemon::current_ms();
            match mmap.modify(now, |state| {
                if let Some(chem) = state.neurochemicals.get_mut(chem_id_enum) {
                    chem.baseline =
                        crate::state::sanitize::finite_clamp(chem.baseline + delta, 0.05, 0.95);
                }
                state.neurochemicals.recompute_derived();
                state.sync_neurochemistry_to_state();
            }) {
                Ok(()) => vec![1],
                Err(e) => {
                    eprintln!("[ipc] NEURO_ADJUST_BASELINE modify failed: {e}");
                    vec![error::INTERNAL_ERROR]
                }
            }
        }

        // ─── Reactive commands (mind-driven) ───────────────────
        // These are handled by the reactive_handler which has access
        // to the TickLoop. If they reach default_handler, it means
        // no TickLoop was provided — return an error.
        cmd::ADVANCE_NEURO
        | cmd::CONSOLIDATE
        | cmd::ASSOCIATE
        | cmd::DREAM
        | cmd::READ_SENSORS
        | cmd::APPLY_BODY_CONTROL
        | cmd::SAVE_INFERENCE => {
            eprintln!(
                "[ipc] reactive command {cmd} sent to default_handler — no TickLoop available"
            );
            vec![error::INTERNAL_ERROR]
        }

        _ => vec![error::UNKNOWN_COMMAND], // Unknown command
    }
}

/// Create a reactive handler that wraps a TickLoop for mind-driven
/// commands, falling through to default_handler for everything else.
///
/// The daemon no longer runs a fixed tick loop. Instead, the
/// cognitive mind drives each function through IPC commands.
/// This handler dispatches the reactive commands to the TickLoop
/// and delegates all other commands to default_handler.
pub fn reactive_handler(
    tick_loop: std::sync::Arc<std::sync::Mutex<super::tick::TickLoop>>,
    data_dir: std::path::PathBuf,
) -> impl Fn(&MmapState, &RingBuffer, &mut LtmStore, u8, &[u8]) -> Vec<u8> + Clone + Send + 'static
{
    move |mmap, stm, ltm, cmd, payload| {
        match cmd {
            cmd::ADVANCE_NEURO => {
                // Request: [f32 dt] (4 bytes)
                if payload.len() < 4 {
                    return vec![
                        error::PAYLOAD_TOO_SHORT,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ];
                }
                let dt = crate::state::sanitize::finite_clamp(
                    f32::from_le_bytes(payload[0..4].try_into().unwrap_or_default()),
                    0.001,
                    10.0,
                );
                let mut tl = match tick_loop.lock() {
                    Ok(tl) => tl,
                    Err(e) => {
                        eprintln!("[ipc] ADVANCE_NEURO: tick_loop poisoned: {e}");
                        return vec![
                            error::INTERNAL_ERROR,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                            0,
                        ];
                    }
                };
                let (surprise, free_energy, precision, allostatic, tick_count) =
                    tl.advance_neuro(mmap, dt);
                let mut resp = vec![1u8]; // ack
                resp.extend_from_slice(&surprise.to_le_bytes());
                resp.extend_from_slice(&free_energy.to_le_bytes());
                resp.extend_from_slice(&precision.to_le_bytes());
                resp.extend_from_slice(&allostatic.to_le_bytes());
                resp.extend_from_slice(&tick_count.to_le_bytes());
                resp
            }

            cmd::CONSOLIDATE => {
                let mut tl = match tick_loop.lock() {
                    Ok(tl) => tl,
                    Err(e) => {
                        eprintln!("[ipc] CONSOLIDATE: tick_loop poisoned: {e}");
                        return vec![error::INTERNAL_ERROR, 0, 0, 0, 0];
                    }
                };
                let count = tl.consolidate(mmap, stm, ltm);
                let mut resp = vec![1u8];
                resp.extend_from_slice(&count.to_le_bytes());
                resp
            }

            cmd::ASSOCIATE => {
                let mut tl = match tick_loop.lock() {
                    Ok(tl) => tl,
                    Err(e) => {
                        eprintln!("[ipc] ASSOCIATE: tick_loop poisoned: {e}");
                        return vec![error::INTERNAL_ERROR, 0, 0, 0, 0];
                    }
                };
                let count = tl.associate(mmap, ltm);
                let mut resp = vec![1u8];
                resp.extend_from_slice(&count.to_le_bytes());
                resp
            }

            cmd::DREAM => {
                let mut tl = match tick_loop.lock() {
                    Ok(tl) => tl,
                    Err(e) => {
                        eprintln!("[ipc] DREAM: tick_loop poisoned: {e}");
                        return vec![error::INTERNAL_ERROR, 0, 0, 0, 0];
                    }
                };
                let insights = tl.dream(mmap, ltm);
                let mut resp = vec![1u8];
                resp.extend_from_slice(&insights.to_le_bytes());
                resp
            }

            cmd::READ_SENSORS => {
                let mut tl = match tick_loop.lock() {
                    Ok(tl) => tl,
                    Err(e) => {
                        eprintln!("[ipc] READ_SENSORS: tick_loop poisoned: {e}");
                        return vec![error::INTERNAL_ERROR];
                    }
                };
                let body = tl.read_sensors(mmap);
                // Serialize with the shared BodyState layout, prefixed
                // by a one-byte ack (same as GET_BODY_STATE otherwise).
                let mut resp = vec![1]; // ack
                resp.extend_from_slice(&serialize_body_state(&body));
                resp
            }

            cmd::APPLY_BODY_CONTROL => {
                let mut tl = match tick_loop.lock() {
                    Ok(tl) => tl,
                    Err(e) => {
                        eprintln!("[ipc] APPLY_BODY_CONTROL: tick_loop poisoned: {e}");
                        return vec![error::INTERNAL_ERROR];
                    }
                };
                let _ = tl.apply_body_control(mmap);
                // Return the control state — the shared layout,
                // prefixed with a one-byte ack.
                let ctrl = super::cpufreq::read_shared_control_state();
                let mut resp = vec![1]; // ack
                resp.extend_from_slice(&serialize_body_control(&ctrl));
                resp
            }

            cmd::SAVE_INFERENCE => {
                let tl = match tick_loop.lock() {
                    Ok(tl) => tl,
                    Err(e) => {
                        eprintln!("[ipc] SAVE_INFERENCE: tick_loop poisoned: {e}");
                        return vec![error::INTERNAL_ERROR];
                    }
                };
                if tl.save_inference_model(&data_dir) {
                    vec![1]
                } else {
                    vec![error::INTERNAL_ERROR]
                }
            }

            // Delegate all other commands to the default handler.
            _ => default_handler(mmap, stm, ltm, cmd, payload),
        }
    }
}
