//! Core-state snapshot — the mind's wire contract for `GET_STATE_SNAPSHOT`.
//!
//! ## Why this type exists
//!
//! `GET_STATE` returns the raw [`GenesisCoreState`] — the brain's
//! authoritative memory image, schema-versioned by `header.version`.
//! That is the right payload for body inspection (the Rust test
//! harness and forensics read it that way), but it is the wrong
//! payload for the cognitive mind: every consumer then has to mirror
//! the brain's byte layout in its own language, and the mirror has no
//! tripwire. A schema bump does not fail — it returns plausible
//! numbers read from the wrong fields. That has happened twice
//! (v3→v4 and v4→v5 offset bugs) and nearly a third time (the
//! transient v6 layout).
//!
//! This struct is the anti-corruption layer between the two: a
//! fixed-layout DTO that carries exactly what the mind consumes,
//! mapped field-by-field in [`From<&GenesisCoreState>`]. The brain's
//! schema can evolve — fields move, blocks insert, versions bump —
//! and the wire contract only changes when this file intentionally
//! changes, under its own [`SNAPSHOT_VERSION`]. A renamed or
//! retyped source field fails to compile here; it cannot silently
//! re-encode on the wire.
//!
//! The payload is self-describing: `magic`, `version`, and
//! `state_size` lead the struct so a mismatched or foreign buffer
//! is rejected before a single field offset is trusted.
//!
//! ## Layout (232 bytes)
//!
//! ```text
//! offset  field                     type
//! ------  -----                     ----
//!    0    magic                     [u8;4]   b"GSS1"
//!    4    version                   u32      SNAPSHOT_VERSION
//!    8    state_size                u32      size_of::<Self>()
//!   12    _pad                      u32      alignment
//!   16    created_at                u64      ms epoch
//!   24    last_updated              u64      ms epoch
//!   32    heartbeat                 u64      tick counter
//!   40    instance_id               u64
//!   48    ltm_episode_count         u64
//!   56    effective_levels          [f32;18] level × sensitivity ×
//!                                          desensitization ×
//!                                          internalization, per
//!                                          NeurochemicalId order
//!  128    global_tone               f32
//!  132    arousal                   f32
//!  136    valence                   f32
//!  140    plasticity_gate           f32      neurochemical gate
//!  144    surprise_ema              f32      ┐
//!  148    free_energy               f32      │
//!  152    expected_free_energy      f32      │
//!  156    allostasis_load           f32      │
//!  160    precision                 f32      │  InferenceSignals,
//!  164    attunement                f32      │  flattened so the
//!  168    dyadic_synchrony          f32      │  brain block can
//!  172    user_valence              f32      │  evolve without
//!  176    user_arousal              f32      │  moving the wire
//!  180    user_engagement           f32      │
//!  184    prediction_error_dopamine f32      │
//!  188    prediction_error_cortisol f32      │
//!  192    prediction_error_serotonin f32     │
//!  196    model_maturity            f32      │
//!  200    policy_authority          f32      ┘
//!  204    inference_tick_count      u32
//!  208    encoding_weight           f32      ┐
//!  212    consolidation_weight      f32      │  MemoryPointers
//!  216    retrieval_weight          f32      ┘  gating weights
//!  220    sleep_pressure            f32      h / H_MAX ∈ [0,1]
//!  224    stm_count                 u32
//!  228    cognitive_zone            u8       CognitiveZone
//!  229    emergent_phase            u8       MentalPhase
//!  230    _pad_tail                 [u8;2]
//!  232    TOTAL
//! ```
//!
//! Deliberately absent: the coupling matrix, per-chemical dynamics
//! (level/baseline/velocity/tonic/phasic/vesicular), the manifest,
//! ions, zones' task bookkeeping, and every other field the mind does
//! not read. The snapshot is a projection, not a copy — fields are
//! added here when a consumer needs them, never because the brain
//! gained them.

use super::GenesisCoreState;
use super::neurochemical::NEUROCHEMICAL_COUNT;
use super::sleep_network;

/// Magic bytes identifying a [`CoreStateSnapshot`] payload. ASCII
/// `"GSS1"` — distinct from the state file's `"GNSX"` so a raw
/// `GenesisCoreState` can never be mistaken for a snapshot.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"GSS1";

/// Snapshot schema version. Bump when this layout changes in any way
/// that breaks binary compatibility — independently of the brain's
/// `SCHEMA_VERSION`, which this type exists to decouple from.
pub const SNAPSHOT_VERSION: u32 = 1;

/// The mind's view of the core state — see the module docs for why
/// this is a separate type from [`GenesisCoreState`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CoreStateSnapshot {
    /// `SNAPSHOT_MAGIC` — payload identity.
    pub magic: [u8; 4],
    /// [`SNAPSHOT_VERSION`].
    pub version: u32,
    /// `size_of::<CoreStateSnapshot>()` as sent.
    pub state_size: u32,
    /// Alignment padding before the u64 block.
    pub _pad: u32,
    /// Milliseconds-timestamp when the state was first created.
    pub created_at: u64,
    /// Milliseconds-timestamp of the last write.
    pub last_updated: u64,
    /// Monotonic heartbeat counter (incremented each tick).
    pub heartbeat: u64,
    /// State instance identifier (creation timestamp in practice).
    pub instance_id: u64,
    /// Long-term memory episode count.
    pub ltm_episode_count: u64,
    /// Effective neurochemical levels, indexed by `NeurochemicalId`
    /// discriminant — what the mind actually feels, not raw levels.
    pub effective_levels: [f32; NEUROCHEMICAL_COUNT],
    /// Affective tone — weighted mean of effective levels.
    pub global_tone: f32,
    /// Arousal axis [0,1].
    pub arousal: f32,
    /// Valence axis [-1,+1].
    pub valence: f32,
    /// Plasticity gate [0,1] — neurochemical memory-forming capacity.
    pub plasticity_gate: f32,
    /// ─── Active-inference projection ────────────────────────────
    /// Exponential moving average of total surprise. [0,1]
    pub surprise_ema: f32,
    /// Current estimated free energy. [0,1]
    pub free_energy: f32,
    /// Expected future free energy — anticipatory demand. [0,1]
    pub expected_free_energy: f32,
    /// Accumulated allostatic load — persistent regulatory burden.
    /// [0,1]
    pub allostasis_load: f32,
    /// Precision of the generative model. [0,1]
    pub precision: f32,
    /// Oxytocin-mediated attunement to the user's affect. [0,1]
    pub attunement: f32,
    /// Rolling valence correlation with the user. [-1,1]
    pub dyadic_synchrony: f32,
    /// Inferred user valence. [-1,1]
    pub user_valence: f32,
    /// Inferred user arousal. [0,1]
    pub user_arousal: f32,
    /// Inferred user engagement. [0,1]
    pub user_engagement: f32,
    /// Dopamine prediction error. Finite and unbounded.
    pub prediction_error_dopamine: f32,
    /// Cortisol prediction error. [-2,2]
    pub prediction_error_cortisol: f32,
    /// Serotonin prediction error. [-2,2]
    pub prediction_error_serotonin: f32,
    /// Evidence the generative model is predictive and constrained.
    /// [0,1]
    pub model_maturity: f32,
    /// Delegated policy authority. [0,1]
    pub policy_authority: f32,
    /// Inference cycles run.
    pub inference_tick_count: u32,
    /// ─── Memory gating ──────────────────────────────────────────
    /// Encoding weight — current ability to form new memories.
    pub encoding_weight: f32,
    /// Consolidation weight — current drive to consolidate STM→LTM.
    pub consolidation_weight: f32,
    /// Retrieval weight — current ability to recall.
    pub retrieval_weight: f32,
    /// Homeostatic sleep pressure `h` as a fraction of its ceiling —
    /// a discharge gauge: it decays as sleep clears the need.
    pub sleep_pressure: f32,
    /// Short-term memory event count.
    pub stm_count: u32,
    /// Cognitive task zone — `CognitiveZone` discriminant.
    pub cognitive_zone: u8,
    /// Emergent mental phase — `MentalPhase` discriminant.
    pub emergent_phase: u8,
    /// Tail padding to the 8-byte boundary.
    pub _pad_tail: [u8; 2],
}

impl CoreStateSnapshot {
    /// Verify the self-describing header of a received payload.
    ///
    /// Call after copying the wire bytes into a snapshot — a mismatch
    /// means the payload is not a [`SNAPSHOT_VERSION`] snapshot and no
    /// field offset may be trusted.
    pub fn verify(&self) -> Result<(), &'static str> {
        if self.magic != SNAPSHOT_MAGIC {
            return Err("bad snapshot magic");
        }
        if self.version != SNAPSHOT_VERSION {
            return Err("snapshot version mismatch");
        }
        if self.state_size as usize != core::mem::size_of::<Self>() {
            return Err("snapshot size mismatch");
        }
        Ok(())
    }
}

impl From<&GenesisCoreState> for CoreStateSnapshot {
    /// Project the authoritative state into the mind's wire contract.
    /// This is the only place that knows both layouts — every field
    /// is named on both sides, so a brain-schema change either
    /// compiles (the projection still resolves) or fails loudly here.
    fn from(state: &GenesisCoreState) -> Self {
        let neuro = &state.neurochemicals;
        let signals = &state.inference_signals;
        let memory = &state.memory;
        Self {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            state_size: core::mem::size_of::<Self>() as u32,
            _pad: 0,
            created_at: state.header.created_at,
            last_updated: state.header.last_updated,
            heartbeat: state.header.heartbeat,
            instance_id: state.header.instance_id,
            ltm_episode_count: memory.ltm_episode_count,
            effective_levels: neuro.effective_levels,
            global_tone: neuro.global_tone,
            arousal: neuro.arousal,
            valence: neuro.valence,
            plasticity_gate: neuro.plasticity_gate,
            surprise_ema: signals.surprise_ema,
            free_energy: signals.free_energy,
            expected_free_energy: signals.expected_free_energy,
            allostasis_load: signals.allostasis_load,
            precision: signals.precision,
            attunement: signals.attunement,
            dyadic_synchrony: signals.dyadic_synchrony,
            user_valence: signals.user_valence,
            user_arousal: signals.user_arousal,
            user_engagement: signals.user_engagement,
            prediction_error_dopamine: signals.prediction_error_dopamine,
            prediction_error_cortisol: signals.prediction_error_cortisol,
            prediction_error_serotonin: signals.prediction_error_serotonin,
            model_maturity: signals.model_maturity,
            policy_authority: signals.policy_authority,
            inference_tick_count: signals.inference_tick_count,
            encoding_weight: memory.encoding_weight,
            consolidation_weight: memory.consolidation_weight,
            retrieval_weight: memory.retrieval_weight,
            sleep_pressure: (state.sleep_network[4] / sleep_network::H_MAX) as f32,
            stm_count: memory.stm_count,
            cognitive_zone: state.zones.cognitive_zone,
            emergent_phase: state.zones.emergent_phase,
            _pad_tail: [0; 2],
        }
    }
}

// --- Compile-time layout assertions ---
const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<CoreStateSnapshot>() == 232);
    assert!(offset_of!(CoreStateSnapshot, magic) == 0);
    assert!(offset_of!(CoreStateSnapshot, version) == 4);
    assert!(offset_of!(CoreStateSnapshot, state_size) == 8);
    assert!(offset_of!(CoreStateSnapshot, created_at) == 16);
    assert!(offset_of!(CoreStateSnapshot, last_updated) == 24);
    assert!(offset_of!(CoreStateSnapshot, heartbeat) == 32);
    assert!(offset_of!(CoreStateSnapshot, instance_id) == 40);
    assert!(offset_of!(CoreStateSnapshot, ltm_episode_count) == 48);
    assert!(offset_of!(CoreStateSnapshot, effective_levels) == 56);
    assert!(offset_of!(CoreStateSnapshot, global_tone) == 128);
    assert!(offset_of!(CoreStateSnapshot, arousal) == 132);
    assert!(offset_of!(CoreStateSnapshot, valence) == 136);
    assert!(offset_of!(CoreStateSnapshot, plasticity_gate) == 140);
    assert!(offset_of!(CoreStateSnapshot, surprise_ema) == 144);
    assert!(offset_of!(CoreStateSnapshot, free_energy) == 148);
    assert!(offset_of!(CoreStateSnapshot, expected_free_energy) == 152);
    assert!(offset_of!(CoreStateSnapshot, allostasis_load) == 156);
    assert!(offset_of!(CoreStateSnapshot, precision) == 160);
    assert!(offset_of!(CoreStateSnapshot, attunement) == 164);
    assert!(offset_of!(CoreStateSnapshot, dyadic_synchrony) == 168);
    assert!(offset_of!(CoreStateSnapshot, user_valence) == 172);
    assert!(offset_of!(CoreStateSnapshot, user_arousal) == 176);
    assert!(offset_of!(CoreStateSnapshot, user_engagement) == 180);
    assert!(offset_of!(CoreStateSnapshot, prediction_error_dopamine) == 184);
    assert!(offset_of!(CoreStateSnapshot, prediction_error_cortisol) == 188);
    assert!(offset_of!(CoreStateSnapshot, prediction_error_serotonin) == 192);
    assert!(offset_of!(CoreStateSnapshot, model_maturity) == 196);
    assert!(offset_of!(CoreStateSnapshot, policy_authority) == 200);
    assert!(offset_of!(CoreStateSnapshot, inference_tick_count) == 204);
    assert!(offset_of!(CoreStateSnapshot, encoding_weight) == 208);
    assert!(offset_of!(CoreStateSnapshot, consolidation_weight) == 212);
    assert!(offset_of!(CoreStateSnapshot, retrieval_weight) == 216);
    assert!(offset_of!(CoreStateSnapshot, sleep_pressure) == 220);
    assert!(offset_of!(CoreStateSnapshot, stm_count) == 224);
    assert!(offset_of!(CoreStateSnapshot, cognitive_zone) == 228);
    assert!(offset_of!(CoreStateSnapshot, emergent_phase) == 229);
};
