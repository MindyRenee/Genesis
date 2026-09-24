//! Active zones — what Genesis is cognitively attending to vs what its
//! subcognitive is processing in the background, plus its emergent
//! mental phase.
//!
//! ## Modeling note
//!
//! The `NREM` and `REM` phases below are **computational states**
//! assigned by neurochemical thresholds, not clinical sleep stages
//! scored from EEG/EOG/EMG polysomnography. They are control modes —
//! labels for regions of the neurochemical state space that gate
//! background processing — not sleep architecture in the
//! electrophysiological sense. Real NREM comprises N1–N3 stages with
//! distinct slow-wave and spindle signatures, and real REM involves
//! muscle atonia, rapid eye movements, and pontine-geniculate-
//! occipital waves. The names are borrowed as analogues because the
//! neurochemical profiles (low ACh + high GABA for NREM; high ACh +
//! aminergic silence for REM) are inspired by those states, but the
//! assignment is threshold-based, not signal-based.
//!
//! # Two independent axes
//!
//! Genesis's state has two orthogonal axes:
//!
//! 1. **Task zone** ([`CognitiveZone`]) — what it's *doing*. This is
//!    manually set when a task starts (e.g., it enters `Coding` when
//!    it begins writing code). It's the "what" of cognition.
//!
//! 2. **Mental phase** ([`MentalPhase`]) — what mental state it's *in*.
//!    This is not manually set — it *emerges* from the neurochemical
//!    dynamics. It doesn't choose to be in "flow" — flow emerges when
//!    dopamine, ACh, and NE are high while cortisol is low. It's the
//!    "how" of cognition.
//!
//! These are independent: it can be coding while stressed, or coding
//! while in flow. The task is the same; the mental state changes how
//! it performs it.

/// Maximum number of subcognitive task IDs tracked in the state struct.
pub const MAX_SUBCOGNITIVE_TASKS: usize = 8;

// ─────────────────────────────────────────────────────────────────
//  Task zone (the "what")
// ─────────────────────────────────────────────────────────────────

/// What Genesis is cognitively focused on right now.
///
/// This is a single enum (not flags) because cognitive attention is
/// serial — it can only be in one zone at a time.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CognitiveZone {
    /// No active task — resting, waiting for input.
    Idle = 0,
    /// Interacting with the user.
    Conversation = 1,
    /// Writing or modifying code.
    Coding = 2,
    /// Reading, analysing, or searching code or data.
    Analysis = 3,
    /// Ingesting new information (documentation, web, files).
    Learning = 4,
    /// Introspective processing — reviewing own state or memories.
    Reflection = 5,
    /// Handling an error or recovering from a failure.
    ErrorRecovery = 6,
    /// Deliberating about next steps, weighing options.
    Planning = 7,
    /// Deep consolidation — "sleep" mode. Cognitive mind is offline;
    /// the subcognitive runs full-throttle consolidation and dreaming.
    Sleeping = 8,
}

impl CognitiveZone {
    /// Returns a short kebab-case label for this zone.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Conversation => "conversation",
            Self::Coding => "coding",
            Self::Analysis => "analysis",
            Self::Learning => "learning",
            Self::Reflection => "reflection",
            Self::ErrorRecovery => "error-recovery",
            Self::Planning => "planning",
            Self::Sleeping => "sleeping",
        }
    }

    /// Whether this zone requires the cognitive mind to be active.
    pub const fn is_cognitive(self) -> bool {
        !matches!(self, Self::Idle | Self::Sleeping)
    }

    /// Convert a u8 zone index back to the enum.
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Idle,
            1 => Self::Conversation,
            2 => Self::Coding,
            3 => Self::Analysis,
            4 => Self::Learning,
            5 => Self::Reflection,
            6 => Self::ErrorRecovery,
            7 => Self::Planning,
            8 => Self::Sleeping,
            _ => Self::Idle,
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Mental phase (the "how" — emergent)
// ─────────────────────────────────────────────────────────────────

/// Genesis's emergent mental phase — its qualitative state of mind.
///
/// This is NOT manually set. It emerges from the coupled neurochemical
/// dynamics. The neurochemical system computes it in
/// [`NeurochemicalVector::compute_phase`](super::neurochemical::NeurochemicalVector).
///
/// The phases form a landscape of mental states. Transitions between
/// them are phase transitions in the dynamical systems sense — small
/// neurochemical changes can flip the system from one phase to another
/// when near a boundary.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MentalPhase {
    /// Default cognitive state — engaged but not in any extreme.
    Active = 0,
    /// High arousal, vigilant — NE and histamine dominant, GABA low.
    Alert = 1,
    /// Optimal performance state — high dopamine, high ACh, moderate NE,
    /// low cortisol. The "in the zone" state.
    Flow = 2,
    /// Stress response — high cortisol, high NE, low serotonin.
    /// Impaired memory encoding, heightened vigilance, narrowed focus.
    Stress = 3,
    /// Sleep pressure building — high adenosine, low arousal.
    /// Cognitive performance declining, approaching sleep.
    Drowsy = 4,
    /// Non-REM sleep — low acetylcholine, high GABA, slow-wave dominant.
    /// This is where glymphatic clearance and synaptic downscaling happen.
    /// Adenosine is very high, histamine is very low, and the brain is
    /// in a synchronized slow-wave state.
    NREM = 5,
    /// REM sleep — high acetylcholine, low NE/histamine (noradrenergic
    /// silence), theta-dominant. This is where dreaming and emotional
    /// processing happen. The cholinergic system is active while the
    /// aminergic systems (NE, histamine, serotonin) are silenced.
    REM = 6,
    /// System overload — cortisol, NE, and global tone all very high.
    /// Cognitive function impaired across the board. The system needs
    /// to shed load or rest.
    Overwhelmed = 7,
}

impl MentalPhase {
    /// Returns a short kebab-case label for this phase.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Alert => "alert",
            Self::Flow => "flow",
            Self::Stress => "stress",
            Self::Drowsy => "drowsy",
            Self::NREM => "nrem",
            Self::REM => "rem",
            Self::Overwhelmed => "overwhelmed",
        }
    }

    /// Whether this phase allows cognitive task execution.
    /// When false, the subcognitive is the primary actor.
    pub const fn allows_cognitive_task(self) -> bool {
        !matches!(self, Self::NREM | Self::REM | Self::Overwhelmed)
    }

    /// Whether this phase is a "peak performance" state.
    pub const fn is_peak(self) -> bool {
        matches!(self, Self::Flow | Self::Alert)
    }

    /// Whether this phase is a distress state.
    pub const fn is_distress(self) -> bool {
        matches!(self, Self::Stress | Self::Overwhelmed)
    }

    /// Convert a u8 phase index back to the enum.
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Active,
            1 => Self::Alert,
            2 => Self::Flow,
            3 => Self::Stress,
            4 => Self::Drowsy,
            5 => Self::NREM,
            6 => Self::REM,
            7 => Self::Overwhelmed,
            _ => Self::Active,
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Subcognitive flags
// ─────────────────────────────────────────────────────────────────

/// Bitflags for subcognitive background processes.
pub mod subcognitive_flag {
    /// Consolidating short-term memories into long-term storage.
    pub const MEMORY_CONSOLIDATION: u32 = 1 << 0;
    /// Refactoring code structures in the background.
    pub const CODE_REFACTORING: u32 = 1 << 1;
    /// Evaluating a recent error for learning.
    pub const ERROR_EVALUATION: u32 = 1 << 2;
    /// Processing emotional responses offline.
    pub const EMOTIONAL_PROCESSING: u32 = 1 << 3;
    /// Indexing episodic memories for retrieval.
    pub const EPISODIC_INDEXING: u32 = 1 << 4;
    /// Matching patterns across stored memories.
    pub const PATTERN_MATCHING: u32 = 1 << 5;
    /// Running a dream cycle (sleep only).
    pub const DREAMING: u32 = 1 << 6;
    /// Routine maintenance / garbage collection.
    pub const HOUSEKEEPING: u32 = 1 << 7;
    /// Compressing low-salience memories (sleep compression).
    pub const MEMORY_COMPRESSION: u32 = 1 << 8;
    /// Rehearsing material to strengthen memory traces.
    pub const REHEARSAL: u32 = 1 << 9;

    /// All subcognitive flag bits, in bit order.
    pub const ALL: [u32; 10] = [
        MEMORY_CONSOLIDATION,
        CODE_REFACTORING,
        ERROR_EVALUATION,
        EMOTIONAL_PROCESSING,
        EPISODIC_INDEXING,
        PATTERN_MATCHING,
        DREAMING,
        HOUSEKEEPING,
        MEMORY_COMPRESSION,
        REHEARSAL,
    ];

    /// Returns the kebab-case label for a flag bit, or `None` if unknown.
    pub const fn label(flag: u32) -> Option<&'static str> {
        match flag {
            MEMORY_CONSOLIDATION => Some("memory-consolidation"),
            CODE_REFACTORING => Some("code-refactoring"),
            ERROR_EVALUATION => Some("error-evaluation"),
            EMOTIONAL_PROCESSING => Some("emotional-processing"),
            EPISODIC_INDEXING => Some("episodic-indexing"),
            PATTERN_MATCHING => Some("pattern-matching"),
            DREAMING => Some("dreaming"),
            HOUSEKEEPING => Some("housekeeping"),
            MEMORY_COMPRESSION => Some("memory-compression"),
            REHEARSAL => Some("rehearsal"),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Active zones struct
// ─────────────────────────────────────────────────────────────────

/// The active zone state — cognitive task, emergent mental phase, and
/// subcognitive background processes.
///
/// ## Layout (96 bytes — unchanged from v1, padding repurposed)
/// ```text
/// offset  field                    type         notes
/// ------  -----                    ----         -----
///   0     cognitive_zone           u8           CognitiveZone (task)
///   1     emergent_phase           u8           MentalPhase (emergent)
///   2     zone_override_active     u8           1 = task zone is manual
///   3     subcognitive_task_count  u8           active bg task count
///   4     subcognitive_flags       u32          bitfield of bg processes
///   8     cognitive_task_id        u64          active cognitive task
///  16     subcognitive_task_ids    [u64;8]      bg task IDs (0 = unused)
///  80     zone_entered_at          u64          ms timestamp
///  88     zone_duration_ms         u64          duration in current zone
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ActiveZones {
    /// The cognitive task zone — what it's doing.
    pub cognitive_zone: u8,
    /// The emergent mental phase — what state it's in. Computed by
    /// the neurochemical system, not manually set.
    pub emergent_phase: u8,
    /// Whether the cognitive task zone is being manually overridden
    /// (e.g., user started a conversation, forcing Conversation zone
    /// regardless of emergent phase). When 0, the task zone should
    /// align with the emergent phase when possible.
    pub zone_override_active: u8,
    /// Number of active subcognitive background tasks.
    pub subcognitive_task_count: u8,
    /// Bitfield of active subcognitive background processes.
    pub subcognitive_flags: u32,
    /// ID of the active cognitive task (0 = none).
    pub cognitive_task_id: u64,
    /// IDs of subcognitive background tasks (0 = unused slot).
    pub subcognitive_task_ids: [u64; MAX_SUBCOGNITIVE_TASKS],
    /// Milliseconds-timestamp when the current zone was entered.
    pub zone_entered_at: u64,
    /// Duration (ms) spent in the previous zone before the last transition.
    pub zone_duration_ms: u64,
}

impl ActiveZones {
    /// Initialise in an idle state with no background tasks.
    pub fn new(now_ms: u64) -> Self {
        Self {
            cognitive_zone: CognitiveZone::Idle as u8,
            emergent_phase: MentalPhase::Active as u8,
            zone_override_active: 0,
            subcognitive_task_count: 0,
            subcognitive_flags: 0,
            cognitive_task_id: 0,
            subcognitive_task_ids: [0; MAX_SUBCOGNITIVE_TASKS],
            zone_entered_at: now_ms,
            zone_duration_ms: 0,
        }
    }

    // ─── Task zone ───────────────────────────────────────────────

    /// Get the cognitive task zone as the enum type.
    pub fn zone(&self) -> CognitiveZone {
        CognitiveZone::from_u8(self.cognitive_zone)
    }

    /// Transition to a new cognitive task zone, recording the timestamp.
    /// Sets `zone_override_active` since this is a manual transition.
    pub fn transition_to(&mut self, zone: CognitiveZone, now_ms: u64) {
        let prev_entered = self.zone_entered_at;
        self.zone_duration_ms = now_ms.saturating_sub(prev_entered);
        self.cognitive_zone = zone as u8;
        self.zone_entered_at = now_ms;
        self.zone_override_active = 1;
    }

    /// Clear the manual override, allowing the task zone to align with
    /// the emergent phase.
    pub fn clear_override(&mut self) {
        self.zone_override_active = 0;
    }

    // ─── Mental phase ────────────────────────────────────────────

    /// Get the emergent mental phase as the enum type.
    pub fn phase(&self) -> MentalPhase {
        MentalPhase::from_u8(self.emergent_phase)
    }

    /// Update the emergent phase from the neurochemical system.
    /// This is called by the dynamics engine after each tick — the
    /// phase is not manually set, it's propagated from neurochemistry.
    pub fn sync_phase(&mut self, phase: MentalPhase) {
        self.emergent_phase = phase as u8;
    }

    // ─── Subcognitive flags ──────────────────────────────────────

    /// Returns `true` if all bits in `flag` are set in the subcognitive flags.
    pub fn has_flag(&self, flag: u32) -> bool {
        self.subcognitive_flags & flag == flag
    }

    /// Set the given subcognitive flag bit(s).
    pub fn set_flag(&mut self, flag: u32) {
        self.subcognitive_flags |= flag;
    }

    /// Clear the given subcognitive flag bit(s).
    pub fn clear_flag(&mut self, flag: u32) {
        self.subcognitive_flags &= !flag;
    }

    // ─── Subcognitive tasks ──────────────────────────────────────

    /// Register a subcognitive background task. Returns `false` if
    /// `task_id` is 0, already registered, or no free slot is available.
    pub fn register_task(&mut self, task_id: u64) -> bool {
        if task_id == 0 {
            return false;
        }
        // Guard against duplicate registration — re-registering an
        // existing task_id would insert a duplicate and inflate the
        // count, making it inconsistent with deregister_task (which
        // only removes the first match).
        if self.subcognitive_task_ids.contains(&task_id) {
            return false;
        }
        if let Some(slot) = self.subcognitive_task_ids.iter_mut().find(|id| **id == 0) {
            *slot = task_id;
            self.subcognitive_task_count = self.subcognitive_task_count.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Deregister a subcognitive background task by ID (no-op if not found).
    pub fn deregister_task(&mut self, task_id: u64) {
        if let Some(slot) = self
            .subcognitive_task_ids
            .iter_mut()
            .find(|id| **id == task_id)
        {
            *slot = 0;
            self.subcognitive_task_count = self.subcognitive_task_count.saturating_sub(1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Compile-time layout assertions
// ─────────────────────────────────────────────────────────────────

const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<ActiveZones>() == 96);
    assert!(offset_of!(ActiveZones, cognitive_zone) == 0);
    assert!(offset_of!(ActiveZones, emergent_phase) == 1);
    assert!(offset_of!(ActiveZones, zone_override_active) == 2);
    assert!(offset_of!(ActiveZones, subcognitive_task_count) == 3);
    assert!(offset_of!(ActiveZones, subcognitive_flags) == 4);
    assert!(offset_of!(ActiveZones, cognitive_task_id) == 8);
    assert!(offset_of!(ActiveZones, subcognitive_task_ids) == 16);
    assert!(offset_of!(ActiveZones, zone_entered_at) == 80);
    assert!(offset_of!(ActiveZones, zone_duration_ms) == 88);
};
