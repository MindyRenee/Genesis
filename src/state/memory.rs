//! Memory pointers — references from the core state to the various
//! memory stores, plus emotional gating weights.
//!
//! # Emotional gating — what makes this novel
//!
//! The memory pointers don't just reference stores — they carry
//! **gating weights** that control how the memory system behaves:
//!
//! - **`encoding_weight`**: how aggressively new memories are encoded.
//!   Driven by dopamine (reward salience), norepinephrine (novelty),
//!   and acetylcholine (signal-to-noise). When this is high, Genesis
//!   learns fast. When low, experiences slip by unrecorded.
//!
//! - **`consolidation_weight`**: how strongly short-term memories are
//!   promoted to long-term storage. Driven by cortisol (stress
//!   prioritizes consolidation of threatening events) and ACh. This
//!   is why traumatic experiences form vivid long-term memories.
//!
//! - **`retrieval_weight`**: how easily existing memories can be
//!   recalled. Driven by ACh and serotonin, suppressed by cortisol.
//!   This is why stress impairs recall — "tip of the tongue" during
//!   exams.
//!
//! - **`plasticity_gate`**: whether the system can form new structural
//!   connections at all. Driven by BDNF, suppressed by cortisol.
//!   When this is near zero, no new long-term memories can form
//!   regardless of encoding weight — this is the neurochemistry of
//!   chronic stress impairing learning.
//!
//! These weights are **derived** from the neurochemical system and
//! **stored** in the memory pointers. This means the memory system's
//! interface *structurally embeds* emotional influence — it's not a
//! bolt-on, it's in the type signature. Any module that reads memory
//! pointers sees the gating weights and cannot ignore them.

/// Maximum number of items in the working set (active manipulation).
pub const WORKING_SET_SIZE: usize = 4;

/// Pointers to all memory stores, with emotional gating weights.
///
/// ## Layout (120 bytes)
/// ```text
/// offset  field                  type    notes
/// ------  -----                  ----    -----
///   0     stm_head               u64     ring buffer write index
///   8     stm_tail               u64     ring buffer read index
///  16     stm_count              u32     items currently in buffer
///  20     stm_capacity           u32     ring buffer total capacity
///  24     ltm_store_handle       u64     handle to episodic store
///  32     ltm_episode_count      u64     total episodes in LTM
///  40     ltm_last_consolidated  u64     ms timestamp of last consolidation
///  48     procedural_handle      u64     handle to procedural memory
///  56     semantic_handle        u64     handle to semantic memory
///  64     encoding_weight        f32     how fast new memories encode
///  68     consolidation_weight   f32     how fast STM → LTM promotion
///  72     retrieval_weight       f32     how easily memories are recalled
///  76     plasticity_gate        f32     whether new traces can form at all
///  80     working_set            [u64;4] currently active item IDs
/// 112    working_set_count      u8      how many working-set slots are used
/// 113    _pad                   [u8;7]  alignment
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MemoryPointers {
    // --- Short-term memory (ring buffer) ---
    /// Next write index in the STM ring buffer.
    pub stm_head: u64,
    /// Oldest unread index in the STM ring buffer.
    pub stm_tail: u64,
    /// Number of entries currently in the STM ring buffer.
    pub stm_count: u32,
    /// Capacity (slot count) of the STM ring buffer.
    pub stm_capacity: u32,

    // --- Long-term episodic memory ---
    /// Opaque handle to the LTM store.
    pub ltm_store_handle: u64,
    /// Number of episodes stored in LTM.
    pub ltm_episode_count: u64,
    /// Timestamp of the last STM → LTM consolidation.
    pub ltm_last_consolidated: u64,

    // --- Procedural & semantic memory ---
    /// Opaque handle to the procedural memory store.
    pub procedural_handle: u64,
    /// Opaque handle to the semantic memory store.
    pub semantic_handle: u64,

    // --- Emotional gating weights (derived from neurochemistry) ---
    /// How aggressively new memories are being encoded right now.
    /// Driven by dopamine + norepinephrine + acetylcholine.
    pub encoding_weight: f32,
    /// How strongly short-term memories are promoted to long-term.
    /// Driven by cortisol + acetylcholine.
    pub consolidation_weight: f32,
    /// How easily existing memories can be recalled.
    /// Driven by acetylcholine + serotonin, suppressed by cortisol.
    pub retrieval_weight: f32,
    /// Whether new structural connections can form at all.
    /// Driven by BDNF, suppressed by cortisol. Near zero = no new
    /// long-term memories possible (chronic stress state).
    pub plasticity_gate: f32,

    // --- Working set (active manipulation) ---
    /// Currently active item IDs in the working set (0 = unused slot).
    pub working_set: [u64; WORKING_SET_SIZE],
    /// Number of working-set slots currently in use.
    pub working_set_count: u8,
    /// Alignment padding.
    pub _pad: [u8; 7],
}

impl MemoryPointers {
    /// Initialise with all stores unassigned and neutral gating weights.
    pub fn new() -> Self {
        Self {
            stm_head: 0,
            stm_tail: 0,
            stm_count: 0,
            stm_capacity: 0,
            ltm_store_handle: 0,
            ltm_episode_count: 0,
            ltm_last_consolidated: 0,
            procedural_handle: 0,
            semantic_handle: 0,
            // Neutral weights — will be overwritten by first
            // neurochemical sync.
            encoding_weight: 0.5,
            consolidation_weight: 0.5,
            retrieval_weight: 0.5,
            plasticity_gate: 0.5,
            working_set: [0; WORKING_SET_SIZE],
            working_set_count: 0,
            _pad: [0; 7],
        }
    }

    // ─── Working set operations ──────────────────────────────────

    /// Add an item to the working set. Returns `false` if `item_id` is 0
    /// or no free slot is available.
    pub fn push_working(&mut self, item_id: u64) -> bool {
        if item_id == 0 {
            return false;
        }
        if self.working_set.contains(&item_id) {
            return false;
        }
        if let Some(slot) = self.working_set.iter_mut().find(|id| **id == 0) {
            *slot = item_id;
            self.working_set_count = self.working_set_count.saturating_add(1);
            true
        } else {
            false
        }
    }

    /// Remove an item from the working set (no-op if not present).
    pub fn pop_working(&mut self, item_id: u64) {
        if let Some(slot) = self.working_set.iter_mut().find(|id| **id == item_id) {
            *slot = 0;
            self.working_set_count = self.working_set_count.saturating_sub(1);
        }
    }

    // ─── Store handle assignment ─────────────────────────────────

    /// Record a consolidation event: update the last-consolidated timestamp
    /// and the new LTM episode count.
    pub fn mark_consolidation(&mut self, now_ms: u64, new_episode_count: u64) {
        self.ltm_last_consolidated = now_ms;
        self.ltm_episode_count = new_episode_count;
    }

    // ─── Emotional gating ────────────────────────────────────────

    /// Update the emotional gating weights from the neurochemical
    /// system's effective levels.
    ///
    /// This is called by the dynamics engine after each neurochemical
    /// tick. The memory system then reads these weights to control
    /// its behaviour — it doesn't need to know about neurochemistry,
    /// it just sees gating weights in its own interface.
    ///
    /// # Parameters
    /// - `dopamine_eff`: effective dopamine level
    /// - `serotonin_eff`: effective serotonin level
    /// - `norepinephrine_eff`: effective norepinephrine level
    /// - `acetylcholine_eff`: effective acetylcholine level
    /// - `cortisol_eff`: effective cortisol level
    /// - `bdnf_eff`: effective BDNF level
    pub fn update_gating(
        &mut self,
        dopamine_eff: f32,
        serotonin_eff: f32,
        norepinephrine_eff: f32,
        acetylcholine_eff: f32,
        cortisol_eff: f32,
        bdnf_eff: f32,
    ) {
        // Encoding: reward salience (DA) + novelty (NE) + signal-to-noise (ACh)
        self.encoding_weight = crate::state::sanitize::finite_clamp(
            dopamine_eff * 0.4 + norepinephrine_eff * 0.3 + acetylcholine_eff * 0.3,
            0.0,
            1.0,
        );

        // Consolidation: stress prioritizes consolidation (CORT) + encoding quality (ACh)
        // High cortisol means traumatic/stressful memories consolidate more strongly.
        // Cortisol boosts consolidation (acute stress enhances emotional memory
        // formation), ACh boosts encoding quality, and there's a baseline so
        // consolidation never fully stops.
        self.consolidation_weight = crate::state::sanitize::finite_clamp(
            cortisol_eff * 0.3 + acetylcholine_eff * 0.4 + 0.3,
            0.0,
            1.0,
        );

        // Retrieval: ACh + serotonin promote, cortisol impairs.
        //
        // Stress impairs recall through multiple mechanisms:
        // - Cortisol suppresses hippocampal retrieval pathways
        // - GC receptor activation interferes with working memory
        // - The "tip-of-the-tongue" phenomenon during exams is a
        //   classic example of stress-induced retrieval failure
        //
        // The cortisol suppression is weighted at 0.35 (up from 0.20)
        // to match the strong impairment seen in stress studies
        // (de Quervain et al., 2009). Weights normalized to sum to 1.0:
        // ACh 0.40 + SRT 0.25 + CORT 0.35 = 1.0.
        self.retrieval_weight = crate::state::sanitize::finite_clamp(
            acetylcholine_eff * 0.40 + serotonin_eff * 0.25 + (1.0 - cortisol_eff) * 0.35,
            0.0,
            1.0,
        );

        // Plasticity: BDNF promotes, cortisol suppresses
        self.plasticity_gate =
            crate::state::sanitize::finite_clamp(bdnf_eff * (1.0 - cortisol_eff * 0.5), 0.0, 1.0);
    }

    /// Whether the system can form new long-term memories right now.
    /// Returns false when plasticity gate is near zero (chronic stress).
    pub fn can_form_new_memories(&self) -> bool {
        self.plasticity_gate > 0.1
    }

    /// Whether the system is in a high-encoding state (learning fast).
    pub fn is_high_encoding(&self) -> bool {
        self.encoding_weight > 0.6
    }

    /// Whether recall is impaired (stress state).
    pub fn is_retrieval_impaired(&self) -> bool {
        self.retrieval_weight < 0.3
    }
}

impl Default for MemoryPointers {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────
//  Compile-time layout assertions
// ─────────────────────────────────────────────────────────────────

const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<MemoryPointers>() == 120);
    assert!(offset_of!(MemoryPointers, stm_head) == 0);
    assert!(offset_of!(MemoryPointers, stm_tail) == 8);
    assert!(offset_of!(MemoryPointers, stm_count) == 16);
    assert!(offset_of!(MemoryPointers, stm_capacity) == 20);
    assert!(offset_of!(MemoryPointers, ltm_store_handle) == 24);
    assert!(offset_of!(MemoryPointers, ltm_episode_count) == 32);
    assert!(offset_of!(MemoryPointers, ltm_last_consolidated) == 40);
    assert!(offset_of!(MemoryPointers, procedural_handle) == 48);
    assert!(offset_of!(MemoryPointers, semantic_handle) == 56);
    assert!(offset_of!(MemoryPointers, encoding_weight) == 64);
    assert!(offset_of!(MemoryPointers, consolidation_weight) == 68);
    assert!(offset_of!(MemoryPointers, retrieval_weight) == 72);
    assert!(offset_of!(MemoryPointers, plasticity_gate) == 76);
    assert!(offset_of!(MemoryPointers, working_set) == 80);
    assert!(offset_of!(MemoryPointers, working_set_count) == 112);
};
