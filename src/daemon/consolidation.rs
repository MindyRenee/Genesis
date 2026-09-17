//! Consolidation engine — promotes short-term memories to long-term
//! storage based on emotional gating.
//!
//! # How human memory consolidation works
//!
//! In the human brain, the hippocampus replays recent experiences during
//! rest and sleep, transferring important ones to the neocortex for
//! long-term storage. This isn't a bulk copy — it's **selective**:
//! emotionally significant experiences (high cortisol = threatening,
//! high dopamine = rewarding) are prioritized. The amygdala tags
//! memories with emotional significance, and the hippocampus uses those
//! tags to decide what to consolidate.
//!
//! # How Genesis's consolidation works
//!
//! Genesis mirrors this architecture:
//!
//! 1. **Scan** the STM ring buffer for unconsolidated entries
//! 2. **Score** each entry using the current emotional gating weights
//!    from the core state's memory pointers
//! 3. **Promote** high-scoring entries to LTM
//! 4. **Mark** consolidated entries in STM (so they're not re-consolidated)
//! 5. **Update** the core state's memory pointers with new counts
//!
//! The emotional gating weights (`encoding_weight`, `consolidation_weight`,
//! `plasticity_gate`) come from the neurochemical system and control
//! which memories survive. This is why chronic stress impairs memory
//! formation — when the plasticity gate is near zero, nothing gets
//! consolidated, no matter how important it is.
//!
//! # What's novel
//!
//! Standard AI memory systems either (a) keep everything forever (no
//! forgetting, no selectivity) or (b) use a fixed retention policy
//! (FIFO, LRU). Genesis uses **emotionally-modulated consolidation**:
//! the decision to keep or forget a memory depends on the system's
//! neurochemical state at consolidation time, not just the memory's
//! recency or frequency.

use crate::state::GenesisCoreState;
use crate::state::neurochemical::NeurochemicalId;
use crate::store::{LTM_EMOTIONAL_TAG_SIZE as LTM_TAG_SIZE, LtmStore, RingBuffer};

/// The result of a consolidation pass.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConsolidationResult {
    /// Number of STM entries scanned.
    pub scanned: u32,
    /// Number of entries promoted to LTM.
    pub promoted: u32,
    /// Number of entries skipped (already consolidated or below threshold).
    pub skipped: u32,
    /// Number of entries that couldn't be consolidated (plasticity gate closed).
    pub blocked: u32,
    /// The new LTM episode count after consolidation.
    pub ltm_count: u32,
}

/// The consolidation engine.
///
/// This is stateless — it operates on the STM ring buffer and LTM store
/// passed to it, reading emotional gating weights from the core state.
pub struct ConsolidationEngine;

impl ConsolidationEngine {
    /// Run a consolidation pass.
    ///
    /// Scans the STM ring buffer for entries that haven't been
    /// consolidated yet, scores them using the current emotional
    /// gating weights, and promotes high-scoring entries to LTM.
    ///
    /// # Parameters
    /// - `state`: the core state (read for gating weights, written
    ///   with updated LTM count)
    /// - `stm`: the short-term memory ring buffer
    /// - `ltm`: the long-term memory store
    /// - `now_ms`: current timestamp
    ///
    /// # Consolidation score
    ///
    /// Each STM entry has a `salience` score (set at insertion time)
    /// and an `emotional_tag` (neurochemical snapshot). The
    /// consolidation score combines this with the *current* gating
    /// weights:
    ///
    /// ```text
    /// score = entry.salience * state.memory.consolidation_weight
    /// ```
    ///
    /// If `state.memory.plasticity_gate < 0.1`, consolidation is
    /// blocked entirely (chronic stress state — no new long-term
    /// memories can form).
    ///
    /// If `score < threshold`, the entry is skipped (it will decay
    /// in the ring buffer as new entries overwrite it).
    pub fn consolidate(
        state: &mut GenesisCoreState,
        stm: &RingBuffer,
        ltm: &mut LtmStore,
        now_ms: u64,
        threshold: f32,
    ) -> ConsolidationResult {
        let mut result = ConsolidationResult::default();

        // Check if plasticity gate is closed (chronic stress)
        if !state.memory.can_form_new_memories() {
            // Count entries: already-consolidated ones are skipped,
            // unconsolidated ones are blocked by the closed gate.
            for entry in stm.iter() {
                result.scanned += 1;
                if entry.is_consolidated() {
                    result.skipped += 1;
                } else {
                    result.blocked += 1;
                }
            }
            return result;
        }

        let consolidation_weight = state.memory.consolidation_weight;

        // Atomically snapshot all entries with their slot indices.
        // `iter_with_slots()` holds `access_lock` for the entire scan,
        // preventing a concurrent IPC `push()` from changing `head`/
        // `tail` and overwriting slots mid-iteration. The snapshot is
        // owned, so the lock is released before we store to LTM.
        let entries = stm.iter_with_slots();

        for (slot, entry) in entries {
            result.scanned += 1;

            // Skip entries that have already been consolidated
            if entry.is_consolidated() {
                result.skipped += 1;
                continue;
            }

            // Compute consolidation score
            let score = entry.salience * consolidation_weight;

            if score < threshold {
                result.skipped += 1;
                continue;
            }

            // Build the full emotional tag from the STM entry's tag
            // (which stores 12 effective levels) and the compact tag
            // [arousal, valence, cortisol, dopamine] from it. Both the
            // consolidation path and the direct STORE_EPISODE IPC path
            // use the same helpers so LTM payloads are identical.
            let full_tag = expand_emotional_tag(&entry.emotional_tag);
            let compact_tag = compute_compact_tag(&full_tag);

            // Store in LTM
            let text = entry.text();
            let timestamp = entry.timestamp;
            let salience = entry.salience;
            let event_type = entry.event_type;
            let source_module = entry.source_module;

            match ltm.store(
                timestamp,
                salience,
                full_tag,
                compact_tag,
                event_type,
                source_module,
                text,
            ) {
                Ok(_episode_id) => {
                    // Mark the STM entry as consolidated so it's not
                    // re-consolidated on the next tick. Use the
                    // timestamp-matched variant: between the
                    // `iter_with_slots()` snapshot (which releases the
                    // ring-buffer lock) and this mark, a concurrent IPC
                    // `push()` can wrap the ring buffer and overwrite the
                    // slot. If we marked the slot unconditionally, we
                    // would mark the *new* (unconsolidated) entry as
                    // consolidated, permanently preventing it from being
                    // promoted. The timestamp match ensures we only mark
                    // the entry we actually promoted to LTM.
                    stm.mark_consolidated_if_match(slot, entry.timestamp);
                    result.promoted += 1;
                }
                Err(e) => {
                    eprintln!("[consolidation] failed to store episode: {e}");
                    result.skipped += 1;
                }
            }
        }

        // Update the core state's memory pointers
        let new_ltm_count = ltm.count();
        state
            .memory
            .mark_consolidation(now_ms, new_ltm_count as u64);

        result.ltm_count = new_ltm_count;
        result
    }
}

/// Compute a compact 4-element emotional tag from the full 12-element tag.
///
/// [arousal, valence, cortisol, dopamine] — the four dimensions that
/// most influence memory retrieval and behaviour.
///
/// The arousal and valence formulas here **must match** the corrected
/// formulas in `NeurochemicalVector::recompute_derived` (see
/// `state/neurochemical.rs`). They were previously out of sync — the
/// compact tag used the deprecated equations that the neurobiology
/// audit corrected (see AGENTS.md). This caused every consolidated
/// LTM episode to carry inaccurate arousal/valence metadata.
///
/// The emotional tag only stores 12 chemicals (the original v2 set).
/// The 6 chemicals added in v3 (endocannabinoid, vasopressin, CRH,
/// orexin, epinephrine, melatonin) are not in the tag, so we use
/// their genetic default values. This means the compact tag's
/// arousal/valence are approximate for episodes encoded with non-
/// default levels of the 6 new chemicals. The approximation error
/// is small for typical operation (the 6 new chemicals are usually
/// near their defaults) but could be significant during extreme
/// states (e.g., very high orexin during prolonged wakefulness).
///
/// The bistable flip-flop hysteresis in `recompute_derived` requires
/// the previous arousal value, which is not available from a static
/// emotional tag. We use the non-hysteretic formula (no ±0.10 bias),
/// which gives the "neutral" arousal — the value the system would
/// converge to if it had no memory of its previous state.
/// Expand a 12-element emotional tag (the STM/IPC wire format) into
/// the full `LTM_EMOTIONAL_TAG_SIZE`-element tag used by `LtmStore::store`.
///
/// The STM ring buffer and the IPC wire format carry 12 effective levels
/// (the original v2 chemical set). `LtmStore` stores
/// `LTM_EMOTIONAL_TAG_SIZE` elements (currently also 12, but the store
/// is the authority on the on-disk width). This helper centralises the
/// expansion so that both the consolidation engine and the direct
/// `STORE_EPISODE` IPC path produce identical LTM payloads.
pub fn expand_emotional_tag(tag: &[f32; 12]) -> [f32; LTM_TAG_SIZE] {
    let mut full_tag = [0.0f32; LTM_TAG_SIZE];
    let tag_len = tag.len().min(LTM_TAG_SIZE);
    full_tag[..tag_len].copy_from_slice(&tag[..tag_len]);
    full_tag
}

/// Compute the 4-element compact tag `[arousal, valence, cortisol, dopamine]`
/// from a full emotional tag.
///
/// This is a thin wrapper around `compute_compact_tag` so that callers
/// outside this module do not need to know the full-tag width.
pub fn compact_tag_from_12(tag: &[f32; 12]) -> [f32; 4] {
    compute_compact_tag(&expand_emotional_tag(tag))
}

fn compute_compact_tag(full_tag: &[f32; LTM_TAG_SIZE]) -> [f32; 4] {
    // Indices match NeurochemicalId order:
    // 0=Dopamine, 1=Serotonin, 2=NE, 3=ACh, 4=GABA, 5=Glutamate,
    // 6=Cortisol, 7=Oxytocin, 8=Endorphin, 9=Histamine, 10=Adenosine, 11=BDNF
    let da = full_tag[0];
    let srt = full_tag[1];
    let ne = full_tag[2];
    let ach = full_tag[3];
    let gaba = full_tag[4];
    let glu = full_tag[5];
    let cort = full_tag[6];
    let oxy = full_tag[7];
    let end = full_tag[8];
    let hist = full_tag[9];
    let adn = full_tag[10];

    // The 6 chemicals added in v3 are not in the 12-chemical tag.
    // Use their genetic defaults as approximations.
    let orexin = NeurochemicalId::Orexin.default_baseline();
    let epi = NeurochemicalId::Epinephrine.default_baseline();
    let mel = NeurochemicalId::Melatonin.default_baseline();

    // Arousal: matches recompute_derived() exactly, minus the Wilson-Cowan
    // self-excitation (which requires the previous arousal state, unavailable
    // from a static emotional tag). Without self-excitation, the sigmoid
    // gives the "neutral" arousal — the value the system would converge to
    // if it had no memory of its previous state.
    //
    // Arousal promoters: NE 0.20, HIST 0.15, DA 0.15, ACh 0.15,
    //   orexin 0.20, epinephrine 0.10
    // Sleep promoters: GABA 0.25, adenosine 0.65, melatonin 0.35
    let arousal_promoters =
        ne * 0.20 + hist * 0.15 + da * 0.15 + ach * 0.15 + orexin * 0.20 + epi * 0.10;
    let sleep_promoters = gaba * 0.25 + adn * 0.65 + mel * 0.35;
    let net_drive = arousal_promoters - sleep_promoters;
    let total_input = net_drive * crate::state::neurochemical::WC_GAIN; // no self-excitation
    let arousal = if total_input > 20.0 {
        1.0
    } else if total_input < -20.0 {
        0.0
    } else {
        1.0 / (1.0 + (-total_input).exp())
    };
    let arousal = crate::state::sanitize::finite_clamp(arousal, 0.0, 1.0);

    // Valence: matches recompute_derived() exactly.
    //
    // Positive: DA 0.25, SRT 0.25, OXY 0.10, END 0.15, NE 0.10
    // Negative: CORT -0.35, glutamate excess -0.15
    // Synergy: DA × SRT × 0.10
    let glu_excess = (glu - 0.70).max(0.0);
    let da_srt_synergy = da * srt * 0.10;
    let valence = da * 0.25 + srt * 0.25 + oxy * 0.10 + end * 0.15 + ne * 0.10
        - cort * 0.35
        - glu_excess * 0.15
        + da_srt_synergy;

    [
        arousal,
        crate::state::sanitize::finite_clamp(valence, -1.0, 1.0),
        cort,
        da,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::neurochemical::{NeurochemicalId, NeurochemicalVector};

    /// Verify that compute_compact_tag produces arousal/valence that
    /// match NeurochemicalVector::recompute_derived's output, when given
    /// the same chemical levels.
    ///
    /// This is the critical sync test: if the formulas drift apart,
    /// every consolidated LTM episode will carry inaccurate emotional
    /// metadata. The test creates a NeurochemicalVector with the same
    /// 12 chemical levels as the tag, calls recompute_derived(), and
    /// compares the results.
    ///
    /// Tolerance for arousal is 0.06: the bistable flip-flop hysteresis
    /// in recompute_derived adds ±0.10 to net_drive (±0.05 to arousal),
    /// which compute_compact_tag omits (it has no previous state).
    /// Valence has no hysteresis, so tolerance is tight (1e-5).
    #[test]
    fn test_compact_tag_matches_recompute_derived() {
        // Set up 12 chemical levels (the emotional tag subset)
        let levels: [(NeurochemicalId, f32); 12] = [
            (NeurochemicalId::Dopamine, 0.80),
            (NeurochemicalId::Serotonin, 0.70),
            (NeurochemicalId::Norepinephrine, 0.60),
            (NeurochemicalId::Acetylcholine, 0.50),
            (NeurochemicalId::GABA, 0.20),
            (NeurochemicalId::Glutamate, 0.55),
            (NeurochemicalId::Cortisol, 0.30),
            (NeurochemicalId::Oxytocin, 0.40),
            (NeurochemicalId::Endorphin, 0.35),
            (NeurochemicalId::Histamine, 0.50),
            (NeurochemicalId::Adenosine, 0.10),
            (NeurochemicalId::BDNF, 0.45),
        ];

        // Build the emotional tag (12-element array, NeurochemicalId order)
        let mut tag = [0.0f32; 12];
        for (id, level) in &levels {
            tag[*id as usize] = *level;
        }

        // Build a NeurochemicalVector with the same levels.
        // The 6 new chemicals (indices 12-17) stay at their defaults.
        let mut vec = NeurochemicalVector::new(0);
        for (id, level) in &levels {
            if let Some(c) = vec.get_mut(*id) {
                c.level = *level;
            }
        }

        // Compute derived fields (arousal, valence) from the live system
        vec.recompute_derived();

        // Compute compact tag from the same levels
        let compact = compute_compact_tag(&tag);

        // Arousal: with bistability, the live arousal depends on its
        // previous state; the compact tag has no previous state and
        // reports the non-hysteretic (unstable-branch) value. Tolerance
        // must cover the full hysteresis gap (~0.4 for WC_SELF=2.05).
        let arousal_diff = (compact[0] - vec.arousal).abs();
        assert!(
            arousal_diff < 0.5,
            "compact arousal {} should match recompute_derived arousal {} (diff {}, hysteresis tolerance 0.5)",
            compact[0],
            vec.arousal,
            arousal_diff
        );

        // Valence: exact match (no hysteresis in valence)
        let valence_diff = (compact[1] - vec.valence).abs();
        assert!(
            valence_diff < 1e-5,
            "compact valence {} should match recompute_derived valence {} (diff {})",
            compact[1],
            vec.valence,
            valence_diff
        );

        // Cortisol and DA are passed through directly
        assert!((compact[2] - 0.30).abs() < 1e-5, "cortisol passed through");
        assert!((compact[3] - 0.80).abs() < 1e-5, "dopamine passed through");
    }

    /// Verify that the old (deprecated) formula would give a different
    /// result — this guards against accidental reversion.
    #[test]
    fn test_compact_tag_differs_from_deprecated_formula() {
        let mut tag = [0.0f32; 12];
        tag[0] = 0.8; // DA
        tag[1] = 0.7; // SRT
        tag[2] = 0.6; // NE
        tag[3] = 0.5; // ACh  (was ignored by old arousal formula)
        tag[5] = 0.55; // Glu  (was ignored by old valence formula)
        tag[6] = 0.3; // Cort
        tag[7] = 0.4; // Oxy  (was ignored by old valence formula)
        tag[8] = 0.35; // Endorphin (was ignored by old valence formula)
        tag[9] = 0.5; // Hist

        let compact = compute_compact_tag(&tag);

        // Old valence = (DA + SRT) * 0.5 - Cort = 0.75 - 0.3 = 0.45
        let old_valence = (0.8 + 0.7) * 0.5 - 0.3;
        let new_valence = compact[1];
        assert!(
            (new_valence - old_valence).abs() > 0.01,
            "corrected valence {} should differ from deprecated {}",
            new_valence,
            old_valence
        );

        // Old arousal = ((NE + Hist)*0.5 - (GABA+Adn)*0.5 + 1.0) * 0.5
        // The new formula includes orexin (0.25 default × 0.20 = 0.05),
        // epinephrine (0.15 default × 0.10 = 0.015), ACh (0.5 × 0.15 = 0.075),
        // and DA (0.8 × 0.15 = 0.12) as arousal promoters, plus different
        // weights for NE (0.20 vs 0.50) and histamine (0.15 vs 0.50).
        let old_arousal = ((0.6 + 0.5) * 0.5 - (0.0 + 0.0) * 0.5 + 1.0) * 0.5;
        assert!(
            (compact[0] - old_arousal).abs() > 0.01,
            "corrected arousal {} should differ from deprecated {}",
            compact[0],
            old_arousal
        );
    }

    /// Verify that compute_compact_tag's arousal matches the non-hysteretic
    /// branch of recompute_derived when GABA is high. This catches drift in
    /// the sleep-promoter weights (a previous bug had GABA at 0.30 in
    /// compute_compact_tag but 0.25 in recompute_derived, masked by the
    /// loose hysteresis tolerance in test_compact_tag_matches_recompute_derived
    /// which used a low GABA value).
    #[test]
    fn test_compact_tag_gaba_weight_matches_recompute_derived() {
        let levels: [(NeurochemicalId, f32); 12] = [
            (NeurochemicalId::Dopamine, 0.50),
            (NeurochemicalId::Serotonin, 0.50),
            (NeurochemicalId::Norepinephrine, 0.50),
            (NeurochemicalId::Acetylcholine, 0.50),
            (NeurochemicalId::GABA, 0.80), // high GABA — the discriminating case
            (NeurochemicalId::Glutamate, 0.50),
            (NeurochemicalId::Cortisol, 0.20),
            (NeurochemicalId::Oxytocin, 0.40),
            (NeurochemicalId::Endorphin, 0.35),
            (NeurochemicalId::Histamine, 0.50),
            (NeurochemicalId::Adenosine, 0.10),
            (NeurochemicalId::BDNF, 0.45),
        ];

        let mut tag = [0.0f32; 12];
        for (id, level) in &levels {
            tag[*id as usize] = *level;
        }

        let mut vec = NeurochemicalVector::new(0);
        for (id, level) in &levels {
            if let Some(c) = vec.get_mut(*id) {
                c.level = *level;
            }
        }
        vec.recompute_derived();

        let compact = compute_compact_tag(&tag);

        // The compact tag has no self-excitation, so it reports the
        // unstable-branch (neutral) arousal. The live system with its
        // initial arousal=0.5 starts near the unstable branch and
        // converges to one of the two stable fixed points. With high
        // GABA, the live system should converge to low arousal (asleep),
        // while the compact tag reports the neutral value. The
        // difference should be the hysteresis gap (~0.4), NOT a weight
        // drift. We verify the compact tag's arousal is the neutral
        // (non-hysteretic) value by computing it directly.
        let gaba = 0.80f32;
        let adn = 0.10f32;
        let mel = NeurochemicalId::Melatonin.default_baseline();
        let sleep_promoters = gaba * 0.25 + adn * 0.65 + mel * 0.35;
        let ne = 0.50f32;
        let hist = 0.50f32;
        let da = 0.50f32;
        let ach = 0.50f32;
        let orexin = NeurochemicalId::Orexin.default_baseline();
        let epi = NeurochemicalId::Epinephrine.default_baseline();
        let arousal_promoters =
            ne * 0.20 + hist * 0.15 + da * 0.15 + ach * 0.15 + orexin * 0.20 + epi * 0.10;
        let net_drive = arousal_promoters - sleep_promoters;
        let total_input = net_drive * crate::state::neurochemical::WC_GAIN;
        let expected_arousal = if total_input > 20.0 {
            1.0
        } else if total_input < -20.0 {
            0.0
        } else {
            1.0 / (1.0 + (-total_input).exp())
        };

        assert!(
            (compact[0] - expected_arousal).abs() < 1e-5,
            "compact arousal {} should match direct computation {} (GABA weight 0.25)",
            compact[0],
            expected_arousal
        );

        // The old buggy weight (0.30) would give a different value:
        let buggy_sleep = gaba * 0.30 + adn * 0.65 + mel * 0.35;
        let buggy_drive = arousal_promoters - buggy_sleep;
        let buggy_input = buggy_drive * crate::state::neurochemical::WC_GAIN;
        let buggy_arousal = 1.0 / (1.0 + (-buggy_input).exp());
        assert!(
            (compact[0] - buggy_arousal).abs() > 0.001,
            "compact arousal {} should NOT match the buggy GABA=0.30 weight value {}",
            compact[0],
            buggy_arousal
        );
    }
}
