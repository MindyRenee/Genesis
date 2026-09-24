//! Tests for the subcognitive daemon — consolidation, association,
//! dreaming, and the tick loop.
//!
//! These are integration tests that exercise the full system:
//! core state + STM + LTM + neurochemistry + consolidation.

use genesis::daemon::*;
use genesis::state::*;
use genesis::store::*;
use std::path::{Path, PathBuf};

fn cleanup_all(state_path: &Path, stm_path: &Path, ltm_base: &Path) {
    let _ = std::fs::remove_file(state_path);
    let _ = std::fs::remove_file(stm_path);
    let _ = std::fs::remove_file(ltm_base.with_extension("idx"));
    let _ = std::fs::remove_file(ltm_base.with_extension("dat"));
}

fn make_emotional_tag() -> [f32; 12] {
    [
        0.50, 0.55, 0.40, 0.45, 0.50, 0.60, 0.20, 0.35, 0.30, 0.45, 0.10, 0.40,
    ]
}

/// Set up a full system: state file, STM ring buffer, LTM store.
struct TestSystem {
    state_path: PathBuf,
    stm_path: PathBuf,
    ltm_base: PathBuf,
    mmap: MmapState,
    stm: RingBuffer,
    ltm: LtmStore,
}

impl TestSystem {
    fn new(test_name: &str, stm_capacity: u32, ltm_capacity: u32) -> Self {
        let state_path = std::env::temp_dir().join(format!(
            "genesis_daemon_{test_name}_state_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let stm_path = std::env::temp_dir().join(format!(
            "genesis_daemon_{test_name}_stm_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let ltm_base = std::env::temp_dir().join(format!(
            "genesis_daemon_{test_name}_ltm_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let mmap = MmapState::create(&state_path, 1, 1000).expect("create state");
        let stm = RingBuffer::create(&stm_path, stm_capacity).expect("create stm");
        let ltm = LtmStore::create(&ltm_base, ltm_capacity).expect("create ltm");

        Self {
            state_path,
            stm_path,
            ltm_base,
            mmap,
            stm,
            ltm,
        }
    }
}

impl Drop for TestSystem {
    fn drop(&mut self) {
        cleanup_all(&self.state_path, &self.stm_path, &self.ltm_base);
    }
}

// ─── Consolidation ────────────────────────────────────────────

#[test]
fn test_consolidation_promotes_high_salience() {
    let mut sys = TestSystem::new("consol_high", 16, 64);

    // Push high-salience entries to STM
    sys.stm.push(RingBufferEntry::new(
        1000,
        EventType::UserInput,
        4,
        0.90,
        make_emotional_tag(),
        "Important user message about water crisis",
    ));
    sys.stm.push(RingBufferEntry::new(
        2000,
        EventType::Observation,
        6,
        0.85,
        make_emotional_tag(),
        "Critical data point observed",
    ));

    // Set up good neurochemistry for consolidation
    sys.mmap
        .modify(3000, |state| {
            state
                .neurochemicals
                .get_mut(NeurochemicalId::BDNF)
                .unwrap()
                .level = 0.60;
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Cortisol)
                .unwrap()
                .level = 0.15;
            state.neurochemicals.recompute_derived();
            state.sync_neurochemistry_to_state();
        })
        .expect("modify");

    // Run consolidation
    let result = ConsolidationEngine::consolidate(
        // SAFETY: single-threaded test — no concurrent access.
        unsafe { sys.mmap.state_mut() },
        &sys.stm,
        &mut sys.ltm,
        3000,
        0.15,
    );

    assert!(
        result.promoted > 0,
        "high-salience entries should be promoted"
    );
    assert_eq!(sys.ltm.count(), result.promoted);
}

#[test]
fn test_consolidation_skips_low_salience() {
    let mut sys = TestSystem::new("consol_low", 16, 64);

    // Push low-salience entries
    sys.stm.push(RingBufferEntry::new(
        1000,
        EventType::Internal,
        0,
        0.05,
        make_emotional_tag(),
        "Routine internal event",
    ));

    // Good neurochemistry
    sys.mmap
        .modify(2000, |state| {
            state
                .neurochemicals
                .get_mut(NeurochemicalId::BDNF)
                .unwrap()
                .level = 0.60;
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Cortisol)
                .unwrap()
                .level = 0.15;
            state.neurochemicals.recompute_derived();
            state.sync_neurochemistry_to_state();
        })
        .expect("modify");

    let result = ConsolidationEngine::consolidate(
        // SAFETY: single-threaded test — no concurrent access.
        unsafe { sys.mmap.state_mut() },
        &sys.stm,
        &mut sys.ltm,
        2000,
        0.15,
    );

    assert_eq!(result.promoted, 0, "low-salience entries should be skipped");
    assert!(result.skipped > 0);
}

#[test]
fn test_consolidation_blocked_by_chronic_stress() {
    let mut sys = TestSystem::new("consol_blocked", 16, 64);

    // Push high-salience entries
    sys.stm.push(RingBufferEntry::new(
        1000,
        EventType::UserInput,
        4,
        0.95,
        make_emotional_tag(),
        "Very important message",
    ));

    // Induce chronic stress — plasticity gate near zero
    sys.mmap
        .modify(2000, |state| {
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Cortisol)
                .unwrap()
                .level = 0.90;
            state
                .neurochemicals
                .get_mut(NeurochemicalId::BDNF)
                .unwrap()
                .level = 0.05;
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Serotonin)
                .unwrap()
                .level = 0.15;
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Acetylcholine)
                .unwrap()
                .level = 0.15;
            state.neurochemicals.recompute_derived();
            state.sync_neurochemistry_to_state();
        })
        .expect("modify");

    let result = ConsolidationEngine::consolidate(
        // SAFETY: single-threaded test — no concurrent access.
        unsafe { sys.mmap.state_mut() },
        &sys.stm,
        &mut sys.ltm,
        2000,
        0.15,
    );

    assert_eq!(
        result.promoted, 0,
        "chronic stress should block consolidation"
    );
    assert!(result.blocked > 0, "entries should be marked as blocked");
}

#[test]
fn test_consolidation_preserves_emotional_tags() {
    let mut sys = TestSystem::new("consol_tags", 16, 64);

    let mut tag = make_emotional_tag();
    tag[0] = 0.90; // high dopamine
    tag[6] = 0.75; // high cortisol

    sys.stm.push(RingBufferEntry::new(
        1000,
        EventType::NeurochemicalShift,
        3,
        0.80,
        tag,
        "Stressful but rewarding event",
    ));

    // Good neurochemistry for consolidation
    sys.mmap
        .modify(2000, |state| {
            state
                .neurochemicals
                .get_mut(NeurochemicalId::BDNF)
                .unwrap()
                .level = 0.60;
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Cortisol)
                .unwrap()
                .level = 0.15;
            state.neurochemicals.recompute_derived();
            state.sync_neurochemistry_to_state();
        })
        .expect("modify");

    ConsolidationEngine::consolidate(
        // SAFETY: single-threaded test — no concurrent access.
        unsafe { sys.mmap.state_mut() },
        &sys.stm,
        &mut sys.ltm,
        2000,
        0.15,
    );

    // Retrieve the consolidated episode and verify tags
    let ep = sys.ltm.retrieve(1).expect("retrieve");
    assert_eq!(ep.full_emotional_tag[0], 0.90, "dopamine tag preserved");
    assert_eq!(ep.full_emotional_tag[6], 0.75, "cortisol tag preserved");
    assert_eq!(ep.text, "Stressful but rewarding event");
}

// ─── Association ──────────────────────────────────────────────

#[test]
fn test_association_finds_similar_memories() {
    let mut sys = TestSystem::new("assoc_find", 16, 64);

    // Store several related memories
    sys.ltm
        .store(
            0,
            0.5,
            make_emotional_tag(),
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "Great Salt Lake water levels are declining",
        )
        .expect("store");
    sys.ltm
        .store(
            0,
            0.5,
            make_emotional_tag(),
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "Data centers consume water for cooling",
        )
        .expect("store");
    sys.ltm
        .store(
            0,
            0.5,
            make_emotional_tag(),
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "Great Salt Lake conservation needs attention",
        )
        .expect("store");

    // Find associations for the first episode
    let hash = sys.ltm.index_entry_ref(0).association_hash;
    let associations = AssociationEngine::find_associations(&sys.ltm, hash, 10);

    assert!(!associations.is_empty(), "should find associations");
    // The "Great Salt Lake conservation" episode should be closer than
    // the "Data centers" episode
    let consol_id = associations.iter().find(|(id, _)| *id == 3);
    assert!(
        consol_id.is_some(),
        "conservation episode should be associated"
    );
}

#[test]
fn test_association_creates_meta_memories() {
    let mut sys = TestSystem::new("assoc_meta", 16, 64);

    // Store related memories
    sys.ltm
        .store(
            0,
            0.5,
            make_emotional_tag(),
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "Great Salt Lake water crisis",
        )
        .expect("store");
    sys.ltm
        .store(
            0,
            0.5,
            make_emotional_tag(),
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "Great Salt Lake drying up",
        )
        .expect("store");

    let count_before = sys.ltm.count();

    // Run association
    let result = AssociationEngine::associate(&mut sys.ltm, &[1, 2], 5000);

    assert!(result.new_associations > 0, "should create meta-memories");
    assert!(sys.ltm.count() > count_before, "LTM should have grown");
}

// ─── Dreaming ─────────────────────────────────────────────────

#[test]
fn test_dreaming_runs_with_memories() {
    let mut sys = TestSystem::new("dream", 16, 64);

    // Store several memories so dreaming has material to work with
    for i in 0..10 {
        sys.ltm
            .store(
                i * 1000,
                0.5,
                make_emotional_tag(),
                [0.4, 0.3, 0.2, 0.5],
                0,
                0,
                &format!("Memory episode number {i} about various topics"),
            )
            .expect("store");
    }

    let _count_before = sys.ltm.count();

    // Run dreaming
    let result = AssociationEngine::dream(&mut sys.ltm, 100_000, 5);

    // Should have traversed a chain
    assert!(!result.chain.is_empty(), "dreaming should produce a chain");
    assert!(result.chain.len() <= 6, "chain should respect max hops");

    // May or may not produce insights (depends on whether it finds
    // novel connections), but the chain should be valid
    for &id in &result.chain {
        assert!(id > 0, "chain IDs should be valid");
    }
}

#[test]
fn test_dream_insight_embeds_episode_text() {
    let mut sys = TestSystem::new("dream_embed", 16, 64);

    // Store memories with distinctive text so we can verify embedding
    sys.ltm
        .store(
            1000,
            0.5,
            make_emotional_tag(),
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "Learning about quantum entanglement physics",
        )
        .expect("store");
    sys.ltm
        .store(
            2000,
            0.5,
            make_emotional_tag(),
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "Learning about classical mechanics motion",
        )
        .expect("store");
    // Store several similar memories to create association chains
    for i in 0..8 {
        sys.ltm
            .store(
                3000 + i * 1000,
                0.5,
                make_emotional_tag(),
                [0.4, 0.3, 0.2, 0.5],
                0,
                0,
                &format!("Physics topic number {i} about energy and momentum"),
            )
            .expect("store");
    }

    // Run dreaming with enough hops to potentially find insights
    let _result = AssociationEngine::dream(&mut sys.ltm, 100_000, 8);

    // Search for dream insight meta-memories and verify they contain
    // embedded text (tab-delimited after the structural header)
    let recent = sys.ltm.recent_episodes(50, Some(9));
    let dream_insights: Vec<_> = recent
        .iter()
        .filter(|ep| ep.text.starts_with("[dream-insight]"))
        .collect();

    if !dream_insights.is_empty() {
        for insight in &dream_insights {
            // The new format must contain a tab delimiter followed by
            // the embedded episode text
            assert!(
                insight.text.contains('\t'),
                "dream insight must embed episode text (tab-delimited): {}",
                insight.text
            );
            let parts: Vec<&str> = insight.text.splitn(3, '\t').collect();
            assert!(
                parts.len() >= 2,
                "dream insight must have at least start text embedded"
            );
            // The embedded text should not be empty (episodes exist at dream time)
            assert!(
                !parts[1].is_empty(),
                "embedded start text must not be empty: {}",
                insight.text
            );
        }
    }
    // If no dream insights were produced, the test still passes —
    // insights are probabilistic. The format check only runs when
    // they exist.
}

#[test]
fn test_association_embeds_episode_text() {
    let mut sys = TestSystem::new("assoc_embed", 16, 64);

    // Store two similar memories
    sys.ltm
        .store(
            0,
            0.5,
            make_emotional_tag(),
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "Great Salt Lake water crisis",
        )
        .expect("store");
    sys.ltm
        .store(
            0,
            0.5,
            make_emotional_tag(),
            [0.4, 0.3, 0.2, 0.5],
            0,
            0,
            "Great Salt Lake drying up",
        )
        .expect("store");

    // Run association
    let result = AssociationEngine::associate(&mut sys.ltm, &[1, 2], 5000);
    assert!(result.new_associations > 0, "should create associations");

    // Find the association meta-memory and verify it contains embedded text
    let recent = sys.ltm.recent_episodes(50, Some(0));
    let associations: Vec<_> = recent
        .iter()
        .filter(|ep| ep.text.starts_with("[association]"))
        .collect();

    assert!(
        !associations.is_empty(),
        "should have at least one association meta-memory"
    );
    for assoc in &associations {
        assert!(
            assoc.text.contains('\t'),
            "association must embed episode text (tab-delimited): {}",
            assoc.text
        );
        let parts: Vec<&str> = assoc.text.splitn(3, '\t').collect();
        assert!(
            parts.len() >= 3,
            "association must have both endpoint texts embedded: {}",
            assoc.text
        );
        assert!(
            !parts[1].is_empty() && !parts[2].is_empty(),
            "embedded texts must not be empty: {}",
            assoc.text
        );
        // Verify the embedded text matches the original episode text
        assert!(
            parts[1].contains("Salt Lake"),
            "embedded text A should contain original episode text: {}",
            parts[1]
        );
        assert!(
            parts[2].contains("Salt Lake"),
            "embedded text B should contain original episode text: {}",
            parts[2]
        );
    }
}

// ─── Tick loop ────────────────────────────────────────────────

#[test]
fn test_tick_loop_advances_neurochemistry() {
    let mut sys = TestSystem::new("tick_neuro", 16, 64);
    let mut tick_loop = TickLoop::new();

    let heartbeat_before = sys.mmap.read().header.heartbeat;

    // Run a few ticks
    for _ in 0..5 {
        tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);
    }

    let heartbeat_after = sys.mmap.read().header.heartbeat;
    assert!(
        heartbeat_after > heartbeat_before,
        "heartbeat should advance with ticks"
    );
}

#[test]
fn test_advance_neuro_circadian_phase_tracks_dt() {
    // The circadian phase must advance at exactly dt/86400 per
    // advance_neuro call — the contract the cognitive mind relies on
    // when it passes real elapsed time as dt. If this rate were
    // wrong, its "day" would not be 24 hours and melatonin would
    // peak at arbitrary times of day.
    let sys = TestSystem::new("advance_dt", 16, 64);
    let mut tick_loop = TickLoop::new();

    let phase_before = sys.mmap.read().neurochemicals.circadian_phase();

    // Two advances of 10 simulated seconds each.
    tick_loop.advance_neuro(&sys.mmap, 10.0);
    tick_loop.advance_neuro(&sys.mmap, 10.0);

    let phase_after = sys.mmap.read().neurochemicals.circadian_phase();

    // 20 seconds of an 86400-second day. Compare as circular
    // distance to be robust against phase wraparound at 1.0.
    let expected = 20.0f32 / 86_400.0;
    let delta = (phase_after - phase_before).rem_euclid(1.0);
    let dist = delta.min(1.0 - delta);
    assert!(
        (dist - expected).abs() < 1e-6,
        "advance_neuro(dt=10) x2 must advance the phase by 20/86400 \
         (before {phase_before}, after {phase_after})"
    );
}

#[test]
fn test_tick_loop_consolidates_stm() {
    let mut sys = TestSystem::new("tick_consol", 16, 64);
    let mut tick_loop = TickLoop::new();

    // Push entries to STM
    sys.stm.push(RingBufferEntry::new(
        1000,
        EventType::UserInput,
        4,
        0.90,
        make_emotional_tag(),
        "Important event to consolidate",
    ));

    // Ensure good neurochemistry
    sys.mmap
        .modify(1000, |state| {
            state
                .neurochemicals
                .get_mut(NeurochemicalId::BDNF)
                .unwrap()
                .level = 0.60;
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Cortisol)
                .unwrap()
                .level = 0.15;
            state.neurochemicals.recompute_derived();
            state.sync_neurochemistry_to_state();
        })
        .expect("modify");

    assert_eq!(sys.ltm.count(), 0);

    // Run a tick — should consolidate
    let result = tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);

    assert!(
        result.consolidated > 0,
        "tick should consolidate high-salience STM entries"
    );
    assert!(sys.ltm.count() > 0, "LTM should have entries after tick");
}

#[test]
fn test_tick_loop_updates_manifest() {
    let mut sys = TestSystem::new("tick_manifest", 16, 64);
    let mut tick_loop = TickLoop::new();

    tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);

    let state = sys.mmap.read();
    let module = state.manifest.get(ModuleId::Subcognitive).expect("module");
    assert_eq!(
        module.status(),
        ModuleStatus::Running,
        "subcognitive module should be Running after tick"
    );
    assert!(module.last_heartbeat > 0, "heartbeat should be set");
}

#[test]
fn test_tick_loop_sets_subcognitive_flags() {
    let mut sys = TestSystem::new("tick_flags", 16, 64);
    let mut tick_loop = TickLoop::new();

    // Push a high-salience entry so consolidation has work to do.
    sys.stm.push(RingBufferEntry::new(
        1000,
        EventType::UserInput,
        4,
        0.90,
        make_emotional_tag(),
        "Important user message",
    ));

    tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);

    let state = sys.mmap.read();
    assert!(
        state
            .zones
            .has_flag(subcognitive_flag::MEMORY_CONSOLIDATION),
        "consolidation flag should be set when consolidation ran"
    );
}

#[test]
fn test_tick_loop_multiple_ticks_stable() {
    let mut sys = TestSystem::new("tick_stable", 16, 64);
    let mut tick_loop = TickLoop::new();

    // Run 50 ticks — should not crash or corrupt state
    for _ in 0..50 {
        let result = tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);
        assert!(result.tick_number > 0);
    }

    // State should still be valid
    assert!(sys.mmap.read().verify().is_ok());
    assert!(sys.mmap.read().verify_checksum().is_ok());
    assert_eq!(tick_loop.tick_count(), 50);
}

#[test]
fn test_tick_loop_stress_blocks_consolidation() {
    let mut sys = TestSystem::new("tick_stress", 16, 64);
    let mut tick_loop = TickLoop::new();

    // Push high-salience entries
    sys.stm.push(RingBufferEntry::new(
        1000,
        EventType::UserInput,
        4,
        0.95,
        make_emotional_tag(),
        "Important event during stress",
    ));

    // Induce chronic stress
    for _ in 0..20 {
        sys.mmap
            .modify(1000, |state| {
                state
                    .neurochemicals
                    .get_mut(NeurochemicalId::Cortisol)
                    .unwrap()
                    .level = 0.90;
                state
                    .neurochemicals
                    .get_mut(NeurochemicalId::BDNF)
                    .unwrap()
                    .level = 0.05;
                state
                    .neurochemicals
                    .get_mut(NeurochemicalId::Serotonin)
                    .unwrap()
                    .level = 0.15;
                state
                    .neurochemicals
                    .get_mut(NeurochemicalId::Acetylcholine)
                    .unwrap()
                    .level = 0.15;
                state.neuro_tick();
            })
            .expect("modify");
    }

    let ltm_before = sys.ltm.count();

    // Run a tick — consolidation should be blocked
    let result = tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);

    assert_eq!(
        result.consolidated, 0,
        "chronic stress should block consolidation during tick"
    );
    assert_eq!(
        sys.ltm.count(),
        ltm_before,
        "LTM should not grow under chronic stress"
    );
}

// ─── Full lifecycle ───────────────────────────────────────────

#[test]
fn test_full_subcognitive_lifecycle() {
    let mut sys = TestSystem::new("lifecycle", 32, 128);
    let mut tick_loop = TickLoop::new();

    // 1. Push several events to STM with varying salience
    sys.stm.push(RingBufferEntry::new(
        1000,
        EventType::UserInput,
        4,
        0.90,
        make_emotional_tag(),
        "User asked about water data",
    ));
    sys.stm.push(RingBufferEntry::new(
        2000,
        EventType::Observation,
        6,
        0.30,
        make_emotional_tag(),
        "Routine system check",
    ));
    sys.stm.push(RingBufferEntry::new(
        3000,
        EventType::Output,
        4,
        0.75,
        make_emotional_tag(),
        "Responded with analysis",
    ));

    // 2. Run ticks — should consolidate high-salience entries
    for _ in 0..10 {
        tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);
    }

    assert!(
        sys.ltm.count() > 0,
        "some entries should be consolidated to LTM"
    );

    // 3. Verify consolidated entries are retrievable
    let ep = sys.ltm.retrieve(1).expect("first episode should exist");
    assert_eq!(ep.text, "User asked about water data");

    // 4. Run more ticks for association
    for _ in 0..60 {
        tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);
    }

    // 5. State should still be valid
    assert!(sys.mmap.read().verify().is_ok());
    assert!(sys.mmap.read().verify_checksum().is_ok());

    // 6. Subcognitive module should be running
    let state = sys.mmap.read();
    let module = state.manifest.get(ModuleId::Subcognitive).expect("module");
    assert_eq!(module.status(), ModuleStatus::Running);
}

#[test]
fn test_tick_loop_consolidates_and_restores_plasticity_while_sleeping() {
    let mut sys = TestSystem::new("tick_sleep", 16, 64);
    let mut tick_loop = TickLoop::new();
    let now = current_ms();

    // Force the cognitive zone to Sleeping and mimic a stressed day:
    // BDNF is depleted, but cortisol is now low as sleep begins.
    // Serotonin is moderate-to-high to drive BDNF recovery.
    sys.mmap
        .modify(now, |s| {
            s.zones.transition_to(CognitiveZone::Sleeping, now);
            s.neurochemicals.chemicals[NeurochemicalId::BDNF as usize].level = 0.12;
            s.neurochemicals.chemicals[NeurochemicalId::BDNF as usize].baseline = 0.40;
            s.neurochemicals.chemicals[NeurochemicalId::Cortisol as usize].level = 0.05;
            s.neurochemicals.chemicals[NeurochemicalId::Serotonin as usize].level = 0.80;
            s.neurochemicals.recompute_derived();
            s.sync_neurochemistry_to_state();
        })
        .expect("set sleep state");

    // Verify pre-sleep plasticity is impaired (low BDNF).
    // `read_consistent` is safe — it copies through raw pointers,
    // no `&GenesisCoreState` to the mmap'd memory is created.
    let pre_state = sys.mmap.read_consistent().expect("read pre-sleep state");
    assert!(
        pre_state.memory.plasticity_gate < 0.2,
        "pre-sleep plasticity should be low due to depleted BDNF"
    );

    // Push a high-salience STM entry.
    sys.stm.push(RingBufferEntry::new(
        1000,
        EventType::UserInput,
        4,
        0.85,
        make_emotional_tag(),
        "User asked about sleep",
    ));

    // Tick many times while sleeping — consolidation should now run as
    // BDNF recovers, and plasticity should strengthen.
    for _ in 0..20 {
        tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);
    }

    // The high-salience STM entry should have been promoted to LTM.
    assert!(
        sys.ltm.count() > 0,
        "STM should be consolidated while sleeping once BDNF recovers"
    );
    assert!(
        sys.stm.count() > 0,
        "STM entry should still be present in the ring buffer"
    );

    // Plasticity should have strengthened during sleep.
    let sleep_state = sys.mmap.read_consistent().expect("read post-sleep state");
    assert!(
        sleep_state.memory.plasticity_gate > pre_state.memory.plasticity_gate,
        "sleep should restore plasticity (BDNF recovery)"
    );

    // Wake up and re-tick — consolidation should continue.
    let wake = current_ms();
    sys.mmap
        .modify(wake, |s| {
            s.zones.transition_to(CognitiveZone::Idle, wake);
            s.neurochemicals.recompute_derived();
            s.sync_neurochemistry_to_state();
        })
        .expect("set idle zone");

    for _ in 0..10 {
        tick_loop.tick(&sys.mmap, &sys.stm, &mut sys.ltm);
    }

    assert!(
        sys.ltm.count() > 0,
        "consolidation should continue after waking"
    );
}
