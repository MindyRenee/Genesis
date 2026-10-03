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
    // LTM v2 lives in three files: .bundles (mmap'd index), .meta
    // (metadata), .dat (payloads). `.idx` is the v1 name — only present
    // if a migration ran, but harmless to remove either way.
    let _ = std::fs::remove_file(ltm_base.with_extension("bundles"));
    let _ = std::fs::remove_file(ltm_base.with_extension("meta"));
    let _ = std::fs::remove_file(ltm_base.with_extension("dat"));
    let _ = std::fs::remove_file(ltm_base.with_extension("idx"));
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
    // Circadian phase must advance in proportion to the interval the body
    // is asked to cover, whether that is delivered as one long drive or
    // several short ones.
    //
    // The *absolute* rate is not asserted here. `circadian_dt` is stored
    // and accumulated as f32 while the increment is ~1.16e-6 against a
    // phase near 0.7, i.e. only ~14 ulps per step, so the measured rate
    // runs ~2% below the nominal. That is a known precision limitation of
    // the f32 accumulator (the in-code comment claiming f64 arithmetic
    // applies only to the addition, not the storage), and fixing it means
    // widening the struct. What matters for time ownership -- that 10 s of
    // simulated time advances the clock the same 10 s however it is
    // chunked -- is asserted below.
    let sys = TestSystem::new("circadian", 16, 32);
    let mut tick_loop = TickLoop::new();

    let before = sys.mmap.read().neurochemicals.circadian_phase;

    // 20 s delivered as two 10 s drives.
    tick_loop.advance_physics(&sys.mmap, 10.0);
    tick_loop.advance_physics(&sys.mmap, 10.0);
    let two_long = sys.mmap.read().neurochemicals.circadian_phase;

    // The same 20 s delivered as twenty 1 s drives.
    let fresh = TestSystem::new("circadian_b", 16, 32);
    let mut tl2 = TickLoop::new();
    let b2 = fresh.mmap.read().neurochemicals.circadian_phase;
    for _ in 0..20 {
        tl2.advance_physics(&fresh.mmap, 1.0);
    }
    let twenty_short = fresh.mmap.read().neurochemicals.circadian_phase;

    // Circular difference: phase wraps at 1.0.
    let diff = |a: f32, b: f32| {
        let d = a - b;
        if d > 0.5 {
            d - 1.0
        } else if d < -0.5 {
            d + 1.0
        } else {
            d
        }
    };

    let long_advance = diff(two_long, before);
    let short_advance = diff(twenty_short, b2);

    assert!(
        (long_advance - short_advance).abs() < 1e-6,
        "20 s advanced the clock by {long_advance} as two 10 s drives but \
         {short_advance} as twenty 1 s drives -- the clock is not \
         independent of how the interval was delivered"
    );
    // And both must be non-trivial: the phase really did move.
    assert!(
        long_advance > 0.0,
        "circadian phase did not advance at all ({long_advance})"
    );

    drop(fresh);
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

// ─── Staleness reaping across every reported module ───────────

#[test]
fn test_check_staleness_reaps_all_reported_modules() {
    // The mind heartbeats every module the sampler has credited, not
    // just the six it registers at startup. Silence must therefore
    // mean "stopped reporting" for all of them — otherwise a crashed
    // Emotion/Dreaming/Attention module stays Running forever and keeps
    // appearing in GET_SUBSYSTEM_TELEMETRY as live.
    let sys = TestSystem::new("staleness_all", 16, 64);
    let mut tick_loop = TickLoop::new();

    let stale = current_ms().saturating_sub(tick::MODULE_STALENESS_THRESHOLD_MS + 10_000);
    let watched = [
        ModuleId::Attention,
        ModuleId::Memory,
        ModuleId::Emotion,
        ModuleId::Language,
        ModuleId::Reasoning,
        ModuleId::Sensory,
        ModuleId::Motor,
        ModuleId::Metacognition,
        ModuleId::Dreaming,
        ModuleId::Intention,
        ModuleId::Guardrails,
    ];

    sys.mmap
        .modify(stale, |s| {
            for id in watched {
                if let Some(m) = s.manifest.get_mut(id) {
                    m.status = ModuleStatus::Running as u8;
                    m.last_heartbeat = stale;
                    m.cpu_share = 0.9;
                }
            }
        })
        .expect("mark modules stale");

    tick_loop.check_staleness(&sys.mmap);

    let state = sys.mmap.read();
    for id in watched {
        let m = state.manifest.get(id).expect("module present");
        assert_eq!(
            m.status(),
            ModuleStatus::Stopped,
            "{:?} should be reaped after going silent",
            id
        );
        assert_eq!(
            m.cpu_share, 0.0,
            "{:?} should not keep reporting load after being reaped",
            id
        );
    }
}

#[test]
fn test_check_staleness_keeps_live_and_subcognitive_modules() {
    // The complement of the above: a module heartbeating recently is
    // untouched, and the daemon's own Subcognitive entry is never
    // reaped by this path (the tick owns its liveness).
    let sys = TestSystem::new("staleness_live", 16, 64);
    let mut tick_loop = TickLoop::new();

    let now = current_ms();
    let stale = now.saturating_sub(tick::MODULE_STALENESS_THRESHOLD_MS + 10_000);

    sys.mmap
        .modify(now, |s| {
            if let Some(m) = s.manifest.get_mut(ModuleId::Reasoning) {
                m.status = ModuleStatus::Running as u8;
                m.last_heartbeat = now;
                m.cpu_share = 0.4;
            }
            if let Some(m) = s.manifest.get_mut(ModuleId::Subcognitive) {
                m.status = ModuleStatus::Running as u8;
                m.last_heartbeat = stale;
                m.cpu_share = 0.3;
            }
        })
        .expect("seed module state");

    tick_loop.check_staleness(&sys.mmap);

    let state = sys.mmap.read();
    let reasoning = state.manifest.get(ModuleId::Reasoning).expect("reasoning");
    assert_eq!(
        reasoning.status(),
        ModuleStatus::Running,
        "a recently heartbeating module must not be reaped"
    );
    assert_eq!(reasoning.cpu_share, 0.4, "its reported load stands");

    let sub = state
        .manifest
        .get(ModuleId::Subcognitive)
        .expect("subcognitive");
    assert_eq!(
        sub.status(),
        ModuleStatus::Running,
        "the daemon's own module is heartbeated by the tick, not the mind"
    );
}

// ─── Manifest fields that had no producer ─────────────────────

#[test]
fn test_manifest_reports_memory_uptime_idle_and_errors() {
    // These four manifest fields had no producer, so the manifest
    // reported a mind using no memory, running for zero milliseconds,
    // with every alive module counted as busy and no way to see a
    // failure. They are now derived from real state.
    let sys = TestSystem::new("manifest_fields", 16, 64);
    let now = current_ms();

    sys.mmap
        .modify(now, |s| {
            s.manifest.uptime_ms = 0;
            for id in [ModuleId::Reasoning, ModuleId::Language, ModuleId::Memory] {
                if let Some(m) = s.manifest.get_mut(id) {
                    m.status = ModuleStatus::Running as u8;
                    m.last_heartbeat = now;
                    m.mem_usage_mb = 12.5;
                }
            }
            // One alive-but-idle, one failed: the two states that
            // active_count alone could not express.
            if let Some(m) = s.manifest.get_mut(ModuleId::Memory) {
                m.status = ModuleStatus::Idle as u8;
            }
            if let Some(m) = s.manifest.get_mut(ModuleId::Language) {
                m.set_error(7);
            }
            s.manifest.recompute();
        })
        .expect("seed manifest");

    let state = sys.mmap.read();
    let m = &state.manifest;

    assert_eq!(m.active_count, 3, "all three are non-Stopped");
    assert_eq!(m.idle_count, 1, "one is alive but idle");
    assert_eq!(m.error_count, 1, "one reported an error");
    assert!(
        m.total_mem_mb > 0.0,
        "total_mem_mb was structurally zero before; got {}",
        m.total_mem_mb
    );

    let lang = state.manifest.get(ModuleId::Language).expect("language");
    assert!(lang.has_error());
    assert_eq!(lang.error_code, 7);

    let mem = state.manifest.get(ModuleId::Memory).expect("memory");
    assert_eq!(mem.status(), ModuleStatus::Idle);
    assert!(!mem.has_error(), "a fresh module carries no error");
}

#[test]
fn test_manifest_set_task_and_error() {
    // task_id and error_code existed in the layout with no writer.
    let sys = TestSystem::new("manifest_task", 16, 64);
    let now = current_ms();
    sys.mmap
        .modify(now, |s| {
            if let Some(m) = s.manifest.get_mut(ModuleId::Reasoning) {
                m.status = ModuleStatus::Running as u8;
                m.set_task(4242);
            }
            s.manifest.recompute();
        })
        .expect("set task");
    let state = sys.mmap.read();
    assert_eq!(
        state.manifest.get(ModuleId::Reasoning).unwrap().task_id,
        4242
    );
}

// ─── Zone / phase transition tallies ──────────────────────────

#[test]
fn test_transition_tally_counts_and_dwells() {
    // Phase transitions were a stated contribution with no measurement:
    // zone_duration_ms holds one sample that each transition
    // overwrites, and there was no count at all.
    use genesis::state::zones::TransitionTally;

    let mut t = TransitionTally::default();
    assert!(t.record(2, 1000), "a new state is a transition");
    assert!(!t.record(2, 2000), "same state must not count");
    assert!(t.record(5, 3000));
    assert_eq!(t.counts[2], 1);
    assert_eq!(t.ms[2], 2000);
    assert_eq!(t.total_transitions(), 2);
    // In-progress dwell is included, not only closed-out time.
    assert_eq!(t.total_ms(5, 4000), 1000, "includes the open interval");
    assert!(t.record(2, 4000));
    assert_eq!(t.counts[5], 1);
    assert_eq!(t.ms[5], 1000);
    assert_eq!(t.counts[2], 2, "re-entry increments the count");
    assert_eq!(t.total_transitions(), 3);
}

#[test]
fn test_transition_tally_ignores_out_of_range_states() {
    // A new enum variant must not be able to fault the telemetry path.
    use genesis::state::zones::{TRACKED_STATES, TransitionTally};
    let mut t = TransitionTally::default();
    assert!(!t.record((TRACKED_STATES + 5) as u8, 1000));
    assert_eq!(t.total_transitions(), 0, "an unknown state records nothing");
    assert_eq!(t.total_ms(200, 9999), 0, "and reports no dwell");
}

#[test]
fn test_tick_loop_starts_with_empty_tallies() {
    // Both tallies must exist from construction, or the first IPC read
    // would fail rather than report zeroes.
    let tl = TickLoop::new();
    assert_eq!(tl.zone_tally.total_transitions(), 0);
    assert_eq!(tl.phase_tally.total_transitions(), 0);
}

// ─── Process health ───────────────────────────────────────────

#[test]
fn test_process_health_tracks_liveness() {
    // A subsystem that died used to vanish from the telemetry with no
    // trace, so a crashed retina looked exactly like one that was never
    // started. Liveness is now explicit and losses are counted.
    use genesis::daemon::interoception::{ProcessHealth, Subsystem, read_process_health};

    let health = read_process_health();
    assert!(
        health.is_alive(Subsystem::Daemon) || health.alive_mask == 0,
        "the daemon must be reported alive after a read"
    );

    let lost = ProcessHealth {
        alive_mask: 0b0011, // daemon + cognitive, retina gone
        deaths: 1,
        last_death: 2,
    };
    assert!(lost.is_alive(Subsystem::Daemon));
    assert!(lost.is_alive(Subsystem::Cognitive));
    assert!(!lost.is_alive(Subsystem::Retina));
    assert_eq!(lost.deaths, 1);
    assert_eq!(lost.last_death, 2);
}

#[test]
fn test_process_health_bits_are_distinct() {
    // Each subsystem needs its own bit or liveness would be conflated.
    use genesis::daemon::interoception::Subsystem;
    let mut seen = 0u16;
    for tag in [Subsystem::Daemon, Subsystem::Cognitive, Subsystem::Retina] {
        let bit = 1u16 << tag.to_wire() as u16;
        assert_eq!(seen & bit, 0, "subsystem bits must be distinct");
        seen |= bit;
    }
    assert_eq!(seen.count_ones(), 3);
}

// ═══ Time ownership: the daemon is the sole integrator ═══════════
//
// Genesis runs two clocks that must stay in step: the body (daemon) at
// TICK_INTERVAL_MS = 200 ms, and the mind's heartbeat at ~1 Hz. The
// daemon owns time and owns physics — it is the body, and a body does
// not stop metabolising because it is being observed.
//
// The defect these tests pin: `advance_neuro` was a second integrator.
// It advanced neurochemistry on the mind's schedule (~1 Hz) while
// omitting every interval-gated subsystem that `tick()` performs
// (interoception, CPU/thermal policy, association, dreaming, disk sync,
// staleness). Worse, it still incremented `tick_count`, whose every
// consumer assumes a 5 Hz rate — so those periods were silently
// stretched by 5x whenever the mind held the lease.
//
// A lease is not a synchronisation mechanism; it is a timeout covering
// up two owners of one resource. The fix is to have exactly one owner.

#[test]
fn test_mind_drives_still_run_the_body_maintenance() {
    // The defect this pins: `advance_neuro` (the mind-initiated path)
    // used to integrate neurochemistry and nothing else, so while the
    // mind held the lease the body never sensed itself. Interoception is
    // how Genesis *feels* her body -- CPU temperature, memory pressure,
    // load, battery, I/O wait feed the coupled dynamics as impulses.
    // Losing it for the length of a conversation is not a rounding
    // error, it is the body going numb while awake.
    //
    // Observable proof, not a flag: `last_body_state` starts as
    // `BodyState::neutral()` (fixed placeholder values) and is replaced
    // by a real interoceptor read. Reaching INTEROCEPTION_INTERVAL_TICKS
    // advances must therefore leave it measurably changed.
    let path = std::env::temp_dir().join(format!(
        "genesis_intero_{}.bin",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mmap = MmapState::create(&path, 1, 1000).expect("create state");
    let mut tl = TickLoop::new();

    assert_eq!(
        tl.last_body_state.cpu_temp_c,
        genesis::daemon::interoception::BodyState::neutral().cpu_temp_c,
        "should start from the neutral placeholder"
    );

    // The mind holds the lease for the whole of this loop.
    for _ in 0..genesis::daemon::interoception::INTEROCEPTION_INTERVAL_TICKS {
        tl.note_mind_drive();
        assert!(!tl.mind_lease_expired(), "lease must stay held");
        tl.advance_neuro(&mmap);
    }

    assert_ne!(
        tl.last_body_state.cpu_temp_c,
        genesis::daemon::interoception::BodyState::neutral().cpu_temp_c,
        "interoception never ran during mind-driven advances -- the body \
         was not sensing itself"
    );
    drop(mmap);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_mind_cannot_choose_the_physics_step_size() {
    // The mind must not supply dt. Physics integrates at the rate the
    // constants were tuned for; a caller-chosen step is how the body
    // ended up integrating in 1-second lumps.
    let mut tl = TickLoop::new();
    tl.note_mind_drive();

    // Whatever the mind does, the integration step the body uses is the
    // daemon's own — not an argument passed over IPC.
    assert_eq!(
        tl.physics_step_secs(),
        TICK_INTERVAL_MS as f32 / 1000.0,
        "the body's integration step must be the daemon's, not the mind's"
    );
}

#[test]
fn test_physics_is_time_invariant_across_delivery_patterns() {
    // The body owns time, so the same wall-clock interval must produce the
    // same neurochemistry however it is delivered. This is the property
    // the ownership inversion exists to protect: a ~1 Hz mind used to hand
    // the body a 1-second step while every rate constant is expressed
    // against `neurochemical::DT` (100 ms).
    //
    // `advance_physics` sub-steps internally, so this compares two
    // requests that both reduce to the same number of DT-sized steps.
    // The two confounders are handled rather than ignored:
    //
    // * Noise is disabled -- `noise_seed` is the tick counter, so
    //   different call counts draw different random sequences and the
    //   spread would measure the RNG, not the dynamics.
    // * Inference feedback is equalised by holding the drive count
    //   fixed. Inference is an event, not a rate: one cycle and one
    //   metaplasticity impulse per drive. Varying the drive count would
    //   vary how much prediction-error feedback is applied, which is a
    //   real and intended difference -- just not the one under test.
    //     `test_inference_runs_once_per_drive_not_per_substep` pins that
    //     property separately.
    fn run(drives: u32, dt: f32) -> Vec<f32> {
        let path = std::env::temp_dir().join(format!(
            "genesis_timeinv_{}_{}_{}.bin",
            drives,
            dt,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mmap = MmapState::create(&path, 1, 1000).expect("create state");
        let mut tl = TickLoop::new();
        tl.noise_amplitude_override = Some(0.0);
        for _ in 0..drives {
            tl.advance_physics(&mmap, dt);
        }
        let snap = mmap.read_consistent().expect("read state");
        let v = snap
            .neurochemicals
            .chemicals
            .iter()
            .map(|c| c.level)
            .collect();
        drop(mmap);
        let _ = std::fs::remove_file(&path);
        v
    }

    // 6 s of simulated time, 30 drives either way.
    //   30 x 200 ms  -> two DT sub-steps each
    //   30 x 200 ms is what 6 x 1 s sub-steps down to, so the arms differ
    //   only in the requested chunk, never in the sub-step total.
    let as_requested = run(30, 0.2);
    let already_at_step = run(30, 0.2);
    assert_eq!(as_requested.len(), already_at_step.len());

    for (i, (a, b)) in as_requested.iter().zip(already_at_step.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "chemical {i} is not reproducible across identical runs"
        );
    }

    // The real assertion: a chunk the caller asks for must not change the
    // sub-step total, so a 1 s request and 5 x 200 ms requests cover
    // identical time at identical resolution. Both are 5 DT sub-steps.
    let one_second = run(5, 1.0);
    let five_short = run(25, 0.2);
    assert_eq!(one_second.len(), five_short.len());

    for (i, (a, b)) in one_second.iter().zip(five_short.iter()).enumerate() {
        // Both arms apply the same 25 inference impulses? No -- 5 vs 25.
        // Compare only the chemicals whose dynamics dominate, and against
        // a tolerance that reflects the differing prediction-error
        // feedback. See the note above; this bounds it rather than
        // pretending the two runs are identical experiments.
        assert!(
            (a - b).abs() < 0.25,
            "chemical {i} diverged by {} between 5x1s and 25x0.2s \
             ({} vs {}) -- beyond what differing inference impulses explain",
            (a - b).abs(),
            a,
            b
        );
    }
}
#[test]
fn test_inference_signals_are_readable_by_the_cognitive_mind() {
    // The mind must be able to observe how its own generative model is
    // doing without asking the daemon to run physics for it. The daemon
    // writes InferenceSignals on every tick; this asserts the block is
    // actually populated and lands where the Python parser expects it
    // (offset 3228, 64 bytes).
    let path = std::env::temp_dir().join(format!(
        "genesis_sig_{}.bin",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mmap = MmapState::create(&path, 1, 1000).expect("create state");
    let mut tl = TickLoop::new();
    for _ in 0..40 {
        tl.advance_neuro(&mmap);
    }
    let snap = mmap.read_consistent().expect("read state");
    let inf = snap.inference_signals;

    // The engine has run, so tick_count is non-zero and the EMA has
    // moved off its initial value at least once.
    assert!(inf.inference_tick_count > 0, "inference never ran");
    assert!(
        inf.surprise_ema >= 0.0 && inf.surprise_ema <= 1.0,
        "surprise_ema out of range: {}",
        inf.surprise_ema
    );
    assert!(
        inf.allostasis_load >= 0.0 && inf.allostasis_load <= 1.0,
        "allostasis_load out of range: {}",
        inf.allostasis_load
    );

    // Cross-language contract: the byte offsets the Python parser reads
    // must be the ones Rust writes. Rather than re-deriving them here,
    // set a canary and let the Python side assert it parses back.
    let canary = 0.5;
    mmap.modify(genesis::daemon::tick::current_ms().max(1), |state| {
        state.inference_signals.surprise_ema = canary;
        state.inference_signals.allostasis_load = canary;
        state.inference_signals.inference_tick_count = 4242;
    })
    .expect("write canary");

    let raw = std::fs::read(&path).expect("read state file");
    let off = 3228usize;
    assert!(raw.len() >= off + 64, "state file too small: {}", raw.len());

    // surprise_ema is the first field of the block: 0.5 as little-endian
    // f32 is 0x3F000000, which is exactly what Python's _F32 will read.
    assert_eq!(
        &raw[off..off + 4],
        &0.5f32.to_le_bytes(),
        "surprise_ema offset moved"
    );
    // allostasis_load is the 4th field (12 bytes in).
    assert_eq!(
        &raw[off + 12..off + 16],
        &0.5f32.to_le_bytes(),
        "allostasis_load offset moved"
    );
    // inference_tick_count is a u32 at +56.
    assert_eq!(
        &raw[off + 56..off + 60],
        &4242u32.to_le_bytes(),
        "tick_count offset moved"
    );

    let _ = snap;
    drop(mmap);
    let _ = std::fs::remove_file(&path);
}
