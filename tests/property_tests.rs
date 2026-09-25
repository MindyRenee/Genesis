//! Property-based tests for the Genesis substrate.
//!
//! These test invariants, not examples. Each test runs many iterations
//! with deterministic seeded inputs, asserting that fundamental
//! properties hold across the entire input space.
//!
//! Categories:
//! 1. Core state invariants — chemicals in [0,1], checksum determinism,
//!    seqlock monotonicity
//! 2. Neurochemical dynamics — stability over 1000 ticks, no NaN/inf,
//!    no runaway
//! 3. Ring buffer — push/peek consistency under wraparound, count ≤
//!    capacity
//! 4. LTM store — store/retrieve round-trip, SimHash determinism,
//!    compression correctness
//! 5. Consolidation — salience ordering preserved, emotional tag
//!    integrity

use genesis::daemon::ConsolidationEngine;
use genesis::state::*;
use genesis::store::*;
use std::path::PathBuf;

// ─────────────────────────────────────────────────────────────────
//  Deterministic PRNG (xorshift64)
// ─────────────────────────────────────────────────────────────────

/// A simple deterministic PRNG so test runs are reproducible.
/// We don't use `rand` to avoid adding a dependency.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xDEADBEEFCAFEBABE } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        // Map to [0.0, 1.0)
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    fn next_u8(&mut self) -> u8 {
        (self.next_u64() & 0xFF) as u8
    }
}

fn temp_path(prefix: &str, ext: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "genesis_prop_{prefix}_{}_{}.{ext}",
        std::process::id(),
        nanos
    ))
}

fn temp_base(prefix: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "genesis_prop_{prefix}_{}_{}",
        std::process::id(),
        nanos
    ))
}

// ─────────────────────────────────────────────────────────────────
//  1. Core state invariants
// ─────────────────────────────────────────────────────────────────

#[test]
fn prop_chemical_levels_stay_in_bounds_after_impulses() {
    let mut rng = Rng::new(42);
    let mut vec = NeurochemicalVector::new(0);

    // Apply 1000 random impulses of varying magnitudes
    for _ in 0..1000 {
        let chem_idx = (rng.next_u64() % NEUROCHEMICAL_COUNT as u64) as usize;
        let magnitude = rng.next_f32_range(-2.0, 2.0); // extreme impulses
        vec.chemicals[chem_idx].apply_impulse(magnitude, 0);
        vec.recompute_derived();
        vec.tick_default();

        // After each tick, all levels must stay in [0, 1]
        for chem in &vec.chemicals {
            assert!(
                chem.level.is_finite(),
                "chemical {} level is not finite: {}",
                chem.id,
                chem.level
            );
            assert!(
                chem.level >= 0.0 && chem.level <= 1.0,
                "chemical {} level out of bounds: {}",
                chem.id,
                chem.level
            );
        }
    }
}

#[test]
fn prop_checksum_is_deterministic() {
    let mut rng = Rng::new(123);
    let mut state = GenesisCoreState::new(1, 1000);

    // Checksum should be the same for the same state
    for _ in 0..100 {
        // Modify some random fields
        let chem_idx = (rng.next_u64() % NEUROCHEMICAL_COUNT as u64) as usize;
        state.neurochemicals.chemicals[chem_idx].level = rng.next_f32();
        state.checksum = state.compute_checksum();

        let cs1 = state.compute_checksum();
        let cs2 = state.compute_checksum();
        assert_eq!(cs1, cs2, "checksum not deterministic for same state");
    }
}

#[test]
fn prop_checksum_changes_on_state_modification() {
    let mut rng = Rng::new(999);
    let mut state = GenesisCoreState::new(1, 1000);
    state.checksum = state.compute_checksum();

    for _ in 0..100 {
        let cs_before = state.compute_checksum();
        let chem_idx = (rng.next_u64() % NEUROCHEMICAL_COUNT as u64) as usize;
        let old_level = state.neurochemicals.chemicals[chem_idx].level;
        let new_level = rng.next_f32();
        if old_level != new_level {
            state.neurochemicals.chemicals[chem_idx].level = new_level;
            let cs_after = state.compute_checksum();
            assert_ne!(
                cs_before, cs_after,
                "checksum didn't change after modifying chemical level"
            );
        }
    }
}

#[test]
fn prop_seqlock_is_monotonically_increasing() {
    let mut state = GenesisCoreState::new(1, 1000);
    let mut last_seq = state.header.seq_lock;

    for i in 0..100 {
        // SAFETY: locally-owned non-aliased state, single-threaded test.
        unsafe { state.write_begin(1000 + i) };
        assert!(
            state.header.seq_lock & 1 == 1,
            "seq_lock should be odd during write (iteration {})",
            i
        );
        state.write_end();
        assert!(
            state.header.seq_lock & 1 == 0,
            "seq_lock should be even after write (iteration {})",
            i
        );
        assert!(
            state.header.seq_lock > last_seq,
            "seq_lock should be monotonically increasing: {} <= {}",
            state.header.seq_lock,
            last_seq
        );
        last_seq = state.header.seq_lock;
    }
}

#[test]
fn prop_read_consistent_returns_valid_state_after_write() {
    let mut state = GenesisCoreState::new(1, 1000);

    for i in 0..50 {
        // SAFETY: single-threaded test, locally-owned state, no concurrent writer.
        unsafe { state.write_begin(1000 + i) };
        state.neurochemicals.chemicals[0].level = 0.5 + i as f32 * 0.01;
        state.write_end();

        // SAFETY: same single-threaded, non-aliased state as above.
        let snapshot = unsafe { state.read_consistent() };
        assert!(
            snapshot.is_some(),
            "read_consistent returned None after write_end"
        );
        let snap = snapshot.unwrap();
        assert!(
            (snap.neurochemicals.chemicals[0].level - (0.5 + i as f32 * 0.01)).abs() < 1e-6,
            "snapshot doesn't match written value"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
//  2. Neurochemical dynamics stability
// ─────────────────────────────────────────────────────────────────

#[test]
fn prop_dynamics_stable_over_1000_ticks_from_random_starts() {
    let mut rng = Rng::new(777);

    // Run 10 different random starting states for 1000 ticks each
    for trial in 0..10 {
        let mut vec = NeurochemicalVector::new(0);

        // Randomize initial levels
        for chem in &mut vec.chemicals {
            chem.level = rng.next_f32();
            chem.tonic_level = rng.next_f32();
            chem.phasic_level = rng.next_f32();
            chem.baseline = rng.next_f32_range(0.1, 0.5);
        }
        vec.recompute_derived();

        for tick in 0..1000 {
            vec.tick_default();

            // Check no NaN/inf in any chemical
            for chem in &vec.chemicals {
                assert!(
                    chem.level.is_finite(),
                    "trial {} tick {}: chemical {} level is not finite: {}",
                    trial,
                    tick,
                    chem.id,
                    chem.level
                );
                assert!(
                    chem.baseline.is_finite(),
                    "trial {} tick {}: chemical {} baseline is not finite: {}",
                    trial,
                    tick,
                    chem.id,
                    chem.baseline
                );
                assert!(
                    chem.receptor_sensitivity.is_finite(),
                    "trial {} tick {}: chemical {} sensitivity is not finite: {}",
                    trial,
                    tick,
                    chem.id,
                    chem.receptor_sensitivity
                );
            }

            // Check derived fields are finite
            assert!(
                vec.arousal.is_finite(),
                "arousal not finite at tick {}",
                tick
            );
            assert!(
                vec.valence.is_finite(),
                "valence not finite at tick {}",
                tick
            );
            assert!(
                vec.plasticity_gate.is_finite(),
                "plasticity_gate not finite at tick {}",
                tick
            );
            assert!(
                vec.global_tone.is_finite(),
                "global_tone not finite at tick {}",
                tick
            );
        }

        // After 1000 ticks, the system should have settled — no extreme values
        for chem in &vec.chemicals {
            assert!(
                chem.level >= 0.0 && chem.level <= 1.0,
                "chemical {} level out of bounds after 1000 ticks: {}",
                chem.id,
                chem.level
            );
        }
    }
}

#[test]
fn prop_coupling_matrix_stays_bounded() {
    let mut vec = NeurochemicalVector::new(0);

    // Tick 2000 times with stress and wellbeing adaptation
    for _ in 0..2000 {
        vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.80;
        vec.tick_default();
    }

    // Check all coupling matrix values are bounded
    for i in 0..NEUROCHEMICAL_COUNT {
        for j in 0..NEUROCHEMICAL_COUNT {
            let val = vec.coupling_matrix[i][j];
            assert!(
                val.is_finite(),
                "coupling_matrix[{}][{}] is not finite: {}",
                i,
                j,
                val
            );
            assert!(
                val.abs() <= 1.0,
                "coupling_matrix[{}][{}] out of bounds: {}",
                i,
                j,
                val
            );
        }
    }
}

#[test]
fn prop_receptor_sensitivity_stays_positive() {
    let mut rng = Rng::new(555);
    let mut vec = NeurochemicalVector::new(0);

    // Apply extreme sustained levels
    for _ in 0..5000 {
        let idx = (rng.next_u64() % NEUROCHEMICAL_COUNT as u64) as usize;
        vec.chemicals[idx].level = if rng.next_u8().is_multiple_of(2) {
            0.95
        } else {
            0.05
        };
        vec.tick_default();
    }

    for chem in &vec.chemicals {
        assert!(
            chem.receptor_sensitivity > 0.0,
            "receptor_sensitivity should never reach zero: chem {} has {}",
            chem.id,
            chem.receptor_sensitivity
        );
        assert!(
            chem.receptor_sensitivity.is_finite(),
            "receptor_sensitivity is not finite: chem {}",
            chem.id
        );
    }
}

// ─────────────────────────────────────────────────────────────────
//  3. Ring buffer invariants
// ─────────────────────────────────────────────────────────────────

#[test]
fn prop_ring_buffer_count_never_exceeds_capacity() {
    let mut rng = Rng::new(321);
    let path = temp_path("rb_count", "bin");
    let cap = 8u32;
    let rb = RingBuffer::create(&path, cap).expect("create");

    // Push 100 entries into a capacity-8 buffer
    for i in 0..100 {
        let entry = RingBufferEntry::new(
            i,
            EventType::from_u8(rng.next_u8() % 8),
            rng.next_u8(),
            rng.next_f32(),
            [0.0f32; 12],
            &format!("entry {}", i),
        );
        rb.push(entry);
        assert!(
            rb.count() <= cap as u64,
            "count {} exceeds capacity {} after push {}",
            rb.count(),
            cap,
            i
        );
    }

    // After 100 pushes into cap-8, count should be exactly cap
    assert_eq!(rb.count(), cap as u64);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn prop_ring_buffer_push_peek_consistency() {
    let _rng = Rng::new(654); // seeded for reproducibility
    let path = temp_path("rb_peek", "bin");
    let cap = 16u32;
    let rb = RingBuffer::create(&path, cap).expect("create");

    // Push entries and verify we can peek them back
    let mut entries = Vec::new();
    for i in 0..cap as u64 {
        let text = format!("test entry {}", i);
        let entry = RingBufferEntry::new(
            i * 1000,
            EventType::Observation,
            (i % 10) as u8,
            i as f32 / 100.0,
            [0.5f32; 12],
            &text,
        );
        entries.push((i, entry.timestamp, entry.salience, text.clone()));
        rb.push(entry);
    }

    // Peek each entry and verify
    for i in 0..cap as u64 {
        let peeked = rb.peek(i);
        assert!(peeked.is_some(), "peek({}) returned None", i);
        let p = peeked.unwrap();
        assert_eq!(p.timestamp, i * 1000, "timestamp mismatch at slot {}", i);
        assert!(
            (p.salience - i as f32 / 100.0).abs() < 1e-6,
            "salience mismatch at slot {}",
            i
        );
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn prop_ring_buffer_wraparound_preserves_latest() {
    let path = temp_path("rb_wrap", "bin");
    let cap = 4u32;
    let rb = RingBuffer::create(&path, cap).expect("create");

    // Push 10 entries into cap-4 buffer — only last 4 should remain
    for i in 0..10u64 {
        let entry = RingBufferEntry::new(
            i,
            EventType::Observation,
            0,
            0.5,
            [0.0f32; 12],
            &format!("entry {}", i),
        );
        rb.push(entry);
    }

    assert_eq!(rb.count(), cap as u64);

    // The last 4 entries (indices 6, 7, 8, 9) should be in the buffer
    let mut found_texts = Vec::new();
    for i in 0..cap as u64 {
        if let Some(e) = rb.peek(i) {
            found_texts.push(e.text().to_string());
        }
    }

    // Should contain entries 6-9 (the latest 4)
    for text in &found_texts {
        let num: u64 = text.strip_prefix("entry ").unwrap().parse().unwrap();
        assert!(
            (6..=9).contains(&num),
            "found stale entry {} after wraparound",
            num
        );
    }

    let _ = std::fs::remove_file(&path);
}

// ─────────────────────────────────────────────────────────────────
//  4. LTM store invariants
// ─────────────────────────────────────────────────────────────────

#[test]
fn prop_ltm_store_retrieve_roundtrip() {
    let mut rng = Rng::new(876);
    let base = temp_base("ltm_rt");
    let mut ltm = LtmStore::create(&base, 128).expect("create");

    // Store 50 episodes with random data
    let mut stored = Vec::new();
    for i in 0..50u64 {
        let timestamp = rng.next_u64();
        let salience = rng.next_f32();
        let text = format!("episode {} with random content {}", i, rng.next_u64());
        let emotional_tag = [
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
        ];

        let id = ltm
            .store(
                timestamp,
                salience,
                emotional_tag,
                [0.5f32; 4], // compact_tag
                EventType::Observation as u8,
                rng.next_u8(),
                &text,
            )
            .expect("store");

        stored.push((id, timestamp, salience, text, emotional_tag));
    }

    // Retrieve each and verify
    for (id, timestamp, salience, text, emotional_tag) in &stored {
        let ep = ltm.retrieve(*id).expect("retrieve");
        assert_eq!(ep.episode_id, *id, "episode_id mismatch");
        assert_eq!(ep.timestamp, *timestamp, "timestamp mismatch for id {}", id);
        assert!(
            (ep.salience - salience).abs() < 1e-6,
            "salience mismatch for id {}",
            id
        );
        assert_eq!(ep.text, *text, "text mismatch for id {}", id);
        for (j, (actual, expected)) in ep
            .full_emotional_tag
            .iter()
            .zip(emotional_tag.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "emotional_tag[{}] mismatch for id {}",
                j,
                id
            );
        }
    }

    let _ = std::fs::remove_file(base.with_extension("bundles"));
    let _ = std::fs::remove_file(base.with_extension("meta"));
    let _ = std::fs::remove_file(base.with_extension("dat"));
    let _ = std::fs::remove_file(base.with_extension("idx"));
}

#[test]
fn prop_simhash_is_deterministic() {
    let mut rng = Rng::new(222);
    let texts = [
        "hello world",
        "the quick brown fox",
        "neurochemistry of cognition",
        "a",
        "",
        "重复的文本",
    ];

    // Same input → same hash, every time
    for _ in 0..100 {
        for text in &texts {
            let h1 = simhash(text);
            let h2 = simhash(text);
            assert_eq!(h1, h2, "simhash not deterministic for '{}'", text);
        }
    }

    // Random text → deterministic within a single run
    for _ in 0..50 {
        let text = format!("random text {}", rng.next_u64());
        let h1 = simhash(&text);
        let h2 = simhash(&text);
        assert_eq!(h1, h2, "simhash not deterministic for random text");
    }
}

#[test]
fn prop_simhash_identical_texts_have_zero_hamming_distance() {
    let texts = ["hello", "world", "cognition", "a longer text here"];
    for text in &texts {
        let h1 = simhash(text);
        let h2 = simhash(text);
        let dist = (h1 ^ h2).count_ones();
        assert_eq!(dist, 0, "identical texts should have zero hamming distance");
    }
}

#[test]
fn prop_ltm_count_grows_without_ceiling() {
    // v2: LTM has no fixed capacity ceiling. Storing 100 episodes
    // into a store created with capacity=32 should all succeed.
    let base = temp_base("ltm_cap");
    let cap = 32u32;
    let mut ltm = LtmStore::create(&base, cap).expect("create");

    for i in 0..100u64 {
        ltm.store(
            i * 1000,
            0.5,
            [0.0f32; 12],
            [0.5f32; 4], // compact_tag
            EventType::Observation as u8,
            0,
            &format!("entry {}", i),
        )
        .expect("store should succeed regardless of capacity");
    }

    assert_eq!(ltm.count(), 100, "v2 LTM should store all 100 episodes");

    let _ = std::fs::remove_file(base.with_extension("bundles"));
    let _ = std::fs::remove_file(base.with_extension("meta"));
    let _ = std::fs::remove_file(base.with_extension("dat"));
}

// ─────────────────────────────────────────────────────────────────
//  5. Consolidation engine invariants
// ─────────────────────────────────────────────────────────────────

#[test]
fn prop_consolidation_preserves_emotional_tag_integrity() {
    let mut rng = Rng::new(444);
    let state_path = temp_path("cons_state", "bin");
    let stm_path = temp_path("cons_stm", "bin");
    let ltm_base = temp_base("cons_ltm");

    let mmap = MmapState::create(&state_path, 1, 1000).expect("create state");
    let stm = RingBuffer::create(&stm_path, 32).expect("create stm");
    let mut ltm = LtmStore::create(&ltm_base, 64).expect("create ltm");

    // Set up good neurochemistry for consolidation
    mmap.modify(1000, |state| {
        state
            .neurochemicals
            .get_mut(NeurochemicalId::BDNF)
            .unwrap()
            .level = 0.70;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Cortisol)
            .unwrap()
            .level = 0.10;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap()
            .level = 0.60;
        state.neurochemicals.recompute_derived();
        state.sync_neurochemistry_to_state();
        state.memory.update_gating(
            0.60, // DA
            0.50, // SRT
            0.50, // NE
            0.50, // ACh
            0.10, // CORT
            0.70, // BDNF
        );
    })
    .expect("modify");

    // Push entries with known emotional tags
    let mut expected_tags = Vec::new();
    for i in 0..10u64 {
        let tag = [
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
            rng.next_f32(),
        ];
        let entry = RingBufferEntry::new(
            i * 1000,
            EventType::Observation,
            0,
            0.8, // high salience → should consolidate
            tag,
            &format!("consolidation test {}", i),
        );
        stm.push(entry);
        expected_tags.push(tag);
    }

    // Consolidate
    mmap.modify(2000, |state| {
        let result = ConsolidationEngine::consolidate(state, &stm, &mut ltm, 2000, 0.1);
        assert!(result.promoted > 0, "no entries were consolidated");
    })
    .expect("modify");

    // Verify emotional tags are preserved in LTM
    for i in 0..expected_tags.len() as u64 {
        // Find the episode by searching recent episodes
        let episodes = ltm.recent_episodes(64, None);
        let ep = episodes
            .iter()
            .find(|e| e.text == format!("consolidation test {}", i));
        assert!(ep.is_some(), "episode {} not found in LTM", i);
        let ep = ep.unwrap();
        let expected = &expected_tags[i as usize];
        for (j, (actual, exp)) in ep
            .full_emotional_tag
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (actual - exp).abs() < 1e-6,
                "emotional_tag[{}] mismatch in episode {}",
                j,
                i
            );
        }
    }

    drop(mmap);
    drop(stm);
    drop(ltm);
    let _ = std::fs::remove_file(&state_path);
    let _ = std::fs::remove_file(&stm_path);
    let _ = std::fs::remove_file(ltm_base.with_extension("bundles"));
    let _ = std::fs::remove_file(ltm_base.with_extension("meta"));
    let _ = std::fs::remove_file(ltm_base.with_extension("dat"));
    let _ = std::fs::remove_file(ltm_base.with_extension("idx"));
}

#[test]
fn prop_consolidation_skips_low_salience() {
    let state_path = temp_path("cons_skip_state", "bin");
    let stm_path = temp_path("cons_skip_stm", "bin");
    let ltm_base = temp_base("cons_skip_ltm");

    let mmap = MmapState::create(&state_path, 1, 1000).expect("create state");
    let stm = RingBuffer::create(&stm_path, 32).expect("create stm");
    let mut ltm = LtmStore::create(&ltm_base, 64).expect("create ltm");

    // Good neurochemistry
    mmap.modify(1000, |state| {
        state
            .neurochemicals
            .get_mut(NeurochemicalId::BDNF)
            .unwrap()
            .level = 0.70;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Cortisol)
            .unwrap()
            .level = 0.10;
        state.neurochemicals.recompute_derived();
        state.sync_neurochemistry_to_state();
        state
            .memory
            .update_gating(0.60, 0.50, 0.50, 0.50, 0.10, 0.70);
    })
    .expect("modify");

    // Push entries with very low salience
    for i in 0..10u64 {
        let entry = RingBufferEntry::new(
            i * 1000,
            EventType::Observation,
            0,
            0.01, // very low salience
            [0.0f32; 12],
            &format!("low salience entry {}", i),
        );
        stm.push(entry);
    }

    // Consolidate with a moderate threshold
    let ltm_count_before = ltm.count();
    mmap.modify(2000, |state| {
        let result = ConsolidationEngine::consolidate(state, &stm, &mut ltm, 2000, 0.5);
        // With salience 0.01 and consolidation_weight < 1.0,
        // score should be below 0.5 threshold
        assert_eq!(
            result.promoted, 0,
            "low-salience entries should not be consolidated"
        );
    })
    .expect("modify");

    assert_eq!(
        ltm.count(),
        ltm_count_before,
        "LTM count should not change when all entries are skipped"
    );

    drop(mmap);
    drop(stm);
    drop(ltm);
    let _ = std::fs::remove_file(&state_path);
    let _ = std::fs::remove_file(&stm_path);
    let _ = std::fs::remove_file(ltm_base.with_extension("bundles"));
    let _ = std::fs::remove_file(ltm_base.with_extension("meta"));
    let _ = std::fs::remove_file(ltm_base.with_extension("dat"));
    let _ = std::fs::remove_file(ltm_base.with_extension("idx"));
}

#[test]
fn prop_consolidation_blocked_by_chronic_stress() {
    let state_path = temp_path("cons_stress_state", "bin");
    let stm_path = temp_path("cons_stress_stm", "bin");
    let ltm_base = temp_base("cons_stress_ltm");

    let mmap = MmapState::create(&state_path, 1, 1000).expect("create state");
    let stm = RingBuffer::create(&stm_path, 32).expect("create stm");
    let mut ltm = LtmStore::create(&ltm_base, 64).expect("create ltm");

    // Chronic stress: high cortisol, low BDNF
    mmap.modify(1000, |state| {
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Cortisol)
            .unwrap()
            .level = 0.85;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::BDNF)
            .unwrap()
            .level = 0.10;
        state.neurochemicals.recompute_derived();
        state.sync_neurochemistry_to_state();
        state
            .memory
            .update_gating(0.20, 0.30, 0.50, 0.30, 0.85, 0.10);
    })
    .expect("modify");

    // Push high-salience entries
    for i in 0..10u64 {
        let entry = RingBufferEntry::new(
            i * 1000,
            EventType::Observation,
            0,
            0.95, // high salience
            [0.5f32; 12],
            &format!("high salience under stress {}", i),
        );
        stm.push(entry);
    }

    let ltm_count_before = ltm.count();
    mmap.modify(2000, |state| {
        let result = ConsolidationEngine::consolidate(state, &stm, &mut ltm, 2000, 0.1);
        assert_eq!(
            result.promoted, 0,
            "consolidation should be blocked by chronic stress"
        );
        assert!(result.blocked > 0, "entries should be blocked, not skipped");
    })
    .expect("modify");

    assert_eq!(
        ltm.count(),
        ltm_count_before,
        "no entries should be consolidated under chronic stress"
    );

    drop(mmap);
    drop(stm);
    drop(ltm);
    let _ = std::fs::remove_file(&state_path);
    let _ = std::fs::remove_file(&stm_path);
    let _ = std::fs::remove_file(ltm_base.with_extension("bundles"));
    let _ = std::fs::remove_file(ltm_base.with_extension("meta"));
    let _ = std::fs::remove_file(ltm_base.with_extension("dat"));
    let _ = std::fs::remove_file(ltm_base.with_extension("idx"));
}

// ─────────────────────────────────────────────────────────────────
//  Numerical convergence invariants
// ─────────────────────────────────────────────────────────────────

/// Integrate the neurochemical system to `target_t` seconds using `dt`.
fn run_dt(target_t: f32, dt: f32) -> NeurochemicalVector {
    let mut vec = NeurochemicalVector::new(0);
    let params = NeuroTickParams {
        dt,
        ..NeuroTickParams::DEFAULT
    };
    let steps = (target_t / dt) as usize;
    for _ in 0..steps {
        vec.tick_with_params(&params);
    }
    vec
}

/// L2 distance between two neurochemical vectors over raw chemical levels.
fn level_l2(a: &NeurochemicalVector, b: &NeurochemicalVector) -> f32 {
    a.chemicals
        .iter()
        .zip(b.chemicals.iter())
        .map(|(ca, cb)| (ca.level - cb.level) * (ca.level - cb.level))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn prop_dt_convergence_for_neurochemical_levels() {
    let target_t = 10.0f32;
    let fine = run_dt(target_t, 0.001);
    let coarse_01 = run_dt(target_t, 0.1);
    let coarse_001 = run_dt(target_t, 0.01);
    let err_01 = level_l2(&coarse_01, &fine);
    let err_001 = level_l2(&coarse_001, &fine);

    assert!(
        err_001 < err_01,
        "dt=0.01 should be more accurate than dt=0.1 (err_001={}, err_01={})",
        err_001,
        err_01
    );
    assert!(
        err_001 < 1e-3,
        "dt=0.01 should be within 1e-3 of fine (got {})",
        err_001
    );
    assert!(
        err_01 < 1e-2,
        "dt=0.1 should be within 1e-2 of fine (got {})",
        err_01
    );
}

#[test]
fn prop_perturbation_growth_is_bounded() {
    let mut base = NeurochemicalVector::new(0);
    let mut perturbed = base;
    perturbed.chemicals[NeurochemicalId::Dopamine as usize].level += 1e-4;
    perturbed.recompute_derived();

    for _ in 0..1000 {
        base.tick_default();
        perturbed.tick_default();
    }

    let l2 = level_l2(&base, &perturbed);
    assert!(
        l2 < 1e-3,
        "1e-4 perturbation should not grow beyond 1e-3 after 1000 ticks (got {})",
        l2
    );
}

#[test]
fn prop_perturbation_growth_remains_bounded_for_random_starts() {
    let mut rng = Rng::new(12345);
    let mut max_final = 0.0f32;

    for _ in 0..20 {
        let mut base = NeurochemicalVector::new(0);

        // Randomize the starting point within the allowed ranges.
        for chem in &mut base.chemicals {
            chem.level = rng.next_f32();
            chem.baseline = rng.next_f32_range(0.1, 0.5);
            chem.tonic_level = chem.level;
        }
        base.recompute_derived();

        let mut perturbed = base;
        let mut delta = [0.0f32; NEUROCHEMICAL_COUNT];
        for d in &mut delta {
            *d = rng.next_f32_range(-1.0, 1.0);
        }
        let norm = delta.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for (i, d) in delta.iter().enumerate() {
                let scaled = *d / norm * 1e-4;
                perturbed.chemicals[i].level =
                    (perturbed.chemicals[i].level + scaled).clamp(0.0, 1.0);
            }
        }
        perturbed.recompute_derived();

        for _ in 0..500 {
            base.tick_default();
            perturbed.tick_default();
        }

        let d1 = level_l2(&base, &perturbed);
        assert!(
            d1 < 1e-2,
            "perturbation exploded for a random start (got {})",
            d1
        );
        max_final = max_final.max(d1);
    }

    // The largest perturbation seen over all random starts stays bounded.
    assert!(
        max_final < 1e-2,
        "worst-case perturbation after 500 ticks is {}",
        max_final
    );
}

#[test]
fn prop_arousal_map_exhibits_bistability_for_current_params() {
    let sigmoid = |x: f32| -> f32 {
        if x > 20.0 {
            1.0
        } else if x < -20.0 {
            0.0
        } else {
            1.0 / (1.0 + (-x).exp())
        }
    };

    let f = |arousal: f32, net_drive: f32| -> f32 {
        let input = WC_GAIN * (net_drive + WC_SELF_EXCITATION * (arousal - 0.5));
        sigmoid(input).clamp(0.0, 1.0)
    };

    let count_roots = |net_drive: f32| -> usize {
        let mut prev_g = f(0.0, net_drive) - 0.0;
        let mut roots = 0;
        for i in 1..=1000 {
            let a = i as f32 / 1000.0;
            let g = f(a, net_drive) - a;
            if g == 0.0 || (g * prev_g < 0.0) {
                roots += 1;
            }
            prev_g = g;
        }
        roots
    };

    let mut saw_three = false;
    let mut saw_one = false;
    let mut steps = 0;
    let mut d = -0.5f32;
    while d <= 0.5 {
        steps += 1;
        let roots = count_roots(d);
        assert!(
            (1..=3).contains(&roots),
            "arousal map should have 1-3 fixed points for net_drive={} (found {})",
            d,
            roots
        );
        if roots == 3 {
            saw_three = true;
        } else if roots == 1 {
            saw_one = true;
        }
        d += 0.01;
    }

    assert!(
        saw_three,
        "arousal map should be bistable for some net_drive"
    );
    assert!(
        saw_one,
        "arousal map should be monostable for some net_drive"
    );
    assert!(
        steps > 50,
        "bifurcation scan should cover a reasonable range"
    );
}
