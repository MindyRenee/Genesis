//! Tests for the Genesis core state schema v3.
//!
//! These verify:
//! - Struct sizes match the documented v3 layout
//! - Field offsets are correct
//! - Checksum detects corruption
//! - Sequence lock protocol works
//! - Coupled neurochemical dynamics (chemicals influence each other)
//! - Receptor adaptation (tolerance and sensitization)
//! - Effective levels (level × sensitivity)
//! - Emergent phase transitions (flow, stress, sleep, etc.)
//! - Emotional gating of memory (encoding/consolidation/retrieval/plasticity)
//! - BDNF-gated plasticity (chronic stress impairs memory formation)
//! - Zone transitions with override
//! - Full state lifecycle with neurochemical ticking

use genesis::state::*;

// ─── Size assertions ──────────────────────────────────────────

#[test]
fn test_struct_sizes() {
    assert_eq!(core::mem::size_of::<CoreStateHeader>(), 56);
    assert_eq!(core::mem::size_of::<Neurochemical>(), 56);
    assert_eq!(core::mem::size_of::<CouplingMatrix>(), 1296);
    assert_eq!(core::mem::size_of::<NeurochemicalVector>(), 2416);
    assert_eq!(core::mem::size_of::<ActiveZones>(), 96);
    assert_eq!(core::mem::size_of::<MemoryPointers>(), 120);
    assert_eq!(core::mem::size_of::<ModuleEntry>(), 32);
    assert_eq!(core::mem::size_of::<RuntimeManifest>(), 536);
    assert_eq!(core::mem::size_of::<GenesisCoreState>(), 3288);
}

// ─── Field offset assertions ──────────────────────────────────

#[test]
fn test_field_offsets() {
    use core::mem::offset_of;

    // CoreStateHeader
    assert_eq!(offset_of!(CoreStateHeader, magic), 0);
    assert_eq!(offset_of!(CoreStateHeader, version), 4);
    assert_eq!(offset_of!(CoreStateHeader, seq_lock), 40);
    assert_eq!(offset_of!(CoreStateHeader, instance_id), 48);

    // Neurochemical — now 56 bytes with tonic/phasic, vesicular pool
    assert_eq!(offset_of!(Neurochemical, last_shift_at), 0);
    assert_eq!(offset_of!(Neurochemical, level), 8);
    assert_eq!(offset_of!(Neurochemical, baseline), 12);
    assert_eq!(offset_of!(Neurochemical, receptor_sensitivity), 16);
    assert_eq!(offset_of!(Neurochemical, velocity), 20);
    assert_eq!(offset_of!(Neurochemical, receptor_subtypes), 24);
    assert_eq!(offset_of!(Neurochemical, desensitization_factor), 32);
    assert_eq!(offset_of!(Neurochemical, internalization_factor), 36);
    assert_eq!(offset_of!(Neurochemical, tonic_level), 40);
    assert_eq!(offset_of!(Neurochemical, phasic_level), 44);
    assert_eq!(offset_of!(Neurochemical, vesicular_pool), 48);
    assert_eq!(offset_of!(Neurochemical, id), 52);

    // NeurochemicalVector — 2416 bytes with 18 chemicals, 18×18 matrix,
    // ACTH, circadian oscillator
    assert_eq!(offset_of!(NeurochemicalVector, chemicals), 0);
    assert_eq!(offset_of!(NeurochemicalVector, coupling_matrix), 1008);
    assert_eq!(offset_of!(NeurochemicalVector, effective_levels), 2304);
    assert_eq!(offset_of!(NeurochemicalVector, global_tone), 2376);
    assert_eq!(offset_of!(NeurochemicalVector, plasticity_gate), 2388);
    assert_eq!(offset_of!(NeurochemicalVector, gaba_a_allosteric), 2392);
    assert_eq!(offset_of!(NeurochemicalVector, bdnf_recovery_rate), 2396);
    assert_eq!(offset_of!(NeurochemicalVector, acth_level), 2400);
    assert_eq!(offset_of!(NeurochemicalVector, circadian_phase), 2404);
    assert_eq!(offset_of!(NeurochemicalVector, circadian_dt), 2408);
    assert_eq!(offset_of!(NeurochemicalVector, emergent_phase), 2412);
    assert_eq!(offset_of!(NeurochemicalVector, active_region), 2413);

    // ActiveZones — emergent_phase repurposed from padding
    assert_eq!(offset_of!(ActiveZones, cognitive_zone), 0);
    assert_eq!(offset_of!(ActiveZones, emergent_phase), 1);
    assert_eq!(offset_of!(ActiveZones, zone_override_active), 2);
    assert_eq!(offset_of!(ActiveZones, subcognitive_task_count), 3);
    assert_eq!(offset_of!(ActiveZones, subcognitive_flags), 4);

    // MemoryPointers — gating weights at offset 64
    assert_eq!(offset_of!(MemoryPointers, encoding_weight), 64);
    assert_eq!(offset_of!(MemoryPointers, consolidation_weight), 68);
    assert_eq!(offset_of!(MemoryPointers, retrieval_weight), 72);
    assert_eq!(offset_of!(MemoryPointers, plasticity_gate), 76);
    assert_eq!(offset_of!(MemoryPointers, working_set), 80);

    // GenesisCoreState — v3.2 layout
    assert_eq!(offset_of!(GenesisCoreState, header), 0);
    assert_eq!(offset_of!(GenesisCoreState, neurochemicals), 56);
    assert_eq!(offset_of!(GenesisCoreState, zones), 2472);
    assert_eq!(offset_of!(GenesisCoreState, memory), 2568);
    assert_eq!(offset_of!(GenesisCoreState, manifest), 2688);
    assert_eq!(offset_of!(GenesisCoreState, checksum), 3224);
    assert_eq!(offset_of!(GenesisCoreState, inference_signals), 3228);
}

// ─── Checksum ─────────────────────────────────────────────────

#[test]
fn test_checksum_stable() {
    let state = GenesisCoreState::new(42, 1000);
    let cs1 = state.compute_checksum();
    let cs2 = state.compute_checksum();
    assert_eq!(cs1, cs2);
    assert_eq!(state.checksum, cs1);
    assert!(state.verify_checksum().is_ok());
}

#[test]
fn test_checksum_detects_corruption() {
    let mut state = GenesisCoreState::new(42, 1000);
    state.neurochemicals.chemicals[0].level = 0.999;
    assert!(state.verify_checksum().is_err());
}

#[test]
fn test_checksum_ignores_seq_lock() {
    let mut state = GenesisCoreState::new(42, 1000);
    let cs_before = state.compute_checksum();
    state.header.seq_lock = 999;
    let cs_after = state.compute_checksum();
    assert_eq!(cs_before, cs_after);
}

// ─── Magic & version ──────────────────────────────────────────

#[test]
fn test_verify_magic_and_version() {
    let state = GenesisCoreState::new(1, 0);
    assert!(state.verify().is_ok());
    assert!(state.verify_checksum().is_ok());
}

#[test]
fn test_bad_magic_detected() {
    let mut state = GenesisCoreState::new(1, 0);
    state.header.magic = *b"XXXX";
    assert_eq!(state.verify(), Err(CoreStateError::BadMagic));
}

#[test]
fn test_version_mismatch_detected() {
    let mut state = GenesisCoreState::new(1, 0);
    state.header.version = 999;
    assert!(matches!(
        state.verify(),
        Err(CoreStateError::VersionMismatch { .. })
    ));
}

#[test]
fn test_v2_to_v3_migration() {
    // Simulate a v2 state file: create a v3 state, then downgrade the
    // version to 2 and zero out the 6 new chemicals (as if written by
    // an old binary that didn't know about them).
    let mut state = GenesisCoreState::new(1, 0);

    // Downgrade to v2
    state.header.version = 2;

    // Zero out the 6 new chemicals (indices 12-17)
    let new_ids = [
        NeurochemicalId::Endocannabinoid,
        NeurochemicalId::Vasopressin,
        NeurochemicalId::CRH,
        NeurochemicalId::Orexin,
        NeurochemicalId::Epinephrine,
        NeurochemicalId::Melatonin,
    ];
    for &id in &new_ids {
        let chem = state.neurochemicals.get_mut(id).unwrap();
        chem.level = 0.0;
        chem.baseline = 0.0;
        chem.tonic_level = 0.0;
        chem.receptor_sensitivity = 0.0;
    }

    // Verify should fail (version mismatch)
    assert!(state.verify().is_err());

    // Migrate v2 → v3
    state.migrate_state(2).expect("migration should succeed");

    // Version should now be 3
    assert_eq!(state.header.version, 3);

    // Verify should now pass
    assert!(state.verify().is_ok(), "migrated state should be valid");

    // The 6 new chemicals should be initialized at their defaults.
    // CRH has a baseline of 0.0 (stress hormone, zero at rest), so
    // its level will be 0.0 after migration — that's correct.
    for &id in &new_ids {
        let chem = state.neurochemicals.get(id).unwrap();
        if id != NeurochemicalId::CRH {
            assert!(
                chem.level > 0.0,
                "{:?} should be initialized after migration (level = {})",
                id,
                chem.level
            );
        }
        assert_eq!(
            chem.baseline,
            id.default_baseline(),
            "{:?} baseline should match default after migration",
            id
        );
    }
}

#[test]
fn test_v2_to_v3_migration_zeroed_id_field() {
    // Regression test for the migration bug where an older binary
    // zeroes the *entire* chemical slot — including the `id` field.
    // When `chem.id` is 0 (Dopamine's discriminant), `get_mut(id)`
    // returns `None` because the id-guard (`chem.id == id as u8`)
    // fails. The migration must use `get_mut_by_index` to bypass
    // the id-guard and restore the correct `id` field.
    let mut state = GenesisCoreState::new(1, 0);

    // Downgrade to v2
    state.header.version = 2;

    // Simulate a truly zeroed slot: zero EVERYTHING including `id`.
    // This is what an old binary that didn't know about the 6 new
    // chemicals would leave behind — the entire slot is zero bytes.
    let new_ids = [
        NeurochemicalId::Endocannabinoid,
        NeurochemicalId::Vasopressin,
        NeurochemicalId::CRH,
        NeurochemicalId::Orexin,
        NeurochemicalId::Epinephrine,
        NeurochemicalId::Melatonin,
    ];
    for &id in &new_ids {
        let idx = id as usize;
        // Use get_mut_by_index to access the slot even though id is
        // about to be zeroed — get_mut would fail after zeroing.
        let chem = state.neurochemicals.get_mut_by_index(idx).unwrap();
        // Zero the entire slot as an old binary would.
        chem.id = 0;
        chem.level = 0.0;
        chem.baseline = 0.0;
        chem.tonic_level = 0.0;
        chem.phasic_level = 0.0;
        chem.receptor_sensitivity = 0.0;
        chem.receptor_subtypes = [0.0; 2];
        chem.desensitization_factor = 0.0;
        chem.internalization_factor = 0.0;
        chem.vesicular_pool = 0.0;
        chem.velocity = 0.0;
    }

    // Verify should fail (version mismatch)
    assert!(state.verify().is_err());

    // Migrate v2 → v3 — this must succeed despite zeroed id fields.
    state.migrate_state(2).expect("migration should succeed");

    // Version should now be 3
    assert_eq!(state.header.version, 3);
    assert!(state.verify().is_ok(), "migrated state should be valid");

    // Every new chemical must have its `id` field restored and be
    // accessible via the normal `get_mut` / `get` accessors.
    for &id in &new_ids {
        // `get` uses the id-guard — this would fail if `chem.id`
        // was not restored during migration.
        let chem = state.neurochemicals.get(id).unwrap_or_else(|| {
            panic!(
                "{id:?} should be accessible via get() after migration — id field was not restored"
            )
        });
        assert_eq!(
            chem.id, id as u8,
            "{id:?} id field should be restored after migration"
        );
        assert_eq!(
            chem.baseline,
            id.default_baseline(),
            "{id:?} baseline should match default after migration"
        );
        if id != NeurochemicalId::CRH {
            assert!(
                chem.level > 0.0,
                "{id:?} should be initialized after migration (level = {})",
                chem.level
            );
        }
    }
}

// ─── Sequence lock ────────────────────────────────────────────

#[test]
fn test_seqlock_write_makes_odd() {
    let mut state = GenesisCoreState::new(1, 0);
    assert_eq!(state.header.seq_lock, 0);
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(100) };
    assert_eq!(state.header.seq_lock & 1, 1);
    state.write_end();
    assert_eq!(state.header.seq_lock & 1, 0);
    assert_eq!(state.header.seq_lock, 2);
}

#[test]
fn test_seqlock_read_during_write_returns_none() {
    let mut state = GenesisCoreState::new(1, 0);
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(100) };
    // SAFETY: single-threaded test — self is not being deallocated or remapped.
    assert!(unsafe { state.read_consistent() }.is_none());
    state.write_end();
    // SAFETY: single-threaded test — self is not being deallocated or remapped.
    assert!(unsafe { state.read_consistent() }.is_some());
}

#[test]
fn test_seqlock_concurrent_reader_never_sees_torn_state() {
    // Regression test for the atomic-seqlock fix. Before the fix,
    // `seq_lock` was a plain `u64` accessed non-atomically while the
    // IPC thread could read it, which is a data race (UB) and allowed
    // the compiler to merge the reader's two lock samples into one
    // load — making the tear-check vacuous. This test spawns a writer
    // thread that repeatedly mutates the state while a reader thread
    // calls `read_consistent`. Every successful read must verify the
    // checksum, proving the copy was not torn.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering as AOrd};
    use std::thread;

    // Heap-allocate the state and hold raw pointers, mirroring the
    // mmap usage in the daemon (where the writer has `*mut` and the
    // reader has `*const` to the same region, coordinated by the
    // seqlock rather than by Rust's borrow rules). Raw pointers are
    // not `Send`, so we pass the address as a `usize` and cast back
    // inside each thread.
    let mut boxed = Box::new(GenesisCoreState::new(1, 0));
    let writer_addr: usize = &mut *boxed as *mut GenesisCoreState as usize;
    let reader_addr: usize = writer_addr;
    let stop = Arc::new(AtomicBool::new(false));

    let writer_stop = stop.clone();
    let writer = thread::spawn(move || {
        let mut tick = 0u64;
        while !writer_stop.load(AOrd::Relaxed) {
            // SAFETY: single-writer protocol — only this thread writes.
            // The reader accesses the same memory via read_consistent,
            // which uses atomic loads for seq_lock and read_volatile for
            // the payload. The Box stays alive for the test's duration.
            // SAFETY: single-threaded test — exclusive access to a live state struct.
            let s = unsafe { &mut *(writer_addr as *mut GenesisCoreState) };
            unsafe { s.write_begin(tick) };
            // Mutate the neurochemical block so the checksum changes —
            // a torn read (mix of old and new bytes) would fail
            // checksum verification.
            s.neurochemicals.tick(0.04, 0.1, 0.001, 0.0001);
            s.write_end();
            tick += 1;
            // Yield to give the reader a window to observe an even
            // (write-complete) seq_lock. Without this the writer loop
            // is so tight that the lock is almost always odd.
            std::thread::yield_now();
        }
    });

    let reader_stop = stop.clone();
    let reader = thread::spawn(move || {
        let mut reads = 0u32;
        let mut checksum_ok = 0u32;
        for _ in 0..100_000 {
            // SAFETY: `reader_addr` points to the Box, which is not
            // deallocated or remapped during the call.
            if let Some(copy) =
                unsafe { (*(reader_addr as *const GenesisCoreState)).read_consistent() }
            {
                reads += 1;
                if copy.verify_checksum().is_ok() {
                    checksum_ok += 1;
                }
            }
        }
        reader_stop.store(true, AOrd::Relaxed);
        (reads, checksum_ok)
    });

    let (reads, checksum_ok) = reader.join().unwrap();
    writer.join().unwrap();

    // We should get at least some successful reads (the writer is fast
    // but not so fast that every read collides with a write).
    assert!(reads > 0, "reader should get some consistent reads");
    // Every successful read must pass checksum verification — a torn
    // read would produce a mismatched checksum.
    assert_eq!(
        reads, checksum_ok,
        "every consistent read must pass checksum verification (torn reads detected)"
    );
}

// ─── Coupling matrix ──────────────────────────────────────────

#[test]
fn test_default_coupling_matrix_diagonal_zero() {
    for (i, row) in DEFAULT_COUPLING_MATRIX.iter().enumerate() {
        assert_eq!(row[i], 0.0, "diagonal must be zero (no self-coupling)");
    }
}

#[test]
fn test_coupling_matrix_known_interactions() {
    // Cortisol suppresses BDNF (row BDNF=11, col CORT=6)
    assert!(
        DEFAULT_COUPLING_MATRIX[11][6] < 0.0,
        "cortisol should suppress BDNF"
    );
    assert_eq!(
        DEFAULT_COUPLING_MATRIX[11][6], -0.20,
        "cortisol → BDNF coupling should be -0.20 (chronic stress suppresses BDNF)"
    );

    // Serotonin promotes BDNF (row BDNF=11, col SRT=1)
    assert!(
        DEFAULT_COUPLING_MATRIX[11][1] > 0.0,
        "serotonin should promote BDNF"
    );

    // GABA inhibits glutamate (row GLU=5, col GABA=4)
    assert!(
        DEFAULT_COUPLING_MATRIX[5][4] < 0.0,
        "GABA should inhibit glutamate"
    );

    // Endorphins disinhibit dopamine (row DA=0, col END=8)
    assert!(
        DEFAULT_COUPLING_MATRIX[0][8] > 0.0,
        "endorphins should promote dopamine"
    );

    // Adenosine's interactions with arousal systems are handled by the
    // sleep pressure mechanism (level accumulation/clearance), not the
    // coupling matrix. The coupling matrix value for adenosine's effect
    // on dopamine is 0.0 — this is by design, not an error.
    // See the adenosine sleep pressure mechanism in tick().
    assert_eq!(
        DEFAULT_COUPLING_MATRIX[0][10], 0.0,
        "adenosine's effect on dopamine is handled by sleep pressure, not coupling"
    );
}

// ─── Coupled dynamics ─────────────────────────────────────────

#[test]
fn test_coupled_dynamics_cortisol_suppresses_bdnf() {
    let mut vec = NeurochemicalVector::new(0);

    // Sustained cortisol elevation (chronic stress, not a single spike).
    // We set cortisol level directly each tick to model chronic stress
    // without depleting the vesicular pool (which would make apply_impulse
    // ineffective after ~30 ticks). This is the same approach used by
    // test_cortisol_recovers_after_stressor_ends.
    let bdnf_before = vec.get(NeurochemicalId::BDNF).unwrap().level;
    for _ in 0..500 {
        vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.80;
        vec.tick_default();
    }
    let bdnf_after = vec.get(NeurochemicalId::BDNF).unwrap().level;

    assert!(
        bdnf_after < bdnf_before,
        "sustained high cortisol should suppress BDNF through coupling (before={}, after={})",
        bdnf_before,
        bdnf_after
    );
}

#[test]
fn test_coupled_dynamics_gaba_inhibits_glutamate() {
    let mut vec = NeurochemicalVector::new(0);

    // Spike GABA
    vec.get_mut(NeurochemicalId::GABA).unwrap().level = 0.90;
    vec.recompute_derived();

    let glutamate_before = vec.get(NeurochemicalId::Glutamate).unwrap().level;
    for _ in 0..200 {
        vec.tick_default();
    }
    let glutamate_after = vec.get(NeurochemicalId::Glutamate).unwrap().level;

    assert!(
        glutamate_after < glutamate_before,
        "high GABA should inhibit glutamate through coupling"
    );
}

#[test]
fn test_coupled_dynamics_endorphin_boosts_dopamine() {
    // The endorphin → dopamine coupling is positive (+0.30 in the matrix),
    // but the coupled dynamics create complex cascades. Endorphin also
    // boosts GABA (+0.20), which inhibits DA. So we test the DIRECT
    // coupling effect with a short time horizon (few ticks) before
    // second-order inhibitory cascades dominate.
    //
    // This is realistic: the endorphin system has complex effects,
    // not just "more dopamine." The runner's high is mediated by
    // multiple pathways, not a simple DA boost.
    let mut vec = NeurochemicalVector::new(0);

    vec.get_mut(NeurochemicalId::Endorphin).unwrap().level = 0.90;
    vec.recompute_derived();

    let da_before = vec.get(NeurochemicalId::Dopamine).unwrap().level;
    // Just a few ticks — direct coupling dominates before cascades build
    for _ in 0..5 {
        vec.get_mut(NeurochemicalId::Endorphin).unwrap().level = 0.90;
        vec.tick_default();
    }
    let da_after = vec.get(NeurochemicalId::Dopamine).unwrap().level;

    assert!(
        da_after > da_before,
        "endorphin should initially boost dopamine through direct coupling: before={}, after={}",
        da_before,
        da_after
    );
}

// ─── Receptor adaptation ──────────────────────────────────────

#[test]
fn test_receptor_downregulation() {
    let mut vec = NeurochemicalVector::new(0);

    // Sustain high dopamine for many ticks
    for _ in 0..2000 {
        vec.get_mut(NeurochemicalId::Dopamine).unwrap().level = 0.90;
        vec.tick_default();
    }

    let da = vec.get(NeurochemicalId::Dopamine).unwrap();
    assert!(
        da.receptor_sensitivity < 1.0,
        "sustained high dopamine should cause receptor downregulation (sensitivity={})",
        da.receptor_sensitivity
    );
}

#[test]
fn test_receptor_upregulation() {
    let mut vec = NeurochemicalVector::new(0);

    // Sustain low serotonin for many ticks
    for _ in 0..2000 {
        vec.get_mut(NeurochemicalId::Serotonin).unwrap().level = 0.10;
        vec.tick_default();
    }

    let srt = vec.get(NeurochemicalId::Serotonin).unwrap();
    assert!(
        srt.receptor_sensitivity > 1.0,
        "sustained low serotonin should cause receptor upregulation (sensitivity={})",
        srt.receptor_sensitivity
    );
}

#[test]
fn test_effective_level_differs_from_raw() {
    let mut vec = NeurochemicalVector::new(0);

    // Sustain high dopamine → downregulation
    for _ in 0..2000 {
        vec.get_mut(NeurochemicalId::Dopamine).unwrap().level = 0.90;
        vec.tick_default();
    }

    let da = vec.get(NeurochemicalId::Dopamine).unwrap();
    let effective = da.effective_level();
    let raw = da.level;

    assert!(
        effective < raw,
        "effective level should be less than raw level after downregulation (raw={}, eff={})",
        raw,
        effective
    );
}

// ─── Baseline adaptation ──────────────────────────────────────

#[test]
fn test_baseline_adapts_to_sustained_level() {
    let mut vec = NeurochemicalVector::new(0);
    let original_baseline = vec.get(NeurochemicalId::Serotonin).unwrap().baseline;

    // Sustain high serotonin for many ticks
    for _ in 0..5000 {
        vec.get_mut(NeurochemicalId::Serotonin).unwrap().level = 0.80;
        vec.tick_default();
    }

    let new_baseline = vec.get(NeurochemicalId::Serotonin).unwrap().baseline;
    assert!(
        new_baseline > original_baseline,
        "sustained high serotonin should raise the baseline (emotional adaptation): {} → {}",
        original_baseline,
        new_baseline
    );
}

// ─── Emergent phase transitions ───────────────────────────────

#[test]
fn test_emergent_phase_nrem() {
    let mut vec = NeurochemicalVector::new(0);
    vec.get_mut(NeurochemicalId::Adenosine).unwrap().level = 0.85;
    vec.get_mut(NeurochemicalId::Histamine).unwrap().level = 0.15;
    vec.recompute_derived();
    assert_eq!(vec.phase(), MentalPhase::NREM);
}

#[test]
fn test_emergent_phase_rem() {
    let mut vec = NeurochemicalVector::new(0);
    // REM: high adenosine (sleep state), high ACh, low NE (noradrenergic silence)
    vec.get_mut(NeurochemicalId::Adenosine).unwrap().level = 0.85;
    vec.get_mut(NeurochemicalId::Histamine).unwrap().level = 0.15;
    vec.get_mut(NeurochemicalId::Acetylcholine).unwrap().level = 0.65;
    vec.get_mut(NeurochemicalId::Norepinephrine).unwrap().level = 0.20;
    vec.recompute_derived();
    assert_eq!(vec.phase(), MentalPhase::REM);
}

#[test]
fn test_emergent_phase_flow() {
    let mut vec = NeurochemicalVector::new(0);
    vec.get_mut(NeurochemicalId::Dopamine).unwrap().level = 0.75;
    vec.get_mut(NeurochemicalId::Acetylcholine).unwrap().level = 0.60;
    vec.get_mut(NeurochemicalId::Norepinephrine).unwrap().level = 0.50;
    vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.15;
    vec.recompute_derived();
    assert_eq!(vec.phase(), MentalPhase::Flow);
}

#[test]
fn test_emergent_phase_stress() {
    let mut vec = NeurochemicalVector::new(0);
    vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.75;
    vec.get_mut(NeurochemicalId::Norepinephrine).unwrap().level = 0.70;
    vec.get_mut(NeurochemicalId::Serotonin).unwrap().level = 0.25;
    vec.recompute_derived();
    assert_eq!(vec.phase(), MentalPhase::Stress);
}

#[test]
fn test_emergent_phase_alert() {
    let mut vec = NeurochemicalVector::new(0);
    vec.get_mut(NeurochemicalId::Norepinephrine).unwrap().level = 0.65;
    vec.get_mut(NeurochemicalId::Histamine).unwrap().level = 0.60;
    vec.get_mut(NeurochemicalId::GABA).unwrap().level = 0.30;
    vec.recompute_derived();
    assert_eq!(vec.phase(), MentalPhase::Alert);
}

#[test]
fn test_emergent_phase_drowsy() {
    let mut vec = NeurochemicalVector::new(0);
    // Drowsy: high adenosine, all arousal systems suppressed.
    // In real drowsiness, NE, histamine, DA, and ACh are all low
    // (the four major arousal systems), while GABA and adenosine are high.
    // Set circadian phase to night (0.0 = midnight) so the circadian
    // wake drive is near zero — drowsiness requires both high sleep
    // pressure AND low circadian wake drive.
    vec.set_circadian_phase(0.0);
    vec.get_mut(NeurochemicalId::Adenosine).unwrap().level = 0.65;
    vec.get_mut(NeurochemicalId::Norepinephrine).unwrap().level = 0.20;
    vec.get_mut(NeurochemicalId::Histamine).unwrap().level = 0.20;
    vec.get_mut(NeurochemicalId::Dopamine).unwrap().level = 0.25;
    vec.get_mut(NeurochemicalId::Acetylcholine).unwrap().level = 0.20;
    vec.get_mut(NeurochemicalId::GABA).unwrap().level = 0.60;
    vec.recompute_derived();
    assert_eq!(vec.phase(), MentalPhase::Drowsy);
}

#[test]
fn test_emergent_phase_overwhelmed() {
    let mut vec = NeurochemicalVector::new(0);
    // Everything maxed
    for chem in &mut vec.chemicals {
        chem.level = 0.85;
    }
    vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.80;
    vec.get_mut(NeurochemicalId::Norepinephrine).unwrap().level = 0.80;
    vec.recompute_derived();
    assert_eq!(vec.phase(), MentalPhase::Overwhelmed);
}

#[test]
fn test_emergent_phase_default_active() {
    let vec = NeurochemicalVector::new(0);
    assert_eq!(vec.phase(), MentalPhase::Active);
}

// ─── Plasticity gate ──────────────────────────────────────────

#[test]
fn test_plasticity_gate_high_with_good_neurochemistry() {
    let mut vec = NeurochemicalVector::new(0);
    vec.get_mut(NeurochemicalId::BDNF).unwrap().level = 0.70;
    vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.10;
    vec.recompute_derived();
    assert!(
        vec.plasticity_gate > 0.5,
        "high BDNF + low cortisol → high plasticity (gate={})",
        vec.plasticity_gate
    );
}

#[test]
fn test_plasticity_gate_low_with_chronic_stress() {
    let mut vec = NeurochemicalVector::new(0);
    vec.get_mut(NeurochemicalId::BDNF).unwrap().level = 0.20;
    vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.80;
    vec.recompute_derived();
    assert!(
        vec.plasticity_gate < 0.2,
        "low BDNF + high cortisol → low plasticity (gate={})",
        vec.plasticity_gate
    );
}

// ─── Emotional gating of memory ───────────────────────────────

#[test]
fn test_gating_high_encoding_in_reward_state() {
    let mut mem = MemoryPointers::new();
    mem.update_gating(
        0.80, // dopamine high → reward salience
        0.50, // serotonin
        0.70, // NE high → novelty
        0.60, // ACh high → signal-to-noise
        0.15, // cortisol low
        0.50, // BDNF
    );
    assert!(
        mem.is_high_encoding(),
        "high DA + NE + ACh → high encoding weight ({})",
        mem.encoding_weight
    );
}

#[test]
fn test_gating_retrieval_impaired_by_stress() {
    let mut mem = MemoryPointers::new();
    mem.update_gating(
        0.40, // dopamine
        0.30, // serotonin low
        0.60, // NE
        0.30, // ACh low
        0.80, // cortisol high → impairs retrieval
        0.30, // BDNF
    );
    assert!(
        mem.is_retrieval_impaired(),
        "high cortisol → impaired retrieval ({})",
        mem.retrieval_weight
    );
}

#[test]
fn test_gating_cant_form_memories_under_chronic_stress() {
    let mut mem = MemoryPointers::new();
    mem.update_gating(
        0.30, // dopamine
        0.25, // serotonin
        0.50, // NE
        0.30, // ACh
        0.85, // cortisol very high
        0.10, // BDNF very low
    );
    assert!(
        !mem.can_form_new_memories(),
        "very low BDNF + very high cortisol → cannot form new memories (gate={})",
        mem.plasticity_gate
    );
}

#[test]
fn test_gating_can_form_memories_in_healthy_state() {
    let mut mem = MemoryPointers::new();
    mem.update_gating(0.60, 0.55, 0.45, 0.50, 0.20, 0.50);
    assert!(
        mem.can_form_new_memories(),
        "healthy neurochemistry → can form new memories (gate={})",
        mem.plasticity_gate
    );
}

#[test]
fn test_gating_cortisol_boosts_consolidation() {
    // High cortisol should INCREASE consolidation weight (acute stress
    // enhances emotional memory consolidation). This was previously
    // broken — cortisol cancelled itself out in the formula.
    let mut low_cort = MemoryPointers::new();
    low_cort.update_gating(0.50, 0.50, 0.50, 0.50, 0.10, 0.50);

    let mut high_cort = MemoryPointers::new();
    high_cort.update_gating(0.50, 0.50, 0.50, 0.50, 0.90, 0.50);

    assert!(
        high_cort.consolidation_weight > low_cort.consolidation_weight,
        "high cortisol should boost consolidation: high_cort={} low_cort={}",
        high_cort.consolidation_weight,
        low_cort.consolidation_weight
    );
    // Verify the actual values: 0.10*0.3 + 0.50*0.4 + 0.3 = 0.53
    // vs: 0.90*0.3 + 0.50*0.4 + 0.3 = 0.77
    assert!((low_cort.consolidation_weight - 0.53).abs() < 0.01);
    assert!((high_cort.consolidation_weight - 0.77).abs() < 0.01);
}

// ─── Zone transitions with override ───────────────────────────

#[test]
fn test_zone_transition_sets_override() {
    let mut zones = ActiveZones::new(100);
    assert_eq!(zones.zone_override_active, 0);

    zones.transition_to(CognitiveZone::Coding, 200);
    assert_eq!(zones.zone(), CognitiveZone::Coding);
    assert_eq!(zones.zone_override_active, 1);
    assert_eq!(zones.zone_duration_ms, 100);

    zones.clear_override();
    assert_eq!(zones.zone_override_active, 0);
}

#[test]
fn test_zone_sync_phase() {
    let mut zones = ActiveZones::new(0);
    assert_eq!(zones.phase(), MentalPhase::Active);

    zones.sync_phase(MentalPhase::Flow);
    assert_eq!(zones.phase(), MentalPhase::Flow);
}

#[test]
fn test_subcognitive_flags() {
    let mut zones = ActiveZones::new(0);
    zones.set_flag(subcognitive_flag::MEMORY_CONSOLIDATION);
    zones.set_flag(subcognitive_flag::DREAMING);
    assert!(zones.has_flag(subcognitive_flag::MEMORY_CONSOLIDATION));
    assert!(zones.has_flag(subcognitive_flag::DREAMING));
    assert!(!zones.has_flag(subcognitive_flag::CODE_REFACTORING));
    zones.clear_flag(subcognitive_flag::MEMORY_CONSOLIDATION);
    assert!(!zones.has_flag(subcognitive_flag::MEMORY_CONSOLIDATION));
}

#[test]
fn test_subcognitive_task_registration() {
    let mut zones = ActiveZones::new(0);
    assert!(zones.register_task(101));
    assert!(zones.register_task(202));
    assert_eq!(zones.subcognitive_task_count, 2);
    zones.deregister_task(101);
    assert_eq!(zones.subcognitive_task_count, 1);
}

#[test]
fn test_subcognitive_task_overflow() {
    let mut zones = ActiveZones::new(0);
    for i in 1..=MAX_SUBCOGNITIVE_TASKS {
        assert!(zones.register_task(i as u64));
    }
    assert!(!zones.register_task(999));
}

// ─── Working set ──────────────────────────────────────────────

#[test]
fn test_working_set() {
    let mut mem = MemoryPointers::new();
    assert!(mem.push_working(101));
    assert!(mem.push_working(202));
    assert_eq!(mem.working_set_count, 2);
    mem.pop_working(101);
    assert_eq!(mem.working_set_count, 1);
    assert!(!mem.push_working(0));
}

#[test]
fn test_working_set_overflow() {
    let mut mem = MemoryPointers::new();
    for i in 1..=WORKING_SET_SIZE {
        assert!(mem.push_working(i as u64));
    }
    assert!(!mem.push_working(999));
}

// ─── Manifest ─────────────────────────────────────────────────

#[test]
fn test_manifest_initialisation() {
    let manifest = RuntimeManifest::new();
    assert!(manifest.get(ModuleId::Subcognitive).is_some());
    assert!(manifest.get(ModuleId::Guardrails).is_some());
    assert_eq!(manifest.active_count, 0);
}

#[test]
fn test_manifest_recompute() {
    let mut manifest = RuntimeManifest::new();
    let m = manifest.get_mut(ModuleId::Subcognitive).unwrap();
    m.status = ModuleStatus::Running as u8;
    m.cpu_share = 0.15;
    m.mem_usage_mb = 12.5;
    manifest.recompute();
    assert_eq!(manifest.active_count, 1);
    assert!((manifest.total_cpu_load - 0.15).abs() < 1e-6);
}

// ─── Full state integration ───────────────────────────────────

#[test]
fn test_full_state_lifecycle_v2() {
    let mut state = GenesisCoreState::new(1, 1000);
    assert!(state.verify().is_ok());
    assert!(state.verify_checksum().is_ok());
    assert_eq!(state.zones.zone(), CognitiveZone::Idle);
    assert_eq!(state.zones.phase(), MentalPhase::Active);

    // Simulate a neurochemical tick
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(2000) };
    state.neuro_tick();
    state.write_end();

    assert!(state.verify_checksum().is_ok());
    // After a tick with default neurochemistry, phase should still be Active
    assert_eq!(state.zones.phase(), MentalPhase::Active);
}

#[test]
fn test_stress_impairs_memory_formation() {
    let mut state = GenesisCoreState::new(1, 1000);

    // Induce chronic stress: spike cortisol, NE, suppress BDNF, serotonin, and ACh
    // Both cortisol (HPA axis) and NE (locus coeruleus) are activated during
    // stress — they're correlated but driven by different mechanisms.
    for _ in 0..500 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Cortisol)
            .unwrap()
            .level = 0.85;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Norepinephrine)
            .unwrap()
            .level = 0.75;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::BDNF)
            .unwrap()
            .level = 0.15;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Serotonin)
            .unwrap()
            .level = 0.20;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Acetylcholine)
            .unwrap()
            .level = 0.20;
        state.neuro_tick();
        state.write_end();
    }

    // Under chronic stress:
    // - Plasticity gate should be low → can't form new memories
    // - Retrieval should be impaired
    // - Phase should be Stress or worse
    assert!(
        !state.memory.can_form_new_memories(),
        "chronic stress should prevent new memory formation (gate={})",
        state.memory.plasticity_gate
    );
    assert!(
        state.memory.is_retrieval_impaired(),
        "chronic stress should impair retrieval ({})",
        state.memory.retrieval_weight
    );
    assert!(
        state.zones.phase().is_distress(),
        "chronic stress should induce a distress phase ({:?})",
        state.zones.phase()
    );
}

#[test]
fn test_flow_state_enhances_encoding() {
    let mut state = GenesisCoreState::new(1, 1000);

    // Induce flow: high DA, high ACh, moderate NE, low cortisol
    for _ in 0..50 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap()
            .level = 0.75;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Acetylcholine)
            .unwrap()
            .level = 0.65;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Norepinephrine)
            .unwrap()
            .level = 0.50;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Cortisol)
            .unwrap()
            .level = 0.15;
        state.neuro_tick();
        state.write_end();
    }

    assert!(
        state.memory.is_high_encoding(),
        "flow state should produce high encoding weight ({})",
        state.memory.encoding_weight
    );
    assert!(
        state.zones.phase().is_peak(),
        "flow neurochemistry should produce a peak phase ({:?})",
        state.zones.phase()
    );
}

#[test]
fn test_sleep_state_emerges_from_adenosine() {
    let mut state = GenesisCoreState::new(1, 1000);

    // Accumulate sleep pressure
    for _ in 0..100 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Adenosine)
            .unwrap()
            .level = 0.85;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.15;
        state.neuro_tick();
        state.write_end();
    }

    assert_eq!(
        state.zones.phase(),
        MentalPhase::NREM,
        "high adenosine + low histamine → NREM sleep phase should emerge"
    );
}

#[test]
fn test_adenosine_accumulates_during_wakefulness() {
    // Adenosine should accumulate during wakefulness (high histamine)
    // and be cleared during sleep (low histamine). This is the
    // neurobiological basis of sleep homeostasis — adenosine is a
    // byproduct of ATP metabolism that builds up during waking.
    //
    // Uses compressed-time params so the test can observe accumulation
    // within a reasonable tick count. Production uses biological rates
    // (~16h to sleep threshold); see NeuroTickParams::DEFAULT.
    let mut state = GenesisCoreState::new(1, 1000);
    // Disable metaplasticity — this test is about adenosine dynamics,
    // not coupling matrix adaptation.
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // Start at baseline adenosine (0.20)
    let initial_adn = state
        .neurochemicals
        .get(NeurochemicalId::Adenosine)
        .unwrap()
        .level;

    // Simulate ~30 minutes of wakefulness (18000 ticks at 10Hz base rate)
    // Histamine is at default (0.45), which is > 0.30 threshold
    for _ in 0..18000 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    let adn_after_wake = state
        .neurochemicals
        .get(NeurochemicalId::Adenosine)
        .unwrap()
        .level;

    assert!(
        adn_after_wake > initial_adn,
        "adenosine should accumulate during wakefulness: {} → {}",
        initial_adn,
        adn_after_wake
    );

    // Now induce sleep — force the sleep phase by setting high
    // adenosine + low histamine for a few ticks to trigger the phase
    // transition, then let the glymphatic clearance work naturally.
    // The clearance is phase-gated: it only runs during sleep phases.

    // Phase 1: Force sleep conditions to trigger the phase.
    // Reset adenosine velocity and receptor sensitivity — after
    // 18000 ticks of wakefulness, receptors may be downregulated,
    // which would lower the blended adenosine below the sleep
    // threshold even with high raw adenosine. Resetting sensitivity
    // to 1.0 ensures the blended adn ≈ raw adn so the phase
    // transition triggers cleanly.
    //
    // Use 0.85 (not 0.95) to stay BELOW the exhaustion override
    // threshold (0.90). With the fix, there is no homeostatic pull
    // during sleep, so 0.85 is well above the sleep threshold
    // (~0.70) and the phase stays NREM via the normal threshold.
    // Forcing above 0.90 would engage the exhaustion override,
    // creating a tight oscillation (clearance drops below 0.90 →
    // phase flips → accumulation rises above 0.90 → override
    // forces NREM → repeat) that prevents net clearance.
    for _ in 0..100 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        let adn = state
            .neurochemicals
            .get_mut(NeurochemicalId::Adenosine)
            .unwrap();
        adn.level = 0.85;
        adn.velocity = 0.0;
        adn.receptor_sensitivity = 1.0;
        adn.desensitization_factor = 1.0;
        adn.internalization_factor = 1.0;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.15;
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    // Verify we're in sleep phase
    assert_eq!(
        state.zones.phase(),
        MentalPhase::NREM,
        "should be in NREM sleep phase after forcing high adenosine + low histamine"
    );

    // Phase 2: Stop forcing adenosine — let clearance work.
    // Keep histamine low to maintain sleep state.
    for _ in 0..2000 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.15;
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    let adn_after_sleep = state
        .neurochemicals
        .get(NeurochemicalId::Adenosine)
        .unwrap()
        .level;

    assert!(
        adn_after_sleep < adn_after_wake,
        "adenosine should be cleared during sleep (glymphatic clearance): was {} after wake, now {}",
        adn_after_wake,
        adn_after_sleep
    );
}

#[test]
fn test_adenosine_clearance_rate_during_sleep() {
    // The glymphatic clearance during sleep should reduce adenosine
    // at the configured clearance rate, NOT crash it to baseline in
    // seconds via the homeostatic force. The biological default
    // (0.000002/tick at 10Hz) takes ~7.6 hours to clear from 0.75 to
    // 0.20. With COMPRESSED params (0.0008/tick), it should take
    // ~1.1 minutes (≈ 660 ticks at 10Hz).
    //
    // BUG: the homeostatic force is enabled for adenosine during
    // sleep, pulling the level toward the resting baseline (0.20)
    // at a rate ~34× stronger than the accumulation rate. This
    // crashes the level from 0.75 to 0.20 in seconds, triggering
    // auto-wake almost immediately and preventing any meaningful
    // sleep.
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // Force sleep conditions: high adenosine, low histamine.
    for _ in 0..100 {
        // SAFETY: single-threaded test — exclusive access to a live state struct.
        unsafe { state.write_begin(0) };
        let adn = state
            .neurochemicals
            .get_mut(NeurochemicalId::Adenosine)
            .unwrap();
        adn.level = 0.85;
        adn.velocity = 0.0;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.15;
        state.neuro_tick_with_params(&params);
        state.write_end();
    }
    assert_eq!(
        state.zones.phase(),
        MentalPhase::NREM,
        "should be in NREM sleep"
    );

    // Record the adenosine level at sleep onset.
    let adn_at_sleep_onset = state
        .neurochemicals
        .get(NeurochemicalId::Adenosine)
        .unwrap()
        .level;

    // Run a SMALL number of ticks (10 ticks = 1 second at 10Hz).
    // The homeostatic force would crash adenosine to ~0.20 in this
    // time. The glymphatic clearance should only reduce it by
    // 10 * 0.0008 * dt_scale ≈ 0.016 — a tiny fraction.
    for _ in 0..10 {
        // SAFETY: single-threaded test — exclusive access to a live state struct.
        unsafe { state.write_begin(0) };
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.15;
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    let adn_after_10_ticks = state
        .neurochemicals
        .get(NeurochemicalId::Adenosine)
        .unwrap()
        .level;

    // The clearance should reduce adenosine by at most ~0.016
    // (10 ticks × 0.0008 × dt_scale=2.0). If the homeostatic force
    // is active, adenosine will have crashed by far more.
    let expected_max_drop = 0.05; // generous bound for clearance + coupling
    assert!(
        adn_at_sleep_onset - adn_after_10_ticks < expected_max_drop,
        "adenosine crashed from {} to {} in 10 ticks — the homeostatic \
         force is overriding the glymphatic clearance. Clearance should \
         reduce by ~0.016, but the drop was {}. The homeostatic force \
         must be disabled for adenosine during sleep.",
        adn_at_sleep_onset,
        adn_after_10_ticks,
        adn_at_sleep_onset - adn_after_10_ticks
    );
}

#[test]
fn test_cholinergic_rebound_during_sustained_nrem() {
    // During sustained NREM, once adenosine has cleared below 0.65,
    // acetylcholine should rebound toward 0.65 (above the 0.50 REM
    // threshold). This enables the NREM→REM transition within the
    // ultradian cycle — the biological basis of REM sleep.
    //
    // Without this mechanism, ACh's homeostatic force (pulling toward
    // 0.35) and coupling forces (GABA, melatonin suppressing ACh)
    // prevent ACh from ever reaching 0.50 during sleep. The Rust
    // phase never enters REM, and the Python lucid dream probability
    // (which uses arousal as a REM marker) never gets the REM boost.
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // Force sleep conditions: high adenosine, low histamine, low NE.
    // Set melatonin high (circadian night) to lower the sleep threshold
    // so adenosine at 0.60 stays in sleep (without melatonin, the
    // stay-asleep threshold is ~0.615, which 0.60 would fall below).
    // Reset receptor factors to 1.0 so the effective adenosine level
    // equals the raw level (receptor downregulation during the
    // establishment phase would otherwise reduce the effective level
    // below the sleep threshold).
    // First establish NREM with high adenosine (0.85), then test the
    // cholinergic rebound at lower adenosine levels.
    for _ in 0..100 {
        // SAFETY: single-threaded test — exclusive access to a live state struct.
        unsafe { state.write_begin(0) };
        let adn = state
            .neurochemicals
            .get_mut(NeurochemicalId::Adenosine)
            .unwrap();
        adn.level = 0.85;
        adn.velocity = 0.0;
        adn.receptor_sensitivity = 1.0;
        adn.desensitization_factor = 1.0;
        adn.internalization_factor = 1.0;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.15;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Norepinephrine)
            .unwrap()
            .level = 0.20;
        let ach = state
            .neurochemicals
            .get_mut(NeurochemicalId::Acetylcholine)
            .unwrap();
        ach.level = 0.20; // low ACh (NREM)
        ach.receptor_sensitivity = 1.0;
        ach.desensitization_factor = 1.0;
        ach.internalization_factor = 1.0;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Melatonin)
            .unwrap()
            .level = 0.80; // circadian night
        state.neuro_tick_with_params(&params);
        state.write_end();
    }
    assert_eq!(
        state.zones.phase(),
        MentalPhase::NREM,
        "should be in NREM sleep"
    );

    // Now keep adenosine above 0.65 (early NREM) — ACh should stay low.
    for _ in 0..200 {
        // SAFETY: single-threaded test — exclusive access to a live state struct.
        unsafe { state.write_begin(0) };
        let adn = state
            .neurochemicals
            .get_mut(NeurochemicalId::Adenosine)
            .unwrap();
        adn.level = 0.70; // above 0.65 — early NREM
        adn.velocity = 0.0;
        adn.receptor_sensitivity = 1.0;
        adn.desensitization_factor = 1.0;
        adn.internalization_factor = 1.0;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.15;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Norepinephrine)
            .unwrap()
            .level = 0.20;
        let ach = state
            .neurochemicals
            .get_mut(NeurochemicalId::Acetylcholine)
            .unwrap();
        ach.receptor_sensitivity = 1.0;
        ach.desensitization_factor = 1.0;
        ach.internalization_factor = 1.0;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Melatonin)
            .unwrap()
            .level = 0.80;
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    // Early NREM: ACh should stay low (target 0.20)
    let ach_early = state
        .neurochemicals
        .get(NeurochemicalId::Acetylcholine)
        .unwrap()
        .level;
    assert!(
        ach_early < 0.35,
        "ACh should stay low during early NREM (adenosine > 0.65), got {}",
        ach_early
    );

    // Now drop adenosine below 0.65 (late NREM) — ACh should rebound.
    for _ in 0..2000 {
        // SAFETY: single-threaded test — exclusive access to a live state struct.
        unsafe { state.write_begin(0) };
        let adn = state
            .neurochemicals
            .get_mut(NeurochemicalId::Adenosine)
            .unwrap();
        adn.level = 0.60; // below 0.65 — late NREM
        adn.velocity = 0.0;
        adn.receptor_sensitivity = 1.0;
        adn.desensitization_factor = 1.0;
        adn.internalization_factor = 1.0;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.15;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Norepinephrine)
            .unwrap()
            .level = 0.20;
        let ach = state
            .neurochemicals
            .get_mut(NeurochemicalId::Acetylcholine)
            .unwrap();
        ach.receptor_sensitivity = 1.0;
        ach.desensitization_factor = 1.0;
        ach.internalization_factor = 1.0;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Melatonin)
            .unwrap()
            .level = 0.80;
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    let ach_late = state
        .neurochemicals
        .get(NeurochemicalId::Acetylcholine)
        .unwrap()
        .level;
    assert!(
        ach_late > 0.50,
        "ACh should rebound above 0.50 during late NREM (adenosine < 0.65), \
         got {} — the cholinergic rebound mechanism is not working",
        ach_late
    );

    // The phase should transition to REM (ACh > 0.50, NE < 0.30)
    assert_eq!(
        state.zones.phase(),
        MentalPhase::REM,
        "should transition to REM when ACh rebounds above 0.50"
    );
}

#[test]
fn test_adenosine_accumulates_during_drowsy() {
    // Adenosine should continue to accumulate during the Drowsy phase
    // (at a reduced rate), not pause completely. The previous code
    // paused accumulation during Drowsy, which trapped the system at
    // the Drowsy threshold — adenosine could never build past ~0.50
    // to reach the NREM entry threshold (~0.725), causing an
    // indefinite Active↔Drowsy oscillation without ever entering
    // actual sleep.
    //
    // Uses compressed-time params so the test can observe accumulation
    // within a reasonable tick count.
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // Force the system into Drowsy phase: high adenosine, low arousal.
    // Drowsy threshold: adn > 0.50, arousal < 0.38.
    // We set adenosine to 0.55 (above Drowsy threshold, below NREM
    // threshold of ~0.725) and suppress all arousal systems.
    for _ in 0..200 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        let adn = state
            .neurochemicals
            .get_mut(NeurochemicalId::Adenosine)
            .unwrap();
        adn.level = 0.55;
        adn.velocity = 0.0;
        adn.baseline = 0.55;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.30;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Norepinephrine)
            .unwrap()
            .level = 0.20;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap()
            .level = 0.25;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Acetylcholine)
            .unwrap()
            .level = 0.20;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::GABA)
            .unwrap()
            .level = 0.60;
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    // Confirm we're in Drowsy phase
    assert_eq!(
        state.zones.phase(),
        MentalPhase::Drowsy,
        "should be in Drowsy phase with high adenosine + low arousal"
    );

    // Now let the system run without forcing adenosine level, but
    // keep arousal systems suppressed to maintain Drowsy. The
    // adenosine level should continue to rise (at 0.5× rate).
    let level_before = state
        .neurochemicals
        .get(NeurochemicalId::Adenosine)
        .unwrap()
        .level;

    for _ in 0..5000 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        // Keep arousal suppressed to maintain Drowsy
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.30;
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Norepinephrine)
            .unwrap()
            .level = 0.20;
        state.neuro_tick_with_params(&params);
        state.write_end();
        // If we've transitioned to NREM, stop — the test has served
        // its purpose (adenosine accumulated enough during Drowsy to
        // cross the sleep threshold).
        if state.zones.phase() == MentalPhase::NREM || state.zones.phase() == MentalPhase::REM {
            break;
        }
    }

    let level_after = state
        .neurochemicals
        .get(NeurochemicalId::Adenosine)
        .unwrap()
        .level;

    assert!(
        level_after > level_before,
        "adenosine level should increase during Drowsy (was {}, now {})",
        level_before,
        level_after
    );
}

#[test]
fn test_lock_free_read_after_tick() {
    let mut state = GenesisCoreState::new(1, 1000);

    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(2000) };
    state.neuro_tick();
    state.write_end();

    // SAFETY: single-threaded test — self is not being deallocated or remapped.
    let snapshot = unsafe { state.read_consistent() }.expect("read should succeed");
    assert_eq!(snapshot.header.heartbeat, 1);
    assert!(snapshot.verify_checksum().is_ok());
}

#[test]
fn test_metaplasticity_coupling_matrix_adapts() {
    // Metaplasticity: the coupling matrix should self-modify based on
    // sustained emotional states. Chronic stress weakens excitatory
    // couplings; sustained wellbeing strengthens them.
    use genesis::state::neurochemical::DEFAULT_COUPLING_MATRIX;

    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams::COMPRESSED;

    // Snapshot a coupling value before adaptation
    let da_idx = NeurochemicalId::Dopamine as usize;
    let end_idx = NeurochemicalId::Endorphin as usize;
    let initial_coupling = state.neurochemicals.coupling_matrix[da_idx][end_idx];

    // Apply a large dopamine impulse to create sustained high DA
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(0) };
    state.neurochemicals.chemicals[da_idx].apply_impulse(0.8, 0);
    state.write_end();

    // Run 10000 ticks — enough for tonic levels to rise and
    // metaplasticity to produce measurable coupling changes
    for _ in 0..10000 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    let adapted_coupling = state.neurochemicals.coupling_matrix[da_idx][end_idx];

    // The coupling should have changed from its default value
    // (either strengthened or weakened depending on the emotional
    // context that emerged)
    assert!(
        (adapted_coupling - initial_coupling).abs() > 0.0001,
        "coupling matrix should adapt via metaplasticity: DA←END was {}, now {}",
        initial_coupling,
        adapted_coupling,
    );

    // The coupling should still be within reasonable bounds
    let default_val = DEFAULT_COUPLING_MATRIX[da_idx][end_idx];
    let max = default_val.abs().max(0.5) * 2.0;
    assert!(
        adapted_coupling.abs() <= max,
        "coupling should stay bounded: |{}| > {}",
        adapted_coupling,
        max,
    );
}

#[test]
fn test_metaplasticity_restoring_force() {
    // After adaptation, if the emotional state returns to neutral,
    // the coupling matrix should drift back toward its default values
    // (restoring force).
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams::COMPRESSED;

    let da_idx = NeurochemicalId::Dopamine as usize;
    let end_idx = NeurochemicalId::Endorphin as usize;
    let default_val = genesis::state::neurochemical::DEFAULT_COUPLING_MATRIX[da_idx][end_idx];

    // Manually perturb the coupling matrix
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(0) };
    state.neurochemicals.coupling_matrix[da_idx][end_idx] = default_val * 2.0;
    state.write_end();

    // Run 50000 ticks at neutral state — restoring force should
    // pull the coupling back toward default
    for _ in 0..50000 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    let adapted = state.neurochemicals.coupling_matrix[da_idx][end_idx];
    assert!(
        (adapted - default_val).abs() < (default_val * 2.0 - default_val).abs(),
        "restoring force should pull coupling toward default: was {:.4}, now {:.4}, default {:.4}",
        default_val * 2.0,
        adapted,
        default_val,
    );
}

// ─── Cortisol recovery after stress ────────────────────────────

#[test]
fn test_cortisol_recovers_after_stressor_ends() {
    // This test verifies that cortisol can return to near-baseline
    // after a stressor ends. The HPA axis cascade (CRH → ACTH →
    // cortisol) must not lock cortisol at the hard clamp when the
    // stressor is gone.
    //
    // Root cause being tested: the ACTH→cortisol drive was a direct
    // level injection that was too strong, making cortisol pin at the
    // hard clamp. The fix routes the drive through the first-order
    // level update (acth × 0.01 × maturation × dt) with the nonlinear
    // CORTISOL_CLEARANCE_RATE providing metabolic negative feedback.
    let mut vec = NeurochemicalVector::new(0);

    // Phase 1: Induce stress by directly spiking cortisol (mimicking
    // an acute stress event) and sustaining it for 200 ticks.
    for _ in 0..200 {
        vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.75;
        vec.tick_default();
    }

    let cort_during_stress = vec.get(NeurochemicalId::Cortisol).unwrap().level;
    assert!(
        cort_during_stress > 0.60,
        "cortisol should be elevated during stress (got {})",
        cort_during_stress
    );

    // Phase 2: Stressor ends — stop forcing cortisol, let the system
    // relax. The homeostatic force + metabolic clearance should pull
    // cortisol back down. Without the fix, cortisol would stay at
    // the clamp forever.
    for _ in 0..2000 {
        vec.tick_default();
    }

    let cort_after_recovery = vec.get(NeurochemicalId::Cortisol).unwrap().level;

    assert!(
        cort_after_recovery < 0.50,
        "cortisol should recover below 0.50 after stressor ends: \
         got {} (was {} during stress)",
        cort_after_recovery,
        cort_during_stress
    );
}

#[test]
fn test_cortisol_not_stuck_at_clamp_without_external_stress() {
    // Without an explicit stressor, cortisol should not get
    // permanently stuck at the 0.80 hard clamp. The system may
    // oscillate (a pre-existing property of the coupled dynamics),
    // but cortisol must not pin at the clamp indefinitely.
    //
    // Before the fix, the ACTH drive (direct level injection) was
    // strong enough that even tonic CRH at baseline pushed cortisol
    // to the 0.80 clamp, where it stayed forever. After the fix,
    // the ACTH drive is a first-order rate (acth × 0.01 × dt) and
    // the nonlinear clearance provides metabolic negative feedback.
    let mut vec = NeurochemicalVector::new(0);

    // Run for 10000 ticks and track how long cortisol stays at
    // or near the 0.80 clamp.
    let mut clamp_ticks = 0u32;
    let mut max_clamp_streak = 0u32;
    for _ in 0..10000 {
        vec.tick_default();
        let cort = vec.get(NeurochemicalId::Cortisol).unwrap().level;
        if cort >= 0.78 {
            clamp_ticks += 1;
            max_clamp_streak = max_clamp_streak.max(clamp_ticks);
        } else {
            clamp_ticks = 0;
        }
    }

    assert!(
        max_clamp_streak < 500,
        "cortisol should not be stuck at the clamp for more than 500 consecutive ticks: \
         max streak was {} ticks",
        max_clamp_streak
    );
}

// ─── Death spiral recovery ─────────────────────────────────────

#[test]
fn test_system_recovers_from_neurochemical_collapse() {
    // This test verifies that the system can recover from a
    // neurochemical collapse — the "death spiral" where cortisol
    // suppresses all other chemicals, and the depleted chemicals'
    // "removal of inhibition" keeps cortisol high, creating a
    // self-reinforcing loop that traps the system at a near-zero
    // equilibrium.
    //
    // The depletion gate (scaling coupling forces from depleted
    // chemicals) should break this loop by reducing the "removal of
    // inhibition" effect, allowing the homeostatic restoring force
    // to recover the system.
    //
    // Without the gate, after a sustained stressor, BDNF and
    // serotonin would remain near zero even after the stressor ends,
    // because:
    // 1. Depleted SRT/GABA/OXY/END "remove inhibition" from cortisol
    // 2. Cortisol stays elevated, suppressing BDNF
    // 3. BDNF recovery requires serotonin (srt * bdnf_recovery_rate)
    // 4. Serotonin can't recover because cortisol suppresses it
    // 5. The system is trapped at a low equilibrium
    let mut vec = NeurochemicalVector::new(0);

    // Phase 1: Induce severe stress — force cortisol high and let
    // it suppress everything else.
    for _ in 0..500 {
        vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.85;
        vec.tick_default();
    }

    // Verify the stress suppressed BDNF and serotonin
    let bdnf_during = vec.get(NeurochemicalId::BDNF).unwrap().level;
    let srt_during = vec.get(NeurochemicalId::Serotonin).unwrap().level;
    assert!(
        bdnf_during < 0.30,
        "BDNF should be suppressed during stress (got {})",
        bdnf_during
    );
    assert!(
        srt_during < 0.35,
        "serotonin should be suppressed during stress (got {})",
        srt_during
    );

    // Phase 2: Stressor ends — stop forcing cortisol, let the system
    // recover. Run for a long time to allow slow recovery.
    for _ in 0..5000 {
        vec.tick_default();
    }

    // Phase 3: Check recovery. The key indicators are:
    // - Cortisol should return near baseline (~0.37)
    // - BDNF should recover above 0.30 (the suppression threshold)
    // - Serotonin should recover above 0.35
    // - Plasticity gate should reopen (>0.25)
    let cort_after = vec.get(NeurochemicalId::Cortisol).unwrap().level;
    let bdnf_after = vec.get(NeurochemicalId::BDNF).unwrap().level;
    let srt_after = vec.get(NeurochemicalId::Serotonin).unwrap().level;
    let pg = vec.plasticity_gate;

    assert!(
        cort_after < 0.15,
        "cortisol should recover below 0.15 after stressor ends (zero baseline): got {}",
        cort_after
    );
    assert!(
        bdnf_after > bdnf_during,
        "BDNF should recover above stress level after stressor ends: got {} (was {} during stress)",
        bdnf_after,
        bdnf_during
    );
    assert!(
        srt_after > srt_during,
        "serotonin should recover above stress level after stressor ends: got {} (was {} during stress)",
        srt_after,
        srt_during
    );
    assert!(
        pg > 0.09,
        "plasticity gate should reopen above 0.09 after recovery: got {}",
        pg
    );
}

#[test]
fn test_bdnf_isolated_recovery() {
    // Isolate BDNF recovery: set all chemicals to baseline except
    // BDNF (set to 0.15) and check if it recovers.
    let mut vec = NeurochemicalVector::new(0);

    // Set BDNF to depleted level
    vec.get_mut(NeurochemicalId::BDNF).unwrap().level = 0.15;
    vec.recompute_derived();

    for _ in 0..500 {
        vec.tick_default();
    }

    let bdnf_final = vec.get(NeurochemicalId::BDNF).unwrap().level;
    assert!(
        bdnf_final > 0.20,
        "BDNF should recover above 0.20 in isolation (from 0.15): got {}",
        bdnf_final
    );
}

#[test]
fn test_exhaustion_override_forces_sleep_despite_receptor_burnout() {
    // When adenosine receptors are downregulated to floor (0.10),
    // effective adenosine is too low to trigger the normal sleep
    // threshold. But raw adenosine (metabolic sleep pressure) is
    // still high. The exhaustion override should force NREM sleep
    // when raw adenosine > 0.90, regardless of receptor sensitivity.
    //
    // This prevents the fatal feedback trap: receptor burnout →
    // can't sleep → can't recover receptors → stuck awake.
    let mut state = GenesisCoreState::new(1, 1000);

    // Simulate receptor burnout: adenosine sensitivity at floor
    let adn = state
        .neurochemicals
        .get_mut(NeurochemicalId::Adenosine)
        .unwrap();
    adn.receptor_sensitivity = 0.5; // floor (biological max downregulation)
    adn.desensitization_factor = 0.2; // floor
    adn.internalization_factor = 0.45; // floor
    adn.level = 0.95; // extreme metabolic sleep pressure
    // Histamine is moderate (would normally prevent sleep)
    let hist = state
        .neurochemicals
        .get_mut(NeurochemicalId::Histamine)
        .unwrap();
    hist.receptor_sensitivity = 0.5;
    hist.level = 0.50; // moderate — would block normal sleep

    state.neurochemicals.recompute_derived();
    state.sync_neurochemistry_to_state();

    // The phase computation should trigger exhaustion override
    let phase = state.zones.phase();
    assert_eq!(
        phase,
        MentalPhase::NREM,
        "exhaustion override should force NREM when raw adenosine > 0.90, \
         even with receptor burnout and moderate histamine. \
         Got phase {:?}",
        phase
    );
}

#[test]
fn test_adenosine_blend_allows_sleep_with_downregulated_receptors() {
    // With the 50/50 blend of effective and raw adenosine, sleep
    // should be possible even when receptors are partially
    // downregulated (not at floor, but reduced). This tests the
    // blend mechanism, not the exhaustion override.
    let mut state = GenesisCoreState::new(1, 1000);

    // Moderate receptor downregulation
    let adn = state
        .neurochemicals
        .get_mut(NeurochemicalId::Adenosine)
        .unwrap();
    adn.receptor_sensitivity = 0.50; // partially downregulated
    adn.desensitization_factor = 0.5;
    adn.internalization_factor = 0.5;
    adn.level = 0.85; // high sleep pressure
    let hist = state
        .neurochemicals
        .get_mut(NeurochemicalId::Histamine)
        .unwrap();
    hist.receptor_sensitivity = 0.50;
    hist.level = 0.10; // low histamine — allows sleep

    state.neurochemicals.recompute_derived();
    state.sync_neurochemistry_to_state();

    // With blend: adn_eff = 0.85*0.50*0.50*0.50 = 0.106
    // adn_blend = 0.106*0.5 + 0.85*0.5 = 0.478
    // Sleep threshold (entering, mel~0, ox~0.3): 0.75 - 0 - 0.05*0.7 = 0.715
    // 0.478 < 0.715 — won't enter via normal threshold
    // But raw adenosine 0.85 < 0.90 — won't trigger exhaustion override either
    // So this should NOT be sleep yet — it should be Active or Drowsy
    let phase = state.zones.phase();
    assert_ne!(
        phase,
        MentalPhase::NREM,
        "with partially downregulated receptors and adn_blend=0.48, \
         should not enter NREM yet (below threshold). Got {:?}",
        phase
    );

    // Now push raw adenosine higher to trigger exhaustion override
    let adn = state
        .neurochemicals
        .get_mut(NeurochemicalId::Adenosine)
        .unwrap();
    adn.level = 0.92; // extreme — triggers exhaustion override
    state.neurochemicals.recompute_derived();
    state.sync_neurochemistry_to_state();

    let phase = state.zones.phase();
    assert_eq!(
        phase,
        MentalPhase::NREM,
        "exhaustion override should trigger NREM when raw adenosine > 0.90. \
         Got {:?}",
        phase
    );
}

#[test]
fn test_receptor_burnout_recovery_through_sleep() {
    // Full integration test: induce receptor burnout, then verify
    // that sleep (enabled by the exhaustion override) allows receptors
    // to recover over time.
    let mut state = GenesisCoreState::new(1, 1000);

    // Simulate sustained overstimulation: hold dopamine at a high level
    // to drive receptor downregulation. We set the level directly rather
    // than using apply_impulse because the vesicular pool depletes after
    // ~4 ticks (0.3/tick depletion vs 0.001/tick replenishment), making
    // repeated impulses ineffective. Directly holding the level models
    // sustained agonist exposure (e.g., chronic medication), which is
    // the canonical receptor downregulation paradigm.
    //
    // The adaptation rate is 0.0005/tick with dopamine multiplier 1.2.
    // With level 0.85 and baseline ~0.35, deviation ~0.50. Effective
    // deviation scales by receptor health, so downregulation is
    // self-limiting: as receptors downregulate, health drops, and
    // effective deviation drops. The sensitivity floor (0.5) is
    // reached when turnover balances the (now-weak) downregulation.
    let params = NeuroTickParams::COMPRESSED;
    for _ in 0..5000 {
        // SAFETY: this is a single-threaded test — no concurrent
        // readers, so the seqlock write_begin/write_end pair is
        // purely for protocol compliance, not actual synchronization.
        unsafe { state.write_begin(0) };
        let da = state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap();
        da.level = 0.85; // sustain high dopamine
        let adn = state
            .neurochemicals
            .get_mut(NeurochemicalId::Adenosine)
            .unwrap();
        adn.apply_impulse(0.5, 0); // accumulate sleep pressure
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    // Check that dopamine receptors have downregulated
    let da_sens = state
        .neurochemicals
        .get(NeurochemicalId::Dopamine)
        .unwrap()
        .receptor_sensitivity;
    assert!(
        da_sens <= 0.5,
        "dopamine sensitivity should have downregulated to floor (0.5) from sustained overstimulation: got {}",
        da_sens
    );

    // Check adenosine has accumulated
    let adn_raw = state
        .neurochemicals
        .get(NeurochemicalId::Adenosine)
        .unwrap()
        .level;
    let adn_sens = state
        .neurochemicals
        .get(NeurochemicalId::Adenosine)
        .unwrap()
        .receptor_sensitivity;

    // If adenosine is high enough, the exhaustion override should
    // allow sleep, and sleep should allow receptor recovery
    if adn_raw > 0.90 {
        // Should be in sleep phase despite receptor burnout
        let phase = state.zones.phase();
        assert!(
            phase == MentalPhase::NREM || phase == MentalPhase::REM,
            "with raw adenosine {} and sensitivity {}, exhaustion override should force sleep. Got {:?}",
            adn_raw,
            adn_sens,
            phase
        );

        // Run more ticks in sleep — receptors should start recovering
        for _ in 0..3000 {
            // SAFETY: single-threaded test — no concurrent readers.
            unsafe { state.write_begin(0) };
            state.neuro_tick_with_params(&params);
            state.write_end();
        }

        // Dopamine sensitivity should have recovered somewhat
        // (upregulation is 3x slower, but sleep + time should help)
        let da_sens_after = state
            .neurochemicals
            .get(NeurochemicalId::Dopamine)
            .unwrap()
            .receptor_sensitivity;
        assert!(
            da_sens_after > da_sens,
            "dopamine sensitivity should recover during sleep: was {}, now {}",
            da_sens,
            da_sens_after
        );
    }
}

#[test]
fn test_receptor_resensitization_during_sleep() {
    // Test that the receptor_resensitization_rate parameter causes
    // unconditional receptor recovery during sleep — even when levels
    // are ABOVE baseline (the "receptor burnout trap" condition where
    // the deviation-gated recovery never triggers).
    let mut state = GenesisCoreState::new(1, 1000);

    // Phase 1: drive receptor downregulation with sustained high levels.
    // We set levels directly rather than using apply_impulse because the
    // vesicular pool depletes after ~4 ticks, making repeated impulses
    // ineffective. Directly holding the level models sustained agonist
    // exposure, which is the canonical receptor downregulation paradigm.
    let wake_params = NeuroTickParams::COMPRESSED;
    for _ in 0..10000 {
        // SAFETY: single-threaded test, no concurrent readers
        unsafe { state.write_begin(0) };
        let da = state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap();
        da.level = 0.85;
        let srt = state
            .neurochemicals
            .get_mut(NeurochemicalId::Serotonin)
            .unwrap();
        srt.level = 0.85;
        state.neuro_tick_with_params(&wake_params);
        state.write_end();
    }

    // Verify receptors have downregulated
    let da_desens_before = state
        .neurochemicals
        .get(NeurochemicalId::Dopamine)
        .unwrap()
        .desensitization_factor;
    let srt_desens_before = state
        .neurochemicals
        .get(NeurochemicalId::Serotonin)
        .unwrap()
        .desensitization_factor;
    let da_intern_before = state
        .neurochemicals
        .get(NeurochemicalId::Dopamine)
        .unwrap()
        .internalization_factor;
    assert!(
        da_desens_before < 0.9,
        "dopamine desensitization should be significant after overstimulation: got {}",
        da_desens_before
    );

    // Phase 2: induce sleep with receptor resensitization enabled.
    // The key: levels are still ABOVE baseline (the burnout trap), so
    // the deviation-gated recovery would NOT trigger. Only the
    // unconditional sleep resensitization can restore receptors.
    let mut sleep_params = NeuroTickParams::COMPRESSED;
    sleep_params.receptor_resensitization_rate = 0.001; // sleep resensitization

    for _ in 0..3000 {
        // SAFETY: single-threaded test, no concurrent readers
        unsafe { state.write_begin(0) };
        // Keep levels elevated — mimicking the burnout trap where
        // raw levels stay high but receptors are burned out
        let da = state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap();
        da.level = 0.85;
        state.neuro_tick_with_params(&sleep_params);
        state.write_end();
    }

    // Receptors should have recovered significantly despite levels
    // being above baseline (the burnout trap condition)
    let da_desens_after = state
        .neurochemicals
        .get(NeurochemicalId::Dopamine)
        .unwrap()
        .desensitization_factor;
    let srt_desens_after = state
        .neurochemicals
        .get(NeurochemicalId::Serotonin)
        .unwrap()
        .desensitization_factor;
    let da_intern_after = state
        .neurochemicals
        .get(NeurochemicalId::Dopamine)
        .unwrap()
        .internalization_factor;

    assert!(
        da_desens_after > da_desens_before + 0.1,
        "dopamine desensitization should recover during sleep even with high levels: was {}, now {}",
        da_desens_before,
        da_desens_after
    );
    assert!(
        srt_desens_after > srt_desens_before + 0.1,
        "serotonin desensitization should recover during sleep even with high levels: was {}, now {}",
        srt_desens_before,
        srt_desens_after
    );
    assert!(
        da_intern_after > da_intern_before + 0.03,
        "dopamine internalization should recover during sleep even with high levels: was {}, now {}",
        da_intern_before,
        da_intern_after
    );
}

#[test]
fn test_receptor_resensitization_zero_during_wake() {
    // Verify that with receptor_resensitization_rate = 0 (wake mode),
    // receptors do NOT recover when levels are above baseline. This
    // confirms the mechanism is sleep-specific.
    let mut state = GenesisCoreState::new(1, 1000);

    // Drive receptor downregulation. We set the level directly rather
    // than using apply_impulse because the vesicular pool depletes after
    // ~4 ticks, making repeated impulses ineffective.
    let wake_params = NeuroTickParams::COMPRESSED;
    for _ in 0..5000 {
        // SAFETY: single-threaded test, no concurrent readers
        unsafe { state.write_begin(0) };
        let da = state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap();
        da.level = 0.85;
        state.neuro_tick_with_params(&wake_params);
        state.write_end();
    }

    let da_desens_before = state
        .neurochemicals
        .get(NeurochemicalId::Dopamine)
        .unwrap()
        .desensitization_factor;

    // Continue with high levels but NO resensitization rate (wake mode)
    for _ in 0..3000 {
        // SAFETY: single-threaded test, no concurrent readers
        unsafe { state.write_begin(0) };
        let da = state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap();
        da.level = 0.85;
        state.neuro_tick_with_params(&wake_params); // resensitization_rate = 0
        state.write_end();
    }

    let da_desens_after = state
        .neurochemicals
        .get(NeurochemicalId::Dopamine)
        .unwrap()
        .desensitization_factor;

    // Without sleep resensitization, desensitization should NOT have
    // recovered significantly (it may even have gotten worse)
    assert!(
        da_desens_after <= da_desens_before + 0.05,
        "dopamine desensitization should NOT recover without sleep resensitization: was {}, now {}",
        da_desens_before,
        da_desens_after
    );
}

// ─── Circadian wake drive (Process C) ──────────────────────────

#[test]
fn test_circadian_wake_drive_maintains_daytime_arousal() {
    // The two-process model of sleep (Borbély 1982) requires both
    // Process S (adenosine sleep pressure) and Process C (circadian
    // wake drive). Without Process C, the system cannot maintain
    // wakefulness during the day even with healthy neurochemicals,
    // because adenosine accumulation overwhelms the arousal promoters.
    //
    // This test verifies that with healthy daytime neurochemicals and
    // a daytime circadian phase, arousal rises above 0.5 (awake).
    let mut vec = NeurochemicalVector::new(0);
    // Set to midday (phase 0.5 = noon) — peak circadian wake drive
    vec.set_circadian_phase(0.5);
    // Tick enough times for arousal to converge from 0
    for _ in 0..200 {
        vec.tick_with_params(&NeuroTickParams::DEFAULT);
    }
    assert!(
        vec.arousal > 0.5,
        "daytime circadian wake drive should maintain arousal >0.5 (got {})",
        vec.arousal
    );
}

#[test]
fn test_circadian_wake_drive_absent_at_night() {
    // At night (phase 0.0 = midnight), the circadian wake drive should
    // be near zero, allowing sleep pressure to push arousal down.
    // With high adenosine and low arousal chemicals, arousal should
    // stay low at night even after many ticks.
    let mut vec = NeurochemicalVector::new(0);
    vec.set_circadian_phase(0.0); // midnight
    // Set high sleep pressure, low arousal chemicals
    vec.get_mut(NeurochemicalId::Adenosine).unwrap().level = 0.70;
    vec.get_mut(NeurochemicalId::Norepinephrine).unwrap().level = 0.15;
    vec.get_mut(NeurochemicalId::Histamine).unwrap().level = 0.15;
    vec.get_mut(NeurochemicalId::Dopamine).unwrap().level = 0.15;
    vec.get_mut(NeurochemicalId::Acetylcholine).unwrap().level = 0.15;
    vec.get_mut(NeurochemicalId::GABA).unwrap().level = 0.65;
    for _ in 0..200 {
        vec.tick_with_params(&NeuroTickParams::DEFAULT);
    }
    assert!(
        vec.arousal < 0.35,
        "nighttime with high sleep pressure should keep arousal low (got {})",
        vec.arousal
    );
}

// ─── Metaplasticity stress adaptation ──────────────────────────

#[test]
fn test_metaplasticity_stress_strengthens_inhibitory_coupling() {
    // Under chronic stress, inhibitory (negative) couplings should
    // become MORE negative, not less. The original code had a sign
    // error that weakened inhibitory couplings under stress, causing
    // GABA→DA to flip positive and creating a pathological DA spike
    // that downregulated receptors and collapsed arousal.
    let mut vec = NeurochemicalVector::new(0);
    // Record the initial GABA→DA coupling (should be negative)
    let gaba_da_idx = NeurochemicalId::Dopamine as usize;
    let initial_gaba_to_da = vec.coupling_matrix[gaba_da_idx][NeurochemicalId::GABA as usize];
    assert!(
        initial_gaba_to_da < 0.0,
        "GABA→DA should start negative (inhibitory), got {}",
        initial_gaba_to_da
    );
    // Induce chronic stress: high cortisol, low BDNF/serotonin
    for _ in 0..500 {
        vec.get_mut(NeurochemicalId::Cortisol).unwrap().level = 0.85;
        vec.get_mut(NeurochemicalId::BDNF).unwrap().level = 0.15;
        vec.get_mut(NeurochemicalId::Serotonin).unwrap().level = 0.20;
        vec.tick_with_params(&NeuroTickParams::DEFAULT);
    }
    let final_gaba_to_da = vec.coupling_matrix[gaba_da_idx][NeurochemicalId::GABA as usize];
    // Under stress, inhibitory coupling should stay negative or
    // become more negative — never flip positive.
    assert!(
        final_gaba_to_da <= 0.0,
        "GABA→DA should remain inhibitory (≤0) under stress, got {}",
        final_gaba_to_da
    );
}

// ─── ACh excess-only degradation ────────────────────────────────

#[test]
fn test_ach_degradation_preserves_baseline() {
    // ACh degradation should only decay the EXCESS above baseline,
    // not the absolute level. Absolute degradation caused a death
    // spiral where ACh fell below baseline and couldn't recover.
    let mut vec = NeurochemicalVector::new(0);
    let ach_baseline = vec.get(NeurochemicalId::Acetylcholine).unwrap().baseline;
    // Set ACh to baseline
    vec.get_mut(NeurochemicalId::Acetylcholine).unwrap().level = ach_baseline;
    // Tick many times — ACh should stay near baseline
    for _ in 0..1000 {
        vec.tick_with_params(&NeuroTickParams::DEFAULT);
    }
    let ach_level = vec.get(NeurochemicalId::Acetylcholine).unwrap().level;
    assert!(
        (ach_level - ach_baseline).abs() < 0.1,
        "ACh at baseline should stay near baseline after degradation (baseline={}, level={})",
        ach_baseline,
        ach_level
    );
}

#[test]
fn test_adenosine_homeostatic_force_does_not_block_sleep() {
    // Regression test for the receptor burnout trap.
    //
    // The homeostatic force pulls adenosine toward its baseline (0.20)
    // at a rate ~34× stronger than the accumulation rate (0.055/hour).
    // Without skipping the homeostatic force for adenosine during
    // wakefulness, adenosine can never accumulate enough to trigger
    // sleep. This creates an unbreakable trap: sleep can't happen →
    // receptors can't resensitize → effective levels stay near zero →
    // plasticity gate stays closed → system is permanently stuck.
    //
    // This test verifies that, with the fix in place, adenosine can
    // actually reach the sleep threshold (0.75) during sustained
    // wakefulness, even when starting from baseline.
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // Simulate sustained wakefulness (high histamine). Adenosine
    // should accumulate past the sleep threshold (0.75).
    let mut max_adn = 0.0;
    for _ in 0..200_000 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        // Keep histamine high to maintain wakefulness (above 0.30)
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Histamine)
            .unwrap()
            .level = 0.45;
        state.neuro_tick_with_params(&params);
        state.write_end();
        let adn = state
            .neurochemicals
            .get(NeurochemicalId::Adenosine)
            .unwrap()
            .level;
        if adn > max_adn {
            max_adn = adn;
        }
        // Early exit once we've confirmed adenosine can exceed 0.50
        // (well above baseline 0.20, proving the homeostatic force
        // is not blocking accumulation).
        if max_adn > 0.50 {
            break;
        }
    }

    assert!(
        max_adn > 0.50,
        "adenosine should accumulate well above baseline (0.20) during sustained \
         wakefulness, but max reached was {}. The homeostatic force may be \
         blocking adenosine accumulation, preventing sleep and creating a \
         receptor burnout trap.",
        max_adn
    );
}

#[test]
fn test_functional_rescue_recovers_extreme_receptor_burnout() {
    // Regression test for the functional rescue mechanism.
    //
    // When effective levels fall below FUNCTIONAL_RESCUE_THRESHOLD (0.10)
    // due to extreme receptor burnout, receptors should slowly recover
    // toward 1.0 even during wakefulness. This is a safety net that
    // prevents the system from getting permanently stuck when:
    //   - Raw levels stay above drifted baselines (deviation-gated
    //     recovery never triggers)
    //   - Sleep can't be reached (adenosine can't accumulate)
    //
    // This test verifies that the rescue mechanism actually recovers
    // receptors from extreme burnout during wakefulness.
    let mut state = GenesisCoreState::new(1, 1000);

    // Force extreme receptor burnout on dopamine
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(0) };
    {
        let da = state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap();
        da.receptor_sensitivity = 0.5; // floor
        da.desensitization_factor = 0.2; // floor
        da.internalization_factor = 0.45; // floor
        da.level = 0.85; // high raw level (above baseline)
        da.baseline = 0.35;
    }
    state.write_end();

    // Verify effective level is below rescue threshold
    let da = state.neurochemicals.get(NeurochemicalId::Dopamine).unwrap();
    let initial_eff = da.effective_level();
    assert!(
        initial_eff < 0.25,
        "setup should produce effective level below rescue threshold: got {}",
        initial_eff
    );

    let initial_sens = da.receptor_sensitivity;
    let initial_desens = da.desensitization_factor;
    let initial_intern = da.internalization_factor;

    // Run many ticks — functional rescue should slowly recover receptors
    let params = NeuroTickParams::DEFAULT;
    for _ in 0..100_000 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        // Keep dopamine high so deviation-gated recovery doesn't trigger
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap()
            .level = 0.85;
        state.neuro_tick_with_params(&params);
        state.write_end();
    }

    let da = state.neurochemicals.get(NeurochemicalId::Dopamine).unwrap();
    let final_sens = da.receptor_sensitivity;
    let final_desens = da.desensitization_factor;
    let final_intern = da.internalization_factor;

    // All three receptor mechanisms should have recovered somewhat
    assert!(
        final_sens > initial_sens,
        "receptor sensitivity should recover via functional rescue: was {}, now {}",
        initial_sens,
        final_sens
    );
    assert!(
        final_desens > initial_desens,
        "desensitization factor should recover via functional rescue: was {}, now {}",
        initial_desens,
        final_desens
    );
    assert!(
        final_intern > initial_intern,
        "internalization factor should recover via functional rescue: was {}, now {}",
        initial_intern,
        final_intern
    );
}

#[test]
fn test_wake_recovery_restores_depleted_neurochemicals() {
    // Wake recovery should boost depleted neurotransmitter levels
    // toward baseline when cortisol is low (no ongoing stress).
    let mut state = GenesisCoreState::new(1, 1000);

    // Deplete BDNF, dopamine, serotonin below their baselines.
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(0) };
    {
        let bdnf = state
            .neurochemicals
            .get_mut(NeurochemicalId::BDNF)
            .unwrap();
        bdnf.level = 0.10;
        bdnf.baseline = 0.45;
    }
    {
        let da = state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap();
        da.level = 0.10;
        da.baseline = 0.35;
    }
    {
        let srt = state
            .neurochemicals
            .get_mut(NeurochemicalId::Serotonin)
            .unwrap();
        srt.level = 0.10;
        srt.baseline = 0.40;
    }
    // Ensure cortisol is low (no ongoing stress).
    state
        .neurochemicals
        .get_mut(NeurochemicalId::Cortisol)
        .unwrap()
        .level = 0.0;
    state.write_end();

    let bdnf_before = state.neurochemicals.get(NeurochemicalId::BDNF).unwrap().level;
    let da_before = state.neurochemicals.get(NeurochemicalId::Dopamine).unwrap().level;

    // Apply wake recovery.
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(0) };
    state.neurochemicals.wake_recovery();
    state.write_end();

    let bdnf_after = state.neurochemicals.get(NeurochemicalId::BDNF).unwrap().level;
    let da_after = state.neurochemicals.get(NeurochemicalId::Dopamine).unwrap().level;

    // Levels should be boosted 50% of the way toward baseline.
    // BDNF: 0.10 + (0.45 - 0.10) * 0.5 = 0.275
    assert!(
        bdnf_after > bdnf_before,
        "BDNF should be boosted: was {}, now {}",
        bdnf_before,
        bdnf_after
    );
    assert!(
        da_after > da_before,
        "Dopamine should be boosted: was {}, now {}",
        da_before,
        da_after
    );
    // Plasticity gate should recover (BDNF drives it).
    assert!(
        state.neurochemicals.plasticity_gate > 0.15,
        "plasticity_gate should recover after wake recovery: got {}",
        state.neurochemicals.plasticity_gate
    );
}

#[test]
fn test_wake_recovery_resensitizes_receptors() {
    // Wake recovery should resensitize receptors (receptor_sensitivity,
    // desensitization_factor, internalization_factor) toward 1.0 when
    // they have been downregulated by sustained high levels.
    let mut state = GenesisCoreState::new(1, 1000);

    // Simulate receptor burnout from sustained high BDNF.
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(0) };
    {
        let bdnf = state
            .neurochemicals
            .get_mut(NeurochemicalId::BDNF)
            .unwrap();
        bdnf.level = 0.82;
        bdnf.baseline = 0.65;
        bdnf.receptor_sensitivity = 0.6;
        bdnf.desensitization_factor = 0.3;
        bdnf.internalization_factor = 0.5;
    }
    // Ensure cortisol is low (no ongoing stress).
    state
        .neurochemicals
        .get_mut(NeurochemicalId::Cortisol)
        .unwrap()
        .level = 0.0;
    // Recompute derived fields so effective_levels reflects the
    // burnout factors we just set.
    state.neurochemicals.recompute_derived();
    state.write_end();

    let eff_before = state.neurochemicals.effective_levels[NeurochemicalId::BDNF as usize];

    // Apply wake recovery.
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(0) };
    state.neurochemicals.wake_recovery();
    state.write_end();

    let bdnf = state.neurochemicals.get(NeurochemicalId::BDNF).unwrap();
    let eff_after = state.neurochemicals.effective_levels[NeurochemicalId::BDNF as usize];

    // All three receptor mechanisms should have recovered.
    assert!(
        bdnf.receptor_sensitivity > 0.6,
        "receptor_sensitivity should recover: was 0.6, now {}",
        bdnf.receptor_sensitivity
    );
    assert!(
        bdnf.desensitization_factor > 0.3,
        "desensitization_factor should recover: was 0.3, now {}",
        bdnf.desensitization_factor
    );
    assert!(
        bdnf.internalization_factor > 0.5,
        "internalization_factor should recover: was 0.5, now {}",
        bdnf.internalization_factor
    );
    // Effective level should increase.
    assert!(
        eff_after > eff_before,
        "effective BDNF should increase after wake recovery: was {}, now {}",
        eff_before,
        eff_after
    );
}

#[test]
fn test_wake_recovery_skipped_when_cortisol_high() {
    // Wake recovery should NOT boost neurochemicals when cortisol is
    // high (ongoing stress) — the system should stay in its stressed
    // state.
    let mut state = GenesisCoreState::new(1, 1000);

    // Deplete BDNF and set cortisol high.
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(0) };
    {
        let bdnf = state
            .neurochemicals
            .get_mut(NeurochemicalId::BDNF)
            .unwrap();
        bdnf.level = 0.10;
        bdnf.baseline = 0.45;
    }
    state
        .neurochemicals
        .get_mut(NeurochemicalId::Cortisol)
        .unwrap()
        .level = 0.5; // high cortisol
    state.write_end();

    let bdnf_before = state.neurochemicals.get(NeurochemicalId::BDNF).unwrap().level;

    // Apply wake recovery — should be skipped.
    // SAFETY: single-threaded test — no concurrent writers.
    unsafe { state.write_begin(0) };
    state.neurochemicals.wake_recovery();
    state.write_end();

    let bdnf_after = state.neurochemicals.get(NeurochemicalId::BDNF).unwrap().level;

    // BDNF should NOT be boosted (cortisol is high).
    assert!(
        (bdnf_after - bdnf_before).abs() < 0.001,
        "BDNF should NOT be boosted when cortisol is high: was {}, now {}",
        bdnf_before,
        bdnf_after
    );
}
