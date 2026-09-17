//! Tests for the memory-mapped state file layer.
//!
//! These verify:
//! - Creating a new state file and reading it back
//! - Opening an existing state file
//! - open_or_create semantics
//! - Modifying state through the mmap layer
//! - Syncing to disk
//! - Crash recovery (checksum detection)
//! - Persistence across MmapState instances
//! - Lock-free reads during writes

use genesis::state::*;
use genesis::store::{MmapState, StateFileError};
use std::path::PathBuf;

/// Unique temp file path for each test run.
fn temp_path(test_name: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("genesis_test_{test_name}_{pid}_{nanos}.bin"))
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
}

/// Current wall-clock ms — used as the state-creation timestamp in
/// persistence tests. `open()` re-anchors the circadian phase to the
/// wall clock when the stored phase has drifted ≥60s; a state born
/// "now" is re-opened within seconds (well under the threshold), so
/// persistence assertions on exact write counts (heartbeat) hold.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

// ─── Create ───────────────────────────────────────────────────

#[test]
fn test_create_state_file() {
    let path = temp_path("create");
    let mmap = MmapState::create(&path, 42, 1000).expect("create should succeed");
    assert!(mmap.was_created());
    assert!(path.exists());

    // Verify the state is correctly initialised
    let state = mmap.read();
    assert!(state.verify().is_ok());
    assert!(state.verify_checksum().is_ok());
    assert_eq!(state.header.instance_id, 42);
    assert_eq!(state.zones.zone(), CognitiveZone::Idle);

    mmap.sync().expect("sync should succeed");
    drop(mmap);
    cleanup(&path);
}

// ─── Open existing ────────────────────────────────────────────

#[test]
fn test_open_rejects_concurrent_writer_lock() {
    let path = temp_path("open_locked");
    {
        let mmap = MmapState::create(&path, 7, now_ms()).expect("create");
        mmap.sync().expect("sync");
    }

    let file = std::fs::File::open(&path).expect("open lock file");
    let fd = std::os::unix::io::AsRawFd::as_raw_fd(&file);
    // SAFETY: fd is valid for the duration of this scope.
    assert_eq!(unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) }, 0);

    let result = MmapState::open(&path);
    assert!(matches!(result, Err(StateFileError::FileLockBusy)));

    // SAFETY: fd is valid and currently holds the flock.
    unsafe { libc::flock(fd, libc::LOCK_UN) };
    drop(file);
    cleanup(&path);
}

#[test]
fn test_open_existing_state_file() {
    let path = temp_path("open");

    // Create and write some state
    {
        let mmap = MmapState::create(&path, 99, 5000).expect("create");
        mmap.modify(5100, |state| {
            state.zones.transition_to(CognitiveZone::Coding, 5100);
        })
        .expect("modify");
        mmap.sync().expect("sync");
    }

    // Re-open and verify persistence
    {
        let mmap = MmapState::open(&path).expect("open should succeed");
        assert!(!mmap.was_created());
        let state = mmap.read();
        assert_eq!(state.header.instance_id, 99);
        assert_eq!(state.zones.zone(), CognitiveZone::Coding);
        assert!(state.verify_checksum().is_ok());
    }

    cleanup(&path);
}

// ─── open_or_create ───────────────────────────────────────────

#[test]
fn test_open_or_create_creates_when_missing() {
    let path = temp_path("ooc_create");
    assert!(!path.exists());

    let mmap = MmapState::open_or_create(&path, 1, 100).expect("should create");
    assert!(mmap.was_created());
    assert!(path.exists());

    drop(mmap);
    cleanup(&path);
}

// ─── Circadian anchoring ──────────────────────────────────────

/// Circular distance between two circadian phases, in [0, 0.5].
fn circadian_distance(a: f32, b: f32) -> f32 {
    let d = (b - a).rem_euclid(1.0);
    d.min(1.0 - d)
}

#[test]
fn test_circadian_phase_anchored_to_wall_clock() {
    // Regression test: the circadian oscillator must be phase-locked
    // to the actual day/night cycle, not to state-file creation time.
    // The constructor used to hardcode phase 0.3 (~07:12) regardless
    // of the wall clock, so melatonin's night window (documented as
    // phase 0.85→0.15 = ~20:24→03:36) peaked at whatever time of day
    // the state file happened to be born, permanently out of sync
    // with the real day.
    //
    // These assertions hold in any timezone: 12h is exactly half a
    // circadian cycle under any fixed UTC offset, and 24h is a full
    // cycle. 2023-11-14/15 contains no DST transition in any zone.
    let base_ms: u64 = 1_699_920_000_000; // 2023-11-14T00:00:00Z

    let phase_at = |ms: u64, path: &PathBuf| -> f32 {
        let mmap = MmapState::create(path, 1, ms).expect("create should succeed");
        let phase = mmap.read().neurochemicals.circadian_phase();
        drop(mmap);
        cleanup(path);
        phase
    };

    // Same timestamp → same phase (deterministic anchoring).
    let phase_a = phase_at(base_ms, &temp_path("circadian_a"));
    let phase_c = phase_at(base_ms, &temp_path("circadian_c"));
    assert_eq!(
        phase_a, phase_c,
        "the same timestamp must yield the same circadian phase"
    );

    // +12h → half a cycle apart. The old hardcoded constant gave a
    // circular distance of 0.0 here, whatever the wall clock said.
    let phase_b = phase_at(base_ms + 12 * 3_600_000, &temp_path("circadian_b"));
    let half = circadian_distance(phase_a, phase_b);
    assert!(
        (half - 0.5).abs() < 0.01,
        "12h apart must be half a circadian cycle apart (got {phase_a} vs {phase_b})"
    );

    // +24h → the same phase (the oscillator's period is one day).
    let phase_d = phase_at(base_ms + 24 * 3_600_000, &temp_path("circadian_d"));
    let full = circadian_distance(phase_a, phase_d);
    assert!(
        full < 0.01,
        "24h apart must be the same circadian phase (got {phase_a} vs {phase_d})"
    );
}

#[test]
fn test_circadian_phase_reanchored_on_open() {
    // Regression test: open() must re-anchor a stale circadian phase
    // to the wall clock. The mmap'd phase only advances while the
    // daemon runs — downtime (power off, Genesis stopped) leaves it
    // stale by the offline duration. A state born 12h in the past
    // must, once opened, carry the same phase as a state born now
    // (both anchored to the current local time of day).
    let now_ms: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let path_stale = temp_path("circadian_reopen_stale");
    let path_fresh = temp_path("circadian_reopen_fresh");

    // Stale: born 12h ago (phase anchored to that time of day).
    let mmap_stale =
        MmapState::create(&path_stale, 1, now_ms - 12 * 3_600_000).expect("create stale");
    let stale_phase = mmap_stale.read().neurochemicals.circadian_phase();
    drop(mmap_stale);

    // Fresh: born now.
    let mmap_fresh = MmapState::create(&path_fresh, 1, now_ms).expect("create fresh");
    let fresh_phase = mmap_fresh.read().neurochemicals.circadian_phase();
    drop(mmap_fresh);

    // Sanity: 12h of birth-time difference is half a cycle apart.
    // (Holds in any timezone; a DST transition inside the window
    // shifts it by at most ~1h/24 ≈ 0.042 of a cycle.)
    let apart = circadian_distance(stale_phase, fresh_phase);
    assert!(
        (apart - 0.5).abs() < 0.06,
        "phases born 12h apart should be ~half a cycle apart (got {stale_phase} vs {fresh_phase})"
    );

    // Open the stale file — the re-anchor must pull its phase to the
    // current wall clock, i.e. to the fresh state's phase.
    let mmap_reopened = MmapState::open(&path_stale).expect("open stale");
    let reopened_phase = mmap_reopened.read().neurochemicals.circadian_phase();
    drop(mmap_reopened);

    let drift = circadian_distance(reopened_phase, fresh_phase);
    assert!(
        drift < 0.01,
        "opened state must be re-anchored to the wall clock \
         (got {reopened_phase}, fresh {fresh_phase})"
    );

    cleanup(&path_stale);
    cleanup(&path_fresh);
}

#[test]
fn test_open_or_create_opens_when_exists() {
    let path = temp_path("ooc_open");

    // First call creates
    {
        let mmap = MmapState::open_or_create(&path, 7, 200).expect("should create");
        assert!(mmap.was_created());
        drop(mmap);
    }

    // Second call opens
    {
        let mmap = MmapState::open_or_create(&path, 999, 300).expect("should open");
        assert!(!mmap.was_created());
        // The instance_id should be from the original creation, not 999
        assert_eq!(mmap.read().header.instance_id, 7);
    }

    cleanup(&path);
}

// ─── Modify ───────────────────────────────────────────────────

#[test]
fn test_modify_state() {
    let path = temp_path("modify");
    let mmap = MmapState::create(&path, 1, 1000).expect("create");

    mmap.modify(2000, |state| {
        state.zones.transition_to(CognitiveZone::Conversation, 2000);
        state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap()
            .apply_impulse(0.3, 2000);
        state.neurochemicals.recompute_derived();
    })
    .expect("modify");

    // Verify the modification took effect
    let state = mmap.read();
    assert_eq!(state.zones.zone(), CognitiveZone::Conversation);
    assert!(
        state.neurochemicals.valence > 0.0,
        "dopamine boost → positive valence"
    );
    assert!(
        state.verify_checksum().is_ok(),
        "checksum should be valid after modify"
    );

    drop(mmap);
    cleanup(&path);
}

#[test]
fn test_multiple_modifies() {
    let path = temp_path("multi_modify");
    let mmap = MmapState::create(&path, 1, 0).expect("create");

    for i in 1..=10 {
        mmap.modify(i * 100, |state| {
            let _ = state.header.heartbeat; // read
            state.neurochemicals.tick_default();
        })
        .expect("modify");
    }

    let state = mmap.read();
    assert_eq!(state.header.heartbeat, 10);
    assert!(state.verify_checksum().is_ok());

    drop(mmap);
    cleanup(&path);
}

// ─── Sync ─────────────────────────────────────────────────────

#[test]
fn test_sync_persists_to_disk() {
    let path = temp_path("sync");

    // Create, modify, sync, drop
    {
        let mmap = MmapState::create(&path, 1, now_ms()).expect("create");
        mmap.modify(2000, |state| {
            state.zones.transition_to(CognitiveZone::Learning, 2000);
        })
        .expect("modify");
        mmap.sync().expect("sync");
    }

    // Re-open and verify the synced state persisted
    {
        let mmap = MmapState::open(&path).expect("open");
        let state = mmap.read();
        assert_eq!(state.zones.zone(), CognitiveZone::Learning);
        assert_eq!(state.header.heartbeat, 1);
    }

    cleanup(&path);
}

// ─── Lock-free read during write ──────────────────────────────

#[test]
fn test_lock_free_read_during_write() {
    let path = temp_path("lockfree");
    let mmap = MmapState::create(&path, 1, 0).expect("create");

    // Read before any write — should succeed.
    // `read_consistent` is safe — it copies through raw pointers,
    // no `&GenesisCoreState` to the mmap'd memory is created.
    let snapshot = mmap.read_consistent();
    assert!(snapshot.is_some(), "read before write should succeed");

    // Modify and read again
    mmap.modify(100, |state| {
        state.neurochemicals.tick_default();
    })
    .expect("modify");

    // `read_consistent` is safe — no unsafe needed.
    let snapshot = mmap.read_consistent();
    assert!(snapshot.is_some(), "read after write should succeed");
    let snapshot = snapshot.unwrap();
    assert_eq!(snapshot.header.heartbeat, 1);

    drop(mmap);
    cleanup(&path);
}

// ─── Neurochemical tick through mmap ──────────────────────────

#[test]
fn test_neuro_tick_through_mmap() {
    let path = temp_path("neuro_tick");
    let mmap = MmapState::create(&path, 1, 0).expect("create");

    // Run several neuro ticks through the mmap layer
    for i in 1..=20 {
        mmap.modify(i * 100, |state| {
            state.neuro_tick();
        })
        .expect("modify");
    }

    let state = mmap.read();
    assert_eq!(state.header.heartbeat, 20);
    assert!(state.verify_checksum().is_ok());

    // The emergent phase should still be Active with default neurochemistry
    assert_eq!(state.zones.phase(), MentalPhase::Active);

    drop(mmap);
    cleanup(&path);
}

// ─── Stress scenario through mmap ──────────────────────────────

#[test]
fn test_stress_scenario_through_mmap() {
    let path = temp_path("stress_mmap");
    let mmap = MmapState::create(&path, 1, 0).expect("create");

    // Induce chronic stress — force cortisol, NE, and suppress
    // BDNF/serotonin/ACh. NE is forced because the reduced arousal
    // coupling (CORT→NE halved) no longer drives NE high enough
    // from cortisol alone to trigger the stress phase.
    for i in 1..=200 {
        mmap.modify(i * 100, |state| {
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Cortisol)
                .unwrap()
                .level = 0.85;
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Norepinephrine)
                .unwrap()
                .level = 0.70;
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
        })
        .expect("modify");
    }

    let state = mmap.read();
    assert!(
        !state.memory.can_form_new_memories(),
        "chronic stress through mmap should prevent new memory formation"
    );
    assert!(
        state.zones.phase().is_distress(),
        "chronic stress should produce distress phase ({:?})",
        state.zones.phase()
    );
    assert!(state.verify_checksum().is_ok());

    drop(mmap);
    cleanup(&path);
}

// ─── File not found ───────────────────────────────────────────

#[test]
fn test_open_nonexistent_returns_error() {
    let path = temp_path("nonexistent");
    assert!(!path.exists());

    let result = MmapState::open(&path);
    assert!(matches!(result, Err(StateFileError::NotFound)));
}

// ─── Corrupted file detection ─────────────────────────────────

#[test]
fn test_corrupted_file_detected() {
    let path = temp_path("corrupted");

    // Create a valid state file
    {
        let mmap = MmapState::create(&path, 1, 0).expect("create");
        mmap.sync().expect("sync");
        drop(mmap);
    }

    // Corrupt the file by overwriting the magic bytes
    {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open for write");
        use std::os::unix::fs::FileExt;
        file.write_all_at(&[0xFF, 0xFF, 0xFF, 0xFF], 0)
            .expect("corrupt magic");
    }

    // Opening should fail with verification error
    let result = MmapState::open(&path);
    assert!(
        matches!(result, Err(StateFileError::VerificationFailed(_))),
        "corrupted file should fail verification"
    );

    cleanup(&path);
}

// ─── Persistence across instances ─────────────────────────────

#[test]
fn test_persistence_across_instances() {
    let path = temp_path("persistence");

    // Instance 1: create and set state
    {
        let mmap = MmapState::create(&path, 12345, now_ms()).expect("create");
        mmap.modify(2000, |state| {
            state.zones.transition_to(CognitiveZone::Analysis, 2000);
            state
                .neurochemicals
                .get_mut(NeurochemicalId::Dopamine)
                .unwrap()
                .apply_impulse(0.4, 2000);
            state.neurochemicals.recompute_derived();
        })
        .expect("modify");
        mmap.sync().expect("sync");
    }

    // Instance 2: open and verify everything persisted
    {
        let mmap = MmapState::open(&path).expect("open");
        let state = mmap.read();
        assert_eq!(state.header.instance_id, 12345);
        assert_eq!(state.zones.zone(), CognitiveZone::Analysis);
        assert!(state.neurochemicals.valence > 0.0);
        assert_eq!(state.header.heartbeat, 1);
        assert!(state.verify_checksum().is_ok());
    }

    cleanup(&path);
}

// ─── File size ────────────────────────────────────────────────

#[test]
fn test_file_size_is_one_page() {
    let path = temp_path("filesize");
    let mmap = MmapState::create(&path, 1, 0).expect("create");
    drop(mmap);

    let metadata = std::fs::metadata(&path).expect("metadata");
    assert_eq!(
        metadata.len(),
        4096,
        "file should be exactly one page (4096 bytes)"
    );

    cleanup(&path);
}

// ─── Cross-process file lock ──────────────────────────────────

#[test]
fn test_cross_process_lock_blocks_second_writer() {
    // Two MmapState instances on the same file simulate two processes.
    // The first modify() acquires the flock; the second modify() on a
    // different instance (different fd, different open file description)
    // should fail with FileLockBusy because the lock is held by the
    // first. This verifies that the flock serializes writers across
    // open file descriptions, not just within one.
    //
    // Note: flock locks are associated with the open file description,
    // not the fd or the process. Two opens of the same file produce two
    // independent open file descriptions, so the second open's flock
    // will conflict with the first's.
    use genesis::store::StateFileError;

    let path = temp_path("xproc_lock");

    let mmap1 = MmapState::create(&path, 1, 1000).expect("create 1");
    let mmap2 = MmapState::open(&path).expect("open 2");

    // Acquire the lock via mmap1's modify. The closure holds the lock
    // for the duration of the call, so we need to test contention
    // from within the closure.
    let result = mmap1.modify(1100, |state| {
        // While mmap1 holds the flock, mmap2 should not be able to
        // acquire it.
        let result2 = mmap2.modify(1200, |_state2| {
            // If we get here, the lock didn't work — two writers
            // are inside the critical section simultaneously.
            panic!("second writer should have been blocked by flock");
        });
        assert!(
            matches!(result2, Err(StateFileError::FileLockBusy)),
            "second modify should return FileLockBusy, got {:?}",
            result2
        );
        // Verify mmap1's state is still accessible inside the closure.
        state.zones.transition_to(CognitiveZone::Learning, 1100);
    });
    assert!(result.is_ok(), "first modify should succeed: {:?}", result);

    // After mmap1's modify releases the lock, mmap2 should succeed.
    let result3 = mmap2.modify(1300, |state| {
        assert_eq!(state.zones.zone(), CognitiveZone::Learning);
    });
    assert!(
        result3.is_ok(),
        "second modify after lock release should succeed: {:?}",
        result3
    );

    cleanup(&path);
}

#[test]
fn test_cross_process_lock_released_after_modify() {
    // Verify that the flock is released after modify() returns, so
    // subsequent modify() calls on the same instance succeed.
    let path = temp_path("xproc_release");

    let mmap = MmapState::create(&path, 1, 1000).expect("create");

    // First modify should acquire and release the lock.
    mmap.modify(1100, |state| {
        state.zones.transition_to(CognitiveZone::Coding, 1100);
    })
    .expect("first modify");

    // Second modify should succeed — the lock was released.
    mmap.modify(1200, |state| {
        assert_eq!(state.zones.zone(), CognitiveZone::Coding);
    })
    .expect("second modify after lock release");

    // sync should also succeed.
    mmap.sync().expect("sync after modify");

    cleanup(&path);
}
