//! Tests for gentle forgetting — power-law trace decay, the
//! accessibility gate, rehearsal as forgetting's opposite, and the
//! prune + compact path that physically reclaims storage.

use genesis::store::forgetting::*;
use genesis::store::*;
use std::path::{Path, PathBuf};

fn temp_base(test_name: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("genesis_forget_{test_name}_{pid}_{nanos}"))
}

fn cleanup(base: &Path) {
    let _ = std::fs::remove_file(base.with_extension("bundles"));
    let _ = std::fs::remove_file(base.with_extension("meta"));
    let _ = std::fs::remove_file(base.with_extension("dat"));
    let _ = std::fs::remove_file(base.with_extension("traces"));
    let _ = std::fs::remove_file(base.with_extension("compacting"));
}

fn make_full_tag() -> [f32; 12] {
    [
        0.50, 0.55, 0.40, 0.45, 0.50, 0.60, 0.20, 0.35, 0.30, 0.45, 0.10, 0.40,
    ]
}

fn make_compact_tag() -> [f32; 4] {
    [0.40, 0.35, 0.20, 0.50]
}

fn store_episode(store: &mut LtmStore, ts: u64, salience: f32, text: &str) -> u64 {
    store
        .store(
            ts,
            salience,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            text,
        )
        .expect("store")
}

const DAY_MS_U64: u64 = 86_400_000;

// ─── Trace mass ───────────────────────────────────────────────

#[test]
fn test_trace_mass_single_trace_exact() {
    // One presentation: mass is exactly age^(−d).
    let m = trace_mass(1, 100.0, 100.0);
    assert!((m - 100.0f64.powf(-0.5)).abs() < 1e-12);
}

#[test]
fn test_trace_mass_decays_and_rehearsal_restores() {
    // Same age, more presentations → more mass.
    let one = trace_mass(1, 30.0, 30.0);
    let five = trace_mass(5, 30.0, 30.0);
    assert!(five > one);
    // Mass falls with age — power law.
    let young = trace_mass(1, 10.0, 10.0);
    let old = trace_mass(1, 40.0, 40.0);
    assert!(young > old);
    // 4× older → half the mass at d = 0.5.
    assert!((young / old - 2.0).abs() < 0.02);
}

#[test]
fn test_trace_mass_rehearsal_after_silence_dominates() {
    // A memory rehearsed once after 200 days of silence: the fresh
    // trace (age ~0 days) should dominate the mass — the endpoint
    // bound keeps the blend from smearing it away.
    let mass = trace_mass(2, 200.0, 1.0);
    // True sum = 200^(−1/2) + 1^(−1/2) = 0.0707 + 1.0 — the blend
    // alone would give 4/(√200+1) ≈ 0.26, badly under-counting.
    assert!(mass > 0.9, "fresh rehearsal must dominate, mass={mass}");
}

// ─── Calibration — the forgetting strength ─────────────────────

#[test]
fn test_forgetting_horizon_is_gentle() {
    // A single-hearing, mid-salience memory should survive ~months
    // of disuse before it is unreachable — not days, not years.
    // days_until_forgotten(n=1, s=0.5, age=0) = (1/e^(−2.5))^(1/0.5)
    // = e^5 ≈ 148 days.
    let horizon = days_until_forgotten(1, 0.5, 0.0);
    assert!(horizon > 100.0 && horizon < 200.0, "horizon={horizon}d");

    // Salience moves the horizon by e^(2·Δs): 0.9 vs 0.5 → e^0.8 ≈ 2.2×.
    let strong = days_until_forgotten(1, 0.9, 0.0);
    let ratio = strong / horizon;
    assert!(ratio > 2.0 && ratio < 2.5, "salience ratio={ratio}");

    // Ten rehearsals stretch the horizon ~100× at n ≥ 2 with a₀≈a_L.
    let rehearsed = days_until_forgotten(10, 0.5, 0.0);
    assert!(rehearsed / horizon > 50.0, "rehearsed={rehearsed}d");
}

#[test]
fn test_noise_is_deterministic_and_bounded() {
    for id in 0..64u64 {
        let a = activation_noise(id, 12345);
        let b = activation_noise(id, 12345);
        assert_eq!(a, b);
        assert!((-1.0..=1.0).contains(&a));
        let c = activation_noise(id, 54321);
        assert_eq!(activation_noise(id, 54321), c);
    }
}

#[test]
fn test_dopamine_shifts_the_boundary() {
    // Above-baseline dopamine lowers the effective threshold —
    // the forgetting-cell pathway easing transient retrieval.
    assert!(effective_threshold(0.9) < RETRIEVAL_TAU);
    assert!(effective_threshold(0.1) > RETRIEVAL_TAU);
    assert_eq!(effective_threshold(DA_BASELINE), RETRIEVAL_TAU);
    // The shift is bounded — chemistry modulates, never rules.
    assert!((effective_threshold(1.0) - (-0.2)).abs() < 1e-9);
}

#[test]
fn test_should_forget_requires_age_and_depth() {
    // Below the bound but too young → protected by the age floor.
    assert!(!should_forget(-3.0, 10.0));
    // Old enough and unreachable → a candidate.
    assert!(should_forget(-2.5, 60.0));
    // Faded but still reachable by a strong cue → not a candidate.
    assert!(!should_forget(-1.5, 60.0));
}

// ─── Integration: gating, rehearsal, prune, compact ────────────

#[test]
fn test_retrieval_gate_fades_with_time() {
    let base = temp_base("gate_fades");
    let mut store = LtmStore::create(&base, 64).expect("create");
    let t0 = 1_000u64;
    let text = "the orchard ladder leaned against the old barn";
    let id = store_episode(&mut store, t0, 0.5, text);

    // Fresh: the memory is accessible.
    let results = store.find_similar_hash_at(text, 5, t0 + DAY_MS_U64, DA_BASELINE);
    assert!(results.iter().any(|(rid, _, _)| *rid == id));

    // ~2 months untouched, mid salience: ordinary weak cues no
    // longer reach it, but an exact-match cue still can.
    let mid = t0 + 60 * DAY_MS_U64;
    let weak = store.find_similar_hash_at("completely unrelated topic", 5, mid, DA_BASELINE);
    assert!(!weak.iter().any(|(rid, _, _)| *rid == id));
    let strong = store.find_similar_hash_at(text, 5, mid, DA_BASELINE);
    assert!(
        strong.iter().any(|(rid, _, _)| *rid == id),
        "a perfect cue should still reach a faded-but-alive memory"
    );

    // A year untouched: below the forget bound — unreachable even
    // by a maximal cue, and a prune candidate.
    let far = t0 + 400 * DAY_MS_U64;
    let gone = store.find_similar_hash_at(text, 5, far, DA_BASELINE);
    assert!(!gone.iter().any(|(rid, _, _)| *rid == id));
    assert!(
        store
            .faded_candidates(far)
            .iter()
            .any(|(cid, _)| *cid == id)
    );

    drop(store);
    cleanup(&base);
}

#[test]
fn test_rehearsal_protects_from_fading() {
    let base = temp_base("rehearsal_protects");
    let mut store = LtmStore::create(&base, 64).expect("create");
    let t0 = 1_000u64;
    let text = "the lighthouse keeper logs the midnight tide";
    let id = store_episode(&mut store, t0, 0.2, text);

    // Rehearse it once — retrieval IS the strengthening event.
    // The trace lands at wall-clock now; a query measured far past
    // that still finds the rehearsal's mark.
    let now_wall = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    store.retrieve(id).expect("rehearse");
    let b_after = store
        .activation_of(id, now_wall + 60 * DAY_MS_U64)
        .expect("activation");
    let b_never = base_level(Trace::encoded(t0), t0, 0.2, now_wall + 60 * DAY_MS_U64);
    assert!(
        b_after > b_never,
        "one rehearsal must lift activation: {b_after} vs {b_never}"
    );

    drop(store);
    cleanup(&base);
}

#[test]
fn test_prune_and_compact_reclaims_disk() {
    let base = temp_base("prune_compact");
    let mut store = LtmStore::create(&base, 64).expect("create");
    // Episodes encoded ~400 days ago on the wall clock, so the
    // rehearsal touches below (stamped at real now) sit on the same
    // timeline as the prune evaluation.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let t0 = now - 400 * DAY_MS_U64;

    // Ten forgettable episodes + one protected (rehearsed to a high
    // access count via repeated retrieves).
    let mut forgettable = Vec::new();
    for i in 0..10 {
        forgettable.push(store_episode(
            &mut store,
            t0,
            0.1,
            &format!("forgettable episode number {i} about ordinary events"),
        ));
    }
    let keeper_text = "an important memory the mind keeps returning to";
    let keeper = store_episode(&mut store, t0, 0.9, keeper_text);
    for _ in 0..5 {
        store.retrieve(keeper).expect("rehearse keeper");
    }

    let before_meta = std::fs::metadata(base.with_extension("meta"))
        .unwrap()
        .len();
    let before_dat = std::fs::metadata(base.with_extension("dat")).unwrap().len();

    // Now: the forgettable cohort is released, the rehearsed memory
    // is not a candidate.
    let candidates: Vec<u64> = store
        .faded_candidates(now)
        .iter()
        .map(|(id, _)| *id)
        .collect();
    for id in &forgettable {
        assert!(candidates.contains(id));
    }
    assert!(!candidates.contains(&keeper));

    let pruned = store.prune_faded(now).expect("prune");
    assert_eq!(pruned, 10);
    let reclaimed = store.compact().expect("compact");
    assert!(reclaimed > 0);

    let after_meta = std::fs::metadata(base.with_extension("meta"))
        .unwrap()
        .len();
    let after_dat = std::fs::metadata(base.with_extension("dat")).unwrap().len();
    assert!(after_meta < before_meta);
    assert!(after_dat < before_dat);

    // The survivor still reads and retrieves; the forgotten are gone.
    let ep = store.retrieve(keeper).expect("keeper survives");
    assert_eq!(ep.text, keeper_text);
    assert!(store.retrieve(forgettable[0]).is_err());
    assert_eq!(store.count(), 1);

    // Reopen: on-disk state must be self-consistent (open() runs
    // the full integrity pass).
    drop(store);
    let mut store = LtmStore::open(&base).expect("reopen after compact");
    assert_eq!(store.count(), 1);
    assert_eq!(
        store.retrieve(keeper).expect("keeper after reopen").text,
        keeper_text
    );
    assert!(store.retrieve(forgettable[0]).is_err());

    // New stores continue with fresh, never-reused episode IDs.
    let next = store_episode(&mut store, t0, 0.5, "a new memory after compaction");
    assert!(next > keeper);
    assert!(store.retrieve(next).is_ok());

    drop(store);
    cleanup(&base);
}

#[test]
fn test_young_episodes_never_prune() {
    let base = temp_base("young_safe");
    let mut store = LtmStore::create(&base, 64).expect("create");
    let t0 = 1_000u64;
    // A very low-salience but recent memory — the age floor protects
    // it even though its activation is low.
    let id = store_episode(&mut store, t0, 0.05, "a faint but recent memory");
    let near = t0 + 10 * DAY_MS_U64; // 10 days < 30-day floor
    assert!(store.faded_candidates(near).is_empty());
    assert_eq!(store.prune_faded(near).expect("prune"), 0);
    assert!(store.retrieve(id).is_ok());
    drop(store);
    cleanup(&base);
}

#[test]
fn test_traces_persist_across_reopen() {
    let base = temp_base("traces_persist");
    let mut store = LtmStore::create(&base, 64).expect("create");
    let t0 = 1_000u64;
    let id = store_episode(&mut store, t0, 0.5, "a memory rehearsed twice");
    store.retrieve(id).expect("rehearse 1");
    store.retrieve(id).expect("rehearse 2");
    store.sync().expect("sync");

    // Reopen — the rehearsal count survives.
    drop(store);
    let store = LtmStore::open(&base).expect("reopen");
    // Activation with 3 touches > activation with the encoding alone.
    let with_traces = store.activation_of(id, t0 + 30 * DAY_MS_U64).expect("act");
    let without = base_level(Trace::encoded(t0), t0, 0.5, t0 + 30 * DAY_MS_U64);
    assert!(with_traces > without);
    drop(store);
    cleanup(&base);
}

#[test]
fn test_missing_traces_file_seeds_fresh_ink() {
    let base = temp_base("traces_missing");
    let mut store = LtmStore::create(&base, 64).expect("create");
    let t0 = 1_000u64;
    // An ancient memory — under the encoded-at-timestamp floor it
    // would already be past the forget horizon.
    let id = store_episode(&mut store, t0, 0.5, "memory with a lost trace file");
    drop(store);

    // Delete the sidecar. Losing the whole ledger must fail toward
    // retention: the store seeds one fresh-ink trace at open rather
    // than treating every memory as never-rehearsed-since-encoding.
    std::fs::remove_file(base.with_extension("traces")).expect("remove traces");
    let store = LtmStore::open(&base).expect("reopen without traces");

    let wall_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let act = store.activation_of(id, wall_now).expect("act");
    let lost_ledger_floor = base_level(Trace::encoded(t0), t0, 0.5, wall_now);
    // Fresh ink outranks the unknowable-history floor by miles.
    assert!(
        act > lost_ledger_floor + 2.0,
        "act={act} floor={lost_ledger_floor}"
    );
    // And the ancient memory must not be a prune candidate yet.
    assert!(store.faded_candidates(wall_now).is_empty());
    drop(store);
    cleanup(&base);
}

#[test]
fn test_seeded_traces_fade_again_honestly() {
    // The migration seed is a delay, not immunity: a memory that is
    // never touched after re-seeding eventually fades past the
    // horizon on its new timeline.
    let base = temp_base("seeded_fades");
    let mut store = LtmStore::create(&base, 64).expect("create");
    let t0 = 1_000u64;
    let id = store_episode(&mut store, t0, 0.3, "seeded then untouched");
    drop(store);
    std::fs::remove_file(base.with_extension("traces")).expect("remove traces");
    let store = LtmStore::open(&base).expect("reopen");

    let wall_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    // A year past the re-seed: n=1, age 365d → base ≈ −2.6 + 0.3
    // < −2, and encoding age clears the floor.
    let far = wall_now + 365 * DAY_MS_U64;
    let candidates = store.faded_candidates(far);
    assert!(
        candidates.iter().any(|(cid, _)| *cid == id),
        "untouched seeded memory should fade on its new timeline"
    );
    drop(store);
    cleanup(&base);
}

#[test]
fn test_body_sweep_respects_meta_archived_and_dreams() {
    // The body's autonomous sweep must never touch the classes the
    // mind's retention policy protects — even when their traces are
    // mathematically dead.
    let base = temp_base("protected_classes");
    let mut store = LtmStore::create(&base, 64).expect("create");
    let t0 = 1_000u64;
    let ancient = t0; // epoch — as faded as it gets

    let ordinary = store_episode(&mut store, ancient, 0.05, "a plain dead memory");
    let archived = store_episode(&mut store, ancient, 0.05, "her cold-storage choice");
    store.archive(archived).expect("archive");
    // store_meta no longer sets the flag (bookkeeping is stored
    // unflagged now), so write the flag the way a legacy store would
    // carry it: META_MEMORY_FLAG at byte 60 of the entry's 64-byte row.
    let meta = store
        .store_meta(
            ancient,
            0.05,
            make_full_tag(),
            make_compact_tag(),
            2,
            4,
            "[association] the residue that justifies forgetting",
        )
        .expect("store_meta");
    let dream = store
        .store(
            ancient,
            0.05,
            make_full_tag(),
            make_compact_tag(),
            2,
            9, // dreaming module, unflagged — protection rides the source
            "a dream-sourced episode",
        )
        .expect("store dream");
    store.sync().expect("sync");
    drop(store);

    // Surgical flag write: meta is the 3rd stored episode → row index 2.
    {
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut meta_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(base.with_extension("meta"))
            .expect("open meta");
        meta_file
            .seek(SeekFrom::Start(64 + 2 * 64 + 60))
            .expect("seek flags");
        let mut flags = [0u8; 1];
        meta_file.read_exact(&mut flags).expect("read flags");
        flags[0] |= 2; // META_MEMORY_FLAG
        meta_file
            .seek(SeekFrom::Start(64 + 2 * 64 + 60))
            .expect("seek back");
        meta_file.write_all(&flags).expect("write flags");
    }
    let store = LtmStore::open(&base).expect("reopen flagged meta");

    let wall_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let candidates: Vec<u64> = store
        .faded_candidates(wall_now)
        .into_iter()
        .map(|(id, _)| id)
        .collect();

    assert!(
        candidates.contains(&ordinary),
        "ordinary dead memory must fade"
    );
    assert!(
        !candidates.contains(&archived),
        "archived is her verdict's call"
    );
    assert!(!candidates.contains(&meta), "meta-memories are never swept");
    assert!(!candidates.contains(&dream), "dream-source is preserved");
    drop(store);
    cleanup(&base);
}

#[test]
fn test_interrupted_compaction_recovers_on_open() {
    let base = temp_base("recover_compact");
    let t0 = 1_000u64;
    let (keeper_id, keeper_text) = {
        let mut store = LtmStore::create(&base, 64).expect("create");
        let keeper = store_episode(&mut store, t0, 0.5, "the survivor");
        let text = store.retrieve(keeper).expect("read").text;
        store_episode(&mut store, t0 + 1, 0.5, "the pruned one");
        store.sync().expect("sync");

        // Simulate a crash after the journal was written: the staged
        // compacted files (survivors only) are durable, waiting to be
        // renamed in. A second store builds byte-exact staged files
        // under the stage dir — a different path, so no flock clash.
        let stage_dir = base.with_extension(format!("compact-{}-stage", std::process::id()));
        std::fs::create_dir(&stage_dir).expect("stage dir");
        {
            let mut staged = LtmStore::create(stage_dir.join("ltm"), 64).expect("staged");
            store_episode(&mut staged, t0, 0.5, "the survivor");
            staged.sync().expect("staged sync");
        }
        std::fs::write(
            base.with_extension("compacting"),
            stage_dir.to_string_lossy().as_bytes(),
        )
        .expect("journal");
        (keeper, text)
        // store drops here without compact() — the "crash".
    };

    // Reopen: the journal is replayed — staged files swap in, the
    // journal and stage dir are gone, and only the survivor remains.
    let mut store = LtmStore::open(&base).expect("recovering open");
    assert_eq!(store.count(), 1, "staged (compacted) meta must win");
    assert_eq!(
        store.retrieve(keeper_id).expect("survivor").text,
        keeper_text
    );
    assert!(
        !base.with_extension("compacting").exists(),
        "journal cleared"
    );
    drop(store);

    // An orphaned stage dir with no journal (crash pre-commit) is
    // swept on the next open without touching live data.
    let orphan = base.with_extension(format!("compact-{}-orphan", std::process::id()));
    std::fs::create_dir(&orphan).expect("orphan dir");
    let store = LtmStore::open(&base).expect("reopen with orphan");
    assert_eq!(store.count(), 1, "live files untouched by sweep");
    assert!(!orphan.exists(), "orphan stage dir swept");
    drop(store);
    cleanup(&base);
}
