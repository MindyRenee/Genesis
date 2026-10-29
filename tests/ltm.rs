//! Tests for the LTM episodic store (v2 SDR/LogHD).

use genesis::store::*;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};

fn temp_base(test_name: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("genesis_ltm_{test_name}_{pid}_{nanos}"))
}

fn cleanup(base: &Path) {
    let _ = std::fs::remove_file(base.with_extension("bundles"));
    let _ = std::fs::remove_file(base.with_extension("meta"));
    let _ = std::fs::remove_file(base.with_extension("dat"));
}

fn make_full_tag() -> [f32; 12] {
    [
        0.50, 0.55, 0.40, 0.45, 0.50, 0.60, 0.20, 0.35, 0.30, 0.45, 0.10, 0.40,
    ]
}

fn make_compact_tag() -> [f32; 4] {
    [0.40, 0.35, 0.20, 0.50] // arousal, valence, cortisol, dopamine
}

// ─── Create ───────────────────────────────────────────────────

#[test]
fn test_create_ltm() {
    let base = temp_base("create");
    let store = LtmStore::create(&base, 64).expect("create");

    // v2: capacity is effectively unlimited
    assert!(store.capacity() > 0);
    assert_eq!(store.count(), 0);

    drop(store);
    assert!(base.with_extension("bundles").exists());
    assert!(base.with_extension("meta").exists());
    assert!(base.with_extension("dat").exists());
    cleanup(&base);
}

// ─── Store and retrieve ───────────────────────────────────────

#[test]
fn test_store_and_retrieve() {
    let base = temp_base("store_retrieve");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let id = store
        .store(
            1000,
            0.75,
            make_full_tag(),
            make_compact_tag(),
            0, // UserInput
            4, // Language module
            "The user asked about the Great Salt Lake water levels.",
        )
        .expect("store");

    assert_eq!(store.count(), 1);
    assert!(id > 0);

    // Retrieve it
    let episode = store.retrieve(id).expect("retrieve");
    assert_eq!(episode.episode_id, id);
    assert_eq!(episode.timestamp, 1000);
    assert_eq!(episode.salience, 0.75);
    assert_eq!(episode.event_type, 0);
    assert_eq!(episode.source_module, 4);
    assert_eq!(
        episode.text,
        "The user asked about the Great Salt Lake water levels."
    );

    // Full emotional tag preserved
    assert_eq!(episode.full_emotional_tag, make_full_tag());

    // Compact tag preserved
    assert_eq!(episode.emotional_tag, make_compact_tag());

    drop(store);
    cleanup(&base);
}

// ─── Multiple episodes ────────────────────────────────────────

#[test]
fn test_multiple_episodes() {
    let base = temp_base("multiple");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let texts = [
        "First event: system started",
        "Second event: user interaction began",
        "Third event: data processed",
        "Fourth event: error occurred in module 3",
        "Fifth event: recovery completed",
    ];

    let mut ids = Vec::new();
    for (i, text) in texts.iter().enumerate() {
        let id = store
            .store(
                (i * 1000) as u64,
                0.3 + i as f32 * 0.1,
                make_full_tag(),
                make_compact_tag(),
                2, // Internal
                0, // Subcognitive
                text,
            )
            .expect("store");
        ids.push(id);
    }

    assert_eq!(store.count(), 5);

    // Retrieve each and verify
    for (i, &id) in ids.iter().enumerate() {
        let ep = store.retrieve(id).expect("retrieve");
        assert_eq!(ep.text, texts[i]);
        assert_eq!(ep.timestamp, (i * 1000) as u64);
    }

    drop(store);
    cleanup(&base);
}

// ─── Persistence ──────────────────────────────────────────────

#[test]
fn test_persistence() {
    let base = temp_base("persist");

    {
        let mut store = LtmStore::create(&base, 64).expect("create");
        store
            .store(
                5000,
                0.8,
                make_full_tag(),
                make_compact_tag(),
                1,
                2,
                "This should survive a restart",
            )
            .expect("store");
        store.sync().expect("sync");
    }

    // Re-open
    {
        let mut store = LtmStore::open(&base).expect("open");
        assert_eq!(store.count(), 1);

        // The first episode ID should be 1
        let ep = store.retrieve(1).expect("retrieve");
        assert_eq!(ep.text, "This should survive a restart");
        assert_eq!(ep.timestamp, 5000);
        assert_eq!(ep.salience, 0.8);
    }

    cleanup(&base);
}

// ─── Compression ──────────────────────────────────────────────

#[test]
fn test_compression() {
    let base = temp_base("compress");
    let mut store = LtmStore::create(&base, 64).expect("create");

    // Highly compressible text
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
    let id = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, &text)
        .expect("store");

    let ep = store.retrieve(id).expect("retrieve");
    assert_eq!(ep.text, text);

    // Check that the data file is smaller than the uncompressed text
    let data_size = std::fs::metadata(base.with_extension("dat"))
        .expect("data file")
        .len();
    assert!(
        data_size < text.len() as u64,
        "compressed data file ({data_size}) should be smaller than raw text ({})",
        text.len()
    );

    drop(store);
    cleanup(&base);
}

// ─── SimHash (retained for association engine) ────────────────

#[test]
fn test_simhash_identical() {
    let h1 = simhash("the great salt lake is drying up");
    let h2 = simhash("the great salt lake is drying up");
    assert_eq!(h1, h2, "identical text should produce identical hash");
}

#[test]
fn test_simhash_similar() {
    let h1 = simhash("the great salt lake is drying up");
    let h2 = simhash("the great salt lake is drying out");
    let dist = hamming_distance(h1, h2);
    assert!(
        dist <= 10,
        "similar texts should have small Hamming distance (got {dist})"
    );
}

#[test]
fn test_simhash_different() {
    let h1 = simhash("the great salt lake is drying up");
    let h2 = simhash("completely unrelated topic about cooking pasta");
    let dist = hamming_distance(h1, h2);
    assert!(
        dist > 15,
        "dissimilar texts should have large Hamming distance (got {dist})"
    );
}

// ─── Associative retrieval ────────────────────────────────────

#[test]
fn test_find_similar() {
    let base = temp_base("find_similar");
    let mut store = LtmStore::create(&base, 64).expect("create");

    // Store several episodes (source_module=4 = Language, a real
    // content module that is NOT auto-marked as meta-memory).
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "Great Salt Lake water levels are declining rapidly",
        )
        .expect("store");
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "Data centers in Utah consume massive amounts of water",
        )
        .expect("store");
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "The recipe for chocolate cake requires cocoa powder",
        )
        .expect("store");
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "Great Salt Lake water conservation efforts need funding",
        )
        .expect("store");

    // Query for something similar to the lake articles
    let results = store.find_similar("Great Salt Lake water crisis", 3);

    assert!(!results.is_empty());
    // The most similar should be one of the lake-related episodes
    let (best_id, _dist, _entry) = results[0];
    // Retrieve the best match to verify it is lake-related
    let best_ep = store.retrieve(best_id).expect("retrieve best match");
    assert!(
        best_ep.text.to_lowercase().contains("lake"),
        "best match should be lake-related: {}",
        best_ep.text
    );

    drop(store);
    cleanup(&base);
}

// ─── Salience ranking ─────────────────────────────────────────

#[test]
fn test_find_most_salient() {
    let base = temp_base("salient");
    let mut store = LtmStore::create(&base, 64).expect("create");

    // Use source_module=4 (Language) — a real content module that is
    // NOT auto-marked as meta-memory. source_module=0 (Subcognitive)
    // would be excluded from find_most_salient by is_excluded_from_search.
    store
        .store(
            0,
            0.2,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "low salience",
        )
        .expect("store");
    store
        .store(
            0,
            0.9,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "high salience",
        )
        .expect("store");
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "medium salience",
        )
        .expect("store");

    let results = store.find_most_salient(3);
    assert_eq!(results.len(), 3);
    // Should be sorted by salience descending
    let (_, top) = results[0];
    assert!(
        (top.salience - 0.9).abs() < 1e-6,
        "top result should have highest salience"
    );

    drop(store);
    cleanup(&base);
}

// ─── Time range ───────────────────────────────────────────────

#[test]
fn test_find_by_time_range() {
    let base = temp_base("time_range");
    let mut store = LtmStore::create(&base, 64).expect("create");

    // Use source_module=4 (Language) — a real content module that is
    // NOT auto-marked as meta-memory. source_module=0 (Subcognitive)
    // would be excluded from find_by_time_range by is_excluded_from_search.
    store
        .store(
            1000,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "t=1000",
        )
        .expect("store");
    store
        .store(
            2000,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "t=2000",
        )
        .expect("store");
    store
        .store(
            3000,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "t=3000",
        )
        .expect("store");
    store
        .store(
            4000,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "t=4000",
        )
        .expect("store");

    let results = store.find_by_time_range(1500, 3500);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].1.timestamp, 2000);
    assert_eq!(results[1].1.timestamp, 3000);

    drop(store);
    cleanup(&base);
}

// ─── Delete ───────────────────────────────────────────────────

#[test]
fn test_delete() {
    let base = temp_base("delete");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let id1 = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, "first")
        .expect("store");
    let id2 = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, "second")
        .expect("store");

    assert_eq!(store.count(), 2);

    // Delete the first
    store.delete(id1).expect("delete");
    assert_eq!(store.count(), 1);

    // Retrieving deleted episode should fail
    assert!(store.retrieve(id1).is_err());

    // Second episode should still be there
    let ep = store.retrieve(id2).expect("retrieve");
    assert_eq!(ep.text, "second");

    drop(store);
    cleanup(&base);
}

#[test]
fn test_archive() {
    let base = temp_base("archive");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let id1 = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, "first")
        .expect("store");
    let _id2 = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, "second")
        .expect("store");

    assert_eq!(store.count(), 2);

    // Archive the first
    store.archive(id1).expect("archive");
    assert_eq!(
        store.count(),
        1,
        "archived episode should not count as active"
    );

    // Archived episode should STILL be retrievable — this is the
    // key difference from deletion.
    let ep = store
        .retrieve(id1)
        .expect("archived episode should be retrievable");
    assert_eq!(ep.text, "first");

    // Archived episode should not appear in find_similar_hash
    let results = store.find_similar_hash("first", 10);
    assert!(
        !results.iter().any(|(id, _, _)| *id == id1),
        "archived episode should not appear in similarity search"
    );

    // Archiving again should be a no-op (not an error)
    store.archive(id1).expect("re-archive should be no-op");

    drop(store);
    cleanup(&base);
}

#[test]
fn test_archive_survives_reopen() {
    let base = temp_base("archive_reopen");
    let id1 = {
        let mut store = LtmStore::create(&base, 64).expect("create");
        let id = store
            .store(
                0,
                0.5,
                make_full_tag(),
                make_compact_tag(),
                0,
                0,
                "persisted memory",
            )
            .expect("store");
        store.archive(id).expect("archive");
        assert_eq!(store.count(), 0, "active count should be 0 after archive");
        id
    };

    // Reopen — archived episode should still be retrievable
    let mut store = LtmStore::open(&base).expect("open");
    assert_eq!(store.count(), 0, "active count should be 0 after reopen");
    let ep = store
        .retrieve(id1)
        .expect("archived episode should survive reopen");
    assert_eq!(ep.text, "persisted memory");

    cleanup(&base);
}

#[test]
fn test_find_similar_archived() {
    let base = temp_base("deep_recall");
    let mut store = LtmStore::create(&base, 64).expect("create");

    // source_module=4 (Language) — a real content module, not
    // auto-flagged as meta-memory.
    let archived_id = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "the orchard ladder leaned against the old barn",
        )
        .expect("store archived");
    let active_id = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "a completely unrelated note about semaphore flags",
        )
        .expect("store active");
    // An archived meta-memory: store_meta flags it regardless of
    // source module; archiving it yields flags = meta|archived.
    let meta_id = store
        .store_meta(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "association trace orchard ladder barn",
        )
        .expect("store meta");
    store.archive(meta_id).expect("archive meta");
    let deleted_id = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "the orchard ladder is also mentioned here",
        )
        .expect("store doomed");
    store.archive(archived_id).expect("archive");
    store.delete(deleted_id).expect("delete");

    // The active scan cannot see the archived episode.
    let active = store.find_similar_hash("orchard ladder barn", 10);
    assert!(
        !active.iter().any(|(id, _, _)| *id == archived_id),
        "archived episode must stay out of the active scan"
    );

    // Deep recall reaches it — a verbatim cue is an exact SimHash
    // match (distance 0) and ranks first.
    let deep = store.find_similar_archived("the orchard ladder leaned against the old barn", 10);
    assert_eq!(
        deep[0].0, archived_id,
        "archived episode should lead deep recall"
    );
    assert_eq!(deep[0].1, 0, "verbatim cue should be an exact match");
    // A fragmentary cue still surfaces it first — SimHash distance
    // reflects whole-text token overlap, so partial recall lands near
    // the 20-bit association boundary rather than at zero.
    let deep = store.find_similar_archived("orchard ladder leaned barn", 10);
    assert_eq!(
        deep[0].0, archived_id,
        "fragmentary cue should still lead deep recall"
    );
    // The other archived mention is reachable too — deep recall
    // returns the whole tier, ranked.
    // Active episodes are NOT part of the archived tier...
    assert!(
        !deep.iter().any(|(id, _, _)| *id == active_id),
        "active episodes do not belong to deep recall"
    );
    // ...nor archived meta-memories (they have their own channels)...
    assert!(
        !deep.iter().any(|(id, _, _)| *id == meta_id),
        "meta-memories stay out of deep recall"
    );
    // ...nor deleted entries.
    assert!(
        !deep.iter().any(|(id, _, _)| *id == deleted_id),
        "deleted entries are gone semantically"
    );

    drop(store);
    cleanup(&base);
}

#[test]
fn test_unarchive_reconsolidates() {
    let base = temp_base("unarchive");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let id = store
        .store(
            0,
            0.3,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "buried treasure under the old dock",
        )
        .expect("store");
    assert_eq!(store.count(), 1);

    store.archive(id).expect("archive");
    assert_eq!(store.count(), 0, "archive should drop the active count");
    assert!(store.find_similar_hash("buried treasure", 10).is_empty());

    store.unarchive(id).expect("unarchive");
    assert_eq!(
        store.count(),
        1,
        "unarchive should restore the active count"
    );

    // Back in the SimHash similarity index…
    let results = store.find_similar_hash("buried treasure under the dock", 10);
    assert!(
        results.iter().any(|(eid, _, _)| *eid == id),
        "revived episode must be searchable again"
    );
    // …and back in the bundle superposition — the association path
    // sees it too.
    let bundled = store.find_similar("buried treasure under the old dock", 10);
    assert!(
        bundled.iter().any(|(eid, _, _)| *eid == id),
        "revived episode must be re-bundled"
    );

    // Reconsolidation salience floor: a weak episode that proved
    // itself reachable is lifted above the compressor's retention
    // threshold.
    let ep = store.retrieve(id).expect("retrieve revived");
    assert!(
        (ep.salience - RECONSOLIDATION_SALIENCE_FLOOR).abs() < 1e-6,
        "salience should floor at reconsolidation level, got {}",
        ep.salience
    );

    // A strong memory is never dragged down to the floor.
    let strong_id = store
        .store(
            0,
            0.9,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "a vivid day at the county fair",
        )
        .expect("store strong");
    store.archive(strong_id).expect("archive strong");
    store.unarchive(strong_id).expect("unarchive strong");
    let strong = store.retrieve(strong_id).expect("retrieve strong");
    assert!(
        (strong.salience - 0.9).abs() < 1e-6,
        "floor must not lower high salience, got {}",
        strong.salience
    );

    // Unarchiving an already-active episode is a no-op; a missing
    // one is an error.
    store.unarchive(id).expect("re-unarchive should be a no-op");
    assert!(store.unarchive(999_999).is_err(), "missing id should error");

    drop(store);
    cleanup(&base);
}

#[test]
fn test_unarchive_survives_reopen() {
    let base = temp_base("unarchive_reopen");
    let id = {
        let mut store = LtmStore::create(&base, 64).expect("create");
        let id = store
            .store(
                0,
                0.4,
                make_full_tag(),
                make_compact_tag(),
                0,
                4,
                "a memory that refuses to stay buried",
            )
            .expect("store");
        store.archive(id).expect("archive");
        store.unarchive(id).expect("unarchive");
        id
    };

    let mut store = LtmStore::open(&base).expect("open");
    assert_eq!(
        store.count(),
        1,
        "revived episode stays active after reopen"
    );
    let ep = store.retrieve(id).expect("retrieve after reopen");
    assert_eq!(ep.text, "a memory that refuses to stay buried");
    let results = store.find_similar_hash("refuses to stay buried", 10);
    assert!(
        results.iter().any(|(eid, _, _)| *eid == id),
        "revived episode stays searchable after reopen"
    );

    drop(store);
    cleanup(&base);
}

// ─── v2: no capacity ceiling ──────────────────────────────────

#[test]
fn test_no_capacity_ceiling() {
    let base = temp_base("no_ceiling");
    let mut store = LtmStore::create(&base, 2).expect("create");

    // v2: even with capacity=2, we can store more than 2 episodes
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, "one")
        .expect("store");
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, "two")
        .expect("store");
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, "three")
        .expect("store");

    assert_eq!(store.count(), 3);

    drop(store);
    cleanup(&base);
}

// ─── Open or create ───────────────────────────────────────────

#[test]
fn test_open_or_create() {
    let base = temp_base("ooc");

    // First call creates
    {
        let mut store = LtmStore::open_or_create(&base, 32).expect("create");
        store
            .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, "test")
            .expect("store");
        store.sync().expect("sync");
    }

    // Second call opens
    {
        let mut store = LtmStore::open_or_create(&base, 32).expect("open");
        assert_eq!(store.count(), 1);
        let ep = store.retrieve(1).expect("retrieve");
        assert_eq!(ep.text, "test");
    }

    cleanup(&base);
}

// ─── Episode not found ────────────────────────────────────────

#[test]
fn test_episode_not_found() {
    let base = temp_base("notfound");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let result = store.retrieve(999);
    assert!(matches!(result, Err(LtmError::EpisodeNotFound)));

    drop(store);
    cleanup(&base);
}

// ─── Large text ───────────────────────────────────────────────

#[test]
fn test_large_text() {
    let base = temp_base("large_text");
    let mut store = LtmStore::create(&base, 64).expect("create");

    // 100 KB of text
    let text = "This is a test of the emergency broadcast system. ".repeat(2000);
    let id = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, &text)
        .expect("store");

    let ep = store.retrieve(id).expect("retrieve");
    assert_eq!(ep.text.len(), text.len());
    assert_eq!(ep.text, text);

    drop(store);
    cleanup(&base);
}

// ─── Unicode text ─────────────────────────────────────────────

#[test]
fn test_unicode_text() {
    let base = temp_base("unicode");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let text = "The Great Salt Lake — 大盐湖 — is drying up. 水位下降。";
    let id = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, text)
        .expect("store");

    let ep = store.retrieve(id).expect("retrieve");
    assert_eq!(ep.text, text);

    drop(store);
    cleanup(&base);
}

// ─── Index entry size ─────────────────────────────────────────

#[test]
fn test_index_entry_size() {
    assert_eq!(core::mem::size_of::<IndexEntry>(), 64);
    assert_eq!(core::mem::size_of::<IndexHeader>(), 64);
    assert_eq!(core::mem::size_of::<DataHeader>(), 16);
}

// ─── Corruption / DoS guards ──────────────────────────────────

#[test]
fn test_retrieve_rejects_oversized_compressed_length() {
    // Regression: corrupted or hand-crafted data files could claim
    // compressed payloads up to u32::MAX bytes. Verify retrieve
    // rejects the length before allocating.
    let base = temp_base("oversize_len");
    let mut store = LtmStore::create(&base, 64).expect("create");

    // Store a small valid episode at data offset 16 (after DataHeader).
    let id = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 0, "hello")
        .expect("store");
    drop(store);

    // Corrupt the 4-byte compressed-length field at offset 16 to u32::MAX.
    let dat_path = base.with_extension("dat");
    let mut dat = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&dat_path)
        .expect("open data");
    dat.seek(std::io::SeekFrom::Start(16)).expect("seek");
    dat.write_all(&u32::MAX.to_le_bytes()).expect("write");
    dat.flush().expect("flush");
    drop(dat);

    // Reopen and verify retrieve is rejected before it allocates.
    let mut store = LtmStore::open(&base).expect("open");
    let err = store
        .retrieve(id)
        .expect_err("should reject oversized length");
    assert!(
        err.to_string().contains("exceeds maximum"),
        "error should mention maximum: {err}"
    );

    drop(store);
    cleanup(&base);
}

#[test]
fn test_rebuild_preserves_episode_ids() {
    let base = temp_base("rebuild_ids");
    let mut store = LtmStore::create(&base, 64).expect("create");
    let mut ids = Vec::new();
    for text in ["first", "second", "third"] {
        ids.push(
            store
                .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, text)
                .unwrap(),
        );
    }
    store.delete(ids[0]).unwrap();
    assert_eq!(store.rebuild_bundles().unwrap(), 2);
    drop(store);

    let mut store = LtmStore::open(&base).unwrap();
    let new_id = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "fourth")
        .unwrap();
    assert!(new_id > ids[2]);
    assert_eq!(store.retrieve(ids[2]).unwrap().text, "third");
    assert_eq!(store.retrieve(new_id).unwrap().text, "fourth");
    assert_eq!(store.active_episode_ids(), &[ids[1], ids[2], new_id]);
    drop(store);
    cleanup(&base);
}

#[test]
fn test_open_reconciles_stale_bundle_id_counter() {
    let base = temp_base("stale_bundle_id");
    let mut store = LtmStore::create(&base, 64).unwrap();
    let first = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "first")
        .unwrap();
    drop(store);
    let mut bundle = std::fs::OpenOptions::new()
        .write(true)
        .open(base.with_extension("bundles"))
        .unwrap();
    bundle.seek(std::io::SeekFrom::Start(24)).unwrap();
    bundle.write_all(&first.to_le_bytes()).unwrap();
    drop(bundle);

    let mut store = LtmStore::open(&base).unwrap();
    let second = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "second")
        .unwrap();
    assert!(second > first);
    assert_eq!(store.retrieve(first).unwrap().text, "first");
    drop(store);
    cleanup(&base);
}

#[test]
fn test_open_rejects_partial_metadata_without_changing_files() {
    let base = temp_base("partial_meta_open");
    let mut store = LtmStore::create(&base, 64).unwrap();
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "first")
        .unwrap();
    drop(store);
    let mut meta = std::fs::OpenOptions::new()
        .append(true)
        .open(base.with_extension("meta"))
        .unwrap();
    meta.write_all(&[0x11; 7]).unwrap();
    drop(meta);
    let before = std::fs::read(base.with_extension("meta")).unwrap();

    let result = LtmStore::open(&base);
    assert!(
        matches!(result, Err(LtmError::Io(ref e)) if e.kind() == std::io::ErrorKind::InvalidData)
    );
    assert_eq!(std::fs::read(base.with_extension("meta")).unwrap(), before);
    cleanup(&base);
}

#[test]
fn test_store_rejects_partial_metadata_before_any_write() {
    let base = temp_base("partial_meta_store");
    let mut store = LtmStore::create(&base, 64).unwrap();
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "first")
        .unwrap();
    let mut meta = std::fs::OpenOptions::new()
        .append(true)
        .open(base.with_extension("meta"))
        .unwrap();
    meta.write_all(&[0x11; 7]).unwrap();
    drop(meta);
    let paths = ["meta", "dat", "bundles"].map(|ext| base.with_extension(ext));
    let before = paths.each_ref().map(|path| std::fs::read(path).unwrap());

    let result = store.store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "second");
    assert!(
        matches!(result, Err(LtmError::Io(ref e)) if e.kind() == std::io::ErrorKind::InvalidData)
    );
    assert_eq!(store.count(), 1);
    for (path, contents) in paths.iter().zip(before) {
        assert_eq!(std::fs::read(path).unwrap(), contents);
    }
    drop(store);
    cleanup(&base);
}

#[test]
fn test_open_or_create_preserves_incomplete_existing_store() {
    let base = temp_base("incomplete_store");
    let mut store = LtmStore::create(&base, 64).unwrap();
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "first")
        .unwrap();
    drop(store);
    let backup = base.with_extension("meta.saved");
    std::fs::rename(base.with_extension("meta"), &backup).unwrap();
    let data_before = std::fs::read(base.with_extension("dat")).unwrap();
    let bundles_before = std::fs::read(base.with_extension("bundles")).unwrap();

    let result = LtmStore::open_or_create(&base, 64);
    assert!(result.is_err());
    assert_eq!(
        std::fs::read(base.with_extension("dat")).unwrap(),
        data_before
    );
    assert_eq!(
        std::fs::read(base.with_extension("bundles")).unwrap(),
        bundles_before
    );
    assert!(!base.with_extension("meta").exists());
    std::fs::rename(backup, base.with_extension("meta")).unwrap();
    cleanup(&base);
}

#[test]
fn test_retrieve_rejects_understated_uncompressed_length() {
    let base = temp_base("short_payload_length");
    let mut store = LtmStore::create(&base, 64).unwrap();
    let id = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "complete text",
        )
        .unwrap();
    let length = store.index_entry_ref(0).expect("slot 0").uncompressed_len;
    drop(store);
    let mut meta = std::fs::OpenOptions::new()
        .write(true)
        .open(base.with_extension("meta"))
        .unwrap();
    meta.seek(std::io::SeekFrom::Start(64 + 36)).unwrap();
    meta.write_all(&(length - 1).to_le_bytes()).unwrap();
    drop(meta);

    let mut store = LtmStore::open(&base).unwrap();
    assert!(matches!(
        store.retrieve(id),
        Err(LtmError::DecompressFailed(_))
    ));
    drop(store);
    cleanup(&base);
}

#[test]
fn test_rebuild_failure_preserves_existing_bundle() {
    let base = temp_base("rebuild_failure");
    let mut store = LtmStore::create(&base, 64).unwrap();
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "first")
        .unwrap();
    let before = std::fs::read(base.with_extension("bundles")).unwrap();
    let mut data = std::fs::OpenOptions::new()
        .write(true)
        .open(base.with_extension("dat"))
        .unwrap();
    data.seek(std::io::SeekFrom::Start(16)).unwrap();
    data.write_all(&u32::MAX.to_le_bytes()).unwrap();
    drop(data);

    assert!(store.rebuild_bundles().is_err());
    assert_eq!(
        std::fs::read(base.with_extension("bundles")).unwrap(),
        before
    );
    drop(store);
    cleanup(&base);
}

#[test]
fn test_rebuild_does_not_truncate_existing_mappings() {
    // The sole-mapper flock makes a second live mapping of the same
    // file impossible, so rebuild's safety can no longer be observed
    // through a stale mapping. The invariant it encodes still holds
    // structurally: rebuild writes a private `bundles.rebuild-*` file
    // and installs it by rename, so the old inode is never truncated
    // under a live mapping — and any process holding the old mapping
    // keeps the old inode by construction.
    let base = temp_base("rebuild_mapping");
    let mut store = LtmStore::create(&base, 64).unwrap();
    let first = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "first")
        .unwrap();
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "second")
        .unwrap();
    store.delete(first).unwrap();
    // A second live mapping of the same file is refused: EWOULDBLOCK.
    match BundleStore::open(base.with_extension("bundles")) {
        Err(err) => assert_eq!(err.raw_os_error(), Some(libc::EWOULDBLOCK)),
        Ok(_) => panic!("second live mapping of a locked file must fail"),
    }
    assert_eq!(store.rebuild_bundles().unwrap(), 1);
    // The rebuilt store contains only the surviving episode.
    assert_eq!(store.count(), 1);
    drop(store);
    cleanup(&base);
}

#[test]
fn test_stale_data_offset_does_not_overwrite_episodes() {
    let base = temp_base("stale_data_offset");
    let mut store = LtmStore::create(&base, 64).unwrap();
    let first = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "first")
        .unwrap();
    drop(store);
    let mut data = std::fs::OpenOptions::new()
        .write(true)
        .open(base.with_extension("dat"))
        .unwrap();
    data.seek(std::io::SeekFrom::Start(8)).unwrap();
    data.write_all(&16_u64.to_le_bytes()).unwrap();
    drop(data);

    let mut store = LtmStore::open(&base).unwrap();
    let second = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "second")
        .unwrap();
    assert_eq!(store.retrieve(first).unwrap().text, "first");
    assert_eq!(store.retrieve(second).unwrap().text, "second");
    drop(store);
    cleanup(&base);
}

fn make_v1_store(base: &Path) -> (Vec<u8>, Vec<u8>) {
    let mut store = LtmStore::create(base, 64).unwrap();
    let first = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "first")
        .unwrap();
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "second")
        .unwrap();
    store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "third")
        .unwrap();
    store.delete(first).unwrap();
    drop(store);
    let mut index = std::fs::read(base.with_extension("meta")).unwrap();
    index[..4].copy_from_slice(b"LTMI");
    index[4..8].copy_from_slice(&1_u32.to_le_bytes());
    index[8..12].copy_from_slice(&3_u32.to_le_bytes());
    std::fs::write(base.with_extension("idx"), &index).unwrap();
    let mut data = std::fs::read(base.with_extension("dat")).unwrap();
    data[4..8].copy_from_slice(&1_u32.to_le_bytes());
    std::fs::write(base.with_extension("dat"), &data).unwrap();
    std::fs::remove_file(base.with_extension("meta")).unwrap();
    std::fs::remove_file(base.with_extension("bundles")).unwrap();
    (index, data)
}

#[test]
fn test_v1_migration_preserves_ids_payloads_and_backups() {
    let base = temp_base("v1_migration");
    let (index_before, data_before) = make_v1_store(&base);
    let store = LtmStore::open_or_create(&base, 64).unwrap();
    assert_eq!(store.active_episode_ids(), &[2, 3]);
    drop(store);
    let mut store = LtmStore::open_or_create(&base, 64).unwrap();
    assert_eq!(store.retrieve(2).unwrap().text, "second");
    assert_eq!(store.retrieve(3).unwrap().text, "third");
    assert_eq!(
        std::fs::read(base.with_extension("idx.v1bak")).unwrap(),
        index_before
    );
    assert_eq!(
        std::fs::read(base.with_extension("dat.v1bak")).unwrap(),
        data_before
    );
    assert!(
        store
            .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "fourth")
            .unwrap()
            > 3
    );
    drop(store);
    cleanup(&base);
    std::fs::remove_file(base.with_extension("idx.v1bak")).unwrap();
    std::fs::remove_file(base.with_extension("dat.v1bak")).unwrap();
}

#[test]
fn test_v1_migration_failure_preserves_original_files() {
    let base = temp_base("v1_failure");
    let (index_before, mut data_before) = make_v1_store(&base);
    let offset = u64::from_le_bytes(index_before[136..144].try_into().unwrap()) as usize;
    data_before[offset..offset + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(base.with_extension("dat"), &data_before).unwrap();

    assert!(LtmStore::open_or_create(&base, 64).is_err());
    assert_eq!(
        std::fs::read(base.with_extension("idx")).unwrap(),
        index_before
    );
    assert_eq!(
        std::fs::read(base.with_extension("dat")).unwrap(),
        data_before
    );
    assert!(!base.with_extension("meta").exists());
    cleanup(&base);
    std::fs::remove_file(base.with_extension("idx")).unwrap();
}

#[test]
fn test_meta_memory_episodes_stay_out_of_bundles() {
    // Modules 0/5/8/9 are auto-flagged meta-memory — internal
    // bookkeeping excluded from find_similar. Their SDRs must not
    // enter the bundle superposition either, or every run re-saturates
    // the store with bookkeeping noise that the next startup rebuild
    // has to mop up again (~260 entries/run observed live).
    let base = temp_base("meta_bundles");
    let mut store = LtmStore::create(&base, 64).unwrap();
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "real episode",
        )
        .unwrap();
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            0,
            "association trace",
        )
        .unwrap();
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            9,
            "dream insight",
        )
        .unwrap();
    // The index holds all three; only the real one may superimpose.
    assert_eq!(store.count(), 3);
    drop(store);
    let bundles = BundleStore::open(base.with_extension("bundles")).unwrap();
    assert_eq!(bundles.episode_count(), 1);
    cleanup(&base);
}

#[test]
fn test_archive_unbundles_episode() {
    // Archiving removes the episode from the active set — its SDR
    // must leave the superposition with it. Before the unbundle
    // path existed, every archive left dead weight that the next
    // startup reconciliation rebuilt away on every boot.
    let base = temp_base("archive_unbundle");
    let mut store = LtmStore::create(&base, 64).unwrap();
    let id = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "real episode about the Great Salt Lake",
        )
        .unwrap();
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            9,
            "dream insight",
        )
        .unwrap();
    drop(store);
    let bundles = BundleStore::open(base.with_extension("bundles")).unwrap();
    assert_eq!(bundles.episode_count(), 1);
    drop(bundles);

    let mut store = LtmStore::open(&base).unwrap();
    store.archive(id).expect("archive");
    drop(store);

    // bundle_count tracks the bundled set, not every stored ID —
    // archive is the moment this episode leaves it.
    let bundles = BundleStore::open(base.with_extension("bundles")).unwrap();
    assert_eq!(bundles.episode_count(), 0);
    cleanup(&base);
}

#[test]
fn test_delete_unbundles_episode() {
    let base = temp_base("delete_unbundle");
    let mut store = LtmStore::create(&base, 64).unwrap();
    let id = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "real episode about the Great Salt Lake",
        )
        .unwrap();
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "another real episode",
        )
        .unwrap();
    drop(store);
    let bundles = BundleStore::open(base.with_extension("bundles")).unwrap();
    assert_eq!(bundles.episode_count(), 2);
    drop(bundles);

    let mut store = LtmStore::open(&base).unwrap();
    store.delete(id).expect("delete");
    drop(store);

    let bundles = BundleStore::open(base.with_extension("bundles")).unwrap();
    assert_eq!(bundles.episode_count(), 1);
    cleanup(&base);
}

#[test]
fn test_delete_of_archived_does_not_double_unbundle() {
    // An archived episode's SDR was already subtracted when it left
    // the active set. Deleting it afterwards must not subtract
    // again — a second subtraction would corrupt the shared
    // superposition and under-count the bundled set.
    let base = temp_base("archive_delete");
    let mut store = LtmStore::create(&base, 64).unwrap();
    let id = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "real episode about the Great Salt Lake",
        )
        .unwrap();
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "another real episode",
        )
        .unwrap();
    store.archive(id).expect("archive");
    drop(store);
    let bundles = BundleStore::open(base.with_extension("bundles")).unwrap();
    assert_eq!(bundles.episode_count(), 1);
    drop(bundles);

    let mut store = LtmStore::open(&base).unwrap();
    store.delete(id).expect("delete archived");
    drop(store);

    let bundles = BundleStore::open(base.with_extension("bundles")).unwrap();
    assert_eq!(
        bundles.episode_count(),
        1,
        "deleting an archived episode must not double-subtract its SDR"
    );
    cleanup(&base);
}

#[test]
fn test_store_meta_non_auto_module_unbundles() {
    // store_meta with a non-auto module (e.g. 4, Language — not in
    // the 0/5/8/9 auto set) goes through store(), which bundles the
    // SDR before the meta flag is applied. The flag path must back
    // the contribution out or every store_meta call leaves dead
    // weight the startup reconciliation rebuilds away.
    let base = temp_base("store_meta_unbundle");
    let mut store = LtmStore::create(&base, 64).unwrap();
    store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "real episode",
        )
        .unwrap();
    store
        .store_meta(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "language-module bookkeeping trace",
        )
        .unwrap();
    drop(store);

    let bundles = BundleStore::open(base.with_extension("bundles")).unwrap();
    assert_eq!(
        bundles.episode_count(),
        1,
        "store_meta on a non-auto module must not leave its SDR bundled"
    );
    cleanup(&base);
}

#[test]
fn test_search_archived_pages_the_cold_tier() {
    let base = temp_base("search_archived");
    let mut store = LtmStore::create(&base, 64).expect("create");

    // Three real episodes archived, one active, one deleted, one
    // archived meta-memory.
    let mut archived = Vec::new();
    for i in 0..3u8 {
        let id = store
            .store(
                0,
                0.5,
                make_full_tag(),
                make_compact_tag(),
                i,
                4,
                &format!("archived episode {i}"),
            )
            .expect("store");
        store.archive(id).expect("archive");
        archived.push(id);
    }
    let active = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "still active",
        )
        .expect("store active");
    let doomed = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "deleted while active",
        )
        .expect("store doomed");
    store.delete(doomed).expect("delete");
    let meta = store
        .store_meta(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "archived meta trace",
        )
        .expect("store meta");
    store.archive(meta).expect("archive meta");

    let page = store.search_archived(100, 0);
    let ids: Vec<u64> = page.iter().map(|(id, _)| *id).collect();

    // Oldest-first ascending order over the archived tier.
    assert_eq!(ids[..3], archived[..3], "archived real episodes, in order");
    // The archived meta-memory rides the tier (the caller's policy
    // filters it); the active episode and the deleted one do not.
    assert!(ids.contains(&meta), "archived meta-memory is enumerable");
    assert!(!ids.contains(&active), "active episode excluded");
    assert!(!ids.contains(&doomed), "deleted episode excluded");
    assert_eq!(ids.len(), 4);

    // Positional paging: skip(1) drops the oldest archived entry.
    let page2 = store.search_archived(2, 1);
    let ids2: Vec<u64> = page2.iter().map(|(id, _)| *id).collect();
    assert_eq!(ids2.len(), 2);
    assert_eq!(ids2[0], archived[1], "offset skips into the tier");

    // Metadata survives the round trip.
    let (_, entry) = page[0];
    assert_eq!(entry.episode_id, archived[0]);
    assert!(entry.is_archived());
    cleanup(&base);
}

#[test]
fn test_delete_removes_from_archived_tier() {
    let base = temp_base("delete_archived");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let id = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "substrate death candidate",
        )
        .expect("store");
    store.archive(id).expect("archive");

    assert_eq!(store.search_archived(10, 0).len(), 1);
    store.delete(id).expect("delete");

    assert!(
        store.search_archived(10, 0).is_empty(),
        "deleted engram leaves the archived tier"
    );
    assert!(
        store
            .find_similar_archived("substrate death", 10)
            .is_empty()
    );
    assert!(store.retrieve(id).is_err(), "deleted payload unreachable");
    cleanup(&base);
}

// ─── Empty payload rejection ──────────────────────────────────

#[test]
fn test_store_rejects_empty_text() {
    let base = temp_base("empty_text");
    let mut store = LtmStore::create(&base, 64).expect("create");

    // Empty and whitespace-only texts are noise, not memories — an
    // empty text's SimHash degenerates to 0, which made every empty
    // episode "associated" with every other at distance 0. That was
    // the seed of the bridge flood.
    assert!(matches!(
        store.store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, ""),
        Err(LtmError::EmptyPayload)
    ));
    assert!(matches!(
        store.store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "   \t\n  "
        ),
        Err(LtmError::EmptyPayload)
    ));

    // A rejected store burns no episode ID and adds nothing.
    assert_eq!(store.count(), 0);
    assert!(store.active_episode_ids().is_empty());

    // A real episode still stores fine.
    let id = store
        .store(0, 0.5, make_full_tag(), make_compact_tag(), 0, 4, "real")
        .expect("non-empty stores");
    assert_eq!(store.count(), 1);
    assert_eq!(store.retrieve(id).expect("retrieve").text, "real");

    drop(store);
    cleanup(&base);
}

// ─── Bulk archive ─────────────────────────────────────────────

#[test]
fn test_archive_many() {
    let base = temp_base("archive_many");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let mut ids = Vec::new();
    for i in 0..5u64 {
        ids.push(
            store
                .store(
                    i,
                    0.5,
                    make_full_tag(),
                    make_compact_tag(),
                    0,
                    0,
                    &format!("episode {i}"),
                )
                .expect("store"),
        );
    }
    assert_eq!(store.count(), 5);

    // Batch archive — unknown, duplicate, and re-archived IDs skipped.
    let n = store
        .archive_many(&[ids[0], ids[2], ids[4], 999_999, ids[2]])
        .expect("archive_many");
    assert_eq!(n, 3);
    assert_eq!(store.count(), 2);

    // Archived payloads remain retrievable by ID — lineage, not
    // amputation.
    assert_eq!(store.retrieve(ids[0]).expect("archived").text, "episode 0");
    assert_eq!(store.retrieve(ids[4]).expect("archived").text, "episode 4");
    assert_eq!(store.retrieve(ids[1]).expect("live").text, "episode 1");

    // Re-archiving is a no-op.
    assert_eq!(store.archive_many(&[ids[0]]).expect("no-op"), 0);
    assert_eq!(store.count(), 2);

    drop(store);
    cleanup(&base);
}

// ─── Meta-memory dedup index ──────────────────────────────────

#[test]
fn test_meta_hash_known_survives_reopen() {
    let base = temp_base("meta_hash");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let text = "[association] episode 1 \u{2194} episode 2 (distance: 9)\ta\tb";
    let fingerprint = simhash(text);
    assert!(!store.meta_hash_known(fingerprint));

    let id = store
        .store_meta(0, 0.5, make_full_tag(), make_compact_tag(), 2, 0, text)
        .expect("store_meta");
    assert!(store.meta_hash_known(fingerprint));
    assert!(store.is_meta_episode(id));

    // A real episode is not meta.
    let real = store
        .store(
            0,
            0.5,
            make_full_tag(),
            make_compact_tag(),
            0,
            4,
            "real memory",
        )
        .expect("store");
    assert!(!store.is_meta_episode(real));

    drop(store);

    // The dedup index is rebuilt from active meta entries on open —
    // dedup survives restarts.
    let reopened = LtmStore::open(&base).expect("open");
    assert!(reopened.meta_hash_known(fingerprint));

    drop(reopened);
    cleanup(&base);
}
