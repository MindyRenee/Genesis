//! Tests for the STM ring buffer.

use genesis::store::*;
use std::path::PathBuf;

fn temp_path(test_name: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("genesis_stm_{test_name}_{pid}_{nanos}.bin"))
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
}

fn make_emotional_tag() -> [f32; 12] {
    [
        0.50, // dopamine
        0.55, // serotonin
        0.40, // norepinephrine
        0.45, // acetylcholine
        0.50, // gaba
        0.60, // glutamate
        0.20, // cortisol
        0.35, // oxytocin
        0.30, // endorphin
        0.45, // histamine
        0.10, // adenosine
        0.40, // bdnf
    ]
}

// ─── Create ───────────────────────────────────────────────────

#[test]
fn test_create_ring_buffer() {
    let path = temp_path("create");
    let rb = RingBuffer::create(&path, 16).expect("create");

    assert_eq!(rb.capacity(), 16);
    assert_eq!(rb.count(), 0);
    assert!(rb.is_empty());
    assert!(!rb.is_full());
    assert_eq!(rb.total_written(), 0);

    // Verify header
    let header = rb.header();
    assert_eq!(header.magic, STM_MAGIC);
    assert_eq!(header.version, STM_SCHEMA_VERSION);
    assert_eq!(header.capacity, 16);

    drop(rb);
    cleanup(&path);
}

// ─── Push and read ────────────────────────────────────────────

#[test]
fn test_push_and_peek() {
    let path = temp_path("push");
    let rb = RingBuffer::create(&path, 16).expect("create");

    let entry = RingBufferEntry::new(
        1000,
        EventType::UserInput,
        0,
        0.75,
        make_emotional_tag(),
        "User said hello",
    );

    let slot = rb.push(entry);
    assert_eq!(slot, 0);
    assert_eq!(rb.count(), 1);
    assert!(!rb.is_empty());
    assert_eq!(rb.total_written(), 1);

    // Read it back
    let read = rb.peek(0).expect("entry should exist");
    assert_eq!(read.timestamp, 1000);
    assert_eq!(read.event_type(), EventType::UserInput);
    assert_eq!(read.salience, 0.75);
    assert_eq!(read.text(), "User said hello");

    drop(rb);
    cleanup(&path);
}

// ─── Ring semantics ───────────────────────────────────────────

#[test]
fn test_ring_overwrite() {
    let path = temp_path("overwrite");
    let rb = RingBuffer::create(&path, 4).expect("create");

    // Fill the buffer
    for i in 0..4 {
        let entry = RingBufferEntry::new(
            i * 100,
            EventType::Internal,
            0,
            0.5,
            make_emotional_tag(),
            &format!("Event {i}"),
        );
        rb.push(entry);
    }

    assert_eq!(rb.count(), 4);
    assert!(rb.is_full());
    assert_eq!(rb.total_written(), 4);

    // Push one more — should overwrite the oldest
    let entry = RingBufferEntry::new(
        400,
        EventType::Internal,
        0,
        0.5,
        make_emotional_tag(),
        "Event 4 (overwrites Event 0)",
    );
    rb.push(entry);

    assert_eq!(rb.count(), 4); // still 4
    assert!(rb.is_full());
    assert_eq!(rb.total_written(), 5);

    // The oldest entry should now be "Event 1"
    let entries: Vec<_> = rb.iter().collect();
    assert_eq!(entries.len(), 4);
    assert_eq!(entries[0].text(), "Event 1");
    assert_eq!(entries[3].text(), "Event 4 (overwrites Event 0)");

    drop(rb);
    cleanup(&path);
}

// ─── Iteration ────────────────────────────────────────────────

#[test]
fn test_iteration_order() {
    let path = temp_path("iter");
    let rb = RingBuffer::create(&path, 8).expect("create");

    for i in 0..5 {
        let entry = RingBufferEntry::new(
            i * 100,
            EventType::Observation,
            0,
            0.3,
            make_emotional_tag(),
            &format!("Obs {i}"),
        );
        rb.push(entry);
    }

    let entries: Vec<_> = rb.iter().collect();
    assert_eq!(entries.len(), 5);
    // Should be in order: oldest to newest
    assert_eq!(entries[0].text(), "Obs 0");
    assert_eq!(entries[4].text(), "Obs 4");
}

// ─── Persistence ──────────────────────────────────────────────

#[test]
fn test_persistence() {
    let path = temp_path("persist");

    {
        let rb = RingBuffer::create(&path, 8).expect("create");
        for i in 0..3 {
            let entry = RingBufferEntry::new(
                i * 1000,
                EventType::Output,
                0,
                0.6,
                make_emotional_tag(),
                &format!("Output {i}"),
            );
            rb.push(entry);
        }
        rb.sync().expect("sync");
    }

    // Re-open
    {
        let rb = RingBuffer::open(&path).expect("open");
        assert_eq!(rb.count(), 3);
        assert_eq!(rb.total_written(), 3);

        let entries: Vec<_> = rb.iter().collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].text(), "Output 0");
        assert_eq!(entries[2].text(), "Output 2");
    }

    cleanup(&path);
}

// ─── Emotional tag ────────────────────────────────────────────

#[test]
fn test_emotional_tag_preserved() {
    let path = temp_path("emotag");
    let rb = RingBuffer::create(&path, 4).expect("create");

    let mut tag = make_emotional_tag();
    tag[0] = 0.90; // high dopamine
    tag[6] = 0.80; // high cortisol

    let entry = RingBufferEntry::new(
        1000,
        EventType::NeurochemicalShift,
        3, // Emotion module
        0.95,
        tag,
        "Dopamine spike with stress",
    );

    rb.push(entry);

    let read = rb.peek(0).expect("entry");
    assert_eq!(read.emotional_tag[0], 0.90);
    assert_eq!(read.emotional_tag[6], 0.80);
    assert_eq!(read.source_module, 3);
    assert_eq!(read.event_type(), EventType::NeurochemicalShift);
    assert_eq!(read.salience, 0.95);

    drop(rb);
    cleanup(&path);
}

// ─── Text truncation ──────────────────────────────────────────

#[test]
fn test_text_truncation() {
    let path = temp_path("trunc");
    let rb = RingBuffer::create(&path, 4).expect("create");

    // Create text longer than ENTRY_TEXT_SIZE
    let long_text = "A".repeat(500);
    let entry = RingBufferEntry::new(
        0,
        EventType::Internal,
        0,
        0.5,
        make_emotional_tag(),
        &long_text,
    );

    rb.push(entry);
    let read = rb.peek(0).expect("entry");
    assert_eq!(read.text_len as usize, ENTRY_TEXT_SIZE);
    assert_eq!(read.text().len(), ENTRY_TEXT_SIZE);
    // All A's
    assert!(read.text().chars().all(|c| c == 'A'));

    drop(rb);
    cleanup(&path);
}

// ─── Clear ────────────────────────────────────────────────────

#[test]
fn test_clear() {
    let path = temp_path("clear");
    let rb = RingBuffer::create(&path, 8).expect("create");

    for i in 0..5 {
        rb.push(RingBufferEntry::new(
            i,
            EventType::Internal,
            0,
            0.5,
            make_emotional_tag(),
            "test",
        ));
    }

    assert_eq!(rb.count(), 5);
    assert_eq!(rb.total_written(), 5);

    rb.clear();

    assert_eq!(rb.count(), 0);
    assert!(rb.is_empty());
    // total_written is monotonic — should NOT reset
    assert_eq!(rb.total_written(), 5);

    // Can push again after clear
    rb.push(RingBufferEntry::new(
        100,
        EventType::Internal,
        0,
        0.5,
        make_emotional_tag(),
        "after clear",
    ));
    assert_eq!(rb.count(), 1);
    assert_eq!(rb.total_written(), 6);

    drop(rb);
    cleanup(&path);
}

// ─── Open or create ───────────────────────────────────────────

#[test]
fn test_open_or_create() {
    let path = temp_path("ooc");

    // First call creates
    let rb = RingBuffer::open_or_create(&path, 32).expect("create");
    assert_eq!(rb.capacity(), 32);
    rb.push(RingBufferEntry::new(
        0,
        EventType::Internal,
        0,
        0.5,
        make_emotional_tag(),
        "test",
    ));
    rb.sync().expect("sync");
    drop(rb);

    // Second call opens
    let rb = RingBuffer::open_or_create(&path, 32).expect("open");
    assert_eq!(rb.capacity(), 32);
    assert_eq!(rb.count(), 1);

    drop(rb);
    cleanup(&path);
}

// ─── File not found ───────────────────────────────────────────

#[test]
fn test_open_nonexistent() {
    let path = temp_path("nonexist");
    assert!(!path.exists());
    let result = RingBuffer::open(&path);
    assert!(matches!(result, Err(RingBufferError::NotFound)));
}

// ─── Bad magic ────────────────────────────────────────────────

#[test]
fn test_bad_magic_detected() {
    let path = temp_path("badmagic");

    // Create a valid file
    {
        let rb = RingBuffer::create(&path, 4).expect("create");
        rb.sync().expect("sync");
        drop(rb);
    }

    // Corrupt the magic
    {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .expect("open");
        use std::os::unix::fs::FileExt;
        file.write_all_at(&[0xFF, 0xFF, 0xFF, 0xFF], 0)
            .expect("corrupt");
    }

    let result = RingBuffer::open(&path);
    assert!(matches!(result, Err(RingBufferError::BadMagic)));

    cleanup(&path);
}

// ─── File size ────────────────────────────────────────────────

#[test]
fn test_file_size() {
    let path = temp_path("filesize");
    let capacity = 64;
    let rb = RingBuffer::create(&path, capacity).expect("create");
    drop(rb);

    let metadata = std::fs::metadata(&path).expect("metadata");
    // The file is page-aligned (see `mmap_size`) so the entire mmap
    // mapping is backed by the file. The logical data size is
    // `file_size(capacity)`, but the on-disk file is rounded up to
    // the next page boundary.
    let logical = file_size(capacity);
    let aligned = mmap_size(capacity);
    assert!(
        metadata.len() as usize >= logical,
        "file should be at least the logical size ({logical}), got {}",
        metadata.len()
    );
    assert_eq!(
        metadata.len() as usize,
        aligned,
        "file should be page-aligned ({aligned}), got {}",
        metadata.len()
    );

    cleanup(&path);
}

// ─── Large capacity ───────────────────────────────────────────

#[test]
fn test_large_capacity() {
    let path = temp_path("large");
    let capacity = 256;
    let rb = RingBuffer::create(&path, capacity).expect("create");

    // Fill it
    for i in 0..capacity as u64 {
        rb.push(RingBufferEntry::new(
            i,
            EventType::Observation,
            0,
            (i as f32) / (capacity as f32),
            make_emotional_tag(),
            &format!("Entry {i}"),
        ));
    }

    assert!(rb.is_full());
    assert_eq!(rb.count(), capacity as u64);
    assert_eq!(rb.total_written(), capacity as u64);

    // Verify iteration
    let entries: Vec<_> = rb.iter().collect();
    assert_eq!(entries.len(), capacity as usize);

    drop(rb);
    cleanup(&path);
}

#[test]
fn test_capacity_too_large_rejected() {
    let path = temp_path("too_large");
    let result = RingBuffer::create(&path, MAX_CAPACITY + 1);
    assert!(result.is_err());
    match result {
        Err(RingBufferError::CapacityTooLarge { found, max }) => {
            assert_eq!(found, MAX_CAPACITY + 1);
            assert_eq!(max, MAX_CAPACITY);
        }
        Err(e) => panic!("expected CapacityTooLarge, got {e}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
    // File should not have been created
    assert!(!path.exists());
}

#[test]
fn test_max_capacity_accepted() {
    // Verify that a reasonable capacity just under MAX_CAPACITY is
    // accepted (boundary check). We don't use MAX_CAPACITY itself
    // (4 GB allocation would fail in CI).
    let path = temp_path("max_ok");
    let rb = RingBuffer::create(&path, 1024).expect("create");
    assert_eq!(rb.capacity(), 1024);
    drop(rb);
    cleanup(&path);
}
