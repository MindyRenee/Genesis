//! Long-term episodic memory (LTM) store — SDR/LogHD architecture.
//!
//! This is Genesis's durable memory. Episodes consolidated from
//! short-term memory persist here across restarts.
//!
//! # Architecture: three-file design
//!
//! ```text
//! ┌─────────────────────────────────────────────────┐
//! │  Bundles file (.bundles) — mmap'd, fixed 800 KB │
//! │  20 dense f32 hypervectors (10,000-dim each)    │
//! │  LogHD class-axis compressed SDR index          │
//! │  Fixed size regardless of episode count         │
//! └─────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────┐
//! │  Metadata file (.meta) — append-only            │
//! │  64-byte IndexEntry per episode                 │
//! │  Grows with episodes (no fixed capacity)        │
//! └─────────────────────────────────────────────────┘
//!
//! ┌─────────────────────────────────────────────────┐
//! │  Data file (.dat) — append-only                 │
//! │  Compressed episode payloads                    │
//! │  Same format as v1                              │
//! └─────────────────────────────────────────────────┘
//! ```
//!
//! # What changed from v1
//!
//! v1 used a fixed-capacity mmap'd index (65,536 slots × 64 bytes).
//! When it filled up, no new memories could be stored — Genesis
//! stopped learning.
//!
//! v2 replaces the fixed index with:
//! - A **LogHD bundle store** (fixed 800 KB) for SDR-based
//!   similarity matching with effectively unlimited capacity
//!   (4^20 ≈ 1 trillion episodes)
//! - An **append-only metadata file** that grows with episodes
//!   (no capacity limit)
//!
//! The SimHash (64-bit) is retained per episode for backward
//! compatibility with the association engine's Hamming distance
//! matching. The SDR provides higher-dimensional similarity for
//! `find_similar` queries.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

use crate::store::sdr::{
    BundleStore, LOGHD_N, code_to_expected_activation, encode_episode, episode_id_to_code,
};

// ─────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────

/// Magic bytes for LTM metadata files: ASCII `"LTMM"`.
pub const LTM_META_MAGIC: [u8; 4] = *b"LTMM";
/// Magic bytes for LTM data files: ASCII `"LTMD"`.
pub const LTM_DATA_MAGIC: [u8; 4] = *b"LTMD";

/// Schema version for LTM metadata and data files.
pub const LTM_SCHEMA_VERSION: u32 = 2;

/// Size of each metadata entry (same as v1 IndexEntry).
pub const INDEX_ENTRY_SIZE: usize = 64;

/// Kept for backward compatibility — no longer a hard limit.
pub const DEFAULT_INDEX_CAPACITY: u32 = u32::MAX;

/// Number of neurochemical levels in the emotional tag.
pub const EMOTIONAL_TAG_SIZE: usize = 12;

/// Maximum uncompressed payload size per episode (256 KB).
pub const MAX_PAYLOAD_SIZE: usize = 256 * 1024;

/// Maximum compressed payload size per episode (320 KB).
/// A near-max incompressible payload (256 KB uncompressed) can
/// legitimately compress to slightly more than 256 KB due to zlib
/// overhead, so the compressed limit is larger than the uncompressed
/// limit.
pub const MAX_COMPRESSED_SIZE: usize = 320 * 1024;

// ─────────────────────────────────────────────────────────────────
//  Index entry (same layout as v1 — 64 bytes)
// ─────────────────────────────────────────────────────────────────

/// A single metadata entry — same layout as v1 for IPC compatibility.
///
/// ## Layout (64 bytes)
/// ```text
/// offset  field              type    notes
/// ------  -----              ----    -----
///   0     episode_id         u64     unique monotonic ID
///   8     data_offset        u64     offset into the data file
///  16     timestamp          u64     when the episode occurred
///  24     association_hash   u64     SimHash fingerprint (backward compat)
///  32     compressed_len     u32     bytes in the data file
///  36     uncompressed_len   u32     original payload size
///  40     salience           f32     emotional importance [0,1]
///  44     emotional_tag      [f32;4] compact: arousal, valence, cortisol, dopamine
///  60     flags              u8      bit 0: deleted, bits 1-7: reserved
///  61     source_module      u8      module that created the episode
///  62     _pad               [u8;2]  alignment
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IndexEntry {
    /// Unique monotonic episode identifier.
    pub episode_id: u64,
    /// Byte offset of the compressed payload in the data file.
    pub data_offset: u64,
    /// Unix timestamp (ms) when the episode was recorded.
    pub timestamp: u64,
    /// 64-bit SimHash fingerprint used for association matching.
    pub association_hash: u64,
    /// Length of the compressed payload in the data file (bytes).
    pub compressed_len: u32,
    /// Original uncompressed payload size (bytes).
    pub uncompressed_len: u32,
    /// Emotional importance of the episode, in `[0,1]`.
    pub salience: f32,
    /// Compact emotional snapshot: arousal, valence, cortisol, dopamine.
    pub emotional_tag: [f32; 4],
    /// Bit flags (bit 0: deleted, bit 1: meta-memory, bit 2: archived).
    pub flags: u8,
    /// Identifier of the module that created the episode.
    pub source_module: u8,
    /// Padding to keep the entry at exactly 64 bytes.
    pub _pad: [u8; 2],
}

impl IndexEntry {
    /// Flag bit marking an episode as logically deleted.
    pub const DELETED_FLAG: u8 = 1 << 0;
    /// Bit 1: this episode is a meta-memory (association trace or
    /// dream insight), not a real source episode. Meta-memories are
    /// excluded from `find_similar` results so that retrieval returns
    /// real conversation/learning episodes, not association bridges.
    pub const META_MEMORY_FLAG: u8 = 1 << 1;
    /// Bit 2: this episode has been archived by sleep compression.
    /// Archived episodes are excluded from `find_similar`,
    /// `recent_episodes`, and the bundle store (same as deleted),
    /// but unlike deleted episodes they remain retrievable by ID.
    /// This preserves dream insight references and conversation
    /// history — nothing is ever lost.
    pub const ARCHIVED_FLAG: u8 = 1 << 2;

    #[allow(clippy::too_many_arguments)]
    /// Construct a new index entry with no flags set.
    pub fn new(
        episode_id: u64,
        data_offset: u64,
        compressed_len: u32,
        uncompressed_len: u32,
        timestamp: u64,
        salience: f32,
        association_hash: u64,
        emotional_tag: [f32; 4],
        source_module: u8,
    ) -> Self {
        Self {
            episode_id,
            data_offset,
            compressed_len,
            uncompressed_len,
            timestamp,
            salience,
            association_hash,
            emotional_tag,
            flags: 0,
            source_module,
            _pad: [0; 2],
        }
    }

    /// Deserialize an `IndexEntry` from a byte array.
    ///
    /// # Safety
    ///
    /// `bytes` must be exactly `INDEX_ENTRY_SIZE` bytes. The function
    /// uses `ptr::read_unaligned` so the buffer does not need to be
    /// aligned to `IndexEntry`'s normal alignment. This is sound because
    /// `IndexEntry` is `#[repr(C)]`, is `Copy`, and contains only
    /// primitive types with no invalid bit patterns.
    pub fn from_bytes(bytes: &[u8; INDEX_ENTRY_SIZE]) -> Self {
        // SAFETY: `bytes` has the same size as `IndexEntry` and the type
        // is plain data. `read_unaligned` avoids alignment requirements
        // so this is safe regardless of the buffer's address.
        unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const IndexEntry) }
    }

    /// Returns `true` if this episode has been logically deleted.
    pub fn is_deleted(&self) -> bool {
        self.flags & Self::DELETED_FLAG != 0
    }

    /// Mark this episode as logically deleted.
    pub fn mark_deleted(&mut self) {
        self.flags |= Self::DELETED_FLAG;
    }

    /// Whether this episode is a meta-memory (association trace or
    /// dream insight) that should be excluded from retrieval results.
    pub fn is_meta_memory(&self) -> bool {
        self.flags & Self::META_MEMORY_FLAG != 0
    }

    /// Mark this episode as a meta-memory.
    pub fn mark_meta_memory(&mut self) {
        self.flags |= Self::META_MEMORY_FLAG;
    }

    /// Whether this episode has been archived by sleep compression.
    /// Archived episodes are excluded from similarity search and
    /// recent-episode scans but remain retrievable by ID.
    pub fn is_archived(&self) -> bool {
        self.flags & Self::ARCHIVED_FLAG != 0
    }

    /// Mark this episode as archived.
    pub fn mark_archived(&mut self) {
        self.flags |= Self::ARCHIVED_FLAG;
    }

    /// Whether this episode should be excluded from similarity search,
    /// recent-episode scans, and the bundle store. Deleted, archived,
    /// and meta-memory entries are all excluded.
    pub fn is_excluded_from_search(&self) -> bool {
        self.is_deleted() || self.is_archived() || self.is_meta_memory()
    }

    /// Hamming distance between this entry's association hash and
    /// `other_hash` (number of differing bits).
    pub fn hash_distance(&self, other_hash: u64) -> u32 {
        (self.association_hash ^ other_hash).count_ones()
    }
}

const _: () = {
    assert!(core::mem::size_of::<IndexEntry>() == INDEX_ENTRY_SIZE);
};

// ─────────────────────────────────────────────────────────────────
//  Metadata file header
// ─────────────────────────────────────────────────────────────────

/// Header for the metadata file (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IndexHeader {
    /// Magic bytes identifying the metadata file (`LTM_META_MAGIC`).
    pub magic: [u8; 4],
    /// Schema version of the metadata file.
    pub version: u32,
    /// Legacy capacity field — always `u32::MAX` in v2 (no fixed limit).
    pub capacity: u32,
    /// Number of episodes currently stored.
    pub episode_count: u32,
    /// Next episode ID to assign (monotonic).
    pub next_episode_id: u64,
    /// Reserved bytes for future use (zeroed).
    pub _reserved: [u8; 40],
}

impl IndexHeader {
    /// Create a fresh metadata header for the given (legacy) capacity.
    pub fn new(capacity: u32) -> Self {
        Self {
            magic: LTM_META_MAGIC,
            version: LTM_SCHEMA_VERSION,
            capacity,
            episode_count: 0,
            next_episode_id: 1,
            _reserved: [0; 40],
        }
    }

    /// Returns `true` if the magic bytes match `LTM_META_MAGIC`.
    pub fn verify_magic(&self) -> bool {
        self.magic == LTM_META_MAGIC
    }
}

const _: () = {
    assert!(core::mem::size_of::<IndexHeader>() == 64);
};

// ─────────────────────────────────────────────────────────────────
//  Data file header (same as v1)
// ─────────────────────────────────────────────────────────────────

/// Header for the LTM data file (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DataHeader {
    /// Magic bytes identifying the data file (`LTM_DATA_MAGIC`).
    pub magic: [u8; 4],
    /// Schema version of the data file.
    pub version: u32,
    /// Byte offset where the next payload append will be written.
    pub next_write_offset: u64,
}

impl DataHeader {
    /// Create a fresh data-file header pointing past the 16-byte header.
    pub fn new() -> Self {
        Self {
            magic: LTM_DATA_MAGIC,
            version: LTM_SCHEMA_VERSION,
            next_write_offset: 16,
        }
    }

    /// Returns `true` if the magic bytes match `LTM_DATA_MAGIC`.
    pub fn verify_magic(&self) -> bool {
        self.magic == LTM_DATA_MAGIC
    }
}

impl Default for DataHeader {
    fn default() -> Self {
        Self::new()
    }
}

const _: () = {
    assert!(core::mem::size_of::<DataHeader>() == 16);
};

// ─────────────────────────────────────────────────────────────────
//  SimHash — retained for backward compatibility with association engine
// ─────────────────────────────────────────────────────────────────

/// Compute a 64-bit SimHash fingerprint of the given text.
pub fn simhash(text: &str) -> u64 {
    let mut counts = [0i32; 64];
    let words: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();
    for word in &words {
        let h = fnv1a_lower(word.as_bytes());
        update_counts(&mut counts, h);
    }
    for i in 0..words.len().saturating_sub(1) {
        let h = fnv1a_lower_bigram(words[i].as_bytes(), words[i + 1].as_bytes());
        update_counts(&mut counts, h);
    }
    let mut result: u64 = 0;
    for (i, &count) in counts.iter().enumerate() {
        if count > 0 {
            result |= 1 << i;
        }
    }
    result
}

fn fnv1a_lower(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= byte.to_ascii_lowercase() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn fnv1a_lower_bigram(a: &[u8], b: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in a {
        hash ^= byte.to_ascii_lowercase() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= b' ' as u64;
    hash = hash.wrapping_mul(0x100000001b3);
    for &byte in b {
        hash ^= byte.to_ascii_lowercase() as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn update_counts(counts: &mut [i32; 64], hash: u64) {
    for (i, count) in counts.iter_mut().enumerate() {
        if hash & (1 << i) != 0 {
            *count += 1;
        } else {
            *count -= 1;
        }
    }
}

/// Hamming distance between two 64-bit hashes (number of differing bits).
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

// ─────────────────────────────────────────────────────────────────
//  Episode — the full in-memory representation (same as v1)
// ─────────────────────────────────────────────────────────────────

/// The full in-memory representation of a single episodic memory.
#[derive(Clone, Debug)]
pub struct Episode {
    /// Unique monotonic episode identifier.
    pub episode_id: u64,
    /// Unix timestamp (ms) when the episode was recorded.
    pub timestamp: u64,
    /// Emotional importance of the episode, in `[0,1]`.
    pub salience: f32,
    /// 64-bit SimHash fingerprint used for association matching.
    pub association_hash: u64,
    /// Compact emotional snapshot: arousal, valence, cortisol, dopamine.
    pub emotional_tag: [f32; 4],
    /// Full emotional tag across all `EMOTIONAL_TAG_SIZE` neurochemicals.
    pub full_emotional_tag: [f32; EMOTIONAL_TAG_SIZE],
    /// Event-type classifier (module-defined).
    pub event_type: u8,
    /// Identifier of the module that created the episode.
    pub source_module: u8,
    /// The episode's text payload.
    pub text: String,
}

// ─────────────────────────────────────────────────────────────────
//  Errors (same as v1)
// ─────────────────────────────────────────────────────────────────

/// Errors that can occur when working with the LTM store.
#[derive(Debug)]
pub enum LtmError {
    /// An underlying I/O error from the filesystem.
    Io(std::io::Error),
    /// The metadata file's magic bytes did not match.
    IndexBadMagic,
    /// The data file's magic bytes did not match.
    DataBadMagic,
    /// A file's schema version did not match the expected one.
    VersionMismatch {
        /// The version the code expected.
        expected: u32,
        /// The version found in the file.
        found: u32,
    },
    /// Memory-mapping a file failed.
    MmapFailed,
    /// `msync` failed while flushing mapped pages to disk.
    MsyncFailed,
    /// `ftruncate` failed while sizing a new file.
    FtruncateFailed,
    /// The (legacy) fixed-capacity index is full.
    IndexFull,
    /// No episode matched the requested ID.
    EpisodeNotFound,
    /// An episode payload exceeded the maximum allowed size.
    PayloadTooLarge {
        /// The actual payload size.
        size: usize,
        /// The maximum allowed size.
        max: usize,
    },
    /// Decompressing an episode payload failed (with a reason string).
    DecompressFailed(String),
    /// Compressing an episode payload failed (with a reason string).
    CompressFailed(String),
}

impl std::fmt::Display for LtmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::IndexBadMagic => write!(f, "metadata file bad magic"),
            Self::DataBadMagic => write!(f, "data file bad magic"),
            Self::VersionMismatch { expected, found } => {
                write!(f, "version mismatch: expected {expected}, found {found}")
            }
            Self::MmapFailed => write!(f, "mmap failed"),
            Self::MsyncFailed => write!(f, "msync failed"),
            Self::FtruncateFailed => write!(f, "ftruncate failed"),
            Self::IndexFull => write!(f, "index is full"),
            Self::EpisodeNotFound => write!(f, "episode not found"),
            Self::PayloadTooLarge { size, max } => {
                write!(f, "payload too large: {size} bytes (max {max})")
            }
            Self::DecompressFailed(e) => write!(f, "decompression failed: {e}"),
            Self::CompressFailed(e) => write!(f, "compression failed: {e}"),
        }
    }
}

impl std::error::Error for LtmError {}

// ─────────────────────────────────────────────────────────────────
//  LTM store — SDR/LogHD architecture
// ─────────────────────────────────────────────────────────────────

/// The LTM episodic store.
///
/// Uses a LogHD bundle store for SDR-based similarity matching and
/// an append-only metadata file for per-episode data. The data file
/// holds compressed payloads (same format as v1).
pub struct LtmStore {
    /// LogHD bundle store (mmap'd, fixed 800 KB)
    bundles: BundleStore,
    /// Metadata file (append-only)
    meta_file: std::fs::File,
    /// Data file (append-only, compressed payloads)
    data_file: std::fs::File,
    /// In-memory metadata entries (all active + deleted)
    entries: Vec<IndexEntry>,
    /// episode_id → index in entries vec
    id_to_idx: HashMap<u64, usize>,
    /// Active (non-deleted) episode IDs
    active_ids: Vec<u64>,
    /// Metadata header (kept in sync with file)
    meta_header: IndexHeader,
    /// Base path (without extension) for rebuilding the bundle store
    base_path: std::path::PathBuf,
}

unsafe impl Send for LtmStore {}
unsafe impl Sync for LtmStore {}

impl LtmStore {
    /// Create a new LTM store at the given base path.
    /// Creates `{base}.bundles`, `{base}.meta`, and `{base}.dat`.
    pub fn create(base_path: impl AsRef<Path>, _capacity: u32) -> Result<Self, LtmError> {
        let base = base_path.as_ref().to_path_buf();
        let bundles_path = base.with_extension("bundles");
        let meta_path = base.with_extension("meta");
        let dat_path = base.with_extension("dat");

        // Create bundle store
        let bundles = BundleStore::create(&bundles_path).map_err(LtmError::Io)?;

        // Create metadata file
        let mut meta_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&meta_path)
            .map_err(LtmError::Io)?;

        let meta_header = IndexHeader::new(DEFAULT_INDEX_CAPACITY);
        meta_file.seek(SeekFrom::Start(0)).map_err(LtmError::Io)?;
        // SAFETY: `IndexHeader` is `#[repr(C)]`, `Copy`, contains only
        // primitive types, and `size_of::<IndexHeader>() == 64`
        // (statically asserted). The slice covers exactly the struct's
        // in-memory representation.
        let meta_header_bytes = unsafe {
            std::slice::from_raw_parts(
                &meta_header as *const IndexHeader as *const u8,
                core::mem::size_of::<IndexHeader>(),
            )
        };
        meta_file
            .write_all(meta_header_bytes)
            .map_err(LtmError::Io)?;
        meta_file.sync_all().map_err(LtmError::Io)?;

        // Create data file
        let mut data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&dat_path)
            .map_err(LtmError::Io)?;

        let data_header = DataHeader::new();
        data_file.seek(SeekFrom::Start(0)).map_err(LtmError::Io)?;
        // SAFETY: `DataHeader` is `#[repr(C)]`, `Copy`, contains only
        // primitive types, and `size_of::<DataHeader>() == 16`
        // (statically asserted). The slice covers exactly the struct's
        // in-memory representation.
        let data_header_bytes = unsafe {
            std::slice::from_raw_parts(
                &data_header as *const DataHeader as *const u8,
                core::mem::size_of::<DataHeader>(),
            )
        };
        data_file
            .write_all(data_header_bytes)
            .map_err(LtmError::Io)?;
        data_file.sync_all().map_err(LtmError::Io)?;

        Ok(Self {
            bundles,
            meta_file,
            data_file,
            entries: Vec::new(),
            id_to_idx: HashMap::new(),
            active_ids: Vec::new(),
            meta_header,
            base_path: base,
        })
    }

    /// Open an existing LTM store.
    pub fn open(base_path: impl AsRef<Path>) -> Result<Self, LtmError> {
        let base = base_path.as_ref().to_path_buf();
        let bundles_path = base.with_extension("bundles");
        let meta_path = base.with_extension("meta");
        let dat_path = base.with_extension("dat");

        // Open bundle store
        let mut bundles = BundleStore::open(&bundles_path).map_err(LtmError::Io)?;

        // Open metadata file
        let mut meta_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&meta_path)
            .map_err(LtmError::Io)?;

        let meta_len = meta_file.metadata().map_err(LtmError::Io)?.len();
        if meta_len < 64 || (meta_len - 64) % INDEX_ENTRY_SIZE as u64 != 0 {
            return Err(LtmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LTM metadata contains an incomplete header or entry",
            )));
        }

        // Read metadata header
        let mut magic = [0u8; 4];
        let mut version_bytes = [0u8; 4];
        let mut capacity_bytes = [0u8; 4];
        let mut count_bytes = [0u8; 4];
        let mut next_id_bytes = [0u8; 8];
        let mut reserved = [0u8; 40];
        meta_file.seek(SeekFrom::Start(0)).map_err(LtmError::Io)?;
        meta_file.read_exact(&mut magic).map_err(LtmError::Io)?;
        meta_file
            .read_exact(&mut version_bytes)
            .map_err(LtmError::Io)?;
        meta_file
            .read_exact(&mut capacity_bytes)
            .map_err(LtmError::Io)?;
        meta_file
            .read_exact(&mut count_bytes)
            .map_err(LtmError::Io)?;
        meta_file
            .read_exact(&mut next_id_bytes)
            .map_err(LtmError::Io)?;
        meta_file.read_exact(&mut reserved).map_err(LtmError::Io)?;

        if magic != LTM_META_MAGIC {
            return Err(LtmError::IndexBadMagic);
        }
        let version = u32::from_le_bytes(version_bytes);
        if version != LTM_SCHEMA_VERSION {
            return Err(LtmError::VersionMismatch {
                expected: LTM_SCHEMA_VERSION,
                found: version,
            });
        }

        let meta_header = IndexHeader {
            magic,
            version,
            capacity: u32::from_le_bytes(capacity_bytes),
            episode_count: u32::from_le_bytes(count_bytes),
            next_episode_id: u64::from_le_bytes(next_id_bytes),
            _reserved: reserved,
        };

        // Read all metadata entries in one bulk read. The per-entry
        // `read_exact` loop was one syscall per entry (320K syscalls
        // at 320K entries); a single bulk read is 1 syscall plus
        // in-memory parsing. Pre-allocate all collections with the
        // exact entry count to avoid reallocations.
        let entry_bytes_len = meta_len as usize - 64;
        let entry_count = entry_bytes_len / INDEX_ENTRY_SIZE;
        let mut all_entry_bytes = vec![0u8; entry_bytes_len];
        meta_file
            .read_exact(&mut all_entry_bytes)
            .map_err(LtmError::Io)?;

        let mut entries: Vec<IndexEntry> = Vec::with_capacity(entry_count);
        let mut id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(entry_count);
        let mut active_ids: Vec<u64> = Vec::with_capacity(entry_count);

        for i in 0..entry_count {
            let offset = i * INDEX_ENTRY_SIZE;
            // SAFETY: `offset` is `i * INDEX_ENTRY_SIZE` where
            // `i < entry_count` and `entry_bytes_len =
            // entry_count * INDEX_ENTRY_SIZE`, so
            // `offset + INDEX_ENTRY_SIZE <= entry_bytes_len`.
            // The slice is exactly `INDEX_ENTRY_SIZE` bytes.
            let entry_bytes: &[u8; INDEX_ENTRY_SIZE] = all_entry_bytes
                [offset..offset + INDEX_ENTRY_SIZE]
                .try_into()
                .expect("entry slice length equals INDEX_ENTRY_SIZE");
            let entry = IndexEntry::from_bytes(entry_bytes);
            let idx = entries.len();
            if !entry.is_deleted() {
                // Archived episodes stay in id_to_idx (retrievable
                // by ID) but are excluded from active_ids (similarity
                // search, recent_episodes, bundle store).
                id_to_idx.insert(entry.episode_id, idx);
                if !entry.is_archived() {
                    active_ids.push(entry.episode_id);
                }
            }
            entries.push(entry);
        }

        // Open data file
        let mut data_file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&dat_path)
            .map_err(LtmError::Io)?;

        // Verify data header
        let mut dmagic = [0u8; 4];
        let mut dversion = [0u8; 4];
        data_file.seek(SeekFrom::Start(0)).map_err(LtmError::Io)?;
        data_file.read_exact(&mut dmagic).map_err(LtmError::Io)?;
        data_file.read_exact(&mut dversion).map_err(LtmError::Io)?;
        if dmagic != LTM_DATA_MAGIC {
            return Err(LtmError::DataBadMagic);
        }
        let dver = u32::from_le_bytes(dversion);
        if dver != LTM_SCHEMA_VERSION {
            return Err(LtmError::VersionMismatch {
                expected: LTM_SCHEMA_VERSION,
                found: dver,
            });
        }

        // Validate next_write_offset: must be >= 16 (past the header)
        // and <= file size. A corrupted or zeroed offset could cause
        // store() to seek to offset 0 and overwrite the header, or
        // to seek beyond EOF into a sparse file.
        let mut offset_bytes = [0u8; 8];
        data_file
            .read_exact(&mut offset_bytes)
            .map_err(LtmError::Io)?;
        let next_write_offset = u64::from_le_bytes(offset_bytes);
        let data_file_len = data_file.metadata().map(|m| m.len()).unwrap_or(0);
        if next_write_offset < 16 || (data_file_len > 0 && next_write_offset > data_file_len) {
            return Err(LtmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "LTM data file has invalid next_write_offset: {} (file size: {})",
                    next_write_offset, data_file_len
                ),
            )));
        }

        // Reconcile in-memory state with disk. The metadata header's
        // episode_count and next_episode_id may be stale if a prior
        // process crashed between writing entries and updating the
        // header. Recompute from the actual loaded entries and the
        // bundle store's next_episode_id to ensure consistency.
        let active_count = active_ids.len() as u32;
        let max_loaded_id = entries.iter().map(|e| e.episode_id).max().unwrap_or(0);
        let bundle_next = bundles.next_episode_id();
        let reconciled_next = max_loaded_id
            .saturating_add(1)
            .max(bundle_next)
            .max(meta_header.next_episode_id);
        bundles.ensure_next_episode_id(reconciled_next);
        bundles.sync().map_err(LtmError::Io)?;

        // Migrate existing meta-memories: mark entries from internal
        // bookkeeping modules with the META_MEMORY_FLAG. These are
        // association traces, dream insights, autobiographical traces,
        // prediction-mismatch records, and error-monitor records —
        // internal cognitive events that should not appear in
        // find_similar results.
        //
        // Source modules marked as meta-memory:
        //   0 = Subcognitive (association traces, dream insights)
        //   5 = Predictive coding ("Surprised by user input" records)
        //   8 = Error monitoring ("Detected error: prediction mismatch")
        //   9 = Dreaming (dream insights)
        //
        // Real content modules (NOT marked):
        //   3 = Autonomous learner (learned facts)
        //   4 = Language (conversation, thoughts, insights)
        //   6 = Visual cortex (visual scenes)
        //  10 = Semantic memory (extracted facts)
        let mut migrated_meta = 0u32;
        for (idx, entry) in entries.iter_mut().enumerate() {
            if !entry.is_meta_memory() && matches!(entry.source_module, 0 | 5 | 8 | 9) {
                entry.mark_meta_memory();
                // Write the updated flags byte back to the metadata file.
                let entry_file_offset = 64 + (idx as u64 * INDEX_ENTRY_SIZE as u64) + 60;
                meta_file
                    .seek(SeekFrom::Start(entry_file_offset))
                    .map_err(LtmError::Io)?;
                meta_file.write_all(&[entry.flags]).map_err(LtmError::Io)?;
                migrated_meta += 1;
            }
        }
        if migrated_meta > 0 {
            meta_file.sync_all().map_err(LtmError::Io)?;
            eprintln!(
                "[ltm] Marked {migrated_meta} existing episodes as meta-memories \
                 (excluded from retrieval)"
            );
        }

        let mut reconciled_store = Self {
            bundles,
            meta_file,
            data_file,
            entries,
            id_to_idx,
            active_ids,
            meta_header: IndexHeader {
                episode_count: active_count,
                next_episode_id: reconciled_next,
                ..meta_header
            },
            base_path: base,
        };
        // Write the reconciled header back to disk so the next open
        // sees consistent values.
        reconciled_store.write_meta_header()?;

        // If we migrated meta-memories, the bundle store is polluted
        // with their SDR contributions. Rebuild it from only real
        // episodes so find_similar returns accurate results.
        //
        // Also rebuild if the bundle store's episode_count is much
        // higher than the number of real (non-meta) episodes — this
        // happens when the migration already ran (flags were
        // persisted) but the bundle store was never rebuilt.
        let real_count = reconciled_store
            .active_ids
            .iter()
            .filter(|&&id| {
                let Some(&idx) = reconciled_store.id_to_idx.get(&id) else {
                    return false;
                };
                !reconciled_store.entries[idx].is_meta_memory()
            })
            .count();
        let bundle_count = reconciled_store.bundles.episode_count() as usize;
        let needs_rebuild = migrated_meta > 0 || (bundle_count > real_count * 2 && real_count > 0);
        if needs_rebuild {
            eprintln!(
                "[ltm] Bundle store has {bundle_count} episodes but only \
                 {real_count} are real — rebuilding to remove pollution"
            );
            reconciled_store.rebuild_bundles()?;
        }
        Ok(reconciled_store)
    }

    /// Open or create.
    pub fn open_or_create(base_path: impl AsRef<Path>, capacity: u32) -> Result<Self, LtmError> {
        let base = base_path.as_ref();
        let has_bundles = base.with_extension("bundles").exists();
        let has_meta = base.with_extension("meta").exists();
        let has_dat = base.with_extension("dat").exists();
        let has_old_idx = base.with_extension("idx").exists();

        // Migrate from v1 if old .idx exists but new files don't
        if has_old_idx && !has_meta && !has_bundles {
            return Self::migrate_from_v1(base, capacity);
        }

        if has_bundles && has_meta && has_dat {
            Self::open(base)
        } else if has_bundles || has_meta || has_dat || has_old_idx {
            Err(LtmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LTM store is incomplete; existing files were preserved",
            )))
        } else {
            Self::create(base, capacity)
        }
    }

    /// Migrate from v1 (.idx/.dat) to v2 (.bundles/.meta/.dat).
    fn migrate_from_v1(base: &Path, _capacity: u32) -> Result<Self, LtmError> {
        eprintln!("[ltm] Migrating from v1 to v2 (SDR/LogHD)...");
        let idx_path = base.with_extension("idx");
        let old_dat_path = base.with_extension("dat");

        // Open old store using v1 format — we need to read entries.
        // We'll read the old index directly since we can't import the
        // old code. The v1 format is: 64-byte header + 64-byte entries.
        let old_idx = std::fs::read(&idx_path).map_err(LtmError::Io)?;
        if old_idx.len() < 64 {
            return Err(LtmError::IndexBadMagic);
        }
        if &old_idx[0..4] != b"LTMI" {
            return Err(LtmError::IndexBadMagic);
        }

        let old_version = u32::from_le_bytes(old_idx[4..8].try_into().unwrap_or([0; 4]));
        if old_version != 1 {
            return Err(LtmError::VersionMismatch {
                expected: 1,
                found: old_version,
            });
        }
        let old_capacity = u32::from_le_bytes(old_idx[8..12].try_into().unwrap_or([0; 4])) as usize;
        if old_idx.len() != 64 + old_capacity * INDEX_ENTRY_SIZE {
            return Err(LtmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "v1 metadata length does not match its capacity",
            )));
        }
        let mut next_id = u64::from_le_bytes(old_idx[16..24].try_into().unwrap_or([0; 8])).max(1);
        let idx_backup = base.with_extension("idx.v1bak");
        let dat_backup = base.with_extension("dat.v1bak");
        if idx_backup.exists() || dat_backup.exists() {
            return Err(LtmError::Io(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "v1 migration backups already exist; recovery is required before retrying",
            )));
        }

        // Read the old data file BEFORE creating the new v2 store.
        // `Self::create()` truncates `{base}.dat`, so if we create
        // the new store first, the old data file is destroyed before
        // we can read it.
        let mut old_data = OpenOptions::new()
            .read(true)
            .open(&old_dat_path)
            .map_err(LtmError::Io)?;

        let mut data_header = [0u8; 16];
        old_data
            .read_exact(&mut data_header)
            .map_err(LtmError::Io)?;
        if data_header[..4] != LTM_DATA_MAGIC {
            return Err(LtmError::DataBadMagic);
        }
        let data_version = u32::from_le_bytes(data_header[4..8].try_into().unwrap_or([0; 4]));
        if data_version != 1 {
            return Err(LtmError::VersionMismatch {
                expected: 1,
                found: data_version,
            });
        }

        // Read all old entries and their payloads into memory before
        // the new store is created in a separate staging directory.
        struct OldEntry {
            episode_id: u64,
            timestamp: u64,
            salience: f32,
            emotional_tag: [f32; 4],
        }
        let mut old_entries: Vec<(OldEntry, Vec<u8>)> = Vec::new();
        for i in 0..old_capacity {
            let offset = 64 + i * INDEX_ENTRY_SIZE;
            let entry_bytes: &[u8; INDEX_ENTRY_SIZE] = old_idx[offset..offset + INDEX_ENTRY_SIZE]
                .try_into()
                .map_err(|_| {
                    LtmError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "v1 index entry slice length mismatch",
                    ))
                })?;
            let entry = IndexEntry::from_bytes(entry_bytes);
            next_id = next_id.max(entry.episode_id.saturating_add(1));
            if entry.episode_id == 0 || entry.is_deleted() {
                continue;
            }
            if entry.data_offset < 16 {
                return Err(LtmError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "v1 episode points into the data header",
                )));
            }

            // Read the compressed payload from the old data file
            old_data
                .seek(SeekFrom::Start(entry.data_offset))
                .map_err(LtmError::Io)?;
            let mut len_bytes = [0u8; 4];
            old_data.read_exact(&mut len_bytes).map_err(LtmError::Io)?;
            let compressed_len = u32::from_le_bytes(len_bytes) as usize;
            if compressed_len > MAX_COMPRESSED_SIZE
                || compressed_len != entry.compressed_len as usize
            {
                return Err(LtmError::DecompressFailed(
                    "invalid v1 compressed length".into(),
                ));
            }
            let mut compressed = vec![0u8; compressed_len];
            old_data.read_exact(&mut compressed).map_err(LtmError::Io)?;

            // Decompress to get the payload
            let payload = decompress(&compressed, entry.uncompressed_len as usize)?;
            if payload.len() < 52 {
                return Err(LtmError::DecompressFailed("v1 payload too short".into()));
            }

            old_entries.push((
                OldEntry {
                    episode_id: entry.episode_id,
                    timestamp: entry.timestamp,
                    salience: entry.salience,
                    emotional_tag: entry.emotional_tag,
                },
                payload,
            ));
        }

        old_entries.sort_unstable_by_key(|(entry, _)| entry.episode_id);
        if old_entries
            .windows(2)
            .any(|pair| pair[0].0.episode_id == pair[1].0.episode_id)
        {
            return Err(LtmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "v1 metadata contains duplicate episode IDs",
            )));
        }

        // Create the new v2 store separately without truncating .dat,
        // and keep the original files intact until it is fully durable.
        let stage_dir = base.with_extension(format!(
            "migration-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        ));
        std::fs::create_dir(&stage_dir).map_err(LtmError::Io)?;
        let result = (|| -> Result<Self, LtmError> {
            let stage_base = stage_dir.join("ltm");
            let mut new_store = Self::create(&stage_base, DEFAULT_INDEX_CAPACITY)?;
            let migrated = old_entries.len();
            for (old, payload) in old_entries {
                let event_type = payload[0];
                let source_module = payload[1];
                // Parse the 12 f32 emotional tag from bytes 4..52.
                // `try_into` to `[u8; 48]` is exact (48 bytes) and
                // avoids the partial-chunk risk of `chunks(4)`.
                let tag_bytes: [u8; 48] = payload[4..52]
                    .try_into()
                    .expect("payload[4..52] is exactly 48 bytes");
                let mut full_tag = [0.0f32; EMOTIONAL_TAG_SIZE];
                for (j, chunk) in tag_bytes.chunks_exact(4).enumerate() {
                    full_tag[j] = f32::from_le_bytes(
                        chunk
                            .try_into()
                            .expect("chunks_exact(4) yields 4-byte chunks"),
                    );
                }
                let text = String::from_utf8_lossy(&payload[52..]).into_owned();

                // Store in the new v2 format
                new_store.bundles.ensure_next_episode_id(old.episode_id);
                new_store.store(
                    old.timestamp,
                    old.salience,
                    full_tag,
                    old.emotional_tag,
                    event_type,
                    source_module,
                    &text,
                )?;
            }
            new_store.meta_header.next_episode_id = next_id;
            new_store.bundles.ensure_next_episode_id(next_id);
            new_store.write_meta_header()?;
            new_store.sync()?;
            let parent = base.parent().filter(|p| !p.as_os_str().is_empty());
            let directory =
                std::fs::File::open(parent.unwrap_or(Path::new("."))).map_err(LtmError::Io)?;
            std::fs::hard_link(&old_dat_path, &dat_backup).map_err(LtmError::Io)?;
            directory.sync_all().map_err(LtmError::Io)?;
            for ext in ["bundles", "dat", "meta"] {
                std::fs::rename(stage_base.with_extension(ext), base.with_extension(ext))
                    .map_err(LtmError::Io)?;
            }
            new_store.base_path = base.to_path_buf();
            eprintln!("[ltm] Migrated {migrated} episodes from v1 to v2");

            // Rename the old index as backup; the original data is already backed up.
            std::fs::rename(&idx_path, &idx_backup).map_err(LtmError::Io)?;
            directory.sync_all().map_err(LtmError::Io)?;
            Ok(new_store)
        })();
        let _ = std::fs::remove_dir_all(&stage_dir);
        result
    }

    // ─── Store operations ────────────────────────────────────────

    /// Store a new episode.
    #[allow(clippy::too_many_arguments)]
    pub fn store(
        &mut self,
        timestamp: u64,
        salience: f32,
        emotional_tag: [f32; EMOTIONAL_TAG_SIZE],
        compact_tag: [f32; 4],
        event_type: u8,
        source_module: u8,
        text: &str,
    ) -> Result<u64, LtmError> {
        let meta_len = self.meta_file.metadata().map_err(LtmError::Io)?.len();
        if meta_len != 64 + self.entries.len() as u64 * INDEX_ENTRY_SIZE as u64 {
            return Err(LtmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LTM metadata changed or has an incomplete entry; reopen before storing",
            )));
        }
        let next_episode_id = self
            .meta_header
            .next_episode_id
            .max(self.bundles.next_episode_id())
            .checked_add(1)
            .ok_or_else(|| {
                LtmError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "LTM episode ID space exhausted",
                ))
            })?;
        let text_bytes = text.as_bytes();
        // The full payload is 52 bytes (header + emotional tag) + text.
        // Check the total payload size, not just the text size, so that
        // a text of exactly MAX_PAYLOAD_SIZE doesn't produce a payload
        // that exceeds the limit by 52 bytes (which would then fail on
        // retrieval).
        if 52 + text_bytes.len() > MAX_PAYLOAD_SIZE {
            return Err(LtmError::PayloadTooLarge {
                size: 52 + text_bytes.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        // Sanitize emotional tag values to prevent NaN/inf from
        // corrupting the SDR encoding and downstream retrieval.
        let emotional_tag: [f32; EMOTIONAL_TAG_SIZE] = {
            let mut sanitized = [0.0f32; EMOTIONAL_TAG_SIZE];
            for (i, &v) in emotional_tag.iter().enumerate() {
                sanitized[i] = crate::state::sanitize::finite_clamp(v, 0.0, 1.0);
            }
            sanitized
        };

        // Build payload: [event_type][source_module][pad:2][tag:48][text...]
        let mut payload = Vec::with_capacity(52 + text_bytes.len());
        payload.push(event_type);
        payload.push(source_module);
        payload.push(0);
        payload.push(0);
        for &v in &emotional_tag {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        payload.extend_from_slice(text_bytes);

        let compressed = compress(&payload)?;

        // Read current data write offset
        self.data_file
            .seek(SeekFrom::Start(8))
            .map_err(LtmError::Io)?;
        let mut offset_bytes = [0u8; 8];
        self.data_file
            .read_exact(&mut offset_bytes)
            .map_err(LtmError::Io)?;
        let recorded_offset = u64::from_le_bytes(offset_bytes);
        let data_offset = self.data_file.metadata().map_err(LtmError::Io)?.len();

        // Validate the recorded offset against the header and EOF.
        // Always append at EOF, not at a potentially stale offset.
        // A crash can leave a payload durable before its offset update;
        // retaining those trailing bytes avoids overwriting episodes.
        // Offsets outside the existing file indicate corruption.
        if recorded_offset < 16 || recorded_offset > data_offset {
            return Err(LtmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("LTM data_offset {recorded_offset} is outside the data file"),
            )));
        }

        // Append compressed payload
        let compressed_len = compressed.len() as u32;
        self.data_file
            .seek(SeekFrom::Start(data_offset))
            .map_err(LtmError::Io)?;
        self.data_file
            .write_all(&compressed_len.to_le_bytes())
            .map_err(LtmError::Io)?;
        self.data_file
            .write_all(&compressed)
            .map_err(LtmError::Io)?;
        // fsync the payload BEFORE writing the offset that points to
        // it. A single fsync after both does not guarantee ordering:
        // the kernel may flush the offset (a small write near the
        // start of the file) before the much larger payload at the
        // end. If a crash occurs between the fsync and the next
        // operation, the offset could point to data that hasn't
        // reached disk yet, producing a corrupt or zero-filled read
        // on reopen.
        self.data_file.sync_all().map_err(LtmError::Io)?;

        // Update data file's next_write_offset, then fsync again to
        // make the offset durable. The payload is already on disk
        // from the sync above, so this fsync only needs to flush the
        // 8-byte offset update.
        let new_offset = data_offset + 4 + compressed.len() as u64;
        self.data_file
            .seek(SeekFrom::Start(8))
            .map_err(LtmError::Io)?;
        self.data_file
            .write_all(&new_offset.to_le_bytes())
            .map_err(LtmError::Io)?;
        self.data_file.sync_all().map_err(LtmError::Io)?;

        // Compute SimHash (backward compat for association engine)
        let assoc_hash = simhash(text);

        // Assign episode ID
        let episode_id = next_episode_id - 1;
        self.bundles.ensure_next_episode_id(episode_id);
        let code = episode_id_to_code(episode_id);

        // Encode SDR and add to bundle store
        let sdr = encode_episode(text, &emotional_tag, timestamp);
        self.bundles.add_episode(&code, &sdr);

        // Make the bundle store durable BEFORE writing the metadata
        // entry that references this episode ID. Without this, a crash
        // after `write_meta_header()` but before the next `sync()` would
        // leave the metadata file with episode N but the bundle file's
        // `next_episode_id` still at N-1. On reopen, `store()` would
        // reuse N-1, producing a duplicate episode ID that corrupts
        // `id_to_idx` and `active_ids`.
        self.bundles.sync().map_err(LtmError::Io)?;

        // Sanitize salience and compact_tag to prevent NaN/inf from
        // corrupting the index entry, SimHash scoring, and SDR encoding.
        let salience = crate::state::sanitize::finite_clamp(salience, 0.0, 1.0);
        let compact_tag = [
            crate::state::sanitize::finite_clamp(compact_tag[0], 0.0, 1.0),
            crate::state::sanitize::finite_clamp(compact_tag[1], -1.0, 1.0),
            crate::state::sanitize::finite_clamp(compact_tag[2], 0.0, 1.0),
            crate::state::sanitize::finite_clamp(compact_tag[3], 0.0, 1.0),
        ];

        // Build index entry
        let mut entry = IndexEntry::new(
            episode_id,
            data_offset,
            compressed_len,
            payload.len() as u32,
            timestamp,
            salience,
            assoc_hash,
            compact_tag,
            source_module,
        );

        // Automatically mark internal bookkeeping modules as
        // meta-memory so they don't pollute find_similar results.
        // This catches all stores regardless of which code path
        // produced them (Python store_event, Rust store_meta, etc.).
        // See the migration in `open` for the full module list.
        if matches!(source_module, 0 | 5 | 8 | 9) {
            entry.mark_meta_memory();
        }

        // Append to metadata file
        // SAFETY: `entry` is a valid local `IndexEntry` reference that lives
        // for the entire scope. `INDEX_ENTRY_SIZE` equals
        // `size_of::<IndexEntry>()`, so the slice covers exactly the struct's
        // in-memory representation with no out-of-bounds access.
        let entry_bytes = unsafe {
            std::slice::from_raw_parts(&entry as *const IndexEntry as *const u8, INDEX_ENTRY_SIZE)
        };
        self.meta_file
            .seek(SeekFrom::End(0))
            .map_err(LtmError::Io)?;
        self.meta_file
            .write_all(entry_bytes)
            .map_err(LtmError::Io)?;

        // Update metadata header in file. `write_meta_header` fsyncs
        // the metadata file, which also makes the entry appended
        // above durable — one fsync for both.
        self.meta_header.episode_count += 1;
        self.meta_header.next_episode_id = next_episode_id;
        self.write_meta_header()?;

        // Update in-memory indices
        let idx = self.entries.len();
        self.entries.push(entry);
        self.id_to_idx.insert(episode_id, idx);
        self.active_ids.push(episode_id);

        Ok(episode_id)
    }

    /// Store a meta-memory (association trace or dream insight).
    ///
    /// This is identical to `store` except it sets the
    /// `META_MEMORY_FLAG` on the index entry. Meta-memories are
    /// excluded from `find_similar` results so they don't flood
    /// retrieval with internal bookkeeping entries.
    ///
    /// `store()` already auto-marks source modules 0/5/8/9 as
    /// meta-memory. When `store_meta` is called with one of those
    /// modules (the common case — all current callers use 0 or 9),
    /// the flag is already set and we skip the redundant disk
    /// write + fsync. When called with a different module, we mark
    /// it and persist the flags byte.
    #[allow(clippy::too_many_arguments)]
    pub fn store_meta(
        &mut self,
        timestamp: u64,
        salience: f32,
        emotional_tag: [f32; EMOTIONAL_TAG_SIZE],
        compact_tag: [f32; 4],
        event_type: u8,
        source_module: u8,
        text: &str,
    ) -> Result<u64, LtmError> {
        let episode_id = self.store(
            timestamp,
            salience,
            emotional_tag,
            compact_tag,
            event_type,
            source_module,
            text,
        )?;
        // `store` appends to `entries` and `active_ids`, so the
        // newest entry is at the back.
        let idx = self.entries.len() - 1;

        // If `store()` already marked this entry as meta-memory
        // (source modules 0/5/8/9), the flags byte on disk already
        // matches — skip the redundant seek + write + fsync.
        if self.entries[idx].is_meta_memory() {
            return Ok(episode_id);
        }

        // Mark the entry and persist the updated flags byte.
        self.entries[idx].mark_meta_memory();

        // The IndexHeader is 64 bytes, each IndexEntry is 64 bytes,
        // and the flags field is at offset 60 within each entry.
        let entry_file_offset = 64 + (idx as u64 * INDEX_ENTRY_SIZE as u64) + 60;
        self.meta_file
            .seek(SeekFrom::Start(entry_file_offset))
            .map_err(LtmError::Io)?;
        self.meta_file
            .write_all(&[self.entries[idx].flags])
            .map_err(LtmError::Io)?;
        self.meta_file.sync_all().map_err(LtmError::Io)?;

        Ok(episode_id)
    }

    /// Write the metadata header back to the file.
    fn write_meta_header(&mut self) -> Result<(), LtmError> {
        self.meta_file
            .seek(SeekFrom::Start(0))
            .map_err(LtmError::Io)?;
        // SAFETY: `IndexHeader` is `#[repr(C)]`, `Copy`, and contains
        // only primitive types. `size_of::<IndexHeader>() == 64`
        // (statically asserted). The slice covers exactly the struct's
        // in-memory representation.
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &self.meta_header as *const IndexHeader as *const u8,
                core::mem::size_of::<IndexHeader>(),
            )
        };
        self.meta_file
            .write_all(header_bytes)
            .map_err(LtmError::Io)?;
        self.meta_file.sync_all().map_err(LtmError::Io)?;
        Ok(())
    }

    /// Retrieve an episode by ID.
    ///
    /// Archived episodes are retrievable (unlike deleted ones) — this
    /// is the key difference between archiving and deletion. Dream
    /// insights and association chains that reference archived episodes
    /// still resolve, preserving memory integrity across sleep
    /// compression cycles.
    pub fn retrieve(&mut self, episode_id: u64) -> Result<Episode, LtmError> {
        let idx = *self
            .id_to_idx
            .get(&episode_id)
            .ok_or(LtmError::EpisodeNotFound)?;
        let entry = self.entries[idx];

        // Guard against deleted entries. `id_to_idx` should not
        // contain deleted entries (delete removes them), but
        // defense-in-depth: if a bug or corruption leaves a stale
        // mapping, don't return a deleted episode.
        // Archived episodes ARE retrievable — that's the point.
        if entry.is_deleted() {
            return Err(LtmError::EpisodeNotFound);
        }

        // Validate entry.data_offset: must be >= 16 (past the data
        // file header). A corrupted index entry with data_offset < 16
        // would read the file header as if it were a compressed payload.
        if entry.data_offset < 16 {
            return Err(LtmError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "LTM entry {} has invalid data_offset {} (< 16)",
                    episode_id, entry.data_offset
                ),
            )));
        }

        self.data_file
            .seek(SeekFrom::Start(entry.data_offset))
            .map_err(LtmError::Io)?;
        let mut len_bytes = [0u8; 4];
        self.data_file
            .read_exact(&mut len_bytes)
            .map_err(LtmError::Io)?;
        let compressed_len = u32::from_le_bytes(len_bytes) as usize;

        if compressed_len > MAX_COMPRESSED_SIZE {
            return Err(LtmError::DecompressFailed(format!(
                "compressed length {compressed_len} exceeds maximum {MAX_COMPRESSED_SIZE}"
            )));
        }

        let mut compressed = vec![0u8; compressed_len];
        self.data_file
            .read_exact(&mut compressed)
            .map_err(LtmError::Io)?;

        // Cap uncompressed_len at MAX_PAYLOAD_SIZE to prevent OOM
        // from a corrupted index entry with a huge uncompressed_len.
        let unc_len = entry.uncompressed_len as usize;
        let payload = decompress(&compressed, unc_len)?;
        if payload.len() > MAX_PAYLOAD_SIZE {
            return Err(LtmError::DecompressFailed(format!(
                "decompressed length {} exceeds maximum {}",
                payload.len(),
                MAX_PAYLOAD_SIZE
            )));
        }
        // Verify the decompressed length matches the index entry's
        // uncompressed_len (capped). A mismatch indicates corruption.
        if payload.len() != unc_len {
            return Err(LtmError::DecompressFailed(format!(
                "decompressed length {} does not match expected {}",
                payload.len(),
                unc_len
            )));
        }

        if payload.len() < 52 {
            return Err(LtmError::DecompressFailed("payload too short".to_string()));
        }
        let event_type = payload[0];
        let source_module = payload[1];

        // Parse the 12 f32 emotional tag from bytes 4..52. The slice
        // is exactly 48 bytes (12 × 4), so `try_into` to `[u8; 48]`
        // is exact — no partial-chunk risk that `chunks(4)` would hide.
        let tag_bytes: [u8; 48] = payload[4..52]
            .try_into()
            .expect("payload[4..52] is exactly 48 bytes");
        let mut full_tag = [0.0f32; EMOTIONAL_TAG_SIZE];
        for (i, chunk) in tag_bytes.chunks_exact(4).enumerate() {
            full_tag[i] = f32::from_le_bytes(
                chunk
                    .try_into()
                    .expect("chunks_exact(4) yields 4-byte chunks"),
            );
        }

        let text = String::from_utf8_lossy(&payload[52..]).into_owned();

        Ok(Episode {
            episode_id: entry.episode_id,
            timestamp: entry.timestamp,
            salience: entry.salience,
            association_hash: entry.association_hash,
            emotional_tag: entry.emotional_tag,
            full_emotional_tag: full_tag,
            event_type,
            source_module,
            text,
        })
    }

    /// Find episodes similar to the given query text.
    ///
    /// Uses the LogHD bundle store for SDR-based similarity matching.
    /// Returns up to `limit` results sorted by distance (ascending).
    pub fn find_similar(&self, query: &str, limit: usize) -> Vec<(u64, u32, &IndexEntry)> {
        // Encode query as SDR (content only — no emotional/temporal
        // context for a text query)
        let query_sdr = crate::store::sdr::encode_content(query);
        let activation = self.bundles.query_activation(&query_sdr);

        // Score each active episode by the raw L2 distance
        // ||A − E||² between the query activation A and the episode's
        // expected activation E. Lower = more similar.
        //
        // We deliberately do NOT normalize the activation vector. The
        // LogHD bundle store is additive — each `add_episode` call
        // superimposes `weight * SDR` onto every bundle — so the
        // activation magnitude encodes how strongly the query overlaps
        // with each episode's content. Normalizing A to unit L2 norm
        // discards that magnitude information. When episode codes are
        // concentrated in a few of the 20 code dimensions (the common
        // case for small episode counts, since IDs 1–4 only use
        // positions 18–19), the expected activations are nearly
        // collinear and normalization makes them indistinguishable
        // regardless of their code digit values. The sort then
        // degenerates, returning wrong matches.
        //
        // Instead, we compute distances as f32 (which cannot overflow
        // regardless of episode count) and then scale to u32
        // adaptively — dividing by the maximum distance so the full
        // u32 range is used without saturation. This preserves
        // ordering for both small and large episode counts.
        let mut scored: Vec<(u64, f32, &IndexEntry)> = Vec::new();

        for &id in &self.active_ids {
            let Some(&idx) = self.id_to_idx.get(&id) else {
                continue;
            };
            let entry = &self.entries[idx];

            // Skip meta-memories (association traces, dream insights),
            // deleted, and archived entries. These are internal
            // bookkeeping episodes or excluded entries, not real
            // experiences. Including them in retrieval results floods
            // find_similar with `[association] episode X ↔ episode Y`
            // entries that crowd out actual conversation/learning
            // memories. `is_excluded_from_search()` is defense-in-depth:
            // active_ids should already exclude deleted/archived entries,
            // but a bug or corruption could leave a stale mapping.
            if entry.is_excluded_from_search() {
                continue;
            }

            let code = episode_id_to_code(id);
            let expected = code_to_expected_activation(&code);

            let mut dist_sq = 0.0f32;
            for j in 0..LOGHD_N {
                let diff = activation[j] - expected[j];
                dist_sq += diff * diff;
            }

            scored.push((id, dist_sq, entry));
        }

        // Sort by f32 distance (ascending) before converting to u32,
        // so the ordering is determined by full-precision distances.
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        // Adaptive scaling: map f32 distances to u32 preserving order.
        // Dividing by the max distance uses the full u32 range without
        // overflow, regardless of whether there are 4 episodes or 4
        // million.
        let max_dist = scored.iter().map(|(_, d, _)| *d).fold(0.0f32, f32::max);
        let scale = if max_dist > 0.0 && max_dist.is_finite() {
            (u32::MAX as f32) / max_dist
        } else {
            1.0
        };

        scored
            .into_iter()
            .map(|(id, dist, entry)| {
                let scaled = if dist.is_finite() {
                    (dist * scale).min(u32::MAX as f32) as u32
                } else {
                    u32::MAX
                };
                (id, scaled, entry)
            })
            .collect()
    }

    /// Find episodes similar to the given query text using SimHash.
    ///
    /// This is an O(n) linear scan that compares the query's SimHash
    /// fingerprint against every active (non-meta, non-deleted) episode's
    /// SimHash. Unlike `find_similar` (which uses the LogHD bundle store's
    /// superposition), this approach has no capacity limit and no
    /// superposition noise — it gives exact Hamming distances regardless
    /// of how many episodes are stored.
    ///
    /// Returns up to `limit` results sorted by Hamming distance (ascending).
    pub fn find_similar_hash(&self, query: &str, limit: usize) -> Vec<(u64, u32, &IndexEntry)> {
        let query_hash = simhash(query);

        let mut scored: Vec<(u64, u32, &IndexEntry)> = Vec::new();
        for &id in &self.active_ids {
            let Some(&idx) = self.id_to_idx.get(&id) else {
                continue;
            };
            let entry = &self.entries[idx];

            // Skip meta-memories, deleted, and archived entries.
            // (active_ids should already exclude deleted/archived,
            // but defense-in-depth.)
            if entry.is_excluded_from_search() {
                continue;
            }

            let dist = entry.hash_distance(query_hash);
            scored.push((id, dist, entry));
        }

        // Sort by Hamming distance (ascending) — lower is more similar.
        scored.sort_by_key(|a| a.1);
        scored.truncate(limit);
        scored
    }

    /// Find the highest-salience episodes.
    pub fn find_most_salient(&self, limit: usize) -> Vec<(u64, &IndexEntry)> {
        let mut results: Vec<(u64, &IndexEntry)> = Vec::with_capacity(self.active_ids.len());
        for &id in &self.active_ids {
            let Some(&idx) = self.id_to_idx.get(&id) else {
                continue;
            };
            let entry = &self.entries[idx];
            // Exclude meta-memories, deleted, and archived entries —
            // same contract as `find_similar` and `find_similar_hash`.
            if entry.is_excluded_from_search() {
                continue;
            }
            results.push((id, entry));
        }
        results.sort_by(|(_, a), (_, b)| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);
        results
    }

    /// Find episodes within a time range [start_ms, end_ms).
    pub fn find_by_time_range(&self, start_ms: u64, end_ms: u64) -> Vec<(u64, &IndexEntry)> {
        let mut results = Vec::new();
        for &id in &self.active_ids {
            let Some(&idx) = self.id_to_idx.get(&id) else {
                continue;
            };
            let entry = &self.entries[idx];
            // Exclude meta-memories, deleted, and archived entries —
            // same contract as `find_similar` and `find_similar_hash`.
            if entry.is_excluded_from_search() {
                continue;
            }
            if entry.timestamp >= start_ms && entry.timestamp < end_ms {
                results.push((id, entry));
            }
        }
        results.sort_by_key(|(_, e)| e.timestamp);
        results
    }

    /// Soft-delete an episode.
    pub fn delete(&mut self, episode_id: u64) -> Result<(), LtmError> {
        let idx = *self
            .id_to_idx
            .get(&episode_id)
            .ok_or(LtmError::EpisodeNotFound)?;

        self.entries[idx].mark_deleted();

        // Only the flags byte changed — write just that byte instead
        // of rewriting the full 64-byte entry. The flags field is at
        // offset 60 within each 64-byte IndexEntry.
        let flags_offset = 64 + (idx as u64 * INDEX_ENTRY_SIZE as u64) + 60;
        self.meta_file
            .seek(SeekFrom::Start(flags_offset))
            .map_err(LtmError::Io)?;
        self.meta_file
            .write_all(&[self.entries[idx].flags])
            .map_err(LtmError::Io)?;
        // Decrement episode_count and write the header (which fsyncs
        // the metadata file, covering the flags write above).
        self.meta_header.episode_count = self.meta_header.episode_count.saturating_sub(1);
        self.write_meta_header()?;

        // Only update in-memory indices after the durable write
        // succeeds. If `write_meta_header` failed above, the entry is
        // marked deleted in `entries` but still present in
        // `id_to_idx`/`active_ids` — the next `open()` will see the
        // on-disk deleted flag and exclude it from `active_ids`.
        self.id_to_idx.remove(&episode_id);
        self.active_ids.retain(|id| *id != episode_id);

        Ok(())
    }

    /// Archive an episode.
    ///
    /// Unlike `delete`, archiving preserves the episode in `id_to_idx`
    /// so it remains retrievable by ID. The episode is only removed
    /// from `active_ids`, which excludes it from similarity search,
    /// recent-episode scans, and the bundle store — but not from
    /// direct retrieval by ID.
    ///
    /// This is the correct operation for sleep compression: the
    /// episode's semantic content has been extracted into the concept
    /// network, so it no longer needs to be in the active similarity
    /// index. But the episode itself must survive — dream insights
    /// and association chains reference it by ID, and the conversation
    /// history is irreplaceable ground truth.
    pub fn archive(&mut self, episode_id: u64) -> Result<(), LtmError> {
        let idx = *self
            .id_to_idx
            .get(&episode_id)
            .ok_or(LtmError::EpisodeNotFound)?;

        // Already archived — nothing to do.
        if self.entries[idx].is_archived() {
            return Ok(());
        }

        self.entries[idx].mark_archived();

        // Only the flags byte changed — write just that byte instead
        // of rewriting the full 64-byte entry. The flags field is at
        // offset 60 within each 64-byte IndexEntry.
        let flags_offset = 64 + (idx as u64 * INDEX_ENTRY_SIZE as u64) + 60;
        self.meta_file
            .seek(SeekFrom::Start(flags_offset))
            .map_err(LtmError::Io)?;
        self.meta_file
            .write_all(&[self.entries[idx].flags])
            .map_err(LtmError::Io)?;
        // Decrement episode_count (active count) and fsync via header.
        self.meta_header.episode_count = self.meta_header.episode_count.saturating_sub(1);
        self.write_meta_header()?;

        // Remove from active_ids but keep in id_to_idx.
        self.active_ids.retain(|id| *id != episode_id);

        Ok(())
    }

    /// Number of active (non-deleted, non-archived) episodes.
    pub fn count(&self) -> u32 {
        self.meta_header.episode_count
    }

    /// Capacity (always u32::MAX in v2 — effectively unlimited).
    pub fn capacity(&self) -> u32 {
        self.meta_header.capacity
    }

    /// Get a reference to an index entry by slot (for compat).
    pub fn index_entry_ref(&self, slot: usize) -> &IndexEntry {
        &self.entries[slot]
    }

    /// Get the most recently stored episode IDs.
    pub fn recent_episode_ids(&self, limit: usize) -> Vec<u64> {
        // `active_ids` is always in ascending episode-ID order: `store`
        // appends monotonically increasing IDs and `delete` preserves
        // order. The most recent IDs are therefore the tail, so no
        // clone-and-sort of the whole list is needed.
        self.active_ids.iter().rev().take(limit).copied().collect()
    }

    /// Read source_module from a compressed payload header.
    ///
    /// After successfully reading the source_module, **backfills** it
    /// into the index entry (both in memory and on disk) so that future
    /// calls never need to decompress the same entry again. This is
    /// critical for performance: without backfilling, every call to
    /// `recent_episodes` with a source_module filter would decompress
    /// all old v2 entries (which have `source_module == 0` in the
    /// index) on every single call — potentially hundreds of thousands
    /// of decompressions per invocation.
    fn source_module_of(&mut self, idx: usize) -> Option<u8> {
        let entry = &self.entries[idx];
        let data_offset = entry.data_offset;
        // Validate data_offset before seeking — a corrupted entry with
        // data_offset < 16 would read the data file header as if it
        // were a compressed payload. `retrieve()` validates this too,
        // but `source_module_of` is called independently during
        // `recent_episodes` filtering.
        if data_offset < 16 {
            return None;
        }
        self.data_file.seek(SeekFrom::Start(data_offset)).ok()?;
        let mut len_bytes = [0u8; 4];
        self.data_file.read_exact(&mut len_bytes).ok()?;
        let compressed_len = u32::from_le_bytes(len_bytes) as usize;
        if compressed_len > MAX_COMPRESSED_SIZE {
            return None;
        }
        let mut compressed = vec![0u8; compressed_len];
        self.data_file.read_exact(&mut compressed).ok()?;
        // Use `take(2)` to read only the first 2 bytes of the
        // decompressed stream (event_type + source_module), avoiding
        // a full decompression just to read the source module byte.
        let decoder = ZlibDecoder::new(&compressed[..]);
        let mut header = [0u8; 2];
        decoder.take(2).read_exact(&mut header).ok()?;
        let module = header[1];

        // Backfill: write the discovered source_module back into the
        // index entry so future calls skip the decompression entirely.
        // The source_module field is at offset 61 within the 64-byte
        // IndexEntry. On disk, the entry starts at 64 + idx * 64.
        if module != 0 {
            let disk_offset = 64 + (idx as u64) * (INDEX_ENTRY_SIZE as u64) + 61;
            // Write to disk BEFORE updating the in-memory entry. If
            // seek fails, we must not write at the wrong position
            // (which could corrupt other index entries or the header).
            // If write fails, the in-memory entry must not diverge
            // from disk. Only update memory after the disk write
            // succeeds.
            if self.meta_file.seek(SeekFrom::Start(disk_offset)).is_ok()
                && self.meta_file.write_all(&[module]).is_ok()
            {
                self.entries[idx].source_module = module;
            }
        }
        Some(module)
    }

    /// Retrieve recent episodes, optionally filtered by source_module.
    pub fn recent_episodes(&mut self, limit: usize, source_filter: Option<u8>) -> Vec<Episode> {
        if limit == 0 {
            return Vec::new();
        }

        let candidate_ids: Vec<u64> = match source_filter {
            None => {
                // `active_ids` is always in ascending episode-ID order
                // (see `recent_episode_ids`), so the most recent IDs
                // are the tail. Iterate in reverse and take `limit`
                // — O(limit) instead of cloning and sorting the full
                // list on every call.
                self.active_ids.iter().rev().take(limit).copied().collect()
            }
            Some(target_module) => {
                // Iterate from most recent (tail of active_ids) to oldest.
                // This lets us stop early once we have enough matches,
                // rather than scanning all 250K+ entries every call.
                //
                // We use a manual index loop instead of cloning
                // active_ids into a reversed Vec. Each field access
                // (active_ids[i], id_to_idx.get, entries[idx]) copies
                // a Copy value (u64, usize, u8) and releases the
                // borrow before source_module_of(&mut self) is called,
                // so the borrow checker is satisfied without any
                // allocation.
                let mut matching: Vec<u64> = Vec::new();
                let mut i = self.active_ids.len();
                while i > 0 {
                    i -= 1;
                    let id = self.active_ids[i];
                    if let Some(&idx) = self.id_to_idx.get(&id) {
                        let module = self.entries[idx].source_module;
                        // Use the source_module field from the index entry.
                        // For old v2 entries where source_module is 0 (was
                        // padding), fall back to decompression + backfill.
                        let module = if module != 0 {
                            module
                        } else {
                            self.source_module_of(idx).unwrap_or(0)
                        };
                        if module == target_module {
                            matching.push(id);
                            if matching.len() >= limit {
                                break;
                            }
                        }
                    }
                }
                // matching is already in descending ID order (newest first)
                // because we iterated active_ids in reverse.
                matching
            }
        };

        let mut results = Vec::with_capacity(candidate_ids.len());
        for id in candidate_ids {
            if let Ok(ep) = self.retrieve(id) {
                results.push(ep);
            }
        }
        results
    }

    /// Get active episode IDs (for dreaming random selection).
    pub fn active_episode_ids(&self) -> &[u64] {
        &self.active_ids
    }

    /// Search episodes by offset — page through all active episodes
    /// in ascending episode ID order. Used by sleep compression to
    /// iterate over the full LTM for compaction.
    ///
    /// Returns up to `limit` episodes starting at `offset` in the
    /// active_ids list. Each result is a fully decompressed Episode
    /// (so the caller can extract concepts from the text).
    pub fn search_episodes(&mut self, limit: usize, offset: usize) -> Vec<Episode> {
        let ids = self.active_episode_ids();
        let start = offset.min(ids.len());
        let end = (start + limit).min(ids.len());
        // Collect IDs first to avoid borrowing self during the loop
        // (retrieve needs &mut self).
        let page: Vec<u64> = ids[start..end].to_vec();
        let mut results = Vec::with_capacity(page.len());
        for id in page {
            if let Ok(ep) = self.retrieve(id) {
                results.push(ep);
            }
        }
        results
    }

    /// Get the SimHash of an episode (for association engine).
    pub fn episode_hash(&self, episode_id: u64) -> Option<u64> {
        let idx = *self.id_to_idx.get(&episode_id)?;
        Some(self.entries[idx].association_hash)
    }

    /// Iterate over all active entries.
    pub fn active_entries(&self) -> impl Iterator<Item = (u64, &IndexEntry)> {
        self.active_ids.iter().filter_map(move |&id| {
            let idx = *self.id_to_idx.get(&id)?;
            Some((id, &self.entries[idx]))
        })
    }

    /// Sync everything to disk.
    ///
    /// This fsyncs the metadata file, data file, and bundle store.
    /// The order is: data file first (so the payload is durable),
    /// then meta file (so the index entries pointing to the payload
    /// are durable), then bundles (the SDR associative index).
    /// This ordering ensures that if a crash occurs between syncs,
    /// we never have an index entry pointing to a non-existent
    /// data record — the data is always at least as durable as
    /// the index that references it.
    pub fn sync(&mut self) -> Result<(), LtmError> {
        self.data_file.sync_all().map_err(LtmError::Io)?;
        self.meta_file.sync_all().map_err(LtmError::Io)?;
        self.bundles.sync().map_err(LtmError::Io)?;
        Ok(())
    }

    /// Rebuild the LogHD bundle store from only real (non-meta,
    /// non-deleted) episodes.
    ///
    /// Over time, the bundle store accumulates SDR contributions from
    /// every episode ever stored, including meta-memories (association
    /// traces, dream insights, autobiographical traces). These
    /// meta-memory SDRs pollute the bundle store's superposition,
    /// drowning out the signal from real episodes and degrading
    /// `find_similar` accuracy. With ~240K meta-memories out of ~292K
    /// total episodes, the bundle store is saturated — every query
    /// returns near-uniform activations.
    ///
    /// This method recreates the bundle file from scratch and re-adds
    /// only real episodes (those without the `META_MEMORY_FLAG` and
    /// without the `DELETED_FLAG`). The episode IDs and codes are
    /// preserved — only the bundle superposition is rebuilt.
    ///
    /// Returns the number of real episodes added to the rebuilt store.
    pub fn rebuild_bundles(&mut self) -> Result<u64, LtmError> {
        let bundles_path = self.base_path.with_extension("bundles");

        // Collect the text, emotional tag, and timestamp of every real
        // episode before we drop the old bundle store. We need to
        // re-encode each episode's SDR, which requires the full text
        // and emotional tag — not just the index entry.
        struct RealEpisode {
            episode_id: u64,
            timestamp: u64,
            text: String,
            full_tag: [f32; EMOTIONAL_TAG_SIZE],
        }

        let mut real_episodes: Vec<RealEpisode> = Vec::new();
        for id in self.active_ids.clone() {
            let Some(&idx) = self.id_to_idx.get(&id) else {
                continue;
            };
            let entry = self.entries[idx];

            // Skip meta-memories, deleted, and archived entries.
            if entry.is_excluded_from_search() {
                continue;
            }

            // Read the episode payload to get the text and full
            // emotional tag. The index entry only has the compact
            // 4-element tag; the full 12-element tag is in the payload.
            let episode = self.retrieve(id)?;
            real_episodes.push(RealEpisode {
                episode_id: id,
                timestamp: episode.timestamp,
                text: episode.text,
                full_tag: episode.full_emotional_tag,
            });
        }

        let real_count = real_episodes.len() as u64;
        eprintln!(
            "[ltm] Rebuilding bundle store: {real_count} real episodes \
             (excluded {} meta-memories and deleted entries)",
            self.active_ids.len().saturating_sub(real_episodes.len())
        );

        // Keep the old bundle store while building a replacement.
        // The Drop impl calls msync + munmap + close, so the old
        // mapping must never share a file that we truncate.
        let rebuild_path = self.base_path.with_extension(format!(
            "bundles.rebuild-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
        ));
        // Create an exclusive temporary store alongside the original
        // so a completed rebuild can be installed by atomic rename.
        let mut rebuilt = BundleStore::create_new(&rebuild_path).map_err(LtmError::Io)?;

        // Populate the fresh (zeroed) store without changing the original.
        let result = (|| -> Result<(), LtmError> {
            // Re-add each real episode's SDR to the rebuilt bundle store.
            // We use the same episode IDs and codes as before — only the
            // bundle superposition is rebuilt.
            for ep in &real_episodes {
                let code = episode_id_to_code(ep.episode_id);
                let sdr = encode_episode(&ep.text, &ep.full_tag, ep.timestamp);
                rebuilt.add_episode(&code, &sdr);
            }
            rebuilt.ensure_next_episode_id(
                self.meta_header
                    .next_episode_id
                    .max(self.bundles.next_episode_id()),
            );

            // Sync the rebuilt bundle store to disk.
            rebuilt.sync().map_err(LtmError::Io)?;
            std::fs::rename(&rebuild_path, &bundles_path).map_err(LtmError::Io)?;
            self.bundles = rebuilt;
            let parent = bundles_path.parent().filter(|p| !p.as_os_str().is_empty());
            std::fs::File::open(parent.unwrap_or(Path::new(".")))
                .and_then(|dir| dir.sync_all())
                .map_err(LtmError::Io)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&rebuild_path);
        }
        result?;

        eprintln!("[ltm] Bundle store rebuilt with {real_count} real episodes");
        Ok(real_count)
    }
}

// ─────────────────────────────────────────────────────────────────
//  Compression helpers (same as v1)
// ─────────────────────────────────────────────────────────────────

fn compress(data: &[u8]) -> Result<Vec<u8>, LtmError> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(data)
        .map_err(|e| LtmError::CompressFailed(e.to_string()))?;
    encoder
        .finish()
        .map_err(|e| LtmError::CompressFailed(e.to_string()))
}

fn decompress(data: &[u8], expected_len: usize) -> Result<Vec<u8>, LtmError> {
    // Cap expected_len at MAX_PAYLOAD_SIZE to prevent OOM from a
    // corrupted index entry with a huge uncompressed_len. The caller
    // already caps it, but defense-in-depth.
    let capped_len = expected_len.min(MAX_PAYLOAD_SIZE);
    let decoder = ZlibDecoder::new(data);
    // Use `take(capped_len)` to bound the decompressed output. Without
    // this, `read_to_end` would ignore `with_capacity` and decompress
    // a zlib bomb to many megabytes before the post-read length check
    // fires. `take` limits the total bytes read from the decoder to
    // `capped_len`, preventing unbounded allocation.
    let mut result = Vec::with_capacity(capped_len);
    decoder
        .take(capped_len as u64 + 1)
        .read_to_end(&mut result)
        .map_err(|e| LtmError::DecompressFailed(e.to_string()))?;
    if expected_len > MAX_PAYLOAD_SIZE || result.len() != expected_len {
        return Err(LtmError::DecompressFailed(format!(
            "decompressed length {} does not match expected {expected_len} within maximum {MAX_PAYLOAD_SIZE}",
            result.len(),
        )));
    }
    Ok(result)
}
