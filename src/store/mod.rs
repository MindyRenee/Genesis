//! Storage layer — persistence for Genesis's state and memories.
//!
//! This module provides the memory-mapped file layer that backs the
//! core state, and (in future) the short-term and long-term memory
//! stores.
//!
//! ## Design principles
//!
//! - **Memory-mapped, not heap-allocated**: on a 4.7 GB RAM machine,
//!   we let the OS manage paging. State lives on disk, mapped into
//!   the process's address space.
//! - **Zero-copy reads**: readers access the mmap'd memory directly
//!   through the sequence lock protocol.
//! - **Crash-safe**: the state file has magic bytes, a schema version,
//!   and a CRC32 checksum. Corruption is detected on open.
//! - **No external services**: no database daemon, no server process.
//!   Just files on disk, mapped into memory.

pub mod ltm;
pub mod mmap_state;
pub mod ring_buffer;
pub mod sdr;

// ─────────────────────────────────────────────────────────────────
//  Page-alignment utility
// ─────────────────────────────────────────────────────────────────

/// Return the OS page size in bytes.
///
/// Uses `sysconf(_SC_PAGESIZE)` at runtime to support architectures
/// with non-4K pages (e.g., aarch64 with 64K pages). Falls back to
/// 4096 if `sysconf` fails, which is the minimum page size on any
/// Linux architecture and therefore always safe.
pub fn page_size() -> usize {
    // SAFETY: `sysconf` is async-signal-safe and has no side effects.
    // `_SC_PAGESIZE` is a valid configuration parameter.
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps > 0 { ps as usize } else { 4096 }
}

/// Round `len` up to the next multiple of the OS page size.
///
/// When `mmap` is called with a length that is not a multiple of the
/// page size, the kernel maps enough pages to cover the requested
/// length, but the tail of the last page (beyond `len`) is not backed
/// by the file. Accessing those tail bytes causes `SIGBUS`. By
/// `ftruncate`-ing the file to the page-aligned length and `mmap`-ing
/// the same, the entire mapping is backed by the file, eliminating
/// this hazard.
pub fn page_align_up(len: usize) -> usize {
    let ps = page_size();
    // If len is already aligned, return as-is. Otherwise round up.
    // This is `len.div_ceil(ps) * ps` but without the nightly
    // `div_ceil` method.
    let remainder = len % ps;
    if remainder == 0 {
        len
    } else {
        // Guard against overflow: `len + (ps - remainder)` can
        // overflow if `len` is near `usize::MAX`. In that case,
        // the rounded-up value would wrap to a small number,
        // causing `ftruncate` to create a tiny file and `mmap` to
        // produce a mapping that `SIGBUS`s on access. Return `len`
        // unchanged — it's already absurdly large and the caller
        // will likely fail on the subsequent `ftruncate`/`mmap`.
        match len.checked_add(ps - remainder) {
            Some(aligned) => aligned,
            None => len,
        }
    }
}

pub use ltm::{
    DEFAULT_INDEX_CAPACITY, DataHeader, EMOTIONAL_TAG_SIZE as LTM_EMOTIONAL_TAG_SIZE, Episode,
    INDEX_ENTRY_SIZE, IndexEntry, IndexHeader, LTM_DATA_MAGIC, LTM_META_MAGIC, LTM_SCHEMA_VERSION,
    LtmError, LtmStore, MAX_PAYLOAD_SIZE, hamming_distance, simhash,
};
pub use mmap_state::{MmapState, StateFileError};
pub use ring_buffer::{
    ENTRY_SIZE, ENTRY_TEXT_SIZE, EventType, MAX_CAPACITY, RingBuffer, RingBufferEntry,
    RingBufferError, RingBufferHeader, STM_MAGIC, STM_SCHEMA_VERSION, file_size, mmap_size,
};
pub use sdr::{
    BUNDLE_FILE_SIZE, BUNDLE_MAGIC, BUNDLE_SCHEMA_VERSION, BundleHeader, BundleStore, LOGHD_K,
    LOGHD_N, SDR_DIM, Sdr, bundle_mmap_size, code_digit, code_to_expected_activation,
    encode_content, encode_emotional, encode_episode, encode_temporal, episode_id_to_code,
};
