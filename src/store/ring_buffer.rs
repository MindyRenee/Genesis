//! Short-term memory (STM) ring buffer.
//!
//! This is Genesis's "working memory" — a fixed-capacity ring buffer
//! of recent events that lives in its own mmap'd file. Events enter
//! the buffer as they happen, and the subcognitive daemon consolidates
//! important ones into long-term memory during idle periods.
//!
//! # What makes this different from a normal ring buffer
//!
//! Each entry carries an **emotional tag** — a snapshot of the
//! neurochemical state at the time the event was recorded. This
//! means the consolidation process can prioritise emotionally
//! significant events (high cortisol = threatening, high dopamine =
//! rewarding) without re-reading the full neurochemical history.
//!
//! The emotional tag also includes a **salience score** — a single
//! float that summarises how important this event is. It's computed
//! at insertion time from the neurochemical state, so the
//! consolidation process can simply sort by salience rather than
//! re-evaluating each event's importance.
//!
//! # Layout
//!
//! The file is a header followed by a fixed array of slots:
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │       RingBufferHeader          │  64 bytes
//! ├─────────────────────────────────┤
//! │  Slot 0: RingBufferEntry        │  256 bytes
//! │  Slot 1: RingBufferEntry        │  256 bytes
//! │  ...                            │
//! │  Slot N-1: RingBufferEntry      │  256 bytes
//! └─────────────────────────────────┘
//! ```

use std::os::unix::io::AsRawFd;
use std::path::Path;

use libc::{
    EEXIST, MAP_FAILED, MAP_SHARED, MS_SYNC, PROT_READ, PROT_WRITE, c_void, close, fdatasync,
    ftruncate, mmap, msync, munmap,
};

// ─────────────────────────────────────────────────────────────────
//  Constants
// ─────────────────────────────────────────────────────────────────

/// Magic bytes for STM files: ASCII `"STMB"`.
pub const STM_MAGIC: [u8; 4] = *b"STMB";

/// STM schema version.
pub const STM_SCHEMA_VERSION: u32 = 1;

/// Size of each ring buffer entry (the event record).
pub const ENTRY_SIZE: usize = 256;

/// Maximum size of the event's text payload (embedded in the entry).
/// 188 bytes of text + 68 bytes of metadata = 256 bytes per entry.
pub const ENTRY_TEXT_SIZE: usize = 188;

/// Number of neurochemical levels stored in the emotional tag.
/// We store the effective levels of the 12 chemicals.
pub const EMOTIONAL_TAG_SIZE: usize = 12;

// ─────────────────────────────────────────────────────────────────
//  Header
// ─────────────────────────────────────────────────────────────────

/// Header for the STM ring buffer file.
///
/// ## Layout (64 bytes)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RingBufferHeader {
    /// Magic bytes identifying the STM file (`STM_MAGIC`).
    pub magic: [u8; 4],
    /// Schema version of the ring buffer file.
    pub version: u32,
    /// Number of slots in the ring buffer.
    pub capacity: u32,
    /// Alignment padding.
    pub _pad0: u32,
    /// Next write index (wraps around at `capacity`).
    pub head: u64,
    /// Oldest unread index.
    pub tail: u64,
    /// Number of entries currently in the buffer.
    pub count: u64,
    /// Total entries ever written (monotonic, never wraps).
    pub total_written: u64,
    /// Reserved bytes for future expansion.
    pub _reserved: [u8; 16],
}

impl RingBufferHeader {
    /// Create a fresh, empty header for a buffer of the given capacity.
    pub fn new(capacity: u32) -> Self {
        Self {
            magic: STM_MAGIC,
            version: STM_SCHEMA_VERSION,
            capacity,
            _pad0: 0,
            head: 0,
            tail: 0,
            count: 0,
            total_written: 0,
            _reserved: [0; 16],
        }
    }

    /// Returns `true` if the magic bytes match `STM_MAGIC`.
    pub fn verify_magic(&self) -> bool {
        self.magic == STM_MAGIC
    }
}

const _: () = {
    assert!(core::mem::size_of::<RingBufferHeader>() == 64);
};

// ─────────────────────────────────────────────────────────────────
//  Entry
// ─────────────────────────────────────────────────────────────────

/// The type of event recorded in the ring buffer.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EventType {
    /// User interaction (message, command, etc.)
    UserInput = 0,
    /// Genesis's own output (response, action, etc.)
    Output = 1,
    /// Internal event (module state change, error, etc.)
    Internal = 2,
    /// Observation from the environment (file change, sensor, etc.)
    Observation = 3,
    /// A memory was recalled from long-term storage.
    MemoryRecall = 4,
    /// A new memory was consolidated to long-term storage.
    MemoryConsolidated = 5,
    /// An error occurred.
    Error = 6,
    /// A neurochemical shift event.
    NeurochemicalShift = 7,
}

impl EventType {
    /// Returns a short kebab-case label for this event type.
    pub const fn label(self) -> &'static str {
        match self {
            Self::UserInput => "user-input",
            Self::Output => "output",
            Self::Internal => "internal",
            Self::Observation => "observation",
            Self::MemoryRecall => "memory-recall",
            Self::MemoryConsolidated => "memory-consolidated",
            Self::Error => "error",
            Self::NeurochemicalShift => "neurochemical-shift",
        }
    }

    /// Convert a raw `u8` into an `EventType`, defaulting to
    /// [`Internal`](Self::Internal) for unknown values.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::UserInput,
            1 => Self::Output,
            2 => Self::Internal,
            3 => Self::Observation,
            4 => Self::MemoryRecall,
            5 => Self::MemoryConsolidated,
            6 => Self::Error,
            7 => Self::NeurochemicalShift,
            _ => Self::Internal,
        }
    }
}

/// A single entry in the STM ring buffer.
///
/// ## Layout (256 bytes)
/// ```text
/// offset  field              type       notes
/// ------  -----              ----       -----
///   0     timestamp          u64        ms since epoch
///   8     event_type         u8         EventType discriminant
///   9     source_module      u8         ModuleId that produced this
///  10     flags              u8         bit 0: consolidated to LTM
///  11     _pad               u8         alignment
///  12     salience           f32        computed importance [0,1]
///  16     emotional_tag      [f32;12]   effective neurochemical levels
///  64     text_len           u16        length of text payload
///  66     _pad2              [u8;2]     alignment
///  68     text               [u8;188]   event description (UTF-8)
/// ```
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RingBufferEntry {
    /// Milliseconds since the Unix epoch when the entry was recorded.
    pub timestamp: u64,
    /// `EventType` discriminant (stored as `u8`).
    pub event_type: u8,
    /// `ModuleId` of the module that produced this entry.
    pub source_module: u8,
    /// Bit 0: entry has been consolidated to LTM (don't re-consolidate).
    /// Other bits reserved for future use.
    pub flags: u8,
    /// Alignment padding.
    pub _pad: u8,
    /// Computed at insertion time from the neurochemical state.
    /// Higher = more important. Used by consolidation to prioritise.
    pub salience: f32,
    /// Snapshot of effective neurochemical levels at insertion time.
    /// Order matches [`NeurochemicalId::all()`].
    pub emotional_tag: [f32; EMOTIONAL_TAG_SIZE],
    /// Length of the UTF-8 text payload (bytes).
    pub text_len: u16,
    /// Alignment padding.
    pub _pad2: [u8; 2],
    /// Fixed-size UTF-8 text payload (truncated to `ENTRY_TEXT_SIZE`).
    pub text: [u8; ENTRY_TEXT_SIZE],
}

impl RingBufferEntry {
    /// Flag bit: entry has been consolidated to LTM.
    pub const CONSOLIDATED: u8 = 1 << 0;

    /// Create a new entry with the given text and emotional tag.
    pub fn new(
        timestamp: u64,
        event_type: EventType,
        source_module: u8,
        salience: f32,
        emotional_tag: [f32; EMOTIONAL_TAG_SIZE],
        text: &str,
    ) -> Self {
        let mut text_buf = [0u8; ENTRY_TEXT_SIZE];
        let bytes = text.as_bytes();
        // Truncate on a UTF-8 char boundary. Cutting mid-character
        // would make `text()`'s `from_utf8` fail and drop the entire
        // payload, not just the tail.
        let mut len = bytes.len().min(ENTRY_TEXT_SIZE);
        while len > 0 && !text.is_char_boundary(len) {
            len -= 1;
        }
        text_buf[..len].copy_from_slice(&bytes[..len]);

        Self {
            timestamp,
            event_type: event_type as u8,
            source_module,
            flags: 0,
            _pad: 0,
            salience: crate::state::sanitize::finite_clamp(salience, 0.0, 1.0),
            emotional_tag,
            text_len: len as u16,
            _pad2: [0; 2],
            text: text_buf,
        }
    }

    /// Whether this entry has been consolidated to LTM.
    pub fn is_consolidated(&self) -> bool {
        self.flags & Self::CONSOLIDATED != 0
    }

    /// Mark this entry as consolidated to LTM.
    pub fn mark_consolidated(&mut self) {
        self.flags |= Self::CONSOLIDATED;
    }

    /// Get the event type as the enum.
    pub fn event_type(&self) -> EventType {
        EventType::from_u8(self.event_type)
    }

    /// Get the text payload as a string (lossy UTF-8).
    pub fn text(&self) -> &str {
        let len = self.text_len as usize;
        if len > ENTRY_TEXT_SIZE {
            return "";
        }
        std::str::from_utf8(&self.text[..len]).unwrap_or("")
    }
}

const _: () = {
    assert!(core::mem::size_of::<RingBufferEntry>() == ENTRY_SIZE);
};

// ─────────────────────────────────────────────────────────────────
//  Ring buffer file
// ─────────────────────────────────────────────────────────────────

/// Errors for the STM ring buffer.
#[derive(Debug)]
pub enum RingBufferError {
    /// An underlying I/O error from the filesystem.
    Io(std::io::Error),
    /// The STM file was not found.
    NotFound,
    /// The file's magic bytes did not match.
    BadMagic,
    /// The file's schema version did not match the expected one.
    VersionMismatch {
        /// The version the code expected.
        expected: u32,
        /// The version found in the file.
        found: u32,
    },
    /// Memory-mapping the file failed.
    MmapFailed,
    /// `msync` failed while flushing mapped pages to disk.
    MsyncFailed,
    /// `fdatasync` failed while flushing the file descriptor.
    FsyncFailed,
    /// `ftruncate` failed while sizing the file.
    FtruncateFailed,
    /// The backing file is smaller than the required size.
    FileTooSmall {
        /// The expected file size.
        expected: u64,
        /// The actual file size found.
        found: u64,
    },
    /// The requested capacity was zero or otherwise invalid.
    InvalidCapacity {
        /// The invalid capacity value found.
        found: u32,
    },
    /// The requested capacity exceeds the maximum allowed.
    CapacityTooLarge {
        /// The capacity value found.
        found: u32,
        /// The maximum allowed capacity.
        max: u32,
    },
}

impl std::fmt::Display for RingBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::NotFound => write!(f, "STM file not found"),
            Self::BadMagic => write!(f, "bad magic bytes"),
            Self::VersionMismatch { expected, found } => {
                write!(f, "version mismatch: expected {expected}, found {found}")
            }
            Self::MmapFailed => write!(f, "mmap failed"),
            Self::MsyncFailed => write!(f, "msync failed"),
            Self::FsyncFailed => write!(f, "fdatasync failed"),
            Self::FtruncateFailed => write!(f, "ftruncate failed"),
            Self::FileTooSmall { expected, found } => {
                write!(f, "file too small: expected {expected}, found {found}")
            }
            Self::InvalidCapacity { found } => {
                write!(f, "invalid capacity: {found} (must be > 0)")
            }
            Self::CapacityTooLarge { found, max } => {
                write!(f, "capacity {found} exceeds maximum {max}")
            }
        }
    }
}

impl std::error::Error for RingBufferError {}

/// Compute the file size for a ring buffer with the given capacity.
///
/// This is the **logical** size — the actual data occupies exactly
/// `header_size + capacity * ENTRY_SIZE` bytes. The on-disk file is
/// page-aligned (see [`mmap_size`]) so the entire `mmap` mapping is
/// backed by the file.
pub fn file_size(capacity: u32) -> usize {
    core::mem::size_of::<RingBufferHeader>() + capacity as usize * ENTRY_SIZE
}

/// Compute the page-aligned mmap size for a ring buffer with the given
/// capacity.
///
/// `ftruncate` and `mmap` use this size so the entire mapping is
/// backed by the file, eliminating the risk of `SIGBUS` when accessing
/// the tail of the last page. Data access is still bounded by
/// [`file_size`] (the logical size).
pub fn mmap_size(capacity: u32) -> usize {
    crate::store::page_align_up(file_size(capacity))
}

/// A memory-mapped STM ring buffer.
///
/// Owns the mmap'd region and file descriptor. When dropped, unmaps
/// and closes.
pub struct RingBuffer {
    ptr: *mut u8,
    fd: i32,
    capacity: u32,
    file_len: usize,
    /// Coarse lock protecting all access to the mmap'd region.
    ///
    /// The ring buffer is shared between the IPC thread (which writes
    /// new events) and the daemon's reactive handler (which iterates
    /// and marks entries as consolidated). This mutex makes every
    /// public operation
    /// atomic with respect to other threads, removing the previous
    /// data race on the header and slot array.
    ///
    /// A poisoned lock means an earlier writer panicked and may have
    /// left the mmap in an inconsistent state. All public methods
    /// `.expect("access_lock poisoned: mmap may be inconsistent")` the guard and intentionally panic the daemon in
    /// that case — crashing is safer than continuing with a torn
    /// in-memory event log.
    access_lock: std::sync::Mutex<()>,
}

unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

/// Maximum capacity of the ring buffer. This prevents `usize` overflow
/// on 32-bit targets when computing `capacity * ENTRY_SIZE` (the mmap
/// region size). On 64-bit targets the limit is effectively unreachable
/// (16M entries × 256 bytes = 4 GB), but it also guards against
/// absurd user-supplied values that would trigger a huge allocation.
pub const MAX_CAPACITY: u32 = 16_777_216; // 2^24 — 4 GB at 256 bytes/entry

impl RingBuffer {
    /// Create a new ring buffer file with the given capacity.
    pub fn create(path: impl AsRef<Path>, capacity: u32) -> Result<Self, RingBufferError> {
        Self::create_inner(path, capacity, false)
    }

    /// Internal create with an `exclusive` flag. When `exclusive` is
    /// true, uses `create_new` (fails with `EEXIST` if the file already
    /// exists) instead of `create` + `truncate`. This is used by
    /// `open_or_create` to avoid the TOCTOU race of `exists()` + `create`.
    fn create_inner(
        path: impl AsRef<Path>,
        capacity: u32,
        exclusive: bool,
    ) -> Result<Self, RingBufferError> {
        if capacity == 0 {
            return Err(RingBufferError::InvalidCapacity { found: 0 });
        }
        if capacity > MAX_CAPACITY {
            return Err(RingBufferError::CapacityTooLarge {
                found: capacity,
                max: MAX_CAPACITY,
            });
        }
        let path = path.as_ref();
        let mmap_len = mmap_size(capacity);

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(!exclusive)
            .create_new(exclusive)
            .truncate(!exclusive)
            .open(path)
            .map_err(RingBufferError::Io)?;

        let fd = file.as_raw_fd();

        // Set the file size to the page-aligned mmap length so the
        // entire mapping is backed by the file. Without this, the tail
        // of the last page would be unbacked and touching it would
        // cause SIGBUS.
        // SAFETY: ftruncate is a POSIX call on a valid, owned fd. The fd is
        // obtained from `file.as_raw_fd()` and `file` is kept alive (forgotten
        // later) so the fd remains valid. `mmap_len` is the page-aligned
        // file size, bounded by the requested capacity.
        unsafe {
            if ftruncate(fd, mmap_len as i64) != 0 {
                return Err(RingBufferError::FtruncateFailed);
            }
        }
        // Persist the file size metadata so a crash doesn't leave a
        // zero-length or truncated file. `msync` only flushes data
        // pages, not the inode's i_size.
        // SAFETY: `fd` is a valid open file descriptor.
        unsafe {
            if fdatasync(fd) != 0 {
                return Err(RingBufferError::FsyncFailed);
            }
        }

        let ptr = Self::do_mmap(fd, mmap_len)?;
        std::mem::forget(file);

        // Initialise the header
        let header = RingBufferHeader::new(capacity);
        // SAFETY: `ptr` was just returned by mmap with PROT_WRITE.
        // The region is exclusively owned (no other references exist).
        unsafe {
            std::ptr::write_volatile(ptr as *mut RingBufferHeader, header);
        }

        // Zero out the entry slots
        // SAFETY: `entries_start` is within the mmap'd region
        // (header size + capacity * ENTRY_SIZE ≤ logical_len ≤ mmap_len).
        unsafe {
            let entries_start = ptr.add(core::mem::size_of::<RingBufferHeader>());
            std::ptr::write_bytes(entries_start, 0, capacity as usize * ENTRY_SIZE);
        }

        Self::do_msync(ptr, mmap_len)?;

        Ok(Self {
            ptr,
            fd,
            capacity,
            file_len: mmap_len,
            access_lock: std::sync::Mutex::new(()),
        })
    }

    /// Open an existing ring buffer file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, RingBufferError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(RingBufferError::NotFound);
        }

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(RingBufferError::Io)?;

        let fd = file.as_raw_fd();
        let metadata = file.metadata().map_err(RingBufferError::Io)?;

        // Read the header to get capacity
        if metadata.len() < core::mem::size_of::<RingBufferHeader>() as u64 {
            return Err(RingBufferError::FileTooSmall {
                expected: core::mem::size_of::<RingBufferHeader>() as u64,
                found: metadata.len(),
            });
        }

        let len = metadata.len() as usize;
        let ptr = Self::do_mmap(fd, len)?;
        std::mem::forget(file);

        // Verify header
        // SAFETY: `ptr` is a valid mmap'd region of `len` bytes,
        // len ≥ size_of::<RingBufferHeader>() (checked above).
        let header: &RingBufferHeader = unsafe { &*(ptr as *const RingBufferHeader) };
        if !header.verify_magic() {
            // SAFETY: `ptr` and `fd` are valid and exclusively owned.
            unsafe {
                munmap(ptr as *mut c_void, len);
                close(fd);
            }
            return Err(RingBufferError::BadMagic);
        }
        if header.version != STM_SCHEMA_VERSION {
            // SAFETY: `ptr` and `fd` are valid and exclusively owned.
            unsafe {
                munmap(ptr as *mut c_void, len);
                close(fd);
            }
            return Err(RingBufferError::VersionMismatch {
                expected: STM_SCHEMA_VERSION,
                found: header.version,
            });
        }
        if header.capacity == 0 {
            // SAFETY: `ptr` and `fd` are valid and exclusively owned.
            unsafe {
                munmap(ptr as *mut c_void, len);
                close(fd);
            }
            return Err(RingBufferError::InvalidCapacity { found: 0 });
        }
        if header.capacity > MAX_CAPACITY {
            // SAFETY: `ptr` and `fd` are valid and exclusively owned.
            unsafe {
                munmap(ptr as *mut c_void, len);
                close(fd);
            }
            return Err(RingBufferError::CapacityTooLarge {
                found: header.capacity,
                max: MAX_CAPACITY,
            });
        }

        // Verify the file is large enough to hold all `capacity` entries.
        // Without this, a truncated file could pass the header check but
        // cause out-of-bounds access when `push` or `iter` compute
        // `entry_ptr(index)` beyond the actual file length.
        let expected = file_size(header.capacity);
        if len < expected {
            // SAFETY: `ptr` and `fd` are valid and exclusively owned.
            unsafe {
                munmap(ptr as *mut c_void, len);
                close(fd);
            }
            return Err(RingBufferError::FileTooSmall {
                expected: expected as u64,
                found: len as u64,
            });
        }

        // Ensure the file is page-aligned so the entire mmap mapping is
        // backed by the file. Files created by older versions of this
        // code may have a non-page-aligned size. We extend the file to
        // the page-aligned mmap size and re-mmap if necessary.
        let needed_mmap_len = mmap_size(header.capacity);
        let (final_ptr, final_mmap_len) = if len < needed_mmap_len {
            // The file is smaller than the page-aligned size. We need
            // to unmap, ftruncate, fsync, and re-mmap.
            // SAFETY: `ptr` and `fd` are valid and exclusively owned.
            unsafe {
                munmap(ptr as *mut c_void, len);
            }
            // SAFETY: `fd` is a valid open file descriptor with write
            // permissions. ftruncate extends the file to
            // `needed_mmap_len`, zero-filling the new bytes.
            unsafe {
                if ftruncate(fd, needed_mmap_len as i64) != 0 {
                    close(fd);
                    return Err(RingBufferError::FtruncateFailed);
                }
            }
            // SAFETY: `fd` is a valid open file descriptor.
            unsafe {
                if fdatasync(fd) != 0 {
                    close(fd);
                    return Err(RingBufferError::FsyncFailed);
                }
            }
            // Re-mmap with the page-aligned size.
            let new_ptr = Self::do_mmap(fd, needed_mmap_len)?;
            (new_ptr, needed_mmap_len)
        } else {
            // File is already large enough (page-aligned or larger).
            // Use the existing mapping.
            (ptr, len)
        };

        Ok(Self {
            ptr: final_ptr,
            fd,
            capacity: header.capacity,
            file_len: final_mmap_len,
            access_lock: std::sync::Mutex::new(()),
        })
    }

    /// Open or create.
    ///
    /// Tries `create_new` first (which fails with `EEXIST` if the file
    /// already exists), then falls back to `open`. This avoids the
    /// TOCTOU race of checking `exists()` then calling `create()` —
    /// between the check and the create, another process could create
    /// the file, and `create` (with truncate) would destroy it.
    pub fn open_or_create(path: impl AsRef<Path>, capacity: u32) -> Result<Self, RingBufferError> {
        match Self::create_inner(path.as_ref(), capacity, true) {
            Ok(rb) => Ok(rb),
            Err(RingBufferError::Io(ref e)) if e.raw_os_error() == Some(EEXIST) => Self::open(path),
            Err(e) => Err(e),
        }
    }

    // ─── Access ──────────────────────────────────────────────────

    /// Raw pointer to the header. Caller must hold `access_lock`.
    fn header_ptr(&self) -> *mut RingBufferHeader {
        self.ptr as *mut RingBufferHeader
    }

    /// Raw pointer to the entry at `index`. Caller must hold `access_lock`.
    fn entry_ptr(&self, index: u64) -> *mut RingBufferEntry {
        let offset = core::mem::size_of::<RingBufferHeader>() + (index as usize) * ENTRY_SIZE;
        // SAFETY: `offset` is header size + index * ENTRY_SIZE, which is
        // within the mmap'd region when index < capacity (callers check).
        unsafe { self.ptr.add(offset) as *mut RingBufferEntry }
    }

    /// Copy of the current header.
    pub fn header(&self) -> RingBufferHeader {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        unsafe { *self.header_ptr() }
    }

    /// Read the entry at the given slot index (without removing).
    pub fn peek(&self, index: u64) -> Option<RingBufferEntry> {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        let header = unsafe { &*self.header_ptr() };
        if index >= header.capacity as u64 {
            return None;
        }
        // SAFETY: `index` < `header.capacity` (checked above), so
        // `entry_ptr(index)` points within the mmap'd entry slot array.
        Some(unsafe { *self.entry_ptr(index) })
    }

    /// Read the entry at the given slot index.
    pub fn entry_at(&self, index: u64) -> Option<RingBufferEntry> {
        self.peek(index)
    }

    /// Mark the entry at `index` as consolidated to LTM.
    ///
    /// **Warning:** This marks whatever entry *currently* occupies the
    /// slot, not necessarily the entry that was there when you snapshotted
    /// it. If a concurrent `push()` wrapped the ring buffer and overwrote
    /// the slot between your snapshot and this call, you would mark the
    /// *new* (unconsolidated) entry as consolidated, permanently preventing
    /// it from being promoted. Use [`mark_consolidated_if_match`] instead
    /// when the snapshot and the mark are separated by lock-releasing
    /// work (e.g. LTM store I/O).
    pub fn mark_consolidated(&self, index: u64) -> bool {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        let header = unsafe { &*self.header_ptr() };
        if index >= header.capacity as u64 {
            return false;
        }
        // SAFETY: `index` < `header.capacity` (checked above), so
        // `entry_ptr(index)` points within the mmap'd entry slot array.
        unsafe {
            let entry = &mut *self.entry_ptr(index);
            entry.flags |= RingBufferEntry::CONSOLIDATED;
        }
        true
    }

    /// Mark the entry at `index` as consolidated, but only if the entry
    /// currently occupying that slot is the same one we consolidated.
    ///
    /// This prevents the race where a concurrent `push()` wraps the ring
    /// buffer and overwrites the slot between the snapshot (from
    /// `iter_with_slots()`, which releases the lock) and this mark. The
    /// `expected_timestamp` identifies the entry we actually promoted to
    /// LTM; if the slot now holds a different entry, we leave it
    /// unconsolidated so it can be promoted on a future tick.
    ///
    /// Returns `true` if the entry was matched and marked, `false` if the
    /// slot was overwritten by a newer entry (race lost) or the index is
    /// out of bounds.
    pub fn mark_consolidated_if_match(&self, index: u64, expected_timestamp: u64) -> bool {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        let header = unsafe { &*self.header_ptr() };
        if index >= header.capacity as u64 {
            return false;
        }
        // SAFETY: `index` < `header.capacity` (checked above), so
        // `entry_ptr(index)` points within the mmap'd entry slot array.
        unsafe {
            let entry = &mut *self.entry_ptr(index);
            // If the slot was overwritten by a push() while we were
            // storing to LTM, the timestamp will differ. Don't mark
            // the new entry — let it be consolidated on a future tick.
            if entry.timestamp != expected_timestamp {
                return false;
            }
            entry.flags |= RingBufferEntry::CONSOLIDATED;
        }
        true
    }

    // ─── Ring buffer operations ──────────────────────────────────

    /// Push a new entry into the ring buffer.
    ///
    /// If the buffer is full, the oldest entry is overwritten.
    /// Returns the slot index where the entry was written.
    pub fn push(&self, entry: RingBufferEntry) -> u64 {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        let header = unsafe { &mut *self.header_ptr() };
        let slot = header.head;
        let cap = header.capacity as u64;

        // Guard against corrupted header with capacity == 0.
        // Return a sentinel (u64::MAX) instead of slot 0 so callers
        // cannot mistake the no-op for a successful write to slot 0.
        if cap == 0 {
            debug_assert!(false, "ring buffer capacity is 0 (corrupt header)");
            return u64::MAX;
        }

        // Write the entry
        // SAFETY: `slot` is `header.head` which is always < `cap`
        // (maintained by modular arithmetic on every push), so
        // `entry_ptr(slot)` points within the mmap'd entry slot array.
        // We use `write_volatile` to ensure the compiler does not
        // reorder or elide this store, which is critical for mmap
        // persistence — the entry must reach the page cache before
        // we update the header below.
        unsafe {
            std::ptr::write_volatile(self.entry_ptr(slot), entry);
        }

        // Release fence: ensures the entry write above is visible
        // before the header update below. On x86 (TSO) this is a
        // no-op, but on ARM/RISC-V the CPU could reorder the header
        // store before the entry store, leaving a crash-recovery
        // scenario where the header points to a slot that hasn't
        // been written yet.
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

        // Advance head
        header.head = (slot + 1) % cap;
        header.total_written = header.total_written.wrapping_add(1);

        // Update count and tail
        if header.count < cap {
            header.count += 1;
        } else {
            // Buffer is full — tail advances too (overwrite oldest)
            header.tail = (header.tail + 1) % cap;
        }

        slot
    }

    /// Iterate over entries in order from oldest to newest.
    ///
    /// Returns an owned snapshot so the lock is not held across the
    /// iteration. Each entry is a 256-byte `Copy` value.
    pub fn iter(&self) -> impl Iterator<Item = RingBufferEntry> {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        let header = unsafe { &*self.header_ptr() };
        let cap = header.capacity as u64;
        // Clamp count to capacity — a corrupted or truncated mmap header
        // could report a count exceeding capacity, causing a huge
        // allocation and iterating over garbage entries.
        let count = header.count.min(cap);
        let start = header.tail;
        let mut entries = Vec::with_capacity(count as usize);
        // Guard against corrupted header with capacity == 0.
        if cap > 0 {
            for i in 0..count {
                let index = (start + i) % cap;
                // SAFETY: `index` < `cap` (modular arithmetic), and `cap`
                // is `header.capacity`, so `entry_ptr(index)` is within the
                // mmap'd entry slot array.
                entries.push(unsafe { *self.entry_ptr(index) });
            }
        }
        entries.into_iter()
    }

    /// Snapshot all entries with their slot indices, holding the lock
    /// for the entire scan.
    ///
    /// This is the atomic variant of `iter()` for callers that need
    /// slot indices (e.g. consolidation, which marks entries
    /// consolidated by slot). Without this, calling `count()`,
    /// `header()`, `entry_at()`, and `mark_consolidated()` separately
    /// would acquire and release `access_lock` for each operation,
    /// allowing a concurrent `push()` to change `head`/`tail` and
    /// overwrite slots mid-scan.
    ///
    /// Returns `(slot_index, entry)` pairs in oldest-to-newest order.
    pub fn iter_with_slots(&self) -> Vec<(u64, RingBufferEntry)> {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        let header = unsafe { &*self.header_ptr() };
        let cap = header.capacity as u64;
        let count = header.count.min(cap);
        let start = header.tail;
        let mut entries = Vec::with_capacity(count as usize);
        if cap > 0 {
            for i in 0..count {
                let slot = (start + i) % cap;
                // SAFETY: `slot` < `cap`, so `entry_ptr(slot)` is within
                // the mmap'd entry slot array.
                entries.push((slot, unsafe { *self.entry_ptr(slot) }));
            }
        }
        entries
    }

    /// Number of entries currently in the buffer.
    pub fn count(&self) -> u64 {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        let header = unsafe { &*self.header_ptr() };
        header.count
    }

    /// Total entries ever written (monotonic counter).
    pub fn total_written(&self) -> u64 {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        let header = unsafe { &*self.header_ptr() };
        header.total_written
    }

    /// Capacity (max entries).
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Whether the buffer is full (new writes will overwrite oldest).
    pub fn is_full(&self) -> bool {
        self.count() == self.capacity as u64
    }

    /// Flush to disk.
    pub fn sync(&self) -> Result<(), RingBufferError> {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        Self::do_msync(self.ptr, self.file_len)
    }

    /// Clear all entries (reset head/tail/count, zero the slots).
    pub fn clear(&self) {
        let _guard = self
            .access_lock
            .lock()
            .expect("access_lock poisoned: mmap may be inconsistent");
        // SAFETY: `self.ptr` is a valid mmap'd region of at least
        // `size_of::<RingBufferHeader>()` bytes, validated on open/create.
        let header = unsafe { &mut *self.header_ptr() };
        header.head = 0;
        header.tail = 0;
        header.count = 0;
        // Don't reset total_written — it's monotonic

        // Zero the entries
        // SAFETY: `entries_start` is within the mmap'd region
        // (header size + capacity * ENTRY_SIZE ≤ file_len).
        unsafe {
            let entries_start = self.ptr.add(core::mem::size_of::<RingBufferHeader>());
            std::ptr::write_bytes(entries_start, 0, self.capacity as usize * ENTRY_SIZE);
        }
    }

    // ─── Internal ────────────────────────────────────────────────

    fn do_mmap(fd: i32, len: usize) -> Result<*mut u8, RingBufferError> {
        // SAFETY: `fd` is a valid open file descriptor with read+write
        // permissions. `len` is the file size (validated above).
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                len,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == MAP_FAILED {
            return Err(RingBufferError::MmapFailed);
        }
        Ok(ptr as *mut u8)
    }

    fn do_msync(ptr: *mut u8, len: usize) -> Result<(), RingBufferError> {
        // SAFETY: `ptr` is a valid mmap'd region of `len` bytes.
        let rc = unsafe { msync(ptr as *mut c_void, len, MS_SYNC) };
        if rc != 0 {
            return Err(RingBufferError::MsyncFailed);
        }
        Ok(())
    }
}

impl Drop for RingBuffer {
    fn drop(&mut self) {
        // Best-effort msync before munmap. Without this, any STM
        // events written since the last explicit sync() are lost
        // when the mapping is removed. We ignore errors here because
        // Drop cannot propagate them, and a failed msync is better
        // than skipping it entirely.
        let _ = Self::do_msync(self.ptr, self.file_len);
        // SAFETY: `self.ptr` and `self.fd` are valid and exclusively
        // owned by this instance. No other references exist at drop time.
        unsafe {
            munmap(self.ptr as *mut c_void, self.file_len);
            close(self.fd);
        }
    }
}
