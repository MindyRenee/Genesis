//! Core state header — the first bytes of the memory-mapped state block.
//!
//! Contains identity, versioning, and the sequence lock that enables
//! lock-free concurrent reads from the mmap'd region.

/// Magic bytes identifying a Genesis core state file. ASCII `"GNSX"`.
pub const MAGIC: [u8; 4] = *b"GNSX";

/// Current schema version. Increment this when the layout of
/// [`GenesisCoreState`](super::core_state::GenesisCoreState) changes
/// in any way that breaks binary compatibility.
///
/// ## Version history
/// - **v1**: initial 12-chemical layout (never released)
/// - **v2**: 12-chemical coupled dynamics with receptor adaptation
/// - **v3**: 18-chemical layout — adds endocannabinoid, vasopressin,
///   CRH, orexin, epinephrine, melatonin. Coupling matrix expanded
///   from 12×12 to 18×18. Arousal/valence equations updated to
///   include the new chemicals. Bistable flip-flop with hysteresis.
///   Adenosine sleep pressure with configurable rates. Melatonin
///   excluded from homeostatic force (circadian-driven). GABA
///   disinhibition uses effective levels.
///
///   The struct size (3288 bytes) did not change from v2 to v3
///   because the v2 binary was already compiled with the full
///   18-chemical NeurochemicalVector — the v2 version label was
///   stale. Migration from v2 to v3 initializes the 6 new chemicals
///   at defaults (in case they were zeroed by an older binary) and
///   fills the new coupling matrix rows/columns.
pub const SCHEMA_VERSION: u32 = 3;

/// Header for the memory-mapped core state.
///
/// ## Layout (56 bytes)
/// ```text
/// offset  field          type   notes
/// ------  -----          ----   -----
///   0     magic          [u8;4] b"GNSX"
///   4     version        u32    schema version
///   8     state_size     u32    sizeof(GenesisCoreState)
///  12     _pad           [u8;4] implicit alignment padding to 8
///  16     created_at     u64    unix epoch milliseconds
///  24     last_updated   u64    unix epoch milliseconds
///  32     heartbeat      u64    monotonic counter, incremented each tick
///  40     seq_lock       u64    sequence lock for lock-free reads
///  48     instance_id    u64    unique per Genesis process instance
/// ```
///
/// ## Sequence lock
/// Writers increment `seq_lock` to an **odd** value before writing,
/// then to the next **even** value after writing. Readers sample
/// `seq_lock` before and after reading the state; if either sample is
/// odd (write in progress) or the two samples differ, the read is
/// retried. This gives us lock-free, wait-free reads with no atomics
/// overhead beyond the seq_lock field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CoreStateHeader {
    /// Magic bytes identifying the state file (`b"GNSX"`).
    pub magic: [u8; 4],
    /// Schema version of the core state.
    pub version: u32,
    /// Total byte size of the core state struct.
    pub state_size: u32,
    /// Explicit alignment padding to 8 bytes. Without this field the
    /// compiler inserts implicit padding that is not guaranteed to be
    /// zero-initialized, which would write stack garbage into the
    /// mmap'd state file. The checksum covers bytes [0..40) (magic
    /// through heartbeat) and [48..checksum_offset) (instance_id
    /// through manifest), so this field at offset 12 IS covered by
    /// the CRC. Only `seq_lock` at offset 40..48 is excluded.
    pub _pad: [u8; 4],
    /// Milliseconds-timestamp when the state was first created.
    pub created_at: u64,
    /// Milliseconds-timestamp of the last write.
    pub last_updated: u64,
    /// Monotonic heartbeat counter (incremented each tick).
    pub heartbeat: u64,
    /// Seqlock counter — odd during a write, even when stable.
    pub seq_lock: u64,
    /// Unique identifier for this state instance (process).
    pub instance_id: u64,
}

impl CoreStateHeader {
    /// Create a header initialised at the current schema version.
    pub fn new(instance_id: u64, state_size: u32, now_ms: u64) -> Self {
        Self {
            magic: MAGIC,
            version: SCHEMA_VERSION,
            state_size,
            _pad: [0; 4],
            created_at: now_ms,
            last_updated: now_ms,
            heartbeat: 0,
            seq_lock: 0, // even = no write in progress
            instance_id,
        }
    }

    /// Verify the magic bytes match.
    pub fn verify_magic(&self) -> bool {
        self.magic == MAGIC
    }
}

// --- Compile-time layout assertions ---
const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<CoreStateHeader>() == 56);
    assert!(offset_of!(CoreStateHeader, magic) == 0);
    assert!(offset_of!(CoreStateHeader, version) == 4);
    assert!(offset_of!(CoreStateHeader, state_size) == 8);
    assert!(offset_of!(CoreStateHeader, _pad) == 12);
    assert!(offset_of!(CoreStateHeader, created_at) == 16);
    assert!(offset_of!(CoreStateHeader, last_updated) == 24);
    assert!(offset_of!(CoreStateHeader, heartbeat) == 32);
    assert!(offset_of!(CoreStateHeader, seq_lock) == 40);
    assert!(offset_of!(CoreStateHeader, instance_id) == 48);
};
