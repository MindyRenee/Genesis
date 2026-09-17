//! Genesis core state — the single memory-mapped structure that holds
//! Genesis's entire runtime state.
//!
//! This is the hub of the brain. Every module reads from and writes to
//! this structure. It is designed to be:
//!
//! - **Memory-mapped**: lives in a single contiguous file, mmap'd into
//!   all Genesis processes.
//! - **Fixed-size**: no heap allocations, no pointers, no variable-length
//!   data. The entire struct is `#[repr(C)]` with a stable binary layout.
//! - **Lock-free readable**: uses a sequence lock in the header so
//!   readers never block writers and vice versa.
//! - **Versioned**: magic bytes + schema version detect format mismatches.
//! - **Checksummed**: CRC32 detects corruption from crashes mid-write.
//!
//! ## Schema v3 layout (3288 bytes)
//!
//! ```text
//! offset  field             type                size
//! ------  -----             ----                ----
//!   0     header            CoreStateHeader      56
//!  56     neurochemicals    NeurochemicalVector 2416
//! 2472    zones             ActiveZones          96
//! 2568    memory            MemoryPointers      120
//! 2688    manifest          RuntimeManifest     536
//! 3224    checksum          u32                   4
//! 3228    inference_signals InferenceSignals      60
//! 3288    TOTAL                                   3288 bytes
//! ```
//!
//! That's ~3.3 KB — fits in a single 4 KB page with room to spare.
//!
//! ## What's new in v3
//!
//! - **6 new neurochemicals**: endocannabinoids (retrograde signaling),
//!   vasopressin (social bonding, HPA axis), CRH (upstream HPA trigger),
//!   orexin (hypocretin, arousal promotion), epinephrine (adrenaline,
//!   fast stress response), melatonin (circadian sleep hormone).
//!   Total: 18 chemicals (up from 12).
//! - **Receptor subtype modeling**: D1/D2 for dopamine, 5-HT1A/5-HT2A
//!   for serotonin, with subtype-specific sensitivities.
//! - **Split receptor adaptation**: desensitization (fast,
//!   phosphorylation-based) and internalization (slow, membrane removal)
//!   as separate mechanisms.
//! - **Allosteric modulation**: GABA-A allosteric site for
//!   benzodiazepine-like enhancement.
//! - **Region-specific serotonin→dopamine coupling**: striatum (5-HT2A,
//!   inhibitory) vs PFC (5-HT1A, facilitatory) vs default (average).
//! - **GABA disinhibition**: high GABA + high ACh disinhibits glutamate.
//! - **BDNF recovery after stress**: serotonin-driven BDNF recovery when
//!   cortisol drops.
//! - **Hard clamps**: cortisol max 0.80, NE max 0.90 (safety mechanism).
//! - **Customizable tick parameters**: NeuroTickParams struct with
//!   tick_with_params() method.
//! - **Increased restoring force**: 20% (up from 10%) for faster
//!   recovery from chronic stress.
//! - **Version migration**: migrate_state() framework for v2→v3 upgrades.
//!
//! ## What's new in v3.1
//!
//! - **HPA axis cascade**: CRH → ACTH → cortisol with time delays and
//!   negative feedback, replacing direct cortisol manipulation.
//! - **Tonic/phasic distinction**: tonic_level and phasic_level fields
//!   model sustained vs burst-firing neurotransmission. For dopamine,
//!   tonic activates D2 (high-affinity) and phasic activates D1
//!   (low-affinity). For serotonin, tonic activates 5-HT1A and phasic
//!   activates 5-HT2A.
//! - **Vesicular pool depletion**: impulses deplete the vesicular pool,
//!   causing synaptic depression when the pool is low. Synthesis and
//!   reuptake slowly replenish it.
//! - **Narcolepsy modeling**: low orexin (<0.10) causes sudden
//!   histamine drops, modeling cataplexy/sleep attacks.
//! - **Improved valence equation**: adds NE, endorphin, glutamate
//!   excess, and DA×SRT nonlinear synergy terms.
//! - **Improved arousal equation**: adds orexin and epinephrine,
//!   weights adenosine more heavily for sleep pressure, and models
//!   the bistable sleep-wake flip-flop with hysteresis.

use super::header::{CoreStateHeader, SCHEMA_VERSION};
use super::inference::InferenceSignals;
use super::manifest::RuntimeManifest;
use super::memory::MemoryPointers;
use super::neurochemical::{NeuroTickParams, NeurochemicalId, NeurochemicalVector};
use super::zones::{ActiveZones, CognitiveZone, MentalPhase};

use core::sync::atomic::{AtomicU64, Ordering, fence};

/// Reserved bytes at the end of the struct for future expansion
/// within schema v3 (does not require a version bump). The
/// [`InferenceSignals`] struct occupies this space — it is 60 bytes,
/// matching the previous reserved region exactly.
pub const RESERVED_BYTES: usize = 60;

/// CRC32 polynomial (IEEE 802.3 — same as zlib / PNG).
const CRC32_POLY: u32 = 0xEDB88320;

/// Pre-computed CRC32 table for fast lookup (8x faster than bit-by-bit).
/// Built once at startup from the polynomial.
static CRC32_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CRC32_POLY
            } else {
                crc >> 1
            };
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// The complete Genesis core state.
///
/// See the [module-level documentation](self) for the full layout.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct GenesisCoreState {
    /// Identity, versioning, and seqlock.
    pub header: CoreStateHeader,
    /// 18-chemical coupled neurochemical dynamics.
    pub neurochemicals: NeurochemicalVector,
    /// Active task zone, emergent phase, and subcognitive flags.
    pub zones: ActiveZones,
    /// Memory store pointers and emotional gating weights.
    pub memory: MemoryPointers,
    /// Runtime module manifest table.
    pub manifest: RuntimeManifest,
    /// CRC32 checksum of all preceding bytes (header.seq_lock excluded).
    pub checksum: u32,
    /// Active inference signals — the generative self-model's
    /// projection. Lives in the former reserved region (60 bytes).
    /// Not covered by the CRC32 checksum (derived state, recomputed
    /// every tick). See [`InferenceSignals`].
    pub inference_signals: InferenceSignals,
}

impl GenesisCoreState {
    /// Total size of the struct in bytes.
    pub const SIZE: usize = core::mem::size_of::<Self>();

    /// Create a fresh core state initialised at the current schema version.
    pub fn new(instance_id: u64, now_ms: u64) -> Self {
        let mut state = Self {
            header: CoreStateHeader::new(instance_id, Self::SIZE as u32, now_ms),
            neurochemicals: NeurochemicalVector::new(now_ms),
            zones: ActiveZones::new(now_ms),
            memory: MemoryPointers::new(),
            manifest: RuntimeManifest::new(),
            checksum: 0,
            inference_signals: InferenceSignals::new(),
        };
        // Sync the emergent phase and emotional gating from the
        // freshly initialised neurochemical system.
        state.sync_neurochemistry_to_state();
        state.checksum = state.compute_checksum();
        state
    }

    /// Propagate the neurochemical system's derived state to the
    /// zones (emergent phase) and memory pointers (gating weights).
    ///
    /// This must be called after every neurochemical tick to keep
    /// the rest of the state consistent with neurochemistry.
    pub fn sync_neurochemistry_to_state(&mut self) {
        // Sync emergent phase
        let neuro_phase = self.neurochemicals.phase();
        // When the cognitive mind has explicitly set the zone to
        // Sleeping (user called /sleep), force the emergent phase to
        // NREM (or REM if already in REM). Without this, the
        // neurochemical phase oscillates between "drowsy" and
        // "active" because adenosine hasn't risen high enough to
        // cross the sleep threshold — even though the mind is
        // cognitively asleep. The zone is the cognitive-layer
        // authority on sleep state; the neurochemical phase should
        // respect it.
        let phase = if self.zones.zone() == CognitiveZone::Sleeping {
            match neuro_phase {
                MentalPhase::REM => MentalPhase::REM,
                _ => MentalPhase::NREM,
            }
        } else {
            neuro_phase
        };
        self.zones.sync_phase(phase);

        // Sync emotional gating weights
        self.memory.update_gating(
            self.neurochemicals.effective(NeurochemicalId::Dopamine),
            self.neurochemicals.effective(NeurochemicalId::Serotonin),
            self.neurochemicals
                .effective(NeurochemicalId::Norepinephrine),
            self.neurochemicals
                .effective(NeurochemicalId::Acetylcholine),
            self.neurochemicals.effective(NeurochemicalId::Cortisol),
            self.neurochemicals.effective(NeurochemicalId::BDNF),
        );
    }

    /// Advance the neurochemical system by one tick and propagate
    /// the derived state to zones and memory.
    ///
    /// This is the main "heartbeat" of the affective system. It should
    /// be called by the dynamics engine at regular intervals (~100ms).
    pub fn neuro_tick(&mut self) {
        self.neurochemicals.tick_default();
        self.sync_neurochemistry_to_state();
    }

    /// Advance the neurochemical system with full custom params and
    /// propagate the derived state.
    pub fn neuro_tick_with_params(&mut self, params: &NeuroTickParams) {
        self.neurochemicals.tick_with_params(params);
        self.sync_neurochemistry_to_state();
    }

    // ─── Version migration ───────────────────────────────────────

    /// Migrate a state from an older schema version to the current one.
    ///
    /// This function provides the framework for upgrading state files
    /// when the schema changes. Currently, v2→v3 migration initializes
    /// the 6 new chemicals and fills the expanded coupling matrix.
    ///
    /// # Migration strategy
    ///
    /// When the schema version changes (e.g., v2 → v3):
    /// 1. Read the old state from disk
    /// 2. Create a new state at the current version
    /// 3. Copy over all fields that haven't changed
    /// 4. Initialize new fields with defaults
    /// 5. Recompute checksum
    ///
    /// # Parameters
    /// - `from_version`: the schema version of the source state
    ///
    /// # Returns
    /// - `Ok(())` if migration succeeded (or was a no-op)
    /// - `Err(CoreStateError)` if migration is not supported
    pub fn migrate_state(&mut self, from_version: u32) -> Result<(), CoreStateError> {
        match from_version {
            // v2 → v3: initialize the 6 new chemicals (endocannabinoid,
            // vasopressin, CRH, orexin, epinephrine, melatonin) at their
            // genetic defaults, fill the new coupling matrix rows/columns
            // with the default values, and recompute derived fields.
            //
            // The struct size is the same (3288 bytes) — the v2 binary
            // was already compiled with the full 18-chemical layout.
            // The 6 new chemicals may be zeroed (if written by an older
            // binary that didn't initialize them) or may have valid data
            // (if written by the current binary with a stale version
            // label). We initialize them unconditionally to ensure
            // consistency.
            2 => {
                let new_chemicals = [
                    NeurochemicalId::Endocannabinoid,
                    NeurochemicalId::Vasopressin,
                    NeurochemicalId::CRH,
                    NeurochemicalId::Orexin,
                    NeurochemicalId::Epinephrine,
                    NeurochemicalId::Melatonin,
                ];

                for &id in &new_chemicals {
                    // Use `get_mut_by_index` instead of `get_mut`: when
                    // an older binary zeroed the slot, `chem.id` is 0
                    // (Dopamine's discriminant), not the expected `id`.
                    // `get_mut`'s id-guard would return `None`, leaving
                    // the chemical uninitialised and mis-classified.
                    // `get_mut_by_index` trusts the array position,
                    // which is correct because `NeurochemicalId` is
                    // `#[repr(u8)]` with stable discriminants and the
                    // `chemicals` array is ordered by discriminant.
                    let idx = id as usize;
                    if let Some(chem) = self.neurochemicals.get_mut_by_index(idx) {
                        let default_baseline = id.default_baseline();
                        // Always restore the correct `id` field — it
                        // may be 0 (zeroed by an old binary) or already
                        // correct (written by current code). Setting it
                        // unconditionally is idempotent and ensures
                        // `get_mut` / `from_u8` work correctly after
                        // migration.
                        chem.id = id as u8;
                        // Only reinitialize the dynamic fields if the
                        // level is 0.0 (zeroed by an old binary). If
                        // the level is non-zero, the data was written
                        // by current code and we preserve it.
                        if chem.level == 0.0 && chem.baseline == 0.0 {
                            chem.level = default_baseline;
                            chem.baseline = default_baseline;
                            chem.tonic_level = default_baseline;
                            chem.receptor_sensitivity = id.default_sensitivity();
                            chem.receptor_subtypes = id.default_receptor_subtypes();
                            chem.desensitization_factor = 1.0;
                            chem.internalization_factor = 1.0;
                            chem.vesicular_pool = 1.0;
                            chem.velocity = 0.0;
                        }
                    }
                }

                // Only fill the new coupling matrix rows/columns when at
                // least one of the new chemicals looked uninitialized
                // (zeroed levels). If the state was written by the
                // current binary with a stale version label, the coupling
                // matrix already holds learned (metaplasticity-adapted)
                // values that must be preserved.
                let new_slots_uninit = new_chemicals.iter().any(|&id| {
                    let idx = id as usize;
                    // A zeroed slot has all-zero coupling row/col; a live
                    // slot preserves whatever the dynamics learned.
                    self.neurochemicals.coupling_matrix[idx]
                        .iter()
                        .all(|&v| v == 0.0)
                });
                if new_slots_uninit {
                    for i in 0..18 {
                        for j in 12..18 {
                            self.neurochemicals.coupling_matrix[i][j] =
                                crate::state::neurochemical::DEFAULT_COUPLING_MATRIX[i][j];
                        }
                    }
                    for i in 12..18 {
                        for j in 0..18 {
                            self.neurochemicals.coupling_matrix[i][j] =
                                crate::state::neurochemical::DEFAULT_COUPLING_MATRIX[i][j];
                        }
                    }
                }

                // Update version
                self.header.version = SCHEMA_VERSION;
                self.header.state_size = Self::SIZE as u32;

                // Recompute derived fields and checksum
                self.neurochemicals.recompute_derived();
                self.sync_neurochemistry_to_state();
                self.checksum = self.compute_checksum();

                Ok(())
            }
            // v3 → v3: no-op (same version)
            3 => {
                self.verify()?;
                Ok(())
            }
            // Unknown version: cannot migrate
            v => Err(CoreStateError::VersionMismatch {
                expected: SCHEMA_VERSION,
                found: v,
            }),
        }
    }

    // ─── Checksum ────────────────────────────────────────────────

    /// Compute the CRC32 checksum over the struct, excluding the
    /// `checksum` field itself and `header.seq_lock`.
    ///
    /// The checksum covers two regions:
    /// 1. offset 0..40 (magic through heartbeat)
    /// 2. offset 48..(SIZE - RESERVED_BYTES - 4) (instance_id through manifest end)
    ///
    /// `header.seq_lock` (offset 40..48) is excluded because it changes
    /// on every write, making the checksum unstable.
    /// `_reserved` (now `inference_signals`) and `checksum` at the
    /// end are excluded.
    pub fn compute_checksum(&self) -> u32 {
        let ptr = self as *const Self as *const u8;
        let size = Self::SIZE;
        let checksum_offset = size - RESERVED_BYTES - 4;

        let (region1, region2) = unsafe {
            // SAFETY: `ptr` is `self as *const Self as *const u8`, pointing to
            // a valid `GenesisCoreState` (repr(C), SIZE = 3288). The two
            // regions are within-bounds non-overlapping sub-slices:
            //   r1: offset 0..40  (40 bytes, before seq_lock)
            //   r2: offset 48..checksum_offset (checksum_offset ≤ SIZE - 64)
            // Both lengths are ≥ 0 and their end offsets are ≤ SIZE.
            // Region 1: offset 0..40 (magic through heartbeat, before seq_lock)
            let r1 = core::slice::from_raw_parts(ptr, 40);
            // Region 2: offset 48..checksum_offset (instance_id through manifest)
            let r2 = core::slice::from_raw_parts(ptr.add(48), checksum_offset - 48);
            (r1, r2)
        };

        let mut crc = 0xFFFFFFFFu32;
        for &byte in region1.iter().chain(region2.iter()) {
            // Table-driven CRC32 — 8x faster than bit-by-bit
            let idx = ((crc ^ byte as u32) & 0xFF) as usize;
            crc = (crc >> 8) ^ CRC32_TABLE[idx];
        }
        crc ^ 0xFFFFFFFFu32
    }

    // ─── Verification ────────────────────────────────────────────

    /// Verify the magic bytes and schema version.
    pub fn verify(&self) -> Result<(), CoreStateError> {
        if !self.header.verify_magic() {
            return Err(CoreStateError::BadMagic);
        }
        if self.header.version != SCHEMA_VERSION {
            return Err(CoreStateError::VersionMismatch {
                expected: SCHEMA_VERSION,
                found: self.header.version,
            });
        }
        if self.header.state_size != Self::SIZE as u32 {
            return Err(CoreStateError::SizeMismatch {
                expected: Self::SIZE as u32,
                found: self.header.state_size,
            });
        }
        Ok(())
    }

    /// Verify the checksum matches.
    pub fn verify_checksum(&self) -> Result<(), CoreStateError> {
        let computed = self.compute_checksum();
        if computed != self.checksum {
            return Err(CoreStateError::ChecksumMismatch {
                expected: computed,
                found: self.checksum,
            });
        }
        Ok(())
    }

    // ─── Sequence lock protocol ──────────────────────────────────

    /// Begin a write transaction. Sets `seq_lock` to odd (write in progress).
    ///
    /// # Safety
    /// The caller must ensure no other writer is active simultaneously.
    pub unsafe fn write_begin(&mut self, now_ms: u64) {
        // Atomically increment seq_lock to odd (write in progress).
        // We use AtomicU64::from_ptr rather than a plain field access
        // because the IPC thread may concurrently read seq_lock via
        // read_consistent. A plain load/store would be a data race
        // (undefined behavior) and the compiler could merge the
        // reader's two lock samples into one load, making the seqlock
        // tear-check vacuous.
        // SAFETY: `self.header.seq_lock` is a valid, properly aligned `u64`
        // field within the `#[repr(C)]` struct (header layout is verified by
        // compile-time asserts in header.rs). `addr_of_mut!` yields a raw
        // pointer without creating a temporary reference, satisfying
        // `AtomicU64::from_ptr`'s alignment and dereferenceability
        // requirements. The caller contract (no concurrent writers) is
        // documented in the function-level `# Safety` doc comment.
        let seq = unsafe { AtomicU64::from_ptr(core::ptr::addr_of_mut!(self.header.seq_lock)) };
        // AcqRel ordering on the fetch_add serves two purposes:
        //
        //   1. **Acquire** half: prevents the compiler/CPU from
        //      reordering the subsequent data writes (last_updated,
        //      heartbeat, and the closure's mutations) *above* the
        //      seqlock increment. Without this, on weakly-ordered
        //      architectures (ARM, RISC-V) a reader could sample an
        //      even `lock1`, observe partially-updated data, then
        //      sample the same even `lock2` and accept a torn state.
        //
        //   2. **Release** half: synchronizes-with the reader's Acquire
        //      load of seq_lock, establishing a happens-before
        //      relationship so that when a reader sees the odd
        //      (write-in-progress) value, it is guaranteed to see the
        //      effects of all writes prior to this write_begin.
        //
        // A plain Release fetch_add does NOT provide the Acquire half:
        // it only prevents *prior* stores from moving below the RMW,
        // not *subsequent* stores from moving above it. A Relaxed
        // fetch_add followed by a Release fence is equally insufficient
        // for the same reason — the fence constrains prior-vs-later
        // ordering but does not give the RMW itself acquire semantics.
        //
        // The Acquire fence below is belt-and-suspenders: it makes the
        // "no subsequent store may move above the RMW" guarantee
        // explicit and architecture-independent, rather than relying
        // solely on the AcqRel RMW's acquire component (which some
        // architectures implement as a no-op fence after the RMW).
        seq.fetch_add(1, Ordering::AcqRel);
        fence(Ordering::Acquire);
        self.header.last_updated = now_ms;
        self.header.heartbeat = self.header.heartbeat.wrapping_add(1);
    }

    /// End a write transaction. Recomputes checksum, sets `seq_lock`
    /// to even (write complete).
    pub fn write_end(&mut self) {
        self.checksum = self.compute_checksum();
        // Release fence ensures all data writes (including the
        // checksum store above) are globally visible before the
        // seq_lock increment that marks the write as complete.
        fence(Ordering::Release);
        // SAFETY: `self.header.seq_lock` is a properly aligned u64
        // (the header layout is verified by compile-time asserts in
        // header.rs, offset 40 with 8-byte alignment). We have
        // exclusive `&mut self` so no concurrent access here.
        let seq = unsafe { AtomicU64::from_ptr(core::ptr::addr_of_mut!(self.header.seq_lock)) };
        seq.fetch_add(1, Ordering::Release);
    }

    /// Attempt a lock-free read of the state.
    ///
    /// Uses the sequence lock protocol: samples `seq_lock` before and
    /// after copying the state. If either sample is odd (write in
    /// progress) or the samples differ, returns `None` and the caller
    /// should retry.
    ///
    /// **When to use this vs [`MmapState::read_consistent`]:**
    ///
    /// - This method is for **stack-allocated** `GenesisCoreState`
    ///   values (e.g. in unit tests) where there is no aliasing
    ///   concern — the caller owns the value exclusively.
    /// - For **mmap'd** state (the production daemon, IPC handler,
    ///   tick loop), use [`MmapState::read_consistent`] instead.
    ///   That method operates on the raw mmap pointer and never
    ///   creates a `&GenesisCoreState` reference to the mmap'd
    ///   memory, eliminating the aliasing UB that would occur if a
    ///   `&` from `MmapState::state()` coexisted with a `&mut` from
    ///   `MmapState::modify()` on another thread.
    ///
    /// # Safety
    /// The caller must ensure that `self` is not being deallocated or
    /// remapped while this function runs, and that no `&mut` reference
    /// to the same memory coexists with the `&self` used to call this
    /// method. For mmap'd state, use [`MmapState::read_consistent`]
    /// instead.
    pub unsafe fn read_consistent(&self) -> Option<GenesisCoreState> {
        // SAFETY: `AtomicU64::from_ptr` requires `*mut`, but atomic
        // loads do not mutate the pointed-to memory through the raw
        // pointer — they only read it atomically. We have `&self`, so
        // we cast away const for the API. No concurrent mutable alias
        // is created here; the writer goes through `&mut self` in
        // `write_begin`/`write_end`.
        let seq =
            unsafe { AtomicU64::from_ptr(core::ptr::addr_of!(self.header.seq_lock) as *mut u64) };

        // Acquire load: synchronizes-with the Release store in
        // write_end, ensuring we see all data writes from the
        // completed transaction that produced this even seq_lock.
        let lock1 = seq.load(Ordering::Acquire);
        if lock1 & 1 != 0 {
            return None;
        }
        // Acquire fence ensures the data reads below happen after the
        // lock1 load. Without this, on weak memory orderings (ARM,
        // RISC-V), the CPU could reorder the data reads before the
        // lock1 load, seeing stale data.
        fence(Ordering::Acquire);

        // SAFETY: `self` is a valid `GenesisCoreState` (caller guarantees it
        // is not being deallocated or remapped — see the function-level
        // `# Safety` doc). `read_volatile` copies `size_of::<Self>()` bytes
        // without creating a reference, avoiding aliasing issues with a
        // concurrent writer. The seqlock protocol (checked below) ensures
        // the copy is consistent.
        let copy = unsafe { core::ptr::read_volatile(self) };

        // Acquire fence ensures all data reads above are completed
        // before we sample lock2. Without this, on weak memory orderings
        // (ARM, RISC-V), the CPU could reorder the data reads after the
        // lock2 load, causing the seqlock validation to pass on stale
        // (torn) data. (On x86's TSO model this is harmless, but the
        // mmap'd state may be shared across architectures.)
        fence(Ordering::Acquire);
        // Relaxed is sufficient here: the fence above provides the
        // ordering we need. We only need an atomic load (not a plain
        // read) so the compiler doesn't merge this with lock1.
        let lock2 = seq.load(Ordering::Relaxed);

        if lock1 != lock2 || lock2 & 1 != 0 {
            return None;
        }
        Some(copy)
    }
}

/// Errors that can occur when verifying the core state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreStateError {
    /// The state's magic bytes did not match `b"GNSX"`.
    BadMagic,
    /// The state's schema version did not match the expected one.
    VersionMismatch {
        /// The version the code expected.
        expected: u32,
        /// The version found in the state.
        found: u32,
    },
    /// The state's byte size did not match the expected size.
    SizeMismatch {
        /// The size the code expected.
        expected: u32,
        /// The size found in the state.
        found: u32,
    },
    /// The state's checksum did not match the recomputed value.
    ChecksumMismatch {
        /// The checksum the code expected.
        expected: u32,
        /// The checksum found in the state.
        found: u32,
    },
}

impl core::fmt::Display for CoreStateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad magic bytes (expected b\"GNSX\")"),
            Self::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "schema version mismatch (expected {expected}, found {found})"
                )
            }
            Self::SizeMismatch { expected, found } => {
                write!(
                    f,
                    "state size mismatch (expected {expected}, found {found})"
                )
            }
            Self::ChecksumMismatch { expected, found } => {
                write!(
                    f,
                    "checksum mismatch (expected {expected:#010x}, found {found:#010x})"
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
//  Compile-time layout assertions
// ─────────────────────────────────────────────────────────────────

const _: () = {
    use core::mem::offset_of;
    assert!(core::mem::size_of::<GenesisCoreState>() == 3288);
    assert!(core::mem::size_of::<GenesisCoreState>() <= 4096);
    assert!(offset_of!(GenesisCoreState, header) == 0);
    assert!(offset_of!(GenesisCoreState, neurochemicals) == 56);
    assert!(offset_of!(GenesisCoreState, zones) == 2472);
    assert!(offset_of!(GenesisCoreState, memory) == 2568);
    assert!(offset_of!(GenesisCoreState, manifest) == 2688);
    assert!(offset_of!(GenesisCoreState, checksum) == 3224);
    assert!(offset_of!(GenesisCoreState, inference_signals) == 3228);
};
