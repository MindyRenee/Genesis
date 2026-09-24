//! Memory-mapped core state file.
//!
//! This is the persistence layer for [`GenesisCoreState`]. It handles:
//!
//! - **Creating** a new state file on disk (zeroed, then initialised)
//! - **Opening** an existing state file and mapping it into memory
//! - **Crash recovery**: detecting corruption via magic, version, and
//!   checksum, and recovering when possible
//! - **Lock-free reads**: readers get an owned copy of the state via
//!   the sequence lock protocol — no Rust reference to the mmap'd
//!   memory is ever created, so there is no aliasing with concurrent
//!   writers
//! - **Atomic writes**: writers use the seqlock protocol
//!   (write_begin → modify → write_end) which makes writes visible
//!   atomically to readers
//! - **Synchronisation**: `sync()` flushes dirty pages to disk so
//!   state survives a crash
//!
//! ## Why mmap instead of read/write?
//!
//! On a 4.7 GB RAM machine, we can't afford to hold copies of state
//! in heap memory. With mmap, the OS manages paging — the 3288-byte
//! state struct occupies a single page (4 KB), and the kernel keeps
//! it cached or pages it out as needed. We get a direct pointer to
//! the file's contents with zero copy.
//!
//! ## Memory safety: why no `&GenesisCoreState`?
//!
//! The mmap'd region is `MAP_SHARED` — it can be modified by another
//! thread (via [`modify`]) or another process at any time. Returning
//! a Rust `&GenesisCoreState` from the mmap'd memory would be
//! **undefined behaviour**: Rust's aliasing rules forbid a `&` from
//! coexisting with a `&mut` to the same bytes, even if the seqlock
//! prevents torn reads. The compiler is free to assume that a `&`
//! reference's pointee does not change, which could cause miscompilation
//! when another thread writes through `&mut`.
//!
//! Instead, the read API ([`read`] and [`read_consistent`]) performs
//! volatile byte copies through raw pointers, returning an **owned**
//! `GenesisCoreState` on the stack. No reference to the mmap'd memory
//! is ever created, so there is no aliasing. The write API ([`modify`])
//! holds an in-process `Mutex` that serialises writers, and the `&mut`
//! reference it creates is the sole reference within this process for
//! the duration of the closure.
//!
//! ## File layout
//!
//! ```text
//! offset 0:     GenesisCoreState (3288 bytes)
//! offset 3288:  (unused, padding to page boundary)
//! ```
//!
//! The file is exactly one page (4096 bytes) — the OS maps it as a
//! single page, and the state struct sits at the start.

use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use core::sync::atomic::{AtomicU64, Ordering, fence};

use libc::{
    LOCK_EX, LOCK_NB, LOCK_UN, MAP_FAILED, MAP_SHARED, MS_SYNC, PROT_READ, PROT_WRITE, c_void,
    close, fdatasync, flock, ftruncate, mmap, msync, munmap,
};

use crate::state::{CoreStateError, GenesisCoreState};

/// The logical file size — one 4K page. The state struct (3288 bytes)
/// fits comfortably, with the rest as padding.
const FILE_SIZE: usize = 4096;

/// The actual mmap/ftruncate length: `FILE_SIZE` rounded up to the OS
/// page size. On 4K-page machines this equals `FILE_SIZE`; on 16K/64K-page
/// machines (aarch64) it ensures the whole mapping is backed by the file
/// (otherwise the tail page is unbacked and faults with SIGBUS) and that
/// `munmap`/`msync` cover the full mapping.
fn mapped_len() -> usize {
    crate::store::page_align_up(FILE_SIZE)
}

/// Derive the initial circadian phase [0, 1) from the local time of
/// day encoded in `now_ms` (Unix epoch milliseconds).
///
/// Phase 0.0 = local midnight, matching the melatonin model's
/// wall-clock semantics ("Phase 0.0 (midnight): 1.0"). The oscillator
/// advances at exactly 24h per cycle, so its initial phase determines
/// its permanent offset from the real day/night cycle — anchoring it
/// to the wall clock at state birth keeps melatonin's night window
/// and the dopamine/histamine daytime elevation aligned with the
/// machine's local day.
///
/// Uses `localtime_r` so the rhythm follows the machine's timezone.
/// If the conversion is unavailable (`now_ms` out of `time_t` range,
/// or `localtime_r` fails), falls back to UTC — deterministic, exact
/// on UTC machines, and in any case anchored to the real day rather
/// than to file-creation time.
fn local_phase_of_day(now_ms: u64) -> f32 {
    let secs = now_ms / 1000;
    // UTC fallback: seconds of day modulo 86400.
    let utc_secs_of_day = (secs % 86_400) as i64;
    let secs_of_day = libc::time_t::try_from(secs)
        .ok()
        .and_then(|t| {
            let mut tm: libc::tm = unsafe { std::mem::zeroed() };
            // SAFETY: `t` is a valid `time_t` (range-checked above) and
            // `tm` is a properly sized, zero-initialised, caller-owned
            // struct. `localtime_r` (unlike `localtime`) is thread-safe:
            // it writes only into `tm` and returns null on failure,
            // leaving `tm` zeroed — handled by the `None` branch.
            let ok = unsafe { libc::localtime_r(&t, &mut tm) };
            if ok.is_null() {
                None
            } else {
                Some(
                    i64::from(tm.tm_hour) * 3600 + i64::from(tm.tm_min) * 60 + i64::from(tm.tm_sec),
                )
            }
        })
        .unwrap_or(utc_secs_of_day);
    (secs_of_day as f32 / 86_400.0).rem_euclid(1.0)
}

/// A memory-mapped Genesis core state file.
///
/// This owns the mmap'd region and the file descriptor. When dropped,
/// it unmaps the memory and closes the file.
///
/// # Concurrency model
///
/// The mmap'd region is `MAP_SHARED` and may be written to by another
/// thread in this process (via [`modify`]) or by another process at
/// any time. To avoid Rust aliasing undefined behaviour, the read API
/// ([`read`], [`read_consistent`]) **never** creates a `&GenesisCoreState`
/// to the mmap'd bytes. Instead, it performs volatile byte copies through
/// raw pointers, returning an owned `GenesisCoreState` on the caller's
/// stack.
///
/// Writers are serialised by an in-process `Mutex` ([`modify`]) and a
/// cross-process `flock(LOCK_EX)` advisory lock on the state file. The
/// `&mut GenesisCoreState` handed to the modify closure is the sole
/// Rust reference to the mmap'd memory within this process for the
/// duration of the closure — no reader creates a competing `&`. The
/// flock ensures that only one process can write at a time, preventing
/// two processes from simultaneously incrementing `seq_lock` and
/// publishing a torn state.
///
/// Cross-process reads remain lock-free — the seqlock protocol ensures
/// tear-free reads regardless of what any process does. Only writers
/// take the flock.
pub struct MmapState {
    /// Pointer to the mmap'd region.
    ptr: *mut u8,
    /// The file descriptor (kept open for msync).
    fd: i32,
    /// Whether we created this file (vs opening an existing one).
    created: bool,
    /// Coarse lock serializing writers.
    ///
    /// `MmapState` is shared between the daemon's reactive handler and
    /// the IPC thread via `Arc`. Both threads can call
    /// [`MmapState::modify`], which
    /// mutates the mmap'd state. The seqlock protects readers from
    /// seeing torn writes, but it does **not** serialize multiple
    /// writers. This mutex ensures only one writer proceeds at a
    /// time, preventing data races on `seq_lock`, checksums, and state
    /// fields. Reads via [`read`] and [`read_consistent`] remain
    /// lock-free.
    ///
    /// If the lock is poisoned, [`modify`] returns
    /// `StateFileError::LockFailed`. The daemon intentionally panics
    /// on this rather than risk writing to a potentially-torn shared
    /// state.
    write_lock: std::sync::Mutex<()>,
}

// SAFETY: `MmapState` can be sent between threads and shared by
// reference. The raw pointer and fd are not tied to a specific thread.
// The safety of concurrent access is ensured by:
//   - Reads: `read`/`read_consistent` use volatile byte copies through
//     raw pointers — no Rust references to the mmap'd memory are
//     created, so no aliasing with writers.
//   - Writes: `modify` acquires `write_lock` before creating the
//     `&mut GenesisCoreState`, ensuring no other thread in this
//     process has a competing reference.
unsafe impl Send for MmapState {}
unsafe impl Sync for MmapState {}

impl MmapState {
    /// Create a new state file at the given path, initialise it with
    /// a fresh `GenesisCoreState`, and map it into memory.
    ///
    /// **Fails if the file already exists.** This prevents accidental
    /// destruction of an existing brain state — a double-start or
    /// misuse of `create` (instead of `open_or_create`) would otherwise
    /// truncate the state file, wiping all neurochemical state, memory
    /// pointers, and runtime manifest. Use `open_or_create` if you want
    /// to open an existing file or create a new one.
    pub fn create(
        path: impl AsRef<Path>,
        instance_id: u64,
        now_ms: u64,
    ) -> Result<Self, StateFileError> {
        let path = path.as_ref();

        // Create the file — create_new fails with EEXIST if the file
        // already exists, preventing accidental truncation of an
        // existing state file.
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(StateFileError::Io)?;

        let fd = file.as_raw_fd();

        // Set the file size
        // SAFETY: ftruncate is a POSIX call on a valid, owned fd. The fd is
        // obtained from `file.as_raw_fd()` and `file` is kept alive (forgotten
        // later) so the fd remains valid. FILE_SIZE is a small constant.
        unsafe {
            if ftruncate(fd, mapped_len() as i64) != 0 {
                return Err(StateFileError::FtruncateFailed);
            }
        }
        // Persist the file size metadata to stable storage. Without
        // this, a crash after ftruncate but before the inode is
        // flushed can leave a zero-length file. `msync` only flushes
        // data pages, not the inode's i_size.
        Self::do_fsync(fd)?;

        // mmap the file
        let ptr = Self::do_mmap(fd)?;

        // Initialise the state
        let mut state = GenesisCoreState::new(instance_id, now_ms);
        // Anchor the circadian oscillator to the actual local time of
        // day (the system clock is the zeitgeber). The constructor's
        // constant phase is only a placeholder for direct construction
        // (tests, tools) — here, at the production birth path, the
        // phase must be derived from the wall clock. Without this the
        // oscillator's 24h cycle is phase-locked to file-creation time,
        // so melatonin's night window and the dopamine/histamine
        // daytime peaks drift permanently out of sync with the real
        // day. Existing state files keep their evolved phase — this
        // runs only on first creation.
        state
            .neurochemicals
            .set_circadian_phase(local_phase_of_day(now_ms));
        // circadian_phase is covered by the CRC32 (region 2 spans the
        // neurochemical vector), so the checksum computed inside the
        // constructor is now stale — recompute it, exactly as
        // `write_end` does after any mutation.
        state.checksum = state.compute_checksum();
        // SAFETY: `ptr` was just returned by mmap with PROT_WRITE and
        // FILE_SIZE ≥ size_of::<GenesisCoreState>(). The region is
        // exclusively owned (no other references exist yet).
        // write_volatile prevents the compiler from optimizing out
        // the store, ensuring the state is written to the mmap'd file.
        unsafe {
            std::ptr::write_volatile(ptr as *mut GenesisCoreState, state);
        }

        // Sync to disk. If this fails we must release the mapping
        // before returning, otherwise the caller has no way to do so
        // because `Self` has not been constructed yet.
        if let Err(e) = Self::do_msync(ptr, mapped_len()) {
            // SAFETY: `ptr` is a valid mmap'd region of FILE_SIZE bytes
            // that we own exclusively. The file is still owned by `file`,
            // which will close the fd when dropped at the end of scope.
            unsafe {
                munmap(ptr as *mut c_void, mapped_len());
            }
            return Err(e);
        }

        // All fallible steps succeeded. Transfer fd ownership to Self.
        std::mem::forget(file);

        Ok(Self {
            ptr,
            fd,
            created: true,
            write_lock: std::sync::Mutex::new(()),
        })
    }

    /// Open an existing state file, map it into memory, and verify
    /// its integrity.
    ///
    /// If the file doesn't exist, returns `StateFileError::NotFound`.
    /// If the file is corrupted (bad magic, version mismatch, or
    /// checksum failure), returns an error with details.
    ///
    /// Verification and migration are performed through volatile reads
    /// and raw pointers — no `&GenesisCoreState` reference to the
    /// mmap'd memory is created. This is safe even though the mmap
    /// is `MAP_SHARED` because no `MmapState` has been returned to
    /// the caller yet, so no other thread in this process can access
    /// the mapping.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StateFileError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(StateFileError::NotFound);
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(StateFileError::Io)?;

        let fd = file.as_raw_fd();

        // Serialize open-time verification and migration with writers
        // in other processes. A volatile snapshot is not coherent while
        // another process is between write_begin and write_end; holding
        // the same flock used by modify() makes the snapshot and any
        // migration atomic with respect to cross-process writes.
        // SAFETY: `fd` is a valid open file descriptor.
        unsafe { Self::lock_file(fd) }?;

        // Check file size — must be at least the state struct size.
        let metadata = match file.metadata() {
            Ok(metadata) => metadata,
            Err(e) => {
                // SAFETY: `fd` is valid and the open lock is held.
                unsafe { Self::unlock_file(fd) };
                return Err(StateFileError::Io(e));
            }
        };
        if metadata.len() < GenesisCoreState::SIZE as u64 {
            // SAFETY: `fd` is valid and the open lock is held.
            unsafe { Self::unlock_file(fd) };
            return Err(StateFileError::FileTooSmall {
                expected: GenesisCoreState::SIZE as u64,
                found: metadata.len(),
            });
        }

        // Ensure the file is at least the page-aligned mapping length so
        // the mmap region is fully backed by the file. Without this, a
        // file that is exactly 3288 bytes (state size) would be mapped
        // as a full page, leaving the tail unbacked — SIGBUS on access.
        if metadata.len() < mapped_len() as u64 {
            // SAFETY: `fd` is a valid open file descriptor with write
            // permissions. ftruncate extends the file to FILE_SIZE,
            // zero-filling the new bytes.
            unsafe {
                if ftruncate(fd, mapped_len() as i64) != 0 {
                    let error = std::io::Error::last_os_error();
                    Self::unlock_file(fd);
                    return Err(StateFileError::Io(error));
                }
            }
            // Persist the size metadata so a crash doesn't leave a
            // truncated file. See `create` for the full rationale.
            if let Err(e) = Self::do_fsync(fd) {
                // SAFETY: `fd` is valid and the open lock is held.
                unsafe { Self::unlock_file(fd) };
                return Err(e);
            }
        }

        // mmap
        let ptr = match Self::do_mmap(fd) {
            Ok(ptr) => ptr,
            Err(e) => {
                // SAFETY: `fd` is valid and the open lock is held.
                unsafe { Self::unlock_file(fd) };
                return Err(e);
            }
        };
        std::mem::forget(file);

        // Verify integrity using a volatile copy — no reference to the
        // mmap'd memory is created. This avoids aliasing issues even
        // though the mapping is MAP_SHARED.
        //
        // SAFETY: `ptr` is a valid mmap'd region of FILE_SIZE bytes,
        // FILE_SIZE ≥ size_of::<GenesisCoreState>(). `read_volatile`
        // copies the bytes without creating a reference. The state
        // is repr(C) with no padding gaps that would make the copy
        // uninitialised.
        // Verify integrity using a single volatile copy — no reference
        // to the mmap'd memory is created. This avoids aliasing issues
        // even though the mapping is MAP_SHARED. Using ONE copy for
        // both verify() and verify_checksum() prevents a torn read: a
        // concurrent writer could modify the MAP_SHARED file between
        // two separate read_volatile calls, so the magic/version check
        // would apply to one snapshot while the checksum check applied
        // to a different (possibly partially written) one.
        //
        // SAFETY: `ptr` is a valid mmap'd region of FILE_SIZE bytes,
        // FILE_SIZE ≥ size_of::<GenesisCoreState>(). `read_volatile`
        // copies the bytes without creating a reference. The state
        // is repr(C) with no padding gaps that would make the copy
        // uninitialised.
        let mut snapshot = unsafe { core::ptr::read_volatile(ptr as *const GenesisCoreState) };
        if let Err(e) = snapshot.verify() {
            // If the version is old (but magic and size are correct),
            // attempt migration before giving up.
            if let crate::state::CoreStateError::VersionMismatch { found: 2, .. } = e {
                // v2 → v3 migration: the struct layout is the same
                // (3288 bytes), just the version label and chemical
                // initialization need updating.
                //
                // SAFETY: `ptr` is a valid mmap'd region, exclusively
                // owned (no `MmapState` has been returned to the caller,
                // so no other thread can access it). The `&mut` is the
                // sole reference and does not alias with anything.
                let state_mut: &mut GenesisCoreState =
                    unsafe { &mut *(ptr as *mut GenesisCoreState) };
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                // Publish migration through the seqlock as well as the
                // flock: the flock excludes writers, while the seqlock
                // keeps lock-free readers from observing a partially
                // migrated state.
                // SAFETY: the flock is held and no other `&mut` exists.
                unsafe { state_mut.write_begin(now_ms) };
                let pre_migration = *state_mut;
                let migration = std::panic::catch_unwind(
                    std::panic::AssertUnwindSafe(|| state_mut.migrate_state(2)),
                );
                match migration {
                    Ok(Ok(())) => state_mut.write_end(),
                    Ok(Err(mig_err)) => {
                        *state_mut = pre_migration;
                        state_mut.write_end();
                        // SAFETY: unlock before munmap/close; fd/ptr are
                        // exclusively owned. No other references exist.
                        unsafe {
                            Self::unlock_file(fd);
                            munmap(ptr as *mut c_void, mapped_len());
                            close(fd);
                        }
                        return Err(StateFileError::VerificationFailed(mig_err));
                    }
                    Err(payload) => {
                        *state_mut = pre_migration;
                        state_mut.write_end();
                        // SAFETY: release the open lock before cleanup.
                        unsafe {
                            Self::unlock_file(fd);
                            munmap(ptr as *mut c_void, mapped_len());
                            close(fd);
                        }
                        std::panic::resume_unwind(payload);
                    }
                }
                // Sync the migrated state to disk. A failed msync here
                // IS significant: the caller believes the migration
                // succeeded and the state is durable, but the on-disk
                // file may still contain the old v2 data. If the
                // process crashes before the next explicit sync(), the
                // migration is lost and the next open() will re-trigger
                // it (which is safe but wasteful) or fail if the file
                // is partially written. Propagate the error so the
                // caller knows the state is not durable.
                if let Err(msync_err) = Self::do_msync(ptr, mapped_len()) {
                    // SAFETY: unlock before munmap/close; fd/ptr are
                    // exclusively owned. No other references exist.
                    unsafe {
                        Self::unlock_file(fd);
                        munmap(ptr as *mut c_void, mapped_len());
                        close(fd);
                    }
                    return Err(msync_err);
                }
                // Re-read the migrated state so the checksum check
                // below validates the post-migration bytes.
                // SAFETY: same as the initial read_volatile above.
                snapshot = unsafe { core::ptr::read_volatile(ptr as *const GenesisCoreState) };
            } else {
                // SAFETY: unlock before munmap/close; fd/ptr are
                // exclusively owned. No other references exist.
                unsafe {
                    Self::unlock_file(fd);
                    munmap(ptr as *mut c_void, mapped_len());
                    close(fd);
                }
                return Err(StateFileError::VerificationFailed(e));
            }
        }

        // Verify the CRC32 checksum on the same snapshot used for
        // verify() above (or the re-read after migration). The
        // `verify()` call only checks magic, version, and state_size —
        // it does NOT validate the checksum. Without this check, a
        // torn write or bit-rot that corrupts the neurochemical/zone
        // state but leaves the header intact is silently accepted,
        // leading to corrupt state being used by every consumer (tick
        // loop, IPC, cpufreq).
        if let Err(e) = snapshot.verify_checksum() {
            // SAFETY: unlock before munmap/close on valid ptr/fd that
            // we own exclusively. No other references exist. (close
            // alone would release the flock; we unlock explicitly to
            // match every other cleanup path in this function.)
            unsafe {
                Self::unlock_file(fd);
                munmap(ptr as *mut c_void, mapped_len());
                close(fd);
            }
            return Err(StateFileError::VerificationFailed(e));
        }

        // Sanity-check created_at: early versions of the daemon passed
        // a hardcoded `now_ms = 1000` to open_or_create, so existing
        // state files have created_at = 1000 (1 second after epoch).
        // This makes ping report an uptime of ~56 years. If created_at
        // is unreasonably old (before 2020-01-01), fix it to the current
        // time. This is a one-time migration — after the fix, the value
        // is correct and this branch is never taken again.
        //
        // 2020-01-01 UTC = 1577836800000 ms
        const REASONABLE_EPOCH_MS: u64 = 1_577_836_800_000;
        // SAFETY: Same invariant as the checksum read above — `ptr`
        // is a valid, mmap'd, page-aligned pointer to a region of
        // exactly `FILE_SIZE` bytes. `read_volatile` ensures we
        // observe the actual on-disk content. `GenesisCoreState` is
        // `#[repr(C)]` with no uninitialized padding.
        let snapshot = unsafe { core::ptr::read_volatile(ptr as *const GenesisCoreState) };
        if snapshot.header.created_at < REASONABLE_EPOCH_MS {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(REASONABLE_EPOCH_MS);
            // SAFETY: `ptr` is a valid mmap'd region, exclusively owned
            // (no `MmapState` has been returned to the caller yet). The
            // `&mut` is the sole reference and does not alias.
            let state_mut: &mut GenesisCoreState = unsafe { &mut *(ptr as *mut GenesisCoreState) };
            // SAFETY: the open-time flock is held; write_begin/write_end
            // publish this repair atomically to lock-free readers.
            unsafe { state_mut.write_begin(now_ms) };
            state_mut.header.created_at = now_ms;
            state_mut.write_end();
            // Sync the repaired state to disk. Propagate the error so
            // the caller knows the state is not durable — the same
            // rationale as the migration msync above.
            if let Err(msync_err) = Self::do_msync(ptr, mapped_len()) {
                // SAFETY: unlock before munmap/close on valid ptr/fd
                // that we own exclusively. No other references exist.
                unsafe {
                    Self::unlock_file(fd);
                    munmap(ptr as *mut c_void, mapped_len());
                    close(fd);
                }
                return Err(msync_err);
            }
        }

        // Re-anchor the circadian oscillator to the wall clock.
        //
        // The mmap'd phase only advances while the daemon is running
        // (each ADVANCE_NEURO tick advances it by the elapsed dt, and
        // dt is clamped to ≤10s, so both stalls and downtime lose
        // time). If the machine was powered off or Genesis was stopped
        // overnight, the stored phase is stale by the offline
        // duration. The phase is purely clock-derived — the dynamics
        // never modulate it (`set_circadian_phase` is reserved for
        // zeitgeber shifts) — so re-anchoring destroys no learned
        // state; it only corrects drift. Melatonin's level approaches
        // its new target smoothly (~2s), so even a large correction
        // produces a gentle transition, not a jump.
        //
        // Sub-minute drift is skipped: a normal restart loses only
        // seconds, and rewriting the file for a correction that small
        // is churn without physiological meaning.
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let wall_phase = local_phase_of_day(now_ms);
        // SAFETY: same invariant as the reads above — `ptr` is a
        // valid, mmap'd, page-aligned region of exactly FILE_SIZE
        // bytes and read_volatile observes the actual content.
        let snapshot = unsafe { core::ptr::read_volatile(ptr as *const GenesisCoreState) };
        let stored_phase = snapshot.neurochemicals.circadian_phase();
        let delta = (wall_phase - stored_phase).rem_euclid(1.0);
        let drift_phase = delta.min(1.0 - delta); // circular distance [0, 0.5]
        // 60 seconds of an 86400-second day ≈ 6.9e-4 of a cycle.
        if drift_phase * 86_400.0 >= 60.0 {
            // Release the open-time lock before modify() re-acquires it.
            // Keeping it held here would still be safe, but routing the
            // write through modify() keeps one protocol for all writes.
            // SAFETY: `fd` is valid and the open lock is held.
            unsafe { Self::unlock_file(fd) };
            let state = Self {
                ptr,
                fd,
                created: false,
                write_lock: std::sync::Mutex::new(()),
            };
            // Route the re-anchor through `modify` — flock + seqlock
            // + checksum recompute — so the write is safe even if
            // another process is concurrently reading or writing the
            // same file. A direct mutation here (like the one-time
            // legacy repairs above) would race any live writer.
            state.modify(now_ms, |s| {
                s.neurochemicals.set_circadian_phase(wall_phase);
            })?;
            // `state` drops here, unmapping and closing the fd.
            return Ok(state);
        }

        // Open-time verification is complete; release the exclusive
        // lock before returning the mapped state to its caller.
        // SAFETY: `fd` is valid and the open lock is held.
        unsafe { Self::unlock_file(fd) };
        Ok(Self {
            ptr,
            fd,
            created: false,
            write_lock: std::sync::Mutex::new(()),
        })
    }

    /// Open an existing state file, or create a new one if it doesn't
    /// exist. This is the most common entry point.
    pub fn open_or_create(
        path: impl AsRef<Path>,
        instance_id: u64,
        now_ms: u64,
    ) -> Result<Self, StateFileError> {
        let path = path.as_ref();
        // Try to create first. If the file already exists, fall back to
        // opening it. This avoids the TOCTOU race of checking `exists()`
        // then calling `create()` — between the check and the create,
        // another process could create the file, and the old `create`
        // (with truncate) would destroy it.
        match Self::create(path, instance_id, now_ms) {
            Ok(state) => Ok(state),
            Err(StateFileError::Io(ref e)) if e.raw_os_error() == Some(libc::EEXIST) => {
                Self::open(path)
            }
            Err(e) => Err(e),
        }
    }

    // ─── Reads ───────────────────────────────────────────────────

    /// Get an owned copy of the core state via a volatile byte read.
    ///
    /// This is the **safe** way to read the mmap'd state. It performs
    /// a `read_volatile` through a raw pointer — no `&GenesisCoreState`
    /// reference to the mmap'd memory is ever created, so there is no
    /// aliasing with concurrent writers.
    ///
    /// The copy may be **torn** if a write is in progress on another
    /// thread. For a seqlock-guarded consistent read, use
    /// [`read_consistent`] instead.
    ///
    /// This method is lock-free and wait-free.
    pub fn read(&self) -> GenesisCoreState {
        // SAFETY: `self.ptr` is a valid mmap'd region of FILE_SIZE
        // bytes (≥ size_of::<GenesisCoreState>()) for the lifetime of
        // `self`. `read_volatile` copies the bytes without creating
        // a reference, so there is no aliasing with a concurrent
        // writer's `&mut`. The copy may be torn if a write is in
        // progress — that is an accepted property of this method
        // (use `read_consistent` for tear-free reads).
        unsafe { core::ptr::read_volatile(self.ptr as *const GenesisCoreState) }
    }

    /// Attempt a lock-free, wait-free, **consistent** read of the
    /// state using the sequence lock protocol.
    ///
    /// Samples `seq_lock` before and after copying the state. If
    /// either sample is odd (write in progress) or the samples differ,
    /// returns `None` and the caller should retry.
    ///
    /// Unlike [`GenesisCoreState::read_consistent`], this method
    /// operates directly on the raw mmap pointer and **never** creates
    /// a `&GenesisCoreState` reference to the mmap'd memory. This
    /// eliminates the aliasing undefined behaviour that would occur if
    /// a `&` from `state()` coexisted with a `&mut` from `modify()`
    /// on another thread.
    ///
    /// The returned `GenesisCoreState` is an owned stack copy — the
    /// caller can use it freely without worrying about concurrent
    /// mutation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Retry up to 8 times for a consistent snapshot.
    /// let mut snapshot = None;
    /// for _ in 0..8 {
    ///     snapshot = mmap.read_consistent();
    ///     if snapshot.is_some() {
    ///         break;
    ///     }
    ///     std::hint::spin_loop();
    /// }
    /// ```
    pub fn read_consistent(&self) -> Option<GenesisCoreState> {
        // SAFETY: `self.ptr` is a valid mmap'd region for the lifetime
        // of `self`. The seq_lock field is at a known offset (40) within
        // the struct, properly aligned (8-byte aligned, verified by
        // compile-time assertions in header.rs). We access it through
        // `AtomicU64::from_ptr` which performs an atomic load without
        // creating a reference. The data copy uses `read_volatile`,
        // which also does not create a reference.
        let state_ptr = self.ptr as *const GenesisCoreState;
        let seq_lock_ptr = unsafe { core::ptr::addr_of!((*state_ptr).header.seq_lock) as *mut u64 };

        // SAFETY: `seq_lock_ptr` is a properly aligned `*mut u64`
        // pointing to the seq_lock field in the mmap'd region. The
        // atomic load does not mutate the pointed-to memory through
        // the raw pointer — it only reads it atomically.
        let seq = unsafe { AtomicU64::from_ptr(seq_lock_ptr) };

        // Acquire load: synchronises-with the Release store in
        // `write_end`, ensuring we see all data writes from the
        // completed transaction that produced this even seq_lock.
        let lock1 = seq.load(Ordering::Acquire);
        if lock1 & 1 != 0 {
            return None;
        }
        // Acquire fence ensures the data reads below happen after the
        // lock1 load. On weak memory orderings (ARM, RISC-V), the CPU
        // could otherwise reorder the data reads before the lock1
        // load, seeing stale data.
        fence(Ordering::Acquire);

        // SAFETY: `state_ptr` is a valid pointer to the mmap'd
        // `GenesisCoreState`. `read_volatile` copies
        // `size_of::<GenesisCoreState>()` bytes without creating a
        // reference, avoiding aliasing issues with a concurrent writer.
        // The seqlock protocol (checked below) ensures the copy is
        // consistent.
        let copy = unsafe { core::ptr::read_volatile(state_ptr) };

        // Acquire fence ensures all data reads above are completed
        // before we sample lock2. On weak memory orderings, the CPU
        // could otherwise reorder the data reads after the lock2
        // load, causing the seqlock validation to pass on stale
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

    // ─── Writes ──────────────────────────────────────────────────

    /// Safely modify the core state within a write transaction.
    ///
    /// This acquires the in-process writer mutex, begins the sequence
    /// lock (`write_begin`), calls the closure with a mutable
    /// reference, then releases the sequence lock (`write_end`).
    ///
    /// The `&mut GenesisCoreState` handed to the closure is the sole
    /// Rust reference to the mmap'd memory within this process for
    /// the duration of the closure — readers use [`read`] /
    /// [`read_consistent`] which copy bytes through raw pointers and
    /// never create a competing `&`.
    ///
    /// The closure receives the current time in milliseconds so it
    /// doesn't need to call a clock itself.
    ///
    /// # Errors
    ///
    /// Returns `StateFileError::LockFailed` if the internal writer
    /// mutex is poisoned (a previous writer thread panicked while
    /// holding the lock).
    pub fn modify<F>(&self, now_ms: u64, f: F) -> Result<(), StateFileError>
    where
        F: FnOnce(&mut GenesisCoreState),
    {
        // Serialize writers within this process. The seqlock only
        // protects readers from a single in-flight writer; it does not
        // protect two writers from each other. Holding this mutex for
        // the duration of the transaction makes `modify` safe across
        // threads.
        let _guard = self
            .write_lock
            .lock()
            .map_err(|_| StateFileError::LockFailed)?;

        // Acquire a cross-process exclusive lock on the state file.
        // The in-process mutex only protects against concurrent writers
        // within this process — but the file is mapped MAP_SHARED, so
        // another process that opens the same path can write to the
        // same bytes concurrently, defeating the seqlock. The flock
        // serializes writers across all processes. We use LOCK_NB
        // (non-blocking) so the daemon doesn't hang if another process
        // holds the lock — it returns FileLockBusy instead, letting
        // the caller retry or skip this tick.
        //
        // The lock is released after the transaction completes (or on
        // drop of the MmapState, which closes the fd).
        // SAFETY: `self.fd` is a valid open file descriptor kept alive
        // for the lifetime of this MmapState.
        unsafe { Self::lock_file(self.fd) }?;

        // SAFETY: We hold `write_lock` and the flock, so we are the
        // sole writer in this process and across all processes. No
        // reader creates a `&` to the mmap'd memory (they use
        // `read`/`read_consistent` which copy through raw pointers),
        // so the `&mut` below does not alias with anything.
        let state = unsafe { self.state_mut() };
        // SAFETY: `state_mut` returned a valid `&mut`; `write_begin`
        // is a seqlock acquire that atomically increments `seq_lock`
        // to odd (write in progress) and updates the heartbeat.
        unsafe { state.write_begin(now_ms) };
        // Snapshot the state before the closure so that if it panics
        // after partially mutating the mmap'd memory, we can restore
        // the pre-write snapshot before calling write_end(). Without
        // this, write_end() would recompute the checksum over torn
        // data and set seq_lock even, publishing a corrupt but
        // apparently-consistent state to all readers.
        let snapshot = *state;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(state)));
        // Hold the cross-process flock until AFTER write_end has fully
        // published the new state (seqlock flipped to even). Releasing
        // the lock before write_end would allow another process to
        // acquire it and call write_begin while we are still finishing
        // our data writes and seqlock flip, producing a torn
        // GenesisCoreState and a corrupted sequence lock.
        if let Err(payload) = result {
            // Restore the pre-write snapshot so readers never see
            // partially-written state.
            *state = snapshot;
            state.write_end();
            // SAFETY: `self.fd` is valid and we hold the lock.
            unsafe { Self::unlock_file(self.fd) };
            std::panic::resume_unwind(payload);
        }
        state.write_end();
        // SAFETY: `self.fd` is valid and we hold the lock.
        unsafe { Self::unlock_file(self.fd) };
        // Don't sync every write — too slow. Caller can sync explicitly.
        Ok(())
    }

    /// Flush dirty pages to disk. Call this after important writes
    /// to ensure state survives a crash.
    ///
    /// Acquires `write_lock` and the cross-process flock to serialize
    /// with any concurrent `modify()` call — including from another
    /// process. Without this, `msync` could flush a partially written
    /// (torn seqlock) state to disk if an IPC `modify()` is between
    /// `write_begin` and `write_end`, or if another process is mid-write.
    pub fn sync(&self) -> Result<(), StateFileError> {
        let _guard = self
            .write_lock
            .lock()
            .map_err(|_| StateFileError::LockFailed)?;
        // SAFETY: `self.fd` is a valid open file descriptor.
        unsafe { Self::lock_file(self.fd) }?;
        let result = Self::do_msync(self.ptr, mapped_len());
        // SAFETY: `self.fd` is valid and we hold the lock.
        unsafe { Self::unlock_file(self.fd) };
        result
    }

    /// Whether this file was created (vs opened from existing).
    pub fn was_created(&self) -> bool {
        self.created
    }

    // ─── Internal helpers ────────────────────────────────────────

    /// Get a mutable reference to the core state.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other thread in this process is
    /// reading or writing the state simultaneously. Readers use [`read`]
    /// / [`read_consistent`] which copy bytes through raw pointers and
    /// never create a `&` to the mmap'd memory, so they do not alias
    /// with the `&mut` returned here. The concern is solely preventing
    /// multiple writers.
    ///
    /// In production code, use [`modify`] instead — it acquires the
    /// writer mutex automatically. This method is exposed for
    /// single-threaded test code that needs to pass `&mut GenesisCoreState`
    /// to an engine that takes it by reference.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn state_mut(&self) -> &mut GenesisCoreState {
        // SAFETY: Caller guarantees no concurrent access. `self.ptr`
        // is a valid mmap'd region for the lifetime of `self`.
        unsafe { &mut *(self.ptr as *mut GenesisCoreState) }
    }

    fn do_mmap(fd: i32) -> Result<*mut u8, StateFileError> {
        // SAFETY: `fd` is a valid open file descriptor with read+write
        // permissions. Length is page-aligned via `mapped_len()`.
        // MAP_SHARED maps the file into memory for inter-process sharing.
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                mapped_len(),
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )
        };
        if ptr == MAP_FAILED {
            return Err(StateFileError::MmapFailed);
        }
        Ok(ptr as *mut u8)
    }

    fn do_msync(ptr: *mut u8, len: usize) -> Result<(), StateFileError> {
        // SAFETY: `ptr` is a valid mmap'd region of `len` bytes.
        let rc = unsafe { msync(ptr as *mut c_void, len, MS_SYNC) };
        if rc != 0 {
            return Err(StateFileError::MsyncFailed);
        }
        Ok(())
    }

    /// Flush file metadata (especially the size change from `ftruncate`)
    /// to stable storage.
    ///
    /// `msync` flushes dirty *data pages* but does **not** guarantee
    /// that the file's *size metadata* (the inode's `i_size`) is
    /// persisted. A crash after `ftruncate` but before the inode is
    /// flushed can leave a zero-length file, which would cause the
    /// next `open` to fail with `FileTooSmall`. `fdatasync` flushes
    /// the inode metadata (without the full `fsync` overhead of
    /// flushing ctime/mtime), making the size change durable.
    fn do_fsync(fd: i32) -> Result<(), StateFileError> {
        // SAFETY: `fd` is a valid open file descriptor.
        let rc = unsafe { fdatasync(fd) };
        if rc != 0 {
            return Err(StateFileError::FsyncFailed);
        }
        Ok(())
    }

    /// Acquire an exclusive cross-process advisory lock on the state
    /// file. Uses `flock(LOCK_EX | LOCK_NB)` — non-blocking, so if
    /// another process holds the lock, this returns
    /// `FileLockBusy` immediately instead of hanging the daemon.
    ///
    /// The lock is associated with the open file description (not the
    /// fd or the process), so it is automatically released when the
    /// last fd referring to this open file description is closed —
    /// including by `Drop` on `MmapState`. For explicit release, use
    /// [`unlock_file`].
    ///
    /// # Safety
    ///
    /// `fd` must be a valid open file descriptor.
    unsafe fn lock_file(fd: i32) -> Result<(), StateFileError> {
        // SAFETY: `fd` is a valid open file descriptor. `flock` is
        // async-signal-safe and does not allocate.
        let rc = unsafe { flock(fd, LOCK_EX | LOCK_NB) };
        if rc != 0 {
            let errno = unsafe { *libc::__errno_location() };
            if errno == libc::EWOULDBLOCK {
                return Err(StateFileError::FileLockBusy);
            }
            return Err(StateFileError::Io(std::io::Error::last_os_error()));
        }
        Ok(())
    }

    /// Release the cross-process advisory lock.
    ///
    /// # Safety
    ///
    /// `fd` must be a valid open file descriptor that currently holds
    /// a lock acquired by [`lock_file`].
    unsafe fn unlock_file(fd: i32) {
        // SAFETY: `fd` is a valid open file descriptor.
        unsafe {
            let _ = flock(fd, LOCK_UN);
        }
    }
}

impl Drop for MmapState {
    fn drop(&mut self) {
        // Best-effort msync before munmap. Without this, any core
        // state writes that the caller did not explicitly sync()
        // before drop are lost. We ignore errors here because Drop
        // cannot propagate them.
        let _ = Self::do_msync(self.ptr, mapped_len());
        // SAFETY: `self.ptr` and `self.fd` are valid and exclusively
        // owned by this instance. No other references exist at drop time.
        unsafe {
            munmap(self.ptr as *mut c_void, mapped_len());
            close(self.fd);
        }
    }
}

/// Errors that can occur when working with the state file.
#[derive(Debug)]
pub enum StateFileError {
    /// I/O error (file open, metadata, etc.)
    Io(std::io::Error),
    /// File doesn't exist (from `open`).
    NotFound,
    /// File is too small to contain a GenesisCoreState.
    FileTooSmall {
        /// The expected file size.
        expected: u64,
        /// The actual file size found.
        found: u64,
    },
    /// `ftruncate` failed.
    FtruncateFailed,
    /// `fdatasync` failed (flushing file metadata after ftruncate).
    FsyncFailed,
    /// `mmap` failed.
    MmapFailed,
    /// `msync` failed.
    MsyncFailed,
    /// State verification failed (bad magic, version, or checksum).
    VerificationFailed(CoreStateError),
    /// The internal writer mutex was poisoned.
    LockFailed,
    /// The cross-process file lock could not be acquired (another
    /// process is writing to the same state file). This is returned
    /// instead of blocking, so the caller can retry or degrade
    /// gracefully rather than hanging the daemon.
    FileLockBusy,
}

impl std::fmt::Display for StateFileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::NotFound => write!(f, "state file not found"),
            Self::FileTooSmall { expected, found } => {
                write!(
                    f,
                    "file too small: expected {expected} bytes, found {found}"
                )
            }
            Self::FtruncateFailed => write!(f, "ftruncate failed"),
            Self::FsyncFailed => write!(f, "fdatasync failed"),
            Self::MmapFailed => write!(f, "mmap failed"),
            Self::MsyncFailed => write!(f, "msync failed"),
            Self::VerificationFailed(e) => write!(f, "verification failed: {e}"),
            Self::LockFailed => write!(f, "state writer lock poisoned"),
            Self::FileLockBusy => write!(f, "state file locked by another process"),
        }
    }
}

impl std::error::Error for StateFileError {}
