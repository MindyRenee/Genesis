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
//! in heap memory. With mmap, the OS manages paging — the ~3.2 KB
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
//! `GenesisCoreState` on the stack. Readers therefore create no
//! reference to the mmap'd memory at all.
//!
//! ### The writer never forms a reference into the mapping
//!
//! Every production write path — [`modify`] and the open-time
//! repairs/migrations — runs its mutation on a **stack copy** and
//! publishes it with `publish_stack` (seqlock odd → single
//! `write_volatile` → seqlock even). No `&` or `&mut` into
//! the shared mapping is ever formed, so there is no aliasing with
//! concurrent readers under Stacked/Tree Borrows: the old pattern
//! (a `&mut GenesisCoreState` materialised into `MAP_SHARED` for the
//! whole transaction, which Miri flagged and which only happened to
//! survive because volatile loads are never dead) is gone.
//!
//! Consequences, kept deliberately:
//!
//! - A panicking [`modify`] closure only ever saw the stack copy, so
//!   the mapping needs no rollback — it was never touched.
//! - `read_volatile` in the readers remains load-bearing: it stops
//!   the compiler eliding or merging the seqlock-guarded loads, it is
//!   not (and never was) what made concurrent access defined.
//! - [`MmapState::state_mut`] still exists, but only single-threaded
//!   test code calls it; grep the tree to confirm before trusting
//!   this.
//!
//! ## File layout
//!
//! ```text
//! offset 0:     GenesisCoreState (3296 bytes)
//! offset 3296:  (unused, padding to page boundary)
//! ```
//!
//! The state struct is 3296 bytes and the file is page-aligned, so the
//! mapping is at least one page and the state sits at its start. (On a
//! host with a 64 KiB page — aarch64, for instance — `page_align_up`
//! expands the file to 64 KiB, not 4 KiB.)

use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use core::sync::atomic::{AtomicU64, Ordering, fence};

use libc::{
    LOCK_EX, LOCK_NB, LOCK_UN, MAP_FAILED, MAP_SHARED, MS_ASYNC, MS_SYNC, PROT_READ, PROT_WRITE,
    c_void, close, fdatasync, flock, ftruncate, mmap, msync, munmap,
};

use crate::state::{CoreStateError, GenesisCoreState};

/// The logical file size — one 4K page. The state struct (3296 bytes)
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
//     created, so readers never alias a writer's reference.
//   - Writes: `modify` acquires `write_lock` before creating the
//     `&mut GenesisCoreState`, ensuring no other thread in this
//     process has a competing reference.
//
// Note: the second bullet gives *exclusion*, not sound aliasing. See
// the module-level "The writer does create a `&mut`" section — a
// reader's `read_volatile` on bytes covered by a live `&mut` is still
// UB under Stacked/Tree Borrows, and this impl is only correct today
// because volatile loads cannot be optimised away.
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

        // Take the open-time lock immediately, for the same reason
        // `open` does: between `ftruncate` and the initialising store
        // below there is a window in which the file exists, is the
        // right size, and is entirely zeros. `open` locks, so a second
        // daemon starting in that window would read the zeros and fail
        // with `BadMagic`. `create_new` has already succeeded, so the
        // EEXIST path is unaffected.
        // SAFETY: `fd` is a valid open file descriptor.
        unsafe { Self::lock_file(fd) }?;

        // Set the file size
        // SAFETY: ftruncate is a POSIX call on a valid, owned fd. The fd is
        // obtained from `file.as_raw_fd()` and `file` is kept alive (forgotten
        // later) so the fd remains valid. FILE_SIZE is a small constant.
        unsafe {
            if ftruncate(fd, mapped_len() as i64) != 0 {
                // This call created the file. Leaving a zero-filled stub
                // behind would make every subsequent `open_or_create`
                // fail with `BadMagic` until it was deleted by hand —
                // turning one transient error into a permanent manual
                // repair. Unlink it on every failure path below.
                let _ = std::fs::remove_file(path);
                return Err(StateFileError::FtruncateFailed);
            }
        }
        // Persist the file size metadata to stable storage. Without
        // this, a crash after ftruncate but before the inode is
        // flushed can leave a zero-length file. `msync` only flushes
        // data pages, not the inode's i_size.
        if let Err(e) = Self::do_fsync(fd) {
            let _ = std::fs::remove_file(path);
            return Err(e);
        }

        // mmap the file
        let ptr = match Self::do_mmap(fd) {
            Ok(p) => p,
            Err(e) => {
                let _ = std::fs::remove_file(path);
                return Err(e);
            }
        };

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
            // See above: unlink the stub this call created.
            let _ = std::fs::remove_file(path);
            return Err(e);
        }

        // The open-time lock only protects initialization. Release it
        // before returning the live mapping so another MmapState can open
        // the file and contend through modify()'s writer lock.
        // SAFETY: fd is valid and the initialization transaction is complete.
        unsafe { Self::unlock_file(fd) };

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

        // Check file size. Current files are at least the current
        // struct size, but a known legacy v3 file is intentionally
        // smaller because the inference-signals tail grew from 60 to
        // 64 bytes. That layout is migrated below after the mapping is
        // established. Reject anything smaller than the known legacy
        // layout rather than trying to interpret arbitrary/truncated
        // bytes as a Genesis state.
        let metadata = match file.metadata() {
            Ok(metadata) => metadata,
            Err(e) => {
                // SAFETY: `fd` is valid and the open lock is held.
                unsafe { Self::unlock_file(fd) };
                return Err(StateFileError::Io(e));
            }
        };
        if metadata.len() < crate::state::core_state::LEGACY_SIZE as u64 {
            // SAFETY: `fd` is valid and the open lock is held.
            unsafe { Self::unlock_file(fd) };
            return Err(StateFileError::FileTooSmall {
                expected: GenesisCoreState::SIZE as u64,
                found: metadata.len(),
            });
        }

        // Ensure the file is at least the page-aligned mapping length so
        // the mmap region is fully backed by the file. Without this, a
        // file that is exactly the state size would be mapped
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
            // Two kinds of mismatch are repairable in place, and the
            // order of `verify()` decides which one we are looking at:
            // it checks the version before the size.
            match e {
                // Old version label, layout still current: migrate the
                // version (and the chemical initialization that came
                // with it).
                crate::state::CoreStateError::VersionMismatch { found: 2, .. } => {
                    // SAFETY: `ptr`/`fd` are exclusively owned here — no
                    // `MmapState` exists yet — and the flock is held.
                    snapshot =
                        unsafe { Self::migrate_and_snapshot(ptr, fd, |s| s.migrate_state(2))? };
                }
                // Correct version, older struct size: the reserve at
                // the end of the struct grew (see
                // `GenesisCoreState::migrate_layout`), so no field
                // before the signals block moved but the file is too
                // short to read as-is.
                crate::state::CoreStateError::SizeMismatch { found, .. } => {
                    // SAFETY: as above.
                    snapshot = unsafe {
                        Self::migrate_and_snapshot(ptr, fd, |s| {
                            s.migrate_layout(found)?;
                            s.write_end_tail();
                            Ok(())
                        })?
                    };
                }
                // Anything else — bad magic, a version we do not know,
                // a size we cannot migrate from — is not a migration,
                // it is a damaged or foreign file. Per the state
                // integrity rules it is a startup error to be repaired
                // explicitly, never something to paper over by starting
                // fresh.
                other => {
                    // SAFETY: unlock before cleanup; fd/ptr are
                    // exclusively owned. No other references exist.
                    unsafe {
                        Self::unlock_file(fd);
                        munmap(ptr as *mut c_void, mapped_len());
                        close(fd);
                    }
                    return Err(StateFileError::VerificationFailed(other));
                }
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

        // Scrub any non-finite float before the state is published to
        // any reader. The CRC cannot catch this: a state file that
        // legitimately contains NaN is byte-exact and passes both
        // `verify` and `verify_checksum`. The per-tick circuit breaker
        // would fix most of it on the first tick, but between here and
        // then a lock-free IPC read would hand NaN to the cognitive
        // mind.
        // SAFETY: same invariant as the reads above — `ptr` is a valid,
        // mmap'd, page-aligned region of exactly `FILE_SIZE` bytes, the
        // flock is held, and no `MmapState` exists yet. The scrub runs
        // on a stack copy and publishes through `publish_stack`: no
        // reference into the mapping is formed.
        let mut scrubbed =
            unsafe { core::ptr::read_volatile(ptr as *const GenesisCoreState) };
        if scrubbed.scrub_non_finite() {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            scrubbed.header.last_updated = now_ms;
            scrubbed.header.heartbeat = scrubbed.header.heartbeat.wrapping_add(1);
            scrubbed.checksum = scrubbed.compute_checksum();
            // SAFETY: valid ptr/len; flock held so no writer can be
            // mid-transaction.
            unsafe { Self::publish_stack(ptr, &scrubbed) };
            if let Err(msync_err) = Self::do_msync(ptr, mapped_len()) {
                // SAFETY: unlock before munmap/close on valid ptr/fd
                // that we own exclusively.
                unsafe {
                    Self::unlock_file(fd);
                    munmap(ptr as *mut c_void, mapped_len());
                    close(fd);
                }
                return Err(msync_err);
            }
        }

        // Repair an inverted seqlock parity.
        //
        // `header.seq_lock` is excluded from the CRC (it changes on
        // every write), and the write order is: data -> checksum store
        // -> `seq_lock` to even. A crash, or a partial page writeback —
        // the kernel offers no atomicity for a 4 KiB page and walks it
        // in address order, so bytes 40..48 can reach disk while bytes
        // 3224..3228 have not — can therefore leave a disk image that
        // is fully written, CRC-valid, and holding an ODD `seq_lock`.
        // Nothing in `verify()`/`verify_checksum()` looks at it.
        //
        // The consequence is severe and permanent. `read_consistent`
        // rejects an odd `seq_lock`, so every `GET_STATE` /
        // `GET_NEURO_SUMMARY` / `PING` fails until a write happens. Then
        // the first `write_begin` increments an odd value to an even
        // one, which readers read as "stable" — publishing a torn
        // snapshot mid-transaction — and `write_end` increments that to
        // odd again, blinding every reader for the rest of the process.
        //
        // Repair the parity while the flock is held. No checksum
        // recomputation is needed: `seq_lock` is outside the CRC.
        // The counter is driven through a raw atomic — no reference
        // into the mapping is formed.
        // SAFETY: `ptr` is a valid, mmap'd, page-aligned region of
        // exactly `FILE_SIZE` bytes; the flock is held and no
        // `MmapState` has been returned to the caller. The offset is
        // `SEQ_LOCK_OFFSET` (layout-pinned, 8-byte aligned).
        if unsafe { core::ptr::read_volatile(ptr as *const GenesisCoreState) }
            .header
            .seq_lock
            & 1
            == 1
        {
            let seq = unsafe { AtomicU64::from_ptr(ptr.add(Self::SEQ_LOCK_OFFSET) as *mut u64) };
            seq.fetch_add(1, Ordering::Release);
            // SAFETY: `ptr`/`len` are a valid mmap'd region; the flock
            // is held so no writer in any process can be mid-transaction.
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
        let mut snapshot = unsafe { core::ptr::read_volatile(ptr as *const GenesisCoreState) };
        if snapshot.header.created_at < REASONABLE_EPOCH_MS {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(REASONABLE_EPOCH_MS);
            // Stack repair published through `publish_stack`: the
            // open-time flock is held and the seqlock window keeps
            // this atomic to lock-free readers. No mapping reference.
            snapshot.header.last_updated = now_ms;
            snapshot.header.heartbeat = snapshot.header.heartbeat.wrapping_add(1);
            snapshot.header.created_at = now_ms;
            snapshot.checksum = snapshot.compute_checksum();
            unsafe { Self::publish_stack(ptr, &snapshot) };
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
        let mut snapshot = unsafe { core::ptr::read_volatile(ptr as *const GenesisCoreState) };
        let old_checksum = snapshot.checksum;
        snapshot.neurochemicals.recompute_derived();
        snapshot.sync_neurochemistry_to_state();
        snapshot.checksum = snapshot.compute_checksum();
        if snapshot.checksum != old_checksum {
            unsafe { Self::publish_stack(ptr, &snapshot) };
            if let Err(e) = Self::do_msync(ptr, mapped_len()) {
                unsafe {
                    Self::unlock_file(fd);
                    munmap(ptr as *mut c_void, mapped_len());
                    close(fd);
                }
                return Err(e);
            }
        }
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
    /// The closure runs against a **stack copy**, never against the
    /// mapping: no `&` or `&mut` into the mmap'd memory is formed on
    /// any path here (see [`publish_stack`]). On success the prepared
    /// copy is published atomically w.r.t. seqlock readers
    /// (odd → volatile write → even) and the checksum with it; on
    /// closure panic the mapping is untouched — there is nothing to
    /// roll back — and the panic resumes after the flock is released.
    ///
    /// This acquires the in-process writer mutex plus the
    /// cross-process flock (non-blocking; `FileLockBusy` when another
    /// process is writing), stamps the heartbeat, runs the closure,
    /// and publishes.
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

        // Stack working copy: `read` copies through a raw pointer and
        // forms no reference, so the closure's `&mut` below aliases
        // nothing in the mapping (or anywhere else). Holding the
        // flock additionally guarantees no other process is
        // publishing concurrently, so this copy is coherent.
        let mut next = self.read();
        // Heartbeat stamp — the non-seqlock half of what
        // `write_begin` used to do inline on the mapping.
        next.header.last_updated = now_ms;
        next.header.heartbeat = next.header.heartbeat.wrapping_add(1);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut next)));
        match result {
            Ok(()) => {
                next.checksum = next.compute_checksum();
                // Hold the flock until AFTER the publish has flipped
                // the seqlock back to even (see `publish_stack`).
                // Releasing earlier would let another process start
                // its own publish mid-window, tearing the state and
                // the sequence lock.
                unsafe { Self::publish_stack(self.ptr, &next) };
                // SAFETY: `self.fd` is valid and we hold the lock.
                unsafe { Self::unlock_file(self.fd) };
                // Don't sync every write — too slow. Caller can sync explicitly.
                Ok(())
            }
            Err(payload) => {
                // The mapping was never touched — the panicking
                // closure only saw the stack copy. Release the flock
                // and resume; the in-process mutex poisons via the
                // guard's drop, as before.
                // SAFETY: `self.fd` is valid and we hold the lock.
                unsafe { Self::unlock_file(self.fd) };
                std::panic::resume_unwind(payload);
            }
        }
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

    /// Periodic (non-critical) flush.
    ///
    /// `sync` uses `msync(MS_SYNC)`, which blocks until the pages have
    /// actually reached the device — tens to hundreds of milliseconds
    /// on a journaled filesystem, and a full fsync stall under load.
    /// The 5 Hz tick performs a dozen-plus `modify` transactions, each
    /// running a full coupled-ODE integration, so blocking the tick
    /// thread on a disk flush also delays every IPC write behind it and
    /// widens the window in which `read_consistent` exhausts its
    /// retries.
    ///
    /// For the periodic path, flush without blocking and without
    /// waiting for a writer to finish: a transaction in flight is
    /// already dirty, and the next flush will carry it. Use `sync` on
    /// the shutdown path, where durability must be confirmed.
    pub fn sync_async(&self) -> Result<(), StateFileError> {
        // `try_lock` rather than `lock`: skipping a flush is always
        // safe here, blocking the tick loop is not.
        let Ok(_guard) = self.write_lock.try_lock() else {
            return Ok(());
        };
        // SAFETY: `self.ptr` is a valid mmap'd region of
        // `mapped_len()` bytes, for the lifetime of `self`.
        let rc = unsafe { msync(self.ptr as *mut c_void, mapped_len(), MS_ASYNC) };
        if rc != 0 {
            return Err(StateFileError::MsyncFailed);
        }
        Ok(())
    }

    /// Whether this file was created (vs opened from existing).
    pub fn was_created(&self) -> bool {
        self.created
    }

    // ─── Internal helpers ────────────────────────────────────────

    /// Byte offset of `header.seq_lock` from the start of the mapping.
    ///
    /// Derived from the layout (`header` is the first field at offset
    /// 0; `seq_lock` is at offset 40 within the header — both pinned
    /// by compile-time asserts in `core_state.rs`/`header.rs`), never
    /// from a live reference, so the seqlock can be driven without
    /// forming any reference into the mapping.
    const SEQ_LOCK_OFFSET: usize = core::mem::offset_of!(GenesisCoreState, header)
        + core::mem::offset_of!(crate::state::header::CoreStateHeader, seq_lock);

    /// Publish a fully-prepared stack state to the mapping.
    ///
    /// The integrity contract, replacing the old in-place
    /// `write_begin`/`write_end` on a mapping reference:
    ///
    /// 1. Flip `seq_lock` to odd (write in progress) through a raw
    ///    atomic — lock-free readers sample this and retry instead of
    ///    trusting bytes written below.
    /// 2. `write_volatile` the whole prepared struct. No `&`/`&mut`
    ///    into the mapping is formed here or by the caller, so there
    ///    is no aliasing with concurrent readers under Stacked/Tree
    ///    Borrows — the previous `&mut`-into-`MAP_SHARED` pattern was
    ///    unsound (Miri-flagged) and is now gone from every
    ///    production path.
    /// 3. Flip `seq_lock` back to even (write complete) with release
    ///    ordering, publishing all preceding bytes to readers.
    ///
    /// The caller must hold `write_lock` and the flock, and
    /// `prepared.checksum` must already cover `prepared` (as
    /// `modify` and the open-time repairs ensure before calling).
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid mmap'd region of at least
    /// `GenesisCoreState::SIZE` bytes with no other writer active
    /// (caller-held locks guarantee this).
    unsafe fn publish_stack(ptr: *mut u8, prepared: &GenesisCoreState) {
        // SAFETY: `ptr.add(SELF::SEQ_LOCK_OFFSET)` is the `seq_lock`
        // u64 by the layout const above (8-byte aligned: 40 % 8 == 0).
        // `AtomicU64::from_ptr` performs atomic RMWs without creating
        // a reference; nothing else here creates one either.
        let seq = unsafe { AtomicU64::from_ptr(ptr.add(Self::SEQ_LOCK_OFFSET) as *mut u64) };
        seq.fetch_add(1, Ordering::AcqRel);
        // Self-heal an inverted parity, mirroring the old
        // `write_begin`: starting from a stale odd lock, one increment
        // lands even ("stable") and would publish a torn snapshot
        // mid-write to readers.
        if seq.load(Ordering::Acquire) & 1 == 0 {
            seq.fetch_add(1, Ordering::AcqRel);
        }
        fence(Ordering::Acquire);
        // SAFETY: `ptr` is a valid writable mapping of at least SIZE
        // bytes; `prepared` is a valid stack value (`Copy`, so this
        // moves bytes without borrowing the mapping). Volatile: the
        // compiler must emit the store.
        unsafe { core::ptr::write_volatile(ptr as *mut GenesisCoreState, *prepared) };
        fence(Ordering::Release);
        // SAFETY: same atomic as above; Release publishes the bytes.
        seq.fetch_add(1, Ordering::Release);
    }

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
    /// In production code, use [`modify`] instead — it runs the
    /// closure on a stack copy and publishes without ever forming a
    /// mapping reference. This method is exposed ONLY for
    /// single-threaded test code that needs to pass
    /// `&mut GenesisCoreState` to an engine that takes it by
    /// reference; no production path calls it (verified: the only
    /// in-tree callers are tests).
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

    /// Run a migration against the mapped state in place, under the
    /// flock `open()` already holds, and return the post-migration
    /// snapshot.
    ///
    /// The returned snapshot is what the caller's `verify_checksum`
    /// must run against: reading the state again afterwards is a
    /// separate volatile copy, so the checksum check has to validate the
    /// same bytes the migration produced.
    ///
    /// On any error the flock is released and the mapping and fd are
    /// torn down before returning — the caller has no `MmapState` yet,
    /// so nothing else owns them. A panicking migration is rolled back
    /// the same way and then resumed.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid mmap'd region of at least
    /// `GenesisCoreState::SIZE` bytes, `fd` the open fd for it, and the
    /// caller must hold the flock and have no other reference to the
    /// mapped state. On error the function consumes `fd` and `ptr`.
    unsafe fn migrate_and_snapshot(
        ptr: *mut u8,
        fd: i32,
        migrate: impl FnOnce(&mut GenesisCoreState) -> Result<(), CoreStateError>,
    ) -> Result<GenesisCoreState, StateFileError> {
        // Stack working copy: `read_volatile` forms no reference, so
        // the migration's `&mut` aliases nothing in the mapping (or
        // anywhere else). A panicking or failing migration leaves the
        // mapping untouched — there is nothing to roll back.
        // SAFETY: `ptr` is a valid mmap'd region of at least
        // `GenesisCoreState::SIZE` bytes; the caller holds the flock
        // and no `MmapState` exists yet.
        let mut next = unsafe { core::ptr::read_volatile(ptr as *const GenesisCoreState) };
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        next.header.last_updated = now_ms;
        next.header.heartbeat = next.header.heartbeat.wrapping_add(1);
        // Publish the migration through the seqlock as well as the
        // flock: the flock excludes writers, while the seqlock keeps
        // lock-free readers from observing a partially migrated state.
        let migration =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| migrate(&mut next)));
        match migration {
            Ok(Ok(())) => {
                // Belt-and-suspenders: the migrations maintain the
                // checksum themselves, but the publish must cover
                // exactly what it writes — recompute over the final
                // bytes (also covers the heartbeat stamp above).
                next.checksum = next.compute_checksum();
                // SAFETY: same mapping invariants as the read above.
                unsafe { Self::publish_stack(ptr, &next) };
            }
            Ok(Err(mig_err)) => {
                // SAFETY: release the lock before cleanup; fd/ptr are
                // exclusively owned. No other references exist.
                unsafe {
                    Self::unlock_file(fd);
                    munmap(ptr as *mut c_void, mapped_len());
                    close(fd);
                }
                return Err(StateFileError::VerificationFailed(mig_err));
            }
            Err(payload) => {
                // SAFETY: release the lock before cleanup; fd/ptr are
                // exclusively owned. No other references exist.
                unsafe {
                    Self::unlock_file(fd);
                    munmap(ptr as *mut c_void, mapped_len());
                    close(fd);
                }
                std::panic::resume_unwind(payload);
            }
        }
        // Sync the migrated state to disk. A failed msync here IS
        // significant: the caller believes the migration succeeded and
        // the state is durable, but the on-disk file may still contain
        // the old data. If the process crashes before the next explicit
        // sync(), the migration is lost and the next open() will
        // re-trigger it (which is safe but wasteful) or fail if the file
        // is partially written. Propagate the error so the caller knows
        // the state is not durable.
        if let Err(msync_err) = Self::do_msync(ptr, mapped_len()) {
            // SAFETY: unlock before cleanup; fd/ptr are exclusively
            // owned. No other references exist.
            unsafe {
                Self::unlock_file(fd);
                munmap(ptr as *mut c_void, mapped_len());
                close(fd);
            }
            return Err(msync_err);
        }
        // Return the exact bytes published — no second copy, so the
        // caller's checksum check validates what migration produced.
        Ok(next)
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
