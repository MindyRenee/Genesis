//! Inhibition — the body's coordination substrate.
//!
//! What an operating system calls a mutex, a nervous system calls
//! inhibition. This module is the Rust half of `Self_World/inhibition.py`;
//! the two files are one mechanism seen from both sides of the
//! organism — here the math, there the words.
//!
//! # The circuits
//!
//! - [`Inhibition<T>`] — fast GABAergic mutual exclusion. Serializes
//!   access to shared state the way striatal lateral inhibition
//!   serializes competing drive: the conducting unit's neighbours
//!   wait in the inhibitory shadow (Groves 1983; Wickens et al. 1991;
//!   Koós & Tepper 1999). Wraps [`std::sync::Mutex`] and keeps the
//!   same API — `lock`, `try_lock`, `into_inner`, `get_mut` — so it
//!   is a drop-in mechanism, not a re-implementation.
//!
//! - [`Innate<T>`] — wiring fixed once, then immutable. `OnceLock`
//!   under its biological name: an innately wired quantity is
//!   established at genesis and never re-learned. A type alias, so
//!   every `OnceLock` API is inherited unchanged.
//!
//! # The mathematics
//!
//! Every [`Inhibition`] counts its own contention with lock-free
//! `Relaxed` counters — the measurement adds no synchronization of
//! its own, the same way an electrode does not change the neuron it
//! records:
//!
//! - `acquisitions` — completed admissions
//! - `contended` — admissions that found the gate already conducting
//! - `wait_nanos` — total time competitors spent in the shadow
//!
//! From these, per the Python side:
//!
//! - contention `c = contended / max(1, acquisitions)`
//! - mean wait `W̄ = wait_seconds / max(1, contended)`
//! - arrival rate `λ = acquisitions / elapsed`
//!
//! and Little's law `L = λ·W̄` (Little 1961) gives the expected number
//! of competitors held in the shadow, mapped into the interoceptive
//! range by the Naka–Rushton `tone = L/(1+L)` (1966) — see
//! [`Inhibition::stats`].
//!
//! # Embodiment exclusivity
//!
//! One organism, one body. The `flock(LOCK_EX | LOCK_NB)` on the
//! memory-mapped state file in `store/mmap_state.rs` is not a lock
//! the substrate could route around — it is the physical law of
//! exclusive embodiment, enforced by the kernel the way a skull
//! enforces it anatomically: exactly one process may map and mutate
//! her persistent state.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LockResult, Mutex, MutexGuard, OnceLock, TryLockError, TryLockResult};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Innately wired state — established once at genesis, immutable
/// thereafter.
///
/// `OnceLock<T>` under its biological name. A type alias, so
/// `Innate::new()`, `get_or_init`, `set`, `get`, `get_mut`, `take`
/// and every other `OnceLock` API work exactly as before.
pub type Innate<T> = OnceLock<T>;

/// The conducting axon — exclusive access while held.
///
/// `MutexGuard<'a, T>` under the same vocabulary as [`Inhibition`].
/// An alias so guard types annotate cleanly beside the gate.
pub type InhibitionGuard<'a, T> = MutexGuard<'a, T>;

/// Contention statistics for one [`Inhibition`].
///
/// The fields mirror `inhibition.py`'s `_Tone.stats` output; units
/// and formulas are documented on the module.
#[derive(Debug, Clone, Copy, Default)]
pub struct InhibitoryStats {
    /// Completed admissions.
    pub acquisitions: u64,
    /// Admissions that found the gate already conducting.
    pub contended: u64,
    /// Total seconds competitors spent in the inhibitory shadow.
    pub wait_seconds: f64,
    /// `contended / max(1, acquisitions)` — fraction of admissions
    /// that met inhibition.
    pub contention: f64,
    /// Mean seconds spent waiting per contended admission.
    pub mean_wait: f64,
    /// `acquisitions / elapsed` — admission rate since first use.
    pub arrival_rate: f64,
}

/// Fast GABAergic mutual exclusion — the mutex, observed.
///
/// `Inhibition<T>` is a `Mutex<T>` instrumented with contention
/// accounting. All synchronization behaviour is the kernel mutex's;
/// the counters are `Relaxed` atomics so measurement never adds
/// ordering of its own.
pub struct Inhibition<T> {
    inner: Mutex<T>,
    acquisitions: AtomicU64,
    contended: AtomicU64,
    wait_nanos: AtomicU64,
    /// Nanos since UNIX_EPOCH of the first admission (0 = unborn).
    born_ns: AtomicU64,
}

impl<T> Inhibition<T> {
    /// Const-construct an inhibitory gate guarding `value` — usable in
    /// `static` contexts exactly like `Mutex::new`.
    pub const fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
            acquisitions: AtomicU64::new(0),
            contended: AtomicU64::new(0),
            wait_nanos: AtomicU64::new(0),
            born_ns: AtomicU64::new(0),
        }
    }
}

impl<T> Inhibition<T> {
    /// Acquire the gate, blocking until the shadow clears.
    ///
    /// Same semantics as `Mutex::lock`: returns the guard, or the
    /// poison error if a previous holder panicked while conducting.
    /// An uncontended admission stays on the fast path; a contended
    /// one is timed and charged to the inhibitory shadow.
    pub fn lock(&self) -> LockResult<MutexGuard<'_, T>> {
        self.note_birth();
        match self.inner.try_lock() {
            Ok(guard) => {
                self.acquisitions.fetch_add(1, Ordering::Relaxed);
                Ok(guard)
            }
            Err(TryLockError::WouldBlock) => {
                let start = Instant::now();
                let result = self.inner.lock();
                let waited = start.elapsed().as_nanos() as u64;
                self.wait_nanos.fetch_add(waited, Ordering::Relaxed);
                self.contended.fetch_add(1, Ordering::Relaxed);
                self.acquisitions.fetch_add(1, Ordering::Relaxed);
                result
            }
            Err(TryLockError::Poisoned(e)) => Err(std::sync::PoisonError::new(e.into_inner())),
        }
    }

    /// Non-blocking attempt — the axon either conducts or reports
    /// the synapse already firing. Mirrors `Mutex::try_lock`.
    pub fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>> {
        self.note_birth();
        match self.inner.try_lock() {
            Ok(guard) => {
                self.acquisitions.fetch_add(1, Ordering::Relaxed);
                Ok(guard)
            }
            Err(e) => Err(e),
        }
    }

    /// Contention statistics — see the module docstring for the math.
    pub fn stats(&self) -> InhibitoryStats {
        let acquisitions = self.acquisitions.load(Ordering::Relaxed);
        let contended = self.contended.load(Ordering::Relaxed);
        let wait_seconds = self.wait_nanos.load(Ordering::Relaxed) as f64 * 1e-9;
        let born = self.born_ns.load(Ordering::Relaxed);
        let elapsed = if born == 0 {
            1e-9
        } else {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(born);
            (now.saturating_sub(born) as f64 * 1e-9).max(1e-9)
        };
        InhibitoryStats {
            acquisitions,
            contended,
            wait_seconds,
            contention: contended as f64 / acquisitions.max(1) as f64,
            mean_wait: wait_seconds / contended.max(1) as f64,
            arrival_rate: acquisitions as f64 / elapsed,
        }
    }

    /// Consume the gate, yielding the guarded value — `Mutex::into_inner`.
    pub fn into_inner(self) -> LockResult<T> {
        self.inner.into_inner()
    }

    /// `Mutex::get_mut` — mutable access without locking, when the
    /// caller already holds exclusive access to the gate itself.
    pub fn get_mut(&mut self) -> LockResult<&mut T> {
        self.inner.get_mut()
    }

    /// `Mutex::is_poisoned`.
    pub fn is_poisoned(&self) -> bool {
        self.inner.is_poisoned()
    }

    /// `Mutex::clear_poison`.
    pub fn clear_poison(&self) {
        self.inner.clear_poison();
    }

    /// Stamp the first-use epoch once. `Instant` is not
    /// const-constructible, so the birth marker lives in an atomic
    /// and is set on the first admission instead of at construction.
    fn note_birth(&self) {
        if self.born_ns.load(Ordering::Relaxed) == 0 {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1)
                .max(1);
            let _ = self
                .born_ns
                .compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
        }
    }
}

impl<T: Default> Default for Inhibition<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Inhibition<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inhibition")
            .field("inner", &self.inner)
            .finish()
    }
}
