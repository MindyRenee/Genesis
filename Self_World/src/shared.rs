//! Shared state with a refractory period.
//!
//! A patch of membrane cannot sit at two potentials at once — while
//! it is depolarized it is refractory, and a second signal arriving
//! then either drops (the impulse is spent, `try_lock` → `Err`) or
//! waits in the bouton until the membrane repolarizes (`lock` spins
//! briefly). The claim is a single atomic state — there is no gate,
//! no queue, no kernel object; the cell is simply busy or it is not.
//!
//! This is the only exclusion biology has, and it is the only one
//! the body needs: without it two threads hold `&mut` to the same
//! tissue and the heap tears (concurrent `Vec` clone vs. drop reads
//! freed memory — measured under ASan). With it the first signal
//! depolarizes, the rest see a busy membrane, and every caller's
//! existing "busy → skip the cycle" arm does exactly what a dropped
//! impulse does: nothing, this time.
//!
//! `Free<T>` is established-once state: the first caller wires the
//! circuit, everyone after reads it.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};

/// The membrane is refractory — the impulse arrived while a signal
/// was already in flight. Callers that can drop an impulse skip the
/// cycle; callers that cannot spin until repolarization.
#[derive(Debug, Clone, Copy)]
pub struct Denied;

impl std::fmt::Display for Denied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("membrane refractory — signal in flight")
    }
}

impl std::error::Error for Denied {}

impl Denied {
    /// `PoisonError::into_inner` — unreachable: the refractory state
    /// is a signal condition, not a poisoned gate.
    pub fn into_inner(self) -> ! {
        unreachable!("refractory membranes never poison")
    }
}

/// A held depolarization — RAII repolarization. While a `Hold`
/// lives the membrane reads busy to every other signal; when it
/// drops the membrane repolarizes and the next impulse lands.
pub struct Hold<'a, T> {
    shared: &'a Shared<T>,
}

impl<T> Deref for Hold<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.shared.cell.get() }
    }
}

impl<T> DerefMut for Hold<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.shared.cell.get() }
    }
}

impl<T> Drop for Hold<'_, T> {
    fn drop(&mut self) {
        // Release pairs with the Acquire claim: everything the
        // depolarization wrote is visible before the membrane
        // reads open again.
        self.shared.refractory.store(false, Ordering::Release);
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Hold<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&**self, f)
    }
}

/// Shared mutable state with a refractory period.
///
/// `claim` is the membrane's depolarized bit: CAS `false → true` to
/// fire, `Release` on repolarization. `Sync`/`Send` remain asserted
/// by hand — the tissue is shared, the refractory flag is what makes
/// sharing it sound.
pub struct Shared<T> {
    cell: UnsafeCell<T>,
    refractory: AtomicBool,
}

unsafe impl<T> Sync for Shared<T> {}
unsafe impl<T> Send for Shared<T> {}

impl<T> Shared<T> {
    pub const fn new(value: T) -> Self {
        Self {
            cell: UnsafeCell::new(value),
            refractory: AtomicBool::new(false),
        }
    }

    /// Fire the impulse — claim the membrane if it is at rest.
    /// `Err(Denied)` means a signal is already in flight: the
    /// impulse drops, exactly like a second stimulus during the
    /// absolute refractory period.
    pub fn try_lock(&self) -> Result<Hold<'_, T>, Denied> {
        self.refractory
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .map(|_| Hold { shared: self })
            .map_err(|_| Denied)
    }

    /// The synaptic terminal: the vesicle holds in the cleft until
    /// the membrane repolarizes, then fires. Spins — bounded by the
    /// in-flight signal's own duration, never by a scheduler gate.
    pub fn lock(&self) -> Result<Hold<'_, T>, Denied> {
        loop {
            match self.try_lock() {
                Ok(hold) => return Ok(hold),
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    /// Bare membrane access — no claim taken, none released. This
    /// is the raw `UnsafeCell` read: sound only where the caller can
    /// prove the membrane has no other live signal (single-owner
    /// phases, establish-once reads, `Shared<()>` markers whose
    /// contents are empty). Everything else goes through `lock` or
    /// `try_lock`, because an unclaimed `&mut` shared across threads
    /// is a torn heap, not a tolerated race.
    #[allow(clippy::mut_from_ref)]
    pub fn get(&self) -> &mut T {
        unsafe { &mut *self.cell.get() }
    }

    /// `Mutex::get_mut` — safe because `&mut self` is exclusive.
    pub fn get_mut(&mut self) -> &mut T {
        self.cell.get_mut()
    }

    /// `Mutex::into_inner`.
    pub fn into_inner(self) -> T {
        self.cell.into_inner()
    }
}

impl<T: Default> Default for Shared<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.try_lock() {
            Ok(hold) => f.debug_struct("Shared").field("inner", &*hold).finish(),
            Err(_) => f
                .debug_struct("Shared")
                .field("inner", &format_args!("<refractory>"))
                .finish(),
        }
    }
}

/// Established-once state — `OnceLock<T>` without the gate.
///
/// The first caller to `get_or_init` wires the value; under a race
/// two callers may each compute one and the last store wins — every
/// site that uses this tolerates that (probes, caches, lazy binds).
pub struct Free<T>(UnsafeCell<Option<T>>);

unsafe impl<T> Sync for Free<T> {}
unsafe impl<T> Send for Free<T> {}

impl<T> Free<T> {
    pub const fn new() -> Self {
        Self(UnsafeCell::new(None))
    }

    /// `OnceLock::get_or_init`.
    #[allow(clippy::mut_from_ref)]
    pub fn get_or_init(&self, f: impl FnOnce() -> T) -> &mut T {
        unsafe { &mut *self.0.get() }.get_or_insert_with(f)
    }

    /// `OnceLock::get`.
    pub fn get(&self) -> Option<&T> {
        unsafe { &*self.0.get() }.as_ref()
    }

    /// Mutable access to the wired value — `get` for writers.
    #[allow(clippy::mut_from_ref)]
    pub fn get_mut_shared(&self) -> Option<&mut T> {
        unsafe { &mut *self.0.get() }.as_mut()
    }

    /// `OnceLock::set` — fails if already wired, returning the value.
    pub fn set(&self, value: T) -> Result<(), T> {
        let slot = unsafe { &mut *self.0.get() };
        if slot.is_some() {
            Err(value)
        } else {
            *slot = Some(value);
            Ok(())
        }
    }

    /// `OnceLock::take`.
    pub fn take(&self) -> Option<T> {
        unsafe { &mut *self.0.get() }.take()
    }

    /// `OnceLock::get_mut`.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.0.get_mut().as_mut()
    }
}

impl<T> Default for Free<T> {
    fn default() -> Self {
        Self::new()
    }
}
