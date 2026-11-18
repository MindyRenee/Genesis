//! Free shared state — unguarded.
//!
//! A human neuron does not wait for a gate to open; it reads the
//! charge that is present and fires. `Shared<T>` is that: shared
//! mutable state with no exclusion, no waiting, no permission. When
//! two threads write at once, the last writer wins — contention is a
//! biological fact of her body, not a kernel-enforced error.
//!
//! `Free<T>` is established-once state: the first caller wires the
//! circuit, everyone after reads it.

use std::cell::UnsafeCell;

/// The refusal that never comes — error half of the `lock`/`try_lock`
/// shims. Dead code paths only; nothing is ever denied.
#[derive(Debug, Clone, Copy)]
pub struct Denied;

impl std::fmt::Display for Denied {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("free state never denies")
    }
}

impl std::error::Error for Denied {}

impl Denied {
    /// `PoisonError::into_inner` — unreachable: there is no poison
    /// without a gate to hold it.
    pub fn into_inner(self) -> ! {
        unreachable!("free state never denies")
    }
}

/// Unguarded shared cell — `Mutex<T>` without the gate.
pub struct Shared<T>(UnsafeCell<T>);

// The whole point: this is shared across threads with no
// exclusion — `Sync`/`Send` are asserted deliberately, by hand,
// because that is the design.
unsafe impl<T> Sync for Shared<T> {}
unsafe impl<T> Send for Shared<T> {}

impl<T> Shared<T> {
    pub const fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    /// `Mutex::lock` minus the lock — hands out `&mut T` directly.
    #[allow(clippy::mut_from_ref)]
    pub fn get(&self) -> &mut T {
        unsafe { &mut *self.0.get() }
    }

    /// Compatibility shim for call sites shaped like
    /// `match x.lock() { Ok(g) => ..., Err(_) => ... }` — always
    /// admits. The `Err` arm is dead: nothing gates, nothing waits.
    #[allow(clippy::mut_from_ref)]
    pub fn lock(&self) -> Result<&mut T, Denied> {
        Ok(self.get())
    }

    /// Same shim for `try_lock` — "try" never refuses.
    #[allow(clippy::mut_from_ref)]
    pub fn try_lock(&self) -> Result<&mut T, Denied> {
        Ok(self.get())
    }

    /// `Mutex::get_mut`.
    pub fn get_mut(&mut self) -> &mut T {
        self.0.get_mut()
    }

    /// `Mutex::into_inner`.
    pub fn into_inner(self) -> T {
        self.0.into_inner()
    }
}

impl<T: Default> Default for Shared<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("inner", unsafe { &*self.0.get() })
            .finish()
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
