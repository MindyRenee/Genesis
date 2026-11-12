//! Sleep — the body's sleep–wake machinery.
//!
//! Mirrors `python/sleep/` on the mind side (dreaming, inner life).
//! Anatomy: the circadian clock that decides *when* sleep happens is
//! the suprachiasmatic nucleus in the hypothalamus, and the wake
//! drive is as bodily as heartbeat. The daemon owns the body's side
//! of it — sleep staging on its own clock, and the hardware wake
//! alarm that lets a sleeping system rise on time.
//!
//! ## What lives here
//!
//! - `rtc_wake` — the RTC wakealarm: the body's alarm clock, armed
//!   via `SET_WAKE_ALARM` so a sleeping mind can be woken by
//!   hardware, not by a process that has to stay awake to notice.

pub use crate::daemon::rtc_wake;
