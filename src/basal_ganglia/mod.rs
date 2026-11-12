//! Basal ganglia — action selection.
//!
//! Anatomy: the basal ganglia are the gatekeepers of action —
//! competing motor and cognitive programs are scored and one is
//! released while the rest stay inhibited (direct/indirect pathway
//! balance, dopamine-gated). Goal and intention selection is their
//! cognitive loop territory.
//!
//! ## What lives here
//!
//! - `intention` — the intention manager: scoring active goals
//!   against the self-model and deciding which cognitive zone to
//!   enter next.

pub use crate::cognition::intention;
