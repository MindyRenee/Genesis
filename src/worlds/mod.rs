//! Worlds — the organism's sensory/actuation boundary with the outer world.
//!
//! Mirrors `python/worlds/`:
//!
//! - `actuators` — real local and relayed output limbs (`daemon::actuators`).
//! - `dyadic_model` — interlocutor affect and attunement dynamics
//!   (`daemon::dyadic_model`).
//! - `optic_nerve` — visual afferent boundary from the outer world
//!   (`daemon::optic_nerve`).

pub use crate::daemon::actuators;
pub use crate::daemon::dyadic_model;
pub use crate::daemon::optic_nerve;
