//! Latent spaces — the Riemannian manifold, fiber bundle, and spectral substrate.
//!
//! Mirrors `python/latent_spaces/`:
//!
//! - `field` — the κ-stereographic sphere (`StereographicSphere`), `GlobalField`,
//!   `SystemFiber`, `SubsystemLayer`, `GaugeTransport`, and `BraidTrajectory`.
//! - `engine` — `SystemEngine` state-transition dynamics (`field::engine`).
//! - `runtime` — `FieldRuntime` multi-fiber integration over `body` (self),
//!   `prediction` (inner), and `agency` (outer) (`field::runtime`).
//! - `spectrum` — band-resolved flux/tension decomposition and carrier
//!   kinematics (`field::spectrum`).
//! - `phase_partition` — thermodynamic partition over state space
//!   (`state::phase_partition`).

pub use crate::field;
pub use crate::field::engine;
pub use crate::field::runtime;
pub use crate::field::spectrum;
pub use crate::state::phase_partition;
