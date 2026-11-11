//! Latent spaces — the shared-latent substrate.
//!
//! One latent space, one home. The body-side half of the shared
//! substrate lives here — the κ-stereographic manifold
//! (`substrate`: `StereographicSphere`, `GlobalField`,
//! `SystemFiber`, `SubsystemLayer`, `GaugeTransport`,
//! `BraidTrajectory`), its transition dynamics (`engine`:
//! `SystemEngine`), the multi-fiber integration over `body`
//! (self), `prediction` (inner), and `agency` (outer)
//! (`runtime`), the band-resolved flux/tension decomposition
//! (`spectrum`), the motor projection (`motor`), and the
//! thermodynamic partition over state space
//! (`state::phase_partition`). The mind's half is
//! `python/latent_spaces/` — the S³ Hopf manifold the deposits
//! live on. The two halves are one space seen from two sides of
//! the IPC boundary.

pub mod engine;
pub mod motor;
pub mod runtime;
pub mod spectrum;
pub mod substrate;

pub use engine::SystemEngine;
pub use motor::MotorTarget;
pub use runtime::{FieldIntegration, FieldRuntime, LATENT_DIMENSION};
pub use substrate::{
    BraidTrajectory, FieldError, GaugeTransport, GlobalField, Matrix, StereographicSphere,
    SubsystemLayer, SystemFiber, Vector, provenance_fingerprint,
};

pub use crate::state::phase_partition;
