//! Self — the body's model of itself.
//!
//! (Named `self_` because `self` is a Rust keyword and cannot be a
//! module name, even raw; it mirrors `Self_World/self/`.)
//!
//! Anatomy: there is no single self region — bodily self-modeling is
//! distributed across insula, medial prefrontal, and posteromedial
//! cortex — but the *schema* of the body, the structured first-person
//! account of its own state, is a prerequisite for any functional
//! selfhood. This region is that account on the body side.
//!
//! ## What lives here
//!
//! - `self_model` — the structured first-person summary of the
//!   system's own state.
//! - `vision` — the eye as mathematics: dioptric optics, the neural
//!   retina, V1 cortex, and the oculomotor gaze plant, composed into
//!   one afferent `VisionSystem`.

pub mod vision;

pub use crate::cognition::self_model;
