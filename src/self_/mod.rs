//! Self — the body's model of itself.
//!
//! (Named `self_` because `self` is a Rust keyword and cannot be a
//! module name, even raw; it mirrors `python/self/`.)
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
//! - `vision` — the body organs as mathematics: the eye's whole
//!   afferent chain (optics → retinal transduction → V1 → gaze
//!   plant) lives in the self region, because it is tissue the
//!   body owns, not a sensor plugged into it.

pub mod vision;

pub use crate::cognition::self_model;
