//! Genesis — the core state schema and subcognitive foundation.
//!
//! ## Layout
//!
//! `python/` opens to brain regions because the mind is organized
//! anatomically; `src/` presents the body's anatomy the same way.
//! The implementation lives in `daemon`, `state`, `store`,
//! `cognition`, and `field`; the region modules (`basal_forebrain`,
//! `brainstem`, `thalamus`, `neurochemical`, `memory`, `limbic_system`,
//! `basal_ganglia`, `self_`, `sleep`, `tools`, `infrastructure`)
//! re-export those modules under the structure they implement, so
//! the crate reads the same way the mind does: click `src/` and the
//! subcortical regions are what you see.

// ─── The body, by region ─────────────────────────────────────────
// Anatomical view of the implementation below — re-exports only,
// no code moves. See each region's mod.rs for what lives there.
pub mod basal_forebrain;
pub mod basal_ganglia;
pub mod brainstem;
pub mod infrastructure;
pub mod limbic_system;
pub mod memory;
pub mod neurochemical;
pub mod self_;
pub mod sleep;
pub mod thalamus;
pub mod tools;

// ─── The spine ───────────────────────────────────────────────────
// The map that holds the layout together — every region and the
// mind-side package it projects to. The mind declares the same
// table back (python/topology.py); the two halves join over the
// region names.
pub mod topology;

// ─── Implementation ──────────────────────────────────────────────
pub mod cognition;
pub mod daemon;
pub mod data_dir;
pub mod field;
pub mod math;
pub mod state;
pub mod store;

pub use data_dir::data_dir;
pub use field::{
    BraidTrajectory, FieldError, GaugeTransport, GlobalField, PoincareBall, SubsystemLayer,
    SystemEngine, SystemFiber, Vector,
};
pub use state::GenesisCoreState;
pub use store::{MmapState, StateFileError};
