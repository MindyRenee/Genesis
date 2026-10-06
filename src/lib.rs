//! Genesis — the core state schema and subcognitive foundation.

pub mod cognition;
pub mod daemon;
pub mod field;
pub mod state;
pub mod store;

pub use field::{BraidTrajectory, FieldError, GaugeTransport, GlobalField, PoincareBall, SubsystemLayer, SystemFiber, SystemEngine, Vector};
pub use state::GenesisCoreState;
pub use store::{MmapState, StateFileError};
