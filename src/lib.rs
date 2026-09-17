//! Genesis — the core state schema and subcognitive foundation.

pub mod cognition;
pub mod daemon;
pub mod state;
pub mod store;

pub use state::GenesisCoreState;
pub use store::{MmapState, StateFileError};
