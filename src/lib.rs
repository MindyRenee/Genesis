//! Genesis — the core state schema and subcognitive foundation.

pub mod cognition;
pub mod daemon;
pub mod data_dir;
pub mod state;
pub mod store;

pub use data_dir::data_dir;
pub use state::GenesisCoreState;
pub use store::{MmapState, StateFileError};
