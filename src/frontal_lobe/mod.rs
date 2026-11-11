//! Frontal lobe — executive intention, zone control, and structural reasoning.
//!
//! Anatomy: prefrontal and premotor cortex — goal maintenance, cognitive
//! zone arbitration, self-monitoring, and structured symbolic/syntactic
//! analysis.
//!
//! Mirrors `python/frontal_lobe/`:
//!
//! - `intention` — executive goal scoring and cognitive zone selection
//!   (`cognition::intention`).
//! - `zones` — active cognitive task zones and transition tallies
//!   (`state::zones`).
//! - `code_analysis` — deterministic AST and dependency-graph analysis
//!   (`cognition::code_analysis`).
//! - `inference` — published precision, expected free energy, and policy
//!   authority signals (`state::inference`).

pub use crate::cognition::code_analysis;
pub use crate::cognition::intention;
pub use crate::state::inference;
pub use crate::state::zones;
