//! Tools — capabilities the body can exercise.
//!
//! Mirrors `python/tools/` on the mind side. Anatomically, tool use
//! recruits a left-lateralized premotor–parietal network: the skill
//! of operating on the world through instruments. Code analysis is
//! the daemon-side such capability — deterministic structural
//! analysis of source, complementary to the mind's neurosymbolic
//! code learner.
//!
//! ## What lives here
//!
//! - `code_analysis` — AST parsing, dependency graphs, semantic
//!   units, refactoring-target detection over Rust/Python source.

pub use crate::cognition::code_analysis;
