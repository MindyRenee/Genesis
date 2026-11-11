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
//!
//! The mind side carries the full workstation (`python/tools/`):
//! the persistent terminal, file surgery, discovery, languages,
//! math, sensors, lint, net, notebook, personal toolchain, git,
//! archive, github, android, sqlite, dictionary, formats, and the
//! fun bench — instruments are description, and description is her
//! side of the anatomy.

pub use crate::cognition::code_analysis;
