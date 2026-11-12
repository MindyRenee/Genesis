//! The body's mathematics — pure machine computation.
//!
//! One file per domain, mirroring the parietal faculty on the
//! mind side (`python/parietal_lobe/math/`). The split is the
//! organism's own: here are numbers, functions, and structures —
//! the engine. The Python twins hold the words: parsing what a
//! question means and composing the answer's shape. Rust computes;
//! the mind speaks.
//!
//! Everything is free functions over plain data — no state, no
//! IO, no interpretation. A camera frame or a cube move arrives as
//! numbers; these modules return numbers.

pub mod abstract_algebra;
pub mod axioms;
pub mod combinatorics;
pub mod complex_analysis;
pub mod discrete;
pub mod geometry;
pub mod linear_algebra;
pub mod number_theory;
pub mod proof;
pub mod real_analysis;
pub mod topology;
pub mod trig;
