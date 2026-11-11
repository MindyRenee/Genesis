//! Parietal lobe — spatial computation, exact mathematics, and motor fields.
//!
//! Anatomy: the intraparietal sulcus (IPS) and superior/inferior parietal
//! lobules form the brain's dorsal spatial-numerical substrate — magnitude,
//! exact mathematical structure, coordinate transformations, and the
//! frontoparietal reach/grasp motor fields.
//!
//! Mirrors `python/parietal_lobe/` (the word and symbolic routing layer):
//!
//! - `math` — the pure machine computation twins of `parietal_lobe/math/*`
//!   (`abstract_algebra`, `axioms`, `combinatorics`, `complex_analysis`,
//!   `discrete`, `geometry`, `linear_algebra`, `number_theory`, `proof`,
//!   `real_analysis`, `topology`, `trig`).
//! - `motor` — distributed spatial and action motor fields (`field::motor`)
//!   decoded over the canonical 64×64 grid.
//! - `tensor_slab` — the named multidimensional field store (`daemon::tensor_slab`)
//!   through which spatial and motor fields are exchanged.

pub use crate::daemon::tensor_slab;
pub use crate::field::motor;
pub use crate::math;
pub use crate::math::{
    abstract_algebra, algebra, arithmetic, axioms, calculus, combinatorics, complex_analysis,
    discrete, geometry, linear_algebra, number_sense, number_theory, proof, real_analysis,
    statistics, topology, trig,
};
