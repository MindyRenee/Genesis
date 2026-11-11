//! Insula — interoceptive sensing and active interoceptive inference.
//!
//! Anatomy: the insular cortex receives homeostatic lamina-I and vagal
//! visceral afferents via the brainstem and ventromedial thalamic nuclei
//! (Craig 2009) and compares them against top-down visceral predictions
//! (Seth 2013; Allen & Friston 2018) to form the felt state of the body
//! (Damasio's proto-self).
//!
//! Mirrors `python/insula/`:
//!
//! - `interoception` — hardware visceral sensing (thermal, CPU, memory,
//!   PSI, power, rest) and sensor-presence bitmasks (`daemon::interoception`).
//! - `active_inference` — interoceptive free energy, surprise, precision,
//!   and allostatic load (`daemon::active_inference`).
//! - `ions` — cellular electrochemical gradients, Nernst potentials, and
//!   membrane potential (`state::ions`).

pub use crate::daemon::active_inference;
pub use crate::daemon::interoception;
pub use crate::state::ions;
