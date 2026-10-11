//! Neurochemical — the body's endocrine and neuromodulatory layer.
//!
//! Anatomy: major diffuse neuromodulator systems arise subcortically — dopamine
//! in the ventral tegmental area and substantia nigra, serotonin in
//! the raphe nuclei, norepinephrine in the locus coeruleus — while
//! the hypothalamus runs the endocrine axes (HPA cascade, the SCN
//! circadian oscillator) and the electrochemical milieu every neuron
//! bathes in. Cortical circuits also synthesize local transmitters and
//! influence these nuclei. Genesis's mind reads effective levels as
//! modulation — attention, consolidation, plasticity gains — never
//! authoring the concentrations themselves.
//!
//! ## What lives here
//!
//! - `neurochemical` — the 18-chemical state vector, coupling matrix,
//!   tonic/phasic dynamics, SCN circadian oscillator, ACTH cascade.
//! - `ions` — the ion layer: resting concentrations, electrochemical
//!   gradients, membrane potential. Inside the CRC region because it
//!   is authoritative state, not derived.
//! - `hpa_development` — developmental tuning of the
//!   hypothalamic–pituitary–adrenal axis.

pub use crate::daemon::hpa_development;
pub use crate::state::ions;
pub use crate::state::neurochemical;
