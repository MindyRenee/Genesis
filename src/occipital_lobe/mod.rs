//! Occipital lobe — early visual afferents and gamma phase-binding.
//!
//! Anatomy: the retinogeniculostriate pathway enters through the optic
//! nerve (subcortical branches to SCN, superior colliculus, pretectum,
//! and dLGN/TRN) and binds retinotopic features into coherent proto-object
//! assemblies in V1/V2 via Kuramoto gamma synchrony (Singer & von der Malsburg).
//!
//! Mirrors `python/occipital_lobe/` (the cortical visual hierarchy and
//! scene description layer):
//!
//! - `optic_nerve` — retinal afferent channels, pupillary light reflex,
//!   ciliary lens accommodation, voluntary eyelid transmission, and TRN
//!   relay gain (`daemon::optic_nerve`).
//! - `gamma` — Kuramoto oscillator binding of colour/contiguity/similarity
//!   into coherent `ProtoObject` assemblies (`daemon::gamma`).

pub use crate::daemon::gamma;
pub use crate::daemon::optic_nerve;
