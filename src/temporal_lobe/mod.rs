//! Temporal lobe — medial-temporal episodic memory, SDRs, and associations.
//!
//! Anatomy: the hippocampal formation, entorhinal/perirhinal cortex, and
//! anterior temporal semantic hub — short-term to long-term consolidation,
//! sparse distributed pattern completion, and associative link discovery.
//!
//! Mirrors `python/temporal_lobe/`:
//!
//! - `ltm` — append-only memory-mapped episodic store and SimHash index
//!   (`store::ltm`).
//! - `ring_buffer` — bounded short-term trace ring (`store::ring_buffer`).
//! - `sdr` — 2048-bit sparse distributed representations (`store::sdr`).
//! - `consolidation` — emotionally gated STM→LTM promotion
//!   (`daemon::consolidation`).
//! - `association` — offline and waking associative bridge discovery
//!   (`daemon::association`).

pub use crate::daemon::association;
pub use crate::daemon::consolidation;
pub use crate::store::ltm;
pub use crate::store::ring_buffer;
pub use crate::store::sdr;
