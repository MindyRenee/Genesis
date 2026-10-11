//! Memory — the body's memory systems.
//!
//! Anatomy: the hippocampus and medial temporal lobe consolidate
//! short-term traces into long-term storage, and associative linking
//! is what lets one memory retrieve another. In Genesis the daemon
//! owns these stores — the mind's memories persist in the body, not
//! in the process that experienced them, which is why recall survives
//! a mind restart.
//!
//! ## What lives here
//!
//! - `ring_buffer` — the STM ring: the bounded short-term trace store.
//! - `ltm` — the long-term episodic store.
//! - `sdr` — sparse distributed representations: the overlap-coded
//!   format memories are written and matched in.
//! - `memory` — the state-block pointers and gating weights (encoding,
//!   consolidation, retrieval, plasticity) neurochemistry modulates.
//! - `consolidation` — STM→LTM consolidation, gated by state.
//! - `association` — association discovery between stored traces.

pub use crate::daemon::association;
pub use crate::daemon::consolidation;
pub use crate::state::memory;
pub use crate::store::ltm;
pub use crate::store::ring_buffer;
pub use crate::store::sdr;
