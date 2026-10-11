//! Infrastructure — the plumbing the regions stand on.
//!
//! Mirrors `python/infrastructure/`. Not itself a brain region:
//! connective tissue — where state files live, and the memory-map
//! substrate the body's authoritative state rides on.
//!
//! ## What lives here
//!
//! - `data_dir` — canonical data-directory resolution (env override,
//!   per-checkout pin, XDG, home fallback) shared by every resolver.
//! - `mmap_state` — the memory-mapped state file machinery: seqlock
//!   reads, page alignment, crash-safe open. The physical substrate
//!   `core_state` is written into.

pub use crate::data_dir;
pub use crate::store::mmap_state;
