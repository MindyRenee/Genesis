//! Thalamus — the relay and integration layer.
//!
//! Anatomy: the thalamus is the brain's relay station — every sensory
//! channel except olfaction passes through thalamic nuclei on the way
//! to cortex, and thalamocortical loops broadcast and bind activity
//! across regions. In Genesis the body↔mind boundary crosses here:
//! the Unix-socket IPC protocol is the fiber tract, the module
//! manifest is the relay's routing table, and the field runtime is
//! the integrative dynamics the broadcast rides on.
//!
//! ## What lives here
//!
//! - `ipc` — the wire protocol between daemon and cognitive mind
//!   (`PROTOCOL_VERSION`, opcodes, request/reply framing). The
//!   geniculate bodies of the architecture: the mind's only line to
//!   the body runs through it.
//! - `manifest` — the module registry: which subsystems are alive,
//!   their status, CPU share. The relay knows its connections.
//! - `engine`, `runtime` — the field architecture: every subsystem a
//!   layer of the system's latent field, gauge transport between
//!   them. The integrative substrate the tick loop drives.

pub use crate::daemon::ipc;
pub use crate::field::engine;
pub use crate::field::runtime;
pub use crate::state::manifest;
