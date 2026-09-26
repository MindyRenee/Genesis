//! Subcognitive daemon — the always-on background process.
//!
//! This is Genesis's subcognitive mind. It runs continuously, even when
//! nobody is interacting with it, and owns the core state, memory
//! stores, and the IPC surface the cognitive mind talks to. It:
//!
//! - Advances the neurochemical dynamics (the "feel" loop)
//! - Consolidates short-term memories into long-term storage
//! - Finds associations between memories
//! - Dreams when sleeping (free association → creative insights)
//! - Maintains the core state and memory stores
//!
//! # Process model
//!
//! The daemon is a standalone process that owns the core state file,
//! STM ring buffer, and LTM store. Other processes (the Python
//! cognitive mind) communicate with it via a Unix domain socket.
//!
//! The daemon is **reactive**, not timer-driven: it never initiates
//! neurochemical, memory, or body-control actions on its own. The
//! cognitive mind requests each function through IPC when its own
//! state (brain waves, thresholds, urges) says it's time. The only
//! autonomous work the daemon does is staleness detection — if the
//! cognitive mind's heartbeats stop, the daemon marks its modules
//! Stopped so the manifest reflects reality.
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                  Subcognitive Daemon                      │
//! │                                                           │
//! │  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
//! │  │ Reactive  │→ │ Neuro    │→ │ Consol-  │→ │ Assoc-   │ │
//! │  │ dispatch  │  │ chem     │  │ idation  │  │ iation   │ │
//! │  └───────────┘  └──────────┘  └──────────┘  └──────────┘ │
//! │       │                                          │        │
//! │       ▼                                          ▼        │
//! │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
//! │  │ Dream   │  │ House-   │  │ IPC      │  │ Manifest  │  │
//! │  │ (sleep) │  │ keeping  │  │ Server   │  │ Update    │  │
//! │  └─────────┘  └──────────┘  └──────────┘  └───────────┘  │
//! └──────────────────────────────────────────────────────────┘
//! ```

pub mod active_inference;
pub mod association;
pub mod consolidation;
pub mod cpufreq;
pub mod dyadic_model;
pub mod interoception;
pub mod ipc;
/// RTC wake alarm — scheduling its own return from suspension.
pub mod rtc_wake;
pub mod tick;

pub use active_inference::{ActiveInferenceEngine, InferenceResult, apply_inference_feedback};
pub use association::{ASSOCIATION_THRESHOLD, AssociationEngine, AssociationResult, DreamResult};
pub use consolidation::{ConsolidationEngine, ConsolidationResult};
pub use cpufreq::{
    BodyControlState, BoostState, CPUFREQ_INTERVAL_TICKS, FreqPolicy, apply_policy,
    capture_hardware_state, derive_boost, derive_policy, freq_range, restore_hardware_state,
};
pub use dyadic_model::{DyadicAffectModel, DyadicResult, UserAffectObservation};
pub use interoception::{BodyState, INTEROCEPTION_INTERVAL_TICKS, Interoceptor};
pub use ipc::cmd;
pub use ipc::{
    BodyControlSummary, BodyStateSummary, IpcClient, IpcServer, LtmAccess, MAX_MESSAGE_LEN,
    MemoryStats, NeuroSummary, PlasticityProfile, default_handler, reactive_handler, read_message,
    write_message,
};
pub use tick::{TICK_INTERVAL_MS, TickLoop, TickResult, current_ms};
