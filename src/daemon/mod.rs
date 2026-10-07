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
//! The daemon is **lease-driven**: the cognitive mind holds a short
//! lease (renewed by every reactive IPC command) and drives each
//! function through IPC while alive. Past the lease
//! (`tick::MIND_LEASE_SECS`), the daemon drives the full loop
//! itself — neurochemistry, consolidation, sleep-gated dreaming,
//! association, sensing, intention expiry with zone arbitration and
//! traces, body-control recommendation publishing, and model
//! checkpoints — so the system keeps living instead of freezing
//! during a stall, crash, or disconnect. Body-control *application*
//! stays mind-requested; the fallback publishes and senses but never
//! acts outward on its own. Staleness detection runs regardless, so
//! a crashed mind's modules still read Stopped.
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
pub mod hpa_development;
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
    capture_hardware_state, derive_boost, derive_policy, freq_range, policy_was_applied,
    report_body_control_outcome, restore_hardware_state,
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
