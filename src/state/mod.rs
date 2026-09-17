//! Genesis core state — the memory-mapped hub of the brain.
//!
//! This module defines the single fixed-layout structure that holds
//! Genesis's entire runtime state. See [`core_state::GenesisCoreState`].
//!
//! ## Schema v3.2 architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────┐
//! │                  GenesisCoreState (3288 bytes)                │
//! │  memory-mapped, lock-free readable, CRC32 checksummed         │
//! │                                                               │
//! │  ┌──────────────┐  ┌────────────────────────────────────────┐ │
//! │  │   Header      │  │  NeurochemicalVector (2416 bytes)      │ │
//! │  │  magic, ver,  │  │  18 chemicals × 56 bytes = 1008        │ │
//! │  │  seqlock, hb  │  │  18×18 coupling matrix = 1296          │ │
//! │  └──────────────┘  │  effective levels, arousal, valence     │ │
//! │                    │  plasticity gate, emergent phase        │ │
//! │                    │  ACTH (HPA cascade), tonic/phasic       │ │
//! │                    │  circadian oscillator (SCN model)       │ │
//! │  ┌──────────────┐  └────────────────────────────────────────┘ │
//! │  │ ActiveZones   │  ┌────────────────────────────────────────┐│
//! │  │  task zone    │  │  MemoryPointers (120 bytes)             ││
//! │  │  + emergent   │  │  STM/LTM/proc/sem pointers              ││
//! │  │    phase      │  │  + emotional gating weights             ││
//! │  │  + subconsc.  │  │    encoding, consolidation,             ││
//! │  │    flags      │  │    retrieval, plasticity                ││
//! │  └──────────────┘  └────────────────────────────────────────┘ │
//! │  ┌──────────────┐  ┌────────────────────────────────────────┐ │
//! │  │ Manifest      │  │  Checksum + InferenceSignals           │ │
//! │  │  16 modules   │  │  CRC32 + 60 bytes active inference     │ │
//! │  └──────────────┘  └────────────────────────────────────────┘ │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## What's novel in v3.2
//!
//! - **18 coupled neurochemicals** (up from 17): added melatonin
//!   (circadian hormone, sleep promotion via SCN oscillator).
//! - **Circadian oscillator**: SCN-model phase oscillator drives
//!   melatonin secretion and diurnal modulation of cortisol, dopamine,
//!   serotonin, and histamine baselines. Cortisol awakening response
//!   (CAR), daytime dopamine/histamine elevation, and night-time
//!   serotonin-to-melatonin conversion are all emergent.
//! - **NREM/REM sleep phases**: the single Sleeping phase is split
//!   into NREM (slow-wave, glymphatic clearance) and REM (dreaming,
//!   emotional processing), driven by the cholinergic/aminergic
//!   flip-flop (Hobson & McCarley reciprocal interaction model).
//! - **HPA axis redesign**: cortisol and CRH are stress hormones with
//!   a zero resting baseline (leaky integrators), produced only by the
//!   maturation-gated CRH → ACTH → cortisol cascade.

pub mod core_state;
pub mod header;
pub mod inference;
pub mod manifest;
pub mod memory;
pub mod neurochemical;
pub mod sanitize;
pub mod zones;

// Re-export the primary types.
pub use core_state::{CoreStateError, GenesisCoreState};
pub use header::{CoreStateHeader, MAGIC, SCHEMA_VERSION};
pub use inference::InferenceSignals;
pub use manifest::{MAX_MODULES, ModuleEntry, ModuleId, ModuleStatus, RuntimeManifest};
pub use memory::{MemoryPointers, WORKING_SET_SIZE};
pub use neurochemical::{
    ActiveRegion, CouplingMatrix, DEFAULT_COUPLING_MATRIX, NEUROCHEMICAL_COUNT, NeuroTickParams,
    Neurochemical, NeurochemicalId, NeurochemicalVector, WC_GAIN, WC_SELF_EXCITATION,
};
pub use sanitize::{finite_clamp, finite_or, sanitize_array_f32, sanitize_matrix_f32};
pub use zones::{
    ActiveZones, CognitiveZone, MAX_SUBCOGNITIVE_TASKS, MentalPhase, subcognitive_flag,
};
