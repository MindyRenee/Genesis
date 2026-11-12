//! Brainstem — the autonomic core of the body.
//!
//! Anatomy: the brainstem runs the rhythms the mind never chooses —
//! arousal and visceral reflexes, interacting with hypothalamic homeostasis.
//! It also houses the monoaminergic nuclei that broadcast
//! neuromodulators upward (locus coeruleus → norepinephrine, raphe →
//! serotonin, VTA/SN → dopamine), and the reticular activating
//! system that sets arousal zones. In Genesis the daemon's own clock
//! and body-maintenance loop is this region: it ticks whether or not
//! the cognitive mind is alive.
//!
//! ## What lives here (implementation stays in `crate::daemon` /
//! `crate::state`; this region is the anatomical view)
//!
//! - `tick` — the pacemaker. Sole authority on time and physics:
//!   integrates neurochemistry, circadian phase, and sleep staging at
//!   `TICK_INTERVAL_MS`, lease-driven by the mind while it lives.
//! - `interoception` — visceral sensing of the machine body: CPU,
//!   thermal, battery, disk. The body's read on its own condition.
//! - `cpufreq` — autonomic frequency/thermal policy (the brainstem's
//!   homeostatic reflexes, applied to silicon).
//! - `active_inference` — the body's anticipatory regulation:
//!   free energy, precision, surprise, allostatic load — published as
//!   `inference` signals the mind observes rather than drives.
//! - `zones` — arousal/task zone state (the reticular activating
//!   system's projection target).
//! - `core_state`, `header`, `sanitize`, `legacy_v3` — the somatic
//!   ground truth: the fixed-layout body state every region shares,
//!   its integrity checks, and the frozen prior schema.

pub use crate::daemon::active_inference;
pub use crate::daemon::cpufreq;
pub use crate::daemon::interoception;
pub use crate::daemon::tick;
pub use crate::state::core_state;
pub use crate::state::header;
pub use crate::state::inference;
pub use crate::state::legacy_v3;
pub use crate::state::sanitize;
pub use crate::state::zones;
