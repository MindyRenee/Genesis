//! Basal forebrain — cortical acetylcholine, attention, encoding, and
//! the somnogenic A1 site of action.
//!
//! Anatomy: the basal forebrain lies ventral to the striatum and
//! anterior to the hypothalamus. It holds the major cholinergic
//! projection to cortex and hippocampus (Ch4 nucleus basalis →
//! neocortex and amygdala; Ch1/Ch2 septal and diagonal-band nuclei →
//! hippocampus). It is also a sleep-wake structure — wake/REM-active
//! cholinergic, parvalbumin GABAergic, and glutamatergic populations
//! intermingled with somatostatin+ GABAergic sleep-promoting neurons
//! (Xu et al., 2015) — and the best-characterized site where the
//! homeostatic sleep-pressure signal acts: extracellular adenosine
//! rises there with time awake and A1-inhibits the wake-active
//! neurons, so the somnogenic signal silences the arousal output this
//! same region produces.
//!
//! The mind's mirror is `python/basal_forebrain` — the words; the
//! math is here.
//!
//! ## What lives here (implementation stays in `crate::state`; this
//! region is the anatomical view)
//!
//! - `basal_forebrain` — the region's own math: the cholinergic
//!   release equation (LC/REM-on release map modulated by A1 adenosine
//!   occupancy and by the expected-uncertainty drive, Yu & Dayan
//!   2005), the waking set-point for the network-free path, the
//!   cholinergic sensory-precision gain that `daemon::active_inference`
//!   applies to its observation channel, and the α = 1 − precision
//!   uncertainty signal.
//! - `neurochemical` — the level dynamics the release equation feeds:
//!   the dedicated adenosine mechanism is Process S itself
//!   (saturating-exponential accumulation, exponential glymphatic
//!   clearance — Daan, Beersma & Borbély 1984), and the AChE excess
//!   decay is this region's enzymatic fast decay.
//! - `sleep_network` — the populations whose `lc`/`rem` drive the
//!   release map's state profile (NREM 0.34, waking 0.58, REM 0.77 —
//!   Vazquez & Baghdoyan 2001).
//! - `memory` — the encoding gate cholinergic tone sets: high ACh
//!   biases encoding over retrieval (Hasselmo), held here as the
//!   encoding/consolidation/retrieval weights the pointers carry.

pub use crate::state::basal_forebrain;
pub use crate::state::memory;
pub use crate::state::neurochemical;
pub use crate::state::sleep_network;
