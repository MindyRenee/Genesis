//! Hypothalamus — neuroendocrine axes, circadian timing, and homeostasis.
//!
//! Anatomy: the diencephalic homeostat — paraventricular nucleus (CRH /
//! HPA stress cascade), suprachiasmatic nucleus (circadian timing /
//! melatonin gating), ventrolateral preoptic area and lateral hypothalamic
//! orexin/TMN histamine populations (the sleep-wake flip-flop).
//!
//! Mirrors `python/hypothalamus/`:
//!
//! - `hpa_development` — developmental maturation and epigenetic tuning
//!   of HPA glucocorticoid feedback (`daemon::hpa_development`).
//! - `sleep_network` — the Booth / Diniz Behn sleep-wake flip-flop and
//!   Serkh–Forger SCN circadian oscillator (`state::sleep_network`).
//! - `neurochemical` — endocrine/neuropeptide release, ACTH cascade, and
//!   allostatic cortisol dynamics (`state::neurochemical`).

pub use crate::daemon::hpa_development;
pub use crate::state::neurochemical;
pub use crate::state::sleep_network;
