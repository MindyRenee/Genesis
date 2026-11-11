//! Cerebellum — forward prediction, sensorimotor timing, and coordination.
//!
//! Anatomy: the cerebellar cortex and deep nuclei run internal forward
//! models over efference copies of motor commands, comparing predicted
//! consequences against sensory reafference to smooth trajectories and
//! coordinate sub-second timing.
//!
//! Mirrors `python/cerebellum/`:
//!
//! - `motor` — spatial and action motor-field decoding, softmax centroids,
//!   spread, and entropy (`latent_spaces::motor`).
//! - `engine` — continuous-time local Jacobian, damping, saturation, and
//!   skew-symmetric spiral dynamics (`latent_spaces::engine`).
//! - `adaptive` — adaptive interval gates driven by prediction/error demand
//!   (`daemon::adaptive`).

pub use crate::daemon::adaptive;
pub use crate::latent_spaces::engine;
pub use crate::latent_spaces::motor;
