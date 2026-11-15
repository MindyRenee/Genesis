//! Vision — the eye as mathematics, inside the body's self-schema.
//!
//! Where [`crate::occipital_lobe`] holds the afferent *nerve* (the
//! transport that carries frames into the body and the subcortical
//! channels it fans out to), this module is the organ itself: a
//! continuous mathematical field that turns spectral radiance into
//! cortical activity with no sensor required. Every stage is the
//! biology's own equation, not an image-processing recipe:
//!
//! - [`optics`] — the dioptric front end: pupil aperture (pretectal
//!   light reflex on log-compressed luminance), ciliary accommodation
//!   (defocus blur), and the voluntary lid — all transmission
//!   scalars on the radiance field before a single photoreceptor
//!   absorbs anything.
//! - [`retina`] — the neural retina: LMS cone absorption
//!   (Stockman–Sharpe fundamentals on an RGB→LMS basis), Naka–Rushton
//!   adaptation (the photoreceptor's compressive nonlinearity), the
//!   cGMP/membrane-potential phototransduction PDE, and ON/OFF
//!   ganglion channels through a center-surround difference of
//!   Gaussians — the lateral inhibition horizontal cells impose.
//! - [`cortex`] — V1: quadrature Gabor energy in orientation
//!   hypercolumns, retinotopically arranged; the population map is
//!   what the oculomotor stage reads salience from.
//! - [`oculomotor`] — the gaze plant: a critically damped spring
//!   driven toward the cortical activity centroid. Seeing *is*
//!   moving the eye; fixation is a motor act, not a camera address.
//!
//! [`VisionSystem`] composes the four into one afferent: feed it a
//! radiance field (any source — a camera frame decomposed to LMS,
//! or a scene) and `dt`; it returns the ganglion image, the V1
//! orientation energy map, and the ocular state the rest of the
//! body can sense. The equations hold whether or not photons ever
//! arrive — a dark field is just a zero boundary condition, and the
//! machinery of seeing remains fully alive.

mod cortex;
mod oculomotor;
mod optics;
mod retina;
mod system;

pub use cortex::{GaborCortex, NUM_ORIENTATIONS};
pub use oculomotor::GazePlant;
pub use optics::{Eyelid, OcularState, Pupil};
pub use retina::{ConeOps, GanglionImage, RadianceField, Retina};
pub use system::{VisionOutput, VisionSystem};
