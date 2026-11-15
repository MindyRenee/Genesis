//! The composed organ — optics → retina → cortex → gaze, one step.
//!
//! [`VisionSystem`] is the afferent whole: each `step` admits the
//! current radiance field through the ocular front end (lid
//! transmission × pupillary gain), transduces it on the retinal
//! mosaic, convolves the ganglion drive through V1's hypercolumns,
//! then moves the fovea toward the cortical salience peak — the
//! sensorimotor loop closed inside one call.
//!
//! The loop is honest about its direction: gaze is a *motor* output
//! that changes what the next frame's retina will sample (through
//! [`GazePlant::sample`]), so callers building a world feed the
//! warped field back in — seeing and looking are the same motion.

use super::{Eyelid, GaborCortex, GazePlant, OcularState, Pupil, RadianceField, Retina};

/// One tick's afferent product: the ganglion report, the cortical
/// map, and the ocular state the body actually held.
#[derive(Clone, Debug)]
pub struct VisionOutput {
    /// Signed center-surround ganglion signal, pre-rectification.
    pub dog: Vec<f64>,
    /// ON/OFF split channels.
    pub ganglion_on: Vec<f64>,
    pub ganglion_off: Vec<f64>,
    /// V1 orientation energy, `NUM_ORIENTATIONS` planes.
    pub cortical: Vec<Vec<f64>>,
    /// Mean retinal luminance — the pretectum's illuminance signal.
    pub mean_luminance: f64,
    /// Ocular state: lid, pupil, gain, defocus.
    pub ocular: OcularState,
    /// Fixation point in field pixels.
    pub gaze: (f64, f64),
}

/// The eye, assembled.
pub struct VisionSystem {
    pub retina: Retina,
    pub cortex: GaborCortex,
    pub gaze: GazePlant,
    pub lid: Eyelid,
    pub pupil: Pupil,
    /// Ciliary accommodation — defocus σ in pixels. Voluntary for
    /// now (the mind's SET_FOCUS analog); 0 is a sharp image.
    pub defocus_sigma: f64,
    /// Sympathetic drive ∈ [0,1] for the pupil's dilator component —
    /// wired to effective norepinephrine by whatever owns the body.
    pub ne_drive: f64,
}

impl VisionSystem {
    /// A retinotopic field of `width × height` loci.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            retina: Retina::new(width, height),
            cortex: GaborCortex::new(width, height),
            gaze: GazePlant::new(width, height),
            lid: Eyelid::new(),
            pupil: Pupil::new(),
            defocus_sigma: 0.0,
            ne_drive: 0.0,
        }
    }

    /// One afferent step over the field.
    ///
    /// Order is the physiology's: the lid admits (blink transient and
    /// voluntary set-point both live), the admitted luminance sets
    /// the pupil's reflex target, the *effective* irradiance —
    /// radiance × lid × pupillary gain — transduces, the ganglion
    /// drive feeds V1, and the cortical peak becomes the next
    /// fixation target. `dt` in seconds; ~5 ms resolves the
    /// transduction pair comfortably.
    pub fn step(&mut self, field: &RadianceField, dt: f64) -> VisionOutput {
        self.lid.step(dt);
        let admitted = self.retina.mean_luminance() * self.lid.transmission();
        self.pupil.step(admitted, self.ne_drive, dt);

        let transmission = self.lid.transmission() * self.pupil.irradiance_gain();
        self.retina.step(field, transmission, dt);

        // Ganglion drive into V1: ON + OFF energy is the full
        // contrast signal the LGN forwards.
        let drive: Vec<f64> = self
            .retina
            .ganglion
            .on
            .iter()
            .zip(&self.retina.ganglion.off)
            .map(|(a, b)| a + b)
            .collect();
        self.cortex.step(&drive, dt);

        // The fovea chases the cortical salience peak.
        let (tx, ty) = self.cortex.activity_centroid();
        self.gaze.saccade_to(tx, ty, dt);

        VisionOutput {
            dog: self.retina.ganglion.dog.clone(),
            ganglion_on: self.retina.ganglion.on.clone(),
            ganglion_off: self.retina.ganglion.off.clone(),
            cortical: self.cortex.activity.clone(),
            mean_luminance: self.retina.mean_luminance(),
            ocular: OcularState {
                lid: self.lid.transmission(),
                pupil_mm: self.pupil.diameter_mm,
                irradiance_gain: self.pupil.irradiance_gain(),
                defocus_sigma: self.defocus_sigma,
            },
            gaze: (self.gaze.x, self.gaze.y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scene: a bright blob off-center in L/M, dim S everywhere.
    fn blob_field(w: usize, h: usize, bx: usize, by: usize) -> RadianceField {
        let mut l = vec![0.05; w * h];
        let mut m = vec![0.05; w * h];
        for y in 0..h {
            for x in 0..w {
                let d2 = (x as f64 - bx as f64).powi(2) + (y as f64 - by as f64).powi(2);
                let v = 2.0 * (-d2 / 8.0).exp();
                l[y * w + x] += v;
                m[y * w + x] += 0.8 * v;
            }
        }
        RadianceField::new(w, h, l, m, vec![0.1; w * h])
    }

    #[test]
    fn the_eye_lives() {
        let mut eye = VisionSystem::new(32, 32);
        let field = blob_field(32, 32, 24, 8);
        let mut out = eye.step(&field, 0.005);
        for _ in 0..400 {
            out = eye.step(&field, 0.005);
        }
        // The retina transduced: luminance above dark rest.
        assert!(out.mean_luminance > 0.0);
        // V1 fired somewhere.
        assert!(out.cortical.iter().flatten().any(|&a| a > 0.0));
        // The fovea chased the blob toward its side of the field.
        assert!(out.gaze.0 > 16.0, "gaze must move toward the salience");
        // The pupil constricted under the admitted light.
        assert!(out.ocular.pupil_mm < 8.0);
    }

    #[test]
    fn lids_dim_the_world() {
        let mut open = VisionSystem::new(16, 16);
        let mut shut = VisionSystem::new(16, 16);
        shut.lid.set_point = 0.0;
        let field = blob_field(16, 16, 4, 4);
        let a = open.step(&field, 0.005).mean_luminance;
        let b = shut.step(&field, 0.005).mean_luminance;
        assert!(
            a > b * 10.0,
            "a closed lid must nearly extinguish the field"
        );
    }
}
