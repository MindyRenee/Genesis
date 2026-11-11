//! Ocular optics — the eye's front end before transduction.
//!
//! Three transmission modulators, each its own dynamics:
//!
//! - The **lids** (`Eyelid`): a voluntary orbicularis oculi held at a
//!   commanded aperture, plus the blink — a brief palpebral closure
//!   the brainstem schedules. The blink is a Gaussian dip in
//!   transmission (~120 ms to close, ~240 ms total), not a step:
//!   real closure sweeps the cornea and the retinal illuminance
//!   follows the lid's area, not a switch.
//! - The **pupil** (`Pupil`): the pretectal light reflex — log-
//!   compressive constriction on retinal illuminance, so the
//!   aperture tracks the log of the light, not the light. Steady
//!   state solves `sphincter = dilator + k·log(L)`; integration is
//!   first-order toward it at the iris muscle's own rate.
//! - **Accommodation** folds in as defocus: the lens state reports a
//!   blur radius so whatever samples the field can convolve at the
//!   PSF the ciliary body actually set.

/// Lid aperture: 1.0 fully open, 0.0 fully closed.
///
/// The voluntary set-point is the mind's (sleep drops it to ~0.15,
/// waking opens it); the blink is a transient occlusion riding on
/// top — the lid can never transmit more than its set-point allows.
#[derive(Clone, Debug)]
pub struct Eyelid {
    /// Commanded aperture, [0, 1]. Voluntary — held until changed.
    pub set_point: f64,
    /// Scheduled blink interval in seconds.
    pub blink_period: f64,
    /// Phase within the blink cycle, [0, blink_period).
    phase: f64,
    /// Time within an active blink, seconds; `None` when open.
    blink_t: Option<f64>,
}

/// One blink's Gaussian closure: total duration and depth.
const BLINK_DURATION_S: f64 = 0.24;
const BLINK_CENTER_S: f64 = 0.12;
const BLINK_WIDTH_S: f64 = 0.05;
const BLINK_DEPTH: f64 = 0.97;

impl Eyelid {
    pub fn new() -> Self {
        Self {
            set_point: 1.0,
            blink_period: 4.0,
            phase: 0.0,
            blink_t: None,
        }
    }

    /// Advance the lid. When the phase crosses the period a blink
    /// starts; inside one, transmission follows a Gaussian dip.
    pub fn step(&mut self, dt: f64) {
        if let Some(t) = self.blink_t {
            let t = t + dt;
            self.blink_t = (t < BLINK_DURATION_S).then_some(t);
        } else {
            self.phase += dt;
            if self.phase >= self.blink_period {
                self.phase -= self.blink_period;
                self.blink_t = Some(0.0);
            }
        }
    }

    /// Current transmission, [0, 1]: the set-point scaled down by any
    /// active blink's Gaussian occlusion.
    pub fn transmission(&self) -> f64 {
        let open = self.set_point.clamp(0.0, 1.0);
        match self.blink_t {
            Some(t) => {
                let dip =
                    BLINK_DEPTH * (-0.5 * ((t - BLINK_CENTER_S) / BLINK_WIDTH_S).powi(2)).exp();
                open * (1.0 - dip)
            }
            None => open,
        }
    }
}

impl Default for Eyelid {
    fn default() -> Self {
        Self::new()
    }
}

/// The iris aperture in millimetres, driven by the light reflex.
///
/// The reflex is logarithmic (Weber–Fechner at the sphincter):
/// the target diameter decreases with log luminance over the
/// mesopic–photopic range, bounded by the anatomy (~2 mm fully
/// constricted, ~8 mm fully dilated). A sympathetic component widens
/// the floor under arousal — carried in as `ne_drive`.
#[derive(Clone, Debug)]
pub struct Pupil {
    /// Diameter in mm.
    pub diameter_mm: f64,
}

/// Anatomical bounds: maximally constricted to maximally dilated.
const PUPIL_MIN_MM: f64 = 2.0;
const PUPIL_MAX_MM: f64 = 8.0;
/// Iris muscle rate constant — the reflex settles over ~300 ms.
const PUPIL_RATE: f64 = 12.0;

impl Pupil {
    pub fn new() -> Self {
        Self { diameter_mm: 5.0 }
    }

    /// Steady-state diameter for a mean luminance `lum` (≥0) and
    /// sympathetic drive `ne` ∈ [0,1]. Log-compressive:
    /// `d* = clamp(d_dark − k·log10(1+L), min, max)`, then the
    /// dilator adds up to its own millimetre at full drive.
    fn target(&self, lum: f64, ne: f64) -> f64 {
        let photopic = PUPIL_MAX_MM - 2.2 * (1.0 + lum.max(0.0)).log10();
        let dilated = photopic + ne.clamp(0.0, 1.0);
        dilated.clamp(PUPIL_MIN_MM, PUPIL_MAX_MM)
    }

    /// First-order approach to the reflex target.
    pub fn step(&mut self, mean_luminance: f64, ne_drive: f64, dt: f64) {
        let target = self.target(mean_luminance, ne_drive);
        self.diameter_mm += (target - self.diameter_mm) * (PUPIL_RATE * dt).min(1.0);
        self.diameter_mm = self.diameter_mm.clamp(PUPIL_MIN_MM, PUPIL_MAX_MM);
    }

    /// Relative retinal irradiance multiplier: pupil *area* scales
    /// the light admitted, so the gain is quadratic in diameter,
    /// normalized so a 4 mm pupil transmits 1.
    pub fn irradiance_gain(&self) -> f64 {
        (self.diameter_mm / 4.0).powi(2)
    }
}

impl Default for Pupil {
    fn default() -> Self {
        Self::new()
    }
}

/// The whole ocular front end at an instant: lid transmission,
/// pupil diameter, and the defocus the ciliary body dialed in.
#[derive(Clone, Copy, Debug)]
pub struct OcularState {
    /// Lid transmission scalar ∈ [0,1].
    pub lid: f64,
    /// Pupil diameter, mm.
    pub pupil_mm: f64,
    /// Pupillary irradiance gain relative to a 4 mm pupil.
    pub irradiance_gain: f64,
    /// Accommodation defocus, Gaussian σ in field pixels (0 = sharp).
    pub defocus_sigma: f64,
}
