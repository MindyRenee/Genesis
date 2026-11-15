//! The neural retina — photons to ganglion channels.
//!
//! The pipeline is the biology's order:
//!
//! 1. **Spectral absorption** — an RGB radiance field is projected
//!    onto the L/M/S cone fundamentals. The matrix is a standard
//!    Hunt–Pointer–Estevez-style basis normalized to unit D65; it is
//!    a *basis*, not a calibration, so callers with true LMS planes
//!    can feed [`RadianceField`] directly and skip the projection.
//! 2. **Naka–Rushton adaptation** — each cone class compresses its
//!    quantal catch by `R = R_max · Iⁿ/(Iⁿ + σⁿ)` (n = 1 in the
//!    steady retina; the semisaturation σ sets the operating point).
//!    This is the photoreceptor's gain control: it is why the same
//!    eye works under starlight and noon.
//! 3. **Phototransduction** — the cGMP cascade as a PDE pair: light
//!    drives phosphodiesterase hydrolysis of cGMP, closing cGMP-gated
//!    channels and hyperpolarizing the membrane toward ~−70 mV. The
//!    graded voltage — not a spike — is what bipolar cells read.
//! 4. **Center-surround antagonism** — horizontal cells pool the
//!    surround; the bipolar/ganglion signal is a difference of
//!    Gaussians (center σ = 1 px, surround σ = 3 px). Uniform fields
//!    cancel; edges survive — the retina reports contrast, not
//!    luminance.
//! 5. **ON/OFF rectification** — ganglion cells can't fire negative;
//!    the DoG output splits into ON (depolarizing-center) and OFF
//!    (hyperpolarizing-center) channels, the two parallel pathways
//!    the LGN receives.
//!
//! Integration is forward Euler on the transduction pair — stable
//! for `dt ≤ ~10 ms` given the rate constants below; [`Retina::step`]
//! clamps the states into their physical ranges so a large `dt`
//! degrades gracefully instead of diverging.

/// An incoming spectral field as L, M, S cone-plane radiances.
///
/// All planes are `width × height` row-major, non-negative; the
/// units are arbitrary but shared (the Naka–Rushton σ is expressed
/// in the same units).
#[derive(Clone, Debug)]
pub struct RadianceField {
    pub width: usize,
    pub height: usize,
    pub l: Vec<f64>,
    pub m: Vec<f64>,
    pub s: Vec<f64>,
}

impl RadianceField {
    pub fn new(width: usize, height: usize, l: Vec<f64>, m: Vec<f64>, s: Vec<f64>) -> Self {
        let n = width * height;
        assert_eq!(l.len(), n, "L plane must be width*height");
        assert_eq!(m.len(), n, "M plane must be width*height");
        assert_eq!(s.len(), n, "S plane must be width*height");
        Self {
            width,
            height,
            l,
            m,
            s,
        }
    }

    /// A uniform dark field — the resting boundary condition.
    pub fn dark(width: usize, height: usize) -> Self {
        Self::new(
            width,
            height,
            vec![0.0; width * height],
            vec![0.0; width * height],
            vec![0.0; width * height],
        )
    }

    /// Build LMS planes from an RGB24 frame (`width*height*3`
    /// row-major bytes) via the cone-fundamental projection.
    pub fn from_rgb24(width: usize, height: usize, rgb: &[u8]) -> Self {
        let n = width * height;
        assert_eq!(rgb.len(), n * 3, "RGB frame must be width*height*3 bytes");
        let mut field = Self::dark(width, height);
        for i in 0..n {
            let r = rgb[3 * i] as f64 / 255.0;
            let g = rgb[3 * i + 1] as f64 / 255.0;
            let b = rgb[3 * i + 2] as f64 / 255.0;
            // sRGB → LMS (HPE-like normalized basis). Rows are the
            // L/M/S fundamentals; the small negative cross-terms of
            // the true transform are clipped at zero — a cone cannot
            // absorb a negative photon count.
            field.l[i] = (0.4002 * r + 0.7075 * g - 0.0807 * b).max(0.0);
            field.m[i] = (-0.2263 * r + 1.1653 * g + 0.0457 * b).max(0.0);
            field.s[i] = (0.9182 * b).max(0.0);
        }
        field
    }
}

/// Cone-opponent readouts after adaptation — the two chromatic
/// channels plus the achromatic (L+M) luminance the magnocellular
/// path carries.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConeOps {
    /// L−M (red–green) opponency.
    pub l_minus_m: f64,
    /// S−(L+M) (blue–yellow) opponency.
    pub s_minus_lm: f64,
    /// Achromatic luminance ≈ 0.6·L + 0.4·M.
    pub luminance: f64,
}

/// Naka–Rushton half-saturation, in the field's radiance units.
const SIGMA: f64 = 0.5;
/// Naka–Rushton exponent; 1.0 is the steady-state retina.
const HILL_N: f64 = 1.0;
/// Peak response (mV-ish scale); only relative values matter.
const R_MAX: f64 = 100.0;

/// Resting (dark) cGMP concentration, relative units.
const CGMP_DARK: f64 = 20.0;
/// cGMP hydrolysis gain — how hard light drives the PDE.
const CGMP_HYDROLYSIS: f64 = 0.5;
/// cGMP synthesis rate — guanylate cyclase restoring the dark state.
const CGMP_SYNTHESIS: f64 = 4.0;
/// Channel Hill coefficient / half-activation for the cGMP-gated
/// conductance (K₁/₂ ≈ 15 in the same relative units).
const CGMP_K_HALF: f64 = 15.0;
/// Channel-open fraction at the dark cGMP level — the baseline the
/// light-driven closure is measured against.
const P_OPEN_DARK: f64 = CGMP_DARK * CGMP_DARK * CGMP_DARK
    / (CGMP_DARK * CGMP_DARK * CGMP_DARK + CGMP_K_HALF * CGMP_K_HALF * CGMP_K_HALF);
/// Membrane leak reversal, mV — drives the photoreceptor back to
/// its depolarized dark state (~−40 mV) when channels reopen.
const V_DARK: f64 = -40.0;
const V_LIGHT: f64 = -70.0;
/// Leak rate and light-driven rate for the membrane ODE.
const V_LEAK: f64 = 0.1;
const V_DRIVE: f64 = 5.0;

/// Gaussian blur helpers — separable kernel, reflect-padded edges.
fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let radius = (3.0 * sigma).ceil().max(1.0) as isize;
    let mut k = Vec::with_capacity((2 * radius + 1) as usize);
    let mut sum = 0.0;
    for i in -radius..=radius {
        let w = (-0.5 * (i as f64 / sigma).powi(2)).exp();
        k.push(w);
        sum += w;
    }
    for w in &mut k {
        *w /= sum;
    }
    k
}

fn convolve_gaussian(plane: &[f64], width: usize, height: usize, sigma: f64) -> Vec<f64> {
    let kernel = gaussian_kernel(sigma);
    let radius = (kernel.len() / 2) as isize;
    let mut tmp = vec![0.0; plane.len()];
    let mut out = vec![0.0; plane.len()];
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (k, &w) in kernel.iter().enumerate() {
                let xx = (x as isize + k as isize - radius).clamp(0, width as isize - 1) as usize;
                acc += w * plane[y * width + xx];
            }
            tmp[y * width + x] = acc;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0;
            for (k, &w) in kernel.iter().enumerate() {
                let yy = (y as isize + k as isize - radius).clamp(0, height as isize - 1) as usize;
                acc += w * tmp[yy * width + x];
            }
            out[y * width + x] = acc;
        }
    }
    out
}

/// The ganglion output image: ON and OFF rectified channels plus the
/// signed DoG they were split from — the retina's true report.
#[derive(Clone, Debug, Default)]
pub struct GanglionImage {
    /// Depolarizing-center channel, ≥0.
    pub on: Vec<f64>,
    /// Hyperpolarizing-center channel, ≥0 (positive = contrast in the
    /// OFF direction).
    pub off: Vec<f64>,
    /// Signed center-minus-surround signal before rectification.
    pub dog: Vec<f64>,
}

/// The retinal field: cone states, transduction states, and the
/// ganglion convolver, over a fixed `width × height` mosaic.
pub struct Retina {
    pub width: usize,
    pub height: usize,
    /// Adapted cone responses (post Naka–Rushton), per locus.
    pub cone_l: Vec<f64>,
    pub cone_m: Vec<f64>,
    pub cone_s: Vec<f64>,
    /// cGMP concentration per locus.
    pub cgmp: Vec<f64>,
    /// Photoreceptor membrane potential, mV, per locus.
    pub membrane_mv: Vec<f64>,
    /// Latest ganglion report.
    pub ganglion: GanglionImage,
}

impl Retina {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            width,
            height,
            cone_l: vec![0.0; n],
            cone_m: vec![0.0; n],
            cone_s: vec![0.0; n],
            cgmp: vec![CGMP_DARK; n],
            membrane_mv: vec![V_DARK; n],
            ganglion: GanglionImage::default(),
        }
    }

    /// One transduction step over the whole mosaic.
    ///
    /// `transmission` multiplies the incident radiance — it carries
    /// the lid, pupil, and defocus the optics stage computed, so the
    /// retina only ever sees the light the eye admitted.
    pub fn step(&mut self, field: &RadianceField, transmission: f64, dt: f64) {
        assert_eq!(field.width, self.width);
        assert_eq!(field.height, self.height);
        let n = self.width * self.height;
        let sigma_n = SIGMA.powf(HILL_N);
        for i in 0..n {
            // Absorption modulated by the admitted fraction.
            let raw_l = field.l[i] * transmission;
            let raw_m = field.m[i] * transmission;
            let raw_s = field.s[i] * transmission;

            // Naka–Rushton compressive adaptation per cone class.
            self.cone_l[i] = R_MAX * raw_l.powf(HILL_N) / (raw_l.powf(HILL_N) + sigma_n);
            self.cone_m[i] = R_MAX * raw_m.powf(HILL_N) / (raw_m.powf(HILL_N) + sigma_n);
            self.cone_s[i] = R_MAX * raw_s.powf(HILL_N) / (raw_s.powf(HILL_N) + sigma_n);

            let lum = self.luminance(i);

            // cGMP: synthesis pushes toward the dark level, light-
            // driven PDE hydrolyzes it down.
            let g = self.cgmp[i];
            let dg =
                CGMP_SYNTHESIS * (CGMP_DARK - g) - CGMP_HYDROLYSIS * lum * g / (CGMP_K_HALF + g);
            let g_next = (g + dg * dt).clamp(0.1, 4.0 * CGMP_DARK);
            self.cgmp[i] = g_next;

            // Membrane: in the dark, high cGMP holds the channels
            // open and the cell rests depolarized at V_DARK; light
            // closes the cGMP-gated conductance and hyperpolarizes
            // toward V_LIGHT. The drive is the light-*added* closure
            // relative to the dark open fraction — zero in the dark,
            // so the cell genuinely rests at V_DARK.
            let p_open = g_next.powi(3) / (g_next.powi(3) + CGMP_K_HALF.powi(3));
            let closed_by_light = P_OPEN_DARK - p_open;
            let v = self.membrane_mv[i];
            let dv = V_DRIVE * closed_by_light * (V_LIGHT - v) - V_LEAK * (v - V_DARK);
            self.membrane_mv[i] = (v + dv * dt).clamp(V_LIGHT - 5.0, V_DARK + 5.0);
        }
        self.refresh_ganglion();
    }

    /// Achromatic luminance at one locus — the magnocellular drive.
    pub fn luminance(&self, i: usize) -> f64 {
        0.6 * self.cone_l[i] + 0.4 * self.cone_m[i]
    }

    /// Mean luminance over the mosaic — the pretectum's illuminance.
    pub fn mean_luminance(&self) -> f64 {
        if self.cone_l.is_empty() {
            return 0.0;
        }
        (0..self.cone_l.len())
            .map(|i| self.luminance(i))
            .sum::<f64>()
            / self.cone_l.len() as f64
    }

    /// Cone-opponent signals at one locus.
    pub fn opponent(&self, i: usize) -> ConeOps {
        let lm = self.cone_l[i] + self.cone_m[i];
        ConeOps {
            l_minus_m: self.cone_l[i] - self.cone_m[i],
            s_minus_lm: self.cone_s[i] - 0.5 * lm,
            luminance: self.luminance(i),
        }
    }

    /// Recompute the ganglion report: DoG of the membrane-signal
    /// plane (depolarization above dark rest = contrast), split into
    /// ON and OFF channels.
    fn refresh_ganglion(&mut self) {
        let n = self.width * self.height;
        // The bipolar drive is the *inverted* deviation from dark
        // rest — ON bipolars sign-invert the photoreceptor, so light
        // (hyperpolarization) reads positive: positive = luminance
        // increment, exactly what the ON channel should carry.
        let drive: Vec<f64> = (0..n).map(|i| V_DARK - self.membrane_mv[i]).collect();
        let center = convolve_gaussian(&drive, self.width, self.height, 1.0);
        let surround = convolve_gaussian(&drive, self.width, self.height, 3.0);
        self.ganglion.dog = (0..n).map(|i| center[i] - surround[i]).collect();
        self.ganglion.on = self.ganglion.dog.iter().map(|&d| d.max(0.0)).collect();
        self.ganglion.off = self.ganglion.dog.iter().map(|&d| (-d).max(0.0)).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naka_rushton_saturates() {
        let mut r = Retina::new(4, 4);
        let dim = RadianceField::new(4, 4, vec![0.1; 16], vec![0.1; 16], vec![0.1; 16]);
        let bright = RadianceField::new(4, 4, vec![10.0; 16], vec![10.0; 16], vec![10.0; 16]);
        r.step(&dim, 1.0, 0.005);
        let dim_l = r.cone_l[0];
        r.step(&bright, 1.0, 0.005);
        let bright_l = r.cone_l[0];
        assert!(bright_l > dim_l);
        assert!(bright_l <= R_MAX + 1e-9, "response saturates at R_MAX");
    }

    #[test]
    fn dark_field_returns_to_rest() {
        let mut r = Retina::new(4, 4);
        for _ in 0..200 {
            r.step(&RadianceField::dark(4, 4), 1.0, 0.005);
        }
        assert!((r.cgmp[0] - CGMP_DARK).abs() < 0.5);
        assert!((r.membrane_mv[0] - V_DARK).abs() < 1.0);
    }

    #[test]
    fn uniform_field_cancels_in_dog() {
        let mut r = Retina::new(16, 16);
        let flat = RadianceField::new(16, 16, vec![1.0; 256], vec![1.0; 256], vec![0.2; 256]);
        for _ in 0..100 {
            r.step(&flat, 1.0, 0.005);
        }
        // Interior loci see equal center and surround → ~zero DoG.
        let mid = 8 * 16 + 8;
        assert!(r.ganglion.dog[mid].abs() < 1e-6);
    }

    #[test]
    fn edge_survives_center_surround() {
        let mut r = Retina::new(16, 16);
        let mut l = vec![0.0; 256];
        for y in 0..16 {
            for x in 8..16 {
                l[y * 16 + x] = 2.0;
            }
        }
        let edge = RadianceField::new(16, 16, l.clone(), l, vec![0.0; 256]);
        for _ in 0..100 {
            r.step(&edge, 1.0, 0.005);
        }
        let energy: f64 = r.ganglion.dog.iter().map(|d| d * d).sum();
        assert!(energy > 1e-3, "a luminance step must survive the DoG");
    }
}
