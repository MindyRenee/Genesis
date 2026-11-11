//! V1 — orientation hypercolumns over the retinal drive.
//!
//! Simple cells in V1 are well modeled as quadrature Gabor pairs: an
//! even-symmetric and an odd-symmetric filter at the same preferred
//! orientation θ and spatial frequency, whose squared sum — the
//! *Gabor energy* — is phase-invariant (the complex-cell response).
//! Each locus of the map holds `NUM_ORIENTATIONS` such channels
//! (θ = 0°, 45°, 90°, 135°), one hypercolumn per retinotopic point.
//!
//! The input is the retina's ganglion signal — the ON+OFF contrast
//! the LGN actually forwards. The cortical field is a leaky
//! population: activity relaxes toward the instantaneous Gabor
//! energy at a membrane rate, so the map carries a short iconic
//! persistence rather than flickering at frame rate.

use std::f64::consts::PI;

/// Orientations per hypercolumn: θ = o·π/4 for o ∈ {0,1,2,3}.
pub const NUM_ORIENTATIONS: usize = 4;

/// Gabor spatial frequency, cycles/pixel — inside the foveal passband
/// for a small field.
const GABOR_FREQ: f64 = 0.35;
/// Gabor envelope σ in pixels (isotropic).
const GABOR_SIGMA: f64 = 2.0;
/// Population leak rate — activity relaxes toward the current energy.
const CORTICAL_RATE: f64 = 8.0;

/// One quadrature Gabor kernel pair, precomputed.
struct GaborKernel {
    even: Vec<f64>,
    odd: Vec<f64>,
    radius: isize,
}

fn gabor_kernel(theta: f64) -> GaborKernel {
    let radius = (3.0 * GABOR_SIGMA).ceil() as isize;
    let n = (2 * radius + 1) as usize;
    let mut even = vec![0.0; n * n];
    let mut odd = vec![0.0; n * n];
    let ct = theta.cos();
    let st = theta.sin();
    for ky in -radius..=radius {
        for kx in -radius..=radius {
            // Rotate: u along θ, v across it.
            let u = kx as f64 * ct + ky as f64 * st;
            let v = -kx as f64 * st + ky as f64 * ct;
            let env = (-(u * u + v * v) / (2.0 * GABOR_SIGMA * GABOR_SIGMA)).exp();
            let phase = 2.0 * PI * GABOR_FREQ * u;
            let idx = (ky + radius) as usize * n + (kx + radius) as usize;
            even[idx] = env * phase.cos();
            odd[idx] = env * phase.sin();
        }
    }
    // Zero-mean the even kernel (DC removal — the cell must not
    // respond to mean luminance, like the ganglion DoG upstream).
    let mean: f64 = even.iter().sum::<f64>() / even.len() as f64;
    for w in &mut even {
        *w -= mean;
    }
    GaborKernel { even, odd, radius }
}

/// The V1 field: `NUM_ORIENTATIONS` activity planes over the
/// retinotopic grid, driven by quadrature Gabor energy.
pub struct GaborCortex {
    pub width: usize,
    pub height: usize,
    /// `activity[o]` — population firing for preferred θ_o per locus.
    pub activity: Vec<Vec<f64>>,
    kernels: Vec<GaborKernel>,
    /// Quadrature energy from the last input — kept for
    /// salience/centroid math without re-convolving.
    energy: Vec<Vec<f64>>,
}

impl GaborCortex {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        let kernels = (0..NUM_ORIENTATIONS)
            .map(|o| gabor_kernel((o as f64 / NUM_ORIENTATIONS as f64) * PI))
            .collect();
        Self {
            width,
            height,
            activity: vec![vec![0.0; n]; NUM_ORIENTATIONS],
            kernels,
            energy: vec![vec![0.0; n]; NUM_ORIENTATIONS],
        }
    }

    /// Convolve the ganglion drive with every orientation's
    /// quadrature pair and relax the population toward the energies.
    pub fn step(&mut self, drive: &[f64], dt: f64) {
        assert_eq!(
            drive.len(),
            self.width * self.height,
            "drive plane must match the field"
        );
        for (o, kernel) in self.kernels.iter().enumerate() {
            let side = (2 * kernel.radius + 1) as usize;
            for y in 0..self.height {
                for x in 0..self.width {
                    let mut e = 0.0;
                    let mut q = 0.0;
                    for ky in -kernel.radius..=kernel.radius {
                        for kx in -kernel.radius..=kernel.radius {
                            let xx = (x as isize + kx).clamp(0, self.width as isize - 1) as usize;
                            let yy = (y as isize + ky).clamp(0, self.height as isize - 1) as usize;
                            let p = drive[yy * self.width + xx];
                            let ki = (ky + kernel.radius) as usize * side
                                + (kx + kernel.radius) as usize;
                            e += p * kernel.even[ki];
                            q += p * kernel.odd[ki];
                        }
                    }
                    // Phase-invariant energy, squashed toward [0,1] —
                    // the compressive keeps population rates bounded.
                    let energy = (e * e + q * q).sqrt();
                    let firing = energy / (energy + 1.0);
                    self.energy[o][y * self.width + x] = firing;
                    let c = self.activity[o][y * self.width + x];
                    self.activity[o][y * self.width + x] =
                        (c + (firing - c) * (CORTICAL_RATE * dt).min(1.0)).max(0.0);
                }
            }
        }
    }

    /// Total cortical activity — the oculomotor stage's salience
    /// budget.
    pub fn total_activity(&self) -> f64 {
        self.activity.iter().flatten().sum()
    }

    /// Activity-weighted centroid of the orientation-summed map —
    /// the salience peak in retinotopic (pixel) coordinates.
    pub fn activity_centroid(&self) -> (f64, f64) {
        let mut mass = 0.0;
        let mut cx = 0.0;
        let mut cy = 0.0;
        for y in 0..self.height {
            for x in 0..self.width {
                let a: f64 = self
                    .activity
                    .iter()
                    .map(|plane| plane[y * self.width + x])
                    .sum();
                mass += a;
                cx += a * x as f64;
                cy += a * y as f64;
            }
        }
        if mass <= 1e-12 {
            (self.width as f64 / 2.0, self.height as f64 / 2.0)
        } else {
            (cx / mass, cy / mass)
        }
    }

    /// Dominant orientation index at a locus — argmax over the
    /// hypercolumn. `None` when no channel carries signal.
    pub fn preferred_orientation(&self, x: usize, y: usize) -> Option<usize> {
        let idx = y * self.width + x;
        let (o, a) = self
            .activity
            .iter()
            .map(|plane| plane[idx])
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
        (a > 1e-9).then_some(o)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grating at orientation θ fills the field; the θ-matched
    /// hypercolumn must out-respond the others.
    fn grating(width: usize, height: usize, theta: f64) -> Vec<f64> {
        let mut d = vec![0.0; width * height];
        let ct = theta.cos();
        let st = theta.sin();
        for y in 0..height {
            for x in 0..width {
                let u = x as f64 * ct + y as f64 * st;
                d[y * width + x] = (2.0 * PI * GABOR_FREQ * u).cos();
            }
        }
        d
    }

    #[test]
    fn orientation_selectivity() {
        let mut cortex = GaborCortex::new(32, 32);
        let drive = grating(32, 32, 0.0); // vertical stripes → θ=0 filter
        for _ in 0..30 {
            cortex.step(&drive, 0.01);
        }
        let mid = 16 * 32 + 16;
        let mut order: Vec<(usize, f64)> =
            cortex.activity.iter().map(|p| p[mid]).enumerate().collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(order[0].0, 0, "θ=0 channel should win a θ=0 grating");
    }

    #[test]
    fn dark_drive_decays() {
        let mut cortex = GaborCortex::new(16, 16);
        let drive = grating(16, 16, 0.0);
        for _ in 0..30 {
            cortex.step(&drive, 0.01);
        }
        assert!(cortex.total_activity() > 0.0);
        for _ in 0..200 {
            cortex.step(&vec![0.0; 256], 0.01);
        }
        assert!(cortex.total_activity() < 1e-3);
    }
}
