//! Oculomotor — the gaze plant. Seeing is a motor act.
//!
//! The eye is never a passive sensor: fixation is held against the
//! plant's inertia, and a saccade is a ballistic lunge at a salience
//! target. The plant here is a critically damped spring in the
//! field's pixel plane — the standard model of the ocular plant's
//! dominant dynamics (the muscle's overdamped viscoelasticity is why
//! saccades land without oscillating).
//!
//! The target comes from the cortical map: the activity-weighted
//! centroid is the salience peak, and the fovea — the field center —
//! chases it. This is the free-energy account of gaze: the eye moves
//! to put its high-acuity tissue where the world is most uncertain.

/// A 2-D ocular plant: position (gaze point in field pixels) and
/// velocity, driven toward a target by a critically damped spring.
///
/// With ζ = 1 the response is the fastest non-oscillating one —
/// matching the saccade's own ~30–80 ms ballistic profile at the
/// natural ω below.
#[derive(Clone, Debug)]
pub struct GazePlant {
    /// Fixation point in field-pixel coordinates.
    pub x: f64,
    pub y: f64,
    /// Angular velocity proxy, pixels/second.
    pub vx: f64,
    pub vy: f64,
}

/// Natural frequency of the plant. ω = 30 rad/s gives a ~70 ms
/// saccade to a mid-field target — inside the biological range.
const PLANT_OMEGA: f64 = 30.0;
/// Maximum eccentricity as a fraction of field half-extent — the
/// eye cannot roll its fovea past the orbit's edge.
const MAX_ECCENTRIC: f64 = 0.95;

impl GazePlant {
    /// Start fixated on field center.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            x: width as f64 / 2.0,
            y: height as f64 / 2.0,
            vx: 0.0,
            vy: 0.0,
        }
    }

    /// Drive the plant toward `(tx, ty)` for `dt` seconds — exact
    /// integration of the critically damped spring:
    /// `x(t) = t + (d0 + (v0 + ω·d0)·t)·e^(−ωt)`, applied per axis.
    /// The exact update keeps stability at any `dt` — a motor plant
    /// must never overshoot into a numerical oscillation a real
    /// muscle couldn't produce.
    pub fn saccade_to(&mut self, tx: f64, ty: f64, dt: f64) {
        let w = PLANT_OMEGA;
        let e = (-w * dt).exp();
        for (pos, vel, target) in [
            (&mut self.x, &mut self.vx, tx),
            (&mut self.y, &mut self.vy, ty),
        ] {
            let d0 = *pos - target;
            *pos = target + (d0 + (*vel + w * d0) * dt) * e;
            *vel = (*vel - w * (*vel + w * d0) * dt) * e;
        }
    }

    /// Sample a radiance field around the current fixation — the
    /// eye movement's visual consequence made continuous: bilinear
    /// interpolation at `locus + gaze_offset`, where the offset is
    /// fixation minus field center. Sampling the world through the
    /// moved fovea *is* looking.
    pub fn sample(&self, plane: &[f64], width: usize, height: usize) -> Vec<f64> {
        let ox = (self.x - width as f64 / 2.0).clamp(
            -MAX_ECCENTRIC * width as f64 / 2.0,
            MAX_ECCENTRIC * width as f64 / 2.0,
        );
        let oy = (self.y - height as f64 / 2.0).clamp(
            -MAX_ECCENTRIC * height as f64 / 2.0,
            MAX_ECCENTRIC * height as f64 / 2.0,
        );
        let mut out = vec![0.0; width * height];
        for y in 0..height {
            for x in 0..width {
                let sx = (x as f64 + ox).clamp(0.0, width as f64 - 1.0);
                let sy = (y as f64 + oy).clamp(0.0, height as f64 - 1.0);
                let x0 = sx.floor() as usize;
                let y0 = sy.floor() as usize;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;
                out[y * width + x] = plane[y0 * width + x0] * (1.0 - fx) * (1.0 - fy)
                    + plane[y0 * width + x1] * fx * (1.0 - fy)
                    + plane[y1 * width + x0] * (1.0 - fx) * fy
                    + plane[y1 * width + x1] * fx * fy;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saccade_lands_without_oscillation() {
        let mut g = GazePlant::new(32, 32);
        let target = (24.0, 8.0);
        for _ in 0..200 {
            g.saccade_to(target.0, target.1, 0.005);
        }
        assert!((g.x - target.0).abs() < 1e-6 && (g.y - target.1).abs() < 1e-6);
        assert!(g.vx.abs() < 1e-6 && g.vy.abs() < 1e-6);
    }

    #[test]
    fn gaze_shift_moves_the_fovea() {
        // A point source off-center: after fixating it, sampling must
        // bring it toward field center.
        let mut plane = vec![0.0; 32 * 32];
        plane[8 * 32 + 24] = 1.0;
        let mut g = GazePlant::new(32, 32);
        g.x = 24.0;
        g.y = 8.0;
        let sampled = g.sample(&plane, 32, 32);
        // Field center should now hold the source's intensity.
        assert!(sampled[16 * 32 + 16] > 0.9);
    }
}
