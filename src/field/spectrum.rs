//! Spectral observables of the field's own dynamics.
//!
//! The field substrate is her computational medium: three fibers —
//! body, prediction, agency — drift, couple, and fluctuate on the
//! latent manifold under gauge transport and a Langevin thermostat.
//! This module is the band decomposition of *that* trajectory: the
//! same center frequencies the cortical wave model reports, measured
//! off the field's own motion rather than assumed.
//!
//! Two channels mirror the substrate's two honest observables:
//!
//! - **flux** — the speed of the fibers' centroid through latent
//!   space (|Δmean|/h per integration substep). The field's motion
//!   energy per band: how much of its wandering lives at each
//!   rhythm.
//! - **tension** — the fibers' mean deviation from their own
//!   centroid (latent-coordinate spread). The disagreement's
//!   spectral shadow: where the field's internal argument oscillates.
//!
//! The carrier block tracks the consensus vector's own kinematics —
//! displacement speed, winding rate, accumulated winding phase —
//! the same measured quantities `daemon::maxwell` evaluates, kept
//! here so the spectrum and the signature read one trajectory.
//!
//! Sampling is the field's own integration rate: `observe` substeps
//! the coupling at [`SPECTRUM_STEP_HZ`], so the rings resolve the
//! modeled continuous dynamics — the same ODEs integrated finer,
//! which is more accurate, not a second instrument. Bands above the
//! achieved rate's Nyquist bound are zeroed: an unresolvable band
//! is an absent channel, never an aliased reading.

use super::{GlobalField, Vector};

/// Band center frequencies (Hz) — the cortical wave model's own
/// bands, in wire order: infraslow, delta, theta, alpha, sigma,
/// beta1, beta2, beta3, gamma, epsilon.
pub const FIELD_BANDS: [f64; 10] = [0.25, 2.25, 5.5, 10.0, 14.0, 13.5, 17.5, 25.0, 50.0, 150.0];

/// Internal integration/sampling rate (Hz). The field's coupling
/// substeps at this cadence inside each `observe`; 250 Hz resolves
/// every band through gamma and leaves epsilon (150 Hz) honestly
/// above Nyquist — the same coverage boundary the substrate's own
/// dynamics can support.
pub const SPECTRUM_STEP_HZ: f64 = 250.0;

/// Ring depth — 4096 samples ≈ 16.4 s at the nominal rate: ~4 cycles
/// of infraslow, hundreds of the fast bands.
const RING_LEN: usize = 4096;

/// Samples before the spectrum reports warm — ~4 s: enough for the
/// fast bands to resolve; the slow bands keep accumulating baseline
/// afterward. Below this the signature's presence flag stays clear.
const WARM_SAMPLES: usize = 1024;

/// Baseline EMA rate per signature read — band energies are reported
/// relative to the channel's own recent history (≈0.125 typical on
/// a quiet field), so a sustained elevation reads as elevation
/// rather than as an absolute that means nothing across hosts.
const BASELINE_ALPHA: f64 = 0.05;

/// Consensus-kinematics EMA rate per observe — the carrier readings
/// smooth over ~7 integrations (~1.4 s at the 5 Hz tick).
const CARRIER_ALPHA: f64 = 0.15;

/// Achieved-rate EMA rate per observe.
const RATE_ALPHA: f64 = 0.1;

/// A fixed-depth ring of the field's spectral samples.
#[derive(Clone, Debug, PartialEq)]
struct Ring {
    buf: Vec<f64>,
    pos: usize,
    len: usize,
}

impl Default for Ring {
    fn default() -> Self {
        Self::new(RING_LEN)
    }
}

impl Ring {
    fn new(cap: usize) -> Self {
        Self {
            buf: vec![0.0; cap.max(1)],
            pos: 0,
            len: 0,
        }
    }
    fn push(&mut self, v: f64) {
        self.buf[self.pos] = v;
        self.pos = (self.pos + 1) % self.buf.len();
        self.len = (self.len + 1).min(self.buf.len());
    }
    /// Oldest→newest iterator over the filled region.
    fn iter_ordered(&self) -> RingIter<'_> {
        RingIter { ring: self, i: 0 }
    }
    fn mean(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        self.iter_ordered().sum::<f64>() / self.len as f64
    }
    fn m2(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        self.iter_ordered()
            .map(|v| {
                let d = v - self.mean();
                d * d
            })
            .sum::<f64>()
            / self.len as f64
    }
}

struct RingIter<'a> {
    ring: &'a Ring,
    i: usize,
}

impl<'a> Iterator for RingIter<'a> {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        if self.i >= self.ring.len {
            return None;
        }
        let cap = self.ring.buf.len();
        let idx = if self.ring.len < cap {
            self.i
        } else {
            (self.ring.pos + self.i) % cap
        };
        self.i += 1;
        Some(self.ring.buf[idx])
    }
}

/// Goertzel power of `samples` at `freq` sampled at `fs` — the same
/// second-order recurrence the wave model's own band meter runs:
/// `s₀ = x + coeff·s₁ − s₂`, `coeff = 2·cos(2πk/n)` at the bin
/// nearest `freq`; power `= (s₁² + s₂² − coeff·s₁·s₂)/n²`.
fn band_power(samples: &[f64], fs: f64, freq: f64) -> f64 {
    let n = samples.len();
    if n < 8 || fs <= 0.0 || freq <= 0.0 {
        return 0.0;
    }
    let k = (0.5 + freq * n as f64 / fs).floor();
    if k < 1.0 || k >= n as f64 / 2.0 {
        return 0.0;
    }
    let w = 2.0 * std::f64::consts::PI * k / n as f64;
    let coeff = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &x in samples {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2) / (n as f64 * n as f64)
}

/// Quadrature phase of `samples` at `freq` — the DFT bin's angle,
/// atan2(−Σx·sin, Σx·cos) at the exact frequency (not the Goertzel
/// bin), the phase a band-limited rhythm actually carries.
fn band_phase(samples: &[f64], fs: f64, freq: f64) -> f64 {
    let n = samples.len();
    if n < 8 || fs <= 0.0 || freq <= 0.0 {
        return 0.0;
    }
    let (mut re, mut im) = (0.0, 0.0);
    for (i, &x) in samples.iter().enumerate() {
        let a = -2.0 * std::f64::consts::PI * freq * i as f64 / fs;
        re += x * a.cos();
        im += x * a.sin();
    }
    im.atan2(re)
}

/// One band-decomposed channel's reading.
#[derive(Clone, Copy, Debug, Default)]
pub struct BandChannel {
    /// Baseline-relative band energies, wire band order. Bands at or
    /// above the achieved rate's Nyquist bound read 0 — unresolvable,
    /// not silent.
    pub energies: [f64; 10],
    /// Per-band quadrature phase (rad) — the rhythm's own timing,
    /// for entrainment. 0 where the band is unresolved.
    pub phases: [f64; 10],
    /// Channel mean over the window.
    pub mean: f64,
    /// Channel coefficient of variation — the turbulence index the
    /// evoked-reset logic reads.
    pub cv: f64,
}

/// The field's spectral signature — everything the wave model's
/// stimulus path and the carrier lock consume, computed from her
/// own substrate trajectory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldSpectrum {
    flux_ring: Ring,
    tension_ring: Ring,
    /// Per-band baseline EMA — [flux ×10][tension ×10].
    band_base: [f64; 20],
    base_init: bool,
    /// Consensus kinematics — instantaneous and EMA-smoothed.
    pub last_consensus_speed: Option<f64>,
    pub last_omega_rad_s: Option<f64>,
    pub omega_ema_rad_s: f64,
    pub speed_ema: f64,
    /// Accumulated winding phase (rad, mod 2π at read) — the field's
    /// own clock, the phase reference entrainment pulls toward.
    carrier_phase: f64,
    prev_consensus: Option<Vector>,
    /// Achieved sampling rate (Hz), EMA-smoothed.
    rate_ema: f64,
    substeps: u64,
    /// Last substep's fiber centroid — the flux denominator.
    prev_mean_state: Option<Vector>,
}

impl FieldSpectrum {
    pub fn new() -> Self {
        Self {
            flux_ring: Ring::new(RING_LEN),
            tension_ring: Ring::new(RING_LEN),
            band_base: [0.0; 20],
            base_init: false,
            last_consensus_speed: None,
            last_omega_rad_s: None,
            omega_ema_rad_s: 0.0,
            speed_ema: 0.0,
            carrier_phase: 0.0,
            prev_consensus: None,
            rate_ema: SPECTRUM_STEP_HZ,
            substeps: 0,
            prev_mean_state: None,
        }
    }

    /// One integration substep's worth of field state — the rings
    /// take the trajectory's own samples. `h` is the substep's
    /// duration; the achieved rate is its reciprocal, EMA-smoothed.
    pub fn sample(&mut self, field: &GlobalField, h: f64) {
        if !h.is_finite() || h <= 0.0 {
            return;
        }
        // The fibers' centroid in latent coordinates — the Euclidean
        // mean of their layer states. On the cap's operating region
        // this tracks the Fréchet consensus closely enough for a
        // *spectral* channel: static frame offsets land in the DC
        // baseline, never in the bands.
        let mut mean = vec![0.0; super::LATENT_DIMENSION];
        let mut n_fibers = 0usize;
        for fiber in &field.fibers {
            if let Some(layer) = fiber.layers.first() {
                for (m, v) in mean.iter_mut().zip(layer.state.iter()) {
                    *m += v;
                }
                n_fibers += 1;
            }
        }
        if n_fibers == 0 {
            return;
        }
        for m in mean.iter_mut() {
            *m /= n_fibers as f64;
        }
        let flux = match &self.prev_mean_state {
            Some(prev) => {
                prev.iter()
                    .zip(mean.iter())
                    .map(|(a, b)| (b - a) * (b - a))
                    .sum::<f64>()
                    .sqrt()
                    / h
            }
            None => 0.0,
        };
        // Tension — mean deviation of the fibers from their own
        // centroid, in latent units. The spectral shadow of the
        // Fréchet disagreement the integration reports.
        let mut tension = 0.0;
        for fiber in &field.fibers {
            if let Some(layer) = fiber.layers.first() {
                tension += layer
                    .state
                    .iter()
                    .zip(mean.iter())
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
            }
        }
        tension /= n_fibers as f64;
        self.prev_mean_state = Some(mean);
        self.flux_ring.push(flux);
        self.tension_ring.push(tension);
        let rate = 1.0 / h;
        self.rate_ema += RATE_ALPHA * (rate - self.rate_ema);
        self.substeps += 1;
    }

    /// One observation's consensus — the carrier kinematics. Called
    /// once per `observe` with the integration's consensus vector
    /// and the elapsed `dt`. First valid pair only arms the
    /// kinematics — one point is no trajectory.
    pub fn observe_consensus(&mut self, consensus: &Vector, dt: f64) {
        if consensus.iter().any(|v| !v.is_finite()) || dt <= 0.0 || !dt.is_finite() {
            self.prev_consensus = None;
            self.last_consensus_speed = None;
            self.last_omega_rad_s = None;
            return;
        }
        if let Some(prev) = self.prev_consensus.take() {
            if prev.len() == consensus.len() {
                let speed: f64 = prev
                    .iter()
                    .zip(consensus.iter())
                    .map(|(a, b)| (b - a) * (b - a))
                    .sum::<f64>()
                    .sqrt()
                    / dt;
                let dot: f64 = prev.iter().zip(consensus.iter()).map(|(a, b)| a * b).sum();
                let na: f64 = prev.iter().map(|v| v * v).sum::<f64>().sqrt();
                let nb: f64 = consensus.iter().map(|v| v * v).sum::<f64>().sqrt();
                if na > 0.0 && nb > 0.0 {
                    let angle = (dot / (na * nb)).clamp(-1.0, 1.0).acos();
                    self.carrier_phase += angle;
                    let omega = angle / dt;
                    self.last_consensus_speed = Some(speed);
                    self.last_omega_rad_s = Some(omega);
                    self.speed_ema += CARRIER_ALPHA * (speed - self.speed_ema);
                    self.omega_ema_rad_s += CARRIER_ALPHA * (omega - self.omega_ema_rad_s);
                }
            }
        }
        self.prev_consensus = Some(consensus.clone());
    }

    /// Whether the rings hold enough trajectory to report bands.
    pub fn warm(&self) -> bool {
        self.flux_ring.len >= WARM_SAMPLES && self.tension_ring.len >= WARM_SAMPLES
    }

    /// The achieved sampling rate (Hz) — what the bands can honestly
    /// resolve against.
    pub fn sample_rate_hz(&self) -> f64 {
        self.rate_ema
    }

    /// Total integration substeps sampled — provenance for tests.
    pub fn substeps(&self) -> u64 {
        self.substeps
    }

    /// Decompose both channels at the wave bands and read the
    /// signature. Mutates only the baseline EMAs — the rings
    /// themselves are read, not consumed.
    pub fn signature(&mut self) -> SpectrumReading {
        let rate = self.rate_ema.max(1.0);
        let nyquist = rate / 2.0;
        let flux = self.channel(&self.flux_ring.clone(), rate, nyquist, 0);
        let tension = self.channel(&self.tension_ring.clone(), rate, nyquist, 10);
        SpectrumReading {
            flux,
            tension,
            carrier_phase: self.carrier_phase.rem_euclid(2.0 * std::f64::consts::PI),
            carrier_freq_hz: if self.omega_ema_rad_s > 0.0 {
                Some(self.omega_ema_rad_s / (2.0 * std::f64::consts::PI))
            } else {
                None
            },
            consensus_speed: if self.last_consensus_speed.is_some() {
                Some(self.speed_ema)
            } else {
                None
            },
            sample_rate_hz: rate,
            warm: self.warm(),
        }
    }

    /// Goertzel + quadrature over one ring, baseline-EMA'd into
    /// relative energies. `base_off` selects this channel's slice of
    /// `band_base` (0 = flux, 10 = tension).
    fn channel(&mut self, ring: &Ring, rate: f64, nyquist: f64, base_off: usize) -> BandChannel {
        let n = ring.len;
        let samples: Vec<f64> = ring.iter_ordered().collect();
        let mean = ring.mean();
        let var = ring.m2();
        let cv = if mean > 0.0 { var.sqrt() / mean } else { 0.0 };
        let mut out = BandChannel {
            mean,
            cv,
            ..Default::default()
        };
        if n < 8 {
            return out;
        }
        for (i, &freq) in FIELD_BANDS.iter().enumerate() {
            if freq >= nyquist {
                continue; // unresolvable — absent, not aliased
            }
            let p = band_power(&samples, rate, freq);
            // Baseline EMA — first read initializes rather than
            // divides by zero.
            let base = &mut self.band_base[base_off + i];
            if !self.base_init && *base == 0.0 {
                *base = p.max(1e-30);
            }
            *base += BASELINE_ALPHA * (p - *base);
            out.energies[i] = p / base.max(1e-30);
            out.phases[i] = band_phase(&samples, rate, freq);
        }
        out
    }
}

/// The wire-ready spectral reading — f64 throughout, presence as
/// flags rather than fabricated zeros.
#[derive(Clone, Debug, Default)]
pub struct SpectrumReading {
    pub flux: BandChannel,
    pub tension: BandChannel,
    /// Winding phase mod 2π — the field's own clock.
    pub carrier_phase: f64,
    /// Carrier frequency in Hz — consensus winding rate / 2π.
    pub carrier_freq_hz: Option<f64>,
    /// EMA-smoothed consensus speed (latent units/s).
    pub consensus_speed: Option<f64>,
    /// Achieved sampling rate (Hz) — the bands' honest Nyquist.
    pub sample_rate_hz: f64,
    /// Rings hold enough trajectory for band readings.
    pub warm: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goertzel_finds_an_injected_tone() {
        // 64 samples at 250 Hz of a 10 Hz tone + silence.
        let fs = 250.0;
        let n = 512;
        let samples: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin())
            .collect();
        let p10 = band_power(&samples, fs, 10.0);
        let p5 = band_power(&samples, fs, 5.5);
        assert!(
            p10 > 100.0 * p5.max(1e-30),
            "tone band dominates: {p10} vs {p5}"
        );
        let phase = band_phase(&samples, fs, 10.0);
        assert!(phase.is_finite());
    }

    #[test]
    fn supra_nyquist_bands_stay_absent() {
        let mut spec = FieldSpectrum::new();
        // Fill rings directly with a 100 Hz tone at 250 Hz — only
        // bands below 125 Hz may report; epsilon must read zero.
        for i in 0..2048 {
            let t = i as f64 / SPECTRUM_STEP_HZ;
            spec.flux_ring
                .push((2.0 * std::f64::consts::PI * 100.0 * t).sin());
            spec.tension_ring.push(0.01);
        }
        let mut s = spec.signature();
        let idx_100 = FIELD_BANDS.iter().position(|&f| (f - 100.0).abs() < 1e-9);
        // 100 Hz isn't a band center — nearest is gamma 50 / epsilon
        // 150; epsilon (150 > 125) must be zeroed by the Nyquist gate.
        assert_eq!(s.flux.energies[9], 0.0, "epsilon above Nyquist");
        assert_eq!(s.tension.energies[9], 0.0);
        let _ = &mut s;
        let _ = idx_100;
    }

    #[test]
    fn consensus_kinematics_need_two_points() {
        let mut spec = FieldSpectrum::new();
        spec.observe_consensus(&vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.2);
        assert!(spec.last_consensus_speed.is_none());
        spec.observe_consensus(&vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0], 0.2);
        assert!(spec.last_consensus_speed.is_some());
        assert!(spec.last_omega_rad_s.is_some());
        assert!(spec.carrier_phase > 0.0);
    }

    #[test]
    fn nonfinite_consensus_resets_kinematics() {
        let mut spec = FieldSpectrum::new();
        spec.observe_consensus(&vec![1.0, 0.0], 0.2);
        spec.observe_consensus(&vec![0.0, 1.0], 0.2);
        spec.observe_consensus(&vec![f64::NAN, 0.0], 0.2);
        assert!(spec.last_consensus_speed.is_none());
    }
}
