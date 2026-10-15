//! Learned phase model — mental phases as latent regimes inferred from
//! the neurochemical trajectory, not boundaries drawn by hand.
//!
//! A Gaussian hidden Markov model over the effective levels of the
//! eighteen neurochemicals. Each hidden state is a regime with its own
//! mean and diagonal variance; the transition matrix says how regimes
//! persist and how they switch. Nothing about dwell time or the number
//! of regimes is set by hand:
//!
//! - **Persistence** comes from the transition matrix. A fixed Dirichlet
//!   pseudo-count on self-transitions keeps regimes from chattering. It
//!   is a finite-state stand-in for the sticky HDP-HMM of Fox et al.
//!   (2008), which is nonparametric and sampled; it is not that model.
//! - **The number of regimes** is chosen by BIC over `1..=MAX_PHASES`,
//!   each fit from several random starts by Baum–Welch. This is known to
//!   overselect when the trajectory is a continuous drift and the emission
//!   model is misspecified, which is what the 48-hour measurement showed.
//!   Cross-validated likelihood, which Celeux & Durand (2008) recommend
//!   for this situation, is not yet implemented.
//! - **Online labelling** is the forward filter: the belief over
//!   regimes after each tick, and the phase is its argmax.
//!
//! Fitting uses scaled forward–backward, so long trajectories do not
//! underflow. A refit aligns the new regimes to the old ones by mean,
//! so a phase keeps its index across refits when the regime it names
//! still exists.
//!
//! Persisted as a checksummed sidecar (see [`PhaseModel::to_bytes`]).
//! The core state keeps only the current index, in `emergent_phase`,
//! so the state schema is unchanged.

use std::f64::consts::PI;

use super::core_state::crc32_ieee;
use super::neurochemical::NEUROCHEMICAL_COUNT as D;

/// Upper bound on regimes. The `emergent_phase` byte indexes them.
pub const MAX_PHASES: usize = 8;

/// BIC needs at least this many samples per regime to be trusted.
const MIN_SAMPLES_PER_REGIME: usize = 40;
/// Floor on each regime's per-dimension variance, so a regime cannot
/// collapse onto a single sample.
const VAR_FLOOR: f64 = 1e-4;
/// Pseudo-count added to each self-transition in the M-step.
const STICKY_KAPPA: f64 = 5.0;
const EM_ITERS: usize = 100;
const EM_TOL: f64 = 1e-7;
const RESTARTS: u64 = 3;

const MAGIC: [u8; 4] = *b"GPHM";
const FORMAT_VERSION: u32 = 2;

/// One sample: the effective level of each neurochemical.
pub type Point = [f32; D];
type Vector = [f64; D];

/// Why a sidecar could not be read back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionError {
    Truncated,
    BadMagic,
    BadVersion,
    BadChecksum,
    Inconsistent,
}

/// A fitted Gaussian HMM plus the online filter state.
#[derive(Clone, Debug, PartialEq)]
pub struct PhaseModel {
    initial: Vec<f64>,
    transition: Vec<Vec<f64>>,
    mean: Vec<Vector>,
    var: Vec<Vector>,
    belief: Vec<f64>,
    current: usize,
}

impl PhaseModel {
    /// Number of regimes.
    pub fn k(&self) -> usize {
        self.initial.len()
    }

    /// The current phase index.
    pub fn current(&self) -> usize {
        self.current
    }

    /// Mean of regime `j` (its typical neurochemical profile).
    pub fn mean(&self, j: usize) -> Option<&Vector> {
        self.mean.get(j)
    }

    /// Transition probability from regime `i` to `j`.
    pub fn transition(&self, i: usize, j: usize) -> f64 {
        self.transition[i][j]
    }

    /// Fold one sample into the filter and return the current phase.
    ///
    /// A non-finite sample is ignored outright: it would poison the
    /// belief for every later tick.
    pub fn observe(&mut self, x: &Point) -> usize {
        if !x.iter().all(|v| v.is_finite()) {
            return self.current;
        }
        self.belief = self.advance(&self.belief, &to_vector(x));
        self.current = argmax(&self.belief);
        self.current
    }

    /// One forward-filter step: predict through the transition matrix,
    /// weight by how well each regime explains `x`, normalize in log
    /// space.
    fn advance(&self, belief: &[f64], x: &Vector) -> Vec<f64> {
        let k = self.k();
        let logw: Vec<f64> = (0..k)
            .map(|j| {
                let pred: f64 = (0..k).map(|i| belief[i] * self.transition[i][j]).sum();
                pred.max(f64::MIN_POSITIVE).ln() + log_gaussian(&self.mean[j], &self.var[j], x)
            })
            .collect();
        normalize_log(&logw)
    }

    /// Filter a whole sequence from the initial distribution.
    fn filtered_belief(&self, xs: &[Vector]) -> Vec<f64> {
        xs.iter()
            .fold(self.initial.clone(), |b, x| self.advance(&b, x))
    }

    /// Refit to a window of recent samples. Keeps the old phase indices
    /// where a regime still exists. Returns false when there is not
    /// enough data to fit, in which case the model is unchanged.
    pub fn refit(&mut self, samples: &[Point]) -> bool {
        let Some(mut fresh) = fit(samples) else {
            return false;
        };
        align_labels(&mut fresh, self);
        let xs = finite_vectors(samples);
        fresh.belief = fresh.filtered_belief(&xs);
        fresh.current = argmax(&fresh.belief);
        *self = fresh;
        true
    }

    /// Serialize as
    /// `MAGIC | version | k | current | initial | transition | mean | var | belief | crc32`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let k = self.k();
        let mut out = Vec::with_capacity(16 + 8 * (k + k * k + 3 * k * D) + 4);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(k as u32).to_le_bytes());
        out.extend_from_slice(&(self.current as u32).to_le_bytes());
        let flat = self
            .initial
            .iter()
            .chain(self.transition.iter().flatten())
            .chain(self.mean.iter().flatten())
            .chain(self.var.iter().flatten())
            .chain(self.belief.iter());
        for v in flat {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let crc = crc32_ieee(&out);
        out.extend_from_slice(&crc.to_le_bytes());
        out
    }

    /// Parse a sidecar, rejecting anything that fails the checksum or
    /// describes a model that cannot exist.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PartitionError> {
        let (body, tail) = bytes
            .split_last_chunk::<4>()
            .ok_or(PartitionError::Truncated)?;
        if body.get(..4) != Some(&MAGIC[..]) {
            return Err(PartitionError::BadMagic);
        }
        if u32::from_le_bytes(take(body, 4)?) != FORMAT_VERSION {
            return Err(PartitionError::BadVersion);
        }
        if u32::from_le_bytes(*tail) != crc32_ieee(body) {
            return Err(PartitionError::BadChecksum);
        }
        let mut r = Reader { buf: body, at: 8 };
        let k = r.u32()? as usize;
        let current = r.u32()? as usize;
        if !(1..=MAX_PHASES).contains(&k) || current >= k {
            return Err(PartitionError::Inconsistent);
        }
        let initial = r.vec(k)?;
        let transition = (0..k).map(|_| r.vec(k)).collect::<Result<Vec<_>, _>>()?;
        let mean = (0..k).map(|_| r.vector()).collect::<Result<Vec<_>, _>>()?;
        let var = (0..k).map(|_| r.vector()).collect::<Result<Vec<_>, _>>()?;
        let belief = r.vec(k)?;
        if r.at != body.len() {
            return Err(PartitionError::Inconsistent);
        }
        let model = Self {
            initial,
            transition,
            mean,
            var,
            belief,
            current,
        };
        if model.is_valid() {
            Ok(model)
        } else {
            Err(PartitionError::Inconsistent)
        }
    }

    fn is_valid(&self) -> bool {
        let stochastic = |row: &[f64]| {
            row.iter().all(|v| v.is_finite() && *v >= 0.0)
                && (row.iter().sum::<f64>() - 1.0).abs() < 1e-6
        };
        stochastic(&self.initial)
            && stochastic(&self.belief)
            && self.transition.iter().all(|r| stochastic(r))
            && self.mean.iter().flatten().all(|v| v.is_finite())
            && self
                .var
                .iter()
                .flatten()
                .all(|v| v.is_finite() && *v >= VAR_FLOOR)
    }
}

/// Fit a regime model to `samples`, choosing the regime count by BIC.
/// Returns `None` when there is not enough data for even one regime.
pub fn fit(samples: &[Point]) -> Option<PhaseModel> {
    let xs = finite_vectors(samples);
    let n = xs.len();
    let mut best: Option<(f64, PhaseModel)> = None;
    for k in 1..=MAX_PHASES {
        if n < MIN_SAMPLES_PER_REGIME * k {
            break;
        }
        let (model, ll) = (0..RESTARTS)
            .map(|r| em(&xs, k, seed(k, r)))
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        let params = (k - 1) + k * (k - 1) + 2 * k * D;
        let bic = -2.0 * ll + params as f64 * (n as f64).ln();
        if best.as_ref().is_none_or(|(b, _)| bic < *b) {
            best = Some((bic, model));
        }
    }
    best.map(|(_, m)| m)
}

/// Baum–Welch from one random start. Returns the fitted model and its
/// log-likelihood.
fn em(xs: &[Vector], k: usize, seed: u64) -> (PhaseModel, f64) {
    let mut rng = Rng::new(seed);
    let global = global_variance(xs);
    let mut means: Vec<Vector> = Vec::with_capacity(k);
    while means.len() < k {
        let c = xs[rng.below(xs.len())];
        if !means.contains(&c) {
            means.push(c);
        }
    }
    let off = if k > 1 { 0.1 / (k - 1) as f64 } else { 0.0 };
    let mut model = PhaseModel {
        initial: vec![1.0 / k as f64; k],
        transition: (0..k)
            .map(|i| {
                (0..k)
                    .map(|j| {
                        if i == j {
                            1.0 - off * (k - 1) as f64
                        } else {
                            off
                        }
                    })
                    .collect()
            })
            .collect(),
        mean: means,
        var: vec![global; k],
        belief: vec![1.0 / k as f64; k],
        current: 0,
    };
    let mut prev = f64::NEG_INFINITY;
    let mut ll = f64::NEG_INFINITY;
    for _ in 0..EM_ITERS {
        let logb: Vec<Vec<f64>> = xs
            .iter()
            .map(|x| {
                (0..k)
                    .map(|j| log_gaussian(&model.mean[j], &model.var[j], x))
                    .collect()
            })
            .collect();
        let (e_ll, gamma, xi) = forward_backward(&model.initial, &model.transition, &logb);
        ll = e_ll;
        if (ll - prev).abs() < EM_TOL * (1.0 + ll.abs()) {
            break;
        }
        prev = ll;
        m_step(&mut model, xs, &gamma, &xi);
    }
    model.belief = model.initial.clone();
    model.current = argmax(&model.belief);
    (model, ll)
}

/// Scaled forward–backward. Returns the log-likelihood, the per-tick
/// regime posteriors `gamma`, and the expected transition counts `xi`.
///
/// Each tick's emissions are shifted by their maximum before
/// exponentiating, and the forward pass renormalizes every step; the
/// log-likelihood accumulates the shifts and scales exactly, so nothing
/// underflows however long the trajectory.
fn forward_backward(
    initial: &[f64],
    trans: &[Vec<f64>],
    logb: &[Vec<f64>],
) -> (f64, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let n = logb.len();
    let k = initial.len();
    let mut ll = 0.0;
    let mut b = vec![vec![0.0; k]; n];
    for t in 0..n {
        let m = logb[t].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        ll += m;
        for j in 0..k {
            b[t][j] = (logb[t][j] - m).exp();
        }
    }
    let mut alpha = vec![vec![0.0; k]; n];
    let mut c = vec![0.0; n];
    for j in 0..k {
        alpha[0][j] = initial[j] * b[0][j];
    }
    c[0] = alpha[0].iter().sum::<f64>().max(f64::MIN_POSITIVE);
    alpha[0].iter_mut().for_each(|v| *v /= c[0]);
    for t in 1..n {
        for j in 0..k {
            let pred: f64 = (0..k).map(|i| alpha[t - 1][i] * trans[i][j]).sum();
            alpha[t][j] = pred * b[t][j];
        }
        c[t] = alpha[t].iter().sum::<f64>().max(f64::MIN_POSITIVE);
        alpha[t].iter_mut().for_each(|v| *v /= c[t]);
    }
    ll += c.iter().map(|v| v.ln()).sum::<f64>();

    let mut beta = vec![vec![1.0; k]; n];
    for t in (0..n.saturating_sub(1)).rev() {
        for i in 0..k {
            beta[t][i] = (0..k)
                .map(|j| trans[i][j] * b[t + 1][j] * beta[t + 1][j])
                .sum::<f64>()
                / c[t + 1];
        }
    }

    let gamma: Vec<Vec<f64>> = (0..n)
        .map(|t| normalize(&(0..k).map(|i| alpha[t][i] * beta[t][i]).collect::<Vec<_>>()))
        .collect();
    let mut xi = vec![vec![0.0; k]; k];
    for t in 0..n.saturating_sub(1) {
        let mut pair = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                pair[i][j] = alpha[t][i] * trans[i][j] * b[t + 1][j] * beta[t + 1][j];
            }
        }
        let total: f64 = pair.iter().flatten().sum::<f64>().max(f64::MIN_POSITIVE);
        for i in 0..k {
            for j in 0..k {
                xi[i][j] += pair[i][j] / total;
            }
        }
    }
    (ll, gamma, xi)
}

/// Re-estimate parameters from the posteriors, with the sticky prior
/// on self-transitions.
fn m_step(model: &mut PhaseModel, xs: &[Vector], gamma: &[Vec<f64>], xi: &[Vec<f64>]) {
    let k = model.k();
    model.initial = gamma[0].clone();
    for (i, xi_row) in xi.iter().enumerate() {
        let mut row: Vec<f64> = xi_row
            .iter()
            .enumerate()
            .map(|(j, x)| x + if i == j { STICKY_KAPPA } else { 0.0 })
            .collect();
        let s: f64 = row.iter().sum();
        row.iter_mut().for_each(|v| *v /= s);
        model.transition[i] = row;
    }
    for j in 0..k {
        let g: f64 = gamma.iter().map(|gt| gt[j]).sum::<f64>().max(1e-12);
        let mut mean = [0.0; D];
        for (x, gt) in xs.iter().zip(gamma) {
            for d in 0..D {
                mean[d] += gt[j] * x[d];
            }
        }
        mean.iter_mut().for_each(|v| *v /= g);
        let mut var = [0.0; D];
        for (x, gt) in xs.iter().zip(gamma) {
            for d in 0..D {
                let e = x[d] - mean[d];
                var[d] += gt[j] * e * e;
            }
        }
        var.iter_mut().for_each(|v| *v = (*v / g).max(VAR_FLOOR));
        model.mean[j] = mean;
        model.var[j] = var;
    }
}

/// Reorder `new`'s regimes so each one takes the index of the nearest
/// unclaimed regime in `old`. Regimes with no counterpart are appended.
fn align_labels(new: &mut PhaseModel, old: &PhaseModel) {
    let k = new.k();
    let mut used = vec![false; k];
    let mut order = Vec::with_capacity(k);
    for o in 0..old.k().min(k) {
        let best = (0..k)
            .filter(|&j| !used[j])
            .min_by(|&a, &b| {
                sq_dist(&new.mean[a], &old.mean[o]).total_cmp(&sq_dist(&new.mean[b], &old.mean[o]))
            })
            .expect("an unclaimed regime exists while o < k");
        used[best] = true;
        order.push(best);
    }
    order.extend((0..k).filter(|&j| !used[j]));
    new.initial = order.iter().map(|&p| new.initial[p]).collect();
    new.transition = order
        .iter()
        .map(|&p| order.iter().map(|&q| new.transition[p][q]).collect())
        .collect();
    new.mean = order.iter().map(|&p| new.mean[p]).collect();
    new.var = order.iter().map(|&p| new.var[p]).collect();
}

fn log_gaussian(mean: &Vector, var: &Vector, x: &Vector) -> f64 {
    (0..D)
        .map(|d| {
            let e = x[d] - mean[d];
            -0.5 * ((2.0 * PI * var[d]).ln() + e * e / var[d])
        })
        .sum()
}

fn normalize_log(logw: &[f64]) -> Vec<f64> {
    let m = logw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    normalize(&logw.iter().map(|l| (l - m).exp()).collect::<Vec<_>>())
}

fn normalize(v: &[f64]) -> Vec<f64> {
    let s: f64 = v.iter().sum::<f64>().max(f64::MIN_POSITIVE);
    v.iter().map(|x| x / s).collect()
}

fn argmax(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn sq_dist(a: &Vector, b: &Vector) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn to_vector(p: &Point) -> Vector {
    p.map(f64::from)
}

fn finite_vectors(samples: &[Point]) -> Vec<Vector> {
    samples
        .iter()
        .filter(|p| p.iter().all(|v| v.is_finite()))
        .map(to_vector)
        .collect()
}

fn global_variance(xs: &[Vector]) -> Vector {
    let n = xs.len().max(1) as f64;
    let mut mean = [0.0; D];
    for x in xs {
        for d in 0..D {
            mean[d] += x[d] / n;
        }
    }
    let mut var = [0.0; D];
    for x in xs {
        for d in 0..D {
            var[d] += (x[d] - mean[d]).powi(2) / n;
        }
    }
    var.map(|v| v.max(VAR_FLOOR))
}

fn seed(k: usize, restart: u64) -> u64 {
    (k as u64) * 1_000 + restart
}

fn take<const N: usize>(body: &[u8], at: usize) -> Result<[u8; N], PartitionError> {
    body.get(at..at + N)
        .and_then(|s| s.try_into().ok())
        .ok_or(PartitionError::Truncated)
}

struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn array<const N: usize>(&mut self) -> Result<[u8; N], PartitionError> {
        let v = take::<N>(self.buf, self.at)?;
        self.at += N;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32, PartitionError> {
        Ok(u32::from_le_bytes(self.array()?))
    }
    fn f64(&mut self) -> Result<f64, PartitionError> {
        Ok(f64::from_le_bytes(self.array()?))
    }
    fn vec(&mut self, n: usize) -> Result<Vec<f64>, PartitionError> {
        (0..n).map(|_| self.f64()).collect()
    }
    fn vector(&mut self) -> Result<Vector, PartitionError> {
        let mut v = [0.0; D];
        for x in &mut v {
            *x = self.f64()?;
        }
        Ok(v)
    }
}

/// Small deterministic generator for start points. Fits must be
/// reproducible: the same trajectory always yields the same model.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(v: f32) -> Point {
        [v; D]
    }

    fn uniform(rng: &mut Rng) -> f64 {
        (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gauss(rng: &mut Rng) -> f64 {
        let u1 = uniform(rng).max(1e-12);
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * uniform(rng)).cos()
    }

    /// Two-regime sticky sequence: regimes at 0.2 and 0.8, switching
    /// with probability `1 - stay` per tick. Noise is drawn
    /// independently per dimension, as sampling noise is.
    fn two_regime(n: usize, stay: f64, noise: f64, seed: u64) -> Vec<Point> {
        let mut rng = Rng::new(seed);
        let mut regime = 0usize;
        (0..n)
            .map(|_| {
                if uniform(&mut rng) > stay {
                    regime = 1 - regime;
                }
                let centre = if regime == 0 { 0.2 } else { 0.8 };
                let mut p = [0.0f32; D];
                for v in &mut p {
                    *v = (centre + noise * gauss(&mut rng)) as f32;
                }
                p
            })
            .collect()
    }

    fn low_state(m: &PhaseModel) -> usize {
        (0..m.k())
            .min_by(|&a, &b| mean_sum(m, a).total_cmp(&mean_sum(m, b)))
            .unwrap()
    }

    fn mean_sum(m: &PhaseModel, j: usize) -> f64 {
        m.mean(j).unwrap().iter().sum()
    }

    #[test]
    fn bic_recovers_two_sticky_regimes() {
        let m = fit(&two_regime(1500, 0.98, 0.05, 7)).expect("enough data");
        assert_eq!(m.k(), 2);
        let lo = m.mean(low_state(&m)).unwrap();
        let hi = m.mean(1 - low_state(&m)).unwrap();
        assert!(lo.iter().all(|v| (v - 0.2).abs() < 0.05));
        assert!(hi.iter().all(|v| (v - 0.8).abs() < 0.05));
    }

    #[test]
    fn bic_does_not_split_a_single_regime() {
        let m = fit(&two_regime(600, 1.0, 0.05, 11)).expect("enough data");
        assert_eq!(m.k(), 1);
    }

    #[test]
    fn sticky_prior_keeps_self_transitions_dominant() {
        let m = fit(&two_regime(1500, 0.98, 0.05, 3)).expect("enough data");
        for i in 0..m.k() {
            assert!(
                m.transition(i, i) > 0.9,
                "self-transition {i} = {}",
                m.transition(i, i)
            );
        }
    }

    #[test]
    fn filter_follows_a_sustained_regime_change() {
        let mut m = fit(&two_regime(1500, 0.98, 0.05, 5)).expect("enough data");
        let lo = low_state(&m);
        for _ in 0..30 {
            m.observe(&flat(0.2));
        }
        assert_eq!(m.current(), lo);
        let mut switched_at = None;
        for t in 1..=30 {
            if m.observe(&flat(0.8)) != lo {
                switched_at = Some(t);
                break;
            }
        }
        assert!(
            switched_at.is_some_and(|t| t <= 10),
            "switched at {switched_at:?}"
        );
    }

    #[test]
    fn filter_holds_through_an_ambiguous_blip() {
        let mut m = fit(&two_regime(1500, 0.98, 0.05, 9)).expect("enough data");
        let lo = low_state(&m);
        for _ in 0..30 {
            m.observe(&flat(0.2));
        }
        m.observe(&flat(0.5));
        assert_eq!(m.current(), lo);
    }

    #[test]
    fn decisive_single_tick_can_switch_the_belief() {
        let mut m = fit(&two_regime(1500, 0.98, 0.05, 9)).expect("enough data");
        let lo = low_state(&m);
        for _ in 0..30 {
            m.observe(&flat(0.2));
        }
        assert_eq!(m.observe(&flat(0.8)), 1 - lo);
    }

    #[test]
    fn refit_on_the_same_trajectory_keeps_labels() {
        let data = two_regime(1500, 0.98, 0.05, 13);
        let mut m = fit(&data).expect("enough data");
        let before = m.mean(low_state(&m)).copied().unwrap();
        assert!(m.refit(&data));
        let after = m.mean(low_state(&m)).copied().unwrap();
        assert!(sq_dist(&before, &after).sqrt() < 0.05);
        assert_eq!(m.k(), 2);
    }

    #[test]
    fn too_little_data_fits_nothing() {
        assert!(fit(&[]).is_none());
        assert!(fit(&two_regime(20, 0.98, 0.05, 1)).is_none());
    }

    #[test]
    fn non_finite_sample_is_ignored() {
        let mut m = fit(&two_regime(600, 1.0, 0.05, 2)).expect("enough data");
        let before = m.clone();
        let mut bad = flat(0.5);
        bad[4] = f32::NAN;
        assert_eq!(m.observe(&bad), before.current());
        assert_eq!(m, before);
    }

    #[test]
    fn long_trajectory_stays_normalized() {
        let m = fit(&two_regime(5000, 0.99, 0.05, 17)).expect("enough data");
        for row in &m.transition {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        }
        assert!((m.belief.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sidecar_round_trips() {
        let m = fit(&two_regime(1500, 0.98, 0.05, 21)).expect("enough data");
        assert_eq!(PhaseModel::from_bytes(&m.to_bytes()).unwrap(), m);
    }

    #[test]
    fn corrupted_sidecar_is_rejected() {
        let m = fit(&two_regime(600, 1.0, 0.05, 23)).expect("enough data");
        let mut bytes = m.to_bytes();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0x01;
        assert_eq!(
            PhaseModel::from_bytes(&bytes),
            Err(PartitionError::BadChecksum)
        );
    }

    #[test]
    fn truncated_and_foreign_bytes_are_rejected() {
        assert_eq!(
            PhaseModel::from_bytes(&[1, 2, 3]),
            Err(PartitionError::Truncated)
        );
        let m = fit(&two_regime(600, 1.0, 0.05, 29)).expect("enough data");
        let mut bytes = m.to_bytes();
        bytes[0] = b'X';
        assert_eq!(
            PhaseModel::from_bytes(&bytes),
            Err(PartitionError::BadMagic)
        );
    }
}
