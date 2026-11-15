//! Real analysis — the mathematics of "arbitrarily close."
//!
//! The epsilon-delta definition is the heart of the domain:
//! lim x->a f(x) = L means that for every eps > 0 there is a
//! delta > 0 such that every x inside the punctured
//! delta-neighborhood of a (x != a, |x-a| < delta) lands within
//! eps of L. It is a game the machine can actually play: given
//! eps, produce a delta; given a claimed delta, verify it by
//! sampling the punctured neighborhood. Verification is dense
//! sampling — evidence, not proof, and the functions here say
//! so by returning witnesses when a delta fails.

/// Does `delta` answer `epsilon` for the claim lim x->a f(x) = L?
/// Samples the punctured neighborhood (a - delta, a + delta) \ {a}
/// and returns the worst offender, if any: the x whose image
/// misses the eps-band by the most. None means the delta held
/// at every sample — evidence for the claim, not a proof.
pub fn check_delta(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    big_l: f64,
    epsilon: f64,
    delta: f64,
    samples: usize,
) -> Option<(f64, f64)> {
    if epsilon <= 0.0 || delta <= 0.0 || samples == 0 {
        return None;
    }
    let mut worst: Option<(f64, f64)> = None;
    let mut worst_gap = 0.0f64;
    for i in 0..samples {
        // Even spread over (-delta, delta), skipping the center.
        let off = delta * (2.0 * (i + 1) as f64 / (samples + 1) as f64 - 1.0);
        if off.abs() < 1e-300 {
            continue;
        }
        let x = a + off;
        let gap = (f(x) - big_l).abs();
        if gap >= epsilon && gap > worst_gap {
            worst_gap = gap;
            worst = Some((x, gap));
        }
    }
    worst
}

/// Find a delta answering epsilon — geometric search: shrink
/// from `start` until no sampled point escapes the eps-band.
/// Returns Some(delta) when a sampled-quiet delta is found within
/// `max_halvings` halvings of `start`; None when none held.
pub fn find_delta(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    big_l: f64,
    epsilon: f64,
    start: f64,
    max_halvings: usize,
) -> Option<f64> {
    if !epsilon.is_finite() || epsilon <= 0.0 || !start.is_finite() || start == 0.0 {
        return None;
    }
    let mut delta = start.abs();
    for _ in 0..max_halvings {
        if check_delta(f, a, big_l, epsilon, delta, 2048).is_none() {
            return Some(delta);
        }
        delta /= 2.0;
    }
    None
}

/// Numeric two-sided limit estimate: evaluate f approaching `a`
/// from both sides along a geometric sequence of offsets; returns
/// the last pair (left, right) when they agree within `tol`.
pub fn limit_numeric(f: &dyn Fn(f64) -> f64, a: f64, tol: f64, steps: usize) -> Option<f64> {
    let mut left = 0.0;
    let mut right = 0.0;
    for k in 1..=steps {
        let h = 0.5f64.powi(k as i32);
        left = f(a - h);
        right = f(a + h);
    }
    if (left - right).abs() <= tol {
        Some((left + right) / 2.0)
    } else {
        None // the two sides disagree — no limit here
    }
}

/// Continuity at a point, sampled: f(a) must exist and every
/// nearby x must land within eps of f(a). Same game as the limit
/// except the neighborhood is NOT punctured — the point itself
/// counts, which is exactly why continuity needs f(a) to equal L.
/// Returns the offending (x, gap) if found.
pub fn check_continuous(
    f: &dyn Fn(f64) -> f64,
    a: f64,
    epsilon: f64,
    delta: f64,
    samples: usize,
) -> Option<(f64, f64)> {
    check_delta(&|x| f(x), a, f(a), epsilon, delta, samples)
}

/// Uniform modulus of continuity on [lo, hi]: the least sampled
/// delta answering epsilon at every sampled point of the interval.
/// The distinction is the quantifier order — pointwise lets delta
/// vary with the point; uniform demands one delta for all. None
/// when no delta in the search held everywhere.
pub fn uniform_modulus(
    f: &dyn Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    epsilon: f64,
    points: usize,
    start_delta: f64,
    max_halvings: usize,
) -> Option<f64> {
    if points == 0
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || !start_delta.is_finite()
        || start_delta == 0.0
    {
        return None;
    }
    let mut delta = start_delta.abs();
    'search: for _ in 0..max_halvings {
        for i in 0..=points {
            let a = lo + (hi - lo) * i as f64 / points as f64;
            if check_continuous(f, a, epsilon, delta, 512).is_some() {
                delta /= 2.0;
                continue 'search;
            }
        }
        return Some(delta);
    }
    None
}

/// Finite-window Cauchy check: do all terms from index `tail`
/// onward lie within eps of each other? The sequence's own
/// coherence — Cauchy converges in R because R is complete.
pub fn is_cauchy_tail(seq: &[f64], tail: usize, epsilon: f64) -> bool {
    if tail >= seq.len() {
        return true;
    }
    let rest = &seq[tail..];
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in rest {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    hi - lo < epsilon
}

/// Estimate of a sequence's limit from its tail: midpoint of the
/// tail's spread when the spread is under tol.
pub fn seq_limit(seq: &[f64], tail: usize, tol: f64) -> Option<f64> {
    if tail >= seq.len() || seq.is_empty() {
        return None;
    }
    let rest = &seq[tail..];
    let lo = rest.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = rest.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if hi - lo <= tol {
        Some((hi + lo) / 2.0)
    } else {
        None
    }
}

/// Riemann sum kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SumKind {
    Left,
    Right,
    Midpoint,
    Trapezoid,
    Simpson,
}

/// Riemann-style integral estimate of f on [a, b] with n panels.
/// Simpson ignores the parity of n's last panel only when n is
/// odd — it needs pairs, so odd n returns None.
pub fn integral(f: &dyn Fn(f64) -> f64, a: f64, b: f64, n: usize, kind: SumKind) -> Option<f64> {
    if n == 0 || a == b {
        return if a == b { Some(0.0) } else { None };
    }
    let h = (b - a) / n as f64;
    let sum = match kind {
        SumKind::Left => (0..n).map(|i| f(a + i as f64 * h)).sum::<f64>() * h,
        SumKind::Right => (1..=n).map(|i| f(a + i as f64 * h)).sum::<f64>() * h,
        SumKind::Midpoint => (0..n).map(|i| f(a + (i as f64 + 0.5) * h)).sum::<f64>() * h,
        SumKind::Trapezoid => {
            let interior: f64 = (1..n).map(|i| f(a + i as f64 * h)).sum();
            (f(a) / 2.0 + interior + f(b) / 2.0) * h
        }
        SumKind::Simpson => {
            if n % 2 == 1 {
                return None;
            }
            let odd: f64 = (1..n).step_by(2).map(|i| f(a + i as f64 * h)).sum();
            let even: f64 = (2..n).step_by(2).map(|i| f(a + i as f64 * h)).sum();
            (f(a) + 4.0 * odd + 2.0 * even + f(b)) * h / 3.0
        }
    };
    Some(sum)
}

/// Bisection root-finder — the Intermediate Value Theorem made
/// algorithmic: a sign change on a continuous function traps a
/// root, and halving the interval keeps it trapped.
pub fn bisection(
    f: &dyn Fn(f64) -> f64,
    mut a: f64,
    mut b: f64,
    tol: f64,
    max_iters: usize,
) -> Option<f64> {
    let mut fa = f(a);
    if fa == 0.0 {
        return Some(a);
    }
    if f(b) == 0.0 {
        return Some(b);
    }
    if fa.signum() == f(b).signum() {
        return None; // IVT gives no promise without a sign change
    }
    for _ in 0..max_iters {
        let m = (a + b) / 2.0;
        let fm = f(m);
        if (b - a).abs() <= tol || fm == 0.0 {
            return Some(m);
        }
        if fa.signum() != fm.signum() {
            b = m;
        } else {
            a = m;
            fa = fm;
        }
    }
    Some((a + b) / 2.0)
}

/// Fixed-point iteration x_{k+1} = g(x_k): converges when g is a
/// contraction near the fixed point. Returns the limit when the
/// step shrinks under tol, else None.
pub fn fixed_point(g: &dyn Fn(f64) -> f64, mut x: f64, tol: f64, max_iters: usize) -> Option<f64> {
    for _ in 0..max_iters {
        let next = g(x);
        if (next - x).abs() <= tol {
            return Some(next);
        }
        x = next;
    }
    None
}

/// Newton–Raphson from a numeric derivative: x - f(x)/f'(x) with
/// a symmetric difference quotient. None when a derivative
/// vanishes or no convergence within max_iters.
pub fn newton(f: &dyn Fn(f64) -> f64, mut x: f64, tol: f64, max_iters: usize) -> Option<f64> {
    let h = 1e-7;
    for _ in 0..max_iters {
        let d = (f(x + h) - f(x - h)) / (2.0 * h);
        if d.abs() < 1e-300 {
            return None;
        }
        let next = x - f(x) / d;
        if (next - x).abs() <= tol {
            return Some(next);
        }
        x = next;
    }
    None
}

/// Ratio test on |a_{n+1}/a_n| — the sequence of ratios over the
/// tail indices `terms` (passes |a_k|, returns sampled limit of
/// the ratio). None when magnitudes hit zero or the ratio's tail
/// spread defeats the tolerance.
pub fn ratio_test(terms: &[f64], tol: f64) -> Option<f64> {
    let ratios: Vec<f64> = terms
        .windows(2)
        .filter(|w| w[0].abs() > 1e-300)
        .map(|w| (w[1] / w[0]).abs())
        .collect();
    if ratios.len() < 4 {
        return None;
    }
    seq_limit(&ratios, ratios.len() / 2, tol)
}

/// Sample a function into (x, y) points over [lo, hi] — the
/// graph's raw material: n evenly spaced abscissae and their
/// images. What gets drawn is this table of pairs.
pub fn plot(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> Vec<(f64, f64)> {
    if n < 2 || lo >= hi {
        return vec![];
    }
    (0..n)
        .map(|i| {
            let x = lo + (hi - lo) * i as f64 / (n - 1) as f64;
            (x, f(x))
        })
        .collect()
}

/// Sup and inf of a finite sample — the analysts' extrema.
pub fn sup_inf(samples: &[f64]) -> Option<(f64, f64)> {
    if samples.is_empty() {
        return None;
    }
    let sup = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let inf = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    Some((sup, inf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_game() {
        // f(x) = 2x at a=1: limit 2. eps=0.1 -> delta=0.05 works.
        let f = |x: f64| 2.0 * x;
        assert!(check_delta(&f, 1.0, 2.0, 0.1, 0.05, 256).is_none());
        // delta too big lets x=1.06 escape the eps band.
        assert!(check_delta(&f, 1.0, 2.0, 0.1, 0.06, 256).is_some());
        assert!(find_delta(&f, 1.0, 2.0, 0.1, 1.0, 16).is_some());
    }

    #[test]
    fn limits() {
        let sinc = |x: f64| if x == 0.0 { 1.0 } else { x.sin() / x };
        let l = limit_numeric(&sinc, 0.0, 1e-6, 24).unwrap();
        assert!((l - 1.0).abs() < 1e-4);
        // Jump function: two sides disagree.
        let jump = |x: f64| if x < 0.0 { -1.0 } else { 1.0 };
        assert_eq!(limit_numeric(&jump, 0.0, 1e-6, 24), None);
    }

    #[test]
    fn uniform() {
        // x^2 on [0, 0.5]: steepest at the right edge, still uniform.
        let sq = |x: f64| x * x;
        assert!(uniform_modulus(&sq, 0.0, 0.5, 0.01, 64, 1.0, 20).is_some());
    }

    #[test]
    fn cauchy_and_seq() {
        let geom: Vec<f64> = (0..20).map(|k| 1.0 - 0.5f64.powi(k)).collect();
        assert!(is_cauchy_tail(&geom, 10, 1e-3));
        let l = seq_limit(&geom, 15, 1e-4).unwrap();
        assert!((l - 1.0).abs() < 1e-4);
    }

    #[test]
    fn integrate_and_roots() {
        let sq = |x: f64| x * x;
        let s = integral(&sq, 0.0, 1.0, 1000, SumKind::Simpson).unwrap();
        assert!((s - 1.0 / 3.0).abs() < 1e-9);
        let t = integral(&sq, 0.0, 1.0, 1000, SumKind::Trapezoid).unwrap();
        assert!((t - 1.0 / 3.0).abs() < 1e-5);

        let root = bisection(&|x| x * x - 2.0, 1.0, 2.0, 1e-12, 100).unwrap();
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-10);
        let fp = fixed_point(&|x| (x + 2.0 / x) / 2.0, 1.0, 1e-14, 50).unwrap();
        assert!((fp - std::f64::consts::SQRT_2).abs() < 1e-12);
        let nr = newton(&|x| x * x - 2.0, 1.0, 1e-12, 50).unwrap();
        assert!((nr - std::f64::consts::SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn ratio() {
        // a_n = 2^-n: ratio tends to 1/2 — absolutely convergent.
        let terms: Vec<f64> = (1..30).map(|k| 0.5f64.powi(k)).collect();
        let r = ratio_test(&terms, 1e-6).unwrap();
        assert!((r - 0.5).abs() < 1e-4);
    }
}
