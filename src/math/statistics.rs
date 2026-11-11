//! Statistics — central tendency, dispersion, and distributional shape.
//!
//! Machine-side counterpart to `python/parietal_lobe/math/statistics.py`.
//! Implements the power-mean hierarchy (arithmetic, geometric, harmonic,
//! quadratic), median, population/sample variance, IQR, skewness, and kurtosis.

/// Full descriptive statistical summary of a finite sample.
#[derive(Clone, Debug, PartialEq)]
pub struct StatSummary {
    pub n: usize,
    pub minimum: f64,
    pub maximum: f64,
    pub arithmetic_mean: f64,
    pub median: f64,
    pub variance: f64,
    pub standard_deviation: f64,
    pub interquartile_range: f64,
    pub skewness: Option<f64>,
    pub kurtosis: Option<f64>,
    pub standard_error: f64,
}

pub fn arithmetic_mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() || xs.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some(xs.iter().sum::<f64>() / xs.len() as f64)
}

pub fn geometric_mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() || xs.iter().any(|&x| !x.is_finite() || x <= 0.0) {
        return None;
    }
    Some((xs.iter().map(|x| x.ln()).sum::<f64>() / xs.len() as f64).exp())
}

pub fn harmonic_mean(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() || xs.iter().any(|&x| !x.is_finite() || x <= 0.0) {
        return None;
    }
    Some(xs.len() as f64 / xs.iter().map(|x| 1.0 / x).sum::<f64>())
}

pub fn power_mean(xs: &[f64], p: f64) -> Option<f64> {
    if !p.is_finite() {
        return None;
    }
    if p == 0.0 {
        return geometric_mean(xs);
    }
    if xs.is_empty() || xs.iter().any(|&x| !x.is_finite() || x <= 0.0) {
        return None;
    }
    Some((xs.iter().map(|x| x.powf(p)).sum::<f64>() / xs.len() as f64).powf(1.0 / p))
}

pub fn median(xs: &[f64]) -> Option<f64> {
    if xs.is_empty() || xs.iter().any(|x| !x.is_finite()) {
        return None;
    }
    let mut vals = xs.to_vec();
    vals.sort_by(f64::total_cmp);
    let mid = vals.len() / 2;
    if vals.len() % 2 == 1 {
        Some(vals[mid])
    } else {
        Some((vals[mid - 1] + vals[mid]) / 2.0)
    }
}

pub fn percentile(xs: &[f64], q: f64) -> Option<f64> {
    if xs.is_empty() || xs.iter().any(|x| !x.is_finite()) || !(0.0..=100.0).contains(&q) {
        return None;
    }
    let mut vals = xs.to_vec();
    vals.sort_by(f64::total_cmp);
    let pos = (q / 100.0) * (vals.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let frac = pos - lo as f64;
    if lo + 1 < vals.len() {
        Some(vals[lo] + frac * (vals[lo + 1] - vals[lo]))
    } else {
        Some(vals[lo])
    }
}

pub fn variance(xs: &[f64], sample: bool) -> Option<f64> {
    let mu = arithmetic_mean(xs)?;
    if sample && xs.len() < 2 {
        return None;
    }
    let denom = (xs.len() - usize::from(sample)) as f64;
    Some(xs.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / denom)
}

pub fn summarize(xs: &[f64]) -> Option<StatSummary> {
    let mu = arithmetic_mean(xs)?;
    let med = median(xs)?;
    let m2 = variance(xs, false)?;
    let q1 = percentile(xs, 25.0)?;
    let q3 = percentile(xs, 75.0)?;
    let n = xs.len() as f64;
    let m3 = xs.iter().map(|x| (x - mu).powi(3)).sum::<f64>() / n;
    let m4 = xs.iter().map(|x| (x - mu).powi(4)).sum::<f64>() / n;
    let min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let sd = m2.sqrt();
    Some(StatSummary {
        n: xs.len(),
        minimum: min,
        maximum: max,
        arithmetic_mean: mu,
        median: med,
        variance: m2,
        standard_deviation: sd,
        interquartile_range: q3 - q1,
        skewness: (m2 > 0.0).then(|| m3 / m2.powf(1.5)),
        kurtosis: (m2 > 0.0).then(|| m4 / (m2 * m2)),
        standard_error: sd / n.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pythagorean_mean_inequality() {
        // HM <= GM <= AM <= QM for positive distinct values
        let xs = [2.0, 4.0, 8.0];
        let hm = harmonic_mean(&xs).unwrap();
        let gm = geometric_mean(&xs).unwrap();
        let am = arithmetic_mean(&xs).unwrap();
        let qm = power_mean(&xs, 2.0).unwrap();
        assert!(hm < gm && gm < am && am < qm);
        let s = summarize(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!((s.arithmetic_mean - 3.0).abs() < 1e-12);
        assert!((s.median - 3.0).abs() < 1e-12);
        assert!(s.skewness.unwrap().abs() < 1e-12);
    }
}
