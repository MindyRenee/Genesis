//! Algebra — polynomial manipulation and closed-form equation solving.
//!
//! Machine-side counterpart to `python/parietal_lobe/math/algebra.py`.
//! Represents univariate polynomials `c[0] + c[1]·x + … + c[d]·x^d` and
//! provides addition, multiplication, Horner evaluation, linear and
//! quadratic real/complex root solving.

use crate::math::complex_analysis::Complex;

/// Univariate polynomial in ascending degree order (`coeffs[k]` is `x^k`).
#[derive(Clone, Debug, PartialEq)]
pub struct Poly {
    pub coeffs: Vec<f64>,
}

impl Poly {
    pub fn new(mut coeffs: Vec<f64>) -> Self {
        while coeffs.len() > 1 && coeffs.last().is_some_and(|c| c.abs() < 1e-15) {
            coeffs.pop();
        }
        if coeffs.is_empty() {
            coeffs.push(0.0);
        }
        Self { coeffs }
    }

    pub fn degree(&self) -> usize {
        if self.coeffs.len() == 1 && self.coeffs[0].abs() < 1e-15 {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    /// Evaluate at `x` by Horner's rule.
    pub fn eval(&self, x: f64) -> f64 {
        self.coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
    }

    pub fn add(&self, rhs: &Poly) -> Poly {
        let n = self.coeffs.len().max(rhs.coeffs.len());
        let coeffs = (0..n)
            .map(|i| {
                self.coeffs.get(i).copied().unwrap_or(0.0)
                    + rhs.coeffs.get(i).copied().unwrap_or(0.0)
            })
            .collect();
        Poly::new(coeffs)
    }

    pub fn mul(&self, rhs: &Poly) -> Poly {
        let mut out = vec![0.0; self.coeffs.len() + rhs.coeffs.len() - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in rhs.coeffs.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        Poly::new(out)
    }
}

/// Solve `a·x + b = 0`.
pub fn solve_linear(a: f64, b: f64) -> Option<f64> {
    if !a.is_finite() || !b.is_finite() || a.abs() < 1e-15 {
        return None;
    }
    Some(-b / a)
}

/// Solve `a·x² + b·x + c = 0` over ℂ (numerically stable Citardauq/quadratic pair).
pub fn solve_quadratic(a: f64, b: f64, c: f64) -> Option<(Complex, Complex)> {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() || a.abs() < 1e-15 {
        return None;
    }
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
        let s = disc.sqrt();
        Some((
            Complex::new((-b + s) / (2.0 * a), 0.0),
            Complex::new((-b - s) / (2.0 * a), 0.0),
        ))
    } else {
        let s = (-disc).sqrt();
        let re = -b / (2.0 * a);
        let im = s / (2.0 * a);
        Some((Complex::new(re, im), Complex::new(re, -im)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly_and_quadratics() {
        // (x + 1)(x - 1) = x^2 - 1
        let p = Poly::new(vec![1.0, 1.0]).mul(&Poly::new(vec![-1.0, 1.0]));
        assert_eq!(p.coeffs, vec![-1.0, 0.0, 1.0]);
        assert!((p.eval(3.0) - 8.0).abs() < 1e-12);
        let (r1, r2) = solve_quadratic(1.0, 0.0, -4.0).unwrap();
        assert!((r1.re - 2.0).abs() < 1e-12 && (r2.re + 2.0).abs() < 1e-12);
    }
}
