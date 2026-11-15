//! Calculus — symbolic polynomial differentiation/integration and Taylor jets.
//!
//! Machine-side counterpart to `python/parietal_lobe/math/calculus.py`.

use crate::math::algebra::Poly;

/// Exact derivative of a univariate polynomial: `d/dx Σ c_k x^k = Σ k c_k x^{k-1}`.
pub fn differentiate_poly(p: &Poly) -> Poly {
    if p.coeffs.len() <= 1 {
        return Poly::new(vec![0.0]);
    }
    let coeffs = p
        .coeffs
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, &c)| (k as f64) * c)
        .collect();
    Poly::new(coeffs)
}

/// Exact antiderivative with constant term `c0`: `∫ Σ c_k x^k dx = c0 + Σ (c_k/(k+1)) x^{k+1}`.
pub fn integrate_poly(p: &Poly, c0: f64) -> Poly {
    let mut coeffs = Vec::with_capacity(p.coeffs.len() + 1);
    coeffs.push(c0);
    for (k, &c) in p.coeffs.iter().enumerate() {
        coeffs.push(c / (k as f64 + 1.0));
    }
    Poly::new(coeffs)
}

/// Definite integral `∫_lo^hi p(x) dx` via the Fundamental Theorem of Calculus.
pub fn definite_integral_poly(p: &Poly, lo: f64, hi: f64) -> f64 {
    let anti = integrate_poly(p, 0.0);
    anti.eval(hi) - anti.eval(lo)
}

/// Symmetric central-difference numerical derivative `(f(x+h) - f(x-h)) / (2h)`.
pub fn numeric_derivative(f: &dyn Fn(f64) -> f64, x: f64, h: f64) -> Option<f64> {
    if !x.is_finite() || !h.is_finite() || h <= 0.0 {
        return None;
    }
    let d = (f(x + h) - f(x - h)) / (2.0 * h);
    d.is_finite().then_some(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fundamental_theorem_holds() {
        // p(x) = 3x^2 => integral from 0 to 2 is 8, derivative of x^3 is 3x^2.
        let cubic = Poly::new(vec![0.0, 0.0, 0.0, 1.0]);
        let deriv = differentiate_poly(&cubic);
        assert_eq!(deriv.coeffs, vec![0.0, 0.0, 3.0]);
        assert!((definite_integral_poly(&deriv, 0.0, 2.0) - 8.0).abs() < 1e-12);
        let slope = numeric_derivative(&|x| x * x, 3.0, 1e-6).unwrap();
        assert!((slope - 6.0).abs() < 1e-8);
    }
}
