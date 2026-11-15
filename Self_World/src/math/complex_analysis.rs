//! Complex analysis — the plane where every root exists.
//!
//! `Complex` is a pair of f64s; the operations are the field's own:
//! arithmetic, conjugate, modulus, argument, exp/log/pow through
//! Euler, roots of unity, polynomial evaluation by Horner, and the
//! Möbius map. The imaginary unit needs no name here — only
//! arithmetic.

use std::f64::consts::TAU;

/// A complex number z = re + im·i.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// From polar form r·e^{iθ}.
    pub fn polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Complex conjugate z̄.
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Modulus |z|.
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// Squared modulus |z|² — cheaper, enough for comparisons.
    pub fn abs2(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// Argument arg(z) in (-π, π].
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    /// (r, θ) polar form.
    pub fn to_polar(self) -> (f64, f64) {
        (self.abs(), self.arg())
    }

    /// Division z/w = z·w̄/|w|²; None at w = 0.
    pub fn checked_div(self, o: Self) -> Option<Self> {
        let d = o.abs2();
        if d == 0.0 {
            return None;
        }
        let n = self * o.conj();
        Some(Self {
            re: n.re / d,
            im: n.im / d,
        })
    }

    /// Multiplicative inverse.
    pub fn recip(self) -> Option<Self> {
        Self::ONE.checked_div(self)
    }

    /// e^z = e^re·(cos im + i sin im) — Euler, machine side.
    pub fn exp(self) -> Self {
        Self::polar(self.re.exp(), self.im)
    }

    /// Principal logarithm: ln|z| + i·arg(z).
    pub fn ln(self) -> Self {
        let (r, t) = self.to_polar();
        Self { re: r.ln(), im: t }
    }

    /// z^w = e^{w·ln z} — principal branch.
    pub fn pow(self, w: Self) -> Self {
        (self.ln() * w).exp()
    }

    /// Integer power by repeated squaring — exact for small n.
    pub fn powi(self, mut n: i64) -> Self {
        if n < 0 {
            return self.recip().map(|r| r.powi(-n)).unwrap_or(Self::ZERO);
        }
        let mut r = Self::ONE;
        let mut b = self;
        while n > 0 {
            if n & 1 == 1 {
                r = r * b;
            }
            b = b * b;
            n >>= 1;
        }
        r
    }

    /// Square root — principal branch (nonnegative real part).
    pub fn sqrt(self) -> Self {
        let (r, t) = self.to_polar();
        Self::polar(r.sqrt(), t / 2.0)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}

/// Complex multiplication — rotation composed with scaling.
impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}

/// The n-th roots of unity: e^{2πik/n} for k = 0..n.
/// The vertices of the regular n-gon on the unit circle.
pub fn roots_of_unity(n: usize) -> Vec<Complex> {
    (0..n)
        .map(|k| Complex::polar(1.0, TAU * k as f64 / n as f64))
        .collect()
}

/// All n n-th roots of z: |z|^{1/n}·e^{i(θ+2πk)/n}.
pub fn roots(z: Complex, n: usize) -> Vec<Complex> {
    if n == 0 || z == Complex::ZERO {
        return vec![];
    }
    let (r, t) = z.to_polar();
    let rr = r.powf(1.0 / n as f64);
    (0..n)
        .map(|k| Complex::polar(rr, (t + TAU * k as f64) / n as f64))
        .collect()
}

/// Polynomial evaluation at a complex point — Horner.
/// `coeffs` are highest-degree first: c₀zⁿ + c₁zⁿ⁻¹ + …
pub fn eval_poly(coeffs: &[Complex], z: Complex) -> Complex {
    coeffs.iter().fold(Complex::ZERO, |acc, &c| acc * z + c)
}

/// Möbius transformation z ↦ (az+b)/(cz+d); None when the
/// denominator vanishes (the point at infinity).
pub fn mobius(a: Complex, b: Complex, c: Complex, d: Complex, z: Complex) -> Option<Complex> {
    (a * z + b).checked_div(c * z + d)
}

/// Is |z - center| <= radius + eps — inside a disk.
pub fn in_disk(z: Complex, center: Complex, radius: f64, eps: f64) -> bool {
    (z - center).abs() <= radius + eps
}

/// Winding-ish count: signed turns of a polygonal path around a
/// point — the discrete argument principle. `path` is the ordered
/// vertex list of the contour.
pub fn winding_number(path: &[Complex], point: Complex) -> i64 {
    if path.len() < 2 {
        return 0;
    }
    let mut turns = 0.0;
    for w in path.windows(2) {
        let a = (w[0] - point).arg();
        let b = (w[1] - point).arg();
        turns += crate::math::trig::normalize(b - a);
    }
    let a = (path[path.len() - 1] - point).arg();
    let b = (path[0] - point).arg();
    turns += crate::math::trig::normalize(b - a);
    (turns / TAU).round() as i64
}

/// De Moivre check: (cos θ + i sin θ)ⁿ = cos nθ + i sin nθ.
pub fn de_moivre_holds(theta: f64, n: i64, tolerance: f64) -> bool {
    let lhs = Complex::polar(1.0, theta).powi(n);
    let rhs = Complex::polar(1.0, n as f64 * theta);
    (lhs - rhs).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Complex, b: Complex) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn field_ops() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.abs(), 5.0);
        assert!(close(z * z.conj(), Complex::new(25.0, 0.0)));
        assert!(close(z.checked_div(z).unwrap(), Complex::ONE));
    }

    #[test]
    fn euler_identity() {
        // e^{iπ} + 1 = 0 — the machine checks the famous line.
        let lhs = (Complex::I * Complex::new(std::f64::consts::PI, 0.0)).exp() + Complex::ONE;
        assert!(lhs.abs() < 1e-12);
    }

    #[test]
    fn unity_roots() {
        let r5 = roots_of_unity(5);
        for z in &r5 {
            assert!((z.abs() - 1.0).abs() < 1e-12);
            assert!(close(z.powi(5), Complex::ONE));
        }
    }

    #[test]
    fn winding() {
        // Unit square around the origin winds once; around (2,2) zero.
        let sq: Vec<Complex> = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
            .iter()
            .map(|&(re, im)| Complex::new(re, im))
            .collect();
        assert_eq!(winding_number(&sq, Complex::ZERO), 1);
        assert_eq!(winding_number(&sq, Complex::new(2.0, 2.0)), 0);
    }

    #[test]
    fn de_moivre() {
        assert!(de_moivre_holds(0.37, 7, 1e-10));
    }
}
