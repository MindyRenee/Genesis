//! Trigonometry — the circle's arithmetic.
//!
//! Angles, the unit circle, and the triangle laws. Everything here
//! is a function over `f64`: angles in radians unless a name says
//! degrees. No parsing, no vocabulary — the machine side only
//! measures.

use std::f64::consts::{PI, TAU};

/// Two full turns in radians.
pub const FULL_TURN: f64 = TAU;

/// Degrees to radians.
pub fn deg(degrees: f64) -> f64 {
    degrees * PI / 180.0
}

/// Radians to degrees.
pub fn rad(radians: f64) -> f64 {
    radians * 180.0 / PI
}

/// Wrap an angle into (-pi, pi] — the canonical turn.
pub fn normalize(angle: f64) -> f64 {
    let mut a = angle % TAU;
    if a <= -PI {
        a += TAU;
    } else if a > PI {
        a -= TAU;
    }
    a
}

/// Point on the unit circle at `angle`.
pub fn unit_point(angle: f64) -> (f64, f64) {
    (angle.cos(), angle.sin())
}

/// Signed angle between two unit-circle bearings, in (-pi, pi].
pub fn angle_between(from: f64, to: f64) -> f64 {
    normalize(to - from)
}

/// The internal angle at vertex B for triangle side lengths
/// (a = |BC|, b = |CA|, c = |AB|). Law of cosines, inverted.
pub fn angle_from_sides(adjacent_a: f64, opposite_b: f64, adjacent_c: f64) -> Option<f64> {
    if adjacent_a <= 0.0 || adjacent_c <= 0.0 {
        return None;
    }
    let cos_b = (adjacent_a * adjacent_a + adjacent_c * adjacent_c - opposite_b * opposite_b)
        / (2.0 * adjacent_a * adjacent_c);
    if (-1.0..=1.0).contains(&cos_b) {
        Some(cos_b.clamp(-1.0, 1.0).acos())
    } else {
        None // no triangle has these sides
    }
}

/// Side opposite the given angle from two adjacent sides.
/// Law of cosines: c² = a² + b² - 2ab·cos(C).
pub fn side_from_angle(a: f64, b: f64, angle_c: f64) -> Option<f64> {
    if a <= 0.0 || b <= 0.0 {
        return None;
    }
    Some((a * a + b * b - 2.0 * a * b * angle_c.cos()).sqrt())
}

/// All three angles of a triangle from its sides (a, b, c).
/// Returns (A, B, C) opposite each side, or None when the
/// sides cannot form a triangle.
pub fn triangle_angles(a: f64, b: f64, c: f64) -> Option<(f64, f64, f64)> {
    if a <= 0.0 || b <= 0.0 || c <= 0.0 || a + b <= c || b + c <= a || c + a <= b {
        return None;
    }
    let big_a = angle_from_sides(b, a, c)?;
    let big_b = angle_from_sides(a, b, c)?;
    Some((big_a, big_b, PI - big_a - big_b))
}

/// Heron's area from three side lengths.
pub fn area_heron(a: f64, b: f64, c: f64) -> Option<f64> {
    if triangle_angles(a, b, c).is_none() {
        return None;
    }
    let s = (a + b + c) / 2.0;
    Some((s * (s - a) * (s - b) * (s - c)).sqrt())
}

/// Area from two sides and the included angle.
pub fn area_sas(a: f64, b: f64, angle_c: f64) -> f64 {
    0.5 * a * b * angle_c.sin()
}

/// Law of sines: given side `a` opposite angle `a_ang`, the side
/// opposite `b_ang`.
pub fn side_law_of_sines(a: f64, a_ang: f64, b_ang: f64) -> Option<f64> {
    let sa = a_ang.sin();
    if sa.abs() < f64::EPSILON {
        return None;
    }
    Some(a * b_ang.sin() / sa)
}

/// The six trigonometric ratios at an angle, gathered.
/// Returns (sin, cos, tan, csc, sec, cot) with None where the
/// ratio diverges.
pub fn ratios(angle: f64) -> (f64, f64, Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    let (s, c) = (angle.sin(), angle.cos());
    let tan = if c.abs() > 1e-15 { Some(s / c) } else { None };
    let csc = if s.abs() > 1e-15 { Some(1.0 / s) } else { None };
    let sec = if c.abs() > 1e-15 { Some(1.0 / c) } else { None };
    let cot = if s.abs() > 1e-15 { Some(c / s) } else { None };
    (s, c, tan, csc, sec, cot)
}

/// The identity sin²θ + cos²θ = 1, checked numerically.
/// The machine verifies what it cannot prove.
pub fn pythagorean_holds(angle: f64, tolerance: f64) -> bool {
    let (s, c) = (angle.sin(), angle.cos());
    (s * s + c * c - 1.0).abs() <= tolerance
}

/// Sum-to-product check: sin(a+b) ≈ sin a cos b + cos a sin b.
pub fn angle_sum_holds(a: f64, b: f64, tolerance: f64) -> bool {
    ((a + b).sin() - (a.sin() * b.cos() + a.cos() * b.sin())).abs() <= tolerance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_wraps() {
        assert!((normalize(3.0 * PI) - PI).abs() < 1e-12);
        assert!((normalize(-PI / 2.0) + PI / 2.0).abs() < 1e-12);
    }

    #[test]
    fn unit_circle_points() {
        let (x, y) = unit_point(0.0);
        assert!((x - 1.0).abs() < 1e-15 && y.abs() < 1e-15);
    }

    #[test]
    fn triangle_345() {
        let (a, b, c) = triangle_angles(3.0, 4.0, 5.0).unwrap();
        assert!((a + b + c - PI).abs() < 1e-12);
        assert!((c - PI / 2.0).abs() < 1e-12);
        assert!((area_heron(3.0, 4.0, 5.0).unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn identities_hold() {
        for k in 0..8 {
            let t = k as f64 * 0.7 - 2.1;
            assert!(pythagorean_holds(t, 1e-12));
            assert!(angle_sum_holds(t, 0.5, 1e-12));
        }
    }
}
