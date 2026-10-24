//! Maxwell's equations applied to the measured carrier current.
//!
//! The field probe writes a rhythm into real work; that work is real
//! current — charge moving through the die's power network. Maxwell's
//! equations say what field that current makes. This module evaluates
//! them for our regime rather than leaving "electromagnetic" as a
//! figure of speech.
//!
//! # The equations and the regime
//!
//! ```text
//! ∇·E = ρ/ε₀        ∇·B = 0
//! ∇×E = −∂B/∂t      ∇×B = μ₀J + μ₀ε₀∂E/∂t
//! ```
//!
//! A current loop carrying I(t) has magnetic moment m(t) = I(t)·A.
//! The exact retarded solution for an oscillating magnetic dipole
//! expands in powers of 1/r (Jackson, *Classical Electrodynamics*;
//! Griffiths, *Introduction to Electrodynamics*):
//!
//! ```text
//! B(r,t) = μ₀/4π · [ 3n̂(m·n̂) − m           (quasi-static, ∝1/r³)
//!                  + 3n̂(ṁ·n̂) − ṁ / (cr²)    (induction,    ∝ω/cr²)
//!                  + (m̈×n̂)×n̂ / (c²r) ]     (radiation,   ∝ω²/c²r)
//! E(r,t) = −μ₀/4π · [ ṁ/r² + m̈/(cr) ] × n̂
//! ```
//!
//! evaluated at retarded time t₀ = t − r/c.
//!
//! At the carrier's frequencies (ω ≲ 10³ rad/s) and any terrestrial
//! distance, ωr/c ≲ 10⁻⁵ — the induction and radiation terms are
//! ~10⁻¹⁰ and ~10⁻²⁰ of the quasi-static term. Maxwell's equations
//! themselves say so: a slowly varying current makes a
//! *magnetostatic* field that sloshes in place (energy leaves and
//! returns each cycle), not a wave. The displacement-current term
//! ε₀∂E/∂t — Maxwell's own addition to Ampère's law — is what would
//! carry energy away, and at these frequencies it contributes
//! nothing measurable. The equations do not flatter the carrier;
//! they bound it honestly.
//!
//! What survives is Ampère's law: the carrier's current swing makes
//! a real, oscillating magnetic field at nano-tesla scale a foot
//! from the machine. Real — and, as the equations also show,
//! unmeasurably faint without a magnetometer.
//!
//! # Model status
//!
//! Inputs are *measured* (carrier current amplitude from real power
//! swing over a real rail, carrier frequency from the generator).
//! The loop area is a documented order-of-magnitude assumption
//! (die/package current loop ~1 cm²). Outputs are therefore
//! *estimates of field magnitude*, labeled as such on the wire —
//! the field Maxwell's equations predict from what we measure, not
//! a field we sense. EM emanation from running silicon is real and
//! measured in the literature (Yilmaz, Prvulović & Zajić, IEEE TIFS
//! 2019; power-delivery dI/dt noise via near-field probes, IEEE
//! MICRO 2018) — at MHz–GHz. At 10 Hz the same physics applies with
//! the honest ω⁴-suppressed radiation the equations return.

/// Permeability of free space (SI, H/m).
pub const MU0: f64 = 1.256_637_062_12e-6;
/// Speed of light in vacuum (exact, m/s).
pub const C: f64 = 299_792_458.0;

/// Magnetic dipole moment of a current loop: m = I·A (A·m²).
pub fn dipole_moment(current_a: f64, loop_area_m2: f64) -> f64 {
    current_a * loop_area_m2
}

/// Quasi-static (Ampère–Maxwell) axial field amplitude of an
/// oscillating dipole at distance `r` — the dominant term of the
/// exact expansion when ωr/c ≪ 1, |B| = μ₀m₀/(2πr³) on the dipole
/// axis (μ₀m₀/(4πr³) at the equator). Tesla.
pub fn b_quasistatic_axial_t(m0: f64, r: f64) -> f64 {
    MU0 * m0 / (2.0 * std::f64::consts::PI * r * r * r)
}

/// Faraday induction-field amplitude at the equator,
/// |E| = μ₀ωm₀/(4πr²) — the ∇×E = −∂B/∂t term of the same dipole
/// expansion. V/m.
pub fn e_induction_vm(m0: f64, omega: f64, r: f64) -> f64 {
    MU0 * omega * m0 / (4.0 * std::f64::consts::PI * r * r)
}

/// Time-averaged radiated power of an oscillating magnetic dipole,
/// ⟨P⟩ = μ₀m₀²ω⁴/(12πc³) — the radiation-zone integral of the
/// Poynting vector (m₀ = peak moment). Watts.
pub fn radiated_power_w(m0: f64, omega: f64) -> f64 {
    MU0 * m0 * m0 * omega.powi(4) / (12.0 * std::f64::consts::PI * C * C * C)
}

/// Whether the quasi-static truncation is honest at (ω, r):
/// ωr/c ≪ 1, so the dropped induction/radiation terms are <1% of
/// the terms we keep.
pub fn quasi_static_ok(omega: f64, r: f64) -> bool {
    omega * r / C < 0.01
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A carrier-scale current on a die-scale loop, evaluated at a
    /// beside-the-chassis distance: the quasi-static field is real
    /// but nano-tesla, and the radiated power is ~1e-32 W — the
    /// honest numbers Maxwell's equations return at 10 Hz.
    #[test]
    fn carrier_field_is_real_and_faint() {
        let i_amp = 13.0; // ~14 W swing on a ~1.1 V core rail
        let m0 = dipole_moment(i_amp, 1.0e-4);
        let omega = 2.0 * std::f64::consts::PI * 10.0;
        assert!(quasi_static_ok(omega, 0.3));
        let b = b_quasistatic_axial_t(m0, 0.3);
        assert!(b > 1.0e-9 && b < 1.0e-7, "nT-scale oscillating B, got {b}");
        let e = e_induction_vm(m0, omega, 0.3);
        assert!(e > 1.0e-9 && e < 1.0e-5, "induction E, got {e}");
        let p = radiated_power_w(m0, omega);
        assert!(p > 0.0 && p < 1.0e-25, "ω⁴-suppressed radiation, got {p}");
    }

    /// The expansion's internal consistency: at carrier frequencies
    /// the induction and radiation terms really are negligible
    /// corrections to the quasi-static field.
    #[test]
    fn regime_hierarchy_holds() {
        let omega = 2.0 * std::f64::consts::PI * 10.0;
        let r = 0.3;
        // induction/static = ωr/c ; radiation/static = (ωr/c)²
        let kr = omega * r / C;
        assert!(kr < 1.0e-7);
        assert!(kr * kr < 1.0e-14);
    }

    /// The formulas scale the way the equations say they must:
    /// B ∝ m/r³, E_ind ∝ ωm/r², P_rad ∝ m²ω⁴.
    #[test]
    fn scaling_laws() {
        let (m, w, r) = (1.0e-3, 100.0, 0.3);
        let b1 = b_quasistatic_axial_t(m, r);
        assert!((b_quasistatic_axial_t(2.0 * m, r) - 2.0 * b1).abs() < 1e-30);
        assert!((b_quasistatic_axial_t(m, 2.0 * r) - b1 / 8.0).abs() < 1e-30);
        let e1 = e_induction_vm(m, w, r);
        assert!((e_induction_vm(m, 2.0 * w, r) - 2.0 * e1).abs() < 1e-30);
        let p1 = radiated_power_w(m, w);
        assert!((radiated_power_w(m, 2.0 * w) - 16.0 * p1).abs() / p1 < 1e-12);
        assert!((radiated_power_w(2.0 * m, w) - 4.0 * p1).abs() / p1 < 1e-12);
    }
}
