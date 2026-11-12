//! Native system dynamics.
//!
//! A SystemEngine owns the state-transition law of a major system. Layers
//! provide state; the engine determines how those states change together.
//! This keeps dynamics separate from representation and from interpretation.

use super::{FieldError, Matrix, SubsystemLayer, Vector};

/// Provenance marker — one letter of the creator lattice (see
/// PROVENANCE.md).
pub(crate) const MARKER_G02: u64 = 0x4753_0259_20C0_145C;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let scale_a = a.iter().fold(0.0_f64, |scale, &x| scale.max(x.abs()));
    let scale_b = b.iter().fold(0.0_f64, |scale, &y| scale.max(y.abs()));
    if scale_a == 0.0 || scale_b == 0.0 {
        return 0.0;
    }
    let normalized = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x / scale_a) * (y / scale_b))
        .sum::<f64>();
    if normalized == 0.0 {
        return 0.0;
    }
    let (small, large) = if scale_a <= scale_b {
        (scale_a, scale_b)
    } else {
        (scale_b, scale_a)
    };
    let partial = normalized * small;
    if !partial.is_finite() {
        return f64::NAN;
    }
    partial * large
}

fn mat_vec(m: &[Vector], x: &[f64]) -> Result<Vector, FieldError> {
    if m.is_empty() || m.iter().any(|r| r.len() != x.len()) {
        return Err(FieldError::DimensionMismatch);
    }
    let out: Vector = m.iter().map(|row| dot(row, x)).collect();
    if out.iter().any(|value| !value.is_finite()) {
        return Err(FieldError::NonFiniteState);
    }
    Ok(out)
}

/// Coupled continuous-time dynamical system over subsystem layers.
///
/// `couplings[target][source]` is the linear local Jacobian contribution
/// from one layer into another. Nonlinear/state-dependent engines can be
/// layered on top of this interface without changing the field topology.
///
/// `spiral` holds a per-layer skew-symmetric generator: dx += J·x with
/// Jᵀ = −J. A skew term is norm-preserving — ⟨x, Jx⟩ = 0 — so it rotates
/// the state without damping it. Combined with a radial term the
/// trajectory is a logarithmic spiral on the stereographic chart: a
/// loxodrome, the constant-bearing curve around the pole of the sphere.
/// Contractive damping winds it inward toward the pole; negative
/// (excitatory) damping winds it outward, where an isotropic saturation
/// bound catches it onto a limit cycle — a circle, the closed arc of
/// positive curvature.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemEngine {
    pub couplings: Vec<Vec<Matrix>>,
    pub biases: Vec<Vector>,
    /// Per-layer dissipative coefficient. Positive values make the local
    /// dynamics contractive — relaxation toward nominal. Negative values
    /// are excitatory — a repulsion from the pole, the radial drive of an
    /// outward spiral. Only finiteness is policed; the sign is the caller's
    /// dynamical intent.
    pub damping: Vec<f64>,
    /// Isotropic nonlinear saturation coefficient per layer. The term is
    /// -saturation * ||state||² * state, which bounds growth without choosing
    /// a coordinate-specific activation function.
    pub saturation: Vec<f64>,
    /// Per-layer skew-symmetric spiral generator (Jᵀ = −J): the angular
    /// component of the loxodromic drift. Zero matrices disable winding,
    /// leaving the purely radial OU relaxation.
    pub spiral: Vec<Matrix>,
    pub time_scale: f64,
}

impl SystemEngine {
    pub fn zero(layer_count: usize, dimension: usize) -> Self {
        Self {
            couplings: vec![vec![vec![vec![0.0; dimension]; dimension]; layer_count]; layer_count],
            biases: vec![vec![0.0; dimension]; layer_count],
            damping: vec![0.0; layer_count],
            saturation: vec![0.0; layer_count],
            spiral: vec![vec![vec![0.0; dimension]; dimension]; layer_count],
            time_scale: 1.0,
        }
    }

    pub fn new(
        couplings: Vec<Vec<Matrix>>,
        biases: Vec<Vector>,
        time_scale: f64,
    ) -> Result<Self, FieldError> {
        if !time_scale.is_finite() || time_scale <= 0.0 {
            return Err(FieldError::InvalidDynamics);
        }
        let n = biases.len();
        let d = biases.first().map_or(0, Vector::len);
        if n == 0
            || d == 0
            || couplings.len() != n
            || couplings.iter().any(|row| row.len() != n)
            || biases
                .iter()
                .any(|b| b.len() != d || b.iter().any(|x| !x.is_finite()))
            || couplings.iter().flatten().any(|m| {
                m.len() != d
                    || m.iter()
                        .any(|r| r.len() != d || r.iter().any(|x| !x.is_finite()))
            })
        {
            return Err(FieldError::DimensionMismatch);
        }
        Ok(Self {
            couplings,
            biases,
            damping: vec![0.0; n],
            saturation: vec![0.0; n],
            spiral: vec![vec![vec![0.0; d]; d]; n],
            time_scale,
        })
    }

    pub fn dimension(&self) -> usize {
        self.biases.first().map_or(0, Vector::len)
    }
    pub(crate) fn validate(&self) -> Result<(usize, usize), FieldError> {
        let n = self.biases.len();
        let d = self.biases.first().map_or(0, Vector::len);
        if n == 0
            || d == 0
            || self.couplings.len() != n
            || self.damping.len() != n
            || self.saturation.len() != n
            || self.spiral.len() != n
            || !self.time_scale.is_finite()
            || self.time_scale <= 0.0
            || self
                .biases
                .iter()
                .any(|b| b.len() != d || b.iter().any(|x| !x.is_finite()))
            || self.damping.iter().any(|x| !x.is_finite())
            || self.saturation.iter().any(|x| !x.is_finite() || *x < 0.0)
            || self.spiral.iter().any(|m| {
                m.len() != d
                    || m.iter()
                        .any(|r| r.len() != d || r.iter().any(|x| !x.is_finite()))
                    || (0..d).any(|i| (0..d).any(|j| m[i][j] != -m[j][i]))
            })
            || self.couplings.iter().any(|row| {
                row.len() != n
                    || row.iter().any(|m| {
                        m.len() != d
                            || m.iter()
                                .any(|r| r.len() != d || r.iter().any(|x| !x.is_finite()))
                    })
            })
        {
            return Err(FieldError::InvalidDynamics);
        }
        Ok((n, d))
    }

    /// Set local damping for every subsystem layer. Positive values
    /// contract toward nominal; negative values excite — they drive
    /// the state outward, and belong paired with a saturation bound
    /// so the spiral lands on a limit cycle rather than the cap rim.
    pub fn with_damping(mut self, damping: Vec<f64>) -> Result<Self, FieldError> {
        if damping.len() != self.biases.len() || damping.iter().any(|x| !x.is_finite()) {
            return Err(FieldError::InvalidDynamics);
        }
        self.damping = damping;
        Ok(self)
    }

    /// Set non-negative isotropic nonlinear saturation for every layer.
    pub fn with_saturation(mut self, saturation: Vec<f64>) -> Result<Self, FieldError> {
        if saturation.len() != self.biases.len()
            || saturation.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err(FieldError::InvalidDynamics);
        }
        self.saturation = saturation;
        Ok(self)
    }

    /// Set a skew-symmetric spiral generator per layer. Each matrix must
    /// satisfy Jᵀ = −J exactly — only then is the drift a pure rotation,
    /// so damping alone decides the radius and the state traces a
    /// logarithmic spiral.
    pub fn with_spiral(mut self, spiral: Vec<Matrix>) -> Result<Self, FieldError> {
        if spiral.len() != self.biases.len() {
            return Err(FieldError::InvalidDynamics);
        }
        self.spiral = spiral;
        self.validate()?;
        Ok(self)
    }

    /// State-dependent Jacobian of the complete nonlinear vector field.
    ///
    /// For the local term -s ||x||² x, the derivative is
    /// -s (||x||² I + 2 x xᵀ). Coupling, spiral, and damping are added
    /// directly.
    pub fn jacobian(&self, layers: &[SubsystemLayer]) -> Result<Vec<Vec<Matrix>>, FieldError> {
        let (n, d) = self.validate()?;
        if layers.len() != n {
            return Err(FieldError::LayerMismatch);
        }
        if d == 0 || layers.iter().any(|l| l.state.len() != d) {
            return Err(FieldError::DimensionMismatch);
        }
        let mut out = vec![vec![vec![vec![0.0; d]; d]; n]; n];
        for (target, couplings) in self.couplings.iter().enumerate() {
            for (source, coupling) in couplings.iter().enumerate() {
                out[target][source] = coupling.clone();
            }
            let saturation = self.saturation[target];
            let r2 = if saturation == 0.0 {
                0.0
            } else {
                let value = dot(&layers[target].state, &layers[target].state);
                if !value.is_finite() {
                    return Err(FieldError::NonFiniteState);
                }
                value
            };
            let state = &layers[target].state;
            for (k, row) in out[target][target].iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell += self.spiral[target][k][j];
                    let nonlinear = if saturation == 0.0 {
                        0.0
                    } else {
                        let xk = state[k];
                        let xj = state[j];
                        saturation * ((if k == j { r2 } else { 0.0 }) + 2.0 * xk * xj)
                    };
                    *cell -= nonlinear;
                }
                row[k] -= self.damping[target];
            }
        }
        if out
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .any(|x| !x.is_finite())
        {
            return Err(FieldError::NonFiniteState);
        }
        Ok(out)
    }

    /// Conservative upper bound on instantaneous linear growth in the
    /// infinity norm. A negative value is a sufficient condition for
    /// contraction of the linearized dynamics; a non-negative value is not
    /// a proof of instability.
    ///
    /// This bound deliberately excludes nonlinear saturation. Saturation is
    /// state-dependent and can stabilize trajectories even when the linear
    /// origin is not contractive.
    pub fn linear_growth_bound(&self) -> Result<f64, FieldError> {
        let (n, d) = self.validate()?;
        if n == 0
            || d == 0
            || self.couplings.len() != n
            || self.couplings.iter().any(|row| row.len() != n)
        {
            return Err(FieldError::DimensionMismatch);
        }
        let mut bound = f64::NEG_INFINITY;
        for target in 0..n {
            for coordinate in 0..d {
                let mut row_sum = -self.damping[target];
                // A skew generator cannot contract, but its entries bound
                // how much it can amplify one coordinate in the worst
                // case, so it belongs in the conservative bound.
                for value in &self.spiral[target][coordinate] {
                    row_sum += value.abs();
                    if !row_sum.is_finite() {
                        return Err(FieldError::NonFiniteState);
                    }
                }
                for source in 0..n {
                    let row = &self.couplings[target][source][coordinate];
                    if row.len() != d {
                        return Err(FieldError::DimensionMismatch);
                    }
                    for value in row {
                        row_sum += value.abs();
                        if !row_sum.is_finite() {
                            return Err(FieldError::NonFiniteState);
                        }
                    }
                }
                bound = bound.max(row_sum);
            }
        }
        Ok(bound)
    }

    pub fn derivatives(&self, layers: &[SubsystemLayer]) -> Result<Vec<Vector>, FieldError> {
        let (n, d) = self.validate()?;
        if layers.len() != n {
            return Err(FieldError::LayerMismatch);
        }
        if layers.iter().any(|l| l.state.len() != d) {
            return Err(FieldError::DimensionMismatch);
        }

        let mut out = self.biases.clone();
        for (i, layer) in layers.iter().enumerate() {
            if self.saturation[i] != 0.0 {
                let radius_sq = dot(&layer.state, &layer.state);
                if !radius_sq.is_finite() {
                    return Err(FieldError::NonFiniteState);
                }
                for (dst, state) in out[i].iter_mut().zip(&layer.state) {
                    *dst -= self.saturation[i] * radius_sq * state;
                }
            }
            for (dst, state) in out[i].iter_mut().zip(&layer.state) {
                *dst -= self.damping[i] * state;
            }
            if self.spiral[i].iter().flatten().any(|v| *v != 0.0) {
                let wound = mat_vec(&self.spiral[i], &layer.state)?;
                for (dst, w) in out[i].iter_mut().zip(wound) {
                    *dst += w;
                }
            }
        }
        for (target, out_row) in out.iter_mut().enumerate() {
            for (source, layer) in layers.iter().enumerate() {
                let contribution = mat_vec(&self.couplings[target][source], &layer.state)?;
                for (dst, src) in out_row.iter_mut().zip(contribution) {
                    *dst += src;
                }
            }
        }
        if out.iter().flatten().any(|x| !x.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod numerical_integrity_tests {
    use super::*;

    #[test]
    fn dot_preserves_finite_cancellation_without_intermediate_overflow() {
        let value = dot(&[1.0e308, 1.0e308], &[1.0e308, -1.0e308]);
        assert_eq!(value, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_bound_rejects_overflowing_accumulation() {
        let mut engine = SystemEngine::zero(1, 2);
        engine.couplings[0][0][0][0] = f64::MAX;
        engine.couplings[0][0][0][1] = f64::MAX;
        assert_eq!(
            engine.linear_growth_bound(),
            Err(FieldError::NonFiniteState)
        );
    }

    #[test]
    fn jacobian_matches_finite_difference_for_nonlinear_layer() {
        let engine = SystemEngine::zero(1, 2)
            .with_damping(vec![0.4])
            .unwrap()
            .with_saturation(vec![0.7])
            .unwrap();
        let layers = vec![SubsystemLayer::new("a", vec![0.2, -0.3])];
        let analytic = engine.jacobian(&layers).unwrap()[0][0].clone();
        let eps = 1e-7;
        let base = engine.derivatives(&layers).unwrap()[0].clone();
        for j in 0..2 {
            let mut perturbed = layers.clone();
            perturbed[0].state[j] += eps;
            let value = engine.derivatives(&perturbed).unwrap()[0].clone();
            for (i, analytic_row) in analytic.iter().enumerate() {
                let finite_difference = (value[i] - base[i]) / eps;
                assert!((analytic_row[j] - finite_difference).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn damped_engine_reduces_unforced_state() {
        let engine = SystemEngine::zero(1, 1).with_damping(vec![2.0]).unwrap();
        let layers = vec![SubsystemLayer::new("a", vec![0.5])];
        let d = engine.derivatives(&layers).unwrap();
        assert!((d[0][0] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn saturation_dissipates_radial_growth() {
        let engine = SystemEngine::zero(1, 1).with_saturation(vec![2.0]).unwrap();
        let layers = vec![SubsystemLayer::new("a", vec![2.0])];
        let d = engine.derivatives(&layers).unwrap();
        assert!(d[0][0] < 0.0);
        assert!((d[0][0] + 16.0).abs() < 1e-12);
    }

    #[test]
    fn isotropic_saturation_limits_growth_direction() {
        let engine = SystemEngine::zero(1, 2).with_saturation(vec![2.0]).unwrap();
        let layers = vec![SubsystemLayer::new("a", vec![2.0, 1.0])];
        let d = engine.derivatives(&layers).unwrap();
        let radius_sq: f64 = 5.0;
        assert!((d[0][0] + 20.0).abs() < 1e-12);
        assert!((d[0][1] + 10.0).abs() < 1e-12);
        assert_eq!(d[0][0] / d[0][1], layers[0].state[0] / layers[0].state[1]);
        assert!((radius_sq - 5.0).abs() < 1e-12);
    }

    #[test]
    fn coupled_growth_bound_includes_cross_subsystem_gain() {
        let engine = SystemEngine::new(
            vec![
                vec![vec![vec![0.0]], vec![vec![2.0]]],
                vec![vec![vec![3.0]], vec![vec![0.0]]],
            ],
            vec![vec![0.0], vec![0.0]],
            1.0,
        )
        .unwrap()
        .with_damping(vec![1.0, 4.0])
        .unwrap();
        assert_eq!(engine.linear_growth_bound().unwrap(), 1.0);
    }

    #[test]
    fn negative_linear_growth_bound_is_a_contraction_certificate() {
        let engine = SystemEngine::zero(2, 1)
            .with_damping(vec![2.0, 3.0])
            .unwrap();
        let bound = engine.linear_growth_bound().unwrap();
        assert_eq!(bound, -2.0);
    }

    #[test]
    fn zero_saturation_handles_large_finite_state_without_zero_times_infinity() {
        let engine = SystemEngine::zero(1, 2);
        let layers = vec![SubsystemLayer::new("a", vec![1.0e308, -1.0e308])];
        let jacobian = engine.jacobian(&layers).unwrap();
        assert_eq!(jacobian[0][0], vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        let derivatives = engine.derivatives(&layers).unwrap();
        assert_eq!(derivatives, vec![vec![0.0, 0.0]]);
    }

    #[test]
    fn spiral_winds_state_without_changing_its_radius() {
        // A skew generator alone is a pure rotation: the derivative
        // is perpendicular to the state, so the radial component of
        // the drift is exactly zero.
        let j = vec![vec![0.0, -1.0], vec![1.0, 0.0]];
        let engine = SystemEngine::zero(1, 2).with_spiral(vec![j]).unwrap();
        let layers = vec![SubsystemLayer::new("a", vec![0.3, 0.4])];
        let d = engine.derivatives(&layers).unwrap();
        assert!((d[0][0] + 0.4).abs() < 1e-12);
        assert!((d[0][1] - 0.3).abs() < 1e-12);
        let radial = layers[0].state[0] * d[0][0] + layers[0].state[1] * d[0][1];
        assert!(radial.abs() < 1e-12);
    }

    #[test]
    fn damped_spiral_contracts_while_rotating() {
        // Damping plus a skew generator is the loxodrome: each step
        // shrinks the radius and turns the phase — a logarithmic
        // spiral into the pole.
        let j = vec![vec![0.0, -0.5], vec![0.5, 0.0]];
        let engine = SystemEngine::zero(1, 2)
            .with_damping(vec![0.5])
            .unwrap()
            .with_spiral(vec![j])
            .unwrap();
        let layers = vec![SubsystemLayer::new("a", vec![0.3, 0.0])];
        let d = engine.derivatives(&layers).unwrap();
        // Radial inflow (−0.5·0.3 = −0.15) plus winding (+0.5·0.3
        // into the second coordinate).
        assert!((d[0][0] + 0.15).abs() < 1e-12);
        assert!((d[0][1] - 0.15).abs() < 1e-12);
    }

    #[test]
    fn spiral_must_be_skew_symmetric() {
        let bad = vec![vec![0.0, -1.0], vec![-1.0, 0.0]];
        assert_eq!(
            SystemEngine::zero(1, 2).with_spiral(vec![bad]),
            Err(FieldError::InvalidDynamics)
        );
        let mut engine = SystemEngine::zero(1, 2);
        engine.spiral[0][0][1] = 1.0;
        let layers = vec![SubsystemLayer::new("a", vec![0.1, 0.2])];
        assert_eq!(
            engine.derivatives(&layers),
            Err(FieldError::InvalidDynamics)
        );
    }

    #[test]
    fn outward_spiral_settles_onto_a_limit_cycle() {
        // Negative damping winds the state out from the pole; the
        // isotropic saturation term bounds it. The fixed point of the
        // radial law ṙ = αr − s·r³ is r* = √(α/s) — a stable ring:
        // inside it the drift pushes out, outside it pulls back.
        let j = vec![vec![0.0, -1.0], vec![1.0, 0.0]];
        let engine = SystemEngine::zero(1, 2)
            .with_damping(vec![-0.5])
            .unwrap()
            .with_spiral(vec![j])
            .unwrap()
            .with_saturation(vec![2.0])
            .unwrap();
        // r* = √(0.5/2) = 0.5. Below it, the radial drift is outward.
        let inner = vec![SubsystemLayer::new("a", vec![0.2, 0.0])];
        let d_in = engine.derivatives(&inner).unwrap();
        let radial_in = inner[0].state[0] * d_in[0][0] + inner[0].state[1] * d_in[0][1];
        assert!(radial_in > 0.0, "inside the ring the spiral pushes out");
        // Above it, saturation dominates — the drift is inward.
        let outer = vec![SubsystemLayer::new("a", vec![0.9, 0.0])];
        let d_out = engine.derivatives(&outer).unwrap();
        let radial_out = outer[0].state[0] * d_out[0][0] + outer[0].state[1] * d_out[0][1];
        assert!(radial_out < 0.0, "outside the ring saturation pulls in");
        // And on it, the radial component vanishes: pure winding.
        let on = vec![SubsystemLayer::new("a", vec![0.5, 0.0])];
        let d_on = engine.derivatives(&on).unwrap();
        let radial_on = on[0].state[0] * d_on[0][0] + on[0].state[1] * d_on[0][1];
        assert!(
            radial_on.abs() < 1e-12,
            "on the ring the drift is pure rotation"
        );
    }

    #[test]
    fn public_engine_mutation_cannot_bypass_dynamics_validation() {
        let mut engine = SystemEngine::zero(1, 1);
        engine.damping = vec![f64::NAN];
        let layers = vec![SubsystemLayer::new("a", vec![0.1])];
        assert_eq!(
            engine.derivatives(&layers),
            Err(FieldError::InvalidDynamics)
        );
        assert_eq!(engine.jacobian(&layers), Err(FieldError::InvalidDynamics));
        assert_eq!(
            engine.linear_growth_bound(),
            Err(FieldError::InvalidDynamics)
        );
    }

    #[test]
    fn coupled_engine_produces_state_derivatives() {
        let engine = SystemEngine::new(
            vec![
                vec![vec![vec![0.5]], vec![vec![1.0]]],
                vec![vec![vec![-1.0]], vec![vec![0.0]]],
            ],
            vec![vec![0.0], vec![0.0]],
            1.0,
        )
        .unwrap();
        let layers = vec![
            SubsystemLayer::new("a", vec![0.1]),
            SubsystemLayer::new("b", vec![0.2]),
        ];
        let d = engine.derivatives(&layers).unwrap();
        assert!((d[0][0] - 0.25).abs() < 1e-12);
        assert!((d[1][0] + 0.1).abs() < 1e-12);
    }
}
