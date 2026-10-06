//! Native system dynamics.
//!
//! A SystemEngine owns the state-transition law of a major system. Layers
//! provide state; the engine determines how those states change together.
//! This keeps dynamics separate from representation and from interpretation.

use super::{FieldError, Matrix, SubsystemLayer, Vector};

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn mat_vec(m: &[Vector], x: &[f64]) -> Result<Vector, FieldError> {
    if m.is_empty() || m.iter().any(|r| r.len() != x.len()) {
        return Err(FieldError::DimensionMismatch);
    }
    Ok(m.iter().map(|row| dot(row, x)).collect())
}

/// Coupled continuous-time dynamical system over subsystem layers.
///
/// couplings[target][source] is the linear local Jacobian contribution
/// from one layer into another. Nonlinear/state-dependent engines can be
/// layered on top of this interface without changing the field topology.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemEngine {
    pub couplings: Vec<Vec<Matrix>>,
    pub biases: Vec<Vector>,
    /// Per-layer dissipative coefficient. Positive values make the local
    /// dynamics contractive; zero preserves the undamped linear system.
    pub damping: Vec<f64>,
    /// Isotropic nonlinear saturation coefficient per layer. The term is
    /// -saturation * ||state||² * state, which bounds growth without choosing
    /// a coordinate-specific activation function.
    pub saturation: Vec<f64>,
    pub time_scale: f64,
}

impl SystemEngine {
    pub fn zero(layer_count: usize, dimension: usize) -> Self {
        Self {
            couplings: vec![
                vec![vec![vec![0.0; dimension]; dimension]; layer_count];
                layer_count
            ],
            biases: vec![vec![0.0; dimension]; layer_count],
            damping: vec![0.0; layer_count],
            saturation: vec![0.0; layer_count],
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
        if n == 0 || d == 0 || couplings.len() != n
            || couplings.iter().any(|row| row.len() != n)
            || biases.iter().any(|b| b.len() != d || b.iter().any(|x| !x.is_finite()))
            || couplings.iter().flatten().any(|m| {
                m.len() != d || m.iter().any(|r| r.len() != d || r.iter().any(|x| !x.is_finite()))
            })
        {
            return Err(FieldError::DimensionMismatch);
        }
        Ok(Self {
            couplings,
            biases,
            damping: vec![0.0; n],
            saturation: vec![0.0; n],
            time_scale,
        })
    }

    pub fn dimension(&self) -> usize { self.biases.first().map_or(0, Vector::len) }
    fn validate(&self) -> Result<(usize, usize), FieldError> {
        let n = self.biases.len();
        let d = self.biases.first().map_or(0, Vector::len);
        if n == 0 || d == 0
            || self.couplings.len() != n
            || self.damping.len() != n
            || self.saturation.len() != n
            || !self.time_scale.is_finite()
            || self.time_scale <= 0.0
            || self.biases.iter().any(|b| b.len() != d || b.iter().any(|x| !x.is_finite()))
            || self.damping.iter().any(|x| !x.is_finite() || *x < 0.0)
            || self.saturation.iter().any(|x| !x.is_finite() || *x < 0.0)
            || self.couplings.iter().any(|row| {
                row.len() != n || row.iter().any(|m| {
                    m.len() != d || m.iter().any(|r| {
                        r.len() != d || r.iter().any(|x| !x.is_finite())
                    })
                })
            })
        {
            return Err(FieldError::InvalidDynamics);
        }
        Ok((n, d))
    }

    /// Set non-negative local damping for every subsystem layer.
    pub fn with_damping(mut self, damping: Vec<f64>) -> Result<Self, FieldError> {
        if damping.len() != self.biases.len()
            || damping.iter().any(|x| !x.is_finite() || *x < 0.0)
        {
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

    /// State-dependent Jacobian of the complete nonlinear vector field.
    ///
    /// For the local term -s ||x||² x, the derivative is
    /// -s (||x||² I + 2 x xᵀ). Coupling and damping are added directly.
    pub fn jacobian(&self, layers: &[SubsystemLayer]) -> Result<Vec<Vec<Matrix>>, FieldError> {
        let (n, d) = self.validate()?;
        if layers.len() != n { return Err(FieldError::LayerMismatch); }
        if d == 0 || layers.iter().any(|l| l.state.len() != d) {
            return Err(FieldError::DimensionMismatch);
        }
        let mut out = vec![vec![vec![vec![0.0; d]; d]; n]; n];
        for target in 0..n {
            for source in 0..n {
                out[target][source] = self.couplings[target][source].clone();
            }
            let r2 = dot(&layers[target].state, &layers[target].state);
            for k in 0..d {
                for j in 0..d {
                    let xk = layers[target].state[k];
                    let xj = layers[target].state[j];
                    let nonlinear = self.saturation[target]
                        * (if k == j { r2 } else { 0.0 } + 2.0 * xk * xj);
                    out[target][target][k][j] -= nonlinear;
                }
                out[target][target][k][k] -= self.damping[target];
            }
        }
        if out.iter().flatten().flatten().flatten().any(|x| !x.is_finite()) {
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
        self.validate()?;
        let (n, d) = self.validate()?;
        if n == 0 || d == 0 || self.couplings.len() != n
            || self.couplings.iter().any(|row| row.len() != n)
        {
            return Err(FieldError::DimensionMismatch);
        }
        let mut bound = f64::NEG_INFINITY;
        for target in 0..n {
            for coordinate in 0..d {
                let mut row_sum = -self.damping[target];
                for source in 0..n {
                    let row = &self.couplings[target][source][coordinate];
                    if row.len() != d {
                        return Err(FieldError::DimensionMismatch);
                    }
                    for value in row {
                        row_sum += value.abs();
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
            let radius_sq = dot(&layer.state, &layer.state);
            for (dst, state) in out[i].iter_mut().zip(&layer.state) {
                *dst -= self.damping[i] * state;
                *dst -= self.saturation[i] * radius_sq * state;
            }
        }
        for target in 0..layers.len() {
            for source in 0..layers.len() {
                let contribution = mat_vec(
                    &self.couplings[target][source],
                    &layers[source].state,
                )?;
                for (dst, src) in out[target].iter_mut().zip(contribution) {
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
mod tests {
    use super::*;

    #[test]
    fn jacobian_matches_finite_difference_for_nonlinear_layer() {
        let engine = SystemEngine::zero(1, 2)
            .with_damping(vec![0.4]).unwrap()
            .with_saturation(vec![0.7]).unwrap();
        let layers = vec![SubsystemLayer::new("a", vec![0.2, -0.3])];
        let analytic = engine.jacobian(&layers).unwrap()[0][0].clone();
        let eps = 1e-7;
        let base = engine.derivatives(&layers).unwrap()[0].clone();
        for j in 0..2 {
            let mut perturbed = layers.clone();
            perturbed[0].state[j] += eps;
            let value = engine.derivatives(&perturbed).unwrap()[0].clone();
            for i in 0..2 {
                let finite_difference = (value[i] - base[i]) / eps;
                assert!((analytic[i][j] - finite_difference).abs() < 1e-6);
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
        let engine = SystemEngine::zero(1, 1)
            .with_saturation(vec![2.0]).unwrap();
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
        let radius_sq = 5.0;
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
            vec![vec![0.0], vec![vec![0.0]]],
            1.0,
        ).unwrap().with_damping(vec![1.0, 4.0]).unwrap();
        assert_eq!(engine.linear_growth_bound().unwrap(), 2.0);
    }

    #[test]
    fn negative_linear_growth_bound_is_a_contraction_certificate() {
        let engine = SystemEngine::zero(2, 1)
            .with_damping(vec![2.0, 3.0]).unwrap();
        let bound = engine.linear_growth_bound().unwrap();
        assert_eq!(bound, -2.0);
    }

    #[test]
    fn public_engine_mutation_cannot_bypass_dynamics_validation() {
        let mut engine = SystemEngine::zero(1, 1);
        engine.damping = vec![-1.0];
        let layers = vec![SubsystemLayer::new("a", vec![0.1])];
        assert_eq!(engine.derivatives(&layers), Err(FieldError::InvalidDynamics));
        assert_eq!(engine.jacobian(&layers), Err(FieldError::InvalidDynamics));
        assert_eq!(engine.linear_growth_bound(), Err(FieldError::InvalidDynamics));
    }

    #[test]
    fn coupled_engine_produces_state_derivatives() {
        let engine = SystemEngine::new(
            vec![
                vec![vec![vec![0.5]], vec![vec![1.0]]],
                vec![vec![vec![-1.0]], vec![vec![0.0]]],
            ],
            vec![vec![0.0], vec![vec![0.0]]],
            1.0,
        ).unwrap();
        let layers = vec![
            SubsystemLayer::new("a", vec![0.1]),
            SubsystemLayer::new("b", vec![0.2]),
        ];
        let d = engine.derivatives(&layers).unwrap();
        assert!((d[0][0] - 0.25).abs() < 1e-12);
        assert!((d[1][0] + 0.1).abs() < 1e-12);
    }
}
