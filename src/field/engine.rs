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
        Ok(Self { couplings, biases, damping: vec![0.0; n], time_scale })
    }

    pub fn dimension(&self) -> usize { self.biases.first().map_or(0, Vector::len) }

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

    pub fn derivatives(&self, layers: &[SubsystemLayer]) -> Result<Vec<Vector>, FieldError> {
        if layers.len() != self.biases.len() {
            return Err(FieldError::LayerMismatch);
        }
        let d = self.biases.first().map_or(0, Vector::len);
        if layers.iter().any(|l| l.state.len() != d) {
            return Err(FieldError::DimensionMismatch);
        }

        let mut out = self.biases.clone();
        for (i, layer) in layers.iter().enumerate() {
            for (dst, state) in out[i].iter_mut().zip(&layer.state) {
                *dst -= self.damping[i] * state;
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
    fn damped_engine_reduces_unforced_state() {
        let engine = SystemEngine::zero(1, 1).with_damping(vec![2.0]).unwrap();
        let layers = vec![SubsystemLayer::new("a", vec![0.5])];
        let d = engine.derivatives(&layers).unwrap();
        assert!((d[0][0] + 1.0).abs() < 1e-12);
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
