//! Runtime bridge for the multi-fiber Genesis architecture.
//!
//! Each major subsystem publishes a normalized local projection into one
//! shared latent coordinate space. The global field measures and reduces
//! cross-system disagreement. The persisted core state remains authoritative;
//! this is the integration substrate rather than a second source of truth.

use crate::state::GenesisCoreState;
use crate::state::neurochemical::NeurochemicalId;

use super::{
    FieldError, GaugeTransport, GlobalField, PoincareBall, SubsystemLayer, SystemEngine,
    SystemFiber, Vector,
};

pub const LATENT_DIMENSION: usize = 6;

#[derive(Clone, Debug, PartialEq)]
pub struct FieldIntegration {
    pub consensus: Vector,
    pub disagreement: f64,
    pub fiber_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldRuntime {
    field: GlobalField,
}

impl FieldRuntime {
    pub fn new() -> Result<Self, FieldError> {
        let manifold = PoincareBall::new(LATENT_DIMENSION, 1.0)?;
        let mut field = GlobalField::new(manifold.clone());

        for id in ["body", "prediction", "agency"] {
            let layer = SubsystemLayer::new(format!("{id}:latent"), vec![0.0; LATENT_DIMENSION]);
            let engine = SystemEngine::zero(1, LATENT_DIMENSION);
            field.add_fiber(SystemFiber::with_engine(
                id,
                manifold.clone(),
                vec![layer],
                engine,
            )?)?;
        }

        for (from, to) in [
            ("body", "prediction"),
            ("prediction", "agency"),
            ("agency", "body"),
        ] {
            field.add_transport(GaugeTransport::new(from, to, identity(LATENT_DIMENSION))?)?;
        }

        Ok(Self { field })
    }

    pub fn field(&self) -> &GlobalField {
        &self.field
    }

    pub fn observe(
        &mut self,
        state: &GenesisCoreState,
        dt: f64,
    ) -> Result<FieldIntegration, FieldError> {
        let projections = [
            ("body", body_projection(state)),
            ("prediction", prediction_projection(state)),
            ("agency", agency_projection(state)),
        ];

        for (id, projection) in projections {
            let fiber = self
                .field
                .fibers
                .iter_mut()
                .find(|fiber| fiber.id == id)
                .ok_or(FieldError::MissingFiber)?;
            fiber.layers[0].state = fiber.manifold.project(&projection)?;
        }

        let consensus = self.field.synchronize(dt)?.ok_or(FieldError::EmptyState)?;
        let disagreement = self.field.disagreement;

        Ok(FieldIntegration {
            consensus,
            disagreement,
            fiber_count: self.field.fibers.len(),
        })
    }
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    matrix
}

fn unit(value: f32, min: f32, max: f32) -> f64 {
    if !value.is_finite() {
        return 0.5;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0) as f64
}

fn effective(state: &GenesisCoreState, id: NeurochemicalId) -> f32 {
    state.neurochemicals.effective(id).clamp(0.0, 2.0)
}

fn body_projection(state: &GenesisCoreState) -> Vector {
    let n = effective(state, NeurochemicalId::Norepinephrine);
    let d = effective(state, NeurochemicalId::Dopamine);
    let s = effective(state, NeurochemicalId::Serotonin);
    let b = effective(state, NeurochemicalId::BDNF);
    let c = effective(state, NeurochemicalId::Cortisol);
    let valence = state.neurochemicals.valence.clamp(-1.0, 1.0);

    vec![
        unit(valence, -1.0, 1.0),
        unit(n, 0.0, 2.0),
        unit(d, 0.0, 2.0),
        unit(s, 0.0, 2.0),
        unit(b, 0.0, 2.0),
        unit(c, 0.0, 2.0),
    ]
}

fn prediction_projection(state: &GenesisCoreState) -> Vector {
    let s = &state.inference_signals;
    vec![
        s.precision.clamp(0.0, 1.0) as f64,
        s.surprise_ema.clamp(0.0, 1.0) as f64,
        s.free_energy.clamp(0.0, 1.0) as f64,
        s.expected_free_energy.clamp(0.0, 1.0) as f64,
        s.allostasis_load.clamp(0.0, 1.0) as f64,
        s.model_maturity.clamp(0.0, 1.0) as f64,
    ]
}

fn agency_projection(state: &GenesisCoreState) -> Vector {
    let s = &state.inference_signals;
    let dopamine = effective(state, NeurochemicalId::Dopamine);
    let norepinephrine = effective(state, NeurochemicalId::Norepinephrine);

    vec![
        s.policy_authority.clamp(0.0, 1.0) as f64,
        unit(dopamine, 0.0, 2.0),
        unit(norepinephrine, 0.0, 2.0),
        s.user_engagement.clamp(0.0, 1.0) as f64,
        s.attunement.clamp(0.0, 1.0) as f64,
        ((s.dyadic_synchrony + 1.0) * 0.5).clamp(0.0, 1.0) as f64,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_has_explicit_major_systems() {
        let runtime = FieldRuntime::new().unwrap();
        let ids: Vec<_> = runtime
            .field()
            .fibers
            .iter()
            .map(|f| f.id.as_str())
            .collect();
        assert_eq!(ids, vec!["body", "prediction", "agency"]);
        assert_eq!(runtime.field().transports.len(), 3);
    }

    #[test]
    fn observation_integrates_real_state_projections() {
        let state = GenesisCoreState::new(1, 1_000);
        let mut runtime = FieldRuntime::new().unwrap();

        let result = runtime.observe(&state, 0.2).unwrap();

        assert_eq!(result.fiber_count, 3);
        assert_eq!(result.consensus.len(), LATENT_DIMENSION);
        assert!(result.consensus.iter().all(|v| v.is_finite()));
        assert!(result.disagreement.is_finite());
        assert!(result.disagreement >= 0.0);
    }

    #[test]
    fn observation_tracks_changed_prediction_state() {
        let mut state = GenesisCoreState::new(1, 1_000);
        let mut runtime = FieldRuntime::new().unwrap();

        let before = runtime.observe(&state, 0.2).unwrap();
        state.inference_signals.surprise_ema = 0.9;
        state.inference_signals.free_energy = 0.8;
        let after = runtime.observe(&state, 0.2).unwrap();

        assert_ne!(before.consensus, after.consensus);
    }
}
