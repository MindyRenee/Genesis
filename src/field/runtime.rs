//! Runtime bridge for the multi-fiber Genesis architecture.
//!
//! Each major subsystem publishes a normalized local projection into one
//! shared latent coordinate space. The global field measures and reduces
//! cross-system disagreement. The persisted core state remains authoritative;
//! this is the integration substrate rather than a second source of truth.

use crate::state::GenesisCoreState;
use crate::state::MentalPhase;
use crate::state::neurochemical::NeurochemicalId;

use super::{
    FieldError, GaugeTransport, GlobalField, StereographicSphere, SubsystemLayer, SystemEngine,
    SystemFiber, Vector,
};

/// Provenance marker — one letter of the creator lattice (see
/// PROVENANCE.md).
pub(crate) const MARKER_G01: u64 = 0x4753_014E_4443_AD12;

pub const LATENT_DIMENSION: usize = 6;

/// Observation coupling timescale (seconds). Each fiber moves a
/// fraction `1 − e^(−dt/τ)` along the geodesic toward its observed
/// projection per call — a first-order manifold filter: the
/// observation informs the field's own state instead of
/// overwriting it.
const OBSERVATION_TAU_SECONDS: f64 = 0.5;

/// Per-fiber homeostatic relaxation times (seconds). Between
/// observations each fiber's engine drifts its state toward nominal
/// — the sphere's pole — at its own clearance rate: an
/// Ornstein–Uhlenbeck inflow in the ambient coordinates the engine
/// integrates, wound by a skew generator so the relaxation traces a
/// loxodrome — a constant-bearing spiral into the pole, arcing
/// inward along the sphere's own positive curvature — projected
/// back into the cap. This is the honest prior for unobserved
/// physiology: monoamines clear in minutes, loads decay, no state
/// holds forever. The observation stays the authority — the drift
/// only fills the gaps it leaves. Rates are first-pass physiological
/// guesses ordered by how slowly each subsystem's measured variables
/// genuinely relax: neurochemical clearance is minutes-scale, the
/// inference signals are faster statistics, the social variables
/// linger longest. The winding rate matches each fiber's own τ —
/// a 45° loxodrome, one radian of turn per e-fold of decay.
const BODY_RELAX_TAU_S: f64 = 120.0;
const PREDICTION_RELAX_TAU_S: f64 = 40.0;
const AGENCY_RELAX_TAU_S: f64 = 60.0;

#[derive(Clone, Debug, PartialEq)]
pub struct FieldIntegration {
    pub consensus: Vector,
    pub disagreement: f64,
    /// The field's thermal state after this step — disagreement
    /// plus the physical bath, cooling at its own rate. The
    /// Langevin kick amplitude for the *next* step is set by it.
    pub temperature: f64,
    pub fiber_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FieldRuntime {
    field: GlobalField,
}

impl FieldRuntime {
    pub fn new() -> Result<Self, FieldError> {
        let manifold = StereographicSphere::new(LATENT_DIMENSION, 1.0)?;
        let mut field = GlobalField::new(manifold.clone());

        for (id, relax_tau) in [
            ("body", BODY_RELAX_TAU_S),
            ("prediction", PREDICTION_RELAX_TAU_S),
            ("agency", AGENCY_RELAX_TAU_S),
        ] {
            let layer = SubsystemLayer::new(format!("{id}:latent"), vec![0.0; LATENT_DIMENSION]);
            let engine = SystemEngine::zero(1, LATENT_DIMENSION)
                .with_damping(vec![1.0 / relax_tau])
                .expect("positive finite damping always validates")
                .with_spiral(vec![loxodrome(LATENT_DIMENSION, 0, 1, 1.0 / relax_tau)])
                .expect("a skew generator always validates");
            field.add_fiber(SystemFiber::with_engine(
                id,
                manifold.clone(),
                vec![layer],
                engine,
            )?)?;
        }

        for (from, to, matrix) in [
            // The afferent edge is the non-flat one: somatic
            // coordinates are *read in* a rotated inferential frame
            // (the generative model's axes are not the raw body
            // axes), so the triangle carries real holonomy. The
            // reverse edge carries the inverse rotation, keeping
            // the pair's own two-way transport consistent.
            (
                "body",
                "prediction",
                rotation(LATENT_DIMENSION, 0, 1, AFFERENT_FRAME_OFFSET),
            ),
            ("prediction", "agency", identity(LATENT_DIMENSION)),
            ("agency", "body", identity(LATENT_DIMENSION)),
            (
                "prediction",
                "body",
                rotation(LATENT_DIMENSION, 0, 1, -AFFERENT_FRAME_OFFSET),
            ),
            ("agency", "prediction", identity(LATENT_DIMENSION)),
            ("body", "agency", identity(LATENT_DIMENSION)),
        ] {
            field.add_transport(GaugeTransport::new(from, to, matrix)?)?;
        }

        Ok(Self { field })
    }

    pub fn field(&self) -> &GlobalField {
        &self.field
    }

    /// Predict → correct → integrate, transactionally on a clone:
    /// any failure leaves the live field untouched rather than
    /// half-observed.
    ///
    /// `bath` is the substrate's measured thermal level in latent²
    /// units (the caller derives it from the physical probe — the
    /// silicon's own timing fluctuation). It floors the field
    /// temperature alongside internal disagreement: the reservoir
    /// the Langevin kick samples is real, not only modeled.
    pub fn observe(
        &mut self,
        state: &GenesisCoreState,
        dt: f64,
        bath: f64,
    ) -> Result<FieldIntegration, FieldError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        let mut next = self.field.clone();

        // Prediction: each fiber advances by its own law — the OU
        // homeostatic drift of its engine, so the field carries a
        // trajectory between observations rather than a frozen
        // snapshot.
        for fiber in &mut next.fibers {
            fiber.step(dt)?;
        }

        // Correction: the observation is an offering, not a command.
        // Each fiber's latent layer moves a bounded fraction along
        // the geodesic toward the observed point — the field carries
        // its own state forward and lets measurement perturb it.
        let gain = 1.0 - (-dt / OBSERVATION_TAU_SECONDS).exp();
        for (id, projection) in [
            ("body", body_projection(state)),
            ("prediction", prediction_projection(state)),
            ("agency", agency_projection(state)),
        ] {
            let fiber = next
                .fibers
                .iter_mut()
                .find(|fiber| fiber.id == id)
                .ok_or(FieldError::MissingFiber)?;
            let manifold = fiber.manifold.clone();
            let observation = manifold.project(&projection)?;
            let current = fiber.layers[0].state.clone();
            fiber.layers[0].state = manifold.geodesic(&current, &observation, gain)?;
        }

        // Integration: while awake, the noradrenergic forebrain
        // coupling pulls the fibers toward consensus — the bounded
        // pull is the body's coherence mechanism. In REM the locus
        // coeruleus is silent and the systems decorrelate, so the
        // honest integration is ascent on the same disagreement
        // potential — each fiber drifts by its own dynamics rather
        // than being held to agreement. Same measured consensus and
        // thermal bookkeeping either way; only the direction the
        // field moves changes with the phase its own carrier runs on.
        let consensus = if state.zones.phase() == MentalPhase::REM {
            next.diverge(dt, bath)?.ok_or(FieldError::EmptyState)?
        } else {
            next.synchronize(dt, bath)?.ok_or(FieldError::EmptyState)?
        };
        let integration = FieldIntegration {
            consensus,
            disagreement: next.disagreement,
            temperature: next.temperature,
            fiber_count: next.fibers.len(),
        };
        self.field = next;
        Ok(integration)
    }
}

/// Frame offset between the body's somatic axes and the prediction
/// system's inferential axes — the one bend in the connection,
/// small enough to stay a frame mismatch and large enough for
/// `path_disagreement` to measure.
const AFFERENT_FRAME_OFFSET: f64 = 0.35;

/// Givens rotation in the (i, j) coordinate plane — the minimal
/// non-identity transport: a fixed frame offset between two
/// systems' latent bases, orthogonal so it preserves norms.
fn rotation(n: usize, i: usize, j: usize, theta: f64) -> Vec<Vec<f64>> {
    let mut m = identity(n);
    let (c, s) = (theta.cos(), theta.sin());
    m[i][i] = c;
    m[j][j] = c;
    m[i][j] = -s;
    m[j][i] = s;
    m
}

fn identity(n: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];
    for (i, row) in matrix.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    matrix
}

/// The loxodromic drift generator: a skew-symmetric rotation in the
/// (i, j) coordinate plane at angular rate `omega` — the winding
/// half of the homeostatic spiral. With damping at the same rate it
/// traces a 45° loxodrome into the pole: each e-fold of radial decay
/// carries one radian of turn.
fn loxodrome(n: usize, i: usize, j: usize, omega: f64) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    m[i][j] = -omega;
    m[j][i] = omega;
    m
}

fn unit(value: f32, min: f32, max: f32) -> f64 {
    if !value.is_finite() {
        return 0.5;
    }
    ((value - min) / (max - min)).clamp(0.0, 1.0) as f64
}

/// Signed latent half-width. Normalized channels live in [0,1];
/// the latent coordinate centers them on the origin at ±0.25 —
/// neutral state is the sphere's pole, and the corner of the
/// operating box (√6·0.25 ≈ 0.61) stays well inside the
/// hemisphere cap, where the log/exp maps are accurate instead of
/// being radially clamped onto the equator rim.
const LATENT_HALF: f64 = 0.25;

fn latent(unit: f64) -> f64 {
    (unit - 0.5) * 2.0 * LATENT_HALF
}

fn unit_finite(value: f32) -> f64 {
    crate::state::sanitize::finite_clamp(value, 0.0, 1.0) as f64
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
        latent(unit(valence, -1.0, 1.0)),
        latent(unit(n, 0.0, 2.0)),
        latent(unit(d, 0.0, 2.0)),
        latent(unit(s, 0.0, 2.0)),
        latent(unit(b, 0.0, 2.0)),
        latent(unit(c, 0.0, 2.0)),
    ]
}

fn prediction_projection(state: &GenesisCoreState) -> Vector {
    let s = &state.inference_signals;
    vec![
        latent(unit_finite(s.precision)),
        latent(unit_finite(s.surprise_ema)),
        latent(unit_finite(s.free_energy)),
        latent(unit_finite(s.expected_free_energy)),
        latent(unit_finite(s.allostasis_load)),
        latent(unit_finite(s.model_maturity)),
    ]
}

fn agency_projection(state: &GenesisCoreState) -> Vector {
    let s = &state.inference_signals;
    let dopamine = effective(state, NeurochemicalId::Dopamine);
    let norepinephrine = effective(state, NeurochemicalId::Norepinephrine);

    vec![
        latent(unit_finite(s.policy_authority)),
        latent(unit(dopamine, 0.0, 2.0)),
        latent(unit(norepinephrine, 0.0, 2.0)),
        latent(unit_finite(s.user_engagement)),
        latent(unit_finite(s.attunement)),
        latent(unit_finite((s.dyadic_synchrony + 1.0) * 0.5)),
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
        // Both directions of every edge — alternative routes exist,
        // so path-dependence is measurable.
        assert_eq!(runtime.field().transports.len(), 6);
    }

    #[test]
    fn alternative_routes_make_holonomy_measurable() {
        let runtime = FieldRuntime::new().unwrap();
        // Every fiber pair has exactly two routes, and the curved
        // afferent edge means each pair's routes disagree — the
        // connection's holonomy is measured, not just permitted.
        for (from, to) in [
            ("body", "prediction"),
            ("prediction", "agency"),
            ("agency", "body"),
            ("prediction", "body"),
            ("agency", "prediction"),
            ("body", "agency"),
        ] {
            let disagreement = runtime
                .field()
                .path_disagreement(from, to)
                .unwrap()
                .expect("two routes must exist");
            assert_eq!(disagreement.paths.len(), 2);
            assert!(
                disagreement.tangent_spread > 1.0e-3,
                "{from}->{to} should disagree across routes: {}",
                disagreement.tangent_spread
            );
        }
    }

    #[test]
    fn observation_blends_into_fiber_state_instead_of_overwriting() {
        // A projection is an offering: the fiber moves a bounded
        // fraction toward it along the geodesic, keeping its own
        // state between calls rather than being overwritten.
        let mut state = GenesisCoreState::new(1, 1_000);
        let mut runtime = FieldRuntime::new().unwrap();
        state.inference_signals.surprise_ema = 1.0;
        state.inference_signals.free_energy = 1.0;

        runtime.observe(&state, 0.1, 0.0).unwrap();
        let projection = prediction_projection(&state);
        let layer_state = &runtime.field().fibers[1].layers[0].state;
        let proj_norm: f64 = projection.iter().map(|v| v * v).sum::<f64>().sqrt();
        let layer_norm: f64 = layer_state.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            layer_norm > 0.0,
            "the fiber must move toward the observation"
        );
        assert!(
            layer_norm < proj_norm,
            "the fiber blends — it does not jump to the observation"
        );
        // And the manifold clocks advanced — the prediction step ran.
        assert!(runtime.field().fibers[1].engine_clock > 0.0);
    }

    #[test]
    fn observation_integrates_real_state_projections() {
        let state = GenesisCoreState::new(1, 1_000);
        let mut runtime = FieldRuntime::new().unwrap();

        let result = runtime.observe(&state, 0.2, 0.0).unwrap();

        assert_eq!(result.fiber_count, 3);
        assert_eq!(result.consensus.len(), LATENT_DIMENSION);
        assert!(result.consensus.iter().all(|v| v.is_finite()));
        assert!(result.disagreement.is_finite());
        assert!(result.disagreement >= 0.0);
        assert!(result.temperature.is_finite());
        assert!(result.temperature >= 0.0);
    }

    #[test]
    fn the_measured_bath_heats_the_field() {
        // A quiet field (fresh, near-nominal projections) plus a hot
        // physical reservoir: the integration must report the bath's
        // temperature, not zero — the Langevin reservoir is the
        // silicon's own measured noise.
        let state = GenesisCoreState::new(1, 1_000);
        let mut runtime = FieldRuntime::new().unwrap();

        let cold = runtime.observe(&state, 0.2, 0.0).unwrap();
        let mut runtime = FieldRuntime::new().unwrap();
        let warm = runtime.observe(&state, 0.2, 0.4).unwrap();

        assert!(warm.temperature > cold.temperature);
        assert!(warm.temperature >= 0.4 - 1.0e-9);
    }

    #[test]
    fn observation_tracks_changed_prediction_state() {
        let mut state = GenesisCoreState::new(1, 1_000);
        let mut runtime = FieldRuntime::new().unwrap();

        let before = runtime.observe(&state, 0.2, 0.0).unwrap();
        state.inference_signals.surprise_ema = 0.9;
        state.inference_signals.free_energy = 0.8;
        let after = runtime.observe(&state, 0.2, 0.0).unwrap();

        assert_ne!(before.consensus, after.consensus);
    }

    #[test]
    fn fibers_spiral_inward_toward_nominal_between_observations() {
        // The engines carry a real law: an unobserved fiber winds
        // toward the sphere's pole at its own clearance rate — the
        // loxodrome. The drift must shrink the state's norm and
        // turn its phase, never grow it, never leave the cap.
        let mut runtime = FieldRuntime::new().unwrap();
        runtime.field.fibers[0].layers[0].state = vec![0.2; LATENT_DIMENSION];
        let before: f64 = runtime.field.fibers[0].layers[0]
            .state
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        let before_first_two: f64 =
            runtime.field.fibers[0].layers[0].state[0] / runtime.field.fibers[0].layers[0].state[1];
        runtime.field.fibers[0].step(1.0).unwrap();
        let state = &runtime.field.fibers[0].layers[0].state;
        let after: f64 = state.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(after < before, "damping must contract the state");
        // 1 s at τ=120 s ≈ 0.83% contraction — real but gentle.
        assert!(after > before * 0.97, "the drift is a leak, not a collapse");
        assert!(after < before * (1.0 - 0.5 / BODY_RELAX_TAU_S));
        // The (0,1)-plane winding turned the phase between the first
        // two coordinates — the spiral, not just the sink.
        let after_first_two = state[0] / state[1];
        assert_ne!(after_first_two, before_first_two, "the skew term must wind");
    }

    #[test]
    fn rem_lets_the_field_diverge_where_wake_pulls_it_together() {
        // Drive both runtimes with projections that genuinely
        // disagree — the prediction fiber hot on its channels,
        // the others near nominal — so each observation deposits
        // real spread for the integration to act on.
        let mut state = GenesisCoreState::new(1, 1_000);
        state.inference_signals.precision = 1.0;
        state.inference_signals.surprise_ema = 0.8;
        state.inference_signals.free_energy = 0.6;

        let mut wake = FieldRuntime::new().unwrap();
        for _ in 0..10 {
            wake.observe(&state, 0.2, 0.0).unwrap();
        }

        state.zones.emergent_phase = MentalPhase::REM as u8;
        let mut rem = FieldRuntime::new().unwrap();
        for _ in 0..10 {
            rem.observe(&state, 0.2, 0.0).unwrap();
        }

        // The same observations, integrated by descent vs ascent
        // on the same potential: REM's LC-silent field must hold
        // strictly more disagreement than the wake-coupled one.
        assert!(
            rem.field().disagreement > wake.field().disagreement,
            "REM decorrelation should out-spread wake coherence: rem={} wake={}",
            rem.field().disagreement,
            wake.field().disagreement
        );
    }

    #[test]
    fn latent_maps_unit_interval_into_signed_half_width() {
        assert_eq!(latent(0.0), -LATENT_HALF);
        assert_eq!(latent(0.5), 0.0); // neutral state is the sphere's pole
        assert_eq!(latent(1.0), LATENT_HALF);
    }

    #[test]
    fn projections_stay_inside_the_cap_unclamped() {
        // Every projection component passes through a [0,1]
        // normalizer and then `latent`, so the reachable set is the
        // box [−0.25, 0.25]^6. Its corner norm is √6·0.25 ≈ 0.61 —
        // interior to the hemisphere cap, so project() must leave
        // even corner projections untouched rather than clamping
        // them onto the equator rim.
        let manifold = StereographicSphere::new(LATENT_DIMENSION, 1.0).unwrap();
        for sign in [1.0, -1.0] {
            let corner = vec![sign * LATENT_HALF; LATENT_DIMENSION];
            assert_eq!(manifold.project(&corner).unwrap(), corner);
        }
        let state = GenesisCoreState::new(1, 1_000);
        for projection in [
            body_projection(&state),
            prediction_projection(&state),
            agency_projection(&state),
        ] {
            assert!(projection.iter().all(|v| v.abs() <= LATENT_HALF));
            assert_eq!(manifold.project(&projection).unwrap(), projection);
        }
    }
}
