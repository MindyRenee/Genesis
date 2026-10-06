//! Machine-native field architecture for Genesis.
//!
//! A major system is a SystemFiber; its subsystems are the layers of that
//! system's latent field. Geometry is separated from semantics: a manifold
//! is a container and metric, while layers own represented state and dynamics.
//!
//! The implementation is dependency-free and deterministic. It does not
//! assume that a fixed number of fibers is correct.

use std::fmt;

pub mod engine;
pub use engine::SystemEngine;

const EPS: f64 = 1.0e-12;
pub type Vector = Vec<f64>;
pub type Matrix = Vec<Vector>;

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
fn norm(a: &[f64]) -> f64 { a.iter().fold(0.0_f64, |acc, &x| acc.hypot(x)) }
fn scaled(a: &[f64], s: f64) -> Vector { a.iter().map(|x| x * s).collect() }
fn add(a: &[f64], b: &[f64]) -> Vector { a.iter().zip(b).map(|(x, y)| x + y).collect() }
fn sub(a: &[f64], b: &[f64]) -> Vector { a.iter().zip(b).map(|(x, y)| x - y).collect() }

#[derive(Clone, Debug, PartialEq)]
pub struct PoincareBall { dimension: usize, curvature: f64 }

impl PoincareBall {
    pub fn new(dimension: usize, c: f64) -> Result<Self, FieldError> {
        if dimension == 0 || !c.is_finite() || c <= 0.0 { return Err(FieldError::InvalidGeometry); }
        Ok(Self { dimension, curvature: c })
    }
    pub fn dimension(&self) -> usize { self.dimension }
    pub fn curvature(&self) -> f64 { self.curvature }
    fn scaled_radius(&self) -> f64 { 1.0 / self.curvature.sqrt() }

    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dimension && x.iter().all(|v| v.is_finite()) && norm(x) < self.scaled_radius()
    }

    pub fn project(&self, x: &[f64]) -> Result<Vector, FieldError> {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::DimensionMismatch);
        }
        let r = self.scaled_radius();
        let n = norm(x);
        if n < r { return Ok(x.to_vec()); }
        let target = r * (1.0 - 1.0e-9);
        Ok(scaled(x, target / n))
    }

    pub fn distance(&self, x: &[f64], y: &[f64]) -> Result<f64, FieldError> {
        self.validate_point(x)?; self.validate_point(y)?;
        let sqrt_c = self.curvature.sqrt();
        // Scale into the unit ball before forming squared norms or
        // differences. This avoids overflow when curvature is very small
        // and coordinates are correspondingly large.
        let ux = scaled(x, sqrt_c);
        let uy = scaled(y, sqrt_c);
        let nx = dot(&ux, &ux);
        let ny = dot(&uy, &uy);
        let dxy = norm(&sub(&ux, &uy));
        let denom = (1.0 - nx).max(EPS) * (1.0 - ny).max(EPS);
        let arg = 1.0 + 2.0 * dxy * dxy / denom;
        Ok(arg.max(1.0).acosh() / sqrt_c)
    }

    pub fn log_origin(&self, x: &[f64]) -> Result<Vector, FieldError> {
        self.validate_point(x)?;
        let r = norm(x);
        if r < EPS { return Ok(vec![0.0; self.dimension]); }
        let scaled_r = self.curvature.sqrt() * r;
        // Floating-point roundoff can turn a mathematically valid radius
        // slightly below one into exactly one. Keep atanh's argument inside
        // its finite domain rather than manufacturing an infinite tangent.
        let atanh_arg = scaled_r.min(1.0 - f64::EPSILON);
        let z = atanh_arg.atanh() * 2.0 / scaled_r;
        Ok(scaled(x, z))
    }

    pub fn exp_origin(&self, v: &[f64]) -> Result<Vector, FieldError> {
        if v.len() != self.dimension || v.iter().any(|x| !x.is_finite()) {
            return Err(FieldError::DimensionMismatch);
        }
        let r = norm(v);
        if r < EPS { return Ok(v.to_vec()); }
        let sqrt_c = self.curvature.sqrt();
        let scaled_r = sqrt_c * r;
        // For large tangent norms, forming sqrt(c) * r can overflow even
        // though the limiting exponential-map scale is simply 1/(sqrt(c)*r).
        // Switch to that asymptotic form before the product can overflow.
        let z = if scaled_r.is_finite() {
            (scaled_r / 2.0).tanh() / scaled_r
        } else {
            1.0 / (sqrt_c * r)
        };
        self.project(&scaled(v, z))
    }

    fn validate_point(&self, x: &[f64]) -> Result<(), FieldError> {
        if self.contains(x) { Ok(()) } else { Err(FieldError::InvalidPoint) }
    }

    /// Compute a consensus by averaging logarithmic coordinates at the origin.
    ///
    /// This is a coordinate-defined tangent-space barycenter, not the intrinsic
    /// Fréchet/Karcher mean of the hyperbolic manifold. Keeping that distinction
    /// explicit prevents a convenient computational operator from being treated
    /// as a geometry-independent mean.
    pub fn origin_barycenter(&self, points: &[Vector]) -> Result<Vector, FieldError> {
        if points.is_empty() { return Err(FieldError::EmptyState); }
        let mut tangent = vec![0.0; self.dimension];
        for p in points {
            let v = self.log_origin(p)?;
            for (dst, src) in tangent.iter_mut().zip(v) { *dst += src / points.len() as f64; }
        }
        self.exp_origin(&tangent)
    }
}

/// One subsystem is one latent-space layer of its parent system.
#[derive(Clone, Debug, PartialEq)]
pub struct SubsystemLayer {
    pub id: String,
    pub state: Vector,
    pub gain: f64,
}

impl SubsystemLayer {
    pub fn new(id: impl Into<String>, state: Vector) -> Self {
        Self { id: id.into(), state, gain: 1.0 }
    }
    pub fn evolve(&mut self, derivative: &[f64], dt: f64) -> Result<(), FieldError> {
        if derivative.len() != self.state.len() || !dt.is_finite() || dt < 0.0 {
            return Err(FieldError::DimensionMismatch);
        }
        for (x, dx) in self.state.iter_mut().zip(derivative) { *x += dt * self.gain * dx; }
        Ok(())
    }
}

/// A major system: one engine operating a multi-layer latent field.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemFiber {
    pub id: String,
    pub engine_clock: f64,
    pub manifold: PoincareBall,
    pub layers: Vec<SubsystemLayer>,
    pub engine: SystemEngine,
}

impl SystemFiber {
    pub fn new(id: impl Into<String>, manifold: PoincareBall, layers: Vec<SubsystemLayer>) -> Result<Self, FieldError> {
        if layers.iter().any(|l| l.state.len() != manifold.dimension()) {
            return Err(FieldError::DimensionMismatch);
        }
        if layers.iter().any(|l| !l.gain.is_finite()) {
            return Err(FieldError::InvalidDynamics);
        }
        let engine = SystemEngine::zero(layers.len(), manifold.dimension());
        Ok(Self { id: id.into(), engine_clock: 0.0, manifold, layers, engine })
    }
    pub fn with_engine(id: impl Into<String>, manifold: PoincareBall, layers: Vec<SubsystemLayer>, engine: SystemEngine) -> Result<Self, FieldError> {
        if engine.biases.len() != layers.len() { return Err(FieldError::LayerMismatch); }
        if engine.dimension() != manifold.dimension() { return Err(FieldError::DimensionMismatch); }
        let mut fiber = Self::new(id, manifold, layers)?;
        fiber.engine = engine;
        Ok(fiber)
    }

    pub fn layer(&self, id: &str) -> Option<&SubsystemLayer> { self.layers.iter().find(|l| l.id == id) }
    pub fn layer_mut(&mut self, id: &str) -> Option<&mut SubsystemLayer> { self.layers.iter_mut().find(|l| l.id == id) }

    /// Advance the fiber by an external elapsed interval.
    ///
    /// `dt` is the shared/global elapsed time. `time_scale` converts that
    /// interval into this engine's internal dynamical time. The engine clock
    /// records the converted interval.
    pub fn step(&mut self, dt: f64) -> Result<(), FieldError> {
        if !dt.is_finite() || dt < 0.0 { return Err(FieldError::InvalidTime); }
        let effective_dt = dt * self.engine.time_scale;
        if !effective_dt.is_finite() || effective_dt < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        if effective_dt == 0.0 {
            return Ok(());
        }

        // Integrate the engine's continuous-time field with classical RK4.
        // The engine remains the source of dynamics; the integrator only
        // advances that field in time. Layer gains are applied at each stage.
        if self.layers.iter().any(|layer| !layer.gain.is_finite()) {
            return Err(FieldError::InvalidDynamics);
        }
        let initial = self.layer_states();
        let scaled = |derivatives: Vec<Vector>| -> Vec<Vector> {
            derivatives.into_iter().enumerate().map(|(i, d)| {
                d.into_iter().map(|x| x * self.layers[i].gain).collect()
            }).collect()
        };
        let add_stage = |state: &[Vector], slope: &[Vector], scale: f64| -> Vec<Vector> {
            state.iter().zip(slope).map(|(x, dx)| {
                x.iter().zip(dx).map(|(v, dv)| v + scale * dv).collect()
            }).collect()
        };

        let k1 = scaled(self.engine.derivatives(&self.layers)?);
        let state2 = add_stage(&initial, &k1, effective_dt * 0.5);
        let mut stage_layers = self.layers.clone();
        for (layer, state) in stage_layers.iter_mut().zip(state2) { layer.state = state; }
        let k2 = scaled(self.engine.derivatives(&stage_layers)?);
        let state3 = add_stage(&initial, &k2, effective_dt * 0.5);
        for (layer, state) in stage_layers.iter_mut().zip(state3) { layer.state = state; }
        let k3 = scaled(self.engine.derivatives(&stage_layers)?);
        let state4 = add_stage(&initial, &k3, effective_dt);
        for (layer, state) in stage_layers.iter_mut().zip(state4) { layer.state = state; }
        let k4 = scaled(self.engine.derivatives(&stage_layers)?);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            for j in 0..layer.state.len() {
                layer.state[j] = initial[i][j]
                    + effective_dt * (k1[i][j] + 2.0 * k2[i][j] + 2.0 * k3[i][j] + k4[i][j]) / 6.0;
            }
            layer.state = self.manifold.project(&layer.state)?;
        }
        let next_clock = self.engine_clock + effective_dt;
        if !next_clock.is_finite() {
            return Err(FieldError::InvalidTime);
        }
        self.engine_clock = next_clock;
        Ok(())
    }

    pub fn evolve(&mut self, derivatives: &[Vector], dt: f64) -> Result<(), FieldError> {
        if !dt.is_finite() || dt < 0.0 { return Err(FieldError::InvalidTime); }
        if derivatives.len() != self.layers.len() { return Err(FieldError::LayerMismatch); }
        for (layer, derivative) in self.layers.iter_mut().zip(derivatives) {
            layer.evolve(derivative, dt)?;
            layer.state = self.manifold.project(&layer.state)?;
        }
        self.engine_clock += dt;
        Ok(())
    }

    pub fn state(&self) -> Result<Vector, FieldError> {
        if self.layers.is_empty() { return Err(FieldError::EmptyState); }
        self.manifold.origin_barycenter(&self.layers.iter().map(|layer| layer.state.clone()).collect::<Vec<_>>())
    }
    pub fn layer_states(&self) -> Vec<Vector> { self.layers.iter().map(|layer| layer.state.clone()).collect() }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaugeTransport {
    pub from: String,
    pub to: String,
    pub matrix: Matrix,
}

impl GaugeTransport {
    pub fn new(from: impl Into<String>, to: impl Into<String>, matrix: Matrix) -> Result<Self, FieldError> {
        let n = matrix.len();
        if n == 0 || matrix.iter().any(|r| r.len() != n || r.iter().any(|v| !v.is_finite())) {
            return Err(FieldError::InvalidTransport);
        }
        Ok(Self { from: from.into(), to: to.into(), matrix })
    }
    pub fn apply_on_manifold(&self, source: &PoincareBall, target: &PoincareBall, x: &[f64]) -> Result<Vector, FieldError> {
        if source.dimension() != target.dimension() || source.curvature() != target.curvature() || self.matrix.len() != source.dimension() {
            return Err(FieldError::GeometryMismatch);
        }
        let tangent = source.log_origin(x)?;
        let mapped = self.apply_tangent(&tangent)?;
        target.exp_origin(&mapped)
    }
    pub fn apply_tangent(&self, x: &[f64]) -> Result<Vector, FieldError> {
        if x.len() != self.matrix.len() { return Err(FieldError::DimensionMismatch); }
        let mut out = vec![0.0; x.len()];
        for (i, row) in self.matrix.iter().enumerate() {
            if row.len() != x.len() { return Err(FieldError::DimensionMismatch); }
            out[i] = dot(row, x);
        }
        if out.iter().any(|v| !v.is_finite()) { return Err(FieldError::NonFiniteState); }
        Ok(out)
    }
    pub fn compose(&self, other: &GaugeTransport) -> Result<GaugeTransport, FieldError> {
        if self.from != other.to || self.matrix.len() != other.matrix.len() { return Err(FieldError::InvalidTransport); }
        let n = self.matrix.len(); let mut m = vec![vec![0.0; n]; n];
        for i in 0..n { for j in 0..n { for k in 0..n { m[i][j] += self.matrix[i][k] * other.matrix[k][j]; } } }
        if m.iter().flatten().any(|v| !v.is_finite()) { return Err(FieldError::NonFiniteState); }
        GaugeTransport::new(other.from.clone(), self.to.clone(), m)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransportPath { pub nodes: Vec<String>, pub matrix: Matrix }
#[derive(Clone, Debug, PartialEq)]
pub struct PathDisagreement { pub from: String, pub to: String, pub paths: Vec<TransportPath>, pub tangent_spread: f64 }

#[derive(Clone, Debug, PartialEq)]
pub struct GlobalField {
    pub manifold: PoincareBall,
    pub fibers: Vec<SystemFiber>,
    pub transports: Vec<GaugeTransport>,
    pub integration_gain: f64,
    pub disagreement: f64,
    pub temperature: f64,
    pub cooling_rate: f64,
}

impl GlobalField {
    pub fn new(manifold: PoincareBall) -> Self { Self { manifold, fibers: Vec::new(), transports: Vec::new(), integration_gain: 1.0, disagreement: 0.0, temperature: 1.0, cooling_rate: 0.1 } }
    pub fn add_fiber(&mut self, fiber: SystemFiber) -> Result<(), FieldError> {
        if fiber.manifold.dimension() != self.manifold.dimension() || fiber.manifold.curvature() != self.manifold.curvature() { return Err(FieldError::GeometryMismatch); }
        if self.fibers.iter().any(|f| f.id == fiber.id) { return Err(FieldError::DuplicateId); }
        self.fibers.push(fiber); Ok(())
    }
    pub fn add_transport(&mut self, transport: GaugeTransport) -> Result<(), FieldError> {
        if transport.from == transport.to || !self.fibers.iter().any(|f| f.id == transport.from) || !self.fibers.iter().any(|f| f.id == transport.to) { return Err(FieldError::InvalidTransport); }
        let n = self.manifold.dimension();
        if transport.matrix.len() != n || transport.matrix.iter().any(|r| r.len() != n || r.iter().any(|v| !v.is_finite())) { return Err(FieldError::InvalidTransport); }
        self.transports.push(transport); Ok(())
    }
    fn matrix_distance(a: &Matrix, b: &Matrix) -> f64 {
        a.iter().zip(b).flat_map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| (x-y).powi(2))).sum::<f64>().sqrt()
    }
    fn mat_mul(a: &Matrix, b: &Matrix) -> Matrix {
        let n = a.len(); let mut out = vec![vec![0.0; n]; n];
        for i in 0..n { for j in 0..n { for k in 0..n { out[i][j] += a[i][k] * b[k][j]; } } }
        out
    }
    fn transport_paths(&self, from: &str, to: &str, max_paths: usize) -> Vec<TransportPath> {
        fn dfs(field: &GlobalField, current: &str, target: &str, nodes: &mut Vec<String>, matrix: &Matrix, out: &mut Vec<TransportPath>, max_paths: usize) {
            if out.len() >= max_paths { return; }
            if current == target { out.push(TransportPath { nodes: nodes.clone(), matrix: matrix.clone() }); return; }
            for edge in field.transports.iter().filter(|t| t.from == current) {
                if nodes.contains(&edge.to) { continue; }
                let composed = GlobalField::mat_mul(&edge.matrix, matrix);
                nodes.push(edge.to.clone()); dfs(field, &edge.to, target, nodes, &composed, out, max_paths); nodes.pop();
                if out.len() >= max_paths { return; }
            }
        }
        let n = self.manifold.dimension(); let identity = (0..n).map(|i| (0..n).map(|j| if i==j {1.0} else {0.0}).collect()).collect();
        let mut nodes = vec![from.to_string()]; let mut out = Vec::new(); dfs(self, from, to, &mut nodes, &identity, &mut out, max_paths); out
    }
    fn path_matrix(&self, from: &str, to: &str) -> Result<Matrix, FieldError> {
        let paths = self.transport_paths(from, to, 2);
        if paths.is_empty() { return Err(FieldError::MissingTransport); }
        if paths.len() == 1 { return Ok(paths[0].matrix.clone()); }
        let spread = Self::matrix_distance(&paths[0].matrix, &paths[1].matrix);
        if spread > 1.0e-9 { return Err(FieldError::TransportMismatch); }
        Ok(paths[0].matrix.clone())
    }
    fn transported_states(&self) -> Result<Vec<Vector>, FieldError> {
        if self.fibers.is_empty() { return Ok(Vec::new()); }
        let reference = &self.fibers[0]; let mut states = vec![reference.state()?];
        for fiber in self.fibers.iter().skip(1) {
            let state = fiber.state()?; let matrix = self.path_matrix(&fiber.id, &reference.id)?;
            let transport = GaugeTransport::new(fiber.id.clone(), reference.id.clone(), matrix)?;
            states.push(transport.apply_on_manifold(&fiber.manifold, &reference.manifold, &state)?);
        }
        Ok(states)
    }
    pub fn consensus(&self) -> Result<Vector, FieldError> { let states = self.transported_states()?; if states.is_empty() { return Err(FieldError::EmptyState); } self.manifold.origin_barycenter(&states) }
    fn matrix_inverse(matrix: &Matrix) -> Result<Matrix, FieldError> {
        let n = matrix.len(); let mut a = matrix.iter().map(|r| { let mut row = r.clone(); row.extend((0..n).map(|j| if j==0 {0.0} else {0.0})); row }).collect::<Vec<_>>();
        for i in 0..n { a[i][n+i] = 1.0; }
        for col in 0..n { let pivot = (col..n).max_by(|&i,&j| a[i][col].abs().partial_cmp(&a[j][col].abs()).unwrap()).unwrap(); if a[pivot][col].abs() < EPS { return Err(FieldError::NonInvertibleTransport); } a.swap(col,pivot); let p=a[col][col]; for j in 0..2*n { a[col][j]/=p; } for i in 0..n { if i==col {continue;} let f=a[i][col]; for j in 0..2*n { a[i][j]-=f*a[col][j]; } } }
        Ok(a.into_iter().map(|r| r[n..].to_vec()).collect())
    }
    pub fn synchronize(&mut self, dt: f64) -> Result<(), FieldError> {
        if !dt.is_finite() || dt < 0.0 { return Err(FieldError::InvalidTime); }
        if self.fibers.is_empty() { return Ok(()); }
        let consensus = self.consensus()?; let states = self.transported_states()?;
        let mut disagreement = 0.0; for state in &states { let d=self.manifold.distance(state,&consensus)?; disagreement += d*d; } self.disagreement=disagreement/states.len() as f64;
        let gain=(self.integration_gain.max(0.0)*dt).clamp(0.0,1.0); let reference_id=self.fibers[0].id.clone(); let consensus_tangent=self.manifold.log_origin(&consensus)?;
        for i in 1..self.fibers.len() { let id=self.fibers[i].id.clone(); let path=self.path_matrix(&id,&reference_id)?; let inverse=Self::matrix_inverse(&path)?; let local_target_tangent=GaugeTransport::new(reference_id.clone(),id.clone(),inverse)?.apply_tangent(&consensus_tangent)?; let local_target=self.fibers[i].manifold.exp_origin(&local_target_tangent)?; let current=self.fibers[i].state()?; let current_tangent=self.fibers[i].manifold.log_origin(&current)?; let target_correction=self.fibers[i].manifold.log_origin(&local_target)?; let correction=target_correction.iter().zip(current_tangent).map(|(t,c)|t-c).collect::<Vec<_>>(); let correction=scaled(&correction,gain); for layer in self.fibers[i].layers.iter_mut() { let tangent=self.fibers[i].manifold.log_origin(&layer.state)?; let updated=tangent.iter().zip(&correction).map(|(x,d)|x+d).collect::<Vec<_>>(); layer.state=self.fibers[i].manifold.exp_origin(&updated)?; } }
        self.temperature=(self.temperature-self.cooling_rate*dt).max(0.0); Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum FieldError {
    InvalidGeometry, InvalidPoint, DimensionMismatch, EmptyState, LayerMismatch,
    InvalidTransport, TransportMismatch, GeometryMismatch, InvalidTime, MissingTransport,
    InvalidDynamics, NonFiniteState, DuplicateId, NonInvertibleTransport,
}

impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{self:?}") }
}
impl std::error::Error for FieldError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hyperbolic_distance_is_zero_at_same_point() {
        let m = PoincareBall::new(3, 1.0).unwrap();
        let p = vec![0.1, 0.2, -0.1];
        assert!(m.distance(&p, &p).unwrap().abs() < 1e-10);
    }

    #[test]
    fn origin_barycenter_is_order_independent() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let points = vec![vec![0.1, 0.0], vec![0.0, 0.2], vec![0.2, 0.1]];
        let mut reversed = points.clone();
        reversed.reverse();
        let a = m.origin_barycenter(&points).unwrap();
        let b = m.origin_barycenter(&reversed).unwrap();
        for (x, y) in a.iter().zip(b) { assert!((x - y).abs() < 1e-12); }
    }

    #[test]
    fn exp_log_round_trip() {
        let m = PoincareBall::new(3, 1.0).unwrap();
        let v = vec![0.1, -0.2, 0.05];
        let p = m.exp_origin(&v).unwrap();
        let recovered = m.log_origin(&p).unwrap();
        for (a, b) in v.iter().zip(recovered) { assert!((a - b).abs() < 1e-10); }
    }

    #[test]
    fn geometry_norm_handles_large_finite_coordinates() {
        let m = PoincareBall::new(2, 1.0e-300).unwrap();
        let p = vec![1.0e149, 1.0e149];
        assert!(m.contains(&p));
        let v = m.log_origin(&p).unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn distance_handles_large_coordinates_without_overflow() {
        let m = PoincareBall::new(2, 1.0e-300).unwrap();
        let x = vec![1.0e149, 0.0];
        let y = vec![-1.0e149, 0.0];
        let d = m.distance(&x, &y).unwrap();
        assert!(d.is_finite());
        assert!(d > 0.0);
    }

    #[test]
    fn empty_fiber_has_no_aggregate_system_state() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let fiber = SystemFiber::new("empty", m, Vec::new()).unwrap();
        assert_eq!(fiber.state(), Err(FieldError::EmptyState));
        assert!(fiber.layer_states().is_empty());
    }

    #[test]
    fn fiber_rejects_nonfinite_layer_gain() {
        let m = PoincareBall::new(1, 1.0).unwrap();
        let layer = SubsystemLayer { id: "a".into(), state: vec![0.0], gain: f64::NAN };
        assert_eq!(SystemFiber::new("x", m, vec![layer]), Err(FieldError::InvalidDynamics));
    }
