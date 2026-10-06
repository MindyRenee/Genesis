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

fn dot(a: &[f64], b: &[f64]) -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
fn norm(a: &[f64]) -> f64 { dot(a, a).sqrt() }
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
        let c = self.curvature;
        let nx = c * dot(x, x); let ny = c * dot(y, y);
        let dxy = norm(&sub(x, y));
        let denom = (1.0 - nx).max(EPS) * (1.0 - ny).max(EPS);
        let arg = 1.0 + 2.0 * c * dxy * dxy / denom;
        Ok(arg.max(1.0).acosh() / c.sqrt())
    }

    pub fn log_origin(&self, x: &[f64]) -> Result<Vector, FieldError> {
        self.validate_point(x)?;
        let r = norm(x);
        if r < EPS { return Ok(vec![0.0; self.dimension]); }
        let z = (self.curvature.sqrt() * r).atanh() * 2.0
            / (self.curvature.sqrt() * r);
        Ok(scaled(x, z))
    }

    pub fn exp_origin(&self, v: &[f64]) -> Result<Vector, FieldError> {
        if v.len() != self.dimension || v.iter().any(|x| !x.is_finite()) {
            return Err(FieldError::DimensionMismatch);
        }
        let r = norm(v);
        if r < EPS { return Ok(v.to_vec()); }
        let z = (self.curvature.sqrt() * r / 2.0).tanh()
            / (self.curvature.sqrt() * r);
        self.project(&scaled(v, z))
    }

    fn validate_point(&self, x: &[f64]) -> Result<(), FieldError> {
        if self.contains(x) { Ok(()) } else { Err(FieldError::InvalidPoint) }
    }

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
        let engine = SystemEngine::zero(layers.len(), manifold.dimension());
        Ok(Self { id: id.into(), engine_clock: 0.0, manifold, layers, engine })
    }
    pub fn with_engine(id: impl Into<String>, manifold: PoincareBall, layers: Vec<SubsystemLayer>, engine: SystemEngine) -> Result<Self, FieldError> {
        if engine.biases.len() != layers.len() { return Err(FieldError::LayerMismatch); }
        let mut fiber = Self::new(id, manifold, layers)?;
        fiber.engine = engine;
        Ok(fiber)
    }

    pub fn layer(&self, id: &str) -> Option<&SubsystemLayer> { self.layers.iter().find(|l| l.id == id) }
    pub fn layer_mut(&mut self, id: &str) -> Option<&mut SubsystemLayer> { self.layers.iter_mut().find(|l| l.id == id) }

    pub fn step(&mut self, dt: f64) -> Result<(), FieldError> {
        if !dt.is_finite() || dt < 0.0 { return Err(FieldError::InvalidTime); }
        let derivatives = self.engine.derivatives(&self.layers)?;
        self.evolve(&derivatives, dt * self.engine.time_scale)
    }

    pub fn evolve(&mut self, derivatives: &[Vector], dt: f64) -> Result<(), FieldError> {
        if derivatives.len() != self.layers.len() { return Err(FieldError::LayerMismatch); }
        for (layer, derivative) in self.layers.iter_mut().zip(derivatives) {
            layer.evolve(derivative, dt)?;
            layer.state = self.manifold.project(&layer.state)?;
        }
        self.engine_clock += dt;
        Ok(())
    }

    pub fn state(&self) -> Result<Vector, FieldError> {
        self.manifold.origin_barycenter(&self.layers.iter().map(|l| l.state.clone()).collect::<Vec<_>>())
    }
}

/// An order-sensitive coordinate transformation between system manifolds.
#[derive(Clone, Debug, PartialEq)]
pub struct GaugeTransport {
    pub from: String,
    pub to: String,
    pub matrix: Vec<Vector>,
}

impl GaugeTransport {
    pub fn new(from: impl Into<String>, to: impl Into<String>, matrix: Vec<Vector>) -> Result<Self, FieldError> {
        let n = matrix.len();
        if n == 0 || matrix.iter().any(|row| row.len() != n || row.iter().any(|x| !x.is_finite())) {
            return Err(FieldError::InvalidTransport);
        }
        Ok(Self { from: from.into(), to: to.into(), matrix })
    }
    /// Apply the transport as a tangent-space coordinate map.
    ///
    /// Points are first mapped to the source tangent space at the common
    /// origin, transformed there, then returned to the target manifold.
    /// This prevents a Euclidean matrix from being mistaken for a direct
    /// map of curved-manifold coordinates.
    pub fn apply_on_manifold(
        &self,
        source: &PoincareBall,
        target: &PoincareBall,
        x: &[f64],
    ) -> Result<Vector, FieldError> {
        if source.dimension() != target.dimension()
            || x.len() != source.dimension()
            || (source.curvature() - target.curvature()).abs() > EPS
            || self.matrix.len() != source.dimension()
        {
            return Err(FieldError::GeometryMismatch);
        }
        let tangent = source.log_origin(x)?;
        let mapped = self.apply_tangent(&tangent)?;
        target.exp_origin(&mapped)
    }

    pub fn apply_tangent(&self, x: &[f64]) -> Result<Vector, FieldError> {
        if x.len() != self.matrix.len() {
            return Err(FieldError::DimensionMismatch);
        }
        let out: Vector = self.matrix.iter().map(|row| dot(row, x)).collect();
        if out.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        Ok(out)
    }
    pub fn compose(&self, other: &Self) -> Result<Self, FieldError> {
        if other.matrix.len() != self.matrix.len() || self.from != other.to {
            return Err(FieldError::TransportMismatch);
        }
        let n = self.matrix.len(); let mut m = vec![vec![0.0; n]; n];
        for i in 0..n { for j in 0..n { for k in 0..n { m[i][j] += self.matrix[i][k] * other.matrix[k][j]; } } }
        Self::new(other.from.clone(), self.to.clone(), m)
    }
}

/// Ordered temporal trajectory. This is deliberately distinct from TDA barcodes.
#[derive(Clone, Debug, PartialEq)]
pub struct BraidTrajectory { pub fiber_id: String, pub samples: Vec<Vector> }

impl BraidTrajectory {
    pub fn new(fiber_id: impl Into<String>) -> Self { Self { fiber_id: fiber_id.into(), samples: Vec::new() } }
    pub fn push(&mut self, state: Vector) { self.samples.push(state); }
    pub fn len(&self) -> usize { self.samples.len() }
    pub fn is_empty(&self) -> bool { self.samples.is_empty() }
}

/// Global execution field. Fiber count is intentionally dynamic.
#[derive(Clone, Debug)]
pub struct GlobalField {
    pub manifold: PoincareBall,
    pub fibers: Vec<SystemFiber>,
    pub transports: Vec<GaugeTransport>,
    /// Global integration gain. Unlike temperature, this is a coupling
    /// strength and does not erase disagreement by itself.
    pub integration_gain: f64,
    /// Running disagreement energy in the common coordinate frame.
    pub disagreement: f64,
    pub temperature: f64,
    pub cooling_rate: f64,
}

impl GlobalField {
    pub fn new(manifold: PoincareBall) -> Self {
        Self { manifold, fibers: Vec::new(), transports: Vec::new(), integration_gain: 1.0, disagreement: 0.0, temperature: 1.0, cooling_rate: 0.1 }
    }
    pub fn add_fiber(&mut self, fiber: SystemFiber) -> Result<(), FieldError> {
        if fiber.manifold.dimension() != self.manifold.dimension()
            || (fiber.manifold.curvature() - self.manifold.curvature()).abs() > EPS {
            return Err(FieldError::GeometryMismatch);
        }
        self.fibers.push(fiber); Ok(())
    }
    pub fn add_transport(&mut self, transport: GaugeTransport) -> Result<(), FieldError> {
        if transport.from == transport.to
            || self.fibers.iter().all(|f| f.id != transport.from)
            || self.fibers.iter().all(|f| f.id != transport.to)
            || transport.matrix.len() != self.manifold.dimension()
        {
            return Err(FieldError::TransportMismatch);
        }
        self.transports.push(transport);
        Ok(())
    }

    fn path_matrix(&self, from: &str, to: &str) -> Result<Vec<Vector>, FieldError> {
        let n = self.manifold.dimension();
        let mut queue: Vec<(String, Vec<Vector>)> = vec![(from.to_string(), identity(n))];
        let mut seen = vec![from.to_string()];
        while let Some((node, matrix)) = queue.first().cloned() {
            queue.remove(0);
            if node == to { return Ok(matrix); }
            for edge in self.transports.iter().filter(|e| e.from == node) {
                if seen.iter().any(|id| id == &edge.to) { continue; }
                seen.push(edge.to.clone());
                queue.push((edge.to.clone(), mat_mul(&edge.matrix, &matrix)));
            }
        }
        Err(FieldError::MissingTransport)
    }

    pub fn transported_states(&self) -> Result<Vec<Vector>, FieldError> {
        let mut out = Vec::with_capacity(self.fibers.len());
        for fiber in &self.fibers {
            let state = fiber.state()?;
            let mapped = if fiber.id == self.fibers[0].id {
                state
            } else {
                let matrix = self.path_matrix(&fiber.id, &self.fibers[0].id)?;
                let tangent = self.manifold.log_origin(&state)?;
                let mapped = matrix_apply(&matrix, &tangent)?;
                self.manifold.exp_origin(&mapped)?
            };
            out.push(mapped);
        }
        Ok(out)
    }

    pub fn consensus(&self) -> Result<Option<Vector>, FieldError> {
        let states = self.transported_states()?;
        if states.is_empty() { return Ok(None); }
        Ok(Some(self.manifold.origin_barycenter(&states)?))
    }

    pub fn synchronize(&mut self, dt: f64) -> Result<Option<Vector>, FieldError> {
        if !dt.is_finite() || dt < 0.0 { return Err(FieldError::InvalidTime); }
        let states = self.transported_states()?;
        if states.is_empty() { return Ok(None); }
        let consensus = self.manifold.origin_barycenter(&states)?;
        let mut disagreement = 0.0;
        for state in &states {
            let delta = sub(state, &consensus);
            disagreement += dot(&delta, &delta);
        }
        self.disagreement = disagreement / states.len() as f64;

        let gain = self.integration_gain.max(0.0);
        let target = (gain * dt).clamp(0.0, 1.0);
        for fiber in &mut self.fibers {
            for layer in &mut fiber.layers {
                let delta = sub(&layer.state, &consensus);
                layer.state = self.manifold.project(&add(&layer.state, &scaled(&delta, -target)))?;
            }
        }
        self.temperature = (self.temperature - self.cooling_rate * dt).max(0.0);
        Ok(Some(consensus))
    }
}



fn identity(n: usize) -> Vec<Vector> {
    (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

fn mat_mul(a: &[Vector], b: &[Vector]) -> Vec<Vector> {
    let n = a.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n { for j in 0..n { for k in 0..n { out[i][j] += a[i][k] * b[k][j]; } } }
    out
}

fn matrix_apply(m: &[Vector], x: &[f64]) -> Result<Vector, FieldError> {
    if m.len() != x.len() || m.iter().any(|r| r.len() != x.len()) {
        return Err(FieldError::DimensionMismatch);
    }
    Ok(m.iter().map(|r| dot(r, x)).collect())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldError {
    InvalidGeometry, InvalidPoint, DimensionMismatch, EmptyState, LayerMismatch,
    InvalidTransport, TransportMismatch, GeometryMismatch, InvalidTime, MissingTransport,
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
    fn exp_log_round_trip() {
        let m = PoincareBall::new(3, 1.0).unwrap();
        let v = vec![0.1, -0.2, 0.05];
        let p = m.exp_origin(&v).unwrap();
        let recovered = m.log_origin(&p).unwrap();
        for (a, b) in v.iter().zip(recovered) { assert!((a - b).abs() < 1e-10); }
    }

    #[test]
    fn subsystem_is_the_layer() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let fiber = SystemFiber::new("affect", m, vec![
            SubsystemLayer::new("emotion", vec![0.1, 0.0]),
            SubsystemLayer::new("appraisal", vec![0.0, 0.1]),
        ]).unwrap();
        assert_eq!(fiber.layers.len(), 2);
        assert_eq!(fiber.layer("emotion").unwrap().id, "emotion");
    }

    #[test]
    fn engine_can_drive_a_fiber_without_external_derivatives() {
        let m = PoincareBall::new(1, 1.0).unwrap();
        let layers = vec![SubsystemLayer::new("a", vec![0.1]), SubsystemLayer::new("b", vec![0.2])];
        let engine = SystemEngine::new(
            vec![vec![vec![vec![0.5]], vec![vec![1.0]]], vec![vec![vec![-1.0]], vec![vec![0.0]]]],
            vec![vec![0.0], vec![vec![0.0]]], 1.0
        ).unwrap();
        let mut fiber = SystemFiber::with_engine("dynamic", m, layers, engine).unwrap();
        fiber.step(0.1).unwrap();
        assert!(fiber.layer("a").unwrap().state[0] > 0.1);
        assert!(fiber.layer("b").unwrap().state[0] < 0.2);
    }

    #[test]
    fn transport_composition_is_order_sensitive() {
        let a = GaugeTransport::new("a", "b", vec![vec![1.0, 1.0], vec![0.0, 1.0]]).unwrap();
        let b = GaugeTransport::new("b", "c", vec![vec![1.0, 0.0], vec![1.0, 1.0]]).unwrap();
        let ab = b.compose(&a).unwrap();
        let ba = a.compose(&b).unwrap();
        assert_ne!(ab.matrix, ba.matrix);
    }

    #[test]
    fn dynamic_fiber_count() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let mut global = GlobalField::new(m.clone());
        for i in 0..3 {
            global.add_fiber(SystemFiber::new(format!("fiber-{i}"), m.clone(),
                vec![SubsystemLayer::new("state", vec![0.01 * i as f64, 0.0])]).unwrap()).unwrap();
        }
        assert_eq!(global.fibers.len(), 3);
    }

    #[test]
    fn synchronization_records_disagreement_and_integrates_gradually() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let mut global = GlobalField::new(m.clone());
        global.integration_gain = 0.5;
        global.add_fiber(SystemFiber::new("a", m.clone(), vec![SubsystemLayer::new("s", vec![0.2, 0.0])]).unwrap()).unwrap();
        global.add_fiber(SystemFiber::new("b", m, vec![SubsystemLayer::new("s", vec![0.4, 0.0])]).unwrap()).unwrap();
        let before = global.disagreement;
        global.synchronize(1.0).unwrap();
        assert!(global.disagreement > before);
        assert!((global.fibers[0].layers[0].state[0] - 0.3).abs() < 0.01);
        assert!((global.fibers[1].layers[0].state[0] - 0.3).abs() < 0.01);
    }

    #[test]
    fn cooling_reduces_dispersion_without_forcing_center() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let mut global = GlobalField::new(m.clone());
        global.add_fiber(SystemFiber::new("a", m.clone(), vec![SubsystemLayer::new("s", vec![0.2, 0.0])]).unwrap()).unwrap();
        global.add_fiber(SystemFiber::new("b", m, vec![SubsystemLayer::new("s", vec![0.4, 0.0])]).unwrap()).unwrap();
        let before = global.consensus().unwrap().unwrap();
        global.cooling_rate = 1.0;
        global.synchronize(1.0).unwrap();
        let after = global.consensus().unwrap().unwrap();
        assert!(after[0] > 0.0);
        assert!((before[0] - after[0]).abs() < 0.05);
        assert_eq!(global.temperature, 0.0);
    }
}
