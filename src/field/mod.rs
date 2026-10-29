//! Machine-native field architecture for Genesis.
//!
//! A major system is a SystemFiber; its subsystems are the layers of that
//! system's latent field. Geometry is separated from semantics: a manifold
//! is a container and metric, while layers own represented state and dynamics.
//!
//! The latent manifold is the κ-stereographic sphere — the constant
//! *positive* curvature counterpart of the Poincaré ball (Bachmann et al.
//! 2020): the sphere of curvature κ projected stereographically onto the
//! chart, with the north pole at the origin. The field operates inside
//! the hemisphere cap ‖x‖ < 1/√κ, a geodesically convex region whose
//! center is the pole — the organism's nominal home.
//!
//! The implementation is dependency-free and deterministic. It does not
//! assume that a fixed number of fibers is correct.

use std::fmt;

pub mod engine;
pub use engine::SystemEngine;
pub mod runtime;
pub use runtime::{FieldIntegration, FieldRuntime, LATENT_DIMENSION};
pub mod motor;
pub use motor::MotorTarget;

const EPS: f64 = 1.0e-12;
/// Upper bound on the field's thermal state — disagreement heats
/// the field only this far, so the Langevin kick stays a
/// fluctuation around the consensus pull, never a takeover.
pub(crate) const TEMPERATURE_CAP: f64 = 0.5;

/// The latent working box's half-width — the thermal kick samples
/// temperature as a *fraction* of the field's operating extent.
/// Unscaled, σ ≈ √(2·gain·T·dt) reaches ~0.45 tangent units on a
/// hot field while the whole box spans ±0.5 — a kick that size
/// scatters the fibers across the manifold instead of perturbing
/// them, and the scattered fibers re-create the disagreement that
/// keeps the temperature pinned: a thermostat stuck on. Scaling σ
/// by the box half-width makes the kick a genuine fluctuation and
/// lets a cooling field actually settle.
pub(crate) const KICK_LATENT_SCALE: f64 = 0.25;

/// Provenance marker — one letter of the creator lattice (see
/// PROVENANCE.md). This one is load-bearing: it is bound into the
/// deterministic noise stream's seed below.
pub(crate) const MARKER_G00: u64 = 0x4753_004D_4975_7005;
/// Golden-ratio stream seed for the thermal kick, bound to the
/// provenance lattice so the noise sequence itself carries a letter.
pub(crate) const FIELD_NOISE_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Fingerprint of the creator lattice (PROVENANCE.md): the XOR fold of
/// all eight Rust provenance markers. The constants are referenced
/// through this function, so removing one is a compile error, not a
/// silent edit; it is logged at daemon startup and asserted by the
/// provenance test. The Python-side markers are scanned by
/// `scripts/provenance.py`.
pub fn provenance_fingerprint() -> u64 {
    MARKER_G00
        ^ runtime::MARKER_G01
        ^ engine::MARKER_G02
        ^ motor::MARKER_G03
        ^ crate::daemon::ipc::MARKER_G04
        ^ crate::daemon::tick::MARKER_G05
        ^ crate::state::header::MARKER_G06
        ^ crate::store::mmap_state::MARKER_G07
}
pub type Vector = Vec<f64>;
pub type Matrix = Vec<Vector>;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let scale_a = a.iter().fold(0.0_f64, |scale, &x| scale.max(x.abs()));
    let scale_b = b.iter().fold(0.0_f64, |scale, &x| scale.max(x.abs()));
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
    (normalized * scale_a) * scale_b
}
fn norm(a: &[f64]) -> f64 {
    a.iter().fold(0.0_f64, |acc, &x| acc.hypot(x))
}
fn scaled(a: &[f64], s: f64) -> Vector {
    a.iter().map(|x| x * s).collect()
}
fn add(a: &[f64], b: &[f64]) -> Vector {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
fn sub(a: &[f64], b: &[f64]) -> Vector {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct StereographicSphere {
    dimension: usize,
    curvature: f64,
}

impl StereographicSphere {
    /// The κ-stereographic sphere: the sphere of constant *positive*
    /// curvature κ, charted by stereographic projection. The origin
    /// of the chart is the image of the north pole — the
    /// distinguished center of the model. The point at infinity is
    /// the south pole, the one point the chart cannot represent.
    ///
    /// `contains`/`project` enforce the field's operating region —
    /// the hemisphere cap around the pole, ‖x‖ < 1/√κ (the
    /// stereographic image of the equator). The cap is geodesically
    /// convex: minimizing geodesics between cap points stay in the
    /// cap, and any set of cap points has a unique Fréchet mean
    /// inside it (Buss & Fillmore 2001). The geometric primitives
    /// (`mobius_add`, `log_*`, `exp_*`, `geodesic`) accept any valid
    /// chart point — finite and right dimension — and return honest
    /// sphere points, which may legitimately lie outside the cap;
    /// the state boundaries re-project.
    pub fn new(dimension: usize, c: f64) -> Result<Self, FieldError> {
        if dimension == 0 || !c.is_finite() || c <= 0.0 {
            return Err(FieldError::InvalidGeometry);
        }
        Ok(Self {
            dimension,
            curvature: c,
        })
    }
    pub fn dimension(&self) -> usize {
        self.dimension
    }
    pub fn curvature(&self) -> f64 {
        self.curvature
    }
    /// The chart radius of the equator — the rim of the operating
    /// cap. Points at exactly this radius sit on the equator;
    /// beyond it lies the southern hemisphere, out to the
    /// unrepresentable south pole at infinity.
    fn scaled_radius(&self) -> f64 {
        1.0 / self.curvature.sqrt()
    }
    /// Whether x is inside the operating cap — the open hemisphere
    /// around the pole where the field works.
    pub fn contains(&self, x: &[f64]) -> bool {
        x.len() == self.dimension
            && x.iter().all(|v| v.is_finite())
            && norm(x) < self.scaled_radius()
    }
    /// Retract x into the cap along its ray. This is the containment
    /// policy, not a geometric projection of the manifold: outside-
    /// cap coordinates are valid sphere points, but the field keeps
    /// its states inside the hemisphere.
    pub fn project(&self, x: &[f64]) -> Result<Vector, FieldError> {
        if x.len() != self.dimension {
            return Err(FieldError::DimensionMismatch);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let r = self.scaled_radius();
        let n = norm(x);
        if !n.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        if n < r {
            return Ok(x.to_vec());
        }
        let target = r * (1.0 - 1.0e-9);
        if !target.is_finite() || target <= 0.0 {
            return Err(FieldError::InvalidGeometry);
        }
        let scale = target / n;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(FieldError::NonFiniteState);
        }
        let projected = scaled(x, scale);
        if projected.iter().any(|v| !v.is_finite()) || !self.contains(&projected) {
            return Err(FieldError::NonFiniteState);
        }
        Ok(projected)
    }
    /// Geodesic distance on the sphere. In stereographic
    /// coordinates, with û = √κ·x the unit-sphere chart, the lifted
    /// chord gives
    ///
    ///   tan(d·√κ/2) = √κ‖x − y‖
    ///                 / √(1 + 2κ⟨x,y⟩ + κ²‖x‖²‖y‖²)
    ///
    /// so d = (2/√κ)·atan2(√κ‖x−y‖, √(1+2κ⟨x,y⟩+κ²‖x‖²‖y‖²)). The
    /// denominator factors as (1+κ⟨x,y⟩)² + κ²(‖x‖²‖y‖² − ⟨x,y⟩²) —
    /// nonnegative by Cauchy–Schwarz, vanishing only for antipodal
    /// pairs, where atan2 correctly returns the maximal distance
    /// π/√κ instead of dividing by zero.
    pub fn distance(&self, x: &[f64], y: &[f64]) -> Result<f64, FieldError> {
        self.validate_point(x)?;
        self.validate_point(y)?;
        let sqrt_c = self.curvature.sqrt();
        let c = self.curvature;
        let nx = dot(x, x);
        let ny = dot(y, y);
        let xy = dot(x, y);
        let dxy = norm(&sub(x, y));
        if !(nx.is_finite() && ny.is_finite() && xy.is_finite() && dxy.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let opposite = (1.0 + 2.0 * c * xy + c * c * nx * ny).max(0.0).sqrt();
        let distance = 2.0 * (sqrt_c * dxy).atan2(opposite) / sqrt_c;
        if !distance.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        Ok(distance)
    }
    /// The tangent at the pole (the chart's origin) pointing along
    /// the geodesic to x, distance-normalized: ‖v‖ = d(0, x). A
    /// point at chart radius r sits at geodesic distance
    /// (2/√κ)·atan(√κ·r) from the pole — unlike the ball's atanh
    /// this needs no clamp: atan is finite for every representable
    /// point, tending to π/(2√κ) as r → ∞ (the south pole).
    pub fn log_origin(&self, x: &[f64]) -> Result<Vector, FieldError> {
        self.validate_point(x)?;
        let r = norm(x);
        if !r.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        if r < EPS {
            return Ok(vec![0.0; self.dimension]);
        }
        let sqrt_c = self.curvature.sqrt();
        let scaled_r = sqrt_c * r;
        if !scaled_r.is_finite() || scaled_r <= 0.0 {
            return Err(FieldError::NonFiniteState);
        }
        let z = scaled_r.atan() * 2.0 / scaled_r;
        if !z.is_finite() || z <= 0.0 {
            return Err(FieldError::NonFiniteState);
        }
        let tangent = scaled(x, z);
        if tangent.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        Ok(tangent)
    }
    /// Step from the pole along direction v for ‖v‖ units of
    /// geodesic distance: the image is the direction of v at chart
    /// radius (1/√κ)·tan(√κ·‖v‖/2). tan is honest about the sphere
    /// being compact: ‖v‖ = π/√κ lands at the unrepresentable south
    /// pole (rejected as non-finite), and beyond it the map wraps
    /// around — the sphere closes on itself.
    pub fn exp_origin(&self, v: &[f64]) -> Result<Vector, FieldError> {
        if v.len() != self.dimension {
            return Err(FieldError::DimensionMismatch);
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let r = norm(v);
        if !r.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        if r < EPS {
            return Ok(v.to_vec());
        }
        let sqrt_c = self.curvature.sqrt();
        let scaled_r = sqrt_c * r;
        if !scaled_r.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        let z = (scaled_r / 2.0).tan() / scaled_r;
        if !z.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        let out = scaled(v, z);
        if out.iter().any(|x| !x.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        Ok(out)
    }
    /// Möbius addition — the gyrovector operation that plays the
    /// role of vector addition on the stereographic sphere (Ungar;
    /// κ > 0, equator image at radius 1/√κ):
    ///
    ///   x ⊕_κ y = [(1 − 2κ⟨x,y⟩ − κ‖y‖²)x + (1 + κ‖x‖²)y]
    ///             / (1 − 2κ⟨x,y⟩ + κ²‖x‖²‖y‖²)
    ///
    /// Non-commutative and non-associative, but left-cancellative:
    /// (−x) ⊕_κ (x ⊕_κ y) = y — the identity that makes the general
    /// log/exp pair below true inverses. Adding a fixed element is a
    /// spherical translation — a rotation of the sphere, so it moves
    /// every point the same geodesic distance. The denominator can
    /// only vanish when y is the antipode of −x (the sum would be
    /// the unrepresentable south pole); inside the cap it is bounded
    /// away from zero by (1 − κ‖x‖‖y‖)².
    pub fn mobius_add(&self, x: &[f64], y: &[f64]) -> Result<Vector, FieldError> {
        self.validate_point(x)?;
        self.validate_point(y)?;
        let c = self.curvature;
        let nx = dot(x, x);
        let ny = dot(y, y);
        let xy = dot(x, y);
        if !(nx.is_finite() && ny.is_finite() && xy.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let x_num = 1.0 - 2.0 * c * xy - c * ny;
        let y_num = 1.0 + c * nx;
        let denom = 1.0 - 2.0 * c * xy + c * c * nx * ny;
        if !(x_num.is_finite() && y_num.is_finite() && denom.is_finite() && denom > EPS) {
            return Err(FieldError::NonFiniteState);
        }
        let mut out = vec![0.0; self.dimension];
        for ((dst, &xi), &yi) in out.iter_mut().zip(x).zip(y) {
            *dst = (x_num * xi + y_num * yi) / denom;
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        Ok(out)
    }
    /// Möbius subtraction: (−x) ⊕_κ y — the displacement from x to
    /// y expressed as a chart point. A spherical translation of y
    /// back toward the origin: d(0, −x ⊕ y) = d(x, y).
    fn mobius_sub(&self, x: &[f64], y: &[f64]) -> Result<Vector, FieldError> {
        self.mobius_add(&scaled(x, -1.0), y)
    }
    /// Logarithm map at an arbitrary basepoint: the tangent vector
    /// at x that points along the geodesic toward y, with the same
    /// distance-normalized convention as `log_origin` (‖v‖ = the
    /// geodesic distance the vector travels). Because x ↦ (−x) ⊕ x
    /// is a spherical translation to the origin, this is just
    /// `log_origin` of the Möbius displacement.
    pub fn log_at(&self, x: &[f64], y: &[f64]) -> Result<Vector, FieldError> {
        self.log_origin(&self.mobius_sub(x, y)?)
    }
    /// Exponential map at an arbitrary basepoint: step from x along
    /// the geodesic in direction v for ‖v‖ units of geodesic
    /// distance — `x ⊕_κ exp_origin(v)`. Inverse of `log_at` by
    /// left-cancellation.
    pub fn exp_at(&self, x: &[f64], v: &[f64]) -> Result<Vector, FieldError> {
        self.validate_point(x)?;
        self.mobius_add(x, &self.exp_origin(v)?)
    }
    /// The geodesic from x toward y at fraction t of the spherical
    /// distance: d(x, γ(t)) = t·d(x,y) exactly. γ(0) = x, γ(1) = y;
    /// t outside \[0,1\] continues along the same great circle — the
    /// geodesic wraps around the sphere.
    pub fn geodesic(&self, x: &[f64], y: &[f64], t: f64) -> Result<Vector, FieldError> {
        if !t.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        self.exp_at(x, &scaled(&self.log_at(x, y)?, t))
    }
    /// Geodesic point reflection through `through`: the point as far
    /// past `through` as x is before it — the symmetric-space
    /// involution s_c(x). `through` is the geodesic midpoint of x
    /// and its image, and the map is its own inverse:
    /// s_c(s_c(x)) = x.
    pub fn point_reflection(&self, x: &[f64], through: &[f64]) -> Result<Vector, FieldError> {
        self.geodesic(x, through, 2.0)
    }
    /// The true spherical antipode — the point maximally distant
    /// from x at d = π/√κ. Under stereographic projection the
    /// antipode map is inversion in the equator circle composed with
    /// negation: −x/(κ‖x‖²). It lands outside the cap (in the
    /// southern hemisphere), the genuine "far side." The pole's own
    /// antipode is the unrepresentable south pole at infinity — an
    /// error, not a clamp.
    pub fn antipode(&self, x: &[f64]) -> Result<Vector, FieldError> {
        self.validate_point(x)?;
        let r2 = dot(x, x);
        if !r2.is_finite() || r2 <= EPS {
            return Err(FieldError::NonFiniteState);
        }
        Ok(scaled(x, -1.0 / (self.curvature * r2)))
    }
    /// A valid chart point: right dimension, finite. Any such point
    /// names a sphere point in the stereographic chart — cap
    /// membership (`contains`) is a separate, stricter question.
    fn validate_point(&self, x: &[f64]) -> Result<(), FieldError> {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            Err(FieldError::InvalidPoint)
        } else {
            Ok(())
        }
    }
    /// The tangent-space mean at the pole — the Euclidean average
    /// of the lifted log coordinates, mapped back. Exact for a
    /// single point and the right seed for `frechet_mean`; away
    /// from the pole it is only the barycenter's first
    /// approximation, since the log map distorts with distance.
    fn tangent_barycenter(&self, points: &[Vector]) -> Result<Vector, FieldError> {
        let tangents = points
            .iter()
            .map(|p| self.log_origin(p))
            .collect::<Result<Vec<_>, _>>()?;
        let scale = tangents
            .iter()
            .flatten()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        if !scale.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        if scale == 0.0 {
            return self.exp_origin(&vec![0.0; self.dimension]);
        }
        let count = points.len() as f64;
        let mut mean = vec![0.0; self.dimension];
        for tangent in &tangents {
            for (dst, &src) in mean.iter_mut().zip(tangent) {
                *dst += (src / scale) / count;
            }
        }
        if mean.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let mean = scaled(&mean, scale);
        if mean.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        self.exp_origin(&mean)
    }
    /// The Fréchet mean: the point minimizing Σᵢ d²(m, xᵢ) — the
    /// true center of mass on the manifold, where the tangent-space
    /// mean is only its polar approximation. Computed by Karcher
    /// iteration: `m ← exp_m(meanᵢ log_m(xᵢ))`, steepest descent on
    /// ½Σd² (whose Riemannian gradient is −Σ log_m(xᵢ)). Positive
    /// curvature changes the guarantee, not the method: on the
    /// sphere the mean is unique whenever the points lie in an open
    /// hemisphere (Buss & Fillmore 2001) — exactly the cap the field
    /// operates in — and ½d² stays strictly convex while every
    /// pairwise distance stays under π/(2√κ). Seeded at the tangent
    /// barycenter, which is already the exact answer at the pole.
    pub fn frechet_mean(&self, points: &[Vector]) -> Result<Vector, FieldError> {
        if points.is_empty() {
            return Err(FieldError::EmptyState);
        }
        if points.iter().any(|p| p.len() != self.dimension) {
            return Err(FieldError::DimensionMismatch);
        }
        let mut m = self.tangent_barycenter(points)?;
        let count = points.len() as f64;
        for _ in 0..64 {
            let mut step = vec![0.0; self.dimension];
            for p in points {
                let v = self.log_at(&m, p)?;
                for (s, &vi) in step.iter_mut().zip(&v) {
                    *s += vi;
                }
            }
            let step = scaled(&step, 1.0 / count);
            let step_norm = norm(&step);
            if !step_norm.is_finite() {
                return Err(FieldError::NonFiniteState);
            }
            // The step IS the negative gradient — its vanishing is
            // the Fréchet condition itself.
            if step_norm < 1.0e-9 {
                return Ok(m);
            }
            m = self.exp_at(&m, &step)?;
        }
        Ok(m)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubsystemLayer {
    pub id: String,
    pub state: Vector,
    pub gain: f64,
}
impl SubsystemLayer {
    pub fn new(id: impl Into<String>, state: Vector) -> Self {
        Self {
            id: id.into(),
            state,
            gain: 1.0,
        }
    }
    /// Euler update by `dt·gain·derivative` on this layer alone.
    /// The layer is geometry-agnostic — the caller owns the
    /// manifold: `SystemFiber::evolve` re-projects the result into
    /// the cap. Crate-private so no other path can strand a layer
    /// outside the operating region.
    pub(crate) fn evolve(&mut self, derivative: &[f64], dt: f64) -> Result<(), FieldError> {
        if derivative.len() != self.state.len() {
            return Err(FieldError::DimensionMismatch);
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        if !self.gain.is_finite() {
            return Err(FieldError::InvalidDynamics);
        }
        if derivative.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let mut next = self.state.clone();
        for (x, dx) in next.iter_mut().zip(derivative) {
            *x += dt * self.gain * dx;
        }
        if next.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        self.state = next;
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SystemFiber {
    pub id: String,
    pub engine_clock: f64,
    pub manifold: StereographicSphere,
    pub layers: Vec<SubsystemLayer>,
    pub engine: SystemEngine,
}
impl SystemFiber {
    pub fn new(
        id: impl Into<String>,
        manifold: StereographicSphere,
        layers: Vec<SubsystemLayer>,
    ) -> Result<Self, FieldError> {
        if layers.iter().any(|l| l.state.len() != manifold.dimension()) {
            return Err(FieldError::DimensionMismatch);
        }
        if layers
            .iter()
            .any(|l| l.state.iter().any(|v| !v.is_finite()))
        {
            return Err(FieldError::NonFiniteState);
        }
        if layers.iter().any(|l| !l.gain.is_finite()) {
            return Err(FieldError::InvalidDynamics);
        }
        let mut ids = std::collections::HashSet::with_capacity(layers.len());
        if layers.iter().any(|l| !ids.insert(&l.id)) {
            return Err(FieldError::DuplicateId);
        }
        Ok(Self {
            id: id.into(),
            engine_clock: 0.0,
            engine: SystemEngine::zero(layers.len(), manifold.dimension()),
            manifold,
            layers,
        })
    }
    pub fn with_engine(
        id: impl Into<String>,
        manifold: StereographicSphere,
        layers: Vec<SubsystemLayer>,
        engine: SystemEngine,
    ) -> Result<Self, FieldError> {
        if engine.biases.len() != layers.len() {
            return Err(FieldError::LayerMismatch);
        }
        if engine.dimension() != manifold.dimension() {
            return Err(FieldError::DimensionMismatch);
        }
        engine.validate()?;
        let mut fiber = Self::new(id, manifold, layers)?;
        fiber.engine = engine;
        Ok(fiber)
    }
    pub fn layer(&self, id: &str) -> Option<&SubsystemLayer> {
        self.layers.iter().find(|l| l.id == id)
    }
    pub fn layer_mut(&mut self, id: &str) -> Option<&mut SubsystemLayer> {
        self.layers.iter_mut().find(|l| l.id == id)
    }
    pub fn step(&mut self, dt: f64) -> Result<(), FieldError> {
        if !self.engine_clock.is_finite() || self.engine_clock < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        self.engine.validate()?;
        if self
            .layers
            .iter()
            .any(|l| l.state.iter().any(|v| !v.is_finite()))
        {
            return Err(FieldError::NonFiniteState);
        }
        let effective_dt = dt * self.engine.time_scale;
        if !effective_dt.is_finite() || effective_dt < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        if effective_dt == 0.0 {
            return Ok(());
        }
        if self.layers.iter().any(|l| !l.gain.is_finite()) {
            return Err(FieldError::InvalidDynamics);
        }
        let initial = self.layer_states();
        let scaled_derivatives = |ds: Vec<Vector>| -> Vec<Vector> {
            ds.into_iter()
                .enumerate()
                .map(|(i, d)| d.into_iter().map(|x| x * self.layers[i].gain).collect())
                .collect()
        };
        let add_stage = |state: &[Vector], slope: &[Vector], scale: f64| -> Vec<Vector> {
            state
                .iter()
                .zip(slope)
                .map(|(x, dx)| x.iter().zip(dx).map(|(v, dv)| v + scale * dv).collect())
                .collect()
        };
        let k1 = scaled_derivatives(self.engine.derivatives(&self.layers)?);
        let state2 = add_stage(&initial, &k1, effective_dt * 0.5);
        let mut stage_layers = self.layers.clone();
        for (layer, state) in stage_layers.iter_mut().zip(state2) {
            layer.state = state
        }
        let k2 = scaled_derivatives(self.engine.derivatives(&stage_layers)?);
        let state3 = add_stage(&initial, &k2, effective_dt * 0.5);
        for (layer, state) in stage_layers.iter_mut().zip(state3) {
            layer.state = state
        }
        let k3 = scaled_derivatives(self.engine.derivatives(&stage_layers)?);
        let state4 = add_stage(&initial, &k3, effective_dt);
        for (layer, state) in stage_layers.iter_mut().zip(state4) {
            layer.state = state
        }
        let k4 = scaled_derivatives(self.engine.derivatives(&stage_layers)?);
        let mut next_layers = self.layers.clone();
        for (i, layer) in next_layers.iter_mut().enumerate() {
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
        self.layers = next_layers;
        self.engine_clock = next_clock;
        Ok(())
    }
    pub fn evolve(&mut self, derivatives: &[Vector], dt: f64) -> Result<(), FieldError> {
        if !self.engine_clock.is_finite() || self.engine_clock < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        if !dt.is_finite() || dt < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        if derivatives.len() != self.layers.len() {
            return Err(FieldError::LayerMismatch);
        }
        self.engine.validate()?;
        if self
            .layers
            .iter()
            .any(|l| l.state.iter().any(|v| !v.is_finite()))
        {
            return Err(FieldError::NonFiniteState);
        }
        let effective_dt = dt * self.engine.time_scale;
        if !effective_dt.is_finite() || effective_dt < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        if effective_dt == 0.0 {
            return Ok(());
        }
        let mut next_layers = self.layers.clone();
        for (layer, derivative) in next_layers.iter_mut().zip(derivatives) {
            layer.evolve(derivative, effective_dt)?;
            layer.state = self.manifold.project(&layer.state)?;
        }
        let next_clock = self.engine_clock + effective_dt;
        if !next_clock.is_finite() {
            return Err(FieldError::InvalidTime);
        }
        self.layers = next_layers;
        self.engine_clock = next_clock;
        Ok(())
    }
    pub fn state(&self) -> Result<Vector, FieldError> {
        if self.layers.is_empty() {
            return Err(FieldError::EmptyState);
        }
        self.manifold.frechet_mean(
            &self
                .layers
                .iter()
                .map(|l| l.state.clone())
                .collect::<Vec<_>>(),
        )
    }
    pub fn layer_states(&self) -> Vec<Vector> {
        self.layers.iter().map(|l| l.state.clone()).collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaugeTransport {
    pub from: String,
    pub to: String,
    pub matrix: Matrix,
}
impl GaugeTransport {
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        matrix: Matrix,
    ) -> Result<Self, FieldError> {
        let n = matrix.len();
        if n == 0
            || matrix
                .iter()
                .any(|r| r.len() != n || r.iter().any(|v| !v.is_finite()))
        {
            return Err(FieldError::InvalidTransport);
        }
        Ok(Self {
            from: from.into(),
            to: to.into(),
            matrix,
        })
    }
    pub fn apply_on_manifold(
        &self,
        source: &StereographicSphere,
        target: &StereographicSphere,
        x: &[f64],
    ) -> Result<Vector, FieldError> {
        if source.dimension() != target.dimension()
            || source.curvature() != target.curvature()
            || self.matrix.len() != source.dimension()
        {
            return Err(FieldError::GeometryMismatch);
        }
        let tangent = source.log_origin(x)?;
        target.exp_origin(&self.apply_tangent(&tangent)?)
    }
    pub fn apply_tangent(&self, x: &[f64]) -> Result<Vector, FieldError> {
        if x.len() != self.matrix.len() {
            return Err(FieldError::DimensionMismatch);
        }
        let mut out = vec![0.0; x.len()];
        for (i, row) in self.matrix.iter().enumerate() {
            if row.len() != x.len() {
                return Err(FieldError::DimensionMismatch);
            }
            out[i] = dot(row, x);
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        Ok(out)
    }
    pub fn compose(&self, other: &GaugeTransport) -> Result<GaugeTransport, FieldError> {
        if self.from != other.to || self.matrix.len() != other.matrix.len() {
            return Err(FieldError::InvalidTransport);
        }
        let n = self.matrix.len();
        let mut m = vec![vec![0.0; n]; n];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                for k in 0..n {
                    let product = self.matrix[i][k] * other.matrix[k][j];
                    if !product.is_finite() {
                        return Err(FieldError::NonFiniteState);
                    }
                    let sum = *cell + product;
                    if !sum.is_finite() {
                        return Err(FieldError::NonFiniteState);
                    }
                    *cell = sum;
                }
            }
        }
        GaugeTransport::new(other.from.clone(), self.to.clone(), m)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BraidTrajectory {
    pub fiber_id: String,
    pub samples: Vec<Vector>,
}
impl BraidTrajectory {
    pub fn new(fiber_id: impl Into<String>) -> Self {
        Self {
            fiber_id: fiber_id.into(),
            samples: Vec::new(),
        }
    }
    pub fn push(&mut self, state: Vector) {
        self.samples.push(state)
    }
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransportPath {
    pub nodes: Vec<String>,
    pub matrix: Matrix,
}
#[derive(Clone, Debug, PartialEq)]
pub struct PathDisagreement {
    pub from: String,
    pub to: String,
    pub paths: Vec<TransportPath>,
    pub tangent_spread: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlobalField {
    pub manifold: StereographicSphere,
    pub fibers: Vec<SystemFiber>,
    pub transports: Vec<GaugeTransport>,
    pub integration_gain: f64,
    pub disagreement: f64,
    /// The field's thermal state — the amplitude of the fluctuation
    /// the consensus pull permits. Starts at zero: the field is
    /// born ordered, and only unresolved disagreement heats it (see
    /// `synchronize` — the temperature floor snaps to the current
    /// disagreement and relaxes at `cooling_rate`).
    pub temperature: f64,
    pub cooling_rate: f64,
    /// Deterministic pseudo-noise state for the thermal kick —
    /// a fixed-seed xorshift64* stream, so the fluctuation model is
    /// stochastic in structure and reproducible in sequence.
    pub rng_state: u64,
}
impl GlobalField {
    pub fn new(manifold: StereographicSphere) -> Self {
        Self {
            manifold,
            fibers: Vec::new(),
            transports: Vec::new(),
            integration_gain: 1.0,
            disagreement: 0.0,
            temperature: 0.0,
            cooling_rate: 0.1,
            rng_state: FIELD_NOISE_SEED ^ MARKER_G00,
        }
    }
    pub fn add_fiber(&mut self, fiber: SystemFiber) -> Result<(), FieldError> {
        if fiber.manifold.dimension() != self.manifold.dimension()
            || (fiber.manifold.curvature() - self.manifold.curvature()).abs() > EPS
        {
            return Err(FieldError::GeometryMismatch);
        }
        if fiber
            .layers
            .iter()
            .any(|l| l.state.iter().any(|v| !v.is_finite()) || !l.gain.is_finite())
        {
            return Err(FieldError::InvalidDynamics);
        }
        fiber.engine.validate()?;
        if !fiber.engine_clock.is_finite() || fiber.engine_clock < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        if self.fibers.iter().any(|f| f.id == fiber.id) {
            return Err(FieldError::DuplicateId);
        }
        self.fibers.push(fiber);
        Ok(())
    }
    pub fn add_transport(&mut self, transport: GaugeTransport) -> Result<(), FieldError> {
        let n = self.manifold.dimension();
        if transport.from == transport.to
            || !self.fibers.iter().any(|f| f.id == transport.from)
            || !self.fibers.iter().any(|f| f.id == transport.to)
            || transport.matrix.len() != n
            || transport
                .matrix
                .iter()
                .any(|r| r.len() != n || r.iter().any(|v| !v.is_finite()))
        {
            return Err(FieldError::TransportMismatch);
        }
        self.transports.push(transport);
        Ok(())
    }
    /// Opposite of `add_fiber`: detach a subsystem carrier and hand
    /// it back. Strict — nothing is silently cascaded: a fiber still
    /// referenced by a transport cannot leave (remove the routes
    /// first), and the reference fiber (`fibers[0]`, the anchor
    /// `synchronize` corrects against) can only leave last, since
    /// dropping it earlier would silently re-anchor every remaining
    /// correction.
    pub fn remove_fiber(&mut self, id: &str) -> Result<SystemFiber, FieldError> {
        let index = self
            .fibers
            .iter()
            .position(|f| f.id == id)
            .ok_or(FieldError::MissingFiber)?;
        let referenced = self.transports.iter().any(|t| t.from == id || t.to == id);
        if referenced || (index == 0 && self.fibers.len() > 1) {
            return Err(FieldError::FiberInUse);
        }
        Ok(self.fibers.remove(index))
    }
    /// Opposite of `add_transport`: sever the directed edge
    /// `from → to` and hand it back.
    pub fn remove_transport(&mut self, from: &str, to: &str) -> Result<GaugeTransport, FieldError> {
        let index = self
            .transports
            .iter()
            .position(|t| t.from == from && t.to == to)
            .ok_or(FieldError::MissingTransport)?;
        Ok(self.transports.remove(index))
    }
    /// The diametric opposite of the field's current localization —
    /// the spherical antipode of the fresh consensus: the point on
    /// the far side of the sphere, at the maximal distance π/√κ,
    /// where the subsystems would sit if they collectively opposed
    /// what they now agree on. `None` when there is no consensus to
    /// oppose (empty field).
    pub fn anticonsensus(&self) -> Result<Option<Vector>, FieldError> {
        match self.consensus()? {
            Some(c) => Ok(Some(self.manifold.antipode(&c)?)),
            None => Ok(None),
        }
    }
    fn transport_paths(
        &self,
        from: &str,
        to: &str,
        max_paths: usize,
    ) -> Result<Vec<TransportPath>, FieldError> {
        /// Mutable state threaded through the depth-first search. Bundled so
        /// the recursion takes one accumulator instead of four separate
        /// out-parameters (which tripped `clippy::too_many_arguments`).
        struct Walk<'a> {
            to: &'a str,
            max_paths: usize,
            seen: Vec<String>,
            out: Vec<TransportPath>,
        }
        fn walk(
            field: &GlobalField,
            node: &str,
            matrix: Matrix,
            nodes: Vec<String>,
            acc: &mut Walk<'_>,
        ) -> Result<(), FieldError> {
            if acc.out.len() >= acc.max_paths {
                return Ok(());
            }
            if node == acc.to {
                acc.out.push(TransportPath { nodes, matrix });
                return Ok(());
            }
            for edge in field.transports.iter().filter(|e| e.from == node) {
                if acc.seen.iter().any(|id| id == &edge.to) {
                    continue;
                }
                acc.seen.push(edge.to.clone());
                let next = mat_mul(&edge.matrix, &matrix)?;
                let mut next_nodes = nodes.clone();
                next_nodes.push(edge.to.clone());
                walk(field, &edge.to, next, next_nodes, acc)?;
                acc.seen.pop();
            }
            Ok(())
        }
        let mut acc = Walk {
            to,
            max_paths: max_paths.max(1),
            seen: vec![from.to_string()],
            out: Vec::new(),
        };
        walk(
            self,
            from,
            identity(self.manifold.dimension()),
            vec![from.to_string()],
            &mut acc,
        )?;
        Ok(acc.out)
    }
    pub fn path_disagreement(
        &self,
        from: &str,
        to: &str,
    ) -> Result<Option<PathDisagreement>, FieldError> {
        if from == to {
            return Err(FieldError::TransportMismatch);
        }
        let paths = self.transport_paths(from, to, 32)?;
        if paths.len() < 2 {
            return Ok(None);
        }
        let mut spread: f64 = 0.0;
        for i in 0..paths.len() {
            for j in i + 1..paths.len() {
                spread = spread.max(matrix_distance(&paths[i].matrix, &paths[j].matrix)?);
            }
        }
        if !spread.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        Ok(Some(PathDisagreement {
            from: from.into(),
            to: to.into(),
            paths,
            tangent_spread: spread,
        }))
    }
    fn path_matrix(&self, from: &str, to: &str) -> Result<Matrix, FieldError> {
        let paths = self.transport_paths(from, to, 32)?;
        match paths.as_slice() {
            [] => Err(FieldError::MissingTransport),
            [p] => Ok(p.matrix.clone()),
            many => {
                // With a non-flat connection, alternative routes
                // legitimately disagree — that disagreement *is* the
                // holonomy. A canonical route still has to exist, so
                // transport prefers the fewest hops (least curvature
                // accumulated); disagreement between equally short
                // routes is a real ambiguity and still an error.
                let shortest = many.iter().map(|p| p.nodes.len()).min().expect("non-empty");
                let mut best = many.iter().filter(|p| p.nodes.len() == shortest);
                let candidate = &best.next().expect("non-empty").matrix;
                for path in best {
                    if matrix_distance(candidate, &path.matrix)? > 1.0e-9 {
                        return Err(FieldError::TransportMismatch);
                    }
                }
                Ok(candidate.clone())
            }
        }
    }
    pub fn transported_states(&self) -> Result<Vec<Vector>, FieldError> {
        if self.fibers.is_empty() {
            return Ok(Vec::new());
        }
        let reference = &self.fibers[0];
        let states = std::iter::once(Ok(reference.state()?))
            .chain(self.fibers.iter().skip(1).map(|fiber| {
                let state = fiber.state()?;
                let matrix = self.path_matrix(&fiber.id, &reference.id)?;
                let transport =
                    GaugeTransport::new(fiber.id.clone(), reference.id.clone(), matrix)?;
                transport.apply_on_manifold(&fiber.manifold, &reference.manifold, &state)
            }))
            .collect::<Result<Vec<_>, FieldError>>()?;
        if states
            .iter()
            .any(|state| state.iter().any(|value| !value.is_finite()))
        {
            return Err(FieldError::NonFiniteState);
        }
        Ok(states)
    }
    pub fn consensus(&self) -> Result<Option<Vector>, FieldError> {
        let states = self.transported_states()?;
        if states.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.manifold.frechet_mean(&states)?))
        }
    }
    pub fn step(&mut self, dt: f64, bath: f64) -> Result<Option<Vector>, FieldError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        let mut next = self.clone();
        for fiber in &mut next.fibers {
            fiber.step(dt)?;
        }
        let consensus = next.synchronize(dt, bath)?;
        *self = next;
        Ok(consensus)
    }
    /// Shared validation for both integration directions — a
    /// finite, nonnegative step and a sane thermostat.
    fn validate_integration(&self, dt: f64) -> Result<(), FieldError> {
        if !dt.is_finite() || dt < 0.0 {
            return Err(FieldError::InvalidTime);
        }
        if !self.integration_gain.is_finite()
            || !self.temperature.is_finite()
            || !self.cooling_rate.is_finite()
            || self.integration_gain < 0.0
            || self.temperature < 0.0
            || self.cooling_rate < 0.0
        {
            return Err(FieldError::InvalidDynamics);
        }
        Ok(())
    }
    /// The measured tension and the integration parameters both
    /// directions share: disagreement is the Fréchet variance of the
    /// transported states against the just-computed consensus —
    /// measured *before* the move, the tension the step resolves.
    /// `gain` is the bounded pull fraction; `sigma` the
    /// overdamped-Langevin kick amplitude √(2γT·dt) per tangent
    /// component — the fluctuation-dissipation relation with
    /// mobility γ = integration_gain, sampling the field's
    /// *pre-update* temperature so a cold field is exact. And the
    /// next temperature: disagreement heats the field, so the floor
    /// snaps up to the current tension (capped so the kick stays a
    /// fluctuation, not a takeover) and relaxes at `cooling_rate` —
    /// snap fast, drift slow, the adaptive-gate asymmetry.
    ///
    /// `bath` is the second heat source: the substrate's *own*
    /// measured thermal level, expressed in the same latent² units
    /// as disagreement. Where disagreement is the field's internal
    /// tension, the bath is the physical reservoir the Langevin
    /// equation actually couples to — real silicon noise driving
    /// real kicks (the effective-temperature construction of the
    /// fluctuation-dissipation literature: T_eff is measured
    /// fluctuation intensity, not a modeled amplitude). It floors
    /// the temperature under the same cap: a hot machine makes the
    /// field tremble whether or not the fibers disagree.
    fn thermal_plan(
        &self,
        states: &[Vector],
        consensus: &Vector,
        dt: f64,
        bath: f64,
    ) -> Result<(f64, f64, f64, f64), FieldError> {
        let bath = if bath.is_finite() {
            bath.clamp(0.0, TEMPERATURE_CAP)
        } else {
            0.0
        };
        let mut disagreement = 0.0;
        for state in states {
            let d = self.manifold.distance(state, consensus)?;
            disagreement += d * d;
        }
        let next_disagreement = disagreement / states.len() as f64;
        let gain = (self.integration_gain.max(0.0) * dt).clamp(0.0, 1.0);
        // σ = √(2·gain·T·dt) · box-half-width — the kick is a
        // fraction-of-extent fluctuation, not a raw-unit scatter.
        let sigma = (2.0 * self.integration_gain.max(0.0) * self.temperature * dt).sqrt()
            * KICK_LATENT_SCALE;
        if !sigma.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        let next_temperature = (self.temperature - self.cooling_rate * dt)
            .max(0.0)
            .max(next_disagreement.min(TEMPERATURE_CAP))
            .max(bath);
        Ok((next_disagreement, gain, sigma, next_temperature))
    }
    /// The correction machinery shared by `synchronize` and
    /// `diverge`: every non-reference fiber moves `signed_gain` of
    /// the way toward (positive) or away from (negative)
    /// `target_tangent` — expressed in the reference frame, mapped
    /// back through the transport inverse — plus a thermal kick of
    /// amplitude `sigma`. Then all staged states commit at once, so
    /// a failure mid-loop leaves the field untouched.
    fn pull_toward(
        &mut self,
        target_tangent: &Vector,
        signed_gain: f64,
        sigma: f64,
    ) -> Result<(), FieldError> {
        let mut rng = self.rng_state;
        let reference_id = self.fibers[0].id.clone();
        let mut next_states: Vec<(usize, Vec<Vector>)> =
            Vec::with_capacity(self.fibers.len().saturating_sub(1));
        for i in 1..self.fibers.len() {
            let matrix = self.path_matrix(&self.fibers[i].id, &reference_id)?;
            let inv = matrix_inverse(&matrix)?;
            let local_target_tangent = mat_vec(&inv, target_tangent)?;
            let current = self.fibers[i].state()?;
            let current_tangent = self.fibers[i].manifold.log_origin(&current)?;
            let correction = scaled(&sub(&local_target_tangent, &current_tangent), signed_gain);
            let manifold = self.fibers[i].manifold.clone();
            let mut states_for_fiber = Vec::with_capacity(self.fibers[i].layers.len());
            for layer in &self.fibers[i].layers {
                let tangent = manifold.log_origin(&layer.state)?;
                let mut updated = add(&tangent, &correction);
                if sigma > 0.0 {
                    for v in updated.iter_mut() {
                        *v += sigma * gaussian(&mut rng);
                    }
                }
                let next = manifold.project(&manifold.exp_origin(&updated)?)?;
                states_for_fiber.push(next);
            }
            next_states.push((i, states_for_fiber));
        }
        for (i, states_for_fiber) in next_states {
            for (layer, next) in self.fibers[i].layers.iter_mut().zip(states_for_fiber) {
                layer.state = next;
            }
        }
        self.rng_state = rng;
        Ok(())
    }
    pub fn synchronize(&mut self, dt: f64, bath: f64) -> Result<Option<Vector>, FieldError> {
        self.validate_integration(dt)?;
        let states = self.transported_states()?;
        if states.is_empty() {
            return Ok(None);
        }
        let consensus = self.manifold.frechet_mean(&states)?;
        let (next_disagreement, gain, sigma, next_temperature) =
            self.thermal_plan(&states, &consensus, dt, bath)?;
        let consensus_tangent = self.manifold.log_origin(&consensus)?;
        self.pull_toward(&consensus_tangent, gain, sigma)?;
        self.disagreement = next_disagreement;
        self.temperature = next_temperature;
        Ok(Some(consensus))
    }
    /// The opposite of `synchronize`: the same step with its sign
    /// negated — gradient *ascent* on the disagreement potential
    /// instead of descent. Each non-reference fiber moves away from
    /// the consensus along its own tangent rather than toward it;
    /// a fiber already at consensus feels no push, so divergence
    /// amplifies existing deviation rather than manufacturing it.
    /// Same thermal bookkeeping either way — divergence is a
    /// direction, not an exemption. Returns the consensus the field
    /// moved away from (the repeller, where `synchronize` returns
    /// the attractor).
    pub fn diverge(&mut self, dt: f64, bath: f64) -> Result<Option<Vector>, FieldError> {
        self.validate_integration(dt)?;
        let states = self.transported_states()?;
        if states.is_empty() {
            return Ok(None);
        }
        let consensus = self.manifold.frechet_mean(&states)?;
        let (next_disagreement, gain, sigma, next_temperature) =
            self.thermal_plan(&states, &consensus, dt, bath)?;
        let consensus_tangent = self.manifold.log_origin(&consensus)?;
        self.pull_toward(&consensus_tangent, -gain, sigma)?;
        self.disagreement = next_disagreement;
        self.temperature = next_temperature;
        Ok(Some(consensus))
    }
}

fn identity(n: usize) -> Matrix {
    let mut m = vec![vec![0.0; n]; n];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// xorshift64* → uniform on [0,1). Fixed-seed deterministic
/// pseudo-noise: the fluctuation model is stochastic in structure
/// and reproducible in sequence — no external entropy, no
/// dependency.
fn next_f64(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// Box–Muller N(0,1) from two uniforms — one branch is enough per
/// draw; the stream stays deterministic.
fn gaussian(state: &mut u64) -> f64 {
    let u1 = 1.0 - next_f64(state); // (0,1] — ln stays finite
    let u2 = next_f64(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn mat_mul(a: &Matrix, b: &Matrix) -> Result<Matrix, FieldError> {
    if a.is_empty()
        || b.is_empty()
        || a.len() != b.len()
        || a.iter().any(|row| row.len() != a.len())
        || b.iter().any(|row| row.len() != b.len())
    {
        return Err(FieldError::DimensionMismatch);
    }
    let n = a.len();
    let mut out = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            out[i][j] = dot(&a[i], &(0..n).map(|k| b[k][j]).collect::<Vec<_>>());
            if !out[i][j].is_finite() {
                return Err(FieldError::NonFiniteState);
            }
        }
    }
    Ok(out)
}
fn mat_vec(a: &Matrix, x: &Vector) -> Result<Vector, FieldError> {
    if a.len() != x.len() || a.iter().any(|r| r.len() != x.len()) {
        return Err(FieldError::DimensionMismatch);
    }
    let mut out = vec![0.0; x.len()];
    for i in 0..a.len() {
        out[i] = dot(&a[i], x);
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(FieldError::NonFiniteState);
    }
    Ok(out)
}
fn matrix_distance(a: &Matrix, b: &Matrix) -> Result<f64, FieldError> {
    if a.len() != b.len() || a.iter().zip(b).any(|(x, y)| x.len() != y.len()) {
        return Err(FieldError::DimensionMismatch);
    }
    let mut scale = 0.0_f64;
    for (row_a, row_b) in a.iter().zip(b) {
        for (&x, &y) in row_a.iter().zip(row_b) {
            if !x.is_finite() || !y.is_finite() {
                return Err(FieldError::NonFiniteState);
            }
            let delta = x - y;
            if !delta.is_finite() {
                return Err(FieldError::NonFiniteState);
            }
            scale = scale.max(delta.abs());
        }
    }
    if scale == 0.0 {
        return Ok(0.0);
    }
    if !scale.is_finite() {
        return Err(FieldError::NonFiniteState);
    }
    let normalized_sum = a
        .iter()
        .zip(b)
        .flat_map(|(row_a, row_b)| row_a.iter().zip(row_b))
        .map(|(&x, &y)| {
            let delta = x - y;
            let normalized = delta / scale;
            normalized * normalized
        })
        .sum::<f64>();
    if !normalized_sum.is_finite() {
        return Err(FieldError::NonFiniteState);
    }
    let distance = scale * normalized_sum.sqrt();
    if !distance.is_finite() {
        return Err(FieldError::NonFiniteState);
    }
    Ok(distance)
}
#[cfg(test)]
mod vector_numerics_tests {
    use super::*;

    #[test]
    fn dot_handles_large_cancelling_products() {
        assert_eq!(dot(&[1.0e308, 1.0e308], &[1.0e308, -1.0e308]), 0.0);
    }
}

#[cfg(test)]
mod subsystem_layer_integrity_tests {
    use super::*;

    #[test]
    fn evolve_reports_invalid_time_separately_from_dimension_mismatch() {
        let mut layer = SubsystemLayer::new("x", vec![0.0, 0.0]);
        assert_eq!(
            layer.evolve(&[1.0, 1.0], f64::NAN),
            Err(FieldError::InvalidTime)
        );
        assert_eq!(
            layer.evolve(&[1.0], 0.1),
            Err(FieldError::DimensionMismatch)
        );
        assert_eq!(layer.state, vec![0.0, 0.0]);
    }
}

#[cfg(test)]
mod distance_integrity_tests {
    use super::*;

    #[test]
    fn distance_rejects_nonfinite_intermediate_result() {
        let manifold = StereographicSphere::new(1, 1.0).unwrap();
        let x = vec![0.0];
        let y = vec![0.999_999_999_999];
        let distance = manifold.distance(&x, &y).unwrap();
        assert!(distance.is_finite());
    }
}

#[cfg(test)]
mod clock_integrity_tests {
    use super::*;

    #[test]
    fn step_rejects_invalid_existing_clock() {
        let manifold = StereographicSphere::new(1, 1.0).unwrap();
        let mut fiber =
            SystemFiber::new("x", manifold, vec![SubsystemLayer::new("layer", vec![0.0])]).unwrap();
        fiber.engine_clock = f64::NAN;
        assert_eq!(fiber.step(0.1), Err(FieldError::InvalidTime));
    }

    #[test]
    fn evolve_rejects_negative_existing_clock() {
        let manifold = StereographicSphere::new(1, 1.0).unwrap();
        let mut fiber =
            SystemFiber::new("x", manifold, vec![SubsystemLayer::new("layer", vec![0.0])]).unwrap();
        fiber.engine_clock = -1.0;
        assert_eq!(
            fiber.evolve(&[vec![0.0]], 0.1),
            Err(FieldError::InvalidTime)
        );
    }
}

#[cfg(test)]
mod matrix_distance_regression {
    use super::*;

    #[test]
    fn matrix_multiply_handles_cancelling_extreme_products() {
        let a = vec![vec![1.0e308, 1.0e308], vec![0.0, 0.0]];
        let b = vec![vec![1.0e308, 0.0], vec![-1.0e308, 0.0]];
        let product = mat_mul(&a, &b).unwrap();
        assert_eq!(product, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn matrix_distance_handles_extreme_finite_differences() {
        let a = vec![vec![5.0e307, -5.0e307]];
        let b = vec![vec![-5.0e307, 5.0e307]];
        let distance = matrix_distance(&a, &b).unwrap();
        assert!(distance.is_finite());
        assert!(distance > 0.0);
    }

    #[test]
    fn matrix_distance_rejects_non_finite_difference() {
        let a = vec![vec![f64::MAX]];
        let b = vec![vec![-f64::MAX]];
        assert_eq!(matrix_distance(&a, &b), Err(FieldError::NonFiniteState));
    }
}

fn matrix_inverse(a: &Matrix) -> Result<Matrix, FieldError> {
    let n = a.len();
    if n == 0
        || a.iter()
            .any(|r| r.len() != n || r.iter().any(|v| !v.is_finite()))
    {
        return Err(FieldError::InvalidTransport);
    }
    let scale = a.iter().flatten().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if scale == 0.0 {
        return Err(FieldError::NonInvertibleTransport);
    }
    let pivot_tol = EPS * scale;
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if aug[row][col].abs() > aug[pivot][col].abs() {
                pivot = row;
            }
        }
        if aug[pivot][col].abs() <= pivot_tol {
            return Err(FieldError::NonInvertibleTransport);
        }
        aug.swap(col, pivot);
        let p = aug[col][col];
        for cell in aug[col].iter_mut() {
            *cell /= p;
        }
        let pivot_row = aug[col].clone();
        for (row, values) in aug.iter_mut().enumerate() {
            if row == col {
                continue;
            }
            let factor = values[col];
            for (j, cell) in values.iter_mut().enumerate() {
                *cell -= factor * pivot_row[j];
            }
        }
    }
    if aug.iter().flatten().any(|v| !v.is_finite()) {
        return Err(FieldError::NonFiniteState);
    }
    Ok(aug.into_iter().map(|row| row[n..].to_vec()).collect())
}

#[derive(Debug, Clone, PartialEq)]
pub enum FieldError {
    DimensionMismatch,
    GeometryMismatch,
    InvalidGeometry,
    InvalidPoint,
    InvalidTransport,
    TransportMismatch,
    MissingTransport,
    NonInvertibleTransport,
    NonFiniteState,
    LayerMismatch,
    EmptyState,
    InvalidTime,
    InvalidDynamics,
    DuplicateId,
    MissingFiber,
    FiberInUse,
}
impl fmt::Display for FieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for FieldError {}

#[cfg(test)]
mod tests {
    #[test]
    fn failed_step_does_not_partially_commit_layer_state_or_clock() {
        let manifold = StereographicSphere::new(1, 1.0).unwrap();
        let mut fiber =
            SystemFiber::new("x", manifold, vec![SubsystemLayer::new("a", vec![0.25])]).unwrap();
        fiber.engine.biases[0][0] = f64::MAX;
        let before = fiber.clone();
        assert_eq!(fiber.step(1.0), Err(FieldError::NonFiniteState));
        assert_eq!(fiber, before);
    }

    use super::*;

    #[test]
    fn distance_invariants_hold() {
        let m = StereographicSphere::new(3, 1.0).unwrap();
        let x = vec![0.1, -0.2, 0.05];
        let y = vec![-0.15, 0.1, 0.2];
        assert!(m.distance(&x, &x).unwrap().abs() < 1.0e-12);
        assert!((m.distance(&x, &y).unwrap() - m.distance(&y, &x).unwrap()).abs() < 1.0e-12);
        assert!(m.distance(&x, &y).unwrap() > 0.0);
    }

    #[test]
    fn distance_is_consistent_with_origin_log() {
        let m = StereographicSphere::new(2, 1.0).unwrap();
        let x = vec![0.2, -0.3];
        let d = m.distance(&[0.0, 0.0], &x).unwrap();
        let tangent = m.log_origin(&x).unwrap();
        assert!((d - norm(&tangent)).abs() < 1.0e-12);
    }

    #[test]
    fn exp_and_log_reject_nonfinite_inputs_with_state_error() {
        let m = StereographicSphere::new(2, 1.0).unwrap();
        assert_eq!(
            m.exp_origin(&[f64::NAN, 0.0]),
            Err(FieldError::NonFiniteState)
        );
        assert_eq!(
            m.log_origin(&[f64::NAN, 0.0]),
            Err(FieldError::InvalidPoint)
        );
    }

    #[test]
    fn exp_log_round_trip() {
        let m = StereographicSphere::new(2, 1.0).unwrap();
        let x = vec![0.2, -0.3];
        let tangent = m.log_origin(&x).unwrap();
        let recovered = m.exp_origin(&tangent).unwrap();
        for (a, b) in x.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1.0e-12);
        }
    }

    #[test]
    fn exp_log_round_trip_near_boundary() {
        let m = StereographicSphere::new(2, 1.0).unwrap();
        let x = vec![0.999_999, 0.0];
        let tangent = m.log_origin(&x).unwrap();
        let recovered = m.exp_origin(&tangent).unwrap();
        assert!((x[0] - recovered[0]).abs() < 1.0e-9);
        assert!(m.contains(&recovered));
    }

    #[test]
    fn project_rejects_finite_coordinates_whose_norm_overflows() {
        let m = StereographicSphere::new(4, 1.0).unwrap();
        let x = vec![f64::MAX; 4];
        assert_eq!(m.project(&x), Err(FieldError::NonFiniteState));
    }

    #[test]
    fn geometry_norm_handles_large_finite_coordinates() {
        let m = StereographicSphere::new(2, 1.0e-300).unwrap();
        let p = vec![1.0e149, 1.0e149];
        assert!(m.contains(&p));
        let v = m.log_origin(&p).unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn distance_handles_large_coordinates_without_overflow() {
        let m = StereographicSphere::new(2, 1.0e-300).unwrap();
        let x = vec![1.0e149, 0.0];
        let y = vec![-1.0e149, 0.0];
        let d = m.distance(&x, &y).unwrap();
        assert!(d.is_finite());
        assert!(d > 0.0);
    }

    #[test]
    fn exp_origin_extreme_tangent_lands_on_a_finite_point() {
        // The sphere is compact: a tangent of 10³⁰⁸ wraps the
        // geodesic around it ~10³⁰⁸ times and lands at an honest,
        // arbitrary chart point — finite, but with no reason to sit
        // inside the operating cap.
        let m = StereographicSphere::new(2, 1.0).unwrap();
        let v = vec![1.0e308, 0.0];
        let x = m.exp_origin(&v).unwrap();
        assert!(x.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn barycenter_handles_cancelling_extreme_tangents() {
        let m = StereographicSphere::new(2, 1.0e-300).unwrap();
        let a = vec![1.0e149, 0.0];
        let b = vec![-1.0e149, 0.0];
        let center = m.frechet_mean(&[a, b]).unwrap();
        assert!(center.iter().all(|v| v.is_finite()));
        assert!(norm(&center) < 1.0e-12);
    }

    #[test]
    fn frechet_mean_is_where_the_tangent_gradient_vanishes() {
        // The defining property, not a Euclidean proxy: at the
        // Fréchet mean, Σᵢ log_m(xᵢ) = 0 — the Riemannian gradient
        // of ½Σd² vanishes.
        let m = StereographicSphere::new(2, 1.0).unwrap();
        let pts = vec![
            vec![0.4, 0.2],
            vec![-0.3, 0.5],
            vec![0.1, -0.35],
            vec![-0.2, -0.2],
        ];
        let mean = m.frechet_mean(&pts).unwrap();
        let mut grad = vec![0.0; 2];
        for p in &pts {
            let v = m.log_at(&mean, p).unwrap();
            grad[0] += v[0];
            grad[1] += v[1];
        }
        assert!(norm(&grad) < 1.0e-7, "residual gradient: {grad:?}");
        // And it beats its own seed: the tangent barycenter sits at
        // a strictly larger Σd² than the iterated mean (unless the
        // points were symmetric to begin with).
        let seed = m.tangent_barycenter(&pts).unwrap();
        let objective = |c: &Vector| -> f64 {
            pts.iter()
                .map(|p| m.distance(c, p).unwrap().powi(2))
                .sum::<f64>()
        };
        assert!(objective(&mean) <= objective(&seed) + 1.0e-12);
        // Symmetric pair: the mean is the geodesic midpoint.
        let a = vec![0.3, -0.1];
        let b = vec![-0.2, 0.4];
        let mid = m.frechet_mean(&[a.clone(), b.clone()]).unwrap();
        let full = m.distance(&a, &b).unwrap();
        assert!((m.distance(&a, &mid).unwrap() - 0.5 * full).abs() < 1.0e-7);
    }

    #[test]
    fn empty_fiber_has_no_aggregate_system_state() {
        let m = StereographicSphere::new(1, 1.0).unwrap();
        let fiber = SystemFiber::new("empty", m, Vec::new()).unwrap();
        assert_eq!(fiber.state(), Err(FieldError::EmptyState));
    }

    #[test]
    fn inverse_accepts_uniformly_scaled_invertible_transport() {
        let matrix = vec![vec![1.0e-15, 0.0], vec![0.0, 2.0e-15]];
        let inverse = matrix_inverse(&matrix).unwrap();
        assert!((inverse[0][0] - 1.0e15).abs() / 1.0e15 < 1.0e-12);
        assert!((inverse[1][1] - 5.0e14).abs() / 5.0e14 < 1.0e-12);
    }

    #[test]
    fn inverse_rejects_singular_transport() {
        let matrix = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert_eq!(
            matrix_inverse(&matrix),
            Err(FieldError::NonInvertibleTransport)
        );
    }

    #[test]
    fn fiber_step_rejects_mutated_invalid_engine() {
        let m = StereographicSphere::new(1, 1.0).unwrap();
        let mut fiber =
            SystemFiber::new("x", m, vec![SubsystemLayer::new("a", vec![0.0])]).unwrap();
        fiber.engine.damping = vec![f64::NAN];
        assert_eq!(fiber.step(0.1), Err(FieldError::InvalidDynamics));
    }

    #[test]
    fn fiber_rejects_nonfinite_layer_state() {
        let m = StereographicSphere::new(1, 1.0).unwrap();
        let layer = SubsystemLayer {
            id: "a".into(),
            state: vec![f64::NAN],
            gain: 1.0,
        };
        assert_eq!(
            SystemFiber::new("x", m, vec![layer]),
            Err(FieldError::NonFiniteState)
        );
    }

    #[test]
    fn fiber_rejects_duplicate_layer_ids() {
        let m = StereographicSphere::new(1, 1.0).unwrap();
        let layers = vec![
            SubsystemLayer::new("a", vec![0.0]),
            SubsystemLayer::new("a", vec![0.0]),
        ];
        assert_eq!(
            SystemFiber::new("x", m, layers),
            Err(FieldError::DuplicateId)
        );
    }

    #[test]
    fn fiber_rejects_nonfinite_layer_gain() {
        let m = StereographicSphere::new(1, 1.0).unwrap();
        let layer = SubsystemLayer {
            id: "a".into(),
            state: vec![0.0],
            gain: f64::NAN,
        };
        assert_eq!(
            SystemFiber::new("x", m, vec![layer]),
            Err(FieldError::InvalidDynamics)
        );
    }

    #[test]
    fn evolve_applies_engine_time_scale_like_step() {
        // `evolve` must respect the engine's time scale the same way
        // `step` does: a fiber at 2× moves the layer by dt·2·gain·dx
        // and advances its clock by the scaled interval.
        let m = StereographicSphere::new(1, 1.0).unwrap();
        let mut engine = SystemEngine::zero(1, 1);
        engine.time_scale = 2.0;
        let mut fiber =
            SystemFiber::with_engine("x", m, vec![SubsystemLayer::new("a", vec![0.0])], engine)
                .unwrap();
        fiber.evolve(&[vec![0.5]], 0.1).unwrap();
        assert!((fiber.layers[0].state[0] - 0.1).abs() < 1.0e-12);
        assert!((fiber.engine_clock - 0.2).abs() < 1.0e-12);
    }

    #[test]
    fn evolve_projects_departed_state_back_into_the_cap() {
        // A derivative that pushes a layer outside the cap must
        // come back through the manifold: the layer state stays a
        // valid cap point, not a raw Euler overshoot.
        let m = StereographicSphere::new(1, 1.0).unwrap();
        let mut fiber =
            SystemFiber::new("x", m, vec![SubsystemLayer::new("a", vec![0.9])]).unwrap();
        fiber.evolve(&[vec![1.0]], 0.5).unwrap();
        assert!(fiber.manifold.contains(&fiber.layers[0].state));
    }

    #[test]
    fn mobius_addition_identity_inverse_and_left_cancellation() {
        let m = StereographicSphere::new(3, 1.0).unwrap();
        let x = vec![0.2, -0.1, 0.3];
        let y = vec![-0.15, 0.25, -0.1];
        let zero = vec![0.0; 3];
        // Identity: x ⊕ 0 = x.
        for (a, b) in x.iter().zip(m.mobius_add(&x, &zero).unwrap().iter()) {
            assert!((a - b).abs() < 1.0e-12);
        }
        // Inverse: (−x) ⊕ x = 0.
        assert!(norm(&m.mobius_add(&scaled(&x, -1.0), &x).unwrap()) < 1.0e-12);
        // Left cancellation: (−x) ⊕ (x ⊕ y) = y — the property the
        // general log/exp pair rides on.
        let back = m
            .mobius_add(&scaled(&x, -1.0), &m.mobius_add(&x, &y).unwrap())
            .unwrap();
        for (a, b) in y.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0e-9);
        }
    }

    #[test]
    fn log_exp_at_arbitrary_basepoint_round_trips() {
        let m = StereographicSphere::new(2, 1.0).unwrap();
        let x = vec![0.3, -0.2];
        let y = vec![-0.1, 0.4];
        let v = m.log_at(&x, &y).unwrap();
        // The tangent's norm is the geodesic distance — same
        // distance-normalized convention as log_origin.
        assert!((norm(&v) - m.distance(&x, &y).unwrap()).abs() < 1.0e-9);
        let recovered = m.exp_at(&x, &v).unwrap();
        for (a, b) in y.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1.0e-9);
        }
    }

    #[test]
    fn geodesic_fraction_matches_distance_fraction() {
        let m = StereographicSphere::new(2, 1.0).unwrap();
        let x = vec![0.2, -0.3];
        let y = vec![-0.4, 0.1];
        let full = m.distance(&x, &y).unwrap();
        for (a, b) in x.iter().zip(m.geodesic(&x, &y, 0.0).unwrap().iter()) {
            assert!((a - b).abs() < 1.0e-9);
        }
        for (a, b) in y.iter().zip(m.geodesic(&x, &y, 1.0).unwrap().iter()) {
            assert!((a - b).abs() < 1.0e-9);
        }
        let mid = m.geodesic(&x, &y, 0.5).unwrap();
        assert!((m.distance(&x, &mid).unwrap() - 0.5 * full).abs() < 1.0e-9);
        assert!((m.distance(&mid, &y).unwrap() - 0.5 * full).abs() < 1.0e-9);
    }

    #[test]
    fn disagreement_heats_the_field_and_fluctuation_is_deterministic() {
        fn build() -> GlobalField {
            let manifold = StereographicSphere::new(1, 1.0).unwrap();
            let mut field = GlobalField::new(manifold);
            field.cooling_rate = 0.0; // isolate the heating term
            field
                .add_fiber(
                    SystemFiber::new(
                        "a",
                        field.manifold.clone(),
                        vec![SubsystemLayer::new("s", vec![0.1])],
                    )
                    .unwrap(),
                )
                .unwrap();
            field
                .add_fiber(
                    SystemFiber::new(
                        "b",
                        field.manifold.clone(),
                        vec![SubsystemLayer::new("s", vec![0.4])],
                    )
                    .unwrap(),
                )
                .unwrap();
            field
                .add_transport(GaugeTransport::new("b", "a", vec![vec![1.0]]).unwrap())
                .unwrap();
            field
        }
        // A cold field starts deterministic: temperature 0 → sigma 0.
        let mut cold = build();
        let before = cold.fibers[1].layers[0].state.clone();
        cold.synchronize(0.1, 0.0).unwrap();
        // The disagreement snapped the temperature floor up — the
        // field remembers the tension it is relaxing.
        assert!(cold.temperature > 0.0);
        assert!(cold.temperature <= TEMPERATURE_CAP + 1.0e-12);
        // Two identical fields take identical kicks — stochastic in
        // structure, reproducible in sequence.
        let mut a = build();
        let mut b = build();
        a.synchronize(0.1, 0.0).unwrap(); // heats both fields
        b.synchronize(0.1, 0.0).unwrap();
        a.synchronize(0.1, 0.0).unwrap(); // now hot: the Langevin kick fires
        b.synchronize(0.1, 0.0).unwrap();
        assert_eq!(a.fibers[1].layers[0].state, b.fibers[1].layers[0].state);
        assert_eq!(a.rng_state, b.rng_state);
        assert_ne!(a.fibers[1].layers[0].state, before);
    }

    #[test]
    fn a_hot_bath_heats_a_coherent_field() {
        // Two fibers already at consensus: internal disagreement is
        // ~zero, so only the measured physical reservoir can raise
        // the temperature. The bath floors the field the same way
        // tension does — the machine's own heat is a real source,
        // not modeled weather.
        let manifold = StereographicSphere::new(1, 1.0).unwrap();
        let mut field = GlobalField::new(manifold);
        field.cooling_rate = 0.0;
        for id in ["a", "b"] {
            field
                .add_fiber(
                    SystemFiber::new(
                        id,
                        field.manifold.clone(),
                        vec![SubsystemLayer::new("s", vec![0.2])],
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        field
            .add_transport(GaugeTransport::new("b", "a", vec![vec![1.0]]).unwrap())
            .unwrap();

        field.synchronize(0.0, 0.3).unwrap();
        assert!(field.disagreement < 1.0e-9);
        assert!(
            (field.temperature - 0.3).abs() < 1.0e-9,
            "the bath must floor the field temperature: {}",
            field.temperature
        );
        // And it obeys the same cap — the reservoir can fill the
        // field, never overflow it.
        field.synchronize(0.0, 99.0).unwrap();
        assert!(field.temperature <= TEMPERATURE_CAP + 1.0e-12);
    }

    #[test]
    fn antipode_is_the_maximally_distant_involution() {
        let m = StereographicSphere::new(3, 1.0).unwrap();
        let x = vec![0.2, -0.1, 0.3];
        // Inversion in the equator circle, negated: −x/(κ‖x‖²) —
        // the far hemisphere, the genuine "other side."
        let r2 = dot(&x, &x);
        let expected = scaled(&x, -1.0 / r2);
        let anti = m.antipode(&x).unwrap();
        for (a, b) in anti.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1.0e-12);
        }
        // Antipodal points saturate the sphere's diameter:
        // d(x, antipode(x)) = π/√κ. The denominator of the distance
        // quotient is exactly zero here; in floating point it
        // survives as ~1e-16, so the measured distance is π to
        // within double precision, not to machine epsilon.
        assert!((m.distance(&x, &anti).unwrap() - std::f64::consts::PI).abs() < 1.0e-6);
        // The antipode lives outside the operating cap — the
        // southern hemisphere — and is still a valid chart point.
        assert!(!m.contains(&anti));
        // The map is its own inverse.
        let back = m.antipode(&anti).unwrap();
        for (a, b) in back.iter().zip(x.iter()) {
            assert!((a - b).abs() < 1.0e-9);
        }
        // The pole's antipode is the unrepresentable south pole —
        // an error, not a silent clamp.
        assert_eq!(m.antipode(&[0.0; 3]), Err(FieldError::NonFiniteState));
    }

    #[test]
    fn point_reflection_places_the_center_at_the_geodesic_midpoint() {
        let m = StereographicSphere::new(2, 1.0).unwrap();
        let x = vec![0.2, -0.3];
        let c = vec![-0.1, 0.15];
        let image = m.point_reflection(&x, &c).unwrap();
        // c is the geodesic midpoint of x and its image.
        let d = m.distance(&x, &image).unwrap();
        assert!((m.distance(&x, &c).unwrap() - 0.5 * d).abs() < 1.0e-9);
        assert!((m.distance(&c, &image).unwrap() - 0.5 * d).abs() < 1.0e-9);
        // Cartan involution: reflecting again returns x.
        let back = m.point_reflection(&image, &c).unwrap();
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0e-9);
        }
        // Reflection through the pole is the polar mirror −x: the
        // great circle through x and the pole continues past it and
        // lands at the same polar angle on the opposite ray. That
        // is NOT the spherical antipode — it stays inside the cap.
        let origin = vec![0.0; 2];
        let reflected = m.point_reflection(&x, &origin).unwrap();
        for (a, b) in scaled(&x, -1.0).iter().zip(reflected.iter()) {
            assert!((a - b).abs() < 1.0e-9);
        }
    }

    fn two_fiber_field() -> GlobalField {
        let manifold = StereographicSphere::new(1, 1.0).unwrap();
        let mut field = GlobalField::new(manifold);
        field.cooling_rate = 0.0;
        field
            .add_fiber(
                SystemFiber::new(
                    "a",
                    field.manifold.clone(),
                    vec![SubsystemLayer::new("s", vec![0.1])],
                )
                .unwrap(),
            )
            .unwrap();
        field
            .add_fiber(
                SystemFiber::new(
                    "b",
                    field.manifold.clone(),
                    vec![SubsystemLayer::new("s", vec![0.4])],
                )
                .unwrap(),
            )
            .unwrap();
        field
            .add_transport(GaugeTransport::new("b", "a", vec![vec![1.0]]).unwrap())
            .unwrap();
        field
    }

    #[test]
    fn diverge_grows_disagreement_where_synchronize_shrinks_it() {
        let mut contracted = two_fiber_field();
        let mut expanded = two_fiber_field();
        // A zero step measures without moving — the standing
        // variance the two directions then act on.
        expanded.synchronize(0.0, 0.0).unwrap();
        let d0 = expanded.disagreement;
        assert!(d0 > 0.0);
        // Cold fields for the moves under test: the Langevin kick
        // samples the pre-update temperature, which the measure step
        // just snapped to the standing disagreement — a seeded draw
        // of the same order as the push can legitimately flip one
        // step's variance. Zeroing it asserts the deterministic
        // direction; fluctuation behavior is covered elsewhere.
        contracted.temperature = 0.0;
        expanded.temperature = 0.0;
        contracted.synchronize(0.2, 0.0).unwrap();
        expanded.diverge(0.2, 0.0).unwrap();
        // The stored disagreement is the pre-move tension; measure
        // again to see where each direction left the field.
        contracted.synchronize(0.0, 0.0).unwrap();
        expanded.synchronize(0.0, 0.0).unwrap();
        // Descent pulled fiber b toward the consensus: variance
        // down. Ascent pushed it away: variance up.
        assert!(contracted.disagreement < d0);
        assert!(expanded.disagreement > d0);
        // Same thermostat either way — and the diverged field is
        // the hotter one, since its tension is now the larger.
        assert!(expanded.temperature >= contracted.temperature);
    }

    #[test]
    fn anticonsensus_is_the_far_side_of_consensus() {
        let mut field = two_fiber_field();
        field.synchronize(0.1, 0.0).unwrap();
        let consensus = field.consensus().unwrap().unwrap();
        let anti = field.anticonsensus().unwrap().unwrap();
        // The true spherical antipode: −c/(κ‖c‖²), at the maximal
        // distance π/√κ on the far side of the sphere.
        let c_norm = norm(&consensus);
        assert_eq!(anti, scaled(&consensus, -1.0 / (c_norm * c_norm)));
        // Singular-point tolerance, as in the antipode test above:
        // the measured distance is π to within double precision.
        assert!(
            (field.manifold.distance(&consensus, &anti).unwrap() - std::f64::consts::PI).abs()
                < 1.0e-6
        );
        // Nothing agreed → nothing to oppose: an empty field has
        // neither a consensus nor an anticonsensus.
        let empty = GlobalField::new(StereographicSphere::new(1, 1.0).unwrap());
        assert_eq!(empty.anticonsensus().unwrap(), None);
    }

    #[test]
    fn removal_is_the_strict_opposite_of_addition() {
        let mut field = two_fiber_field();
        // A referenced fiber cannot leave while its routes stand.
        assert_eq!(field.remove_fiber("b"), Err(FieldError::FiberInUse));
        // The anchor fiber cannot leave while others remain.
        assert_eq!(field.remove_fiber("a"), Err(FieldError::FiberInUse));
        // Sever the route first — remove_transport hands it back.
        let transport = field.remove_transport("b", "a").unwrap();
        assert_eq!(transport.from, "b");
        assert_eq!(
            field.remove_transport("b", "a"),
            Err(FieldError::MissingTransport)
        );
        // Now b may leave — and it leaves as the fiber it was.
        let fiber = field.remove_fiber("b").unwrap();
        assert_eq!(fiber.id, "b");
        assert_eq!(field.fibers.len(), 1);
        // The lone anchor may finally go; the field is empty, not
        // corrupted.
        field.remove_fiber("a").unwrap();
        assert!(field.fibers.is_empty());
        assert_eq!(field.remove_fiber("a"), Err(FieldError::MissingFiber));
    }
}

#[cfg(test)]
mod global_step_tests {
    use super::*;

    fn identity_transport(from: &str, to: &str) -> GaugeTransport {
        GaugeTransport::new(from, to, vec![vec![1.0]]).unwrap()
    }

    #[test]
    fn global_step_evolves_fibers_then_synchronizes() {
        let manifold = StereographicSphere::new(1, 1.0).unwrap();
        let first = SystemFiber::new(
            "first",
            manifold.clone(),
            vec![SubsystemLayer::new("state", vec![0.1])],
        )
        .unwrap();
        let second = SystemFiber::new(
            "second",
            manifold.clone(),
            vec![SubsystemLayer::new("state", vec![0.2])],
        )
        .unwrap();

        let mut field = GlobalField::new(manifold);
        field.integration_gain = 1.0;
        field.cooling_rate = 0.0;
        field.add_fiber(first).unwrap();
        field.add_fiber(second).unwrap();
        field
            .add_transport(identity_transport("second", "first"))
            .unwrap();

        let consensus = field.step(0.1, 0.0).unwrap().unwrap();

        assert!((field.fibers[0].engine_clock - 0.1).abs() < 1.0e-12);
        assert!((field.fibers[1].engine_clock - 0.1).abs() < 1.0e-12);
        assert!(consensus[0] > 0.1 && consensus[0] < 0.2);
        assert!(field.fibers[1].state().unwrap()[0] < 0.2);
        assert!(field.disagreement.is_finite());
    }

    #[test]
    fn global_step_is_transactional_across_fibers() {
        let manifold = StereographicSphere::new(1, 1.0).unwrap();
        let first = SystemFiber::new(
            "first",
            manifold.clone(),
            vec![SubsystemLayer::new("state", vec![0.1])],
        )
        .unwrap();
        let second = SystemFiber::new(
            "second",
            manifold.clone(),
            vec![SubsystemLayer::new("state", vec![0.2])],
        )
        .unwrap();

        let mut field = GlobalField::new(manifold);
        field.add_fiber(first).unwrap();
        field.add_fiber(second).unwrap();
        field
            .add_transport(identity_transport("second", "first"))
            .unwrap();
        field.fibers[1].engine.damping[0] = f64::NAN;

        let before_first_state = field.fibers[0].state().unwrap();
        let before_second_state = field.fibers[1].state().unwrap();
        let before_first_clock = field.fibers[0].engine_clock;
        let before_second_clock = field.fibers[1].engine_clock;
        let before_disagreement = field.disagreement;
        let before_temperature = field.temperature;
        let before_cooling_rate = field.cooling_rate;

        assert_eq!(field.step(0.1, 0.0), Err(FieldError::InvalidDynamics));

        assert_eq!(field.fibers[0].state().unwrap(), before_first_state);
        assert_eq!(field.fibers[1].state().unwrap(), before_second_state);
        assert_eq!(field.fibers[0].engine_clock, before_first_clock);
        assert_eq!(field.fibers[1].engine_clock, before_second_clock);
        assert!(field.fibers[1].engine.damping[0].is_nan());
        assert_eq!(field.disagreement, before_disagreement);
        assert_eq!(field.temperature, before_temperature);
        assert_eq!(field.cooling_rate, before_cooling_rate);
    }

    #[test]
    fn provenance_lattice_verifies() {
        // The creator lattice (PROVENANCE.md): every marker carries the
        // 0x4753 tag, its index, its message fragment, and an
        // fnv1a-24 chaincheck that commits to the previous fragment.
        // This test is the excision tripwire — a removed or edited
        // letter breaks the fingerprint and the chain checks.
        fn fnv24(data: &[u8]) -> u32 {
            let mut h: u32 = 0x811C_9DC5;
            for &b in data {
                h = (h ^ b as u32).wrapping_mul(0x0100_0193);
            }
            h & 0xFF_FFFF
        }

        let markers: [u64; 8] = [
            MARKER_G00,
            runtime::MARKER_G01,
            engine::MARKER_G02,
            motor::MARKER_G03,
            crate::daemon::ipc::MARKER_G04,
            crate::daemon::tick::MARKER_G05,
            crate::state::header::MARKER_G06,
            crate::store::mmap_state::MARKER_G07,
        ];

        let mut prev: [u8; 2] = *b"GS";
        let mut fragment = Vec::new();
        for (idx, &m) in markers.iter().enumerate() {
            assert_eq!(m >> 48, 0x4753, "marker {idx} lost its tag");
            assert_eq!(
                ((m >> 40) & 0xFF) as usize,
                idx,
                "marker {idx} out of position"
            );
            let frag = [((m >> 32) & 0xFF) as u8, ((m >> 24) & 0xFF) as u8];
            let mut buf = vec![0x47, 0x53, idx as u8];
            buf.extend_from_slice(&frag);
            buf.extend_from_slice(&prev);
            assert_eq!(
                (m & 0xFF_FFFF) as u32,
                fnv24(&buf),
                "chain check failed at marker {idx}"
            );
            fragment.extend_from_slice(&frag);
            prev = frag;
        }
        assert_eq!(
            &fragment[..7],
            b"MINDY B",
            "the field's half of the message drifted"
        );
        assert_eq!(provenance_fingerprint(), 0x007f_63f1_b36f);
    }
}
