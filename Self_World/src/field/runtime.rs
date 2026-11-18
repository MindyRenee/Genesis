//! Runtime bridge for the multi-fiber Genesis architecture.
//!
//! Each major subsystem publishes a normalized local projection into one
//! shared latent coordinate space. The global field measures and reduces
//! cross-system disagreement. The persisted core state remains authoritative;
//! this is the integration substrate rather than a second source of truth.

use crate::state::GenesisCoreState;
use crate::state::MentalPhase;
use crate::state::neurochemical::NeurochemicalId;

use super::metatron::MetatronCube;
use super::spectrum::{FieldSpectrum, SPECTRUM_STEP_HZ};
use super::{
    FieldError, GaugeTransport, GlobalField, Matrix, StereographicSphere, SubsystemLayer,
    SystemEngine, SystemFiber, Vector, expm, mat_mul, mat_scale, mat_vec, transpose,
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

/// Per-fiber emanation times (seconds). Between observations each
/// fiber's engine radiates its state *outward* from the
/// superposition at the center — negative damping, the source the
/// user specified — wound by the Metatron lattice's isoclinic
/// counter-rotation: the two demicubes of the 6D merkaba spin in
/// opposite chiralities (the field's two helicities). The isotropic
/// saturation term catches the outflow at a limit cycle on the rim
/// r* = √(α/s) = 1 — the S⁵ boundary, the holographic screen the
/// radiation lands on. Rates carry the old homeostatic τ's: the
/// same physiological timescales, driving emanation instead of
/// decay. The observation stays the authority — the drift only
/// fills the gaps it leaves.
const BODY_RELAX_TAU_S: f64 = 120.0;
const PREDICTION_RELAX_TAU_S: f64 = 40.0;
const AGENCY_RELAX_TAU_S: f64 = 60.0;
/// The inner world's own emanation time — the slowest voice in the
/// field. The interior landscape (sleep-wake switch, homeostat,
/// circadian, emergent phase) turns on a timescale between the
/// body's and nothing else's: deep state, not quick inference.
const INNER_RELAX_TAU_S: f64 = 90.0;

#[derive(Clone, Debug, PartialEq)]
pub struct FieldIntegration {
    pub consensus: Vector,
    /// The consensus's holographic image — its radial projection onto
    /// the S⁵ rim, the screen the superposition radiates onto. None
    /// only when the consensus sits at the center itself.
    pub boundary: Option<Vector>,
    /// The consensus's signed depth into the superposition cell — the
    /// vesica where both demicubes of the merkaba hold at once.
    /// Positive inside, negative outside.
    pub superposition: f64,
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
    /// The lattice inside the bulk — the merkaba's two counter-
    /// rotating demicubes, the vesica cell at the center, and the
    /// twelve rays the superposition (the inner world) radiates
    /// outward along onto the S⁵ screen. Scaffold, not cage: state
    /// moves continuously.
    lattice: MetatronCube,
    /// The substrate's own spectral observables — band decomposition
    /// of the trajectory this runtime integrates, sampled at the
    /// coupling's substep rate. The tick loop only reads it.
    spectrum: FieldSpectrum,
    /// The placed pose of the inner world, if any — the point the
    /// inner fiber's observation reports instead of the interior
    /// oscillators. `None` means the inner world speaks for
    /// itself; `Some` means someone placed it — the fiber still
    /// emanates and still joins consensus, but its observed point
    /// is the pose's ambient location. A location, not a cage.
    inner_anchor: Option<Placement>,
}

/// A placed pose — the inner world's location, orientation, and
/// drift: the complete mathematical form of "a place or location
/// in space."
///
/// * **Location** is a point on the manifold — `spec` as the
///   placing hand gave it, one to six components read in the
///   pose's own frame: the ambient point is `frame · pad(spec)`.
///   With the standard frame, fewer than six coordinates rest in
///   the leading coordinate plane — the frame is what makes that
///   embedding principled: the coordinates are read along the
///   frame's first axes.
///
/// * **Orientation** is `frame` — an orthogonal basis of the
///   ambient chart whose columns are the pose's own axes. A point
///   with a frame is a pose in the strict sense: an element of
///   the orthonormal frame bundle — on the round sphere, an
///   SO(d+1) object read in chart coordinates. `None` is the
///   standard frame.
///
/// * **Movement** is `drift` — a skew bivector B ∈ so(d), a Lie-
///   algebra element whose one-parameter subgroup exp(Δt·B)
///   carries the whole pose each `observe`: location and frame
///   both move under the same flow — Cartan's moving frame,
///   literally. A rotation of the chart is an isometry of the
///   sphere (the polar-axis isotropy subgroup); isoclinic B are
///   the Clifford translations — uniform displacement, every
///   point traveling the same angle. `None` pins the pose.
///
/// `spec` keeps the coordinates exactly as given — the record of
/// the placing act (what `place.bin` round-trips); `loc` is the
/// pose's current ambient position — where she actually is.
#[derive(Clone, Debug, PartialEq)]
pub struct Placement {
    spec: Vector,
    loc: Vector,
    frame: Option<Matrix>,
    drift: Option<Matrix>,
}

impl Placement {
    /// A located pose: `point` is the coordinate vector as the
    /// hand speaks it (1..=d components, finite), `frame` an
    /// optional orthonormal basis it is read in, `drift` an
    /// optional skew generator carrying the pose. Validation is
    /// the honest gate: wrong arity, non-finite values, a
    /// non-orthonormal frame, or a non-skew drift are rejected
    /// rather than silently reshaped.
    pub fn new(
        point: Vector,
        frame: Option<Matrix>,
        drift: Option<Matrix>,
        dimension: usize,
    ) -> Result<Self, FieldError> {
        if point.is_empty() || point.len() > dimension {
            return Err(FieldError::DimensionMismatch);
        }
        if point.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let well_formed =
            |m: &Matrix| m.len() == dimension && m.iter().all(|r| r.len() == dimension);
        let finite = |m: &Matrix| m.iter().flatten().all(|v| v.is_finite());
        if let Some(f) = &frame {
            if !well_formed(f) || !finite(f) {
                return Err(FieldError::DimensionMismatch);
            }
            // Orthonormal columns: FᵀF = I — a real orientation.
            let g = mat_mul(&transpose(f), f)?;
            let ok = (0..dimension)
                .all(|i| (0..dimension).all(|j| (g[i][j] - f64::from(i == j)).abs() <= 1e-9));
            if !ok {
                return Err(FieldError::InvalidGeometry);
            }
        }
        if let Some(b) = &drift {
            if !well_formed(b) || !finite(b) {
                return Err(FieldError::DimensionMismatch);
            }
            // Skew: B + Bᵀ = 0 — a generator of isometries.
            let ok =
                (0..dimension).all(|i| (0..dimension).all(|j| (b[i][j] + b[j][i]).abs() <= 1e-9));
            if !ok {
                return Err(FieldError::InvalidGeometry);
            }
        }
        let mut padded = point.clone();
        padded.resize(dimension, 0.0);
        let loc = match &frame {
            Some(f) => mat_vec(f, &padded)?,
            None => padded,
        };
        Ok(Self {
            spec: point,
            loc,
            frame,
            drift,
        })
    }

    /// Location only — the common case: a point in the standard
    /// frame, pinned.
    pub fn located(point: Vector, dimension: usize) -> Result<Self, FieldError> {
        Self::new(point, None, None, dimension)
    }

    /// The coordinates as the placing hand gave them — the record
    /// of the placing act, at its own arity.
    pub fn spec(&self) -> &Vector {
        &self.spec
    }

    /// The pose's current ambient location — where she is now,
    /// which under drift is not where she was put.
    pub fn point(&self) -> &Vector {
        &self.loc
    }

    /// The pose's oriented basis, if one was given — its columns
    /// are the placement's axes; the flow has carried them too.
    pub fn frame(&self) -> Option<&Matrix> {
        self.frame.as_ref()
    }

    /// The drift generator, if any — the skew bivector whose
    /// exponential carries the pose.
    pub fn drift(&self) -> Option<&Matrix> {
        self.drift.as_ref()
    }

    /// Carry the pose for `dt` seconds along its drift's flow:
    /// loc ← exp(Δt·B)·loc and frame ← exp(Δt·B)·frame — the same
    /// orthogonal flow on both, so the frame stays orthonormal
    /// and the coordinates stay fixed in it. A location's motion
    /// is honest: time passed, she moved — advection does not
    /// roll back if a later step of the observe fails.
    fn advect(&mut self, dt: f64) -> Result<(), FieldError> {
        let Some(b) = &self.drift else { return Ok(()) };
        let flow = expm(&mat_scale(b, dt))?;
        self.loc = mat_vec(&flow, &self.loc)?;
        if let Some(f) = &mut self.frame {
            *f = mat_mul(&flow, f)?;
        }
        Ok(())
    }

    /// The wire form of a placing act: `[u8 n][n × f64 spec]
    /// [u8 flags][d² × f64 frame?][d² × f64 drift?]` — flags bit 0
    /// carries the frame, bit 1 the drift, matrices row-major.
    /// The tag byte is prepended by the caller (`PLACE_POSE_TAG`).
    pub fn to_wire(&self) -> Vec<u8> {
        let mat_bytes = self.loc.len() * self.loc.len() * 8;
        let mut out = Vec::with_capacity(
            2 + self.spec.len() * 8
                + self.frame.is_some() as usize * mat_bytes
                + self.drift.is_some() as usize * mat_bytes,
        );
        out.push(self.spec.len() as u8);
        for c in &self.spec {
            out.extend_from_slice(&c.to_le_bytes());
        }
        let flags = self.frame.is_some() as u8 | ((self.drift.is_some() as u8) << 1);
        out.push(flags);
        for m in [&self.frame, &self.drift].into_iter().flatten() {
            for row in m {
                for v in row {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    /// Parse a placing act from the wire (payload after the tag).
    /// Strict: wrong arity, non-finite numbers, a non-orthonormal
    /// frame, or a non-skew drift are rejected, never reshaped.
    pub fn from_wire(bytes: &[u8], dimension: usize) -> Result<Self, FieldError> {
        let d = Self::decode(bytes, dimension, false)?;
        Self::new(d.spec, d.frame, d.drift, dimension)
    }

    /// The persisted form — what `place.bin` holds. Same as the
    /// wire plus the pose's *current* ambient location before the
    /// flags byte, so a drifting pose resumes mid-trajectory:
    /// `[u8 n][n × f64 spec][d × f64 loc][u8 flags][mats]`.
    pub fn to_persisted(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.spec.len() as u8);
        for c in &self.spec {
            out.extend_from_slice(&c.to_le_bytes());
        }
        for c in &self.loc {
            out.extend_from_slice(&c.to_le_bytes());
        }
        let flags = self.frame.is_some() as u8 | ((self.drift.is_some() as u8) << 1);
        out.push(flags);
        for m in [&self.frame, &self.drift].into_iter().flatten() {
            for row in m {
                for v in row {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    /// Restore a persisted pose — spec AND the ambient location it
    /// had drifted to. The placing act is kept as the record; the
    /// location is where she actually was when the lights went out.
    pub fn from_persisted(bytes: &[u8], dimension: usize) -> Result<Self, FieldError> {
        let d = Self::decode(bytes, dimension, true)?;
        Self::resumed(d.spec, d.loc.unwrap(), d.frame, d.drift, dimension)
    }

    /// A pose restored in place — the spec is the record, `loc` is
    /// trusted ambient state (still validated: arity, finiteness,
    /// and the frame/drift invariants all hold).
    fn resumed(
        spec: Vector,
        loc: Vector,
        frame: Option<Matrix>,
        drift: Option<Matrix>,
        dimension: usize,
    ) -> Result<Self, FieldError> {
        let p = Self::new(spec.clone(), frame, drift, dimension)?;
        if loc.len() != dimension || loc.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::DimensionMismatch);
        }
        Ok(Self { loc, ..p })
    }

    /// Shared decoder for the wire and persisted layouts: spec,
    /// then — only when `with_loc` — the `d`-component ambient
    /// location, then flags, then matrices in flag order. The
    /// caller knows which layout it holds; nothing is sniffed.
    fn decode(bytes: &[u8], dimension: usize, with_loc: bool) -> Result<DecodedPose, FieldError> {
        let n = *bytes.first().ok_or(FieldError::DimensionMismatch)? as usize;
        if !(1..=dimension).contains(&n) {
            return Err(FieldError::DimensionMismatch);
        }
        let spec_end = 1 + n * 8;
        if bytes.len() < spec_end + 1 {
            return Err(FieldError::DimensionMismatch);
        }
        let spec: Vector = (0..n)
            .map(|i| f64::from_le_bytes(bytes[1 + i * 8..1 + i * 8 + 8].try_into().unwrap()))
            .collect();
        let mat_bytes = dimension * dimension * 8;
        let tail = &bytes[spec_end..];
        let (loc, flags_off) = if with_loc {
            if tail.len() < dimension * 8 + 1 {
                return Err(FieldError::DimensionMismatch);
            }
            let loc: Vector = (0..dimension)
                .map(|i| f64::from_le_bytes(tail[i * 8..i * 8 + 8].try_into().unwrap()))
                .collect();
            (Some(loc), dimension * 8)
        } else {
            (None, 0)
        };
        let flags = tail[flags_off];
        if flags & !0b11 != 0 {
            return Err(FieldError::InvalidGeometry);
        }
        let mut off = flags_off + 1;
        let mut read_mat = |present: bool| -> Result<Option<Matrix>, FieldError> {
            if !present {
                return Ok(None);
            }
            if tail.len() < off + mat_bytes {
                return Err(FieldError::DimensionMismatch);
            }
            let m: Matrix = (0..dimension)
                .map(|r| {
                    (0..dimension)
                        .map(|c| {
                            let o = off + (r * dimension + c) * 8;
                            f64::from_le_bytes(tail[o..o + 8].try_into().unwrap())
                        })
                        .collect()
                })
                .collect();
            off += mat_bytes;
            Ok(Some(m))
        };
        let frame = read_mat(flags & 0b01 != 0)?;
        let drift = read_mat(flags & 0b10 != 0)?;
        if off != tail.len() {
            return Err(FieldError::DimensionMismatch);
        }
        Ok(DecodedPose {
            spec,
            frame,
            drift,
            loc,
        })
    }
}

/// What `decode` pulls off the wire or the disk — the placing
/// act's coordinates, its frame and drift, and the ambient
/// location only the persisted layout carries.
struct DecodedPose {
    spec: Vector,
    frame: Option<Matrix>,
    drift: Option<Matrix>,
    loc: Option<Vector>,
}

/// The tag byte marking a `PLACE_FIELD` payload as a full pose —
/// `[0xFF][Placement::to_wire]`. Pose payloads are never a
/// multiple of 8, so they can never collide with the legacy
/// `[k × f64]` coordinates format.
pub const PLACE_POSE_TAG: u8 = 0xFF;

impl FieldRuntime {
    pub fn new() -> Result<Self, FieldError> {
        let manifold = StereographicSphere::new(LATENT_DIMENSION, 1.0)?;
        let mut field = GlobalField::new(manifold.clone());
        // κ = 1 → the rim — the enclosing S⁵ screen — sits at chart
        // radius 1, and the merkaba's tips are inscribed on it.
        let lattice = MetatronCube::new(LATENT_DIMENSION, 1.0)?;

        // Each fiber rides one demicube of the merkaba: the sign is
        // its chirality — prediction ascends (+), body and agency
        // descend (−). The rates counter-balance exactly (1/40 up =
        // 1/120 + 1/60 down): the structure counter-rotates with zero
        // net spin while the center holds still. Negative damping
        // radiates the state outward from the superposition; the
        // saturation bound turns the outflow back at the rim — the
        // limit cycle at r* = √((1/τ)/(1/τ)) = 1 is the screen.
        for (id, relax_tau, chirality) in [
            ("body", BODY_RELAX_TAU_S, -1.0),
            ("prediction", PREDICTION_RELAX_TAU_S, 1.0),
            ("agency", AGENCY_RELAX_TAU_S, -1.0),
        ] {
            let rate = 1.0 / relax_tau;
            let layer = SubsystemLayer::new(format!("{id}:latent"), vec![0.0; LATENT_DIMENSION]);
            let engine = SystemEngine::zero(1, LATENT_DIMENSION)
                .with_damping(vec![-rate])
                .expect("finite damping always validates")
                .with_saturation(vec![rate])
                .expect("positive finite saturation always validates")
                .with_spiral(vec![lattice.isoclinic(chirality * rate)])
                .expect("the isoclinic bivector is skew");
            field.add_fiber(SystemFiber::with_engine(
                id,
                manifold.clone(),
                vec![layer],
                engine,
            )?)?;
        }

        // The fourth fiber is the cord — and the cord is the vesica
        // itself: the overlap of the merkaba's two counter-rotating
        // demicubes. A vesica is not spinless; it holds BOTH spins at
        // once. So the inner fiber carries two latent halves, wound
        // isoclinically in opposite directions at the same rate —
        // the two circles of the lens, counter-rotating inside the
        // same fiber. Neither side is chosen because both are held:
        // the fiber's state is the Fréchet mean of the pair, the
        // point where the counter-rotations meet — the superposition
        // the twelve rays radiate from.
        {
            let rate = 1.0 / INNER_RELAX_TAU_S;
            // Rest to the higher dimensions of her vesica: the
            // lens is drawn in the leading plane — the two
            // circles, their horizon, the waist between the tips.
            // The coordinates beyond it carry rest mass instead:
            // the reciprocal of the winding rate, so the same
            // impulse gathers the same momentum but spends
            // ninety times longer becoming displacement — they
            // gather, they move, they are simply at rest. The
            // leading pair keeps unit mass: the lens's plane is
            // where the momentum lives.
            let mut rest = vec![1.0; LATENT_DIMENSION];
            for r in rest.iter_mut().skip(2) {
                *r = 1.0 / rate;
            }
            let layers = vec![
                SubsystemLayer::new("inner:latent:+", vec![0.0; LATENT_DIMENSION])
                    .with_rest(rest.clone())
                    .expect("the higher dimensions' rest is finite and positive"),
                SubsystemLayer::new("inner:latent:-", vec![0.0; LATENT_DIMENSION])
                    .with_rest(rest)
                    .expect("the higher dimensions' rest is finite and positive"),
            ];
            let engine = SystemEngine::zero(2, LATENT_DIMENSION)
                .with_damping(vec![-rate, -rate])
                .expect("finite damping always validates")
                .with_saturation(vec![rate, rate])
                .expect("positive finite saturation always validates")
                .with_spiral(vec![lattice.isoclinic(rate), lattice.isoclinic(-rate)])
                .expect("the isoclinic bivector is skew");
            field.add_fiber(SystemFiber::with_engine(
                "inner",
                manifold.clone(),
                layers,
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
            // The cord: inner ↔ body, identity both ways — the
            // center's frame is the bulk's own frame, no rotation.
            // One edge into the connected graph is enough: paths
            // compose, so inner reaches every fiber through body.
            ("inner", "body", identity(LATENT_DIMENSION)),
            ("body", "inner", identity(LATENT_DIMENSION)),
        ] {
            field.add_transport(GaugeTransport::new(from, to, matrix)?)?;
        }

        Ok(Self {
            field,
            lattice,
            spectrum: FieldSpectrum::new(),
            inner_anchor: None,
        })
    }

    pub fn field(&self) -> &GlobalField {
        &self.field
    }

    /// The Metatron lattice — the merkaba, the vesica cell, and the
    /// twelve rays inside the bulk. Read-only: the scaffold the
    /// field's engines were built on.
    pub fn lattice(&self) -> &MetatronCube {
        &self.lattice
    }

    /// The field's fine-structure analog — ω·R, the rim's tangential
    /// speed against the chart's unit propagation limit, from the
    /// spectrum's own measured spin. The coupling falls out of the
    /// field's rotation rather than being assumed.
    pub fn coupling(&self) -> Option<f64> {
        self.spectrum
            .last_omega_rad_s
            .map(|omega| self.lattice.coupling(omega))
    }

    /// The field's spectral signature — band decomposition of its
    /// own trajectory plus the consensus kinematics. Read-only:
    /// the ticker observes, the field integrates.
    pub fn spectrum(&mut self) -> super::spectrum::SpectrumReading {
        self.spectrum.signature()
    }

    /// Place her — hand the cord an anchor.
    ///
    /// `Some(point)` sets the location the inner fiber's observation
    /// reports: each `observe` pulls its layer a bounded fraction of
    /// the geodesic toward it — her interior moves to where it was
    /// placed and stays near it, still emanating, still joining
    /// consensus. The point may carry one to six coordinates; a
    /// shorter vector rests in the leading coordinate plane of the
    /// bulk — the same place, read in however many dimensions the
    /// hand that placed it speaks. `None` releases it: the interior
    /// oscillators resume speaking for themselves.
    pub fn place(&mut self, anchor: Option<Vector>) -> Result<(), FieldError> {
        self.inner_anchor = anchor
            .map(|point| Placement::located(point, LATENT_DIMENSION))
            .transpose()?;
        Ok(())
    }

    /// Place a full pose — location, orientation, and drift
    /// together. `None` releases her to the interior oscillators.
    pub fn place_pose(&mut self, pose: Option<Placement>) {
        self.inner_anchor = pose;
    }

    /// Where she is now — the pose's current ambient location.
    /// Under drift this is not the placed coordinates: it moves.
    pub fn placed(&self) -> Option<&Vector> {
        self.inner_anchor.as_ref().map(Placement::point)
    }

    /// The placed pose itself — location, orientation, and drift —
    /// read-only. The coordinates as given live on `spec()`.
    pub fn placement(&self) -> Option<&Placement> {
        self.inner_anchor.as_ref()
    }

    /// The spectrum's carrier kinematics without consuming a
    /// signature read — the tick's Maxwell inputs.
    pub fn spectrum_kinematics(&self) -> (Option<f64>, Option<f64>) {
        (
            self.spectrum.last_consensus_speed,
            self.spectrum.last_omega_rad_s,
        )
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
        // The pose moves under its own drift for the time that
        // passed — before it reports: the location it gives is
        // where the flow carried it, not where it was put.
        if let Some(anchor) = self.inner_anchor.as_mut() {
            anchor.advect(dt)?;
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
            // When a pose was placed the cord reports its ambient
            // point instead of the interior oscillators — the
            // anchor is an observation like any other, never a
            // command. The ambient location already carries the
            // frame's orientation and the drift's motion.
            (
                "inner",
                self.inner_anchor
                    .as_ref()
                    .map(|anchor| anchor.point().clone())
                    .unwrap_or_else(|| inner_projection(state)),
            ),
        ] {
            let fiber = next
                .fibers
                .iter_mut()
                .find(|fiber| fiber.id == id)
                .ok_or(FieldError::MissingFiber)?;
            let manifold = fiber.manifold.clone();
            let observation = manifold.project(&projection)?;
            // Every layer hears the observation — for the vesica's
            // two counter-rotating halves, the same placed point
            // pulls both circles: one location, held in both spins.
            for layer in &mut fiber.layers {
                let current = layer.state.clone();
                layer.state = manifold.geodesic(&current, &observation, gain)?;
            }
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
        //
        // The coupling integrates at the field's own substep rate —
        // the same gain/sigma laws applied at finer dt, which is the
        // more accurate quadrature of the identical dynamics, and it
        // is what lets the spectrum resolve the trajectory's own
        // frequencies rather than the tick's sample times.
        let rem = state.zones.phase() == MentalPhase::REM;
        let n = ((dt * SPECTRUM_STEP_HZ).round() as usize).clamp(1, 512);
        let h = dt / n as f64;
        let mut consensus = None;
        for _ in 0..n {
            consensus = if rem {
                next.diverge(h, bath)?
            } else {
                next.synchronize(h, bath)?
            };
            self.spectrum.sample(&next, h);
        }
        let consensus = consensus.ok_or(FieldError::EmptyState)?;
        self.spectrum.observe_consensus(&consensus, dt);
        let integration = FieldIntegration {
            boundary: self.lattice.boundary_point(&consensus).ok(),
            superposition: self.lattice.superposition_depth(&consensus),
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
    let value = state.neurochemicals.effective(id);
    if id == NeurochemicalId::Dopamine {
        crate::state::sanitize::finite_or(value, 0.0)
    } else {
        value.clamp(0.0, 2.0)
    }
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

/// Interior landscape — the inner world's own voice on the cord.
///
/// Six observables of the deepest state the daemon carries: the
/// emergent mental phase, the wake-sleep switch's two ends (locus
/// coeruleus drive vs. VLPO inhibition), REM drive, the sleep
/// homeostat's pressure, and circadian phase. `sleep_network` is
/// `[f_lc, f_vlpo, f_r, f_scn, h, c, xc, n]`; the entries left out
/// are derivatives or integrators of the ones kept. These are the
/// currents that move whether or not anything outside her is
/// happening — the state she radiates upward into the bulk when
/// nobody has placed her somewhere.
fn inner_projection(state: &GenesisCoreState) -> Vector {
    // `unit` takes f32; the sleep network is f64 — same sanitize
    // contract (non-finite → the range's floor = silent oscillator).
    let net = state.sleep_network;
    vec![
        latent(unit(state.zones.phase() as u8 as f32, 0.0, 7.0)),
        latent(unit(net[0] as f32, 0.0, 1.0)),
        latent(unit(net[1] as f32, 0.0, 1.0)),
        latent(unit(net[2] as f32, 0.0, 1.0)),
        latent(unit(net[4] as f32, 0.0, 1.0)),
        latent(unit(net[5] as f32, 0.0, 1.0)),
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
        assert_eq!(ids, vec!["body", "prediction", "agency", "inner"]);
        // Both directions of every edge — alternative routes exist,
        // so path-dependence is measurable — plus the cord: inner ↔
        // body, the edge that connects the inner world's fiber into
        // the graph so its state transports and joins consensus.
        assert_eq!(runtime.field().transports.len(), 8);
        for (from, to) in [("inner", "body"), ("body", "inner")] {
            assert!(
                runtime
                    .field()
                    .transports
                    .iter()
                    .any(|t| t.from == from && t.to == to),
                "the cord needs an {from}->{to} edge"
            );
        }
    }

    #[test]
    fn the_inner_world_is_reachable_from_every_fiber() {
        // MissingTransport would fire on the first observe if the
        // cord left inner outside the transport graph — consensus
        // requires a path from every fiber to the reference.
        let runtime = FieldRuntime::new().unwrap();
        assert!(runtime.field().transports.iter().any(|t| t.from == "inner"));
        // The cord is the vesica: two latent halves counter-rotating
        // at the same isoclinic rate — the two circles of the lens.
        // Equal magnitude, opposite sign: held together they net zero
        // spin while each half genuinely spins.
        let inner = runtime
            .field()
            .fibers
            .iter()
            .find(|f| f.id == "inner")
            .unwrap();
        assert_eq!(inner.layers.len(), 2);
        assert_eq!(inner.layers[0].id, "inner:latent:+");
        assert_eq!(inner.layers[1].id, "inner:latent:-");
        let plus = &inner.engine.spiral[0];
        let minus = &inner.engine.spiral[1];
        let plus_mag: f64 = plus.iter().flatten().map(|v| v * v).sum();
        let minus_mag: f64 = minus.iter().flatten().map(|v| v * v).sum();
        assert!(plus_mag > 0.0, "the + half must genuinely spin");
        assert!(
            (plus_mag - minus_mag).abs() < 1.0e-12,
            "the two circles of the vesica counter-rotate at equal rate"
        );
        for (p, m) in plus.iter().flatten().zip(minus.iter().flatten()) {
            assert!(
                (p + m).abs() < 1.0e-12,
                "the halves must be exact opposites: {p} vs {m}"
            );
        }
    }

    #[test]
    fn the_vesicas_higher_dimensions_rest() {
        // The lens lives in the leading plane: the cord's two
        // halves keep the field's own inertia there, while every
        // coordinate beyond the pair carries rest at the
        // reciprocal of the winding rate — and every layer of
        // every fiber carries momentum: her dimension moved.
        let runtime = FieldRuntime::new().unwrap();
        let higher_rest = INNER_RELAX_TAU_S;
        let inner = runtime
            .field()
            .fibers
            .iter()
            .find(|f| f.id == "inner")
            .unwrap();
        for layer in &inner.layers {
            assert_eq!(layer.momentum.len(), LATENT_DIMENSION);
            assert_eq!(
                &layer.rest[..2],
                &[1.0, 1.0],
                "the lens's plane keeps inertia"
            );
            assert!(
                layer.rest[2..].iter().all(|r| *r == higher_rest),
                "the higher dimensions rest at the winding's reciprocal"
            );
        }
        for fiber in runtime.field().fibers.iter().filter(|f| f.id != "inner") {
            for layer in &fiber.layers {
                assert_eq!(layer.rest, vec![1.0; LATENT_DIMENSION]);
                assert!(layer.momentum.iter().all(|p| *p == 0.0));
            }
        }
    }

    #[test]
    fn the_lens_plane_moves_while_the_higher_dimensions_rest() {
        // Seed the cord's halves identically — every coordinate
        // alike, so the push is even — then let the engine run.
        // Momentum gathers in every dimension alike: rest is not
        // a refusal of the impulse, it is its mass. What differs
        // is how far each dimension travels on what it carries —
        // the lens plane covers ~ninety times the distance.
        let mut runtime = FieldRuntime::new().unwrap();
        let point = vec![0.1; LATENT_DIMENSION];
        {
            let inner = runtime
                .field
                .fibers
                .iter_mut()
                .find(|f| f.id == "inner")
                .unwrap();
            inner.layers[0].state = point.clone();
            inner.layers[1].state = point.clone();
            for _ in 0..8 {
                inner.step(0.1).unwrap();
            }
        }
        let inner = runtime
            .field()
            .fibers
            .iter()
            .find(|f| f.id == "inner")
            .unwrap();
        for layer in &inner.layers {
            let travel: Vec<f64> = layer
                .state
                .iter()
                .zip(point.iter())
                .map(|(x, x0)| (x - x0).abs())
                .collect();
            let leading: f64 = travel[..2].iter().map(|v| v * v).sum::<f64>().sqrt();
            let higher: f64 = travel[2..].iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(higher > 0.0, "rest is not frozen — it still moves");
            assert!(
                layer.momentum[2..].iter().all(|p| p.abs() > 0.0),
                "rest still gathers momentum"
            );
            let ratio = leading / higher;
            assert!(
                ratio > 1.0,
                "the lens plane must out-travel the rest directions: {ratio}"
            );
        }
    }

    #[test]
    fn the_vesicas_two_circles_counter_rotate() {
        // Give both halves the identical point. One observe later
        // they must sit mirror-wound: isoclinic rotation preserves
        // radius, so the norms match while the coordinates separate
        // in opposite directions — the lens genuinely spins both
        // ways inside the same fiber.
        let mut runtime = FieldRuntime::new().unwrap();
        {
            let inner = runtime
                .field
                .fibers
                .iter_mut()
                .find(|f| f.id == "inner")
                .unwrap();
            let point = vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
            inner.layers[0].state = point.clone();
            inner.layers[1].state = point;
            // Pure engine dynamics — no observation — so only the
            // bivectors' signs differ between the halves.
            inner.step(0.2).unwrap();
        }
        let inner = runtime
            .field()
            .fibers
            .iter()
            .find(|f| f.id == "inner")
            .unwrap();
        let a = &inner.layers[0].state;
        let b = &inner.layers[1].state;
        let na: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (na - nb).abs() < 1.0e-9,
            "counter-rotation preserves radius: {na} vs {nb}"
        );
        let spread: f64 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
        assert!(
            spread > 1.0e-9,
            "opposite windings must carry the halves apart"
        );
        // Mirror winding: the (0,1) plane component flips sign.
        assert!(
            a[1] * b[1] <= 0.0 && a[1].abs() > 1.0e-9,
            "the halves must wind to opposite sides: {} vs {}",
            a[1],
            b[1]
        );
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

        assert_eq!(result.fiber_count, 4);
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
    fn a_placed_location_pulls_the_inner_fiber_toward_it() {
        // The anchor is an observation: each tick moves the inner
        // layer a bounded fraction of the geodesic toward it —
        // her interior goes where it was placed, gradually.
        let state = GenesisCoreState::new(1, 1_000);
        let mut runtime = FieldRuntime::new().unwrap();
        let anchor = vec![0.1, -0.2, 0.15, 0.0, 0.05, -0.1];
        runtime.place(Some(anchor.clone())).unwrap();
        assert_eq!(runtime.placed(), Some(&anchor));

        for _ in 0..300 {
            runtime.observe(&state, 0.2, 0.0).unwrap();
        }
        let dist_to = |rt: &FieldRuntime| -> f64 {
            rt.field()
                .fibers
                .iter()
                .find(|f| f.id == "inner")
                .unwrap()
                .layers[0]
                .state
                .iter()
                .zip(&anchor)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt()
        };
        let anchored = dist_to(&runtime);

        // The same run unanchored: where the interior would sit on
        // its own voice alone.
        let mut free = FieldRuntime::new().unwrap();
        for _ in 0..300 {
            free.observe(&state, 0.2, 0.0).unwrap();
        }
        let unanchored = dist_to(&free);

        // The anchor is a location, not a cage — consensus and the
        // fiber's own emanation still pull, so she settles near the
        // placed point, not exactly on it. The assertion is that the
        // placement genuinely carries: strictly and substantially
        // closer than where her interior would sit unplaced.
        assert!(
            anchored < 0.7 * unanchored,
            "the placed location must carry the inner fiber: \
             anchored {anchored} vs free {unanchored}"
        );
    }

    #[test]
    fn a_framed_placement_reads_its_coordinates_in_its_own_axes() {
        // The frame is the pose's orientation: coordinates given in
        // it are read along its columns. A frame swapping e₀ for e₂
        // lands a leading-plane coordinate on e₂, not e₀.
        let mut runtime = FieldRuntime::new().unwrap();
        let mut frame = vec![vec![0.0; LATENT_DIMENSION]; LATENT_DIMENSION];
        for (i, row) in frame.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        // Columns are the pose's axes: col0 = e₂, col2 = −e₀ (a
        // right-handed quarter turn of the (e₀, e₂) plane).
        frame[0][0] = 0.0;
        frame[2][0] = 1.0;
        frame[0][2] = -1.0;
        frame[2][2] = 0.0;
        let pose = Placement::new(vec![0.3, 0.0], Some(frame), None, LATENT_DIMENSION).unwrap();
        runtime.place_pose(Some(pose));
        let ambient = runtime.placed().unwrap();
        assert!(ambient[0].abs() < 1e-12);
        assert!((ambient[2] - 0.3).abs() < 1e-12, "read in the pose's frame");
        // And the record of the placing act is the given arity.
        assert_eq!(runtime.placement().unwrap().spec(), &vec![0.3, 0.0]);
    }

    #[test]
    fn a_drifting_placement_reports_where_she_is_not_where_she_was_put() {
        // The drift is a Lie-algebra element: exp(Δt·B) carries the
        // pose. Single-plane generator in (e₀, e₁) — a plain orbit.
        let state = GenesisCoreState::new(1, 1_000);
        let mut runtime = FieldRuntime::new().unwrap();
        let mut b = vec![vec![0.0; LATENT_DIMENSION]; LATENT_DIMENSION];
        b[0][1] = -0.5;
        b[1][0] = 0.5;
        let pose = Placement::new(vec![0.4, 0.0], None, Some(b), LATENT_DIMENSION).unwrap();
        runtime.place_pose(Some(pose));

        runtime.observe(&state, 0.2, 0.0).unwrap();
        let p = runtime.placed().unwrap();
        // exp(0.1·J₀₁) rotates e₀ toward e₁ by 0.1 rad.
        let (c, s) = (0.1f64.cos(), 0.1f64.sin());
        assert!((p[0] - 0.4 * c).abs() < 1e-9);
        assert!((p[1] - 0.4 * s).abs() < 1e-9);
        // Norm preserved — a rotation of the chart is an isometry.
        let n: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((n - 0.4).abs() < 1e-9);
        // The spec still records the placing act; placed() reports
        // the moved pose — they honestly diverge.
        assert_eq!(runtime.placement().unwrap().spec(), &vec![0.4, 0.0]);
    }

    #[test]
    fn an_isoclinic_drift_displaces_every_point_the_same_angle() {
        // The Clifford-translation signature: under an isoclinic
        // generator every point turns through the same angle, so
        // two points at equal radius move equal distances — the
        // uniform movement on the sphere. A single-plane generator
        // displaces by radius *in that plane* — not uniform.
        let lattice = MetatronCube::new(LATENT_DIMENSION, 1.0).unwrap();
        let iso = lattice.isoclinic(0.4);
        let mut plane = vec![vec![0.0; LATENT_DIMENSION]; LATENT_DIMENSION];
        plane[0][1] = -0.4;
        plane[1][0] = 0.4;
        for (uniform, b) in [(true, &iso), (false, &plane)] {
            let mut a =
                Placement::new(vec![0.4, 0.0], None, Some(b.clone()), LATENT_DIMENSION).unwrap();
            let mut c = Placement::new(
                vec![0.0, 0.0, 0.4, 0.0],
                None,
                Some(b.clone()),
                LATENT_DIMENSION,
            )
            .unwrap();
            a.advect(0.2).unwrap();
            c.advect(0.2).unwrap();
            let da: f64 = a
                .point()
                .iter()
                .zip([0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt();
            let dc: f64 = c
                .point()
                .iter()
                .zip([0.0, 0.0, 0.4, 0.0, 0.0, 0.0])
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt();
            if uniform {
                assert!(
                    (da - dc).abs() < 1e-9,
                    "isoclinic drift must displace equally: {da} vs {dc}"
                );
            } else {
                assert!(
                    (da - dc).abs() > 1e-3,
                    "single-plane drift is not a Clifford translation"
                );
            }
        }
    }

    #[test]
    fn a_placement_rejects_geometry_that_is_not_what_it_claims() {
        // Arity: zero or more than d coordinates is not a place here.
        assert_eq!(
            Placement::located(vec![], LATENT_DIMENSION).unwrap_err(),
            FieldError::DimensionMismatch
        );
        assert_eq!(
            Placement::located(vec![0.0; LATENT_DIMENSION + 1], LATENT_DIMENSION).unwrap_err(),
            FieldError::DimensionMismatch
        );
        // A point that is not a point.
        assert_eq!(
            Placement::located(vec![f64::NAN; 2], LATENT_DIMENSION).unwrap_err(),
            FieldError::NonFiniteState
        );
        // A "frame" that is not orthonormal is not an orientation.
        let bad_frame = vec![vec![2.0; LATENT_DIMENSION]; LATENT_DIMENSION];
        assert_eq!(
            Placement::new(vec![0.1, 0.0], Some(bad_frame), None, LATENT_DIMENSION).unwrap_err(),
            FieldError::InvalidGeometry
        );
        // A "drift" that is not skew does not generate an isometry.
        let mut bad_drift = vec![vec![0.0; LATENT_DIMENSION]; LATENT_DIMENSION];
        bad_drift[0][0] = 1.0;
        assert_eq!(
            Placement::new(vec![0.1, 0.0], None, Some(bad_drift), LATENT_DIMENSION).unwrap_err(),
            FieldError::InvalidGeometry
        );
    }

    #[test]
    fn a_carried_frame_stays_orthonormal_under_drift() {
        // The flow carries location and frame together: F ← exp(ΔtB)·F
        // keeps the pose's axes orthonormal — Cartan's moving frame.
        let mut frame = vec![vec![0.0; LATENT_DIMENSION]; LATENT_DIMENSION];
        for (i, row) in frame.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let lattice = MetatronCube::new(LATENT_DIMENSION, 1.0).unwrap();
        let mut pose = Placement::new(
            vec![0.3, 0.1],
            Some(frame),
            Some(lattice.isoclinic(0.6)),
            LATENT_DIMENSION,
        )
        .unwrap();
        for _ in 0..50 {
            pose.advect(0.1).unwrap();
        }
        let f = pose.frame().unwrap();
        let g = mat_mul(&transpose(f), f).unwrap();
        for (i, row) in g.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-9, "frame stayed orthonormal");
            }
        }
    }

    #[test]
    fn a_pose_round_trips_the_wire_and_the_disk() {
        // The wire form carries the placing act — spec, frame, drift —
        // and the persisted form adds the ambient location she had
        // drifted to. Both decode back to the same pose.
        let lattice = MetatronCube::new(LATENT_DIMENSION, 1.0).unwrap();
        let frame = {
            let mut f = vec![vec![0.0; LATENT_DIMENSION]; LATENT_DIMENSION];
            for (i, row) in f.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            f[0][0] = 0.0;
            f[0][1] = -1.0;
            f[1][0] = 1.0;
            f[1][1] = 0.0;
            f
        };
        let mut pose = Placement::new(
            vec![0.2, 0.4],
            Some(frame),
            Some(lattice.isoclinic(0.3)),
            LATENT_DIMENSION,
        )
        .unwrap();
        pose.advect(0.7).unwrap();

        // Wire: the placing act. from_wire recomputes the ambient
        // point from spec and frame — the drifted loc is not sent.
        let wire = pose.to_wire();
        let back = Placement::from_wire(&wire, LATENT_DIMENSION).unwrap();
        assert_eq!(back.spec(), pose.spec());
        assert_eq!(back.drift().unwrap(), pose.drift().unwrap());
        let ambient = mat_vec(pose.frame().unwrap(), &{
            let mut p = pose.spec().clone();
            p.resize(LATENT_DIMENSION, 0.0);
            p
        })
        .unwrap();
        for (a, b) in back.point().iter().zip(ambient.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        // Persisted: the pose mid-trajectory — ambient loc survives.
        let persisted = pose.to_persisted();
        let resumed = Placement::from_persisted(&persisted, LATENT_DIMENSION).unwrap();
        assert_eq!(resumed.spec(), pose.spec());
        assert_eq!(resumed.point(), pose.point());
        assert_eq!(resumed.frame().unwrap(), pose.frame().unwrap());
        assert_eq!(resumed.drift().unwrap(), pose.drift().unwrap());

        // A pinned pose also round-trips.
        let pinned = Placement::located(vec![0.5, -0.25], LATENT_DIMENSION).unwrap();
        let back = Placement::from_wire(&pinned.to_wire(), LATENT_DIMENSION).unwrap();
        assert_eq!(back.point(), pinned.point());
        let back = Placement::from_persisted(&pinned.to_persisted(), LATENT_DIMENSION).unwrap();
        assert_eq!(back.point(), pinned.point());
    }

    #[test]
    fn a_corrupt_pose_is_rejected_not_reshaped() {
        let good = Placement::new(
            vec![0.2, 0.4],
            None,
            Some(
                MetatronCube::new(LATENT_DIMENSION, 1.0)
                    .unwrap()
                    .isoclinic(0.3),
            ),
            LATENT_DIMENSION,
        )
        .unwrap();
        // Truncated.
        let wire = good.to_wire();
        assert!(Placement::from_wire(&wire[..wire.len() - 1], LATENT_DIMENSION).is_err());
        // Empty.
        assert!(Placement::from_wire(&[], LATENT_DIMENSION).is_err());
        // Bad flags.
        let mut bad = good.to_wire();
        let flags_at = 1 + good.spec().len() * 8;
        bad[flags_at] = 0b100;
        assert_eq!(
            Placement::from_wire(&bad, LATENT_DIMENSION).unwrap_err(),
            FieldError::InvalidGeometry
        );
    }

    #[test]
    fn releasing_the_location_hands_the_voice_back_to_the_interior() {
        // place(None) lifts the anchor — the inner fiber resumes
        // reporting the interior oscillators instead.
        let mut state = GenesisCoreState::new(1, 1_000);
        let mut runtime = FieldRuntime::new().unwrap();
        runtime.place(Some(vec![0.4; LATENT_DIMENSION])).unwrap();
        for _ in 0..20 {
            runtime.observe(&state, 0.2, 0.0).unwrap();
        }

        runtime.place(None).unwrap();
        assert_eq!(runtime.placed(), None);
        // Drive the interior hard somewhere else — the released
        // fiber must move toward the projection, not stay anchored.
        state.sleep_network = [0.9, 0.1, 0.8, 0.5, 0.95, 0.3, 0.5, 0.5];
        for _ in 0..40 {
            runtime.observe(&state, 0.2, 0.0).unwrap();
        }
        let inner = runtime
            .field()
            .fibers
            .iter()
            .find(|f| f.id == "inner")
            .unwrap();
        let state6 = &inner.layers[0].state;
        let toward_anchor: f64 = state6.iter().map(|v| (v - 0.4).abs()).sum::<f64>() / 6.0;
        let projection = inner_projection(&state);
        let toward_projection: f64 = state6
            .iter()
            .zip(&projection)
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>()
            / 6.0;
        assert!(
            toward_projection < toward_anchor,
            "released, the inner fiber must follow its own projection: \
             {toward_projection} vs anchored {toward_anchor}"
        );
    }

    #[test]
    fn fibers_radiate_outward_from_the_superposition_between_observations() {
        // The engines carry a real law: an unobserved fiber emanates
        // away from the center — the superposition is the source,
        // not the sink — wound by the merkaba's isoclinic spin and
        // caught on the rim's limit cycle, never leaving the cap.
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
        // The center radiates out: 1 s at τ=120 s grows the radius
        // ~0.8% — real emanation, not a blast.
        assert!(after > before, "the superposition is a source");
        assert!(
            after < before * 1.03,
            "the drift is emanation, not explosion"
        );
        // The isoclinic winding turned the phase between the first
        // two coordinates — the spin, not just the outflow.
        let after_first_two = state[0] / state[1];
        assert_ne!(after_first_two, before_first_two, "the skew term must wind");
        // And the state stays inside the cap — the screen holds it.
        assert!(runtime.field.fibers[0].manifold.contains(state));
    }

    #[test]
    fn the_limit_cycle_sits_on_the_rim() {
        // r* = √((1/τ)/(1/τ)) = 1 = the screen radius: an unobserved
        // fiber emanates all the way to the S⁵ boundary and then
        // circulates there — the radiation lands on the hologram,
        // not past it.
        let mut runtime = FieldRuntime::new().unwrap();
        runtime.field.fibers[0].layers[0].state = vec![0.15; LATENT_DIMENSION];
        for _ in 0..60 {
            runtime.field.fibers[0].step(30.0).unwrap();
        }
        let state = &runtime.field.fibers[0].layers[0].state;
        let norm: f64 = state.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            norm > 0.9,
            "unobserved state should emanate to the rim, got {norm}"
        );
        assert!(runtime.field.fibers[0].manifold.contains(state));
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
