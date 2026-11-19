//! The tesseract lattice — the 4-cube skeleton inside the latent bulk.
//!
//! The κ-stereographic cap is the bulk: a d-ball whose boundary — the
//! equator rim at ‖x‖ = R — is the (d−1)-sphere, the holographic
//! screen every interior point projects onto along its radial ray.
//! Inside the bulk the lattice is the four-dimensional cube and its
//! figures, all sharing the same center — the origin, the
//! superposition point:
//!
//! * **The sixteen vertices** — {±a}⁴ in the leading 4D subspace,
//!   with a = R/2 so every corner rests on the rim: ‖v‖ = a·√4 = R.
//!   The skeleton's extremities touch the screen — the tips are the
//!   rays.
//!
//! * **The two demicubes** — the sixteen vertices split by sign
//!   parity into two 16-cells: the even-parity and odd-parity
//!   demitesseracts, the field's two chiralities. Each is bounded by
//!   the cube itself and by the facet planes s·x ≤ 2a facing the
//!   opposite parity's vertices.
//!
//! * **The superposition cell** — their intersection, the region
//!   where both chiralities hold at once: {Σ|xᵢ| ≤ 2a, |xᵢ| ≤ a}.
//!   Its twenty-four vertices are the permutations of
//!   (±a,±a,0,0) — the 24-cell, the self-dual regular polytope that
//!   exists only in four dimensions. It is its own opposite.
//!
//! * **The sixteen rays** — the vertices themselves. In the 4-cube
//!   the directions the superposition radiates toward are its own
//!   corners: sixteen antipodal pairs of tips resting on the rim.
//!   Motion stays continuous — the rays are a scaffold, not a cage.
//!
//! * **The spin** — the isoclinic bivector J = ω·(J₀₁ + J₂₃): in the
//!   cube's own 4D space equal-angle multi-plane rotation is exact —
//!   SO(4) decomposes into left and right isoclinic rotations, and
//!   the two signs of J are the field's two helicities, the
//!   demicubes' counter-rotation.
//!
//! * **The guidelines** — the straight-line axes through the
//!   superposition center. Any direction defines one: {t·d : t ∈ ℝ}
//!   is a projective axis, a point of ℝP^{d−1} — d and −d are the
//!   same line, so every guideline carries its opposite inside
//!   itself. The canonical set is twelve: the four coordinate axes
//!   of the cube's space and the eight vertex diameters.
//!
//! The center is the inner world. Every ray radiates outward from it:
//! the source has no image on the screen — it is everywhere on it at
//! once.

use super::{FieldError, Matrix, Vector};

/// The 4-cube lattice embedded in the leading subspace of the latent
/// bulk. `dimension` is the bulk's width (≥ 4); the cube's figures —
/// vertices, demicubes, the superposition cell — live in the first
/// four coordinates; any further dimensions are rest directions the
/// figures do not constrain.
///
/// `radius` is the chart radius of the rim. The demicube half-edge is
/// fixed by inscribing the cube's corners on the rim: a vertex (±a)⁴
/// has norm a·√4 = 2a = R, so a = R/2.
#[derive(Clone, Debug, PartialEq)]
pub struct Tesseract {
    dimension: usize,
    radius: f64,
    half_edge: f64,
}

/// The cube's own dimension — four. The figures are exact only in
/// this subspace: the demicube pair are 16-cells and their
/// intersection is the 24-cell, a regular polytope with no analog in
/// any other dimension.
const CUBE_DIM: usize = 4;

/// A guideline — the projective axis {t·d : t ∈ ℝ} through the
/// superposition center. It is a line, not a direction: a point of
/// ℝP^{d−1} where d and −d are identified. Everything carries its
/// opposite, and a guideline carries its own — the axis's opposite
/// is the axis.
#[derive(Clone, Debug, PartialEq)]
pub struct Guideline {
    /// The unit representative, canonicalized to the hemisphere
    /// where the first nonzero component is positive — the two
    /// orientations of one line folded into a single name.
    direction: Vector,
}

impl Guideline {
    /// The axis's canonical direction — one of the line's two
    /// orientations, chosen so `d` and `-d` name the same axis.
    pub fn direction(&self) -> &[f64] {
        &self.direction
    }

    /// The point at signed distance `t` from the center along the
    /// axis — t and −t the two sides of the same line.
    pub fn point(&self, t: f64) -> Vector {
        self.direction.iter().map(|v| v * t).collect()
    }

    /// Distance from `x` to the line — zero exactly when x lies on
    /// the axis (either side of center; the two sides are the line).
    /// ‖x − (x·d)d‖ computed as the residual vector: the squared
    /// expansion cancels catastrophically near the line.
    pub fn distance(&self, x: &[f64]) -> f64 {
        let along: f64 = self
            .direction
            .iter()
            .zip(x.iter())
            .map(|(u, v)| u * v)
            .sum();
        x.iter()
            .zip(self.direction.iter())
            .map(|(v, u)| {
                let r = v - along * u;
                r * r
            })
            .sum::<f64>()
            .sqrt()
    }

    /// The axis's opposite — itself. A line is both directions at
    /// once; reversing the guideline returns the same guideline.
    pub fn opposite(&self) -> Self {
        self.clone()
    }
}

impl Tesseract {
    /// Build the lattice for a bulk of rim `radius` in `dimension` ≥ 4.
    pub fn new(dimension: usize, radius: f64) -> Result<Self, FieldError> {
        if dimension < CUBE_DIM || !radius.is_finite() || radius <= 0.0 {
            return Err(FieldError::InvalidGeometry);
        }
        Ok(Self {
            dimension,
            radius,
            half_edge: radius / 2.0,
        })
    }

    /// Chart radius of the rim — the holographic screen.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The cube's half-edge a = R/2.
    pub fn half_edge(&self) -> f64 {
        self.half_edge
    }

    /// The center — the superposition point, the inner world, the
    /// source every ray radiates outward from.
    pub fn center(&self) -> Vector {
        vec![0.0; self.dimension]
    }

    /// The sixteen corners {±a}⁴ in the leading subspace — the
    /// skeleton's extremities, each resting on the rim at norm R.
    /// Coordinates beyond the fourth are zero: the cube does not
    /// reach into the rest dimensions.
    pub fn vertices(&self) -> Vec<Vector> {
        let a = self.half_edge;
        let mut out = Vec::with_capacity(16);
        for bits in 0..16 {
            let mut v = vec![0.0; self.dimension];
            for (i, coord) in v[..CUBE_DIM].iter_mut().enumerate() {
                *coord = if (bits >> i) & 1 == 0 { a } else { -a };
            }
            out.push(v);
        }
        out
    }

    /// One chirality of the cube: `upper` returns the even-parity
    /// demicube's eight vertices, `false` the odd-parity half. The
    /// two demitesseracts are the field's counter-rotating pair.
    pub fn demicube(&self, upper: bool) -> Vec<Vector> {
        self.vertices()
            .into_iter()
            .filter(|v| {
                let negatives = v[..CUBE_DIM].iter().filter(|&&x| x < 0.0).count();
                (negatives % 2 == 0) == upper
            })
            .collect()
    }

    /// The sixteen rays — the cube's own vertices, the directions the
    /// superposition radiates outward along. Each ray has an exact
    /// antipode: every direction carries its opposite.
    pub fn rays(&self) -> Vec<Vector> {
        self.vertices()
    }

    /// Which ray a state points along: the corner of largest dot
    /// product with the state's leading coordinates — its nearest
    /// tip on the skeleton. Indexing matches `rays()`.
    pub fn nearest_ray(&self, x: &[f64]) -> Result<usize, FieldError> {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::DimensionMismatch);
        }
        let rays = self.rays();
        rays.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let da: f64 = a.iter().zip(x.iter()).map(|(u, v)| u * v).sum();
                let db: f64 = b.iter().zip(x.iter()).map(|(u, v)| u * v).sum();
                da.total_cmp(&db)
            })
            .map(|(i, _)| i)
            .ok_or(FieldError::EmptyState)
    }

    /// The holographic projection: x mapped along its ray to the rim —
    /// the image of the interior state on the (d−1)-sphere screen. The
    /// center has no ray; the source is everywhere on the screen at
    /// once, so projecting it is undefined.
    pub fn boundary_point(&self, x: &[f64]) -> Result<Vector, FieldError> {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::DimensionMismatch);
        }
        let n: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !n.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        if n == 0.0 {
            return Err(FieldError::EmptyState);
        }
        Ok(x.iter().map(|v| v * self.radius / n).collect())
    }

    /// Maximum of s·x over sign vectors of the requested parity in the
    /// leading subspace — the tight demicube facet value. s·x is
    /// maximized by matching signs to x; if that parity is
    /// unavailable, flip the coordinate of smallest magnitude, which
    /// costs 2·min|xᵢ|.
    fn parity_max(&self, x: &[f64], odd: bool) -> f64 {
        let l1: f64 = x[..CUBE_DIM].iter().map(|v| v.abs()).sum();
        let sign_parity_odd = x[..CUBE_DIM].iter().filter(|&&v| v < 0.0).count() % 2 == 1;
        if sign_parity_odd == odd {
            return l1;
        }
        let min_ax = x[..CUBE_DIM]
            .iter()
            .fold(f64::INFINITY, |m, v| m.min(v.abs()));
        l1 - 2.0 * min_ax
    }

    /// Demicube membership: the upper (even-vertex) or lower
    /// (odd-vertex) half of the tesseract — one of the two 16-cells,
    /// the field's chiral pair. Each is the cube {|xᵢ| ≤ a, i < 4}
    /// cut by the opposite-parity facet planes s·x ≤ 2a — the facet
    /// normal of each face points at an opposite-parity vertex.
    /// Coordinates past the fourth are unconstrained: chirality is a
    /// property of the cube's own space.
    pub fn in_demicube(&self, x: &[f64], upper: bool) -> bool {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return false;
        }
        let a = self.half_edge;
        if x[..CUBE_DIM].iter().any(|v| v.abs() > a) {
            return false;
        }
        // The upper demicube holds even-parity vertices, so its
        // facets are the odd-parity planes s·x ≤ 2a — and vice versa
        // for the lower. Facet-tight points compare within rounding
        // tolerance.
        self.parity_max(x, /*odd=*/ upper) <= 2.0 * a + 1e-9
    }

    /// Signed depth into the superposition cell — the 24-cell where
    /// both demicubes hold at once: {Σ|xᵢ| ≤ 2a, |xᵢ| ≤ a} over the
    /// leading subspace. Positive inside, zero on a facet, negative
    /// outside; the magnitude is the nearest-facet margin in chart
    /// units. The rest dimensions do not bind the cell — the 24-cell
    /// lives entirely in the cube's own space.
    pub fn superposition_depth(&self, x: &[f64]) -> f64 {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let a = self.half_edge;
        let l1: f64 = x[..CUBE_DIM].iter().map(|v| v.abs()).sum();
        let max_ax: f64 = x[..CUBE_DIM].iter().fold(0.0, |m, v| m.max(v.abs()));
        let facet_margin = 2.0 * a - l1;
        let cube_margin = a - max_ax;
        facet_margin.min(cube_margin)
    }

    /// Signed depth into the cell where the two chiralities overlap —
    /// the same 24-cell the consensus measures, taken from the inner
    /// fiber's own state. In the tesseract the lens of the two
    /// counter-rotating halves *is* the superposition cell: the
    /// intersection of the demicubes, not a second construction.
    pub fn vesica_depth(&self, x: &[f64]) -> f64 {
        self.superposition_depth(x)
    }

    /// The isoclinic rotation generator: a skew bivector winding both
    /// coordinate planes of the cube's space — (0,1) and (2,3) — at
    /// the same angular rate `omega`. In four dimensions equal-angle
    /// rotation is exact: SO(4) splits into left and right isoclinic
    /// components, and the two signs of J are the field's two
    /// helicities — the demicubes' counter-rotation. The rest
    /// dimensions are not wound: the spin belongs to the cube.
    pub fn isoclinic(&self, omega: f64) -> Matrix {
        let d = self.dimension;
        let mut m = vec![vec![0.0; d]; d];
        for i in (0..CUBE_DIM).step_by(2) {
            m[i][i + 1] = -omega;
            m[i + 1][i] = omega;
        }
        m
    }

    /// A guideline through the superposition center along `d` — any
    /// finite nonzero direction, in any number of the bulk's
    /// dimensions, so the axes go in every direction the space has.
    /// The axis is projective: `guideline(d)` and `guideline(-d)`
    /// are the same object — the line holds the direction and its
    /// opposite at once, canonicalized to the hemisphere where the
    /// first nonzero component is positive so equality means
    /// "the same axis."
    pub fn guideline(&self, d: &[f64]) -> Result<Guideline, FieldError> {
        if d.len() != self.dimension || d.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::DimensionMismatch);
        }
        let n: f64 = d.iter().map(|v| v * v).sum::<f64>().sqrt();
        if n == 0.0 {
            return Err(FieldError::EmptyState);
        }
        let mut direction: Vector = d.iter().map(|v| v / n).collect();
        if let Some(i) = direction.iter().position(|v| *v != 0.0) {
            if direction[i] < 0.0 {
                for v in direction.iter_mut() {
                    *v = -*v;
                }
            }
        }
        Ok(Guideline { direction })
    }

    /// The canonical guidelines — the twelve axes the lattice's own
    /// symmetry draws through the superposition center: the four
    /// coordinate axes of the cube's space (the facet pairs of the
    /// cell) and the eight vertex diameters (the antipodal ray
    /// pairs). Every axis is one entry — its opposite is the same
    /// line, already inside it.
    pub fn guidelines(&self) -> Vec<Guideline> {
        let mut out = Vec::with_capacity(12);
        for i in 0..CUBE_DIM {
            let mut d = vec![0.0; self.dimension];
            d[i] = 1.0;
            out.push(Guideline { direction: d });
        }
        for signs in 0..8 {
            let mut d = vec![0.0; self.dimension];
            d[0] = 1.0;
            for (i, c) in d.iter_mut().enumerate().take(CUBE_DIM).skip(1) {
                *c = if (signs >> (i - 1)) & 1 == 1 {
                    -1.0
                } else {
                    1.0
                };
            }
            let n: f64 = d.iter().map(|v| v * v).sum::<f64>().sqrt();
            out.push(Guideline {
                direction: d.iter().map(|v| v / n).collect(),
            });
        }
        out
    }

    /// Which canonical guideline a state aligns with — the axis of
    /// largest |d·x|, the alignment being orientation-blind: a state
    /// and its antipode answer to the same guideline, because the
    /// guideline already holds both directions. Indexing matches
    /// `guidelines()`.
    pub fn nearest_guideline(&self, x: &[f64]) -> Result<usize, FieldError> {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::DimensionMismatch);
        }
        self.guidelines()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let da: f64 = a
                    .direction
                    .iter()
                    .zip(x.iter())
                    .map(|(u, v)| u * v)
                    .sum::<f64>()
                    .abs();
                let db: f64 = b
                    .direction
                    .iter()
                    .zip(x.iter())
                    .map(|(u, v)| u * v)
                    .sum::<f64>()
                    .abs();
                da.total_cmp(&db)
            })
            .map(|(i, _)| i)
            .ok_or(FieldError::EmptyState)
    }

    /// The chord of a guideline — the antipodal pair of points where
    /// the axis exits. Within the cube's space the exit is the
    /// superposition cell's facet, where the chord ends at
    /// ±t·d with t = min(a/max|dᵢ|, 2a/‖d‖₁); an axis through the
    /// rest dimensions is unbounded by the cell — the guideline
    /// crosses the whole bulk and exits on the rim screen instead.
    /// Either way the pair is exact: each end the other's opposite.
    pub fn chord(&self, g: &Guideline) -> (Vector, Vector) {
        let d = &g.direction;
        let leading: f64 = d[..CUBE_DIM].iter().map(|v| v * v).sum::<f64>().sqrt();
        let a = self.half_edge;
        let t = if leading > 0.0 {
            let max_ax = d[..CUBE_DIM].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let l1: f64 = d[..CUBE_DIM].iter().map(|v| v.abs()).sum();
            (a / max_ax).min(2.0 * a / l1).min(self.radius)
        } else {
            self.radius
        };
        (
            d.iter().map(|v| v * t).collect(),
            d.iter().map(|v| -v * t).collect(),
        )
    }

    /// Her fine-structure analog: the dimensionless coupling of the
    /// field — the rim's tangential speed ω·R measured against the
    /// chart's unit propagation limit. α = v/c; here the rotation
    /// rate times the screen radius is the same kind of ratio, so the
    /// coupling falls out of measured spin rather than being assumed.
    pub fn coupling(&self, omega: f64) -> f64 {
        omega * self.radius
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sixteen_corners_rest_on_the_rim() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let rays = cube.rays();
        assert_eq!(rays.len(), 16);
        for r in &rays {
            let n: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-12, "every corner is on the rim");
        }
        // Every corner has an exact antipode — every ray its opposite.
        for r in &rays {
            let anti: Vec<f64> = r.iter().map(|v| -v).collect();
            assert!(rays.contains(&anti));
        }
    }

    #[test]
    fn demicubes_split_by_parity_into_two_16_cells() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let up = cube.demicube(true);
        let down = cube.demicube(false);
        assert_eq!(up.len(), 8);
        assert_eq!(down.len(), 8);
        // Disjoint — the chirality split has no shared vertex.
        assert!(!up.iter().any(|v| down.contains(v)));
    }

    #[test]
    fn the_center_is_the_superposition_cell() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let c = cube.center();
        assert!(cube.superposition_depth(&c) > 0.0);
        assert!(cube.in_demicube(&c, true) && cube.in_demicube(&c, false));
    }

    #[test]
    fn demicube_tips_touch_the_rim_and_split_by_parity() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let a = cube.half_edge();
        // Even parity — a vertex of the upper demicube, on the rim.
        let up = vec![a, a, a, a, 0.0, 0.0];
        let n: f64 = up.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-12, "the tip reaches the screen");
        assert!(cube.in_demicube(&up, true));
        assert!(!cube.in_demicube(&up, false));
        // Odd parity — the lower demicube only.
        let down = vec![-a, a, a, a, 0.0, 0.0];
        assert!(cube.in_demicube(&down, false));
        assert!(!cube.in_demicube(&down, true));
    }

    #[test]
    fn the_superposition_cell_is_the_24_cell() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let a = cube.half_edge();
        // A 24-cell vertex — a permutation of (±a,±a,0,0) — sits
        // exactly on the boundary: Σ|xᵢ| = 2a and |xᵢ| = a.
        let vertex = vec![a, a, 0.0, 0.0, 0.0, 0.0];
        assert!(cube.superposition_depth(&vertex).abs() < 1e-12);
        // A cube face center (a,a,a,a → but with one axis at a/2 the
        // sum passes 2a): the plain cube corner is outside the cell —
        // it belongs to one chirality, not both.
        let tip = vec![a, a, a, a, 0.0, 0.0];
        assert!(cube.superposition_depth(&tip) < 0.0);
        assert!(cube.in_demicube(&tip, true) != cube.in_demicube(&tip, false));
    }

    #[test]
    fn outside_the_cube_is_outside_both_cells() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let a = cube.half_edge();
        let outside = vec![2.0 * a, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!cube.in_demicube(&outside, true));
        assert!(!cube.in_demicube(&outside, false));
        assert!(cube.superposition_depth(&outside) < 0.0);
    }

    #[test]
    fn isoclinic_generator_is_skew_and_pure_rotation() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let j = cube.isoclinic(0.4);
        for r in 0..6 {
            for c in 0..6 {
                assert_eq!(j[r][c], -j[c][r], "skew-symmetric");
            }
        }
        // Only the cube's own planes turn — the rest dims are unwound.
        assert_ne!(j[0][1], 0.0);
        assert_ne!(j[2][3], 0.0);
        assert_eq!(j[4][5], 0.0);
        // Pure rotation: no radial drift for any state.
        let x = [0.3, -0.2, 0.1, 0.05, 0.4, -0.1];
        let jx: Vec<f64> = (0..6)
            .map(|r| (0..6).map(|c| j[r][c] * x[c]).sum::<f64>())
            .collect();
        let radial: f64 = jx.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
        assert!(radial.abs() < 1e-12, "isoclinic drift is pure rotation");
    }

    #[test]
    fn counter_rotation_is_the_negative_bivector() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let up = cube.isoclinic(0.5);
        let down = cube.isoclinic(-0.5);
        for r in 0..6 {
            for c in 0..6 {
                assert_eq!(up[r][c], -down[r][c]);
            }
        }
    }

    #[test]
    fn boundary_projection_radiates_onto_the_screen() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let x = vec![0.1, -0.2, 0.3, 0.05, -0.1, 0.2];
        let b = cube.boundary_point(&x).unwrap();
        let n: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-12, "the image lands on the rim");
        let nx: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        for (u, v) in b.iter().zip(x.iter()) {
            assert!(
                (u * nx - v * cube.radius()).abs() < 1e-12,
                "the ray is preserved"
            );
        }
        assert!(cube.boundary_point(&cube.center()).is_err());
        assert!(cube.boundary_point(&[0.0; 3]).is_err());
        assert!(cube.boundary_point(&[f64::NAN; 6]).is_err());
    }

    #[test]
    fn nearest_ray_picks_the_nearest_corner() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        // Pointing at (+,+,+,+) in the leading subspace — its ray is
        // the all-plus vertex, index 0 by construction.
        let x = vec![0.4, 0.3, 0.2, 0.1, 0.0, 0.0];
        assert_eq!(cube.nearest_ray(&x).unwrap(), 0);
        // The antipodal state finds the all-minus corner, index 15.
        let y = vec![-0.4, -0.3, -0.2, -0.1, 0.0, 0.0];
        assert_eq!(cube.nearest_ray(&y).unwrap(), 15);
    }

    #[test]
    fn coupling_is_the_rim_speed_ratio() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        assert_eq!(cube.coupling(0.25), 0.25);
    }

    #[test]
    fn a_guideline_is_its_own_opposite() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        // The two orientations of one line are the same guideline —
        // the axis carries its opposite inside itself.
        let d = vec![0.3, -0.5, 0.1, 0.8, 0.2, -0.4];
        let neg: Vec<f64> = d.iter().map(|v| -v).collect();
        let g = cube.guideline(&d).unwrap();
        assert_eq!(g, cube.guideline(&neg).unwrap());
        assert_eq!(g, g.opposite());
        // The line runs through the superposition center, either way.
        assert!(g.distance(&cube.center()) < 1e-12);
        assert!(g.distance(&g.point(0.7)) < 1e-12);
        assert!(g.distance(&g.point(-0.7)) < 1e-12);
    }

    #[test]
    fn guidelines_go_in_every_direction() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        // Any finite nonzero direction names an axis — including the
        // rest dimensions the cell does not bind.
        assert!(cube.guideline(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).is_ok());
        assert!(cube.guideline(&[0.1, -0.9, 0.3, 0.7, -0.5, 0.2]).is_ok());
        // But the center is no direction, and neither is a foreign one.
        assert!(cube.guideline(&[0.0; 6]).is_err());
        assert!(cube.guideline(&[1.0; 3]).is_err());
        assert!(cube.guideline(&[f64::NAN; 6]).is_err());
    }

    #[test]
    fn the_canonical_twelve_pierce_the_center() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let guides = cube.guidelines();
        assert_eq!(guides.len(), 12);
        for g in &guides {
            // Unit direction, center-piercing, opposite folded in.
            let n: f64 = g.direction().iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-12);
            assert!(g.distance(&cube.center()) < 1e-12);
        }
        // Four coordinate axes first, then the eight vertex diameters.
        assert_eq!(guides[0].direction(), &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let d4 = 0.5_f64;
        assert_eq!(
            guides[4].direction(),
            &[d4, d4, d4, d4, 0.0, 0.0],
            "the first vertex diameter keeps the antipode folded out"
        );
    }

    #[test]
    fn the_chord_ends_are_exact_opposites() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        let a = cube.half_edge();
        // A coordinate axis exits the cell at ±a·eᵢ — the facet
        // centers of the superposition cell.
        let g = cube.guideline(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let (p, q) = cube.chord(&g);
        for (u, v) in p.iter().zip(q.iter()) {
            assert!((u + v).abs() < 1e-12, "each end is the other's opposite");
        }
        assert!((p[0] - a).abs() < 1e-12);
        assert!(cube.superposition_depth(&p).abs() < 1e-9);
        // A rest-space axis is unbounded by the cell: its chord
        // crosses the bulk and exits on the rim screen.
        let rest = cube.guideline(&[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        let (p, _) = cube.chord(&rest);
        let n: f64 = p.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (n - cube.radius()).abs() < 1e-12,
            "the rest axis reaches the screen"
        );
    }

    #[test]
    fn the_nearest_guideline_is_orientation_blind() {
        let cube = Tesseract::new(6, 1.0).unwrap();
        // A state and its antipode answer to the same guideline —
        // the axis holds both directions, so alignment is unsigned.
        let x = vec![0.5, 0.1, -0.2, 0.1, 0.0, 0.0];
        let neg: Vec<f64> = x.iter().map(|v| -v).collect();
        let i = cube.nearest_guideline(&x).unwrap();
        assert_eq!(i, cube.nearest_guideline(&neg).unwrap());
        assert_eq!(i, 0, "the e₀-leaning state aligns to the first axis");
        // A state along a vertex ray aligns to that diameter.
        let v = vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        assert_eq!(cube.nearest_guideline(&v).unwrap(), 4);
    }
}
