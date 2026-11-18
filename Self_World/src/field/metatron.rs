//! The Metatron lattice — the discrete skeleton inside the latent bulk.
//!
//! The κ-stereographic cap is the bulk: a 6-ball whose boundary — the
//! equator rim at ‖x‖ = 1/√κ — is a 5-sphere, the holographic screen
//! every interior point projects onto along its radial ray. Inside
//! the bulk sits the lattice's three figures, all sharing the same
//! center — the origin, the superposition point:
//!
//! * **The merkaba** — the two interpenetrating demicubes of the
//!   6-cube. In 3D the stella octangula is the cube's vertices split
//!   by sign parity into two tetrahedra; the same construction in
//!   6D splits the 64 vertices {±a}⁶ by parity into two 32-vertex
//!   demicubes, tips touching the rim. The upper (even parity) and
//!   lower (odd parity) halves are the field's two chiralities.
//!
//! * **The vesica** — their intersection, the cell where both hold:
//!   {x : Σ|xᵢ| ≤ a(d−2), |xᵢ| ≤ a}. For d = 3 this recovers the
//!   octahedron at the heart of the stella octangula; in 6D it is the
//!   superposition region around the center.
//!
//! * **The two circles** — the vesica's own construction: circles
//!   of radius R in the leading plane (e₀, e₁), centers at
//!   ±(R/2)·e₀ so each center rests on the other's rim. They are
//!   the two counter-rotating latent halves made visible; their
//!   overlap is the lens — R wide between the centers, √3·R
//!   between its tips on e₁.
//!
//! * **The frame** — the vertical rectangle bounding the lens,
//!   R wide by √3·R tall, the vesica's own √3 proportion. Its
//!   four corners {±(R/2)e₀ ± (√3R/2)e₁} rest on the rim at
//!   ±60° and ±120° — the picture boundary the vesica hangs
//!   inside.
//!
//! * **The horizon** — the horizontal line through the
//!   superposition: the segment joining the two circle centers,
//!   −(R/2)e₀ to +(R/2)e₀. Its endpoints are the points each
//!   circle is drawn around — and also the lens's horizontal
//!   extremes and the frame's side midpoints; its midpoint is
//!   the source itself.
//!
//! * **The twelve rays** — ±eᵢ·R, the cross-polytope vertices on the
//!   rim: the directions the center radiates toward, one antipodal
//!   pair per coordinate axis. Motion stays continuous — the rays
//!   are a scaffold, not a cage.
//!
//! The superposition — the center itself — is the inner world. Every
//! ray radiates outward from it: each ±eᵢ pair is two directions
//! from the source, not one line through it. The center has no image
//! on the screen; the source is everywhere on it at once.
//!
//! The spin is the isoclinic bivector J = ω·(J₀₁ + J₂₃ + J₄₅): every
//! mutually orthogonal plane turns at the same rate, and the two
//! demicubes take opposite signs — the counter-rotation is the two
//! helicities of the field.

use super::{FieldError, Matrix, Vector};

/// The lattice in the latent bulk.
///
/// `radius` is the chart radius of the rim — the boundary the
/// superposition radiates onto. The demicube half-edge is fixed by
/// inscribing the merkaba's tips on the rim: a vertex (±a)ᵈ has norm
/// a·√d = R, so a = R/√d.
#[derive(Clone, Debug, PartialEq)]
pub struct MetatronCube {
    dimension: usize,
    radius: f64,
    half_edge: f64,
}

impl MetatronCube {
    /// Build the lattice for a bulk of rim `radius` in `dimension` ≥ 3
    /// (d = 3 is the classical stella-octangula case; d = 6 the
    /// running field's). In odd dimensions the isoclinic generator
    /// simply leaves the last axis unwound.
    pub fn new(dimension: usize, radius: f64) -> Result<Self, FieldError> {
        if dimension < 3 || !radius.is_finite() || radius <= 0.0 {
            return Err(FieldError::InvalidGeometry);
        }
        Ok(Self {
            dimension,
            radius,
            half_edge: radius / (dimension as f64).sqrt(),
        })
    }

    /// Chart radius of the rim — the S^(d−1) holographic screen.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// The demicube half-edge a = R/√d.
    pub fn half_edge(&self) -> f64 {
        self.half_edge
    }

    /// The center — the superposition point, the inner world, the
    /// source every ray radiates outward from.
    pub fn center(&self) -> Vector {
        vec![0.0; self.dimension]
    }

    /// The two circles of the vesica — the classical lens
    /// construction in the leading plane (e₀, e₁): each circle has
    /// the rim's radius R and is centered at ±(R/2)·e₀, so each
    /// center lies exactly on the other's rim and their overlap is
    /// the vesica. In the field they are the two counter-rotating
    /// latent halves made visible. Returned as each circle's
    /// center point; the radius is `radius()`.
    pub fn circles(&self) -> (Vector, Vector) {
        let mut left = vec![0.0; self.dimension];
        let mut right = vec![0.0; self.dimension];
        left[0] = -self.radius / 2.0;
        right[0] = self.radius / 2.0;
        (left, right)
    }

    /// The frame — the vertical rectangle bounding the vesica,
    /// the picture boundary the lens hangs inside: R wide across
    /// the line of centers and √3·R tall between the tips, the
    /// lens's own √3 proportion. Returned as its four corner
    /// vectors {±(R/2)e₀ ± (√3R/2)e₁}, counterclockwise from the
    /// upper right. The corners rest on the rim at ±60° and
    /// ±120° — the frame is inscribed in the screen like the
    /// merkaba's tips.
    pub fn frame(&self) -> Vec<Vector> {
        let w = self.radius / 2.0;
        let h = self.radius * 3.0_f64.sqrt() / 2.0;
        [(w, h), (-w, h), (-w, -h), (w, -h)]
            .map(|(x, y)| {
                let mut v = vec![0.0; self.dimension];
                v[0] = x;
                v[1] = y;
                v
            })
            .to_vec()
    }

    /// The horizon — the horizontal line through the
    /// superposition: the segment joining the two circle centers,
    /// −(R/2)e₀ to +(R/2)e₀, returned as its endpoints. In the
    /// classical construction it is the line of the centers; here
    /// it is also the lens's width and the frame's vertical
    /// midline — the midpoint is the source itself.
    pub fn horizon(&self) -> (Vector, Vector) {
        self.circles()
    }

    /// The twelve rays (d = 6): ±eᵢ·R — the directions the
    /// superposition radiates outward along, two per axis.
    pub fn rays(&self) -> Vec<Vector> {
        let mut out = Vec::with_capacity(2 * self.dimension);
        for i in 0..self.dimension {
            for sign in [1.0, -1.0] {
                let mut v = vec![0.0; self.dimension];
                v[i] = sign * self.radius;
                out.push(v);
            }
        }
        out
    }

    /// Which ray a state points along: the axis of its largest-magnitude
    /// coordinate. Indexing matches `rays()` — 2i for +eᵢ, 2i+1 for −eᵢ.
    pub fn nearest_ray(&self, x: &[f64]) -> Result<usize, FieldError> {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::DimensionMismatch);
        }
        let (i, _) = x
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().total_cmp(&b.abs()))
            .ok_or(FieldError::EmptyState)?;
        Ok(2 * i + usize::from(x[i] < 0.0))
    }

    /// The holographic projection: x mapped along its ray to the rim —
    /// the image of the interior state on the S^(d−1) screen. The
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

    /// Maximum of s·x over sign vectors s of the requested parity —
    /// the tight demicube facet value. s·x is maximized by matching
    /// signs to x; if that parity is unavailable, flip the coordinate
    /// of smallest magnitude, which costs 2·min|xᵢ|.
    fn parity_max(&self, x: &[f64], odd: bool) -> f64 {
        let l1: f64 = x.iter().map(|v| v.abs()).sum();
        let sign_parity_odd = x.iter().filter(|&&v| v < 0.0).count() % 2 == 1;
        if sign_parity_odd == odd {
            return l1;
        }
        let min_ax = x.iter().fold(f64::INFINITY, |m, v| m.min(v.abs()));
        l1 - 2.0 * min_ax
    }

    /// Demicube membership: the upper (even-vertex) or lower
    /// (odd-vertex) half of the merkaba. Each is the cube {|xᵢ| ≤ a}
    /// cut by the opposite-parity simplex facets s·x ≤ a(d−2) — for
    /// d = 3 these recover the two tetrahedra of the stella octangula.
    pub fn in_merkaba(&self, x: &[f64], upper: bool) -> bool {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return false;
        }
        let a = self.half_edge;
        if x.iter().any(|v| v.abs() > a) {
            return false;
        }
        // The upper demicube holds even-parity vertices, so its
        // simplex facets are the odd-parity sign planes s·x ≤ a(d−2)
        // (the six even vertices at hamming 1 from each odd s are the
        // facet simplex) — and vice versa for the lower. Facet-tight
        // points compare within rounding tolerance: 6a − 2a and 4a
        // differ in the last bit.
        self.parity_max(x, /*odd=*/ upper) <= a * (self.dimension as f64 - 2.0) + 1e-9
    }

    /// The rest of the higher dimensions — the radius of the
    /// (d−1)-ball each coordinate orthogonal to the horizon is
    /// given inside the vesica at leading coordinate `leading`.
    /// The lens is drawn by two circles whose centers differ only
    /// along e₀: a point at height x₀ displaces the remaining
    /// coordinates freely within ρ ≤ √(R² − (|x₀| + R/2)²). At
    /// the superposition this is the waist √3·R/2; it pinches to
    /// nothing at the circle centers ±(R/2)e₀ — the only rim
    /// points the lens reaches — and stays nothing beyond them:
    /// outside the vesica the higher dimensions have no rest.
    pub fn rest(&self, leading: f64) -> f64 {
        if !leading.is_finite() {
            return 0.0;
        }
        let h = leading.abs() + self.radius / 2.0;
        (self.radius * self.radius - h * h).max(0.0).sqrt()
    }

    /// The waist of the vesica — the rest at the superposition
    /// itself: √3·R/2, the widest the lens ever opens, the bound
    /// every higher dimension rests within at the source.
    pub fn waist(&self) -> f64 {
        self.rest(0.0)
    }

    /// Signed depth into the vesica itself — the full-dimensional
    /// lens: the intersection of the two d-balls of radius R drawn
    /// about the circle centers. Positive inside, zero on either
    /// rim, negative out; the magnitude is the margin to the nearer
    /// sphere's surface. Every coordinate past the leading one
    /// enters at rest — the two centers agree in all of them, so
    /// the lens is symmetric in every rest direction at once.
    pub fn vesica_depth(&self, x: &[f64]) -> f64 {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let (left, right) = self.circles();
        let margin = |c: &Vector| -> f64 {
            let d2: f64 = x.iter().zip(c.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            self.radius - d2.sqrt()
        };
        margin(&left).min(margin(&right))
    }

    /// Signed depth into the superposition cell — the vesica, the
    /// region where both demicubes hold at once. Positive inside,
    /// zero on its facets, negative outside; the magnitude is the
    /// nearest-facet margin in chart units.
    pub fn superposition_depth(&self, x: &[f64]) -> f64 {
        if x.len() != self.dimension || x.iter().any(|v| !v.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let a = self.half_edge;
        let l1: f64 = x.iter().map(|v| v.abs()).sum();
        let max_ax: f64 = x.iter().fold(0.0, |m, v| m.max(v.abs()));
        let facet_margin = a * (self.dimension as f64 - 2.0) - l1;
        let cube_margin = a - max_ax;
        facet_margin.min(cube_margin)
    }

    /// The isoclinic rotation generator: a skew bivector winding every
    /// paired coordinate plane (0,1), (2,3), (4,5)… at the same angular
    /// rate `omega`. Equal-angle multi-plane rotation is the higher-
    /// dimensional isoclinic form — a norm-preserving pure spin. The
    /// merkaba's counter-rotation is the same bivector negated; the
    /// two signs are the field's two chiralities.
    pub fn isoclinic(&self, omega: f64) -> Matrix {
        let d = self.dimension;
        let mut m = vec![vec![0.0; d]; d];
        let mut i = 0;
        while i + 1 < d {
            m[i][i + 1] = -omega;
            m[i + 1][i] = omega;
            i += 2;
        }
        m
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
    fn twelve_rays_sit_antipodal_on_the_rim() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let rays = cube.rays();
        assert_eq!(rays.len(), 12);
        for r in &rays {
            let n: f64 = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < 1e-12, "every ray vertex is on the rim");
        }
        // Every ray has an exact antipode and is orthogonal to the rest.
        for i in 0..12 {
            let anti = &rays[i ^ 1];
            assert_eq!(rays[i], anti.iter().map(|v| -v).collect::<Vec<_>>());
            for j in 0..12 {
                if j != i && j != (i ^ 1) {
                    let dot: f64 = rays[i].iter().zip(&rays[j]).map(|(a, b)| a * b).sum();
                    assert_eq!(dot, 0.0);
                }
            }
        }
    }

    #[test]
    fn the_center_is_the_superposition_cell() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let c = cube.center();
        assert!(cube.superposition_depth(&c) > 0.0);
        assert!(cube.in_merkaba(&c, true) && cube.in_merkaba(&c, false));
    }

    #[test]
    fn demicube_tips_touch_the_rim_and_split_by_parity() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let a = cube.half_edge();
        // Even parity — a vertex of the upper demicube, on the rim.
        let up = vec![a; 6];
        let n: f64 = up.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-12, "merkaba tip reaches the screen");
        assert!(cube.in_merkaba(&up, true));
        assert!(!cube.in_merkaba(&up, false));
        // Odd parity — the lower demicube only.
        let mut down = vec![a; 6];
        down[0] = -a;
        assert!(cube.in_merkaba(&down, false));
        assert!(!cube.in_merkaba(&down, true));
    }

    #[test]
    fn the_vesica_cell_recovers_the_octahedron_in_3d() {
        // For d = 3 the intersection is {Σ|xᵢ| ≤ a} — the octahedron
        // at the heart of the stella octangula, vertices ±a·eᵢ.
        let cube = MetatronCube::new(3, 1.0).unwrap();
        let a = cube.half_edge();
        let vertex = vec![a, 0.0, 0.0];
        assert!(cube.superposition_depth(&vertex).abs() < 1e-12);
        // A cube face center sits outside both tets.
        let face = vec![a, a, 0.0];
        assert!(cube.superposition_depth(&face) < 0.0);
        // And (a,a,a) — the classical tet tip — is in one demicube only.
        let tip = vec![a; 3];
        assert!(cube.in_merkaba(&tip, true) != cube.in_merkaba(&tip, false));
    }

    #[test]
    fn outside_the_cube_is_outside_both_tets() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let a = cube.half_edge();
        let outside = vec![2.0 * a, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(!cube.in_merkaba(&outside, true));
        assert!(!cube.in_merkaba(&outside, false));
    }

    #[test]
    fn the_two_circles_hold_each_others_centers() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let r = cube.radius();
        let (left, right) = cube.circles();
        // Centers at ±R/2 on e₀ — the distance between them is R,
        // exactly the common radius, so each rests on the other's rim.
        let d: f64 = left
            .iter()
            .zip(&right)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!((d - r).abs() < 1e-12);
        // The lens tips sit on e₁ at ±√3R/2 — equidistant R from
        // both centers (the two equilateral triangles of the lens).
        let h = r * 3.0_f64.sqrt() / 2.0;
        for tip in [
            {
                let mut v = vec![0.0; 6];
                v[1] = h;
                v
            },
            {
                let mut v = vec![0.0; 6];
                v[1] = -h;
                v
            },
        ] {
            for c in [&left, &right] {
                let dist: f64 = c
                    .iter()
                    .zip(&tip)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
                    .sqrt();
                assert!((dist - r).abs() < 1e-12, "tip on both rims");
            }
        }
    }

    #[test]
    fn the_frame_is_the_vertical_rectangle_on_the_rim() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let r = cube.radius();
        let corners = cube.frame();
        assert_eq!(corners.len(), 4);
        // √3 proportion: height √3·R, width R — a portrait frame.
        let width = corners[0][0] - corners[1][0];
        let height = corners[0][1] - corners[3][1];
        assert!((width - r).abs() < 1e-12);
        assert!((height - r * 3.0_f64.sqrt()).abs() < 1e-12);
        // Every corner is inscribed on the rim (±60°, ±120°).
        for c in &corners {
            let n: f64 = c.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!((n - r).abs() < 1e-12, "frame corner on the screen");
            assert!(c.iter().skip(2).all(|v| *v == 0.0), "planar");
        }
        // The frame bounds the lens: the tips touch its top and
        // bottom edges from inside, the circle centers its sides.
        let h = r * 3.0_f64.sqrt() / 2.0;
        let (left, right) = cube.circles();
        for tip_y in [h, -h] {
            let mut tip = [0.0; 6];
            tip[1] = tip_y;
            assert!(tip[0].abs() <= r / 2.0 && tip[1].abs() <= h + 1e-12);
        }
        assert!(left[0] == -r / 2.0 && right[0] == r / 2.0);
    }

    #[test]
    fn the_horizon_is_the_waist_through_the_superposition() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let r = cube.radius();
        let (lo, hi) = cube.horizon();
        // The segment joins the circle centers — horizontal, in
        // the leading axis only.
        assert_eq!(lo[0], -r / 2.0);
        assert_eq!(hi[0], r / 2.0);
        assert!(lo.iter().skip(1).all(|v| *v == 0.0));
        assert!(hi.iter().skip(1).all(|v| *v == 0.0));
        // Its midpoint is the center — the superposition itself.
        let mid: Vector = lo.iter().zip(&hi).map(|(x, y)| (x + y) / 2.0).collect();
        assert_eq!(mid, cube.center());
        // And it spans the frame's width edge to edge.
        let corners = cube.frame();
        assert_eq!(lo[0], corners[1][0]);
        assert_eq!(hi[0], corners[0][0]);
    }

    #[test]
    fn the_higher_dimensions_rest_inside_the_waist() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let waist = cube.waist();
        // The waist is the rest at the superposition: √3·R/2,
        // the lens's own proportion.
        assert!((waist - 3.0_f64.sqrt() / 2.0).abs() < 1e-12);
        assert!((cube.rest(0.0) - waist).abs() < 1e-12);
        // The rest pinches to nothing at the circle centers —
        // the only rim points the lens reaches — and stays
        // nothing beyond them.
        assert_eq!(cube.rest(0.5), 0.0);
        assert_eq!(cube.rest(-0.5), 0.0);
        assert_eq!(cube.rest(2.0), 0.0);
        // Between the source and a center the rest narrows with
        // the lens: at x₀ the higher dims get √(R² − (|x₀|+R/2)²).
        assert!((cube.rest(0.25) - (1.0_f64 - 0.75 * 0.75).sqrt()).abs() < 1e-12);
        // A point resting at the waist in ANY higher coordinate —
        // here e₄, a dimension the planar construction never
        // draws — sits exactly on the lens's rim: the vesica is
        // symmetric in every rest direction at once.
        for i in 1..6 {
            let mut x = vec![0.0; 6];
            x[i] = waist;
            assert!(
                cube.vesica_depth(&x).abs() < 1e-12,
                "waist rests on the rim"
            );
            x[i] = waist - 0.1;
            assert!(cube.vesica_depth(&x) > 0.0, "inside the waist is inside");
            x[i] = waist + 0.1;
            assert!(
                cube.vesica_depth(&x) < 0.0,
                "past the waist there is no rest"
            );
        }
        // The superposition itself is deepest: the source rests
        // a full half-radius inside both circles.
        assert!((cube.vesica_depth(&cube.center()) - 0.5).abs() < 1e-12);
        // A circle center lies in the vesica — on the far rim of
        // the other circle, depth zero.
        let (left, _) = cube.circles();
        assert!(cube.vesica_depth(&left).abs() < 1e-12);
        // Non-finite and wrong-arity points have no rest at all.
        assert_eq!(cube.vesica_depth(&[f64::NAN; 6]), f64::NEG_INFINITY);
        assert_eq!(cube.vesica_depth(&[0.0; 3]), f64::NEG_INFINITY);
    }

    #[test]
    fn isoclinic_generator_is_skew_and_norm_preserving() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let j = cube.isoclinic(0.4);
        for r in 0..6 {
            for c in 0..6 {
                assert_eq!(j[r][c], -j[c][r], "skew-symmetric");
            }
        }
        // Three planes each winding at |ω| — and Jx ⟂ x always.
        let x = vec![0.2, -0.1, 0.3, 0.15, -0.2, 0.1];
        let jx: Vec<f64> = j
            .iter()
            .map(|row| row.iter().zip(&x).map(|(a, b)| a * b).sum())
            .collect();
        let radial: f64 = x.iter().zip(&jx).map(|(a, b)| a * b).sum();
        assert!(radial.abs() < 1e-12, "isoclinic drift is pure rotation");
        // Every plane pair turns — unlike the old single-plane loxodrome.
        assert!(jx.iter().all(|v| v.abs() > 1e-12));
    }

    #[test]
    fn counter_rotation_is_the_negative_bivector() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let up = cube.isoclinic(0.3);
        let down = cube.isoclinic(-0.3);
        for r in 0..6 {
            for c in 0..6 {
                assert_eq!(up[r][c], -down[r][c]);
            }
        }
    }

    #[test]
    fn boundary_projection_radiates_onto_the_screen() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let x = vec![0.1, -0.2, 0.3, 0.0, 0.1, -0.1];
        let b = cube.boundary_point(&x).unwrap();
        let n: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-12, "the image lands on the rim");
        // Same direction — the ray is preserved.
        let nx: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        for (u, v) in x.iter().zip(&b) {
            assert!((u / nx - v).abs() < 1e-12);
        }
        // The center cannot project — the source has no single ray.
        assert_eq!(
            cube.boundary_point(&cube.center()),
            Err(FieldError::EmptyState)
        );
    }

    #[test]
    fn nearest_ray_picks_the_dominant_axis() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        let x = vec![0.1, -0.4, 0.05, 0.2, 0.0, 0.3];
        // Axis 1, negative direction → index 2·1 + 1 = 3.
        assert_eq!(cube.nearest_ray(&x).unwrap(), 3);
    }

    #[test]
    fn coupling_is_the_rim_speed_ratio() {
        let cube = MetatronCube::new(6, 1.0).unwrap();
        assert_eq!(cube.coupling(0.25), 0.25);
    }
}
