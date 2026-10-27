//! Geometry — the plane, computed.
//!
//! Points are plain pairs of f64. The machine's toolkit: distances
//! and orientations, segment intersections, the shoelace area,
//! Andrew's monotone-chain convex hull, triangle centers, and
//! ray-cast containment. Exact where exactness is possible,
//! optional where the math can fail.

/// A point in the plane.
pub type Point = (f64, f64);

/// Euclidean distance.
pub fn dist(a: Point, b: Point) -> f64 {
    ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt()
}

/// Squared distance — ordering without the sqrt.
pub fn dist2(a: Point, b: Point) -> f64 {
    (b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)
}

/// Orientation: the signed cross product (b-a) x (c-a).
/// Positive = counterclockwise, negative = clockwise, zero =
/// collinear. This one number decides most planar questions.
pub fn orient(a: Point, b: Point, c: Point) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Do segments ab and cd properly cross (not just touch)?
pub fn segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool {
    let o1 = orient(a, b, c);
    let o2 = orient(a, b, d);
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);
    ((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0))
        && ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))
}

/// Signed double area of a polygon — the shoelace sum.
/// Positive when vertices wind counterclockwise.
pub fn shoelace2(poly: &[Point]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    (0..n)
        .map(|i| {
            let (x1, y1) = poly[i];
            let (x2, y2) = poly[(i + 1) % n];
            x1 * y2 - x2 * y1
        })
        .sum()
}

/// Unsigned polygon area.
pub fn area(poly: &[Point]) -> f64 {
    shoelace2(poly).abs() / 2.0
}

/// Polygon perimeter.
pub fn perimeter(poly: &[Point]) -> f64 {
    let n = poly.len();
    if n < 2 {
        return 0.0;
    }
    (0..n).map(|i| dist(poly[i], poly[(i + 1) % n])).sum()
}

/// Point-in-polygon by ray casting — count crossings of a ray
/// to +infinity; odd means inside.
pub fn contains(poly: &[Point], p: Point) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    for i in 0..n {
        let (xi, yi) = poly[i];
        let (xj, yj) = poly[(i + 1) % n];
        if (yi > p.1) != (yj > p.1) && p.0 < (xj - xi) * (p.1 - yi) / (yj - yi) + xi {
            inside = !inside;
        }
    }
    inside
}

/// Convex hull — Andrew's monotone chain, O(n log n).
/// Returns hull vertices counterclockwise, no repeats.
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    if points.len() <= 1 {
        return points.to_vec();
    }
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    pts.dedup_by(|a, b| (a.1 - b.1).abs() < 1e-15 && (a.0 - b.0).abs() < 1e-15);
    if pts.len() <= 2 {
        return pts;
    }
    let mut hull: Vec<Point> = Vec::with_capacity(pts.len() + 1);
    // Lower then upper chain.
    for _ in 0..2 {
        let start = hull.len();
        for &p in &pts {
            while hull.len() >= start + 2
                && orient(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
        pts.reverse();
    }
    hull
}

/// Centroid of a polygon — area-weighted, not vertex-averaged.
/// None when the polygon degenerates to zero area.
pub fn centroid(poly: &[Point]) -> Option<Point> {
    let n = poly.len();
    if n < 3 {
        return None;
    }
    let a2 = shoelace2(poly);
    if a2.abs() < 1e-15 {
        return None;
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        let (x1, y1) = poly[i];
        let (x2, y2) = poly[(i + 1) % n];
        let w = x1 * y2 - x2 * y1;
        cx += (x1 + x2) * w;
        cy += (y1 + y2) * w;
    }
    Some((cx / (3.0 * a2), cy / (3.0 * a2)))
}

/// Vertex centroid of a point set — the plain average.
pub fn mean_point(points: &[Point]) -> Option<Point> {
    if points.is_empty() {
        return None;
    }
    let n = points.len() as f64;
    Some((
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
    ))
}

/// Circumcenter of a triangle — where the perpendicular
/// bisectors meet; the circle through all three vertices.
/// None when the points are collinear.
pub fn circumcenter(a: Point, b: Point, c: Point) -> Option<Point> {
    let d = 2.0 * orient(a, b, c);
    if d.abs() < 1e-15 {
        return None;
    }
    let (a2, b2, c2) = (
        a.0 * a.0 + a.1 * a.1,
        b.0 * b.0 + b.1 * b.1,
        c.0 * c.0 + c.1 * c.1,
    );
    Some((
        (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d,
        (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d,
    ))
}

/// Closest pair among points — brute force, exact.
pub fn closest_pair(points: &[Point]) -> Option<(Point, Point, f64)> {
    let mut best: Option<(Point, Point, f64)> = None;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let d = dist(points[i], points[j]);
            if best.is_none_or(|b| d < b.2) {
                best = Some((points[i], points[j], d));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shoelace_and_orient() {
        let tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)];
        assert_eq!(shoelace2(&tri), 12.0); // CCW positive
        assert_eq!(area(&tri), 6.0);
        assert!(orient(tri[0], tri[1], tri[2]) > 0.0);
        assert!(segments_cross(
            (0.0, 0.0),
            (2.0, 2.0),
            (0.0, 2.0),
            (2.0, 0.0)
        ));
        assert!(!segments_cross(
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 1.0)
        ));
    }

    #[test]
    fn containment() {
        let sq = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];
        assert!(contains(&sq, (2.0, 2.0)));
        assert!(!contains(&sq, (5.0, 2.0)));
    }

    #[test]
    fn hull() {
        let pts = [
            (0.0, 0.0),
            (4.0, 0.0),
            (4.0, 4.0),
            (0.0, 4.0),
            (2.0, 2.0),
            (1.0, 1.0), // interior points drop out
        ];
        let h = convex_hull(&pts);
        assert_eq!(h.len(), 4);
    }

    #[test]
    fn centers() {
        let tri = [(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)];
        let (cx, cy) = centroid(&tri).unwrap();
        assert!((cx - 4.0 / 3.0).abs() < 1e-12 && (cy - 1.0).abs() < 1e-12);
        let (ox, oy) = circumcenter(tri[0], tri[1], tri[2]).unwrap();
        assert!((ox - 2.0).abs() < 1e-12 && (oy - 1.5).abs() < 1e-12);
    }
}
