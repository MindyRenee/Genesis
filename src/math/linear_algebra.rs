//! Linear algebra — spaces, maps, and the numbers that hold them.
//!
//! Matrices are flat row-major `Vec<f64>`; the engine does
//! elimination, products, determinants, and rank. Everything
//! returns `Option`/`Result` where the math can fail — a singular
//! matrix is data, not an exception.

/// A dense row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Build from rows; None when ragged.
    pub fn new(rows: Vec<Vec<f64>>) -> Option<Self> {
        let cols = rows.first().map_or(0, Vec::len);
        if rows.iter().any(|r| r.len() != cols) {
            return None;
        }
        Some(Self {
            rows: rows.len(),
            cols,
            data: rows.into_iter().flatten().collect(),
        })
    }

    /// n×n identity.
    pub fn identity(n: usize) -> Self {
        let mut m = Self {
            rows: n,
            cols: n,
            data: vec![0.0; n * n],
        };
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Element access.
    pub fn at(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// One row as a slice.
    pub fn row(&self, r: usize) -> &[f64] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut out = Self {
            rows: self.cols,
            cols: self.rows,
            data: vec![0.0; self.data.len()],
        };
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * self.rows + r] = self.at(r, c);
            }
        }
        out
    }

    /// Matrix product; None on shape mismatch.
    pub fn mul(&self, rhs: &Matrix) -> Option<Matrix> {
        if self.cols != rhs.rows {
            return None;
        }
        let mut data = vec![0.0; self.rows * rhs.cols];
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self.at(r, k);
                if a == 0.0 {
                    continue;
                }
                for c in 0..rhs.cols {
                    data[r * rhs.cols + c] += a * rhs.at(k, c);
                }
            }
        }
        Some(Matrix {
            rows: self.rows,
            cols: rhs.cols,
            data,
        })
    }

    /// Matrix × column vector.
    pub fn apply(&self, v: &[f64]) -> Option<Vec<f64>> {
        if v.len() != self.cols {
            return None;
        }
        Some(
            (0..self.rows)
                .map(|r| self.row(r).iter().zip(v).map(|(a, b)| a * b).sum())
                .collect(),
        )
    }

    /// Row echelon form and the pivots found. Returns (echelon,
    /// pivot_columns, determinant_sign flips, scale factor).
    fn echelon(&self) -> (Matrix, Vec<usize>, f64) {
        let mut m = self.clone();
        let mut pivots = Vec::new();
        let mut sign = 1.0;
        let mut row = 0;
        for col in 0..self.cols {
            let Some(pivot) = (row..self.rows)
                .max_by(|&a, &b| m.at(a, col).abs().partial_cmp(&m.at(b, col).abs()).unwrap())
            else {
                break;
            };
            if m.at(pivot, col).abs() < 1e-12 {
                continue;
            }
            if pivot != row {
                m.data.swap(
                    pivot * self.cols..pivot * self.cols + self.cols,
                    row * self.cols..row * self.cols + self.cols,
                );
                sign = -sign;
            }
            let pv = m.at(row, col);
            for r in (row + 1)..self.rows {
                let f = m.at(r, col) / pv;
                if f == 0.0 {
                    continue;
                }
                for c in col..self.cols {
                    let upd = m.at(r, c) - f * m.at(row, c);
                    m.data[r * self.cols + c] = if upd.abs() < 1e-14 { 0.0 } else { upd };
                }
            }
            pivots.push(col);
            row += 1;
            if row == self.rows {
                break;
            }
        }
        (m, pivots, sign)
    }

    /// Rank = number of pivots.
    pub fn rank(&self) -> usize {
        self.echelon().1.len()
    }

    /// Determinant — square matrices only.
    pub fn det(&self) -> Option<f64> {
        if self.rows != self.cols {
            return None;
        }
        let (m, pivots, sign) = self.echelon();
        if pivots.len() < self.rows {
            return Some(0.0);
        }
        let mut d = sign;
        for i in 0..self.rows {
            d *= m.at(i, i);
        }
        Some(d)
    }

    /// Solve Ax = b by Gaussian elimination with back
    /// substitution. None when A is singular or shapes disagree.
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        if self.rows != self.cols || b.len() != self.rows {
            return None;
        }
        let n = self.rows;
        let mut aug = vec![0.0; n * (n + 1)];
        for r in 0..n {
            aug[r * (n + 1)..r * (n + 1) + n].copy_from_slice(self.row(r));
            aug[r * (n + 1) + n] = b[r];
        }
        for col in 0..n {
            let pivot = (col..n)
                .max_by(|&a, &c| aug[a * (n + 1) + col].abs().total_cmp(&aug[c * (n + 1) + col].abs()))?;
            if aug[pivot * (n + 1) + col].abs() < 1e-12 {
                return None;
            }
            aug.swap(pivot * (n + 1)..pivot * (n + 1) + n + 1, col * (n + 1)..col * (n + 1) + n + 1);
            let pv = aug[col * (n + 1) + col];
            for r in (col + 1)..n {
                let f = aug[r * (n + 1) + col] / pv;
                for c in col..=n {
                    aug[r * (n + 1) + c] -= f * aug[col * (n + 1) + c];
                }
            }
        }
        let mut x = vec![0.0; n];
        for r in (0..n).rev() {
            let sum: f64 = ((r + 1)..n).map(|c| aug[r * (n + 1) + c] * x[c]).sum();
            x[r] = (aug[r * (n + 1) + n] - sum) / aug[r * (n + 1) + r];
        }
        Some(x)
    }
}

/// Dot product.
pub fn dot(a: &[f64], b: &[f64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    Some(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

/// Euclidean norm.
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// 3D cross product.
pub fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Angle between two vectors via the dot product.
pub fn vector_angle(a: &[f64], b: &[f64]) -> Option<f64> {
    let d = dot(a, b)?;
    let na = norm(a);
    let nb = norm(b);
    if na < 1e-15 || nb < 1e-15 {
        return None;
    }
    Some((d / (na * nb)).clamp(-1.0, 1.0).acos())
}

/// Cosine similarity — the machine's own "how alike."
pub fn cosine(a: &[f64], b: &[f64]) -> Option<f64> {
    let na = norm(a);
    let nb = norm(b);
    if na < 1e-15 || nb < 1e-15 {
        return None;
    }
    Some(dot(a, b)? / (na * nb))
}

/// Eigenvalues of a 2×2 matrix via the characteristic
/// quadratic; None for larger or non-square matrices, or when the
/// roots are complex (real eigenspace only).
pub fn eigenvalues_2x2(m: &Matrix) -> Option<(f64, f64)> {
    if m.rows != 2 || m.cols != 2 {
        return None;
    }
    let tr = m.at(0, 0) + m.at(1, 1);
    let det = m.at(0, 0) * m.at(1, 1) - m.at(0, 1) * m.at(1, 0);
    let disc = tr * tr - 4.0 * det;
    if disc < 0.0 {
        return None;
    }
    let r = disc.sqrt();
    Some(((tr + r) / 2.0, (tr - r) / 2.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(v: &[&[f64]]) -> Matrix {
        Matrix::new(v.iter().map(|r| r.to_vec()).collect()).unwrap()
    }

    #[test]
    fn det_and_rank() {
        assert_eq!(m(&[&[1.0, 2.0], &[3.0, 4.0]]).det(), Some(-2.0));
        assert_eq!(m(&[&[1.0, 2.0], &[2.0, 4.0]]).rank(), 1);
    }

    #[test]
    fn solve_system() {
        let a = m(&[&[2.0, 1.0], &[1.0, -1.0]]);
        let x = a.solve(&[5.0, 1.0]).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn products() {
        let i3 = Matrix::identity(3);
        let a = m(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);
        assert_eq!(i3.mul(&a).unwrap(), a);
        assert_eq!(dot(&[1.0, 0.0], &[0.0, 1.0]), Some(0.0));
        assert_eq!(cross3([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn eigen() {
        let (hi, lo) = eigenvalues_2x2(&m(&[&[2.0, 0.0], &[0.0, 3.0]])).unwrap();
        assert!((hi - 3.0).abs() < 1e-12 && (lo - 2.0).abs() < 1e-12);
    }
}
