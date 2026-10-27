//! Axioms — the first pillar of mathematical thinking.
//!
//! An axiom system is a set of demands; a structure satisfies it
//! or it does not. On finite carriers the demands are decidable:
//! an operation is a table, and each axiom is a finite check over
//! the table's cells. This module climbs the classical ladder —
//!
//!   magma → semigroup → monoid → group → abelian group,
//!   then ring and field,
//!
//! reporting the highest rung a table reaches. The machine does
//! not believe a structure is a group because it is called one —
//! it checks every triple for associativity and every element for
//! an inverse, and the ladder rung is whatever survives.

/// A binary operation on {0,..,n-1} as an n×n table.
pub type Table = Vec<Vec<usize>>;

/// The ladder rungs, ordered weakest to strongest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rung {
    NotClosed = 0,
    Magma,
    Semigroup,
    Monoid,
    Group,
    AbelianGroup,
}

/// Closed: every table cell lands back in the carrier.
pub fn is_closed(table: &Table, n: usize) -> bool {
    table.len() == n
        && table
            .iter()
            .all(|row| row.len() == n && row.iter().all(|&v| v < n))
}

/// Associative: (a∘b)∘c = a∘(b∘c) at every triple — the cubic
/// check, decidable because n is finite.
pub fn is_associative(table: &Table, n: usize) -> bool {
    if !is_closed(table, n) {
        return false;
    }
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                if table[table[a][b]][c] != table[a][table[b][c]] {
                    return false;
                }
            }
        }
    }
    true
}

/// Identity: some e with e∘a = a∘e = a for all a. Returns the
/// identity element when one exists.
pub fn identity(table: &Table, n: usize) -> Option<usize> {
    (0..n).find(|&e| (0..n).all(|a| table[e][a] == a && table[a][e] == a))
}

/// Commutative: a∘b = b∘a everywhere.
pub fn is_commutative(table: &Table, n: usize) -> bool {
    is_closed(table, n) && (0..n).all(|a| (0..n).all(|b| table[a][b] == table[b][a]))
}

/// Every element has an inverse under the given identity:
/// for each a, some b with a∘b = b∘a = e.
pub fn all_invertible(table: &Table, n: usize, e: usize) -> bool {
    (0..n).all(|a| (0..n).any(|b| table[a][b] == e && table[b][a] == e))
}

/// The inverse map a ↦ a⁻¹ under identity e.
pub fn inverse_map(table: &Table, n: usize, e: usize) -> Vec<usize> {
    (0..n)
        .map(|a| {
            (0..n)
                .find(|&b| table[a][b] == e && table[b][a] == e)
                .unwrap_or(usize::MAX)
        })
        .collect()
}

/// Which rung does this operation reach?
pub fn classify(table: &Table, n: usize) -> Rung {
    if n == 0 || !is_closed(table, n) {
        return Rung::NotClosed;
    }
    if !is_associative(table, n) {
        return Rung::Magma;
    }
    let Some(e) = identity(table, n) else {
        return Rung::Semigroup;
    };
    if !all_invertible(table, n, e) {
        return Rung::Monoid;
    }
    if is_commutative(table, n) {
        Rung::AbelianGroup
    } else {
        Rung::Group
    }
}

/// Cancellation: a∘x = a∘y implies x = y (both sides). Holds in
/// every group; the check distinguishes it on weaker rungs.
pub fn has_cancellation(table: &Table, n: usize) -> bool {
    if !is_closed(table, n) {
        return false;
    }
    (0..n).all(|a| {
        (0..n).all(|x| {
            (0..n).all(|y| {
                (table[a][x] != table[a][y] || x == y) && (table[x][a] != table[y][a] || x == y)
            })
        })
    })
}

/// A ring: two tables — add (abelian group) and mul (monoid) —
/// plus distributivity a∘(b+c) = a∘b + a∘c on both sides.
pub fn is_ring(add: &Table, mul: &Table, n: usize) -> bool {
    if n == 0 || classify(add, n) != Rung::AbelianGroup {
        return false;
    }
    if !is_associative(mul, n) || identity(mul, n).is_none() {
        return false;
    }
    for a in 0..n {
        for b in 0..n {
            for c in 0..n {
                if mul[a][add[b][c]] != add[mul[a][b]][mul[a][c]]
                    || mul[add[b][c]][a] != add[mul[b][a]][mul[c][a]]
                {
                    return false;
                }
            }
        }
    }
    true
}

/// A field: a commutative ring whose nonzero elements invert.
pub fn is_field(add: &Table, mul: &Table, n: usize) -> bool {
    if !is_ring(add, mul, n) || !is_commutative(mul, n) {
        return false;
    }
    let Some(zero) = identity(add, n) else {
        return false;
    };
    let Some(one) = identity(mul, n) else {
        return false;
    };
    zero != one
        && (0..n)
            .filter(|&a| a != zero)
            .all(|a| (0..n).any(|b| mul[a][b] == one && mul[b][a] == one))
}

/// Build the ℤₙ addition table.
pub fn zn_add(n: usize) -> Table {
    (0..n)
        .map(|a| (0..n).map(|b| (a + b) % n).collect())
        .collect()
}

/// Build the ℤₙ multiplication table.
pub fn zn_mul(n: usize) -> Table {
    (0..n)
        .map(|a| (0..n).map(|b| (a * b) % n).collect())
        .collect()
}

/// Metric axioms on a distance matrix d[i][j]:
/// nonnegativity, identity of indiscernibles, symmetry, and the
/// triangle inequality — the minimal notion of "space."
pub fn is_metric(d: &[Vec<f64>], n: usize, tol: f64) -> bool {
    if d.len() != n || !tol.is_finite() || tol < 0.0 || d.iter().any(|row| row.len() != n) {
        return false;
    }
    for i in 0..n {
        for j in 0..n {
            let dij = d[i][j];
            if !dij.is_finite() || dij < -tol {
                return false; // negative distance
            }
            if (i == j) != (dij.abs() <= tol) {
                return false; // d(x,x)=0 and d(x,y)>0 for x!=y
            }
            if (dij - d[j][i]).abs() > tol {
                return false; // symmetry
            }
            if d[i]
                .iter()
                .enumerate()
                .any(|(k, &dik)| dij > dik + d[k][j] + tol)
            {
                return false; // triangle inequality
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder() {
        // ℤ₄ under +: abelian group.
        assert_eq!(classify(&zn_add(4), 4), Rung::AbelianGroup);
        // ℤ₄ under ×: commutative monoid — 0,1,2,3 with 1 the
        // identity but 0,2 lack inverses.
        assert_eq!(classify(&zn_mul(4), 4), Rung::Monoid);
        // Positive naturals under + have no identity — semigroup.
        let plus_no_zero: Table = (0..4)
            .map(|a| (0..4).map(|b| (a + b + 2) % 4).collect())
            .collect();
        assert_eq!(classify(&plus_no_zero, 4), Rung::AbelianGroup); // still a group — it IS ℤ₄ shifted
    }

    #[test]
    fn nonassoc_is_magma() {
        // Subtraction on ℤ₃: closed, not associative.
        let sub: Table = (0..3)
            .map(|a| (0..3).map(|b| (a + 3 - b) % 3).collect())
            .collect();
        assert_eq!(classify(&sub, 3), Rung::Magma);
    }

    #[test]
    fn rings_and_fields() {
        // ℤ₅ is a field; ℤ₄ is a ring but not a field (2 has no inverse).
        assert!(is_ring(&zn_add(5), &zn_mul(5), 5));
        assert!(is_field(&zn_add(5), &zn_mul(5), 5));
        assert!(is_ring(&zn_add(4), &zn_mul(4), 4));
        assert!(!is_field(&zn_add(4), &zn_mul(4), 4));
    }

    #[test]
    fn groups_have_cancellation() {
        assert!(has_cancellation(&zn_add(4), 4));
        assert!(!has_cancellation(&zn_mul(4), 4)); // 2·0 = 2·2
    }

    #[test]
    fn metric() {
        // Discrete metric on 3 points.
        let d: Vec<Vec<f64>> = (0..3)
            .map(|i| (0..3).map(|j| if i == j { 0.0 } else { 1.0 }).collect())
            .collect();
        assert!(is_metric(&d, 3, 1e-12));
        // Asymmetric — not a metric.
        let mut bad = d.clone();
        bad[0][1] = 0.5;
        assert!(!is_metric(&bad, 3, 1e-12));
    }
}
