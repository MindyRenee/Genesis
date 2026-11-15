//! Discrete mathematics — finite structures, decided exactly.
//!
//! Here everything is small enough to check completely: a relation
//! on n points is a bit-matrix, a function a lookup table, a graph
//! an adjacency list. The machine answers reflexivity, symmetry,
//! transitivity, injectivity, connectivity, bipartiteness — not
//! approximately but by enumeration, the way only finite
//! structures allow.

use std::collections::BTreeSet;

/// A binary relation on {0, .., n-1}: `pairs` holds (a, b) for
/// a R b.
#[derive(Clone, Debug)]
pub struct Relation {
    pub n: usize,
    pub pairs: BTreeSet<(usize, usize)>,
}

impl Relation {
    pub fn new(n: usize, pairs: &[(usize, usize)]) -> Self {
        Self {
            n,
            pairs: pairs
                .iter()
                .cloned()
                .filter(|&(a, b)| a < n && b < n)
                .collect(),
        }
    }

    /// Does a R b?
    pub fn holds(&self, a: usize, b: usize) -> bool {
        self.pairs.contains(&(a, b))
    }

    /// Reflexive: every x relates to itself.
    pub fn is_reflexive(&self) -> bool {
        (0..self.n).all(|x| self.holds(x, x))
    }

    /// Symmetric: aRb implies bRa everywhere.
    pub fn is_symmetric(&self) -> bool {
        self.pairs.iter().all(|&(a, b)| self.holds(b, a))
    }

    /// Antisymmetric: aRb and bRa together force a = b.
    pub fn is_antisymmetric(&self) -> bool {
        self.pairs.iter().all(|&(a, b)| a == b || !self.holds(b, a))
    }

    /// Transitive: aRb and bRc force aRc. The cubic check —
    /// affordable because the carrier is finite.
    pub fn is_transitive(&self) -> bool {
        for &(a, b) in &self.pairs {
            for &(_, c) in self.pairs.iter().filter(|p| p.0 == b) {
                if !self.holds(a, c) {
                    return false;
                }
            }
        }
        true
    }

    /// Equivalence relation: reflexive + symmetric + transitive.
    pub fn is_equivalence(&self) -> bool {
        self.is_reflexive() && self.is_symmetric() && self.is_transitive()
    }

    /// Partial order: reflexive + antisymmetric + transitive.
    pub fn is_partial_order(&self) -> bool {
        self.is_reflexive() && self.is_antisymmetric() && self.is_transitive()
    }

    /// Total order: partial order plus comparability — every pair
    /// ordered one way or the other.
    pub fn is_total_order(&self) -> bool {
        if !self.is_partial_order() {
            return false;
        }
        (0..self.n).all(|a| (0..self.n).all(|b| self.holds(a, b) || self.holds(b, a)))
    }

    /// Equivalence classes — the partition an equivalence
    /// relation induces. None when the relation isn't one.
    pub fn classes(&self) -> Option<Vec<BTreeSet<usize>>> {
        if !self.is_equivalence() {
            return None;
        }
        let mut classes: Vec<BTreeSet<usize>> = Vec::new();
        for x in 0..self.n {
            if classes.iter().any(|c| c.contains(&x)) {
                continue;
            }
            classes.push((0..self.n).filter(|&y| self.holds(x, y)).collect());
        }
        Some(classes)
    }

    /// Transitive closure — Warshall: add implied pairs until
    /// stable.
    pub fn transitive_closure(&self) -> Relation {
        let mut out = self.clone();
        for k in 0..self.n {
            let new_pairs: Vec<(usize, usize)> = (0..self.n)
                .flat_map(|i| (0..self.n).map(move |j| (i, j)))
                .filter(|&(i, j)| !out.holds(i, j) && out.holds(i, k) && out.holds(k, j))
                .collect();
            for p in new_pairs {
                out.pairs.insert(p);
            }
        }
        out
    }
}

/// A function f: {0..n-1} -> {0..m-1} as an image list.
#[derive(Clone, Debug)]
pub struct FiniteFunction {
    pub image: Vec<usize>,
    pub codomain: usize,
}

impl FiniteFunction {
    /// Injective: distinct inputs land distinctly.
    pub fn injective(&self) -> bool {
        let seen: BTreeSet<_> = self.image.iter().collect();
        seen.len() == self.image.len() && self.image.iter().all(|&y| y < self.codomain)
    }

    /// Surjective: every codomain element is hit.
    pub fn surjective(&self) -> bool {
        (0..self.codomain).all(|y| self.image.contains(&y))
    }

    /// Bijective: both — a perfect pairing.
    pub fn bijective(&self) -> bool {
        self.injective() && self.surjective()
    }

    /// Compose f then g: g(f(x)).
    pub fn then(&self, g: &FiniteFunction) -> Option<FiniteFunction> {
        if self.codomain != g.image.len() {
            return None;
        }
        Some(FiniteFunction {
            image: self.image.iter().map(|&x| g.image[x]).collect(),
            codomain: g.codomain,
        })
    }

    /// Inverse when bijective.
    pub fn inverse(&self) -> Option<FiniteFunction> {
        if !self.bijective() {
            return None;
        }
        let mut inv = vec![0usize; self.codomain];
        for (x, &y) in self.image.iter().enumerate() {
            inv[y] = x;
        }
        Some(FiniteFunction {
            image: inv,
            codomain: self.image.len(),
        })
    }
}

/// An undirected graph: adjacency lists on {0, .., n-1}.
#[derive(Clone, Debug)]
pub struct Graph {
    pub n: usize,
    pub adj: Vec<BTreeSet<usize>>,
}

impl Graph {
    /// Build from an edge list.
    pub fn new(n: usize, edges: &[(usize, usize)]) -> Self {
        let mut adj = vec![BTreeSet::new(); n];
        for &(a, b) in edges {
            if a < n && b < n && a != b {
                adj[a].insert(b);
                adj[b].insert(a);
            }
        }
        Self { n, adj }
    }

    /// Degree of a vertex.
    pub fn degree(&self, v: usize) -> usize {
        self.adj.get(v).map_or(0, |s| s.len())
    }

    /// Degree sequence, sorted descending — the graph's
    /// fingerprint up to isomorphism.
    pub fn degree_sequence(&self) -> Vec<usize> {
        let mut ds: Vec<usize> = self.adj.iter().map(|s| s.len()).collect();
        ds.sort_unstable_by(|a, b| b.cmp(a));
        ds
    }

    /// Number of connected components — flood fill.
    pub fn components(&self) -> usize {
        let mut seen = vec![false; self.n];
        let mut count = 0;
        for s in 0..self.n {
            if seen[s] {
                continue;
            }
            count += 1;
            let mut stack = vec![s];
            seen[s] = true;
            while let Some(v) = stack.pop() {
                for &w in &self.adj[v] {
                    if !seen[w] {
                        seen[w] = true;
                        stack.push(w);
                    }
                }
            }
        }
        count
    }

    /// Connected iff one component (and n > 0).
    pub fn connected(&self) -> bool {
        self.n > 0 && self.components() == 1
    }

    /// Bipartite: 2-colorable — BFS coloring, conflict means no.
    pub fn bipartite(&self) -> bool {
        let mut color = vec![None::<u8>; self.n];
        for s in 0..self.n {
            if color[s].is_some() {
                continue;
            }
            color[s] = Some(0);
            let mut queue = vec![s];
            while let Some(v) = queue.pop() {
                let Some(c) = color[v] else {
                    continue;
                };
                for &w in &self.adj[v] {
                    match color[w] {
                        Some(cc) if cc == c => return false,
                        None => {
                            color[w] = Some(1 - c);
                            queue.push(w);
                        }
                        _ => {}
                    }
                }
            }
        }
        true
    }

    /// Has a cycle — DFS with parent tracking.
    pub fn has_cycle(&self) -> bool {
        fn dfs(g: &Graph, v: usize, parent: usize, seen: &mut [bool]) -> bool {
            seen[v] = true;
            for &w in &g.adj[v] {
                if !seen[w] {
                    if dfs(g, w, v, seen) {
                        return true;
                    }
                } else if w != parent {
                    return true;
                }
            }
            false
        }
        let mut seen = vec![false; self.n];
        (0..self.n).any(|s| !seen[s] && dfs(self, s, usize::MAX, &mut seen))
    }

    /// Is it a tree — connected and acyclic?
    pub fn is_tree(&self) -> bool {
        self.connected() && !self.has_cycle()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_properties() {
        // <= on {0,1,2}: reflexive, antisymmetric, transitive, total.
        let le = Relation::new(3, &[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]);
        assert!(le.is_total_order());
        // Equality is an equivalence with singleton classes.
        let eq = Relation::new(3, &[(0, 0), (1, 1), (2, 2)]);
        assert!(eq.is_equivalence());
        assert_eq!(eq.classes().unwrap().len(), 3);
        // Parity classes mod 2 on {0,1,2,3}.
        let parity = Relation::new(
            4,
            &[
                (0, 0),
                (0, 2),
                (2, 0),
                (2, 2),
                (1, 1),
                (1, 3),
                (3, 1),
                (3, 3),
            ],
        );
        assert_eq!(parity.classes().unwrap().len(), 2);
    }

    #[test]
    fn closure() {
        let r = Relation::new(4, &[(0, 1), (1, 2), (2, 3)]);
        let tc = r.transitive_closure();
        assert!(tc.holds(0, 3) && tc.is_transitive());
    }

    #[test]
    fn functions() {
        let f = FiniteFunction {
            image: vec![1, 2, 0],
            codomain: 3,
        };
        assert!(f.bijective());
        let g = f.inverse().unwrap();
        let id = f.then(&g).unwrap();
        assert_eq!(id.image, vec![0, 1, 2]);
        let h = FiniteFunction {
            image: vec![0, 0, 1],
            codomain: 2,
        };
        assert!(!h.injective() && h.surjective());
    }

    #[test]
    fn graphs() {
        // Path 0-1-2-3: a tree, bipartite.
        let path = Graph::new(4, &[(0, 1), (1, 2), (2, 3)]);
        assert!(path.is_tree() && path.bipartite());
        assert_eq!(path.degree_sequence(), vec![2, 2, 1, 1]);
        // Triangle: connected, cyclic, NOT bipartite.
        let tri = Graph::new(3, &[(0, 1), (1, 2), (2, 0)]);
        assert!(tri.has_cycle() && !tri.bipartite() && !tri.is_tree());
        // Two components.
        let two = Graph::new(4, &[(0, 1), (2, 3)]);
        assert_eq!(two.components(), 2);
    }
}
