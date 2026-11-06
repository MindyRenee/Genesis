//! Gamma-band phase binding — the synchrony layer that forms proto-objects.
//!
//! The brain does not *compute* grouping; it *resonates* it. In V1/V2,
//! features that belong to the same object fire in gamma phase-lock while
//! features of different objects desynchronize (binding-by-synchrony,
//! Singer & von der Malsburg). A proto-object is literally a gamma-coherent
//! assembly: an internal token the system can attend, track, and reason
//! about before any recognition happens.
//!
//! The math is Kuramoto: each visible cell carries a phase oscillator whose
//! natural frequency is seeded by colour; coupling pulls phase-locked
//! neighbours together (contiguity, the dominant Gestalt term) and adds a
//! weak long-range pull between same-colour cells inside a receptive window
//! (similarity, the second term). After integration, cells sharing phase
//! and linkage are emitted as [`ProtoObject`] assemblies — same-colour
//! components whose collective phases locked get bound even when not
//! adjacent, which is the Gestalt capability a plain flood-fill lacks.
//!
//! Discipline: this layer forms objects; it never recognizes them. And a
//! frame it cannot bind reports absence — an empty scene, not a fabricated
//! object.

const TWO_PI: f64 = std::f64::consts::TAU;

/// One bound assembly — a gamma-coherent proto-object.
#[derive(Debug, Clone)]
pub struct ProtoObject {
    /// Palette index (0–15) the assembly was bound under.
    pub color: u8,
    /// Number of member cells.
    pub size: usize,
    /// Member cells as flat indices (row * width + col).
    pub cells: Vec<usize>,
    /// Bounding box: (min_row, min_col, max_row, max_col).
    pub bbox: (usize, usize, usize, usize),
    /// Centroid (row, col).
    pub centroid: (f64, f64),
    /// Mean phase the assembly locked to — its binding signature.
    pub phase: f64,
    /// Mean-field coherence 0..1 of the member oscillators.
    pub coherence: f64,
}

/// The V2 grouping field: Kuramoto phases bound to an optic frame.
pub struct GammaField {
    width: usize,
    height: usize,
    /// Per-cell phase after integration; `f64::NAN` where background.
    phases: Vec<f64>,
    /// Component id per cell; `usize::MAX` where background.
    labels: Vec<usize>,
}

impl GammaField {
    /// Bind a colour grid into proto-object assemblies.
    ///
    /// `grid` is `height * width` palette indices; `background` is the
    /// palette index treated as ground (no oscillators form there).
    /// `steps` is the Kuramoto integration horizon — ~40 iterations
    /// settles most fields.
    pub fn bind(grid: &[u8], width: usize, height: usize, background: u8, steps: usize) -> Vec<ProtoObject> {
        let n = width * height;
        if grid.len() != n || n == 0 {
            return Vec::new();
        }
        let mut field = Self::segment(grid, width, height, background);
        field.integrate(grid, steps);
        field.assemblies(grid)
    }

    /// First pass — contiguity: label 4-connected same-colour components.
    /// This is the contour-integration stage; grouping is seeded by
    /// adjacency before similarity coupling acts.
    fn segment(grid: &[u8], width: usize, height: usize, background: u8) -> Self {
        let n = width * height;
        let mut labels = vec![usize::MAX; n];
        let phases = vec![f64::NAN; n];
        let mut next_label = 0usize;
        let mut stack = Vec::new();
        for seed in 0..n {
            if grid[seed] == background || labels[seed] != usize::MAX {
                continue;
            }
            let colour = grid[seed];
            labels[seed] = next_label;
            stack.push(seed);
            while let Some(idx) = stack.pop() {
                let (r, c) = (idx / width, idx % width);
                for (nr, nc) in [
                    (r.wrapping_sub(1), c),
                    (r + 1, c),
                    (r, c.wrapping_sub(1)),
                    (r, c + 1),
                ] {
                    if nr < height && nc < width {
                        let nidx = nr * width + nc;
                        if grid[nidx] == colour && labels[nidx] == usize::MAX {
                            labels[nidx] = next_label;
                            stack.push(nidx);
                        }
                    }
                }
            }
            next_label += 1;
        }
        Self { width, height, phases, labels }
    }

    /// Kuramoto integration — natural frequency seeded by colour, strong
    /// coupling to same-component neighbours, weak similarity coupling to
    /// same-colour cells in a local window (the Gestalt similarity term
    /// that lets separated fragments of one colour bind).
    fn integrate(&mut self, grid: &[u8], steps: usize) {
        const K_LOCAL: f64 = 0.7;
        const K_SIMILAR: f64 = 0.05;
        const WINDOW: usize = 8;
        let (w, h) = (self.width, self.height);
        for (i, &lab) in self.labels.iter().enumerate() {
            if lab != usize::MAX {
                // Colour seeds the natural frequency — distinct hues enter
                // distinct phase basins, so binding separates at colour.
                self.phases[i] = (grid[i] as f64 + 1.0) * 0.5;
            }
        }
        for _ in 0..steps {
            let cur = self.phases.clone();
            for idx in 0..cur.len() {
                if cur[idx].is_nan() {
                    continue;
                }
                let lab = self.labels[idx];
                let (r, c) = (idx / w, idx % w);
                let mut drive = 0.0;
                let mut count = 0.0;
                // Strong: same-component 4-neighbours (contiguity).
                for (nr, nc) in [
                    (r.wrapping_sub(1), c),
                    (r + 1, c),
                    (r, c.wrapping_sub(1)),
                    (r, c + 1),
                ] {
                    if nr < h && nc < w {
                        let nidx = nr * w + nc;
                        if self.labels[nidx] == lab {
                            drive += (cur[nidx] - cur[idx]).sin();
                            count += 1.0;
                        }
                    }
                }
                let mut delta = if count > 0.0 { K_LOCAL * drive / count } else { 0.0 };
                // Weak: same-colour cells in a window (similarity Gestalt —
                // remote fragments of one hue pull toward shared phase).
                let mut sim = 0.0;
                let mut scount = 0.0;
                let r0 = r.saturating_sub(WINDOW);
                let r1 = (r + WINDOW).min(h - 1);
                let c0 = c.saturating_sub(WINDOW);
                let c1 = (c + WINDOW).min(w - 1);
                for nr in r0..=r1 {
                    for nc in c0..=c1 {
                        let nidx = nr * w + nc;
                        if nidx != idx && self.labels[nidx] != lab && grid[nidx] == grid[idx] {
                            sim += (cur[nidx] - cur[idx]).sin();
                            scount += 1.0;
                        }
                    }
                }
                if scount > 0.0 {
                    delta += K_SIMILAR * sim / scount;
                }
                self.phases[idx] = (cur[idx] + delta).rem_euclid(TWO_PI);
            }
        }
    }

    /// Read out assemblies: contiguous components, merged when their mean
    /// phases locked within tolerance — the similarity binding. Each object
    /// carries its locked phase and mean-field coherence.
    fn assemblies(&self, grid: &[u8]) -> Vec<ProtoObject> {
        const PHASE_TOL: f64 = 0.35;
        let mut comps: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for (i, &lab) in self.labels.iter().enumerate() {
            if lab != usize::MAX {
                comps.entry(lab).or_default().push(i);
            }
        }
        let mut stats: Vec<(usize, f64, f64)> = Vec::new(); // (label, mean_phase, coherence)
        for (&lab, cells) in &comps {
            let (mut sx, mut sy) = (0.0, 0.0);
            for &i in cells {
                sx += self.phases[i].cos();
                sy += self.phases[i].sin();
            }
            let m = cells.len() as f64;
            let coherence = (sx * sx + sy * sy).sqrt() / m;
            stats.push((lab, sy.atan2(sx), coherence));
        }
        // Merge components whose collective phases locked (phase-tolerant
        // union — non-adjacent same-phase fragments form one assembly).
        let mut parent: Vec<usize> = (0..stats.len()).collect();
        fn find(p: &mut Vec<usize>, x: usize) -> usize {
            if p[x] != x { p[x] = find(p, p[x]); }
            p[x]
        }
        for a in 0..stats.len() {
            for b in (a + 1)..stats.len() {
                let d = (stats[a].1 - stats[b].1).abs();
                let d = d.min(TWO_PI - d);
                if d < PHASE_TOL
                    && grid[comps[&stats[a].0][0]] == grid[comps[&stats[b].0][0]]
                {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb { parent[rb] = ra; }
                }
            }
        }
        let mut merged: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
        for (i, &(lab, phase, coh)) in stats.iter().enumerate() {
            merged.entry(find(&mut parent, i)).or_default().extend(comps[&lab].iter().copied());
            let _ = (phase, coh);
        }
        let mut out = Vec::new();
        for (_root, cells) in merged {
            let n = cells.len();
            let (mut rmin, mut rmax, mut cmin, mut cmax) = (usize::MAX, 0, usize::MAX, 0);
            let (mut sr, mut sc) = (0.0, 0.0);
            let (mut sx, mut sy) = (0.0, 0.0);
            for &i in &cells {
                let (r, c) = (i / self.width, i % self.width);
                rmin = rmin.min(r); rmax = rmax.max(r);
                cmin = cmin.min(c); cmax = cmax.max(c);
                sr += r as f64; sc += c as f64;
                sx += self.phases[i].cos(); sy += self.phases[i].sin();
            }
            let first = cells[0];
            out.push(ProtoObject {
                color: grid[first],
                size: n,
                cells,
                bbox: (rmin, cmin, rmax, cmax),
                centroid: (sr / n as f64, sc / n as f64),
                phase: sy.atan2(sx),
                coherence: (sx * sx + sy * sy).sqrt() / n as f64,
            });
        }
        out
    }
}
