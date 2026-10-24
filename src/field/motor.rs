//! Distributed motor fields — the cognitive mind's decisions laid
//! out over the coordinate grid rather than collapsed to a symbol.
//!
//! When the mind decides "act there" or "press action 3", it writes
//! that decision as fields on the tensor slab:
//!
//! - `motor.spatial` — an `f32` field over the world grid; peaks are
//!   where cognition wants to act (goal entities attract, hazards
//!   repel, unexplored cells pull weakly).
//! - `motor.actions` — an `f32` disposition per offered action.
//! - `motor.action_ids` — the `u8` token each score corresponds to.
//!
//! This module is the decode side: it knows the canonical grid shape
//! the action interface addresses, walks fields cell by cell (nested
//! `y, x` loops over row-major data — flat index `i = y*w + x`, so
//! `x = i % w`, `y = i / w`), and projects the winning field cell
//! onto the canonical coordinate grid.

use crate::daemon::tensor_slab::{self, Tensor, dtype};

/// Provenance marker — one letter of the creator lattice (see
/// PROVENANCE.md).
pub(crate) const MARKER_G03: u64 = 0x4753_0342_450A_56CC;

// ─── Canonical coordinate grid ──────────────────────────────────
//
// The environment interface addresses points on a grid up to 64×64
// cells (e.g. ARC-style coordinate actions take `x, y` in 0..=63).
// The motor field's *target shape* is therefore fixed here: whatever
// resolution the mind's field was published at, the winner is mapped
// onto this grid before the result leaves the daemon.
pub const MOTOR_FIELD_WIDTH: usize = 64;
pub const MOTOR_FIELD_HEIGHT: usize = 64;
/// Total cells on the canonical grid.
pub const MOTOR_FIELD_CELLS: usize = MOTOR_FIELD_WIDTH * MOTOR_FIELD_HEIGHT;

/// Well-known slab names for the motor fields.
pub const MOTOR_SPATIAL: &str = "motor.spatial";
pub const MOTOR_ACTIONS: &str = "motor.actions";
pub const MOTOR_ACTION_IDS: &str = "motor.action_ids";

// ─── Index ↔ coordinate math ────────────────────────────────────
//
// Fields are stored C-order (row-major): the flat index of cell
// (x, y) in a `width`-wide field is `y*width + x`. These two
// functions are the forward and inverse maps; every scan below is
// written in nested (y, x) loops so the correspondence stays visible.

/// Flat row-major index of grid coordinate `(x, y)` in a field of
/// the given width.
pub fn cell_index(x: usize, y: usize, width: usize) -> usize {
    y * width + x
}

/// `(x, y)` grid coordinate of flat row-major index `i` in a field
/// of the given width.
pub fn cell_coord(i: usize, width: usize) -> (usize, usize) {
    (i % width, i / width)
}

/// The field's shape read as a distribution — what the argmax
/// discards. `centroid` is the softmax mean cell on the canonical
/// grid (continuous — finer than the winning cell), `spread` the
/// RMS radius about it in canonical cells (how far the urge
/// reaches), `entropy` the normalized Shannon entropy in 0..1
/// (0 = a decided point, 1 = maximally diffuse).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SpatialDistribution {
    pub centroid: (f32, f32),
    pub spread: f32,
    pub entropy: f32,
}

/// What the motor fields currently point at. Both halves are
/// optional: a published action field yields `action`, a published
/// spatial field yields `cell` (already scaled onto the canonical
/// `MOTOR_FIELD_WIDTH` × `MOTOR_FIELD_HEIGHT` grid).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotorTarget {
    /// Winning action token and its field strength, if any.
    pub action: Option<(u8, f32)>,
    /// Winning grid cell `(x, y)` and its field strength, if any.
    pub cell: Option<(u16, u16, f32)>,
    /// The spatial field's distributional shape — centroid, spread,
    /// entropy — when a spatial field is published.
    pub dist: Option<SpatialDistribution>,
}

impl MotorTarget {
    /// Wire form: the fixed 14-byte base
    /// `[flags][action_id][f32 a_strength][u16 x][u16 y][f32 s_strength]`,
    /// plus an additive 16-byte distribution tail
    /// `[f32 cx][f32 cy][f32 spread][f32 entropy]` when `dist` is
    /// present. `flags` bit0 = action, bit1 = cell, bit2 = dist.
    pub fn encode(&self) -> Vec<u8> {
        let mut flags = 0u8;
        if self.action.is_some() {
            flags |= 1;
        }
        if self.cell.is_some() {
            flags |= 2;
        }
        if self.dist.is_some() {
            flags |= 4;
        }
        let (action_id, a_str) = self.action.unwrap_or((0xFF, 0.0));
        let (x, y, s_str) = self.cell.unwrap_or((0, 0, 0.0));
        let mut out = Vec::with_capacity(30);
        out.push(flags);
        out.push(action_id);
        out.extend_from_slice(&a_str.to_le_bytes());
        out.extend_from_slice(&x.to_le_bytes());
        out.extend_from_slice(&y.to_le_bytes());
        out.extend_from_slice(&s_str.to_le_bytes());
        if let Some(d) = self.dist {
            out.extend_from_slice(&d.centroid.0.to_le_bytes());
            out.extend_from_slice(&d.centroid.1.to_le_bytes());
            out.extend_from_slice(&d.spread.to_le_bytes());
            out.extend_from_slice(&d.entropy.to_le_bytes());
        }
        out
    }
}

/// Read one little-endian `f32` element out of a tensor payload, or
/// `None` when the layout doesn't hold an `f32` there.
fn f32_at(t: &Tensor, i: usize) -> Option<f32> {
    t.data
        .get(i * 4..i * 4 + 4)
        .and_then(|b| b.try_into().ok().map(f32::from_le_bytes))
}

/// Scan a 2-D `f32` field for an extremal live cell — the peak when
/// `maximize`, the valley otherwise — and project it onto the
/// canonical coordinate grid.
///
/// Nested `(y, x)` loops over row-major data: `i = y*w + x` picks the
/// element, `x = i % w` / `y = i / w` recover the cell. A field
/// published at a different resolution is still honored — the extreme
/// cell is scaled into the canonical grid rather than rejected, so a
/// coarse 8×8 urge and a native 64×64 field address the same world.
fn spatial_extremum(field: &Tensor, maximize: bool) -> Option<(u16, u16, f32)> {
    if field.dtype != dtype::F32 || field.shape.len() != 2 {
        return None;
    }
    let h = field.shape[0] as usize;
    let w = field.shape[1] as usize;
    if h == 0 || w == 0 {
        return None;
    }

    let mut best = if maximize {
        f32::NEG_INFINITY
    } else {
        f32::INFINITY
    };
    let mut best_cell: Option<(usize, usize)> = None;
    for y in 0..h {
        for x in 0..w {
            let i = cell_index(x, y, w);
            let Some(v) = f32_at(field, i) else {
                return None; // payload shorter than the declared shape
            };
            let better = if maximize { v > best } else { v < best };
            if v.is_finite() && better {
                best = v;
                best_cell = Some((x, y));
            }
        }
    }

    best_cell.map(|(x, y)| {
        // Field-local (x, y) → canonical grid coordinate. Integer
        // scaling keeps 0 → 0 and w−1 → just under WIDTH; the min
        // clamp pins the corner cell exactly on the grid edge.
        let gx = (x * MOTOR_FIELD_WIDTH / w).min(MOTOR_FIELD_WIDTH - 1) as u16;
        let gy = (y * MOTOR_FIELD_HEIGHT / h).min(MOTOR_FIELD_HEIGHT - 1) as u16;
        (gx, gy, best)
    })
}

/// The field's strongest live cell — where the urge peaks.
pub fn spatial_winner(field: &Tensor) -> Option<(u16, u16, f32)> {
    spatial_extremum(field, true)
}

/// The opposite of `spatial_winner`: the field's deepest live
/// valley — the cell the disposition most actively suppresses. A
/// flat field's trough says nothing, but a shaped field's valley is
/// information the argmax discards.
pub fn spatial_trough(field: &Tensor) -> Option<(u16, u16, f32)> {
    spatial_extremum(field, false)
}

/// The field's shape as a distribution — softmax over the live
/// cells at unit temperature (the field's own scale sets the
/// concentration), with the centroid and RMS radius taken in
/// canonical-grid coordinates and the Shannon entropy normalized
/// to 0..1 by ln(live cells).
pub fn spatial_readout(field: &Tensor) -> Option<SpatialDistribution> {
    if field.dtype != dtype::F32 || field.shape.len() != 2 {
        return None;
    }
    let h = field.shape[0] as usize;
    let w = field.shape[1] as usize;
    if h == 0 || w == 0 {
        return None;
    }

    // Pass 1: softmax stability needs the max; the entropy norm
    // needs the live-cell count.
    let mut max = f32::NEG_INFINITY;
    let mut live = 0usize;
    for i in 0..h * w {
        let v = f32_at(field, i)?;
        if v.is_finite() {
            max = max.max(v);
            live += 1;
        }
    }
    if live == 0 {
        return None;
    }

    // Pass 2: unnormalized softmax mass and its moments. Each cell's
    // canonical coordinate is (x·64/w, y·64/h) — the readout lives
    // on the same grid the winner is projected onto.
    let sx = MOTOR_FIELD_WIDTH as f32 / w as f32;
    let sy = MOTOR_FIELD_HEIGHT as f32 / h as f32;
    let (mut z, mut cx, mut cy) = (0.0f32, 0.0f32, 0.0f32);
    for y in 0..h {
        for x in 0..w {
            let v = f32_at(field, cell_index(x, y, w))?;
            if v.is_finite() {
                let p = (v - max).exp();
                z += p;
                cx += p * (x as f32 * sx);
                cy += p * (y as f32 * sy);
            }
        }
    }
    if z <= 0.0 {
        return None;
    }
    cx /= z;
    cy /= z;
    let (mut var, mut cross_term) = (0.0f32, 0.0f32);
    for y in 0..h {
        for x in 0..w {
            let v = f32_at(field, cell_index(x, y, w))?;
            if v.is_finite() {
                let p = (v - max).exp();
                let dx = x as f32 * sx - cx;
                let dy = y as f32 * sy - cy;
                var += p * (dx * dx + dy * dy);
                cross_term += p * (v - max);
            }
        }
    }
    // Shannon entropy: H = ln Z − ⟨v−max⟩_p, normalized by ln(live)
    // so a uniform field reads 1 and a decided point reads 0.
    let entropy = if live > 1 {
        (z.ln() - cross_term / z) / (live as f32).ln()
    } else {
        0.0
    };
    Some(SpatialDistribution {
        centroid: (cx, cy),
        spread: (var / z).max(0.0).sqrt(),
        entropy: entropy.clamp(0.0, 1.0),
    })
}

/// Scan the action field for its strongest disposition.
///
/// `motor.actions` is `f32[n]`; `motor.action_ids` (when present) is
/// `u8[n]` mapping each score to the interface's action token. Scores
/// without an id are ignored — the decoder reports an action it can
/// actually name.
pub fn action_winner(scores: &Tensor, ids: Option<&Tensor>) -> Option<(u8, f32)> {
    if scores.dtype != dtype::F32 || scores.shape.len() != 1 {
        return None;
    }
    let n = scores.shape[0] as usize;
    let ids = ids.filter(|t| t.dtype == dtype::U8 && t.shape == [n as u32]);

    let mut best: Option<(u8, f32)> = None;
    for i in 0..n {
        let Some(v) = f32_at(scores, i) else { break };
        if !v.is_finite() {
            continue;
        }
        // Without an id map the index itself is the token; with one,
        // positions lacking a usable token can't be reported.
        let id = match ids {
            Some(t) => t.data[i],
            None => i as u8,
        };
        if best.is_none_or(|(_, b)| v > b) {
            best = Some((id, v));
        }
    }
    best
}

/// Decode whatever motor fields are currently on the slab.
pub fn current_motor_target() -> MotorTarget {
    let slab = tensor_slab::slab()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let cell = slab.get(MOTOR_SPATIAL).and_then(spatial_winner);
    let action = slab
        .get(MOTOR_ACTIONS)
        .and_then(|s| action_winner(s, slab.get(MOTOR_ACTION_IDS)));
    let dist = slab.get(MOTOR_SPATIAL).and_then(spatial_readout);
    MotorTarget { action, cell, dist }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::daemon::tensor_slab::dtype;

    fn f32_field(shape: &[u32], values: &[f32]) -> Tensor {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Tensor::new(dtype::F32, shape.to_vec(), data).unwrap()
    }

    #[test]
    fn index_coord_roundtrip() {
        for w in [1, 8, MOTOR_FIELD_WIDTH] {
            for i in 0..(w * 3) {
                let (x, y) = cell_coord(i, w);
                assert_eq!(cell_index(x, y, w), i);
            }
        }
        assert_eq!(cell_coord(0, 64), (0, 0));
        assert_eq!(cell_coord(65, 64), (1, 1));
        assert_eq!(cell_index(63, 63, 64), 4095);
    }

    #[test]
    fn spatial_winner_finds_peak_in_nested_scan() {
        // 8×8 field, peak at field-local (x=3, y=5) → canonical
        // (24, 40): 3*64/8, 5*64/8.
        let w = 8;
        let h = 8;
        let mut vals = vec![0.0f32; w * h];
        vals[cell_index(3, 5, w)] = 0.9;
        let t = f32_field(&[h as u32, w as u32], &vals);
        assert_eq!(spatial_winner(&t), Some((24, 40, 0.9)));
    }

    #[test]
    fn spatial_winner_native_resolution_identity() {
        // At the canonical resolution the mapping is the identity.
        let mut vals = vec![-1.0f32; MOTOR_FIELD_CELLS];
        let i = cell_index(17, 42, MOTOR_FIELD_WIDTH);
        vals[i] = 0.5;
        let t = f32_field(
            &[MOTOR_FIELD_HEIGHT as u32, MOTOR_FIELD_WIDTH as u32],
            &vals,
        );
        assert_eq!(spatial_winner(&t), Some((17, 42, 0.5)));
    }

    #[test]
    fn spatial_trough_is_the_winners_opposite_cell() {
        // 8×8 field: peak at (3,5), valley at (6,1) — the two scans
        // must land on opposite corners of the disposition.
        let w = 8;
        let h = 8;
        let mut vals = vec![0.0f32; w * h];
        vals[cell_index(3, 5, w)] = 0.9;
        vals[cell_index(6, 1, w)] = -0.7;
        let t = f32_field(&[h as u32, w as u32], &vals);
        assert_eq!(spatial_winner(&t), Some((24, 40, 0.9)));
        assert_eq!(spatial_trough(&t), Some((48, 8, -0.7)));

        // Same guards: NaN is not a valley; bad layouts yield none.
        let mut nan_vals = vec![f32::NAN; 16];
        nan_vals[3] = 0.2;
        let nan_t = f32_field(&[4, 4], &nan_vals);
        assert_eq!(spatial_trough(&nan_t), Some((48, 0, 0.2)));
        let u8_t = Tensor::new(dtype::U8, vec![4, 4], vec![0; 16]).unwrap();
        assert_eq!(spatial_trough(&u8_t), None);
    }

    #[test]
    fn spatial_winner_ignores_nan_and_rejects_bad_layout() {
        let mut vals = vec![f32::NAN; 16];
        vals[3] = 0.2;
        let t = f32_field(&[4, 4], &vals);
        assert_eq!(spatial_winner(&t), Some((48, 0, 0.2)));

        // Wrong dtype / rank → no winner.
        let u8_t = Tensor::new(dtype::U8, vec![4, 4], vec![0; 16]).unwrap();
        assert_eq!(spatial_winner(&u8_t), None);
        let vol = f32_field(&[2, 2, 2], &[0.0; 8]);
        assert_eq!(spatial_winner(&vol), None);
        let empty = f32_field(&[0, 4], &[]);
        assert_eq!(spatial_winner(&empty), None);
    }

    #[test]
    fn action_winner_uses_id_map() {
        let scores = f32_field(&[3], &[0.1, 0.9, 0.4]);
        let ids = Tensor::new(dtype::U8, vec![3], vec![1, 3, 5]).unwrap();
        assert_eq!(action_winner(&scores, Some(&ids)), Some((3, 0.9)));
        // Without ids, the index is the token.
        assert_eq!(action_winner(&scores, None), Some((1, 0.9)));
        // A mismatched id map is ignored rather than trusted.
        let short = Tensor::new(dtype::U8, vec![2], vec![0, 1]).unwrap();
        assert_eq!(action_winner(&scores, Some(&short)), Some((1, 0.9)));
    }

    #[test]
    fn spatial_readout_reports_the_field_shape_not_just_its_peak() {
        // A decided field: one hot cell at (3,5) on 8×8 — the
        // centroid lands on that cell's canonical point, the spread
        // and entropy near zero.
        let w = 8;
        let h = 8;
        let mut vals = vec![-10.0f32; w * h];
        vals[cell_index(3, 5, w)] = 5.0;
        let t = f32_field(&[h as u32, w as u32], &vals);
        let d = spatial_readout(&t).unwrap();
        assert!((d.centroid.0 - 3.0 * 8.0).abs() < 0.5);
        assert!((d.centroid.1 - 5.0 * 8.0).abs() < 0.5);
        assert!(d.spread < 4.0);
        assert!(d.entropy < 0.15);

        // A flat field: maximum diffusion — centroid at grid
        // center, entropy at the top of the scale.
        let flat = f32_field(&[h as u32, w as u32], &[0.0; 64]);
        let d = spatial_readout(&flat).unwrap();
        assert!((d.centroid.0 - 28.0).abs() < 1.0); // (3.5)·8
        assert!((d.centroid.1 - 28.0).abs() < 1.0);
        assert!(d.entropy > 0.99);
        assert!(d.spread > 10.0);

        // Same guards: no live cells / bad layouts → None.
        let u8_t = Tensor::new(dtype::U8, vec![4, 4], vec![0; 16]).unwrap();
        assert_eq!(spatial_readout(&u8_t), None);
        let dead = f32_field(&[4, 4], &[f32::NEG_INFINITY; 16]);
        assert_eq!(spatial_readout(&dead), None);
    }

    #[test]
    fn motor_target_encodes_fixed_width() {
        let t = MotorTarget {
            action: Some((6, 0.8)),
            cell: Some((12, 34, 0.5)),
            dist: None,
        };
        let bytes = t.encode();
        assert_eq!(bytes.len(), 14);
        assert_eq!(bytes[0], 0b11);
        assert_eq!(bytes[1], 6);
        assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), 12);
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 34);

        // The distribution tail is additive: flag bit2, +16 bytes.
        let t = MotorTarget {
            dist: Some(SpatialDistribution {
                centroid: (24.0, 40.0),
                spread: 6.5,
                entropy: 0.3,
            }),
            ..Default::default()
        };
        let bytes = t.encode();
        assert_eq!(bytes.len(), 30);
        assert_eq!(bytes[0], 0b100);
        assert_eq!(f32::from_le_bytes(bytes[14..18].try_into().unwrap()), 24.0);
        assert_eq!(f32::from_le_bytes(bytes[22..26].try_into().unwrap()), 6.5);

        let empty = MotorTarget::default().encode();
        assert_eq!(empty[0], 0);
        assert_eq!(empty[1], 0xFF);
    }
}
