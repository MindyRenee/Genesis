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
}

impl MotorTarget {
    /// Fixed 14-byte wire form:
    /// `[flags][action_id][f32 a_strength][u16 x][u16 y][f32 s_strength]`.
    /// `flags` bit0 = action present, bit1 = cell present.
    pub fn encode(&self) -> Vec<u8> {
        let mut flags = 0u8;
        if self.action.is_some() {
            flags |= 1;
        }
        if self.cell.is_some() {
            flags |= 2;
        }
        let (action_id, a_str) = self.action.unwrap_or((0xFF, 0.0));
        let (x, y, s_str) = self.cell.unwrap_or((0, 0, 0.0));
        let mut out = Vec::with_capacity(14);
        out.push(flags);
        out.push(action_id);
        out.extend_from_slice(&a_str.to_le_bytes());
        out.extend_from_slice(&x.to_le_bytes());
        out.extend_from_slice(&y.to_le_bytes());
        out.extend_from_slice(&s_str.to_le_bytes());
        out
    }
}

/// Read one little-endian `f32` element out of a tensor payload, or
/// `None` when the layout doesn't hold an `f32` there.
fn f32_at(t: &Tensor, i: usize) -> Option<f32> {
    t.data
        .get(i * 4..i * 4 + 4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
}

/// Scan a 2-D `f32` field for its strongest live cell and project the
/// winner onto the canonical coordinate grid.
///
/// Nested `(y, x)` loops over row-major data: `i = y*w + x` picks the
/// element, `x = i % w` / `y = i / w` recover the cell. A field
/// published at a different resolution is still honored — the winning
/// cell is scaled into the canonical grid rather than rejected, so a
/// coarse 8×8 urge and a native 64×64 field address the same world.
pub fn spatial_winner(field: &Tensor) -> Option<(u16, u16, f32)> {
    if field.dtype != dtype::F32 || field.shape.len() != 2 {
        return None;
    }
    let h = field.shape[0] as usize;
    let w = field.shape[1] as usize;
    if h == 0 || w == 0 {
        return None;
    }

    let mut best = f32::NEG_INFINITY;
    let mut best_cell: Option<(usize, usize)> = None;
    for y in 0..h {
        for x in 0..w {
            let i = cell_index(x, y, w);
            let Some(v) = f32_at(field, i) else {
                return None; // payload shorter than the declared shape
            };
            if v.is_finite() && v > best {
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
    MotorTarget { action, cell }
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
    fn motor_target_encodes_fixed_width() {
        let t = MotorTarget {
            action: Some((6, 0.8)),
            cell: Some((12, 34, 0.5)),
        };
        let bytes = t.encode();
        assert_eq!(bytes.len(), 14);
        assert_eq!(bytes[0], 0b11);
        assert_eq!(bytes[1], 6);
        assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), 12);
        assert_eq!(u16::from_le_bytes([bytes[8], bytes[9]]), 34);

        let empty = MotorTarget::default().encode();
        assert_eq!(empty[0], 0);
        assert_eq!(empty[1], 0xFF);
    }
}
