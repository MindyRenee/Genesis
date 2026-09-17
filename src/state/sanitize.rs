//! NaN / infinity sanitization primitives.
//!
//! ## Why this module exists
//!
//! Rust's `f32::clamp(min, max)` returns `NaN` when `self` is `NaN`.
//! This means every `.clamp()` call in the codebase — and there are
//! hundreds — provides **zero** protection against NaN propagation.
//! A single NaN in a neurochemical level propagates through the
//! 18×18 coupling matrix to all 18 chemicals within one tick, and
//! is then persisted to the mmap'd state file. The system never
//! recovers — the NaN is permanent.
//!
//! The entry points for NaN are:
//! - **IPC**: `f32::from_le_bytes` on wire data from the cognitive
//!   mind. A bug in the Python layer (or a malicious client) can
//!   send NaN or ±inf as an impulse magnitude, user-affect field,
//!   or event salience.
//! - **Model file loading**: `f32::from_le_bytes` on the active
//!   inference model file. A crash during `save()` can corrupt the
//!   file, injecting NaN into the transition matrix.
//! - **Computation**: `subtype_effective_level` for dopamine can
//!   produce values up to ~6.0 (breaking the [0, 2] contract),
//!   and the Pearson correlation in the dyadic model can produce
//!   NaN when variance is zero.
//!
//! This module provides the atomic building blocks for a multi-layer
//! defense:
//!
//! 1. **Entry-point sanitization**: reject NaN/inf at every IPC
//!    handler and file loader before the value enters the system.
//! 2. **`apply_impulse` sanitization**: the last line of defense
//!    before NaN enters a chemical's level/velocity.
//! 3. **Tick-level circuit breaker**: scan all chemicals at the
//!    start of each tick and reset any with non-finite fields.
//! 4. **Derived-value sanitization**: sanitize arousal, valence,
//!    global_tone, and plasticity_gate after computation.

/// Return `v` if it is finite (not NaN, not ±inf), else `default`.
///
/// This is the atomic NaN-replacement primitive. Use it when you
/// need a specific fallback for a non-finite value (e.g., replacing
/// a corrupted model-file weight with 0.0).
#[inline]
#[must_use]
pub fn finite_or(v: f32, default: f32) -> f32 {
    if v.is_finite() { v } else { default }
}

/// Clamp `v` to `[min, max]` if it is finite; return `min` if it is
/// NaN or ±inf.
///
/// This is the NaN-safe replacement for `f32::clamp`. Rust's
/// `f32::clamp` returns `NaN` when `self` is `NaN`, making it useless
/// for NaN defense. This function guarantees a finite return value
/// within `[min, max]`.
///
/// The choice of `min` (rather than `default`) as the NaN fallback
/// is deliberate: for neurochemical levels, receptor sensitivities,
/// and other [0, max] quantities, `min = 0.0` is the safe "empty"
/// state — a NaN level is treated as "no signal," not "maximum
/// signal."
#[inline]
#[must_use]
pub fn finite_clamp(v: f32, min: f32, max: f32) -> f32 {
    if v.is_finite() {
        v.clamp(min, max)
    } else {
        min
    }
}

/// Sanitize a 2D array in place: replace every non-finite element
/// with `default`.
pub fn sanitize_matrix_f32<const M: usize, const N: usize>(mat: &mut [[f32; N]; M], default: f32) {
    for row in mat.iter_mut() {
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = default;
            }
        }
    }
}

/// Sanitize a 1D array in place: replace every non-finite element
/// with `default`.
pub fn sanitize_array_f32<const N: usize>(arr: &mut [f32; N], default: f32) {
    for v in arr.iter_mut() {
        if !v.is_finite() {
            *v = default;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finite_or_returns_finite() {
        assert_eq!(finite_or(1.23, 0.0), 1.23);
        assert_eq!(finite_or(-1.0, 0.0), -1.0);
        assert_eq!(finite_or(0.0, 42.0), 0.0);
    }

    #[test]
    fn test_finite_or_replaces_nan() {
        assert_eq!(finite_or(f32::NAN, 0.0), 0.0);
        assert_eq!(finite_or(f32::NAN, 42.0), 42.0);
    }

    #[test]
    fn test_finite_clamp_clamps_finite() {
        assert_eq!(finite_clamp(0.5, 0.0, 1.0), 0.5);
        assert_eq!(finite_clamp(-1.0, 0.0, 1.0), 0.0);
        assert_eq!(finite_clamp(2.0, 0.0, 1.0), 1.0);
    }

    #[test]
    fn test_finite_clamp_replaces_nan_with_min() {
        assert_eq!(finite_clamp(f32::NAN, 0.0, 1.0), 0.0);
        assert_eq!(finite_clamp(f32::NAN, -1.0, 1.0), -1.0);
    }

    #[test]
    fn test_sanitize_matrix() {
        let mut mat = [[1.0, f32::NAN], [f32::INFINITY, 2.0]];
        sanitize_matrix_f32(&mut mat, 0.0);
        assert_eq!(mat, [[1.0, 0.0], [0.0, 2.0]]);
    }

    #[test]
    fn test_sanitize_array() {
        let mut arr = [1.0, f32::NAN, 3.0, f32::NEG_INFINITY];
        sanitize_array_f32(&mut arr, 0.0);
        assert_eq!(arr, [1.0, 0.0, 3.0, 0.0]);
    }
}
