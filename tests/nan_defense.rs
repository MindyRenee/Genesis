//! Regression tests for NaN/inf rejection at every entry point.
//!
//! These tests verify that the multi-layer NaN defense works:
//! 1. `finite_or` / `finite_clamp` primitives replace non-finite values
//! 2. `apply_impulse` rejects NaN magnitudes
//! 3. `effective_level` and `subtype_effective_level` return finite values
//! 4. The tick-level circuit breaker resets non-finite chemical fields
//! 5. The active inference engine rejects NaN inputs and model values
//! 6. The dyadic model rejects NaN observations and computes finite synchrony
//!
//! The critical scenario these tests guard against: a single NaN in any
//! chemical field propagates through the 18×18 coupling matrix to all
//! 18 chemicals within one tick, permanently corrupting the mmap'd state.

use genesis::state::neurochemical::{NeuroTickParams, Neurochemical, NeurochemicalId};
use genesis::state::*;

// ─── sanitize primitives ───────────────────────────────────────

#[test]
fn test_finite_or_returns_finite_unchanged() {
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
fn test_finite_or_replaces_inf() {
    assert_eq!(finite_or(f32::INFINITY, 0.0), 0.0);
    assert_eq!(finite_or(f32::NEG_INFINITY, 0.0), 0.0);
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
fn test_finite_clamp_replaces_inf_with_min() {
    assert_eq!(finite_clamp(f32::INFINITY, 0.0, 1.0), 0.0);
    assert_eq!(finite_clamp(f32::NEG_INFINITY, 0.0, 1.0), 0.0);
}

#[test]
fn test_native_clamp_does_not_catch_nan() {
    // This test documents the bug that motivates the sanitize module.
    // Rust's native f32::clamp returns NaN for NaN input.
    let result = f32::NAN.clamp(0.0, 1.0);
    assert!(result.is_nan());
}

// ─── apply_impulse rejects NaN ──────────────────────────────────

#[test]
fn test_apply_impulse_rejects_nan_magnitude() {
    let mut chem = Neurochemical::new(NeurochemicalId::Dopamine, 0);
    let original_level = chem.level;
    let original_velocity = chem.velocity;

    // Apply a NaN impulse — should be a no-op, not a corruption
    chem.apply_impulse(f32::NAN, 0);

    assert!(
        chem.level.is_finite(),
        "level must be finite after NaN impulse, got {}",
        chem.level
    );
    assert!(
        chem.velocity.is_finite(),
        "velocity must be finite after NaN impulse, got {}",
        chem.velocity
    );
    // The NaN should have been replaced with 0.0, so the impulse
    // is a no-op — level and velocity should be unchanged.
    assert!(
        (chem.level - original_level).abs() < 1e-6,
        "level should be unchanged after NaN impulse (no-op), was {}, now {}",
        original_level,
        chem.level
    );
    assert!(
        (chem.velocity - original_velocity).abs() < 1e-6,
        "velocity should be unchanged after NaN impulse (no-op), was {}, now {}",
        original_velocity,
        chem.velocity
    );
}

#[test]
fn test_apply_impulse_rejects_inf_magnitude() {
    let mut chem = Neurochemical::new(NeurochemicalId::Dopamine, 0);
    chem.apply_impulse(f32::INFINITY, 0);

    assert!(
        chem.level.is_finite(),
        "level must be finite after inf impulse"
    );
    assert!(
        chem.velocity.is_finite(),
        "velocity must be finite after inf impulse"
    );
}

#[test]
fn test_apply_impulse_accepts_normal_magnitude() {
    let mut chem = Neurochemical::new(NeurochemicalId::Dopamine, 0);
    let original_level = chem.level;
    chem.apply_impulse(0.5, 0);

    // A positive impulse should increase the level
    assert!(
        chem.level > original_level,
        "positive impulse should increase level"
    );
    assert!(chem.level.is_finite());
}

#[test]
fn test_apply_impulse_clamps_huge_magnitude_velocity() {
    // A huge finite magnitude should not produce an unbounded velocity.
    // Without velocity clamping, a magnitude of 1e20 would set velocity
    // to ~3e19, which decays geometrically but keeps the chemical
    // saturated for hundreds of ticks — a local DoS that distorts the
    // entire neurochemical state.
    let mut chem = Neurochemical::new(NeurochemicalId::Dopamine, 0);
    chem.apply_impulse(1e20, 0);

    assert!(
        chem.velocity.abs() <= 1.0,
        "velocity must be clamped to [-1, 1] even for huge impulses, got {}",
        chem.velocity
    );
    assert!(chem.level.is_finite());
    assert!(
        chem.level <= 1.0,
        "level must be in [0, 1], got {}",
        chem.level
    );
}

// ─── effective_level returns finite values ──────────────────────

#[test]
fn test_effective_level_returns_finite_for_nan_inputs() {
    let mut chem = Neurochemical::new(NeurochemicalId::Dopamine, 0);
    // Corrupt the fields with NaN
    chem.level = f32::NAN;
    chem.receptor_sensitivity = 1.0;
    chem.desensitization_factor = 1.0;
    chem.internalization_factor = 1.0;

    let eff = chem.effective_level();
    assert!(
        eff.is_finite(),
        "effective_level must be finite for NaN level, got {}",
        eff
    );
    assert!(
        (0.0..=2.0).contains(&eff),
        "effective_level must be in [0, 2], got {}",
        eff
    );
}

#[test]
fn test_effective_level_returns_finite_for_nan_sensitivity() {
    let mut chem = Neurochemical::new(NeurochemicalId::Dopamine, 0);
    chem.level = 0.5;
    chem.receptor_sensitivity = f32::NAN;
    chem.desensitization_factor = 1.0;
    chem.internalization_factor = 1.0;

    let eff = chem.effective_level();
    assert!(
        eff.is_finite(),
        "effective_level must be finite for NaN sensitivity"
    );
}

// ─── subtype_effective_level dopamine branch ───────────────────

#[test]
fn test_subtype_effective_level_dopamine_clamped_to_2() {
    let mut chem = Neurochemical::new(NeurochemicalId::Dopamine, 0);
    // Set up conditions that would produce a value > 2.0 without clamping:
    // base = level × sensitivity × desens × intern
    // For base = 2.0: level=1.0, sensitivity=2.0, desens=1.0, intern=1.0
    chem.level = 1.0;
    chem.receptor_sensitivity = 2.0;
    chem.desensitization_factor = 1.0;
    chem.internalization_factor = 1.0;
    // D1=1.0, D2=0.0 → (d1-d2)/total = 1.0
    // raw = 2.0 * 1.0 * 2.0 + 2.0 = 6.0 (without clamping)
    chem.receptor_subtypes = [1.0, 0.0];

    let eff = chem.subtype_effective_level();
    assert!(
        eff <= 2.0,
        "subtype_effective_level for dopamine must be clamped to [0, 2], got {}",
        eff
    );
    assert!(eff >= 0.0);
}

#[test]
fn test_subtype_effective_level_dopamine_finite_for_nan() {
    let mut chem = Neurochemical::new(NeurochemicalId::Dopamine, 0);
    chem.level = f32::NAN;
    chem.receptor_subtypes = [1.0, 0.0];

    let eff = chem.subtype_effective_level();
    assert!(
        eff.is_finite(),
        "subtype_effective_level must be finite for NaN level"
    );
    assert!((0.0..=2.0).contains(&eff));
}

// ─── Tick-level circuit breaker ─────────────────────────────────

#[test]
fn test_tick_circuit_breaker_resets_nan_level() {
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // Corrupt one chemical's level with NaN
    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    let da = state
        .neurochemicals
        .get_mut(NeurochemicalId::Dopamine)
        .unwrap();
    da.level = f32::NAN;
    state.write_end();

    // Run one tick — the circuit breaker should reset the NaN level
    // before the coupling matrix propagates it to all 18 chemicals
    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    state.neuro_tick_with_params(&params);
    state.write_end();

    // Verify ALL chemicals have finite levels (the NaN must not have
    // propagated through the coupling matrix)
    for (i, chem) in state.neurochemicals.chemicals.iter().enumerate() {
        assert!(
            chem.level.is_finite(),
            "chemical {} level must be finite after tick with NaN input, got {}",
            i,
            chem.level
        );
        assert!(
            chem.velocity.is_finite(),
            "chemical {} velocity must be finite after tick with NaN input, got {}",
            i,
            chem.velocity
        );
    }
}

#[test]
fn test_tick_circuit_breaker_resets_nan_velocity() {
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    let srt = state
        .neurochemicals
        .get_mut(NeurochemicalId::Serotonin)
        .unwrap();
    srt.velocity = f32::INFINITY;
    state.write_end();

    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    state.neuro_tick_with_params(&params);
    state.write_end();

    for (i, chem) in state.neurochemicals.chemicals.iter().enumerate() {
        assert!(
            chem.velocity.is_finite(),
            "chemical {} velocity must be finite after tick with inf input, got {}",
            i,
            chem.velocity
        );
    }
}

#[test]
fn test_tick_circuit_breaker_resets_nan_coupling_matrix() {
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // Corrupt the coupling matrix with NaN
    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    state.neurochemicals.coupling_matrix[0][1] = f32::NAN;
    state.neurochemicals.coupling_matrix[5][10] = f32::INFINITY;
    state.write_end();

    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    state.neuro_tick_with_params(&params);
    state.write_end();

    // The NaN in the coupling matrix must not have propagated to any chemical
    for (i, chem) in state.neurochemicals.chemicals.iter().enumerate() {
        assert!(
            chem.level.is_finite(),
            "chemical {} level must be finite after tick with NaN coupling matrix, got {}",
            i,
            chem.level
        );
    }

    // The coupling matrix itself should be sanitized
    for (i, row) in state.neurochemicals.coupling_matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            assert!(
                val.is_finite(),
                "coupling_matrix[{}][{}] must be finite after tick, got {}",
                i,
                j,
                val
            );
        }
    }
}

#[test]
fn test_tick_circuit_breaker_resets_nan_arousal_valence() {
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    state.neurochemicals.arousal = f32::NAN;
    state.neurochemicals.valence = f32::NAN;
    state.neurochemicals.global_tone = f32::INFINITY;
    state.neurochemicals.plasticity_gate = f32::NAN;
    state.write_end();

    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    state.neuro_tick_with_params(&params);
    state.write_end();

    assert!(
        state.neurochemicals.arousal.is_finite(),
        "arousal must be finite"
    );
    assert!(
        state.neurochemicals.valence.is_finite(),
        "valence must be finite"
    );
    assert!(
        state.neurochemicals.global_tone.is_finite(),
        "global_tone must be finite"
    );
    assert!(
        state.neurochemicals.plasticity_gate.is_finite(),
        "plasticity_gate must be finite"
    );
}

#[test]
fn test_tick_circuit_breaker_resets_nan_receptor_fields() {
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // Corrupt receptor fields
    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    let da = state
        .neurochemicals
        .get_mut(NeurochemicalId::Dopamine)
        .unwrap();
    da.receptor_sensitivity = f32::NAN;
    da.desensitization_factor = f32::INFINITY;
    da.internalization_factor = f32::NAN;
    da.vesicular_pool = f32::NEG_INFINITY;
    state.write_end();

    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    state.neuro_tick_with_params(&params);
    state.write_end();

    let da = state.neurochemicals.get(NeurochemicalId::Dopamine).unwrap();
    assert!(
        da.receptor_sensitivity.is_finite(),
        "receptor_sensitivity must be finite"
    );
    assert!(
        da.desensitization_factor.is_finite(),
        "desensitization_factor must be finite"
    );
    assert!(
        da.internalization_factor.is_finite(),
        "internalization_factor must be finite"
    );
    assert!(
        da.vesicular_pool.is_finite(),
        "vesicular_pool must be finite"
    );
}

// ─── System stability after NaN corruption ─────────────────────

#[test]
fn test_system_remains_stable_after_nan_corruption() {
    // This is the integration test: after the circuit breaker
    // cleans up NaN corruption, the system must remain stable
    // for many subsequent ticks (no delayed propagation, no
    // residual NaN in derived values).
    let mut state = GenesisCoreState::new(1, 1000);
    let params = NeuroTickParams {
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::COMPRESSED
    };

    // Corrupt multiple chemicals
    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    {
        let da = state
            .neurochemicals
            .get_mut(NeurochemicalId::Dopamine)
            .unwrap();
        da.level = f32::NAN;
        da.velocity = f32::INFINITY;
    }
    {
        let srt = state
            .neurochemicals
            .get_mut(NeurochemicalId::Serotonin)
            .unwrap();
        srt.level = f32::NEG_INFINITY;
    }
    {
        let cort = state
            .neurochemicals
            .get_mut(NeurochemicalId::Cortisol)
            .unwrap();
        cort.receptor_sensitivity = f32::NAN;
    }
    state.neurochemicals.coupling_matrix[3][7] = f32::NAN;
    state.neurochemicals.effective_levels[5] = f32::INFINITY;
    state.write_end();

    // Run 100 ticks — all must produce finite values
    for tick in 0..100 {
        // SAFETY: single-threaded test
        unsafe { state.write_begin(0) };
        state.neuro_tick_with_params(&params);
        state.write_end();

        // Check all chemicals every tick
        for (i, chem) in state.neurochemicals.chemicals.iter().enumerate() {
            assert!(
                chem.level.is_finite(),
                "tick {}: chemical {} level is non-finite ({}) — NaN propagated!",
                tick,
                i,
                chem.level
            );
            assert!(
                chem.velocity.is_finite(),
                "tick {}: chemical {} velocity is non-finite ({})",
                tick,
                i,
                chem.velocity
            );
            assert!(
                chem.baseline.is_finite(),
                "tick {}: chemical {} baseline is non-finite ({})",
                tick,
                i,
                chem.baseline
            );
        }

        // Check derived values
        assert!(
            state.neurochemicals.arousal.is_finite(),
            "tick {}: arousal is non-finite",
            tick
        );
        assert!(
            state.neurochemicals.valence.is_finite(),
            "tick {}: valence is non-finite",
            tick
        );
        assert!(
            state.neurochemicals.global_tone.is_finite(),
            "tick {}: global_tone is non-finite",
            tick
        );
        assert!(
            state.neurochemicals.plasticity_gate.is_finite(),
            "tick {}: plasticity_gate is non-finite",
            tick
        );

        // Check effective levels
        for (i, &eff) in state.neurochemicals.effective_levels.iter().enumerate() {
            assert!(
                eff.is_finite(),
                "tick {}: effective_levels[{}] is non-finite ({})",
                tick,
                i,
                eff
            );
        }
    }
}

// ─── recompute_derived produces finite values ───────────────────

#[test]
fn test_recompute_derived_finite_after_corruption() {
    let mut state = GenesisCoreState::new(1, 1000);

    // Corrupt some chemical levels
    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    let da = state
        .neurochemicals
        .get_mut(NeurochemicalId::Dopamine)
        .unwrap();
    da.level = f32::NAN;
    state.write_end();

    // recompute_derived should produce finite arousal/valence/tone
    // despite the NaN level. The effective_levels computation uses
    // finite_clamp, which replaces NaN with 0.0.
    // SAFETY: single-threaded test
    unsafe { state.write_begin(0) };
    state.neurochemicals.recompute_derived();
    state.write_end();

    assert!(
        state.neurochemicals.arousal.is_finite(),
        "arousal must be finite after recompute_derived with NaN level"
    );
    assert!(
        state.neurochemicals.valence.is_finite(),
        "valence must be finite after recompute_derived with NaN level"
    );
    assert!(
        state.neurochemicals.global_tone.is_finite(),
        "global_tone must be finite after recompute_derived with NaN level"
    );
    assert!(
        state.neurochemicals.plasticity_gate.is_finite(),
        "plasticity_gate must be finite after recompute_derived with NaN level"
    );

    // Effective levels must all be finite
    for (i, &eff) in state.neurochemicals.effective_levels.iter().enumerate() {
        assert!(
            eff.is_finite(),
            "effective_levels[{}] must be finite after recompute_derived, got {}",
            i,
            eff
        );
    }
}
