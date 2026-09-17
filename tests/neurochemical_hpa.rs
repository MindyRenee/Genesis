//! Regression tests for the HPA axis cascade and global_tone IPC clamp.
//!
//! ## HPA cascade maturation gating
//!
//! The Rust HPA cascade (CRH → ACTH → cortisol) is maturation-gated
//! via the stress hyporesponsive period (SHRP). At maturation 0.0,
//! the cascade is fully dormant — CRH impulses must NOT produce
//! cortisol. At maturation 1.0, the cascade is fully online.
//!
//! This is the architectural contract that the Python emotional
//! regulator relies on: it sends CRH impulses (the stress signal),
//! and the Rust cascade handles cortisol production with proper
//! maturation gating. If the cascade fails to gate at maturation 0,
//! the SHRP protection is broken and the developing brain is exposed
//! to cortisol's neurotoxic effects.
//!
//! ## global_tone IPC clamp
//!
//! `global_tone` is the mean of effective levels, each in [0, 2].
//! The Rust internal state clamps it to [0, 2]. The IPC
//! GET_NEURO_SUMMARY handler must use the same range — clamping to
//! [-1, 1] truncates values above 1.0 in extreme states, causing
//! the Python cognitive mind to see a different value than the Rust
//! daemon stores.

use genesis::state::neurochemical::{NeuroTickParams, NeurochemicalId, NeurochemicalVector};
use genesis::state::sanitize::finite_clamp;

// ─── HPA cascade maturation gating ─────────────────────────────

/// At maturation 0.0 (newborn / SHRP), CRH impulses must NOT
/// produce cortisol. The entire cascade is dormant.
#[test]
fn test_hpa_cascade_dormant_at_zero_maturation() {
    let mut neuro = NeurochemicalVector::new(0);

    // Drive CRH up with a strong impulse
    neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.5, 0);
    neuro.recompute_derived();

    // Tick with maturation = 0.0 (SHRP)
    let params = NeuroTickParams {
        maturation_level: 0.0,
        ..NeuroTickParams::DEFAULT
    };

    // Run several ticks to allow the cascade to propagate
    for _ in 0..100 {
        neuro.tick_with_params(&params);
    }

    let cortisol = neuro.chemicals[NeurochemicalId::Cortisol as usize].level;
    let acth = neuro.acth_level;

    // CRH may be elevated (the impulse was applied), but ACTH and
    // cortisol must remain at zero — the cascade is gated by
    // maturation.
    assert!(acth < 0.01, "ACTH must be ~0 at maturation=0, got {acth}");
    assert!(
        cortisol < 0.01,
        "Cortisol must be ~0 at maturation=0 (SHRP), got {cortisol}"
    );
}

/// At maturation 1.0 (mature), CRH impulses MUST produce cortisol
/// through the cascade. This verifies the cascade is functional.
#[test]
fn test_hpa_cascade_active_at_full_maturation() {
    let mut neuro = NeurochemicalVector::new(0);

    // Drive CRH up with a strong impulse
    neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.5, 0);
    neuro.recompute_derived();

    // Tick with maturation = 1.0 (fully mature)
    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0, // deterministic
        ..NeuroTickParams::DEFAULT
    };

    // Run enough ticks for the cascade to propagate:
    // CRH → ACTH (~5s at rate 0.2/s) → cortisol (~100s at rate 0.01/s)
    // At 10 Hz, 100 seconds = 1000 ticks. Run 2000 to be safe.
    for _ in 0..2000 {
        // Re-apply CRH impulse periodically to sustain the signal
        neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.01, 0);
        neuro.tick_with_params(&params);
    }

    let cortisol = neuro.chemicals[NeurochemicalId::Cortisol as usize].level;
    let acth = neuro.acth_level;

    // ACTH should be elevated (CRH drives it)
    assert!(
        acth > 0.01,
        "ACTH must be elevated at maturation=1.0 with CRH drive, got {acth}"
    );
    // Cortisol should be elevated (ACTH drives it)
    assert!(
        cortisol > 0.01,
        "Cortisol must be elevated at maturation=1.0 with CRH drive, got {cortisol}"
    );
}

/// At maturation 0.5 (partial), the cascade should be partially
/// active — cortisol production is attenuated but not zero.
#[test]
fn test_hpa_cascade_partial_at_half_maturation() {
    let mut neuro_half = NeurochemicalVector::new(0);
    let mut neuro_full = NeurochemicalVector::new(0);

    // Drive CRH up in both
    neuro_half.apply_impulse_capped(NeurochemicalId::CRH, 0.5, 0);
    neuro_full.apply_impulse_capped(NeurochemicalId::CRH, 0.5, 0);
    neuro_half.recompute_derived();
    neuro_full.recompute_derived();

    let params_half = NeuroTickParams {
        maturation_level: 0.5,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let params_full = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };

    for _ in 0..2000 {
        neuro_half.apply_impulse_capped(NeurochemicalId::CRH, 0.01, 0);
        neuro_full.apply_impulse_capped(NeurochemicalId::CRH, 0.01, 0);
        neuro_half.tick_with_params(&params_half);
        neuro_full.tick_with_params(&params_full);
    }

    let cortisol_half = neuro_half.chemicals[NeurochemicalId::Cortisol as usize].level;
    let cortisol_full = neuro_full.chemicals[NeurochemicalId::Cortisol as usize].level;

    // Half maturation should produce less cortisol than full
    assert!(
        cortisol_half < cortisol_full,
        "Half maturation ({cortisol_half}) should produce less cortisol than full ({cortisol_full})"
    );
    // But not zero — the cascade is partially active
    assert!(
        cortisol_half > 0.001,
        "Half maturation should produce some cortisol, got {cortisol_half}"
    );
}

// ─── global_tone IPC clamp range ───────────────────────────────

/// The IPC GET_NEURO_SUMMARY handler must clamp global_tone to
/// [0, 2], matching the Rust internal state's range. The previous
/// clamp of [-1, 1] truncated values above 1.0 in extreme states.
///
/// This test verifies the clamp range directly — the IPC handler
/// uses `finite_clamp(state.neurochemicals.global_tone, 0.0, 2.0)`,
/// which must preserve values in [0, 2] and reject NaN.
#[test]
fn test_global_tone_ipc_clamp_preserves_full_range() {
    // Verify finite_clamp with [0, 2] preserves values in the full range
    assert_eq!(finite_clamp(0.0, 0.0, 2.0), 0.0);
    assert_eq!(finite_clamp(0.5, 0.0, 2.0), 0.5);
    assert_eq!(finite_clamp(1.0, 0.0, 2.0), 1.0);
    assert_eq!(finite_clamp(1.5, 0.0, 2.0), 1.5);
    assert_eq!(finite_clamp(2.0, 0.0, 2.0), 2.0);
    // Above 2.0 clamps to 2.0
    assert_eq!(finite_clamp(3.0, 0.0, 2.0), 2.0);
    // Below 0.0 clamps to 0.0
    assert_eq!(finite_clamp(-0.5, 0.0, 2.0), 0.0);
    // NaN returns 0.0 (the min)
    assert_eq!(finite_clamp(f32::NAN, 0.0, 2.0), 0.0);
}

/// Verify that global_tone can actually exceed 1.0 in the Rust
/// internal state (the scenario the old IPC clamp would truncate).
/// This happens when effective levels are high across most chemicals.
#[test]
fn test_global_tone_can_exceed_one() {
    let mut neuro = NeurochemicalVector::new(0);

    // Push all chemicals to high levels with high receptor sensitivity
    for i in 0..18 {
        let chem = &mut neuro.chemicals[i];
        chem.level = 0.9;
        chem.receptor_sensitivity = 1.5;
        chem.desensitization_factor = 1.0;
        chem.internalization_factor = 1.0;
    }
    neuro.recompute_derived();

    // global_tone is the mean of effective levels
    let gt = neuro.global_tone;
    assert!(
        gt > 1.0,
        "global_tone should exceed 1.0 with high levels + sensitivity, got {gt}"
    );
    assert!(gt <= 2.0, "global_tone must not exceed 2.0, got {gt}");
}

// ─── CRH impulse routing through the cascade ───────────────────

/// Verify that CRH impulses (from the Python regulator or
/// interoception) route through the HPA cascade to produce
/// cortisol, and that the cascade's negative feedback prevents
/// runaway.
#[test]
fn test_crh_impulse_routes_through_cascade() {
    let mut neuro = NeurochemicalVector::new(0);

    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };

    // Phase 1: Apply CRH impulse and verify cortisol rises
    neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.3, 0);
    for _ in 0..500 {
        neuro.tick_with_params(&params);
    }

    let cortisol_peak = neuro.chemicals[NeurochemicalId::Cortisol as usize].level;
    assert!(
        cortisol_peak > 0.01,
        "Cortisol should rise after CRH impulse, got {cortisol_peak}"
    );

    // Phase 2: Stop CRH impulse and verify cortisol decays
    // (negative feedback + clearance)
    for _ in 0..2000 {
        neuro.tick_with_params(&params);
    }

    let cortisol_after = neuro.chemicals[NeurochemicalId::Cortisol as usize].level;
    assert!(
        cortisol_after < cortisol_peak,
        "Cortisol should decay after CRH stops: peak={cortisol_peak}, after={cortisol_after}"
    );
}

/// Verify that the cortisol hard clamp prevents runaway even under
/// sustained CRH drive.
#[test]
fn test_cortisol_hard_clamp_under_sustained_crh() {
    let mut neuro = NeurochemicalVector::new(0);

    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        cortisol_max: 0.80,
        ..NeuroTickParams::DEFAULT
    };

    // Sustained CRH drive
    for _ in 0..5000 {
        neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.05, 0);
        neuro.tick_with_params(&params);
    }

    let cortisol = neuro.chemicals[NeurochemicalId::Cortisol as usize].level;
    assert!(
        cortisol <= 0.80 + 0.001,
        "Cortisol must not exceed hard clamp 0.80, got {cortisol}"
    );
}
