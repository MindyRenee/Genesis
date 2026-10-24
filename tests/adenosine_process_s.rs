//! Process S regression tests — the adenosine homeostat.
//!
//! Sleep staging downstream of the daemon reads `sleep_pressure`,
//! which is adenosine normalized over its resting range. N2 needs
//! pressure ≥ 0.15 (adenosine ≥ ~0.31) and N3 needs ≥ 0.38
//! (adenosine ≥ ~0.49). These tests pin the two invariants that make
//! those stages reachable:
//!
//! 1. A waking day actually builds the pressure — the dedicated
//!    accumulation mechanism is not starved.
//! 2. External impulses can add load but can never subtract it —
//!    Process S is one-way; only glymphatic clearance (sleep) and
//!    offline rest may drain it. Periodic "alertness" impulses from
//!    the mind once erased the level far faster than the ~0.055/h
//!    mechanism could rebuild it, pinning the level at its floor and
//!    leaving every NREM episode staged N1.

use genesis::state::neurochemical::{
    NetworkLevels, NeuroTickParams, NeurochemicalId, NeurochemicalVector,
};

/// 36_000 ticks of 100 ms ≈ one simulated hour.
const TICKS_PER_HOUR: u32 = 36_000;

const WAKE: NetworkLevels = NetworkLevels {
    lc: 0.963,
    vlpo: 0.021,
    rem: 0.0,
};
const NREM: NetworkLevels = NetworkLevels {
    lc: 0.151,
    vlpo: 0.771,
    rem: 0.011,
};

fn params(network: NetworkLevels) -> NeuroTickParams {
    NeuroTickParams {
        maturation_level: 1.0,
        network_levels: Some(network),
        ..Default::default()
    }
}

fn adenosine(v: &NeurochemicalVector) -> f32 {
    v.get(NeurochemicalId::Adenosine).unwrap().level
}

#[test]
fn half_day_of_wake_reaches_n2_pressure() {
    // Six waking hours from the resting floor must cross the N2
    // staging threshold (adenosine 0.3125 ≡ pressure 0.15). The
    // saturating accumulation (U=0.95, r≈0.055/h before activity
    // coupling) reaches ~0.45 at 6 h, so this bound has real margin —
    // a failure means the homeostat is starved, not unlucky.
    let mut v = NeurochemicalVector::new(1);
    let wake = params(WAKE);
    for _ in 0..6 * TICKS_PER_HOUR {
        v.tick_with_params(&wake);
    }
    assert!(
        adenosine(&v) >= 0.3125,
        "6 h of wake must reach N2-range pressure; adenosine={:.3}",
        adenosine(&v)
    );
}

#[test]
fn waking_day_reaches_n3_pressure() {
    // A full 16-hour waking day must reach deep-sleep pressure
    // (adenosine 0.485 ≡ pressure 0.38).
    let mut v = NeurochemicalVector::new(1);
    let wake = params(WAKE);
    for _ in 0..16 * TICKS_PER_HOUR {
        v.tick_with_params(&wake);
    }
    assert!(
        adenosine(&v) >= 0.485,
        "16 h of wake must reach N3-range pressure; adenosine={:.3}",
        adenosine(&v)
    );
}

#[test]
fn nrem_clears_pressure_toward_floor() {
    // Build real pressure, then an hour of NREM must reduce it —
    // glymphatic clearance is exponential toward the 0.20 floor, so
    // the level falls strictly while never undershooting it.
    let mut v = NeurochemicalVector::new(1);
    let wake = params(WAKE);
    for _ in 0..16 * TICKS_PER_HOUR {
        v.tick_with_params(&wake);
    }
    let built = adenosine(&v);
    let nrem = params(NREM);
    for _ in 0..TICKS_PER_HOUR {
        v.tick_with_params(&nrem);
    }
    let after_1h = adenosine(&v);
    assert!(
        after_1h < built,
        "NREM must clear adenosine: before={built:.3} after={after_1h:.3}"
    );
    assert!(
        after_1h >= 0.19,
        "clearance must not undershoot the resting floor: {after_1h:.3}"
    );
}

#[test]
fn adenosine_impulses_are_one_way() {
    // External impulses may add metabolic load (the mind's heartbeat
    // reports its compute work this way) but must never drain the
    // homeostat — clearance is owned by sleep and offline rest.
    let mut v = NeurochemicalVector::new(1);
    let adn = NeurochemicalId::Adenosine;

    let baseline = adenosine(&v);
    v.apply_impulse_capped(adn, 0.1, 0);
    let raised = adenosine(&v);
    assert!(
        raised > baseline,
        "positive adenosine impulse (metabolic load) must apply: {baseline:.3} -> {raised:.3}"
    );

    let velocity_before = v.get(adn).unwrap().velocity;
    v.apply_impulse_capped(adn, -0.5, 1);
    assert_eq!(
        adenosine(&v),
        raised,
        "negative adenosine impulse must be ignored entirely"
    );
    // No velocity residue either — a rejected impulse must not leak
    // decay into the next ticks.
    assert_eq!(
        v.get(adn).unwrap().velocity,
        velocity_before,
        "negative impulse must not set adenosine velocity"
    );

    // Control: ordinary chemicals still accept negative impulses.
    let da = NeurochemicalId::Dopamine;
    let d0 = v.get(da).unwrap().level;
    v.apply_impulse_capped(da, -0.2, 2);
    assert!(
        v.get(da).unwrap().level < d0,
        "control impulse on dopamine must still apply"
    );
}
