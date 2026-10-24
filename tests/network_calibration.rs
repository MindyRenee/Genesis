//! Calibration of the sleep-wake network's release targets against the data.
//!
//! The network's stage means are the inputs (measured from the source model
//! over days 8–10): wake, NREM and REM. Each release level is held at its
//! steady state under those inputs and compared with the level its data give.
//!
//! Sources for the targets:
//! - Acetylcholine: basal forebrain microdialysis, NREM 0.34, waking 0.58,
//!   REM 0.77 pmol/10 min (Vazquez & Baghdoyan, 2001).
//! - Serotonin: dorsal raphe, wake high, NREM reduced, REM near silent
//!   (Monti & Jantos, 2008; Sakai & Crochet, 2001). Levels 0.39 / 0.22 / 0.07.
//! - Norepinephrine: tracks LC firing, silent in REM (Aston-Jones & Bloom,
//!   1981). Proportional to the network's LC.
//! - GABA: cortex NREM highest, waking 39% lower, REM 63% lower
//!   (SLEEP 2012). Anchored to the chemistry's waking baseline of 0.50 (the
//!   genetic default), so NREM 0.82 and REM 0.30 follow from the ratios.

use genesis::state::neurochemical::{
    NetworkLevels, NeuroTickParams, NeurochemicalId, NeurochemicalVector,
};

const NREM: NetworkLevels = NetworkLevels {
    lc: 0.151,
    vlpo: 0.771,
    rem: 0.011,
};
const WAKE: NetworkLevels = NetworkLevels {
    lc: 0.963,
    vlpo: 0.021,
    rem: 0.0,
};
const REM: NetworkLevels = NetworkLevels {
    lc: 0.133,
    vlpo: 0.896,
    rem: 0.976,
};

/// Steady-state levels of (ACh, serotonin, NE, GABA) under fixed network input.
fn equilibrium(net: NetworkLevels) -> [f32; 4] {
    let params = NeuroTickParams {
        maturation_level: 1.0,
        network_levels: Some(net),
        ..Default::default()
    };
    let mut v = NeurochemicalVector::new(1);
    for _ in 0..300_000u32 {
        v.tick_with_params(&params);
    }
    [
        NeurochemicalId::Acetylcholine,
        NeurochemicalId::Serotonin,
        NeurochemicalId::Norepinephrine,
        NeurochemicalId::GABA,
    ]
    .map(|id| v.get(id).unwrap().level)
}

#[test]
fn steady_release_levels_match_the_calibrated_data() {
    let (n, w, r) = (equilibrium(NREM), equilibrium(WAKE), equilibrium(REM));
    let want = [
        ("ACh", [0.34, 0.58, 0.77]),
        ("serotonin", [0.22, 0.39, 0.07]),
        ("NE", [0.05, 0.30, 0.00]),
        ("GABA", [0.82, 0.50, 0.30]),
    ];
    // Histamine and orexin are wake-active and silent in sleep.
    let (hist_n, hist_w, hist_r) = (
        equilibrium_hist(NREM),
        equilibrium_hist(WAKE),
        equilibrium_hist(REM),
    );
    assert!(
        hist_n < 0.02 && hist_r < 0.02,
        "histamine in sleep: NREM {hist_n} REM {hist_r}"
    );
    assert!((hist_w - 0.30).abs() < 0.02, "histamine awake {hist_w}");
    for (k, (name, target)) in want.iter().enumerate() {
        for (state, got, goal) in [
            ("NREM", n[k], target[0]),
            ("wake", w[k], target[1]),
            ("REM", r[k], target[2]),
        ] {
            assert!(
                (got - goal).abs() < 0.02,
                "{name} in {state}: steady {got:.3}, calibrated {goal:.3}"
            );
        }
    }
}

fn equilibrium_hist(net: NetworkLevels) -> f32 {
    let params = NeuroTickParams {
        maturation_level: 1.0,
        network_levels: Some(net),
        ..Default::default()
    };
    let mut v = NeurochemicalVector::new(1);
    for _ in 0..300_000u32 {
        v.tick_with_params(&params);
    }
    v.get(NeurochemicalId::Histamine).unwrap().level
}

/// Arousal per stage: NREM is the lowest, REM moderate, waking highest, and
/// the NREM and REM bands match the ranges the scripted clamps enforced.
#[test]
fn arousal_orders_the_stages_and_sits_in_the_bands() {
    let arousal = |net: NetworkLevels| {
        let params = NeuroTickParams {
            maturation_level: 1.0,
            network_levels: Some(net),
            ..Default::default()
        };
        let mut v = NeurochemicalVector::new(1);
        for _ in 0..300_000u32 {
            v.tick_with_params(&params);
        }
        v.arousal
    };
    let (n, w, r) = (arousal(NREM), arousal(WAKE), arousal(REM));
    assert!(
        n < r && r < w,
        "arousal ordering: NREM {n} REM {r} wake {w}"
    );
    assert!(n <= 0.25, "NREM arousal {n} should be low");
    assert!(
        (0.35..=0.55).contains(&r),
        "REM arousal {r} should be moderate"
    );
}

#[test]
fn release_ordering_matches_the_data() {
    let (n, w, r) = (equilibrium(NREM), equilibrium(WAKE), equilibrium(REM));
    // ACh: REM > waking > NREM.
    assert!(r[0] > w[0] && w[0] > n[0], "ACh ordering {r:?} {w:?} {n:?}");
    // Serotonin: waking > NREM > REM (near silent).
    assert!(w[1] > n[1] && n[1] > r[1], "serotonin ordering");
    // GABA: NREM > waking > REM.
    assert!(n[3] > w[3] && w[3] > r[3], "GABA ordering");
    // NE: silent in REM.
    assert!(r[2] < 0.02, "NE should be silent in REM, got {}", r[2]);
}

/// Process S clears when the *network* is asleep — the body's own state,
/// not the mind's zone, is the sleep signal. When the network has crossed
/// into sleep while the zone still reads awake (idle-deferral, a stalled
/// mind, any desync), adenosine would otherwise keep accumulating during
/// actual sleep.
#[test]
fn adenosine_clears_when_the_network_sleeps() {
    let tick = |net: NetworkLevels| {
        let mut v = NeurochemicalVector::new(1);
        v.get_mut(NeurochemicalId::Adenosine).unwrap().level = 0.80;
        let params = NeuroTickParams {
            maturation_level: 1.0,
            network_levels: Some(net),
            ..NeuroTickParams::COMPRESSED
        };
        for _ in 0..600u32 {
            v.tick_with_params(&params);
        }
        v.get(NeurochemicalId::Adenosine).unwrap().level
    };
    let asleep = tick(NREM);
    let awake = tick(WAKE);
    assert!(
        asleep < 0.80,
        "adenosine must clear while the network sleeps (zone still awake): {asleep}"
    );
    assert!(
        awake >= asleep,
        "adenosine must not clear faster while the network is awake: asleep {asleep} awake {awake}"
    );
}
