//! Probe: do the field spectrum's absolute band powers differentiate
//! the sleep-wake stages? Each stage gets a fresh organism state at
//! its calibrated chemical equilibrium and phase, then the field
//! integrates ~2 minutes of observations; the spectrum signature is
//! read at the end.

use genesis::field::FieldRuntime;
use genesis::state::core_state::GenesisCoreState;
use genesis::state::neurochemical::NeurochemicalId;
use genesis::state::zones::MentalPhase;

fn stage_state(phase: MentalPhase, levels: &[(NeurochemicalId, f32)]) -> GenesisCoreState {
    let mut s = GenesisCoreState::new(1, 0);
    s.zones.sync_phase(phase);
    for (id, lv) in levels {
        if let Some(c) = s.neurochemicals.get_mut(*id) {
            c.level = *lv;
        }
    }
    s
}

fn run_stage(name: &str, phase: MentalPhase, levels: &[(NeurochemicalId, f32)]) {
    let mut state = stage_state(phase, levels);
    let mut rt = FieldRuntime::new().expect("field runtime");
    // ~120 s of observations at the daemon's 5 Hz cadence.
    for _ in 0..600 {
        let _ = rt.observe(&state, 0.2, 0.0);
        // Keep the phase pinned — nothing else resyncs it here.
        state.zones.sync_phase(phase);
    }
    let sig = rt.spectrum();
    let bands = [
        "infraslow",
        "delta",
        "theta",
        "alpha",
        "sigma",
        "beta1",
        "beta2",
        "beta3",
        "gamma",
        "epsilon",
    ];
    println!("== {name} (warm={})", sig.warm);
    println!("   flux_power:   {:?}", sig.flux_power);
    println!("   tension_power:{:?}", sig.tension_power);
    println!(
        "   carrier_freq={:?} speed={:?} cv(f/t)={:.3}/{:.3}",
        sig.carrier_freq_hz, sig.consensus_speed, sig.flux.cv, sig.tension.cv
    );
    // Log-scaled band powers for readability.
    let fp: Vec<String> = sig
        .flux_power
        .iter()
        .enumerate()
        .map(|(i, p)| format!("{}={:.2e}", bands[i], p))
        .collect();
    let tp: Vec<String> = sig
        .tension_power
        .iter()
        .enumerate()
        .map(|(i, p)| format!("{}={:.2e}", bands[i], p))
        .collect();
    println!("   flux: {}", fp.join(" "));
    println!("   tens: {}", tp.join(" "));
}

#[test]
fn spectrum_bands_by_sleep_stage() {
    use NeurochemicalId as C;
    // Calibrated stage equilibria (tests/network_calibration.rs +
    // the histamine/orexin probe).
    let wake = [
        (C::Acetylcholine, 0.58),
        (C::Serotonin, 0.39),
        (C::Norepinephrine, 0.30),
        (C::GABA, 0.50),
        (C::Histamine, 0.30),
        (C::Orexin, 0.30),
    ];
    let nrem = [
        (C::Acetylcholine, 0.34),
        (C::Serotonin, 0.22),
        (C::Norepinephrine, 0.05),
        (C::GABA, 0.82),
        (C::Histamine, 0.0),
        (C::Orexin, 0.014),
    ];
    let rem = [
        (C::Acetylcholine, 0.77),
        (C::Serotonin, 0.07),
        (C::Norepinephrine, 0.0),
        (C::GABA, 0.30),
        (C::Histamine, 0.0),
        (C::Orexin, 0.010),
    ];
    run_stage("WAKE", MentalPhase::Alert, &wake);
    run_stage("NREM", MentalPhase::NREM, &nrem);
    run_stage("REM", MentalPhase::REM, &rem);
}
