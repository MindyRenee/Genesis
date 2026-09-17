//! Long-run stability test: 10000 ticks to verify no runaway dynamics,
//! phase flickering, or state corruption after the audit fixes.
//!
//! Run with: cargo run --example stability_test

use genesis::state::*;

fn main() {
    let mut state = GenesisCoreState::new(1, 0);
    let params = NeuroTickParams::COMPRESSED;

    let mut phases = Vec::new();
    let mut arousal_min = 1.0f32;
    let mut arousal_max = 0.0f32;
    let mut valence_min = 1.0f32;
    let mut valence_max = -1.0f32;
    let mut cort_max = 0.0f32;
    let mut da_max = 0.0f32;
    let mut phase_changes = 0u32;
    let mut prev_phase = state.zones.phase();

    for tick in 0..10000 {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(tick as u64 * 100) };
        state.neuro_tick_with_params(&params);
        state.write_end();

        let phase = state.zones.phase();
        if phase != prev_phase {
            phase_changes += 1;
            prev_phase = phase;
        }

        let arousal = state.neurochemicals.arousal;
        let valence = state.neurochemicals.valence;
        let cort = state.neurochemicals.effective_levels[NeurochemicalId::Cortisol as usize];
        let da = state.neurochemicals.effective_levels[NeurochemicalId::Dopamine as usize];

        arousal_min = arousal_min.min(arousal);
        arousal_max = arousal_max.max(arousal);
        valence_min = valence_min.min(valence);
        valence_max = valence_max.max(valence);
        cort_max = cort_max.max(cort);
        da_max = da_max.max(da);

        if tick % 1000 == 0 {
            phases.push((tick, phase, arousal, valence, cort, da));
        }
    }

    println!("=== 10000-tick stability test (COMPRESSED params) ===");
    println!();
    for (tick, phase, arousal, valence, cort, da) in &phases {
        println!(
            "tick {:5}: phase={:?}  arousal={:.3}  valence={:.3}  cort={:.3}  da={:.3}",
            tick, phase, arousal, valence, cort, da
        );
    }
    println!();
    println!("arousal range:  [{:.3}, {:.3}]", arousal_min, arousal_max);
    println!("valence range:  [{:.3}, {:.3}]", valence_min, valence_max);
    println!("cortisol max:   {:.3}", cort_max);
    println!("dopamine max:   {:.3}", da_max);
    println!("phase changes:  {}", phase_changes);
    println!();

    // Check for runaway (effective levels can exceed 1.0 due to
    // receptor upregulation and phasic boosts — this is by design)
    let mut warnings = 0;
    if cort_max > 0.98 {
        println!("WARNING: cortisol runaway detected (max={:.3})", cort_max);
        warnings += 1;
    }
    if arousal_max - arousal_min < 0.01 {
        println!("WARNING: arousal is flat (no dynamics)");
        warnings += 1;
    }
    if phase_changes > 100 {
        println!(
            "WARNING: excessive phase flickering ({} changes)",
            phase_changes
        );
        warnings += 1;
    }

    // Verify state integrity
    assert!(
        state.verify().is_ok(),
        "state should be valid after 10000 ticks"
    );
    println!("State verification: OK (checksum {:#x})", state.checksum);

    if warnings == 0 {
        println!();
        println!("All stability checks PASSED.");
    } else {
        println!();
        println!("{} warning(s) — review above.", warnings);
    }
}
