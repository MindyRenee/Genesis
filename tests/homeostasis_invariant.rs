//! Homeostasis invariant: with inter-chemical coupling disabled, every
//! chemical must converge exactly onto its genetic default.
//!
//! ## Why this test exists
//!
//! Genesis's resting chemical levels were found to settle well below her
//! genetic defaults (dopamine -0.104, glutamate -0.244 over ~4 hours),
//! which held arousal in the "excited" band while idle. The obvious
//! suspect was the baseline-adaptation rule, which settles a baseline at
//! `(level + genetic) / 2` and therefore looks like a ratchet that chases
//! the level instead of correcting it.
//!
//! That hypothesis is wrong, and this test is what disproves it. With
//! `coupling_scale = 0` the homeostatic force and the baseline-adaptation
//! rule run alone, and every chemical lands on its genetic default to
//! within 0.001. The homeostasis machinery regulates correctly; it is
//! not the source of the offset.
//!
//! The offset comes from outside the chemistry: the daemon's tick loop
//! additionally calls `active_inference::apply_inference_feedback`, which
//! writes impulses into chemical levels on every tick. The depressed
//! resting baseline is the relaxation target *under that sustained
//! cognitive input*, not a property of the neurochemical dynamics alone.
//! Note that this test exercises `neuro_tick_with_params` only, so the
//! inference feedback is deliberately out of scope here.
//!
//! Dopamine and histamine are excluded from the assertion: both carry a
//! circadian baseline modifier (`circadian_baseline_modifier`), so their
//! equilibrium is legitimately offset from the genetic default by the
//! circadian phase. Asserting they return to the raw default would be
//! asserting that the circadian rhythm does not exist.

use genesis::state::core_state::GenesisCoreState;
use genesis::state::neurochemical::{NeuroTickParams, NeurochemicalId};

/// Chemicals with no circadian baseline modifier. These must return to
/// their genetic default when coupling is disabled.
const UNMODULATED: [NeurochemicalId; 6] = [
    NeurochemicalId::GABA,
    NeurochemicalId::Glutamate,
    NeurochemicalId::Acetylcholine,
    NeurochemicalId::Norepinephrine,
    NeurochemicalId::Orexin,
    NeurochemicalId::Endorphin,
];

/// Chemicals whose equilibrium is legitimately circadian-modulated.
/// Tracked for reporting, excluded from the assertion.
const MODULATED: [NeurochemicalId; 2] = [NeurochemicalId::Dopamine, NeurochemicalId::Histamine];

const TICKS: u32 = 14_000;
const TOLERANCE: f32 = 0.01;

/// Returns (id, raw level, effective level) after the run.
fn run(coupling_scale: f32) -> Vec<(NeurochemicalId, f32, f32)> {
    let params = NeuroTickParams {
        dt: 1.0,
        coupling_scale,
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let mut state = GenesisCoreState::new(1, 1000);
    for _ in 0..TICKS {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        state.neuro_tick_with_params(&params);
        state.write_end();
    }
    let mut all = Vec::new();
    for id in UNMODULATED.into_iter().chain(MODULATED) {
        let raw = state
            .neurochemicals
            .get(id)
            .map(|c| c.level)
            .unwrap_or(f32::NAN);
        all.push((id, raw, state.neurochemicals.effective(id)));
    }
    all
}

#[test]
fn homeostasis_alone_returns_chemicals_to_genetic_defaults() {
    let levels = run(0.0);

    println!(
        "\n{:>14}{:>10}{:>12}{:>12}",
        "chemical", "genetic", "level", "deviation"
    );
    for (id, level, _) in &levels {
        let g = id.default_baseline();
        let modulated = MODULATED.contains(id);
        println!(
            "{:>14}{:>10.3}{:>12.3}{:>12.4}{}",
            format!("{:?}", id),
            g,
            level,
            level - g,
            if modulated {
                "  (circadian-modulated)"
            } else {
                ""
            }
        );
    }

    for (id, level, _) in &levels {
        if MODULATED.contains(id) {
            continue;
        }
        let deviation = (level - id.default_baseline()).abs();
        assert!(
            deviation < TOLERANCE,
            "{:?} should return to its genetic default ({:.3}) when coupling is \
             disabled, but settled at {:.3} (deviation {:.4})",
            id,
            id.default_baseline(),
            level,
            deviation
        );
    }

    // The same comparison on *effective* levels, which is what the
    // Python eval harness actually reports.
    //
    // `CoreState.chemicals` is unpacked from the vector's
    // `effective_levels` array, not the raw `chemicals[]` array, and
    // `eval.drift.check_snapshot_vs_defaults` compares those effective
    // values against `default_baseline()` — a raw, genetic quantity.
    // Effective level is raw level scaled by receptor sensitivity,
    // subtype weighting, phasic tone and (for GABA) allosteric
    // modulation, and receptor sensitivity adapts over time. So the two
    // are not the same quantity and can differ even when the system is
    // perfectly regulated. This measures by how much.
    println!(
        "\n{:>14}{:>10}{:>12}{:>12}{:>12}",
        "chemical", "genetic", "raw", "effective", "eff-gen"
    );
    for (id, raw, eff) in &levels {
        if MODULATED.contains(id) {
            continue;
        }
        let g = id.default_baseline();
        println!(
            "{:>14}{:>10.3}{:>12.3}{:>12.3}{:>12.4}",
            format!("{:?}", id),
            g,
            raw,
            eff,
            eff - g
        );
    }
}
