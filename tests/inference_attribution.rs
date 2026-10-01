//! Attribution: how much of the resting chemistry is driven by the
//! active-inference layer rather than by the neurochemical dynamics?
//!
//! `tests/homeostasis_invariant.rs` established that the homeostatic force
//! and the baseline-adaptation rule hold every chemical *exactly* on its
//! genetic default when inter-chemical coupling is disabled. Yet the
//! running daemon settles far below those defaults (dopamine -0.104,
//! glutamate -0.244 over ~4 hours).
//!
//! ## What this test shows
//!
//! The active-inference feedback is what holds Genesis's resting
//! chemistry below her genetic baseline. Measured on effective levels
//! (the quantity the eval harness reports and the one that drives
//! arousal):
//!
//! | chemical | genetic | chemistry only | + inference | daemon rest |
//! |---|---|---|---|---|
//! | dopamine    | 0.350 | 0.288 | 0.238 | ~0.25 |
//! | serotonin   | 0.400 | 0.398 | 0.304 | 0.304 |
//! | GABA        | 0.500 | 0.471 | 0.337 | 0.337 |
//! | glutamate   | 0.600 | 0.553 | 0.356 | 0.356 |
//!
//! Without the inference layer the chemistry sits essentially on its
//! defaults (worst deviation 0.047). With it, four chemicals are pushed
//! 0.05-0.20 further down. So the depression is not a property of the
//! neurochemical dynamics or of the coupling matrix — it comes from the
//! cognitive layer's feedback into the substrate.
//!
//! The inference layer acts selectively rather than globally: it drives
//! dopamine, serotonin, GABA and glutamate substantially, while
//! acetylcholine and norepinephrine are left essentially untouched.
//!
//! ## Resolved: the apparent sign mismatch was a units mismatch
//!
//! This test reads raw `.level`, while the Python client reads
//! `effective_levels` (see `tests/resting_production_path.rs`). The two
//! disagree, which made it look as though this harness pushed levels up
//! while the daemon pushed them down. Reported on the same quantity, the
//! production path and the daemon agree.
//!
//! The magnitude finding here stands: the inference feedback is the
//! dominant term in the chemistry, roughly an order of magnitude above
//! coupling and homeostasis, and large enough on its own to saturate a
//! chemical. What the raw-vs-effective split adds is that raw and
//! effective levels diverge substantially once receptor sensitivity has
//! adapted, so the depression in the *reported* level is larger than the
//! depression in the raw level.

use genesis::daemon::active_inference::{
    ActiveInferenceEngine, apply_inference_feedback, impulse_offset_to_rate,
};
use genesis::state::core_state::GenesisCoreState;
use genesis::state::neurochemical::{NEUROCHEMICAL_COUNT, NeuroTickParams, NeurochemicalId};
use std::time::{SystemTime, UNIX_EPOCH};

const TRACKED: [NeurochemicalId; 6] = [
    NeurochemicalId::Dopamine,
    NeurochemicalId::Serotonin,
    NeurochemicalId::GABA,
    NeurochemicalId::Glutamate,
    NeurochemicalId::Acetylcholine,
    NeurochemicalId::Norepinephrine,
];

const TICKS: u32 = 14_000;

fn effective_all(v: &GenesisCoreState) -> [f32; NEUROCHEMICAL_COUNT] {
    let mut out = [0.0f32; NEUROCHEMICAL_COUNT];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = v
            .neurochemicals
            .effective(NeurochemicalId::from_u8(i as u8));
    }
    out
}

/// Run the daemon's chemistry path. With `with_inference = false` the
/// active-inference cycle and its feedback are omitted, leaving only
/// `neuro_tick_with_params`.
type PolicyCounts = std::collections::HashMap<String, u32>;
type ImpulseTotals = std::collections::HashMap<u8, f32>;

fn run(
    with_inference: bool,
    policy_counts: &mut PolicyCounts,
    impulse_totals: &mut ImpulseTotals,
) -> Vec<(NeurochemicalId, f32)> {
    let params = NeuroTickParams {
        dt: 1.0,
        metaplasticity_rate: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let mut state = GenesisCoreState::new(1, 1000);
    let mut engine = ActiveInferenceEngine::new();
    let start_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    for t in 0..TICKS {
        // SAFETY: single-threaded test — no concurrent writers.
        unsafe { state.write_begin(0) };
        let pre = effective_all(&state);
        state.neuro_tick_with_params(&params);
        if with_inference {
            let post = effective_all(&state);
            let baselines = state.neurochemicals.baseline_levels();
            let result = engine.cycle(&pre, &post, params.dt, &baselines);
            *policy_counts
                .entry(result.selected_policy.to_string())
                .or_insert(0u32) += 1;
            for &(chem_id, magnitude) in &result.impulses {
                *impulse_totals.entry(chem_id).or_insert(0.0f32) += magnitude;
            }
            apply_inference_feedback(
                &mut state.neurochemicals,
                &result,
                start_ms + t as u64,
                false,
                impulse_offset_to_rate(&params),
            );
        }
        state.write_end();
    }

    // Report EFFECTIVE levels, not raw. `CoreState.chemicals` — what the
    // Python eval harness reads — is unpacked from the vector's
    // `effective_levels` array, and effective is what drives arousal.
    // Comparing raw levels here compares a different quantity from the
    // one the system actually experiences.
    TRACKED
        .iter()
        .map(|id| (*id, state.neurochemicals.effective(*id)))
        .collect()
}

#[test]
fn attribute_resting_offset_to_active_inference() {
    let mut policy_counts: PolicyCounts = std::collections::HashMap::new();
    let mut impulse_totals: ImpulseTotals = std::collections::HashMap::new();
    let chemistry_only = run(false, &mut policy_counts, &mut impulse_totals);
    let with_inference = run(true, &mut policy_counts, &mut impulse_totals);

    // Which policy does the system actually settle on at rest, and what
    // is the resulting impulse budget per chemical?
    let mut counts: Vec<(&String, &u32)> = policy_counts.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1));
    println!("\npolicy selection at rest (of {TICKS} ticks):");
    for (name, n) in &counts {
        println!(
            "  {:<10} {:>6}  ({:>5.1}%)",
            name,
            n,
            100.0 * **n as f32 / TICKS as f32
        );
    }
    println!("\ncumulative impulse per chemical over the run:");
    let mut imps: Vec<(&u8, &f32)> = impulse_totals.iter().collect();
    imps.sort_by(|a, b| a.0.cmp(b.0));
    for (chem_id, total) in imps {
        let id = NeurochemicalId::from_u8(*chem_id);
        println!(
            "  {:<16} {:>+9.2}   (mean {:+.5}/tick)",
            format!("{:?}", id),
            total,
            total / TICKS as f32
        );
    }

    println!(
        "\n{:<20}{:>9}{:>14}{:>16}{:>12}",
        "chemical", "genetic", "chem-only", "+inference", "delta"
    );
    let mut total_delta = 0.0f32;
    for ((id, chem), (_, inf)) in chemistry_only.iter().zip(with_inference.iter()) {
        let g = id.default_baseline();
        let d = inf - chem;
        total_delta += d.abs();
        println!(
            "{:<20}{:>9.3}{:>14.3}{:>16.3}{:>12.3}",
            format!("{:?}", id),
            g,
            chem,
            inf,
            d
        );
    }
    println!(
        "\nsum |delta| from adding the inference layer = {:.3}",
        total_delta
    );

    // The property this test protects.
    //
    // With the inference layer active, the resting chemistry must stay
    // near its genetic baseline. It previously did not: the
    // excitation/inhibition loop had a gain of -0.030 (GLU->GABA +0.10
    // times GABA->GLU -0.30), so inhibition was three times stronger
    // than the excitation recruiting it. That contractionary loop
    // collapsed excitation instead of restoring balance, and Genesis
    // rested with glutamate 39% below baseline. Rebalancing
    // GABA->GLU brought the loop near neutral and the resting state
    // back onto its defaults.
    //
    // The assertion is on the resting level, not on the size of the
    // inference layer's influence: a strong influence that is
    // homeostatically correct is fine, and asserting a minimum
    // magnitude would only re-encode the bug.
    for ((id, chem), (_, inf)) in chemistry_only.iter().zip(with_inference.iter()) {
        assert!(
            chem.is_finite() && inf.is_finite(),
            "{:?} went non-finite (chem={}, inference={})",
            id,
            chem,
            inf
        );
        let g = id.default_baseline();
        assert!(
            (chem - g).abs() < 0.10,
            "{:?} without inference should sit near its genetic default \
             ({:.3}), but was {:.3} — the chemistry alone is not at baseline",
            id,
            g,
            chem
        );
        assert!(
            (inf - g).abs() < 0.12,
            "{:?} rested at {:.3}, {:.3} from its genetic default ({:.3}). \
             The excitation/inhibition loop should settle near balance, \
             not collapse excitation — see the module docs.",
            id,
            inf,
            inf - g,
            g
        );
    }
}
