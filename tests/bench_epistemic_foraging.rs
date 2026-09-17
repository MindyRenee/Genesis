//! Benchmark 4: Regime-Dependent Policy Selection.
//!
//! Tests whether Genesis's active inference engine selects different
//! policies under different regimes (stable, stressed, uncertain),
//! rather than always defaulting to the same action. This is the
//! core property of a regime-dependent action selection system.
//!
//! ## Relationship to the T-maze
//!
//! The T-maze (Friston et al., 2015; Pezzulo et al., 2016) is the
//! canonical active inference benchmark: an agent must seek
//! information (visit a cue) before exploiting (go to the reward).
//! It tests epistemic foraging — information-seeking behavior
//! driven by epistemic value under uncertainty.
//!
//! This benchmark was originally framed as a T-maze structural
//! equivalent, but the results show the system does NOT exhibit
//! epistemic foraging: the uncertain regime selects the same
//! pragmatic corrective policy ("calm") as the stressed regime,
//! not an epistemic policy ("explore"). The epistemic value term
//! in the EFE is not strong enough to override pragmatic value
//! under the tested conditions. This is an honest limitation: the
//! system shows regime-dependent policy selection, but not the
//! epistemic-pragmatic tradeoff that the T-maze tests.
//!
//! ## Methodology
//!
//! Three scenarios, each testing a different regime:
//!
//! 1. **Stable at baseline** (200 ticks, no perturbation):
//!    The system is predictable and at its homeostatic set-points.
//!    Expected: a low-effort policy (rest or noop) — no deviation
//!    to correct, no information to seek.
//!
//! 2. **Stressed** (cortisol impulse, then observe):
//!    The system has deviated from baseline (high cortisol).
//!    Expected: a corrective policy wins (pragmatic action —
//!    reduce the deviation).
//!
//! 3. **Uncertain** (sustained unpredictable input, then observe):
//!    The system has low precision and high belief_var — it can't
//!    predict its own trajectory. An ideal active inference agent
//!    would select an epistemic policy (explore) to reduce
//!    uncertainty. In practice, the system selects "calm" (the
//!    same pragmatic policy as the stressed regime), indicating
//!    the epistemic value term is not strong enough to override
//!    pragmatic value under these conditions.
//!
//! ## What this proves
//!
//! The system selects different policies under different regimes
//! (rest when stable, calm when stressed or uncertain). This
//! confirms regime-dependent policy selection — the engine adapts
//! its action selection to its current state. It does NOT confirm
//! epistemic foraging — the uncertain regime does not select an
//! epistemic policy.

use std::path::Path;

use genesis::daemon::active_inference::ActiveInferenceEngine;
use genesis::state::neurochemical::{NeuroTickParams, NeurochemicalId, NeurochemicalVector};

/// A scenario result: which policy was selected and its EFE.
#[derive(Clone)]
struct ScenarioResult {
    name: String,
    selected_policy: String,
    selected_policy_efe: f32,
    precision: f32,
    surprise: f32,
    free_energy: f32,
    model_maturity: f32,
}

/// Scenario 1: Stable at baseline.
///
/// The system runs with no perturbation. Adenosine naturally
/// accumulates (sleep pressure), so the system drifts from baseline
/// in that dimension. The selected policy reflects what the system
/// needs given its current state — typically a gentle corrective
/// policy like "rest" (to clear sleep pressure) or "noop" (if truly
/// at balance).
fn scenario_stable() -> ScenarioResult {
    let mut neuro = NeurochemicalVector::new(0);
    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let baselines = neuro.baseline_levels();
    let mut engine = ActiveInferenceEngine::new();

    // Run 200 stable ticks.
    for _ in 0..200 {
        let pre = neuro.effective_levels;
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        engine.cycle(&pre, &post, params.dt, &baselines);
    }

    // Record the last cycle's policy selection.
    let pre = neuro.effective_levels;
    neuro.tick_with_params(&params);
    let post = neuro.effective_levels;
    let result = engine.cycle(&pre, &post, params.dt, &baselines);

    ScenarioResult {
        name: "stable_baseline".to_string(),
        selected_policy: result.selected_policy.to_string(),
        selected_policy_efe: result.selected_policy_efe,
        precision: engine.precision(),
        surprise: result.surprise,
        free_energy: result.free_energy,
        model_maturity: engine.signals().model_maturity,
    }
}

/// Scenario 2: Stressed — a corrective policy should win.
fn scenario_stressed() -> ScenarioResult {
    let mut neuro = NeurochemicalVector::new(0);
    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let baselines = neuro.baseline_levels();
    let mut engine = ActiveInferenceEngine::new();

    // Train on stable for 100 ticks.
    for _ in 0..100 {
        let pre = neuro.effective_levels;
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        engine.cycle(&pre, &post, params.dt, &baselines);
    }

    // Apply cortisol stress impulse.
    neuro.apply_impulse_capped(NeurochemicalId::Cortisol, 0.5, 0);
    neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.3, 0);
    neuro.recompute_derived();

    // Run 50 ticks with the stressor active.
    for _ in 0..50 {
        let pre = neuro.effective_levels;
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        engine.cycle(&pre, &post, params.dt, &baselines);
    }

    // Record policy selection under stress.
    let pre = neuro.effective_levels;
    neuro.tick_with_params(&params);
    let post = neuro.effective_levels;
    let result = engine.cycle(&pre, &post, params.dt, &baselines);

    ScenarioResult {
        name: "stressed".to_string(),
        selected_policy: result.selected_policy.to_string(),
        selected_policy_efe: result.selected_policy_efe,
        precision: engine.precision(),
        surprise: result.surprise,
        free_energy: result.free_energy,
        model_maturity: engine.signals().model_maturity,
    }
}

/// Scenario 3: Uncertain — sustained unpredictable input lowers
/// precision and raises belief_var. The epistemic value term should
/// boost information-seeking policies.
///
/// We record the policy selection DURING the uncertain period (not
/// after), while precision is still low and surprise is still high.
fn scenario_uncertain() -> ScenarioResult {
    let mut neuro = NeurochemicalVector::new(0);
    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let baselines = neuro.baseline_levels();
    let mut engine = ActiveInferenceEngine::new();

    // Train on stable for 50 ticks (build initial model).
    for _ in 0..50 {
        let pre = neuro.effective_levels;
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        engine.cycle(&pre, &post, params.dt, &baselines);
    }

    // Feed 200 ticks of unpredictable input: random impulses to
    // different chemicals each tick. This makes the trajectory
    // unpredictable, lowering precision and raising belief_var.
    let chem_ids = [
        NeurochemicalId::Dopamine,
        NeurochemicalId::Serotonin,
        NeurochemicalId::Norepinephrine,
        NeurochemicalId::GABA,
        NeurochemicalId::Glutamate,
        NeurochemicalId::Acetylcholine,
    ];

    let mut last_result = None;
    for i in 0..200 {
        let pre = neuro.effective_levels;

        // Apply a different random-ish impulse each tick.
        let chem = chem_ids[i % chem_ids.len()];
        let mag = if i % 3 == 0 {
            0.15
        } else if i % 3 == 1 {
            -0.10
        } else {
            0.05
        };
        neuro.apply_impulse_capped(chem, mag, 0);
        neuro.recompute_derived();

        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        let result = engine.cycle(&pre, &post, params.dt, &baselines);
        last_result = Some(result);
    }

    let result = last_result.unwrap();

    ScenarioResult {
        name: "uncertain".to_string(),
        selected_policy: result.selected_policy.to_string(),
        selected_policy_efe: result.selected_policy_efe,
        precision: engine.precision(),
        surprise: result.surprise,
        free_energy: result.free_energy,
        model_maturity: engine.signals().model_maturity,
    }
}

/// Write JSON results.
fn write_json(results: &[ScenarioResult], path: &Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"benchmark\": \"regime_dependent_policy_selection\",").unwrap();
    writeln!(
        f,
        "  \"reference\": \"T-maze (Friston et al., 2015; Pezzulo et al., 2016) — structural property NOT demonstrated; system shows regime-dependent selection but not epistemic foraging\","
    )
    .unwrap();
    writeln!(f, "  \"structural_property\": \"regime-dependent policy selection (NOT epistemic foraging)\",").unwrap();
    writeln!(f, "  \"scenarios\": [").unwrap();
    for (i, r) in results.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"name\": \"{}\",", r.name).unwrap();
        writeln!(f, "      \"selected_policy\": \"{}\",", r.selected_policy).unwrap();
        writeln!(
            f,
            "      \"selected_policy_efe\": {:.6},",
            r.selected_policy_efe
        )
        .unwrap();
        writeln!(f, "      \"precision\": {:.6},", r.precision).unwrap();
        writeln!(f, "      \"surprise\": {:.6},", r.surprise).unwrap();
        writeln!(f, "      \"free_energy\": {:.6},", r.free_energy).unwrap();
        writeln!(f, "      \"model_maturity\": {:.6}", r.model_maturity).unwrap();
        writeln!(f, "    }}{}", if i + 1 < results.len() { "," } else { "" }).unwrap();
    }
    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();
}

#[test]
fn bench_epistemic_foraging() {
    let stable = scenario_stable();
    let stressed = scenario_stressed();
    let uncertain = scenario_uncertain();

    let results = vec![stable.clone(), stressed.clone(), uncertain.clone()];

    // Print summary
    let bar = "=".repeat(70);
    let dash = "-".repeat(70);
    println!("\n{bar}");
    println!("Benchmark 4: Regime-Dependent Policy Selection");
    println!("{bar}");
    println!("Reference: T-maze (Friston 2015, Pezzulo 2016) — structural property NOT demonstrated");
    println!("Property: regime-dependent policy selection (NOT epistemic foraging)");
    println!("{dash}");
    println!(
        "{:<20} {:>12} {:>10} {:>10} {:>10}",
        "Scenario", "Policy", "Precision", "Surprise", "EFE"
    );
    println!("{dash}");
    for r in &results {
        println!(
            "{:<20} {:>12} {:>10.4} {:>10.4} {:>10.4}",
            r.name, r.selected_policy, r.precision, r.surprise, r.selected_policy_efe
        );
    }
    println!();

    // ─── Assertions ──────────────────────────────────────────────

    // Claim 1: Policy selection is regime-dependent — at least 2
    // different policies are selected across the 3 regimes. This is
    // the core structural property: the system adapts its action
    // selection to its current state and uncertainty, rather than
    // always selecting the same policy.
    let policies: Vec<&str> = results.iter().map(|r| r.selected_policy.as_str()).collect();
    let unique: std::collections::HashSet<&str> = policies.iter().copied().collect();
    assert!(
        unique.len() >= 2,
        "At least 2 different policies should be selected across regimes, got {} unique: {:?}",
        unique.len(),
        policies
    );

    // Claim 2: In the stressed regime, a corrective policy is
    // selected (NOT noop — the system needs to act to correct the
    // cortisol deviation). This is the "exploitation" phase: the
    // system has a clear deviation and selects a pragmatic action.
    assert_ne!(
        stressed.selected_policy, "noop",
        "Stressed regime should NOT select 'noop' (system needs corrective action), got '{}'",
        stressed.selected_policy
    );

    // Claim 3: The uncertain regime has higher surprise than the
    // stable regime (the system can't predict its trajectory under
    // unpredictable input).
    assert!(
        uncertain.surprise > stable.surprise,
        "Uncertain regime should have higher surprise: stable={:.6}, uncertain={:.6}",
        stable.surprise,
        uncertain.surprise
    );

    // Claim 4: The stressed regime has higher surprise than the
    // stable regime (the cortisol impulse is a prediction error).
    assert!(
        stressed.surprise > stable.surprise || stressed.free_energy > stable.free_energy,
        "Stressed regime should have higher surprise or free energy than stable: \
         stable surprise={:.6} FE={:.6}, stressed surprise={:.6} FE={:.6}",
        stable.surprise,
        stable.free_energy,
        stressed.surprise,
        stressed.free_energy
    );

    // Claim 5: The EFE of the selected policy differs across regimes
    // (policy selection is sensitive to the regime — the system
    // evaluates policies differently depending on its state).
    let efe_values: Vec<f32> = results.iter().map(|r| r.selected_policy_efe).collect();
    let efe_min = efe_values.iter().cloned().fold(f32::INFINITY, f32::min);
    let efe_max = efe_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        efe_max - efe_min > 0.001,
        "EFE should vary across regimes (range={:.6}), indicating regime-sensitive selection",
        efe_max - efe_min
    );

    // Write JSON output.
    std::fs::create_dir_all("benchmarks/results").ok();
    let out_path = std::path::Path::new("benchmarks/results/bench_epistemic_foraging.json");
    write_json(&results, out_path);
    println!("Results written to: {}", out_path.display());

    println!("\nAll assertions passed.");
    println!(
        "  - Regime-dependent policy selection: YES ({:?})",
        policies
    );
    println!(
        "  - Stressed → corrective policy (not noop): YES ({})",
        stressed.selected_policy
    );
    println!(
        "  - Uncertain → higher surprise than stable: YES ({:.6} > {:.6})",
        uncertain.surprise, stable.surprise
    );
    println!("  - Stressed → higher surprise/FE than stable: YES");
    println!(
        "  - EFE varies across regimes: YES (range={:.6})",
        efe_max - efe_min
    );
    println!("  - Epistemic foraging (uncertain → explore): NO (uncertain selected '{}')", uncertain.selected_policy);
}
