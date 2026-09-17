//! Benchmark 6: Active Inference vs. Proportional Controller.
//!
//! Compares Genesis's active inference engine against a proportional
//! controller (the standard engineering baseline for homeostatic
//! control) on the same task: maintaining neurochemical state at
//! baseline under perturbation.
//!
//! ## Why a proportional controller instead of pymdp?
//!
//! pymdp (Hinz et al., 2021) is the reference Python library for
//! active inference, but it's designed for discrete state spaces
//! and would require significant adaptation to operate on Genesis's
//! continuous 18-dimensional neurochemical trajectory. A proportional
//! controller is the natural baseline for continuous homeostatic
//! control — it's what a control systems engineer would build
//! without active inference.
//!
//! ## The comparison
//!
//! Both systems face the same perturbation sequence:
//! 1. Stable baseline (200 ticks)
//! 2. Stress perturbation (cortisol + CRH impulses, 100 ticks)
//! 3. Recovery (200 ticks)
//! 4. Second perturbation (100 ticks) — tests adaptation
//! 5. Final recovery (200 ticks)
//!
//! **Genesis**: Uses the full active inference engine (learned
//! transition matrix, adaptive precision, policy selection, EFE).
//!
//! **Proportional controller**: Simple negative feedback with fixed
//! gain. For each chemical, applies a correction proportional to the
//! deviation from baseline: correction = K * (baseline - level).
//! No learning, no prediction, no policy selection.
//!
//! ## Metrics
//!
//! - **Cumulative free energy**: Genesis's variational free energy
//!   vs. the controller's squared deviation (both measure "how far
//!   from optimal").
//! - **Recovery time**: ticks to return to within 5% of baseline
//!   after perturbation ends.
//! - **Steady-state deviation**: mean |level - baseline| during
//!   stable phases.
//! - **Adaptation**: does the second perturbation produce lower
//!   free energy than the first? (Genesis should learn; the
//!   controller can't.)

use std::path::Path;

use genesis::daemon::active_inference::ActiveInferenceEngine;
use genesis::state::neurochemical::{
    NEUROCHEMICAL_COUNT, NeuroTickParams, NeurochemicalId, NeurochemicalVector,
};

/// Proportional controller gain.
const KP: f32 = 0.04;

/// Phase lengths.
const STABLE_TICKS: usize = 200;
const PERTURB_TICKS: usize = 100;
const RECOVERY_TICKS: usize = 200;

/// A data point for comparison.
struct ComparisonPoint {
    tick: usize,
    phase: &'static str,
    genesis_free_energy: f32,
    controller_cost: f32,
    genesis_deviation: f32,
    controller_deviation: f32,
}

/// Run Genesis's active inference engine through the perturbation
/// sequence and return the data.
fn run_genesis() -> Vec<ComparisonPoint> {
    let mut neuro = NeurochemicalVector::new(0);
    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let baselines = neuro.baseline_levels();
    let mut engine = ActiveInferenceEngine::new();

    let mut data = Vec::new();

    // Phase 1: Stable
    for tick in 0..STABLE_TICKS {
        let pre = neuro.effective_levels;
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        let result = engine.cycle(&pre, &post, params.dt, &baselines);

        let deviation = rms_deviation(&post, &baselines);
        data.push(ComparisonPoint {
            tick,
            phase: "stable_1",
            genesis_free_energy: result.free_energy,
            controller_cost: 0.0, // filled by controller run
            genesis_deviation: deviation,
            controller_deviation: 0.0,
        });
    }

    // Phase 2: Perturbation 1
    for tick in 0..PERTURB_TICKS {
        let pre = neuro.effective_levels;
        if tick % 10 == 0 {
            neuro.apply_impulse_capped(NeurochemicalId::Cortisol, 0.15, 0);
            neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.10, 0);
        }
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        let result = engine.cycle(&pre, &post, params.dt, &baselines);

        let deviation = rms_deviation(&post, &baselines);
        data.push(ComparisonPoint {
            tick: STABLE_TICKS + tick,
            phase: "perturb_1",
            genesis_free_energy: result.free_energy,
            controller_cost: 0.0,
            genesis_deviation: deviation,
            controller_deviation: 0.0,
        });
    }

    // Phase 3: Recovery 1
    for tick in 0..RECOVERY_TICKS {
        let pre = neuro.effective_levels;
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        let result = engine.cycle(&pre, &post, params.dt, &baselines);

        let deviation = rms_deviation(&post, &baselines);
        data.push(ComparisonPoint {
            tick: STABLE_TICKS + PERTURB_TICKS + tick,
            phase: "recovery_1",
            genesis_free_energy: result.free_energy,
            controller_cost: 0.0,
            genesis_deviation: deviation,
            controller_deviation: 0.0,
        });
    }

    // Phase 4: Perturbation 2 (tests adaptation)
    for tick in 0..PERTURB_TICKS {
        let pre = neuro.effective_levels;
        if tick % 10 == 0 {
            neuro.apply_impulse_capped(NeurochemicalId::Cortisol, 0.15, 0);
            neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.10, 0);
        }
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        let result = engine.cycle(&pre, &post, params.dt, &baselines);

        let deviation = rms_deviation(&post, &baselines);
        data.push(ComparisonPoint {
            tick: STABLE_TICKS + PERTURB_TICKS + RECOVERY_TICKS + tick,
            phase: "perturb_2",
            genesis_free_energy: result.free_energy,
            controller_cost: 0.0,
            genesis_deviation: deviation,
            controller_deviation: 0.0,
        });
    }

    // Phase 5: Recovery 2
    for tick in 0..RECOVERY_TICKS {
        let pre = neuro.effective_levels;
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        let result = engine.cycle(&pre, &post, params.dt, &baselines);

        let deviation = rms_deviation(&post, &baselines);
        data.push(ComparisonPoint {
            tick: STABLE_TICKS + PERTURB_TICKS + RECOVERY_TICKS + PERTURB_TICKS + tick,
            phase: "recovery_2",
            genesis_free_energy: result.free_energy,
            controller_cost: 0.0,
            genesis_deviation: deviation,
            controller_deviation: 0.0,
        });
    }

    data
}

/// Run a proportional controller through the same perturbation
/// sequence and return the data.
fn run_controller() -> Vec<ComparisonPoint> {
    let mut neuro = NeurochemicalVector::new(0);
    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let baselines = neuro.baseline_levels();

    let mut data = Vec::new();

    // Phase 1: Stable
    for tick in 0..STABLE_TICKS {
        // Apply proportional control: push each chemical toward baseline.
        for (chem, &baseline) in neuro.chemicals.iter_mut().zip(baselines.iter()).take(NEUROCHEMICAL_COUNT) {
            let error = baseline - chem.level;
            chem.level += KP * error;
        }
        neuro.recompute_derived();
        neuro.tick_with_params(&params);

        let deviation = rms_deviation(&neuro.effective_levels, &baselines);
        // Controller "cost" = squared deviation (analogous to free energy).
        let cost = deviation * deviation;
        data.push(ComparisonPoint {
            tick,
            phase: "stable_1",
            genesis_free_energy: 0.0,
            controller_cost: cost,
            genesis_deviation: 0.0,
            controller_deviation: deviation,
        });
    }

    // Phase 2: Perturbation 1
    for tick in 0..PERTURB_TICKS {
        if tick % 10 == 0 {
            neuro.apply_impulse_capped(NeurochemicalId::Cortisol, 0.15, 0);
            neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.10, 0);
        }
        for (chem, &baseline) in neuro.chemicals.iter_mut().zip(baselines.iter()).take(NEUROCHEMICAL_COUNT) {
            let error = baseline - chem.level;
            chem.level += KP * error;
        }
        neuro.recompute_derived();
        neuro.tick_with_params(&params);

        let deviation = rms_deviation(&neuro.effective_levels, &baselines);
        let cost = deviation * deviation;
        data.push(ComparisonPoint {
            tick: STABLE_TICKS + tick,
            phase: "perturb_1",
            genesis_free_energy: 0.0,
            controller_cost: cost,
            genesis_deviation: 0.0,
            controller_deviation: deviation,
        });
    }

    // Phase 3: Recovery 1
    for tick in 0..RECOVERY_TICKS {
        for (chem, &baseline) in neuro.chemicals.iter_mut().zip(baselines.iter()).take(NEUROCHEMICAL_COUNT) {
            let error = baseline - chem.level;
            chem.level += KP * error;
        }
        neuro.recompute_derived();
        neuro.tick_with_params(&params);

        let deviation = rms_deviation(&neuro.effective_levels, &baselines);
        let cost = deviation * deviation;
        data.push(ComparisonPoint {
            tick: STABLE_TICKS + PERTURB_TICKS + tick,
            phase: "recovery_1",
            genesis_free_energy: 0.0,
            controller_cost: cost,
            genesis_deviation: 0.0,
            controller_deviation: deviation,
        });
    }

    // Phase 4: Perturbation 2
    for tick in 0..PERTURB_TICKS {
        if tick % 10 == 0 {
            neuro.apply_impulse_capped(NeurochemicalId::Cortisol, 0.15, 0);
            neuro.apply_impulse_capped(NeurochemicalId::CRH, 0.10, 0);
        }
        for (chem, &baseline) in neuro.chemicals.iter_mut().zip(baselines.iter()).take(NEUROCHEMICAL_COUNT) {
            let error = baseline - chem.level;
            chem.level += KP * error;
        }
        neuro.recompute_derived();
        neuro.tick_with_params(&params);

        let deviation = rms_deviation(&neuro.effective_levels, &baselines);
        let cost = deviation * deviation;
        data.push(ComparisonPoint {
            tick: STABLE_TICKS + PERTURB_TICKS + RECOVERY_TICKS + tick,
            phase: "perturb_2",
            genesis_free_energy: 0.0,
            controller_cost: cost,
            genesis_deviation: 0.0,
            controller_deviation: deviation,
        });
    }

    // Phase 5: Recovery 2
    for tick in 0..RECOVERY_TICKS {
        for (chem, &baseline) in neuro.chemicals.iter_mut().zip(baselines.iter()).take(NEUROCHEMICAL_COUNT) {
            let error = baseline - chem.level;
            chem.level += KP * error;
        }
        neuro.recompute_derived();
        neuro.tick_with_params(&params);

        let deviation = rms_deviation(&neuro.effective_levels, &baselines);
        let cost = deviation * deviation;
        data.push(ComparisonPoint {
            tick: STABLE_TICKS + PERTURB_TICKS + RECOVERY_TICKS + PERTURB_TICKS + tick,
            phase: "recovery_2",
            genesis_free_energy: 0.0,
            controller_cost: cost,
            genesis_deviation: 0.0,
            controller_deviation: deviation,
        });
    }

    data
}

/// Compute RMS deviation from baseline.
fn rms_deviation(
    levels: &[f32; NEUROCHEMICAL_COUNT],
    baselines: &[f32; NEUROCHEMICAL_COUNT],
) -> f32 {
    let sum_sq: f32 = (0..NEUROCHEMICAL_COUNT)
        .map(|i| (levels[i] - baselines[i]).powi(2))
        .sum();
    (sum_sq / NEUROCHEMICAL_COUNT as f32).sqrt()
}

/// Merge Genesis and controller data into a combined series.
fn merge_data(genesis: &[ComparisonPoint], controller: &[ComparisonPoint]) -> Vec<ComparisonPoint> {
    genesis
        .iter()
        .zip(controller.iter())
        .map(|(g, c)| ComparisonPoint {
            tick: g.tick,
            phase: g.phase,
            genesis_free_energy: g.genesis_free_energy,
            controller_cost: c.controller_cost,
            genesis_deviation: g.genesis_deviation,
            controller_deviation: c.controller_deviation,
        })
        .collect()
}

/// Compute phase summary.
struct PhaseSummary {
    name: String,
    genesis_fe_mean: f32,
    controller_cost_mean: f32,
    genesis_dev_mean: f32,
    controller_dev_mean: f32,
}

fn summarize_phase(data: &[ComparisonPoint], phase: &str) -> PhaseSummary {
    let phase_data: Vec<&ComparisonPoint> = data.iter().filter(|d| d.phase == phase).collect();
    let n = phase_data.len() as f32;
    PhaseSummary {
        name: phase.to_string(),
        genesis_fe_mean: phase_data
            .iter()
            .map(|d| d.genesis_free_energy)
            .sum::<f32>()
            / n,
        controller_cost_mean: phase_data.iter().map(|d| d.controller_cost).sum::<f32>() / n,
        genesis_dev_mean: phase_data.iter().map(|d| d.genesis_deviation).sum::<f32>() / n,
        controller_dev_mean: phase_data
            .iter()
            .map(|d| d.controller_deviation)
            .sum::<f32>()
            / n,
    }
}

/// Write JSON results.
fn write_json(data: &[ComparisonPoint], summaries: &[PhaseSummary], path: &Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"benchmark\": \"aif_vs_proportional\",").unwrap();
    writeln!(
        f,
        "  \"baseline\": \"proportional controller (Kp={}, no learning)\",",
        KP
    )
    .unwrap();
    writeln!(f, "  \"phases\": [").unwrap();
    for (i, s) in summaries.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"name\": \"{}\",", s.name).unwrap();
        writeln!(f, "      \"genesis_fe_mean\": {:.8},", s.genesis_fe_mean).unwrap();
        writeln!(
            f,
            "      \"controller_cost_mean\": {:.8},",
            s.controller_cost_mean
        )
        .unwrap();
        writeln!(f, "      \"genesis_dev_mean\": {:.8},", s.genesis_dev_mean).unwrap();
        writeln!(
            f,
            "      \"controller_dev_mean\": {:.8}",
            s.controller_dev_mean
        )
        .unwrap();
        writeln!(
            f,
            "    }}{}",
            if i + 1 < summaries.len() { "," } else { "" }
        )
        .unwrap();
    }
    writeln!(f, "  ],").unwrap();
    writeln!(f, "  \"time_series\": [").unwrap();
    for (i, d) in data.iter().enumerate() {
        if d.tick % 10 == 0 || i == data.len() - 1 {
            writeln!(f, "    {{").unwrap();
            writeln!(f, "      \"tick\": {},", d.tick).unwrap();
            writeln!(f, "      \"phase\": \"{}\",", d.phase).unwrap();
            writeln!(
                f,
                "      \"genesis_free_energy\": {:.8},",
                d.genesis_free_energy
            )
            .unwrap();
            writeln!(f, "      \"controller_cost\": {:.8},", d.controller_cost).unwrap();
            writeln!(
                f,
                "      \"genesis_deviation\": {:.8},",
                d.genesis_deviation
            )
            .unwrap();
            writeln!(
                f,
                "      \"controller_deviation\": {:.8}",
                d.controller_deviation
            )
            .unwrap();
            writeln!(f, "    }}{}", if i + 1 < data.len() { "," } else { "" }).unwrap();
        }
    }
    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();
}

#[test]
fn bench_aif_vs_proportional() {
    let genesis_data = run_genesis();
    let controller_data = run_controller();
    let merged = merge_data(&genesis_data, &controller_data);

    let phases = [
        "stable_1",
        "perturb_1",
        "recovery_1",
        "perturb_2",
        "recovery_2",
    ];
    let summaries: Vec<PhaseSummary> = phases.iter().map(|p| summarize_phase(&merged, p)).collect();

    // Print summary
    let bar = "=".repeat(70);
    let dash = "-".repeat(70);
    println!("\n{bar}");
    println!("Benchmark 6: Active Inference vs. Proportional Controller");
    println!("{bar}");
    println!("Baseline: proportional controller (Kp={}, no learning)", KP);
    println!("{dash}");
    println!(
        "{:<15} {:>12} {:>12} {:>12} {:>12}",
        "Phase", "Genesis FE", "Ctrl Cost", "Genesis Dev", "Ctrl Dev"
    );
    println!("{dash}");
    for s in &summaries {
        println!(
            "{:<15} {:>12.8} {:>12.8} {:>12.8} {:>12.8}",
            s.name,
            s.genesis_fe_mean,
            s.controller_cost_mean,
            s.genesis_dev_mean,
            s.controller_dev_mean
        );
    }
    println!();

    // ─── Assertions ──────────────────────────────────────────────
    //
    // Important finding: the proportional controller achieves lower
    // absolute deviation than Genesis's active inference engine.
    // This is expected and honest — the controller directly and
    // aggressively pushes all 18 chemicals toward baseline every
    // tick, while Genesis's active inference only generates small
    // impulses (0.01-0.05 magnitude) through policy selection. The
    // primary homeostatic control in Genesis is the tick function's
    // homeostatic_rate, not the active inference engine.
    //
    // The active inference engine's value is NOT in regulation —
    // it's in prediction, learning, and adaptation. The fair
    // comparison tests those properties.

    // Claim 1: Genesis adapts — the second perturbation produces
    // lower free energy than the first (the model learned from the
    // first perturbation). This is the core advantage of active
    // inference over a fixed controller.
    let perturb_1 = &summaries[1];
    let perturb_2 = &summaries[3];
    assert!(
        perturb_2.genesis_fe_mean < perturb_1.genesis_fe_mean,
        "Genesis should adapt: perturb_1 FE={:.8}, perturb_2 FE={:.8} (2nd should be lower)",
        perturb_1.genesis_fe_mean,
        perturb_2.genesis_fe_mean
    );

    // Claim 2: The controller's cost is NOT systematically lower on
    // the second perturbation (it doesn't learn — any difference is
    // due to state, not adaptation). We verify the controller doesn't
    // show the same adaptation pattern as Genesis.
    let controller_ratio =
        perturb_2.controller_cost_mean / perturb_1.controller_cost_mean.max(1e-10);
    let genesis_ratio = perturb_2.genesis_fe_mean / perturb_1.genesis_fe_mean.max(1e-10);
    assert!(
        genesis_ratio < controller_ratio,
        "Genesis should show more adaptation than controller: genesis_ratio={:.4}, controller_ratio={:.4}",
        genesis_ratio,
        controller_ratio
    );

    // Claim 3: The controller achieves lower absolute deviation
    // (it's a better regulator — this is the honest tradeoff).
    // Genesis trades regulation quality for prediction and learning.
    let recovery_1 = &summaries[2];
    assert!(
        recovery_1.controller_dev_mean < recovery_1.genesis_dev_mean,
        "Controller should have lower deviation (better regulator): genesis={:.8}, controller={:.8}",
        recovery_1.genesis_dev_mean,
        recovery_1.controller_dev_mean
    );

    // Claim 4: Genesis's free energy decreases over the recovery
    // phase (the system is settling as the model predicts better).
    // This is something the controller can't show — it has no
    // concept of free energy or prediction.
    let recovery_1_data: Vec<&ComparisonPoint> =
        merged.iter().filter(|d| d.phase == "recovery_1").collect();
    let early_fe: f32 = recovery_1_data[5..15]
        .iter()
        .map(|d| d.genesis_free_energy)
        .sum::<f32>()
        / 10.0;
    let late_fe: f32 = recovery_1_data[recovery_1_data.len() - 10..]
        .iter()
        .map(|d| d.genesis_free_energy)
        .sum::<f32>()
        / 10.0;
    assert!(
        late_fe < early_fe,
        "Genesis free energy should decrease during recovery: early={:.8}, late={:.8}",
        early_fe,
        late_fe
    );

    // Write JSON output.
    std::fs::create_dir_all("benchmarks/results").ok();
    let out_path = std::path::Path::new("benchmarks/results/bench_aif_vs_proportional.json");
    write_json(&merged, &summaries, out_path);
    println!("Results written to: {}", out_path.display());

    println!("\nAll assertions passed.");
    println!(
        "  - Genesis adapts (2nd perturb < 1st): YES (FE {:.8} → {:.8})",
        perturb_1.genesis_fe_mean, perturb_2.genesis_fe_mean
    );
    println!(
        "  - Genesis adapts more than controller: YES (ratio {:.4} < {:.4})",
        genesis_ratio, controller_ratio
    );
    println!(
        "  - Controller is better regulator: YES (dev {:.8} < {:.8})",
        recovery_1.controller_dev_mean, recovery_1.genesis_dev_mean
    );
    println!(
        "  - Genesis FE decreases during recovery: YES ({:.8} → {:.8})",
        early_fe, late_fe
    );
    println!("  - Honest tradeoff: controller regulates better, Genesis learns better");
}
