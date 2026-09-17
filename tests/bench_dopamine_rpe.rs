//! Benchmark 3: Dopamine Prediction Error — Implementation Correctness.
//!
//! Verifies that Genesis's dopamine prediction error computation
//! correctly produces the Schultz (1998, *J Neurophysiol*; 2015,
//! *Physiol Rev*) sign pattern: positive for better-than-predicted,
//! zero for as-predicted, negative for worse-than-predicted.
//!
//! ## What this benchmark tests
//!
//! This is an **implementation correctness test**, not empirical
//! biological validation. It verifies that the PE computation is
//! wired correctly — the sign tracks the deviation, the zero case
//! produces ~zero, and the magnitude is proportional to the impulse.
//!
//! ## Why the correlation is 1.0 by construction
//!
//! The engine is trained on a stable trajectory (no change), so it
//! learns to predict "no change" (identity transition). When an
//! impulse is applied:
//!
//! ```text
//! deviation = post - pre            ≈ impulse
//! PE        = actual - predicted    ≈ actual - "no change" ≈ impulse
//! ```
//!
//! The deviation and PE are measuring the same thing by construction.
//! A Pearson correlation of 1.0 is the expected outcome of a correctly
//! implemented PE computation, not an empirical discovery. This
//! benchmark would produce 1.0 for any system that computes PE as
//! (actual - predicted) when the model predicts "no change."
//!
//! ## Semantic distinction
//!
//! Genesis's `da_prediction_error` is a **state** prediction error
//! (actual dopamine level minus the generative model's predicted
//! dopamine level), not a reward prediction error (RPE) in the
//! TD-learning sense. A true RPE compares outcome value to expected
//! outcome value, which requires a separate reward/value model. The
//! *sign pattern* matches Schultz; the *semantics* differ.
//!
//! ## Methodology
//!
//! 1. Train the engine on a stable trajectory (all chemicals at
//!    baseline, no change) so it learns to predict "no change."
//! 2. Apply positive dopamine impulses of varying magnitudes
//!    (0.05, 0.10, 0.15, 0.20, 0.25, 0.30) and measure the PE.
//! 3. Return to stable (no impulse) and measure PE (should be ~0).
//! 4. Apply negative dopamine impulses of varying magnitudes
//!    (-0.05, -0.10, -0.15, -0.20) and measure PE.
//! 5. Compute:
//!    - Sign accuracy: does the PE sign match the deviation sign?
//!    - Pearson correlation between deviation and PE.
//!    - The "as predicted" PE magnitude (should be near zero).
//!
//! ## Schultz reference data
//!
//! Schultz (1998) showed that dopamine neurons:
//! - Are activated by rewards that are better than predicted (positive RPE)
//! - Are uninfluenced by rewards that are as predicted (zero RPE)
//! - Are depressed by rewards that are worse than predicted (negative RPE)
//! - The response is proportional to the prediction error magnitude
//!
//! This benchmark verifies that the PE computation reproduces the
//! sign pattern. It does not validate the biological claim — the
//! pattern is matched by construction, not discovered.

use std::path::Path;

use genesis::daemon::active_inference::ActiveInferenceEngine;
use genesis::state::neurochemical::{
    NeuroTickParams, NeurochemicalId, NeurochemicalVector,
};

/// Number of training ticks to stabilize the model.
const TRAIN_TICKS: usize = 50;

/// A single trial: impulse magnitude and the resulting PE.
struct Trial {
    impulse_magnitude: f32,
    da_prediction_error: f32,
    da_deviation: f32, // actual DA - predicted DA
}

/// Run a single trial: train on stable, apply impulse, measure PE.
fn run_trial(impulse: f32) -> Trial {
    let mut neuro = NeurochemicalVector::new(0);
    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let baselines = neuro.baseline_levels();

    let mut engine = ActiveInferenceEngine::new();

    // Train on stable trajectory.
    for _ in 0..TRAIN_TICKS {
        let pre = neuro.effective_levels;
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        engine.cycle(&pre, &post, params.dt, &baselines);
    }

    // Record the predicted DA (the model's prediction for this tick).
    let pre = neuro.effective_levels;
    let predicted_da = engine.signals().surprise_ema; // Not directly available;
    let _ = predicted_da;

    // Apply the dopamine impulse.
    if impulse.abs() > 1e-6 {
        neuro.apply_impulse_capped(NeurochemicalId::Dopamine, impulse, 0);
    }
    neuro.recompute_derived();

    // Tick and measure.
    neuro.tick_with_params(&params);
    let post = neuro.effective_levels;

    let result = engine.cycle(&pre, &post, params.dt, &baselines);

    // The DA deviation is the actual DA minus what the model predicted.
    // The model predicts "no change" (identity), so deviation ≈ post - pre.
    let da_idx = NeurochemicalId::Dopamine as usize;
    let da_deviation = post[da_idx] - pre[da_idx];

    Trial {
        impulse_magnitude: impulse,
        da_prediction_error: result.da_prediction_error,
        da_deviation,
    }
}

/// Compute Pearson correlation between two vectors.
fn pearson_correlation(x: &[f32], y: &[f32]) -> f64 {
    let n = x.len() as f64;
    let mean_x: f64 = x.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mean_y: f64 = y.iter().map(|v| *v as f64).sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_x = 0.0f64;
    let mut var_y = 0.0f64;

    for i in 0..x.len() {
        let dx = x[i] as f64 - mean_x;
        let dy = y[i] as f64 - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom > 1e-12 { cov / denom } else { 0.0 }
}

/// Write JSON results.
fn write_json(
    trials: &[Trial],
    correlation: f64,
    sign_accuracy: f64,
    zero_trial_pe: f32,
    path: &Path,
) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"benchmark\": \"dopamine_rpe_pattern\",").unwrap();
    writeln!(
        f,
        "  \"reference\": \"Schultz (1998, J Neurophysiol; 2015, Physiol Rev)\","
    )
    .unwrap();
    writeln!(f, "  \"pattern\": \"positive for better-than-predicted, zero for as-predicted, negative for worse-than-predicted\",").unwrap();
    writeln!(f, "  \"semantic_note\": \"state prediction error, not reward prediction error — pattern matches, semantics differ\",").unwrap();
    writeln!(f, "  \"summary\": {{").unwrap();
    writeln!(f, "    \"pearson_correlation\": {:.6},", correlation).unwrap();
    writeln!(f, "    \"sign_accuracy\": {:.4},", sign_accuracy).unwrap();
    writeln!(f, "    \"zero_trial_pe\": {:.6}", zero_trial_pe).unwrap();
    writeln!(f, "  }},").unwrap();
    writeln!(f, "  \"trials\": [").unwrap();
    for (i, t) in trials.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"impulse\": {:.4},", t.impulse_magnitude).unwrap();
        writeln!(f, "      \"da_deviation\": {:.6},", t.da_deviation).unwrap();
        writeln!(
            f,
            "      \"da_prediction_error\": {:.6}",
            t.da_prediction_error
        )
        .unwrap();
        writeln!(f, "    }}{}", if i + 1 < trials.len() { "," } else { "" }).unwrap();
    }
    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();
}

#[test]
fn bench_dopamine_rpe_pattern() {
    // Test impulses: positive, zero, negative.
    let impulses: Vec<f32> = vec![
        0.30, 0.25, 0.20, 0.15, 0.10, 0.05, // positive (better than predicted)
        0.0,  // as predicted
        -0.05, -0.10, -0.15, -0.20, // negative (worse than predicted)
    ];

    let trials: Vec<Trial> = impulses.iter().map(|&imp| run_trial(imp)).collect();

    // Extract deviations and PEs for correlation.
    let deviations: Vec<f32> = trials.iter().map(|t| t.da_deviation).collect();
    let pes: Vec<f32> = trials.iter().map(|t| t.da_prediction_error).collect();

    // Sign accuracy: does the PE sign match the deviation sign?
    let mut sign_matches = 0;
    for (dev, pe) in deviations.iter().zip(pes.iter()) {
        if dev.abs() < 1e-6 {
            // Zero deviation: PE should be ~0 (counts as match if |PE| < 0.01)
            if pe.abs() < 0.01 {
                sign_matches += 1;
            }
        } else if dev.signum() == pe.signum() {
            sign_matches += 1;
        }
    }
    let sign_accuracy = sign_matches as f64 / trials.len() as f64;

    // Pearson correlation between deviation and PE.
    let correlation = pearson_correlation(&deviations, &pes);

    // The zero-impulse trial's PE (should be near zero).
    let zero_trial = trials
        .iter()
        .find(|t| t.impulse_magnitude.abs() < 1e-6)
        .unwrap();
    let zero_trial_pe = zero_trial.da_prediction_error;

    // Print summary
    let bar = "=".repeat(60);
    let dash = "-".repeat(60);
    println!("\n{bar}");
    println!("Benchmark 3: Dopamine PE Pattern (Schultz 1998)");
    println!("{bar}");
    println!("Reference: Schultz (1998, 2015) — primate dopamine neurons");
    println!("Pattern: +PE for better-than-predicted, 0 for as-predicted, -PE for worse");
    println!("{dash}");
    println!(
        "{:>10} {:>12} {:>12} {:>10}",
        "Impulse", "DA Deviation", "DA PE", "Sign OK?"
    );
    println!("{dash}");
    for t in &trials {
        let sign_ok = if t.impulse_magnitude.abs() < 1e-6 {
            t.da_prediction_error.abs() < 0.01
        } else {
            t.da_deviation.signum() == t.da_prediction_error.signum()
        };
        println!(
            "{:>10.4} {:>12.6} {:>12.6} {:>10}",
            t.impulse_magnitude,
            t.da_deviation,
            t.da_prediction_error,
            if sign_ok { "YES" } else { "NO" }
        );
    }
    println!("{dash}");
    println!("  Pearson correlation (deviation ↔ PE): {:.6}", correlation);
    println!(
        "  Sign accuracy: {:.1}% ({}/{})",
        sign_accuracy * 100.0,
        sign_matches,
        trials.len()
    );
    println!("  Zero-impulse PE: {:.6} (should be ~0)", zero_trial_pe);
    println!();

    // ─── Assertions ──────────────────────────────────────────────

    // Claim 1: Positive impulses produce positive PE (better than predicted).
    let positive_trials: Vec<&Trial> = trials
        .iter()
        .filter(|t| t.impulse_magnitude > 0.01)
        .collect();
    for t in &positive_trials {
        assert!(
            t.da_prediction_error > 0.0,
            "Positive impulse ({:.2}) should produce positive PE, got {:.6}",
            t.impulse_magnitude,
            t.da_prediction_error
        );
    }

    // Claim 2: Zero impulse produces ~zero PE (as predicted).
    assert!(
        zero_trial_pe.abs() < 0.01,
        "Zero impulse should produce ~zero PE, got {:.6}",
        zero_trial_pe
    );

    // Claim 3: Negative impulses produce negative PE (worse than predicted).
    let negative_trials: Vec<&Trial> = trials
        .iter()
        .filter(|t| t.impulse_magnitude < -0.01)
        .collect();
    for t in &negative_trials {
        assert!(
            t.da_prediction_error < 0.0,
            "Negative impulse ({:.2}) should produce negative PE, got {:.6}",
            t.impulse_magnitude,
            t.da_prediction_error
        );
    }

    // Claim 4: Sign accuracy ≥ 90% (the pattern matches Schultz).
    assert!(
        sign_accuracy >= 0.90,
        "Sign accuracy should be ≥ 90%, got {:.1}%",
        sign_accuracy * 100.0
    );

    // Claim 5: Strong positive correlation between deviation and PE.
    assert!(
        correlation > 0.80,
        "Correlation between deviation and PE should be > 0.80, got {:.6}",
        correlation
    );

    // Write JSON output.
    std::fs::create_dir_all("benchmarks/results").ok();
    let out_path = std::path::Path::new("benchmarks/results/bench_dopamine_rpe.json");
    write_json(&trials, correlation, sign_accuracy, zero_trial_pe, out_path);
    println!("Results written to: {}", out_path.display());

    println!("\nAll assertions passed.");
    println!("  - Positive impulses → positive PE: YES");
    println!("  - Zero impulse → ~zero PE: YES ({:.6})", zero_trial_pe);
    println!("  - Negative impulses → negative PE: YES");
    println!(
        "  - Sign accuracy ≥ 90%: YES ({:.1}%)",
        sign_accuracy * 100.0
    );
    println!("  - Correlation > 0.80: YES ({:.6})", correlation);
    println!("  - Pattern matches Schultz (1998): YES");
}
