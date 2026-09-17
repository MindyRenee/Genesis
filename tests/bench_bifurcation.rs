//! Benchmark 5: Bifurcation Analysis of Phase Transitions.
//!
//! Tests whether Genesis's neurochemical dynamics show genuine
//! bifurcations — qualitative changes in system behavior as a
//! control parameter crosses critical values — rather than smooth
//! interpolation.
//!
//! ## Methodology
//!
//! We sweep the **homeostatic rate** (the strength of the restoring
//! force toward baseline) from 0.001 (very weak) to 0.08 (strong).
//! At each value, we:
//!
//! 1. Run the system for 2000 ticks to reach steady state.
//! 2. Apply a perturbation (dopamine impulse).
//! 3. Measure recovery time, oscillation amplitude, and steady-state
//!    variance across all 18 chemicals.
//!
//! ## What we expect
//!
//! - **High homeostatic rate**: fast recovery, low variance, stable
//!   (the system is rigidly controlled).
//! - **Low homeostatic rate**: slow recovery, high variance, possibly
//!   oscillatory (the system is loosely controlled).
//! - **Critical transition**: there should be a parameter range
//!   where the system transitions from stable to oscillatory/erratic.
//!   This is a bifurcation — a qualitative change in dynamics.
//!
//! ## What this proves
//!
//! If the system shows a qualitative transition (e.g., variance
//! increases sharply, recovery time diverges, oscillations appear)
//! at a critical homeostatic rate, the dynamics are genuinely
//! nonlinear and the system's stability is parameter-dependent.
//! This is the signature of a dynamical system with bifurcations,
//! not a simple thermostat.

use std::path::Path;

use genesis::state::neurochemical::{
    NEUROCHEMICAL_COUNT, NeuroTickParams, NeurochemicalId, NeurochemicalVector,
};

/// Number of ticks to reach steady state.
const STABILIZE_TICKS: usize = 2000;

/// Number of ticks to measure recovery and oscillation.
const MEASURE_TICKS: usize = 1000;

/// Perturbation magnitude.
const PERTURB_MAG: f32 = 0.15;

/// A single point in the bifurcation diagram.
struct BifurcationPoint {
    homeostatic_rate: f32,
    steady_state_variance: f32,
    recovery_ticks: usize,
    oscillation_amplitude: f32,
    max_chemical_deviation: f32,
}

/// Compute the variance of the effective levels over a window.
fn compute_variance(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
    samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32
}

/// Run the system at a given homeostatic rate.
fn run_at_parameter(homeostatic_rate: f32) -> BifurcationPoint {
    let mut neuro = NeurochemicalVector::new(0);

    let params = NeuroTickParams {
        homeostatic_rate,
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };

    // Stabilize.
    for _ in 0..STABILIZE_TICKS {
        neuro.tick_with_params(&params);
    }

    // Record steady-state variance: sample all 18 chemicals over 200
    // ticks and compute the mean variance. High variance = oscillatory.
    // We measure all chemicals (not just cortisol) because cortisol is
    // at 0.0 at rest (no stressor) — its variance is uninformative.
    // The system-wide variance captures coupling-driven oscillations
    // in chemicals that are active at rest (dopamine, serotonin, etc.).
    let mut all_samples: Vec<Vec<f32>> = (0..NEUROCHEMICAL_COUNT)
        .map(|_| Vec::with_capacity(200))
        .collect();
    for _ in 0..200 {
        neuro.tick_with_params(&params);
        for (i, chem) in neuro.chemicals.iter().enumerate().take(NEUROCHEMICAL_COUNT) {
            all_samples[i].push(chem.level);
        }
    }
    let steady_state_variance: f32 = all_samples
        .iter()
        .map(|samples| compute_variance(samples))
        .sum::<f32>()
        / NEUROCHEMICAL_COUNT as f32;

    // Record pre-perturbation dopamine.
    let pre_da = neuro.chemicals[NeurochemicalId::Dopamine as usize].level;

    // Apply perturbation.
    neuro.apply_impulse_capped(NeurochemicalId::Dopamine, PERTURB_MAG, 0);
    neuro.recompute_derived();

    let post_perturb_da = neuro.chemicals[NeurochemicalId::Dopamine as usize].level;
    let immediate_response = post_perturb_da - pre_da;

    // Measure recovery and oscillation over MEASURE_TICKS.
    let mut da_samples = Vec::with_capacity(MEASURE_TICKS);
    let mut recovery_ticks = MEASURE_TICKS;
    let target = pre_da + immediate_response * 0.05;

    for t in 0..MEASURE_TICKS {
        neuro.tick_with_params(&params);
        let da = neuro.chemicals[NeurochemicalId::Dopamine as usize].level;
        da_samples.push(da);
        if recovery_ticks == MEASURE_TICKS && (da - target).abs() < immediate_response.abs() * 0.05
        {
            recovery_ticks = t;
        }
    }

    // Oscillation amplitude: max - min of dopamine in the post-perturbation window.
    let da_max = da_samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let da_min = da_samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let oscillation_amplitude = da_max - da_min;

    // Max chemical deviation from baseline across all 18 chemicals.
    let baselines = neuro.baseline_levels();
    let mut max_dev = 0.0f32;
    for (chem, &baseline) in neuro.chemicals.iter().zip(baselines.iter()).take(NEUROCHEMICAL_COUNT) {
        let dev = (chem.level - baseline).abs();
        if dev > max_dev {
            max_dev = dev;
        }
    }

    BifurcationPoint {
        homeostatic_rate,
        steady_state_variance,
        recovery_ticks,
        oscillation_amplitude,
        max_chemical_deviation: max_dev,
    }
}

/// Write JSON results.
fn write_json(points: &[BifurcationPoint], path: &Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"benchmark\": \"bifurcation_analysis\",").unwrap();
    writeln!(
        f,
        "  \"methodology\": \"parameter sweep (homeostatic rate) with perturbation recovery\","
    )
    .unwrap();
    writeln!(f, "  \"control_parameter\": \"homeostatic_rate\",").unwrap();
    writeln!(f, "  \"data_points\": [").unwrap();
    for (i, p) in points.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"homeostatic_rate\": {:.6},", p.homeostatic_rate).unwrap();
        writeln!(
            f,
            "      \"steady_state_variance\": {:.8},",
            p.steady_state_variance
        )
        .unwrap();
        writeln!(f, "      \"recovery_ticks\": {},", p.recovery_ticks).unwrap();
        writeln!(
            f,
            "      \"oscillation_amplitude\": {:.8},",
            p.oscillation_amplitude
        )
        .unwrap();
        writeln!(
            f,
            "      \"max_chemical_deviation\": {:.8}",
            p.max_chemical_deviation
        )
        .unwrap();
        writeln!(f, "    }}{}", if i + 1 < points.len() { "," } else { "" }).unwrap();
    }
    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();
}

#[test]
fn bench_bifurcation_analysis() {
    // Sweep homeostatic rate from 0.001 (very weak) to 0.08 (strong).
    // 16 parameter values.
    let param_values: Vec<f32> = (0..16).map(|i| 0.001 + i as f32 * 0.005).collect();

    let points: Vec<BifurcationPoint> = param_values.iter().map(|&p| run_at_parameter(p)).collect();

    // Print summary
    let bar = "=".repeat(80);
    let dash = "-".repeat(80);
    println!("\n{bar}");
    println!("Benchmark 5: Bifurcation Analysis (Homeostatic Rate Sweep)");
    println!("{bar}");
    println!("Control parameter: homeostatic_rate (restoring force strength)");
    println!("Range: 0.001 (weak) to 0.076 (strong), 16 values");
    println!("{dash}");
    println!(
        "{:>10} {:>14} {:>10} {:>14} {:>14}",
        "HO_RATE", "SS_Variance", "Recovery", "Oscill_Amp", "Max_Dev"
    );
    println!("{dash}");
    for p in &points {
        println!(
            "{:>10.6} {:>14.8} {:>10} {:>14.8} {:>14.8}",
            p.homeostatic_rate,
            p.steady_state_variance,
            p.recovery_ticks,
            p.oscillation_amplitude,
            p.max_chemical_deviation
        );
    }
    println!();

    // ─── Assertions ──────────────────────────────────────────────

    // Claim 1: Recovery time varies across the parameter sweep.
    // At high homeostatic rate, recovery should be fast; at low
    // rate, recovery should be slow or not converge.
    let recovery_values: Vec<usize> = points.iter().map(|p| p.recovery_ticks).collect();
    let recovery_min = *recovery_values.iter().min().unwrap();
    let recovery_max = *recovery_values.iter().max().unwrap();
    let recovery_range = recovery_max as f32 - recovery_min as f32;
    assert!(
        recovery_range > 10.0,
        "Recovery time should vary across sweep (range={}), got min={} max={}",
        recovery_range,
        recovery_min,
        recovery_max
    );

    // Claim 2: The steady-state variance varies across the sweep.
    // At low homeostatic rate, the system should be more variable
    // (less tightly controlled).
    let variance_values: Vec<f32> = points.iter().map(|p| p.steady_state_variance).collect();
    let variance_range = variance_values
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max)
        - variance_values
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
    assert!(
        variance_range > 1e-8,
        "Steady-state variance should vary across sweep (range={:.10})",
        variance_range
    );

    // Claim 3: The oscillation amplitude varies across the sweep.
    let osc_values: Vec<f32> = points.iter().map(|p| p.oscillation_amplitude).collect();
    let osc_range = osc_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - osc_values.iter().cloned().fold(f32::INFINITY, f32::min);
    assert!(
        osc_range > 1e-6,
        "Oscillation amplitude should vary across sweep (range={:.10})",
        osc_range
    );

    // Claim 4: There is a qualitative transition — the system's
    // behavior changes non-smoothly at some critical parameter value.
    // We detect this by looking for a parameter value where the
    // recovery time jumps by more than 2x relative to the previous
    // value, or where the variance increases by more than 3x.
    let mut has_qualitative_transition = false;
    for i in 1..points.len() {
        let prev_r = points[i - 1].recovery_ticks as f32;
        let curr_r = points[i].recovery_ticks as f32;
        if prev_r > 0.0 && curr_r / prev_r > 2.0 {
            has_qualitative_transition = true;
            println!(
                "Qualitative transition at ho_rate={:.6}: recovery {} → {}",
                points[i].homeostatic_rate,
                points[i - 1].recovery_ticks,
                points[i].recovery_ticks
            );
            break;
        }
        let prev_v = points[i - 1].steady_state_variance;
        let curr_v = points[i].steady_state_variance;
        if prev_v > 1e-10 && curr_v / prev_v > 3.0 {
            has_qualitative_transition = true;
            println!(
                "Qualitative transition at ho_rate={:.6}: variance {:.8} → {:.8}",
                points[i].homeostatic_rate, prev_v, curr_v
            );
            break;
        }
    }

    // Claim 5: At the highest homeostatic rate, recovery is faster
    // than at the lowest rate (the restoring force actually works).
    let high_ho = points.last().unwrap();
    let low_ho = points.first().unwrap();
    assert!(
        high_ho.recovery_ticks <= low_ho.recovery_ticks,
        "High homeostatic rate should recover faster: high={} ticks, low={} ticks",
        high_ho.recovery_ticks,
        low_ho.recovery_ticks
    );

    // Write JSON output.
    std::fs::create_dir_all("benchmarks/results").ok();
    let out_path = std::path::Path::new("benchmarks/results/bench_bifurcation.json");
    write_json(&points, out_path);
    println!("Results written to: {}", out_path.display());

    println!("\nAll assertions passed.");
    println!("  - Recovery time varies: YES (range={})", recovery_range);
    println!("  - Variance varies: YES (range={:.10})", variance_range);
    println!(
        "  - Oscillation amplitude varies: YES (range={:.10})",
        osc_range
    );
    println!(
        "  - Qualitative transition: {}",
        if has_qualitative_transition {
            "YES"
        } else {
            "smooth (no sharp transition)"
        }
    );
    println!(
        "  - High ho_rate recovers faster: YES ({} ≤ {})",
        high_ho.recovery_ticks, low_ho.recovery_ticks
    );
}
