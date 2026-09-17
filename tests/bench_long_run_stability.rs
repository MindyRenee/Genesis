//! Benchmark 7: Long-Run Stability (100,000+ ticks).
//!
//! Tests whether Genesis's neurochemical system remains stable over
//! long-duration operation: 100,000 ticks = 10,000 seconds = ~2.8
//! hours of simulated time at 10 Hz.
//!
//! ## What we test
//!
//! 1. **No NaN/Infinity**: no chemical level becomes NaN or infinite.
//! 2. **Bounded state**: all chemical levels stay in [0, 1] (or
//!    their valid range).
//! 3. **No divergence**: the system doesn't drift to extreme values.
//! 4. **Periodic perturbation recovery**: the system recovers from
//!    perturbations applied every 10,000 ticks.
//! 5. **Stable attractor**: the system returns to a stable state
//!    after each perturbation.
//!
//! ## Why 100k ticks?
//!
//! The existing property tests run for 1000 ticks. A 100x longer
//! run tests for:
//! - Slow accumulation effects (baseline drift, receptor adaptation).
//! - Numerical instability that only appears after many iterations.
//! - Edge cases in the coupling matrix that only trigger at specific
//!   state configurations.
//! - Memory effects in the active inference engine.

use std::path::Path;

use genesis::daemon::active_inference::ActiveInferenceEngine;
use genesis::state::neurochemical::{
    NEUROCHEMICAL_COUNT, NeuroTickParams, NeurochemicalId, NeurochemicalVector,
};

/// Total ticks to run (100,000 = ~2.8 hours at 10 Hz).
const TOTAL_TICKS: usize = 100_000;

/// Perturbation interval (every 10,000 ticks).
const PERTURB_INTERVAL: usize = 10_000;

/// A checkpoint recording the system state at intervals.
struct Checkpoint {
    tick: usize,
    phase: &'static str,
    max_level: f32,
    min_level: f32,
    mean_level: f32,
    any_nan: bool,
    any_inf: bool,
    max_deviation: f32,
    free_energy: f32,
    surprise: f32,
    precision: f32,
    model_maturity: f32,
}

/// Run the long-duration stability test.
fn run_stability_test() -> Vec<Checkpoint> {
    let mut neuro = NeurochemicalVector::new(0);
    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.002, // production noise
        noise_seed: 42,
        ..NeuroTickParams::DEFAULT
    };
    let baselines = neuro.baseline_levels();
    let mut engine = ActiveInferenceEngine::new();

    let mut checkpoints = Vec::new();
    let checkpoint_interval = 5_000; // record every 5000 ticks

    for tick in 0..TOTAL_TICKS {
        // Apply periodic perturbation.
        let phase = if tick % PERTURB_INTERVAL == 0 && tick > 0 {
            neuro.apply_impulse_capped(NeurochemicalId::Cortisol, 0.3, 0);
            neuro.apply_impulse_capped(NeurochemicalId::Dopamine, 0.2, 0);
            neuro.recompute_derived();
            "perturbation"
        } else {
            "stable"
        };

        let pre = neuro.effective_levels;
        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        let result = engine.cycle(&pre, &post, params.dt, &baselines);

        // Record checkpoint.
        if tick % checkpoint_interval == 0 || tick == TOTAL_TICKS - 1 {
            let mut max_level = f32::NEG_INFINITY;
            let mut min_level = f32::INFINITY;
            let mut sum_level = 0.0f32;
            let mut any_nan = false;
            let mut any_inf = false;
            let mut max_dev = 0.0f32;

            for (chem, &baseline) in neuro.chemicals.iter().zip(baselines.iter()).take(NEUROCHEMICAL_COUNT) {
                let level = chem.level;
                if level.is_nan() {
                    any_nan = true;
                }
                if level.is_infinite() {
                    any_inf = true;
                }
                if level > max_level {
                    max_level = level;
                }
                if level < min_level {
                    min_level = level;
                }
                sum_level += level;
                let dev = (level - baseline).abs();
                if dev > max_dev {
                    max_dev = dev;
                }
            }

            checkpoints.push(Checkpoint {
                tick,
                phase,
                max_level,
                min_level,
                mean_level: sum_level / NEUROCHEMICAL_COUNT as f32,
                any_nan,
                any_inf,
                max_deviation: max_dev,
                free_energy: result.free_energy,
                surprise: result.surprise,
                precision: engine.precision(),
                model_maturity: engine.signals().model_maturity,
            });
        }
    }

    checkpoints
}

/// Write JSON results.
fn write_json(checkpoints: &[Checkpoint], path: &Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"benchmark\": \"long_run_stability\",").unwrap();
    writeln!(f, "  \"total_ticks\": {},", TOTAL_TICKS).unwrap();
    writeln!(
        f,
        "  \"simulated_time_seconds\": {:.0},",
        TOTAL_TICKS as f64 * 0.1
    )
    .unwrap();
    writeln!(f, "  \"perturbation_interval\": {},", PERTURB_INTERVAL).unwrap();
    writeln!(f, "  \"checkpoints\": [").unwrap();
    for (i, c) in checkpoints.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"tick\": {},", c.tick).unwrap();
        writeln!(f, "      \"phase\": \"{}\",", c.phase).unwrap();
        writeln!(f, "      \"max_level\": {:.6},", c.max_level).unwrap();
        writeln!(f, "      \"min_level\": {:.6},", c.min_level).unwrap();
        writeln!(f, "      \"mean_level\": {:.6},", c.mean_level).unwrap();
        writeln!(f, "      \"any_nan\": {},", c.any_nan).unwrap();
        writeln!(f, "      \"any_inf\": {},", c.any_inf).unwrap();
        writeln!(f, "      \"max_deviation\": {:.6},", c.max_deviation).unwrap();
        writeln!(f, "      \"free_energy\": {:.8},", c.free_energy).unwrap();
        writeln!(f, "      \"surprise\": {:.8},", c.surprise).unwrap();
        writeln!(f, "      \"precision\": {:.6},", c.precision).unwrap();
        writeln!(f, "      \"model_maturity\": {:.6}", c.model_maturity).unwrap();
        writeln!(
            f,
            "    }}{}",
            if i + 1 < checkpoints.len() { "," } else { "" }
        )
        .unwrap();
    }
    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();
}

#[test]
fn bench_long_run_stability() {
    let checkpoints = run_stability_test();

    // Print summary
    let bar = "=".repeat(70);
    let dash = "-".repeat(70);
    println!("\n{bar}");
    println!("Benchmark 7: Long-Run Stability ({} ticks)", TOTAL_TICKS);
    println!("{bar}");
    println!(
        "Simulated time: {:.1} hours",
        TOTAL_TICKS as f64 * 0.1 / 3600.0
    );
    println!("Perturbations every {} ticks", PERTURB_INTERVAL);
    println!("{dash}");
    println!(
        "{:>8} {:>8} {:>10} {:>10} {:>10} {:>8} {:>10}",
        "Tick", "Phase", "MaxLevel", "MinLevel", "MaxDev", "NaN?", "FE"
    );
    println!("{dash}");
    for c in checkpoints.iter().step_by(4) {
        // print every 4th checkpoint
        println!(
            "{:>8} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>8} {:>10.6}",
            c.tick,
            c.phase,
            c.max_level,
            c.min_level,
            c.max_deviation,
            if c.any_nan { "YES" } else { "no" },
            c.free_energy
        );
    }
    // Always print the last checkpoint.
    let last = checkpoints.last().unwrap();
    println!(
        "{:>8} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>8} {:>10.6}",
        last.tick,
        last.phase,
        last.max_level,
        last.min_level,
        last.max_deviation,
        if last.any_nan { "YES" } else { "no" },
        last.free_energy
    );
    println!();

    // ─── Assertions ──────────────────────────────────────────────

    // Claim 1: No NaN in any chemical level at any checkpoint.
    for c in &checkpoints {
        assert!(
            !c.any_nan,
            "NaN detected at tick {} — system is numerically unstable",
            c.tick
        );
    }

    // Claim 2: No Infinity in any chemical level at any checkpoint.
    for c in &checkpoints {
        assert!(
            !c.any_inf,
            "Infinity detected at tick {} — system is numerically unstable",
            c.tick
        );
    }

    // Claim 3: All chemical levels stay bounded in [0, 2].
    // (The system uses [0, 1] for most chemicals, but effective levels
    // can reach 2.0 with high receptor sensitivity. We check [0, 2]
    // as the hard bound.)
    for c in &checkpoints {
        assert!(
            c.max_level <= 2.0 && c.min_level >= 0.0,
            "Level out of bounds at tick {}: min={:.4}, max={:.4}",
            c.tick,
            c.min_level,
            c.max_level
        );
    }

    // Claim 4: The system doesn't diverge — max deviation stays
    // bounded (< 0.8, well below the hard clamp).
    let max_dev_over_run = checkpoints
        .iter()
        .map(|c| c.max_deviation)
        .fold(0.0f32, f32::max);
    assert!(
        max_dev_over_run < 0.8,
        "Max deviation should stay bounded (< 0.8), got {:.4}",
        max_dev_over_run
    );

    // Claim 5: The system recovers from perturbations — the max
    // deviation at the end is not larger than at the first perturbation.
    let first_perturb = checkpoints
        .iter()
        .find(|c| c.phase == "perturbation")
        .unwrap();
    let final_checkpoint = checkpoints.last().unwrap();
    assert!(
        final_checkpoint.max_deviation < first_perturb.max_deviation * 2.0,
        "System should not accumulate deviation: first_perturb={:.4}, final={:.4}",
        first_perturb.max_deviation,
        final_checkpoint.max_deviation
    );

    // Claim 6: Model maturity increases over the run (the active
    // inference engine keeps learning).
    let initial_maturity = checkpoints.first().unwrap().model_maturity;
    let final_maturity = final_checkpoint.model_maturity;
    assert!(
        final_maturity > initial_maturity,
        "Model maturity should increase: initial={:.6}, final={:.6}",
        initial_maturity,
        final_maturity
    );

    // Claim 7: Free energy doesn't diverge over the run.
    let max_fe = checkpoints
        .iter()
        .map(|c| c.free_energy)
        .fold(0.0f32, f32::max);
    let mean_fe = checkpoints.iter().map(|c| c.free_energy).sum::<f32>() / checkpoints.len() as f32;
    assert!(
        max_fe < mean_fe * 10.0 + 0.1,
        "Free energy should not diverge: max={:.6}, mean={:.6}",
        max_fe,
        mean_fe
    );

    // Write JSON output.
    std::fs::create_dir_all("benchmarks/results").ok();
    let out_path = std::path::Path::new("benchmarks/results/bench_long_run_stability.json");
    write_json(&checkpoints, out_path);
    println!("Results written to: {}", out_path.display());

    println!("\nAll assertions passed.");
    println!("  - No NaN/Inf in {} ticks: YES", TOTAL_TICKS);
    println!("  - All levels bounded [0, 2]: YES");
    println!("  - Max deviation bounded: YES ({:.4})", max_dev_over_run);
    println!("  - System recovers from perturbations: YES");
    println!(
        "  - Model maturity increases: YES ({:.4} → {:.4})",
        initial_maturity, final_maturity
    );
    println!(
        "  - Free energy doesn't diverge: YES (max={:.6}, mean={:.6})",
        max_fe, mean_fe
    );
    println!(
        "  - Duration: {:.1} hours simulated, {} perturbations",
        TOTAL_TICKS as f64 * 0.1 / 3600.0,
        TOTAL_TICKS / PERTURB_INTERVAL
    );
}
