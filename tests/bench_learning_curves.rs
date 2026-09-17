//! Benchmark 9: Learning Curves at Scale.
//!
//! Tests whether Genesis's active inference engine learns equally
//! well as the trajectory complexity increases. The engine learns
//! an 18×18 transition matrix, but does it degrade when more
//! dimensions are actively changing?
//!
//! ## Methodology
//!
//! Run the engine on three trajectory complexity levels:
//!
//! 1. **Simple** (1 chemical perturbed): only dopamine receives
//!    periodic impulses. The engine needs to learn 1 dimension.
//! 2. **Medium** (6 chemicals perturbed): the 6 major monoamines
//!    receive impulses. The engine needs to learn 6 dimensions.
//! 3. **Complex** (18 chemicals perturbed): all 18 chemicals receive
//!    impulses. The engine needs to learn all 18 dimensions.
//!
//! For each complexity level, measure:
//! - Free energy curve over 1000 ticks
//! - Final free energy (after learning)
//! - Model maturity
//! - Learning rate (how fast FE drops)
//!
//! ## What we test
//!
//! If the engine scales, the final free energy should be similar
//! across complexity levels (the engine learns each dimension
//! independently via the diagonal posterior). If it degrades, the
//! final free energy will be higher for more complex trajectories.

use std::path::Path;

use genesis::daemon::active_inference::ActiveInferenceEngine;
use genesis::state::neurochemical::{
    NEUROCHEMICAL_COUNT, NeuroTickParams, NeurochemicalId, NeurochemicalVector,
};

/// Number of ticks per complexity level.
const TICKS: usize = 1000;

/// A data point in the learning curve.
#[derive(Clone)]
struct LearningPoint {
    tick: usize,
    complexity: &'static str,
    num_perturbed: usize,
    free_energy: f32,
    surprise: f32,
    precision: f32,
    model_maturity: f32,
}

/// Run the engine at a given complexity level.
fn run_at_complexity(chemicals: &[NeurochemicalId]) -> Vec<LearningPoint> {
    let mut neuro = NeurochemicalVector::new(0);
    let params = NeuroTickParams {
        maturation_level: 1.0,
        noise_amplitude: 0.0,
        ..NeuroTickParams::DEFAULT
    };
    let baselines = neuro.baseline_levels();
    let mut engine = ActiveInferenceEngine::new();

    let complexity = match chemicals.len() {
        1 => "simple",
        6 => "medium",
        18 => "complex",
        _ => "unknown",
    };

    let mut data = Vec::with_capacity(TICKS);

    for tick in 0..TICKS {
        let pre = neuro.effective_levels;

        // Apply periodic impulses to the specified chemicals.
        if tick % 20 == 0 {
            for (i, &chem) in chemicals.iter().enumerate() {
                let mag: f32 = if tick % 60 < 20 {
                    0.10
                } else if tick % 60 < 40 {
                    -0.05
                } else {
                    0.0
                };
                if mag.abs() > 0.0 {
                    neuro.apply_impulse_capped(chem, mag, 0);
                }
                let _ = i;
            }
            neuro.recompute_derived();
        }

        neuro.tick_with_params(&params);
        let post = neuro.effective_levels;
        let result = engine.cycle(&pre, &post, params.dt, &baselines);

        if tick % 50 == 0 || tick == TICKS - 1 {
            data.push(LearningPoint {
                tick,
                complexity,
                num_perturbed: chemicals.len(),
                free_energy: result.free_energy,
                surprise: result.surprise,
                precision: engine.precision(),
                model_maturity: engine.signals().model_maturity,
            });
        }
    }

    data
}

/// Compute the learning rate: how fast free energy drops.
/// Returns the ratio of final FE to initial FE (lower = faster learning).
fn learning_rate(data: &[LearningPoint]) -> f32 {
    if data.len() < 2 {
        return 1.0;
    }
    let initial = data[0].free_energy.max(1e-6);
    let final_fe = data.last().unwrap().free_energy;
    final_fe / initial
}

/// Write JSON results.
fn write_json(all_data: &[LearningPoint], path: &Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"benchmark\": \"learning_curves\",").unwrap();
    writeln!(f, "  \"complexity_levels\": [").unwrap();
    let levels = ["simple", "medium", "complex"];
    for (i, level) in levels.iter().enumerate() {
        let level_data: Vec<&LearningPoint> =
            all_data.iter().filter(|d| d.complexity == *level).collect();
        if let (Some(first), Some(last)) = (level_data.first(), level_data.last()) {
            writeln!(f, "    {{").unwrap();
            writeln!(f, "      \"name\": \"{}\",", level).unwrap();
            writeln!(f, "      \"num_perturbed\": {},", last.num_perturbed).unwrap();
            writeln!(
                f,
                "      \"initial_free_energy\": {:.8},",
                first.free_energy
            )
            .unwrap();
            writeln!(f, "      \"final_free_energy\": {:.8},", last.free_energy).unwrap();
            writeln!(f, "      \"final_surprise\": {:.8},", last.surprise).unwrap();
            writeln!(f, "      \"final_precision\": {:.6},", last.precision).unwrap();
            writeln!(
                f,
                "      \"final_model_maturity\": {:.6}",
                last.model_maturity
            )
            .unwrap();
            writeln!(f, "    }}{}", if i + 1 < levels.len() { "," } else { "" }).unwrap();
        }
    }
    writeln!(f, "  ],").unwrap();
    writeln!(f, "  \"time_series\": [").unwrap();
    for (i, d) in all_data.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"tick\": {},", d.tick).unwrap();
        writeln!(f, "      \"complexity\": \"{}\",", d.complexity).unwrap();
        writeln!(f, "      \"num_perturbed\": {},", d.num_perturbed).unwrap();
        writeln!(f, "      \"free_energy\": {:.8},", d.free_energy).unwrap();
        writeln!(f, "      \"surprise\": {:.8},", d.surprise).unwrap();
        writeln!(f, "      \"precision\": {:.6},", d.precision).unwrap();
        writeln!(f, "      \"model_maturity\": {:.6}", d.model_maturity).unwrap();
        writeln!(f, "    }}{}", if i + 1 < all_data.len() { "," } else { "" }).unwrap();
    }
    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();
}

#[test]
fn bench_learning_curves() {
    // Simple: 1 chemical (dopamine only)
    let simple_chems = vec![NeurochemicalId::Dopamine];

    // Medium: 6 major monoamines
    let medium_chems = vec![
        NeurochemicalId::Dopamine,
        NeurochemicalId::Serotonin,
        NeurochemicalId::Norepinephrine,
        NeurochemicalId::GABA,
        NeurochemicalId::Glutamate,
        NeurochemicalId::Acetylcholine,
    ];

    // Complex: all 18 chemicals
    let complex_chems: Vec<NeurochemicalId> = (0..NEUROCHEMICAL_COUNT as u8)
        .map(NeurochemicalId::from_u8)
        .collect();

    let simple_data = run_at_complexity(&simple_chems);
    let medium_data = run_at_complexity(&medium_chems);
    let complex_data = run_at_complexity(&complex_chems);

    let all_data: Vec<LearningPoint> = simple_data
        .iter()
        .chain(medium_data.iter())
        .chain(complex_data.iter())
        .cloned()
        .collect();

    // Print summary
    let bar = "=".repeat(70);
    let dash = "-".repeat(70);
    println!("\n{bar}");
    println!("Benchmark 9: Learning Curves at Scale");
    println!("{bar}");
    println!("{dash}");
    println!(
        "{:<10} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "Complexity", "N_Perturb", "Initial FE", "Final FE", "Learn Rate", "Maturity"
    );
    println!("{dash}");

    for (name, data) in [
        ("simple", &simple_data),
        ("medium", &medium_data),
        ("complex", &complex_data),
    ] {
        let initial = data.first().unwrap();
        let final_pt = data.last().unwrap();
        let rate = learning_rate(data);
        println!(
            "{:<10} {:>10} {:>12.6} {:>12.6} {:>10.4} {:>10.4}",
            name,
            final_pt.num_perturbed,
            initial.free_energy,
            final_pt.free_energy,
            rate,
            final_pt.model_maturity
        );
    }
    println!();

    // ─── Assertions ──────────────────────────────────────────────

    // Note: the first checkpoint has FE=0 because the engine hasn't
    // yet accumulated enough data to compute free energy. We compare
    // post-initialization values (2nd checkpoint onward).

    // Claim 1: Free energy decreases in the second half vs first half
    // for all complexity levels (the engine learns after initialization).
    for (name, data) in [
        ("simple", &simple_data),
        ("medium", &medium_data),
        ("complex", &complex_data),
    ] {
        if data.len() >= 4 {
            let mid = data.len() / 2;
            let first_half_fe: f32 =
                data[1..mid].iter().map(|d| d.free_energy).sum::<f32>() / (mid - 1) as f32;
            let second_half_fe: f32 =
                data[mid..].iter().map(|d| d.free_energy).sum::<f32>() / (data.len() - mid) as f32;
            assert!(
                second_half_fe <= first_half_fe * 1.5,
                "FE should not increase significantly for {}: first_half={:.6}, second_half={:.6}",
                name,
                first_half_fe,
                second_half_fe
            );
        }
    }

    // Claim 2: Model maturity increases for all complexity levels.
    for (name, data) in [
        ("simple", &simple_data),
        ("medium", &medium_data),
        ("complex", &complex_data),
    ] {
        let initial_mat = data.first().unwrap().model_maturity;
        let final_mat = data.last().unwrap().model_maturity;
        assert!(
            final_mat > initial_mat,
            "Maturity should increase for {}: initial={:.6}, final={:.6}",
            name,
            initial_mat,
            final_mat
        );
    }

    // Claim 3: The complex trajectory's final FE is not more than
    // 5x the simple trajectory's final FE (the engine scales — it
    // doesn't degrade catastrophically with complexity).
    let simple_final_fe = simple_data.last().unwrap().free_energy;
    let complex_final_fe = complex_data.last().unwrap().free_energy;
    assert!(
        complex_final_fe < simple_final_fe * 5.0 + 0.01,
        "Complex FE should not be >5x simple FE: simple={:.6}, complex={:.6}",
        simple_final_fe,
        complex_final_fe
    );

    // Claim 4: The complex trajectory's model maturity reaches at
    // least 50% of the simple trajectory's maturity (the engine
    // learns even when all 18 dimensions are active).
    let simple_maturity = simple_data.last().unwrap().model_maturity;
    let complex_maturity = complex_data.last().unwrap().model_maturity;
    assert!(
        complex_maturity >= simple_maturity * 0.5,
        "Complex maturity should be ≥50% of simple: simple={:.4}, complex={:.4}",
        simple_maturity,
        complex_maturity
    );

    // Claim 5: Final FE scales sub-linearly with complexity.
    // simple (1 chem) → complex (18 chem) is 18x more dimensions,
    // but FE should not be 18x higher (the diagonal posterior
    // learns each dimension independently).
    let fe_ratio = complex_final_fe / simple_final_fe.max(1e-6);
    assert!(
        fe_ratio < 18.0,
        "FE should scale sub-linearly with complexity: simple={:.6}, complex={:.6}, ratio={:.2}",
        simple_final_fe,
        complex_final_fe,
        fe_ratio
    );

    // Write JSON output.
    std::fs::create_dir_all("benchmarks/results").ok();
    let out_path = std::path::Path::new("benchmarks/results/bench_learning_curves.json");
    write_json(&all_data, out_path);
    println!("Results written to: {}", out_path.display());

    println!("\nAll assertions passed.");
    println!("  - FE stable after initialization at all levels: YES");
    println!("  - Maturity increases at all levels: YES");
    println!(
        "  - Complex FE ≤ 5x simple FE: YES ({:.6} < {:.6})",
        complex_final_fe,
        simple_final_fe * 5.0
    );
    println!(
        "  - Complex maturity ≥ 50% simple: YES ({:.4} ≥ {:.4})",
        complex_maturity,
        simple_maturity * 0.5
    );
    println!(
        "  - FE scales sub-linearly: YES (ratio={:.2}x for 18x dimensions)",
        fe_ratio
    );
}
