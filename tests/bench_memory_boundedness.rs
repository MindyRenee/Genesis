//! Benchmark 8: Memory Boundedness Over Time.
//!
//! Tests whether Genesis's long-term memory (LTM) store grows
//! predictably and bounded as episodes accumulate. The key claims:
//!
//! 1. **Linear growth**: file size grows linearly with episode count
//!    (no superlinear blowup).
//! 2. **Bounded per-episode size**: each episode adds a bounded
//!    amount of storage (no episodes that grow without limit).
//! 3. **Stable retrieval latency**: retrieval time doesn't degrade
//!    catastrophically as the store grows.
//! 4. **Count accuracy**: the store's count matches the number of
//!    stored episodes.
//!
//! ## Methodology
//!
//! Store 1000 episodes with varying text lengths, measuring file
//! sizes and retrieval latency at intervals (10, 100, 500, 1000).
//! Fit a linear model to the file size vs. episode count and
//! verify the R² is high (growth is linear).

use std::path::{Path, PathBuf};
use std::time::Instant;

use genesis::store::LtmStore;

fn temp_base(test_name: &str) -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("genesis_bench_{test_name}_{pid}_{nanos}"))
}

fn cleanup(base: &Path) {
    let _ = std::fs::remove_file(base.with_extension("bundles"));
    let _ = std::fs::remove_file(base.with_extension("meta"));
    let _ = std::fs::remove_file(base.with_extension("dat"));
}

fn make_full_tag() -> [f32; 12] {
    [
        0.50, 0.55, 0.40, 0.45, 0.50, 0.60, 0.20, 0.35, 0.30, 0.45, 0.10, 0.40,
    ]
}

fn make_compact_tag() -> [f32; 4] {
    [0.40, 0.35, 0.20, 0.50]
}

struct Measurement {
    episode_count: u32,
    dat_size: u64,
    meta_size: u64,
    bundles_size: u64,
    total_size: u64,
    retrieve_latency_us: u64,
}

fn measure_store(store: &mut LtmStore, base: &Path) -> Measurement {
    let count = store.count();

    let dat_size = std::fs::metadata(base.with_extension("dat"))
        .map(|m| m.len())
        .unwrap_or(0);
    let meta_size = std::fs::metadata(base.with_extension("meta"))
        .map(|m| m.len())
        .unwrap_or(0);
    let bundles_size = std::fs::metadata(base.with_extension("bundles"))
        .map(|m| m.len())
        .unwrap_or(0);
    let total_size = dat_size + meta_size + bundles_size;

    // Measure retrieval latency: retrieve the first episode.
    let retrieve_latency_us = if count > 0 {
        let start = Instant::now();
        let _ = store.retrieve(1);
        let elapsed = start.elapsed();
        elapsed.as_micros() as u64
    } else {
        0
    };

    Measurement {
        episode_count: count,
        dat_size,
        meta_size,
        bundles_size,
        total_size,
        retrieve_latency_us,
    }
}

/// Fit a line y = a + b*x to the data and return (slope, r_squared).
fn linear_fit(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean_x: f64 = xs.iter().sum::<f64>() / n;
    let mean_y: f64 = ys.iter().sum::<f64>() / n;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;

    for i in 0..xs.len() {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }

    if ss_xx.abs() < 1e-12 {
        return (0.0, 0.0);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    let ss_res: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let predicted = intercept + slope * x;
            (y - predicted).powi(2)
        })
        .sum();

    let r_squared = if ss_yy.abs() > 1e-12 {
        1.0 - ss_res / ss_yy
    } else {
        0.0
    };

    (slope, r_squared)
}

fn write_json(measurements: &[Measurement], slope: f64, r_squared: f64, path: &Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"benchmark\": \"memory_boundedness\",").unwrap();
    writeln!(f, "  \"growth_model\": {{").unwrap();
    writeln!(f, "    \"type\": \"linear\",").unwrap();
    writeln!(f, "    \"slope_bytes_per_episode\": {:.2},", slope).unwrap();
    writeln!(f, "    \"r_squared\": {:.6}", r_squared).unwrap();
    writeln!(f, "  }},").unwrap();
    writeln!(f, "  \"measurements\": [").unwrap();
    for (i, m) in measurements.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"episode_count\": {},", m.episode_count).unwrap();
        writeln!(f, "      \"dat_size\": {},", m.dat_size).unwrap();
        writeln!(f, "      \"meta_size\": {},", m.meta_size).unwrap();
        writeln!(f, "      \"bundles_size\": {},", m.bundles_size).unwrap();
        writeln!(f, "      \"total_size\": {},", m.total_size).unwrap();
        writeln!(
            f,
            "      \"retrieve_latency_us\": {}",
            m.retrieve_latency_us
        )
        .unwrap();
        writeln!(
            f,
            "    }}{}",
            if i + 1 < measurements.len() { "," } else { "" }
        )
        .unwrap();
    }
    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();
}

#[test]
fn bench_memory_boundedness() {
    let base = temp_base("memory_bounded");
    let mut store = LtmStore::create(&base, 64).expect("create");

    let total_episodes = 1000;
    let measure_points = [10, 100, 500, 1000];

    let mut measurements = Vec::new();

    // Varying text lengths to test boundedness across payload sizes.
    let texts: Vec<String> = (0..total_episodes)
        .map(|i| {
            let len = 20 + (i % 200); // 20-219 bytes
            format!("Episode {}: {}", i, "x".repeat(len))
        })
        .collect();

    for (i, text) in texts.iter().enumerate() {
        let salience = 0.5 + (i as f32 % 100.0) / 200.0;
        store
            .store(
                i as u64 * 100,
                salience,
                make_full_tag(),
                make_compact_tag(),
                0,
                4,
                text,
            )
            .expect("store");

        if measure_points.contains(&(i + 1)) {
            measurements.push(measure_store(&mut store, &base));
        }
    }

    // Print summary
    let bar = "=".repeat(70);
    let dash = "-".repeat(70);
    println!("\n{bar}");
    println!(
        "Benchmark 8: Memory Boundedness ({} episodes)",
        total_episodes
    );
    println!("{bar}");
    println!("{dash}");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Count", "DAT (B)", "Meta (B)", "Bundles (B)", "Total (B)", "Retrieve (us)"
    );
    println!("{dash}");
    for m in &measurements {
        println!(
            "{:>8} {:>12} {:>12} {:>12} {:>12} {:>12}",
            m.episode_count,
            m.dat_size,
            m.meta_size,
            m.bundles_size,
            m.total_size,
            m.retrieve_latency_us
        );
    }
    println!();

    // Fit linear model to total size vs. episode count.
    let xs: Vec<f64> = measurements
        .iter()
        .map(|m| m.episode_count as f64)
        .collect();
    let ys: Vec<f64> = measurements.iter().map(|m| m.total_size as f64).collect();
    let (slope, r_squared) = linear_fit(&xs, &ys);

    println!(
        "Linear fit: {:.2} bytes/episode, R² = {:.6}",
        slope, r_squared
    );
    println!();

    // ─── Assertions ──────────────────────────────────────────────

    // Claim 1: File size grows linearly with episode count (R² > 0.99).
    assert!(
        r_squared > 0.99,
        "Growth should be linear (R² > 0.99), got R²={:.6}",
        r_squared
    );

    // Claim 2: The per-episode storage is bounded (slope < 1KB/episode).
    assert!(
        slope < 1024.0,
        "Per-episode storage should be bounded (< 1KB), got {:.2} bytes",
        slope
    );

    // Claim 3: The store's count matches the number of stored episodes.
    assert_eq!(
        store.count(),
        total_episodes as u32,
        "Store count should match episodes stored"
    );

    // Claim 4: Retrieval latency stays bounded (< 10ms even at 1000 episodes).
    let max_latency = measurements
        .iter()
        .map(|m| m.retrieve_latency_us)
        .max()
        .unwrap();
    assert!(
        max_latency < 10_000,
        "Retrieval latency should stay bounded (< 10ms), got {}us",
        max_latency
    );

    // Claim 5: No file grows superlinearly. Check the ratio of size
    // increase between 100→500 vs 10→100 episodes. If growth is
    // linear, the ratio should be ~4x (400/100 = 4).
    if measurements.len() >= 3 {
        let size_10 = measurements[0].total_size as f64;
        let size_100 = measurements[1].total_size as f64;
        let size_500 = measurements[2].total_size as f64;
        let early_growth = size_100 - size_10;
        let late_growth = size_500 - size_100;
        if early_growth > 0.0 {
            let ratio = late_growth / early_growth;
            let expected_ratio = 400.0 / 90.0; // 4.44x
            assert!(
                ratio < expected_ratio * 1.5,
                "Growth should not be superlinear: early={:.0}, late={:.0}, ratio={:.2} (expected ~{:.2})",
                early_growth,
                late_growth,
                ratio,
                expected_ratio
            );
        }
    }

    // Write JSON output.
    std::fs::create_dir_all("benchmarks/results").ok();
    let out_path = std::path::Path::new("benchmarks/results/bench_memory_boundedness.json");
    write_json(&measurements, slope, r_squared, out_path);
    println!("Results written to: {}", out_path.display());

    // Cleanup.
    drop(store);
    cleanup(&base);

    println!("\nAll assertions passed.");
    println!("  - Linear growth (R² > 0.99): YES (R²={:.6})", r_squared);
    println!("  - Bounded per-episode size: YES ({:.2} bytes)", slope);
    println!("  - Count accuracy: YES ({})", total_episodes);
    println!("  - Retrieval latency bounded: YES (max={}us)", max_latency);
    println!("  - No superlinear growth: YES");
}
