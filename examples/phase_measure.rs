//! Measures the learned phase model against the neurochemical
//! classifier over a no-input simulation.
//!
//! Runs the coupled neurochemistry at its 100 ms integration step with
//! no external input, samples the eighteen effective levels every 10
//! simulated seconds, fits a regime model to the samples, and reports
//! how the regimes line up with the classifier's `emergent_phase`.
//!
//! Usage: `cargo run --release --example phase_measure -- [hours]`
//! (default 48 simulated hours).

use genesis::state::MentalPhase;
use genesis::state::neurochemical::NeurochemicalVector;
use genesis::state::phase_partition::{self, MAX_PHASES, Point};

const TICKS_PER_SECOND: u64 = 10;
const SAMPLE_EVERY_SECONDS: u64 = 10;

fn main() {
    let hours: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(48.0);
    let ticks = (hours * 3600.0 * TICKS_PER_SECOND as f64) as u64;
    let stride = SAMPLE_EVERY_SECONDS * TICKS_PER_SECOND;

    let mut v = NeurochemicalVector::new(0);
    let mut occupancy = [0u64; MAX_PHASES];
    let mut points: Vec<Point> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();
    for t in 0..ticks {
        v.tick_default();
        occupancy[usize::from(v.emergent_phase)] += 1;
        if t % stride == 0 {
            points.push(std::array::from_fn(|i| v.effective_levels[i]));
            labels.push(v.emergent_phase);
        }
    }

    println!("simulated {hours} h, {} samples", points.len());
    println!("classifier occupancy:");
    for (p, n) in occupancy.iter().enumerate().filter(|(_, n)| **n > 0) {
        let name = MentalPhase::from_u8(p as u8).label();
        println!("  {name:<12} {:>6.2}%", 100.0 * *n as f64 / ticks as f64);
    }

    let Some(mut model) = phase_partition::fit(&points) else {
        println!("not enough samples to fit a regime model");
        return;
    };
    println!("learned model: {} regimes (BIC-selected)", model.k());

    let mut table = vec![vec![0u64; model.k()]; MAX_PHASES];
    for (p, &label) in points.iter().zip(&labels) {
        let regime = model.observe(p);
        table[usize::from(label)][regime] += 1;
    }
    println!("classifier label × learned regime (sample counts):");
    print!("  {:<12}", "");
    for r in 0..model.k() {
        print!(" {:>7}", format!("R{r}"));
    }
    println!();
    for (label, row) in table.iter().enumerate().filter(|(_, row)| row.iter().any(|n| *n > 0)) {
        print!("  {:<12}", MentalPhase::from_u8(label as u8).label());
        for n in row {
            print!(" {n:>7}");
        }
        println!();
    }
    for r in 0..model.k() {
        let m = model.mean(r).expect("regime exists");
        let profile: Vec<String> = m.iter().map(|x| format!("{x:.2}")).collect();
        println!("R{r} mean: [{}]", profile.join(" "));
    }
}
