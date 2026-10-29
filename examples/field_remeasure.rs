//! Remeasure the field's standing disagreement under the
//! stereographic-sphere geometry, to check the precision-attenuation
//! calibration (`DISAGREEMENT_PRECISION_SCALE` in src/daemon/tick.rs).
//!
//! Drives `FieldRuntime::observe` — the real predict → correct →
//! integrate path the tick loop uses — at the mind's ~1 Hz cadence
//! with a quiet-host operating point: mid-range effective levels,
//! modest inference signals, and small stationary jitter (the host
//! breathes; it is never frozen). After a settle burn-in it reports
//! the disagreement distribution, then drives a genuine incoherence
//! excursion (sustained surprise) so both ends of the attenuation
//! curve can be checked against the calibrated scale.

use genesis::field::FieldRuntime;
use genesis::state::{GenesisCoreState, NeurochemicalId};

struct SplitMix64(u64);

impl SplitMix64 {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Quiet-host operating point: representative mid-range levels —
/// nothing stressed, nothing idle-frozen.
fn quiet_state(rng: &mut SplitMix64) -> GenesisCoreState {
    let mut state = GenesisCoreState::new(1, 1_000);
    let jit = |center: f32, amp: f32, rng: &mut SplitMix64| -> f32 {
        (center + (rng.next_f64() as f32 - 0.5) * 2.0 * amp).clamp(0.0, 2.0)
    };
    let c = &mut state.neurochemicals.chemicals;
    c[NeurochemicalId::Norepinephrine as usize].level = jit(0.45, 0.03, rng);
    c[NeurochemicalId::Dopamine as usize].level = jit(0.40, 0.03, rng);
    c[NeurochemicalId::Serotonin as usize].level = jit(0.60, 0.03, rng);
    c[NeurochemicalId::BDNF as usize].level = jit(0.45, 0.03, rng);
    c[NeurochemicalId::Cortisol as usize].level = jit(0.08, 0.02, rng);
    state.neurochemicals.recompute_derived();
    state.neurochemicals.valence = jit(0.10, 0.04, rng);

    let s = &mut state.inference_signals;
    s.precision = jit(0.65, 0.04, rng);
    s.surprise_ema = jit(0.06, 0.02, rng);
    s.free_energy = jit(0.35, 0.04, rng);
    s.expected_free_energy = jit(0.30, 0.04, rng);
    s.allostasis_load = jit(0.10, 0.02, rng);
    s.model_maturity = jit(0.55, 0.02, rng);
    s.policy_authority = jit(0.50, 0.03, rng);
    s.user_engagement = jit(0.35, 0.04, rng);
    s.attunement = jit(0.55, 0.03, rng);
    s.dyadic_synchrony = jit(0.0, 0.05, rng);
    state
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

fn main() {
    let mut rng = SplitMix64(0x5EED);
    let mut runtime = FieldRuntime::new().expect("runtime builds");
    let dt = 1.0; // the mind's ~1 Hz observation cadence
    let mut state;

    // Burn-in: let the fibers reach their operating positions.
    for _ in 0..300 {
        state = quiet_state(&mut rng);
        runtime.observe(&state, dt, 0.0).unwrap();
    }

    // Standing disagreement — one simulated hour of quiet host.
    let mut standing = Vec::with_capacity(3600);
    for _ in 0..3600 {
        state = quiet_state(&mut rng);
        let r = runtime.observe(&state, dt, 0.0).unwrap();
        standing.push(r.disagreement);
    }
    standing.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = standing.iter().sum::<f64>() / standing.len() as f64;

    // Excursion: sustained incoherence — surprise pinned high for
    // 30 s while everything else stays quiet.
    let mut excursion = Vec::with_capacity(30);
    for _ in 0..30 {
        state = quiet_state(&mut rng);
        state.inference_signals.surprise_ema = 0.9;
        let r = runtime.observe(&state, dt, 0.0).unwrap();
        excursion.push(r.disagreement);
    }

    const SCALE: f64 = 0.15; // DISAGREEMENT_PRECISION_SCALE
    let atten = |d: f64| 1.0 / (1.0 + d / SCALE);
    println!("standing disagreement (3600 s, quiet host):");
    println!("  mean  {:.4}   attenuates to {:.2}", mean, atten(mean));
    println!(
        "  p50   {:.4}   attenuates to {:.2}",
        percentile(&standing, 0.50),
        atten(percentile(&standing, 0.50))
    );
    println!(
        "  p95   {:.4}   attenuates to {:.2}",
        percentile(&standing, 0.95),
        atten(percentile(&standing, 0.95))
    );
    println!("  p99   {:.4}", percentile(&standing, 0.99));
    println!("  max   {:.4}", standing.last().unwrap());
    println!("excursion (surprise_ema=0.9, 30 s):");
    println!(
        "  disagreement {:.4} -> {:.4}",
        excursion.first().unwrap(),
        excursion.last().unwrap()
    );
    println!(
        "  peak {:.4}",
        excursion.iter().cloned().fold(0.0, f64::max)
    );
    println!();
    println!("reference: SCALE=0.15, floor 0.5 — standing ~0.07–0.08 should");
    println!("attenuate ~33%, excursion d>=0.15 should reach the floor.");
    for d in [0.2, 0.3, 0.4, 0.5, 0.8, 1.0] {
        println!("  d={d:.1} -> precision factor {:.2}", atten(d));
    }
}
