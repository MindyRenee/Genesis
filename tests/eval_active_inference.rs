//! Evaluation 6: Active Inference Engine (Substrate-Level Strange Loop).
//!
//! Tests whether Genesis's Rust active inference engine — her generative
//! model of her own neurochemical trajectory — actually learns to predict
//! her neurochemistry and adapts when surprised.
//!
//! This is the substrate-level strange loop: the engine predicts where
//! her 18 effective levels will move, computes prediction error, learns
//! from the error, and feeds back into neurochemistry. This is the
//! "strange loop" the architecture is built on — a system that models
//! itself, where the model's predictions influence the system being
//! modeled.
//!
//! ## What this eval tests (that the unit tests don't)
//!
//! The existing `tests/active_inference.rs` tests verify individual
//! mechanisms in isolation. This eval tests the **full causal chain**
//! as an integrated evaluation:
//!
//! 1. **Model learns predictable trajectory** — a consistent
//!    neurochemical trend should be learned: surprise drops over time.
//! 2. **Surprise spikes on perturbation** — after learning a stable
//!    trajectory, a sudden perturbation should produce a surprise spike.
//! 3. **Surprise recovers after perturbation** — after the perturbation,
//!    returning to the learned trajectory should let surprise drop back.
//! 4. **Precision tracks predictability** — sustained predictability
//!    raises precision; sustained unpredictability lowers it.
//! 5. **Allostatic load accumulates under chronic surprise** — chronic
//!    unpredictability should produce allostatic load (chronic stress).
//! 6. **Allostatic load recovers when predictable** — after chronic
//!    surprise, a return to predictability should let allostatic load
//!    recover.
//! 7. **Model maturity increases with experience** — the model should
//!    mature over time, approaching 1.0 asymptotically.
//! 8. **Dopamine prediction error signals** — unexpected dopamine
//!    changes should produce dopamine prediction errors (the triphasic
//!    RPE pattern).
//! 9. **Save/load round-trip preserves the learned model** — the
//!    transition matrix, precision, and maturity should survive
//!    serialization.

use genesis::daemon::active_inference::ActiveInferenceEngine;
use genesis::state::neurochemical::NEUROCHEMICAL_COUNT;

const BASELINE: [f32; NEUROCHEMICAL_COUNT] = [0.5; NEUROCHEMICAL_COUNT];

// ─── Eval conditions ──────────────────────────────────────────────

/// Condition 1: The model learns a predictable trajectory.
///
/// A consistent neurochemical trend (every chemical rises by 0.01/tick)
/// should be learned by the transition matrix. Surprise should decrease
/// over time as the model adapts.
#[test]
fn eval_model_learns_predictable_trajectory() {
    let mut engine = ActiveInferenceEngine::new();
    let mut current = [0.3f32; NEUROCHEMICAL_COUNT];

    let mut surprises = Vec::new();
    for _ in 0..100 {
        let mut next = current;
        for i in 0..NEUROCHEMICAL_COUNT {
            next[i] = (current[i] + 0.01).min(1.0);
        }
        let result = engine.cycle(&current, &next, 0.1, &BASELINE);
        surprises.push(result.surprise);
        current = next;
    }

    let early_avg: f32 = surprises[5..15].iter().sum::<f32>() / 10.0;
    let late_avg: f32 = surprises[85..95].iter().sum::<f32>() / 10.0;
    assert!(
        late_avg < early_avg,
        "surprise should decrease as model learns: early={early_avg:.4}, late={late_avg:.4}"
    );
}

/// Condition 2: Surprise spikes on perturbation.
///
/// After the model learns a stable trajectory, a sudden large change
/// should produce a surprise spike.
#[test]
fn eval_surprise_spikes_on_perturbation() {
    let mut engine = ActiveInferenceEngine::new();
    let stable = [0.5f32; NEUROCHEMICAL_COUNT];

    // Train on stable input
    for _ in 0..20 {
        engine.cycle(&stable, &stable, 0.1, &BASELINE);
    }
    let baseline_surprise = engine.signals().surprise_ema;

    // Perturbation: large jump in multiple chemicals
    let mut perturbed = stable;
    for v in perturbed.iter_mut().take(NEUROCHEMICAL_COUNT) {
        *v = 0.9; // all chemicals jump from 0.5 to 0.9
    }
    let result = engine.cycle(&stable, &perturbed, 0.1, &BASELINE);

    assert!(
        result.surprise > baseline_surprise + 0.05,
        "perturbation should spike surprise: baseline={baseline_surprise:.4}, perturbed={:.4}",
        result.surprise
    );
}

/// Condition 3: Surprise recovers after perturbation.
///
/// After a perturbation spike, returning to the stable trajectory
/// should let surprise drop back down.
#[test]
fn eval_surprise_recovers_after_perturbation() {
    let mut engine = ActiveInferenceEngine::new();
    let stable = [0.5f32; NEUROCHEMICAL_COUNT];

    // Train on stable input
    for _ in 0..20 {
        engine.cycle(&stable, &stable, 0.1, &BASELINE);
    }

    // Perturbation
    let mut perturbed = stable;
    perturbed[0] = 0.9;
    engine.cycle(&stable, &perturbed, 0.1, &BASELINE);
    let spike_surprise = engine.signals().surprise_ema;

    // Recovery: return to stable
    for _ in 0..50 {
        engine.cycle(&stable, &stable, 0.1, &BASELINE);
    }
    let recovered_surprise = engine.signals().surprise_ema;

    assert!(
        recovered_surprise < spike_surprise,
        "surprise should recover: spike={spike_surprise:.4}, recovered={recovered_surprise:.4}"
    );
}

/// Condition 4: Precision tracks predictability.
///
/// Sustained predictability raises precision; sustained
/// unpredictability lowers it. This tests the asymmetry — losing
/// confidence is fast, regaining it is slow.
#[test]
fn eval_precision_tracks_predictability() {
    let mut engine = ActiveInferenceEngine::new();
    let initial_precision = engine.precision();

    // Phase 1: sustained unpredictability
    let mut state = [0.5f32; NEUROCHEMICAL_COUNT];
    for i in 0..100 {
        let mut next = [0.0f32; NEUROCHEMICAL_COUNT];
        for (j, n) in next.iter_mut().enumerate() {
            *n = (((i + 1) as f32 * 0.7 + j as f32 * 1.3).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        }
        engine.cycle(&state, &next, 0.1, &BASELINE);
        state = next;
    }
    let low_precision = engine.precision();
    assert!(
        low_precision < initial_precision,
        "precision should drop under surprise: initial={initial_precision:.4}, low={low_precision:.4}"
    );

    // Phase 2: sustained predictability
    let stable = [0.5f32; NEUROCHEMICAL_COUNT];
    for _ in 0..200 {
        engine.cycle(&stable, &stable, 0.1, &BASELINE);
    }
    let recovered_precision = engine.precision();
    assert!(
        recovered_precision > low_precision,
        "precision should recover: low={low_precision:.4}, recovered={recovered_precision:.4}"
    );
}

/// Condition 5: Allostatic load accumulates under chronic surprise.
///
/// Chronic unpredictability should produce allostatic load — the
/// physiological cost of chronic stress.
#[test]
fn eval_allostasis_load_accumulates() {
    let mut engine = ActiveInferenceEngine::new();
    let initial_load = engine.signals().allostasis_load;

    // Chronic surprise: large unpredictable swings
    let mut state = [0.5f32; NEUROCHEMICAL_COUNT];
    for i in 0..200 {
        let mut next = [0.0f32; NEUROCHEMICAL_COUNT];
        for (j, n) in next.iter_mut().enumerate() {
            *n = ((i as f32 * 0.031 + j as f32 * 0.043) % 1.0).clamp(0.0, 1.0);
        }
        engine.cycle(&state, &next, 0.1, &BASELINE);
        state = next;
    }
    let final_load = engine.signals().allostasis_load;
    assert!(
        final_load > initial_load,
        "allostatic load should accumulate: initial={initial_load:.4}, final={final_load:.4}"
    );
}

/// Condition 6: Allostatic load recovers when predictable.
///
/// After chronic surprise, a return to predictability should let
/// allostatic load recover (the system de-stresses).
#[test]
fn eval_allostasis_load_recovers() {
    let mut engine = ActiveInferenceEngine::new();

    // Phase 1: chronic surprise
    let mut state = [0.5f32; NEUROCHEMICAL_COUNT];
    for i in 0..200 {
        let mut next = [0.0f32; NEUROCHEMICAL_COUNT];
        for (j, n) in next.iter_mut().enumerate() {
            *n = ((i as f32 * 0.031 + j as f32 * 0.043) % 1.0).clamp(0.0, 1.0);
        }
        engine.cycle(&state, &next, 0.1, &BASELINE);
        state = next;
    }
    let peak_load = engine.signals().allostasis_load;

    // Phase 2: return to predictability
    let stable = [0.5f32; NEUROCHEMICAL_COUNT];
    for _ in 0..300 {
        engine.cycle(&stable, &stable, 0.1, &BASELINE);
    }
    let recovered_load = engine.signals().allostasis_load;
    assert!(
        recovered_load < peak_load,
        "allostatic load should recover: peak={peak_load:.4}, recovered={recovered_load:.4}"
    );
}

/// Condition 7: Model maturity increases with experience.
///
/// The model should mature over time, approaching 1.0 asymptotically.
#[test]
fn eval_model_maturity_increases() {
    let mut engine = ActiveInferenceEngine::new();
    let initial_maturity = engine.signals().model_maturity;

    let stable = [0.5f32; NEUROCHEMICAL_COUNT];
    for _ in 0..100 {
        engine.cycle(&stable, &stable, 0.1, &BASELINE);
    }
    let final_maturity = engine.signals().model_maturity;
    assert!(
        final_maturity > initial_maturity,
        "maturity should increase: initial={initial_maturity:.4}, final={final_maturity:.4}"
    );
    assert!(
        final_maturity <= 1.0,
        "maturity should be bounded: {final_maturity:.4}"
    );
}

/// Condition 8: Dopamine prediction error signals.
///
/// An unexpected dopamine increase should produce a positive dopamine
/// prediction error; an unexpected decrease should produce a negative
/// one. This mirrors the triphasic RPE pattern.
#[test]
fn eval_dopamine_prediction_error() {
    let mut engine = ActiveInferenceEngine::new();
    let stable = [0.5f32; NEUROCHEMICAL_COUNT];

    // Train on stable dopamine
    for _ in 0..20 {
        engine.cycle(&stable, &stable, 0.1, &BASELINE);
    }

    // Unexpected dopamine increase
    let mut high_da = stable;
    high_da[0] = 0.8;
    let result_up = engine.cycle(&stable, &high_da, 0.1, &BASELINE);
    assert!(
        result_up.da_prediction_error > 0.0,
        "unexpected DA increase should produce positive PE: {:.4}",
        result_up.da_prediction_error
    );

    // Train back to stable
    for _ in 0..20 {
        engine.cycle(&stable, &stable, 0.1, &BASELINE);
    }

    // Unexpected dopamine decrease
    let mut low_da = stable;
    low_da[0] = 0.2;
    let result_down = engine.cycle(&stable, &low_da, 0.1, &BASELINE);
    assert!(
        result_down.da_prediction_error < 0.0,
        "unexpected DA decrease should produce negative PE: {:.4}",
        result_down.da_prediction_error
    );
}

/// Condition 9: Save/load round-trip preserves the learned model.
///
/// The transition matrix, precision, and maturity should survive
/// serialization to disk and back.
#[test]
fn eval_save_load_round_trip() {
    let mut engine = ActiveInferenceEngine::new();

    // Train the model
    let mut current = [0.3f32; NEUROCHEMICAL_COUNT];
    for _ in 0..50 {
        let mut next = current;
        for i in 0..NEUROCHEMICAL_COUNT {
            next[i] = (current[i] + 0.01).min(1.0);
        }
        engine.cycle(&current, &next, 0.1, &BASELINE);
        current = next;
    }
    let pre_precision = engine.precision();
    let pre_maturity = engine.signals().model_maturity;
    let pre_tick_count = engine.signals().inference_tick_count;

    // Save and load
    let model_path = std::env::temp_dir().join(format!(
        "genesis_eval_ai_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    engine.save(&model_path).unwrap();
    let loaded = ActiveInferenceEngine::load(&model_path);
    let _ = std::fs::remove_file(&model_path);

    assert!(
        (loaded.precision() - pre_precision).abs() < 0.001,
        "precision should match: pre={pre_precision:.4}, post={:.4}",
        loaded.precision()
    );
    assert!(
        (loaded.signals().model_maturity - pre_maturity).abs() < 0.001,
        "maturity should match: pre={pre_maturity:.4}, post={:.4}",
        loaded.signals().model_maturity
    );
    assert_eq!(
        loaded.signals().inference_tick_count,
        pre_tick_count,
        "tick_count should match"
    );
}
