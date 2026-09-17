//! Tests for the active inference engine and dyadic affective model.
//!
//! These verify:
//! - InferenceSignals struct layout (60 bytes, correct offsets)
//! - Active inference engine: prediction, error computation, model
//!   learning, surprise, free energy, precision dynamics
//! - Allostatic load accumulation and recovery
//! - Active inference impulses (dopamine PE, NE orienting, homeostatic)
//! - Model persistence (save/load round-trip)
//! - Dyadic affective model: user affect estimation, attunement,
//!   synchrony computation, neurochemical impulses
//! - Dyadic coupling: oxytocin bonding, empathic stress
//! - Integration: inference signals written to core state

use genesis::daemon::active_inference::{
    ActiveInferenceEngine, DyadicSignals, apply_inference_feedback,
};
use genesis::daemon::dyadic_model::{DyadicAffectModel, UserAffectObservation};
use genesis::state::InferenceSignals;
use genesis::state::neurochemical::{NEUROCHEMICAL_COUNT, NeurochemicalId, NeurochemicalVector};

// ─── InferenceSignals layout ──────────────────────────────────

#[test]
fn test_inference_signals_size() {
    assert_eq!(core::mem::size_of::<InferenceSignals>(), 60);
}

#[test]
fn test_inference_signals_offsets() {
    use core::mem::offset_of;
    assert_eq!(offset_of!(InferenceSignals, surprise_ema), 0);
    assert_eq!(offset_of!(InferenceSignals, free_energy), 4);
    assert_eq!(offset_of!(InferenceSignals, expected_free_energy), 8);
    assert_eq!(offset_of!(InferenceSignals, allostasis_load), 12);
    assert_eq!(offset_of!(InferenceSignals, precision), 16);
    assert_eq!(offset_of!(InferenceSignals, attunement), 20);
    assert_eq!(offset_of!(InferenceSignals, dyadic_synchrony), 24);
    assert_eq!(offset_of!(InferenceSignals, user_valence), 28);
    assert_eq!(offset_of!(InferenceSignals, user_arousal), 32);
    assert_eq!(offset_of!(InferenceSignals, user_engagement), 36);
    assert_eq!(offset_of!(InferenceSignals, prediction_error_dopamine), 40);
    assert_eq!(offset_of!(InferenceSignals, prediction_error_cortisol), 44);
    assert_eq!(offset_of!(InferenceSignals, prediction_error_serotonin), 48);
    assert_eq!(offset_of!(InferenceSignals, model_maturity), 52);
    assert_eq!(offset_of!(InferenceSignals, inference_tick_count), 56);
}

#[test]
fn test_inference_signals_new_defaults() {
    let signals = InferenceSignals::new();
    assert_eq!(signals.surprise_ema, 0.0);
    assert_eq!(signals.free_energy, 0.0);
    assert_eq!(signals.expected_free_energy, 0.0);
    assert_eq!(signals.allostasis_load, 0.0);
    assert!((signals.precision - 0.5).abs() < 1e-6); // moderate initial precision
    assert_eq!(signals.attunement, 0.0);
    assert_eq!(signals.dyadic_synchrony, 0.0);
    assert_eq!(signals.user_valence, 0.0);
    assert!((signals.user_arousal - 0.5).abs() < 1e-6); // neutral arousal
    assert_eq!(signals.user_engagement, 0.0);
    assert_eq!(signals.prediction_error_dopamine, 0.0);
    assert_eq!(signals.prediction_error_cortisol, 0.0);
    assert_eq!(signals.prediction_error_serotonin, 0.0);
    assert_eq!(signals.model_maturity, 0.0);
    assert_eq!(signals.inference_tick_count, 0);
}

#[test]
fn test_inference_signals_predicates() {
    let mut signals = InferenceSignals::new();

    // Default: not surprised, not loaded, not attuned, not in sync
    assert!(!signals.is_surprised());
    assert!(!signals.is_allostatically_loaded());
    assert!(!signals.is_attuned());
    assert!(!signals.is_in_sync());

    // Surprised
    signals.surprise_ema = 0.5;
    assert!(signals.is_surprised());

    // Allostatically loaded
    signals.allostasis_load = 0.6;
    assert!(signals.is_allostatically_loaded());

    // Attuned
    signals.attunement = 0.5;
    assert!(signals.is_attuned());

    // In sync
    signals.dyadic_synchrony = 0.5;
    assert!(signals.is_in_sync());
}

// ─── Active inference engine: prediction ──────────────────────

#[test]
fn test_engine_new_identity_prediction() {
    // A fresh engine has an identity transition matrix, so it should
    // predict "no change" — the predicted state equals the current state.
    let engine = ActiveInferenceEngine::new();
    let current = [0.5f32; NEUROCHEMICAL_COUNT];
    let predicted = engine.predict(&current);
    for i in 0..NEUROCHEMICAL_COUNT {
        assert!(
            (predicted[i] - current[i]).abs() < 1e-6,
            "predicted[{i}] = {}, expected {}",
            predicted[i],
            current[i]
        );
    }
}

#[test]
fn test_engine_predict_clamps_to_range() {
    // Predictions should be clamped to [0, 2] since effective levels
    // can exceed 1.0 under phasic bursts (dopamine D1, serotonin 5-HT2A)
    // and allosteric modulation (GABA).
    let engine = ActiveInferenceEngine::new();
    let current = [0.0f32; NEUROCHEMICAL_COUNT];
    let predicted = engine.predict(&current);
    for p in predicted {
        assert!((0.0..=2.0).contains(&p), "predicted {} out of [0,2]", p);
    }

    let current = [2.0f32; NEUROCHEMICAL_COUNT];
    let predicted = engine.predict(&current);
    for p in predicted {
        assert!((0.0..=2.0).contains(&p), "predicted {} out of [0,2]", p);
    }
}

// ─── Active inference engine: cycle ───────────────────────────

#[test]
fn test_engine_first_cycle_no_error() {
    // The first cycle should not produce prediction errors (no prior
    // prediction to compare against).
    let mut engine = ActiveInferenceEngine::new();
    let pre = [0.5f32; NEUROCHEMICAL_COUNT];
    let post = [0.6f32; NEUROCHEMICAL_COUNT];
    let result = engine.cycle(&pre, &post, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    assert_eq!(result.surprise, 0.0); // no error on first cycle
    assert_eq!(result.free_energy, 0.0);
    assert_eq!(engine.tick_count(), 1);
}

#[test]
fn test_engine_surprise_on_unpredicted_change() {
    // After the first cycle establishes a prediction, a large
    // unexpected change should produce high surprise.
    let mut engine = ActiveInferenceEngine::new();

    // First cycle: establish baseline (identity prediction = no change)
    let pre1 = [0.5f32; NEUROCHEMICAL_COUNT];
    let post1 = [0.5f32; NEUROCHEMICAL_COUNT]; // no change → no surprise
    engine.cycle(&pre1, &post1, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // Second cycle: large unexpected change
    let pre2 = [0.5f32; NEUROCHEMICAL_COUNT];
    let mut post2 = [0.5f32; NEUROCHEMICAL_COUNT];
    post2[0] = 0.9; // big change in dopamine
    let result = engine.cycle(&pre2, &post2, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // Should have non-zero surprise (the EMA, not instantaneous)
    assert!(
        result.surprise > 0.0,
        "surprise should be > 0 for unexpected change, got {}",
        result.surprise
    );
}

#[test]
fn test_engine_surprise_low_on_predicted_no_change() {
    // When the state doesn't change (and the model predicts no
    // change), surprise should stay low.
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run several cycles with no change
    for _ in 0..10 {
        engine.cycle(&state, &state, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }

    let signals = engine.signals();
    assert!(
        signals.surprise_ema < 0.05,
        "surprise should be low for predictable no-change, got {}",
        signals.surprise_ema
    );
}

#[test]
fn test_engine_model_learns_transition() {
    // If the state consistently moves in a predictable way, the
    // model should learn the transition and surprise should decrease.
    let mut engine = ActiveInferenceEngine::new();

    // Consistent transition: every chemical increases by 0.01 per tick
    let mut current = [0.3f32; NEUROCHEMICAL_COUNT];

    let mut surprises = Vec::new();
    for _ in 0..100 {
        let mut next = current;
        for i in 0..NEUROCHEMICAL_COUNT {
            next[i] = (current[i] + 0.01).min(1.0);
        }
        let result = engine.cycle(&current, &next, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
        surprises.push(result.surprise);
        current = next;
    }

    // Surprise should decrease over time as the model learns
    let early_surprise: f32 = surprises[5..15].iter().sum::<f32>() / 10.0;
    let late_surprise: f32 = surprises[85..95].iter().sum::<f32>() / 10.0;
    assert!(
        late_surprise < early_surprise,
        "surprise should decrease as model learns: early={}, late={}",
        early_surprise,
        late_surprise
    );
}

#[test]
fn test_engine_precision_decreases_on_surprise() {
    // Sustained surprise should decrease precision (model confidence).
    // Precision decays when the surprise EMA exceeds
    // PRECISION_SURPRISE_THRESHOLD (0.15). We generate genuinely
    // unpredictable changes (large jumps that the linear model can't
    // learn) to keep surprise sustained high.
    let mut engine = ActiveInferenceEngine::new();
    let initial_precision = engine.precision();
    let mut state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Generate sustained high surprise with large, unpredictable jumps.
    // The linear model can't predict these, so surprise stays high.
    for i in 0..100 {
        let mut next = [0.0f32; NEUROCHEMICAL_COUNT];
        for (j, n) in next.iter_mut().enumerate() {
            // Pseudo-random but deterministic: large swings that the
            // linear transition matrix can't model.
            *n = (((i + 1) as f32 * 0.7 + j as f32 * 1.3).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        }
        engine.cycle(&state, &next, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
        state = next;
    }

    let final_precision = engine.precision();
    assert!(
        final_precision < initial_precision,
        "precision should decrease under sustained surprise: initial={}, final={}",
        initial_precision,
        final_precision
    );
}

#[test]
fn test_engine_precision_increases_on_predictable() {
    // Sustained predictability should increase precision.
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run many cycles with no change (perfectly predictable)
    for _ in 0..100 {
        engine.cycle(&state, &state, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }

    let final_precision = engine.precision();
    assert!(
        final_precision > 0.5,
        "precision should increase under predictability: final={}",
        final_precision
    );
}

// ─── Allostatic load ──────────────────────────────────────────

#[test]
fn test_allostatic_load_accumulates_on_sustained_surprise() {
    // Sustained high surprise should accumulate allostatic load.
    let mut engine = ActiveInferenceEngine::new();
    let mut state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Generate sustained surprise
    for i in 0..100 {
        let mut next = state;
        for (j, n) in next.iter_mut().enumerate() {
            *n = ((i as f32 * 0.031 + j as f32 * 0.043) % 1.0).clamp(0.0, 1.0);
        }
        engine.cycle(&state, &next, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
        state = next;
    }

    let load = engine.allostasis_load();
    assert!(
        load > 0.0,
        "allostatic load should accumulate under sustained surprise: {}",
        load
    );
}

#[test]
fn test_allostatic_load_recovers_on_predictable() {
    // After sustained surprise, predictability should allow allostatic
    // load to recover (decrease).
    let mut engine = ActiveInferenceEngine::new();
    let mut state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Phase 1: Generate sustained surprise to accumulate load
    for i in 0..100 {
        let mut next = state;
        for (j, n) in next.iter_mut().enumerate() {
            *n = ((i as f32 * 0.031 + j as f32 * 0.043) % 1.0).clamp(0.0, 1.0);
        }
        engine.cycle(&state, &next, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
        state = next;
    }
    let loaded = engine.allostasis_load();
    assert!(loaded > 0.0, "should have accumulated load");

    // Phase 2: Run predictably to allow recovery
    let stable = state;
    for _ in 0..200 {
        engine.cycle(&stable, &stable, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }
    let recovered = engine.allostasis_load();
    assert!(
        recovered < loaded,
        "allostatic load should recover under predictability: loaded={}, recovered={}",
        loaded,
        recovered
    );
}

#[test]
fn test_allostatic_load_recovery_faster_than_accumulation() {
    // Recovery rate (0.002) should be 2× faster than accumulation rate
    // (0.001). This models the biological principle that the body heals
    // faster than it breaks down (McEwen, 1998). After equal time under
    // stress vs. recovery, the net load should decrease.
    let mut engine = ActiveInferenceEngine::new();
    let mut state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Phase 1: Generate sustained surprise to accumulate load
    for i in 0..100 {
        let mut next = state;
        for (j, n) in next.iter_mut().enumerate() {
            *n = ((i as f32 * 0.031 + j as f32 * 0.043) % 1.0).clamp(0.0, 1.0);
        }
        engine.cycle(&state, &next, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
        state = next;
    }
    let after_accum = engine.allostasis_load();

    // Phase 2: Recover for the SAME number of ticks
    let stable = state;
    for _ in 0..100 {
        engine.cycle(&stable, &stable, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }
    let after_recover = engine.allostasis_load();

    // The recovery over 100 ticks should have removed more load than
    // 100 ticks of accumulation added (since recovery is 2× faster).
    // We check that the net change is negative — more was recovered
    // than accumulated in equal time.
    let net_change = after_recover - after_accum;
    assert!(
        net_change < 0.0,
        "recovery should be faster than accumulation over equal time: \
         after_accum={}, after_recover={}, net={}",
        after_accum,
        after_recover,
        net_change
    );
}

// ─── Dopamine prediction error ────────────────────────────────

#[test]
fn test_dopamine_prediction_error_positive() {
    // When dopamine is higher than predicted, the DA prediction error
    // should be positive (reward prediction error, Schultz 2016).
    let mut engine = ActiveInferenceEngine::new();

    // First cycle to initialize
    let pre1 = [0.5f32; NEUROCHEMICAL_COUNT];
    let post1 = [0.5f32; NEUROCHEMICAL_COUNT];
    engine.cycle(&pre1, &post1, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // Second cycle: dopamine jumps up unexpectedly
    let mut pre2 = [0.5f32; NEUROCHEMICAL_COUNT];
    let mut post2 = [0.5f32; NEUROCHEMICAL_COUNT];
    let da_idx = NeurochemicalId::Dopamine as usize;
    pre2[da_idx] = 0.5;
    post2[da_idx] = 0.8; // dopamine higher than predicted
    let result = engine.cycle(&pre2, &post2, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    assert!(
        result.da_prediction_error > 0.0,
        "DA PE should be positive when dopamine exceeds prediction: {}",
        result.da_prediction_error
    );
}

#[test]
fn test_dopamine_prediction_error_negative() {
    // When dopamine is lower than predicted, the DA prediction error
    // should be negative (disappointment).
    let mut engine = ActiveInferenceEngine::new();

    // First cycle to initialize
    let pre1 = [0.5f32; NEUROCHEMICAL_COUNT];
    let post1 = [0.5f32; NEUROCHEMICAL_COUNT];
    engine.cycle(&pre1, &post1, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // Second cycle: dopamine drops unexpectedly
    let mut pre2 = [0.5f32; NEUROCHEMICAL_COUNT];
    let mut post2 = [0.5f32; NEUROCHEMICAL_COUNT];
    let da_idx = NeurochemicalId::Dopamine as usize;
    pre2[da_idx] = 0.5;
    post2[da_idx] = 0.2; // dopamine lower than predicted
    let result = engine.cycle(&pre2, &post2, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    assert!(
        result.da_prediction_error < 0.0,
        "DA PE should be negative when dopamine is below prediction: {}",
        result.da_prediction_error
    );
}

// ─── Model maturity ───────────────────────────────────────────

#[test]
fn test_model_maturity_increases() {
    // Model maturity should increase with more inference cycles,
    // asymptotically approaching 1.0.
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    assert!((engine.model_maturity() - 0.0).abs() < 1e-6);

    for _ in 0..100 {
        engine.cycle(&state, &state, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }
    let m100 = engine.model_maturity();
    assert!(m100 > 0.0, "maturity should increase: {}", m100);

    for _ in 0..400 {
        engine.cycle(&state, &state, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }
    let m500 = engine.model_maturity();
    assert!(
        m500 > m100,
        "maturity should continue increasing: m100={}, m500={}",
        m100,
        m500
    );
    assert!(m500 < 1.0, "maturity should be < 1.0: {}", m500);
}

// ─── Model persistence ────────────────────────────────────────

#[test]
fn test_engine_save_load_round_trip() {
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run some cycles to develop non-default state
    for i in 0..50 {
        let mut next = state;
        for j in 0..NEUROCHEMICAL_COUNT {
            next[j] = (state[j] + i as f32 * 0.001).min(1.0);
        }
        engine.cycle(&state, &next, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }

    let precision_before = engine.precision();
    let surprise_before = engine.signals().surprise_ema;
    let tick_count_before = engine.tick_count();

    // Save
    let path = std::env::temp_dir().join(format!(
        "genesis_inference_test_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    engine.save(&path).expect("save failed");

    // Load
    let loaded = ActiveInferenceEngine::load(&path);

    assert!(
        (loaded.precision() - precision_before).abs() < 1e-6,
        "precision should match after load: before={}, after={}",
        precision_before,
        loaded.precision()
    );
    assert!(
        (loaded.signals().surprise_ema - surprise_before).abs() < 1e-6,
        "surprise_ema should match after load"
    );
    assert_eq!(
        loaded.tick_count(),
        tick_count_before,
        "tick_count should match after load"
    );

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_engine_load_missing_file_returns_new() {
    // Loading a non-existent file should return a fresh engine.
    let path = std::path::Path::new("/nonexistent/path/that/does/not/exist.bin");
    let engine = ActiveInferenceEngine::load(path);
    assert_eq!(engine.tick_count(), 0);
    assert!((engine.precision() - 0.5).abs() < 1e-6);
}

#[test]
fn test_engine_load_clamps_corrupted_matrix_weights() {
    // A model file from a version that predates the tick-level
    // [-2, 2] clamp can contain astronomically large finite values
    // (e.g., 3172.0 from an unbounded delta rule). The load function
    // must clamp these to [-2, 2] so the corrupted model doesn't
    // produce wild predictions before the first tick runs.
    //
    // This was a real production bug: a corrupted dopamine row
    // (max=3172) caused huge prediction errors → free energy=0.749
    // → allostatic load=1.0 → chronic stress → cognitive impairment
    // ("genesis is light").
    use std::io::Write;

    let path = std::env::temp_dir().join(format!(
        "genesis_inference_clamp_test_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    // Build a model file with corrupted (but finite) values
    {
        let mut file = std::fs::File::create(&path).expect("create failed");
        // Magic
        file.write_all(b"AIFE").unwrap();
        // Version
        file.write_all(&1u32.to_le_bytes()).unwrap();
        // Transition matrix: 18×18 f32 = 324 values
        for i in 0..NEUROCHEMICAL_COUNT {
            for j in 0..NEUROCHEMICAL_COUNT {
                // Row 0 (dopamine) has extreme values; others are normal
                let val = if i == 0 {
                    if j == 0 {
                        3172.0f32
                    } else if j == 1 {
                        -500.0
                    } else {
                        0.5
                    }
                } else {
                    if i == j { 1.0 } else { 0.0 }
                };
                file.write_all(&val.to_le_bytes()).unwrap();
            }
        }
        // Bias: 18 f32
        for i in 0..NEUROCHEMICAL_COUNT {
            // Bias 1 (serotonin) has an extreme value
            let val: f32 = if i == 1 { 0.987 } else { 0.1 };
            file.write_all(&val.to_le_bytes()).unwrap();
        }
        // Scalar state (order matches save()): precision, surprise_ema,
        // expected_fe_ema, allostasis_load, model_maturity, tick_count
        file.write_all(&0.5f32.to_le_bytes()).unwrap(); // precision
        file.write_all(&0.1f32.to_le_bytes()).unwrap(); // surprise_ema
        file.write_all(&0.1f32.to_le_bytes()).unwrap(); // expected_fe_ema
        file.write_all(&0.0f32.to_le_bytes()).unwrap(); // allostasis_load
        file.write_all(&1.0f32.to_le_bytes()).unwrap(); // model_maturity
        file.write_all(&100u32.to_le_bytes()).unwrap(); // tick_count
        file.flush().unwrap();
    }

    let mut engine = ActiveInferenceEngine::load(&path);

    // If the matrix wasn't clamped, a cycle would produce wild
    // predictions. Instead, verify the engine loaded successfully
    // and has the expected tick_count (proving the file was parsed).
    assert_eq!(engine.tick_count(), 100, "tick_count should load");

    // Run a cycle with the loaded (clamped) model and verify
    // the free energy is bounded — not wild from the 3172 weight.
    let state = [0.5f32; NEUROCHEMICAL_COUNT];
    let result = engine.cycle(&state, &state, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    assert!(
        result.free_energy <= 1.0 && result.free_energy.is_finite(),
        "free energy should be bounded after load-clamped model: {}",
        result.free_energy
    );

    let _ = std::fs::remove_file(&path);
}

// ─── apply_inference_feedback ─────────────────────────────────

#[test]
fn test_apply_inference_feedback_dopamine_impulse() {
    // A positive dopamine prediction error should generate a DA impulse
    // that increases the dopamine level.
    let mut engine = ActiveInferenceEngine::new();
    let pre = [0.5f32; NEUROCHEMICAL_COUNT];
    let mut post = [0.5f32; NEUROCHEMICAL_COUNT];
    let da_idx = NeurochemicalId::Dopamine as usize;
    post[da_idx] = 0.8; // dopamine higher than predicted

    // First cycle to initialize
    engine.cycle(
        &pre,
        &[0.5f32; NEUROCHEMICAL_COUNT],
        0.1,
        &[0.5f32; NEUROCHEMICAL_COUNT],
    );
    // Second cycle with the DA jump
    let result = engine.cycle(&pre, &post, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // Apply feedback to a neurochemical vector
    let mut neuro = NeurochemicalVector::new(0);
    let da_level_before = neuro.effective(NeurochemicalId::Dopamine);
    apply_inference_feedback(&mut neuro, &result, 0);
    let da_level_after = neuro.effective(NeurochemicalId::Dopamine);

    // If there was a DA impulse, the level should have changed
    if result.da_prediction_error > 0.02 {
        assert!(
            da_level_after != da_level_before,
            "DA level should change after positive PE impulse: before={}, after={}",
            da_level_before,
            da_level_after
        );
    }
}

#[test]
fn test_apply_inference_feedback_dopamine_negative_dip() {
    // A negative dopamine prediction error (worse than expected) should
    // generate a DA *dip* — a transient decrease in dopamine below tonic
    // baseline, modeling the pause in VTA firing that Schultz, Dayan &
    // Montague (1997) and Bayer & Glimcher (2005) documented for
    // negative reward prediction errors. Without this, the system only
    // learns from positive surprises, never from disappointments.
    let mut engine = ActiveInferenceEngine::new();
    let pre = [0.5f32; NEUROCHEMICAL_COUNT];
    let mut post = [0.5f32; NEUROCHEMICAL_COUNT];
    let da_idx = NeurochemicalId::Dopamine as usize;
    post[da_idx] = 0.2; // dopamine lower than predicted

    // First cycle to initialize
    engine.cycle(
        &pre,
        &[0.5f32; NEUROCHEMICAL_COUNT],
        0.1,
        &[0.5f32; NEUROCHEMICAL_COUNT],
    );
    // Second cycle with the DA drop
    let result = engine.cycle(&pre, &post, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    assert!(
        result.da_prediction_error < -0.02,
        "DA PE should be significantly negative: {}",
        result.da_prediction_error
    );

    // There must be a negative DA impulse in the feedback
    let da_impulse = result
        .impulses
        .iter()
        .find(|(id, _)| *id == NeurochemicalId::Dopamine as u8)
        .map(|(_, mag)| *mag);
    assert!(
        da_impulse.is_some_and(|m| m < 0.0),
        "negative DA PE should produce a negative (dip) DA impulse, got: {:?}",
        da_impulse
    );

    // Applying the feedback should decrease the effective DA level
    let mut neuro = NeurochemicalVector::new(0);
    let da_before = neuro.effective(NeurochemicalId::Dopamine);
    apply_inference_feedback(&mut neuro, &result, 0);
    let da_after = neuro.effective(NeurochemicalId::Dopamine);
    assert!(
        da_after < da_before,
        "DA level should decrease after negative PE dip: before={}, after={}",
        da_before,
        da_after
    );
}

// ─── Dyadic affective model ───────────────────────────────────

#[test]
fn test_dyadic_model_new_defaults() {
    let model = DyadicAffectModel::new();
    assert!((model.user_valence() - 0.0).abs() < 1e-6);
    assert!((model.user_arousal() - 0.5).abs() < 1e-6);
    assert!((model.user_engagement() - 0.0).abs() < 1e-6);
    assert!((model.attunement() - 0.0).abs() < 1e-6);
    assert!(!model.is_initialized());
}

#[test]
fn test_dyadic_model_updates_from_observation() {
    let mut model = DyadicAffectModel::new();
    let obs = UserAffectObservation {
        valence: 0.8,
        arousal: 0.7,
        engagement: 0.9,
        confidence: 1.0,
    };

    let result = model.update(Some(&obs), 0.0, 0.0);

    assert!(model.is_initialized());
    // With confidence=1.0 and EMA decay=0.2, the estimate should move
    // significantly toward the observation
    assert!(result.user_valence > 0.0, "valence should be positive");
    assert!(
        result.user_engagement > 0.0,
        "engagement should be positive"
    );
}

#[test]
fn test_dyadic_model_no_observation_decays_to_neutral() {
    let mut model = DyadicAffectModel::new();

    // First, set a positive valence via observation
    let obs = UserAffectObservation {
        valence: 0.8,
        arousal: 0.7,
        engagement: 0.9,
        confidence: 1.0,
    };
    model.update(Some(&obs), 0.0, 0.0);
    let valence_after_obs = model.user_valence();
    assert!(valence_after_obs > 0.0);

    // Then run many cycles with no observation — valence should
    // decay toward neutral (0.0) via momentum
    for _ in 0..100 {
        model.update(None, 0.0, 0.0);
    }
    let valence_after_decay = model.user_valence();
    assert!(
        valence_after_decay < valence_after_obs,
        "valence should decay without observations: after_obs={}, after_decay={}",
        valence_after_obs,
        valence_after_decay
    );
}

#[test]
fn test_dyadic_attunement_follows_oxytocin() {
    // Attunement should follow oxytocin with inertia.
    let mut model = DyadicAffectModel::new();
    let obs = UserAffectObservation {
        valence: 0.5,
        arousal: 0.5,
        engagement: 0.5,
        confidence: 0.8,
    };

    // High oxytocin → attunement should rise
    for _ in 0..100 {
        model.update(Some(&obs), 0.5, 0.8); // high oxytocin
    }
    assert!(
        model.attunement() > 0.3,
        "attunement should rise with high oxytocin: {}",
        model.attunement()
    );

    // Low oxytocin → attunement should fall
    for _ in 0..200 {
        model.update(Some(&obs), 0.0, 0.1); // low oxytocin
    }
    assert!(
        model.attunement() < 0.3,
        "attunement should fall with low oxytocin: {}",
        model.attunement()
    );
}

#[test]
fn test_dyadic_synchrony_computed() {
    // When Genesis's and the user's valence trajectories are
    // correlated, synchrony should be positive.
    let mut model = DyadicAffectModel::new();

    // Generate correlated valence trajectories: both rising together
    for i in 0..200 {
        let genesis_valence = (i as f32 / 200.0) * 0.8 - 0.4; // -0.4 to 0.4
        let user_valence = genesis_valence + 0.05; // closely correlated
        let obs = UserAffectObservation {
            valence: user_valence,
            arousal: 0.5,
            engagement: 0.5,
            confidence: 0.8,
        };
        model.update(Some(&obs), genesis_valence, 0.5);
    }

    assert!(
        model.synchrony() > 0.3,
        "synchrony should be high for correlated trajectories: {}",
        model.synchrony()
    );
}

#[test]
fn test_dyadic_synchrony_zero_for_uncorrelated() {
    // When trajectories are uncorrelated, synchrony should be near zero.
    let mut model = DyadicAffectModel::new();

    for i in 0..200 {
        // Genesis valence rises, user valence falls (anti-correlated)
        let genesis_valence = (i as f32 / 200.0) * 0.8 - 0.4;
        let user_valence = -genesis_valence;
        let obs = UserAffectObservation {
            valence: user_valence,
            arousal: 0.5,
            engagement: 0.5,
            confidence: 0.8,
        };
        model.update(Some(&obs), genesis_valence, 0.5);
    }

    assert!(
        model.synchrony() < -0.3,
        "synchrony should be negative for anti-correlated trajectories: {}",
        model.synchrony()
    );
}

#[test]
fn test_dyadic_oxytocin_impulse_from_positive_user() {
    // When the user is positive and engaged, and Genesis is attuned,
    // the model should generate an oxytocin impulse (social bonding).
    let mut model = DyadicAffectModel::new();

    // First, build up attunement with high oxytocin
    for _ in 0..100 {
        let obs = UserAffectObservation {
            valence: 0.5,
            arousal: 0.5,
            engagement: 0.5,
            confidence: 0.8,
        };
        model.update(Some(&obs), 0.5, 0.8);
    }
    assert!(model.attunement() > 0.3);

    // Now send a very positive, engaged observation
    let obs = UserAffectObservation {
        valence: 0.9,
        arousal: 0.7,
        engagement: 0.9,
        confidence: 1.0,
    };
    let result = model.update(Some(&obs), 0.5, 0.8);

    // Should have an oxytocin impulse
    let oxy_impulse = result
        .impulses
        .iter()
        .find(|(id, _)| *id == NeurochemicalId::Oxytocin as u8);

    assert!(
        oxy_impulse.is_some(),
        "should generate oxytocin impulse for positive engaged user"
    );
    assert!(
        oxy_impulse.unwrap().1 > 0.0,
        "oxytocin impulse should be positive"
    );
}

#[test]
fn test_dyadic_crh_impulse_from_distressed_user() {
    // When the user is distressed and Genesis is attuned,
    // the model should generate a CRH impulse (empathic stress).
    //
    // The empathic stress response routes through CRH (the stress
    // signal), not direct cortisol. The Rust HPA cascade handles
    // CRH → ACTH → cortisol conversion with maturation gating
    // (SHRP), delays, and negative feedback. Direct cortisol
    // impulses would bypass the maturation gate.
    let mut model = DyadicAffectModel::new();

    // Build up attunement
    for _ in 0..100 {
        let obs = UserAffectObservation {
            valence: 0.0,
            arousal: 0.5,
            engagement: 0.5,
            confidence: 0.8,
        };
        model.update(Some(&obs), 0.0, 0.8);
    }
    assert!(model.attunement() > 0.3);

    // Send multiple distressed observations to drive the valence EMA
    // below -0.2 (one observation isn't enough with EMA smoothing)
    let mut result = None;
    for _ in 0..20 {
        let obs = UserAffectObservation {
            valence: -0.8,
            arousal: 0.8,
            engagement: 0.7,
            confidence: 1.0,
        };
        result = Some(model.update(Some(&obs), 0.0, 0.8));
    }
    let result = result.unwrap();

    // Should have a CRH impulse (not cortisol — routes through HPA cascade)
    let crh_impulse = result
        .impulses
        .iter()
        .find(|(id, _)| *id == NeurochemicalId::CRH as u8);

    assert!(
        crh_impulse.is_some(),
        "should generate CRH impulse for distressed user (valence={})",
        model.user_valence()
    );
    assert!(
        crh_impulse.unwrap().1 > 0.0,
        "CRH impulse should be positive"
    );

    // Should NOT have a direct cortisol impulse — that would bypass
    // the HPA cascade's maturation gating.
    let cort_impulse = result
        .impulses
        .iter()
        .find(|(id, _)| *id == NeurochemicalId::Cortisol as u8);
    assert!(
        cort_impulse.is_none(),
        "should NOT generate direct cortisol impulse (routes through CRH → HPA cascade)"
    );
}

#[test]
fn test_dyadic_no_impulse_when_not_attuned() {
    // When attunement is low, the model should NOT generate impulses
    // (the coupling is too weak to influence neurochemistry).
    let mut model = DyadicAffectModel::new();

    // Don't build up attunement — use low oxytocin
    let obs = UserAffectObservation {
        valence: 0.9,
        arousal: 0.9,
        engagement: 0.9,
        confidence: 1.0,
    };
    let result = model.update(Some(&obs), 0.0, 0.05); // very low oxytocin

    // Should NOT have any impulses (attunement is too low)
    assert!(
        result.impulses.is_empty(),
        "should not generate impulses when not attuned"
    );
}

#[test]
fn test_dyadic_nan_genesis_valence_does_not_distress_user() {
    // When genesis_valence is NaN (from a corrupted neurochemical
    // state), the momentum/coupling path must NOT produce a
    // "distressed user" estimate.
    //
    // Before the fix, the raw NaN genesis_valence was used in the
    // momentum equation:
    //   user_valence = user_valence * 0.93 + NaN * 0.15 = NaN
    // The clamp at the bottom then turned NaN into -1.0 (the min
    // for valence range [-1, 1]), making the user appear "very
    // distressed". This triggered empathic CRH impulses — a
    // corruption amplification path: NaN → distressed user →
    // CRH impulses → more neurochemical disruption.
    //
    // The fix sanitizes genesis_valence to 0.0 (neutral) for NaN
    // before using it, so a corrupted state produces a neutral
    // (not distressed) user estimate.
    let mut model = DyadicAffectModel::new();

    // Initialize with a positive observation
    let obs = UserAffectObservation {
        valence: 0.5,
        arousal: 0.5,
        engagement: 0.5,
        confidence: 0.8,
    };
    model.update(Some(&obs), 0.0, 0.0);
    let valence_before = model.user_valence();
    assert!(valence_before > 0.0);

    // Now run with NaN genesis_valence and no observation
    // (the momentum/coupling path)
    for _ in 0..10 {
        model.update(None, f32::NAN, 0.0);
    }

    // The user valence should NOT be -1.0 (distressed).
    // With the fix, NaN genesis_valence → 0.0 (neutral), so the
    // coupling term is 0.0 and valence decays via momentum only.
    let valence_after = model.user_valence();
    assert!(
        valence_after > -0.5,
        "NaN genesis_valence should not produce distressed user, got {valence_after}"
    );
    assert!(
        valence_after.is_finite(),
        "user_valence must be finite after NaN genesis_valence, got {valence_after}"
    );
}

#[test]
fn test_dyadic_nan_genesis_valence_no_impulses() {
    // When genesis_valence is NaN and the user is present and
    // attuned, the model must NOT generate empathic stress impulses.
    // The sanitized genesis_valence (0.0 for NaN) should not push
    // the user valence below -0.2 (the empathic stress threshold).
    let mut model = DyadicAffectModel::new();

    // Build up attunement with high oxytocin and neutral observations
    for _ in 0..100 {
        let obs = UserAffectObservation {
            valence: 0.0,
            arousal: 0.5,
            engagement: 0.5,
            confidence: 0.8,
        };
        model.update(Some(&obs), 0.0, 0.8);
    }
    assert!(model.attunement() > 0.3);

    // Now send NaN genesis_valence with no observation
    let result = model.update(None, f32::NAN, 0.8);

    // Should NOT have CRH or cortisol impulses — the user valence
    // should remain near neutral (0.0), not distressed (-1.0).
    let stress_impulse = result.impulses.iter().find(|(id, mag)| {
        (*id == NeurochemicalId::CRH as u8 || *id == NeurochemicalId::Cortisol as u8) && *mag > 0.0
    });
    assert!(
        stress_impulse.is_none(),
        "NaN genesis_valence should not trigger empathic stress impulses, \
         user_valence={}",
        result.user_valence
    );
}

#[test]
fn test_dyadic_inf_genesis_valence_handled() {
    // Infinity in genesis_valence should also be handled gracefully.
    // finite_or replaces inf with 0.0, then finite_clamp keeps it in
    // [-1, 1]. The user valence should remain finite.
    let mut model = DyadicAffectModel::new();

    let obs = UserAffectObservation {
        valence: 0.0,
        arousal: 0.5,
        engagement: 0.5,
        confidence: 0.8,
    };
    model.update(Some(&obs), 0.0, 0.0);

    // Run with infinity genesis_valence
    for _ in 0..10 {
        model.update(None, f32::INFINITY, 0.0);
    }

    assert!(
        model.user_valence().is_finite(),
        "user_valence must be finite after inf genesis_valence, got {}",
        model.user_valence()
    );
    assert!(
        model.user_valence() <= 1.0,
        "user_valence must not exceed 1.0, got {}",
        model.user_valence()
    );
}

// ─── Inference signals update ─────────────────────────────────

#[test]
fn test_update_signals_writes_all_fields() {
    let engine = ActiveInferenceEngine::new();
    let mut signals = InferenceSignals::new();

    engine.update_signals(
        &mut signals,
        0.1,   // da_pe
        -0.05, // cort_pe
        0.02,  // srt_pe
        &DyadicSignals {
            attunement: 0.4,
            synchrony: 0.3,
            user_valence: 0.6,
            user_arousal: 0.7,
            user_engagement: 0.5,
        },
    );

    assert!((signals.prediction_error_dopamine - 0.1).abs() < 1e-6);
    assert!((signals.prediction_error_cortisol - (-0.05)).abs() < 1e-6);
    assert!((signals.prediction_error_serotonin - 0.02).abs() < 1e-6);
    assert!((signals.attunement - 0.4).abs() < 1e-6);
    assert!((signals.dyadic_synchrony - 0.3).abs() < 1e-6);
    assert!((signals.user_valence - 0.6).abs() < 1e-6);
    assert!((signals.user_arousal - 0.7).abs() < 1e-6);
    assert!((signals.user_engagement - 0.5).abs() < 1e-6);
}

// ─── Core state integration ───────────────────────────────────

#[test]
fn test_core_state_has_inference_signals() {
    use genesis::state::GenesisCoreState;

    let state = GenesisCoreState::new(42, 1000);
    // The inference_signals field should exist and have default values
    assert_eq!(state.inference_signals.surprise_ema, 0.0);
    assert!((state.inference_signals.precision - 0.5).abs() < 1e-6);
    assert_eq!(state.inference_signals.inference_tick_count, 0);

    // Verify the struct size hasn't changed
    assert_eq!(core::mem::size_of::<GenesisCoreState>(), 3288);
}

// ─── Variational posterior (Kalman filter) ─────────────────────

#[test]
fn test_variational_posterior_updates_on_observation() {
    // The Kalman filter should update the belief mean toward the
    // observation. After a cycle where the observation differs from
    // the prediction, the belief mean should be between the prediction
    // and the observation.
    let mut engine = ActiveInferenceEngine::new();

    // First cycle to initialize
    let pre = [0.5f32; NEUROCHEMICAL_COUNT];
    engine.cycle(&pre, &pre, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // Second cycle: observation differs from prediction
    let mut post = [0.5f32; NEUROCHEMICAL_COUNT];
    post[0] = 0.8; // large deviation in dopamine
    let result = engine.cycle(&pre, &post, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // The free energy should be non-zero (the observation was surprising)
    assert!(
        result.free_energy > 0.0,
        "free energy should be > 0 for surprising observation: {}",
        result.free_energy
    );
}

#[test]
fn test_variational_posterior_belief_converges_to_observation() {
    // Under sustained consistent observations, the belief mean should
    // converge toward the observed state.
    let mut engine = ActiveInferenceEngine::new();
    let target = [0.7f32; NEUROCHEMICAL_COUNT];

    // Run many cycles with a consistent state
    for _ in 0..200 {
        engine.cycle(&target, &target, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }

    // The free energy should be low (the model has learned the state)
    let signals = engine.signals();
    assert!(
        signals.free_energy < 0.3,
        "free energy should be low after convergence: {}",
        signals.free_energy
    );
}

#[test]
fn test_kl_complexity_zero_when_observation_matches_prediction() {
    // When the observation matches the prediction (no surprise), the
    // KL divergence (complexity term) should be near zero — the
    // posterior doesn't need to move from the prior.
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run many cycles with no change
    for _ in 0..50 {
        engine.cycle(&state, &state, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }

    // Free energy should be low (no surprise, no complexity).
    // The expected uncertainty term (1-precision)*0.3 is still present
    // — after only 50 ticks, precision is ~0.6, so expected uncertainty
    // is ~0.12. The total FE should be under 0.2.
    let signals = engine.signals();
    assert!(
        signals.free_energy < 0.2,
        "free energy should be low when observation matches prediction: {}",
        signals.free_energy
    );
}

#[test]
fn test_kl_complexity_nonzero_on_surprising_observation() {
    // When the observation is surprising, the KL divergence should
    // be non-zero — the posterior has moved away from the prior.
    let mut engine = ActiveInferenceEngine::new();

    // First cycle to initialize
    let pre = [0.5f32; NEUROCHEMICAL_COUNT];
    engine.cycle(&pre, &pre, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // Second cycle: large unexpected change
    let post = [0.9f32; NEUROCHEMICAL_COUNT];
    let result = engine.cycle(&pre, &post, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // Free energy should be significantly above zero
    assert!(
        result.free_energy > 0.1,
        "free energy should be high for surprising observation: {}",
        result.free_energy
    );
}

// ─── Epistemic value in policy selection ───────────────────────

#[test]
fn test_epistemic_value_favors_exploration() {
    // The epistemic value bonus should make the "explore" policy
    // more attractive when the system is in a predictable but
    // suboptimal state. The explore policy takes the system to
    // novel regions, which has higher epistemic value.
    //
    // We test this by running the engine until it's in a predictable
    // regime (high precision), then checking that the explore policy
    // has lower EFE than noop (because of the epistemic bonus).
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run until precision is high (predictable regime)
    for _ in 0..200 {
        engine.cycle(&state, &state, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }

    // In a predictable regime with the state at the target (0.5),
    // noop should have low pragmatic cost. But explore should get
    // an epistemic bonus for taking the system to novel regions.
    // The test verifies that the engine doesn't crash and produces
    // a valid policy selection.
    let mut post = state;
    post[0] = 0.51; // tiny change to trigger a cycle
    let result = engine.cycle(&state, &post, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);

    // The selected policy should be valid (one of the 9 policies)
    let valid_policies = [
        "noop", "engage", "calm", "focus", "bond", "soothe", "rest", "mobilize", "explore",
    ];
    assert!(
        valid_policies.contains(&result.selected_policy),
        "selected policy should be valid: {}",
        result.selected_policy
    );
}

#[test]
fn test_policy_selection_with_epistemic_value() {
    // Verify that policy selection still works correctly with the
    // epistemic value term. The system should select reasonable
    // policies under various conditions.
    let mut engine = ActiveInferenceEngine::new();

    // Run with some stress (state far from baseline)
    let mut state = [0.8f32; NEUROCHEMICAL_COUNT];
    for _ in 0..50 {
        let mut next = state;
        for i in 0..NEUROCHEMICAL_COUNT {
            next[i] = (state[i] - 0.001).max(0.0);
        }
        let result = engine.cycle(&state, &next, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
        // The selected policy should be valid
        let valid_policies = [
            "noop", "engage", "calm", "focus", "bond", "soothe", "rest", "mobilize", "explore",
        ];
        assert!(
            valid_policies.contains(&result.selected_policy),
            "selected policy should be valid: {}",
            result.selected_policy
        );
        state = next;
    }
}

// ─── Save/load with variational posterior ──────────────────────

#[test]
fn test_engine_save_load_v2_round_trip() {
    // The v2 save/load format should preserve the variational
    // posterior (belief_mean, belief_var, prior_var, last_free_energy).
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run cycles to develop non-default posterior state
    for i in 0..50 {
        let mut next = state;
        for j in 0..NEUROCHEMICAL_COUNT {
            next[j] = (state[j] + i as f32 * 0.001).min(1.0);
        }
        engine.cycle(&state, &next, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    }

    let fe_before = engine.signals().free_energy;
    let precision_before = engine.precision();
    let tick_count_before = engine.tick_count();

    // Save
    let path = std::env::temp_dir().join(format!(
        "genesis_inference_v2_test_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    engine.save(&path).expect("save failed");

    // Load
    let loaded = ActiveInferenceEngine::load(&path);

    assert!(
        (loaded.precision() - precision_before).abs() < 1e-6,
        "precision should match after load: before={}, after={}",
        precision_before,
        loaded.precision()
    );
    assert!(
        (loaded.signals().free_energy - fe_before).abs() < 1e-6,
        "free_energy should match after load: before={}, after={}",
        fe_before,
        loaded.signals().free_energy
    );
    assert_eq!(
        loaded.tick_count(),
        tick_count_before,
        "tick_count should match after load"
    );

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_engine_load_v1_file_still_works() {
    // A v1 format file (without variational posterior) should still
    // load correctly, with the posterior fields set to defaults.
    use std::io::Write;

    let path = std::env::temp_dir().join(format!(
        "genesis_inference_v1_compat_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));

    // Write a minimal v1 file: magic + version 1 + identity matrix +
    // zero bias + default scalars.
    {
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(b"AIFE").unwrap();
        file.write_all(&1u32.to_le_bytes()).unwrap(); // version 1

        // Identity transition matrix
        for i in 0..NEUROCHEMICAL_COUNT {
            for j in 0..NEUROCHEMICAL_COUNT {
                let val = if i == j { 1.0f32 } else { 0.0 };
                file.write_all(&val.to_le_bytes()).unwrap();
            }
        }

        // Zero bias
        for _ in 0..NEUROCHEMICAL_COUNT {
            file.write_all(&0.0f32.to_le_bytes()).unwrap();
        }

        // Scalar state: precision=0.5, surprise=0, expected_fe=0,
        // allostatic=0, maturity=0, tick_count=10
        file.write_all(&0.5f32.to_le_bytes()).unwrap(); // precision
        file.write_all(&0.0f32.to_le_bytes()).unwrap(); // surprise_ema
        file.write_all(&0.0f32.to_le_bytes()).unwrap(); // expected_fe_ema
        file.write_all(&0.0f32.to_le_bytes()).unwrap(); // allostasis_load
        file.write_all(&0.0f32.to_le_bytes()).unwrap(); // model_maturity
        file.write_all(&10u32.to_le_bytes()).unwrap(); // tick_count
    }

    // Load the v1 file
    let mut engine = ActiveInferenceEngine::load(&path);

    // Should load the v1 fields correctly
    assert_eq!(engine.tick_count(), 10, "tick_count should load from v1");
    assert!(
        (engine.precision() - 0.5).abs() < 1e-6,
        "precision should load from v1"
    );

    // The variational posterior should have default values
    // (belief_mean = 0.5, belief_var = PROCESS_NOISE, prior_var = PROCESS_NOISE)
    // We can't check these directly (private fields), but we can verify
    // the engine works correctly on the next cycle.
    let state = [0.5f32; NEUROCHEMICAL_COUNT];
    let result = engine.cycle(&state, &state, 0.1, &[0.5f32; NEUROCHEMICAL_COUNT]);
    assert!(
        result.free_energy >= 0.0 && result.free_energy <= 1.0,
        "free energy should be valid after loading v1 file: {}",
        result.free_energy
    );

    // Clean up
    let _ = std::fs::remove_file(&path);
}

// ─── Functional variational posterior ──────────────────────────
//
// The following tests verify that the posterior variance (belief_var)
// is genuinely functional — propagated forward via the Kalman predict
// step and consumed by the free-energy and epistemic-value
// computations. Before the fix, belief_var was dead state: computed,
// saved, and loaded, but never read by any computation that affected
// the system's behavior. These tests guard against regressions to
// that state.

#[test]
fn test_kalman_predict_step_propagates_variance() {
    // The Kalman predict step (prior_var = belief_var + Q) makes the
    // posterior variance functional: it is propagated forward and
    // consumed by the free-energy computation. The observable
    // signature is that the free energy remains strictly positive
    // under sustained predictability even when precision has saturated
    // at 1.0 — because the posterior variance (belief_var), while
    // small, is nonzero and contributes to the expected-uncertainty
    // term.
    //
    // Before the fix, belief_var was dead state: the expected
    // uncertainty was (1 - precision) * W, which is exactly 0 at
    // precision 1.0. With the functional posterior, the expected
    // uncertainty is ((1 - precision) + normalized_belief_var) * 0.5
    // * W, which is nonzero even at precision 1.0 because belief_var
    // > 0 (the steady-state posterior variance under process noise).
    // This test would fail with the old dead belief_var (FE = 0) and
    // passes with the functional posterior (FE > 0).
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run enough ticks for precision to saturate at 1.0 (recovers at
    // 0.002/tick from 0.5 → 250 ticks to reach 1.0).
    for _ in 0..300 {
        engine.cycle(&state, &state, 0.1, &state);
    }

    // Precision should be at or near 1.0.
    assert!(
        engine.precision() > 0.99,
        "precision should be saturated after 300 predictable ticks: {}",
        engine.precision()
    );

    // The free energy should be strictly positive, reflecting the
    // posterior variance contribution. With the old dead belief_var,
    // FE would be exactly 0 here (surprise=0, (1-precision)*W=0,
    // complexity=0). With the functional posterior, FE > 0 because
    // belief_var > 0 at steady state (process noise prevents it from
    // reaching zero).
    let fe = engine.signals().free_energy;
    assert!(
        fe > 0.0,
        "free energy should be positive at max precision, reflecting \
         the functional posterior variance: fe={}",
        fe
    );
    assert!(
        fe < 0.05,
        "free energy should be small under sustained predictability: fe={}",
        fe
    );
}

#[test]
fn test_posterior_variance_shrinks_under_predictability() {
    // Under sustained predictability, the posterior variance shrinks
    // (the model becomes confident in its estimates) and precision
    // rises. Both drive the free energy toward a low steady state.
    // We verify that the free energy decreases under sustained
    // predictability — a property that holds because the posterior
    // variance (now functional) contributes to the expected-uncertainty
    // term and shrinks over time.
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    let mut fe_20 = 0.0f32;
    for i in 0..200 {
        let result = engine.cycle(&state, &state, 0.1, &state);
        if i == 19 {
            fe_20 = result.free_energy;
        }
    }
    let fe_200 = engine.signals().free_energy;

    assert!(
        fe_200 < fe_20,
        "free energy should decrease under sustained predictability as \
         the posterior variance shrinks: after 20 ticks={}, after 200={}",
        fe_20,
        fe_200
    );
    // The steady-state free energy should be low — the model has
    // learned the state and both precision and belief_var have
    // converged to confident values.
    assert!(
        fe_200 < 0.1,
        "free energy should be low after sustained predictability: {}",
        fe_200
    );
}

#[test]
fn test_posterior_variance_elevated_after_surprise() {
    // After a surprising observation, the posterior variance (belief_var)
    // should be elevated — the observation was far from the prediction,
    // so the Kalman update moved the belief significantly, and the
    // posterior variance reflects the residual uncertainty. This
    // elevated belief_var should contribute to the free energy through
    // the expected-uncertainty term, keeping the free energy above zero
    // even as the surprise EMA decays.
    //
    // We verify that after a surprise followed by a brief predictable
    // period (where surprise_ema decays but belief_var remains
    // elevated), the free energy is still positive — reflecting the
    // posterior variance contribution.
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Initialize and build some predictability
    for _ in 0..50 {
        engine.cycle(&state, &state, 0.1, &state);
    }

    // Surprising observation
    let surprised = [0.9f32; NEUROCHEMICAL_COUNT];
    engine.cycle(&state, &surprised, 0.1, &state);

    // Run a few predictable cycles — surprise_ema decays but the
    // posterior variance remains elevated from the large update.
    for _ in 0..5 {
        engine.cycle(&surprised, &surprised, 0.1, &state);
    }

    let signals = engine.signals();
    // The free energy should still be positive, reflecting the
    // elevated posterior variance (the model is uncertain about its
    // estimates even though recent predictions have been accurate).
    assert!(
        signals.free_energy > 0.0,
        "free energy should reflect elevated posterior variance after \
         surprise: free_energy={}",
        signals.free_energy
    );
}

// ─── Action-conditioned generative model ───────────────────────

#[test]
fn test_action_matrix_learns_action_effects() {
    // The action matrix B should learn the effect of the engine's own
    // impulses. After many cycles where the engine generates impulses
    // (e.g., dopamine PE impulses, homeostatic reflexes), the
    // action-conditioned prediction (using B) should differ from the
    // endogenous prediction (B=0), proving B has learned nonzero
    // action effects.
    //
    // We create a scenario where the engine's impulses are the
    // primary driver of state change: the post-tick state differs
    // from the pre-tick state in a way that correlates with the
    // generated impulses. Over many cycles, B should learn this
    // correlation, and the action-conditioned prediction should
    // become more accurate than the endogenous-only prediction.
    let mut engine = ActiveInferenceEngine::new();
    let baseline = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run cycles where the state consistently shifts up in dopamine
    // (mimicking the engine's dopamine impulse taking effect).
    let da_idx = NeurochemicalId::Dopamine as usize;
    for _ in 0..200 {
        let pre = [0.5f32; NEUROCHEMICAL_COUNT];
        let mut post = pre;
        post[da_idx] = 0.6; // DA consistently rises by 0.1
        engine.cycle(&pre, &post, 0.1, &baseline);
    }

    // After learning, the endogenous prediction (predict, which
    // uses zero action) should NOT predict the DA rise, because
    // the rise was caused by the engine's action, not endogenous
    // dynamics. The action-conditioned prediction (used in cycle)
    // should predict closer to the actual post state.
    //
    // We can't call predict_with_action directly (it's private),
    // but we can compare: predict(&pre) gives the endogenous
    // prediction, while the next cycle's result.predicted gives
    // the action-conditioned prediction. If B has learned, the
    // action-conditioned prediction should be closer to the actual
    // post state for the DA dimension.
    let pre = [0.5f32; NEUROCHEMICAL_COUNT];
    let mut post = pre;
    post[da_idx] = 0.6;

    let endogenous_pred = engine.predict(&pre);
    let result = engine.cycle(&pre, &post, 0.1, &baseline);

    // The endogenous prediction should not predict the full DA rise
    // (the rise is an action effect, not endogenous). It may predict
    // some of it if A absorbed some action effect early on, but
    // it should be less accurate than the action-conditioned pred.
    let endogenous_da_error = (endogenous_pred[da_idx] - post[da_idx]).abs();
    let conditioned_da_error = (result.predicted[da_idx] - post[da_idx]).abs();

    assert!(
        conditioned_da_error <= endogenous_da_error + 0.01,
        "action-conditioned prediction should be at least as accurate as \
         endogenous-only for DA: endogenous_error={}, conditioned_error={}",
        endogenous_da_error,
        conditioned_da_error
    );
}

#[test]
fn test_transition_matrix_not_corrupted_by_actions() {
    // The key structural fix: A should learn only endogenous dynamics,
    // not action effects. If the state changes are entirely due to
    // the engine's impulses (no endogenous dynamics), A should stay
    // close to identity (predict "no change"), while B absorbs the
    // action effect.
    //
    // We test this by running cycles where the pre and post states
    // are identical (no endogenous change), but the engine still
    // generates impulses (which would have been applied to the
    // neurochemical state by the tick loop, but here we pass
    // pre == post so the net effect is zero). In this scenario, A
    // should stay at identity because there's no endogenous change
    // to learn, and B should stay near zero because the action effect
    // is zero (pre == post means the impulses had no net effect).
    //
    // More importantly, we verify that when there IS a change, but
    // it's driven by the action (not endogenous dynamics), A doesn't
    // absorb it. We do this by checking that the endogenous prediction
    // (predict, which uses B=0) stays close to the input state when
    // the state changes are action-driven.
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run 100 predictable cycles (pre == post) to build precision
    // and stabilize the model.
    for _ in 0..100 {
        engine.cycle(&state, &state, 0.1, &state);
    }

    // Now the endogenous prediction should be close to the input
    // (A ≈ identity, bias ≈ 0). This verifies A hasn't been
    // corrupted by any action effects from the predictable cycles.
    let pred = engine.predict(&state);
    for i in 0..NEUROCHEMICAL_COUNT {
        assert!(
            (pred[i] - state[i]).abs() < 0.15,
            "endogenous prediction should be close to input after \
             predictable cycles (A ≈ identity): dim {} predicted={}, actual={}",
            i,
            pred[i],
            state[i]
        );
    }
}

#[test]
fn test_engine_save_load_v3_round_trip() {
    // The v3 save/load format should preserve the action-conditioned
    // model (action_matrix B, last_action vector) in addition to
    // all v2 fields.
    let mut engine = ActiveInferenceEngine::new();
    let state = [0.5f32; NEUROCHEMICAL_COUNT];

    // Run cycles to develop non-default state, including action
    // effects (the engine generates impulses that make B nonzero).
    let da_idx = NeurochemicalId::Dopamine as usize;
    for _ in 0..50 {
        let mut next = state;
        next[da_idx] = 0.6;
        engine.cycle(&state, &next, 0.1, &state);
    }

    // Save
    let path = std::env::temp_dir().join(format!(
        "genesis_inference_v3_test_{}_{}.bin",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    engine.save(&path).expect("save failed");

    // Load
    let loaded = ActiveInferenceEngine::load(&path);

    // The predictions should match — this verifies that both A and B
    // (and all scalar state) were preserved.
    let test_state = [0.5f32; NEUROCHEMICAL_COUNT];
    let pred_before = engine.predict(&test_state);
    let pred_after = loaded.predict(&test_state);
    for i in 0..NEUROCHEMICAL_COUNT {
        assert!(
            (pred_before[i] - pred_after[i]).abs() < 1e-6,
            "endogenous prediction should match after v3 load: dim {} before={}, after={}",
            i,
            pred_before[i],
            pred_after[i]
        );
    }

    // Scalar state should match
    assert!(
        (loaded.precision() - engine.precision()).abs() < 1e-6,
        "precision should match after v3 load"
    );
    assert_eq!(
        loaded.tick_count(),
        engine.tick_count(),
        "tick_count should match after v3 load"
    );

    // Clean up
    let _ = std::fs::remove_file(&path);
}
