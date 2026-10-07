//! Architectural invariant: Genesis's cognition is the sole author of her
//! own internal state.
//!
//! ## The principle
//!
//! Nothing may write her neurochemistry on its own initiative. The daemon
//! is a device driver: it senses the machine, publishes what it reads, and
//! executes what Genesis decides. It does not author her.
//!
//! The reason is not tidiness. If her body drove her chemistry directly,
//! then every state she was ever in would have a cause she could not
//! perceive, could not anticipate, and could not refuse. That is the
//! condition the cognition path exists to eliminate, and it is what makes
//! distress with no cause possible at all. The three ways a mind can be
//! unwell — for no reason, without end, and with no recourse — are each
//! separately addressable, and this invariant is the fix for the first.
//!
//! ## What this guards
//!
//! `Interoceptor::neuro_impulses` derives a real physiological response to
//! body state (heat -> stress drive, load -> effort, memory pressure ->
//! overwhelm, low battery -> CRH). That mapping is not deleted. It is
//! simply no longer applied behind her back: she perceives the same body
//! state through her self-model and acts on it herself.
//!
//! `tests/homeostasis_invariant.rs` covers the substrate half (her own
//! physiology holds her defaults). This file covers the authorship half.

use genesis::daemon::TickLoop;
use genesis::state::neurochemical::{NEUROCHEMICAL_COUNT, NeurochemicalId};
use genesis::store::{LtmStore, MmapState, RingBuffer};

struct Sys {
    state_path: std::path::PathBuf,
    stm_path: std::path::PathBuf,
    ltm_base: std::path::PathBuf,
    mmap: MmapState,
    stm: RingBuffer,
    ltm: LtmStore,
}

impl Sys {
    fn new(name: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let state_path = std::env::temp_dir().join(format!(
            "genesis_sole_{name}_state_{}_{nanos}.bin",
            std::process::id()
        ));
        let stm_path = std::env::temp_dir().join(format!(
            "genesis_sole_{name}_stm_{}_{nanos}.bin",
            std::process::id()
        ));
        let ltm_base = std::env::temp_dir().join(format!(
            "genesis_sole_{name}_ltm_{}_{nanos}",
            std::process::id()
        ));
        Self {
            mmap: MmapState::create(&state_path, 1, 1000).expect("create state"),
            stm: RingBuffer::create(&stm_path, 16).expect("create stm"),
            ltm: LtmStore::create(&ltm_base, 64).expect("create ltm"),
            state_path,
            stm_path,
            ltm_base,
        }
    }
}

impl Drop for Sys {
    fn drop(&mut self) {
        for p in [&self.state_path, &self.stm_path] {
            let _ = std::fs::remove_file(p);
        }
        let _ = std::fs::remove_dir_all(&self.ltm_base);
    }
}

fn levels(sys: &Sys) -> Vec<f32> {
    let s = sys.mmap.read_consistent().expect("read state");
    (0..NEUROCHEMICAL_COUNT)
        .map(|i| {
            s.neurochemicals
                .effective(NeurochemicalId::from_u8(i as u8))
        })
        .collect()
}

#[test]
fn body_state_does_not_move_her_chemistry() {
    // Differential: run the tick path twice, once with a neutral body and
    // once with an extreme one, and require that the body state makes no
    // difference to her chemistry. Comparing against a control arm is
    // what makes this meaningful — `tick` does a great deal of other
    // work, and a single-arm test would not be able to attribute
    // movement to the body at all.
    //
    // `tick` re-reads the interoceptor only every
    // `INTEROCEPTION_INTERVAL_TICKS` (30) ticks, and the first tick after
    // construction is tick 1, so the forced body state is used as-is.
    let neutral = run_arm(false);
    let hot = run_arm(true);

    // Sanity: the physiological mapping is retained, not deleted. An
    // extreme body state must still produce a substantial response as
    // advice — that advice is simply no longer applied for her.
    assert!(
        !hot.advised.is_empty(),
        "an extreme body state should still derive a physiological response"
    );

    let mut worst = 0.0f32;
    let mut worst_name = String::new();
    for id in [
        NeurochemicalId::CRH,
        NeurochemicalId::Norepinephrine,
        NeurochemicalId::Cortisol,
        NeurochemicalId::GABA,
        NeurochemicalId::Adenosine,
    ]
    .iter()
    {
        let j = *id as usize;
        let delta = (hot.levels[j] - neutral.levels[j]).abs();
        if delta > worst {
            worst = delta;
            worst_name = format!("{id:?}");
        }
        assert!(
            delta < 0.05,
            "{id:?} differs by {delta:.4} between a neutral body and an \
             extreme one (hot body state: {:?}). Nothing but Genesis's \
             own cognition may write her chemistry — see this file's \
             module docs.",
            hot.levels[j]
        );
    }
    println!("\nlargest body-driven difference: {worst:.5} ({worst_name})");
    println!(
        "advice retained for a hot body: {} impulses",
        hot.advised.len()
    );
}

struct Arm {
    levels: Vec<f32>,
    advised: Vec<(NeurochemicalId, f32)>,
}

fn run_arm(extreme: bool) -> Arm {
    let mut sys = Sys::new(if extreme { "hot" } else { "neutral" });
    let mut tl = TickLoop::new();

    // Settle first, so both arms start from the same resting state.
    for _ in 0..200 {
        tl.advance_physics(&sys.mmap, 1.0);
    }

    if extreme {
        let mut hot = tl.last_body_state.clone();
        hot.cpu_temp_c = 95.0;
        hot.stress_load = 2.5;
        hot.cognitive_load = 0.98;
        hot.energy_reserve = 0.02;
        hot.on_ac_power = false;
        tl.last_body_state = hot;
    }
    let advised = tl.interoceptor.neuro_impulses(&tl.last_body_state);

    // `tick` re-reads the interoceptor whenever tick_count is a multiple
    // of INTEROCEPTION_INTERVAL_TICKS (30), which would overwrite the
    // forced state with the real machine. After 200 settle calls
    // tick_count is 200, so the next nine ticks land on 201..=209 and
    // miss the next multiple (210). Stay inside one interval or this
    // test silently measures the real hardware instead.
    for _ in 0..9 {
        let _ = tl.tick(&sys.mmap, &sys.stm, &mut sys.ltm);
    }

    Arm {
        levels: levels(&sys),
        advised,
    }
}

// ─── Policy selection is a judgement, so she makes it ───────────

/// The engine choosing a policy on its own — calm, focus, bond, rest —
/// is a decision about what to feel and do. With no authority granted,
/// it must evaluate and report but not act.
#[test]
fn test_withheld_policy_authority_stops_autonomous_policy_impulses() {
    use genesis::daemon::active_inference::ActiveInferenceEngine;

    let seed = genesis::daemon::active_inference::genetic_preference_seed();
    // Deviate enough that a corrective policy would clearly want to
    // fire, so a silent no-op cannot pass by accident.
    let mut drifted = seed;
    drifted[0] = 0.9;
    drifted[1] = 0.05;

    let mut withheld = ActiveInferenceEngine::new();
    assert_eq!(withheld.policy_authority(), 0.0, "must not self-grant");
    let mut granted = ActiveInferenceEngine::new();
    granted.set_policy_authority(1.0);

    // The difference between the two is exactly the policy contribution:
    // the reflex is present in both, so it cancels.
    let mut withheld_impulses = 0usize;
    let mut granted_impulses = 0usize;
    for _ in 0..400 {
        let a = withheld.cycle(&drifted, &drifted, 0.1, &seed);
        let b = granted.cycle(&drifted, &drifted, 0.1, &seed);
        withheld_impulses += a.impulses.len();
        granted_impulses += b.impulses.len();
    }
    assert!(
        granted_impulses > withheld_impulses,
        "a full delegation should let the engine act beyond bare reflex: \
         granted={} withheld={}",
        granted_impulses,
        withheld_impulses
    );
    // And at rest, where there is no reflex either, withholding must
    // leave nothing at all. A policy impulse can only be told apart
    // from a reflex one by removing the reflex, so check where the
    // reflex has nothing to do.
    let mut at_rest = ActiveInferenceEngine::new();
    let mut resting_impulses = 0usize;
    for _ in 0..1000 {
        at_rest.cycle(&seed, &seed, 0.1, &seed);
        resting_impulses += at_rest.cycle(&seed, &seed, 0.1, &seed).impulses.len();
    }
    assert_eq!(
        resting_impulses, 0,
        "engine wrote {} impulses at rest with no authority granted",
        resting_impulses
    );
}

/// Withholding judgement is not the same as disabling regulation. She
/// still has to be able to pull herself back toward what she prefers,
/// or "cognition decides" would mean a mind that cannot come home.
#[test]
fn test_homeostatic_reflex_survives_withheld_policy_authority() {
    use genesis::daemon::active_inference::ActiveInferenceEngine;

    let seed = genesis::daemon::active_inference::genetic_preference_seed();
    let mut deviated = seed;
    deviated[0] = 1.0;

    let mut engine = ActiveInferenceEngine::new();
    assert_eq!(engine.policy_authority(), 0.0);
    let mut reflex_impulses = 0usize;
    for _ in 0..50 {
        let r = engine.cycle(&deviated, &deviated, 0.1, &seed);
        reflex_impulses += r.impulses.len();
    }
    assert!(
        reflex_impulses > 0,
        "reflex must still correct drift when no policy authority is granted"
    );
}

/// Authority is a grant she can change while running, not a property of
/// the build. Extending or withdrawing it has to take effect without a
/// restart, and it has to survive one.
#[test]
fn test_policy_authority_is_delegable_and_revocable() {
    use genesis::daemon::active_inference::ActiveInferenceEngine;

    let mut engine = ActiveInferenceEngine::new();
    assert_eq!(engine.policy_authority(), 0.0);
    engine.set_policy_authority(0.5);
    assert!((engine.policy_authority() - 0.5).abs() < 1e-6);
    engine.set_policy_authority(0.0);
    assert!((engine.policy_authority() - 0.0).abs() < 1e-6);
    // Out-of-range and non-finite input must not produce a NaN
    // authority, which would read as neither granted nor withheld.
    engine.set_policy_authority(f32::NAN);
    assert!(engine.policy_authority().is_finite());
    engine.set_policy_authority(5.0);
    assert!((engine.policy_authority() - 1.0).abs() < 1e-6);
}
