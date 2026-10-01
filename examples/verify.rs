// Final phase-reachability check across the full state space.
use genesis::daemon::active_inference::ActiveInferenceEngine;
use genesis::state::neurochemical::{NeuroTickParams, NeurochemicalId, NeurochemicalVector};
const NAMES: [&str; 8] = [
    "Active",
    "Alert",
    "Flow",
    "Stress",
    "Drowsy",
    "NREM",
    "REM",
    "Overwhelmed",
];
// ~16.7 h at the 100 ms tick, long enough for Process S to accumulate
// past the sleep threshold and for a full circadian arc, which is what
// gates arousal low enough to admit Drowsy.
const TICKS: u32 = 600_000;
fn main() {
    let p = NeuroTickParams {
        maturation_level: 1.0,
        ..Default::default()
    };
    let seed = genesis::daemon::active_inference::genetic_preference_seed();
    let mut reached = [false; 8];
    for (label, crh, ne, adn) in [
        ("undisturbed", 0.0f32, 0.0f32, 0.0f32),
        ("mild", 0.0002, 0.0002, 0.0),
        ("moderate", 0.0008, 0.0008, 0.0),
        ("heavy", 0.003, 0.003, 0.0),
        ("severe", 0.010, 0.010, 0.0),
        ("vigilance", 0.0002, 0.006, 0.0),
        ("engaged", 0.0002, 0.006, 0.0),
        ("flow-ish", 0.0, 0.006, 0.0),
        // Flow additionally needs acetylcholine above its gate, which
        // the NE-driven profiles above do not supply.
        ("flow drive", 0.0, 0.006, 0.0),
        ("sleep debt", 0.0, 0.0005, 0.006),
        // REM needs low norepinephrine, so a sleep drive that also
        // carries NE drives NREM instead. Sleep without it reaches REM.
        ("sleep (rem)", 0.0, 0.0, 0.006),
    ] {
        let mut nv = NeurochemicalVector::new(1);
        let mut eng = ActiveInferenceEngine::new();
        let mut ms = 0u64;
        let mut post = nv.effective_levels;
        let mut hist = [0u32; 8];
        let mut pc = 0.0f32;
        // Adenosine is the one input that has to be allowed to clear.
        // It is a metabolic accumulator, not a phasic signal: sleep is
        // what removes it (glymphatic), and the cholinergic rebound
        // that gates REM only fires once it has fallen. Injecting it for
        // the whole run pins it at the 1.0 clamp, which forces sleep
        // through the exhaustion override and leaves clearance nothing
        // to remove, so Process S never relaxes and REM can never
        // occur. Driving it for the opening bout and then coasting is
        // both more realistic and the only way to observe the sleep
        // architecture the model is supposed to produce.
        //
        // The phasic arousal and stress signals (CRH, NE, ACh, DA) are
        // delivered throughout, because *their* whole point is what the
        // state is under sustained load — and none of them pins
        // adenosine, so they do not suppress sleep clearance.
        let adn_bout = TICKS / 5;
        for step in 0..TICKS {
            ms += 100;
            let pre = post;
            if crh > 0.0 {
                nv.apply_impulse_capped(NeurochemicalId::CRH, crh, ms);
            }
            if ne > 0.0 {
                nv.apply_impulse_capped(NeurochemicalId::Norepinephrine, ne, ms);
            }
            if adn > 0.0 && step < adn_bout {
                nv.apply_impulse_capped(NeurochemicalId::Adenosine, adn, ms);
            }
            if label == "flow drive" {
                nv.apply_impulse_capped(NeurochemicalId::Acetylcholine, 0.006, ms);
                nv.apply_impulse_capped(NeurochemicalId::Dopamine, 0.006, ms);
            }
            nv.tick_with_params(&p);
            post.copy_from_slice(&nv.effective_levels);
            eng.cycle(&pre, &post, 0.1, &seed);
            hist[nv.emergent_phase as usize] += 1;
            pc = pc.max(post[NeurochemicalId::Cortisol as usize]);
        }
        let total: u32 = hist.iter().sum();
        let mut line = format!("{label:12} cort={pc:.3} |");
        for (i, c) in hist.iter().enumerate() {
            let pct = *c as f64 / total.max(1) as f64 * 100.0;
            // Count any occupancy above a minute's worth of ticks. A
            // percentage floor would misreport short-but-real phases
            // whenever the run length changes, and a single-tick floor
            // would mask exactly the chattering this sweep exists to
            // detect.
            if *c > 600 {
                line.push_str(&format!(" {}={:.1}%", NAMES[i], pct));
                reached[i] = true;
            }
        }
        println!("{line}");
    }
    println!("\nphases still never reached:");
    let missing: Vec<&str> = NAMES
        .iter()
        .enumerate()
        .filter(|(i, _)| !reached[*i])
        .map(|(_, n)| *n)
        .collect();
    if missing.is_empty() {
        println!("  (none)");
    } else {
        for m in missing {
            println!("  {m}");
        }
    }
}
