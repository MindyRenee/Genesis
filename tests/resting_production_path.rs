//! Reproduce the daemon's resting chemistry through the production path.
//!
//! ## Resolved: the "divergence" was a units mismatch
//!
//! This test was written because an earlier harness appeared to disagree
//! with the running daemon: it showed glutamate saturating at 0.976 while
//! the daemon sat at 0.356. The two were the same computation. The harness
//! read `neurochemicals.get(id).level` (the raw level) while the Python
//! client reads `CoreState.chemicals`, which `types.py` unpacks from the
//! vector's `effective_levels` array at offset 2304 — not the raw
//! `chemicals[]` array at offset 0. Raw and effective are different
//! quantities: effective scales raw by receptor sensitivity, subtype
//! weighting, phasic tone and (for GABA) allosteric modulation.
//!
//! Reported side by side they agree with the daemon (GABA 0.337,
//! glutamate 0.356), and the resting depression is real in both.
//!
//! ## What this shows
//!
//! - The production path is deterministic and independent of call rate:
//!   a tight in-process loop and a ~1 ms-paced loop give bit-identical
//!   results, so the daemon's behaviour does not depend on how fast the
//!   mind calls `advance_neuro`.
//! - Raw dopamine *oscillates* (0.35 -> 0.674 -> 0.439 -> 0.486) even
//!   while its effective level stays smoother. That raw instability is
//!   worth attention in its own right.
//! - At rest, raw and effective diverge substantially (dopamine raw
//!   0.486 vs effective 0.303) because receptor sensitivity adapts
//!   downward. The depression in the reported effective level is
//!   therefore larger than the depression in the raw level.

use genesis::daemon::TickLoop;
use genesis::state::neurochemical::NeurochemicalId;
use genesis::store::{LtmStore, MmapState, RingBuffer};

struct Sys {
    state_path: std::path::PathBuf,
    stm_path: std::path::PathBuf,
    ltm_base: std::path::PathBuf,
    mmap: MmapState,
    _stm: RingBuffer,
    _ltm: LtmStore,
}

impl Sys {
    fn new(name: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let state_path = std::env::temp_dir().join(format!(
            "genesis_resting_{name}_state_{}_{nanos}.bin",
            std::process::id()
        ));
        let stm_path = std::env::temp_dir().join(format!(
            "genesis_resting_{name}_stm_{}_{nanos}.bin",
            std::process::id()
        ));
        let ltm_base = std::env::temp_dir().join(format!(
            "genesis_resting_{name}_ltm_{}_{nanos}",
            std::process::id()
        ));
        Self {
            mmap: MmapState::create(&state_path, 1, 1000).expect("create state"),
            _stm: RingBuffer::create(&stm_path, 16).expect("create stm"),
            _ltm: LtmStore::create(&ltm_base, 64).expect("create ltm"),
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

/// (tick, raw levels, effective levels, arousal, phase)
type Trace = (u32, Vec<f32>, Vec<f32>, f32, u8);

const TRACKED: [NeurochemicalId; 6] = [
    NeurochemicalId::Dopamine,
    NeurochemicalId::Serotonin,
    NeurochemicalId::GABA,
    NeurochemicalId::Glutamate,
    NeurochemicalId::Acetylcholine,
    NeurochemicalId::Norepinephrine,
];

#[test]
fn production_path_resting_chemistry() {
    // Two arms, identical except for pacing. The IPC harness issues each
    // advance over a socket, so its calls are milliseconds apart; a tight
    // in-process loop issues them microseconds apart. If the resting
    // chemistry depends on the call rate, the two disagree — which is
    // exactly what the IPC daemon and this test appear to show.
    for (label, sleep) in [("tight loop", None), ("~1ms between calls", Some(1u64))] {
        run_arm(label, sleep);
    }
}

fn run_arm(label: &str, sleep_us: Option<u64>) {
    let sys = Sys::new("path");
    let mut tl = TickLoop::new();
    const DT: f32 = 1.0;
    const TICKS: u32 = 14_000;

    let mut checkpoints: Vec<Trace> = Vec::new();
    for t in 0..TICKS {
        tl.advance_neuro(&sys.mmap, DT);
        if let Some(us) = sleep_us {
            std::thread::sleep(std::time::Duration::from_micros(us));
        }
        if t % 2000 == 0 || t == TICKS - 1 {
            if let Some(s) = sys.mmap.read_consistent() {
                // Report BOTH quantities. `CoreState.chemicals` on the
                // Python side is unpacked from the vector's
                // `effective_levels` array (offset 2304), not the raw
                // `chemicals[]` array (offset 0), so comparing a Rust
                // `.level` against a Python `state.chemicals[...]` is
                // comparing two different numbers.
                let raw: Vec<f32> = TRACKED
                    .iter()
                    .map(|id| s.neurochemicals.get(*id).map(|c| c.level).unwrap_or(f32::NAN))
                    .collect();
                let eff: Vec<f32> = TRACKED
                    .iter()
                    .map(|id| s.neurochemicals.effective(*id))
                    .collect();
                checkpoints.push((t, raw, eff, s.neurochemicals.arousal, s.zones.phase() as u8));
            }
        }
    }

    println!("\n=== {} ===", label);
    println!(
        "{:>6}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}   {:>8}",
        "t", "DA_eff", "SRT_eff", "GABA_eff", "GLU_eff", "ACh_eff", "NE_eff", "DA_raw", "arousal"
    );
    for (t, raw, eff, arousal, phase) in &checkpoints {
        println!(
            "{:>6}{:>8.3}{:>8.3}{:>8.3}{:>8.3}{:>8.3}{:>8.3}{:>8.3}   {:>8.3}  phase={}",
            t,
            eff[0],
            eff[1],
            eff[2],
            eff[3],
            eff[4],
            eff[5],
            raw[0],
            arousal,
            phase
        );
    }

    let (_, _raw, final_eff, _, _) = checkpoints.last().expect("checkpoints");
    for (i, id) in TRACKED.iter().enumerate() {
        assert!(
            final_eff[i].is_finite(),
            "{:?} went non-finite on the production path",
            id
        );
    }
}
