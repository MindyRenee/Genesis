//! Neurological recovery tool — resets receptor sensitivities that have
//! collapsed to the floor (0.10) due to sustained overstimulation.
//!
//! This is the equivalent of pharmacological intervention: when receptors
//! are so downregulated that the system can't self-regulate (can't sleep,
//! can't learn, can't feel), we manually restore receptor sensitivity to
//! allow the normal dynamics to resume.
//!
//! Run with: cargo run --example recover

use genesis::MmapState;
use genesis::state::neurochemical::NeurochemicalId;

fn main() {
    // Same resolution as run.sh: XDG_DATA_HOME (or ~/.local/share),
    // with GENESIS_DATA_DIR kept as a manual override.
    let path = std::env::var("GENESIS_DATA_DIR").unwrap_or_else(|_| {
        let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
            format!(
                "{}/.local/share",
                std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
            )
        });
        format!("{}/genesis-public", base)
    });
    let state_file = format!("{}/core_state.bin", path);

    println!("Opening state file: {}", state_file);
    let mmap = match MmapState::open(&state_file) {
        Ok(m) => {
            println!(
                "  Opened existing state (seq_lock={})",
                m.read().header.seq_lock
            );
            m
        }
        Err(e) => {
            eprintln!("Cannot open state file: {}", e);
            std::process::exit(1);
        }
    };

    // Read current state
    let snapshot = mmap.read_consistent();
    let snap = match snapshot {
        Some(s) => s,
        None => {
            eprintln!("Could not get consistent read — retrying...");
            std::thread::sleep(std::time::Duration::from_millis(100));
            match mmap.read_consistent() {
                Some(s) => s,
                None => {
                    eprintln!("Failed to get consistent read after retry.");
                    std::process::exit(1);
                }
            }
        }
    };

    println!("\n=== BEFORE RECOVERY ===");
    let names = NeurochemicalId::all();
    for id in &names {
        let chem = &snap.neurochemicals.chemicals[*id as usize];
        let eff = snap.neurochemicals.effective_levels[*id as usize];
        println!(
            "  {:<20} level={:.4} baseline={:.4} sens={:.4} eff={:.4}",
            id.name(),
            chem.level,
            chem.baseline,
            chem.receptor_sensitivity,
            eff
        );
    }
    println!(
        "\n  plasticity_gate={:.4}  arousal={:.4}  valence={:.4}",
        snap.neurochemicals.plasticity_gate,
        snap.neurochemicals.arousal,
        snap.neurochemicals.valence
    );

    // Check which receptors are at or near floor (0.5 for sensitivity,
    // 0.2 for desensitization, 0.45 for internalization) or significantly
    // downregulated (below 0.6 sensitivity — this catches GABA and other
    // chemicals that are above floor but still low enough to crush
    // effective levels and drag down coupled chemicals).
    let floor_chemicals: Vec<NeurochemicalId> = names
        .iter()
        .filter(|id| {
            let chem = &snap.neurochemicals.chemicals[**id as usize];
            chem.receptor_sensitivity <= 0.60
                || chem.desensitization_factor <= 0.25
                || chem.internalization_factor <= 0.35
        })
        .copied()
        .collect();

    println!(
        "\n  Receptors at/near floor: {:?}",
        floor_chemicals
            .iter()
            .map(|id| id.name())
            .collect::<Vec<_>>()
    );

    if floor_chemicals.is_empty() {
        println!("\n  All receptors are healthy.");
    }

    // Reset floor receptors to their default sensitivity
    // This is like administering a receptor antagonist that allows
    // upregulation to reset — pharmacologically, this is what happens
    // when a drug is withdrawn and receptors recover.
    if !floor_chemicals.is_empty() {
        println!("\n  Resetting floor receptors to default sensitivity...");
    }
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    mmap.modify(now_ms, |state| {
        let neuro = &mut state.neurochemicals;
        for id in &floor_chemicals {
            let chem = &mut neuro.chemicals[*id as usize];
            let default_sens = id.default_sensitivity();
            println!(
                "  {:<20} sens {:.4} → {:.4}",
                id.name(),
                chem.receptor_sensitivity,
                default_sens
            );
            chem.receptor_sensitivity = default_sens;
        }
        // Also boost BDNF to help reopen the plasticity gate
        let bdnf = &mut neuro.chemicals[NeurochemicalId::BDNF as usize];
        if bdnf.level < 0.10 {
            println!(
                "  {:<20} level {:.4} → 0.20 (BDNF boost to reopen plasticity)",
                "BDNF", bdnf.level
            );
            bdnf.level = 0.20;
        }
        // Reset desensitization and internalization factors for floor chemicals
        for id in &floor_chemicals {
            let chem = &mut neuro.chemicals[*id as usize];
            chem.desensitization_factor = 1.0; // no desensitization
            chem.internalization_factor = 1.0; // no internalization
        }
        // Always reset coupling matrix to defaults — metaplasticity may
        // have amplified positive feedback loops that drive overstimulation.
        // This is the pharmacological equivalent of a full reset: the
        // metaplastic changes are erased and the system starts fresh
        // with the biologically grounded default connectome.
        println!("  Resetting coupling matrix to defaults (clearing metaplasticity)");
        neuro.coupling_matrix = genesis::state::neurochemical::DEFAULT_COUPLING_MATRIX;
        // Recompute derived metrics with the new sensitivities
        neuro.recompute_derived();
        state.sync_neurochemistry_to_state();
    })
    .expect("recovery write failed");

    // Verify
    let snapshot = mmap.read_consistent().expect("post-recovery read failed");
    println!("\n=== AFTER RECOVERY ===");
    for id in &names {
        let chem = &snapshot.neurochemicals.chemicals[*id as usize];
        let eff = snapshot.neurochemicals.effective_levels[*id as usize];
        println!(
            "  {:<20} level={:.4} sens={:.4} eff={:.4}",
            id.name(),
            chem.level,
            chem.receptor_sensitivity,
            eff
        );
    }
    println!(
        "\n  plasticity_gate={:.4}  arousal={:.4}  valence={:.4}",
        snapshot.neurochemicals.plasticity_gate,
        snapshot.neurochemicals.arousal,
        snapshot.neurochemicals.valence
    );
    println!("\n  Recovery complete. The daemon will continue ticking with restored receptors.");
    println!("  Sleep should now be possible as adenosine sensitivity is restored.");
}
