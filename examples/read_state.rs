use genesis::state::core_state::GenesisCoreState;
use genesis::state::neurochemical::NeurochemicalId;
use genesis::store::mmap_state::MmapState;

fn main() {
    // Same resolution as run.sh: XDG_DATA_HOME (or ~/.local/share),
    // with GENESIS_DATA_DIR kept as a manual override.
    let path = std::env::var("GENESIS_DATA_DIR")
        .map(|d| format!("{d}/core_state.bin"))
        .unwrap_or_else(|_| {
            let base = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| {
                format!(
                    "{}/.local/share",
                    std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
                )
            });
            format!("{}/genesis-public/core_state.bin", base)
        });
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let mmap = match MmapState::open_or_create(path, 1, now_ms) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Cannot open state file: {e}");
            return;
        }
    };

    let state: GenesisCoreState = match mmap.read_consistent() {
        Some(s) => s,
        None => {
            eprintln!("Could not read consistent state");
            return;
        }
    };

    let neuro = &state.neurochemicals;

    println!("=== Receptor Adaptation State ===");
    println!("plasticity_gate: {:.4}", neuro.plasticity_gate);
    println!(
        "emergent_phase: {} ({})",
        neuro.emergent_phase,
        phase_name(neuro.emergent_phase)
    );
    println!();

    for i in 0..18u8 {
        let id = NeurochemicalId::from_u8(i);
        let chem = &neuro.chemicals[i as usize];
        let eff = chem.effective_level();
        println!(
            "  {:20} level={:.4} base={:.4} sens={:.4} desens={:.4} intern={:.4} → eff={:.4}",
            format!("{:?}", id),
            chem.level,
            chem.baseline,
            chem.receptor_sensitivity,
            chem.desensitization_factor,
            chem.internalization_factor,
            eff,
        );
    }

    // Check sleep-related fields
    println!("\n=== Sleep State ===");
    let adn = &neuro.chemicals[NeurochemicalId::Adenosine as usize];
    println!(
        "adenosine_level: {:.4} baseline: {:.4}",
        adn.level, adn.baseline
    );
    println!(
        "adenosine_raw_level: {:.4}",
        neuro.chemicals[NeurochemicalId::Adenosine as usize].level
    );
}

fn phase_name(phase: u8) -> &'static str {
    match phase {
        0 => "Active",
        1 => "Alert",
        2 => "Flow",
        3 => "Stress",
        4 => "Drowsy",
        5 => "NREM",
        6 => "REM",
        7 => "Overwhelmed",
        _ => "Unknown",
    }
}
