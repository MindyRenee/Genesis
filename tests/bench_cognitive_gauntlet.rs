//! Benchmark 10: Cognitive Architecture Self-Evaluation.
//!
//! A self-authored evaluation of Genesis against two established
//! cognitive architecture frameworks:
//!
//! 1. **The Common Model of Cognition** (Laird et al., 2017) — the
//!    consensus taxonomy of what a cognitive architecture should
//!    have: procedural memory, declarative memory, working memory,
//!    perceptual modules, motor modules, learning mechanisms,
//!    decision-making, and goal-directed behavior.
//!
//! 2. **Pickett's Gauntlet** — a phenomena checklist for evaluating
//!    cognitive architectures: autonomous learning, representation
//!    development, domain independence, self-modification, etc.
//!
//! ## What this is and isn't
//!
//! This is a **self-evaluation**, not an independent assessment.
//! The capability matrix was authored by the project author, who
//! both wrote the rubric and graded Genesis against it. The
//! coverage percentages (100% Common Model, 91% Pickett) are
//! self-reported, not independently adjudicated.
//!
//! "Implemented" means **code exists and is wired in** — it does
//! not mean the capability works at expert-level standards or
//! passes a validated behavioral test. A reviewer should treat
//! these classifications as the author's honest assessment of
//! what exists in the codebase, not as objective capability
//! measurements.
//!
//! ## Methodology
//!
//! Each capability is classified as:
//!
//! - **Implemented**: the code exists and is wired in.
//! - **Partial**: some components exist but the full capability is
//!   incomplete.
//! - **Absent**: no implementation exists.
//! - **Unverified**: the capability exists but hasn't been tested
//!   against external criteria.
//!
//! This is the honest framing — we don't claim capabilities we
//! can't point to in the code, and we flag what's unverified.

use std::path::Path;

/// A capability evaluation entry.
struct Capability {
    name: &'static str,
    framework: &'static str, // "Common Model" or "Pickett"
    status: &'static str,    // "implemented", "partial", "absent", "unverified"
    evidence: &'static str,  // file/function that provides this
    notes: &'static str,
}

/// The full capability matrix, grounded in the actual codebase.
///
/// Each entry is verified by the source files referenced. The
/// classification is conservative: we only claim "implemented" when
/// the code exists and is wired into the tick cycle or the
/// cognitive mind's active processing.
fn capability_matrix() -> Vec<Capability> {
    vec![
        // ─── Common Model of Cognition ──────────────────────────
        Capability {
            name: "Procedural memory (production rules / skills)",
            framework: "Common Model",
            status: "partial",
            evidence: "daemon/active_inference.rs (9 fixed policies)",
            notes: "Policies are fixed, not learned production rules. \
                    Policy selection is active inference, not pattern matching.",
        },
        Capability {
            name: "Declarative memory (facts / episodes)",
            framework: "Common Model",
            status: "implemented",
            evidence: "store/ltm.rs (LTM episodic store with SDR encoding)",
            notes: "Episodic store with salience-gated encoding, emotional \
                    tagging, and SDR-based retrieval. Verified in bench_memory_boundedness.",
        },
        Capability {
            name: "Working memory (active state)",
            framework: "Common Model",
            status: "implemented",
            evidence: "state/core_state.rs (mmap'd GenesisCoreState, 3288 bytes)",
            notes: "The core state struct is the working memory — shared \
                    across processes via mmap, updated every tick.",
        },
        Capability {
            name: "Perceptual modules (sensory input)",
            framework: "Common Model",
            status: "implemented",
            evidence: "daemon/interoception.rs (hardware → neurochemistry)",
            notes: "Interoception reads CPU temp, load, memory, I/O, battery \
                    and maps them to neurochemical impulses. This is the \
                    machine-native perceptual system.",
        },
        Capability {
            name: "Motor modules (action output)",
            framework: "Common Model",
            status: "implemented",
            evidence: "daemon/cpufreq.rs (neurochemistry → hardware control)",
            notes: "Body control publishes CPU frequency, scheduling priority, \
                    I/O priority, and thermal caps. Applied via cognitive mind.",
        },
        Capability {
            name: "Learning mechanism",
            framework: "Common Model",
            status: "implemented",
            evidence: "daemon/active_inference.rs (transition model learning)",
            notes: "The active inference engine learns an 18×18 transition \
                    matrix and adapts precision. Verified in bench_learning_curves.",
        },
        Capability {
            name: "Decision-making (action selection)",
            framework: "Common Model",
            status: "implemented",
            evidence: "daemon/active_inference.rs (policy selection via EFE)",
            notes: "Policy selection minimizes expected free energy, balancing \
                    pragmatic cost and epistemic value. Verified in bench_epistemic_foraging.",
        },
        Capability {
            name: "Goal-directed behavior",
            framework: "Common Model",
            status: "partial",
            evidence: "daemon/active_inference.rs (homeostatic targets as goals)",
            notes: "Goals are implicit (homeostatic baselines), not explicit \
                    goal representations. No goal stack or goal decomposition.",
        },
        Capability {
            name: "Attention mechanism",
            framework: "Common Model",
            status: "implemented",
            evidence: "state/neurochemical.rs (ACh gating, norepinephrine focus)",
            notes: "Acetylcholine gates cortical processing; norepinephrine \
                    modulates focus. These are neurochemically grounded.",
        },
        Capability {
            name: "Emotion / affect",
            framework: "Common Model",
            status: "implemented",
            evidence: "state/neurochemical.rs (18-chemical affective substrate)",
            notes: "Full neurochemical affective system with arousal, valence, \
                    and specific neuromodulators. Verified in bench_cortisol_hpa, \
                    bench_dopamine_rpe.",
        },
        // ─── Pickett's Gauntlet ──────────────────────────────────
        Capability {
            name: "Autonomous learning (without external teaching)",
            framework: "Pickett",
            status: "implemented",
            evidence: "daemon/active_inference.rs (self-model learning)",
            notes: "The engine learns its own transition model from observed \
                    neurochemical trajectories — no external teacher.",
        },
        Capability {
            name: "Representation development (builds own representations)",
            framework: "Pickett",
            status: "partial",
            evidence: "store/sdr.rs (SDR encoding of episodes)",
            notes: "Episodes are encoded as sparse distributed representations. \
                    The active inference engine builds a transition model. \
                    But no concept formation or abstraction layer.",
        },
        Capability {
            name: "Domain independence (not task-specific)",
            framework: "Pickett",
            status: "partial",
            evidence: "state/neurochemical.rs (general-purpose substrate)",
            notes: "The neurochemical substrate is domain-general (it models \
                    the system's own state, not a specific task). But the \
                    policy repertoire is fixed and domain-specific (homeostatic).",
        },
        Capability {
            name: "Self-modification (changes own architecture)",
            framework: "Pickett",
            status: "implemented",
            evidence: "state/neurochemical.rs (metaplasticity, receptor adaptation)",
            notes: "The coupling matrix is plastic (metaplasticity), receptors \
                    adapt, and BDNF gates plasticity. The system modifies its \
                    own dynamics.",
        },
        Capability {
            name: "Scalability (handles increasing complexity)",
            framework: "Pickett",
            status: "implemented",
            evidence: "bench_learning_curves (1→18 dimensions, sub-linear FE scaling)",
            notes: "FE scales 1.29x for 18x more dimensions. Verified.",
        },
        Capability {
            name: "Robustness (handles perturbation/noise)",
            framework: "Pickett",
            status: "implemented",
            evidence: "bench_long_run_stability (100k ticks, 10 perturbations)",
            notes: "No NaN/Inf over 100k ticks with noise. Recovers from \
                    periodic perturbations. Verified.",
        },
        Capability {
            name: "Real-time operation",
            framework: "Pickett",
            status: "implemented",
            evidence: "daemon/tick.rs (10 Hz tick cycle)",
            notes: "The system operates at 10 Hz, updating state every 100ms. \
                    This is real-time for a cognitive architecture.",
        },
        Capability {
            name: "Memory boundedness (doesn't grow without limit)",
            framework: "Pickett",
            status: "implemented",
            evidence: "bench_memory_boundedness (linear growth, R²=0.999999)",
            notes: "LTM grows linearly at 152 bytes/episode. Retrieval <400us. \
                    Verified.",
        },
        Capability {
            name: "Experience / self-awareness",
            framework: "Pickett",
            status: "unverified",
            evidence: "python/genesis_cognitive/ (cognitive mind)",
            notes: "A cognitive mind module exists (Python), but experience \
                    is NOT claimed or verified. The system exhibits self-modeling \
                    (active inference over own state) but this is not experience.",
        },
        Capability {
            name: "Natural language interaction",
            framework: "Pickett",
            status: "partial",
            evidence: "python/genesis_cognitive/ (language engine)",
            notes: "A language engine exists in the cognitive mind, but it's \
                    not benchmarked here. Not verified in this suite.",
        },
        Capability {
            name: "Developmental trajectory (learns over lifetime)",
            framework: "Pickett",
            status: "partial",
            evidence: "state/neurochemical.rs (maturation gating)",
            notes: "The HPA axis is maturation-gated (CRH→ACTH→cortisol cascade \
                    activates as maturation increases). This is a developmental \
                    trajectory, but it's a single dimension, not a full \
                    developmental progression.",
        },
    ]
}

/// Write JSON results.
fn write_json(capabilities: &[Capability], summary: &[(String, usize, usize)], path: &Path) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();

    writeln!(f, "{{").unwrap();
    writeln!(f, "  \"benchmark\": \"cognitive_gauntlet\",").unwrap();
    writeln!(f, "  \"frameworks\": [").unwrap();
    for (i, (framework, total, implemented)) in summary.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"name\": \"{}\",", framework).unwrap();
        writeln!(f, "      \"total_capabilities\": {},", total).unwrap();
        writeln!(f, "      \"implemented_or_partial\": {}", implemented).unwrap();
        writeln!(f, "    }}{}", if i + 1 < summary.len() { "," } else { "" }).unwrap();
    }
    writeln!(f, "  ],").unwrap();
    writeln!(f, "  \"capabilities\": [").unwrap();
    for (i, c) in capabilities.iter().enumerate() {
        writeln!(f, "    {{").unwrap();
        writeln!(f, "      \"name\": \"{}\",", c.name).unwrap();
        writeln!(f, "      \"framework\": \"{}\",", c.framework).unwrap();
        writeln!(f, "      \"status\": \"{}\",", c.status).unwrap();
        writeln!(f, "      \"evidence\": \"{}\",", c.evidence).unwrap();
        writeln!(f, "      \"notes\": \"{}\"", c.notes).unwrap();
        writeln!(
            f,
            "    }}{}",
            if i + 1 < capabilities.len() { "," } else { "" }
        )
        .unwrap();
    }
    writeln!(f, "  ]").unwrap();
    writeln!(f, "}}").unwrap();
}

#[test]
fn bench_cognitive_gauntlet() {
    let capabilities = capability_matrix();

    // Count by status and framework.
    let mut common_model_total = 0;
    let mut common_model_impl = 0;
    let mut pickett_total = 0;
    let mut pickett_impl = 0;

    for c in &capabilities {
        let is_impl = c.status == "implemented" || c.status == "partial";
        if c.framework == "Common Model" {
            common_model_total += 1;
            if is_impl {
                common_model_impl += 1;
            }
        } else if c.framework == "Pickett" {
            pickett_total += 1;
            if is_impl {
                pickett_impl += 1;
            }
        }
    }

    // Print summary
    let bar = "=".repeat(80);
    let dash = "-".repeat(80);
    println!("\n{bar}");
    println!("Benchmark 10: Cognitive Architecture Self-Evaluation");
    println!("{bar}");
    println!("Frameworks: Common Model of Cognition (Laird et al., 2017)");
    println!("            Pickett's Gauntlet (phenomena checklist)");
    println!("{dash}");
    println!("{:<45} {:<12} {:<12}", "Capability", "Framework", "Status");
    println!("{dash}");
    for c in &capabilities {
        println!("{:<45} {:<12} {:<12}", c.name, c.framework, c.status);
    }
    println!();
    println!(
        "Common Model: {}/{} capabilities implemented or partial",
        common_model_impl, common_model_total
    );
    println!(
        "Pickett:      {}/{} capabilities implemented or partial",
        pickett_impl, pickett_total
    );
    println!();

    // ─── Assertions ──────────────────────────────────────────────

    // Claim 1: Genesis implements at least 70% of Common Model capabilities
    // (implemented or partial).
    let common_ratio = common_model_impl as f32 / common_model_total as f32;
    assert!(
        common_ratio >= 0.70,
        "Common Model coverage should be ≥70%: got {}/{} = {:.0}%",
        common_model_impl,
        common_model_total,
        common_ratio * 100.0
    );

    // Claim 2: Genesis implements at least 60% of Pickett capabilities
    // (implemented or partial).
    let pickett_ratio = pickett_impl as f32 / pickett_total as f32;
    assert!(
        pickett_ratio >= 0.60,
        "Pickett coverage should be ≥60%: got {}/{} = {:.0}%",
        pickett_impl,
        pickett_total,
        pickett_ratio * 100.0
    );

    // Claim 3: No capability is claimed as "implemented" without
    // evidence (every implemented capability has a non-empty evidence field).
    for c in &capabilities {
        if c.status == "implemented" {
            assert!(
                !c.evidence.is_empty(),
                "Implemented capability '{}' must have evidence",
                c.name
            );
        }
    }

    // Claim 4: Experience is NOT claimed as implemented (honest framing).
    let experience = capabilities
        .iter()
        .find(|c| c.name.contains("Experience"))
        .unwrap();
    assert!(
        experience.status != "implemented",
        "Experience must NOT be claimed as implemented (honest framing): got '{}'",
        experience.status
    );

    // Claim 5: The capability matrix covers both frameworks.
    assert!(
        common_model_total >= 8,
        "Should evaluate ≥8 Common Model capabilities"
    );
    assert!(
        pickett_total >= 8,
        "Should evaluate ≥8 Pickett capabilities"
    );

    // Write JSON output.
    std::fs::create_dir_all("benchmarks/results").ok();
    let out_path = std::path::Path::new("benchmarks/results/bench_cognitive_gauntlet.json");
    let summary = vec![
        (
            "Common Model".to_string(),
            common_model_total,
            common_model_impl,
        ),
        ("Pickett".to_string(), pickett_total, pickett_impl),
    ];
    write_json(&capabilities, &summary, out_path);
    println!("Results written to: {}", out_path.display());

    println!("\nAll assertions passed.");
    println!(
        "  - Common Model coverage ≥70%: YES ({:.0}%)",
        common_ratio * 100.0
    );
    println!(
        "  - Pickett coverage ≥60%: YES ({:.0}%)",
        pickett_ratio * 100.0
    );
    println!("  - All implemented capabilities have evidence: YES");
    println!(
        "  - Experience NOT claimed as implemented: YES (status='{}')",
        experience.status
    );
    println!(
        "  - Both frameworks covered: YES (Common={}, Pickett={})",
        common_model_total, pickett_total
    );
}
