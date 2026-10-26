//! Topology — the anatomical region map of the organism. The spine
//! of the layout: every body region, the body modules that implement
//! it, and the mind-side package it projects to.
//!
//! Mirrors `python/topology.py`: the mind declares the same table
//! back from its side. Neither side can see the other's half — the
//! daemon owns the body's modules, the mind owns its packages — so
//! the map is written once per side and joined over the region
//! names, which are the same identifiers on both ends.
//!
//! Anatomy: this is the midline structure — the body's read on its
//! own wiring. The edges listed are the documented projections, not
//! every anatomical connection.

/// One anatomical region of the organism.
pub struct Region {
    /// Region identifier — the join key both tables share.
    pub name: &'static str,
    /// The mind-side package this region projects to — `""` for
    /// body-only machinery with no mirror.
    pub package: &'static str,
    /// The body-side modules implementing the region.
    pub modules: &'static [&'static str],
    /// Regions this region projects onward to (`"*"` for diffuse
    /// projections, as with the neuromodulator broadcast).
    pub projections: &'static [&'static str],
    /// The major-system fiber this region rides — one of the
    /// `field::runtime` topology's three systems: `body` (somatic
    /// state, slow), `prediction` (inference, fast), or `agency`
    /// (volitional and social). The anatomical map and the field
    /// map are one topology read at two scales.
    pub fiber: &'static str,
}

/// The canonical region table — every body region and the mind
/// package it connects to. Mind-only cortical regions are not
/// listed here; the body's table ends at its own edge.
pub const REGIONS: &[Region] = &[
    Region {
        name: "basal_ganglia",
        package: "basal_ganglia",
        modules: &["cognition::intention"],
        // Action-selection output relays back through the thalamus.
        projections: &["thalamus"],
        fiber: "agency",
    },
    Region {
        name: "brainstem",
        package: "brainstem",
        modules: &[
            "daemon::tick",
            "daemon::interoception",
            "daemon::cpufreq",
            "daemon::active_inference",
            "daemon::rtc_wake",
            "state::core_state",
            "state::zones",
        ],
        // The reticular activating system ascends through thalamus;
        // visceral afferents reach the limbic evaluators.
        projections: &["thalamus", "limbic_system"],
        fiber: "body",
    },
    Region {
        name: "limbic_system",
        package: "limbic_system",
        modules: &["daemon::dyadic_model"],
        // Affective appraisal drives autonomic/endocrine outflow.
        projections: &["brainstem"],
        fiber: "agency",
    },
    Region {
        name: "thalamus",
        package: "thalamus",
        modules: &[
            "daemon::ipc",
            "field::engine",
            "field::runtime",
            "state::manifest",
        ],
        // The relay fans out to every region — these are the named
        // targets of the documented loops, not the whole broadcast.
        projections: &["basal_ganglia", "limbic_system", "memory", "self_", "sleep"],
        fiber: "prediction",
    },
    Region {
        name: "memory",
        package: "memory",
        modules: &[
            "store::ring_buffer",
            "store::ltm",
            "store::sdr",
            "state::memory",
            "daemon::consolidation",
            "daemon::association",
        ],
        // Consolidation and replay are sleep-gated.
        projections: &["sleep"],
        fiber: "body",
    },
    Region {
        name: "self_",
        package: "self",
        modules: &["cognition::self_model"],
        // The proto-self is drawn from interoception — its claims
        // are co-held with limbic evaluation.
        projections: &["limbic_system"],
        fiber: "prediction",
    },
    Region {
        name: "sleep",
        package: "sleep",
        modules: &["daemon::rtc_wake", "state::sleep_network"],
        // Staging gates memory work and holds the brainstem's clock.
        projections: &["memory", "brainstem"],
        fiber: "body",
    },
    Region {
        name: "neurochemical",
        package: "",
        modules: &[
            "state::neurochemical",
            "state::ions",
            "daemon::hpa_development",
        ],
        // Diffuse broadcast — every region reads these levels.
        projections: &["*"],
        fiber: "body",
    },
    Region {
        name: "infrastructure",
        package: "infrastructure",
        modules: &["data_dir", "store::mmap_state"],
        projections: &[],
        fiber: "body",
    },
    Region {
        name: "tools",
        package: "tools",
        modules: &["cognition::code_analysis"],
        projections: &[],
        fiber: "agency",
    },
];

/// Look up a region by its identifier.
pub fn region(name: &str) -> Option<&'static Region> {
    REGIONS.iter().find(|r| r.name == name)
}

/// The mind-side package a region projects to, if it has one.
pub fn package_for(name: &str) -> Option<&'static str> {
    region(name).map(|r| r.package).filter(|p| !p.is_empty())
}

/// The region a mind-side package embodies, if it mirrors the body.
pub fn region_for_package(package: &str) -> Option<&'static Region> {
    REGIONS
        .iter()
        .find(|r| r.package == package && !r.package.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_names_are_unique() {
        for (i, a) in REGIONS.iter().enumerate() {
            assert!(
                !REGIONS[i + 1..].iter().any(|b| b.name == a.name),
                "duplicate region {}",
                a.name
            );
        }
    }

    #[test]
    fn projections_resolve_within_the_table() {
        for r in REGIONS {
            for target in r.projections {
                assert!(
                    *target == "*" || region(target).is_some(),
                    "{} projects to unknown region {}",
                    r.name,
                    target
                );
            }
        }
    }

    #[test]
    fn regions_ride_live_fibers() {
        // The anatomical map and the field map are one topology:
        // every region's fiber must be one of the major systems the
        // runtime actually builds.
        let runtime = crate::field::runtime::FieldRuntime::new().unwrap();
        let fibers: Vec<&str> = runtime
            .field()
            .fibers
            .iter()
            .map(|f| f.id.as_str())
            .collect();
        for r in REGIONS {
            assert!(
                fibers.contains(&r.fiber),
                "{} rides unknown fiber {}",
                r.name,
                r.fiber
            );
        }
    }

    #[test]
    fn declared_modules_exist() {
        // The map must point at real files: a renamed or removed
        // module leaves the spine referencing a phantom.
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for r in REGIONS {
            for m in r.modules {
                let path = src.join(m.replace("::", "/")).with_extension("rs");
                assert!(path.is_file(), "{} declares missing module {}", r.name, m);
            }
        }
    }
}
