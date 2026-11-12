"""Topology — the anatomical region map of the organism. The spine
of the layout: every body region, the body modules that implement
it, and the mind-side package it projects to.

Mirrors ``src/topology.rs``: the body declares the same table from
its side. Neither side can see the other's half — the daemon owns
the body's modules, the mind owns its packages — so the map is
written once per side and joined over the region names, which are
the same identifiers on both ends.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Region:
    """One anatomical region of the organism."""

    # Region identifier — the join key both tables share.
    name: str
    # The mind-side package this region afferents into — "" for
    # body-only machinery with no mirror.
    package: str
    # The body-side modules implementing the region (Rust paths).
    modules: tuple[str, ...]
    # Regions this region projects onward to ("*" for diffuse
    # projections, as with the neuromodulator broadcast).
    projections: tuple[str, ...]
    # The major-system fiber this region rides — one of the field
    # runtime's three systems: "body" (somatic, slow), "prediction"
    # (inference, fast), "agency" (volitional and social).
    fiber: str


# The canonical region table — every body region and the mind
# package it connects to. Body regions only; the cortical tree
# beyond the body's edge is listed separately below.
REGIONS: tuple[Region, ...] = (
    Region(
        name="basal_forebrain",
        package="basal_forebrain",
        modules=(
            "state::basal_forebrain",
            "state::neurochemical",
            "state::sleep_network",
            "state::memory",
        ),
        projections=("memory", "sleep"),
        fiber="prediction",
    ),
    Region(
        name="basal_ganglia",
        package="basal_ganglia",
        modules=("cognition::intention", "daemon::actuators"),
        projections=("thalamus", "worlds"),
        fiber="agency",
    ),
    Region(
        name="brainstem",
        package="brainstem",
        modules=(
            "daemon::tick",
            "daemon::adaptive",
            "daemon::continuity",
            "daemon::interoception",
            "daemon::cpufreq",
            "daemon::active_inference",
            "daemon::maxwell",
            "daemon::mind_shepherd",
            "daemon::rtc_wake",
            "daemon::rtkit",
            "state::core_state",
            "state::header",
            "state::sanitize",
            "state::zones",
        ),
        projections=("thalamus", "insula", "limbic_system"),
        fiber="body",
    ),
    Region(
        name="cerebellum",
        package="cerebellum",
        modules=("latent_spaces::motor", "latent_spaces::engine", "daemon::adaptive"),
        projections=("thalamus", "frontal_lobe"),
        fiber="prediction",
    ),
    Region(
        name="frontal_lobe",
        package="frontal_lobe",
        modules=(
            "cognition::intention",
            "cognition::code_analysis",
            "state::inference",
            "state::zones",
        ),
        projections=("basal_ganglia", "parietal_lobe", "temporal_lobe", "thalamus"),
        fiber="agency",
    ),
    Region(
        name="hypothalamus",
        package="hypothalamus",
        modules=(
            "daemon::hpa_development",
            "state::neurochemical",
            "state::sleep_network",
        ),
        projections=("brainstem", "sleep", "limbic_system"),
        fiber="body",
    ),
    Region(
        name="insula",
        package="insula",
        modules=(
            "daemon::interoception",
            "daemon::active_inference",
            "state::ions",
        ),
        projections=("limbic_system", "self_", "frontal_lobe"),
        fiber="body",
    ),
    Region(
        name="latent_spaces",
        package="latent_spaces",
        modules=(
            "latent_spaces::substrate",
            "latent_spaces::engine",
            "latent_spaces::runtime",
            "latent_spaces::spectrum",
            "latent_spaces::motor",
            "state::phase_partition",
        ),
        projections=("*",),
        fiber="prediction",
    ),
    Region(
        name="limbic_system",
        package="limbic_system",
        modules=("daemon::dyadic_model",),
        projections=("brainstem", "hypothalamus"),
        fiber="agency",
    ),
    Region(
        name="occipital_lobe",
        package="occipital_lobe",
        modules=("daemon::optic_nerve", "daemon::gamma"),
        projections=("parietal_lobe", "temporal_lobe"),
        fiber="prediction",
    ),
    Region(
        name="parietal_lobe",
        package="parietal_lobe",
        modules=("math", "latent_spaces::motor", "daemon::tensor_slab"),
        projections=("frontal_lobe", "cerebellum"),
        fiber="prediction",
    ),
    Region(
        name="temporal_lobe",
        package="temporal_lobe",
        modules=(
            "store::ltm",
            "store::ring_buffer",
            "store::sdr",
            "daemon::consolidation",
            "daemon::association",
        ),
        projections=("frontal_lobe", "memory", "limbic_system"),
        fiber="prediction",
    ),
    Region(
        name="thalamus",
        package="thalamus",
        modules=(
            "daemon::ipc",
            "latent_spaces::engine",
            "latent_spaces::runtime",
            "latent_spaces::spectrum",
            "state::manifest",
        ),
        projections=(
            "basal_ganglia",
            "frontal_lobe",
            "limbic_system",
            "memory",
            "occipital_lobe",
            "parietal_lobe",
            "self_",
            "sleep",
            "temporal_lobe",
        ),
        fiber="prediction",
    ),
    Region(
        name="memory",
        package="memory",
        modules=(
            "store::ring_buffer",
            "store::ltm",
            "store::sdr",
            "state::memory",
            "daemon::consolidation",
            "daemon::association",
        ),
        projections=("sleep",),
        fiber="body",
    ),
    Region(
        name="self_",
        package="self",
        modules=("cognition::self_model",),
        projections=("limbic_system",),
        fiber="prediction",
    ),
    Region(
        name="sleep",
        package="sleep",
        modules=("daemon::rtc_wake", "state::sleep_network"),
        projections=("memory", "brainstem"),
        fiber="body",
    ),
    Region(
        name="neurochemical",
        package="",
        modules=("state::neurochemical", "state::ions", "daemon::hpa_development"),
        projections=("*",),
        fiber="body",
    ),
    Region(
        name="infrastructure",
        package="infrastructure",
        modules=("data_dir", "store::mmap_state"),
        projections=(),
        fiber="body",
    ),
    Region(
        name="tools",
        package="tools",
        modules=("cognition::code_analysis",),
        projections=(),
        fiber="agency",
    ),
    Region(
        name="worlds",
        package="worlds",
        modules=(
            "daemon::actuators",
            "daemon::dyadic_model",
            "daemon::optic_nerve",
        ),
        projections=("occipital_lobe", "limbic_system", "temporal_lobe"),
        fiber="agency",
    ),
)

# Every cortical and subcortical package now has its matching Rust body
# region in REGIONS; MIND_ONLY remains empty for backward compatibility.
MIND_ONLY: tuple[str, ...] = ()


def region(name: str) -> Region | None:
    """Look up a region by its identifier."""
    return next((r for r in REGIONS if r.name == name), None)


def package_for(name: str) -> str | None:
    """The mind-side package a region projects to, if it has one."""
    r = region(name)
    if r is None or not r.package:
        return None
    return r.package


def region_for_package(package: str) -> Region | None:
    """The region a mind-side package embodies, if it mirrors the body."""
    return next((r for r in REGIONS if r.package and r.package == package), None)


@dataclass(frozen=True)
class RegionWiring:
    """One region's ends resolved against the live tree."""

    region: Region
    # The mind-side package resolved to an importable module.
    package_live: bool
    # Body modules declared but missing under src/.
    missing_modules: tuple[str, ...]

    @property
    def connected(self) -> bool:
        """Both ends present (a body-only region counts on modules)."""
        return self.package_live and not self.missing_modules


def _module_paths(module: str, src: Path) -> tuple[Path, Path]:
    """``daemon::tick`` -> ``src/daemon/tick.rs`` and ``src/daemon/tick/mod.rs``.

    Rust resolves ``mod name`` from either layout — a flat file or a
    directory module — so the declared name is satisfied by both.
    """
    parts = module.split("::")
    stem = src.joinpath(*parts)
    return stem.with_suffix(".rs"), stem / "mod.rs"


def connect(root: Path | None = None) -> tuple[RegionWiring, ...]:
    """Join the topology: resolve every declared end against the tree.

    For each region the table names, the mind-side package must
    import (``importlib.util.find_spec``) and every declared body
    module must exist as a ``src/**.rs`` file. Returns one
    :class:`RegionWiring` per region — a region is *connected* when
    both ends are present; a body-only region (``package == ""``)
    counts on its modules alone.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    src = root / "src"
    wiring = []
    for r in REGIONS:
        package_live = importlib.util.find_spec(r.package) is not None if r.package else True
        missing = tuple(
            m for m in r.modules if not any(path.is_file() for path in _module_paths(m, src))
        )
        wiring.append(RegionWiring(r, package_live, missing))
    return tuple(wiring)


def dangling(root: Path | None = None) -> list[str]:
    """Region names whose declared ends do not resolve — breaks in
    the map, where the two halves have drifted apart."""
    return [
        w.region.name
        for w in connect(root)
        if (w.region.package and not w.package_live) or w.missing_modules
    ]


__all__ = [
    "MIND_ONLY",
    "REGIONS",
    "Region",
    "RegionWiring",
    "connect",
    "dangling",
    "package_for",
    "region",
    "region_for_package",
]
