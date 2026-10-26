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
        name="basal_ganglia",
        package="basal_ganglia",
        modules=("cognition::intention",),
        projections=("thalamus",),
        fiber="agency",
    ),
    Region(
        name="brainstem",
        package="brainstem",
        modules=(
            "daemon::tick",
            "daemon::interoception",
            "daemon::cpufreq",
            "daemon::active_inference",
            "daemon::rtc_wake",
            "state::core_state",
            "state::zones",
        ),
        projections=("thalamus", "limbic_system"),
        fiber="body",
    ),
    Region(
        name="limbic_system",
        package="limbic_system",
        modules=("daemon::dyadic_model",),
        projections=("brainstem",),
        fiber="agency",
    ),
    Region(
        name="thalamus",
        package="thalamus",
        modules=("daemon::ipc", "field::engine", "field::runtime", "state::manifest"),
        projections=("basal_ganglia", "limbic_system", "memory", "self_", "sleep"),
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
)

# The cortical tree beyond the body's edge — mind-only regions the
# body's table never names (it ends at its own edge). Kept apart
# from REGIONS: the join table must match the body's row for row.
MIND_ONLY: tuple[str, ...] = (
    "basal_forebrain",
    "cerebellum",
    "frontal_lobe",
    "hypothalamus",
    "insula",
    "occipital_lobe",
    "parietal_lobe",
    "temporal_lobe",
    "worlds",
)


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


def _module_path(module: str, src: Path) -> Path:
    """``daemon::tick`` -> ``src/daemon/tick.rs``; ``data_dir`` -> ``src/data_dir.rs``."""
    parts = module.split("::")
    return src.joinpath(*parts).with_suffix(".rs")


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
        missing = tuple(m for m in r.modules if not _module_path(m, src).is_file())
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
