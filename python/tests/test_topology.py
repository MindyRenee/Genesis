"""Topology — the spine stays connected.

``src/topology.rs`` declares the body's half of the region map;
``python/topology.py`` declares the same table back. The join key is
the region name, and the ends are real files: a moved, renamed, or
deleted package or module leaves a dangling end, which is exactly
the drift this guards against.
"""

from __future__ import annotations

import topology


def test_every_region_package_imports() -> None:
    """Every mirrored region's package resolves to a live module."""
    for r in topology.REGIONS:
        if not r.package:
            continue  # body-only region — no mind-side end to check
        wiring = next(w for w in topology.connect() if w.region is r)
        assert wiring.package_live, f"{r.name} package {r.package} does not resolve"


def test_every_declared_body_module_exists() -> None:
    """Every module the table names exists as a src/*.rs file."""
    for w in topology.connect():
        assert not w.missing_modules, (
            f"{w.region.name} declares missing modules: {w.missing_modules}"
        )


def test_nothing_dangles() -> None:
    """The whole spine connects: no region left unresolved."""
    assert topology.dangling() == []


def test_lookup_accessors_agree() -> None:
    """The two directions of the map are consistent."""
    for r in topology.REGIONS:
        assert topology.region(r.name) is r
        if r.package:
            assert topology.package_for(r.name) == r.package
            assert topology.region_for_package(r.package) is r


def test_projection_targets_exist() -> None:
    """No region projects to a name the table does not declare."""
    for r in topology.REGIONS:
        for target in r.projections:
            assert target == "*" or topology.region(target) is not None, (
                f"{r.name} projects to unknown region {target}"
            )


def test_regions_ride_field_fibers() -> None:
    """Every region rides a real major system.

    The three ids mirror ``src/field/runtime.rs``'s fibers — body
    (somatic, slow), prediction (inference, fast), agency (volitional
    and social). The Rust side cross-checks its own table against the
    live FieldRuntime; this side pins the same three names.
    """
    field_fibers = {"body", "prediction", "agency"}
    for r in topology.REGIONS:
        assert r.fiber in field_fibers, f"{r.name} rides unknown fiber {r.fiber}"
