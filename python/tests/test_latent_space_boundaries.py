"""Tests for typed latent-space boundaries and category-preserving flow."""

import pytest

from latent_spaces.boundaries import (
    Boundary,
    LatentDomain,
    LatentSpaceGraph,
    LatentState,
    energy_from_mass,
)


def test_boundary_connects_spaces_without_merging_categories() -> None:
    edge = Boundary(
        name="perception_to_self",
        source=LatentDomain.E,
        target=LatentDomain.S,
        matrix=((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)),
        allowed_categories=frozenset({"observation"}),
    )
    graph = LatentSpaceGraph([edge])
    source = LatentState(LatentDomain.E, (2.0, 3.0), "observation", 0.8, 4.0)

    result = graph.flow("perception_to_self", source)

    assert result.domain is LatentDomain.S
    assert result.values == (2.0, 3.0, 5.0)
    assert result.category == "observation"
    assert result.confidence == 0.8
    assert result.timestamp == 4.0
    assert graph.history[-1].source is LatentDomain.E
    assert graph.history[-1].target is LatentDomain.S


def test_category_boundary_rejects_unpermitted_flow() -> None:
    graph = LatentSpaceGraph([
        Boundary(
            "language_to_inner",
            LatentDomain.L,
            LatentDomain.I,
            ((1.0,),),
            allowed_categories=frozenset({"intent"}),
        )
    ])
    with pytest.raises(ValueError, match="not permitted"):
        graph.flow(
            "language_to_inner",
            LatentState(LatentDomain.L, (1.0,), "unrelated", timestamp=1.0),
        )


def test_boundary_rejects_wrong_domain_and_dimension() -> None:
    graph = LatentSpaceGraph([
        Boundary("outer_to_language", LatentDomain.E, LatentDomain.L, ((1.0, 2.0),))
    ])
    with pytest.raises(ValueError, match="accepts"):
        graph.flow(
            "outer_to_language",
            LatentState(LatentDomain.I, (1.0, 2.0), "state"),
        )
    with pytest.raises(ValueError, match="expects"):
        graph.flow(
            "outer_to_language",
            LatentState(LatentDomain.E, (1.0,), "observation"),
        )


def test_reachability_uses_directed_edges_not_implicit_mixing() -> None:
    graph = LatentSpaceGraph([
        Boundary("E_to_S", LatentDomain.E, LatentDomain.S, ((1.0,),)),
        Boundary("S_to_L", LatentDomain.S, LatentDomain.L, ((1.0,),)),
        Boundary("I_to_S", LatentDomain.I, LatentDomain.S, ((1.0,),)),
    ])
    assert graph.reachable(LatentDomain.E, LatentDomain.L)
    assert graph.reachable(LatentDomain.I, LatentDomain.L)
    assert not graph.reachable(LatentDomain.L, LatentDomain.E)
    assert graph.reachable(LatentDomain.S, LatentDomain.S)


def test_duplicate_boundary_names_are_rejected() -> None:
    edge = Boundary("edge", LatentDomain.E, LatentDomain.I, ((1.0,),))
    graph = LatentSpaceGraph([edge])
    with pytest.raises(ValueError, match="already exists"):
        graph.add_boundary(edge)


def test_e_equals_mc_squared_is_physical_and_separate_from_latent_flow() -> None:
    assert energy_from_mass(1.0) == pytest.approx(299_792_458.0**2)
    assert energy_from_mass(0.0) == 0.0
    with pytest.raises(ValueError):
        energy_from_mass(-1.0)
