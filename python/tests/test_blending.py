"""Tests for conceptual blending — generic space, selective
projection, emergent structure."""

from frontal_lobe.reasoning.blending import Space, align, blend


def test_generic_space_is_the_shared_structure():
    """The generic space is what the inputs have in common — the
    counterpart pairs under the mapping."""
    a = Space("medieval", {"knight": frozenset({"brave", "armored"})})
    b = Space("space", {"astronaut": frozenset({"brave", "suited"})})
    bl = blend(a, b, {"knight": "astronaut"})
    assert ("knight", "astronaut") in bl.generic


def test_fused_elements_carry_both_sides():
    """The fused element's properties are the union — selective
    projection brings both inputs' features into one."""
    a = Space("a", {"knight": frozenset({"brave", "armored"})})
    b = Space("b", {"astronaut": frozenset({"brave", "suited"})})
    bl = blend(a, b, {"knight": "astronaut"})
    assert bl.elements["knight|astronaut"] == frozenset({"brave", "armored", "suited"})


def test_relations_project_through_the_fusion():
    """An input's relation re-instantiates on the blended elements —
    'knight rides horse' becomes the astronaut riding the rocket."""
    a = Space(
        "medieval",
        {
            "knight": frozenset({"brave"}),
            "horse": frozenset({"carries"}),
        },
        relations=frozenset({("rides", "knight", "horse")}),
    )
    b = Space(
        "space",
        {
            "astronaut": frozenset({"brave"}),
            "rocket": frozenset({"carries"}),
        },
        relations=frozenset(),
    )
    bl = blend(a, b, {"knight": "astronaut", "horse": "rocket"})
    assert ("rides", "knight|astronaut", "horse|rocket") in bl.relations


def test_emergence_is_structure_neither_input_had():
    """The innovation: a relation binding one input's structure to
    the other's material — the astronaut inherits 'rides'."""
    a = Space(
        "medieval",
        {
            "knight": frozenset({"brave"}),
            "horse": frozenset({"carries"}),
        },
        relations=frozenset({("rides", "knight", "horse")}),
    )
    b = Space(
        "space",
        {"astronaut": frozenset({"brave"}), "rocket": frozenset({"fast"})},
    )
    bl = blend(a, b, {"knight": "astronaut", "horse": "rocket"})
    # The blend asserts rides(astronaut-part, rocket-part) — neither
    # input could say that; B's material under A's structure.
    assert bl.novel
    assert ("rides", "knight|astronaut", "horse|rocket") in bl.emergent_relations


def test_emergent_properties_cross_contribute():
    """The fused element gains properties its source lacked —
    the astronaut becomes armored in the blend."""
    a = Space("a", {"knight": frozenset({"armored"})})
    b = Space("b", {"astronaut": frozenset({"suited"})})
    bl = blend(a, b, {"knight": "astronaut"})
    assert "armored" in bl.emergent_properties["knight|astronaut"]
    assert "suited" in bl.emergent_properties["knight|astronaut"]


def test_align_matches_by_shared_structure():
    """Counterpart alignment is structural — the brave-armored pair
    aligns on properties, not on name."""
    a = Space("a", {"x": frozenset({"red", "square"})})
    b = Space("b", {"y": frozenset({"red", "square"}), "z": frozenset({"blue"})})
    assert align(a, b) == {"x": "y"}


def test_empty_inputs_have_nothing_to_say():
    a = Space("a", {})
    b = Space("b", {})
    bl = blend(a, b)
    assert not bl.novel
    assert not bl.elements
