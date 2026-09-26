"""Tests for the language-to-concept grounding boundary."""

from genesis_cognitive.concepts.network import ConceptNetwork
from genesis_cognitive.language.comprehension import (
    ComprehensionEngine,
    Proposition,
    SemanticRole,
)
from genesis_cognitive.language.grounding import SemanticGrounder


def test_grounder_binds_existing_concepts_without_mutating_network() -> None:
    network = ConceptNetwork()
    network.add_concept("cat", confidence=0.9)
    network.add_concept("mat", confidence=0.8)

    proposition = Proposition(
        subject="cat",
        predicate="sits",
        object="mat",
        roles={SemanticRole.LOCATION: "mat"},
    )
    before = len(network._concepts)

    grounded = SemanticGrounder(network).ground(proposition)

    assert grounded.subject.concept_id == "cat"
    assert grounded.object.concept_id == "mat"
    assert grounded.roles[SemanticRole.LOCATION].concept_id == "mat"
    assert grounded.predicate == "sits"
    assert grounded.fully_grounded
    assert len(network._concepts) == before


def test_grounder_preserves_unknowns_for_learning() -> None:
    network = ConceptNetwork()
    network.add_concept("cat", confidence=0.9)

    grounded = SemanticGrounder(network).ground(
        Proposition(subject="cat", predicate="sees", object="quokka")
    )

    assert grounded.subject.concept_id == "cat"
    assert grounded.object.concept_id is None
    assert grounded.object.surface == "quokka"
    assert not grounded.fully_grounded


def test_grounder_uses_the_same_polysemy_resolution_as_network() -> None:
    network = ConceptNetwork()
    network.add_concept(
        "bank",
        confidence=0.9,
        properties={"definition": "a financial institution"},
    )
    network.add_concept(
        "bank",
        confidence=0.8,
        properties={"definition": "the edge of a river"},
    )

    grounded = SemanticGrounder(network).ground(
        Proposition(subject="bank", predicate="exists")
    )

    assert grounded.subject.concept_id == "bank"
    assert grounded.subject.confidence == 0.9


def test_comprehension_output_can_be_grounded_without_reparsing() -> None:
    network = ConceptNetwork()
    network.add_concept("dog", confidence=0.9)
    network.add_concept("water", confidence=0.8)

    result = ComprehensionEngine().comprehend("The dog drank water.")
    grounded = SemanticGrounder(network).ground_all(result.propositions)

    assert grounded
    assert grounded[0].subject.concept_id == "dog"
    assert grounded[0].object.concept_id == "water"
