"""Regression tests for critical-thinking evidence provenance."""

from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.reasoning import (
    CriticalThinkingEngine,
    ReasoningResult,
    ReasoningType,
)


def test_knowledge_payload_without_provenance_is_not_evidence() -> None:
    """Composition-only knowledge triples must not fabricate support."""
    network = ConceptNetwork()
    network.add_concept("fire")
    network.add_concept("smoke")
    network.add_concept("rain")
    network.add_edge("rain", "smoke", RelationType.CAUSES, weight=1.0)

    result = ReasoningResult(
        conclusion="fire might cause smoke",
        reasoning_type=ReasoningType.HYPOTHESIS,
        evidence=[],
        confidence=0.7,
        novel=True,
        knowledge=[("causes", "smoke", 0.9)],
    )

    assessment = CriticalThinkingEngine(network).evaluate(result)

    assert assessment.supporting_evidence == []
    assert "unsupported_assertion" in assessment.fallacies
    assert assessment.recommendation.name == "INVESTIGATE"
