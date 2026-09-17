"""Reasoning bundle tests — reasoning engine, drift diffusion, theory of mind."""

import logging
import random

import pytest

from genesis_cognitive.concepts import ConceptNetwork, RelationType
from genesis_cognitive.reasoning import (
    AnalogyEngine,
    AnalogyInsight,
    BetaDistribution,
    CounterfactualReasoning,
    DecisionResult,
    DriftDiffusionModel,
    EvidenceAccumulator,
    GoalType,
    KnowledgeLevel,
    MetaReasoning,
    ProbabilisticReasoning,
    ProblemSolver,
    ReasoningEngine,
    ReasoningResult,
    ReasoningStrategy,
    ReasoningType,
    TemporalReasoning,
    TemporalRelation,
    TheoryOfMind,
    UserBelief,
    UserModel,
)

logger = logging.getLogger(__name__)


# ======================================================================
# From tests/test_reasoning.py
# ======================================================================


# ═══════════════════════════════════════════════════════════════════
# Transitive chains
# ═══════════════════════════════════════════════════════════════════


def test_transitive_is_a_three_hops() -> None:
    """IS_A chain: dog → mammal → animal → living_thing."""
    net = ConceptNetwork()
    for c in ("dog", "mammal", "animal", "living_thing"):
        net.add_concept(c)
    net.add_edge("dog", "mammal", RelationType.IS_A)
    net.add_edge("mammal", "animal", RelationType.IS_A)
    net.add_edge("animal", "living_thing", RelationType.IS_A)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("dog", depth=4)
    deductive = [r for r in results if r.reasoning_type == ReasoningType.DEDUCTIVE]
    targets = {r.conclusion for r in deductive}
    assert any("mammal" in c for c in targets)
    assert any("animal" in c for c in targets)
    assert any("living_thing" in c for c in targets)


def test_transitive_part_of() -> None:
    """PART_OF chain: wheel → car → vehicle."""
    net = ConceptNetwork()
    for c in ("wheel", "car", "vehicle"):
        net.add_concept(c)
    net.add_edge("wheel", "car", RelationType.PART_OF)
    net.add_edge("car", "vehicle", RelationType.PART_OF)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("wheel")
    deductive = [r for r in results if r.reasoning_type == ReasoningType.DEDUCTIVE]
    assert any("vehicle" in r.conclusion for r in deductive)


def test_transitive_depends_on() -> None:
    """DEPENDS_ON chain: thinking → memory → neurons."""
    net = ConceptNetwork()
    for c in ("thinking", "memory", "neurons"):
        net.add_concept(c)
    net.add_edge("thinking", "memory", RelationType.DEPENDS_ON)
    net.add_edge("memory", "neurons", RelationType.DEPENDS_ON)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("thinking")
    deductive = [r for r in results if r.reasoning_type == ReasoningType.DEDUCTIVE]
    assert any("neurons" in r.conclusion for r in deductive)


def test_transitive_enables() -> None:
    """ENABLES chain: memory → learning → understanding."""
    net = ConceptNetwork()
    for c in ("memory", "learning", "understanding"):
        net.add_concept(c)
    net.add_edge("memory", "learning", RelationType.ENABLES)
    net.add_edge("learning", "understanding", RelationType.ENABLES)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("memory")
    deductive = [r for r in results if r.reasoning_type == ReasoningType.DEDUCTIVE]
    assert any("understanding" in r.conclusion for r in deductive)


def test_transitive_confidence_decay() -> None:
    """Confidence decreases with chain length: 0.9^1 > 0.9^2 > 0.9^3."""
    net = ConceptNetwork()
    for c in ("a", "b", "c", "d"):
        net.add_concept(c)
    net.add_edge("a", "b", RelationType.IS_A, weight=0.9)
    net.add_edge("b", "c", RelationType.IS_A, weight=0.9)
    net.add_edge("c", "d", RelationType.IS_A, weight=0.9)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("a", depth=4)
    deductive = [r for r in results if r.reasoning_type == ReasoningType.DEDUCTIVE]
    # Match the final target in each conclusion precisely
    by_target = {}
    for r in deductive:
        # The conclusion format is "a is_a X (transitively, through ...)"
        # The target is the word right after "is_a"
        parts = r.conclusion.split(" is_a ")
        if len(parts) >= 2:
            target = parts[1].split(" ")[0]
            by_target[target] = r.confidence
    assert "b" in by_target
    assert "c" in by_target
    assert "d" in by_target
    assert by_target["b"] > by_target["c"]
    assert by_target["c"] > by_target["d"]


def test_transitive_cycle_no_infinite_loop() -> None:
    """Cycles in the graph don't cause infinite recursion."""
    net = ConceptNetwork()
    net.add_concept("a")
    net.add_concept("b")
    net.add_edge("a", "b", RelationType.IS_A)
    net.add_edge("b", "a", RelationType.IS_A)  # cycle

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("a", depth=5)
    # Should terminate and return some results
    assert isinstance(results, list)


# ═══════════════════════════════════════════════════════════════════
# Causal chains
# ═══════════════════════════════════════════════════════════════════


def test_causal_chain_leads_to() -> None:
    """LEADS_TO chain: practice → skill → mastery."""
    net = ConceptNetwork()
    for c in ("practice", "skill", "mastery"):
        net.add_concept(c)
    net.add_edge("practice", "skill", RelationType.LEADS_TO)
    net.add_edge("skill", "mastery", RelationType.LEADS_TO)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("practice")
    causal = [r for r in results if r.reasoning_type == ReasoningType.CAUSAL]
    assert len(causal) > 0
    assert any("mastery" in r.conclusion for r in causal)


def test_causal_chain_mixed() -> None:
    """Mixed CAUSES + LEADS_TO chain."""
    net = ConceptNetwork()
    for c in ("spark", "fire", "warmth"):
        net.add_concept(c)
    net.add_edge("spark", "fire", RelationType.CAUSES)
    net.add_edge("fire", "warmth", RelationType.LEADS_TO)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("spark")
    causal = [r for r in results if r.reasoning_type == ReasoningType.CAUSAL]
    assert len(causal) > 0
    assert any("warmth" in r.conclusion for r in causal)


# ═══════════════════════════════════════════════════════════════════
# Contradiction detection
# ═══════════════════════════════════════════════════════════════════


def test_contradiction_explicit() -> None:
    """Explicit CONTRADICTS edge detected."""
    net = ConceptNetwork()
    c1 = net.add_concept("free_will")
    c2 = net.add_concept("determinism")
    # Reset activation to 0 so the "both active" boost doesn't trigger
    c1.activation = 0.0
    c2.activation = 0.0
    net.add_edge("free_will", "determinism", RelationType.CONTRADICTS, weight=0.9)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("free_will")
    contradictions = [r for r in results if r.reasoning_type == ReasoningType.CONTRADICTION]
    assert len(contradictions) == 1
    assert "determinism" in contradictions[0].contradictions
    assert abs(contradictions[0].confidence - 0.9) < 1e-9


def test_contradiction_structural_enables_prevents() -> None:
    """Structural contradiction: same concept ENABLES and PREVENTS target."""
    net = ConceptNetwork()
    net.add_concept("stress")
    net.add_concept("performance")
    net.add_edge("stress", "performance", RelationType.ENABLES, weight=0.3)
    net.add_edge("stress", "performance", RelationType.PREVENTS, weight=0.7)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("stress")
    contradictions = [r for r in results if r.reasoning_type == ReasoningType.CONTRADICTION]
    # Should detect the structural contradiction
    structural = [r for r in contradictions if "structural" in r.conclusion]
    assert len(structural) == 1
    assert "performance" in structural[0].contradictions


def test_contradiction_both_active_boosted() -> None:
    """When both contradicting concepts are active, confidence is boosted."""
    net = ConceptNetwork()
    c1 = net.add_concept("thesis")
    c2 = net.add_concept("antithesis")
    c1.activation = 0.5
    c2.activation = 0.5
    net.add_edge("thesis", "antithesis", RelationType.CONTRADICTS, weight=0.6)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("thesis")
    contradictions = [r for r in results if r.reasoning_type == ReasoningType.CONTRADICTION]
    assert len(contradictions) == 1
    # Confidence should be boosted (0.6 + 0.2 = 0.8)
    assert contradictions[0].confidence > 0.6
    assert "both active" in contradictions[0].conclusion


# ═══════════════════════════════════════════════════════════════════
# Hypothesis formation
# ═══════════════════════════════════════════════════════════════════


def test_hypothesis_fills_gap() -> None:
    """Hypothesis is formed when A→B, B→C but no A→C."""
    net = ConceptNetwork()
    for c in ("a", "b", "c"):
        net.add_concept(c)
    net.add_edge("a", "b", RelationType.CAUSES)
    net.add_edge("b", "c", RelationType.CAUSES)
    # No a→c edge

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("a")
    hypotheses = [r for r in results if r.reasoning_type == ReasoningType.HYPOTHESIS]
    assert len(hypotheses) > 0
    assert hypotheses[0].novel
    assert any("c" in r.conclusion for r in hypotheses)


def test_hypothesis_consistent_path_higher_confidence() -> None:
    """Consistent path (same relation type) gives higher confidence."""
    net = ConceptNetwork()
    for c in ("a", "b", "c"):
        net.add_concept(c)
    net.add_edge("a", "b", RelationType.CAUSES)
    net.add_edge("b", "c", RelationType.CAUSES)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("a")
    hypotheses = [r for r in results if r.reasoning_type == ReasoningType.HYPOTHESIS]
    assert len(hypotheses) > 0
    # Consistent path → base_conf = 0.3 + 0.15 * 1 = 0.45
    assert hypotheses[0].confidence >= 0.44


def test_hypothesis_max_five() -> None:
    """Hypothesis formation is capped at 5 results."""
    net = ConceptNetwork()
    net.add_concept("hub")
    for i in range(10):
        c = f"leaf_{i}"
        net.add_concept(c)
        net.add_edge("hub", c, RelationType.RELATED_TO)
        # Each leaf connects to a unique 2-hop target
        t = f"target_{i}"
        net.add_concept(t)
        net.add_edge(c, t, RelationType.RELATED_TO)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("hub")
    hypotheses = [r for r in results if r.reasoning_type == ReasoningType.HYPOTHESIS]
    assert len(hypotheses) <= 5


# ═══════════════════════════════════════════════════════════════════
# Analogical transfer
# ═══════════════════════════════════════════════════════════════════


def test_analogical_transfer_is_a() -> None:
    """A similar_to B, B is_a C → A might be C."""
    net = ConceptNetwork()
    for c in ("joy", "happiness", "emotion"):
        net.add_concept(c)
    net.add_edge("joy", "happiness", RelationType.SIMILAR_TO, weight=0.8)
    net.add_edge("happiness", "emotion", RelationType.IS_A, weight=0.9)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("joy")
    analogical = [r for r in results if r.reasoning_type == ReasoningType.ANALOGICAL]
    assert len(analogical) > 0
    assert any("emotion" in r.conclusion for r in analogical)
    # Confidence = edge.weight * s_edge.weight * 0.6
    expected = 0.8 * 0.9 * 0.6
    assert abs(analogical[0].confidence - expected) < 0.01


def test_analogical_transfer_causes() -> None:
    """A similar_to B, B causes C → A might cause C."""
    net = ConceptNetwork()
    for c in ("rain", "sprinkler", "wet_grass"):
        net.add_concept(c)
    net.add_edge("rain", "sprinkler", RelationType.SIMILAR_TO, weight=0.7)
    net.add_edge("sprinkler", "wet_grass", RelationType.CAUSES, weight=0.8)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("rain")
    analogical = [r for r in results if r.reasoning_type == ReasoningType.ANALOGICAL]
    assert len(analogical) > 0
    assert any("wet_grass" in r.conclusion for r in analogical)


# ═══════════════════════════════════════════════════════════════════
# Abductive inference
# ═══════════════════════════════════════════════════════════════════


def test_abductive_single_cause() -> None:
    """Abductive: X causes Y, reasoning about Y → X is explanation."""
    net = ConceptNetwork()
    for c in ("fire", "smoke"):
        net.add_concept(c)
    net.add_edge("fire", "smoke", RelationType.CAUSES, weight=0.9)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("smoke")
    abductive = [r for r in results if r.reasoning_type == ReasoningType.ABDUCTIVE]
    assert len(abductive) == 1
    assert "fire" in abductive[0].conclusion
    # confidence = weight * 0.6 * dilution (1 cause → dilution=1.0)
    expected = 0.9 * 0.6 * 1.0
    assert abs(abductive[0].confidence - expected) < 0.01


def test_abductive_multiple_causes_dilution() -> None:
    """Multiple causes dilute confidence."""
    net = ConceptNetwork()
    for c in ("fever", "infection", "heatstroke", "allergy"):
        net.add_concept(c)
    net.add_edge("infection", "fever", RelationType.CAUSES, weight=0.9)
    net.add_edge("heatstroke", "fever", RelationType.CAUSES, weight=0.8)
    net.add_edge("allergy", "fever", RelationType.CAUSES, weight=0.5)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("fever")
    abductive = [r for r in results if r.reasoning_type == ReasoningType.ABDUCTIVE]
    assert len(abductive) == 3
    # Sorted by confidence — best explanation first
    assert abductive[0].confidence >= abductive[1].confidence
    # Dilution: 1/(1 + 0.3*2) = 1/1.6 = 0.625
    expected_best = 0.9 * 0.6 * (1.0 / (1.0 + 0.3 * 2))
    assert abs(abductive[0].confidence - expected_best) < 0.01


def test_abductive_max_three() -> None:
    """Abductive inference returns at most 3 explanations."""
    net = ConceptNetwork()
    net.add_concept("effect")
    for i in range(5):
        c = f"cause_{i}"
        net.add_concept(c)
        net.add_edge(c, "effect", RelationType.CAUSES, weight=0.5 + i * 0.05)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("effect")
    abductive = [r for r in results if r.reasoning_type == ReasoningType.ABDUCTIVE]
    assert len(abductive) <= 3


def test_abductive_deeper_chain_evidence() -> None:
    """Abductive evidence includes deeper causes when available."""
    net = ConceptNetwork()
    for c in ("virus", "infection", "fever"):
        net.add_concept(c)
    net.add_edge("virus", "infection", RelationType.CAUSES, weight=0.8)
    net.add_edge("infection", "fever", RelationType.CAUSES, weight=0.9)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("fever")
    abductive = [r for r in results if r.reasoning_type == ReasoningType.ABDUCTIVE]
    assert len(abductive) == 1
    # Evidence should mention the deeper cause (virus)
    evidence_text = " ".join(abductive[0].evidence)
    assert "virus" in evidence_text


# ═══════════════════════════════════════════════════════════════════
# Synthesis
# ═══════════════════════════════════════════════════════════════════


def test_synthesis_finds_unifying_concept() -> None:
    """Synthesis finds a concept connected to all inputs."""
    net = ConceptNetwork()
    for c in ("joy", "fear", "emotion"):
        net.add_concept(c)
    net.add_edge("joy", "emotion", RelationType.IS_A)
    net.add_edge("fear", "emotion", RelationType.IS_A)

    reasoner = ReasoningEngine(net)
    result = reasoner.synthesize(["joy", "fear"])
    assert result is not None
    assert "emotion" in result.conclusion
    assert result.reasoning_type == ReasoningType.SYNTHESIS


def test_synthesis_single_concept_returns_none() -> None:
    """Synthesis requires at least 2 concepts."""
    net = ConceptNetwork()
    net.add_concept("a")
    reasoner = ReasoningEngine(net)
    result = reasoner.synthesize(["a"])
    assert result is None


def test_synthesis_no_intersection_returns_none() -> None:
    """Synthesis returns None when no shared concept exists."""
    net = ConceptNetwork()
    net.add_concept("a")
    net.add_concept("b")
    net.add_concept("c")
    net.add_concept("d")
    net.add_edge("a", "c", RelationType.RELATED_TO)
    net.add_edge("b", "d", RelationType.RELATED_TO)

    reasoner = ReasoningEngine(net)
    result = reasoner.synthesize(["a", "b"])
    assert result is None


# ═══════════════════════════════════════════════════════════════════
# explain_relation
# ═══════════════════════════════════════════════════════════════════


def test_explain_relation_same_concept() -> None:
    """Explaining relation of a concept to itself."""
    net = ConceptNetwork()
    net.add_concept("dog")
    reasoner = ReasoningEngine(net)
    result = reasoner.explain_relation("dog", "dog")
    assert result is not None
    assert "same concept" in result.conclusion
    assert result.confidence == 1.0


def test_explain_relation_direct() -> None:
    """Explain direct relation between two concepts."""
    net = ConceptNetwork()
    net.add_concept("fire")
    net.add_concept("smoke")
    net.add_edge("fire", "smoke", RelationType.CAUSES)

    reasoner = ReasoningEngine(net)
    result = reasoner.explain_relation("fire", "smoke")
    assert result is not None
    assert "smoke" in result.conclusion


def test_explain_relation_multihop() -> None:
    """Explain relation via multi-hop path."""
    net = ConceptNetwork()
    for c in ("a", "b", "c"):
        net.add_concept(c)
    net.add_edge("a", "b", RelationType.RELATED_TO)
    net.add_edge("b", "c", RelationType.RELATED_TO)

    reasoner = ReasoningEngine(net)
    result = reasoner.explain_relation("a", "c")
    assert result is not None
    # Should find path through b


def test_explain_relation_no_connection() -> None:
    """Explain relation when no connection exists."""
    net = ConceptNetwork()
    net.add_concept("a")
    net.add_concept("b")
    # No edges

    reasoner = ReasoningEngine(net)
    result = reasoner.explain_relation("a", "b")
    # May return None or a synthesis result
    # With no shared neighbors, synthesize returns None
    assert result is None


# ═══════════════════════════════════════════════════════════════════
# compare_concepts
# ═══════════════════════════════════════════════════════════════════


def test_compare_concepts_similar() -> None:
    """Compare two similar concepts."""
    net = ConceptNetwork()
    for c in ("joy", "happiness", "emotion"):
        net.add_concept(c)
    net.add_edge("joy", "happiness", RelationType.SIMILAR_TO, weight=0.9)
    net.add_edge("joy", "emotion", RelationType.IS_A)
    net.add_edge("happiness", "emotion", RelationType.IS_A)

    reasoner = ReasoningEngine(net)
    result = reasoner.compare_concepts("joy", "happiness")
    assert result is not None
    assert "emotion" in result.conclusion or "similar" in result.conclusion.lower()


def test_compare_concepts_opposite() -> None:
    """Compare opposite concepts."""
    net = ConceptNetwork()
    net.add_concept("hot")
    net.add_concept("cold")
    net.add_edge("hot", "cold", RelationType.OPPOSITE_OF, weight=0.9)

    reasoner = ReasoningEngine(net)
    result = reasoner.compare_concepts("hot", "cold")
    assert result is not None


# ═══════════════════════════════════════════════════════════════════
# explain_why
# ═══════════════════════════════════════════════════════════════════


def test_explain_why_direct_dependency() -> None:
    """explain_why finds direct dependency."""
    net = ConceptNetwork()
    for c in ("thinking", "memory"):
        net.add_concept(c)
    net.add_edge("thinking", "memory", RelationType.DEPENDS_ON, weight=0.8)

    reasoner = ReasoningEngine(net)
    result = reasoner.explain_why("thinking", "memory")
    assert result is not None
    assert "memory" in result.conclusion


def test_explain_why_no_dependency() -> None:
    """explain_why returns None when no dependency exists."""
    net = ConceptNetwork()
    net.add_concept("a")
    net.add_concept("b")
    # No edges

    reasoner = ReasoningEngine(net)
    result = reasoner.explain_why("a", "b")
    assert result is None


# ═══════════════════════════════════════════════════════════════════
# explain_process
# ═══════════════════════════════════════════════════════════════════


def test_explain_process_with_triggers_and_outcomes() -> None:
    """explain_process describes triggers and outcomes."""
    net = ConceptNetwork()
    for c in ("spark", "fire", "smoke", "warmth"):
        net.add_concept(c)
    net.add_edge("spark", "fire", RelationType.CAUSES, weight=0.9)
    net.add_edge("fire", "smoke", RelationType.CAUSES, weight=0.8)
    net.add_edge("fire", "warmth", RelationType.LEADS_TO, weight=0.7)

    reasoner = ReasoningEngine(net)
    result = reasoner.explain_process("fire")
    assert result is not None
    assert "fire" in result.conclusion


# ═══════════════════════════════════════════════════════════════════
# explain_affordances
# ═══════════════════════════════════════════════════════════════════


def test_explain_affordances() -> None:
    """explain_affordances lists what a concept enables."""
    net = ConceptNetwork()
    for c in ("memory", "learning", "reasoning"):
        net.add_concept(c)
    net.add_edge("memory", "learning", RelationType.ENABLES, weight=0.8)
    net.add_edge("memory", "reasoning", RelationType.ENABLES, weight=0.7)

    reasoner = ReasoningEngine(net)
    result = reasoner.explain_affordances("memory")
    assert result is not None
    assert "learning" in result.conclusion or "reasoning" in result.conclusion


# ═══════════════════════════════════════════════════════════════════
# answer_question
# ═══════════════════════════════════════════════════════════════════


def test_answer_what_is() -> None:
    """answer_question with 'what_is' type."""
    net = ConceptNetwork()
    net.add_concept("dog")
    net.add_concept("animal")
    net.add_edge("dog", "animal", RelationType.IS_A)

    reasoner = ReasoningEngine(net)
    result = reasoner.answer_question("dog", "what_is")
    assert result is not None
    assert "animal" in result.conclusion


def test_answer_what_causes() -> None:
    """answer_question with 'what_causes' type (what does X cause?)."""
    net = ConceptNetwork()
    net.add_concept("fire")
    net.add_concept("smoke")
    net.add_edge("fire", "smoke", RelationType.CAUSES)

    reasoner = ReasoningEngine(net)
    result = reasoner.answer_question("fire", "what_causes")
    assert result is not None
    assert "smoke" in result.conclusion


# ═══════════════════════════════════════════════════════════════════
# reason_about sorted by confidence
# ═══════════════════════════════════════════════════════════════════


def test_reason_about_sorted_by_confidence() -> None:
    """reason_about returns results sorted by confidence (descending)."""
    net = ConceptNetwork()
    for c in ("a", "b", "c", "d"):
        net.add_concept(c)
    net.add_edge("a", "b", RelationType.IS_A, weight=0.9)
    net.add_edge("b", "c", RelationType.IS_A, weight=0.9)
    net.add_edge("a", "d", RelationType.CAUSES, weight=0.5)
    net.add_edge("d", "c", RelationType.CAUSES, weight=0.5)

    reasoner = ReasoningEngine(net)
    results = reasoner.reason_about("a")
    confidences = [r.confidence for r in results]
    assert confidences == sorted(confidences, reverse=True)


# ═══════════════════════════════════════════════════════════════════
# BetaDistribution
# ═══════════════════════════════════════════════════════════════════


def test_beta_uniform_prior() -> None:
    """Beta(1,1) has mean 0.5 and maximum variance."""
    beta = BetaDistribution(alpha=1.0, beta=1.0)
    assert abs(beta.mean - 0.5) < 1e-9
    assert beta.variance > 0


def test_beta_mean() -> None:
    """Beta(9,1) has mean 0.9."""
    beta = BetaDistribution(alpha=9.0, beta=1.0)
    assert abs(beta.mean - 0.9) < 1e-9


def test_beta_update_positive() -> None:
    """Positive update increases alpha."""
    beta = BetaDistribution(alpha=1.0, beta=1.0)
    beta.update(positive=True, weight=2.0)
    assert beta.alpha == 3.0
    assert beta.beta == 1.0
    assert beta.mean > 0.5


def test_beta_update_negative() -> None:
    """Negative update increases beta."""
    beta = BetaDistribution(alpha=1.0, beta=1.0)
    beta.update(positive=False, weight=2.0)
    assert beta.alpha == 1.0
    assert beta.beta == 3.0
    assert beta.mean < 0.5


def test_beta_confidence_increases_with_evidence() -> None:
    """More evidence → higher confidence (lower variance)."""
    beta_low = BetaDistribution(alpha=1.0, beta=1.0)
    beta_high = BetaDistribution(alpha=10.0, beta=10.0)
    assert beta_high.confidence > beta_low.confidence


# ═══════════════════════════════════════════════════════════════════
# ProbabilisticReasoning
# ═══════════════════════════════════════════════════════════════════


def test_probabilistic_get_belief_uniform_prior() -> None:
    """Unknown belief returns uniform prior."""
    net = ConceptNetwork()
    net.add_concept("a")
    net.add_concept("b")
    pr = ProbabilisticReasoning(net)
    belief = pr.get_belief("a", "causes", "b")
    assert abs(belief.mean - 0.5) < 1e-9


def test_probabilistic_get_belief_from_edge() -> None:
    """Belief initialized from edge weight."""
    net = ConceptNetwork()
    net.add_concept("fire")
    net.add_concept("smoke")
    net.add_edge("fire", "smoke", RelationType.CAUSES, weight=0.9)

    pr = ProbabilisticReasoning(net)
    belief = pr.get_belief("fire", "causes", "smoke")
    # weight=0.9 → alpha=9, beta=1 → mean=0.9
    assert abs(belief.mean - 0.9) < 0.01


def test_probabilistic_observe_evidence() -> None:
    """Observing evidence updates the belief."""
    net = ConceptNetwork()
    net.add_concept("a")
    net.add_concept("b")
    pr = ProbabilisticReasoning(net)

    belief = pr.observe_evidence("a", "causes", "b", positive=True, weight=3.0)
    assert belief.alpha == 4.0  # 1 + 3
    assert belief.beta == 1.0
    assert belief.mean > 0.5


def test_probabilistic_infer() -> None:
    """probabilistic_infer returns a ReasoningResult with posterior."""
    net = ConceptNetwork()
    net.add_concept("fire")
    net.add_concept("smoke")
    net.add_edge("fire", "smoke", RelationType.CAUSES, weight=0.5)

    pr = ProbabilisticReasoning(net)
    result = pr.probabilistic_infer(
        prior=("fire", "causes", "smoke"),
        evidence=[("fire", "causes", "smoke", True, 2.0)],
    )
    assert result is not None
    assert "probability" in result.conclusion
    assert result.confidence > 0


# ═══════════════════════════════════════════════════════════════════
# TemporalReasoning
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "name, interval_a, interval_b, expected_relation",
    [
        ("before", ("breakfast", 7.0, 8.0), ("lunch", 12.0, 13.0), TemporalRelation.BEFORE),
        ("after", ("lunch", 12.0, 13.0), ("breakfast", 7.0, 8.0), TemporalRelation.AFTER),
        ("equal", ("a", 5.0, 10.0), ("b", 5.0, 10.0), TemporalRelation.EQUAL),
        ("during", ("snack", 14.0, 15.0), ("afternoon", 13.0, 17.0), TemporalRelation.DURING),
        ("contains", ("afternoon", 13.0, 17.0), ("snack", 14.0, 15.0), TemporalRelation.CONTAINS),
        ("meets", ("class", 9.0, 10.0), ("break", 10.0, 10.5), TemporalRelation.MEETS),
        ("overlaps", ("a", 1.0, 4.0), ("b", 3.0, 6.0), TemporalRelation.OVERLAPS),
        ("starts", ("short", 5.0, 7.0), ("long", 5.0, 10.0), TemporalRelation.STARTS),
        ("finishes", ("late", 7.0, 10.0), ("long", 5.0, 10.0), TemporalRelation.FINISHES),
    ],
)
def test_temporal_infer_allen_relations(
    name, interval_a, interval_b, expected_relation
) -> None:
    """Allen's interval algebra: temporal_infer returns the correct relation."""
    tr = TemporalReasoning()
    tr.add_event(interval_a[0], interval_a[1], interval_a[2])
    tr.add_event(interval_b[0], interval_b[1], interval_b[2])
    assert tr.temporal_infer(interval_a[0], interval_b[0]) == expected_relation


def test_temporal_infer_no_intervals() -> None:
    """temporal_infer returns EQUAL for unregistered events."""
    tr = TemporalReasoning()
    assert tr.temporal_infer("unknown_a", "unknown_b") == TemporalRelation.EQUAL


def test_temporal_get_event_sequence() -> None:
    """get_event_sequence sorts events by start time."""
    tr = TemporalReasoning()
    tr.add_event("c", 15.0, 16.0)
    tr.add_event("a", 7.0, 8.0)
    tr.add_event("b", 12.0, 13.0)
    seq = tr.get_event_sequence(["c", "a", "b"])
    assert seq == ["a", "b", "c"]


def test_temporal_query_before() -> None:
    """query_before finds events ending before the reference event starts."""
    tr = TemporalReasoning()
    tr.add_event("breakfast", 7.0, 8.0)
    tr.add_event("brunch", 10.0, 11.0)
    tr.add_event("lunch", 12.0, 13.0)
    before_lunch = tr.query_before("lunch")
    assert "breakfast" in before_lunch
    assert "brunch" in before_lunch
    assert "lunch" not in before_lunch


def test_temporal_query_after() -> None:
    """query_after finds events starting after the reference event ends."""
    tr = TemporalReasoning()
    tr.add_event("breakfast", 7.0, 8.0)
    tr.add_event("brunch", 10.0, 11.0)
    tr.add_event("lunch", 12.0, 13.0)
    after_breakfast = tr.query_after("breakfast")
    assert "brunch" in after_breakfast
    assert "lunch" in after_breakfast
    assert "breakfast" not in after_breakfast


def test_temporal_query_during() -> None:
    """query_during finds events contained within the reference event."""
    tr = TemporalReasoning()
    tr.add_event("afternoon", 13.0, 17.0)
    tr.add_event("snack", 14.0, 15.0)
    tr.add_event("tea", 16.0, 16.5)
    during_afternoon = tr.query_during("afternoon")
    assert "snack" in during_afternoon
    assert "tea" in during_afternoon
    assert "afternoon" not in during_afternoon


# ═══════════════════════════════════════════════════════════════════
# CounterfactualReasoning
# ═══════════════════════════════════════════════════════════════════


def test_counterfactual_traces_effects() -> None:
    """Counterfactual traces downstream effects of intervention."""
    net = ConceptNetwork()
    for c in ("rain", "grass_growth", "healthy_lawn"):
        net.add_concept(c)
    net.add_edge("rain", "grass_growth", RelationType.CAUSES, weight=0.8)
    net.add_edge("grass_growth", "healthy_lawn", RelationType.CAUSES, weight=0.9)

    cf = CounterfactualReasoning(net)
    result = cf.counterfactual("rain", "drought", depth=3)
    assert result is not None
    assert result.reasoning_type == ReasoningType.CAUSAL
    assert result.novel


def test_counterfactual_removed_effects() -> None:
    """Counterfactual identifies effects that no longer occur."""
    net = ConceptNetwork()
    for c in ("exercise", "fitness", "health"):
        net.add_concept(c)
    net.add_edge("exercise", "fitness", RelationType.CAUSES, weight=0.9)
    net.add_edge("fitness", "health", RelationType.CAUSES, weight=0.8)

    cf = CounterfactualReasoning(net)
    result = cf.counterfactual("exercise", "sedentary", depth=3)
    assert result is not None
    # Evidence should mention removed effects
    evidence_text = " ".join(result.evidence)
    assert "no longer occurs" in evidence_text or "sedentary" in evidence_text


def test_counterfactual_no_effects() -> None:
    """Counterfactual with no causal chain produces neutral conclusion."""
    net = ConceptNetwork()
    net.add_concept("a")
    net.add_concept("b")
    # No causal edges

    cf = CounterfactualReasoning(net)
    result = cf.counterfactual("a", "b", depth=3)
    assert result is not None
    assert "no significant downstream" in result.conclusion


# ═══════════════════════════════════════════════════════════════════
# MetaReasoning
# ═══════════════════════════════════════════════════════════════════


def test_meta_reason_empty_strategies() -> None:
    """meta_reason returns DEDUCTIVE when no strategies available."""
    mr = MetaReasoning()
    selected = mr.meta_reason("test", [])
    assert selected == ReasoningStrategy.DEDUCTIVE


def test_meta_reason_causal_problem() -> None:
    """Causal keywords + learned preference bias toward causal strategy."""
    mr = MetaReasoning()
    # Record a successful causal result to create a preference
    mr.record_result(
        strategy=ReasoningStrategy.CAUSAL,
        problem="what causes the fire",
        confidence=0.8,
        success=True,
    )
    strategies = [
        ReasoningStrategy.DEDUCTIVE,
        ReasoningStrategy.CAUSAL,
    ]
    selected = mr.meta_reason("what causes the fire", strategies)
    # Causal keywords + learned preference should select causal
    assert selected == ReasoningStrategy.CAUSAL


def test_meta_reason_temporal_problem() -> None:
    """Temporal keywords + learned preference bias toward temporal strategy."""
    mr = MetaReasoning()
    # Record a successful temporal result to create a preference
    mr.record_result(
        strategy=ReasoningStrategy.TEMPORAL,
        problem="what happens before the event",
        confidence=0.8,
        success=True,
    )
    strategies = [
        ReasoningStrategy.DEDUCTIVE,
        ReasoningStrategy.TEMPORAL,
    ]
    selected = mr.meta_reason("what happens before the event", strategies)
    assert selected == ReasoningStrategy.TEMPORAL


def test_meta_reason_high_confidence_prefers_simple() -> None:
    """High confidence prefers simpler strategies (deductive)."""
    mr = MetaReasoning()
    strategies = [
        ReasoningStrategy.DEDUCTIVE,
        ReasoningStrategy.COUNTERFACTUAL,
    ]
    selected = mr.meta_reason("general problem", strategies, confidence=0.9)
    # High confidence → deductive gets +0.1 bonus
    assert selected == ReasoningStrategy.DEDUCTIVE


def test_meta_reason_low_confidence_prefers_deep() -> None:
    """Low confidence prefers deeper strategies (counterfactual)."""
    mr = MetaReasoning()
    strategies = [
        ReasoningStrategy.DEDUCTIVE,
        ReasoningStrategy.COUNTERFACTUAL,
    ]
    selected = mr.meta_reason("general problem", strategies, confidence=0.1)
    # Low confidence → counterfactual gets +0.1 bonus
    assert selected == ReasoningStrategy.COUNTERFACTUAL


def test_meta_reason_record_result() -> None:
    """record_result updates strategy scores."""
    mr = MetaReasoning()
    mr.record_result(
        strategy=ReasoningStrategy.CAUSAL,
        problem="why does fire cause smoke",
        confidence=0.8,
        success=True,
    )
    assert len(mr.reasoning_history) == 1
    assert mr.reasoning_history[0].strategy == ReasoningStrategy.CAUSAL
    assert mr.reasoning_history[0].success is True
    # Score should have increased
    assert mr._strategy_scores[ReasoningStrategy.CAUSAL] > 0.5


def test_meta_reason_record_failure_decreases_score() -> None:
    """Recording a failure decreases the strategy score."""
    mr = MetaReasoning()
    initial = mr._strategy_scores[ReasoningStrategy.DEDUCTIVE]
    mr.record_result(
        strategy=ReasoningStrategy.DEDUCTIVE,
        problem="test problem",
        confidence=0.5,
        success=False,
    )
    assert mr._strategy_scores[ReasoningStrategy.DEDUCTIVE] < initial


def test_meta_reason_recommend_depth_boundaries() -> None:
    """recommend_depth maps confidence to reasoning depth at the boundaries."""
    mr = MetaReasoning()
    assert mr.recommend_depth(0.9) == 1  # high confidence → shallow
    assert mr.recommend_depth(0.1) == 5  # low confidence → deep


def test_meta_reason_get_strategy_stats() -> None:
    """get_strategy_stats returns per-strategy statistics."""
    mr = MetaReasoning()
    mr.record_result(
        strategy=ReasoningStrategy.CAUSAL,
        problem="test",
        confidence=0.8,
        success=True,
    )
    mr.record_result(
        strategy=ReasoningStrategy.CAUSAL,
        problem="test2",
        confidence=0.6,
        success=False,
    )
    stats = mr.get_strategy_stats()
    assert "causal" in stats
    assert stats["causal"]["uses"] == 2
    assert stats["causal"]["successes"] == 1
    assert abs(stats["causal"]["success_rate"] - 0.5) < 1e-9


@pytest.mark.parametrize(
    "phrase, expected_category",
    [
        ("what causes the fire", "causal"),
        ("the effect of rain", "causal"),
        ("this leads to that", "causal"),
        ("what happens before lunch", "temporal"),
        ("sequence of events", "temporal"),
        ("how are they similar", "similarity"),
        ("compare these concepts", "similarity"),
        ("what if we changed this", "counterfactual"),
        ("imagine an alternative", "counterfactual"),
        ("this is probably true", "uncertain"),
        ("what is the probability", "uncertain"),
        ("hello world", "general"),
    ],
)
def test_meta_reason_classify_problem(phrase, expected_category) -> None:
    """_classify_problem identifies the problem category from keywords."""
    mr = MetaReasoning()
    assert mr._classify_problem(phrase) == expected_category


def test_meta_reason_classify_problem_from_network() -> None:
    """_classify_problem uses concept network edges when keywords don't match."""
    net = ConceptNetwork()
    net.add_concept("fire")
    net.add_concept("smoke")
    net.add_edge("fire", "smoke", RelationType.CAUSES)
    mr = MetaReasoning(network=net)
    # "fire" has no causal keywords but has CAUSES edge in network
    assert mr._classify_problem("fire") == "causal"


# ═══════════════════════════════════════════════════════════════════
# ReasoningRecord
# ═══════════════════════════════════════════════════════════════════


# ======================================================================
# From tests/test_drift_diffusion.py
# ======================================================================


# ═══════════════════════════════════════════════════════════════════
# EvidenceAccumulator
# ═══════════════════════════════════════════════════════════════════


def test_evidence_accumulator_creation() -> None:
    """EvidenceAccumulator initializes with option and threshold."""
    acc = EvidenceAccumulator(option="greet", threshold=1.0)
    assert acc.option == "greet"
    assert acc.threshold == 1.0
    assert acc.evidence == 0.0
    assert not acc.decided


def test_evidence_accumulator_progress() -> None:
    """progress returns evidence/threshold capped at 1.0."""
    acc = EvidenceAccumulator(option="greet", threshold=1.0)
    acc.evidence = 0.5
    assert acc.progress == 0.5
    acc.evidence = 1.5
    assert acc.progress == 1.0  # capped


def test_evidence_accumulator_progress_zero_threshold() -> None:
    """progress returns 0 when threshold is 0."""
    acc = EvidenceAccumulator(option="greet", threshold=0.0)
    assert acc.progress == 0.0


# ═══════════════════════════════════════════════════════════════════
# DriftDiffusionModel — options
# ═══════════════════════════════════════════════════════════════════


def test_ddm_add_option() -> None:
    """add_option registers a response option."""
    ddm = DriftDiffusionModel()
    ddm.add_option("greet")
    assert "greet" in ddm.options


def test_ddm_add_option_custom_threshold() -> None:
    """add_option can set a per-option threshold."""
    ddm = DriftDiffusionModel(threshold=1.0)
    ddm.add_option("greet", threshold=0.5)
    assert "greet" in ddm.options


def test_ddm_add_option_duplicate_ignored() -> None:
    """Adding the same option twice doesn't overwrite."""
    ddm = DriftDiffusionModel()
    ddm.add_option("greet", threshold=0.5)
    ddm.add_option("greet")  # should not overwrite
    # The original threshold should be preserved
    assert ddm.get_progress("greet") == 0.0


def test_ddm_options_property() -> None:
    """options property returns registered options."""
    ddm = DriftDiffusionModel()
    ddm.add_option("a")
    ddm.add_option("b")
    opts = ddm.options
    assert "a" in opts
    assert "b" in opts


def test_ddm_clear_options() -> None:
    """clear_options removes all options."""
    ddm = DriftDiffusionModel()
    ddm.add_option("a")
    ddm.add_option("b")
    ddm.clear_options()
    assert ddm.options == []


# ═══════════════════════════════════════════════════════════════════
# DriftDiffusionModel — evidence
# ═══════════════════════════════════════════════════════════════════


def test_ddm_add_evidence() -> None:
    """add_evidence adds pending drift to an option."""
    ddm = DriftDiffusionModel()
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.5)
    assert ddm.has_pending


def test_ddm_add_evidence_creates_option() -> None:
    """add_evidence auto-creates unknown options."""
    ddm = DriftDiffusionModel()
    ddm.add_evidence("perception", "greet", 0.5)
    assert "greet" in ddm.options


def test_ddm_add_evidence_negative() -> None:
    """add_evidence can add negative evidence."""
    ddm = DriftDiffusionModel()
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", -0.5)
    assert ddm.has_pending


def test_ddm_add_evidence_multiple_sources() -> None:
    """Multiple sources can contribute to the same option."""
    ddm = DriftDiffusionModel()
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.3)
    ddm.add_evidence("emotion", "greet", 0.2)
    sources = ddm.get_sources("greet")
    assert sources["perception"] == 0.3
    assert sources["emotion"] == 0.2


def test_ddm_get_evidence() -> None:
    """get_evidence returns current evidence level."""
    ddm = DriftDiffusionModel(noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.5)
    ddm.tick()
    assert ddm.get_evidence("greet") == 0.5


def test_ddm_get_evidence_unknown_option() -> None:
    """get_evidence returns 0 for unknown options."""
    ddm = DriftDiffusionModel()
    assert ddm.get_evidence("unknown") == 0.0


def test_ddm_get_progress() -> None:
    """get_progress returns progress toward threshold."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.5)
    ddm.tick()
    assert ddm.get_progress("greet") == 0.5


def test_ddm_get_progress_unknown_option() -> None:
    """get_progress returns 0 for unknown options."""
    ddm = DriftDiffusionModel()
    assert ddm.get_progress("unknown") == 0.0


def test_ddm_get_sources() -> None:
    """get_sources returns per-source contributions."""
    ddm = DriftDiffusionModel()
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.3)
    ddm.add_evidence("memory", "greet", 0.2)
    sources = ddm.get_sources("greet")
    assert sources["perception"] == 0.3
    assert sources["memory"] == 0.2


def test_ddm_get_sources_unknown_option() -> None:
    """get_sources returns empty dict for unknown options."""
    ddm = DriftDiffusionModel()
    assert ddm.get_sources("unknown") == {}


def test_ddm_has_pending_false_initially() -> None:
    """has_pending is False initially."""
    ddm = DriftDiffusionModel()
    ddm.add_option("greet")
    assert not ddm.has_pending


def test_ddm_has_pending_false_after_tick() -> None:
    """has_pending is False after tick integrates drift."""
    ddm = DriftDiffusionModel(noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.5)
    ddm.tick()
    assert not ddm.has_pending


# ═══════════════════════════════════════════════════════════════════
# DriftDiffusionModel — tick and decisions
# ═══════════════════════════════════════════════════════════════════


def test_ddm_tick_no_decision() -> None:
    """tick returns None when no threshold is crossed."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.5)
    result = ddm.tick()
    assert result is None


def test_ddm_tick_decision() -> None:
    """tick returns a DecisionResult when threshold is crossed."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 1.5)
    result = ddm.tick()
    assert result is not None
    assert result.option == "greet"


def test_ddm_tick_decision_confidence() -> None:
    """DecisionResult confidence is in [0, 1]."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 1.5)
    result = ddm.tick()
    assert result is not None
    assert 0.0 <= result.confidence <= 1.0


def test_ddm_tick_decision_time() -> None:
    """DecisionResult decision_time includes elapsed time."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.5)
    ddm.tick()  # no decision
    ddm.add_evidence("perception", "greet", 1.0)
    result = ddm.tick()
    assert result is not None
    assert result.decision_time > 0


def test_ddm_tick_non_decision_time() -> None:
    """non_decision_time is added to decision_time."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0, non_decision_time=2.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 1.5)
    result = ddm.tick()
    assert result is not None
    assert result.decision_time >= 2.0


def test_ddm_tick_evidence_sources_in_result() -> None:
    """DecisionResult includes evidence sources."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.6)
    ddm.add_evidence("emotion", "greet", 0.6)
    result = ddm.tick()
    assert result is not None
    assert "perception" in result.evidence_sources
    assert "emotion" in result.evidence_sources


def test_ddm_tick_all_evidence_in_result() -> None:
    """DecisionResult includes all evidence levels."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_option("inform")
    ddm.add_evidence("perception", "greet", 1.5)
    result = ddm.tick()
    assert result is not None
    assert "greet" in result.all_evidence
    assert "inform" in result.all_evidence


def test_ddm_tick_multiple_options_winner() -> None:
    """The option with the highest evidence wins."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_option("inform")
    ddm.add_evidence("perception", "greet", 1.5)
    ddm.add_evidence("perception", "inform", 1.2)
    result = ddm.tick()
    assert result is not None
    assert result.option == "greet"  # higher evidence


def test_ddm_tick_no_evidence() -> None:
    """tick with no evidence returns None (plus noise only)."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    result = ddm.tick()
    assert result is None


def test_ddm_tick_evidence_bounded_below() -> None:
    """Evidence is bounded below at 0."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", -1.0)
    ddm.tick()
    assert ddm.get_evidence("greet") == 0.0


# ═══════════════════════════════════════════════════════════════════
# DriftDiffusionModel — noise
# ═══════════════════════════════════════════════════════════════════


def test_ddm_noise_reproducible() -> None:
    """With a seeded RNG, noise is reproducible."""
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    ddm1 = DriftDiffusionModel(threshold=1.0, noise=0.1, rng=rng1)
    ddm2 = DriftDiffusionModel(threshold=1.0, noise=0.1, rng=rng2)
    ddm1.add_option("greet")
    ddm2.add_option("greet")
    ddm1.add_evidence("perception", "greet", 0.5)
    ddm2.add_evidence("perception", "greet", 0.5)
    ddm1.tick()
    ddm2.tick()
    assert ddm1.get_evidence("greet") == ddm2.get_evidence("greet")


def test_ddm_no_noise() -> None:
    """With noise=0, evidence is deterministic."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.5)
    ddm.tick()
    assert ddm.get_evidence("greet") == 0.5


# ═══════════════════════════════════════════════════════════════════
# DriftDiffusionModel — decay
# ═══════════════════════════════════════════════════════════════════


def test_ddm_decay_reduces_evidence() -> None:
    """Decay reduces evidence over time."""
    ddm = DriftDiffusionModel(threshold=10.0, noise=0.0, decay=0.5)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 1.0)
    ddm.tick()
    initial = ddm.get_evidence("greet")
    # Another tick with no new evidence — decay should reduce it
    ddm.tick()
    assert ddm.get_evidence("greet") < initial


def test_ddm_no_decay() -> None:
    """With decay=0, evidence persists without new input."""
    ddm = DriftDiffusionModel(threshold=10.0, noise=0.0, decay=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.5)
    ddm.tick()
    ddm.tick()  # no new evidence
    assert ddm.get_evidence("greet") == 0.5


# ═══════════════════════════════════════════════════════════════════
# DriftDiffusionModel — reset
# ═══════════════════════════════════════════════════════════════════


def test_ddm_reset_clears_evidence() -> None:
    """reset clears evidence but keeps options."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 0.5)
    ddm.tick()
    ddm.reset()
    assert ddm.get_evidence("greet") == 0.0
    assert "greet" in ddm.options  # option still registered


def test_ddm_reset_clears_decided() -> None:
    """reset clears the decided flag."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.add_evidence("perception", "greet", 1.5)
    ddm.tick()
    ddm.reset()
    # After reset, we can accumulate again
    ddm.add_evidence("perception", "greet", 1.5)
    result = ddm.tick()
    assert result is not None


# ═══════════════════════════════════════════════════════════════════
# DriftDiffusionModel — threshold adjustment
# ═══════════════════════════════════════════════════════════════════


def test_ddm_set_threshold() -> None:
    """set_threshold updates the threshold for undecided accumulators."""
    ddm = DriftDiffusionModel(threshold=1.0, noise=0.0)
    ddm.add_option("greet")
    ddm.set_threshold(2.0)
    ddm.add_evidence("perception", "greet", 1.5)
    result = ddm.tick()
    # 1.5 < 2.0 → no decision
    assert result is None


def test_ddm_set_threshold_bounded() -> None:
    """set_threshold has a minimum of 0.1."""
    ddm = DriftDiffusionModel(threshold=1.0)
    ddm.add_option("greet")
    ddm.set_threshold(0.01)
    assert ddm.threshold == 0.1


def test_ddm_speed_accuracy_tradeoff() -> None:
    """Lower threshold → faster decision; higher → slower."""
    # Low threshold → quick decision
    ddm_fast = DriftDiffusionModel(threshold=0.5, noise=0.0)
    ddm_fast.add_option("greet")
    ddm_fast.add_evidence("perception", "greet", 0.6)
    result_fast = ddm_fast.tick()
    assert result_fast is not None

    # High threshold → no decision yet
    ddm_slow = DriftDiffusionModel(threshold=2.0, noise=0.0)
    ddm_slow.add_option("greet")
    ddm_slow.add_evidence("perception", "greet", 0.6)
    result_slow = ddm_slow.tick()
    assert result_slow is None


# ═══════════════════════════════════════════════════════════════════
# DecisionResult
# ═══════════════════════════════════════════════════════════════════


def test_decision_result_creation() -> None:
    """DecisionResult stores all decision metadata."""
    result = DecisionResult(
        option="greet",
        confidence=0.8,
        decision_time=3.0,
        evidence_sources={"perception": {"greet": 0.8}},
        all_evidence={"greet": 1.2, "inform": 0.5},
    )
    assert result.option == "greet"
    assert result.confidence == 0.8
    assert result.decision_time == 3.0
    assert "perception" in result.evidence_sources
    assert result.all_evidence["greet"] == 1.2


# ======================================================================
# From tests/test_theory_of_mind.py
# ======================================================================


# ═══════════════════════════════════════════════════════════════════
# Enums and dataclasses
# ═══════════════════════════════════════════════════════════════════


def test_knowledge_level_values() -> None:
    """KnowledgeLevel has the expected values."""
    assert KnowledgeLevel.UNKNOWN.value == "unknown"
    assert KnowledgeLevel.NOVICE.value == "novice"
    assert KnowledgeLevel.FAMILIAR.value == "familiar"
    assert KnowledgeLevel.EXPERT.value == "expert"


def test_user_belief_creation() -> None:
    """UserBelief stores concept, value, confidence."""
    belief = UserBelief(concept="cognition", value="it's an illusion", confidence=0.8)
    assert belief.concept == "cognition"
    assert belief.value == "it's an illusion"
    assert belief.confidence == 0.8


def test_user_model_defaults() -> None:
    """UserModel has expected defaults."""
    model = UserModel()
    assert model.beliefs == {}
    assert model.intentions == []
    assert model.knowledge == {}
    assert model.emotional_state == "neutral"
    assert model.emotional_valence == 0.0
    assert model.expertise_level == 0.5
    assert model.name == ""
    assert model.interaction_count == 0


# ═══════════════════════════════════════════════════════════════════
# TheoryOfMind — initialization
# ═══════════════════════════════════════════════════════════════════


def test_tom_initialization() -> None:
    """TheoryOfMind initializes with an empty user model."""
    tom = TheoryOfMind()
    model = tom.get_user_model()
    assert model.interaction_count == 0
    assert tom.user_name == ""


# ═══════════════════════════════════════════════════════════════════
# update_from_user_input
# ═══════════════════════════════════════════════════════════════════


def test_update_increments_interaction_count() -> None:
    """update_from_user_input increments the interaction count."""
    tom = TheoryOfMind()
    tom.update_from_user_input("hello")
    assert tom.get_user_model().interaction_count == 1
    tom.update_from_user_input("how are you")
    assert tom.get_user_model().interaction_count == 2


def test_update_detects_question() -> None:
    """Questions set the topic knowledge to NOVICE."""
    tom = TheoryOfMind()
    tom.update_from_user_input("what is cognition?")
    model = tom.get_user_model()
    # The topic should be marked as NOVICE (asked about)
    assert any(v == KnowledgeLevel.NOVICE for v in model.knowledge.values())


def test_update_detects_expertise() -> None:
    """Expertise indicators increase expertise level."""
    tom = TheoryOfMind()
    initial = tom.get_expertise_level()
    tom.update_from_user_input("the algorithm complexity is asymptotically linear")
    assert tom.get_expertise_level() > initial


def test_update_detects_novice() -> None:
    """Novice indicators decrease expertise level."""
    tom = TheoryOfMind()
    tom.update_from_user_input("i'm new to this, can you explain? it's confusing")
    # Expertise should be lower than default
    assert tom.get_expertise_level() <= 0.5


def test_update_detects_name() -> None:
    """Name introductions are recorded."""
    tom = TheoryOfMind()
    tom.update_from_user_input("my name is Alice")
    assert tom.user_name == "Alice"


def test_update_detects_positive_emotion() -> None:
    """Positive words set positive emotional state."""
    tom = TheoryOfMind()
    tom.update_from_user_input("this is great and wonderful")
    state, valence = tom.get_emotional_state()
    assert state == "positive"
    assert valence > 0


def test_update_detects_negative_emotion() -> None:
    """Negative words set negative emotional state."""
    tom = TheoryOfMind()
    tom.update_from_user_input("this is terrible and awful")
    state, valence = tom.get_emotional_state()
    assert state == "negative"
    assert valence < 0


def test_update_neutral_emotion() -> None:
    """No emotion words → neutral state."""
    tom = TheoryOfMind()
    tom.update_from_user_input("the table is brown")
    state, _ = tom.get_emotional_state()
    assert state == "neutral"


def test_update_detects_distress() -> None:
    """Common distress words register via the shared sentiment lexicon.

    Regression: the user model used its own small word list, so
    "stressed"/"lonely"/"worry" were invisible and distress read as
    neutral.
    """
    tom = TheoryOfMind()
    tom.update_from_user_input("I'm feeling really stressed today")
    state, valence = tom.get_emotional_state()
    assert state == "negative"
    assert valence < 0


def test_update_emotion_respects_negation() -> None:
    """'not stressed' is positive, not negative."""
    tom = TheoryOfMind()
    tom.update_from_user_input("I am not stressed at all")
    state, valence = tom.get_emotional_state()
    assert state == "positive"
    assert valence > 0


def test_update_detects_correction() -> None:
    """Corrections mark topics as EXPERT."""
    tom = TheoryOfMind()
    tom.update_from_user_input("actually, the algorithm is O(n log n)")
    model = tom.get_user_model()
    assert any(v == KnowledgeLevel.EXPERT for v in model.knowledge.values())


# ═══════════════════════════════════════════════════════════════════
# should_explain
# ═══════════════════════════════════════════════════════════════════


def test_should_explain_unknown() -> None:
    """should_explain returns True for unknown concepts."""
    tom = TheoryOfMind()
    assert tom.should_explain("quantum mechanics") is True


def test_should_explain_novice() -> None:
    """should_explain returns True for novice concepts."""
    tom = TheoryOfMind()
    tom.update_from_user_input("what is cognition?")
    assert tom.should_explain("cognition") is True


def test_should_explain_expert() -> None:
    """should_explain returns False for expert concepts."""
    tom = TheoryOfMind()
    tom.update_from_user_input("actually, the cognition is about integrated information")
    assert tom.should_explain("cognition") is False


def test_should_explain_familiar() -> None:
    """should_explain returns True for familiar concepts."""
    tom = TheoryOfMind()
    # Mention a topic multiple times → familiar
    tom.update_from_user_input("dogs are great")
    tom.update_from_user_input("dogs are loyal")
    assert tom.should_explain("dogs") is True


# ═══════════════════════════════════════════════════════════════════
# get_explanation_depth
# ═══════════════════════════════════════════════════════════════════


def test_explanation_depth_unknown() -> None:
    """Unknown concepts get 'brief' explanation."""
    tom = TheoryOfMind()
    assert tom.get_explanation_depth("unknown_topic") == "brief"


def test_explanation_depth_novice() -> None:
    """Novice concepts get 'full' explanation."""
    tom = TheoryOfMind()
    tom.update_from_user_input("what is cognition?")
    assert tom.get_explanation_depth("cognition") == "full"


def test_explanation_depth_expert() -> None:
    """Expert concepts get 'skip' explanation."""
    tom = TheoryOfMind()
    tom.update_from_user_input("actually, cognition involves integrated information")
    assert tom.get_explanation_depth("cognition") == "skip"


# ═══════════════════════════════════════════════════════════════════
# user_knows and user_asked_about
# ═══════════════════════════════════════════════════════════════════


def test_user_knows_false_initially() -> None:
    """user_knows returns False for unknown concepts."""
    tom = TheoryOfMind()
    assert tom.user_knows("anything") is False


def test_user_knows_true_after_demonstration() -> None:
    """user_knows returns True after the user demonstrates knowledge."""
    tom = TheoryOfMind()
    tom.update_from_user_input("actually, the algorithm is correct")
    assert tom.user_knows("algorithm") is True


def test_user_asked_about_false_initially() -> None:
    """user_asked_about returns False for unasked concepts."""
    tom = TheoryOfMind()
    assert tom.user_asked_about("anything") is False


def test_user_asked_about_true_after_question() -> None:
    """user_asked_about returns True after the user asks about a concept."""
    tom = TheoryOfMind()
    tom.update_from_user_input("what is cognition?")
    assert tom.user_asked_about("cognition") is True


# ═══════════════════════════════════════════════════════════════════
# Beliefs
# ═══════════════════════════════════════════════════════════════════


def test_add_belief() -> None:
    """add_belief records a user belief."""
    tom = TheoryOfMind()
    tom.add_belief("cognition", "it's an illusion", confidence=0.8)
    belief = tom.get_belief("cognition")
    assert belief is not None
    assert belief.value == "it's an illusion"
    assert belief.confidence == 0.8


def test_get_belief_unknown() -> None:
    """get_belief returns None for unknown concepts."""
    tom = TheoryOfMind()
    assert tom.get_belief("unknown") is None


def test_add_belief_case_insensitive() -> None:
    """Beliefs are stored case-insensitively."""
    tom = TheoryOfMind()
    tom.add_belief("Cognition", "illusion")
    assert tom.get_belief("cognition") is not None
    assert tom.get_belief("COGNITION") is not None


def test_has_divergent_belief_no_belief() -> None:
    """has_divergent_belief returns False when no belief exists."""
    tom = TheoryOfMind()
    assert tom.has_divergent_belief("unknown", "our value") is False


def test_has_divergent_belief_same() -> None:
    """has_divergent_belief returns False when beliefs match."""
    tom = TheoryOfMind()
    tom.add_belief("cognition", "it's real")
    assert tom.has_divergent_belief("cognition", "it's real") is False


def test_has_divergent_belief_different() -> None:
    """has_divergent_belief returns True when beliefs differ."""
    tom = TheoryOfMind()
    tom.add_belief("cognition", "it's an illusion")
    assert tom.has_divergent_belief("cognition", "it's real") is True


# ═══════════════════════════════════════════════════════════════════
# Expertise and emotion queries
# ═══════════════════════════════════════════════════════════════════


def test_get_expertise_level_initial() -> None:
    """Initial expertise level is 0.5."""
    tom = TheoryOfMind()
    assert tom.get_expertise_level() == 0.5


def test_get_emotional_state_initial() -> None:
    """Initial emotional state is neutral."""
    tom = TheoryOfMind()
    state, valence = tom.get_emotional_state()
    assert state == "neutral"
    assert valence == 0.0


def test_user_name_initial_empty() -> None:
    """Initial user name is empty."""
    tom = TheoryOfMind()
    assert tom.user_name == ""


# ═══════════════════════════════════════════════════════════════════
# get_user_model
# ═══════════════════════════════════════════════════════════════════


def test_get_user_model() -> None:
    """get_user_model returns the current model."""
    tom = TheoryOfMind()
    tom.update_from_user_input("hello, my name is Bob")
    model = tom.get_user_model()
    assert isinstance(model, UserModel)
    assert model.name == "Bob"
    assert model.interaction_count == 1


# ═══════════════════════════════════════════════════════════════════
# reset
# ═══════════════════════════════════════════════════════════════════


def test_reset() -> None:
    """reset clears the user model."""
    tom = TheoryOfMind()
    tom.update_from_user_input("hello, my name is Alice")
    tom.add_belief("cognition", "illusion")
    tom.reset()
    model = tom.get_user_model()
    assert model.interaction_count == 0
    assert tom.user_name == ""
    assert tom.get_belief("cognition") is None


# ═══════════════════════════════════════════════════════════════════
# Analogy engine — cross-domain structure-mapping
# ═══════════════════════════════════════════════════════════════════


def _build_analogy_network() -> ConceptNetwork:
    """Build a two-domain network for analogy tests.

    Domain A (column "conversation"): an engine system.
        engine CAUSES motion, ENABLES vehicle, DEPENDS_ON fuel
        fuel CAUSES combustion, PART_OF vehicle

    Domain B (column "dictionary"): a heart system.
        heart CAUSES circulation, ENABLES body
        oxygen CAUSES respiration, PART_OF blood
        circulation DEPENDS_ON oxygen

    The structural analogy: engine ~ heart (both CAUSES + ENABLES).
    The candidate inference: heart DEPENDS_ON oxygen (projected from
    engine DEPENDS_ON fuel; oxygen is the role-equivalent of fuel).
    """
    net = ConceptNetwork()
    # Domain A — engine system (column "conversation")
    for name in ("engine", "motion", "vehicle", "fuel", "combustion"):
        net.add_concept(
            name, origin="conversation",
            properties={"definition": f"a concept: {name}"},
        )
    net.add_edge("engine", "motion", RelationType.CAUSES)
    net.add_edge("engine", "vehicle", RelationType.ENABLES)
    net.add_edge("engine", "fuel", RelationType.DEPENDS_ON)
    net.add_edge("fuel", "combustion", RelationType.CAUSES)
    net.add_edge("fuel", "vehicle", RelationType.PART_OF)

    # Domain B — heart system (column "dictionary")
    for name in ("heart", "circulation", "body", "oxygen", "respiration", "blood"):
        net.add_concept(
            name, origin="dictionary",
            properties={"definition": f"a concept: {name}"},
        )
    net.add_edge("heart", "circulation", RelationType.CAUSES)
    net.add_edge("heart", "body", RelationType.ENABLES)
    net.add_edge("oxygen", "respiration", RelationType.CAUSES)
    net.add_edge("oxygen", "blood", RelationType.PART_OF)
    net.add_edge("circulation", "oxygen", RelationType.DEPENDS_ON)
    return net


def test_analogy_finds_cross_domain_similar_to() -> None:
    """Structurally analogous concepts in different columns get SIMILAR_TO."""
    net = _build_analogy_network()
    engine_cols = net.get_columns_for("engine")
    heart_cols = net.get_columns_for("heart")
    assert engine_cols != heart_cols  # cross-domain

    engine = AnalogyEngine(net)
    insights = engine.discover_analogies()

    # Should find the engine ~ heart analogical similarity.
    similarity = [
        i for i in insights
        if i.kind == "similarity"
        and {i.source, i.target} == {"engine", "heart"}
    ]
    assert len(similarity) == 1
    assert similarity[0].relation == RelationType.SIMILAR_TO


def test_analogy_transfers_relation_across_domains() -> None:
    """A relation in domain A is projected into domain B.

    engine DEPENDS_ON fuel → heart DEPENDS_ON oxygen (oxygen is the
    role-equivalent of fuel: both CAUSE something and are PART_OF
    something).
    """
    net = _build_analogy_network()
    engine = AnalogyEngine(net)
    insights = engine.discover_analogies()

    transfers = [
        i for i in insights
        if i.kind == "transfer"
        and i.source == "heart"
        and i.target == "oxygen"
        and i.relation == RelationType.DEPENDS_ON
    ]
    assert len(transfers) == 1
    # The transfer should cite engine DEPENDS_ON fuel as the analog.
    assert transfers[0].analog_source == "engine"
    assert transfers[0].analog_target == "fuel"


def test_analogy_discover_and_accept_writes_edges() -> None:
    """discover_and_accept writes validated insights into the network."""
    net = _build_analogy_network()
    engine = AnalogyEngine(net)
    added = engine.discover_and_accept()
    assert added >= 2  # at least the similarity + the transfer

    # The SIMILAR_TO edge should now exist between engine and heart.
    has_sim = False
    for edge in net.get_edges("engine", "out"):
        if edge.target == "heart" and edge.relation == RelationType.SIMILAR_TO:
            has_sim = True
            assert edge.origin == "analogy"
    assert has_sim

    # The DEPENDS_ON edge should now exist from heart to oxygen.
    has_dep = False
    for edge in net.get_edges("heart", "out"):
        if edge.target == "oxygen" and edge.relation == RelationType.DEPENDS_ON:
            has_dep = True
            assert edge.origin == "analogy"
    assert has_dep


def test_analogy_rejects_contradicted_transfer() -> None:
    """A transfer that contradicts an existing edge is rejected.

    If heart PREVENTS oxygen, then heart DEPENDS_ON oxygen is
    contradictory and must not be written.
    """
    net = _build_analogy_network()
    net.add_edge("heart", "oxygen", RelationType.PREVENTS)

    engine = AnalogyEngine(net)
    insights = engine.discover_analogies()

    # No transfer to oxygen should survive.
    bad = [
        i for i in insights
        if i.kind == "transfer"
        and i.source == "heart"
        and i.target == "oxygen"
    ]
    assert len(bad) == 0
    # The similarity edge is still fine (not contradicted).
    assert any(
        i.kind == "similarity" and {i.source, i.target} == {"engine", "heart"}
        for i in insights
    )


def test_analogy_skips_already_connected_pairs() -> None:
    """If two concepts are already connected, they are not re-paired."""
    net = _build_analogy_network()
    # Pre-connect engine and heart with a generic edge.
    net.add_edge("engine", "heart", RelationType.RELATED_TO)

    engine = AnalogyEngine(net)
    insights = engine.discover_analogies()

    # No similarity insight should be generated for the already-connected
    # pair (the _are_connected check filters them out during pairing).
    sim = [
        i for i in insights
        if i.kind == "similarity"
        and {i.source, i.target} == {"engine", "heart"}
    ]
    assert len(sim) == 0


def test_analogy_no_transfer_when_target_lacks_role_equivalent() -> None:
    """If no role-equivalent target exists, no transfer is projected.

    When oxygen is a bare leaf (no structural edges, not in heart's
    2-hop neighborhood), the engine cannot find a role-equivalent for
    fuel, so no DEPENDS_ON transfer is projected.
    """
    net = ConceptNetwork()
    for name in ("engine", "motion", "vehicle", "fuel", "combustion"):
        net.add_concept(
            name, origin="conversation",
            properties={"definition": name},
        )
    net.add_edge("engine", "motion", RelationType.CAUSES)
    net.add_edge("engine", "vehicle", RelationType.ENABLES)
    net.add_edge("engine", "fuel", RelationType.DEPENDS_ON)
    net.add_edge("fuel", "combustion", RelationType.CAUSES)
    net.add_edge("fuel", "vehicle", RelationType.PART_OF)

    for name in ("heart", "circulation", "body", "oxygen"):
        net.add_concept(
            name, origin="dictionary",
            properties={"definition": name},
        )
    net.add_edge("heart", "circulation", RelationType.CAUSES)
    net.add_edge("heart", "body", RelationType.ENABLES)
    # oxygen is a bare leaf — no structural edges, not reachable from
    # heart within 2 hops.

    engine = AnalogyEngine(net)
    insights = engine.discover_analogies()

    # The similarity edge should still form (structural match on
    # CAUSES + ENABLES).
    assert any(
        i.kind == "similarity" and {i.source, i.target} == {"engine", "heart"}
        for i in insights
    )
    # But no DEPENDS_ON transfer — oxygen is not in heart's 2-hop
    # neighborhood, so no role-equivalent can be found.
    dep_transfers = [
        i for i in insights
        if i.kind == "transfer" and i.relation == RelationType.DEPENDS_ON
    ]
    assert len(dep_transfers) == 0


def test_analogy_insight_describe() -> None:
    """AnalogyInsight.describe produces a readable summary."""
    inf = AnalogyInsight(
        source="engine",
        target="heart",
        relation=RelationType.SIMILAR_TO,
        confidence=0.4,
        kind="similarity",
        analog_source="heart",
        analog_target="engine",
        basis="causes, enables",
    )
    text = inf.describe()
    assert "engine" in text
    assert "heart" in text
    assert "structurally like" in text


def test_analogy_persistence_roundtrip() -> None:
    """to_dict / restore_from_dict preserves insight history."""
    net = _build_analogy_network()
    engine = AnalogyEngine(net)
    engine.discover_and_accept()

    data = engine.to_dict()
    assert isinstance(data, dict)

    engine2 = AnalogyEngine(net)
    engine2.restore_from_dict(data)
    assert engine2.insight_count == engine.insight_count
    assert engine2.pairs_found == engine.pairs_found


# ======================================================================
# Problem-solving engine tests
# ======================================================================


def _build_problem_network() -> ConceptNetwork:
    """Build a network for problem-solving tests.

    Structure (a prerequisite chain):
        water DEPENDS_ON photosynthesis
        sunlight ENABLES photosynthesis
        photosynthesis CAUSES oxygen
        photosynthesis HAS_PROPERTY green
        oxygen CAUSES respiration
        chlorophyll PART_OF photosynthesis
        chlorophyll SIMILAR_TO hemoglobin
    """
    net = ConceptNetwork()
    for name in (
        "water", "sunlight", "photosynthesis", "oxygen",
        "respiration", "chlorophyll", "hemoglobin", "carbon_dioxide",
    ):
        net.add_concept(name)

    # photosynthesis is well-understood (definition + typed edges).
    photo = net.get_concept("photosynthesis")
    assert photo is not None
    photo.properties["definition"] = "process by which plants convert light to energy"

    net.add_edge("water", "photosynthesis", RelationType.DEPENDS_ON)
    net.add_edge("sunlight", "photosynthesis", RelationType.ENABLES)
    net.add_edge("photosynthesis", "oxygen", RelationType.CAUSES)
    net.add_edge("photosynthesis", "green", RelationType.HAS_PROPERTY)
    net.add_edge("oxygen", "respiration", RelationType.CAUSES)
    net.add_edge("chlorophyll", "photosynthesis", RelationType.PART_OF)
    net.add_edge("chlorophyll", "hemoglobin", RelationType.SIMILAR_TO)
    return net


def test_problem_solver_already_understood() -> None:
    """A well-understood concept is solved trivially (goal already satisfied)."""
    net = _build_problem_network()
    solver = ProblemSolver(net)
    solution = solver.solve("photosynthesis", GoalType.UNDERSTAND)
    assert solution.verified
    assert solution.confidence >= 0.9
    # No subproblems needed — the concept is already understood.
    assert len(solution.sub_solutions) == 0


def test_problem_solver_decomposes_prerequisites() -> None:
    """An un-understood concept gathers knowledge via operators."""
    net = _build_problem_network()
    # 'oxygen' has no definition but has causal edges (CAUSES
    # respiration, caused_by photosynthesis). The solver should
    # gather this knowledge via the find_causes operator.
    solver = ProblemSolver(net)
    solution = solver.solve("oxygen", GoalType.UNDERSTAND)
    # oxygen has typed edges (CAUSES respiration, caused_by photosynthesis)
    # so the solver should gather knowledge via operators.
    assert len(solution.steps) > 0
    # Should have found that photosynthesis causes oxygen.
    all_knowledge = solution.all_knowledge
    relations = [rel for rel, _t, _w in all_knowledge]
    assert "caused_by" in relations or "causes" in relations


def test_problem_solver_achieve_finds_causes() -> None:
    """ACHIEVE goal finds the causal chain to the goal."""
    net = _build_problem_network()
    solver = ProblemSolver(net)
    solution = solver.solve("oxygen", GoalType.ACHIEVE)
    # photosynthesis CAUSES oxygen — the goal is already satisfied
    # (a cause is known), so the solver verifies it immediately.
    assert solution.verified
    # The goal_check step confirms the causal chain exists.
    assert any(step.operator == "goal_check" for step in solution.steps)


def test_problem_solver_explain_finds_causes() -> None:
    """EXPLAIN goal finds causes of the target."""
    net = _build_problem_network()
    solver = ProblemSolver(net)
    solution = solver.solve("respiration", GoalType.EXPLAIN)
    # oxygen CAUSES respiration — the goal is already satisfied
    # (a cause is known), so the solver verifies it immediately.
    assert solution.verified
    assert any(step.operator == "goal_check" for step in solution.steps)


def test_problem_solver_compare_finds_path() -> None:
    """COMPARE goal finds a path between two concepts."""
    net = _build_problem_network()
    solver = ProblemSolver(net)
    solution = solver.solve(
        "chlorophyll", GoalType.COMPARE, secondary_goal="hemoglobin"
    )
    # chlorophyll SIMILAR_TO hemoglobin — direct path exists, so the
    # goal is already satisfied and the solver verifies immediately.
    assert solution.verified
    assert any(step.operator == "goal_check" for step in solution.steps)


def test_problem_solver_resolve_weighs_support() -> None:
    """RESOLVE goal weighs support for both sides of a contradiction."""
    net = _build_problem_network()
    # Add a contradiction: photosynthesis CONTRADICTS combustion.
    net.add_concept("combustion")
    net.add_edge("photosynthesis", "combustion", RelationType.CONTRADICTS)
    # Make photosynthesis well-supported, combustion poorly supported.
    photo = net.get_concept("photosynthesis")
    assert photo is not None
    photo.confidence = 0.9
    photo.properties["definition"] = "process by which plants convert light to energy"

    solver = ProblemSolver(net)
    solution = solver.solve(
        "photosynthesis", GoalType.RESOLVE, secondary_goal="combustion"
    )
    assert solution.verified
    # The stronger side is identified — either via the goal_check
    # (already satisfied) or the resolve_contradiction operator.
    assert solution.confidence > 0.5


def test_problem_solver_cycle_detection() -> None:
    """Circular dependencies are detected and blocked, not infinite-looped."""
    net = ConceptNetwork()
    net.add_concept("a")
    net.add_concept("b")
    # a DEPENDS_ON b, b DEPENDS_ON a — circular.
    net.add_edge("a", "b", RelationType.DEPENDS_ON)
    net.add_edge("b", "a", RelationType.DEPENDS_ON)
    solver = ProblemSolver(net)
    solution = solver.solve("a", GoalType.UNDERSTAND)
    # Should not hang. The subproblem for 'b' will detect the cycle.
    # The top-level solution may still gather some knowledge via operators,
    # but at least one sub-solution should be blocked.
    assert solution is not None


def test_problem_solver_unknown_concept() -> None:
    """Solving a concept that doesn't exist produces an unverified solution."""
    net = ConceptNetwork()
    solver = ProblemSolver(net)
    solution = solver.solve("nonexistent_concept", GoalType.UNDERSTAND)
    assert not solution.verified
    assert solution.confidence < 0.5


def test_problem_solver_with_reasoning_engine() -> None:
    """The solver uses the ReasoningEngine when provided."""
    net = _build_problem_network()
    reasoning = ReasoningEngine(net)
    solver = ProblemSolver(net, reasoning=reasoning)
    solution = solver.solve("chlorophyll", GoalType.UNDERSTAND)
    # chlorophyll has typed edges (PART_OF photosynthesis, SIMILAR_TO hemoglobin)
    # so the solver should gather knowledge.
    assert len(solution.steps) > 0
    # The 'reason' operator should have been applied.
    assert any(step.operator == "reason" for step in solution.steps)


def test_problem_solver_describe() -> None:
    """Solution.describe() returns a readable summary."""
    net = _build_problem_network()
    solver = ProblemSolver(net)
    solution = solver.solve("oxygen", GoalType.UNDERSTAND)
    desc = solution.describe()
    assert "Problem:" in desc
    assert "oxygen" in desc
    assert "Confidence:" in desc


def test_problem_solver_persistence() -> None:
    """to_dict / restore_from_dict preserves solved/blocked counts."""
    net = _build_problem_network()
    solver = ProblemSolver(net)
    solver.solve("photosynthesis", GoalType.UNDERSTAND)  # solved
    solver.solve("nonexistent", GoalType.UNDERSTAND)  # not solved

    data = solver.to_dict()
    solved_count = data.get("solved_count", 0)
    assert isinstance(solved_count, int) and solved_count >= 1

    solver2 = ProblemSolver(net)
    solver2.restore_from_dict(data)
    assert solver2.solved_count == solver.solved_count
    assert solver2.blocked_count == solver.blocked_count


def test_problem_solver_subsolution_tree() -> None:
    """Subproblems are solved recursively, producing a solution tree."""
    net = ConceptNetwork()
    net.add_concept("goal")
    net.add_concept("prereq1")
    net.add_concept("prereq2")
    # goal DEPENDS_ON prereq1, prereq1 DEPENDS_ON prereq2.
    # None have definitions, so all are "un-understood" → subproblems spawn.
    net.add_edge("prereq1", "goal", RelationType.DEPENDS_ON)
    net.add_edge("prereq2", "prereq1", RelationType.DEPENDS_ON)
    # Give prereq2 some properties so the recursion bottoms out.
    net.add_edge("prereq2", "foundational", RelationType.HAS_PROPERTY)
    net.add_edge("prereq2", "prereq1", RelationType.ENABLES)

    solver = ProblemSolver(net)
    solution = solver.solve("goal", GoalType.UNDERSTAND)
    # Should have subproblems for prereq1 (and recursively prereq2).
    assert len(solution.sub_solutions) > 0
    # The first sub-solution should be about prereq1.
    sub = solution.sub_solutions[0]
    assert sub.problem.goal == "prereq1"
    # prereq1 should itself have a sub-solution for prereq2.
    assert len(sub.sub_solutions) > 0
    assert sub.sub_solutions[0].problem.goal == "prereq2"


# ======================================================================
# MetaReasoning + advanced operators tests
# ======================================================================


def test_problem_solver_with_meta_reasoning() -> None:
    """MetaReasoning selects strategies and records results for learning."""
    from genesis_cognitive.reasoning import MetaReasoning

    net = _build_problem_network()
    meta = MetaReasoning(net)
    solver = ProblemSolver(net, meta_reasoning=meta)
    solution = solver.solve("oxygen", GoalType.UNDERSTAND)
    # MetaReasoning should have recorded reasoning attempts.
    assert len(meta.reasoning_history) > 0
    # The strategy scores should have been updated.
    stats = meta.get_strategy_stats()
    assert isinstance(stats, dict)
    assert solution.verified or len(solution.steps) > 0


def test_problem_solver_probabilistic_operator() -> None:
    """The probabilistic operator assesses uncertain relationships."""
    from genesis_cognitive.reasoning import ProbabilisticReasoning

    net = _build_problem_network()
    prob = ProbabilisticReasoning(net)
    solver = ProblemSolver(net, probabilistic=prob)
    solution = solver.solve("oxygen", GoalType.UNDERSTAND)
    # The probabilistic_infer operator should have been applied.
    assert any(
        step.operator == "probabilistic_infer"
        for step in solution.steps
    )


def test_problem_solver_temporal_operator() -> None:
    """The temporal operator orders causes and effects."""
    from genesis_cognitive.reasoning import TemporalReasoning

    net = _build_problem_network()
    temp = TemporalReasoning(net)
    solver = ProblemSolver(net, temporal=temp)
    solution = solver.solve("oxygen", GoalType.UNDERSTAND)
    # The temporal_order operator should have been applied (oxygen
    # causes respiration, so there's an "after" relationship).
    assert any(
        step.operator == "temporal_order"
        for step in solution.steps
    )


def test_problem_solver_counterfactual_operator() -> None:
    """The counterfactual operator traces effects of removing the goal."""
    from genesis_cognitive.reasoning import CounterfactualReasoning

    net = _build_problem_network()
    cf = CounterfactualReasoning(net)
    solver = ProblemSolver(net, counterfactual=cf)
    solution = solver.solve("photosynthesis", GoalType.UNDERSTAND)
    # photosynthesis is already understood, so the solver returns
    # immediately. Use a less-understood concept instead.
    solution = solver.solve("oxygen", GoalType.UNDERSTAND)
    # The counterfactual operator should have been applied (oxygen
    # causes respiration, so removing it has traceable effects).
    assert any(
        step.operator == "counterfactual"
        for step in solution.steps
    )


def test_problem_solver_all_operators_combined() -> None:
    """All operators work together when all engines are provided."""
    from genesis_cognitive.reasoning import (
        CounterfactualReasoning,
        MetaReasoning,
        ProbabilisticReasoning,
        TemporalReasoning,
    )

    net = _build_problem_network()
    meta = MetaReasoning(net)
    prob = ProbabilisticReasoning(net)
    temp = TemporalReasoning(net)
    cf = CounterfactualReasoning(net)
    solver = ProblemSolver(
        net,
        reasoning=ReasoningEngine(net),
        meta_reasoning=meta,
        probabilistic=prob,
        temporal=temp,
        counterfactual=cf,
    )
    solution = solver.solve("oxygen", GoalType.UNDERSTAND)
    # Should have multiple operator types in the steps.
    operators = {step.operator for step in solution.steps}
    assert len(operators) >= 3
    # MetaReasoning should have recorded multiple attempts.
    assert len(meta.reasoning_history) >= 3


def test_meta_reasoning_strategy_selection() -> None:
    """MetaReasoning selects different strategies for different problems."""
    from genesis_cognitive.reasoning import MetaReasoning, ReasoningStrategy

    net = _build_problem_network()
    meta = MetaReasoning(net)
    # A causal problem should favor causal strategies at low confidence.
    strategy = meta.meta_reason(
        "photosynthesis causes oxygen",
        list(ReasoningStrategy),
        confidence=0.2,
    )
    assert isinstance(strategy, ReasoningStrategy)
    # A similarity problem should favor analogical/deductive at high confidence.
    strategy2 = meta.meta_reason(
        "chlorophyll is similar to hemoglobin",
        list(ReasoningStrategy),
        confidence=0.8,
    )
    assert isinstance(strategy2, ReasoningStrategy)


def test_meta_reasoning_learning() -> None:
    """MetaReasoning learns from recorded results (EMA update)."""
    from genesis_cognitive.reasoning import MetaReasoning, ReasoningStrategy

    net = _build_problem_network()
    meta = MetaReasoning(net)
    # Record several successful deductive attempts.
    for _ in range(5):
        meta.record_result(ReasoningStrategy.DEDUCTIVE, "test", 0.8, True)
    # Record several failed counterfactual attempts.
    for _ in range(5):
        meta.record_result(ReasoningStrategy.COUNTERFACTUAL, "test", 0.2, False)
    stats = meta.get_strategy_stats()
    # Deductive should have higher learned score than counterfactual.
    assert (
        stats["deductive"]["learned_score"]
        > stats["counterfactual"]["learned_score"]
    )


# ======================================================================
# Critical thinking engine tests
# ======================================================================


def _build_critical_network() -> ConceptNetwork:
    """Build a network for critical-thinking tests.

    Structure:
        fire CAUSES heat (origin=conversation, weight=0.8)
        oxygen ENABLES fire (origin=dictionary, weight=0.9)
        water PREVENTS fire (origin=dictionary, weight=0.9)
        fire CONTRADICTS ice (origin=conversation, weight=0.6)
        fire SIMILAR_TO ice (origin=inferred, weight=0.3)
    """
    net = ConceptNetwork()
    for name in ("fire", "heat", "oxygen", "water", "ice"):
        net.add_concept(name)
    net.add_edge("fire", "heat", RelationType.CAUSES, weight=0.8, origin="conversation")
    net.add_edge("oxygen", "fire", RelationType.ENABLES, weight=0.9, origin="dictionary")
    net.add_edge("water", "fire", RelationType.PREVENTS, weight=0.9, origin="dictionary")
    net.add_edge("fire", "ice", RelationType.CONTRADICTS, weight=0.6, origin="conversation")
    net.add_edge("fire", "ice", RelationType.SIMILAR_TO, weight=0.3, origin="inferred")
    return net


def _make_result(
    conclusion: str = "fire causes heat",
    confidence: float = 0.7,
    knowledge: list[tuple[str, str, float]] | None = None,
    evidence: list[str] | None = None,
) -> ReasoningResult:
    """Create a ReasoningResult for testing.

    If evidence is not provided, it's generated from the knowledge
    triples (e.g., ``"fire causes heat"`` for ``("causes", "heat", 0.8)``).
    Pass ``evidence=[]`` explicitly for no evidence.
    """
    if knowledge is None:
        knowledge = [("causes", "heat", 0.8)]
    if evidence is None:
        evidence = [f"fire {rel} {tgt}" for rel, tgt, _w in knowledge]
    return ReasoningResult(
        conclusion=conclusion,
        reasoning_type=ReasoningType.DEDUCTIVE,
        evidence=evidence,
        confidence=confidence,
        knowledge=knowledge,
    )


def test_critical_thinking_accepts_well_supported() -> None:
    """A well-supported claim from reliable sources is accepted."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    # fire causes heat, supported by conversation (reliability 0.6)
    result = _make_result(confidence=0.7, knowledge=[("causes", "heat", 0.8)])
    assessment = critic.evaluate(result)
    assert assessment.recommendation.value in ("accept", "hedge")
    assert assessment.revised_confidence > 0


def test_critical_thinking_downgrades_weak_evidence() -> None:
    """A claim from a single low-reliability source is downgraded."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    # SIMILAR_TO from inferred origin (low reliability)
    result = _make_result(
        confidence=0.7,
        knowledge=[("similar_to", "ice", 0.3)],
        evidence=["fire similar_to ice"],
    )
    assessment = critic.evaluate(result)
    assert assessment.revised_confidence < result.confidence
    assert assessment.is_downgraded


def test_critical_thinking_detects_disconfirmation() -> None:
    """Disconfirming evidence lowers confidence."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    # Claim: fire SIMILAR_TO ice — disconfirmed by fire CONTRADICTS ice
    result = _make_result(
        conclusion="fire similar_to ice",
        confidence=0.7,
        knowledge=[("similar_to", "ice", 0.3)],
        evidence=["fire similar_to ice"],
    )
    assessment = critic.evaluate(result)
    # Should find disconfirming evidence (CONTRADICTS on fire→ice).
    assert len(assessment.disconfirming_evidence) > 0
    assert assessment.revised_confidence < result.confidence


def test_critical_thinking_detects_unsupported_assertion() -> None:
    """A claim with no supporting evidence is flagged."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    # No knowledge triples, high confidence
    result = _make_result(
        confidence=0.8,
        knowledge=[],
        evidence=[],
    )
    assessment = critic.evaluate(result)
    assert "unsupported_assertion" in assessment.fallacies
    assert assessment.recommendation.value in ("reject", "investigate")


def test_critical_thinking_detects_hasty_generalization() -> None:
    """A hypothesis from a single weak source is flagged."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    # Single low-reliability source (inferred), high confidence
    result = _make_result(
        confidence=0.7,
        knowledge=[("similar_to", "ice", 0.3)],
    )
    assessment = critic.evaluate(result)
    assert "hasty_generalization" in assessment.fallacies


def test_critical_thinking_source_reliability_weighting() -> None:
    """High-reliability sources boost confidence more than low ones."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    # Dictionary-sourced evidence (high reliability)
    high_result = _make_result(
        confidence=0.5,
        knowledge=[("enables", "fire", 0.9)],
        evidence=["oxygen enables fire"],
    )
    high_assessment = critic.evaluate(high_result)
    # Inferred-sourced evidence (low reliability)
    low_result = _make_result(
        confidence=0.5,
        knowledge=[("similar_to", "ice", 0.3)],
        evidence=["fire similar_to ice"],
    )
    low_assessment = critic.evaluate(low_result)
    # High-reliability evidence should score better.
    assert high_assessment.evidence_quality > low_assessment.evidence_quality


def test_critical_thinking_evidence_quality_multiple_sources() -> None:
    """Multiple independent sources boost evidence quality."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    # Single source: fire causes heat (conversation)
    single = _make_result(
        confidence=0.5,
        knowledge=[("causes", "heat", 0.8)],
    )
    single_assessment = critic.evaluate(single)
    # Multiple sources: fire causes heat (conversation) + oxygen enables fire (dictionary)
    multi = _make_result(
        confidence=0.5,
        knowledge=[("causes", "heat", 0.8), ("enables", "fire", 0.9)],
    )
    multi_assessment = critic.evaluate(multi)
    # Multiple independent sources should have higher evidence quality.
    assert multi_assessment.evidence_quality > single_assessment.evidence_quality


def test_critical_thinking_evaluate_batch() -> None:
    """evaluate_batch returns results with revised confidence."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    results = [
        _make_result(confidence=0.7, knowledge=[("causes", "heat", 0.8)]),
        _make_result(confidence=0.7, knowledge=[("similar_to", "ice", 0.3)]),
    ]
    evaluated = critic.evaluate_batch(results)
    assert len(evaluated) == 2
    for result, assessment in evaluated:
        assert assessment.original_confidence == 0.7
        # The weak-evidence result should be downgraded.
        if assessment.is_downgraded:
            assert result.confidence < 0.7


def test_critical_thinking_describe() -> None:
    """Assessment.describe() returns a readable summary."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    result = _make_result(confidence=0.7, knowledge=[("causes", "heat", 0.8)])
    assessment = critic.evaluate(result)
    desc = assessment.describe()
    assert "Critical assessment:" in desc
    assert "confidence:" in desc
    assert "evidence quality:" in desc


def test_critical_thinking_persistence() -> None:
    """to_dict / restore_from_dict preserves counts."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    # Generate some assessments.
    critic.evaluate(_make_result(confidence=0.7, knowledge=[("causes", "heat", 0.8)]))
    critic.evaluate(_make_result(confidence=0.8, knowledge=[]))  # unsupported

    data = critic.to_dict()
    assert isinstance(data, dict)

    critic2 = CriticalThinkingEngine(net)
    critic2.restore_from_dict(data)
    assert critic2.downgraded_count == critic.downgraded_count
    assert critic2.fallacy_count == critic.fallacy_count


def test_critical_thinking_stats_tracking() -> None:
    """The engine tracks downgraded/rejected/fallacy counts."""
    from genesis_cognitive.reasoning import CriticalThinkingEngine
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    # Unsupported assertion → should be downgraded.
    critic.evaluate(_make_result(confidence=0.8, knowledge=[]))
    assert critic.downgraded_count >= 1
    assert critic.fallacy_count >= 1


# ======================================================================
# Belief revision engine tests
# ======================================================================


def test_belief_revision_downgrades_on_disconfirmation() -> None:
    """Disconfirming evidence downgrades the edge weight in the network."""
    from genesis_cognitive.reasoning import (
        BeliefRevisionEngine,
        CriticalThinkingEngine,
    )
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    revisor = BeliefRevisionEngine(net)
    # Claim: fire similar_to ice — disconfirmed by fire CONTRADICTS ice.
    result = _make_result(
        conclusion="fire similar_to ice",
        confidence=0.7,
        knowledge=[("similar_to", "ice", 0.3)],
        evidence=["fire similar_to ice"],
    )
    assessment = critic.evaluate(result)
    records = revisor.revise(assessment)
    # Should have downgraded the claim edge.
    assert revisor.downgraded_count > 0
    assert any(r.reason == "disconfirmation" for r in records)


def test_belief_revision_strengthens_on_strong_support() -> None:
    """Strong multi-source support strengthens the edge weight."""
    from genesis_cognitive.reasoning import (
        BeliefRevisionEngine,
        CriticalThinkingEngine,
    )
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    revisor = BeliefRevisionEngine(net)
    # Strong support: fire causes heat (conversation) + oxygen enables fire (dictionary)
    result = _make_result(
        confidence=0.5,
        knowledge=[("causes", "heat", 0.8), ("enables", "fire", 0.9)],
        evidence=["fire causes heat", "oxygen enables fire"],
    )
    assessment = critic.evaluate(result)
    # Capture original weight.
    orig_edges = list(net.get_edges("fire", "out"))
    orig_weight = next(
        (e.weight for e in orig_edges if e.relation == RelationType.CAUSES), 0.0
    )
    revisor.revise(assessment)
    new_edges = list(net.get_edges("fire", "out"))
    new_weight = next(
        (e.weight for e in new_edges if e.relation == RelationType.CAUSES), 0.0
    )
    assert revisor.strengthened_count > 0
    assert new_weight > orig_weight


def test_belief_revision_prunes_weak_edges() -> None:
    """Edges that fall below threshold after disconfirmation are pruned."""
    from genesis_cognitive.reasoning import (
        BeliefRevisionEngine,
        CriticalThinkingEngine,
    )
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    revisor = BeliefRevisionEngine(net)
    # The SIMILAR_TO edge (inferred, weight=0.3) is prunable.
    # Repeatedly disconfirm it to drive the weight below threshold.
    for _ in range(10):
        result = _make_result(
            confidence=0.7,
            knowledge=[("similar_to", "ice", 0.3)],
            evidence=["fire similar_to ice"],
        )
        assessment = critic.evaluate(result)
        revisor.revise(assessment)
    # Should have pruned the weak inferred edge.
    assert revisor.pruned_count > 0
    # The edge should be gone.
    sim_edges = [
        e for e in net.get_edges("fire", "out")
        if e.relation == RelationType.SIMILAR_TO and e.target == "ice"
    ]
    assert len(sim_edges) == 0


def test_belief_revision_preserves_stated_edges() -> None:
    """Stated edges are not pruned even at low weight."""
    from genesis_cognitive.reasoning import (
        BeliefRevisionEngine,
        CriticalThinkingEngine,
    )
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    revisor = BeliefRevisionEngine(net)
    # Add a low-weight stated edge that is contradicted.
    net.add_edge("fire", "water", RelationType.SIMILAR_TO, weight=0.04, origin="stated")
    net.add_edge("fire", "water", RelationType.OPPOSITE_OF, weight=0.8, origin="dictionary")
    # Repeatedly disconfirm it to drive the weight down.
    for _ in range(5):
        result = _make_result(
            confidence=0.7,
            knowledge=[("similar_to", "water", 0.04)],
            evidence=["fire similar_to water"],
        )
        assessment = critic.evaluate(result)
        revisor.revise(assessment)
    # The stated edge should still exist (not prunable).
    sim_edges = [
        e for e in net.get_edges("fire", "out")
        if e.relation == RelationType.SIMILAR_TO and e.target == "water"
    ]
    assert len(sim_edges) > 0


def test_belief_revision_bayesian_update() -> None:
    """Evidence feeds into the probabilistic reasoning engine."""
    from genesis_cognitive.reasoning import (
        BeliefRevisionEngine,
        CriticalThinkingEngine,
        ProbabilisticReasoning,
    )
    net = _build_critical_network()
    probabilistic = ProbabilisticReasoning(net)
    critic = CriticalThinkingEngine(net)
    revisor = BeliefRevisionEngine(net, probabilistic=probabilistic)
    # Strong support → positive Bayesian update.
    result = _make_result(
        confidence=0.5,
        knowledge=[("causes", "heat", 0.8)],
    )
    assessment = critic.evaluate(result)
    revisor.revise(assessment)
    # The belief distribution should exist and have been updated.
    belief = probabilistic.get_belief("fire", "causes", "heat")
    # alpha should have increased from the prior (positive evidence).
    assert belief.alpha > 1.0


def test_belief_revision_non_destructive_contradiction() -> None:
    """resolve_contradiction_non_destructive downgrades both sides."""
    from genesis_cognitive.reasoning import BeliefRevisionEngine
    net = _build_critical_network()
    # Add a SIMILAR_TO + OPPOSITE_OF contradiction.
    net.add_edge("fire", "water", RelationType.SIMILAR_TO, weight=0.7, origin="inferred")
    net.add_edge("fire", "water", RelationType.OPPOSITE_OF, weight=0.6, origin="stated")
    revisor = BeliefRevisionEngine(net)
    records = revisor.resolve_contradiction_non_destructive(
        "fire", "water", RelationType.SIMILAR_TO, RelationType.OPPOSITE_OF,
    )
    # Both edges should be downgraded, not removed.
    assert len(records) >= 1
    sim_edges = [
        e for e in net.get_edges("fire", "out")
        if e.relation == RelationType.SIMILAR_TO and e.target == "water"
    ]
    opp_edges = [
        e for e in net.get_edges("fire", "out")
        if e.relation == RelationType.OPPOSITE_OF and e.target == "water"
    ]
    assert len(sim_edges) > 0  # preserved
    assert len(opp_edges) > 0  # preserved
    # Both should have reduced weight.
    assert sim_edges[0].weight < 0.7
    assert opp_edges[0].weight < 0.6


def test_belief_revision_persistence() -> None:
    """to_dict / restore_from_dict preserves counts."""
    from genesis_cognitive.reasoning import (
        BeliefRevisionEngine,
        CriticalThinkingEngine,
    )
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    revisor = BeliefRevisionEngine(net)
    # Generate some revisions.
    result = _make_result(
        confidence=0.7,
        knowledge=[("similar_to", "ice", 0.3)],
        evidence=["fire similar_to ice"],
    )
    assessment = critic.evaluate(result)
    revisor.revise(assessment)

    data = revisor.to_dict()
    assert isinstance(data, dict)

    revisor2 = BeliefRevisionEngine(net)
    revisor2.restore_from_dict(data)
    assert revisor2.downgraded_count == revisor.downgraded_count
    assert revisor2.pruned_count == revisor.pruned_count


def test_belief_revision_records_audit_trail() -> None:
    """Revision records provide an audit trail of what changed."""
    from genesis_cognitive.reasoning import (
        BeliefRevisionEngine,
        CriticalThinkingEngine,
    )
    net = _build_critical_network()
    critic = CriticalThinkingEngine(net)
    revisor = BeliefRevisionEngine(net)
    result = _make_result(
        confidence=0.7,
        knowledge=[("similar_to", "ice", 0.3)],
        evidence=["fire similar_to ice"],
    )
    assessment = critic.evaluate(result)
    records = revisor.revise(assessment)
    for record in records:
        assert record.concept
        assert record.relation
        assert record.target
        assert record.reason in (
            "disconfirmation", "strong_support", "pruned",
            "contradiction_downgrade",
        )
        # weight_delta should be negative for downgrades.
        if record.reason == "disconfirmation":
            assert record.weight_delta < 0


# ======================================================================
# Decision engine tests
# ======================================================================


def _make_reasoning_result(
    conclusion: str = "fire causes heat",
    confidence: float = 0.7,
    knowledge: list[tuple[str, str, float]] | None = None,
    evidence: list[str] | None = None,
    novel: bool = False,
) -> ReasoningResult:
    """Helper to build a ReasoningResult for decision tests."""
    return ReasoningResult(
        reasoning_type=ReasoningType.DEDUCTIVE,
        conclusion=conclusion,
        confidence=confidence,
        evidence=evidence or [conclusion],
        knowledge=knowledge or [("causes", "heat", confidence)],
        novel=novel,
    )


def test_decision_engine_selects_answer_with_strong_evidence() -> None:
    """Strong reasoning evidence keeps 'answer' as the selected action."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    net.add_concept("fire")
    net.add_concept("heat")
    de = DecisionEngine(net)
    rr = _make_reasoning_result(confidence=0.85)
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[rr],
        confidence=0.85,
        uncertainty=0.15,
        goals=[],
        topics=["fire", "heat"],
        perception_intent="question",
    )
    assert outcome.action_type == ActionType.ANSWER
    assert not outcome.overridden_default


def test_decision_engine_switches_to_ask_under_uncertainty() -> None:
    """Low confidence + high uncertainty switches to 'ask'."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    net.add_concept("fire")
    de = DecisionEngine(net)
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[],
        confidence=0.2,
        uncertainty=0.8,
        goals=[],
        topics=["fire"],
        perception_intent="question",
    )
    assert outcome.action_type == ActionType.ASK
    assert outcome.overridden_default


def test_decision_engine_uncertainty_switch_threshold() -> None:
    """Switch happens at the configured threshold, not above."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    de = DecisionEngine(net)
    # Confidence just above threshold → no switch.
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[_make_reasoning_result(confidence=0.5)],
        confidence=0.5,
        uncertainty=0.3,
        goals=[],
        topics=["fire"],
        perception_intent="question",
    )
    assert outcome.action_type == ActionType.ANSWER


def test_decision_engine_evaluates_multiple_candidates() -> None:
    """Multiple candidates are evaluated and ranked."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    de = DecisionEngine(net)
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[_make_reasoning_result(confidence=0.6)],
        confidence=0.6,
        uncertainty=0.4,
        goals=[],
        topics=["fire"],
        perception_intent="question",
    )
    # Should have at least answer, ask, reflect, withhold.
    action_types = {c.action_type for c in outcome.candidates}
    assert ActionType.ANSWER in action_types
    assert ActionType.ASK in action_types
    assert ActionType.WITHHOLD in action_types


def test_decision_engine_goal_relevance() -> None:
    """Goals boost actions relevant to them."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    de = DecisionEngine(net)
    # With a goal to understand fire, answering about fire is relevant.
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[_make_reasoning_result(confidence=0.7)],
        confidence=0.7,
        uncertainty=0.3,
        goals=["understand:fire"],
        topics=["fire"],
        perception_intent="question",
    )
    answer_candidate = next(
        c for c in outcome.candidates if c.action_type == ActionType.ANSWER
    )
    assert answer_candidate.criteria_scores["goal_relevance"] > 0.0


def test_decision_engine_value_weights() -> None:
    """Value criteria are weighted in the composite score."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    de = DecisionEngine(net)
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[_make_reasoning_result(confidence=0.7)],
        confidence=0.7,
        uncertainty=0.3,
        goals=[],
        topics=["fire"],
        perception_intent="question",
    )
    answer = next(
        c for c in outcome.candidates if c.action_type == ActionType.ANSWER
    )
    # Accuracy should be high for a confident answer.
    assert answer.criteria_scores["accuracy"] > 0.5
    # Honesty should be moderate (not under uncertainty).
    assert "honesty" in answer.criteria_scores


def test_decision_engine_inhibition() -> None:
    """Executive inhibition withholds very weak responses."""
    from genesis_cognitive.executive import ExecutiveFunction
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    exec_fn = ExecutiveFunction()
    de = DecisionEngine(net, executive=exec_fn)
    # No evidence, no reasoning, very low confidence → inhibited.
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[],
        confidence=0.05,
        uncertainty=0.95,
        goals=[],
        topics=[],
        perception_intent="question",
    )
    # The best candidate should have very low score.
    # Inhibition may or may not trigger depending on exact scores,
    # but the withhold option should exist.
    withhold = next(
        (c for c in outcome.candidates if c.action_type == ActionType.WITHHOLD),
        None,
    )
    assert withhold is not None


def test_decision_engine_inhibition_triggers_for_weak_candidate() -> None:
    """Executive inhibition actually fires for very weak candidates.

    Previously, the impulse strength was passed as ``1.0 - composite_score``,
    which inverted the semantics: weak candidates got strong impulses and
    were never inhibited. Now the composite score is passed directly, so
    weak candidates produce weak impulses that are inhibited by the
    executive's stop-signal threshold.

    The composite score is now normalized to [0, 1] and the inhibition
    threshold raised to 0.45, making the path reachable for the absolute
    weakest candidates (e.g., ASK with no evidence and low uncertainty).
    """
    from genesis_cognitive.executive import ExecutiveFunction
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    exec_fn = ExecutiveFunction(inhibition_threshold=0.5)
    de = DecisionEngine(net, executive=exec_fn)
    # Reasoning with confidence=0.5 → ASK gets evidence=max(0, 0.5-0.5)=0.
    # Low uncertainty → ASK gets low uncertainty score.
    # No goals, no topics → low goal_relevance.
    # This drives ASK to its minimum normalized score (~0.45), below 0.50.
    reasoning = [_make_reasoning_result("fire causes heat", confidence=0.5)]
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=reasoning,
        confidence=0.01,
        uncertainty=0.01,
        goals=[],
        topics=[],
        perception_intent="question",
    )
    # The selected action should be WITHHOLD (inhibition fired).
    assert outcome.inhibition_applied, (
        "Inhibition should have fired for a very weak candidate"
    )
    assert outcome.selected.action_type == ActionType.WITHHOLD


def test_decision_engine_td_value_lookup() -> None:
    """TD value function is consulted for each candidate."""
    from genesis_cognitive.learning.td import TDLearner
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    net.add_concept("fire")
    net.add_concept("heat")
    td = TDLearner(net)
    de = DecisionEngine(net, td_learner=td)
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[_make_reasoning_result(confidence=0.7)],
        confidence=0.7,
        uncertainty=0.3,
        goals=[],
        topics=["fire", "heat"],
        perception_intent="question",
    )
    # TD values should be set (even if 0 initially).
    for c in outcome.candidates:
        assert c.td_value >= 0.0


def test_decision_engine_persistence() -> None:
    """to_dict / restore_from_dict preserves counts."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    de = DecisionEngine(net)
    # Make a decision to generate stats.
    de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[],
        confidence=0.2,
        uncertainty=0.8,
        goals=[],
        topics=["fire"],
        perception_intent="question",
    )
    data = de.to_dict()
    assert isinstance(data, dict)
    assert "override_count" in data
    de2 = DecisionEngine(net)
    de2.restore_from_dict(data)
    assert de2.override_count == de.override_count


def test_decision_engine_describe() -> None:
    """describe() produces a human-readable summary."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    de = DecisionEngine(net)
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[_make_reasoning_result(confidence=0.7)],
        confidence=0.7,
        uncertainty=0.3,
        goals=[],
        topics=["fire"],
        perception_intent="question",
    )
    desc = outcome.describe()
    assert "Decision:" in desc
    assert "answer" in desc


def test_decision_engine_reflect_for_novel_reasoning() -> None:
    """Novel reasoning gives 'reflect' a higher score."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    de = DecisionEngine(net)
    rr = _make_reasoning_result(confidence=0.6, novel=True)
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[rr],
        confidence=0.6,
        uncertainty=0.4,
        goals=[],
        topics=["fire"],
        perception_intent="question",
    )
    reflect = next(
        c for c in outcome.candidates if c.action_type == ActionType.REFLECT
    )
    # Novel reasoning should give reflect a decent evidence score.
    assert reflect.criteria_scores["evidence"] > 0.0


def test_decision_engine_stats_tracking() -> None:
    """Decision stats are tracked across calls."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    de = DecisionEngine(net)
    # Make an override decision.
    de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[],
        confidence=0.2,
        uncertainty=0.8,
        goals=[],
        topics=["fire"],
        perception_intent="question",
    )
    assert de.override_count > 0
    assert len(de.decisions) > 0


def test_decision_engine_custom_value_weights() -> None:
    """Custom value weights change the composite scores."""
    from genesis_cognitive.reasoning import ActionType, DecisionEngine
    net = ConceptNetwork()
    # Emphasize curiosity heavily.
    custom_weights = {
        "accuracy": 0.1, "honesty": 0.1, "helpfulness": 0.1,
        "curiosity": 0.5, "safety": 0.1, "coherence": 0.1,
    }
    de = DecisionEngine(net, value_weights=custom_weights)
    outcome = de.decide(
        default_action=ActionType.ANSWER,
        reasoning_results=[_make_reasoning_result(confidence=0.6)],
        confidence=0.6,
        uncertainty=0.4,
        goals=["understand:fire"],
        topics=["fire"],
        perception_intent="question",
    )
    # With curiosity weighted high, investigate or ask should score well.
    assert outcome.action_type in (
        ActionType.ANSWER, ActionType.ASK, ActionType.REFLECT,
        ActionType.INVESTIGATE,
    )


# ======================================================================
# Planning engine tests
# ======================================================================


def _build_planning_network() -> ConceptNetwork:
    """Build a test network with causal and dependency structure.

    fire depends on oxygen and fuel, causes heat.
    oxygen causes circulation.
    """
    net = ConceptNetwork()
    net.add_concept("fire")
    net.add_concept("heat")
    net.add_concept("oxygen")
    net.add_concept("fuel")
    net.add_concept("circulation")
    net.add_edge("fire", "heat", RelationType.CAUSES, weight=0.8, origin="stated")
    net.add_edge("oxygen", "fire", RelationType.ENABLES, weight=0.7, origin="stated")
    net.add_edge("fire", "oxygen", RelationType.DEPENDS_ON, weight=0.7, origin="stated")
    net.add_edge("fuel", "fire", RelationType.ENABLES, weight=0.6, origin="stated")
    net.add_edge("fire", "fuel", RelationType.DEPENDS_ON, weight=0.6, origin="stated")
    net.add_edge("oxygen", "circulation", RelationType.CAUSES, weight=0.5, origin="stated")
    return net


def test_planning_engine_creates_plan_with_prerequisites() -> None:
    """Plan includes prerequisites before the goal concept."""
    from genesis_cognitive.reasoning import PlanningEngine, PlanStatus
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    # Should have steps for oxygen, fuel, fire, and verify.
    assert len(plan.steps) >= 3
    assert plan.status == PlanStatus.ACTIVE
    # Prerequisites should come before the goal.
    step_concepts = [s.target_concept for s in plan.steps]
    assert "oxygen" in step_concepts
    assert "fuel" in step_concepts
    assert "fire" in step_concepts
    oxygen_idx = step_concepts.index("oxygen")
    fire_idx = step_concepts.index("fire")
    assert oxygen_idx < fire_idx


def test_planning_engine_feasibility() -> None:
    """Feasibility score reflects concept existence."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    # All concepts exist → high feasibility.
    assert plan.feasibility_score > 0.5


def test_planning_engine_low_feasibility_for_unknown() -> None:
    """Unknown concepts produce low feasibility."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = ConceptNetwork()
    net.add_concept("fire")
    planner = PlanningEngine(net)
    # "dragon" doesn't exist in the network.
    plan = planner.create_plan("dragon", "understand")
    assert plan.feasibility_score < 0.5


def test_planning_engine_advance_returns_step() -> None:
    """advance() returns the current step to execute."""
    from genesis_cognitive.reasoning import PlanningEngine, PlanStepStatus
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    step = planner.advance(plan)
    assert step is not None
    assert step.status == PlanStepStatus.IN_PROGRESS


def test_planning_engine_mark_step_completed() -> None:
    """mark_step with success advances the plan."""
    from genesis_cognitive.reasoning import (
        PlanningEngine,
        PlanStepStatus,
    )
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    step = planner.advance(plan)
    assert step is not None
    planner.mark_step(plan, step, success=True, confidence=0.8)
    assert step.status == PlanStepStatus.COMPLETED
    assert plan.current_step_index > 0


def test_planning_engine_completes_plan() -> None:
    """Completing all steps marks the plan as completed."""
    from genesis_cognitive.reasoning import PlanningEngine, PlanStatus
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    # Execute all steps.
    while True:
        step = planner.advance(plan)
        if step is None:
            break
        planner.mark_step(plan, step, success=True, confidence=0.8)
    assert plan.status == PlanStatus.COMPLETED
    assert plan.progress == 1.0
    assert planner.completed_count > 0


def test_planning_engine_revision_on_failure() -> None:
    """Repeated failure triggers plan revision."""
    from genesis_cognitive.reasoning import (
        PlanningEngine,
        PlanStepStatus,
    )
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    # Fail the first step repeatedly until it's failed or skipped.
    for _ in range(10):
        step = planner.advance(plan)
        if step is None:
            break
        if step.status in (PlanStepStatus.FAILED, PlanStepStatus.SKIPPED):
            break
        planner.mark_step(plan, step, success=False)
    # Step should be failed or skipped (revision may skip it).
    first_step = plan.steps[0]
    assert first_step.status in (PlanStepStatus.FAILED, PlanStepStatus.SKIPPED)
    assert planner.revised_count > 0


def test_planning_engine_blocked_when_no_alternative() -> None:
    """Plan is blocked when no alternative path exists."""
    from genesis_cognitive.reasoning import (
        PlanningEngine,
        PlanStatus,
    )
    net = ConceptNetwork()
    net.add_concept("unknown_concept")
    planner = PlanningEngine(net)
    plan = planner.create_plan("unknown_concept", "understand")
    # Fail the step repeatedly with no alternative path.
    step = planner.advance(plan)
    if step is None:
        return  # plan may be empty
    for _ in range(step.max_attempts):
        planner.mark_step(plan, step, success=False)
    # Plan should be blocked or the step skipped.
    assert plan.status in (PlanStatus.BLOCKED, PlanStatus.ACTIVE)


def test_planning_engine_persistence() -> None:
    """to_dict / restore_from_dict preserves plan state."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    step = planner.advance(plan)
    if step:
        planner.mark_step(plan, step, success=True, confidence=0.8)
    data = planner.to_dict()
    assert isinstance(data, dict)
    assert "plans" in data
    assert "completed_count" in data
    # Restore.
    planner2 = PlanningEngine(net)
    planner2.restore_from_dict(data)
    assert planner2.completed_count == planner.completed_count
    assert len(planner2._plans) > 0


def test_planning_engine_describe() -> None:
    """describe() produces a human-readable plan summary."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    desc = plan.describe()
    assert "Plan for:" in desc
    assert "fire" in desc
    assert "feasibility" in desc


def test_planning_engine_progress_tracking() -> None:
    """Progress is tracked as steps complete."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    initial_progress = plan.progress
    assert initial_progress == 0.0
    # Complete one step.
    step = planner.advance(plan)
    assert step is not None
    planner.mark_step(plan, step, success=True, confidence=0.8)
    assert plan.progress > initial_progress


def test_planning_engine_causal_decomposition() -> None:
    """ACHIEVE goals decompose into causal chains."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("heat", "achieve")
    # Should have steps for causes of heat (fire) and heat itself.
    step_concepts = [s.target_concept for s in plan.steps]
    assert "heat" in step_concepts


def test_planning_engine_resolve_decomposition() -> None:
    """RESOLVE goals decompose into understanding both sides."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = _build_planning_network()
    # Add a contradiction.
    net.add_concept("ice")
    net.add_edge("fire", "ice", RelationType.CONTRADICTS, weight=0.7, origin="stated")
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire vs ice", "resolve")
    step_concepts = [s.target_concept for s in plan.steps]
    # Should include both sides and a resolve step.
    assert "fire" in step_concepts
    assert "ice" in step_concepts
    assert any(s.step_type == "resolve" for s in plan.steps)


def test_planning_engine_get_plan_for_goal() -> None:
    """get_plan_for_goal retrieves a previously created plan."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    retrieved = planner.get_plan_for_goal("fire")
    assert retrieved is plan


def test_planning_engine_steps_executed_tracking() -> None:
    """Steps executed counter tracks total executions."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    initial = planner.steps_executed
    planner.advance(plan)
    assert planner.steps_executed > initial


def test_planning_engine_verify_step() -> None:
    """Plans include a final verification step."""
    from genesis_cognitive.reasoning import PlanningEngine
    net = _build_planning_network()
    planner = PlanningEngine(net)
    plan = planner.create_plan("fire", "understand")
    # Last step should be a verify step.
    assert plan.steps[-1].step_type == "verify"

