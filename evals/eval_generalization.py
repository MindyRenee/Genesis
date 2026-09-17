"""Evaluation 2: Generalization (Transitive Inference).

Tests whether Genesis can infer new relationships from taught facts.
This is the next level up from basic teaching/recall — it tests
whether the inference cycle actually produces new knowledge.

Conditions:
1. Transitive IS_A hierarchy — teach "A is_a B", "B is_a C", check
   if "A is_a C" is inferred after the inference cycle.
2. Transitive PART_OF hierarchy — teach "X is_part_of Y", "Y is_part_of Z",
   check if "X is_part_of Z" is inferred.
3. Multi-hop inference — teach a 4-level hierarchy, check if the
   inference engine closes the full chain (A→B→C→D implies A→D).
4. No false transitivity — teach non-transitive relations (CAUSES,
   RELATED_TO), verify the inference engine does NOT infer
   spurious edges.

The key distinction from the teaching eval: here we check edges that
were NEVER directly taught — they should emerge from the inference
cycle's transitive closure. A passing result means Genesis is
genuinely reasoning, not just recording.
"""

from __future__ import annotations

import time

from genesis_cognitive.concepts import RelationType
from harness import (
    EvalMind,
    EvalResult,
    Fact,
    FactResult,
    print_result,
    run_condition,
)

# ─── Test hierarchies ─────────────────────────────────────────────

# IS_A hierarchy: tardigrade → micro-animal → animal
# Teach: "A tardigrade is a micro-animal." and "A micro-animal is an animal."
# Check: tardigrade IS_A animal (inferred, never taught)
_IS_A_FACTS = [
    Fact(
        teach_input="A tardigrade is a microscopic animal.",
        concept="tardigrade",
        expected_substrings=("microscopic", "animal"),
        relation_type=RelationType.IS_A,
        relation_target="animal",
    ),
    Fact(
        teach_input="A microscopic animal is a kind of animal.",
        concept="microscopic animal",
        expected_substrings=("animal",),
        relation_type=RelationType.IS_A,
        relation_target="animal",
    ),
]

# The inferred edge we check for:
_IS_A_INFERRED = Fact(
    teach_input="",  # never taught
    concept="tardigrade",
    expected_substrings=(),  # definition check not needed
    relation_type=RelationType.IS_A,
    relation_target="animal",
)


# PART_OF hierarchy: cortex → cerebrum → brain
# Teach: "The cortex is part of the cerebrum." and "The cerebrum is part of the brain."
# Check: cortex PART_OF brain (inferred)
_PART_OF_FACTS = [
    Fact(
        teach_input="The cortex is part of the cerebrum.",
        concept="cortex",
        expected_substrings=("cerebrum",),
        relation_type=RelationType.PART_OF,
        relation_target="cerebrum",
    ),
    Fact(
        teach_input="The cerebrum is part of the brain.",
        concept="cerebrum",
        expected_substrings=("brain",),
        relation_type=RelationType.PART_OF,
        relation_target="brain",
    ),
]

_PART_OF_INFERRED = Fact(
    teach_input="",
    concept="cortex",
    expected_substrings=(),
    relation_type=RelationType.PART_OF,
    relation_target="brain",
)


# Multi-hop: hippocampus → temporal lobe → cerebrum → brain
# Teach 3 edges, check if the inference engine closes the full chain
_MULTI_HOP_FACTS = [
    Fact(
        teach_input="The hippocampus is part of the temporal lobe.",
        concept="hippocampus",
        expected_substrings=("temporal", "lobe"),
        relation_type=RelationType.PART_OF,
        relation_target="temporal lobe",
    ),
    Fact(
        teach_input="The temporal lobe is part of the cerebrum.",
        concept="temporal lobe",
        expected_substrings=("cerebrum",),
        relation_type=RelationType.PART_OF,
        relation_target="cerebrum",
    ),
    Fact(
        teach_input="The cerebrum is part of the brain.",
        concept="cerebrum",
        expected_substrings=("brain",),
        relation_type=RelationType.PART_OF,
        relation_target="brain",
    ),
]

_MULTI_HOP_INFERRED = Fact(
    teach_input="",
    concept="hippocampus",
    expected_substrings=(),
    relation_type=RelationType.PART_OF,
    relation_target="brain",
)


# Non-transitive: CAUSES should NOT be transitively closed
# Teach: "Cortisol causes stress." and "Stress causes insomnia."
# Check: cortisol CAUSES insomnia should NOT exist
_NON_TRANSITIVE_FACTS = [
    Fact(
        teach_input="Cortisol causes stress.",
        concept="cortisol",
        expected_substrings=("stress",),
        relation_type=RelationType.CAUSES,
        relation_target="stress",
    ),
    Fact(
        teach_input="Stress causes insomnia.",
        concept="stress",
        expected_substrings=("insomnia",),
        relation_type=RelationType.CAUSES,
        relation_target="insomnia",
    ),
]

_NON_TRANSITIVE_SHOULD_NOT_EXIST = Fact(
    teach_input="",
    concept="cortisol",
    expected_substrings=(),
    relation_type=RelationType.CAUSES,
    relation_target="insomnia",
)


# ─── Conditions ───────────────────────────────────────────────────


def _check_inferred_edge(em: EvalMind, fact: Fact) -> FactResult:
    """Check if a specific edge exists (regardless of how it was created)."""
    for edge in em.network.get_edges(fact.concept, direction="out"):
        if edge.relation == fact.relation_type and edge.target == fact.relation_target:
            return FactResult(
                concept=fact.concept, passed=True,
                detail=f"{fact.relation_type.value} edge to '{fact.relation_target}' exists",
            )
    return FactResult(
        concept=fact.concept, passed=False,
        detail=f"no {fact.relation_type.value} edge to '{fact.relation_target}'",
    )


def _check_no_spurious_edge(em: EvalMind, fact: Fact) -> FactResult:
    """Check that a specific edge does NOT exist (negative test)."""
    for edge in em.network.get_edges(fact.concept, direction="out"):
        if edge.relation == fact.relation_type and edge.target == fact.relation_target:
            return FactResult(
                concept=fact.concept, passed=False,
                detail=(
                    f"spurious {fact.relation_type.value} edge to "
                    f"'{fact.relation_target}' should not exist"
                ),
            )
    return FactResult(
        concept=fact.concept, passed=True,
        detail=f"correctly no {fact.relation_type.value} edge to '{fact.relation_target}'",
    )


def _check_taught_edges(em: EvalMind, facts: list[Fact]) -> list[FactResult]:
    """Check that the directly-taught edges exist (not definitions)."""
    results: list[FactResult] = []
    for fact in facts:
        if fact.relation_type is not None and fact.relation_target is not None:
            results.append(_check_inferred_edge(em, fact))
        else:
            # No relation to check — just verify the concept exists
            concept = em.network.get_concept(fact.concept)
            if concept is not None:
                results.append(FactResult(
                    concept=fact.concept, passed=True,
                    detail=f"concept '{fact.concept}' exists",
                ))
            else:
                results.append(FactResult(
                    concept=fact.concept, passed=False,
                    detail=f"concept '{fact.concept}' not found",
                ))
    return results


def _condition_transitive_is_a(em: EvalMind) -> list[FactResult]:
    """Helper: transitive is a."""
    em.teach_all(_IS_A_FACTS)
    em.mind.cognition.self_learner.run_inference_cycle(force=True)
    taught = _check_taught_edges(em, _IS_A_FACTS)
    inferred = _check_inferred_edge(em, _IS_A_INFERRED)
    return [*taught, inferred]


def _condition_transitive_part_of(em: EvalMind) -> list[FactResult]:
    """Helper: transitive part of."""
    em.teach_all(_PART_OF_FACTS)
    em.mind.cognition.self_learner.run_inference_cycle(force=True)
    taught = _check_taught_edges(em, _PART_OF_FACTS)
    inferred = _check_inferred_edge(em, _PART_OF_INFERRED)
    return [*taught, inferred]


def _condition_multi_hop(em: EvalMind) -> list[FactResult]:
    """Helper: multi hop."""
    em.teach_all(_MULTI_HOP_FACTS)
    em.mind.cognition.self_learner.run_inference_cycle(force=True)
    taught = _check_taught_edges(em, _MULTI_HOP_FACTS)
    inferred = _check_inferred_edge(em, _MULTI_HOP_INFERRED)
    return [*taught, inferred]


def _condition_no_false_transitivity(em: EvalMind) -> list[FactResult]:
    """Helper: no false transitivity."""
    em.teach_all(_NON_TRANSITIVE_FACTS)
    em.mind.cognition.self_learner.run_inference_cycle(force=True)
    taught = _check_taught_edges(em, _NON_TRANSITIVE_FACTS)
    no_spurious = _check_no_spurious_edge(em, _NON_TRANSITIVE_SHOULD_NOT_EXIST)
    return [*taught, no_spurious]


# ─── Eval entry point ─────────────────────────────────────────────


def run(seed: int = 42) -> EvalResult:
    """Run the evaluation."""
    start = time.time()
    result = EvalResult(
        name="Generalization (Transitive Inference)",
        description=(
            "Can Genesis infer new relationships from taught facts? "
            "Tests the inference cycle's transitive closure: if A→B "
            "and B→C are taught, does A→C emerge? Also verifies "
            "non-transitive relations are NOT falsely closed."
        ),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    conditions = [
        ("Transitive IS_A",
         "Teach tardigrade→micro-animal, micro-animal→animal. Check tardigrade→animal inferred.",
         _condition_transitive_is_a),
        ("Transitive PART_OF",
         "Teach cortex→cerebrum, cerebrum→brain. Check cortex→brain inferred.",
         _condition_transitive_part_of),
        ("Multi-hop PART_OF",
         "Teach 3-level chain: hippocampus→temporal lobe→cerebrum→brain. Check hippocampus→brain.",
         _condition_multi_hop),
        ("No False Transitivity",
         "Teach cortisol→stress, stress→insomnia. Verify cortisol→insomnia NOT inferred.",
         _condition_no_false_transitivity),
    ]

    all_accuracies: list[float] = []
    for name, desc, fn in conditions:
        with EvalMind(seed=seed) as em:
            cond = run_condition(name, desc, lambda em=em, fn=fn: fn(em))
        result.conditions.append(cond)
        all_accuracies.append(cond.accuracy)

    result.overall_accuracy = (
        sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
    )
    result.duration_seconds = round(time.time() - start, 3)
    return result


if __name__ == "__main__":
    res = run()
    print_result(res)
