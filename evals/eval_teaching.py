"""Evaluation 1: Teaching and Recall.

Tests the most fundamental cognitive capability: can Genesis learn a
novel fact from a teaching statement and recall it correctly?

This is the foundation. If she can't learn and recall, nothing else
matters. The eval drives the real learning path (learn_from_input +
run_inference_cycle) and checks recall through both the concept
network (structural) and the reasoning engine (functional).

Conditions:
1. Immediate recall — teach facts, check each is in the network.
2. Reasoning recall — teach facts, ask the reasoning engine "what
   is X?" and check the answer contains the taught content.
3. Delayed recall after interference — teach facts, inject unrelated
   fillers, check originals survive.
4. Persistence across save/load — teach, save, load, check facts
   survive the round-trip.

The facts are deliberately novel — not in the initial concept network
seeds — so a passing result means Genesis actually learned them, not
that they were preloaded.
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

# ─── Test facts ────────────────────────────────────────────────────
# Novel facts not in the initial concept network seeds.
# Each is (teach_input, concept, expected_definition_substrings,
#           relation_type, relation_target, question, question_type,
#           expected_answer_substrings)

_FACTS: list[Fact] = [
    Fact(
        teach_input="A tardigrade is a microscopic animal.",
        concept="tardigrade",
        expected_substrings=("microscopic", "animal"),
        relation_type=RelationType.IS_A,
        relation_target="animal",
        question="What is a tardigrade?",
        question_type="what_is",
        expected_answer_substrings=("tardigrade",),
    ),
    Fact(
        teach_input="Cryptobiosis is a state of suspended animation.",
        concept="cryptobiosis",
        expected_substrings=("suspended", "animation"),
    ),
    Fact(
        teach_input="A mycelium is a network of fungal threads.",
        concept="mycelium",
        expected_substrings=("network", "fungal"),
    ),
    Fact(
        teach_input="Photosynthesis is a process that converts light into energy.",
        concept="photosynthesis",
        expected_substrings=("light", "energy"),
    ),
    Fact(
        teach_input="A star is a luminous sphere of plasma.",
        concept="star",
        expected_substrings=("luminous", "plasma"),
    ),
    Fact(
        teach_input="A quasar is an extremely luminous active galactic nucleus.",
        concept="quasar",
        expected_substrings=("luminous", "galactic"),
    ),
    Fact(
        teach_input="A lichen is a symbiotic organism of fungi and algae.",
        concept="lichen",
        expected_substrings=("symbiotic", "organism"),
    ),
    Fact(
        teach_input="A prism is a transparent optical element that refracts light.",
        concept="prism",
        expected_substrings=("transparent", "light"),
    ),
]

# Filler facts for interference testing — unrelated to the test facts.
_FILLERS: list[str] = [
    "A guitar is a stringed musical instrument.",
    "The Amazon is a river in South America.",
    "A glacier is a slow-moving mass of ice.",
    "Quartz is a hard crystalline mineral.",
    "A savanna is a grassland with scattered trees.",
    "The violin is a stringed instrument played with a bow.",
    "A canyon is a deep gorge carved by a river.",
    "Granite is a coarse-grained igneous rock.",
]


# ─── Conditions ───────────────────────────────────────────────────


def _condition_immediate_recall(em: EvalMind) -> list[FactResult]:
    """Helper: immediate recall."""
    em.teach_all(_FACTS)
    return em.check_all(_FACTS)


def _condition_reasoning_recall(em: EvalMind) -> list[FactResult]:
    """Helper: reasoning recall."""
    em.teach_all(_FACTS)
    results: list[FactResult] = []
    for fact in _FACTS:
        if fact.question_type:
            results.append(em.answer(fact))
        else:
            results.append(em.check_fact(fact))
    return results


def _condition_interference(em: EvalMind) -> list[FactResult]:
    """Helper: interference."""
    em.teach_all(_FACTS)
    # Inject interference
    em.teach_fillers(_FILLERS)
    # Check delayed recall
    delayed = em.check_all(_FACTS)
    return delayed


def _condition_persistence(em: EvalMind) -> list[FactResult]:
    """Helper: persistence."""
    em.teach_all(_FACTS)
    em.save()
    em.load()
    return em.check_all(_FACTS)


# ─── Eval entry point ─────────────────────────────────────────────


def run(seed: int = 42) -> EvalResult:
    """Run the teaching and recall evaluation."""
    start = time.time()
    result = EvalResult(
        name="Teaching and Recall",
        description=(
            "Can Genesis learn novel facts from teaching statements "
            "and recall them correctly? Tests the foundational "
            "learning path: learn_from_input → inference cycle → "
            "concept network → reasoning engine."
        ),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Each condition gets a fresh mind so they don't interfere.
    conditions = [
        ("Immediate Recall",
         "Teach facts, check each is in the network.",
         _condition_immediate_recall),
        ("Reasoning Recall",
         "Teach facts, ask the reasoning engine 'what is X?'.",
         _condition_reasoning_recall),
        ("Interference",
         "Teach facts, inject unrelated fillers, check originals survive.",
         _condition_interference),
        ("Persistence",
         "Teach facts, save to disk, reload, check facts survived.",
         _condition_persistence),
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
