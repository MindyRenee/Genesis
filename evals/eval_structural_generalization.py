"""Evaluation: blind structural generalization on unseen synthetic concepts.

This evaluation is deliberately different from the hand-authored
generalization eval. It generates arbitrary concept names at runtime,
teaches only pairwise PART_OF facts, and computes the expected closure
from the generated graph itself.

A passing result therefore cannot come from memorizing words such as
"tardigrade", "cortex", or "brain". The evaluator knows only the graph
structure it generated.

Design:
- 50 independent worlds per condition.
- Each world is a 6-node PART_OF chain using random synthetic tokens.
- Genesis receives only adjacent edges.
- The evaluator asks for a non-adjacent edge (4 hops apart).
- Distractor edges form separate components and must not bridge worlds.
- Inferred edges are required to have origin="inferred".
- A negative control checks that unrelated endpoints are not connected.

This is evidence for structural generalization, not a general-intelligence
or consciousness test.
"""

from __future__ import annotations

import random
import string
import time

from harness import EvalMind, EvalResult, FactResult, RelationType, print_result, run_condition


_WORLD_COUNT = 50
_CHAIN_LENGTH = 6
_QUERY_HOPS = 4


def _token(rng: random.Random, prefix: str, world: int, node: int) -> str:
    """Create an opaque concept token with no semantic content."""
    suffix = "".join(rng.choice(string.ascii_lowercase) for _ in range(8))
    return f"{prefix}_{world}_{node}_{suffix}"


def _teach_part_of(em: EvalMind, source: str, target: str) -> None:
    """Teach a relation using the same natural-language form used elsewhere."""
    em.teach(f"The {source} is part of the {target}.")


def _edge_exists(em: EvalMind, source: str, target: str) -> bool:
    """Check for the exact relation, not merely any edge."""
    return any(
        edge.relation == RelationType.PART_OF and edge.target == target
        for edge in em.network.get_edges(source, direction="out")
    )


def _edge_origin(em: EvalMind, source: str, target: str) -> str | None:
    """Return the origin of the exact relation if present."""
    for edge in em.network.get_edges(source, direction="out"):
        if edge.relation == RelationType.PART_OF and edge.target == target:
            return edge.origin
    return None


def _run_closure(em: EvalMind, max_cycles: int = 30) -> int:
    """Run inference until no new edges are produced or the safety cap is hit."""
    total_new = 0
    for _ in range(max_cycles):
        result = em.mind.cognition.self_learner.run_inference_cycle(force=True)
        total_new += result.new_edges
        if result.new_edges == 0:
            break
    return total_new


def _condition_randomized_structural_generalization(
    em: EvalMind, seed: int
) -> list[FactResult]:
    """Generate and evaluate opaque graph worlds."""
    rng = random.Random(seed)

    worlds: list[list[str]] = []
    for world_index in range(_WORLD_COUNT):
        nodes = [
            _token(rng, "node", world_index, node_index)
            for node_index in range(_CHAIN_LENGTH)
        ]
        worlds.append(nodes)

        # Independent distractor component. It is never connected to the
        # chain and therefore must not create a path to the queried target.
        distractor = [
            _token(rng, "distractor", world_index, node_index)
            for node_index in range(2)
        ]

        for left, right in zip(nodes, nodes[1:]):
            _teach_part_of(em, left, right)

        _teach_part_of(em, distractor[0], distractor[1])

    inferred = _run_closure(em)

    results: list[FactResult] = []

    # Positive tests: four-hop relationships were never directly taught.
    for world_index, nodes in enumerate(worlds):
        source = nodes[0]
        target = nodes[_QUERY_HOPS]
        exists = _edge_exists(em, source, target)
        origin = _edge_origin(em, source, target)

        results.append(
            FactResult(
                concept=source,
                passed=exists and origin == "inferred",
                detail=(
                    f"blind {_QUERY_HOPS}-hop PART_OF closure to '{target}': "
                    f"exists={exists}, origin={origin!r}"
                ),
            )
        )

    # Negative tests: unrelated endpoints must remain disconnected.
    for world_index, nodes in enumerate(worlds):
        source = nodes[0]
        other_world = worlds[(world_index + 1) % len(worlds)]
        target = other_world[_QUERY_HOPS]
        exists = _edge_exists(em, source, target)

        results.append(
            FactResult(
                concept=source,
                passed=not exists,
                detail=(
                    f"cross-world negative control to '{target}': "
                    f"edge_exists={exists}"
                ),
            )
        )

    # Structural audit: every direct edge was supplied by the evaluator,
    # while the queried long-range edge must be an inference artifact.
    for world_index, nodes in enumerate(worlds):
        direct_target = nodes[1]
        origin = _edge_origin(em, nodes[0], direct_target)
        results.append(
            FactResult(
                concept=nodes[0],
                passed=origin != "inferred",
                detail=f"direct taught edge origin={origin!r}",
            )
        )

    # Report inference volume as a diagnostic without using it as the score.
    results.append(
        FactResult(
            concept="evaluation",
            passed=inferred > 0,
            detail=f"inference cycle produced {inferred} new edges",
        )
    )

    return results


def run(seed: int = 42) -> EvalResult:
    """Run the blind randomized structural generalization evaluation."""
    start = time.time()
    result = EvalResult(
        name="Blind Randomized Structural Generalization",
        description=(
            "Tests whether Genesis can learn and transitively close "
            "unseen graph structures built from opaque, randomly generated "
            "concept names. No semantic meaning of the names is supplied."
        ),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    with EvalMind(seed=seed) as em:
        condition = run_condition(
            "Opaque synthetic worlds",
            (
                f"{_WORLD_COUNT} independent {_CHAIN_LENGTH}-node PART_OF "
                f"chains; test {_QUERY_HOPS}-hop inferred edges and "
                "cross-world negative controls."
            ),
            lambda em=em: _condition_randomized_structural_generalization(
                em, seed
            ),
        )
    result.conditions.append(condition)
    result.overall_accuracy = condition.accuracy
    result.duration_seconds = round(time.time() - start, 3)
    return result


if __name__ == "__main__":
    print_result(run())
