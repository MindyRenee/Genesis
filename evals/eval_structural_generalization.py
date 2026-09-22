# ruff: isort: skip_file

"""Evaluation: blind structural generalization on unseen synthetic concepts.

This evaluation is deliberately different from the hand-authored
generalization eval. It generates arbitrary concept names at runtime,
teaches only pairwise PART_OF facts, and computes the expected closure
from the generated graph itself.

A passing result therefore cannot come from memorizing words such as
"tardigrade", "cortex", or "brain". The evaluator knows only the graph
structure it generated.

Design:
- One independent world per condition — a minimal one-shot probe of
  whether she can close a structure she has never seen. Raise
  _WORLD_COUNT for repeated trials.
- Each world is a 6-node PART_OF chain using random synthetic tokens.
- Genesis receives only adjacent edges.
- The evaluator asks for a non-adjacent edge (4 hops apart).
- A distractor edge forms a disconnected component that closure must
  not bridge (the negative control).
- Inferred edges are required to have origin="inferred".
- A configurable cooldown separates worlds when _WORLD_COUNT > 1.

This is evidence for structural generalization, not a general-intelligence
or consciousness test.
"""

from __future__ import annotations

import argparse
import random
import string
import time
from itertools import pairwise

from harness import (
    EvalMind,
    EvalResult,
    FactResult,
    RelationType,
    print_result,
    run_condition,
)


_WORLD_COUNT = 1
_CHAIN_LENGTH = 6
_QUERY_HOPS = 4
_DEFAULT_COOLDOWN_SECONDS = 1.0


def _token(rng: random.Random, kind: str, world: int, node: int) -> str:
    """Create an opaque concept token with no semantic content.

    Tokens must be single lowercase words: her fact extractor only
    recognizes alphabetic concept names, and a trailing "s" would be
    singularized away by concept normalization. They contain no real
    morphemes — kind ("n"/"d") and world/node position are encoded as
    letters purely so failures stay debuggable; the random suffix keeps
    each token unique and meaningless to her.
    """
    letters = string.ascii_lowercase
    position = f"{letters[world // 26]}{letters[world % 26]}{letters[node]}"
    suffix = "".join(rng.choice(letters) for _ in range(7))
    suffix += rng.choice(letters.replace("s", ""))
    return f"{kind}{position}{suffix}"


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
    em: EvalMind,
    seed: int,
    cooldown_seconds: float,
) -> list[FactResult]:
    """Generate and evaluate opaque graph worlds one at a time."""
    rng = random.Random(seed)

    all_results: list[FactResult] = []
    total_inferred = 0

    for world_index in range(_WORLD_COUNT):
        nodes = [
            _token(rng, "n", world_index, node_index)
            for node_index in range(_CHAIN_LENGTH)
        ]

        # Independent distractor component. It is never connected to the
        # chain and therefore must not create a path to the queried target.
        distractor = [
            _token(rng, "d", world_index, node_index)
            for node_index in range(2)
        ]

        # Blindness precondition: every generated token must be absent
        # from her network — otherwise the trial isn't "never seen".
        unseen = all(
            em.network.get_concept(t) is None for t in (*nodes, *distractor)
        )
        all_results.append(
            FactResult(
                concept=nodes[0],
                passed=unseen,
                detail=(
                    f"world {world_index + 1}: all generated tokens "
                    f"previously unseen={unseen}"
                ),
            )
        )

        for left, right in pairwise(nodes):
            _teach_part_of(em, left, right)
        _teach_part_of(em, distractor[0], distractor[1])

        inferred = _run_closure(em)
        total_inferred += inferred

        source = nodes[0]
        target = nodes[_QUERY_HOPS]
        exists = _edge_exists(em, source, target)
        origin = _edge_origin(em, source, target)
        passed = exists and origin == "inferred"

        positive = FactResult(
            concept=source,
            passed=passed,
            detail=(
                f"world {world_index + 1}: blind {_QUERY_HOPS}-hop PART_OF "
                f"closure to '{target}': exists={exists}, origin={origin!r}"
            ),
        )
        all_results.append(positive)

        # Negative control: the distractor is a disconnected component —
        # closure must never bridge from the chain into it.
        negative_exists = _edge_exists(em, source, distractor[1])
        all_results.append(
            FactResult(
                concept=source,
                passed=not negative_exists,
                detail=(
                    f"world {world_index + 1}: distractor negative control "
                    f"to '{distractor[1]}': edge_exists={negative_exists}"
                ),
            )
        )

        # Structural audit: the direct edge supplied by this evaluator must
        # exist as a taught ("stated") edge — a check that also proves the
        # teaching landed instead of passing vacuously on a missing edge.
        direct_target = nodes[1]
        direct_origin = _edge_origin(em, source, direct_target)
        all_results.append(
            FactResult(
                concept=source,
                passed=direct_origin == "stated",
                detail=(
                    f"world {world_index + 1}: direct taught edge "
                    f"origin={direct_origin!r}"
                ),
            )
        )

        if cooldown_seconds > 0 and world_index < _WORLD_COUNT - 1:
            print(
                f"      Cooldown: {cooldown_seconds:.1f}s before the next world."
            )
            time.sleep(cooldown_seconds)

    # Diagnostic only: do not let inference volume determine the score.
    all_results.append(
        FactResult(
            concept="evaluation",
            passed=total_inferred > 0,
            detail=f"inference cycle produced {total_inferred} new edges",
        )
    )

    return all_results


def run(seed: int = 42, cooldown_seconds: float = _DEFAULT_COOLDOWN_SECONDS) -> EvalResult:
    """Run the blind randomized structural generalization evaluation."""
    start = time.time()
    result = EvalResult(
        name="Blind Randomized Structural Generalization",
        description=(
            "Tests whether Genesis can learn and transitively close unseen "
            "graph structures built from opaque, randomly generated concept "
            "names."
        ),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    with EvalMind(seed=seed) as em:
        condition = run_condition(
            "Opaque synthetic worlds",
            (
                f"{_WORLD_COUNT} independent {_CHAIN_LENGTH}-node PART_OF "
                f"chains; process one world at a time; test {_QUERY_HOPS}-hop "
                "inferred edges, distractor negative controls, and direct "
                "edge provenance."
            ),
            lambda em=em: _condition_randomized_structural_generalization(
                em, seed, cooldown_seconds
            ),
        )
    result.conditions.append(condition)
    result.overall_accuracy = condition.accuracy
    result.duration_seconds = round(time.time() - start, 3)
    return result


def main() -> None:
    """Run the evaluation from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cooldown",
        type=float,
        default=_DEFAULT_COOLDOWN_SECONDS,
        help="Seconds between worlds; use 0 to disable.",
    )
    args = parser.parse_args()
    if args.cooldown < 0:
        parser.error("--cooldown must be >= 0")
    print_result(run(seed=args.seed, cooldown_seconds=args.cooldown))


if __name__ == "__main__":
    main()
