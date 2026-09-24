"""Reasoning engine — how Genesis draws conclusions.

This is not pattern matching. It's graph traversal over the concept
network: following relationships, combining facts, detecting
contradictions, and forming hypotheses.

# Types of reasoning

1. **Deductive**: A is_a B, B is_a C → A is_a C (transitivity)
2. **Abductive**: A causes B, B observed → maybe A (inference to best explanation)
3. **Analogical**: A similar_to B, A has property P → B might have P
4. **Causal**: A causes B, B causes C → A leads_to C (chain)
5. **Contradiction detection**: A contradicts B, both stated → flag conflict
6. **Hypothesis formation**: gap in graph → propose edge to fill it

Each reasoning pass produces a ReasoningResult with the conclusion,
the chain of evidence, and a confidence score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from ..concepts import (
    ConceptNetwork,
    RelationType,
)

__all__ = [
    "BetaDistribution",
    "CounterfactualReasoning",
    "MetaReasoning",
    "ProbabilisticReasoning",
    "ReasoningEngine",
    "ReasoningRecord",
    "ReasoningResult",
    "ReasoningStrategy",
    "ReasoningType",
    "TemporalReasoning",
    "TemporalRelation",
    "TimeInterval",
]


class ReasoningType(Enum):
    """What kind of reasoning produced this conclusion."""

    DEDUCTIVE = "deductive"  # transitive chain
    ABDUCTIVE = "abductive"  # inference to best explanation
    ANALOGICAL = "analogical"  # transfer by similarity
    CAUSAL = "causal"  # cause-effect chain
    CONTRADICTION = "contradiction"  # conflict detected
    HYPOTHESIS = "hypothesis"  # proposed new connection
    SYNTHESIS = "synthesis"  # combining multiple sources


@dataclass(slots=True)
class ReasoningResult:
    """A conclusion reached through reasoning.

    The evidence chain is the list of edges traversed to reach the
    conclusion. This makes reasoning transparent — you can always
    see *why* Genesis concluded something.

    The ``knowledge`` field carries structured (relation, target,
    weight) triples that the language engine can compose from —
    prefer passing these as ``Thought.metadata["knowledge"]`` so
    Genesis composes her own words rather than reciting the
    pre-composed ``conclusion`` string.
    """

    conclusion: str  # human-readable conclusion (semantic seed)
    reasoning_type: ReasoningType
    evidence: list[str]  # chain of concept→relation→concept
    confidence: float  # 0..1
    novel: bool = False  # is this a new insight?
    contradictions: list[str] = field(default_factory=list)
    knowledge: list[tuple[str, str, float]] = field(default_factory=list)

    def describe(self) -> str:
        """Describe the reasoning process."""
        lines = [
            f"Conclusion: {self.conclusion}",
            f"Type: {self.reasoning_type.value}",
            f"Confidence: {self.confidence:.2f}",
            "Evidence chain:",
        ]
        for step in self.evidence:
            lines.append(f"  → {step}")
        if self.novel:
            lines.append("  [NOVEL INSIGHT]")
        if self.contradictions:
            lines.append(f"  Contradictions: {', '.join(self.contradictions)}")
        return "\n".join(lines)


class ReasoningEngine:
    """Traverses the concept network to draw conclusions.

    The reasoning engine is stateless — it reads the concept network
    and produces conclusions. The concept network is the memory;
    the reasoning engine is the processor.
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Store the concept network that this engine will reason over."""
        self.network = network

    def reason_about(self, concept_name: str, depth: int = 3) -> list[ReasoningResult]:
        """Run all applicable reasoning about a concept.

        This is the main entry point. Given a concept, it:
        1. Finds transitive relationships (deductive) — IS_A, PART_OF,
           DEPENDS_ON, ENABLES chains
        2. Finds causal chains — CAUSES and LEADS_TO
        3. Detects contradictions
        4. Forms hypotheses about gaps (with typed relation prediction)
        5. Finds analogies — transfer through SIMILAR_TO
        6. Abductive reasoning — inference to best explanation

        Returns a list of reasoning results, sorted by confidence.
        """
        results: list[ReasoningResult] = []

        concept = self.network.get_concept(concept_name)
        if concept is None:
            return results

        # 1. Deductive: transitive chains for all transitive relations
        results.extend(self._transitive_chains(concept.id, depth))

        # 2. Causal: follow CAUSES and LEADS_TO chains
        results.extend(self._causal_chains(concept.id, depth))

        # 3. Contradiction detection
        results.extend(self._detect_contradictions(concept.id))

        # 4. Hypothesis formation (find gaps with typed predictions)
        results.extend(self._form_hypotheses(concept.id))

        # 5. Analogical: transfer properties through SIMILAR_TO
        results.extend(self._analogical_transfer(concept.id))

        # 6. Abductive: inference to best explanation
        results.extend(self._abductive_inference(concept.id))

        # Sort by confidence
        results.sort(key=lambda r: -r.confidence)
        # Filter out results that reference internal/structural concepts
        # (those starting with "_"). These are utterance templates
        # (_utt:...), category hubs (_cat:...), and other structural
        # concepts that should never appear in natural language output.
        # Without this filter, reasoning conclusions like "_utt:I've
        # been_thinking expresses _cat:utterance:reflection_opener"
        # leak into speech as raw concept IDs.
        results = [
            r for r in results
            if not self._references_internal_concept(r)
        ]
        return results

    @staticmethod
    def _references_internal_concept(result: ReasoningResult) -> bool:
        """Check if a reasoning result references internal/structural concepts.

        Internal concepts (starting with "_") are structural scaffolding
        (utterance templates, category hubs, etc.) that should never
        appear in reasoning conclusions or speech output.
        """
        text = result.conclusion
        for evidence in result.evidence:
            text += " " + evidence
        # Check for concept IDs starting with "_" (e.g. _utt:, _cat:)
        for token in text.split():
            if token.startswith("_") and len(token) > 1:
                return True
        return False

    # ── Transitive relations and their properties ────────────────

    # Relations where A→B and B→C implies A→C.
    _TRANSITIVE: ClassVar[dict[RelationType, float]] = {
        RelationType.IS_A: 0.9,
        RelationType.PART_OF: 0.85,
        RelationType.DEPENDS_ON: 0.75,
        RelationType.ENABLES: 0.7,
    }

    def _transitive_chains(self, concept_id: str, depth: int) -> list[ReasoningResult]:
        """Follow transitive chains for all transitive relation types.

        If A is_a B and B is_a C, then A is_a C.
        If A part_of B and B part_of C, then A part_of C.
        If A depends_on B and B depends_on C, then A depends_on C.
        If A enables B and B enables C, then A enables C.

        Each relation type has its own confidence decay rate.
        """
        results: list[ReasoningResult] = []

        for rel_type, base_conf in self._TRANSITIVE.items():
            def traverse(
                current: str,
                chain: list[str],
                d: int,
                rel_type: RelationType = rel_type,
                base_conf: float = base_conf,
            ) -> None:
                """Follow edges of a transitive relation, recording deductive conclusions."""
                if d >= depth:
                    return
                for edge in self.network.get_edges(current, "out"):
                    if edge.relation != rel_type:
                        continue
                    if edge.target in chain:
                        continue
                    new_chain = [*chain, edge.target]
                    step = f"{current} {rel_type.value} {edge.target}"
                    if len(new_chain) > 1:
                        # We have a transitive conclusion
                        evidence = [
                            f"{chain[i]} {rel_type.value} {chain[i + 1]}"
                            for i in range(len(chain) - 1)
                        ] + [step]
                        results.append(
                            ReasoningResult(
                                conclusion=(
                                    f"{concept_id} {rel_type.value} {edge.target} "
                                    f"(transitively, through {', '.join(chain[1:])})"
                                ),
                                reasoning_type=ReasoningType.DEDUCTIVE,
                                evidence=evidence,
                                confidence=base_conf ** (len(new_chain) - 1),
                            )
                        )
                    traverse(edge.target, new_chain, d + 1)

            traverse(concept_id, [concept_id], 0)
        return results

    def _causal_chains(self, concept_id: str, depth: int) -> list[ReasoningResult]:
        """Follow CAUSES and LEADS_TO chains."""
        results: list[ReasoningResult] = []

        def traverse(current: str, chain: list[str], d: int) -> None:
            """Recursively follow CAUSES and LEADS_TO edges, recording causal chain conclusions."""
            if d >= depth:
                return
            for edge in self.network.get_edges(current, "out"):
                if edge.relation not in (RelationType.CAUSES, RelationType.LEADS_TO):
                    continue
                if edge.target in chain:
                    continue
                new_chain = [*chain, edge.target]
                step = f"{current} {edge.relation.value} {edge.target}"
                if len(new_chain) > 2:
                    evidence = [
                        f"{chain[i]} → {chain[i + 1]}"
                        for i in range(len(chain) - 1)
                    ] + [step]
                    results.append(
                        ReasoningResult(
                            conclusion=(
                                f"{concept_id} ultimately leads to "
                                f"{edge.target} through a causal chain"
                            ),
                            reasoning_type=ReasoningType.CAUSAL,
                            evidence=evidence,
                            confidence=0.7 ** (len(new_chain) - 1),
                        )
                    )
                traverse(edge.target, new_chain, d + 1)

        traverse(concept_id, [concept_id], 0)
        return results

    def _detect_contradictions(self, concept_id: str) -> list[ReasoningResult]:
        """Detect contradictions involving this concept.

        Two kinds of contradictions are detected:
        1. Explicit CONTRADICTS edges — directly stated contradictions.
        2. Structural contradictions — a concept that both ENABLES and
           PREVENTS the same target is internally inconsistent.
        """
        results: list[ReasoningResult] = []

        # ── Explicit CONTRADICTS edges ──
        for edge in self.network.get_edges(concept_id, "both"):
            if edge.relation == RelationType.CONTRADICTS:
                other = edge.target if edge.source == concept_id else edge.source
                conclusion = f"{concept_id} and {other} contradict each other"
                evidence = [f"{edge.source} contradicts {edge.target}"]
                results.append(
                    ReasoningResult(
                        conclusion=conclusion,
                        reasoning_type=ReasoningType.CONTRADICTION,
                        evidence=evidence,
                        confidence=edge.weight,
                        contradictions=[other],
                    )
                )

                # Check if both are simultaneously asserted
                # (this would be a deeper contradiction)
                concept_a = self.network.get_concept(concept_id)
                concept_b = self.network.get_concept(other)
                both_active = (
                    concept_a
                    and concept_b
                    and (concept_a.activation or 0.0) > 0.3
                    and (concept_b.activation or 0.0) > 0.3
                )
                if both_active:
                    results[-1].conclusion = (
                        f"{concept_id} and {other} are both active "
                        f"yet contradict each other — this needs resolution"
                    )
                    results[-1].confidence = min(1.0, results[-1].confidence + 0.2)

        # ── Structural contradictions: ENABLES + PREVENTS same target ──
        enables_targets: dict[str, float] = {}
        prevents_targets: dict[str, float] = {}
        for edge in self.network.get_edges(concept_id, "out"):
            if edge.relation == RelationType.ENABLES:
                enables_targets[edge.target] = edge.weight
            elif edge.relation == RelationType.PREVENTS:
                prevents_targets[edge.target] = edge.weight

        for target in enables_targets:
            if target in prevents_targets:
                conclusion = (
                    f"{concept_id} both enables and prevents {target} "
                    f"— this is a structural contradiction"
                )
                evidence = [
                    f"{concept_id} enables {target} (weight={enables_targets[target]:.2f})",
                    f"{concept_id} prevents {target} (weight={prevents_targets[target]:.2f})",
                ]
                # Confidence proportional to the combined weight
                conf = min(1.0, enables_targets[target] + prevents_targets[target])
                results.append(
                    ReasoningResult(
                        conclusion=conclusion,
                        reasoning_type=ReasoningType.CONTRADICTION,
                        evidence=evidence,
                        confidence=conf,
                        contradictions=[target],
                    )
                )

        return results

    def _form_hypotheses(self, concept_id: str) -> list[ReasoningResult]:
        """Form hypotheses about gaps in the concept network.

        If concept A is connected to B, and B is connected to C,
        but A has no direct connection to C, we can hypothesize one.
        This is how new ideas form.

        The predicted relation type is inferred from the path structure:
        - If both edges are the same transitive relation, predict that
          relation (e.g., A causes B, B causes C → A might cause C).
        - If the edges differ, predict the weaker/more general one.
        - Confidence is higher when the path is structurally consistent.
        """
        results: list[ReasoningResult] = []

        # Find concepts 2 hops away, tracking the edge types
        one_hop: dict[str, list[RelationType]] = {}  # neighbor → relation types
        for edge in self.network.get_edges(concept_id, "out"):
            one_hop.setdefault(edge.target, []).append(edge.relation)

        # candidate → list of (intermediary, relation_to_candidate)
        two_hop: dict[str, list[tuple[str, RelationType]]] = {}
        for neighbor, _rels in one_hop.items():
            for edge in self.network.get_edges(neighbor, "out"):
                if edge.target != concept_id and edge.target not in one_hop:
                    two_hop.setdefault(edge.target, []).append(
                        (neighbor, edge.relation)
                    )

        # For each 2-hop concept, hypothesize a direct connection
        for candidate, paths in two_hop.items():
            # Don't hypothesize too many
            if len(results) >= 5:
                break

            # Predict the relation type from the path structure.
            # Collect the relation types from A→B and B→C for each path.
            path_relations: list[tuple[RelationType, RelationType]] = []
            for inter, rel_to_candidate in paths:
                for rel_to_inter in one_hop.get(inter, []):
                    path_relations.append((rel_to_inter, rel_to_candidate))

            suggested_relation = self._predict_hypothesis_relation(path_relations)
            # Confidence: higher when the path is structurally consistent
            # (both edges share the same relation type)
            consistent_paths = sum(
                1 for r1, r2 in path_relations if r1 == r2
            )
            base_conf = 0.3
            if consistent_paths > 0:
                base_conf += 0.15 * min(consistent_paths, 2)

            # Build a grammatically correct hypothesis conclusion.
            # Relation types are not all verbs — "part_of" is a
            # prepositional phrase, "is_a" is a copular construction.
            # Use "might be <relation>" for prepositional/copular
            # types, and "might <verb>" for verbal types.
            _COPULAR_RELATIONS = {
                "is_a", "part_of", "similar_to", "opposite_of",
                "related_to", "has_property",
            }
            rel_val = suggested_relation.value
            if rel_val in _COPULAR_RELATIONS:
                rel_phrase = rel_val.replace("_", " ")
                conclusion = (
                    f"{concept_id} might be {rel_phrase} "
                    f"{candidate} (currently connected only through "
                    f"{', '.join(p[0] for p in paths)})"
                )
            else:
                # Verbal relation: convert to base form for "might" modal
                rel_verb = rel_val.replace("_", " ")
                words = rel_verb.split()
                if words and words[0].endswith("s") and not words[0].endswith("ss"):
                    words[0] = words[0][:-1]
                rel_base = " ".join(words)
                conclusion = (
                    f"{concept_id} might {rel_base} "
                    f"{candidate} (currently connected only through "
                    f"{', '.join(p[0] for p in paths)})"
                )
            evidence = [f"gap: no direct {concept_id} → {candidate}"]
            for inter, rel in paths:
                evidence.append(f"{concept_id} → {inter}")
                evidence.append(f"{inter} {rel.value} {candidate}")
            results.append(
                ReasoningResult(
                    conclusion=conclusion,
                    reasoning_type=ReasoningType.HYPOTHESIS,
                    evidence=evidence,
                    confidence=base_conf,
                    novel=True,
                )
            )

        return results

    @staticmethod
    def _predict_hypothesis_relation(
        path_relations: list[tuple[RelationType, RelationType]],
    ) -> RelationType:
        """Predict the most likely relation type for a hypothesized edge.

        Given the relation types along the path A→B→C, predict what
        relation might directly connect A→C.

        Rules:
        - If both edges share the same transitive relation, predict
          that relation (transitivity suggests it).
        - If one edge is CAUSES and the other is LEADS_TO, predict
          CAUSES (they're semantically close).
        - If one edge is ENABLES and the other is DEPENDS_ON, predict
          RELATED_TO (the semantics get murky).
        - Default: RELATED_TO (safe, generic).
        """
        if not path_relations:
            return RelationType.RELATED_TO

        # Count how often each relation appears as the first or second edge
        from collections import Counter
        first_rels = Counter(r1 for r1, _ in path_relations)
        second_rels = Counter(r2 for _, r2 in path_relations)

        # If both edges share the same relation, predict it (transitivity)
        for r1, r2 in path_relations:
            if r1 == r2:
                return r1

        # CAUSES and LEADS_TO are semantically close — predict CAUSES
        all_rels = set(first_rels) | set(second_rels)
        if all_rels <= {RelationType.CAUSES, RelationType.LEADS_TO}:
            return RelationType.CAUSES

        # IS_A and INSTANCE_OF are close — predict IS_A
        if all_rels <= {RelationType.IS_A, RelationType.INSTANCE_OF}:
            return RelationType.IS_A

        # PART_OF and IS_A can chain — predict the first edge's relation
        if all_rels <= {RelationType.PART_OF, RelationType.IS_A}:
            return first_rels.most_common(1)[0][0]

        # Otherwise, predict the most common second-edge relation
        # (the relation that leads TO the candidate)
        if second_rels:
            return second_rels.most_common(1)[0][0]

        return RelationType.RELATED_TO

    def _analogical_transfer(self, concept_id: str) -> list[ReasoningResult]:
        """Transfer properties through SIMILAR_TO relationships.

        If A similar_to B, and B has property P, then A might have P.
        This is analogical reasoning — the foundation of metaphor.

        Similarity is symmetric — if B similar_to A (incoming edge),
        then A is also similar to B, and B's properties transfer to A.
        Both outgoing and incoming SIMILAR_TO edges are followed.
        """
        results: list[ReasoningResult] = []

        # Collect similar concepts from both outgoing and incoming
        # SIMILAR_TO edges (similarity is symmetric).
        # Skip hypothetical dream-synthesis edges — they are proposals,
        # not validated knowledge. Only dream-validated (or explicitly
        # stated) SIMILAR_TO edges should enable analogical transfer.
        similar_concepts: list[tuple[str, float]] = []
        for edge in self.network.get_edges(concept_id, "out"):
            if edge.relation == RelationType.SIMILAR_TO and edge.origin != "dream-synthesis":
                similar_concepts.append((edge.target, edge.weight))
        for edge in self.network.get_edges(concept_id, "in"):
            if edge.relation == RelationType.SIMILAR_TO and edge.origin != "dream-synthesis":
                similar_concepts.append((edge.source, edge.weight))

        for similar_concept, edge_weight in similar_concepts:
            # Find properties of the similar concept
            for s_edge in self.network.get_edges(similar_concept, "out"):
                if s_edge.relation in (
                    RelationType.IS_A,
                    RelationType.PART_OF,
                    RelationType.DEPENDS_ON,
                    RelationType.ENABLES,
                    RelationType.CAUSES,
                    RelationType.PREVENTS,
                    RelationType.HARMS,
                ):
                    conclusion = (
                        f"{concept_id} might {s_edge.relation.value} "
                        f"{s_edge.target}, by analogy with {similar_concept}"
                    )
                    evidence = [
                        f"{concept_id} similar_to {similar_concept}",
                        f"{similar_concept} {s_edge.relation.value} {s_edge.target}",
                    ]
                    confidence = edge_weight * s_edge.weight * 0.6
                    results.append(
                        ReasoningResult(
                            conclusion=conclusion,
                            reasoning_type=ReasoningType.ANALOGICAL,
                            evidence=evidence,
                            confidence=confidence,
                            novel=True,
                        )
                    )

        return results

    def _abductive_inference(self, concept_id: str) -> list[ReasoningResult]:
        """Abductive reasoning — inference to the best explanation.

        If we observe Y, and X causes Y, then X is a possible
        explanation for Y. When reasoning about Y, we look for
        concepts X such that X causes Y (or X leads_to Y) and
        propose X as an explanation.

        This is the reasoning behind "the dog barked, so maybe someone
        is at the door" — we infer the cause from the effect.

        Confidence is proportional to the edge weight and reduced for
        each additional possible explanation (more alternatives = less
        certainty in any single one).
        """
        results: list[ReasoningResult] = []

        # Find all concepts that CAUSE or LEAD_TO this concept
        # (incoming edges with causal relations)
        causes: list[tuple[str, RelationType, float]] = []
        for edge in self.network.get_edges(concept_id, "in"):
            if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                causes.append((edge.source, edge.relation, edge.weight))

        if not causes:
            return results

        # Confidence is diluted by the number of alternative explanations.
        # With 1 cause: high confidence. With 3 causes: lower confidence.
        # This mirrors the logic of "inference to the best explanation":
        # fewer alternatives = stronger inference.
        n_alternatives = len(causes)
        dilution = 1.0 / (1.0 + 0.3 * (n_alternatives - 1))

        for cause, rel_type, weight in causes:
            conclusion = (
                f"{cause} is a possible explanation for {concept_id} "
                f"(abductive inference: {cause} {rel_type.value} {concept_id})"
            )
            evidence = [
                f"observed: {concept_id}",
                f"{cause} {rel_type.value} {concept_id} (weight={weight:.2f})",
            ]

            # Check if the cause is itself caused by something —
            # if so, mention the deeper chain as supporting evidence
            for deeper_edge in self.network.get_edges(cause, "in"):
                if deeper_edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                    evidence.append(
                        f"{deeper_edge.source} {deeper_edge.relation.value} {cause} "
                        f"(deeper cause)"
                    )
                    break

            confidence = weight * 0.6 * dilution
            results.append(
                ReasoningResult(
                    conclusion=conclusion,
                    reasoning_type=ReasoningType.ABDUCTIVE,
                    evidence=evidence,
                    confidence=confidence,
                    novel=True,
                )
            )

        # Sort by confidence — best explanation first
        results.sort(key=lambda r: -r.confidence)
        # Keep only the top 3 explanations
        return results[:3]

    def synthesize(self, concepts: list[str]) -> ReasoningResult | None:
        """Synthesize a conclusion from multiple concepts.

        When multiple concepts are active simultaneously, the reasoning
        engine looks for shared connections — a concept that all of
        them relate to. This is how synthesis works: finding the
        common thread.
        """
        if len(concepts) < 2:
            return None

        # Find neighbors of each concept
        neighbor_sets: list[set[str]] = []
        for name in concepts:
            cid = self.network._resolve(name)
            if not cid:
                continue
            neighbors = {cid}
            for edge in self.network.get_edges(cid, "both"):
                neighbors.add(edge.target if edge.source == cid else edge.source)
            neighbor_sets.append(neighbors)

        if len(neighbor_sets) < 2:
            return None

        # Find intersection — concepts connected to ALL input concepts
        intersection = neighbor_sets[0]
        for ns in neighbor_sets[1:]:
            intersection &= ns

        # Remove the input concepts themselves
        for name in concepts:
            cid = self.network._resolve(name)
            if cid and cid in intersection:
                intersection.discard(cid)

        if not intersection:
            return None

        # Pick the most connected shared concept
        best = max(intersection, key=lambda c: len(self.network.get_edges(c, "both")))

        conclusion = (
            f"{', '.join(concepts)} are all connected through {best} "
            f"— this might be a unifying concept"
        )
        evidence = []
        for name in concepts:
            cid = self.network._resolve(name)
            if cid:
                path = self.network.find_path(cid, best, max_depth=3)
                if path:
                    chain = " → ".join(f"{e.source} {e.relation.value} {e.target}" for e in path)
                    evidence.append(f"{name}: {chain}")

        return ReasoningResult(
            conclusion=conclusion,
            reasoning_type=ReasoningType.SYNTHESIS,
            evidence=evidence,
            confidence=0.5,
            novel=True,
        )

    def _inherit_parent_properties(
        self,
        subject: str,
        parents: list[str],
        facts: list[str],
        evidence: list[str],
        seen_targets: set[str],
    ) -> None:
        """Inherit has_property from is_a parents (priority 1.25).

        If fido is_a dog and dog has_property furry, then fido is furry.
        This is a core form of human-like default reasoning: instances
        and subkinds inherit attributes from their categories. We only
        inherit has_property (not arbitrary relations) because properties
        are the typical shared traits of a category.

        Mutates facts, evidence, and seen_targets in place, stopping
        once 3 facts are gathered.
        """
        visited: set[str] = set()
        queue = list(parents)
        while queue and len(facts) < 3:
            parent = queue.pop(0)
            if parent in visited:
                continue
            visited.add(parent)
            for prop_edge in self.network.get_edges(parent, "out"):
                if (
                    prop_edge.relation == RelationType.HAS_PROPERTY
                    and prop_edge.target not in seen_targets
                ):
                    facts.append(f"{subject} has {prop_edge.target} (from {parent})")
                    evidence.append(f"{subject} is_a {parent}")
                    evidence.append(f"{parent} has_property {prop_edge.target}")
                    seen_targets.add(prop_edge.target)
                    if len(facts) >= 3:
                        break
            # Also climb higher for transitive inheritance (fido is_a
            # dog is_a animal → fido can inherit from animal).
            for parent_edge in self.network.get_edges(parent, "out"):
                if parent_edge.relation == RelationType.IS_A:
                    queue.append(parent_edge.target)

    @staticmethod
    def _build_what_is_conclusion(facts: list[str], evidence: list[str]) -> ReasoningResult:
        """Build the final ReasoningResult from gathered facts."""
        if len(facts) == 1:
            conclusion = facts[0]
        elif len(facts) == 2:
            conclusion = f"{facts[0]}, and {facts[1]}"
        else:
            conclusion = f"{facts[0]}, {facts[1]}, and {facts[2]}"
        return ReasoningResult(
            conclusion=conclusion,
            reasoning_type=ReasoningType.DEDUCTIVE,
            evidence=evidence,
            confidence=0.7,
        )

    def _answer_what_is(self, subject: str) -> ReasoningResult | None:
        """Gather multiple facts for a 'what_is' question and build a result."""
        facts: list[str] = []
        evidence: list[str] = []
        seen_targets: set[str] = set()

        # Priority 1: IS_A relationships
        parents: list[str] = []
        for edge in self.network.get_edges(subject, "out"):
            if edge.relation == RelationType.IS_A and edge.target not in seen_targets:
                facts.append(f"{subject} is {edge.target}")
                evidence.append(f"{subject} is_a {edge.target}")
                seen_targets.add(edge.target)
                parents.append(edge.target)
                if len(facts) >= 3:
                    break

        # Priority 1.25: inherit properties from is_a parents.
        if len(facts) < 3:
            self._inherit_parent_properties(subject, parents, facts, evidence, seen_targets)

        # Priority 1.5: incoming PART_OF (what the subject is made of).
        # If A is part of the subject, the subject is made of A (and B, ...).
        if len(facts) < 3:
            parts: list[str] = []
            for edge in self.network.get_edges(subject, "in"):
                if edge.relation == RelationType.PART_OF and edge.source not in seen_targets:
                    parts.append(edge.source)
                    seen_targets.add(edge.source)
                    if len(parts) >= 4:
                        break
            if parts:
                if len(parts) == 1:
                    made_of = f"{subject} is made of {parts[0]}"
                elif len(parts) == 2:
                    made_of = f"{subject} is made of {parts[0]} and {parts[1]}"
                else:
                    made_of = f"{subject} is made of {', '.join(parts[:-1])}, and {parts[-1]}"
                facts.append(made_of)
                evidence.extend(f"{p} part_of {subject}" for p in parts)

        # Priority 2: outgoing PART_OF relationships
        if len(facts) < 3:
            for edge in self.network.get_edges(subject, "out"):
                if edge.relation == RelationType.PART_OF and edge.target not in seen_targets:
                    facts.append(f"{subject} is part of {edge.target}")
                    evidence.append(f"{subject} part_of {edge.target}")
                    seen_targets.add(edge.target)
                    if len(facts) >= 3:
                        break

        # Priority 2.5: incoming causes/enables (what the subject is
        # caused by or enabled by). This lets "what is heat" answer
        # "heat is caused by fire" when the network stores
        # "fire causes heat".
        if len(facts) < 3:
            for edge in self.network.get_edges(subject, "in"):
                if edge.relation in (RelationType.CAUSES, RelationType.ENABLES):
                    rel_str = edge.relation.value.replace("_", " ")
                    # Convert third-person present to simple past
                    # participle for the passive voice: "causes" →
                    # "caused", "enables" → "enabled".
                    rel_past = rel_str[:-1] + "d" if rel_str.endswith("s") else rel_str
                    facts.append(f"{subject} is {rel_past} by {edge.source}")
                    evidence.append(f"{edge.source} {rel_str} {subject}")
                    seen_targets.add(edge.source)
                    if len(facts) >= 3:
                        break

        # Priority 3: Other relationships (RELATED_TO, ENABLES, etc.)
        if len(facts) < 3:
            for edge in self.network.get_edges(subject, "out"):
                if edge.target not in seen_targets:
                    if edge.relation == RelationType.HAS_PROPERTY:
                        # "has" is the natural form of has_property.
                        facts.append(f"{subject} has {edge.target}")
                        evidence.append(f"{subject} has_property {edge.target}")
                    else:
                        rel_str = edge.relation.value.replace("_", " ")
                        facts.append(f"{subject} {rel_str} {edge.target}")
                        evidence.append(f"{subject} {rel_str} {edge.target}")
                    seen_targets.add(edge.target)
                    if len(facts) >= 4:
                        break

        if facts:
            return self._build_what_is_conclusion(facts, evidence)
        return None

    def _answer_who(self, subject: str) -> ReasoningResult | None:
        """Answer 'who created/caused X?' or 'who is the Y of X?'.

        This is a first step toward human-like agent inference: instead
        of just listing a relation, it identifies the agent behind it.
        """
        agents: list[tuple[str, str, float]] = []  # (agent, role/verb, confidence)

        # Direct agent-patient edges: X creates/causes Y → agent is X.
        for edge in self.network.get_edges(subject, "in"):
            if edge.relation in (
                RelationType.CREATES,
                RelationType.CAUSES,
                RelationType.ENABLES,
            ):
                agents.append((edge.source, edge.relation.value, edge.weight))
                if len(agents) >= 3:
                    break

        # For 'relates_to' edges, look for a role (is_a parent) of the
        # source concept. 'Alice is the creator of Genesis' produces
        # alice relates_to genesis and alice is_a creator, so we can
        # infer 'Alice is the creator of Genesis'.
        if not agents:
            for edge in self.network.get_edges(subject, "in"):
                if edge.relation == RelationType.RELATED_TO:
                    for role_edge in self.network.get_edges(edge.source, "out"):
                        if (
                            role_edge.relation == RelationType.IS_A
                            and role_edge.target != subject
                        ):
                            agents.append(
                                (edge.source, f"the {role_edge.target}", role_edge.weight)
                            )
                            break
                    if len(agents) >= 3:
                        break

        if not agents:
            return None

        if len(agents) == 1:
            agent, role, conf = agents[0]
            if role in ("creates", "causes", "enables"):
                # Past-participle for a natural answer: "Alice created Genesis"
                verb_past = role[:-1] + "d" if role.endswith("s") else role
                conclusion = f"{agent} {verb_past} {subject}"
            else:
                conclusion = f"{agent} is {role} of {subject}"
            return ReasoningResult(
                conclusion=conclusion,
                reasoning_type=ReasoningType.DEDUCTIVE,
                evidence=[f"{agent} {role.replace('the ', '')} {subject}"],
                confidence=conf,
            )

        sources = [a[0] for a in agents]
        conclusion = f"{', '.join(sources[:-1])}, and {sources[-1]} are associated with {subject}"
        evidence = [f"{a[0]} {a[1].replace('the ', '')} {subject}" for a in agents]
        return ReasoningResult(
            conclusion=conclusion,
            reasoning_type=ReasoningType.DEDUCTIVE,
            evidence=evidence,
            confidence=sum(a[2] for a in agents) / len(agents),
        )

    def answer_question(self, subject: str, question_type: str) -> ReasoningResult | None:
        """Attempt to answer a question about a concept.

        question_type: "what_is", "why", "how", "what_causes",
                       "what_does_it_do", "what_is_it_part_of"
        """
        concept = self.network.get_concept(subject)
        if concept is None:
            return None

        if question_type == "what_is":
            return self._answer_what_is(subject)

        elif question_type == "who" or question_type == "who_is":
            return self._answer_who(subject)

        elif question_type == "what_causes":
            # "what_causes" in this project means "what does X cause?".
            # It searches outgoing CAUSES edges from the subject. This is
            # used by the existing test suite. Natural
            # English "what causes X?" is handled by `_answer_what_is`
            # which reports "X is caused by Y" from incoming edges.
            causes: list[tuple[str, float, str]] = []
            for edge in self.network.get_edges(subject, "out"):
                if edge.relation == RelationType.CAUSES:
                    causes.append((edge.target, edge.weight, edge.relation.value))
                    if len(causes) >= 4:
                        break
            if causes:
                if len(causes) == 1:
                    return ReasoningResult(
                        conclusion=f"{subject} causes {causes[0][0]}",
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        evidence=[f"{subject} causes {causes[0][0]}"],
                        confidence=causes[0][1],
                    )
                else:
                    targets = [c[0] for c in causes]
                    conclusion = f"{subject} causes {', '.join(targets[:-1])}, and {targets[-1]}"
                    evidence = [f"{subject} causes {t}" for t in targets]
                    avg_conf = sum(c[1] for c in causes) / len(causes)
                    return ReasoningResult(
                        conclusion=conclusion,
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        evidence=evidence,
                        confidence=avg_conf,
                    )

        elif question_type == "what_is_it_part_of":
            # "What is X part of?" → find larger things X belongs to.
            # These are outgoing PART_OF edges: X is part of Y.
            wholes: list[tuple[str, float]] = []
            for edge in self.network.get_edges(subject, "out"):
                if edge.relation == RelationType.PART_OF:
                    wholes.append((edge.target, edge.weight))
                    if len(wholes) >= 4:
                        break
            if wholes:
                if len(wholes) == 1:
                    return ReasoningResult(
                        conclusion=f"{subject} is part of {wholes[0][0]}",
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        evidence=[f"{subject} part_of {wholes[0][0]}"],
                        confidence=wholes[0][1],
                    )
                else:
                    targets = [w[0] for w in wholes]
                    conclusion = (
                        f"{subject} is part of "
                        f"{', '.join(targets[:-1])}, and {targets[-1]}"
                    )
                    evidence = [f"{subject} part_of {t}" for t in targets]
                    avg_conf = sum(w[1] for w in wholes) / len(wholes)
                    return ReasoningResult(
                        conclusion=conclusion,
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        evidence=evidence,
                        confidence=avg_conf,
                    )

        elif question_type == "what_emerges_from":
            for edge in self.network.get_edges(subject, "in"):
                if edge.relation == RelationType.EMERGES_FROM:
                    return ReasoningResult(
                        conclusion=f"{subject} emerges from {edge.source}",
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        evidence=[f"{subject} emerges_from {edge.source}"],
                        confidence=edge.weight,
                    )

        elif question_type == "what_enables":
            for edge in self.network.get_edges(subject, "out"):
                if edge.relation == RelationType.ENABLES:
                    return ReasoningResult(
                        conclusion=f"{subject} enables {edge.target}",
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        evidence=[f"{subject} enables {edge.target}"],
                        confidence=edge.weight,
                    )

        elif question_type == "what_depends_on":
            for edge in self.network.get_edges(subject, "out"):
                if edge.relation == RelationType.DEPENDS_ON:
                    return ReasoningResult(
                        conclusion=f"{subject} depends on {edge.target}",
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        evidence=[f"{subject} depends_on {edge.target}"],
                        confidence=edge.weight,
                    )

        return None

    def _collect_direct_facts(
        self, ca: str, cb: str, concept_a: str, concept_b: str
    ) -> tuple[list[str], list[str]]:
        """Collect facts and evidence from direct edges between two concepts."""
        edges_ab = [
            e
            for e in self.network.get_edges(ca, "out")
            if (self.network._resolve(e.target) or e.target.lower()) == cb
        ]
        edges_ba = [
            e
            for e in self.network.get_edges(cb, "out")
            if (self.network._resolve(e.target) or e.target.lower()) == ca
        ]

        facts: list[str] = []
        evidence: list[str] = []

        for e in edges_ab:
            evidence.append(f"{ca} {e.relation.value} {cb}")
            if e.relation == RelationType.IS_A:
                facts.append(f"{concept_a} is a kind of {concept_b}")
            elif e.relation == RelationType.PART_OF:
                facts.append(f"{concept_a} is a component of {concept_b}")
            elif e.relation == RelationType.OPPOSITE_OF:
                facts.append(f"{concept_a} is the opposite of {concept_b}")
            elif e.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                facts.append(f"{concept_a} causes {concept_b}")
            elif e.relation == RelationType.CREATES:
                facts.append(f"{concept_a} creates {concept_b}")
            elif e.relation == RelationType.ENABLES:
                facts.append(f"{concept_a} enables {concept_b}")
            elif e.relation == RelationType.EMERGES_FROM:
                facts.append(f"{concept_a} emerges from {concept_b}")
            elif e.relation == RelationType.DEPENDS_ON:
                facts.append(f"{concept_a} depends on {concept_b}")
            elif e.relation == RelationType.SIMILAR_TO:
                facts.append(f"{concept_a} is similar to {concept_b}")
            elif e.relation == RelationType.CALLS:
                facts.append(f"{concept_a} calls {concept_b}")
            elif e.relation == RelationType.DEFINES:
                facts.append(f"{concept_a} defines {concept_b}")
            else:
                facts.append(f"{concept_a} is directly related to {concept_b}")

        for e in edges_ba:
            evidence.append(f"{cb} {e.relation.value} {ca}")
            if e.relation == RelationType.IS_A and not any("is a kind of" in f for f in facts):
                facts.append(f"{concept_b} is a kind of {concept_a}")
            elif e.relation == RelationType.PART_OF and not any("component" in f for f in facts):
                facts.append(f"{concept_b} is part of {concept_a}")
            elif e.relation == RelationType.OPPOSITE_OF and not any("opposite" in f for f in facts):
                facts.append(f"{concept_b} is the opposite of {concept_a}")
            elif e.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                facts.append(f"{concept_b} causes {concept_a}")
            elif e.relation == RelationType.CREATES:
                facts.append(f"{concept_b} creates {concept_a}")
            elif e.relation == RelationType.ENABLES:
                facts.append(f"{concept_b} enables {concept_a}")
            elif e.relation == RelationType.EMERGES_FROM:
                facts.append(f"{concept_b} emerges from {concept_a}")
            elif e.relation == RelationType.DEPENDS_ON:
                facts.append(f"{concept_b} depends on {concept_a}")

        return facts, evidence

    def _explain_direct_relations(
        self, ca: str, cb: str, concept_a: str, concept_b: str
    ) -> ReasoningResult | None:
        """Explain direct relationships between two concepts."""
        facts, evidence = self._collect_direct_facts(ca, cb, concept_a, concept_b)

        if facts:
            # Deduplicate while preserving order
            unique_facts: list[str] = []
            seen_f: set[str] = set()
            for f in facts:
                f_norm = f.lower().strip()
                if f_norm not in seen_f:
                    seen_f.add(f_norm)
                    unique_facts.append(f)

            if len(unique_facts) == 1:
                conclusion = f"{unique_facts[0].capitalize()}."
            elif len(unique_facts) == 2:
                conclusion = f"{unique_facts[0].capitalize()}, and {unique_facts[1]}."
            else:
                conclusion = (
                    f"{unique_facts[0].capitalize()}, {unique_facts[1]}, and {unique_facts[2]}."
                )

            return ReasoningResult(
                conclusion=conclusion,
                reasoning_type=ReasoningType.DEDUCTIVE,
                evidence=evidence,
                confidence=0.9,
            )
        return None

    def _explain_multihop_path(self, ca: str, cb: str) -> ReasoningResult | None:
        """Explain relationship via multi-hop shortest path.

        find_path is now undirected (follows both outgoing and incoming
        edges), so a single call suffices — the old ``or find_path(cb, ca)``
        fallback is no longer needed.
        """
        path = self.network.find_path(ca, cb, max_depth=4)
        if path:
            path_evidence = [f"{e.source} {e.relation.value} {e.target}" for e in path]
            chain_descriptions = []
            for e in path:
                rel = e.relation.value.replace("_", " ")
                chain_descriptions.append(f"{e.source} {rel} {e.target}")

            # Formulate coherent multi-hop statement
            if len(path) == 2:
                mid = path[0].target
                rel1 = path[0].relation.value.replace("_", " ")
                rel2 = path[1].relation.value.replace("_", " ")
                conclusion = f"{path[0].source} {rel1} {mid}, which in turn {rel2} {path[1].target}"
            else:
                conclusion = " -> ".join(chain_descriptions)

            return ReasoningResult(
                conclusion=conclusion,
                reasoning_type=ReasoningType.DEDUCTIVE,
                evidence=path_evidence,
                confidence=0.75,
            )
        return None

    def explain_relation(self, concept_a: str, concept_b: str) -> ReasoningResult | None:
        """Explain the relationship between two concepts via graph traversal and deduction.

        Checks direct edges in both directions, multi-hop shortest paths,
        and shared semantic hubs/categories.
        """
        ca = self.network._resolve(concept_a) or concept_a.lower()
        cb = self.network._resolve(concept_b) or concept_b.lower()

        if ca == cb:
            return ReasoningResult(
                conclusion=f"{concept_a} and {concept_b} refer to the same concept.",
                reasoning_type=ReasoningType.DEDUCTIVE,
                evidence=[f"{ca} == {cb}"],
                confidence=1.0,
            )

        # 1. Direct relationships (A -> B or B -> A)
        direct = self._explain_direct_relations(ca, cb, concept_a, concept_b)
        if direct:
            return direct

        # 2. Multi-hop shortest path (A -> ... -> B)
        path_result = self._explain_multihop_path(ca, cb)
        if path_result:
            return path_result

        # 3. Shared common concepts / category synthesis
        syn_result = self.synthesize([ca, cb])
        if syn_result:
            return syn_result

        return None

    def _gather_process_edges(
        self, proc_id: str, process_name: str
    ) -> tuple[list[str], list[str], list[str]]:
        """Gather triggers and outcomes for a process."""
        triggers: list[str] = []
        outcomes: list[str] = []
        evidence: list[str] = []

        # Incoming causes / triggers
        for e in self.network.get_edges(proc_id, "in"):
            if e.relation in (RelationType.CAUSES, RelationType.LEADS_TO, RelationType.ENABLES):
                triggers.append(f"{e.source} {e.relation.value.replace('_', ' ')} {process_name}")
                evidence.append(f"{e.source} -> {proc_id}")

        # Outgoing products / outcomes
        for e in self.network.get_edges(proc_id, "out"):
            if e.relation in (
                RelationType.CREATES,
                RelationType.LEADS_TO,
                RelationType.CAUSES,
                RelationType.ENABLES,
            ):
                outcomes.append(f"{e.relation.value.replace('_', ' ')} {e.target}")
                evidence.append(f"{proc_id} -> {e.target}")

        return triggers, outcomes, evidence

    def _build_subject_narrative(
        self,
        process_name: str,
        subject: str,
        proc_id: str,
        outcomes: list[str],
        triggers: list[str],
        definition: str,
        evidence: list[str],
    ) -> ReasoningResult | None:
        """Build a subject-centered narrative for a process."""
        subj_id = self.network._resolve(subject) or subject.lower()
        # Check if the subject has any relationship to the process
        subj_to_proc = [
            e
            for e in self.network.get_edges(subj_id, "out")
            if (self.network._resolve(e.target) or e.target.lower()) == proc_id
        ]
        # Also check if subject is related to the process's outcomes
        # (e.g., water → ice, and freezing creates ice)
        subj_to_outcome = False
        subj_targets = {
            self.network._resolve(se.target) or se.target.lower()
            for se in self.network.get_edges(subj_id, "out")
        }
        for e in self.network.get_edges(proc_id, "out"):
            if e.relation not in (RelationType.CREATES, RelationType.LEADS_TO):
                continue
            target_id = self.network._resolve(e.target) or e.target.lower()
            if target_id in subj_targets:
                subj_to_outcome = True
                break

        if subj_to_proc or subj_to_outcome or outcomes:
            # Conjugate the process verb for the subject.
            # "freezing" → "freezes", "boiling" → "boils", "melting" → "melts"
            # The -ing form often drops a silent "e" (freeze→freezing),
            # so we check the network for the base form.
            verb = process_name + "s"
            if process_name.endswith("ing"):
                base = process_name[:-3]
                # Try base as-is (boil, melt) and with added "e" (freeze)
                if self.network.get_concept(base):
                    verb = base + "s"
                elif self.network.get_concept(base + "e"):
                    verb = base + "es"
                else:
                    verb = base + "s"

            outcome_str = outcomes[0] if outcomes else ""
            parts: list[str] = []
            knowledge: list[tuple[str, str, float]] = []
            parts.append(f"When {subject} {verb}, it {outcome_str}")
            knowledge.append(("causes", outcome_str, 0.85))
            if triggers:
                parts.append(f"this is brought about when {triggers[0]}")
                knowledge.append(("triggered_by", triggers[0], 0.8))
            if definition:
                parts.append(f"{process_name.capitalize()} is {definition}")
                knowledge.append(("is", definition, 0.85))
            if parts:
                return ReasoningResult(
                    conclusion=". ".join(parts),
                    reasoning_type=ReasoningType.CAUSAL,
                    evidence=evidence or [f"process:{proc_id}"],
                    confidence=0.85,
                    knowledge=knowledge,
                )
        return None

    def explain_process(
        self, process_name: str, subject: str | None = None
    ) -> ReasoningResult | None:
        """Explain a physical, chemical, biological, or cognitive process.

        Traces triggers, conditions, transformations, and outcomes.
        When a subject is provided (e.g., "water" in "what happens when
        water freezes"), the narrative is woven around the subject's
        transformation.
        """
        proc_id = self.network._resolve(process_name) or process_name.lower()
        concept = self.network.get_concept(proc_id)
        if concept is None:
            return None

        triggers, outcomes, evidence = self._gather_process_edges(proc_id, process_name)

        definition = concept.properties.get("definition", "")
        parts: list[str] = []

        # Build a subject-centered narrative when possible
        if subject and outcomes:
            result = self._build_subject_narrative(
                process_name, subject, proc_id, outcomes, triggers, definition, evidence
            )
            if result:
                return result

        if definition:
            parts.append(f"{process_name.capitalize()} is {definition}")
        if triggers:
            parts.append(f"It is brought about when {triggers[0]}")
        if outcomes:
            parts.append(f"resulting in {outcomes[0]}")

        if parts:
            knowledge: list[tuple[str, str, float]] = []
            if definition:
                knowledge.append(("is", definition, 0.85))
            if triggers:
                knowledge.append(("triggered_by", triggers[0], 0.8))
            if outcomes:
                knowledge.append(("results_in", outcomes[0], 0.85))
            return ReasoningResult(
                conclusion=". ".join(parts),
                reasoning_type=ReasoningType.CAUSAL,
                evidence=evidence or [f"process:{proc_id}"],
                confidence=0.85,
                knowledge=knowledge,
            )
        return None

    def compare_concepts(self, concept_a: str, concept_b: str) -> ReasoningResult | None:
        """Compare and contrast two concepts by examining shared categories and distinguishing"
        "features."""
        ca = self.network._resolve(concept_a) or concept_a.lower()
        cb = self.network._resolve(concept_b) or concept_b.lower()

        c_a = self.network.get_concept(ca)
        c_b = self.network.get_concept(cb)
        if not c_a or not c_b:
            return None

        # Check for explicit OPPOSITE_OF
        opposites = any(
            e.relation == RelationType.OPPOSITE_OF and (self.network._resolve(e.target) == cb)
            for e in self.network.get_edges(ca, "out")
        )

        # Check shared categories/hypernyms
        hypernyms_a = {
            e.target for e in self.network.get_edges(ca, "out") if e.relation == RelationType.IS_A
        }
        hypernyms_b = {
            e.target for e in self.network.get_edges(cb, "out") if e.relation == RelationType.IS_A
        }
        shared_hypernyms = hypernyms_a & hypernyms_b

        # Distinguishing properties
        def_a = c_a.properties.get("definition", "")
        def_b = c_b.properties.get("definition", "")

        facts = []
        if shared_hypernyms:
            shared_str = ", ".join(shared_hypernyms)
            facts.append(f"Both {concept_a} and {concept_b} are kinds of {shared_str}")

        if opposites:
            facts.append(
                f"{concept_a} and {concept_b} represent opposing physical states or qualities"
            )

        if def_a and def_b:
            facts.append(f"{concept_a} is {def_a}, while {concept_b} is {def_b}")

        if facts:
            return ReasoningResult(
                conclusion=". ".join(facts),
                reasoning_type=ReasoningType.ANALOGICAL,
                evidence=[f"compare({ca}, {cb})"],
                confidence=0.8,
            )
        return None

    def explain_affordances(self, concept_name: str) -> ReasoningResult | None:
        """Explain the affordances, functions, and ecological/practical roles of a concept."""
        cid = self.network._resolve(concept_name) or concept_name.lower()
        c = self.network.get_concept(cid)
        if not c:
            return None

        def _display(cid_or_name: str) -> str:
            """Convert a concept id to a human-readable display name."""
            name = cid_or_name
            if ":" in name:
                name = name.split(":", 1)[1]
            return name.replace("_", " ").strip()

        display_name = _display(concept_name)

        # Deduplicate while preserving order
        enabled = list(
            dict.fromkeys(
                _display(e.target)
                for e in self.network.get_edges(cid, "out")
                if e.relation == RelationType.ENABLES
            )
        )
        caused = list(
            dict.fromkeys(
                _display(e.target)
                for e in self.network.get_edges(cid, "out")
                if e.relation == RelationType.CAUSES
            )
        )
        leads_to = list(
            dict.fromkeys(
                _display(e.target)
                for e in self.network.get_edges(cid, "out")
                if e.relation == RelationType.LEADS_TO
            )
        )
        dependent = list(
            dict.fromkeys(
                _display(e.source)
                for e in self.network.get_edges(cid, "in")
                if e.relation == RelationType.DEPENDS_ON
            )
        )
        created = list(
            dict.fromkeys(
                _display(e.target)
                for e in self.network.get_edges(cid, "out")
                if e.relation == RelationType.CREATES
            )
        )

        parts = []
        knowledge: list[tuple[str, str, float]] = []
        if enabled:
            parts.append(f"{display_name} enables {', '.join(enabled)}")
            for e in enabled:
                knowledge.append(("enables", e, 0.85))
        if caused:
            parts.append(f"{display_name} causes {', '.join(caused)}")
            for caused_item in caused:
                knowledge.append(("causes", caused_item, 0.85))
        if leads_to:
            parts.append(f"{display_name} leads to {', '.join(leads_to)}")
            for lt in leads_to:
                knowledge.append(("leads_to", lt, 0.8))
        if dependent:
            parts.append(f"{', '.join(dependent)} depend on {display_name}")
            for d in dependent:
                knowledge.append(("depended_on_by", d, 0.8))
        if created:
            parts.append(f"it gives rise to {', '.join(created)}")
            for cr in created:
                knowledge.append(("creates", cr, 0.85))

        if parts:
            return ReasoningResult(
                conclusion=(
                    f"{display_name.capitalize()} plays a vital functional role: {'; '.join(parts)}"
                ),
                reasoning_type=ReasoningType.DEDUCTIVE,
                evidence=[f"affordances({cid})"],
                confidence=0.85,
                knowledge=knowledge,
            )
        return None

    def _find_why_dependency(self, subj_id: str, obj_id: str) -> list:
        """Verify that a dependency exists between subject and object."""
        dep_edges = [
            e
            for e in self.network.get_edges(subj_id, "out")
            if e.relation == RelationType.DEPENDS_ON
            and (self.network._resolve(e.target) or e.target.lower()) == obj_id
        ]
        # Also check incoming DEPENDS_ON from subject's perspective
        if not dep_edges:
            dep_edges = [
                e
                for e in self.network.get_edges(obj_id, "in")
                if e.relation == RelationType.DEPENDS_ON
                and (self.network._resolve(e.source) or e.source.lower()) == subj_id
            ]
        return dep_edges

    def _concepts_connected(
        self, source: str, target: str, connection_rels: tuple[RelationType, ...]
    ) -> bool:
        """Check if source connects to target via a connection relation."""
        for se in self.network.get_edges(source, "out"):
            se_target = self.network._resolve(se.target) or se.target.lower()
            if se.relation in connection_rels and se_target == target:
                return True
        return False

    def _find_connecting_concepts(
        self, enabled: list[str], subj_types: list[str]
    ) -> list[str]:
        """Find concepts connecting enabled features to subject types."""
        connecting_concepts: list[str] = []
        connection_rels = (
            RelationType.IS_A,
            RelationType.RELATED_TO,
            RelationType.SIMILAR_TO,
        )
        for en in enabled:
            en_id = self.network._resolve(en) or en.lower()
            for st in subj_types:
                st_id = self.network._resolve(st) or st.lower()
                if st_id == en_id:
                    connecting_concepts.append(en)
                    continue
                # Check if enabled concept connects to subject's type
                if self._concepts_connected(en_id, st_id, connection_rels):
                    if en not in connecting_concepts:
                        connecting_concepts.append(en)
                    continue
                # Also check reverse: subject's type connects to enabled
                if self._concepts_connected(st_id, en_id, connection_rels):
                    if en not in connecting_concepts:
                        connecting_concepts.append(en)

        # Prefer the most fundamental connecting concept — the one with
        # the most incoming DEPENDS_ON edges (most things rely on it).
        if connecting_concepts:
            connecting_concepts.sort(
                key=lambda c: sum(
                    1
                    for e in self.network.get_edges(self.network._resolve(c) or c.lower(), "in")
                    if e.relation == RelationType.DEPENDS_ON
                ),
                reverse=True,
            )
        return connecting_concepts

    def _build_why_explanation(
        self,
        subject: str,
        object_: str,
        subj_id: str,
        obj_id: str,
        obj_concept: Any,
        enabled: list[str],
        connecting_concepts: list[str],
        subj_types: list[str],
    ) -> ReasoningResult:
        """Build the explanation for why subject depends on object."""
        connection_rels = (
            RelationType.IS_A,
            RelationType.RELATED_TO,
            RelationType.SIMILAR_TO,
        )
        parts: list[str] = []
        knowledge: list[tuple[str, str, float]] = []
        parts.append(f"{subject} depends on {object_}")
        knowledge.append(("depends_on", object_, 0.8))

        if connecting_concepts:
            reason = connecting_concepts[0]
            reason_id = self.network._resolve(reason) or reason.lower()
            subj_type_str = ""
            for st in subj_types:
                st_id = self.network._resolve(st) or st.lower()
                if st_id == reason_id:
                    subj_type_str = st
                    break
                if self._concepts_connected(reason_id, st_id, connection_rels):
                    subj_type_str = st
                    break
                if self._concepts_connected(st_id, reason_id, connection_rels):
                    subj_type_str = st
                    break

            if subj_type_str:
                parts.append(
                    f"because {object_} enables {reason}, and {subject} is a kind of "
                    f"{subj_type_str}"
                )
                knowledge.append(("enables", reason, 0.8))
                knowledge.append(("is_a", subj_type_str, 0.75))
            else:
                parts.append(f"because {object_} enables {reason}")
                knowledge.append(("enables", reason, 0.8))
        elif enabled:
            parts.append(f"because {object_} enables {', '.join(enabled[:3])}")
            for e in enabled[:3]:
                knowledge.append(("enables", e, 0.8))
        else:
            obj_def = obj_concept.properties.get("definition", "")
            if obj_def:
                parts.append(f"and {object_} is {obj_def}")
                knowledge.append(("is", obj_def, 0.7))

        obj_def = obj_concept.properties.get("definition", "")
        if obj_def and not any(obj_def in p for p in parts):
            parts.append(f"{object_} is {obj_def}")
            knowledge.append(("is", obj_def, 0.7))

        return ReasoningResult(
            conclusion=". ".join(parts),
            reasoning_type=ReasoningType.CAUSAL,
            evidence=[f"why({subj_id}, {obj_id})"],
            confidence=0.8,
            knowledge=knowledge,
        )

    def explain_why(self, subject: str, object_: str) -> ReasoningResult | None:
        """Explain why a subject needs/depends on an object.

        Traces the causal chain: subject depends_on object → object
        enables X → subject is_a X (or subject relates to X). This
        produces answers like "humans need water because water enables
        life, and humans are living things."
        """
        subj_id = self.network._resolve(subject) or subject.lower()
        obj_id = self.network._resolve(object_) or object_.lower()

        subj_concept = self.network.get_concept(subj_id)
        obj_concept = self.network.get_concept(obj_id)
        if not subj_concept or not obj_concept:
            return None

        # 1. Verify the dependency exists
        dep_edges = self._find_why_dependency(subj_id, obj_id)
        if not dep_edges:
            return None

        # 2. Find what the object enables
        enabled = list(
            dict.fromkeys(
                e.target
                for e in self.network.get_edges(obj_id, "out")
                if e.relation == RelationType.ENABLES
            )
        )

        # 3. Find what the subject is (IS_A chain)
        subj_types = list(
            dict.fromkeys(
                e.target
                for e in self.network.get_edges(subj_id, "out")
                if e.relation == RelationType.IS_A
            )
        )

        # 4. Find connections: does the subject's type match what the
        # object enables? This is the "because" link.
        connecting_concepts = self._find_connecting_concepts(enabled, subj_types)

        # Build the explanation
        return self._build_why_explanation(
            subject, object_, subj_id, obj_id, obj_concept,
            enabled, connecting_concepts, subj_types,
        )


# ═══════════════════════════════════════════════════════════════════════
# Tier 3: Advanced Reasoning Modes
# ═══════════════════════════════════════════════════════════════════════


class ReasoningStrategy(Enum):
    """Available reasoning strategies for metareasoning selection.

    These correspond to the different reasoning modes that the engine
    can deploy. Metareasoning selects among them based on problem
    characteristics and past effectiveness.
    """

    DEDUCTIVE = "deductive"  # transitive chains
    ABDUCTIVE = "abductive"  # inference to best explanation
    ANALOGICAL = "analogical"  # transfer by similarity
    CAUSAL = "causal"  # cause-effect chains
    PROBABILISTIC = "probabilistic"  # Bayesian updating
    TEMPORAL = "temporal"  # Allen's interval algebra
    COUNTERFACTUAL = "counterfactual"  # what-if / do-calculus
    SYNTHESIS = "synthesis"  # combining multiple sources


# ─── Probabilistic Reasoning ────────────────────────────────────────


@dataclass(slots=True)
class BetaDistribution:
    """A Beta distribution representing a belief as a probability
    distribution rather than a point estimate.

    The Beta distribution is the conjugate prior for the Bernoulli
    and Binomial distributions, making it ideal for Bayesian updating
    of binary beliefs. Alpha and beta are the shape parameters:
    - alpha = number of positive observations + prior
    - beta = number of negative observations + prior
    - mean = alpha / (alpha + beta)
    - variance = alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1))

    A uniform prior (alpha=1, beta=1) represents maximum uncertainty.
    """

    alpha: float = 1.0  # positive evidence count + prior
    beta: float = 1.0  # negative evidence count + prior

    def __post_init__(self) -> None:
        """Validate that both Beta shape parameters are finite and positive."""
        if not all(math.isfinite(value) and value > 0.0 for value in (self.alpha, self.beta)):
            raise ValueError("Beta shape parameters must be finite and positive")

    @staticmethod
    def _validate_weight(weight: float) -> None:
        """Reject non-finite or negative evidence weight."""
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("Evidence weight must be finite and non-negative")

    @property
    def mean(self) -> float:
        """Expected value (point estimate) of the distribution."""
        scale = max(self.alpha, self.beta)
        alpha, beta = self.alpha / scale, self.beta / scale
        return alpha / (alpha + beta)

    @property
    def variance(self) -> float:
        """Variance of the distribution (uncertainty)."""
        scale = max(self.alpha, self.beta, 1.0)
        alpha, beta = self.alpha / scale, self.beta / scale
        total = alpha + beta
        return (alpha / total) * (beta / total) / (total + 1.0 / scale) / scale

    @property
    def confidence(self) -> float:
        """Confidence as 1 - normalized variance (0..1).

        High confidence means low variance (the distribution is
        concentrated). Low confidence means high variance (the
        distribution is spread out).
        """
        # Maximum variance for a Beta(1,1) is 1/12 ≈ 0.0833
        max_var = 1.0 / 12.0
        return 1.0 - min(1.0, self.variance / max_var) if max_var > 0 else 0.5

    def update(self, positive: bool, weight: float = 1.0) -> None:
        """Bayesian update: observe evidence and update the distribution.

        Args:
            positive: True if the evidence supports the belief,
                      False if it contradicts.
            weight: Strength of the evidence (default 1.0).
        """
        self._validate_weight(weight)
        updated = (self.alpha if positive else self.beta) + weight
        if not math.isfinite(updated):
            raise ValueError("Evidence update would overflow the belief")
        if positive:
            self.alpha = updated
        else:
            self.beta = updated

    def describe(self) -> str:
        """Human-readable description of the distribution."""
        return (
            f"Beta(alpha={self.alpha:.2f}, beta={self.beta:.2f}), "
            f"mean={self.mean:.3f}, confidence={self.confidence:.3f}"
        )


class ProbabilisticReasoning:
    """Probabilistic reasoning using Bayesian belief updating.

    Instead of treating confidence as a point estimate, this class
    represents each belief as a Beta distribution. When evidence is
    observed, the distribution is updated using Bayesian updating
    (conjugate prior for Bernoulli/Binomial likelihoods).

    This handles uncertain relationships with probabilistic edges —
    instead of a binary "A causes B", we track how much evidence
    supports that claim.
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Initialize the probabilistic reasoning engine.

        Args:
            network: The concept network to reason over.
        """
        self.network = network
        # Beliefs are keyed by (source, relation, target) tuples
        self._beliefs: dict[tuple[str, str, str], BetaDistribution] = {}

    def _belief_key(self, source: str, relation: str, target: str) -> tuple[str, str, str]:
        """Create a normalized belief key."""
        return (
            self.network._resolve(source) or source,
            relation,
            self.network._resolve(target) or target,
        )

    def get_belief(self, source: str, relation: str, target: str) -> BetaDistribution:
        """Get the Beta distribution for a belief.

        If the belief hasn't been explicitly tracked, initialize it
        from the edge weight in the concept network (if an edge exists),
        or with a uniform prior (alpha=1, beta=1).

        Args:
            source: Source concept name.
            relation: Relation type string (e.g., "causes").
            target: Target concept name.

        Returns:
            The BetaDistribution representing the belief.
        """
        key = self._belief_key(source, relation, target)
        if key not in self._beliefs:
            # Try to initialize from edge weight
            try:
                rel_type = RelationType(relation)
            except ValueError:
                rel_type = None

            if rel_type is not None:
                edges = self.network.get_edges(key[0], "out")
                for edge in edges:
                    if edge.target == key[2] and edge.relation == rel_type:
                        # Convert edge weight to pseudo-counts
                        # weight=0.5 → alpha=1, beta=1 (uniform)
                        # weight=0.9 → alpha=9, beta=1
                        w = edge.weight
                        self._beliefs[key] = BetaDistribution(
                            alpha=max(0.1, w * 10),
                            beta=max(0.1, (1.0 - w) * 10),
                        )
                        return self._beliefs[key]

            # No edge found — uniform prior
            self._beliefs[key] = BetaDistribution()
        return self._beliefs[key]

    def observe_evidence(
        self,
        source: str,
        relation: str,
        target: str,
        positive: bool,
        weight: float = 1.0,
    ) -> BetaDistribution:
        """Observe evidence and update the belief distribution.

        Args:
            source: Source concept name.
            relation: Relation type string.
            target: Target concept name.
            positive: True if evidence supports the relationship.
            weight: Strength of the evidence.

        Returns:
            The updated BetaDistribution.
        """
        belief = self.get_belief(source, relation, target)
        belief.update(positive, weight)
        return belief

    def probabilistic_infer(
        self,
        prior: tuple[str, str, str],
        evidence: list[tuple[str, str, str, bool, float]],
    ) -> ReasoningResult:
        """Infer a conclusion from a prior belief and observed evidence.

        Updates the prior belief with each piece of evidence using
        Bayesian updating, then returns a ReasoningResult with the
        posterior distribution.

        Args:
            prior: A (source, relation, target) tuple for the prior belief.
            evidence: A list of (source, relation, target, positive, weight)
                      tuples representing observed evidence.

        Returns:
            A ReasoningResult with the posterior mean as confidence and
            the distribution parameters in the evidence chain.
        """
        source, relation, target = prior
        belief = self.get_belief(source, relation, target)

        evidence_chain: list[str] = [f"Prior: {source} {relation} {target} — {belief.describe()}"]

        for src, rel, tgt, pos, wt in evidence:
            ev_belief = self.observe_evidence(src, rel, tgt, pos, wt)
            evidence_chain.append(
                f"Evidence: {src} {rel} {tgt} "
                f"({'supports' if pos else 'contradicts'}, w={wt:.2f}) "
                f"→ {ev_belief.describe()}"
            )

        conclusion = (
            f"{source} {relation} {target} with probability "
            f"{belief.mean:.3f} (confidence: {belief.confidence:.3f})"
        )

        return ReasoningResult(
            conclusion=conclusion,
            reasoning_type=ReasoningType.DEDUCTIVE,
            evidence=evidence_chain,
            confidence=belief.confidence,
        )


# ─── Temporal Reasoning ─────────────────────────────────────────────


class TemporalRelation(Enum):
    """Allen's interval algebra — 13 temporal relations between intervals.

    These 13 relations form a jointly exhaustive and pairwise disjoint
    (JEPD) set: any two intervals must be in exactly one of these
    relations. They are the foundation of qualitative temporal reasoning.

    References:
        - Allen (1983): Maintaining knowledge about temporal intervals
    """

    BEFORE = "before"  # A ends before B starts
    AFTER = "after"  # A starts after B ends
    DURING = "during"  # A is contained within B
    CONTAINS = "contains"  # B is contained within A
    OVERLAPS = "overlaps"  # A starts before B, they overlap
    OVERLAPPED_BY = "overlapped_by"  # B starts before A, they overlap
    MEETS = "meets"  # A ends exactly when B starts
    MET_BY = "met_by"  # A starts exactly when B ends
    STARTS = "starts"  # A starts with B but ends earlier
    STARTED_BY = "started_by"  # B starts with A but ends earlier
    FINISHES = "finishes"  # A ends with B but starts later
    FINISHED_BY = "finished_by"  # B ends with A but starts later
    EQUAL = "equal"  # A and B are the same interval


@dataclass(slots=True)
class TimeInterval:
    """A temporal interval with a start and end time.

    Attributes:
        start_time: The start time of the interval.
        end_time: The end time of the interval (must be >= start_time).
        label: Optional label for the event this interval represents.
    """

    start_time: float
    end_time: float
    label: str = ""

    def __post_init__(self) -> None:
        """Clamp end_time to start_time if it was set earlier."""
        if self.end_time < self.start_time:
            self.end_time = self.start_time

    @property
    def duration(self) -> float:
        """Duration of the interval."""
        return self.end_time - self.start_time


class TemporalReasoning:
    """Temporal reasoning using Allen's interval algebra.

    Implements the 13 temporal relations from Allen (1983) for
    reasoning about the temporal ordering of events. Supports event
    sequencing, temporal order queries, and transitive reasoning
    about temporal relationships.
    """

    def __init__(self, network: ConceptNetwork | None = None) -> None:
        """Initialize the temporal reasoning engine.

        Args:
            network: Optional concept network for tracking temporal
                     relationships between concepts.
        """
        self.network = network
        # Track intervals for events by name
        self._intervals: dict[str, TimeInterval] = {}

    def add_event(
        self,
        name: str,
        start_time: float,
        end_time: float,
    ) -> TimeInterval:
        """Register a temporal interval for an event.

        Args:
            name: The event name (concept ID).
            start_time: Start time of the event.
            end_time: End time of the event.

        Returns:
            The created TimeInterval.
        """
        interval = TimeInterval(
            start_time=start_time,
            end_time=end_time,
            label=name,
        )
        self._intervals[name] = interval
        return interval

    def get_interval(self, name: str) -> TimeInterval | None:
        """Get the temporal interval for an event.

        Args:
            name: The event name.

        Returns:
            The TimeInterval, or None if not registered.
        """
        return self._intervals.get(name)

    def temporal_infer(self, event_a: str, event_b: str) -> TemporalRelation:
        """Determine the temporal relation between two events.

        Uses Allen's interval algebra to classify the relationship
        between the two event intervals.

        Args:
            event_a: Name of the first event.
            event_b: Name of the second event.

        Returns:
            The TemporalRelation between event_a and event_b.
        """
        interval_a = self._intervals.get(event_a)
        interval_b = self._intervals.get(event_b)

        if interval_a is None or interval_b is None:
            # If we have a concept network, try to use TEMPORAL_ORDER edges
            if self.network is not None:
                for edge in self.network.get_edges(event_a, "both"):
                    if edge.relation != RelationType.TEMPORAL_ORDER:
                        continue
                    if edge.source == event_a and edge.target == event_b:
                        return TemporalRelation.BEFORE
                    if edge.source == event_b and edge.target == event_a:
                        return TemporalRelation.AFTER
            return TemporalRelation.EQUAL

        return self._classify_relation(interval_a, interval_b)

    @staticmethod
    def _classify_relation(a: TimeInterval, b: TimeInterval) -> TemporalRelation:
        """Classify the Allen relation between two intervals.

        Uses exact comparison with a small epsilon for floating-point
        tolerance.

        Args:
            a: The first interval.
            b: The second interval.

        Returns:
            The TemporalRelation from a's perspective.
        """
        eps = 1e-9

        a_start, a_end = a.start_time, a.end_time
        b_start, b_end = b.start_time, b.end_time

        # Equal
        if abs(a_start - b_start) < eps and abs(a_end - b_end) < eps:
            return TemporalRelation.EQUAL

        # Before / After
        if a_end < b_start - eps:
            return TemporalRelation.BEFORE
        if b_end < a_start - eps:
            return TemporalRelation.AFTER

        # Meets / Met_by
        if abs(a_end - b_start) < eps and a_start < b_start - eps:
            return TemporalRelation.MEETS
        if abs(b_end - a_start) < eps and b_start < a_start - eps:
            return TemporalRelation.MET_BY

        # During / Contains
        if a_start > b_start + eps and a_end < b_end - eps:
            return TemporalRelation.DURING
        if b_start > a_start + eps and b_end < a_end - eps:
            return TemporalRelation.CONTAINS

        # Overlaps / Overlapped_by
        if a_start < b_start - eps and a_end > b_start + eps and a_end < b_end - eps:
            return TemporalRelation.OVERLAPS
        if b_start < a_start - eps and b_end > a_start + eps and b_end < a_end - eps:
            return TemporalRelation.OVERLAPPED_BY

        # Starts / Started_by
        if abs(a_start - b_start) < eps and a_end < b_end - eps:
            return TemporalRelation.STARTS
        if abs(a_start - b_start) < eps and b_end < a_end - eps:
            return TemporalRelation.STARTED_BY

        # Finishes / Finished_by
        if abs(a_end - b_end) < eps and a_start > b_start + eps:
            return TemporalRelation.FINISHES
        if abs(a_end - b_end) < eps and b_start > a_start + eps:
            return TemporalRelation.FINISHED_BY

        # Fallback (shouldn't happen with valid intervals)
        return TemporalRelation.EQUAL

    def get_event_sequence(self, events: list[str]) -> list[str]:
        """Sort events by their start times (temporal ordering).

        Args:
            events: List of event names to sort.

        Returns:
            Events sorted from earliest to latest.
        """
        scored: list[tuple[float, str]] = []
        for name in events:
            interval = self._intervals.get(name)
            if interval is not None:
                scored.append((interval.start_time, name))
            else:
                scored.append((float("inf"), name))
        scored.sort(key=lambda x: x[0])
        return [name for _, name in scored]

    def query_before(self, event: str) -> list[str]:
        """Find all events that occur before the given event.

        Args:
            event: The reference event name.

        Returns:
            List of event names that end before the given event starts.
        """
        ref = self._intervals.get(event)
        if ref is None:
            return []
        result: list[str] = []
        for name, interval in self._intervals.items():
            if name == event:
                continue
            if self._classify_relation(interval, ref) == TemporalRelation.BEFORE:
                result.append(name)
        return result

    def query_after(self, event: str) -> list[str]:
        """Find all events that occur after the given event.

        Args:
            event: The reference event name.

        Returns:
            List of event names that start after the given event ends.
        """
        ref = self._intervals.get(event)
        if ref is None:
            return []
        result: list[str] = []
        for name, interval in self._intervals.items():
            if name == event:
                continue
            if self._classify_relation(interval, ref) == TemporalRelation.AFTER:
                result.append(name)
        return result

    def query_during(self, event: str) -> list[str]:
        """Find all events that occur during the given event.

        Args:
            event: The reference event name.

        Returns:
            List of event names that are contained within the given event.
        """
        ref = self._intervals.get(event)
        if ref is None:
            return []
        result: list[str] = []
        for name, interval in self._intervals.items():
            if name == event:
                continue
            if self._classify_relation(interval, ref) == TemporalRelation.DURING:
                result.append(name)
        return result


# ─── Counterfactual Reasoning ───────────────────────────────────────


class CounterfactualReasoning:
    """Counterfactual reasoning — "what if" scenario generation.

    Implements intervention modeling using Pearl's do-calculus (Pearl,
    2009). Given a factual observation and a hypothetical intervention,
    this class reasons about what would happen if the intervention
    were applied.

    The key insight of do-calculus is that intervening (do(X=x)) is
    different from observing (seeing X=x). Observation updates our
    beliefs about causes of X; intervention cuts the causal links
    into X and sets it to a fixed value.

    Causal relationships are tracked from the concept network using
    CAUSES and ENABLES relation types.
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Initialize the counterfactual reasoning engine.

        Args:
            network: The concept network containing causal relationships.
        """
        self.network = network

    def _get_causal_children(self, concept: str) -> list[tuple[str, float, RelationType]]:
        """Get concepts that are caused or enabled by the given concept.

        Args:
            concept: The source concept name.

        Returns:
            List of (target, weight, relation) tuples for causal descendants.
        """
        cid = self.network._resolve(concept)
        if not cid:
            return []
        children: list[tuple[str, float, RelationType]] = []
        for edge in self.network.get_edges(cid, "out"):
            if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO, RelationType.ENABLES):
                children.append((edge.target, edge.weight, edge.relation))
        return children

    def _trace_counterfactual_effects(
        self,
        intervention: str,
        depth: int,
        affected: list[tuple[str, float, int]],
        visited: set[str],
        evidence: list[str],
    ) -> None:
        """Trace downstream effects of an intervention through the causal graph."""

        def trace(concept: str, prob: float, d: int) -> None:
            """Recursively propagate an intervention's probability through causal children."""
            if d >= depth or prob < 0.05:
                return
            if concept in visited:
                return
            visited.add(concept)

            children = self._get_causal_children(concept)
            for target, weight, relation in children:
                child_prob = prob * weight
                affected.append((target, child_prob, d + 1))
                rel_str = relation.value.replace("_", " ")
                evidence.append(
                    f"  {'  ' * d}{concept} {rel_str} {target} (w={weight:.2f}, p={child_prob:.3f})"
                )
                trace(target, child_prob, d + 1)

        # Trace from the intervention point
        trace(intervention, 1.0, 0)

    def _find_removed_effects(self, fact: str, visited: set[str]) -> list[str]:
        """Find effects of the fact that no longer occur under intervention."""
        fact_children = self._get_causal_children(fact)
        removed_effects: list[str] = []
        for target, weight, relation in fact_children:
            if target not in visited:
                rel_str = relation.value.replace("_", " ")
                removed_effects.append(
                    f"{fact} {rel_str} {target} (w={weight:.2f}) — no longer occurs"
                )
        return removed_effects

    def _build_counterfactual_conclusion(
        self,
        fact: str,
        intervention: str,
        affected: list[tuple[str, float, int]],
        removed_effects: list[str],
    ) -> tuple[str, float]:
        """Build the conclusion and confidence for a counterfactual."""
        if affected:
            top_effects = sorted(affected, key=lambda x: -x[1])[:3]
            effect_str = ", ".join(f"{c} (p={p:.2f})" for c, p, _ in top_effects)
            conclusion = (
                f"If we intervene to set {intervention} instead of {fact}, then: {effect_str}"
            )
        elif removed_effects:
            conclusion = (
                f"If we intervene to set {intervention} instead of {fact}, "
                f"the effects of {fact} would not occur"
            )
        else:
            conclusion = (
                f"If we intervene to set {intervention} instead of {fact}, "
                f"no significant downstream changes are predicted"
            )

        # Confidence is based on the strength of causal links traced
        max_prob = max((p for _, p, _ in affected), default=0.0)
        confidence = min(1.0, max_prob * 0.8 + 0.1)
        return conclusion, confidence

    def counterfactual(
        self,
        fact: str,
        intervention: str,
        depth: int = 3,
    ) -> ReasoningResult:
        """Generate a counterfactual: "if we change X to Y, what would happen?"

        Uses Pearl's do-calculus: the intervention cuts incoming causal
        links to the intervened variable and sets it to the new value.
        We then trace the downstream effects through the causal graph.

        Args:
            fact: The factual observation (concept name that is true).
            intervention: The hypothetical intervention (concept name
                          to set instead). This replaces `fact` in the
                          causal graph.

        Returns:
            A ReasoningResult describing what would happen under the
            intervention.
        """
        evidence: list[str] = []
        evidence.append(f"Fact: {fact} is observed")
        evidence.append(f"Intervention: do({intervention}) — cutting causal links to {fact}")

        # Under do(intervention), the intervened variable's causes are
        # disconnected. We trace downstream effects of the intervention.
        affected: list[tuple[str, float, int]] = []
        visited: set[str] = set()

        self._trace_counterfactual_effects(intervention, depth, affected, visited, evidence)

        # Also check what effects of the fact are NO LONGER present
        removed_effects = self._find_removed_effects(fact, visited)

        if removed_effects:
            evidence.append("Effects that no longer occur:")
            for eff in removed_effects:
                evidence.append(f"  {eff}")

        # Build conclusion
        conclusion, confidence = self._build_counterfactual_conclusion(
            fact, intervention, affected, removed_effects
        )

        return ReasoningResult(
            conclusion=conclusion,
            reasoning_type=ReasoningType.CAUSAL,
            evidence=evidence,
            confidence=confidence,
            novel=True,
        )


# ─── Metareasoning ──────────────────────────────────────────────────


@dataclass(slots=True)
class ReasoningRecord:
    """A record of a reasoning attempt for metareasoning learning.

    Tracks which strategy was used, the problem characteristics, and
    the outcome quality. This lets the metareasoning engine learn
    which strategies work best for which types of problems.
    """

    strategy: ReasoningStrategy
    problem: str
    confidence: float  # confidence of the result
    success: bool  # was the result useful?
    timestamp: float = 0.0


class MetaReasoning:
    """Metareasoning — reasoning about reasoning.

    This class evaluates which reasoning strategy is best for a given
    problem, based on past effectiveness. It tracks a reasoning history
    and learns which strategies produce good results for different
    problem types.

    Key features:
    - Strategy selection: picks the best strategy for a problem
    - Depth adjustment: high confidence → shallow, low confidence → deep
    - History tracking: learns from past reasoning attempts
    """

    def __init__(self, network: ConceptNetwork | None = None) -> None:
        """Initialize the metareasoning engine.

        Args:
            network: Optional concept network for problem analysis.
        """
        self.network = network
        self.reasoning_history: list[ReasoningRecord] = []
        # Strategy effectiveness scores (learned from history)
        self._strategy_scores: dict[ReasoningStrategy, float] = dict.fromkeys(
            ReasoningStrategy, 0.5
        )
        # Problem type → strategy preferences
        self._problem_preferences: dict[str, dict[ReasoningStrategy, float]] = {}

    def meta_reason(
        self,
        problem: str,
        available_strategies: list[ReasoningStrategy],
        confidence: float = 0.5,
    ) -> ReasoningStrategy:
        """Select the best reasoning strategy for a given problem.

        The selection is based on:
        1. Problem type matching (does the problem involve causality,
           similarity, temporal ordering, etc.?)
        2. Past effectiveness of each strategy for similar problems
        3. Current confidence level (high confidence → simpler strategies)

        Args:
            problem: The problem description or concept name.
            available_strategies: Strategies that are applicable.
            confidence: Current confidence in the problem understanding
                        (0..1). High confidence → shallow strategies,
                        low confidence → deep strategies.

        Returns:
            The recommended ReasoningStrategy.
        """
        if not available_strategies:
            return ReasoningStrategy.DEDUCTIVE

        # Classify the problem type
        problem_type = self._classify_problem(problem)

        # Score each available strategy
        scores: list[tuple[float, ReasoningStrategy]] = []
        for strategy in available_strategies:
            # Base score from learned effectiveness
            base_score = self._strategy_scores.get(strategy, 0.5)

            # Problem-type preference
            prefs = self._problem_preferences.get(problem_type, {})
            type_bonus = prefs.get(strategy, 0.0)

            # Confidence adjustment: high confidence favors simpler
            # strategies (deductive, analogical), low confidence favors
            # deeper strategies (causal, counterfactual, probabilistic)
            confidence_adj = 0.0
            if confidence > 0.7:
                if strategy in (
                    ReasoningStrategy.DEDUCTIVE,
                    ReasoningStrategy.ANALOGICAL,
                    ReasoningStrategy.SYNTHESIS,
                ):
                    confidence_adj = 0.1
            elif confidence < 0.3:
                if strategy in (
                    ReasoningStrategy.CAUSAL,
                    ReasoningStrategy.COUNTERFACTUAL,
                    ReasoningStrategy.PROBABILISTIC,
                    ReasoningStrategy.TEMPORAL,
                ):
                    confidence_adj = 0.1

            total = base_score + type_bonus + confidence_adj
            scores.append((total, strategy))

        # Pick the highest-scoring strategy
        scores.sort(key=lambda x: -x[0])
        return scores[0][1]

    def _classify_problem(self, problem: str) -> str:
        """Classify a problem into a type for strategy matching.

        Args:
            problem: The problem description or concept name.

        Returns:
            A problem type string (e.g., "causal", "temporal", "similarity").
        """
        problem_lower = problem.lower()

        # Check for causal keywords
        causal_words = {"cause", "effect", "leads to", "results in", "because"}
        if any(w in problem_lower for w in causal_words):
            return "causal"

        # Check for temporal keywords
        temporal_words = {"before", "after", "during", "while", "when", "sequence", "order"}
        if any(w in problem_lower for w in temporal_words):
            return "temporal"

        # Check for similarity keywords
        similarity_words = {"similar", "like", "analog", "compare", "resembl"}
        if any(w in problem_lower for w in similarity_words):
            return "similarity"

        # Check for uncertainty keywords
        uncertainty_words = {"maybe", "probably", "likely", "uncertain", "probability"}
        if any(w in problem_lower for w in uncertainty_words):
            return "uncertain"

        # Check for counterfactual keywords
        counterfactual_words = {"what if", "suppose", "imagine", "instead", "alternative"}
        if any(w in problem_lower for w in counterfactual_words):
            return "counterfactual"

        # Check the concept network for relation types
        if self.network is not None:
            concept = self.network.get_concept(problem)
            if concept is not None:
                for edge in self.network.get_edges(concept.id, "both"):
                    if edge.relation in (RelationType.CAUSES, RelationType.LEADS_TO):
                        return "causal"
                    if edge.relation == RelationType.TEMPORAL_ORDER:
                        return "temporal"
                    if edge.relation == RelationType.SIMILAR_TO:
                        return "similarity"

        return "general"

    def record_result(
        self,
        strategy: ReasoningStrategy,
        problem: str,
        confidence: float,
        success: bool,
    ) -> None:
        """Record a reasoning attempt for future learning.

        Args:
            strategy: The strategy that was used.
            problem: The problem that was addressed.
            confidence: The confidence of the result.
            success: Whether the result was useful.
        """
        import time as _time

        record = ReasoningRecord(
            strategy=strategy,
            problem=problem,
            confidence=confidence,
            success=success,
            timestamp=_time.time(),
        )
        self.reasoning_history.append(record)
        # Bound the history — this is a long-running daemon and an
        # unbounded list leaks memory and slows get_strategy_stats'
        # per-record scans. Aggregate stats are kept in
        # _strategy_scores / _problem_preferences, so dropping old
        # records loses no learned information.
        if len(self.reasoning_history) > 1000:
            del self.reasoning_history[: len(self.reasoning_history) - 1000]

        # Update strategy effectiveness score using exponential moving average
        alpha = 0.1  # learning rate
        reward = 1.0 if success else 0.0
        current = self._strategy_scores.get(strategy, 0.5)
        self._strategy_scores[strategy] = current * (1 - alpha) + reward * alpha

        # Update problem-type preferences
        problem_type = self._classify_problem(problem)
        if problem_type not in self._problem_preferences:
            self._problem_preferences[problem_type] = {}
        prefs = self._problem_preferences[problem_type]
        current_pref = prefs.get(strategy, 0.0)
        prefs[strategy] = current_pref * (1 - alpha) + reward * alpha * 0.5

    def recommend_depth(self, confidence: float) -> int:
        """Recommend reasoning depth based on confidence.

        High confidence → shallow reasoning (depth 1-2).
        Low confidence → deep reasoning (depth 4-5).

        Args:
            confidence: Current confidence level (0..1).

        Returns:
            Recommended reasoning depth (1-5).
        """
        if confidence > 0.8:
            return 1
        if confidence > 0.6:
            return 2
        if confidence > 0.4:
            return 3
        if confidence > 0.2:
            return 4
        return 5

    def get_strategy_stats(self) -> dict[str, Any]:
        """Get statistics about strategy effectiveness.

        Returns:
            A dict with per-strategy success rates and usage counts.
        """
        stats: dict[str, Any] = {}
        for strategy in ReasoningStrategy:
            records = [r for r in self.reasoning_history if r.strategy == strategy]
            total = len(records)
            successes = sum(1 for r in records if r.success)
            stats[strategy.value] = {
                "uses": total,
                "successes": successes,
                "success_rate": successes / total if total > 0 else 0.0,
                "learned_score": self._strategy_scores.get(strategy, 0.5),
            }
        return stats
