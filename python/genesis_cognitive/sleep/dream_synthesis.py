"""Sleep-driven synthesis — dreaming that produces real insight.

During REM sleep (high acetylcholine, low norepinephrine), Genesis enters
a "synthesis" dream mode that goes beyond random association. She finds
concepts that are **structurally similar but far apart** in the graph,
proposes **novel typed edges** between them, and stores these as
hypothetical edges with low confidence.

On the next wake cycle (or a dedicated "dream review" stage), a
validation pass runs abductive/analogical inference over each
hypothetical edge. If the edge is supported by independent evidence
(>= 2 corroborating paths), it is promoted to a real edge with raised
confidence. If contradicted, it is dropped. If neutral, it remains
hypothetical for re-evaluation next sleep.

This is the dream -> novel connection -> test -> consolidate loop.
Validated dream edges are "insights" — they are the mechanism that
should drive analogical reasoning improvement over time.

## Structural similarity

Two concepts are structurally similar if they occupy similar roles in
the network, even if they're in different domains:

- **Same hub distance**: both are leaves of the same IS_A hub, or both
  are mid-level nodes at the same depth from a hub.
- **Same relation-type signature**: both have outgoing CAUSES edges, or
  both have incoming DEPENDS_ON edges, etc.

The relation-type signature is a frozenset of (relation, direction)
pairs. Two concepts with the same signature play similar structural
roles — e.g. two "causers" (both have outgoing CAUSES edges) or two
"enablers" (both have outgoing ENABLES edges).

## Typed edge prediction

When two structurally similar concepts are found, the relation type for
the proposed edge is predicted from the shared structural signature:

- Two leaves of the same IS_A hub -> ``SIMILAR_TO``
- Two nodes in the same CAUSES chain position -> ``CORRELATED_WITH``
  (mapped to ``RELATED_TO`` since there's no CORRELATED_WITH type)
- Two nodes with the same ENABLES signature -> ``SIMILAR_TO``
- Default: ``RELATED_TO``

## Validation

A hypothetical edge A -> B (relation R) is validated by checking whether
independent evidence in the network supports it:

1. **Analogical transfer**: if A similar_to C, and C R B, then the edge
   is supported by analogy.
2. **Transitive chain**: if A R C and C R B (for transitive relations
   like IS_A, CAUSES), the edge is supported by transitivity.
3. **Shared context**: if both A and B are connected to a common hub H
   via the same relation, the edge is plausibly supported.

An edge needs >= 2 corroborating paths to be promoted. Contradictions
(an existing OPPOSITE_OF or CONTRADICTS edge) cause it to be dropped.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..brain_waves import BrainWave, BrainWaveState
from ..concepts import ConceptNetwork, RelationType

if TYPE_CHECKING:
    from ..reasoning import ReasoningEngine

__all__ = [
    "DreamInsight",
    "DreamProposal",
    "DreamSynthesisEngine",
    "DreamValidationResult",
]

logger = logging.getLogger(__name__)

# Relations that are transitive (A R C and C R B implies A R B)
_TRANSITIVE_RELATIONS: frozenset[RelationType] = frozenset({
    RelationType.IS_A,
    RelationType.PART_OF,
    RelationType.CAUSES,
    RelationType.DEPENDS_ON,
})

# Relations that indicate contradiction
_CONTRADICTION_RELATIONS: frozenset[RelationType] = frozenset({
    RelationType.OPPOSITE_OF,
    RelationType.CONTRADICTS,
    RelationType.PREVENTS,
    RelationType.HARMS,
})

# Minimum number of corroborating paths needed to promote a hypothetical edge
_MIN_CORROBORATIONS = 2

# Confidence for hypothetical dream edges
_HYPOTHETICAL_WEIGHT = 0.15
_PROMOTED_WEIGHT = 0.45

# Origin tag for dream-synthesized edges
_ORIGIN_DREAM_SYNTHESIS = "dream-synthesis"
_ORIGIN_DREAM_VALIDATED = "dream-validated"


@dataclass(slots=True)
class DreamProposal:
    """A hypothetical edge proposed during dream synthesis."""

    source: str
    target: str
    relation: RelationType
    structural_similarity: float  # 0..1, how similar the structural roles are
    predicted_from: str  # what structural signal predicted this relation
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def key(self) -> tuple[str, str, RelationType]:
        """Unique identifier tuple of (source, target, relation) for deduplication."""
        return (self.source, self.target, self.relation)


@dataclass(slots=True)
class DreamValidationResult:
    """Result of validating a dream proposal."""

    proposal: DreamProposal
    corroborations: int
    contradicted: bool
    promoted: bool
    reason: str


@dataclass(slots=True)
class DreamInsight:
    """A dream-synthesized edge that survived validation."""

    source: str
    target: str
    relation: RelationType
    corroborations: int
    insight_text: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def describe(self) -> str:
        """Return a human-readable summary of the validated dream insight."""
        return (
            f"Dream insight: {self.source} {self.relation.value} {self.target} "
            f"({self.corroborations} corroborating paths)"
        )


class DreamSynthesisEngine:
    """Sleep-driven structural synthesis and validation.

    During REM sleep, this engine finds structurally similar but
    unconnected concepts and proposes novel typed edges between them.
    During wake (or a dream-review stage), it validates those proposals
    using the reasoning engine and promotes or drops them.

    Usage::

        engine = DreamSynthesisEngine(network)
        proposals = engine.synthesize(max_proposals=5)
        results = engine.validate(proposals)
        insights = [r for r in results if r.promoted]
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Initialize the engine with a concept network and empty insight tracking."""
        self.network = network
        self._validated_insights: list[DreamInsight] = []
        self._rejected_count = 0
        self._proposed_count = 0

    @property
    def insight_count(self) -> int:
        """Number of validated dream insights."""
        return len(self._validated_insights)

    @property
    def proposed_count(self) -> int:
        """Total proposals ever generated."""
        return self._proposed_count

    @property
    def rejected_count(self) -> int:
        """Total proposals rejected by validation."""
        return self._rejected_count

    @property
    def insights(self) -> list[DreamInsight]:
        """All validated dream insights (copy)."""
        return list(self._validated_insights)

    # ─── Synthesis (during REM sleep) ────────────────────────────

    def synthesize(
        self,
        max_proposals: int = 5,
        brain_waves: BrainWaveState | None = None,
    ) -> list[DreamProposal]:
        """Find structurally similar unconnected concepts and propose edges.

        This is the creative step — it runs during REM sleep. It:

        1. Computes the structural signature of each concept (relation
           types + directions + hub distance).
        2. Groups concepts by signature.
        3. For each group, finds pairs that are **not currently connected**.
        4. Proposes a typed edge based on the shared structural role.

        Args:
            max_proposals: Maximum number of proposals to generate.
            brain_waves: Optional brain wave state. REM sleep is
                theta-dominant — when theta is dominant, more proposals
                are generated (the brain is in associative synthesis
                mode). When delta is dominant (NREM deep sleep), fewer
                proposals are generated (the brain is in replay mode,
                not synthesis mode).

        Returns:
            List of dream proposals (hypothetical edges).
        """
        # Brain-wave-modulated proposal count.
        # REM = theta-dominant → more synthesis (associative binding).
        # NREM deep = delta-dominant → less synthesis (replay only).
        effective_max = max_proposals
        if brain_waves is not None:
            dom = brain_waves.dominant
            if dom == BrainWave.THETA:
                effective_max = int(max_proposals * 1.5)
            elif dom == BrainWave.DELTA:
                effective_max = max(1, int(max_proposals * 0.3))
            elif dom == BrainWave.GAMMA:
                # Gamma during sleep is unusual but can occur in
                # lucid dreaming — boost synthesis.
                effective_max = int(max_proposals * 1.3)
        concept_ids = self.network.dream_concept_ids
        if len(concept_ids) < 2:
            return []

        # Compute structural signatures
        signatures = self._compute_structural_signatures(concept_ids)

        # Group concepts by signature
        groups: dict[frozenset, list[str]] = {}
        for cid, sig in signatures.items():
            groups.setdefault(sig, []).append(cid)

        proposals: list[DreamProposal] = []
        for sig, members in groups.items():
            if len(members) < 2:
                continue
            # Find unconnected pairs within this structural group
            for i in range(len(members)):
                if len(proposals) >= effective_max:
                    break
                for j in range(i + 1, len(members)):
                    if len(proposals) >= effective_max:
                        break
                    c1, c2 = members[i], members[j]
                    if self._are_connected(c1, c2):
                        continue
                    proposal = self._propose_edge(c1, c2, sig)
                    if proposal is not None:
                        proposals.append(proposal)

        # Store hypothetical edges in the network
        for p in proposals:
            self.network.add_edge(
                p.source,
                p.target,
                p.relation,
                weight=_HYPOTHETICAL_WEIGHT,
                origin=_ORIGIN_DREAM_SYNTHESIS,
            )
            self._proposed_count += 1

        if proposals:
            logger.info(
                f"Dream synthesis: proposed {len(proposals)} hypothetical edges "
                f"from {len(groups)} structural groups"
            )

        return proposals

    def _compute_structural_signatures(
        self, concept_ids: list[str]
    ) -> dict[str, frozenset[tuple[str, str]]]:
        """Compute the structural signature of each concept.

        The signature is a frozenset of (relation_value, direction)
        pairs, where direction is "out" or "in". This captures the
        structural role of the concept — what kinds of relations it
        participates in and from which direction.

        Two concepts with the same signature play similar structural
        roles in the network.
        """
        signatures: dict[str, frozenset[tuple[str, str]]] = {}
        for cid in concept_ids:
            sig_parts: set[tuple[str, str]] = set()
            for edge in self.network.get_edges(cid, "out"):
                sig_parts.add((edge.relation.value, "out"))
            for edge in self.network.get_edges(cid, "in"):
                sig_parts.add((edge.relation.value, "in"))
            signatures[cid] = frozenset(sig_parts)
        return signatures

    def _are_connected(self, c1: str, c2: str) -> bool:
        """Check if two concepts already have any edge between them."""
        for edge in self.network.get_edges(c1, "out"):
            if edge.target == c2:
                return True
        for edge in self.network.get_edges(c2, "out"):
            if edge.target == c1:
                return True
        return False

    def _propose_edge(
        self, c1: str, c2: str, signature: frozenset[tuple[str, str]]
    ) -> DreamProposal | None:
        """Propose a typed edge between two structurally similar concepts.

        The relation type is predicted from the shared structural
        signature.
        """
        relation, predicted_from = self._predict_relation(signature)
        if relation is None:
            return None

        # Compute structural similarity (Jaccard of signatures)
        sig1 = self._get_signature_set(c1)
        sig2 = self._get_signature_set(c2)
        if not sig1 or not sig2:
            return None
        intersection = sig1 & sig2
        union = sig1 | sig2
        similarity = len(intersection) / len(union) if union else 0.0

        # Only propose if similarity is meaningful
        if similarity < 0.3:
            return None

        return DreamProposal(
            source=c1,
            target=c2,
            relation=relation,
            structural_similarity=similarity,
            predicted_from=predicted_from,
        )

    def _get_signature_set(self, cid: str) -> set[tuple[str, str]]:
        """Get the structural signature of a concept as a set."""
        sig: set[tuple[str, str]] = set()
        for edge in self.network.get_edges(cid, "out"):
            sig.add((edge.relation.value, "out"))
        for edge in self.network.get_edges(cid, "in"):
            sig.add((edge.relation.value, "in"))
        return sig

    @staticmethod
    def _predict_relation(
        signature: frozenset[tuple[str, str]]
    ) -> tuple[RelationType | None, str]:
        """Predict the relation type from the shared structural signature.

        Returns (relation, reason) or (None, "") if no prediction.
        """
        sig_strs = {s[0] for s in signature}

        # Two leaves of the same IS_A hub -> SIMILAR_TO
        if "is_a" in sig_strs:
            return RelationType.SIMILAR_TO, "shared IS_A hub membership"

        # Two nodes in the same CAUSES chain -> RELATED_TO (correlated)
        if "causes" in sig_strs:
            return RelationType.RELATED_TO, "shared CAUSES chain position"

        # Two enablers -> SIMILAR_TO
        if "enables" in sig_strs:
            return RelationType.SIMILAR_TO, "shared ENABLES role"

        # Two nodes that depend on similar things -> SIMILAR_TO
        if "depends_on" in sig_strs:
            return RelationType.SIMILAR_TO, "shared DEPENDS_ON pattern"

        # Default: generic association
        if signature:
            return RelationType.RELATED_TO, "structural role similarity"

        return None, ""

    # ─── Validation (during wake or dream review) ────────────────

    def validate(
        self,
        proposals: list[DreamProposal],
        reasoning_engine: ReasoningEngine | None = None,
    ) -> list[DreamValidationResult]:
        """Validate dream proposals using the reasoning engine.

        For each proposal, checks whether independent evidence in the
        network supports the hypothetical edge. If >= 2 corroborating
        paths are found and no contradictions, the edge is promoted to
        a real edge with raised confidence. If contradicted, it is
        dropped. Otherwise it remains hypothetical.

        Args:
            proposals: The dream proposals to validate.
            reasoning_engine: Optional reasoning engine for deeper
                validation. If None, uses structural checks only.

        Returns:
            List of validation results.
        """
        results: list[DreamValidationResult] = []

        for proposal in proposals:
            corroborations = self._count_corroborations(proposal)
            contradicted = self._check_contradictions(proposal)

            if contradicted:
                self._drop_hypothetical_edge(proposal)
                self._rejected_count += 1
                results.append(DreamValidationResult(
                    proposal=proposal,
                    corroborations=corroborations,
                    contradicted=True,
                    promoted=False,
                    reason="contradicted by existing OPPOSITE_OF/CONTRADICTS edge",
                ))
            elif corroborations >= _MIN_CORROBORATIONS:
                self._promote_edge(proposal)
                insight = DreamInsight(
                    source=proposal.source,
                    target=proposal.target,
                    relation=proposal.relation,
                    corroborations=corroborations,
                    insight_text=(
                        f"dream insight: {proposal.source} "
                        f"{proposal.relation.value} {proposal.target} "
                        f"— {corroborations} corroborating paths"
                    ),
                )
                self._validated_insights.append(insight)
                results.append(DreamValidationResult(
                    proposal=proposal,
                    corroborations=corroborations,
                    contradicted=False,
                    promoted=True,
                    reason=f"promoted: {corroborations} corroborating paths",
                ))
                logger.info(f"Dream insight validated: {insight.describe()}")
            else:
                results.append(DreamValidationResult(
                    proposal=proposal,
                    corroborations=corroborations,
                    contradicted=False,
                    promoted=False,
                    reason=(
                        f"insufficient evidence: {corroborations}/"
                        f"{_MIN_CORROBORATIONS} corroborations — "
                        f"remains hypothetical"
                    ),
                ))

        promoted = sum(1 for r in results if r.promoted)
        rejected = sum(1 for r in results if r.contradicted)
        logger.info(
            f"Dream validation: {promoted} promoted, {rejected} rejected, "
            f"{len(results) - promoted - rejected} still hypothetical"
        )

        return results

    def _count_corroborations(self, proposal: DreamProposal) -> int:
        """Count independent evidence paths supporting the proposal.

        Checks:
        1. Analogical transfer: A similar_to C, C R B
        2. Transitive chain: A R C, C R B (for transitive relations)
        3. Shared hub: both A and B connect to a common concept H

        Only counts paths through **validated or explicitly stated**
        edges — hypothetical dream-synthesis edges are proposals, not
        evidence, and must not corroborate each other (that would
        create a circular validation cascade).
        """
        count = 0
        src, tgt, rel = proposal.source, proposal.target, proposal.relation

        # 1. Analogical transfer: A similar_to C, and C has an edge
        #    with the same relation to B (or B to C)
        for edge in self.network.get_edges(src, "out"):
            if edge.relation != RelationType.SIMILAR_TO:
                continue
            if edge.origin == _ORIGIN_DREAM_SYNTHESIS:
                continue  # hypothetical — not evidence
            intermediate = edge.target
            if intermediate == tgt:
                continue
            # Check if intermediate has the proposed relation to target
            for e2 in self.network.get_edges(intermediate, "out"):
                if e2.origin == _ORIGIN_DREAM_SYNTHESIS:
                    continue
                if e2.target == tgt and e2.relation == rel:
                    count += 1
                    break

        # 2. Transitive chain: A R C, C R B
        if rel in _TRANSITIVE_RELATIONS:
            for edge in self.network.get_edges(src, "out"):
                if edge.relation != rel:
                    continue
                if edge.origin == _ORIGIN_DREAM_SYNTHESIS:
                    continue
                intermediate = edge.target
                if intermediate == tgt:
                    continue
                for e2 in self.network.get_edges(intermediate, "out"):
                    if e2.origin == _ORIGIN_DREAM_SYNTHESIS:
                        continue
                    if e2.target == tgt and e2.relation == rel:
                        count += 1
                        break

        # 3. Shared hub: both A and B connect to a common concept H
        #    via the same relation type
        src_targets = {
            e.target for e in self.network.get_edges(src, "out")
            if e.relation == rel and e.origin != _ORIGIN_DREAM_SYNTHESIS
        }
        tgt_targets = {
            e.target for e in self.network.get_edges(tgt, "out")
            if e.relation == rel and e.origin != _ORIGIN_DREAM_SYNTHESIS
        }
        shared = src_targets & tgt_targets
        if shared:
            count += 1  # shared hub counts as one corroboration

        return count

    def _check_contradictions(self, proposal: DreamProposal) -> bool:
        """Check if the proposed edge is contradicted by existing edges."""
        src, tgt = proposal.source, proposal.target

        # Check for direct contradiction edges
        for edge in self.network.get_edges(src, "out"):
            if edge.target == tgt and edge.relation in _CONTRADICTION_RELATIONS:
                return True
        for edge in self.network.get_edges(tgt, "out"):
            if edge.target == src and edge.relation in _CONTRADICTION_RELATIONS:
                return True

        return False

    def _promote_edge(self, proposal: DreamProposal) -> None:
        """Promote a hypothetical edge to a validated edge.

        Updates the existing edge's weight and origin tag in place.
        The edge transitions from "dream-synthesis" (hypothetical)
        to "dream-validated" (corroborated by ≥2 independent paths).
        """
        # Find the existing edge and update it directly
        for edge in self.network.get_edges(proposal.source, "out"):
            if (
                edge.target == proposal.target
                and edge.relation == proposal.relation
            ):
                edge.weight = _PROMOTED_WEIGHT
                edge.origin = _ORIGIN_DREAM_VALIDATED
                return
        # If not found (shouldn't happen), add it
        self.network.add_edge(
            proposal.source,
            proposal.target,
            proposal.relation,
            weight=_PROMOTED_WEIGHT,
            origin=_ORIGIN_DREAM_VALIDATED,
        )

    def _drop_hypothetical_edge(self, proposal: DreamProposal) -> None:
        """Remove a contradicted hypothetical edge.

        Marks the edge as very low weight with a "dream-rejected" origin.
        The edge remains but is effectively negligible.
        """
        for edge in self.network.get_edges(proposal.source, "out"):
            if (
                edge.target == proposal.target
                and edge.relation == proposal.relation
            ):
                edge.weight = 0.01
                edge.origin = "dream-rejected"
                return
        self.network.add_edge(
            proposal.source,
            proposal.target,
            proposal.relation,
            weight=0.01,
            origin="dream-rejected",
        )

    # ─── Dream review (called on wake) ───────────────────────────

    def review_pending_proposals(
        self, reasoning_engine: ReasoningEngine | None = None
    ) -> list[DreamValidationResult]:
        """Review all pending dream-synthesis edges.

        Finds all edges with origin ``dream-synthesis`` that haven't
        been validated yet, and runs the validation pass on them.

        This is called on wake (or during a dedicated dream-review
        stage) to close the loop: dreams propose, waking validates.
        """
        pending: list[DreamProposal] = []
        for edge in self.network.edges:
            if edge.origin != _ORIGIN_DREAM_SYNTHESIS:
                continue
            pending.append(DreamProposal(
                source=edge.source,
                target=edge.target,
                relation=edge.relation,
                structural_similarity=0.0,  # not needed for validation
                predicted_from="pending review",
            ))

        if not pending:
            return []

        return self.validate(pending, reasoning_engine)

    # ─── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, list[dict[str, object]] | dict[str, int]]:
        """Serialize state for persistence."""
        return {
            "insights": [
                {
                    "source": i.source,
                    "target": i.target,
                    "relation": i.relation.value,
                    "corroborations": i.corroborations,
                    "insight_text": i.insight_text,
                    "timestamp": i.timestamp,
                }
                for i in self._validated_insights
            ],
            "stats": {
                "proposed_count": self._proposed_count,
                "rejected_count": self._rejected_count,
            },
        }

    def restore_from_dict(self, data: dict[str, Any]) -> None:
        """Restore state from persistence."""
        self._validated_insights = []
        for item in data.get("insights", []):
            relation_str = item.get("relation", "related_to")
            try:
                relation = RelationType(relation_str)
            except ValueError:
                relation = RelationType.RELATED_TO
            self._validated_insights.append(DreamInsight(
                source=item["source"],
                target=item["target"],
                relation=relation,
                corroborations=item.get("corroborations", 0),
                insight_text=item.get("insight_text", ""),
                timestamp=item.get("timestamp", 0),
            ))
        stats = data.get("stats", {})
        self._proposed_count = stats.get("proposed_count", 0)
        self._rejected_count = stats.get("rejected_count", 0)
