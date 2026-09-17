"""Belief revision — closing the loop between evaluation and belief.

The critical thinking engine evaluates claims and produces
``CriticalAssessment`` objects with revised confidence, evidence
quality scores, and disconfirmation findings. But until now, those
assessments were ephemeral — they lowered the confidence of a single
reasoning result but didn't change the underlying beliefs in the
concept network.

This module closes that loop. The ``BeliefRevisionEngine`` takes
critical assessments and applies them to the network:

1. **Edge weight revision**: when disconfirming evidence is found,
   the corresponding edge's weight is downgraded. When evidence is
   strong and from multiple high-reliability sources, the edge is
   strengthened.

2. **Bayesian belief updating**: each piece of evidence is fed to
   ``ProbabilisticReasoning.observe_evidence()``, updating the Beta
   distributions that track belief strength. This is the update loop
   that was missing — the Bayesian infrastructure existed but was
   never fed evidence.

3. **Structure learning (Bayesian Model Reduction)**: edges whose
   weight falls below a threshold after repeated disconfirmation are
   pruned. This is structure learning in the active inference sense
   (Friston et al., 2021; Nature Communications, 2026) — not just
   updating parameters, but revising the structure of the world model
   itself when evidence no longer supports it.

4. **Non-destructive contradiction management**: when a contradiction
   is found, both edges are preserved but their weights are
   downgraded. The system retains the competing claims with reduced
   confidence rather than destroying knowledge. This matches the
   2026 epistemic integrity literature (Kumiho, arXiv 2026) which
   emphasizes preserving revision history.

## Integration

The engine is wired into the cognition engine and called after
critical evaluation. Every critical assessment that finds
disconfirming evidence or strong support now feeds back into the
network, so Genesis's beliefs actually change in response to
evidence — not just her confidence in individual reasoning results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..concepts import ConceptNetwork, Edge, RelationType

if TYPE_CHECKING:
    from .critical_thinking import CriticalAssessment
    from .engine import ProbabilisticReasoning

__all__ = ["BeliefRevisionEngine", "RevisionRecord"]

logger = logging.getLogger(__name__)

# Edge weight adjustment magnitudes.
_DOWNGRADE_FACTOR = 0.8  # multiply weight by this on disconfirmation
_STRENGTHEN_FACTOR = 1.15  # multiply weight by this on strong support
_MAX_WEIGHT = 1.0
_MIN_WEIGHT = 0.0

# Edges below this weight after revision are candidates for pruning.
_PRUNE_THRESHOLD = 0.05

# Only inferred/discovered/dream-validated/analogy edges are prunable.
# Stated, observed, and corrected edges are preserved even at low
# weight — they represent direct knowledge, not inference.
_PRUNABLE_ORIGINS = frozenset({
    "inferred", "discovered", "dream-validated", "analogy",
    "transferred",
})


@dataclass(slots=True)
class RevisionRecord:
    """A record of a single belief revision.

    Tracks what was revised, why, and how much it changed. This is
    the audit trail — Genesis can explain *why* she changed her mind
    about something, not just that she did.
    """

    concept: str
    relation: str
    target: str
    old_weight: float
    new_weight: float
    reason: str  # "disconfirmation", "strong_support", "pruned"
    evidence_summary: str = ""

    @property
    def weight_delta(self) -> float:
        """How much the weight changed."""
        return self.new_weight - self.old_weight

    @property
    def was_pruned(self) -> bool:
        """Whether this revision pruned the edge."""
        return self.reason == "pruned"


class BeliefRevisionEngine:
    """Closes the loop between critical evaluation and belief storage.

    Given a ``CriticalAssessment``, the engine revises the underlying
    beliefs in the concept network:

    - Disconfirming evidence → edge weight downgraded
    - Strong multi-source support → edge weight strengthened
    - Each evidence item → ``observe_evidence()`` called on the
      probabilistic reasoning engine (Bayesian update)
    - Edges below threshold after revision → pruned (structure learning)

    This is the mechanism by which Genesis *changes her mind*. The
    critical thinking engine judges whether a claim is justified;
    the belief revision engine actually revises the belief.

    Usage::

        revisor = BeliefRevisionEngine(network, probabilistic)
        for assessment in assessments:
            records = revisor.revise(assessment)

    The engine is stateless across calls (it modifies the network and
    the probabilistic reasoning engine, but doesn't hold state itself).
    Revision history is retained for introspection.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        probabilistic: ProbabilisticReasoning | None = None,
    ) -> None:
        """Initialize with a concept network and optional probabilistic engine.

        Args:
            network: The concept network — the belief store.
            probabilistic: The probabilistic reasoning engine —
                provides ``observe_evidence()`` for Bayesian updating.
                If None, Bayesian updates are skipped but edge weight
                revision still occurs.
        """
        self.network = network
        self.probabilistic = probabilistic
        self._records: list[RevisionRecord] = []
        self._pruned_count = 0
        self._downgraded_count = 0
        self._strengthened_count = 0

    @property
    def records(self) -> list[RevisionRecord]:
        """All revision records (copy)."""
        return list(self._records)

    @property
    def pruned_count(self) -> int:
        """Total edges pruned by belief revision."""
        return self._pruned_count

    @property
    def downgraded_count(self) -> int:
        """Total edges downgraded by disconfirmation."""
        return self._downgraded_count

    @property
    def strengthened_count(self) -> int:
        """Total edges strengthened by strong support."""
        return self._strengthened_count

    # ─── Public API ───────────────────────────────────────────────

    def revise(self, assessment: CriticalAssessment) -> list[RevisionRecord]:
        """Apply a critical assessment to the network.

        This is the main entry point. Given a critical assessment with
        supporting and disconfirming evidence, the engine:

        1. Downgrades claim edges targeted by disconfirming evidence.
        2. Strengthens claim edges targeted by strong supporting evidence.
        3. Feeds each evidence item to the probabilistic reasoning
           engine for Bayesian updating.
        4. Prunes edges that fall below the threshold after revision.

        Args:
            assessment: The critical assessment to apply.

        Returns:
            A list of revision records documenting what changed.
        """
        records: list[RevisionRecord] = []

        # 1. Process disconfirming evidence → downgrade the CLAIM edges.
        # The disconfirming evidence points at concepts involved in the
        # claim; we downgrade the claim's edges, not the disconfirming
        # edges themselves.
        if assessment.disconfirming_evidence:
            for evidence in assessment.supporting_evidence:
                record = self._downgrade_claim_edge(evidence)
                if record is not None:
                    records.append(record)
                    self._downgraded_count += 1
        # Feed disconfirming evidence to Bayesian engine (negative).
        for evidence in assessment.disconfirming_evidence:
            self._observe_evidence(evidence, positive=False)

        # 2. Process supporting evidence → strengthen claim edges.
        if assessment.evidence_quality > 0.5 and not assessment.fallacies:
            for evidence in assessment.supporting_evidence:
                record = self._strengthen_edge(evidence)
                if record is not None:
                    records.append(record)
                    self._strengthened_count += 1
                # Feed to Bayesian engine (positive evidence).
                self._observe_evidence(evidence, positive=True)
        elif not assessment.fallacies:
            # Still feed supporting evidence to Bayesian engine even
            # when not strengthening (always update beliefs).
            for evidence in assessment.supporting_evidence:
                self._observe_evidence(evidence, positive=True)

        # 3. Prune edges that fell below threshold.
        pruned = self._prune_weak_edges(records)
        records.extend(pruned)

        self._records.extend(records)
        return records

    def revise_batch(
        self, assessments: list[CriticalAssessment],
    ) -> list[RevisionRecord]:
        """Apply a batch of critical assessments to the network.

        Args:
            assessments: The critical assessments to apply.

        Returns:
            A list of all revision records.
        """
        all_records: list[RevisionRecord] = []
        for assessment in assessments:
            all_records.extend(self.revise(assessment))
        return all_records

    # ─── Edge revision ────────────────────────────────────────────

    def _downgrade_claim_edge(self, evidence: object) -> RevisionRecord | None:
        """Downgrade a claim edge based on disconfirming evidence.

        The evidence item represents the *claim* (supporting evidence);
        when disconfirming evidence exists, we downgrade the claim's
        edge. The edge is preserved (non-destructive) — only its weight
        changes.
        """
        target = getattr(evidence, "target_concept", "")
        relation_str = getattr(evidence, "relation", "")
        source = getattr(evidence, "source_concept", "")

        if not target or not relation_str:
            return None

        try:
            rel_type = RelationType(relation_str)
        except ValueError:
            return None

        # Find the claim edge (source → target with the claim's relation).
        edge = self._find_edge(source, target, rel_type)
        if edge is None:
            return None

        old_weight = edge.weight
        new_weight = max(_MIN_WEIGHT, old_weight * _DOWNGRADE_FACTOR)
        edge.weight = new_weight

        return RevisionRecord(
            concept=edge.source,
            relation=edge.relation.value,
            target=edge.target,
            old_weight=old_weight,
            new_weight=new_weight,
            reason="disconfirmation",
            evidence_summary="claim disconfirmed by opposing evidence",
        )

    def _strengthen_edge(self, evidence: object) -> RevisionRecord | None:
        """Strengthen an edge based on strong supporting evidence.

        Multiplies the edge's weight by the strengthen factor, capped
        at 1.0. Only called when evidence quality is high and no
        fallacies were detected.
        """
        target = getattr(evidence, "target_concept", "")
        relation_str = getattr(evidence, "relation", "")
        source = getattr(evidence, "source_concept", "")

        if not target or not relation_str:
            return None

        try:
            rel_type = RelationType(relation_str)
        except ValueError:
            return None

        edge = self._find_edge(source, target, rel_type)
        if edge is None:
            return None

        old_weight = edge.weight
        new_weight = min(_MAX_WEIGHT, old_weight * _STRENGTHEN_FACTOR)
        if new_weight <= old_weight + 0.001:
            return None  # already at max

        edge.weight = new_weight

        return RevisionRecord(
            concept=edge.source,
            relation=edge.relation.value,
            target=edge.target,
            old_weight=old_weight,
            new_weight=new_weight,
            reason="strong_support",
            evidence_summary=f"supported by {source} ({relation_str})",
        )

    def _prune_weak_edges(
        self, records: list[RevisionRecord],
    ) -> list[RevisionRecord]:
        """Prune edges that fell below the threshold after revision.

        Only prunes edges from prunable origins (inferred, discovered,
        dream-validated, analogy, transferred). Stated, observed, and
        corrected edges are preserved even at low weight — they
        represent direct knowledge.

        This is structure learning in the active inference sense: the
        structure of the world model is revised when evidence no
        longer supports it.
        """
        pruned: list[RevisionRecord] = []
        for record in records:
            if record.reason != "disconfirmation":
                continue
            if record.new_weight > _PRUNE_THRESHOLD:
                continue
            # Find the edge and check if it's prunable.
            edge = self._find_edge(
                record.concept, record.target,
                RelationType(record.relation),
            )
            if edge is None or edge.origin not in _PRUNABLE_ORIGINS:
                continue
            self.network.remove_edge(
                record.concept, record.target, edge.relation,
            )
            self._pruned_count += 1
            pruned.append(RevisionRecord(
                concept=record.concept,
                relation=record.relation,
                target=record.target,
                old_weight=record.new_weight,
                new_weight=0.0,
                reason="pruned",
                evidence_summary=f"pruned: weight fell below {_PRUNE_THRESHOLD}",
            ))
        return pruned

    # ─── Bayesian updating ────────────────────────────────────────

    def _observe_evidence(self, evidence: object, positive: bool) -> None:
        """Feed evidence to the probabilistic reasoning engine.

        This is the missing Bayesian update loop. Each piece of
        evidence (supporting or disconfirming) updates the Beta
        distribution for the corresponding belief, so the probabilistic
        reasoning engine's beliefs evolve with evidence.
        """
        if self.probabilistic is None:
            return
        source = getattr(evidence, "source_concept", "")
        relation = getattr(evidence, "relation", "")
        target = getattr(evidence, "target_concept", "")
        weight = getattr(evidence, "adjusted_weight", 0.5)
        if not source or not relation or not target:
            return
        try:
            self.probabilistic.observe_evidence(
                source, relation, target, positive, weight,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug(f'silent except: {e}')

    # ─── Non-destructive contradiction resolution ─────────────────

    def resolve_contradiction_non_destructive(
        self,
        concept_a: str,
        concept_b: str,
        relation_a: RelationType,
        relation_b: RelationType,
    ) -> list[RevisionRecord]:
        """Resolve a contradiction by downgrading both sides, not removing.

        When two edges contradict (e.g., SIMILAR_TO and OPPOSITE_OF on
        the same pair), this method downgrades *both* edges rather than
        removing the weaker one. Both claims are preserved with
        reduced confidence — the system retains the competing claims
        and can re-evaluate them later as more evidence accumulates.

        This replaces the destructive resolution in
        ``consistency_sweep()`` which removes the lower-weight edge.
        """
        records: list[RevisionRecord] = []
        edge_a = self._find_edge(concept_a, concept_b, relation_a)
        edge_b = self._find_edge(concept_a, concept_b, relation_b)

        for edge in [edge_a, edge_b]:
            if edge is None:
                continue
            old_weight = edge.weight
            new_weight = max(_MIN_WEIGHT, old_weight * _DOWNGRADE_FACTOR)
            edge.weight = new_weight
            records.append(RevisionRecord(
                concept=edge.source,
                relation=edge.relation.value,
                target=edge.target,
                old_weight=old_weight,
                new_weight=new_weight,
                reason="contradiction_downgrade",
                evidence_summary=(
                    f"contradiction with {concept_b}: both sides downgraded"
                ),
            ))
            self._downgraded_count += 1

        self._records.extend(records)
        return records

    # ─── Helpers ─────────────────────────────────────────────────

    def _find_edge(
        self, source: str, target: str, relation: RelationType,
    ) -> Edge | None:
        """Find an edge by source, target, and relation."""
        src_id = self.network._resolve(source)
        tgt_id = self.network._resolve(target)
        if src_id is None or tgt_id is None:
            return None
        for edge in self.network.get_edges(src_id, "out"):
            if edge.relation == relation and edge.target == tgt_id:
                return edge
        # Try reverse direction.
        for edge in self.network.get_edges(tgt_id, "out"):
            if edge.relation == relation and edge.target == src_id:
                return edge
        return None

    # ─── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Serialize state for persistence."""
        return {
            "pruned_count": self._pruned_count,
            "downgraded_count": self._downgraded_count,
            "strengthened_count": self._strengthened_count,
            "recent_records": [
                {
                    "concept": r.concept,
                    "relation": r.relation,
                    "target": r.target,
                    "old_weight": r.old_weight,
                    "new_weight": r.new_weight,
                    "reason": r.reason,
                }
                for r in self._records[-20:]
            ],
        }

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Restore state from persistence."""
        pruned = data.get("pruned_count", 0)
        downgraded = data.get("downgraded_count", 0)
        strengthened = data.get("strengthened_count", 0)
        self._pruned_count = (
            int(pruned) if isinstance(pruned, (int, float)) else 0
        )
        self._downgraded_count = (
            int(downgraded) if isinstance(downgraded, (int, float)) else 0
        )
        self._strengthened_count = (
            int(strengthened) if isinstance(strengthened, (int, float)) else 0
        )
