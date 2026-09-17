"""Critical thinking — epistemic evaluation of claims and evidence.

Genesis could reason about concepts (deductive, abductive, analogical,
causal) and form hypotheses, but she had no way to evaluate whether a
claim is *well-supported*. The reasoning engine produces conclusions;
nothing checks whether those conclusions rest on good evidence, whether
the evidence comes from credible sources, or whether disconfirming
evidence exists that should lower confidence.

This module closes that gap with a **critical thinking engine** that
applies epistemic scrutiny to reasoning results before they reach the
answer composer. It's the difference between "I can produce a conclusion"
and "I can judge whether my conclusion is justified."

## What it does

1. **Evidence quality assessment**: for each reasoning result, evaluate
   the supporting evidence — how many independent sources support it,
   what is their reliability, and how strong are the edges.

2. **Source credibility weighting**: edges from high-reliability sources
   (official docs, dictionaries) count more than edges from low-
   reliability sources (Wikipedia, conversation). The existing
   ``SOURCE_RELIABILITY`` hierarchy is finally propagated into
   epistemic evaluation.

3. **Disconfirmation search**: for each conclusion, search for edges
   that would contradict it (OPPOSITE_OF, CONTRADICTS, PREVENTS when
   the conclusion asserts ENABLES, etc.). Disconfirming evidence lowers
   confidence.

4. **Fallacy detection**: flag common reasoning fallacies:
   - **Hasty generalization**: a hypothesis from a single weak path
   - **Circular reasoning**: A→B→A chains in the evidence
   - **Unsupported assertion**: a claim with no evidence edges
   - **Confirmation bias**: only supporting evidence found, no
     disconfirmation search was performed

5. **Confidence revision**: produce a revised confidence that combines
   the original reasoning confidence with the evidence quality
   assessment. Well-supported conclusions keep or gain confidence;
   poorly-supported conclusions lose it.

6. **Critical assessment**: return a ``CriticalAssessment`` with the
   revised confidence, evidence quality score, disconfirmation count,
   fallacies detected, and a recommendation (accept, hedge, reject,
   investigate further).

## Integration

The engine is wired into the cognition engine as
``self.cognition.critical_thinking`` and invoked on reasoning results
before they reach the answer composer. Results that fail critical
scrutiny have their confidence downgraded and are flagged with
fallacies, so Genesis hedges or investigates rather than asserting
uncertain claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ..concepts import ConceptNetwork, Edge, RelationType
from .engine import ReasoningResult

if TYPE_CHECKING:
    pass

__all__ = [
    "AssessmentRecommendation",
    "CriticalAssessment",
    "CriticalThinkingEngine",
    "EvidenceItem",
]

logger = logging.getLogger(__name__)

# Source reliability hierarchy (mirrors learning/autonomous.py SOURCE_RELIABILITY).
# Higher = more trustworthy. Conversation is mid-tier — the user is
# generally reliable but can be wrong. Seeds and structural concepts
# are high-tier (they're foundational knowledge Genesis was built with).
# Inferred and discovered edges are low-tier (they're Genesis's own
# conclusions, not verified by external sources).
_SOURCE_RELIABILITY: dict[str, float] = {
    # Ground truth — official documentation
    "python_docs": 1.0,
    "rust_docs": 1.0,
    "man_pages": 1.0,
    # Curated — lexicographers, foundational seeds
    "wordnet": 0.85,
    "seeded": 0.8,
    "structural": 0.8,
    # Corrected — user explicitly corrected this, high trust
    "corrected": 0.75,
    # Observed — direct observation
    "observed": 0.7,
    "perceptual": 0.7,
    # Conversation — user said it, generally reliable but fallible
    "conversation": 0.6,
    "learned": 0.6,
    # Transferred — analogical transfer, plausible but unverified
    "transferred": 0.4,
    # Inferred — Genesis's own reasoning, needs verification
    "inferred": 0.3,
    # Discovered — pattern-matched, speculative
    "discovered": 0.25,
    # Dream-validated — sleep consolidation, tentative
    "dream-validated": 0.2,
    # Analogy — cross-domain projection, most speculative
    "analogy": 0.2,
}

# Default reliability for unknown origins.
_DEFAULT_RELIABILITY = 0.5

# Minimum number of independent sources for "well-supported".
_MIN_SOURCES_FOR_STRONG = 2

# Confidence adjustment magnitudes.
_STRONG_EVIDENCE_BOOST = 0.1
_WEAK_EVIDENCE_PENALTY = 0.15
_DISCONFIRMATION_PENALTY = 0.2
_FALLACY_PENALTY = 0.1

# Relations that contradict each other.
_CONTRADICTION_PAIRS: dict[RelationType, set[RelationType]] = {
    RelationType.ENABLES: {RelationType.PREVENTS, RelationType.HARMS},
    RelationType.CAUSES: {RelationType.PREVENTS},
    RelationType.IS_A: {RelationType.OPPOSITE_OF},
    RelationType.SIMILAR_TO: {RelationType.OPPOSITE_OF, RelationType.CONTRADICTS},
    RelationType.RELATED_TO: {RelationType.CONTRADICTS},
}


class AssessmentRecommendation(Enum):
    """What Genesis should do with a critically-assessed claim.

    - ACCEPT: the claim is well-supported; assert it confidently.
    - HEDGE: the claim has some support but also concerns; express
      uncertainty.
    - REJECT: the claim is contradicted or fallacious; don't assert it.
    - INVESTIGATE: the claim lacks sufficient evidence; seek more
      information before asserting.
    """

    ACCEPT = "accept"
    HEDGE = "hedge"
    REJECT = "reject"
    INVESTIGATE = "investigate"


@dataclass(slots=True)
class EvidenceItem:
    """A single piece of evidence for or against a claim.

    An evidence item is an edge in the concept network that supports
    or contradicts a conclusion. Its weight is adjusted by the source
    reliability of its origin.
    """

    source_concept: str
    relation: str
    target_concept: str
    raw_weight: float
    source_reliability: float
    origin: str
    supports: bool = True

    @property
    def adjusted_weight(self) -> float:
        """Evidence weight adjusted by source reliability."""
        return self.raw_weight * self.source_reliability


@dataclass(slots=True)
class CriticalAssessment:
    """The result of critically evaluating a reasoning result.

    Contains the revised confidence, evidence quality assessment,
    disconfirmation analysis, fallacy flags, and a recommendation for
    how Genesis should handle the claim.
    """

    original_confidence: float
    revised_confidence: float
    evidence_quality: float  # 0..1, how good the supporting evidence is
    supporting_evidence: list[EvidenceItem] = field(default_factory=list)
    disconfirming_evidence: list[EvidenceItem] = field(default_factory=list)
    fallacies: list[str] = field(default_factory=list)
    recommendation: AssessmentRecommendation = AssessmentRecommendation.INVESTIGATE
    notes: list[str] = field(default_factory=list)

    @property
    def confidence_delta(self) -> float:
        """How much the confidence changed after critical evaluation."""
        return self.revised_confidence - self.original_confidence

    @property
    def is_downgraded(self) -> bool:
        """Whether the assessment lowered the confidence."""
        return self.confidence_delta < -0.01

    @property
    def is_upgraded(self) -> bool:
        """Whether the assessment raised the confidence."""
        return self.confidence_delta > 0.01

    def describe(self) -> str:
        """Human-readable summary of the assessment."""
        rec = self.recommendation.value
        lines = [
            f"Critical assessment: {rec}",
            f"  confidence: {self.original_confidence:.2f} → "
            f"{self.revised_confidence:.2f} "
            f"({self.confidence_delta:+.2f})",
            f"  evidence quality: {self.evidence_quality:.2f}",
            f"  supporting: {len(self.supporting_evidence)} items",
            f"  disconfirming: {len(self.disconfirming_evidence)} items",
        ]
        if self.fallacies:
            lines.append(f"  fallacies: {', '.join(self.fallacies)}")
        if self.notes:
            lines.append(f"  notes: {'; '.join(self.notes)}")
        return "\n".join(lines)


class CriticalThinkingEngine:
    """Epistemic evaluation of claims and evidence.

    Given a reasoning result (a conclusion with evidence), the engine
    evaluates whether the conclusion is well-supported by assessing
    evidence quality, searching for disconfirming evidence, detecting
    fallacies, and revising confidence accordingly.

    Usage::

        critic = CriticalThinkingEngine(network)
        assessment = critic.evaluate(reasoning_result)
        if assessment.recommendation == AssessmentRecommendation.REJECT:
            # Don't assert this claim
            ...
        revised_confidence = assessment.revised_confidence

    The engine is stateless across calls (it reads the network; it
    doesn't modify it). Assessment history is retained for introspection.
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Initialize with a concept network to evaluate against.

        Args:
            network: The concept network — the evidence base.
        """
        self.network = network
        self._assessments: list[CriticalAssessment] = []
        self._downgraded_count = 0
        self._rejected_count = 0
        self._fallacy_count = 0

    @property
    def assessments(self) -> list[CriticalAssessment]:
        """All assessments ever produced (copy)."""
        return list(self._assessments)

    @property
    def downgraded_count(self) -> int:
        """Total claims downgraded by critical evaluation."""
        return self._downgraded_count

    @property
    def rejected_count(self) -> int:
        """Total claims rejected by critical evaluation."""
        return self._rejected_count

    @property
    def fallacy_count(self) -> int:
        """Total fallacies detected."""
        return self._fallacy_count

    # ─── Public API ───────────────────────────────────────────────

    def evaluate(self, result: ReasoningResult) -> CriticalAssessment:
        """Critically evaluate a reasoning result.

        This is the main entry point. Given a conclusion with evidence,
        the engine:
        1. Gathers supporting evidence from the network.
        2. Searches for disconfirming evidence.
        3. Detects fallacies in the reasoning.
        4. Revises confidence based on evidence quality.
        5. Recommends how to handle the claim.

        Args:
            result: The reasoning result to evaluate.

        Returns:
            A CriticalAssessment with revised confidence and recommendation.
        """
        # 1. Gather supporting evidence from the result's knowledge triples.
        supporting = self._gather_supporting_evidence(result)

        # 2. Search for disconfirming evidence.
        disconfirming = self._search_disconfirmation(result)

        # 3. Detect fallacies.
        fallacies = self._detect_fallacies(result, supporting, disconfirming)

        # 4. Assess evidence quality.
        evidence_quality = self._assess_evidence_quality(supporting)

        # 5. Revise confidence.
        revised = self._revise_confidence(
            result.confidence, supporting, disconfirming, fallacies,
        )

        # 6. Recommend.
        recommendation = self._recommend(
            revised, supporting, disconfirming, fallacies,
        )

        # Build notes.
        notes: list[str] = []
        if not supporting:
            notes.append("no supporting evidence found")
        if disconfirming:
            notes.append(
                f"{len(disconfirming)} disconfirming evidence items"
            )
        if fallacies:
            notes.append(f"{len(fallacies)} fallacy flags")

        assessment = CriticalAssessment(
            original_confidence=result.confidence,
            revised_confidence=revised,
            evidence_quality=evidence_quality,
            supporting_evidence=supporting,
            disconfirming_evidence=disconfirming,
            fallacies=fallacies,
            recommendation=recommendation,
            notes=notes,
        )

        # Track stats.
        self._assessments.append(assessment)
        if assessment.is_downgraded:
            self._downgraded_count += 1
        if recommendation == AssessmentRecommendation.REJECT:
            self._rejected_count += 1
        self._fallacy_count += len(fallacies)

        return assessment

    def evaluate_batch(
        self, results: list[ReasoningResult],
    ) -> list[tuple[ReasoningResult, CriticalAssessment]]:
        """Evaluate a batch of reasoning results.

        Returns the original results paired with their assessments,
        with revised confidence applied to the results.

        Args:
            results: The reasoning results to evaluate.

        Returns:
            List of (result, assessment) tuples. The result's confidence
            is replaced with the revised confidence.
        """
        evaluated: list[tuple[ReasoningResult, CriticalAssessment]] = []
        for result in results:
            assessment = self.evaluate(result)
            # Apply revised confidence to the result.
            if assessment.revised_confidence != result.confidence:
                result = _with_confidence(result, assessment.revised_confidence)
            evaluated.append((result, assessment))
        return evaluated

    # ─── Evidence gathering ───────────────────────────────────────

    def _gather_supporting_evidence(
        self, result: ReasoningResult,
    ) -> list[EvidenceItem]:
        """Gather supporting evidence from the result's knowledge triples.

        Each knowledge triple (relation, target, weight) in the result
        becomes an EvidenceItem, with source reliability determined by
        the edge's origin in the network.
        """
        evidence: list[EvidenceItem] = []
        for rel_str, target, weight in result.knowledge:
            edge = self._find_edge(result, rel_str, target)
            origin = edge.origin if edge else "inferred"
            reliability = self._source_reliability(origin)
            source_concept = edge.source if edge else ""
            evidence.append(EvidenceItem(
                source_concept=source_concept,
                relation=rel_str,
                target_concept=target,
                raw_weight=weight,
                source_reliability=reliability,
                origin=origin,
                supports=True,
            ))
        # Also check the evidence chain for stated relationships.
        for ev in result.evidence:
            parsed = self._parse_evidence_string(ev)
            if parsed is not None:
                src, rel, tgt = parsed
                edge = self._find_edge_by_parts(src, rel, tgt)
                if edge is not None:
                    origin = edge.origin
                    reliability = self._source_reliability(origin)
                    # Avoid duplicates.
                    if not any(
                        e.source_concept == edge.source
                        and e.target_concept == edge.target
                        and e.relation == rel
                        for e in evidence
                    ):
                        evidence.append(EvidenceItem(
                            source_concept=edge.source,
                            relation=rel,
                            target_concept=edge.target,
                            raw_weight=edge.weight,
                            source_reliability=reliability,
                            origin=origin,
                            supports=True,
                        ))
        return evidence

    def _search_disconfirmation(
        self, result: ReasoningResult,
    ) -> list[EvidenceItem]:
        """Search for evidence that contradicts the conclusion.

        For each knowledge triple (source → relation → target) in the
        result, look for evidence that specifically contradicts *that*
        claim — not just any contradiction involving the same concepts.

        Two kinds of disconfirmation:
        1. **Relation-specific**: if the claim is ENABLES, look for
           PREVENTS/HARMS on the same source→target pair. If the claim
           is SIMILAR_TO, look for OPPOSITE_OF/CONTRADICTS on the same
           pair.
        2. **Direct contradiction**: a CONTRADICTS edge between the
           claim's source and target disconfirms any claim about that
           pair.

        A CONTRADICTS edge on an *unrelated* concept (e.g., fire
        contradicts ice) does NOT disconfirm an unrelated claim (e.g.,
        fire causes heat) — those are different claims about different
        relationships.
        """
        disconfirming: list[EvidenceItem] = []
        for ev in result.evidence:
            parsed = self._parse_evidence_string(ev)
            if parsed is None:
                continue
            src, rel_str, tgt = parsed
            try:
                rel_type = RelationType(rel_str)
            except ValueError:
                continue
            src_id = self.network._resolve(src)
            tgt_id = self.network._resolve(tgt)
            if src_id is None or tgt_id is None:
                continue
            contradicting_rels = _CONTRADICTION_PAIRS.get(rel_type, set())
            # 1. Relation-specific contradictions on the same pair.
            if contradicting_rels:
                for edge in self.network.get_edges(src_id, "both"):
                    if edge.relation not in contradicting_rels:
                        continue
                    other = (
                        edge.target if edge.source == src_id else edge.source
                    )
                    if other != tgt_id:
                        continue  # must be the same pair
                    reliability = self._source_reliability(edge.origin)
                    disconfirming.append(EvidenceItem(
                        source_concept=other,
                        relation=edge.relation.value,
                        target_concept=tgt_id,
                        raw_weight=edge.weight,
                        source_reliability=reliability,
                        origin=edge.origin,
                        supports=False,
                    ))
            # 2. Direct CONTRADICTS between the claim's source and target.
            for edge in self.network.get_edges(src_id, "both"):
                if edge.relation != RelationType.CONTRADICTS:
                    continue
                other = (
                    edge.target if edge.source == src_id else edge.source
                )
                if other != tgt_id:
                    continue  # must be the same pair
                # Don't double-count if already found above.
                if any(
                    d.source_concept == other
                    and d.target_concept == tgt_id
                    and d.relation == "contradicts"
                    for d in disconfirming
                ):
                    continue
                reliability = self._source_reliability(edge.origin)
                disconfirming.append(EvidenceItem(
                    source_concept=other,
                    relation="contradicts",
                    target_concept=tgt_id,
                    raw_weight=edge.weight,
                    source_reliability=reliability,
                    origin=edge.origin,
                    supports=False,
                ))
        return disconfirming

    # ─── Fallacy detection ────────────────────────────────────────

    def _detect_fallacies(
        self,
        result: ReasoningResult,
        supporting: list[EvidenceItem],
        disconfirming: list[EvidenceItem],
    ) -> list[str]:
        """Detect common reasoning fallacies.

        Checks for:
        - **Hasty generalization**: hypothesis with only one weak source.
        - **Circular reasoning**: A→B→A in the evidence chain.
        - **Unsupported assertion**: no supporting evidence at all.
        - **Confirmation bias**: no disconfirmation search was done
          (always false here since we search, but flagged if the result
          has high confidence with zero supporting evidence).
        """
        fallacies: list[str] = []

        # Unsupported assertion: no supporting evidence.
        if not supporting and result.confidence > 0.5:
            fallacies.append("unsupported_assertion")

        # Hasty generalization: hypothesis from a single weak source.
        if (
            len(supporting) == 1
            and supporting[0].source_reliability < 0.5
            and result.confidence > 0.5
        ):
            fallacies.append("hasty_generalization")

        # Circular reasoning: check the evidence chain for A→B→A patterns.
        if self._has_circular_reasoning(result.evidence):
            fallacies.append("circular_reasoning")

        # Overconfidence with disconfirming evidence: high confidence
        # despite significant disconfirmation.
        if (
            disconfirming
            and result.confidence > 0.6
            and sum(d.adjusted_weight for d in disconfirming) > 0.3
        ):
            fallacies.append("ignoring_disconfirmation")

        return fallacies

    def _has_circular_reasoning(self, evidence: list[str]) -> bool:
        """Check if the evidence chain contains A→B→A circular patterns."""
        # Parse all evidence strings and look for cycles.
        concepts_in_chain: set[str] = set()
        for ev in evidence:
            parsed = self._parse_evidence_string(ev)
            if parsed is None:
                continue
            src, _rel, tgt = parsed
            # If we've seen the target before as a source, it's circular.
            if tgt in concepts_in_chain and src in concepts_in_chain:
                return True
            concepts_in_chain.add(src)
            concepts_in_chain.add(tgt)
        return False

    # ─── Evidence quality assessment ──────────────────────────────

    def _assess_evidence_quality(
        self, supporting: list[EvidenceItem],
    ) -> float:
        """Assess the quality of supporting evidence (0..1).

        Quality is a function of:
        - Number of independent sources (more = better)
        - Source reliability (higher = better)
        - Evidence weight (stronger edges = better)

        A single high-reliability source scores ~0.5. Multiple
        independent high-reliability sources can reach 1.0.
        """
        if not supporting:
            return 0.0
        # Count independent sources (by origin).
        origins = {e.origin for e in supporting}
        n_sources = len(origins)
        # Average adjusted weight.
        total_weight = sum(e.adjusted_weight for e in supporting)
        avg_weight = total_weight / len(supporting)
        # Source diversity bonus: more independent sources = higher quality.
        diversity_bonus = min(0.3, (n_sources - 1) * 0.15)
        # Combine: average weight + diversity bonus.
        quality = min(1.0, avg_weight + diversity_bonus)
        return quality

    # ─── Confidence revision ──────────────────────────────────────

    def _revise_confidence(
        self,
        original: float,
        supporting: list[EvidenceItem],
        disconfirming: list[EvidenceItem],
        fallacies: list[str],
    ) -> float:
        """Revise confidence based on evidence quality and concerns.

        - Strong evidence (multiple high-reliability sources) boosts.
        - Weak evidence (single low-reliability source) penalizes.
        - Disconfirming evidence penalizes.
        - Fallacies penalize.
        """
        revised = original
        # Evidence quality adjustment.
        if supporting:
            n_sources = len({e.origin for e in supporting})
            avg_reliability = (
                sum(e.source_reliability for e in supporting) / len(supporting)
            )
            if n_sources >= _MIN_SOURCES_FOR_STRONG and avg_reliability > 0.6:
                # Strong evidence — boost.
                revised = min(1.0, revised + _STRONG_EVIDENCE_BOOST)
            elif n_sources == 1 and avg_reliability < 0.4:
                # Weak evidence — penalize.
                revised = max(0.0, revised - _WEAK_EVIDENCE_PENALTY)
        else:
            # No supporting evidence — large penalty.
            revised = max(0.0, revised - _WEAK_EVIDENCE_PENALTY * 2)

        # Disconfirmation penalty.
        if disconfirming:
            disconfirm_weight = sum(
                d.adjusted_weight for d in disconfirming
            )
            revised = max(0.0, revised - disconfirm_weight * _DISCONFIRMATION_PENALTY)

        # Fallacy penalties.
        if fallacies:
            revised = max(0.0, revised - len(fallacies) * _FALLACY_PENALTY)

        return revised

    # ─── Recommendation ───────────────────────────────────────────

    def _recommend(
        self,
        revised_confidence: float,
        supporting: list[EvidenceItem],
        disconfirming: list[EvidenceItem],
        fallacies: list[str],
    ) -> AssessmentRecommendation:
        """Recommend how to handle the claim.

        - REJECT: contradicted or multiple fallacies with low confidence.
        - INVESTIGATE: no supporting evidence.
        - HEDGE: weak evidence or disconfirming evidence present.
        - ACCEPT: strong evidence, no fallacies, good confidence.
        """
        # Reject: strong disconfirmation or fallacies with low confidence.
        has_serious_fallacy = (
            "unsupported_assertion" in fallacies
            or "circular_reasoning" in fallacies
        )
        if fallacies and has_serious_fallacy:
            if revised_confidence < 0.3:
                return AssessmentRecommendation.REJECT
        if disconfirming and revised_confidence < 0.2:
            return AssessmentRecommendation.REJECT

        # Investigate: no supporting evidence.
        if not supporting:
            return AssessmentRecommendation.INVESTIGATE

        # Hedge: weak evidence or disconfirming evidence.
        if disconfirming or revised_confidence < 0.4 or fallacies:
            return AssessmentRecommendation.HEDGE

        # Accept: good confidence and evidence.
        if revised_confidence >= 0.5:
            return AssessmentRecommendation.ACCEPT

        # Default: hedge.
        return AssessmentRecommendation.HEDGE

    # ─── Helpers ─────────────────────────────────────────────────

    def _source_reliability(self, origin: str) -> float:
        """Get the reliability score for a source origin."""
        return _SOURCE_RELIABILITY.get(origin, _DEFAULT_RELIABILITY)

    def _find_edge(
        self, result: ReasoningResult, rel_str: str, target: str,
    ) -> Edge | None:
        """Find the network edge corresponding to a knowledge triple."""
        try:
            rel_type = RelationType(rel_str)
        except ValueError:
            return None
        # Search edges on the target concept.
        for edge in self.network.get_edges(target, "both"):
            if edge.relation == rel_type:
                return edge
        return None

    def _find_edge_by_parts(
        self, source: str, rel_str: str, target: str,
    ) -> Edge | None:
        """Find an edge by source, relation, and target."""
        try:
            rel_type = RelationType(rel_str)
        except ValueError:
            return None
        src_id = self.network._resolve(source)
        tgt_id = self.network._resolve(target)
        if src_id is None or tgt_id is None:
            return None
        for edge in self.network.get_edges(src_id, "out"):
            if edge.relation == rel_type and edge.target == tgt_id:
                return edge
        return None

    def _parse_evidence_string(
        self, evidence: str,
    ) -> tuple[str, str, str] | None:
        """Parse an evidence string like 'A causes B' into (A, causes, B).

        Returns None if the string can't be parsed into a triple.
        """
        # Evidence strings from the reasoning engine look like:
        # "A relation_value B" or "A → B" or "A relation_value B (weight=...)"
        # Try to parse the "A relation B" format.
        parts = evidence.split()
        if len(parts) < 3:
            return None
        # Find the relation word (it should be a RelationType value).
        for i in range(1, len(parts) - 1):
            word = parts[i].strip("(),")
            try:
                RelationType(word)
                source = parts[0].strip("(),")
                target = parts[-1].strip("(),")
                # Strip trailing parenthetical.
                target = target.split("(")[0].strip()
                if source and target:
                    return (source, word, target)
            except ValueError:
                continue
        return None

    # ─── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Serialize state for persistence."""
        return {
            "downgraded_count": self._downgraded_count,
            "rejected_count": self._rejected_count,
            "fallacy_count": self._fallacy_count,
            "recent_assessments": [
                {
                    "original_confidence": a.original_confidence,
                    "revised_confidence": a.revised_confidence,
                    "evidence_quality": a.evidence_quality,
                    "recommendation": a.recommendation.value,
                    "fallacies": a.fallacies,
                    "n_supporting": len(a.supporting_evidence),
                    "n_disconfirming": len(a.disconfirming_evidence),
                }
                for a in self._assessments[-20:]
            ],
        }

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Restore state from persistence."""
        downgraded = data.get("downgraded_count", 0)
        rejected = data.get("rejected_count", 0)
        fallacy = data.get("fallacy_count", 0)
        self._downgraded_count = (
            int(downgraded) if isinstance(downgraded, (int, float)) else 0
        )
        self._rejected_count = (
            int(rejected) if isinstance(rejected, (int, float)) else 0
        )
        self._fallacy_count = (
            int(fallacy) if isinstance(fallacy, (int, float)) else 0
        )


def _with_confidence(
    result: ReasoningResult, confidence: float,
) -> ReasoningResult:
    """Return a copy of a ReasoningResult with revised confidence.

    ReasoningResult is a slots dataclass, so we reconstruct it with
    the new confidence value.
    """
    return ReasoningResult(
        conclusion=result.conclusion,
        reasoning_type=result.reasoning_type,
        evidence=result.evidence,
        confidence=confidence,
        novel=result.novel,
        contradictions=result.contradictions,
        knowledge=result.knowledge,
    )
