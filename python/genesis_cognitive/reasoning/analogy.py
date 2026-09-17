"""Analogy engine — structure-mapping across domains.

This is Genesis's cross-domain analogy system. It implements a
simplified Structure-Mapping Engine (SME, Gentner 1983) over the
concept network: it finds concepts in **different cortical columns**
whose **relational structure** matches, aligns their neighborhoods,
and projects relations from one domain into the other as **candidate
inferences**.

This is the mechanism that finds connections humans haven't seen.
Pair-level similarity (edge proposer, dream synthesis) finds concepts
that *look* alike. Structure-mapping finds concepts that *work* alike —
concepts whose role in their own domain is the same, even when the
domains share no surface features. "The atom is like the solar system"
is not a similarity of appearance; it is a similarity of relational
structure (central body, orbiting bodies, attractive force). That kind
of insight is what this engine produces.

## How it works

1. **Relation profile**: each concept gets a profile — a count of
   ``(relation, direction)`` pairs. Two concepts with the same profile
   play the same structural role (both are "causers" with two outgoing
   CAUSES edges, both are "enablers", etc.).

2. **Analogous pairs**: concepts in **different cortical columns** with
   overlapping profiles are candidate analogs. Cross-column is the
   novelty criterion — within-column analogies are usually obvious;
   cross-column analogies are the insights humans miss.

3. **Candidate inference (relation transfer)**: once S (domain A) and C
   (domain B) are aligned, any relation S has to a target T that C does
   *not* have to any target is projected into C's domain. The projected
   target T' is the concept near C whose own relation profile best
   matches T's — the "role equivalent" of T in C's domain. The
   predicted edge is ``C --R--> T'``.

4. **Analogical similarity**: the analogical mapping itself is written
   as a ``SIMILAR_TO`` edge between S and C — "S is structurally like
   C" — so downstream analogical transfer (reasoning engine) can use it.

5. **Validation**: each candidate inference is checked for
   contradictions (OPPOSITE_OF, CONTRADICTS, PREVENTS, HARMS) and for
   corroborating evidence (shared hubs, transitive chains, embedding
   proximity). Contradicted inferences are dropped; corroborated ones
   are written with ``origin="analogy"``.

## Lifecycle

Analogy edges are auto-generated knowledge (like ``discovered`` and
``dream-validated`` edges). They are migrated to the holographic graph
during sleep compression to prevent unbounded explicit-edge growth,
and they are eligible for normal Hebbian strengthening / SHY decay.

The engine runs during N3 slow-wave sleep (alongside the edge proposer)
via ``discover_and_accept()``.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..concepts import ConceptNetwork, RelationType

if TYPE_CHECKING:
    from ..concepts import EmbeddingStore

__all__ = ["AnalogyEngine", "AnalogyInsight", "AnalogyPair"]

logger = logging.getLogger(__name__)

# Relations that carry structural meaning — used to decide whether two
# concepts share enough structure to be analogs. RELATED_TO alone is not
# structural (it's generic co-occurrence), so it doesn't count.
_STRUCTURAL_RELATIONS: frozenset[RelationType] = frozenset({
    RelationType.IS_A,
    RelationType.PART_OF,
    RelationType.CAUSES,
    RelationType.LEADS_TO,
    RelationType.ENABLES,
    RelationType.DEPENDS_ON,
    RelationType.HAS_PROPERTY,
    RelationType.EMERGES_FROM,
    RelationType.PREVENTS,
    RelationType.HARMS,
    RelationType.CREATES,
    RelationType.OPPOSITE_OF,
})

# Relations that indicate a proposed edge would be contradictory.
_CONTRADICTION_RELATIONS: frozenset[RelationType] = frozenset({
    RelationType.OPPOSITE_OF,
    RelationType.CONTRADICTS,
    RelationType.PREVENTS,
    RelationType.HARMS,
})

# Minimum structural similarity (profile cosine) for two concepts to be
# considered analogs. Below this, the structural overlap is too weak.
_MIN_ANALOGY_SIMILARITY = 0.4

# Minimum target-match score for a relation transfer. Below this, we
# can't confidently identify the role-equivalent target in the analog's
# domain, so we don't project the relation.
_MIN_TARGET_MATCH = 0.3

# Edge weights for analogy-derived edges.
_SIMILARITY_WEIGHT = 0.4
_TRANSFER_WEIGHT_BASE = 0.35

# Origin tag for all analogy-derived edges.
_ORIGIN_ANALOGY = "analogy"

# Caps to keep the sleep-time discovery tractable.
_DEFAULT_MAX_PAIRS = 20
_DEFAULT_MAX_INFERENCES_PER_PAIR = 3
_DEFAULT_MAX_TARGET_CANDIDATES = 30


@dataclass(slots=True)
class AnalogyPair:
    """Two concepts in different domains with matching relational structure."""

    source: str  # concept in domain A
    target: str  # concept in domain B (the analog)
    structural_similarity: float  # 0..1, profile cosine similarity
    shared_relations: frozenset[RelationType]  # structural relations both have
    cross_domain: bool  # whether they live in different cortical columns

    @property
    def key(self) -> tuple[str, str]:
        """Canonical (sorted) key for deduplication."""
        a, b = sorted((self.source, self.target))
        return (a, b)


@dataclass(slots=True)
class AnalogyInsight:
    """A validated analogy-derived edge written into the network."""

    source: str  # edge source concept
    target: str  # edge target concept
    relation: RelationType
    confidence: float  # 0..1
    kind: str  # "similarity" (S~C mapping) or "transfer" (projected relation)
    analog_source: str  # the analog this was projected from
    analog_target: str  # the original target in the analog's domain
    basis: str  # human-readable basis for the insight
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def describe(self) -> str:
        """Return a human-readable summary of the analogy insight."""
        if self.kind == "similarity":
            return (
                f"Analogy: {self.source} is structurally like {self.target} "
                f"(shared relations: {self.basis})"
            )
        return (
            f"Analogy: {self.source} {self.relation.value} {self.target}, "
            f"by structural mapping from {self.analog_source} "
            f"{self.relation.value} {self.analog_target}"
        )


class AnalogyEngine:
    """Structure-mapping engine for cross-domain analogical inference.

    Finds concepts in different cortical columns whose relational
    structure matches, aligns their neighborhoods, and projects
    relations from one domain into the other as candidate inferences.

    Usage::

        engine = AnalogyEngine(network, embeddings)
        insights = engine.discover_analogies()
        written = engine.discover_and_accept()

    The engine is stateless across calls (it reads the network and
    writes edges). Insight history is retained for introspection.
    """

    def __init__(
        self,
        network: ConceptNetwork,
        embeddings: EmbeddingStore | None = None,
        min_similarity: float = _MIN_ANALOGY_SIMILARITY,
    ) -> None:
        """Initialize with a concept network and optional embeddings.

        Args:
            network: The concept network to read from and write to.
            embeddings: Optional embedding store. When present, embedding
                proximity is used as a corroborating signal during
                validation. When absent, validation is structural only.
            min_similarity: Minimum profile cosine similarity for two
                concepts to be considered analogs.
        """
        self.network = network
        self.embeddings = embeddings
        self.min_similarity = min_similarity
        self._insights: list[AnalogyInsight] = []
        self._pairs_found = 0
        self._inferences_proposed = 0
        self._inferences_rejected = 0

    @property
    def insights(self) -> list[AnalogyInsight]:
        """All validated analogy insights (copy)."""
        return list(self._insights)

    @property
    def insight_count(self) -> int:
        """Number of validated analogy insights."""
        return len(self._insights)

    @property
    def pairs_found(self) -> int:
        """Total analogous pairs ever discovered."""
        return self._pairs_found

    @property
    def inferences_proposed(self) -> int:
        """Total candidate inferences ever proposed."""
        return self._inferences_proposed

    @property
    def inferences_rejected(self) -> int:
        """Total candidate inferences rejected by validation."""
        return self._inferences_rejected

    # ─── Public API ───────────────────────────────────────────────

    def discover_analogies(
        self,
        max_pairs: int = _DEFAULT_MAX_PAIRS,
        max_inferences_per_pair: int = _DEFAULT_MAX_INFERENCES_PER_PAIR,
    ) -> list[AnalogyInsight]:
        """Find cross-domain analogies and return validated insights.

        This is the main discovery pass. It:

        1. Computes relation profiles for all quality concepts.
        2. Finds analogous pairs across cortical columns.
        3. For each pair, generates candidate inferences (similarity +
           relation transfers).
        4. Validates each inference against contradictions and for
           corroborating evidence.
        5. Returns the validated insights (does NOT write edges — call
           ``accept_insights`` or ``discover_and_accept`` to write).

        Args:
            max_pairs: Maximum number of analogous pairs to process.
            max_inferences_per_pair: Maximum candidate inferences per
                pair.

        Returns:
            List of validated analogy insights.
        """
        concept_ids = self.network.dream_concept_ids
        if len(concept_ids) < 2:
            return []

        # Build the inverted index: structural relation type -> concepts
        # that have it. This lets us find candidates that share at least
        # one structural relation with a seed without scanning all
        # concepts.
        profiles = self._compute_profiles(concept_ids)
        inverted = self._build_inverted_index(concept_ids, profiles)

        # Find analogous pairs across columns.
        pairs = self._find_analogous_pairs(
            concept_ids, profiles, inverted, max_pairs
        )
        self._pairs_found += len(pairs)

        if not pairs:
            return []

        # Generate and validate candidate inferences for each pair.
        all_insights: list[AnalogyInsight] = []
        for pair in pairs:
            inferences = self._generate_inferences(
                pair, profiles, max_inferences_per_pair
            )
            for inf in inferences:
                self._inferences_proposed += 1
                if self._validate_inference(inf, pair):
                    all_insights.append(inf)
                else:
                    self._inferences_rejected += 1

        if all_insights:
            logger.info(
                f"Analogy engine: {len(pairs)} pairs, "
                f"{self._inferences_proposed} inferences proposed, "
                f"{len(all_insights)} validated"
            )

        return all_insights

    def accept_insights(self, insights: list[AnalogyInsight] | None) -> int:
        """Write validated analogy insights into the concept network.

        Args:
            insights: The insights to accept. If None, runs
                ``discover_analogies`` first.

        Returns:
            Number of edges added.
        """
        if insights is None:
            insights = self.discover_analogies()

        added = 0
        for inf in insights:
            # Double-check the concepts still exist.
            if self.network.get_concept(inf.source) is None:
                continue
            if self.network.get_concept(inf.target) is None:
                continue
            self.network.add_edge(
                inf.source,
                inf.target,
                inf.relation,
                inf.confidence,
                origin=_ORIGIN_ANALOGY,
            )
            self._insights.append(inf)
            added += 1
        return added

    def discover_and_accept(
        self,
        max_pairs: int = _DEFAULT_MAX_PAIRS,
        max_inferences_per_pair: int = _DEFAULT_MAX_INFERENCES_PER_PAIR,
    ) -> int:
        """One-shot: discover analogies and write them into the network.

        Convenience method for the sleep consolidation phase.
        Returns the number of edges added.
        """
        insights = self.discover_analogies(max_pairs, max_inferences_per_pair)
        return self.accept_insights(insights)

    # ─── Relation profiles ───────────────────────────────────────

    def _relation_profile(self, cid: str) -> Counter[tuple[str, str]]:
        """Compute the relation profile of a concept.

        The profile is a Counter of ``(relation_value, direction)``
        pairs, where direction is "out" or "in". This captures both
        *which* relations a concept participates in and *how many* of
        each — a concept with 3 outgoing CAUSES edges has a different
        profile from one with 1, even though their signatures (the set
        of relation types) are identical.
        """
        profile: Counter[tuple[str, str]] = Counter()
        for edge in self.network.get_edges(cid, "out"):
            profile[(edge.relation.value, "out")] += 1
        for edge in self.network.get_edges(cid, "in"):
            profile[(edge.relation.value, "in")] += 1
        return profile

    def _compute_profiles(
        self, concept_ids: list[str]
    ) -> dict[str, Counter[tuple[str, str]]]:
        """Compute relation profiles for a list of concepts."""
        return {cid: self._relation_profile(cid) for cid in concept_ids}

    def _build_inverted_index(
        self,
        concept_ids: list[str],
        profiles: dict[str, Counter[tuple[str, str]]],
    ) -> dict[RelationType, set[str]]:
        """Build an inverted index: structural relation -> concept IDs.

        Maps each structural relation type to the set of concepts that
        participate in it (in either direction). This lets us find
        candidates sharing a structural relation without scanning all
        concepts.
        """
        inverted: dict[RelationType, set[str]] = {}
        for cid in concept_ids:
            profile = profiles[cid]
            for (rel_val, _direction) in profile:
                try:
                    rel = RelationType(rel_val)
                except ValueError:
                    continue
                if rel in _STRUCTURAL_RELATIONS:
                    inverted.setdefault(rel, set()).add(cid)
        return inverted

    @staticmethod
    def _profile_cosine(
        p1: Counter[tuple[str, str]],
        p2: Counter[tuple[str, str]],
    ) -> float:
        """Cosine similarity between two relation profiles.

        Treats each profile as a sparse vector over
        ``(relation, direction)`` keys. Cosine similarity captures both
        which relations are present and their relative proportions.
        """
        if not p1 or not p2:
            return 0.0
        # Dot product over shared keys.
        shared_keys = set(p1) & set(p2)
        dot = sum(p1[k] * p2[k] for k in shared_keys)
        if dot == 0:
            return 0.0
        # Magnitudes.
        mag1 = sum(v * v for v in p1.values()) ** 0.5
        mag2 = sum(v * v for v in p2.values()) ** 0.5
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return dot / (mag1 * mag2)

    @staticmethod
    def _shared_structural_relations(
        p1: Counter[tuple[str, str]],
        p2: Counter[tuple[str, str]],
    ) -> frozenset[RelationType]:
        """Structural relation types that both profiles share."""
        shared: set[RelationType] = set()
        for (rel_val, _direction) in set(p1) & set(p2):
            try:
                rel = RelationType(rel_val)
            except ValueError:
                continue
            if rel in _STRUCTURAL_RELATIONS:
                shared.add(rel)
        return frozenset(shared)

    # ─── Pair finding ────────────────────────────────────────────

    def _find_analogous_pairs(
        self,
        concept_ids: list[str],
        profiles: dict[str, Counter[tuple[str, str]]],
        inverted: dict[RelationType, set[str]],
        max_pairs: int,
    ) -> list[AnalogyPair]:
        """Find cross-domain concept pairs with matching structure.

        For each concept, looks up candidates that share at least one
        structural relation (via the inverted index), filters to
        cross-column pairs, and ranks by profile cosine similarity.
        """
        pairs: list[AnalogyPair] = []
        seen: set[tuple[str, str]] = set()

        for cid in concept_ids:
            if len(pairs) >= max_pairs:
                break
            profile = profiles[cid]
            if not profile:
                continue
            columns = self.network.get_columns_for(cid)

            # Gather candidate set: concepts sharing any structural
            # relation with cid (via inverted index).
            candidates: set[str] = set()
            for (rel_val, _direction) in profile:
                try:
                    rel = RelationType(rel_val)
                except ValueError:
                    continue
                if rel in _STRUCTURAL_RELATIONS:
                    candidates |= inverted.get(rel, set())

            # Rank candidates by profile cosine, prefer cross-column.
            scored: list[tuple[float, bool, str]] = []
            for cand in candidates:
                if cand == cid:
                    continue
                pair_key: tuple[str, str] = (cid, cand) if cid < cand else (cand, cid)
                if pair_key in seen:
                    continue
                # Skip if already connected (any edge between them).
                if self._are_connected(cid, cand):
                    continue
                cand_profile = profiles.get(cand)
                if cand_profile is None:
                    cand_profile = self._relation_profile(cand)
                    profiles[cand] = cand_profile
                sim = self._profile_cosine(profile, cand_profile)
                if sim < self.min_similarity:
                    continue
                shared = self._shared_structural_relations(profile, cand_profile)
                if not shared:
                    continue
                cand_cols = self.network.get_columns_for(cand)
                cross_domain = not (columns & cand_cols)
                # Prefer cross-domain, then higher similarity.
                scored.append((sim, cross_domain, cand))

            if not scored:
                continue

            # Sort: cross-domain first (True > False), then by similarity.
            scored.sort(key=lambda x: (not x[1], -x[0]))
            for sim, cross_domain, cand in scored:
                if len(pairs) >= max_pairs:
                    break
                pair_key = (cid, cand) if cid < cand else (cand, cid)
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                shared = self._shared_structural_relations(
                    profile, profiles[cand]
                )
                pairs.append(AnalogyPair(
                    source=cid,
                    target=cand,
                    structural_similarity=sim,
                    shared_relations=shared,
                    cross_domain=cross_domain,
                ))

        return pairs

    def _are_connected(self, c1: str, c2: str) -> bool:
        """Check if two concepts already have any edge between them."""
        for edge in self.network.get_edges(c1, "out"):
            if edge.target == c2:
                return True
        for edge in self.network.get_edges(c2, "out"):
            if edge.target == c1:
                return True
        return False

    # ─── Candidate inference generation ──────────────────────────

    def _generate_inferences(
        self,
        pair: AnalogyPair,
        profiles: dict[str, Counter[tuple[str, str]]],
        max_inferences: int,
    ) -> list[AnalogyInsight]:
        """Generate candidate inferences for an analogous pair.

        Two kinds of inference:
        1. **Similarity**: S SIMILAR_TO C — the analogical mapping
           itself, written so downstream analogical transfer can use it.
        2. **Transfer**: for each relation S has to a target T that C
           does not have to any target, project C --R--> T' where T' is
           the role-equivalent of T in C's domain. Symmetrically from
           C to S.
        """
        inferences: list[AnalogyInsight] = []

        s, c = pair.source, pair.target

        # 1. Analogical similarity edge (S ~ C).
        if not self._has_edge(s, c, RelationType.SIMILAR_TO):
            shared_names = ", ".join(
                sorted(r.value for r in pair.shared_relations)
            )
            inferences.append(AnalogyInsight(
                source=s,
                target=c,
                relation=RelationType.SIMILAR_TO,
                confidence=min(_SIMILARITY_WEIGHT, pair.structural_similarity),
                kind="similarity",
                analog_source=c,
                analog_target=s,
                basis=shared_names,
            ))

        # 2. Relation transfer in both directions.
        # S -> C: for each relation S has that C lacks, project.
        transfer_sc = self._transfer_relations(
            source=s,
            analog=c,
            profiles=profiles,
            max_inferences=max(0, max_inferences - len(inferences)),
        )
        inferences.extend(transfer_sc)

        # C -> S: symmetric.
        if len(inferences) < max_inferences:
            transfer_cs = self._transfer_relations(
                source=c,
                analog=s,
                profiles=profiles,
                max_inferences=max(0, max_inferences - len(inferences)),
            )
            inferences.extend(transfer_cs)

        return inferences[:max_inferences]

    def _transfer_relations(
        self,
        source: str,
        analog: str,
        profiles: dict[str, Counter[tuple[str, str]]],
        max_inferences: int,
    ) -> list[AnalogyInsight]:
        """Project relations from ``analog`` into ``source``'s domain.

        For each relation R that ``analog`` has (outgoing) to a target
        T, if ``source`` does not have R to any target, find the
        role-equivalent T' near ``source`` (the concept whose profile
        best matches T's) and predict ``source --R--> T'``.
        """
        if max_inferences <= 0:
            return []

        inferences: list[AnalogyInsight] = []

        # Relations the analog has outgoing, grouped by type.
        analog_out_by_rel: dict[RelationType, list[str]] = {}
        for edge in self.network.get_edges(analog, "out"):
            if edge.origin == _ORIGIN_ANALOGY:
                continue  # don't project from analogy edges (no cascades)
            if edge.relation not in _STRUCTURAL_RELATIONS:
                continue
            analog_out_by_rel.setdefault(edge.relation, []).append(edge.target)

        # Relations the source already has outgoing.
        source_out_rels: set[RelationType] = set()
        for edge in self.network.get_edges(source, "out"):
            source_out_rels.add(edge.relation)

        for rel, analog_targets in analog_out_by_rel.items():
            if len(inferences) >= max_inferences:
                break
            # Only transfer if source lacks this relation entirely.
            if rel in source_out_rels:
                continue

            for analog_target in analog_targets:
                if len(inferences) >= max_inferences:
                    break
                # Find the role-equivalent target near source.
                t_prime = self._find_role_equivalent(
                    source=source,
                    analog_target=analog_target,
                    profiles=profiles,
                )
                if t_prime is None:
                    continue
                if t_prime == source:
                    continue
                # Skip if the edge already exists.
                if self._has_edge(source, t_prime, rel):
                    continue

                # Confidence: structural similarity × target match × base.
                target_match = self._profile_cosine(
                    profiles.get(analog_target, self._relation_profile(analog_target)),
                    profiles.get(t_prime, self._relation_profile(t_prime)),
                )
                confidence = _TRANSFER_WEIGHT_BASE * max(target_match, 0.1)
                inferences.append(AnalogyInsight(
                    source=source,
                    target=t_prime,
                    relation=rel,
                    confidence=confidence,
                    kind="transfer",
                    analog_source=analog,
                    analog_target=analog_target,
                    basis=f"role-equivalent of {analog_target}",
                ))

        return inferences

    def _find_role_equivalent(
        self,
        source: str,
        analog_target: str,
        profiles: dict[str, Counter[tuple[str, str]]],
    ) -> str | None:
        """Find the concept near ``source`` that plays the same role as
        ``analog_target`` plays in the analog's domain.

        Searches source's 2-hop neighborhood for the concept whose
        relation profile best matches analog_target's profile. The
        role-equivalent is the concept that "does the same thing" in
        source's domain as analog_target does in the analog's domain.
        """
        target_profile = profiles.get(
            analog_target, self._relation_profile(analog_target)
        )
        if not target_profile:
            return None

        # Gather candidates: source's 1-hop and 2-hop neighbors.
        candidates: set[str] = set()
        one_hop: set[str] = set()
        for edge in self.network.get_edges(source, "out"):
            one_hop.add(edge.target)
            candidates.add(edge.target)
        for edge in self.network.get_edges(source, "in"):
            one_hop.add(edge.source)
            candidates.add(edge.source)
        for neighbor in one_hop:
            for edge in self.network.get_edges(neighbor, "out"):
                if edge.target != source:
                    candidates.add(edge.target)
            for edge in self.network.get_edges(neighbor, "in"):
                if edge.source != source:
                    candidates.add(edge.source)

        if not candidates:
            return None

        # Rank by profile cosine to the analog target.
        best: str | None = None
        best_score = 0.0
        checked = 0
        for cand in candidates:
            if checked >= _DEFAULT_MAX_TARGET_CANDIDATES:
                break
            checked += 1
            cand_profile = profiles.get(cand, self._relation_profile(cand))
            profiles[cand] = cand_profile  # cache for reuse
            score = self._profile_cosine(target_profile, cand_profile)
            if score > best_score:
                best_score = score
                best = cand

        if best is None or best_score < _MIN_TARGET_MATCH:
            return None
        return best

    # ─── Validation ──────────────────────────────────────────────

    def _validate_inference(
        self, inf: AnalogyInsight, pair: AnalogyPair
    ) -> bool:
        """Validate a candidate inference before writing it.

        Rejects if:
        - The edge contradicts existing edges (OPPOSITE_OF, CONTRADICTS,
          PREVENTS, HARMS between source and target).
        - A transfer inference has no corroborating evidence.

        Corroborating evidence (need >= 1 for transfers; similarity
        edges are always accepted if non-contradictory since the
        structural match itself is the evidence):
        - Shared hub: source and target both connect to a common
          concept via the same relation.
        - Embedding proximity: source and target are close in latent
          space (when embeddings are available).
        """
        # Contradiction check — always applies.
        if self._is_contradicted(inf):
            return False

        if inf.kind == "similarity":
            # The structural match IS the evidence for similarity.
            return True

        # Transfer inferences need at least one corroboration.
        corroboration = self._count_corroborations(inf)
        if corroboration < 1:
            return False
        return True

    def _is_contradicted(self, inf: AnalogyInsight) -> bool:
        """Check if the proposed edge contradicts existing edges."""
        src, tgt = inf.source, inf.target

        # Direct contradiction edges in either direction.
        for edge in self.network.get_edges(src, "out"):
            if edge.target == tgt and edge.relation in _CONTRADICTION_RELATIONS:
                return True
        for edge in self.network.get_edges(tgt, "out"):
            if edge.target == src and edge.relation in _CONTRADICTION_RELATIONS:
                return True

        # An IS_A proposal is contradicted if src and tgt are opposites.
        if inf.relation == RelationType.IS_A:
            for edge in self.network.get_edges(src, "out"):
                if (
                    edge.target == tgt
                    and edge.relation == RelationType.OPPOSITE_OF
                ):
                    return True

        return False

    def _count_corroborations(self, inf: AnalogyInsight) -> int:
        """Count independent evidence paths supporting a transfer inference.

        1. **Shared hub**: source and target both connect to a common
           concept H via the same relation type (structural context).
        2. **Embedding proximity**: source and target are close in
           latent space (when embeddings available, threshold 0.5).

        Only counts paths through non-analogy edges — analogy edges
        must not corroborate each other (circular validation).
        """
        count = 0
        src, tgt = inf.source, inf.target

        # 1. Shared hub: both connect to a common concept via the same
        #    relation type (any relation, not just the proposed one).
        src_neighbors: dict[str, set[RelationType]] = {}
        for edge in self.network.get_edges(src, "out"):
            if edge.origin == _ORIGIN_ANALOGY:
                continue
            src_neighbors.setdefault(edge.target, set()).add(edge.relation)
        for edge in self.network.get_edges(src, "in"):
            if edge.origin == _ORIGIN_ANALOGY:
                continue
            src_neighbors.setdefault(edge.source, set()).add(edge.relation)

        tgt_neighbors: dict[str, set[RelationType]] = {}
        for edge in self.network.get_edges(tgt, "out"):
            if edge.origin == _ORIGIN_ANALOGY:
                continue
            tgt_neighbors.setdefault(edge.target, set()).add(edge.relation)
        for edge in self.network.get_edges(tgt, "in"):
            if edge.origin == _ORIGIN_ANALOGY:
                continue
            tgt_neighbors.setdefault(edge.source, set()).add(edge.relation)

        shared_hubs = set(src_neighbors) & set(tgt_neighbors)
        if shared_hubs:
            count += 1

        # 2. Embedding proximity (when available).
        if self.embeddings is not None and self.embeddings.has_embeddings:
            try:
                similar = self.embeddings.find_similar_concepts(
                    src, k=20, threshold=0.5,
                )
                for neighbor, _score in similar:
                    if neighbor == tgt:
                        count += 1
                        break
            except Exception as e:  # noqa: BLE001
                logger.debug(f"embedding corroboration failed: {e}")

        return count

    def _has_edge(
        self, source: str, target: str, relation: RelationType
    ) -> bool:
        """Check if a specific typed edge exists between two concepts."""
        for edge in self.network.get_edges(source, "out"):
            if edge.target == target and edge.relation == relation:
                return True
        # SIMILAR_TO is symmetric — check the reverse too.
        if relation == RelationType.SIMILAR_TO:
            for edge in self.network.get_edges(target, "out"):
                if edge.source == source and edge.relation == relation:
                    return True
            for edge in self.network.get_edges(source, "in"):
                if edge.source == target and edge.relation == relation:
                    return True
        return False

    # ─── Persistence ─────────────────────────────────────────────

    def to_dict(self) -> dict[str, object]:
        """Serialize state for persistence."""
        return {
            "insights": [
                {
                    "source": i.source,
                    "target": i.target,
                    "relation": i.relation.value,
                    "confidence": i.confidence,
                    "kind": i.kind,
                    "analog_source": i.analog_source,
                    "analog_target": i.analog_target,
                    "basis": i.basis,
                    "timestamp": i.timestamp,
                }
                for i in self._insights
            ],
            "stats": {
                "pairs_found": self._pairs_found,
                "inferences_proposed": self._inferences_proposed,
                "inferences_rejected": self._inferences_rejected,
            },
        }

    def restore_from_dict(self, data: dict[str, object]) -> None:
        """Restore state from persistence."""
        self._insights = []
        insights_raw = data.get("insights", [])
        if not isinstance(insights_raw, list):
            return
        for item in insights_raw:
            if not isinstance(item, dict):
                continue
            relation_str = item.get("relation", "related_to")
            try:
                relation = RelationType(str(relation_str))
            except ValueError:
                relation = RelationType.RELATED_TO
            self._insights.append(AnalogyInsight(
                source=str(item["source"]),
                target=str(item["target"]),
                relation=relation,
                confidence=float(item.get("confidence", 0.0)),
                kind=str(item.get("kind", "transfer")),
                analog_source=str(item.get("analog_source", "")),
                analog_target=str(item.get("analog_target", "")),
                basis=str(item.get("basis", "")),
                timestamp=int(item.get("timestamp", 0)),
            ))
        stats = data.get("stats", {})
        if isinstance(stats, dict):
            self._pairs_found = int(stats.get("pairs_found", 0))
            self._inferences_proposed = int(stats.get("inferences_proposed", 0))
            self._inferences_rejected = int(stats.get("inferences_rejected", 0))
