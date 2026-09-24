"""Concept consolidation — sleep processing, bridges, review."""

from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING, Any

from .classify import _BRIDGE_ORIGINS, _column_of, _strip_prefix
from .types import Concept, Edge, RelationType

logger = logging.getLogger(__name__)


class ConsolidationMixin:
    """Mixin for :class:`ConceptNetwork` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        def __getattr__(self, name: str) -> Any: ...


    def create_bridge(self, concept_a: str, concept_b: str, weight: float = 0.3) -> Edge | None:
        """Create a pioneer bridge between two concepts in different columns.

        Bridges are inter-column connections (like cortical association
        fibers). They connect the same concept across different domains
        (e.g., "cognition" in dictionary ↔ "cognition" in identity).

        Bridge weight starts low (0.3) and strengthens through Hebbian
        co-activation during conversations. This mirrors how pioneer
        neurons create initial pathways that are later refined by
        experience.
        """
        col_a = self.get_column(concept_a)
        col_b = self.get_column(concept_b)
        if col_a == col_b:
            return None  # No bridge needed within same column

        return self.add_edge(
            concept_a, concept_b, RelationType.BRIDGES, weight=weight, origin="bridge"
        )
    def get_bridges(self, concept_id: str) -> list[Edge]:
        """Get all bridge edges connected to a concept (deduplicated)."""
        edges = self.get_edges(concept_id, direction="both")
        bridges = [e for e in edges if e.relation == RelationType.BRIDGES]
        # Deduplicate by edge identity (same edge can appear in both
        # outgoing and incoming indices)
        seen: set[int] = set()
        unique: list[Edge] = []
        for e in bridges:
            if id(e) not in seen:
                seen.add(id(e))
                unique.append(e)
        return unique
    def backfill_columns(self) -> dict[str, int]:
        """Backfill column membership for all existing concepts.

        Concepts loaded from persistence (before the `columns` field
        existed) have empty column sets. This method populates each
        concept's `columns` set based on its origin and ID prefix.

        Also creates pioneer bridges between concepts that share the
        same base name but live in different columns (e.g., "mind" in
        dictionary vs "python:Mind" in code).

        **Bridge creation only runs for concepts that were actually
        backfilled** (had empty `columns`). If no backfilling was
        needed, bridges are not recreated — they either already exist
        or were intentionally pruned during sleep consolidation.
        Recreating them every startup would defeat the pruning and
        cause unbounded edge growth (13K bridges per startup).

        Returns a dict with backfill statistics.
        """
        backfilled_ids = self._backfill_column_membership_tracked()

        if not backfilled_ids:
            # No concepts needed backfilling — bridges were already
            # created in a previous run. Don't recreate them.
            return {
                "concepts_backfilled": 0,
                "bridges_created": 0,
            }

        bridges_created = self._create_pioneer_bridges(backfilled_ids)
        semantic_bridges = self._create_semantic_bridges(backfilled_ids)
        bridges_created += semantic_bridges

        return {
            "concepts_backfilled": len(backfilled_ids),
            "bridges_created": bridges_created,
        }
    def _backfill_column_membership_tracked(self) -> set[str]:
        """Populate and remap column sets, returning the set of backfilled concept IDs."""
        backfilled_ids: set[str] = set()

        # Step 1: Populate columns for every concept
        # Also remap existing columns (e.g., "introspection" → "identity"
        # when the column mapping changes)
        for cid, concept in list(self._concepts.items()):
            if not concept.columns:
                col = _column_of(concept.origin, cid)
                concept.columns = {col}
                backfilled_ids.add(cid)
            else:
                # Remap existing columns using current _column_of logic
                new_cols = {
                    "identity" if col == "introspection" else col
                    for col in concept.columns
                }
                if new_cols != concept.columns:
                    concept.columns = new_cols
                    backfilled_ids.add(cid)

        return backfilled_ids
    def _create_pioneer_bridges(self, backfilled_ids: set[str] | None = None) -> int:
        """Create pioneer bridges between same-base-name concepts across columns.

        If ``backfilled_ids`` is provided, only create bridges involving
        at least one backfilled concept. This prevents recreating bridges
        that were intentionally pruned during sleep consolidation.
        """
        bridges_created = 0

        # Step 2: Find same-base-name concepts across columns and
        # create pioneer bridges between them.
        #
        # Extract the base name from each concept ID (strip python:/rust:
        # prefix, take last component after dots). Group by base name.
        # If a base name appears in multiple columns, create bridges.
        by_base: dict[str, list[str]] = {}
        for cid in list(self._concepts):
            base = _strip_prefix(cid, ("python:", "rust:", "identity:", "code:"))
            if "." in base:
                base = base.split(".")[-1]
            base_lower = base.lower()
            by_base.setdefault(base_lower, []).append(cid)

        for _base_lower, cids in by_base.items():
            if len(cids) < 2:
                continue

            # Group by column
            by_column: dict[str, str] = {}
            for cid in cids:
                col = self.get_column(cid)
                if col not in by_column:
                    by_column[col] = cid

            # If the base name exists in multiple columns, create bridges
            # between the first concept in each column
            if len(by_column) <= 1:
                continue
            col_concepts = list(by_column.values())
            for i in range(len(col_concepts)):
                for j in range(i + 1, len(col_concepts)):
                    bridges_created += self._try_create_bridge(
                        col_concepts[i], col_concepts[j], backfilled_ids
                    )

        return bridges_created
    def _add_bridge_if_new(
        self, a: str, b: str, weight: float, origin: str
    ) -> int:
        """Add a bridge edge between two concepts if it doesn't already exist.

        Returns 1 if a new bridge was created, 0 otherwise.
        """
        existing = self._edge_key_index.get((a, b, RelationType.BRIDGES))
        if existing is None:
            existing = self._edge_key_index.get((b, a, RelationType.BRIDGES))
        if existing is None:
            self.add_edge(
                a, b, RelationType.BRIDGES, weight=weight, origin=origin
            )
            return 1
        return 0
    def _try_create_bridge(
        self, a: str, b: str, backfilled_ids: set[str] | None
    ) -> int:
        """Try to create a pioneer bridge between two concepts.

        Returns 1 if a new bridge was created, 0 otherwise.
        Skips pairs where neither concept was just backfilled (when
        filtering by ``backfilled_ids``), and skips pairs that already
        have a bridge edge.
        """
        # If filtering by backfilled IDs, skip pairs where
        # neither concept was just backfilled
        if (
            backfilled_ids is not None
            and a not in backfilled_ids
            and b not in backfilled_ids
        ):
            return 0
        return self._add_bridge_if_new(a, b, 0.3, "bridge")
    def _create_semantic_bridges(self, backfilled_ids: set[str] | None = None) -> int:
        """Create semantic bridges between related concepts across columns.

        Same-name bridges only fire when base names match. But many
        cross-column connections are semantic, not lexical: "cognition"
        in the dictionary relates to "cognition" in code, "daemon" in
        dictionary relates to "start_daemon" in code.

        If ``backfilled_ids`` is provided, only create bridges involving
        at least one backfilled concept.

        Two methods are used:

        1. Manual semantic mapping for key concepts that don't share
           substrings (cognition→cognition, awareness→perception).
           These are the cortical association fibers that connect
           different-name but same-meaning concepts across domains.

        2. Substring matching for concepts where the dictionary name
           appears within the code name (daemon→start_daemon,
           emotion→emotion.EmotionalState). Optimized by indexing
           code concept names by their component words.
        """
        bridges_created = 0

        # Method 1: Manual semantic mapping
        bridges_created += self._create_manual_semantic_bridges(backfilled_ids)

        # Method 2: Substring matching (optimized)
        code_concepts, code_word_index = self._build_code_concept_index()
        bridges_created += self._create_substring_and_word_bridges(
            code_concepts, code_word_index, backfilled_ids
        )

        # Method 3: Code-to-code word matching
        bridges_created += self._create_code_to_code_bridges(code_concepts, backfilled_ids)

        return bridges_created
    def _create_manual_semantic_bridges(self, backfilled_ids: set[str] | None = None) -> int:
        """Create bridges via manual semantic mapping for key concepts."""
        bridges_created = 0

        # These are concepts where the dictionary term and code term
        # are different words for the same thing. Missing targets are
        # silently skipped (the code checks if the concept exists).
        semantic_map = {
            "cognition": ["python:cognition.CognitionEngine"],
            "awareness": ["python:perception", "python:perception.Perception"],
            "mind": ["python:mind", "python:mind.Mind", "python:genesis_cognitive.Mind"],
            "learning": [
                "python:autonomous_learner",
                "python:autonomous_learner.AutonomousLearner",
            ],
            "curiosity": ["python:curiosity", "python:curiosity.CuriosityEngine"],
            "reflection": ["python:reflection", "python:reflection.ReflectionEngine"],
            "reasoning": ["python:reasoning", "python:reasoning.ReasoningEngine"],
            "emotion": ["python:emotion", "python:emotion.EmotionalState"],
            "imagination": ["python:sleep.inner_life", "python:sleep.inner_life.InnerLife"],
            "self-awareness": ["python:self_model", "python:self_model.SelfModel"],
            "narrative": ["python:narrative", "python:narrative.NarrativeEngine"],
            "vocabulary": ["python:language"],
            "daemon": ["python:genesis_cli.start_daemon", "python:start_daemon"],
            "subcognitive": ["python:genesis_cli.start_daemon", "python:start_daemon"],
        }

        for dict_name, code_names in semantic_map.items():
            dict_cid = self._resolve(dict_name)
            if not dict_cid:
                continue
            # If filtering by backfilled IDs, skip if dict concept wasn't backfilled
            if backfilled_ids is not None and dict_cid not in backfilled_ids:
                continue
            for code_name in code_names:
                code_cid = self._resolve(code_name)
                if not code_cid:
                    continue
                bridges_created += self._add_bridge_if_new(
                    dict_cid, code_cid, 0.3, "semantic_bridge"
                )

        return bridges_created
    def _build_code_concept_index(
        self,
    ) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
        """Build index of code concept base names for fast lookup."""
        # Build index of code concept base names for fast lookup
        code_concepts: list[tuple[str, str]] = []  # (cid, base_lower)
        for cid, concept in list(self._concepts.items()):
            cols = concept.columns or {_column_of(concept.origin, cid)}
            if "code" not in cols:
                continue
            base = _strip_prefix(cid)
            code_concepts.append((cid, base.lower()))

        # For each non-code concept, check if it's a substring
        # of any code concept name. This connects dictionary/conversation/
        # learned concepts to code concepts that mention them.
        #
        # Also: split code concept names into words and match each
        # word to non-code concepts. E.g., "python:emotion.EmotionalState"
        # splits into "emotion", "emotional", "state" — each can match
        # a dictionary concept.
        # Build a word → code_concepts index for fast lookup
        code_word_index: dict[str, list[str]] = {}
        for code_cid, code_base_lower in code_concepts:
            # Split on non-alphanumeric to get component words
            import re as _re

            words = _re.findall(r"[a-z]+", code_base_lower)
            for w in words:
                code_word_index.setdefault(w, []).append(code_cid)

        return code_concepts, code_word_index
    def _create_substring_and_word_bridges(
        self,
        code_concepts: list[tuple[str, str]],
        code_word_index: dict[str, list[str]],
        backfilled_ids: set[str] | None = None,
    ) -> int:
        """Create substring and exact-word-match bridges for non-code concepts."""
        bridges_created = 0

        for cid, concept in list(self._concepts.items()):
            # If filtering by backfilled IDs, skip non-backfilled concepts
            if backfilled_ids is not None and cid not in backfilled_ids:
                continue
            cols = concept.columns or {_column_of(concept.origin, cid)}
            if "code" in cols:
                continue  # skip code-to-code (already connected)
            # Only single-word concepts (multi-word are less likely to match)
            if " " in cid:
                continue
            cid_lower = cid.lower()
            if len(cid_lower) < 4:  # skip very short words (too noisy)
                continue

            # Method 2a: substring match (original)
            for code_cid, code_base_lower in code_concepts:
                if cid_lower in code_base_lower and cid_lower != code_base_lower:
                    bridges_created += self._add_bridge_if_new(
                        cid, code_cid, 0.25, "semantic_bridge"
                    )

            # Method 2b: exact word match (new — catches more connections)
            for code_cid in code_word_index.get(cid_lower, []):
                if code_cid == cid:
                    continue
                bridges_created += self._add_bridge_if_new(
                    cid, code_cid, 0.3, "semantic_bridge"
                )

        return bridges_created
    def _create_code_to_code_bridges(
        self,
        code_concepts: list[tuple[str, str]],
        backfilled_ids: set[str] | None = None,
    ) -> int:
        """Connect code concepts that share significant words."""
        bridges_created = 0

        # Method 3: Code-to-code word matching
        # Connect code concepts that share significant words.
        # E.g., "python:emotion.EmotionalState" and
        # "python:emotion.assess_emotion" share "emotion" — bridge them.
        # This creates shortcuts across the module hierarchy.
        word_to_code: dict[str, list[str]] = {}
        for code_cid, code_base_lower in code_concepts:
            # If filtering by backfilled IDs, skip non-backfilled code concepts
            if backfilled_ids is not None and code_cid not in backfilled_ids:
                continue
            import re as _re2

            words = _re2.findall(r"[a-z]+", code_base_lower)
            for w in words:
                word_to_code.setdefault(w, []).append(code_cid)

        for _word, cids in word_to_code.items():
            if len(cids) < 2:
                continue
            # Connect pairs that share this word (cap per word)
            import itertools

            pairs = list(itertools.combinations(cids, 2))[:20]  # cap at 20 per word
            for a, b in pairs:
                bridges_created += self._add_bridge_if_new(
                    a, b, 0.15, "semantic_bridge"
                )

        return bridges_created
    def _compute_sleep_bridge_limits(
        self, n: int, edge_count: int
    ) -> tuple[int, int]:
        """Compute max bridges and hub attachments for sleep consolidation.

        Bridge creation scales DOWN as the network grows — a large
        network already has enough connectivity, and unbounded
        bridging produces noise. There is no hard edge cap; Genesis
        learns freely. But bridge creation is throttled for large
        networks to keep the signal-to-noise ratio healthy.
        """
        # Scale bridges inversely with network size. A small network
        # needs more bridges to build connectivity; a large network
        # already has rich connectivity and needs fewer new bridges.
        if n > 50000:
            return 25, 50
        elif n > 10000:
            return 50, 100
        elif n > 5000:
            return 50, 100
        elif n > 1000:
            return 100, 200
        else:
            return 50, 200
    def _run_sleep_associative_processing(self) -> int:
        """Run REM-sleep associative bridging and hub attachment.

        Returns total bridges created (associative + hub).
        """
        n = self.size
        edge_count = len(self._edges)

        # ── 6: Associative bridging (REM sleep) ──────────────────
        # During REM sleep, the brain finds novel connections between
        # seemingly unrelated memories. This is the "overnight insight"
        # effect — you wake up understanding something you didn't the
        # night before.
        #
        # We find pairs of concepts that:
        #   1. Are NOT directly connected
        #   2. Share 2+ common neighbors
        #   3. Are in different columns (cross-domain)
        # And create weak bridges between them.
        max_bridges, hub_max = self._compute_sleep_bridge_limits(n, edge_count)
        bridges_created = self._create_associative_bridges(max_new=max_bridges)

        # ── 7: Hub attachment — connect orphaned concepts to hubs ──
        # Concepts with only 1 edge are "orphaned" — they're connected
        # to their parent but nothing else. The brain connects new
        # neurons to established hubs. We find the top hub concepts
        # (by degree) and connect orphans to the hub that shares the
        # most word overlap with them.
        hub_bridges = self._attach_orphans_to_hubs(max_new=hub_max)

        return bridges_created + hub_bridges
    def _pre_spill_archive_edges(self) -> None:
        """Archive edges for dormant concepts before edge pruning.

        Edge pruning (``_strengthen_and_prune_edges``) runs early in
        ``consolidate_during_sleep``. If we wait until ``spill_dormant``
        (which runs late, after pruning) to archive edges, the edges
        involving dormant concepts will already have been pruned and
        there will be nothing to archive.

        This method identifies the same dormant concepts that
        ``spill_dormant`` would spill (activation < threshold, origin
        not protected, review_count == 0) and archives their edges
        BEFORE pruning happens. The edges stay in working memory too
        (for fast traversal), but the archive copy survives pruning.

        This is idempotent — ``archive_edges_batch`` uses INSERT OR
        REPLACE, so calling it multiple times for the same edges is
        safe. If a concept is not spilled (e.g. it gets activated
        before spill_dormant runs), its archived edges simply sit in
        the archive unused until the concept is eventually spilled or
        permanently deleted.
        """
        if self._archive is None:
            return

        # Mirror the spill criteria from spill_dormant
        protected = frozenset({
            "identity", "introspection", "foundational",
            "seeded", "structural",
        })
        activation_threshold = 0.01

        # Find dormant concepts (same criteria as spill_dormant)
        dormant_ids: set[str] = set()
        for cid, concept in self._concepts.items():
            if concept.origin in protected:
                continue
            if (concept.activation or 0.0) >= activation_threshold:
                continue
            if concept.review_count > 0:
                continue
            dormant_ids.add(cid)

        if not dormant_ids:
            return

        # Collect edges involving dormant concepts
        edge_dicts: list[dict[str, Any]] = []
        for edge in self._edges:
            if edge.source in dormant_ids or edge.target in dormant_ids:
                edge_dicts.append({
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation.value,
                    "weight": edge.weight,
                    "created_at": edge.created_at,
                    "origin": edge.origin,
                })

        if edge_dicts:
            try:
                self._archive.archive_edges_batch(edge_dicts)
            except (sqlite3.Error, RuntimeError) as e:
                logger.debug(f"pre-spill edge archive failed: {e}")
    def consolidate_during_sleep(self, max_in_memory: int = 15000) -> dict[str, int]:
        """Consolidate the concept network during sleep.

        This is the concept-network analogue of the Rust daemon's
        dreaming pass. During sleep, the network:

        1. **Strengthens** edges that have been activated (high weight
           or recently used). Well-established connections get stronger.
        2. **Prunes** weak edges (weight < 0.1) that haven't been
           reinforced. This is synaptic pruning — the brain removes
           unused connections during sleep.
        3. **Boosts** confidence of well-connected concepts. Concepts
           with many connections are probably real and important.
        4. **Decays** activation of all concepts. Sleep clears
           short-term activation, preparing for a fresh day.
        5. **Removes** isolated low-confidence concepts (calls
           clean_noise). Sleep is when the brain takes out the trash.
        6. **Spills** dormant concepts to the long-term archive (if
           attached). Instead of deleting dormant knowledge, it moves
           to disk where it can be recalled on demand. This is the
           brain's consolidation of short-term to long-term memory.
        7. **Vacuums** the archive (if attached and modified this
           cycle) to reclaim disk space from recalled or deleted rows.

        Args:
            max_in_memory: Maximum concepts to keep in working memory.
                If the network exceeds this, dormant concepts are
                spilled to the archive.

        Returns a dict with counts:
        - 'strengthened': number of edges strengthened
        - 'pruned': number of edges pruned
        - 'boosted': number of concepts boosted
        - 'decay_cleared': number of concepts decayed
        - 'noise_removed': number of noise concepts removed
        - 'bridges_created': number of associative + hub bridges created
        - 'spilled': number of concepts moved to long-term archive
        """
        # ── Pre-spill: Archive edges for dormant concepts BEFORE pruning ──
        # Edge pruning happens in step 1 below. If we wait until step 6
        # (spill_dormant) to archive edges, the edges involving dormant
        # concepts will already have been pruned and there will be
        # nothing to archive. By snapshotting the dormant concepts and
        # archiving their edges NOW, we preserve the relationships even
        # if they're later pruned from working memory.
        self._pre_spill_archive_edges()

        strengthened, pruned = self._strengthen_and_prune_edges()
        self._rebuild_edge_indices()
        boosted = self._boost_concept_confidence()
        decay_cleared = self._decay_concept_activations()

        # ── 5-7: Associative bridging, hub attachment, THEN noise removal
        # Order matters: we must try to connect orphans to hubs BEFORE
        # pruning them. Otherwise we'd delete concepts that could have
        # been connected. The old order (prune → connect) caused
        # permanent orphan accumulation because clean_noise ran first
        # and removed zero-edge concepts before _attach_orphans_to_hubs
        # ever had a chance to connect them.
        bridges_created = self._run_sleep_associative_processing()

        # ── 5: Remove noise (sleep takes out the trash) ──────────
        # Now prune concepts that STILL have no edges after hub
        # attachment attempted. These are truly unconnectable noise.
        noise_removed = len(self.clean_noise(min_confidence=0.4, min_edges=1))

        # ── 6: Spill dormant concepts to long-term archive ───────
        # After noise removal, move remaining dormant concepts to the
        # archive (if attached). This keeps working memory bounded
        # while preserving knowledge. The activation decay in step 4
        # has just pushed many concepts below the spill threshold,
        # making this the ideal moment to spill.
        spilled = self.spill_dormant(max_in_memory=max_in_memory)

        # ── 7: Vacuum the archive to reclaim disk space ──────────
        # Recalls (transparent lookups that pull concepts back from
        # the archive) and permanent deletions (remove_concept) leave
        # free pages in the SQLite file. VACUUM rebuilds the database
        # compactly. This is expensive, so only run it when the
        # archive was actually modified this cycle (concepts spilled
        # or noise removed, which deletes archived edges).
        if self._archive is not None and (spilled > 0 or noise_removed > 0):
            try:
                self._archive.vacuum()
            except (sqlite3.Error, RuntimeError) as e:
                logger.debug(f"archive vacuum during sleep failed: {e}")

        return {
            "strengthened": strengthened,
            "pruned": pruned,
            "boosted": boosted,
            "decay_cleared": decay_cleared,
            "noise_removed": noise_removed,
            "bridges_created": bridges_created,
            "spilled": spilled,
        }
    def _strengthen_and_prune_edges(self) -> tuple[int, int]:
        """Strengthen active edges and prune weak ones."""
        strengthened = 0
        pruned = 0

        # ── 1 & 2: Strengthen active edges, prune weak ones ──────
        #
        # Strengthening uses logistic growth: w += α * (1 - w)
        # This gives diminishing returns — edges near 1.0 get smaller
        # increments than edges near 0.5, preventing saturation and
        # preserving discriminative power. The fixed increment (w += 0.05)
        # caused all strong edges to saturate at 1.0, making them
        # indistinguishable.
        #
        # Logistic growth: w_new = w + α * (1 - w)
        # At w=0.5: increment = 0.05 * 0.5 = 0.025
        # At w=0.9: increment = 0.05 * 0.1 = 0.005
        # At w=0.99: increment = 0.05 * 0.01 = 0.0005
        STRENGTHEN_RATE = 0.05
        edges_to_keep = []
        for edge in list(self._edges):
            # Pruning threshold depends on origin:
            # - Taught knowledge (protected): prune at 0.1 (original)
            # - Auto-generated bridges: prune at 0.3 (aggressive)
            #   Bridge edges start at 0.15 and were never pruned because
            #   the 0.1 floor was below their creation weight. The higher
            #   threshold ensures weak bridges get recycled.
            threshold = 0.3 if edge.origin in _BRIDGE_ORIGINS else 0.1
            if edge.weight < threshold:
                pruned += 1
                continue  # remove this edge
            # Logistic strengthening (diminishing returns)
            if edge.weight > 0.4:
                edge.weight = min(1.0, edge.weight + STRENGTHEN_RATE * (1.0 - edge.weight))
                strengthened += 1
            edges_to_keep.append(edge)

        self._edges: list[Edge] = edges_to_keep
        return strengthened, pruned
    def _rebuild_edge_indices(self) -> None:
        """Rebuild edge indices after edge modification."""
        # Rebuild edge indices
        self._edge_index: dict[str, list[Edge]] = {}
        self._reverse_index: dict[str, list[Edge]] = {}
        self._edge_key_index = {}
        for edge in list(self._edges):
            self._edge_index.setdefault(edge.source, []).append(edge)
            self._reverse_index.setdefault(edge.target, []).append(edge)
            self._edge_key_index[(edge.source, edge.target, edge.relation)] = edge
    def _boost_concept_confidence(self) -> int:
        """Boost confidence of well-connected concepts."""
        boosted = 0

        # ── 3: Boost confidence of well-connected concepts ───────
        for cid, concept in list(self._concepts.items()):
            edge_count = len(self._edge_index.get(cid, [])) + len(self._reverse_index.get(cid, []))
            if edge_count >= 3 and concept.confidence < 0.8:
                concept.confidence = min(0.9, concept.confidence + 0.1)
                boosted += 1

        return boosted
    def _decay_concept_activations(self) -> int:
        """Decay activation of all concepts during sleep."""
        decay_cleared = 0

        # ── 4: Decay activation (sleep clears short-term activation) ──
        for concept in list(self._concepts.values()):
            activation = concept.activation or 0.0
            if activation > 0.1:
                concept.activation = max(0.0, activation - 0.3)
                decay_cleared += 1

        return decay_cleared
    def _create_associative_bridges(self, max_new: int = 50) -> int:
        """Create bridges between concepts that share common neighbors.

        This is the REM-sleep associative function: concepts that share
        2+ neighbors but aren't directly connected get a weak bridge.
        This reduces average path length and creates cross-domain links.

        Only runs on concepts with at least 2 edges (well-connected
        enough to have meaningful neighbors). Skips pairs that already
        have any edge between them.
        """
        if self.size < 10:
            return 0

        # Build neighbor sets for each concept
        neighbors: dict[str, set[str]] = {}
        for edge in list(self._edges):
            neighbors.setdefault(edge.source, set()).add(edge.target)
            neighbors.setdefault(edge.target, set()).add(edge.source)

        # Only consider concepts with 2+ neighbors
        candidates = {cid: neigh for cid, neigh in neighbors.items() if len(neigh) >= 2}

        if len(candidates) < 2:
            return 0

        # For each pair, count shared neighbors
        cids = list(candidates.keys())
        created = 0
        import random as _rng

        _rng.shuffle(cids)

        for i in range(len(cids)):
            if created >= max_new:
                break
            a = cids[i]
            a_neigh = candidates[a]
            a_cols = self._concepts[a].columns or set()
            for j in range(i + 1, len(cids)):
                if created >= max_new:
                    break
                created += self._try_associative_bridge(
                    a, cids[j], a_neigh, candidates, a_cols
                )

        return created
    def _try_associative_bridge(
        self,
        a: str,
        b: str,
        a_neigh: set[str],
        candidates: dict[str, set[str]],
        a_cols: set[str],
    ) -> int:
        """Try to create an associative bridge between two concepts.

        Returns 1 if a new bridge was created, 0 otherwise. Bridges are
        created when the two concepts share 2+ neighbors and are not
        already directly connected. Cross-column bridges get a bonus.
        """
        # Skip if already connected
        if b in a_neigh or a in candidates[b]:
            return 0
        # Count shared neighbors
        shared = a_neigh & candidates[b]
        if len(shared) < 2:
            return 0
        # Prefer cross-column bridges
        b_cols = self._concepts[b].columns or set()
        cross_domain = not (a_cols & b_cols)
        # Create the bridge
        weight = 0.2 + 0.1 * min(len(shared), 5)  # 0.3–0.7
        if cross_domain:
            weight += 0.1  # bonus for cross-domain
        self.add_edge(
            a,
            b,
            RelationType.RELATED_TO,
            weight=min(weight, 0.6),
            origin="associative_bridge",
        )
        return 1
    def _attach_orphans_to_hubs(self, max_new: int = 200) -> int:
        """Connect orphaned concepts (≤2 edges) to relevant hubs.

        Finds concepts with only 1-2 edges and connects them to hub
        concepts (5+ edges) that share word overlap. This is
        preferential attachment — new concepts gravitate toward
        well-connected hubs, which is how scale-free networks
        like the brain grow.

        Each orphan is connected to up to 3 matching hubs to ensure
        it's within 2-3 hops of the network core. Orphans with no
        word match get connected to super hubs (top 20 by degree)
        to guarantee global connectivity.
        """
        if self.size < 50:
            return 0

        import re as _re

        # Find hubs (concepts with 5+ edges) and rank by degree
        degree: dict[str, int] = {}
        for edge in list(self._edges):
            degree[edge.source] = degree.get(edge.source, 0) + 1
            degree[edge.target] = degree.get(edge.target, 0) + 1

        hubs = {cid: deg for cid, deg in degree.items() if deg >= 5}
        if not hubs:
            return 0

        # Build hub word index
        hub_words: dict[str, list[str]] = {}  # word → list of hub cids
        for hub_cid in hubs:
            base = _strip_prefix(hub_cid.lower())
            words = _re.findall(r"[a-z]+", base)
            for w in words:
                hub_words.setdefault(w, []).append(hub_cid)

        # Also pick top 20 hubs by degree as "super hubs" — connect
        # orphans with no word match to these to ensure connectivity
        super_hubs = sorted(hubs.items(), key=lambda x: -x[1])[:20]
        super_hub_cids = [cid for cid, _ in super_hubs]

        # Find orphans (concepts with ≤2 edges).
        # CRITICAL: also include concepts with ZERO edges — they don't
        # appear in the degree dict at all (it's built from edges), so
        # without this they're invisible to the orphan-attachment system.
        # This was the root cause of 7,961 disconnected "learned" concepts
        # that Genesis "knew" but could never use in reasoning.
        orphans = [cid for cid, deg in degree.items() if deg <= 2]
        zero_edge_concepts = [
            cid for cid in self._concepts
            if cid not in degree
        ]
        orphans.extend(zero_edge_concepts)

        # Don't try to attach concepts that are below the clean_noise
        # confidence threshold. Those are likely noise and should be
        # left for clean_noise to remove instead of being connected to
        # hubs (which would give them edges and let them survive).
        MIN_ATTACHMENT_CONFIDENCE = 0.4
        orphans = [
            cid for cid in orphans
            if self._concepts[cid].confidence >= MIN_ATTACHMENT_CONFIDENCE
        ]

        if not orphans:
            return 0

        created = 0
        import random as _rng

        _rng.shuffle(orphans)

        for orphan in orphans:
            if created >= max_new:
                break
            created = self._connect_orphan(
                orphan, hub_words, super_hub_cids, created, max_new, _re, _rng)

        return created
    def _connect_orphan(
        self,
        orphan: str,
        hub_words: dict[str, list[str]],
        super_hub_cids: list[str],
        created: int,
        max_new: int,
        _re,
        _rng,
    ) -> int:
        """Connect a single orphan to matching hubs or super hubs."""
        # Get words from orphan concept name
        base = _strip_prefix(orphan.lower())
        orphan_words = set(_re.findall(r"[a-z]+", base))

        # Determine if the orphan is a code concept or a world concept.
        # Code concepts (python:..., rust:...) should not be word-matched
        # to world concepts (Wikipedia phrases, dictionary entries) —
        # "cartoon network" should NOT bridge to
        # python:genesis_cognitive.concepts.relationtype just
        # because they share the word "network". Cross-domain bridges
        # are created by _create_semantic_bridges, which uses curated
        # manual mappings and substring matching, not raw word overlap.
        orphan_is_code = orphan.startswith(("python:", "rust:"))

        # Find matching hubs by word overlap
        matches: list[tuple[int, str]] = []  # (overlap, hub_cid)
        for word in orphan_words:
            for hub_cid in hub_words.get(word, []):
                if hub_cid == orphan or hub_cid in self._edge_index.get(orphan, []):
                    continue
                # Skip cross-domain word-overlap matches: code↔world
                # connections via shared words (e.g. "network" in both
                # "cartoon network" and a code concept) are semantically
                # meaningless and pollute the concept network.
                hub_is_code = hub_cid.startswith(("python:", "rust:"))
                if orphan_is_code != hub_is_code:
                    continue
                hub_base = _strip_prefix(hub_cid.lower())
                hub_word_set = set(_re.findall(r"[a-z]+", hub_base))
                overlap = len(orphan_words & hub_word_set)
                if overlap > 0:
                    matches.append((overlap, hub_cid))

        # Sort by overlap (best matches first)
        matches.sort(key=lambda x: -x[0])

        # Connect to top 3 matching hubs
        connected = 0
        for _, hub_cid in matches[:3]:
            if created >= max_new:
                break
            self.add_edge(
                orphan,
                hub_cid,
                RelationType.RELATED_TO,
                weight=0.2,
                origin="hub_attachment",
            )
            created += 1
            connected += 1

        # If no word match, connect to 2 random super hubs
        # to ensure the orphan is reachable from the core.
        # Apply the same cross-domain filter: don't connect world
        # concepts to code super hubs (and vice versa).
        if connected == 0 and super_hub_cids:
            _rng.shuffle(super_hub_cids)
            for hub_cid in super_hub_cids[:2]:
                if created >= max_new:
                    break
                if hub_cid == orphan or hub_cid in self._edge_index.get(orphan, []):
                    continue
                hub_is_code = hub_cid.startswith(("python:", "rust:"))
                if orphan_is_code != hub_is_code:
                    continue
                self.add_edge(
                    orphan,
                    hub_cid,
                    RelationType.RELATED_TO,
                    weight=0.1,
                    origin="hub_attachment",
                )
                created += 1

        return created
    def connect_all_orphans(self) -> int:
        """One-shot: connect EVERY weakly-connected concept to super hubs.

        This is the nuclear option — after all learning is done,
        run this to guarantee every concept is within 2-3 hops
        of the network core. No caps, no limits.

        Connects concepts with ≤5 edges to 3 super hubs each.
        """
        if self.size < 50:
            return 0

        import random as _rng

        # Compute degree
        degree: dict[str, int] = {}
        for edge in list(self._edges):
            degree[edge.source] = degree.get(edge.source, 0) + 1
            degree[edge.target] = degree.get(edge.target, 0) + 1

        # Top 50 hubs by degree = super hubs
        ranked = sorted(degree.items(), key=lambda x: -x[1])
        super_hubs = [cid for cid, deg in ranked[:50] if deg >= 10]
        if not super_hubs:
            return 0

        # Connect every weakly-connected concept (≤5 edges) to 3 super hubs
        weak = [cid for cid, deg in degree.items() if deg <= 5]
        created = 0
        for orphan in weak:
            # Pick 3 random super hubs
            picks = _rng.sample(super_hubs, min(3, len(super_hubs)))
            for hub in picks:
                if hub == orphan or hub in self._edge_index.get(orphan, []):
                    continue
                self.add_edge(
                    orphan,
                    hub,
                    RelationType.RELATED_TO,
                    weight=0.1,
                    origin="global_connect",
                )
                created += 1

        return created
    def get_review_queue(self, now_ms: int | None = None, max_items: int = 10) -> list[str]:
        """Get concepts due for review (spaced repetition).

        Returns a list of concept IDs that are due for review,
        sorted by urgency (most overdue first).

        A concept is due for review if:
        - It has been reviewed before and enough time has passed
          since the last review based on the spaced repetition schedule
        - OR it has never been reviewed (review_count == 0) and was
          created more than 1 minute ago

        Concepts with high confidence are reviewed less frequently.
        Concepts with low confidence are reviewed more frequently.
        """
        if now_ms is None:
            now_ms = int(time.time() * 1000)

        due: list[tuple[int, str]] = []  # (overdue_ms, concept_id)

        for cid, concept in list(self._concepts.items()):
            overdue = self._compute_review_overdue(concept, now_ms)
            if overdue is not None:
                due.append((overdue, cid))

        # Sort by urgency (most overdue first)
        due.sort(key=lambda x: -x[0])
        return [cid for _, cid in due[:max_items]]
    def _compute_review_overdue(self, concept: Concept, now_ms: int) -> int | None:
        """Compute how overdue a concept is for review, or None if not due.

        - Never reviewed: due if created > 1 minute ago.
        - Previously reviewed: due if enough time has passed for the
          next spaced-repetition interval (adjusted by confidence).
        """
        if concept.review_count == 0:
            # Never reviewed — due if created > 1 minute ago
            age = now_ms - concept.created_at
            if age > 60_000:
                return age - 60_000
            return None
        # Get the review interval for this concept
        interval_idx = min(concept.review_count, len(self._REVIEW_INTERVALS_MS) - 1)
        interval = self._REVIEW_INTERVALS_MS[interval_idx]

        # High-confidence concepts get longer intervals
        if concept.confidence > 0.7:
            interval = int(interval * 1.5)
        # Low-confidence concepts get shorter intervals
        elif concept.confidence < 0.4:
            interval = int(interval * 0.5)

        time_since = now_ms - concept.last_reviewed
        if time_since >= interval:
            return time_since - interval
        return None
    def review_concept(
        self, concept_id: str, success: bool = True, now_ms: int | None = None
    ) -> None:
        """Mark a concept as reviewed.

        If success=True, increment the review count (which extends
        the next review interval). If success=False, reset the
        review count (which shortens the next review interval) and
        slightly lower confidence.

        This is called when Genesis actively thinks about or uses
        a concept — confirming it still "knows" it.
        """
        concept = self._concepts.get(concept_id)
        if not concept:
            return

        if now_ms is None:
            now_ms = int(time.time() * 1000)

        concept.last_reviewed = now_ms
        if success:
            concept.review_count += 1
            # Successful review slightly boosts confidence
            concept.confidence = min(1.0, concept.confidence + 0.02)
        else:
            # Failed review — reset count, lower confidence
            concept.review_count = max(0, concept.review_count - 1)
            concept.confidence = max(0.1, concept.confidence - 0.05)
