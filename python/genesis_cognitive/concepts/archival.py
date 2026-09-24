"""Concept archive — long-term spill/recall to cold storage."""

from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Any

from .classify import _column_of
from .types import Concept, ConceptCategory, ConceptModality, Edge, RelationType

logger = logging.getLogger(__name__)


class ArchivalMixin:
    """Mixin for :class:`ConceptNetwork` — see module docstring."""
    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see the package's core module).
        def __getattr__(self, name: str) -> Any: ...


    def attach_archive(self, archive: Any) -> None:
        """Attach a long-term memory archive to this network.

        Once attached, dormant concepts are spilled to the archive
        instead of being pruned. Spilled concepts can be recalled
        transparently via ``_resolve`` / ``get_concept``.

        Args:
            archive: A ``ConceptArchive`` instance.
        """
        self._archive = archive
    def has_concept_anywhere(self, cid: str) -> bool:
        """Check if a concept ID exists in working memory OR the archive.

        Unlike :meth:`has_concept` (which only checks working memory),
        this also checks the long-term archive — but WITHOUT recalling
        the concept. Use this when you need to know if a concept is
        known anywhere in the system without the side effect of
        pulling it back into working memory.
        """
        if cid in self._concepts:
            return True
        if self._archive is None:
            return False
        try:
            return self._archive.has_concept(cid)
        except (sqlite3.Error, RuntimeError) as e:
            logger.debug(f"archive has_concept failed for {cid!r}: {e}")
            return False
    @property
    def all_concept_ids(self) -> list[str]:
        """All concept IDs across working memory AND the archive.

        Unlike :attr:`concept_ids` (working memory only), this
        includes dormant concepts spilled to the long-term archive.
        Useful for diagnostics, deduplication checks, and any code
        that needs the complete set of known concepts.
        """
        ids = set(self._concepts.keys())
        if self._archive is not None:
            try:
                ids.update(self._archive.get_all_ids())
            except (sqlite3.Error, RuntimeError) as e:
                logger.debug(f"archive get_all_ids failed: {e}")
        return list(ids)
    def restore_archive_to_working_memory(self) -> int:
        """Recall ALL archived concepts back into working memory.

        Migrates every concept from the long-term archive back into
        the in-memory network. This is the inverse of
        :meth:`spill_dormant` — it loads the entire archive into RAM.

        Use this when you need the full concept network in working
        memory (e.g., before a bulk analysis pass, or to reset the
        working/archive split). After this call, the archive will be
        empty and all concepts will be in working memory.

        Returns:
            The number of concepts recalled from the archive.
        """
        if self._archive is None:
            return 0
        try:
            all_concepts = self._archive.get_all_concepts()
        except (sqlite3.Error, RuntimeError) as e:
            logger.warning(f"archive restore failed: {e}")
            return 0
        if not all_concepts:
            return 0
        recalled = 0
        for concept_id, concept_data in all_concepts:
            # Skip concepts already in working memory (e.g. recalled
            # by a concurrent thread between get_all_concepts and here).
            if concept_id in self._concepts:
                continue
            concept = self._dict_to_concept(concept_data)
            self._concepts[concept.id] = concept
            self._add_alias(concept.id, concept.id)
            for alias in concept.aliases:
                self._add_alias(alias, concept.id)
            for col in (concept.columns or {_column_of(concept.origin, concept.id)}):
                self._column_index.setdefault(col, set()).add(concept.id)
            # Recall archived edges for this concept.
            try:
                archived_edges = self._archive.recall_edges(concept.id)
            except (sqlite3.Error, RuntimeError) as e:
                logger.debug(f"edge recall during restore failed for {concept_id!r}: {e}")
                archived_edges = []
            for e_data in archived_edges:
                try:
                    relation = RelationType(e_data["relation"])
                except ValueError:
                    continue
                key = (e_data["source"], e_data["target"], relation)
                if key in self._edge_key_index:
                    continue
                edge = Edge(
                    source=e_data["source"],
                    target=e_data["target"],
                    relation=relation,
                    weight=e_data["weight"],
                    created_at=e_data["created_at"],
                    origin=e_data["origin"],
                )
                self._edges.append(edge)
                self._edge_index.setdefault(edge.source, []).append(edge)
                self._reverse_index.setdefault(edge.target, []).append(edge)
                self._edge_key_index[key] = edge
            # Remove from the archive (concept + aliases).
            try:
                self._archive.recall_concept(concept_id)
            except (sqlite3.Error, RuntimeError) as e:
                logger.debug(f"archive cleanup during restore failed for {concept_id!r}: {e}")
            recalled += 1
        # Invalidate caches
        self._concept_ids_cache: list[str] | None = None
        self._world_concept_ids_cache: list[str] | None = None
        self._quality_concept_ids_cache: list[str] | None = None
        self._search_index: dict[str, set[str]] | None = None
        if recalled > 0:
            logger.info(
                "Restored %d concepts from archive to working memory "
                "(working: %d, archive: %d)",
                recalled, len(self._concepts), self.archive_size,
            )
        return recalled
    def dedupe_archive(self) -> int:
        """Drop archive rows for concepts present in working memory.

        The archive's invariant is "dormant concepts live here instead
        of working memory." That invariant breaks when a save file
        written before a spill (autosave cadence, or a SIGKILL between
        spill and save) is restored: the concept re-enters working
        memory via the scratch-network transplant, which bypasses
        ``add_concept``'s archive check, and the stale archive row is
        never cleaned because every lookup finds the hot concept first.

        Call after state restore. Working memory always wins — the
        archived copy is by definition older. Returns the number of
        stale rows removed.
        """
        if self._archive is None:
            return 0
        try:
            archived_ids = self._archive.get_all_ids()
            dupes = [cid for cid in archived_ids if cid in self._concepts]
            if not dupes:
                return 0
            removed = self._archive.remove_concepts_batch(dupes)
        except (sqlite3.Error, RuntimeError) as e:
            logger.debug(f"archive dedupe failed: {e}")
            return 0
        if removed > 0:
            logger.info(
                "Deduped %d stale archive rows duplicated in working "
                "memory (archive: %d)",
                removed, self.archive_size,
            )
        return removed

    def _archive_spilled_edges(self, to_spill: set[str]) -> None:
        """Archive edges involving spilled concepts.

        Edges stay in working memory too (for fast traversal), but the
        archive copy survives if the working-memory edge is later pruned.
        """
        edge_dicts: list[dict[str, Any]] = []
        for edge in self._edges:
            if edge.source in to_spill or edge.target in to_spill:
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
                logger.debug(f"edge archive during spill failed: {e}")
    def _remove_spilled_from_working_memory(self, to_spill: set[str]) -> None:
        """Remove spilled concepts from working memory (concepts + aliases)."""
        for cid in to_spill:
            self._concepts.pop(cid, None)
            # Remove this concept's aliases from the alias map
            keys_to_clean: list[str] = []
            for k, cids in list(self._alias_map.items()):
                if cid in cids:
                    cids.remove(cid)
                    if not cids:
                        keys_to_clean.append(k)
            for k in keys_to_clean:
                self._alias_map.pop(k, None)
            # Remove from column index
            for col_cids in self._column_index.values():
                col_cids.discard(cid)

        # Invalidate caches
        self._concept_ids_cache = None
        self._world_concept_ids_cache = None
        self._quality_concept_ids_cache = None
        self._search_index = None
    def spill_dormant(
        self,
        activation_threshold: float = 0.01,
        protected_origins: frozenset[str] | None = None,
        max_in_memory: int = 15000,
    ) -> int:
        """Spill dormant concepts from working memory to the long-term archive.

        A concept is spillable if ALL of:
        - activation < ``activation_threshold`` (dormant — not recently used)
        - origin not in ``protected_origins`` (structural seeds are kept)
        - review_count == 0 (never reviewed via spaced repetition)

        Concepts are spilled in ascending order of activation (most
        dormant first) until the in-memory network is at or below
        ``max_in_memory`` concepts.

        Edges involving spilled concepts are also archived in the
        ``archived_edges`` table. They stay in working memory too
        (for fast traversal by hot concepts), but the archive copy
        survives if the working-memory edge is later pruned during
        sleep consolidation or edge cleanup.

        Returns the number of concepts spilled. Returns 0 if no
        archive is attached.
        """
        if self._archive is None:
            return 0

        # The spill protected set is smaller than _PROTECTED_ORIGINS
        # (which protects taught knowledge from edge pruning). For
        # spilling, we only protect structural seeds — identity,
        # introspection, etc. Dormant "learned" concepts (the bulk of
        # autonomous learner output) SHOULD be spilled, not kept in
        # working memory forever.
        if protected_origins is None:
            protected_origins = frozenset({
                "identity", "introspection", "foundational",
                "seeded", "structural",
            })

        # If we're under the limit, nothing to spill
        if len(self._concepts) <= max_in_memory:
            return 0

        # Find spillable concepts, sorted by activation (most dormant first)
        spillable: list[tuple[float, str]] = []
        for cid, concept in self._concepts.items():
            if concept.origin in protected_origins:
                continue
            activation = concept.activation or 0.0
            if activation >= activation_threshold:
                continue
            if concept.review_count > 0:
                continue
            spillable.append((activation, cid))

        # Sort by activation ascending (most dormant first)
        spillable.sort(key=lambda x: x[0])

        # How many to spill
        n_to_spill = min(len(spillable), len(self._concepts) - max_in_memory)
        if n_to_spill <= 0:
            return 0

        to_spill = {cid for _, cid in spillable[:n_to_spill]}

        # Serialize and archive in batch
        items: list[tuple[str, dict[str, Any], set[str]]] = []
        for cid in to_spill:
            c: Concept | None = self._concepts.get(cid)
            if c is None:
                continue
            concept_data = self._concept_to_dict(c)
            aliases = set(c.aliases)
            items.append((cid, concept_data, aliases))

        try:
            spilled = self._archive.archive_concepts_batch(items)
        except (sqlite3.Error, RuntimeError) as e:
            logger.warning(f"archive batch spill failed: {e}")
            spilled = 0

        if spilled > 0:
            # Archive edges involving spilled concepts.
            self._archive_spilled_edges(to_spill)
            # Remove from working memory (concepts + aliases, NOT
            # edges). Only reached when the batch COMMIT succeeded —
            # archive_concepts_batch is transactional, so spilled > 0
            # means every item is on disk. If the archive write failed,
            # the concepts must stay in working memory: removing them
            # here would silently revert the spill to destructive
            # pruning and lose the knowledge entirely.
            self._remove_spilled_from_working_memory(to_spill)
            try:
                archive_count = self._archive.count()
            except (sqlite3.Error, RuntimeError):
                archive_count = -1
            logger.info(
                "Spilled %d dormant concepts to archive "
                "(working: %d, archive: %d)",
                spilled,
                len(self._concepts),
                archive_count,
            )

        return spilled
    def _concept_to_dict(self, concept: Concept) -> dict[str, Any]:
        """Serialize a Concept to a dict for archiving."""
        return {
            "id": concept.id,
            "aliases": list(concept.aliases),
            "activation": concept.activation,
            "confidence": concept.confidence,
            "origin": concept.origin,
            "columns": list(concept.columns) if concept.columns else [],
            "created_at": concept.created_at,
            "review_count": concept.review_count,
            "last_reviewed": concept.last_reviewed,
            "properties": concept.properties,
            "category": concept.category.value,
            "is_animacy_detected": concept.is_animacy_detected,
            "modality": concept.modality.value,
            "is_semantic_hub": concept.is_semantic_hub,
        }
    def _dict_to_concept(self, data: dict[str, Any]) -> Concept:
        """Deserialize a dict back into a Concept."""
        try:
            category = ConceptCategory(data.get("category", "unknown"))
        except ValueError:
            category = ConceptCategory.UNKNOWN
        try:
            modality = ConceptModality(data.get("modality", "unknown"))
        except ValueError:
            modality = ConceptModality.UNKNOWN
        return Concept(
            id=data["id"],
            aliases=set(data.get("aliases") or []),
            activation=data.get("activation") or 0.0,
            confidence=data.get("confidence") or 0.5,
            origin=data.get("origin") or "conversation",
            columns=set(data.get("columns") or []),
            created_at=data.get("created_at") or 0,
            review_count=data.get("review_count") or 0,
            last_reviewed=data.get("last_reviewed") or 0,
            properties=data.get("properties") or {},
            category=category,
            is_animacy_detected=data.get("is_animacy_detected") or False,
            modality=modality,
            is_semantic_hub=bool(data.get("is_semantic_hub", False)),
        )
    def _recall_from_archive_by_id(self, concept_id: str) -> Concept | None:
        """Recall a concept from the archive by canonical ID.

        Restores the concept to working memory (``_concepts``,
        ``_alias_map``, ``_column_index``). Also restores any
        archived edges involving this concept that are no longer in
        working memory (e.g. they were pruned during sleep while the
        concept was archived). Edges that are still live in working
        memory are left untouched — the live edge wins.

        Returns the restored Concept, or None if not in the archive
        or if the archive is temporarily unavailable.
        """
        if self._archive is None:
            return None
        try:
            data = self._archive.recall_concept(concept_id)
        except (sqlite3.Error, RuntimeError) as e:
            logger.debug(f"archive recall by id failed for {concept_id!r}: {e}")
            return None
        if data is None:
            return None
        concept = self._dict_to_concept(data)
        self._concepts[concept.id] = concept
        # Restore aliases
        self._add_alias(concept.id, concept.id)
        for alias in concept.aliases:
            self._add_alias(alias, concept.id)
        # Restore column index
        for col in (concept.columns or {_column_of(concept.origin, concept.id)}):
            self._column_index.setdefault(col, set()).add(concept.id)

        # Restore archived edges that are no longer in working memory.
        # Edges that survived in working memory are left untouched.
        try:
            archived_edges = self._archive.recall_edges(concept.id)
        except (sqlite3.Error, RuntimeError) as e:
            logger.debug(f"edge recall failed for {concept_id!r}: {e}")
            archived_edges = []
        for e_data in archived_edges:
            try:
                relation = RelationType(e_data["relation"])
            except ValueError:
                continue  # unknown relation type
            key = (e_data["source"], e_data["target"], relation)
            if key in self._edge_key_index:
                continue  # live edge exists — skip
            edge = Edge(
                source=e_data["source"],
                target=e_data["target"],
                relation=relation,
                weight=e_data["weight"],
                created_at=e_data["created_at"],
                origin=e_data["origin"],
            )
            self._edges.append(edge)
            self._edge_index.setdefault(edge.source, []).append(edge)
            self._reverse_index.setdefault(edge.target, []).append(edge)
            self._edge_key_index[key] = edge

        # Invalidate caches
        self._concept_ids_cache = None
        self._world_concept_ids_cache = None
        self._quality_concept_ids_cache = None
        self._search_index = None
        return concept
    def _recall_from_archive_by_alias(self, alias: str) -> str | None:
        """Recall a concept from the archive by alias.

        Looks up the alias in the archive, recalls the best matching
        concept, and returns its canonical ID. Returns None if no
        matching concept is in the archive or if the archive is
        temporarily unavailable (e.g. connection closed during
        shutdown). Archive errors are non-fatal — a failed recall
        just means the concept isn't available, not that the entire
        cognitive pipeline should crash.

        If multiple concepts match (polysemy), each is tried in order
        until one is successfully recalled. This handles the case
        where an earlier match has orphaned aliases (concept deleted
        but alias row remains from a crash mid-recall).
        """
        if self._archive is None:
            return None
        try:
            concept_ids = self._archive.find_by_alias(alias)
        except (sqlite3.Error, RuntimeError) as e:
            logger.debug(f"archive recall by alias failed for {alias!r}: {e}")
            return None
        if not concept_ids:
            return None
        # Try each matching concept ID in order. If the first has
        # orphaned aliases (concept deleted but alias row remains),
        # fall through to the next match.
        for concept_id in concept_ids:
            concept = self._recall_from_archive_by_id(concept_id)
            if concept is not None:
                return concept.id
        return None
