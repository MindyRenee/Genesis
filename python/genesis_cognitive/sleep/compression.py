"""Algorithmic sleep compression — the compaction pass that keeps Genesis bounded.

## Modeling note

This module implements **memory compaction** — VQ codebook retraining,
edge migration into the holographic graph, and episode summarization —
inspired by but not equivalent to biological sleep consolidation.
Biological memory consolidation is selective replay and systems-level
transformation in which hippocampal episodic traces are gradually
integrated into neocortical semantic networks (Rasch & Born, 2013);
it is not destructive deletion. The "compact LTM" phase below deletes
raw episodes after extracting their semantic content, which is a
storage-management decision, not a model of memory transformation.
The synaptic-homeostasis renormalization is a weight-scaling operation
inspired by Tononi & Cirelli (2006), but it operates on graph edge
weights, not synaptic strengths. The biological language used below
("hippocampal → neocortical transfer," "synaptic reorganization")
marks analogies, not mechanistic claims.

This module implements the sleep compression algorithm that runs during
N3 slow-wave sleep. It is the third pillar of Genesis's bounded-growth
architecture, alongside vector quantization (VQ) and holographic
associative graphs.

# What sleep compression does

During wakefulness, Genesis accumulates:
- New concepts from learning and conversation
- New edges from reasoning and bridge creation
- New LTM episodes from every experience

Without compression, this growth is monotonic — she will eventually
exhaust disk and memory. Algorithmic sleep is the compaction pass that
keeps her bounded.

Three phases, mirroring biological sleep:

1. **Quantize** (hippocampal → neocortical transfer):
   Retrain the VQ codebook with current concept embeddings. Identify
   near-duplicate concepts (same prototype, tiny residual) as merge
   candidates. The codebook compresses the embedding space.

2. **Holographize** (synaptic reorganization):
   Migrate all auto-generated bridge edges into the holographic
   associative graph. Delete them from the explicit edge list. The
   explicit graph retains only "real" edges — stated, learned, inferred.
   The holographic graph handles fuzzy associations at fixed size.

3. **Compact LTM** (memory consolidation):
   Replay recent LTM episodes, extract their semantic content (concepts
   + associations), bind the associations into the holographic graph,
   and delete the raw episodes. Only high-salience episodes are retained
   verbatim. The episode's *meaning* persists in the compressed
   representation; the raw text is transient.

Finally, **synaptic homeostasis** (Tononi & Cirelli): renormalize all
edge weights so there's headroom for tomorrow's learning. This is the
global downscaling that prevents saturation and keeps the system
learnable.

# Integration

This module is called from ``Mind._consolidate_n3_heavy()`` after the
existing ``consolidate_during_sleep`` pass. It augments — does not
replace — the existing sleep consolidation.

# Neuroscience grounding

- **Replay & consolidation**: episodic memories (hippocampus) are
  replayed and transferred to semantic memory (neocortex). This is
  literally VQ codebook update — the day's episodes are compressed
  into prototypes and residuals. (McClelland et al., 1995)

- **Synaptic homeostasis**: synaptic weights that grew during the day
  are globally downscaled. This prevents saturation and keeps the
  system learnable. Algorithmically, this is renormalization. (Tononi
  & Cirelli, 2006)

- **Memory pruning**: weak/noisy associations are eliminated. In the
  holographic model, this is denoising — the sleep replay process
  re-encodes the strongest associations and lets the weak noise decay.

She wakes up smaller than she went to sleep — not because she forgot,
but because she *compressed*.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..concepts import HolographicGraph
from ..vq_codebook import VQCodebook

__all__ = ["SleepCompressor"]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork, EmbeddingStore


# ─── Constants ──────────────────────────────────────────────────

# VQ codebook size. 2K prototypes is sufficient for 10-20K concepts.
DEFAULT_VQ_K = 2048

# VQ training iterations.
DEFAULT_VQ_ITERS = 25

# LTM episode salience threshold for retention. Episodes above this
# salience are kept verbatim; lower-salience episodes are replayed and
# deleted after their semantic content is extracted.
DEFAULT_SALIENCE_THRESHOLD = 0.6

# Maximum LTM episodes to keep verbatim. Even high-salience episodes
# beyond this count are compacted.
DEFAULT_MAX_LTM_EPISODES = 500

# Maximum concepts to extract associations from per episode. Episodes
# that mention many concepts would generate excessive associations
# (most of which are noise — distant concepts that co-occur in a long
# text). Capping this mirrors biological memory consolidation: only
# the most salient features are consolidated, not every detail.
DEFAULT_MAX_CONCEPTS_PER_EPISODE = 20

# Synaptic homeostasis: global downscaling factor. All edge weights
# are multiplied by this factor after compression. 0.9 means 10%
# downscaling — enough to create headroom without erasing recent
# learning.
DEFAULT_HOMEOSTASIS_SCALE = 0.9

# Merge threshold for VQ: concepts with reconstructed L2 distance
# below this are considered merge candidates.
DEFAULT_MERGE_THRESHOLD = 0.02


class SleepCompressor:
    """Algorithmic sleep compression pass.

    Runs during N3 sleep to compress the concept network, migrate
    bridge edges to the holographic graph, and compact LTM episodes.
    This keeps Genesis's memory bounded across indefinite operation.

    The compressor is stateful — it holds references to the VQ codebook
    and holographic graph that persist across sleep cycles.
    """

    def __init__(
        self,
        data_dir: str,
        vq_k: int = DEFAULT_VQ_K,
        vq_iters: int = DEFAULT_VQ_ITERS,
        salience_threshold: float = DEFAULT_SALIENCE_THRESHOLD,
        max_ltm_episodes: int = DEFAULT_MAX_LTM_EPISODES,
        max_concepts_per_episode: int = DEFAULT_MAX_CONCEPTS_PER_EPISODE,
        homeostasis_scale: float = DEFAULT_HOMEOSTASIS_SCALE,
        merge_threshold: float = DEFAULT_MERGE_THRESHOLD,
    ) -> None:
        """Initialize the sleep compressor with persistent components.

        Loads (or creates) the VQ codebook and holographic graph from
        ``data_dir``, and prepares the concept matcher for episode
        scanning. All parameters have module-level defaults tuned for
        long-running operation.
        """
        self.data_dir = data_dir
        self.vq_k = vq_k
        self.vq_iters = vq_iters
        self.salience_threshold = salience_threshold
        self.max_ltm_episodes = max_ltm_episodes
        self.max_concepts_per_episode = max_concepts_per_episode
        self.homeostasis_scale = homeostasis_scale
        self.merge_threshold = merge_threshold

        # Persistent components
        self.vq_codebook = VQCodebook(k=vq_k)
        self.holographic_graph = HolographicGraph()

        # Paths
        self._vq_path = os.path.join(data_dir, "vq_codebook.npz")
        self._hgraph_path = os.path.join(data_dir, "holographic_graph.npz")

        # Load existing state if available
        self._load()

    # ─── Public API ─────────────────────────────────────────────

    def compress(
        self,
        network: ConceptNetwork,
        embeddings: EmbeddingStore | None = None,
        ltm_client: object | None = None,
        consolidation_intensity: float = 0.5,
    ) -> dict[str, int | float | str]:
        """Run the full sleep compression pass.

        This should be called during N3 sleep, after the existing
        ``consolidate_during_sleep`` pass has run.

        Args:
            network: The concept network to compress.
            embeddings: The embedding store (for VQ training). If None,
                VQ training is skipped.
            ltm_client: An LTM client for episode compaction. If None,
                LTM compaction is skipped. Expected to have methods
                ``search_episodes(limit, offset)`` and
                ``archive_episode(episode_id)``.
            consolidation_intensity: Brain-wave-derived consolidation
                drive (0..1). Scales the aggressiveness of LTM
                compaction and synaptic downscaling. High consolidation
                (delta/theta-dominant deep sleep) → more aggressive
                compression. Low consolidation (stress-disrupted sleep)
                → gentler compression to avoid losing fragile memories.

        Returns:
            Dict with compression statistics.

        Atomicity: The edge list is snapshotted before any mutations.
        If any phase fails, the snapshot is restored so the network is
        unchanged — the next sleep cycle can try again. Without this,
        a partial failure (e.g. phase 2 succeeds but phase 4 raises)
        would leave bridge edges removed from the network but not yet
        saved to the holographic graph on disk, causing data loss on
        restart.
        """
        stats: dict[str, int | float | str] = {}
        t0 = time.time()

        # Snapshot the edge list (with copied edges) before any
        # mutations. If a phase fails, we restore this snapshot so
        # the network is unchanged. This makes the compression pass
        # atomic — either it fully succeeds, or the network is
        # unchanged and the next pass can try again.
        #
        # We deep-copy the edges because homeostasis mutates
        # edge.weight in place — a shallow list copy wouldn't
        # protect against that.
        edge_snapshot = [copy.copy(e) for e in network.edges]

        try:
            # Phase 1: Quantize
            if embeddings is not None:
                stats.update(self._phase_quantize(network, embeddings))

            # Phase 2: Holographize
            stats.update(self._phase_holographize(network))

            # Phase 3: Compact LTM
            if ltm_client is not None:
                stats.update(self._phase_compact_ltm(network, ltm_client))

            # Synaptic homeostasis (scaled by brain-wave-derived
            # consolidation intensity — deep sleep downscales more
            # aggressively, stress-disrupted sleep is gentler).
            stats.update(
                self._apply_homeostasis(network, consolidation_intensity)
            )

        except Exception as e:  # noqa: BLE001
            # Restore the edge snapshot — the network is unchanged.
            # The holographic graph may have received some additive
            # associations (from phase 2 or 3), but those are
            # superimposed and idempotent — on the next pass, the
            # bridge edges will still be in the network (restored
            # from snapshot) and will be re-migrated. The holographic
            # graph's additive nature means duplicate bindings just
            # reinforce the same association, not create noise.
            network.replace_edges(edge_snapshot)
            logger.warning(
                "Sleep compression failed, network restored: %s", e,
            )
            stats["error"] = str(e)
            stats["elapsed_s"] = time.time() - t0
            return stats

        # Save state — only if all phases succeeded.
        # Periodically rebuild the holographic graph from the current
        # edge set to remove accumulated noise from additive
        # superposition. The holographic graph uses circular
        # convolution binding, and repeated add_edges calls can
        # accumulate interference. Rebuilding from a clean state
        # every ~10 compression passes restores signal clarity.
        self._maybe_rebuild_holographic_graph(network)
        self._save()

        stats["elapsed_s"] = time.time() - t0
        logger.info("Sleep compression complete: %s", stats)
        return stats

    def _maybe_rebuild_holographic_graph(self, network: ConceptNetwork) -> None:
        """Periodically rebuild the holographic graph from current bridge
        edges to clear accumulated additive-superposition noise.

        The holographic graph uses circular convolution binding, and
        repeated ``add_edges`` calls accumulate interference. Rebuilding
        from a clean state every ~10 compression passes restores signal
        clarity.
        """
        self._compress_cycle = getattr(self, "_compress_cycle", 0) + 1
        if self._compress_cycle % 10 == 0 and self.holographic_graph.edge_count > 0:
            try:
                # Rebuild from the current bridge edges in the network
                from ..concepts import _BRIDGE_ORIGINS
                all_bridge_edges: list[tuple[str, str, str, float]] = []
                for edge in network.edges:
                    if edge.origin in _BRIDGE_ORIGINS:
                        rel = edge.relation.value if hasattr(
                            edge.relation, "value"
                        ) else str(edge.relation)
                        all_bridge_edges.append((edge.source, rel, edge.target, edge.weight))
                self.holographic_graph.rebuild_from_edges(all_bridge_edges)
                logger.info(
                    "Rebuilt holographic graph from %d edges (cycle %d)",
                    len(all_bridge_edges), self._compress_cycle,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"holographic graph rebuild failed: {e}")

    # ─── Phase 1: Quantize ──────────────────────────────────────

    def _phase_quantize(
        self,
        _network: ConceptNetwork,  # kept for API symmetry with other phases
        embeddings: EmbeddingStore,
    ) -> dict[str, int | float]:
        """Retrain VQ codebook and identify merge candidates."""
        stats: dict[str, int | float] = {}

        # Use the existing concept matrix (already built by EmbeddingStore).
        # The matrix and names come from the embeddings store, not the
        # network — the network may have concepts that don't have
        # embeddings yet (e.g. newly learned concepts before refresh).
        matrix, names = embeddings.get_concept_matrix()

        if matrix is None or len(names) == 0:
            return {"vq_trained": 0, "merge_candidates": 0}

        # Train the codebook
        train_stats = self.vq_codebook.fit(
            matrix,
            names,
            iters=self.vq_iters,
        )
        stats["vq_trained"] = 1
        stats["vq_reconstruction_error"] = train_stats["final_error"]
        stats["vq_compression_ratio"] = self.vq_codebook.compression_ratio()

        # Find merge candidates
        merge_candidates = self.vq_codebook.find_merge_candidates(
            threshold=self.merge_threshold,
        )
        stats["merge_candidates"] = len(merge_candidates)

        if merge_candidates:
            logger.info(
                "VQ found %d merge candidates (threshold=%.3f)",
                len(merge_candidates), self.merge_threshold,
            )

        return stats

    # ─── Phase 2: Holographize ──────────────────────────────────

    def _phase_holographize(
        self,
        network: ConceptNetwork,
    ) -> dict[str, int | float]:
        """Migrate bridge edges to holographic graph."""
        from ..concepts import _BRIDGE_ORIGINS

        bridge_edges: list[tuple[str, str, str, float]] = []
        kept_edges: list = []

        for edge in network.edges:
            if edge.origin in _BRIDGE_ORIGINS:
                rel = edge.relation.value if hasattr(edge.relation, "value") else str(edge.relation)
                bridge_edges.append((
                    edge.source,
                    rel,
                    edge.target,
                    edge.weight,
                ))
            else:
                kept_edges.append(edge)

        # Add bridge edges to holographic graph
        self.holographic_graph.add_edges(bridge_edges)

        # Remove bridge edges from the explicit graph
        network.replace_edges(kept_edges)

        logger.info(
            "Holographize: migrated %d bridge edges, %d explicit edges remain",
            len(bridge_edges), len(kept_edges),
        )

        return {
            "bridges_migrated": len(bridge_edges),
            "explicit_edges_remaining": len(kept_edges),
            "holographic_edges": self.holographic_graph.edge_count,
        }

    # ─── Phase 3: Compact LTM ───────────────────────────────────

    @staticmethod
    def _build_concept_matcher(network: ConceptNetwork) -> Callable[[str], list[str]]:
        """Build a compiled regex that finds concept IDs in text.

        Returns a callable ``pattern(text) -> list[str]`` that finds
        all concept IDs mentioned in the given text. This replaces the
        old O(N) per-episode loop with a single O(T) regex scan.

        Concept IDs use underscores as word separators. We match both
        the underscore form and the space-separated form, with word
        boundaries to avoid false positives (e.g. "cat" matching
        "category"). Short concept IDs (≤2 chars) are skipped to
        avoid noise.
        """
        # Build alternation of escaped concept name patterns.
        # Sort by length descending so longer names match first
        # (e.g. "long_term_memory" before "long").
        names: list[str] = []
        for cid in network.concept_ids:
            if len(cid) <= 2:
                continue
            names.append(cid)
        names.sort(key=len, reverse=True)

        if not names:
            # No concepts to match — return a no-op matcher
            def _no_match(text: str) -> list[str]:
                """Return no matches when the network has no concepts."""
                return []
            return _no_match

        # Build a regex alternation. For each concept ID, match either
        # the underscore form or the space-separated form, with word
        # boundaries. We use a single alternation pattern for
        # efficiency — the regex engine compiles it into a trie.
        alternatives: list[str] = []
        for cid in names:
            escaped = re.escape(cid)
            space_form = re.escape(cid.replace("_", " "))
            alternatives.append(escaped)
            if space_form != escaped:
                alternatives.append(space_form)

        # Use word boundaries. For multi-word forms, \b applies at
        # the start and end of the full alternative.
        pattern_str = r"\b(?:" + "|".join(alternatives) + r")\b"
        try:
            compiled = re.compile(pattern_str, re.IGNORECASE)
        except re.error:
            # If the pattern is too complex or has an error, fall
            # back to a simple per-episode scan. This should never
            # happen with re.escape'd inputs, but we guard anyway.
            logger.warning("Concept matcher regex failed, using fallback")
            lower_names = [(cid, cid.replace("_", " ").lower()) for cid in names]

            def _fallback_match(text: str) -> list[str]:
                """Match concepts via substring scan when regex compilation fails."""
                text_lower = text.lower()
                return [cid for cid, lower_name in lower_names if lower_name in text_lower]
            return _fallback_match

        def _regex_match(text: str) -> list[str]:
            """Match concepts in text using the compiled regex, preserving order.

            Finds all matches and maps them back to concept IDs. The
            regex matches either the underscore form or the space form
            — results are normalized back to the underscore form.
            """
            seen: list[str] = []
            seen_set: set[str] = set()
            for m in compiled.finditer(text):
                matched = m.group()
                # Normalize: space form → underscore form
                cid = matched.lower().replace(" ", "_")
                if network.has_concept(cid) and cid not in seen_set:
                    seen_set.add(cid)
                    seen.append(cid)
            return seen

        return _regex_match

    def _phase_compact_ltm(
        self,
        network: ConceptNetwork,
        ltm_client: object,
    ) -> dict[str, int | float]:
        """Replay and compact LTM episodes.

        For each episode below the salience threshold:
        1. Extract concepts mentioned in the episode text
        2. Create associations between co-occurring concepts
        3. Bind the associations into the holographic graph
        4. Delete the raw episode

        High-salience episodes are retained verbatim (up to max_ltm_episodes).

        Paginates through all LTM episodes — the daemon caps each page
        at 1000 regardless of the requested limit, so we page through
        with increasing offsets until a page comes back short. Without
        pagination, only the first 1000 episodes would ever be compacted
        and the LTM would grow unboundedly past that point.
        """
        stats: dict[str, int | float] = {
            "ltm_replayed": 0,
            "ltm_archived": 0,
            "ltm_retained": 0,
            "ltm_associations_extracted": 0,
            "ltm_total_seen": 0,
        }

        # Paginate through all LTM episodes. The daemon caps each page
        # at 1000 (ipc.rs: limit.min(1000)), so we must issue repeated
        # calls with increasing offsets. We fetch all pages before
        # processing because the retain/compact decision depends on
        # global salience ranking — a low-salience episode on page 1
        # might rank higher than one on page 5.
        episodes = self._paginate_ltm_episodes(ltm_client)

        if not episodes:
            return stats

        stats["ltm_total_seen"] = len(episodes)

        # Sort by salience descending
        episodes_sorted = sorted(
            episodes,
            key=lambda e: getattr(e, "salience", 0.0),
            reverse=True,
        )

        # Retain high-salience episodes verbatim, capped at
        # max_ltm_episodes. Retention is salience-based, not
        # rank-based: episodes below the threshold are compacted
        # even when they'd rank inside the cap, and episodes at or
        # above the threshold are compacted when they fall beyond it.
        retained = [
            e for e in episodes_sorted
            if getattr(e, "salience", 0.0) >= self.salience_threshold
        ][: self.max_ltm_episodes]
        stats["ltm_retained"] = len(retained)
        retained_ids = {id(e) for e in retained}

        # Compact everything not retained — low-salience episodes and
        # high-salience overflow alike.
        to_compact = [e for e in episodes_sorted if id(e) not in retained_ids]
        associations_extracted = 0

        # Build a compiled regex for concept matching once, rather
        # than iterating every concept for every episode. The old
        # approach was O(N×M) — 7.5M substring searches for 15K
        # concepts × 500 episodes. A compiled regex alternation scans
        # the text in a single pass per episode, making it O(M×T)
        # where T is the average text length.
        #
        # Concept IDs use underscores as word separators (e.g.
        # "long_term_memory"). We match the underscore form and the
        # space-separated form, with word boundaries to avoid
        # false positives (e.g. "cat" matching "category").
        concept_pattern = self._build_concept_matcher(network)

        for episode in to_compact:
            associations_extracted += self._compact_episode(
                episode, concept_pattern, ltm_client, stats,
            )

        stats["ltm_associations_extracted"] = associations_extracted

        logger.info(
            "LTM compaction: seen %d, replayed %d, archived %d, retained %d, "
            "extracted %d associations",
            stats["ltm_total_seen"], stats["ltm_replayed"], stats["ltm_archived"],
            stats["ltm_retained"], associations_extracted,
        )

        return stats

    def _paginate_ltm_episodes(
        self, ltm_client: object,
    ) -> list:
        """Fetch all LTM episodes via pagination.

        The daemon caps each page at 1000 (ipc.rs: limit.min(1000)),
        so we must issue repeated calls with increasing offsets.
        Returns all episodes fetched, or an empty list on failure.
        """
        PAGE_SIZE = 1000
        episodes: list = []
        offset = 0
        try:
            while True:
                page = ltm_client.search_episodes(  # type: ignore[attr-defined]
                    limit=PAGE_SIZE, offset=offset,
                )
                if not page:
                    break
                episodes.extend(page)
                if len(page) < PAGE_SIZE:
                    break  # partial page → end of active set
                offset += PAGE_SIZE
        except AttributeError as e:
            logger.warning("LTM compaction skipped: %s", e)
            return []
        except OSError as e:
            if not episodes:
                logger.warning("LTM compaction skipped: %s", e)
                return []
            logger.warning(
                "LTM compaction partial: fetched %d episodes before %s",
                len(episodes), e,
            )
        return episodes

    def _compact_episode(
        self,
        episode,
        concept_pattern,
        ltm_client: object,
        stats: dict[str, int | float],
    ) -> int:
        """Extract associations from one episode and archive it.

        Every episode handed here is compacted — the retention
        decision is made by the caller (top salient episodes within
        the cap are kept verbatim and never reach this method).
        High-salience episodes beyond the cap are still compacted:
        their associations carry proportionally more weight via the
        ``salience * 0.5`` scaling.

        Returns the number of associations extracted.
        """
        salience = getattr(episode, "salience", 0.0)
        text = getattr(episode, "text", "")
        if not text:
            return 0

        mentioned = concept_pattern(text)
        if len(mentioned) > self.max_concepts_per_episode:
            mentioned = mentioned[:self.max_concepts_per_episode]

        associations = 0
        for i in range(len(mentioned)):
            for j in range(i + 1, min(i + 5, len(mentioned))):
                a, b = mentioned[i], mentioned[j]
                if a != b:
                    weight = salience * 0.5
                    self.holographic_graph.add(a, "related_to", b, weight)
                    associations += 1

        try:
            episode_id = getattr(episode, "episode_id", None)
            if episode_id is not None:
                ltm_client.archive_episode(episode_id)  # type: ignore[attr-defined]
                stats["ltm_archived"] += 1
        except AttributeError as e:
            logger.debug(f"LTM episode archive skipped (no episode_id): {e}")
        except OSError as e:
            logger.debug(f"LTM episode archive failed: {e}")

        stats["ltm_replayed"] += 1
        return associations

    # ─── Synaptic homeostasis ───────────────────────────────────

    def _apply_homeostasis(
        self, network: ConceptNetwork, consolidation_intensity: float = 0.5,
    ) -> dict[str, int | float]:
        """Apply global synaptic downscaling (SHY hypothesis).

        All edge weights are multiplied by ``homeostasis_scale``. This
        creates headroom for tomorrow's learning and prevents saturation.
        Edges that fall below the pruning threshold after downscaling
        are removed.

        The downscaling intensity is modulated by the brain-wave-derived
        consolidation drive. High consolidation (deep delta/theta sleep)
        → full downscaling. Low consolidation (stress-disrupted sleep)
        → gentler downscaling to preserve fragile memories.
        """
        # Scale the downscaling: high consolidation → more downscaling
        # (closer to the configured scale), low consolidation → less
        # downscaling (closer to 1.0 = no change). This mirrors how
        # stress-disrupted sleep impairs synaptic renormalization.
        effective_scale = 1.0 - (1.0 - self.homeostasis_scale) * max(
            0.0, min(1.0, consolidation_intensity)
        )

        scaled = 0
        pruned = 0
        kept_edges: list = []

        for edge in network.edges:
            edge.weight *= effective_scale
            if edge.weight < 0.05:
                pruned += 1
                continue
            kept_edges.append(edge)
            scaled += 1

        network.replace_edges(kept_edges)

        logger.info(
            "Synaptic homeostasis: scaled %d edges, pruned %d (scale=%.2f)",
            scaled, pruned, effective_scale,
        )

        return {
            "homeostasis_scaled": scaled,
            "homeostasis_pruned": pruned,
        }

    # ─── Persistence ────────────────────────────────────────────

    def _save(self) -> None:
        """Save VQ codebook and holographic graph to disk."""
        try:
            if self.vq_codebook.is_trained:
                self.vq_codebook.save(self._vq_path)
            self.holographic_graph.save(self._hgraph_path)
        except OSError as e:
            logger.warning("Failed to save sleep compression state: %s", e)

    def _load(self) -> None:
        """Load VQ codebook and holographic graph from disk."""
        vq_loaded = self.vq_codebook.load(self._vq_path)
        hg_loaded = self.holographic_graph.load(self._hgraph_path)
        if vq_loaded:
            logger.info("Loaded VQ codebook: %s", self.vq_codebook.stats())
        if hg_loaded:
            logger.info("Loaded holographic graph: %s", self.holographic_graph.stats())
