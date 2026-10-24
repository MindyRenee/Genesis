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
- New edges from frontal_lobe.reasoning and bridge creation
- New LTM episodes from every experience

Without compression, this growth is monotonic — it will eventually
exhaust disk and memory. Algorithmic sleep is the compaction pass that
keeps it bounded.

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

Synaptic homeostasis is *not* a phase of this pass: SHY
renormalisation is a continuous process that lives in the
``SynapticDownscaler`` on her inner-life loop — every N3 tick
downscales weights proportionally at a rate set by accumulated
synaptic load and the body's sleep pressure (Tononi & Cirelli:
discharge rate ∝ sleep pressure). The old lumped renormalisation
here — one multiplicative sweep per pass — was replaced by that
continuous model; what remains is the compression itself.

# Integration

This module's steps are queued by ``Mind._build_n3_work()`` after the
existing ``consolidate_during_sleep`` pass. It augments — does not
replace — the existing sleep consolidation.

# Neuroscience grounding

- **Replay & consolidation**: episodic memories (hippocampus) are
  replayed and transferred to semantic memory (neocortex). This is
  literally VQ codebook update — the day's episodes are compressed
  into prototypes and residuals. (McClelland et al., 1995)

- **Synaptic homeostasis**: synaptic weights that grew during the day
  are globally downscaled. This prevents saturation and keeps the
  system learnable. Modelled continuously by the
  ``SynapticDownscaler`` — proportional ``w −= rate·dt·w`` per N3
  tick, rate gated by synaptic load and the body's adenosine/sleep
  pressure — not by a pass here. (Tononi & Cirelli, 2006)

- **Memory pruning**: weak/noisy associations are eliminated. In the
  holographic model, this is denoising — the sleep replay process
  re-encodes the strongest associations and lets the weak noise decay.

It wakes up smaller than it went to sleep — not because it forgot,
but because it *compressed*.
"""

from __future__ import annotations

import copy
import logging
import os
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from temporal_lobe.concepts import HolographicGraph

from genesis_client.swallow import note_swallowed
from infrastructure.vq_codebook import VQCodebook

__all__ = ["SleepCompressor"]

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from temporal_lobe.concepts import ConceptNetwork, EmbeddingStore


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

# Merge threshold for VQ: concepts with reconstructed L2 distance
# below this are considered merge candidates.
DEFAULT_MERGE_THRESHOLD = 0.02

# LTM source module for dream insights (the daemon's Dreaming
# module). Dream episodes are not compaction fodder: they are
# internal cognitive events that exist to be surfaced by recency —
# wake residues, ``/sleep`` status, and dream notifications all read
# ``get_recent_episodes(source_module=9)``, which sees only the
# active list. Compaction extracts nothing from their
# ``[dream-insight]`` bookkeeping text, and archiving keeps the
# payload on disk anyway, so sweeping them buys nothing and costs
# the entire visible dream record. They sit outside the
# retain/compact economy entirely.
_DREAM_MODULE = 9


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
        """Run the full sleep compression pass in one call.

        The whole step list pumped end-to-end — kept for callers that
        genuinely want a monolithic pass (tests, maintenance). The
        sleep loop uses :meth:`compress_steps` instead so the work is
        spread across the stage.

        Args:
            network: The concept network to compress.
            embeddings: The embedding store (for VQ training). If None,
                VQ training is skipped.
            ltm_client: An LTM client for episode compaction. If None,
                LTM compaction is skipped. Expected to have methods
                ``search_episodes(limit, offset)`` and
                ``archive_episode(episode_id)``.
            consolidation_intensity: Retained for API compatibility;
                consolidation gating now lives in the sleep loop's
                chemistry, not in this pass.

        Returns:
            Dict with compression statistics.

        Atomicity: The edge list is snapshotted before any mutations.
        If any step fails, the snapshot is restored so the network is
        unchanged — the next sleep cycle can try again.
        """
        stats: dict[str, int | float | str] = {}
        t0 = time.time()

        edge_snapshot = [copy.copy(e) for e in network.edges]

        try:
            for _name, step in self.compress_steps(
                network, embeddings, ltm_client
            ):
                out = step()
                if isinstance(out, dict):
                    stats.update(out)
        except Exception as e:  # noqa: BLE001
            # Restore the edge snapshot — the network is unchanged.
            # The holographic graph may have received some additive
            # associations, but those are superimposed and idempotent.
            network.replace_edges(edge_snapshot)
            logger.warning(
                "Sleep compression failed, network restored: %s",
                e,
            )
            stats["error"] = str(e)
            stats["elapsed_s"] = time.time() - t0
            return stats

        stats["elapsed_s"] = time.time() - t0
        logger.info("Sleep compression complete: %s", stats)
        return stats

    def compress_steps(
        self,
        network: ConceptNetwork,
        embeddings: EmbeddingStore | None = None,
        ltm_client: object | None = None,
    ) -> list[tuple[str, Callable[[], dict[str, int | float]]]]:
        """The compression pass as discrete safe steps.

        She runs one step per sleep tick, so the pass is spread
        through the stage like real consolidation — thousands of
        ripples, not one dump. Each step is self-committing:

        - **quantize** — one k-means iteration per step via
          :meth:`VQCodebook.fit_steps`; the codebook converges across
          the stage instead of blocking in a single fit.
        - **holographize+save** — migrates bridge edges into the
          holographic graph and persists it in the same step, so a
          mid-queue exit never leaves migrated edges unsaved.
        - **compact_ltm** — archives episodes one at a time; each
          archived episode is already durable.
        - **save** — VQ codebook + holographic graph, plus the
          periodic holographic rebuild.

        Steps not yet run when the stage ends simply wait — nothing
        half-committed needs rolling back.
        """
        steps: list[tuple[str, Callable[[], dict[str, int | float]]]] = []

        if embeddings is not None:
            steps.extend(self._quantize_steps(embeddings))

        def _holographize() -> dict[str, int | float]:
            stats = self._phase_holographize(network)
            # Save in the same step: migrated bridge edges are gone
            # from the explicit network the moment _phase_holographize
            # returns — persisting here keeps a mid-queue exit lossless.
            self._save()
            return stats

        steps.append(("holographize", _holographize))

        if ltm_client is not None:
            steps.append(
                ("compact_ltm", lambda: self._phase_compact_ltm(network, ltm_client))
            )

        def _save_step() -> dict[str, int | float]:
            hg_stats = self._maybe_rebuild_holographic_graph(network)
            self._save()
            hg_stats["saved"] = 1
            return hg_stats

        steps.append(("save", _save_step))
        return steps

    def _quantize_steps(
        self,
        embeddings: EmbeddingStore,
    ) -> list[tuple[str, Callable[[], dict[str, int | float]]]]:
        """VQ training as one-Lloyd-iteration steps + a merge scan."""
        matrix, names = embeddings.get_concept_matrix()
        if matrix is None or len(names) == 0:
            return [("quantize", lambda: {"vq_trained": 0, "merge_candidates": 0})]

        gen = self.vq_codebook.fit_steps(
            matrix, names, iters=self.vq_iters, seed=42
        )
        holder: dict[str, object] = {"stats": None}

        def _pump() -> dict[str, int | float]:
            if holder["stats"] is not None:
                return {"vq_done": 1}
            try:
                return {"vq_iter": float(next(gen)["iter"])}
            except StopIteration as done:
                raw = dict(done.value or {})
                stats: dict[str, int | float] = {
                    "vq_trained": 1,
                    "vq_reconstruction_error": raw.get("final_error", 0.0),
                    "vq_relative_reconstruction_error": raw.get("relative_error", 0.0),
                    "vq_reconstruction_cosine": raw.get("cosine", 0.0),
                    "vq_residual_saturation": raw.get("saturation", 0.0),
                    "vq_compression_ratio": self.vq_codebook.compression_ratio(),
                }
                holder["stats"] = stats
                return stats

        steps: list[tuple[str, Callable[[], dict[str, int | float]]]] = [
            ("quantize", _pump) for _ in range(self.vq_iters)
        ]

        def _merge_count() -> dict[str, int | float]:
            # Ensure training finished even if convergence yielded
            # fewer steps than `iters` — pump any remainder.
            while holder["stats"] is None:
                _pump()
            n = self.vq_codebook.count_merge_candidates(
                threshold=self.merge_threshold
            )
            if n:
                logger.info(
                    "VQ found %d merge candidates (threshold=%.3f)",
                    n,
                    self.merge_threshold,
                )
            return {"merge_candidates": n}

        steps.append(("merge_candidates", _merge_count))
        return steps

    def _maybe_rebuild_holographic_graph(self, network: ConceptNetwork) -> dict[str, float]:
        """Periodically rebuild the holographic graph from its journal
        to clear accumulated additive-superposition noise.

        The holographic graph uses circular convolution binding, and
        repeated ``add`` calls accumulate interference — duplicates
        reinforce in memory and journal evictions leave ghost bindings.
        Rebuilding from the journal every ~10 compression passes
        re-encodes the clean, deduplicated association set and restores
        SNR. The journal is the rebuild source because holographic
        memory cannot enumerate its contents, and materialized bridge
        edges in ``network.edges`` are nearly always absent — the old
        scan rebuilt from an empty list every time, so the memory only
        ever grew. ``network`` is retained for signature compatibility.
        """
        stats: dict[str, float] = {}
        self._compress_cycle = getattr(self, "_compress_cycle", 0) + 1
        journal_size = self.holographic_graph.distinct_edges
        # Saturation trigger: raw adds beyond ~2× the journal's capacity
        # are ghost/duplicate bindings — LTM compaction alone can add
        # ~57K pairs a pass, so waiting 10 cycles lets the memory
        # drown its journaled signal long before the cadence fires.
        saturated = (
            journal_size > 0
            and self.holographic_graph.edge_count > self.holographic_graph.journal_cap * 2
        )
        if (self._compress_cycle % 10 == 0 or saturated) and self.holographic_graph.edge_count > 0:
            try:
                # Do NOT rebuild from an empty journal — the guard
                # checks the holographic graph's *pre-rebuild* edge
                # count, and a journalless legacy load would pass it
                # and then clear() would destroy the store.
                rebuilt = self.holographic_graph.rebuild_from_journal()
                if rebuilt == 0:
                    logger.debug(
                        "Skipping holographic rebuild at cycle %d: "
                        "journal is empty (pre-journal state or nothing "
                        "bound yet). Graph left intact.",
                        self._compress_cycle,
                    )
                    return stats
                stats["hg_rebuilt"] = 1.0
                stats["hg_distinct_edges"] = float(rebuilt)
                stats["hg_snr_estimate"] = self.holographic_graph.snr_estimate
                logger.info(
                    "Rebuilt holographic graph from %d journaled edges "
                    "(cycle %d, snr ~%.2f)",
                    rebuilt,
                    self._compress_cycle,
                    self.holographic_graph.snr_estimate,
                )
            except Exception as e:  # noqa: BLE001
                note_swallowed(
                    "temporal_lobe.compression._maybe_rebuild_holographic_graph",
                    e,
                )
        return stats

    # ─── Phase 2: Holographize ──────────────────────────────────

    def _phase_holographize(
        self,
        network: ConceptNetwork,
    ) -> dict[str, int | float]:
        """Migrate bridge edges to holographic graph."""
        from temporal_lobe.concepts import _BRIDGE_ORIGINS

        bridge_edges: list[tuple[str, str, str, float]] = []
        kept_edges: list = []

        for edge in network.edges:
            if edge.origin in _BRIDGE_ORIGINS:
                rel = edge.relation.value if hasattr(edge.relation, "value") else str(edge.relation)
                bridge_edges.append(
                    (
                        edge.source,
                        rel,
                        edge.target,
                        edge.weight,
                    )
                )
            else:
                kept_edges.append(edge)

        # Add bridge edges to holographic graph
        self.holographic_graph.add_edges(bridge_edges)

        # Remove bridge edges from the explicit graph
        network.replace_edges(kept_edges)

        logger.info(
            "Holographize: migrated %d bridge edges, %d explicit edges remain",
            len(bridge_edges),
            len(kept_edges),
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

        Dream-insight episodes (``_DREAM_MODULE``) are neither retained
        nor compacted — they bypass this phase untouched. See the
        constant for why.

        Paginates through all LTM episodes — the daemon caps each page
        at 1000 regardless of the requested limit, so we page through
        with increasing offsets until a page comes back short. Without
        pagination, only the first 1000 episodes would ever be compacted
        and the LTM would grow unboundedly past that point.
        """
        stats: dict[str, int | float] = {
            "ltm_replayed": 0,
            "ltm_archived": 0,
            "ltm_archive_failed": 0,
            "ltm_retained": 0,
            "ltm_associations_extracted": 0,
            "ltm_total_seen": 0,
            "ltm_dreams_preserved": 0,
        }

        # A dead control channel cannot archive. search_episodes rides
        # the read channel and may still succeed, so without this check
        # the pass commits extracted associations while every archive
        # call raises ConnectionError — a half-committed sweep reported
        # as success. Skip cleanly instead; the next pass retries.
        if not getattr(ltm_client, "is_connected", True):
            logger.warning("LTM compaction skipped: client not connected")
            stats["ltm_skipped"] = 1
            return stats

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

        # Dream insights are preserved, not compacted — see
        # _DREAM_MODULE. Excluded before salience ranking so they
        # neither consume retention slots nor reach the archive step.
        episodes = [e for e in episodes if getattr(e, "source_module", None) != _DREAM_MODULE]
        stats["ltm_dreams_preserved"] = stats["ltm_total_seen"] - len(episodes)

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
            e for e in episodes_sorted if getattr(e, "salience", 0.0) >= self.salience_threshold
        ][: self.max_ltm_episodes]
        stats["ltm_retained"] = len(retained)
        retained_ids = {id(e) for e in retained}

        # Compact everything not retained — low-salience episodes and
        # high-salience overflow alike.
        to_compact = [e for e in episodes_sorted if id(e) not in retained_ids]
        associations_extracted = self._compact_episodes(to_compact, network, ltm_client, stats)
        stats["ltm_associations_extracted"] = associations_extracted

        logger.info(
            "LTM compaction: seen %d, replayed %d, archived %d, "
            "archive_failed %d, retained %d, dreams preserved %d, "
            "extracted %d associations",
            stats["ltm_total_seen"],
            stats["ltm_replayed"],
            stats["ltm_archived"],
            stats["ltm_archive_failed"],
            stats["ltm_retained"],
            stats["ltm_dreams_preserved"],
            associations_extracted,
        )

        return stats

    def _compact_episodes(
        self,
        to_compact: list,
        network: ConceptNetwork,
        ltm_client: object,
        stats: dict[str, int | float],
    ) -> int:
        """Replay each non-retained episode and return the number of
        associations extracted into the concept network."""
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

        associations_extracted = 0
        for episode in to_compact:
            associations_extracted += self._compact_episode(
                episode,
                concept_pattern,
                ltm_client,
                stats,
            )
        return associations_extracted

    def _paginate_ltm_episodes(
        self,
        ltm_client: object,
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
                    limit=PAGE_SIZE,
                    offset=offset,
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
                len(episodes),
                e,
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
            mentioned = mentioned[: self.max_concepts_per_episode]

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
            if (
                episode_id is not None and ltm_client.archive_episode(episode_id)  # type: ignore[attr-defined]
            ):
                stats["ltm_archived"] += 1
        except AttributeError as e:
            note_swallowed(
                "temporal_lobe.compression._compact_episode",
                e,
            )
        except OSError as e:
            stats["ltm_archive_failed"] += 1
            note_swallowed(
                "temporal_lobe.compression._compact_episode",
                e,
            )

        stats["ltm_replayed"] += 1
        return associations

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
