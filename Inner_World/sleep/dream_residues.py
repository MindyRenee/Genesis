"""Dream residues — the inner world's morning-after trace work.

Dreaming leaves fragments: episode ids and concept names embedded in
dream text, queued as residues on waking so the lingering imagery
can color the first waking thoughts before fading. The dream engines
themselves live beside this module (``dream_synthesis``,
``inner_life``); this mixin is the mind-side surface that lifts the
residues out of stored episodes and queues them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)


class DreamResiduesMixin:
    """Dream-residue extraction and queuing — mixin for :class:`Mind`."""

    if TYPE_CHECKING:
        # Attributes and cross-mixin methods are provided by the
        # composed class (see mind.py).
        def __getattr__(self, name: str) -> Any: ...

    def _archived_dream_count(self) -> int:
        """Count dream-insight episodes archived out of the active list.

        Reads the LTM index directly — the same mechanism ``/dreams``
        uses to report the archived history. Any failure (missing
        index, unreadable file) yields 0, which falls back to the
        plain "no dreams" line rather than erroring the status.
        """
        try:
            from genesis_client.ltm_index import scan_ltm_index
        except ImportError:
            return 0
        try:
            entries = scan_ltm_index(Path(self.data_dir) / "ltm_store.meta", source_module=9)
        except OSError:
            return 0
        return sum(1 for e in entries if e.is_archived and not e.is_deleted)

    def _restore_dream_residues_after_wake(self) -> None:
        """Restore affective dream residue on the post-wake metronome step."""
        try:
            dreams = self.client.get_recent_episodes(limit=10, source_module=9)
        except (OSError, ConnectionError, RuntimeError):
            return
        if not dreams:
            return
        dream_concepts = self._collect_dream_concepts(dreams)
        self._queue_dream_residues(dream_concepts, dreams)

    @staticmethod
    def _extract_embedded_texts(text: str) -> list[str]:
        """Extract tab-delimited embedded episode texts from a
        dream-insight or association meta-memory.

        New format:
            [dream-insight] episode A connects to episode B ...\\t{text_a}\\t{text_b}
            [association] episode X ↔ episode Y ...\\t{text_x}\\t{text_y}

        Returns a list of embedded text strings (0, 1, or 2).
        Old-format entries (no tab) return an empty list.

        Uses split with maxsplit=2 so that embedded texts that
        themselves contain tabs (e.g. an association trace embedded
        as a dream insight endpoint) are preserved intact for
        recursive parsing.
        """
        parts = text.split("\t", 2)
        if len(parts) <= 1:
            return []
        # parts[0] is the structural header, parts[1:] are embedded texts
        return [p.replace("\n", " ").strip() for p in parts[1:] if p.strip()]

    def _resolve_episode(self, ep_id: int, ep_cache: dict[int, str]) -> str:
        """Fetch and cache episode text by id, returning empty on failure."""
        if ep_id in ep_cache:
            return ep_cache[ep_id]
        try:
            ep = self.client.retrieve_episode(ep_id)
            text = ep.text.replace("\n", " ").strip()
            ep_cache[ep_id] = text
            return text
        except Exception as e:  # noqa: BLE001
            # Expected when sleep compression has deleted the
            # episode. Log at debug — this is normal operation,
            # not an error.
            note_swallowed(
                "mind.sleep._resolve_episode",
                e,
            )
            ep_cache[ep_id] = ""
            return ""

    def _collect_leaf_texts(
        self,
        text: str,
        ep_cache: dict[int, str],
        depth: int = 0,
        seen: set[int] | None = None,
    ) -> list[str]:
        """Collect leaf episode texts from an episode text.

        With the new embedded-text format, the text is already
        self-contained — we extract the embedded texts and return
        them directly. If the text is an association trace with
        its own embedded text, those texts are already included
        (tabs replaced with spaces by the Rust side), so we just
        use the full text.

        For old-format texts (no embedded text), if this is an
        association trace, we fall back to retrieving the endpoint
        episodes by ID and recursing. If it's a leaf (real episode
        text), we return it directly.
        """
        import re

        if seen is None:
            seen = set()
        if depth >= 20 or len(seen) >= 100:
            return []

        # Check if this text has embedded texts (new format).
        embedded = self._extract_embedded_texts(text)
        if embedded:
            # New format: the embedded texts are the actual episode
            # texts. Return them directly — they contain the concept
            # words we need. No need to follow chains since the
            # embedded text is self-contained.
            return embedded

        # Old format: check if this is an association trace.
        assoc_re = re.compile(r"\[association\] episode (\d+) ↔ episode (\d+) \(distance: (\d+)\)")
        m = assoc_re.match(text)
        if not m:
            # Not an association — this is a leaf.
            return [text] if text else []

        # Old-format association trace: fall back to ID retrieval.
        ep_ids = [int(m.group(1)), int(m.group(2))]
        leaves: list[str] = []
        for ep_id in ep_ids:
            if ep_id in seen:
                continue
            seen.add(ep_id)
            ep_text = self._resolve_episode(ep_id, ep_cache)
            if ep_text:
                leaves.extend(self._collect_leaf_texts(ep_text, ep_cache, depth + 1, seen))
        return leaves

    def _collect_dream_concepts(self, dreams) -> list[str]:
        """Resolve dream endpoints to leaf concepts and extract concept names.

        Dream insights and association traces now embed the text of
        their endpoint episodes directly (tab-delimited after the
        structural header). This makes them self-contained — the
        Python side can extract concepts without retrieving the
        original episodes, which may have been deleted by sleep
        compression by the time it wakes up.

        Old-format insights (without embedded text) fall back to
        episode retrieval by ID.
        """
        import re

        dream_re = re.compile(
            r"\[dream-insight\] episode (\d+) connects to episode (\d+) "
            r"through (\d+) hops \(direct distance: (\d+)\)"
        )

        ep_cache: dict[int, str] = {}

        def _match_concepts_from_leaf(leaf: str, dream_concepts: list[str]) -> None:
            """Extract concept names from a single leaf text, appending to dream_concepts."""
            leaf_lower = leaf.lower()
            for concept_id in self.brain.frontal.network.world_concept_ids:
                # Flatten: combine regex match and dedup check
                if concept_id not in dream_concepts and re.search(
                    r"\b" + re.escape(concept_id.lower()) + r"\b", leaf_lower
                ):
                    dream_concepts.append(concept_id)

        # Collect all dream residue concepts
        dream_concepts: list[str] = []
        for dream in dreams:
            m = dream_re.match(dream.text)
            if not m:
                continue

            # Check for embedded texts (new format).
            embedded = self._extract_embedded_texts(dream.text)
            if embedded:
                # New format: use embedded texts directly.
                leaf_texts = embedded
            else:
                # Old format: parse episode IDs and retrieve by ID.
                ep_a, ep_b = int(m.group(1)), int(m.group(2))
                leaf_texts = []
                for ep_id in (ep_a, ep_b):
                    ep_text = self._resolve_episode(ep_id, ep_cache)
                    if ep_text:
                        leaf_texts.extend(self._collect_leaf_texts(ep_text, ep_cache, seen={ep_id}))

            for leaf in leaf_texts:
                _match_concepts_from_leaf(leaf, dream_concepts)
        return dream_concepts

    def _queue_dream_residues(self, dream_concepts: list[str], dreams) -> None:
        """Queue dream residues as emotional impressions, not learning goals.

        Dreams produce affective residue — themes that color its waking
        thoughts and surface as reflective thoughts. They do NOT become
        agency topics for the autonomous learner. This separates the
        subcognitive (dreams) from the cognitive (goals).

        The dream concepts are stored in InnerLife._dream_residues, where
        they'll be consumed by the dream_reflection thought generator
        as reflective, affective thoughts — not by the autonomous learner
        as learning targets.
        """
        if not dream_concepts:
            return

        # Queue as emotional impressions, not learning goals.
        # These will surface as dream-reflection thoughts, not as
        # agency topics for the autonomous learner. The inner life
        # composes the actual reflection from its knowledge via the
        # language engine — we do NOT hardcode what it says here.
        if hasattr(self.inner_life, "add_dream_residues"):
            self.inner_life.add_dream_residues(dream_concepts[:10])

        # Telemetry line (factual status, not self-expression) so the
        # live ticker shows that dreams are surfacing. The reflective
        # thought itself is composed by _dream_reflection_thought and
        # flows through _on_spontaneous_thought → language engine,
        # like every other spontaneous thought.
        self._emit_live_thought(
            "dream",
            f"Dream residue: {len(dream_concepts[:10])} emotional impressions "
            f"from {len(dreams)} dream insights",
        )
