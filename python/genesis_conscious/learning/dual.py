"""Dual-system learning architecture — hippocampus and neocortex.

The brain has two complementary learning systems (McClelland et al.,
1995; O'Reilly & Norman, 2002):

1. **Hippocampus** — a *fast* learner. It rapidly encodes individual
   episodes with high plasticity, using *pattern separation* to
   orthogonalize similar inputs into distinct, non-overlapping
   representations. This prevents new memories from overwriting old
   ones (catastrophic interference). But hippocampal storage is
   capacity-limited and temporary.

2. **Neocortex** — a *slow* learner. It gradually extracts general
   structure across many experiences, using overlapping
   representations that capture shared regularities. Neocortical
   learning is robust and long-lasting but too slow to encode a
   single episode in one shot.

**Systems consolidation**: during sleep (especially slow-wave and
REM sleep), hippocampal memories are *replayed* and gradually
transferred to neocortical storage. The hippocampus teaches the
neocortex, episode by episode, until the neocortex has extracted
the general knowledge and the hippocampal trace can fade
(Diekelmann & Born, 2010; Frankland & Bontempi, 2005).

This complementary systems architecture solves the *stability-
plasticity dilemma*: the hippocampus can learn fast without
disrupting neocortical knowledge, and the neocortex can learn
slowly without forgetting the details that the hippocampus holds.

# Implementation

This module models the two systems as separate stores:

- The **fast (hippocampal) store** keeps recent episodes as
  pattern-separated sparse vectors. Pattern separation is achieved
  by adding orthogonalizing noise proportional to input similarity:
  similar inputs get pushed apart so they don't interfere.

- The **slow (neocortical) store** keeps general knowledge as
  overlapping dense vectors, updated gradually by averaging over
  consolidated hippocampal episodes.

Retrieval checks both systems and blends their outputs — the
hippocampus provides fast, specific recall; the neocortex provides
slow, generalized knowledge.

References:
    - McClelland et al. (1995): Why there are complementary learning
      systems in the hippocampus and neocortex. Neural Computation,
      7(2), 249-267.
    - O'Reilly & Norman (2002): Hippocampal conjunctive encoding,
      representation, and consolidation. Hippocampus, 12, 175-188.
    - Frankland & Bontempi (2005): The organization of recent and
      remote memories. Nature Reviews Neuroscience, 6, 119-130.
    - Diekelmann & Born (2010): The memory function of sleep.
      Nature Reviews Neuroscience, 11, 114-126.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork, EmbeddingStore

__all__ = ["DualSystemLearner", "HippocampalEpisode", "NeocorticalMemory"]


@dataclass(slots=True)
class HippocampalEpisode:
    """A single episode in the fast (hippocampal) store.

    Attributes:
        key: A human-readable key (e.g., the topic or concept set).
        pattern: The pattern-separated sparse vector representation.
        content: The raw experience content (text or concept list).
        timestamp: When the episode was encoded (ms).
        replay_count: How many times this episode has been replayed
            during consolidation. Episodes with high replay counts
            are closer to being fully transferred to neocortex.
        consolidated: Whether this episode has been transferred to
            the slow (neocortical) store.
    """

    key: str
    pattern: np.ndarray
    content: str
    timestamp: int
    replay_count: int = 0
    consolidated: bool = False


@dataclass(slots=True)
class NeocorticalMemory:
    """A generalized memory in the slow (neocortical) store.

    Attributes:
        key: The generalized key (often shared across episodes).
        pattern: The overlapping dense vector representation.
        strength: How strongly this memory is established (grows
            with consolidation).
        episode_count: How many hippocampal episodes contributed.
        last_updated: Timestamp of the last consolidation update.
    """

    key: str
    pattern: np.ndarray
    strength: float
    episode_count: int
    last_updated: int


class DualSystemLearner:
    """Dual-system learning: fast hippocampal + slow neocortical.

    Models the complementary learning systems of the brain. The
    hippocampus rapidly encodes episodes with pattern separation;
    the neocortex gradually extracts general knowledge through
    systems consolidation.

    Attributes:
        fast_lr: Hippocampal (fast) learning rate — high.
        slow_lr: Neocortical (slow) learning rate — low.
        consolidation_rate: How much each replay shifts a hippocampal
            episode toward neocortical storage.
        separation_gain: Strength of pattern separation (how much
            similar inputs are orthogonalized).
        fast_capacity: Maximum number of hippocampal episodes retained.
    """

    # Default parameters
    FAST_LR: float = 0.5  # hippocampal: rapid encoding
    SLOW_LR: float = 0.02  # neocortical: gradual
    CONSOLIDATION_RATE: float = 0.1  # per-replay transfer strength
    SEPARATION_GAIN: float = 0.3  # pattern separation strength
    FAST_CAPACITY: int = 200  # hippocampal episode limit
    PATTERN_DIM: int = 64  # dimensionality of internal representations

    def __init__(
        self,
        network: ConceptNetwork,
        embeddings: EmbeddingStore | None = None,
        *,
        fast_lr: float = FAST_LR,
        slow_lr: float = SLOW_LR,
        consolidation_rate: float = CONSOLIDATION_RATE,
        separation_gain: float = SEPARATION_GAIN,
        fast_capacity: int = FAST_CAPACITY,
        pattern_dim: int = PATTERN_DIM,
    ) -> None:
        """Initialize both learning stores with configurable rates and capacity."""
        self.network = network
        self.embeddings = embeddings

        self.fast_lr = fast_lr
        self.slow_lr = slow_lr
        self.consolidation_rate = consolidation_rate
        self.separation_gain = separation_gain
        self.fast_capacity = fast_capacity
        self.pattern_dim = pattern_dim

        # Fast (hippocampal) store
        self._fast_store: list[HippocampalEpisode] = []
        # Slow (neocortical) store, keyed by generalized key
        self._slow_store: dict[str, NeocorticalMemory] = {}

        self._lock = threading.Lock()

        # Statistics
        self.total_encoded = 0
        self.total_consolidated = 0

    # ─── Public API ─────────────────────────────────────────────

    def encode_fast(self, experience: str | list[str]) -> HippocampalEpisode:
        """Encode an experience into the fast (hippocampal) store.

        Pattern separation ensures that similar experiences get
        distinct hippocampal representations, preventing interference.
        The input is converted to a feature vector (from embeddings if
        available, otherwise a hash-based vector), then orthogonalizing
        noise is added proportional to similarity with existing
        episodes.

        Args:
            experience: A text description or list of concept names.

        Returns:
            The encoded HippocampalEpisode.
        """
        key = self._experience_key(experience)
        base_pattern = self._featurize(experience)

        with self._lock:
            # Pattern separation: push this pattern away from similar
            # existing patterns. The repulsion is proportional to
            # cosine similarity, so near-duplicates get maximally
            # separated while unrelated inputs are untouched.
            pattern = base_pattern.copy()
            for ep in self._fast_store:
                sim = self._cosine(pattern, ep.pattern)
                if sim > 0.3:
                    # Orthogonalizing push: move away from the similar
                    # pattern, scaled by similarity and separation gain.
                    diff = pattern - ep.pattern
                    n = np.linalg.norm(diff)
                    if n > 1e-8:
                        pattern = pattern + self.separation_gain * sim * (diff / n)

            # Normalize the separated pattern
            n = np.linalg.norm(pattern)
            if n > 1e-8:
                pattern = pattern / n

            episode = HippocampalEpisode(
                key=key,
                pattern=pattern.astype(np.float32),
                content=experience if isinstance(experience, str) else " ".join(experience),
                timestamp=int(time.time() * 1000),
            )
            self._fast_store.append(episode)

            # Enforce capacity (drop oldest, like hippocampal forgetting)
            if len(self._fast_store) > self.fast_capacity:
                self._fast_store.pop(0)

            self.total_encoded += 1

        return episode

    def consolidate_to_slow(self, max_episodes: int = 10) -> dict[str, int]:
        """Transfer hippocampal memories to neocortical storage.

        This is systems consolidation. Unconsolidated hippocampal
        episodes are replayed and gradually merged into the slow
        (neocortical) store. Each replay moves the neocortical
        representation toward the hippocampal pattern by a small
        amount (slow_lr), and increments the episode's replay count.
        Once an episode's accumulated transfer exceeds a threshold,
        it is marked consolidated.

        Args:
            max_episodes: Maximum number of episodes to replay per call.

        Returns:
            A dict with 'replayed', 'consolidated', 'neocortical_updated'.
        """
        replayed = 0
        consolidated = 0
        neocortical_updated = 0

        with self._lock:
            # Pick unconsolidated episodes (oldest first — they've had
            # the most time to be replayed during sleep).
            candidates = [ep for ep in self._fast_store if not ep.consolidated]
            candidates.sort(key=lambda e: e.timestamp)
            batch = candidates[:max_episodes]

            for ep in batch:
                ep.replay_count += 1
                replayed += 1

                # Update or create the neocortical memory
                existing = self._slow_store.get(ep.key)
                if existing is None:
                    self._slow_store[ep.key] = NeocorticalMemory(
                        key=ep.key,
                        pattern=ep.pattern.copy(),
                        strength=self.consolidation_rate,
                        episode_count=1,
                        last_updated=int(time.time() * 1000),
                    )
                    neocortical_updated += 1
                else:
                    # Gradual neocortical update: move toward the
                    # hippocampal pattern by slow_lr.
                    existing.pattern = (
                        existing.pattern * (1.0 - self.slow_lr) + ep.pattern * self.slow_lr
                    )
                    n = np.linalg.norm(existing.pattern)
                    if n > 1e-8:
                        existing.pattern = existing.pattern / n
                    existing.strength = min(1.0, existing.strength + self.consolidation_rate)
                    existing.episode_count += 1
                    existing.last_updated = int(time.time() * 1000)
                    neocortical_updated += 1

                # Mark consolidated once enough replays have occurred
                # and the neocortical trace is strong enough.
                neo = self._slow_store[ep.key]
                if ep.replay_count >= 3 and neo.strength >= 0.3:
                    ep.consolidated = True
                    consolidated += 1

            self.total_consolidated += consolidated

        return {
            "replayed": replayed,
            "consolidated": consolidated,
            "neocortical_updated": neocortical_updated,
        }

    def retrieve(self, query: str | list[str]) -> list[tuple[str, float, str]]:
        """Retrieve memories matching a query from both systems.

        Checks both the fast (hippocampal) and slow (neocortical)
        stores. The hippocampus provides specific, recent recall;
        the neocortex provides generalized knowledge. Results are
        blended and sorted by similarity.

        Args:
            query: A text query or list of concept names.

        Returns:
            A list of (key, similarity, system) tuples sorted by
            similarity descending. `system` is 'hippocampal' or
            'neocortical'.
        """
        query_pattern = self._featurize(query)
        results: list[tuple[str, float, str]] = []

        with self._lock:
            # Fast system: specific episodic recall
            for ep in self._fast_store:
                sim = self._cosine(query_pattern, ep.pattern)
                if sim > 0.1:
                    results.append((ep.key, sim, "hippocampal"))

            # Slow system: generalized knowledge
            for mem in self._slow_store.values():
                sim = self._cosine(query_pattern, mem.pattern)
                # Weight by neocortical strength (well-consolidated
                # memories are more reliable).
                sim *= 0.5 + 0.5 * mem.strength
                if sim > 0.1:
                    results.append((mem.key, sim, "neocortical"))

        results.sort(key=lambda x: -x[1])
        return results

    def get_statistics(self) -> dict[str, int | float]:
        """Return statistics about both learning systems."""
        with self._lock:
            return {
                "fast_episodes": len(self._fast_store),
                "fast_consolidated": sum(1 for e in self._fast_store if e.consolidated),
                "slow_memories": len(self._slow_store),
                "total_encoded": self.total_encoded,
                "total_consolidated": self.total_consolidated,
                "avg_replay_count": (
                    sum(e.replay_count for e in self._fast_store) / len(self._fast_store)
                    if self._fast_store
                    else 0.0
                ),
            }

    # ─── Internal ───────────────────────────────────────────────

    def _experience_key(self, experience: str | list[str]) -> str:
        """Derive a generalized key from an experience."""
        if isinstance(experience, list):
            return " ".join(sorted(experience)).lower()
        return experience.lower().strip()

    def _featurize(self, experience: str | list[str]) -> np.ndarray:
        """Convert an experience into a feature vector.

        Uses concept embeddings when available (semantic features),
        falling back to a deterministic hash-based vector otherwise.
        """
        if isinstance(experience, list):
            concepts = experience
        else:
            # Extract concept names from text via the network
            concepts = self.network.extract_from_text(experience)

        pattern = np.zeros(self.pattern_dim, dtype=np.float32)

        if concepts and self.embeddings is not None and self.embeddings.has_embeddings:
            # Average the concept vectors, projected to pattern_dim
            vecs = []
            for c in concepts:
                v = self.embeddings.get_concept_vector(c)
                if v is not None:
                    vecs.append(v)
            if vecs:
                avg = np.mean(vecs, axis=0)
                # Project to pattern_dim (truncate or pad)
                if len(avg) >= self.pattern_dim:
                    pattern = avg[: self.pattern_dim]
                else:
                    pattern[: len(avg)] = avg
        else:
            # Hash-based fallback: deterministic bag-of-words features
            text = self._experience_key(experience)
            for i, ch in enumerate(text):
                idx = (ord(ch) * (i + 1)) % self.pattern_dim
                pattern[idx] += 1.0

        n = np.linalg.norm(pattern)
        if n > 1e-8:
            pattern = pattern / n
        return pattern.astype(np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
