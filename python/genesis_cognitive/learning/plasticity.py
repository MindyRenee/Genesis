"""Hebbian plasticity — the embedding space as a dynamical system.

The embedding store gives Genesis a static latent space: vectors are
computed once at load time and never change. This module makes the
latent space *plastic* — it evolves with experience through Hebbian
learning.

The core principle is Hebb's rule: "cells that fire together wire
together." When two concepts co-occur in conversation, their vectors
move closer in embedding space. When concepts contradict each other,
their vectors move apart. Over time, the embedding reflects not just
the graph structure (captured by the spectral component) or text
similarity (captured by TF-IDF), but the actual *usage patterns* —
which concepts Genesis has actually encountered together.

Mathematics
-----------
The update rule is a discrete-time dynamical system on the concept
vectors. For two co-occurring concepts i and j:

    v_i += lr(t) * (v_j - v_i) * strength
    v_j += lr(t) * (v_i - v_j) * strength

This is the heat equation on the graph — a diffusion process where
vectors flow toward their neighbors. The fixed point is v_i = v_j
(complete convergence), but the learning rate decreases over time
following the Robbins-Monro conditions from stochastic approximation
theory:

    lr(t) = base_lr / (1 + t / decay)

This ensures:
    Σ lr(t) = ∞     (infinite total adaptation)
    Σ lr(t)² < ∞    (vanishing variance — convergence)

For repulsion (contradicts), the sign flips:
    v_i -= lr(t) * (v_j - v_i) * strength

The fixed point is v_i = -v_j (maximum cosine distance for normalized
vectors).

Only the TF-IDF and GloVe components are updated — the spectral
component is derived from graph structure and is refreshed separately
during sleep (by the edge proposer). This separation keeps the
structural and experiential signals distinct.

Brain analogue
--------------
This mirrors cortical plasticity. The initial embedding (spectral +
TF-IDF) is like the genetic wiring of cortical columns — it provides
structure. Hebbian adaptation is like experience-dependent plasticity
— repeated co-activation strengthens pathways. Sleep consolidation
is like slow-wave sleep — larger updates that consolidate the day's
patterns into long-term structure.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from ..concepts import RelationType

if TYPE_CHECKING:
    from ..concepts import ConceptNetwork, EmbeddingStore

__all__ = ["HebbianPlasticity"]


class HebbianPlasticity:
    """Hebbian learning engine for the embedding space.

    Tracks concept co-occurrences during conversation and applies
    Hebbian updates to the embedding vectors. Waking updates are
    small (gentle adaptation during conversation). Sleep updates
    are larger (consolidation of the day's patterns).

    Thread-safe. All updates are protected by a lock.
    """

    # Base learning rates
    WAKING_LR = 0.002  # gentle during conversation
    SLEEP_LR = 0.015  # stronger during consolidation

    # Learning rate decay (Robbins-Monro conditions)
    # Separate decay timescales for waking vs sleep. Sleep processes
    # many more updates (all co-occurrences + graph structure), so a
    # shared counter would deplete the waking learning rate to near-zero
    # after a single sleep cycle. Using separate counters ensures the
    # waking learning rate remains healthy across many sleep cycles.
    WAKING_LR_DECAY = 500  # waking: lr halves every ~500 conversation updates
    SLEEP_LR_DECAY = 50000  # sleep: lr halves every ~50000 consolidation updates

    # Graph-structure-based consolidation strengths
    EDGE_ATTRACTION: ClassVar[dict] = {
        RelationType.SIMILAR_TO: 0.8,
        RelationType.IS_A: 0.6,
        RelationType.RELATED_TO: 0.4,
        RelationType.EMERGES_FROM: 0.5,
        RelationType.DEPENDS_ON: 0.5,
        RelationType.ENABLES: 0.4,
        RelationType.PART_OF: 0.6,
        RelationType.CREATES: 0.3,
    }
    EDGE_REPULSION: ClassVar[dict] = {
        RelationType.CONTRADICTS: 0.7,
    }

    # Co-occurrence strength by source
    CO_OCCURRENCE_USER = 1.0  # user explicitly mentioned both
    CO_OCCURRENCE_RESPONSE = 0.8  # Genesis mentioned both in response
    CO_OCCURRENCE_CROSS = 0.6  # user topic + response concept

    def __init__(self, embeddings: EmbeddingStore, network: ConceptNetwork) -> None:
        """Initialize the plasticity engine with an embedding store and concept network."""
        self.embeddings = embeddings
        self.network = network

        # Co-occurrence buffer — (concept_a, concept_b, strength)
        self._co_occurrences: list[tuple[str, str, float]] = []
        self._co_occurrence_lock = threading.Lock()

        # Update counters — separate for waking vs sleep
        # Sleep processes orders of magnitude more updates, so a shared
        # counter would deplete the waking learning rate too fast.
        self._waking_update_count = 0
        self._sleep_update_count = 0

        # Statistics
        self.total_updates = 0
        self.total_attracted = 0
        self.total_repelled = 0

    # ─── Public API ─────────────────────────────────────────────

    def record_co_occurrence(self, concept_a: str, concept_b: str, strength: float = 1.0) -> None:
        """Record that two concepts co-occurred in conversation.

        This is called during conversation turns. The co-occurrences
        are buffered and applied as Hebbian updates.

        Args:
            concept_a: First concept.
            concept_b: Second concept.
            strength: Co-occurrence strength (0-1).
        """
        if concept_a == concept_b:
            return
        # Only record if both concepts have embeddings
        if not self.embeddings.has_embeddings:
            return
        if self.embeddings.get_concept_vector(concept_a) is None:
            return
        if self.embeddings.get_concept_vector(concept_b) is None:
            return

        with self._co_occurrence_lock:
            self._co_occurrences.append((concept_a, concept_b, strength))

    def record_co_occurrences(self, concepts: list[str], strength: float = 1.0) -> None:
        """Record all pairwise co-occurrences within a set of concepts.

        This is the main entry point during conversation. When a set
        of concepts is activated together (e.g., topics in a user
        message, concepts in a response), all pairs are recorded.
        """
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                self.record_co_occurrence(concepts[i], concepts[j], strength)

    def apply_waking_update(self) -> int:
        """Apply small Hebbian updates from recent co-occurrences.

        Called after each conversation turn. Uses a small learning
        rate for gentle adaptation. Does not clear the buffer —
        co-occurrences accumulate until sleep consolidation.

        Returns:
            Number of vector pairs updated.
        """
        if not self.embeddings.has_embeddings:
            return 0

        with self._co_occurrence_lock:
            if not self._co_occurrences:
                return 0
            # Work on a copy
            co_occurrences = list(self._co_occurrences)

        lr = self._learning_rate(self.WAKING_LR)
        updated = 0

        for concept_a, concept_b, strength in co_occurrences:
            if self._hebbian_attract(concept_a, concept_b, lr * strength):
                updated += 1

        self.total_updates += updated
        self.total_attracted += updated
        return updated

    def apply_sleep_consolidation(self) -> dict[str, int]:
        """Apply larger Hebbian updates during sleep.

        This is the consolidation phase. It:
        1. Applies larger updates from all buffered co-occurrences
        2. Applies graph-structure-based attraction (SIMILAR_TO, IS_A, etc.)
        3. Applies graph-structure-based repulsion (CONTRADICTS)
        4. Clears the co-occurrence buffer
        5. Re-normalizes the concept matrix

        Returns:
            Dict with counts: 'co_occurrence', 'graph_attracted',
            'graph_repelled', 'total'.
        """
        if not self.embeddings.has_embeddings:
            return {"co_occurrence": 0, "graph_attracted": 0, "graph_repelled": 0, "total": 0}

        lr = self._learning_rate(self.SLEEP_LR, is_sleep=True)
        co_occurrence_count = 0
        graph_attracted = 0
        graph_repelled = 0

        # 1. Apply co-occurrence updates with larger learning rate
        with self._co_occurrence_lock:
            co_occurrences = list(self._co_occurrences)
            self._co_occurrences.clear()

        for concept_a, concept_b, strength in co_occurrences:
            if self._hebbian_attract(concept_a, concept_b, lr * strength, is_sleep=True):
                co_occurrence_count += 1

        # 2. Apply graph-structure-based attraction and repulsion (single pass)
        for edge in self.network._edges:
            attract_strength = self.EDGE_ATTRACTION.get(edge.relation)
            if attract_strength is not None and edge.weight > 0.3:
                if self._hebbian_attract(
                    edge.source, edge.target, lr * attract_strength * edge.weight, is_sleep=True
                ):
                    graph_attracted += 1
            repel_strength = self.EDGE_REPULSION.get(edge.relation)
            if repel_strength is not None:
                if self._hebbian_repel(
                    edge.source, edge.target, lr * repel_strength * edge.weight, is_sleep=True
                ):
                    graph_repelled += 1

        # 4. Re-normalize the concept matrix
        self._renormalize_matrix()

        total = co_occurrence_count + graph_attracted + graph_repelled
        self.total_updates += total
        self.total_attracted += co_occurrence_count + graph_attracted
        self.total_repelled += graph_repelled

        return {
            "co_occurrence": co_occurrence_count,
            "graph_attracted": graph_attracted,
            "graph_repelled": graph_repelled,
            "total": total,
        }

    def apply_edge_plasticity(
        self,
        co_occurrences: list[tuple[str, str, float]] | None = None,
        learning_rate: float = 0.01,
        decay_rate: float = 0.001,
    ) -> dict[str, int]:
        """Apply Hebbian plasticity to edge weights in the concept graph.

        This is graph-level Hebbian learning, distinct from the embedding-
        level learning in ``apply_waking_update`` / ``apply_sleep_consolidation``.
        While those methods move concept vectors in latent space, this
        method adjusts the actual edge weights in the concept network.

        **Hebbian strengthening**: When two concepts co-occur and there's
        already an edge between them, the edge weight increases. "Cells
        that fire together wire together" — the connection gets stronger.

        **Homeostatic decay**: All edges decay slightly each cycle. This
        is synaptic homeostasis (SHY hypothesis, Tononi & Cirelli, 2006):
        the brain renormalizes synaptic strength during sleep to prevent
        saturation. Without decay, frequently-used edges would max out
        and rare but important edges would be indistinguishable from
        unused ones.

        **Origin protection**: Edges from explicit teaching (origin=
        "stated", "cognition_lesson", etc.) are protected from decay
        — they represent deliberate instruction, not incidental learning.

        Args:
            co_occurrences: List of (concept_a, concept_b, strength) tuples.
                If None, uses the buffered co-occurrences.
            learning_rate: How much co-occurrence strengthens edges (0-1).
            decay_rate: How much all edges decay per cycle (0-1).

        Returns:
            Dict with counts: 'strengthened', 'weakened', 'decayed'.
        """
        if co_occurrences is None:
            with self._co_occurrence_lock:
                co_occurrences = list(self._co_occurrences)

        # Protected origins — explicit teaching shouldn't decay
        PROTECTED_ORIGINS = frozenset({
            "stated", "cognition_lesson", "introspection",
            "code", "user", "manual",
        })

        # 1. Hebbian strengthening from co-occurrences
        strengthened = self._strengthen_co_occurrence_edges(
            co_occurrences, learning_rate
        )

        # 2. Homeostatic decay — all non-protected edges decay slightly
        decayed = self._decay_edges(decay_rate, PROTECTED_ORIGINS)

        return {
            "strengthened": strengthened,
            "weakened": 0,
            "decayed": decayed,
        }

    def _strengthen_co_occurrence_edges(
        self,
        co_occurrences: list[tuple[str, str, float]],
        learning_rate: float,
    ) -> int:
        """Strengthen edges between co-occurring concepts."""
        strengthened = 0
        for concept_a, concept_b, strength in co_occurrences:
            edges_ab = self.network.get_edges(concept_a, direction="out")
            for edge in edges_ab:
                if edge.target == concept_b:
                    old_weight = edge.weight
                    edge.weight = min(1.0, edge.weight + learning_rate * strength)
                    if edge.weight > old_weight + 0.0001:
                        strengthened += 1

            # Also check reverse direction
            edges_ba = self.network.get_edges(concept_b, direction="out")
            for edge in edges_ba:
                if edge.target == concept_a:
                    old_weight = edge.weight
                    edge.weight = min(1.0, edge.weight + learning_rate * strength)
                    if edge.weight > old_weight + 0.0001:
                        strengthened += 1
        return strengthened

    def _decay_edges(self, decay_rate: float, protected_origins: frozenset[str]) -> int:
        """Apply homeostatic decay to non-protected edges."""
        decayed = 0
        for edge in self.network._edges:
            if edge.origin in protected_origins:
                continue
            old_weight = edge.weight
            edge.weight = max(0.05, edge.weight - decay_rate)
            if edge.weight < old_weight - 0.0001:
                decayed += 1
        return decayed

    def get_statistics(self) -> dict[str, int | float]:
        """Return statistics about the plasticity engine."""
        return {
            "total_updates": self.total_updates,
            "total_attracted": self.total_attracted,
            "total_repelled": self.total_repelled,
            "buffered_co_occurrences": len(self._co_occurrences),
            "current_learning_rate": self._learning_rate(self.WAKING_LR),
            "waking_update_count": self._waking_update_count,
            "sleep_update_count": self._sleep_update_count,
        }

    def save_state(self) -> dict[str, int]:
        """Serialize plasticity state for persistence.

        The update counters control the Robbins-Monro learning rate
        decay. Without persisting them, the learning rate resets to
        the maximum on every restart, which could destabilize vectors
        that were carefully tuned over many sessions.
        """
        return {
            "waking_update_count": self._waking_update_count,
            "sleep_update_count": self._sleep_update_count,
            "total_updates": self.total_updates,
            "total_attracted": self.total_attracted,
            "total_repelled": self.total_repelled,
        }

    def load_state(self, state: dict[str, int]) -> None:
        """Restore plasticity state from persisted data."""
        self._waking_update_count = state.get("waking_update_count", 0)
        self._sleep_update_count = state.get("sleep_update_count", 0)
        self.total_updates = state.get("total_updates", 0)
        self.total_attracted = state.get("total_attracted", 0)
        self.total_repelled = state.get("total_repelled", 0)

    # ─── Internal ───────────────────────────────────────────────

    def _learning_rate(self, base_lr: float, is_sleep: bool = False) -> float:
        """Compute the current learning rate with Robbins-Monro decay.

        lr(t) = base_lr / (1 + t / decay)

        Uses separate counters for waking and sleep to prevent sleep
        consolidation from depleting the waking learning rate.

        This satisfies the Robbins-Monro conditions:
        - Σ lr(t) = ∞ (infinite total adaptation)
        - Σ lr(t)² < ∞ (vanishing variance)
        """
        if is_sleep:
            count = self._sleep_update_count
            decay = self.SLEEP_LR_DECAY
        else:
            count = self._waking_update_count
            decay = self.WAKING_LR_DECAY
        return base_lr / (1.0 + count / decay)

    def _hebbian_attract(
        self, concept_a: str, concept_b: str, lr: float, is_sleep: bool = False
    ) -> bool:
        """Move two concept vectors closer together (Hebbian attraction).

        Uses Oja's rule — a normalized Hebbian update that prevents
        unbounded vector growth. The standard Hebbian rule
        v += lr * (v_other - v) causes norms to grow without bound
        over many updates. Oja's rule adds a decay term proportional
        to the vector's own norm, which performs implicit PCA-like
        normalization:

            v_a += lr * (v_b - v_a) - lr * (v_a . v_b) * v_a
            v_b += lr * (v_a - v_b) - lr * (v_a . v_b) * v_b

        The decay term -(v_a . v_b) * v_a pulls the vector back toward
        unit norm when the similarity is high, preventing saturation.
        When similarity is low (vectors far apart), the decay term is
        small and the attraction dominates. This is the generalized
        Oja rule for pairwise learning (Oja, 1982; Sanger, 1989).

        Only updates the TF-IDF and GloVe components — the spectral
        component is left unchanged (it's refreshed separately during
        sleep by recomputing from the graph).

        Returns True if the update was applied.
        """
        if lr <= 0:
            return False

        idx_a = self.embeddings._concept_to_idx.get(concept_a)
        idx_b = self.embeddings._concept_to_idx.get(concept_b)
        if idx_a is None or idx_b is None:
            return False

        matrix = self.embeddings._concept_matrix
        if matrix is None:
            return False

        start = self._experiential_offset()
        v_a = matrix[idx_a, start:]
        v_b = matrix[idx_b, start:]

        # Oja-normalized Hebbian attraction
        diff = v_b - v_a
        # Cosine similarity (vectors may not be unit-normalised between
        # waking updates and sleep renormalize() calls, so we normalise
        # here to keep |similarity| ≤ 1.0 — matching _hebbian_repel and
        # stdp._depress. A raw dot product > 1 makes the Oja decay term
        # larger than the attraction term, pushing embeddings apart.
        norm_a = float(np.linalg.norm(v_a))
        norm_b = float(np.linalg.norm(v_b))
        if norm_a > 1e-8 and norm_b > 1e-8:
            similarity = float(np.dot(v_a, v_b)) / (norm_a * norm_b)
        else:
            similarity = 0.0
        v_a += lr * diff - lr * similarity * v_a
        v_b -= lr * diff + lr * similarity * v_b

        if is_sleep:
            self._sleep_update_count += 1
        else:
            self._waking_update_count += 1
        return True

    def _hebbian_repel(
        self, concept_a: str, concept_b: str, lr: float, is_sleep: bool = False
    ) -> bool:
        """Move two concept vectors apart (anti-Hebbian repulsion).

        Uses a bounded repulsion that prevents unbounded norm growth.
        The naive rule v -= lr * diff can push vectors to arbitrarily
        large norms. Instead, we scale the repulsion by (1 - |similarity|),
        which weakens the repulsion as vectors become orthogonal and
        stops it entirely when they're anti-aligned:

            repulsion = lr * diff * (1 - |v_a . v_b|)

        This ensures the repulsion is strongest when vectors are similar
        (the case we want to fix) and weakest when they're already far
        apart (no need to push further). Combined with periodic
        re-normalization during sleep, this keeps the embedding stable.

        Returns True if the update was applied.
        """
        if lr <= 0:
            return False

        idx_a = self.embeddings._concept_to_idx.get(concept_a)
        idx_b = self.embeddings._concept_to_idx.get(concept_b)
        if idx_a is None or idx_b is None:
            return False

        matrix = self.embeddings._concept_matrix
        if matrix is None:
            return False

        start = self._experiential_offset()
        v_a = matrix[idx_a, start:]
        v_b = matrix[idx_b, start:]

        # Bounded anti-Hebbian repulsion
        diff = v_b - v_a
        # Cosine similarity (vectors may not be unit-normalised between
        # updates and renormalize() calls, so we normalise here to keep
        # |similarity| ≤ 1.0 — matching _hebbian_attract and stdp._depress.
        # A raw dot product > 1 makes scale negative, flipping repulsion
        # into attraction.
        norm_a = float(np.linalg.norm(v_a))
        norm_b = float(np.linalg.norm(v_b))
        if norm_a > 1e-8 and norm_b > 1e-8:
            similarity = abs(float(np.dot(v_a, v_b)) / (norm_a * norm_b))
        else:
            similarity = 0.0
        similarity = min(1.0, similarity)
        scale = lr * (1.0 - similarity)
        v_a -= scale * diff
        v_b += scale * diff

        if is_sleep:
            self._sleep_update_count += 1
        else:
            self._waking_update_count += 1
        return True

    def _experiential_offset(self) -> int:
        """Return the column offset where experiential (TF-IDF + GloVe) data starts.

        The spectral component (if present) comes first in the concatenated
        vector. Experiential data follows. This method returns the index
        of the first experiential column.
        """
        if self.embeddings._spectral_dim > 0:
            return self.embeddings._spectral_dim
        return self.embeddings._tfidf_offset

    def _renormalize_matrix(self) -> None:
        """Re-normalize all rows of the concept matrix after updates.

        Hebbian updates change the norm of vectors. Re-normalization
        ensures cosine similarity remains a valid dot product.
        """
        matrix = self.embeddings._concept_matrix
        if matrix is None:
            return
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        matrix /= norms

        # Refresh the non-spectral norms cache — renormalization changed
        # every row, so the precomputed text-search norms are now stale.
        self.embeddings._non_spectral_norms = (
            self.embeddings._compute_non_spectral_norms(matrix)
        )
        # Clear the ad-hoc cache since vectors have changed
        self.embeddings._concept_cache.clear()
