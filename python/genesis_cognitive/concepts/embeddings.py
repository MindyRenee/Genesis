"""Embedding store — Genesis's latent space.

This is the sub-symbolic layer beneath the concept network. It provides
a continuous vector space where concepts with similar meanings are close
together. This enables three things the symbolic graph cannot do alone:

1. **Pattern recognition** — "the tall plant in my yard" activates
   "tree" even though no word matches exactly. The input text shares
   semantic features with the tree concept.

2. **Generalization** — she knows "joy" and encounters "elation."
   They share many neighbors in the concept network, so their vectors
   are close. She can reason about elation by analogy to joy, even
   with no explicit graph edge.

3. **Discovery** — during sleep, she scans the latent space for nearby
   concepts that lack a graph edge and proposes new SIMILAR_TO
   relationships. This is where generalization becomes knowledge.

Architecture
------------
The embedding store uses two complementary representations:

1. **Spectral embedding** of the concept graph — concepts that share
   neighbors are close in this space. Computed via truncated SVD of
   the graph Laplacian. This captures structural similarity (the shape
   of the knowledge graph).

2. **TF-IDF text embedding** — concepts are represented by bag-of-words
   vectors over their definitions and names, weighted by inverse
   document frequency. This captures lexical similarity (words that
   appear in similar definitions).

The final concept vector is the concatenation of both, L2-normalized.
This dual representation mirrors the brain's dual coding: structural
(hippocampal attractor basins from co-activation) and lexical
(cortical semantic networks from language).

If pre-trained GloVe embeddings are available (downloaded via
setup_embeddings.py), they are used as an additional component,
providing distributional semantics from general English. This
supplements but does not replace the self-grounded representations.

All vectors are L2-normalized at load time, so cosine similarity
reduces to a single dot product — O(N) per query with numpy
vectorization, fast enough for interactive use.

The symbolic concept network remains the source of truth for reasoning
and explanation. The embedding space provides the similarity substrate
that enables recognition and generalization. They are two systems,
connected, each doing what it's good at.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import threading
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .network import ConceptNetwork


class EmbeddingStore:
    """Latent space for semantic similarity.

    Combines graph spectral embedding, TF-IDF text embedding, and
    optional pre-trained word embeddings into a unified representation.

    Thread-safe after initialization. All vectors are read-only.
    """

    # Spectral embedding dimensions (graph structure)
    SPECTRAL_DIM = 32
    # TF-IDF embedding dimensions (text similarity)
    TFIDF_DIM = 128
    # Experiential embedding dimensions (sleep-discovered associations)
    # Built from SVD of the holographic graph's association matrix.
    EXPERIENTIAL_DIM = 32

    # Toroidal geometry: spectral + experiential dimensions are
    # mapped to angles on a torus T^d, where d = SPECTRAL_DIM +
    # EXPERIENTIAL_DIM. This makes the space periodic, enabling
    # representation of cyclical relationships (antonyms, recurrence)
    # that flat R^n cannot capture.
    #
    # The toroidal block is [0 : spectral_dim + experiential_dim].
    # The flat block is [spectral_dim + experiential_dim : end].
    # Hybrid similarity = α * geodesic(toroidal) + (1-α) * cosine(flat).
    TOROIDAL_WEIGHT = 0.3  # weight for toroidal similarity in hybrid

    # Stopwords to exclude from TF-IDF text tokens. These words carry
    # little semantic signal and create spurious similarity between
    # unrelated concepts that happen to share them in their definitions.
    _STOPWORDS: frozenset[str] = frozenset({
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "must", "shall",
        "can", "need", "dare", "ought", "used", "to", "of", "in", "for",
        "on", "at", "by", "with", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "over",
        "again", "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
        "too", "very", "just", "now", "also", "its", "it", "he", "she",
        "they", "we", "you", "i", "me", "him", "her", "us", "them", "my",
        "your", "his", "our", "their", "this", "that", "these", "those",
        "which", "who", "whom", "what", "whose",
    })
    # GloVe embedding dimensions (if available)
    GLOVE_DIM = 50

    def __init__(
        self,
        network: ConceptNetwork,
        data_dir: str = ".",
        filename: str = "embeddings.npz",
        dimension: int | None = None,
    ) -> None:
        """Initialize the embedding store.

        Loads cached embeddings from disk if present; otherwise prepares
        the store for incremental learning from the concept network.

        Args:
            network: The concept network whose nodes will be embedded.
            data_dir: Directory holding the cached ``embeddings.npz``.
            filename: Cache filename for the embedding matrix.
            dimension: Optional fixed target dimension. If None, the
                dimension is derived from the embedding composition.
        """
        self.network = network
        self._path = os.path.join(data_dir, filename)

        # Pre-trained word vectors (from GloVe, optional)
        self._word_vectors: np.ndarray | None = None
        self._word_list: np.ndarray | None = None
        self._word_to_idx: dict[str, int] = {}

        # Concept vectors (computed from all available sources)
        self._concept_matrix: np.ndarray | None = None
        self._concept_names: list[str] = []
        self._concept_to_idx: dict[str, int] = {}

        # Cold layer — flat-block vectors for archived (dormant)
        # concepts. Archived concepts have no live graph position, so
        # they get no spectral/experiential block; only the flat
        # TF-IDF + GloVe portion is stored. Consulted as a fallback
        # when a hot-matrix search returns fewer than k results, so
        # dormant knowledge stays discoverable by similarity without
        # inflating the hot search path.
        self._archive_matrix: np.ndarray | None = None
        self._archive_names: list[str] = []
        self._archive_to_idx: dict[str, int] = {}

        # TF-IDF vocabulary
        self._tfidf_vocab: dict[str, int] = {}
        self._idf: np.ndarray | None = None

        # Per-concept cache for ad-hoc queries
        self._concept_cache: dict[str, np.ndarray | None] = {}

        self._loaded = False
        self._lock = threading.Lock()

        # Dimensionality (set at build time)
        self._dim = 0

        # Target dimensionality — if set, vectors are padded/truncated
        # to this size. Allows configurable higher-dimensional embeddings
        # (256, 512, 1024). When None, dimensionality is determined by
        # the natural concatenation of spectral + TF-IDF + GloVe.
        self._target_dim = dimension

        # Component offsets within the concatenated vector
        # (spectral, experiential, tfidf, glove) — used for
        # dimension-aware search and toroidal geometry.
        self._spectral_offset = 0
        self._spectral_dim = 0
        self._experiential_offset = 0
        self._experiential_dim = 0
        self._tfidf_offset = 0
        self._tfidf_dim = 0
        self._glove_offset = 0
        self._glove_dim = 0

        # Holographic graph reference — used to build the experiential
        # layer. Set via set_holographic_graph() after mind startup.
        self._holographic_graph: Any = None

        # Toroidal geometry: the toroidal block is the spectral +
        # experiential dimensions. Each dimension is mapped to an
        # angle θ ∈ [0, 2π) via min-max scaling. The angle matrix
        # and ranges are computed at build time.
        self._toroidal_dim = 0  # spectral_dim + experiential_dim
        self._angle_matrix: np.ndarray | None = None  # (N, toroidal_dim)
        self._toroidal_ranges: np.ndarray | None = None  # (toroidal_dim, 2)

        # Precomputed L2 norms of each concept row with the spectral
        # portion zeroed out. Used to avoid copying the full concept
        # matrix during non-spectral (text) searches.
        self._non_spectral_norms: np.ndarray | None = None

    # ─── Public API ─────────────────────────────────────────────

    def set_holographic_graph(self, hgraph: Any) -> None:
        """Wire the holographic graph for experiential layer building.

        After calling this, the embedding store can build the
        experiential layer from the holographic graph's sleep-
        discovered associations. Call ``refresh()`` to rebuild.
        """
        self._holographic_graph = hgraph

    @property
    def has_embeddings(self) -> bool:
        """Whether concept embeddings are available."""
        self._ensure_loaded()
        return self._concept_matrix is not None

    @property
    def has_glove(self) -> bool:
        """Whether pre-trained GloVe vectors are available."""
        self._ensure_loaded()
        return self._word_vectors is not None

    @property
    def dimensionality(self) -> int:
        """Dimensionality of the embedding space (0 if not loaded)."""
        self._ensure_loaded()
        return self._dim

    @property
    def concept_count(self) -> int:
        """Number of concepts with computed vectors."""
        self._ensure_loaded()
        return len(self._concept_names)

    @property
    def archive_concept_count(self) -> int:
        """Number of archived concepts with cold-layer vectors."""
        self._ensure_loaded()
        return len(self._archive_names)

    def get_concept_matrix(
        self,
    ) -> tuple[np.ndarray | None, list[str]]:
        """Return the concept matrix and corresponding concept names.

        This is the public accessor for the internal matrix used by
        VQ training and sleep compression. Ensures embeddings are
        loaded before returning.

        Returns:
            (matrix, names) where matrix is shape (N, D) float32 or
            None if no embeddings are available, and names is the
            list of concept IDs aligned with matrix rows.
        """
        self._ensure_loaded()
        return self._concept_matrix, self._concept_names

    def get_word_vector(self, word: str) -> np.ndarray | None:
        """Get the GloVe vector for a single word.

        Returns None if GloVe is not loaded or the word is not in
        the vocabulary.
        """
        self._ensure_loaded()
        if self._word_vectors is None:
            return None
        idx = self._word_to_idx.get(word.lower())
        if idx is None:
            return None
        return self._word_vectors[idx]

    def get_concept_vector(self, concept_name: str) -> np.ndarray | None:
        """Get the vector for a concept in the network.

        The vector combines spectral (graph structure), TF-IDF (text),
        and GloVe (distributional) components.
        """
        self._ensure_loaded()
        if self._concept_matrix is None:
            return None

        # Check the precomputed index first
        idx = self._concept_to_idx.get(concept_name)
        if idx is not None:
            return self._concept_matrix[idx]

        # Check the ad-hoc cache
        if concept_name in self._concept_cache:
            return self._concept_cache[concept_name]

        # Check the cold layer — archived concepts have a flat-block
        # vector stored without the toroidal (spectral + experiential)
        # block. Expand to full width with zeros so callers can compare
        # against hot vectors; the zeros contribute nothing to cosine.
        archive_idx = self._archive_to_idx.get(concept_name)
        if archive_idx is not None and self._archive_matrix is not None:
            full = np.zeros(self._dim, dtype=np.float32)
            flat_start = self._spectral_dim + self._experiential_dim
            full[flat_start:] = self._archive_matrix[archive_idx]
            self._concept_cache[concept_name] = full
            return full

        # Compute on the fly
        vec = self._compute_concept_vector(concept_name)
        self._concept_cache[concept_name] = vec
        return vec

    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (n1 * n2))

    def find_similar_concepts(
        self,
        concept_name: str,
        k: int = 5,
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Find concepts similar to the given concept using hybrid distance.

        Uses the concept's own vector in the unified latent space,
        combining toroidal similarity (experiential layer via geodesic
        distance on T^d) with flat similarity (TF-IDF + GloVe via
        cosine distance). The spectral component is excluded from
        the toroidal block because it's polluted by hub attachments
        — concepts attached to the same hub appear close even when
        semantically unrelated.

        When the experiential layer is not available (no holographic
        graph wired), falls back to text-only search (the original
        behavior).

        The query concept itself is excluded from results.

        Args:
            concept_name: The concept to find neighbors for.
            k: Maximum number of results.
            threshold: Minimum hybrid similarity (0-1).

        Returns:
            List of (concept_name, similarity) pairs, sorted by
            similarity descending.
        """
        self._ensure_loaded()
        if self._concept_matrix is None:
            return []

        # If no experiential layer, fall back to text search.
        # The toroidal block without the experiential layer is just
        # the spectral component, which is polluted by hub attachments.
        if self._experiential_dim == 0:
            concept = self.network.get_concept(concept_name)
            query = concept_name.replace("_", " ")
            if concept:
                definition = concept.properties.get("definition", "")
                if definition and definition != "NO DEF":
                    query = f"{query} {definition}"
                for alias in getattr(concept, "aliases", set()):
                    if alias and alias != concept_name:
                        query = f"{query} {alias.replace('_', ' ')}"
            if not query:
                return []
            return self.find_similar_to_text(
                query, k=k, threshold=threshold, exclude=concept_name
            )

        # Resolve the concept to its index in the concept matrix
        idx = self._concept_to_idx.get(concept_name)
        if idx is None:
            # Concept not in the matrix — fall back to text search
            concept = self.network.get_concept(concept_name)
            query = concept_name.replace("_", " ")
            if concept:
                definition = concept.properties.get("definition", "")
                if definition and definition != "NO DEF":
                    query = f"{query} {definition}"
                for alias in getattr(concept, "aliases", set()):
                    if alias and alias != concept_name:
                        query = f"{query} {alias.replace('_', ' ')}"
            if not query:
                return []
            return self.find_similar_to_text(
                query, k=k, threshold=threshold, exclude=concept_name
            )

        # Use the concept's own vector for hybrid similarity search
        query_vec = self._concept_matrix[idx]
        return self._search_hybrid(
            query_vec, k, threshold, exclude=concept_name
        )

    def _search_hybrid(
        self,
        query_vec: np.ndarray,
        k: int,
        threshold: float,
        exclude: str | None = None,
    ) -> list[tuple[str, float]]:
        """Search using hybrid toroidal + flat distance.

        The toroidal block (spectral + experiential) uses geodesic
        distance on T^d. The flat block (TF-IDF + GloVe) uses cosine
        similarity. The two are combined with TOROIDAL_WEIGHT.

        This is the vesica-aware search: concepts that are close in
        BOTH toroidal and flat space rank higher than those close in
        only one — the geometric intersection of neighborhoods.
        """
        if self._concept_matrix is None:
            return []

        # Compute flat similarity (cosine over TF-IDF + GloVe)
        flat_start = self._spectral_dim + self._experiential_dim
        if flat_start < self._dim:
            query_flat = query_vec[flat_start:]
            matrix_flat = self._concept_matrix[:, flat_start:]
            # Normalize
            q_norm = np.linalg.norm(query_flat)
            if q_norm < 1e-8:
                flat_sim = np.zeros(self._concept_matrix.shape[0], dtype=np.float32)
            else:
                query_flat = query_flat / q_norm
                m_norms = np.linalg.norm(matrix_flat, axis=1)
                safe = m_norms > 1e-8
                flat_sim = np.where(
                    safe,
                    (matrix_flat @ query_flat) / np.where(safe, m_norms, 1.0),
                    0.0,
                )
        else:
            flat_sim = np.ones(self._concept_matrix.shape[0], dtype=np.float32)

        # Compute toroidal similarity (geodesic on T^d)
        query_angles = self._map_vector_to_angles(query_vec)
        if query_angles is not None:
            toroidal_sim = self._toroidal_similarity(query_angles)
        else:
            toroidal_sim = np.ones(self._concept_matrix.shape[0], dtype=np.float32)

        # Combine: weighted sum of toroidal and flat similarities
        alpha = self.TOROIDAL_WEIGHT
        similarities = (
            alpha * toroidal_sim + (1.0 - alpha) * flat_sim
        ).astype(np.float32)

        results = self._collect_search_results(similarities, threshold, k, exclude)
        if len(results) < k:
            results.extend(
                self._search_archive_flat(
                    query_vec, k - len(results), threshold,
                    exclude_names={exclude, *(n for n, _ in results)},
                )
            )
        return results

    def find_similar_to_text(
        self,
        text: str,
        k: int = 5,
        threshold: float = 0.3,
        exclude: str | None = None,
    ) -> list[tuple[str, float]]:
        """Find concepts similar to arbitrary text (semantic matching).

        This is the pattern recognition method. It takes any text —
        a sentence, a phrase, a question — and finds concepts in the
        network whose vectors are closest.

        The text vector is computed from:
        1. TF-IDF: matching words in concept definitions
        2. GloVe (if available): word vector average for distributional match

        Example: "the tall plant in my yard" → [("tree", 0.82),
        ("plant", 0.75), ("garden", 0.68), ...]

        Args:
            text: Input text to match.
            k: Maximum number of results.
            threshold: Minimum cosine similarity.
            exclude: Optional concept name to exclude from results.

        Returns:
            List of (concept_name, similarity) pairs.
        """
        self._ensure_loaded()
        if self._concept_matrix is None:
            return []

        vec = self._compute_text_vector(text)
        if vec is None:
            return []

        return self._search_concept_matrix(
            vec, k, threshold, exclude=exclude, query_has_spectral=False
        )

    def find_similar_to_vector(
        self,
        vector: np.ndarray,
        k: int = 5,
        threshold: float = 0.5,
        query_has_spectral: bool = True,
    ) -> list[tuple[str, float]]:
        """Find concepts similar to a given vector."""
        self._ensure_loaded()
        if self._concept_matrix is None:
            return []
        return self._search_concept_matrix(
            vector, k, threshold, query_has_spectral=query_has_spectral
        )

    def find_concepts_related_to_both(
        self,
        concept_a: str,
        concept_b: str,
        k: int = 5,
        threshold: float = 0.3,
    ) -> list[tuple[str, float]]:
        """Vesica query: find concepts in the intersection of two neighborhoods.

        This is the geometric join operation — the vesica piscis of two
        concept neighborhoods on the torus. A concept ranks high if it
        is close to BOTH A and B, not just one. The combined score is
        the minimum of the two similarities (intersection, not union).

        This enables analogical reasoning: "A is to B as C is to ?"
        requires finding C in the vesica of A's and B's neighborhoods.

        Args:
            concept_a: First concept.
            concept_b: Second concept.
            k: Maximum number of results.
            threshold: Minimum combined similarity.

        Returns:
            List of (concept_name, combined_similarity) pairs.
        """
        self._ensure_loaded()
        if self._concept_matrix is None:
            return []

        idx_a = self._concept_to_idx.get(concept_a)
        idx_b = self._concept_to_idx.get(concept_b)
        if idx_a is None or idx_b is None:
            return []

        vec_a = self._concept_matrix[idx_a]
        vec_b = self._concept_matrix[idx_b]

        # Compute hybrid similarities for both concepts
        sim_a = self._compute_hybrid_similarities(vec_a)
        sim_b = self._compute_hybrid_similarities(vec_b)

        # Vesica: intersection = minimum of the two similarities
        combined = np.minimum(sim_a, sim_b)

        # Exclude A and B from results
        exclude_set = {idx_a, idx_b}
        results = self._collect_search_results_excluding(
            combined, threshold, k, exclude_set
        )
        if len(results) < k:
            results.extend(
                self._search_archive_vesica(
                    vec_a, vec_b, k - len(results), threshold,
                    exclude_names={
                        concept_a, concept_b, *(n for n, _ in results)
                    },
                )
            )
        return results

    def _compute_hybrid_similarities(
        self, query_vec: np.ndarray
    ) -> np.ndarray:
        """Compute hybrid similarities for all concepts (no filtering)."""
        assert self._concept_matrix is not None
        n = self._concept_matrix.shape[0]

        # Flat similarity
        flat_start = self._spectral_dim + self._experiential_dim
        if flat_start < self._dim:
            query_flat = query_vec[flat_start:]
            matrix_flat = self._concept_matrix[:, flat_start:]
            q_norm = np.linalg.norm(query_flat)
            if q_norm < 1e-8:
                flat_sim = np.zeros(n, dtype=np.float32)
            else:
                query_flat = query_flat / q_norm
                m_norms = np.linalg.norm(matrix_flat, axis=1)
                safe = m_norms > 1e-8
                flat_sim = np.where(
                    safe,
                    (matrix_flat @ query_flat) / np.where(safe, m_norms, 1.0),
                    0.0,
                )
        else:
            flat_sim = np.ones(n, dtype=np.float32)

        # Toroidal similarity
        query_angles = self._map_vector_to_angles(query_vec)
        if query_angles is not None:
            toroidal_sim = self._toroidal_similarity(query_angles)
        else:
            toroidal_sim = np.ones(n, dtype=np.float32)

        return (
            self.TOROIDAL_WEIGHT * toroidal_sim
            + (1.0 - self.TOROIDAL_WEIGHT) * flat_sim
        ).astype(np.float32)

    def _collect_search_results_excluding(
        self,
        similarities: np.ndarray,
        threshold: float,
        k: int,
        exclude_indices: set[int],
    ) -> list[tuple[str, float]]:
        """Filter, sort, and build results, excluding multiple indices."""
        above = np.where(similarities >= threshold)[0]
        if len(above) == 0:
            return []
        sorted_idx = above[np.argsort(-similarities[above])]
        results: list[tuple[str, float]] = []
        for idx in sorted_idx:
            if idx not in exclude_indices:
                results.append((self._concept_names[idx], float(similarities[idx])))
            if len(results) >= k:
                break
        return results

    def refresh(self) -> None:
        """Recompute concept vectors after the network changes.

        Call this after adding new concepts to the network.

        This rebuilds the spectral component (which depends on graph
        structure) while preserving the experiential components
        (TF-IDF + GloVe) that carry Hebbian learning. New concepts
        that weren't in the previous matrix get their experiential
        vectors computed from scratch; existing concepts retain their
        Hebbian-adapted vectors.
        """
        self._refresh_preserve_experiential()
        self._concept_cache.clear()

    def _restore_hebbian_experiential(
        self,
        matrix: np.ndarray,
        concepts: list[str],
        old_matrix: np.ndarray | None,
        old_to_idx: dict[str, int],
        old_tfidf_vocab: dict[str, int],
        old_tfidf_dim: int,
        old_spectral_dim: int,
        old_experiential_dim: int = 0,
    ) -> None:
        """Restore Hebbian-adapted experiential vectors for existing concepts.

        Only overwrite the TF-IDF + GloVe portion; leave the freshly-
        computed spectral and experiential portions as-is.

        TF-IDF columns are re-mapped by word: old column j (word w) →
        new column self._tfidf_vocab[w]. This prevents vocabulary drift
        from silently scrambling learned weights. GloVe columns are
        indexed by a fixed word list and can be copied directly.
        """
        if old_matrix is None or self._spectral_dim <= 0:
            return
        # Skip both spectral AND experiential in the new matrix.
        # The old matrix may or may not have an experiential layer —
        # skip both spectral and experiential in the old matrix too.
        exp_start = self._spectral_dim + self._experiential_dim
        old_exp_start = old_spectral_dim + old_experiential_dim

        # Build old→new TF-IDF column mapping
        new_tfidf_vocab = self._tfidf_vocab
        tfidf_col_map: dict[int, int] = {}  # old_col → new_col
        if old_tfidf_vocab and new_tfidf_vocab and old_tfidf_dim > 0:
            for word, old_col in old_tfidf_vocab.items():
                new_col = new_tfidf_vocab.get(word)
                if new_col is not None:
                    tfidf_col_map[old_col] = new_col

        for i, name in enumerate(concepts):
            old_i = old_to_idx.get(name)
            if old_i is not None and exp_start < old_matrix.shape[1]:
                old_exp = old_matrix[old_i, old_exp_start:]
                new_exp_len = matrix.shape[1] - exp_start

                # Re-map TF-IDF columns by word
                if tfidf_col_map and self._tfidf_dim > 0:
                    for old_col, new_col in tfidf_col_map.items():
                        if old_col < old_exp.shape[0] and new_col < new_exp_len:
                            matrix[i, exp_start + new_col] = old_exp[old_col]
                    # Copy GloVe columns directly (fixed word list, no drift)
                    glove_start = old_tfidf_dim
                    new_glove_start = self._tfidf_dim
                    if self._glove_dim > 0 and glove_start < old_exp.shape[0]:
                        glove_copy_len = min(
                            self._glove_dim,
                            old_exp.shape[0] - glove_start,
                            new_exp_len - new_glove_start,
                        )
                        if glove_copy_len > 0:
                            matrix[i, exp_start + new_glove_start:
                                   exp_start + new_glove_start + glove_copy_len] = (
                                old_exp[glove_start:glove_start + glove_copy_len]
                            )
                else:
                    # No TF-IDF mapping (e.g., first refresh or no vocab):
                    # copy by position as a fallback
                    copy_len = min(old_exp.shape[0], new_exp_len)
                    matrix[i, exp_start:exp_start + copy_len] = old_exp[:copy_len]

    def _refresh_preserve_experiential(self) -> None:
        """Refresh the spectral component while preserving Hebbian learning.

        The spectral component is derived from graph structure and must
        be rebuilt when the graph changes (new edges, weight updates).
        The experiential components (TF-IDF + GloVe) carry Hebbian
        adaptation from waking and sleep consolidation — rebuilding
        them from scratch would erase that learning.

        For concepts that existed before, we keep their experiential
        vectors. For new concepts, we compute fresh experiential vectors.
        The spectral component is always rebuilt.

        **Vocabulary alignment**: The TF-IDF vocabulary is rebuilt from
        the current corpus, so words may shift to different column
        indices. We re-map the old TF-IDF columns to the new vocabulary
        positions by word, ensuring that a Hebbian-adapted weight for
        "brain" stays on the "brain" column. GloVe columns are indexed
        by a fixed word list and do not drift.
        """
        concepts = list(self.network._concepts.keys())
        if not concepts:
            self._concept_matrix = None
            self._concept_names = []
            self._concept_to_idx = {}
            self._non_spectral_norms = None
            self._archive_matrix = None
            self._archive_names = []
            self._archive_to_idx = {}
            return

        # Save the old experiential data and vocabulary before rebuilding
        old_matrix = self._concept_matrix
        old_to_idx = self._concept_to_idx
        old_tfidf_vocab = self._tfidf_vocab  # word → old column index
        old_tfidf_dim = self._tfidf_dim
        old_spectral_dim = self._spectral_dim  # save before it gets overwritten
        old_experiential_dim = self._experiential_dim  # save before overwrite

        # Build fresh spectral component
        spectral = self._build_spectral_embedding(concepts)
        experiential = self._build_experiential_embedding(concepts)
        tfidf = self._build_tfidf_embedding(concepts)
        glove = self._build_glove_embedding(concepts)

        # Concatenate components, tracking offsets
        # Order: spectral, experiential, tfidf, glove
        components = []
        offset = 0

        if spectral is not None:
            self._spectral_offset = offset
            self._spectral_dim = spectral.shape[1]
            components.append(spectral)
            offset += self._spectral_dim

        if experiential is not None:
            self._experiential_offset = offset
            self._experiential_dim = experiential.shape[1]
            components.append(experiential)
            offset += self._experiential_dim

        if tfidf is not None:
            self._tfidf_offset = offset
            self._tfidf_dim = tfidf.shape[1]
            components.append(tfidf)
            offset += self._tfidf_dim

        if glove is not None:
            self._glove_offset = offset
            self._glove_dim = glove.shape[1]
            components.append(glove)
            offset += self._glove_dim

        if not components:
            self._concept_matrix = None
            self._concept_names = []
            self._concept_to_idx = {}
            self._non_spectral_norms = None
            return

        matrix = np.concatenate(components, axis=1).astype(np.float32)

        # Restore Hebbian-adapted experiential vectors for existing concepts.
        self._restore_hebbian_experiential(
            matrix, concepts, old_matrix, old_to_idx,
            old_tfidf_vocab, old_tfidf_dim, old_spectral_dim,
            old_experiential_dim,
        )

        # Normalize
        matrix = self._normalize_rows(matrix)

        # Handle target dimension
        if self._target_dim is not None and matrix.shape[1] != self._target_dim:
            matrix = self._pad_or_truncate_matrix(matrix, self._target_dim)
            matrix = self._normalize_rows(matrix)

        self._concept_matrix = matrix
        self._concept_names = concepts
        self._concept_to_idx = {n: i for i, n in enumerate(concepts)}
        self._dim = matrix.shape[1]
        self._build_toroidal_angles(matrix)
        self._non_spectral_norms = self._compute_non_spectral_norms(matrix)
        self._build_archive_matrix()

    # ─── Persistence ─────────────────────────────────────────────

    def save_experiential(self, data_dir: str | None = None) -> None:
        """Save the Hebbian-adapted experiential vectors to disk.

        The spectral component is always recomputed from the graph, so
        only the experiential portion (TF-IDF + GloVe) needs persisting.
        This is called after sleep consolidation to preserve Hebbian
        learning across restarts.

        The TF-IDF vocabulary is saved alongside the vectors so that
        ``load_experiential`` can re-map columns to the current
        vocabulary, preventing drift when the corpus changes between
        sessions.

        Args:
            data_dir: Directory to save to. Defaults to the store's
                data_dir.
        """
        self._ensure_loaded()
        if self._concept_matrix is None:
            return

        ddir = data_dir or os.path.dirname(self._path)
        path = os.path.join(ddir, "concept_vectors.npz")

        # Extract only the TF-IDF + GloVe portion (skip spectral AND
        # experiential — the experiential layer is rebuilt from the
        # holographic graph on each startup, not persisted).
        exp_start = self._spectral_dim + self._experiential_dim
        if exp_start == 0:
            exp_start = self._spectral_dim if self._spectral_dim > 0 else 0
        exp_matrix = self._concept_matrix[:, exp_start:]

        # Save the TF-IDF vocabulary so columns can be re-mapped on load
        tfidf_words = np.array(
            list(self._tfidf_vocab.keys()), dtype=object
        ) if self._tfidf_vocab else np.array([], dtype=object)
        tfidf_indices = np.array(
            list(self._tfidf_vocab.values()), dtype=np.int32
        ) if self._tfidf_vocab else np.array([], dtype=np.int32)

        try:
            np.savez(
                path,
                concepts=np.array(self._concept_names, dtype=object),
                vectors=exp_matrix.astype(np.float32),
                spectral_dim=np.array(self._spectral_dim),
                tfidf_dim=np.array(self._tfidf_dim),
                tfidf_words=tfidf_words,
                tfidf_indices=tfidf_indices,
            )
        except (OSError, ValueError) as e:
            logger.debug(f"Hebbian learning save failed: {e}")

    def load_experiential(self, data_dir: str | None = None) -> bool:
        """Load persisted Hebbian-adapted experiential vectors from disk.

        Called after _build() on startup. Restores the experiential
        portion of the concept matrix from a previous session's
        Hebbian learning. Only restores vectors for concepts that still
        exist in the network.

        **Vocabulary alignment**: The saved TF-IDF vocabulary may differ
        from the current vocabulary (concepts added/removed between
        sessions). We re-map saved TF-IDF columns to the current
        vocabulary positions by word, ensuring that a learned weight
        for "brain" stays on the "brain" column. GloVe columns are
        indexed by a fixed word list and can be copied directly.

        For backwards compatibility, files saved without vocabulary
        metadata fall back to positional copying (the old behavior).

        Args:
            data_dir: Directory to load from. Defaults to the store's
                data_dir.

        Returns:
            True if any vectors were restored.
        """
        if self._concept_matrix is None:
            return False

        ddir = data_dir or os.path.dirname(self._path)
        path = os.path.join(ddir, "concept_vectors.npz")

        if not os.path.exists(path):
            return False

        try:
            data = np.load(path, allow_pickle=True)
            saved_concepts = [str(c) for c in data["concepts"]]
            saved_vectors = data["vectors"]
        except (KeyError, ValueError, OSError):
            return False

        # Skip both spectral AND experiential — only restore TF-IDF + GloVe.
        # The experiential layer is rebuilt from the holographic graph.
        exp_start = self._spectral_dim + self._experiential_dim
        if exp_start == 0:
            exp_start = self._spectral_dim if self._spectral_dim > 0 else 0
        restored = 0

        # Load saved TF-IDF vocabulary for column re-mapping
        saved_tfidf_vocab: dict[str, int] = {}
        saved_tfidf_dim = 0
        try:
            saved_tfidf_dim = int(data["tfidf_dim"])
            words = data["tfidf_words"]
            indices = data["tfidf_indices"]
            saved_tfidf_vocab = {
                str(w): int(idx)
                for w, idx in zip(words, indices, strict=False)
            }
        except (KeyError, ValueError) as e:
            logger.debug(f'silent except: {e}')

        # Build saved→current TF-IDF column mapping
        tfidf_col_map: dict[int, int] = {}  # saved_col → current_col
        if saved_tfidf_vocab and self._tfidf_vocab and saved_tfidf_dim > 0:
            for word, saved_col in saved_tfidf_vocab.items():
                current_col = self._tfidf_vocab.get(word)
                if current_col is not None:
                    tfidf_col_map[saved_col] = current_col

        for i, name in enumerate(saved_concepts):
            current_idx = self._concept_to_idx.get(name)
            if current_idx is not None:
                saved_exp = saved_vectors[i]
                new_exp_len = self._concept_matrix.shape[1] - exp_start
                if self._restore_saved_vector(
                    saved_exp, current_idx, exp_start, new_exp_len,
                    tfidf_col_map, saved_tfidf_dim,
                ):
                    restored += 1

        if restored > 0:
            # Re-normalize after restoring
            self._concept_matrix = self._normalize_rows(self._concept_matrix)
            self._non_spectral_norms = self._compute_non_spectral_norms(
                self._concept_matrix
            )
            self._concept_cache.clear()

        return restored > 0

    def _restore_saved_vector(
        self,
        saved_exp: np.ndarray,
        current_idx: int,
        exp_start: int,
        new_exp_len: int,
        tfidf_col_map: dict[int, int],
        saved_tfidf_dim: int,
    ) -> bool:
        """Restore one concept's experiential vector from a saved session.

        Re-maps TF-IDF columns by word and copies GloVe columns directly
        (fixed word list, no drift). Falls back to positional copying
        when no vocabulary mapping is available (backwards-compatible).
        """
        if self._concept_matrix is None:
            return False
        if tfidf_col_map and self._tfidf_dim > 0:
            # Re-map TF-IDF columns by word
            for saved_col, current_col in tfidf_col_map.items():
                if saved_col < saved_exp.shape[0] and current_col < new_exp_len:
                    self._concept_matrix[current_idx, exp_start + current_col] = (
                        saved_exp[saved_col]
                    )
            # Copy GloVe columns directly (fixed word list, no drift)
            glove_start = saved_tfidf_dim
            new_glove_start = self._tfidf_dim
            if self._glove_dim > 0 and glove_start < saved_exp.shape[0]:
                glove_copy_len = min(
                    self._glove_dim,
                    saved_exp.shape[0] - glove_start,
                    new_exp_len - new_glove_start,
                )
                if glove_copy_len > 0:
                    self._concept_matrix[
                        current_idx,
                        exp_start + new_glove_start:
                        exp_start + new_glove_start + glove_copy_len
                    ] = saved_exp[glove_start:glove_start + glove_copy_len]
        else:
            # Backwards-compatible fallback: copy by position
            copy_len = min(saved_exp.shape[0], new_exp_len)
            self._concept_matrix[current_idx, exp_start:exp_start + copy_len] = (
                saved_exp[:copy_len]
            )
        return True

    # ─── Internal: loading and building ─────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazily build/load embeddings on first access (thread-safe)."""
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._load()
            self._loaded = True  # Set AFTER load completes

    def _load(self) -> None:
        """Load GloVe (if available) and build concept embeddings."""
        # Try loading pre-trained GloVe vectors
        self._load_glove()

        # Build concept embeddings from all available sources
        self._build()

        # Restore Hebbian-adapted experiential vectors from previous session
        self.load_experiential()

    def _load_glove(self) -> None:
        """Load pre-trained GloVe vectors from .npz file if it exists."""
        if not os.path.exists(self._path):
            return

        try:
            data = np.load(self._path, allow_pickle=False)
            self._word_list = data["words"]
            self._word_vectors = data["vectors"].astype(np.float32)
            self._word_to_idx = {str(w): i for i, w in enumerate(self._word_list)}
        except (KeyError, ValueError, OSError):
            self._word_vectors = None
            self._word_list = None
            self._word_to_idx = {}

    def _build(self) -> None:
        """Build the concept embedding matrix from all sources.

        Combines:
        1. Spectral embedding of the graph (always available)
        2. Experiential embedding from holographic graph (if wired)
        3. TF-IDF over concept definitions (always available)
        4. GloVe word vectors (if available)

        The spectral + experiential dimensions form the toroidal
        block — they are mapped to angles on T^d for geodesic
        distance. The TF-IDF + GloVe dimensions form the flat block
        — they use cosine similarity.
        """
        concepts = list(self.network._concepts.keys())
        if not concepts:
            self._concept_matrix = None
            self._concept_names = []
            self._concept_to_idx = {}
            self._non_spectral_norms = None
            self._angle_matrix = None
            self._toroidal_ranges = None
            self._archive_matrix = None
            self._archive_names = []
            self._archive_to_idx = {}
            return

        # Build each component
        spectral = self._build_spectral_embedding(concepts)
        experiential = self._build_experiential_embedding(concepts)
        tfidf = self._build_tfidf_embedding(concepts)
        glove = self._build_glove_embedding(concepts)

        # Concatenate all available components, tracking offsets.
        # Order: spectral, experiential, tfidf, glove.
        # The toroidal block is spectral + experiential.
        components = []
        offset = 0

        if spectral is not None:
            self._spectral_offset = offset
            self._spectral_dim = spectral.shape[1]
            components.append(spectral)
            offset += self._spectral_dim

        if experiential is not None:
            self._experiential_offset = offset
            self._experiential_dim = experiential.shape[1]
            components.append(experiential)
            offset += self._experiential_dim

        if tfidf is not None:
            self._tfidf_offset = offset
            self._tfidf_dim = tfidf.shape[1]
            components.append(tfidf)
            offset += self._tfidf_dim

        if glove is not None:
            self._glove_offset = offset
            self._glove_dim = glove.shape[1]
            components.append(glove)
            offset += self._glove_dim

        if not components:
            self._concept_matrix = None
            self._concept_names = []
            self._concept_to_idx = {}
            self._non_spectral_norms = None
            self._angle_matrix = None
            self._toroidal_ranges = None
            self._archive_matrix = None
            self._archive_names = []
            self._archive_to_idx = {}
            return

        # Concatenate and normalize
        matrix = np.concatenate(components, axis=1).astype(np.float32)
        matrix = self._normalize_rows(matrix)

        # If a target dimension is set, pad with zeros (or truncate)
        # to match the desired dimensionality. This allows configurable
        # higher-dimensional embeddings (256, 512, 1024).
        if self._target_dim is not None and matrix.shape[1] != self._target_dim:
            matrix = self._pad_or_truncate_matrix(matrix, self._target_dim)
            # Re-normalize after padding (padding with zeros doesn't
            # change the norm, but truncation does)
            matrix = self._normalize_rows(matrix)

        self._concept_matrix = matrix
        self._concept_names = concepts
        self._concept_to_idx = {n: i for i, n in enumerate(concepts)}
        self._dim = matrix.shape[1]

        # Build the toroidal angle mapping for the toroidal block
        # (spectral + experiential dimensions). Each dimension is
        # mapped to an angle θ ∈ [0, 2π) via min-max scaling.
        self._build_toroidal_angles(matrix)

        # Precompute the L2 norms of each row with the spectral portion
        # zeroed out, so non-spectral (text) searches can avoid copying
        # and re-normalizing the full concept matrix.
        self._non_spectral_norms = self._compute_non_spectral_norms(matrix)

        # Cold layer — flat-block vectors for archived concepts
        self._build_archive_matrix()

    # ─── Toroidal geometry ───────────────────────────────────────

    def _build_toroidal_angles(self, matrix: np.ndarray) -> None:
        """Map the experiential layer to angles on a torus.

        The experiential dimensions (from the holographic graph's
        sleep-discovered associations) are mapped to angles θ ∈ [0, 2π)
        via min-max scaling. This makes the space periodic: concepts
        at opposite ends of a dimension (in flat space) are close on
        the torus (wrap-around distance).

        The spectral dimensions are NOT included in the toroidal block
        because they're polluted by hub attachments — concepts attached
        to the same hub appear close even when semantically unrelated.
        The experiential layer is clean (it comes from sleep-discovered
        associations, not graph topology).
        """
        toroidal_dim = self._experiential_dim
        if toroidal_dim == 0 or matrix is None:
            self._toroidal_dim = 0
            self._angle_matrix = None
            self._toroidal_ranges = None
            return

        # Extract the experiential block
        exp_start = self._spectral_dim
        exp_end = exp_start + self._experiential_dim
        toroidal_block = matrix[:, exp_start:exp_end]

        # Compute min/max for each dimension
        ranges = np.zeros((toroidal_dim, 2), dtype=np.float32)
        for d in range(toroidal_dim):
            col = toroidal_block[:, d]
            ranges[d, 0] = float(col.min())
            ranges[d, 1] = float(col.max())

        # Map to angles: θ = 2π * (x - min) / (max - min)
        # Handle constant dimensions (max == min) by setting θ = 0
        angles = np.zeros_like(toroidal_block)
        for d in range(toroidal_dim):
            span = ranges[d, 1] - ranges[d, 0]
            if span > 1e-8:
                angles[:, d] = 2.0 * np.pi * (
                    toroidal_block[:, d] - ranges[d, 0]
                ) / span

        self._toroidal_dim = toroidal_dim
        self._angle_matrix = angles.astype(np.float32)
        self._toroidal_ranges = ranges

    def _toroidal_similarity(
        self, query_angles: np.ndarray
    ) -> np.ndarray:
        """Compute geodesic similarity on the torus for all concepts.

        Given a query angle vector (toroidal_dim,), returns an (N,)
        array of similarities in [0, 1], where 1 means identical
        position on the torus and 0 means maximally distant.

        Geodesic distance on T^d:
            d(θ, φ) = sqrt(sum_i min(|θ_i - φ_i|, 2π - |θ_i - φ_i|)^2)

        Similarity = 1 - d / (π * sqrt(d)), normalized to [0, 1].
        """
        if self._angle_matrix is None or self._toroidal_dim == 0:
            n = self._concept_matrix.shape[0] if self._concept_matrix is not None else 0
            return np.ones(n, dtype=np.float32)

        # Compute angular differences: (N, toroidal_dim)
        diff = np.abs(self._angle_matrix - query_angles[np.newaxis, :])

        # Wrap to [0, π]: min(diff, 2π - diff)
        diff = np.minimum(diff, 2.0 * np.pi - diff)

        # Geodesic distance: sqrt(sum of squared wrapped diffs)
        dist = np.sqrt(np.sum(diff * diff, axis=1))

        # Normalize to [0, 1] similarity
        max_dist = np.pi * np.sqrt(self._toroidal_dim)
        return 1.0 - dist / max_dist

    def _map_vector_to_angles(self, vec: np.ndarray) -> np.ndarray | None:
        """Map a vector's experiential block to angles.

        Used for concept-to-concept queries where the query vector
        has experiential components that need to be mapped to the same
        angle space as the concept matrix.
        """
        if self._toroidal_dim == 0 or self._toroidal_ranges is None:
            return None

        # Extract the experiential block from the vector
        exp_start = self._spectral_dim
        exp_end = exp_start + self._experiential_dim
        exp_block = vec[exp_start:exp_end]

        angles = np.zeros(self._toroidal_dim, dtype=np.float32)
        for d in range(self._toroidal_dim):
            span = self._toroidal_ranges[d, 1] - self._toroidal_ranges[d, 0]
            if span > 1e-8:
                angles[d] = 2.0 * np.pi * (
                    exp_block[d] - self._toroidal_ranges[d, 0]
                ) / span
        return angles

    # ─── Spectral embedding ─────────────────────────────────────

    def _build_spectral_embedding(self, concepts: list[str]) -> np.ndarray | None:
        """Build spectral embedding from graph structure.

        Concepts that share many neighbors are close in this space.
        Uses truncated SVD of the normalized adjacency matrix, which
        is equivalent to a spectral embedding of the graph Laplacian
        but more numerically stable for large graphs.

        For N concepts, this builds an NxN adjacency matrix (weighted
        by edge weights), normalizes it, and decomposes it into the
        top SPECTRAL_DIM singular vectors. The result is a low-dimensional
        embedding where structurally similar concepts are close.

        For very large graphs (>10k concepts), we use a sparse
        representation to avoid O(N²) memory.
        """
        n = len(concepts)
        if n < 2:
            return None

        # Limit spectral embedding to concepts with at least one edge
        # This reduces the matrix size for large networks
        concept_to_idx = {c: i for i, c in enumerate(concepts)}

        # Build adjacency matrix (weighted, symmetric)
        # For large networks, use sparse matrix
        if n > 5000:
            return self._build_spectral_sparse(concepts, concept_to_idx)

        # Dense path (small networks)
        adj = np.zeros((n, n), dtype=np.float32)

        for concept_id in concepts:
            i = concept_to_idx[concept_id]
            # Outgoing edges
            for edge in self.network.get_edges(concept_id, direction="out"):
                j = concept_to_idx.get(edge.target)
                if j is not None:
                    w = float(edge.weight)
                    adj[i, j] = max(adj[i, j], w)
                    adj[j, i] = max(adj[j, i], w)  # symmetric

        # Normalized adjacency (symmetric Laplacian variant)
        # D^{-1/2} A D^{-1/2}
        degrees = adj.sum(axis=1)
        degrees[degrees < 1e-8] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        normalized = adj * d_inv_sqrt[:, np.newaxis] * d_inv_sqrt[np.newaxis, :]

        # Truncated SVD of the normalized adjacency matrix.
        #
        # The normalized adjacency D^{-1/2}AD^{-1/2} shares eigenvectors
        # with the normalized Laplacian L = I - D^{-1/2}AD^{-1/2}: if
        # L*v = λ*v, then (D^{-1/2}AD^{-1/2})*v = (1-λ)*v. So the top
        # singular vectors of the normalized adjacency correspond to the
        # bottom eigenvectors of the Laplacian — the standard spectral
        # embedding.
        #
        # The FIRST singular vector is approximately constant (eigenvalue
        # ≈ 1, corresponding to λ ≈ 0 of the Laplacian). It carries no
        # discriminative information — it's the trivial eigenvector. We
        # skip it, taking vectors 1..dim+1 instead of 0..dim.
        #
        # Scaling by singular values (u * s) gives the diffusion map
        # with t=1, which weights structurally important dimensions more
        # heavily. This is standard practice (Coifman & Lafon, 2006).
        dim = min(self.SPECTRAL_DIM, n - 2)  # -2: skip trivial + bounds
        if dim < 1:
            dim = 1
        try:
            u, s, _vt = np.linalg.svd(normalized, full_matrices=False)
            # Skip the first singular vector (trivial/constant)
            embedding = u[:, 1 : dim + 1] * s[1 : dim + 1]
            return embedding.astype(np.float32)
        except np.linalg.LinAlgError:
            return None

    def _build_spectral_sparse(
        self,
        concepts: list[str],
        concept_to_idx: dict[str, int],
    ) -> np.ndarray | None:
        """Spectral embedding for large graphs using sparse operations.

        For large networks, building a dense NxN matrix is infeasible.
        Instead, we build a sparse adjacency list and use randomized
        projection to approximate the spectral embedding.

        This uses the Nyström method: sample a subset of landmark
        nodes, compute the exact spectral embedding for the landmark
        subgraph, then extend to all nodes via the landmark similarities.
        """
        n = len(concepts)

        landmarks, landmark_set, landmark_idx, n_landmarks = self._select_landmarks(
            concepts, n
        )

        landmark_embedding = self._compute_landmark_embedding(
            landmarks, landmark_set, landmark_idx, n_landmarks
        )
        if landmark_embedding is None:
            return None

        dim = landmark_embedding.shape[1]
        return self._extend_nystrom_embedding(
            concepts,
            concept_to_idx,
            landmarks,
            landmark_set,
            landmark_idx,
            n_landmarks,
            landmark_embedding,
            dim,
            n,
        )

    def _select_landmarks(
        self, concepts: list[str], n: int
    ) -> tuple[list[str], set[str], dict[str, int], int]:
        """Sample landmark nodes (concepts with most edges)."""
        # Use the edge index directly instead of calling get_edges
        # (which does name resolution on every call — O(n) dict lookups
        # for 125k concepts is the main bottleneck).
        edge_counts = []
        for c in concepts:
            cid = self.network._resolve(c) or c
            out_count = len(self.network._edge_index.get(cid, []))
            in_count = len(self.network._reverse_index.get(cid, []))
            edge_counts.append((out_count + in_count, c))

        # Sort by edge count, take top landmarks.
        # Limited to 500 to keep the SVD fast — SVD is O(n³), and on
        # systems with suboptimal BLAS, 2000x2000 SVD can take 30+ seconds.
        # 500 landmarks gives a 500³ SVD (~64x faster) while still
        # providing good coverage of the graph's structure.
        edge_counts.sort(reverse=True)
        n_landmarks = min(500, n)
        landmarks = [c for _, c in edge_counts[:n_landmarks]]
        landmark_set = set(landmarks)
        landmark_idx = {c: i for i, c in enumerate(landmarks)}
        return landmarks, landmark_set, landmark_idx, n_landmarks

    def _compute_landmark_embedding(
        self,
        landmarks: list[str],
        landmark_set: set[str],
        landmark_idx: dict[str, int],
        n_landmarks: int,
    ) -> np.ndarray | None:
        """Build landmark adjacency and compute its spectral embedding."""
        # Build landmark adjacency matrix using edge index directly
        adj_lm = np.zeros((n_landmarks, n_landmarks), dtype=np.float32)
        for c in landmarks:
            i = landmark_idx[c]
            cid = self.network._resolve(c) or c
            for edge in self.network._edge_index.get(cid, []):
                if edge.target in landmark_set:
                    j = landmark_idx[edge.target]
                    w = float(edge.weight)
                    adj_lm[i, j] = max(adj_lm[i, j], w)
                    adj_lm[j, i] = max(adj_lm[j, i], w)

        # Spectral embedding of landmark subgraph
        degrees = adj_lm.sum(axis=1)
        degrees[degrees < 1e-8] = 1.0
        d_inv_sqrt = 1.0 / np.sqrt(degrees)
        normalized = adj_lm * d_inv_sqrt[:, np.newaxis] * d_inv_sqrt[np.newaxis, :]

        dim = min(self.SPECTRAL_DIM, n_landmarks - 2)  # -2: skip trivial + bounds
        if dim < 1:
            dim = 1
        try:
            u, s, _vt = np.linalg.svd(normalized, full_matrices=False)
            # Skip first singular vector (trivial/constant), same as dense path
            return u[:, 1 : dim + 1] * s[1 : dim + 1]
        except np.linalg.LinAlgError:
            return None

    def _extend_nystrom_embedding(
        self,
        concepts: list[str],
        concept_to_idx: dict[str, int],
        landmarks: list[str],
        landmark_set: set[str],
        landmark_idx: dict[str, int],
        n_landmarks: int,
        landmark_embedding: np.ndarray,
        dim: int,
        n: int,
    ) -> np.ndarray:
        """Extend landmark embedding to all nodes via Nyström approximation."""
        embedding = np.zeros((n, dim), dtype=np.float32)

        # Fill in landmark embeddings
        for c in landmarks:
            i = concept_to_idx[c]
            j = landmark_idx[c]
            embedding[i] = landmark_embedding[j]

        # For non-landmark nodes, compute similarity to landmarks
        # and project. Use edge index directly for speed.
        for c in concepts:
            i = concept_to_idx[c]
            if c in landmark_set:
                continue

            cid = self.network._resolve(c) or c
            sim = np.zeros(n_landmarks, dtype=np.float32)
            for edge in self.network._edge_index.get(cid, []):
                if edge.target in landmark_set:
                    j = landmark_idx[edge.target]
                    sim[j] = max(sim[j], float(edge.weight))
            for edge in self.network._reverse_index.get(cid, []):
                if edge.source in landmark_set:
                    j = landmark_idx[edge.source]
                    sim[j] = max(sim[j], float(edge.weight))

            if sim.sum() > 0:
                # L2-normalize the similarity vector and project onto
                # the landmark embedding. This is an approximation of
                # the Nyström extension — the full formula includes
                # D^{-1/2} degree normalization on both the node and
                # landmark sides, which we omit here. The spectral
                # component is only 32 of ~210 total dimensions, so
                # the approximation error has limited impact on the
                # combined concept vector.
                sim = sim / (np.linalg.norm(sim) + 1e-8)
                embedding[i] = sim @ landmark_embedding

        return embedding

    # ─── Experiential embedding ──────────────────────────────────

    def _build_experiential_embedding(
        self, concepts: list[str]
    ) -> np.ndarray | None:
        """Build experiential embedding from the holographic graph.

        The holographic graph stores sleep-discovered associations —
        cross-domain connections found during N3 sleep compression
        and LTM episode replay. These are experiential associations
        that neither the graph structure (spectral) nor the text
        (TF-IDF/GloVe) can capture.

        This method:
        1. Extracts the feature matrix from the holographic graph
           (n_relations × n_buckets features per concept)
        2. Normalizes and truncates to EXPERIENTIAL_DIM dimensions
        3. Returns the experiential embedding

        Concepts not in the holographic graph get zero vectors —
        they haven't been through sleep compression yet.
        """
        if self._holographic_graph is None:
            return None

        try:
            hgraph_concepts, features = (
                self._holographic_graph.extract_association_matrix()
            )
        except Exception:  # noqa: BLE001
            return None

        if not hgraph_concepts or features.shape[1] == 0:
            return None

        n_total = len(concepts)
        concept_set = set(concepts)
        hgraph_set = set(hgraph_concepts)

        # Filter out code concepts from the experiential layer.
        # The holographic graph contains code concepts (python:exceptions,
        # module:function, etc.) from autonomous code learning. These are
        # code structure, not experiential semantic associations — including
        # them pollutes the latent space and pulls attention toward code
        # instead of the conversation topic.
        def _is_clean_concept(cid: str) -> bool:
            base = cid.split("#")[0]
            if ":" in base and not base.startswith("http"):
                return False
            if "." in base:
                return False
            if "__" in base:
                return False
            return True

        # Map holographic graph concepts to the full concept list
        shared = [
            c for c in hgraph_concepts
            if c in concept_set and _is_clean_concept(c)
        ]
        if len(shared) < 2:
            return None

        # Build index maps
        hgraph_to_feat = {c: i for i, c in enumerate(hgraph_concepts)}
        shared_feat_idx = [hgraph_to_feat[c] for c in shared]

        # Extract the shared feature submatrix
        sub_features = features[shared_feat_idx]

        # Normalize rows
        norms = np.linalg.norm(sub_features, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        sub_features = sub_features / norms

        # Truncate or pad to EXPERIENTIAL_DIM
        dim = self.EXPERIENTIAL_DIM
        n_features = sub_features.shape[1]
        if n_features >= dim:
            shared_embedding = sub_features[:, :dim]
        else:
            # Pad with zeros if fewer features than dim
            shared_embedding = np.zeros(
                (len(shared), dim), dtype=np.float32
            )
            shared_embedding[:, :n_features] = sub_features

        # Map back to the full concept list
        embedding = np.zeros((n_total, dim), dtype=np.float32)
        shared_to_full = {c: i for i, c in enumerate(concepts) if c in hgraph_set}
        for j, c in enumerate(shared):
            full_idx = shared_to_full.get(c)
            if full_idx is not None:
                embedding[full_idx] = shared_embedding[j]

        return embedding

    # ─── TF-IDF embedding ───────────────────────────────────────

    def _build_tfidf_embedding(self, concepts: list[str]) -> np.ndarray | None:
        """Build TF-IDF embedding from concept definitions and names.

        Each concept is represented as a bag-of-words vector over its
        name and definition, weighted by inverse document frequency.
        This captures lexical similarity — concepts that share words
        in their definitions are close in this space.

        The vocabulary is selected to maximize matching utility: words
        that appear in a moderate number of concepts (not too rare,
        not too common) are the most useful for semantic matching.
        Words that appear in only 1-2 concepts can't match input text,
        and words that appear in almost every concept are uninformative.
        """
        # Collect documents (name + definition for each concept)
        documents: list[list[str]] = []
        for c in concepts:
            tokens = self._extract_text_tokens(c)
            documents.append(tokens)

        # Build document frequency
        n_docs = len(documents)
        doc_freq: dict[str, int] = {}
        for tokens in documents:
            seen = set(tokens)
            for token in seen:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        # Select vocabulary: words that appear in at least 2 documents
        # (so they can match between concepts) but not in more than 5%
        # of documents (so they're still informative). The old 50% cap
        # let words like "that" dominate and create spurious similarity.
        min_freq = 2
        max_freq = max(n_docs // 20, 10)
        useful_words = {
            word: freq for word, freq in doc_freq.items() if min_freq <= freq <= max_freq
        }

        # Compute IDF for selected words
        idf_scores = {
            word: math.log(n_docs / (freq + 1)) + 1.0 for word, freq in useful_words.items()
        }

        # Sort by combined score: freq * idf — favors words that appear
        # in many concepts but are still distinctive
        combined_scores = {word: useful_words[word] * idf_scores[word] for word in useful_words}

        sorted_vocab = sorted(combined_scores.items(), key=lambda x: -x[1])
        vocab_size = min(self.TFIDF_DIM, len(sorted_vocab))
        self._tfidf_vocab = {word: i for i, (word, _) in enumerate(sorted_vocab[:vocab_size])}
        self._idf = np.array(
            [idf_scores[word] for word, _ in sorted_vocab[:vocab_size]],
            dtype=np.float32,
        )

        # Build TF-IDF matrix
        matrix = np.zeros((n_docs, vocab_size), dtype=np.float32)
        for i, tokens in enumerate(documents):
            for token in tokens:
                j = self._tfidf_vocab.get(token)
                if j is not None:
                    matrix[i, j] += 1.0

        # Apply IDF weighting
        matrix *= self._idf[np.newaxis, :]

        # L2 normalize rows
        return self._normalize_rows(matrix)


    def _extract_text_tokens(self, concept_name: str) -> list[str]:
        """Extract text tokens from a concept's name and definition."""
        tokens: list[str] = []

        def _maybe_add(token: str) -> None:
            """Add a token to the list, filtering stopwords and short tokens."""
            cleaned = token.strip(".,;:!?()\"'`")
            if not cleaned or cleaned in self._STOPWORDS or len(cleaned) < 3:
                return
            tokens.append(cleaned)

        # Tokenize the concept name
        name = concept_name.replace("_", " ").replace("-", " ")
        for w in name.lower().split():
            _maybe_add(w)

        # Get the concept's definition. Use _concepts directly
        # instead of get_concept to avoid recalling from the
        # archive as a side effect — this runs during embedding
        # builds and refreshes, which should not disturb working
        # memory. The concept_name comes from the concepts list
        # which was built from _concepts.keys(), so it's already
        # a canonical ID in working memory.
        concept = self.network._concepts.get(concept_name)
        if concept:
            definition = concept.properties.get("definition", "")
            if definition and definition != "NO DEF":
                # Tokenize the definition
                for w in definition.lower().split():
                    _maybe_add(w)

        return tokens

    # ─── GloVe embedding ────────────────────────────────────────

    def _build_glove_embedding(self, concepts: list[str]) -> np.ndarray | None:
        """Build GloVe-based concept embeddings (if available).

        Multi-word concepts are the average of their constituent word
        vectors. Returns None if GloVe is not loaded.
        """
        if self._word_vectors is None:
            return None

        n = len(concepts)
        dim = self._word_vectors.shape[1]
        matrix = np.zeros((n, dim), dtype=np.float32)

        for i, concept_name in enumerate(concepts):
            vec = self._compute_glove_vector(concept_name)
            if vec is not None:
                matrix[i] = vec

        return matrix

    def _compute_glove_vector(self, concept_name: str) -> np.ndarray | None:
        """Compute GloVe vector for a concept by averaging word vectors."""
        if self._word_vectors is None:
            return None

        tokens = self._tokenize(concept_name)
        vecs = [self._word_vectors[self._word_to_idx[t]] for t in tokens if t in self._word_to_idx]
        if not vecs:
            return None
        return np.mean(vecs, axis=0).astype(np.float32)

    # ─── Combined vector computation ────────────────────────────

    def _compute_concept_vector(self, concept_name: str) -> np.ndarray | None:
        """Compute a vector for a concept not in the precomputed index.

        Combines all available embedding sources.
        """
        components: list[np.ndarray] = []

        # TF-IDF component
        if self._idf is not None and self._tfidf_vocab:
            tokens = self._extract_text_tokens(concept_name)
            vec = np.zeros(len(self._tfidf_vocab), dtype=np.float32)
            for token in tokens:
                j = self._tfidf_vocab.get(token)
                if j is not None:
                    vec[j] += 1.0
            vec *= self._idf
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                components.append(vec / norm)

        # GloVe component
        glove_vec = self._compute_glove_vector(concept_name)
        if glove_vec is not None:
            norm = np.linalg.norm(glove_vec)
            if norm > 1e-8:
                components.append(glove_vec / norm)

        if not components:
            return None

        # Re-normalize the concatenated vector: each component was
        # individually L2-normalized, so the concatenation of N
        # unit vectors has norm √N, not 1. The precomputed concept
        # matrix is normalized after concatenation (_normalize_rows),
        # so ad-hoc vectors must be normalized too for cosine
        # similarity comparisons to be valid.
        return self._normalize(np.concatenate(components).astype(np.float32))

    def _compute_text_vector(self, text: str) -> np.ndarray | None:
        """Compute a vector for arbitrary text.

        Uses TF-IDF (matching against concept vocabulary) and GloVe
        (if available) to create a vector that can be compared to
        concept vectors.
        """
        components: list[np.ndarray] = []

        # TF-IDF component
        if self._idf is not None and self._tfidf_vocab:
            tokens = [
                w.strip(".,;:!?()\"'`").lower()
                for w in text.split()
                if w.strip(".,;:!?()\"'`")
            ]
            vec = np.zeros(len(self._tfidf_vocab), dtype=np.float32)
            for token in tokens:
                j = self._tfidf_vocab.get(token)
                if j is not None:
                    vec[j] += 1.0
            vec *= self._idf
            norm: np.floating[Any] = np.linalg.norm(vec)
            if norm > 1e-8:
                components.append(vec / norm)

        # GloVe component — filter stopwords and short tokens so
        # function words ("the", "in", "my") don't dilute the
        # content-word semantics in the averaged vector.
        if self._word_vectors is not None:
            tokens = [
                w.strip(".,;:!?()\"'`").lower()
                for w in text.split()
                if w.strip(".,;:!?()\"'`")
            ]
            vecs = [
                self._word_vectors[self._word_to_idx[t]]
                for t in tokens
                if t in self._word_to_idx
                and t not in self._STOPWORDS
                and len(t) >= 3
            ]
            if vecs:
                avg = np.mean(vecs, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 1e-8:
                    components.append(avg / norm)

        if not components:
            return None

        # Re-normalize after concatenation (see _compute_concept_vector).
        return self._normalize(np.concatenate(components).astype(np.float32))

    # ─── Search ─────────────────────────────────────────────────

    def _search_concept_matrix(
        self,
        query: np.ndarray,
        k: int,
        threshold: float,
        exclude: str | None = None,
        query_has_spectral: bool = True,
    ) -> list[tuple[str, float]]:
        """Find the k most similar concepts to a query vector.

        Uses vectorized dot product against the precomputed concept
        matrix. O(N) where N is the number of concepts with vectors.

        When query_has_spectral is False (text queries), the spectral
        component of the concept matrix is zeroed out so the comparison
        is only over TF-IDF and GloVe dimensions that the query has.
        """
        if self._concept_matrix is None:
            return []

        # Normalize the query
        normalized_query = self._normalize(query)
        if normalized_query is None:
            return []

        aligned = self._align_search_query(normalized_query, query_has_spectral)
        if aligned is None:
            return []
        query, matrix, non_spectral_search = aligned

        # Vectorized cosine similarity
        similarities = matrix @ query

        # For non-spectral (text) searches, divide by the precomputed
        # non-spectral norms to obtain cosine similarities over only the
        # TF-IDF and GloVe dimensions, without copying the matrix.
        if non_spectral_search and self._non_spectral_norms is not None:
            norms = self._non_spectral_norms
            safe = norms > 1e-8
            similarities = np.where(safe, similarities / np.where(safe, norms, 1.0), 0.0)

        results = self._collect_search_results(similarities, threshold, k, exclude)
        if len(results) < k:
            # Hot search came up short — fall back to the cold layer
            # (archived concepts). Cold vectors only carry the flat
            # block, so the comparison is flat cosine regardless of
            # whether the hot query used hybrid scoring.
            results.extend(
                self._search_archive_flat(
                    query, k - len(results), threshold,
                    exclude_names={exclude, *(n for n, _ in results)},
                )
            )
        return results

    def _align_search_query(
        self,
        query: np.ndarray,
        query_has_spectral: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool] | None:
        """Align query dimensions to the concept matrix.

        Returns (query, matrix, non_spectral_search) or None if the
        re-normalized query is a zero vector.
        """
        if self._concept_matrix is None:
            return None

        non_spectral_search = False

        # Handle dimension mismatch
        if query.shape[0] != self._concept_matrix.shape[1]:
            if not query_has_spectral and (
                self._spectral_dim > 0 or self._experiential_dim > 0
            ):
                # Text query: build a full-size vector with zeros for
                # the spectral AND experiential components, and the
                # query values for the TF-IDF and GloVe components
                full_query = np.zeros(self._dim, dtype=np.float32)
                # Place the query's TF-IDF component
                tfidf_end = min(self._tfidf_dim, query.shape[0])
                full_query[self._tfidf_offset : self._tfidf_offset + tfidf_end] = query[:tfidf_end]
                # Place the query's GloVe component (if present)
                if query.shape[0] > tfidf_end and self._glove_dim > 0:
                    glove_end = min(self._glove_dim, query.shape[0] - tfidf_end)
                    full_query[self._glove_offset : self._glove_offset + glove_end] = query[
                        tfidf_end : tfidf_end + glove_end
                    ]
                # Re-normalize
                normalized = self._normalize(full_query)
                if normalized is None:
                    return None
                query = normalized

                # The query has zeros in the spectral and experiential
                # regions, so the dot product with the original concept
                # matrix naturally excludes those contributions. We only
                # need to account for the fact that the concept rows have
                # non-unit norm in their non-spectral portion (the spectral
                # and experiential portions contributed to their overall
                # norm at build time). We divide by the precomputed
                # non-spectral norms instead of copying and re-normalizing
                # the full matrix.
                matrix = self._concept_matrix
                non_spectral_search = True
            else:
                # Pad or truncate to match
                if query.shape[0] < self._concept_matrix.shape[1]:
                    padded = np.zeros(self._concept_matrix.shape[1], dtype=np.float32)
                    padded[: query.shape[0]] = query
                    query = padded
                else:
                    query = query[: self._concept_matrix.shape[1]]
                matrix = self._concept_matrix
        else:
            matrix = self._concept_matrix

        return query, matrix, non_spectral_search

    def _collect_search_results(
        self,
        similarities: np.ndarray,
        threshold: float,
        k: int,
        exclude: str | None,
    ) -> list[tuple[str, float]]:
        """Filter, sort, and build the search result list."""
        # Filter by threshold
        above = np.where(similarities >= threshold)[0]
        if len(above) == 0:
            return []

        # Sort by similarity (descending)
        sorted_idx = above[np.argsort(-similarities[above])]

        # Build results, excluding the query concept
        results: list[tuple[str, float]] = []
        for idx in sorted_idx:
            name = self._concept_names[idx]
            if name != exclude:
                results.append((name, float(similarities[idx])))
            if len(results) >= k:
                break

        return results

    # ─── Cold layer (archived concepts) ─────────────────────────
    #
    # The hot matrix only covers working memory. Concepts spilled to
    # the SQLite archive get a parallel "cold" matrix holding just the
    # flat block (TF-IDF + GloVe) — they have no live graph position,
    # so spectral and experiential components would be meaningless.
    # Searches consult the cold layer only when the hot layer returns
    # fewer than k results, keeping dormant knowledge discoverable by
    # similarity without slowing the hot path or disturbing working
    # memory (no recalls happen here — the matrix is built straight
    # from the archive's serialized dicts).

    def _build_archive_matrix(self) -> None:
        """Build the cold matrix for archived concepts.

        Reads ``archive.get_all_concepts()`` (id → serialized dict) and
        computes a flat-block vector per concept from its name,
        aliases, and definition. Concepts currently in working memory
        (recalled between archive writes and this build) are skipped —
        their hot row wins.
        """
        self._archive_matrix = None
        self._archive_names = []
        self._archive_to_idx = {}

        archive = getattr(self.network, "_archive", None)
        if archive is None or self._dim == 0:
            return
        flat_start = self._spectral_dim + self._experiential_dim
        flat_dim = self._dim - flat_start
        if flat_dim <= 0:
            return
        try:
            archived = archive.get_all_concepts()
        except (sqlite3.Error, RuntimeError) as e:
            logger.debug(f"archive listing for cold matrix failed: {e}")
            return
        items = [
            (cid, data) for cid, data in archived
            if cid not in self._concept_to_idx
        ]
        if not items:
            return

        tfidf_pos = self._tfidf_offset - flat_start
        glove_pos = self._glove_offset - flat_start
        matrix = np.zeros((len(items), flat_dim), dtype=np.float32)
        names: list[str] = []
        for cid, data in items:
            row = np.zeros(flat_dim, dtype=np.float32)

            # TF-IDF block over name + aliases + definition
            if (
                self._idf is not None and self._tfidf_vocab
                and self._tfidf_dim > 0
            ):
                vec = np.zeros(self._tfidf_dim, dtype=np.float32)
                for token in self._archived_text_tokens(cid, data):
                    j = self._tfidf_vocab.get(token)
                    if j is not None:
                        vec[j] += 1.0
                vec *= self._idf
                row[tfidf_pos:tfidf_pos + self._tfidf_dim] = vec

            # GloVe block — average over name and alias word vectors
            if self._glove_dim > 0:
                texts = [cid, *(str(a) for a in (data.get("aliases") or []))]
                vecs = [
                    v for t in texts
                    if (v := self._compute_glove_vector(t)) is not None
                ]
                if vecs:
                    row[glove_pos:glove_pos + self._glove_dim] = np.mean(
                        vecs, axis=0
                    )

            norm = np.linalg.norm(row)
            if norm < 1e-8:
                continue
            matrix[len(names)] = row / norm
            names.append(cid)

        if not names:
            return
        self._archive_matrix = matrix[: len(names)]
        self._archive_names = names
        self._archive_to_idx = {n: i for i, n in enumerate(names)}
        logger.debug(
            "Cold layer built: %d archived concepts embedded", len(names)
        )

    def _archived_text_tokens(
        self, concept_id: str, data: dict[str, Any]
    ) -> list[str]:
        """Extract filtered tokens from an archived concept dict.

        Mirrors ``_extract_text_tokens`` but reads the serialized
        archive dict (id, aliases, properties.definition) instead of a
        live ``Concept`` — no working-memory access, no recall.
        """
        texts = [concept_id.replace("_", " ").replace("-", " ")]
        for alias in data.get("aliases") or []:
            texts.append(str(alias).replace("_", " ").replace("-", " "))
        definition = str((data.get("properties") or {}).get("definition", ""))
        if definition and definition != "NO DEF":
            texts.append(definition)

        tokens: list[str] = []
        for text in texts:
            for w in text.lower().split():
                cleaned = w.strip(".,;:!?()\"'`")
                if (
                    cleaned
                    and cleaned not in self._STOPWORDS
                    and len(cleaned) >= 3
                ):
                    tokens.append(cleaned)
        return tokens

    def _archive_flat_sims(self, query_vec: np.ndarray) -> np.ndarray | None:
        """Flat-block cosine similarities of a query against the cold layer.

        The query is expected at full width (``self._dim``); its flat
        block is sliced out, normalized, and dotted against the
        normalized cold rows. Returns None if the cold layer is empty
        or the query's flat block is ~zero.
        """
        if self._archive_matrix is None or self._dim == 0:
            return None
        flat_start = self._spectral_dim + self._experiential_dim
        if query_vec.shape[0] != self._dim:
            # Dimension mismatch — pad/truncate to full width using the
            # same convention as _align_search_query.
            full = np.zeros(self._dim, dtype=np.float32)
            n = min(query_vec.shape[0], self._dim)
            full[:n] = query_vec[:n]
            query_vec = full
        q = query_vec[flat_start:]
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return None
        return (self._archive_matrix @ (q / norm)).astype(np.float32)

    def _collect_archive_results(
        self,
        sims: np.ndarray,
        k: int,
        threshold: float,
        exclude_names: set[str | None],
    ) -> list[tuple[str, float]]:
        """Filter/sort cold-layer similarities into result pairs."""
        above = np.where(sims >= threshold)[0]
        if len(above) == 0:
            return []
        sorted_idx = above[np.argsort(-sims[above])]
        results: list[tuple[str, float]] = []
        for idx in sorted_idx:
            name = self._archive_names[idx]
            # Skip excluded names and anything that has been recalled
            # to working memory since the cold matrix was built — its
            # hot row already had a chance to rank.
            if name in exclude_names or name in self._concept_to_idx:
                continue
            results.append((name, float(sims[idx])))
            if len(results) >= k:
                break
        return results

    def _search_archive_flat(
        self,
        query_vec: np.ndarray,
        k: int,
        threshold: float,
        exclude_names: set[str | None],
    ) -> list[tuple[str, float]]:
        """Search the cold layer by flat-block cosine similarity."""
        sims = self._archive_flat_sims(query_vec)
        if sims is None:
            return []
        return self._collect_archive_results(sims, k, threshold, exclude_names)

    def _search_archive_vesica(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        k: int,
        threshold: float,
        exclude_names: set[str | None],
    ) -> list[tuple[str, float]]:
        """Cold-layer vesica: cold concepts close to BOTH queries."""
        sims_a = self._archive_flat_sims(vec_a)
        sims_b = self._archive_flat_sims(vec_b)
        if sims_a is None or sims_b is None:
            return []
        combined = np.minimum(sims_a, sims_b)
        return self._collect_archive_results(
            combined, k, threshold, exclude_names
        )

    # ─── Utilities ──────────────────────────────────────────────

    @staticmethod
    def _tokenize(concept_name: str) -> list[str]:
        """Tokenize a concept name into lowercase words."""
        name = concept_name.replace("_", " ").replace("-", " ")
        return [w for w in name.split() if w]

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray | None:
        """L2-normalize a vector. Returns None for zero vectors."""
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return None
        return vec / norm

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize each row of a matrix."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return matrix / norms

    def _compute_non_spectral_norms(self, matrix: np.ndarray) -> np.ndarray:
        """L2 norms of each row with spectral AND experiential zeroed out.

        Precomputed once at build time so that non-spectral (text)
        searches can avoid copying and re-normalizing the full concept
        matrix. Text queries only have TF-IDF + GloVe components, so
        the spectral and experiential portions must be excluded from
        the norm. When there is no spectral/experiential component,
        the norms are just the per-row L2 norms of the matrix.
        """
        if self._spectral_dim <= 0 and self._experiential_dim <= 0:
            return np.linalg.norm(matrix, axis=1)
        non_spectral = matrix.copy()
        if self._spectral_dim > 0:
            s = self._spectral_offset
            non_spectral[:, s : s + self._spectral_dim] = 0
        if self._experiential_dim > 0:
            e = self._experiential_offset
            non_spectral[:, e : e + self._experiential_dim] = 0
        return np.linalg.norm(non_spectral, axis=1)

    # ─── Dimensionality management ────────────────────────────────

    @staticmethod
    def _pad_or_truncate_matrix(matrix: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad or truncate a matrix to the target column dimension.

        When padding, zeros are appended to the right. When truncating,
        columns are removed from the right. This preserves the
        component offsets (spectral, TF-IDF, GloVe) since they're at
        the beginning of the vector.

        Args:
            matrix: The matrix to resize (N × D).
            target_dim: The desired column dimension.

        Returns:
            A matrix of shape (N × target_dim).
        """
        current_dim = matrix.shape[1]
        if current_dim == target_dim:
            return matrix
        elif current_dim < target_dim:
            # Pad with zeros
            padding = np.zeros(
                (matrix.shape[0], target_dim - current_dim),
                dtype=matrix.dtype,
            )
            return np.concatenate([matrix, padding], axis=1)
        else:
            # Truncate
            return matrix[:, :target_dim]

    def resize(self, new_dim: int) -> None:
        """Resize the embedding space to a new dimensionality.

        Pads with zeros if new_dim > current dim, or truncates if
        new_dim < current dim. After resizing, the concept matrix is
        re-normalized.

        Args:
            new_dim: The new target dimensionality.
        """
        self._ensure_loaded()
        self._target_dim = new_dim

        if self._concept_matrix is not None:
            self._concept_matrix = self._pad_or_truncate_matrix(self._concept_matrix, new_dim)
            self._concept_matrix = self._normalize_rows(self._concept_matrix)
            self._dim = self._concept_matrix.shape[1]
            self._non_spectral_norms = self._compute_non_spectral_norms(self._concept_matrix)
            # Clear the ad-hoc cache since dimensions changed
            self._concept_cache.clear()

    # ─── Systematic compositionality ──────────────────────────────
    #
    # Tensor product representations (TPR) for concept combination.
    # Following Smolensky (1990) and Plate (1995), the tensor product
    # of a filler (concept) vector and a role vector creates a
    # structured representation that preserves the binding between
    # the concept and its role. This enables systematic composition:
    # "red car" = compose(car, red, role=modifier).
    #
    # The tensor product of two vectors a (dim D) and b (dim D) is
    # the outer product a ⊗ b, which has dimension D². To keep
    # computation tractable, we use a circular convolution (Plate,
    # 1995) which reduces the result back to dimension D while
    # approximately preserving the binding information.
    #
    # References:
    # - Smolensky, P. (1990). Tensor product variable binding and
    #   the representation of symbolic structures in connectionist
    #   systems. *Artificial Intelligence*, 46(1-2), 159–216.
    # - Plate, T. A. (1995). Holographic reduced representations.
    #   *IEEE Transactions on Neural Networks*, 6(3), 623–641.

    def compose_concepts(
        self,
        concept_a: str,
        concept_b: str,
        relation: str = "modifier",
    ) -> np.ndarray | None:
        """Combine two concept embeddings using tensor product composition.

        Uses circular convolution (Plate, 1995) to bind the two
        concepts into a single vector of the same dimensionality.
        This enables the system to build representations for phrases
        like "red car" or "fast runner" from individual concept vectors.

        The relation parameter determines the role binding:
        - "modifier": concept_b modifies concept_a (e.g. "red car")
        - "relation": concept_a is in relation to concept_b
        - "compound": both concepts combine equally

        Args:
            concept_a: The primary concept (e.g. "car").
            concept_b: The secondary concept (e.g. "red").
            relation: The binding relation — "modifier", "relation",
                or "compound".

        Returns:
            A composed embedding vector, or None if either concept
            has no embedding.
        """
        self._ensure_loaded()

        vec_a = self.get_concept_vector(concept_a)
        vec_b = self.get_concept_vector(concept_b)

        if vec_a is None or vec_b is None:
            return None

        # Ensure both vectors have the same dimensionality
        dim = max(vec_a.shape[0], vec_b.shape[0])
        va = self._pad_or_truncate_vector(vec_a, dim)
        vb = self._pad_or_truncate_vector(vec_b, dim)

        if relation == "modifier":
            # b modifies a: use circular convolution of a and b
            # This binds the modifier (b) to the head (a) — the
            # result is order-sensitive and preserves structure.
            composed = self._circular_convolution(va, vb)
        elif relation == "relation":
            # a is related to b: superposition (weighted sum). This is
            # NOT compositional — the result is order-invariant and
            # the binding between concept and role is lost. Use this
            # when you want a joint representation of two concepts
            # without structural distinction, not for phrase-level
            # composition.
            composed = 0.6 * va + 0.4 * vb
        elif relation == "compound":
            # Equal superposition — same caveat as "relation": the
            # result is order-invariant and non-compositional. Both
            # concepts contribute equally but the structure is lost.
            composed = 0.5 * va + 0.5 * vb
        else:
            # Default: circular convolution (compositional binding)
            composed = self._circular_convolution(va, vb)

        # Normalize the result
        norm = np.linalg.norm(composed)
        if norm > 1e-8:
            composed = composed / norm

        return composed.astype(np.float32)

    def compose_with_role(
        self,
        concept: str,
        role: str,
    ) -> np.ndarray | None:
        """Compose a concept embedding with a role vector.

        Role-based composition binds a concept to a structural role
        (subject, object, modifier, etc.). The role vector is derived
        from a hash of the role name, ensuring different roles produce
        different bindings.

        This is the building block for structured composition: to
        represent "the red car", you compose(car, role="head") and
        compose(red, role="modifier"), then combine the results.

        Args:
            concept: The concept to bind.
            role: The role name (e.g. "subject", "object",
                "modifier", "head").

        Returns:
            A role-bound embedding vector, or None if the concept
            has no embedding.
        """
        self._ensure_loaded()

        vec = self.get_concept_vector(concept)
        if vec is None:
            return None

        # Generate a deterministic role vector from the role name
        role_vec = self._role_vector(role, vec.shape[0])

        # Bind concept to role via circular convolution
        composed = self._circular_convolution(vec, role_vec)

        # Normalize
        norm = np.linalg.norm(composed)
        if norm > 1e-8:
            composed = composed / norm

        return composed.astype(np.float32)

    def compose_recursive(
        self,
        vectors: list[np.ndarray],
        relation: str = "compound",
    ) -> np.ndarray | None:
        """Recursively compose multiple vectors.

        Allows composing already-composed representations, enabling
        hierarchical structure: compose(compose(red, car), fast) for
        "fast red car".

        Args:
            vectors: A list of vectors to compose (left to right).
            relation: The composition relation.

        Returns:
            A composed vector, or None if the list is empty.
        """
        if not vectors:
            return None

        if len(vectors) == 1:
            return vectors[0].copy()

        result = vectors[0]
        for vec in vectors[1:]:
            # Ensure same dimensionality
            dim = max(result.shape[0], vec.shape[0])
            ra = self._pad_or_truncate_vector(result, dim)
            rb = self._pad_or_truncate_vector(vec, dim)

            if relation == "modifier":
                result = self._circular_convolution(ra, rb)
            elif relation == "compound":
                result = 0.5 * ra + 0.5 * rb
            else:
                result = self._circular_convolution(ra, rb)

        # Normalize
        norm = np.linalg.norm(result)
        if norm > 1e-8:
            result = result / norm

        return result.astype(np.float32)

    # ─── Compositionality helpers ─────────────────────────────────

    @staticmethod
    def _circular_convolution(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular convolution of two vectors (Plate, 1995).

        The circular convolution of a and b produces a vector c where:
            c[k] = sum_j a[j] * b[(k - j) mod n]

        This is the binding operation in Holographic Reduced
        Representations (HRR). It binds two vectors into a single
        vector of the same dimensionality, approximately preserving
        the information from both.

        Computed efficiently via FFT: conv(a, b) = IFFT(FFT(a) * FFT(b)).
        """
        # Use FFT for efficient computation
        fa = np.fft.fft(a)
        fb = np.fft.fft(b)
        conv = np.fft.ifft(fa * fb).real
        return conv.astype(np.float32)

    @staticmethod
    def _pad_or_truncate_vector(vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad or truncate a single vector to the target dimension."""
        current = vec.shape[0]
        if current == target_dim:
            return vec
        elif current < target_dim:
            padding = np.zeros(target_dim - current, dtype=vec.dtype)
            return np.concatenate([vec, padding])
        else:
            return vec[:target_dim]

    @staticmethod
    def _role_vector(role_name: str, dim: int) -> np.ndarray:
        """Generate a deterministic unit vector for a role name.

        Uses a hash of the role name to seed a random vector, ensuring
        the same role always produces the same vector. Different roles
        produce approximately orthogonal vectors (high-dimensional
        random vectors are nearly orthogonal).
        """
        import hashlib

        # Hash the role name to get a deterministic seed
        hash_bytes = hashlib.md5(role_name.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], "little")

        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec
