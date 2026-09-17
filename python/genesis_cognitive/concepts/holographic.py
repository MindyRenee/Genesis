"""Holographic associative graph — fixed-size associative memory for Genesis.

This module replaces the auto-generated bridge edges (semantic_bridge,
hub_attachment, associative_bridge) with a **fixed-size** holographic
associative memory based on Holographic Reduced Representations (HRRs)
/ Vector Symbolic Architectures (VSAs).

# The problem

The concept network's edge list grows monotonically. 70% of edges are
auto-generated bridges — ``semantic_bridge``, ``hub_attachment``,
``associative_bridge`` — that materialize fuzzy similarity into explicit
graph edges. These edges are noisy, regenerate during every sleep cycle,
and dominate the serialized state size (11+ MB of JSON).

# The solution

Holographic Reduced Representations store associations *implicitly* in
fixed-size high-dimensional vectors. The key operations:

- **Binding** (circular convolution ``⊗``): binds two vectors into a new
  vector representing their association. ``dog ⊗ chases`` produces a
  vector that IS the "dog-chases" association.

- **Unbinding** (circular correlation ``⊙``): recovers one element from
  the bound pair. ``(dog ⊗ chases) ⊙ chases ≈ dog``. It's approximate —
  there's noise — but the association is there.

- **Superposition** (vector addition ``+``): piles multiple bindings into
  the same vector. ``dog⊗chases + cat⊗runs + bird⊗flies`` all coexist in
  one fixed-size vector.

The critical property: **the vector is fixed-size regardless of how many
associations it holds**. 10 associations or 10,000 — same vector. The
tradeoff is signal-to-noise ratio degrades as you superpose more
bindings, but that's neuroscientifically correct — real associative
memory gets noisier as you load it.

# Implementation

We use **bucketed per-relation memory** to keep the signal-to-noise
ratio healthy:

- D = 2048 dimensions (fixed)
- B = 16 buckets per relation type
- Each concept gets a deterministic random unit vector (its "address")
- Each relation type gets a deterministic random unit vector (its "role")
- Adding edge (A, R, B): ``memory_R[hash(A) % B] += bind(addr_A, addr_B)``
- Querying (A, R): ``unbind(addr_A, memory_R[hash(A) % B])`` → compare
  to all concept addresses

With 49K bridge edges across 15 relation types and 16 buckets, each
bucket holds ~200 bindings. The SNR is sqrt(D/N) = sqrt(2048/200) ≈ 3.2,
which is sufficient for candidate generation.

Storage: 15 relations × 16 buckets × 2048 dims × 4 bytes = 1.9 MB **fixed**.

# Neuroscience grounding

This models the hippocampal associative memory system. The hippocampus
performs pattern completion — given a partial cue, it retrieves the full
association. The holographic graph does the same: given a source concept
and a relation, it retrieves the most likely target. The noise in the
retrieval mirrors the probabilistic nature of biological memory recall.

(Plate, 1995; Kanerva, 2009; Sutherland & McNaughton, 2000)
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────

# Dimensionality of the holographic vectors. Higher = better SNR but
# more storage. 2048 is a good balance: SNR ~3 with 200 bindings per
# bucket, total storage ~1.9 MB.
DEFAULT_DIM = 2048

# Number of buckets per relation type. More buckets = fewer bindings
# per bucket = better SNR, but more storage. 16 is sufficient for
# up to ~100K bridge edges.
DEFAULT_BUCKETS = 16


# ─── Vector operations ──────────────────────────────────────────

def _seeded_unit_vector(seed: int, dim: int) -> np.ndarray:
    """Generate a deterministic random unit vector from a seed.

    The same seed always produces the same vector, which is critical
    for consistency across save/load cycles.
    """
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        v[0] = 1.0
        norm = np.float32(1.0)
    return v / norm


def _hash_to_seed(key: str) -> int:
    """Hash a string key to a deterministic integer seed."""
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def circular_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution (binding operation).

    bind(a, b) = IFFT(FFT(a) * FFT(b))

    This is the HRR binding operation. The result is a new vector
    that represents the association of a and b.
    """
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    result = np.fft.irfft(fa * fb, n=a.shape[0])
    # Normalize to unit length
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result = result / norm
    return result.astype(np.float32)


def circular_correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """HRR unbinding operation (circular convolution with approximate inverse).

    unbind(c, b) = IFFT(FFT(c) * conj(FFT(b)))

    This recovers an approximation of the vector that was bound with
    ``b`` to produce ``c``. If ``c = bind(x, b)``, then
    ``unbind(c, b) ≈ x`` (plus noise from other superposed bindings).

    Note: this is NOT the standard mathematical circular correlation
    (which would be IFFT(conj(FFT(a)) * FFT(b))). HRR unbinding uses
    the convolution-with-inverse: IFFT(FFT(a) * conj(FFT(b))).
    """
    fa = np.fft.rfft(a)
    fb = np.fft.rfft(b)
    result = np.fft.irfft(fa * np.conj(fb), n=a.shape[0])
    return result.astype(np.float32)


# ─── Holographic graph ──────────────────────────────────────────

class HolographicGraph:
    """Fixed-size holographic associative memory for concept associations.

    Replaces auto-generated bridge edges with a fixed-size vector
    representation. The graph stores associations implicitly via
    circular convolution binding and superposition.

    Thread-safe after initialization. All vectors are read-only post-build.
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        n_buckets: int = DEFAULT_BUCKETS,
    ) -> None:
        """Initialize the holographic graph store."""
        self.dim = dim
        self.n_buckets = n_buckets

        # Per-relation bucketed memory: relation_name → (B, D) float32
        self._memory: dict[str, np.ndarray] = {}

        # Concept address vectors: concept_id → (D,) float32
        self._addresses: dict[str, np.ndarray] = {}

        # Role vectors: relation_name → (D,) float32
        self._roles: dict[str, np.ndarray] = {}

        # Track which concepts and relations have been registered
        self._registered_concepts: set[str] = set()
        self._registered_relations: set[str] = set()

        # Edge count for statistics
        self._edge_count: int = 0

        # LRU cache for query results. The graph is read-only after
        # build (sleep compression writes, then the mind reads), so
        # caching is safe. Each query involves FFT + dot products
        # against ~3000 concepts, so caching is critical for the
        # cognition hot path (get_neighbors is called frequently).
        # The cache is invalidated on any write (add/store/load).
        self._query_cache: OrderedDict[tuple, object] = OrderedDict()
        self._query_cache_max = 512

    @property
    def edge_count(self) -> int:
        """Number of associations stored in the holographic graph."""
        return self._edge_count

    @property
    def n_concepts(self) -> int:
        """Number of registered concepts."""
        return len(self._addresses)

    @property
    def n_relations(self) -> int:
        """Number of registered relation types."""
        return len(self._roles)

    @property
    def storage_bytes(self) -> int:
        """Total storage size in bytes."""
        total = 0
        for mem in self._memory.values():
            total += mem.nbytes
        for addr in self._addresses.values():
            total += addr.nbytes
        for role in self._roles.values():
            total += role.nbytes
        return total

    def _invalidate_cache(self) -> None:
        """Clear the query cache. Called after any write to the graph."""
        self._query_cache.clear()

    def _cache_get(self, key: tuple) -> list | dict | None:
        """Get a value from the LRU cache, moving it to the end."""
        if key in self._query_cache:
            self._query_cache.move_to_end(key)
            val = self._query_cache[key]
            if isinstance(val, (list, dict)):
                return val
            return None
        return None

    def _cache_put(self, key: tuple, value: list | dict) -> None:
        """Put a value in the LRU cache, evicting the oldest if full."""
        self._query_cache[key] = value
        self._query_cache.move_to_end(key)
        if len(self._query_cache) > self._query_cache_max:
            self._query_cache.popitem(last=False)

    # ─── Registration ───────────────────────────────────────────

    def _get_address(self, concept_id: str) -> np.ndarray:
        """Get or create the address vector for a concept."""
        if concept_id not in self._addresses:
            self._addresses[concept_id] = _seeded_unit_vector(
                _hash_to_seed(concept_id), self.dim,
            )
            self._registered_concepts.add(concept_id)
        return self._addresses[concept_id]

    def _get_role(self, relation: str) -> np.ndarray:
        """Get or create the role vector for a relation type."""
        if relation not in self._roles:
            self._roles[relation] = _seeded_unit_vector(
                _hash_to_seed(f"role:{relation}"), self.dim,
            )
            self._registered_relations.add(relation)
            # Initialize memory for this relation
            self._memory[relation] = np.zeros(
                (self.n_buckets, self.dim), dtype=np.float32,
            )
        return self._roles[relation]

    def _bucket_index(self, concept_id: str) -> int:
        """Hash a concept ID to a bucket index."""
        return _hash_to_seed(concept_id) % self.n_buckets

    # ─── Core operations ────────────────────────────────────────

    def add(self, source: str, relation: str, target: str, weight: float = 1.0) -> None:
        """Add an association to the holographic graph.

        The association (source, relation, target) is stored via
        superposition of the bound role×target vector into the source's
        bucket in the relation's memory.

        Args:
            source: Source concept ID.
            relation: Relation type name (e.g. "related_to", "similar_to").
            target: Target concept ID.
            weight: Association strength (0-1). Scales the superposed
                vector. Weaker associations contribute less signal.
        """
        # Register the source address (needed for querying)
        self._get_address(source)
        addr_target = self._get_address(target)
        role = self._get_role(relation)

        # Bind role with target: this creates the "association signal"
        bound = circular_convolve(role, addr_target) * weight

        # Superpose into the source's bucket
        bucket = self._bucket_index(source)
        self._memory[relation][bucket] += bound
        self._edge_count += 1
        self._invalidate_cache()

    def query(
        self,
        source: str,
        relation: str,
        candidate_ids: list[str] | None = None,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Query the holographic graph for associations.

        Given a source concept and a relation type, retrieve the most
        likely target concepts. The result is approximate — the
        holographic graph returns candidates that should be validated
        against the embedding space or the explicit graph.

        Args:
            source: Source concept ID.
            relation: Relation type name.
            candidate_ids: Optional list of candidate concept IDs to
                score. If None, all registered concepts are scored.
            top_k: Maximum number of results.

        Returns:
            List of (concept_id, similarity) tuples, sorted by
            similarity descending. Similarity is the dot product
            between the unbound result and each candidate's address
            vector (higher = more likely association).
        """
        # Cache lookup — only cache when candidate_ids is None (the
        # common case from get_neighbors). Custom candidate lists
        # are rare and would pollute the cache.
        cache_key: tuple | None = None
        if candidate_ids is None:
            cache_key = (source, relation, top_k)
            cached = self._cache_get(cache_key)
            if isinstance(cached, list):
                return list(cached)

        if relation not in self._memory:
            return []
        if source not in self._addresses:
            return []

        role = self._get_role(relation)
        bucket = self._bucket_index(source)

        # Unbind: recover the target from the memory
        # memory[bucket] ≈ bind(role, target) + noise
        # unbind(role, memory[bucket]) ≈ target + noise
        result = circular_correlate(self._memory[relation][bucket], role)

        # Score against candidate addresses
        if candidate_ids is None:
            candidate_ids = list(self._addresses.keys())

        if not candidate_ids:
            return []

        # Exclude the source itself
        candidates = [c for c in candidate_ids if c != source]
        if not candidates:
            return []

        # Compute dot product with each candidate's address
        scores = []
        for cid in candidates:
            addr = self._addresses.get(cid)
            if addr is not None:
                sim = float(np.dot(result, addr))
                scores.append((cid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        result_list = scores[:top_k]

        if cache_key is not None:
            self._cache_put(cache_key, list(result_list))

        return result_list

    def query_all_relations(
        self,
        source: str,
        candidate_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> dict[str, list[tuple[str, float]]]:
        """Query all relation types for associations from a source.

        Returns a dict mapping relation name → list of (target, similarity).
        """
        # Cache lookup — only cache when candidate_ids is None
        cache_key: tuple | None = None
        if candidate_ids is None:
            cache_key = (source, "all", top_k)
            cached = self._cache_get(cache_key)
            if isinstance(cached, dict):
                return dict(cached)

        results: dict[str, list[tuple[str, float]]] = {}
        for relation in self._memory:
            hits = self.query(source, relation, candidate_ids, top_k)
            if hits:
                results[relation] = hits

        if cache_key is not None:
            self._cache_put(cache_key, dict(results))

        return results

    # ─── Bulk operations ────────────────────────────────────────

    def extract_association_matrix(self) -> tuple[list[str], np.ndarray]:
        """Extract a feature matrix from the holographic graph.

        For each (relation, bucket) pair, the unbound vector is
        scored against all concept addresses in one matrix multiply.
        This gives each concept a 32-dimensional feature vector
        (2 relations × 16 buckets) representing its association
        pattern across the holographic memory.

        The resulting matrix is (N, n_relations * n_buckets) —
        already low-dimensional, suitable for use as an experiential
        embedding without further SVD.

        Returns:
            (concept_ids, feature_matrix) where concept_ids is the
            list of registered concepts and feature_matrix is a
            (N, n_relations * n_buckets) float32 matrix.
        """
        concept_ids = sorted(self._addresses.keys())
        if not concept_ids:
            return [], np.zeros((0, 0), dtype=np.float32)

        n = len(concept_ids)
        addr_matrix = np.stack([self._addresses[c] for c in concept_ids])
        # addr_matrix: (N, D)

        features: list[np.ndarray] = []

        for relation in sorted(self._memory.keys()):
            role = self._roles.get(relation)
            if role is None:
                continue
            mem = self._memory[relation]
            for bucket_idx in range(self.n_buckets):
                # Unbind once for this (relation, bucket) pair
                result = circular_correlate(mem[bucket_idx], role)
                # Score against all concept addresses at once: (N,)
                scores = addr_matrix @ result
                features.append(scores.astype(np.float32))

        if not features:
            return concept_ids, np.zeros((n, 0), dtype=np.float32)

        feature_matrix = np.stack(features, axis=1)  # (N, n_features)
        return concept_ids, feature_matrix

    def add_edges(
        self,
        edges: list[tuple[str, str, str, float]],
    ) -> int:
        """Bulk add associations from a list of edges.

        Args:
            edges: List of (source, relation, target, weight) tuples.

        Returns:
            Number of edges added.
        """
        count = 0
        for source, relation, target, weight in edges:
            self.add(source, relation, target, weight)
            count += 1
        return count

    def clear(self) -> None:
        """Clear all associations from the holographic graph."""
        for relation in self._memory:
            self._memory[relation] = np.zeros(
                (self.n_buckets, self.dim), dtype=np.float32,
            )
        self._edge_count = 0

    def rebuild_from_edges(
        self,
        edges: list[tuple[str, str, str, float]],
    ) -> int:
        """Clear and rebuild the graph from a list of edges.

        Args:
            edges: List of (source, relation, target, weight) tuples.

        Returns:
            Number of edges stored.
        """
        self.clear()
        return self.add_edges(edges)

    # ─── Persistence ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the holographic graph to an .npz file."""
        # Pack addresses into a matrix
        concept_ids = sorted(self._addresses.keys())
        if concept_ids:
            addr_matrix = np.stack([self._addresses[c] for c in concept_ids])
        else:
            addr_matrix = np.zeros((0, self.dim), dtype=np.float32)

        # Pack memory into a dict of arrays
        relation_names = sorted(self._memory.keys())
        if relation_names:
            mem_matrix = np.stack([self._memory[r] for r in relation_names])
        else:
            mem_matrix = np.zeros((0, self.n_buckets, self.dim), dtype=np.float32)

        # Pack role vectors
        if relation_names:
            role_matrix = np.stack([self._roles[r] for r in relation_names])
        else:
            role_matrix = np.zeros((0, self.dim), dtype=np.float32)

        np.savez(
            path,
            dim=np.array(self.dim, dtype=np.int32),
            n_buckets=np.array(self.n_buckets, dtype=np.int32),
            edge_count=np.array(self._edge_count, dtype=np.int64),
            concept_ids=np.array(concept_ids, dtype=object),
            addr_matrix=addr_matrix,
            relation_names=np.array(relation_names, dtype=object),
            mem_matrix=mem_matrix,
            role_matrix=role_matrix,
        )

    def load(self, path: str) -> bool:
        """Load the holographic graph from an .npz file.

        Returns True if loaded, False if file doesn't exist or is
        incompatible.
        """
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=True)
            self.dim = int(data["dim"])
            self.n_buckets = int(data["n_buckets"])
            self._edge_count = int(data["edge_count"])

            # Restore addresses
            concept_ids = list(data["concept_ids"])
            addr_matrix = data["addr_matrix"]
            self._addresses = {}
            for i, cid in enumerate(concept_ids):
                self._addresses[cid] = addr_matrix[i].astype(np.float32)
            self._registered_concepts = set(concept_ids)

            # Restore memory and roles
            relation_names = list(data["relation_names"])
            mem_matrix = data["mem_matrix"]
            role_matrix = data["role_matrix"]
            self._memory = {}
            self._roles = {}
            for i, rname in enumerate(relation_names):
                self._memory[rname] = mem_matrix[i].astype(np.float32)
                self._roles[rname] = role_matrix[i].astype(np.float32)
            self._registered_relations = set(relation_names)

            self._invalidate_cache()
            return True
        except (KeyError, ValueError, OSError) as e:
            logger.warning("Failed to load holographic graph: %s", e)
            return False

    # ─── Statistics ─────────────────────────────────────────────

    def stats(self) -> dict[str, float]:
        """Return a summary of holographic graph statistics."""
        return {
            "dim": float(self.dim),
            "n_buckets": float(self.n_buckets),
            "edge_count": float(self._edge_count),
            "n_concepts": float(self.n_concepts),
            "n_relations": float(self.n_relations),
            "storage_mb": self.storage_bytes / 1e6,
        }
