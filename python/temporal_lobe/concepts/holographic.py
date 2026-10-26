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
- B = 64 buckets per relation type
- Each concept gets a deterministic random unit vector (its "address")
- Each relation type gets a deterministic random unit vector (its "role")
- Adding edge (A, R, B): ``memory_R[hash(A) % B] += bind(addr_A, addr_B)``
- Querying (A, R): ``unbind(addr_A, memory_R[hash(A) % B])`` → compare
  to all concept addresses

With ~40K journaled edges across ~11 relation types and 64 buckets, each
bucket holds ~50-200 bindings. The SNR is sqrt(D/N) ≈ 3-6, sufficient
for candidate generation — with the proviso that a hub concept carries
all its own bindings in one bucket, so the journal's weight-ordered
capacity bound is what keeps hub buckets retrievable.

Storage: 15 relations × 64 buckets × 2048 dims × 4 bytes = 7.9 MB **fixed**.

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

from infrastructure._npz_io import load_npz, save_npz, str_array

logger = logging.getLogger(__name__)


# ─── Constants ──────────────────────────────────────────────────

# Dimensionality of the holographic vectors. Higher = better SNR but
# more storage. 2048 is a good balance: SNR ~3 with 200 bindings per
# bucket, total storage ~1.9 MB.
DEFAULT_DIM = 2048

# Number of buckets per relation type. More buckets = fewer sources
# sharing each memory vector = better SNR for everyone but the hubs.
# Storage is still trivial: ~8 MB for 15 relations × 64 × 2048.
DEFAULT_BUCKETS = 64

# Maximum distinct associations the journal tracks. Holographic
# memory cannot enumerate its contents — superposition is one-way —
# so the journal is the rebuild record: every (source, relation,
# target) ever bound, deduplicated with accumulated weight. Its cap
# is the capacity budget: at ~280 bindings per (relation, bucket)
# vector the SNR is sqrt(D/N) ≈ 2.7, near the designed operating
# point. When the journal is full the weakest entries are evicted —
# their bindings remain in memory until the next rebuild, a slow
# forgetting rather than a cliff.
DEFAULT_JOURNAL_CAP = 49152

# Trace encoding versions. Every write and query honors the graph's
# own encoding, and every rebuild re-encodes at the newest version.
#
# v1 (legacy): ``bind(role, target)`` — the source concept only
#   hashed to a bucket, so every source sharing that bucket returned
#   its cohabitants' targets as signal. With ~400 sources per bucket
#   at production scale, recall was cohabitation, not association.
#
# v2: ``bind(source_addr, target_addr)`` — the documented design.
#   Unbinding by source address recovers only that source's targets;
#   cohabitant bindings become true noise instead of rival signal.
#   Relation separation was already structural (per-relation memory
#   matrices), so the role binding was redundant anyway.
ENCODING = 2


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
        journal_cap: int = DEFAULT_JOURNAL_CAP,
    ) -> None:
        """Initialize the holographic graph store."""
        self.dim = dim
        self.n_buckets = n_buckets
        self.journal_cap = journal_cap

        # Per-relation bucketed memory: relation_name → (B, D) float32
        self._memory: dict[str, np.ndarray] = {}

        # Concept address vectors: concept_id → (D,) float32
        self._addresses: dict[str, np.ndarray] = {}

        # Role vectors: relation_name → (D,) float32
        self._roles: dict[str, np.ndarray] = {}

        # Track which concepts and relations have been registered
        self._registered_concepts: set[str] = set()
        self._registered_relations: set[str] = set()

        # Trace encoding version of the in-memory contents — see
        # ENCODING. A v1 file loads as 1 and keeps v1 semantics (and
        # v1 binds) until a rebuild re-encodes it as 2.
        self._encoding = ENCODING

        # Edge journal: (source, relation, target) → accumulated weight.
        # Bounded at journal_cap — see DEFAULT_JOURNAL_CAP.
        self._journal: dict[tuple[str, str, str], float] = {}

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
    def distinct_edges(self) -> int:
        """Number of distinct associations in the journal."""
        return len(self._journal)

    @property
    def bindings_per_bucket(self) -> float:
        """Mean raw bindings per (relation, bucket) memory vector.

        The SNR driver: noise grows with the number of superposed
        bindings per memory vector, not with the total edge count.
        """
        n_vecs = max(1, len(self._memory) * self.n_buckets)
        return self._edge_count / n_vecs

    @property
    def snr_estimate(self) -> float:
        """Expected signal-to-noise ratio sqrt(D/N) at current load."""
        return float(np.sqrt(self.dim / max(1.0, self.bindings_per_bucket)))

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
                _hash_to_seed(concept_id),
                self.dim,
            )
            self._registered_concepts.add(concept_id)
        return self._addresses[concept_id]

    def _get_role(self, relation: str) -> np.ndarray:
        """Get or create the role vector for a relation type."""
        if relation not in self._roles:
            self._roles[relation] = _seeded_unit_vector(
                _hash_to_seed(f"role:{relation}"),
                self.dim,
            )
            self._registered_relations.add(relation)
            # Initialize memory for this relation
            self._memory[relation] = np.zeros(
                (self.n_buckets, self.dim),
                dtype=np.float32,
            )
        return self._roles[relation]

    def _bucket_index(self, concept_id: str) -> int:
        """Hash a concept ID to a bucket index."""
        return _hash_to_seed(concept_id) % self.n_buckets

    # ─── Core operations ────────────────────────────────────────

    def _bind(self, source: str, relation: str, target: str, weight: float) -> None:
        """Superpose one association into memory without bookkeeping.

        The raw memory write shared by ``add`` (which also journals)
        and rebuilds (which re-encode the journal verbatim).
        """
        # Register the source address (needed for querying)
        addr_source = self._get_address(source)
        addr_target = self._get_address(target)
        role = self._get_role(relation)

        # Bind the association trace. v2 binds source⊗target (the
        # documented design); v1 files keep role⊗target until rebuilt.
        if self._encoding >= 2:
            bound = circular_convolve(addr_source, addr_target) * weight
        else:
            bound = circular_convolve(role, addr_target) * weight

        # Superpose into the source's bucket
        bucket = self._bucket_index(source)
        self._memory[relation][bucket] += bound

    def _journal_add(self, source: str, relation: str, target: str, weight: float) -> None:
        """Record an association in the bounded journal.

        Repeated bindings accumulate weight (repetition strengthens,
        saturating at 1.0). When the journal is over capacity the
        weakest entries are evicted; their superposed bindings stay in
        memory until the next rebuild — gradual forgetting, not a wipe.
        """
        key = (source, relation, target)
        prev = self._journal.get(key)
        self._journal[key] = weight if prev is None else min(1.0, prev + weight)
        if len(self._journal) > self.journal_cap:
            self._evict_journal()

    def _evict_journal(self) -> None:
        """Drop the weakest journal entries back under the cap.

        Evicts down to 90% of the cap so eviction is an occasional
        batch, not a per-add sort.
        """
        keep = int(self.journal_cap * 0.9)
        ranked = sorted(self._journal.items(), key=lambda kv: kv[1], reverse=True)
        self._journal = dict(ranked[:keep])

    def add(self, source: str, relation: str, target: str, weight: float = 1.0) -> None:
        """Add an association to the holographic graph.

        The association (source, relation, target) is stored via
        superposition of the bound role×target vector into the source's
        bucket in the relation's memory, and journaled so periodic
        rebuilds can re-encode the graph from a clean state.

        Args:
            source: Source concept ID.
            relation: Relation type name (e.g. "related_to", "similar_to").
            target: Target concept ID.
            weight: Association strength (0-1). Scales the superposed
                vector. Weaker associations contribute less signal.
        """
        self._bind(source, relation, target, weight)
        self._journal_add(source, relation, target, weight)
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

        bucket = self._bucket_index(source)

        # Unbind: recover the target from the memory
        # v2: memory[bucket] ≈ bind(source, target) + noise
        #     unbind(source, memory[bucket]) ≈ target + noise
        # v1: memory[bucket] ≈ bind(role, target) + noise (pooled per bucket)
        if self._encoding >= 2:
            probe = self._addresses[source]
        else:
            probe = self._get_role(relation)
        result = circular_correlate(self._memory[relation][bucket], probe)

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

    def association_fidelity(
        self,
        edges: list[tuple[str, str, str, float]],
        top_k: int = 10,
    ) -> dict[str, float]:
        """Measure retrieval fidelity against known associations.

        Holographic memory is an approximate associative representation, so
        exact vector MSE is not an appropriate reconstruction metric. This
        evaluates whether stored associations survive the encode/unbind
        transform: recall@k, mean reciprocal rank, and target similarity.
        The supplied edges are evaluation input and are not retained.
        """
        if not edges or top_k < 1:
            return {
                "recall_at_k": 0.0,
                "mean_reciprocal_rank": 0.0,
                "mean_target_similarity": 0.0,
                "n_evaluated": 0.0,
            }

        reciprocal_ranks: list[float] = []
        target_similarities: list[float] = []
        hits = 0

        for source, relation, target, _weight in edges:
            if source not in self._addresses or target not in self._addresses:
                continue
            results = self.query(source, relation, top_k=max(top_k, 1))
            rank = next(
                (i + 1 for i, (cid, _score) in enumerate(results) if cid == target),
                None,
            )
            if rank is not None:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            result_ids = {cid for cid, score in results}
            if target in result_ids:
                target_similarities.append(next(score for cid, score in results if cid == target))
            else:
                bucket = self._bucket_index(source)
                probe = (
                    self._addresses[source] if self._encoding >= 2 else self._roles.get(relation)
                )
                if probe is not None:
                    recovered = circular_correlate(self._memory[relation][bucket], probe)
                    target_similarities.append(float(np.dot(recovered, self._addresses[target])))

        n = len(reciprocal_ranks)
        if n == 0:
            return {
                "recall_at_k": 0.0,
                "mean_reciprocal_rank": 0.0,
                "mean_target_similarity": 0.0,
                "n_evaluated": 0.0,
            }
        return {
            "recall_at_k": float(hits / n),
            "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
            "mean_target_similarity": float(np.mean(target_similarities)),
            "n_evaluated": float(n),
        }

    # ─── Bulk operations ────────────────────────────────────────

    def extract_association_matrix(self) -> tuple[list[str], np.ndarray]:
        """Extract a feature matrix from the holographic graph.

        For each (relation, bucket) pair — or, under v2 encoding,
        each (concept, relation) pair — the unbound vector is scored
        into a compact feature representing the concept's association
        pattern across the holographic memory.

        The resulting matrix is (N, n_relations * n_buckets) for a
        v1-encoded graph and (N, n_relations) for v2 — in v2 a bucket's
        contents only unbind meaningfully per-source, so the feature is
        each concept's own recovered-association strength per relation.
        Either way it is already low-dimensional, suitable for use as an
        experiential embedding without further SVD.

        Returns:
            (concept_ids, feature_matrix) where concept_ids is the
            list of registered concepts and feature_matrix is an
            (N, n_features) float32 matrix.
        """
        concept_ids = sorted(self._addresses.keys())
        if not concept_ids:
            return [], np.zeros((0, 0), dtype=np.float32)

        n = len(concept_ids)
        addr_matrix = np.stack([self._addresses[c] for c in concept_ids])
        # addr_matrix: (N, D)

        features: list[np.ndarray] = []

        if self._encoding >= 2:
            # Per-source unbind: for each concept, recover its own
            # association signal from its (relation, bucket) home and
            # score that recovered vector against all addresses. The
            # norm of the scored vector is the concept's association
            # strength in that relation — a compact per-source feature.
            for relation in sorted(self._memory.keys()):
                mem = self._memory[relation]
                col = np.zeros(n, dtype=np.float32)
                for i, cid in enumerate(concept_ids):
                    result = circular_correlate(mem[self._bucket_index(cid)], self._addresses[cid])
                    col[i] = float(np.linalg.norm(result))
                features.append(col)
        else:
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
        """Clear all associations from the holographic graph.

        Also invalidates the query cache and drops the concept/relation
        registrations. Every other write path calls
        `_invalidate_cache`, and `clear` did not — so after a clear,
        `query()` short-circuited on the cache and returned the
        pre-wipe associations out of a provably empty store. That is
        fabricated memory, returned with the same confidence as a real
        one, and it is reachable: `rebuild_from_edges([])` is
        `clear()` plus nothing.

        `_addresses` is dropped for the same reason. `query()` scores
        candidates against the address map, so leaving it behind means
        a cleared graph still enumerates concepts it no longer has any
        association for.
        """
        for relation in self._memory:
            self._memory[relation] = np.zeros(
                (self.n_buckets, self.dim),
                dtype=np.float32,
            )
        self._addresses = {}
        self._roles = {}
        self._registered_concepts = set()
        self._registered_relations = set()
        self._journal = {}
        self._edge_count = 0
        self._invalidate_cache()

    def rebuild_from_edges(
        self,
        edges: list[tuple[str, str, str, float]],
    ) -> int:
        """Clear and rebuild the graph from a list of edges.

        The journal is replaced wholesale by the deduplicated input
        (max weight per edge triple) — a rebuild declares the complete
        content set, so input edges are not accumulated on top of the
        old journal.

        Args:
            edges: List of (source, relation, target, weight) tuples.

        Returns:
            Number of edges stored.
        """
        journal: dict[tuple[str, str, str], float] = {}
        for source, relation, target, weight in edges:
            key = (source, relation, target)
            if weight > journal.get(key, 0.0):
                journal[key] = weight
        self.clear()
        self._encoding = ENCODING  # rebuilds always re-encode newest
        self._journal = journal
        if len(self._journal) > self.journal_cap:
            self._evict_journal()
        for (source, relation, target), weight in self._journal.items():
            self._bind(source, relation, target, weight)
        self._edge_count = len(self._journal)
        self._invalidate_cache()
        return self._edge_count

    def journal_edges(self) -> list[tuple[str, str, str, float]]:
        """The journaled association set: (source, relation, target, weight)."""
        return [
            (source, relation, target, weight)
            for (source, relation, target), weight in self._journal.items()
        ]

    def rebuild_from_journal(self) -> int:
        """Clear and re-encode the graph from its own journal.

        This is the maintenance rebuild: the journal is the clean,
        deduplicated record of what was bound, so re-binding it drops
        the duplicate reinforcement and evicted-ghost bindings the
        memory accumulated since the last rebuild. An empty journal
        rebuilds nothing — the memory is left intact.
        """
        if not self._journal:
            return 0
        return self.rebuild_from_edges(self.journal_edges())

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

        # Pack the journal — the rebuild record — as parallel arrays
        j_sources = [k[0] for k in self._journal]
        j_relations = [k[1] for k in self._journal]
        j_targets = [k[2] for k in self._journal]
        j_weights = np.array(list(self._journal.values()), dtype=np.float32)

        save_npz(
            path,
            dim=np.array(self.dim, dtype=np.int32),
            n_buckets=np.array(self.n_buckets, dtype=np.int32),
            edge_count=np.array(self._edge_count, dtype=np.int64),
            encoding=np.array(self._encoding, dtype=np.int32),
            concept_ids=str_array(concept_ids),
            addr_matrix=addr_matrix,
            relation_names=str_array(relation_names),
            mem_matrix=mem_matrix,
            role_matrix=role_matrix,
            j_source=str_array(j_sources),
            j_relation=str_array(j_relations),
            j_target=str_array(j_targets),
            j_weight=j_weights,
        )

    def load(self, path: str) -> bool:
        """Load the holographic graph from an .npz file.

        Returns True if loaded, False if file doesn't exist or is
        incompatible.
        """
        if not os.path.exists(path):
            return False
        try:
            data = load_npz(path)
            self.dim = int(data["dim"])
            self.n_buckets = int(data["n_buckets"])
            self._edge_count = int(data["edge_count"])
            # Files written before source-binding carry no encoding
            # field — they are v1, and stay v1 until a rebuild.
            self._encoding = int(data["encoding"]) if "encoding" in data else 1

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

            # Restore the journal. Older files predate it — an absent
            # journal loads empty and the first rebuild after the cap
            # fills it skips the wipe via rebuild_from_journal's guard.
            self._journal = {}
            if "j_source" in data:
                js = list(data["j_source"])
                jr = list(data["j_relation"])
                jt = list(data["j_target"])
                jw = data["j_weight"]
                for i in range(len(js)):
                    self._journal[(js[i], jr[i], jt[i])] = float(jw[i])
                if len(self._journal) > self.journal_cap:
                    self._evict_journal()

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
            "distinct_edges": float(self.distinct_edges),
            "n_concepts": float(self.n_concepts),
            "n_relations": float(self.n_relations),
            "bindings_per_bucket": self.bindings_per_bucket,
            "snr_estimate": self.snr_estimate,
            "journal_cap": float(self.journal_cap),
            "encoding": float(self._encoding),
            "storage_mb": self.storage_bytes / 1e6,
        }
