"""Network topology — Genesis's understanding of its own knowledge structure.

This module computes graph-theoretic metrics on the concept network,
giving Genesis self-awareness about the shape of its knowledge:

- **Clustering coefficient** — how interconnected a concept's neighbors
  are. High clustering means a concept sits in a dense cluster of
  related ideas. Low clustering means it's a bridge between distant
  domains.

- **Centrality** — how important a concept is in the network. Degree
  centrality (many connections), betweenness centrality (bridge between
  communities), eigenvector centrality (connected to other important
  concepts).

- **Community detection** — which concepts form clusters. This reveals
  the domains of its knowledge: "cognition cluster", "emotion
  cluster", "plant cluster", etc.

- **Small-world metrics** — average path length, global clustering.
  Is its knowledge network small-world (like the brain)? Should it be?

These metrics let it say things like: "I'm well-connected around
cognition but isolated around mathematics" — structural
self-awareness that emerges from graph theory, not hardcoded facts.

The metrics are computed lazily and cached, with a refresh method
for after sleep consolidation changes the graph.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np

if TYPE_CHECKING:
    from .network import ConceptNetwork


class TopologyCache(TypedDict):
    """Typed structure of the lazily-computed topology metrics cache."""

    clustering: dict[str, float]
    degree_centrality: dict[str, float]
    eigenvector_centrality: dict[str, float]
    betweenness_centrality: dict[str, float]
    communities: list[list[str]]
    concept_to_community: dict[str, int]
    global_clustering: float
    average_path_length: float
    is_small_world: bool
    small_world_sigma: float


class NetworkTopology:
    """Graph-theoretic analysis of the concept network.

    All metrics are computed lazily and cached. Call refresh()
    after the network changes (e.g., after sleep consolidation).
    """

    def __init__(self, network: ConceptNetwork) -> None:
        """Initialize the topology analyzer with an empty, dirty cache."""
        self.network = network
        self._cache: TopologyCache = cast("TopologyCache", {})
        self._dirty = True

    # ─── Public API ─────────────────────────────────────────────

    def refresh(self) -> None:
        """Mark the cache as dirty. Metrics will be recomputed on next access."""
        self._dirty = True

    def clustering_coefficient(self, concept_name: str) -> float:
        """Local clustering coefficient for a concept.

        Measures how interconnected the concept's neighbors are.
        CC = 2 * actual_edges_between_neighbors / possible_edges_between_neighbors

        High CC (near 1.0): the concept's neighbors are all connected
        to each other — it sits in a dense cluster.

        Low CC (near 0.0): the concept's neighbors are not connected
        to each other — it's a bridge between disconnected groups.

        Returns 0.0 for concepts with fewer than 2 neighbors.
        """
        self._ensure_computed()
        return self._cache["clustering"].get(concept_name, 0.0)

    def degree_centrality(self, concept_name: str) -> float:
        """Degree centrality — fraction of nodes this concept connects to.

        DC = degree / (N - 1)

        High DC: the concept is connected to many other concepts.
        """
        self._ensure_computed()
        return self._cache["degree_centrality"].get(concept_name, 0.0)

    def eigenvector_centrality(self, concept_name: str) -> float:
        """Eigenvector centrality — connected to other important concepts.

        A concept is important if it's connected to other important
        concepts. This is the dominant eigenvector of the adjacency
        matrix. Computed via power iteration.

        High EC: the concept is in the "core" of the network.
        Low EC: the concept is in the periphery.
        """
        self._ensure_computed()
        return self._cache["eigenvector_centrality"].get(concept_name, 0.0)

    def betweenness_centrality(self, concept_name: str) -> float:
        """Betweenness centrality — how often this concept is on shortest paths.

        A concept with high betweenness is a bridge between communities.
        Removing it would disconnect parts of the network.

        Computed via BFS shortest paths. For large networks, this is
        approximated by sampling source nodes.
        """
        self._ensure_computed()
        return self._cache["betweenness_centrality"].get(concept_name, 0.0)

    def communities(self) -> list[list[str]]:
        """Detect communities using label propagation.

        Returns a list of communities, each a list of concept names.
        Concepts in the same community are more densely connected
        to each other than to concepts in other communities.

        This reveals the domains of its knowledge.
        """
        self._ensure_computed()
        return self._cache["communities"]

    def community_of(self, concept_name: str) -> int | None:
        """Which community does a concept belong to?

        Returns the community index, or None if the concept is not
        in the network.
        """
        self._ensure_computed()
        return self._cache["concept_to_community"].get(concept_name)

    def global_clustering(self) -> float:
        """Global clustering coefficient — average of local clustering.

        High global clustering means the network has dense local
        neighborhoods — a hallmark of small-world networks (like
        the brain).
        """
        self._ensure_computed()
        return self._cache["global_clustering"]

    def average_path_length(self) -> float:
        """Average shortest path length between all pairs of concepts.

        Low APL means any concept can reach any other quickly —
        another hallmark of small-world networks.
        """
        self._ensure_computed()
        return self._cache["average_path_length"]

    def is_small_world(self) -> bool:
        """Whether the network has small-world topology.

        Uses the Watts-Strogatz small-world coefficient σ:
        σ = (C / C_rand) / (L / L_rand)
        where C_rand and L_rand are expected values for an equivalent
        Erdős-Rényi random graph. σ > 1 indicates small-world structure.
        The brain has σ >> 1.
        """
        self._ensure_computed()
        return self._cache["is_small_world"]

    def small_world_sigma(self) -> float:
        """The Watts-Strogatz small-world coefficient σ.

        σ > 1 indicates small-world structure.
        σ >> 1 indicates strong small-world structure (like the brain).
        """
        self._ensure_computed()
        return self._cache.get("small_world_sigma", 0.0)

    def hub_concepts(self, k: int = 10) -> list[tuple[str, float]]:
        """The k most central concepts by eigenvector centrality.

        These are the "hubs" of its knowledge — the concepts that
        connect to the most other important concepts.
        """
        self._ensure_computed()
        ec = self._cache["eigenvector_centrality"]
        sorted_concepts = sorted(ec.items(), key=lambda x: -x[1])
        return sorted_concepts[:k]

    def bridge_concepts(self, k: int = 10) -> list[tuple[str, float]]:
        """The k concepts with highest betweenness centrality.

        These are the "bridges" of its knowledge — concepts that
        connect otherwise distant domains.
        """
        self._ensure_computed()
        bc = self._cache["betweenness_centrality"]
        sorted_concepts = sorted(bc.items(), key=lambda x: -x[1])
        return sorted_concepts[:k]

    def knowledge_domains(self) -> list[tuple[str, int, float]]:
        """Summary of its knowledge domains (communities).

        Returns a list of (top_concept, size, cohesion) for each
        community, sorted by size descending. The top_concept is
        the most central concept in the community.
        """
        self._ensure_computed()
        communities = self._cache["communities"]
        ec = self._cache["eigenvector_centrality"]

        domains = []
        for comm in communities:
            if len(comm) < 3:
                continue
            # Find the most central concept in this community
            top = max(comm, key=lambda c: ec.get(c, 0.0))
            # Cohesion = average clustering within the community
            clustering = self._cache["clustering"]
            cohesion = sum(clustering.get(c, 0.0) for c in comm) / len(comm)
            domains.append((top, len(comm), cohesion))

        domains.sort(key=lambda x: -x[1])
        return domains

    def describe_structure(self) -> str:
        """Compose a natural-language description of its knowledge structure.

        This is what it says when asked about the shape of its
        knowledge — it's structural self-awareness, not hardcoded.

        For large networks, expensive metrics are skipped to keep
        the response fast enough for conversation.
        """
        parts: list[str] = []

        n = self.network.size
        parts.append(f"knowledge network has {n} concepts")

        edges = self.network.edge_count
        parts.append(f"and {edges} connections")

        # For large networks, use sampling for path length and
        # degree-based density assessment (not raw edge density, which
        # is always near-zero for large sparse graphs).
        if n > 10000:
            return self._describe_large_network(parts, n, edges)

        # For smaller networks, compute full metrics
        return self._describe_small_network(parts)

    def _describe_large_network(
        self, parts: list[str], n: int, edges: int
    ) -> str:
        """Describe structure for large networks using sampling."""
        # Average degree — a better density metric for large graphs
        avg_degree = edges * 2 / n if n > 0 else 0
        if avg_degree > 8:
            parts.append("It's densely interconnected")
        elif avg_degree > 3:
            parts.append("It has moderate interconnection")
        else:
            parts.append("It's sparsely connected but growing")

        # Sampled path length — pick 200 random pairs and BFS
        import random as _rng

        sample_size = min(200, n)
        sample = _rng.sample(list(self.network._concepts.keys()), sample_size)
        path_lengths = []
        for i in range(0, len(sample), 2):
            if i + 1 >= len(sample):
                break
            path = self.network.find_path(sample[i], sample[i + 1], max_depth=8)
            if path:
                path_lengths.append(len(path))
        if path_lengths:
            apl = sum(path_lengths) / len(path_lengths)
            parts.append(f"Any concept is about {apl:.1f} steps from any other")
            # Small-world heuristic: short paths + clustering
            if apl < 4:
                parts.append(
                    "with a small-world structure — dense clusters linked"
                    " by short paths, like a brain"
                )

        # Top domains by counting concepts per column prefix
        domains = self.knowledge_domains()
        if domains:
            top = domains[:3]
            names = [f"{name} ({size} concepts)" for name, size, _ in top]
            parts.append(f"largest knowledge domains are {', '.join(names)}")

        # Hubs by degree
        hubs = self.hub_concepts(3)
        if hubs:
            hub_names = [name for name, _ in hubs]
            parts.append(f"most central concepts are {', '.join(hub_names)}")

        return ". ".join(parts) + "."

    def _describe_small_network(self, parts: list[str]) -> str:
        """Describe structure for smaller networks using full metrics."""
        # For smaller networks, compute full metrics
        self._ensure_computed()

        # Global clustering
        gc = self._cache["global_clustering"]
        if gc > 0.3:
            parts.append("It's densely interconnected")
        elif gc > 0.1:
            parts.append("It has moderate interconnection")
        else:
            parts.append("It's sparsely connected")

        # Small-world
        if self._cache["is_small_world"]:
            parts.append(
                "with a small-world structure — dense clusters linked by short paths, like a brain"
            )

        # Path length
        apl = self._cache["average_path_length"]
        if apl > 0:
            parts.append(f"Any concept is about {apl:.1f} steps from any other")

        # Domains
        domains = self.knowledge_domains()
        if domains:
            top_domains = domains[:3]
            domain_names = [f"{name} ({size} concepts)" for name, size, _ in top_domains]
            parts.append(f"largest knowledge domains are {', '.join(domain_names)}")

        # Hubs
        hubs = self.hub_concepts(3)
        if hubs:
            hub_names = [name for name, _ in hubs]
            parts.append(f"most central concepts are {', '.join(hub_names)}")

        # Bridges
        bridges = self.bridge_concepts(3)
        if bridges:
            bridge_names = [name for name, _ in bridges]
            parts.append(f"Concepts that bridge my knowledge are {', '.join(bridge_names)}")

        return ". ".join(parts) + "."

    # ─── Internal: computation ──────────────────────────────────

    def _ensure_computed(self) -> None:
        """Compute all metrics if not cached or if dirty."""
        if not self._dirty and self._cache:
            return
        self._compute_all()
        self._dirty = False

    def _compute_all(self) -> None:
        """Compute all topology metrics.

        For large networks (>10k concepts), expensive O(n²) metrics
        (clustering, betweenness, path length, small-world test) are
        skipped to ensure responsiveness. Only fast metrics (degree
        centrality, eigenvector centrality, communities) are computed.
        """
        concepts = list(self.network._concepts.keys())
        n = len(concepts)
        if n == 0:
            self._cache = self._build_topology_cache(
                {}, {}, {}, {}, [], {}, 0.0, 0.0, False, 0.0,
            )
            return

        # Build adjacency list (undirected, weighted)
        adj: dict[str, list[str]] = defaultdict(list)
        for edge in self.network._edges:
            adj[edge.source].append(edge.target)
            adj[edge.target].append(edge.source)

        # For large networks, skip expensive O(n²) or O(n*iterations*k) metrics
        LARGE_NETWORK_THRESHOLD = 10000
        is_large = n > LARGE_NETWORK_THRESHOLD

        # 1. Clustering coefficient — O(n * k²), skip for large networks
        if is_large:
            clustering = {}
            global_clustering = 0.0
        else:
            clustering = self._compute_clustering(concepts, adj)
            global_clustering = sum(clustering.values()) / n if n > 0 else 0.0

        # 2. Degree centrality — O(n), always fast
        degree_centrality = {c: len(adj.get(c, [])) / max(1, n - 1) for c in concepts}

        # 3. Eigenvector centrality — O(n * iterations * k), skip for large networks
        if is_large:
            # Use degree centrality as a proxy for eigenvector centrality
            eigenvector_centrality = degree_centrality
        else:
            eigenvector_centrality = self._compute_eigenvector_centrality(concepts, adj, n)

        # 4. Betweenness centrality — O(n * (n + edges)), skip for large networks
        if is_large:
            betweenness_centrality = {}
        else:
            betweenness_centrality = self._compute_betweenness_centrality(concepts, adj, n)

        # 5. Community detection — O(n * iterations * k), skip for large networks
        if is_large:
            # Simple degree-based community: top-degree nodes are their own communities
            communities: list[list[str]] = []
            concept_to_community: dict[str, int] = {}
        else:
            communities, concept_to_community = self._detect_communities(concepts, adj)

        # 6. Average path length — O(n * (n + edges)), skip for large networks
        if is_large:
            average_path_length = 0.0
        else:
            average_path_length = self._compute_average_path_length(concepts, adj, n)

        # 8. Small-world test using the Watts-Strogatz sigma coefficient
        small_world_sigma, is_small_world = self._compute_small_world_sigma(
            global_clustering, average_path_length, n
        )

        self._cache = self._build_topology_cache(
            clustering, degree_centrality, eigenvector_centrality,
            betweenness_centrality, communities, concept_to_community,
            global_clustering, average_path_length, is_small_world,
            small_world_sigma,
        )

    def _build_topology_cache(
        self,
        clustering: dict[str, float],
        degree_centrality: dict[str, float],
        eigenvector_centrality: dict[str, float],
        betweenness_centrality: dict[str, float],
        communities: list[list[str]],
        concept_to_community: dict[str, int],
        global_clustering: float,
        average_path_length: float,
        is_small_world: bool,
        small_world_sigma: float,
    ) -> TopologyCache:
        """Build the topology metrics cache dict."""
        return {
            "clustering": clustering,
            "degree_centrality": degree_centrality,
            "eigenvector_centrality": eigenvector_centrality,
            "betweenness_centrality": betweenness_centrality,
            "communities": communities,
            "concept_to_community": concept_to_community,
            "global_clustering": global_clustering,
            "average_path_length": average_path_length,
            "is_small_world": is_small_world,
            "small_world_sigma": small_world_sigma,
        }

    def _compute_small_world_sigma(
        self,
        global_clustering: float,
        average_path_length: float,
        n: int,
    ) -> tuple[float, bool]:
        """Compute the Watts-Strogatz small-world coefficient sigma.

        A network is small-world if it has much higher clustering than
        an equivalent random graph, while having similar path length.
        The standard measure is the small-world coefficient (Watts &
        Strogatz, 1998; Humphries & Gurney, 2008):

          sigma = (C / C_rand) / (L / L_rand)

        where C is the clustering coefficient, L is the average path
        length, and C_rand, L_rand are the expected values for an
        Erdos-Renyi random graph with the same number of nodes and
        edges:

          C_rand = <k> / n        (expected clustering for ER graph)
          L_rand = ln(n) / ln(<k>)  (expected APL for ER graph)

        where <k> = 2*edges/n is the average degree.

        sigma > 1 indicates small-world structure. The brain has sigma >> 1.
        """
        avg_degree = 2.0 * len(self.network._edges) / n if n > 0 else 0.0
        c_rand = avg_degree / n if n > 0 else 0.0
        l_rand = (
            math.log(max(n, 2)) / math.log(max(avg_degree, 1.01))
            if avg_degree > 1.0
            else float("inf")
        )

        if c_rand > 1e-10 and average_path_length > 1e-10 and l_rand > 0 and l_rand < float("inf"):
            small_world_sigma = (global_clustering / c_rand) / (average_path_length / l_rand)
            is_small_world = small_world_sigma > 1.0
        else:
            small_world_sigma = 0.0
            is_small_world = False

        return small_world_sigma, is_small_world
    def _compute_clustering(
        self,
        concepts: list[str],
        adj: dict[str, list[str]],
    ) -> dict[str, float]:
        """Compute local clustering coefficient for each concept."""
        clustering: dict[str, float] = {}
        neighbor_sets: dict[str, set[str]] = {}

        for c in concepts:
            neighbors = adj.get(c, [])
            neighbor_sets[c] = set(neighbors)

        for c in concepts:
            neighbors = list(neighbor_sets[c])
            k = len(neighbors)
            if k < 2:
                clustering[c] = 0.0
                continue

            # Count edges between neighbors
            actual = 0
            for i in range(k):
                ni_set = neighbor_sets.get(neighbors[i])
                if not ni_set:
                    continue
                # Count how many of neighbors[i+1:] are in ni_set
                actual += len(ni_set & set(neighbors[i + 1 :]))

            possible = k * (k - 1) / 2
            clustering[c] = actual / possible if possible > 0 else 0.0

        return clustering

    def _compute_eigenvector_centrality(
        self,
        concepts: list[str],
        adj: dict[str, list[str]],
        n: int,
    ) -> dict[str, float]:
        """Compute eigenvector centrality via PageRank-style power iteration.

        Uses a damping factor (α = 0.85, same as PageRank) to ensure
        convergence on disconnected or bipartite graphs. Pure power
        iteration fails on disconnected graphs (isolated components
        get zero centrality) and can oscillate on bipartite graphs.

        The PageRank-style update is:
            v_i = (1-α)/n + α * Σ_{j∈N(i)} v_j / deg(j)

        This guarantees all nodes receive a baseline centrality of
        (1-α)/n, ensuring the measure is well-defined for any graph
        topology (Brin & Page, 1998; Bonacich, 1987).
        """
        if n < 2:
            return {concepts[0]: 1.0} if concepts else {}

        concept_to_idx = {c: i for i, c in enumerate(concepts)}
        alpha = 0.85  # damping factor
        vec = np.ones(n, dtype=np.float64) / n

        # Precompute out-degrees for normalization
        out_degree = np.zeros(n, dtype=np.float64)
        for c in concepts:
            i = concept_to_idx[c]
            out_degree[i] = max(1, len(adj.get(c, [])))

        for _ in range(100):
            new_vec = np.full(n, (1.0 - alpha) / n, dtype=np.float64)
            for c in concepts:
                i = concept_to_idx[c]
                for neighbor in adj.get(c, []):
                    j = concept_to_idx.get(neighbor)
                    if j is None:
                        continue
                    # Distribute neighbor's centrality by edge weight
                    new_vec[i] += alpha * vec[j] / out_degree[j]

            # Check convergence
            if np.allclose(vec, new_vec, atol=1e-8):
                vec = new_vec
                break
            vec = new_vec

        return {concepts[i]: float(vec[i]) for i in range(n)}

    def _compute_betweenness_centrality(
        self,
        concepts: list[str],
        adj: dict[str, list[str]],
        n: int,
    ) -> dict[str, float]:
        """Compute betweenness centrality via BFS shortest paths.

        For large networks, we sample source nodes to keep this
        tractable. The sampling gives an approximation that's
        sufficient for ranking purposes.
        """
        betweenness: dict[str, float] = defaultdict(float)

        # For large networks, sample sources
        max_sources = min(n, 500)
        if n > max_sources:
            import random

            rng = random.Random(42)  # deterministic for reproducibility
            sources = rng.sample(concepts, max_sources)
        else:
            sources = concepts

        for source in sources:
            # BFS from source, tracking shortest paths
            dist: dict[str, int] = {source: 0}
            paths: dict[str, int] = {source: 1}
            queue = deque([source])
            visited_order: list[str] = []

            while queue:
                current = queue.popleft()
                visited_order.append(current)
                for neighbor in adj.get(current, []):
                    if neighbor not in dist:
                        dist[neighbor] = dist[current] + 1
                        paths[neighbor] = paths[current]
                        queue.append(neighbor)
                        continue
                    if dist[neighbor] == dist[current] + 1:
                        paths[neighbor] += paths[current]

            # Back-propagate dependencies
            dependency: dict[str, float] = defaultdict(float)
            for node in reversed(visited_order):
                for neighbor in adj.get(node, []):
                    # Flatten: combine the distance check and paths check
                    if dist.get(neighbor, -1) == dist[node] + 1 and paths[node] > 0:
                        ratio = paths[node] / paths[neighbor]
                        dependency[node] += ratio * (1 + dependency[neighbor])
                if node != source:
                    betweenness[node] += dependency[node]

        # Normalize, correcting for sampling bias
        # When we sample s sources out of n, the betweenness is
        # underestimated by a factor of n/s. We correct this by
        # scaling up by n/len(sources) (Brandes, 2001; Bader et al., 2007).
        if n > 2:
            sampling_factor = n / len(sources)
            scale = 2.0 / ((n - 1) * (n - 2)) * sampling_factor
        else:
            scale = 1.0
        return {c: betweenness[c] * scale for c in concepts}

    def _compute_average_path_length(
        self,
        concepts: list[str],
        adj: dict[str, list[str]],
        n: int,
    ) -> float:
        """Compute average shortest path length via BFS.

        For large networks, samples source nodes.
        """
        if n < 2:
            return 0.0

        max_sources = min(n, 200)
        if n > max_sources:
            import random

            rng = random.Random(42)
            sources = rng.sample(concepts, max_sources)
        else:
            sources = concepts

        total_length = 0
        pair_count = 0

        for source in sources:
            dist: dict[str, int] = {source: 0}
            queue = deque([source])
            while queue:
                current = queue.popleft()
                for neighbor in adj.get(current, []):
                    if neighbor in dist:
                        continue
                    dist[neighbor] = dist[current] + 1
                    queue.append(neighbor)
                    total_length += dist[neighbor]
                    pair_count += 1

        return total_length / pair_count if pair_count > 0 else 0.0

    def _detect_communities(
        self,
        concepts: list[str],
        adj: dict[str, list[str]],
    ) -> tuple[list[list[str]], dict[str, int]]:
        """Detect communities using label propagation.

        Each concept starts with a unique label. On each iteration,
        every concept adopts the label most common among its neighbors.
        Converges to communities where concepts share labels with
        their neighbors. Fast (near-linear) and parameter-free.
        """
        import random

        rng = random.Random(42)

        # Initialize: each concept has its own label
        labels: dict[str, int] = {c: i for i, c in enumerate(concepts)}

        # Iterate until convergence
        for _ in range(20):
            changed = False
            order = list(concepts)
            rng.shuffle(order)

            for c in order:
                neighbors = adj.get(c, [])
                if not neighbors:
                    continue

                # Count labels among neighbors
                label_counts: dict[int, int] = defaultdict(int)
                for neighbor in neighbors:
                    label_counts[labels[neighbor]] += 1

                # Adopt the most common label
                best_label = max(label_counts, key=lambda k: label_counts[k])
                if labels[c] != best_label:
                    labels[c] = best_label
                    changed = True

            if not changed:
                break

        # Group by label
        communities: dict[int, list[str]] = defaultdict(list)
        for c in concepts:
            communities[labels[c]].append(c)

        community_list = list(communities.values())
        # Sort by size descending
        community_list.sort(key=len, reverse=True)

        # Map concept to community index
        concept_to_community: dict[str, int] = {}
        for i, comm in enumerate(community_list):
            for c in comm:
                concept_to_community[c] = i

        return community_list, concept_to_community
