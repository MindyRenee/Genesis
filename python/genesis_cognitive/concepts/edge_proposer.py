"""Edge proposer — discovers new relationships from embedding proximity.

During sleep, Genesis scans its latent space for pairs of concepts that
are close together in embedding space but have no graph edge between
them. These are candidate SIMILAR_TO relationships — connections it
wasn't explicitly told but can infer from semantic similarity.

This is where generalization happens. The symbolic graph contains only
what it was taught. The embedding space contains implicit relationships
from the distributional structure of language. The edge proposer bridges
the two: it uses the latent space to discover patterns, then writes them
into the symbolic graph as explicit knowledge.

The process is conservative:
- Only proposes edges above a high similarity threshold (default 0.85)
- Skips pairs that already have any edge (avoids redundancy)
- Limits proposals per concept (avoids flooding the graph)
- Marks proposed edges with origin="discovered" so they can be
  distinguished from taught knowledge

This mirrors hippocampal consolidation: during sleep, the hippocampus
replays experiences, extracts patterns, and writes them into cortical
long-term memory. The embedding space is the hippocampal attractor
basin; the concept network is the cortical knowledge store.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .network import RelationType

if TYPE_CHECKING:
    from .embeddings import EmbeddingStore
    from .network import ConceptNetwork


class EdgeProposer:
    """Proposes new graph edges from embedding proximity."""

    def __init__(
        self,
        network: ConceptNetwork,
        embeddings: EmbeddingStore,
        threshold: float = 0.85,
        max_per_concept: int = 3,
        max_per_session: int = 200,
    ) -> None:
        """Initialize the edge proposer with a concept network."""
        self.network = network
        self.embeddings = embeddings
        self.threshold = threshold
        self.max_per_concept = max_per_concept
        self.max_per_session = max_per_session

    def propose_edges(self) -> list[tuple[str, str, float]]:
        """Scan embedding space for candidate SIMILAR_TO edges.

        Returns a list of (concept_a, concept_b, similarity) tuples
        for pairs that are close in embedding space but have no
        existing graph edge. Does not modify the network — the caller
        decides whether to accept the proposals.

        This uses the TF-IDF (text/definition) component of the
        embedding space rather than the spectral (graph-structure)
        component. The spectral component is dominated by structural
        bridge/hub-attachment edges, which produces nonsense pairings
        like "subcognitive ~ gpu". Text-based similarity is noisier
        for short single-word concepts but produces semantically
        plausible pairings for concepts with multi-word names or
        definitions.
        """
        if not self.embeddings.has_embeddings:
            return []

        proposals: list[tuple[str, str, float]] = []
        seen: set[tuple[str, str]] = set()

        for concept_name in self.embeddings._concept_names:
            if len(proposals) >= self.max_per_session:
                break

            # Skip concepts that have been spilled to the archive
            # since the embeddings were last refreshed. The embedding
            # matrix may still contain their vectors, but they're no
            # longer in working memory — proposing edges to them
            # would recall them (via add_edge → _resolve) as a side
            # effect, which is not what sleep consolidation should do.
            if concept_name not in self.network._concepts:
                continue

            # Build a text query from the concept name and definition.
            # This searches the TF-IDF dimensions only (query_has_spectral=False),
            # avoiding the polluted graph-spectral component.
            query = self._concept_text_query(concept_name)
            if not query:
                continue

            similar = self.embeddings.find_similar_to_text(
                query,
                k=self.max_per_concept,
                threshold=self.threshold,
                exclude=concept_name,
            )

            count_for_this = 0
            for neighbor, score in similar:
                if count_for_this >= self.max_per_concept:
                    break
                if len(proposals) >= self.max_per_session:
                    break

                # Skip the query concept itself (find_similar_to_text may
                # return it since the query includes the concept name).
                if neighbor == concept_name:
                    continue

                # Skip neighbors that have been spilled to the archive.
                # Same reasoning as above — don't recall during sleep.
                if neighbor not in self.network._concepts:
                    continue

                # Canonical ordering (avoid a→b and b→a duplicates)
                a, b = sorted([concept_name, neighbor])
                pair = (a, b)
                if pair in seen:
                    continue
                seen.add(pair)

                # Skip if any edge already exists between them
                if self._has_any_edge(concept_name, neighbor):
                    continue

                proposals.append((concept_name, neighbor, score))
                count_for_this += 1

        return proposals

    def _concept_text_query(self, concept_name: str) -> str:
        """Build a text query for a concept using name + definition."""
        parts = [concept_name.replace("_", " ")]
        concept = self.network.get_concept(concept_name)
        if concept:
            definition = concept.properties.get("definition", "")
            if definition and definition != "NO DEF":
                parts.append(definition)
            # Aliases are also useful text signals.
            for alias in getattr(concept, "aliases", []):
                if alias and alias != concept_name:
                    parts.append(alias.replace("_", " "))
        return " ".join(p for p in parts if p)

    def accept_proposals(
        self,
        proposals: list[tuple[str, str, float]] | None = None,
    ) -> int:
        """Write proposed edges into the concept network.

        Args:
            proposals: The proposals to accept. If None, runs
                       propose_edges() first.

        Returns:
            Number of edges added.
        """
        if proposals is None:
            proposals = self.propose_edges()

        added = 0
        for concept_a, concept_b, score in proposals:
            # Double-check the concepts still exist in working memory.
            # Use _concepts directly rather than get_concept (which
            # would recall from the archive as a side effect —
            # sleep consolidation should not pull concepts back into
            # working memory).
            if concept_a not in self.network._concepts:
                continue
            if concept_b not in self.network._concepts:
                continue

            # Add the edge with origin="discovered"
            self.network.add_edge(
                concept_a,
                concept_b,
                RelationType.SIMILAR_TO,
                score,
                origin="discovered",
            )
            added += 1

        return added

    def discover_and_accept(self) -> int:
        """One-shot: propose edges and write them into the network.

        Convenience method for the sleep consolidation phase.
        """
        return self.accept_proposals(self.propose_edges())

    def _has_any_edge(self, concept_a: str, concept_b: str) -> bool:
        """Check if any edge exists between two concepts (either direction).

        Uses the network's edge indices directly rather than
        ``get_edges`` (which resolves aliases and allocates new lists
        on every call). This is called once per candidate pair during
        ``propose_edges``, so it must be fast.
        """
        # Outgoing: concept_a → concept_b
        for edge in self.network._edge_index.get(concept_a, ()):
            if edge.target == concept_b:
                return True
        # Incoming: concept_b → concept_a
        for edge in self.network._reverse_index.get(concept_a, ()):
            if edge.source == concept_b:
                return True
        return False
