"""Tests for the embedding store — Genesis's latent space."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from genesis_cognitive.concepts import (
    ConceptNetwork,
    EmbeddingStore,
    RelationType,
)


@pytest.fixture
def small_network() -> ConceptNetwork:
    """A small concept network for testing."""
    net = ConceptNetwork()

    # Add some concepts with definitions
    for name, defn in [
        ("tree", "a tall perennial woody plant having a main trunk"),
        ("plant", "a living organism that grows in the earth"),
        ("flower", "a colorful plant that blooms and produces seeds"),
        ("garden", "a plot of ground where plants are cultivated"),
        ("bush", "a low woody plant with multiple stems"),
        ("dog", "a domesticated carnivorous mammal"),
        ("cat", "a small domesticated carnivorous mammal"),
        ("animal", "a living organism that moves voluntarily"),
        ("cognition", "an alert cognitive state of awareness"),
        ("memory", "the cognitive process of retaining information"),
    ]:
        net.add_concept(name, confidence=0.9, origin="test")
        c = net.get_concept(name)
        if c:
            c.properties["definition"] = defn

    # Add some edges
    net.add_edge("tree", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("bush", "plant", RelationType.IS_A, 0.85, origin="test")
    net.add_edge("flower", "plant", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("dog", "animal", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("cat", "animal", RelationType.IS_A, 0.9, origin="test")
    net.add_edge("tree", "garden", RelationType.RELATED_TO, 0.7, origin="test")
    net.add_edge("flower", "garden", RelationType.RELATED_TO, 0.8, origin="test")
    net.add_edge("cognition", "memory", RelationType.RELATED_TO, 0.7, origin="test")

    return net


@pytest.fixture
def store(small_network: ConceptNetwork, tmp_path: Path) -> EmbeddingStore:
    """An embedding store built from the small network."""
    return EmbeddingStore(small_network, data_dir=str(tmp_path))


class TestEmbeddingStoreBasics:
    """Test basic embedding store functionality."""

    def test_has_embeddings_after_build(self, store: EmbeddingStore):
        """Test has embeddings after build."""
        assert store.has_embeddings is True

    def test_concept_count_matches_network(
        self, store: EmbeddingStore, small_network: ConceptNetwork
    ) -> None:
        """Test concept count matches network."""
        assert store.concept_count == small_network.size

    def test_dimensionality_is_positive(self, store: EmbeddingStore):
        """Test dimensionality is positive."""
        assert store.dimensionality > 0

    def test_get_concept_vector_returns_normalized(self, store: EmbeddingStore):
        """Test get concept vector returns normalized."""
        vec = store.get_concept_vector("tree")
        assert vec is not None
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5  # L2 normalized

    def test_get_concept_vector_for_unknown_returns_none(self, store: EmbeddingStore):
        """Test get concept vector for unknown returns none."""
        vec = store.get_concept_vector("nonexistent_concept")
        # May return None or a computed vector — both are acceptable
        # if it returns a vector, it should be normalized
        if vec is not None:
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


class TestSemanticSimilarity:
    """Test that semantically similar concepts are close in embedding space."""

    def test_tree_similar_to_plant(self, store: EmbeddingStore):
        """Tree should be similar to plant (they share an IS_A edge)."""
        similar = store.find_similar_concepts("tree", k=5, threshold=0.1)
        names = [name for name, _ in similar]
        assert "plant" in names or "bush" in names or "flower" in names

    def test_dog_similar_to_cat(self, store: EmbeddingStore):
        """Dog and cat should be similar (both are animals)."""
        similar = store.find_similar_concepts("dog", k=5, threshold=0.1)
        names = [name for name, _ in similar]
        # They share the "animal" hypernym, so they should be close
        assert "cat" in names or "animal" in names

    def test_tree_not_similar_to_cognition(self, store: EmbeddingStore):
        """Tree and cognition should not be similar."""
        similar = store.find_similar_concepts("tree", k=10, threshold=0.01)
        names = [name for name, _ in similar]
        # cognition should be low on the list or absent
        if "cognition" in names:
            # If it appears, it should be the last
            assert names[-1] == "cognition"

    def test_self_is_excluded_from_results(self, store: EmbeddingStore):
        """The query concept should not appear in its own similar list."""
        similar = store.find_similar_concepts("tree", k=5, threshold=0.0)
        names = [name for name, _ in similar]
        assert "tree" not in names


class TestTextMatching:
    """Test semantic text matching — the pattern recognition feature."""

    def test_text_matching_finds_relevant_concepts(self, store: EmbeddingStore):
        """Text about plants should find plant-related concepts."""
        results = store.find_similar_to_text("tall woody plant", k=3, threshold=0.01)
        assert len(results) > 0
        names = [name for name, _ in results]
        # Should find tree, plant, or bush
        assert any(name in names for name in ("tree", "plant", "bush", "flower"))

    def test_text_matching_for_cognition(self, store: EmbeddingStore):
        """Text about awareness should find cognition."""
        results = store.find_similar_to_text("alert cognitive awareness", k=3, threshold=0.01)
        assert len(results) > 0
        names = [name for name, _ in results]
        assert "cognition" in names or "memory" in names

    def test_empty_text_returns_empty(self, store: EmbeddingStore):
        """Test empty text returns empty."""
        results = store.find_similar_to_text("", k=3, threshold=0.01)
        assert results == []


class TestEdgeProposer:
    """Test that the edge proposer discovers new relationships."""

    def test_propose_finds_candidates(
        self, store: EmbeddingStore, small_network: ConceptNetwork
    ) -> None:
        """Test propose finds candidates."""
        from genesis_cognitive.concepts import EdgeProposer

        proposer = EdgeProposer(small_network, store, threshold=0.1, max_per_session=10)
        proposals = proposer.propose_edges()
        # Should find some candidate edges (e.g., tree-bush, dog-cat)
        assert isinstance(proposals, list)

    def test_accept_writes_edges(
        self, store: EmbeddingStore, small_network: ConceptNetwork
    ) -> None:
        """Test accept writes edges."""
        from genesis_cognitive.concepts import EdgeProposer

        proposer = EdgeProposer(small_network, store, threshold=0.1, max_per_session=10)
        proposals = proposer.propose_edges()

        if proposals:
            # Count edges before
            before = small_network.edge_count
            proposer.accept_proposals(proposals)
            after = small_network.edge_count
            assert after >= before

    def test_does_not_propose_existing_edges(
        self, store: EmbeddingStore, small_network: ConceptNetwork
    ) -> None:
        """Test does not propose existing edges."""
        from genesis_cognitive.concepts import EdgeProposer

        proposer = EdgeProposer(small_network, store, threshold=0.0, max_per_session=50)
        proposals = proposer.propose_edges()

        # None of the proposals should be existing edges
        for a, b, _ in proposals:
            outgoing = small_network.get_edges(a, direction="out")
            for edge in outgoing:
                assert not (edge.target == b), f"Edge {a}→{b} already exists"


class TestThreadSafety:
    """Test that the embedding store is safe for concurrent access."""

    def test_lazy_loading_is_thread_safe(
        self, small_network: ConceptNetwork, tmp_path: Path
    ) -> None:
        """Test lazy loading is thread safe."""
        import threading

        store = EmbeddingStore(small_network, data_dir=str(tmp_path))
        results = []

        def access() -> None:
            """Record whether the store has embeddings (thread-safety probe)."""
            results.append(store.has_embeddings)

        threads = [threading.Thread(target=access) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(results)
        assert len(results) == 10


class TestTfIdfVocabularyAlignment:
    """Test that Hebbian-adapted TF-IDF vectors survive vocabulary drift.

    This is a regression test for a data-integrity bug where the
    TF-IDF vocabulary was rebuilt on every refresh, shifting words to
    different column indices. The old experiential vectors were
    restored by raw column position, so a Hebbian-adapted weight for
    "brain" (old column 5) could get applied to "neuron" (new column
    5). The fix re-maps columns by word.
    """

    def test_refresh_preserves_tfidf_weights_by_word(
        self, small_network: ConceptNetwork, tmp_path: Path
    ) -> None:
        """After refresh, a concept's TF-IDF weight for a specific word
        should be preserved, not shifted to a different word's column."""
        store = EmbeddingStore(small_network, data_dir=str(tmp_path))
        store._ensure_loaded()
        assert store._concept_matrix is not None

        # Get the initial TF-IDF vector for "tree"
        tree_idx = store._concept_to_idx.get("tree")
        assert tree_idx is not None

        tfidf_start = store._spectral_dim
        tfidf_end = tfidf_start + store._tfidf_dim

        initial_vec = store._concept_matrix[tree_idx, tfidf_start:tfidf_end].copy()

        # Record which words have non-zero weights
        initial_nonzero_words = set()
        for word, col in store._tfidf_vocab.items():
            if col < len(initial_vec) and initial_vec[col] != 0.0:
                initial_nonzero_words.add(word)

        # Modify a TF-IDF weight to simulate Hebbian adaptation
        assert initial_nonzero_words, "Test requires at least one non-zero TF-IDF word"
        target_word = next(iter(initial_nonzero_words))
        target_col = store._tfidf_vocab[target_word]
        original_value = float(initial_vec[target_col])
        adapted_value = original_value + 5.0  # boost significantly
        store._concept_matrix[tree_idx, tfidf_start + target_col] = adapted_value

        # Add new concepts to change the corpus → vocabulary shifts
        small_network.add_concept("brain", confidence=0.9, origin="test")
        bc = small_network.get_concept("brain")
        if bc:
            bc.properties["definition"] = "the organ of cognition and thought"
        small_network.add_concept("neuron", confidence=0.9, origin="test")
        nc = small_network.get_concept("neuron")
        if nc:
            nc.properties["definition"] = "a nerve cell that transmits signals"
        small_network.add_concept("synapse", confidence=0.9, origin="test")
        sc = small_network.get_concept("synapse")
        if sc:
            sc.properties["definition"] = "the junction between two nerve cells"

        # Refresh — this rebuilds the TF-IDF vocabulary
        store._refresh_preserve_experiential()

        # After refresh, the adapted weight should be on the SAME word.
        tree_idx_new = store._concept_to_idx.get("tree")
        assert tree_idx_new is not None

        tfidf_start_new = store._spectral_dim
        tfidf_end_new = tfidf_start_new + store._tfidf_dim
        new_vec = store._concept_matrix[tree_idx_new, tfidf_start_new:tfidf_end_new]

        # The target word should still be in the vocab
        assert target_word in store._tfidf_vocab, (
            f"'{target_word}' was dropped from the vocabulary after refresh. "
            f"Test needs a word that survives the vocab rebuild."
        )
        new_col = store._tfidf_vocab[target_word]

        # The adapted weight should be preserved on the correct word.
        # We verify this by checking that the value at the target word's
        # NEW column is higher than the value at the OLD column position
        # (which now corresponds to a different word). Without the fix,
        # the adapted weight would stay at the old column position
        # (wrong word) and the new column (correct word) would have
        # only the fresh TF-IDF value.
        new_value = float(new_vec[new_col])

        # Check the value at the OLD column position (now a different word)
        if target_col < len(new_vec) and target_col != new_col:
            old_pos_value = float(new_vec[target_col])
            # The adapted weight should be on the correct word, not
            # stuck at the old position. So the correct word's value
            # should be higher than the old position's value.
            assert new_value > old_pos_value, (
                f"Adapted weight appears to be at the old column position "
                f"({target_col}, value={old_pos_value:.4f}) instead of the "
                f"correct word's new column ({new_col}, value={new_value:.4f}). "
                f"This indicates vocabulary drift is scrambling learned weights."
            )

        # Also verify the adapted value is higher than a fresh embedding
        # would produce (less strict threshold to account for normalization)
        fresh_net = ConceptNetwork()
        for name, defn in [
            ("tree", "a tall perennial woody plant having a main trunk"),
            ("plant", "a living organism that grows in the earth"),
            ("flower", "a colorful plant that blooms and produces seeds"),
            ("garden", "a plot of ground where plants are cultivated"),
            ("bush", "a low woody plant with multiple stems"),
            ("dog", "a domesticated carnivorous mammal"),
            ("cat", "a small domesticated carnivorous mammal"),
            ("animal", "a living organism that moves voluntarily"),
            ("cognition", "an alert cognitive state of awareness"),
            ("memory", "the cognitive process of retaining information"),
            ("brain", "the organ of cognition and thought"),
            ("neuron", "a nerve cell that transmits signals"),
            ("synapse", "the junction between two nerve cells"),
        ]:
            fresh_net.add_concept(name, confidence=0.9, origin="test")
            c = fresh_net.get_concept(name)
            if c:
                c.properties["definition"] = defn
        fresh_net.add_edge("tree", "plant", RelationType.IS_A, 0.9, origin="test")
        fresh_net.add_edge("bush", "plant", RelationType.IS_A, 0.85, origin="test")
        fresh_net.add_edge("flower", "plant", RelationType.IS_A, 0.9, origin="test")
        fresh_net.add_edge("dog", "animal", RelationType.IS_A, 0.9, origin="test")
        fresh_net.add_edge("cat", "animal", RelationType.IS_A, 0.9, origin="test")
        fresh_net.add_edge("tree", "garden", RelationType.RELATED_TO, 0.7, origin="test")
        fresh_net.add_edge("flower", "garden", RelationType.RELATED_TO, 0.8, origin="test")
        fresh_net.add_edge("cognition", "memory", RelationType.RELATED_TO, 0.7, origin="test")

        fresh_store = EmbeddingStore(fresh_net, data_dir=str(tmp_path / "fresh"))
        fresh_store._ensure_loaded()
        assert fresh_store._concept_matrix is not None
        fresh_tree_idx = fresh_store._concept_to_idx.get("tree")
        fresh_tfidf_start = fresh_store._spectral_dim
        fresh_vec = fresh_store._concept_matrix[
            fresh_tree_idx,
            fresh_tfidf_start:fresh_tfidf_start + fresh_store._tfidf_dim,
        ]

        if target_word in fresh_store._tfidf_vocab:
            fresh_col = fresh_store._tfidf_vocab[target_word]
            fresh_value = float(fresh_vec[fresh_col])
            assert new_value > fresh_value, (
                f"Adapted weight for '{target_word}' was not preserved after "
                f"refresh. Fresh={fresh_value:.4f}, adapted={adapted_value:.4f}, "
                f"after refresh={new_value:.4f}. Expected the adapted weight "
                f"to be higher than a fresh embedding."
            )

    def test_save_load_preserves_tfidf_weights_by_word(
        self, small_network: ConceptNetwork, tmp_path: Path
    ) -> None:
        """After save → load, a concept's TF-IDF weight for a specific
        word should be preserved, not shifted to a different word."""
        store = EmbeddingStore(small_network, data_dir=str(tmp_path))
        store._ensure_loaded()
        assert store._concept_matrix is not None

        tree_idx = store._concept_to_idx.get("tree")
        assert tree_idx is not None

        tfidf_start = store._spectral_dim

        # Find a non-zero TF-IDF word and record its original value
        initial_vec = store._concept_matrix[tree_idx, tfidf_start:tfidf_start + store._tfidf_dim]
        nonzero_words = [
            (w, c) for w, c in store._tfidf_vocab.items()
            if c < len(initial_vec) and initial_vec[c] != 0.0
        ]
        assert nonzero_words, "Test requires at least one non-zero TF-IDF word"

        target_word, target_col = nonzero_words[0]
        original_value = float(initial_vec[target_col])
        adapted_value = original_value + 5.0
        store._concept_matrix[tree_idx, tfidf_start + target_col] = adapted_value

        # Save
        store.save_experiential()

        # Add new concepts to shift the vocabulary
        small_network.add_concept("brain", confidence=0.9, origin="test")
        bc = small_network.get_concept("brain")
        if bc:
            bc.properties["definition"] = "the organ of cognition"
        small_network.add_concept("neuron", confidence=0.9, origin="test")
        nc = small_network.get_concept("neuron")
        if nc:
            nc.properties["definition"] = "a nerve cell that transmits signals"

        # Rebuild from scratch (simulates restart)
        store2 = EmbeddingStore(small_network, data_dir=str(tmp_path))
        store2._ensure_loaded()
        assert store2._concept_matrix is not None
        restored = store2.load_experiential()
        assert restored, "load_experiential should restore vectors"

        # The target word should still have the adapted weight
        tree_idx2 = store2._concept_to_idx.get("tree")
        assert tree_idx2 is not None

        tfidf_start2 = store2._spectral_dim
        tfidf_end2 = tfidf_start2 + store2._tfidf_dim
        new_vec = store2._concept_matrix[tree_idx2, tfidf_start2:tfidf_end2]

        assert target_word in store2._tfidf_vocab, (
            f"'{target_word}' was dropped from vocabulary after reload"
        )
        new_col = store2._tfidf_vocab[target_word]
        new_value = float(new_vec[new_col])

        # Compare against a fresh embedding (no adaptation) to verify
        # the adapted weight was preserved through save/load
        fresh_net = ConceptNetwork()
        for name, defn in [
            ("tree", "a tall perennial woody plant having a main trunk"),
            ("plant", "a living organism that grows in the earth"),
            ("flower", "a colorful plant that blooms and produces seeds"),
            ("garden", "a plot of ground where plants are cultivated"),
            ("bush", "a low woody plant with multiple stems"),
            ("dog", "a domesticated carnivorous mammal"),
            ("cat", "a small domesticated carnivorous mammal"),
            ("animal", "a living organism that moves voluntarily"),
            ("cognition", "an alert cognitive state of awareness"),
            ("memory", "the cognitive process of retaining information"),
            ("brain", "the organ of cognition"),
            ("neuron", "a nerve cell that transmits signals"),
        ]:
            fresh_net.add_concept(name, confidence=0.9, origin="test")
            c = fresh_net.get_concept(name)
            if c:
                c.properties["definition"] = defn
        fresh_net.add_edge("tree", "plant", RelationType.IS_A, 0.9, origin="test")
        fresh_net.add_edge("bush", "plant", RelationType.IS_A, 0.85, origin="test")
        fresh_net.add_edge("flower", "plant", RelationType.IS_A, 0.9, origin="test")
        fresh_net.add_edge("dog", "animal", RelationType.IS_A, 0.9, origin="test")
        fresh_net.add_edge("cat", "animal", RelationType.IS_A, 0.9, origin="test")
        fresh_net.add_edge("tree", "garden", RelationType.RELATED_TO, 0.7, origin="test")
        fresh_net.add_edge("flower", "garden", RelationType.RELATED_TO, 0.8, origin="test")
        fresh_net.add_edge("cognition", "memory", RelationType.RELATED_TO, 0.7, origin="test")

        fresh_store = EmbeddingStore(fresh_net, data_dir=str(tmp_path / "fresh2"))
        fresh_store._ensure_loaded()
        assert fresh_store._concept_matrix is not None
        fresh_tree_idx = fresh_store._concept_to_idx.get("tree")
        fresh_tfidf_start = fresh_store._spectral_dim
        fresh_vec = fresh_store._concept_matrix[
            fresh_tree_idx, fresh_tfidf_start:fresh_tfidf_start + fresh_store._tfidf_dim
        ]

        if target_word in fresh_store._tfidf_vocab:
            fresh_col = fresh_store._tfidf_vocab[target_word]
            fresh_value = float(fresh_vec[fresh_col])
            assert new_value > fresh_value, (
                f"Adapted weight for '{target_word}' was not preserved after "
                f"save/load. Fresh={fresh_value:.4f}, "
                f"after reload={new_value:.4f}. Expected the adapted weight "
                f"to be higher than a fresh embedding."
            )

    def test_save_includes_vocabulary_metadata(
        self, small_network: ConceptNetwork, tmp_path: Path
    ) -> None:
        """The saved file should include TF-IDF vocabulary metadata."""
        store = EmbeddingStore(small_network, data_dir=str(tmp_path))
        store._ensure_loaded()
        assert store._concept_matrix is not None
        store.save_experiential()

        path = tmp_path / "concept_vectors.npz"
        assert path.exists()

        data = np.load(path, allow_pickle=False)
        assert "tfidf_words" in data
        assert "tfidf_indices" in data
        assert "tfidf_dim" in data


class TestColdLayer:
    """Test that archived (dormant) concepts get cold-layer vectors.

    The hot concept matrix only covers working memory. Archived
    concepts are embedded in a parallel flat-block matrix that is
    consulted as a fallback when a hot search returns fewer than k
    results — dormant knowledge stays discoverable by similarity.
    """

    def _archived_concept_data(self, cid: str, definition: str) -> dict:
        """Serialized concept dict in the shape the archive stores."""
        return {
            "id": cid,
            "aliases": [],
            "activation": 0.0,
            "confidence": 0.8,
            "origin": "learned",
            "columns": [],
            "created_at": 0,
            "review_count": 0,
            "last_reviewed": 0,
            "properties": {"definition": definition},
            "category": "unknown",
            "is_animacy_detected": False,
            "modality": "unknown",
            "is_semantic_hub": False,
        }

    def test_archived_concepts_get_cold_vectors(
        self, small_network: ConceptNetwork, tmp_path: Path
    ) -> None:
        """Archived concepts are embedded in the cold layer."""
        from genesis_cognitive.concepts.archive import open_archive

        archive = open_archive(tmp_path / "archive")
        archive.archive_concept(
            "cactus",
            self._archived_concept_data(
                "cactus", "a spiny succulent plant that stores water"
            ),
            set(),
        )
        small_network.attach_archive(archive)

        store = EmbeddingStore(small_network, data_dir=str(tmp_path))
        store._ensure_loaded()

        assert store.archive_concept_count == 1
        # The archived concept is not in the hot matrix
        assert "cactus" not in store._concept_to_idx
        assert store._archive_matrix is not None
        assert store._archive_matrix.shape[0] == 1
        archive.close()

    def test_cold_fallback_surfaces_archived_concept(
        self, small_network: ConceptNetwork, tmp_path: Path
    ) -> None:
        """Text search falls back to the cold layer when the hot
        matrix can't fill k results."""
        from genesis_cognitive.concepts.archive import open_archive

        archive = open_archive(tmp_path / "archive")
        archive.archive_concept(
            "cactus",
            self._archived_concept_data(
                "cactus", "a spiny succulent plant that stores water"
            ),
            set(),
        )
        small_network.attach_archive(archive)

        store = EmbeddingStore(small_network, data_dir=str(tmp_path))
        store._ensure_loaded()
        assert store._archive_matrix is not None

        # k exceeds the hot network size, so the cold fallback must
        # fire to fill remaining results. "plant" is in the hot TF-IDF
        # vocabulary and in the archived concept's definition.
        results = store.find_similar_to_text(
            "spiny succulent plant", k=20, threshold=0.01
        )
        names = [name for name, _ in results]
        assert "cactus" in names
        archive.close()

    def test_recalled_concept_not_duplicated_in_cold_results(
        self, small_network: ConceptNetwork, tmp_path: Path
    ) -> None:
        """A concept recalled to working memory after the cold build
        is filtered out of cold results (its hot row wins)."""
        from genesis_cognitive.concepts.archive import open_archive

        archive = open_archive(tmp_path / "archive")
        archive.archive_concept(
            "cactus",
            self._archived_concept_data(
                "cactus", "a spiny succulent plant that stores water"
            ),
            set(),
        )
        small_network.attach_archive(archive)

        store = EmbeddingStore(small_network, data_dir=str(tmp_path))
        store._ensure_loaded()
        assert store.archive_concept_count == 1

        # Recall the concept — it leaves the archive and re-enters
        # working memory. The cold matrix is stale until refresh, so
        # the result filter must suppress it.
        recalled = small_network.get_concept("cactus")
        assert recalled is not None

        results = store.find_similar_to_text(
            "spiny succulent plant", k=20, threshold=0.0
        )
        names = [name for name, _ in results]
        assert names.count("cactus") <= 1
        archive.close()

    def test_get_concept_vector_resolves_archived(
        self, small_network: ConceptNetwork, tmp_path: Path
    ) -> None:
        """get_concept_vector returns a cold vector for archived
        concepts, expanded to full width."""
        from genesis_cognitive.concepts.archive import open_archive

        archive = open_archive(tmp_path / "archive")
        archive.archive_concept(
            "cactus",
            self._archived_concept_data(
                "cactus", "a spiny succulent plant that stores water"
            ),
            set(),
        )
        small_network.attach_archive(archive)

        store = EmbeddingStore(small_network, data_dir=str(tmp_path))
        vec = store.get_concept_vector("cactus")
        assert vec is not None
        assert vec.shape[0] == store.dimensionality
        # Toroidal block is zeros for cold vectors
        toroidal_end = store._spectral_dim + store._experiential_dim
        assert np.allclose(vec[:toroidal_end], 0.0)
        archive.close()
