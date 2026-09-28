"""Regression tests for holographic associative reconstruction."""

from genesis_cognitive.concepts.holographic import HolographicGraph


def test_holographic_fidelity_measures_association_retrieval() -> None:
    graph = HolographicGraph(dim=256, n_buckets=8)
    edges = [
        ("a", "rel", "b", 1.0),
        ("c", "rel", "d", 1.0),
    ]
    graph.add_edges(edges)

    metrics = graph.association_fidelity(edges, top_k=2)

    assert metrics["n_evaluated"] == 2.0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert 0.0 <= metrics["mean_reciprocal_rank"] <= 1.0
    assert metrics["mean_target_similarity"] > 0.0


def test_holographic_fidelity_does_not_retain_evaluation_edges() -> None:
    graph = HolographicGraph(dim=128, n_buckets=4)
    graph.add("a", "rel", "b")

    before = graph.edge_count
    metrics = graph.association_fidelity(
        [("a", "rel", "b", 1.0)], top_k=1
    )

    assert metrics["n_evaluated"] == 1.0
    assert graph.edge_count == before
