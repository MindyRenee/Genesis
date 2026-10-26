"""Regression tests for holographic associative reconstruction."""

import random

from temporal_lobe.concepts.holographic import HolographicGraph


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
    metrics = graph.association_fidelity([("a", "rel", "b", 1.0)], top_k=1)

    assert metrics["n_evaluated"] == 1.0
    assert graph.edge_count == before


def test_journal_dedups_and_accumulates_weight() -> None:
    """Repeated adds superpose repeatedly but journal once."""
    graph = HolographicGraph(dim=128, n_buckets=4)
    graph.add("a", "rel", "b", weight=0.4)
    graph.add("a", "rel", "b", weight=0.4)
    graph.add("a", "rel", "b", weight=0.4)

    assert graph.edge_count == 3
    assert graph.distinct_edges == 1
    assert graph._journal[("a", "rel", "b")] == 1.0


def test_journal_cap_evicts_weakest() -> None:
    """Over-cap journals drop the weakest entries, keep the strong."""
    graph = HolographicGraph(dim=128, n_buckets=4, journal_cap=10)
    for i in range(9):
        graph.add("s", "rel", f"weak_{i}", weight=0.1)
    for i in range(5):
        graph.add("s", "rel", f"strong_{i}", weight=0.9)

    assert graph.distinct_edges <= 10
    kept = {t for (_s, _r, t) in graph._journal}
    assert all(f"strong_{i}" in kept for i in range(5))


def test_rebuild_from_journal_restores_fidelity() -> None:
    """The saturation regression: more distinct edges than the journal
    can hold flood the memory — journal-evicted bindings stay bound at
    full strength as pure noise. Rebuilding re-encodes only the
    journaled set and recall recovers. This is the failure mode of the
    live 1.9M-binding graph (distinct LTM co-occurrence pairs).
    """
    rng = random.Random(0)
    # Few buckets keeps the overload real — at the production 64 the
    # same edge volume barely saturates, which is the point of 64.
    graph = HolographicGraph(n_buckets=4, journal_cap=2000)
    concepts = [f"c{i}" for i in range(400)]
    strong = [(rng.choice(concepts), "related_to", rng.choice(concepts), 0.9) for _ in range(150)]
    weak = [(rng.choice(concepts), "related_to", rng.choice(concepts), 0.3) for _ in range(8000)]
    for s, r, t, w in strong + weak:
        if s != t:
            graph.add(s, r, t, w)
    strong = [e for e in strong if e[0] != e[2]]

    sample = strong[:80]
    rebuilt = graph.rebuild_from_journal()
    after = graph.association_fidelity(sample, top_k=10)["recall_at_k"]

    assert rebuilt == graph.distinct_edges <= 2000
    # Under source-bound encoding the journaled set must stay
    # retrievable after a rebuild — the ghost bindings are gone and
    # what remains is the clean deduplicated association set.
    assert after >= 0.7, f"journaled recall degraded after rebuild: {after:.3f}"


def test_rebuild_from_journal_empty_is_noop() -> None:
    graph = HolographicGraph(dim=128, n_buckets=4)
    assert graph.rebuild_from_journal() == 0


def test_rebuild_from_edges_replaces_journal() -> None:
    graph = HolographicGraph(dim=128, n_buckets=4)
    graph.add("x", "rel", "y", weight=0.5)
    graph.rebuild_from_edges(
        [("a", "rel", "b", 0.9), ("a", "rel", "b", 0.3), ("c", "rel", "d", 0.7)]
    )
    assert graph.edge_count == 2
    assert graph.distinct_edges == 2
    assert graph._journal[("a", "rel", "b")] == 0.9  # max weight wins


def test_journal_survives_save_load(tmp_path) -> None:
    path = str(tmp_path / "hg.npz")
    graph = HolographicGraph(dim=128, n_buckets=4)
    graph.add("dog", "related_to", "cat", weight=0.9)
    graph.add("dog", "related_to", "cat", weight=0.9)
    graph.save(path)

    loaded = HolographicGraph(dim=128, n_buckets=4)
    assert loaded.load(path)
    assert loaded.distinct_edges == 1
    assert loaded._journal[("dog", "related_to", "cat")] == 1.0


def test_legacy_npz_without_journal_loads_intact(tmp_path) -> None:
    """Pre-journal files have no journal arrays — load keeps the memory
    and the rebuild guard refuses an empty-journal wipe."""
    import numpy as np

    path = str(tmp_path / "hg.npz")
    graph = HolographicGraph(dim=128, n_buckets=4)
    graph.add("dog", "related_to", "cat", weight=0.9)
    graph.save(path)

    # Strip journal keys to emulate a legacy file.
    data = dict(np.load(path, allow_pickle=True).items())
    np.savez(
        path,
        **{k: v for k, v in data.items() if not k.startswith("j_")},
    )

    loaded = HolographicGraph(dim=128, n_buckets=4)
    assert loaded.load(path)
    assert loaded.edge_count == 1
    assert loaded.distinct_edges == 0
    assert loaded.rebuild_from_journal() == 0
    assert loaded.edge_count == 1  # memory intact
