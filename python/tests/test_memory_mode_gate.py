"""Tests for the Hasselmo acetylcholine memory-mode gate.

Hasselmo's model: the same cortical/hippocampal circuitry switches
between encode mode (high ACh — strong afferent write, suppressed
recurrent transmission to prevent interference) and retrieve mode
(low ACh — released recurrent collaterals). The body computes the
mode as NeuroSummary encoding/consolidation/retrieval weights;
ConceptNetwork.set_memory_mode maps them to multiplicative gains
(0.5 baseline → 1.0 open).
"""

import pytest
from temporal_lobe.concepts.network import ConceptNetwork


@pytest.fixture
def chain() -> ConceptNetwork:
    """cat → dog → bone, a two-hop retrieval chain."""
    n = ConceptNetwork()
    for c in ("cat", "dog", "bone"):
        n.add_concept(c)
    n.add_edge("cat", "dog", "similar_to", weight=1.0)
    n.add_edge("dog", "bone", "eats", weight=1.0)
    return n


def test_baseline_weights_leave_spread_open(chain: ConceptNetwork) -> None:
    """Neutral drive (0.5) is the open gate — unchanged spread."""
    base = chain.spread_activation(["cat"], amount=1.0, depth=2)
    assert base["dog"] > 0.4
    assert base["bone"] > 0.0


def test_encode_mode_suppresses_recurrent_spread(chain: ConceptNetwork) -> None:
    """High-ACh encode mode blocks the recurrent cascade — the afferent
    seed activates but nothing propagates through association fibers."""
    chain.set_memory_mode(encoding=1.0, retrieval=0.0)
    out = chain.spread_activation(["cat"], amount=1.0, depth=2)
    assert out["cat"] == pytest.approx(1.0)
    assert out["dog"] == pytest.approx(0.0)
    assert out.get("bone", 0.0) == pytest.approx(0.0)


def test_retrieve_mode_amplifies_spread(chain: ConceptNetwork) -> None:
    """Low-ACh retrieve mode releases recurrent collaterals — the same
    stimulus carries further through the graph."""
    chain.set_memory_mode(encoding=0.5, retrieval=1.0)
    out = chain.spread_activation(["cat"], amount=1.0, depth=2)
    base = ConceptNetwork()
    for c in ("cat", "dog", "bone"):
        base.add_concept(c)
    base.add_edge("cat", "dog", "similar_to", weight=1.0)
    base.add_edge("dog", "bone", "eats", weight=1.0)
    ref = base.spread_activation(["cat"], amount=1.0, depth=2)
    assert out["dog"] > ref["dog"]
    assert out["bone"] > ref["bone"]


def test_suppressed_encoding_blocks_edge_writes(chain: ConceptNetwork) -> None:
    """No ACh drive → the synaptic increment itself is gated to zero."""
    chain.set_memory_mode(encoding=0.0, retrieval=0.5)
    edge = chain.add_edge("cat", "bone", "similar_to", weight=0.8)
    assert edge is not None
    assert edge.weight == pytest.approx(0.0)


def test_amplified_encoding_scales_writes(chain: ConceptNetwork) -> None:
    """Full encoding drive doubles the plasticity increment."""
    chain.set_memory_mode(encoding=1.0, retrieval=0.5)
    edge = chain.add_edge("cat", "bone", "similar_to", weight=0.6)
    assert edge is not None
    assert edge.weight == pytest.approx(1.0)


def test_repeated_add_under_suppression_does_not_strengthen(
    chain: ConceptNetwork,
) -> None:
    """Re-exposure during encode-suppressed mode cannot strengthen an
    existing edge — interference cannot masquerade as learning."""
    chain.set_memory_mode(encoding=0.0, retrieval=0.5)
    e1 = chain.add_edge("cat", "dog", "similar_to", weight=0.4)
    assert e1 is not None
    w1 = e1.weight
    e2 = chain.add_edge("cat", "dog", "similar_to", weight=0.4)
    assert e2 is not None
    assert e2.weight == pytest.approx(w1)


def test_nonfinite_weights_fall_back_to_open(chain: ConceptNetwork) -> None:
    """A corrupt summary degrades to the open gate, not a crash."""
    chain.set_memory_mode(encoding=float("nan"), retrieval=float("inf"))
    out = chain.spread_activation(["cat"], amount=1.0, depth=2)
    assert out["dog"] > 0.0
