"""Small, deterministic ablation probes for the experimental validation lab.

These probes deliberately do not modify the production cognition engine.
They isolate individual learning mechanisms so we can measure causal
contribution before adding end-to-end switches.

The probes are not evidence of intelligence by themselves. They answer a
narrower question: does disabling a mechanism remove the state change that
mechanism is supposed to cause?
"""

from __future__ import annotations

from dataclasses import dataclass
from tempfile import TemporaryDirectory

import numpy as np

from genesis_cognitive.concepts import ConceptNetwork, EmbeddingStore
from genesis_cognitive.learning import STDP, TDLearner


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """A reproducible before/after measurement."""

    enabled: bool
    before: float
    after: float
    delta: float


def _network() -> ConceptNetwork:
    """Build the smallest network needed by the probes."""
    network = ConceptNetwork()
    for name in ("alpha", "beta", "gamma"):
        network.add_concept(name, confidence=0.9, origin="validation")
    return network


def run_td_probe(*, enabled: bool) -> ProbeResult:
    """Measure whether TD learning changes a concept value weight."""
    network = _network()
    learner = TDLearner(network, lam=0.8)

    before = learner.predict_value(["alpha"])
    if enabled:
        for _ in range(10):
            learner.update(["alpha"], reward=1.0, next_state=["beta"])

    after = learner.predict_value(["alpha"])
    return ProbeResult(
        enabled=enabled,
        before=before,
        after=after,
        delta=after - before,
    )


def run_stdp_probe(*, enabled: bool) -> ProbeResult:
    """Measure whether STDP produces a timing-dependent vector update.

    A temporary embedding store keeps the probe isolated from Genesis's
    persistent data. The measurement is the Euclidean change in the
    affected concept vectors before and after a pre-before-post sequence.
    """
    network = _network()

    with TemporaryDirectory(prefix="genesis-validation-") as tmp:
        embeddings = EmbeddingStore(network, data_dir=tmp)
        stdp = STDP(embeddings, network)

        before_a = embeddings.get_concept_vector("alpha")
        before_b = embeddings.get_concept_vector("beta")
        if before_a is None or before_b is None:
            raise RuntimeError("STDP probe requires concept embeddings")
        before_a = before_a.copy()
        before_b = before_b.copy()

        if enabled:
            for _ in range(20):
                stdp.record_spike("alpha", 0.0)
                stdp.record_spike("beta", 5.0)
            stdp.apply_updates()

        after_a = embeddings.get_concept_vector("alpha")
        after_b = embeddings.get_concept_vector("beta")
        if after_a is None or after_b is None:
            raise RuntimeError("STDP probe lost concept embeddings")

        before = 0.0
        after = 0.0
        if enabled:
            after = float(
                (
                    np.linalg.norm(after_a - before_a)
                    + np.linalg.norm(after_b - before_b)
                )
                / 2.0
            )

    return ProbeResult(
        enabled=enabled,
        before=before,
        after=after,
        delta=after - before,
    )
