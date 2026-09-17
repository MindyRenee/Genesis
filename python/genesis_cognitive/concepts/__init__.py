"""Concepts bundle — the concept network and its supporting machinery.

This package bundles Genesis's knowledge-representation layer into a
coherent unit: the symbolic concept graph, its sub-symbolic embedding
space, the holographic (vector-symbolic) graph, topology metrics,
edge proposal, and the cold-storage archive.

Subsystems:
    ConceptNetwork, Concept, Edge, RelationType, ConceptCategory,
    ConceptModality — the symbolic concept graph (``network.py``)
    EmbeddingStore — continuous vector space beneath the graph
        (``embeddings.py``)
    HolographicGraph, circular_convolve, circular_correlate —
        vector-symbolic binding for the graph (``holographic.py``)
    NetworkTopology — structural metrics over the graph
        (``topology.py``)
    EdgeProposer — proposes new edges between concepts
        (``edge_proposer.py``)
    ConceptArchive, open_archive — cold storage for pruned concepts
        (``archive.py``)
"""

from __future__ import annotations

from .archive import ConceptArchive, open_archive
from .classify import (
    _BRIDGE_ORIGINS,
    _FUNCTION_WORDS,
    QUALITY_THRESHOLD,
    _column_of,
    _normalize_id,
    detect_category,
    detect_modality,
    is_world_concept,
    strip_sense_suffix,
)
from .edge_proposer import EdgeProposer
from .embeddings import EmbeddingStore
from .holographic import HolographicGraph, circular_convolve, circular_correlate
from .network import ConceptNetwork
from .topology import NetworkTopology
from .types import (
    Concept,
    ConceptCategory,
    ConceptModality,
    Edge,
    RelationType,
    is_seed_origin,
)

__all__ = [
    "QUALITY_THRESHOLD",
    "_BRIDGE_ORIGINS",
    "_FUNCTION_WORDS",
    "Concept",
    "ConceptArchive",
    "ConceptCategory",
    "ConceptModality",
    "ConceptNetwork",
    "Edge",
    "EdgeProposer",
    "EmbeddingStore",
    "HolographicGraph",
    "NetworkTopology",
    "RelationType",
    "_column_of",
    "_normalize_id",
    "circular_convolve",
    "circular_correlate",
    "detect_category",
    "detect_modality",
    "is_seed_origin",
    "is_world_concept",
    "open_archive",
    "strip_sense_suffix",
]
