"""Consolidation engine — word-layer companion to ``src/daemon/consolidation.rs``.

The hippocampal-to-neocortical gating organ on the body side: scans
the short-term trace ring (``store::ring_buffer``), scores each entry
against the live neurochemical encoding, consolidation, and plasticity
weights, and promotes surviving traces into the memory-mapped long-term
store (``store::ltm``).
"""
