"""Tick — word-layer companion to ``src/daemon/tick.rs``.

The body's heartbeat: the ~5 Hz daemon loop that steps every organ —
interoception reads, the probe samples, adaptive gates fire, state
syncs. The mind paces off a slower ~1 Hz ``advance_neuro``; between
them is the rhythm the whole body keeps. When the tick says so,
the body's chemistry moves; when the mind says so, the mind's does.
"""
