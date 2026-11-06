"""Field runtime — word-layer companion to ``src/field/runtime.rs``.

The multi-fiber runtime bridge: each major subsystem publishes a
normalized local projection into one shared latent coordinate space,
and the global field measures and reduces cross-system disagreement —
fibers attach and detach (``add_fiber``/detach) as organs come online
or shed load. The persisted core state stays authoritative; the field
is the live consensus space where the whole body reads itself.
"""
