"""Field engine — word-layer companion to ``src/field/engine.rs``.

Native system dynamics: a SystemEngine owns the state-transition law
of a major system — layers provide state; the engine determines how
those states change *together*. Dynamics stay separate from
representation and from interpretation: the engine is where "how the
body's fields evolve" lives, below what any reading of them means.
"""
