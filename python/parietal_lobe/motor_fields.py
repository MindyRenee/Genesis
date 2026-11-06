"""Motor fields — word-layer companion to ``src/field/motor.rs``.

The mind's decisions laid out over a coordinate grid instead of
collapsed to a symbol: when cognition decides "act there" or "press
action 3", it writes the decision as *fields* on the tensor slab —
``motor.spatial`` (peaks where it wants to act: goals attract,
hazards repel, unexplored pulls weakly), ``motor.actions`` and
``motor.action_ids`` (a disposition per offered action). The decode
side projects the winning field cell onto the canonical 64×64 grid —
the layer between deciding and acting, spatial like real motor cortex.
"""
