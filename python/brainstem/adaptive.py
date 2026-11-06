"""Adaptive interval gates — word-layer companion to ``src/daemon/adaptive.rs``.

Every maintenance gear spins at the rate its own signal demands,
bounded so nothing stalls or redlines. A ``tick_count % N`` gate is a
gear keyed to the axle: change the tick rate (or pace it from the
mind's ~1 Hz ``advance_neuro`` instead of the daemon's 5 Hz ``tick``)
and every gate silently changes period. The organ's own solution is
signal-paced gates — each periodic job waits on the counter that
actually matters to it (LTM sync on plasticity, probe on samples,
persist on events), never on raw ticks.
"""
