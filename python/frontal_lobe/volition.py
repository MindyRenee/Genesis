"""Volition — internal urges that decide when Genesis acts on herself.

Unlike scheduled background loops, urges grow organically based on
internal and environmental triggers. When an urge crosses its
threshold, Genesis "feels like" doing the corresponding action and
does it once.

Mathematical form: each urge is a leaky accumulator (Usher &
McClelland, 2001, leaky competing accumulator), the same
integrate-to-threshold dynamics that model striatal and cortical
evidence accumulation. With value ``v`` in [0, 1], intrinsic growth
``g``, stimulus weights ``w_i`` on signals ``s_i(t)``, leak ``d``, and
an optional drive ``f(t, dt)`` that returns its own increment:

    v(t + dt) = clip₀₁( v + (g + Σ_i w_i·s_i(t)) · dt + f(t, dt) − d · dt )

An urge is ready when ``v ≥ θ`` and the refractory period since its last
action has elapsed. Acting resets ``v`` to 0, so each crossing is a
single discrete decision rather than a sustained signal. The basal
ganglia selection loop (``basal_ganglia.volition``) arbitrates among the
ready urges, which is the striatal gate on this accumulator.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Urge:
    """A named internal drive with a value that grows and decays."""

    name: str
    threshold: float = 0.7
    growth: float = 0.001
    decay: float = 0.0005
    cooldown: float = 120.0
    value: float = 0.0
    last_action: float = 0.0  # monotonic-clock timestamp
    stimuli: dict[str, float] = field(default_factory=dict)
    update_fn: Callable[[str, dict[str, object], float], float] | None = None

    def tick(self, context: dict[str, object], dt: float) -> None:
        """Update the urge value based on time and stimuli."""
        change = self.growth * dt
        for signal, weight in self.stimuli.items():
            raw = context.get(signal, 0.0)
            if isinstance(raw, (int, float)):
                change += float(raw) * weight * dt
        if self.update_fn:
            change += self.update_fn(self.name, context, dt)
        self.value = max(0.0, min(1.0, self.value + change - self.decay * dt))

    def is_ready(self) -> bool:
        """Whether the urge is above threshold and off cooldown."""
        now = time.monotonic()
        return self.value >= self.threshold and (now - self.last_action) >= self.cooldown

    def consume(self) -> None:
        """Mark the urge as just acted on."""
        self.last_action = time.monotonic()
        self.value = 0.0


class Volition:
    """Track multiple urges and report which ones are ready to act."""

    def __init__(self) -> None:
        """Initialize an empty urge registry and record the first tick time."""
        self.urges: dict[str, Urge] = {}
        self._last_tick = time.monotonic()

    def register(self, urge: Urge) -> None:
        """Add an urge to the engine."""
        self.urges[urge.name] = urge

    def get(self, name: str) -> Urge | None:
        """Look up an urge by name, or None if not found."""
        return self.urges.get(name)

    def tick(self, context: dict[str, object]) -> list[str]:
        """Advance all urges and return the names of those that are ready.

        The engine uses a monotonic clock for elapsed-time accounting so
        wall-clock corrections (NTP, manual changes, DST) cannot create a
        negative ``dt`` or prematurely satisfy a cooldown.

        Ready urges are **not** consumed — the caller must call
        :meth:`consume` for each urge it actually acts on. This lets
        the caller apply external filters (e.g. wake-delay gating)
        without wasting an urge's cooldown on a fire that was filtered
        out before it could run.
        """
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        ready: list[str] = []
        for urge in self.urges.values():
            urge.tick(context, dt)
            if urge.is_ready():
                ready.append(urge.name)
        return ready

    def consume(self, name: str) -> None:
        """Mark an urge as just acted on (resets value, starts cooldown)."""
        urge = self.urges.get(name)
        if urge is not None:
            urge.consume()

    def state(self) -> dict[str, float]:
        """Return current urge values for introspection."""
        return {name: u.value for name, u in self.urges.items()}
