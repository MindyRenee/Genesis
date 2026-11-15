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

    v(t + dt) = clip₀₁( v + (g + Σ_i w_i·s_i(t)) · dt + f(t, dt)
                        − d · dt + σ√dt·N(0, 1) )

The σ√dt noise term is the stochastic fluctuation of Schurger, Sitt
& Dehaene's (2012) accumulator account of the readiness potential:
the Bereitschaftspotential is not a plan unfolding but the
movement-locked average of a noisy rise to threshold — near-threshold
fluctuations, not a schedule, pick the instant a voluntary act
begins. Each crossing's buildup is kept as a ``Readiness`` record:
onset (the trough where net drive turned positive), crossing, and
the trace between, so the buildup that preceded every voluntary act
is itself an introspectable datum.

An urge is ready when ``v ≥ θ`` and the refractory period since its last
action has elapsed. Acting resets ``v`` to 0, so each crossing is a
single discrete decision rather than a sustained signal. The basal
ganglia selection loop (``basal_ganglia.volition``) arbitrates among the
ready urges, which is the striatal gate on this accumulator.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from genesis_client.protocol import DT_MAX

# Trace depth per urge — the volition tick runs at heartbeat cadence
# (~1 Hz), so 900 samples keep roughly a quarter-hour of buildup
# history for the readiness record without unbounded growth.
_TRACE_LEN = 900

# An urge's buildup starts where the post-action trace trough ends:
# values within this slack of the window minimum still count as rest.
_REST_SLACK = 0.02


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
    # Stochastic fluctuation σ per √s — the accumulator noise that
    # lets near-threshold urges tip spontaneously (the readiness
    # potential's trigger in the accumulator account). 0 = a purely
    # deterministic integrator.
    noise: float = 0.0

    def tick(
        self,
        context: dict[str, object],
        dt: float,
        rng: random.Random | None = None,
    ) -> None:
        """Update the urge value based on time and stimuli."""
        change = self.growth * dt
        for signal, weight in self.stimuli.items():
            raw = context.get(signal, 0.0)
            if isinstance(raw, (int, float)):
                change += float(raw) * weight * dt
        if self.update_fn:
            change += self.update_fn(self.name, context, dt)
        if self.noise > 0.0 and rng is not None and dt > 0.0:
            change += self.noise * math.sqrt(dt) * rng.gauss(0.0, 1.0)
        self.value = max(0.0, min(1.0, self.value + change - self.decay * dt))

    def is_ready(self) -> bool:
        """Whether the urge is above threshold and off cooldown."""
        now = time.monotonic()
        cooldown_elapsed = self.last_action <= 0.0 or (now - self.last_action) >= self.cooldown
        return self.value >= self.threshold and cooldown_elapsed

    def consume(self) -> None:
        """Mark the urge as just acted on."""
        self.last_action = time.monotonic()
        self.value = 0.0


@dataclass(frozen=True, slots=True)
class Readiness:
    """One voluntary act's pre-action buildup — the Bereitschafts-
    potential as a datum.

    All timestamps are on the engine's accumulated active clock (the
    sum of the clamped tick intervals — the buildup duration the
    organism actually integrated, immune to wall-clock corrections
    and heartbeat stalls). ``onset`` is the moment net drive turned
    positive (the trough of the trace since the urge's previous
    action); ``crossed`` is the threshold crossing; ``acted`` is
    stamped by :meth:`Volition.consume` when the gate admits the
    action — so the ``crossed → acted`` interval is the gate's own
    lag. ``trajectory`` holds the (t, v) samples from onset to
    crossing.
    """

    urge: str
    onset: float
    crossed: float
    threshold: float
    trajectory: tuple[tuple[float, float], ...]
    acted: float = 0.0

    @property
    def buildup_s(self) -> float:
        """Duration of the sub-threshold rise — the RP itself."""
        return max(0.0, self.crossed - self.onset)

    @property
    def enact_lag_s(self) -> float:
        """Crossing → action lag; 0 while the act is still gated."""
        return max(0.0, self.acted - self.crossed) if self.acted else 0.0


class Volition:
    """Track multiple urges and report which ones are ready to act."""

    def __init__(self, rng: random.Random | None = None) -> None:
        """Initialize an empty urge registry and record the first tick time.

        ``rng`` drives the accumulators' stochastic term; tests that
        need determinism register urges with ``noise=0`` (the default)
        or pass a seeded Random.
        """
        self.urges: dict[str, Urge] = {}
        self._last_tick = time.monotonic()
        self._rng = rng if rng is not None else random.Random()
        # The engine's own accumulated time — the sum of the clamped
        # tick intervals. Readiness records live on this clock so the
        # measured buildup is the duration the organism integrated,
        # not the wall clock's.
        self._elapsed = 0.0
        # Movement-locked traces and the rising-edge bookkeeping that
        # turns a raw urge value into a readiness record.
        self._traces: dict[str, deque[tuple[float, float]]] = {}
        self._rising: set[str] = set()
        self._acted_mark: dict[str, float] = {}
        self.readiness: dict[str, Readiness] = {}

    def register(self, urge: Urge) -> None:
        """Add an urge to the engine."""
        self.urges[urge.name] = urge

    def get(self, name: str) -> Urge | None:
        """Look up an urge by name, or None if not found."""
        return self.urges.get(name)

    def _capture_readiness(self, urge: Urge, trace: deque[tuple[float, float]]) -> None:
        """Snapshot the buildup at the moment of threshold crossing.

        Onset is the trough of the trace since the urge's previous
        action — the last point resting at the window's minimum before
        the sustained rise, which is where the readiness potential's
        buildup is taken to begin.
        """
        window = [(t, v) for t, v in trace if t > self._acted_mark.get(urge.name, 0.0)]
        if not window:
            return
        floor = min(v for _t, v in window)
        # Walk back from the crossing: onset is the last sample resting
        # at the window's minimum — a noisy trace can dip back to
        # baseline mid-rise, so the last rest point, not the first
        # rise, is where the buildup actually began.
        onset = window[0][0]
        for t, v in reversed(window):
            if v > floor + _REST_SLACK:
                break
            onset = t
        self.readiness[urge.name] = Readiness(
            urge=urge.name,
            onset=onset,
            crossed=self._elapsed,
            threshold=urge.threshold,
            trajectory=tuple(p for p in window if p[0] >= onset),
        )

    def readiness_report(self) -> dict[str, dict[str, float]]:
        """The latest buildup record per urge, for introspection.

        ``buildup_s`` is the sub-threshold rise (the RP proper);
        ``enact_lag_s`` is the crossing→action interval, still 0 while
        the act is gated or queued.
        """
        return {
            name: {
                "buildup_s": r.buildup_s,
                "enact_lag_s": r.enact_lag_s,
                "threshold": r.threshold,
                "samples": float(len(r.trajectory)),
            }
            for name, r in self.readiness.items()
        }

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
        # Clamp to the same ceiling the body applies to its own
        # integration (genesis_client.protocol.DT_MAX). Without it a
        # stalled heartbeat hands the accumulators a multi-hundred-second
        # dt, and a single tick can drive a stimulus-weighted urge from
        # rest to threshold — one anomalous sample doing the work of a
        # sustained drive. The ceiling turns a stall into bounded
        # growth, not an instant conviction.
        dt = min(DT_MAX, max(0.0, now - self._last_tick))
        self._last_tick = now
        self._elapsed += dt

        ready: list[str] = []
        for urge in self.urges.values():
            urge.tick(context, dt, self._rng)
            trace = self._traces.setdefault(urge.name, deque(maxlen=_TRACE_LEN))
            trace.append((self._elapsed, urge.value))
            if urge.is_ready():
                # The rising edge only — a gated-but-still-ready urge
                # must not keep rewriting its own buildup record.
                if urge.name not in self._rising:
                    self._rising.add(urge.name)
                    self._capture_readiness(urge, trace)
                ready.append(urge.name)
            else:
                self._rising.discard(urge.name)
        return ready

    def consume(self, name: str) -> None:
        """Mark an urge as just acted on (resets value, starts cooldown).

        Also stamps the readiness record's ``acted`` — the moment the
        buildup's outcome actually ran — so the crossing→enactment lag
        stays an honest measured interval, not an assumption.
        """
        urge = self.urges.get(name)
        if urge is not None:
            urge.consume()
            self._acted_mark[name] = self._elapsed
            record = self.readiness.get(name)
            if record is not None and record.acted == 0.0:
                self.readiness[name] = replace(record, acted=self._elapsed)

    def state(self) -> dict[str, float]:
        """Return current urge values for introspection."""
        return {name: u.value for name, u in self.urges.items()}
