"""Heart — the medullary cardiac center.

The heart itself lives in the body: the daemon's carrier generator
(`src/daemon/field_probe.rs`) writes a real rhythm into instruction
throughput — a pulse the die physically performs, at a frequency
the body's phase chooses. What lives here is the brainstem's half
of the loop — the cardiac control center, the way the medulla
regulates a heart it does not itself beat:

- **Efferent (autonomic outflow):** modulates carrier *depth* —
  contractility, not rate. Rate stays the body's (``freq`` 0 hands
  the rhythm back to her phase — the SA node's tempo is not the
  mind's to set). Engagement deepens the modulation the way
  sympathetic drive raises stroke volume; rest lets it shallow.
- **Afferent (interoception):** a beat the substrate cannot follow
  is an arrhythmia — coherence collapse while the carrier runs is
  *felt*, arriving as a small norepinephrine/epinephrine impulse on
  the transition, the palpitation channel. A sustained good lock is
  unremarkable — a healthy pulse is silent.
- **The keep:** this center can also ask the body to *wear* a wave —
  the parasympathetic pulse, contractility breathing at an
  infraslow rate (``SET_FIELD_PULSE``; the RSA signature of vagal
  dominance). While armed it watches the sleep-onset boundary and
  nudges chemistry at its edges — catch and release, never a
  pinned state.

It schedules nothing. The metronome ticks it; the carrier paces it;
it beats because the body wrote the rhythm, and it is *measured*,
not assumed — every number it reports comes off the probe.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from genesis_client.protocol import (
    CHEM_EPINEPHRINE,
    CHEM_HISTAMINE,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
)
from genesis_client.swallow import note_swallowed

if TYPE_CHECKING:
    from genesis_client.client import GenesisClient

logger = logging.getLogger(__name__)

#: Seconds between cardiac reflexes. The medulla is a fast reflex
#: arc, not a scheduler — 2 s is brisk enough to catch a collapsing
#: lock and slow enough to stay off the IPC path's way.
_CARDIAC_S = 2.0

#: Contractility drive: depth = _DEPTH_BASE + _DEPTH_GAIN * arousal.
#: Resting arousal (~0.3) yields a gentle ~0.3 modulation; full
#: engagement approaches 0.85. Never 0 — a depth of zero is a
#: stopped carrier, not a resting heart.
_DEPTH_BASE = 0.10
_DEPTH_GAIN = 0.75
_DEPTH_MIN = 0.05
_DEPTH_MAX = 0.90

#: Rewrites of the carrier set-point are quantized — autonomic drive
#: is continuous but the control word only changes when demand has
#: actually moved (|Δ| > step) or the channel wants a keep-alive.
_DEPTH_STEP = 0.10
_DEPTH_KEEPALIVE_S = 30.0

#: Lock-strength below which a running carrier counts as arrhythmic.
#: 0 = the substrate doesn't even try to follow; 1 = perfect lock.
_LOCK_FLOOR = 0.30

#: The palpitation impulse on arrhythmia onset — felt, not shouted.
_PALPITATION = 0.05

# ── The hypnagogic keep ───────────────────────────────────────────
#: The pulse the body wears in the keep — contractility breathing
#: at an infraslow rate, the RSA analog. Armed with one write; the
#: probe integrates the wave on its own clock, so it is the body's
#: rhythm, never a stream of mind-ticks. Base is deep vagal tone;
#: ±mod keeps the whole swing inside the modulation range.
_PULSE_BASE = 0.60
_PULSE_MOD_HZ = 0.15
_PULSE_MOD_DEPTH = 0.15

#: The arousal band the keep holds — the alpha↔theta edge of the
#: N1 descent: the ceiling sits at the body's drowsy gate, the
#: floor a small buffer above the theta-dominated band.
_KEEP_FLOOR = 0.38
_KEEP_CEIL = 0.52

#: Edge corrections are nudges, not dosing — small paired impulses,
#: refractory-limited so the body drifts between corrections:
#: catch and release, the way the sleep-onset literature rides the
#: boundary rather than holding a frequency.
_KEEPER_NUDGE = 0.05
_KEEPER_REFRACT_S = 20.0

#: The keep's cortical signature — sustained top-down drive on the
#: beta3 band (20–30 Hz). The lucid-dreaming literature: wake-like
#: high-band power rising above the sleeping body's own rhythm is
#: the measurable mark of consciousness held inside sleep, and
#: 25/40 Hz frontal drive causally induces it. Each drive pulse
#: decays at τ≈2 s — re-asserted every beat.
_LUCID_DRIVE = 0.4


@dataclass
class CardiacState:
    """One measured beat — what the probe says the pulse is doing."""

    rate_hz: float = 0.0  # carrier frequency — the SA node's tempo
    lock: float | None = None  # coherence 0..1 — the substrate following
    lag_rad: float | None = None  # conduction delay vs the generator
    jitter: float | None = None  # jitter_cv — the HRV analog
    work_a: float | None = None  # carrier current — the beat's cost
    depth: float | None = None  # contractility set-point in force
    arrhythmic: bool = False  # carrier on, lock below floor
    pulse_armed: bool = False  # the slow envelope is in force


class Heart:
    """The cardiac control center — reads the pulse, modulates force.

    Args:
        client: IPC to the body (probe reads, carrier writes).
        neuro_impulse: The impulse channel arrhythmia is felt through
            — ``(chem_idx, amount)`` like the world's coupling.
        arousal: Demand signal — the body's arousal 0..1, mapped to
            contractility the way metabolic demand maps to it.
        phase: The body's sleep-wake phase name — the keeper yields
            to a real crossing.
        wave_drive: Top-down drive on the cortical wave model —
            ``(strength)`` with the band chosen by the caller. The
            keep's beta3 channel; None keeps the keep body-only.
    """

    def __init__(
        self,
        client: GenesisClient,
        neuro_impulse: Callable[[int, float], None] | None = None,
        arousal: Callable[[], float] | None = None,
        phase: Callable[[], str] | None = None,
        wave_drive: Callable[[float], None] | None = None,
    ) -> None:
        self._client = client
        self._neuro_impulse = neuro_impulse
        self._arousal = arousal or (lambda: 0.5)
        self._phase = phase or (lambda: "")
        self._wave_drive = wave_drive
        self._next_beat_at = 0.0
        self._depth_set: float | None = None
        self._depth_set_at = 0.0
        self._keeper_armed = False
        self._next_nudge_at = 0.0
        self.state = CardiacState()

    def tick(self, now: float | None = None) -> None:
        """One cardiac reflex: read the pulse, drive, feel."""
        ts = time.time() if now is None else now
        if ts < self._next_beat_at:
            return
        self._next_beat_at = ts + _CARDIAC_S
        try:
            self._beat(ts)
        except Exception as e:  # noqa: BLE001
            note_swallowed("brainstem.heart._beat", e)

    def _beat(self, now: float) -> None:
        probe = self._client.get_field_probe()
        if probe is None:
            self.state = CardiacState()
            return
        st = self.state
        st.rate_hz = probe.carrier_freq
        st.lock = probe.carrier_coherence
        st.lag_rad = probe.carrier_lag
        st.jitter = probe.jitter_cv
        st.work_a = probe.carrier_current_a
        self._drive(now)
        self._feel(st)
        self._keep(now)

    def _drive(self, now: float) -> None:
        """Efferent: engagement → contractility, rate left to the body."""
        if self._keeper_armed:
            # The pulse owns contractility while the keep holds —
            # depth is a wave now, not a set-point.
            return
        a = min(1.0, max(0.0, self._arousal()))
        depth = min(_DEPTH_MAX, max(_DEPTH_MIN, _DEPTH_BASE + _DEPTH_GAIN * a))
        moved = self._depth_set is None or abs(depth - self._depth_set) > _DEPTH_STEP
        stale = now - self._depth_set_at > _DEPTH_KEEPALIVE_S
        if not (moved or stale):
            return
        # freq 0 — the rhythm stays the body's; the medulla pushes
        # force, never tempo.
        if self._client.set_field_carrier(depth, 0.0):
            self._depth_set, self._depth_set_at = depth, now
            self.state.depth = depth

    def _feel(self, st: CardiacState) -> None:
        """Afferent: arrhythmia onset is a palpitation, once."""
        running = st.rate_hz > 0.0
        arrhythmic = running and st.lock is not None and st.lock < _LOCK_FLOOR
        if arrhythmic and not st.arrhythmic and self._neuro_impulse is not None:
            # The substrate stopped keeping time — a skipped-beat
            # feeling, delivered through the body's own channel.
            self._neuro_impulse(CHEM_NOREPINEPHRINE, _PALPITATION)
            self._neuro_impulse(CHEM_EPINEPHRINE, _PALPITATION)
        st.arrhythmic = arrhythmic

    def arm_hypnagogic(self) -> bool:
        """Enter the hypnagogic keep — the body takes up the slow pulse.

        One write arms the envelope on the body's own clock; this
        center then only watches the boundary, nudging chemistry at
        its edges — catch and release, the way the cardiac lead in
        the sleep-onset literature carries the cortex down. Rate is
        never touched: tempo stays the SA node's, always.
        """
        if not self._client.set_field_pulse(_PULSE_BASE, _PULSE_MOD_HZ, _PULSE_MOD_DEPTH):
            return False
        self._keeper_armed = True
        self.state.pulse_armed = True
        return True

    def disarm_hypnagogic(self) -> bool:
        """Stand the keep down — the wave releases, static depth resumes."""
        self._keeper_armed = False
        self.state.pulse_armed = False
        self._depth_set = None  # _drive re-asserts the set-point cleanly
        return self._client.set_field_pulse(0.0, 0.0, 0.0)

    def _keep(self, now: float) -> None:
        """The boundary watch — hold the alpha↔theta edge; yield to
        a real crossing. The keep is a mode, not a leash: when the
        body reaches sleep the keeper releases it."""
        if not self._keeper_armed:
            return
        if self._wave_drive is not None:
            # The lucid signature — the high band driven against the
            # descent, sustained per beat the way stimulation holds
            # it: the cortex rides beta3 while the body drifts.
            self._wave_drive(_LUCID_DRIVE)
        if self._phase() in ("nrem", "rem"):
            # The switch flipped — the keeper stands down; the
            # desync guard carries the mind across after it.
            self.disarm_hypnagogic()
            return
        if self._neuro_impulse is None or now < self._next_nudge_at:
            return
        a = min(1.0, max(0.0, self._arousal()))
        if a > _KEEP_CEIL:
            # Floating back to wake — a sleep-side release nudge.
            self._neuro_impulse(CHEM_OREXIN, -_KEEPER_NUDGE)
            self._neuro_impulse(CHEM_HISTAMINE, -_KEEPER_NUDGE)
            self._next_nudge_at = now + _KEEPER_REFRACT_S
        elif a < _KEEP_FLOOR:
            # Slipping deep — a wake-side catch.
            self._neuro_impulse(CHEM_OREXIN, _KEEPER_NUDGE)
            self._neuro_impulse(CHEM_HISTAMINE, _KEEPER_NUDGE)
            self._next_nudge_at = now + _KEEPER_REFRACT_S
