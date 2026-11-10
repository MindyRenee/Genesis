"""Heart — the medullary cardiac center.

The heart itself lives in the body: her field substrate's coherent
motion *is* the rhythm — the consensus of the body/prediction/
agency fibers winding across the latent manifold, with Maxwell's
equations saying what field that motion makes
(``brainstem.maxwell``, ``src/daemon/maxwell.rs``). What lives here
is the brainstem's half of the loop — the cardiac control center,
the way the medulla regulates a heart it does not itself beat:

- **Afferent (interoception):** the field signature off the wire —
  the carrier frequency is the heartbeat itself (the consensus'
  winding rate; there is no separate cardiac rhythm to observe),
  and the Fréchet disagreement is the coherence. A field whose
  fibers cannot hold consensus is an arrhythmia — coherence
  collapse while the carrier runs is *felt*, arriving as a small
  norepinephrine/epinephrine impulse on the transition, the
  palpitation channel. A sustained good lock is unremarkable — a
  healthy pulse is silent.
- **Efferent (autonomic outflow):** contractility demand expressed
  as norepinephrine tone — the only chemistry channel the body
  sanctions (bounded, decaying impulses). Engagement deepens
  sympathetic drive the way it raises stroke volume; rest withdraws
  it. Demand is re-asserted on a quantized cadence — a tonic kick
  the chemistry integrates, never a pinned level.
- **The keep:** this center also holds the hypnagogic boundary
  watch — the parasympathetic keep that rides the alpha↔theta edge
  of sleep onset. What remains is the central keep —
  chemistry nudges at the edges and the lucid band's sustained
  drive, catch and release, never a pinned state.

It schedules nothing. The metronome ticks it; the field's own
dynamics set the carrier — her heartbeat *is* the field's carrier
frequency — and it is *measured*, not assumed: every number it
reports comes off the signature, never a template.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from brainstem.maxwell import carrier_hz, coherence, field_signature

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

# ── Contractility drive ──────────────────────────────────────────
#: depth = _DEPTH_BASE + _DEPTH_GAIN * arousal — the autonomic
#: demand the heart computes. Resting arousal (~0.3) yields a
#: gentle ~0.33 demand; full engagement approaches 0.85.
_DEPTH_BASE = 0.10
_DEPTH_GAIN = 0.75
_DEPTH_MIN = 0.05
_DEPTH_MAX = 0.90

#: The depth at which autonomic drive is neutral — the demand a
#: resting arousal (~0.3) computes to. Offsets from it are the
#: signed contractility signal the NE channel carries.
_DEPTH_NEUTRAL = _DEPTH_BASE + _DEPTH_GAIN * 0.3

#: Demand→impulse gain: a full-scale swing (0.85→0.05) maps to a
#: ±~0.12 NE kick — phasic, inside the impulse cap, re-asserted on
#: the keep-alive cadence so sustained demand holds a level through
#: the chemistry's own decay rather than a pinned write.
_CONTRACTILITY_GAIN = 0.15

#: Re-assertion of the contractility demand is quantized — autonomic
#: drive is continuous but the impulse only fires when demand has
#: actually moved (|Δ| > step) or the channel wants a keep-alive.
_DEPTH_STEP = 0.10
_DEPTH_KEEPALIVE_S = 30.0

#: Coherence below which a running rhythm counts as arrhythmic.
#: 0 = the fibers do not even try to hold consensus; 1 = perfect
#: lock. The same tension scale the body's precision attenuation
#: uses (see brainstem.maxwell).
_LOCK_FLOOR = 0.30

#: The palpitation impulse on arrhythmia onset — felt, not shouted.
_PALPITATION = 0.05

# ── The hypnagogic keep ───────────────────────────────────────────
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

#: Recent tempos kept for the jitter read — eight beats at the
#: medulla's 2 s cadence is ~16 s of rhythm, enough for a
#: coefficient of variation without turning the center into a
#: history book.
_JITTER_WINDOW = 8


@dataclass
class CardiacState:
    """One measured beat — what the field's signature says the
    pulse is doing."""

    rate_hz: float = 0.0  # the carrier — the field's winding rate, i.e. her heartbeat
    lock: float | None = None  # coherence 0..1 — the fibers holding consensus
    lag_rad: float | None = None  # conduction delay — no analog on the signature; unreported
    jitter: float | None = None  # CV of recent tempos — the HRV analog
    work_a: float | None = None  # coherent current — the beat's cost in the model's units
    depth: float | None = None  # contractility demand in force
    arrhythmic: bool = False  # rhythm running, coherence below floor
    pulse_armed: bool = False  # the hypnagogic keep is armed


class Heart:
    """The cardiac control center — reads the field's pulse,
    modulates force through autonomic chemistry.

    Args:
        client: IPC to the body (the field signature read).
        neuro_impulse: The impulse channel contractility and
            arrhythmia are expressed through — ``(chem_idx, amount)``
            like the world's coupling. Without it the heart still
            reads and reports; it just cannot push.
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
        self._depth_sent: float | None = None
        self._depth_sent_at = 0.0
        self._keeper_armed = False
        self._next_nudge_at = 0.0
        self._carriers: deque[float] = deque(maxlen=_JITTER_WINDOW)
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
        sig = field_signature(self._client)
        if sig is None:
            self.state = CardiacState()
            self._carriers.clear()
            return
        st = self.state
        st.rate_hz = carrier_hz(sig)
        st.lock = coherence(sig)
        st.work_a = sig.coherent_current_a if sig.signature_present else None
        if st.rate_hz > 0.0:
            self._carriers.append(st.rate_hz)
        st.jitter = self._tempo_cv()
        self._drive(now)
        self._feel(st)
        self._keep(now)

    def _tempo_cv(self) -> float | None:
        """Coefficient of variation of the recent carrier
        frequencies — the heart-rate-variability analog read off
        the field's own winding-rate wobble. ``None`` until enough
        beats exist to say anything honest."""
        carriers = list(self._carriers)
        if len(carriers) < 3:
            return None
        mean = sum(carriers) / len(carriers)
        if mean <= 0.0:
            return None
        var = sum((c - mean) ** 2 for c in carriers) / len(carriers)
        return (var**0.5) / mean

    def _drive(self, now: float) -> None:
        """Efferent: engagement → contractility, expressed as
        norepinephrine tone — the autonomic outflow the medulla
        actually commands. The carrier itself is never touched:
        tempo stays the field's own, always — the heart modulates
        the chemistry the field reads, not the rhythm it computes."""
        if self._keeper_armed or self._neuro_impulse is None:
            # The keep owns contractility while armed — the pulse
            # is a wave now, not a set-point. And with no impulse
            # channel there is no outflow to write.
            return
        a = min(1.0, max(0.0, self._arousal()))
        depth = min(_DEPTH_MAX, max(_DEPTH_MIN, _DEPTH_BASE + _DEPTH_GAIN * a))
        moved = self._depth_sent is None or abs(depth - self._depth_sent) > _DEPTH_STEP
        stale = now - self._depth_sent_at > _DEPTH_KEEPALIVE_S
        if not (moved or stale):
            return
        # The demand as a signed offset from neutral autonomic
        # tone — above it sympathetic (NE+), below it withdrawal
        # (NE−). Bounded phasic kick; the chemistry integrates it.
        kick = (depth - _DEPTH_NEUTRAL) * _CONTRACTILITY_GAIN
        if abs(kick) < 1.0e-4:
            self._depth_sent, self._depth_sent_at = depth, now
            self.state.depth = depth
            return
        self._neuro_impulse(CHEM_NOREPINEPHRINE, kick)
        self._depth_sent, self._depth_sent_at = depth, now
        self.state.depth = depth

    def _feel(self, st: CardiacState) -> None:
        """Afferent: arrhythmia onset is a palpitation, once."""
        running = st.rate_hz > 0.0
        arrhythmic = running and st.lock is not None and st.lock < _LOCK_FLOOR
        if arrhythmic and not st.arrhythmic and self._neuro_impulse is not None:
            # The field stopped holding consensus — a skipped-beat
            # feeling, delivered through the body's own channel.
            self._neuro_impulse(CHEM_NOREPINEPHRINE, _PALPITATION)
            self._neuro_impulse(CHEM_EPINEPHRINE, _PALPITATION)
        st.arrhythmic = arrhythmic

    def arm_hypnagogic(self) -> bool:
        """Enter the hypnagogic keep — the boundary watch engages.

        The keep that remains is hers to
        hold — it watches the alpha↔theta edge, nudges chemistry at
        its margins, and sustains the lucid band's drive. Rate is
        never touched: tempo stays the field's, always. There is
        nothing left to fail — the keep is central, so arming is
        the watch itself.
        """
        self._keeper_armed = True
        self.state.pulse_armed = True
        return True

    def disarm_hypnagogic(self) -> bool:
        """Stand the keep down — the watch releases, static
        contractility resumes on the next beat."""
        self._keeper_armed = False
        self.state.pulse_armed = False
        self._depth_sent = None  # _drive re-asserts the demand cleanly
        return True

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
