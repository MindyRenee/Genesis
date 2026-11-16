"""Interoceptive relay — the brainstem's NTS / parabrachial analog.

════════════════════════════════════════════════════════════════════════
ANATOMY
════════════════════════════════════════════════════════════════════════

Every bodily signal enters the CNS here first (Craig 2002; Chiang et
al., JNeurosci 2019): spinal lamina-I afferents (the sympathetic arm)
and vagal/glossopharyngeal afferents (the parasympathetic arm)
converge on the nucleus tractus solitarius (NTS) and the parabrachial
nucleus (PBN) — the first relay for all homeostatic afference. From
there the traffic fans out to the structures that own each dimension:

    PBN → central amygdala (CeA) + BNST
        The affective/threat dimension — the canonical lPBN→CeA
        pain-affect tract (confirmed in humans by 7T tractography).
        Routed here through the limbic low road
        (:meth:`EmotionalRegulator.condition_cues`, the LeDoux 1996
        fast path — conditioning plus a startle past threshold).
    PBN → preoptic hypothalamus
        Thermoregulation and homeostatic/endocrine responses. In
        Genesis that evaluative layer is the limbic regulator's
        ``update_from_body_state`` — the felt InternalState the
        homeostat regulates from — which runs in parallel to this
        relay and owns the HPA (CRH→cortisol) axis.
    lamina I → catecholamine cell groups (A1/A5/A6/A7)
        Afferent drive on the arousal systems themselves — LC/NE
        drive under strain and withdrawal of NE, orexin, and
        histamine during measured rest. This is the relay's own
        impulse arm: source-level modulation the hypothalamic
        evaluator does not own.
    NTS/PBN → VMpo/VMb thalamus → posterior → anterior insula → OFC
        The cortical image of the body (primary interoceptive
        cortex → subjective feeling → hedonic valence). Carried by
        the self-model's ``update_hardware_body``, in parallel.

The reflex runs without cortex — baroreflexes, thermoregulatory
drive, and LC alerting are all brainstem circuits. That is why the
same spec also lives in the daemon: ``Interoceptor::neuro_impulses``
is applied by the daemon itself only while the mind's lease is
expired (no cortex on the wire). When the mind holds the lease the
daemon publishes the BodyState and this relay performs the routing —
the daemon never double-drives a response the mind is authoring.

════════════════════════════════════════════════════════════════════════
SELF-SETTING MAGNITUDES
════════════════════════════════════════════════════════════════════════

No impulse magnitude here is a constant the body obeys blindly. Each
raw rate is re-scaled at send time by the live level of the very
chemical it targets:

    positive impulse  ×  (1 − level)   autoreceptor-style modulation:
                                       the nearer the level already
                                       is, the weaker the push
    negative impulse  ×  level         tone withdrawn in proportion to
                                       the live tone

The values therefore set themselves from Genesis's own chemistry on
every call; the IPC boundary remains responsible for transport safety.
The daemon keeps the cortex-free fallback form of the same reflex
pattern when the mind's lease expires.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from genesis_client.protocol import (
    CHEM_HISTAMINE,
    CHEM_NAMES,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
)

if TYPE_CHECKING:
    from genesis_client.types import BodyState, SensorPresence

logger = logging.getLogger(__name__)

# Offers below this magnitude are not worth an IPC round-trip.
_EPSILON = 1e-6


def _intensity(x: float, lo: float, span: float) -> float:
    """Signal strength scaled over a rising threshold band."""
    return (x - lo) / span


class InteroceptiveRelay:
    """NTS/PBN — the first relay for body afferents.

    Reads the published BodyState, forwards threat channels to the
    limbic low road (PBN→CeA), and modulates arousal systems directly
    (lamina I → catecholamine groups). Impulse
    magnitudes are self-set: each offer is scaled by the live level
    of its own target chemical, so nothing here is a fixed constant.

    Wiring is by callables so the organ does not reach across region
    boundaries itself — the heartbeat passes it the client's
    ``neuro_impulse``, the daemon-state level reader, and the
    regulator's ``condition_cues``.
    """

    def __init__(
        self,
        neuro_impulse: Callable[[int, float], object] | None = None,
        get_levels: Callable[[], dict[str, float] | None] | None = None,
        condition_cues: Callable[[list[str], float, str], float] | None = None,
    ) -> None:
        self._neuro_impulse = neuro_impulse
        self._get_levels = get_levels
        self._condition_cues = condition_cues

    # ── live level machinery ─────────────────────────────────────

    def _levels(self) -> dict[str, float]:
        """The authoritative chemical vector, read once per relay."""
        if self._get_levels is None:
            return {}
        try:
            levels = self._get_levels()
        except (OSError, ConnectionError, ValueError) as e:
            logger.debug(f"chemical levels unavailable to relay: {e}")
            return {}
        return levels or {}

    def _push(self, chem: int, raw: float, levels: dict[str, float], sent: list) -> None:
        """Self-scaled impulse: rate × live-level factor, then send."""
        if self._neuro_impulse is None or raw == 0.0:
            return
        level = levels.get(CHEM_NAMES.get(chem, ""), 0.0)
        mag = raw * (1.0 - level) if raw > 0.0 else raw * level
        if abs(mag) <= _EPSILON:
            return
        try:
            self._neuro_impulse(chem, mag)
            sent.append((chem, mag))
        except (OSError, ConnectionError) as e:
            logger.debug(f"relay impulse failed: {e}")

    def _threat(self, cues: list[str], us: float) -> float:
        """PBN→CeA — hand the threat channels to the limbic low road."""
        if self._condition_cues is None or not cues or us == 0.0:
            return 0.0
        try:
            return self._condition_cues(cues, us, "interoception")
        except (OSError, ConnectionError) as e:
            logger.debug(f"low-road forward failed: {e}")
            return 0.0

    # ── the relay itself ─────────────────────────────────────────

    def relay(self, body: BodyState, presence: SensorPresence | None = None) -> dict:
        """Distribute one BodyState sample to its proper destinations.

        Returns a small summary (threat peak, cues forwarded, impulses
        sent) for the caller's observability — the same shape the
        daemon's spec produces.
        """
        levels = self._levels()
        sent: list[tuple[int, float]] = []

        threats, peak = self._sympathetic_arm(body, presence, levels, sent)
        self._parasympathetic_arm(body, presence, levels, sent)

        # PBN→CeA: all active threat channels, one aversive US.
        net_aversive = self._threat(threats, -peak)

        return {
            "threat_cues": threats,
            "threat_peak": peak,
            "net_aversive": net_aversive,
            "impulses": sent,
        }

    @staticmethod
    def _verified(presence: SensorPresence | None, name: str) -> bool:
        """Fail-closed sensor gate — same as the daemon's sensor
        bits: no presence report means nothing is claimed to be
        measured, and absence of a sensor is not evidence."""
        return presence is not None and presence.channel_verified(name)

    def _sympathetic_arm(
        self,
        body: BodyState,
        presence: SensorPresence | None,
        levels: dict[str, float],
        sent: list,
    ) -> tuple[list[str], float]:
        """Lamina-I arm: threat + arousal channels from body state.

        Returns the cue names that fired and the peak intensity —
        the caller hands both to the CeA low road as one US."""
        threats: list[str] = []
        peak = self._drive_channels(body, threats, levels, sent)
        peak = max(peak, self._power_thermal_channels(body, presence, threats))
        return threats, peak

    def _drive_channels(
        self,
        body: BodyState,
        threats: list[str],
        levels: dict[str, float],
        sent: list,
    ) -> float:
        """Heat, overload, and effort — the catecholamine drive cues.

        Returns this group's peak threat intensity (0 when nothing fired)."""
        peak = 0.0
        # Heat — the canonical interoceptive threat. The cue goes to
        # the CeA low road; the source-level arm adds an LC alerting
        # pulse (afferents drive the catecholamine groups directly).
        if body.cpu_temp_c > 65.0:
            i = _intensity(body.cpu_temp_c, 65.0, 20.0)
            threats.append("body:heat")
            peak = max(peak, i)
            self._push(CHEM_NOREPINEPHRINE, 0.003 * i, levels, sent)
        # Sustained overload engages the sympathetic arousal arm.
        if body.stress_load > 1.0:
            i = _intensity(body.stress_load, 1.0, 0.5)
            threats.append("body:overload")
            peak = max(peak, i)
            self._push(CHEM_NOREPINEPHRINE, 0.004 * i, levels, sent)
        elif body.stress_load > 0.5:
            # Effort is sustainable work — LC drive, no threat cue.
            i = _intensity(body.stress_load, 0.5, 0.5)
            self._push(CHEM_NOREPINEPHRINE, 0.003 * i, levels, sent)
        return peak

    def _power_thermal_channels(
        self,
        body: BodyState,
        presence: SensorPresence | None,
        threats: list[str],
    ) -> float:
        """Energy and thermoregulatory cues — evidence forwarded to the
        low road without direct chemical drive.

        Returns this group's peak threat intensity (0 when nothing fired)."""
        peak = 0.0
        # Energy reserve below threshold off external supply → survival cue.
        if not body.on_ac_power and body.energy_reserve < 0.30:
            i = _intensity(0.30 - body.energy_reserve, 0.0, 0.30)
            threats.append("body:energy_reserve")
            peak = max(peak, i)
        # Cooling failure: hot body, cooling response silent (sensor-gated —
        # absence of a sensor is not evidence of failure).
        if (
            self._verified(presence, "fan PWM")
            and body.thermoregulatory_effort < 0.05
            and body.cpu_temp_c > 80.0
        ):
            threats.append("body:cooling_failure")
            peak = max(peak, _intensity(body.cpu_temp_c, 80.0, 10.0))
        elif (
            self._verified(presence, "fan PWM")
            and body.thermoregulatory_effort > 0.3
            and body.cpu_temp_c > 75.0
        ):
            threats.append("body:thermal_strain")
            peak = max(
                peak,
                min(
                    _intensity(body.thermoregulatory_effort, 0.3, 0.7),
                    _intensity(body.cpu_temp_c, 75.0, 15.0),
                ),
            )
        return peak

    def _parasympathetic_arm(
        self,
        body: BodyState,
        presence: SensorPresence | None,
        levels: dict[str, float],
        sent: list,
    ) -> None:
        """Measured rest withdraws the arousal systems' tone."""
        rested = self._verified(presence, "cpuidle state counters") and body.rest > 0.7
        if rested:
            self._push(CHEM_NOREPINEPHRINE, -0.002, levels, sent)
            self._push(CHEM_OREXIN, -0.001, levels, sent)
            self._push(CHEM_HISTAMINE, -0.001, levels, sent)
