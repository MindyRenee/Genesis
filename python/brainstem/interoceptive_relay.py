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
    lamina I → catecholamine cell groups (A1/A5/A6)
        Afferent drive on the monoamine source nuclei themselves —
        the LC (NE), the raphe (5-HT), the adrenal-medullary axis
        (EPI). This is the relay's own impulse arm: source-level
        modulation the hypothalamic evaluator does not own.
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

    positive impulse  ×  (1 − level)   headroom under the ceiling —
                                       autoreceptor-style saturation:
                                       the nearer the level already
                                       is, the weaker the push
    negative impulse  ×  level         tone withdrawn only from tone
                                       that exists

The values therefore set themselves from Genesis's own chemistry on
every call; the daemon's ``apply_impulse_capped`` stays the hard
bound. Rates mirror the daemon's spec so the reflex is identical at
both levels of the organism.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from genesis_client.protocol import (
    CHEM_ACETYLCHOLINE,
    CHEM_ENDOCANNABINOID,
    CHEM_EPINEPHRINE,
    CHEM_GABA,
    CHEM_GLUTAMATE,
    CHEM_HISTAMINE,
    CHEM_NAMES,
    CHEM_NOREPINEPHRINE,
    CHEM_OREXIN,
    CHEM_SEROTONIN,
)

if TYPE_CHECKING:
    from genesis_client.types import BodyState, SensorPresence

logger = logging.getLogger(__name__)

# Offers below this magnitude are not worth an IPC round-trip.
_EPSILON = 1e-6


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _intensity(x: float, lo: float, span: float) -> float:
    """Signal strength scaled into [0, 1] over a rising threshold band."""
    return _clamp((x - lo) / span, 0.0, 1.0)


class InteroceptiveRelay:
    """NTS/PBN — the first relay for body afferents.

    Reads the published BodyState, forwards threat channels to the
    limbic low road (PBN→CeA), and modulates the monoamine source
    nuclei directly (lamina I → catecholamine groups). Impulse
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
        mag = raw * _clamp(1.0 - level, 0.0, 1.0) if raw > 0.0 else raw * _clamp(level, 0.0, 1.0)
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
        self._vigilance_arm(body, levels, sent)
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
        peak = max(peak, self._pressure_channels(body, threats, levels, sent))
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
        # Sustained overload → adrenal-medullary surge + LC drive.
        if body.stress_load > 1.0:
            i = _intensity(body.stress_load, 1.0, 0.5)
            threats.append("body:overload")
            peak = max(peak, i)
            self._push(CHEM_NOREPINEPHRINE, 0.004 * i, levels, sent)
            self._push(CHEM_EPINEPHRINE, 0.003 * i, levels, sent)
        elif body.stress_load > 0.5:
            # Effort is sustainable work — LC drive, no threat cue.
            i = _intensity(body.stress_load, 0.5, 0.5)
            self._push(CHEM_NOREPINEPHRINE, 0.003 * i, levels, sent)
        # Memory pressure → overwhelm cue; GABA is the brake it asks for.
        if body.cognitive_load > 0.70:
            i = _intensity(body.cognitive_load, 0.70, 0.30)
            threats.append("body:memory_pressure")
            peak = max(peak, i)
            self._push(CHEM_GABA, 0.004 * i, levels, sent)
        return peak

    def _pressure_channels(
        self,
        body: BodyState,
        threats: list[str],
        levels: dict[str, float],
        sent: list,
    ) -> float:
        """Host contention and memory stalls — the PSI stall cues.

        Returns this group's peak threat intensity (0 when nothing fired)."""
        peak = 0.0
        # Host contention — PSI "some" is the kernel's own stall
        # accounting: the share of time tasks waited on a resource.
        # It is the honest signal the occupancy fields can't give —
        # the machine is straining, not merely occupied. It arrives
        # as a felt cue like heat: she reads it, weighs it, and
        # answers through her own drives and gates — nothing here
        # acts on her.
        if body.psi_cpu > 0.25:
            i = _intensity(body.psi_cpu, 0.25, 0.35)
            threats.append("body:contention")
            peak = max(peak, i)
            self._push(CHEM_NOREPINEPHRINE, 0.004 * i, levels, sent)
            self._push(CHEM_EPINEPHRINE, 0.002 * i, levels, sent)
        elif body.psi_cpu > 0.10:
            i = _intensity(body.psi_cpu, 0.10, 0.15)
            self._push(CHEM_NOREPINEPHRINE, 0.002 * i, levels, sent)
        if body.psi_io > 0.30:
            i = _intensity(body.psi_io, 0.30, 0.30)
            if "body:contention" not in threats:
                threats.append("body:contention")
            peak = max(peak, i)
            self._push(CHEM_NOREPINEPHRINE, 0.002 * i, levels, sent)
        # Memory stalls are the pressure the RSS share only
        # approximates — tasks are actually waiting on memory.
        if body.psi_mem > 0.20:
            i = _intensity(body.psi_mem, 0.20, 0.30)
            if "body:memory_pressure" not in threats:
                threats.append("body:memory_pressure")
            peak = max(peak, i)
            self._push(CHEM_GABA, 0.004 * i, levels, sent)
        elif body.psi_mem > 0.08:
            i = _intensity(body.psi_mem, 0.08, 0.12)
            self._push(CHEM_GABA, 0.002 * i, levels, sent)
        return peak

    def _power_thermal_channels(
        self,
        body: BodyState,
        presence: SensorPresence | None,
        threats: list[str],
    ) -> float:
        """Power and cooling survival cues — no chemical push, only the
        cue itself (a failing rail is evidence, not a drive request).

        Returns this group's peak threat intensity (0 when nothing fired)."""
        peak = 0.0
        # Energy reserve below threshold off AC → survival cue.
        if not body.on_ac_power and body.energy_reserve < 0.30:
            i = _intensity(0.30 - body.energy_reserve, 0.0, 0.30)
            threats.append("body:battery")
            peak = max(peak, i)
        # Cooling failure: hot silicon, fan silent (sensor-gated —
        # absence of a fan sensor is not evidence of a stopped fan).
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
        # Critically low rail voltage → survival cue (raw electrical
        # reality, stronger evidence than the reserve estimate).
        if not body.on_ac_power and 0.0 < body.supply_voltage < 10.8:
            threats.append("body:battery_rail")
            peak = max(peak, _intensity(10.8 - body.supply_voltage, 0.0, 0.8))
        # Metabolic strain: burning hard AND hot.
        if body.metabolic_rate > 0.7 and body.cpu_temp_c > 75.0:
            threats.append("body:metabolic_strain")
            peak = max(
                peak,
                min(
                    _intensity(body.metabolic_rate, 0.7, 0.3),
                    _intensity(body.cpu_temp_c, 75.0, 15.0),
                ),
            )
        return peak

    def _vigilance_arm(
        self,
        body: BodyState,
        levels: dict[str, float],
        sent: list,
    ) -> None:
        """Work that is not yet distress — LC/ACh alerting pushes."""
        if body.io_activity > 0.10:
            i = _intensity(body.io_activity, 0.10, 0.40)
            self._push(CHEM_NOREPINEPHRINE, 0.003 * i, levels, sent)
        if body.core_activity > 0.40:
            i = _intensity(body.core_activity, 0.40, 0.60)
            self._push(CHEM_NOREPINEPHRINE, 0.002 * i, levels, sent)
        # DRAM switching = memory traffic → ACh (encoding/retrieval).
        if body.dram_activity > 0.30:
            i = _intensity(body.dram_activity, 0.30, 0.70)
            self._push(CHEM_ACETYLCHOLINE, 0.003 * i, levels, sent)
        # Hardware prediction errors → small LC orienting pulses.
        if body.branch_miss_rate > 0.08:
            i = _intensity(body.branch_miss_rate, 0.08, 0.12)
            self._push(CHEM_NOREPINEPHRINE, 0.002 * i, levels, sent)
        if body.cache_miss_rate > 0.15:
            i = _intensity(body.cache_miss_rate, 0.15, 0.25)
            self._push(CHEM_NOREPINEPHRINE, 0.002 * i, levels, sent)

    def _parasympathetic_arm(
        self,
        body: BodyState,
        presence: SensorPresence | None,
        levels: dict[str, float],
        sent: list,
    ) -> None:
        """Vagal arm: rest is evidence — calming pushes and
        alerting-tone withdrawal."""
        # cpuidle residency is the machine's vagal signal — a core
        # parked in a real C-state is positive evidence of a
        # hypometabolic substrate. Gated on the sensor: absence of
        # the measurement is not evidence of calm.
        rested = self._verified(presence, "cpuidle state counters") and body.rest > 0.0
        if rested and body.rest > 0.5 and body.cpu_temp_c < 60.0 and body.stress_load < 0.25:
            self._push(CHEM_SEROTONIN, 0.002, levels, sent)
            self._push(CHEM_GABA, 0.001, levels, sent)
            # eCB — the homeostatic brake; the documented stress-
            # termination signal (Hill et al. 2011).
            self._push(CHEM_ENDOCANNABINOID, 0.002, levels, sent)
        if rested and body.rest > 0.7:
            # Deepest rest withdraws alerting tone — reciprocal
            # autonomic antagonism, not silence. Scaled by level:
            # tone can only be withdrawn where it exists.
            self._push(CHEM_NOREPINEPHRINE, -0.002, levels, sent)
            self._push(CHEM_GLUTAMATE, -0.001, levels, sent)
            self._push(CHEM_OREXIN, -0.001, levels, sent)
            self._push(CHEM_HISTAMINE, -0.001, levels, sent)
        # Cooling after strain → eCB (glucocorticoid feedback
        # terminates stress responses by mobilizing endocannabinoids,
        # Evanson et al. 2010). Sensor-gated on a real thermometer.
        if self._verified(presence, "temperature") and body.temp_trend < -0.15:
            self._push(CHEM_ENDOCANNABINOID, 0.002, levels, sent)

        # Contentment: low load + cool — the healthy-body baseline.
        if body.stress_load < 0.25 and body.cpu_temp_c < 60.0 and body.cognitive_load < 0.60:
            self._push(CHEM_SEROTONIN, 0.002, levels, sent)
            self._push(CHEM_GABA, 0.001, levels, sent)
