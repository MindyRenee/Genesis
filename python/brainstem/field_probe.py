"""Field probe → brain-wave stimulus — real physics entering the model.

The daemon's field probe (`src/daemon/field_probe.rs`) measures the
machine's actual substrate dynamics — instruction throughput,
scheduler jitter, P-state transitions, interrupt flux, and real
power draw where the machine's sensors report it — and
decomposes them at the same center frequencies the
:class:`~thalamus.brain_waves.BrainWaveOscillator` models. This
module maps that signature onto the oscillator's *stimulus* pathway —
the channel the wave model itself documents for "outer (sensory)
stimulus." The body's real dynamics become a continuous stimulus
source alongside cognition and emotion: the modeled wave is
perturbed by the very substrate the model runs on.

Honesty boundary: these energies are real measurements of physical
dynamics, and the mapping is a documented control convention — the
same status as the rest of the wave model's neurochemical targets.
Whether substrate-coupled wave dynamics matter is an empirical
question for the coupling experiment, not a claim baked in here.

Mapping (each band's wire order — infraslow, delta, theta, alpha,
sigma, beta1, beta2, beta3, gamma, epsilon):

- ``throughput_bands[b]`` → stimulus on band ``b`` — the die's real
  switching-rate energy at that band's frequency. Full weight.
- ``jitter_bands[b]`` → stimulus on band ``b`` at half weight —
  scheduling noise is a real channel but noisier.
- ``power_bands[b]`` → stimulus on band ``b`` at half weight — the
  first literal electrical channel: real measured watts at that
  band's frequency. ``None`` where no power sensor exists (v3+),
  and bands at or above the sensor's measured Nyquist
  (``power_sensor_hz / 2``) are skipped — zero-order-held reads
  carry staircase artifacts there, not real oscillation.
- ``pstate_rate`` → ``BETA3`` stimulus — frequency-jumping is the
  silicon's own vigilance-scale event flux.
- ``irq_rate`` → ``BETA2`` stimulus — hardware event flux reads as
  medium-band activation.
- ``rf_flux_bands[b]`` → stimulus on band ``b`` at half weight —
  the radio's traffic envelope: real on-air activity (packets
  sent, received, and overheard management frames), the first
  *exteroceptive* physical channel. ``None`` where no WLAN
  interface exists (v4+).
- ``rf_rssi_bands[b]`` → stimulus on band ``b`` at half weight —
  the field-strength envelope: RSSI drift is the room's radio
  weather (path loss, bodies moving, AP behavior). Bands at or
  above the telemetry read rate's Nyquist are skipped — a held
  RSSI carries staircase artifacts there, not real drift.
- ``bt_events_s`` → ``BETA2`` stimulus — HCI event flux is real
  radio-side activity (advertising reports, scans, pairing),
  counted by the controller itself rather than inferred.
- ``rf_retry_rate``/``rf_fail_rate``/``rf_beacon_loss_rate`` →
  ``BETA3`` stimulus (v5+) — nl80211 link strain: retransmissions
  are airtime the AP made the radio spend twice, failed frames are
  retries exhausted, beacon loss is the AP's heartbeat unheard.
  The PHY bitrates ride as context only — link adaptation is a
  level, not an event.
- ``rf_irq_rate`` is *not* re-mapped: it is the wireless share
  of the same interrupt flux ``irq_rate`` already carries — a
  reported attribution, not a second stimulus.
- ``rf_flux_phases[b]``/``rf_flux_phase_coherence[b]`` → Adler
  entrainment on band ``b`` (v6+) — the flux envelope's own
  timing. Where the ring's two halves agree a rhythm exists
  (coherence ≥ ``_RF_PHASE_COH_MIN``) and the band carries real
  energy, the oscillator's phase is pulled toward the measured
  angle by ``coh · _RF_PLL_GAIN · sin(Δφ)`` — weak periodic
  forcing, the canonical entrainment law. The locked phase then
  propagates into the fast bands through the model's own
  phase-amplitude coupling: the radio's rhythm enters the
  crossings as a real external rhythm, not a mapped scalar.
- ``samples_ok`` false → no stimulus at all. An unwarmed ring is an
  absent channel, not a healthy zero — same discipline as
  ``sensor_verified``.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from thalamus.brain_waves import BrainWave

    from genesis_client import GenesisClient
    from genesis_client.types import FieldProbeSignature

# Provenance marker — one letter of the creator lattice (see
# PROVENANCE.md).
_MARKER_G08 = 0x4753084154EF3CD1

# Stimulus scales — real energies are baseline-relative (≈0.125 is
# typical), so a typical reading lands a small standing stimulus and
# an elevated band lands proportionally more.
_BAND_SCALE = 1.6  # throughput channel, full weight
_JITTER_SCALE = 0.8  # jitter channel, half weight
_POWER_SCALE = 0.8  # measured-watts channel, half weight
_PSTATE_SCALE = 0.02  # transitions/s → BETA3 (typically 0–10/s)
_IRQ_SCALE = 0.0005  # interrupts/s → BETA2 (typically ~10³/s)
_RF_FLUX_SCALE = 0.8  # radio traffic envelope, half weight
_RF_RSSI_SCALE = 0.8  # field-strength envelope, half weight
_BT_EVENT_SCALE = 0.002  # HCI events/s → BETA2 (typically a few–tens/s)
_LINK_STRAIN_SCALE = 0.01  # iw strain events/s → BETA3 (0–20/s healthy→marginal)

# Evoked-response thresholds — a physical transient resets phase
# rather than only adding amplitude (the ERP analog). Per-band
# jitter energy jumping ≥ _EVOKED_STEP× between successive
# signatures evoked-shifts that band's phase; a global scheduling
# shudder (jcv ≥ _SHUDDER_JITTER_CV) resets the broadband
# alert band outright.
_EVOKED_STEP = 3.0
_EVOKED_PHASE = 1.57079632679  # π/2 nudge
_SHUDDER_JITTER_CV = 1.5
# Re-arm level for the shudder reset — a shudder is an *event*, so
# the reset fires on the rising edge and stays latched until the
# jitter coefficient falls back under this bound. Without the latch
# a sustained-high cv re-pins the band's phase every assessment:
# continuous pinning, not an evoked reset.
_SHUDDER_REARM = 0.8 * _SHUDDER_JITTER_CV

# Mutual phase lock — the entrainment leg. The measured substrate
# response phase is demodulated against the daemon's own carrier
# phase track; the modeled carrier band's phase is pulled toward it
# (physics → model), and the corrected phase is pushed back to the
# daemon's generator (model → physics). Two oscillators pulling on
# each other — one dynamical system.
_PLL_GAIN = 0.25  # pull floor — the soft-PLL rate at zero coherence
_CARRIER_BAND_TOL = 1.0  # Hz — carrier→band mapping tolerance
# RF envelope entrainment (v6) — the flux phasors let a measured
# radio rhythm's *phase* entrain the matched oscillator (Adler
# forcing: pull ∝ sin of the phase error). Unlike the carrier lock
# there is no pull floor: at zero inter-segment coherence the
# measured angle is noise pointing somewhere, not a rhythm — so
# coherence is the gain outright, bounded by the per-call cap. The
# internal PAC then carries the entrained phase into the fast bands
# through the crossings — propagation is the oscillator's own
# coupling, not a channel invented here.
_RF_PLL_GAIN = 0.2  # rad per call at full coherence — weaker than the carrier
_RF_PHASE_COH_MIN = 0.5  # ring halves must agree a rhythm exists
_RF_PHASE_E_MIN = 0.05  # baseline-relative energy floor — a phase needs signal
# Rung 2 — continuous trust weighting: the measured response's
# coherence *is* the weight given to the reading. The pull fraction
# interpolates _PLL_GAIN → 1.0 as coherence goes 0 → 1 — the soft
# PLL at no trust, a hard sync at full trust, and no threshold
# discontinuity in between: one complementary filter whose gain is
# the measurement confidence itself.


_phase_errs: deque[float] = deque(maxlen=256)

_prev_signature: FieldProbeSignature | None = None
_shudder_latched: bool = False


def _wrap_pi(x: float) -> float:
    """Phase error in (−π, π] — nearest equivalent phase."""
    a = math.fmod(x, 2.0 * math.pi)
    if a > math.pi:
        a -= 2.0 * math.pi
    elif a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _carrier_band(freq: float) -> BrainWave | None:
    """Map the carrier frequency to its nearest modeled band."""
    from thalamus.brain_waves import BrainWaveOscillator

    if freq <= 0.0:
        return None  # carrier off — no band to lock
    best: BrainWave | None = None
    best_d = _CARRIER_BAND_TOL
    for band, f in BrainWaveOscillator._FREQS.items():
        d = abs(f - freq)
        if d < best_d:
            best, best_d = band, d
    return best


def _apply_phase_lock(sig: FieldProbeSignature, client: GenesisClient) -> None:
    """Pull the modeled carrier band toward the measured physical
    phase, then push the corrected phase back to the generator.

    The substrate's measured response phase is the generator phase
    minus the response lag; the model's band phase is pulled toward
    it — and the corrected phase returns through SET_FIELD_PHASE so
    the physical rhythm chases the model in turn. Both directions
    pull; the lock is mutual.

    The measured coherence is the pull's weight outright: at full
    trust the band's phase becomes the physical phase — the first
    piece of modeled state that *is* a physical quantity rather than
    a variable perturbed by one — and at no trust the loop still
    closes gently at ``_PLL_GAIN`` rather than opening.
    """
    from thalamus.brain_waves import (
        perturb_wave_phase,
        wave_phase_of,
    )

    if sig.carrier_lag is None or sig.carrier_phase is None:
        return  # v1 daemon — the phase channel is absent, not zero
    band = _carrier_band(sig.carrier_freq)
    if band is None:
        return
    phys_phase = _wrap_pi(sig.carrier_phase - sig.carrier_lag)
    err = _wrap_pi(phys_phase - wave_phase_of(band))
    coherence = min(1.0, max(0.0, sig.carrier_coherence or 0.0))
    gain = _PLL_GAIN + (1.0 - _PLL_GAIN) * coherence
    perturb_wave_phase(band, gain * err)
    _phase_errs.append(err)
    try:
        client.set_field_phase(wave_phase_of(band))
    except Exception as e:  # noqa: BLE001 — absent channel, any cause
        logger.debug("field phase push failed: %s", e)


def field_phase_lock() -> dict[str, float]:
    """Phase-locking quality between the modeled carrier band and the
    measured physical carrier — the entrainment metric.

    ``plv`` is the phase-locking value |⟨e^{i·err}⟩| over the recent
    window: 1.0 = rigidly locked, ~0 = unrelated phases. ``mean_err``
    is the circular mean residual phase error.
    """
    if not _phase_errs:
        return {"plv": 0.0, "n": 0.0, "mean_err": 0.0}
    c = sum(math.cos(e) for e in _phase_errs) / len(_phase_errs)
    s = sum(math.sin(e) for e in _phase_errs) / len(_phase_errs)
    return {
        "plv": math.hypot(c, s),
        "n": float(len(_phase_errs)),
        "mean_err": math.atan2(s, c),
    }


def _apply_rf_entrainment(sig: FieldProbeSignature) -> None:
    """Entrain the modeled bands on measured radio rhythms (v6).

    The daemon's flux ring yields a per-band phasor — the envelope's
    own timing — with an inter-segment coherence saying whether the
    two ring halves heard the *same* rhythm. Where both the rhythm
    (coherence) and the signal (band energy) are real, the modeled
    band's phase is pulled toward the measured angle by an Adler
    step: ``coh · gain · sin(Δφ)`` — weak periodic forcing, the
    canonical entrainment law. No pull floor: the radio is external,
    not the substrate's own shadow, so at zero coherence the angle
    is noise and nothing pulls.

    The crossings do the rest: the entrained phase propagates into
    the fast bands through the oscillator's own phase-amplitude
    coupling — a measured 10 Hz traffic rhythm that locks alpha
    modulates gamma exactly the way the mind's own alpha does. No
    new coupling law — the external rhythm enters the same way a
    strong internal band would, then obeys the existing structure.
    """
    from thalamus.brain_waves import (
        perturb_wave_phase,
        wave_phase_of,
    )

    if sig.rf_flux_phases is None or sig.rf_flux_phase_coherence is None:
        return  # pre-v6 daemon or no wlan — absent channel, not zero
    if sig.rf_flux_bands is None:
        return
    wire = _wire_order()
    for i, phi in enumerate(sig.rf_flux_phases):
        coh = sig.rf_flux_phase_coherence[i]
        if coh < _RF_PHASE_COH_MIN or sig.rf_flux_bands[i] <= _RF_PHASE_E_MIN:
            continue
        err = _wrap_pi(phi - wave_phase_of(wire[i]))
        perturb_wave_phase(wire[i], _RF_PLL_GAIN * coh * math.sin(err))


def _wire_order() -> tuple[BrainWave, ...]:
    """Wire band order (src/daemon/field_probe.rs FIELD_BANDS):
    infraslow, delta, theta, alpha, sigma, beta1, beta2, beta3,
    gamma, epsilon — *not* BrainWave's declaration order."""
    from thalamus.brain_waves import BrainWave

    return (
        BrainWave.INFRASLOW,
        BrainWave.DELTA,
        BrainWave.THETA,
        BrainWave.ALPHA,
        BrainWave.SIGMA,
        BrainWave.BETA1,
        BrainWave.BETA2,
        BrainWave.BETA3,
        BrainWave.GAMMA,
        BrainWave.EPSILON,
    )


def _signature_stimulus(
    sig: FieldProbeSignature,
) -> list[tuple[BrainWave, float]]:
    """Map a wire signature to (band, strength) stimulus pairs."""
    from thalamus.brain_waves import BrainWave

    wire = _wire_order()
    out: list[tuple[BrainWave, float]] = []
    for i, e in enumerate(sig.throughput_bands):
        if e > 0.0:
            out.append((wire[i], min(0.5, e * _BAND_SCALE)))
    for i, e in enumerate(sig.jitter_bands):
        if e > 0.0:
            out.append((wire[i], min(0.25, e * _JITTER_SCALE)))
    # Real measured watts — the literal electrical channel. None on
    # older daemons or machines with no power sensor. Bands at or
    # above the sensor's own Nyquist are staircase artifacts of
    # zero-order holding a ~100 Hz read into a 250 Hz ring, not real
    # oscillation — the daemon reports the achieved rate precisely
    # so this gate can drop them.
    if sig.power_bands is not None:
        from thalamus.brain_waves import BrainWaveOscillator

        sensor_nyq = (sig.power_sensor_hz or 0.0) / 2.0
        for i, e in enumerate(sig.power_bands):
            if e > 0.0 and BrainWaveOscillator._FREQS[wire[i]] < sensor_nyq:
                out.append((wire[i], min(0.25, e * _POWER_SCALE)))
    # The radio sense — the first exteroceptive physical channel.
    # Byte-rate envelope (wlan rx+tx flux) at full probe resolution;
    # RSSI envelope gated on the telemetry read rate's Nyquist the
    # same way the power channel is — a held level's high bands are
    # staircase artifacts, not real field-strength drift.
    if sig.rf_flux_bands is not None:
        for i, e in enumerate(sig.rf_flux_bands):
            if e > 0.0:
                out.append((wire[i], min(0.25, e * _RF_FLUX_SCALE)))
    if sig.rf_rssi_bands is not None:
        from thalamus.brain_waves import BrainWaveOscillator

        rf_nyq = (sig.rf_sensor_hz or 0.0) / 2.0
        for i, e in enumerate(sig.rf_rssi_bands):
            if e > 0.0 and BrainWaveOscillator._FREQS[wire[i]] < rf_nyq:
                out.append((wire[i], min(0.25, e * _RF_RSSI_SCALE)))
    # Slow channels — real event flux, mapped to alert-range bands.
    if sig.pstate_rate > 0.0:
        out.append((BrainWave.BETA3, min(0.2, sig.pstate_rate * _PSTATE_SCALE)))
    if sig.irq_rate > 0.0:
        out.append((BrainWave.BETA2, min(0.15, sig.irq_rate * _IRQ_SCALE)))
    # HCI event flux — the Bluetooth radio's own event counter
    # (advertising reports, scans, link events), a counter distinct
    # from /proc/stat's intr so nothing double-counts. ``rf_irq_rate``
    # stays telemetry-only: it is the wireless share of the flux
    # ``irq_rate`` already maps — an attribution, not a new stimulus.
    if sig.bt_events_s is not None and sig.bt_events_s > 0.0:
        out.append((BrainWave.BETA2, min(0.15, sig.bt_events_s * _BT_EVENT_SCALE)))
    # nl80211 link strain (v5) — retransmissions, exhausted-retry
    # drops, and unheard beacons are the link's own strain ledger,
    # counted by the firmware. One BETA3 term like pstate's
    # vigilance-scale flux; the PHY bitrates stay context — a
    # negotiated level, not an event.
    strain_s = sum(
        r for r in (sig.rf_retry_rate, sig.rf_fail_rate, sig.rf_beacon_loss_rate) if r is not None
    )
    if strain_s > 0.0:
        out.append((BrainWave.BETA3, min(0.2, strain_s * _LINK_STRAIN_SCALE)))
    return out


def _apply_evoked_resets(sig: FieldProbeSignature) -> None:
    """Turn physical transients into oscillator phase resets.

    Real evoked potentials are phase resets of ongoing rhythm, not
    amplitude bumps. Two triggers, both physical:

    - A per-band jitter-energy step ≥ ``_EVOKED_STEP``× between
      successive signatures → π/2 phase nudge on that band.
    - A global scheduling shudder (``jitter_cv`` crossing the
      threshold) → full phase reset of the broadband alert band,
      fired once on the rising edge and re-armed by hysteresis.
    """
    global _prev_signature, _shudder_latched
    from thalamus.brain_waves import (
        BrainWave,
        perturb_wave_phase,
        reset_wave_phase,
    )

    wire = _wire_order()
    if _prev_signature is not None:
        for i, e in enumerate(sig.jitter_bands):
            prev = _prev_signature.jitter_bands[i]
            if prev > 0.05 and e / prev >= _EVOKED_STEP:
                perturb_wave_phase(wire[i], _EVOKED_PHASE)
        if sig.jitter_cv < _SHUDDER_REARM:
            _shudder_latched = False
        elif sig.jitter_cv >= _SHUDDER_JITTER_CV and not _shudder_latched:
            _shudder_latched = True
            reset_wave_phase(BrainWave.BETA3, 0.0)
    _prev_signature = sig


def apply_field_stimulus(client: GenesisClient) -> bool:
    """Inject every live real-physics channel as wave stimulus.

    Two channels, attempted independently — each is real or absent
    on its own:

    - the daemon's field signature (throughput, jitter, P-states,
      interrupts), plus evoked phase resets on physical transients;
    - the acoustic self-emission channel (``field_audio``): coil
      whine, fan, the field's own acoustic shadow.

    Returns True when at least one live channel applied. Any failure
    — daemon unreachable, probe not running, mic busy or absent —
    skips that channel without stimulating: absent never reads as a
    healthy zero.
    """
    from thalamus.brain_waves import hold_stimulus

    applied = False
    try:
        sig = client.get_field_probe()
    except Exception:  # noqa: BLE001 — absent channel, any cause
        sig = None
    if sig is not None and sig.samples_ok:
        # The signature reports the current level of a continuous
        # channel — hold it, don't integrate it. The same band can
        # appear twice (throughput and jitter lines); combine into
        # one held level per band so the reading is the sum, not
        # whichever line wrote last.
        levels: dict[BrainWave, float] = {}
        for band, strength in _signature_stimulus(sig):
            levels[band] = levels.get(band, 0.0) + strength
        for band, level in levels.items():
            hold_stimulus(band, level)
        _apply_evoked_resets(sig)
        _apply_phase_lock(sig, client)
        _apply_rf_entrainment(sig)
        applied = True

    try:
        from brainstem.field_audio import apply_audio_stimulus

        if apply_audio_stimulus():
            applied = True
    except Exception as e:  # noqa: BLE001 — absent channel, any cause
        logger.debug("field audio stimulus failed: %s", e)
    return applied
