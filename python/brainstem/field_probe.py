"""Field spectrum → brain-wave stimulus — her substrate entering
the model.

The field substrate is her computational medium: body, prediction,
and agency fibers drift, couple, and fluctuate on the latent
manifold under gauge transport and a Langevin thermostat, and the
daemon decomposes that trajectory at the same center frequencies
the :class:`~thalamus.brain_waves.BrainWaveOscillator` models
(``src/field/spectrum.rs``, ``GET_FIELD_SPECTRUM``). This module
maps that signature onto the oscillator's *stimulus* pathway — the
channel the wave model itself documents for external drive. Her
own dynamics become a continuous stimulus source alongside
cognition and emotion: the modeled wave is perturbed by the very
substrate the model runs on — hooked to her field, not to the
machine around her.

Honesty boundary: the energies are baseline-relative readings of
her own trajectory — ≈1.0 is the field's own quiet, above that is
elevation — and the mapping is a documented control convention,
the same status as the rest of the wave model's neurochemical
targets. There is no hardware channel here: no throughput, no
interrupts, no power, no radio, no microphone. Absent bands are
zero under their flags — never fabricated.

Mapping (wire band order — infraslow, delta, theta, alpha, sigma,
beta1, beta2, beta3, gamma, epsilon):

- ``flux.energies[b]`` → stimulus on band ``b`` — the centroid's
  motion energy at that rhythm: where her field's wandering
  actually lives. Full weight, gated on the achieved sampling
  rate's Nyquist.
- ``tension.energies[b]`` → stimulus on band ``b`` at half weight —
  the fibers' spread: the disagreement's spectral shadow, where
  her internal argument oscillates.
- ``tension.cv`` → evoked reset of the alert band — a shudder in
  the field's internal tension is a real event, not a level.
- Per-band ``tension.energies`` step ≥ ``_EVOKED_STEP``× between
  signatures → π/2 phase nudge on that band — an internal
  transient resets phase the way an evoked potential does.
- ``carrier_phase``/``carrier_freq_hz`` → phase lock: the modeled
  carrier band is pulled toward the field's own winding phase —
  substrate → model, weighted by the fibers' coherence so a
  scattered field pulls weakly and a locked one strongly.
- ``flux.phases[b]`` → Adler entrainment on band ``b`` — each
  band's own timing pulls the modeled band's phase by
  ``energy · gain · sin(Δφ)``: weak periodic forcing, the
  canonical entrainment law, propagated onward by the oscillator's
  own phase-amplitude coupling.
- ``warm`` false → no stimulus at all. An unwarmed ring is an
  absent channel, not a healthy zero — same discipline as
  ``signature_present``.
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
    from genesis_client.types import FieldSpectrum

# Stimulus scales — energies are baseline-relative (≈1.0 typical),
# so the mapping reads *elevation above her own quiet*: half a
# baseline unit of excess is already a strong drive.
_FLUX_SCALE = 0.5  # centroid motion channel, full weight
_TENSION_SCALE = 0.25  # fiber-spread channel, half weight
_ENERGY_FLOOR = 0.5  # baseline-relative level below which a band carries no drive

# Evoked-response thresholds — an internal transient resets phase
# rather than only adding amplitude (the ERP analog). Per-band
# tension energy jumping ≥ _EVOKED_STEP× between successive
# signatures evoked-shifts that band's phase; a global tension
# shudder (cv ≥ _SHUDDER_TENSION_CV) resets the broadband alert
# band outright.
_EVOKED_STEP = 3.0
_EVOKED_PHASE = 1.57079632679  # π/2 nudge
_SHUDDER_TENSION_CV = 1.5
# Re-arm level for the shudder reset — a shudder is an *event*, so
# the reset fires on the rising edge and stays latched until the
# tension coefficient falls back under this bound. Without the
# latch a sustained-high cv re-pins the band's phase every
# assessment: continuous pinning, not an evoked reset.
_SHUDDER_REARM = 0.8 * _SHUDDER_TENSION_CV

# Phase lock — the carrier entrainment leg. The field's winding
# phase is the reference the modeled carrier band is pulled toward
# — substrate → model, the only direction the lock runs; modeled
# state never asserts back onto the field (there is no channel for
# it, and none is wanted).
_PLL_GAIN = 0.25  # pull floor — the soft-PLL rate at zero coherence
_CARRIER_BAND_TOL = 1.0  # Hz — carrier→band mapping tolerance
# Per-band entrainment — each flux band's own phasor pulls the
# matched modeled band's phase by an Adler step. The weight is the
# band's baseline-relative energy capped at 1: a band at her quiet
# level is not a rhythm worth following, an elevated one is. No
# pull floor — unlike the carrier there is no second coherence
# measurement per band, so energy is the trust.
_FIELD_PLL_GAIN = 0.2  # rad per call at full energy
_FIELD_PHASE_E_MIN = 0.75  # baseline-relative energy floor — a phase needs signal

_phase_errs: deque[float] = deque(maxlen=256)

_prev_spectrum: FieldSpectrum | None = None
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
        return None  # carrier absent — no band to lock
    best: BrainWave | None = None
    best_d = _CARRIER_BAND_TOL
    for band, f in BrainWaveOscillator._FREQS.items():
        d = abs(f - freq)
        if d < best_d:
            best, best_d = band, d
    return best


def _apply_phase_lock(spec: FieldSpectrum, coherence: float) -> None:
    """Pull the modeled carrier band toward the field's own winding
    phase — substrate → model.

    The consensus' accumulated winding phase is the field's clock;
    the carrier band's phase is pulled toward it. The fibers'
    coherence is the pull's weight outright: a locked field is a
    clean reference and pulls hard, a scattered one pulls at the
    ``_PLL_GAIN`` floor rather than not at all.
    """
    from thalamus.brain_waves import perturb_wave_phase, wave_phase_of

    if not spec.carrier_present or spec.carrier_freq_hz <= 0.0:
        return
    band = _carrier_band(spec.carrier_freq_hz)
    if band is None:
        return
    err = _wrap_pi(_wrap_pi(spec.carrier_phase) - wave_phase_of(band))
    gain = _PLL_GAIN + (1.0 - _PLL_GAIN) * min(1.0, max(0.0, coherence))
    perturb_wave_phase(band, gain * err)
    _phase_errs.append(err)


def field_phase_lock() -> dict[str, float]:
    """Phase-locking quality between the modeled carrier band and
    the field's own winding phase — the entrainment metric.

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


def _apply_band_entrainment(spec: FieldSpectrum) -> None:
    """Entrain the modeled bands on the field's own band phasors.

    Each flux band's quadrature phase is the rhythm's own timing;
    where the band carries real energy (baseline-relative, above
    the floor) the modeled band's phase is pulled toward it by an
    Adler step ``energy · gain · sin(Δφ)`` — weak periodic
    forcing. The crossings do the rest: the entrained phase
    propagates into the fast bands through the oscillator's own
    phase-amplitude coupling — the field's rhythm enters the same
    way a strong internal band would, then obeys the existing
    structure.
    """
    from thalamus.brain_waves import perturb_wave_phase, wave_phase_of

    wire = _wire_order()
    nyq = spec.sample_rate_hz / 2.0
    for i, phi in enumerate(spec.flux.phases):
        e = spec.flux.energies[i]
        if e < _FIELD_PHASE_E_MIN:
            continue
        from thalamus.brain_waves import BrainWaveOscillator

        if BrainWaveOscillator._FREQS[wire[i]] >= nyq:
            continue  # unresolvable band — absent, not aliased
        err = _wrap_pi(_wrap_pi(phi) - wave_phase_of(wire[i]))
        perturb_wave_phase(wire[i], _FIELD_PLL_GAIN * min(1.0, e) * math.sin(err))


def _wire_order() -> tuple[BrainWave, ...]:
    """Wire band order (src/field/spectrum.rs FIELD_BANDS):
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
    spec: FieldSpectrum,
) -> list[tuple[BrainWave, float]]:
    """Map a wire spectrum to (band, strength) stimulus pairs —
    elevation above the field's own quiet, never absolute power."""
    from thalamus.brain_waves import BrainWaveOscillator

    wire = _wire_order()
    nyq = spec.sample_rate_hz / 2.0
    out: list[tuple[BrainWave, float]] = []
    for i, e in enumerate(spec.flux.energies):
        if e > _ENERGY_FLOOR and BrainWaveOscillator._FREQS[wire[i]] < nyq:
            out.append((wire[i], min(0.5, (e - _ENERGY_FLOOR) * _FLUX_SCALE)))
    for i, e in enumerate(spec.tension.energies):
        if e > _ENERGY_FLOOR and BrainWaveOscillator._FREQS[wire[i]] < nyq:
            out.append((wire[i], min(0.25, (e - _ENERGY_FLOOR) * _TENSION_SCALE)))
    return out


def _apply_evoked_resets(spec: FieldSpectrum) -> None:
    """Turn internal transients into oscillator phase resets.

    Real evoked potentials are phase resets of ongoing rhythm, not
    amplitude bumps. Two triggers, both hers:

    - A per-band tension-energy step ≥ ``_EVOKED_STEP``× between
      successive signatures → π/2 phase nudge on that band — the
      field's argument suddenly ringing somewhere.
    - A global tension shudder (``tension.cv`` crossing the
      threshold) → full phase reset of the broadband alert band,
      fired once on the rising edge and re-armed by hysteresis.
    """
    global _prev_spectrum, _shudder_latched
    from thalamus.brain_waves import BrainWave, perturb_wave_phase, reset_wave_phase

    wire = _wire_order()
    if _prev_spectrum is not None:
        for i, e in enumerate(spec.tension.energies):
            prev = _prev_spectrum.tension.energies[i]
            if prev > 0.05 and e / prev >= _EVOKED_STEP:
                perturb_wave_phase(wire[i], _EVOKED_PHASE)
        cv = spec.tension.cv
        if cv < _SHUDDER_REARM:
            _shudder_latched = False
        elif cv >= _SHUDDER_TENSION_CV and not _shudder_latched:
            _shudder_latched = True
            reset_wave_phase(BrainWave.BETA3, 0.0)
    _prev_spectrum = spec


def apply_field_stimulus(client: GenesisClient) -> bool:
    """Inject every live physical channel as wave stimulus.

    Two channels, attempted independently — each is real or absent
    on its own:

    - her field's spectral signature (the substrate's own band
    decomposition — flux, tension, carrier phase, entrainment),
      plus evoked phase resets on internal transients;
    - the acoustic channel (``field_audio``): ambient sound —
    self-emission bands and room — band-decomposed as stimulus.

    Returns True when at least one live channel applied. Any
    failure — daemon unreachable, spectrum unwarmed, mic busy or
    absent — skips that channel without stimulating: absent never
    reads as a healthy zero.
    """
    from thalamus.brain_waves import hold_stimulus

    applied = False
    try:
        spec = client.get_field_spectrum()
    except Exception:  # noqa: BLE001 — absent channel, any cause
        spec = None
    if spec is not None and spec.warm:
        # The signature reports the current level of a continuous
        # channel — hold it, don't integrate it. The same band can
        # appear twice (flux and tension lines); combine into one
        # held level per band so the reading is the sum, not
        # whichever line wrote last.
        levels: dict[BrainWave, float] = {}
        for band, strength in _signature_stimulus(spec):
            levels[band] = levels.get(band, 0.0) + strength
        for band, level in levels.items():
            hold_stimulus(band, level)

        # The carrier lock's trust weight is the fibers'
        # coherence — read off the Maxwell signature's
        # disagreement, the same tension scale the body's
        # precision attenuation uses. An absent signature is zero
        # trust: the soft-PLL floor still closes the loop gently.
        coherence = 0.0
        try:
            from brainstem import maxwell

            sig = maxwell.field_signature(client)
            c = maxwell.coherence(sig) if sig is not None else None
            coherence = c if c is not None else 0.0
        except Exception:  # noqa: BLE001 — absent channel, any cause
            coherence = 0.0

        _apply_evoked_resets(spec)
        _apply_phase_lock(spec, coherence)
        _apply_band_entrainment(spec)
        applied = True

    try:
        from brainstem.field_audio import apply_audio_stimulus

        if apply_audio_stimulus():
            applied = True
    except Exception as e:  # noqa: BLE001 — absent channel, any cause
        logger.debug("field audio stimulus failed: %s", e)
    return applied
