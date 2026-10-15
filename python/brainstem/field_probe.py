"""Field probe → brain-wave stimulus — real physics entering the model.

The daemon's field probe (`src/daemon/field_probe.rs`) measures the
machine's actual substrate dynamics — instruction throughput,
scheduler jitter, P-state transitions, interrupt flux — and
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

Mapping (each band's wire order matches ``BrainWave`` definition
order — epsilon, delta, theta, alpha, sigma, beta1, beta2, beta3,
gamma, lambda):

- ``throughput_bands[b]`` → stimulus on band ``b`` — the die's real
  switching-rate energy at that band's frequency. Full weight.
- ``jitter_bands[b]`` → stimulus on band ``b`` at half weight —
  scheduling noise is a real channel but noisier.
- ``pstate_rate`` → ``BETA3`` stimulus — frequency-jumping is the
  silicon's own vigilance-scale event flux.
- ``irq_rate`` → ``BETA2`` stimulus — hardware event flux reads as
  medium-band activation.
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

# Stimulus scales — real energies are baseline-relative (≈0.125 is
# typical), so a typical reading lands a small standing stimulus and
# an elevated band lands proportionally more.
_BAND_SCALE = 1.6      # throughput channel, full weight
_JITTER_SCALE = 0.8    # jitter channel, half weight
_PSTATE_SCALE = 0.02   # transitions/s → BETA3 (typically 0–10/s)
_IRQ_SCALE = 0.0005    # interrupts/s → BETA2 (typically ~10³/s)

# Evoked-response thresholds — a physical transient resets phase
# rather than only adding amplitude (the ERP analog). Per-band
# jitter energy jumping ≥ _EVOKED_STEP× between successive
# signatures evoked-shifts that band's phase; a global scheduling
# shudder (jcv ≥ _SHUDDER_JITTER_CV) resets the broadband
# alert band outright.
_EVOKED_STEP = 3.0
_EVOKED_PHASE = 1.57079632679  # π/2 nudge
_SHUDDER_JITTER_CV = 1.5

# Mutual phase lock — the entrainment leg. The measured substrate
# response phase is demodulated against the daemon's own carrier
# phase track; the modeled carrier band's phase is pulled toward it
# (physics → model), and the corrected phase is pushed back to the
# daemon's generator (model → physics). Two oscillators pulling on
# each other — one dynamical system.
_PLL_GAIN = 0.25          # bounded pull per assessment
_CARRIER_BAND_TOL = 1.0   # Hz — carrier→band mapping tolerance
# Rung 2 — hybrid state: once the measured response is coherent
# enough that the phase reading is trustworthy, the modeled band's
# phase *is* the physical phase — a hard sync, not a pull. Below the
# gate the model free-runs on the soft PLL.
_SYNC_COHERENCE = 0.20

_phase_errs: deque[float] = deque(maxlen=256)

_prev_signature: FieldProbeSignature | None = None


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

    When the measured response is coherent enough to trust
    (``_SYNC_COHERENCE``), the pull becomes a sync: the band's phase
    is set to the measured physical phase outright — the first piece
    of modeled state that *is* a physical quantity rather than a
    variable perturbed by one.
    """
    from thalamus.brain_waves import (
        perturb_wave_phase,
        reset_wave_phase,
        wave_phase_of,
    )

    if sig.carrier_lag is None or sig.carrier_phase is None:
        return  # v1 daemon — the phase channel is absent, not zero
    band = _carrier_band(sig.carrier_freq)
    if band is None:
        return
    phys_phase = _wrap_pi(sig.carrier_phase - sig.carrier_lag)
    err = _wrap_pi(phys_phase - wave_phase_of(band))
    if (sig.carrier_coherence or 0.0) >= _SYNC_COHERENCE:
        reset_wave_phase(band, phys_phase)
    else:
        perturb_wave_phase(band, _PLL_GAIN * err)
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


def _wire_order() -> tuple[BrainWave, ...]:
    """Wire band order (src/daemon/field_probe.rs FIELD_BANDS):
    epsilon, delta, theta, alpha, sigma, beta1, beta2, beta3,
    gamma, lambda — *not* BrainWave's declaration order."""
    from thalamus.brain_waves import BrainWave

    return (
        BrainWave.EPSILON,
        BrainWave.DELTA,
        BrainWave.THETA,
        BrainWave.ALPHA,
        BrainWave.SIGMA,
        BrainWave.BETA1,
        BrainWave.BETA2,
        BrainWave.BETA3,
        BrainWave.GAMMA,
        BrainWave.LAMBDA,
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
    # Slow channels — real event flux, mapped to alert-range bands.
    if sig.pstate_rate > 0.0:
        out.append((BrainWave.BETA3, min(0.2, sig.pstate_rate * _PSTATE_SCALE)))
    if sig.irq_rate > 0.0:
        out.append((BrainWave.BETA2, min(0.15, sig.irq_rate * _IRQ_SCALE)))
    return out


def _apply_evoked_resets(sig: FieldProbeSignature) -> None:
    """Turn physical transients into oscillator phase resets.

    Real evoked potentials are phase resets of ongoing rhythm, not
    amplitude bumps. Two triggers, both physical:

    - A per-band jitter-energy step ≥ ``_EVOKED_STEP``× between
      successive signatures → π/2 phase nudge on that band.
    - A global scheduling shudder (``jitter_cv`` over threshold)
      → full phase reset of the broadband alert band.
    """
    global _prev_signature
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
        if sig.jitter_cv >= _SHUDDER_JITTER_CV:
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
    from thalamus.brain_waves import add_stimulus

    applied = False
    try:
        sig = client.get_field_probe()
    except Exception:  # noqa: BLE001 — absent channel, any cause
        sig = None
    if sig is not None and sig.samples_ok:
        for band, strength in _signature_stimulus(sig):
            add_stimulus(band, strength)
        _apply_evoked_resets(sig)
        _apply_phase_lock(sig, client)
        applied = True

    try:
        from brainstem.field_audio import apply_audio_stimulus

        if apply_audio_stimulus():
            applied = True
    except Exception as e:  # noqa: BLE001 — absent channel, any cause
        logger.debug("field audio stimulus failed: %s", e)
    return applied
