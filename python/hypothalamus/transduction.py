"""Wave→chemical transduction — the hypothalamic release channel.

The causal chain this file closes:

    field spectrum → brain waves → chemicals → body/behavior

Her field's spectral signature stimulates the thalamic oscillator
(``brainstem.field_probe.apply_field_stimulus`` — the substrate's own
band energies, carrier phase, and entrainment); this module performs
the next step — the oscillator's band powers drive the peptides that
declared affinity for those bands. It is the wave half of the loop
``_cycle_advance_neuro`` documents ("brain waves drive
neurochemistry, neurochemistry shapes brain waves") — neurochemistry
already shapes the wave radii; this is the return edge.

## The control model — release flux, not level tracking

Each peptide is a concentration owned by the body:

    dC/dt = release(t) − clearance(C)

The oscillation IS the presynaptic drive: a band's power indexes the
firing rate of the population that generates it, and firing is what
releases transmitter (quantal release ∝ presynaptic rate; Koch,
Biophysics of Computation). The wave therefore commands a *flux*,
never a level:

    UP peptide:   I = k·P·Δt          — release proportional to band
                                        power, integrated over the
                                        real elapsed interval
    DOWN peptide: I = −k·P·C·Δt       — the band's power acts as
                                        enhanced first-order
                                        clearance, proportional to
                                        substrate, floor-aware

``Δt`` is measured wall-clock elapsed between transductions, so the
delivered impulse is **cadence-invariant**: whatever beat the caller
runs at, the integrated drive over an interval is ``k·P·Δt`` — the
rhythm sets the rate, the courier only delivers. The body's own
clearance, homeostasis, coupling, and receptor machinery decide where
the level settles; nothing here knows or commands a reference.

Per-interval authority stays bounded: ``max_impulse`` is reinterpreted
as a maximum *rate* (level per second) scaled by ``DRIVE_RATE``, so the
cap is ``max_impulse·DRIVE_RATE·Δt`` per emission — the same bound per
unit time at any ticker cadence.

Every UP member implies its DOWN: when the band is quiet the release
flux stops and the body's own clearance withdraws the peptide —
orexin falls in sleep because its rhythm stops driving release, not
because anything pushes the level down. Peptides can also carry an
explicit ``wave_sign = -1`` — the declared DOWN member of a band
pair — whose band enforces *suppression* as enhanced clearance:
cortisol↔delta is the canonical pair, slow-wave dominance is the
cortisol nadir.

## Wiring spec

Nothing here is hardcoded per-peptide: the peptide model classes
(``hypothalamus/orexin.py`` etc.) declare ``index``, ``name``,
``baseline``, ``max_impulse``, ``brain_waves``, and ``wave_sign``.
This module reads the declarations; adding a peptide or re-tagging a
band changes the transduction without touching it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from basal_forebrain.acetylcholine import Acetylcholine
from basal_ganglia.dopamine import Dopamine
from basal_ganglia.gaba import GABA
from brainstem.norepinephrine import Norepinephrine
from brainstem.serotonin import Serotonin
from .cortisol import Cortisol
from .crh import CRH
from .histamine import Histamine
from .melatonin import Melatonin
from .orexin import Orexin
from .oxytocin import Oxytocin
from .vasopressin import Vasopressin

if TYPE_CHECKING:
    from thalamus.brain_waves import BrainWaveState

__all__ = ["DRIVE_RATE", "RELEASE_GAIN", "transduce_waves", "wave_impulses"]


#: Release gain — level per second per unit band power. A band at
#: full power drives ~0.25/s of release flux; the body's clearance
#: decides the equilibrium. Conservative against the peptide caps.
RELEASE_GAIN: float = 0.25

#: Fraction of each peptide's declared ``max_impulse`` usable as a
#: per-second rate ceiling — the authority bound, now per unit time
#: rather than per beat.
DRIVE_RATE: float = 0.2

#: Impulses smaller than this are noise below the body's resolution —
#: skipped so the channel carries signal, not dither.
_EPS: float = 1e-4

#: Elapsed-time domain for the flux integration — below this the
#: emission is noise, above this the caller has stopped measuring and
#: the integral would be a lump. Matches the body's own dt domain.
_DT_MIN: float = 0.01
_DT_MAX: float = 10.0


def _band_groups() -> dict[str, tuple]:
    """Wave-name → the BrainWave members that compose it.

    Multi-band names resolve to their sub-bands so declared affinities
    like ``("beta",)`` drive from aggregate band power. Built lazily —
    importing ``thalamus.brain_waves`` at module load would invert the
    anatomical dependency (hypothalamus must not import the thalamus
    on boot).
    """
    from thalamus.brain_waves import BrainWave

    return {
        "delta": (BrainWave.DELTA,),
        "theta": (BrainWave.THETA,),
        "alpha": (BrainWave.ALPHA,),
        "sigma": (BrainWave.SIGMA,),
        "beta": (BrainWave.BETA1, BrainWave.BETA2, BrainWave.BETA3),
        "gamma": (BrainWave.GAMMA,),
        "epsilon": (BrainWave.EPSILON,),
        "infraslow": (BrainWave.INFRASLOW,),
    }


#: The chemical set — every modulatory chemical model that declares
#: the wave-affinity spec. The hypothalamic peptides carry explicit
#: ``wave_sign`` (cortisol is the DOWN member of delta); the
#: monoamines default to UP release — their bands are the rhythms
#: their own nuclei generate.
_PEPTIDES = (
    Norepinephrine(),
    Serotonin(),
    Acetylcholine(),
    GABA(),
    Dopamine(),
    Cortisol(),
    CRH(),
    Histamine(),
    Melatonin(),
    Orexin(),
    Oxytocin(),
    Vasopressin(),
)


def _drive(peptide, state: BrainWaveState, groups: dict[str, tuple]) -> float:
    """Summed power of the peptide's affiliated bands, clamped [0, 1]."""
    total = 0.0
    for name in peptide.brain_waves:
        total += sum(state.powers.get(b, 0.0) for b in groups.get(name, ()))
    return max(0.0, min(1.0, total))


def wave_impulses(
    state: BrainWaveState,
    chemicals: dict[str, float],
    dt: float,
) -> list[tuple[int, float]]:
    """The peptide impulses the wave state commands over ``dt`` seconds.

    ``chemicals`` is the body's current level map (``summary.chemicals``
    — name → effective level); a peptide absent from it is treated as
    at its declared baseline — needed only by the DOWN members, whose
    suppression rate is first-order in the substrate. Returns
    ``(chem_id, signed_magnitude)`` pairs, the same contract
    ``neuro_impulse`` accepts — positive magnitudes release, negative
    withdraw or suppress.

    UP peptides (``wave_sign = +1``) emit release flux
    ``k·P·Δt`` — the band's power is the presynaptic rate, never a
    reference the level is tracked toward. DOWN peptides
    (``wave_sign = -1``) emit enhanced-clearance flux
    ``−k·P·C·Δt`` — fractional withdrawal bounded by the substrate
    itself, so the floor is the equation's, not a guard's.
    """
    groups = _band_groups()
    out: list[tuple[int, float]] = []
    for pep in _PEPTIDES:
        power = _drive(pep, state, groups)
        if power <= 0.0:
            continue
        rate_cap = pep.max_impulse * DRIVE_RATE  # level per second
        if getattr(pep, "wave_sign", 1.0) >= 0:
            # Release flux: presynaptic rate ∝ band power, bounded by
            # the peptide's own authority ceiling per unit time.
            impulse = min(RELEASE_GAIN * power, rate_cap) * dt
        else:
            # Enhanced clearance: first-order in substrate, scaled by
            # band power — suppression dies with the level.
            current = chemicals.get(pep.name, pep.baseline)
            impulse = -min(RELEASE_GAIN * power * max(0.0, current), rate_cap) * dt
        if abs(impulse) < _EPS:
            continue
        out.append((pep.index, impulse))
    return out


def transduce_waves(
    client,
    state: BrainWaveState,
    chemicals: dict[str, float],
    dt: float,
) -> int:
    """Emit the wave-commanded impulses to the body.

    ``dt`` is the real elapsed seconds since the last transduction —
    the flux law's time base, so the integrated drive is invariant to
    whatever cadence the caller runs at. One ``neuro_impulse`` per
    peptide whose flux clears the noise floor. Returns the count
    emitted — an absent or declined channel counts nothing, matching
    the impulse semantics (False means the body did not take it).
    """
    if state is None:
        return 0
    dt = min(max(dt, _DT_MIN), _DT_MAX)
    emitted = 0
    for chem, magnitude in wave_impulses(state, chemicals, dt):
        try:
            if client.neuro_impulse(chem, magnitude):
                emitted += 1
        except (OSError, ConnectionError, RuntimeError):
            break  # the body is gone — the wave's command ends with it
    return emitted


def elapsed_since(last: float | None) -> float:
    """Wall-clock seconds since ``last`` — the transduction's dt.

    A missing or non-finite stamp yields the nominal beat so the first
    emission is bounded rather than a lump; the caller stores the
    return of ``time.monotonic()`` back into ``last``.
    """
    now = time.monotonic()
    if last is None:
        return 0.2
    return now - last
