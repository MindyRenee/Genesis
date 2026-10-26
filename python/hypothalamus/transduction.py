"""Wave→chemical transduction — the hypothalamic release channel.

The causal chain this file closes:

    EM field (field_probe) → brain waves → chemicals → body/behavior

The probe's measured oscillations stimulate the thalamic oscillator
(``brainstem.field_probe.apply_field_stimulus``); this module performs
the next step — the oscillator's band powers drive the peptides that
declared affinity for those bands. It is the wave half of the loop
``_heartbeat_advance_neuro`` documents ("brain waves drive
neurochemistry, neurochemistry shapes brain waves") — neurochemistry
already shapes the wave radii; this is the return edge.

## The control model

Each peptide is a first-order integrator owned by the body:

    dC/dt = -λ·(C - C0) + Σᵢ Iᵢ·δ(t - tᵢ)

with impulsive inputs Iᵢ (``neuro_impulse`` kicks) and body-owned
decay λ. The wave state supplies the *reference*: a peptide's
commanded level is the summed power of its affiliated bands — power
is additive across sub-bands (energies add, per Parseval), so
"beta" means BETA1+BETA2+BETA3.

The control law is saturated exact tracking:

    I = clip(r - C, ±max_impulse·RATE)

a contraction toward r for *any* λ — no estimate of the body's decay
constants is needed for stability, only for speed, and rate-limiting
by the peptide's own ``max_impulse`` caps authority per beat.

Every UP member implies its DOWN: when the band is quiet the same
law emits negative impulses, withdrawing the peptide's hold (orexin
must fall in sleep, not merely stop rising). Peptides can also carry
an explicit ``wave_sign = -1`` — the declared DOWN member of a band
pair. For those the band enforces *suppression* rather than a level:

    I = -min(max_impulse·RATE·p, C)

exponential decay at a rate the band power sets, floor-aware so it
can never drive below zero. Cortisol↔delta is the canonical pair:
slow-wave dominance is the cortisol nadir.

## Wiring spec

Nothing here is hardcoded per-peptide: the peptide model classes
(``hypothalamus/orexin.py`` etc.) declare ``index``, ``name``,
``baseline``, ``max_impulse``, ``brain_waves``, and ``wave_sign``.
This module reads the declarations; adding a peptide or re-tagging a
band changes the transduction without touching it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .cortisol import Cortisol
from .crh import CRH
from .histamine import Histamine
from .melatonin import Melatonin
from .orexin import Orexin
from .oxytocin import Oxytocin
from .vasopressin import Vasopressin

if TYPE_CHECKING:
    from thalamus.brain_waves import BrainWaveState

__all__ = ["DRIVE_RATE", "transduce_waves", "wave_impulses"]


#: Fraction of each peptide's declared ``max_impulse`` used per beat.
#: Keeps per-call kicks inside the emotional regulator's ±0.01–0.02
#: envelope while leaving exact tracking intact over a few beats.
DRIVE_RATE: float = 0.2

#: Impulses smaller than this are noise below the body's resolution —
#: skipped so the channel carries signal, not dither.
_EPS: float = 1e-4


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


#: The hypothalamic peptide set — instantiated once; the models are
#: frozen data tables so the instances are pure wiring spec.
_PEPTIDES = (Cortisol(), CRH(), Histamine(), Melatonin(), Orexin(), Oxytocin(), Vasopressin())


def _drive(peptide, state: BrainWaveState, groups: dict[str, tuple]) -> float:
    """Summed power of the peptide's affiliated bands, clamped [0, 1]."""
    total = 0.0
    for name in peptide.brain_waves:
        total += sum(state.powers.get(b, 0.0) for b in groups.get(name, ()))
    return max(0.0, min(1.0, total))


def wave_impulses(
    state: BrainWaveState,
    chemicals: dict[str, float],
) -> list[tuple[int, float]]:
    """The peptide impulses the wave state commands this beat.

    ``chemicals`` is the body's current level map (``summary.chemicals``
    — name → effective level); a peptide absent from it is treated as
    at its declared baseline. Returns ``(chem_id, signed_magnitude)``
    pairs, the same contract ``neuro_impulse`` accepts — positive
    magnitudes release, negative withdraw or suppress.

    UP peptides (``wave_sign = +1``) track their band power as the
    commanded level — saturated exact tracking. DOWN peptides
    (``wave_sign = -1``) are suppressed at a rate proportional to
    their band's power — the pair's opposing member.
    """
    groups = _band_groups()
    out: list[tuple[int, float]] = []
    for pep in _PEPTIDES:
        drive = _drive(pep, state, groups)
        cap = pep.max_impulse * DRIVE_RATE
        current = chemicals.get(pep.name, pep.baseline)
        if pep.wave_sign >= 0:
            # Saturated exact tracking: I = clip(r - C, ±cap), r = drive.
            impulse = max(-cap, min(cap, drive - current))
        else:
            # DOWN member: exponential suppression enforced by the
            # band — never positive, never deeper than the floor.
            impulse = -min(cap * drive, max(0.0, current))
        if abs(impulse) < _EPS:
            continue
        out.append((pep.index, impulse))
    return out


def transduce_waves(client, state: BrainWaveState, chemicals: dict[str, float]) -> int:
    """Emit the wave-commanded impulses to the body.

    One ``neuro_impulse`` per peptide whose commanded correction
    clears the noise floor. Returns the count emitted — an absent or
    declined channel counts nothing, matching the impulse semantics
    (False means the body did not take it).
    """
    if state is None:
        return 0
    emitted = 0
    for chem, magnitude in wave_impulses(state, chemicals):
        try:
            if client.neuro_impulse(chem, magnitude):
                emitted += 1
        except (OSError, ConnectionError, RuntimeError):
            break  # the body is gone — the wave's command ends with it
    return emitted
