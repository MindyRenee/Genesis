"""Melatonin — circadian-driven sleep hormone; pineal secretion.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Pineal gland — secretes melatonin during the biological night
    under SCN (suprachiasmatic nucleus) control.

Projections:
    Bloodstream (hormonal). Acts on MT1 and MT2 receptors
    throughout the brain.

Receptors:
    MT1 — Gi-coupled, high density in SCN. Promotes sleep onset.
    MT2 — Gi-coupled, circadian phase shifting.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.10

Circadian-driven. Melatonin is low during the day and high during
the biological night. The baseline reflects the daytime resting
level; the circadian oscillator drives the level up at night.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.5

Melatonin receptors adapt slowly — hormonal receptors have slow
turnover.


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT MELATONIN (matrix row 17)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    Melatonin is primarily driven by the circadian oscillator (SCN).
    Wake-promoting signals suppress melatonin: NE, cortisol, orexin.

    NE   -> MEL: -0.05  (weak volume transmission.)
    CORT -> MEL: -0.05  (weak volume transmission.)
    OX   -> MEL: -0.05  (weak volume transmission.)
    DA   -> MEL: 0.00   (no direct coupling)
    SRT  -> MEL: 0.00   (no direct coupling)
    ACh  -> MEL: 0.00   (no direct coupling)
    GABA -> MEL: 0.00   (no direct coupling)
    GLU  -> MEL: 0.00   (no direct coupling)
    OXY  -> MEL: 0.00   (no direct coupling)
    END  -> MEL: 0.00   (no direct coupling)
    HIST -> MEL: 0.00   (no direct coupling)
    ADN  -> MEL: 0.00   (no direct coupling)
    BDNF -> MEL: 0.00   (no direct coupling)
    ECB  -> MEL: 0.00   (no direct coupling)
    VP   -> MEL: 0.00   (no direct coupling)
    CRH  -> MEL: 0.00   (no direct coupling)
    EPI  -> MEL: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW MELATONIN AFFECTS OTHERS (matrix column 17)
════════════════════════════════════════════════════════════════════════

    MEL -> DA :  -0.05  (sleep suppression of dopaminergic arousal.)
    MEL -> NE :  -0.05  (sleep suppression of noradrenergic arousal.)
    MEL -> ACh:  -0.05  (sleep suppression of cholinergic arousal.)
    MEL -> GABA:  +0.05  (sleep facilitation.)
    MEL -> GLU:  -0.05  (sleep suppression of cortical arousal.)
    MEL -> HIST:  -0.10  (sleep suppression of wake center.)
    MEL -> OX :  -0.05  (melatonin pushes the sleep-wake switch toward sleep.)
    MEL -> SRT:  0.00   (no direct coupling)
    MEL -> CORT:  0.00   (no direct coupling)
    MEL -> OXY:  0.00   (no direct coupling)
    MEL -> END:  0.00   (no direct coupling)
    MEL -> ADN:  0.00   (no direct coupling)
    MEL -> BDNF:  0.00   (no direct coupling)
    MEL -> ECB:  0.00   (no direct coupling)
    MEL -> VP :  0.00   (no direct coupling)
    MEL -> CRH:  0.00   (no direct coupling)
    MEL -> EPI:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Circadian oscillator drive:
    Melatonin is driven by the circadian oscillator directly, NOT
    by homeostatic regulation. The homeostatic force is SKIPPED for
    melatonin — it would fight the circadian approach, pulling
    melatonin toward its 0.10 baseline even at night when the
    oscillator is driving it toward 0.80.

    The melatonin_factor() gives a smooth [0,1] envelope that is
    high at night and low during the day, with a piecewise
    smoothstep model:
    - Day (0.30-0.70): melatonin is essentially 0.
    - Evening onset (0.70-0.85): melatonin rises from 0 to 1.
    - Biological night (0.85-0.15, wrap): melatonin stays at 1.
    - Morning offset (0.15-0.30): melatonin falls from 1 to 0.

    The level approaches a target set by the circadian factor:
    mel_target = mel_factor * 0.80  (peak ~0.80 at midnight)
    mel_approach_rate = 0.05 * dt_scale  (fast approach, ~2 seconds)
    mel_delta = (mel_target - level) * mel_approach_rate
    level += mel_delta

    Coupling forces from other chemicals (NE, cortisol, orexin
    inhibition) still apply — they can suppress melatonin even at
    night if arousal is high enough.

No baseline adaptation:
    Melatonin's baseline is not adapted by the general baseline
    adaptation loop — it is driven by the circadian oscillator.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force:
    homeostatic_force = 0.0  (skipped — circadian-driven)

Melatonin secretion (circadian approach):
    mel_factor = piecewise smoothstep based on circadian_phase
    mel_target = mel_factor * 0.80
    mel_approach_rate = 0.05 * dt_scale
    mel_delta = (mel_target - level) * mel_approach_rate
    level += mel_delta

Melatonin factor (piecewise smoothstep):
    smoothstep(t) = t^2 * (3 - 2t)  for t in [0, 1]

    if phase in [0.70, 0.85):  evening onset
        mel_factor = smoothstep((phase - 0.70) / 0.15)
    elif phase not in [0.15, 0.85):  biological night (wrap)
        mel_factor = 1.0
    elif phase in [0.15, 0.30):  morning offset
        mel_factor = 1.0 - smoothstep((phase - 0.15) / 0.15)
    else:  day
        mel_factor = 0.0

    Example values:
    - Phase 0.0 (midnight): 1.0
    - Phase 0.10 (02:24): 1.0
    - Phase 0.20 (04:48): ~0.74
    - Phase 0.25 (06:00): ~0.26
    - Phase 0.30 (07:12): 0.0
    - Phase 0.70 (16:48): 0.0
    - Phase 0.80 (19:12): ~0.26
    - Phase 0.90 (21:36): ~0.74

Effective level, coupling force, vesicular pool, tonic/phasic,
receptor adaptation: same general formulas as dopamine.py.


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no baseline modulation)

Melatonin is driven by the circadian oscillator directly (handled
by melatonin secretion, not baseline modulation). The baseline
stays at 0.10 throughout — the oscillator drives the LEVEL, not
the baseline.


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Melatonin -> delta (sleep, circadian night)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Arousal equation (sleep promoter):
    sleep_promoters = GABA*0.25 + ADN*0.65 + MEL*0.35
    (melatonin weight = 0.35 — hormonal sleep-promoter, MT1/MT2
    receptor activation, supplements the direct GABA/adenosine
    sleep drive, especially at the circadian night)

Sleep phase gating:
    Melatonin lowers the adenosine/histamine barrier for entering
    sleep:
    sleep_adn = 0.75 - 0.15 * mel - 0.05 * (1.0 - ox)
    sleep_hist = 0.25 + 0.05 * mel - 0.05 * ox

    High melatonin (circadian night) makes it easier to fall asleep.

Alert phase:
    MEL < 0.30 (enter) / 0.40 + h (stay) required for alertness


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.1
(Conservative — melatonin is a hormone with slow clearance)

Python modules that send melatonin impulses:
    - (Currently no direct Python impulses — melatonin is driven
      by the circadian oscillator in the Rust dynamics)

Python modules that read melatonin:
    - emotional_regulator.py: homeostatic regulation
    - emotion.py: maps to EmotionalState (drowsiness, sleep)
    - sleep/ (Python): sleep timing, circadian phase

Typical impulse scenarios:
    - (No direct impulses — driven by circadian oscillator in Rust)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_MELATONIN

__all__ = ["CHEM_MELATONIN", "Melatonin"]


@dataclass(frozen=True, slots=True)
class Melatonin:
    """Melatonin — circadian-driven sleep hormone; pineal secretion.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_MELATONIN
    name: str = "melatonin"
    baseline: float = 0.10
    receptor_adaptation_multiplier: float = 0.4
    receptor_subtypes: tuple[str, str] = ("MT1", "MT2")
    receptor_subtype_balance: tuple[float, float] = (0.5, 0.5)
    max_impulse: float = 0.1
    origin: tuple[str, ...] = ("pineal gland",)
    projections: tuple[str, ...] = ("bloodstream", "SCN")
    receptors: tuple[str, ...] = ("MT1", "MT2")
    brain_waves: tuple[str, ...] = ("delta",)
    circadian_modifier: str = "1.0 (driven by oscillator, not baseline modulation)"
    role: str = "circadian sleep hormone, MT1/MT2 activation, sleep onset"
    peak_level: float = 0.80
    approach_rate: float = 0.05
