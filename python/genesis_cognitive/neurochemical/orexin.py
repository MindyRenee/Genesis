"""Orexin — wake stabilization; bistable flip-flop stabilizer.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Lateral hyporelay (LHA), perifornical area, posterior
    hyporelay. A small, specific population of orexin neurons
    (~70k in humans).

Projections:
    Throughout the brain, with especially dense projections to
    monoaminergic neurons (locus coeruleus, raphe, TMN, VTA).

Receptors:
    OX1R (HCRTR1) — Gq-coupled, excitatory. Higher affinity for
        orexin-A. Both OX1R and OX2R are excitatory (Gq/11-coupled)
        and cause strong depolarization via non-selective cation
        channels and K+ channel inhibition (Sakurai & Mieda, 2011).
    OX2R (HCRTR2) — Gq-coupled, excitatory. Equal affinity for
        orexin-A and orexin-B. OX2R has the more pivotal wake-
        promoting role (OX2R knockout mice exhibit disrupted
        wakefulness; Mochizuki et al., J. Neurosci. 2011).

Both receptor subtypes are excitatory — the "OX1R excitatory,
OX2R wake-promoting" labels in the model are a simplification.
Both promote wakefulness; OX2R has the larger contribution.

Receptor subtypes (default_receptor_subtypes):
    [OX1R, OX2R] = [0.50, 0.50] — balanced at birth.


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.30

Picomolar. Wake stabilization. Orexin is the key stabilizer of the
sleep-wake flip-flop — it excites histamine and NE to maintain
wakefulness. Loss of orexin neurons causes narcolepsy.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.8

Orexin receptor trafficking is moderate — slightly slower than the
baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT OREXIN (matrix row 15)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    SRT  -> OX : -0.05  (serotonin promotes sleep onset.)
    NE   -> OX : +0.05  (weak volume transmission.)
    ACh  -> OX : +0.05  (weak volume transmission.)
    GABA -> OX : -0.10  (VLPO sleep-promoting neurons inhibit orexin.)
    GLU  -> OX : +0.05  (weak volume transmission.)
    HIST -> OX : +0.05  (weak volume transmission.)
    ADN  -> OX : -0.10  (sleep pressure inhibits orexin (the sleep switch).)
    ECB  -> OX : -0.05  (stress reduction.)
    VP   -> OX : +0.05  (weak volume transmission.)
    EPI  -> OX : +0.05  (weak volume transmission.)
    MEL  -> OX : -0.05  (melatonin pushes the sleep-wake switch toward sleep.)
    DA   -> OX : 0.00   (dopamine does not directly promote orexin
    — the DA->OX->DA positive feedback loop caused runaway
    overstimulation. Orexin is driven by metabolic state, not reward.)
    CORT -> OX : 0.00   (no direct coupling)
    OXY  -> OX : 0.00   (no direct coupling)
    END  -> OX : 0.00   (no direct coupling)
    BDNF -> OX : 0.00   (no direct coupling)
    CRH  -> OX : 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW OREXIN AFFECTS OTHERS (matrix column 15)
════════════════════════════════════════════════════════════════════════

    OX  -> DA :  +0.05  (weak volume transmission.)
    OX  -> NE :  +0.05  (arousal mutual excitation.)
    OX  -> ACh:  +0.05  (orexin-driven cortical activation.)
    OX  -> GABA:  -0.05  (weak volume transmission.)
    OX  -> GLU:  +0.05  (weak volume transmission.)
    OX  -> HIST:  +0.05  (orexin excites histamine — wake-stabilization circuit.)
    OX  -> ADN:  -0.05  (wake promotion — orexin mildly inhibits adenosine.)
    OX  -> EPI:  +0.05  (sympathetic activation.)
    OX  -> MEL:  -0.05  (melatonin pushes the sleep-wake switch toward sleep (reciprocal: OX
                        suppresses MEL).)
    OX  -> SRT:  0.00   (no direct coupling)
    OX  -> CORT:  0.00   (no direct coupling)
    OX  -> OXY:  0.00   (no direct coupling)
    OX  -> END:  0.00   (no direct coupling)
    OX  -> BDNF:  0.00   (no direct coupling)
    OX  -> ECB:  0.00   (no direct coupling)
    OX  -> VP :  0.00   (no direct coupling)
    OX  -> CRH:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Narcolepsy modeling:
    When orexin is very low (<0.10), the sleep-wake flip-flop
    becomes unstable — the system can suddenly transition to sleep
    without sufficient adenosine buildup. This models narcolepsy/
    cataplexy, where loss of orexin neurons causes sudden sleep
    attacks.

    Orexin normally stabilizes the flip-flop by exciting histamine
    and NE. Without orexin, histamine drops rapidly, causing sudden
    sleep transitions even when adenosine is not high enough to
    trigger normal sleep.

    if orexin_level < 0.10:
        narcolepsy_force = (0.10 - orexin_level) * 0.05 * dt_scale
        histamine_level -= narcolepsy_force


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force, effective level, coupling force, vesicular pool,
tonic/phasic, receptor adaptation, baseline adaptation: same
general formulas as dopamine.py.

Narcolepsy force (applied to histamine when orexin is low):
    if orexin_level < 0.10:
        narcolepsy_force = (0.10 - orexin_level) * 0.05 * dt_scale
        histamine_level -= narcolepsy_force


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no circadian modulation)

Orexin's circadian behavior is driven by the arousal equation and
sleep pressure, not by baseline modulation.


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Orexin -> beta (wakefulness, arousal stabilization)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Arousal equation:
    arousal_promoters = NE*0.20 + HIST*0.15 + DA*0.15 + ACh*0.15
                       + OX*0.20 + EPI*0.10
    (orexin weight = 0.20 — major arousal stabilizer, tied with NE
    for the highest weight)

Sleep phase gating:
    Orexin raises the adenosine/histamine barrier for entering sleep:
    sleep_adn = 0.75 - 0.15 * mel - 0.05 * (1.0 - ox)
    sleep_hist = 0.25 + 0.05 * mel - 0.05 * ox

    High orexin makes it harder to fall asleep; low orexin makes it
    easier. This makes sleep a multi-system decision, not adenosine
    alone.

Alert phase:
    OX > 0.20 (enter) / 0.15 (stay) required for alertness

Narcolepsy:
    OX < 0.10 -> sudden sleep transitions (cataplexy)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.1
(Conservative — orexin is potent for wake stabilization)

Python modules that send orexin impulses:
    - emotional_regulator.py: drowsiness (negative), calming

Python modules that read orexin:
    - emotion.py: maps to EmotionalState (arousal)
    - emotional_regulator.py: homeostatic regulation
    - sleep/ (Python): sleep timing, wake stabilization

Typical impulse scenarios:
    - Calming: -0.02 (emotional_regulator)
    - Drowsiness: -0.02 (emotional_regulator)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_OREXIN

__all__ = ["CHEM_OREXIN", "Orexin"]


@dataclass(frozen=True, slots=True)
class Orexin:
    """Orexin — wake stabilization; bistable flip-flop stabilizer.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_OREXIN
    name: str = "orexin"
    baseline: float = 0.30
    receptor_adaptation_multiplier: float = 0.9
    receptor_subtypes: tuple[str, str] = ("OX1R", "OX2R")
    receptor_subtype_balance: tuple[float, float] = (0.50, 0.50)
    max_impulse: float = 0.1
    origin: tuple[str, ...] = ("lateral hyporelay", "perifornical area")
    projections: tuple[str, ...] = ("locus coeruleus", "raphe", "TMN", "VTA", "throughout brain")
    receptors: tuple[str, ...] = ("OX1R", "OX2R")
    brain_waves: tuple[str, ...] = ("beta",)
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "wake stabilization, flip-flop stabilizer, narcolepsy modeling"
    narcolepsy_threshold: float = 0.10
