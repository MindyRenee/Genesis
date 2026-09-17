"""Histamine — wakefulness promotion and arousal maintenance.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Tuberomammillary nucleus (TMN) — posterior hyporelay

Projections:
    Throughout the brain. Histamine is one of the key wake-
    promoting neuromodulators, alongside orexin and NE.

Receptors:
    H1 — Gq-coupled, excitatory. Target of antihistamines
        (drowsiness side effect).
    H2 — Gs-coupled, excitatory.
    H3 — Gi-coupled, autoreceptor (inhibitory).
    H4 — immune system, not central.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.30

Low nM. Wake promotion. Histamine is high during wakefulness and
low during sleep.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 1.0

Histamine receptors adapt at the baseline reference rate.


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT HISTAMINE (matrix row 9)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> HIST: +0.05  (weak volume transmission.)
    SRT  -> HIST: -0.05  (weak volume transmission.)
    NE   -> HIST: +0.05  (weak volume transmission.)
    ACh  -> HIST: +0.05  (weak volume transmission.)
    GABA -> HIST: -0.10  (inhibition of histaminergic tuberomammillary nucleus.)
    GLU  -> HIST: +0.05  (weak volume transmission.)
    END  -> HIST: -0.05  (weak volume transmission.)
    OX   -> HIST: +0.05  (weak volume transmission.)
    MEL  -> HIST: -0.10  (sleep suppression of wake center.)
    CORT -> HIST: 0.00   (no direct coupling)
    OXY  -> HIST: 0.00   (no direct coupling)
    ADN  -> HIST: 0.00   (no direct coupling)
    BDNF -> HIST: 0.00   (no direct coupling)
    ECB  -> HIST: 0.00   (no direct coupling)
    VP   -> HIST: 0.00   (no direct coupling)
    CRH  -> HIST: 0.00   (no direct coupling)
    EPI  -> HIST: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW HISTAMINE AFFECTS OTHERS (matrix column 9)
════════════════════════════════════════════════════════════════════════

    HIST -> DA :  +0.05  (weak volume transmission.)
    HIST -> NE :  +0.05  (arousal mutual excitation.)
    HIST -> ACh:  +0.05  (weak volume transmission.)
    HIST -> GABA:  -0.05  (weak volume transmission.)
    HIST -> GLU:  +0.05  (weak volume transmission.)
    HIST -> OX :  +0.05  (arousal mutual excitation.)
    HIST -> SRT:  0.00   (no direct coupling)
    HIST -> CORT:  0.00   (no direct coupling)
    HIST -> OXY:  0.00   (no direct coupling)
    HIST -> END:  0.00   (no direct coupling)
    HIST -> ADN:  0.00   (no direct coupling)
    HIST -> BDNF:  0.00   (no direct coupling)
    HIST -> ECB:  0.00   (no direct coupling)
    HIST -> VP :  0.00   (no direct coupling)
    HIST -> CRH:  0.00   (no direct coupling)
    HIST -> EPI:  0.00   (no direct coupling)
    HIST -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Narcolepsy modeling:
    When orexin is very low (<0.10), the sleep-wake flip-flop
    becomes unstable. Without orexin, histamine drops rapidly,
    causing sudden sleep transitions (cataplexy/sleep attack) even
    when adenosine is not high enough to trigger normal sleep.

    if orexin_level < 0.10:
        narcolepsy_force = (0.10 - orexin_level) * 0.05 * dt_scale
        histamine_level -= narcolepsy_force


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force, effective level, coupling force, vesicular pool,
tonic/phasic, receptor adaptation, baseline adaptation: same
general formulas as dopamine.py.

Narcolepsy force:
    if orexin_level < 0.10:
        narcolepsy_force = (0.10 - orexin_level) * 0.05 * dt_scale
        histamine_level -= narcolepsy_force


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 + 0.20 * sin(2π * (phase - 0.25))

Daytime elevation: promotes wakefulness, peaking around noon.


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Histamine -> beta (wakefulness, cortical arousal)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Arousal equation:
    arousal_promoters = NE*0.20 + HIST*0.15 + DA*0.15 + ACh*0.15
                       + OX*0.20 + EPI*0.10
    (histamine weight = 0.15 — wake maintenance)

Sleep phase:
    HIST < 0.25 (enter) / 0.38 (stay) required for sleep
    (The stay-asleep threshold (0.38) is above the histamine
    baseline (0.30) so that small perturbations from her own normal
    activity don't break the sleep state. A deliberate wake cascade
    sends histamine +0.20, pushing it to ~0.48 — well above 0.38.)

Alert phase:
    HIST > 0.55 (enter) / 0.45 (stay) + NE > 0.55 + GABA < 0.40
    + MEL < 0.30 + OX > 0.20


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.1
(Conservative — histamine is potent for wakefulness)

Python modules that send histamine impulses:
    - emotional_regulator.py: wakefulness, drowsiness (negative)

Python modules that read histamine:
    - emotion.py: maps to EmotionalState (arousal)
    - emotional_regulator.py: homeostatic regulation
    - brain_waves.py: beta baseline

Typical impulse scenarios:
    - Wakefulness: +0.04 (emotional_regulator)
    - Drowsiness: -0.02 (emotional_regulator)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_HISTAMINE

__all__ = ["CHEM_HISTAMINE", "Histamine"]


@dataclass(frozen=True, slots=True)
class Histamine:
    """Histamine — wakefulness promotion and arousal maintenance.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_HISTAMINE
    name: str = "histamine"
    baseline: float = 0.30
    receptor_adaptation_multiplier: float = 1.0
    receptor_subtypes: tuple[str, str] = ("H1", "H2")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.1
    origin: tuple[str, ...] = ("tuberomammillary nucleus",)
    projections: tuple[str, ...] = ("throughout brain",)
    receptors: tuple[str, ...] = ("H1", "H2", "H3")
    brain_waves: tuple[str, ...] = ("beta",)
    circadian_modifier: str = "1.0 + 0.20 * sin(2π * (phase - 0.25))"
    role: str = "wakefulness, arousal maintenance, narcolepsy modeling"
