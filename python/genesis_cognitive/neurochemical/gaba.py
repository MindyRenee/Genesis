"""GABA — primary inhibitory neurotransmitter; calms glutamate.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Inhibitory interneurons (ubiquitous throughout the brain)
    VLPO (ventrolateral preoptic nucleus) — sleep-promoting GABA
    neurons

Projections:
    Throughout the brain. GABA is the brain's main inhibitory
    neurotransmitter, present in ~25-40% of all synapses.

Receptors:
    GABA-A — ionotropic, Cl- channel, fast inhibition. Target of
        benzodiazepines (allosteric modulation).
    GABA-B — metabotropic, Gi-coupled, slow inhibition.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.50

~200 nM extracellular. Main inhibitory transmitter. Always active.
The GLU/GABA balance is the primary E/I (excitation/inhibition) axis.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.9

GABA-A receptor turnover is moderate — slightly slower than the
baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT GABA (matrix row 4)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> GABA: -0.05  (weak volume transmission.)
    SRT  -> GABA: +0.05  (weak volume transmission.)
    NE   -> GABA: -0.05  (weak volume transmission.)
    ACh  -> GABA: -0.05  (weak volume transmission.)
    GLU  -> GABA: +0.10  (activity-driven feedback inhibition — when
    GLU rises, it drives GABAergic interneurons to fire, providing automatic gain control.
                         This is the core E/I balance mechanism.)
    CORT -> GABA: -0.10  (stress disinhibition (cortisol suppresses GABA).)
    OXY  -> GABA: +0.05  (weak volume transmission.)
    END  -> GABA: +0.10  (endorphin promotes GABA (analgesic relaxation).)
    HIST -> GABA: -0.05  (weak volume transmission.)
    ADN  -> GABA: +0.05  (sleep pressure promotes inhibition.)
    BDNF -> GABA: +0.05  (weak volume transmission.)
    ECB  -> GABA: -0.05  (depolarization-induced suppression of inhibition.)
    OX   -> GABA: -0.05  (weak volume transmission.)
    EPI  -> GABA: -0.05  (weak volume transmission.)
    MEL  -> GABA: +0.05  (sleep facilitation.)
    VP   -> GABA: 0.00   (no direct coupling)
    CRH  -> GABA: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW GABA AFFECTS OTHERS (matrix column 4)
════════════════════════════════════════════════════════════════════════

    GABA -> DA :  -0.10  (GABAergic inhibition of VTA DA neurons.)
    GABA -> SRT:  -0.05  (weak volume transmission.)
    GABA -> NE :  -0.10  (GABAergic inhibition of locus coeruleus.)
    GABA -> ACh:  -0.10  (inhibition of cholinergic basal forebrain.)
    GABA -> GLU:  -0.30  (THE primary inhibitory brake. Strongest coupling in the matrix.)
    GABA -> CORT:  -0.15  (GABA suppresses cortisol — strongest suppressive coupling for cortisol.)
    GABA -> OXY:  +0.05  (weak volume transmission.)
    GABA -> END:  +0.05  (relaxation promotes endorphin.)
    GABA -> HIST:  -0.10  (inhibition of histaminergic tuberomammillary nucleus.)
    GABA -> ADN:  +0.05  (sleep pressure promotes inhibition.)
    GABA -> ECB:  +0.05  (neural activity stimulates eCB synthesis.)
    GABA -> CRH:  -0.05  (GABA suppresses CRH.)
    GABA -> OX :  -0.10  (VLPO sleep-promoting neurons inhibit orexin.)
    GABA -> EPI:  -0.05  (weak volume transmission.)
    GABA -> BDNF:  0.00   (no direct coupling)
    GABA -> VP :  0.00   (no direct coupling)
    GABA -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Allosteric modulation (GABA-A):
    When gaba_a_allosteric > 0, GABA's effective level is multiplied
    by (1 + allosteric * 0.5). This models benzodiazepine-like
    enhancement of GABA-A sensitivity.

    effective = base * (1 + gaba_a_allosteric * 0.5)
    (clamped to [0, 2])

    This field is reserved for future IPC-driven pharmacological
    modulation. It is not currently driven by any daemon dynamics —
    the Python cognitive mind boosts GABA directly via impulses
    instead.

GABA disinhibition:
    When GABA is very high (>0.8) AND ACh is high (>0.6), GABA
    interneurons inhibit other GABA interneurons, disinhibiting
    glutamate. Paradoxically, very high GABA can promote excitatory
    transmission through disinhibition circuits.

    if gaba_eff > 0.8 and ach_eff > 0.6:
        disinhibition = (gaba_eff - 0.8) * (ach_eff - 0.6) * 0.5
        glutamate_level += disinhibition * dt


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force, coupling force, vesicular pool, tonic/phasic,
receptor adaptation, baseline adaptation: same general formulas as
dopamine.py.

Effective level (with allosteric modulation):
    base = subtype_effective_level()
    effective = base * (1 + gaba_a_allosteric * 0.5)
    (clamped to [0, 2])

GABA disinhibition:
    if gaba_eff > 0.8 and ach_eff > 0.6:
        disinhibition = (gaba_eff - 0.8) * (ach_eff - 0.6) * 0.5
        glutamate_level += disinhibition * dt


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no circadian modulation)

GABA has no dedicated circadian baseline modifier. Its sleep-
promoting function is driven by the arousal equation (sleep
promoter weight 0.25) and the VLPO inhibition of orexin.


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    GABA -> alpha/beta (inhibition, rhythm stabilization)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Arousal equation (sleep promoter):
    sleep_promoters = GABA*0.25 + ADN*0.65 + MEL*0.35
    (GABA weight = 0.25 — primary inhibitory. Toned down from 0.30
    because GABA's baseline (0.50) is high even during wakefulness;
    at 0.30 the sleep promoters exceeded the arousal promoters at
    baseline, making the Wilson-Cowan equilibrium dip too low at
    night.)

Alert phase:
    GABA < 0.40 + hysteresis (low GABA required for alertness)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.5
(Moderate anxiolytic doses are 0.3-0.5; higher doses cause
sedation, Nestler et al., 2008)

Python modules that send GABA impulses:
    - emotional_regulator.py: calming, comfort, stress reduction,
      sleep preparation, social bonding

Python modules that read GABA:
    - emotion.py: maps to EmotionalState (arousal reduction)
    - emotional_regulator.py: homeostatic regulation
    - brain_waves.py: alpha/beta baseline

Typical impulse scenarios:
    - Calming: +0.03 (emotional_regulator)
    - Comfort: +0.04 (emotional_regulator)
    - Social bonding: +0.3 * social_modulation (emotional_regulator)
    - Stress reset: +0.5 * social_modulation (emotional_regulator)
    - Sleep preparation: +0.04 (emotional_regulator)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_GABA

__all__ = ["CHEM_GABA", "GABA"]


@dataclass(frozen=True, slots=True)
class GABA:
    """GABA — primary inhibitory neurotransmitter.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_GABA
    name: str = "gaba"
    baseline: float = 0.50
    receptor_adaptation_multiplier: float = 0.9
    receptor_subtypes: tuple[str, str] = ("GABA-A", "GABA-B")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.5
    origin: tuple[str, ...] = ("inhibitory interneurons", "VLPO")
    projections: tuple[str, ...] = ("throughout brain",)
    receptors: tuple[str, ...] = ("GABA-A", "GABA-B")
    brain_waves: tuple[str, ...] = ("alpha", "beta")
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "primary inhibition, E/I balance, anxiety reduction, sleep"
