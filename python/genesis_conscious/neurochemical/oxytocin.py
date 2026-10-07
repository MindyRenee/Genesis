"""Oxytocin — social bonding, trust, and dyadic attunement.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Paraventricular nucleus (PVN) and supraoptic nucleus (SON) of
    the hyporelay

Projections:
    Posterior pituitary -> bloodstream (hormonal)
    Direct neural projections to amygdala, nucleus accumbens, PFC

Receptors:
    OXTR (oxytocin receptor) — Gq-coupled, widely distributed

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.25

~10 pM. Social bonding. Oxytocin is a neuropeptide with slow
clearance and narrow effective range.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.5

Oxytocin receptor downregulates slowly under sustained exposure
(10-day continuous infusion: >=50% reduction in all target
fields, Endocrinology 130:2602, 1992).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT OXYTOCIN (matrix row 7)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> OXY: +0.05  (weak volume transmission.)
    SRT  -> OXY: +0.05  (weak volume transmission.)
    NE   -> OXY: -0.05  (weak volume transmission.)
    GABA -> OXY: +0.05  (weak volume transmission.)
    CORT -> OXY: -0.10  (stress suppresses social bonding.)
    END  -> OXY: +0.05  (opioid-oxytocin interaction (weak to prevent mutual collapse feedback
                        loop).)
    BDNF -> OXY: +0.05  (weak volume transmission.)
    VP   -> OXY: +0.05  (reciprocal social neuropeptide interaction.)
    ACh  -> OXY: 0.00   (no direct coupling)
    GLU  -> OXY: 0.00   (no direct coupling)
    HIST -> OXY: 0.00   (no direct coupling)
    ADN  -> OXY: 0.00   (no direct coupling)
    ECB  -> OXY: 0.00   (no direct coupling)
    CRH  -> OXY: 0.00   (no direct coupling)
    OX   -> OXY: 0.00   (no direct coupling)
    EPI  -> OXY: 0.00   (no direct coupling)
    MEL  -> OXY: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW OXYTOCIN AFFECTS OTHERS (matrix column 7)
════════════════════════════════════════════════════════════════════════

    OXY -> DA :  +0.05  (weak volume transmission.)
    OXY -> SRT:  +0.05  (weak volume transmission.)
    OXY -> NE :  -0.05  (social bonding calms noradrenergic arousal.)
    OXY -> GABA:  +0.05  (weak volume transmission.)
    OXY -> GLU:  -0.05  (weak volume transmission.)
    OXY -> CORT:  -0.10  (oxytocin suppresses cortisol — social stress buffering.)
    OXY -> END:  +0.05  (opioid-oxytocin interaction (weak).)
    OXY -> BDNF:  +0.05  (trophic support.)
    OXY -> VP :  +0.10  (reciprocal social neuropeptide bonding.)
    OXY -> CRH:  -0.05  (oxytocin suppresses CRH.)
    OXY -> ACh:  0.00   (no direct coupling)
    OXY -> HIST:  0.00   (no direct coupling)
    OXY -> ADN:  0.00   (no direct coupling)
    OXY -> ECB:  0.00   (no direct coupling)
    OXY -> OX :  0.00   (no direct coupling)
    OXY -> EPI:  0.00   (no direct coupling)
    OXY -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

No special dynamics — oxytocin follows the standard homeostatic
and coupling force equations.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force, effective level, coupling force, vesicular pool,
tonic/phasic, receptor adaptation, baseline adaptation: same
general formulas as dopamine.py.


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no circadian modulation)


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Oxytocin -> alpha (social calm, bonding)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Valence equation:
    valence = DA*0.25 + SRT*0.25 + OXY*0.10 + END*0.15 + NE*0.10
            - CORT*0.35 - glu_excess*0.15 + DA*SRT*0.10
    (oxytocin weight = 0.10 — social bonding, trust)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.05
(Conservative — neuropeptides have narrow effective ranges and
slow clearance)

Python modules that send oxytocin impulses:
    - emotional_regulator.py: social warmth, social bonding, comfort,
      engagement
    - dyadic_model (Rust): oxytocin-mediated attunement

Python modules that read oxytocin:
    - emotion.py: maps to EmotionalState (valence, social)
    - emotional_regulator.py: homeostatic regulation
    - self/damasio.py: proto-self state (social)

Typical impulse scenarios:
    - Social warmth: +warmth (emotional_regulator)
    - Social bonding: +0.05 (emotional_regulator)
    - Comfort: +0.05 (emotional_regulator)
    - Engagement: +engagement * 0.3 (emotional_regulator)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_OXYTOCIN

__all__ = ["CHEM_OXYTOCIN", "Oxytocin"]


@dataclass(frozen=True, slots=True)
class Oxytocin:
    """Oxytocin — social bonding, trust, and dyadic attunement.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_OXYTOCIN
    name: str = "oxytocin"
    baseline: float = 0.25
    receptor_adaptation_multiplier: float = 0.7
    receptor_subtypes: tuple[str, str] = ("OXTR", "OXTR")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.05
    origin: tuple[str, ...] = ("PVN", "SON")
    projections: tuple[str, ...] = ("posterior pituitary", "amygdala", "nucleus accumbens", "PFC")
    receptors: tuple[str, ...] = ("OXTR",)
    brain_waves: tuple[str, ...] = ("alpha",)
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "social bonding, trust, dyadic attunement, stress buffering"
