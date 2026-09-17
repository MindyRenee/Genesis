"""Endorphin — euphoria, pleasure, and runner's high.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Hyporelay (arcuate nucleus)
    Pituitary gland

Projections:
    Throughout the brain via volume transmission. Endorphins are
    endogenous opioid peptides that act on the same receptors as
    morphine.

Receptors:
    Mu-opioid receptor (MOR) — Gi-coupled, analgesia, euphoria.
        Primary target of beta-endorphin.
    Delta-opioid receptor (DOR) — Gi-coupled, analgesia.
    Kappa-opioid receptor (KOR) — Gi-coupled, dysphoria, stress.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.25

~5-15 fmol/mL. Pleasure/pain modulation. Very low tonic levels;
released in bursts during exercise, social bonding, and pain.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.7

MOR downregulates slowly under sustained agonist exposure (chronic
morphine: "small reduction in total receptor number" — the 70-80%
loss is functional desensitization, not density, Chen et al., PNAS
1989).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT ENDORPHIN (matrix row 8)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> END: +0.05  (weak volume transmission.)
    SRT  -> END: +0.05  (weak volume transmission.)
    NE   -> END: -0.05  (weak volume transmission.)
    GABA -> END: +0.05  (relaxation promotes endorphin.)
    GLU  -> END: -0.05  (excitatory drive reduces endorphin.)
    CORT -> END: -0.10  (stress suppresses endogenous opioids.)
    OXY  -> END: +0.05  (opioid-oxytocin interaction (weak).)
    BDNF -> END: +0.05  (weak volume transmission.)
    ACh  -> END: 0.00   (no direct coupling)
    HIST -> END: 0.00   (no direct coupling)
    ADN  -> END: 0.00   (no direct coupling)
    ECB  -> END: 0.00   (no direct coupling)
    VP   -> END: 0.00   (no direct coupling)
    CRH  -> END: 0.00   (no direct coupling)
    OX   -> END: 0.00   (no direct coupling)
    EPI  -> END: 0.00   (no direct coupling)
    MEL  -> END: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW ENDORPHIN AFFECTS OTHERS (matrix column 8)
════════════════════════════════════════════════════════════════════════

    END -> DA :  +0.10  (endorphin disinhibition of DA — mu-opioid
    -> GABA interneuron -> DA disinhibition. Strongest positive
    coupling for DA, appropriate for runner's high.)
    END -> SRT:  +0.05  (weak volume transmission.)
    END -> NE :  -0.05  (endorphin calming effect.)
    END -> GABA:  +0.10  (endorphin promotes GABA — analgesic relaxation.)
    END -> GLU:  -0.05  (excitatory drive reduces endorphin.)
    END -> CORT:  -0.10  (endorphin suppresses cortisol — stress analgesia.)
    END -> OXY:  +0.05  (opioid-oxytocin interaction (weak).)
    END -> HIST:  -0.05  (weak volume transmission.)
    END -> BDNF:  +0.05  (trophic support.)
    END -> ACh:  0.00   (no direct coupling)
    END -> ADN:  0.00   (no direct coupling)
    END -> ECB:  0.00   (no direct coupling)
    END -> VP :  0.00   (no direct coupling)
    END -> CRH:  0.00   (no direct coupling)
    END -> OX :  0.00   (no direct coupling)
    END -> EPI:  0.00   (no direct coupling)
    END -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

No special dynamics — endorphin follows the standard homeostatic
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

    Endorphin -> alpha/theta (pleasure, relaxation)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Valence equation:
    valence = DA*0.25 + SRT*0.25 + OXY*0.10 + END*0.15 + NE*0.10
            - CORT*0.35 - glu_excess*0.15 + DA*SRT*0.10
    (endorphin weight = 0.15 — euphoria, pleasure)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.05
(Conservative — neuropeptides have narrow effective ranges)

Python modules that send endorphin impulses:
    - emotional_regulator.py: satisfaction, pleasure, comfort

Python modules that read endorphin:
    - emotion.py: maps to EmotionalState (valence, pleasure)
    - emotional_regulator.py: homeostatic regulation

Typical impulse scenarios:
    - Satisfaction: +satisfaction (emotional_regulator)
    - Comfort: +0.015 (emotional_regulator)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_ENDORPHIN

__all__ = ["CHEM_ENDORPHIN", "Endorphin"]


@dataclass(frozen=True, slots=True)
class Endorphin:
    """Endorphin — euphoria, pleasure, and runner's high.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_ENDORPHIN
    name: str = "endorphin"
    baseline: float = 0.25
    receptor_adaptation_multiplier: float = 0.9
    receptor_subtypes: tuple[str, str] = ("MOR", "DOR")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.05
    origin: tuple[str, ...] = ("hyporelay (arcuate)", "pituitary")
    projections: tuple[str, ...] = ("throughout brain (volume transmission)",)
    receptors: tuple[str, ...] = ("MOR", "DOR", "KOR")
    brain_waves: tuple[str, ...] = ("alpha", "theta")
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "euphoria, pleasure, pain modulation, runner's high"
