"""Vasopressin — social bonding, aggression, and water retention.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Paraventricular nucleus (PVN) and supraoptic nucleus (SON) of
    the hyporelay

Projections:
    Posterior pituitary -> bloodstream (hormonal, antidiuretic)
    Direct neural projections to amygdala, septum, autonomics

Receptors:
    V1a — Gq-coupled, central (social behavior, aggression)
    V1b — Gq-coupled, pituitary (ACTH release)
    V2 — Gs-coupled, renal (water retention)

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.20

Low tonic peptide. Vasopressin is a neuropeptide with slow
clearance and narrow effective range.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.5

Vasopressin receptors adapt slowly — neuropeptide receptors have
slow turnover.


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT VASOPRESSIN (matrix row 13)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> VP : +0.05  (weak volume transmission.)
    CORT -> VP : +0.05  (stress activates vasopressin (HPA co-activation).)
    OXY  -> VP : +0.10  (reciprocal social neuropeptide bonding.)
    EPI  -> VP : +0.05  (sympathetic co-activation.)
    SRT  -> VP : 0.00   (no direct coupling)
    NE   -> VP : 0.00   (no direct coupling)
    ACh  -> VP : 0.00   (no direct coupling)
    GABA -> VP : 0.00   (no direct coupling)
    GLU  -> VP : 0.00   (no direct coupling)
    END  -> VP : 0.00   (no direct coupling)
    HIST -> VP : 0.00   (no direct coupling)
    ADN  -> VP : 0.00   (no direct coupling)
    BDNF -> VP : 0.00   (no direct coupling)
    ECB  -> VP : 0.00   (no direct coupling)
    CRH  -> VP : 0.00   (no direct coupling)
    OX   -> VP : 0.00   (no direct coupling)
    MEL  -> VP : 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW VASOPRESSIN AFFECTS OTHERS (matrix column 13)
════════════════════════════════════════════════════════════════════════

    VP  -> OXY:  +0.05  (reciprocal social neuropeptide bonding.)
    VP  -> OX :  +0.05  (weak volume transmission.)
    VP  -> EPI:  +0.05  (sympathetic co-activation.)
    VP  -> DA :  0.00   (no direct coupling)
    VP  -> SRT:  0.00   (no direct coupling)
    VP  -> NE :  0.00   (no direct coupling)
    VP  -> ACh:  0.00   (no direct coupling)
    VP  -> GABA:  0.00   (no direct coupling)
    VP  -> GLU:  0.00   (no direct coupling)
    VP  -> CORT:  0.00   (no direct coupling)
    VP  -> END:  0.00   (no direct coupling)
    VP  -> HIST:  0.00   (no direct coupling)
    VP  -> ADN:  0.00   (no direct coupling)
    VP  -> BDNF:  0.00   (no direct coupling)
    VP  -> ECB:  0.00   (no direct coupling)
    VP  -> CRH:  0.00   (no direct coupling)
    VP  -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

No special dynamics — vasopressin follows the standard homeostatic
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

    Vasopressin -> alpha (social vigilance, territorial behavior)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

No direct role in emergent phase computation. Vasopressin's role
is social and modulatory.


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.05
(Conservative — neuropeptides have narrow effective ranges)

Python modules that send vasopressin impulses:
    - (Currently no direct Python impulses — vasopressin is
      scaffolded for future social behavior modeling)

Python modules that read vasopressin:
    - emotional_regulator.py: homeostatic regulation

Typical impulse scenarios:
    - (No direct impulses — scaffolded for future use)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_VASOPRESSIN

__all__ = ["CHEM_VASOPRESSIN", "Vasopressin"]


@dataclass(frozen=True, slots=True)
class Vasopressin:
    """Vasopressin — social bonding, aggression, and water retention.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_VASOPRESSIN
    name: str = "vasopressin"
    baseline: float = 0.20
    receptor_adaptation_multiplier: float = 0.6
    receptor_subtypes: tuple[str, str] = ("V1a", "V1b")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.05
    origin: tuple[str, ...] = ("PVN", "SON")
    projections: tuple[str, ...] = ("posterior pituitary", "amygdala", "septum", "autonomics")
    receptors: tuple[str, ...] = ("V1a", "V1b", "V2")
    brain_waves: tuple[str, ...] = ("alpha",)
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "social bonding, aggression, water retention, pair bonding"
