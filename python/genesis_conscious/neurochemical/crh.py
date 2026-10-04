"""CRH — corticotropin-releasing hormone; HPA cascade initiator.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Paraventricular nucleus (PVN) of the hyporelay

Projections:
    Median eminence -> anterior pituitary (endocrine)
    Direct neural projections throughout the brain (central CRH
    system — anxiety, arousal)

Receptors:
    CRF1 — Gs-coupled, anterior pituitary and brain. Primary
        stress receptor.
    CRF2 — Gs-coupled, less well characterized. May mediate
        recovery from stress.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.0

CRH is a stress hormone — zero at rest, released on demand. Like
cortisol, it has no homeostatic set-point. It rises in response to
stressors and decays via clearance.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.5

CRF receptors adapt slowly — stress hormone receptors have slow
turnover.


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT CRH (matrix row 14)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    ALL positive couplings removed.
    CRH is produced only by the stress detection mechanism (emotional
    regulator) and the HPA cascade.
    Only suppressive feedback: GABA, cortisol (negative feedback), oxytocin, ECB.

    GABA -> CRH: -0.05  (weak volume transmission.)
    CORT -> CRH: -0.10  (weak volume transmission.)
    OXY  -> CRH: -0.05  (weak volume transmission.)
    ECB  -> CRH: -0.05  (weak volume transmission.)
    DA   -> CRH: 0.00   (no direct coupling)
    SRT  -> CRH: 0.00   (no direct coupling)
    NE   -> CRH: 0.00   (no direct coupling)
    ACh  -> CRH: 0.00   (no direct coupling)
    GLU  -> CRH: 0.00   (no direct coupling)
    END  -> CRH: 0.00   (no direct coupling)
    HIST -> CRH: 0.00   (no direct coupling)
    ADN  -> CRH: 0.00   (no direct coupling)
    BDNF -> CRH: 0.00   (no direct coupling)
    VP   -> CRH: 0.00   (no direct coupling)
    OX   -> CRH: 0.00   (no direct coupling)
    EPI  -> CRH: 0.00   (no direct coupling)
    MEL  -> CRH: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW CRH AFFECTS OTHERS (matrix column 14)
════════════════════════════════════════════════════════════════════════

    CRH -> EPI:  +0.05  (sympathetic stress pathway.)
    CRH -> DA :  0.00   (no direct coupling)
    CRH -> SRT:  0.00   (no direct coupling)
    CRH -> NE :  0.00   (no direct coupling)
    CRH -> ACh:  0.00   (no direct coupling)
    CRH -> GABA:  0.00   (no direct coupling)
    CRH -> GLU:  0.00   (no direct coupling)
    CRH -> CORT:  0.00   (no direct coupling)
    CRH -> OXY:  0.00   (no direct coupling)
    CRH -> END:  0.00   (no direct coupling)
    CRH -> HIST:  0.00   (no direct coupling)
    CRH -> ADN:  0.00   (no direct coupling)
    CRH -> BDNF:  0.00   (no direct coupling)
    CRH -> ECB:  0.00   (no direct coupling)
    CRH -> VP :  0.00   (no direct coupling)
    CRH -> OX :  0.00   (no direct coupling)
    CRH -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

No homeostatic set-point:
    CRH is a stress hormone, not a neurotransmitter. It has no
    homeostatic set-point — its resting level is zero. It rises
    in response to stressors and decays via clearance.

    The homeostatic force is zero for CRH. Instead, it gets a
    constant decay toward zero (leaky integrator):
    level *= exp(-0.005 * dt)  (half-life ~138 seconds)

HPA axis cascade initiator:
    CRH is the initiator of the HPA cascade:
    1. CRH (hyporelay) -> ACTH (anterior pituitary) — fast
    2. ACTH -> cortisol (adrenal cortex) — slower (minutes)
    3. Cortisol -> negative feedback to CRH and ACTH — slow

    Step 1: CRH drives ACTH production
    acth_target = crh_eff * 0.8 * maturation
    acth_alpha = 1.0 - exp(-0.2 * dt)
    acth_level += (acth_target - acth_level) * acth_alpha

    The cascade is gated by maturation_level (SHRP). At maturation
    0.0 (newborn), the cascade is fully dormant.

No baseline adaptation:
    CRH's baseline should never adapt — it has no homeostatic
    set-point, just a leaky integrator to zero. The general baseline
    adaptation loop skips CRH.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force:
    homeostatic_force = 0.0  (no homeostatic set-point)

Metabolic decay (leaky integrator):
    level *= exp(-0.005 * dt)  (half-life ~138 seconds)

HPA cascade (CRH -> ACTH):
    acth_target = crh_eff * 0.8 * maturation
    acth_alpha = 1.0 - exp(-0.2 * dt)
    acth_level += (acth_target - acth_level) * acth_alpha

Effective level, coupling force, vesicular pool, tonic/phasic,
receptor adaptation: same general formulas as dopamine.py.


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no circadian modulation)


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    CRH -> beta (anxiety, arousal, stress anticipation)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

No direct role in emergent phase computation. CRH's role is
upstream — it initiates the HPA cascade that produces cortisol,
which then drives stress/overwhelmed phases.


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.1
(Conservative — CRH is the HPA initiator, potent at low levels)

Python modules that send CRH impulses:
    - emotional_regulator.py: stress (via _raw_impulse)

Python modules that read CRH:
    - emotional_regulator.py: homeostatic regulation

Typical impulse scenarios:
    - Stress: +crh_output * 0.01 (emotional_regulator, raw impulse)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_CRH

__all__ = ["CHEM_CRH", "CRH"]


@dataclass(frozen=True, slots=True)
class CRH:
    """CRH — corticotropin-releasing hormone; HPA cascade initiator.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_CRH
    name: str = "crh"
    baseline: float = 0.0
    receptor_adaptation_multiplier: float = 0.5
    receptor_subtypes: tuple[str, str] = ("CRF1", "CRF2")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.1
    origin: tuple[str, ...] = ("PVN (hyporelay)",)
    projections: tuple[str, ...] = (
        "median eminence -> pituitary",
        "throughout brain (central CRH)",
    )
    receptors: tuple[str, ...] = ("CRF1", "CRF2")
    brain_waves: tuple[str, ...] = ("beta",)
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "HPA cascade initiator, stress response, anxiety"
    metabolic_decay_rate: float = 0.005
