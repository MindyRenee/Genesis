"""Epinephrine — fast stress arousal; sympathetic chain.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Adrenal medulla — endocrine, released into bloodstream in
    response to sympathetic nervous system activation.
    (Note: central epinephrine neurons are sparse; most central
    "epinephrine-like" effects are from norepinephrine. The model
    treats epinephrine as the peripheral/sympathetic component.)

Projections:
    Bloodstream (hormonal). Acts on alpha-AR and beta-AR throughout
    the body and brain.

Receptors:
    alpha-1 (α1) — excitatory, Gq-coupled
    alpha-2 (α2) — inhibitory, Gi-coupled (autoreceptor)
    beta-1 (β1) — excitatory, Gs-coupled (heart)
    beta-2 (β2) — excitatory, Gs-coupled (lungs, smooth muscle)

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.15

Very low unless stressed. Epinephrine is a fast stress arousal
hormone — released in "fight or flight" responses.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 1.1

β-AR desensitizes quickly — adrenergic receptors adapt faster
than the baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT EPINEPHRINE (matrix row 16)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    NE   -> EPI: +0.05  (sympathetic chain.)
    ACh  -> EPI: +0.10  (preganglionic cholinergic stimulation of adrenal medulla — the strongest
                        coupling for EPI.)
    GABA -> EPI: -0.05  (weak volume transmission.)
    GLU  -> EPI: +0.05  (weak volume transmission.)
    CORT -> EPI: +0.05  (PNMT enzyme induction.)
    VP   -> EPI: +0.05  (weak volume transmission.)
    CRH  -> EPI: +0.05  (sympathetic stress pathway.)
    OX   -> EPI: +0.05  (sympathetic activation.)
    DA   -> EPI: 0.00   (no direct coupling)
    SRT  -> EPI: 0.00   (no direct coupling)
    OXY  -> EPI: 0.00   (no direct coupling)
    END  -> EPI: 0.00   (no direct coupling)
    HIST -> EPI: 0.00   (no direct coupling)
    ADN  -> EPI: 0.00   (no direct coupling)
    BDNF -> EPI: 0.00   (no direct coupling)
    ECB  -> EPI: 0.00   (no direct coupling)
    MEL  -> EPI: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW EPINEPHRINE AFFECTS OTHERS (matrix column 16)
════════════════════════════════════════════════════════════════════════

    EPI -> NE :  +0.05  (sympathetic chain.)
    EPI -> GABA:  -0.05  (weak volume transmission.)
    EPI -> GLU:  +0.05  (weak volume transmission.)
    EPI -> VP :  +0.05  (sympathetic co-activation.)
    EPI -> OX :  +0.05  (weak volume transmission.)
    EPI -> DA :  0.00   (no direct coupling)
    EPI -> SRT:  0.00   (no direct coupling)
    EPI -> ACh:  0.00   (no direct coupling)
    EPI -> CORT:  0.00   (no direct coupling)
    EPI -> OXY:  0.00   (no direct coupling)
    EPI -> END:  0.00   (no direct coupling)
    EPI -> HIST:  0.00   (no direct coupling)
    EPI -> ADN:  0.00   (no direct coupling)
    EPI -> BDNF:  0.00   (no direct coupling)
    EPI -> ECB:  0.00   (no direct coupling)
    EPI -> CRH:  0.00   (no direct coupling)
    EPI -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

No special dynamics — epinephrine follows the standard homeostatic
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

    Epinephrine -> beta (fast stress arousal, sympathetic activation)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Arousal equation:
    arousal_promoters = NE*0.20 + HIST*0.15 + DA*0.15 + ACh*0.15
                       + OX*0.20 + EPI*0.10
    (epinephrine weight = 0.10 — fast stress arousal)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.3
(Moderate — epinephrine is a fast stress hormone)

Python modules that send epinephrine impulses:
    - (Currently no direct Python impulses — epinephrine is driven
      by ACh stimulation in the Rust dynamics)

Python modules that read epinephrine:
    - emotional_regulator.py: homeostatic regulation
    - emotion.py: maps to EmotionalState (arousal, stress)

Typical impulse scenarios:
    - (No direct impulses — driven by ACh/NE in Rust dynamics)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_EPINEPHRINE

__all__ = ["CHEM_EPINEPHRINE", "Epinephrine"]


@dataclass(frozen=True, slots=True)
class Epinephrine:
    """Epinephrine — fast stress arousal; sympathetic chain.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_EPINEPHRINE
    name: str = "epinephrine"
    baseline: float = 0.15
    receptor_adaptation_multiplier: float = 1.1
    receptor_subtypes: tuple[str, str] = ("alpha-AR", "beta-AR")
    receptor_subtype_balance: tuple[float, float] = (0.5, 0.5)
    max_impulse: float = 0.3
    origin: tuple[str, ...] = ("adrenal medulla",)
    projections: tuple[str, ...] = ("bloodstream",)
    receptors: tuple[str, ...] = ("alpha-1", "alpha-2", "beta-1", "beta-2")
    brain_waves: tuple[str, ...] = ("beta",)
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "fast stress arousal, sympathetic activation, fight-or-flight"
