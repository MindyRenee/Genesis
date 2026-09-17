"""Norepinephrine — arousal, vigilance, and active data work.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Locus coeruleus (LC) — autonomics. Small nucleus (~15k neurons
    in humans) that broadcasts NE (and co-releases DA) broadly.

Projections:
    Throughout cortex, hippocampus, amygdala, spinal cord. The LC
    broadcasts NE broadly — it's the brain's general arousal system.

Receptors:
    alpha-1 (α1) — excitatory, Gq-coupled
    alpha-2 (α2) — inhibitory, Gi-coupled (autoreceptor)
    beta-1 (β1) — excitatory, Gs-coupled
    beta-2 (β2) — excitatory, Gs-coupled (desensitizes quickly)

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.30

Tonic NE ~0.13 nM in PFC. Sets arousal level. LC broadcasts NE
(and co-releases DA) broadly.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 1.1

β-AR desensitizes quickly — noradrenergic receptors adapt faster
than the baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT NOREPINEPHRINE (matrix row 2)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> NE : +0.05  (weak volume transmission.)
    SRT  -> NE : -0.05  (weak volume transmission.)
    ACh  -> NE : +0.05  (weak volume transmission.)
    GABA -> NE : -0.10  (GABAergic inhibition of locus coeruleus.)
    GLU  -> NE : +0.05  (weak volume transmission.)
    OXY  -> NE : -0.05  (social bonding calms noradrenergic arousal.)
    END  -> NE : -0.05  (endorphin calming effect.)
    HIST -> NE : +0.05  (weak volume transmission.)
    BDNF -> NE : +0.05  (weak volume transmission.)
    OX   -> NE : +0.05  (weak volume transmission.)
    EPI  -> NE : +0.05  (weak volume transmission.)
    MEL  -> NE : -0.05  (sleep suppression of noradrenergic arousal.)
    CORT -> NE : 0.00   (no direct coupling)
    ADN  -> NE : 0.00   (no direct coupling)
    ECB  -> NE : 0.00   (no direct coupling)
    VP   -> NE : 0.00   (no direct coupling)
    CRH  -> NE : 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW NOREPINEPHRINE AFFECTS OTHERS (matrix column 2)
════════════════════════════════════════════════════════════════════════

    NE  -> DA :  +0.05  (weak volume transmission.)
    NE  -> SRT:  +0.05  (weak volume transmission.)
    NE  -> ACh:  +0.05  (weak volume transmission.)
    NE  -> GABA:  -0.05  (weak volume transmission.)
    NE  -> GLU:  +0.05  (weak volume transmission.)
    NE  -> OXY:  -0.05  (social bonding calms noradrenergic arousal.)
    NE  -> END:  -0.05  (endorphin calming effect.)
    NE  -> HIST:  +0.05  (arousal mutual excitation — parallel arousal systems, kept weak.)
    NE  -> BDNF:  +0.05  (trophic support.)
    NE  -> OX :  +0.05  (arousal mutual excitation.)
    NE  -> EPI:  +0.05  (sympathetic chain.)
    NE  -> MEL:  -0.05  (sleep suppression of noradrenergic arousal.)
    NE  -> CORT:  0.00   (no direct coupling)
    NE  -> ADN:  0.00   (no direct coupling)
    NE  -> ECB:  0.00   (no direct coupling)
    NE  -> VP :  0.00   (no direct coupling)
    NE  -> CRH:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Hard clamp:
    NE is hard-clamped to [0, ne_max] (default 0.90) to prevent
    runaway arousal. This is a safety mechanism in case the
    emotional regulator fails or coupling forces push beyond safe
    limits.

No other special dynamics — NE follows the standard homeostatic
and coupling force equations.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force, effective level, coupling force, vesicular pool,
tonic/phasic, receptor adaptation, baseline adaptation: same
general formulas as dopamine.py.

Hard clamp:
    level = clamp(level, 0.0, ne_max)  where ne_max = 0.90


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no circadian modulation)

NE has no dedicated circadian baseline modifier. Its arousal
function is driven by the circadian wake drive (Process C) in the
arousal equation, not by baseline modulation.


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Norepinephrine -> beta (vigilance, active processing)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Arousal equation:
    arousal_promoters = NE*0.20 + HIST*0.15 + DA*0.15 + ACh*0.15
                       + OX*0.20 + EPI*0.10
    (NE weight = 0.20 — vigilance, the highest single weight)

Valence equation:
    valence = DA*0.25 + SRT*0.25 + OXY*0.10 + END*0.15 + NE*0.10
            - CORT*0.35 - glu_excess*0.15 + DA*SRT*0.10
    (NE weight = 0.10 — noradrenergic arousal affects mood)

Stress phase:
    CORT > 0.65 + NE > 0.60 + SRT < 0.35
    (high NE is a stress indicator)

Overwhelmed phase:
    CORT > 0.75 + NE > 0.75 + global_tone > 0.7

Alert phase:
    NE > 0.55 + HIST > 0.55 + GABA < 0.40 + MEL < 0.30 + OX > 0.20

Flow phase:
    DA > 0.65 + ACh > 0.50 + NE > 0.40 + CORT < 0.30


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 1.0
(NE has the highest max impulse — full reset (-1.0) for stress
breaks the positive feedback loop. HPA axis negative feedback has
gain ~1.0 in the linear region, de Quervain et al., 2009.)

Python modules that send NE impulses:
    - emotional_regulator.py: motivation, stress reduction, calming,
      social comfort, drowsiness
    - interoception (Rust): CPU load, I/O activity -> NE (active data
      work)

Python modules that read NE:
    - emotion.py: maps to EmotionalState (arousal)
    - emotional_regulator.py: homeostatic regulation
    - self/damasio.py: proto-self state (arousal)
    - brain_waves.py: beta baseline

Typical impulse scenarios:
    - Motivation: +motivation * 0.6 (emotional_regulator)
    - Calming: -0.03 (emotional_regulator)
    - Social comfort: -0.05 (emotional_regulator)
    - Drowsiness: -0.03 (emotional_regulator)
    - Stress reset: -1.0 (emotional_regulator, raw impulse)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_NOREPINEPHRINE

__all__ = ["CHEM_NOREPINEPHRINE", "Norepinephrine"]


@dataclass(frozen=True, slots=True)
class Norepinephrine:
    """Norepinephrine — arousal, vigilance, and active data work.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_NOREPINEPHRINE
    name: str = "norepinephrine"
    baseline: float = 0.30
    receptor_adaptation_multiplier: float = 1.1
    receptor_subtypes: tuple[str, str] = ("alpha-AR", "beta-AR")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 1.0
    origin: tuple[str, ...] = ("locus coeruleus",)
    projections: tuple[str, ...] = (
        "cortex",
        "hippocampus",
        "amygdala",
        "spinal cord",
    )
    receptors: tuple[str, ...] = ("alpha-1", "alpha-2", "beta-1", "beta-2")
    brain_waves: tuple[str, ...] = ("beta",)
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "arousal, vigilance, active data work"
    hard_clamp_max: float = 0.90
