"""BDNF — brain-derived neurotrophic factor; plasticity gate.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Throughout the brain — BDNF is expressed by neurons broadly,
    with high levels in hippocampus, cortex, and amygdala.

Projections:
    Local paracrine/autocrine signaling. BDNF is a neurotrophin,
    not a neurotransmitter — it acts on TrkB receptors to promote
    synaptic plasticity, neurogenesis, and neuronal survival.

Receptors:
    TrkB — receptor tyrosine kinase, primary BDNF receptor.
    p75NTR — pan-neurotrophin receptor, modulatory.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.45

Constitutive neurotrophin. BDNF is always present at a moderate
level to support baseline plasticity. Chronic stress (cortisol)
suppresses it below baseline, blocking learning.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.5

TrkB receptor turnover is slow — neurotrophin receptors adapt
much slower than the baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT BDNF (matrix row 11)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> BDNF: +0.05  (weak volume transmission.)
    SRT  -> BDNF: +0.10  (the antidepressant mechanism — serotonin is the strongest BDNF supporter.
                         SSRIs raise SRT -> raise BDNF -> neuroplasticity -> depression recovery.)
    NE   -> BDNF: +0.05  (weak volume transmission.)
    ACh  -> BDNF: +0.05  (weak volume transmission.)
    GLU  -> BDNF: +0.05  (weak volume transmission.)
    CORT -> BDNF: -0.20  (chronic stress suppresses BDNF — THE key mechanism by which stress
                         impairs learning and memory.)
    OXY  -> BDNF: +0.05  (weak volume transmission.)
    END  -> BDNF: +0.05  (weak volume transmission.)
    GABA -> BDNF: 0.00   (no direct coupling)
    HIST -> BDNF: 0.00   (no direct coupling)
    ADN  -> BDNF: 0.00   (no direct coupling)
    ECB  -> BDNF: 0.00   (no direct coupling)
    VP   -> BDNF: 0.00   (no direct coupling)
    CRH  -> BDNF: 0.00   (no direct coupling)
    OX   -> BDNF: 0.00   (no direct coupling)
    EPI  -> BDNF: 0.00   (no direct coupling)
    MEL  -> BDNF: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW BDNF AFFECTS OTHERS (matrix column 11)
════════════════════════════════════════════════════════════════════════

    BDNF -> DA :  +0.05  (trophic support.)
    BDNF -> SRT:  +0.10  (the antidepressant positive feedback — SSRIs raise SRT -> raise BDNF -> .
                         raise SRT. Bounded by homeostatic forces.)
    BDNF -> NE :  +0.05  (trophic support.)
    BDNF -> ACh:  +0.05  (trophic support.)
    BDNF -> GABA:  +0.05  (trophic support.)
    BDNF -> GLU:  +0.05  (trophic support.)
    BDNF -> CORT:  -0.05  (BDNF suppresses cortisol (weak).)
    BDNF -> OXY:  +0.05  (trophic support.)
    BDNF -> END:  +0.05  (trophic support.)
    BDNF -> HIST:  0.00   (no direct coupling)
    BDNF -> ADN:  0.00   (no direct coupling)
    BDNF -> ECB:  0.00   (no direct coupling)
    BDNF -> VP :  0.00   (no direct coupling)
    BDNF -> CRH:  0.00   (no direct coupling)
    BDNF -> OX :  0.00   (no direct coupling)
    BDNF -> EPI:  0.00   (no direct coupling)
    BDNF -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

BDNF recovery after stress:
    When cortisol has dropped below 0.3 (stress is over) and BDNF
    is still below its baseline (depleted by chronic stress), BDNF
    gets a recovery boost proportional to the serotonin level.
    Serotonin drives BDNF expression — this is why SSRIs help
    depression (increasing serotonin promotes BDNF recovery).

    if cort_level < 0.3 and bdnf_level < bdnf_baseline:
        recovery = srt_level * bdnf_recovery_rate * dt_scale
        bdnf_level += recovery

    This captures the neurobiological basis of stress recovery:
    after chronic stress depletes BDNF (via cortisol suppression),
    the recovery is driven by serotonin signaling, not just passive
    homeostatic drift.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force, effective level, coupling force, vesicular pool,
tonic/phasic, receptor adaptation, baseline adaptation: same
general formulas as dopamine.py.

BDNF recovery after stress:
    if cort_level < 0.3 and bdnf_level < bdnf_baseline:
        recovery = srt_level * bdnf_recovery_rate * dt_scale
        bdnf_level += recovery
    where bdnf_recovery_rate = 0.001 (default)


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no circadian modulation)


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    BDNF -> gamma (plasticity, learning)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Plasticity gate:
    plasticity_gate = BDNF * (1 - CORT * 0.5)
    (clamped to [0, 1])

    BDNF promotes plasticity, cortisol suppresses it. This is the
    structural basis of why chronic stress impairs learning:
    cortisol suppresses BDNF -> suppresses plasticity_gate ->
    prevents new memory formation.

Metaplasticity gating:
    The coupling matrix self-modifies only when plasticity_gate >
    0.1. Chronic stress (high cortisol) suppresses BDNF, which
    suppresses plasticity_gate, which suppresses metaplasticity —
    a protective mechanism that prevents stress from permanently
    rewiring the brain.


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.05
(Physiological boosts are 0.02-0.05; larger boosts trigger
receptor downregulation, Lu et al., 2014)

Python modules that send BDNF impulses:
    - emotional_regulator.py: learning, comfort, social bonding,
      stress reduction

Python modules that read BDNF:
    - emotion.py: maps to EmotionalState (plasticity)
    - emotional_regulator.py: homeostatic regulation
    - memory/engine.py: plasticity gate (encoding weight)
    - learning/ (Python): plasticity for all learning systems

Typical impulse scenarios:
    - Learning: +0.02 (emotional_regulator)
    - Comfort: +0.08 (emotional_regulator)
    - Social bonding: +0.15 * social_modulation (emotional_regulator)
    - Stress reduction: +0.10 (emotional_regulator, raw impulse)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_BDNF

__all__ = ["BDNF", "CHEM_BDNF"]


@dataclass(frozen=True, slots=True)
class BDNF:
    """BDNF — brain-derived neurotrophic factor; plasticity gate.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_BDNF
    name: str = "bdnf"
    baseline: float = 0.45
    receptor_adaptation_multiplier: float = 0.3
    receptor_subtypes: tuple[str, str] = ("TrkB", "p75NTR")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.05
    origin: tuple[str, ...] = ("throughout brain", "hippocampus", "cortex", "amygdala")
    projections: tuple[str, ...] = ("local paracrine/autocrine",)
    receptors: tuple[str, ...] = ("TrkB", "p75NTR")
    brain_waves: tuple[str, ...] = ("gamma",)
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "plasticity gate, neurogenesis, synaptic strengthening"
    recovery_rate: float = 0.001
