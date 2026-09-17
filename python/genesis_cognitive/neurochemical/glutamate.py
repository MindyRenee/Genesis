"""Glutamate — primary excitatory neurotransmitter; drives cortical activity.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Cortical pyramidal neurons (ubiquitous)
    Thalamic relay neurons

Projections:
    Throughout the brain. Glutamate is the brain's main excitatory
    neurotransmitter, present in the majority of all synapses.

Receptors:
    NMDA — voltage-gated, Ca2+ permeable, slow. Key for LTP/LTD.
        Blocked by Mg2+ at rest, requires depolarization to open.
    AMPA — fast Na+/K+ channel, primary EPSC.
    Kainate — fast, less common.
    Metabotropic (mGluR1-8) — Gq/Gi-coupled, modulatory.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.60

~2 µM extracellular. Main excitatory transmitter, highest
concentration. The GLU/GABA balance is the primary E/I axis.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 1.0

Glutamate receptors are the baseline reference for adaptation rate.


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT GLUTAMATE (matrix row 5)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> GLU: +0.05  (weak volume transmission.)
    SRT  -> GLU: -0.05  (weak volume transmission.)
    NE   -> GLU: +0.05  (weak volume transmission.)
    ACh  -> GLU: +0.05  (weak volume transmission.)
    GABA -> GLU: -0.30  (THE primary inhibitory brake. Strongest
                        coupling in the matrix. GABA is the brain's
                        main mechanism for controlling glutamatergic
                        excitation. Glutamate is ~1000x more
                        concentrated than monoamines.)
    OXY  -> GLU: -0.05  (weak volume transmission.)
    END  -> GLU: -0.05  (weak volume transmission.)
    HIST -> GLU: +0.05  (weak volume transmission.)
    BDNF -> GLU: +0.05  (weak volume transmission.)
    ECB  -> GLU: -0.05  (weak volume transmission.)
    OX   -> GLU: +0.05  (weak volume transmission.)
    EPI  -> GLU: +0.05  (weak volume transmission.)
    MEL  -> GLU: -0.05  (weak volume transmission.)
    CORT -> GLU: 0.00   (no direct coupling)
    ADN  -> GLU: 0.00   (no direct coupling)
    VP   -> GLU: 0.00   (no direct coupling)
    CRH  -> GLU: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW GLUTAMATE AFFECTS OTHERS (matrix column 5)
════════════════════════════════════════════════════════════════════════

    GLU -> DA :  +0.05  (VTA glutamate input drives DA.)
    GLU -> SRT:  +0.05  (weak volume transmission.)
    GLU -> NE :  +0.05  (weak volume transmission.)
    GLU -> ACh:  +0.05  (weak volume transmission.)
    GLU -> GABA:  +0.10  (activity-driven feedback inhibition — when
    GLU rises, it drives GABAergic interneurons to fire, providing automatic gain control.
                         This is the core E/I balance mechanism.)
    GLU -> END:  -0.05  (weak volume transmission.)
    GLU -> HIST:  +0.05  (weak volume transmission.)
    GLU -> BDNF:  +0.05  (trophic support.)
    GLU -> ECB:  +0.05  (neural activity stimulates eCB synthesis.)
    GLU -> OX :  +0.05  (weak volume transmission.)
    GLU -> EPI:  +0.05  (weak volume transmission.)
    GLU -> CORT:  0.00   (no direct coupling)
    GLU -> OXY:  0.00   (no direct coupling)
    GLU -> ADN:  0.00   (no direct coupling)
    GLU -> VP :  0.00   (no direct coupling)
    GLU -> CRH:  0.00   (no direct coupling)
    GLU -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

GABA disinhibition target:
    When GABA is very high (>0.8) AND ACh is high (>0.6), GABA
    interneurons inhibit other GABA interneurons, disinhibiting
    glutamate. Glutamate is the target of this disinhibition.

    if gaba_eff > 0.8 and ach_eff > 0.6:
        disinhibition = (gaba_eff - 0.8) * (ach_eff - 0.6) * 0.5
        glutamate_level += disinhibition * dt

Endocannabinoid synthesis trigger:
    Glutamate activity drives endocannabinoid synthesis. eCB is
    produced on demand by postsynaptic neurons when strongly
    activated by glutamate (excitation) or GABA (inhibition).

    activity = eff_glu + eff_gaba
    synthesis = activity * ecb_activity_coupling * dt


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

    Glutamate -> gamma (active cortical processing)
    High GLU + high ACh -> gamma (attention)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Valence equation:
    valence = DA*0.25 + SRT*0.25 + OXY*0.10 + END*0.15 + NE*0.10
            - CORT*0.35 - glu_excess*0.15 + DA*SRT*0.10
    where glu_excess = max(0, GLU - 0.70)
    (glutamate excess > 0.70 contributes -0.15 — models
    excitotoxicity-adjacent agitation from excess excitatory drive)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.3
(Conservative — excess glutamate is excitotoxic)

Python modules that send glutamate impulses:
    - emotional_regulator.py: calming (negative), stress reduction

Python modules that read glutamate:
    - emotion.py: maps to EmotionalState (arousal, agitation)
    - emotional_regulator.py: homeostatic regulation
    - brain_waves.py: gamma baseline

Typical impulse scenarios:
    - Calming: -0.04 (emotional_regulator)
    - Stress reduction: -0.04 (emotional_regulator)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_GLUTAMATE

__all__ = ["CHEM_GLUTAMATE", "Glutamate"]


@dataclass(frozen=True, slots=True)
class Glutamate:
    """Glutamate — primary excitatory neurotransmitter.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_GLUTAMATE
    name: str = "glutamate"
    baseline: float = 0.60
    receptor_adaptation_multiplier: float = 1.0
    receptor_subtypes: tuple[str, str] = ("NMDA", "AMPA")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.3
    origin: tuple[str, ...] = ("cortical pyramidal neurons", "relay")
    projections: tuple[str, ...] = ("throughout brain",)
    receptors: tuple[str, ...] = ("NMDA", "AMPA", "kainate", "mGluR1-8")
    brain_waves: tuple[str, ...] = ("gamma",)
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "primary excitation, E/I balance, cortical activity"
