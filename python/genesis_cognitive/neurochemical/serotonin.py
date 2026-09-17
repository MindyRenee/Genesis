"""Serotonin — mood stability, wellbeing, and the antidepressant mechanism.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Dorsal raphe nucleus (DRN) — upper autonomics
    Median raphe nucleus (MRN) — upper autonomics
    (Note: the model uses a single "serotonin" scalar, eliding the
    dorsal vs median raphe projection differences.)

Projections:
    Throughout the cortex, hippocampus, amygdala, basal ganglia,
    hyporelay. Serotonin is one of the most widely projecting
    neuromodulators.

Receptors:
    5-HT1A — anxiolytic, high-affinity, Gi-coupled. Activated by
        tonic serotonin (mood stability, calm).
    5-HT2A — hallucinogenic, low-affinity, Gq-coupled. Activated by
        phasic serotonin bursts (excitatory).
    5-HT1B/1D, 5-HT2C, 5-HT3, 5-HT4, 5-HT5, 5-HT6, 5-HT7

Receptor subtypes (default_receptor_subtypes):
    [5-HT1A, 5-HT2A] = [0.50, 0.50] — balanced anxiolytic/excitatory.

The 5-HT1A/5-HT2A distinction mirrors the D1/D2 split: tonic
serotonin activates high-affinity 5-HT1A (anxiolytic), phasic
bursts activate low-affinity 5-HT2A (excitatory). This is why SSRIs
take weeks to work — they raise tonic serotonin, which gradually
upregulates BDNF and restores mood.


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.40

Tonic 5-HT is ~10 nM but is a potent modulator. Oscillates with
~10 min periods (FSCAV data). Regulates mood, sleep, and BDNF
expression.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.9

5-HT1A/2A receptors show moderate trafficking — slightly slower
than the baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT SEROTONIN (matrix row 1)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> SRT: -0.05  (weak volume transmission.)
    NE   -> SRT: +0.05  (weak volume transmission.)
    GABA -> SRT: -0.05  (weak volume transmission.)
    GLU  -> SRT: +0.05  (weak volume transmission.)
    CORT -> SRT: -0.10  (chronic stress depletes serotonin.)
    OXY  -> SRT: +0.05  (weak volume transmission.)
    END  -> SRT: +0.05  (weak volume transmission.)
    BDNF -> SRT: +0.10  (the antidepressant positive feedback — SSRIs raise SRT -> raise BDNF -> .
                        raise SRT. Bounded by homeostatic forces.)
    ACh  -> SRT: 0.00   (no direct coupling)
    HIST -> SRT: 0.00   (no direct coupling)
    ADN  -> SRT: 0.00   (no direct coupling)
    ECB  -> SRT: 0.00   (no direct coupling)
    VP   -> SRT: 0.00   (no direct coupling)
    CRH  -> SRT: 0.00   (no direct coupling)
    OX   -> SRT: 0.00   (no direct coupling)
    EPI  -> SRT: 0.00   (no direct coupling)
    MEL  -> SRT: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW SEROTONIN AFFECTS OTHERS (matrix column 1)
════════════════════════════════════════════════════════════════════════

    SRT -> DA :  -0.05  (weak volume transmission (default region); region-specific:
    -0.025 default, -0.20 striatum, +0.15 PFC.)
    SRT -> NE :  -0.05  (weak volume transmission.)
    SRT -> GABA:  +0.05  (weak volume transmission.)
    SRT -> GLU:  -0.05  (weak volume transmission.)
    SRT -> CORT:  -0.10  (serotonin suppresses cortisol.)
    SRT -> OXY:  +0.05  (weak volume transmission.)
    SRT -> END:  +0.05  (weak volume transmission.)
    SRT -> HIST:  -0.05  (serotonin promotes sleep onset.)
    SRT -> BDNF:  +0.10  (the antidepressant mechanism — SSRIs raise SRT -> raise BDNF -> .
                         neuroplasticity -> depression recovery.)
    SRT -> OX :  -0.05  (serotonin promotes sleep onset.)
    SRT -> ACh:  0.00   (no direct coupling)
    SRT -> ADN:  0.00   (no direct coupling)
    SRT -> ECB:  0.00   (no direct coupling)
    SRT -> VP :  0.00   (no direct coupling)
    SRT -> CRH:  0.00   (no direct coupling)
    SRT -> EPI:  0.00   (no direct coupling)
    SRT -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Subtype-weighted effective level:
    5-HT1A (anxiolytic) promotes calm signaling, 5-HT2A
    (hallucinogenic) promotes excitatory signaling. The net effect
    is neutral at a 50/50 balance, biased up or down by the
    relative receptor weights.

    balance = (ht1a - ht2a) / total
    subtype_effective = base * (1.0 + 0.3 * balance)
    (clamped to [0, 2])

Phasic boost:
    Phasic bursts activate low-affinity 5-HT2A (excitatory)
    receptors, adding signaling beyond the tonic baseline.

    effective = base + phasic_level * ht2a_weight * 0.2
    (clamped to [0, 2])

    Tonic level activates high-affinity 5-HT1A (anxiolytic). Phasic
    bursts activate low-affinity 5-HT2A (excitatory).

BDNF recovery driver:
    Serotonin drives BDNF expression — this is why SSRIs help
    depression. When cortisol drops below 0.3 and BDNF is below
    baseline, BDNF gets a recovery boost proportional to serotonin
    level.

    recovery = srt_level * bdnf_recovery_rate * dt_scale


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force (with autoreceptor nonlinearity):
    Same as dopamine — see dopamine.py for the general formula.

Effective level:
    effective = level * receptor_sensitivity * desensitization * internalization
    (clamped to [0, 2])

Subtype-weighted effective level:
    ht1a = receptor_subtypes[0]
    ht2a = receptor_subtypes[1]
    total = ht1a + ht2a
    balance = (ht1a - ht2a) / total
    subtype_effective = base * (1.0 + 0.3 * balance)
    (clamped to [0, 2])

Phasic boost:
    effective = base + phasic_level * ht2a_weight * 0.2
    (clamped to [0, 2])

Coupling force, vesicular pool, tonic/phasic, receptor adaptation,
baseline adaptation: same general formulas as dopamine.py.


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 - 0.10 * melatonin_factor

Serotonin is the precursor to melatonin. Reduced at night when
melatonin production is high (biochemical conversion).


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Serotonin -> alpha (mood stability, cortical idling)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Valence equation:
    valence = DA*0.25 + SRT*0.25 + OXY*0.10 + END*0.15 + NE*0.10
            - CORT*0.35 - glu_excess*0.15 + DA*SRT*0.10
    (serotonin weight = 0.25 — mood stability)
    (DA × SRT synergy = 0.10 — balanced mood when both high)

Stress phase:
    CORT > 0.65 + SRT < 0.35 + hysteresis
    (low serotonin is a stress indicator)

BDNF/plasticity:
    plasticity_gate = BDNF * (1 - CORT * 0.5)
    (serotonin drives BDNF, which drives plasticity)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.05
(Gentle modulation — 0.01-0.03 is typical, Belmaker & Agam, 2008)

Python modules that send serotonin impulses:
    - emotional_regulator.py: comfort, pleasure, social bonding,
      motivation, stress reduction
    - learning/autonomous.py: learning satisfaction

Python modules that read serotonin:
    - emotion.py: maps to EmotionalState (valence, mood)
    - emotional_regulator.py: homeostatic regulation
    - self/damasio.py: proto-self state (tone)
    - memory/engine.py: consolidation weight
    - brain_waves.py: alpha baseline

Typical impulse scenarios:
    - Comfort: +0.03 (emotional_regulator)
    - Pleasure: +pleasure (emotional_regulator)
    - Social bonding: +0.02 (emotional_regulator)
    - Stress reduction: +0.01 (emotional_regulator)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_SEROTONIN

__all__ = ["CHEM_SEROTONIN", "Serotonin"]


@dataclass(frozen=True, slots=True)
class Serotonin:
    """Serotonin — mood stability and the antidepressant mechanism.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_SEROTONIN
    name: str = "serotonin"
    baseline: float = 0.40
    receptor_adaptation_multiplier: float = 0.9
    receptor_subtypes: tuple[str, str] = ("5-HT1A", "5-HT2A")
    receptor_subtype_balance: tuple[float, float] = (0.50, 0.50)
    max_impulse: float = 0.05
    origin: tuple[str, ...] = ("dorsal raphe", "median raphe")
    projections: tuple[str, ...] = (
        "cortex",
        "hippocampus",
        "amygdala",
        "basal ganglia",
        "hyporelay",
    )
    receptors: tuple[str, ...] = (
        "5-HT1A", "5-HT1B", "5-HT1D", "5-HT2A", "5-HT2C",
        "5-HT3", "5-HT4", "5-HT5", "5-HT6", "5-HT7",
    )
    brain_waves: tuple[str, ...] = ("alpha",)
    circadian_modifier: str = "1.0 - 0.10 * melatonin_factor"
    role: str = "mood stability, wellbeing, BDNF expression, sleep"
