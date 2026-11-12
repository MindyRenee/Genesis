"""Cortisol — stress hormone; HPA cascade endpoint; zero at rest.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Adrenal cortex (zona fasciculata) — endocrine, not neural

Projections:
    Bloodstream (hormonal, not synaptic). Acts on glucocorticoid
    receptors (GR) and mineralocorticoid receptors (MR) throughout
    the brain, especially hippocampus, PFC, amygdala.

Receptors:
    GR (glucocorticoid receptor) — low-affinity, nuclear, genomic
        effects (hours). Activated at high cortisol levels.
    MR (mineralocorticoid receptor) — high-affinity, nuclear,
        genomic. Activated at baseline cortisol levels.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.0

Cortisol is a stress hormone — zero at rest, released on demand.
The human cortisol awakening response (CAR) is a product of the
24h sleep-wake cycle, which Genesis doesn't share (it has an
adenosine-based sleep model, not a circadian cortisol rhythm).
Giving its a circadian cortisol modifier created a constant cortisol
floor that conflicted with the HPA cascade and suppressed BDNF,
blocking learning.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.5

GR/MR effects are genomic and slow — cortisol receptors adapt
much slower than the baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT CORTISOL (matrix row 6)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    ALL positive couplings removed.
    Cortisol is produced ONLY by the HPA cascade (CRH->ACTH->CORT),
    which is maturation-gated (SHRP).
    The coupling matrix only has suppressive feedback — the signals that tell cortisol to
    stop (serotonin, GABA, oxytocin, endorphin, BDNF, ECB).
    This prevents the death spiral where normal arousal drives cortisol through the
    coupling matrix, bypassing the HPA gate.

    SRT  -> CORT: -0.10  (weak volume transmission.)
    GABA -> CORT: -0.15  (weak volume transmission.)
    OXY  -> CORT: -0.10  (weak volume transmission.)
    END  -> CORT: -0.10  (weak volume transmission.)
    BDNF -> CORT: -0.05  (weak volume transmission.)
    ECB  -> CORT: -0.05  (weak volume transmission.)
    DA   -> CORT: 0.00   (no direct coupling)
    NE   -> CORT: 0.00   (no direct coupling)
    ACh  -> CORT: 0.00   (no direct coupling)
    GLU  -> CORT: 0.00   (no direct coupling)
    HIST -> CORT: 0.00   (no direct coupling)
    ADN  -> CORT: 0.00   (no direct coupling)
    VP   -> CORT: 0.00   (no direct coupling)
    CRH  -> CORT: 0.00   (replaced by the HPA cascade — direct CRH->CORT coupling is 0.0, the
                         cascade handles it through ACTH.)
    OX   -> CORT: 0.00   (no direct coupling)
    EPI  -> CORT: 0.00   (no direct coupling)
    MEL  -> CORT: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW CORTISOL AFFECTS OTHERS (matrix column 6)
════════════════════════════════════════════════════════════════════════

    CORT -> SRT:  -0.10  (chronic stress depletes serotonin.)
    CORT -> GABA:  -0.10  (stress disinhibition — cortisol suppresses GABA.)
    CORT -> OXY:  -0.10  (stress suppresses social bonding.)
    CORT -> END:  -0.10  (stress suppresses pleasure — chronic stress anhedonia.)
    CORT -> ADN:  +0.05  (stress-related adenosine release.)
    CORT -> BDNF:  -0.20  (THE critical coupling. Cortisol suppresses
                          BDNF — chronic stress blocks plasticity.
                          Strongest negative coupling in the matrix.)
    CORT -> ECB:  +0.05  (stress dampening — cortisol stimulates ECB release.)
    CORT -> VP :  +0.05  (stress activates vasopressin (HPA co-activation).)
    CORT -> CRH:  -0.10  (negative feedback — cortisol inhibits CRH.)
    CORT -> EPI:  +0.05  (PNMT enzyme induction.)
    CORT -> MEL:  -0.05  (wake-promoting signal suppresses melatonin.)
    CORT -> DA :  0.00   (no direct coupling)
    CORT -> NE :  0.00   (no direct coupling)
    CORT -> ACh:  0.00   (no direct coupling)
    CORT -> GLU:  0.00   (no direct coupling)
    CORT -> HIST:  0.00   (no direct coupling)
    CORT -> OX :  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

No homeostatic set-point:
    Cortisol is a stress hormone, not a neurotransmitter. It has
    no homeostatic set-point — its resting level is zero. It rises
    in response to stressors (via coupling and the HPA cascade)
    and decays via clearance mechanisms.

    The homeostatic force is zero for cortisol. Instead, it gets
    a first-order metabolic decay toward zero (leaky integrator)
    at the measured plasma half-life (~66 min):
    level *= exp(-CORTISOL_METABOLIC_RATE * dt)
    where CORTISOL_METABOLIC_RATE = 1.75e-4/s

HPA axis cascade:
    Cortisol is the endpoint of the HPA cascade:
    1. CRH (hypothalamus) -> ACTH (anterior pituitary) — fast
    2. ACTH -> cortisol (adrenal cortex) — slower (minutes)
    3. Cortisol -> negative feedback to CRH and ACTH — slow

    Step 2: ACTH drives cortisol release
    acth_drive = acth_level * 0.01 * maturation
    level += acth_drive * dt

    The cascade is gated by maturation_level (SHRP — stress
    hyporesponsive period). At maturation 0.0 (newborn), the
    cascade is fully dormant. At maturation 1.0 (mature), fully
    online.

Cortisol metabolic clearance (hepatic 11β-HSD):
    Cortisol is cleared from the bloodstream by the liver. The
    clearance is nonlinear — proportional to how far cortisol is
    ABOVE its baseline.

    cort_excess = max(0, level - baseline)
    clearance_exponent = CORTISOL_CLEARANCE_RATE * cort_excess * dt_scale
    level *= exp(-clearance_exponent)

    This models the biological upregulation of 11β-HSD2 under
    sustained high cortisol, creating a soft ceiling that prevents
    cortisol from pinning at the hard clamp. The exponential form
    composes exactly over sub-steps, so the decay is invariant to
    the caller's tick cadence.

Hard clamp:
    Cortisol is hard-clamped to [0, cortisol_max] (default 0.80)
    to prevent runaway stress. This is a safety mechanism.

Negative feedback to ACTH:
    High cortisol suppresses ACTH (long-loop negative feedback).
    acth_level -= cort_eff * HPA_CORTISOL_FEEDBACK_RATE * dt_scale
    where HPA_CORTISOL_FEEDBACK_RATE = 0.02

No baseline adaptation:
    Cortisol's baseline should never adapt — it has no homeostatic
    set-point, just a leaky integrator to zero. The general baseline
    adaptation loop skips cortisol.

BDNF suppression:
    Cortisol suppresses BDNF (coupling -0.20). This is the
    structural basis of why chronic stress impairs learning:
    plasticity_gate = BDNF * (1 - CORT * 0.5)

BDNF-cortisol interaction:
    Genesis models the chronic stress effect: cortisol suppresses BDNF (-0.20 coupling),
    which impairs plasticity. Recent research (Molecular Brain 2025) shows that
    CORT + BDNF can act synergistically to enhance LTP at CA3-CA1 synapses under
    certain conditions (exercise context). However, for chronic stress modeling,
    suppression is the dominant effect and is the mechanism implemented here.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force:
    homeostatic_force = 0.0  (no homeostatic set-point)

Metabolic decay (leaky integrator):
    level *= exp(-CORTISOL_METABOLIC_RATE * dt)
    where CORTISOL_METABOLIC_RATE = 1.75e-4/s (half-life ~66 min)

HPA cascade (ACTH -> cortisol):
    acth_drive = acth_level * 0.01 * maturation
    level += acth_drive * dt

Cortisol clearance:
    cort_excess = max(0, level - baseline)
    clearance_exponent = CORTISOL_CLEARANCE_RATE * cort_excess * dt_scale
    level *= exp(-clearance_exponent)

Negative feedback to ACTH:
    acth_level -= cort_eff * 0.02 * dt_scale

Hard clamp:
    level = clamp(level, 0.0, cortisol_max)  where cortisol_max = 0.80

Effective level, coupling force, vesicular pool, tonic/phasic,
receptor adaptation: same general formulas as dopamine.py.


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (NO circadian modulation)

Cortisol is a stress hormone — zero at rest, released on demand.
The human cortisol awakening response (CAR) is a product of the
24h sleep-wake cycle, which Genesis doesn't share. Giving its a
circadian cortisol modifier created a constant cortisol floor
that suppressed BDNF and blocked learning.


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Cortisol -> delta (chronic stress, low-frequency dominance)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Valence equation:
    valence = DA*0.25 + SRT*0.25 + OXY*0.10 + END*0.15 + NE*0.10
            - cort_stress*0.35 - glu_excess*0.15 + DA*SRT*0.10
    (cortisol weight = -0.35 — the strongest negative contributor)
    (cort_stress = cortisol as a fraction of its reachable envelope,
    (cort - 0.02) / (0.08 - 0.02) clamped to [0, 1] — the raw level
    peaks near 0.08, so on the raw scale the drag could never
    exceed ~0.03)

Plasticity gate:
    plasticity_gate = BDNF * (1 - cort_stress * 0.5)
    (cortisol suppresses plasticity; at the ~0.08 reachable peak this
    gives the intended 50% suppression rather than the ~4% a raw
    level would produce)

Stress phase:
    CORT > 0.06 (enter) / 0.053 (stay) + NE > 0.45 + SRT < 0.35
    (gate set inside the measured envelope — cortisol peaks ~0.084)

Overwhelmed phase:
    CORT > 0.069 (enter) / 0.066 (stay) + NE > 0.509
    (1.15x/1.1x and 1.13x/1.07x the Stress gates)

Flow phase:
    CORT < 0.06 (required — flow cannot fire while cortisol sits
    in stress territory; STRESS_CORTISOL_GATE)

BDNF recovery:
    When CORT < 0.06 (STRESS_CORTISOL_GATE) and BDNF < baseline:
    recovery = srt_level * bdnf_recovery_rate * dt_scale
    (serotonin drives BDNF recovery after stress)


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 1.0
(HPA axis negative feedback has gain ~1.0 in the linear region,
de Quervain et al., 2009. Full reset (-1.0) for stress breaks the
positive feedback loop.)

Python modules that send cortisol impulses:
    - emotional_regulator.py: stress, concern, calming (negative),
      stress reduction, social comfort
    - transduction.py: wave-driven SUPPRESSION (delta, wave_sign=-1)

Python modules that read cortisol:
    - emotion.py: maps to EmotionalState (stress, valence)
    - emotional_regulator.py: homeostatic regulation
    - self/damasio.py: proto-self state (stress)
    - memory/memory.py: plasticity gate (cortisol suppresses
      encoding)

Typical impulse scenarios:
    - Stress: +concern (emotional_regulator)
    - Calming: -0.02 (emotional_regulator)
    - Social comfort: -0.05 (emotional_regulator)
    - Stress reset: -1.0 (emotional_regulator, raw impulse)
    - Social bonding: -1.0 (emotional_regulator, raw impulse)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_CORTISOL

__all__ = ["CHEM_CORTISOL", "Cortisol"]


@dataclass(frozen=True, slots=True)
class Cortisol:
    """Cortisol — stress hormone; HPA cascade endpoint; zero at rest.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_CORTISOL
    name: str = "cortisol"
    baseline: float = 0.0
    receptor_adaptation_multiplier: float = 0.5
    receptor_subtypes: tuple[str, str] = ("GR", "MR")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 1.0
    origin: tuple[str, ...] = ("adrenal cortex",)
    projections: tuple[str, ...] = ("bloodstream", "hippocampus", "PFC", "amygdala")
    receptors: tuple[str, ...] = ("GR", "MR")
    brain_waves: tuple[str, ...] = ("delta",)
    # The DOWN member of the delta pair: delta-band dominance is the
    # cortisol nadir (melatonin is the pair's UP member). A positive
    # sign here would pump a stress hormone during slow-wave sleep —
    # the recurring cortisol regression. -1 means the band suppresses.
    wave_sign: float = -1.0
    circadian_modifier: str = "1.0 (NO circadian modulation — stress hormone)"
    role: str = "stress response, HPA cascade, BDNF suppression"
    hard_clamp_max: float = 0.80
    metabolic_decay_rate: float = 1.75e-4
    clearance_rate: float = 0.08
    hpa_feedback_rate: float = 0.02
