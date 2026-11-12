"""Endocannabinoid — retrograde signaling; activity-dependent synthesis.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Postsynaptic neurons — endocannabinoids are synthesized on
    demand by postsynaptic neurons when strongly activated by
    glutamate (excitation) or GABA (inhibition).

Projections:
    Retrograde — eCB travels from postsynaptic to presynaptic
    terminals, where it binds CB1 receptors to suppress
    neurotransmitter release. This is the canonical retrograde
    signaling mechanism (Kano et al., 2009).

Receptors:
    CB1 — Gi-coupled, abundant on presynaptic terminals.
        Primary central receptor.
    CB2 — immune system, less central.

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.25

On-demand synthesis. eCB is produced when needed, not stored.
The baseline reflects tonic retrograde signaling.


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 0.8

CB1 receptor trafficking is moderate — slightly slower than the
baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT ENDOCANNABINOID (matrix row 12)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    Endocannabinoids are synthesized on demand from postsynaptic activity.
    DA (reward), GABA/GLU (neural activity), and cortisol (stress
    dampening) all stimulate ECB release.

    DA   -> ECB: +0.05  (weak volume transmission.)
    GABA -> ECB: +0.05  (weak volume transmission.)
    GLU  -> ECB: +0.05  (weak volume transmission.)
    CORT -> ECB: +0.05  (weak volume transmission.)
    SRT  -> ECB: 0.00   (no direct coupling)
    NE   -> ECB: 0.00   (no direct coupling)
    ACh  -> ECB: 0.00   (no direct coupling)
    OXY  -> ECB: 0.00   (no direct coupling)
    END  -> ECB: 0.00   (no direct coupling)
    HIST -> ECB: 0.00   (no direct coupling)
    ADN  -> ECB: 0.00   (no direct coupling)
    BDNF -> ECB: 0.00   (no direct coupling)
    VP   -> ECB: 0.00   (no direct coupling)
    CRH  -> ECB: 0.00   (no direct coupling)
    OX   -> ECB: 0.00   (no direct coupling)
    EPI  -> ECB: 0.00   (no direct coupling)
    MEL  -> ECB: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW ENDOCANNABINOID AFFECTS OTHERS (matrix column 12)
════════════════════════════════════════════════════════════════════════

    ECB -> DA :  +0.05  (weak volume transmission.)
    ECB -> GABA:  -0.05  (depolarization-induced suppression of inhibition (DSI).)
    ECB -> GLU:  -0.05  (depolarization-induced suppression of excitation (DSE).)
    ECB -> CORT:  -0.05  (stress dampening — ECB suppresses cortisol.)
    ECB -> CRH:  -0.05  (stress reduction — ECB suppresses CRH.)
    ECB -> OX :  -0.05  (stress reduction — ECB suppresses orexin.)
    ECB -> SRT:  0.00   (no direct coupling)
    ECB -> NE :  0.00   (no direct coupling)
    ECB -> ACh:  0.00   (no direct coupling)
    ECB -> OXY:  0.00   (no direct coupling)
    ECB -> END:  0.00   (no direct coupling)
    ECB -> HIST:  0.00   (no direct coupling)
    ECB -> ADN:  0.00   (no direct coupling)
    ECB -> BDNF:  0.00   (no direct coupling)
    ECB -> VP :  0.00   (no direct coupling)
    ECB -> EPI:  0.00   (no direct coupling)
    ECB -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Activity-dependent synthesis:
    Endocannabinoids are synthesized on demand by postsynaptic
    neurons when strongly activated by glutamate (excitation) or
    GABA (inhibition). This is the canonical trigger for retrograde
    endocannabinoid signaling (Kano et al., 2009), providing a fast,
    local negative feedback that dampens both excitatory and
    inhibitory input.

    activity = eff_glu + eff_gaba
    synthesis = activity * ecb_activity_coupling * dt
    level += synthesis

    where ecb_activity_coupling = 0.01 (default)

    A highly active cortex can push eCB ~50% above its homeostatic
    baseline through retrograde feedback, producing a mild negative-
    feedback brake on excitation/inhibition.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force, effective level, coupling force, vesicular pool,
tonic/phasic, receptor adaptation, baseline adaptation: same
general formulas as dopamine.py.

Activity-dependent synthesis:
    activity = eff_glu + eff_gaba
    synthesis = activity * ecb_activity_coupling * dt
    level += synthesis
    where ecb_activity_coupling = 0.01 (default)


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no circadian modulation)


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Endocannabinoid -> theta/alpha (relaxation, retrograde feedback)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

No direct role in emergent phase computation. eCB's role is
modulatory — it provides retrograde feedback that stabilizes
neural activity, indirectly supporting all phases.


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.05
(Conservative — eCB is on-demand synthesized, not stored)

Python modules that send eCB impulses:
    - (Currently no direct Python impulses — eCB is driven by
      activity-dependent synthesis in the Rust dynamics)

Python modules that read eCB:
    - emotional_regulator.py: homeostatic regulation
    - emotion.py: maps to EmotionalState (relaxation)

Typical impulse scenarios:
    - (No direct impulses — driven by GLU/GABA activity in Rust)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_ENDOCANNABINOID

__all__ = ["CHEM_ENDOCANNABINOID", "Endocannabinoid"]


@dataclass(frozen=True, slots=True)
class Endocannabinoid:
    """Endocannabinoid — retrograde signaling; activity-dependent synthesis.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_ENDOCANNABINOID
    name: str = "endocannabinoid"
    baseline: float = 0.25
    receptor_adaptation_multiplier: float = 1.0
    receptor_subtypes: tuple[str, str] = ("CB1", "CB2")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.05
    origin: tuple[str, ...] = ("postsynaptic neurons (on-demand)",)
    projections: tuple[str, ...] = ("retrograde (postsynaptic -> presynaptic)",)
    receptors: tuple[str, ...] = ("CB1", "CB2")
    brain_waves: tuple[str, ...] = ("theta", "alpha")
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "retrograde signaling, DSI/DSE, activity feedback"
    activity_coupling: float = 0.01
