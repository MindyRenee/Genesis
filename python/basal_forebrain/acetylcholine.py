"""Acetylcholine — attention, focus, cortical arousal, and memory encoding.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Basal forebrain (nucleus of Meynert) — cortical projections
    Pedunculopontine/laterodorsal tegmental nuclei — autonomics
    (REM-on cholinergic neurons)

Projections:
    Throughout cortex (attention, cortical arousal)
    Hippocampus (memory encoding)
    Thalamus (arousal)
    Brainstem REM-on neurons (dreaming)

Receptors:
    Nicotinic (nAChR) — ionotropic, fast, desensitize in ms
    Muscarinic (mAChR) — Gq/Gi-coupled, slower, M1-M5 subtypes

No subtype differentiation modeled (default_receptor_subtypes = [1.0, 1.0]).


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.35

Tonic ACh ~1 nM in PFC. Sets cortical activation level. Co-
fluctuates with DA at ~2 Hz (Howe et al., 2023).


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 1.2

Nicotinic receptors desensitize in ms — cholinergic receptors adapt
faster than the baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT ACETYLCHOLINE (matrix row 3)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    DA   -> ACh: +0.05  (co-fluctuation at ~2 Hz (Howe et al., 2023).)
    NE   -> ACh: +0.05  (weak volume transmission.)
    GABA -> ACh: -0.10  (inhibition of cholinergic basal forebrain.)
    GLU  -> ACh: +0.05  (weak volume transmission.)
    HIST -> ACh: +0.05  (weak volume transmission.)
    BDNF -> ACh: +0.05  (weak volume transmission.)
    OX   -> ACh: +0.05  (orexin-driven cortical activation.)
    MEL  -> ACh: -0.05  (sleep suppression of cholinergic arousal.)
    SRT  -> ACh: 0.00   (no direct coupling)
    CORT -> ACh: 0.00   (no direct coupling)
    OXY  -> ACh: 0.00   (no direct coupling)
    END  -> ACh: 0.00   (no direct coupling)
    ADN  -> ACh: 0.00   (no direct coupling)
    ECB  -> ACh: 0.00   (no direct coupling)
    VP   -> ACh: 0.00   (no direct coupling)
    CRH  -> ACh: 0.00   (no direct coupling)
    EPI  -> ACh: 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW ACETYLCHOLINE AFFECTS OTHERS (matrix column 3)
════════════════════════════════════════════════════════════════════════

    ACh -> DA :  +0.05  (co-fluctuation at ~2 Hz (Howe et al., 2023).)
    ACh -> NE :  +0.05  (weak volume transmission.)
    ACh -> GABA:  -0.05  (weak volume transmission.)
    ACh -> GLU:  +0.05  (weak volume transmission.)
    ACh -> HIST:  +0.05  (weak volume transmission.)
    ACh -> BDNF:  +0.05  (trophic support.)
    ACh -> OX :  +0.05  (weak volume transmission.)
    ACh -> EPI:  +0.10  (preganglionic cholinergic stimulation of adrenal medulla — the strongest
                        coupling for EPI.)
    ACh -> SRT:  0.00   (no direct coupling)
    ACh -> CORT:  0.00   (no direct coupling)
    ACh -> OXY:  0.00   (no direct coupling)
    ACh -> END:  0.00   (no direct coupling)
    ACh -> ADN:  0.00   (no direct coupling)
    ACh -> ECB:  0.00   (no direct coupling)
    ACh -> VP :  0.00   (no direct coupling)
    ACh -> CRH:  0.00   (no direct coupling)
    ACh -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Acetylcholinesterase (AChE) degradation:
    ACh is rapidly broken down in the synaptic cleft by AChE, one
    of the fastest enzymes in the body (turnover ~25,000 molecules/
    sec). This happens on a millisecond timescale — far faster than
    the homeostatic restoring force (seconds to minutes).

    The degradation applies only to the EXCESS above the homeostatic
    baseline (not the absolute level). This preserves the biological
    intent — high ACh bursts decay rapidly — while allowing the
    homeostatic force to maintain ACh near baseline at rest.

    ACH_DEGRADATION_RATE = 0.03 per 100 ms step (0.3/sec)
    An excess of 0.10 above baseline decays by 3% per step, reaching
    baseline in ~3.5 steps (~0.35 seconds) without homeostatic help.

    decay_factor = (1 - ACH_DEGRADATION_RATE)^dt_scale
    level -= ach_excess * (1 - decay_factor)
    where ach_excess = max(0, level - homeostatic_baseline)

    This is compressed from the biological millisecond timescale
    but preserves the key property: ACh is the fastest-degrading
    neurotransmitter in the model.

Network-driven release (sleep):
    The scripted cholinergic rebound is retired. When the sleep-wake
    network is running, ACh is driven directly by the network's
    populations (LC for waking, REM-on for REM) toward a linear
    release target — the network, not a threshold latch, produces
    the cholinergic rebound into REM:

        ach_target = ACH_BASE + ACH_FROM_LC * lc + ACH_FROM_REM * rem
        level relaxes toward target: level -> target - exp(-rate) * (target - level)

    Steady states follow the microdialysis profile (Vazquez &
    Baghdoyan, 2001): NREM 0.34, waking 0.58, REM 0.77 — the
    cholinergic/aminergic flip-flop (Hobson & McCarley, 1975):
    REM-on cells (cholinergic) are active while REM-off cells
    (noradrenergic, serotonergic) are silenced.

    While network-driven, the main loop skips ACh entirely (continue)
    so the homeostatic/coupling forces don't fight the release target.

GABA disinhibition:
    When GABA is very high (>0.8) AND ACh is high (>0.6), GABA
    interneurons inhibit other GABA interneurons, disinhibiting
    glutamate. ACh is the co-trigger for this paradoxical
    disinhibition.

    if gaba_eff > 0.8 and ach_eff > 0.6:
        disinhibition = (gaba_eff - 0.8) * (ach_eff - 0.6) * 0.5
        glutamate_level += disinhibition * dt


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force, effective level, coupling force, vesicular pool,
tonic/phasic, receptor adaptation, baseline adaptation: same
general formulas as dopamine.py.

AChE degradation (excess-only):
    ach_excess = max(0, level - homeostatic_baseline)
    decay_factor = (1 - ACH_DEGRADATION_RATE)^dt_scale
    level -= ach_excess * (1 - decay_factor)
    where ACH_DEGRADATION_RATE = 0.03

Network release target (network-driven states):
    ach_target = ACH_BASE + ACH_FROM_LC * lc + ACH_FROM_REM * rem
    level -> ach_target - exp(-rate * dt_scale) * (ach_target - level)
    steady states: NREM 0.34, waking 0.58, REM 0.77


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 (no circadian modulation)

ACh has no dedicated circadian baseline modifier. Its arousal
function is driven by the arousal equation and the cholinergic
rebound during sleep.


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Acetylcholine -> gamma (cortical activation, attention)
    High ACh + high DA -> gamma (active processing)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Arousal equation:
    arousal_promoters = NE*0.20 + HIST*0.15 + DA*0.15 + ACh*0.15
                       + OX*0.20 + EPI*0.10
    (ACh weight = 0.15 — cortical activation)

REM phase:
    Owned by the Rust sleep-wake network — REM-on firing above 3 Hz.
    The chemistry-side ACh/NE REM gates are retired; the network's
    drive produces the cholinergic-high, aminergic-silent REM profile
    (ACh 0.77, NE ~0, serotonin 0.07).

Flow phase:
    DA > 0.40 + ACh > 0.44 + NE > 0.40 + CORT < 0.30

GABA disinhibition:
    ACh > 0.6 + GABA > 0.8 -> glutamate disinhibition


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.1
(Conservative — ACh is potent and rapidly degrading)

Python modules that send ACh impulses:
    - emotional_regulator.py: engagement, wakefulness, curiosity,
      drowsiness reduction

Python modules that read ACh:
    - emotion.py: maps to EmotionalState (arousal)
    - emotional_regulator.py: homeostatic regulation
    - memory/memory.py: encoding weight (attention to memory)
    - brain_waves.py: gamma baseline

Typical impulse scenarios:
    - Engagement: +engagement * 0.6 (emotional_regulator)
    - Wakefulness: +0.06 (emotional_regulator)
    - Curiosity: +novelty * 0.7 (emotional_regulator)
    - Drowsiness: -0.02 (emotional_regulator)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_ACETYLCHOLINE

__all__ = ["CHEM_ACETYLCHOLINE", "Acetylcholine"]


@dataclass(frozen=True, slots=True)
class Acetylcholine:
    """Acetylcholine — attention, focus, cortical arousal, memory encoding.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_ACETYLCHOLINE
    name: str = "acetylcholine"
    baseline: float = 0.35
    receptor_adaptation_multiplier: float = 1.2
    receptor_subtypes: tuple[str, str] = ("nicotinic", "muscarinic")
    receptor_subtype_balance: tuple[float, float] = (1.0, 1.0)
    max_impulse: float = 0.1
    origin: tuple[str, ...] = ("basal forebrain", "PPT/LDT")
    projections: tuple[str, ...] = (
        "cortex",
        "hippocampus",
        "relay",
        "autonomics REM-on",
    )
    receptors: tuple[str, ...] = ("nicotinic", "M1", "M2", "M3", "M4", "M5")
    brain_waves: tuple[str, ...] = ("gamma",)
    circadian_modifier: str = "1.0 (no circadian modulation)"
    role: str = "attention, focus, cortical arousal, memory encoding, REM"
    degradation_rate: float = 0.03
