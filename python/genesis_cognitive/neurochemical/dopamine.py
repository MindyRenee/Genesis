"""Dopamine — reward prediction error, motivation, and the Go/NoGo pathway.

════════════════════════════════════════════════════════════════════════
BIOLOGY
════════════════════════════════════════════════════════════════════════

Origin:
    Ventral tegmental area (VTA) — reward/motivation pathway
    Substantia nigra pars compacta (SNc) — motor pathway

Projections:
    Mesolimbic:    VTA -> nucleus accumbens (reward, motivation)
    Mesocortical:  VTA -> prefrontal cortex (cognitive control)
    Nigrostriatal: SNc -> dorsal striatum (motor control, habits)

Receptors:
    D1 family (D1, D5) — excitatory, Gs-coupled, Go pathway (direct,
        facilitative). Activation facilitates action.
    D2 family (D2, D3, D4) — inhibitory, Gi-coupled, NoGo pathway
        (indirect, suppressive). Activation suppresses action.

Receptor subtypes (default_receptor_subtypes):
    [D1, D2] = [0.50, 0.50] — balanced Go/NoGo at birth.

The D1/D2 distinction is critical: the same dopamine concentration
produces opposite effects depending on which receptor it binds to.
D1 activation facilitates action (Go); D2 activation suppresses
action (NoGo). This is the basal ganglia's action selection mechanism.


════════════════════════════════════════════════════════════════════════
DEFAULT BASELINE
════════════════════════════════════════════════════════════════════════

    baseline = 0.35

Tonic DA is very low (~0.06 nM in PFC, ~5 nM in VTA) but phasic
bursts are large. The baseline reflects tonic firing rate, not
concentration. Spontaneous impulses at ~0.01/sec (Mohebi et al.,
2019).


════════════════════════════════════════════════════════════════════════
RECEPTOR ADAPTATION MULTIPLIER
════════════════════════════════════════════════════════════════════════

    multiplier = 1.2

D2/D3 receptors traffic rapidly. Dopaminergic receptors adapt
faster than the baseline reference (glutamate at 1.0×).


════════════════════════════════════════════════════════════════════════
COUPLING: HOW OTHERS AFFECT DOPAMINE (matrix row 0)
════════════════════════════════════════════════════════════════════════

Index order: DA, SRT, NE, ACh, GABA, GLU, CORT, OXY, END, HIST,
ADN, BDNF, ECB, VP, CRH, OX, EPI, MEL

    SRT  -> DA : -0.05  (weak volume transmission.)
    NE   -> DA : +0.05  (weak volume transmission.)
    ACh  -> DA : +0.05  (weak volume transmission.)
    GABA -> DA : -0.10  (GABAergic inhibition of VTA DA neurons.)
    GLU  -> DA : +0.05  (weak volume transmission.)
    OXY  -> DA : +0.05  (weak volume transmission.)
    END  -> DA : +0.10  (endorphin disinhibition of DA — mu-opioid
    -> GABA interneuron -> DA disinhibition. Strongest positive
    coupling for DA, appropriate for runner's high.)
    HIST -> DA : +0.05  (weak volume transmission.)
    BDNF -> DA : +0.05  (weak volume transmission.)
    ECB  -> DA : +0.05  (weak volume transmission.)
    OX   -> DA : +0.05  (weak volume transmission.)
    MEL  -> DA : -0.05  (sleep suppression of dopaminergic arousal.)
    CORT -> DA : 0.00   (cortisol does not directly promote dopamine.)
    ADN  -> DA : 0.00   (adenosine's effect on dopamine is handled by the sleep pressure
    mechanism (level accumulation/clearance), not direct coupling.)
    VP   -> DA : 0.00   (no direct coupling)
    CRH  -> DA : 0.00   (no direct coupling)
    EPI  -> DA : 0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
COUPLING: HOW DOPAMINE AFFECTS OTHERS (matrix column 0)
════════════════════════════════════════════════════════════════════════

    DA  -> SRT:  -0.05  (weak volume transmission.)
    DA  -> NE :  +0.05  (weak volume transmission.)
    DA  -> ACh:  +0.05  (co-fluctuation at ~2 Hz (Howe et al., 2023).)
    DA  -> GABA:  -0.05  (weak volume transmission.)
    DA  -> GLU:  +0.05  (weak volume transmission.)
    DA  -> OXY:  +0.05  (weak volume transmission.)
    DA  -> END:  +0.05  (weak volume transmission.)
    DA  -> HIST:  +0.05  (weak volume transmission.)
    DA  -> BDNF:  +0.05  (trophic support.)
    DA  -> ECB:  +0.05  (reward-driven eCB synthesis.)
    DA  -> VP :  +0.05  (weak volume transmission.)
    DA  -> CORT:  0.00   (no direct coupling)
    DA  -> ADN:  0.00   (no direct coupling)
    DA  -> CRH:  0.00   (no direct coupling)
    DA  -> OX :  0.00   (no direct coupling)
    DA  -> EPI:  0.00   (no direct coupling)
    DA  -> MEL:  0.00   (no direct coupling)

════════════════════════════════════════════════════════════════════════
SPECIAL DYNAMICS
════════════════════════════════════════════════════════════════════════

Subtype-weighted effective level:
    Dopamine's coupling to other chemicals uses its D1/D2 receptor
    subtype-weighted effective level. When D1 dominates, dopamine's
    coupling to arousal chemicals (NE, ACh, GLU) is amplified; when
    D2 dominates, dopamine's coupling to inhibitory chemicals (GABA)
    is amplified instead. The same dopamine concentration produces
    different coupling depending on receptor state.

    subtype_effective_level = base * (d1 - d2) / total * 2.0 + base
    (clamped to [0, 2])

Phasic boost:
    Phasic bursts activate low-affinity D1 (excitatory) receptors,
    adding signaling beyond the tonic baseline.

    effective = base + phasic_level * d1_weight * 0.3
    (clamped to [0, 2])

    Tonic level activates high-affinity D2 receptors (mood
    maintenance). Phasic bursts activate low-affinity D1 receptors
    (reward signaling).

Region-specific SRT->DA coupling override:
    The default matrix has SRT->DA = -0.05, but this is always
    overridden by the region-specific path:
    - Default region:  -0.025  (average of striatum and PFC paths)
    - Striatum:        -0.20   (5-HT2A, inhibitory)
    - PFC:             +0.15   (5-HT1A, facilitatory)


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FORMULAS
════════════════════════════════════════════════════════════════════════

Homeostatic force (with autoreceptor nonlinearity):

    deviation = baseline - level
    abs_dev = |deviation|
    excess = max(0, abs_dev - AUTORECEPTOR_THRESHOLD)
    nonlinear_factor = 1 + AUTORECEPTOR_GAIN * excess^2
    homeostatic_force = deviation * homeostatic_rate * nonlinear_factor

    where:
        AUTORECEPTOR_THRESHOLD = 0.15
        AUTORECEPTOR_GAIN = 200.0
        homeostatic_rate = 0.04 (default)

    Calibration:
    - At deviation 0.10: factor = 1.0 (purely linear)
    - At deviation 0.20: factor = 1.5 (mild correction)
    - At deviation 0.30: factor = 5.5 (moderate)
    - At deviation 0.50: factor = 25.5 (strong)

Effective level:

    effective = level * receptor_sensitivity * desensitization * internalization
    (clamped to [0, 2])

Subtype-weighted effective level:

    d1 = receptor_subtypes[0]
    d2 = receptor_subtypes[1]
    total = d1 + d2
    subtype_effective = base * (d1 - d2) / total * 2.0 + base
    (clamped to [0, 2])

Phasic boost:

    effective = base + phasic_level * d1_weight * 0.3
    (clamped to [0, 2])

Coupling force (sum over all chemicals j != i):

    coupling_force = sum_j  C[i][j] * deviation_j * level_gate_j * coupling_gain_j
    coupling_force *= coupling_scale  (0.15)

    where:
        deviation_j = max(-AUTORECEPTOR_THRESHOLD, eff_j - baseline_j)
        level_gate_j = eff_j^2 / (baseline_j^2 + eff_j^2)  (Hill, n=2)
        coupling_gain_j = clamp(1 + 0.3 * tanh(self_dev * j_dev), 0.7, 1.3)

Vesicular pool:
    On impulse: pool -= |magnitude| * 0.3 (depletion)
    Per tick:   pool += (1 - pool) * 0.001 * dt_scale  (synthesis)
                pool += level * 0.0005 * dt_scale  (reuptake)

Tonic/phasic:
    phasic *= 0.80^dt_scale  (fast decay, ~1 sec)
    tonic += (level - tonic) * 0.002 * dt_scale  (slow, ~50 sec)

Receptor adaptation:
    sensitivity: downreg fast (1.0×), upreg 3× slower (0.3×)
    desensitization: 0.2× rate, floor 0.2
    internalization: 0.05× rate, floor 0.45
    All have sleep resensitization + receptor turnover

Baseline adaptation:
    baseline += (level - baseline) * rate + (genetic - baseline) * rate
    (rate = baseline_adaptation_rate * dt_scale)


════════════════════════════════════════════════════════════════════════
CIRCADIAN MODULATION
════════════════════════════════════════════════════════════════════════

    modifier = 1.0 + 0.15 * sin(2π * (phase - 0.25))

Daytime elevation: slightly higher during the day, peaking around
noon (phase 0.5).


════════════════════════════════════════════════════════════════════════
BRAIN WAVE COUPLING
════════════════════════════════════════════════════════════════════════

    Dopamine -> theta/gamma (reward encoding, memory formation)
    High dopamine + high ACh -> gamma (active processing)


════════════════════════════════════════════════════════════════════════
ROLE IN EMERGENT PHASES
════════════════════════════════════════════════════════════════════════

Arousal equation:
    arousal_promoters = NE*0.20 + HIST*0.15 + DA*0.15 + ACh*0.15
                       + OX*0.20 + EPI*0.10
    (dopamine weight = 0.15 — reward-driven alertness)

Valence equation:
    valence = DA*0.25 + SRT*0.25 + OXY*0.10 + END*0.15 + NE*0.10
            - CORT*0.35 - glu_excess*0.15 + DA*SRT*0.10
    (dopamine weight = 0.25 — reward/wanting)
    (DA × SRT synergy = 0.10 — balanced mood when both high)

Flow phase:
    DA > 0.65 (enter) / 0.55 (stay) + ACh > 0.50 + NE > 0.40
    + CORT < 0.30


════════════════════════════════════════════════════════════════════════
APPLICATION IN GENESIS
════════════════════════════════════════════════════════════════════════

Maximum impulse magnitude: 0.05
(Conservative — dopamine is potent and easily overstimulated)

Python modules that send dopamine impulses:
    - emotional_regulator.py: reward, motivation, novelty, engagement,
      pleasure, joy, social warmth
    - learning/td.py: reward prediction error
    - learning/curiosity.py: curiosity-driven exploration
    - vision.py: novel/salient visual input
    - cognition/engine.py: curiosity-driven exploration

Python modules that read dopamine:
    - emotion.py: maps to EmotionalState (valence, arousal)
    - emotional_regulator.py: homeostatic regulation
    - self/damasio.py: proto-self state
    - memory/engine.py: encoding weight (novel, arousing)
    - brain_waves.py: theta/gamma baseline

Typical impulse scenarios:
    - Novel visual input: +0.05 (vision.py)
    - Reward prediction error: proportional to RPE (td.py)
    - Social warmth: +warmth * 0.7 (emotional_regulator)
    - Curiosity: +engagement (emotional_regulator)
    - Pleasure: +pleasure (emotional_regulator)
"""

from __future__ import annotations

from dataclasses import dataclass

from genesis_client.protocol import CHEM_DOPAMINE

__all__ = ["CHEM_DOPAMINE", "Dopamine"]


@dataclass(frozen=True, slots=True)
class Dopamine:
    """Dopamine — reward prediction error and motivation.

    All values synchronized with src/state/neurochemical.rs.
    """

    index: int = CHEM_DOPAMINE
    name: str = "dopamine"
    baseline: float = 0.35
    receptor_adaptation_multiplier: float = 1.2
    receptor_subtypes: tuple[str, str] = ("D1", "D2")
    receptor_subtype_balance: tuple[float, float] = (0.50, 0.50)
    max_impulse: float = 0.05
    origin: tuple[str, ...] = ("VTA", "SNc")
    projections: tuple[str, ...] = (
        "nucleus accumbens",
        "prefrontal cortex",
        "dorsal striatum",
    )
    receptors: tuple[str, ...] = ("D1", "D2", "D3", "D4", "D5")
    brain_waves: tuple[str, ...] = ("theta", "gamma")
    circadian_modifier: str = "1.0 + 0.15 * sin(2π * (phase - 0.25))"
    role: str = "reward prediction error, motivation, motor control"
