"""Neurochemical system — Genesis's 18-chemical affective engine.

════════════════════════════════════════════════════════════════════════
WHAT THIS IS
════════════════════════════════════════════════════════════════════════

The neurochemical system is the brain's neuromodulatory layer. Unlike
the subsystems (which process information), neurochemicals modulate HOW
information is processed — they change the gain, plasticity, and
mode of every brain region simultaneously.

In the biological brain, neuromodulators are produced by small
nuclei in the autonomics and hyporelay and projected widely across
the cortex. Each neurotransmitter system is a separate pathway with
its own origin, projections, receptor types, and computational role.

In Genesis, the 18-chemical coupled dynamics live in the Rust daemon
(``src/state/neurochemical.rs``). The Python side reads levels via
``client.get_neuro_summary()`` and sends impulses via
``client.neuro_impulse(CHEM_X, value)``. This folder gives each
chemical its own home: documentation, mathematical model, and
application helpers.

The 18 chemicals:

    Index  Chemical          Origin (biological)                Baseline
    -----  --------          ------------------                  -------
    0      Dopamine          VTA, SNc (midbrain)                 0.35
    1      Serotonin         Dorsal/median raphe (autonomics)     0.40
    2      Norepinephrine    Locus coeruleus (autonomics)         0.30
    3      Acetylcholine     Basal forebrain (nucleus of Meynert) 0.35
    4      GABA              Inhibitory interneurons (ubiquitous) 0.50
    5      Glutamate         Excitatory neurons (ubiquitous)     0.60
    6      Cortisol          HPA axis (hyporelay -> adrenal)  0.00
    7      Oxytocin          Hyporelay (paraventricular)      0.25
    8      Endorphin         Hyporelay, pituitary             0.25
    9      Histamine         Tuberomammillary nucleus            0.30
    10     Adenosine         Astrocytes, basal forebrain         0.20
    11     BDNF              Hippocampus, cortex (neurotrophin)  0.45
    12     Endocannabinoid   Postsynaptic neurons (retrograde)   0.25
    13     Vasopressin       Hyporelay (supraoptic)           0.20
    14     CRH               Hyporelay (paraventricular)       0.00
    15     Orexin            Lateral hyporelay                0.30
    16     Epinephrine       Adrenal medulla (sympathetic)       0.15
    17     Melatonin         Pineal gland (circadian)            0.10

Cortisol and CRH have zero baseline — they are stress hormones with no
homeostatic set-point. They rise only via the HPA cascade and external
impulses, then decay via clearance.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
════════════════════════════════════════════════════════════════════════

The full coupled dynamics are in Rust. This documents the math.

Coupled Dynamics (18x18 coupling matrix)
-----------------------------------------

The 18 chemicals are not independent. An 18x18 coupling matrix
defines how each chemical modulates the others' rates of change.

    dx_i/dt = f_i(x, u_i) + sum_j  C_ij * g_j(x_j)

    where:
        x_i       = level of chemical i
        f_i(x,u)  = intrinsic dynamics (decay, production, feedback)
        C_ij      = coupling matrix entry (how chemical j affects i)
        g_j(x_j)  = coupling function (nonlinear, Hill-function gated)
        u_i       = external input (impulse from Python)

The coupling is nonlinear via three mechanisms:

1. Hill-function threshold gating (n=2 cooperativity):

    g_j(x_j) = x_j^n / (EC50^n + x_j^n)

    where:
        EC50 = half-maximal effective concentration
        n    = Hill coefficient (cooperativity, n=2)

    Below EC50, coupling is subthreshold; above, it activates
    steeply. This models receptor dose-response curves.

2. State-dependent gain modulation:

    C_ij_effective = C_ij * (1 + alpha * sign(x_i - baseline) *
                                  sign(x_j - baseline))

    where:
        alpha = up to 0.3 (30% amplification/suppression)

    When two chemicals co-activate (both above baseline), coupling
    is amplified (synergy). When they antagonize (one above, one
    below), coupling is reduced (antagonism).

3. Dopamine subtype-aware coupling:

    DA_effective = D1_weight * DA + D2_weight * DA

    where:
        D1_weight = D1 receptor fraction (excitatory, Go pathway)
        D2_weight = D2 receptor fraction (inhibitory, NoGo pathway)

    The same dopamine concentration produces different coupling
    depending on receptor state.

In Genesis: implemented in ``src/state/neurochemical.rs``. The
coupling matrix is mmap'd and plastic (metaplasticity — the system
learns how to learn).


Receptor Adaptation
---------------------

Each chemical has a receptor_sensitivity field. The effective level
— what consumers actually see — is not the raw concentration.

    effective_level = level * receptor_sensitivity

    Receptor dynamics:
    dS/dt = (S_target - S) / tau_adapt

    S_target = 1 / (1 + (level / EC50_adapt)^n)

    where:
        S             = receptor_sensitivity (0..1)
        S_target      = target sensitivity
        EC50_adapt    = adaptation threshold
        tau_adapt     = adaptation time constant (slow, minutes)
        n             = Hill coefficient

    Sustained high level -> S_target decreases (downregulation,
    tolerance). Sustained low level -> S_target increases
    (upregulation, sensitization).

In Genesis: implemented in Rust. The Python side reads effective
levels via ``NeuroSummary``.


Metaplasticity
---------------

The coupling matrix itself is plastic. It lives in the mmap'd
struct and can be modified by the dynamics engine.

    dC_ij/dt = eta_meta * error_signal * eligibility_ij

    where:
        C_ij          = coupling matrix entry
        eta_meta      = metaplastic learning rate
        error_signal  = prediction error from active inference
        eligibility   = which couplings were recently active

The system learns how to learn — the coupling between chemicals
adapts based on experience. This is implemented in the Rust
daemon's active inference module.


Emotional Gating of Memory
----------------------------

Memory pointers carry weights derived from neurochemistry:

    encoding_weight    = f(dopamine, norepinephrine, acetylcholine)
    consolidation_weight = f(BDNF, serotonin, sleep_stage)
    retrieval_weight    = f(acetylcholine, serotonin)
    plasticity_weight   = f(BDNF, cortisol)

    where:
        High dopamine + NE + ACh -> strong encoding (novel, arousing)
        High BDNF + low cortisol -> strong consolidation (plastic)
        High ACh -> strong retrieval (attention to memory)
        High cortisol -> suppressed plasticity (stress blocks learning)

In Genesis: the memory system reads these weights from the
neurochemical state and modulates memory operations accordingly.


Emergent Phase Transitions
-----------------------------

Mental states emerge from neurochemical dynamics, not manual switches:

    Phase = {
        FLOW:        high dopamine + high ACh + low cortisol + low adenosine
        STRESS:      high cortisol + high NE + low BDNF
        DROWSY:      high adenosine + low orexin + low histamine
        SLEEPING:    high melatonin + high adenosine + low orexin
        OVERWHELMED: high cortisol + low serotonin + low GABA
    }

Two independent axes:
    Task zone  (what it's doing): IDLE, CONVERSATION, LEARNING, etc.
    Mental phase (what state): FLOW, STRESS, DROWSY, SLEEPING, etc.

In Genesis: the daemon computes emergent phases from the
neurochemical vector. The Python side reads them via
``NeuroSummary``.


════════════════════════════════════════════════════════════════════════
BRAIN WAVES
════════════════════════════════════════════════════════════════════════

Neurochemicals and brain waves are deeply coupled:

    Acetylcholine -> gamma (cortical activation, attention)
    Serotonin -> alpha (mood stability, cortical idling)
    GABA -> alpha/beta (inhibition, rhythm stabilization)
    Glutamate -> gamma (excitation, active processing)
    Dopamine -> theta/gamma (reward, memory encoding)
    Melatonin -> delta (sleep, slow waves)
    Adenosine -> delta (sleep pressure, slow waves)
    Orexin -> beta/gamma (wakefulness, alertness)

The neurochemical state determines which brain wave bands are
dominant. The brain wave system (``brain_waves.py``) reads the
neurochemical state to set its baseline rhythm.


════════════════════════════════════════════════════════════════════════
MODULE ORGANIZATION
════════════════════════════════════════════════════════════════════════

Each chemical has its own file, documenting its biological origin,
projections, receptor types, computational role, mathematical model,
and application in Genesis. The files re-export the protocol
constant for that chemical.

    dopamine.py         Reward prediction error, motivation, D1/D2
    serotonin.py        Mood, impulse control, satiety
    norepinephrine.py   Arousal, vigilance, attention
    acetylcholine.py    Attention, plasticity, memory encoding
    gaba.py             Inhibition, anxiety reduction, rhythm
    glutamate.py        Excitation, the main neurotransmitter
    cortisol.py         Stress response, HPA axis, BDNF suppression
    oxytocin.py         Social bonding, trust, dyadic attunement
    endorphin.py        Pain relief, reward, euphoria
    histamine.py        Wakefulness, arousal, inflammation
    adenosine.py        Sleep pressure, fatigue, ATP byproduct
    bdnf.py             Neurotrophin, plasticity, growth
    endocannabinoid.py  Retrograde signaling, appetite, stress buffer
    vasopressin.py      Social behavior, water balance, aggression
    crh.py              Stress cascade trigger, HPA axis initiator
    orexin.py           Wakefulness, appetite, arousal maintenance
    epinephrine.py      Fight-or-flight, sympathetic arousal
    melatonin.py        Circadian rhythm, sleep onset, antioxidant

The coupled dynamics (18x18 coupling matrix, receptor adaptation,
metaplasticity) are in Rust (``src/state/neurochemical.rs``). This
folder documents each chemical and provides Python-side helpers
for applying impulses.

Related top-level modules:

    (top-level) emotion.py
        Reads neurochemistry -> maps to EmotionalState. The
        consumer of the neurochemical system's output.

    (top-level) emotional_regulator.py
        Monitors neurochemistry -> sends corrective impulses. The
        Python-side homeostatic regulator.

    (top-level) brain_waves.py
        Reads neurochemistry -> sets brain wave baseline. The
        neurochemical-brain-wave coupling.
"""

from __future__ import annotations

from .acetylcholine import CHEM_ACETYLCHOLINE, Acetylcholine
from .adenosine import CHEM_ADENOSINE, Adenosine
from .bdnf import BDNF, CHEM_BDNF
from .cortisol import CHEM_CORTISOL, Cortisol
from .crh import CHEM_CRH, CRH
from .dopamine import CHEM_DOPAMINE, Dopamine
from .endocannabinoid import CHEM_ENDOCANNABINOID, Endocannabinoid
from .endorphin import CHEM_ENDORPHIN, Endorphin
from .epinephrine import CHEM_EPINEPHRINE, Epinephrine
from .gaba import CHEM_GABA, GABA
from .glutamate import CHEM_GLUTAMATE, Glutamate
from .histamine import CHEM_HISTAMINE, Histamine
from .melatonin import CHEM_MELATONIN, Melatonin
from .norepinephrine import CHEM_NOREPINEPHRINE, Norepinephrine
from .orexin import CHEM_OREXIN, Orexin
from .oxytocin import CHEM_OXYTOCIN, Oxytocin
from .serotonin import CHEM_SEROTONIN, Serotonin
from .vasopressin import CHEM_VASOPRESSIN, Vasopressin

__all__ = [
    "BDNF",
    "CHEM_ACETYLCHOLINE",
    "CHEM_ADENOSINE",
    "CHEM_BDNF",
    "CHEM_CORTISOL",
    "CHEM_CRH",
    "CHEM_DOPAMINE",
    "CHEM_ENDOCANNABINOID",
    "CHEM_ENDORPHIN",
    "CHEM_EPINEPHRINE",
    "CHEM_GABA",
    "CHEM_GLUTAMATE",
    "CHEM_HISTAMINE",
    "CHEM_MELATONIN",
    "CHEM_NOREPINEPHRINE",
    "CHEM_OREXIN",
    "CHEM_OXYTOCIN",
    "CHEM_SEROTONIN",
    "CHEM_VASOPRESSIN",
    "CRH",
    "GABA",
    "Acetylcholine",
    "Adenosine",
    "Cortisol",
    "Dopamine",
    "Endocannabinoid",
    "Endorphin",
    "Epinephrine",
    "Glutamate",
    "Histamine",
    "Melatonin",
    "Norepinephrine",
    "Orexin",
    "Oxytocin",
    "Serotonin",
    "Vasopressin",
]
