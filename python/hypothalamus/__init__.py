"""Hypothalamus — homeostasis, the stress axis, and the neuroendocrine hub.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The hypothalamus sits in the diencephalon, above the brainstem and
below the thalamus. It is the controller of the body's set-points:
it senses blood chemistry, temperature, and energy state, and it
drives the endocrine glands through the pituitary. Its nuclei are the
origin of several neuromodulator and neuropeptide systems that the
rest of the brain reads as state:

    Nucleus                        Output               Claimed as
    -----------------------------  -------------------  ------------------
    Paraventricular (PVN)          CRH, vasopressin     HPA axis, osmotic
    Supraoptic (SON)               Vasopressin, oxytoc. fluid balance, bonding
    Lateral hypothalamus           Orexin/hypocretin    wake stabilization
    Tuberomammillary (TMN)         Histamine            cortical arousal
    Suprachiasmatic (SCN)          Circadian timing     melatonin gating

Melatonin is synthesized by the pineal gland, but its timing is set
by the SCN, so it is claimed here as the circadian output.

Two of these systems are shared with other regions, and the sharing
is deliberate. The sleep-wake switch (VLPO, orexin, SCN) is one
circuit spanning this region and brainstem; the cortisol endpoint of
the stress axis is read by limbic and frontal cortex. Each shared
claim below is listed in those regions' ``_EXPORTS`` as well.

════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION: THE HPA STRESS CASCADE
════════════════════════════════════════════════════════════════════════

The HPA axis is a three-stage cascade with transport delays and
negative feedback from the end hormone. Delay-differential models of
this loop (Lenbury & Pornsawad, 2005) are the physiological reference
for its structure; the coefficients here are this module's own, not
fitted to that paper. Receptor physiology is from de Kloet, Joëls &
Holsboer (2005). With CRH ``x``, ACTH ``a``, and cortisol ``c``, each
clamped to [0, 1], and stress indicator ``S(t)`` in {0, 1}, the
implemented dynamics are:

    CRH       dx/dt = S·r_x − d_x·(1 − S) − k_x·c
    ACTH      da/dt = [x(t − τ_a) > θ]·r_a·x(t − τ_a)
                      − d_a·[x(t − τ_a) ≤ θ] − k_a·c
    Cortisol  dc/dt = [a(t − τ_c) > θ]·r_c·a(t − τ_c)
                      − d_c·[a(t − τ_c) ≤ θ]

with delays τ_a = 300 s (pituitary, about 5 minutes) and τ_c = 600 s
(adrenal, about 10 minutes), gate threshold θ = ``hpa_hormone_gate``,
and the feedback gains ``k`` scaled by 0.01 as implemented. The delays
make the response lag its trigger: in simulation, cortisol first rises
about 16 minutes after onset and peaks about 27 minutes after it, the
range measured for human cortisol responses to psychosocial stress
(peak ~38 min in the TSST meta-analysis, Frontiers in Neuroendocrinology
2022). Every level is clamped to [0, 1], so the state stays in the unit
cube, and the feedback term returns it to baseline once the stressor
ends.

════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION: ALLOSTASIS AND CIRCADIAN TIMING
════════════════════════════════════════════════════════════════════════

Set-points move with load (Sterling, 2012): the cumulative cost of
repeated HPA activation is allostatic load (McEwen, 1998). The tracker
in ``limbic_system`` takes the median of three domains (substrate load,
cortisol, body stress), accumulates the excess of that median above an
accumulation threshold, and drains it below a recovery threshold. The
median makes one isolated spike unable to create load on its own.

Circadian timing is the two-process model (Borbély, 1982): a homeostatic
sleep pressure ``S`` that rises in wake and falls in sleep, and a
circadian drive ``C``. Sleep onset occurs when ``S`` reaches an upper
threshold, and waking when it falls to a lower one, with ``C`` shifting
those thresholds. The implemented sleep switch in ``brainstem.sleep`` maps
``S`` to adenosine (``AUTO_SLEEP_ADENOSINE``) and ``C`` to melatonin,
the SCN's output: melatonin lowers the sleep threshold at circadian
night. The wake rescue applies only while melatonin is below
``WAKE_RESCUE_MELATONIN``; above it, the circadian drive takes precedence.

════════════════════════════════════════════════════════════════════════
CLAIMS
════════════════════════════════════════════════════════════════════════

Stress and allostasis (co-claimed with ``limbic_system``): the HPA
cascade and the load tracker.

Neuroendocrine and neuromodulator outputs (co-claimed with
``neurochemical/``, whose files stay at top level because the hormone
systems are distributed): CRH, cortisol, vasopressin, oxytocin,
orexin, histamine, melatonin, endorphin.

The sleep-wake circuit's hypothalamic half (co-claimed with
``brainstem``, which runs the switch): the VLPO/orexin/SCN side of
the flip-flop and the orexin-timescale sleep inertia that follows
the wake transition. The VLPO's sleep-promoting output is GABAergic
(with galanin); the GABA chemical model itself stays shared in
``neurochemical/`` because inhibitory interneurons are ubiquitous —
only the region-specific nuclei are claimed.
"""

from __future__ import annotations

from _views import view_getattr

_EXPORTS: dict[str, str] = {
    "SleepStage": "thalamus.brain_waves",
    "SleepCycleTracker": "sleep.architecture",
    "SleepMixin": "brainstem.sleep",
    "SleepStageWorker": "brainstem.sleep",
    "AUTO_SLEEP_ADENOSINE": "brainstem.thresholds",
    # The orexin-stabilization lag — the hypothalamic peptide holds
    # the monoaminergic pole of the flip-flop, so the grogginess after
    # the wake flip decays on orexin's timescale (co-claimed with
    # brainstem, which runs the switch)
    "SleepInertia": "sleep.events",
    # The HPA stress cascade — CRH -> ACTH -> cortisol with delays
    "HPAAxis": "limbic_system.emotional_regulator",
    "HPAState": "limbic_system.emotional_regulator",
    # Allostatic load — the cumulative cost of HPA activation
    "AllostaticLoadTracker": "limbic_system.emotional_regulator",
    "AllostaticState": "limbic_system.emotional_regulator",
    # PVN neuropeptide — the first stage of the stress axis
    "CRH": "neurochemical.crh",
    # Adrenal endpoint of the axis, which the hypothalamus regulates by feedback
    "Cortisol": "neurochemical.cortisol",
    # Magnocellular (SON, PVN) neuropeptides — osmotic balance, bonding
    "Vasopressin": "neurochemical.vasopressin",
    "Oxytocin": "neurochemical.oxytocin",
    # Arcuate/pituitary opioid peptide — pain relief and reward
    "Endorphin": "neurochemical.endorphin",
    # Lateral hypothalamus — orexin/hypocretin wake stabilization
    "Orexin": "neurochemical.orexin",
    # Tuberomammillary nucleus — histamine cortical arousal
    "Histamine": "neurochemical.histamine",
    # SCN -> pineal circadian output
    "Melatonin": "neurochemical.melatonin",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
