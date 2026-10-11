"""Basal forebrain — cortical acetylcholine, attention, and encoding.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The basal forebrain lies ventral to the striatum and anterior to the
hypothalamus. It holds the major cholinergic projection to the cortex
and the hippocampus. In the nomenclature of Mesulam and colleagues,
Ch1 and Ch2 (medial septum and vertical diagonal band) supply the
hippocampal complex, Ch3 (horizontal diagonal band) supplies the
olfactory bulb, and Ch4 (nucleus basalis of Meynert) supplies the
neocortex and amygdala. The brainstem's pedunculopontine and
laterodorsal nuclei (Ch5, Ch6) project mainly to the thalamus, not the
cortex, so they belong to ``brainstem``, not here.

Acetylcholine from this region controls cortical arousal, attention,
and the encoding of new memories.

The region is also a sleep-wake structure, not only a cholinergic
one. The basal forebrain holds three intermingled populations —
wake-promoting cholinergic and parvalbumin GABAergic neurons,
glutamatergic arousal neurons, and sleep-promoting GABAergic
(somatostatin+) neurons — and it is the best-characterized site
where the homeostatic sleep-pressure signal acts.

════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION: ADENOSINE AS PROCESS S
════════════════════════════════════════════════════════════════════════

In the two-process model (Borbély, 1982; documented under
``hypothalamus/`` where the circadian half lives), homeostatic
pressure S accumulates during wake and dissipates during sleep:

    dS/dt = +r_wake   (awake)
    dS/dt = -S / τ_sleep  (asleep)

The molecular correlate of the wake side is adenosine, the ATP
byproduct. Extracellular adenosine in the basal forebrain rises with
time spent awake and falls during sleep (Porkka-Heiskanen et al.,
Science 1997); blocking its reuptake mimics sleep deprivation, and
blocking A1 receptors in the cholinergic basal forebrain blunts the
recovery-sleep response (Strecker et al., J. Neurosci. 2000;
Basheer et al., 2004). The mechanism is A1-mediated inhibition of
the region's wake-active neurons — the sleep signal literally
silences the arousal output that this same region produces.

The honest caveat, kept current as of October 2026: lesioning the
basal-forebrain cholinergic neurons abolishes the local adenosine
rise but does NOT abolish sleep drive (Blanco-Centurion et al.,
J. Neurosci. 2006), and conditional A1R knockouts retain much of
the rebound response (Bjorness et al., 2009). Adenosine is now
understood to accumulate across cortex and basal forebrain as a
distributed, largely astrocytic signal — the basal forebrain is the
canonical measurement site, not the sole locus. The claim here is
the site-of-action claim, not an exclusivity claim.

In Genesis: ``neurochemical/adenosine.py`` is the chemical model;
the daemon integrates its level with wake time. ``brainstem/
thresholds.py`` AUTO_SLEEP_ADENOSINE is the setpoint the flip-flop
reads — the threshold lives with the switch, the signal is claimed
here.


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION: ACETYLCHOLINE AS UNCERTAINTY SIGNAL
════════════════════════════════════════════════════════════════════════

Yu & Dayan (2005, Neuron, "Uncertainty, neuromodulation, and
attention") separate two kinds of uncertainty. Expected uncertainty is
the known unreliability of prior knowledge in a stable environment, and
they propose acetylcholine as its signal. Unexpected uncertainty is a
change in the environment, and they propose norepinephrine as its
signal. Acetylcholine therefore sets how strongly a learner relies on
its prior against new evidence. The implemented model in
``neurochemical.acetylcholine`` covers the attention and encoding roles
described in its own module docstring; its equations live there.

════════════════════════════════════════════════════════════════════════
CLAIMS
════════════════════════════════════════════════════════════════════════

Cortical and hippocampal acetylcholine (co-claimed with ``neurochemical/``
and ``brainstem``, whose PPT/LDT nuclei feed the thalamic branch):
the acetylcholine system.

The homeostatic sleep signal (claimed alone): adenosine, the
process-S correlate whose best-characterized site of action is this
region's wake-active neurons.
"""

from __future__ import annotations

from _views import view_getattr

_EXPORTS: dict[str, str] = {
    # Ch4 and Ch1-2 — cortical and hippocampal cholinergic projections
    "Acetylcholine": "neurochemical.acetylcholine",
    # Process S — the adenosine that accumulates here during wake and
    # A1-inhibits this region's wake-active neurons (the somnogenic
    # signal that silences its own arousal output)
    "Adenosine": "neurochemical.adenosine",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
