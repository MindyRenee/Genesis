"""Insula — interoception: sensing the body's internal state.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The insular cortex lies folded inside the lateral sulcus, between the
temporal and frontal opercula. It receives the visceral afferents
relayed through several thalamic nuclei, including VMpo and VMb in
primate lamina-I and vagal pathways, rather than a single VPM channel.
It builds representations of heartbeat, breath, temperature, pain,
hunger, and fatigue from these signals. Craig
(2009) describes it as a posterior-to-anterior hierarchy. The posterior
insula represents the body's current state; the anterior insula
integrates that state with salience and emotion and is the hub of the
salience network with the anterior cingulate.

The sensing machinery is in the brainstem (the autonomic heartbeat that
samples the body) and in the limbic system (the interoceptive layer that
turns samples into stress). This region is the anatomical place where
those signals become a felt state; it co-claims that machinery rather
than duplicating it.

════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION: INTEROCEPTIVE PREDICTIVE CODING
════════════════════════════════════════════════════════════════════════

Interoceptive predictive coding (Seth, Suzuki & Critchley, 2011; Seth,
2013) treats the insula as a comparator between a top-down prediction
of the body state and the bottom-up afferent signal. For an afferent
reading ``s`` and a prediction ``ŝ``, the prediction error is

    ε = s − ŝ

and the posterior estimate is updated by precision-weighted errors,

    μ ← μ + κ · π · ε,      π = 1 / σ²

so a channel that is reliable (low variance σ²) moves the felt state more
than a noisy one. Allen and Friston (2018) extend this with a hierarchy of
such comparators, which matches the insula's posterior-to-anterior
organisation (Royal Society Open Science 9(6), 2022, "An insula
hierarchical network architecture for active interoceptive inference":
gINS is the lowest level, dINS the middle, aINS the top, with forward
prediction errors and backward predictions between them).

The implemented body-state layer follows the same structure: each reading
is a sample with a freshness window, and a reading that is older than
the window stops contributing distress while its last value is retained.
This is a freshness heuristic, not an implemented Bayesian precision update.

Felt stress is then the salience arbitration over the two sensing layers,

    stress = min(1, max(stress_process, stress_body))

a winner-take-all salience signal. It is a max rather than a precision-
weighted mean on purpose: a saturated channel (for example, 90% CPU) must
reach the felt state even when the other channel is quiet, because the
software needs to preserve a saturated distress channel. This max rule
is an engineering choice, not a measured anterior-insula transfer function.

════════════════════════════════════════════════════════════════════════
CLAIMS
════════════════════════════════════════════════════════════════════════

Interoceptive sensing and the felt state (co-claimed with
``limbic_system``, which owns the regulator that consumes them): the
interoception system and the composed internal state.

Autonomic sampling (co-claimed with ``brainstem``, which owns the
heartbeat loop): the heartbeat mixin that samples the body each cycle.

The interoceptive body map (co-claimed with ``self``, where the
proto-self is defined): Damasio's proto-self, the insula-based map of
the body that the self-model is built on — and the somatic-marker
congruence check that compares the felt body against the expressed
body (co-claimed with ``limbic_system``).
"""

from __future__ import annotations

from _views import view_getattr

_EXPORTS: dict[str, str] = {
    # The interoceptive sensor — process and body layers
    "InteroceptionSystem": "limbic_system.emotional_regulator",
    "InternalState": "limbic_system.emotional_regulator",
    "SystemMonitor": "parietal_lobe.system_monitor",
    "SystemSnapshot": "parietal_lobe.system_monitor",
    # The autonomic sampling loop that reads the body each cycle
    "HeartbeatMixin": "brainstem.heartbeat",
    # The body map the self is built on
    "ProtoSelf": "insula.damasio",
    "ProtoSelfState": "insula.damasio",
    # Somatic-marker congruence — how well the felt body matches the
    # expressed body: an interoceptive comparison, read in the
    # insular cortex (co-claimed with limbic_system)
    "SomaticSnapshot": "insula.somatic",
    "somatic_congruence": "insula.somatic",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
