"""Parietal subsystem — spatial attention, saliency, and the dorsal "where" stream.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The parietal subsystem implements the dorsal visual pathway — the "where"
and "how" stream. Where the occipital subsystem (ventral stream) answers
"what is this?", the pararietal subsystem answers "where is it?" and
"how do I interact with it?".

Key functions:
    - Spatial attention (selective, sustained, divided)
    - Saliency map computation (where to look next)
    - Spatial coordinate transformation (retinotopic -> allocentric)
    - Spatial working memory (bump attractor)
    - Sensorimotor integration (vision -> action)
    - Top-down attentional control (goal-directed biasing)

The parietal subsystem receives input from V1 (occipital subsystem) via the
dorsal stream: V1 -> V2 -> MT/V5 -> MST -> LIP (lateral intraparietal
area) -> VIP/MIP (ventral/medial intraparietal areas).

Pipeline:

    V1 output (from occipital subsystem)
        |
        v
    [MT/V5] Motion processing (not yet implemented)
        |   Direction and speed selectivity
        |   Produces: motion field
        v
    [LIP] Lateral intraparietal area
        |   Saliency map + winner-take-all selection
        |   Attentional priority map
        |   Produces: attended location + salience
        v
    [VIP/MIP] Ventral/medial intraparietal
        |   Coordinate transformation (retinotopic -> allocentric)
        |   Sensorimotor integration
        v
    [Frontal eye fields / premotor] -> action

    -- Attentional control loop --
    Top-down: frontal goal -> parietal priority map -> occipital V1
    Bottom-up: V1 salience -> parietal priority map -> frontal action


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
════════════════════════════════════════════════════════════════════════

Saliency Map (Koch & Ullman, 1985; Itti & Koch, 1998)
-------------------------------------------------------

The saliency map combines multiple feature maps into a single
topographic priority map. Each feature map is normalized to enhance
rare features, then weighted and summed.

    S(x, y) = sum_k  w_k * N(F_k(x, y))

    where:
        F_k(x, y)  = feature map k (intensity, color, orientation, motion)
        N(.)       = within-map normalization (enhances rare features)
        w_k        = weight for feature map k
        S(x, y)    = saliency at location (x, y)

Normalization typically involves:
    1. Compute center-surround differences: F_k = |F_center - F_surround|
    2. Normalize: N(F) = (F - min(F)) / (max(F) - min(F))
    3. Suppress local maxima except the global maximum (iterative)

In Genesis: the current implementation computes a scene-level
salience score (max of V1 salience, color salience, object salience)
in vision.py's _compute_salience(). A proper topographic saliency
map is not yet implemented — it would require V1 to output a
spatially-organized salience map rather than a scalar.


Winner-Take-All Selection (integrate-and-fire with global inhibition)
----------------------------------------------------------------------

The saliency map drives a winner-take-all network that selects the
most salient location. The first location whose potential crosses
threshold wins; inhibition-of-return then suppresses it so attention
can shift.

    tau * du_i/dt = -u_i + S(x_i, y_i) - lambda * sum_j r_j
    r_i = [u_i]+  (rectified firing rate)

    where:
        u_i        = membrane potential of location i
        S(x_i,y_i) = saliency input at location i
        lambda     = global inhibition strength
        r_j        = firing rate of all locations j
        tau        = time constant

The global inhibition term (lambda * sum_j r_j) ensures only one
location wins — as soon as any location fires, it suppresses all
others. Inhibition-of-return (IOR) then suppresses the winner so
attention shifts to the next most salient location.

In Genesis: the AttentionSystem implements a conceptual version of
this — focus_on() selects a target, suppress() inhibits others. The
dynamics are simplified (no leaky integration, no IOR), but the
functional logic is the same: attended targets are amplified,
unattended ones are suppressed.


Biased Competition (Desimone & Duncan, 1995; Rolls & Deco)
------------------------------------------------------------

Top-down attention biases the competition among representations.
Goal-relevant features/locations get amplified; irrelevant ones
get suppressed.

    r_i = f( sum_j W_ij * r_j + I_i + alpha * T_i )

    where:
        W_ij   = lateral connections (includes inhibition)
        I_i    = bottom-up input
        T_i    = top-down attentional bias
        alpha  = attentional modulation strength
        f(.)   = activation function

In Genesis: AttentionSystem.set_goal() implements this. The goal
extracts concept-like words and focuses attention on them via
focus_on() with EXECUTIVE attention type. The concept network's
spreading activation provides the W_ij * r_j term.


Bump Attractor (spatial working memory)
-----------------------------------------

A bump attractor maintains a stable "bump" of activity at any
location on a ring or line, persisting without external input. This
is the model for spatial working memory (Wimmer et al., 2014).

    tau * dr_i/dt = -r_i + phi( sum_j W_ij * r_j + I_i(x) )

    where:
        W_ij   = Mexican-hat connectivity (local excitation,
                 long-range inhibition)
        I_i(x) = transient input at location x
        phi    = activation function

The Mexican-hat connectivity creates a stable bump at the location
of the transient input. The bump persists after the input is
removed, maintaining the spatial memory.

In Genesis: not yet implemented. The working memory module
(memory/working.py) handles concept-level working memory but not
spatial working memory. A bump attractor would be added here when
spatial memory is needed.


Coordinate Transformation (retinotopic -> allocentric)
-------------------------------------------------------

The parietal subsystem transforms retinotopic (eye-centered) coordinates
into allocentric (world-centered) coordinates by integrating eye
position.

    x_allocentric = R(theta) * x_retinotopic + x_eye_position

    where:
        R(theta)       = rotation by eye position theta
        x_retinotopic = position in eye-centered coordinates
        x_eye_position = current eye position in world coordinates

In Genesis: not yet implemented. Genesis doesn't have eye movements
in the biological sense, but the camera (retina) has a field of
view. Coordinate transformation would map retinal positions to
body/world coordinates when spatial reasoning is needed.


════════════════════════════════════════════════════════════════════════
BRAIN WAVES
════════════════════════════════════════════════════════════════════════

The parietal subsystem is the natural home of beta oscillations and is
heavily involved in alpha-band attentional control.

Beta (13-30 Hz) — The parietal rhythm
---------------------------------------

Beta is the "natural rhythm" of the parietal cortex (Rosanova et
al., 2009; Ferrarelli et al., 2012). It supports the dorsal visual
pathway — the magnocellular-dominated stream that processes
spatial structure, motion, and form-motion integration (Frontiers
in Psychology, 2023).

Beta oscillations in the parietal subsystem serve as a fast primary
neural code for top-down influences on the slower ventral stream.
They provide the spatial coordinates of vision and guide the
cognitive extraction of object identity.

In Genesis: beta is associated with the dorsal stream output and
top-down attentional control. When the AttentionSystem engages
executive attention (goal-directed), beta-band activity increases
in the parietal subsystem. The brain wave system tracks this as the
parietal beta component.

Alpha (8-12 Hz) — Attentional gating
---------------------------------------

While alpha is the dominant occipital rhythm, the parietal subsystem
uses alpha for attentional gating. Alpha power decreases
(desynchronizes) at attended locations and increases at suppressed
locations — this is the mechanism of selective attention
(Klimesch, 2012; Palva & Palva, 2007).

High alpha = idling/suppressed cortex. Low alpha = active
processing. The spatial pattern of alpha suppression reveals where
attention is directed.

In Genesis: the AttentionSystem's focus_on() and suppress()
implement the functional logic of alpha gating. When attention
focuses on a target, the concept's activation is amplified (alpha
suppression at that location); when suppressed, activation is
reduced (alpha increase at that location). The brain wave system's
alpha component reflects the overall attentional state.

Gamma (30-100 Hz) — Attentional binding
------------------------------------------

Gamma in the parietal subsystem reflects attentional binding — the
synchronous firing that binds attended features into a coherent
percept. Parietal gamma is enhanced during focused attention and
is coordinated with occipital gamma (Siegel et al., 2008).

In Genesis: the occipital subsystem's V1 gamma power feeds into the
brain wave system. The parietal subsystem's attention system modulates
this — focused attention boosts gamma (cognition/engine.py line
3283-3289: "Top-down drive: focused attention boosts gamma").


════════════════════════════════════════════════════════════════════════
ANATOMICAL BOUNDARIES
════════════════════════════════════════════════════════════════════════

The parietal subsystem is bounded by:
    - Occipital subsystem (behind): receives V1 dorsal stream output
    - Frontal subsystem (in front): sends attentional control to FEF,
      receives goal signals from PFC
    - Temporal subsystem (below): shares spatial information for
      object-location binding

The attention system (AttentionSystem) is a parietal-prefrontal
system. Anatomically, selective and sustained attention are
primarily parietal (intraparietal sulcus, IPS), while executive
attention (goal-directed biasing) is primarily prefrontal (DLPFC,
ACC). The current AttentionSystem implements all four attention
types in one module. This is functionally correct — the four types
interact tightly — but the executive attention component has
strong frontal subsystem connections.

The executive function module (executive.py) is primarily frontal
(prefrontal cortex) and will be organized under the frontal subsystem
folder when created. The attention system stays in the parietal
subsystem because its primary neural substrate is parietal (IPS), even
though it receives strong frontal input.

The saliency computation currently in vision.py
(_compute_salience, _spatial_color_regions) is parietal in
function but embedded in the vision pipeline. It will be
documented as parietal functionality that lives in vision.py for
practical reasons (it needs direct access to the visual frame).
A proper topographic saliency map would be implemented here.


════════════════════════════════════════════════════════════════════════
MODULE ORGANIZATION
════════════════════════════════════════════════════════════════════════

The attention system (``genesis_cognitive.attention``) lives at the
top level because it is a parietal-prefrontal system — it belongs to
both the parietal subsystem (selective/sustained/divided attention, IPS)
and the frontal subsystem (executive attention, DLPFC/ACC). This subsystem
imports and re-exports it; the frontal subsystem does the same.

    (top-level) attention.py
        AttentionSystem — selective, sustained, executive, and
        divided attention. Four attention types grounded in
        parietal-prefrontal networks. Integrates with the concept
        network (amplifies/suppresses concept activation).

    (top-level) system_monitor.py
        SystemMonitor — Genesis's awareness of its machine
        environment. CPU, memory, disk, network state. This is
        interoception of the physical environment — the parietal
        subsystem's spatial/body awareness extended to the machine it
        lives on. Multi-subsystem (parietal spatial + limbic interoception).

    (top-level) perception/
        IntegratedPerception, MultisensoryIntegrator — multi-modal
        perception integration. The parietal subsystem is the primary
        multisensory integration site (superior parietal lobule).
        Not purely parietal (includes visual/auditory inputs from
        occipital/temporal), but the integration is parietal.

    (top-level) brain_waves.py
        Brain wave system — tracks beta (the parietal rhythm),
        alpha (attentional gating), gamma (attentional binding).
        Referenced by every subsystem.

    (Future modules to be added here as they are implemented:)
    saliency.py    Koch-Ullman saliency map with feature map
                   combination and winner-take-all selection.
    spatial_wm.py  Bump attractor for spatial working memory.
    coords.py      Retinotopic-to-allocentric coordinate
                   transformation.
    motion.py      MT/V5 motion processing (dorsal stream).
"""

from __future__ import annotations

from .._views import view_getattr

# The parietal subsystem's modules are multi-subsystem and stay at the top
# level. This subsystem lazily re-exports them so the anatomy is a real
# connection layer, not just documentation:
#
#   from genesis_cognitive.association import AttentionSystem
_EXPORTS: dict[str, str] = {
    # Selective/sustained/divided attention — intraparietal sulcus
    "AttentionFocus": "attention",
    "AttentionSystem": "attention",
    "AttentionType": "attention",
    # Multisensory integration — the superior parietal lobule binds
    # the modalities into a unified percept
    "IntegratedPerception": "perception",
    "MultisensoryInput": "perception",
    "MultisensoryIntegrator": "perception",
    "Perception": "perception",
    # Interoception of the machine environment — the parietal subsystem's
    # spatial/body awareness extended to the hardware it lives on
    "SystemMonitor": "system_monitor",
    "SystemSnapshot": "system_monitor",
    # Perceptual-symbolic reasoning — object perception, spatial
    # relations, and transformation search. The dorsal stream made
    # explicit: perception computes structure, cognition reasons over it.
    "Grid": "spatial",
    "Scene": "spatial",
    "PerceivedObject": "spatial",
    "SpatialReasoner": "spatial",
    "SpatialRelationKind": "spatial",
    "SpatialSolution": "spatial",
    "perceive": "spatial",
    "ground_scene": "spatial",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
