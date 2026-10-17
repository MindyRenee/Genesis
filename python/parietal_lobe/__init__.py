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
    [MT/V5] Motion processing (motion.py)
        |   Direction and speed selectivity — per-cell normal flow
        |   from consecutive frames (constant-brightness least squares)
        |   Produces: MotionField + grouped MotionRegions
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

In Genesis: saliency.py SaliencyMap implements the topographic
map — bottom-up feature maps (intensity centre-surround, color
double-opponency, ...) combined into one spatial priority map, with
SaliencyMap.select() choosing the next attended location by
winner-take-all and inhibition-of-return returning an
AttendedLocation. This replaces the scalar scene-salience the
vision pipeline used before.


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


Motion (MT/V5 — constant-brightness flow)
-------------------------------------------

motion.py takes two consecutive luminance frames and produces a
motion field: per-cell direction and speed, plus the moving regions
those cells group into. The estimator is the constant-brightness
approximation of optical flow:

    Ix * u + Iy * v + It = 0

    where:
        Ix, Iy = spatial luminance gradients
        It     = temporal luminance gradient
        (u, v) = the cell's flow velocity

Solved per coarse cell by least squares over its pixels (a single
equation per pixel cannot determine both components; aggregating
over a cell's pixel set makes the system over-determined and
solvable). This is *normal flow* — the component of motion visible
under the aperture constraint — which is exactly the input LIP's
saliency map and the sensorimotor areas need: where things move and
how fast, not what they are.

In Genesis: occipital_lobe/motion.py MotionProcessor implements this
occipitotemporal computation, producing MotionField (per-cell flow)
and grouped MotionRegions projected to parietal attention.


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
this — focused attention boosts gamma (frontal_lobe/prefrontal.py:
"Top-down drive: focused attention boosts gamma").


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
(prefrontal cortex) and lives under ``frontal_lobe/``. The attention
system stays in the parietal subsystem because its primary neural
substrate is parietal (IPS), even though it receives strong frontal
input.

The scalar scene salience inside the vision pipeline is parietal in
function but stays in ``perception/`` for practical reasons (it needs
direct access to the visual frame); the topographic map it used to
approximate is now implemented here in ``parietal_lobe/saliency.py``.


════════════════════════════════════════════════════════════════════════
MODULE ORGANIZATION
════════════════════════════════════════════════════════════════════════

The attention system (``parietal_lobe.attention``) is a
parietal-prefrontal system — it belongs to both the parietal
subsystem (selective/sustained/divided attention, IPS) and the
frontal subsystem (executive attention, DLPFC/ACC). This subsystem
imports and re-exports it; the frontal subsystem does the same.

    (parietal_lobe/) attention.py
        AttentionSystem — selective, sustained, executive, and
        divided attention. Four attention types grounded in
        parietal-prefrontal networks. Integrates with the concept
        network (amplifies/suppresses concept activation).

    (parietal_lobe/) system_monitor.py
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

    (thalamus/) brain_waves.py
        Brain wave system — tracks beta (the parietal rhythm),
        alpha (attentional gating), gamma (attentional binding).
        Referenced by every subsystem.

    (Future modules to be added here as they are implemented:)
    spatial_wm.py  Bump attractor for spatial working memory.
    coords.py      Retinotopic-to-allocentric coordinate
                   transformation.

    Implemented here (see the saliency and motion sections above):
    saliency.py    LIP topographic saliency map — feature-map
                   combination, winner-take-all selection,
                   inhibition-of-return.
    motion.py      MT/V5 motion processing — per-cell normal flow
                   via the constant-brightness approximation,
                   moving-region grouping.

Top-level modules newly claimed by this subsystem:

    (frontal_lobe/reasoning/) math_reasoning.py
        MathReasoningEngine — numerical magnitude and symbolic
        manipulation. The intraparietal sulcus (IPS) handles
        magnitude; the PFC handles symbolic manipulation
        (frontoparietal — also re-exported by frontal_lobe).

    (self/) model.py — the minimal self and the agency comparator
        MinimalSelf, AgencyDetector. The temporoparietal junction is
        the canonical seat of self-location, first-person
        visuospatial perspective, and ownership (Blanke & Metzinger;
        out-of-body illusions from rTPJ disruption). The agency
        detector is the TPJ's forward-model comparator: expected
        consequences of its own acts matched against actual input —
        mismatch marks the event as caused by another (Decety et
        al.; rTPJ TMS impairs self/other attribution). Co-claimed
        with limbic_system, which owns the Damasio stack.

    belief.py (formerly world/belief.py) — posteriors about who is out there
        PresenceBelief, Beta. Mentalizing over the social world:
        Beta posteriors on external presences' responsiveness and
        mood, updated by evidence. Right TPJ is the selective
        mentalizing node — belief attribution, not mere person
        detection (Saxe & Kanwisher 2003; Schurz et al.
        meta-analyses). Co-claimed with frontal_lobe (mPFC, the
        other mentalizing node).

    (sleep/) inner_life.py — the posterior DMN node
        InnerLife. The default-mode network is not only frontal:
        its posterior hub is posterior cingulate/precuneus
        (parietal midline) and its lateral nodes are angular
        gyrus/TPJ — the regions active in mind-wandering and
        self-projection (Buckner et al. 2008; Andrews-Hanna). The
        co-claim records the parietal half of the frontoparietal
        loop InnerLife models.

    (parietal_lobe/) situation_model.py
        SituationModel — the episodic state of the world the current
        discourse describes ("Mary is in the bathroom", "Sandra is
        carrying the milk"), updated every turn by comprehension
        propositions and consulted before semantic memory or external
        lookup. Event comprehension is parietal-temporal; the model
        of where things are right now is parietal (the world_model's
        discourse sibling).

    (this package) grid.py, scene.py, grounding.py, delta.py,
                  transforms.py
        The dorsal spatial substrate: Grid (the 2-D cell structure
        imagery and embodiment share), perceive()/Scene (object
        segmentation + spatial relations), grounding of perceived
        scenes into checkable relations, PairDelta/TaskDelta
        (perceive_pair/perceive_task — before/after structure
        deltas), and the transformation operators (gravity
        direction, recolor maps, shifts, fills) that spatial
        imagery composes.
"""

from __future__ import annotations

from _views import view_getattr

# The parietal subsystem's modules are multi-subsystem and stay at the top
# level. This subsystem lazily re-exports them so the anatomy is a real
# connection layer, not just documentation:
#
#   from parietal_lobe import AttentionSystem
_EXPORTS: dict[str, str] = {
    # Selective/sustained/divided attention — intraparietal sulcus
    "AttentionFocus": "parietal_lobe.attention",
    "AttentionSystem": "parietal_lobe.attention",
    "AttentionType": "parietal_lobe.attention",
    # LIP — the topographic saliency map and winner-take-all
    # selection with inhibition-of-return
    "AttendedLocation": "parietal_lobe.saliency",
    "SaliencyMap": "parietal_lobe.saliency",
    "SaliencyResult": "parietal_lobe.saliency",
    # MT/V5 — per-cell normal flow and the moving regions it groups
    "MotionCell": "occipital_lobe.motion",
    "MotionField": "occipital_lobe.motion",
    "MotionProcessor": "occipital_lobe.motion",
    "MotionRegion": "occipital_lobe.motion",
    "PhonologicalLoop": "temporal_lobe.memory.working",
    "VisuoSpatialSketchpad": "temporal_lobe.memory.working",
    "WorkingMemory": "temporal_lobe.memory.working",
    "TheoryOfMind": "frontal_lobe.reasoning.theory_of_mind",
    # Multisensory integration — the superior parietal lobule binds
    # the modalities into a unified percept
    "IntegratedPerception": "occipital_lobe.perception.core",
    "MultisensoryInput": "occipital_lobe.perception.core",
    "MultisensoryIntegrator": "occipital_lobe.perception.core",
    "Perception": "occipital_lobe.perception.core",
    # Interoception of the machine environment — the parietal subsystem's
    # spatial/body awareness extended to the hardware it lives on
    "SystemMonitor": "parietal_lobe.system_monitor",
    "SystemSnapshot": "parietal_lobe.system_monitor",
    # The dorsal spatial substrate — the 2-D cell grid, object
    # perception into scenes, spatial relations, before/after
    # structure deltas, transformation operators, and grounding of
    # scenes into checkable relations
    "Grid": "parietal_lobe.grid",
    "Scene": "parietal_lobe.scene",
    "ObjectRelation": "parietal_lobe.scene",
    "PerceivedObject": "parietal_lobe.scene",
    "SpatialRelationKind": "parietal_lobe.scene",
    "perceive": "parietal_lobe.scene",
    "PairDelta": "parietal_lobe.delta",
    "perceive_pair": "parietal_lobe.delta",
    "perceive_task": "parietal_lobe.delta",
    "TaskDelta": "parietal_lobe.delta",
    "ground_scene": "parietal_lobe.grounding",
    "scene_facts": "parietal_lobe.grounding",
    "scene_relations_for": "parietal_lobe.grounding",
    "Transform": "parietal_lobe.transforms",
    # Numerical cognition — the intraparietal sulcus handles
    # magnitude; the PFC handles symbolic manipulation. Co-claimed
    # with frontal_lobe (frontoparietal).
    "MathResult": "frontal_lobe.reasoning.math_reasoning",
    "try_math": "frontal_lobe.reasoning.math_reasoning",
    "CalcResult": "parietal_lobe.math.arithmetic",
    "NumberSense": "parietal_lobe.math.number_sense",
    # The discourse situation model — the episodic state of the world
    # the current conversation describes (event comprehension;
    # distinct from the semantic network)
    "DialogueAct": "parietal_lobe.situation_model",
    "EntityState": "parietal_lobe.situation_model",
    "SituationEvent": "parietal_lobe.situation_model",
    "SituationModel": "parietal_lobe.situation_model",
    # Relational reasoning (IPS) and the egocentric world model
    "AbstractReasoner": "parietal_lobe.abstract_reasoning",
    "RelationalComplexity": "parietal_lobe.abstract_reasoning",
    "RelationalSchema": "parietal_lobe.abstract_reasoning",
    "RelationalSystem": "parietal_lobe.abstract_reasoning",
    "SchemaAtom": "parietal_lobe.abstract_reasoning",
    "SpatialImagery": "parietal_lobe.spatial_imagery",
    "WorldModel": "parietal_lobe.world_model",
    "WorldEntity": "parietal_lobe.world_model",
    "WorldEvent": "parietal_lobe.world_model",
    "SimState": "parietal_lobe.world_model",
    # Sensorimotor body loop — observe/act (body schema is parietal)
    "EmbodimentMixin": "parietal_lobe.embodiment",
    # The temporoparietal minimal self — self-location, first-person
    # perspective, and the ownership comparator that marks which
    # events are self-caused (co-claimed with limbic_system)
    "MinimalSelf": "frontal_lobe.self.model",
    "AgencyDetector": "frontal_lobe.self.model",
    # Mentalizing over the social world — Beta posteriors on the
    # external presences' responsiveness and mood (rTPJ belief
    # attribution; co-claimed with frontal_lobe's medial PFC)
    "PresenceBelief": "parietal_lobe.belief",
    "Beta": "parietal_lobe.belief",
    # The posterior default-mode hub — posterior cingulate/precuneus
    # and the angular gyrus are the parietal half of the mind-
    # wandering loop InnerLife models (co-claimed with frontal_lobe)
    "InnerLife": "frontal_lobe.inner_life",
    # The region itself — dorsal attention + situation model
    "ParietalLobe": "parietal_lobe.parietal",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
