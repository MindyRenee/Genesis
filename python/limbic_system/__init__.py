"""Limbic system — emotion, motivation, and the self.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The limbic system is the brain's emotional and motivational core. It
sits beneath the cerebral cortex and wraps around the brainstem,
forming a ring (limbus = border). It translates raw neurochemical
state into emotional experience, drives approach/avoidance behavior,
and forms the substrate of the self.

Key structures:
    - Amygdala — emotional evaluation, fear conditioning, reward
      salience. Tags sensory input with emotional valence.
    - Hippocampus — memory formation and retrieval (anatomically
      limbic; the memory systems are documented in ``temporal_lobe``,
      which claims them).
    - Cingulate cortex — emotion-cognition integration, conflict
      monitoring, pain processing.
    - Insular cortex (insula) — interoception, body-state awareness,
      the felt sense of the body.
    - Hypothalamus — homeostasis, neurochemical control, the HPA axis.
    - Parahippocampal gyrus — memory integration.
    - Entorhinal cortex — gateway to hippocampus, grid cells.

Key functions:
    - Emotional evaluation (amygdala tags input with valence)
    - Interoception (insula reads body state)
    - Homeostatic regulation (hypothalamus maintains setpoints)
    - Emotional memory (amygdala-hippocampus interaction)
    - Self-awareness (proto-self, core self — Damasio)
    - Motivation and reward (amygdala-ventral striatum)
    - Emotional regulation (prefrontal-limbic control loop)
    - Fear conditioning (amygdala)
    - Affective decision-making (somatic marker hypothesis)

Pipeline:

    [Sensory input] (from all subsystems)
        |
        v
    [Amygdala] (emotion.py)
        |   Tags input with emotional valence
        |   Fast, pre-cognitive evaluation
        |   Produces: EmotionalState (category, valence, arousal)
        v
    [Insula] (self/damasio.py — ProtoSelf)
        |   Reads body/neurochemical state (interoception)
        |   Produces: ProtoSelfState (arousal, valence, tone, plasticity)
        v
    [Cingulate cortex] (self/reflection.py — ErrorMonitor, on FrontalLobe)
        |   Integrates emotion with cognition
        |   Conflict monitoring, error detection
        |   Produces: caution level, error signals
        v
    [Prefrontal cortex] (emotional_regulator.py)
        |   Top-down regulation of emotion
        |   Veto power over emotional responses
        |   Produces: regulated emotional state

    -- Emotional memory loop --
    [Amygdala] <-> [Hippocampus]
        |   Amygdala tags memories with emotional valence
        |   Hippocampus stores the episodic memory
        |   Emotional memories are preferentially consolidated


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
════════════════════════════════════════════════════════════════════════

Amygdala — Emotional Evaluation
---------------------------------

The amygdala evaluates sensory input for emotional significance.
It computes a fast, pre-cognitive valence and arousal signal that
tags the input for downstream processing.

    valence = f(neurochemistry)
        = w_dopamine * DA + w_serotonin * 5HT - w_cortisol * CORT
    arousal = g(neurochemistry)
        = w_norepi * NE + w_acetylcholine * ACh - w_gaba * GABA

    category = argmax_c  P(c | valence, arousal, tone, plasticity)

    where:
        DA, 5HT, CORT  = dopamine, serotonin, cortisol levels
        NE, ACh, GABA  = norepinephrine, acetylcholine, GABA levels
        w_*            = weights (learned or fixed)
        valence        = emotional valence (-1..1)
        arousal        = arousal level (0..1)
        category       = EmotionCategory (structural ID)

The amygdala's evaluation is fast and pre-cognitive — it happens
before the cortex has fully processed the input. This is the
"low road" (LeDoux, 1996): relay -> amygdala directly, bypassing
the cortex. The "high road" (relay -> cortex -> amygdala) provides
a slower, more refined evaluation.

In Genesis: emotion.py implements this. EmotionalState is computed
from neurochemistry via assess_emotion(), which maps the
neurochemical vector to EmotionCategory, CognitiveMode, and
CauseCategory. The mapping uses threshold-based rules (not learned
weights), but the functional logic is the same: neurochemistry
determines emotional state.


Insula — Interoception and Proto-Self
---------------------------------------

The insular cortex reads the body's internal state (interoception)
and creates a non-cognitive representation of "how the body feels."
This is Damasio's proto-self — the background hum of existence.

    proto_self = {
        arousal:           f(neurochemistry),
        valence:           g(neurochemistry),
        tone:              h(neurochemistry),
        plasticity:        p(neurochemistry),
        homeostatic_balance: 1 - |deviation_from_setpoint|,
        chemicals:         {DA: ..., 5HT: ..., CORT: ..., ...},
    }

    where:
        arousal, valence, tone, plasticity = derived axes from
            the 18-chemical neurochemical vector
        homeostatic_balance = how close to ideal the state is
        chemicals           = raw neurochemical levels

The proto-self is not cognitive — it's the raw interoceptive signal
that the core self monitors for changes. When the proto-self changes
(e.g., cortisol spikes), the core self registers the change as a
"feeling episode" — the moment of "something is happening to me."

In Genesis: self/damasio.py ProtoSelf and ProtoSelfState implement
this. The proto-self reads from neurochemistry (via the daemon's
shared memory) and computes the derived axes. The core self
(CoreSelf) monitors the proto-self for changes and records
significant changes as CoreSelfEpisode objects.


Hypothalamus — Homeostatic Regulation
---------------------------------------

The hypothalamus maintains homeostatic setpoints and drives the
HPA axis (stress response). It's the brain's thermostat.

    For each homeostatic variable v:
        error = v - setpoint(v)
        if |error| > threshold:
            trigger corrective neurochemical impulse

    HPA axis cascade:
        stress -> CRH (hypothalamus) -> ACTH (pituitary) ->
        cortisol (adrenal) -> feedback to hippocampus/prefrontal

    where:
        v            = current level (e.g., cortisol, glucose)
        setpoint(v)  = homeostatic setpoint for v
        error        = deviation from setpoint
        CRH          = corticotropin-releasing hormone
        ACTH         = adrenocorticotropic hormone

In Genesis: the daemon's neurochemical system implements the
homeostatic dynamics. The emotional_regulator.py
EmotionalRegulator implements the cognitive-layer homeostatic
self-regulation — monitoring neurochemistry and sending corrective
impulses (GABA for stress, ACh for drowsiness, etc.).


Emotional Regulation (Prefrontal-Limbic Control Loop)
-------------------------------------------------------

The prefrontal cortex (vmPFC) regulates the amygdala. This is the
neural basis of emotional self-control — the ability to suppress
or modulate emotional responses.

    regulated_response = raw_emotion * regulation_gain

    regulation_gain = f(prefrontal_control, current_state)

    where:
        raw_emotion      = amygdala's initial response
        regulation_gain  = [0, 1] — 0 = full suppression, 1 = no regulation
        prefrontal_control = top-down control signal from PFC

The regulation gain is state-dependent: if already stressed, the
PFC has less control (regulation gain is lower); if calm, the PFC
can fully regulate. This captures the clinical observation that
stress impairs emotional regulation.

In Genesis: emotional_regulator.py EmotionalRegulator implements
this. It has "veto power" over emotional responses — the cognition
engine suggests an emotional response, and the regulator decides
whether to allow it, dampen it, or block it. The regulation is
state-dependent (current stress level modulates the gain).


Somatic Marker Hypothesis (Damasio)
--------------------------------------

Damasio's somatic marker hypothesis proposes that emotions guide
decision-making by marking options with gut feelings. The body
(as-if body loop) signals the emotional value of each option,
biasing choice toward emotionally positive options.

    for each option a:
        somatic_marker(a) = emotional_valence(recall(a))
        value(a) = rational_value(a) + alpha * somatic_marker(a)

    best_action = argmax_a  value(a)

    where:
        somatic_marker(a) = emotional valence of recalling option a
        rational_value(a) = computed rational value
        alpha              = weight on emotional vs rational

In Genesis: self/damasio.py implements the "as-if body loop" as a
linear blend of neurochemical levels that modulates current state.
The DamasioSelfHierarchy integrates proto-self (body state) with
core self (emotional awareness) and autobiographical self
(identity) to produce the felt sense that guides behavior.


════════════════════════════════════════════════════════════════════════
BRAIN WAVES
════════════════════════════════════════════════════════════════════════

The limbic system interacts with all brain wave bands, but its
primary rhythm is theta.

Theta (4-8 Hz) — The limbic rhythm
-------------------------------------

Theta is the dominant rhythm of the limbic system, particularly
the hippocampus. Theta reflects memory processing, emotional
evaluation, and the integration of emotion with memory.

    Amygdala theta: emotional evaluation, fear conditioning
    Hippocampal theta: memory formation, retrieval
    Cingulate theta: conflict monitoring, cognitive control

Theta-gamma coupling in the limbic system is the mechanism by
which emotional memories are encoded: gamma cycles nest within
theta cycles, and the theta phase determines encoding vs retrieval.

In Genesis: the brain wave system tracks theta globally. The
limbic system's contribution to theta is through emotional
evaluation (emotion.py) and memory processing (memory/). The
sleep system's hippocampal replay occurs during theta-dominant
REM sleep.

Gamma (30-100 Hz) — Active emotional processing
--------------------------------------------------

Limbic gamma reflects active emotional processing — the amygdala
firing during emotional evaluation, the hippocampus during memory
retrieval. Gamma is coordinated with theta (theta-gamma coupling).

In Genesis: gamma is tracked globally. The limbic system's
emotional evaluation and memory retrieval contribute to gamma
power.

Alpha (8-12 Hz) — Emotional regulation
-----------------------------------------

Limbic alpha reflects emotional regulation. Alpha increases in
the amygdala during successful emotional suppression (the PFC
down-regulating the amygdala). Alpha decreases during emotional
reactivity.

In Genesis: alpha is tracked globally. The emotional regulator's
suppression of emotional responses would increase alpha in the
limbic system.


════════════════════════════════════════════════════════════════════════
ANATOMICAL BOUNDARIES
════════════════════════════════════════════════════════════════════════

The limbic system is not a cortical subsystem — it's a set of
subcortical and periallocortical structures that sit beneath and
around the cortex. It interacts with all subsystems:

    - Frontal subsystem: vmPFC regulates the amygdala (emotional
      regulation). The ACC (part of the limbic system) is in the
      frontal subsystem and does conflict monitoring.
    - Temporal subsystem: the hippocampus is anatomically in the
      temporal subsystem but functionally part of the limbic system.
      The amygdala sits in the medial temporal subsystem.
    - Parietal subsystem: the insula (interoception) borders the
      parietal and temporal subsystems.
    - Brainstem: the hypothalamus connects to the brainstem for
      autonomic control.

Limbic system modules are multi-subsystem — they interact with
cognition, memory, perception, and action. The limbic-specific
ones live in this package; the rest stay at the top level and are
referenced by this subsystem.

    (limbic_system/) emotion.py
        EmotionalState, EmotionCategory, CognitiveMode, CauseCategory
        — translates neurochemistry into emotional state. Used by
        ~30 modules across all subsystems. This is the most cross-cutting
        module after brain_waves.py.

    (limbic_system/) emotional_regulator.py
        EmotionalRegulator — self-regulation of neurochemistry.
        The prefrontal-limbic control loop. Used by mind.py,
        frontal_lobe/prefrontal.py, infrastructure/persistence.py.

    (limbic_system/) curiosity.py
        CuriosityEngine — the SEEKING drive. Intrinsic motivation
        that grows with surprise, novelty, and knowledge gaps;
        drives the questions Genesis asks itself and the topics
        the autonomous learner pursues.

    (limbic_system/) active_inference.py
        ActiveInference — the self-affect/self-state reader:
        minimizing free energy over what the user's words imply
        about its own standing. Insula-style appraisal.

    (self/) damasio.py
        DamasioSelfHierarchy — proto-self (body state/interoception),
        core self (emotional awareness), autobiographical self
        (identity). The proto-self is limbic (insula/hypothalamus),
        the core self is limbic-cortical, the autobiographical self
        is cortical (temporal-frontal).

    (self/) (the self package)
        SelfModel, SelfComposer, Introspection, etc. The
        self-model spans the limbic system (proto-self) to the
        frontal subsystem (autobiographical self, metacognition). The
        self package is multi-subsystem.

    (sleep/) (the sleep package)
        Sleep architecture, dreams, inner life. Sleep is a global
        brain state controlled by the brainstem (reticular activating
        system) and the hypothalamus (circadian rhythm). The sleep
        package is multi-subsystem — it affects all cortical areas.

    sentiment.py
        Sentiment analysis — lexicon-based sentiment with negation,
        intensifier, and diminisher handling. This is the limbic
        system's emotional evaluation of language input. The
        amygdala tags incoming text with emotional valence. Multi-
        subsystem (limbic evaluation + temporal language).

    (infrastructure/) growth_ledger.py
        GrowthLedger — legible tracking of Genesis's development.
        Persistent record of milestones. This is the limbic system's
        reward/motivation tracking — the dopamine-driven sense of
        progress and accomplishment. Multi-subsystem (limbic reward +
        frontal self-assessment).

    (infrastructure/) user_profile.py
        UserProfile — what Genesis knows about the human. A
        dedicated, persistent model of the user. This is the
        limbic system's social bonding function — the oxytocin-
        mediated attachment system that tracks the relationship.
        Multi-subsystem (limbic bonding + temporal memory + frontal
        theory of mind).

    (thalamus/) brain_waves.py
        Brain wave system — tracks theta, gamma, alpha, beta, delta.
        Referenced by all subsystems. The limbic system's primary rhythm
        is theta.

Limbic-specific modules live in this package (emotion, the
regulator, the feeling reporter, sentiment, the SEEKING-drive
curiosity engine, and the active-inference self-reader); the shared
multi-subsystem subsystems stay at the top level and are re-exported.
"""

from __future__ import annotations

from _views import view_getattr

# Limbic modules either live in this package or stay at the top level
# when they are multi-subsystem. This subsystem references them,
# documents their limbic role, and lazily re-exports them so the
# anatomy is a real connection layer (the view-package pattern):
#
#   from limbic_system import EmotionalState
_EXPORTS: dict[str, str] = {
    # Amygdala — emotional evaluation: neurochemistry -> state
    "CauseCategory": "limbic_system.emotion",
    "CognitiveMode": "limbic_system.emotion",
    "EmotionalState": "limbic_system.emotion",
    "EmotionCategory": "limbic_system.emotion",
    "assess_emotion": "limbic_system.emotion",
    # The organ itself — conditioned appetitive/aversive value fields
    # (BLA acquisition, ITC/vmPFC extinction, Pearce-Hall associability)
    "Amygdala": "limbic_system.amygdala",
    # Prefrontal-limbic control loop — self-regulation, HPA axis,
    # allostatic load (hypothalamic homeostasis)
    "AllostaticLoadTracker": "limbic_system.emotional_regulator",
    "AllostaticState": "limbic_system.emotional_regulator",
    "EmotionalRegulator": "limbic_system.emotional_regulator",
    "HPAAxis": "limbic_system.emotional_regulator",
    "HPAState": "limbic_system.emotional_regulator",
    # Interoception — the sensor and the composed felt state the
    # regulator consumes (insular sensing, co-claimed with insula)
    "InteroceptionSystem": "limbic_system.emotional_regulator",
    "InternalState": "limbic_system.emotional_regulator",
    # Insula / proto-self — the Damasio hierarchy: proto-self (body),
    # core self (feeling), autobiographical self (identity; cortical)
    "AutobiographicalSelf": "insula.damasio",
    "CoreSelf": "insula.damasio",
    "CoreSelfEpisode": "insula.damasio",
    "DamasioSelfHierarchy": "insula.damasio",
    "ProtoSelf": "insula.damasio",
    "ProtoSelfState": "insula.damasio",
    "SelfModel": "frontal_lobe.self.model",
    # Interoceptive self-reading — active inference over its own
    # neurochemical trajectory (insula: felt sense of the body);
    # SEEKING drive (curiosity) and dyadic user-affect modeling
    "ActiveInferenceReader": "limbic_system.active_inference",
    "Curiosity": "limbic_system.curiosity",
    "CuriosityType": "limbic_system.curiosity",
    "DyadicState": "limbic_system.active_inference",
    "Question": "limbic_system.curiosity",
    "SelfModelReading": "limbic_system.active_inference",
    "SelfModelState": "limbic_system.active_inference",
    "UserAffectEstimate": "limbic_system.active_inference",
    # Amygdala evaluation of language — sentiment tagging
    "analyze_sentiment": "limbic_system.sentiment",
    # The learned sentiment lexicon — valence acquired from experience
    "clear_learned_sentiment": "limbic_system.sentiment",
    "has_learned_sentiment": "limbic_system.sentiment",
    "learn_sentiment": "limbic_system.sentiment",
    "lookup_sentiment": "limbic_system.sentiment",
    # Affective self-report — composing how it feels and what it is
    # concerned about, from the emotional state (insula → expression)
    "FeelingReporter": "limbic_system.feeling_reporter",
    # The self-model's structural components — traits, the machine
    # body-substrate, the minimal self, agency detection, values,
    # self-esteem (SelfModel itself is claimed above)
    "AgencyDetector": "frontal_lobe.self.model",
    "ComputationalSubstrate": "frontal_lobe.self.model",
    "MinimalSelf": "frontal_lobe.self.model",
    "PersonalityTraits": "frontal_lobe.self.model",
    "SelfEsteem": "frontal_lobe.self.model",
    "Value": "frontal_lobe.self.model",
    # Interoceptive congruence — how well the felt body matches the
    # expressed body (somatic_marker congruence; co-claimed with
    # insula, where the interoceptive comparison is read)
    "SomaticSnapshot": "insula.somatic",
    "somatic_congruence": "insula.somatic",
    # Amygdala-modulated consolidation — emotional salience tags the
    # memory and gates its strength (McGaugh's amygdala-hippocampal
    # interaction; co-claimed with temporal_lobe, which owns the
    # episodic store)
    "EmotionalMemory": "temporal_lobe.memory.systems",
    "EmotionalMemorySystem": "temporal_lobe.memory.systems",
    # Reward/motivation tracking — the dopamine sense of progress
    "GrowthLedger": "infrastructure.growth_ledger",
    "GrowthMilestone": "infrastructure.growth_ledger",
    # Social bonding — the oxytocin-mediated attachment model
    "UserProfile": "infrastructure.user_profile",
    # The region itself — affect as one object
    "LimbicSystem": "limbic_system.limbic",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
