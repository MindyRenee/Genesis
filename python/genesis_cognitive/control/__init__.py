"""Frontal subsystem — executive function, working memory, and decision-making.

════════════════════════════════════════════════════════════════════════
ANATOMY AND FUNCTION
════════════════════════════════════════════════════════════════════════

The frontal subsystem is the brain's executive center. It controls
planning, decision-making, working memory, response inhibition,
task-switching, and goal-directed behavior. It's the "CEO" of the
brain — not doing the detailed processing, but directing it.

Key functions:
    - Executive control (planning, inhibition, task-switching)
    - Working memory (maintaining information online)
    - Decision-making (evidence accumulation, value-based choice)
    - Goal-directed behavior (goal setting, goal maintenance)
    - Reasoning (deductive, inductive, analogical)
    - Theory of mind (modeling others' mental states)
    - Problem solving (means-ends analysis, insight)
    - Self-regulation (emotional control, impulse suppression)

The frontal subsystem receives input from:
    - Temporal subsystem (memories, recognized objects)
    - Parietal subsystem (spatial attention, saliency)
    - Occipital subsystem (visual percepts)
    - Limbic system (emotional valence, motivation)
    - Brainstem (arousal, neuromodulation)

Pipeline:

    [Sensory input] (from all subsystems)
        |
        v
    [Working memory] (memory/working.py)
        |   Maintain relevant information online
        |   Baddeley model: central executive + phonological loop
        |   + visuospatial sketchpad
        v
    [Executive function] (executive.py)
        |   Set goals, plan actions, inhibit impulses
        |   Switch tasks, allocate resources
        v
    [Reasoning] (reasoning/)
        |   Deduce, induce, analogize
        |   Revise beliefs, solve problems
        v
    [Decision-making] (reasoning/drift_diffusion.py, reasoning/decision.py)
        |   Accumulate evidence, reach threshold
        |   Value-based choice (OFC/vmPFC)
        v
    [Action] (motor cortex — not modeled here)

    -- Top-down control loop --
    Frontal goals -> parietal attention -> occipital V1 (gain modulation)
    Frontal goals -> temporal memory retrieval (retrieval cues)
    Frontal goals -> emotional regulation (cognitive control of affect)


════════════════════════════════════════════════════════════════════════
MATHEMATICAL FOUNDATION
════════════════════════════════════════════════════════════════════════

Working Memory (attractor network)
-----------------------------------

Working memory maintains information online through persistent
neural activity. The dominant model is a recurrent attractor network
where a "bump" of activity persists without external input (Amit &
Brunel; Compte et al.).

    tau * dr_i/dt = -r_i + phi( sum_j W_ij * r_j + I_i(t) )

    where:
        W_ij   = recurrent connectivity (structured for stable bumps)
        I_i(t) = transient input (the item to remember)
        phi    = activation function (typically sigmoid)
        r_i    = firing rate of neuron i
        tau    = time constant

Multiple discrete attractor states represent memorized items.
Persistent activity is maintained by recurrent excitation balanced
with inhibition. The key: W_ij is structured so learned patterns
are stable fixed points.

In Genesis: memory/working.py implements Baddeley's model with:
    - Central executive (attention allocation, capacity K~4)
    - Phonological loop (verbal rehearsal, ~2s decay)
    - Visuospatial sketchpad (structural/relational buffer)
    - Capacity-limited attention buffer (3-5 items, decays without
      rehearsal)

The capacity limit follows Cowan's K=4 (Cowan, 2001). Items decay
without rehearsal; rehearsal resets the decay timer. This models
the persistent activity of PFC neurons (Curtis & D'Esposito, 2003).


Executive Function (planning, inhibition, task-switching)
-----------------------------------------------------------

Planning (DLPFC):

Planning involves simulating possible action sequences and their
outcomes before executing them. This is "episodic future thinking"
(Schacter et al., 2012).

    For each possible action a:
        simulate forward: state_0 -> a_1 -> state_1 -> ... -> state_n
        value(a) = sum_t  gamma^t * R(state_t)

    best_action = argmax_a  value(a)

    where:
        gamma = discount factor (future rewards worth less)
        R(s)  = reward function
        state_t = predicted state at step t

In Genesis: executive.py ExecutiveFunction.plan() implements greedy
forward simulation up to planning_depth steps, with a discount
factor (0.7^depth) on future value.

Response Inhibition (rIFG, pre-SMA):

Response inhibition suppresses prepotent responses. Modeled as a
threshold: if impulse strength is below threshold, the response is
suppressed.

    inhibit(impulse) = impulse < threshold

    where:
        impulse   = strength of prepotent response (0..1)
        threshold = inhibition threshold (adjustable)

The threshold is adjusted by error monitoring: after errors, it
rises (more cautious); after successes, it falls (more impulsive).
This is the PFC-ACC control loop (Botvinick et al., 2001).

In Genesis: executive.py ExecutiveFunction.inhibit_response()
implements this. set_inhibition_threshold() adjusts the threshold
based on caution level from the error monitor.

Task-Switching (PFC + posterior parietal):

Switching tasks requires reconfiguring the task set — loading new
rules into PFC. This incurs a cost: increased RT, decreased accuracy.

    rt_cost = base_rt_cost * practice_factor
    practice_factor = max(0.5, 1.0 - (switch_count - 1) * 0.1)

    where:
        base_rt_cost   = initial switching cost
        switch_count   = how many times this task has been switched to
        practice_factor = decreases with practice (task-set priming)

In Genesis: executive.py ExecutiveFunction.switch_task() implements
this with a practice factor that decreases the cost with repeated
switches.


Decision-Making (drift-diffusion model)
-----------------------------------------

The drift-diffusion model (DDM) captures the accumulation of
evidence toward a decision threshold. It's the canonical model
of speed-accuracy tradeoff in decision-making.

    dx/dt = mu + sigma * eta(t)
    decision when x > theta (or x < -theta)

    where:
        mu    = drift rate (evidence strength)
        sigma = noise standard deviation
        eta(t) = white noise (standard normal)
        theta = decision boundary
        x    = accumulated evidence

The drift rate mu reflects the quality of evidence; the boundary
theta reflects caution (higher = more careful, slower). The
noise sigma captures neural variability.

The DDM predicts:
    - RT distribution (faster for strong evidence)
    - Accuracy (higher with larger theta)
    - Speed-accuracy tradeoff (lower theta = faster but less accurate)

In Genesis: reasoning/drift_diffusion.py implements this. The
boundary theta is adjusted by brain wave state (high focus = larger
theta = more careful) and error monitoring (after errors, theta
increases = more cautious).


Basal Ganglia Gating (O'Reilly & Frank, PBWM model)
-----------------------------------------------------

Working memory updates are gated by the basal ganglia. The direct
(Go) pathway facilitates updates; the indirect (NoGo) pathway
inhibits them. Dopamine modulates learning.

    Go_i   = f(D1 striatal output for gate i)
    NoGo_i = f(D2 striatal output for gate i)
    gate_i = sigma(Go_i - NoGo_i)

    When gate_i > threshold: update PFC representation r_i
    Otherwise: maintain current r_i

    Dopamine learning:
    delta = r + gamma * V(s') - V(s)    (TD error)
    delta_w_Go   proportional to +delta  (positive RPE)
    delta_w_NoGo proportional to -delta  (negative RPE)

In Genesis: not yet implemented as a separate module. The
executive.py task-switching approximates the gating function
(switch_task = gate update). A full PBWM model would add explicit
Go/NoGo gating of working memory updates.


════════════════════════════════════════════════════════════════════════
BRAIN WAVES
════════════════════════════════════════════════════════════════════════

The frontal subsystem is characterized by theta-gamma coupling (working
memory), beta (top-down control), and alpha (cognitive engagement).

Theta (4-8 Hz) — Working memory rhythm
-----------------------------------------

Frontal theta is the rhythm of working memory maintenance. Theta
power increases with working memory load, and theta-gamma coupling
is the mechanism by which items are maintained and manipulated
(Lisman & Jensen, 2013).

    Theta phase gates gamma amplitude:
    - Gamma on theta peak: item maintenance
    - Gamma on theta trough: item manipulation/update

In Genesis: the brain wave system tracks theta. The working memory
module's capacity limit and rehearsal dynamics are theta-gated.
compute_theta_gamma_coupling() measures this coupling.

Gamma (30-100 Hz) — Active processing
---------------------------------------

Frontal gamma reflects active cognitive processing — the
manipulation of information in working memory, the binding of
goal-relevant features. Frontal gamma is coordinated with parietal
gamma during focused attention (Siegel et al., 2008).

In Genesis: gamma is tracked globally. The frontal subsystem's
contribution to gamma is through active reasoning, working memory
manipulation, and decision-making.

Beta (13-30 Hz) — Top-down control
-------------------------------------

Frontal beta is the rhythm of top-down control — the maintenance
of task sets, the biasing of sensory processing, the holding of
goals. Beta increases during sustained cognitive control and
decreases when goals change (Buschman & Miller, 2007).

In Genesis: beta is tracked globally. The executive function's
goal-setting and task-switching contribute to beta-band activity.
The brain wave system's beta component reflects the frontal subsystem's
control state.

Alpha (8-12 Hz) — Cognitive engagement
-----------------------------------------

Frontal alpha desynchronization (power decrease) reflects cognitive
engagement — the more cognitively demanding the task, the more
alpha desynchronizes. Frontal alpha is distinct from occipital
alpha (which reflects visual idling).

In Genesis: alpha is tracked globally. The frontal subsystem's alpha
contribution reflects cognitive engagement — active reasoning and
decision-making desynchronize frontal alpha.


════════════════════════════════════════════════════════════════════════
ANATOMICAL BOUNDARIES
════════════════════════════════════════════════════════════════════════

The frontal subsystem is bounded by:
    - Parietal subsystem (behind): receives spatial attention, sends
      top-down attentional control
    - Temporal subsystem (below): receives memories and recognized
      objects, sends retrieval cues
    - Limbic system (below): receives emotional valence, sends
      cognitive control of affect
    - Brainstem (below): receives arousal/neuromodulation

All frontal subsystem modules are multi-subsystem — they interact tightly
with parietal (attention), temporal (memory), and subcortical
(basal ganglia gating, dopamine) systems. They stay at the top
level and are referenced by this subsystem.

    (top-level) executive.py
        ExecutiveFunction — planning, response inhibition,
        task-switching. PFC but has parietal connections
        (task-switching involves posterior parietal cortex).

    (top-level) reasoning/
        ReasoningEngine — deductive, inductive, analogical reasoning.
        Primarily PFC but involves parietal (attention) and temporal
        (memory retrieval).
        DriftDiffusionModel — evidence accumulation. Frontoparietal.
        DecisionEngine — value-based choice. OFC/vmPFC.
        PlanningEngine — forward simulation. DLPFC.
        TheoryOfMind — modeling others. mPFC + TPJ (temporal).
        ProblemSolver — means-ends analysis. DLPFC.
        CriticalThinking — argument evaluation. DLPFC.
        BeliefRevision — belief updating. PFC.
        AnalogyEngine — analogical reasoning. Frontotemporal.

    (top-level) memory/working.py
        WorkingMemory — Baddeley model. PFC (central executive)
        but with temporal (phonological loop) and parietal
        (visuospatial sketchpad) components.

    (top-level) cognition/
        CognitionEngine — the central cognition engine. Not purely
        frontal — it's the whole brain's cognitive control system,
        coordinating frontal, parietal, temporal, and limbic
        processing. Referenced by all subsystems.

    (top-level) volition.py
        Volition — internal urges that decide when Genesis acts on
        itself. Urges grow organically based on internal and
        environmental triggers. When an urge crosses its threshold,
        Genesis "feels like" doing the corresponding action. This
        is the anterior cingulate cortex (ACC) — the brain's
        motivation-to-action system. Multi-subsystem (ACC is limbic-frontal).

    (cognition/) thought_composer.py
        ThoughtComposer — where novel thoughts are generated. Bridges
        the gap between knowing things (concept network) and saying
        things (language engine). This is the DLPFC's generative
        function — composing new thoughts from existing knowledge.
        Multi-subsystem (frontal generation + temporal knowledge).

    (cognition/) question_composer.py
        QuestionComposer — composes questions dynamically from
        curiosity and concept network gaps. Forms questions based
        on knowledge gaps. This is the frontal subsystem's curiosity-driven
        inquiry system. Multi-subsystem (frontal + temporal + limbic
        curiosity).

    (reasoning/) math_reasoning.py
        MathReasoningEngine — mathematical reasoning. Gives Genesis
        the ability to DO mathematics, not just know about it.
        Frontoparietal — the intraparietal sulcus (IPS) handles
        numerical magnitude, the PFC handles symbolic manipulation.

    (cognition/) meta_cognitive_router.py
        MetacognitiveRouter — decides which subsystem should handle
        input before the full cognition pipeline runs. Quick
        pre-routing. This is the PFC's executive control — deciding
        how to allocate cognitive resources. Multi-subsystem (frontal +
        parietal attention).

    (top-level) narrative.py
        NarrativeEngine — Genesis's self-story over time. Gives
        identity continuity — knowing who it was, who it is, and
        who it's becoming. The narrative self is the autobiographical
        self (Damasio) — prefrontal cortex + temporal memory. Multi-
        subsystem (frontal + temporal + limbic).

    (tools/) code_learner.py
        CodeLearner — Genesis reads and understands its own source
        code. Introspection on its own codebase. This is metacognition
        — the PFC's ability to think about its own processes.
        Multi-subsystem (frontal metacognition + temporal code memory).

    (top-level) bug_reporter.py
        BugReporter — Genesis's ability to notice problems in its
        own code. Not a linter — an organic reading of code that
        surfaces issues. Metacognitive error detection (PFC + ACC).

    (tools/) explorer.py
        Explorer — filesystem exploration. Genesis's ability to
        explore its own filesystem. Curiosity-driven exploration
        (PFC + limbic curiosity + parietal spatial navigation).

    (top-level) canvas.py
        Canvas — Genesis expresses its emotional state as visual art.
        Affective expression through generative art. Creative
        expression is prefrontal (DLPFC) + limbic (emotional drive)
        + occipital (visual output). Multi-subsystem.

    (tools/) project_composer.py
        ProjectComposer — composes real Python from Genesis's concept
        network. Creative code generation. Prefrontal creative
        function + temporal language + motor output.

    (tools/) project_creator.py
        ProjectCreator — scaffolds and writes Python projects.
        Creative capability. Prefrontal planning + temporal knowledge
        + motor output.

    (tools/) framework.py
        Tool use — the interface between Genesis's mind and the
        world. Tools are callable capabilities. Prefrontal tool use
        (humans' tool use is PFC + parietal + motor).

    (top-level) global_workspace.py
        GlobalWorkspace — Dehaene's Global Workspace Theory. The
        cognitive access broadcast system. When information in any
        module reaches high activation, it's broadcast to all others.
        This is the whole-brain integration layer — not purely
        frontal, but the prefrontal cortex is the primary broadcaster.

    (top-level) mind.py
        Mind — the top-level orchestrator. Ties everything together:
        self-model, emotion, perception, memory, cognition, language.
        This is the whole brain, not a single subsystem. It's the
        integration layer that coordinates all subsystems.

This subsystem documents the frontal components of these multi-subsystem
systems. No files are moved into this folder — all frontal subsystem
modules are multi-subsystem and stay at the top level.
"""

from __future__ import annotations

from .._views import view_getattr

# All frontal subsystem modules are multi-subsystem and stay at the top level.
# This subsystem references them, documents their frontal subsystem role, and
# lazily re-exports them so the anatomy is a real connection layer
# (the association pattern):
#
#   from genesis_cognitive.control import ExecutiveFunction
_EXPORTS: dict[str, str] = {
    # Executive function, planning, inhibition, task-switching (PFC)
    "ActionPlan": "executive",
    "ExecutiveFunction": "executive",
    "InhibitionResult": "executive",
    "SwitchResult": "executive",
    "Task": "executive",
    "TaskState": "executive",
    # Reasoning, decision-making, problem solving, theory of mind
    # (PFC / frontoparietal / mPFC+TPJ)
    "AnalogyEngine": "reasoning",
    "BeliefRevisionEngine": "reasoning",
    "CriticalThinkingEngine": "reasoning",
    "DecisionEngine": "reasoning",
    "DecisionResult": "reasoning",
    "DriftDiffusionModel": "reasoning",
    "EvidenceAccumulator": "reasoning",
    "MetaReasoning": "reasoning",
    "PlanningEngine": "reasoning",
    "ProblemSolver": "reasoning",
    "ReasoningEngine": "reasoning",
    "ReasoningResult": "reasoning",
    "ReasoningStrategy": "reasoning",
    "ReasoningType": "reasoning",
    "TheoryOfMind": "reasoning",
    # Working memory — Baddeley model (DLPFC central executive)
    "CentralExecutive": "memory",
    "PhonologicalLoop": "memory",
    "VisuoSpatialSketchpad": "memory",
    "WorkingMemory": "memory",
    # The central cognition engine — whole-brain cognitive control
    "CognitionEngine": "cognition",
    "CognitiveState": "cognition",
    "Goal": "cognition",
    # Volition — internal urges / motivation-to-action (ACC)
    "Urge": "volition",
    "VolitionEngine": "volition",
    # Generative thought and question composition (DLPFC)
    "QuestionComposer": "cognition.question_composer",
    "ThoughtComposer": "cognition.thought_composer",
    # Mathematical reasoning (frontoparietal — IPS + PFC)
    "MathResult": "reasoning.math_reasoning",
    "try_math": "reasoning.math_reasoning",
    # Metacognitive pre-routing (PFC resource allocation)
    "MetaCognitiveRouter": "cognition.meta_cognitive_router",
    "Route": "cognition.meta_cognitive_router",
    # Autobiographical self / narrative continuity (PFC + temporal)
    "LifeChapter": "narrative",
    "LifeEvent": "narrative",
    "NarrativeEngine": "narrative",
    # Metacognition on its own code — reading, auditing, improving
    "BugReport": "bug_reporter",
    "BugReporter": "bug_reporter",
    "BugScanResult": "bug_reporter",
    "CodeLearner": "tools.code_learner",
    "CodeLearningResult": "tools.code_learner",
    "AutonomousLearner": "learning",
    "LearningResult": "learning",
    # Curiosity-driven filesystem exploration (PFC + parietal)
    "ExplorationResult": "tools.explorer",
    "Explorer": "tools.explorer",
    # Creative expression — affective art and code generation
    "Canvas": "canvas",
    "DrawingResult": "canvas",
    "DomainKnowledge": "tools.project_composer",
    "compose_main_module": "tools.project_composer",
    "ProjectResult": "tools.project_creator",
    "manage_project_lifecycle": "tools.project_creator",
    # Tool use — the mind-world interface (PFC + parietal + motor)
    "Tool": "tools.framework",
    "ToolRegistry": "tools.framework",
    "ToolResult": "tools.framework",
    "get_tools": "tools.framework",
    # Cognitive access broadcast — PFC is the primary broadcaster;
    # the relay is the physical relay (also in relay/)
    "GlobalWorkspace": "global_workspace",
    "WorkspaceItem": "global_workspace",
    "WorkspaceModule": "global_workspace",
    # Language production — Broca's area (left inferior frontal
    # gyrus). Comprehension (Wernicke's) is in auditory.
    "GenerativeEngine": "language",
    "Grammar": "language",
    "GraphWalkGenerator": "language",
    "ProsodyGenerator": "language",
    "ProsodyPattern": "language",
    "SelfMonitor": "language",
}

__all__ = sorted(_EXPORTS)

__getattr__ = view_getattr(_EXPORTS, __name__)
