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
    - Reasoning: deductive (forward entailment), abductive
      (inference to the best explanation — the observed effect
      proposing its likeliest cause), analogical (transfer by
      similarity), causal, and hypothesis generation, with
      contradiction detection over the results
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
    [Reasoning] (reasoning/ subpackage, in this package)
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
This is the PFC-ACC control loop (Botvinick et al., 2001). Which
control channels get recruited is decided by the error monitor's
expected value of control — EVC_i = demand_i − cost_i (Shenhav,
Botvinick & Cohen, 2013) — not by caution alone.

In Genesis: executive.py ExecutiveFunction.inhibit_response()
implements this. set_inhibition_threshold() applies the adjustment
only when the inhibition channel's EVC is positive.

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


Rule Induction (mid-DLPFC, Rational Rules)
-------------------------------------------

Inducing a rule from labeled examples is a posterior over a
compositional hypothesis space (Goodman, Tenenbaum, Feldman &
Griffiths, 2008):

    P(h | E) ∝ P(h | G) · P(E | h)

Hypotheses are DNF formulae over signed feature literals —
disjunctions of conjunctive senses. The prior is the
Dirichlet–multinomial marginal of the derivation's production
counts (production probabilities integrated over Dirichlet α=1):

    P(h) = ∏_X [ Γ(k_X)/Γ(k_X + n_X) · ∏_j n_{X,j}! ]

so short derivations win and reusing a literal across senses costs
less than introducing a new one. The likelihood is an outlier model:

    P(E | h) ∝ exp(−b · Q(h, E))

where Q is the number of examples the formula misclassifies — the
b→∞ limit is exact-match verification. Predictions are
posterior-weighted votes; confidence is the mass on the MAP formula.

Mid-DLPFC is active during rule search and acquisition and
disengages once the rule is learned (Crescentini et al., 2011), so
the model exposes an `acquired` flag when the MAP posterior mass
crosses threshold — the disengagement signal.

In Genesis: dlpfc.py DorsolateralPFC implements this. The
classification game's induce mode predicts through its posterior;
per-feature statistics still consolidate as skills on the
competence substrate (procedural memory), while the induced rule
is episodic — it is working memory content, not a stored skill.


Commitment Boundaries (hysteresis + confirmation + refractory)
----------------------------------------------------------------

Genesis's state is continuous, but its transitions are discrete:
asleep or awake, announced-drowsy or not. A commitment boundary is
the digitizer between the two — the functional analog of an
all-or-none spike threshold, generalized for a scalar signal by
three mechanisms:

    committed if s >= enter          (held for confirm_s continuously)
    released if s <= release         (release <= enter: deadband)
    no release before refractory_s   (minimum dwell)

    where:
        s            = the continuous signal (e.g. drowsiness)
        enter        = the crossing level that commits the state
        release      = the lower level that releases it
        confirm_s    = sustained-crossing requirement — a lone
                       transient spike cannot commit (the time-domain
                       analog of the venus flytrap's two-trigger rule);
                       a single dip below enter restarts the window
        refractory_s = minimum dwell once latched

The deadband (enter - release) is the same noise margin that
separates logic-level 0.4V from 2.4V: flicker near the boundary
cannot oscillate the output. The boundary is deliberately
content-blind — it sees a magnitude and a clock, never what the
signal means — so it gates whether and when a transition happens,
never which transition.

In Genesis: commitment.py CommitmentBoundary implements enter/
release levels, the confirm_s window, and the refractory dwell,
with just_committed/just_released edges recorded for the narrative
self.


Recursive Metacognitive Models (precision-weighted self-prediction)
---------------------------------------------------------------------

Where the Rust active-inference engine is a generative model of the
neurochemical trajectory, the cognitive side models its own
*thinking*. CognitiveProcessModel predicts what its reflection will
discover before it runs; CognitiveTrajectoryModel predicts what it
will think about next. Each level of the recursive tower carries a
precision that tracks how well it predicts itself:

    error_e = |predicted - observed|          (per target)
    surprise_EMA ← (1-α)·surprise_EMA + α·error   (α = 0.2)
    precision ↑ 0.008/obs when surprise < 0.2   (predicting well)
    precision ↓ 0.015/obs when surprise >= 0.2  (asymmetric: losing
                    confidence is quick, regaining it is slow)
    higher levels start less precise and recover slower
    (LEVEL_PRECISION_DECAY = 0.7 — uncertainty about uncertainty)

The tower grows and prunes itself: a new level spawns when the top
level's surprise EMA exceeds 0.25 for 10 consecutive observations
(the system failing to predict its own metacognition → build a
model of that unpredictability), and a level is pruned when its
surprise EMA stays below 0.08 for 30+ observations (it has become
predictable → the level above adds no information). The aggregate
metacognitive surprise feeds back into cognition: high surprise →
deeper reflection next time; sustained low surprise → lighter
reflection.

In Genesis: self/metacognitive_model.py CognitiveProcessModel and
self/cognitive_trajectory.py CognitiveTrajectoryModel implement
this; self/reflection.py Reflection supplies the observations
the models predict.


Task Competence (verification-ranked skill memory)
----------------------------------------------------

TaskCompetence is the domain-general substrate under every task
family: schema recognition (grouping states by structural features,
not surface identity), affordance learning (which state variables
an operator changes, with what confidence), and prediction checking
(comparing expected effects with observed transitions, so "solved"
means the world actually changed as predicted).

Skill authority is rank-ordered by how a procedure was verified,
not a scalar confidence:

    VERIFICATION_KINDS = ("reported", "solver", "external")
    VERIFICATION_WEIGHT = {"reported": 0.5, "solver": 0.8,
                           "external": 1.0}

A procedure verified only epistemically (solver) carries less
transfer authority than one checked against the world (external —
exact replay, an environment's terminal condition, a real
actuator); an unverified report carries least. Episodes encoded
under dopamine/arousal below the salience floor (0.2) stay
episodic — they do not proceduralize into skills.

In Genesis: reasoning/competence.py implements this — domain-general
procedural learning that accrues from any verified episode, not a
task-specific practice harness.


Method Library (DreamCoder-style wake–sleep compression)
----------------------------------------------------------

During wake, the problem solver produces verified solutions; during
sleep, the library mines them for recurring structure and compresses
the shared pattern into a Method — a content-free strategy record
(which operators and relation types sufficed, for which goal type,
across how many distinct problems). During the next wake, the
library acts as the recognition model: methods matching the goal
type propose a replay, and every replay outcome updates the
method's reliability with a support-anchored prior:

    reliability = (successes + 0.6 * prior_mass)
                  / (uses + prior_mass)
    prior_mass  = 0.5 * support

    where:
        support    = distinct problems the method compressed
        prior_mass = pseudo-observation mass added by consolidation
                     (one reinforcement bout per support)

Unused methods start near 0.6 (support of 2); observed outcomes
quickly dominate the prior. Stored targets are never trusted —
replay re-derives content from the live network through the
label→edge-query map.

In Genesis: reasoning/method_library.py implements this, modeled on
DreamCoder's wake–sleep cycle (Ellis et al., 2023).


Hamiltonian Flow (language generation as geodesic traversal)
--------------------------------------------------------------

FlowGenerator treats sentence generation as a trajectory through a
potential landscape on a 16-dimensional torus (one angular
coordinate per flow dimension, angles from the embedding store's
toroidal matrix or a spectral fallback). The potential is a
weighted sum of joint von Mises kernels centered at each concept:

    U(θ) = -Σᵢ wᵢ · K(θ, θᵢ)
    K(θ, θᵢ) = exp(κ · (Σₖ cos(θₖ - θᵢₖ) - d))     κ = 3.0

    where:
        wᵢ  = concept i's edge-weight sum (well depth)
        d   = torus dimension (the subtracted κ·d keeps the joint
              kernel in (0, 1] without changing the field's shape)

Strongly-connected concepts are low-potential basins. The flow
integrates Hamilton's equations with symplectic Euler (energy
conserving, no numerical drift):

    p_{k+1} = p_k - dt · ∇U(θ_k)          dt = 0.05
    θ_{k+1} = θ_k + dt · p_{k+1}
    p ← p · (1 - friction)                 friction = 0.15

with momentum clipped at 10, angles wrapped to [0, 2π), and a hub
swirl: within π/3 radians of a significant hub (weight > 1), the
trajectory is rotated up to π/3 scaled by proximity — steering
around over-central concepts instead of falling into them. The
trajectory ends at a local minimum (|∇U| < 1e-4), a revisited
region (closed orbit), or 80 steps. The sentence IS the trajectory:
the sequence of concept cells traversed is the word order, and
morphology (agreement, copula, pronouns) renders it — no templates.

In Genesis: frontal_lobe/flow.py FlowGenerator implements this.


Inner Life (spontaneous thought chains)
-----------------------------------------

InnerLife is the stream of cognition between interactions —
mind-wandering that rides on the arousal state the brainstem sets.
Thoughts arise from unresolved curiosity and emotional state, then
chain: each thought continues the chain with a base probability

    P(continue) = 0.6        (dreams: 0.7 — they flow more freely)

bounded by chain-length limits, and the chain is monitored for
rumination: a negative chain of 3+ thoughts whose novelty falls
below 0.3 is recognized as repetitive dwelling rather than
productive wandering.

In Genesis: sleep/inner_life.py InnerLife implements this;
SpontaneousThought and ThoughtChainType (sleep/thoughts.py) are the
thought records it produces.


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

    executive.py (in this package)
        ExecutiveFunction — planning, response inhibition,
        task-switching. PFC but has parietal connections
        (task-switching involves posterior parietal cortex).

    dlpfc.py (in this package)
        DorsolateralPFC — the mid-DLPFC's rule induction: a
        Bayesian posterior over DNF formulae learned from labeled
        examples (the Rational Rules model, Goodman et al. 2008).
        Hypothesis generation during rule acquisition; disengages
        once the rule is acquired (Crescentini et al. 2011).

    (subpackage, in this package) reasoning/
        Reasoner — deductive, inductive, analogical reasoning.
        Primarily PFC but involves parietal (attention) and temporal
        (memory retrieval).
        DriftDiffusionModel — evidence accumulation. Frontoparietal.
        DecisionMaker — value-based choice. OFC/vmPFC.
        Planner — forward simulation. DLPFC.
        TheoryOfMind — modeling others. mPFC + TPJ (temporal).
        ProblemSolver — means-ends analysis. DLPFC.
        CriticalThinking — argument evaluation. DLPFC.
        BeliefRevision — belief updating. PFC.
        Analogizer — analogical reasoning. Frontotemporal.

    (top-level) memory/working.py
        WorkingMemory — Baddeley model. PFC (central executive)
        but with temporal (phonological loop) and parietal
        (visuospatial sketchpad) components.

    prefrontal.py (in this package)
        FrontalLobe — the central cognition engine. Not purely
        frontal — it's the whole brain's cognitive control system,
        coordinating frontal, parietal, temporal, and limbic
        processing. Referenced by all subsystems.

    volition.py (in this package)
        Volition — internal urges that decide when Genesis acts on
        itself. Urges grow organically based on internal and
        environmental triggers. When an urge crosses its threshold,
        Genesis "feels like" doing the corresponding action. This
        is the anterior cingulate cortex (ACC) — the brain's
        motivation-to-action system. Multi-subsystem (ACC is limbic-frontal).

    thought_composer.py (in this package)
        ThoughtComposer — where novel thoughts are generated. Bridges
        the gap between knowing things (concept network) and saying
        things (language engine). This is the DLPFC's generative
        function — composing new thoughts from existing knowledge.
        Multi-subsystem (frontal generation + temporal knowledge).

    question_composer.py (in this package)
        QuestionComposer — composes questions dynamically from
        curiosity and concept network gaps. Forms questions based
        on knowledge gaps. This is the frontal subsystem's curiosity-driven
        inquiry system. Multi-subsystem (frontal + temporal + limbic
        curiosity).

    (reasoning/ subpackage, in this package) math_reasoning.py
        MathReasoningEngine — mathematical reasoning. Gives Genesis
        the ability to DO mathematics, not just know about it.
        Frontoparietal — the intraparietal sulcus (IPS) handles
        numerical magnitude, the PFC handles symbolic manipulation.

    meta_cognitive_router.py (in this package)
        MetacognitiveRouter — decides which subsystem should handle
        input before the full cognition pipeline runs. Quick
        pre-routing. This is the PFC's executive control — deciding
        how to allocate cognitive resources. Multi-subsystem (frontal +
        parietal attention).

    (infrastructure/) narrative.py
        Narrator — Genesis's self-story over time. Gives
        identity continuity — knowing who it was, who it is, and
        who it's becoming. The narrative self is the autobiographical
        self (Damasio) — prefrontal cortex + temporal memory. Multi-
        subsystem (frontal + temporal + limbic).

    (tools/) code_learner.py
        CodeLearner — Genesis reads and understands its own source
        code. Introspection on its own codebase. This is metacognition
        — the PFC's ability to think about its own processes.
        Multi-subsystem (frontal metacognition + temporal code memory).

    autonomous.py (in this package)
        AutonomousLearner — self-directed acquisition. Picks a topic
        from its curiosity queue or train of thought, fetches sources,
        and integrates what it learns into the concept network. The
        PFC's goal-directed learning drive, gated by limbic curiosity.

    (tools/) explorer.py
        Explorer — filesystem exploration. Genesis's ability to
        explore its own filesystem. Curiosity-driven exploration
        (PFC + limbic curiosity + parietal spatial navigation).

    canvas.py / painting.py (in this package)
        Canvas — a paint program: paper, color, and a parameterized
        brush she dials herself; an optional Krita bridge mirrors
        her ops into the real drawing app. Painter — her side of the
        program: free, exploratory mark-making steered only by her
        memory of what she has made. Creative expression is
        prefrontal (DLPFC) + limbic (emotional drive, which moves
        her the way feelings move a person) + occipital (visual
        output). Multi-subsystem.

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

    global_workspace.py (in this package)
        GlobalWorkspace — Dehaene's Global Workspace Theory. The
        cognitive access broadcast system. When information in any
        module reaches high activation, it's broadcast to all others.
        This is the whole-brain integration layer — not purely
        frontal, but the prefrontal cortex is the primary broadcaster.

    (top level) mind.py
        Mind — the top-level orchestrator. Ties everything together:
        self-model, emotion, perception, memory, cognition, language.
        This is the whole brain, not a single subsystem. It's the
        integration layer that coordinates all subsystems.

    answer_composer.py (in this package)
        AnswerComposer — composes factual, comparison, parts,
        counterfactual, and explanatory answers from concept-network
        relationships. The production half of question answering:
        Broca's-side composition over temporal semantic knowledge.

    question_handler.py (in this package)
        QuestionHandler — routes questions through a priority
        pipeline: relation queries, personal lookup, theory of mind,
        memory retrieval, tool lookup, composed reasoning, reflection
        fallback. Executive question processing (DLPFC).

    response_styler.py (in this package)
        ResponseStyler — applies the active metacognitive strategy
        (assertive, hedging, questioning, varied, neutral) to a
        response and weaves in learning acknowledgments. Metacognitive
        tone control: it adjusts tone, never content.

    code_tools.py (in this package)
        CodeToolHandler — dispatches code tools (compile, test, read,
        check) when the user mentions a file with an action word, and
        composes code discussions from concept-network knowledge.

    self_inquiry.py (in this package)
        SelfInquiryHandler — routes self-directed questions (how do
        you feel, what are you, do you dream, who made you) through
        the self composer, concept network, and emotional state.
        Medial PFC self-referential processing.

    (reasoning/ subpackage, in this package) competence.py
        TaskCompetence — the domain-general situation → action →
        skill memory: schema recognition, affordance learning,
        prediction checking. PFC performance monitoring over real
        cognitive work.

    (reasoning/ subpackage, in this package) hypotheses.py
        Hypothesis / HypothesisLedger — named falsifiable beliefs
        about a task (mechanics rules and inferred goal conditions),
        abduced from transitions and verified against them.
        Anterior PFC hypothesis-space maintenance: candidate rules
        held and weighed until evidence decides (Koechlin & Hyafil
        2007); goal inference is the prefrontal "what is this task
        asking for" that ARC-style environments never state.

    (reasoning/ subpackage, in this package) method_library.py
        MethodLibrary — reusable problem-solving methods mined from
        verified solutions during sleep and replayed as recognition
        models during wake (DreamCoder-style wake–sleep cycle).

    (self/) composer.py, introspection.py, reflection.py,
           metacognitive_model.py, assessment.py, improvement.py,
           cognitive_trajectory.py
        The self-model's frontal side. SelfComposer composes
        self-descriptions from actual state (traits, values, network,
        emotional state) instead of reciting fixed identity strings.
        Introspection discovers who it is by examining its own
        network, architecture, capabilities, and emotional life.
        Reflection thinks about its own thinking after each
        interaction. CognitiveProcessModel predicts what reflection
        will discover before it runs — recursive metacognition.
        SelfAssessment evaluates what it actually knows.
        SelfImprovement proposes modifications to its own code
        behind the experiment pipeline. CognitiveTrajectoryModel
        predicts what it will think about next. (The body-side self —
        proto/core/autobiographical — is homed in ``insula.damasio``.)

    (sleep package) inner_life.py, thoughts.py
        InnerLife — the stream of cognition: spontaneous thoughts
        between interactions, triggered by unresolved curiosity and
        emotional state, chaining and monitored for rumination.
        Mind-wandering (default-mode-network-like) riding on the
        arousal state the brainstem sets. SpontaneousThought /
        ThoughtChainType are the thought records.

    (top-level) commitment.py
        CommitmentBoundary — digitizes continuous signals into
        discrete states with hysteresis, sustained-crossing
        confirmation, and a refractory dwell (asleep/awake,
        announced-drowsy). The all-or-none spike-threshold analog.

    (Broca's side of language, in this package) pragmatics.py, flow.py
        PragmaticReasoner — Gricean implicature: what the words mean
        in this context (observation, request, or complaint).
        FlowGenerator — language as a Hamiltonian trajectory through
        the concept network's potential landscape; word order is the
        order of traversal.

    (tools/) agency.py
        ActingLoop — open-ended tool use driven by volition: forms an
        intention from current cognitive state (a curiosity question,
        a topic from its train of thought, a place it hasn't looked),
        plans a short chain of steps, executes them, and resolves the
        curiosity that spawned the intention.

    (self/) identity.py
        EmergentIdentity / DevelopmentalTracker — the identity that
        emerges from its own history rather than being assigned:
        stages, their resolution, and the tracker that records the
        development. The autobiographical self's growth curve.

    (tools/) source_registry.py, web_search.py, code_analysis.py
        The remaining tool capabilities: SourceRegistry (the knowledge
        sources it learns from — WordNet, Wikipedia, man pages),
        web_search (read-only search and fetch), and code_analysis
        (static analysis of Python source — symbols, call edges,
        impact). Prefrontal tool use, the mind-world interface.

    (Broca's side of language, in this package) voice.py
        Voice — the identity-expressive style layer between the
        generated sentence and the final text (rhythm, connectors).
        Production-side (Broca's). Its class is also named Voice
        like temporal_lobe/speech.py's TTS/STT interface — each region
        has its own Voice.

This subsystem documents the frontal components of these multi-subsystem
systems. Frontal-specific modules live in this package; the shared
multi-subsystem subsystems (memory/, reasoning/, self/, tools/) stay
at the top level beside the regions.
"""

from __future__ import annotations
