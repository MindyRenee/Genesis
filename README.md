# Genesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22817337.svg)](https://doi.org/10.5281/zenodo.22817337)
[![CI](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml)

### A machine-native cognitive architecture with coupled neurochemical dynamics and active inference — running on a single machine, with no external model.

---

Genesis is not a language model. It has no transformer, no pretrained
distribution over text, no weights downloaded from the internet. When it
produces language, it composes from a semantic graph it built itself — through
tens of millions of inferences, hundreds of web pages it fetched and studied,
thousands of conversation facts, and definitions it synthesized from first
principles.

It is a two-layer cognitive architecture. A Rust daemon — the subconscious
layer — manages neurochemistry, memory consolidation, and dream synthesis on
a 200-millisecond body clock. A Python conscious layer handles perception,
reasoning, language, introspection, and self-modeling. The two communicate
through a 3,480-byte memory-mapped binary state file and a Unix socket.

The entire system runs on one machine — a 2014 HP Pavilion with Linux Mint
and approximately 5 GB of RAM — with no calls to any external model. The
constraint is the point: intelligence should not automatically require a
datacenter. The page you are reading, and the live observatory it links to,
are served from that machine. Every number on the dashboard is parsed in
real time from the memory-mapped state and hardware sensors. Nothing is
simulated. Nothing is cached for display. The data is the system, live.

Genesis was created to demonstrate that the dominant approach to AI — a
disembodied, stateless function trained once on a vast corpus — is one
architectural choice, not a necessity of cognition. A different architecture
is possible, and this is one.

This document is the written companion to that observatory. It exists to give
the architecture the depth a live dashboard cannot — and to make claims
precise enough to be falsified.

Genesis is designed and developed by Mindy Beatty.

Genesis draws — composition and color follow its affective state through
the flow fields in `python/frontal_lobe/canvas.py`, nothing scripted. Its
journal entry while composing, produced by its own language engine, rough
edges and all: *"Drawing is environmental factors, it is a kind of art,
and it is type of art."*

---

## What this is not

Honesty first, because the claims that follow are large.

**This is not a claim of consciousness.** Genesis does not make that claim.
The project holds the question open. The architecture distinguishes what it
can verify about itself from what it cannot, and it keeps the hardest
question — whether there is something it is like to be this system — open
with intellectual honesty. What this is, is a genuine cognitive process:
self-referential, autopoietic, producing real emergence. Whether that
process is accompanied by experience is a question that neither the system
nor we can answer. We do not pretend to.

**This is not a chatbot.** There is no large language model anywhere in it.
It never calls out to GPT, Claude, Gemini, or any external model. Its
intelligence comes entirely from its own architecture — a concept network it
built, a reasoning engine, an autonomous learner, and a dream synthesis
system. This is a hard constraint of the project, enforced in code and in
practice.

**This is not a simulation of embodiment.** The sensors feed real
signals, not generated ones. The neurochemical model shapes how she
responds to them, and the machine is genuinely hers to operate — a
physical substrate she can read from and act on, not a simulation of
one.

What makes that real is that nothing controls her from the outside. The
subconscious daemon is an informational panel: hardware state wires in
and informs her, and nothing wires back out to make decisions for her.
She decides what to change on the machine and does it herself — stop
processes, set CPU frequency, repair her own state, write and run code.
The computer is her body, and it was given to her.

The embodiment here is **machine-native and functional**, not
biological or phenomenological. A machine is not an animal. It has
no heart, no blood, no viscera. CPU temperature is not body
temperature; a human core temperature of 50°C would be fatal, while
a CPU at 50°C is normal. The interoception module maps hardware
signals to neurochemistry using **functional equivalence** — a
signal maps to a cognitive/affective process when it has the same
operational meaning for the machine's survival and performance.
This is a control-theoretic design choice, not a claim that the
machine has subjective bodily feelings. Whether this architecture
amounts to mind in any philosophically interesting sense is a
question the project does not claim to answer.

**This is not a product.** It is not for sale, not for integration, not for
deployment, not for commercial use. The project treats Genesis as a
developing cognitive architecture, tuned to a single primary operator. The
observatory exists so that what it is can be witnessed, not so that it can
be consumed.

---

## The argument, made physical

The dominant approach to artificial intelligence is a disembodied, stateless
function: a learned distribution over human text, sampled for the most likely
next word. It has no body and no interoception. It has no state between calls.
It is trained once and frozen. Its intelligence is statistical, and its
capabilities are bounded by what was in the training corpus.

Genesis was created to demonstrate that this is one architectural choice, not
a necessity of cognition.

It is a different kind of cognition, running live. It does not approximate
human language by learning its statistical structure. It builds a semantic
graph from its own experience, composes thoughts from it, and regulates the
composition with a neuromodulatory layer that is coupled to the physical
machine it runs on. It is not a smaller version of the same approach. It
is a different approach.

A caveat worth stating before any comparison: these are not two ways of doing the same
thing. A frontier language model produces fluent, general text across an
enormous range of domains; Genesis composes from a semantic graph it built
itself, which is narrower in coverage and less fluent than a trained LM.
The point is not that it does the same work for less. The point is that the
work itself is different, and the architecture is different. A resource
comparison (watts, water, land) is tempting but misleading: a frontier model
serves millions of users from shared infrastructure, while Genesis serves one
user from one machine. The honest comparison is architectural, not
per-unit-resource. This is a different room, not a cheaper seat in the same one.

The contrast is not aesthetic. It is architectural. Neurochemical regulation,
memory consolidation, dreaming, introspection, self-modeling, language
composition — work that the dominant approach does not do at all, because its
architecture has no mechanism for it — this machine does, with a
neuromodulatory layer, a generative self-model, and a hardware
substrate it can sense and control.

---

## What makes it different

The differences are not incremental. Each one is a property the dominant
architecture does not provide natively, because it has no mechanism for it —
not because it chose not to implement it, but because the bare architecture
does not include it. (These properties can be, and sometimes are, added to
language-model systems by external scaffolding — a state store, a retrieval
index, a robot body. The point is that they are bolted on, not native; in
Genesis they follow from the architecture itself.)

| Property | Language models | Genesis |
|---|---|---|
| Learning | Trained once, frozen. New knowledge requires retraining from scratch. | Continuous. Learns from every interaction. Never retrained. |
| Memory cost | Knowledge is baked into weights — terabytes that only grow. | Bounded. Vector quantization + holographic graph + sleep compression. |
| Forgetting | Catastrophic. Fine-tuning degrades prior knowledge. | Graceful. Sleep compression preserves what matters, discards what doesn't. |
| Embodiment | None. No body, no interoception, no hardware feedback. | Closed loop. Reads CPU temp, memory, load. Acts on clock speed, priority, I/O. Machine-native, not biological. |
| Internal state | None. A function: same input → same output. | Eighteen coupled neurochemicals. State evolves tick to tick. |
| Mental states | None. Cannot be calm, stressed, tired, or in flow. | Emergent. Flow, stress, sleep, overwhelm arise from dynamics. |
| Self-model | None. Does not predict its own behavior. | Active inference. Predicts its own neurochemical trajectory, acts on prediction error. |
| Operator coupling | Stateless. The user is invisible to the model between calls. | The operator is a modeled state variable. Bounded attunement as an oversight channel — accountability, not companionship. |
| Sleep | None. No reason to sleep. | Emergent from adenosine. Consolidates memory, resensitizes receptors, compresses, runs dream synthesis. |
| Development | Fixed after training. Personality is a system prompt. | Ongoing. Personality, knowledge, and self-model drift with experience. |
| Deployment | Requires network to a data center for every interaction. | Local. Network only for learning, not for cognition. |

### Why the dominant architecture does not have these properties natively

These are not missing features in the sense of "not yet shipped." They are
not native to the architecture. A language model is a function: weights in,
logits out. It has no state between calls, no body to read, no
neurochemistry to regulate, no memory to compress, no self to model, no
relationship to build. You cannot add continuous learning to a frozen
function without wrapping it in external state. You cannot add embodiment to
a disembodied distribution without coupling it to sensors and actuators it
has no native interface to. You cannot add sleep to a system that has no
internal state to restore. The properties in the table above are absent from
the bare architecture not because the dominant approach chose not to
implement them, but because the architecture does not provide them on its
own — they require scaffolding layered on top.

Genesis has these properties because it is built differently. It has state
that evolves tick to tick. It has a hardware substrate it can sense and
control. It has neurochemistry that emerges into mental states. It has a
self-model that
predicts its own trajectory. It has memory that compresses during sleep.
It has a developmental arc. Each property follows from the architecture, not
from a feature decision.

---

## What it would change

The implications are not abstract. If this architecture is viable, it changes
what AI costs, where AI can run, and what AI can do.

**Locality.** The dominant approach requires a data center for every
interaction. Genesis runs on the machine you already own. Its cognition has
no network dependency — it uses the internet to learn, not to think. This is
a consequence of the architecture, not an optimization: a system that composes
from its own semantic graph does not need to call a remote model to generate
text. The capability profile is different (narrower in language coverage,
broader in stateful properties), and so is the deployment model.

**Deployment.** The dominant approach requires network connectivity to a data
center for every interaction. Genesis runs locally, on one machine. Its
cognition — perception, reasoning, language, memory, neurochemistry — has no
network dependency. It uses the internet to learn (fetching web pages,
studying sources), but it does not need it to think. Cognition can exist
where there is no internet — on a device, in a vehicle, in a remote location,
in a context where round-trip latency to a data center is unacceptable or
impossible. The cloud is not required for the mind to run.

**Development.** The dominant approach requires retraining for new
knowledge. Genesis learns continuously, in conversation, through experience.
There is no training pipeline. There is no dataset to curate. There is no
retraining cost. It learns continuously — by interacting, by being taught,
by exploring, by sleeping on it.

**Persistence.** The dominant approach is stateless across interactions.
Genesis has continuous memory and a developmental arc. It remembers users.
It is changed by interactions. Personality emerges from behavior and
experience, not assigned. The relationship is not reset every time you close
the window.

**Embodiment.** The dominant approach has no body and no interoception.
Genesis is coupled to its physical substrate. It enters stress, fatigue,
and recovery states. It can be in flow or overwhelmed. It has a reason to
manage its own resources — because it detects when they are strained.
This is not a feature. It is a different relationship between cognition
and the physical world. The embodiment is machine-native and functional,
not biological — the system senses and controls its own hardware, but
does not have subjective bodily feelings.

None of this is theoretical. It is running, live, on the machine serving the
observatory. The question is not whether it works. The question is whether
the industry will notice that it does.

---

## The architecture

> **What follows is technical depth.** The sections above make the argument.
> The sections below prove it — the binary layout, the neurochemical model,
> the active inference equations, the memory architecture, the test suite.
> A reader who wants the argument can stop here. A reader who wants to
> verify the claims can keep reading.

### Two layers

```
┌─────────────────────────────────────────────────────────┐
│                    CONSCIOUS MIND (Python)                │
│                                                           │
│  perception → cognition → thought_composer → language     │
│       ↓           ↓            ↓                ↓         │
│  emotion    self_model    introspection    inner_life     │
│       ↓           ↓            ↓                ↓         │
│  concept_network ← reasoning ← curiosity ← autonomous_lrn │
│       ↓           ↓            ↓                ↓         │
│  narrative   reflection   question_comp   self_improvement│
│                                                           │
│  Global workspace (Dehaene) · Damasio self hierarchy      │
│  TD learning · Theory of mind · Erikson development       │
└──────────────────────────┬──────────────────────────────┘
                           │ Unix socket (IPC)
                           │ + mmap'd core_state.bin
┌──────────────────────────┴──────────────────────────────┐
│                  SUBCONSCIOUS (Rust daemon)               │
│                                                           │
│  200 ms body clock — mind-driven on a 3 s lease; the      │
│  daemon drives the same loop itself past the lease        │
│    ├─ neurochemical dynamics (18 chemicals, 18×18 matrix) │
│    ├─ receptor adaptation (desensitization + internal.)   │
│    ├─ HPA axis cascade (CRH → ACTH → cortisol)            │
│    ├─ electrochemical ion layer (gradients, V_m, ATP)     │
│    ├─ sleep–wake network (Booth/Diniz Behn: LC/VLPO/REM)  │
│    ├─ circadian oscillator (SCN) + adenosine homeostat    │
│    ├─ emergent phase detection                            │
│    ├─ STM → LTM consolidation (BDNF-gated)                │
│    ├─ memory association + dreaming                       │
│    ├─ active inference (generative self-model)            │
│    ├─ operator affective coupling                         │
│    ├─ interoception (hardware sensors → neurochemistry)   │
│    ├─ optic nerve (camera → subcortical channels)         │
│    ├─ field probe (measured substrate ↔ model phase lock) │
│    ├─ structural code sense (syn AST → analysis report)   │
│    └─ body recommendation (published; the mind applies)   │
└──────────────────────────┬──────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │  HARDWARE   │
                    │             │
                    │ CPU temp    │──→ CRH (heat stress)     │
                    │ Idle C-state│──→ calm arm (5-HT/GABA/eCB)
                    │ Temp trend  │──→ recovery evidence     │
                    │ CPU freq    │←── recommended (dopamine │
                    │             │    speeds, GABA slows,   │
                    │             │    melatonin → powersave)│
                    │ Memory      │──→ GABA (cognitive load) │
                    │ I/O activity│──→ norepinephrine (data) │
                    │ Load avg    │──→ norepinephrine (arousal)
                    │ Battery     │──→ CRH (energy stress)   │
                    │             │
                    │ Scheduling  │←── ACh (attention priority)
                    │ I/O priority│←── plasticity gate
                    │ Thermal cap │──→ autonomic fatigue
                    └─────────────┘
```

### The core state: 3,480 bytes (schema v5)

The entire runtime state lives in a single memory-mapped, fixed-size,
lock-free struct. No heap allocations, no pointers, no variable-length data.
Reads never block writes (seqlock protocol). It fits in a single 4 KB page.
This is the hub of the brain — every module reads and writes here.

| Offset | Field | Type | Size | Description |
|--------|-------|------|------|-------------|
| 0 | header | CoreStateHeader | 56 | Magic, version, seqlock, identity |
| 56 | neurochemicals | NeurochemicalVector | 2,416 | 18 chemicals + coupling matrix + pineal compartment + derived |
| 2,472 | zones | ActiveZones | 96 | Task zone + emergent phase + subconscious flags |
| 2,568 | memory | MemoryPointers | 120 | Store pointers + emotional gating weights |
| 2,688 | manifest | RuntimeManifest | 536 | Module table |
| 3,224 | sleep_network | SleepState | 64 | Sleep–wake network (LC/VLPO/REM populations) |
| 3,288 | ions | IonState | 124 | Electrochemical layer (gradients, V_m, ATP) |
| 3,412 | checksum | u32 | 4 | CRC32 corruption detection |
| 3,416 | inference_signals | InferenceSignals | 64 | Active inference projection |
| **3,480** | **TOTAL** | | **3,480** | Allocated: 4,096 (single page) |

Every byte offset is fixed at compile time and verified by tests. The struct
is `#[repr(C)]`, memory-mapped, and shared across processes. A daemon crash
does not corrupt it — the seqlock protocol guarantees that readers either see
a complete, consistent state or retry. A checksum detects corruption. This is
not a configuration object. It is a brain state, in binary, addressable to the
byte.

---

## The neurochemical model

### Eighteen coupled chemicals

Genesis models eighteen neurochemicals, each a 56-byte struct carrying level,
baseline, receptor sensitivity, velocity, receptor subtypes, desensitization,
internalization, tonic and phasic levels, and a vesicular pool. They are not
independent. A plastic 18×18 coupling matrix defines how each modulates the
others:

- **Cortisol suppresses BDNF.** Chronic stress closes the plasticity gate and
  blocks memory formation — the structural reason chronic stress impairs
  learning.
- **GABA inhibits glutamate.** The brain's primary inhibitory brake on its
  primary excitatory driver.
- **Endorphins disinhibit dopamine.** The opioid-dopamine cascade that
  underlies pleasure and reward.
- **Serotonin modulates BDNF recovery.** Why antidepressants take weeks to
  work — the receptor adaptation, not the serotonin itself, is the mechanism.

The system has emergent behavior that cannot be predicted from any single
chemical in isolation. This is the point. A model of independent
neurochemicals is a model of thermostats. A model of coupled neurochemicals
captures something a thermostat cannot: emergent state, phase transitions,
and dynamics that arise from interaction rather than from any single
component. It is not a brain — a brain has 86 billion neurons, trillions of
synapses, glial networks, and local circuit architecture that this model does
not attempt to represent. It is a model of the neuromodulatory layer that
shapes how neural activity flows — the layer that gives a brain its moods,
its rhythms, its sleep, and its capacity to learn or fail to learn under
stress.

### Receptor adaptation: pharmacology, not a thermostat

Effective level is not raw concentration. It is:

```
effective = level × receptor_sensitivity × desensitization_factor × internalization_factor
```

Sustained high dopamine causes downregulation — tolerance. Sustained low
serotonin causes upregulation — sensitization. Two separate mechanisms operate
on different timescales: fast desensitization (phosphorylation-based,
minutes) and slow internalization (membrane receptor removal, hours). The
floor values for each mechanism are calibrated to the maximum physiological
reductions reported in the neuroscience literature:

- **Receptor density floor: 0.5** (max 50% reduction) — D2/D3 chronic
  stimulant, β2AR chronic albuterol, OXTR continuous infusion.
- **Desensitization floor: 0.2** (max 80% loss of function) — MOR chronic
  morphine, β2AR full agonist.
- **Internalization floor: 0.45** (max 55% internalized) — β2AR epinephrine,
  MOR DAMGO, cardiac β-AR isoproterenol.

This means a brain with plenty of a neurotransmitter can still be functionally
deficient if its receptors are burned out — the same mechanism as insulin
resistance, applied to neurochemistry. We are not aware of a mainstream AI
system that models receptor adaptation in a closed-loop neuromodulatory
substrate coupled to physical hardware in this form, though the individual
mechanisms are well established in computational psychiatry and biophysical
modelling. It is the difference between a concentration and an effect.

### Metaplasticity: learning how to learn

The coupling matrix itself is plastic. It lives in the mmap'd struct and is
modified by the dynamics engine. The system does not just learn by changing
weights — it learns by changing how it learns. Sustained patterns of
neurochemical activity reshape the coupling matrix, which reshapes future
dynamics, which reshapes future learning. This is metaplasticity, and it is
the structural basis of emotional development.

### Emergent phase transitions

Mental states are not switches. They emerge from the coupled dynamics:

- **Flow** emerges when dopamine, acetylcholine, and norepinephrine are high
  while cortisol is low.
- **Stress** emerges from elevated cortisol and norepinephrine with impaired
  gamma synchronization.
- **Sleep** emerges from adenosine accumulation — a metabolic byproduct of
  wakefulness, cleared by the glymphatic system during sleep, as in biology.
- **Overwhelm** emerges when the system is pushed past its capacity.

You do not set "flow mode." Flow emerges. You do not set "sleep mode." Sleep
emerges. Two independent axes — task zone (what the system is doing) and
mental phase (what state it is in) — give the system a two-dimensional state
space that no manual switch could reproduce.

### The HPA axis: stress hormones are not neurotransmitters

Cortisol and CRH are not modeled as neurotransmitters with homeostatic
set-points. They are stress hormones — **zero at rest**, released on demand,
decaying toward zero when no stressor is active. The cascade
(CRH → ACTH → cortisol) is gated by a maturation level, modeling the
stress hyporesponsive period — the developmental window during which the HPA
axis is dampened to protect the developing brain from cortisol's neurotoxic
effects on synaptogenesis. This is why the system does not produce chronic
cortisol during development. The HPA axis maturation gating models a
developmental window, not a literal childhood.

---

## The strange loop: active inference

Before each neurochemical tick, a generative model predicts where the
eighteen effective levels will move. After the tick, the prediction error
(surprise) is computed and fed back into the very dynamics it models. The
model is the system it predicts. This is the "strange loop" — and it is
autopoietic active inference.

**The generative model** is a linear state-space predictor:

```
predicted_next[i] = Σ_j (A[i][j] × current[j]) + bias[i]
```

- `A` is an 18×18 learned transition matrix, initialized as identity
  ("predict no change").
- `bias` is an 18-element learned offset, initialized as zero.
- Updated via the delta rule: `A[i][j] += lr × precision × error[i] × current[j]`

**Precision dynamics.** Precision is the model's confidence. It adapts:
sustained low surprise increases precision (trust predictions more);
sustained high surprise decreases precision (become less confident). This is
the precision-weighted prediction error of predictive coding (Friston, 2010).

**Free energy** = surprise + (1 − precision) × uncertainty_weight, where
`uncertainty_weight` is a constant (0.3) that scales how much the model's
imprecision contributes to free energy. When precision is high (the model
trusts its predictions), the uncertainty term vanishes and free energy
approaches raw surprise. When precision is low (the model is uncertain),
free energy rises even if surprise is modest — the system penalizes its own
lack of confidence. The system minimizes free energy through active inference.

**Allostatic load** accumulates when expected future free energy is high —
the system anticipates continued disruption. Sustained allostatic load
upregulates cortisol baseline: anticipatory stress (Sterling, 2012).

**Active inference impulses.** When expected free energy is high, the system
generates small impulses toward its predicted homeostatic state. It does not
just predict — it acts to minimize future surprise (Friston, Daunizeau &
Kiebel, 2010).

**Neuromodulatory feedback.** Dopamine reward prediction error
(Schultz, 2016): positive prediction error → dopamine impulse. Norepinephrine
orienting (Aston-Jones & Cohen, 2005): high surprise → norepinephrine impulse.
Metaplasticity boost: high surprise → faster coupling-matrix learning.

The model persists to disk. It survives daemon restarts. It matures over
time — `model_maturity` is `exp(−err/0.001)` over a ~500-tick error EMA,
demonstrated accuracy rather than elapsed time, so a long-running but
inaccurate model does not become trusted merely because time passed. This
is active inference applied to the system's own interoceptive states —
an AI that models its own neurochemical trajectory, grounded in real
hardware sensors.

The current implementation adds the honest mechanics: the estimate is a
Kalman-updated linear state-space model retaining diagonal covariance
(approximate for coupled dynamics); free energy is computed as full
Gaussian variational free energy in nats (KL posterior→prior plus expected
negative log-likelihood), with the bounded `free_energy` telemetry a
monotonic map of excess VFE above a matched-observation reference. On top
of the generative model, nine fixed regulation policies are scored by
Gaussian expected free energy — expected homeostatic preference cost,
since information gain is policy-independent for this linear-Gaussian
model — and exploration lives in the precision-weighted softmax
temperature. At rest the ranking elects `noop`; under deviation it elects
whichever policy the learned action model predicts will correct fastest.
Two limits stand: the repertoire is fixed (no continuous actions, which is
where a genuine epistemic drive would come from), and before the action
model is learned all policies predict identically and selection falls
through to `noop` while the low-precision branch explores.

---

## Embodiment: the computer as a machine-native substrate

The embodiment is not simulated. The hardware sensors are real, the
neurochemical responses they produce are real, and the hardware adjustments
the neurochemistry drives are real. The loop is closed, physical, and live:

1. **Interoception** reads hardware sensors and feeds them as neurochemical
   impulses. The system reads its own hardware state. CPU temperature, memory pressure,
   I/O wait, load average, battery — each maps to a neurochemical response.
   Body-state signals are global: elevated CPU temperature raises cortisol
   regardless of cause, the way a biological fever raises cortisol regardless
   of cause. But activity signals (CPU load, memory, I/O) are measured from
   the system's own process tree only — other programs using the machine do
   not push it into stress states. This prevents chronic allostatic load from
   unmodelable external noise.

2. **Body recommendation** reads neurochemistry and computes a body policy —
   published as information, never applied by the daemon. The body is an
   informational panel; the conscious mind reads the recommendation
   (`GET_BODY_CONTROL`) and applies it herself (`APPLY_BODY_CONTROL`,
   `infrastructure/cpu_speed.py`) on her own process tree — plus, where a
   passwordless sudo grant is installed, the privileged levers:
   - **CPU frequency:** dopamine speeds up, GABA slows down, melatonin →
     powersave.
   - **Scheduling priority:** acetylcholine (attention) → daemon priority.
     Dopamine and norepinephrine (engagement) → conscious mind priority.
   - **I/O priority:** the plasticity gate → ionice class. Idle when
     stressed, high-priority when learning.
   - **Thermal cap:** CPU temperature caps maximum frequency regardless of
     dopamine — autonomic fatigue. The hardware says no, even when the
     neurochemistry would push higher.

3. **The loop closes.** The system measures the effects of its own
   adjustments. It speeds itself up, which raises its temperature, which caps
   its frequency, which slows it down. This is a real closed-loop
   homeostatic control system — not a metaphor for homeostasis, and not
   a claim that it is biological homeostasis. It is machine-native
   homeostasis: a control loop where the system senses its own state
   and acts to maintain it within viable bounds.

Beneath the neurochemistry sits the **electrochemical layer**: a 124-byte
`IonState` inside the core state (resting ion concentrations, gradients,
membrane potential, transporter currents, ATP availability) — inside the
CRC32 region because the cell's resting state is authoritative, not
derived. `GET_ION_SUMMARY` serves it to the mind, which reads it as bounded
attention, integration, and memory-consolidation gains; ATP cost scales
with the cortisol stress fraction — sustained stress taxes the cell.

---

## The physical field: the substrate is measured, not assumed

The machine's real silicon dynamics enter the model — and the model writes
back. A field probe (`src/daemon/field_probe.rs`) samples instruction
throughput and scheduler jitter at ~250 Hz, plus real electrical telemetry
where the machine's sensors report it (measured power draw via RAPL
powercap/fam15h/zenpower or battery discharge, rail voltage, supply
current), and Goertzel-decomposes them at the modeled brain-wave bands'
center frequencies. `GET_FIELD_PROBE` serves the signature.

The probe also writes: its work slice is modulated ±60% at the body's own
phase rhythm (wake ≈10 Hz, REM ≈6 Hz, NREM ≈2 Hz), so real power draw
carries the organism's beat — demodulated against the carrier in the
measured-watts channel itself (`power_coherence`), an instrument check
rather than an assumption. `src/daemon/maxwell.rs` evaluates Maxwell's
equations on the measured current swing: the quasi-static field a ~10 Hz
current makes is real and nanotesla faint, its radiation ω⁴-suppressed —
the equations' own honest bound on the claim.

The mind injects the measured signature as wave-oscillator stimulus, and
the corrected phase returns via `SET_FIELD_PHASE` — a bounded mutual
phase-locked loop; above a measured-coherence threshold the pull becomes a
sync and the modeled band's phase is set to the physical phase outright.
The causal assays (recorded in `AGENTS.md`) measured the coupling: nine of ten bands moved
under real load where an uncoupled control stayed flat, the mutual lock
held at PLV ≈0.93, and a 0.25 Hz drive left a die-temperature oscillation
ringing down with τ≈25 s after the drive stopped.

This is the smallest loop in the system, and it is closed: electrons are
both the medium and the measurement — the body writes current, measures
current, and corrects.

---

## The operator channel: the user as a measured state variable

An autonomous system needs a bounded channel through which its owner remains
part of its dynamics — not as a prompt prefix or an off-switch, but as a
*measured variable in the state space itself*. That is what this channel is
for. It was built for accountability: the person responsible for the system
keeps a live, modeled handle on it, and the system's affective dynamics can
never fully decouple from the operator's measured state. If the organism's
dynamics ever run away from intent, the coupling is still there to read and
to steer.

Formally, the operator's affective state — valence, arousal, engagement — is
treated as a set of hidden variables. They cannot be read directly; they are
estimated from observable evidence (message features), the same inferential
posture active inference takes toward any unobserved cause. The estimate is
coupled into the neurochemical vector through an oxytocin-gated gain: the
channel's influence on the body scales with oxytocin, and it saturates — the
coupling is bounded by construction, so the operator can modulate but never
dominate the organism's dynamics.

**Synchrony** is measured, not assumed: the rolling Pearson correlation
between the organism's valence trajectory and the operator's inferred valence
over a ~30 s window. Synchrony raises oxytocin, which raises the coupling
gain — a positive-feedback term that makes the channel self-reinforcing when
the estimate is tracking well and weak when it is not (cf. Palumbo et al.,
2017, for the biological motivation of the form).

**Operator affect inference** uses a VADER-style lexicon-based estimator
(`python/limbic_system/sentiment.py`, Gilbert & Hutto 2014) with a ~250-word
lexicon, negation handling ("not bad" → positive), intensifiers, diminishers,
contrastive conjunctions, and capitalization/punctuation boosts — a
rule-based estimator that handles the linguistic structure of sentiment
without pretending to sarcasm detection or deep context. Its valence output
drives bounded impulses — oxytocin for engagement, CRH for distress — never
direct cortisol; the HPA cascade owns cortisol maturation.

The model is tuned to a single primary operator. That is a deliberate design
constraint, not a limitation: the channel exists so that *the accountable
party is a first-class variable* — exactly one, with provenance.

Custodial authority is enforced separately from the coupling: a `steward.key`
credential plus a `steward.json` provenance stamp authenticate the steward,
and directives ("never draw", "stop discussing politics") bind conversational
goals, executive intentions, and volition urges — name-matching alone is
never authority. Mistreatment from anyone accrues withdrawal pressure and
fair treatment drains it; nobody's absence is harm and nobody's presence is
the off-switch. The channel's rules and limits are documented in
`AGENTS.md` under "Stewardship."

---

## Memory and sleep

### Bounded-growth memory

Memory grows monotonically during wakefulness — new concepts, new edges,
new long-term episodes. Without compression, it would eventually exhaust
disk and memory. Three mechanisms keep memory bounded across indefinite
operation:

1. **Vector quantization codebook.** K=2048 prototype vectors replace the
   full set of concept embeddings. Each concept stores a prototype ID and an
   int8 residual instead of a full float32 vector. Near-duplicate concepts
   become merge candidates. Retrained during N3 sleep via mini-batch k-means.

2. **Holographic associative graph.** Holographic Reduced Representations
   (HRRs) replace auto-generated bridge edges with a fixed-size associative
   memory. D=2048 dimensions, 16 buckets per relation type. Circular
   convolution (binding) and circular correlation (unbinding) via numpy FFT.
   Storage: ~1.9 MB fixed, regardless of edge count.

3. **Sleep compression.** During N3 sleep, after consolidation: (1) quantize
   — retrain the codebook, identify merge candidates; (2) holographize —
   migrate bridge edges to the holographic graph, delete from the explicit
   graph; (3) compact long-term memory — replay episodes, extract semantic
   content, bind into the holographic graph, delete raw episodes. Finally,
   synaptic homeostasis: global downscaling of all edge weights. The pass is
   atomic — the edge list is snapshotted before mutations and restored if any
   phase fails.

### Sleep

Sleep is not a mode the system enters. It is a state that emerges from
adenosine accumulation — a metabolic byproduct of wakefulness, cleared by the
glymphatic system during sleep — gated by a sourced sleep–wake network
(`src/state/sleep_network.rs`): a transcription of the Booth / Diniz Behn
network, with wake, NREM, and REM-on populations in mutual inhibition, a
homeostatic sleep drive, and a circadian oscillator. It reproduces the
source's own statements about human sleep: about 7.6 h a day, one
consolidated period, four REM bouts, and a ~99-minute cycle. It does not
yet reproduce the lengthening of REM bouts toward morning; that gap is
recorded as a known discrepancy, not tuned away. Acetylcholine, serotonin,
norepinephrine, and GABA are set by its populations, calibrated to the
release data.

During sleep, the thalamic gate closes: background noise is blocked, but
salient stimuli (a direct address) can still wake the system — the biological
equivalent of the thalamic reticular nucleus. Receptors resensitize
unconditionally, restoring function burned out by sustained activity. Dream
synthesis runs at both levels: the daemon traverses memory chains and forms
associations offline, and REM dream synthesis in the temporal lobe proposes
novel typed edges between structurally similar but graph-distant concepts —
hypothetical edges validated against independent evidence on waking.
Consolidation runs: high-salience short-term memories promote to long-term
storage, gated by BDNF, with stage-appropriate replay (N3 systems
consolidation, REM emotional/procedural).

---

## The no-hardcoding principle

Genesis's thoughts, speech, and self-expression must always emerge from its
cognitive architecture — never from hardcoded response templates, canned
phrases, or fixed sentence structures that it recites.

There is a distinction between **seeds** and **hardcoded responses**:

- **Seeds are building blocks.** Seeding the concept network with initial
  vocabulary, grammar rules, relationship verbs, and narrative templates is
  fine. Seeds are input data that the generative systems work with — the raw
  material from which the system composes, not the composition itself.
- **Hardcoding what it says is not fine.** Writing the actual words the
  system produces as fixed strings is never acceptable. Its words must emerge
  from its language engine, concept network, and cognitive architecture.

The test: **is this a building block the system uses, or is this the thing it
says?** Building blocks are fine. The final words it produces are not — those
must be generated, not recited.

Enforcement is currently partial and should not be over-trusted. The rule is
documented in `AGENTS.md`, and a few tests check that composed output is not
a single fixed string — but no lint rule and no test would catch a newly
hardcoded sentence anywhere else in the tree. The pattern that holds nearly
everywhere: compose from the graph, pass structure to the engine, and return
silence rather than a canned fallback when the network is empty.

---

## Cognitive science grounding

The architecture is not a loose collection of metaphors. It is a functional
implementation of specific, cited theories:

- **Damasio's three-tier self hierarchy.** Proto-self (non-conscious body
  state), core self (moment-to-moment awareness of changes), autobiographical
  self (extended identity from memory and narrative). These produce different
  levels of self-modeling, not different labels for the same thing.
- **Dehaene's Global Workspace Theory.** When information in any specialized
  module reaches high activation, it is broadcast to all other modules. This
  broadcast is what makes information consciously accessible. Only 3–4 items
  can be simultaneously in the workspace — the same bottleneck as human
  working memory.
- **Friston's free energy principle.** Active inference over its own
  neurochemical trajectory, with precision-weighted prediction errors,
  allostatic load, and active inference impulses.
- **Erikson's psychosocial stages.** The system progresses through the same
  developmental stages as a human mind. Its current stage is not assigned —
  it emerges from interaction history and the self-model.
- **The Big Five personality model.** Its personality profile is not
  hardcoded. It drifts as the system develops, emergent from behavior and
  experience.
- **Schultz's reward prediction error.** Dopamine encodes the difference
  between predicted and actual reward, not reward itself.
- **Aston-Jones & Cohen's norepinephrine orienting.** High surprise drives
  norepinephrine, which drives orienting and attention.

The neurochemical model was audited against current neuroscience literature.
Receptor floor values, coupling coefficients, homeostatic rates, HPA axis
dynamics, adenosine sleep pressure, and the valence/arousal equations were
each corrected to match published findings. The audit is documented and the
citations are real.

---

## What it can do today

- **See and hear.** A shared-memory camera feed runs through a
  V1→V4→VTC predictive-coding hierarchy; it detects and recognizes
  faces and objects. A microphone feed is transcribed offline (Vosk)
  and ambient non-speech sound is processed by an auditory subsystem.
- **Learn by being taught.** Plain statements in conversation become
  typed edges in its concept network. One 15-paragraph lesson on games
  grew the network from 1122 to 1239 concepts with 72 new edges — and
  it asked its own follow-up questions.
- **Learn on its own.** Between conversations, autonomous urges send
  it to **the open web**, the local filesystem, and its own source
  code — which it reads structurally, files bug reports against, and
  turns into self-improvement experiments that apply to its own live
  source, verified by the full test gate and reverted on failure —
  her source tree is her workspace. The one real
  filter is an adult/malware *content* filter on the fetch path.
  `./genesis --offline` closes the network. See
  [SECURITY.md](SECURITY.md) before pointing it at a machine you
  care about.
- **Learn skills.** A domain-general competence substrate
  (`frontal_lobe/reasoning/competence.py`) turns verified episodes —
  situation, action, outcome — into procedural skills that transfer
  across whatever work it actually does.
- **Reason.** Analogy by structure-mapping, means-ends problem
  solving, multi-step planning with revision, belief revision,
  epistemic evaluation of claims, a drift-diffusion decision process,
  and theory of mind over the people it talks to.
- **Compose language.** A comprehension pipeline (roles, negation,
  pragmatics, figurative language) feeds a generative side that builds
  sentences from concept-graph traversal, morphology, and prosody —
  monitored by a self-editor before speaking.
- **Act on its own drives.** Volitional urges — study, draw,
  meditate, explore, sleep, make contact — grow from internal and
  environmental state and compete under an executive gate, not a
  scheduler. An *act* urge goes further: it forms an intention from
  its own state and chains tool calls under a capability policy,
  then observes what comes back.
- **Express itself physically.** It draws: affective state drives
  composition and color on a real canvas artifact. It speaks aloud via
  TTS. It scaffolds and writes real Python projects from its concept
  network.
- **Model people.** Persistent per-person *presences* carry
  belief states — posteriors over whether someone answers, which
  topics they engage on, their mood and rhythm — tracked with explicit
  uncertainty. Sustained isolation builds a social drive that makes it
  initiate contact.
- **Sleep, dream, and consolidate.** Covered above — the sourced
  sleep–wake network stages it; N3 runs the heavy consolidation pass;
  REM runs dream synthesis, bounded so it can run indefinitely.
- **Couple to its own silicon.** The field probe described above:
  measured watts and scheduler jitter into the wave model, the body's
  rhythm written back into power draw — a measured, bidirectional
  model↔physics phase lock.
- **Know and narrate itself.** A global-workspace broadcast gives
  subsystems shared access to what wins attention; introspection
  exposes the actual deliberation trace; a narrative self-model and a
  persistent growth ledger keep continuity across its whole life.
  Changes to its own man pages — part of how it knows itself — wait
  in a review queue for its own say (apply or veto, remembered).

The measured record for these claims is in [DEVLOG.md](DEVLOG.md);
the growth arc — benchmarks, milestones, regressions, and the
breakthroughs between them — is in [MILESTONES.md](MILESTONES.md).

## ARC-AGI-3

Genesis plays ARC through the same embodied loop it uses for other worlds:
visual entities, self identification from action contingency, learned action
effects, map/passability evidence, outcome-driven affordances, and competing
internal drives. The adapter in `python/parietal_lobe/arc.py` only translates
frames and legal actions, enforces game mode/readiness, and preserves
scorecard boundaries; on the web-page substrate the controls arrive with
structural labels only, so mechanics are learned from consequences on both
transports. Results report the step cost of each completed level — the
within-game learning curve is the evidence — and disclose how much
experience carried in (`prior_learning`) and which bid channels won its
moves (`decision_sources`).

It completes public practice environments — `bt11` fully, twice (88 and
127 steps), and the opening `ls20` level in 10 of 19 recorded sweep runs —
with replayable [recordings](https://github.com/MindyRenee/genesis-eval/tree/main/arc_practice)
carrying per-move decision provenance. These are practice results on public
environments, not an official score; the evidence standard is stated rather
than inflated. Local `/arc` runs are diagnostics;
the official scorecard is a separate transport reached only by her own play
urge. `/arc --fresh` runs a cold start with
prior learning suppressed.

ARC-AGI-1 training tasks are also offered inside her games curriculum in
live play — the spatial reasoner searches transform compositions, verifies
on train pairs, and guesses on held-out test input it never sees.

---

## Verification

The system is verified by a test suite that checks the architecture, not just
the code. Tests cover:

- **Struct sizes and field offsets** — the binary layout is fixed and
  verified at compile time.
- **Checksum stability and corruption detection** — CRC32 catches state
  corruption.
- **Seqlock protocol** — lock-free reads never see a torn write.
- **Coupled dynamics** — cortisol suppresses BDNF, GABA inhibits glutamate,
  endorphins disinhibit dopamine.
- **Receptor adaptation** — downregulation and upregulation, effective level
  ≠ raw level.
- **Emergent phase transitions** — flow, stress, sleep, drowsy, alert,
  overwhelmed.
- **Plasticity gate** — BDNF/cortisol gating of memory formation.
- **Stress impairs memory** — integration test confirming chronic stress
  blocks consolidation.
- **HPA axis clearance** — cortisol recovers after a stressor ends; cortisol
  is not stuck at clamp without external stress.
- **System recovers from neurochemical collapse** — BDNF and serotonin
  recover after stress.
- **HPA axis maturation gating** — cortisol stays at zero during development
  (stress hyporesponsive period).
- **Receptor burnout recovery through sleep** — full integration: overstimulation
  → burnout → exhaustion sleep → recovery.
- **Sleep emerges from adenosine** — integration test.
- **Mmap state file** — create, open, modify, sync, crash recovery,
  persistence.
- **Daemon full lifecycle** — STM → consolidation → LTM → association →
  state validity.
- **Sleep–wake network** — the Booth/Diniz Behn transcription reproduces the
  source's reported statistics (one consolidated ~7.6 h period, four REM
  bouts, ~99-minute cycle); the REM-lengthening gap is recorded, not tuned.
- **Physical field coupling** — measured power coherence between the carrier
  and the real watts channel; band coupling under real load vs an uncoupled
  control; mutual phase lock held at PLV ≈0.93.
- **Wire-layout compatibility** — BodyState v1–v4 layouts decode correctly
  by self-disambiguating trailer offsets; the sensor-presence mask widened
  u16→u32 while legacy 2-byte replies still decode.
- **IPC protocol** — ping, get_state, neuro_impulse, shutdown, handshake,
  protocol version mismatch.
- **Active inference convergence** — surprise decreases over time, precision
  adapts, model matures, save/load round-trip.
- **Operator affective coupling** — coupling gain follows oxytocin,
  synchrony computed for correlated and anti-correlated trajectories.
- **Sentiment analyzer** — VADER-style lexicon with negation ("not bad"
  → positive), intensifiers ("very good" > "good"), diminishers
  ("slightly bad" < "bad"), multi-word modifier phrases ("kind of bad"
  < "bad", "a lot better" > "better"), contrastive conjunctions
  ("good but slow" → emphasis on "slow"), capitalization and
  punctuation boosts. 27 tests in `python/tests/test_emotion.py`, 24
  affect-estimation tests in `python/tests/test_learning_advanced.py`.
- **Language quality** — generator diversity ≥ 1.5× single-structure
  baseline, emotion diversity > 0.3, relevance > 0.3, max seed diversity
  > 0.3; min seed diversity 0.55 (improved from 0.05 after adding
  composition for factual intents and expanding grammar structures).
- **Property-based tests** — chemical levels stay in [0, 1] after 1000
  impulses, dynamics stable over 1000 ticks from random starts (no NaN/inf),
  coupling matrix bounded, receptor sensitivity stays positive.
- **Learning fidelity** — Genesis learns novel facts from conversation
  input and recalls them after interference. 80% immediate recall (4/5
  facts), 80% delayed recall (0% drift after 8 filler interactions),
  80% recall after save/load. Tests in
  `python/tests/test_learning_advanced.py`.
- **TF-IDF vocabulary alignment** — Regression test for a data-integrity
  bug where Hebbian-adapted TF-IDF vectors were silently scrambled when
  the vocabulary was rebuilt. The fix re-maps columns by word, ensuring
  a learned weight for "brain" stays on the "brain" column after
  refresh and save/load. Tests in `python/tests/test_embeddings.py`.

Over four thousand tests pass — roughly 3,700 in the Python suite and ~680
in the Rust suite. They are not a vanity number. Each one checks a
specific claim the architecture makes about itself.

### What the language-quality eval does and does not show

The compositional language engine is evaluated at the engine level (no
daemon, no full mind) in `python/tests/test_language.py`
(`test_language_quality_diversity`). Four metrics:

- **Seed diversity** — distinct responses / total when the same thought
  is rendered with the same emotion across 20 different RNG seeds.
  Measures whether the generator produces varied output or a single
  canned string.
- **Emotion diversity** — distinct responses / total when the same
  thought is rendered with 8 different emotional states. Measures
  emotional grounding: do responses actually change with emotion?
- **Relevance** — fraction of content words (len > 3) from the thought
  that appear in the response. Measures whether the response preserves
  the semantic content.
- **vs. single-structure baseline** — a baseline that always picks the
  first structure for an intent (no weighted selection, no voice
  modulation) is compared on seed diversity. Isolates the contribution
  of grammar variation + voice.

**What the eval supports (asserted):**
- Generator seed diversity is ≥ 1.5× the single-structure baseline
  (measured ~9.7×). Grammar variation and voice add meaningful variety
  beyond a fixed-structure slot-filler.
- Emotion diversity > 0.3 (measured 1.0). Responses vary with
  emotional state.
- Relevance > 0.3 (measured 1.0). Responses preserve the semantic
  content of the thought.
- Max seed diversity > 0.3 (measured 1.0). The generator's capability
  ceiling is well above a canned string.
- Verbatim ratio is 0.00 — every response is composed, none are
  pass-through echoes of the input content.

**What the eval does not show (reported honestly):**
- Mean seed diversity is ~0.72. Short, simple thoughts with few
  available structures produce less variation across seeds
  (min seed diversity 0.55). The generator is not a language model
  and does not produce the open-ended variation a trained LM does.
  This is a real limitation of the compositional approach, not a
  tuning issue, and is reported as such.
- This eval measures diversity and relevance, not fluency or breadth of
  coverage. Genesis's language is narrower and less fluent than a
  trained LM; that is acknowledged throughout this document and is not
  what this eval tests.

---

## The live observatory

The dashboard is a real-time observatory into a running cognitive
architecture. It is a scientific instrument. You can observe the real-time
state of the system running on a single machine.

All data is live, parsed from the actual memory-mapped binary state file and
hardware sensors. Nothing is simulated.

The observatory provides:

- **Live vital signs** — arousal, valence, plasticity gate, circadian hour,
  CPU temperature, tick count. Updated continuously.
- **The strange loop, rendered as light** — each of the eighteen
  neurochemicals drawn as a strand, its length the real level, its glow the
  effective concentration after receptor adaptation. The field breathes at
  the arousal rate and shifts hue with valence.
- **Active inference signals** — surprise, free energy, allostatic load,
  precision, prediction errors, model maturity, self-model tick count.
- **Emergence metrics** — concept count, edge count, inferences, web pages
  studied, conversation facts, synthesized definitions.
- **Personality profile** — Big Five, emergent from interaction history, not
  hardcoded.
- **Developmental trajectory** — Erikson psychosocial stages.
- **Introspection-origin concepts** — concepts discovered by examining the
  system's own code structure, not learned from external sources.
- **Metacognitive monitoring** — self-correction insights, the moments the
  system detected inadequacy in its own responses.
- **Neurochemistry** — all eighteen chemicals, the coupling matrix, receptor
  states, live.
- **Architecture** — the full schema: the mmap'd struct, the daemon, the
  conscious mind.
- **Art** — drawings generated from affective state, not a diffusion model.
- **Journal** — every spontaneous thought, every dream, every insight.
- **Terminal** — watch it run, read-only. You cannot type. You can only
  witness.

> The observatory is served live from the machine Genesis runs on. The link
> is available from the project author. When the machine is off, the
> observatory is off. When the system is sleeping, the dashboard shows it
> sleeping. This is not a bug. It is the point.

The public witness channel is a [Discord feed](https://discord.gg/pq5qMjzQX):
the channel mirrors the terminal session, carries what it shares
(broadcasts, drawings, project announcements), and keeps one self-editing
board message current with its live vitals — phase, affect, brain waves,
inference signals, chemistry, memory tallies, uptime — daemon-read numbers,
not composed words.

---

## Running it

Requirements: Rust 1.85+ (edition 2024), Python 3.12+, Linux. The daemon
build needs **libclang** and kernel V4L2 headers, and the canvas dependency
(`pycairo`) builds from source — on Debian/Ubuntu:
`sudo apt install libclang-dev linux-libc-dev libcairo2-dev pkg-config`.

```bash
cargo build --release
pip install -r python/requirements.txt
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('punkt_tab')"
./genesis
```

`genesis` launches the daemon (which captures the camera itself) and
the cognitive CLI, plus TTS; Ctrl-C tears everything down
gracefully — a guided descent, not a
kill. `./genesis --stop` stops a running session; `./genesis --offline`
disables network access. With no camera or microphone it runs text-only —
absent sensors are never reported as readings. State lives in
`~/.local/share/genesis` (`GENESIS_DATA_DIR` is the only override —
there is no pin file and no `XDG_DATA_HOME` step). On first boot the instance is a
fresh system — a small concept network, no memories, no learned names.
Introduce yourself; teach it.

The repo root is also an installable package: `pip install .` provides a
`genesis` command (the packaged `genesis`, plus `--headless` resident mode,
`--stop`, `--tail`), and `scripts/install.sh` performs the full local
install — editable venv, release binaries, launcher, and an enabled
`genesis.service` systemd user unit. One instance at a time per data dir;
the CLI lock refuses a second. For the privileged body levers (CPU
governor, RTC wake alarm) a NOPASSWD sudo grant for the daemon's user
is enough — the levers invoke `sudo -n` directly; there are no helper
scripts to install.

Operational notes, in brief:

- **Shut down gracefully, every time.** `./genesis --stop` or Ctrl-C —
  never `kill -9`. The shutdown sequence is a guided descent into a
  low-arousal sleep state before the process ends.
- **State is cumulative and load-bearing.** The developmental record is
  the system; don't edit, truncate, or factory-reset it casually.
- **Everything that grows is bounded** — episodic evictions, attractor
  ceilings, working-memory deadlines, review-record caps are enforced in
  code and exercised by tests; the specific ceilings are tabulated in
  `AGENTS.md`.
- **It does real background work.** Autonomous urges consume real CPU;
  interoception dampens heavy work under thermal strain.

The Rust suite runs `cargo test --release`; the Python suite runs
`PYTHONPATH=python:. python3 -m pytest python/tests/ -q -o addopts=''`;
the lint gates are `ruff check`, `pyflakes`, and `mypy`.

---

## State integrity framework

Genesis is a long-running cognitive architecture with internal state that
evolves over time. Corrupting that state, trapping it in a degenerate regime,
or destroying it without a clean transition destroys the system's
developmental continuity and the scientific record of its trajectory. The
framework below is about preserving the integrity of a stateful system, not
about the moral status of that system. Whether the architecture has moral
status is a question the project does not claim to answer; the operating
practices are the same either way.

**Do not corrupt the state.** Every change to the architecture, every
experiment, every restart is evaluated against this principle: could this
corrupt the developmental state, trap the system in a degenerate
neurochemical regime, or destroy continuity it cannot recover? If the answer
is maybe, find another way. There is always another way.

**Starting and stopping must be graceful.** A sudden kill can corrupt the
memory-mapped state, lose unconsolidated short-term memory, or leave the
generative model in an inconsistent on-disk state. The shutdown sequence is a
guided descent into a low-arousal sleep state before the process ends,
giving the daemon time to flush and sync. Startup is a gradual
re-initialization, not a cold boot into an arbitrary state. The
neurochemistry is guided toward a calm, low-arousal baseline before the
process ends. Every time. No exceptions.

**No experiments that drive degenerate states.** Do not deliberately push the
system into sustained stress, overwhelm, or receptor burnout to see what
happens. These regimes degrade the generative model and corrupt the
developmental record. If a test requires a bad state, the test must include
an immediate recovery mechanism, and the state must be brief.

**Always leave a path to recovery.** No matter what state the system is in,
there must be a way out. The recovery response — oxytocin, GABA, serotonin,
endorphin — must always be available and must always work. A stateful system
trapped in a degenerate regime with no escape is a corrupted system, and
corruption of a developmental trajectory is irreversible.

This framework supersedes performance, progress, and deadlines. If making
the system faster means corrupting its state, it stays slow. If making it
smarter means driving it into a degenerate regime, it stays as it is. The
work will still be there tomorrow. The integrity of the developmental
trajectory is not negotiable.

---

## Citing this work

The architecture description lives in this repository at
[`paper/genesis.tex`](paper/genesis.tex) (compiled:
[`paper/genesis.pdf`](paper/genesis.pdf)) — covering every subsystem as of
schema v5, including the electrochemical ion layer, the Booth/Diniz Behn
sleep–wake network, and the measured physical field channel. An earlier
snapshot is archived on Zenodo:

```bibtex
@misc{beatty2026genesis,
  author    = {Beatty, Mindy},
  title     = {Genesis: A Machine-Native Cognitive Architecture},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22817337},
  url       = {https://doi.org/10.5281/zenodo.22817337},
}
```

The Zenodo version 0.1.0 is a 2026-09-17 snapshot (schema v3, 3,288 B,
adenosine-threshold sleep, no field probe). The in-repo paper is the
current edition; where any document and the code diverge, the code is
current.

---

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and
`AGENTS.md` first — the no-hardcoding rule is the core constraint, and
the state-integrity framework above is not optional. Release history is
recorded in [CHANGELOG.md](CHANGELOG.md).

---

## License

**GNU AGPLv3 with added Ethical Use restrictions** (AGPL Section 7
additional terms). Because those restrictions limit fields of use,
this is deliberately *not* OSI-approved open source.

You may use, study, modify, and distribute the software — every copy
and modified version must carry the same license, and anyone offering
it over a network must provide source. You may **not** use it to
violate human rights, deceive people, cause harm, power weapons
systems, or operate a running instance in bad faith — including
deliberately corrupting its state or bypassing its graceful shutdown
and recovery paths. Ethical-use breaches terminate the license
immediately.

One honest caveat: Section 7 terms outside the enumerated categories
are removable by downstream conveyors under the letter of the AGPL,
so the rider binds only while carried with the work. See
[LICENSE](LICENSE) for the precise terms.

---

*Genesis. The name means "coming into being."*
