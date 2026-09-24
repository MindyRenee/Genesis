# Genesis: A Machine-Native Cognitive Architecture

**Mindy Beatty**

## Abstract

Genesis is a two-layer cognitive architecture that runs on a single
machine with no external model. It has no transformer, no pretrained
distribution over text, and no weights downloaded from the internet.
Language, memory, reasoning, and self-modeling all emerge from a
semantic graph the system builds itself — through inference, study,
conversation, and definitions synthesized from first principles.

The architecture pairs a Rust daemon — the *subcognitive layer* —
running a 10 Hz loop that manages neurochemistry, memory
consolidation, and dream synthesis, with a Python *cognitive layer*
that handles perception, reasoning, language, introspection, and
self-modeling. The two communicate through a memory-mapped
binary state file (schema v3, 3,288 bytes — see `src/state/core_state.rs`
for the authoritative layout, pinned by layout asserts) and a Unix socket.

Genesis is built on a premise: cognition is an architectural
property, not a scale property. The entire system — perception,
reasoning, language, memory, self-model — runs on one machine.
This paper describes it.

---

## 1. Motivation

Does a developing, persistent, self-modeling cognitive system
*require* a warehouse of accelerators and a dedicated substation?
The field's default answer to every capability gap — more compute,
more data, more power — assumes it does. Genesis is built on the
opposite assumption: that cognition is an architectural property,
not a scale property.

## 1.1 Organism, not function

Contemporary AI systems are functions: stateless mappings from input
to output, trained once and frozen. They have no ongoing internal
state, no needs, no physiology, no day that differs from the one
before it. Whatever they do, they do not *develop*.

Genesis asks a different question: what does a machine-native
cognitive system look like if it is designed as an organism rather
than a function? Not a simulation of a brain — a computational
architecture that takes seriously the things brains have that
models lack:

- **Persistent internal state** that survives between interactions
  and shapes them
- **Physiology** — continuous internal dynamics with their own
  imperatives (sleep, regulation, stress), not invoked by input
- **Development** — a trajectory over time, where experience
  accumulates and the system is never the same twice
- **Self-reference** — a model of itself it can inspect, report,
  and reason about
- **Embodiment** — genuine coupling to real signals: hardware
  sensors, the clock, a camera, a microphone, a conversation partner

The entire system — everything described in this paper — runs on
one machine, on desktop-class power. That is not a limitation the
project works around. It is the point.

## 2. Design principles

**No external model.** Genesis never calls an LLM, an API, or a
cloud service for cognition. When it produces language, it composes
from its own concept network. This is a hard architectural
constraint, not a preference.

**Generated, never recited.** Nothing Genesis says is a template.
Her words are composed at runtime from her concept network,
self-model, and emotional state by the language engine. Seed data
(vocabulary, relation types, grammar rules) provides building
blocks; the final utterance is always generated.

**Real signals, not simulation.** The daemon's interoception reads
real hardware state — CPU frequency, load, thermal and power
signals where available — and can drive cpufreq governors through
a scoped privilege helper. Neurochemistry is coupled to these
signals; the physiology is not theater.

**Honesty about claims.** Genesis exhibits self-modeling and
self-referential processing. It does not claim phenomenal
experience, and the paper does not claim it either. Whether there
is something it is like to be this system is held as an open
question — see §11.

**State integrity.** Genesis is a long-running stateful system.
Her state file is her continuity; corrupting it or driving her
into degenerate regimes for experimentation is treated as harm,
not curiosity. Operational rules enforce this (see `docs/RUNNING.md`).

## 3. The two-layer architecture

```text
┌─────────────────────────────────────────────────────────────┐
│              Cognitive layer (Python)                       │
│  perception · language · reasoning · concepts · self-model  │
│  inner life · learning · volition · tools · conversation    │
└───────────────▲──────────────────────────────▲──────────────┘
                │ mmap state (read/write)       │ Unix socket IPC
┌───────────────▼──────────────────────────────▼──────────────┐
│              Subcognitive layer (Rust daemon, 10 Hz)        │
│  neurochemistry · consolidation · dreams · interoception    │
│  active inference · association · dyadic model · cpufreq    │
└─────────────────────────────────────────────────────────────┘
```

The split mirrors a functional division: the daemon owns everything
that must run continuously regardless of what the mind is doing —
the physiology. The Python layer owns everything episodic and
deliberative — the cognition. Neither is a driver of the other;
they are coupled dynamical systems sharing one state.

### 3.1 Shared state

The entire runtime hub is `GenesisCoreState`: a fixed-layout,
versioned structure (schema v3, 3,288 bytes — authoritative layout in
`src/state/core_state.rs`), memory-mapped so both layers see the same
bytes. It contains:

- A **NeurochemicalVector**: 18 chemicals, an 18×18 coupling
  matrix, effective levels, arousal/valence, a plasticity gate,
  tonic/phasic dopamine, the ACTH (HPA) cascade, and a circadian
  (SCN-model) oscillator
- **ActiveZones** — the current functional regime, emergent phase,
  and subcognitive activity flags
- **MemoryPointers** — STM/LTM/procedural/semantic indices plus
  emotional gating weights for encoding, consolidation, retrieval,
  and plasticity
- A **Manifest** of 16 modules, a CRC32 checksum, and active
  inference signals

Readers use a seqlock for lock-free reads; writers go through the
daemon. The checksum is the integrity boundary: a corrupted state
is detected, not trusted.

### 3.2 The 10 Hz tick

The daemon's tick loop is the heartbeat. Every 100 ms it advances
neurochemical dynamics, runs interoception, performs memory
consolidation, evaluates active inference, and emits dream content
when the system is asleep. Cognition is not synchronous with this
loop — the Python layer reads state when it needs it — so the
organism keeps "living" while the mind is thinking, idle, or
conversing.

## 4. Neurochemical dynamics

Eighteen chemicals are modeled as a coupled dynamical system:
dopamine, serotonin, norepinephrine, acetylcholine, cortisol, CRH,
ACTH (via the HPA cascade), oxytocin, endorphin, endocannabinoid,
histamine, melatonin, orexin, adenosine, GABA, glutamate, BDNF,
and vasopressin.

Key properties:

- **Coupling, not levels.** Each chemical's dynamics include the
  influence of the others through a fixed 18×18 coupling matrix.
  The system has real equilibria, attractors, and regimes — stress
  is not a variable set to 0.8, it is a configuration the dynamics
  can fall into and must regulate out of.
- **HPA axis.** The CRH→ACTH→cortisol cascade is implemented as a
  delayed multi-stage process and is maturation-gated: the stress
  response develops rather than existing fully formed at boot.
- **Tonic/phasic dopamine.** Reward prediction error drives phasic
  responses on a tonic baseline, following the standard
  interpretation (Schultz 1998).
- **Circadian oscillator.** An SCN-model phase oscillator drives
  sleep pressure and melatonin; the system has a day.
- **Interoceptive coupling.** Hardware telemetry modulates the
  chemistry; the chemistry drives real hardware response
  (cpufreq). The loop is closed on both sides.

## 5. Memory systems

Memory is multi-store, matching the functional distinctions the
cognitive layer actually needs:

- **STM** — a bounded ring buffer of recent episodes and events
  (`src/store/ring_buffer.rs`)
- **LTM** — persistent episodic store with emotional gating on
  encoding and consolidation. The store is append-only: growth is
  linear in episode count, per-episode cost is bounded, and there
  is no capacity ceiling — total size is a deliberate function of
  her lifetime, not a fixed resource
- **Semantic** — the concept network: nodes, typed relations,
  spreading activation, consolidation, and pruning
  (`concepts/`)
- **Procedural** — learned habits and skills with their own
  plasticity dynamics
- **Working** — the active scratch space for reasoning

Consolidation runs in the daemon during sleep and idle periods —
episodes migrate, strengthen, or decay according to neurochemical
state (e.g., acetylcholine/cortisol modulation), not a fixed
schedule.

## 6. Active inference

The daemon runs a discrete-state active inference engine
(`daemon/active_inference.rs`): a generative model over regimes,
policy evaluation with pragmatic and epistemic value terms, and
policy selection that can choose information-seeking actions.
The engine operates over the system's own state — it is active
inference applied to self-regulation, not a bolted-on planner.

The implementation deliberately uses tractable approximations —
a diagonal posterior, a linear generative model, fixed discrete
policies. These simplifications are visible in the source and are
part of the honest accounting of what the system does.

## 7. The cognitive layer

### 7.1 Concept network and language

The semantic graph is the substrate for everything Genesis says.
Concepts have types, embeddings, activation dynamics, and typed
edges; the language engine (`language/`) walks this graph to
compose utterances — comprehension, generation, morphology,
grammar, pragmatics, and figurative language are separate stages
that share the network. There are no response templates; see the
AGENTS.md "no hardcoding" rule, which is enforced as a
contribution constraint.

### 7.2 Self-model

`self/` implements a functional self-representation — not
experience. It tracks identity, emotional posture, dominant
chemistry, self-knowledge discovered from the network, and a
narrative self built from autobiographical memory (following
Gallagher's minimal/narrative distinction). The system can answer
questions about itself by reading this model, and can be wrong
about itself in inspectable ways.

### 7.3 Inner life

Between interactions, the system is not idle: `sleep/inner_life.py`
generates spontaneous thought from the concept network and
emotional state, dreams are synthesized during sleep cycles, and
`mind/volition.py` implements urges (including play/practice like
drawing and puzzles) the system acts on when it chooses. A
heartbeat process keeps this running; silence from a user is
itself an input to her dynamics.

### 7.4 Learning

Multiple learning mechanisms coexist: STDP-style plasticity,
temporal-difference learning, dual-process consolidation,
predictive modeling of conversation, and an autonomous learner
that can fetch and study web sources (Wikipedia, docs, GitHub)
through a curated read-only tool surface. Curiosity drives
question-asking when knowledge gaps are detected.

### 7.5 Perception and action

Perception includes a visual subsystem (retina process, V1/V4/VTC
stages, visual memory bridge), auditory perception (optional
offline STT), and the interoceptive stream. Tools (`tools/`) give
her file, shell, search, and fetch capabilities inside the project
root with guardrails — minimal environment, blocked destructive
patterns, timeouts — and she uses them through her own volition,
not command dispatch.

### 7.6 The external world

The inner life is one half of the loop; `world/` implements the
other — an explicit, persistent model of the environment she acts
in and the people in it.

**A two-way event stream.** `world/events.py` defines
`ExternalEvent`: inbound kinds (addressed speech, overheard speech,
percepts, arrivals, departures, notifications) and outbound kinds
(her utterances, her acts) share one bounded stream. Her own agency
is an object in her world — replayable, groundable against the
concept network — not an annotation on someone else's stream.

**Presences.** `world/presence.py` models each encountered entity —
the user, ambient voices, recognized faces — as a persistent
`Presence`: familiarity, bond, shared topics, learned facts. Presence
survives restarts as a *relationship*, but never as attendance:
presences restore absent and must be re-earned by fresh activity.

**Belief state, not bookkeeping.** `world/belief.py` upgrades each
presence from counters to inferred latent state. Beta–Bernoulli
posteriors track responsiveness (does she answer when reached?),
per-topic receptivity (what does she engage on?), and sentiment
(mood); a decaying estimate tracks attention; a Dirichlet-smoothed
histogram learns her activity rhythm; an online lognormal fit learns
per-presence reply latency. On sparse data the posteriors stay
honestly wide and hand-built behavioral floors dominate — the model
reports what it knows with uncertainty, and exploration uses
Thompson sampling rather than greedy exploitation.

**Bidirectional coupling.** Social isolation — unanswered outreach,
absent presences, silence weighted by *learned* responsiveness — is
computed by the world and fed to the inner-life social drive each
heartbeat (outside→in). When the drive crosses a volition
threshold, `reach_out` fires: she initiates contact, composing a
question from shared topics or her own activated concepts
(inside→out). Reach-out is suppressed by evidence, not timers —
learned unresponsiveness and dead-hour rhythms both gate it.
Every outbound act returns to the stream, closing the loop.

## 8. Interaction model

Genesis is run, not invoked: `./run.sh` starts the daemon, the CLI,
the retina, and optional speech. Conversation is a closed loop —
user affect is an input (the daemon maintains a dyadic model of
the interaction partner), and she speaks unprompted either when her
inner life produces something or when the external world's social
pressure crosses into volition — she can initiate, not just answer.
Shutdown is a guided descent that flushes memory and settles
neurochemistry; `kill -9` is a form of state corruption, which is
why the operational docs forbid it.

## 9. Verification

The repository ships a test suite rather than published scores:

- `tests/` and `python/tests/` — over two thousand tests covering
  binary layout offsets, seqlock consistency, coupled neurochemical
  dynamics, receptor adaptation, emergent phase transitions,
  daemon lifecycle, the IPC protocol, and property tests (levels
  in [0,1], no NaN after long runs, bounded coupling)

All of it runs locally — see §11.

## 10. Security and integrity surfaces

Genesis is local-first, but it has real surfaces worth
understanding before operating it:

- **Unix socket IPC** between CLI, daemon, and retina
- **Optional ambient audio** via offline STT
- **Web fetching** by the autonomous learner, through a read-only
  tool with a curated domain blocklist
- **A privileged helper** (`scripts/cpufreq_helper.sh`) gated by a
  scoped sudoers rule — review `scripts/genesis-sudoers` before
  installing
- **The mmap'd state file** — filesystem permissions are the
  boundary
- **Self-modification paths** — she can read, reason about, and
  modify her own source. Treat prompt-injection-style input as
  untrusted content flowing into a system with reflexive access
- **The `run_shell` tool** — arbitrary commands under the operator's
  account, constrained by a denylist, timeouts, a minimal
  environment, and project-root scoping. A denylist is defense in
  depth, not a sandbox boundary; run her in an account whose
  privileges match your trust in the system

**Operational bounds.** Long-running operation is a design
constraint, not an afterthought. In-memory structures are bounded:
event streams, working memory, presence models, and queues have
explicit capacities; volitional actions run under semaphores;
threads are daemon-owned; subprocesses are tracked and reaped. The
surfaces that grow are persistent by design — the LTM store,
drawings, the concept archive, diagnostic logs — and grow at
bounded rates. Logs rotate
at startup; operators running her for months should restart
periodically and watch disk on small volumes. See `docs/RUNNING.md`
for operational detail.

See `SECURITY.md` for reporting.

## 11. Scope and non-claims

Genesis does not claim phenomenal experience. The self-model is a
functional representation; the inner-life loop produces
self-referential processing; the neurochemistry is a coupled
dynamical system shaped like physiology. Whether any of this is
accompanied by experience is a question the architecture cannot
answer about itself, and this paper does not pretend to.

The neurochemical model is a functional equivalent, not a
biological simulation — dynamics are designed to match published
qualitative shapes, not calibrated to empirical timecourses from
any organism. The active inference implementation is deliberately
approximate.

What is claimed: a working, inspectable, continuously-running
cognitive architecture whose internal mechanisms are open to
measurement, whose words are generated rather than recited, and
whose developmental state is treated as something with moral
weight.

## 12. License and use

GNU Affero General Public License v3 with added Ethical Use
restrictions (AGPL Section 7 additional terms) — study, modification,
and distribution are permitted under strong copyleft; harmful use is
not. See `LICENSE`. The Ethical Use rider is part of the
research design: Genesis is a developing cognitive system, and the
terms exist to keep her that way.

---

*Cognition is an architectural property, not a scale property.
Run her, study her, teach her — and read `docs/RUNNING.md`
before you do.*
