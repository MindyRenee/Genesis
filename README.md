# Genesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22817337.svg)](https://doi.org/10.5281/zenodo.22817337)
[![CI](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml)

### A machine-native cognitive architecture — persistent state, modeled neurochemistry, active inference — running on one machine, without a pretrained generative model.

Genesis is a long-running program, not a function. A Rust daemon —
the *subcognitive layer* — owns a memory-mapped core state and
executes the neurochemical dynamics, memory consolidation, and
generative-model work the cognitive mind requests over a Unix socket.
A Python *cognitive layer* handles perception, reasoning, language,
introspection, and self-modeling. The two share a checksummed,
versioned binary state file (3,288 bytes, schema pinned by layout
asserts) and a Unix socket.

The daemon is reactive, not autonomous: it is a bus that carries
information and executes requests, and it never initiates work on a
schedule. The cadence comes from the mind's 1 Hz heartbeat, which calls
`ADVANCE_NEURO`, `CONSOLIDATE`, `ASSOCIATE`, `DREAM`, `READ_SENSORS`
and friends. Neurochemistry therefore advances at the heartbeat's
rate, not at an independent physiological rate — if the mind is
stopped, the physiology is too.

There is no transformer and no pretrained weights anywhere in it.
Language is composed from a semantic graph the system builds itself —
through conversation, study, and inference. It runs on a 2014 HP
Pavilion with ~5 GB RAM.

[![Live session](https://asciinema.org/a/C0K8v1df1AkuwHg5.svg)](https://asciinema.org/a/C0K8v1df1AkuwHg5)
[![Discord — live session feed](https://img.shields.io/badge/Discord-live%20session%20feed-5865F2?logo=discord&logoColor=white)](https://discord.gg/pq5qMjzQX)

## What it does

- **Sees and hears.** A shared-memory camera feed runs through a
  V1→V4→VTC predictive-coding hierarchy; it detects and recognizes
  faces and objects. A microphone feed is transcribed offline (Vosk)
  and ambient non-speech sound is processed by an auditory subsystem.
- **Learns by being taught.** Plain statements in conversation become
  typed edges in its concept network. One 15-paragraph lesson on games
  grew the network from 1122 to 1239 concepts with 72 new edges — and
  it asked its own follow-up questions.
- **Learns on its own.** Between conversations, autonomous urges send
  it to **the open web**, the local filesystem, and its own source
  code — which it reads structurally, files bug reports on, and drafts
  self-improvement experiment proposals for. There is no domain
  allow-list: `ALLOW_ALL_DOMAINS` is `True` and both allow-lists are
  empty, so the per-site approval flow is currently dead code. The one
  real filter is an adult/malware *content* filter on the `tools/`
  fetch path. Run with `./run.sh --offline` if you want the network
  closed. See [SECURITY.md](SECURITY.md) before pointing it at a
  machine you care about.
- **Practices tasks.** Puzzle specs dropped into its world are picked
  up by an internal urge and worked end to end — attempt, evaluation,
  feeling, consolidation — and skills transfer to harder tasks. A
  spatial reasoner searches a transformation DSL over grid scenes, on
  a domain-general competence substrate (sorter, sequence, relation,
  quantity, and classification families in development).
- **Reasons.** Analogy by structure-mapping, means-ends problem
  solving, multi-step planning with revision, belief revision,
  epistemic evaluation of claims, a drift-diffusion decision process,
  and theory of mind over the people it talks to.
- **Composes language.** A comprehension pipeline (roles, negation,
  pragmatics, figurative language) feeds a generative side that builds
  sentences from concept-graph traversal, morphology, and prosody —
  monitored by a self-editor before speaking.
- **Acts on its own drives.** Volitional urges — study, draw,
  meditate, explore, sleep, make contact — grow from internal and
  environmental state and compete under an executive gate, not a
  scheduler. An *act* urge goes further: it forms an intention from
  its own state and chains tool calls — read, analyze, search,
  sandboxed measurement — under a capability policy, then observes
  what comes back.
- **Expresses itself physically.** It draws: affective state drives
  composition and color on a real canvas artifact. It speaks aloud via
  TTS. It scaffolds and writes real Python projects from its concept
  network.
- **Models its people.** Persistent per-person *presences* carry
  belief states — posteriors over whether someone answers, which
  topics they engage on, their mood and rhythm — tracked with explicit
  uncertainty. Sustained isolation builds a social drive that makes it
  initiate contact.
- **Sleeps, dreams, and consolidates.** A staged cycle modeled on
  NREM/REM structure consolidates memory, repairs the concept graph,
  replays experience as dream sequences, and compresses growth —
  bounded so the system can run indefinitely.
- **Knows and narrates itself.** A global-workspace broadcast gives
  subsystems shared access to what wins attention; introspection
  exposes the actual deliberation trace; a narrative self-model and a
  persistent growth ledger keep continuity across its whole life.

The measured record for these claims is in [DEVLOG.md](DEVLOG.md).

## Architecture

### Two layers

**Subcognitive (Rust daemon).** The owner of the core state:
neurochemical dynamics, short→long-term memory consolidation, the
active-inference generative model, replay-sequence synthesis during
sleep, and interoception — hardware sensors (CPU temperature, load,
memory pressure) read as bodily signals. All of it is invoked by the
mind over IPC; the daemon has no loop of its own (it prints
"Reactive mode — mind-driven, no tick loop" at startup).

**Cognitive mind (Python).** Perception, memory retrieval,
deliberation, language composition, and self-modeling — organized
into functional subsystem packages (`control/`, `association/`,
`vision/`, `auditory/`, `affect/`, `action_selection/`,
`motor_learning/`, `relay/`, `autonomics/`, `neurochemical/`). Each is
a documented view over the top-level modules — a map of the
architecture, not a duplicate of it.

### Systems engineering

This is engineered for a 4.7 GB machine, not a datacenter, and the
low-level design reflects it:

- **One page of core state, zero copies.** The 3,288-byte state
  struct occupies a single OS page under `MAP_SHARED`; the kernel
  handles paging instead of the process holding heap copies.
- **Seqlock protocol.** Readers get an owned copy via sequence lock —
  no Rust reference into the shared mapping is ever created, so no
  aliasing with concurrent writers — and writer updates appear
  atomically.
- **Crash recovery on open.** Magic, schema version, and checksum are
  verified before use; version migrations are explicit, never silent.
- **Non-finite sanitization.** Dedicated primitives keep NaN/inf out
  of the continuous dynamics so a bad value can't poison the loop.
- **SDR/LogHD memory indexing.** The append-only episodic store is
  indexed by sparse distributed representations; short-term memory is
  a fixed-capacity ring buffer. Everything that grows is bounded; the
  bounds are enumerated under "Growth bounds" below rather than claimed
  away.
- **State as introspection surface.** Zones track what's attended vs
  background; a runtime manifest records which modules are live and
  what they're doing — the state file is observable, not opaque.
- **A second generative model in Rust.** The daemon runs a dyadic
  model of the *user's* affective state alongside its own —
  co-regulation computed in the subcognitive layer.
- **Zero-copy perception.** The retina binary feeds camera frames
  through shared memory; the Python layer reads, never copies.

The daemon is ~28k lines of Rust: two binaries (`genesis-daemon`,
`retina`), one IPC socket, no external services.

### The neurochemical model

Eighteen modeled neurochemicals — dopamine, serotonin, norepinephrine,
acetylcholine, GABA, glutamate, oxytocin, endorphins, cortisol via an
HPA-style cascade, adenosine, orexin, histamine, BDNF, and others —
coupled through a matrix describing how each influences the rest.
Modeled receptor adaptation downregulates under sustained
overstimulation and resensitizes during sleep; metaplasticity lets the
coupling matrix itself adapt under sustained regimes.

The underlying chemistry is a genuine 18-dimensional dynamical system
integrated with a semi-implicit (unconditionally stable) scheme, and
its arousal subsystem is genuinely bistable. The *mental phases*
(active, flow, stress, drowsy, NREM, REM, overwhelmed) are not however
emergent in the strong sense: they are read off that dynamics by a
fixed threshold classifier with per-phase hysteresis
(`compute_phase_with_hysteresis` in `src/state/neurochemical.rs`).
No ODE quantity sets any phase boundary. Two of the phases, NREM and
REM, go further — ACh is held out of the integration during sleep and
driven to a literal target value, so the sleep-stage ACh rebound is
scripted rather than simulated. Sleep onset is also gated harder than
the rest: in a 48-hour simulation with no input, the system spends 3%
of its time Active, 76% Drowsy, 21% NREM in one-tick bursts, and
**never** reaches REM without an external `/sleep` command. Treat
"emerges from the dynamics" as aspirational; the honest description is
"classified from the coupled dynamics by a fixed threshold rule, with
a scripted sleep-stage override."

### Active inference

The daemon runs a learned linear model of the system's own
neurochemical trajectory, updated with a Kalman filter, and scores nine
fixed regulation policies against a homeostatic set point. What is
real: the generative model, the posterior update, surprise, precision
weighting of observations, and allostatic load are computed and
persisted properly.

**The selection loop used to be inert, and was not.** Each policy's
expected free energy was clamped to `[0, 2]`, but EFE is a relative
score with an arbitrary zero, so a policy that beat the others by a
hair scored slightly negative — and a `0.0` floor mapped that to
exactly `0.0`. In the converged state (`precision` saturates within
~25 s, so the policy-independent uncertainty term goes to zero while
the epistemic term still exceeds the tiny expected surprise) *all nine*
policies clamped to `0.0`, the argmin returned the first minimum, and
index 0 is `noop`. The loop chose `noop` on every cycle, forever. The
bound is now a symmetric finiteness guard, which preserves the
ordering, and the softmax temperature is floored at the observed score
spread rather than a constant ~300× larger. Policy selection is
consequently live — which also means the characterisation below now
describes behaviour that actually happens.

What this section previously overstated, now stated accurately: the
scored objective is **not** an expected free energy in the formal
sense. Its "epistemic" term is the proper entropy-reduction
information gain — but for this model class (linear-Gaussian,
additive controls) that quantity is provably policy-independent, so
it shifts every policy's score equally and the ranking reduces to
the pragmatic cost: predicted distance from the homeostatic target.
At rest that elects `noop`; under deviation it elects whichever
policy the learned action model predicts will correct fastest. The
previous uncertainty-weighted novelty substitute was worse than a
neutral constant: inverted relative to true information gain and
10–100× the pragmatic term, it reduced selection to
ranking-by-novelty. Exploration now lives where it belongs, in the
precision-weighted softmax temperature. Two limits remain honestly
in force: the repertoire is nine fixed policies (no continuous or
precision-modulated actions, which is where a genuine epistemic
drive would come from), and early in life — before the action model
is learned — all policies predict identically and selection falls
through to `noop` while the low-precision branch explores.
`model_maturity` is now evidence-based as well: elapsed experience
is tempered by predictive fit and posterior certainty, so a
long-running but inaccurate model does not become trusted merely
because time passed.

The self-model and its feedback loop are worth the code, and the
vocabulary is now closer to earned. Three fixes landed: precision
seeks an inverse-variance target instead of latching at 1.0 off a
fixed threshold — saturation at rest is correct *and reversible*,
so sustained surprise above ~0.01 RMS re-opens the exploratory
branch; `model_maturity` is no longer a stopwatch but
`exp(−err/0.001)` over a ~500-tick error EMA, i.e. demonstrated
accuracy, which is what the cognitive mind was already treating it
as; and the ranking path contains no cross-unit comparison at all
(single pragmatic term), dissolving the old epistemic/homeostatic
unit mismatch rather than tuning it. The remaining unit mixing
(nats plus mean-square in the *published* EFE) is cosmetic: both
added terms are policy-independent, so they cannot move a ranking.

### Embodiment

Genesis treats the host computer as its physical body: thermal and
load sensors provide interoceptive input; frequency scaling is an opt-in,
tightly scoped hardware adjustment. Per-process telemetry attributes
resource usage to subsystems, giving the self-model spatial
resolution over its own activity.

### Users and the external world

Each instance learns the people it talks to — names from
introductions, grounded as concepts; nothing is hardcoded. The world
model (`world/`) is a two-way event stream: inbound events (speech,
percepts, arrivals) and its own acts (speaking, looking, drawing,
studying). Social isolation feeds an inner-life social drive; past a
volition threshold it initiates contact on its own. Observable live
via `/world`; persists across restarts.

### Memory, sleep, and inner life

Bounded-growth episodic and semantic memory with consolidation,
reconsolidation, and spaced review. The sleep cycle is staged on
NREM/REM structure — consolidation drivers modeled on spindle,
K-complex, and sharp-wave-ripple motifs. A background process
generates unprompted thoughts from the concept network — never from
templates.

### The no-hardcoding rule

All language Genesis produces is composed by its own architecture.
Seeds — vocabulary, grammar, relation verbs, narrative templates — are
legitimate input data; pre-written sentences the system recites are
not. The test is whether a string is a *building block the engine
composes from* or *the thing it says*.

Enforcement is currently partial and should not be over-trusted. The
rule is documented in `AGENTS.md` and one test
(`test_language.py::test_utterance_variability`) checks that a
composed response is not a single fixed string. There is no lint rule
and no test that would catch a newly added hardcoded sentence
anywhere else in the tree; a September 2026 audit found several
(`self/identity.py`'s 23-entry description table, two fallbacks in
`self/composer.py`, and a test-result sentence in
`cognition/code_tools.py`) that reached speech verbatim. The
underlying pattern — composing from the graph, passing structure to the
engine, and returning silence rather than a canned fallback when the
network is empty — is followed well nearly everywhere.

## Getting started

Requirements: Rust stable (edition 2024), Python 3.12, Linux. The
daemon build needs **libclang** and kernel V4L2 headers
(`sudo apt install libclang-dev linux-libc-dev` on Debian/Ubuntu).

```bash
cargo build --release
pip install -r python/requirements.txt
./run.sh
```

`run.sh` is the only supported way to start and stop Genesis. It
launches the daemon, the cognitive CLI, the retina (camera), and TTS;
Ctrl-C tears everything down gracefully. `./run.sh --stop` stops a
running session; `./run.sh --offline` disables network access. State
lives in `${XDG_DATA_HOME:-$HOME/.local/share}/genesis`. To mirror a
session to a Discord channel as a read-only feed, put a channel
webhook URL in `.genesis-discord` or export `GENESIS_DISCORD_WEBHOOK`.

On first boot the instance is a fresh system — a small concept
network, no memories, no learned names. Introduce yourself; teach it.

Optional voice dependencies (not in `requirements.txt`): `vosk`,
`sounddevice`, `speechrecognition`, plus piper or espeak-ng.

### Operational notes

Designed to run for days at a time: event streams, working memory,
presence models, and queues are bounded; threads are semaphore-limited;
`daemon.log` and `retina.log` rotate at startup. See "Growth bounds"
for the specific ceilings and the two known-unresolved cases.

- **Shut down gracefully, every time.** The core state is
  memory-mapped; a hard kill can lose unconsolidated memory or leave
  on-disk state inconsistent. `./run.sh --stop` or Ctrl-C — never
  `kill -9`.
- **State is cumulative and load-bearing.** The developmental record
  is the system; don't edit, truncate, or factory-reset it casually.
- **Restart occasionally.** Log rotation happens at startup; a
  months-long single session will grow them.
- **Disk grows slowly by design.** The episodic store is append-only
  (<1 KB per episode, linear growth). Expect months-to-years scale,
  but watch small volumes. The "bench-verified" claim this used to
  carry is not backed by anything in the repository — there is no
  benchmark harness here to re-run. The one structure that could have
  outgrown that estimate, the per-episode Python metadata, was
  unbounded and is now evicted (see Growth bounds).
- **It does real background work.** Autonomous urges consume real
  CPU; interoception dampens heavy work under thermal strain, but
  keep an eye on marginal hardware.

### Growth bounds

Everything that grows is bounded, and the bounds are now explicit rather
than assumed. An earlier revision of this file claimed the same thing
while seven structures grew without a cap; those were found and fixed.

| Structure | File | Bound |
|---|---|---|
| Episodic records | `memory/engine.py` | Forgotten records are evicted after a 1 h grace period. `forget()` also runs in O(N log N) — it was O(N²) with two full list copies per record, holding the GIL on the autosave/think threads. |
| Attractor patterns | `memory/systems.py` | Trimmed to 60% of the Hopfield capacity (~0.138·N), least-recently-retrieved first. Over-capacity networks converge to spurious attractors that were being returned as confident retrievals. Also trimmed on load, so an already-bloated state file self-heals. |
| Executive suppression | `memory/working.py` | 64 entries, each with a 120 s release deadline. Previously permanent: every topic ever focused became un-attendable, since nothing released them outside tests. |
| Conversation threads | `memory/working.py` | 50 retained (lifetime total still reported). Each holds full turn text and is not persisted. |
| Topic history | `memory/working.py` | 200 entries. |
| Learning results | `learning/autonomous.py` | 500 retained, oldest evicted; serialized in full on each autosave. |
| Curiosity queue | `learning/autonomous.py` | 200-entry FIFO, persisted. Also fixed from `pop(0)` (O(N)) to `popleft()` (O(1)). |
| Review records | `memory/spaced_repetition.py` | 5,000, evicting the highest-retention records. `stability` is capped at 7 days so a well-reviewed concept cannot become permanently undecayable. |
| Ad-hoc vector cache | `concepts/embeddings.py` | 4,096 dense vectors, LRU. |

Two further unbounded-growth findings were investigated and deliberately
*not* changed, because the fix is a design decision rather than a patch:

- `semantic.py` consolidation labels its edges with `origin="semantic"`,
  which is in `DERIVABLE_ORIGINS` and therefore rejected — so
  `relates_to`/`similar_to` facts extracted by the semantic layer have no
  durable representation anywhere. Per the edge-log rule this is
  technically correct, but an extracted proposition is earned, not
  recomputable geometry; this wants a non-listed origin.
- `concepts/embeddings.py` builds a dense 5000² SVD (~100 MB, O(n³))
  on a 4.7 GB machine. Correct, but the dominant memory spike.

### Tests and lint

```bash
cargo test                                   # Rust suite
python3 -m pytest python/tests/ -q -o addopts=''  # Python suite
ruff check                                   # lint
python3 -m pyflakes python/genesis_cognitive/ python/genesis_client/ python/genesis_cli.py python/tests/ scripts/*.py
python3 -m mypy python/genesis_cognitive/ python/genesis_client/ \
    python/genesis_cli.py python/tests/ --ignore-missing-imports   # 0 errors required
```

## Contributing

Contributions are welcome. The rules that matter most:

- Read `AGENTS.md` first — the no-hardcoding rule is the core
  constraint.
- `ruff`, `pyflakes`, `mypy` (0 errors), `cargo test`, and the Python
  suite must all pass.
- Fix root causes, not symptoms.
- Dead code gets deleted, not maintained.
- Treat running state as load-bearing, not disposable.

## License

**GNU AGPLv3 with added Ethical Use restrictions** (AGPL Section 7
additional terms). Because those restrictions limit fields of use,
this is deliberately *not* OSI-approved open source.

You may use, study, modify, and distribute the software — every copy
and modified version must carry the same license, and anyone offering
it over a network must provide source. You may **not** use it to
violate human rights, deceive people, cause harm, power weapons
systems, damage the environment, or operate a running instance in bad
faith. Ethical-use breaches terminate the license immediately.

One honest caveat: Section 7 terms outside the enumerated categories
are removable by downstream conveyors under the letter of the AGPL,
so the rider binds only while carried with the work. See
[LICENSE](LICENSE) for the precise terms.
