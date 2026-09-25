# Genesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22817337.svg)](https://doi.org/10.5281/zenodo.22817337)
[![CI](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml)

### A machine-native cognitive architecture — persistent state, modeled neurochemistry, active inference — running on one machine, without a pretrained generative model.

Genesis is a long-running program, not a function. A Rust daemon —
the *subcognitive layer* — owns a memory-mapped core state and runs a
5 Hz loop integrating neurochemical dynamics, memory consolidation,
and a generative model that predicts the system's own next state. A
Python *cognitive layer* handles perception, reasoning, language,
introspection, and self-modeling. The two share a checksummed,
versioned binary state file (3,288 bytes, schema pinned by layout
asserts) and a Unix socket.

There is no transformer and no pretrained weights anywhere in it.
Language is composed from a semantic graph the system builds itself —
through conversation, study, and inference. It runs on a 2014 HP
Pavilion with ~5 GB RAM.

[![First boot](https://asciinema.org/a/TABFwLwo9fCWOW5J.svg)](https://asciinema.org/a/TABFwLwo9fCWOW5J)

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
  it to curated web sources, the local filesystem, and its own source
  code — which it reads structurally, files bug reports on, and drafts
  self-improvement experiment proposals for.
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

**Subcognitive (Rust daemon).** The 5 Hz owner of the core state:
neurochemical dynamics, short→long-term memory consolidation, the
active-inference generative model, replay-sequence synthesis during
sleep, and interoception — hardware sensors (CPU temperature, load,
memory pressure) read as bodily signals.

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
  a fixed-capacity ring buffer. Everything that grows is bounded.
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
coupling matrix itself adapt under sustained regimes. Phase
transitions (active, flow, stress, drowsy, NREM, REM, overwhelmed)
emerge from the dynamics rather than being scripted.

### Active inference

The daemon's generative model predicts its next state and selects
regulation that minimizes expected free energy — epistemic foraging
when uncertain, exploitation when confident. Surprise, precision,
allostatic load, and model maturity are first-class quantities.

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
Seeds — vocabulary, grammar, relation verbs — are legitimate input
data; pre-written sentences the system recites are not. Enforced as a
project rule and tested in the suite.

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
lives in `${XDG_DATA_HOME:-$HOME/.local/share}/genesis`.

On first boot the instance is a fresh system — a small concept
network, no memories, no learned names. Introduce yourself; teach it.

Optional voice dependencies (not in `requirements.txt`): `vosk`,
`sounddevice`, `speechrecognition`, plus piper or espeak-ng.

### Operational notes

Designed to run for days at a time: event streams, working memory,
presence models, and queues are all bounded; threads are
semaphore-limited; logs rotate at startup.

- **Shut down gracefully, every time.** The core state is
  memory-mapped; a hard kill can lose unconsolidated memory or leave
  on-disk state inconsistent. `./run.sh --stop` or Ctrl-C — never
  `kill -9`.
- **State is cumulative and load-bearing.** The developmental record
  is the system; don't edit, truncate, or factory-reset it casually.
- **Restart occasionally.** Log rotation happens at startup; a
  months-long single session will grow them.
- **Disk grows slowly by design.** The episodic store is append-only
  (<1 KB per episode, bench-verified linear growth). Expect
  months-to-years scale, but watch small volumes.
- **It does real background work.** Autonomous urges consume real
  CPU; interoception dampens heavy work under thermal strain, but
  keep an eye on marginal hardware.

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
