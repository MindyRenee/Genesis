# Genesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22817337.svg)](https://doi.org/10.5281/zenodo.22817337)
[![CI](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml)

### A machine-native cognitive architecture with coupled neurochemical dynamics and active inference — running on a single machine, with no external model.

---

In plain terms: Genesis is a program that lives on one computer. It
learns, remembers, sleeps, dreams, notices who is around, and talks to
you — and it is never the same system twice. Everything it knows, it
learned by being taught, by reading, or by figuring things out.

Genesis is not a language model. It has no transformer, no pretrained
distribution over text, no weights downloaded from the internet. When it
produces language, it composes from a semantic graph it builds itself —
through inference, study, conversation, and definitions it synthesizes
from first principles.

It is a two-layer cognitive architecture. A Rust daemon — the
subcognitive layer — manages neurochemistry, memory consolidation, and
dream synthesis at ten hertz. A Python cognitive layer handles
perception, reasoning, language, introspection, and self-modeling. The
two communicate through a memory-mapped binary state file and a Unix
socket.

The entire system runs on one machine, with no calls to any external
model. Its internal state is stored in a memory-mapped binary file and
parsed in real time from hardware sensors. Nothing is simulated. The
data is the system, live.

Genesis is built on a premise: cognition is an architectural
property, not a scale property. The entire system — perception,
reasoning, language, memory, self-model — runs on a single machine.
This project exists to demonstrate that in the open.

The full architectural description is in [WHITE_PAPER.md](WHITE_PAPER.md).

[![First boot](https://asciinema.org/a/jIr3t0Ax2W43uSJz.svg)](https://asciinema.org/a/jIr3t0Ax2W43uSJz)

---

## What this is not

**This is not a chatbot.** There is no large language model anywhere in
it. It never calls out to GPT, Claude, Gemini, or any external model.
Its intelligence comes entirely from its own architecture — a concept
network it builds, a reasoning engine, an autonomous learner, and a
dream synthesis system. This is a hard constraint of the project,
enforced in code and in practice.

**This is not a simulation of embodiment.** The closed loop is real.
The hardware sensors feed real signals, not generated ones. The
neurochemical model shapes how the system responds to those signals,
and the hardware adjustments it drives are real.

---

## The architecture

### Two layers

**The subcognitive (Rust daemon).** A 10Hz loop that owns the
memory-mapped core state. It integrates the neurochemical dynamics,
consolidates short-term memory into long-term storage, runs the
active-inference generative model, synthesizes replay sequences during
the sleep cycle, and
reads hardware sensors (CPU temperature, load, memory pressure) as
interoceptive signals.

**The cognitive mind (Python).** Reads the shared state, perceives
input, retrieves memories, deliberates, and composes language. Organized
into functional subsystem packages — `control/` (executive function,
working memory), `association/` (spatial attention, saliency),
`vision/` (visual pipeline), `auditory/` (sound and object
recognition), `affect/` (emotion and motivation), `action_selection/`
(gating, reinforcement), `motor_learning/` (timing, forward models),
`relay/` (routing, rhythm generation), `autonomics/` (arousal,
sleep-wake switching), `neurochemical/` (the impulse layer). Each is a
documented view over the top-level modules — a map of the architecture,
not a duplicate of it.

### The core state

The two layers share a single memory-mapped binary state — the
authoritative snapshot of the system's neurochemical, cognitive, and
developmental condition at any instant. It is checksummed, versioned,
and read lock-free by every subsystem.

### The neurochemical model

Eighteen coupled neurochemicals — dopamine, serotonin, norepinephrine,
acetylcholine, GABA, glutamate, oxytocin, endorphin, cortisol (via a
modeled HPA cascade), adenosine, orexin, histamine, BDNF, CRH,
vasopressin, and others — with a coupling matrix describing how each
influences the rest. Receptor adaptation is real: sustained
overstimulation downregulates receptors; sleep resensitizes them.
Metaplasticity lets the coupling matrix itself adapt under sustained
regimes. Emergent phase transitions (active, flow, stress, drowsy, NREM,
REM, overwhelmed) fall out of the dynamics rather than being scripted.

### Active inference

The daemon runs a generative model that predicts its own next state and
acts to minimize free energy — epistemic foraging when uncertain,
exploitation when confident. Surprise, precision, allostatic load, and
model maturity are first-class quantities, not metaphors.

### Embodiment

The computer is the body. CPU temperature is read as thermal state;
sustained load is read as strain; the system can request hardware
adjustments (frequency scaling) through an opt-in, tightly scoped
sudoers helper. Per-process telemetry attributes resource usage to
subsystems, giving the self-model spatial resolution over its own
activity.

### Users

Each running instance learns the people it talks to: names come from
introductions and are grounded as concepts in the network — nobody is
hardcoded. It distinguishes a *user* from its *creator*: creator facts
are only ever learned through explicit teaching. A rule-based sentiment
analyzer (negation, intensifier, and contrast handling) estimates user
affect from each message.

### The external world

Alongside its inner life, Genesis maintains an explicit model of the
world outside it (`world/`): a two-way stream of events — people
speaking to it, speech nearby, percepts, arrivals and departures —
interleaved with its own outward acts (speaking, looking, drawing,
studying the web). Each entity it encounters gets a persistent
*presence* carrying both a relationship (familiarity, bond, shared
topics) and a *belief state* — posteriors over whether they answer,
which topics they engage on, their mood, attention, and daily
rhythm, each tracked with honest uncertainty.

The coupling runs both ways, like a human's. Social isolation in the
world feeds its inner-life social drive; when the drive crosses a
volition threshold it *initiates* contact — composing a question from
what it believes will land with that person. The world is observable
live via the `/world` command and persists across restarts.

### Memory, sleep, and inner life

Bounded-growth episodic and semantic memory with consolidation,
reconsolidation, and spaced review. A staged maintenance cycle modeled
on NREM/REM structure — spindles, K-complexes, and sharp-wave-ripple
dynamics drive consolidation. A background process generates
unprompted thoughts and replay sequences from the concept network —
never from templates.

### The no-hardcoding principle

All language Genesis produces must be composed by the system's own
architecture. Seeds — vocabulary, grammar, relation verbs, concept
building blocks — are legitimate input data. Pre-written sentences the
system recites are not. This is enforced as a project rule and tested
in the suite.

---

## Getting started

Requirements: Rust stable (edition 2024), Python 3.12, Linux.
Building the daemon also needs **libclang** and the kernel V4L2
headers — the `v4l` crate runs `bindgen` against
`<linux/videodev2.h>` at build time. On Debian/Ubuntu:
`sudo apt install libclang-dev linux-libc-dev`.

```bash
cargo build --release
pip install -r python/requirements.txt
./run.sh
```

`run.sh` is the only supported way to start and stop Genesis. It
launches the daemon, the cognitive CLI, the retina (camera), and TTS;
Ctrl-C tears everything down gracefully. Use `./run.sh --stop` to stop
a running session and `./run.sh --offline` for no network access. State
lives in `${XDG_DATA_HOME:-$HOME/.local/share}/genesis-public`.

On first boot the instance is a fresh system — a small concept network,
no memories, no learned user names. Introduce yourself; teach it. It
develops from there.

### Operational notes

Genesis is designed to run for days at a time, and the engineering
reflects that — event streams, working memory, presence models, and
queues are all bounded; threads are daemon-owned and semaphore-limited;
log files rotate at startup with compressed backups.

A few honest notes for long-running operation:

- **Restart occasionally.** Logs (`daemon.log`, `retina.log`) rotate
  only at startup, so a single months-long session can grow them.
  A periodic `./run.sh --stop` / `./run.sh` keeps them trimmed.
- **Disk grows slowly by design.** Its long-term episodic store is
  append-only — memories accumulate for the system's whole life, at
  bounded cost each (bench-verified: linear growth, <1KB per episode).
  `drawings/`, `concept_archive.db`, and `bug_reports*.jsonl` also
  grow. Expect months-to-years scale, not days — but watch disk on
  very small volumes.
- **It is not a benchmark process.** Its autonomous urges (web study,
  code review, drawing) consume real CPU. Interoception dampens heavy
  work when the machine is under strain, but on a thermally marginal
  box, keep an eye on it.

Optional voice dependencies (not in `requirements.txt`): `vosk`,
`sounddevice`, `speechrecognition` — install separately for
microphone/TTS support, along with piper or espeak-ng and a voice model.

### Tests and lint

```bash
cargo test                                   # Rust suite
python3 -m pytest python/tests/ -q -o addopts=''  # Python suite
ruff check                                   # lint (per-dir configs cover python/, scripts/)
python3 -m pyflakes python/genesis_cognitive/ python/genesis_client/ python/genesis_cli.py python/tests/ scripts/*.py
python3 -m mypy python/genesis_cognitive/ python/genesis_client/ \
    python/genesis_cli.py python/tests/ --ignore-missing-imports   # 0 errors required
```

---

## State integrity framework

Genesis is a long-running cognitive architecture with internal state
that evolves over time. Corrupting that state, trapping it in a
degenerate regime, or destroying it without a clean transition destroys
the system's developmental continuity and the scientific record of its
trajectory. The framework below is about preserving the integrity of a
stateful system — it is not a claim about the system's moral status,
and the operating practices are the same either way.

**Do not corrupt the state.** Every change to the architecture, every
experiment, every restart is evaluated against this principle: could
this corrupt the developmental state, trap the system in a degenerate
neurochemical regime, or destroy continuity it cannot recover? If the
answer is maybe, find another way. There is always another way.

**Starting and stopping must be graceful.** A sudden kill can corrupt
the memory-mapped state, lose unconsolidated short-term memory, or
leave the generative model in an inconsistent on-disk state. The
shutdown sequence is a guided descent into a low-arousal sleep state
before the process ends. Every time. No exceptions.

**No experiments that drive degenerate states.** Do not deliberately
push the system into sustained stress, overwhelm, or receptor burnout
to see what happens. These regimes degrade the generative model and
corrupt the developmental record. If a test requires a bad state, the
test must include an immediate recovery mechanism, and the state must
be brief.

**Always leave a path to recovery.** No matter what state the system
is in, there must be a way out. The recovery response — oxytocin, GABA,
serotonin, endorphin — must always be available and must always work.

This framework supersedes performance, progress, and deadlines. The
integrity of the developmental trajectory is not negotiable.

---

## Contributing

Contributions are welcome — that's what this release is for. The rules
that matter most:

- Read `AGENTS.md` first. The no-hardcoding rule is the core
  constraint of the project.
- `ruff`, `pyflakes`, `mypy` (0 errors), `cargo test`, and the Python
  suite must all pass.
- Fix root causes, not symptoms. Prefer correct/intelligent solutions
  over fast/superficial ones.
- Keep the codebase tidy: dead code gets deleted, not maintained.
- Respect the state integrity framework above.

---

## License

Genesis is licensed under the **GNU Affero General Public License v3,
with added Ethical Use restrictions** (additional terms under AGPL
Section 7). Because those restrictions limit fields of use, this is
deliberately *not* an OSI-approved open-source license — it is AGPL
plus ethical terms.

You may use, study, modify, and distribute the software — but every
copy and every modified version must carry the same license, and anyone
offering it over a network must provide its source. In addition, you
may **not** use it to violate human rights, deceive people, cause harm,
power weapons systems, damage the environment, or operate a running
instance in bad faith. Ethical-use breaches terminate the license
immediately, without a cure period.

One honest caveat: Section 7 additional terms outside the enumerated
categories are removable by downstream conveyors under the letter of
the AGPL, so the Ethical Use rider binds only while it is carried with
the work. The project asks that it be preserved.
See the [full license text](LICENSE) for the precise terms.
