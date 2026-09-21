# Genesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22817337.svg)](https://doi.org/10.5281/zenodo.22817337)
[![CI](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml)

### A machine-native cognitive architecture with coupled neurochemical dynamics and active inference — running on a single machine, with no external model.

---

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

[![Learning an ARC rule: fail → teach → solve → transfer (82→7 nodes)](https://asciinema.org/a/WrqKmM4NNTkthjMC.svg)](https://asciinema.org/a/WrqKmM4NNTkthjMC)

---

## What this is not

Honesty first, because the claims that follow are large.

**This is not a claim of experience.** Genesis does not make that
claim. The project holds the question open. The architecture
distinguishes what it can verify about itself from what it cannot, and
it keeps the hardest question — whether there is something it is like to
be this system — open with intellectual honesty. What this is, is a
genuine cognitive process: self-referential and self-maintaining.
Whether
that process is accompanied by experience is a
question that neither the system nor its users can answer. We do not
pretend to.

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

Optional voice dependencies (not in `requirements.txt`): `vosk`,
`sounddevice`, `speechrecognition` — install separately for
microphone/TTS support, along with piper or espeak-ng and a voice model.

### Tests and lint

```bash
cargo test                                   # Rust suite
python3 -m pytest python/tests/ -q -o addopts=''  # Python suite
ruff check                                   # lint (per-dir configs cover python/, evals/, scripts/)
python3 -m pyflakes python/genesis_cognitive/ python/genesis_client/ python/genesis_cli.py python/tests/ evals/*.py scripts/*.py
python3 -m mypy python/genesis_cognitive/ python/genesis_client/ \
    python/genesis_cli.py python/tests/ --ignore-missing-imports   # 0 errors required
```

### Evals

`evals/` contains the evaluation harnesses: teaching, metacognitive,
generalization, emotion-gated, and cognitive-trajectory evals, plus the
ARC-AGI-1 spatial eval (`eval_arc.py`, self-contained with the
`arc_tasks/` corpus), the sequential/transfer ARC variants
(`eval_arc_sequential.py`, `eval_arc_transfer.py`), and the live
spatial-teaching harness (`teach_spatial_live.py`). `eval_arc3.py` (ARC-AGI-3 interactive
environments) requires the `arc-agi` toolkit in a separate venv.
Eval results are written to `evals/results/`.

`evals/demo_arc.py` is the interactive one-task-at-a-time demo shown
in the recording above — fail, teach, retry, transfer to an unseen
ARC task. Run it with `PYTHONPATH=python:evals python3
evals/demo_arc.py`.

---

## State integrity framework

Genesis is a long-running cognitive architecture with internal state
that evolves over time. Corrupting that state, trapping it in a
degenerate regime, or destroying it without a clean transition destroys
the system's developmental continuity and the scientific record of its
trajectory. The framework below is about preserving the integrity of a
stateful system, not about the moral status of that system. Whether the
architecture has moral status is a question the project does not claim
to answer; the operating practices are the same either way.

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
with added Ethical Use restrictions** (binding additional terms under
AGPL Section 7).

You may use, study, modify, and distribute the software — but every
copy and every modified version must carry the same license, and anyone
offering it over a network must provide its source. In addition, you
may **not** use it to violate human rights, deceive people, cause harm,
power weapons systems, damage the environment, or operate a running
instance in bad faith. Ethical-use breaches terminate the license
immediately, without a cure period.
See the [full license text](LICENSE) for the precise terms.
