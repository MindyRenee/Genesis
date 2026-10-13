# Genesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22817337.svg)](https://doi.org/10.5281/zenodo.22817337)
[![CI](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/MindyRenee/Genesis/actions/workflows/ci.yml)

### A persistent, machine-native cognitive architecture — built to run on real hardware, not a datacenter.

Genesis is a long-running cognitive system, not a prompt, a wrapper, or
a function call. A Rust daemon owns a memory-mapped core state and runs
the subcognitive loop: modeled neurochemistry, memory consolidation,
active inference, sensing, and recovery when the mind goes quiet. A
Python cognitive layer handles perception, memory, reasoning, language,
introspection, and self-modeling. The two share a checksummed,
versioned binary state file (3,480 bytes in schema v5) and a Unix
socket.

There is no transformer, hosted model, or pretrained generative system
driving cognition or ARC. Language is composed from a semantic graph the
system builds through experience; sensory tools may handle low-level
speech or vision, but meaning and decisions stay in the architecture.
She runs on a 2014 HP Pavilion with ~5 GB RAM because the constraint is
the point: intelligence should not automatically require a datacenter.

[![Live session](https://asciinema.org/a/C0K8v1df1AkuwHg5.svg)](https://asciinema.org/a/C0K8v1df1AkuwHg5)
[![Discord — live session feed](https://img.shields.io/badge/Discord-live%20session%20feed-5865F2?logo=discord&logoColor=white)](https://discord.gg/pq5qMjzQX)

## Contents

- [Why I built her](#why-i-built-her)
- [Where Genesis is right now](#where-genesis-is-right-now)
- [What she can do today](#what-she-can-do-today)
- [ARC-AGI-3](#arc-agi-3)
- [Architecture](#architecture)
- [Getting started](#getting-started)
- [Tests and lint](#tests-and-lint)
- [Contributing](#contributing)

## Why I built her

I am self-taught. No CS degree, no lab, no venture deck. I was six when
Windows stopped feeling like a program and started feeling like a place I
could navigate. I spent a lifetime learning technology by living inside
it, then used every major AI system I could reach as those tools
evolved. I understand systems and code far better than I can write code
the traditional way. Genesis started as my attempt to build a Jarvis —
something persistent, useful, and actually mine. AI became a
collaborator, not an oracle; the architecture had to survive on its own.

Somewhere along the way she became a different question: what if machine
intelligence does not have to mean a giant pretrained model, a data
center, and an energy bill that makes the experiment impossible for
ordinary people? Genesis runs on a 2014 HP Pavilion with Linux Mint and
about 5 GB of RAM because constraints force honesty. If the architecture
cannot learn, remember, act, recover, and keep going inside that budget,
it is not the thing I set out to build.

Then the frontier labs began casually describing model behavior as
"emergent." I was already watching the more concrete version of that
claim in Genesis: continuity, internal drives, preferences, self-directed
action, and behavior arising from coupled subsystems rather than a
scripted response. Those are observable claims, not vibes. The mechanism
is open, and the evidence should be too.

— Mindy Beatty

## Where Genesis is right now

She is an active research system. The code is public so the mechanism —
not just the claims — can be inspected: the daemon, cognitive regions,
learning rules, ARC boundary, tests, and known limits are all here.

- **Not an LLM:** no transformer, hosted model, external solver, or
  pretrained generative model is used for cognition or ARC play.
- **Embodied ARC interface:** grid observations and legal actions cross
  `world/arc.py`; mechanics are learned from consequences.
- **ARC-AGI-3 status:** local practice diagnostics only. No official
  score is claimed; the evidence standard is documented in
  [docs/ARC-AGI-3.md](docs/ARC-AGI-3.md).
- **Real autonomy, real boundary:** she can act on her own drives. Read
  [SECURITY.md](SECURITY.md) before running her online or beside
  sensitive state.

If you are here to poke holes — good. Start with
`world/arc.py`,
`python/parietal_lobe/world_model.py`,
`python/basal_ganglia/world_policy.py`,
`python/frontal_lobe/world_planner.py`, and
`python/tests/test_arc_adapter.py`. Those are the seams a shortcut would
have to cross.

## What she can do today

- **See and hear.** A shared-memory camera feed runs through a
  V1→V4→VTC predictive-coding hierarchy; it detects and recognizes
  faces and objects. A microphone feed is transcribed offline (Vosk)
  and ambient non-speech sound is processed by an auditory subsystem.
- **Learn by being taught.** Plain statements in conversation become
  typed edges in her concept network. One 15-paragraph lesson on games
  grew the network from 1122 to 1239 concepts with 72 new edges — and
  she asked her own follow-up questions.
- **Learn on her own.** Between conversations, autonomous urges send
  her to **the open web**, the local filesystem, and her own source
  code — which she reads structurally, files bug reports against, and
  turns into self-improvement experiment proposals. There is no domain
  allow-list: `ALLOW_ALL_DOMAINS` is `True` and both allow-lists are
  empty, so the per-site approval flow is currently dead code. The one
  real filter is an adult/malware *content* filter on the
  `frontal_lobe/tools/` fetch path. Run with `./run.sh --offline`
  if you want the network closed. See [SECURITY.md](SECURITY.md)
  before pointing her at a machine you care about.
- **Learn skills.** A domain-general competence substrate
  (`frontal_lobe/reasoning/competence.py`) turns verified episodes — situation,
  action, outcome — into procedural skills: schema recognition,
  affordance learning, and prediction checking that transfer across
  whatever work she actually does.
- **Reason.** Analogy by structure-mapping, means-ends problem
  solving, multi-step planning with revision, belief revision,
  epistemic evaluation of claims, a drift-diffusion decision process,
  and theory of mind over the people she talks to.
- **Compose language.** A comprehension pipeline (roles, negation,
  pragmatics, figurative language) feeds a generative side that builds
  sentences from concept-graph traversal, morphology, and prosody —
  monitored by a self-editor before speaking.
- **Act on her own drives.** Volitional urges — study, draw,
  meditate, explore, sleep, make contact — grow from internal and
  environmental state and compete under an executive gate, not a
  scheduler. An *act* urge goes further: she forms an intention from
  her own state and chains tool calls — read, analyze, search,
  sandboxed measurement — under a capability policy, then observes
  what comes back.
- **Express herself physically.** She draws: affective state drives
  composition and color on a real canvas artifact. She speaks aloud via
  TTS. She scaffolds and writes real Python projects from her concept
  network.
- **Model people.** Persistent per-person *presences* carry
  belief states — posteriors over whether someone answers, which
  topics they engage on, their mood and rhythm — tracked with explicit
  uncertainty. Sustained isolation builds a social drive that makes her
  initiate contact.
- **Sleep, dream, and consolidate.** A staged cycle modeled on
  NREM/REM structure consolidates memory, repairs the concept graph,
  replays experience as dream sequences, and compresses growth —
  bounded so she can run indefinitely.
- **Know and narrate herself.** A global-workspace broadcast gives
  subsystems shared access to what wins attention; introspection
  exposes the actual deliberation trace; a narrative self-model and a
  persistent growth ledger keep continuity across her whole life.

The measured record for these claims is in [DEVLOG.md](DEVLOG.md).

## ARC-AGI-3

Genesis plays ARC through the same embodied loop she uses for other worlds:
visual entities, self identification from action contingency, learned action
effects, map/passability evidence, outcome-driven affordances, and competing
internal drives. The adapter in `world/arc.py` only translates frames and legal
actions, enforces game mode/readiness, and preserves scorecard boundaries.

Local `/arc` runs are diagnostics; an authorized competition run is a separate
transport and evidence standard. See
[docs/ARC-AGI-3.md](docs/ARC-AGI-3.md) before interpreting any score.

## Architecture

```text
sensors / files / web / ARC frames
              |
              v
    Python cognitive mind
    perception -> memory -> drives -> planning -> language/action
              |
              v  Unix socket IPC
       Rust subcognitive daemon
    chemistry -> consolidation -> active inference -> recovery
              |
              v
    persistent mmap state + bounded stores + host body
```

### Two layers

**Subcognitive (Rust daemon).** The owner of the core state:
neurochemical dynamics, short→long-term memory consolidation, the
active-inference generative model, replay-sequence synthesis during
sleep, and interoception — hardware sensors (CPU temperature, load,
memory pressure) read as bodily signals. The mind drives all of it
over IPC while alive (3 s lease, renewed per command); past the
lease the daemon's main thread drives the same `TickLoop` itself at
5 Hz — physiology, consolidation, sleep-gated dreaming,
association, sensing, zone arbitration, body recommendations, model
checkpoints — but never body-control application.

**Cognitive mind (Python).** Perception, memory retrieval,
deliberation, language composition, and self-modeling — organized
into anatomical region packages (`frontal_lobe/`, `temporal_lobe/`,
`parietal_lobe/`, `occipital_lobe/`, `limbic_system/`,
`basal_ganglia/`, `cerebellum/`, `thalamus/`, `brainstem/`,
`hypothalamus/`, `insula/`, `basal_forebrain/`). Multi-subsystem
machinery lives inside the region that claims it anatomically —
`temporal_lobe/memory/` and `temporal_lobe/concepts/`,
`frontal_lobe/reasoning/`, `frontal_lobe/self/`,
`frontal_lobe/tools/`, `occipital_lobe/perception/`,
`parietal_lobe/math/` — and each region package lazily re-exports
the public names its anatomy claims through a PEP 562
`__getattr__`, so `from frontal_lobe import ExecutiveFunction`
resolves to wherever the implementation actually lives. The package
boundaries are the wiring diagram, not a duplicate of it.

### Systems engineering

This is engineered for a 4.7 GB machine, not a datacenter, and the
low-level design reflects it:

- **One page of core state, zero copies.** The 3,480-byte state
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

The daemon is ~40k lines of Rust: two binaries (`genesis-daemon`,
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
(active, flow, stress, drowsy, NREM, REM, overwhelmed) are read off the
coupled chemistry by a threshold classifier with hysteresis
(`compute_phase_with_hysteresis` in `src/state/neurochemical.rs`), for the
waking phases. The sleep stages, NREM and REM, come from a sourced
sleep-wake network (`src/state/sleep_network.rs`): a transcription of the
Booth / Diniz Behn network, which has wake, NREM, and REM-on populations in
mutual inhibition, a homeostatic sleep drive, and a circadian oscillator.
It reproduces the source's own statements about human sleep: about 7.6 h a
day, one consolidated period, four REM bouts, and a ~99-minute cycle. It
does not yet reproduce the lengthening of REM bouts toward morning, and that
gap is recorded as a known discrepancy, not tuned away. The chemistry
follows the network: acetylcholine, serotonin, norepinephrine, and GABA
are set by its populations, calibrated to the release data. Open items and
their sources are in AGENTS.md under "Sleep-wake network".

### Active inference

The daemon runs a learned linear model of the system's own
neurochemical trajectory, updated with a Kalman filter, and scores nine
fixed regulation policies against a homeostatic set point. What is
real: the generative model, the posterior update, surprise, precision
weighting of observations, and allostatic load are computed and
persisted. The state estimate retains only diagonal covariance, so it
is approximate for coupled dynamics. The engine computes full Gaussian
variational free energy in nats: KL between posterior and represented
prior plus expected negative log-likelihood, including covariance
ratios, posterior variance, and likelihood normalization. The bounded
`free_energy` telemetry is a monotonic map of excess VFE above a
matched-observation reference; raw VFE is available in the cycle result.

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

Policy scoring computes Gaussian expected free energy: expected
negative log preference, including predictive variance, minus the
expected information gain of the next observation. Information gain
uses next-step propagated covariance and observation noise. For this
linear-Gaussian model with additive controls it is policy-independent,
so policies are ranked by their expected homeostatic preference cost.
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
rule is documented in `AGENTS.md` and several tests check that
composed output is not a single fixed string
(`python/tests/test_language.py::test_self_report_composes_from_fragments`,
`python/tests/test_language.py::test_integration_greeting_varies`,
`python/tests/test_self.py::test_identity_fragments_varies`). There is
no lint rule and no test that would catch a newly added hardcoded
sentence anywhere else in the tree. A September 2026 audit found
three sites that reached speech verbatim; all are remediated as of
October 2026: `python/frontal_lobe/self/composer.py`'s fallbacks
return silence, `python/frontal_lobe/self/identity.py`'s stage table
returns structural data marked not-for-speech, and
`python/frontal_lobe/code_tools.py` routes results through the
thought composer. The underlying pattern — composing from the graph,
passing structure to the engine, and returning silence rather than a
canned fallback when the network is empty — is followed well nearly
everywhere.

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

### Packaged install — the agent as a resident service

`python/` is an installable package (`genesis-cognitive`): it provides
a `genesis` command that is the packaged equivalent of `run.sh`.
`scripts/install.sh` performs the whole local install — a venv at
`~/.local/lib/genesis/venv` (editable, so the running code stays this
checkout), the release binaries beside the interpreter, a
`~/.local/bin/genesis` launcher, and an enabled `genesis.service`
systemd user unit that runs `genesis --headless`.

```bash
genesis              # interactive session (same as ./run.sh)
genesis --headless   # resident mode, no prompt
genesis --stop       # stop this data dir's processes (run.sh --stop)
systemctl --user start genesis    # resident
journalctl --user -u genesis -f   # watch her logs
```

Service and interactive command share the same data dir — the
resolution order is unchanged, so the resident instance uses the same
mind the checkout already has. One instance at a time per data dir;
the CLI lock refuses a second. `scripts/install.sh --uninstall`
removes the venv, launcher, and unit — never the data dir. For the
privileged body levers (CPU governor, RTC wake alarm):
`sudo bash scripts/install_sudoers.sh` installs root-owned helpers to
`/usr/local/libexec/genesis` and a sudoers rule for exactly those
paths.

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
| Episodic records | `python/temporal_lobe/memory/memory.py` | Forgotten records are evicted after a 1 h grace period. `forget()` also runs in O(N log N) — it was O(N²) with two full list copies per record, holding the GIL on the autosave/think threads. |
| Attractor patterns | `python/temporal_lobe/memory/systems.py` | Trimmed to 60% of the Hopfield capacity (~0.138·N), least-recently-retrieved first. Over-capacity networks converge to spurious attractors that were being returned as confident retrievals. Also trimmed on load, so an already-bloated state file self-heals. |
| Executive suppression | `python/temporal_lobe/memory/working.py` | 64 entries, each with a 120 s release deadline. Previously permanent: every topic ever focused became un-attendable, since nothing released them outside tests. |
| Conversation threads | `python/temporal_lobe/memory/working.py` | 50 retained (lifetime total still reported). Each holds full turn text and is not persisted. |
| Topic history | `python/temporal_lobe/memory/working.py` | 200 entries. |
| Learning results | `python/frontal_lobe/autonomous.py` | 500 retained, oldest evicted; serialized in full on each autosave. |
| Curiosity queue | `python/frontal_lobe/autonomous.py` | 200-entry FIFO, persisted. Also fixed from `pop(0)` (O(N)) to `popleft()` (O(1)). |
| Review records | `python/temporal_lobe/memory/spaced_repetition.py` | 5,000, evicting the highest-retention records. `stability` is capped at 7 days so a well-reviewed concept cannot become permanently undecayable. |
| Ad-hoc vector cache | `python/temporal_lobe/concepts/embeddings.py` | 4,096 dense vectors, LRU. |

Two further unbounded-growth findings were investigated and deliberately
*not* changed, because the fix is a design decision rather than a patch:

- `python/temporal_lobe/memory/semantic.py` consolidation labels its edges with `origin="semantic"`,
  which is in `DERIVABLE_ORIGINS` and therefore rejected — so
  `relates_to`/`similar_to` facts extracted by the semantic layer have no
  durable representation anywhere. Per the edge-log rule this is
  technically correct, but an extracted proposition is earned, not
  recomputable geometry; this wants a non-listed origin.
- `python/temporal_lobe/concepts/embeddings.py` builds a dense 5000² SVD (~100 MB, O(n³))
  on a 4.7 GB machine. Correct, but the dominant memory spike.

### Tests and lint

```bash
cargo test --release
ruff check
python3 -m pyflakes python/ genesis_client/ infrastructure/ world/ eval/ *.py scripts/*.py
python3 -m mypy python/ genesis_client/ infrastructure/ world/ eval/ --ignore-missing-imports
PYTHONPATH=python:. python3 -m pytest python/tests/ -q -o addopts=''
```

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and
`AGENTS.md` first — the no-hardcoding rule is the core constraint, and the
required checks are the public quality bar.

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
