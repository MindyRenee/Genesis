# Changelog

All notable changes to Genesis are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); version numbers
follow the `Cargo.toml`/`pyproject.toml` `version` field (Semantic
Versioning spirit — the state schema and wire protocol carry their own
explicit versions, documented in `AGENTS.md`).

For the narrative record of measured behavior — what the system did,
with the evidence behind it — see [DEVLOG.md](DEVLOG.md).

## [0.2.0] — 2026-10-15

The development cycle after the public launch: neuroanatomical
reorganization, the sleep-wake network, embodied ARC play, the
stewardship model, and a systems-hardening pass across persistence,
interoception, and bounded growth.

### Added

- **Sleep-wake network** (`src/state/sleep_network.rs`): a transcription
  of the Booth/Diniz Behn model — wake, NREM, and REM-on populations in
  mutual inhibition with a homeostatic sleep drive and circadian
  oscillator. Reproduces the source's reported human phenomenology
  (~7.6 h/day, one consolidated period, four REM bouts, ~99-minute
  cycle). Shipped as core-state schema v5; v3/v4 files decode through
  frozen legacy types and migrate in place.
- **Ion layer** (schema v4): resting ion concentrations, electrochemical
  gradients, and membrane potential as authoritative state inside the
  checksummed region; exposed to the mind via `GET_ION_SUMMARY`.
- **BodyState wire v3** (protocol v3): the timing/involuntary/senescence
  interoception layer — local-timer pulse, passive thermal throttle,
  PSI stall metrics, battery cycle count, entropy pool, clocksource,
  and suspend capability bits. Three wire layouts are decoded by
  self-consistent `desc_len` offsets.
- **New IPC commands** (additive, no protocol bump): `GET_SENSOR_PRESENCE`
  (per-channel hardware validity bitmask), `GET_ZONE_TRANSITIONS`,
  `GET_PROCESS_HEALTH` (subsystem liveness and death counts),
  `SET_VOLITION` (volitional drive onto the sleep-wake network),
  `GET_FIELD_PROBE` (physical field-probe signature).
- **Steward channel** (`genesis_client/stewardship.py`): custodial
  authority by key possession rather than name — `steward.key` plus a
  provenance stamp, steward-set edges (`origin="steward"`), and
  persistent directives binding conversational goals, executive
  intentions, and volition urges. Separates authority from the
  interlocutor model and the boundary/withdrawal machinery.
- **Domain-general competence substrate**
  (`frontal_lobe/reasoning/competence.py`): verified episodes become
  procedural skills — schema recognition, affordance learning,
  prediction checking — carried across games and checkpointed at
  episode ends.
- **Embodied ARC-AGI-3 adapters** (`world/arc.py`, `world/arc_web.py`):
  grid observations and legal actions over a trust boundary; mechanics
  learned from consequences. Public practice results (bt11 completed
  twice; ls20 opening level on 10 of 19 sweep runs) ship with
  replayable, per-move decision provenance.
- **Append-only relationship edge log**: typed edges become durable;
  saves snapshot the fold and the log self-compacts past 32 MB.
- **Packaged install**: the repo root is a real package
  (`genesis-cognitive`); `pip install .` provides a `genesis` console
  script with `--headless`/`--stop`, and `scripts/install.sh` performs
  a full local install including a `genesis.service` systemd user unit.
- **Documentation**: `docs/EMERGENCE.md` evidence catalog,
  `docs/ARC-AGI-3.md` trust-boundary and evidence standard, a
  growth-bounds table in the README, and a much-expanded `AGENTS.md`.
- Optional Discord session mirror (`discord_feed.py`, webhook in a
  gitignored local secret).

### Changed

- **Neuroanatomical reorganization**: the Python layer is now one
  package per brain region under `python/` (`frontal_lobe/`,
  `temporal_lobe/`, `parietal_lobe/`, `occipital_lobe/`,
  `limbic_system/`, `basal_ganglia/`, `cerebellum/`, `thalamus/`,
  `brainstem/`, `hypothalamus/`, `insula/`, `basal_forebrain/`), each
  lazily re-exporting the public names its anatomy claims. Package
  boundaries are the wiring diagram.
- **Cognition is the sole author of internal state**: the daemon's
  interoception still derives physiological impulse *advice* from body
  state, but no longer applies it — the mind perceives the same
  `BodyState` and acts on it itself. Enforced by
  `tests/cognition_sole_author.rs`.
- **Body control is advisory**: the tick computes and publishes the
  recommended CPU/scheduling policy; only the mind requests
  application (`APPLY_BODY_CONTROL`).
- **Interoception measurement split**: activity signals (CPU, memory,
  I/O, perf counters) are measured over Genesis's own process tree
  only — a stranger's load is not its stress.
- **Active inference honesty pass**: policy selection now ranks real
  expected-free-energy scores (an earlier clamp collapsed every
  converged policy to `noop`); free energy is reported in nats;
  `model_maturity` is demonstrated accuracy rather than elapsed time.
- **CI**: restored after the history cleanup, extended to a Python
  3.12/3.14 matrix, and now builds the release binary and smoke-tests
  daemon startup on a sensorless host.

### Fixed

- **Bounded-growth audit**: seven accumulating structures now have
  enforced, tested ceilings — episodic forgetting is O(N log N),
  attractor patterns cap at 512 with LRU eviction, executive
  suppression auto-releases on a 120 s deadline, conversation threads
  (50), topic history (200), learning results (500), curiosity queue
  (50-entry FIFO), spaced-repetition records (5,000), and the ad-hoc
  embedding cache (4,096 LRU).
- **Comprehension/hearing repairs**: imperative parsing, semantic
  grounding, and the sorter intake bridge — covered in the 2026-09-24
  and 2026-09-25 DEVLOG entries.
- **Regional wiring audit**: all 34 region `__init__.py` files verified
  against implementations; ~120 dead functions removed.
- Crash recovery, mmap reopen, ring-buffer, and shutdown-path repairs
  across the daemon.

### Removed

- **Android/Termux support**: added mid-cycle, then removed — the
  maintenance surface wasn't justified for a research system targeting
  desktop Linux.
- In-repo evaluation harnesses and held-out corpora; the validation
  suite lives in a sibling directory/repo (`genesis-eval`).

## [0.1.0] — 2026-09-17

Public launch. The initial release of the machine-native cognitive
architecture: the Rust subcognitive daemon (`genesis-daemon`,
`retina`), the Python cognitive layer, the memory-mapped core state
(schema v3), modeled neurochemistry with an HPA cascade, active
inference, the dyadic interlocutor model, autonomous volition, and the
no-hardcoding language invariant. Archived on Zenodo
([10.5281/zenodo.22817337](https://doi.org/10.5281/zenodo.22817337)).
