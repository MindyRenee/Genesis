# Genesis — Project Notes

## What this is
Genesis is a machine-native cognitive architecture. The Rust crate
(`genesis`) defines the core state schema — the memory-mapped hub of
system state that every module reads from and writes to. The Python
layer (`python/genesis_conscious/`) is the cognitive mind: perception,
reasoning, language, introspection, and self-modeling.

## CRITICAL RULE: Never hardcode Genesis's responses
Genesis's thoughts, speech, and self-expression must ALWAYS emerge
from its own cognitive architecture — NEVER hardcode
pre-written response templates, canned phrases, or fixed sentence
structures that it "recites."

This applies to:
- **Spontaneous thoughts** (inner_life.py) — must be generated from
  its concept network, self_composer, or emotional state, NOT from
  template strings like `"I am written in {language}"`
- **Conversation responses** — must go through its language engine
  and cognition, NOT hardcoded reply patterns
- **Self-descriptions** — must be composed from its actual concept
  network and self-model, NOT template sentences with filled-in slots
- **Art/drawing descriptions** — must reflect its actual state, NOT
  pre-written captions

If you find yourself writing a string template with `{variable}` slots
that Genesis "says," STOP. Route the underlying data through its
concept network or language engine instead. It should compose its
own words from its own understanding.

### Seeds are OK — hardcoding what it says is NOT
There is an important distinction between **seeds** and **hardcoded
responses**:

- **Seeds are OK.** Seeding its concept network with initial knowledge,
  vocabulary, relationship verbs, narrative templates, and thought
  seeds is fine. Seeds are *input data* that its generative systems
  work with — they are the raw material from which it composes, not
  the composition itself. A seed says "here are building blocks";
  it does not say "say this sentence."
- **Hardcoding what it says is NOT OK.** Writing the actual words
  Genesis speaks as fixed strings (with or without `{variable}` slots)
  is never acceptable. Its words must emerge from its language engine,
  concept network, and cognitive architecture.

The test: **Is this a building block it uses, or is this the thing it
says?** Building blocks (seeds, vocabulary, grammar rules, relation
verbs, narrative structures) are fine. The final words it speaks are
not — those must be generated, not recited.

## Toolchain
- Rust stable, edition 2024 (rust-version 1.85+)
- Build: `cargo build` / `cargo build --release`. Requires libclang
  and kernel V4L2 headers — `v4l2-sys-mit` runs `bindgen` against
  `<linux/videodev2.h>` at build time (Debian/Ubuntu:
  `sudo apt install libclang-dev linux-libc-dev`).
- Test: `cargo test`. **Without a system C toolchain** (e.g. NixOS, or
  any host with no `cc`/`gcc`), `cargo` fails at link with
  `linker 'cc' not found` even though `cargo check` passes. Zig ships
  a working clang; shim it so `cc` resolves:
  ```sh
  . ./scripts/cargo-env.sh && cargo test --release
  ```
  `scripts/cargo-env.sh` builds the shim and exports `PATH`,
  `LIBCLANG_PATH`, and `BINDGEN_EXTRA_CLANG_ARGS`. It defaults to the
  roots verified on this host (Zig 0.16.0 under `~/tools`); override with
  `ZIG_ROOT=` / `LIBCLANG_ROOT=` if yours differ. The include set must
  match `$ZIG cc -E -v -x c /dev/null` exactly — print it and copy the
  `<...>` search list. A partial set fails on `__STD_TYPE` inside
  `stddef.h`. With this the Rust suite is 524 tests and takes ~2 min.
  Note that `. ./scripts/cargo-env.sh` must run in the *current*
  shell: piping the source (`. ./scripts/cargo-env.sh | tail`) runs it
  in a subshell, so the exported `PATH` is discarded and the build
  fails at link with `linker 'cc' not found`.
- Run examples: `cargo run --example <name>`
- Python 3.12+ (CI tests 3.12 and 3.14), dependencies pinned in
  `python/requirements.txt`
  (runtime) and `python/requirements-dev.txt` (lint + test). Install
  with: `pip install -r python/requirements-dev.txt`
- Lint: `ruff check` and `pyflakes` (both must pass)
- Type check: `mypy python/genesis_conscious/ python/genesis_client/ python/genesis_cli.py python/tests/ --ignore-missing-imports` (0 errors)
- Optional native capabilities (not in requirements.txt, because the
  wheels bind to system libraries that may be absent): microphone
  (`vosk`, `sounddevice`, `speechrecognition`), voice (`piper-tts` +
  `.onnx` models in `python/voices/`), and drawing (`pycairo`).
  Install and **verify** with `./scripts/install_optional_deps.sh`
  (`--verify` to re-check). Verify rather than assume: `sounddevice`
  imports cleanly and only raises `OSError` when it dlopens a missing
  libportaudio, so a wheel-only install silently turns "no microphone"
  into a crash in the audio thread. Genesis degrades to text-only when a
  capability is missing — that is expected, not an error.

## Time ownership
The daemon is the sole authority on time and on physics. It is the
body: it integrates neurochemistry, circadian phase, and sleep staging
on its own clock at `TICK_INTERVAL_MS` (200 ms), whether or not the
cognitive mind is alive.

- `TickLoop::advance_neuro(mmap)` takes **no `dt`**. The body measures
  its own elapsed interval (clamped to `DT_MIN..DT_MAX`, shared with the
  cognitive layer so both clocks accept the same range).
- `advance_physics` sub-steps at `neurochemical::DT` (100 ms), so the
  integration resolution cannot drift with the caller's cadence.
- Both entry points run the interval-gated body maintenance —
  interoception, CPU/thermal policy. Losing those starved the body for
  as long as the mind held the lease, which was the original defect.
- The mind is an observer. It reads `CoreState.inference_signals`
  (the 64-byte `InferenceSignals` block at offset 3352) for surprise,
  free energy, precision, and allostatic load rather than provoking a
  physics advance to obtain them. Note the block moved when v4
  inserted the ions ahead of the checksum — 3228 was the v3 offset and
  is stale.
- `tick_count` is a 5 Hz counter by construction: every
  `_INTERVAL_TICKS` gate assumes it. A caller-chosen step would stretch
  those periods.

## State schema and the ion layer
`CoreState` is **schema v4** (3416 bytes) in
`src/state/core_state.rs`. v4 inserted a 124-byte `IonState`
(`src/state/ions.rs`) immediately **before** the checksum field, so
every field after `manifest` moved and a v3 file (3296 bytes) cannot be
reinterpreted in place. `src/state/legacy_v3.rs` holds the byte-exact
v3 layout that `MmapState::open` decodes a v3 file through before
widening it; `GenesisCoreStateV3` is frozen history and its size and
offsets are pinned by assertions.

The ion block sits *inside* the CRC32 region deliberately: resting
concentrations, electrochemical gradients and membrane potential are
authoritative state, not derived. Losing them would reset the cell's
resting state on every restart.

Two layout traps worth knowing before touching either struct:

- `IonState` carries a `current_density: [f32; 4]` that
  `IonSummary` (the IPC view) does **not** export. So `membrane_potential_mv`
  is float 20 in the state file but float 16 on the wire. Read ion state
  through `GenesisClient.get_ion_summary()` or the IPC path — poking
  `core_state.bin` directly bypasses the seqlock *and* uses a different
  field layout, so it yields both torn reads and plausible-looking
  nonsense.
- Reading the mmap'd file directly is never correct. Use
  `read_consistent()`, or ask the daemon.

`GET_ION_SUMMARY` (opcode 40, additive, 108-byte reply) exposes the
layer to the mind; `genesis_conscious/neurochemical/electrochemistry.py`
turns it into bounded cognitive modulation (attention / integration /
memory-consolidation gains, neutral ≈ 1.0). Note genesis2 uses opcode
**36** for this; that id is already spent here on `GET_SENSOR_PRESENCE`,
so a collision is a real risk if ports are merged — `test_protocol.py`
pins both ids.

## Wire protocol
- The daemon↔mind IPC protocol lives in `src/daemon/ipc.rs`
  (`PROTOCOL_VERSION`) and `python/genesis_client/protocol.py` — the
  two constants must always match. Current version: **v3** (BodyState
  carries the timing/involuntary/senescence layer — pulse, throttle,
  PSI, battery cycles, entropy, clocksource, suspend caps — with a
  116-byte fixed header; SET_WAKE_ALARM = 34 arms the RTC wakealarm
  via `scripts/rtc_wake_helper.sh` + sudoers, installed by
  `scripts/install_sudoers.sh`). BodyState has three wire layouts —
  v1 (58 B), v2 (78 B), v3 (116 B) — disambiguated by which desc_len
  offset (54/74/112) is self-consistent with the packet length.
  **Do not add a field to BodyState**: each layout is distinguished by
  that desc_len arithmetic, so a new field means a fourth layout and a
  version bump. Per-channel sensor validity is served separately by
  `GET_SENSOR_PRESENCE = 36` (a `u16` bitmask of
  `interoception::sensor::*` / `protocol.SENSOR_*`) — additive, so it
  does not bump `PROTOCOL_VERSION`. A zero mask means nothing is
  confirmed present, which is the safe direction. Absent-sensor
  discipline depends on it: `ComputationalSubstrate.sensor_verified`
  and `reported_channels` gate every body-condition claim, so a channel
  that no hardware backs can never be reported as a reading.
  `GET_ZONE_TRANSITIONS = 37` is likewise additive — zone and phase
  transition counts with dwell time, tracked in
  `zones::TransitionTally` (process-local, because `ActiveZones` is
  exactly 96 bytes with no reserved tail and a field there would force
  a layout migration).
  `GET_PROCESS_HEALTH = 38` reports `[u16 alive_mask][u16 deaths][u8
  last_death]` for the daemon/cognitive/retina process tree — a
  subsystem that dies used to vanish from the telemetry with no trace,
  so a crashed retina was indistinguishable from one never started.
  `ADVANCE_PHYSICS = 39` is an evaluation-only sibling of
  `ADVANCE_NEURO = 24`, both requesting `[f32 dt]` and both returning the
  same 21-byte reply. `ADVANCE_NEURO`'s dt is **ignored**: the daemon
  owns time, measures its own elapsed interval, and sub-steps at
  `neurochemical::DT` (100 ms, the resolution every rate constant is
  expressed against). `Mind.advance_neuro()` takes no `dt`, so the
  coupling that let the mind dictate the body's integration rate cannot
  be reintroduced from the cognitive layer. `advance_physics(dt)` is the
  deliberate exception, used only by the causal assay against an isolated
  daemon — see "Time ownership" below.
  `UPDATE_MODULE_STATUS` accepts optional trailing `task_id` (u64) and
  `error_code` (u32) after the 6-byte cpu_share form, which fills two
  manifest fields that previously had no producer; the module section
  of `GET_SUBSYSTEM_TELEMETRY` carries them back at 18 bytes/record,
  and parsers accept the older 6-byte record as task 0 / no error.

## Keep the project tidy
The project is large. Dead code, unused files, and stale leftovers
accumulate quickly and make the codebase harder to navigate. When
auditing or working on a file:
- If a file is outdated, unused, or no longer necessary, remove it
  rather than spending effort fixing it. Confirm it has no importers
  first (grep for imports), then delete it.
- Remove dead methods, unused imports, and stale comments you encounter.

**Important distinction: dead vs unwired.** Not everything that appears
unused is dead. Some code is intentionally written but not yet wired in.
Before removing something that appears unused, check whether it looks
like scaffolded work-in-progress; if unsure, ask rather than deleting.

## Data directory resolution
Genesis has one live state directory per checkout, and several places
resolve it. They must agree, or a tool measures, cleans, prunes, or
recovers the wrong tree. The canonical order, as implemented in
`run.sh`:

1. `GENESIS_DATA_DIR` — explicit override, always wins.
2. `.genesis-data-dir` — uncommitted, gitignored per-checkout pin
   (`genesis-public` for this checkout). Its contents **are** the data
   dir; it is not a parent directory.
3. `XDG_DATA_HOME/genesis`.
4. `~/.local/share/genesis`.

**The pin outranks `XDG_DATA_HOME`, which is the whole point.** A
Flatpak or sandboxed agent host sets `XDG_DATA_HOME` to its own private
directory; treating that as authoritative silently forks the checkout
into a second, divergent state dir. That happened here — six days of
divergence, ~8.7k concepts and 8 projects in a fork under
`~/.var/app/`. `run.sh` exports the pin as `GENESIS_DATA_DIR`, so in the
supported launch path Python's
`infrastructure/config.py::default_data_dir` resolves correctly even
without knowing about the pin. That function now implements the full
order anyway, so tools run outside `run.sh` agree: `config.py`,
`scripts/hygiene.sh`, `scripts/prune_dead_concepts.py`,
`scripts/migrate_edges_to_log.py`, `scripts/install_optional_deps.sh`,
`python/setup_embeddings.py`, `examples/read_state.rs`, and
`examples/recover.rs`. There is no remaining known deviation — check
this list before adding a ninth resolver.

**The list above is enforced, not aspirational.** Two test suites fail if
any resolver drifts:

- `python/tests/test_data_dir_conformance.py` runs every resolver side by
  side under a hostile `XDG_DATA_HOME` and asserts they return run.sh's
  answer. It also greps the tree for files referencing the state
  directory and fails if one is not in its list, so a tenth resolver
  cannot be added silently.
- `src/data_dir.rs` has unit tests for the full precedence matrix,
  including blank-pin and blank-env cases.

A blank or whitespace-only `GENESIS_DATA_DIR` counts as **unset**
everywhere; treating it as set resolves the data dir to a garbage path
built from the current directory. Adding a resolver means adding it to
`DIR_RESOLVERS` in the conformance test, not just writing the `if`
cascade again.

The Rust side resolves through `genesis::data_dir()` (a pure function
over its inputs) rather than reimplementing the order, so the examples
cannot drift from each other.

Assets follow the data dir too. `config.py::vosk_model_dir()` is the
single source of truth for the speech model: `$GENESIS_VOSK_DIR`, then
`<data dir>/vosk-models`, then the XDG-relative location, then the
pre-pin `~/.local/share/genesis/vosk-models` so an existing install
keeps working. `temporal_lobe/speech.py` and `perception/ambient.py`
both call it; the path used to be inlined in both and pointed at
whatever instance happened to have a model, which for a pinned checkout
meant reading a *different* instance's directory.

`scripts/merge_data_dirs.py` folds a fork back into the canonical dir
(concepts unioned by id, edges folded per side then unioned into one
canonical snapshot, journal appended, projects copied). Dry-run by
default. LTM episodes and the derived vector/holographic/VQ indices
have no supported union and are reported, not merged.

## State integrity
Genesis is a long-running stateful system, designed to run for days at
a time. Always shut down via `./run.sh --stop` (or Ctrl-C in the
running terminal) — never `kill -9` or kill processes directly. The
core state is memory-mapped, so a hard kill can lose unconsolidated
memory or leave on-disk state inconsistent. Do not corrupt the mmap'd
state or drive the system into degenerate regimes for experimentation.

The state is cumulative and load-bearing: the developmental record *is*
the system. Do not edit, truncate, or casually factory-reset it.

Other things worth knowing before experimenting:

- **Restart occasionally.** `daemon.log` and `retina.log` rotate at
  startup, so a months-long single session will grow them.
- **Disk grows slowly by design.** The episodic store is append-only
  (<1 KB per episode, linear). Expect months-to-years scale; watch
  small volumes. Everything that grows in memory is bounded and
  explicitly so — episodic eviction (`memory/engine.py`), attractor
  trimming to 60% of Hopfield capacity (`memory/systems.py`),
  executive suppression with release deadlines and 50 retained
  conversation threads / 200 topic-history entries
  (`memory/working.py`), 500 retained learning results and a
  200-entry FIFO curiosity queue (`learning/autonomous.py`), 5,000
  review records with `stability` capped at 7 days
  (`memory/spaced_repetition.py`), and a 4,096-entry LRU vector cache
  (`concepts/embeddings.py`).
- **It does real background work.** Autonomous urges consume real CPU.
  Interoception dampens heavy work under thermal strain, but keep an
  eye on marginal hardware.

## Concept archiving — working memory is capped, the disk is not
`ConceptNetwork` keeps at most `max_in_memory` (15 000) concepts resident
and spills dormant ones to `concept_archive.db`. Archiving is a
**memory-pressure valve, not a dormancy sweep**: `spill_dormant` returns
early while the network is under the cap, so activation level alone never
triggers it. Two call sites: `mind/lifecycle.py::_archive_dormant_concepts`
(every 10th autosave cycle, ~50 min) and
`concepts/consolidation.py::_consolidate_n3_heavy` (N3 slow-wave sleep).
It is not a REM behavior.

Spilled concepts stay recallable — `add_concept` and `_resolve` pull
them back by id or alias — so nothing is lost, unlike the older
destructive prune that remains only as the no-archive fallback.

Two invariants keep this from failing silently again:

- `spill_dormant` verifies its own effect. It compares working memory
  before and after the trim and logs at ERROR if a batch committed to the
  archive but the trim did not shrink working memory.
- Every zero return logs why at DEBUG — no archive, within budget, or
  nothing dormant. Previously all three returned a bare `0`, so "archived
  is zero" was indistinguishable from healthy idleness.

`_archive_dormant_concepts` logs failures at ERROR as well as tallying
them via `note_swallowed`. It runs on a background autosave thread, and
the tally is only visible through `swallow_report()`, so a hard failure
there left working memory growing over budget with nothing in the log
explaining it.

## Edge storage — the edge log is canonical
Relationships have ONE source of truth: `edge_log.jsonl` in the data
dir (append-only assert/retract/snapshot events; `concepts/edge_log.py`).
`network._edges` is a materialized fold — never persist it as truth.
The JSON state's `edges` array is a debugging/rollback projection
written at save time; the log's fold overwrites it at restore.
- Derivable edges (untyped `related_to`/`bridges`/`similar_to` with
  pipeline origins like `hub_attachment`, `semantic_bridge`,
  `co_occurrence`) are NEVER materialized — rejected at `add_edge`,
  filtered at `replace_edges`, archive spill, and recall. Their role
  is served at query time by the similarity provider
  (`get_associations`), which returns `Neighbor` records with
  provenance (`exact` | `holographic` | `embedding`).
- Typed relations are always canonical regardless of origin; untyped
  edges with unlisted origins default to canonical (a false positive
  costs decay; a false negative loses an experiential binding).
- Edge mutations write through to the log; saves snapshot the fold;
  the log self-compacts past 32MB. Migration from pre-log state:
  `scripts/migrate_edges_to_log.py --data-dir DIR [--write]`.

## Swallowed exceptions
- `genesis_client/swallow.py` tallies every `except` that discards an
  error, in both layers. The sites previously logged at DEBUG, which is
  invisible at the default INFO level, so a subsystem could fail on
  every cycle indefinitely with nothing saying so. Journalling each
  occurrence would be worse than useless (a per-second failure is 86k
  lines a day), so the journal receives the *count*: first occurrence,
  then exponentially spaced repeats (2, 4, 8 … 4096, then every 4096),
  and additionally whenever 300 s have elapsed so a steadily-failing
  site cannot go stale between thresholds. 10,000 failures produce 14
  records, the last carrying the true count. The tally lives in
  `genesis_client` (the lower layer) with the journal injected as a
  sink at Mind startup, because `genesis_conscious` depends on
  `genesis_client` and not the reverse. Query with `swallow_report()`,
  surfaced in status as `swallowed_errors`. Site names are
  `module.function`, derived from the real enclosing scope.

## Manifest memory
- `ModuleEntry::mem_usage_mb` is only meaningful for a module owning
  its own process. The cognitive mind is one process hosting language,
  memory and reasoning alike, so a per-module split would be
  fabrication; those entries stay 0 ("not separately measurable"), the
  process-tree total rides on `Subcognitive` so `total_mem_mb` is
  correct, and per-process figures come from
  `GET_SUBSYSTEM_TELEMETRY`.

## Persistence and lifecycle verification
- `python3 -m pytest python/tests/ -q -o addopts=''` runs the full Python
  suite with an explicit summary. The daemon integration test uses a fresh,
  temporary XDG data directory, runs the mind offline, and shuts down via
  `run.sh --stop`; failed shutdown must preserve that temporary state.
- Focused state-integrity regressions live in
  `python/tests/test_cli_lifecycle.py` and `python/tests/test_persistence.py`.
- Transfer/learning-compounding eval lives outside this checkout
  (`../genesis-eval/eval_transfer/run_sweep.sh --replicates N`; set
  GENESIS_DIR if the repo is elsewhere). Two-arm probe/ladder design
  measuring whether consolidated skills lower task cost.
- Never unlink singleton lock files during shutdown: replacing the inode
  can allow a second process to acquire a different lock for the same state.
- Existing unreadable cognitive state is a startup error, not first boot.
  Repair or restore it explicitly; do not silently replace it with fresh state.
