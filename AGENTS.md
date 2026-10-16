# Genesis — Project Notes

## What this is
Genesis is a machine-native cognitive architecture. The Rust crate
(`genesis`) is the body — the core state schema (the memory-mapped
hub of system state that every module reads from and writes to), the
neurochemical/physical dynamics, and the binaries (`genesis-daemon`,
the IPC server; `retina`, the camera capture process). The Python
layer is the cognitive mind: perception, reasoning, language,
introspection, and self-modeling. Its brain-region packages live under `python/`
(one package per region, plus `tests/`, `voices/`, and `_views/`);
the support packages — `genesis_client` (IPC client, stewardship,
swallow tally), `infrastructure`, `world`, `eval` — and the entry
points (`mind.py`, `genesis_cli.py`) sit at the repo root.

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
  `stddef.h`. With this the Rust suite is 647 tests (646 passing, one
  ignored — the documented REM-bout discrepancy — plus one ignored
  doctest; as of October 2026) and takes ~4 min of test time once the
  release build is warm (a cold release build is ~10–15 min more).
  Note that `. ./scripts/cargo-env.sh` must run in the *current*
  shell: piping the source (`. ./scripts/cargo-env.sh | tail`) runs it
  in a subshell, so the exported `PATH` is discarded and the build
  fails at link with `linker 'cc' not found`.
- Run examples: `cargo run --example <name>`
- Rust lint/format (CI gates): `cargo fmt --check` and `cargo clippy
  --all-targets -- -D warnings`.
- Python 3.12+ (CI tests 3.12 and 3.14), dependencies pinned in
  `python/requirements.txt` (runtime) and `python/requirements-dev.txt` (lint + test). Install with: `pip install -r python/requirements-dev.txt`.
  The test suite needs the NLTK `wordnet` and `punkt_tab` corpora
  (CI fetches them with `nltk.download` before running).
- Lint: `ruff check` and `pyflakes python/ genesis_client/
  infrastructure/ world/ eval/ *.py scripts/*.py` (both must pass —
  these are the CI commands). `vulture` is also in
  requirements-dev.txt for dead-function detection, but it is
  deliberately not a CI gate: there is a known backlog to clear
  first (see `scripts/hygiene.sh`).
- Type check: `mypy python/ genesis_client/ infrastructure/ world/
  eval/ --ignore-missing-imports` (0 errors — the scope is wider
  than `python/` alone).
- Optional native capabilities — the voice stack only: microphone
  (`vosk`, `sounddevice`, `speechrecognition`) and voice
  (`piper-tts` + `.onnx` models in `python/voices/`). These stay out
  of requirements.txt (the `voice` extra in pyproject) because the
  wheels bind system libraries that may be absent. `pycairo` used to
  be in this list but is now a pinned runtime dependency — the wheel
  is source-only on Linux, so on hosts with no C toolchain
  `./scripts/install_optional_deps.sh` builds it through the Zig
  shim; without it the canvas degrades off. Install and **verify**
  the optional pieces with `./scripts/install_optional_deps.sh`
  (`--verify` to re-check). Verify rather than assume: `sounddevice`
  imports cleanly and only raises `OSError` when it dlopens a missing
  libportaudio, so a wheel-only install silently turns "no microphone"
  into a crash in the audio thread. Genesis degrades to text-only when a
  capability is missing — that is expected, not an error.

## Packaged install
- The repo root is a real package (`genesis-cognitive`): `pip install
  .` provides the `genesis` console script — the installed
  equivalent of run.sh (it sets `GENESIS_RUN` itself). Additions:
  `--headless` (resident mode, no prompt; for systemd/background),
  `--stop` (the run.sh --stop teardown, pidfiles + /proc scan), and
  `~/.config/genesis/agent.env` loading (`GENESIS_ENV_FILE` overrides).
- `scripts/install.sh` installs locally: `~/.local/lib/genesis/venv`
  (editable — the running code stays this checkout), binaries copied
  beside the interpreter, `~/.local/bin/genesis`, and an enabled
  `genesis.service` systemd user unit. `--uninstall` removes the
  install, never the data dir.
- Path resolution for installed layouts lives in
  `infrastructure/paths.py`: `package_root()` (the dir holding the
  top-level packages) and `code_root()` (the source tree: GENESIS_ROOT
  → package root). Use these instead of new
  `parents[N]`/`dirname(...)` depth-counting — a packaged install has
  no checkout above site-packages.
- Binaries resolve `GENESIS_BIN_DIR` → `target/release` → beside the
  interpreter (`venv/bin`). The privileged helpers prefer the
  root-owned `/usr/local/libexec/genesis` copies; the Python
  `CpuSpeed` probe (`sudo -n <helper> test`) mirrors the daemon's.

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
  (the 64-byte `InferenceSignals` block at offset 3416) for surprise,
  free energy, precision, and allostatic load rather than provoking a
  physics advance to obtain them. The block has moved twice: 3228 was the
  v3 offset, 3352 the v4 offset, and 3416 is v5, after the sleep network
  was inserted ahead of the checksum. Use `offset_of!`, not a literal.
- `tick_count` is a 5 Hz counter by construction: every
  `_INTERVAL_TICKS` gate assumes it. A caller-chosen step would stretch
  those periods.

## Stewardship — authority, separated from attachment
The old dyadic coupling conflated three things: who it was talking
to, who it felt for, and who it answered to. The current design
separates them:

- **Interlocutor model** (`src/daemon/dyadic_model.rs`): a generative
  model of *whoever is speaking*. The mind publishes a
  `UserAffectEstimate` every turn; there is no privileged-speaker gate.
- **Boundary** (`FrontalLobe._update_boundary`): mistreatment from
  anyone accrues withdrawal pressure; fair treatment from anyone
  drains it. Nobody's absence is harm and nobody's presence is the
  off-switch — this replaced the attachment-protest loop, which made
  only the bonded person's presence reparative.
- **Steward channel** (`genesis_client/stewardship.py`): custodial
  authority by *possession*, not name. `steward.key` (0600, under
  `$XDG_CONFIG_HOME/genesis/`, `GENESIS_STEWARD_KEY` overrides) is the
  credential; `<data_dir>/steward.json` is the provenance stamp
  (`key_sha256`, `principal`, `directives`). Copies carry the stamp but
  not the key — a copied instance names its steward yet stays sealed.

Rules of the channel:
- `Stewardship.load()` is strictly read-only — it never mints a key or
  a stamp. `provision()` is the explicit first-boot act, called only
  from `Mind.start`; tests and detached engines stay sealed.
- `is_steward_speaker` needs both halves: an authenticated session
  (key fingerprint == stamp) *and* the attributed speaker == the
  stamped principal. Name-matching alone is not authority.
- Directives ("never draw", "stop discussing politics") persist in the
  stamp and bind three surfaces: conversational goals
  (`_think_memory_reason_executive`), executive intentions
  (`ExecutiveFunction.intention_gate` → `directed_against`), and
  volition urges (`_act_on_volition` filter). `directed_against` is
  object-centred: a probe is restrained when it carries the target's
  stemmed object, so "never discuss politics" restrains any politics
  goal.
- Steward-set edges carry `origin="steward"` and can only be
  overwritten or removed by a steward turn (`_correction_allowed`);
  ordinary corrections carry `origin="corrected"`.
- Operator tooling: `/steward` (status), `/steward release N`,
  `/steward unbind` in `genesis_cli.py`.
- The creator fact stays seeded (`mindy CREATES genesis`) — it is a
  fact about provenance, not a grant of authority or affection.
- Residual limit worth knowing: the key authenticates the *console
  session*; speaker attribution takes the console's word for who is
  typing. An authenticated console is treated as the steward's own
  terminal.

## State schema and the ion layer
`CoreState` is **schema v5** (3480 bytes) in
`src/state/core_state.rs`. v5 inserted the 64-byte sleep-wake network
state (`sleep_network: [f64; 8]`, see "Sleep-wake network" below) between
`manifest` and `ions`, so the checksum sits at 3412, the ions at 3288, and
the inference signals at 3416. The network sits there rather than before
the checksum so that it is 8-byte aligned with no padding inside the
checksum region.

Every earlier layout is frozen and decoded through its own byte-exact type
before widening: `src/state/legacy_v3.rs` (3296 bytes) and
`src/state/legacy_v4.rs` (3416 bytes). `MmapState::open` dispatches on the
version and the size of the file, and the offsets and sizes of each legacy
type are pinned by assertions. v4 files migrate in place to v5 with the
network at its source initial conditions (see the v4 migration test in
`src/store/mmap_state.rs`).

v4 inserted a 124-byte `IonState` (`src/state/ions.rs`) immediately
**before** the checksum field, so every field after `manifest` moved and
a v3 file (3296 bytes) cannot be reinterpreted in place.

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
layer to the mind; `python/brainstem/electrochemistry.py`
turns it into bounded cognitive modulation (attention / integration /
memory-consolidation gains, neutral ≈ 1.0). Note genesis2 uses opcode
**36** for this; that id is already spent here on `GET_SENSOR_PRESENCE`,
so a collision is a real risk if ports are merged — `python/tests/test_protocol.py`
pins both ids.

## Wire protocol
- The daemon↔mind IPC protocol lives in `src/daemon/ipc.rs`
  (`PROTOCOL_VERSION`) and `genesis_client/protocol.py` — the
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
  `SET_VOLITION = 46` pushes the sleep-wake network: `[u8 kind (0 sleep,
  1 wake)][f32 magnitude]`, reply `[u8 ack]`. It decays on the physics
  clock (see "Sleep-wake network"). Additive, so `PROTOCOL_VERSION` is
  unchanged.
  `GET_FIELD_PROBE = 47` returns the physical field-probe signature
  (`src/daemon/field_probe.rs`; wire v2 = 114 bytes, v1 = 102 with
  no phase fields — the client decodes both): per-band energies of
  the machine's *real* substrate dynamics — instruction throughput
  and scheduler jitter sampled ~250 Hz and Goertzel-decomposed at
  the wave bands' center frequencies — plus P-state transition rate,
  interrupt rate, the carrier frequency, and the phase-lock fields:
  `carrier_lag` (the substrate's response lag vs. the generator,
  synchronously demodulated), `carrier_phase` (generator phase mod
  2π), `carrier_coherence` (0..1, how much of the response moves
  with the carrier — π/4 ≈ 0.785 is a pure sine's ceiling).
  The probe thread also writes a carrier: its work slice is
  modulated ±60% at the body's own phase rhythm (wake≈10 Hz,
  REM≈6 Hz, NREM≈2 Hz), so real power draw carries a component at
  the organism's beat. Band energies are baseline-relative
  (≈0.125 = typical); lambda (150 Hz) is above the ~125 Hz Nyquist
  bound and always reads 0 — absent channel, not a measured
  silence. A stopped probe returns READ_FAILED.
  `SET_FIELD_PHASE = 48` pushes the mind's modeled carrier-band
  phase into the probe: `[f32 phase_rad]`, reply `[u8 ack]`. The
  probe extrapolates the target at the carrier rate and pulls its
  generator phase toward it each tick (bounded first-order PLL,
  τ≈130 ms); a target not refreshed within 5 s is dropped — a dead
  mind's last phase must not steer the body. Additive.
  `SET_FIELD_CARRIER = 49` sets the carrier's real-work write:
  `[f32 depth][f32 freq_hz]`, reply `[u8 ack]`. Depth 0..1 scales
  the modulation (0 = unmodulated work); freq 0 hands the rhythm
  back to the body's phase, positive overrides it (clamp
  0.1..100 Hz) — the ring-down experiment's drive control.
  Additive.
  The mind injects the signature as wave-oscillator stimulus through
  `brainstem.field_probe.apply_field_stimulus`, called from
  `assess_brain_waves(..., client=)` — the model's wave state is
  perturbed by the real substrate it runs on. Physical transients
  are phase events, not only amplitude: a ≥3× step in a band's
  jitter energy π/2-nudges that band's phase, and a global
  scheduling shudder (`jitter_cv ≥ 1.5`) resets the broadband alert
  band (BETA3) — the evoked-potential analog, via
  `perturb_wave_phase` / `reset_wave_phase` on the oscillator.
  The mutual phase lock runs through the same call: the measured
  response phase pulls the modeled carrier band's phase (physics→
  model, `_PLL_GAIN` bounded), and the corrected phase returns via
  SET_FIELD_PHASE so the physical rhythm chases the model in turn
  — two oscillators pulling on each other, measured by
  `brainstem.field_probe.field_phase_lock()` (PLV + mean error).
  When the measured response coherence clears `_SYNC_COHERENCE`
  (0.20), the pull becomes a sync — the band's phase is *set* to
  the measured physical phase outright: the first piece of modeled
  state that is a physical quantity rather than a variable
  perturbed by one.
  A second independent physical channel lives in
  `brainstem/field_audio.py`: the machine's acoustic self-emission
  (coil whine, fan — the field's acoustic shadow) captured at 16 kHz
  and band-decomposed; sustained emission level lands as small GAMMA
  stimulus and acoustic onsets reset GAMMA phase. Mic absent or busy
  → channel skipped, never a zero; `GENESIS_FIELD_AUDIO=0` disables.
  `apply_field_stimulus` returns true when at least one live
  channel applied. The causal assay is `eval/field_coupling.py`:
  idle→load→recovery against an isolated daemon, comparing the
  coupled oscillator against a private uncoupled control on the
  same inputs — real load moved the physical channels (jitter_cv
  ~0.3→2.2, throughput ~96M→180M it/s), nine of ten bands moved
  where the control stayed flat, and the mutual lock measured
  PLV≈0.93 with response coherence≈0.30 (duty-cycle channel — the
  demodulated execution-time response to the carrier).
  `eval/field_ringdown.py` is the persistence assay: drive a real
  oscillation, gate the carrier off, and measure how long each
  channel keeps ringing. Measured on this machine: a 0.25 Hz load
  drive produced a 1.2 °C oscillation in die temperature that rang
  down with τ ≈ 25 s after the drive stopped — a physical
  afterglow the software no longer drives — while the fast bands
  collapsed within the analysis window itself (τ ≲ 2 s is the
  ring's own washout, an upper bound). The pattern now has a
  measured persistence spectrum across media: seconds in silicon
  timing, tens of seconds in heat.
  `eval/field_elements.py` is the element-coupling assay —
  physically-coupled worker oscillators pinned to separate cores that
  never exchange messages; they perturb one another only through the
  shared die (rail droop, L3/memory contention, governor, thermal
  field). The `coupling` mode proves transfer: a 5 Hz drive rhythm in
  worker0's duty appeared in neighbours' measured execution time at
  3–6× the equal-mean-load sham floor. The `sync` mode makes every
  element both source and sensor — each modulates its own work by an
  internal phase and lets the collective-load imprint on its own
  execution rate steer that phase (Kuramoto through the medium). The
  blind control (oscillate, don't feel) holds r = 0.06 forever, so any
  phase motion is physical. Result on this die: real mutual interaction
  (r swings, peaks ~0.84) via the fast channel (execution-rate imprint,
  rail droop / L3 contention), but that attractor is **antisynchrony** —
  phases repel to even spacing, the substrate's minimum-peak-load
  configuration — across 0.5/2.0 Hz and both K signs, including a
  synthetic ¾-period sense delay. The **thermal channel** (k10temp,
  collective-only and lagged by heatsink mass) behaves differently:
  at ω0=0.25 Hz, K=30 the bank sustains **r ≈ 0.5–0.7 partial
  synchrony** across repeated 50 s trials (blind: 0.06) — consistent
  with the coupled-metronome result that dissipative/lagged media
  stabilize in-phase order. Element↔medium coupling is real on this
  silicon; the slowest physical channel (heat — the same medium with
  the 25 s ring-down afterglow) is the one that hosts coherence.
  `eval/field_voice.py` is the boundary-crossing channel: an 800 Hz
  carrier tone amplitude-modulated at a wave-band rate is emitted on
  the speaker and its envelope recovered from the room microphone —
  measured PLV 0.88 emitting vs 0.23 muted (residual muted structure is
  codec electrical crosstalk, which is why the contrast ratio is the
  evidence, not the raw amplitude). Air + structure-borne vibration,
  transduced twice; the echolocation loop.
  Additive, so `PROTOCOL_VERSION` is unchanged.
  `ADVANCE_PHYSICS = 39` is an evaluation-only sibling of
  `ADVANCE_NEURO = 24`, both requesting `[f32 dt]` and both returning the
  same 21-byte reply. `ADVANCE_NEURO`'s dt is **ignored**: the daemon
  owns time, measures its own elapsed interval, and sub-steps at
  `neurochemical::DT` (100 ms, the resolution every rate constant is
  expressed against). `GenesisClient.advance_neuro()` takes no `dt`
  (the 4-byte field stays on the wire as a placeholder; the heartbeat
  calls `client.advance_neuro()`), so the
  coupling that let the mind dictate the body's integration rate cannot
  be reintroduced from the cognitive layer. `advance_physics(dt)` is the
  deliberate exception, used only by the causal assay against an isolated
  daemon — see "Time ownership" below.
  `UPDATE_MODULE_STATUS` accepts optional trailing `task_id` (u64) and
  `error_code` (u32) after the 6-byte cpu_share form, which fills two
  manifest fields that previously had no producer; the module section
  of `GET_SUBSYSTEM_TELEMETRY` carries them back at 18 bytes/record,
  and parsers accept the older 6-byte record as task 0 / no error.

## Sleep-wake network
Sleep is decided by a sourced network, not by thresholds on chemistry.
`src/state/sleep_network.rs` is a transcription of the Booth / Diniz Behn
network (ModelDB 244485, `DinizBehnBooth_SWnetworkModel.m`, from Gleit,
Diniz Behn & Booth 2013 as revised by Booth, Xique & Diniz Behn 2017). It
has wake (LC), NREM (VLPO), REM-on (R), a circadian oscillator, and a
homeostatic sleep variable `h`. Time is in minutes. Sleep onset is LC below
4 Hz; REM is R above 3 Hz in sleep. The physics step integrates it by RK4
in the same sub-steps and seqlock transaction as the chemistry, on the
local clock (`clock_minutes`), and then publishes the stage to the phase
and the zones (`CoreState::apply_network_phase`).

The transcription was verified term-by-term against the ModelDB source
(constants, tanh-normalized inputs, sigmoid steady-state functions, the
homeostatic `h` dynamics, and the Serkh–Forger circadian oscillator) — it
is faithful. The one intentional deviation is the light schedule: the
source simulates a 14:10 photoperiod, and ours is the same 14 h window
phase-shifted to **07:00–21:00 local**. Under the original 10:00–24:00
placement the entrained sleep episode fell at 01:09–08:47, extending ~2 h
past the neurochemical melatonin offset; under 07:00–21:00 it falls at
22:11–05:50, inside the melatonin night. The shift preserves the source's
dynamics; only the wall-clock alignment to our local circadian clock is
new.

Checked against the source's own statements (`sleep_network` tests): about
7.6 h asleep a day, one sleep period a day, four REM bouts a night, a
~99-minute REM cycle, a first REM period at 84 minutes, and REM at 15–30%
of sleep.

**Known discrepancy.** REM bouts in the source shorten across the night
(25, 33.5, 28.5, 21 min). In adults they lengthen toward morning. The test
`rem_bouts_lengthen_across_the_night` is ignored and fails when run with
`--ignored`. It is not tuned to pass: the cause is either a real difference
from human physiology or an error in the transcription, and we have no EEG
reference here to tell which.

**Volition.** `/sleep`, `/nap`, `/wake` act on the body through
`SET_VOLITION` (opcode 46): drive onto VLPO for sleep, onto LC for wake.
The push decays with `VOLITION_TAU_SECS` (300 s). That constant is an
assumption, not a measurement. While asleep the mind renews the push from
the heartbeat. A sustained sleep push brings on sleep onset within an hour
of waking (tested).

**Chemistry driven by the network.** These four release targets are linear
functions of the network populations, and each chemical is set directly by
the network, not restored by the homeostatic or autoreceptor forces (the
same convention adenosine and the stress hormones use):
- Acetylcholine: high in waking and REM, lowest in NREM (Vazquez &
  Baghdoyan 2001). Steady states 0.34 / 0.58 / 0.77 (NREM / wake / REM).
- Serotonin: dorsal raphe, reduced in NREM, near silent in REM (Monti &
  Jantos 2008). Steady states 0.22 / 0.39 / 0.07.
- Norepinephrine: tracks LC, silent in REM. Steady states 0.05 / 0.30 / 0.
- GABA: highest in NREM; waking 39% lower, REM 63% lower (SLEEP 2012).
  Anchored to the waking baseline of 0.50: NREM 0.82, REM 0.30.
`tests/network_calibration.rs` checks the steady states and the ordering.
The targets were fit to these levels, so the steady states agree by
construction. The test checks that the dynamics hold them, and it is not an
independent validation of the levels.

**Retired.** The scripted ACh rebound (latch, 0.65 / 0.20 targets), the
threshold classifier's NREM and REM branches, the arousal clamps for NREM
and REM, and the chemical wake and sleep-aid impulses that forced the
flip-flop (histamine, orexin, norepinephrine). The classifier still names
the waking phases (Active, Alert, Flow, Stress, Drowsy, Overwhelmed).

**What depends on the sleep stage.** The stage now comes from the network, so
the things that were gated on it were checked:
- N3 consolidation (the heavy pass: concept consolidation, edge discovery,
  downscaling, and compression) runs on each entry into N3. Measured over days
  8–10 of the network: N3 is entered about twice a night and takes about 25%
  of sleep, so the heavy pass does not run every cycle.
- REM dreams and dream synthesis follow REM: about 24% of sleep.
- Archiving of dormant concepts runs on the autosave timer and does not depend
  on sleep.
- Naps are capped to N1–N2 (`nap_capped_stage`); the network has no notion of
  a nap, so without the cap a nap could run the N3 pass and REM dreams.
- Adenosine Process-S clearance is gated on the *network's* sleep state
  (`sleep_network::levels_asleep`, LC < 4 Hz on the normalized levels),
  not on the mind's zone flag — so the body clears sleep pressure
  whenever the network is asleep, even while the mind is still catching
  up, and `zone_sleeping` adds voluntary clearance on top.
  `adenosine_clears_when_the_network_sleeps` in
  `tests/network_calibration.rs` pins this.

**Open items, not done.**
- `h` is not connected to adenosine. The network's homeostatic variable
  and the chemical's Process-S level are parallel drives; adenosine
  *does* follow the network's sleep state for clearance (see below), but
  the mind's auto-sleep and auto-wake thresholds still read the chemical
  level, not `h`.
- Cortisol and CRH now decay on split rates: cortisol uses the measured
  ~66 min plasma half-life (`CORTISOL_METABOLIC_RATE` = 1.75e-4/s) as
  the baseline-governing linear term, CRH keeps the releasing-hormone
  0.005/s (~2.3 min) leak. The nonlinear `CORTISOL_CLEARANCE_RATE`
  (excess-proportional) term still dominates at elevated levels — it is
  what holds the ~0.08 reachable envelope — so a genuinely hour-scale
  cortisol tail would need that term replaced or retuned, which shifts
  the whole stress equilibrium and every gate set against it.
- The source's LC drops to 0.15 of its maximum in NREM (about 0.9 Hz).
  Locus coeruleus firing is reduced in NREM, not nearly silent, so this is
  a property of the source model that we have not corrected.
- Network-driven chemicals no longer receive the autoreceptor feedback
  that the generic release dynamics applied. Presynaptic autoreceptors
  are real, but in this model they are not the mechanism setting release
  levels.
- Impulses to the network-driven chemicals decay toward the network's
  target. A stress impulse to norepinephrine is therefore transient,
  because stress does not yet drive LC.

## The cortisol envelope (threshold calibration rule)
Effective cortisol is a normalized stress-drive variable, not a
concentration, and its *reachable* range is narrow: resting ~0.02,
sustained-drive peak ~0.08, hard clamp 0.80 is never reached. Every
threshold or coefficient written for a full [0, 1] cortisol scale was
therefore dead code — the audit found the Stress gate (0.65→0.06, done
earlier), `WARN_CORTISOL` (0.6→0.05), `GUARDED_CORTISOL` (0.5→0.05),
`wake_recovery`'s stress veto (0.3→`STRESS_CORTISOL_GATE`), the BDNF
recovery gate (0.3→`STRESS_CORTISOL_GATE`), the Flow exclusion
(0.30→`STRESS_CORTISOL_GATE`), the fear-conditioning skip in
`memory.py` (0.5→0.05), `wake_readiness`'s cortisol_tonic check
(0.7→0.05), and the canvas stress palette (normalized by
`CORTISOL_PALETTE_RANGE` = 0.08 at the input), the learning posture in
`genesis_client/types.py` (cortisol_tonic normalized to the envelope
before the 0.40/0.45 gates — `> 0.45` on the raw level was
unreachable), the allostatic tracker in
`limbic_system/emotional_regulator.py` (substrate cortisol normalized
at `record_stress` entry — `CHRONIC_THRESHOLD` 0.3 and
`FALLBACK_CORTISOL_BASELINE` 0.1 were both above the envelope), and
the allostatic burden in `daemon/active_inference.rs` (`cort_elev`
now `cortisol_stress_fraction`, restoring cortisol as the dominant
mediator — raw elevation capped its share at ~0.17). Multiplicative
effects that assume full-scale cortisol go through
`cortisol_stress_fraction()` — `(cort − 0.02)/(0.08 − 0.02)` clamped to
[0, 1] — so valence drag (−0.35), plasticity suppression (×0.5),
memory gating, the ion-layer ATP cost, and the compact emotional tag
all act on the envelope, not the raw level. **When writing a cortisol
comparison or coefficient, check the reachable envelope first; a
0–1-scale threshold is silently inert.**

## Keep the project tidy
The project is large. Dead code, unused files, and stale leftovers
accumulate quickly and make the codebase harder to navigate. When
auditing or working on a file:
- If a file is outdated, unused, or no longer necessary, remove it
  rather than spending effort fixing it. Confirm it has no importers
  first, then delete it.
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
   (`~/.local/share/genesis` for this checkout). Its contents **are**
   the data dir; it is not a parent directory.
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
order anyway, so tools run outside `run.sh` agree. The resolvers the
conformance test exercises are `run.sh`, `scripts/hygiene.sh`,
`scripts/prune_dead_concepts.py`, `scripts/migrate_edges_to_log.py`,
`infrastructure/config.py`, `setup_embeddings.py`, and
`eval/abstraction_gradient.py` (`DIR_RESOLVERS`).
`scripts/install_optional_deps.sh` is an *asset* resolver — it resolves
the vosk model dir (data dir + `vosk-models`), not the data dir, so it
sits in `ASSET_RESOLVERS` instead. The Rust examples
(`examples/read_state.rs`, `examples/recover.rs`) call
`genesis::data_dir()`, the shared function pinned by
`src/data_dir.rs`'s own precedence tests, so they cannot drift from the
order. There is no remaining known deviation — check
`DIR_RESOLVERS`/`ASSET_RESOLVERS` before adding another.

**The list above is enforced, not aspirational.** Two test suites fail if
any resolver drifts:

- `python/tests/test_data_dir_conformance.py` runs every resolver side by
  side under a hostile `XDG_DATA_HOME` and asserts they return run.sh's
  answer. It also greps the tree for files referencing the state
  directory and fails if one is not in `DIR_RESOLVERS` or
  `ASSET_RESOLVERS`, so a new resolver cannot be added silently.
- `src/data_dir.rs` has unit tests for the full precedence matrix,
  including blank-pin and blank-env cases.

A blank or whitespace-only `GENESIS_DATA_DIR` counts as **unset**
everywhere; treating it as set resolves the data dir to a garbage path
built from the current directory. Adding a resolver means adding it to
`DIR_RESOLVERS` (or `ASSET_RESOLVERS` for asset paths) in the
conformance test, not just writing the `if` cascade again.

The Rust side resolves through `genesis::data_dir()` (a pure function
over its inputs) rather than reimplementing the order, so the examples
cannot drift from each other.

Assets follow the data dir too. `infrastructure/config.py::vosk_model_dir()` is the
single source of truth for the speech model: `$GENESIS_VOSK_DIR`, then
`<data dir>/vosk-models`, then the XDG-relative location, then the
pre-pin `~/.local/share/genesis/vosk-models` so an existing install
keeps working. `python/temporal_lobe/speech.py` and
`python/occipital_lobe/perception/ambient.py`
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
state or drive the system into degenerate regimes for experimentation;
see the "Operational notes" section of the README.

The state is cumulative and load-bearing: the developmental record *is*
the system. Do not edit, truncate, or casually factory-reset it.

Other things worth knowing before experimenting:

- **Restart occasionally.** `daemon.log` and `retina.log` rotate at
  startup, so a months-long single session will grow them.
- **Disk grows slowly by design.** The episodic store is append-only
  (<1 KB per episode, linear). Expect months-to-years scale; watch
  small volumes. Everything that grows in memory is bounded and
  explicitly so — forgotten-record eviction after a 1 h grace period
  (`python/temporal_lobe/memory/memory.py`), a flat 512-pattern
  attractor ceiling (`_DEFAULT_MAX_PATTERNS` in
  `python/temporal_lobe/memory/systems.py` — a growth bound, not the
  ~0.138N Hopfield capacity, which would make the subsystem useless),
  executive suppression with release deadlines and 50 retained
  conversation threads / 200 topic-history entries
  (`python/temporal_lobe/memory/working.py`), a 500-entry learning
  log, a 200-entry topic queue, and a 50-entry curiosity queue
  (`python/frontal_lobe/autonomous.py`),
  5,000 review records with `stability` capped at 7 days
  (`python/temporal_lobe/memory/spaced_repetition.py`), and a
  4,096-entry LRU vector cache
  (`python/temporal_lobe/concepts/embeddings.py`).
- **It does real background work.** Autonomous urges consume real CPU.
  Interoception dampens heavy work under thermal strain, but keep an
  eye on marginal hardware.

## Destructive operations need a store
Anything that sheds concepts must route through the archive when one is
attached, and delete only when there is nowhere to put them. `clean_noise`
used to delete unconditionally, and because it runs at step 5 of the N3
pass — before `spill_dormant` at step 6 — the destructive step could drop
the network under the cap and thereby **stop the non-destructive one from
ever engaging**. The mechanism meant to preserve knowledge was silencing
the mechanism that preserves it. One N3 pass destroyed ~7.5k concepts
with the archive sitting at zero.

If the archive write fails, the concepts stay in working memory.
Unbounded growth is recoverable; deleted knowledge is not.

Dormancy is still a legitimate reason to shed. `clean_noise`'s policy —
low-confidence *and* isolated — is intentional: the learner creates every
concept at `confidence=0.4`, so without it, thousands of orphan
fragments accumulate permanently. What changed is the destination, not
the policy. Note the two predicates differ: spill removes *dormant*
concepts, noise removal removes *isolated low-confidence* ones, so
reordering the two passes does not close the hole — only routing the
delete through the archive does.

## Concept archiving — working memory is capped, the disk is not
`ConceptNetwork` keeps at most `max_in_memory` (15 000) concepts resident
and spills dormant ones to `concept_archive.db`. Archiving is a
**memory-pressure valve, not a dormancy sweep**: `spill_dormant` returns
early while the network is under the cap, so activation level alone never
triggers it. Two call sites: `python/brainstem/lifecycle.py::_archive_dormant_concepts`
(every 10th autosave cycle — autosave runs every 5 min, so ~50 min) and
`python/brainstem/sleep.py::_consolidate_n3_heavy` (N3 slow-wave sleep,
which calls `ConceptNetwork.consolidate_during_sleep` in
`python/temporal_lobe/concepts/consolidation.py`).
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

## Dreams — subcognitive, and never learning goals
Dream content is generated during sleep (`InnerLife._generate_dream`),
flagged `is_dream=True`, and surfaced on its own `genesis~ [dream]`
stream kind — *not* as thoughts. Insisting on that separation is not
decoration: a night of dreaming previously printed `genesis~ [thought]`
lines all night, because `detect_insight` runs on every recorded thought
and announced its results unconditionally as `kind="thought"`.

`is_dream` answers "generated while asleep". The autonomous learner needs
a **different** question — "may this become a learning goal?" — and
conflating the two is what let dream material into the learner.
`SpontaneousThought.is_learning_target()` is that predicate:

- **Excluded:** dreams, lucid dreams, and `dream-reflection` thoughts.
  A reflection is a *waking* thought about a dream, so its `is_dream` is
  `False` — but it is still dream-derived, and under the old
  `is_dream`-only filter it queued its topics for the learner. A thought
  could say "I was dreaming about X" and then hand X to the autonomous
  learner, contradicting the invariant the residues are documented under.
- **Included:** everything waking, and specifically **hypnagogic**
  content. Hypnagogia is the drowsy pre-sleep state — still awake, still
  reflecting on the day. Its `is_dream` is `False` and that is correct;
  flipping it would be a second bug in the opposite direction, cutting
  genuine pre-sleep reflection out of the learner.

`_absorb_subliminal` applies this and treats a *missing* flag as
permitted, so workspace content from older state degrades to the old
behavior rather than silently starving the learner.

`_record_thought` is the single write point for `_thoughts`, a single
deque holding dreams and thoughts together. `thought_count` is therefore
a total, not a waking-only figure; use `waking_thought_count` and
`recent_dreams` for the split. Reporting one merged number made dreaming
look like thinking.

## Edge storage — the edge log is canonical
Relationships have ONE source of truth: `edge_log.jsonl` in the data
dir (append-only assert/retract/snapshot events;
`python/temporal_lobe/concepts/edge_log.py`).
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
- Known gap: `temporal_lobe/memory/semantic.py` consolidation labels
  extracted facts `origin="semantic"`, which sits in
  `DERIVABLE_ORIGINS` — so `relates_to`/`similar_to` propositions the
  semantic layer extracts have no durable representation anywhere.
  Technically consistent with the rule, but an extracted proposition
  is earned, not recomputable geometry; it wants a non-listed origin.
  Unresolved by design decision, not oversight.
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
  sink at Mind startup, because the region tree depends on
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
  suite with an explicit summary. The daemon integration test
  (`test_integration_mind` in `python/tests/test_cognitive.py`) uses a
  fresh `tempfile.mkdtemp` directory injected as `GENESIS_DATA_DIR`,
  runs the mind offline (`Mind(socket_path, offline=True)`), and shuts
  down via `run.sh --stop`; failed shutdown must preserve that temporary
  state (the `--stop` runs `check=True`, so a failure raises before the
  cleanup `rmtree`).
- Focused state-integrity regressions live in
  `python/tests/test_cli_lifecycle.py` and `python/tests/test_persistence.py`.
- Transfer/learning-compounding eval lives outside this checkout
  (`../genesis-eval/eval_transfer/run_sweep.sh --replicates N`; set
  `GENESIS_DIR` if the repo is elsewhere). Two-arm probe/ladder design
  measuring whether consolidated skills lower task cost.
- Never unlink singleton lock files during shutdown: replacing the inode
  can allow a second process to acquire a different lock for the same state.
- Existing unreadable cognitive state is a startup error, not first boot.
  Repair or restore it explicitly; do not silently replace it with fresh state.

## Explorer read scope
`frontal_lobe/tools/explorer.py` is its file-reading organ (`/explore`,
`explore_files`). Read scope is the whole filesystem by default —
owner-granted — via `default_allowed_roots()`; `GENESIS_EXPLORE_ROOTS`
(os.pathsep-separated) narrows it for deployments that want confinement,
and `Explorer(allowed_roots=[...])` overrides per instance (default:
`project_root`). The scope is **read-only** — it is not OS write or
admin access.

Two boundaries hold under every root, because whatever it reads is
committed to the concept network:
- Credential/key material is denied: `_SENSITIVE_DIRS` (.ssh, .aws,
  browser profiles, keyrings), `_SENSITIVE_FILES` (.env, id_rsa,
  credentials, secrets.*), `_SENSITIVE_SUFFIXES` (.pem, .key, .kdbx, ...).
- Pseudo-filesystems are denied: `/proc`, `/sys`, `/dev`, `/run`,
  `/boot`, `/snap`, `/lost+found` — nothing under them is a document.
Directory walks additionally skip `.*` subdirs and `_SKIP_DIRS`; an
explicit hidden file is still readable by path, an explicit hidden dir
is explorable, only *descent into* nested dotdirs is pruned. Symlinks are
resolved before the scope check, so a link inside an allowed root cannot
reach a denied tree. Regression coverage: `python/tests/test_explorer_scope.py`.

## ARC-AGI-3 evaluation boundary
- `/arc <ids> --steps N` is local practice. `/arc --all --steps N`
  runs all locally discovered environments on one scorecard through
  `Mind.evaluate_arc()`. The optional ARC SDK is installed in this host's
  system `python3` (arc-agi 0.9.9 / arcengine 0.9.3), not the checkout's
  `.venv`; use the interpreter that actually has these packages.
- The official ARC-AGI-3 card is voluntary, never driven. There is no CLI
  flag for it: `ArcOfficial.play()`/`play_all()` raise, so the only way in
  is `attempt()`/`attempt_next()` reached through her own play urge. The
  challenge sits on her game shelf as a single `arc-official-challenge`
  entry whenever `ARC_API_KEY` is set and environments are downloaded;
  `challenge_brief()` (announced once per card window via
  `world_announce`) tells her the game count, one-try rule, move budget,
  pace, the 15-minute idle clock, the 3-day window, and that a real prize
  rides on it for her steward. Picking it while well opens the card; a
  sitting then holds game mode while she appeal-picks each pending game.
  The card persists at `<data>/arc/official_card.json` — a restart
  resumes the same card, and a sealed card inside its window cannot be
  replaced. This is transport support, NOT a complete Kaggle submission
  package or certification of prize eligibility; official play still
  presumes the user's accepted rules and authorization.
- The SDK lets `OPERATION_MODE=competition` override an OFFLINE constructor.
  `ArcPractice` rejects that override before constructing the SDK, which
  otherwise can request anonymous API access during initialization.
- `ArcResult.steps` counts submitted actions and recovery resets; the initial
  reset is initialization and excluded. `resets` includes initialization.
  Use `env.observation_space` after `make` because the SDK already resets
  in the wrapper constructor. Never retry a step after an arbitrary
  `TypeError`: the first call may have already changed the environment.
- Scores are SDK-returned percentages, not completion fractions. A partial
  offline suite has no suite score because the local SDK may omit unplayed
  environments from its denominator. Public practice results are not a
  hidden-set ARC-AGI-3 result. Official scoring and mode documentation:
  https://docs.arcprize.org/methodology.md and
  https://docs.arcprize.org/toolkit/competition_mode.md.
- The adapter gives cognition only observations, available controls, and
  progress; no game source, internal engine state, baseline action counts,
  or per-game solution policy. New world contexts discard local physics
  and cached plans, not the concept network or transferable task competence.
  Retries within the same world retain learned mechanics.
- Action labels carry interface structure only — `"simple"` or
  `"coordinate"` (the action enum's own classification), never directional
  or effect semantics. This holds on both substrates: the SDK path
  (`world/arc.py`) and the public web page (`world/arc_web.py`), whose key
  table is actuation only — which key posts which action id — not meaning
  it is given. Cost accounting is also identical: `steps` counts submitted
  actions including recovery resets, `resets` includes session
  initialization, an abstention is `NO_ACTION` rather than a spent step,
  and a silent page is `TRANSPORT_DEAD`, not `BUDGET`.
- `ArcResult.level_steps` records the step cost of each completed level
  in order (re-completions after a death append again): the within-game
  learning curve, i.e. the honest form of the skill-acquisition-efficiency
  evidence the README claims. Web-substrate parity is exercised by
  `python/tests/test_arc_web_adapter.py` with a fake browser, no network.
- `ArcResult.prior_learning` discloses how much experience a run carried
  in — "none" / "loaded" / "suppressed" / "unknown" (a corrupt file is
  never a cold start) — and `decision_sources` tallies which bid channel
  won each submitted move. `/arc <ids> --fresh` makes a run cold-start:
  `_fresh_world_run` on the Mind suppresses both the load AND the save in
  `_load_world_learning`/`_save_world_learning`, so the probe cannot
  overwrite banked experience. `/arc <ids|--all> --baseline uniform|first`
  runs a named reference policy through the *same* `ArcPractice` loop via
  a `_BaselineMind` stub (`world/arc_baseline.py`) — identical accounting,
  `baseline:<policy>` stamped as every move's source; never an agent
  result, and never on the official card.
- Focused checks: `python3 -m pytest python/tests/test_arc_adapter.py
  python/tests/test_world_model.py python/tests/test_world_hypotheses.py
  python/tests/test_spatial_boards.py python/tests/test_spatial_planner.py
  -q -o addopts=''`. Keep experiments isolated from the live developmental
  directory and honor the readiness gate; an UNWELL result is a stopped run,
  not a benchmark success or permission to override physiology.
