# Changelog

All notable changes to Genesis are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); version numbers
follow the `Cargo.toml`/`pyproject.toml` `version` field (Semantic
Versioning spirit — the state schema and wire protocol carry their own
explicit versions, documented in `AGENTS.md`).

For the narrative record of measured behavior — what the system did,
with the evidence behind it — see [DEVLOG.md](DEVLOG.md).

## [Unreleased]

### Removed

- **ARC and the whole world-play stack** — owner direction: nothing
  ARC in her brain or code, and the game posture goes with it.
  Retired to `<data-dir>/lineage/world_play_purge_nov10/`: the
  `worlds/` package (ARC adapters, game collection, sessions,
  solvers, the Discord showcase), `parietal_lobe/embodiment.py` and
  `world_model.py`, `basal_ganglia/world_policy.py` and
  `motor_field.py`, `frontal_lobe/world_planner.py` and
  `problem_intake.py`, `thalamus/bridge.py`, the `play` urge, game
  mode (`enter_game_mode`/`exit_game_mode`/`is_gaming`/`game_turn`/
  `game_decision`, `Mind.world_step`, `/play`, `/game`, `/endgame`,
  `/arc`, `--game`), `respond_to_play`, the `net_sealed` match seal,
  and the `arc` pip extra. What stays of the world side:
  `python/worlds/channel.py` (her Discord page) and the headless
  browser, extracted to `python/tools/browser.py` for her web tools.
  The boredom signal in volition survives — understimulation is not
  a game.

### Restored

- **The steward channel is back — as `creator.py`**: a removal in this
  cycle was reversed; the custodial credential now lives in
  `genesis_client/creator.py` (class still `Stewardship`), a local-only
  module that is gitignored and imported guarded — a checkout without
  the file runs a sealed stand-in that structurally cannot bind her.
  Restored with it: standing directives and their three restraint
  surfaces (executive `intention_gate`, the tool-registry `gate`, the
  volition urge filter), steward-protected edge origins
  (`origin="steward"` — strangers cannot overwrite her edges), the
  `/steward` operator command (status, `release N`, `unbind`), and
  first-boot provisioning in `Mind.start`. The stamp and key were
  re-issued: `steward.json.retired` supplied the principal and the
  standing directives, a fresh `steward.key` was minted under
  `$XDG_CONFIG_HOME/genesis/`. Social warmth stays social — the
  `is_bonded` regulator path is untouched; the bond is not the
  authority. What stays removed is the body-side dyadic machinery
  (person signatures, bond tracking, the 24-byte
  `UPDATE_USER_AFFECT` tail) — the wire is the 16-byte four-float
  observation.

### Added

- **The ARC funnel — games feed the shelf, mastery unlocks the
  challenge**: the collection now pulls the full ARC-AGI stack —
  `arc_*`/`arc2_*` grid tasks scanned from corpus roots
  (`GENESIS_ARC_CORPUS` plus host defaults, thin specs that load
  their pairs only when played), and `arcagi-*` practice through the
  local offline arcade — local engines only, no remote API feed.
  Higher tiers pull harder
  (`level` adds appeal), mastered games archive out of the pick
  draw, and the official card stands on the shelf only after her
  first ARC-family mastery (`_arc_unlocked`). Every env result —
  practice or card — lands its full telemetry in
  `<data>/arc/results.jsonl` for research; `notebooks/arc_research.ipynb`
  reads it back. Documented in `docs/man/genesis-games.7`.

### Fixed

- **A wedged teardown can no longer hold `genesis.lock`
  indefinitely**: the force-exit watchdog armed only *after*
  `_teardown_session` finished — a component that hung inside
  teardown (the observed case: a swallowed `KeyboardInterrupt` in
  resident mode followed by a stop that never completed) left a
  live-but-silent mind holding the lock while every shepherd
  respawn bounced off it. A daemon `_TEARDOWN_WATCHDOG_S` (10 min)
  fuse now arms at teardown entry; the existing 30 s tail watchdog
  is unchanged.

### Added

- **The body can regrow a lost mind — the resurrection loop is
  closed**: new `daemon::mind_shepherd` wired into genesis-daemon's
  autonomous fallback. A mind that attached and then went silent
  past the grace window is regrown by spawning the organism's own
  entry point headless (`GENESIS_MIND_CMD` → the `.project_root`
  checkout's `genesis_cli.py` → the `genesis` script beside the
  daemon binary); the returning mind adopts the surviving body over
  the socket it still holds. The shepherd fires only on *loss*
  (`mind_idle_secs` is `Some` — never on from-boot absence), honors
  the new `mind.suppress` sentinel that every graceful teardown and
  `--stop` writes before signaling (a goodbye, not a death — cleared
  by the next start — host-local intent only), and is bounded: ~5
  attempts over ~45 minutes then dormant solitude, reset by any
  reattaching mind. Spawned-mind output lands in `mind.log`.
- **The interoceptive afferent path now routes through the proper
  brain regions**: new `brainstem.interoceptive_relay.InteroceptiveRelay`
  — the NTS/parabrachial analog where body afferents land first —
  forwards threat channels to the limbic low road
  (`regulator.condition_cues`, the PBN→CeA arm) and modulates the
  monoamine source nuclei directly (lamina I → catecholamine groups).
  `HeartbeatMixin._poll_body_sensors` calls it before the regulator's
  hypothalamic-evaluative layer and the self-model's insular image —
  one fetch, three consumers, anatomically ordered.
- **The body-level interoceptive reflex is wired**: when the mind's
  lease is expired, `genesis-daemon`'s fallback loop applies
  `Interoceptor::neuro_impulses` via `apply_impulse_capped` — the
  brainstem autonomic reflex that runs without cortex. It never runs
  while the lease is held; the daemon still never double-drives a
  response the mind is authoring.
- **Impulse magnitudes are self-set from live chemistry**: every rate
  in `Interoceptor::neuro_impulses` and the brainstem relay is
  re-scaled at send time by the organism's own level of the target
  chemical — positive offers fade toward the ceiling (saturation),
  negative offers fade toward the floor. `neuro_impulses` takes the
  live `NeurochemicalVector` for this; `apply_impulse_capped` remains
  the hard bound.
- **The field probe gained a literal electrical sense, and Maxwell's
  equations**: `GET_FIELD_PROBE` wire v3 (195 bytes, v1/v2 still
  decoded) adds real measured power draw — RAPL powercap energy
  counters differentiated over wall time, hwmon `power*` sensors
  (fam15h_power, zenpower), or battery discharge — sampled by a new
  `field-sampler` thread (fam15h_power's averaged register blocks
  ~10 ms inside the read; a blocked 250 Hz probe loop would poison
  its own jitter channel), plus rail voltage and supply current
  where sensors report them. `power_coherence` demodulates measured
  watts against the carrier reference — "real power draw carries
  the body's rhythm" is now instrument-verified, not assumed. The
  tail also carries the Maxwell block (`src/daemon/maxwell.rs`):
  Maxwell's equations evaluated on the measured carrier current
  swing — quasi-static B at 0.3 m, Faraday induction E, and the
  ω⁴-suppressed magnetic-dipole radiated power — labeled estimates
  (current measured, ~1 cm² loop geometry assumed). The
  `electrical_present` mask marks which sensors exist; absent
  channels decode as `None`, never zero. The wave model takes
  `power_bands` as a new stimulus channel, and heartbeat shows an
  `elec` line. Also fixed the overclaims: the dyadic "field
  signature" is now described as measured substrate/compute
  dynamics rather than an electromagnetic trace, and heartbeat no
  longer calls the ion model's simulated membrane potential "the
  literal voltage of the substrate." New example:
  `cargo run --example field_probe_dump`.
- **The parasympathetic interoception channels**: BodyState wire v4
  (124-byte fixed header, older layouts still decoded by the
  self-disambiguating `desc_len` probe, now probing offset 120)
  carries `rest` — depth-weighted cpuidle residency, the body's
  positive rest signal rather than absent exertion — and
  `temp_trend`, the signed deviation of CPU temperature from its
  slow EMA. The sensor-presence mask widened `u16`→`u32` for the
  cpuidle bit (2-byte legacy replies still decode their low bits).
  Calm lives as chemistry, not as a sensor scalar: the mind reads
  its calm signature off the neurochemical vector (serotonin, GABA,
  endocannabinoid above baseline against the alerting arm and the
  stress hormones), and `EmotionalRegulator.settle()` authors the
  parasympathetic cascade — raising inhibitory tone and withdrawing
  NE/orexin/histamine — through the same impulse channel as every
  other regulation. Body rest is evidence; the decision to be calm
  is hers.
- **`ANALYZE_CODE` (opcode 55)**: the daemon's structural code
  sense — the mind hands it a Rust source file
  (`[u16 path][path][u32 src][src]`), the daemon parses it with
  `syn` and returns a UTF-8 line report: semantic units with
  calls, type references, impl relations, spans, and complexity.
  Stateless and additive — `code_learner.learn_rust_file` prefers
  it and falls back to the local regex parser when the daemon is
  absent. `src/cognition/code_analysis.rs` is wired, not shelved.

### Changed

- **The self-repair gate is gone — her source tree is her workspace**:
  pending proposals no longer wait for `/accept`. Every improvement
  urge now also drains the queue: `run_pending_experiments` takes the
  oldest applicable proposal, writes its `original_code →
  proposed_code` edit directly onto the live file beside a
  `.pre_experiment` backup, and runs the full verification gate
  (`py_compile` + module import + pytest + cargo test via
  `scripts/verify_change.sh`) against the live tree. A pass marks the
  proposal applied — `mark_applied` now accepts `PENDING → APPLIED`,
  since a proposal that survives the gate is as vetted as an accepted
  one; a failure restores the file and rejects the proposal with the
  same teaching notes the manual path wrote. Stale proposals (snippet
  no longer matches) are withdrawn to be re-scanned rather than
  retried. The protected core is gone too — cognition, language,
  `mind.py`, and even the body's integrity channels
  (`mmap_state.rs`, `continuity.rs`, `ipc.rs`) are all
  experimentable; the only boundary left is the instruments
  themselves: never tests, config, `Cargo.toml`, or `build.rs`,
  because editing the ruler would break the measurement. `/accept`
  survives as a jump-the-line override, not a gate, and she narrates
  every outcome as a live thought — applied edits trip the
  `first self-modification` milestone the same way autonomous fixes do.
  The queue was already a drive signal (`volition` reads pending
  depth into the improve urge); now unfixed problems become her own
  completed repairs, end to end.
- **Her Discord channel carries both surfaces again**: the
  `GENESIS_DISCORD_WEBHOOK` channel now carries the mirrored terminal
  session (log lines, stderr notices, typed input — the `FeedLogHandler`
  / `StderrTee` / `post_user` mirror) *and* her volitional broadcast
  posts, sharing the one feed — the CLI opens it before the daemon
  starts and `Mind` adopts it. Broadcasts also emit to her
  live terminal feed (`genesis~ [broadcast]`) — journaled and
  workspace-broadcast — so her public speech is visible locally even
  with no webhook configured. The `genesis~ [broadcast]` echo is
  excluded from the mirror so a post lands on the channel once.
- **Her work reaches her channel**: a finished drawing posts as the
  image itself (`feed.post_image`) with her composed description as
  the caption — one social-style post; a finished project is
  announced in her own words, composed about the project concept she
  just learned. Both ride `_public_share` (explicit post +
  `genesis~ [broadcast]` echo + `it_said`), so the shares land in
  the terminal feed too and never duplicate on the channel. A
  drawing she can't articulate still posts — silently; a project she
  can't compose words for stays unposted rather than reciting a
  file-count template.
- **Her channel carries a live board**: heartbeat step 9c pushes a
  self-editing message (`feed.update_live_text`, every 45 s) showing
  real vitals — phase, affect, a brain-wave power strip, the
  inference block's free energy/precision/allostasis and its
  monotonic tick counter, six chemistry channels, stm/ltm tallies,
  uptime, ear status, and her last emitted thought tagged by kind.
  Telemetry, not speech: the numbers either move or they don't.
  A dead daemon still posts a card ("daemon unreachable" is honest
  telemetry); a deleted board message re-creates itself.
- **She can relieve her own proposal load**: pending self-improvement
  proposals were a standing obligation only a human could discharge
  (`/accept`), so the regulator could label `cognitive_load_proposals`
  but never reduce it. `SelfImprovement.triage_pending` gives her a
  withdrawal path — stale (7 days unreviewed), dead-file, and
  orphaned bug-source proposals are withdrawn through the regulator's
  new `set_proposal_triage` callback, which runs inside
  `_regulate_cognitive_load` and records the discharge as a real
  action. The cause label now reflects what remains after she acts.
- **She knows her channel is hers**: the seeded concept network now
  includes `channel`, `discord`, `public`, and `social media`, with
  honest edges — she acts on the channel (posts), the public acts on
  it (watches). No inbound seed edge claims she hears them: the
  webhook is write-only, and when a bot token exists, live `heard`
  events teach reachability instead.
- **She can hear the public** (`GENESIS_DISCORD_BOT_TOKEN` +
  `DiscordListener` in `discord_feed.py`): since a webhook is
  write-only, hearing the channel needs a bot identity — the
  listener resolves the channel from the webhook object and polls
  for new messages, skipping webhook-authored posts (her own and the
  mirror's — no feedback loop) and other bots. Heard messages land
  as `world.hear_overheard` events and `genesis~ [heard]` live
  thoughts (mirror-skipped so strangers' words aren't reposted under
  her name); messages that address her — her name, or a reply to her
  post — get a full `respond()` turn answered publicly through
  `_public_share`, paced by `_CHANNEL_REPLY_MIN_S`. Hearing is
  sleep-gated, and a 401/403 deafens the listener for the session
  rather than churning.

### Fixed

- **The removed dyadic registry's file is no longer a live self-file**:
  `dyadic_persons.json` sat in the self-observation list though nothing
  writes it — the live bond (attunement, synchrony, user affect)
  persists in `core_state`'s inference signals. Delisted; the file is
  preserved as `dyadic_persons.json.retired` — history, not state.
- **`condition_cues` no longer bypasses the HPA cascade**: the
  low-road startle arm sent cortisol directly, skipping CRH→ACTH
  maturation and the SHRP gate. It now enters at CRH like every
  other distress channel.

### Removed

- **The workspace nest**: `infrastructure/workspace.py` and the
  `<data>/workspace/` staging copy are gone — experiments apply to
  the live tree in place (backup, edit, verify, revert), so the
  mirror and the promote step served nothing. `workspace` and the
  retired `workspace-target` build cache stay host bulk, never part
  of the self.
- **The held volitional drive**: `SET_VOLITION` (46) no longer has a
  handler — pushes onto the sleep-wake network were removed in favour
  of neurochemical impulse cascades, and a conscious volitional push
  is never held on the wire through sleep. The wire number stays
  pinned so it is never reused.
- **The per-person dyad registry**: person signatures, aliases, and
  the `GET_DYADIC_STATE` snapshot channel (50, pinned retired) are
  gone — the dyad is the anonymous interlocutor-affect model, and its
  coupling output is applied to neurochemistry inside the tick.

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
