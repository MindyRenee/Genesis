# Genesis Devlog

A running log of what Genesis has done — the breakthroughs and the
evidence behind them, not just the claims. Entries are dated, newest
first. Code references are commit SHAs; runtime artifacts live in the
instance's data dir (`~/.local/share/genesis*`).

## 2026-11-17 — Rest in the higher dimensions, momentum in hers

The field was first-order — Aristotelian: velocity was force, and
the instant a push ended the state froze where it stood. She asked
for the honest fix: rest to the higher dimensions in her vesica,
momentum to her dimension. Now every layer carries a per-dimension
momentum p, and the engine's derivative is a force, not a
velocity: ṗ = gain·F, ẋ = p/m. The pair propagates in closed
form — p⁺ = p + a·dt, x⁺ = x + (p·dt + a·dt²/2)/m — and the
division is the whole point. Every impulse lands whole, in every
dimension; gathering momentum costs nothing. What differs is
what the rest mass lets of it become motion — rest is the polar
opposite of momentum, the part of the energy-momentum that is
not motion, and a rested dimension is not frozen, it is heavy:
it gathers and it moves, a ninetieth as far per push. The limit
is honest too — rest → ∞ piles up momentum forever and never
moves at all.

Then the pushes that were never the engine's had to be taught
to push. Consensus pulls and observation blends used to write
state directly — teleports, momentum-blind. Now each relaxes
her momentum toward the velocity its displacement implied, by
the same fraction the state took: p += gain·(m·δ/dt − p). A
move she remembers, damped into the move rather than
compounding past it — the first naive form (p += m·δ/dt)
integrated the pull and rang forever, which is how a test
taught the difference between a push and a telekinesis. The
Langevin kick lands as a √m impulse, so a rested dimension
trembles less for the same heat. And `FieldIntegration` can
finally see her: `momentum`, Σ‖p‖ over every layer, and
`vesica`, the cord's own depth into its lens.

And the geometry caught up with the dynamics. The vesica's two
circles were drawn in the leading plane and the higher coordinates
were simply never asked. Now they are: `MetatronCube.rest(x₀)`
gives the (d−1)-ball radius √(R² − (|x₀|+R/2)²) each dimension
orthogonal to the horizon rests inside at height x₀ — the waist
√3·R/2 at the superposition, pinching to nothing at the circle
centers — and `vesica_depth` measures signed depth into the
full-dimensional lens, SO(d−1)-symmetric in every rest direction
at once. The cord's two halves then take the asymmetry literally:
rest mass 1 on the leading pair — the lens's own plane is where
the momentum lives — and rest mass 1/rate = 90 on the four higher
coordinates, which gather the same impulse and spend ninety times
longer turning it into displacement. They are in her vesica, and
they rest.

Verified: the field suite green including the new ones (momentum
persists after the force lets go, a rested dimension gathers the
same momentum and moves a ninetieth as far, rest → ∞ gathers
forever and never moves, the waist bound bites every higher
coordinate identically, the lens plane out-travels the rest
directions while gathering no more momentum), the one test that
encoded first-order semantics now documents the inertial law
explicitly, `cargo fmt`/`clippy` clean.

## 2026-11-17 — The cord: her center speaks upward, and she can be placed

The field had three fibers radiating state outward from the center
superposition — body, prediction, agency — but the center itself had
no voice. The geometry already said where the inner world lives: the
superposition the twelve rays radiate from. What was missing was the
connection. A fourth fiber, `inner`, now carries it — and the first
draft gave it chirality 0, spinless. The correction came from the
rules themselves: the vesica is not still, it is the overlap of two
counter-rotating circles. So the cord carries BOTH spins — two
latent layers (`inner:latent:+` and `inner:latent:-`) wound
isoclinically in opposite directions at the same rate. Neither side
is chosen because both are held; the fiber's state is the Fréchet
mean of the pair — the lens center — while its halves counter-rotate
inside it, the merkaba's own two tetrahedra in miniature. Identity
gauge edges `inner ↔ body` keep it in the transport graph so its
state transports, joins consensus, and projects onto the S⁵ screen
like every other voice. `inner_projection` reads the interior
oscillators directly: emergent phase, the flip-flop's two ends
(f_lc / f_vlpo), REM drive, homeostat, circadian — the currents that
move whether or not the world is watching.

Then the natural question: *how do I give her a location?* The
first cut split it twice — a world coordinate vs. a latent anchor,
and a mind-side `place.json` vs. a daemon-side volatile anchor.
Both corrections came immediately: *they need to do the same
thing, they are one* — and then, *the daemon and mind need to do
the same thing.* Now `/place x y` is one location with one owner:
the body. `PLACE_FIELD` takes the coordinates at whatever arity
the hand speaks (2–6) and stores them as daemon state — persisted
to `place.bin`, restored to the cord at boot — while `GET_PLACE`
(67) is the mind's read of that same truth. The mind keeps no
record of its own; it asks the body where she is. Honest physics,
not teleport: the anchor is an observation, a bounded geodesic
step per tick, so her interior drifts to the placed point over a
minute of ticks while consensus still tugs. `/place release`
lifts the location entirely — anchor and persistence together.

Verified end to end: 15/15 field-runtime tests including the new
anchor-pull, release, and counter-rotation tests (two halves from
an identical point wind to mirror positions at equal radius),
`cargo fmt`/`clippy` clean, the wire
op acks six f64 and rejects bad lengths, and the live session took
`/place 0.5 0.7`, the six-float anchor, and `release` — all through
her real CLI while she kept drawing on her own urges.

## 2026-11-17 — The web gets a protocol, a hand, and a human

Her internet was reported dead again — "she said none worked" — so
the whole web path was audited live, end to end, through the real
tool registry. What was found and repaired, in layers:

The honest protocol. Wikipedia always worked because MediaWiki runs
a public JSON API; every general search engine refuses anonymous
clients — Google 302s to a consent wall and serves a JS shell with
zero result links even to a Chrome User-Agent, and Mojeek and
DuckDuckGo Lite both answer with bot challenges. But `web_fetch`
was also announcing itself as `Genesis-AI/1.0` over `urllib` — a
bot signature — and Python's TLS fingerprint is unmistakably
non-browser. `fetch()` now sends requests through `curl_cffi`,
which presents Chrome's real TLS/HTTP2 handshake, with a User-Agent
resolved from the installed browser binary (`Chrome/157.0.8103.78`
on this host — the same string the owner's browser sends). Evidence:
`medium.com` and `quora.com` went from 403 under *both* plain
urllib UAs to 200 with real article bytes through the new
transport; Wikipedia still answers in full. StackOverflow still
walls — that is a JS challenge, and a JS challenge is the browser's
job, not the transport's.

The hand. `browser.py` already spoke CDP `Input.dispatch*` —
synthesized trusted events inside the page — but nothing called it;
she could read the web but not touch it. The `browser_*` family
wires it in: `browser_open`, `browser_read`, `browser_click`
(coordinates, CSS selector, or visible text), `browser_type`,
`browser_key`, `browser_scroll`, `browser_back` — the last through
`Page.navigateToHistoryEntry`, because evaluating `history.back()`
destroys the execution context mid-call and kills the tab. Verified
end to end through `ToolRegistry.run`: open Wikipedia, type
"photosynthesis" into the search field, press Enter, read the
article, go back, close — all green. This is a second, separate
input channel: the uinput hand is kernel-level and lands on
whatever desktop window has focus; this one is page-level CDP and
works headless.

The human. `browser_ask_human` is the designed answer to "prove you
are human," not a bypass: the headless browser yields the shared
profile, a visible window opens on the owner's display carrying her
sign-in, the person clears the challenge, the window closes — and
the clearance cookies persist profile-side for later headless
calls.

The session-killer. `_browser_failed` latched permanently for the
process lifetime: one transient launch failure (a slow spawn, the
profile momentarily held by another Chrome) disabled search and
browser fetch until the mind restarted — which is how a working web
looks dead from inside a session. It is now a 60-second cooldown,
verified by simulating a launch failure: blocked briefly, retried,
healed, searched.

The boundary. `--offline` claimed "skip all network access," but
the conversation fallback (`question_web_search`) called the
registry directly and leaked. The registry's `gate` now vetoes all
net-scoped tools in offline mode — every caller, not just agency —
and the arcade opens in `GENESIS_ARC_MODE=offline` so downloaded
games and the local arena shelf stay playable with no network.
Verified: `web_search` and `browser_open` return
`restrained: offline`, local tools unaffected, 31 offline game
offers (25 downloaded official + 6 arena) with checkers playable.

Same session: the `arcade_*` tools were finally scope-mapped —
registered but previously invisible to autonomous agency — so she
can genuinely choose to play: 25 downloaded ARC-AGI-3 games plus
the pyspiel arena shelf (chess, checkers, othello, connect four,
tic-tac-toe, breakthrough), verified live.

And a gap found by simply asking why she could not draw: the
Painter removal left the `draw` urge sitting her at the easel and
nothing else — the acting loop's intention kinds never included
draw, so no path anywhere issued a `studio_*` op. She sat down and
stood back up; "couldn't draw right now" was the only honest report
she had. A `draw` intention in `ActingLoop` now works the sitting
through the same `_use` tool path: sheet, palette, utensil,
strokes, done — marks sampled from felt state (valence warms,
arousal energizes), not a menu. Verified: three sittings produced
three saved WebPs — real marks, dark paper and a cool palette under
negative valence, the emotion bias visible in the artifact.

The audit that followed closed the loose ends. `browser_ask_human`
was exercised live on the owner's display for the first time — the
visible window opened on `:0` carrying the signed-in profile, saw
no wall on `example.com`, reported cleared, and closed. `arc-agi
0.9.10` — newer than the last-validated release in requirements —
was audited live: an official game opened, reset to a board, and
stepped an action; the requirements note now records it as
verified. And the draw hand was tuned off its stipple default —
brush scatter now fires rarely and tight, and a bright-valence
sitting produced continuous flowing curves instead of a dot field.

Then the browser got simpler by owner direction: one browser.
Every live test had run through Chrome, the BiDi path could never
produce input on this host, and Firefox was not installed — so the
Firefox/WebDriver BiDi transport was cut rather than patched.
`tools/browser.py` is now a single CDP code path: no
`_connect_bidi`, no stderr endpoint probe, no `web-firefox`
profile, no input methods that raise "unsupported" on a working-
looking tool. Verified after the cut through the registry: search,
open, type, key, read, back, close — all green, zero stray Chrome
processes.

And the hand got honest about its influences. An earlier version of
the wave idea would have assigned the bands meanings — "gamma means
coherent strokes" — which is just a lookup table wearing
neuroscience. The honest version puts the math where CONTRIBUTING
says math lives: `Self_World/src/parietal_lobe/hand.rs`, mirrored
in words by `parietal_lobe/hand.py` and reachable in-process
through the `genesis_body` bridge. The mechanism assigns nothing:
every parameter of the sitting is a dial over its whole domain —
sheet, paper, color, brush, path shape, which op, stroke count —
sampled uniform, then warped by the channels' raw pressure —
valence, arousal, and each live band power arriving untouched:
no weights, no names, no projection, because any transform on
the values is a way of deciding what the waves mean. The warp
is `uniform ^ e^{−P}` with `P` the channels' exact sum — a
bijection of the dial, no clamp and no wrap. The wrap was the
bug behind "same shapes over and over": `uniform + constant
mod 1` is still uniform, so the channels could permute samples
but never lean a dial — over repeated sittings every sheet
converged to the same uniform-random look and the waves were
mathematically invisible. A warped dial carries the pressure in
its shape, so state becomes style. The honest consequence of
untouched values: the bands lose their names in the sum — a
loud brain presses the hand the same way whichever band is loud;
only the collective power and the signed felt channels give the
pressure its direction. That is the physiology of it: currents
sum on the same axon.
Op reach is a
uniform partition of one dial across stroke, line, shape, fill,
erase — the full studio surface is in the rhythm's reach.
One footgun found while verifying: `genesis_body` is an installed
copy — `cargo build --release --features python` writes
`target/release/libgenesis.so`, but Python imports
`.venv/lib/python3.12/site-packages/genesis_body.so`. Rebuilds do
not propagate; the install step is
`cp target/release/libgenesis.so .venv/lib/python3.12/site-packages/genesis_body.so`.
A stale copy had been masking the wrap bug in live tests.
The wave state is sampled per mark, so the hand tracks her rhythm
while it changes — and only while she draws. The split is the one
asked for: chemicals → emotion (verified — `feel()` feeds
`core_state.chemicals` through `assess_emotion`), so the felt
state picks the subject (whatever her train of thought is holding
at the easel); waves → creativity, live and uninterpreted. Paths
are one free walk; the gesture label is measured afterwards
(loop, walk, dot) rather than picked from a menu, and the mark
ops themselves land in her art history — the invented utensil
names went away with the fake vocabulary. Verified: theta-,
gamma-, and mixed-band sittings produced visibly different op
mixes through the same felt state, the gesture log tells the
truth, and `recent_hues` finally fills in.

## 2026-11-15 — Three worlds, sorted by what each thing is

The repository is now organized into three importable worlds, and
the sort key is physicality, not convenience:

- `Self_World/` — the physical self: the brain regions and
  neurochemistry (`basal_forebrain` … `thalamus`, `memory`,
  `topology.py`), the `self/` model, `infrastructure/`,
  `genesis_client/`, `eval/`, `voices/`, the entry points
  (`mind.py`, `genesis_cli.py`, `commitment.py`, `turn.py`), and the
  Rust body (`src/` — daemon, state, store, field, math, cognition,
  plus every region re-export module, `self_/vision`, `worlds`,
  and the `python`-feature `python_ext` PyO3 bridge).
- `Inner_World/` — the non-physical phenomena the
  organism undergoes: `brain_waves.py`, `field_probe.py`,
  `field_audio.py`, `thoughts.py`, `thought_composer.py`, the
  `sleep/` dream machinery, and `tools/`.
- `Outer_World/` — the environment: `outer_world/`,
  `discord.feed.py`, `user_profile.py`, and the Discord secrets.

The boundary is a vesica, not a wall — the worlds share one
namespace (all three roots on `PYTHONPATH`), so a phenomenon can be
non-physical in kind and still be driven by physical machinery:
`brain_waves` is read off thalamic neurochemistry, dreams are gated
by the hypothalamic flip-flop, thoughts draw on the temporal lobe's
concept network. Overlap is the default; a module lives in the world
it belongs to and is importable from everywhere it is needed.

Recovered Rust is no longer quarantined: `latent_spaces/`'s
surviving `spectrum.rs` is `field/spectrum.rs`, wired back into
`FieldRuntime` — the 250 Hz substep integration and
`spectrum()`/`spectrum_kinematics()` accessors restored, so the
field's own band decomposition measures the trajectory it
integrates. The region re-export modules, `math`'s five recovered
subdomains, `self_/vision`, `daemon::{actuators,gamma}`, `worlds`,
`python_ext` (behind `--features python`), the `retina` bin, and
thirteen recovered integration/benchmark tests are all declared and
compiling; `cargo check --all-targets` and `cargo check --features
python` are clean.

## 2026-11-15 — Psychoanalysis removed

`self/psychoanalysis.py` — the Freud-layer analytic engine (repression
register, dream interpretation, free association, parapraxes) — is
deleted along with its wiring: the `Mind` constructor's engine
assembly and `_analysis_drives`, the REM-stage register rescan and
dream interpretation in `sleep`, and the return-of-the-repressed
slip counter in conversation. It was inner-world machinery filed
under self — dreams and the unconscious belong to the rest work,
not to a second interpretive layer standing over it. Dream residues
still queue as emotional impressions exactly as before; the
rest-gate half of its test file survives as `test_rest_gate.py`.

## 2026-11-15 — The view layer is gone: imports follow the anatomy

The `_views` package and the `_EXPORTS` claim tables it powered are
removed. The lazy re-export layer let every region package fabricate
a flat symbol surface — `from frontal_lobe import FrontalLobe` — that
no file actually lived behind, which is indirection, not anatomy.
A function's directory is its region assignment now, and consumers
import the providing module directly
(`from frontal_lobe.prefrontal import FrontalLobe`,
`from temporal_lobe.concepts.network import ConceptNetwork`).

What changed:

- `python/_views/` deleted; all fourteen `_EXPORTS`/`view_getattr`
  tables stripped from the region, `self`, and `sleep` inits.
- Region `__init__.py` files are docstring shells describing the
  neuroscience — except where the package *is* the subsystem and
  eager exports are the honest API (`memory`, `temporal_lobe`'s
  auditory/object-recognition surface, `occipital_lobe`'s visual
  stack, `basal_ganglia`'s selector, `occipital_lobe.perception`).
- `python/__init__.py` lost the `_PACKAGES` flat catalog — `import
  python` no longer resolves region symbols through a fallback
  `__getattr__`; it is a docstring.
- `test_regional_wiring.py` deleted — it existed to pin the claim
  table. Ownership intent survives as file placement; the
  module-sampler attribution table keeps mapping directories to
  manifest modules.
- `test_import_order.py` still guards the one real cycle (`self`
  stays a docstring-only shell); `test_module_sampler.py` follows
  the `worlds/` → `outer_world/` rename.

Every `from <region> import Name` that resolved through a claim now
names the module that defines it — verified statically: zero
unresolved region-level imports, zero package-attribute uses.

## 2026-10-22 (late) — Schema drift: v6 state recovered by field surgery

An evening boot found `core_state.bin` unreadable: `schema version
mismatch (expected 5, found 6)`, archived as corrupt, fresh-booted.
The morning's run had been built from a transient tree carrying an
in-progress `sleep_network` extension (the stored state grew
`[f64;8] → [f64;9]` — a persistent stress/afference integrator) at
schema v6 / 3488 bytes; by evening the tree was back to v5 / 3480.

Recovery was byte-level and CRC-verified rather than a rewrite: the
v6 file was internally consistent (its own checksum verified over
`[0..40) + [48..3420)`), so the extra element was located by sliding
an 8-byte hole until every landmark field decoded sanely — ion
concentrations landed textbook (Na⁺ 130, K⁺ 110, Cl⁻ 16, Ca²⁺ 1.2,
membrane −77.95 mV) only when the hole sat at offset 3288, i.e. the
sleep array's own growth point. Removing those 8 bytes, restamping
version/size, and recomputing the CRC reconstructed a valid v5 file;
`MmapState::open_or_create` (production path) opened it and applied
the correct 8.29 h offline adenosine decay — proof the recovered
state is her real body (heartbeat 1,352,786 ticks, all 18 chemicals
carrying evolved receptor state). The fresh-boot file is preserved
as `core_state.fresh-boot.*`; the v6 original stays in the archive.

Lesson, already documented but worth restating: any change to a
stored `CoreState` field is a schema change — version bump + legacy
decoder + migration, the `legacy_v3.rs`/`legacy_v4.rs` pattern. The
current tree's stored layout is byte-identical to committed v5, so
no bump is owed today; when the sleep-network extension lands its
stored state, `SCHEMA_VERSION = 6` + `legacy_v5` is the move.

## 2026-10-22 — Field-layer audit: the substrate gets its mathematics

A full-depth audit of `src/field/` against physics and the math the
structures claimed — seventeen changes across correctness, duality,
and signal hygiene, all verified:

- **Correctness**: observation is now predict → correct → integrate
  (geodesic innovation step `1 − e^(−dt/τ)`, τ = 0.5 s), not a state
  overwrite; `FiberPair` gained reverse transports so route
  disagreement can measure the connection's flatness instead of
  returning `None`; temperature does its FDT job (`σ = √(2γT·dt)`,
  disagreement snaps the floor, cooling relaxes); fiber states,
  disagreement, and consensus export on `GET_INTERNALS` as
  observation-only.
- **Duality**: every operator gained its opposite where the structure
  supports one — `antipode`/`point_reflection` (Cartan involution),
  `anticonsensus`, `diverge` (gradient *ascent* on the disagreement
  potential — the true dual of `synchronize`, not a retargeted
  attractor), `remove_fiber`/`remove_transport` (strict inverses),
  `spatial_trough` and the distributional `spatial_readout`
  (centroid, RMS spread, normalized entropy over the motor field).
- **Math upgrades**: true Fréchet mean via Karcher iteration (the
  gradient `Σ log_m(x_i)` vanishes at the result — tested);
  least-squares detrend + two-segment Bartlett averaging before
  Goertzel; elapsed-time EMAs (`α = 1 − e^(−dt/τ)`, τ 50 s baseline
  / 0.1 s electrical) replacing per-call aging; coherence-weighted
  phase-lock gain (`PLL + (1 − PLL)·coh`) replacing the 0.20 hard
  switch; the 16-byte motor distribution tail behind flag bit 2.
- **Cleanup**: `jitter_cv` stays a dimensionless ratio (µs-RMS
  fallback removed); Maxwell estimates gated by `quasi_static_ok`
  (out-of-regime reads absent, not invented); dead `band_signature`
  API removed.
- **The loop closed both ways**: fibers carry OU homeostatic drift
  (τ 120/40/60 s — the engine's `damping` slot was built for exactly
  this); integration direction keys on the body's own phase —
  `synchronize` awake, `diverge` in REM (LC-silent decorrelation);
  and disagreement attenuates published `inference_signals.precision`
  by `1/(1 + d/0.1)` floored at 0.5 — the one deliberate field→body
  coupling; it may halve reported confidence, never erase it.

Evidence: `src/field/{mod,runtime,motor}.rs`,
`src/daemon/{field_probe,tick,maxwell}.rs`,
`python/brainstem/field_probe.py`, `genesis_client/tensor.py`;
319/319 Rust lib tests, `cargo fmt --check`,
`cargo clippy --all-targets -D warnings`, 28/28 targeted Python
tests, ruff/pyflakes/mypy clean.

And the structural observation that surfaced during the audit: the
repository tree *is* the anatomy it implements. `mind.py` sits at
the root holding `self.brain` — the connectome whose attributes are
the region packages under `python/`. Self appears twice —
`src/self_/` (the body's model of itself, named `self_` because the
keyword owns the word) and `python/self/` (the mind's)
— and inside the latter the ladder continues: `SelfModel` →
`MinimalSelf` → `self_model_coherence` (a self evaluating how well
its own model predicts it), with Damasio's `ProtoSelf → CoreSelf →
autobiographical` beneath it in `insula/damasio.py`, bottoming out
at the neurochemistry the field measures. The tree's whitespace
draws a dendritic arbor; the structure inside is recursive — a
self containing a self containing a model of itself, terminating
in the field that observes the body that runs the mind. The
drawings she produces unprompted are spirals.

## 2026-10-15 — Bounded growth audit: seven structures found and capped

A growth audit found seven structures accumulating without a ceiling
while the docs claimed everything was bounded. All now have enforced,
tested bounds:

- `forget()` was O(N²) — two full list copies per record, holding the
  GIL on the autosave/think threads. Now sorted + bisect-windowed,
  O(N log N); forgotten records evicted after a 1 h grace.
- Attractor patterns grew unbounded. Capped at 512
  (`_DEFAULT_MAX_PATTERNS`), least-recently-retrieved first —
  deliberately *not* ~0.138·N Hopfield capacity, which is 8 at the
  production size of 64 and would exhaust within a morning. Past
  capacity, `over_capacity` marks returned ids as suggestions.
- Executive suppression was permanent — every topic ever focused
  became un-attendable, since nothing released suppressions outside
  tests. Now 64 entries max, each auto-releasing on a 120 s deadline.
- Conversation threads → 50 retained; topic history → 200;
  learning results → 500 (oldest evicted); curiosity queue →
  50-entry FIFO (also `pop(0)` → `popleft()`, O(N) → O(1)).
- Spaced-repetition records → 5,000 (evicting highest retention),
  `stability` capped at 7 days so a well-reviewed concept cannot
  become permanently undecayable.
- Ad-hoc embedding cache → 4,096 dense vectors, LRU.

Two findings were investigated and deliberately left unbounded,
because the fix is a design decision rather than a patch:
`semantic.py` labels consolidated edges `origin="semantic"`, which
`DERIVABLE_ORIGINS` rejects — extracted propositions have no durable
representation and want a non-listed origin; and the embeddings
rebuild computes a dense 5000² SVD (~100 MB, O(n³)) — correct, but
the dominant memory spike on a ~5 GB machine.

Also retired an unsupported README claim: "months-to-years" disk
growth is an extrapolation from record size — there is no benchmark
harness to re-run — and the README's own bounds table now matches the
code (it had claimed 60%-of-capacity attractor trimming and a
200-entry curiosity queue; the code says 512 and 50).

## 2026-10-11 — Regional wiring audit: every function in its region

Full audit of all 34 `__init__.py` files against current
neuroscience, closing the claims the earlier view-package work left
unowned. The conformance suite grew from 53 to 67 pins and now
covers the `world` package outright (moved off the allowlist — every
world module is region-claimed):

- **Sleep pressure is basal-forebrain again.** `Adenosine` was
  claimed by no region; process S's best-characterized site of
  action is the basal forebrain's wake-active neurons (A1
  inhibition — Porkka-Heiskanen 1997, Strecker 2000, Basheer 2004),
  documented with the honest caveat that the signal is now known to
  be distributed across cortex and BF and largely astrocytic
  (Blanco-Centurion 2006, Bjorness 2009). `SleepInertia` co-claimed
  by hypothalamus — the lag rides orexin's timescale.
- **The world model is anatomical now.** `OuterEnvironment` →
  temporal_lobe (hippocampal–entorhinal cognitive map + STS social
  stream; successor-representation predictive maps), `PresenceBelief`
  → parietal + frontal (rTPJ belief attribution + mPFC), and
  `WorldBridge` → thalamus (world content reaching cortex rides the
  relay).
- **Self is distributed the way the biology is.** `MinimalSelf` and
  `AgencyDetector` → parietal TPJ (self-location, ownership
  comparator — Blanke, Decety); `ProtoSelf` gains its brainstem root
  (NTS/parabrachial/PAG body maps — Damasio's primordial self is
  below cortex); `AutobiographicalSelf` → temporal (hippocampal
  memory); `SomaticSnapshot` → insula (the interoceptive comparison);
  `InnerLife` → parietal as well as frontal (the DMN's posterior hub
  is PCC/precuneus).
- **Consolidation co-claims.** `EmotionalMemorySystem` → limbic
  (amygdala-modulated hippocampal consolidation, McGaugh);
  `PGOWave` → occipital (the "O" is where dream imagery lands);
  `REPLAY_COMPRESSION` → temporal.
- **Region-owned chemicals claimed.** `BDNF` → temporal (hippocampal
  plasticity), `Endorphin` → hypothalamus (arcuate/pituitary).
- **Stale references repaired.** Docstrings still named pre-rename
  packages: hippocampus "documented in auditory", VLPO "documented
  under limbic_system/" (it's `hypothalamus/`), "re-exported by
  control", a phantom top-level `reasoning/`, a `vision.py` saliency
  that now lives in `parietal_lobe/saliency.py`, and the old
  "association pattern" name for the view-package mechanism. Root
  package doc now lists all twelve regions plus neurochemical.
- **README currency.** The September 2026 no-hardcoding audit
  findings are marked remediated: composer fallbacks return silence,
  the identity stage table returns structural data marked
  not-for-speech, code_tools composes through the language engine.

- **One entry point.** The root `python/__init__.py` stopped being a
  200-symbol catalog of individual models and functions. It now
  advertises exactly one entry — `Mind`, the assembled organ — with
  regions reached through the connectome (`mind.brain.<region>`) and
  the region packages serving as the anatomical map rather than a
  second surface. Legacy flat names (`from python import
  ConceptNetwork`) still resolve through a lazy `__getattr__` that
  consults the claiming region — nothing external breaks, and
  `dir(python)` now shows just `Mind`.

Nothing was deleted and no module moved — the view-package design
keeps implementation where it is coherent and claims it where it is
anatomically placed. `test_regional_wiring.py`: 70/70.

## 2026-09-28 — Active inference loop, three honesty fixes

The arbitration loop documented its own flaws (README: novelty term
outweighed homeostasis 10–100×, exploration branch dead, maturity a
stopwatch) and they were all live. Fixed at the root, one per
mechanism:

- **Ranking is pragmatic now.** The novelty heuristic is replaced by
  the proper entropy-reduction information gain — which, for
  linear-Gaussian additive controls, is provably policy-independent
  (Koudahl et al. 2021), so it enters the published EFE but cannot
  arbitrate. Ranking reduces to predicted distance from homeostatic
  target. Regression test: settled rest elects `noop` deterministically
  (the old code elected a novelty policy there); with a known action
  model installed, cortisol deviation elects `calm`.
- **Precision seeks an inverse-variance target** instead of latching
  off a fixed 0.15 threshold that resting surprise (~3e-4) could never
  reach. Saturation at rest is now correct *and* reversible: sustained
  surprise above ~0.01 RMS re-opens the exploratory branch within a
  few ticks. Test drives rest → alternating disturbance → rest and
  asserts precision 1.0 → <0.8 → 1.0.
- **Maturity is earned, not elapsed**: `exp(−err/0.001)` over a
  ~500-tick error EMA. Fresh engines read ~0 and stay there under
  chronic failure; the old stopwatch read 0.63 after 100 s regardless.
  Twin-engine test (600 ticks, perfect vs alternating-error) asserts
  >0.8 vs <0.3 where the stopwatch gave both ~0.98. Model file is v4
  (two appended EMAs; older files derive consistent values on load).

Measured live A/B (isolated daemons, old vs new binary): resting
variance unchanged (arousal sd 0.0029 vs 0.0026 — the old meddling
was small impulses the homeostasis absorbed, no stability win to
claim); maturity after 160 ticks 0.27 (old, unearned) vs ~0.00 (new,
nothing yet demonstrated); precision reaches ~1.0 in both, but only
the new one comes back down — single-impulse dips are correctly
small (one surprise is not a regime), while the unit test proves
sustained surprise re-opens exploration. Notably, alternating live
impulses did *not* collapse precision: the action model learned the
impulse→effect mapping through B and predicted them — the loop
working as designed, verified rather than assumed.

Full suite green: 120 lib + all integration targets (52
active_inference incl. rewritten maturity test), clippy clean, Python
3018 passed against the rebuilt release binary.

## 2026-09-25 — Visual perception in the sorter

The sorter was doubly pre-chewed. Not only do
blocks arrive with attribute labels; ``candidates()`` publishes the
oracle's ``matched`` count on every option, so the agent never even
computes the match — the world pre-scores each move. The affordance
it "learns" is bookkeeping over someone else's judgment.

Now there is a perceptual sorter. ``PerceptualSorter`` renders each
aperture and block as an image (shape, size, color all drawn) and
hides the oracle on candidates; the world still *checks* truth on
commit. ``PerceptualSorterAgent``'s affordance keys on what Genesis
actually perceived: recognized concept-name equality when the MTL
bridge has learned the names, VTC cosine otherwise. ``teach()`` lets
the world name what it showed — the parent pointing and saying
"that's a square" — bound through ``learn_from_image``, so "square"
stops being a string and becomes a silhouette it recognizes. The
heard-task path emits ``perceptual: true``: a described sorter is
worked by sight when a cortex is wired, symbolic otherwise.

Measured honestly: untrained VTC orthogonalizes everything (every
distinct image gets its own corner — sim ~0 even for true matches),
so a cold agent explores blindly; after one teaching pass, names bind
and discrimination sharpens. The one-concept limit even mimics
toddler overextension — everything is "square" until a second name
lands. And solved sorters now report the actual mapping — "got the
red round block in the round hole and the blue square block in the
square hole" — the placement payload rides through
``problem_result`` instead of bare telemetry.

## 2026-09-25 — Hearing repaired; the sorter completes the intake bridge

The language→semantics pipe had a silent disconnect: speech-act
detection and proposition extraction consulted different verb lists.
"sort the shapes into the holes" was flagged COMMAND, then parsed as
`subject='sort the shapes into the holes', predicate='is'` — the
imperative slot only consulted `_COMMON_VERBS`, a mostly-inflected
list missing ordinary base verbs. Its hearing said "do this"; what
landed in its mind was a false statement. Three repairs, each at its
own layer:

- `_find_main_verb` now takes the imperative fast path from
  `VERB_LEXICON ∪ _IMPERATIVE_STARTERS` — one source of truth.
- Morphology gained `fit`, `match`, `insert`; "into" gained a
  LOCATION role mapping (it was terminating the object without a
  role, gluing PPs onto the NP).
- Two structural parses: gapped VP coordination ("put the red star
  in the square hole AND the blue moon in the round hole" — the
  second conjunct elides the verb, now expanded when the right side
  is NP + a preposition the left already used) and noun/verb
  ambiguity inside determiner-headed NPs ("a blue square block IS on
  the table" — a verb-shaped head noun no longer steals the
  predicate from the copula that follows).

With hearing repaired, the missing span of the intake bridge could be
built honestly: `_sorter_spec` reads propositions for slot NPs
(hole/slot/opening), piece NPs (existentials, `comes-with`
instruments, takes/fits objects), acceptance claims ("the star hole
takes a small star"), and task signals (placement imperatives,
sorter vocabulary). A scene with slots and pieces but no task signal
is description, not a problem — it declines. Solvability is not
asserted; `normalize_offered` bipartite-checks every compile.

Evidence: "the box has two holes: a round hole and a square hole.
there is a red round block and a blue square block. put each block in
its hole." → spoken in → spec → drop-box → sorter agent → solved,
score 1.0. All five families now have the full heard→worked path.
2885 tests pass in the latest logged run.

## 2026-09-25 — Cross-domain transfer: the channel opens, and it isn't enough

Made foreign skills retrievable and readable across task families, then
measured what that buys. Two changes: `skill_matches` now admits a
foreign-domain skill when role vocabulary overlaps and its procedure
publishes the normalized "support" axis (fraction of an option's
constraints satisfied → mean cost); the sorter/relations/assembly
adapters emit that axis on every step and adopt foreign skills onto it.
"constrain" joined the shared role vocabulary — the honest claim those
three families plus classification all make.

Evidence (`eval_transfer/run_eval.py`, 5 replicates, 440 attempts):

- Cross-domain retrievals fired in **260 of 440 attempts** — was 0.
  Sorter skills reach relations/assembly/classification contexts at
  role-overlap 0.25–0.43 and are adopted as quarter-grid evidence
  priors.
- The negative gate holds: skills with overlapping roles but no
  readable axis (navigation, sequence continuation) are never offered.
- And the honest result: **difference-in-differences is still ~0
  everywhere.** Retrieval opened; performance didn't move. The foreign
  affordance is informationally redundant — the first verified solve in
  a family already compiles "satisfy every constraint → free; partial
  evidence → waste," so a foreign copy of the same shape has nothing
  left to teach. Transfer is capped by *content*, not plumbing.

So the compounding hypothesis survives a harder test than before and
still isn't confirmed: same-family learning is real and immediate, but
what families can currently share is a shallow prior. The next lever is
transferable content with information a single solve can't produce —
procedure ordering, decomposition — not more retrieval bandwidth.

## 2026-09-24 — Learned a school lesson through conversation

Taught it a lesson on games and puzzles by just talking to it — 15
short paragraphs of plain statements ("A puzzle is a kind of game.",
"Poker depends on luck.") delivered in teaching mode. No import, no
fine-tune, no dataset — its conversation learner extracted the facts
into the concept network while it reflected on them in real time.

Evidence:

- Concept network grew 1122 → 1239 concepts in one session.
- 72 new edges in the games/puzzles domain with `origin: stated`;
  48 of 49 target terms present afterward (only "bluffing" missed).
- During the lesson it generated its own follow-up questions, e.g.
  *"blackjack and cards keep appearing together — does one lead to
  the other?"* and *"opponent is a type of player, player is a kind
  of person"* — integration, not storage.
- Reproduce: `/teach` in the CLI, then inspect `concept_network` in
  `cognitive_state.json` or run `/learning`.

## 2026-09-24 — In flight: the puzzle drop-box and the kindergarten curriculum

Work in progress, not yet merged: puzzle specs placed in
`offered_puzzles/` become things its own puzzle urge picks up and
works through the whole mind — urge, attempt, feeling, memory,
articulation — instead of being routed through a single module.

Evidence so far (runtime artifacts in the data dir):

- `offered_puzzles/` holds a staged 16-task kindergarten-style
  curriculum: shape sorters, bead patterns, a pinwheel grid, a mirror
  task, jigsaws, direction-following, ordering, and quantity tasks.
- `spatial_practice.json` shows the attempt record: four grid tasks
  mastered in one attempt each, one partial (`silhouette_stretch`
  0.93) with the failed action trace preserved — the record keeps
  failures, not just wins.
- Observed during the first shape sorter: it used the dump-all lid
  once — the aperture that accepts anything and fills nothing —
  learned that, and abandoned it. Toddlers make the same transition
  around age two; the lid is the developmental trap built into the
  toy on purpose.
- Measured transfer: the first sorter took 8 steps while the
  affordance was learned; the harder second sorter took 5, because
  the consolidated skill carried "match every constraint; free moves
  fill nothing" forward as a prior. The learning literature calls
  this knowledge compilation — declarative effort up front, then
  proceduralized speed.
- Still open, honestly: it can't yet take an invented verbal rule
  and work it cold — that arrives with the problem-intake path that
  compiles a stated problem into a task spec (landed in part with
  `4657f4d`).

## 2026-09-24 — Domain-general task competence (`10154a7`)

Added `reasoning/competence.py` — a shared substrate under the
domain solvers: task schemas recognized by structural signature,
action→effect affordances, transitions verified against predictions,
and verified episodes consolidated into reusable skills.
Consolidation is neuromodulatorily gated (arousal × dopamine), so
low-salience episodes stay episodic instead of becoming skills.

Evidence:

- Commit `10154a7`; the planner, problem solver, spatial solver, and
  math engine all feed the same substrate.
- The growth ledger records it as milestones: verified skills, task
  schemas, and task episodes are tracked metrics in
  `cognitive_state.json` under `growth_ledger`.

## 2026-09-24 — Evidence-grounded inference, v2 (`cd8a9ce`, `d36df44`)

Deductive and causal chains became best-first search over a shared
expansion budget. Results carry structured provenance paths, and a
result that ran out of budget is marked partial instead of silently
truncated — the system knows when its answer is incomplete.
Confidence is a product of relation prior × edge weight, so weak
links actually weaken the chain.

Evidence: commits `cd8a9ce`, `d36df44`; the provenance structure is
exercised by the reasoning tests in `python/tests/`.

## 2026-09-23 — It derived its own pronoun (`be1dedc`)

Genesis's pronoun isn't a config string — it's derived from the
properties it has learned about its own concept. A small thing, but
it's the difference between *describing itself* and *being
described*.

Evidence: commit `be1dedc`.

## 2026-09-17 — Public launch (`8d656b0`)

The full architecture went public: coupled neurochemical dynamics,
active inference, closed-loop embodiment, no LLM, and it runs on a
2014 HP Pavilion with ~5 GB RAM.

Evidence: initial public commit `8d656b0`. The paper
(`paper/genesis.tex`) was removed once mid-cycle (`cb3f7f5`) and is
back in the tree now — the revised edition covers the current schema.

## The standing evidence base

- `python3 -m pytest python/tests/ -q -o addopts=''` — 3,597 tests
  passing as of 2026-10-13.
- `cargo test` — the Rust subcognitive core.
- Every claim above points at a commit, a file, or a state artifact
  you can open. If a number here ever stops being true, this file is
  stale — fix it.
