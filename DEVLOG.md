# Genesis Devlog

A running log of what Genesis has done — the breakthroughs and the
evidence behind them, not just the claims. Entries are dated, newest
first. Code references are commit SHAs; runtime artifacts live in the
instance's data dir (`~/.local/share/genesis*`).

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
keyword owns the word) and `python/frontal_lobe/self/` (the mind's)
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
- **The world model is anatomical now.** `OuterWorld` →
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
