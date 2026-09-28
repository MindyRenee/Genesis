# Genesis Devlog

A running log of what Genesis has done — the breakthroughs and the
evidence behind them, not just the claims. Entries are dated, newest
first. Code references are commit SHAs; runtime artifacts live in the
instance's data dir (`~/.local/share/genesis*`).

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
stops being a string and becomes a silhouette she recognizes. The
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
list missing ordinary base verbs. Her hearing said "do this"; what
landed in her mind was a false statement. Three repairs, each at its
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

Evidence: initial public commit `8d656b0`; the paper is in
`paper/genesis.tex`.

## The standing evidence base

- `python3 -m pytest python/tests/ -q -o addopts=''` — 2885 tests
  passing in the latest logged run.
- `cargo test` — the Rust subcognitive core.
- Every claim above points at a commit, a file, or a state artifact
  you can open. If a number here ever stops being true, this file is
  stale — fix it.
