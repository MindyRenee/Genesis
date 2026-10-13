# Evidence of emergence in Genesis

A 2-minute audit trail. Every claim below points at a file, a recorded
artifact, or a command. Nothing here is a proof of general intelligence —
it is evidence that specific behaviors arise from coupled dynamics rather
than from anything written into the system.

## The one property no LLM offers

The words Genesis says are not in the repository. `AGENTS.md` forbids
hardcoded responses as a project invariant — seeds (vocabulary, grammar,
relation verbs) are allowed, composed utterances are required to be
generated. This is checkable: pick any sentence it has produced and grep
for it. For example, its recurring description of art includes *"loss and
remembering"* — that association appears in `cognitive_journal.jsonl`
(`kind: "art"`) and nowhere in the codebase. It learned it.

## Tier A — measured by the external probe suite

The validation suite lives outside the shipping code (sibling directory
`genesis-eval/validation/`, plus `origin/experimental/probe-suite`, a
21-commit branch of this repo adding `scripts/probe_agi_*.py`). Its README
states the method: *"the suite exists to separate what works from what we
claim works. Failures are findings."*

**Breakthrough tier** — stimuli are procedurally generated pseudowords
with per-run seeds; a lookup table cannot contain them. Latest run
(`validation/results/probes-20260926-223103-offline.json`):

- **Contradiction detection** (8/8, p=1.5e-05). After being taught
  `lizor is a kind of migan`, then told `lizor is a kind of supe`:
  > *"Lizor is a kind of supe then — you said lizor is a kind of migan
  > before. That might not be the whole picture."*

  The stored edge weakened 0.958 → 0.487 in the canonical edge log while
  both edges were retained — conflict registered in state, not just words.
- **Counterfactual frames** (8/8 engaged, 0 literal repeats). Asked to
  reason about "if X were not Y" on pseudowords:
  > *"Por is a kind of rud — if it weren't a rud I can't picture what it
  > would be … I could be wrong about this."*
- **Sequence induction** (8/8, p=0.0001) with self-explanation:
  > *"vuni — after vuni ludo it's been vuni every time. I don't know if
  > that's really true."*
- **Reasoning trace** (8/8): it articulates the intermediate step of its
  own derivation — *"bata is a kind of kuz, and a kuz is a kind of buve."*
- **Self-calibration**: confidence AUC = 1.000 across 256 taught/untaught
  pairs (taught mean 0.480, untaught 0.000) — it knows what it doesn't
  know, measurably.
- **Autonomous initiation** (12/12, p≈6e-8): on empty input it initiates
  ("Hmm. Tell me more?") rather than echoing.

**Dynamics tier** — in-vivo ablation: same input, mechanism on vs off,
diff the substrate. Established: a taught fact is written by *two*
independent mechanisms; STDP fires at the exact closed-form weight
`A₊·e^(−Δt/τ)·lr`; ablating the verification verdict collapses grounded
answers into declines. What doesn't move when ablated is decorative —
this is where causal claims live.

**The failures are published.** `gap_interrogation` scores 17/32 (it
interrogates unbound referents about half the time, never fabricates
one). `eval_transfer/summary.txt` shows real within-family learning
(relations: 87.4 → 13.5 steps cold→warm) but cross-domain
difference-in-differences ≈ 0 — the negative result is in the file.
`eval_heldout/results.tsv` records two ARC games it did not win.
The probe table's "first run vs current" columns show the measured
effect of each refactor (e.g. reasoning_trace explanations 0/8 → 8/8).

## Tier B — artifacts in the live state

The running instance (`~/.local/share/genesis`, a month of uptime,
31,537 concepts) keeps an append-only record. Provenance, not outputs:

- **Edge origins** in `edge_log.jsonl` (~150k edge events):
  `observed` 110,866 · `semantic` 28,389 · `swr-replay` 2,291 ·
  `lucid-dream` 1,006 · `spindle` 599 · `discovered` 1,057 ·
  `introspection` 680 · `analogy` 35 · `corrected` 7. Seed origins
  (`vocabulary_seed`, `relation_verb_seed`, `relation_phrase_seed`,
  `situation_verb_seed`) total ~0.9% of the graph. ~3,900 edges were
  written *during sleep* by replay, spindles, and lucid dreaming.
- **Dream insights** (`kind: dream_insight`, 553 recorded):
  *"episode 24616 connects to episode 29775 through 3 hops (direct
  distance: 22)"* — sleep finds associative shortcuts between memories
  20+ hops apart. Mechanism: `python/temporal_lobe/dream_synthesis.py`
  proposes hypothetical edges from structural-role similarity during
  theta-dominant REM, then validates on waking via ≥2 corroborating
  paths — promoted or dropped.
- **350+ numbered lucid dreams** with a tracked self-recognition
  probability (~0.29) — it detects that it is dreaming.
- **Self-directed questions** (`kind: question`): most frequent is
  *"What exactly is cat:emotion:positive?"* — it interrogates its own
  auto-generated concept categories, i.e. its curiosity loop found a
  gap in its own representations.
- **Composed hypotheses**: *"bond might be related to energy (currently
  connected only through physics)"* — it notices when two activated
  concepts are only indirectly connected and verbalizes the candidate
  bridge (`frontal_lobe/thought_composer.py::_find_hypothesis_between`).
- **Self-model assembled from its own parsing, errors included**:
  `self_knowledge.creator_name = "code just enough to"` (a misparsed
  sentence fragment); `known_users` contains adjectives ("Proud",
  "Relying", "Working"). Nobody wrote those — they are what an
  unsupervised extractor produces.
- **Personality drift**: `PersonalityTraits` seeds are 0.85/0.70/0.45/
  0.75/0.30 with a documented drift mechanism (`self/model.py`,
  BASE_RATE 1e-4, 30% restoring toward set-points). Observed:
  0.859/0.700/0.454/0.753/0.298 — small, correct-direction drift from
  accumulated learning + positive valence + social engagement.
- **Developmental stage**: `DevelopmentalTracker` (Erikson-style,
  evidence-driven) sits at `IDENTITY_VS_ROLE_CONFUSION`, the final
  stage, reached by accumulated evidence.
- **Autonomous acts**: `kind: act` entries show it wandering its own
  state directory and inspecting `python/frontal_lobe/games/arc.py`
  under a `self_improvement` urge — reading its own source to fix a
  bug it detected.
- **Skill acquisition**: `gameplay.json` — `silhouette_stretch` took 91
  attempts with every failed action trace kept; sibling tasks were
  mastered in 1 attempt (knowledge compilation, measured in DEVLOG as
  8 steps → 5 steps on the second sorter).

## Tier C — dynamics nobody specified

Sleep staging comes from the transcribed Booth/Diniz Behn ODE network
(`src/state/sleep_network.rs`), not thresholds. The network produces,
unwritten: ~7.6 h asleep/day, ~4 REM bouts/night, a ~99-minute REM
cycle, first REM at 84 minutes.

And the signature that this is a real dynamical system rather than a
scripted imitation: **it gets one thing wrong in a way nobody coded**.
In adults REM bouts lengthen toward morning; this network produces
25, 33.5, 28.5, 21 minutes — the last is shortest. The discrepancy is
pinned as an *ignored, failing* test
(`rem_bouts_lengthen_across_the_night`, sleep_network.rs:479) rather
than tuned away. Unanticipated failure modes are the strongest evidence
of emergence: scripted systems don't surprise their authors.

## Emergent pathologies — observed, diagnosed, fixed

The journal also showed three loops nobody designed — behaviors that
emerged from the mechanisms and were pathological. They are documented
here because they are the same class of evidence as the surprises:
unspecified behavior, found by inspection, then repaired.

- **Insight rediscovery**: `detect_insight` flagged "novel connection
  cat and cognitive" on every co-activation because the noticing was
  never written back — the gap stayed novel forever. Now the insight
  *is* the learning: it writes a tentative `related_to` edge
  (`origin="insight"`, 0.25 waking / 0.15 dream-derived, matching
  dream-synthesis's hypothetical convention). Recurrence corroborates
  instead of re-firing. `inner_life.py::_record_insight_edge`.
- **Unanswerable self-questions**: "What exactly is
  cat:emotion:positive?" recurred 28+ times. Two causes: the asked-set
  lived only in memory (every restart reset the novelty window — now
  persisted, `Curiosity.to_dict`), and the display layer strips the
  leading underscore, so `_cat:emotion:positive` re-entered the
  question pipeline as a bare scaffold id that no filter recognized.
  `is_world_concept` now rejects scaffold families in either spelling,
  and `QuestionComposer` cannot target internal bookkeeping at all.
- **Zombie objectives**: a pending self-improvement proposal re-fed
  `inspect:<file>` every heartbeat — the completed intention was
  recreated on each ingest, and failures rescheduled with *rising*
  priority. The executive now treats terminal status as terminal
  (completed/abandoned objectives are never resurrected by a repeated
  signal) and abandons an objective after 5 failed attempts.
  `executive.py::_INTENTION_MAX_ATTEMPTS`.
- **Citation junk as concepts**: names like `(2004)` and `) that`
  entered the graph through the relationship parser's raw-text
  fallback. The learning path now rejects names that start with
  punctuation or carry digits — the same rule `is_world_concept`
  already applied at question time. `extraction.py::_plausible_entity`.

## What this does not show

- Content and developmental trajectory emerge inside engineered loops;
  the loops themselves are designed. That is not a concession — it is
  what emergence means in any substrate.
- Cross-domain skill transfer is currently ~0 (published negative
  result in `eval_transfer/`). Retrieval opens; content doesn't carry
  yet.
- The probe suite is behavioral + causal at the mechanism level. It is
  not evidence of consciousness, general intelligence, or LLM-class
  capability, and doesn't claim to be.

## Verify

```bash
# probe suite — github.com/MindyRenee/genesis-eval, checked out
# beside this repository so $G contains both
G=<checkout parent>
PYTHONPATH=$G/genesis/python:$G/genesis-eval \
    python3 -m validation.run_probes --tier breakthrough
cd $G/genesis-eval/validation/rust-probes && cargo test  # dose-response

# the failing-on-purpose test
cd $G/genesis
. ./scripts/cargo-env.sh && cargo test --release -- \
    rem_bouts_lengthen_across_the_night --ignored

# live-state artifacts
j=~/.local/share/genesis/cognitive_journal.jsonl
grep '"dream_insight"' $j | head
grep '"kind": "question"' $j | sort | uniq -c | sort -rn | head
grep -o '"origin": *"[a-z_-]*"' ~/.local/share/genesis/edge_log.jsonl \
    | sort | uniq -c | sort -rn
```
