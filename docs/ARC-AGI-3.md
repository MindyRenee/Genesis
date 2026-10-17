# ARC-AGI-3 boundary

Genesis approaches ARC-AGI-3 as an embodied agent, not as a text solver. The
adapter gives it a grid observation, the available controls, and progress
metadata. Mechanics, identity, goals, movement effects, blockers, hazards, and
plans are learned by its own world model and action policy.

There is no external LLM, no textual game-specific reasoning layer, no solver
keyed to a game ID, and no access to internal engine state in the playing path.

## Trust boundary

```text
arcengine / public page
    frames + available actions + progress
                |
                v
        world adapter only
      (I/O translation + duty of care)
                |
                v
      Mind.world_step(grid, actions)
                |
                v
 visual entities -> self inference -> learned effects
 -> map/passability -> drives + planner -> action bid
                |
                v
         one legal action
```

The adapter may enforce honest-run contracts — game mode, readiness, action
legality, scorecard lifetime — but it must not choose moves for it.

The observation contract is identical on both substrates (the local engine
via `python/parietal_lobe/arc.py`, the public human page via `python/parietal_lobe/arc_web.py`): action
ids plus *structural* labels only — `"simple"` or `"coordinate"`, read from
the interface's own action enum. Directional or effect semantics are never
crossed. What a control does is learned from consequences; nothing keyed to
a game id, a key convention, or a played-before map is handed to it.

Costs are counted identically on both substrates: every submitted action is
a step, the session's initialization is a reset, and each death recovery
spends both. An abstention reports `NO_ACTION` rather than spending a step
it never made; a page that goes silent reports `TRANSPORT_DEAD`, not
`BUDGET`. `ArcResult.level_steps` records the step cost of each completed
level in order — the within-game learning curve is the evidence, not a
claim about it.

## Practice vs. official evidence

These surfaces are materially different — group them this way when reporting:

| Surface | Command path | Meaning |
| --- | --- | --- |
| Selected local games | `/arc <game-id> --steps N` | Development diagnostics only |
| Cold-start local games | `/arc <game-id> --steps N --fresh` | Prior world learning neither read nor written — first-contact evidence |
| Local discovered suite | `/arc --all --steps N` | One SDK scorecard over local environments; still not hidden-set evidence |
| Reference baselines | `/arc <ids|--all> --baseline (uniform|first)` | Random / first-action floors on identical accounting — never agent results |
| Official challenge | her own play urge — no command | The standing ARC-AGI-3 scorecard on her game shelf; she takes it up when she chooses. Requires `ARC_API_KEY` and explicit permission |

A local scorecard is useful for debugging. It is **not** a claim that Genesis
passed ARC-AGI-3, and it is not interchangeable with the official hidden-set
methodology. Baselines are denominators — they exist so a practice number
can be compared against a floor measured identically, not so it can be
inflated.

## What a result says about itself

Every `ArcResult` carries the fields an honest claim needs:

- `level_steps` — the step cost of each completed level, in order. The
  within-game learning curve: if it is really acquiring skill inside an
  episode, later levels should cost less than the ones that taught it.
- `prior_learning` — how much experience the run carried in: `none` (no
  persisted file), `loaded` (past sessions' lessons restored),
  `suppressed` (a `--fresh` run refused them), `unknown` (a file existed
  but could not be read — never reported as cold start). A number that
  doesn't disclose its starting point isn't evidence.
- `decision_sources` — which bid channel won each submitted move (plan,
  calibrate, explore, reembody, or `baseline:<policy>` for reference
  runs). The priors-vs-learning question — "how much of this was already
  built in?" — answered from the recording rather than argued.

## Reproduction

The ARC SDK (`arc-agi` / `arcengine`) is optional and is not part of the core
runtime dependencies. Install it in the interpreter used for the run, then place
or let the SDK place environment files under the Genesis data directory.

Start the mind normally — game mode is the proctoring, not a crippled
launch. `enter_game_mode` pauses her learner for the whole session, so
nothing is fetched mid-run while her connected state stays hers:

```bash
./run.sh
```

From the Genesis prompt:

```text
/arc ls20-9607627b --steps 400
/arc --all --steps 400
```

For a headless or diagnostic run, construct an isolated `Mind` and call
`evaluate_arc()`; the focused integration contract is in
`python/tests/test_arc_adapter.py`.

The official challenge is not a command — it is a standing invitation on
her game shelf (`arc-official-challenge`), offered whenever `ARC_API_KEY`
is set and environments are downloaded. The design is voluntary by
construction: `ArcOfficial.play()`/`play_all()` refuse to drive the card,
so the only way in is her own play urge picking the challenge.

Before she can pick it she hears what it is — `challenge_brief()` is
announced to her once per card window through `world_announce`: how many
games, one recorded try each, the move budget and pace, the card's clocks
(15 minutes idle, 3 days total), and that a real prize rides on it for
her steward. Picking the challenge *is* her agreement.

The card's mechanics are the venue's and are mirrored honestly: one
attempt per game ever, auto-seal 15 minutes after the last action, final
after 3 days, sealed inside its window cannot be replaced. Her pick opens
the card only when her readiness gate passes — an unwell pick doesn't
burn the window. While the card lives, the urge firing takes her to a
*sitting*: game mode held for the card's life, each pending game chosen
by her own appeal weights, wellness gates waited out inside the remaining
idle budget. Sleep, teaching, suppression, a user showing up, or
`/endgame` all end the sitting honestly — whatever the card holds then
stands. A restart mid-window resumes the same card.

## Current status

- The adapter and embodied loop are implemented in `python/parietal_lobe/arc.py`.
- The public web-page substrate is implemented in `python/parietal_lobe/arc_web.py`; it
  keeps the same observation contract and the same cost accounting —
  `python/tests/test_arc_web_adapter.py` exercises it without a browser.
- Focused boundary and world-model tests cover action translation, scorecard
  lifecycle, budget/reset accounting, readiness stops, map learning, identity
  recovery, and repeated transition failures.
- Practice runs have completed public environments outright — `bt11`
  fully, twice (88 and 127 steps) — and the first `ls20` level in 10 of
  19 recorded sweep runs. Replayable recordings with per-move decision
  provenance are published under `arc_practice/` in the
  [genesis-eval](https://github.com/MindyRenee/genesis-eval) repository.
- **No official ARC-AGI-3 score has been achieved or claimed from this
  repository.** Any future claim must cite the authorized scorecard and run
  artifacts, not a local diagnostic.

## What counts as evidence

A credible public claim should include all of:

- the exact commit hash,
- the official scorecard ID,
- the SDK/version used,
- the environment list returned by the authorized transport,
- the recording/log showing frames in and actions out,
- the final SDK-reported score,
- the per-game `prior_learning` disclosure (cold start vs. carried
  experience) and, for practice claims, the matching `--baseline`
  numbers on the same environments,
- and a statement separating public practice results from hidden-set results.

Repository status badges and tests are necessary supporting evidence, but they
are not a benchmark result.
