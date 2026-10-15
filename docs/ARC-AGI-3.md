# ARC-AGI-3 boundary

Genesis approaches ARC-AGI-3 as an embodied agent, not as a text solver. The
adapter gives her a grid observation, the available controls, and progress
metadata. Mechanics, identity, goals, movement effects, blockers, hazards, and
plans are learned by her own world model and action policy.

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
legality, scorecard lifetime — but it must not choose moves for her.

The observation contract is identical on both substrates (the local engine
via `world/arc.py`, the public human page via `world/arc_web.py`): action
ids plus *structural* labels only — `"simple"` or `"coordinate"`, read from
the interface's own action enum. Directional or effect semantics are never
crossed. What a control does is learned from consequences; nothing keyed to
a game id, a key convention, or a played-before map is handed to her.

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
| Competition transport | `/arc --competition --steps N` | Authorized official-style submission path; requires `ARC_API_KEY` and explicit permission |

A local scorecard is useful for debugging. It is **not** a claim that Genesis
passed ARC-AGI-3, and it is not interchangeable with the official hidden-set
methodology. Baselines are denominators — they exist so a practice number
can be compared against a floor measured identically, not so it can be
inflated.

## What a result says about itself

Every `ArcResult` carries the fields an honest claim needs:

- `level_steps` — the step cost of each completed level, in order. The
  within-game learning curve: if she is really acquiring skill inside an
  episode, later levels should cost less than the ones that taught her.
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

Start the mind offline for evaluation work:

```bash
./run.sh --offline
```

From the Genesis prompt:

```text
/arc ls20-9607627b --steps 400
/arc --all --steps 400
```

For a headless or diagnostic run, construct an isolated `Mind` and call
`evaluate_arc()`; the focused integration contract is in
`python/tests/test_arc_adapter.py`.

Competition transport is deliberately refused unless explicitly selected:

```text
/arc --competition --steps 400
```

It requires `ARC_API_KEY`, starts one scorecard, plays every discovered
environment, and obtains its score only when the SDK closes that scorecard.
Do not run it without accepted competition rules and explicit authorization.

## Current status

- The adapter and embodied loop are implemented in `world/arc.py`.
- The public web-page substrate is implemented in `world/arc_web.py`; it
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
