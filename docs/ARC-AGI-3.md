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

## Practice vs. official evidence

There are three materially different surfaces:

| Surface | Command path | Meaning |
| --- | --- | --- |
| Selected local games | `/arc <game-id> --steps N` | Development diagnostics only |
| Local discovered suite | `/arc --all --steps N` | One SDK scorecard over local environments; still not hidden-set evidence |
| Competition transport | `/arc --competition --steps N` | Authorized official-style submission path; requires `ARC_API_KEY` and explicit permission |

A local scorecard is useful for debugging. It is **not** a claim that Genesis
passed ARC-AGI-3, and it is not interchangeable with the official hidden-set
methodology.

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
- The public web-page substrate is implemented in `world/arc_web.py`.
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
- and a statement separating public practice results from hidden-set results.

Repository status badges and tests are necessary supporting evidence, but they
are not a benchmark result.
