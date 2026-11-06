# ARC upstream compatibility audit

**Audit date:** 2026-11-02  
**Branch:** `genesis3`  
**Upstream references checked:**

- ARC-AGI Toolkit repository and API reference: https://github.com/arcprize/ARC-AGI
- Toolkit documentation: https://docs.arcprize.org/
- Full Play Test: https://docs.arcprize.org/full-play-test
- ARC Prize ARC-AGI-3 overview: https://arcprize.org/arc-agi/3

## Compatibility decisions

Genesis uses the official `arc-agi` Python toolkit for ARC-AGI-3 environments. The adapter uses the documented `Arcade`, `make`, `get_environments`, `EnvironmentWrapper.action_space`, `step`, `reset`, and scorecard APIs. Genesis supplies its own policy; the toolkit is the environment/API layer, not the reasoning system.

ARC-AGI-1 remains a separate static-grid path. Its 0–9 color validation and task format are not applied to ARC-AGI-3 frames. ARC-AGI-3 click coordinates are validated separately as integer `x` and `y` values in the toolkit's 0–63 action coordinate range.

## Fixes made in this audit

- **No-action state:** `step_arc3_game` now refuses to submit a non-reset action when the toolkit exposes an empty legal-action set. It no longer treats an empty set as permission to send any action.
- **Action-name validation:** malformed or empty action names are rejected before importing or looking up toolkit enum members.
- **Terminal-state integrity:** the phone bridge now rejects actions after `WIN` instead of returning a successful-looking response while silently ignoring the requested action.
- **Reward feedback:** Genesis' phone-bridge autonomous step now passes the same level-progress and terminal reward calculation used by the bounded episode runner into the exploration policy. Previously, that path omitted explicit reward feedback.
- **Termux install path:** the phone bridge instructions now install `arc-agi==0.9.9` directly on Termux, avoiding the full editable project's desktop-oriented base dependencies. Desktop installation through `.[arc]` remains documented separately.
- **Regression tests:** tests were added for empty legal-action sets, malformed action names, post-WIN action rejection, and reward propagation.

## Runtime verification status

The code and tests were updated on `genesis3`. The test suite has **not** been executed in this audit environment, and no live ARC game was opened here. Run the following from the repository root in Termux after the toolkit is installed:

```sh
PYTHONPATH=python:. python -m pytest python/tests/test_arc_sources.py python/tests/test_arc_play.py python/tests/test_arc_phone_bridge.py
```

Then check the CLI import path without opening a live game:

```sh
PYTHONPATH=python:. python scripts/arc_phone_bridge.py --help
```

A passing fake-environment test suite verifies adapter logic; it does not establish that a live API key, network connection, available game catalog, or autonomous solving performance has been verified.
