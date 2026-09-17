"""Evaluation: ARC-AGI-3 interactive environments (pilot).

Plays a small set of ARC-AGI-3 games offline with the novelty-driven
SpatialAgent. Unlike the static ARC eval, there are no taught
examples — the agent must discover what each action does by acting.

Environments are downloaded once via the arc_agi toolkit (anonymous
API key) into evals/arc3_envs/, then run fully offline.

Conditions: one per game. A game "passes" if the agent completes at
least one level within the step budget — for a v1 novelty policy,
most games will fail; the eval exists to establish the baseline and
to verify the perception→action loop mechanics work end to end.

Usage:
    PYTHONPATH=python .venv-arc3/bin/python evals/eval_arc3.py
    (or via run_evals.py once the venv has arc_agi installed)
"""

from __future__ import annotations

import time
from pathlib import Path

from harness import (
    EvalResult,
    FactResult,
    print_result,
    run_condition,
)

_ENVS_DIR = Path(__file__).resolve().parent / "arc3_envs"

# Small, representative pilot set — downloaded on first run.
_PILOT_GAMES = ["ls20", "tu93", "ft09"]


def _play_game(game_id: str, max_steps: int, seed: int) -> FactResult:
    """Play one game with the SpatialAgent; report outcome."""
    try:
        import arc_agi
        from arcengine import GameState
    except ImportError:
        return FactResult(game_id, False, "arc_agi toolkit not installed")

    from genesis_cognitive.spatial import SpatialAgent

    # Prefer OFFLINE once downloaded — no API calls, no scorecards.
    # NORMAL (which downloads missing games) only as fallback.
    env = None
    last_err: Exception | None = None
    for mode in (arc_agi.OperationMode.OFFLINE, arc_agi.OperationMode.NORMAL):
        try:
            arc = arc_agi.Arcade(
                operation_mode=mode,
                environments_dir=str(_ENVS_DIR),
            )
            env = arc.make(game_id)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if env is None:
        return FactResult(
            game_id, False,
            f"could not load game: {last_err}",
        )

    agent = SpatialAgent(seed=seed)
    obs = env.reset()
    agent.observe(obs.frame)

    steps = 0
    won_levels = 0
    while steps < max_steps:
        action = agent.choose_action(list(env.action_space))
        data = None
        is_complex = getattr(action, "is_complex", False)
        is_complex_value = is_complex() if callable(is_complex) else bool(is_complex)
        if is_complex_value:
            frame = getattr(agent, "_last_frame", None)
            shape = getattr(frame, "shape", None)
            if shape is not None and len(shape) >= 2:
                h, w = int(shape[0]), int(shape[1])
            else:
                h, w = (64, 64)
            data = agent.action_data(action, (h, w))
        obs = env.step(action, data=data) if data else env.step(action)
        steps += 1
        if obs is None:
            break
        agent.observe(obs.frame)
        agent.mark_goal_reached()
        state = getattr(obs, "state", None)
        if state == GameState.WIN:
            won_levels = getattr(obs, "levels_completed", won_levels + 1)
            agent.on_episode_end("WIN")
            obs = env.reset()
            agent.observe(obs.frame)
        elif state == GameState.GAME_OVER:
            agent.on_episode_end("GAME_OVER")
            obs = env.reset()
            agent.observe(obs.frame)

    tried = {
        a.name: f"n={s.attempts},Δ={s.mean_change:.3f}"
        for a, s in agent.stats.items()
    }
    if won_levels > 0:
        return FactResult(
            game_id, True,
            f"completed {won_levels} level(s) in {steps} steps; "
            f"actions: {tried}",
        )
    return FactResult(
        game_id, False,
        f"no level completed in {steps} steps; actions: {tried}",
    )


def run(seed: int = 42, max_steps: int = 500) -> EvalResult:
    """Run the ARC-AGI-3 pilot evaluation."""
    start = time.time()
    result = EvalResult(
        name="arc3_interactive",
        description=(
            "Interactive environments: discover goals through action "
            "(ARC-AGI-3 pilot, novelty-driven policy)"
        ),
    )
    for game_id in _PILOT_GAMES:
        result.conditions.append(
            run_condition(
                game_id,
                f"ARC-AGI-3 environment {game_id}, {max_steps}-step budget",
                lambda gid=game_id: [
                    _play_game(gid, max_steps=max_steps, seed=seed)
                ],
            )
        )

    result.duration_seconds = round(time.time() - start, 3)
    total = sum(c.total for c in result.conditions)
    correct = sum(c.correct for c in result.conditions)
    result.overall_accuracy = correct / total if total else 0.0
    result.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    return result


if __name__ == "__main__":
    print_result(run())
