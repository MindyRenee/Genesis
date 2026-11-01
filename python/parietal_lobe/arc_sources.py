"""Official ARC-AGI-1 data and ARC-AGI-3 environment adapters.

This module does not choose actions or solve tasks. It exposes validated
ARC-AGI-1 examples and the official ARC-AGI-3 environment API to Genesis'
own reasoning/action-selection systems. The optional arc extra provides
the ARC-AGI-3 toolkit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_ARC1_ROOT = Path("data/arc/ARC-AGI-1")


class ARCSourceError(RuntimeError):
    """Raised when official ARC resources are missing or malformed."""


def arc1_root(root: str | Path | None = None) -> Path:
    """Resolve the local clone of the official ARC-AGI-1 repository."""
    if root is not None:
        return Path(root).expanduser()
    configured = os.environ.get("GENESIS_ARC1_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_ARC1_ROOT


def list_arc1_tasks(
    split: str = "training", root: str | Path | None = None
) -> list[Path]:
    """List task JSON files from the official ARC-AGI-1 data split."""
    if split not in {"training", "evaluation"}:
        raise ValueError("split must be 'training' or 'evaluation'")
    directory = arc1_root(root) / "data" / split
    if not directory.is_dir():
        raise ARCSourceError(
            f"ARC-AGI-1 {split} data not found at {directory}. "
            "Run scripts/install_arc_resources.sh or set GENESIS_ARC1_DIR."
        )
    return sorted(directory.glob("*.json"))


def _validate_grid(grid: Any, *, context: str) -> list[list[int]]:
    """Validate an ARC color grid and return it unchanged.

    Args:
        grid: Candidate grid represented as nested lists of integer colors.
        context: Location label included in validation errors.

    Raises:
        ARCSourceError: If the grid is empty, ragged, oversized, or invalid.
    """
    if not isinstance(grid, list) or not grid:
        raise ARCSourceError(f"{context}: grid must be a non-empty list")
    if any(not isinstance(row, list) or not row for row in grid):
        raise ARCSourceError(f"{context}: each grid row must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ARCSourceError(f"{context}: grid must be rectangular")
    if len(grid) > 30 or width > 30:
        raise ARCSourceError(f"{context}: ARC grid dimensions cannot exceed 30x30")
    for row in grid:
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise ARCSourceError(f"{context}: cells must be integers in range 0..9")
    return grid


def load_arc1_task(path: str | Path) -> dict[str, Any]:
    """Load and validate an ARC task JSON file.

    Both training and test inputs are validated; outputs are validated when
    present. Test outputs remain in the loaded source payload but should be
    removed with :func:`arc1_problem_view` before solving.
    """
    task_path = Path(path)
    try:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ARCSourceError(f"Could not read ARC task {task_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("train"), list):
        raise ARCSourceError(f"{task_path}: expected a task object with a train list")
    if not isinstance(payload.get("test"), list) or not payload["test"]:
        raise ARCSourceError(f"{task_path}: expected a non-empty test list")
    for section in ("train", "test"):
        for index, pair in enumerate(payload[section]):
            if not isinstance(pair, dict) or "input" not in pair:
                raise ARCSourceError(f"{task_path}: {section}[{index}] lacks input grid")
            _validate_grid(pair["input"], context=f"{task_path}:{section}[{index}].input")
            if "output" in pair:
                _validate_grid(pair["output"], context=f"{task_path}:{section}[{index}].output")
    return payload


def arc1_problem_view(
    task: dict[str, Any], *, include_test_outputs: bool = False
) -> dict[str, Any]:
    """Build the agent-facing task view without leaking test answers.

    Training input/output pairs are retained. Test inputs are always retained,
    while test outputs are included only when ``include_test_outputs`` is true.
    """
    train = [
        {"input": pair["input"], "output": pair["output"]}
        for pair in task["train"]
        if "output" in pair
    ]
    tests = []
    for pair in task["test"]:
        test_pair = {"input": pair["input"]}
        if include_test_outputs and "output" in pair:
            test_pair["output"] = pair["output"]
        tests.append(test_pair)
    return {"train": train, "test": tests}


def _arc3_toolkit() -> tuple[Any, Any]:
    """Import the official ARC-AGI-3 toolkit only when requested."""
    try:
        from arc_agi import Arcade, OperationMode
    except ImportError as exc:
        raise ARCSourceError(
            "ARC-AGI-3 toolkit is not installed. Install with "
            "python -m pip install -e '.[arc]'."
        ) from exc
    return Arcade, OperationMode


def list_arc3_games(*, offline: bool = False) -> list[dict[str, Any]]:
    """List game IDs and titles from the official ARC-AGI-3 toolkit."""
    Arcade, OperationMode = _arc3_toolkit()
    mode = OperationMode.OFFLINE if offline else OperationMode.NORMAL
    arcade = Arcade(operation_mode=mode)
    games = arcade.get_environments()
    return [
        {
            "game_id": game.game_id,
            "title": getattr(game, "title", ""),
            "tags": list(getattr(game, "tags", []) or []),
        }
        for game in games
    ]


def open_arc3_game(
    game_id: str,
    *,
    offline: bool = False,
    seed: int = 0,
    save_recording: bool = True,
) -> tuple[Any, Any]:
    """Create an official environment and return it with its initial observation.

    Args:
        game_id: Official environment identifier.
        offline: Whether to request offline operation mode.
        seed: Initial environment seed.
        save_recording: Whether the toolkit should save a recording.

    Raises:
        ARCSourceError: If the toolkit is unavailable or environment startup fails.
    """
    Arcade, OperationMode = _arc3_toolkit()
    mode = OperationMode.OFFLINE if offline else OperationMode.NORMAL
    arcade = Arcade(operation_mode=mode)
    env = arcade.make(game_id, seed=seed, save_recording=save_recording)
    if env is None:
        raise ARCSourceError(f"ARC-AGI-3 could not open game {game_id!r}")
    observation = env.reset()
    if observation is None:
        raise ARCSourceError(f"ARC-AGI-3 game {game_id!r} returned no initial observation")
    # The environment wrapper owns the interaction; callers retain it for the session.
    return env, observation


def arc3_legal_actions(environment: Any) -> list[str]:
    """Return the action names currently exposed as legal by the environment.

    The available action set is state-dependent; callers should request it
    again after every observation rather than assume every action is legal.
    """
    actions = getattr(environment, "action_space", None)
    if actions is None:
        raise ARCSourceError("ARC-AGI-3 environment does not expose action_space")
    names = sorted(
        {
            str(getattr(action, "name", action))
            for action in actions
        }
    )
    return names


def step_arc3_game(
    environment: Any,
    action_name: str,
    *,
    data: dict[str, Any] | None = None,
    reasoning: dict[str, Any] | None = None,
) -> Any:
    """Apply one named official action and return the next observation.

    ``data`` and ``reasoning`` are passed through to the toolkit unchanged.
    This adapter validates the action name and, when exposed, the current
    legal-action set; it does not choose an action for Genesis.

    Raises:
        ARCSourceError: If the toolkit is unavailable or stepping fails to
            produce an observation.
        ValueError: If the action name is unknown or currently illegal.
    """
    try:
        from arcengine import GameAction
    except ImportError as exc:
        raise ARCSourceError(
            "The arcengine dependency is missing; install the optional arc extra."
        ) from exc
    action = getattr(GameAction, action_name, None)
    if action is None:
        valid = sorted(name for name in dir(GameAction) if name.startswith("ACTION"))
        raise ValueError(f"Unknown ARC action {action_name!r}; valid actions: {valid}")
    legal = arc3_legal_actions(environment)
    if legal and action_name not in legal:
        raise ValueError(
            f"ARC action {action_name!r} is not legal in the current state; "
            f"available actions: {legal}"
        )
    result = environment.step(action, data=data, reasoning=reasoning)
    if result is None:
        raise ARCSourceError("ARC-AGI-3 environment returned no observation after action")
    return result
