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


TERMINAL_WIN = "WIN"
GAME_OVER = "GAME_OVER"
RESET_ACTION = "RESET"


def observation_state(observation: Any) -> str | None:
    """Return the game-state name of a toolkit, mapping, or namespace observation."""
    state = (
        observation.get("state")
        if isinstance(observation, dict)
        else getattr(observation, "state", None)
    )
    if state is None:
        return None
    if isinstance(state, dict):
        state = state.get("name", state.get("value", state))
    name = getattr(state, "name", None)
    return str(name if name is not None else state)


def _int_grid(value: Any) -> list[list[int]] | None:
    """Return ``value`` as a rectangular list of int rows, or None."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        return None
    rows: list[list[int]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            return None
        if any(isinstance(cell, bool) or not isinstance(cell, int) for cell in row):
            return None
        rows.append([int(cell) for cell in row])
    if any(len(row) != len(rows[0]) for row in rows):
        return None
    return rows


def observation_frames(observation: Any) -> list[list[list[int]]]:
    """Return every visible frame layer of an observation as int grids.

    The official toolkit returns ``FrameDataRaw`` whose board lives in the
    private ``_frame`` attribute (exposed as the ``frame`` property) as a
    list of 2-D arrays; generic serializers that read public fields drop
    it. Mapping and namespace observations may supply ``frame`` or
    ``grid`` as a single 2-D grid or a list of grids.
    """
    raw: Any = None
    for key in ("frame", "grid"):
        if isinstance(observation, dict):
            raw = observation.get(key)
        else:
            raw = getattr(observation, key, None)
        if raw is not None:
            break
    if raw is None:
        return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    single = _int_grid(raw)
    if single is not None:
        return [single]
    if not isinstance(raw, (list, tuple)):
        return []
    frames = [_int_grid(layer) for layer in raw]
    return [layer for layer in frames if layer is not None]


def observation_grid(observation: Any) -> list[list[int]] | None:
    """Return the current board: the last frame layer, or None when absent."""
    frames = observation_frames(observation)
    return frames[-1] if frames else None


def _int_field(observation: Any, key: str) -> int | None:
    """Read a non-boolean integer field from a mapping or object, else None."""
    value = (
        observation.get(key)
        if isinstance(observation, dict)
        else getattr(observation, key, None)
    )
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def observation_view(observation: Any) -> dict[str, Any]:
    """Return the game-relevant, JSON-native content of an observation.

    Session identifiers, echoed action inputs, and timestamps are excluded
    so equal boards in equal game states compare equal.
    """
    available = (
        observation.get("available_actions")
        if isinstance(observation, dict)
        else getattr(observation, "available_actions", None)
    )
    return {
        "state": observation_state(observation),
        "grid": observation_grid(observation),
        "levels_completed": _int_field(observation, "levels_completed"),
        "win_levels": _int_field(observation, "win_levels"),
        "available_actions": (
            sorted(int(item) for item in available)
            if isinstance(available, (list, tuple))
            and all(isinstance(item, int) and not isinstance(item, bool) for item in available)
            else None
        ),
    }


def observation_percept(observation: Any) -> Any:
    """Expose an observation in the grid format Genesis' perception parses."""
    grid = observation_grid(observation)
    if grid is not None:
        return {"grid": grid}
    return observation if isinstance(observation, dict) else None


def is_complex_action(action_name: str) -> bool:
    """Return whether an action needs ``{"x", "y"}`` coordinates (ACTION6)."""
    try:
        from arcengine import GameAction
    except ImportError:
        return action_name == "ACTION6"
    action = getattr(GameAction, action_name, None)
    return bool(action is not None and action.is_complex())


def arc3_arcade(*, offline: bool = False) -> Any:
    """Create one toolkit ``Arcade`` to share across a run's games.

    One Arcade owns one API key and one default scorecard; reuse it for
    every game in a run and close it with :func:`close_arc3_scorecard`.
    """
    Arcade, OperationMode = _arc3_toolkit()
    mode = OperationMode.OFFLINE if offline else OperationMode.NORMAL
    return Arcade(operation_mode=mode)


def close_arc3_scorecard(arcade: Any) -> Any:
    """Close the Arcade's default scorecard and return the final scorecard."""
    return arcade.close_scorecard()


def list_arc3_games(
    *, offline: bool = False, arcade: Any | None = None
) -> list[dict[str, Any]]:
    """List game IDs and titles from the official ARC-AGI-3 toolkit."""
    active = arcade if arcade is not None else arc3_arcade(offline=offline)
    games = active.get_environments()
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
    arcade: Any | None = None,
) -> tuple[Any, Any]:
    """Create an official environment and return it with its initial observation.

    Args:
        game_id: Official environment identifier.
        offline: Whether to request offline operation mode.
        seed: Initial environment seed.
        save_recording: Whether the toolkit should save a recording.
        arcade: Shared Arcade from :func:`arc3_arcade`. Pass the same one
            for every game in a run so they share one scorecard; when
            omitted, a fresh Arcade (and scorecard) is created.

    Raises:
        ARCSourceError: If the toolkit is unavailable or environment startup fails.
    """
    active = arcade if arcade is not None else arc3_arcade(offline=offline)
    env = active.make(game_id, seed=seed, save_recording=save_recording)
    if env is None:
        raise ARCSourceError(f"ARC-AGI-3 could not open game {game_id!r}")
    observation = env.reset()
    if observation is None:
        raise ARCSourceError(f"ARC-AGI-3 game {game_id!r} returned no initial observation")
    return env, observation


def arc3_legal_actions(environment: Any) -> list[str]:
    """Return the action names currently exposed as legal by the environment.

    The available action set is state-dependent; callers should request it
    again after every observation rather than assume every action is legal.
    ``RESET`` is never listed here; use :func:`step_arc3_game` with
    ``"RESET"`` to recover from ``GAME_OVER``.
    """
    actions = getattr(environment, "action_space", None)
    if actions is None:
        raise ARCSourceError("ARC-AGI-3 environment does not expose action_space")
    return sorted({str(getattr(action, "name", action)) for action in actions})


def _validate_click(action_name: str, data: dict[str, Any] | None) -> None:
    """Require in-range integer x/y coordinates for complex actions."""
    if not is_complex_action(action_name):
        return
    if not isinstance(data, dict):
        raise ValueError(f"{action_name} requires data with integer x and y in 0..63")
    for axis in ("x", "y"):
        value = data.get(axis)
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 63:
            raise ValueError(f"{action_name} requires data with integer x and y in 0..63")


def step_arc3_game(
    environment: Any,
    action_name: str,
    *,
    data: dict[str, Any] | None = None,
    reasoning: dict[str, Any] | None = None,
) -> Any:
    """Apply one named official action and return the next observation.

    ``RESET`` is always accepted (it is how a ``GAME_OVER`` is recovered).
    Other actions must be in the current legal set, and complex actions
    (ACTION6) must carry explicit ``x``/``y`` coordinates; the toolkit
    would otherwise silently click (0, 0).

    Raises:
        ARCSourceError: If the toolkit is unavailable or stepping fails to
            produce an observation.
        ValueError: If the action name is unknown, currently illegal, or
            missing required coordinates.
    """
    try:
        from arcengine import GameAction
    except ImportError as exc:
        raise ARCSourceError(
            "The arcengine dependency is missing; install the optional arc extra."
        ) from exc
    action = getattr(GameAction, action_name, None)
    if action is None:
        valid = sorted(
            name for name in dir(GameAction) if name.startswith("ACTION") or name == "RESET"
        )
        raise ValueError(f"Unknown ARC action {action_name!r}; valid actions: {valid}")
    if action_name != RESET_ACTION:
        legal = arc3_legal_actions(environment)
        if legal and action_name not in legal:
            raise ValueError(
                f"ARC action {action_name!r} is not legal in the current state; "
                f"available actions: {legal}"
            )
        _validate_click(action_name, data)
    result = environment.step(action, data=data, reasoning=reasoning)
    if result is None:
        raise ARCSourceError("ARC-AGI-3 environment returned no observation after action")
    return result
