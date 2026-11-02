"""Tests against real ``arcengine.FrameDataRaw`` observations.

Earlier tests used fake ``.grid`` namespaces, which hid that the official
toolkit stores the board in the private ``frame`` field.
"""

from __future__ import annotations

import json

import pytest

arcengine = pytest.importorskip("arcengine")
np = pytest.importorskip("numpy")

from parietal_lobe.arc_ascii import encode_game_record  # noqa: E402
from parietal_lobe.arc_phone_bridge import ArcPhoneBridge, _jsonable  # noqa: E402
from parietal_lobe.arc_policy import (  # noqa: E402
    ARC3ExplorationPolicy,
    click_targets,
    observation_signature,
)
from parietal_lobe.arc_sources import (  # noqa: E402
    is_complex_action,
    observation_grid,
    observation_percept,
    observation_state,
    step_arc3_game,
)


def _frame(board, *, guid="a", state=None, levels=0):
    """Build a real FrameDataRaw with a 64x64 board."""
    raw = arcengine.FrameDataRaw(
        game_id="g", guid=guid, state=state or arcengine.GameState.NOT_FINISHED,
        levels_completed=levels, win_levels=3, available_actions=[1, 6],
    )
    raw.frame = [np.array(board, dtype=np.int8)]
    return raw


def _board(fill=0, mark=None):
    """Return a 64x64 board, optionally with one marked cell."""
    grid = [[fill] * 64 for _ in range(64)]
    if mark:
        r, c, v = mark
        grid[r][c] = v
    return grid


def test_board_is_read_from_private_frame() -> None:
    """Extract the board, state, and percept from FrameDataRaw."""
    obs = _frame(_board(mark=(3, 4, 9)))
    assert observation_grid(obs)[3][4] == 9
    assert observation_state(obs) == "NOT_FINISHED"
    assert observation_percept(obs)["grid"][3][4] == 9


def test_signature_tracks_board_not_session_metadata() -> None:
    """Different boards differ; identical boards with new guids match."""
    a = _frame(_board(mark=(1, 1, 2)), guid="x")
    b = _frame(_board(mark=(1, 1, 2)), guid="y")
    c = _frame(_board(mark=(1, 2, 2)), guid="x")
    assert observation_signature(a) == observation_signature(b)
    assert observation_signature(a) != observation_signature(c)


def test_serializers_include_board() -> None:
    """Bridge JSON and ASCII records both carry the frame."""
    obs = _frame(_board(mark=(0, 0, 5)))
    assert _jsonable(obs)["frame"][0][0][0] == 5
    record = json.loads(bytes(encode_game_record({"o": obs}).codes))
    assert record["o"]["frame"][0][0][0] == 5


def test_action6_requires_coordinates() -> None:
    """Reject ACTION6 without in-range x/y instead of clicking (0, 0)."""
    calls = []

    class Env:
        """Fake env exposing ACTION6."""

        def __init__(self):
            self.action_space = [arcengine.GameAction.ACTION6]

        def step(self, action, data=None, reasoning=None):
            """Record the step."""
            calls.append(data)
            return _frame(_board())

    assert is_complex_action("ACTION6") and not is_complex_action("ACTION1")
    for bad in (None, {}, {"x": 64, "y": 0}, {"x": True, "y": 1}):
        with pytest.raises(ValueError):
            step_arc3_game(Env(), "ACTION6", data=bad)
    step_arc3_game(Env(), "ACTION6", data={"x": 3, "y": 4})
    step_arc3_game(Env(), "RESET")
    assert calls == [{"x": 3, "y": 4}, None]


def test_click_targets_hit_objects() -> None:
    """Click candidates land on non-background objects, smallest first."""
    grid = _board()
    grid[10][20] = 3
    for r in range(30, 34):
        for c in range(30, 34):
            grid[r][c] = 4
    targets = click_targets(grid)
    assert targets[0] == (20, 10)
    assert grid[targets[1][1]][targets[1][0]] == 4


def test_policy_reads_real_frames() -> None:
    """The policy produces ACTION6 data on a real observation."""
    obs = _frame(_board(mark=(7, 8, 1)))
    decision = ARC3ExplorationPolicy().choose_action(obs, ["ACTION6"], [])
    assert decision.action == "ACTION6"
    assert decision.data == {"x": 8, "y": 7}


def test_bridge_game_over_offers_only_reset() -> None:
    """After GAME_OVER the bridge lists RESET and rejects other actions."""
    over = _frame(_board(), state=arcengine.GameState.GAME_OVER)
    fresh = _frame(_board())
    steps = []

    def step(env, action, data=None, reasoning=None):
        steps.append(action)
        return fresh

    bridge = ArcPhoneBridge(
        game_lister=lambda **_: [],
        game_opener=lambda game_id, **_: (object(), over),
        legal_actions=lambda env: ["ACTION1"],
        step_game=step,
    )
    view = bridge.create_session({"game_id": "g"})
    assert view["legal_actions"] == ["RESET"]
    assert view["state"] == "GAME_OVER"
    assert len(view["board"]) == 64
    with pytest.raises(ValueError):
        bridge.step_session(view["session_id"], {"action": "ACTION1"})
    after = bridge.step_session(view["session_id"], {"action": "RESET"})
    assert steps == ["RESET"]
    assert after["legal_actions"] == ["ACTION1"]
    assert after["levels_completed"] == 0
