"""ARC web adapter tests — the public-page substrate boundary.

No browser, no network: ``world.arc_web._Browser`` is replaced with a
fake that speaks the same surface — ``bodies`` of FrameDataRaw-shaped
dicts, plus ``call``/``js``/``key``/``navigate``/``reconnect``/
``close`` — so the run loop is exercised exactly. What these tests
guard is parity with the SDK adapter's honest-run contract: no action
semantics are handed to her, every submitted move and reset is
counted, and an abstention reports as itself rather than as budget.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from world.arc_web import ArcWebPractice


def _frame(actions=(1, 2, 3, 5), state="NOT_FINISHED", levels=0, win=2):
    grid = [[(x + y) % 16 for x in range(8)] for y in range(8)]
    return {
        "frame": [grid],
        "available_actions": list(actions),
        "state": state,
        "levels_completed": levels,
        "win_levels": win,
    }


class _FakeBrowser:
    """A scripted page: each actuation publishes the next frame.

    One real input event answers with one frame, the way the page
    answers one key press with one ``/api/cmd`` response — the click
    the ``js`` shim performs is what a button does, not a back door.
    """

    def __init__(self, frames, headless: bool = True):
        self._frames = list(frames)
        self.bodies: dict[str, dict] = {}
        self.responses: dict[str, str] = {}
        self.calls: list[str] = []
        self.keys: list[str] = []
        self.js_calls: list[str] = []

    def _publish(self) -> None:
        if self._frames:
            self.bodies[f"r{len(self.bodies)}"] = self._frames.pop(0)

    def call(self, method, **params):
        self.calls.append(method)
        if (
            method == "Input.dispatchKeyEvent"
            and params.get("type") == "rawKeyDown"
        ) or (
            method == "Input.dispatchMouseEvent"
            and params.get("type") == "mousePressed"
        ):
            self._publish()
        return {}

    def js(self, expression):
        self.js_calls.append(expression)
        if "getBoundingClientRect" in expression:
            return json.dumps({"x": 0, "y": 0, "width": 640, "height": 640})
        if ".click()" in expression:
            # A real button press makes the page fetch a fresh frame.
            self._publish()
        return True

    def navigate(self, url):
        pass

    def key(self, name, code):
        self.keys.append(name)
        for kind in ("rawKeyDown", "keyUp"):
            self.call("Input.dispatchKeyEvent", type=kind, key=name, code=name)

    def reconnect(self):
        pass

    def close(self):
        pass


class _StubMind:
    """The boundary the web adapter consumes."""

    def __init__(self, *, gaming=True, ready=True, descriptors=None):
        self.is_gaming = gaming
        self._ready = ready
        self._descriptors = descriptors or [{"action": 1}]
        self.turns: list[dict] = []
        self.episodes: list[str] = []
        self.last_world_decision = {"action": "ACTION1", "source": "plan"}

    def readiness(self, *, require_body=True):
        return SimpleNamespace(
            ok=self._ready,
            reasons=("asleep",) if not self._ready else (),
        )

    def world_step(self, frame, available_actions=None, info=None):
        self.turns.append({"frame": frame, "actions": available_actions, "info": info})
        desc = self._descriptors[(len(self.turns) - 1) % len(self._descriptors)]
        if desc is None:
            return None
        offered = {item["action"] for item in available_actions}
        if desc["action"] in offered or desc["action"] == 0:
            return desc
        return {"action": min(offered)}

    def world_episode_end(self, outcome, game="the game"):
        self.episodes.append(outcome)

    @property
    def game_decision(self):
        return self.last_world_decision


def _practice(mind, frames) -> ArcWebPractice:
    """An ArcWebPractice on a fake browser — no chromium, no network."""
    practice = ArcWebPractice.__new__(ArcWebPractice)
    practice._mind = mind
    practice._instructions = ""
    practice._info_sent = set()
    practice._require_live = False
    practice._browser = _FakeBrowser(frames)  # type: ignore[assignment]
    return practice


def test_play_never_labels_direction_semantics():
    """The contract world.arc keeps: structure, never what a key does.

    Labels may mark interface structure ("simple"/"coordinate") but
    must never carry directional semantics — "up"/"down"/"left"/
    "right" is the developer's knowledge of the WASD convention, and
    in a game whose ACTION1 isn't up it is also wrong.
    """
    mind = _StubMind()
    practice = _practice(mind, [_frame()] + [_frame() for _ in range(4)])
    practice.play("fake-web", max_steps=3)
    offered = [a for turn in mind.turns for a in turn["actions"]]
    assert offered
    assert {a["label"] for a in offered} <= {"simple", "coordinate"}


def test_coordinate_actions_marked_by_structure_not_hand():
    """Coordinate-taking actions are labeled by the interface's own
    action enum, never by a hand-built direction map."""
    labels = {aid: ArcWebPractice._action_label(aid) for aid in range(8)}
    assert set(labels.values()) <= {"simple", "coordinate"}
    try:
        from arcengine import ComplexAction, GameAction
    except ImportError:
        return
    for action in GameAction:
        if action.action_type is ComplexAction:
            assert labels[int(action.value)] == "coordinate"
        else:
            assert labels[int(action.value)] == "simple"


def test_recovery_reset_counts_like_sdk_path():
    """Death recovery is a submitted action with a reset's cost."""
    frames = [
        _frame(),                         # session start (init reset)
        _frame(state="GAME_OVER"),        # her first move kills her
        _frame(),                         # world after the reset
        _frame(state="WIN", levels=1, win=1),
    ]
    mind = _StubMind()
    result = _practice(mind, frames).play("fake", max_steps=10)
    assert result.won and result.state == "WIN"
    assert result.steps == 3   # action + recovery reset + action
    assert result.resets == 2  # initialization + recovery
    assert mind.episodes == ["over", "win"]


def test_chosen_reset_counts_as_reset():
    """Her own RESET bid is both a step and a reset — like the SDK."""
    frames = [
        _frame(),
        _frame(),                          # after her reset
        _frame(),
        _frame(state="WIN", levels=1, win=1),
    ]
    mind = _StubMind(descriptors=[{"action": 0}, {"action": 1}, {"action": 1}])
    result = _practice(mind, frames).play("fake", max_steps=10)
    assert result.won
    assert result.resets == 2
    assert result.steps == 3


def test_abstention_reports_no_action_without_spending_step():
    """Declining to act is NO_ACTION — not a phantom step, not budget."""
    mind = _StubMind()
    mind.world_step = lambda *args, **kwargs: None
    result = _practice(mind, [_frame()]).play("fake", max_steps=3)
    assert result.state == "NO_ACTION"
    assert result.steps == 0
    assert result.resets == 1  # the world did initialize


def test_unavailable_action_is_rejected():
    """A choice outside the whitelist raises before it spends a step."""
    mind = _StubMind(descriptors=[{"action": 99}])
    mind.world_step = lambda *args, **kwargs: {"action": 99}
    practice = _practice(mind, [_frame()])
    with pytest.raises(ValueError, match="available"):
        practice.play("fake", max_steps=3)


def test_coordinate_action_without_coords_is_rejected():
    """A coordinate-taking action chosen without a target raises —
    the SDK's ``_to_step`` contract, kept on this substrate too."""
    practice = _practice(_StubMind(), [_frame()])
    if ArcWebPractice._action_label(6) != "coordinate":
        pytest.skip("no engine installed to classify action 6")
    with pytest.raises(ValueError, match="x,y"):
        practice._act({"action": 6})


def test_level_steps_record_per_level_costs():
    """Steps each level cost — the within-game learning curve."""
    frames = [
        _frame(levels=0),
        _frame(levels=1),
        _frame(levels=2),
        _frame(state="WIN", levels=3, win=3),
    ]
    result = _practice(_StubMind(), frames).play("fake", max_steps=10)
    assert result.won
    assert result.level_steps == [1, 1, 1]
    assert result.steps == 3


def test_death_recompletion_appends_again():
    """A level re-completed after a death appends a second cost."""
    frames = [
        _frame(levels=0),
        _frame(levels=1),                  # completes level one
        _frame(state="GAME_OVER", levels=1),  # dies on level two
        _frame(levels=0),                  # world restarts
        _frame(levels=1),                  # re-completes level one
        _frame(state="WIN", levels=2, win=2),
    ]
    result = _practice(_StubMind(), frames).play("fake", max_steps=10)
    assert result.won
    assert result.level_steps == [1, 1, 1]
    assert result.steps == 5   # 4 submitted actions + 1 recovery reset
    assert result.resets == 2


def test_no_frame_is_transport_dead_not_unfinished(monkeypatch):
    """A page that never produced a frame is a dead transport."""
    monkeypatch.setattr(ArcWebPractice, "_start_game", lambda self: None)
    result = _practice(_StubMind(), []).play("fake", max_steps=3)
    assert result.state == "TRANSPORT_DEAD"
    assert result.steps == 0 and result.resets == 0


def test_unwell_refuses_before_browser_work():
    """The health gate stops the run before the page is touched."""
    mind = _StubMind(ready=False)
    practice = _practice(mind, [_frame()])
    result = practice.play("fake", max_steps=3)
    assert result.state == "UNWELL"
    assert "asleep" in result.reasons
    browser = practice._browser
    assert not browser.calls and not browser.js_calls and not browser.keys


def test_play_refuses_outside_game_mode():
    practice = _practice(_StubMind(gaming=False), [_frame()])
    with pytest.raises(RuntimeError, match="game mode"):
        practice.play("fake", max_steps=3)


def test_dead_transport_after_move_is_not_budget(monkeypatch):
    """A page that goes quiet mid-run reports TRANSPORT_DEAD."""
    monkeypatch.setattr(
        ArcWebPractice, "_wait_frame", lambda self, marker, timeout=6.0: None
    )
    result = _practice(_StubMind(), [_frame()]).play("fake", max_steps=3)
    assert result.state == "TRANSPORT_DEAD"
    assert result.steps == 1


def test_decision_sources_and_prior_learning_reported():
    """The web substrate tallies the same evidence as the SDK path."""
    frames = [_frame(), _frame(state="WIN", levels=1, win=1)]
    mind = _StubMind()
    mind._world_learning_loaded = "loaded"
    result = _practice(mind, frames).play("fake", max_steps=5)
    assert result.decision_sources == {"plan": 1}
    assert result.prior_learning == "loaded"
