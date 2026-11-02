"""Tests for the ARC phone bridge using a fake deterministic environment."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import ClassVar

import pytest
from parietal_lobe.arc_phone_bridge import ArcPhoneBridge, serve


class FakeEnvironment:
    """Environment stand-in for bridge tests."""
    action_space: ClassVar[list[str]] = ["ACTION1", "ACTION2"]


def test_session_exposes_legal_actions_and_advances_observation() -> None:
    """Check that session exposes legal actions and advances observation."""
    env = FakeEnvironment()
    bridge = ArcPhoneBridge(
        game_lister=lambda: [{"game_id": "fake", "title": "Fake"}],
        game_opener=lambda *args, **kwargs: (env, {"state": "PLAY", "frame": 0}),
        legal_actions=lambda environment: environment.action_space,
        step_game=lambda environment, action, **kwargs: {
            "state": "WIN" if action == "ACTION1" else "PLAY",
            "frame": 1,
        },
        max_steps=3,
    )
    created = bridge.create_session({"game_id": "fake"})
    assert created["legal_actions"] == ["ACTION1", "ACTION2"]
    assert created["step_budget_remaining"] == 3

    stepped = bridge.step_session(created["session_id"], {"action": "ACTION1"})
    assert stepped["observation"]["frame"] == 1
    assert stepped["terminal_state"] == "WIN"
    assert stepped["legal_actions"] == []
    assert stepped["step"] == 1


def test_rejects_illegal_action_without_advancing() -> None:
    """Check that rejects illegal action without advancing."""
    env = FakeEnvironment()
    bridge = ArcPhoneBridge(
        game_opener=lambda *args, **kwargs: (env, {"state": "PLAY"}),
        legal_actions=lambda environment: environment.action_space,
        step_game=lambda *args, **kwargs: pytest.fail("illegal action reached environment"),
    )
    created = bridge.create_session({"game_id": "fake"})
    with pytest.raises(ValueError, match="illegal action"):
        bridge.step_session(created["session_id"], {"action": "ACTION_BAD"})
    assert bridge._sessions[created["session_id"]].steps == 0


def test_validates_session_input_and_budget() -> None:
    """Check that validates session input and budget."""
    bridge = ArcPhoneBridge(
        game_opener=lambda *args, **kwargs: (FakeEnvironment(), {"state": "PLAY"}),
        legal_actions=lambda environment: environment.action_space,
        step_game=lambda *args, **kwargs: {"state": "PLAY"},
        max_steps=1,
    )
    with pytest.raises(ValueError):
        bridge.create_session({"game_id": ""})
    created = bridge.create_session({"game_id": "fake"})
    bridge.step_session(created["session_id"], {"action": "ACTION2"})
    with pytest.raises(OverflowError):
        bridge.step_session(created["session_id"], {"action": "ACTION2"})


def test_http_endpoints_and_error_mapping() -> None:
    """Check that http endpoints and error mapping."""
    bridge = ArcPhoneBridge(
        game_lister=lambda: [{"game_id": "fake"}],
        game_opener=lambda *args, **kwargs: (FakeEnvironment(), {"state": "PLAY"}),
        legal_actions=lambda environment: environment.action_space,
        step_game=lambda environment, action, **kwargs: {"state": "PLAY", "action": action},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), bridge.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/health") as response:
            assert json.load(response)["protocol"] == 1
        with urllib.request.urlopen(base + "/games") as response:
            assert json.load(response)["games"][0]["game_id"] == "fake"
        request = urllib.request.Request(
            base + "/sessions",
            data=json.dumps({"game_id": "fake"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            session = json.load(response)
            assert response.status == 201
        bad = urllib.request.Request(
            base + f"/sessions/{session['session_id']}/step",
            data=json.dumps({"action": "NOPE"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(bad)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_service_refuses_non_loopback_binding() -> None:
    """Check that service refuses non loopback binding."""
    with pytest.raises(ValueError, match="loopback"):
        serve("0.0.0.0", 8765)



def test_bridge_publishes_initial_action_and_result_events():
    """Check that bridge publishes initial action and result events."""
    events = []
    env = FakeEnvironment()
    bridge = ArcPhoneBridge(
        game_opener=lambda *args, **kwargs: (env, {"state": "PLAY", "frame": 0}),
        legal_actions=lambda environment: environment.action_space,
        step_game=lambda environment, action, **kwargs: {
            "state": "WIN", "frame": 1, "action": action,
        },
        observer=lambda event, payload: events.append((event, payload)),
    )
    created = bridge.create_session({"game_id": "fake"})
    assert bridge.health()["genesis_observer_connected"] is True
    assert events[0][0] == "observation"
    assert events[0][1]["phase"] == "initial"

    bridge.step_session(created["session_id"], {"action": "ACTION1"})
    assert [event for event, _ in events] == ["observation", "action", "observation"]
    assert events[1][1]["action"] == "ACTION1"
    assert events[2][1]["observation"]["state"] == "WIN"
