"""Loopback HTTP bridge exposing the official ARC-AGI-3 adapter to a phone client.

The bridge is transport only: it lists games, creates bounded sessions, exposes
current legal actions, and applies explicitly requested legal actions. It does
not choose actions or claim to solve games. Bind to loopback by default; do not
expose this unauthenticated development service to a network.
"""
from __future__ import annotations

import ipaddress
import json
import secrets
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit

from parietal_lobe.arc_sources import (
    ARCSourceError,
    arc3_legal_actions,
    list_arc3_games,
    open_arc3_game,
    step_arc3_game,
)


def _jsonable(value: Any, depth: int = 0) -> Any:
    """Convert common toolkit values into bounded JSON-compatible data."""
    if depth > 12:
        return {"truncated_type": type(value).__name__}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, depth + 1) for item in value]
    if hasattr(value, "name"):
        return str(value.name)
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist(), depth + 1)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item, depth + 1)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    return str(value)


def _state_name(observation: Any) -> str | None:
    state = (
        observation.get("state")
        if isinstance(observation, dict)
        else getattr(observation, "state", None)
    )
    if state is None:
        return None
    if isinstance(state, dict):
        state = state.get("name", state.get("value", state))
    return str(getattr(state, "name", state))


@dataclass
class _Session:
    game_id: str
    environment: Any
    observation: Any
    steps: int = 0
    closed: bool = False


class ArcPhoneBridge:
    """Thread-safe session registry over the existing ARC source adapter."""

    def __init__(
        self,
        *,
        game_lister: Callable[..., list[dict[str, Any]]] = list_arc3_games,
        game_opener: Callable[..., tuple[Any, Any]] = open_arc3_game,
        legal_actions: Callable[[Any], list[str]] = arc3_legal_actions,
        step_game: Callable[..., Any] = step_arc3_game,
        max_steps: int = 500,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self._game_lister = game_lister
        self._game_opener = game_opener
        self._legal_actions = legal_actions
        self._step_game = step_game
        self._max_steps = max_steps
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def health(self) -> dict[str, Any]:
        return {"ok": True, "service": "genesis-arc-phone-bridge", "protocol": 1}

    def games(self) -> dict[str, Any]:
        return {"games": _jsonable(self._game_lister())}

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        game_id = payload.get("game_id")
        if not isinstance(game_id, str) or not game_id.strip() or len(game_id) > 256:
            raise ValueError("game_id must be a non-empty string of at most 256 characters")
        seed = payload.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        offline = payload.get("offline", False)
        if not isinstance(offline, bool):
            raise ValueError("offline must be a boolean")
        env, observation = self._game_opener(
            game_id, offline=offline, seed=seed, save_recording=True
        )
        session_id = secrets.token_urlsafe(18)
        session = _Session(game_id=game_id, environment=env, observation=observation)
        with self._lock:
            self._sessions[session_id] = session
        return self._session_view(session_id, session)

    def step_session(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        if not isinstance(action, str) or not action:
            raise ValueError("action must be a non-empty string")
        data = payload.get("data")
        reasoning = payload.get("reasoning")
        if data is not None and not isinstance(data, dict):
            raise ValueError("data must be an object or null")
        if reasoning is not None and not isinstance(reasoning, dict):
            raise ValueError("reasoning must be an object or null")
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.closed:
                raise KeyError("session not found")
            if session.steps >= self._max_steps:
                raise OverflowError("session step budget exhausted")
            state = _state_name(session.observation)
            if state in {"WIN", "GAME_OVER"}:
                return self._session_view(session_id, session)
            legal = self._legal_actions(session.environment)
            if action not in legal:
                raise ValueError(f"illegal action {action!r}; legal actions: {legal}")
            next_observation = self._step_game(
                session.environment, action, data=data, reasoning=reasoning
            )
            session.observation = next_observation
            session.steps += 1
            return self._session_view(session_id, session)

    def close_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("session not found")
            session.closed = True
            del self._sessions[session_id]
        return {"closed": True, "session_id": session_id}

    def _session_view(self, session_id: str, session: _Session) -> dict[str, Any]:
        state = _state_name(session.observation)
        terminal = state in {"WIN", "GAME_OVER"}
        return {
            "session_id": session_id,
            "game_id": session.game_id,
            "step": session.steps,
            "observation": _jsonable(session.observation),
            "legal_actions": (
                [] if terminal or session.closed else self._legal_actions(session.environment)
            ),
            "terminal_state": state if terminal else None,
            "closed": session.closed,
            "step_budget_remaining": max(0, self._max_steps - session.steps),
        }

    def handler(self) -> type[BaseHTTPRequestHandler]:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "GenesisArcBridge/1"

            def log_message(self, fmt: str, *args: Any) -> None:
                # Avoid logging request bodies, observations, or credentials.
                return

            def _send(self, status: int, value: dict[str, Any]) -> None:
                body = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> dict[str, Any]:
                raw_length = self.headers.get("Content-Length", "0")
                try:
                    length = int(raw_length)
                except ValueError as exc:
                    raise ValueError("invalid Content-Length") from exc
                if length < 0 or length > 1_048_576:
                    raise ValueError("request body must be at most 1 MiB")
                raw = self.rfile.read(length)
                if not raw:
                    return {}
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("JSON request body must be an object")
                return value

            def do_GET(self) -> None:
                path = urlsplit(self.path).path
                try:
                    if path == "/health":
                        self._send(200, bridge.health())
                    elif path == "/games":
                        self._send(200, bridge.games())
                    else:
                        self._send(404, {"error": "not_found"})
                except RuntimeError as exc:
                    self._send(503, {"error": "arc_unavailable", "message": str(exc)})

            def do_POST(self) -> None:
                path = urlsplit(self.path).path
                try:
                    payload = self._body()
                    if path == "/sessions":
                        self._send(201, bridge.create_session(payload))
                        return
                    parts = path.strip("/").split("/")
                    if len(parts) == 3 and parts[0] == "sessions" and parts[2] == "step":
                        self._send(200, bridge.step_session(parts[1], payload))
                        return
                    self._send(404, {"error": "not_found"})
                except KeyError as exc:
                    self._send(404, {"error": "session_not_found", "message": str(exc)})
                except OverflowError as exc:
                    self._send(409, {"error": "step_budget_exhausted", "message": str(exc)})
                except ValueError as exc:
                    self._send(400, {"error": "invalid_request", "message": str(exc)})
                except RuntimeError as exc:
                    self._send(503, {"error": "arc_unavailable", "message": str(exc)})

            def do_DELETE(self) -> None:
                parts = urlsplit(self.path).path.strip("/").split("/")
                if len(parts) != 2 or parts[0] != "sessions":
                    self._send(404, {"error": "not_found"})
                    return
                try:
                    self._send(200, bridge.close_session(parts[1]))
                except KeyError as exc:
                    self._send(404, {"error": "session_not_found", "message": str(exc)})

        return Handler


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the unauthenticated bridge on loopback only."""
    if not ipaddress.ip_address(host).is_loopback:
        raise ValueError("the unauthenticated phone bridge must bind to loopback only")
    server = ThreadingHTTPServer((host, port), ArcPhoneBridge().handler())
    try:
        print(f"Genesis ARC phone bridge listening at http://{host}:{server.server_port}")
        server.serve_forever()
    finally:
        server.server_close()
