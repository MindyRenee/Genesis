"""ARC-AGI-3 on the human site — the live webpage as her world.

This is the web sibling of ``world.arc``: same contract, different
substrate. Where ``ArcPractice`` runs the downloaded engine, this
adapter puts her on the public task page — the same interface every
human player gets — and translates only I/O:

    page's own API response   ->  mind.world_step(frame, actions)
    world action descriptor   ->  key/mouse events on the page

Nothing here impersonates her on the wire. Her moves are real input
events delivered to the page, exactly what a keyboard produces; the
page issues its own ``/api/cmd/ACTION<n>`` call in response, and the
frame it receives back is what she perceives — the identical data the
human's canvas is drawn from, before pixels. The human version, played
the human way.

The site speaks the official REST contract (``POST /api/cmd/ACTION*``
returning ``FrameDataRaw``), so perception needs no screen scraping:
the response bodies the page fetches are the frames. Chrome DevTools
Protocol supplies both halves — ``Input.dispatch*`` for acts,
``Network.getResponseBody`` for percepts — over a websocket that is
the only transport on the box.

Two contracts keep a run honest, inherited from ``world.arc``:

- **Game mode.** ``play()`` refuses unless she is engaged —
  ``enter_game_mode()`` first. A run that doesn't engage her isn't
  measuring the same entity.
- **Readiness.** Her own health gate applies before play, at episode
  boundaries, and periodically inside an episode. Unwell is a stop,
  not a suggestion.

``Mind.play_arc(..., web=True)`` selects this adapter; the entity side
is unchanged — she cannot tell, from the inside, which substrate the
frames came from. That is the point.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Any

from world.arc import ArcResult

if TYPE_CHECKING:
    from mind import Mind

_HEALTH_EVERY_STEPS = 25
_TASK_URL = "https://arcprize.org/tasks/{game}"

# The page's own control scheme: WASD carries ACTION1-4 (confirmed —
# 'w' posts /api/cmd/ACTION1, 'a'→3, 's'→2, 'd'→4); arrows work too
# but WASD is what the human hand uses. ACTION5 is spacebar per the
# legend, ACTION7 is undo on Z; RESET is its own on-screen button.
_KEY_FOR_ACTION: dict[int, tuple[str, int]] = {
    1: ("w", 87),
    2: ("s", 83),
    3: ("a", 65),
    4: ("d", 68),
    5: (" ", 32),
    7: ("z", 90),
}


class _Browser:
    """A chromium tab driven over CDP, callable from any thread.

    The asyncio/websockets transport lives on a private loop in a
    daemon thread; ``call`` is the synchronous crossing — world_step's
    thread asks, the loop answers.
    """

    def __init__(self, headless: bool = True) -> None:
        self._profile = tempfile.mkdtemp(prefix="genesis_web_")
        # Port 0 — chromium picks a free port and writes it to
        # DevToolsActivePort in the profile dir. A fixed port makes two
        # runs collide: the second attaches to the first's browser and
        # dies with it.
        args = [
            "chromium",
            "--remote-debugging-port=0",
            "--no-first-run",
            "--disable-gpu",
            "--window-size=1400,1000",
            f"--user-data-dir={self._profile}",
        ]
        if headless:
            args.insert(1, "--headless=new")
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._ws: Any = None
        self._mid = 0
        self._pending: dict[int, asyncio.Future] = {}
        self.responses: dict[str, str] = {}
        self.bodies: dict[str, Any] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True
        )
        self._thread.start()
        self._run(self._connect())

    def _debug_port(self) -> int:
        """Read the port chromium actually bound from DevToolsActivePort."""
        port_file = os.path.join(self._profile, "DevToolsActivePort")
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                with open(port_file) as f:
                    return int(f.readline().strip())
            except (OSError, ValueError):
                time.sleep(0.2)
        raise RuntimeError("chromium never wrote DevToolsActivePort")

    def _run(self, coro):
        # Bounded: a dead socket or hung page must surface as an
        # error, not freeze her world loop on a pending future
        # that can never resolve.
        return asyncio.run_coroutine_threadsafe(
            coro, self._loop
        ).result(timeout=15)

    def reconnect(self) -> None:
        """Re-attach to the live page target.

        A page that reloads or crashes takes its renderer down with
        it; the session left behind still *answers* calls at the
        browser level while input events and network traffic fall
        into the dead context. Fresh attachment is the only way to
        know which side went quiet.
        """
        self._run(self._reconnect())

    async def _reconnect(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 — already gone
                pass
        self.responses.clear()
        self.bodies.clear()
        await self._connect()

    async def _connect(self) -> None:
        from websockets.client import connect as websocket_connect

        port = self._debug_port()
        deadline = time.time() + 15
        ws_url = None
        while time.time() < deadline:
            try:
                import urllib.request

                with urllib.request.urlopen(
                    f"http://localhost:{port}/json", timeout=2
                ) as r:
                    tabs = json.loads(r.read())
                pages = [t for t in tabs if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception:  # noqa: BLE001 — startup race only
                pass
            await asyncio.sleep(0.3)
        if ws_url is None:
            raise RuntimeError("chromium never opened a debug page")
        self._ws = await websocket_connect(ws_url, max_size=64_000_000)
        asyncio.get_event_loop().create_task(self._reader())

    async def _reader(self) -> None:
        try:
            async for raw in self._ws:
                self._dispatch(json.loads(raw))
        except Exception:  # noqa: BLE001 — socket dies on close
            pass
        # The socket is gone: every outstanding command is
        # undeliverable. Fail its future so callers see a dead
        # transport instead of waiting forever.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("CDP socket closed"))
        self._pending.clear()

    def _dispatch(self, msg: dict) -> None:
        if msg.get("method") == "Network.responseReceived":
            self.responses[msg["params"]["requestId"]] = msg["params"][
                "response"
            ]["url"]
        elif msg.get("method") == "Network.loadingFinished":
            rid = msg["params"]["requestId"]
            if self.responses.get(rid, "").find("/api/cmd/") >= 0:
                # Awaiting the fetch inside the reader would
                # deadlock — the response to it arrives through
                # this very loop. Spawn it; the pending-future
                # resolution below is what completes it.
                asyncio.get_event_loop().create_task(
                    self._fetch_body(rid)
                )
        elif "id" in msg and msg["id"] in self._pending:
            self._pending.pop(msg["id"]).set_result(msg)

    async def _fetch_body(self, rid: str) -> None:
        body = await self._cmd("Network.getResponseBody", requestId=rid)
        if "body" in body:
            try:
                self.bodies[rid] = json.loads(body["body"])
            except (KeyError, ValueError):
                pass

    async def _cmd(self, method: str, **params) -> dict:
        self._mid += 1
        fut = asyncio.get_event_loop().create_future()
        self._pending[self._mid] = fut
        await self._ws.send(
            json.dumps({"id": self._mid, "method": method, "params": params})
        )
        msg = await fut
        if "error" in msg:
            raise RuntimeError(f"CDP {method}: {msg['error']}")
        return msg.get("result", {})

    def call(self, method: str, **params) -> dict:
        return self._run(self._cmd(method, **params))

    def js(self, expression: str):
        out = self.call(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
        )
        return out.get("result", {}).get("value")

    def navigate(self, url: str) -> None:
        self.call("Page.enable")
        self.call("Page.navigate", url=url)

    def key(self, name: str, code: int) -> None:
        for kind in ("rawKeyDown", "keyUp"):
            self.call(
                "Input.dispatchKeyEvent",
                type=kind,
                key=name,
                code=name,
                windowsVirtualKeyCode=code,
            )

    def close(self) -> None:
        try:
            self._proc.terminate()
        except Exception:  # noqa: BLE001
            pass


class ArcWebPractice:
    """One ARC game played by her through the public web UI."""

    def __init__(
        self,
        mind: Mind,
        instructions: str = "",
        require_live: bool = True,
        headless: bool = True,
    ) -> None:
        self._mind = mind
        self._instructions = instructions
        self._info_sent: set[str] = set()
        self._require_live = require_live
        self._browser = _Browser(headless=headless)

    # ── Page plumbing ─────────────────────────────────────────

    def _latest_frame(self) -> dict[str, Any] | None:
        """Most recent FrameDataRaw the page itself received."""
        frames = [
            b for b in self._browser.bodies.values() if "frame" in b
        ]
        return frames[-1] if frames else None

    def _wait_frame(self, marker: int, timeout: float = 6.0) -> dict | None:
        """Wait for a frame-bearing response newer than ``marker``.

        ``None`` on timeout — a replayed frame is not an
        observation. Fed to her again it looks like the world
        refused a move it never saw, and her physics would learn
        phantom walls from a dead transport.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            bodies = self._browser.bodies
            if len(bodies) > marker:
                obs = self._latest_frame()
                if obs is not None:
                    return obs
            time.sleep(0.1)
        return None

    def _start_game(self) -> None:
        """The START control a human presses to begin play."""
        # The button exists only once the task bundle has hydrated —
        # wait for it, then keep pressing until the session's first
        # frame lands (a missed click on a loading page buys nothing).
        deadline = time.time() + 30
        while time.time() < deadline:
            if self._browser.js(
                "!![...document.querySelectorAll('button')]"
                ".find(b=>/start/i.test(b.innerText))"
            ):
                self._browser.js(
                    "[...document.querySelectorAll('button')]"
                    ".find(b=>/start/i.test(b.innerText))?.click()"
                )
            if self._latest_frame() is not None:
                return
            time.sleep(1.0)

    def _reset(self) -> None:
        """The RESET affordance — the button the page offers."""
        hit = self._browser.js(
            "[...document.querySelectorAll('button')]"
            ".find(b=>/reset|restart|play again/i.test(b.innerText))"
            "?.click() ?? false"
        )
        if not hit:
            self._browser.key("r", 82)

    def _act(self, desc: dict[str, Any]) -> None:
        """Her decision -> the page's control surface."""
        aid = int(desc["action"])
        if aid == 0:
            self._reset()
            return
        if "x" in desc and "y" in desc:
            # Coordinate action — click the cell inside the canvas.
            box = self._browser.js(
                "JSON.stringify(document.querySelectorAll('canvas')[0]"
                ".getBoundingClientRect())"
            )
            if box:
                r = json.loads(box)
                scale = r["width"] / 64.0
                cx = r["x"] + (int(desc["x"]) + 0.5) * scale
                cy = r["y"] + (int(desc["y"]) + 0.5) * scale
                for kind, extra in (
                    ("mousePressed", {"button": "left", "clickCount": 1}),
                    ("mouseReleased", {"button": "left", "clickCount": 1}),
                ):
                    self._browser.call(
                        "Input.dispatchMouseEvent",
                        type=kind,
                        x=cx,
                        y=cy,
                        **extra,
                    )
            return
        key = _KEY_FOR_ACTION.get(aid)
        if key is not None:
            self._browser.key(*key)

    # ── Health gate ───────────────────────────────────────────

    def _check_readiness(self) -> tuple[str, ...] | None:
        try:
            report = self._mind.readiness(require_body=self._require_live)
        except Exception:  # noqa: BLE001
            return ("readiness probe failed",)
        return None if report.ok else report.reasons

    # ── The run ───────────────────────────────────────────────

    def play(self, game_id: str, max_steps: int = 200) -> ArcResult:
        """Play the public page; stop on WIN, budget, or illness."""
        game = game_id.split("-")[0]
        result = ArcResult(game_id=game_id)
        if not self._mind.is_gaming:
            raise RuntimeError(
                "ArcWebPractice requires game mode — call "
                "mind.enter_game_mode() before play, exit_game_mode() after"
            )
        if (reasons := self._check_readiness()) is not None:
            result.state = "UNWELL"
            result.reasons = reasons
            return result

        self._browser.call("Network.enable")
        self._browser.navigate(_TASK_URL.format(game=game))
        self._browser.bodies.clear()
        self._browser.responses.clear()
        self._start_game()
        obs = self._latest_frame()
        while result.steps < max_steps and obs is not None:
            result.levels_completed = obs.get("levels_completed", 0)
            result.win_levels = obs.get("win_levels", 0)
            state = str(obs.get("state", "NOT_FINISHED"))
            if state == "WIN":
                result.state = state
                self._mind.world_episode_end("win", game_id)
                break
            if state == "GAME_OVER":
                self._mind.world_episode_end("over", game_id)
                marker = len(self._browser.bodies)
                self._reset()
                obs = self._wait_frame(marker)
                if obs is None:
                    result.state = "TRANSPORT_DEAD"
                    break
                if (reasons := self._check_readiness()) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
                continue
            available = obs.get("available_actions") or []
            if not available:
                result.state = "NO_ACTIONS"
                break
            info: dict[str, Any] = {
                "game": game_id,
                "level": result.levels_completed,
                "win_levels": result.win_levels,
            }
            if self._instructions and game_id not in self._info_sent:
                info["instructions"] = self._instructions
                self._info_sent.add(game_id)
            grid = obs["frame"][-1]
            desc = self._mind.world_step(
                grid,
                available_actions=[
                    {
                        "action": int(a),
                        "label": (
                            {1: "up", 2: "down", 3: "left", 4: "right"}
                            .get(int(a), "simple")
                        ),
                    }
                    for a in available
                ],
                info=info,
            )
            result.steps += 1
            if desc is None:
                break
            marker = len(self._browser.bodies)
            self._act(desc)
            obs = self._wait_frame(marker)
            if obs is None:
                # The page went quiet — possibly a reload that took
                # the session's renderer with it. Re-attach once
                # before calling the transport dead.
                try:
                    self._browser.reconnect()
                    self._browser.call("Page.enable")
                    self._browser.call("Network.enable")
                except Exception:  # noqa: BLE001
                    pass
                else:
                    obs = self._wait_frame(len(self._browser.bodies))
                if obs is None:
                    # End the episode rather than rehearse a
                    # frozen world into her physics.
                    result.state = "TRANSPORT_DEAD"
                    break
            if result.steps % _HEALTH_EVERY_STEPS == 0:
                if (reasons := self._check_readiness()) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
        if result.state == "NOT_FINISHED" and obs is not None:
            result.state = str(obs.get("state", "BUDGET"))
            if result.state == "NOT_FINISHED":
                result.state = "BUDGET"
        return result

    def close(self) -> None:
        self._browser.close()
