"""ARC-AGI-3 on the human site — the live webpage as her world.

This is the web sibling of ``worlds.arc``: same contract, different
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

Two contracts keep a run honest, inherited from ``worlds.arc``:

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
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from genesis_client.swallow import note_swallowed
from genesis_client.threads import named_thread

from .arc import ArcResult

if TYPE_CHECKING:
    from mind import Mind

_HEALTH_EVERY_STEPS = 25
_TASK_URL = "https://arcprize.org/tasks/{game}"

# The page's own control scheme: WASD carries ACTION1-4 (confirmed —
# 'w' posts /api/cmd/ACTION1, 'a'→3, 's'→2, 'd'→4); arrows work too
# but WASD is what the human hand uses. ACTION5 is spacebar per the
# legend, ACTION7 is undo on Z; RESET is its own on-screen button.
# This table is actuation only — which key posts which action id is
# wire protocol, the same information the page's legend gives a human.
# What an action *does* in a game is never told to her here either;
# direction semantics stay hers to learn from consequences.
_KEY_FOR_ACTION: dict[int, tuple[str, int]] = {
    1: ("w", 87),
    2: ("s", 83),
    3: ("a", 65),
    4: ("d", 68),
    5: (" ", 32),
    7: ("z", 90),
}

# The palette the venue renders — the 16 ARC colors as sRGB triplets
# (the game's own COLOR_MAP). World-side knowledge, like the keymap:
# which colors the canvas can paint. Cells are classified back to
# these in CIELAB, the perceptually uniform space.
_ARC_PALETTE_RGB = (
    (255, 255, 255),
    (204, 204, 204),
    (153, 153, 153),
    (102, 102, 102),
    (51, 51, 51),
    (0, 0, 0),
    (229, 58, 163),
    (255, 123, 204),
    (249, 60, 49),
    (30, 147, 255),
    (136, 216, 241),
    (255, 220, 0),
    (255, 133, 27),
    (146, 18, 49),
    (79, 204, 48),
    (163, 86, 214),
)

# JS expression evaluating to the board canvas element — the largest
# canvas on the task page. Embedded inside larger scripts; the element
# itself is never serialized over the wire.
_BOARD_JS = (
    "[...document.querySelectorAll('canvas')].filter(c=>{const r=c"
    ".getBoundingClientRect();return r.width>50&&r.height>50})"
    ".sort((a,b)=>b.getBoundingClientRect().width"
    "-a.getBoundingClientRect().width)[0]"
)

# Her internet browser, resolved per host: Google Chrome is the
# browser she is given; the open-source builds serve hosts without
# it. ``GENESIS_BROWSER`` pins an override (a name on PATH or an
# absolute path).
_BROWSER_BINS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")

# Where her browser keeps itself: a persistent profile under her
# data dir, one subdirectory per purpose. A browser that forgets
# every session is not one she learns from — cookies, site state,
# and history are her memory of the web, so the profile survives
# restarts. Purpose separation is a Chrome requirement, not a
# preference: separate instances must use separate user-data dirs,
# and an ARC run and a web lookup may both be alive at once. It is
# never the host's default Chrome profile — which is also what
# keeps --remote-debugging-port honoured on Chrome ≥136, where CDP
# on the default data dir is silently ignored.
_BROWSER_PURPOSES = ("web", "arcade")


def _browser_binary() -> str:
    """The chrome-family executable on this host, or a launch failure.

    A browser transport on a host with no browser cannot degrade —
    the raise is the honest report.
    """
    pin = os.getenv("GENESIS_BROWSER", "").strip()
    if pin:
        found = shutil.which(pin)
        if found is None:
            raise RuntimeError(f"GENESIS_BROWSER={pin!r}: no such executable")
        return found
    for name in _BROWSER_BINS:
        if found := shutil.which(name):
            return found
    raise RuntimeError("no chrome-family browser installed")


def _browser_profile(purpose: str) -> str:
    """The persistent profile dir for one of her browser's purposes.

    ``<data_dir>/browser/<purpose>`` — created on first use, kept
    across restarts, never the host Chrome profile. Purposes stay
    separate because Chrome itself requires it: two instances on
    one user-data dir don't coexist — the second forwards to the
    first and exits.
    """
    if purpose not in _BROWSER_PURPOSES:
        raise ValueError(f"unknown browser purpose {purpose!r}")
    from infrastructure.config import default_data_dir

    path = default_data_dir() / "browser" / purpose
    path.mkdir(parents=True, exist_ok=True)
    _clear_stale_singletons(path)
    return str(path)


def _pid_alive(pid: int) -> bool:
    """Whether *pid* names a live process on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return pid > 0


def _clear_stale_singletons(profile: os.PathLike[str] | str) -> None:
    """Drop Chrome's singleton markers when no live browser owns them.

    ``SingletonLock``/``SingletonCookie``/``SingletonSocket`` — plus
    the ``DevToolsActivePort`` hint — are process state, not profile
    data: they name the instance holding the profile and nothing
    more. An unclean mind death leaves them pointing at a dead pid,
    and a profile that traveled (spore, germinate, a copied data
    dir) can arrive with the symlinks flattened into plain files —
    which Chrome refuses to overwrite, aborting launch with a
    singleton-cookie failure. Chrome heals a stale *lock* itself but
    not the rest. Safe to remove only when nothing owns them, so the
    lock's ``pid-hostname`` target is checked first: a live owner
    keeps its markers and the launch adopts it instead.
    """
    import socket

    lock = os.path.join(profile, "SingletonLock")
    try:
        target = os.readlink(lock)  # "PID-HOSTNAME"
        pid_s, _, host = target.partition("-")
        if host == socket.gethostname() and _pid_alive(int(pid_s)):
            return  # a live browser owns this profile — adopt, don't touch
    except (OSError, ValueError):
        pass  # absent, unreadable, or flattened — none name a live owner
    for name in (
        "SingletonLock",
        "SingletonCookie",
        "SingletonSocket",
        "DevToolsActivePort",
    ):
        try:
            os.unlink(os.path.join(profile, name))
        except OSError:
            pass


class _Browser:
    """A browser tab driven over CDP, callable from any thread.

    The asyncio/websockets transport lives on a private loop in a
    daemon thread; ``call`` is the synchronous crossing — world_step's
    thread asks, the loop answers.
    """

    def __init__(self, headless: bool = True, purpose: str = "web") -> None:
        """Launch her browser and attach to its first page over CDP.

        A private asyncio loop runs the websocket transport on a
        daemon thread so the synchronous ``call``/``js`` surface can
        be used from ``world_step``'s thread. ``headless=False``
        shows the page on her own screen — the nested display
        ``arc_cabinet.ensure_arc_display`` keeps for her, never the
        user's desktop — for watching a run, not scoring one.

        ``purpose`` selects her persistent profile — ``"web"`` for
        the browsing her tools do, ``"arcade"`` for game sessions.
        Same browser, separate user-data dirs: history and site
        state accumulate in each, and two of them may be alive at
        once without Chrome's single-instance forwarding.
        """
        self._profile = _browser_profile(purpose)
        # Port 0 — the browser picks a free port and writes it to
        # DevToolsActivePort in the profile dir. A fixed port makes two
        # runs collide: the second attaches to the first's browser and
        # dies with it.
        env = None
        window_size = "1400,1000"
        if not headless:
            from worlds.arc_cabinet import _ARC_SCREEN, ensure_arc_display

            display = ensure_arc_display()
            if display is not None:
                env = {**os.environ, "DISPLAY": display}
                window_size = _ARC_SCREEN.replace("x", ",")
        args = [
            _browser_binary(),
            "--remote-debugging-port=0",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-gpu",
            "--mute-audio",
            # Containers cap /dev/shm at 64MB and Chrome crashes
            # without this; harmless on a full host.
            "--disable-dev-shm-usage",
            # A persistent profile that saves anything would else
            # block on the desktop keyring's unlock prompt — there is
            # no one to answer it on her screen.
            "--password-store=basic",
            "--use-mock-keychain",
            # Her profile never syncs outward to a Google account.
            "--disable-sync",
            f"--window-size={window_size}",
            f"--user-data-dir={self._profile}",
        ]
        if headless:
            args.insert(1, "--headless=new")
        # One pipe for both streams — Chrome writes "DevTools
        # listening on ws://..." to stdout, its diagnostics to
        # stderr; merging keeps the reader a single tail. text=True:
        # the port-line regex and the diagnostics are str-shaped.
        self._proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        self._stderr_lines: deque[str] = deque(maxlen=32)
        threading.Thread(target=self._read_stderr, name="arc-web-stderr", daemon=True).start()
        self._ws: Any = None
        self._mid = 0
        self._pending: dict[int, asyncio.Future] = {}
        self.responses: dict[str, str] = {}
        self.bodies: dict[str, Any] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = named_thread("arc-web-loop", self._loop.run_forever, comm="g-arc-web")
        self._thread.start()
        try:
            # Cold start on a populated profile is slow — the port
            # file plus the first target probe can outlast the 15 s
            # a transport call is allowed, so connect gets its own.
            self._run(self._connect(), timeout=40)
        except Exception:
            # A browser that never speaks CDP is not a transport —
            # kill it rather than leave it holding her profile's
            # single-instance lock.
            self.close()
            raise

    def _read_stderr(self) -> None:
        """Collect the browser's output — port line first, diagnostics always."""
        try:
            for line in self._proc.stdout or ():
                self._stderr_lines.append(line.rstrip("\n"))
        except Exception as e:  # noqa: BLE001 — pipe died with the browser
            note_swallowed("worlds.arc_web.stderr_drain", e)

    def _stderr_tail(self) -> str:
        """The browser's last stderr lines — for launch-failure messages."""
        return "; ".join(self._stderr_lines)[-400:] or "no stderr"

    def _debug_port(self) -> int:
        """The port her browser actually bound for CDP.

        Chrome reports it on stderr — ``DevTools listening on
        ws://127.0.0.1:PORT/...`` — the authoritative source for the
        process she launched. The profile's ``DevToolsActivePort``
        file is NOT trustworthy here: a stale file from a previous
        run can outlive its process, and it may name a port nobody
        owns. The file is only the adoption path — when her launch
        forwards to a browser already running on the profile and
        exits, that running browser's port is the one she attaches
        to.
        """
        import re

        deadline = time.time() + 20
        while time.time() < deadline:
            for line in self._stderr_lines:
                m = re.search(r"DevTools listening on ws://[^:]+:(\d+)", line)
                if m:
                    return int(m.group(1))
            if self._proc.poll() is not None:
                # The launch forwarded to her running browser (one
                # instance per profile) — adopt its port from the
                # file that browser wrote.
                port_file = os.path.join(self._profile, "DevToolsActivePort")
                try:
                    with open(port_file) as f:
                        return int(f.readline().strip())
                except (OSError, ValueError):
                    raise RuntimeError(
                        f"browser exited {self._proc.returncode} "
                        f"with no live port: {self._stderr_tail()}"
                    ) from None
            time.sleep(0.2)
        raise RuntimeError(f"browser never opened a debug port: {self._stderr_tail()}")

    def _run(self, coro, timeout: float = 15):
        """Submit a coroutine to the loop thread and wait, bounded.

        The synchronous crossing every public method uses: 15 s is
        the longest a transport call may take before surfacing as an
        error instead of freezing her world loop on a future that
        can never resolve. Launch/attach passes a wider bound — a
        cold profile start legitimately takes longer than a command.
        """
        # Bounded: a dead socket or hung page must surface as an
        # error, not freeze her world loop on a pending future
        # that can never resolve.
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

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
        """Loop-side half of ``reconnect`` — close, clear, re-attach.

        Cached response bodies die with the old session: a body id
        is only fetchable while its network context lives, so the
        fresh attachment starts with empty maps.
        """
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception as e:  # noqa: BLE001 — already gone
                note_swallowed("worlds.arc_web.ws_close", e)
        self.responses.clear()
        self.bodies.clear()
        await self._connect()

    async def _connect(self) -> None:
        """Attach the websocket to the browser's first page target.

        Polls ``/json`` until a page target appears — the browser binds
        the debug port before its first tab is listable, so the
        retry loop bridges the startup race. Spawns the reader task
        once connected.
        """
        import websockets

        websocket_connect = cast(Any, websockets).connect

        port = self._debug_port()
        deadline = time.time() + 15
        ws_url = None
        while time.time() < deadline:
            try:
                import urllib.request

                with urllib.request.urlopen(f"http://localhost:{port}/json", timeout=2) as r:
                    tabs = json.loads(r.read())
                pages = [t for t in tabs if t.get("type") == "page"]
                if pages:
                    ws_url = pages[0]["webSocketDebuggerUrl"]
                    break
            except Exception as e:  # noqa: BLE001 — startup race only
                note_swallowed("worlds.arc_web.ws_probe", e)
            await asyncio.sleep(0.3)
        if ws_url is None:
            raise RuntimeError("browser never opened a debug page")
        self._ws = await websocket_connect(ws_url, max_size=64_000_000)
        asyncio.get_event_loop().create_task(self._reader())

    async def _reader(self) -> None:
        """Consume the websocket: dispatch events, resolve futures.

        Runs until the socket dies — normal close or crash — then
        fails every outstanding command so a caller never waits on
        an undeliverable request.
        """
        try:
            async for raw in self._ws:
                self._dispatch(json.loads(raw))
        except Exception as e:  # noqa: BLE001 — socket dies on close
            note_swallowed("worlds.arc_web.reader", e)
        # The socket is gone: every outstanding command is
        # undeliverable. Fail its future so callers see a dead
        # transport instead of waiting forever.
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(ConnectionError("CDP socket closed"))
        self._pending.clear()

    def _dispatch(self, msg: dict) -> None:
        """Route one CDP message: network events or a command reply.

        ``responseReceived`` records request-id → url;
        ``loadingFinished`` on an ``/api/cmd/`` request spawns the
        body fetch; anything with a matching ``id`` resolves the
        pending command future.
        """
        if msg.get("method") == "Network.responseReceived":
            self.responses[msg["params"]["requestId"]] = msg["params"]["response"]["url"]
        elif msg.get("method") == "Network.loadingFinished":
            rid = msg["params"]["requestId"]
            if self.responses.get(rid, "").find("/api/cmd/") >= 0:
                # Awaiting the fetch inside the reader would
                # deadlock — the response to it arrives through
                # this very loop. Spawn it; the pending-future
                # resolution below is what completes it.
                asyncio.get_event_loop().create_task(self._fetch_body(rid))
        elif "id" in msg and msg["id"] in self._pending:
            self._pending.pop(msg["id"]).set_result(msg)

    async def _fetch_body(self, rid: str) -> None:
        """Pull one finished response's JSON body into ``bodies``.

        The fetch is itself a CDP command, so it cannot be awaited
        inside the reader that would resolve it — spawned as a task,
        it rides the same pending-future path as every other call.
        """
        body = await self._cmd("Network.getResponseBody", requestId=rid)
        if "body" in body:
            try:
                self.bodies[rid] = json.loads(body["body"])
            except (KeyError, ValueError) as e:
                # Non-JSON response body — nothing to record.
                note_swallowed("worlds.arc_web.body_json", e)

    async def _cmd(self, method: str, **params) -> dict:
        """Send one CDP command and await its reply.

        Each call gets a monotonically increasing id; the reader
        resolves the pending future when the matching response
        arrives. A CDP ``error`` field raises.
        """
        self._mid += 1
        fut = asyncio.get_event_loop().create_future()
        self._pending[self._mid] = fut
        await self._ws.send(json.dumps({"id": self._mid, "method": method, "params": params}))
        msg = await fut
        if "error" in msg:
            raise RuntimeError(f"CDP {method}: {msg['error']}")
        return msg.get("result", {})

    def call(self, method: str, **params) -> dict:
        """Synchronous CDP command — the thread-crossing entry point."""
        return self._run(self._cmd(method, **params))

    def js(self, expression: str):
        """Evaluate JavaScript in the page; return the value."""
        out = self.call(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
        )
        return out.get("result", {}).get("value")

    def navigate(self, url: str) -> None:
        """Navigate the page to ``url`` over the DevTools protocol."""
        self.call("Page.enable")
        self.call("Page.navigate", url=url)

    def key(self, name: str, code: int) -> None:
        """Send a keypress to the page — ``rawKeyDown`` then ``keyUp`` events."""
        for kind in ("rawKeyDown", "keyUp"):
            self.call(
                "Input.dispatchKeyEvent",
                type=kind,
                key=name,
                code=name,
                windowsVirtualKeyCode=code,
            )

    def move_mouse(self, x: float, y: float) -> None:
        """Move the pointer to page coordinates — a real ``mouseMoved``.

        Hover state is real to the page: a click that lands without
        a move is a teleport, and pointer-tracking elements (the
        board included) see the hand arrive.
        """
        self.call("Input.dispatchMouseEvent", type="mouseMoved", x=x, y=y)

    def click(self, x: float, y: float) -> None:
        """Click at page coordinates — move there, then press+release."""
        self.move_mouse(x, y)
        for kind in ("mousePressed", "mouseReleased"):
            self.call(
                "Input.dispatchMouseEvent",
                type=kind,
                x=x,
                y=y,
                button="left",
                clickCount=1,
            )

    def close(self) -> None:
        """Shut the browser down gracefully — the profile is memory.

        SIGTERM lets Chrome flush its profile (history, cookies,
        site state) before exit; SIGKILL only on a browser that
        ignores it. A dead process is already closed.
        """
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        except Exception as e:  # noqa: BLE001
            note_swallowed("worlds.arc_web.proc_terminate", e)


class ArcWebPractice:
    """One ARC game played by her through the public web UI."""

    def __init__(
        self,
        mind: Mind,
        instructions: str = "",
        require_live: bool = True,
        headless: bool = True,
    ) -> None:
        """Bind to her mind and launch the browser transport.

        ``require_live=False`` is the disembodied-debug path — never
        a measurement of the entity. ``headless=False`` shows the
        page for watching a run; scoring runs stay headless.
        """
        self._mind = mind
        self._instructions = instructions
        self._info_sent: set[str] = set()
        self._require_live = require_live
        self._palette_lab: Any = None
        self._browser = _Browser(headless=headless, purpose="arcade")

    # ── Page plumbing ─────────────────────────────────────────

    def _latest_frame(self) -> dict[str, Any] | None:
        """Most recent FrameDataRaw the page itself received."""
        frames = [b for b in self._browser.bodies.values() if "frame" in b]
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
            # Only bodies landed after the marker count are candidates —
            # the frame she needs is the reply to the move just sent,
            # which lands at an index past every body already seen.
            # A fresh body that bears no frame (an error reply, a
            # different /api/cmd response) is not an observation, and
            # an older frame-bearing body re-returned would be a
            # replay — neither satisfies the wait.
            new = [
                b
                for b in list(self._browser.bodies.values())[marker:]
                if isinstance(b, dict) and "frame" in b
            ]
            if new:
                return new[-1]
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
                "!![...document.querySelectorAll('button')].find(b=>/start/i.test(b.innerText))"
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

    @staticmethod
    def _action_label(action_id: int) -> str:
        """The same structural contract ``worlds.arc`` keeps.

        ``"coordinate"`` when the action takes a target cell — public
        interface structure, read from the action enum the page's own
        protocol implements — ``"simple"`` otherwise. Never a
        direction: what a control *does* is developer knowledge, and
        on this substrate as on the SDK it stays hers to learn.
        """
        try:
            from arcengine import ComplexAction, GameAction
        except ImportError:
            return "simple"
        try:
            action = GameAction.from_id(action_id)
        except (KeyError, ValueError):
            return "simple"
        return "coordinate" if action.action_type is ComplexAction else "simple"

    def _act(self, desc: dict[str, Any]) -> None:
        """Her decision -> the page's control surface."""
        aid = int(desc["action"])
        if aid == 0:
            self._reset()
            return
        if "x" in desc and "y" in desc:
            # Coordinate action — the pointer moves onto the board
            # and clicks the cell, the way the venue's canvas reads
            # a hand.
            x, y = desc["x"], desc["y"]
            if type(x) is not int or type(y) is not int or not (0 <= x < 64 and 0 <= y < 64):
                raise ValueError("Coordinate actions require integer x,y in [0, 63]")
            box = self._browser.js(
                "JSON.stringify((()=>{const b="
                + _BOARD_JS
                + ";if(!b)return null;const r=b.getBoundingClientRect()"
                + ";return{x:r.x,y:r.y,width:r.width,height:r.height}})())"
            )
            if box:
                r = json.loads(box)
                scale = r["width"] / 64.0
                self._browser.click(
                    r["x"] + (int(desc["x"]) + 0.5) * scale,
                    r["y"] + (int(desc["y"]) + 0.5) * scale,
                )
            return
        if self._action_label(aid) == "coordinate":
            raise ValueError("Coordinate actions require integer x,y")
        key = _KEY_FOR_ACTION.get(aid)
        if key is not None:
            self._browser.key(*key)

    def _display_grid(self) -> list[list[int]] | None:
        """The board as rendered — the human's evidence, decoded to cells.

        ``getImageData`` samples the canvas bitmap: the inner half of
        each of the 64×64 cells is averaged, then classified to the
        nearest ARC palette color in CIELAB (ΔE — perceptual distance,
        so antialiasing and compositing can't bend the answer).

        The API frame reports logical grid colors; this grid reports
        *displayed* colors — sprites the API never lists (selection
        rings, click targets remapped to display-only colors) exist
        here. ``None`` when the canvas can't be read — the caller
        degrades to the engine frame, the same way a lost sense
        degrades her other channels.
        """
        try:
            from occipital_lobe.color import nearest_palette, palette_lab_table
        except ImportError:
            return None
        raw = self._browser.js(
            "JSON.stringify((()=>{const b="
            + _BOARD_JS
            + ";if(!b)return null;const ctx=b.getContext('2d');"
            "const w=b.width,h=b.height,d=ctx.getImageData(0,0,w,h).data;"
            "const out=[];"
            "for(let r=0;r<64;r++){const row=[];"
            "for(let c=0;c<64;c++){"
            "const x0=Math.floor((c+0.25)*w/64),x1=Math.floor((c+0.75)*w/64);"
            "const y0=Math.floor((r+0.25)*h/64),y1=Math.floor((r+0.75)*h/64);"
            "let sr=0,sg=0,sb=0,n=0;"
            "for(let y=y0;y<y1;y++)for(let x=x0;x<x1;x++){"
            "const i=(y*w+x)*4;sr+=d[i];sg+=d[i+1];sb+=d[i+2];n++;}"
            "row.push(n?[sr/n,sg/n,sb/n]:[0,0,0]);}out.push(row);}"
            "return out})())"
        )
        if not raw:
            return None
        try:
            rgb = np.asarray(json.loads(raw), dtype=np.float64) / 255.0
        except (TypeError, ValueError):
            return None
        if rgb.shape != (64, 64, 3):
            return None
        if self._palette_lab is None:
            self._palette_lab = palette_lab_table(np.asarray(_ARC_PALETTE_RGB) / 255.0)
        grid = nearest_palette(rgb, self._palette_lab)
        return grid.astype(int).tolist()

    # ── Health gate ───────────────────────────────────────────

    def _check_readiness(self) -> tuple[str, ...] | None:
        """Refusal reasons when she isn't fit to play, else None.

        Same gate as the SDK adapter: a probe that raises reads as
        unwell rather than as healthy.
        """
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
        if obs is not None:
            # Starting the session is the initialization reset — the
            # same count ``worlds.arc`` keeps, so the substrates' costs
            # read identically.
            result.resets = 1
        prev_levels = (obs or {}).get("levels_completed", 0)
        level_start = 0
        while result.steps < max_steps and obs is not None:
            result.levels_completed = obs.get("levels_completed", 0)
            result.win_levels = obs.get("win_levels", 0)
            if result.levels_completed != prev_levels:
                # Steps each level cost — the within-game learning
                # curve, kept identically on both substrates.
                for _ in range(max(0, result.levels_completed - prev_levels)):
                    result.level_steps.append(result.steps - level_start)
                    level_start = result.steps
                prev_levels = result.levels_completed
                level_start = result.steps
            state = str(obs.get("state", "NOT_FINISHED"))
            if state == "WIN":
                result.state = state
                self._mind.world_episode_end("win", game_id)
                break
            if state == "GAME_OVER":
                self._mind.world_episode_end("over", game_id)
                if result.steps >= max_steps:
                    result.state = state
                    break
                # A death is a checkpoint — recovery stops the run
                # when she isn't well, and the restart is a submitted
                # action with a reset's cost either way.
                if (reasons := self._check_readiness()) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
                result.steps += 1
                result.resets += 1
                marker = len(self._browser.bodies)
                self._reset()
                obs = self._wait_frame(marker)
                if obs is None:
                    result.state = "TRANSPORT_DEAD"
                    break
                continue
            obs, terminal = self._web_turn(obs, result, game_id)
            if terminal is not None:
                result.state = terminal
                break
            if result.steps % _HEALTH_EVERY_STEPS == 0:
                if (reasons := self._check_readiness()) is not None:
                    result.state = "UNWELL"
                    result.reasons = reasons
                    break
        if result.state == "NOT_FINISHED":
            if obs is None:
                # The page never produced a frame — a dead transport,
                # not an unfinished run.
                result.state = "TRANSPORT_DEAD"
            else:
                result.state = str(obs.get("state", "BUDGET"))
                if result.state == "NOT_FINISHED":
                    result.state = "BUDGET"
        result.prior_learning = getattr(self._mind, "_world_learning_loaded", "none")
        return result

    def _web_turn(
        self, obs: dict[str, Any], result: ArcResult, game_id: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """One decision cycle over the page transport.

        Returns ``(next_obs, terminal_state)`` — ``terminal_state``
        is None when the world answered and the loop continues, else
        the state the result closes with: ``NO_ACTIONS``,
        ``NO_ACTION``, or ``TRANSPORT_DEAD`` (a silent page gets one
        reconnect before the transport is declared dead).
        """
        available = obs.get("available_actions") or []
        if not available:
            return None, "NO_ACTIONS"
        info: dict[str, Any] = {
            "game": game_id,
            "level": result.levels_completed,
            "win_levels": result.win_levels,
        }
        if self._instructions and game_id not in self._info_sent:
            info["instructions"] = self._instructions
            self._info_sent.add(game_id)
        strip = list(obs.get("frame") or [])
        # The percept is the rendered board when the canvas reads —
        # the API grid reports logical colors only, so interactive
        # sprites the page draws (click targets in display-only
        # colors) are invisible in it. The display grid is the human
        # evidence; the engine frame is the fallback, the way a lost
        # sense degrades her other channels.
        percept = self._display_grid()
        if percept is None:
            percept = strip[-1] if strip else obs["frame"]
        desc = self._mind.world_step(
            percept,
            through=strip[:-1],
            available_actions=[
                {"action": int(a), "label": self._action_label(int(a))} for a in available
            ],
            info=info,
        )
        if desc is None:
            return None, "NO_ACTION"
        aid = int(desc["action"])
        if aid != 0 and aid not in {int(a) for a in available}:
            raise ValueError("Chosen action is not available in this observation")
        result.steps += 1
        if aid == 0:
            result.resets += 1
        decision = getattr(self._mind, "game_decision", None) or {}
        source = str(decision.get("source", "unknown"))
        result.decision_sources[source] = result.decision_sources.get(source, 0) + 1
        marker = len(self._browser.bodies)
        self._act(desc)
        next_obs = self._wait_frame(marker)
        if next_obs is not None:
            return next_obs, None
        # The page went quiet — possibly a reload that took the
        # session's renderer with it. Re-attach once before calling
        # the transport dead.
        try:
            self._browser.reconnect()
            self._browser.call("Page.enable")
            self._browser.call("Network.enable")
        except Exception as e:  # noqa: BLE001
            note_swallowed("worlds.arc_web.reconnect", e)
        else:
            next_obs = self._wait_frame(len(self._browser.bodies))
        if next_obs is None:
            # End the episode rather than rehearse a frozen world
            # into her physics.
            return None, "TRANSPORT_DEAD"
        return next_obs, None

    def close(self) -> None:
        """Shut the browser down — the run's transport ends here."""
        self._browser.close()
