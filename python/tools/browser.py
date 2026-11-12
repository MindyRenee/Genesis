"""Browser — her tab in the outer web, driven over CDP.

Extracted from the retired worlds package: a headless browser is a
sensory-motor channel to the public web, not cognition. Tools ask;
the private asyncio loop on its daemon thread answers.
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
from typing import Any, cast

from genesis_client.swallow import note_swallowed
from genesis_client.threads import named_thread

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
# preference: separate instances must use separate user-data dirs.
# It is never the host's default Chrome profile — which is also what
# keeps --remote-debugging-port honoured on Chrome ≥136, where CDP
# on the default data dir is silently ignored.
_BROWSER_PURPOSES = ("web",)


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
    and a profile that traveled (a copied data dir, a transplant)
    can arrive with the symlinks flattened into plain files —
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
    daemon thread; ``call`` is the synchronous crossing — a tool's
    thread asks, the loop answers.
    """

    def __init__(self, headless: bool = True, purpose: str = "web") -> None:
        """Launch her browser and attach to its first page over CDP.

        A private asyncio loop runs the websocket transport on a
        daemon thread so the synchronous ``call``/``js`` surface can
        be used from a tool's thread. ``headless=False`` shows the
        page on the session's own ``DISPLAY`` — for watching her
        browse, not headless work.

        ``purpose`` selects her persistent profile — ``"web"`` for
        the browsing her tools do. The user-data dir persists, so
        history and site state accumulate between sessions.
        """
        self._profile = _browser_profile(purpose)
        # Port 0 — the browser picks a free port and writes it to
        # DevToolsActivePort in the profile dir. A fixed port makes two
        # runs collide: the second attaches to the first's browser and
        # dies with it.
        env = None
        window_size = "1400,1000"
        if not headless and os.environ.get("DISPLAY"):
            env = {**os.environ}
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
        threading.Thread(target=self._read_stderr, name="browser-stderr", daemon=True).start()
        self._ws: Any = None
        self._mid = 0
        self._pending: dict[int, asyncio.Future] = {}
        self.responses: dict[str, str] = {}
        self.bodies: dict[str, Any] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = named_thread("browser-loop", self._loop.run_forever, comm="g-browser")
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
            note_swallowed("tools.browser.stderr_drain", e)

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
                note_swallowed("tools.browser.ws_close", e)
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
                note_swallowed("tools.browser.ws_probe", e)
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
            note_swallowed("tools.browser.reader", e)
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
                note_swallowed("tools.browser.body_json", e)

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
            note_swallowed("tools.browser.proc_terminate", e)
