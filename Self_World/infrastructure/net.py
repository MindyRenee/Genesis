"""Net — bounded HTTP plumbing the body and the transports share.

The Discord ear (``body.hearing.discord_ear``) and the Discord poster
(``discord.feed``) both talk to the same API; the plumbing that keeps
those calls from wedging lives here so the ear never has to reach
into the outer world for it.
"""

from __future__ import annotations

import json
import queue
import time
import urllib.error
import urllib.request
from urllib.response import addinfourl

from genesis_client.threads import named_thread

from infrastructure.pulse import Pulse


def bounded_urlopen(
    req: urllib.request.Request,
    timeout: float,
    stop: Pulse | None = None,
) -> addinfourl:
    """urlopen with a wall-clock bound that also covers DNS.

    urllib's ``timeout`` bounds connect/read but NOT ``getaddrinfo`` —
    a blackholed resolver can hang the call indefinitely (observed at
    shutdown: ``discord-feed`` parked in ``getaddrinfo`` outlived the
    teardown join). The request runs on a daemon helper thread; after
    ``timeout`` the caller abandons it and the orphan finishes whenever
    DNS returns — it is a daemon, so it can never block process exit.

    ``stop`` cuts the wait short: ``box.get`` is polled in small
    slices, so a set event abandons the request immediately instead
    of parking the caller for the rest of ``timeout``.
    """
    box: queue.Queue[addinfourl | BaseException] = queue.Queue(maxsize=1)

    def _do() -> None:
        try:
            box.put(urllib.request.urlopen(req, timeout=timeout))
        except Exception as e:  # noqa: BLE001 — relayed to the caller
            box.put(e)

    named_thread("bounded-io", _do, comm="g-io").start()
    deadline = time.monotonic() + timeout
    while True:
        if stop is not None and stop.is_set():
            raise urllib.error.URLError("request cancelled — shutdown")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise urllib.error.URLError(
                f"request exceeded {timeout:.0f}s total bound (incl. DNS)"
            ) from None
        try:
            item = box.get(timeout=min(remaining, 0.05))
        except queue.Empty:
            continue
        break
    if isinstance(item, BaseException):
        raise item
    return item


def parse_retry_after(error: urllib.error.HTTPError) -> float | None:
    """Extract the server's retry_after seconds from a 429 response."""
    try:
        body = json.loads(error.read().decode("utf-8"))
        value = float(body.get("retry_after", 0))
        return value if value > 0 else None
    except (ValueError, OSError, AttributeError):
        return None
