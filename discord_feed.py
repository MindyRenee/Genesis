"""Discord feed — the webhook transport behind Genesis's channels.

Three channels exist, each on its own webhook:

- ``GENESIS_DISCORD_WEBHOOK`` — *her* channel. Her social media: the
  only messages posted here are the ones she chooses to broadcast,
  composed from her own cognition when the ``broadcast`` urge fires
  (see ``basal_ganglia.volition``). Nothing mirrors to this channel —
  no session log, no telemetry, no user input.
- ``GENESIS_DISCORD_ARC_WEBHOOK`` — an opt-in spectator feed the ARC
  showcase narrates practice runs to.
- ``GENESIS_DISCORD_PULSE_WEBHOOK`` — an opt-in ops telemetry feed
  driven by ``scripts/discord_pulse.py``.

A webhook can only *post* — it cannot read the channel — so every
feed is read-only by construction. Channel permissions decide who
watches.

Design notes:

- ``DiscordFeed`` is a thread-safe bounded queue plus a single daemon
  poster thread. Producers enqueue; the poster coalesces bursts into
  single messages (Discord caps content at 2000 chars), paces
  requests, and honors 429 ``retry_after``. Posting never blocks or
  raises into Genesis — a dead webhook costs a queue, nothing more.
- ``allowed_mentions`` is empty — posted text can never ping anyone.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Iterator

from genesis_client.swallow import note_swallowed

logger = logging.getLogger(__name__)

_WEBHOOK_ENV = "GENESIS_DISCORD_WEBHOOK"

# Discord's content limit is 2000 chars; leave margin so coalesced
# batches and markdown never push a message over.
_MAX_CONTENT = 1900
# Seconds between webhook POSTs. Webhook routes are limited to a
# handful of requests per few seconds; pacing + coalescing keeps a
# talkative session under the cap without dropping lines.
_POST_INTERVAL = 1.0
# Bounded queue: if Discord is unreachable for a long stretch the
# oldest lines are dropped rather than letting memory grow.
_QUEUE_MAX = 512
# How long close() keeps flushing buffered lines before giving up.
_DRAIN_DEADLINE = 5.0
# Backoff ceiling (seconds) for transient network/5xx failures.
_BACKOFF_CAP = 60.0
# Consecutive transient failures after which the feed disables itself.
# A dead webhook must not cost the session a churning poster thread —
# without a cap, the queue can never drain.
_FAILURE_DISABLE = 20
# Seconds the poster waits after the first queued line for more of the
# same author to arrive, so a burst coalesces into one message.
_COALESCE_WINDOW = 0.15

def _chunks(text: str) -> Iterator[str]:
    """Yield ``text`` in pieces that fit Discord's content limit.

    Splits on newlines when possible so coalesced batches break at
    line boundaries; hard-splits single lines that exceed the limit.
    """
    while len(text) > _MAX_CONTENT:
        cut = text.rfind("\n", 0, _MAX_CONTENT + 1)
        if cut <= 0:
            cut = _MAX_CONTENT
        yield text[:cut]
        text = text[cut:].lstrip("\n")
    if text:
        yield text


def _multipart(
    payload: dict, filename: str, data: bytes, field: str = "files[0]"
) -> tuple[bytes, str]:
    """Encode a webhook multipart/form-data body (payload_json + one file)."""
    boundary = f"----genesis{int(time.time() * 1e6):x}"
    parts: list[bytes] = []
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="payload_json"\r\n'
        f"Content-Type: application/json\r\n\r\n"
        f"{json.dumps(payload)}\r\n".encode()
    )
    parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; '
        f'filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n".encode()
        + data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class DiscordFeed:
    """Background webhook poster. All public methods are thread-safe.

    Args:
        webhook_url: Discord channel webhook URL.
        post_interval: Minimum seconds between POSTs.
        queue_max: Maximum buffered lines before oldest are dropped.
    """

    def __init__(
        self,
        webhook_url: str,
        *,
        post_interval: float = _POST_INTERVAL,
        queue_max: int = _QUEUE_MAX,
    ) -> None:
        self._url = webhook_url
        self._post_interval = post_interval
        self._queue_max = queue_max
        self._queue: deque[tuple] = deque()
        self._cond = threading.Condition()
        self._closing = False
        self._disabled = False
        self._fail_streak = 0
        # A webhook may edit its own messages; the live-board view is
        # one message kept current rather than a scroll of frames.
        self._live_id: str | None = None
        self._thread = threading.Thread(
            target=self._run, name="discord-feed", daemon=True
        )
        self._thread.start()

    @classmethod
    def from_env(cls) -> DiscordFeed | None:
        """Create her public channel feed from ``GENESIS_DISCORD_WEBHOOK``.

        Returns None when the webhook is unset — her channel is simply
        silent then; nothing she says is mirrored anywhere.
        """
        url = os.environ.get(_WEBHOOK_ENV, "").strip()
        if not url:
            return None
        feed = cls(url)
        logger.info(
            f"[discord] Public channel enabled ({_WEBHOOK_ENV}) — "
            "posts are composed by her volition only."
        )
        return feed

    # ── Producer API ─────────────────────────────────────────────────

    def post(self, text: str, author: str | None = None) -> None:
        """Enqueue one message. ``author`` overrides the webhook username."""
        if not text.strip():
            return
        with self._cond:
            if self._disabled:
                return
            while len(self._queue) >= self._queue_max:
                self._queue.popleft()
            self._queue.append(("text", author, text))
            self._cond.notify()

    def post_image(
        self,
        png: bytes,
        caption: str = "",
        author: str | None = None,
        filename: str = "frame.png",
    ) -> None:
        """Enqueue one image attachment."""
        if not png:
            return
        with self._cond:
            if self._disabled:
                return
            while len(self._queue) >= self._queue_max:
                self._queue.popleft()
            self._queue.append(("image", author, caption, filename, png))
            self._cond.notify()

    def update_live(
        self,
        png: bytes,
        caption: str = "",
        author: str | None = None,
    ) -> None:
        """Queue an update to the single live-board message.

        Webhooks can edit their own messages, so the board view is one
        message kept current rather than a scroll of frames. Pending
        live updates collapse — only the newest frame is ever sent.
        """
        if not png:
            return
        with self._cond:
            if self._disabled:
                return
            self._queue = deque(
                item for item in self._queue if item[0] != "live"
            )
            self._queue.append(("live", author, caption, png))
            self._cond.notify()

    # ── Poster thread ────────────────────────────────────────────────

    def _run(self) -> None:
        backoff = 0.0
        drain_deadline: float | None = None
        while True:
            if self._closing:
                if drain_deadline is None:
                    drain_deadline = time.monotonic() + _DRAIN_DEADLINE
                elif time.monotonic() > drain_deadline:
                    return
            batch = self._take_batch()
            if batch is None:
                return
            kind = batch[0]
            if backoff:
                wait = backoff
                if drain_deadline is not None:
                    wait = min(wait, max(0.0, drain_deadline - time.monotonic()))
                time.sleep(wait)
            if kind == "text":
                _, author, content = batch
                ok, retry_after, permanent = self._post(content, author)
            elif kind == "image":
                _, author, caption, filename, png = batch
                ok, retry_after, permanent = self._post_image_once(
                    caption, author, filename, png
                )
            else:  # "live"
                _, author, caption, png = batch
                ok, retry_after, permanent = self._post_live(
                    caption, author, png
                )
            if ok:
                backoff = 0.0
                self._fail_streak = 0
            elif permanent:
                with self._cond:
                    self._disabled = True
                logger.info(
                    "[discord] Webhook rejected a post — feed disabled "
                    "for this session."
                )
                return
            else:
                self._fail_streak += 1
                if self._fail_streak >= _FAILURE_DISABLE:
                    with self._cond:
                        self._disabled = True
                    logger.info(
                        "[discord] Webhook unreachable after "
                        f"{_FAILURE_DISABLE} attempts — feed disabled "
                        "for this session."
                    )
                    return
                backoff = (
                    retry_after
                    if retry_after is not None
                    else min(_BACKOFF_CAP, max(2.0, backoff * 2))
                )
                # Re-queue the failed batch so it isn't lost.
                with self._cond:
                    self._queue.appendleft(batch)
            time.sleep(0.0 if self._closing else self._post_interval)

    def _take_batch(self) -> tuple | None:
        """Pop one item, coalescing same-author text up to the limit."""
        with self._cond:
            while not self._queue:
                if self._closing:
                    return None
                self._cond.wait(timeout=1.0)
            item = self._queue.popleft()
            if item[0] != "text":
                return item
            _, author, text = item
            parts = [text]
            total = len(text)
            window_end = time.monotonic() + _COALESCE_WINDOW
            while total < _MAX_CONTENT:
                if not self._queue:
                    if self._closing:
                        break
                    remaining = window_end - time.monotonic()
                    if remaining <= 0:
                        break
                    self._cond.wait(timeout=remaining)
                    continue
                nxt = self._queue[0]
                if nxt[0] != "text":
                    break
                next_author, next_text = nxt[1], nxt[2]
                if next_author != author or total + 1 + len(next_text) > _MAX_CONTENT:
                    break
                self._queue.popleft()
                parts.append(next_text)
                total += 1 + len(next_text)
            return ("text", author, "\n".join(parts))

    def _post(
        self, content: str, author: str | None
    ) -> tuple[bool, float | None, bool]:
        """POST content (chunked). Returns (ok, retry_after, permanent)."""
        for chunk in _chunks(content):
            ok, retry_after, permanent = self._post_once(chunk, author)
            if not ok:
                return ok, retry_after, permanent
        return True, None, False

    def _post_once(
        self, content: str, author: str | None
    ) -> tuple[bool, float | None, bool]:
        payload: dict[str, object] = {
            "content": content,
            "allowed_mentions": {"parse": []},
        }
        if author:
            payload["username"] = author
        req = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                # Discord's edge (Cloudflare 1010) rejects the default
                # Python-urllib user agent as a bot signature.
                "User-Agent": "GenesisSessionMirror/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                return True, None, False
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = _parse_retry_after(e)
                logger.debug(f"[discord] rate-limited, retry_after={retry_after}")
                return False, retry_after, False
            permanent = 400 <= e.code < 500
            logger.debug(f"[discord] webhook POST failed: HTTP {e.code}")
            return False, None, permanent
        except (urllib.error.URLError, OSError) as e:
            logger.debug(f"[discord] webhook POST error: {e}")
            return False, None, False

    def _post_image_once(
        self,
        caption: str,
        author: str | None,
        filename: str,
        png: bytes,
    ) -> tuple[bool, float | None, bool]:
        payload: dict[str, object] = {
            "content": caption,
            "allowed_mentions": {"parse": []},
            "attachments": [
                {"id": 0, "filename": filename, "description": caption[:1024]}
            ],
        }
        if author:
            payload["username"] = author
        body, content_type = _multipart(payload, filename, png)
        return self._send(self._url, "POST", body, content_type)

    def _post_live(
        self, caption: str, author: str | None, png: bytes
    ) -> tuple[bool, float | None, bool]:
        """PATCH the live message if it exists, else create it (wait=true)."""
        payload: dict[str, object] = {
            "content": caption,
            "allowed_mentions": {"parse": []},
            "attachments": [{"id": 0, "filename": "board.png"}],
        }
        if self._live_id:
            url = f"{self._url}/messages/{self._live_id}"
            body, content_type = _multipart(payload, "board.png", png)
            ok, retry_after, permanent = self._send(url, "PATCH", body, content_type)
            if ok or retry_after is not None or not permanent:
                return ok, retry_after, permanent
            # Permanent failure — the message may have been deleted.
            # Fall through and create a fresh live message.
            self._live_id = None
        if author:
            payload["username"] = author
        body, content_type = _multipart(payload, "board.png", png)
        url = self._url + ("&" if "?" in self._url else "?") + "wait=true"
        status, retry_after, permanent, resp = self._send(
            url, "POST", body, content_type, want_body=True
        )
        if status and resp:
            try:
                self._live_id = json.loads(resp).get("id")
            except ValueError as e:
                note_swallowed("discord_feed.live_status_parse", e)
        return status, retry_after, permanent

    def _send(
        self,
        url: str,
        method: str,
        body: bytes,
        content_type: str,
        want_body: bool = False,
    ) -> tuple:
        """Raw request. Returns (ok, retry_after, permanent[, body])."""
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": content_type,
                "User-Agent": "GenesisSessionMirror/1.0",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                resp = r.read() if want_body else None
            return (True, None, False, resp) if want_body else (True, None, False)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = _parse_retry_after(e)
                logger.debug(f"[discord] rate-limited, retry_after={retry_after}")
                out = (False, retry_after, False)
            else:
                permanent = 400 <= e.code < 500
                logger.debug(f"[discord] webhook {method} failed: HTTP {e.code}")
                out = (False, None, permanent)
            return (*out, None) if want_body else out
        except (urllib.error.URLError, OSError) as e:
            logger.debug(f"[discord] webhook {method} error: {e}")
            out = (False, None, False)
            return (*out, None) if want_body else out

    def close(self) -> None:
        """Flush remaining lines (best-effort) and stop the poster thread."""
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        self._thread.join(timeout=_DRAIN_DEADLINE + 2.0)


def _parse_retry_after(error: urllib.error.HTTPError) -> float | None:
    """Extract Discord's retry_after seconds from a 429 response."""
    try:
        body = json.loads(error.read().decode("utf-8"))
        value = float(body.get("retry_after", 0))
        return value if value > 0 else None
    except (ValueError, OSError, AttributeError):
        return None
