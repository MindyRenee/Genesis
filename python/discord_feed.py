"""Discord feed — mirror the terminal session to a Discord channel.

Set ``GENESIS_DISCORD_WEBHOOK`` to a Discord channel webhook URL and
the CLI mirrors the whole session there: ``genesis>`` replies,
``genesis~`` thoughts, slash-command output, system notices, and the
user's typed input (which the terminal itself never writes to the
log, so it is enqueued explicitly).

A webhook can only *post* — it cannot read the channel — so the feed
is read-only by construction. Channel permissions decide who watches.

Design notes:

- ``DiscordFeed`` is a thread-safe bounded queue plus a single daemon
  poster thread. Producers enqueue; the poster coalesces bursts into
  single messages (Discord caps content at 2000 chars), paces
  requests, and honors 429 ``retry_after``. Posting never blocks or
  raises into Genesis — a dead webhook costs a queue, nothing more.
- ``FeedLogHandler`` attaches to the root logger so every line the
  terminal shows is also enqueued. ``StderrTee`` wraps ``sys.stderr``
  for the few direct prints that bypass logging.
- ``allowed_mentions`` is empty — mirrored text can never ping anyone.
- Line prefixes map to usernames: ``genesis>``/``genesis~`` post as
  "Genesis", ``you>`` posts as "you", everything else uses the
  webhook's configured name.
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
# Seconds the poster waits after the first queued line for more of the
# same author to arrive, so a burst coalesces into one message.
_COALESCE_WINDOW = 0.15

# Line-prefix → Discord username overrides. The webhook's own name is
# used for anything without a prefix (system notices, status dumps).
_PREFIX_AUTHORS: tuple[tuple[str, str], ...] = (
    ("genesis>", "Genesis"),
    ("genesis~", "Genesis"),
    ("you>", "you"),
)


def _classify(line: str) -> tuple[str | None, str]:
    """Split a terminal line into (Discord username, content)."""
    stripped = line.strip("\n")
    lead = stripped.lstrip()
    for marker, author in _PREFIX_AUTHORS:
        if lead.startswith(marker):
            return author, lead[len(marker):].lstrip()
    return None, stripped


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
        self._queue: deque[tuple[str | None, str]] = deque()
        self._cond = threading.Condition()
        self._closing = False
        self._disabled = False
        self._thread = threading.Thread(
            target=self._run, name="discord-feed", daemon=True
        )
        self._thread.start()

    @classmethod
    def from_env(cls) -> DiscordFeed | None:
        """Create a feed from ``GENESIS_DISCORD_WEBHOOK``, or None."""
        url = os.environ.get(_WEBHOOK_ENV, "").strip()
        if not url:
            return None
        feed = cls(url)
        logger.info(f"[discord] Mirroring this session to {_WEBHOOK_ENV}.")
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
            self._queue.append((author, text))
            self._cond.notify()

    def post_line(self, line: str) -> None:
        """Enqueue one terminal line, classifying genesis>/you> prefixes."""
        author, content = _classify(line)
        if content.strip():
            self.post(content, author)

    def post_user(self, text: str) -> None:
        """Enqueue user input verbatim (typed input never reaches the log)."""
        self.post(text, "you")

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
            author, content = batch
            if backoff:
                wait = backoff
                if drain_deadline is not None:
                    wait = min(wait, max(0.0, drain_deadline - time.monotonic()))
                time.sleep(wait)
            ok, retry_after, permanent = self._post(content, author)
            if ok:
                backoff = 0.0
            elif permanent:
                with self._cond:
                    self._disabled = True
                logger.info(
                    "[discord] Webhook rejected a post — feed disabled "
                    "for this session."
                )
                return
            else:
                backoff = (
                    retry_after
                    if retry_after is not None
                    else min(_BACKOFF_CAP, max(2.0, backoff * 2))
                )
                # Re-queue the failed batch so it isn't lost.
                with self._cond:
                    self._queue.appendleft((author, content))
            time.sleep(0.0 if self._closing else self._post_interval)

    def _take_batch(self) -> tuple[str | None, str] | None:
        """Pop one item, coalescing same-author items up to the limit."""
        with self._cond:
            while not self._queue:
                if self._closing:
                    return None
                self._cond.wait(timeout=1.0)
            author, text = self._queue.popleft()
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
                next_author, next_text = self._queue[0]
                if next_author != author or total + 1 + len(next_text) > _MAX_CONTENT:
                    break
                self._queue.popleft()
                parts.append(next_text)
                total += 1 + len(next_text)
            return author, "\n".join(parts)

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


class FeedLogHandler(logging.Handler):
    """Logging handler that mirrors every emitted record to a feed.

    Attach to the root logger with the same ``%(message)s`` format the
    console uses, so the feed sees exactly what the terminal shows.
    """

    def __init__(self, feed: DiscordFeed) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter("%(message)s"))
        self._feed = feed

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._feed.post_line(self.format(record))
        except Exception:  # noqa: BLE001 — mirroring must never break logging
            pass


class StderrTee:
    """File-like wrapper that mirrors stderr writes into the feed.

    Forwards every write to the real stderr, then feeds complete lines
    to ``post_line``. Assign over ``sys.stderr``; keep the return value
    to restore it later.
    """

    def __init__(self, real, feed: DiscordFeed) -> None:
        self._real = real
        self._feed = feed
        self._buf = ""

    def write(self, s: str) -> int:
        self._real.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._feed.post_line(line)
        return len(s)

    def flush(self) -> None:
        self._real.flush()
        if self._buf.strip():
            self._feed.post_line(self._buf)
            self._buf = ""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._real, name)
