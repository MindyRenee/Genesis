"""Discord feed — the webhook transport behind Genesis's channel.

The channel itself — *her page*, a place in the outer world where
the public gathers — is ``discord.channel.Channel``, held by
``OuterEnvironment``. This module is only the transport: posting out
(``DiscordFeed``) and the session-mirror plumbing (``FeedLogHandler``,
``StderrTee``). Polling in — the ear — is the body's, and lives in
``body.hearing.discord_ear`` (the organ is hers even though it
listens outward).

Three channels exist, each on its own webhook:

- ``GENESIS_DISCORD_WEBHOOK`` — *her* channel, the self∩outer
  overlap. It carries the mirrored terminal session — ``genesis>``
  replies, ``you>`` input, system notices — plus the posts she
  chooses to broadcast, composed from her own cognition when the
  ``broadcast`` urge fires (see ``basal_ganglia.volition``). Her
  ``genesis~ [broadcast]`` echo is excluded from the mirror so a
  broadcast lands on the channel once, untagged.
- ``GENESIS_DISCORD_INNER_WEBHOOK`` — optional: her inner world's
  own page. When set, ``genesis~`` thoughts and dreams post there
  instead of the shared channel — inner life is inner world, not
  part of the conversation. Unset, they stay on the main channel.
- ``GENESIS_DISCORD_PULSE_WEBHOOK`` — an opt-in ops telemetry feed
  driven by ``scripts/discord_pulse.py``.
- ``GENESIS_DISCORD_INFO_WEBHOOK`` — optional: a read-only info page.
  When set, the "about this channel" card (``discord.guide``) is kept
  there as one maintained message, so the explanation never scrolls
  away. Unset, the guide posts once to the shared channel at page
  open.

A webhook can only *post* — it cannot read the channel. Hearing the
public needs a bot identity: ``body.hearing.discord_ear`` polls the
channel with ``GENESIS_DISCORD_BOT_TOKEN`` when one is configured —
that is the one inbound path, and everything it hears is attributed
to the public. Without the token her channel stays read-only by
construction, exactly as before.

Design notes:

- ``DiscordFeed`` is a thread-safe bounded queue plus a single daemon
  poster thread. Producers enqueue; the poster coalesces bursts into
  single messages (Discord caps content at 2000 chars), paces
  requests, and honors 429 ``retry_after``. Posting never blocks or
  raises into Genesis — a dead webhook costs a queue, nothing more.
- ``FeedLogHandler`` attaches to the root logger so every line the
  terminal shows is also enqueued. ``StderrTee`` wraps ``sys.stderr``
  for the few direct prints that bypass logging.
- ``allowed_mentions`` is empty — posted text can never ping anyone.
- Line prefixes map to message *kinds* (``_KIND_STYLE``): ``genesis>``
  posts as "Genesis" on a blurple card, ``you>`` as the profile name
  in plain text, and everything else — notices, panels, command
  output — as "session" on muted grey. ``genesis~`` splits by its
  ``[tag]``: outward tags (question, expression, distress — the
  kinds the mind itself counts as her voice reaching out) ride the
  speech card; ``thought``/``dream`` post as "Genesis 💭" on purple
  (and onto her inner page when ``GENESIS_DISCORD_INNER_WEBHOOK``
  is set); every other tag — play, code, learning, art — is her
  acting in the world, teal on the shared page. A uniform tag lifts
  out of the words into the card's header.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Iterator

from genesis_client.swallow import note_swallowed
from genesis_client.threads import named_thread
from infrastructure.net import bounded_urlopen as _bounded_urlopen
from infrastructure.net import parse_retry_after as _parse_retry_after
from infrastructure.pulse import Pulse

logger = logging.getLogger(__name__)

_WEBHOOK_ENV = "GENESIS_DISCORD_WEBHOOK"
# Optional second page — her inner world. When set, genesis~ inner
# lines post there instead of the shared channel: conversation is
# the self∩outer overlap (her voice, the public's words, the
# session's noise all in one place), thoughts and dreams are inner
# world and live apart from it. Unset, inner life stays on the main
# page under its purple card — the mirror stays whole.
_INNER_WEBHOOK_ENV = "GENESIS_DISCORD_INNER_WEBHOOK"
# Optional third page — a read-only info channel holding the guide
# card (who she is, how to talk to her, how to read the status
# card). One maintained message rather than a post that scrolls off.
_INFO_WEBHOOK_ENV = "GENESIS_DISCORD_INFO_WEBHOOK"
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
# Consecutive transient failures after which the outage is announced
# once. Transient failures never disable the feed — retries continue
# at the backoff cap until the channel answers.
_OUTAGE_NOTICE = 20
# Seconds the poster waits after the first queued line for more of the
# same kind to arrive, so a burst coalesces into one message.
_COALESCE_WINDOW = 0.15

# Message kinds — every mirrored or posted line carries one, and the
# kind decides how it renders on the channel. A reader scans the
# color bar and the speaker name to tell her voice from her inner
# life from session noise; without kinds the mirror is one flat
# stream of identical messages.
KIND_SPEECH = "speech"  # genesis> replies, broadcasts, outward utterances
KIND_ACTIVITY = "activity"  # genesis~ play/code/learning/art — her acting in the world
KIND_INNER = "inner"  # genesis~ thoughts, dreams — her inner life
KIND_USER = "user"  # you> — the human's own input
KIND_SYSTEM = "session"  # everything else the terminal shows
KIND_BOARD = "board"  # the self-editing vitals card
KIND_PULSE = "pulse"  # ops telemetry (separate webhook)
KIND_GUIDE = "guide"  # the "what is this channel" explainer card

# kind → (embed color, default username). A kind's color is the
# message's left bar; the username is who the words are attributed
# to. ``user`` posts stay plain content — a human's words read as
# chat, not a card. "session" rather than "system" for the noise:
# Discord reserves "system" in webhook usernames (a rejected name
# 400s the whole post).
_KIND_STYLE: dict[str, tuple[int | None, str | None]] = {
    KIND_SPEECH: (0x5865F2, "Genesis"),  # blurple — her public voice
    KIND_ACTIVITY: (0x1ABC9C, "Genesis"),  # teal — her acting, not musing
    KIND_INNER: (0x9B59B6, "Genesis 💭"),  # purple — the inner life
    KIND_USER: (None, "you"),  # plain text — a person talking
    KIND_SYSTEM: (0x4F545C, "session"),  # muted grey — terminal noise
    KIND_BOARD: (0x57F287, None),  # green — the instrument card
    KIND_PULSE: (0xFAA61A, None),  # amber — ops telemetry
    KIND_GUIDE: (0x3498DB, "about genesis"),  # light blue — orientation
}

# genesis~ tags that are her *voice reaching out* — the same set the
# mind itself records into the outer world via ``it_said``
# (``_emit_live_thought``: "expressions, questions, and distress
# calls are its voice reaching out"). They post on the shared page —
# speech is the overlap, not the inner world — while ``thought`` and
# ``dream`` (incl. ``thought:N``/``dream:N`` chain positions) are the
# inner kinds routed to her inner page. Every other genesis~ tag —
# play, code, learning, art, listening, sleeping — is activity:
# her *doing* in the world, shown on the shared page on teal.
_OUTWARD_TAGS = frozenset(("question", "expression", "distress"))


# Terminal control bytes that reach the mirror would render on
# Discord as literal noise: SGR colour/dim escapes (``\033[2m``),
# erase/cursor CSIs (``\r\033[K``), OSC sequences, charset shifts,
# and stray C0 controls. Ordered so full escape sequences are
# consumed before the fallback single-byte class strips leftovers.
_ANSI_RE = re.compile(
    r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC … terminated by BEL or ST
    r"|\x1b\[[0-9;?]*[ -/]*[@-~]"  # CSI … SGR, erase, cursor motion
    r"|\x1b[()#%][0-9A-Za-z]?"  # charset designation / shift escapes
    r"|[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]"  # other C0 + DEL (\t \n \r survive —
    # \r is the frame separator the rewrite-collapse needs)
)

# Glyphs that only align under a monospace font — a mirrored status
# panel full of them needs a code fence to survive Discord's
# proportional rendering.
_BOX_CHARS = frozenset("│─┌┐└┘┃═║╔╗╚╝├┤┬┴┼╭╮╰╯")


def _clean(text: str) -> str:
    """Render terminal text Discord-readable.

    Strips ANSI/CSI escape sequences, then collapses ``\\r``-rewritten
    text to the final frame a terminal would have shown (a progress
    rewrite posts as one line, not a smear of every intermediate
    state).
    """
    if "\x1b" in text or "\r" in text:
        text = _ANSI_RE.sub("", text)
    if "\r" in text:
        frames = [f for f in text.split("\r") if f.strip()]
        text = frames[-1] if frames else ""
    return text


def _fence_if_panel(chunk: str) -> str:
    """Fence a chunk that carries box-drawing — columns only survive
    in monospace. Runs per-chunk so a long panel split by ``_chunks``
    still fences both halves, and skipped when the content already
    fences itself or the fence would breach Discord's limit."""
    if "```" in chunk or not (_BOX_CHARS & set(chunk)):
        return chunk
    fenced = f"```\n{chunk.strip()}\n```"
    return fenced if len(fenced) <= 2000 else chunk


# genesis~ lines arrive tagged — ``genesis~ [thought] a musing``.
# The tag both decides the kind here and, at post time, lifts out of
# the words into the embed's small header when it is uniform.
_TAG_RE = re.compile(r"^\[([^\[\]\n]{1,32})\][ \t]*")


def _classify(line: str) -> tuple[str | None, str | None, str]:
    """Split a terminal line into (kind, Discord username, content).

    ``(None, None, "")`` marks a line that must not mirror at all.
    The terminal's leading indent carries no meaning on the channel —
    stripping it here is a uniform left-shift, so panel columns keep
    their internal alignment under the fence."""
    stripped = _clean(line).strip("\n").lstrip()
    lead = stripped
    # Two genesis~ kinds must never mirror onto the channel:
    # - [broadcast]: the terminal echo of a public share — the
    #   explicit post already carries the utterance untagged, so
    #   mirroring it would land the same utterance twice.
    # - [heard]: the terminal echo of a *public* message she heard
    #   through DiscordListener — mirroring it would repost the
    #   public's own words under her name.
    if lead.startswith(("genesis~ [broadcast]", "genesis~ [heard]")):
        return None, None, ""
    if lead.startswith("genesis>"):
        return KIND_SPEECH, _KIND_STYLE[KIND_SPEECH][1], lead[len("genesis>") :].lstrip()
    if lead.startswith("you>"):
        return KIND_USER, _KIND_STYLE[KIND_USER][1], lead[len("you>") :].lstrip()
    if lead.startswith("genesis~"):
        rest = lead[len("genesis~") :].lstrip()
        tag = _TAG_RE.match(rest)
        if tag is None:
            # Untagged genesis~ chrome — "(thinking...)" and friends —
            # is session noise, not a thought.
            return KIND_SYSTEM, _KIND_STYLE[KIND_SYSTEM][1], rest
        name = tag.group(1)
        if name in _OUTWARD_TAGS:
            kind = KIND_SPEECH
        elif name in ("thought", "dream") or name.startswith(("thought:", "dream:")):
            kind = KIND_INNER
        else:
            kind = KIND_ACTIVITY
        return kind, _KIND_STYLE[kind][1], rest
    return KIND_SYSTEM, _KIND_STYLE[KIND_SYSTEM][1], stripped


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
        live_kind: str = KIND_BOARD,
    ) -> None:
        self._url = webhook_url
        self._post_interval = post_interval
        self._queue_max = queue_max
        # Which card identity the maintained message wears — the
        # instrument board by default; the info feed's single live
        # message is the guide card instead.
        self._live_kind = live_kind
        self._queue: deque[tuple] = deque()
        self._cond = Pulse()
        self._closing = False
        self._disabled = False
        self._fail_streak = 0
        # True once a transient outage has been announced — one
        # "unreachable" notice per outage, and one "restored" when a
        # post lands again, instead of a line per failed attempt.
        self._outage = False
        # Who ``user``-kind words are attributed to. The session mirror
        # sets this from the user profile once the mind is up; until
        # then the channel's human reads as plain "you".
        self.user_name = "you"
        # Her inner world page, when a second webhook is configured —
        # post_line routes ``inner``-kind lines there. None keeps the
        # whole mirror on this channel.
        self.inner_feed: DiscordFeed | None = None
        # The info page, when a third webhook is configured — the
        # guide card lives there as one maintained message.
        self.info_feed: DiscordFeed | None = None
        # Set at the drain deadline — in-flight requests poll it so a
        # wedged urlopen expires *with* the flush grace, not after.
        self._abandon = Pulse()
        # Armed by close() itself, under the cond: the flush grace is
        # measured from the close call, not from whenever the poster
        # first notices — a request parked through the abandon must
        # land on an already-expired deadline, not a fresh window.
        self._drain_deadline: float | None = None
        # A webhook may edit its own messages; the live-board view is
        # one message kept current rather than a scroll of frames.
        self._live_id: str | None = None
        self._thread = named_thread("discord-feed", self._run, comm="g-disc-feed")
        self._thread.start()

    @property
    def live_message_id(self) -> str | None:
        """The maintained message's id, once the poster has created it."""
        return self._live_id

    @live_message_id.setter
    def live_message_id(self, message_id: str | None) -> None:
        """Seed the maintained-message id — e.g. one saved from a
        previous boot so a card is edited, not reposted."""
        self._live_id = message_id

    @classmethod
    def from_env(cls) -> DiscordFeed | None:
        """Create her public channel feed from ``GENESIS_DISCORD_WEBHOOK``.

        Returns None when the webhook is unset — there is no channel
        then, so the session stays local and her broadcasts have
        nowhere to post.
        """
        url = os.environ.get(_WEBHOOK_ENV, "").strip()
        if not url:
            return None
        feed = cls(url)
        logger.info(
            f"[discord] Channel enabled ({_WEBHOOK_ENV}) — carries the "
            "session feed and her broadcast posts."
        )
        inner_url = os.environ.get(_INNER_WEBHOOK_ENV, "").strip()
        if inner_url:
            feed.inner_feed = cls(inner_url)
            logger.info(
                f"[discord] Inner page enabled ({_INNER_WEBHOOK_ENV}) — "
                "her thoughts and dreams post apart from the shared channel."
            )
        info_url = os.environ.get(_INFO_WEBHOOK_ENV, "").strip()
        if info_url:
            feed.info_feed = cls(info_url, live_kind=KIND_GUIDE)
            logger.info(
                f"[discord] Info page enabled ({_INFO_WEBHOOK_ENV}) — "
                "the channel guide lives there as one maintained card."
            )
        return feed

    # ── Producer API ─────────────────────────────────────────────────

    def post(self, text: str, author: str | None = None, kind: str = KIND_SPEECH) -> None:
        """Enqueue one message. ``author`` overrides the kind's speaker
        name; ``kind`` picks the card's color."""
        text = _clean(text)
        if not text.strip():
            return
        if self._disabled:
            return
        while len(self._queue) >= self._queue_max:
            self._queue.popleft()
        self._queue.append(("text", kind, author, text))
        self._cond.notify()

    def post_line(self, line: str) -> None:
        """Enqueue one terminal line, classifying genesis>/genesis~/you>."""
        kind, author, content = _classify(line)
        if kind and content.strip():
            if kind == KIND_INNER and self.inner_feed is not None:
                # Inner world lives on its own page when one exists —
                # her thoughts and dreams don't mix with the overlap.
                self.inner_feed.post(content, author, kind=kind)
                return
            if kind == KIND_USER:
                author = self.user_name
            self.post(content, author, kind=kind)

    def post_user(self, text: str) -> None:
        """Enqueue user input verbatim (typed input never reaches the log)."""
        self.post(text, self.user_name, kind=KIND_USER)

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
        if self._disabled:
            return
        self._queue = deque(item for item in self._queue if item[0] not in ("live", "live_text"))
        self._queue.append(("live", author, caption, png))
        self._cond.notify()

    def update_live_text(self, text: str, author: str | None = None) -> None:
        """Queue a text-only update to the live-board message.

        The board doesn't always need a rendered frame — her channel
        card is state readout, plain content. Pending live updates of
        either kind collapse (both target the same message), so only
        the newest board is ever sent.
        """
        text = _clean(text)
        if not text.strip():
            return
        if self._disabled:
            return
        self._queue = deque(item for item in self._queue if item[0] not in ("live", "live_text"))
        self._queue.append(("live_text", author, text))
        self._cond.notify()

    # ── Poster thread ────────────────────────────────────────────────

    def _run(self) -> None:
        backoff = 0.0
        while True:
            drain_deadline = self._drain_deadline
            if drain_deadline is not None and time.monotonic() > drain_deadline:
                return
            batch = self._take_batch()
            if batch is None:
                return
            if backoff:
                wait = backoff
                if drain_deadline is not None:
                    wait = min(wait, max(0.0, drain_deadline - time.monotonic()))
                # Interruptible — close()'s notify wakes the drain
                # now instead of sleeping out the backoff.
                self._cond.wait_for(lambda: self._closing or self._disabled, timeout=wait)
            # An in-flight request must expire *with* the drain
            # deadline, not the urlopen bound past it — otherwise a
            # dead network turns close() into its full 15 s timeout.
            bound = (
                max(0.0, drain_deadline - time.monotonic()) if drain_deadline is not None else None
            )
            if bound is not None and bound <= 0.0:
                return
            if batch[0] == "text":
                _, msg_kind, author, content = batch
                ok, retry_after, permanent = self._post(content, author, msg_kind, bound)
            elif batch[0] == "image":
                _, author, caption, filename, png = batch
                ok, retry_after, permanent = self._post_image_once(
                    caption, author, filename, png, bound
                )
            elif batch[0] == "live_text":
                _, author, text = batch
                ok, retry_after, permanent = self._post_live_text(text, author, bound)
            else:  # "live"
                _, author, caption, png = batch
                ok, retry_after, permanent = self._post_live(caption, author, png, bound)
            if ok:
                if self._outage:
                    logger.info("[discord] Webhook reachable again — feed restored.")
                    self._outage = False
                backoff = 0.0
                self._fail_streak = 0
            elif permanent:
                self._disabled = True
                logger.info("[discord] Webhook rejected a post — feed disabled for this session.")
                return
            else:
                self._fail_streak += 1
                # A transient failure is the network, not the webhook —
                # a wifi roam or DNS stall outlasts any fixed retry
                # count, so the feed never disables on it. Retry at
                # the backoff cap until the channel answers; the
                # bounded queue sheds oldest lines meanwhile. Only a
                # rejected post (4xx above) kills the session feed —
                # that cannot heal.
                if self._fail_streak >= _OUTAGE_NOTICE and not self._outage:
                    self._outage = True
                    logger.info(
                        "[discord] Webhook unreachable — posts queue "
                        "and keep retrying until the channel answers."
                    )
                backoff = (
                    min(_BACKOFF_CAP, retry_after)
                    if retry_after is not None
                    else min(_BACKOFF_CAP, max(2.0, backoff * 2))
                )
                # Re-queue the failed batch so it isn't lost.
                self._queue.appendleft(batch)
            self._cond.wait_for(lambda: self._closing, timeout=self._post_interval)

    def _take_batch(self) -> tuple | None:
        """Pop one item, coalescing same-kind text up to the limit."""
        while not self._queue:
            if self._closing:
                return None
            self._cond.wait(timeout=1.0)
        item = self._queue.popleft()
        if item[0] != "text":
            return item
        _, kind, author, text = item
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
            next_kind, next_author, next_text = nxt[1], nxt[2], nxt[3]
            if (next_kind, next_author) != (kind, author) or total + 1 + len(
                next_text
            ) > _MAX_CONTENT:
                break
            self._queue.popleft()
            parts.append(next_text)
            total += 1 + len(next_text)
        return ("text", kind, author, "\n".join(parts))

    def _post(
        self,
        content: str,
        author: str | None,
        kind: str = KIND_SPEECH,
        bound: float | None = None,
    ) -> tuple[bool, float | None, bool]:
        """POST content (chunked). Returns (ok, retry_after, permanent)."""
        for chunk in _chunks(content):
            chunk = _fence_if_panel(chunk)
            ok, retry_after, permanent = self._post_once(chunk, author, kind, bound)
            if not ok:
                return ok, retry_after, permanent
        return True, None, False

    def _post_once(
        self,
        content: str,
        author: str | None,
        kind: str = KIND_SPEECH,
        bound: float | None = None,
    ) -> tuple[bool, float | None, bool]:
        color, default_author = _KIND_STYLE.get(kind, _KIND_STYLE[KIND_SYSTEM])
        name = author or default_author
        if color is None:
            # ``user`` words post as plain content — a person talking,
            # not a card.
            payload: dict[str, object] = {"content": content}
        else:
            embed: dict[str, object] = {"description": content, "color": color}
            # A uniform [tag] on every line lifts out of the text into
            # the card's small header — [thought] on an inner card,
            # [question] on a speech card, [learning] on an activity
            # card, [genesis] on a session card. Mixed or untagged
            # text stays verbatim so no words are ever dropped.
            tagged: list[tuple[str, re.Match[str]]] = []
            for line in content.split("\n"):
                match = _TAG_RE.match(line)
                if match is None:
                    tagged = []
                    break
                tagged.append((line, match))
            if tagged and all(m.group(1) == tagged[0][1].group(1) for _, m in tagged):
                embed["author"] = {"name": tagged[0][1].group(1)}
                embed["description"] = "\n".join(line[m.end() :] for line, m in tagged)
            payload = {"embeds": [embed]}
        payload["allowed_mentions"] = {"parse": []}
        if name:
            payload["username"] = name
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
        timeout = min(10.0, bound) if bound is not None else 10.0
        if timeout <= 0.0:
            return False, None, False
        try:
            with _bounded_urlopen(req, timeout=timeout, stop=self._abandon):
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
        bound: float | None = None,
    ) -> tuple[bool, float | None, bool]:
        # The drawing lands inside her speech card — one colored
        # embed holding both the image and her caption, rather than
        # a bare upload next to loose text.
        color, default_author = _KIND_STYLE[KIND_SPEECH]
        payload: dict[str, object] = {
            "allowed_mentions": {"parse": []},
            "attachments": [{"id": 0, "filename": filename, "description": caption[:1024]}],
            "embeds": [
                {
                    "color": color,
                    "image": {"url": f"attachment://{filename}"},
                    **({"description": caption} if caption else {}),
                }
            ],
        }
        name = author or default_author
        if name:
            payload["username"] = name
        body, content_type = _multipart(payload, filename, png)
        return self._send(self._url, "POST", body, content_type, bound=bound)

    def _post_live(
        self,
        caption: str,
        author: str | None,
        png: bytes,
        bound: float | None = None,
    ) -> tuple[bool, float | None, bool]:
        """PATCH the live message if it exists, else create it (wait=true)."""
        # Same card treatment as the text board — the frame rides
        # inside the instrument-colored embed, not as a bare upload.
        embed: dict[str, object] = {
            "color": _KIND_STYLE[self._live_kind][0],
            "image": {"url": "attachment://board.png"},
        }
        if caption:
            embed["description"] = caption
        payload: dict[str, object] = {
            "allowed_mentions": {"parse": []},
            "attachments": [{"id": 0, "filename": "board.png"}],
            "embeds": [embed],
        }
        if self._live_id:
            url = f"{self._url}/messages/{self._live_id}"
            body, content_type = _multipart(payload, "board.png", png)
            ok, retry_after, permanent = self._send(url, "PATCH", body, content_type, bound=bound)
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
            url, "POST", body, content_type, want_body=True, bound=bound
        )
        if status and resp:
            try:
                self._live_id = json.loads(resp).get("id")
            except ValueError as e:
                note_swallowed("discord.feed.live_status_parse", e)
        return status, retry_after, permanent

    def _post_live_text(
        self,
        text: str,
        author: str | None,
        bound: float | None = None,
    ) -> tuple[bool, float | None, bool]:
        """PATCH the live message's card, or create it (wait=true).

        Plain JSON — the board carries no attachment, so no multipart
        body is needed; the card itself is a single embed (its color
        bar is what marks it as an instrument, not chat). Same
        live_id lifecycle as the image board: a permanent PATCH
        failure drops the id and re-creates.
        """
        payload: dict[str, object] = {
            "embeds": [{"description": text, "color": _KIND_STYLE[self._live_kind][0]}],
            "allowed_mentions": {"parse": []},
        }
        if self._live_id:
            body = json.dumps(payload).encode("utf-8")
            ok, retry_after, permanent = self._send(
                f"{self._url}/messages/{self._live_id}",
                "PATCH",
                body,
                "application/json",
                bound=bound,
            )
            if ok or retry_after is not None or not permanent:
                return ok, retry_after, permanent
            self._live_id = None
        # username is a create-time field — a message PATCH doesn't
        # take it, so it only joins the fresh-message payload.
        if author:
            payload["username"] = author
        body = json.dumps(payload).encode("utf-8")
        url = self._url + ("&" if "?" in self._url else "?") + "wait=true"
        status, retry_after, permanent, resp = self._send(
            url, "POST", body, "application/json", want_body=True, bound=bound
        )
        if status and resp:
            try:
                self._live_id = json.loads(resp).get("id")
            except ValueError as e:
                note_swallowed("discord.feed.live_status_parse", e)
        return status, retry_after, permanent

    def _send(
        self,
        url: str,
        method: str,
        body: bytes,
        content_type: str,
        want_body: bool = False,
        bound: float | None = None,
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
        timeout = min(15.0, bound) if bound is not None else 15.0
        if timeout <= 0.0:
            return (False, None, False, None) if want_body else (False, None, False)
        try:
            with _bounded_urlopen(req, timeout=timeout, stop=self._abandon) as r:
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
        if self.inner_feed is not None:
            # The inner page is hers too — it drains inside the same
            # teardown grace rather than outliving the session.
            self.inner_feed.close()
        if self.info_feed is not None:
            self.info_feed.close()
        if not self._closing:
            self._closing = True
            self._drain_deadline = time.monotonic() + _DRAIN_DEADLINE
        self._cond.notify_all()
        # Requests still on the wire when the drain deadline hits are
        # abandoned — without this a wedged urlopen keeps the poster
        # (and the join below) parked past the grace it was given.
        timer = threading.Timer(_DRAIN_DEADLINE, self._abandon.set)
        timer.daemon = True  # a non-daemon Timer would hold the exit itself
        timer.start()
        self._thread.join(timeout=_DRAIN_DEADLINE + 2.0)


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
        # Never feed this module's own log records back into the queue.
        # A webhook outage logs an error here every failed POST; if that
        # record re-enqueues, the queue never drains and the poster
        # churns forever — the reported hang.
        if record.name == logger.name:
            return
        try:
            self._feed.post_line(self.format(record))
        except Exception:  # noqa: BLE001 — mirroring must never break logging
            # NOTE: must not use logger here (it would recurse into this
            # handler). Delegate to the standard logging error handling,
            # which writes to stderr when raiseExceptions is enabled.
            self.handleError(record)


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
